#!/usr/bin/env python3
"""
"Rainy Window" — a lo-fi / downtempo sketch.

Everything is synthesized from scratch with numpy (no samples, no libraries
beyond numpy).  Output: 44.1 kHz / 16-bit stereo WAV.

Tweakables: BPM, CHORDS, the MEL_* lists (bar, beat, midi, beats, velocity).
"""
import wave
import numpy as np

SR = 44100
BPM = 84.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
NBARS = 16
TAIL = 3.0
DUR = NBARS * BAR + TAIL
N = int(DUR * SR)

rng = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# harmony / arrangement data
# ----------------------------------------------------------------------------
# key: A minor.  Am9 | Fmaj9 | Cmaj9 | G6/9, one bar each, four times.
CHORDS = [
    dict(name="Am9",   root=33, rhodes=[57, 60, 64, 67, 71], pad=[57, 60, 64, 71]),
    dict(name="Fmaj9", root=29, rhodes=[53, 57, 60, 64, 67], pad=[53, 57, 60, 64, 67]),
    dict(name="Cmaj9", root=36, rhodes=[52, 55, 59, 62, 64], pad=[52, 55, 59, 62]),
    dict(name="G6/9",  root=31, rhodes=[55, 59, 62, 64, 69], pad=[55, 59, 62, 69]),
]

# bass: (beat, semitones above root, length in beats, velocity)
BASS_PAT = [
    [(0.00, 0, 1.70, 0.90), (1.75, 0, 0.50, 0.55), (2.50, 7, 0.50, 0.70), (3.50, 0, 0.70, 0.60)],
    [(0.00, 0, 1.20, 0.90), (1.50, 12, 0.50, 0.50), (2.50, 0, 0.90, 0.70), (3.50, 7, 0.40, 0.50)],
    [(0.00, 0, 1.70, 0.90), (1.75, 7, 0.40, 0.50), (2.50, 0, 0.50, 0.65), (3.50, 12, 0.60, 0.60)],
    [(0.00, 0, 1.20, 0.85), (1.50, 0, 0.50, 0.50), (2.75, 7, 0.50, 0.60), (3.50, 9, 0.50, 0.55)],
]

# melody: (bar offset, beat, midi note, length in beats, velocity)
MEL_A = [
    (0, 2.5, 76, 1.0, 0.70), (0, 3.5, 79, 0.5, 0.60), (0, 4.0, 81, 0.9, 0.65),
    (1, 1.0, 81, 1.4, 0.70), (1, 2.5, 79, 0.7, 0.60), (1, 3.5, 76, 0.9, 0.65),
    (2, 1.0, 74, 0.7, 0.65), (2, 2.0, 76, 0.45, 0.60), (2, 2.5, 74, 0.7, 0.60), (2, 3.5, 72, 1.2, 0.65),
    (3, 1.0, 71, 0.7, 0.60), (3, 2.0, 74, 0.7, 0.60), (3, 3.0, 76, 0.45, 0.55), (3, 3.5, 74, 1.6, 0.60),
]

MEL_B = [
    (0, 1.0, 81, 0.7, 0.75), (0, 1.5, 79, 0.5, 0.60), (0, 2.0, 76, 1.0, 0.70),
    (0, 3.5, 79, 0.5, 0.60), (0, 4.0, 81, 1.0, 0.70),
    (1, 1.0, 84, 0.7, 0.75), (1, 2.0, 81, 1.0, 0.70), (1, 3.0, 79, 0.5, 0.60), (1, 3.5, 77, 0.9, 0.65),
    (2, 1.0, 76, 0.7, 0.70), (2, 2.0, 79, 0.7, 0.65), (2, 3.0, 76, 0.5, 0.60), (2, 3.5, 74, 1.5, 0.60),
    (3, 1.0, 71, 1.0, 0.65), (3, 2.0, 74, 0.7, 0.60), (3, 3.0, 76, 0.5, 0.55), (3, 3.5, 79, 1.2, 0.60),
]

# reprise of A, but bar 4 settles on the tonic
MEL_C = MEL_A[:-1] + [(3, 3.5, 69, 3.0, 0.70)]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def midi2f(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def gate(n, attack, release, curve=2.0):
    """Click-free attack / release envelope, returns length-n array."""
    e = np.ones(n)
    a = min(max(int(attack * SR), 1), n)
    e[:a] = np.linspace(0, 1, a, endpoint=False) ** curve
    r = min(max(int(release * SR), 1), n)
    e[n - r:] = np.minimum(e[n - r:], np.linspace(1, 0, r, endpoint=False) ** curve)
    return e


def fft_filter(x, lo=None, hi=None, order=2):
    """Zero-phase (FFT) Butterworth-magnitude filter."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    H = np.ones_like(f)
    if hi:
        H *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        H *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-6)) ** (2 * order))
    return np.fft.irfft(X * H, len(x))


def add(bus, sig, t0, pan=0.0, gain=1.0):
    """Equal-power pan and mix a mono signal into a (2, N) bus."""
    i = int(round(t0 * SR))
    if i >= bus.shape[1] or i + 1 <= 0:
        return
    j = min(i + len(sig), bus.shape[1])
    seg = sig[: j - i]
    th = (pan + 1) * np.pi / 4
    bus[0, i:j] += seg * gain * np.cos(th)
    bus[1, i:j] += seg * gain * np.sin(th)


# ----------------------------------------------------------------------------
# instruments
# ----------------------------------------------------------------------------
_cache = {}


def rhodes(freq, dur, vel=0.7, bright=1.0):
    """Warm, slightly tine-y electric piano."""
    key = ("rhodes", round(freq, 3), round(dur, 3), round(bright, 2))
    if key in _cache:
        return _cache[key] * vel
    tail = 0.55
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio, amp, dec in ((1.0, 1.0, 1.9), (2.0, 0.30 * bright, 1.0),
                            (3.0, 0.10 * bright, 0.55), (4.02, 0.05 * bright, 0.30),
                            (6.05, 0.02 * bright, 0.18)):
        f = freq * ratio
        if f < SR * 0.45:
            y += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * np.exp(-t / dec)
    # tine / hammer click
    click = rng.standard_normal(n) * np.exp(-t / 0.006) * 0.10 * bright
    y += fft_filter(click, lo=1500, hi=9000)
    y *= gate(n, 0.005, tail)
    y *= 1 - 0.05 * np.sin(2 * np.pi * 5.2 * t)          # gentle tremolo
    y /= 2.2
    _cache[key] = y
    return y * vel


def lead(freq, dur, vel=0.7):
    """Soft, breathy sine-ish lead with delayed vibrato."""
    tail = 0.5
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.0035 * np.sin(2 * np.pi * 5.3 * t) * np.clip((t - 0.18) / 0.5, 0, 1)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    y = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    y *= gate(n, 0.05, tail, curve=1.5) * np.exp(-t / 3.0)
    y *= 1 - 0.03 * np.sin(2 * np.pi * 0.9 * t)
    return y * vel * 0.30


def bass(freq, dur, vel=0.8):
    tail = 0.25
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * freq * t) + 0.22 * np.sin(4 * np.pi * freq * t)
         + 0.08 * np.sin(6 * np.pi * freq * t))
    y *= np.exp(-t / 1.4)
    y = np.tanh(y * 1.7) / 1.7
    y *= gate(n, 0.008, tail)
    return y * vel * 0.55


def pad(freq, dur, vel=0.6):
    """Detuned, filtered saw (band-limited additive), slow attack."""
    key = ("pad", round(freq, 3), round(dur, 2))
    if key in _cache:
        return _cache[key] * vel
    attack, release = 1.1, 1.8
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for dt in (-7.0, 7.0):
        f = freq * 2 ** (dt / 1200)
        K = max(1, min(8, int(0.40 * SR / f)))
        for k in range(1, K + 1):
            out += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi)) / k
    out = fft_filter(out, hi=1500, order=2)
    out *= gate(n, attack, release, curve=1.6)
    out *= 1 - 0.08 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28))
    out /= 9.0
    _cache[key] = out
    return out * vel


def kick(vel=1.0):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 48 + 80 * np.exp(-t / 0.028)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    y += rng.standard_normal(n) * np.exp(-t / 0.004) * 0.25
    y *= gate(n, 0.002, 0.30)
    return y * vel * 0.9


def rim(vel=1.0):
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    noise = fft_filter(rng.standard_normal(n), lo=1400, hi=5200, order=2)
    y = noise * np.exp(-t / 0.045) * 1.6
    y += 0.25 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    y *= gate(n, 0.002, 0.22)
    return y * vel * 0.30


def hat(open_=False, vel=1.0):
    n = int((0.32 if open_ else 0.12) * SR)
    t = np.arange(n) / SR
    y = fft_filter(rng.standard_normal(n), lo=6800, hi=15000, order=2)
    y *= np.exp(-t / (0.16 if open_ else 0.028))
    y *= gate(n, 0.001, 0.05)
    return y * vel * (0.13 if open_ else 0.155)


# ----------------------------------------------------------------------------
# arrangement
# ----------------------------------------------------------------------------
mix = np.zeros((2, N))
send = np.zeros((2, N))          # reverb send
SWING = 0.055                    # delay applied to off-8ths (in beats)


def place(sig, t0, pan, gain, rev=0.0, jitter=0.006):
    t = max(0.0, t0 + rng.normal(0, jitter))
    add(mix, sig, t, pan, gain)
    if rev:
        add(send, sig, t, pan * 0.6, gain * rev)


print(f"{NBARS} bars @ {BPM:.0f} BPM  ->  {DUR:.1f}s")
for bar in range(1, NBARS + 1):
    t_bar = (bar - 1) * BAR
    ch = CHORDS[(bar - 1) % 4]
    intro = bar <= 4
    build = 5 <= bar <= 8
    main = 9 <= bar <= 12
    outro = bar >= 13

    r_vel = 0.55 if intro else (0.68 if build else (0.78 if main else 0.62))
    brightness = 0.8 if intro else (1.0 if build else 1.05)

    # --- electric piano: rolled chord on beat 1, little arpeggio on beat 3
    for k, note in enumerate(ch["rhodes"]):
        place(rhodes(midi2f(note), 2.2 * BEAT, r_vel * (1 - 0.05 * k), brightness),
              t_bar + 0.028 * k, 0.12 + 0.03 * k, 1.0, rev=0.30)
    for k, idx in enumerate((2, 4, 3, 4)):
        place(rhodes(midi2f(ch["rhodes"][idx]), 0.55 * BEAT, r_vel * 0.55, brightness * 0.9),
              t_bar + (2.0 + 0.5 * k) * BEAT + (SWING * BEAT if k % 2 else 0),
              0.30 + 0.05 * k, 0.75, rev=0.35)

    # --- pad
    p_gain = 0.85 if intro else (0.75 if build else (0.95 if main else 0.7))
    for k, note in enumerate(ch["pad"]):
        place(pad(midi2f(note), BAR, 1.0), t_bar, -0.55 + 0.35 * k, p_gain * 0.55,
              rev=0.45)
    if main:  # airy octave on top during the main section
        place(pad(midi2f(ch["pad"][0] + 12), BAR, 0.8), t_bar, 0.4, 0.20, rev=0.55)

    # --- bass
    if not intro:
        for beat, semi, blen, bvel in BASS_PAT[(bar - 1) % 4]:
            place(bass(midi2f(ch["root"] + semi), blen * BEAT, bvel),
                  t_bar + beat * BEAT + (SWING * BEAT if beat % 1 else 0),
                  0.0, 1.0, rev=0.05)

    # --- drums
    if not intro and bar != 16:
        e = 0.45 if build else (1.0 if main else 0.55)
        place(kick(0.78 * e), t_bar + 0 * BEAT, 0.0, 1.0, rev=0.08)
        if bar % 4 != 1 or bar <= 8:
            place(kick(0.64 * e), t_bar + 2.5 * BEAT, 0.0, 1.0, rev=0.06)
        if e > 0.8:
            place(kick(0.50 * e), t_bar + 3.5 * BEAT, 0.0, 1.0, rev=0.05)
        for b in (1, 3):
            place(rim(0.75 * e), t_bar + b * BEAT, -0.12, 1.0, rev=0.22)
        if e > 0.8 and bar % 2 == 0:
            place(rim(0.28), t_bar + 3.75 * BEAT, -0.18, 1.0, rev=0.2)
        for k in range(8):
            hb = k * 0.5 + (SWING if k % 2 else 0.0)
            hvel = (0.9 if k % 2 == 0 else 0.62) * e
            place(hat(False, hvel), t_bar + hb * BEAT, 0.28, 1.0, rev=0.12)
        if e > 0.8:
            place(hat(True, 0.5 * e), t_bar + 3.5 * BEAT, 0.3, 1.0, rev=0.25)

    # --- melody
    if build or main or outro:
        notes, first_bar = (MEL_A, 5) if build else ((MEL_B, 9) if main else (MEL_C, 13))
        m_gain = 1.0 if main else (0.85 if build else 0.8)
        shift = 0   # keep the phrase in one register   # lift the last two bars
        for obar, beat, note, blen, mvel in notes:
            if obar != bar - first_bar:
                continue
            t0 = t_bar + beat * BEAT + (SWING * BEAT if beat % 1 else 0)
            sig = lead(midi2f(note + shift), blen * BEAT, mvel * m_gain)
            place(sig, t0, -0.10, 1.0, rev=0.40)
            # dotted-8th echo, opposite side
            add(mix, sig, t0 + 0.75 * BEAT, 0.35, 0.22)
            add(mix, sig, t0 + 1.50 * BEAT, -0.35, 0.09)

# ----------------------------------------------------------------------------
# space: stereo reverb from a synthetic impulse response
# ----------------------------------------------------------------------------
def make_ir(sec=2.6, decay=0.95):
    n = int(sec * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / decay)
    ir = fft_filter(ir, lo=180, hi=6500, order=1)
    return ir / np.sqrt(np.sum(ir ** 2))


def convolve(x, h):
    n = len(x) + len(h) - 1
    Nfft = 1 << int(np.ceil(np.log2(n)))
    y = np.fft.irfft(np.fft.rfft(x, Nfft) * np.fft.rfft(h, Nfft), Nfft)
    return y[:len(x)]


print("reverb...")
rev = np.stack([convolve(send[0], make_ir()), convolve(send[1], make_ir())])
mix += 0.55 * rev

# ----------------------------------------------------------------------------
# tape / vinyl character
# ----------------------------------------------------------------------------
print("tape + vinyl...")
t_all = np.arange(N) / SR
rate = (1 + 0.0011 * np.sin(2 * np.pi * 0.37 * t_all)
        + 0.0003 * np.sin(2 * np.pi * 3.1 * t_all + 1.2))
pos = np.cumsum(rate)
pos -= pos[0]
idx = np.arange(N)
for c in range(2):
    mix[c] = np.interp(pos, idx, mix[c])

# crackle + hiss: sparse, decaying ticks rather than ideal impulses
clicks = np.zeros(N)
n_click = int(N / SR * 2.2)
L = int(0.005 * SR)
burst_t = np.arange(L) / SR
for start in rng.uniform(0, N - L, n_click).astype(int):
    burst = rng.standard_normal(L) * np.exp(-burst_t / 0.0009)
    clicks[start:start + L] += burst * rng.uniform(0.15, 1.0)
clicks = fft_filter(clicks, lo=500, hi=5200, order=2)
clicks *= 0.055 / max(np.max(np.abs(clicks)), 1e-9)
hiss = fft_filter(rng.standard_normal(N), lo=3000, hi=9000, order=2)
noise = clicks + 0.0018 * hiss
noise *= np.clip(t_all / 1.5, 0, 1)                       # fade noise in
mix += noise

# gentle bus EQ, saturation, fade, normalise
for c in range(2):
    mix[c] = fft_filter(mix[c], lo=32, hi=13500, order=2)
mix *= 0.95 / max(np.max(np.abs(mix)), 1e-9)      # tame into the saturator
mix = np.tanh(mix * 1.1) / np.tanh(1.1)

fade = np.ones(N)
fade_s = int(4.2 * SR)
fade[-fade_s:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_s)))
mix *= fade
mix *= np.clip(t_all / 0.08, 0, 1)
mix *= 0.89 / max(np.max(np.abs(mix)), 1e-9)

# ----------------------------------------------------------------------------
# write
# ----------------------------------------------------------------------------
from pathlib import Path as _P
OUT = str(_P(__file__).resolve().parent / "rainy_window.wav")
pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
with wave.open(OUT, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())

rms = float(np.sqrt(np.mean(mix ** 2)))
print(f"wrote {OUT}: {N / SR:.2f}s, peak {20*np.log10(np.max(np.abs(mix))):.2f} dBFS, "
      f"rms {20*np.log10(rms):.2f} dBFS")
