#!/usr/bin/env python3
"""
"West Lake, Jiangnan" — 西湖·江南

A sparser, more 江南丝竹 reading: 84 BPM, 箫 leads the first statement,
笛子 takes the second, 二胡 joins only on long tones, and the accompaniment
is reduced to single plucked notes on the 板, a quiet 笙 drone, and a very
light 木鱼 / 碰铃 pulse. Most of the bar is left empty on purpose.
"""
import wave
from pathlib import Path

import numpy as np

SR = 44100
BPM = 84.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
N_BARS = 36
TAIL = 6.0
DUR = N_BARS * BAR + TAIL
N = int(DUR * SR)

rng = np.random.default_rng(7788)


def midi2f(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def bt(bar, beat=0.0):
    return (bar - 1) * BAR + beat * BEAT


# ------------------------------------------------------------------ material
PENTA = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84, 86, 88]

CHORD_TONES = {
    "C":  [48, 55, 60, 64, 67],
    "G":  [43, 50, 55, 59, 62],
    "Am": [45, 52, 57, 60, 64],
    "F":  [41, 48, 53, 57, 60],
}
ROOT = {"C": 36, "G": 43, "Am": 45, "F": 41}
DRONE = {"C": 48, "G": 43, "Am": 45, "F": 41}

PROG = ["C", "G", "Am", "F"] * 4
PROG += ["F", "G", "C", "Am", "F", "G", "C", "C"]
PROG += ["C", "G", "Am", "F"] * 2
PROG += ["C", "G", "F", "C"]


# --------------------------------------------------------------------- buses
mix = np.zeros((2, N))
send = np.zeros((2, N))


def add(bus, sig, t0, pan=0.0, gain=1.0):
    i = int(round(t0 * SR))
    if i >= bus.shape[1] or i + len(sig) <= 0:
        return
    j = min(i + len(sig), bus.shape[1])
    seg = sig[: j - i]
    th = (pan + 1) * np.pi / 4
    bus[0, i:j] += seg * gain * np.cos(th)
    bus[1, i:j] += seg * gain * np.sin(th)


def place(sig, t0, pan=0.0, gain=1.0, rev=0.0, jitter=0.009):
    t = max(0.0, t0 + rng.normal(0, jitter))
    add(mix, sig, t, pan, gain)
    if rev:
        add(send, sig, t, pan * 0.7, gain * rev)


def fft_filter(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    H = np.ones_like(f)
    if hi:
        H *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        H *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-6)) ** (2 * order))
    return np.fft.irfft(X * H, len(x))


def gate(n, attack, release, curve=2.0):
    e = np.ones(n)
    a = min(max(int(attack * SR), 1), n)
    e[:a] = np.linspace(0, 1, a, endpoint=False) ** curve
    r = min(max(int(release * SR), 1), n)
    e[n - r:] = np.minimum(e[n - r:], np.linspace(1, 0, r, endpoint=False) ** curve)
    return e


# ---------------------------------------------------------------- instruments
def xiao(freq, dur, vel=0.8, slide_from=None):
    """End-blown flute: darker, breathier, softer than the dizi."""
    tail = 0.45
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 4.7 * t) * np.clip((t - 0.25) / 0.6, 0, 1)
    if slide_from:
        ft = freq * (1 + (slide_from / freq - 1) * np.exp(-t / 0.09)) * vib
    else:
        ft = freq * vib
    ph = 2 * np.pi * np.cumsum(ft) / SR
    y = (np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph))
    y += fft_filter(rng.standard_normal(n), lo=1200, hi=8000) * (0.055 + 0.06 * np.exp(-t / 0.5))
    y = fft_filter(y, hi=5200)
    y *= np.exp(-t / 5.0)
    y *= gate(n, 0.09, tail, curve=1.4)
    return y * vel * 0.30


def dizi(freq, dur, vel=0.8, slide_from=None):
    tail = 0.35
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.005 * np.sin(2 * np.pi * 5.6 * t) * np.clip((t - 0.18) / 0.5, 0, 1)
    if slide_from:
        ft = freq * (1 + (slide_from / freq - 1) * np.exp(-t / 0.07)) * vib
    else:
        ft = freq * vib
    ph = 2 * np.pi * np.cumsum(ft) / SR
    y = (np.sin(ph) + 0.36 * np.sin(2 * ph) + 0.18 * np.sin(3 * ph)
         + 0.08 * np.sin(4 * ph) + 0.04 * np.sin(5 * ph))
    y += fft_filter(rng.standard_normal(n), lo=2400, hi=9500) * 0.06 * np.exp(-t / 0.35)
    y *= np.exp(-t / 4.5)
    y *= gate(n, 0.045, tail, curve=1.4)
    return y * vel * 0.26


def erhu(freq, dur, vel=0.7, slide_from=None):
    tail = 0.3
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.8 * t) * np.clip((t - 0.1) / 0.45, 0, 1)
    if slide_from:
        ft = freq * (1 + (slide_from / freq - 1) * np.exp(-t / 0.11)) * vib
    else:
        ft = freq * vib
    ph = 2 * np.pi * np.cumsum(ft) / SR
    y = np.zeros(n)
    for k in range(1, 12):
        if k * freq < 8000:
            y += np.sin(k * ph + rng.uniform(0, 6.28)) / k
    y = fft_filter(y, lo=180, hi=4800)
    y = 0.78 * y + 0.5 * fft_filter(y, lo=900, hi=2500, order=2)
    y += fft_filter(rng.standard_normal(n), lo=900, hi=3500) * 0.02
    y *= gate(n, 0.09, tail, curve=1.4)
    return y * vel * 0.22


def pipa_pluck(freq, dur, vel=0.7):
    """Single stroke — no tremolo; the 轮指 is saved for the climax."""
    tail = 1.0
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, 9):
        f = freq * k * (1 + 0.0002 * k * k)
        if f < SR * 0.45:
            y += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / k ** 0.9 * np.exp(-t / (0.9 / k ** 0.7))
    y += fft_filter(rng.standard_normal(n), lo=2000, hi=8000) * np.exp(-t / 0.006) * 0.10
    y *= gate(n, 0.002, tail)
    y /= 2.2
    return y * vel


def pipa_tremolo(freq, dur, vel=0.6):
    n = int((dur + 1.0) * SR)
    y = np.zeros(n)
    step = 0.075
    for s in range(max(1, int(dur / step))):
        seg = pipa_pluck(freq, 0.12, 1.0)
        k = int(s * step * SR)
        j = min(k + len(seg), n)
        y[k:j] += seg[: j - k] * np.exp(-s * step / 1.2)
    return y * vel * 0.5


def yangqin(freq, dur, vel=0.6):
    tail = 1.4
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, 10):
        f = freq * k * (1 + 0.0003 * k * k)
        if f < SR * 0.45:
            y += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / (1.2 / k ** 0.7)) / k ** 0.9
    y += fft_filter(rng.standard_normal(n), lo=2500, hi=9000) * np.exp(-t / 0.005) * 0.10
    y *= gate(n, 0.0015, tail)
    y /= 2.2
    return y * vel


def guzheng(freq, dur, vel=0.8, bend=0.0):
    tail = 2.0
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    if bend:
        ph = 2 * np.pi * np.cumsum(freq * (1 + bend * np.exp(-t / 0.16))) / SR
    else:
        ph = 2 * np.pi * freq * t
    y = np.zeros(n)
    for k in range(1, 13):
        y += (1.0 / k ** 0.9) * np.sin(k * ph + rng.uniform(0, 6.28)) * np.exp(-t / (2.6 / k ** 0.8))
    y += fft_filter(rng.standard_normal(n), lo=1600, hi=8000) * np.exp(-t / 0.006) * 0.10
    y *= gate(n, 0.002, tail)
    y /= 2.6
    return y * vel


def gliss(notes, dur, vel=0.6):
    n = int((dur + 1.6) * SR)
    out = np.zeros(n)
    step = dur / max(1, len(notes))
    for i, m in enumerate(notes):
        seg = guzheng(midi2f(m), 0.5 * BEAT, vel * (0.55 + 0.45 * i / len(notes)))
        k = int(i * step * SR)
        j = min(k + len(seg), n)
        out[k:j] += seg[: j - k]
    return out


def sheng(freq, dur, vel=0.5):
    """Reed mouth-organ drone: soft, slightly beating, no chord."""
    n = int((dur + 1.2) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for cents in (-5.0, 5.0):
        f = freq * 2 ** (cents / 1200)
        y += (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)
              + 0.15 * np.sin(6 * np.pi * f * t))
    y = fft_filter(y, hi=1600, order=2)
    y *= gate(n, 0.8, 1.2, curve=1.4)
    return y * vel * 0.09


def muyu(vel=1.0):
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * 620 * t) + 0.5 * np.sin(2 * np.pi * 1180 * t)) * np.exp(-t / 0.02)
    y += fft_filter(rng.standard_normal(n), lo=900, hi=5000) * np.exp(-t / 0.004) * 0.4
    y *= gate(n, 0.001, 0.06)
    return y * vel * 0.14


def bell(freq=2349.0, vel=1.0):
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio, amp, dec in ((1, 1.0, 0.9), (2.32, 0.5, 0.5), (4.25, 0.3, 0.28)):
        y += amp * np.sin(2 * np.pi * freq * ratio * t + rng.uniform(0, 6.28)) * np.exp(-t / dec)
    y *= gate(n, 0.0015, 0.6)
    return y * vel * 0.07


def drum(vel=1.0):
    n = int(0.7 * SR)
    t = np.arange(n) / SR
    f = 62 + 28 * np.exp(-t / 0.05)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.25)
    y *= gate(n, 0.002, 0.4)
    return y * vel * 0.4


def bass_pluck(freq, dur, vel=0.8):
    tail = 0.7
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
         + 0.1 * np.sin(6 * np.pi * freq * t))
    y *= np.exp(-t / 0.9)
    y *= gate(n, 0.012, tail)
    return y * vel * 0.42


def pentatonic_above(m):
    for k in sorted(PENTA):
        if k > m:
            return k
    return m + 2



# --------------------------------------------------------------- new material
CHORD_TONES["Em"] = [52, 55, 59, 64]
ROOT["Em"] = 40
DRONE["Em"] = 52

PROG = ["C", "G", "Am", "F"]
PROG += ["C", "G", "Am", "F", "C", "G", "Am", "F"]        # 5-12   A
PROG += ["F", "G", "Em", "Am", "F", "G", "Em", "Am"]      # 13-20  B
PROG += ["C", "G", "Am", "F", "C", "G", "Am", "F"]        # 21-28  A'
PROG += ["F", "G", "C", "G"]                              # 29-32  C
PROG += ["C", "G", "F", "C"]                              # 33-36  outro

print(f"{N_BARS} bars @ {BPM:.0f} BPM  ->  {DUR:.1f}s")
SEC = {}
for b in range(1, N_BARS + 1):
    SEC[b] = ("intro" if b <= 4 else "A" if b <= 12 else "B" if b <= 20
              else "A2" if b <= 28 else "C" if b <= 32 else "outro")

# ---------------------------------------------------------------- arrangement
for bar in range(1, N_BARS + 1):
    ch = PROG[bar - 1]
    t0 = bt(bar)
    sec = SEC[bar]
    tones = CHORD_TONES[ch]

    if bar < 34:
        place(sheng(midi2f(DRONE[ch]), BAR * 0.98, 0.55 if sec in ("A", "A2") else 0.4),
              t0, -0.15, 1.0, rev=0.35)

    # 琵琶: one stroke per bar, a second only where the melody breathes
    if bar >= 5 and sec != "C":
        pat = [(0.0, 0)]
        if sec == "B" and bar % 2 == 0:
            pat.append((2.5, 2))
        if sec == "A2" and bar % 2 == 1:
            pat.append((1.5, 2))
        for beat, ix in pat:
            place(pipa_pluck(midi2f(tones[min(ix, len(tones) - 1)] + 12), 0.7 * BEAT, 0.48),
                  t0 + beat * BEAT, 0.35, 1.0, rev=0.35)

    # 扬琴: only at the two biggest cadences
    if bar in (12, 20, 28, 32):
        ix = 3 if bar % 8 == 0 else 1
        place(yangqin(midi2f(tones[min(ix, len(tones) - 1)] + 12), 0.9 * BEAT, 0.30),
              t0 + 3.0 * BEAT, -0.45, 1.0, rev=0.4)

    # 古筝: a short answer only where the melody truly rests
    if bar in (20, 28):
        for k, note in enumerate((72, 76, 79)):
            place(guzheng(midi2f(note), 0.6 * BEAT, 0.30), t0 + (3.05 + 0.28 * k) * BEAT,
                  0.5, 1.0, rev=0.45)
    if bar in (1, 5, 13, 21, 29, 33):
        run = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81]
        place(gliss(run if bar != 29 else run[::-1], 0.6, 0.5), t0 - 0.5, 0.35, 1.0, rev=0.45)

    place(bass_pluck(midi2f(ROOT[ch]), 2.0 * BEAT, 0.95), t0, 0.0, 1.25, rev=0.05)

    if bar >= 5:
        place(muyu(0.7), t0, -0.2, 1.0, rev=0.12)
    if bar in (9, 17, 25, 33):
        place(bell(2349.0, 0.4), t0 + 2 * BEAT, 0.45, 1.0, rev=0.35)
    if bar == 29:
        place(drum(0.4), t0, 0.0, 1.0, rev=0.25)

# ------------------------------------------------------------- one motif, developed
# cell: 3-5-6 up, then 6-5-3 down  (E-G-A | A-G-E), rhythm: 8th 8th quarter
A1 = [
    (0, 0.0, 76, .5, .72), (0, 0.5, 79, .5, .66), (0, 1.0, 81, 1.0, .76),
    (0, 2.0, 79, .5, .64), (0, 2.5, 76, 1.5, .70),
    (1, 0.0, 74, .5, .66), (1, 0.5, 76, .5, .64), (1, 1.0, 79, 1.0, .72),
    (1, 2.0, 81, .5, .68), (1, 2.5, 79, 1.5, .68),
    (2, 0.0, 81, 1.0, .74), (2, 1.0, 79, .5, .64), (2, 1.5, 76, .5, .62),
    (2, 2.0, 74, 1.0, .66), (2, 3.0, 72, 1.0, .64),
    (3, 0.0, 74, .5, .62), (3, 0.5, 76, .5, .60), (3, 1.0, 79, 1.0, .68),
    (3, 2.0, 76, 2.0, .66),
]
# consequent: the same cell a third higher, cadencing on C
A2 = [
    (0, 0.0, 79, .5, .72), (0, 0.5, 81, .5, .68), (0, 1.0, 84, 1.0, .78),
    (0, 2.0, 81, .5, .68), (0, 2.5, 79, 1.5, .70),
    (1, 0.0, 81, .5, .70), (1, 0.5, 79, .5, .64), (1, 1.0, 76, 1.0, .70),
    (1, 2.0, 74, .5, .62), (1, 2.5, 76, 1.5, .66),
    (2, 0.0, 76, 1.0, .68), (2, 1.0, 74, .5, .60), (2, 1.5, 72, .5, .58),
    (2, 2.0, 74, 1.0, .62), (2, 3.0, 76, 1.0, .64),
    (3, 0.0, 74, .5, .60), (3, 0.5, 76, .5, .58), (3, 1.0, 81, 2.5, .66),
]
# B: the cell inverted (down first), longer notes, same rhythm underneath
B1 = [
    (0, 0.0, 81, .5, .74), (0, 0.5, 79, .5, .66), (0, 1.0, 76, 1.5, .70),
    (0, 2.5, 79, .5, .62), (0, 3.0, 81, 1.0, .70),
    (1, 0.0, 79, 1.0, .70), (1, 1.0, 76, .5, .62), (1, 1.5, 74, 1.5, .66),
    (1, 3.0, 76, 1.0, .64),
    (2, 0.0, 76, .5, .66), (2, 0.5, 79, .5, .64), (2, 1.0, 83, 1.5, .74),
    (2, 2.5, 81, .5, .64), (2, 3.0, 79, 1.0, .66),
    (3, 0.0, 81, 2.5, .72), (3, 2.5, 79, .5, .60), (3, 3.0, 76, 1.0, .62),
]
B2 = [
    (0, 0.0, 84, .5, .76), (0, 0.5, 81, .5, .68), (0, 1.0, 79, 1.5, .72),
    (0, 2.5, 76, .5, .62), (0, 3.0, 79, 1.0, .68),
    (1, 0.0, 81, 1.0, .72), (1, 1.0, 79, .5, .64), (1, 1.5, 76, 1.5, .68),
    (1, 3.0, 74, 1.0, .62),
    (2, 0.0, 76, .5, .64), (2, 0.5, 79, .5, .62), (2, 1.0, 81, 1.0, .70),
    (2, 2.0, 83, .5, .66), (2, 2.5, 81, .5, .64), (2, 3.0, 79, 1.0, .64),
    (3, 0.0, 81, 3.0, .72),
]
# A' : same opening as A, but the consequent climbs to the climax
A3 = [
    (0, 0.0, 76, .5, .72), (0, 0.5, 79, .5, .66), (0, 1.0, 81, 1.0, .76),
    (0, 2.0, 79, .5, .64), (0, 2.5, 76, 1.5, .70),
    (1, 0.0, 74, .5, .66), (1, 0.5, 76, .5, .64), (1, 1.0, 79, 1.0, .72),
    (1, 2.0, 81, .5, .68), (1, 2.5, 79, 1.5, .68),
    (2, 0.0, 79, .5, .70), (2, 0.5, 81, .5, .68), (2, 1.0, 84, 1.0, .76),
    (2, 2.0, 81, .5, .68), (2, 2.5, 79, 1.5, .70),
    (3, 0.0, 81, .5, .70), (3, 0.5, 84, .5, .68), (3, 1.0, 86, 1.0, .76),
    (3, 2.0, 84, .5, .68), (3, 2.5, 81, 1.5, .70),
]
# climax: the cell in sequence, then a cadence back to A'
C1 = [
    (0, 0.0, 81, .5, .78), (0, 0.5, 84, .5, .74), (0, 1.0, 86, 1.0, .82),
    (0, 2.0, 84, .5, .72), (0, 2.5, 81, 1.5, .74),
    (1, 0.0, 79, .5, .72), (1, 0.5, 81, .5, .70), (1, 1.0, 84, 1.0, .78),
    (1, 2.0, 81, .5, .70), (1, 2.5, 79, 1.5, .72),
    (2, 0.0, 76, .5, .70), (2, 0.5, 79, .5, .68), (2, 1.0, 81, 1.0, .76),
    (2, 2.0, 79, .5, .68), (2, 2.5, 76, 1.5, .70),
    (3, 0.0, 74, .5, .68), (3, 0.5, 76, .5, .66), (3, 1.0, 79, 1.0, .72),
    (3, 2.0, 81, 2.0, .74),
]
OUT_END = [
    (0, 0.0, 79, 1.0, .66), (0, 1.0, 76, .5, .58), (0, 1.5, 74, 1.0, .62), (0, 2.5, 72, 1.0, .60),
    (1, 0.0, 74, 1.5, .60), (1, 1.5, 76, .5, .56), (1, 2.0, 79, 1.0, .62),
    (2, 0.0, 76, 1.0, .60), (2, 1.0, 74, 1.0, .58), (2, 2.0, 72, 2.0, .62),
]


def play(notes, bar0, inst, gain=1.0, octave=0, grace=0.0):
    prev = None
    for obar, beat, note, blen, vel in notes:
        bar = bar0 + obar
        m = note + octave
        src = midi2f(prev - 2) if prev is not None else None
        if grace and blen >= 0.75 and rng.random() < grace:
            place(inst(midi2f(pentatonic_above(m)), 0.12 * BEAT, vel * 0.42),
                  bt(bar, beat) - 0.12 * BEAT, 0.2, 0.8, rev=0.4)
        place(inst(midi2f(m), blen * BEAT, vel * gain, slide_from=src),
              bt(bar, beat), -0.1, 1.0, rev=0.42)
        prev = m


# a short xiao call before the tune enters (bars 3-4)
INTRO_CALL = [(2, 2.0, 69, 1.0, .55), (2, 3.0, 72, 1.0, .55),
              (3, 0.0, 76, 1.5, .60), (3, 1.5, 74, 1.0, .52), (3, 2.5, 72, 1.5, .55)]
play(INTRO_CALL, 1, xiao, 0.85, grace=0.25)

# main line: dizi throughout, xiao doubles the first statement for depth
play(A1, 5, dizi, 0.98, grace=0.22)
play(A1, 5, xiao, 0.45, octave=-12, grace=0.0)
play(A2, 9, dizi, 0.98, grace=0.20)
play(B1, 13, dizi, 1.0, grace=0.18)
play(B2, 17, dizi, 1.0, grace=0.18)
play(A3, 21, dizi, 1.02, grace=0.20)
play(C1, 29, dizi, 1.05, grace=0.12)
play(OUT_END, 33, xiao, 0.9, octave=-12, grace=0.2)

# erhu shadows the long notes all the way through (legato glue)
for phrase, bar0 in ((A2, 9), (B1, 13), (B2, 17), (A3, 21), (C1, 29)):
    for obar, beat, note, blen, vel in phrase:
        if blen >= 1.0:
            place(erhu(midi2f(note - 12), blen * BEAT, vel * 0.36),
                  bt(bar0 + obar, beat) + 0.03, 0.5, 1.0, rev=0.45)

for obar, beat, note, blen, vel in C1:
    if blen >= 1.0:
        place(pipa_tremolo(midi2f(note - 12), blen * BEAT, vel * 0.35),
              bt(29 + obar, beat), 0.45, 1.0, rev=0.4)

for k, note in enumerate((36, 48, 55, 60, 64)):
    place(guzheng(midi2f(note), 1.2 * BEAT, 0.45 - 0.04 * k),
          bt(36, 0.0) + 0.07 * k, -0.3 + 0.12 * k, 1.0, rev=0.5)
place(bell(2349.0, 0.4), bt(36, 0.0) + 0.5, 0.4, 1.0, rev=0.5)

# --------------------------------------------------------------------- space
def make_ir(sec=2.2, decay=0.9):
    n = int(sec * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / decay)
    ir = fft_filter(ir, lo=250, hi=6000, order=1)
    return ir / np.sqrt(np.sum(ir ** 2))


def convolve(x, h):
    n = len(x) + len(h) - 1
    Nf = 1 << int(np.ceil(np.log2(n)))
    y = np.fft.irfft(np.fft.rfft(x, Nf) * np.fft.rfft(h, Nf), Nf)
    return y[:len(x)]


print("reverb...")
rev = np.stack([convolve(send[0], make_ir()), convolve(send[1], make_ir())])
mix += 0.36 * rev

for c in range(2):
    mix[c] = fft_filter(mix[c], lo=40, hi=17000, order=2)

mid = (mix[0] + mix[1]) * 0.5
side = (mix[0] - mix[1]) * 0.5 * 1.45
mix[0], mix[1] = mid + side, mid - side


def compress(x, thresh_db=-30.0, ratio=2.0, win_s=0.06):
    win = int(win_s * SR)
    env = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, "same") + 1e-12)
    env_db = 20 * np.log10(env + 1e-9)
    over = np.maximum(env_db - thresh_db, 0.0)
    gain = 10 ** (-(over * (1 - 1 / ratio)) / 20)
    k = int(0.04 * SR)
    gain = np.convolve(gain, np.ones(k) / k, "same")
    return x * gain


for c in range(2):
    mix[c] = compress(mix[c])
mix *= 0.98 / max(np.max(np.abs(mix)), 1e-9)
mix = np.tanh(mix * 1.02) / np.tanh(1.02)
mix *= 0.88 / max(np.max(np.abs(mix)), 1e-9)

t_all = np.arange(N) / SR
fade = np.ones(N)
fs = int(5.0 * SR)
fade[-fs:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, fs)))
mix *= fade
mix *= np.clip(t_all / 0.06, 0, 1)

OUT = str(Path(__file__).resolve().parent / "westlake_jiangnan3.wav")
pcm = (np.clip(mix.T, -1, 1) * 32767).astype("<i2")
with wave.open(OUT, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())

rms = float(np.sqrt(np.mean(mix ** 2)))
print(f"wrote {OUT}: {N / SR:.2f}s, peak {20*np.log10(np.max(np.abs(mix))):.2f} dBFS, "
      f"rms {20*np.log10(rms):.2f} dBFS")
