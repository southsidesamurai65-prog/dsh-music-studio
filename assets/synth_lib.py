#!/usr/bin/env python3
"""
synth_lib.py — a small, dependency-free instrument library for DSH Music Studio.

Everything is additive synthesis with numpy: no samples, no scipy.
Import what you need, or call `demo()` for a two-bar cell.

    import synth_lib as sl
    sr = sl.SR
    y = sl.guzheng(sl.midi2f(69), 1.0, 0.8)
"""
import wave

import numpy as np

SR = 44100
_rng = np.random.default_rng(0)
_cache = {}


def seed(n):
    """Make renders reproducible."""
    global _rng
    _rng = np.random.default_rng(n)


def midi2f(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# --------------------------------------------------------------------- helpers
def fft_filter(x, lo=None, hi=None, order=2):
    """Zero-phase Butterworth-magnitude filter."""
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


def add(bus, sig, t0, pan=0.0, gain=1.0):
    """Mix a mono signal into a (2, N) bus with equal-power panning."""
    i = int(round(t0 * SR))
    if i >= bus.shape[1] or i + len(sig) <= 0:
        return
    j = min(i + len(sig), bus.shape[1])
    seg = sig[: j - i]
    th = (pan + 1) * np.pi / 4
    bus[0, i:j] += seg * gain * np.cos(th)
    bus[1, i:j] += seg * gain * np.sin(th)


# --------------------------------------------------------------------- stems
class Stems:
    """Role-separated render buffers, so the mixer can still tell a snare from
    a pad after the notes are placed.

    Rendering into one `bus` is irreversible: once the kick and the strings are
    summed, no compressor can tell them apart — which is why multiband
    compression on a full mix blunts the backbeat. Render into stems, then let
    `mix.mixdown()` decide what gets compressed.

        st = sl.Stems(seconds)
        st.add("drums", sl.snare(), t0, 0.05, 1.0)
        st.add("music", sl.piano(f, 2.0), t0, 0.2, 0.5)
        bus = mx.mixdown(st, preset="pop", ir=ir)
    """

    def __init__(self, seconds, sr=SR, roles=("drums", "bass", "music", "fx")):
        self.sr = sr
        self.n = int(round(seconds * sr))
        self.buf = {r: np.zeros((2, self.n)) for r in roles}

    def add(self, role, sig, t0, pan=0.0, gain=1.0):
        """Place a mono signal into a stem (same semantics as `add`)."""
        if sig is None or len(sig) == 0:
            return
        if role not in self.buf:
            self.buf[role] = np.zeros((2, self.n))
        add(self.buf[role], sig, t0, pan, gain)

    def play(self, instrument, sig, t0, pan=0.0, gain=1.0):
        """Like `add`, but pick the stem from the instrument's role in
        INSTRUMENTS ("snare" -> drums, "ebass" -> bass, "dizi" -> music)."""
        self.add(role_of(instrument), sig, t0, pan, gain)

    def __getitem__(self, role):
        return self.buf[role]

    def __contains__(self, role):
        return role in self.buf

    def keys(self):
        return list(self.buf)

    def total(self):
        out = np.zeros((2, self.n))
        for v in self.buf.values():
            out += v
        return out

    def peak_db(self, role):
        p = float(np.max(np.abs(self.buf[role]))) if role in self.buf else 0.0
        return 20 * np.log10(p + 1e-12)

    def rms_db(self, role):
        x = self.buf[role] if role in self.buf else np.zeros((2, self.n))
        return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)

    def trim(self, role, db):
        """Set the stem's fader. Balance is the composer's call, loudness the
        limiter's."""
        if role in self.buf:
            self.buf[role] *= 10 ** (db / 20)
        return self

    def report(self):
        rows = []
        for r in self.buf:
            if np.any(self.buf[r]):
                rows.append(f"  {r:6s} peak {self.peak_db(r):6.1f} dB  "
                            f"rms {self.rms_db(r):6.1f} dB")
        return "stems:\n" + "\n".join(rows)


def role_of(instrument):
    """Map an instrument name to its mix stem via INSTRUMENTS."""
    for role, names in INSTRUMENTS.items():
        if instrument in names:
            return "drums" if role == "drums" else ("bass" if role == "bass" else "music")
    return "music"


# ------------------------------------------------------------------ plucked
def guzheng(freq, dur, vel=0.8, bend=0.0, bright=1.0):
    """Chinese zither: bright pick, inharmonic shimmer, long decay."""
    key = ("gz", round(freq, 3), round(dur, 2), round(bend, 3), round(bright, 2))
    if key in _cache:
        return _cache[key] * vel
    tail = 2.4
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    if bend:
        ph = 2 * np.pi * np.cumsum(freq * (1 + bend * np.exp(-t / 0.16))) / SR
    else:
        ph = 2 * np.pi * freq * t
    y = np.zeros(n)
    for k in range(1, 15):
        y += (1.0 / k ** 0.9) * np.sin(k * ph + _rng.uniform(0, 2 * np.pi)) * np.exp(-t / (3.0 / k ** 0.8))
    y += fft_filter(_rng.standard_normal(n), lo=1500, hi=8000) * np.exp(-t / 0.006) * 0.11 * bright
    y *= gate(n, 0.002, tail)
    _cache[key] = y / 2.6
    return _cache[key] * vel


def pipa_pluck(freq, dur, vel=0.7):
    """Pipa single stroke."""
    tail = 1.0
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, 9):
        f = freq * k * (1 + 0.0002 * k * k)
        if f < SR * 0.45:
            y += np.sin(2 * np.pi * f * t + _rng.uniform(0, 6.28)) * np.exp(-t / (0.9 / k ** 0.7)) / k ** 0.9
    y += fft_filter(_rng.standard_normal(n), lo=2000, hi=8000) * np.exp(-t / 0.006) * 0.10
    y *= gate(n, 0.002, tail)
    return y / 2.2 * vel


def pipa_tremolo(freq, dur, vel=0.6, rate=0.075):
    """轮指: rapid repeated strokes."""
    n = int((dur + 1.0) * SR)
    y = np.zeros(n)
    for s in range(max(1, int(dur / rate))):
        seg = pipa_pluck(freq, 0.12, 1.0)
        k = int(s * rate * SR)
        j = min(k + len(seg), n)
        y[k:j] += seg[: j - k] * np.exp(-s * rate / 1.2)
    return y * vel * 0.5


def yangqin(freq, dur, vel=0.6):
    """Hammered dulcimer."""
    tail = 1.4
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k in range(1, 10):
        f = freq * k * (1 + 0.0003 * k * k)
        if f < SR * 0.45:
            y += np.sin(2 * np.pi * f * t + _rng.uniform(0, 6.28)) * np.exp(-t / (1.2 / k ** 0.7)) / k ** 0.9
    y += fft_filter(_rng.standard_normal(n), lo=2500, hi=9000) * np.exp(-t / 0.005) * 0.10
    y *= gate(n, 0.0015, tail)
    return y / 2.2 * vel


# ------------------------------------------------------------------- winds
def dizi(freq, dur, vel=0.8, slide_from=None):
    """Membrane bamboo flute: breath, vibrato, portamento."""
    tail = 0.35
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.005 * np.sin(2 * np.pi * 5.6 * t) * np.clip((t - 0.18) / 0.5, 0, 1)
    ft = freq * vib * ((1 + (slide_from / freq - 1) * np.exp(-t / 0.07)) if slide_from else 1)
    ph = 2 * np.pi * np.cumsum(ft) / SR
    y = (np.sin(ph) + 0.38 * np.sin(2 * ph) + 0.18 * np.sin(3 * ph)
         + 0.08 * np.sin(4 * ph) + 0.04 * np.sin(5 * ph))
    y += fft_filter(_rng.standard_normal(n), lo=2400, hi=9000) * 0.05 * np.exp(-t / 0.35)
    y *= np.exp(-t / 4.5) * gate(n, 0.045, tail, curve=1.4)
    return y * vel * 0.27


def xiao(freq, dur, vel=0.8, slide_from=None):
    """End-blown flute: darker, breathier than dizi."""
    tail = 0.45
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 4.7 * t) * np.clip((t - 0.25) / 0.6, 0, 1)
    ft = freq * vib * ((1 + (slide_from / freq - 1) * np.exp(-t / 0.09)) if slide_from else 1)
    ph = 2 * np.pi * np.cumsum(ft) / SR
    y = np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    y += fft_filter(_rng.standard_normal(n), lo=1200, hi=6500) * (0.05 + 0.05 * np.exp(-t / 0.5))
    y = fft_filter(y, hi=5200) * np.exp(-t / 5.0) * gate(n, 0.09, tail, curve=1.4)
    return y * vel * 0.30


def erhu(freq, dur, vel=0.7, slide_from=None):
    """Bowed two-string: saw body, nasal formant, portamento."""
    tail = 0.3
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.9 * t) * np.clip((t - 0.1) / 0.45, 0, 1)
    ft = freq * vib * ((1 + (slide_from / freq - 1) * np.exp(-t / 0.11)) if slide_from else 1)
    ph = 2 * np.pi * np.cumsum(ft) / SR
    y = np.zeros(n)
    for k in range(1, 12):
        if k * freq < 8000:
            y += np.sin(k * ph + _rng.uniform(0, 6.28)) / k
    y = fft_filter(y, lo=180, hi=5000)
    y = 0.78 * y + 0.5 * fft_filter(y, lo=900, hi=2500, order=2)
    y += fft_filter(_rng.standard_normal(n), lo=900, hi=3500) * 0.02
    y *= gate(n, 0.09, tail, curve=1.4)
    return y * vel * 0.23


# ---------------------------------------------------------------- keys / pads
def rhodes(freq, dur, vel=0.7, bright=1.0):
    """Electric piano."""
    key = ("rh", round(freq, 3), round(dur, 2), round(bright, 2))
    if key in _cache:
        return _cache[key] * vel
    tail = 0.55
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio, amp, dec in ((1.0, 1.0, 1.9), (2.0, 0.30 * bright, 1.0), (3.0, 0.10 * bright, 0.55),
                            (4.02, 0.05 * bright, 0.30), (6.05, 0.02 * bright, 0.18)):
        if freq * ratio < SR * 0.45:
            y += amp * np.sin(2 * np.pi * freq * ratio * t + _rng.uniform(0, 2 * np.pi)) * np.exp(-t / dec)
    y += fft_filter(_rng.standard_normal(n), lo=1500, hi=9000) * np.exp(-t / 0.006) * 0.10 * bright
    y *= gate(n, 0.005, tail)
    _cache[key] = y / 2.2
    return _cache[key] * vel


def pad(freq, dur, vel=1.0, cutoff=1600):
    """Detuned saw pad: slow attack, no transients."""
    key = ("pad", round(freq, 3), round(dur, 2), cutoff)
    if key in _cache:
        return _cache[key] * vel
    attack, release = 1.2, 2.2
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for dt in (-6.0, 6.0):
        f = freq * 2 ** (dt / 1200)
        for k in range(1, max(1, min(9, int(0.42 * SR / f))) + 1):
            out += np.sin(2 * np.pi * f * k * t + _rng.uniform(0, 2 * np.pi)) / k
    out = fft_filter(out, lo=90, hi=cutoff, order=2) * gate(n, attack, release, curve=1.5)
    _cache[key] = out / 8.0
    return _cache[key] * vel


def piano(freq, dur, vel=0.8, bright=1.0):
    """Struck-string piano: inharmonic partials, hammer noise, long decay."""
    key = ("pno", round(freq, 3), round(dur, 2), round(bright, 2))
    if key in _cache:
        return _cache[key] * vel
    n = int((dur + 2.2) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    B = 0.00035 * (1.0 + 120.0 / max(freq, 60.0))
    for k in range(1, 13):
        f = freq * k * np.sqrt(1 + B * k * k)
        if f > 0.45 * SR:
            break
        amp = (bright ** (k - 1)) / k ** 1.5
        y += amp * np.sin(2 * np.pi * f * t + _rng.uniform(0, 6.28)) * np.exp(-t / (1.9 / (1 + 0.45 * k)))
    y += fft_filter(_rng.standard_normal(n), lo=1800, hi=8000) * np.exp(-t / 0.005) * 0.18
    y = fft_filter(y * gate(n, 0.003, 0.35, curve=1.2), lo=35, hi=11000, order=1)
    _cache[key] = y * 0.30
    return _cache[key] * vel


def strings(freq, dur, vel=0.6, bright=1.0):
    """String section: three detuned bows, slow attack, vibrato."""
    n = int((dur + 1.1) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.0045 * np.sin(2 * np.pi * 5.1 * t + _rng.uniform(0, 6.28)) * np.clip((t - 0.15) / 0.6, 0, 1)
    y = np.zeros(n)
    for cents in (-7.0, 0.0, 7.0):
        f = freq * 2 ** (cents / 1200)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        top = max(1, int(min(6000.0, 0.42 * SR) / max(f, 40.0)))
        for k in range(1, top + 1):
            y += np.sin(k * ph + _rng.uniform(0, 6.28)) / (k ** 1.2)
    y = fft_filter(y, lo=110, hi=5200, order=2) * gate(n, 0.22, 0.9, curve=1.3)
    return y * vel * 0.055


def ebass(freq, dur, vel=0.8, slide=0.0):
    """Electric finger bass with an optional slide into the note."""
    n = int((dur + 0.4) * SR)
    t = np.arange(n) / SR
    f = freq * (1 + slide * np.exp(-t / 0.05))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
    y = np.tanh(y * 1.8) * np.exp(-t / 1.4) * gate(n, 0.006, 0.3)
    return fft_filter(y, hi=2600) * vel * 0.42


def sheng(freq, dur, vel=0.5):
    """Reed mouth-organ drone: soft, slightly beating."""
    n = int((dur + 1.2) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for cents in (-5.0, 5.0):
        f = freq * 2 ** (cents / 1200)
        y += np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.15 * np.sin(6 * np.pi * f * t)
    y = fft_filter(y, hi=1600, order=2) * gate(n, 0.8, 1.2, curve=1.4)
    return y * vel * 0.20


# --------------------------------------------------------------------- bass
def bass(freq, dur, vel=0.8):
    tail = 0.35
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * freq * t) + 0.22 * np.sin(4 * np.pi * freq * t) + 0.06 * np.sin(6 * np.pi * freq * t)
    y = np.tanh(y * 1.5) / 1.5 * np.exp(-t / 1.6) * gate(n, 0.01, tail)
    return y * vel * 0.5


def bass_pluck(freq, dur, vel=0.8):
    """Short, woody low pluck (ruan-like)."""
    tail = 0.6
    n = int((dur + tail) * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
         + 0.1 * np.sin(6 * np.pi * freq * t)) * np.exp(-t / 0.8) * gate(n, 0.01, tail)
    return y * vel * 0.42


# ---------------------------------------------------------------- percussion
def kick(vel=1.0):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    f = 48 + 80 * np.exp(-t / 0.028)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    y += _rng.standard_normal(n) * np.exp(-t / 0.004) * 0.2
    return y * gate(n, 0.002, 0.3) * vel * 0.8


def rim(vel=1.0):
    n = int(0.28 * SR)
    t = np.arange(n) / SR
    y = fft_filter(_rng.standard_normal(n), lo=1400, hi=5200) * np.exp(-t / 0.045) * 1.6
    y += 0.25 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    return y * gate(n, 0.002, 0.22) * vel * 0.3


def snare(vel=1.0):
    """Tight pop backbeat: shell tone plus band-limited noise."""
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    body = 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.055)
    body += 0.28 * np.sin(2 * np.pi * 330 * t) * np.exp(-t / 0.035)
    noise = fft_filter(_rng.standard_normal(n), lo=1400, hi=9500) * np.exp(-t / 0.10)
    return (body + noise) * gate(n, 0.001, 0.12) * vel * 0.42


def shaker(vel=1.0):
    """Hi-hat/shaker 16th layer."""
    n = int(0.14 * SR)
    t = np.arange(n) / SR
    y = fft_filter(_rng.standard_normal(n), lo=6000, hi=15000) * np.exp(-t / 0.05)
    return y * gate(n, 0.001, 0.05) * vel * 0.10


def hat(open_=False, vel=1.0):
    n = int((0.32 if open_ else 0.12) * SR)
    t = np.arange(n) / SR
    y = fft_filter(_rng.standard_normal(n), lo=6800, hi=15000) * np.exp(-t / (0.16 if open_ else 0.028))
    return y * gate(n, 0.001, 0.05) * vel * (0.13 if open_ else 0.16)


def drum_low(vel=1.0):
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    f = 55 + 45 * np.exp(-t / 0.05)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.35)
    y += _rng.standard_normal(n) * np.exp(-t / 0.01) * 0.12
    return y * gate(n, 0.003, 0.6) * vel * 0.5


def muyu(vel=1.0):
    n = int(0.18 * SR)
    t = np.arange(n) / SR
    y = (np.sin(2 * np.pi * 620 * t) + 0.5 * np.sin(2 * np.pi * 1180 * t)) * np.exp(-t / 0.02)
    y += fft_filter(_rng.standard_normal(n), lo=900, hi=5000) * np.exp(-t / 0.004) * 0.4
    return y * gate(n, 0.001, 0.06) * vel * 0.14


def bell(freq=2349.0, vel=1.0):
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio, amp, dec in ((1, 1.0, 0.9), (2.32, 0.5, 0.5), (4.25, 0.3, 0.28)):
        y += amp * np.sin(2 * np.pi * freq * ratio * t + _rng.uniform(0, 6.28)) * np.exp(-t / dec)
    return y * gate(n, 0.0015, 0.6) * vel * 0.08


def gong(freq=98, vel=1.0):
    n = int(7.0 * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for ratio, amp, dec in ((1.0, 1.0, 3.2), (2.01, 0.5, 2.2), (2.98, 0.3, 1.5),
                            (4.2, 0.2, 1.0), (5.4, 0.12, 0.7), (7.1, 0.07, 0.45)):
        y += amp * np.sin(2 * np.pi * freq * ratio * t + _rng.uniform(0, 6.28)) * np.exp(-t / dec)
    return y * gate(n, 0.004, 3.0) * vel * 0.2


def violin(freq, dur, vel=0.7, bright=1.0):
    """Bowed violin: brighter and more focused than the erhu."""
    n = int((dur + 0.7) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.005 * np.sin(2 * np.pi * 5.6 * t) * np.clip((t - 0.12) / 0.5, 0, 1)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    y = np.zeros(n)
    for k in range(1, 16):
        if k * freq < 9000:
            y += np.sin(k * ph + _rng.uniform(0, 6.28)) / k ** 0.85
    y = fft_filter(y, lo=180, hi=6500) * gate(n, 0.10, 0.55, curve=1.4)
    y += fft_filter(_rng.standard_normal(n), lo=2000, hi=7000) * 0.015
    return y * vel * 0.10 * bright


def cello(freq, dur, vel=0.7):
    """Bowed cello: dark, wide, slow attack."""
    n = int((dur + 0.9) * SR)
    t = np.arange(n) / SR
    vib = 1 + 0.004 * np.sin(2 * np.pi * 4.9 * t) * np.clip((t - 0.18) / 0.6, 0, 1)
    ph = 2 * np.pi * np.cumsum(freq * vib) / SR
    y = np.zeros(n)
    for k in range(1, 12):
        if k * freq < 5000:
            y += np.sin(k * ph + _rng.uniform(0, 6.28)) / k ** 1.1
    y = fft_filter(y, lo=60, hi=3200) * gate(n, 0.16, 0.7, curve=1.4)
    return y * vel * 0.16


def choir(freq, dur, vel=0.6):
    """'Ah' choir: detuned voices shaped by an /a/ formant envelope."""
    n = int((dur + 1.2) * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for cents in (-9.0, 0.0, 9.0):
        f = freq * 2 ** (cents / 1200)
        ph = 2 * np.pi * np.cumsum(f * (1 + 0.003 * np.sin(2 * np.pi * 4.6 * t))) / SR
        for k in range(1, 18):
            if k * f < 7000:
                y += np.sin(k * ph + _rng.uniform(0, 6.28)) / k ** 0.9
    # /a/ formant peaks (male), applied over the whole note
    X = np.fft.rfft(y)
    fr = np.fft.rfftfreq(n, 1 / SR)
    env = np.zeros_like(fr)
    for fc, bw, g in ((730, 90, 1.0), (1090, 110, 0.7), (2440, 160, 0.35), (3400, 220, 0.2)):
        env += g / (1 + ((fr - fc) / (bw / 2)) ** 2)
    y = np.fft.irfft(X * env, n)
    y = fft_filter(y, lo=110, hi=5200) * gate(n, 0.35, 1.0, curve=1.3)
    return y * vel * 0.20


def clap(vel=1.0):
    """Three quick noise bursts — a hand clap."""
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for d in (0.0, 0.011, 0.023):
        i = int(d * SR)
        seg = fft_filter(_rng.standard_normal(n - i), lo=1200, hi=7000) * np.exp(-t[: n - i] / 0.045)
        y[i:] += seg * 0.6
    return y * gate(n, 0.001, 0.15) * vel * 0.30


# --------------------------------------------------------------------- space
def make_ir(sec=2.4, decay=1.0, lo=200, hi=6500):
    n = int(sec * SR)
    t = np.arange(n) / SR
    ir = _rng.standard_normal(n) * np.exp(-t / decay)
    return fft_filter(ir, lo=lo, hi=hi, order=1) / np.sqrt(n)


def convolve(x, h):
    n = len(x) + len(h) - 1
    nf = 1 << int(np.ceil(np.log2(n)))
    return np.fft.irfft(np.fft.rfft(x, nf) * np.fft.rfft(h, nf), nf)[:len(x)]


def apply_reverb(mix, send, wet=0.5, sec=2.4, decay=1.0, lo=200, hi=6500):
    rev = np.stack([convolve(send[0], make_ir(sec, decay, lo, hi)),
                    convolve(send[1], make_ir(sec, decay, lo, hi))])
    return mix + wet * rev


# ----------------------------------------------------------------- mastering
def compress(x, thresh_db=-30.0, ratio=2.0, win_s=0.06):
    win = int(win_s * SR)
    env = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, "same") + 1e-12)
    over = np.maximum(20 * np.log10(env + 1e-9) - thresh_db, 0.0)
    gain = 10 ** (-(over * (1 - 1 / ratio)) / 20)
    k = int(0.04 * SR)
    return x * np.convolve(gain, np.ones(k) / k, "same")


def master(mix, peak=0.89, lowpass=16000, highpass=35, air_db=0.0, sat=1.05):
    """High-pass, low-pass, optional air shelf, soft saturation, peak normalise."""
    for c in range(2):
        x = fft_filter(mix[c], lo=highpass, hi=lowpass, order=2)
        if air_db:
            X = np.fft.rfft(x)
            f = np.fft.rfftfreq(len(x), 1 / SR)
            X *= 10 ** (air_db / 20 * f ** 2 / (f ** 2 + 5000 ** 2))
            x = np.fft.irfft(X, len(x))
        mix[c] = x
    mix *= 0.95 / max(np.max(np.abs(mix)), 1e-9)
    mix = np.tanh(mix * sat) / np.tanh(sat)
    return mix * peak / max(np.max(np.abs(mix)), 1e-9)


def write_wav(path, stereo, fade_out=3.0):
    """Write a (2, N) float bus as 16-bit stereo, with a cosine fade."""
    x = np.array(stereo, dtype=np.float64)
    n = x.shape[1]
    if fade_out > 0:
        fs = min(int(fade_out * SR), n)
        x *= np.concatenate([np.ones(n - fs), 0.5 * (1 + np.cos(np.linspace(0, np.pi, fs)))])
    pcm = (np.clip(x.T, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def demo(seconds=8.0, bpm=96.0):
    """Render a short C-major pentatonic cell to demo.wav."""
    seed(7)
    beat = 60 / bpm
    n = int(seconds * SR)
    mix, send = np.zeros((2, n)), np.zeros((2, n))
    notes = [72, 74, 76, 79, 81, 79, 76, 74]
    for i, m in enumerate(notes):
        t0 = i * beat
        sig = guzheng(midi2f(m), 0.8 * beat, 0.7)
        add(mix, sig, t0, -0.4 + 0.1 * i, 1.0)
        add(send, sig, t0, 0.0, 0.4)
        add(mix, dizi(midi2f(m + 12), 0.9 * beat, 0.35), t0 + 0.02, 0.1, 1.0)
    for i in range(0, int(seconds / beat), 2):
        add(mix, muyu(0.5), i * beat, -0.2, 1.0)
    mix = apply_reverb(mix, send, 0.4)
    mix = master(mix, 0.5)
    return write_wav("demo.wav", mix)


if __name__ == "__main__":
    print(demo())


# ------------------------------------------------------------------- palettes
# Pick by ROLE, not by "the one I used last time".
INSTRUMENTS = {
    "lead": ["dizi", "xiao", "erhu", "violin", "cello", "piano", "rhodes", "choir"],
    "counter": ["erhu", "violin", "cello", "xiao", "pipa_tremolo"],
    "rhythm": ["piano", "guzheng", "pipa_pluck", "yangqin", "rhodes"],
    "bass": ["bass", "bass_pluck", "ebass", "cello"],
    "drums": ["kick", "snare", "hat", "shaker", "clap", "rim", "drum_low", "muyu"],
    "pad": ["pad", "strings", "choir", "sheng"],
    "colour": ["guzheng", "pipa_pluck", "yangqin", "bell", "gong"],
}

# A mix preset is a starting point, not a law: re-measure and adjust.
# `drums` runs on its own bus (glue + parallel) and is NOT multiband
# compressed; `music` is. `width_target` is solved for by scaling the side
# signal; `target_rms_db` is the loudness the limiter aims at.
MIX_PRESETS = {
    "chamber": {
        "reverb_wet": 0.30, "drum_wet": 0.22, "reverb_sec": 2.4, "lowcut_hz": 160,
        "compress": [(-34, 3.0)], "peak": 0.66, "width_target": 0.35,
        "target_rms_db": -17.0, "knee": 0.6,
        "targets": {"centroid_hz": (450, 900), "width": (0.30, 0.55), "crest_db": (12, 16)},
        "crossovers": (200.0, 2000.0), "mb_thresh_db": (-32.0, -30.0, -33.0),
        "mb_ratio": 1.6, "mb_attack_ms": 15.0, "mb_release_ms": 180.0,
        "ceiling_db": -1.0, "predelay_ms": 18.0,
        "drum_thresh_db": -17.0, "drum_ratio": 2.2, "drum_attack_ms": 30.0,
        "drum_release_ms": 180.0, "drum_parallel": 0.20,
    },
    "pop": {
        "reverb_wet": 0.20, "drum_wet": 0.10, "reverb_sec": 2.0, "lowcut_hz": 200,
        "compress": [(-32, 2.6)], "peak": 0.90, "width_target": 0.42,
        "target_rms_db": -13.0, "knee": 0.5,
        "targets": {"centroid_hz": (500, 800), "width": (0.35, 0.55), "crest_db": (10, 14)},
        "crossovers": (200.0, 2000.0), "mb_thresh_db": (-30.0, -28.0, -32.0),
        "mb_ratio": 2.0, "mb_attack_ms": 12.0, "mb_release_ms": 150.0,
        "ceiling_db": -1.0, "predelay_ms": 22.0,
        "drum_thresh_db": -16.0, "drum_ratio": 2.4, "drum_attack_ms": 28.0,
        "drum_release_ms": 170.0, "drum_parallel": 0.25,
    },
    "lofi": {
        "reverb_wet": 0.25, "drum_wet": 0.14, "reverb_sec": 1.8, "lowcut_hz": 180,
        "compress": [(-30, 2.4)], "peak": 0.72, "width_target": 0.25,
        "target_rms_db": -14.5, "knee": 0.4,
        "targets": {"centroid_hz": (400, 900), "width": (0.15, 0.35), "crest_db": (12, 18)},
        "crossovers": (250.0, 2500.0), "mb_thresh_db": (-34.0, -32.0, -34.0),
        "mb_ratio": 1.4, "mb_attack_ms": 20.0, "mb_release_ms": 200.0,
        "ceiling_db": -1.5, "predelay_ms": 10.0,
        "drum_thresh_db": -19.0, "drum_ratio": 2.0, "drum_attack_ms": 35.0,
        "drum_release_ms": 220.0, "drum_parallel": 0.30,
    },
    "cinematic": {
        "reverb_wet": 0.42, "drum_wet": 0.28, "reverb_sec": 3.2, "lowcut_hz": 140,
        "compress": [(-38, 2.2)], "peak": 0.70, "width_target": 0.45,
        "target_rms_db": -17.5, "knee": 0.7,
        "targets": {"centroid_hz": (400, 1000), "width": (0.35, 0.60), "crest_db": (14, 20)},
        "crossovers": (180.0, 1800.0), "mb_thresh_db": (-34.0, -32.0, -36.0),
        "mb_ratio": 1.5, "mb_attack_ms": 25.0, "mb_release_ms": 260.0,
        "ceiling_db": -2.0, "predelay_ms": 30.0,
        "drum_thresh_db": -14.0, "drum_ratio": 3.0, "drum_attack_ms": 35.0,
        "drum_release_ms": 240.0, "drum_parallel": 0.30,
    },
}


def palette(preset="pop"):
    """Return a printable palette brief: roles, choices and mix starting points."""
    p = MIX_PRESETS.get(preset, MIX_PRESETS["pop"])
    lines = [f"preset: {preset}  (targets: {p['targets']})", "roles:"]
    for role, names in INSTRUMENTS.items():
        lines.append(f"  {role:8s} {', '.join(names)}")
    lines.append(f"mix: drums glue {p['drum_ratio']}:1 @ {p['drum_thresh_db']:.0f} dB "
                 f"(attack {p['drum_attack_ms']:.0f} ms, parallel {p['drum_parallel']:.2f}), "
                 f"music MB {p['mb_ratio']}:1, width target {p.get('width_target', 0.4):.2f}")
    return "\n".join(lines)


# ----------------------------------------------------------------- soundfont
_SF = None
_SF_CACHE = {}


def use_soundfont(path):
    """Load an .sf2 bank for sampled instruments (see sf2.py). Returns it."""
    global _SF
    from sf2 import SoundFont
    _SF = SoundFont(path)
    _SF_CACHE.clear()
    return _SF


def soundfont():
    return _SF


def sf_play(program, bank=0):
    """Make an instrument function (freq, dur, vel) from the loaded bank."""
    def play(freq, dur, vel=0.8):
        if _SF is None:
            raise RuntimeError("call synth_lib.use_soundfont(path) first")
        key = 69 + 12 * np.log2(max(freq, 1e-6) / 440.0)
        ck = (program, bank, round(key, 2), round(dur, 2), round(vel, 2))
        if ck not in _SF_CACHE:
            _SF_CACHE[ck] = _SF.note(program, key, vel, dur, bank=bank)
        return _SF_CACHE[ck] * (vel if False else 1.0)
    return play
