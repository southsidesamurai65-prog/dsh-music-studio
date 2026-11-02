#!/usr/bin/env python3
"""
mix.py — a small mixing bus: multiband compression, real-IR convolution and a
look-ahead limiter. numpy only.

The synth library could render notes; it could not *mix*. This module is the
missing half of P1: split the bus, compress the bands independently, put it in a
real room measured as an impulse response, then limit.

    import mix
    ir = mix.load_ir("ConradPrebysConcertHall.wav")
    wet = mix.reverb(bus, ir, wet=0.3)
    bus = mix.multiband_compress(wet, [250.0, 2500.0], thresh_db=[-28, -30, -32])
    bus = mix.limiter(bus, ceiling_db=-1.0)
"""
import subprocess

import numpy as np

SR = 44100


# ------------------------------------------------------------------- impulse
def load_ir(path, sr=SR, max_sec=3.0, normalize=True):
    """Decode an IR wav (any bit depth) to a stereo float array via ffmpeg."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(sr),
         "-f", "f32le", "-"], capture_output=True, check=True)
    x = np.frombuffer(p.stdout, dtype="<f4").astype(np.float64).reshape(-1, 2).T
    x = x[:, : int(max_sec * sr)]
    if normalize and x.size:
        x /= np.sqrt((x ** 2).sum() / 2) + 1e-12
    return x


def reverb(mix, ir, wet=0.3, predelay=0.0, highpass=120.0):
    """Convolve the send bus with a real IR and add it to the dry mix.

    `mix` is (2, N); the returned array is the same length, with the reverb
    tail of the dry material included (the tail beyond the end is folded in by
    padding before the convolution).
    """
    n = mix.shape[1]
    ir_len = ir.shape[1]
    pad = max(ir_len, int(predelay * SR))
    ir = ir.copy()
    if predelay > 0:
        d = int(predelay * SR)
        ir = np.concatenate([np.zeros((2, d)), ir], axis=1)
        pad += d
    src = np.concatenate([mix, np.zeros((2, pad))], axis=1)
    nf = 1 << int(np.ceil(np.log2(src.shape[1] + ir.shape[1])))
    out = np.zeros((2, src.shape[1]))
    for c in range(2):
        X = np.fft.rfft(src[c], nf)
        H = np.fft.rfft(ir[c], nf)
        out[c] = np.fft.irfft(X * H, nf)[: src.shape[1]]
    if highpass:
        out = np.stack([_highpass(out[c], highpass) for c in range(2)])
    return mix + wet * out[:, :n]


def _highpass(x, fc, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (fc / np.maximum(f, 1e-6)) ** (2 * order))
    return np.fft.irfft(X, len(x))


# ------------------------------------------------------------------- bands
def band_split(x, crossovers):
    """Split one channel into complementary bands (zero phase, unity sum)."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X = np.fft.rfft(x)
    edges = [0.0] + list(crossovers) + [SR / 2]
    masks = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = np.ones_like(f)
        if lo > 0:
            m *= 1 / (1 + (lo / np.maximum(f, 1.0)) ** 4)
        if hi < SR / 2:
            m *= 1 / (1 + (f / hi) ** 4)
        masks.append(m)
    total = np.sum(masks, axis=0) + 1e-12
    return [np.fft.irfft(X * (m / total), n) for m in masks]


def _movavg(x, k):
    """Running mean in O(n) via a cumulative sum."""
    k = max(1, int(k))
    if k <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    pad = len(x) - len(out)
    return np.concatenate([out, np.full(pad, out[-1] if len(out) else 0.0)])


def _smooth_gain(target, attack_ms, release_ms, decim=64):
    """Vectorised stand-in for a one-pole follower: a short attack mean, a long
    release mean, and never louder than the instantaneous requirement."""
    n = len(target)
    m = max(1, n // decim)
    coarse = target[: m * decim].reshape(m, decim).min(axis=1)
    a = max(1, int(attack_ms * 1e-3 * SR / decim))
    r = max(1, int(release_ms * 1e-3 * SR / decim))
    fast = _movavg(coarse, a)
    win_r = np.hanning(max(r, 3))
    win_r /= win_r.sum()
    slow = np.convolve(coarse, win_r, "same")
    sm = np.repeat(np.minimum(np.minimum(fast, slow), coarse), decim)[:n]
    if len(sm) < n:
        sm = np.concatenate([sm, np.full(n - len(sm), sm[-1] if len(sm) else 1.0)])
    return np.minimum(sm, target + 1e-9)


def _compress_mono(x, thresh_db, ratio, attack_ms, release_ms, knee_db=3.0):
    """Feed-forward compressor with a smoothed gain envelope."""
    env = _movavg(np.abs(x), int(0.005 * SR))
    level = 20 * np.log10(env + 1e-9)
    over = level - thresh_db
    # soft knee
    gain_db = np.where(over > knee_db / 2, -over * (1 - 1 / ratio),
                       np.where(over < -knee_db / 2, 0.0,
                                -((over + knee_db / 2) ** 2) / (2 * knee_db) * (1 - 1 / ratio)))
    target = 10 ** (gain_db / 20)
    return x * _smooth_gain(target, attack_ms, release_ms)


def multiband_compress(mix, crossovers=(250.0, 2500.0), thresh_db=(-28, -30, -32),
                       ratio=2.0, attack_ms=8.0, release_ms=120.0, makeup_db=None):
    """Compress each band independently, then sum. `thresh_db` is per band."""
    out = np.zeros_like(mix)
    for c in range(2):
        bands = band_split(mix[c], crossovers)
        acc = np.zeros_like(mix[c])
        for i, b in enumerate(bands):
            th = thresh_db[min(i, len(thresh_db) - 1)]
            g = _compress_mono(b, th, ratio, attack_ms, release_ms)
            if makeup_db is not None:
                g *= 10 ** (makeup_db[min(i, len(makeup_db) - 1)] / 20)
            acc += g
        out[c] = acc
    return out


# ------------------------------------------------------------------ limiter
def limiter(mix, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=90.0):
    """Look-ahead peak limiter: block-max envelope, smooth gain, hard ceiling."""
    ceiling = 10 ** (ceiling_db / 20)
    w = max(1, int(lookahead_ms * 1e-3 * SR))
    n = mix.shape[1]
    peak = np.max(np.abs(mix), axis=0)
    blocks = max(1, n // w)
    env = peak[: blocks * w].reshape(blocks, w).max(axis=1)
    env = np.repeat(env, w)[:n]
    if len(env) < n:
        env = np.concatenate([env, np.full(n - len(env), env[-1] if len(env) else 0.0)])
    gain = np.minimum(1.0, ceiling / (env + 1e-9))
    smooth = _smooth_gain(gain, lookahead_ms, release_ms)
    out = mix * smooth
    return np.clip(out, -ceiling, ceiling)


def soft_clip(mix, ceiling_db=-1.0, drive_db=3.0):
    """Soft clipper: the last few dB of loudness that a peak limiter cannot give
    without sounding pumped. Standard, and audible if pushed too far."""
    c = 10 ** (ceiling_db / 20)
    y = np.tanh(mix * 10 ** (drive_db / 20) / c) * c
    peak = np.max(np.abs(y)) + 1e-12
    return y * (c / peak) if peak > c else y


# ------------------------------------------------------------------- preset
def apply_bus(mix, send, preset="pop", ir=None):
    """Convenience chain: reverb -> multiband -> limiter, driven by a preset dict."""
    from synth_lib import MIX_PRESETS
    p = MIX_PRESETS.get(preset, MIX_PRESETS["pop"])
    wet = p.get("reverb_wet", 0.25)
    if ir is not None:
        mix = reverb(mix, ir, wet=wet, predelay=p.get("predelay_ms", 15) / 1000.0)
    cross = p.get("crossovers", (250.0, 2500.0))
    th = p.get("mb_thresh_db", (-30.0, -28.0, -30.0))
    mix = multiband_compress(mix, cross, th, ratio=p.get("mb_ratio", 2.0),
                             attack_ms=p.get("mb_attack_ms", 10.0),
                             release_ms=p.get("mb_release_ms", 140.0))
    return limiter(mix, ceiling_db=p.get("ceiling_db", -1.0))
