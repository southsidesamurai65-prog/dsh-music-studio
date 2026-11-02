#!/usr/bin/env python3
"""
mix.py — the mix bus: a drum bus, multiband compression, real-IR convolution,
stereo width and loudness targets, a look-ahead limiter and a knee clipper.
numpy only.

The synth library could render notes; it could not *mix*. The one decision that
matters most is what the multiband compressor is allowed to see: if it sees the
kit, it flattens the snare transient along with the pad. So the input is stems.

    import synth_lib as sl, mix
    st = sl.Stems(seconds)                 # drums / bass / music / fx
    ...
    ir = mix.load_ir("ConradPrebysConcertHall.wav")
    bus = mix.mixdown(st, preset="pop", ir=ir)

Single-bus material still works the old way:

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


def send_return(send, ir, predelay=0.0, highpass=120.0, sr=SR):
    """Convolve a send bus with an IR; return only the wet signal.

    `ir` may be mono (1, N) — `synth_lib.make_ir` returns that — or stereo
    (2, N) as measured impulse responses are.
    """
    n = send.shape[1]
    ir = np.asarray(ir, dtype=float)
    if ir.ndim == 1:
        ir = np.stack([ir, ir])
    ir = ir.copy()
    if predelay > 0:
        d = int(predelay * sr)
        ir = np.concatenate([np.zeros((2, d)), ir], axis=1)
    pad = max(ir.shape[1], 1)
    src = np.concatenate([send, np.zeros((2, pad))], axis=1)
    nf = 1 << int(np.ceil(np.log2(src.shape[1] + ir.shape[1])))
    out = np.zeros((2, src.shape[1]))
    for c in range(2):
        X = np.fft.rfft(src[c], nf)
        H = np.fft.rfft(ir[c], nf)
        out[c] = np.fft.irfft(X * H, nf)[: src.shape[1]]
    if highpass:
        out = np.stack([_highpass(out[c], highpass) for c in range(2)])
    return out[:, :n]


def reverb(mix, ir, wet=0.3, predelay=0.0, highpass=120.0):
    """Convolve the send bus with a real IR and add it to the dry mix.

    `mix` is (2, N); the returned array is the same length, with the reverb
    tail of the dry material included (the tail beyond the end is folded in by
    padding before the convolution).
    """
    return mix + wet * send_return(mix, ir, predelay, highpass)


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
    slow = np.convolve(coarse, win_r, "same")[:m]
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
def limiter(mix, ceiling_db=-1.0, lookahead_ms=5.0, release_ms=90.0, makeup=True):
    """Look-ahead peak limiter.

    The gain is solved on a 1.45 ms block grid: a *forward* minimum over the
    look-ahead window (so the reduction is already in place when the peak
    arrives), a release that only lets the gain back up slowly, and a short
    centred smoothing pass the width of the look-ahead.

    With `makeup`, the remaining headroom is handed back so the loudest peak
    lands on the ceiling. The release always lags a dense mix, and without this
    the output sits 2-3 dB under the ceiling with a low crest — which reads as
    "loud but flat". Pass makeup=False for a limiter that may only attenuate.

    This matters more than it looks. An earlier version smoothed with a centred
    90 ms window, which began pulling the gain down 45 ms *before* every snare
    hit — flattening exactly the attacks the drum bus had just been split out
    to protect (backbeat 2.01 -> 1.63 on a test mix).
    """
    ceiling = 10 ** (ceiling_db / 20)
    n = mix.shape[1]
    w = 64
    nb = max(1, n // w)
    peak = np.max(np.abs(mix[:, : nb * w]), axis=0).reshape(nb, w).max(axis=1)
    req = np.minimum(1.0, ceiling / (peak + 1e-12))
    L = max(1, int(round(lookahead_ms * 1e-3 * SR / w)))
    anti = req.copy()
    for k in range(1, L + 1):
        anti[:-k] = np.minimum(anti[:-k], req[k:])
    r = float(np.exp(-1.0 / max(1.0, release_ms * 1e-3 * SR / w)))
    g = np.empty(nb)
    cur = 1.0
    for i in range(nb):
        a = anti[i]
        cur = a if a < cur else (r * cur + (1.0 - r) * a)
        g[i] = cur
    k_win = np.hanning(2 * L + 1)
    gs = np.convolve(g, k_win / k_win.sum(), "same")
    # np.convolve(..., "same") returns max(len(g), len(kernel)) elements, which
    # grows a short bus; trim back to the block grid
    g = gs[:nb] if len(gs) >= nb else np.concatenate(
        [gs, np.full(nb - len(gs), gs[-1] if len(gs) else 1.0)])
    gain = np.interp(np.arange(n), np.arange(nb) * w + 0.5 * w, np.clip(g, 0.0, 1.0))
    out = np.clip(mix * gain, -ceiling, ceiling)
    # the release lags the requirement, so a loud mix ends up a couple of dB
    # below the ceiling: hand that headroom back, which is what keeps the crest
    # (and therefore the punch) up
    if makeup and np.any(gain < 0.999):
        p = np.max(np.abs(out))
        if 1e-9 < p < ceiling:
            out = out * (ceiling / p)
    return np.clip(out, -ceiling, ceiling)


def soft_clip(mix, ceiling_db=-1.0, knee=0.5, drive_db=None):
    """Knee clipper: linear below `knee` of the ceiling, a tangent tanh above,
    hard at the ceiling.

    A tanh across the whole signal is not a clipper — it quietly attenuates
    everything and compresses the whole mix (measured: 2.4 dB of peak loss).
    This one leaves anything under the knee alone, so it only shaves the last
    few dB that the limiter could not take without pumping.
    """
    c = 10 ** (ceiling_db / 20)
    if drive_db is not None:          # 1.4.0 and earlier took a drive amount
        knee = 0.5 * 10 ** (-drive_db / 20)
    k = float(np.clip(knee, 0.05, 0.95)) * c
    span = c - k
    a = np.abs(mix)
    y = np.where(a <= k, a, k + span * np.tanh((a - k) / span))
    return np.sign(mix) * np.minimum(y, c)


# ---------------------------------------------------------------- drum bus
def _lowpass(x, fc, order=2):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / max(fc, 1.0)) ** (2 * order))
    return np.fft.irfft(X, len(x))


def _level_to(x, target_db):
    """Scale a bus so its peak sits at target_db; return (scaled, gain)."""
    p = float(np.max(np.abs(x))) if x.size else 0.0
    if p < 1e-9:
        return x, 1.0
    g = 10 ** (target_db / 20) / p
    return x * g, g


def _peak_env(x, win_ms=2.0):
    """Block-peak level follower (stereo linked, in samples).

    A 5 ms *mean* of a kick reads ~11 dB below its sample peak, which makes any
    threshold above it useless; a block peak does not have that problem.
    """
    p = np.max(np.abs(x), axis=0)
    w = max(1, int(win_ms * 1e-3 * SR))
    blocks = max(1, len(p) // w)
    env = p[: blocks * w].reshape(blocks, w).max(axis=1)
    env = np.repeat(env, w)
    if len(env) < len(p):
        env = np.concatenate([env, np.full(len(p) - len(env), env[-1] if len(env) else 0.0)])
    return env[: len(p)]


def _follow_gain(target, attack_ms, release_ms, decim=64):
    """Gain follower with separate attack and release time constants.

    Block-min decimation keeps the worst case of each block; the loop runs on
    the coarse grid (1.45 ms at 44.1 kHz), so a 3-minute bus takes ~0.2 s.
    """
    t = np.minimum(target, 1.0)
    decim = max(1, min(decim, len(t) // 8 or 1))
    m = max(1, len(t) // decim)
    coarse = t[: m * decim].reshape(m, decim).min(axis=1)
    a = float(np.exp(-1.0 / max(1.0, attack_ms * 1e-3 * SR / decim)))
    r = float(np.exp(-1.0 / max(1.0, release_ms * 1e-3 * SR / decim)))
    out = np.empty(m)
    g = 1.0
    for i in range(m):
        x = coarse[i]
        g = (a * g + (1.0 - a) * x) if x < g else (r * g + (1.0 - r) * x)
        out[i] = g
    gain = np.repeat(out, decim)
    if len(gain) < len(t):
        gain = np.concatenate([gain, np.full(len(t) - len(gain), gain[-1])])
    return gain[: len(t)]


def bus_compress(x, thresh_db=-16.0, ratio=2.0, attack_ms=25.0, release_ms=170.0,
                 knee_db=4.0, detector_ms=2.0):
    """Stereo-linked, peak-detected compressor.

    One gain curve for both channels, so the stereo image does not wander when
    one side is momentarily louder. `detector_ms` is the block-peak window: the
    shorter it is, the more transient detail the detector sees.
    """
    env = _peak_env(x, detector_ms)
    level = 20 * np.log10(env + 1e-9)
    over = level - thresh_db
    gain_db = np.where(over > knee_db / 2, -over * (1 - 1 / ratio),
                       np.where(over < -knee_db / 2, 0.0,
                                -((over + knee_db / 2) ** 2) / (2 * knee_db) * (1 - 1 / ratio)))
    return x * _follow_gain(10 ** (gain_db / 20), attack_ms, release_ms)


def drum_bus(drums, thresh_db=-16.0, ratio=2.4, attack_ms=28.0, release_ms=170.0,
             parallel=0.25, parallel_thresh_db=-26.0, parallel_ratio=8.0,
             parallel_attack_ms=1.5, parallel_release_ms=90.0,
             reference_db=-3.0):
    """Glue-compress the kit on its own bus.

    A slow attack lets the stick transient through, so the kick and snare keep
    their punch; a fast, hard parallel copy adds body and density without the
    pumping that a fast attack on the main path would cause. The output is
    peak-normalised back to the input, which is the make-up gain.

    `reference_db` normalises the kit before compressing so the thresholds mean
    the same thing on a quiet demo and a loud one.
    """
    if not np.any(drums):
        return drums
    x, g = _level_to(drums, reference_db)
    glue = bus_compress(x, thresh_db, ratio, attack_ms, release_ms)
    out = glue
    if parallel > 0:
        crush = bus_compress(x, parallel_thresh_db, parallel_ratio,
                             parallel_attack_ms, parallel_release_ms)
        out = (1.0 - parallel) * glue + parallel * crush
    out = out * (np.max(np.abs(x)) / max(np.max(np.abs(out)), 1e-9))
    return out / g


# ------------------------------------------------------------------ mixdown
def _rms_db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)


def _width(x):
    """Stereo width as the probe measures it: rms(side) / rms(mid)."""
    mid = (x[0] + x[1]) / 2
    side = (x[0] - x[1]) / 2
    return float(np.sqrt((side ** 2).mean()) / (np.sqrt((mid ** 2).mean()) + 1e-9))


def mixdown(stems, preset="pop", ir=None, eq=None, section_env=None,
            target_rms_db=None, width=None, drum_comp=True, music_comp=True,
            drum_gain_db=0.0, music_gain_db=0.0, ceiling_db=None,
            knee=None, drum_wet=None, reverb_wet=None,
            highpass_hz=30.0, lowpass_hz=18000.0, sr=SR):
    """Stem-wise mixdown: drums get their own bus, everything else is the music.

    Chain: drums -> glue+parallel -> (send) ; music -> eq -> multiband ->
    (send) ; send bus -> real IR -> return ; sum -> width -> section dynamics
    -> gain to target loudness -> limiter -> soft clip.

    The point of the split is that the multiband compressor never sees the kit,
    so the snare transient survives the loudness stage. `stems` is a
    `synth_lib.Stems` or a plain {role: (2, N)} dict.
    """
    from synth_lib import MIX_PRESETS
    p = MIX_PRESETS.get(preset, MIX_PRESETS["pop"])
    buf = stems.buf if hasattr(stems, "buf") else dict(stems)
    if not buf:
        raise ValueError("mixdown needs at least one stem")
    n = next(iter(buf.values())).shape[1]
    drums = buf.get("drums", np.zeros((2, n)))
    send_only = buf.get("fx", np.zeros((2, n)))
    music = np.zeros((2, n))
    for role, v in buf.items():
        if role not in ("drums", "fx"):
            music += v
    drums = drums * 10 ** (drum_gain_db / 20)
    music = music * 10 ** (music_gain_db / 20)

    if eq is not None:
        music = eq(music)
    if drum_comp and np.any(drums):
        drums = drum_bus(drums, p.get("drum_thresh_db", -16.0),
                         p.get("drum_ratio", 2.4),
                         p.get("drum_attack_ms", 28.0),
                         p.get("drum_release_ms", 170.0),
                         parallel=p.get("drum_parallel", 0.25))
    if music_comp and np.any(music):
        scaled, g = _level_to(music, -6.0)
        music = multiband_compress(scaled, p.get("crossovers", (200.0, 2000.0)),
                                   p.get("mb_thresh_db", (-30.0, -28.0, -32.0)),
                                   ratio=p.get("mb_ratio", 2.0),
                                   attack_ms=p.get("mb_attack_ms", 12.0),
                                   release_ms=p.get("mb_release_ms", 150.0)) / g

    wet = np.zeros((2, n))
    if ir is not None:
        d_wet = p.get("drum_wet", 0.1) if drum_wet is None else drum_wet
        m_wet = p.get("reverb_wet", 0.2) if reverb_wet is None else reverb_wet
        send = d_wet * drums + m_wet * music + send_only
        wet = send_return(send, ir, p.get("predelay_ms", 20.0) / 1000.0,
                          highpass=p.get("reverb_lowcut_hz", 140.0))
    bus = drums + music + wet

    # Width: a fixed side multiplier cannot fit both a narrow synth demo and a
    # wide sampled one. With width=None the preset's *target* is solved for
    # instead; pass a number to force a side gain.
    if width is None:
        target_w = p.get("width_target", 0.40)
        cur_w = _width(bus)
        # 10x is the cap: beyond that the material is mono by design and
        # widening it only amplifies noise
        side_gain = float(np.clip(target_w / cur_w, 0.0, 10.0)) if cur_w > 1e-6 else 1.0
    else:
        side_gain = float(width)
    if abs(side_gain - 1.0) > 1e-3:
        mid = 0.5 * (bus[0] + bus[1])
        side = 0.5 * (bus[0] - bus[1]) * side_gain
        bus = np.stack([mid + side, mid - side])

    for c in range(2):
        bus[c] = _highpass(_lowpass(bus[c], lowpass_hz), highpass_hz)
    if section_env is not None:
        bus = bus * section_env

    if target_rms_db is None:
        target_rms_db = p.get("target_rms_db", -14.0)
    if knee is None:
        knee = p.get("knee", 0.5)
    ceiling = 10 ** ((p.get("ceiling_db", -1.0) if ceiling_db is None else ceiling_db) / 20)
    ceiling_db_ = 20 * np.log10(ceiling)
    bus = bus * 10 ** ((target_rms_db - _rms_db(bus)) / 20)
    bus = limiter(bus, ceiling_db=ceiling_db_)
    # Pushing harder only lowers the crest, which raises RMS; settling the
    # loudness this way converges instead of ratcheting. Turning it *down* never
    # needs another limiter pass, which is what keeps the peaks intact.
    for _ in range(4):
        diff = target_rms_db - _rms_db(bus)
        if abs(diff) < 0.2:
            break
        if diff < 0:
            bus = bus * 10 ** (diff / 20)
        else:
            bus = bus * 10 ** (min(diff, 1.5) / 20)
            bus = limiter(bus, ceiling_db=ceiling_db_)
        bus = np.clip(bus, -ceiling, ceiling)
    if knee < 0.95:
        bus = soft_clip(bus, ceiling_db=ceiling_db_, knee=knee)
        diff = target_rms_db - _rms_db(bus)
        if diff < -0.1:
            bus = bus * 10 ** (diff / 20)
        bus = np.clip(bus, -ceiling, ceiling)
    return bus


def write_stems(prefix, stems, fade_out=0.0):
    """Dump each stem as its own wav: the fastest way to hear what the bus is
    doing is to listen to the kit alone."""
    from synth_lib import write_wav
    buf = stems.buf if hasattr(stems, "buf") else dict(stems)
    paths = []
    for role, v in buf.items():
        if not np.any(v):
            continue
        path = f"{prefix}_{role}.wav"
        write_wav(path, v, fade_out=fade_out)
        paths.append(path)
    return paths


# ------------------------------------------------------------------- preset
def apply_bus(mix, send=None, preset="pop", ir=None):
    """Convenience chain for a single bus: reverb -> multiband -> limiter.

    Kept for material that was never split into stems. Prefer `mixdown`, which
    can keep the kit out of the multiband compressor.
    """
    from synth_lib import MIX_PRESETS
    if hasattr(mix, "buf"):
        return mixdown(mix, preset, ir=ir)
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
