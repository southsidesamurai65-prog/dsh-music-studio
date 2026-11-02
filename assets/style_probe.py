#!/usr/bin/env python3
"""
style_probe.py — turn a reference track into a style profile.

    python3 style_probe.py reference.mp3 [--json out.json | --json -]

Reliable: tempo, beat/bar grid, key, timbre balance, stereo width,
dynamics, arrangement map.
Approximate: per-bar chords (reported with candidates + root confidence).

Everything is numpy; ffmpeg is only used to decode.
"""
import json
import os
import subprocess
import sys

import numpy as np

SR = 44100
PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KRUMHANSL_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

CHORDS = {
    "maj":  [(0, 1.2), (4, 1.0), (7, 0.8)],
    "min":  [(0, 1.2), (3, 1.0), (7, 0.8)],
    "7":    [(0, 1.2), (4, 0.9), (7, 0.7), (10, 0.7)],
    "maj7": [(0, 1.2), (4, 0.9), (7, 0.7), (11, 0.7)],
    "min7": [(0, 1.2), (3, 0.9), (7, 0.7), (10, 0.7)],
    "6":    [(0, 1.2), (4, 0.9), (7, 0.7), (9, 0.7)],
    "dim":  [(0, 1.2), (3, 0.9), (6, 0.8)],
    "sus4": [(0, 1.2), (5, 1.0), (7, 0.8)],
}
SUF = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "6": "6", "dim": "dim", "sus4": "sus4"}
SCALES = {"major": {0, 2, 4, 5, 7, 9, 11}, "minor": {0, 2, 3, 5, 7, 8, 10}}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]


# ------------------------------------------------------------------ audio io
def load_audio(path):
    if not os.path.exists(path):
        raise SystemExit("找不到文件: %s" % path)
    if os.path.isdir(path):
        raise SystemExit("这是一个目录，不是音频文件: %s" % path)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True)
    if p.returncode != 0:
        detail = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise SystemExit("ffmpeg 无法解码 %s: %s" % (path, detail[-1] if detail else "未知错误"))
    x = np.frombuffer(p.stdout, dtype="<f4").astype(np.float64)
    if x.size == 0:
        raise SystemExit("音频为空，无法分析: %s" % path)
    return x.reshape(-1, 2)


def stft(mono, n_fft, hop):
    n = 1 + max(0, (len(mono) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    return np.abs(np.fft.rfft(mono[idx] * np.hanning(n_fft), axis=1))


# ------------------------------------------------------------------ chroma
def chroma_log(mono, n_fft=16384, hop=4096, lo=55, hi=2000, bpo=24, floor=10):
    """Linear-magnitude chroma on a log-frequency grid (log compression would
    flatten the window leakage into neighbouring semitones)."""
    S = stft(mono, n_fft, hop)
    bin_hz = SR / n_fft
    m = int(np.log2(hi / lo) * bpo)
    f = lo * 2 ** (np.arange(m) / bpo)
    pos = f / bin_hz
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, S.shape[1] - 1)
    fr = pos - i0
    ok = (i0 >= 1) & (i0 + 2 < S.shape[1])
    mag = np.zeros((len(S), m))
    mag[:, ok] = S[:, i0[ok]] * (1 - fr[ok]) + S[:, i1[ok]] * fr[ok]
    pc = np.round(69 + 12 * np.log2(f / 440)).astype(int) % 12
    C = np.stack([mag[:, pc == k].sum(axis=1) for k in range(12)], axis=1)
    C = np.maximum(C - np.percentile(C, floor, axis=1, keepdims=True), 0)
    return C / (C.sum(axis=1, keepdims=True) + 1e-9)


def key_of(chroma, bass_votes=None):
    """Krumhansl correlation on the full chroma, blended with the bass-note
    histogram: the bass roots disambiguate relative keys (Am vs C vs Em)."""
    g = chroma.mean(axis=0)
    g = g / (g.sum() + 1e-9)
    b = None
    if bass_votes is not None and bass_votes.sum() > 0:
        bv = bass_votes / bass_votes.sum()
        # trust the bass histogram only if it agrees with the chroma at all
        if np.corrcoef(bv, g)[0, 1] > 0.25:
            b = bv
    out = []
    for root in range(12):
        for mode, prof in (("major", KRUMHANSL_MAJ), ("minor", KRUMHANSL_MIN)):
            p = np.roll(prof, root)
            p = p / p.sum()
            r = float(np.corrcoef(g, p)[0, 1])
            if b is not None:
                r = 0.6 * r + 0.4 * float(np.corrcoef(b, p)[0, 1])
            out.append((r, root, mode))
    out.sort(reverse=True)
    return out


# ------------------------------------------------------------- arrangement
def arrangement_of(S, fps, period, phase, mono):
    """Heuristics for *what kind of texture* a track has.

    The probe can hear level and spectrum; it cannot see the score. These
    numbers are the closest it gets to "is there a drum kit, and how busy is
    the arrangement": attack density, transient share, how deeply the low band
    is modulated on the beat, and whether beats 2/4 are stronger than 1/3.
    Treat them as hints, not facts.
    """
    freqs = np.fft.rfftfreq(4096, 1 / SR)
    d = np.maximum(np.diff(np.log1p(S * 20), axis=0), 0.0)
    f_all = np.concatenate([[0.0], d.sum(axis=1)])
    z = (f_all - f_all.mean()) / (f_all.std() + 1e-9)
    dur = max(len(z) / fps, 1e-6)

    core = z[1:-1]
    mad = np.median(np.abs(core - np.median(core))) + 1e-9
    thr = np.median(core) + 2.0 * mad
    onsets = (core > thr) & (core > z[:-2]) & (core >= z[2:])
    onset_rate = float(onsets.sum() / dur)
    transient_share = float((z > 1.0).mean())

    low = S[:, (freqs >= 30) & (freqs < 160)]
    low_env = low.mean(axis=1) if low.shape[1] else np.zeros(len(S))
    low_mod = float(low_env.std() / (low_env.mean() + 1e-9))
    # half-time and double-time readings are equally valid: take the best
    low_periodic = 0.0
    if len(low_env) > 4 * period:
        a = low_env - low_env.mean()
        ac = np.correlate(a, a, "full")[len(a) - 1:]
        norm = ac[0] + 1e-9
        for mult in (0.5, 1.0, 2.0):
            lag = int(round(period * mult))
            if 0 < lag < len(low_env) // 2:
                low_periodic = max(low_periodic, float(ac[lag] / norm))

    # the beat-phase estimate can lock onto the backbeat itself, which would
    # report a real backbeat as 0.5: take the strongest of the four rotations
    backbeat = None
    beats = []
    k = 0
    while True:
        c = int(round(phase + k * period))
        if c + 1 >= len(f_all):
            break
        beats.append(f_all[c])
        k += 1
    if len(beats) >= 16:
        best = 0.0
        for shift in range(4):
            strong = [beats[i] for i in range(len(beats)) if (i + shift) % 4 in (1, 3)]
            weak = [beats[i] for i in range(len(beats)) if (i + shift) % 4 in (0, 2)]
            if len(strong) >= 4 and len(weak) >= 4:
                best = max(best, (np.mean(strong) + 1e-9) / (np.mean(weak) + 1e-9))
        backbeat = float(best) if best > 0 else None

    n = min(len(mono), SR * 60)
    spec = np.abs(np.fft.rfft(mono[:n] * np.hanning(n)))
    f = np.fft.rfftfreq(len(spec) * 2 - 2, 1 / SR)
    def band_db(lo, hi):
        m = (f >= lo) & (f < hi)
        return 20 * np.log10(np.sqrt((spec[m] ** 2).mean()) + 1e-9) if m.any() else -99.0
    presence = band_db(1000, 2000) - band_db(200, 500)

    pulse = low_periodic * min(low_mod, 1.5)
    if pulse > 0.55:
        drums = "strong"
    elif pulse > 0.25:
        drums = "moderate"
    else:
        drums = "weak"
    if onset_rate >= 7:
        texture = "busy"
    elif onset_rate >= 3.5:
        texture = "mid"
    else:
        texture = "sparse"

    return {"onset_rate_per_s": round(onset_rate, 2),
            "transient_share": round(transient_share, 3),
            "low_pulse_mod": round(low_mod, 2),
            "low_pulse_periodicity": round(low_periodic, 2),
            "backbeat_ratio": None if backbeat is None else round(backbeat, 2),
            "presence_db": round(float(presence), 1),
            "drums": drums,
            "texture": texture}


# ------------------------------------------------------------------ rhythm
def flux(S):
    d = np.diff(np.log1p(S * 20), axis=0)
    f = np.concatenate([[0], np.maximum(d, 0).sum(axis=1)])
    f = f - f.mean()
    return f / (f.std() + 1e-9)


def comb(onset, period, phase):
    pos = phase + np.arange(max(1, int((len(onset) - phase) / period))) * period
    pos = pos[(pos >= 0) & (pos < len(onset))].astype(int)
    if len(pos) < 4:
        return -1e9
    return float(onset[pos].mean())


def tempo(onset, fps, lo=60, hi=185):
    """Beat period + phase via a comb filter, with a log-normal prior around a
    typical pulse so half-time patterns are not reported as very slow tempos."""
    def prior(per):
        bpm = 60 * fps / per
        return float(np.exp(-0.5 * (np.log2(bpm / 112.0) / 0.45) ** 2))

    def score(per, ph):
        return comb(onset, per, ph) * prior(per)

    n = len(onset)
    ac = np.fft.irfft(np.abs(np.fft.rfft(onset, 2 * n)) ** 2)[:n]
    ac /= ac[0] + 1e-9
    lags = np.arange(int(fps * 60 / hi), int(fps * 60 / lo) + 1)
    best = (-1e9, 0.0, 0.0)
    for lag in sorted(lags, key=lambda L: ac[L], reverse=True)[:6]:
        for mult in (1.0, 2.0, 0.5):
            per = lag * mult
            if not (fps * 60 / hi <= per <= fps * 60 / lo):
                continue
            for ph in np.linspace(0, per, 24, endpoint=False):
                sc = score(per, ph)
                if sc > best[0]:
                    best = (sc, per, ph)
    per0, ph0 = best[1], best[2]
    for per in np.linspace(per0 * 0.95, per0 * 1.05, 161):
        for ph in np.linspace(ph0 - 2, ph0 + 2, 9):
            sc = score(per, ph)
            if sc > best[0]:
                best = (sc, per, ph)
    # surface the runners-up: slow/fast ambiguities are common in sparse or
    # rubato material, and the octave relationship matters to a composer.
    ranked = sorted(lags, key=lambda L: ac[L] * prior(L), reverse=True)
    candidates, seen = [], set()
    for lag in ranked:
        bpm_c = round(60 * fps / lag, 1)
        if any(abs(bpm_c - seen_bpm) / seen_bpm < 0.04 for seen_bpm in seen):
            continue
        seen.add(bpm_c)
        candidates.append({"bpm": bpm_c, "score": round(float(ac[lag] * prior(lag)), 3)})
        if len(candidates) == 3:
            break
    return 60 * fps / best[1], best[1], best[2], candidates


def swing(onset, period, phase):
    offs = []
    for b in range(int((len(onset) - phase) / period)):
        c = phase + b * period
        lo, hi = int(c + 0.3 * period), int(min(c + 0.75 * period, len(onset) - 1))
        if hi <= lo + 1:
            continue
        seg = onset[lo:hi]
        if seg.max() <= 0:
            continue
        k = int(np.argmax(seg)) + lo
        if 0 < k < len(onset) - 1:
            a, m_, z = onset[k - 1], onset[k], onset[k + 1]
            den = a - 2 * m_ + z
            if abs(den) > 1e-9:
                k = k + 0.5 * (a - z) / den
        offs.append((k - c) / period)
    return float(np.median(offs)) if len(offs) >= 8 else None


# ------------------------------------------------------------------ harmony
def rank_chords(v):
    v = v / (np.linalg.norm(v) + 1e-9)
    out = []
    for root in range(12):
        for qual, spec in CHORDS.items():
            t = np.zeros(12)
            for pc, w in spec:
                t[(root + pc) % 12] = w
            out.append((float(v @ (t / np.linalg.norm(t))), root, qual))
    out.sort(reverse=True)
    return out


def lowest_pitch_classes(mono, lo=45, hi=1300, n_fft=32768, hop=4096, rel=0.12):
    """Per frame: pc of the lowest significant spectral peak — the bass when
    there is one, otherwise the bottom of the voicing."""
    S = stft(mono, n_fft, hop)
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    band = (f >= lo) & (f <= hi)
    Sb, fb = S[:, band], f[band]
    out = []
    for row in Sb:
        mx = row.max()
        if mx <= 1e-9:
            out.append(-1)
            continue
        idx = np.where(row > rel * mx)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
        g = groups[0]
        w = fb[g] * row[g]
        fpk = float((fb[g] * row[g]).sum() / (row[g].sum() + 1e-12))
        out.append(int(round(69 + 12 * np.log2(fpk / 440))) % 12)
    return np.array(out), n_fft / SR


def roman(root, qual, key_root, mode):
    deg = (root - key_root) % 12
    scale = SCALES[mode]
    if deg in scale:
        num = ROMAN[sorted(scale).index(deg)]
    else:
        below = max((d for d in scale if d < deg), default=max(scale))
        num = ROMAN[sorted(scale).index(below)] + "#"
    return num.lower() if qual in ("min", "min7", "dim") else num


# ------------------------------------------------------------------ analysis
def analyse(path):
    x = load_audio(path)
    mono = x.mean(axis=1)
    dur = len(mono) / SR
    S = stft(mono, 4096, 1024)
    onset = flux(S)
    fps = SR / 1024
    bpm, period, phase, tempo_alts = tempo(onset, fps)
    beat_s = period / fps
    bar_s = 4 * beat_s
    sw = swing(onset, period, phase)

    C = chroma_log(mono)
    c_fps = SR / 4096
    lp, lp_hop = lowest_pitch_classes(mono)
    lp_fps = SR / lp_hop
    bass_votes = np.bincount(lp[lp >= 0], minlength=12).astype(float)
    keys = key_of(C, bass_votes)
    key_r, key_root, mode = keys[0]
    key_alts = [{"name": f"{PITCH_NAMES[r]} {mo}", "score": round(r_, 3)} for r_, r, mo in keys[1:3]]

    # align the bar grid: the true bar phase makes every bar internally
    # homogeneous (chords change on the downbeat), so maximise mean cosine
    # similarity of frames to their own bar's mean chroma.
    fpb = max(4, int(round(bar_s * c_fps)))
    n_bars = max(1, len(C) // fpb)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    best = (-1.0, 0)
    for ph in range(fpb):
        sc = []
        for k in range(n_bars):
            a, b = ph + k * fpb, min(ph + (k + 1) * fpb, len(Cn))
            if b - a < 0.8 * fpb:
                continue
            m = Cn[a:b].mean(axis=0)
            m /= np.linalg.norm(m) + 1e-9
            sc.append(float(np.mean(Cn[a:b] @ m)))
        if sc and float(np.mean(sc)) > best[0]:
            best = (float(np.mean(sc)), ph)
    frame0 = best[1]

    # per-bar chroma + root votes
    bar_vecs, bar_votes = [], []
    for b in range(n_bars):
        f0 = frame0 + b * fpb
        f1 = f0 + fpb
        if f0 < 0 or f1 > len(C):
            continue
        seg = np.median(C[f0:f1], axis=0)
        if seg.sum() < 1e-9:
            continue
        bar_vecs.append(seg / (np.linalg.norm(seg) + 1e-9))
        a = int((f0 / c_fps) * lp_fps)
        e = int((f1 / c_fps) * lp_fps)
        v = lp[a:e]
        hist = np.bincount(v[v >= 0], minlength=12).astype(float)
        bar_votes.append(hist)

    def pick_bar(vec, hist):
        cand = rank_chords(vec)
        pick = cand[0]
        conf = 0.0
        if hist.sum() > 0:
            root = int(np.argmax(hist))
            conf = float(hist.max() / hist.sum())
            if conf >= 0.5:
                for sc, r, q in cand[:8]:
                    if r == root:
                        pick = (sc, r, q)
                        break
        alts = [f"{PITCH_NAMES[r]}{SUF[q]}" for _, r, q in cand[:3]]
        return pick, conf, alts

    # loop detection: compare bars p apart by chroma similarity
    loop, loop_sim = None, 0.0
    n = len(bar_vecs)
    for p in range(1, min(9, max(2, n // 3) + 1)):
        sims = [float(bar_vecs[i] @ bar_vecs[i + p]) for i in range(n - p)]
        if len(sims) >= 3:
            m = float(np.mean(sims))
            if m > 0.90 and m > loop_sim:
                loop, loop_sim = p, m

    bars, chord_names, chord_alts, roots, chord_quals, confs = [], [], [], [], [], []
    if loop:
        for s_i in range(loop):
            idxs = list(range(s_i, n, loop))
            vec = np.mean([bar_vecs[i] for i in idxs], axis=0)
            hist = np.sum([bar_votes[i] for i in idxs], axis=0)
            pick, conf, alts = pick_bar(vec, hist)
            chord_names.append(f"{PITCH_NAMES[pick[1]]}{SUF[pick[2]]}")
            chord_alts.append(alts)
            roots.append(pick[1])
            chord_quals.append(pick[2])
            confs.append(round(conf, 2))
        loop_names = chord_names[:]
        loop_quals = list(zip(roots, chord_quals))
        per_bar = [chord_names[i % loop] for i in range(n)]
        bars = [[round((frame0 + i * fpb) / c_fps, 2), round((frame0 + (i + 1) * fpb) / c_fps, 2)]
                for i in range(n)]
    else:
        for i in range(n):
            pick, conf, alts = pick_bar(bar_vecs[i], bar_votes[i])
            chord_names.append(f"{PITCH_NAMES[pick[1]]}{SUF[pick[2]]}")
            chord_alts.append(alts)
            roots.append(pick[1])
            chord_quals.append(pick[2])
            confs.append(round(conf, 2))
            bars.append([round((frame0 + i * fpb) / c_fps, 2), round((frame0 + (i + 1) * fpb) / c_fps, 2)])
        loop_names, loop_quals, per_bar = [], [], chord_names

    # ---- timbre / space / dynamics
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    fm = np.fft.rfftfreq(len(mono), 1 / SR)
    tot = np.sum(spec ** 2) + 1e-12
    centroid = float(np.sum(fm * spec ** 2) / tot)
    bands = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000),
             (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]
    bal = {f"{lo}-{hi}": round(float(10 * np.log10((np.sum(spec[(fm >= lo) & (fm < hi)] ** 2) + 1e-12) / tot)), 1)
           for lo, hi in bands}

    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    width = float(np.sqrt((side ** 2).mean()) / (np.sqrt((mid ** 2).mean()) + 1e-9))
    rms = float(np.sqrt((x ** 2).mean()))
    crest = float(20 * np.log10(np.max(np.abs(x)) / (rms + 1e-9)))

    env = np.sqrt(np.convolve(mono ** 2, np.ones(512) / 512, "same"))
    pk = np.where((env[1:-1] > np.percentile(env, 99.5)) &
                  (env[1:-1] >= env[:-2]) & (env[1:-1] >= env[2:]))[0] + 1
    ratios = [env[p + int(0.15 * SR)] / (env[p + int(0.02 * SR)] + 1e-9)
              for p in pk[:200] if p + int(0.15 * SR) < len(env)]
    tail = float(np.median(ratios)) if ratios else float("nan")

    nb = 8
    arc = [round(float(20 * np.log10(np.sqrt((mono[int(k * len(mono) / nb):int((k + 1) * len(mono) / nb)] ** 2).mean()) + 1e-9)), 1)
           for k in range(nb)]

    blocks = []
    for i in range(0, len(mono), SR):
        s = mono[i:i + SR]
        if len(s) < SR // 2:
            continue
        sp = np.abs(np.fft.rfft(s * np.hanning(len(s))))
        f = np.fft.rfftfreq(len(s), 1 / SR)
        blocks.append([round(i / SR, 1), round(float(20 * np.log10(np.sqrt((s ** 2).mean()) + 1e-9)), 1),
                       round(float(np.sum(f * sp) / (sp.sum() + 1e-9)))])

    def char(c):
        return "warm/dark" if c < 1200 else ("balanced" if c < 2600 else "bright")

    return {
        "file": path,
        "duration_s": round(dur, 2),
        "tempo": {"bpm": round(bpm, 1), "candidates": tempo_alts, "meter": "4/4 (assumed)",
                  "swing_offbeat_position": None if sw is None else round(sw, 3)},
        "key": {"name": f"{PITCH_NAMES[key_root]} {mode}", "score": round(key_r, 3),
                "alternatives": key_alts},
        "chords": {
            "bar_seconds": round(bar_s, 2),
            "per_bar": per_bar,
            "candidates": chord_alts,
            "root_confidence": confs,
            "note": "chord labels are estimates; candidates list nearby readings",
            "loop_bars": loop,
            "loop": loop_names if loop else None,
            "loop_similarity": round(loop_sim, 3),
            "roman": [roman(r, q, key_root, mode) for r, q in loop_quals] if loop else None,
        },
        "timbre": {"centroid_hz": round(centroid), "character": char(centroid),
                   "band_balance_db": bal},
        "space": {"stereo_width": round(width, 3),
                  "tail_150ms_ratio": None if tail != tail else round(tail, 3)},
        "dynamics": {"rms_dbfs": round(20 * np.log10(rms + 1e-9), 1),
                     "crest_db": round(crest, 1), "arc_8ths_db": arc},
        "arrangement": arrangement_of(S, fps, period, phase, mono),
        "arrangement_1s": blocks,
    }


def brief(p):
    t = p["tempo"]["bpm"]
    ch = p["chords"]
    if ch["loop"]:
        loop_txt = f"{ch['loop_bars']} 小节: {'–'.join(ch['loop'])} ({' '.join(ch['roman'])})"
    else:
        loop_txt = "未检出稳定循环（见每小节候选）"
    alt = p["key"]["alternatives"]
    alt_txt = ""
    if alt and p["key"]["score"] - alt[0]["score"] < 0.06:
        alt_txt = f" (也可能是 {alt[0]['name']})"
    out = [f"{p['duration_s']:.0f}s | {t:.0f} BPM | {p['key']['name']}{alt_txt} | 和弦循环 {loop_txt}",
           f"音色重心 {p['timbre']['centroid_hz']} Hz ({p['timbre']['character']}) | 低频/中频/高频 "
           f"{p['timbre']['band_balance_db']['20-60']:+.1f} / "
           f"{p['timbre']['band_balance_db']['500-1000']:+.1f} / "
           f"{p['timbre']['band_balance_db']['4000-8000']:+.1f} dB",
           f"立体声宽度 {p['space']['stereo_width']:.2f} | 动态 {p['dynamics']['crest_db']:.0f} dB | "
           f"8 段能量 " + " ".join(f"{v:.0f}" for v in p["dynamics"]["arc_8ths_db"])]
    if p["tempo"]["swing_offbeat_position"] is not None:
        out.append(f"摇摆: 后半拍在 {p['tempo']['swing_offbeat_position']:.2f} 拍位 (0.50 平直, >0.55 有拖拽感)")
    out.append("每小节: " + " ".join(f"{c}" for c in p["chords"]["per_bar"]))
    a = p.get("arrangement")
    if a:
        bb = "" if a["backbeat_ratio"] is None else f", 反拍/正拍 {a['backbeat_ratio']:.2f}"
        out.append(f"织体: {a['texture']} | 起音 {a['onset_rate_per_s']:.1f}/秒 | "
                   f"低频调制 {a['low_pulse_mod']:.2f}/周期性 {a['low_pulse_periodicity']:.2f}"
                   f" | 节拍脉冲 {a['drums']}{bb} | 存在感(1-2k) {a['presence_db']:+.1f} dB")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    out = None
    if "--json" in sys.argv:
        i = sys.argv.index("--json")
        out = sys.argv[i + 1] if i + 1 < len(sys.argv) else "-"
    prof = analyse(sys.argv[1])
    if out != "-":
        print(brief(prof))
    if out is not None:
        if out == "-":
            print(json.dumps(prof, ensure_ascii=False))
        else:
            with open(out, "w") as fh:
                json.dump(prof, fh, ensure_ascii=False, indent=2)
            print("wrote", out)
