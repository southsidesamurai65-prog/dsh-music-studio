#!/usr/bin/env python3
"""
ablate.py — paired ablations for the style probe.

The probe is a pile of heuristics. "It reported 2.26 on my reference" proves
nothing on its own; the question is whether each number *moves when its cause
moves*, by how much, and whether the other numbers stay still while it does.

So: render one short piece from stems, then mix it several times changing
exactly one dimension per pair — drums out, bowed lead swapped for a plucked
one, pad muted, top end filtered, width doubled, reverb drenched, level pushed,
the whole bus compressed — and ask the probe about every version.

Each row is a *paired* comparison: same notes, same seed, same loudness unless
loudness is the variable. That is the only way to tell a real sensitivity from a
coincidence.

    python3 ablate.py                  # table + verdicts
    python3 ablate.py --keep /tmp/ab   # keep the wavs
    python3 ablate.py --json ab.json   # machine-readable, for regressions

Exit status is 1 when an expected movement does not happen, so this doubles as
a regression test for probe changes.
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mix as mx                      # noqa: E402
import style_probe as sp              # noqa: E402
import synth_lib as sl                # noqa: E402

SR = sl.SR
BPM = 76.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 8
TAIL = 3.0
N = int((BARS * BAR + TAIL) * SR)

VOICE = {
    "D":     (38, [50, 57, 62, 66]),
    "D/F#":  (30, [42, 50, 54, 57]),
    "Gmaj7": (31, [43, 50, 54, 59]),
    "Asus4": (33, [45, 52, 57, 62]),
    "A/C#":  (37, [49, 57, 61, 64]),
    "Bm7":   (35, [47, 54, 57, 62]),
    "A7":    (33, [45, 52, 55, 61]),
}
PROG = ["D", "D/F#", "Gmaj7", "Asus4", "D", "A/C#", "Bm7", "A7"]

MELODY = [  # (bar, beat, beats, midi)
    (1, 0.0, 0.5, 74), (1, 0.5, 0.5, 76), (1, 1.0, 1.0, 78), (1, 2.0, 1.0, 76),
    (1, 3.0, 0.5, 74), (1, 3.5, 0.5, 73),
    (2, 0.0, 1.5, 71), (2, 1.5, 0.5, 73), (2, 2.0, 1.0, 74), (2, 3.0, 1.0, 76),
    (3, 0.0, 1.0, 78), (3, 1.0, 1.0, 81), (3, 2.0, 1.5, 79), (3, 3.5, 0.5, 78),
    (4, 0.0, 2.0, 76), (4, 2.0, 1.0, 74), (4, 3.0, 1.0, 73),
    (5, 0.0, 1.0, 74), (5, 1.0, 1.0, 76), (5, 2.0, 1.0, 78), (5, 3.0, 1.0, 81),
    (6, 0.0, 1.5, 79), (6, 1.5, 0.5, 78), (6, 2.0, 1.0, 76), (6, 3.0, 1.0, 74),
    (7, 0.0, 1.0, 73), (7, 1.0, 1.0, 71), (7, 2.0, 2.0, 69),
    (8, 0.0, 4.0, 74),
]


def bt(bar, beat=0.0):
    return (bar - 1) * BAR + beat * BEAT


# --------------------------------------------------------------------- render
def render_stems():
    """One piece, split into the parts the ablations will pull apart."""
    sl.seed(7)
    P = sl.midi2f
    st = sl.Stems(N / SR, roles=("drums", "bass", "piano", "pad",
                                 "lead_bow", "lead_pluck"))
    for bar in range(1, BARS + 1):
        chord = PROG[bar - 1]
        bass_m, notes = VOICE[chord]
        t0 = bt(bar)
        for off, idx, v in ((0.0, 0, .55), (0.5, 2, .40), (1.0, 1, .45), (1.5, 3, .40),
                            (2.0, 2, .50), (2.5, 1, .38), (3.0, 3, .46), (3.5, 2, .36)):
            m = notes[idx]
            while m < 50:
                m += 12
            st.add("piano", sl.piano(P(m), 1.2, v), t0 + off * BEAT,
                   0.35 if idx % 2 else -0.25, 0.55)
        for i, m in enumerate(notes[1:4]):
            while m < 55:
                m += 12
            st.add("pad", sl.strings(P(m), BAR * 1.02, .5), t0, -0.80 + 0.65 * i, 0.30)
        st.add("bass", sl.ebass(P(bass_m), 0.6, .9), t0, 0.0, 0.80)
        st.add("bass", sl.ebass(P(bass_m + 7), 0.5, .6), t0 + 2 * BEAT, 0.0, 0.70)
        st.add("drums", sl.kick(.85), t0, 0.0, 0.90)
        st.add("drums", sl.kick(.65), t0 + 2 * BEAT, 0.0, 0.85)
        st.add("drums", sl.snare(.80), t0 + BEAT, 0.05, 0.85)
        st.add("drums", sl.snare(.80), t0 + 3 * BEAT, 0.05, 0.85)
        for k in range(8):
            st.add("drums", sl.hat(k % 4 == 2, .28), t0 + 0.5 * k * BEAT,
                   0.62 if k % 2 else -0.35, 0.70)
    for bar, beat, dur_b, m in MELODY:
        d = dur_b * BEAT * 0.95
        st.add("lead_bow", sl.erhu(P(m), d, 0.85), bt(bar, beat), 0.05, 0.90)
        st.add("lead_pluck", sl.pipa_pluck(P(m), d, 0.85), bt(bar, beat), 0.05, 0.90)

    # A benchmark is only as good as its balance: a layer 17 dB under the lead
    # is not "removed" by an ablation, it was never there. Pin every part to a
    # common RMS so pulling one out is a real change.
    for role, target in (("drums", -21.0), ("bass", -22.0), ("piano", -22.0),
                         ("pad", -22.0), ("lead_bow", -20.0), ("lead_pluck", -20.0)):
        st.trim(role, target - st.rms_db(role))
    return st


# --------------------------------------------------------------------- mixing
IR = sl.make_ir(2.2, 1.1, 180, 6500)


def build(st, drums=True, lead="bow", pad=True, lowpass=None, width=1.0,
          wet=0.25, level_db=-16.0, compress=False):
    parts = [st["piano"], st["bass"]]
    if drums:
        parts.append(st["drums"])
    if pad:
        parts.append(st["pad"])
    parts.append(st["lead_bow" if lead == "bow" else "lead_pluck"])
    bus = np.sum(parts, axis=0)
    if compress:
        bus = mx.multiband_compress(bus, (200.0, 2000.0), (-30.0, -28.0, -32.0),
                                    ratio=2.0, attack_ms=12.0, release_ms=150.0,
                                    makeup_db=(2.5, 2.0, 2.5))
    bus = mx.reverb(bus, IR, wet=wet, predelay=0.022)
    if abs(width - 1.0) > 1e-3:
        mid = 0.5 * (bus[0] + bus[1])
        side = 0.5 * (bus[0] - bus[1]) * width
        bus = np.stack([mid + side, mid - side])
    if lowpass:
        bus = np.stack([mx._lowpass(bus[c], lowpass, order=2) for c in range(2)])
    bus = bus * 10 ** ((level_db - mx._rms_db(bus)) / 20)
    # a loud master is a limited master: use the limiter rather than hard
    # clipping, otherwise the loudness pair only measures my clipper
    bus = mx.limiter(bus, ceiling_db=-1.0)
    return np.clip(bus, -1.0, 1.0)


# ------------------------------------------------------------------ ablation
# `must`  : a movement this benchmark insists on — it is the documented purpose
#           of the metric. A failure here means the probe regressed.
# `watch` : a movement worth knowing about but not guaranteed on this material;
#           reported with its size, never fails the run.
# `stable`: metrics that must not move (specificity). Crossing the threshold is
#           a warning, because it means the number is not isolated.
ABLATIONS = [
    dict(key="no_drums", label="去掉鼓组", variant=dict(drums=False),
         must=[("backbeat", "down", 0.10), ("low_pulse_periodicity", "down", 0.05)],
         watch=[("onset_rate", "down", 0.4), ("crest_db", "up", 0.5),
                ("bpm", "same", 0.5)],
         stable=[("band_mid_db", 2.0)]),
    dict(key="plucked", label="主奏 弓弦→拨弦", variant=dict(lead="pluck"),
         must=[("attack_mid_ms", "down", 3.0)],
         watch=[("sustain_ratio", "down", 0.01), ("flat_high", "up", 0.02)],
         stable=[("bpm", 0.5), ("width", 0.08)]),
    dict(key="no_pad", label="去掉铺底", variant=dict(pad=False),
         must=[("band_lowmid_db", "down", 0.8)],
         watch=[("width", "down", 0.03), ("sustain_ratio", "down", 0.01)],
         stable=[("bpm", 0.5)]),
    dict(key="dark", label="低通 3 kHz", variant=dict(lowpass=3000.0),
         must=[("centroid_hz", "down", 120.0), ("band_high_db", "down", 1.0)],
         watch=[("backbeat", "same", 0.20), ("flat_high", "down", 0.02)],
         stable=[("bpm", 0.5)]),
    dict(key="wide", label="侧面 ×2.2", variant=dict(width=2.2),
         must=[("width", "up", 0.08)],
         watch=[("centroid_hz", "same", 60.0), ("crest_db", "same", 0.5)],
         stable=[("bpm", 0.5), ("backbeat", 0.20)]),
    dict(key="wet", label="混响 0.25→0.60", variant=dict(wet=0.60),
         must=[("tail", "up", 0.03)],
         watch=[("centroid_hz", "same", 80.0), ("attack_mid_ms", "down", 2.0)],
         stable=[("bpm", 0.5)]),
    dict(key="loud", label="响度 −16→−12 dBFS", variant=dict(level_db=-12.0),
         must=[("rms_db", "up", 3.0), ("crest_db", "down", 0.4)],
         watch=[("band_mid_db", "same", 1.0), ("centroid_hz", "same", 80.0)],
         stable=[("bpm", 0.5)]),
    dict(key="compressed", label="整条总线压缩", variant=dict(compress=True),
         must=[("crest_db", "down", 0.4)],
         watch=[("backbeat", "down", 0.10), ("presence_db", "up", 0.5)],
         stable=[("bpm", 0.5)]),
]

# below these a delta is noise, not signal: used to surface cross-talk
NOISE = {"bpm": 0.5, "key": 0.0, "centroid_hz": 40.0, "band_low_db": 0.5,
         "band_lowmid_db": 0.5, "band_mid_db": 0.5, "band_high_db": 0.5, "width": 0.03, "tail": 0.01,
         "crest_db": 0.3, "rms_db": 0.3, "onset_rate": 0.2, "backbeat": 0.08,
         "low_pulse_periodicity": 0.03, "presence_db": 0.5, "attack_mid_ms": 2.0,
         "attack_high_ms": 2.0, "sustain_ratio": 0.01, "flat_mid": 0.01,
         "flat_high": 0.01}

FEATURES = [
    ("bpm", "BPM"), ("key", "调性"),
    ("centroid_hz", "音色重心 Hz"), ("band_low_db", "低频 60–120 dB"),
    ("band_lowmid_db", "中低频 250–500 dB"), ("band_mid_db", "中频 500–1 k dB"),
    ("band_high_db", "高频 4–8 k dB"),
    ("width", "立体声宽度"), ("tail", "混响尾比"),
    ("crest_db", "crest dB"), ("rms_db", "RMS dBFS"),
    ("onset_rate", "起音密度 /s"), ("backbeat", "反拍比"),
    ("low_pulse_periodicity", "低频脉冲周期性"), ("presence_db", "存在感 dB"),
    ("attack_mid_ms", "中频起音 ms"), ("attack_high_ms", "高频起音 ms"),
    ("sustain_ratio", "持续音占比"), ("flat_mid", "中频平坦度"),
    ("flat_high", "高频平坦度"),
]


def features(prof):
    t = prof["timbre"]
    bal = t["band_balance_db"]
    f = prof["fingerprint"]
    return {
        "bpm": prof["tempo"]["bpm"],
        "key": prof["key"]["name"],
        "centroid_hz": t["centroid_hz"],
        "band_low_db": bal["60-120"],
        "band_lowmid_db": bal["250-500"],
        "band_mid_db": bal["500-1000"],
        "band_high_db": bal["4000-8000"],
        "width": prof["space"]["stereo_width"],
        "tail": prof["space"]["tail_150ms_ratio"],
        "crest_db": prof["dynamics"]["crest_db"],
        "rms_db": prof["dynamics"]["rms_dbfs"],
        "onset_rate": prof["arrangement"]["onset_rate_per_s"],
        "backbeat": prof["arrangement"]["backbeat_ratio"],
        "low_pulse_periodicity": prof["arrangement"]["low_pulse_periodicity"],
        "presence_db": prof["arrangement"]["presence_db"],
        "attack_mid_ms": f["attack_ms"]["mid"],
        "attack_high_ms": f["attack_ms"]["high"],
        "sustain_ratio": f["sustain_ratio"],
        "flat_mid": f["flatness"]["mid"],
        "flat_high": f["flatness"]["high"],
    }


def delta(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, str):
        return 0.0 if a == b else 1.0
    return float(b) - float(a)


def fmt(v):
    if v is None:
        return "  —  "
    if isinstance(v, str):
        return f"{v:>5s}"
    if abs(v) >= 1000:
        return f"{v:5.0f}"
    if abs(v) >= 100:
        return f"{v:5.1f}"
    return f"{v:5.2f}"


def run(keep=None, json_path=None):
    st = render_stems()
    out_dir = keep or tempfile.mkdtemp(prefix="ablate_")

    def render(name, **kw):
        bus = build(st, **kw)
        path = os.path.join(out_dir, f"{name}.wav")
        sl.write_wav(path, bus, fade_out=0.0)
        return path

    base_path = render("base")
    base = features(sp.analyse(base_path))
    rows, failures, warnings = [], [], []

    print(f"base: {os.path.basename(base_path)}  ({N / SR:.1f}s, {BPM:.0f} BPM)")
    print(f"      {base['key']} | centroid {base['centroid_hz']} Hz | width {base['width']:.2f}"
          f" | backbeat {base['backbeat']} | crest {base['crest_db']} dB")
    print()

    for abl in ABLATIONS:
        path = render(abl["key"], **abl["variant"])
        cur = features(sp.analyse(path))
        report = {"key": abl["key"], "label": abl["label"], "variant": abl["variant"],
                  "features": cur, "checks": []}
        print(f"■ {abl['label']}")
        print(f"  {'feature':<20s} {'base':>6s} {'variant':>7s} {'delta':>7s}   expectation")
        checked = set()
        for kind, items in (("must", abl["must"]), ("watch", abl["watch"])):
            for name, direction, thresh in items:
                checked.add(name)
                d = delta(base[name], cur[name])
                ok = d is not None and (
                    (direction == "down" and d <= -thresh) or
                    (direction == "up" and d >= thresh) or
                    (direction == "same" and abs(d) <= thresh))
                if kind == "must" and not ok:
                    failures.append((abl["label"], name, d, direction, thresh))
                    mark = "FAIL"
                elif not ok:
                    mark = "not seen"
                else:
                    mark = "ok"
                print(f"  {label_of(name):<20s} {fmt(base[name]):>6s} {fmt(cur[name]):>7s} "
                      f"{fmt(d):>7s}   {direction} ≥{thresh:g} → {mark}")
                report["checks"].append({"feature": name, "kind": kind, "delta": d,
                                         "expect": direction, "threshold": thresh,
                                         "ok": bool(ok)})
        for name, thresh in abl["stable"]:
            checked.add(name)
            d = delta(base[name], cur[name])
            if d is not None and abs(d) > thresh:
                warnings.append((abl["label"], name, d, thresh))
                print(f"  {label_of(name):<20s} {fmt(base[name]):>6s} {fmt(cur[name]):>7s} "
                      f"{fmt(d):>7s}   should hold still (<{thresh:g}) → WARN")
        other = []
        for name, _ in FEATURES:
            if name in checked:
                continue
            d = delta(base[name], cur[name])
            if d is None:
                continue
            if abs(d) > NOISE.get(name, 0.05):
                other.append((name, d))
        if other:
            txt = ", ".join(f"{label_of(n)} {d:+.2f}" for n, d in other)
            print(f"  其它变动: {txt}")
        report["other"] = [{"feature": n, "delta": d} for n, d in other]
        rows.append(report)
        print()

    print("─" * 64)
    total = sum(1 for r in rows for c in r["checks"] if c["kind"] == "must")
    print(f"{total - len(failures)}/{total} required movements held"
          f" · {len(warnings)} specificity warnings")
    for label, name, d, direction, thresh in failures:
        print(f"  FAIL  {label} / {label_of(name)}: Δ={fmt(d).strip()} "
              f"(wanted {direction} ≥{thresh:g})")

    if json_path:
        with open(json_path, "w") as fh:
            json.dump({"base": base, "ablations": rows}, fh, ensure_ascii=False, indent=2)
        print("wrote", json_path)
    if keep:
        print("kept", keep)
    return 0 if not failures else 1


def label_of(name):
    return dict(FEATURES).get(name, name)


def main(argv=None):
    ap = argparse.ArgumentParser(description="paired ablations for the style probe")
    ap.add_argument("--keep", metavar="DIR", help="keep the rendered wavs here")
    ap.add_argument("--json", metavar="PATH", help="write the numbers to a json file")
    args = ap.parse_args(argv)
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
    return run(keep=args.keep, json_path=args.json)


if __name__ == "__main__":
    sys.exit(main())
