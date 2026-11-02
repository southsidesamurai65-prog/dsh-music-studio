#!/usr/bin/env python3
"""
pop_ballad.py — starting point for a 2000s Chinese-pop ballad.

Piano 16ths, strings, electric bass, a real backbeat, and a solo lead on top.
Copy this file into pieces/<name>/, change BPM/key/progression/melody there;
do not copy the previous piece's script instead.

    python3 pop_ballad.py            # writes pop_ballad_demo.wav next to itself
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import harmony as hm
import mix as mx
import synth_lib as sl

SR = sl.SR
BPM = 76.0
BEAT = 60 / BPM
BAR = 4 * BEAT
BARS = 16
TAIL = 4.0
N = int((BARS * BAR + TAIL) * SR)

sl.seed(11)
P = sl.midi2f

# render into stems, not one bus: the kit needs its own compressor, and a
# single bus cannot be un-mixed afterwards
st = sl.Stems(N / SR)


def bt(bar, beat=0.0):
    return (bar - 1) * BAR + beat * BEAT


def place(sig, t0, pan=0.0, gain=1.0, rev=0.25, role="music"):
    st.add(role, sig, t0, pan, gain)
    if rev:
        st.add("fx", sig, t0, pan * 0.7, gain * rev)


# ---------------------------------------------------------------- harmony
# Chords are scale degrees, and each voicing is led from the previous one, so
# the inner parts become lines instead of blocks.
KEY = hm.Key("D")
PROG = ["Imaj7", "V/3", "vim7", "IVmaj7", "I/3", "iim7", "Vsus4", "V7"] * 2
VOICINGS = hm.realize(PROG, KEY)          # four voices, nearest motion
BASSES = [v[0] - 12 for v in VOICINGS]    # bass an octave under the chord

# ---------------------------------------------------------------- melody
MELODY = [  # (bar, beat, beats, midi, velocity)
    (1, 0.0, 1.0, 66), (1, 1.0, 1.0, 69), (1, 2.0, 2.0, 71),
    (2, 0.0, 1.0, 69), (2, 1.0, 1.0, 66), (2, 2.0, 2.0, 64),
    (3, 0.0, 0.5, 66), (3, 0.5, 0.5, 69), (3, 1.0, 1.0, 71), (3, 2.0, 2.0, 74),
    (4, 0.0, 1.5, 71), (4, 1.5, 0.5, 69), (4, 2.0, 2.0, 66),
    (5, 0.0, 1.0, 69), (5, 1.0, 1.0, 71), (5, 2.0, 2.0, 74),
    (6, 0.0, 1.0, 76), (6, 1.0, 1.0, 74), (6, 2.0, 2.0, 71),
    (7, 0.0, 0.5, 69), (7, 0.5, 0.5, 71), (7, 1.0, 1.0, 74), (7, 2.0, 2.0, 71),
    (8, 0.0, 2.0, 66), (8, 2.0, 2.0, 69),
] + [
    (9, 0.0, 1.0, 74), (9, 1.0, 1.0, 76), (9, 2.0, 2.0, 78),
    (10, 0.0, 1.0, 76), (10, 1.0, 1.0, 74), (10, 2.0, 2.0, 71),
    (11, 0.0, 0.5, 74), (11, 0.5, 0.5, 71), (11, 1.0, 1.0, 74), (11, 2.0, 2.0, 78),
    (12, 0.0, 2.0, 76), (12, 2.0, 2.0, 74),
    (13, 0.0, 1.0, 71), (13, 1.0, 1.0, 74), (13, 2.0, 2.0, 78),
    (14, 0.0, 1.0, 76), (14, 1.0, 1.0, 74), (14, 2.0, 2.0, 71),
    (15, 0.0, 0.5, 69), (15, 0.5, 0.5, 71), (15, 1.0, 1.0, 74), (15, 2.0, 2.0, 71),
    (16, 0.0, 4.0, 74),
]

# ---------------------------------------------------------------- backing
for bar in range(1, BARS + 1):
    if bar == 1:
        continue
    bar_idx = (bar - 1) % len(PROG)
    notes = VOICINGS[bar_idx]              # four voice-led chord tones
    bass_m = BASSES[bar_idx]
    t0 = bt(bar)
    chorus = bar >= 9

    # piano: broken eighths over the voicing, not a fixed block
    seq = [(0.0, 0, .55), (0.5, 2, .40), (1.0, 1, .45), (1.5, 3, .40),
           (2.0, 2, .50), (2.5, 1, .38), (3.0, 3, .46), (3.5, 2, .36)]
    for off, voice, v in seq:
        place(sl.piano(P(notes[voice % len(notes)]), 1.2, v),
              t0 + off * BEAT, 0.16, 0.55, 0.22)
    place(sl.piano(P(bass_m + 12), 1.6, .40), t0, -0.06, 0.50, 0.20)

    # strings from bar 5, the upper three voices an octave up
    if bar >= 5:
        for i, m in enumerate(notes[1:4]):
            place(sl.strings(P(m + 12), BAR * 1.03, .5), t0, -0.55 + 0.45 * i,
                  0.34 if not chorus else 0.5, 0.32)

    # bass
    place(sl.ebass(P(bass_m), 0.6, .9), t0, 0.0, 0.8, 0.05, role="bass")
    place(sl.ebass(P(bass_m + 7), 0.5, .6), t0 + 2 * BEAT, 0.0, 0.7, 0.05, role="bass")
    if chorus:
        place(sl.ebass(P(bass_m + 12), 0.4, .5), t0 + 3.5 * BEAT, 0.0, 0.6, 0.05,
              role="bass")

    # drums: backbeat (snare on 2 and 4)
    place(sl.kick(.85), t0, 0.0, 0.9, 0.02, role="drums")
    place(sl.kick(.65), t0 + 2 * BEAT, 0.0, 0.85, 0.02, role="drums")
    place(sl.snare(.55 if not chorus else .85), t0 + BEAT, 0.05, 0.85, 0.03,
          role="drums")
    place(sl.snare(.55 if not chorus else .85), t0 + 3 * BEAT, 0.05, 0.85, 0.03,
          role="drums")
    for k in range(8):
        place(sl.hat(k % 4 == 2, .28), t0 + 0.5 * k * BEAT,
              0.55 if k % 2 else -0.2, 0.7, 0.02, role="drums")
    if chorus:
        for k in range(16):
            place(sl.shaker(.30), t0 + 0.25 * k * BEAT,
                  -0.6 if k % 2 else 0.55, 0.8, 0.02, role="drums")

# ---------------------------------------------------------------- lead
prev = None
for bar, beat, dur_b, m in MELODY:
    d = dur_b * BEAT * 0.95
    sig = sl.dizi(P(m), d, 0.85, slide_from=prev if d > 0.4 else None)
    place(sig, bt(bar, beat), 0.05, 0.9, 0.34)
    prev = P(m)

# ---------------------------------------------------------------- mix
# drums -> glue + parallel bus; music -> multiband; both -> one reverb -> limiter
ir = sl.make_ir(2.2, 1.05, 180, 6500)
mix = mx.mixdown(st, preset="pop", ir=ir)

sl.write_wav(str(HERE / "pop_ballad_demo.wav"), mix, fade_out=3.0)
print("wrote pop_ballad_demo.wav  %.1fs" % (len(mix[0]) / SR))
