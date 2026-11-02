#!/usr/bin/env python3
"""
phrase.py — motif development and phrase building for melodic writing.

The complaint this answers: melodies that are one bar repeated. A motif is not
a melody; a melody is what you do to a motif. This module provides the
operations, so a piece can be written as a *plan* over a motif rather than as a
list of bars.

Everything works on scale degrees, so transposition and inversion stay in key.

    motif = [(0.0, 1.0, 3), (1.0, 0.5, 4), (1.5, 0.5, 2), (2.0, 2.0, 0)]
    a  = render(motif, scale_degrees, root_midi=62, t0=bar(5))
    a2 = render(sequence(motif, 1, +1), ...)          # tonal sequence
    b  = render(invert(motif), ...)                    # contrast
    a3 = render(ornament(augment(motif, 1.5)), ...)    # closing statement
"""
import numpy as np


# ------------------------------------------------------------------ material
def _norm(motif):
    """(offset, dur, degree) tuples, sorted, offsets relative to the first."""
    m = sorted(motif, key=lambda e: e[0])
    off0 = m[0][0] if m else 0.0
    return [(round(e[0] - off0, 4), e[1], e[2]) for e in m]


def transpose(motif, steps):
    """Shift every degree by `steps` scale steps (tonal, stays in key)."""
    return [(o, d, g + steps) for o, d, g in _norm(motif)]


def invert(motif, pivot, anchor=None):
    """Mirror the contour around a pivot degree. anchor re-seats the result."""
    m = _norm(motif)
    inv = [(o, d, pivot - (g - pivot)) for o, d, g in m]
    if anchor is not None:
        shift = anchor - inv[0][2]
        inv = [(o, d, g + shift) for o, d, g in inv]
    return inv


def retrograde(motif):
    """Play the shape backwards, keeping the rhythm forward."""
    m = _norm(motif)
    total = max(o + d for o, d, _ in m)
    return [(round(total - o - d, 4), d, g) for o, d, g in m]


def augment(motif, factor=2.0):
    """Stretch the rhythm (rhythmic augmentation)."""
    return [(o * factor, d * factor, g) for o, d, g in _norm(motif)]


def diminish(motif, factor=0.5):
    return augment(motif, factor)


def fragment(motif, start, length):
    """Keep a slice of the motif (fragmentation / sequencing material)."""
    m = _norm(motif)
    part = [e for e in m if start <= e[0] < start + length]
    return _norm(part) if part else m[:1]


def sequence(motif, times, step, keep_last=False):
    """Repeat the motif up (or down) by `step` degrees, as a tonal sequence."""
    m = _norm(motif)
    span = max(o + d for o, d, _ in m)
    out = []
    for k in range(times):
        out += [(o + k * span, d, g + k * step) for o, d, g in m]
    if keep_last:
        out[-1] = (out[-1][0], out[-1][1], m[-1][2])
    return out


def ornament(motif, neighbor=1, passing=True):
    """Add upper neighbours on long notes and passing tones in big leaps."""
    m = _norm(motif)
    out = []
    for o, d, g in m:
        out.append((o, d, g))
        if d >= 1.0:
            # upper neighbour on the tail of a long note
            out.append((round(o + d * 0.75, 4), round(d * 0.25, 4), g + neighbor))
    if passing:
        fixed = []
        for i, (o, d, g) in enumerate(sorted(out)):
            fixed.append((o, d, g))
            if i + 1 < len(out):
                no, nd, ng = sorted(out)[i + 1]
                if abs(ng - g) >= 3 and no - (o + d) > 0.2:
                    fixed.append((round(o + d, 4), 0.25, g + (1 if ng > g else -1)))
        out = fixed
    return _norm(out)


def cadence(motif, target_degree, approach=1):
    """Force the phrase to land on a chord tone: last note = target."""
    m = _norm(motif)
    out = m[:-1]
    o, d, _ = m[-1]
    if len(out) >= 2:
        prev = out[-1]
        out[-1] = (prev[0], prev[1], target_degree + approach
                   if target_degree + approach - prev[2] <= 2 else prev[2])
    out.append((o, d, target_degree))
    return _norm(out)


def humanize(motif, rng, timing=0.012, velocity=None):
    """Micro-timing and velocity variation so a phrase stops sounding gridded."""
    out = []
    for o, d, g in motif:
        jitter = rng.normal(0, timing)
        out.append((max(0.0, o + jitter), d, g))
    return _norm(out)


# -------------------------------------------------------------------- render
def render(motif, scale, root_midi, t0, beat_s, octave=0, vel=0.85):
    """Turn degrees into (start_sec, dur_sec, midi, vel) note events."""
    events = []
    for o, d, g in _norm(motif):
        idx = g
        oct_shift = 0
        while idx < 0:
            idx += len(scale)
            oct_shift -= 12
        while idx >= len(scale):
            idx -= len(scale)
            oct_shift += 12
        midi = root_midi + scale[idx] + octave + oct_shift
        events.append((t0 + o * beat_s, d * beat_s, midi, vel))
    return events


def repetition_score(events, n=4):
    """Share of pitch-contour n-grams that occur more than once (0 = varied)."""
    contour = [e[2] for e in events]
    if len(contour) < n:
        return 1.0
    grams = [tuple(np.sign(np.diff(contour[i:i + n + 1])))
             for i in range(len(contour) - n)]
    grams = [g for g in grams if len(g) == n]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    return round(1 - uniq / len(grams), 3)
