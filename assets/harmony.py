#!/usr/bin/env python3
"""
harmony.py — chords as scale degrees, and voicings that move as little as
possible. numpy only (in fact, no numpy needed: this is score-level).

The old library stored harmony as a dictionary of absolute pitches:

    VOICE = {"Dmaj7": (38, [50, 57, 62, 66])}

Three things go wrong with that. The voicing is a fixed block, so every chord
change makes the inner voices jump. The chord has no function, so "this is the
dominant" cannot be expressed. And the pitches are absolute, so transposing a
piece means rewriting the table.

Here a chord is a *degree* plus a quality, and a voicing is the cheapest motion
from the previous one:

    k = hm.Key("D")
    ch = hm.parse("V7", k)            # A7
    v = hm.voice(ch, prev=None)       # first voicing: from the centre out
    v = hm.voice(hm.parse("I", k), prev=v)   # every voice moves as little as it can

`realize()` does that for a whole progression. The point is not that the chords
change — they are the same chords — but that the inner voices become lines
instead of blocks.
"""
import itertools
import re

NOTE_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10,
    "Bb": 10, "B": 11, "Cb": 11,
}
PC_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

SCALES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "minor":          [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "dorian":         [0, 2, 3, 5, 7, 9, 10],
    "mixolydian":     [0, 2, 4, 5, 7, 9, 10],
}

QUALITIES = {
    "":      (0, 4, 7),
    "m":     (0, 3, 7),
    "dim":   (0, 3, 6),
    "aug":   (0, 4, 8),
    "5":     (0, 7),
    "6":     (0, 4, 7, 9),
    "m6":    (0, 3, 7, 9),
    "7":     (0, 4, 7, 10),
    "maj7":  (0, 4, 7, 11),
    "m7":    (0, 3, 7, 10),
    "m7b5":  (0, 3, 6, 10),
    "dim7":  (0, 3, 6, 9),
    "9":     (0, 4, 7, 10, 14),
    "maj9":  (0, 4, 7, 11, 14),
    "m9":    (0, 3, 7, 10, 14),
    "sus2":  (0, 2, 7),
    "sus4":  (0, 5, 7),
    "7sus4": (0, 5, 7, 10),
    "add9":  (0, 4, 7, 14),
}

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]
_SUFFIX_ORDER = sorted(QUALITIES, key=len, reverse=True)   # "m7b5" before "m"


class Chord:
    """A chord as pitch classes: root, quality, optional bass."""

    __slots__ = ("root", "quality", "bass", "label")

    def __init__(self, root, quality="", bass=None, label=None):
        if quality not in QUALITIES:
            raise ValueError(f"unknown chord quality {quality!r}")
        self.root = int(root) % 12
        self.quality = quality
        self.bass = self.root if bass is None else int(bass) % 12
        self.label = label or self.name()

    def pcs(self):
        return [(self.root + i) % 12 for i in QUALITIES[self.quality]]

    def name(self):
        s = PC_NAME[self.root] + self.quality
        if self.bass != self.root:
            s += "/" + PC_NAME[self.bass]
        return s

    def __repr__(self):
        return f"Chord({self.label})"

    def __eq__(self, other):
        return (isinstance(other, Chord) and self.root == other.root
                and self.quality == other.quality and self.bass == other.bass)


class Key:
    """Tonic + mode; everything else is a degree of it."""

    def __init__(self, tonic="C", mode="major"):
        self.tonic = NOTE_PC[tonic] if isinstance(tonic, str) else int(tonic) % 12
        if mode not in SCALES:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.scale = SCALES[mode]

    def pc(self, degree):
        """Pitch class of a 1-based scale degree (7 == 1 an octave up)."""
        return (self.tonic + self.scale[(degree - 1) % 7]) % 12

    def degree(self, pc):
        """Scale degree nearest to a pitch class; ties resolve to the flat side."""
        d, _ = self.degree_delta(pc)
        return d

    def degree_delta(self, pc):
        """(degree, chromatic offset): +1 means #, 11 means b, 0 is diatonic."""
        best = None
        for d in range(1, 8):
            delta = (pc - self.pc(d)) % 12
            dist = min(delta, 12 - delta)
            # prefer exact, then the flatter neighbour on a tie
            rank = (0 if delta == 0 else 1, dist, 0 if delta > 6 else 1, d)
            if best is None or rank < best[0]:
                best = (rank, d, delta)
        return best[1], best[2]

    def label(self, chord):
        """Roman numeral of a chord, e.g. 'V7', 'bVI', 'ii7'."""
        d, delta = self.degree_delta(chord.root)
        deg = ROMAN[d - 1]
        if delta:
            deg = ("b" if delta > 6 else "#") + deg
        if chord.quality in ("m", "m7", "m9", "m6", "m7b5"):
            deg = deg.lower()
        return deg + _quality_suffix(chord.quality)

    def __repr__(self):
        return f"Key({PC_NAME[self.tonic]} {self.mode})"


def _quality_suffix(q):
    return "" if q in ("", "m") else q


def _parse_numeral(token):
    """'bVII' -> (-1, 7); 'V' -> (0, 5); 'vi' -> (0, 6, minor)."""
    m = re.fullmatch(r"([b#]*)([IViv]+)(.*)", token)
    if not m:
        raise ValueError(f"cannot read roman numeral {token!r}")
    acc, rom, suffix = m.groups()
    upper = rom.isupper()
    if rom.upper() not in ROMAN:
        raise ValueError(f"unknown degree {token!r}")
    degree = ROMAN.index(rom.upper()) + 1
    shift = sum(1 if c == "#" else -1 for c in acc)
    return degree, shift, upper, suffix


def parse(symbol, key):
    """Parse a roman numeral (or a chord name) into a Chord.

    Handles: 'I', 'vi', 'V7', 'ii7b5', 'bVI', '#IVdim7', 'Isus4',
    'V/V' (secondary dominant), 'ii/V', and inversions 'I/3', 'V7/5'.
    A literal chord name ('Dmaj7', 'Bb', 'F#m7') is accepted too.
    """
    symbol = symbol.strip()
    if symbol in NOTE_PC or re.fullmatch(r"[A-G][#b]?.*", symbol):
        return _parse_chord_name(symbol)

    # secondary dominant / applied chord: everything before the last '/' is
    # read in the key of the target
    if "/" in symbol:
        head, tail = symbol.rsplit("/", 1)
        if re.fullmatch(r"[IViv]+", tail):
            target = parse(tail, key)
            tkey = Key(target.root, "major" if target.quality in ("", "7", "maj7") else "minor")
            chord = parse(head, tkey)
            chord.label = symbol
            return chord
        if re.fullmatch(r"[1-7]", tail):
            chord = parse(head, key)
            return _invert(chord, int(tail))
        if tail in NOTE_PC:
            chord = parse(head, key)
            chord.bass = NOTE_PC[tail]
            chord.label = symbol
            return chord

    degree, shift, upper, suffix = _parse_numeral(symbol)
    root = (key.pc(degree) + shift) % 12
    quality = _quality_from(upper, suffix, key, degree, shift)
    return Chord(root, quality, label=symbol)


_SUFFIX_MAP = {
    "": ("", "m"), "7": ("7", "m7"), "maj7": ("maj7", "maj7"),
    "m7": ("m7", "m7"), "6": ("6", "m6"), "9": ("9", "m9"),
    "maj9": ("maj9", "maj9"), "m9": ("m9", "m9"), "5": ("5", "5"),
    "dim": ("dim", "dim"), "dim7": ("dim7", "dim7"),
    "sus2": ("sus2", "sus2"), "sus4": ("sus4", "sus4"),
    "7sus4": ("7sus4", "7sus4"), "add9": ("add9", "add9"),
    "7b5": ("m7b5", "m7b5"), "m7b5": ("m7b5", "m7b5"),
}


def _quality_from(upper, suffix, key, degree, shift):
    """Pick the quality: the written suffix wins, otherwise the case does.

    Case, not the diatonic third: 'IV' is major and 'iv' is minor whatever the
    mode, which is also what makes borrowed chords readable ('bVI' -> Bb major,
    not Bbm in a major key). A suffix like '7' follows the case: 'V7' is a
    dominant seventh, 'ii7' a minor seventh.
    """
    if suffix in _SUFFIX_MAP:
        return _SUFFIX_MAP[suffix][0 if upper else 1]
    if suffix in QUALITIES:
        return suffix
    raise ValueError(f"unknown chord suffix {suffix!r}")


def _invert(chord, n):
    """'I/3' means the third in the bass — we only need the bass pitch class."""
    pcs = chord.pcs()
    if n < 1 or n > len(pcs):
        raise ValueError(f"no chord tone {n} in {chord.name()}")
    chord.bass = pcs[n - 1]
    chord.label = f"{chord.label}/{n}"
    return chord


def _parse_chord_name(s):
    m = re.fullmatch(r"([A-G][#b]?)(.*?)(?:/([A-G][#b]?))?", s)
    if not m:
        raise ValueError(f"cannot read chord name {s!r}")
    root_s, quality, bass_s = m.groups()
    if quality not in QUALITIES:
        raise ValueError(f"unknown chord quality {quality!r} in {s!r}")
    return Chord(NOTE_PC[root_s], quality,
                 None if bass_s is None else NOTE_PC[bass_s], label=s)


# ------------------------------------------------------------------ voicing
def _candidates(pc, prev, lo, hi, near=9):
    """Every octave of a pitch class within range, closest ones first."""
    out = []
    for p in range(lo, hi + 1):
        if p % 12 == pc % 12:
            out.append(p)
    if prev is not None and out:
        out.sort(key=lambda p: (abs(p - prev), p))
        out = out[:near]
    return out


def _motion(a, b):
    return abs(b - a)


def _fit_prev(prev, voices):
    """Adapt a previous voicing to a different voice count (4 <-> 5 chords)."""
    p = list(prev)
    if len(p) < voices:
        while len(p) < voices:
            p.append(p[-1] + 12)
        return p
    if len(p) > voices:
        idx = sorted({round(k * (len(p) - 1) / (voices - 1)) for k in range(voices)})
        while len(idx) < voices:
            idx.append(idx[-1] - 1)
        return [p[i] for i in sorted(idx)]
    return p


def _parallel_penalty(prev, cur):
    """Forbid (heavily) parallel fifths and octaves between adjacent voices."""
    pen = 0.0
    for i in range(len(prev)):
        for j in range(i + 1, len(prev)):
            a, b = prev[i], prev[j]
            c, d = cur[i], cur[j]
            if a == b or c == d:
                continue
            ia, ib = (b - a) % 12, (d - c) % 12
            if ia == ib and ia in (0, 7):
                if (b - a) * (d - c) > 0 and a != c:
                    pen += 1e6   # effectively forbidden
    return pen


def voice(chord, prev=None, center=62, voices=None, lo=45, hi=84):
    """Nearest voicing of `chord`, moving as little as possible from `prev`.

    Cost = total semitone motion + parallel perfect penalties + spacing and
    doubling rules. The search is exhaustive over a small candidate set per
    voice (<= 9 options each), which is cheap and exact enough for 4 voices.
    """
    pcs = chord.pcs()
    if voices is None:
        voices = max(4, len(set(pcs)))      # a 9th needs five voices to speak
    if prev is not None:
        prev = _fit_prev(prev, voices)
    if chord.bass not in pcs:
        pcs = [chord.bass] + pcs
    if prev is None:
        # first chord: bass in the low register, the rest stacked above it
        base = [p for p in range(lo + 3, center + 1) if p % 12 == chord.bass % 12]
        bass = max(base) if base else lo + 3
        upper = sorted({p for p in range(bass + 1, bass + 25) if p % 12 in pcs})
        chosen = [bass] + upper[: voices - 1]
        while len(chosen) < voices:      # a two-note chord still needs a voice
            chosen.append(chosen[-1] + 12)
        return chosen

    def candidates(prev, tight):
        """Per-voice pools near each voice's previous pitch (or a wide pool)."""
        k = (4 if tight else 9)
        cap = (7 if voices >= 5 else 9) if tight else 14
        out = []
        for i in range(voices):
            if i == 0:
                opts = [p for p in range(lo, hi + 1) if p % 12 == chord.bass % 12]
                near = [p for p in opts if prev is None or abs(p - prev[0]) <= 7]
                opts = near or opts
                opts.sort(key=lambda p: abs(p - (prev[0] if prev else center)))
                out.append(opts[:5])
            else:
                pool = {p for pc in pcs for p in _candidates(pc, None if prev is None else prev[i],
                                                             lo, hi, near=k)}
                ref = center if prev is None else prev[i]
                pool = sorted(pool, key=lambda p: abs(p - ref))[:cap]
                out.append(sorted(pool) or [ref])
        return out

    cands = candidates(prev, tight=True)
    best, best_cost = None, float("inf")
    for combo in itertools.product(*cands):
        if len(set(combo)) < len(combo):
            continue
        if any(combo[i] >= combo[i + 1] for i in range(voices - 1)):
            continue
        cost = sum(_motion(prev[i], combo[i]) for i in range(voices))
        cost += _parallel_penalty(prev, combo)
        # keep each voice near its own register, or the nearest-motion rule
        # will let the bass walk down an octave every time the loop repeats
        targets = (center - 11, center - 3, center + 2, center + 7)
        cost += 0.6 * sum(abs(combo[i] - targets[min(i, 3)]) for i in range(voices))
        for i in range(voices - 1):
            gap = combo[i + 1] - combo[i]
            if gap < 3:
                cost += 6.0
            if gap > 12:
                cost += 2.0 * (gap - 12)
        if combo[0] < 40 or combo[-1] > 86:
            cost += 50.0
        missing = [pc for pc in pcs if pc not in [p % 12 for p in combo]]
        cost += 8.0 * len(missing)
        if cost < best_cost:
            best, best_cost = combo, cost

    if best is None:                     # dense chord, tight pool: widen once
        for combo in itertools.product(*candidates(prev, tight=False)):
            if len(set(combo)) < len(combo):
                continue
            if any(combo[i] >= combo[i + 1] for i in range(voices - 1)):
                continue
            cost = sum(_motion(prev[i], combo[i]) for i in range(voices))
            if cost < best_cost:
                best, best_cost = combo, cost
    if best is None:
        return list(prev)
    return list(best)


def realize(chords, key, center=62, voices=4):
    """Voice a whole progression, each chord starting from the previous one."""
    out, prev = [], None
    for ch in chords:
        if isinstance(ch, str):
            ch = parse(ch, key)
        prev = voice(ch, prev, center=center, voices=voices)
        out.append(prev)
    return out


def motion_total(voicings):
    """Total semitone motion across a progression — the score-level test."""
    return sum(sum(_motion(a, b) for a, b in zip(prev, cur))
               for prev, cur in zip(voicings, voicings[1:]))


def parallels(voicings):
    """Count parallel fifths/octaves between consecutive voicings."""
    n = 0
    for prev, cur in zip(voicings, voicings[1:]):
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                a, b = prev[i], prev[j]
                c, d = cur[i], cur[j]
                if (b - a) % 12 == (d - c) % 12 and (b - a) % 12 in (0, 7) \
                        and (b - a) * (d - c) > 0 and a != c:
                    n += 1
    return n


def block(names, key, center=62):
    """The old way: stack each chord from a fixed register, no voice leading."""
    out = []
    for name in names:
        ch = parse(name, key) if isinstance(name, str) else name
        pcs = ch.pcs()
        base = [p for p in range(center - 12, center + 1) if p % 12 == ch.bass]
        v = [max(base) if base else center - 12]
        for p in sorted({p for pc in pcs for p in range(v[0], 85) if p % 12 == pc}):
            if len(v) >= 4:
                break
            if p > v[0]:
                v.append(p)
        out.append(v)
    return out


# ------------------------------------------------------------- progression
T, PD, D = "T", "PD", "D"
FUNCTION = {"I": T, "vi": T, "iii": T, "ii": PD, "IV": PD, "V": D, "vii": D}


class Progression:
    """A chord per bar, written as scale degrees, with a functional generator.

    The generator is not a random walk over chords: it fills a phrase with
    functions (tonic / predominant / dominant), lands the cadence, and only
    then spends a chromatic budget on applied dominants and borrowed chords.
    """

    def __init__(self, key, bars=8):
        self.key = key
        self.bars = bars
        self.slots = []          # one symbol per bar
        self.slot_keys = []      # the key each slot is written in (modulation)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_symbols(cls, symbols, key):
        p = cls(key, len(symbols))
        p.slots = list(symbols)
        p.slot_keys = [key] * len(symbols)
        return p

    def generate(self, bars=None, cadence="PAC", chromatic=2, seed=0,
                 rhythm=None):
        """Build a phrase. `chromatic` is the maximum number of altered chords."""
        import random
        rng = random.Random(seed)
        n = bars or self.bars
        self.bars = n
        self.slots = self._skeleton(n, cadence, rng)
        self._add_chromatic(n, cadence, chromatic, rng)
        self.slot_keys = [self.key] * n
        return self

    def _skeleton(self, n, cadence, rng):
        """Function-first: 4-bar groups, each closing on a dominant."""
        deg = []
        for b in range(n):
            pos = b % 4
            if pos == 0:
                deg.append(rng.choice(["I", "I", "vi", "iii"]))
            elif pos == 1:
                deg.append(rng.choice(["IV", "ii", "IV"]))
            elif pos == 2:
                deg.append(rng.choice(["V", "V", "ii"]))
            else:
                deg.append(rng.choice(["I", "I", "vi", "V"]))
        # final bar is the cadence target
        if n >= 2:
            if cadence == "PAC":
                deg[-2], deg[-1] = "V", "I"
            elif cadence == "HC":
                deg[-2], deg[-1] = "I", "V"
            elif cadence == "DC":
                deg[-2], deg[-1] = "V", "vi"
            elif cadence == "PC":
                deg[-2], deg[-1] = "IV", "I"
        return deg

    def _add_chromatic(self, n, cadence, budget, rng):
        """Spend the budget on applied dominants, borrowed chords, one tritone sub.

        Each insertion must resolve forward: a V7/x sits immediately before x, a
        borrowed chord only before a tonic or subdominant, and the tritone
        substitute only before the tonic. Two altered chords never touch.
        """
        if budget <= 0 or n < 4:
            return
        used, spend = set(), 0
        order = list(range(1, n - 2))
        rng.shuffle(order)
        for i in order:
            if spend >= budget:
                break
            if (i - 1) in used or i in used or i - 1 <= 0 or i >= n - 2:
                continue
            target, prev = self.slots[i], self.slots[i - 1]
            if prev not in FUNCTION or target not in FUNCTION or FUNCTION[prev] == D:
                continue
            roll = rng.random()
            if roll < 0.60 and target in ("ii", "IV", "V", "vi"):
                slot = f"V7/{target}"
            elif roll < 0.80 and target in ("I", "IV") and self.key.mode == "major":
                slot = rng.choice(["bVI", "bVII", "iv"])
            elif roll < 0.92 and target == "I":
                slot = "bII7"
            else:
                continue
            if slot == self.slots[i - 1] or slot in used:
                continue
            self.slots[i - 1] = slot
            used.add(i - 1)
            spend += 1

    # -- modulation ------------------------------------------------------
    def modulate(self, target, at_bar, via=None, cadence="PAC", seed=0,
                 chromatic=1):
        """Pivot into another key at `at_bar` (1-based).

        The pivot is a chord that is diatonic in both keys, so the ear hears it
        as an old chord that happens to mean something new; the bars after it are
        generated in the target key.
        """
        if not (1 <= at_bar <= len(self.slots)):
            raise ValueError("at_bar out of range")
        head = self.slots[:at_bar - 1]
        tail_len = len(self.slots) - at_bar
        pivot, pivot_note = self._pivot(target, via)
        tail = Progression(target, max(tail_len, 0))
        if tail_len > 0:
            tail.generate(cadence=cadence, chromatic=chromatic, seed=seed)
        self.slots = head + [pivot] + tail.slots
        self.slot_keys = ([self.key] * at_bar) + [target] * tail_len
        self.bars = len(self.slots)
        self._last_pivot = pivot_note
        return self

    def _pivot(self, target, via=None):
        """A chord diatonic in both keys, preferring the requested symbol."""
        def pcs_of(sym, key):
            return set(parse(sym, key).pcs())

        target_pcs = {(target.tonic + d) % 12 for d in target.scale}
        candidates = []
        for d in range(1, 8):
            sym = ROMAN[d - 1] if d in (1, 4, 5) else ROMAN[d - 1].lower()
            if pcs_of(sym, self.key) <= target_pcs:
                candidates.append(sym)
        if via is not None:
            if pcs_of(via, self.key) <= target_pcs:
                return via, self._pivot_note(via, target)
            raise ValueError(f"{via} is not diatonic in both keys")
        if not candidates:
            return "V", self._pivot_note("V", target)
        # prefer a pivot that already means something in the new key (a
        # subdominant or dominant) over one that turns into a weak mediant
        prefer = ["iv", "IV", "VI", "VII", "V", "i", "I", "ii", "v", "vi", "III", "iii"]
        def rank(sym):
            lab = target.label(parse(sym, self.key))
            return (prefer.index(lab) if lab in prefer else 99, sym)
        best = min(candidates, key=rank)
        return best, self._pivot_note(best, target)

    def _pivot_note(self, sym, target):
        ch = parse(sym, self.key)          # the symbol is written in the old key
        return (f"{sym} = {self.key.label(ch)} in {self.key} / "
                f"{target.label(ch)} in {target}")

    # -- reading ---------------------------------------------------------
    def chords(self):
        return [parse(s, k) for s, k in zip(self.slots, self.slot_keys or [self.key] * len(self.slots))]

    def roman(self):
        return list(self.slots)

    def names(self):
        return [c.name() for c in self.chords()]

    def realize(self, center=62, voices=4):
        out, prev = [], None
        for ch in self.chords():
            prev = voice(ch, prev, center=center, voices=voices)
            out.append(prev)
        return out

    def validate(self):
        """Cheap syntax checks: applied chords must resolve to their target."""
        problems = []
        ch = self.chords()
        keys = self.slot_keys or [self.key] * len(self.slots)
        for i, (sym, k) in enumerate(zip(self.slots, keys)):
            try:
                parse(sym, k)
            except ValueError as exc:
                problems.append(f"bar {i + 1}: {sym} does not parse in {k}: {exc}")
        for i, sym in enumerate(self.slots):
            if "/" in sym and re.fullmatch(r"V7/[IViv]+", sym):
                target = sym.split("/")[1]
                nxt = self.slots[i + 1] if i + 1 < len(self.slots) else None
                if nxt != target:
                    problems.append(f"bar {i + 1}: {sym} does not resolve to {target}")
            if i and self.slots[i - 1] == sym and sym not in ("I", "V"):
                problems.append(f"bar {i + 1}: {sym} repeats")
        return problems

    def __len__(self):
        return len(self.slots)

    def __repr__(self):
        return "Progression(" + " ".join(self.slots) + ")"


def check_melody(melody, progression, key, bars_per_chord=1):
    """Flag melody notes that actually fight the chord.

    Only two cases are reported: the flat ninth (a semitone above the *root*,
    the classic "avoid" tone) and the tritone against the root. A 6th, 9th,
    11th or major seventh over the chord is colour; so is a semitone above a
    chord tone that is not the root (the 11th over a major triad), which pop
    melodies use constantly — flagging those makes the check useless.
    """
    ch = progression.chords()
    out = []
    for bar, beat, dur, midi in melody:
        idx = int(bar - 1) // bars_per_chord
        if idx < 0 or idx >= len(ch):
            continue
        if abs(beat - round(beat)) > 1e-6 or int(round(beat)) % 2 != 0:
            continue                      # only strong beats (0 and 2 in 4/4)
        pc = midi % 12
        c = ch[idx]
        tones = set(c.pcs())
        if pc in tones:
            continue
        if (c.root + 1) % 12 == pc:
            out.append((bar, beat, midi, f"b9 {PC_NAME[pc]} over {c.name()}"))
        elif (pc - c.root) % 12 == 6:
            out.append((bar, beat, midi, f"tritone {PC_NAME[pc]} over {c.name()}"))
    return out


def selftest():
    """Assertions for the parts that are easy to get subtly wrong."""
    D = Key("D")
    Bm = Key("B", "minor")
    assert parse("I", D).name() == "D"
    assert parse("vi", D).name() == "Bm"
    assert parse("V7", D).name() == "A7"
    assert parse("ii7", D).name() == "Em7"
    assert parse("ii7b5", D).name() == "Em7b5"
    assert parse("bVI", D).name() == "Bb"
    assert parse("bVII", D).name() == "C"
    assert parse("V/V", D).name() == "E"
    assert parse("V7/V", D).name() == "E7"
    assert parse("ii/V", D).name() == "Bm"
    assert parse("i", Bm).name() == "Bm"
    assert parse("V7", Bm).name() == "F#7"
    assert parse("A/C#", D).bass == 1
    assert D.label(parse("bVI", D)) == "bVI"
    assert D.label(parse("V7", D)) == "V7"
    assert parse("I", D).pcs() == [2, 6, 9]

    prog = ["I", "V", "vi", "IV", "I", "V", "vi", "IV"]
    led = realize(prog, D)
    blk = block(["D", "A", "Bm", "G", "D", "A", "Bm", "G"], D)
    assert motion_total(led) < motion_total(blk), "voice leading must move less"
    assert parallels(led) == 0, f"parallels in led voicing: {parallels(led)}"
    for v in led:
        assert v == sorted(v) and len(set(v)) == len(v), "voicing must ascend, no unisons"
    k2 = Key("G")
    assert parse("I", k2).name() == "G"
    assert parse("V7", k2).name() == "D7"
    return True


if __name__ == "__main__":
    assert selftest()
    print("selftest ok")
    k = Key("D")
    prog = ["Dmaj7", "A/C#", "Bm7", "Gmaj7", "D/F#", "Em7", "Asus4", "A7"]
    led = realize(prog, k)
    blk = block(prog, k)
    print("chord      roman   block voicing      voice-led")
    for name, b, v in zip(prog, blk, led):
        print(f"{name:9s} {k.label(parse(name, k)):6s} {str(b):18s} {v}")
    print(f"motion: block {motion_total(blk)} semitones, led {motion_total(led)}")
    print(f"parallels: block {parallels(blk)}, led {parallels(led)}")
