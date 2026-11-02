#!/usr/bin/env python3
"""
sf2.py — a minimal SoundFont 2 reader and player, numpy only.

Why: an additive synth can carry an arrangement, but it cannot fake a real
piano or a string section. This module plays .sf2 sample banks so the library
can render with real recorded instruments.

Scope (deliberate): mono output, one matching zone per note, linear
interpolation, volume envelope + loop points, coarse/fine tuning, pan and
attenuation. It ignores filters, LFOs, modulators and stereo link — enough for
sketching, not a full synth engine.

    sf = SoundFont("GeneralUser-GS.sf2")
    y = sf.note(program=0, key=60, vel=0.8, dur=2.0)   # acoustic grand
"""
import struct

import numpy as np

SF2_HEADER = 12

# generator operators we honour
GEN_KEYRANGE = 43
GEN_VELRANGE = 44
GEN_START_OFF = 0
GEN_END_OFF = 1
GEN_STARTLOOP_OFF = 2
GEN_ENDLOOP_OFF = 3
GEN_START_COARSE = 4
GEN_END_COARSE = 12
GEN_PAN = 17
GEN_ATTACK = 34
GEN_HOLD = 35
GEN_DECAY = 36
GEN_SUSTAIN = 37
GEN_RELEASE = 38
GEN_INITIAL_ATT = 48
GEN_FINE_TUNE = 52
GEN_COARSE_TUNE = 51
GEN_SAMPLE_ID = 53
GEN_SAMPLE_MODES = 54
GEN_ROOT_KEY = 58

TIME_GENS = (GEN_ATTACK, GEN_HOLD, GEN_DECAY, GEN_RELEASE)


def _chunks(buf, start, end):
    pos = start
    while pos + 8 <= end:
        cid = buf[pos:pos + 4]
        size = struct.unpack_from("<I", buf, pos + 4)[0]
        data = pos + 8
        yield cid, data, size
        pos = data + size + (size & 1)


class SoundFont:
    """Parse the core of an .sf2 file and play single notes from it."""

    def __init__(self, path):
        self.path = str(path)
        with open(self.path, "rb") as fh:
            self.raw = fh.read()
        if self.raw[:4] != b"RIFF" or self.raw[8:12] not in (b"sfbk", b"sf2"):
            raise ValueError(f"{path}: not a SoundFont 2 file")
        self._parse()

    # ------------------------------------------------------------------ parse
    def _parse(self):
        raw = self.raw
        self.samples = None
        self.phdr = self.pbag = self.pmod = self.pgen = None
        self.inst = self.ibag = self.imod = self.igen = self.shdr = None
        for cid, data, size in _chunks(raw, SF2_HEADER, len(raw)):
            if cid != b"LIST":
                continue
            kind = raw[data:data + 4]
            body = data + 4
            for sid, sdata, ssize in _chunks(raw, body, data + size):
                if kind == b"sdta" and sid == b"smpl":
                    self.samples = np.frombuffer(raw, dtype="<i2", count=ssize // 2, offset=sdata)
                elif kind == b"pdta":
                    setattr(self, sid.decode("ascii"), (sdata, ssize))
        if self.samples is None or self.shdr is None:
            raise ValueError(f"{self.path}: missing pdta/sdta sections")
        self._parse_headers()

    def _parse_headers(self):
        def table(name, fmt, count):
            off, size = getattr(self, name)
            return np.frombuffer(self.raw, dtype=fmt, count=count, offset=off)

        phdr_off, phdr_size = self.phdr
        self.phdr = table("phdr", np.dtype([
            ("name", "S20"), ("preset", "<u2"), ("bank", "<u2"), ("bag", "<u2"),
            ("library", "<u4"), ("genre", "<u4"), ("morph", "<u4")]), phdr_size // 38)
        self.pbag = table("pbag", np.dtype([("gen", "<u2"), ("mod", "<u2")]), self.pbag[1] // 4)
        self.pgen = table("pgen", np.dtype([("op", "<u2"), ("amount", "<h")]), self.pgen[1] // 4)
        self.inst = table("inst", np.dtype([("name", "S20"), ("bag", "<u2")]), self.inst[1] // 22)
        self.ibag = table("ibag", np.dtype([("gen", "<u2"), ("mod", "<u2")]), self.ibag[1] // 4)
        self.igen = table("igen", np.dtype([("op", "<u2"), ("amount", "<h")]), self.igen[1] // 4)
        self.shdr = table("shdr", np.dtype([
            ("name", "S20"), ("start", "<u4"), ("end", "<u4"), ("lstart", "<u4"),
            ("lend", "<u4"), ("rate", "<u4"), ("root", "u1"), ("corr", "i1"),
            ("link", "<u2"), ("type", "<u2")]), self.shdr[1] // 46)

    # ------------------------------------------------------------- generators
    @staticmethod
    def _gen_list(gens, start, end):
        out = {}
        for k in range(start, end):
            out[int(gens[k]["op"])] = int(gens[k]["amount"])
        return out

    # ------------------------------------------------------------------ notes
    def presets(self):
        return [(int(p["bank"]), int(p["preset"]), p["name"].split(b"\x00")[0].decode("latin-1"))
                for p in self.phdr[:-1]]

    def _instrument_zones(self, inst_index):
        start = int(self.inst[inst_index]["bag"])
        end = int(self.inst[inst_index + 1]["bag"])
        zones = []
        for b in range(start, end):
            g0 = int(self.ibag[b]["gen"])
            g1 = int(self.ibag[b + 1]["gen"])
            zones.append(self._gen_list(self.igen, g0, g1))
        return zones

    def note(self, program, key, vel=0.8, dur=1.0, bank=0, sr=44100, gain=1.0):
        """Render one note. `key` may be fractional (in semitones)."""
        key_int = int(round(key))
        preset_idx = self._find_preset(program, bank)
        if preset_idx is None:
            return np.zeros(int(0.01 * sr))
        zone = self._find_zone(preset_idx, key_int, vel)
        if zone is None:
            return np.zeros(int(0.01 * sr))
        return self._render(zone, key, vel, dur, sr, gain)

    def _find_preset(self, program, bank):
        best = None
        for i, p in enumerate(self.phdr[:-1]):
            if int(p["preset"]) == program and int(p["bank"]) == bank:
                return i
            if int(p["preset"]) == program and best is None:
                best = i
        return best

    def _find_zone(self, preset_idx, key, vel):
        pstart = int(self.phdr[preset_idx]["bag"])
        pend = int(self.phdr[preset_idx + 1]["bag"])
        for b in range(pstart, pend):
            g0 = int(self.pbag[b]["gen"])
            g1 = int(self.pbag[b + 1]["gen"])
            pgens = self._gen_list(self.pgen, g0, g1)
            inst_idx = pgens.get(41)          # instrument generator
            if inst_idx is None or inst_idx >= len(self.inst) - 1:
                continue
            global_zone = {}
            for z in self._instrument_zones(inst_idx):
                if GEN_SAMPLE_ID not in z:
                    global_zone.update(z)      # instrument-wide defaults
                    continue
                lo, hi = self._range(z, GEN_KEYRANGE, 0, 127)
                if not (lo <= key <= hi):
                    continue
                vlo, vhi = self._range(z, GEN_VELRANGE, 0, 127)
                if not (vlo <= vel * 127 <= vhi):
                    continue
                merged = dict(global_zone)
                merged.update(z)
                # preset-zone generators offset the instrument ones; envelope
                # times are in timecents, so summing them multiplies the time
                for k, v in pgens.items():
                    if k in TIME_GENS:
                        merged[k] = int(np.clip(merged.get(k, -12000) + v, -32768, 32767))
                    elif k in (GEN_INITIAL_ATT, GEN_PAN):
                        merged[k] = merged.get(k, 0) + v
                return merged
        return None

    @staticmethod
    def _range(zone, op, dlo, dhi):
        # generator ranges are stored as (lo, hi) on the same operator: the
        # spec packs them into one 16-bit word; parsers commonly expose lo/hi
        # as consecutive entries, so accept either shape.
        if op in zone:
            v = zone[op] & 0xFFFF
            return (v & 0xFF, (v >> 8) & 0xFF)
        lo = zone.get(op)
        hi = zone.get(op + 1)
        if lo is None or hi is None:
            return dlo, dhi
        return lo, hi

    def _render(self, zone, key, vel, dur, sr, gain):
        sid = zone.get(GEN_SAMPLE_ID, 0)
        if sid >= len(self.shdr) - 1:
            return np.zeros(int(0.01 * sr))
        h = self.shdr[sid]
        start = int(h["start"]) + zone.get(GEN_START_OFF, 0) + 32768 * zone.get(GEN_START_COARSE, 0)
        end = int(h["end"]) + zone.get(GEN_END_OFF, 0) + 32768 * zone.get(GEN_END_COARSE, 0)
        lstart = int(h["lstart"]) + zone.get(GEN_STARTLOOP_OFF, 0)
        lend = int(h["lend"]) + zone.get(GEN_ENDLOOP_OFF, 0)
        start = max(0, min(start, len(self.samples) - 2))
        end = max(start + 2, min(end, len(self.samples)))
        lstart = max(start, min(lstart, end - 2))
        lend = max(lstart + 2, min(lend, end))

        root = zone.get(GEN_ROOT_KEY, int(h["root"]))
        cents = (key - root) * 100 + zone.get(GEN_COARSE_TUNE, 0) * 100 \
            + zone.get(GEN_FINE_TUNE, 0) + int(h["corr"])
        ratio = 2 ** (cents / 1200.0) * (int(h["rate"]) / sr)

        attack = self._time(zone.get(GEN_ATTACK, -12000))
        hold = self._time(zone.get(GEN_HOLD, -12000))
        decay = self._time(zone.get(GEN_DECAY, -12000))
        release = self._time(zone.get(GEN_RELEASE, -12000))
        sustain = 10 ** (-(zone.get(GEN_SUSTAIN, 0) / 10.0) / 20.0)   # 0.1 dB units
        att = 10 ** (-(zone.get(GEN_INITIAL_ATT, 0) / 10.0) / 20.0)

        total = dur + release + 0.05
        n = int(total * sr)
        pos = start + np.arange(n) * ratio
        loop_mode = zone.get(GEN_SAMPLE_MODES, 0)
        if loop_mode in (1, 3) and lend > lstart:
            over = pos >= lend
            if over.any():
                pos[over] = lstart + np.mod(pos[over] - lstart, lend - lstart)
            idx = pos - start
            sig = np.interp(idx, np.arange(end - start),
                            self.samples[start:end].astype(np.float64))
        else:
            live = pos < end
            idx = np.clip(pos - start, 0, end - start - 1)
            sig = np.interp(idx, np.arange(end - start),
                            self.samples[start:end].astype(np.float64)) * live

        sig /= 32768.0
        sig *= self._envelope(n, sr, attack, hold, decay, sustain, dur, release)
        sig *= att * (vel ** 1.3) * gain
        return sig

    @staticmethod
    def _time(tc):
        return 0.0 if tc <= -32700 else 2 ** (tc / 1200.0)

    @staticmethod
    def _envelope(n, sr, attack, hold, decay, sustain, dur, release):
        t = np.arange(n) / sr
        env = np.ones(n)
        if attack > 1e-4:
            env *= np.clip(t / attack, 0.0, 1.0)
        td = t - (attack + hold)
        if decay > 1e-4:
            dec = sustain + (1 - sustain) * np.exp(-np.maximum(td, 0) / decay)
            env *= np.where(td > 0, dec, 1.0)
        else:
            env *= np.where(td > 0, sustain, 1.0)
        rel = np.clip((t - dur) / max(release, 1e-3), 0.0, 1.0)
        env *= (1.0 - rel)
        return env
