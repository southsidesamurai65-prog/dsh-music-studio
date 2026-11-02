# Music Studio

Turn “write something that sounds like these songs” into measured parameters and a
reproducible render.

This bundle adds two capabilities to a DSH profile:

- **`music_style_probe`** — analyses a reference audio file (wav/mp3/m4a/flac) and
  returns a style profile: tempo and beat grid, key with alternatives, per-bar chord
  estimate and loop, timber (spectral centroid, band balance), stereo width, reverb
  tail, dynamics and an arrangement arc. The per-bar harmony is an estimate — treat
  it as a hypothesis and prefer a user-supplied chord chart or MIDI when available.
- **`compose-in-style`** — a skill that carries the workflow: probe the references,
  write a six-layer brief, render with the bundled synth library, verify the render
  with the same probe, then iterate one layer at a time.

## Requirements

- `python3` with `numpy`
- `ffmpeg` for decoding anything that is not WAV

## Assets

| Path | Purpose |
|---|---|
| `assets/style_probe.py` | the analysis probe used by the tool |
| `assets/synth_lib.py` | additive-synthesis instruments: guzheng, dizi, xiao, erhu, pipa, yangqin, sheng, rhodes, pad, bass, drums, bells, gong, reverb, mastering |
| `assets/sf2.py` | minimal SoundFont 2 reader/player (samples are not shipped) |
| `assets/mix.py` | stem-wise mixdown: drum bus, multiband compression, IR reverb, limiter, knee clipper |
| `assets/phrase.py` | motif development: transpose, invert, retrograde, augment, sequence, cadence |
| `assets/templates/lofi.py` | full lo-fi example (49 s) |
| `assets/templates/jiangnan.py` | full Jiangnan-style example (109 s) |
| `assets/templates/pop_ballad.py` | pop-ballad starting point (piano/strings/bass/drums) |
| `assets/STYLE_GUIDE.md` | the long-form methodology |

## Example

1. `music_style_probe({ path: "/music/reference.mp3" })`
2. Write a brief from the profile, then a `compose_track.py` that imports
   `synth_lib` (the skill points at its absolute path).
3. Render, then probe the render and compare tempo, key, band balance and dynamics
   against the reference.

## Fingerprint (1.1.0)

`fingerprint` describes *how* notes behave, not which instrument plays them:
attack time per band, decay time, per-band spectral flatness (tonal vs noisy) and
a sustain ratio. Compare a render against its reference on the same numbers —
a commercial master and a synthetic render differ systematically (noise floor,
mastering colour), so read them relatively.

Naming an instrument from a mixed master would need stem separation; the probe
deliberately does not guess.

## Harmony (1.7.1)

`assets/harmony.py` replaces the absolute-pitch chord table with scale degrees
and voice-led voicings: `hm.parse("V7", Key("D"))` is A7, `hm.realize(prog, key)`
voices a whole progression with the least motion consistent with no parallel
fifths or octaves and voices kept in their registers. `V/3` is a first inversion,
`V/V` a secondary dominant, `I/C` a named bass.

Measured on a 16-bar A/B with the same melody: block voicings move 159 semitones
and contain 9 parallel perfects, voice-led ones move 91 with none. The probe
cannot see the difference — it measures the result, not the counterpoint — so
acceptance is by ear plus the score-level `motion_total()` / `parallels()`.

### Functional generation (1.7.2)

`hm.Progression.generate()` fills a phrase by function (tonic / predominant /
dominant, four bars per group), then spends a chromatic budget: applied dominants
(`V7/ii`, `V7/V`), borrowed chords (`bVI`, `bVII`, `iv`) and one tritone
substitution (`bII7`). Insertions must resolve forward, never touch each other,
and leave the first and last bars plain; `validate()` checks that every applied
chord lands on its target.

`check_melody()` reports only a flat ninth over the root and a tritone against
it — a 6th, 9th, 11th or major seventh is colour, and flagging those would make
the check cry wolf.

The probe gained `chords.chroma_novelty` (cosine distance between consecutive bar
chroma): 0.389 for a diatonic loop, 0.464 for the generated progression. Swapping
only the voicings moves it by ±0.01, so it measures *rate of change*, not
richness. A `chromatic_content` metric was tried and removed: it went down when
chromatic chords were added, because it was measuring spectral leakage.

### Pivot modulation (1.7.3)

Each slot now remembers which key it is written in. `modulate(target, at_bar)`
keeps the bars before the change, inserts a chord diatonic in both keys as the
pivot, and regenerates the rest in the new key to its own cadence. Pivots are
ranked by what they mean in the *new* key, so a subdominant or dominant wins over
a chord that turns into a weak mediant (D -> B minor gives Em: ii -> iv).

The probe reports one key for a whole file, so a mid-piece modulation averages
out — in the demo the full render reads B minor because the second half dominates.
Verifying a modulation means splitting the audio at the section boundary.

## Stems and the drum bus (1.5.0)

Rendering into one bus is irreversible: once the kick and the strings are summed,
no compressor can separate them again, which is why multiband compression over a
full mix blunts the backbeat (measured: 2.01 -> 1.63 on the kit's own contrast).

`synth_lib.Stems(seconds)` renders into `drums` / `bass` / `music` / `fx` and
`mix.mixdown()` mixes them: the kit gets glue compression with a slow attack plus a
parallel fast-attack copy, everything else gets the multiband compressor, and the
two meet only at the sum. `fx` is a reverb-send-only stem.

Two defects surfaced during this work and are fixed: the limiter's gain envelope
was smoothed with a centred 90 ms window (it started ducking 45 ms before every
snare hit), and `soft_clip` applied a tanh across the whole signal (2.4 dB of peak
loss). Loudness is now targeted in dBFS and stereo width against a preset target
rather than a fixed side multiplier.

## Ablation benchmark (1.6.0)

`assets/ablate.py` renders one 28 s piece into stems, then mixes it again with a
single dimension changed per pair — drums out, bowed lead swapped for a plucked
one, pad muted, top end filtered, width doubled, reverb drenched, level pushed,
the bus compressed — and asks the probe about every version. Same notes, same
seed, same loudness unless loudness is the variable.

It exists to keep the probe honest: each `must` entry is a documented promise
(backbeat falls when the kit goes, attack time shortens when the bow becomes a
plectrum), and a failure exits non-zero. It also records effect sizes, so a
difference smaller than the measured sensitivity is known to be noise.

It found three false positives on the first run: `backbeat_ratio` follows
spectral balance (a 3 kHz low-pass moved it from 2.29 to 3.30), `onset_rate` goes
*up* when the drums are removed (loud transients raise the adaptive threshold),
and tempo is read at double time on drum-less material. It also caught two real
bugs elsewhere: `make_ir` was 11-14 dB quieter than a measured IR, and the
limiter's centred gain smoothing started ducking 45 ms before every hit.

## Notes

- Everything is synthesized from scratch; no samples are shipped.
- Renders are deterministic when a seed is set (`synth_lib.seed(n)`).
- The bundle records no telemetry and writes only the files the agent creates.
