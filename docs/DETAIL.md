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

## Notes

- Everything is synthesized from scratch; no samples are shipped.
- Renders are deterministic when a seed is set (`synth_lib.seed(n)`).
- The bundle records no telemetry and writes only the files the agent creates.
