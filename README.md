# dsh-music-studio

A DSH bundle for reference-driven music sketching.

- tool `music_style_probe` — audio → style profile (tempo, key, chords, timbre, space, dynamics, structure)
- skill `compose-in-style` — the workflow and the layered brief template
- `assets/synth_lib.py` — numpy instruments (Chinese ensemble + lo-fi kit), reverb, mastering
- `assets/templates/` — three starting points (lo-fi, 江南, pop ballad)
- `assets/sf2.py` — SoundFont 2 player for real sampled instruments (bring your own bank)
- `assets/mix.py` — convolution reverb, multiband compression, limiter, soft clip

Install into a profile from the directory that contains this package:

```sh
dsh plugin --profile <name> add ./dsh-music-studio
```

Requirements: python3 + numpy, ffmpeg.

Path handling: the probe runs on Linux, so Windows paths are translated for you —
`\\wsl.localhost\<distro>\home\...` → `/home/...`, `C:\...` → `/mnt/c/...`.
If the file still cannot be found, the tool reports the original and the translated
path instead of a Python traceback.

## Changelog

- 1.3.0 — mixing bus. `assets/mix.py`: real-IR convolution reverb (predelay +
  high-pass), zero-phase multiband compression with soft knee and makeup gain,
  a look-ahead peak limiter, and a soft clipper. `MIX_PRESETS` gained the bus
  parameters and `apply_bus()` chains them. Everything is numpy; 180 s of audio
  processes in about 5 s.
- 1.2.0 — sampled instruments. `assets/sf2.py` is a numpy-only SoundFont 2 reader
  (zones, velocity layers, loop points, volume envelope, tuning, attenuation, pan)
  and `synth_lib.use_soundfont()` / `synth_lib.sf_play()` expose it with the same
  call signature as the synth instruments. No sample data is shipped: users point
  `SF2` at their own bank and are responsible for its licence.
- 1.1.0 — `fingerprint` block: attack/decay time per band, per-band spectral
  flatness and a sustain ratio, so a render and its reference can be compared on
  *how notes start and ring*, not just on the overall spectrum. It describes
  behaviour, not instruments: naming a piano from a mixed master would need stem
  separation, which this probe deliberately does not fake.
- 1.0.0 — first release that is meant to be used. The 0.x line was a series of
  pre-releases: 0.1.x could not load the skill or read Windows paths, 0.2.x had
  a library that only spoke Chinese-ensemble, and 0.3.x added the arrangement
  hints, palettes, mix presets and the phase-invariant backbeat metric. Nothing
  changed in the code between 0.3.1 and 1.0.0 — this is a statement about the
  state of the tool, not a feature drop.
- 0.3.1 — `backbeat_ratio` is now phase-invariant: the beat tracker can lock onto
  the backbeat itself, which reported a real 2&4 accent as 0.5. The probe now
  takes the strongest of the four beat rotations, so the number means "does the
  pattern alternate accented beats", not "which beat is beat 1".
- 0.3.0 — anti-homogenisation release. The probe reports arrangement hints
  (onset rate, low-band pulse, backbeat, presence) and the skill now requires an
  orchestration checklist with a diversity self-check. `synth_lib` gained
  `violin`, `cello`, `choir`, `clap`, a role registry (`sl.palette()`) and four
  mix presets (`MIX_PRESETS`). New `templates/pop_ballad.py` gives a non-folk
  starting point.
- 0.2.0 — new pop-section instruments in `synth_lib`: `piano`, `strings`, `ebass`,
  `snare`, `shaker` (the library was Chinese-ensemble-only, which made every
  arrangement sound like the same chamber piece).
- 0.1.2 — the skill now registers with the `source` the skill registry requires;
  before this, `compose-in-style` appeared in the catalog but failed to load with
  "loaded skill ... source must be a string".
- 0.1.1 — accept Windows/WSL paths in `music_style_probe`; readable errors on
  missing or undecodable files; probe no longer emits a traceback for bad input;
  tempo candidates in the summary.
- 0.1.0 — first release: probe tool, compose-in-style skill, synth library, templates.
