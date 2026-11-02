# dsh-music-studio

A DSH bundle for reference-driven music sketching.

- tool `music_style_probe` — audio → style profile (tempo, key, chords, timbre, space, dynamics, structure)
- skill `compose-in-style` — the workflow and the layered brief template
- `assets/synth_lib.py` — numpy instruments (Chinese ensemble + lo-fi kit), stems, mixing, mastering
- `assets/templates/` — three starting points (lo-fi, 江南, pop ballad)
- `assets/sf2.py` — SoundFont 2 player for real sampled instruments (bring your own bank)
- `assets/mix.py` — stem-wise mixdown: drum bus, multiband compression, convolution
  reverb, width and loudness targets, limiter and knee clipper
- `assets/phrase.py` — motif development and phrase building
- `assets/harmony.py` — chords as scale degrees, with voice-led voicings
- `assets/ablate.py` — paired ablations: does each probe metric move when its cause moves?

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

- 1.7.5 — a pipa that does not sound like a piano. `pipa_pluck` had borrowed the
  piano's recipe: eight partials on smooth long decays with a quiet click. On C4,
  92.6% of its energy sat below 1 kHz and it rang for 1.28 s, which is exactly why
  a listener heard a piano. Rewritten around what actually identifies a pipa — a
  hard nail click, upper partials gone within a tenth of a second, two detuned
  courses, body resonances at 1.6 and 3.3 kHz — it now measures 1.24 kHz attack
  centroid against the piano's 0.34 kHz, and decays to -20 dB in 0.23 s instead of
  1.28 s. `pipa_tremolo` inherits the fix. Renders made before this will change if
  regenerated.
- 1.7.4 — packaging. The tarball had grown to 9 MB because running
  `templates/pop_ballad.py` writes a demo wav next to itself and `files: ["assets"]`
  swept it in. `files` now excludes rendered audio and images, so the package is
  80 KB again; the 1.7.1–1.7.3 tarballs in `dist/` were deleted (their tags on
  GitHub are source-only, so nothing is lost).
- 1.7.3 — harmony, part three: pivot modulation. Slots now remember their key, so
  `Progression.modulate(target, at_bar)` can keep the bars before the change,
  insert a chord diatonic in both keys as a pivot, and regenerate the rest in the
  new key to its own cadence. Pivots are ranked by what they mean in the *new*
  key, preferring a subdominant or dominant over one that turns into a weak
  mediant (D→B minor chooses Em: ii→iv). The demo caught a real melody clash at
  the seam (G over F#, a flat ninth) which `check_melody` flagged before anyone
  heard it. The probe reports one key per file, so a modulation averages out in a
  whole-track profile; split the audio to verify it.
- 1.7.2 — harmony, part two: function and chromaticism. `hm.Progression.generate()`
  fills a phrase by function (tonic / predominant / dominant) and then spends a
  chromatic budget on applied dominants (`V7/ii`, `V7/V`), borrowed chords
  (`bVI`, `bVII`, `iv`) and one tritone substitution (`bII7`), with every
  insertion resolving forward and a budget per phrase so the piece does not turn
  into secondary dominants. `validate()` checks the resolutions, `check_melody()`
  flags only a flat ninth over the root and a tritone against it — a 6th, 9th or
  11th is colour. The probe gained `chroma_novelty` / `chroma_repeat_rate`
  (diatonic loop 0.389, generated progression 0.464; voicing swaps move it ±0.01,
  so it measures change, not richness). A `chromatic_content` metric was tried
  and removed: it fell when chromatic chords were added, because it measured
  spectral leakage.
- 1.7.1 — harmony, part one: degrees and voice leading. `assets/harmony.py` stops
  storing chords as absolute pitch blocks. A chord is a degree plus a quality
  (`I`, `vi`, `V7`, `ii7b5`, `bVI`, `#IVdim7`, `V/3` for a first inversion, `V/V`
  for a secondary dominant), and `realize()` voices a progression with the least
  motion that avoids parallel fifths and octaves and keeps each voice in its
  register. Measured on a 16-bar A/B with the same melody: block voicings move
  159 semitones with 9 parallel perfects, voice-led ones 91 with none. The probe
  cannot see this, so the acceptance test is `motion_total()` / `parallels()`
  plus the ear. The pop-ballad template now writes its progression in roman
  numerals and uses the voicings.
- 1.6.0 — an ablation benchmark, and two bugs it caught. `assets/ablate.py` renders
  one 28 s piece into stems, then re-mixes it with exactly one dimension changed per
  pair (drums out, bow→plectrum, pad muted, 3 kHz low-pass, width ×2.2, reverb
  0.25→0.60, −16→−12 dBFS, whole-bus compression) and compares what the probe says.
  Required movements must hold or the run exits non-zero, so a probe change is now a
  regression. Measured sensitivity: backbeat 2.29→1.18 without drums, 2.29→1.60
  under bus compression; mid attack 34.8→29.0 ms bow→pluck; centroid 728→489 Hz
  through a 3 kHz low-pass; width 0.21→0.46 at ×2.2. Three false positives came out
  of it and are documented: `backbeat_ratio` follows spectral balance (a 3 kHz
  low-pass pushes it to 3.30), `onset_rate` *rises* when the drums are removed
  (loud transients raise the adaptive threshold), and drum-less material reads at
  double tempo. The benchmark also caught two real bugs: `make_ir` was 11–14 dB
  quieter than a measured IR (every synthetic-reverb mix was far drier than its
  `wet` value), and the width target was silently capped at 6× so narrow material
  missed the preset target.
- 1.5.0 — stems and a drum bus. `synth_lib.Stems` renders into named stems
  (`drums`/`bass`/`music`/`fx`) instead of one bus, and `mix.mixdown()` mixes them:
  drums get glue compression (slow attack) plus parallel compression, everything
  else gets the multiband compressor, and only then do the two meet. Two bugs fell
  out of it: the limiter's gain envelope was smoothed with a *centred* 90 ms window
  that started ducking 45 ms before every snare hit, and `soft_clip` applied a tanh
  across the whole signal (2.4 dB of peak loss). Fixing both — forward-looking
  look-ahead, make-up gain, and a proper knee clipper — is where the extra punch
  comes from. Measured on a 177 s piece: backbeat ratio 1.68 -> 1.92 (the kit's own
  contrast is 2.01, so retention went from 84% to 96%) at the same crest. Stereo
  width is now solved against a preset target instead of a fixed side multiplier,
  and loudness is targeted in dBFS rather than guessed.
- 1.4.0 — phrasing. `assets/phrase.py`: motif operations (transpose, invert,
  retrograde, augment/diminish, fragment, sequence, ornament, cadence,
  humanise), scale-degree rendering and a rough contour-repetition score. The
  point is that a motif is not a melody: the example piece states one motif and
  then sequences, mirrors and stretches it.
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
