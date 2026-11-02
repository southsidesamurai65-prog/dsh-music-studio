/**
 * dsh-music-studio — reference-driven music sketching for DSH.
 *
 * Registers:
 *   - tool  `music_style_probe`: runs the bundled numpy MIR probe on an audio
 *     file and returns a compact style profile (tempo, key, chords, timbre,
 *     space, dynamics, arrangement).
 *   - skill `compose-in-style`: the workflow for turning a reference into a
 *     parameterised brief and rendering it with the bundled synth library.
 *
 * Runtime needs: python3 + numpy, and ffmpeg for decoding non-WAV input.
 */
import { spawn } from 'node:child_process'
import { readFileSync, existsSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const assets = join(here, 'assets')

export const name = 'music-studio'
export const inject = ['tools', 'skills']

/**
 * Accept the path forms people actually paste: Windows UNC paths into WSL
 * (`\\wsl.localhost\Ubuntu\home\me\a.wav`, `\\wsl$\...`, `\\?\UNC\...`),
 * Windows drive paths (`C:\Users\me\a.wav`), and quoted paths with spaces.
 * Everything else is passed through untouched.
 */
function normalizePath(input) {
  let p = String(input).trim().replace(/^["']|["']$/g, '')
  if (p.startsWith('~')) p = join(process.env.HOME ?? '~', p.slice(1))
  const unc = p.replace(/^\\\\\?\\UNC\\/i, '\\\\')
  const wsl = unc.match(/^\\\\wsl(?:\.localhost|\$)\\[^\\]+\\(.+)$/i)
  if (wsl) return '/' + wsl[1].replace(/\\/g, '/')
  const drive = unc.match(/^([A-Za-z]):\\(.*)$/)
  if (drive) return `/mnt/${drive[1].toLowerCase()}/${drive[2].replace(/\\/g, '/')}`
  return p
}

/** Turn python stderr into the one line a user can act on. */
function cleanStderr(text) {
  const trimmed = text.trim()
  if (!trimmed) return 'no stderr'
  const lines = trimmed.split('\n').filter(l => l.trim())
  if (lines.length > 0 && /^Traceback \(most recent call last\)/.test(lines[0])) {
    return lines[lines.length - 1]
  }
  return trimmed.slice(0, 800)
}

/** Run a command, resolving to its stdout; rejects with stderr on failure. */
function run(command, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] })
    const out = []
    const err = []
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      reject(new Error(`${command} timed out after ${timeoutMs} ms`))
    }, timeoutMs)
    child.stdout.on('data', chunk => out.push(chunk))
    child.stderr.on('data', chunk => err.push(chunk))
    child.on('error', (error) => {
      clearTimeout(timer)
      reject(new Error(`failed to start ${command}: ${error.message}`))
    })
    child.on('close', (code) => {
      clearTimeout(timer)
      const stdout = Buffer.concat(out).toString('utf8')
      if (code !== 0) {
        const stderr = cleanStderr(Buffer.concat(err).toString('utf8'))
        reject(new Error(`${command} exited with ${code}: ${stderr}`))
        return
      }
      resolve(stdout)
    })
  })
}

/** One line on texture: attacks, pulse, backbeat — hears a rhythm section, not a score. */
function fingerprintLine(f) {
  if (!f) return 'not measured'
  const atk = f.attack_ms ?? {}
  const dec = f.decay_ms ?? {}
  const flat = f.flatness ?? {}
  const fmt = o => ['low', 'mid', 'high'].map(k => (o[k] == null ? '?' : o[k])).join('/')
  return `attack ${fmt(atk)} ms, decay ${fmt(dec)} ms, noise (flatness) ${fmt(flat)},`
    + ` sustain ${f.sustain_ratio}`
}

function arrangementLine(a) {
  if (!a) return 'not measured'
  const bb = a.backbeat_ratio == null ? '' : `, backbeat ${a.backbeat_ratio}`
  return `${a.texture}, onsets ${a.onset_rate_per_s}/s, low-band pulse ${a.drums}`
    + ` (mod ${a.low_pulse_mod}, periodicity ${a.low_pulse_periodicity})${bb}`
    + `, presence ${a.presence_db} dB`
}

function summarize(profile) {
  const chords = profile.chords ?? {}
  const loop = Array.isArray(chords.loop) && chords.loop.length > 0
    ? `${chords.loop.join('–')} (${(chords.roman ?? []).join(' ')})`
    : 'not detected'
  const balance = profile.timbre?.band_balance_db ?? {}
  const bands = ['20-60', '250-500', '500-1000', '4000-8000']
    .map(b => `${b}:${balance[b] ?? '?'}`)
    .join(' ')
  return [
    `file: ${profile.file}`,
    `duration: ${profile.duration_s}s`,
    `tempo: ${profile.tempo?.bpm} BPM (${profile.tempo?.meter}); candidates ${(profile.tempo?.candidates ?? []).map(c => `${c.bpm}`).join(', ')}`,
    `key: ${profile.key?.name} (score ${profile.key?.score}; alternatives ${(profile.key?.alternatives ?? []).map(k => k.name).join(', ') || 'none'})`,
    `chord loop: ${loop}${chords.loop_bars ? `, ${chords.loop_bars} bars` : ''}`,
    `per bar: ${(chords.per_bar ?? []).slice(0, 16).join(' ')}${(chords.per_bar ?? []).length > 16 ? ' …' : ''}`,
    `timbre: centroid ${profile.timbre?.centroid_hz} Hz (${profile.timbre?.character}); ${bands}`,
    `space: width ${profile.space?.stereo_width}; tail ${profile.space?.tail_150ms_ratio}`,
    `dynamics: rms ${profile.dynamics?.rms_dbfs} dBFS, crest ${profile.dynamics?.crest_db} dB`,
    `arc (8 slices, dB): ${(profile.dynamics?.arc_8ths_db ?? []).join(' ')}`,
    `arrangement: ${arrangementLine(profile.arrangement)}`,
    `texture: ${fingerprintLine(profile.fingerprint)}`,
  ].join('\n')
}

export function apply(ctx) {
  ctx.tools.register({
    name: 'music_style_probe',
    description: [
      'Analyse a reference audio file into a music style profile: tempo and beat grid,',
      'key (with alternatives), per-bar chord estimate and loop, timbre (spectral centroid',
      'and band balance), stereo width and reverb tail, dynamics, an arrangement arc, and',
      'Also reports arrangement hints (onset rate, low-band pulse, backbeat ratio, presence)',
      'and texture hints (attack/decay time, per-band noise, sustain ratio) to compare a render',
      'check the match. Requires python3 with numpy; ffmpeg decodes non-WAV input.',
    ].join(' '),
    parameters: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Audio file path (wav, mp3, m4a, flac, ...). Windows UNC paths from WSL '
            + '(\\\\wsl.localhost\\Distro\\home\\...) and drive paths (C:\\...) are converted automatically.',
        },
        full: { type: 'boolean', description: 'Return the complete profile JSON instead of a summary.' },
      },
      required: ['path'],
      additionalProperties: false,
    },
    output: {
      schema: { type: 'string' },
      render: (_args, value) => [{ type: 'text', text: value }],
    },
    timeoutMs: 180000,
    async execute(args) {
      const raw = String(args.path)
      const file = normalizePath(raw)
      if (!existsSync(file)) {
        const hint = file !== raw
          ? ` (converted from "${raw}" — that is a Windows path, this agent runs in Linux)`
          : ''
        throw new Error(`audio file not found: ${file}${hint}`)
      }
      const stdout = await run('python3', [join(assets, 'style_probe.py'), file, '--json', '-'], 170000)
      let profile
      try {
        profile = JSON.parse(stdout)
      } catch {
        throw new Error(`style_probe returned non-JSON output: ${stdout.slice(0, 400)}`)
      }
      return args.full === true
        ? JSON.stringify(profile, null, 2)
        : summarize(profile)
    },
  })

  const skill = readFileSync(join(assets, 'skill.md'), 'utf8').replaceAll('{{ASSETS}}', assets)
  ctx.skills.register({
    name: 'compose-in-style',
    description: 'Compose original music in the style of reference tracks: probe the references, write a layered brief, render with the bundled numpy synth library, then verify and iterate.',
    whenToUse: 'The user asks for music in a described style, or similar in feel to specific songs, or wants an existing sketch reworked in a direction (brighter, sparser, more folk, more like X).',
    // required by the skill registry: without a source string the skill lists but
    // fails to load ("loaded skill ... source must be a string")
    source: 'runtime',
    resourceBase: { kind: 'directory', path: assets },
    content: skill,
  })
}
