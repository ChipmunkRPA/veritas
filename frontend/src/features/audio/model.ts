/** The audio exhibit model (UX §5.10 lane 4, PRD DC-05 / EA-05; BUILD_SPEC A-1): the voice lane from the temporal stage's series and
 *  segments exhibits, with music, noise and silence hatched with their labels from the content table, and the examiner's interval
 *  notes. It fills the video lane model, so the timeline, the segment list and the table view read the same way for both kinds.
 *  - Lane scores are smoothed window outputs; headline calibration does not calibrate lanes or segment thresholds.
 *  - Recordings carry no frames: segment frame indices are 0. */
import { frameStep } from '../../lib/api'
import type { Annotation, ExhibitPayload, TableExhibit } from '../../lib/api/queries'
import { ptsSeconds } from '../../lib/pts'
import { noteNumbers } from '../review/review'
import { laneName, letterFor, windowSpans, type Gap, type LaneModel, type NoteItem, type SeriesExhibit, type SegmentsExhibit, type VideoModel, type VideoSegment, type WaveBin } from '../video/model'

export type ContentLabel = 'speech' | 'music' | 'noise' | 'silence'

/** The v1 view's `audio` block (`engine/stages/audio.py::fuse`). */
export interface AudioFacts {
  container?: string | null
  codec?: string | null
  sample_rate?: number | null
  channels?: number | null
  bit_rate?: number | null
  duration_s?: number | null
  lossless?: boolean
  seconds_analysed?: number
  shares?: Partial<Record<ContentLabel, number>>
  assessable?: boolean
  not_assessable_reason?: string | null
}

export interface ContentRun { start: number; end: number; label: string; scored: boolean }

const CONTENT_ORDER: readonly ContentLabel[] = ['speech', 'music', 'noise', 'silence']
const GAP_LABEL: Record<string, string> = { music: 'music, no speech', noise: 'noise, no speech', silence: 'silence', speech: 'speech, not scored' }
const NOT_ANALYSED = 'not analysed'

/** "62 % music, 20 % speech, 18 % silence": largest share first, shares that round to 0 % left out (`detectors/audio/content.py::describe`). */
export function describeShares(shares: AudioFacts['shares']): string {
  return CONTENT_ORDER.map((label) => [label, shares?.[label] ?? 0] as const)
    .filter(([, v]) => Math.round(v * 100) > 0)
    .sort((a, b) => b[1] - a[1] || CONTENT_ORDER.indexOf(a[0]) - CONTENT_ORDER.indexOf(b[0]))
    .map(([label, v]) => `${Math.round(v * 100)} % ${label}`)
    .join(', ')
}

/** The content runs of the temporal stage's table exhibit (columns start_s, end_s, content, voice detectors). */
export function contentRuns(payloads: readonly ExhibitPayload[]): ContentRun[] {
  const table = payloads.find((p): p is TableExhibit => p.kind === 'table' && p.media_ref === 'audio' && p.columns.includes('content'))
  if (!table) return []
  const col = (name: string) => table.columns.indexOf(name)
  const [s, e, c, d] = [col('start_s'), col('end_s'), col('content'), col('voice detectors')]
  return table.rows
    .map((row) => ({ start: Number(row[s]), end: Number(row[e]), label: String(row[c] ?? ''), scored: d >= 0 && row[d] === 'scored' }))
    .filter((r) => Number.isFinite(r.start) && Number.isFinite(r.end) && r.end > r.start)
}

/** The waveform envelope of the temporal stage's series exhibit (`detectors/audio/waveform.py`): the lowest and highest sample and
 *  the RMS level per bin. Empty for a result recorded before the exhibit existed. */
export function waveBins(payloads: readonly ExhibitPayload[]): { bins: WaveBin[]; binS: number } {
  const exhibit = payloads.find((p): p is SeriesExhibit => p.kind === 'series' && p.media_ref === 'audio' && p.series.some((s) => s.name === 'level_rms'))
  const points = (name: string) => exhibit?.series.find((s) => s.name === name)?.points ?? []
  const [low, high, rms] = [points('level_low'), points('level_high'), points('level_rms')]
  const bins = rms.flatMap((r, i): WaveBin[] => {
    const [l, h] = [low[i]?.[1], high[i]?.[1]]
    return r[1] == null || l == null || h == null ? [] : [{ t: Number(r[0]), low: Number(l), high: Number(h), rms: Number(r[1]) }]
  })
  return { bins, binS: bins.length > 1 ? bins[1].t - bins[0].t : 0.01 }
}

/** Parts of [from, to] outside every interval, as "not analysed" gaps (at least a millisecond wide). */
function uncovered(intervals: readonly { start: number; end: number }[], from: number, to: number): Gap[] {
  const out: Gap[] = []
  let at = from
  for (const x of [...intervals].sort((a, b) => a.start - b.start)) {
    if (x.start > at + 1e-3) out.push({ start: at, end: Math.min(x.start, to), label: NOT_ANALYSED })
    at = Math.max(at, x.end)
    if (at >= to) break
  }
  if (to > at + 1e-3) out.push({ start: at, end: to, label: NOT_ANALYSED })
  return out
}

export function buildAudioModel({ facts, payloads, annotations, spectrogramUrl = null }: {
  facts: AudioFacts | null | undefined
  payloads: readonly ExhibitPayload[]
  annotations: readonly Annotation[]
  /** The spectrogram picture of the temporal stage, with its access token; null when the result has none or the token is not ready. */
  spectrogramUrl?: string | null
}): VideoModel & { content: ContentRun[] } {
  const content = contentRuns(payloads)
  const voiceSeries = payloads.filter((p): p is SeriesExhibit => p.kind === 'series' && p.media_ref === 'audio' && p.series.some((s) => s.name === 'p_smoothed'))
  const points = voiceSeries.flatMap((p) => (p.series.find((s) => s.name === 'p_smoothed')?.points ?? []).map((x) => [Number(x[0]), x[1] == null ? null : Number(x[1])] as [number, number | null]))
  const wave = waveBins(payloads)
  const pictured = wave.bins.length ? wave.bins[wave.bins.length - 1].t + wave.binS / 2 : 0  // seconds of signal the pictures span
  const duration = Math.max(facts?.duration_s ?? 0, ...content.map((r) => r.end), ...points.map((x) => x[0]), pictured, 0)

  const segments: VideoSegment[] = payloads
    .filter((p): p is SegmentsExhibit => p.kind === 'segments' && p.media_ref === 'audio')
    .flatMap((p) => p.segments.map((s, i) => {
      const lane = s.lane ?? p.lane
      return {
        key: `${p.id}:${i}`, letter: '', lane, laneName: laneName(lane), trackId: null, trackName: null,
        start: ptsSeconds(s.start_pts, s.time_base), end: ptsSeconds(s.end_pts, s.time_base), startFrame: 0, endFrame: 0,
        peak: s.peak, step: frameStep(s.peak), exhibitId: p.id, hypothesis: s.hypothesis, sources: s.sources ?? [], displayId: s.display_id ?? null,
      }
    }))
    .sort((a, b) => a.start - b.start || a.end - b.end)
    .map(({ displayId, ...seg }, n) => ({ ...seg, letter: displayId ? displayId.replace(/^Seg\.\s*/, '') : letterFor(n) }))

  const base = (key: string, kind: LaneModel['kind'], label: string, detail: string): LaneModel =>
    ({ key, kind, label, detail, trackId: null, state: 'ready', spans: [], gaps: [], segments: [] })

  // The voice lane: window cells where the detectors scored; non-speech runs hatched with their label; the rest not analysed.
  const voice = base('voice', 'voice', 'Audio', 'synthetic-speech check · p uncalibrated')
  const labelled: Gap[] = content.filter((r) => r.label !== 'speech' || !r.scored).map((r) => ({ start: r.start, end: r.end, label: GAP_LABEL[r.label] ?? r.label }))
  if (points.length) {
    const { spans } = windowSpans(points, duration)
    Object.assign(voice, { spans, gaps: [...labelled, ...uncovered([...spans, ...labelled], 0, duration)], segments })
  } else if (content.length) {
    Object.assign(voice, { gaps: [...labelled, ...uncovered(labelled, 0, duration)] })
  } else {
    Object.assign(voice, { state: 'not_run', reason: facts?.not_assessable_reason ?? 'no voice detector produced windows' })
  }

  const numbers = noteNumbers(annotations)
  const notes: NoteItem[] = annotations
    .filter((a) => a.target_type === 'interval' && !a.deleted_at && a.geometry)
    .map((a, i): NoteItem | null => {
      const g = a.geometry as { start_pts?: number; end_pts?: number; time_base?: string }
      const ok = typeof g.start_pts === 'number' && typeof g.end_pts === 'number' && typeof g.time_base === 'string'
      return ok ? { id: a.id, label: numbers.get(a.id) ?? `N${i + 1}`, start: ptsSeconds(g.start_pts as number, g.time_base as string), end: ptsSeconds(g.end_pts as number, g.time_base as string), body: a.body ?? '', visibility: a.visibility } : null
    })
    .filter((n): n is NoteItem => n != null)

  // The signal itself under the voice lane (L-7): what the recording looks like, with the content runs named. Not detector outputs.
  const pictures: LaneModel[] = []
  if (wave.bins.length) {
    pictures.push({ ...base('waveform', 'waveform', 'Waveform', 'peak and RMS level'), explain: 'not a detector output', wave: wave.bins, binS: wave.binS,
      runs: content.map((r) => ({ start: r.start, end: r.end, label: r.label })) })
    if (spectrogramUrl) pictures.push({ ...base('spectrogram', 'spectrogram', 'Spectrogram', '0–8 kHz, loudness in dB'), explain: 'brighter is louder, down to 80 dB below the loudest cell; not a detector output', imageUrl: spectrogramUrl, imageSeconds: pictured })
  }
  return { duration, lanes: [voice, ...pictures, { ...base('notes', 'notes', 'Notes', 'examiner intervals'), notes }], segments, tracks: [], stills: [], notes, content }
}

/** `pcm_s16le · 16 kHz · mono · 8.0 s · lossless` for the header and the exhibit. */
export function describeFacts(facts: AudioFacts | null | undefined): string | null {
  if (!facts) return null
  const channels = facts.channels === 1 ? 'mono' : facts.channels === 2 ? 'stereo' : facts.channels ? `${facts.channels} channels` : null
  const parts = [
    facts.codec ?? null,
    facts.sample_rate ? `${Math.round(facts.sample_rate / 100) / 10} kHz` : null,
    channels,
    facts.duration_s != null ? `${facts.duration_s.toFixed(1)} s` : null,
    facts.lossless ? 'lossless' : facts.bit_rate ? `${Math.round(facts.bit_rate / 1000)} kb/s` : null,
  ].filter((x): x is string => x != null)
  return parts.length ? parts.join(' · ') : null
}
