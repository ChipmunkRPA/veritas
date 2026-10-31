/** The video workspace model (UX §5.10, §8.4; ARCHITECTURE §7.5): lanes, segments, tracks, stills and notes derived from the frame
 *  table, the lane index and the exhibit payloads. Pure, so the lane stack, the lists and the report share one reading.
 *  - Lane scores are smoothed window outputs; headline calibration does not calibrate lanes or segment thresholds.
 *  - Segment letters follow start time, then lane order, then track (D-26 `Seg. a`, until the backend assigns display ids).
 *  - Frame indices are the pts table's (0-based), as in `TrackFrame.frame_index` and the stills. */
import { frameStep } from '../../lib/api'
import type { AnalysisDetail, Annotation, ExhibitMeta, ExhibitPayload, FrameTable } from '../../lib/api/queries'
import { frameAt, frameAtMs, ptsSeconds, type PtsIndex } from '../../lib/pts'
import { estimateOf } from '../analysis/answer'
import { noteNumbers } from '../review/review'
import type { LaneStep } from '../../components/ui'

export type SeriesExhibit = Extract<ExhibitPayload, { kind: 'series' }>
export type SegmentsExhibit = Extract<ExhibitPayload, { kind: 'segments' }>
export type FaceTrackExhibit = Extract<ExhibitPayload, { kind: 'face_track' }>
export type TrackFrame = FaceTrackExhibit['frames'][number]
export type LaneIndexEntry = FrameTable['lanes'][number]
export interface Box { x: number; y: number; w: number; h: number }

export interface Span { start: number; end: number; p: number; step: LaneStep }
export interface Gap { start: number; end: number; label: string }
export interface CurvePoint { start: number; end: number; t: number; ms: number; confidence: number | null }
export interface Tick { t: number; frame: number; kind: 'key' | 'irregular' | 'cut' | 'sampled' }

export interface VideoSegment {
  key: string
  letter: string
  lane: string
  laneName: string
  trackId: string | null
  trackName: string | null
  start: number
  end: number
  startFrame: number
  endFrame: number
  peak: number
  step: LaneStep
  exhibitId: string
  hypothesis: string
  sources: string[]
}

export interface TrackItem {
  trackId: string
  letter: string | null
  name: string
  exhibitId: string
  frames: TrackFrame[]
  firstFrame: number
  lastFrame: number
  start: number
  end: number
  detections: number
  /** Highest smoothed face-lane probability on this track (uncalibrated); without a face lane, the highest per-frame face score
   *  on it; null when neither scored it. */
  peak: number | null
}

export interface NoteItem { id: string; label: string; start: number; end: number; body: string; visibility: string }
export interface StillItem { frame: number; ms: number; exhibitId: string }

export type LaneKind = 'estimate' | 'face' | 'lip_sync' | 'voice' | 'synthesis' | 'other' | 'codec' | 'shots' | 'provenance' | 'notes' | 'sampled' | 'waveform' | 'spectrogram'

/** One bin of the waveform envelope: the lowest and highest sample and the RMS level, in shares of full scale. */
export interface WaveBin { t: number; low: number; high: number; rms: number }

export interface LaneModel {
  key: string
  kind: LaneKind
  label: string
  detail: string
  trackId: string | null
  state: 'ready' | 'not_run'
  /** Display restriction only: executed checks and stored scores remain unchanged. */
  inspectionOnly?: boolean
  /** N-19k (Q2-12): the lane of a check that scored and is not counted in the estimate (the fusion's record on its signal). */
  notCounted?: boolean
  reason?: string
  spans: Span[]
  gaps: Gap[]
  segments: VideoSegment[]
  points?: [number, number][]
  /** Discrete observations, never a claim of continuous scored coverage. */
  samples?: [number, number][]
  coverageNote?: string
  curve?: CurvePoint[]
  ticks?: Tick[]
  badge?: string
  notes?: NoteItem[]
  /** A longer reading note for the lane key and the accessible summary; the gutter shows `detail` only. */
  explain?: string
  /** Waveform lane (L-7): the envelope, the width of a bin in seconds and the labelled content runs drawn over it. */
  wave?: WaveBin[]
  binS?: number
  runs?: { start: number; end: number; label: string }[]
  /** Spectrogram lane (L-7): the picture and the seconds of signal it spans from 0. */
  imageUrl?: string
  imageSeconds?: number
}

export type VideoInspectionReason = 'not_assessable' | 'no_estimate'

export interface VideoModel {
  inspectionReason?: VideoInspectionReason
  duration: number
  lanes: LaneModel[]
  segments: VideoSegment[]
  /** N-19k (Q2-12): the segments of checks that scored and are not counted in the estimate. They stay on their lanes, drawn
   *  for inspection, and are in no list of flagged or strongest segments. */
  uncounted?: VideoSegment[]
  tracks: TrackItem[]
  stills: StillItem[]
  notes: NoteItem[]
}

// N-19d (design review P2): a lane and a segment are named as the check that drew them ("face-swap check", "synthetic-speech
// check"), not as a result: "Audio · synthetic speech" and "a 00:01.0–00:02.0 face-swap · Face A" stood on a file that could
// not be assessed
export const LANE_NAMES: Record<string, string> = { face_swap: 'Face-swap check', lip_sync: 'Lip-sync check', voice: 'Synthetic-speech check', synthesis: 'Synthesis check', partial_edit: 'Partial-edit check' }
const LANE_ORDER = ['face_swap', 'partial_edit', 'lip_sync', 'voice', 'synthesis']
const NOT_ANALYSED = 'not analysed'
const LEGACY_FUSER = 'fuse'
const CODEC_DETECTOR = 'vid.codec.ffprobe'

export const laneName = (lane: string): string => LANE_NAMES[lane] ?? lane.replace(/_/g, ' ')

function median(values: readonly number[]): number {
  if (!values.length) return 0
  const s = [...values].sort((a, b) => a - b)
  const mid = s.length >> 1
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

/** Letters a…z, then aa, ab…: the segment callouts (UX §5.10). */
export function letterFor(n: number): string {
  let out = ''
  let k = n + 1
  while (k > 0) {
    const r = (k - 1) % 26
    out = String.fromCharCode(97 + r) + out
    k = Math.floor((k - 1) / 26)
  }
  return out
}

/** Window cells from point samples (window centres or sampled frames): cells meet halfway between neighbours; a spacing above twice
 *  the median leaves a "not analysed" gap. Null values leave their window unscored. */
export function windowSpans(points: readonly (readonly [number, number | null])[], duration: number): { spans: Span[]; gaps: Gap[] } {
  // Keep null/invalid observations in the spacing grid so they leave an unscored cell.
  const pts = points.filter(p => Number.isFinite(p[0])).slice().sort((a, b) => a[0] - b[0])
  if (!pts.length) return { spans: [], gaps: [] }
  const spacing = pts.slice(1).map((p, i) => p[0] - pts[i][0]).filter((d) => d > 0)
  const m = spacing.length ? median(spacing) : duration > 0 ? duration : 1
  const half = m / 2
  const clamp = (t: number) => (duration > 0 ? Math.min(duration, Math.max(0, t)) : Math.max(0, t))
  const spans: Span[] = []
  const gaps: Gap[] = []
  for (let i = 0; i < pts.length; i++) {
    const [t, p] = pts[i]
    const prev = pts[i - 1]
    const next = pts[i + 1]
    const joinedPrev = prev != null && t - prev[0] <= 2 * m
    const joinedNext = next != null && next[0] - t <= 2 * m
    if (prev && !joinedPrev) gaps.push({ start: clamp(prev[0] + half), end: clamp(t - half), label: NOT_ANALYSED })
    const start = clamp(joinedPrev ? (prev[0] + t) / 2 : t - half)
    const end = clamp(joinedNext ? (t + next[0]) / 2 : t + half)
    if (p == null || !Number.isFinite(p)) gaps.push({ start, end, label: NOT_ANALYSED })
    else spans.push({ start, end, p, step: frameStep(p) })
  }
  return { spans, gaps }
}

/** Uncovered parts of [from, to] outside the spans, as "not analysed" gaps (at least a millisecond wide). */
function uncovered(spans: readonly Span[], from: number, to: number): Gap[] {
  const out: Gap[] = []
  let at = from
  for (const s of [...spans].sort((a, b) => a.start - b.start)) {
    if (s.start > at + 1e-3) out.push({ start: at, end: Math.min(s.start, to), label: NOT_ANALYSED })
    at = Math.max(at, s.end)
    if (at >= to) break
  }
  if (to > at + 1e-3) out.push({ start: at, end: to, label: NOT_ANALYSED })
  return out
}

/** The saved bounds are authoritative. Older series contain centres only, so retain values as discrete observations.
 *  Point samples are discrete, but a lane that is sampled at a regular spacing was analysed along its length: when gaps are
 *  looked for, a sample stands for the lane's median spacing either side, so nothing is "not analysed" between neighbours
 *  (N-17; browser QA P1-5 d). A spacing above twice the median still leaves a gap, as in `windowSpans`. `from` and `to` bound
 *  the time the lane is about (a face track's own span). */
function laneCoverage(windows: LaneIndexEntry['windows'], points: readonly [number, number | null][], duration: number, from = 0, to = duration) {
  const spans: Span[] = []
  const samples: [number, number][] = []
  const clamp = (t: number) => Math.min(duration, Math.max(0, t))
  const known = windows != null
  if (known) {
    for (const w of windows) {
      if (!Number.isFinite(w.start_s) || !Number.isFinite(w.end_s) || w.end_s < w.start_s || w.end_s < 0 || w.start_s > duration ||
          typeof w.p !== 'number' || !Number.isFinite(w.p) || w.p < 0 || w.p > 1) continue
      if (w.kind === 'point' && w.start_s === w.end_s) samples.push([clamp(w.start_s), w.p])
      else if (w.kind === 'interval' && clamp(w.end_s) > clamp(w.start_s)) {
        spans.push({ start: clamp(w.start_s), end: clamp(w.end_s), p: w.p, step: frameStep(w.p) })
      }
    }
  } else {
    for (const [t, p] of points) {
      if (Number.isFinite(t) && t >= 0 && t <= duration && p != null && Number.isFinite(p) && p >= 0 && p <= 1) samples.push([t, p])
    }
  }
  spans.sort((a, b) => a.start - b.start || a.end - b.end)
  samples.sort((a, b) => a[0] - b[0])
  const spacing = samples.slice(1).map(([t], i) => t - samples[i][0]).filter((d) => d > 0)
  const reach = known && spacing.length ? median(spacing) : 0
  const occupied = [...spans, ...samples.map(([t, p]) => ({ start: t - reach, end: t + reach, p, step: frameStep(p) }))]
  const gaps = uncovered(occupied, from, to).map(g => ({ ...g, label: known ? NOT_ANALYSED : 'coverage unknown' }))
  const coverageNote = !known ? 'window boundaries unavailable' : samples.length ? 'sampled values only' : undefined
  return { spans, gaps, samples, coverageNote }
}

const series = (payload: SeriesExhibit | undefined, name: string, trackId?: string | null) =>
  payload?.series.find((s) => s.name === name && (trackId === undefined || (s.track_id ?? null) === trackId)) ?? null

const pointsOf = (s: ReturnType<typeof series>): [number, number | null][] => (s?.points ?? []).map((p) => [Number(p[0]), p[1] == null ? null : Number(p[1])])

/** The face box of a track at frame k: the tracker's box on its own frames (a box without a detection score was interpolated by
 *  the tracker), else a linear interpolation between the nearest track frames, marked interpolated (ARCHITECTURE §7.5). */
export function boxAt(track: TrackItem, k: number): { box: Box; interpolated: boolean; score: number | null } | null {
  const fr = track.frames
  if (!fr.length || k < fr[0].frame_index || k > fr[fr.length - 1].frame_index) return null
  let lo = 0
  let hi = fr.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (fr[mid].frame_index <= k) lo = mid
    else hi = mid - 1
  }
  const a = fr[lo]
  if (a.frame_index === k) return { box: { ...a.bbox }, interpolated: a.det_score == null, score: a.det_score ?? null }
  const b = fr[lo + 1]
  const u = (k - a.frame_index) / (b.frame_index - a.frame_index)
  const mix = (x: number, y: number) => x + (y - x) * u
  return { box: { x: mix(a.bbox.x, b.bbox.x), y: mix(a.bbox.y, b.bbox.y), w: mix(a.bbox.w, b.bbox.w), h: mix(a.bbox.h, b.bbox.h) }, interpolated: true, score: null }
}

export function trackName(track: { identity?: string | null; track_id: string }): string {
  if (track.identity) return `Face ${track.identity}`
  const n = Number(track.track_id.split(':').pop())
  return Number.isFinite(n) ? `Face track ${n + 1}` : track.track_id
}

export function videoInspectionReason(decision: AnalysisDetail['decision']): VideoInspectionReason | undefined {
  if (decision?.code === 'not_assessable') return 'not_assessable'
  const estimate = decision ? estimateOf(decision) : null
  return estimate == null || !Number.isFinite(estimate) ? 'no_estimate' : undefined
}

export const videoInspectionCaption = (reason: VideoInspectionReason): string =>
  `${reason === 'not_assessable' ? 'Not assessable' : 'No estimate'}: values shown for inspection only.`

/** What the model reads of a signal: whether the fusion recorded that its check is not counted in the estimate. */
export interface LaneSignal { id: string; detector_id: string; outside_estimate?: unknown }

export const UNCOUNTED_DETAIL = 'not counted in this estimate'

/** N-19k (Q2-12): what the list of segments says of the ones it leaves out (`forensic.uncounted_note`). */
export function uncountedNote(uncounted: readonly VideoSegment[] | undefined, lanes: readonly LaneModel[]): string | null {
  if (!uncounted?.length) return null
  const labels = [...new Set(lanes.filter((l) => l.notCounted && l.segments.length > 0).map((l) => l.label))]
  const where = labels.length === 1 ? `the ${labels[0]} lane` : `the lanes ${labels.join(', ')}`
  const one = uncounted.length === 1
  return `Not listed: ${uncounted.length} segment${one ? '' : 's'} on ${where}. ${labels.length === 1 ? 'Its check is' : 'Their checks are'} not counted in this estimate; the ${labels.length === 1 ? 'lane shows' : 'lanes show'} ${one ? 'it' : 'them'} for inspection.`
}

export function buildVideoModel({ index, frames, exhibits, payloads, annotations, c2paState, decision, signals = [] }: {
  /** The analysis' signals; without them no check is known to be outside the estimate (a result stored before N-19k). */
  signals?: readonly LaneSignal[]
  index: PtsIndex
  frames: FrameTable
  exhibits: readonly ExhibitMeta[]
  payloads: readonly ExhibitPayload[]
  annotations: readonly Annotation[]
  /** The H3 assessment's Content Credentials state, or null when none was established. */
  c2paState: string | null
  decision: AnalysisDetail['decision']
}): VideoModel {
  const inspectionReason = videoInspectionReason(decision)
  const duration = index.duration
  const byId = new Map(payloads.map((p) => [p.id, p]))
  const seriesById = (id: string | null | undefined) => {
    const p = id ? byId.get(id) : undefined
    return p?.kind === 'series' ? p : undefined
  }
  const first = frames.lanes.filter((l) => l.pass === 'first')
  // The per-frame scores of the earlier method (`fuse`): p_frame, and p_face on the largest face of a sampled frame.
  const fused = payloads.find((p): p is SeriesExhibit => p.kind === 'series' && p.created_by === LEGACY_FUSER && p.series.some((s) => s.name === 'p_frame'))
  const numeric = (x: [number, number | null]): x is [number, number] => x[1] != null && Number.isFinite(x[0]) && Number.isFinite(x[1])

  // --- tracks ---------------------------------------------------------------------------------------------
  const faceLanes = first.filter((l) => l.track_id != null && (l.lane === 'face_swap' || l.lane === 'partial_edit'))
  const tracks: TrackItem[] = payloads.filter((p): p is FaceTrackExhibit => p.kind === 'face_track' && p.frames.length > 0).map((p) => {
    const fr = [...p.frames].sort((a, b) => a.frame_index - b.frame_index)
    const peaks = faceLanes.filter((l) => l.track_id === p.track.track_id)
      .flatMap((l) => pointsOf(series(seriesById(l.series_exhibit_id), 'p_smoothed')).map((x) => x[1]))
      .filter((x): x is number => x != null)
    const firstFrame = fr[0].frame_index
    const lastFrame = fr[fr.length - 1].frame_index
    return {
      trackId: p.track.track_id, letter: p.track.identity ?? null, name: trackName(p.track), exhibitId: p.id, frames: fr, firstFrame, lastFrame,
      start: index.times[Math.min(firstFrame, index.count - 1)] ?? 0, end: index.times[Math.min(lastFrame, index.count - 1)] ?? 0,
      detections: fr.filter((f) => f.det_score != null).length, peak: peaks.length ? Math.max(...peaks) : null as number | null,
    }
  }).sort((a, b) => a.start - b.start || a.trackId.localeCompare(b.trackId, undefined, { numeric: true }))
  const trackById = new Map(tracks.map((t) => [t.trackId, t]))
  // N-17 (browser QA P1-5 b): the face-swap check of the per-frame set scores the largest face of each sampled frame and has no
  // lane of its own. Its scores belong to the track whose box is the largest at that frame, so that the track's lane shows what
  // the ledger holds, in place of "Not run".
  const frameFaceScores = new Map<string, [number, number][]>()
  for (const [t, p] of pointsOf(series(fused, 'p_face')).filter(numeric)) {
    const k = frameAt(index, t)
    let best: TrackItem | null = null
    let area = 0
    for (const track of tracks) {
      const at = boxAt(track, k)
      if (at && at.box.w * at.box.h > area) { best = track; area = at.box.w * at.box.h }
    }
    if (best) frameFaceScores.set(best.trackId, [...(frameFaceScores.get(best.trackId) ?? []), [t, p]])
  }
  for (const track of tracks) {
    const scores = frameFaceScores.get(track.trackId)
    if (track.peak == null && scores?.length) track.peak = Math.max(...scores.map((x) => x[1]))
  }

  // --- segments -------------------------------------------------------------------------------------------
  const rawSegments = first.flatMap((l) => {
    const p = l.segments_exhibit_id ? byId.get(l.segments_exhibit_id) : undefined
    return p?.kind === 'segments' ? p.segments.map((s, i) => ({ s, entry: l, exhibitId: p.id, i })) : []
  })
  const order = (lane: string) => (LANE_ORDER.indexOf(lane) + LANE_ORDER.length + 1) % (LANE_ORDER.length + 1)
  const segments: VideoSegment[] = rawSegments
    .map(({ s, entry, exhibitId, i }) => {
      const start = ptsSeconds(s.start_pts, s.time_base)
      const end = ptsSeconds(s.end_pts, s.time_base)
      const tid = s.track_id ?? entry.track_id ?? null
      return {
        key: `${exhibitId}:${i}`, letter: '', lane: s.lane ?? entry.lane, laneName: laneName(s.lane ?? entry.lane), trackId: tid,
        trackName: tid ? trackById.get(tid)?.name ?? tid : null, start, end, startFrame: frameAt(index, start), endFrame: frameAt(index, end),
        peak: s.peak, step: frameStep(s.peak), exhibitId, hypothesis: s.hypothesis, sources: s.sources ?? [], displayId: s.display_id ?? null,
      }
    })
    .sort((a, b) => a.start - b.start || order(a.lane) - order(b.lane) || (a.trackId ?? '').localeCompare(b.trackId ?? '', undefined, { numeric: true }))
    .map(({ displayId, ...seg }, n) => ({ ...seg, letter: displayId ? displayId.replace(/^Seg\.\s*/, '') : letterFor(n) }))
  const segmentsOf = (entry: LaneIndexEntry) => segments.filter((s) => s.lane === entry.lane && s.trackId === (entry.track_id ?? null))
  // N-19k (Q2-12): a check that scored and is not counted in the estimate carries the fusion's record on its signal. The lane
  // of such a check is drawn for inspection and its segments are in no list of flagged segments: "Audio" stood in solid red,
  // and "a 00:00.0–00:10.0 synthetic-speech check 0.91" was the strongest segment of a result that counted 0 flags
  const outsideSignals = new Set(signals.filter((x) => x.outside_estimate != null).map((x) => x.id))
  const outsideDetectors = new Set(signals.filter((x) => x.outside_estimate != null).map((x) => x.detector_id))
  const countedDetectors = new Set(signals.filter((x) => x.outside_estimate == null).map((x) => x.detector_id))
  const laneUncounted = (entry: LaneIndexEntry) => entry.detectors.length > 0 && entry.detectors.every((d) => outsideDetectors.has(d) && !countedDetectors.has(d))
  const uncountedLanes = new Set(first.filter(laneUncounted).map((l) => `${l.lane}|${l.track_id ?? ''}`))
  const uncountedLaneKeys = new Set(first.filter(laneUncounted).map((l) => (l.lane === 'lip_sync' ? `lip_sync:${l.track_id ?? 'global'}`
    : l.lane === 'voice' || l.lane === 'synthesis' ? `${l.lane}:${l.series_exhibit_id}` : `other:${l.series_exhibit_id}`)))
  const segmentUncounted = (seg: VideoSegment) => (seg.sources.length > 0 ? seg.sources.every((id) => outsideSignals.has(id)) : uncountedLanes.has(`${seg.lane}|${seg.trackId ?? ''}`))

  const lanes: LaneModel[] = []
  const base = (key: string, kind: LaneKind, label: string, detail: string, trackId: string | null = null): LaneModel =>
    ({ key, kind, label, detail, trackId, state: 'ready', spans: [], gaps: [], segments: [] })

  // 1. Estimate: the legacy fuser's per-frame p at the sampled frames. It is the estimate of the uncalibrated fallback only: a
  // fusion model does not read these per-frame scores, so its result has no such lane (N-17; browser QA P1-5 a). The series
  // stays an exhibit of the checks that cite it.
  const pointSupport = (samples: readonly [number, number][]) => samples.map(([t, p]) => ({ kind: 'point' as const, start_s: t, end_s: t, p }))
  if (decision?.fusion_mode !== 'artefact') {
    const estimate = base('estimate', 'estimate', 'Estimate', 'per sampled frame · uncalibrated')
    if (fused) {
      const pts = pointsOf(series(fused, 'p_frame'))
      const samples = pts.filter(numeric)
      Object.assign(estimate, laneCoverage(pointSupport(samples), pts, duration), { points: samples, coverageNote: 'sampled frames only' })
    } else Object.assign(estimate, { state: 'not_run', reason: 'no per-frame fusion' })
    lanes.push(estimate)
  }

  // 2. Face lanes, one per track (letter = on-frame box label).
  const lanePoints = (entry: LaneIndexEntry) => laneCoverage(entry.windows, pointsOf(series(seriesById(entry.series_exhibit_id), 'p_smoothed')), duration)
  if (!tracks.length && !faceLanes.length) lanes.push({ ...base('faces', 'face', 'Faces', 'face-swap check'), state: 'not_run', reason: 'no face tracks' })
  for (const t of tracks) {
    const entry = faceLanes.find((l) => l.track_id === t.trackId)
    const lane = base(`face:${t.trackId}`, 'face', t.name, entry ? `${laneName(entry.lane).toLowerCase()} · p uncalibrated` : 'face-swap check', t.trackId)
    const scores = frameFaceScores.get(t.trackId)
    if (!entry && scores?.length) {
      Object.assign(lane, laneCoverage(pointSupport(scores), scores, duration, t.start, t.end),
        { detail: 'face-swap check · p uncalibrated', coverageNote: 'per sampled frame, largest face' })
    } else if (!entry) Object.assign(lane, { state: 'not_run', reason: 'no face-swap detector scored this track' })
    else {
      Object.assign(lane, lanePoints(entry), { segments: segmentsOf(entry) })
    }
    lanes.push(lane)
  }
  for (const entry of faceLanes.filter((l) => !trackById.has(l.track_id ?? ''))) {
    lanes.push({ ...base(`face:${entry.track_id}`, 'face', entry.track_id ?? 'Face', laneName(entry.lane).toLowerCase(), entry.track_id), ...lanePoints(entry), segments: segmentsOf(entry) })
  }

  // 3. Lip-sync per track: offset curve in ms from the detector's `offset_frames` series and its confidence.
  const syncLanes = first.filter((l) => l.lane === 'lip_sync')
  if (!syncLanes.length) lanes.push({ ...base('lip_sync', 'lip_sync', 'Lip-sync', 'offset ms'), state: 'not_run', reason: 'no lip-sync detector ran' })
  for (const entry of syncLanes) {
    const name = entry.track_id ? trackById.get(entry.track_id)?.name ?? entry.track_id : 'whole video'
    const raw = payloads.find((p): p is SeriesExhibit => p.kind === 'series' && entry.detectors.includes(p.created_by ?? '') && p.series.some((s) => s.name === 'offset_frames' && (s.track_id ?? null) === entry.track_id))
    const offsets = pointsOf(series(raw, 'offset_frames', entry.track_id))
    const conf = new Map(pointsOf(series(raw, 'confidence', entry.track_id) ?? series(raw, 'peak_correlation', entry.track_id)).map((x) => [x[0], x[1]]))
    const valid = offsets.filter((x): x is [number, number] => x[1] != null).sort((a, b) => a[0] - b[0])
    const curve = valid.map(([t, f]) => {
      const support = entry.windows?.find(w => w.kind === 'interval' && Math.abs((w.start_s + w.end_s) / 2 - t) < 1e-3)
      return { t, ms: f * index.medianStep * 1000, confidence: conf.get(t) ?? null, start: support?.start_s ?? t, end: support?.end_s ?? t }
    })
    lanes.push({ ...base(`lip_sync:${entry.track_id ?? 'global'}`, 'lip_sync', 'Lip-sync', `${name} · offset ms`, entry.track_id), ...lanePoints(entry), curve, segments: segmentsOf(entry) })
  }

  // 4–5. Audio (voice windows) and synthesis (frame and temporal channels), whole video.
  for (const [kind, lane, label, detail, reason] of [
    ['voice', 'voice', 'Audio', 'synthetic-speech check · p uncalibrated', 'no voice detector produced windows'],
    ['synthesis', 'synthesis', 'Synthesis', 'synthesis check, whole frame · p uncalibrated', 'no synthesis detector ran'],
  ] as const) {
    const entries = first.filter((l) => l.lane === lane && l.track_id == null)
    if (!entries.length) lanes.push({ ...base(kind, kind, label, detail), state: 'not_run', reason })
    for (const entry of entries) {
      lanes.push({ ...base(`${kind}:${entry.series_exhibit_id}`, kind, label, detail), ...lanePoints(entry), segments: segmentsOf(entry) })
    }
  }
  for (const entry of first.filter((l) => !['face_swap', 'partial_edit', 'lip_sync', 'voice', 'synthesis'].includes(l.lane))) {
    lanes.push({ ...base(`other:${entry.series_exhibit_id}`, 'other', laneName(entry.lane), entry.track_id ?? 'whole video', entry.track_id), ...lanePoints(entry), segments: segmentsOf(entry) })
  }

  // 6. Codec: the GOP strip from the pts table (key frames), irregular key frames from the codec lane when it ran.
  const codec = payloads.find((p): p is SeriesExhibit => p.kind === 'series' && p.created_by === CODEC_DETECTOR && p.series.some((s) => s.name === 'gop_frames'))
  const irregular = new Set(pointsOf(series(codec, 'irregular_key_frames')).map((x) => frameAt(index, x[0])))
  const gops = frames.key_frames.slice(1).map((k, i) => k - frames.key_frames[i])
  const gop = gops.length ? median(gops) : null
  lanes.push({
    ...base('codec', 'codec', 'Codec', 'GOP · key frames'),
    ticks: frames.key_frames.map((k) => ({ t: index.times[k] ?? 0, frame: k, kind: irregular.has(k) ? 'irregular' as const : 'key' as const })),
    badge: `${frames.key_frames.length} key frame${frames.key_frames.length === 1 ? '' : 's'}${gop != null ? ` · GOP ${gop} frames` : ''}${irregular.size ? ` · ${irregular.size} irregular` : ''}`,
  })

  // 7. Shots.
  lanes.push(frames.shots_detected
    ? { ...base('shots', 'shots', 'Shots', 'detected cuts'), ticks: frames.shot_cuts.map((k) => ({ t: index.times[k] ?? 0, frame: k, kind: 'cut' as const })), badge: frames.shot_cuts.length ? `${frames.shot_cuts.length + 1} shots` : 'one shot' }
    : { ...base('shots', 'shots', 'Shots', 'detected cuts'), state: 'not_run', reason: 'not in this profile' })

  // 8. Provenance badge row.
  lanes.push({ ...base('provenance', 'provenance', 'Provenance', 'Content Credentials'), badge: c2paState ? `Content Credentials ${c2paState}` : 'No Content Credentials established' })

  // 9. Notes: interval annotations of the examiner layer, with the N-numbers of the notes thread.
  const numbers = noteNumbers(annotations)
  const notes: NoteItem[] = annotations
    .filter((a) => a.target_type === 'interval' && !a.deleted_at && a.geometry)
    .map((a, i): NoteItem | null => {
      const g = a.geometry as { start_pts?: number; end_pts?: number; time_base?: string }
      const ok = typeof g.start_pts === 'number' && typeof g.end_pts === 'number' && typeof g.time_base === 'string'
      return ok ? { id: a.id, label: numbers.get(a.id) ?? `N${i + 1}`, start: ptsSeconds(g.start_pts as number, g.time_base as string), end: ptsSeconds(g.end_pts as number, g.time_base as string), body: a.body ?? '', visibility: a.visibility } : null
    })
    .filter((n): n is NoteItem => n != null)
  lanes.push({ ...base('notes', 'notes', 'Notes', 'examiner intervals'), notes })

  // 10. Persisted sampling plan; this is not a detector-execution count.
  lanes.push({ ...base('sampled', 'sampled', 'Sampled', `${frames.sampled.length} frames in sampling plan`), ticks: frames.sampled.map((k) => ({ t: index.times[k] ?? 0, frame: k, kind: 'sampled' as const })) })

  const stills: StillItem[] = exhibits
    .filter((x) => x.kind === 'image' && x.role === 'still' && x.frame_pts != null)
    .map((x) => ({ frame: frameAtMs(index, x.frame_pts as number), ms: x.frame_pts as number, exhibitId: x.id }))
    .sort((a, b) => a.frame - b.frame)

  // Inspection is a view restriction, not a claim that the detector did not execute. Preserve every raw value and gap.
  const explained = lanes.map(lane => lane.coverageNote ? { ...lane, detail: `${lane.detail} · ${lane.coverageNote}` } : lane)
  const inspectionLanes = explained.map((lane) => inspectionReason && ['estimate', 'face', 'lip_sync', 'voice', 'synthesis', 'other'].includes(lane.kind)
    ? { ...lane, inspectionOnly: true, detail: lane.detail.replace(' · p uncalibrated', '') } : lane)
  // N-19k: under a result with an estimate, the lanes of the checks it does not count (never the lane of a face track: the
  // check of the frames, which the model holds, is drawn on it beside the track's own)
  const uncounted = inspectionReason ? [] : segments.filter((seg) => !['face_swap', 'partial_edit'].includes(seg.lane) && segmentUncounted(seg))
  const left = new Set(uncounted.map((seg) => seg.key))
  const drawn = inspectionLanes.map((lane) => {
    const mine = !inspectionReason && ['lip_sync', 'voice', 'synthesis', 'other'].includes(lane.kind) && lane.state === 'ready'
      && (lane.segments.length > 0 ? lane.segments.every((seg) => left.has(seg.key)) : uncountedLaneKeys.has(lane.key))
    return mine ? { ...lane, inspectionOnly: true, notCounted: true, detail: lane.detail.includes(' · p uncalibrated') ? lane.detail.replace(' · p uncalibrated', ` · ${UNCOUNTED_DETAIL}`) : `${lane.detail} · ${UNCOUNTED_DETAIL}` } : lane
  })
  return { duration, lanes: drawn, segments: segments.filter((seg) => !left.has(seg.key)), uncounted, tracks, stills, notes, inspectionReason }
}

/** Plain-text summary of one lane for its `aria-label` and the table view. */
export function describeLane(lane: LaneModel): string {
  if (lane.state === 'not_run') return `${lane.label}: not run, ${lane.reason ?? 'no data'}`
  const parts = [lane.explain ? `${lane.detail} · ${lane.explain}` : lane.detail]
  if (lane.segments.length) parts.push(`segments ${lane.segments.map((s) => s.letter).join(', ')}`)
  if (lane.inspectionOnly) parts.push('values shown for inspection only')
  if (lane.spans.length && !lane.inspectionOnly) parts.push(`highest ${Math.max(...lane.spans.map((s) => s.p)).toFixed(2)}`)
  if (lane.curve?.length) parts.push(`offsets ${Math.round(Math.min(...lane.curve.map((c) => c.ms)))} to ${Math.round(Math.max(...lane.curve.map((c) => c.ms)))} ms`)
  if (lane.badge) parts.push(lane.badge)
  if (lane.samples?.length) parts.push(`${lane.samples.length} sampled value${lane.samples.length === 1 ? '' : 's'}`)
  for (const label of [...new Set(lane.gaps.map(g => g.label))]) {
    const count = lane.gaps.filter(g => g.label === label).length
    parts.push(`${count} ${label} span${count === 1 ? '' : 's'}`)
  }
  if (lane.notes?.length) parts.push(`${lane.notes.length} note${lane.notes.length === 1 ? '' : 's'}`)
  if (lane.wave?.length) parts.push(`peak ${Math.max(...lane.wave.map((b) => Math.max(Math.abs(b.low), Math.abs(b.high)))).toFixed(2)} of full scale`)
  return `${lane.label}: ${parts.join('; ')}`
}
