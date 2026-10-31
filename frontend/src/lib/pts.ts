/** Frame and time mapping for video analyses (ARCHITECTURE §7.1 `pts.ts`, §7.5).
 *  - The probe's per-frame pts table is in display order; frame k shows from `pts[k] × time_base` until the next frame's pts.
 *  - A media time maps to the frame shown at that time: the last frame whose start is at or before it (binary search).
 *  - A frame step seeks a quarter frame past the frame's start, so the browser lands inside the frame rather than on its edge.
 *  - Times are seconds from the stream's pts origin, the same seconds the backend writes into segments (`start_s`). */

export interface PtsTable {
  time_base: string
  pts: readonly number[]
}

export interface PtsIndex {
  count: number
  timeBase: readonly [number, number]
  /** Frame start in seconds, exactly pts × time_base. */
  times: Float64Array
  /** Median frame step in seconds (0 for a single frame). */
  medianStep: number
  /** Frames per second from the median step, or null for a single frame. */
  fps: number | null
  /** True when a frame step departs from the median by more than one timestamp tick and 1 %. */
  vfr: boolean
  /** End of the last frame: its start plus the median step. */
  duration: number
}

/** Media times within this tolerance of a frame start count as that frame (float noise in `mediaTime`, never a real frame gap). */
const EPSILON_S = 1e-4
const QUARTER = 0.25

export function parseTimeBase(value: string): [number, number] {
  const m = /^([1-9]\d*)\/([1-9]\d*)$/.exec(value.trim())
  if (!m) throw new Error(`time base ${JSON.stringify(value)} must look like 1/30000`)
  return [Number(m[1]), Number(m[2])]
}

function median(values: readonly number[]): number {
  if (!values.length) return 0
  const s = [...values].sort((a, b) => a - b)
  const mid = s.length >> 1
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

export function buildPtsIndex(table: PtsTable): PtsIndex {
  const timeBase = parseTimeBase(table.time_base)
  const tick = timeBase[0] / timeBase[1]
  const count = table.pts.length
  const times = new Float64Array(count)
  for (let k = 0; k < count; k++) times[k] = (table.pts[k] * timeBase[0]) / timeBase[1]
  const steps: number[] = []
  for (let k = 1; k < count; k++) steps.push(table.pts[k] - table.pts[k - 1])
  const medianTicks = median(steps)
  const vfr = steps.some((d) => Math.abs(d - medianTicks) > Math.max(1, 0.01 * medianTicks))
  const medianStep = medianTicks * tick
  const span = count > 1 ? times[count - 1] - times[0] : 0
  return {
    count, timeBase, times, medianStep, vfr,
    // The average rate over the table: millisecond time bases round 1/30 s to 33 or 34 ms, so the median step alone reads 30.303.
    fps: span > 0 ? (count - 1) / span : null,
    duration: count ? times[count - 1] + medianStep : 0,
  }
}

/** The frame shown at media time `t`: the last frame starting at or before it, clamped to the table. */
export function frameAt(index: PtsIndex, t: number): number {
  const { times, count } = index
  if (!count || !Number.isFinite(t) || t <= times[0] + EPSILON_S) return 0
  let lo = 0
  let hi = count - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (times[mid] <= t + EPSILON_S) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** The frame whose start rounds to `ms` milliseconds (still exhibits carry `frame_pts` in rounded ms): half a millisecond of slack. */
export function frameAtMs(index: PtsIndex, ms: number): number {
  return frameAt(index, (ms + 0.5) / 1000)
}

export function clampFrame(index: PtsIndex, k: number): number {
  return Math.min(Math.max(0, Math.round(k)), Math.max(0, index.count - 1))
}

/** Seconds frame k is shown for; the last frame takes the median step. */
export function frameDuration(index: PtsIndex, k: number): number {
  const i = clampFrame(index, k)
  return i + 1 < index.count ? index.times[i + 1] - index.times[i] : index.medianStep
}

/** The media time a seek to frame k aims at: a quarter frame past its start. */
export function seekTime(index: PtsIndex, k: number): number {
  const i = clampFrame(index, k)
  return index.count ? index.times[i] + QUARTER * frameDuration(index, i) : 0
}

export function stepFrame(index: PtsIndex, k: number, delta: number): number {
  return clampFrame(index, k + delta)
}

/** The analysed frame nearest to k in a sorted list of frame indices (ties go to the earlier frame), or null for an empty list. */
export function nearestFrame(sorted: readonly number[], k: number): number | null {
  if (!sorted.length) return null
  let lo = 0
  let hi = sorted.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (sorted[mid] < k) lo = mid + 1
    else hi = mid
  }
  const after = sorted[lo]
  const before = lo > 0 ? sorted[lo - 1] : after
  return k - before <= after - k ? before : after
}

/** Seconds of a pts value in its own time base (segments and annotations carry theirs). */
export function ptsSeconds(pts: number, timeBase: string): number {
  const [num, den] = parseTimeBase(timeBase)
  return (pts * num) / den
}

/** `HH:MM:SS.mmm` (UX §5.10); negative and non-finite times read 00:00:00.000. */
export function timecode(seconds: number): string {
  const ms = Number.isFinite(seconds) && seconds > 0 ? Math.round(seconds * 1000) : 0
  const h = Math.floor(ms / 3_600_000)
  const m = Math.floor(ms / 60_000) % 60
  const s = Math.floor(ms / 1000) % 60
  const pad = (n: number, w = 2) => String(n).padStart(w, '0')
  return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(ms % 1000, 3)}`
}

/** 29.97, 30, 23.98: two decimals with trailing zeros removed (an average over rounded timestamps is not exact past that). */
export function fmtFps(fps: number | null): string {
  if (fps == null || !Number.isFinite(fps)) return '—'
  return String(Math.round(fps * 100) / 100)
}
