import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { ChevronLeft, ChevronRight, Pause, Play, StepBack, StepForward } from 'lucide-react'
import type { FrameTable } from '../../lib/api/queries'
import { cx } from '../../lib/cx'
import { clampFrame, fmtFps, frameAt, nearestFrame, seekTime, stepFrame, timecode, type PtsIndex } from '../../lib/pts'
import { Menu } from '../../components/ui'
import { boxAt, type StillItem, type TrackItem, type VideoModel } from './model'

type FrameCallback = (now: number, metadata: { mediaTime: number }) => void
type FrameCallbackVideo = HTMLVideoElement & { requestVideoFrameCallback: (cb: FrameCallback) => number; cancelVideoFrameCallback: (handle: number) => void }
const hasFrameCallback = (v: HTMLVideoElement): v is FrameCallbackVideo => typeof (v as Partial<FrameCallbackVideo>).requestVideoFrameCallback === 'function'

export const RATES = [0.25, 0.5, 1, 2] as const
/** A GOP above this many frames makes browser seeking land away from the analysed frame often enough to say so (UX §5.10). */
const LONG_GOP_FRAMES = 60

export interface FramePlayerApi {
  ref: (el: HTMLVideoElement | null) => void
  frame: number
  playing: boolean
  seeking: boolean
  ready: boolean
  error: string | null
  approximate: boolean
  clockMapped: boolean
  rate: number
  seekFrame: (k: number) => void
  step: (delta: number) => void
  toggle: () => void
  setRate: (rate: number) => void
}

/** `<video>` + `requestVideoFrameCallback` (ARCHITECTURE §7.5): the presented frame's `mediaTime` maps to a frame index through the
 *  pts table. A frame step seeks a quarter frame past the frame's start and the callback (or `seeked`) confirms the landing frame.
 *  Without the callback the index follows `currentTime` and is marked approximate. */
export function playbackSeekTime(index: PtsIndex, frame: number, timelineOrigin: number | null): number | null {
  if (timelineOrigin == null || !Number.isFinite(timelineOrigin)) return null
  const value = seekTime(index, frame) - timelineOrigin
  return Number.isFinite(value) && value >= 0 ? value : null
}

export function useFramePlayer(index: PtsIndex | null, initialFrame: number | null, source: string | null,
  timelineOrigin: number | null = null): FramePlayerApi {
  const clockMapped = timelineOrigin != null && Number.isFinite(timelineOrigin)
  const [video, setVideo] = useState<HTMLVideoElement | null>(null)
  const [frame, setFrame] = useState(initialFrame ?? 0)
  const frameRef = useRef(frame)
  frameRef.current = frame
  const [playing, setPlaying] = useState(false)
  const [seeking, setSeeking] = useState(false)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rate, setRateState] = useState(1)
  const [approximate, setApproximate] = useState(false)
  const initial = useRef(initialFrame)
  const rateRef = useRef(rate)

  useEffect(() => {
    setReady(false)
    setPlaying(false)
    setSeeking(false)
    setError(null)
    if (!video || !index || !source || !clockMapped || timelineOrigin == null) return
    // A refreshed signed URL reloads the same video element. Preserve the live
    // frame, including navigation performed while media access was unavailable.
    initial.current ??= frameRef.current
    let live = true
    let metadataReady = false
    let handle = 0
    const callback = hasFrameCallback(video)
    setApproximate(!callback)
    const onPresented: FrameCallback = (_now, meta) => {
      if (!live || !callback) return
      if (metadataReady && video.readyState >= 1) setFrame(frameAt(index, meta.mediaTime + timelineOrigin))
      handle = video.requestVideoFrameCallback(onPresented)
    }
    if (callback) handle = video.requestVideoFrameCallback(onPresented)
    const sync = () => { if (metadataReady && video.readyState >= 1) setFrame(frameAt(index, video.currentTime + timelineOrigin)) }
    const onSeeking = () => setSeeking(true)
    const onSeeked = () => { setSeeking(false); sync() }
    const onTime = () => { if (!callback) sync() }
    const onPlay = () => setPlaying(true)
    const onPause = () => { setPlaying(false); if (!callback) sync() }
    const onMeta = () => {
      metadataReady = true
      setReady(true)
      video.playbackRate = rateRef.current
      if (initial.current != null) {
        const time = playbackSeekTime(index, initial.current, timelineOrigin)
        if (time != null) video.currentTime = time
        initial.current = null
      }
    }
    const onError = () => setError('This browser cannot decode the video; the stills show the analysed frames.')
    const events: [string, () => void][] = [['seeking', onSeeking], ['seeked', onSeeked], ['timeupdate', onTime], ['play', onPlay], ['pause', onPause], ['loadedmetadata', onMeta], ['error', onError]]
    for (const [name, fn] of events) video.addEventListener(name, fn)
    if (video.readyState >= 1) onMeta()
    return () => {
      live = false
      if (callback) video.cancelVideoFrameCallback(handle)
      for (const [name, fn] of events) video.removeEventListener(name, fn)
    }
  }, [video, index, source, timelineOrigin, clockMapped])

  const seekFrame = useCallback((k: number) => {
    if (!index) return
    const target = clampFrame(index, k)
    setFrame(target)
    // Metadata navigation stays available while secure media links are pending.
    // Apply the latest requested frame when the video finally attaches and can seek.
    const time = playbackSeekTime(index, target, timelineOrigin)
    if (video && video.readyState >= 1 && !error && time != null) video.currentTime = time
    else initial.current = target
  }, [video, index, error, timelineOrigin])
  const step = useCallback((delta: number) => { if (index) seekFrame(stepFrame(index, frameRef.current, delta)) }, [index, seekFrame])
  const toggle = useCallback(() => {
    if (!video || error || !clockMapped) return
    if (video.paused) void video.play().catch(() => undefined)
    else video.pause()
  }, [video, error, clockMapped])
  const setRate = useCallback((r: number) => {
    rateRef.current = r
    if (video) video.playbackRate = r
    setRateState(r)
  }, [video])
  return { ref: setVideo, frame, playing, seeking, ready, error, approximate, clockMapped, rate, seekFrame, step, toggle, setRate }
}

function Scrubber({ index, frame, onFrame }: { index: PtsIndex; frame: number; onFrame: (k: number) => void }) {
  const dragging = useRef(false)
  const last = Math.max(0, index.count - 1)
  const at = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    onFrame(frameAt(index, (Math.min(r.width, Math.max(0, e.clientX - r.left)) / Math.max(1, r.width)) * index.duration))
  }
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowDown: -1, ArrowUp: 1, PageDown: -10, PageUp: 10 }[e.key]
    if (delta !== undefined) onFrame(stepFrame(index, frame, delta))
    else if (e.key === 'Home') onFrame(0)
    else if (e.key === 'End') onFrame(last)
    else return
    e.preventDefault()
    e.stopPropagation()
  }
  const u = index.duration > 0 ? (index.times[frame] ?? 0) / index.duration : 0
  return (
    <div role="slider" tabIndex={0} data-testid="scrubber" aria-label="Frame" aria-valuemin={0} aria-valuemax={last} aria-valuenow={frame}
      aria-valuetext={`${(index.times[frame] ?? 0).toFixed(1)} seconds, frame ${frame}`} onKeyDown={onKey}
      onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); at(e) }}
      onPointerMove={(e) => { if (dragging.current) at(e) }} onPointerUp={() => { dragging.current = false }}
      className="relative h-6 cursor-pointer touch-none rounded-sm">
      <span aria-hidden className="absolute left-0 right-0 top-[11px] h-0.5 bg-border-strong" />
      <span aria-hidden className="absolute left-0 top-[11px] h-0.5 bg-text" style={{ width: `${u * 100}%` }} />
      <span aria-hidden className="absolute top-[5px] h-3.5 w-0.5 -ml-px bg-text" style={{ left: `${u * 100}%` }} />
    </div>
  )
}

// N-19d: the view badge stands 12 px from the top left of the stage; its width is taken from its text (micro mono type)
const BADGE = { x: 12, y: 12, h: 16 }
export const badgeWidth = (text: string): number => text.length * 6.1 + 12
export interface Rect { x: number; y: number; w: number; h: number }
const CORNERS = ['top-left', 'top-right', 'bottom-left', 'bottom-right'] as const
export type BadgeCorner = typeof CORNERS[number]
const CORNER_CLASS: Record<BadgeCorner, string> = { 'top-left': 'left-3 top-3', 'top-right': 'right-3 top-3', 'bottom-left': 'left-3 bottom-3', 'bottom-right': 'right-3 bottom-3' }

/** The rectangle of the view badge in a corner of the stage. */
export function badgeRect(corner: BadgeCorner, badgeW: number, stage: { width: number; height: number }): Rect {
  return { x: corner.endsWith('left') ? BADGE.x : stage.width - BADGE.x - badgeW, y: corner.startsWith('top') ? BADGE.y : stage.height - BADGE.y - BADGE.h, w: badgeW, h: BADGE.h }
}

const meets = (a: Rect, b: Rect, halo = 2): boolean => a.x < b.x + b.w + halo && a.x + a.w + halo > b.x && a.y < b.y + b.h + halo && a.y + a.h + halo > b.y

/** N-19k (design re-review D2-12): the view badge stood at the top left whatever was under it and covered a corner of a face
 *  box there. It takes the first corner of the stage in which it stands on no face box; the top left when a box is in each. */
export function badgeCorner(boxes: readonly Rect[], badgeW: number, stage: { width: number; height: number }): BadgeCorner {
  return CORNERS.find((corner) => boxes.every((box) => !meets(badgeRect(corner, badgeW, stage), box))) ?? 'top-left'
}

/** The top of a face's label: above its box, or under the box when the label would stand on the view badge (`badge`: where
 *  the badge is; at the top left unless a face box is there). */
export function faceLabelTop(box: { x: number; y: number; h: number }, labelWidth: number, badgeW: number, badge: Rect = { x: BADGE.x, y: BADGE.y, w: badgeW, h: BADGE.h }): number {
  const above = Math.max(0, box.y - 18)
  const onBadge = box.x < badge.x + badge.w && box.x + labelWidth > badge.x && above < badge.y + badge.h && above + 16 > badge.y
  return onBadge ? box.y + box.h + 2 : above
}

const transportButton = 'h-8 min-w-8 px-1.5 inline-flex items-center justify-center gap-1 rounded-sm text-small text-text-muted enabled:hover:text-text enabled:hover:bg-surface-3 transition-colors duration-fast disabled:opacity-40 disabled:cursor-not-allowed'

/**
 * Frame player (UX §5.10 transport; ARCHITECTURE §7.5): the source video on the dark stage with face boxes, the "detector view" still
 * when paused on a sampled frame, the transport (timecode, frame n / N, fps and CFR/VFR, previous/next flagged, rate) and the
 * frame-snapping scrubber. Keyboard: the workspace handles Space , . [ ] ← → Home End.
 */
export default function FramePlayer({ player, index, frames, model, src, stillUrl, isolatedTrack, onPrevFlagged, onNextFlagged }: {
  player: FramePlayerApi
  index: PtsIndex
  frames: FrameTable
  model: VideoModel
  src: string | null
  stillUrl: (still: StillItem) => string | null
  isolatedTrack: string | null
  onPrevFlagged: (() => void) | null
  onNextFlagged: (() => void) | null
}) {
  const { frame } = player
  const stage = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })
  useEffect(() => {
    const el = stage.current
    if (!el) return
    const ro = new ResizeObserver((entries) => setSize({ width: Math.floor(entries[0].contentRect.width), height: Math.floor(entries[0].contentRect.height) }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const sampled = useMemo(() => model.stills.map((s) => s.frame), [model.stills])
  const stillAt = model.stills.find((s) => s.frame === frame) ?? null
  const nearest = nearestFrame(sampled, frame)
  const nearestStill = nearest == null ? null : model.stills.find((s) => s.frame === nearest) ?? null
  const paused = !player.playing && !player.seeking
  const still = player.error || !player.clockMapped ? nearestStill : paused ? stillAt : null
  const [loaded, setLoaded] = useState<string | null>(null)
  const stillSrc = still ? stillUrl(still) : null
  const detectorView = still != null && loaded === stillSrc

  const media = { width: frames.width, height: frames.height }
  const scale = media.width > 0 && media.height > 0 ? Math.min(size.width / media.width, size.height / media.height) : 0
  const offset = { x: (size.width - media.width * scale) / 2, y: (size.height - media.height * scale) / 2 }
  const boxes = model.tracks
    .filter((t: TrackItem) => !isolatedTrack || t.trackId === isolatedTrack)
    .map((t) => ({ track: t, at: boxAt(t, detectorView && still ? still.frame : frame) }))
    .filter((b): b is { track: TrackItem; at: NonNullable<ReturnType<typeof boxAt>> } => b.at != null)

  const gop = frames.key_frames.length > 1 ? Math.max(...frames.key_frames.slice(1).map((k, i) => k - frames.key_frames[i])) : frames.pict_types.length
  const caption = !player.clockMapped ? 'Playback timing is not verified for this analysis. Inspect the recorded stills; a new analysis can capture the source clock.' : player.error
    ? player.error
    : player.approximate ? 'This browser has no frame callback: the frame index follows the playback clock and is approximate.'
      : index.vfr || gop > LONG_GOP_FRAMES ? 'Browser seeking can land ±1 frame from the analysed frame; key frames show the exact analysed image.' : null
  const t = index.times[frame] ?? 0

  const placed = scale > 0 ? boxes.map(({ at }) => ({ x: offset.x + at.box.x * scale, y: offset.y + at.box.y * scale, w: at.box.w * scale, h: at.box.h * scale })) : []
  const badge = !src ? 'Preview unavailable' : detectorView && still ? `Detector view · analysed frame ${still.frame}` : !player.clockMapped ? 'Playback clock unavailable' : player.playing ? 'Playing' : stillAt ? 'Loading the analysed frame…' : `Video · frame ${frame} was not sampled`
  const corner = badgeCorner(placed, badgeWidth(badge), size)
  const badgeAt = badgeRect(corner, badgeWidth(badge), size)
  return (
    <div data-testid="frame-player" className="min-w-0">
      <div ref={stage} role="region" tabIndex={0} aria-label="Video stage" aria-describedby="video-stage-description"
        data-testid="video-stage" data-frame={frame} data-view={!src ? 'unavailable' : detectorView ? 'still' : 'video'}
        className="viewer-stage stage relative h-[min(480px,42vh)] min-h-[200px] bg-stage overflow-hidden outline-none">
        <p id="video-stage-description" className="sr-only">Space plays, comma and full stop step frames, brackets jump between flagged segments.</p>
        {src && player.clockMapped && <video ref={player.ref} src={src} preload="auto" playsInline className="absolute inset-0 w-full h-full object-contain" aria-label="Source video" />}
        {stillSrc && (
          <img key={stillSrc} src={stillSrc} alt={`Detector view: the analysed frame ${still?.frame} as the detectors scored it`} onLoad={() => setLoaded(stillSrc)}
            data-testid="detector-still" className={cx('absolute inset-0 w-full h-full object-contain bg-stage', !detectorView && 'invisible')} />
        )}
        {src && scale > 0 && (player.clockMapped || detectorView) && (
          <svg aria-hidden width={size.width} height={size.height} className="absolute inset-0 pointer-events-none">
            {boxes.map(({ track, at }) => {
              const x = offset.x + at.box.x * scale
              const y = offset.y + at.box.y * scale
              const w = at.box.w * scale
              const h = at.box.h * scale
              const text = `${track.letter ?? track.name}${at.interpolated ? ' · interpolated' : ''}`
              // N-19d (design review P2): the view badge covered the label of a face at the top left; the label then goes
              // under its box. N-19k: the badge itself stands in a corner without a face box
              const labelY = faceLabelTop({ x, y, h }, text.length * 6.7 + 8, badgeWidth(badge), badgeAt)
              return (
                <g key={track.trackId} data-box={track.letter ?? track.trackId} data-interpolated={at.interpolated}>
                  <rect x={x} y={y} width={w} height={h} fill="none" style={{ stroke: 'var(--color-lens-halo)', strokeWidth: 3 }} />
                  <rect x={x} y={y} width={w} height={h} fill="none" strokeDasharray={at.interpolated ? '4 3' : undefined} style={{ stroke: 'var(--color-lens-ink)', strokeWidth: 1.5 }} />
                  <rect data-testid="face-label" x={x} y={labelY} width={text.length * 6.7 + 8} height={16} rx={2} style={{ fill: 'var(--color-stage-material)' }} />
                  <text x={x + 4} y={labelY + 11.5} className="font-mono" style={{ fill: 'var(--color-stage-text)', fontSize: 11 }}>{text}</text>
                </g>
              )
            })}
          </svg>
        )}
        <p data-testid="view-badge" data-corner={corner} className={cx('absolute m-0 px-1.5 rounded-xs bg-stage-material font-mono text-micro text-stage-text whitespace-nowrap', CORNER_CLASS[corner])}>
          {badge}
        </p>
      </div>

      <div role="toolbar" aria-label="Transport" className="mt-2 flex items-center gap-1 flex-wrap min-h-8">
        <button type="button" className={transportButton} onClick={player.toggle} disabled={!src || !!player.error || !player.clockMapped} aria-label={player.playing ? 'Pause (Space)' : 'Play (Space)'} title={player.playing ? 'Pause (Space)' : 'Play (Space)'}>
          {player.playing ? <Pause size={16} strokeWidth={1.5} aria-hidden /> : <Play size={16} strokeWidth={1.5} aria-hidden />}
        </button>
        <button type="button" className={transportButton} onClick={() => player.step(-1)} aria-label="Previous frame (,)" title="Previous frame (, · Shift for 10)"><StepBack size={16} strokeWidth={1.5} aria-hidden /></button>
        <button type="button" className={transportButton} onClick={() => player.step(1)} aria-label="Next frame (.)" title="Next frame (. · Shift for 10)"><StepForward size={16} strokeWidth={1.5} aria-hidden /></button>
        <span className="ml-2 font-mono text-mono text-text num" data-testid="timecode">{timecode(t)}</span>
        <span className="ml-3 font-mono text-mono text-text num" data-testid="frame-readout">frame {frame} / {index.count}</span>
        <span className="ml-3 font-mono text-mono text-text-muted num">{fmtFps(index.fps)} fps {index.vfr ? 'VFR' : 'CFR'}</span>
        <span className="ml-auto flex items-center gap-1">
          <button type="button" className={transportButton} onClick={onPrevFlagged ?? undefined} disabled={!onPrevFlagged} title="Previous flagged segment ([)"><ChevronLeft size={14} strokeWidth={1.5} aria-hidden />prev flagged</button>
          <button type="button" className={transportButton} onClick={onNextFlagged ?? undefined} disabled={!onNextFlagged} title="Next flagged segment (])">next flagged<ChevronRight size={14} strokeWidth={1.5} aria-hidden /></button>
          <Menu ariaLabel="Playback rate" align="right" triggerClassName="h-8 px-2 font-mono num" label={`${player.rate}×`}
            items={RATES.map((r) => ({ id: String(r), label: `${r}×` }))} onSelect={(r) => player.setRate(Number(r))} />
        </span>
      </div>
      <Scrubber index={index} frame={frame} onFrame={player.seekFrame} />
      <p className="m-0 mt-1 text-micro text-text-muted min-h-4" data-testid="player-caption">
        {caption ?? ''}
        {!detectorView && nearest != null && nearest !== frame && <span>{caption ? ' · ' : ''}Nearest analysed frame: {timecode(index.times[nearest] ?? 0)} (frame {nearest}).</span>}
      </p>
    </div>
  )
}
