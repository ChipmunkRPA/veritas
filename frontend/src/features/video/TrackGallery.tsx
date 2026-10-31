import { useEffect, useRef } from 'react'
import { cx } from '../../lib/cx'
import { nearestFrame } from '../../lib/pts'
import { shortTime } from './SegmentList'
import type { StillItem, TrackItem } from './model'

const THUMB = 64
const MARGIN = 0.25

/** A 64 px face crop drawn from the analysed still nearest the track's first detected frame. */
function Thumbnail({ track, stills, stillUrl, mediaWidth }: { track: TrackItem; stills: readonly StillItem[]; stillUrl: (s: StillItem) => string | null; mediaWidth: number }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const source = track.frames.find((f) => f.det_score != null) ?? track.frames[0]
  const nearest = nearestFrame(stills.map((s) => s.frame).filter((k) => k >= track.firstFrame && k <= track.lastFrame), source.frame_index)
  const still = nearest == null ? null : stills.find((s) => s.frame === nearest) ?? null
  const src = still ? stillUrl(still) : null
  const box = still ? track.frames.reduce((best, f) => (Math.abs(f.frame_index - still.frame) < Math.abs(best.frame_index - still.frame) ? f : best), source).bbox : null
  useEffect(() => {
    const c = canvas.current
    const ctx = c?.getContext('2d')
    if (!c || !ctx || !src || !box) return
    let live = true
    const img = new Image()
    img.onload = () => {
      if (!live) return
      const s = mediaWidth > 0 ? img.naturalWidth / mediaWidth : 1
      const side = Math.max(box.w, box.h) * (1 + 2 * MARGIN)
      const cx0 = (box.x + box.w / 2 - side / 2) * s
      const cy0 = (box.y + box.h / 2 - side / 2) * s
      ctx.clearRect(0, 0, THUMB, THUMB)
      ctx.drawImage(img, cx0, cy0, side * s, side * s, 0, 0, THUMB, THUMB)
    }
    img.src = src
    return () => { live = false }
  }, [src, box, mediaWidth])
  return <canvas ref={canvas} width={THUMB} height={THUMB} aria-hidden className="block shrink-0 rounded-xs bg-stage" style={{ width: THUMB, height: THUMB }} />
}

/** Face tracks (ARCHITECTURE §7.5 TrackGallery): thumbnail, screen time and the highest face-lane probability. Selecting a track isolates
 *  its lanes and its boxes; selecting it again shows every track. */
export default function TrackGallery({ tracks, stills, stillUrl, mediaWidth, isolated, onIsolate, inspectionOnly = false }: {
  tracks: readonly TrackItem[]
  inspectionOnly?: boolean
  stills: readonly StillItem[]
  stillUrl: (s: StillItem) => string | null
  mediaWidth: number
  isolated: string | null
  onIsolate: (trackId: string | null) => void
}) {
  if (!tracks.length) return <p className="m-0 text-small text-text-muted">No face was tracked in this video.</p>
  return (
    <ul className="m-0 p-0 list-none flex flex-col gap-2" data-testid="track-gallery">
      {tracks.map((t) => (
        <li key={t.trackId}>
          <button type="button" data-testid="track-row" data-track={t.letter ?? t.trackId} aria-pressed={isolated === t.trackId} onClick={() => onIsolate(isolated === t.trackId ? null : t.trackId)}
            className={cx('w-full flex items-center gap-3 p-1.5 rounded-sm text-left transition-colors duration-fast', isolated === t.trackId ? 'bg-surface-3' : 'hover:bg-surface-3')}>
            <Thumbnail track={t} stills={stills} stillUrl={stillUrl} mediaWidth={mediaWidth} />
            <span className="min-w-0">
              <span className="block text-small font-medium text-text">{t.name}</span>
              <span className="block font-mono text-micro text-text-muted num">{shortTime(t.start)}–{shortTime(t.end)} · {(t.end - t.start).toFixed(1)} s</span>
              <span className="block text-micro text-text-muted">{inspectionOnly ? 'For inspection only' : t.peak != null ? <>face-lane peak <span className="font-mono num text-text">{t.peak.toFixed(2)}</span> · uncalibrated</> : 'no face-lane score'} · {t.detections} detections</span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  )
}
