import { SYNC_RANGE_MS, SYNC_TOLERANCE_MS } from './lanesCanvas'
import type { LaneModel } from './model'

const W = 288
const H = 132
const LEFT = 36
const STRIP = 4
const PLOT_H = H - 22

/** Sync-offset plot (UX §5.10 lip-sync lane, §8.4): the detector's per-window audio-video offset in ms over ±200 ms with the display
 *  tolerance band, out-of-tolerance runs thickened, confidence as a strip under the curve, and the playhead. Click to seek. */
export default function SyncOffsetPlot({ lane, time, duration, onSeek }: { lane: LaneModel; time: number | null; duration: number; onSeek: (t: number) => void }) {
  const curve = lane.curve ?? []
  const x = (t: number) => LEFT + (duration > 0 ? (t / duration) * (W - LEFT - 4) : 0)
  const y = (ms: number) => 6 + (1 - (Math.max(-SYNC_RANGE_MS, Math.min(SYNC_RANGE_MS, ms)) + SYNC_RANGE_MS) / (2 * SYNC_RANGE_MS)) * (PLOT_H - 12)
  const worst = curve.reduce((m, c) => (Math.abs(c.ms) > Math.abs(m) ? c.ms : m), 0)
  return (
    <figure className="m-0" data-testid="sync-offset-plot">
      <svg width={W} height={H} role="img" className="block cursor-pointer" aria-label={`Sync offset for ${lane.detail.replace(' · offset ms', '')}: ${curve.length} windows, largest offset ${Math.round(worst)} ms`}
        onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); onSeek(Math.max(0, Math.min(duration, ((e.clientX - r.left - LEFT) / (W - LEFT - 4)) * duration))) }}>
        <rect x={LEFT} y={y(SYNC_TOLERANCE_MS)} width={W - LEFT - 4} height={y(-SYNC_TOLERANCE_MS) - y(SYNC_TOLERANCE_MS)} style={{ fill: 'var(--color-lane-tolerance)' }} />
        {[SYNC_RANGE_MS, 0, -SYNC_RANGE_MS].map((v) => (
          <g key={v}>
            <text x={LEFT - 6} y={y(v)} textAnchor="end" dominantBaseline="central" className="font-mono text-micro" style={{ fill: 'var(--color-text-muted)' }}>{v > 0 ? `+${v}` : v === 0 ? '0' : `−${-v}`}</text>
            <line x1={LEFT} x2={W - 4} y1={y(v)} y2={y(v)} strokeDasharray={v === 0 ? '2 3' : undefined} style={{ stroke: 'var(--color-graphic)', strokeWidth: v === 0 ? 1 : 0.5 }} />
          </g>
        ))}
        {curve.slice(1).map((b, i) => {
          const a = curve[i]
          const out = Math.abs(a.ms) > SYNC_TOLERANCE_MS || Math.abs(b.ms) > SYNC_TOLERANCE_MS
          return <line key={b.t} x1={x(a.t)} y1={y(a.ms)} x2={x(b.t)} y2={y(b.ms)} data-out={out || undefined} style={{ stroke: 'var(--color-text)', strokeWidth: out ? 2.5 : 1.5 }} />
        })}
        {curve.map((c) => c.confidence != null && <rect key={`c-${c.t}`} x={x(c.start)} y={PLOT_H} width={Math.max(1, x(c.end) - x(c.start))} height={STRIP} style={{ fill: 'var(--color-text)', opacity: Math.max(0, Math.min(1, c.confidence)) }} />)}
        <text x={LEFT} y={H - 4} className="font-mono text-micro" style={{ fill: 'var(--color-text-muted)' }}>0s</text>
        <text x={W - 4} y={H - 4} textAnchor="end" className="font-mono text-micro" style={{ fill: 'var(--color-text-muted)' }}>{duration.toFixed(1)}s</text>
        {time != null && <line x1={x(time)} x2={x(time)} y1={2} y2={PLOT_H + STRIP} style={{ stroke: 'var(--color-text)', strokeWidth: 1 }} />}
      </svg>
      <figcaption className="mt-1 text-micro text-text-muted">
        Offset in ms per window; above zero the sound is later than the mouth. The band at ±{SYNC_TOLERANCE_MS} ms is a reading aid, not a detector threshold; the strip under the curve is the detector's confidence.
      </figcaption>
    </figure>
  )
}
