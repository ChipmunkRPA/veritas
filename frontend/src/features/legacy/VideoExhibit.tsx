import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, frameStep, type Analysis } from '../../lib/api'
import { fmtBytes, fmtSeconds, prob } from '../../lib/format'
import { cx } from '../../lib/cx'
import { Eyebrow } from '../../components/ui'
import { Plate, captionFor, keyFrameIndex, titleFor, useLightbox, type PlateItem } from './Plates'

type Row = { t: number; index: number; frame: number; face: number | null; general: number; faces: number }
const STEP_BG = ['bg-verdict-1', 'bg-verdict-2', 'bg-verdict-3', 'bg-verdict-4', 'bg-verdict-5'] as const
const tickStyle = { fontSize: 12, fontFamily: 'var(--font-mono)', fill: 'var(--color-stage-muted)' }

/** Stage budget: 16 + video 420 + timeline (32 header + 200 chart + strip) + meta line ≈ 712 px ≤ 720. */
const VIDEO_MAX = 420
const CHART_H = 200

/** Whole-second ticks from 0 to the rounded duration, at most ~7 of them, always ending on the last second. */
function secondTicks(end: number): number[] {
  if (!Number.isFinite(end) || end <= 0) return [0]
  const step = Math.max(1, Math.ceil(end / 6))
  const ticks: number[] = []
  for (let t = 0; t < end; t += step) ticks.push(t)
  if (ticks[ticks.length - 1] !== end) ticks.push(end)
  return ticks
}

function ChartTip({ active, payload, label }: { active?: boolean; payload?: { payload: Row }[]; label?: number }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const row = (k: string, v: number | null) => (
    <div className="flex justify-between gap-6"><span className="text-small text-text-muted">{k}</span><span className="font-mono text-mono text-text num">{prob(v, 3)}</span></div>
  )
  return (
    <div className="bg-surface border border-border rounded-md shadow-popover px-3 py-2 min-w-44">
      <div className="font-mono text-caption text-text-muted num">t = {Number(label).toFixed(2)} s · frame {d.index}{d.faces ? ` · ${d.faces} face${d.faces === 1 ? '' : 's'}` : ''}</div>
      {row('fused', d.frame)}{row('face-swap', d.face)}{row('whole-frame', d.general)}
    </div>
  )
}

function audioStatus(video: Record<string, unknown> | undefined): string {
  const info = (video as { audio_analysis?: { audio?: string; seconds?: number; p_synthetic?: number } } | undefined)?.audio_analysis
  if (!info) return 'n/a'
  if (info.audio === 'analysed') return `${info.seconds ?? '?'} s analysed, p(synthetic) ${info.p_synthetic != null ? info.p_synthetic.toFixed(2) : '—'}`
  return info.audio ?? 'n/a'
}

export default function VideoExhibit({ a, token = '', figIndex = {} }: { a: Analysis; token?: string; figIndex?: Record<string, number> }) {
  const r = a.result!
  const frames = r.frames ?? []
  const arts = r.artifacts ?? []
  const data: Row[] = useMemo(() => frames.map((f) => ({ t: +f.time.toFixed(2), index: f.index, frame: f.p_frame, face: f.p_face, general: f.p_general, faces: f.faces })), [frames])
  const videoRef = useRef<HTMLVideoElement>(null)
  const [time, setTime] = useState(0)
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const [series, setSeries] = useState({ frame: true, face: true, general: true })
  const duration = Number(r.video?.duration ?? (data.at(-1)?.t ?? 0))
  const domainEnd = Math.max(1, Math.ceil(duration))
  const ticks = useMemo(() => secondTicks(domainEnd), [domainEnd])
  // Poster only when the engine emitted one; otherwise the `#t=0.001` fragment still paints the first frame in most browsers.
  const poster = arts.includes('poster.jpg') ? api.artifactUrl(a.id, 'poster.jpg', token) : undefined

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const on = () => setTime(v.currentTime)
    v.addEventListener('timeupdate', on); v.addEventListener('seeked', on)
    return () => { v.removeEventListener('timeupdate', on); v.removeEventListener('seeked', on) }
  }, [])

  const seek = useCallback((t: number) => { const v = videoRef.current; if (v && Number.isFinite(t)) { v.currentTime = t; setTime(t) } }, [])

  // Per-frame crops (`keyframe_N.jpg`) render as a thumbnail strip with `t = …` captions taken from the frame record;
  // without them the composite contact sheet is shown cropped to its image strip (its lower half is unreadable captions).
  const perFrame: PlateItem[] = useMemo(() => arts
    .map((n) => ({ n, k: keyFrameIndex(n) }))
    .filter((x): x is { n: string; k: number } => x.k != null)
    .sort((x, y) => x.k - y.k)
    .map(({ n, k }) => {
      const f = frames.find((fr) => fr.index === k) ?? frames[k]
      const caption = f ? `t = ${f.time.toFixed(1)} s · fused ${prob(f.p_frame)}` : captionFor(n)
      return { name: n, src: api.artifactUrl(a.id, n, token), title: f ? `Frame ${f.index}` : titleFor(n), caption, n: figIndex[n] }
    }), [arts, frames, a.id, token, figIndex])
  const composite: PlateItem[] = useMemo(() => arts.filter((n) => n === 'key_frames.jpg').map((n) => ({ name: n, src: api.artifactUrl(a.id, n, token), title: titleFor(n), caption: captionFor(n), n: figIndex[n] })), [arts, a.id, token, figIndex])
  const keyFrames = perFrame.length ? perFrame : composite
  const lb = useLightbox(keyFrames)

  const legend = (k: keyof typeof series, label: string, dash: string) => (
    <button type="button" aria-pressed={series[k]} onClick={() => setSeries((s) => ({ ...s, [k]: !s[k] }))} title={`Toggle the ${label} series`}
      className={cx('inline-flex items-center gap-2 text-small text-stage-text rounded-sm px-1 min-h-8 transition-opacity duration-fast', !series[k] && 'opacity-40')}>
      <svg width="20" height="4" aria-hidden><line x1="0" y1="2" x2="20" y2="2" strokeWidth={k === 'frame' ? 2 : 1.5} strokeDasharray={dash} stroke={k === 'frame' ? 'var(--color-stage-text)' : k === 'face' ? 'var(--color-stage-series-face)' : 'var(--color-stage-series-general)'} /></svg>
      {label}
    </button>
  )

  return (
    <section aria-label="Exhibit" data-tour="exhibit">
      <Eyebrow as="h2">Exhibit</Eyebrow>
      <div className="stage mt-3 rounded-md bg-stage text-stage-text border border-stage-border flex flex-col">
        <div className="flex items-center justify-center p-4 pb-0">
          {/* `#t=0.001` media fragment makes browsers paint the first frame while paused (fragment is never sent to the server). */}
          <video ref={videoRef} src={`${api.sourceUrl(a.id, token)}#t=0.001`} poster={poster} controls preload="metadata" playsInline
            className="block w-full rounded-[2px]" style={{ maxHeight: VIDEO_MAX, objectFit: 'contain' }} />
        </div>

        {data.length > 0 && (
          <div className="px-4 pt-2" data-tour="timeline">
            <div className="flex items-center justify-between flex-wrap gap-2 px-2 min-h-8">
              <Eyebrow as="h3" className="text-stage-muted">Timeline</Eyebrow>
              <div className="flex items-center gap-4">{legend('frame', 'fused', '0')}{legend('face', 'face-swap', '6 3')}{legend('general', 'whole-frame', '2 4')}</div>
            </div>
            <div style={{ height: CHART_H }} title="Click to seek the video; the playhead follows playback" aria-label="Per-frame detector scores over time; click to seek">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}
                  onClick={(s) => { const l = (s as { activeLabel?: unknown } | null)?.activeLabel; if (l != null) seek(Number(l)) }}
                  onMouseMove={(s) => { const i = (s as { activeTooltipIndex?: number } | null)?.activeTooltipIndex; setHoverIdx(typeof i === 'number' ? i : null) }}
                  onMouseLeave={() => setHoverIdx(null)} style={{ cursor: 'pointer' }}>
                  <CartesianGrid horizontal vertical={false} stroke="var(--color-stage-border)" />
                  <ReferenceArea y1={0.5} y2={1} fill="var(--color-stage-series-face)" fillOpacity={0.07} />
                  <XAxis dataKey="t" type="number" domain={[0, domainEnd]} ticks={ticks} tick={tickStyle} axisLine={false} tickLine={false} tickFormatter={(s: number) => `${Math.round(s)}s`} />
                  <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tick={tickStyle} axisLine={false} tickLine={false} width={36} />
                  <Tooltip content={<ChartTip />} cursor={{ stroke: 'var(--color-stage-muted)' }} isAnimationActive={false} />
                  <ReferenceLine y={0.5} stroke="var(--color-stage-muted)" strokeDasharray="2 4" label={{ value: 'threshold', position: 'insideTopLeft', fontSize: 11, fill: 'var(--color-stage-muted)', fontFamily: 'var(--font-mono)' }} />
                  {series.frame && <Line type="linear" dataKey="frame" name="fused" stroke="var(--color-stage-text)" strokeWidth={2} dot={false} isAnimationActive={false} activeDot={{ r: 3, fill: 'var(--color-stage-text)', stroke: 'var(--color-stage)', strokeWidth: 2 }} />}
                  {series.face && <Line type="linear" dataKey="face" name="face-swap" stroke="var(--color-stage-series-face)" strokeWidth={1.5} strokeDasharray="6 3" dot={false} connectNulls isAnimationActive={false} activeDot={{ r: 3, fill: 'var(--color-stage-series-face)', stroke: 'var(--color-stage)', strokeWidth: 2 }} />}
                  {series.general && <Line type="linear" dataKey="general" name="whole-frame" stroke="var(--color-stage-series-general)" strokeWidth={1.5} strokeDasharray="2 4" dot={false} isAnimationActive={false} activeDot={{ r: 3, fill: 'var(--color-stage-series-general)', stroke: 'var(--color-stage)', strokeWidth: 2 }} />}
                  <ReferenceLine x={time} stroke="var(--color-stage-text)" strokeWidth={1} ifOverflow="extendDomain" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {/* Band strip: one segment per frame, coloured by its band (direction A §4.7). */}
            <div className="ml-7 mr-3 mt-1 flex h-2 gap-px" role="img" aria-label="Per-frame band strip; colour indicates where the fused score falls on the verdict scale">
              {data.map((d, i) => {
                const step = frameStep(d.frame)
                return (
                  <button key={d.index} type="button" tabIndex={-1} aria-label={`Frame ${d.index} at ${d.t} seconds, fused ${prob(d.frame)}`} onClick={() => seek(d.t)}
                    className={cx('flex-1 min-w-0 transition-[box-shadow] duration-fast', STEP_BG[step - 1], step === 3 && 'opacity-50', hoverIdx === i && 'shadow-[0_0_0_1px_var(--color-stage-text)] opacity-100 relative z-10')} style={{ borderRadius: 1 }} />
                )
              })}
            </div>
          </div>
        )}

        <p className="px-4 py-2 text-right font-mono text-caption text-stage-muted num">
          {r.video?.width ? `${r.video.width} × ${r.video.height}` : ''}{r.video?.fps ? ` @ ${Number(r.video.fps).toFixed(2)} fps` : ''} · {fmtSeconds(duration)} · {frames.length} frames analysed · {fmtBytes(a.size_bytes)} · audio {audioStatus(r.video)}
        </p>
      </div>
      <p className="mt-2 text-caption text-text-muted">Exhibit — Source video with per-frame detector scores{data.length ? ' and the fused band per frame' : ''}.</p>

      {keyFrames.length > 0 && (
        <div className="mt-8" data-tour="keyframes">
          <Eyebrow as="h3">Key frames</Eyebrow>
          {perFrame.length ? (
            <div className="mt-3 flex gap-3 overflow-x-auto pb-2" style={{ scrollSnapType: 'x mandatory' }}>
              {perFrame.map((k, i) => <Plate key={k.name} item={k} height={120} onOpen={() => lb.open(i)} className="shrink-0" />)}
            </div>
          ) : (
            <div className="mt-3 max-w-[760px]">
              {/* The composite's lower half is caption text too small to read at this width; show only the image strip. */}
              <figure className="m-0">
                <button type="button" onClick={() => lb.open(0)} aria-label={`Open ${composite[0].title} in viewer`} className="block w-full rounded-md text-left">
                  <span className="block h-[180px] overflow-hidden rounded-md border border-border bg-surface-2">
                    <img src={composite[0].src} alt={`${composite[0].title}. ${composite[0].caption}`} loading="lazy" className="w-full h-full object-cover object-top" />
                  </span>
                </button>
                <figcaption className="mt-2">
                  <span className="block text-small font-medium text-text">{composite[0].n ? <span className="num">Fig. {composite[0].n} — </span> : null}{composite[0].title}</span>
                  <span className="block text-caption text-text-muted">{composite[0].caption}; open the viewer for per-frame times and scores</span>
                </figcaption>
              </figure>
            </div>
          )}
          {lb.node}
        </div>
      )}
    </section>
  )
}
