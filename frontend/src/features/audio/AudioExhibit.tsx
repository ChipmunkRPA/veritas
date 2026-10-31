import { useEffect, useMemo, useRef, useState } from 'react'
import { Pause, Play } from 'lucide-react'
import { api, hasApiKey } from '../../lib/api'
import { useAnnotations, useExhibitPayloads, useExhibits, type AnalysisDetail, type Annotation, type Finding } from '../../lib/api/queries'
import { timecode } from '../../lib/pts'
import { Disclosure, ErrorState, HeatLegend, SkeletonRows } from '../../components/ui'
import SegmentList from '../video/SegmentList'
import TimelineLanes from '../video/TimelineLanes'
import type { VideoSegment } from '../video/model'
import { exhibitUrl } from '../viewer/useBase'
import { buildAudioModel, describeFacts, describeShares, type AudioFacts } from './model'

const STEP_S = 1
const NO_ANNOTATIONS: Annotation[] = []

const tool = 'h-8 min-w-8 px-2 inline-flex items-center justify-center gap-1.5 rounded-sm text-small text-text-muted enabled:hover:text-text enabled:hover:bg-surface-3 transition-colors duration-fast disabled:opacity-40 disabled:cursor-not-allowed'

/** Exhibit of an audio analysis on the result page (UX §5.10 lane 4; PRD DC-05, EA-05; BUILD_SPEC A-1): playback of the original, the
 *  recording facts and content shares, the flagged segments, the voice lane with music, noise and silence hatched, and the content
 *  runs. The original loads only when played (`preload="none"`), so opening the page records no `original_viewed` event. */
export default function AudioExhibit({ analysis, facts, findings, token }: { analysis: AnalysisDetail; facts: AudioFacts | null; findings: readonly Finding[]; token: string }) {
  const exhibits = useExhibits(analysis.id)
  const annotations = useAnnotations(analysis.id)
  const ids = useMemo(() => (exhibits.data ?? []).filter((x) => x.kind === 'series' || x.kind === 'segments' || x.kind === 'table').map((x) => x.id), [exhibits.data])
  const payloads = useExhibitPayloads(analysis.id, ids, exhibits.isSuccess)
  const payloadKey = payloads.data.map((p) => p.id).join(',')
  const annotationData = annotations.data ?? NO_ANNOTATIONS
  const spectrogramId = (exhibits.data ?? []).find((x) => x.kind === 'image' && x.role === 'spectrum' && x.media_ref === 'audio')?.id ?? null
  const spectrogramUrl = spectrogramId && (!hasApiKey || token !== '') ? exhibitUrl(spectrogramId, 'payload', token) : null
  const model = useMemo(
    () => (exhibits.data && !payloads.isPending ? buildAudioModel({ facts, payloads: payloads.data, annotations: annotationData, spectrogramUrl }) : null),
    // payloads.data is a fresh array on every render; payloadKey identifies it
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [exhibits.data, payloadKey, payloads.isPending, annotationData, facts, spectrogramUrl],
  )

  const audio = useRef<HTMLAudioElement>(null)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [playError, setPlayError] = useState<string | null>(null)
  const [selected, setSelected] = useState<VideoSegment | null>(null)
  useEffect(() => {
    if (!playing) return
    let raf = 0
    const tick = () => {
      if (audio.current) setTime(audio.current.currentTime)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing])

  const heading = <h2 id="exhibit-heading" className="m-0 eyebrow">Exhibit</h2>
  const error = exhibits.error ?? payloads.error ?? null
  if (error) return <section id="section-exhibit" className="scroll-mt-14" aria-labelledby="exhibit-heading">{heading}<ErrorState className="mt-4" onRetry={() => { void exhibits.refetch(); payloads.refetch() }}>{error.message}</ErrorState></section>
  if (!model) return <section id="section-exhibit" className="scroll-mt-14" aria-labelledby="exhibit-heading">{heading}<SkeletonRows className="mt-4" /></section>

  const duration = model.duration
  const tokenReady = !hasApiKey || token !== ''
  const seek = (t: number) => {
    const next = Math.max(0, Math.min(duration, t))
    if (audio.current) audio.current.currentTime = next
    setTime(next)
  }
  const toggle = () => {
    const el = audio.current
    if (!el) return
    if (!el.paused) { el.pause(); return }
    setPlayError(null)
    el.currentTime = time >= duration ? 0 : time
    void el.play().catch(() => setPlaying(false))
  }
  const pick = (s: VideoSegment | null) => {
    setSelected(s)
    if (s) seek(s.start)
  }
  const shares = describeShares(facts?.shares)
  const factLine = describeFacts(facts)
  const rows: [string, React.ReactNode, string?][] = [
    ['Recording', <span className="font-mono text-mono num">{factLine ?? 'not recorded'}</span>, 'audio-facts'],
    ['Content', shares || 'not recorded', 'audio-content'],
    ['Analysed', facts?.seconds_analysed != null ? `${facts.seconds_analysed.toFixed(1)} s as 16 kHz mono (at most 600 s)` : 'not recorded'],
  ]
  if (facts?.assessable === false) rows.push(['Voice checks', `Not assessable. ${facts.not_assessable_reason ?? 'Less than 1 s of speech was found.'}`, 'audio-not-assessable'])

  return (
    <section id="section-exhibit" data-testid="audio-exhibit" aria-labelledby="exhibit-heading" className="scroll-mt-14">
      {heading}
      <div className="mt-3 rounded-md bg-surface-2 border border-border p-4 grid gap-6 min-[960px]:grid-cols-[minmax(0,1fr)_300px]">
        <div className="min-w-0">
          <div role="group" aria-label="Playback" className="flex items-center gap-1 flex-wrap">
            <button type="button" className={`${tool} text-text`} onClick={toggle} disabled={!tokenReady}>
              {playing ? <Pause size={14} strokeWidth={1.5} aria-hidden /> : <Play size={14} strokeWidth={1.5} aria-hidden />}
              {playing ? 'Pause' : 'Play'}
            </button>
            <button type="button" className={tool} onClick={() => seek(time - STEP_S)} aria-label="Back 1 second">−1 s</button>
            <button type="button" className={tool} onClick={() => seek(time + STEP_S)} aria-label="Forward 1 second">+1 s</button>
            <span data-testid="audio-timecode" className="ml-3 font-mono text-mono text-text num">{timecode(time)} / {timecode(duration)}</span>
          </div>
          {playError && <p className="mt-2 mb-0 text-caption text-danger" role="alert">{playError}</p>}
          <dl className="mt-4 mb-0 grid grid-cols-[120px_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-small max-[639px]:grid-cols-1 max-[639px]:gap-y-0.5">
            {rows.map(([key, value, testId]) => (
              <div key={key} className="contents">
                <dt className="text-text-muted">{key}</dt>
                <dd data-testid={testId} className="m-0 text-text min-w-0 break-words max-[639px]:mb-1">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="min-w-0">
          <h3 className="m-0 eyebrow">Flagged segments</h3>
          <div className="mt-2">
            <SegmentList analysisId={analysis.id} segments={model.segments} selected={selected?.key ?? null} findings={findings} onSelect={pick}
              emptyText="No voice window reached the segment threshold (0.60, held above 0.45 for at least 0.5 s)." />
          </div>
        </div>
      </div>
      <TimelineLanes className="mt-6" medium="recording" model={model} time={time} selected={selected} onSeek={seek} onSelectSegment={pick} />
      {/* N-19d (design review P2): the spectrogram is drawn in the ramp of the evidence maps; its legend says that it shows
          loudness in dB, not a score */}
      {model.lanes.some((l) => l.kind === 'spectrogram') && (
        <HeatLegend testId="spectrogram-legend" className="mt-3" showUnreliable={false} ticks={['−80 dB', '−40 dB', '0 dB']}
          title="Spectrogram: loudness below the loudest cell of this recording. It is not a detector output." />
      )}
      {model.content.length > 0 && (
        <Disclosure className="mt-6" testId="content-runs" label="Content runs" summary={`${model.content.length} run${model.content.length === 1 ? '' : 's'}`}>
          <table className="w-full border-collapse text-small">
            <caption className="sr-only">Each second of the recording labelled speech, music, noise or silence; consecutive seconds with the same label merged</caption>
            <thead>
              <tr className="text-left text-text-muted border-b border-border">
                <th scope="col" className="py-1.5 pr-4 font-medium">Start</th><th scope="col" className="py-1.5 pr-4 font-medium">End</th>
                <th scope="col" className="py-1.5 pr-4 font-medium">Content</th><th scope="col" className="py-1.5 font-medium">Voice detectors</th>
              </tr>
            </thead>
            <tbody>
              {model.content.map((r) => (
                <tr key={`${r.start}-${r.end}`} className="border-b border-border">
                  <td className="py-1 pr-4 font-mono text-mono num">{timecode(r.start)}</td><td className="py-1 pr-4 font-mono text-mono num">{timecode(r.end)}</td>
                  <td className="py-1 pr-4 text-text">{r.label}</td><td className="py-1 text-text-muted">{r.scored ? 'scored' : 'not scored'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Disclosure>
      )}
      <p className="mt-2 mb-0 text-caption text-text-muted max-w-[80ch]">
        Exhibit — Source recording: playback of the original, the voice lane with music, noise and silence hatched, and the flagged segments.
        Each second is labelled by a heuristic classifier; only windows at least half inside speech were scored by the voice detectors.
        {model.lanes.some((l) => l.kind === 'waveform') && ' The waveform and the spectrogram show the analysed 16 kHz mono signal itself; they are not detector outputs.'}
      </p>
      {tokenReady && (
        <audio ref={audio} src={api.sourceUrl(analysis.id, token)} preload="none" aria-label="Source recording"
          onPlay={() => setPlaying(true)}
          onPause={() => { setPlaying(false); if (audio.current) setTime(audio.current.currentTime) }}
          onError={() => { setPlaying(false); setPlayError('The recording could not be played in this browser. The analysis used its own decoder (ffmpeg).') }} />
      )}
    </section>
  )
}
