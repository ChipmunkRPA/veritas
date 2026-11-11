import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Button } from '../../components/ui'
import AudioInspectionTimeline from './AudioInspectionTimeline'
import AudioBackgroundContext from './AudioBackgroundContext'
import AudioSoundEvents from './AudioSoundEvents'
import AudioStereoContext from './AudioStereoContext'
import { EvidenceAudio, EvidenceLimits, EvidenceMeasurements, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import { measurementValue } from './imagePackets'
import { rationalSeconds, seconds, type AudioEvidencePacket, type AudioEvidenceSet } from './multimediaPackets'

export default function AudioEvidenceContent({ set, token, mediaReady }: { set: AudioEvidenceSet; token: string; mediaReady: boolean }) {
  const rate = set.source.sample_rate
  const { hash } = useLocation()
  const [selection, setSelection] = useState(set.packets[0]?.id ?? null)
  const root = useRef<HTMLDivElement>(null)
  useEffect(() => {
    // Keep the node for cleanup: React clears root.current when the video
    // channel is unmounted. Also catches players added after media access loads.
    const node = root.current
    return () => node?.querySelectorAll('audio').forEach((audio) => { if (!audio.paused) audio.pause() })
  }, [])
  const selectedId = set.packets.some((packet) => packet.id === selection) ? selection : set.packets[0]?.id ?? null
  const pauseAudio = () => root.current?.querySelectorAll('audio').forEach((audio) => { if (!audio.paused) audio.pause() })
  const select = (packet: AudioEvidencePacket) => {
    // A hidden clip must not keep playing after a different interval is chosen.
    pauseAudio()
    setSelection(packet.id)
    requestAnimationFrame(() => {
      const article = Array.from(root.current?.querySelectorAll('article') ?? []).find((element) => element.id === `media-evidence-${packet.id}`)
      if (article && !article.hidden) {
        article.focus({ preventScroll: true })
        article.scrollIntoView?.({ block: 'start' })
      }
    })
  }
  useEffect(() => {
    const packet = set.packets.find((item) => hash === `#media-evidence-${encodeURIComponent(item.id)}`)
    if (!packet) return
    root.current?.querySelectorAll('audio').forEach((audio) => { if (!audio.paused) audio.pause() })
    setSelection(packet.id)
  }, [hash, set.packets])
  return <div ref={root} data-testid="audio-evidence-content">
    <AudioSoundEvents set={set} token={token} mediaReady={mediaReady} pauseAudio={pauseAudio} />
    <AudioBackgroundContext set={set} token={token} mediaReady={mediaReady} pauseAudio={pauseAudio} />
    <AudioStereoContext set={set} />
    {set.packets.length > 0 && <AudioInspectionTimeline set={set} token={token} mediaReady={mediaReady} selectedId={selectedId} onSelect={select} />}
    {set.coverage && rate ? <div className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <p className="m-0">Voice checks cover {seconds(set.coverage.scored_samples / rate)} of the {seconds(set.coverage.decoded_samples / rate)} decoded input.
        {' '}{seconds(set.coverage.unscored_samples / rate)} of that input has no window score. Overlapping windows are counted once in this coverage.</p>
      <p className="mt-2 mb-0">{set.coverage.windows.length} scored windows. Responses apply to complete windows, normally four seconds; they do not identify individual words or exact voice-replacement boundaries.</p>
      {set.coverage.decode_limit_reached && <p className="mt-2 mb-0">The decoding limit was reached. Audio beyond this input was not examined.</p>}
      <p className="mt-2 mb-0">Clips reproduce the exact analyzed {measurementValue(rate)} Hz mono input. The source had {set.source.source_channels ?? 'unrecorded'} channel(s); this is a derived listening copy.</p>
    </div> : <p className="mt-3 text-small text-text-muted">Scored audio coverage is unavailable. Unexamined audio is not a negative finding.</p>}
    {set.overview && set.packets.length === 0 && <div className={`${evidenceGrid} mt-4`}>
      {(['waveform', 'scores'] as const).map((role) => set.overview?.[role] && <figure key={role} className="m-0 min-w-0">
        <EvidencePicture set={set} artifact={set.overview[role]} token={token} ready={mediaReady} chart alt={role === 'waveform' ? 'Waveform of the analyzed audio input' : 'Recorded voice-check window responses over the analyzed input'} />
        <figcaption className="mt-2 text-caption text-text-muted">{role === 'waveform' ? 'Signal shape provides listening context; it does not prove synthesis.' : 'Recorded window responses. Gaps have no score; scores are not probabilities of editing.'}</figcaption>
      </figure>)}
    </div>}
    <div className="mt-6 space-y-7">{set.packets.map((packet) => <article key={packet.id} id={`media-evidence-${packet.id}`} hidden={packet.id !== selectedId} tabIndex={-1} className={evidenceCard}
      aria-label={`Selected audio evidence ${packet.id}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-body font-medium"><span className="font-mono text-accent mr-3">{packet.id}</span>{packet.title}</h3>
        {set.packets.length > 1 && <Button variant="text" onClick={() => {
          const navigator = root.current?.querySelector<HTMLElement>('[aria-label="Audio interval navigator"]')
          navigator?.querySelector<HTMLElement>('button[aria-pressed="true"]')?.focus({ preventScroll: true })
          navigator?.scrollIntoView?.({ block: 'start' })
        }}>Choose another interval</Button>}
      </div>
      <p className="mt-2 mb-0 text-body">{packet.observation}</p>
      <p className="mt-2 mb-0 font-mono text-caption text-text-muted">Analyzed input: {seconds(packet.start_s)}–{seconds(packet.end_s)} · samples [{packet.interval.start_sample}, {packet.interval.end_sample})</p>
      <figure className="m-0 mt-4">
        <EvidencePicture set={set} artifact={packet.artifacts.waveform} token={token} ready={mediaReady} chart alt={`${packet.id}: waveform with the selected window marked`} />
        <figcaption className="mt-2 text-caption text-text-muted">The highlighted interval is the exact scored window selected above. The waveform is source context; it does not identify a splice or prove synthesis.</figcaption>
      </figure>
      <div className={`${evidenceGrid} mt-4`}>
        <EvidenceAudio set={set} artifact={packet.artifacts.selected_clip} token={token} ready={mediaReady} alt={`${packet.id}: exact scored window`} />
        <EvidenceAudio set={set} artifact={packet.artifacts.context_clip} token={token} ready={mediaReady} alt={`${packet.id}: window with surrounding audio`} />
      </div>
      <p className="mt-2 mb-0 text-caption text-text-muted">Context samples [{packet.context_interval.start_sample}, {packet.context_interval.end_sample}); surrounding audio is listening context, not an additional confirmation.</p>
      <p className="mt-4 mb-0 text-small">{packet.interpretation}</p>
      <EvidenceMeasurements values={packet.observations.map((observation) => ({ ...observation,
        label: `${observation.signal_refs.map((ref) => ref.display_id ?? 'Check').join(', ')} · ${observation.label}` }))} />
      <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">Responses around this interval</summary>
        <div className="mt-3"><EvidencePicture set={set} artifact={packet.artifacts.scores} token={token} ready={mediaReady} chart
          alt={`${packet.id}: recorded responses for the selected window and nearby windows`} /></div>
      </details>
      <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">Original recording timestamps</summary>
        <p className="mt-2 mb-0">These intervals preserve the source clock, including gaps. The clip controls start at zero within each exported clip.</p>
        <ul className="pl-4 list-disc">{packet.anchor.source_timestamp_ranges.map((range, i) => {
          const start = rationalSeconds(range.source_start_pts, range.time_base), end = rationalSeconds(range.source_end_pts, range.time_base)
          return <li key={i}>Samples [{range.start_sample}, {range.end_sample}): {start !== null && end !== null ? `${seconds(start)}–${seconds(end)}` : 'time conversion unavailable'}
            {' '}({range.source_start_pts}–{range.source_end_pts} ticks, time base {range.time_base})</li>
        })}</ul>
      </details>
      <EvidenceLimits lines={packet.limitations} /><EvidenceReferences packet={packet} />
    </article>)}</div>
  </div>
}
