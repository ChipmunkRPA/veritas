import { useEffect, useRef, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { Button } from '../../components/ui'
import AudioInspectionTimeline from './AudioInspectionTimeline'
import AudioBackgroundContext from './AudioBackgroundContext'
import AudioSoundEvents from './AudioSoundEvents'
import AudioStereoContext from './AudioStereoContext'
import AudioRepetitionContext from './AudioRepetitionContext'
import { EvidenceAudio, EvidenceLimits, EvidenceMeasurements, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import { measurementValue } from './imagePackets'
import { rationalSeconds, seconds, type AudioEvidencePacket, type AudioEvidenceSet } from './multimediaPackets'

const longVoiceLabels = {
  ai: 'Leans AI-generated in this window',
  original: 'Leans original in this window',
  inconclusive: 'Inconclusive window',
  unassessed: 'No usable speech response',
}
const longVoiceSampleRate = 16_000

function SourceTimestamps({ ranges, videoOriginS, showSeconds = false }: {
  ranges: AudioEvidencePacket['anchor']['source_timestamp_ranges']; videoOriginS?: number | null; showSeconds?: boolean
}) {
  return <>{ranges.map((range) => {
    const start = rationalSeconds(range.source_start_pts, range.time_base)
    const end = rationalSeconds(range.source_end_pts, range.time_base)
    return <p key={range.start_sample} className="m-0 text-caption">
      {showSeconds && start != null && end != null && <>Original audio {seconds(start)}–{seconds(end)}. </>}
      Source PTS [{range.source_start_pts}, {range.source_end_pts}) at {range.time_base}.
      {videoOriginS != null && start != null && end != null && start >= videoOriginS
        && ` Video playback ${seconds(start - videoOriginS)}–${seconds(end - videoOriginS)}.`}
    </p>
  })}</>
}

export default function AudioEvidenceContent({ set, token, mediaReady, videoOriginS, standalone = true }: {
  set: AudioEvidenceSet; token: string; mediaReady: boolean; videoOriginS?: number | null; standalone?: boolean
}) {
  const rate = set.source.sample_rate
  const tracks = set.source.audio_stream_indices
  const clock = set.coverage?.source_clock_context
  const shortReview = standalone ? set.short_voice_review : undefined
  const longReview = standalone ? set.long_voice_review : undefined
  const intervals = standalone && set.packets.length === 0 ? [...new Map(set.coverage?.windows?.map((window) =>
    [`${window.start_sample}:${window.end_sample}`, window] as const)).values()]
    .sort((a, b) => a.start_sample - b.start_sample || a.end_sample - b.end_sample) : []
  const stereo = set.stereo_context
  const stereoOverlaps = set.coverage?.stereo_attenuation_overlaps?.flatMap((overlap) => {
    const measurement = stereo?.windows[overlap.stereo_window_index]?.downmix
    return measurement?.strong_attenuation ? [{ overlap, measurement }] : []
  }) ?? []
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
  const pauseAudio = (playing?: HTMLAudioElement) => root.current?.querySelectorAll('audio').forEach((audio) => {
    if (audio !== playing && !audio.paused) audio.pause()
  })
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
  return <div ref={root} data-testid="audio-evidence-content"
    onPlayCapture={longReview?.status === 'corroborated' ? (event) => {
      if (event.target instanceof HTMLAudioElement) pauseAudio(event.target)
    } : undefined}>
    {tracks && tracks.length > 1 && <section aria-label="Audio track coverage" className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <h3 className="m-0 font-medium">Audio track coverage</h3>
      <p className="mt-2 mb-0">The file contains {tracks.length} audio tracks. Audio checks used the first track (source stream index {tracks[0]}).{' '}
        {tracks.length === 2 ? 'The second track was unassessed.' : `The other ${tracks.length - 1} tracks were unassessed.`}</p>
    </section>}
    <AudioSoundEvents set={set} token={token} mediaReady={mediaReady} pauseAudio={pauseAudio} />
    <AudioBackgroundContext set={set} token={token} mediaReady={mediaReady} pauseAudio={pauseAudio} />
    <AudioStereoContext set={set} token={token} mediaReady={mediaReady} videoOriginS={videoOriginS} pauseAudio={pauseAudio} />
    <AudioRepetitionContext set={set} token={token} mediaReady={mediaReady} videoOriginS={videoOriginS} />
    {shortReview && <section id="media-evidence-short-voice-review" aria-label={shortReview.title}
      className="mt-5 rounded-md border border-border p-4 scroll-mt-20"
      onPlayCapture={(event) => { if (event.target instanceof HTMLAudioElement) pauseAudio(event.target) }}>
      <h3 className="m-0 text-body font-medium">{shortReview.title}</h3>
      <p className="mt-2 mb-0 text-small">{shortReview.reason}</p>
      {shortReview.status === 'flagged' && shortReview.interval && shortReview.start_s != null && shortReview.end_s != null && <>
        <p className="mt-2 mb-0 font-mono text-caption text-text-muted">
          Analyzed input {seconds(shortReview.start_s)}–{seconds(shortReview.end_s)} · samples [{shortReview.interval.start_sample}, {shortReview.interval.end_sample})
        </p>
        <SourceTimestamps ranges={shortReview.source_timestamp_ranges} />
        {shortReview.artifacts.waveform && <div className="mt-4">
          <EvidencePicture set={set} artifact={shortReview.artifacts.waveform} token={token} ready={mediaReady} chart
            alt="Waveform with the flagged two-second speech window marked" />
        </div>}
        {shortReview.artifacts.selected_clip && <div className="mt-4">
          <EvidenceAudio set={set} artifact={shortReview.artifacts.selected_clip} token={token} ready={mediaReady}
            alt="Flagged two-second speech window" />
        </div>}
      </>}
      {rate && shortReview.checked_windows.length > 0 && <details className="mt-3 text-small text-text-muted">
        <summary className="cursor-pointer font-medium">Checked two-second windows</summary>
        <ul className="mt-2 mb-0 list-disc pl-4">{shortReview.checked_windows.map((window) =>
          <li key={window.start_sample}>{seconds(window.start_sample / rate)}–{seconds(window.end_sample / rate)}</li>)}</ul>
      </details>}
      <p className="mt-3 mb-0 text-caption text-text-muted">{shortReview.limitation}</p>
    </section>}
    {longReview && <section id="media-evidence-long-voice-review" aria-label={longReview.title}
      className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
      <h3 className="m-0 text-body font-medium">{longReview.title}</h3>
      <p className="mt-2 mb-0 text-small" role="status">{longReview.reason}</p>
      {longReview.windows.length > 0 && <ol className="mt-4 mb-0 space-y-3 list-none p-0">
        {longReview.windows.map((window) => {
          const span = `${seconds(window.start_sample / longVoiceSampleRate)}–${seconds(window.end_sample / longVoiceSampleRate)}`
          const showMedia = longReview.status === 'corroborated' && window.direction === 'ai'
          return <li key={window.start_sample} className="rounded-md border border-border p-3">
            <p className="m-0 text-small font-medium">{longVoiceLabels[window.direction]}</p>
            <p className="mt-2 mb-0 font-mono text-caption text-text-muted">
              Analyzed input {span} · samples [{window.start_sample}, {window.end_sample})
            </p>
            <SourceTimestamps ranges={window.source_timestamp_ranges} showSeconds />
            {showMedia && window.artifacts.waveform && <div className="mt-3">
              <EvidencePicture set={set} artifact={window.artifacts.waveform} token={token} ready={mediaReady} chart
                alt={`Waveform for the corroborating speech window at ${span}`} />
            </div>}
            {showMedia && window.artifacts.selected_clip && <div className="mt-3">
              <EvidenceAudio set={set} artifact={window.artifacts.selected_clip} token={token} ready={mediaReady}
                alt={`Corroborating speech window at ${span}`} />
            </div>}
          </li>
        })}
      </ol>}
      <p className="mt-3 mb-0 text-caption text-text-muted">{longReview.limitation}</p>
    </section>}
    {set.packets.length > 0 && <AudioInspectionTimeline set={set} token={token} mediaReady={mediaReady} selectedId={selectedId} onSelect={select} />}
    {set.coverage && rate ? <div id={standalone ? 'media-evidence-voice-coverage' : undefined} className="scroll-mt-14 mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <p className="m-0">{shortReview || longReview ? 'Primary voice checks' : 'Voice checks'} cover {seconds(set.coverage.scored_samples / rate)} of the {seconds(set.coverage.decoded_samples / rate)} decoded input.
        {' '}{seconds(set.coverage.unscored_samples / rate)} of that input has no window score. Overlapping windows are counted once in this coverage.</p>
      <p className="mt-2 mb-0">{set.coverage.windows.length} scored windows. Responses apply to complete windows, normally four seconds; they do not identify individual words or exact voice-replacement boundaries.</p>
      {intervals.length > 0 && <details className="mt-3" open={hash === '#media-evidence-voice-coverage' || undefined}>
        <summary className="cursor-pointer font-medium">Scored voice intervals</summary>
        <ul className="mt-2 mb-0 pl-4 list-disc">{intervals.map((window) => <li key={`${window.start_sample}:${window.end_sample}`}>
          {seconds(window.start_sample / rate)}–{seconds(window.end_sample / rate)} · samples [{window.start_sample}, {window.end_sample})
        </li>)}</ul>
      </details>}
      {clock && <section aria-label="Audio source clock breaks" className="mt-3">
        <h3 className="m-0 font-medium">Audio source clock breaks</h3>
        {clock.recording_voice_estimate_withheld && <p className="mt-2 mb-0">The source audio timestamps contain a gap or overlap. Voice checks use only continuous intervals, so a recording-wide voice estimate is unavailable.</p>}
        {clock.total_discontinuities ? <>
          <p className="mt-2 mb-0">The decoded listening copy joins these source spans. {clock.scored_windows_crossing} recorded voice {clock.scored_windows_crossing === 1 ? 'window crosses' : 'windows cross'} a source clock break. This timing check carries zero weight in the AI assessment.</p>
          <ul className="mt-2 mb-0 space-y-2 pl-4 list-disc">
            {clock.breaks.map((item) => <li key={item.boundary_sample}>
              <p className="m-0">At {seconds(item.boundary_sample / rate)} of analyzed audio. Source timestamp {item.jump_samples > 0 ? 'gap' : 'overlap'} of {measurementValue(Math.abs(item.jump_samples) * 1000 / rate)} ms.</p>
              <p className="m-0 text-caption">On the original timeline, the audio spans {seconds(item.before.source_start_pts / rate)} to {seconds(item.before.source_end_pts / rate)} before the break and {seconds(item.after.source_start_pts / rate)} to {seconds(item.after.source_end_pts / rate)} after it.</p>
              <p className="m-0">{item.scored_windows_crossing} recorded voice {item.scored_windows_crossing === 1 ? 'window crosses' : 'windows cross'} this break.</p>
            </li>)}
          </ul>
          <p className="mt-2 mb-0">This check lists jumps of at least {measurementValue(clock.minimum_jump_samples * 1000 / rate)} ms.
            {clock.total_discontinuities > clock.breaks.length && ` Showing the ${clock.breaks.length} largest of ${clock.total_discontinuities} breaks. The full source timeline is in JSON.`}</p>
        </> : <p className="mt-2 mb-0">No source clock break of at least {measurementValue(clock.minimum_jump_samples * 1000 / rate)} ms was measured in {seconds(clock.checked_samples / rate)} of decoded audio. This timing check carries zero weight in the AI assessment.</p>}
      </section>}
      {!!set.coverage.unscored_intervals?.length && <details className="mt-3">
        <summary className="cursor-pointer font-medium">Unscored voice intervals</summary>
        <p className="mt-2 mb-0">These times have no recorded voice response. Background sound can affect coverage, and the gap alone does not establish its cause.</p>
        <ul className="mt-2 mb-0 space-y-2 pl-4 list-disc">
          {set.coverage.unscored_intervals.map((gap) => <li key={gap.start_sample}>
            <p className="m-0">Analyzed audio {seconds(gap.start_sample / rate)}–{seconds(gap.end_sample / rate)}.</p>
            <SourceTimestamps ranges={gap.source_timestamp_ranges} videoOriginS={videoOriginS} />
          </li>)}
        </ul>
      </details>}
      {stereoOverlaps.length > 0 && <section aria-label="Unscored voice during a mono level drop" className="mt-3">
        <p className="m-0 font-medium">Unscored voice during a mono level drop</p>
        <p className="mt-2 mb-0">The <a className="underline" href="#media-evidence-stereo">stereo measurements</a> and unscored voice times overlap below. The cause of the missing voice score is unknown.</p>
        <ul className="mt-2 mb-0 space-y-2 pl-4 list-disc">
          {stereoOverlaps.map(({ overlap, measurement }) => <li key={overlap.start_sample}>
            <p className="m-0">Overlap {seconds(overlap.start_sample / rate)}–{seconds(overlap.end_sample / rate)} of analyzed audio.</p>
            <p className="m-0">{measurement.attenuation_db == null ? 'Mono was digitally silent in the overlapping stereo measurement.'
              : `The overlapping one-second stereo measurement found mono ${measurement.attenuation_db} dB quieter than the louder channel.`}</p>
            <SourceTimestamps ranges={overlap.source_timestamp_ranges} videoOriginS={videoOriginS} />
          </li>)}
        </ul>
      </section>}
      {set.coverage.decode_limit_reached && <p className="mt-2 mb-0">The decoding limit was reached. Audio beyond this input was not examined.</p>}
      <p className="mt-2 mb-0">Voice inspection clips reproduce the exact analyzed {measurementValue(rate)} Hz mono input. The source had {set.source.source_channels ?? 'unrecorded'} channel(s); this is a derived listening copy.</p>
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
