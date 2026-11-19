import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Button } from '../../components/ui'
import AudioInspectionTimeline from './AudioInspectionTimeline'
import AudioBackgroundContext from './AudioBackgroundContext'
import AudioSoundEvents from './AudioSoundEvents'
import AudioStereoContext from './AudioStereoContext'
import AudioRepetitionContext from './AudioRepetitionContext'
import { EvidenceAudio, EvidenceLimits, EvidenceMeasurements, EvidencePicture, EvidenceReferences, evidenceCard, evidenceGrid } from './EvidenceMedia'
import { measurementValue } from './imagePackets'
import { rationalSeconds, seconds, type AudioEvidencePacket, type AudioEvidenceSet, type BriefVoiceReview, type LongVoiceCoverageStatus } from './multimediaPackets'

const longVoiceLabels = {
  ai: 'Leans AI-generated in this window',
  original: 'Leans original in this window',
  inconclusive: 'Inconclusive window',
  unassessed: 'No usable speech response',
}
const longVoiceSampleRate = 16_000
const longVoiceCoverageLabels: Record<LongVoiceCoverageStatus, string> = {
  directional: 'At least one directional response',
  inconclusive: 'Only inconclusive responses',
  below_speech_minimum: 'All covering windows were below the speech minimum',
  outside_schedule: 'Outside the sampled windows',
}

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

const selectionLabels = {
  scored: 'Scored',
  below_speech_minimum: 'Below the speech minimum',
  outside_window_limit: 'Outside the window limit',
}

function SpeechWindowSelection({ selection, rate, videoOriginS }: {
  selection: NonNullable<NonNullable<AudioEvidenceSet['coverage']>['speech_window_selection']>
  rate: number
  videoOriginS?: number | null
}) {
  const { counts, windows } = selection
  const unscored = windows.filter((window) => window.status !== 'scored')
  return <>
    <p className="mt-2 mb-0">{counts.scored} scored · {counts.below_speech_minimum} below the speech minimum · {counts.outside_window_limit} outside the window limit</p>
    <p className="mt-2 mb-0">The voice check requires detected speech in at least half of each candidate window. This measures speech coverage in the mixed track and carries zero weight in the AI estimate.</p>
    {unscored.length > 0 && <ul className="mt-2 mb-0 space-y-2 pl-4 list-disc">
      {unscored.map((window) => <li key={window.start_sample}>
        <p className="m-0">Analyzed audio {seconds(window.start_sample / rate)}–{seconds(window.end_sample / rate)}.
          {' '}Detected speech {measurementValue(100 * window.speech_samples / (window.end_sample - window.start_sample))}%.
          {' '}{selectionLabels[window.status]}.</p>
        <SourceTimestamps ranges={window.source_timestamp_ranges} videoOriginS={videoOriginS} />
      </li>)}
    </ul>}
  </>
}

function BriefSpeechWindows({ review, set, token, mediaReady }: {
  review: BriefVoiceReview; set: AudioEvidenceSet; token: string; mediaReady: boolean
}) {
  return <section id="media-evidence-brief-voice-review" aria-label="Brief speech windows"
    className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
    <h3 className="m-0 text-body font-medium">Brief speech windows</h3>
    <p className="mt-2 mb-0 text-small" role="status">{review.reason}</p>
    {review.status !== 'skipped' && <p className="mt-2 mb-0 text-caption text-text-muted">
      {review.speech_windows_checked} of {review.scheduled_windows} scheduled 750 ms windows checked.
    </p>}
    {review.status === 'flagged' && review.windows.length > 0 && <>
      <p className="mt-2 mb-0 text-caption text-text-muted">Each marked excerpt appears once, even when it supports two pairs.</p>
      <ol className={`mt-4 mb-0 list-none p-0 ${evidenceGrid}`}>
        {review.windows.map(window => {
          const span = `${seconds(window.start_s)}–${seconds(window.end_s)}`
          return <li key={window.id} id={`media-evidence-${window.id}`} className="rounded-md border border-border p-3 scroll-mt-20">
            <h4 className="m-0 text-small font-medium">{span}</h4>
            <p className="mt-1 mb-0 font-mono text-caption text-text-muted">
              {window.id} · Analyzed samples [{window.start_sample}, {window.end_sample})
            </p>
            <SourceTimestamps ranges={window.source_timestamp_ranges} />
            <div className="mt-3"><EvidencePicture set={set} artifact={window.artifacts.waveform} token={token} ready={mediaReady}
              chart alt={`Waveform for brief speech window ${window.id} at ${span}`} /></div>
            <div className="mt-3"><EvidenceAudio set={set} artifact={window.artifacts.selected_clip} token={token} ready={mediaReady}
              alt={`Brief speech window ${window.id} at ${span}`} /></div>
          </li>
        })}
      </ol>
    </>}
    <p className="mt-3 mb-0 text-caption text-text-muted">{review.limitation}</p>
  </section>
}

export default function AudioEvidenceContent({ set, token, mediaReady, videoOriginS, standalone = true, hasTrackComparison = false }: {
  set: AudioEvidenceSet; token: string; mediaReady: boolean; videoOriginS?: number | null; standalone?: boolean; hasTrackComparison?: boolean
}) {
  const rate = set.source.sample_rate
  const tracks = set.source.audio_stream_indices
  const clock = set.coverage?.source_clock_context
  const shortReview = standalone ? set.short_voice_review : undefined
  const focusedEdit = standalone ? set.focused_speech_edit_review : undefined
  const shortInterval = shortReview?.interval ?? focusedEdit?.interval
  const shortContext = shortInterval
    ? (set.acoustic_context?.events ?? []).filter(event =>
      event.interval.start_sample < shortInterval.end_sample && event.interval.end_sample > shortInterval.start_sample)
    : []
  const longReview = standalone ? set.long_voice_review : undefined
  const twoSecond = standalone ? set.two_second_review ?? longReview?.two_second_review : undefined
  const twoSecondCoverage = twoSecond?.sample_coverage
  const twoSecondCount = twoSecond?.excerpts.reduce((total, row) =>
    total + row.speech_samples.filter(samples => samples >= longVoiceSampleRate).length, 0)
  const briefReview = standalone && set.brief_voice_review?.schema === 'chipmunk.brief-voice-review/1'
    && set.brief_voice_review.affects_overall_confidence === false ? set.brief_voice_review : undefined
  const focusedSpeech = Array.isArray(longReview?.regions)
  const hasMeasuredSpeech = longReview?.windows.some((window) => window.direction !== 'unassessed')
  const backgroundReview = standalone ? set.background_reduced_voice_review : undefined
  const pairedBackgroundWaveforms = backgroundReview?.schema === 'chipmunk.background-reduced-voice-review/2'
  const intervals = standalone && set.packets.length === 0 ? [...new Map(set.coverage?.windows?.map((window) =>
    [`${window.start_sample}:${window.end_sample}`, window] as const)).values()]
    .sort((a, b) => a.start_sample - b.start_sample || a.end_sample - b.end_sample) : []
  const stereo = set.stereo_context
  const stereoOverlaps = set.coverage?.stereo_attenuation_overlaps?.flatMap((overlap) => {
    const measurement = stereo?.windows[overlap.stereo_window_index]?.downmix
    return measurement?.strong_attenuation ? [{ overlap, measurement }] : []
  }) ?? []
  const { hash, search } = useLocation()
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
    onPlayCapture={longReview?.status === 'corroborated' || twoSecond?.status === 'flagged'
      || briefReview?.status === 'flagged' || backgroundReview?.status === 'flagged' ? (event) => {
      if (event.target instanceof HTMLAudioElement) pauseAudio(event.target)
    } : undefined}>
    {tracks && tracks.length > 1 && <section aria-label="Audio track coverage" className="mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <h3 className="m-0 font-medium">Audio track coverage</h3>
      <p className="mt-2 mb-0">The file contains {tracks.length} audio tracks. Audio checks used the first track (source stream index {tracks[0]}).{' '}
        {hasTrackComparison ? 'See the timing comparison between tracks below.'
          : tracks.length === 2 ? 'The second track was unassessed.' : `The other ${tracks.length - 1} tracks were unassessed.`}</p>
    </section>}
    {focusedEdit && <section id="media-evidence-focused-speech-edit" aria-label={focusedEdit.title}
      className="mt-5 rounded-md border border-border p-4 scroll-mt-20"
      onPlayCapture={(event) => { if (event.target instanceof HTMLAudioElement) pauseAudio(event.target) }}>
      <h3 className="m-0 text-body font-medium">{focusedEdit.title}</h3>
      <p className="mt-2 mb-0 text-small">{focusedEdit.reason}</p>
      {focusedEdit.checked_windows > 0 && <p className="mt-2 mb-0 text-small">
        Checked {focusedEdit.checked_windows} windows lasting {focusedEdit.checked_scope ? '0.25' : '0.5'} to 1 second.
      </p>}
      {focusedEdit.checked_scope && <p className="mt-2 mb-0 text-small">
        Search scope {seconds(focusedEdit.checked_scope.start_sample / 16000)}–{seconds(focusedEdit.checked_scope.end_sample / 16000)}.
      </p>}
      {focusedEdit.schema === 'chipmunk.focused-speech-edit-review/2' && <p className="mt-2 mb-0 text-small">
        The check uses 16 kHz audio decoded from the 48 kHz recording.
      </p>}
      {focusedEdit.interval && focusedEdit.start_s != null && focusedEdit.end_s != null && <>
        <p className="mt-2 mb-0 font-mono text-caption text-text-muted">
          Flagged window {seconds(focusedEdit.start_s)}–{seconds(focusedEdit.end_s)} · samples [{focusedEdit.interval.start_sample}, {focusedEdit.interval.end_sample})
        </p>
        {focusedEdit.original_timestamp_range ? <p className="m-0 text-caption">
          Original samples [{focusedEdit.original_timestamp_range.start_pts}, {focusedEdit.original_timestamp_range.end_pts}) at 48000 Hz.
        </p> : <SourceTimestamps ranges={focusedEdit.source_timestamp_ranges} />}
        {focusedEdit.artifacts.waveform && <div className="mt-4">
          <EvidencePicture set={set} artifact={focusedEdit.artifacts.waveform} token={token} ready={mediaReady} chart
            alt="Waveform with the focused speech window marked" />
        </div>}
        {focusedEdit.artifacts.selected_clip && <div className="mt-4">
          <EvidenceAudio set={set} artifact={focusedEdit.artifacts.selected_clip} token={token} ready={mediaReady}
            alt="Focused speech window" />
        </div>}
        {shortContext.length > 0 && <p className="mt-3 mb-0 text-caption text-text-muted">
          Recording observations at the same time{' '}
          {shortContext.map((event, index) => <span key={event.id}>{index > 0 && ', '}
            <Link to={{ search, hash: `#media-evidence-${encodeURIComponent(event.id)}` }} className="underline underline-offset-2">{event.id}</Link>
          </span>)}.
        </p>}
      </>}
      <p className="mt-3 mb-0 text-caption text-text-muted">{focusedEdit.limitation}</p>
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
      {standalone && set.source.identical_stereo_channel_pcm_sha256 && <p className="mt-2 mb-0 text-small">
        Both recorded channels contain the same samples.
      </p>}
      {set.source.source_sample_rate === 48000 && shortReview.window_coverage && <p className="mt-2 mb-0 text-small">
        This check uses 16 kHz mono audio decoded from the 48 kHz source.
      </p>}
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
        {shortContext.length > 0 && <p className="mt-3 mb-0 text-caption text-text-muted">
          Overlapping recording observations{' '}
          {shortContext.map((event, index) => <span key={event.id}>{index > 0 && ', '}
            <Link to={{ search, hash: `#media-evidence-${encodeURIComponent(event.id)}` }} className="underline underline-offset-2">{event.id}</Link>
          </span>)}. They describe the same time in the mixed track. The cause of the speech response is undetermined.
        </p>}
      </>}
      {rate && shortReview.nearby_window_review && <div className="mt-4 text-small">
        <h4 className="m-0 font-medium">Nearby speech windows</h4>
        <p className="mt-1 mb-2 text-caption text-text-muted">
          {{ repeated: 'The elevated response repeated in every checked nearby window.',
            varied: 'At least one nearby window fell below the review point. Small timing changes affected this finding.',
            incomplete: 'Some nearby windows had too little detected speech to compare.' }[shortReview.nearby_window_review.status]}
        </p>
        <ul className="mb-0 grid list-none gap-2 p-0 sm:grid-cols-2">
          {shortReview.nearby_window_review.windows.map((window) => <li key={window.start_sample} className="rounded-md border border-border p-2">
            <p className="m-0 font-medium">{seconds(window.start_sample / rate)}–{seconds(window.end_sample / rate)}</p>
            <p className="mb-0 mt-1 text-caption text-text-muted">
              {{ elevated: 'Elevated response', below_review_point: 'Below the review point',
                below_speech_minimum: 'Below the speech minimum' }[window.status]} · {seconds(window.speech_samples / rate)} detected speech
            </p>
            <SourceTimestamps ranges={window.source_timestamp_ranges} />
          </li>)}
        </ul>
      </div>}
      {rate && shortReview.window_coverage ? <div className="mt-4 text-small">
        <h4 className="m-0 font-medium">Two-second window coverage</h4>
        <p className="mt-1 mb-2 text-caption text-text-muted">Each window requires at least one second of detected speech in the mixed track.</p>
        <ul className="mb-0 grid gap-2 list-none p-0 sm:grid-cols-2">
          {shortReview.window_coverage.map((window) => <li key={window.start_sample} className="rounded-md border border-border p-2">
            <p className="m-0 font-medium">{seconds(window.start_sample / rate)}–{seconds(window.end_sample / rate)}
              {' · '}{window.status === 'checked' ? 'Checked' : 'Below the speech minimum'}</p>
            <p className="m-0 text-caption">Detected speech {seconds(window.speech_samples / rate)}.</p>
            <SourceTimestamps ranges={window.source_timestamp_ranges} />
          </li>)}
        </ul>
      </div> : rate && shortReview.checked_windows.length > 0 && <details className="mt-3 text-small text-text-muted">
        <summary className="cursor-pointer font-medium">Checked two-second windows</summary>
        <ul className="mt-2 mb-0 list-disc pl-4">{shortReview.checked_windows.map((window) =>
          <li key={window.start_sample}>{seconds(window.start_sample / rate)}–{seconds(window.end_sample / rate)}</li>)}</ul>
      </details>}
      <p className="mt-3 mb-0 text-caption text-text-muted">{shortReview.limitation}</p>
    </section>}
    {twoSecond && <section id="media-evidence-two-second-speech-review" aria-label={twoSecond.title}
      className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
      <h3 className="m-0 text-body font-medium">{twoSecond.title}</h3>
      <p className="mt-2 mb-0 text-small" role="status">{twoSecond.reason}</p>
      {twoSecondCoverage && <div className="mt-3">
        <h4 className="m-0 text-small font-medium">Where the two-second check ran</h4>
        <p className="mt-1 mb-0 text-caption text-text-muted">
          {twoSecondCoverage.checked_windows} of {twoSecondCoverage.scheduled_windows} distinct windows checked, covering{' '}
          {seconds(twoSecondCoverage.checked_samples / longVoiceSampleRate)} of {seconds(twoSecondCoverage.scope_samples / longVoiceSampleRate)}.
          Overlapping checked windows are counted once. Coverage describes where the check ran.
        </p>
        <table aria-label="Two-second speech coverage intervals" className="mt-2 w-full text-left text-caption">
          <thead><tr><th scope="col">Source interval</th><th scope="col">Coverage</th></tr></thead>
          <tbody>{twoSecondCoverage.intervals.map(interval => <tr key={interval.start_sample} className="border-t border-border">
            <td className="py-2 pr-3 align-top">
              <p className="m-0 font-mono">{(interval.start_sample / longVoiceSampleRate).toFixed(7)}–{(interval.end_sample / longVoiceSampleRate).toFixed(7)} s</p>
              <p className="m-0 font-mono text-text-muted">Samples [{interval.start_sample}, {interval.end_sample})</p>
              {interval.native_source_range ? <p className="m-0 font-mono text-text-muted">
                Original ticks [{interval.native_source_range.start_pts}, {interval.native_source_range.end_pts}) at {interval.native_source_range.time_base}
              </p> : <SourceTimestamps ranges={interval.source_timestamp_ranges} />}
            </td>
            <td className="py-2 align-top">{interval.status === 'checked' ? 'Checked' : 'Insufficient detected speech'}</td>
          </tr>)}</tbody>
        </table>
      </div>}
      {twoSecond.regions.length > 0 && <ol className="mt-4 mb-0 space-y-3 list-none p-0">
        {twoSecond.regions.filter(region => region.same_pair_as === undefined).map(region => {
          const windows = region.confirming_window ? [region, region.confirming_window] : [region]
          const time = (sample: number) => region.confirming_window
            ? `${(sample / longVoiceSampleRate).toFixed(7)} s` : seconds(sample / longVoiceSampleRate)
          const context = (set.acoustic_context?.events ?? []).filter(event => windows.some(window =>
            event.interval.start_sample < window.end_sample && event.interval.end_sample > window.start_sample))
          return <li key={region.start_sample} className="rounded-md border border-border p-3">
            {windows.map((window, index) => {
              const span = `${time(window.start_sample)}–${time(window.end_sample)}`
              const playerLabel = index === 0 ? 'Two-second speech window' : 'Confirming two-second speech window'
              const waveformLabel = index === 0 ? 'two-second speech window' : 'confirming two-second speech window'
              return <div key={window.start_sample} className={index > 0 ? 'mt-4 border-t border-border pt-3' : undefined}>
                <p className="m-0 text-small font-medium">{index === 0 ? 'AI-like speech response' : 'Confirming response'} at {span}</p>
                <p className="mt-1 mb-0 font-mono text-caption text-text-muted">Samples [{window.start_sample}, {window.end_sample})</p>
                {window.native_source_range ? <p className="mt-1 mb-0 font-mono text-caption text-text-muted">
                  Original source ticks [{window.native_source_range.start_pts}, {window.native_source_range.end_pts}) at {window.native_source_range.time_base}
                </p> : <SourceTimestamps ranges={window.source_timestamp_ranges} />}
                <div className="mt-3"><EvidencePicture set={set} artifact={window.artifacts.waveform} token={token} ready={mediaReady}
                  chart alt={`Waveform for the ${waveformLabel} at ${span}`} /></div>
                <div className="mt-3"><EvidenceAudio set={set} artifact={window.artifacts.selected_clip} token={token} ready={mediaReady}
                  alt={`${playerLabel} at ${span}`} /></div>
              </div>
            })}
            {context.length > 0 && <p className="mt-3 mb-0 text-caption text-text-muted">Overlapping recording observations{' '}
              {context.map((event, index) => <span key={event.id}>{index > 0 && ', '}
                <Link to={{ search, hash: `#media-evidence-${encodeURIComponent(event.id)}` }} className="underline underline-offset-2">{event.id}</Link>
              </span>)}.</p>}
          </li>
        })}
      </ol>}
      {!twoSecondCoverage && twoSecond.excerpts.length > 0 && <details className="mt-3 text-small">
        <summary className="cursor-pointer">{twoSecondCount} checks across {twoSecond.excerpts.length} overlapping six-second excerpts</summary>
        <ul className="mt-2 mb-0 list-disc pl-4">{twoSecond.excerpts.map(excerpt => <li key={excerpt.start_sample}>
          {seconds(excerpt.start_sample / longVoiceSampleRate)}–{seconds(excerpt.start_sample / longVoiceSampleRate + 6)}
          {' · '}{excerpt.speech_samples.filter(samples => samples >= longVoiceSampleRate).length}
          {' of '}{excerpt.speech_samples.length} windows had enough speech to check
        </li>)}</ul>
      </details>}
      <p className="mt-3 mb-0 text-caption text-text-muted">{twoSecond.limitation}</p>
    </section>}
    {briefReview && <BriefSpeechWindows review={briefReview} set={set} token={token} mediaReady={mediaReady} />}
    {longReview && <section id="media-evidence-long-voice-review" aria-label={longReview.title}
      className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
      <h3 className="m-0 text-body font-medium">{longReview.title}</h3>
      <p className="mt-2 mb-0 text-small" role="status">{focusedSpeech && longReview.windows.length > 0 && !hasMeasuredSpeech
        ? 'None of the sampled windows produced a usable speech response.' : longReview.reason}</p>
      {longReview.sample_coverage && <details className="mt-3 text-small" open={longReview.status === 'no_corroboration'}>
        <summary className="cursor-pointer">Where the long speech review reached a response</summary>
        <p className="mt-2 mb-0 text-text-muted">Intervals are listed once even when windows overlap. A directional window does not assign a result to every sample in it.</p>
        <table className="mt-3 w-full text-left text-caption">
          <thead><tr><th scope="col">Source interval</th><th scope="col">Window coverage</th></tr></thead>
          <tbody>{longReview.sample_coverage.intervals.map(interval => <tr key={interval.start_sample}>
            <td className="py-1 pr-3 font-mono">{seconds(interval.start_sample / longVoiceSampleRate)}–{seconds(interval.end_sample / longVoiceSampleRate)}</td>
            <td>{longVoiceCoverageLabels[interval.status]}</td>
          </tr>)}</tbody>
        </table>
      </details>}
      {!!longReview.regions?.length && <ol className="mt-4 mb-0 space-y-3 list-none p-0">
        {longReview.regions.map((region) => {
          const span = `${seconds(region.start_sample / longVoiceSampleRate)}–${seconds(region.end_sample / longVoiceSampleRate)}`
          return <li key={`${region.start_sample}:${region.end_sample}`} className="rounded-md border border-border p-3">
            <p className="m-0 text-small font-medium">{region.basis === 'overlap' ? 'Overlap of two AI-leaning windows' : 'Corroborated three-second window'}</p>
            <p className="mt-2 mb-0 font-mono text-caption text-text-muted">Selected range {span} · samples [{region.start_sample}, {region.end_sample})</p>
            <SourceTimestamps ranges={region.source_timestamp_ranges} showSeconds />
            <ul aria-label="Supporting speech readings" className="mt-2 mb-0 flex flex-wrap gap-x-4 gap-y-1 list-none p-0 text-small">
              {region.supporting_windows.map((index) => {
                const window = longReview.windows[index]
                if (!window) return null
                return <li key={index}><a href={`#long-speech-window-${index}`} className="underline"
                  onClick={(event) => {
                    event.preventDefault()
                    const reading = root.current?.querySelector<HTMLElement>(`#long-speech-window-${index}`)
                    if (!reading) return
                    reading.focus({ preventScroll: true })
                    reading.scrollIntoView?.({ block: 'nearest' })
                  }}>{seconds(window.start_sample / longVoiceSampleRate)}–{seconds(window.end_sample / longVoiceSampleRate)}</a></li>
              })}
            </ul>
            <div className="mt-3">
              <EvidencePicture set={set} artifact={region.artifacts.waveform} token={token} ready={mediaReady} chart alt={`Waveform for the selected speech range at ${span}`} />
            </div>
            <div className="mt-3">
              <EvidenceAudio set={set} artifact={region.artifacts.selected_clip} token={token} ready={mediaReady} alt={`Selected speech range at ${span}`} />
            </div>
          </li>
        })}
      </ol>}
      {longReview.windows.length > 0 && <ol className="mt-4 mb-0 space-y-3 list-none p-0">
        {longReview.windows.map((window, index) => {
          const span = `${seconds(window.start_sample / longVoiceSampleRate)}–${seconds(window.end_sample / longVoiceSampleRate)}`
          const showMedia = longReview.status === 'corroborated' && window.direction === 'ai'
          return <li key={`${window.start_sample}:${window.end_sample}`} id={focusedSpeech ? `long-speech-window-${index}` : undefined}
            tabIndex={focusedSpeech ? -1 : undefined} className="rounded-md border border-border p-3 scroll-mt-20">
            <p className="m-0 text-small font-medium">{longVoiceLabels[window.direction]}</p>
            <p className="mt-2 mb-0 font-mono text-caption text-text-muted">
              Analyzed input {span} · samples [{window.start_sample}, {window.end_sample})
            </p>
            {window.speech_samples !== undefined && <p className="mt-1 mb-0 text-caption">
              Detected speech {seconds(window.speech_samples / longVoiceSampleRate)} of 3.000 s.
              {window.direction === 'unassessed' && ' Below the 1.500 s speech minimum.'}
            </p>}
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
    {backgroundReview && <section id="media-evidence-background-reduced-voice-review" aria-label={backgroundReview.title}
      className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
      <h3 className="m-0 text-body font-medium">{backgroundReview.title}</h3>
      <p className="mt-2 mb-0 text-small">{backgroundReview.reason}</p>
      {backgroundReview.status === 'flagged' && <>
        <p className="mt-2 mb-0 text-caption text-text-muted">
          Loud background sound can make genuine speech look AI-generated in this check.
          When voices overlap, the check cannot identify which speaker drove the result.
        </p>
        <SourceTimestamps ranges={backgroundReview.source_timestamp_ranges} showSeconds />
        <p className="m-0 font-mono text-caption text-text-muted">
          Samples [{backgroundReview.interval.start_sample}, {backgroundReview.interval.end_sample})
        </p>
        {pairedBackgroundWaveforms && <p className="mt-3 mb-0 text-small text-text-muted">
          Both waveforms cover the same 0.000–3.000 s source interval on a fixed digital amplitude scale.
          Processing can change the sound of overlapping voices.
        </p>}
        <div className="mt-4 space-y-4">
          <div className="space-y-3">
            {pairedBackgroundWaveforms && <h4 className="m-0 text-small font-medium">Original recording</h4>}
            {backgroundReview.artifacts.waveform && <EvidencePicture set={set} artifact={backgroundReview.artifacts.waveform}
              token={token} ready={mediaReady} chart alt="Original waveform across the flagged three-second interval" />}
            {backgroundReview.artifacts.selected_clip && <EvidenceAudio set={set} artifact={backgroundReview.artifacts.selected_clip}
              token={token} ready={mediaReady} alt="Original three-second interval" />}
          </div>
          <div className="space-y-3">
            {pairedBackgroundWaveforms && <h4 className="m-0 text-small font-medium">Processed copy</h4>}
            {pairedBackgroundWaveforms && backgroundReview.artifacts.processed_waveform && <EvidencePicture set={set}
              artifact={backgroundReview.artifacts.processed_waveform} token={token} ready={mediaReady} chart
              alt="Processed waveform across the same three-second interval" />}
            {backgroundReview.artifacts.processed_clip && <EvidenceAudio set={set} artifact={backgroundReview.artifacts.processed_clip}
              token={token} ready={mediaReady} alt="Processed copy with reduced background sound" />}
          </div>
        </div>
      </>}
      <p className="mt-3 mb-0 text-caption text-text-muted">{backgroundReview.limitation}</p>
    </section>}
    {set.packets.length > 0 && <AudioInspectionTimeline set={set} token={token} mediaReady={mediaReady} selectedId={selectedId} onSelect={select} />}
    {set.coverage && rate ? <div id={standalone ? 'media-evidence-voice-coverage' : undefined} className="scroll-mt-14 mt-4 rounded-md border border-border p-4 text-small text-text-muted">
      <p className="m-0">{shortReview || longReview || backgroundReview || briefReview ? 'Primary voice checks' : 'Voice checks'} cover {seconds(set.coverage.scored_samples / rate)} of the {seconds(set.coverage.decoded_samples / rate)} decoded input.
        {' '}{seconds(set.coverage.unscored_samples / rate)} of that input has no window score. Overlapping windows are counted once in this coverage.</p>
      <p className="mt-2 mb-0">{set.coverage.windows.length} scored windows. Responses apply to complete windows, normally four seconds; they do not identify individual words or exact voice-replacement boundaries.</p>
      {set.coverage.speech_window_selection && <details className="mt-3"
        open={set.coverage.windows.length === 0 || hash === '#media-evidence-voice-coverage' || undefined}>
        <summary className="cursor-pointer font-medium">How speech windows were selected</summary>
        <SpeechWindowSelection selection={set.coverage.speech_window_selection} rate={rate} videoOriginS={videoOriginS} />
      </details>}
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
    <div className="mt-6 space-y-7">{set.packets.map((packet) => {
      const background = set.acoustic_context?.events.filter((event) => event.related_voice_packet_ids.includes(packet.id)) ?? []
      const sounds = set.sound_events?.windows.filter((window) => window.start_sample < packet.interval.end_sample
        && window.end_sample > packet.interval.start_sample) ?? []
      const contextLinks = [
        ...background.map((event) => ({ id: event.id, label: `Acoustic observation ${event.id}` })),
        ...sounds.map((window) => ({ id: window.id, label: `Sound categories ${window.id}` })),
      ]
      return <article key={packet.id} id={`media-evidence-${packet.id}`} hidden={packet.id !== selectedId} tabIndex={-1} className={evidenceCard}
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
      {contextLinks.length > 0 && <div className="mt-3 text-small">
        <p className="m-0">Listen to overlapping background and recording evidence.</p>
        <ul className="mt-1 mb-0 pl-4 list-disc">{contextLinks.map(({ id, label }) => <li key={id}>
          <Link className="underline underline-offset-2" to={{ search, hash: `#media-evidence-${id}` }}>{label}</Link>
        </li>)}</ul>
      </div>}
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
    </article>})}</div>
  </div>
}
