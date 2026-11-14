import AudioEvidenceContent from './AudioEvidenceContent'
import { EvidenceAudio, EvidenceLimits, EvidencePicture, evidenceGrid } from './EvidenceMedia'
import VideoAVEvidenceContent from './VideoAVEvidenceContent'
import { rationalSeconds, seconds, type AudioEvidenceSet, type SoundtrackPair, type VideoAudioChannel, type VideoEvidenceSet } from './multimediaPackets'

/** Presentation adapter only: nested audio never receives a separate authority or hash. */
export function videoAudioView(parent: VideoEvidenceSet, channel: VideoAudioChannel): AudioEvidenceSet {
  return { ...channel, schema: 'chipmunk.audio-evidence/1', status: 'ready',
    id: parent.id, analysis_id: parent.analysis_id, revision: parent.revision, version: parent.version,
    sha256: parent.sha256, original_result_hash: parent.original_result_hash,
    coverage: channel.coverage ?? undefined, omitted_windows: channel.omitted_windows ?? 0 }
}

function UnavailableChannel({ reason }: { reason?: string | null }) {
  return <p role="status" className="mt-3 text-small text-text-muted">{reason || 'Source-bound evidence is unavailable for this channel.'}
    {' '}Missing measurements do not establish authenticity or synchronization.</p>
}

function SourceClock({ source }: { source: AudioEvidenceSet['source'] }) {
  const rate = source.sample_rate
  if (!rate || !source.timeline_time_base || !source.timeline?.length) return null
  return <details className="mt-3 text-small text-text-muted"><summary className="cursor-pointer">Recorded audio source clock</summary>
    <p className="mt-2 mb-0">These frozen input ranges preserve the original recording clock. They describe decoded context, not scored coverage.</p>
    <ul className="pl-4 list-disc">{source.timeline.map((range, index) => {
      const start = rationalSeconds(range.source_pts, source.timeline_time_base!)
      return <li key={index}>Samples [{range.start_sample}, {range.end_sample}): {start === null ? 'time conversion unavailable'
        : `${seconds(start)}–${seconds(start + (range.end_sample - range.start_sample) / rate)}`}
        {' '}(starts at {range.source_pts} ticks, time base {source.timeline_time_base})</li>
    })}</ul>
  </details>
}

const count = (value: unknown) => typeof value === 'number' && Number.isFinite(value) && value >= 0

const soundtrackMessages: Record<SoundtrackPair['status'], string> = {
  duration_outside_scope: 'This check covers soundtracks from 1.5 to 12 seconds.',
  second_stream_unavailable: 'The second audio track could not be decoded on a verified sample clock.',
  unsupported_clock: 'This check needs two continuous audio tracks with the same start and duration, from 1.5 to 12 seconds.',
  insufficient_same_time_support: 'The tracks shared too little distinctive sound at the same times to assess a local shift.',
  no_local_shift: 'No qualifying local shift was found within the matched audio windows.',
  local_shift: 'This excerpt occurs at different times in the two soundtracks. Other matched sound is aligned on both sides.',
}

function SoundtrackComparison({ pair, set, token, mediaReady }: {
  pair: SoundtrackPair; set: VideoEvidenceSet; token: string; mediaReady: boolean
}) {
  const location = pair.location
  const offsetMs = location ? location.shift_samples / 16 : 0
  return <section id="media-evidence-soundtrack-pair" aria-label="Timing between audio tracks" className="mt-5 rounded-md border border-border p-4">
    <h4 className="m-0 text-body font-medium">Timing between audio tracks</h4>
    <p className="mt-2 mb-0 text-small">{soundtrackMessages[pair.status]}</p>
    {pair.coverage.candidate_windows != null && <p className="mt-2 mb-0 text-caption text-text-muted">
      {pair.coverage.same_time_windows} same-time matches among {pair.coverage.matched_windows} matched windows.{' '}
      {pair.coverage.candidate_windows} windows checked.</p>}
    {location && <>
      <p className="mt-2 mb-0 text-small">The matching excerpt is {Math.abs(offsetMs).toFixed(1)} ms {offsetMs > 0 ? 'later' : 'earlier'} in the second track.</p>
      <div className={`mt-3 ${evidenceGrid}`} onPlayCapture={event => {
        if (event.target instanceof HTMLAudioElement) {
          event.currentTarget.querySelectorAll('audio').forEach(audio => { if (audio !== event.target && !audio.paused) audio.pause() })
        }
      }}>
        {location.excerpts.map(track => {
          const span = track.source_timestamp_ranges[0]
          const start = span && rationalSeconds(span.source_start_pts, span.time_base)
          const end = span && rationalSeconds(span.source_end_pts, span.time_base)
          const stream = `Stream ${track.stream_index}`
          const origin = set.source.timeline_origin_s
          return <div key={track.stream_index} className="space-y-2">
            <p className="m-0 text-caption">{stream}{start != null && end != null && ` · source ${seconds(start)}–${seconds(end)}`}
              {start != null && end != null && origin != null && start >= origin && ` · playback ${seconds(start - origin)}–${seconds(end - origin)}`}</p>
            <EvidencePicture set={set} token={token} ready={mediaReady} artifact={track.waveform} alt={`${stream} waveform`} chart />
            <EvidenceAudio set={set} token={token} ready={mediaReady} artifact={track.clip} alt={`${stream} matching excerpt`} />
          </div>
        })}
      </div>
    </>}
    <p className="mt-2 mb-0 text-caption text-text-muted">{pair.limitation}</p>
  </section>
}

export default function VideoNestedEvidenceContent({ set, token, mediaReady, channel = 'all' }: {
  set: VideoEvidenceSet; token: string; mediaReady: boolean; channel?: 'all' | 'audio' | 'av'
}) {
  const audio = set.audio, av = set.av
  const audioBound = audio?.source.sha256 === set.source.sha256
  const avBound = av?.source.sha256 === set.source.sha256
  const audioCoverage = audio?.coverage
  const audioHasCoverage = audioCoverage && [audioCoverage.decoded_samples, audioCoverage.scored_samples, audioCoverage.unscored_samples].every(count)
    && Array.isArray(audioCoverage.windows) && !!audio?.source.sample_rate && audio.source.sample_rate > 0
  const avCoverage = av?.coverage
  const avHasCoverage = avCoverage && [avCoverage.recorded_windows, avCoverage.selected_windows, avCoverage.omitted_windows].every(count)
    && Array.isArray(avCoverage.unsupported_checks)
  const showAudio = audioBound && audioHasCoverage && (audio?.status === 'ready' || audio?.status === 'not_assessed')
  const showAV = avBound && avHasCoverage && (av?.status === 'ready' || av?.status === 'not_assessed' || !!avCoverage.timing_windows)
  return <div data-testid="video-nested-evidence" className="mt-8 space-y-8">
    {channel !== 'av' && <section id="media-evidence-voice" aria-label="Voice evidence from this video" className="border-t border-border pt-6 scroll-mt-20">
      <h3 className="m-0 text-body font-medium">Voice evidence from this video</h3>
      <p className="mt-2 mb-0 text-small text-text-muted">Audio windows use the same original video and saved evidence record. Samples begin at decoded audio zero; original recording timestamps remain available below.</p>
      {showAudio && audio ? <>
        {audio.status === 'not_assessed' && <UnavailableChannel reason={audio.reason} />}
        <AudioEvidenceContent set={videoAudioView(set, audio)} token={token} mediaReady={mediaReady} videoOriginS={set.source.timeline_origin_s}
          standalone={false} hasTrackComparison={!!audio.soundtrack_pair} />
        {audio.soundtrack_pair && <SoundtrackComparison pair={audio.soundtrack_pair} set={set} token={token} mediaReady={mediaReady} />}
        {audio.packets.length === 0 && <SourceClock source={audio.source} />}
        {audio.packets.length === 0 && <p className="mt-3 text-small text-text-muted">No voice window was selected for inspection. Recorded coverage remains available; this does not establish a natural voice.</p>}
        {(audio.omitted_windows ?? 0) > 0 && <p className="mt-3 text-small text-text-muted">{audio.omitted_windows} additional voice window(s) are omitted from this bounded presentation.</p>}
      </> : <UnavailableChannel reason={!audioBound && audio ? 'The audio record does not match this original video.' : audio?.reason} />}
      <EvidenceLimits lines={audio?.limitations ?? []} />
    </section>}
    {channel !== 'audio' && <section id="media-evidence-timing" aria-label="Audio and picture timing evidence" className="border-t border-border pt-6 scroll-mt-20">
      <h3 className="m-0 text-body font-medium">Audio and picture timing evidence</h3>
      {showAV && av ? <>
        {av.status !== 'ready' && <UnavailableChannel reason={av.reason} />}
        <VideoAVEvidenceContent set={set} channel={av} token={token} mediaReady={mediaReady} />
        {av.packets.length === 0 && <SourceClock source={av.source} />}
      </>
        : <UnavailableChannel reason={!avBound && av ? 'The alignment record does not match this original video.' : av?.reason} />}
      <EvidenceLimits lines={av?.limitations ?? []} />
    </section>}
  </div>
}
