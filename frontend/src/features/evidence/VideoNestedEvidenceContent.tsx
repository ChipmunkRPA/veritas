import AudioEvidenceContent from './AudioEvidenceContent'
import { EvidenceLimits } from './EvidenceMedia'
import VideoAVEvidenceContent from './VideoAVEvidenceContent'
import { rationalSeconds, seconds, type AudioEvidenceSet, type VideoAudioChannel, type VideoEvidenceSet } from './multimediaPackets'

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
  const showAV = avBound && avHasCoverage && (av?.status === 'ready' || av?.status === 'not_assessed')
  return <div data-testid="video-nested-evidence" className="mt-8 space-y-8">
    {channel !== 'av' && <section id="media-evidence-voice" aria-label="Voice evidence from this video" className="border-t border-border pt-6 scroll-mt-20">
      <h3 className="m-0 text-body font-medium">Voice evidence from this video</h3>
      <p className="mt-2 mb-0 text-small text-text-muted">Audio windows use the same original video and saved evidence record. Samples begin at decoded audio zero; original recording timestamps remain available below.</p>
      {showAudio && audio ? <>
        {audio.status === 'not_assessed' && <UnavailableChannel reason={audio.reason} />}
        <AudioEvidenceContent set={videoAudioView(set, audio)} token={token} mediaReady={mediaReady} videoOriginS={set.source.timeline_origin_s} standalone={false} />
        {audio.packets.length === 0 && <SourceClock source={audio.source} />}
        {audio.packets.length === 0 && <p className="mt-3 text-small text-text-muted">No voice window was selected for inspection. Recorded coverage remains available; this does not establish a natural voice.</p>}
        {(audio.omitted_windows ?? 0) > 0 && <p className="mt-3 text-small text-text-muted">{audio.omitted_windows} additional voice window(s) are omitted from this bounded presentation.</p>}
      </> : <UnavailableChannel reason={!audioBound && audio ? 'The audio record does not match this original video.' : audio?.reason} />}
      <EvidenceLimits lines={audio?.limitations ?? []} />
    </section>}
    {channel !== 'audio' && <section id="media-evidence-timing" aria-label="Audio and picture timing evidence" className="border-t border-border pt-6 scroll-mt-20">
      <h3 className="m-0 text-body font-medium">Audio and picture timing evidence</h3>
      {showAV && av ? <>
        {av.status === 'not_assessed' && <UnavailableChannel reason={av.reason} />}
        <VideoAVEvidenceContent set={set} channel={av} token={token} mediaReady={mediaReady} />
        {av.packets.length === 0 && <SourceClock source={av.source} />}
      </>
        : <UnavailableChannel reason={!avBound && av ? 'The alignment record does not match this original video.' : av?.reason} />}
      <EvidenceLimits lines={av?.limitations ?? []} />
    </section>}
  </div>
}
