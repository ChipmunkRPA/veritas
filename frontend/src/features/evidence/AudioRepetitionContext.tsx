import { EvidenceAudio, EvidencePicture, evidenceGrid } from './EvidenceMedia'
import { rationalSeconds, seconds, type AudioAlignedRepetitionContext, type AudioEvidenceSet, type AudioGainRepetitionContext, type AudioRepetitionContext as ExactContext, type AudioSimilarityContext } from './multimediaPackets'

type Props = {
  set: AudioEvidenceSet; token: string; mediaReady: boolean; videoOriginS?: number | null
}

function RepeatCheck({ set, token, mediaReady, videoOriginS, context, channel }: Props & {
  context?: ExactContext | AudioSimilarityContext | AudioGainRepetitionContext | AudioAlignedRepetitionContext
  channel?: 'left' | 'right' | 'difference' | 'aligned'
}) {
  if (!context || context.source_sha256 !== set.source.sha256
    || (!channel && context.pcm_sha256 !== set.source.pcm_sha256)) return null
  const aac = context.schema === 'chipmunk.aligned-stereo-repetition/3'
  const nativeAligned = aac || context.schema === 'chipmunk.aligned-stereo-repetition/2'
  const { sample_rate: decodedRate, samples: decodedSamples } = set.source
  if (!decodedRate || !decodedSamples) return null
  const sampleRate = nativeAligned ? 48000 : decodedRate
  const sampleCount = aac ? set.source.source_duration_pts : decodedSamples * sampleRate / decodedRate
  if (context.sample_rate !== sampleRate || context.input_samples !== sampleCount
    || (nativeAligned && set.source.source_sample_rate !== sampleRate)) return null
  const gain = context.schema === 'chipmunk.audio-gain-repetition/1'
  const similar = !gain && context.schema !== 'chipmunk.audio-exact-repetition/1'
  const waveform = context.schema === 'chipmunk.audio-repeat-similarity/2'
  const difference = channel === 'difference'
  const aligned = nativeAligned || context.schema === 'chipmunk.aligned-stereo-repetition/1'
  const method = gain
    ? 'This check compares the recorded channel for exact repeats after a fixed level or polarity change. Mixing new sound into that channel can hide repeats.'
    : waveform
      ? 'The exact and frequency-band searches yielded no reportable pair. A second similarity check compared waveform patterns in the same continuous 2.5–4 second mono input. Background noise can mask repeats.'
      : aligned
        ? 'This check compares quieter sounds after reducing the dominant sound shared across the stereo channels. It needs a stable timing and level relationship between channels.'
        : similar
        ? 'This check compares patterns across frequency bands in continuous 2.5–4 second mono inputs. Background noise can mask repeats.'
        : difference
          ? 'Subtracting the stereo channels cancels sound recorded identically in both. The match can include off-center speech, music, or background sound.'
          : channel ? 'This check compares the recorded channel before it is mixed into mono.' : 'This check compares samples in continuous 2.5–4 second mono inputs.'
  const id = channel ? `media-evidence-${channel}-${gain ? 'gain-' : ''}repetition` : similar ? 'media-evidence-similar-audio' : 'media-evidence-repetition'
  const title = aligned ? 'Repeated sound after stereo alignment'
    : difference ? 'Repeated sound in the stereo difference'
    : channel ? `${gain ? 'Gain-adjusted' : 'Exact'} repeats in the ${channel} channel`
    : similar ? 'Similar repeated audio' : 'Exact repeated audio'
  return <section id={id} aria-labelledby={`${id}-heading`} className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id={`${id}-heading`} className="m-0 text-body font-medium">{title}</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">{method} Loops, recurring alerts, and editing can all repeat audio. Matches carry no AI confidence weight.</p>
    {nativeAligned && <p className="mt-2 mb-0 text-caption text-text-muted">
      {aac ? 'This comparison uses lower frequency bands of the presented 48 kHz stereo audio. The waveforms show those bands.'
        : 'Times and excerpts use the original 48 kHz stereo recording.'} Each waveform uses its own peak scale for visibility.
    </p>}
    {aligned && context.searched_interval && context.alignment && <p className="mt-2 mb-0 text-caption text-text-muted">
      Compared {seconds(context.searched_interval.start_sample / context.sample_rate)}–{seconds(context.searched_interval.end_sample / context.sample_rate)} of analyzed audio.
      The right channel uses sample offset {context.alignment.right_offset_samples} relative to the left.
    </p>}
    {context.status === 'skipped'
      ? <p className="mt-2 mb-0 text-small">{context.reason === 'exact_channel_difference_found'
        ? 'The direct stereo comparison already found a repeat.'
        : 'The shared sound required no timing or level adjustment.'}</p>
      : context.status !== 'measured'
      ? <p className="mt-2 mb-0 text-small">{context.status === 'ambiguous'
        ? (waveform ? 'The waveform search could not establish a reliable pair.' : 'The search reached its comparison limit before it could establish a result.')
        : 'This input could not be checked within the supported duration, source timing, and signal conditions.'}</p>
      : context.pairs.length === 0
        ? <p className="mt-2 mb-0 text-small">No {gain ? 'gain-adjusted' : waveform ? 'waveform' : similar ? 'similar' : 'exact active'} repeat of at least 0.15 seconds was found in the checked input.
          {waveform && context.frequency_band_search?.status !== 'measured' && ' The frequency-band search was inconclusive.'}</p>
        : <div className="mt-3 space-y-4">{context.pairs.map((pair) => <article key={pair.id} id={`media-evidence-${pair.id}`} className="border-t border-border pt-3 scroll-mt-20">
          <h4 className="m-0 text-small font-medium">{pair.id} · {seconds(pair.sample_count / context.sample_rate)} of {gain ? 'samples matched after a gain adjustment' : similar ? 'similar sound' : 'identical samples'}</h4>
          {'scaled_occurrence' in pair && <p className="mt-2 mb-0 text-small">Multiplying the {pair.scaled_occurrence === 'second' ? 'first' : 'second'} occurrence by {pair.gain_numerator}/{pair.gain_denominator} and rounding to recorded precision reproduces the {pair.scaled_occurrence} occurrence. These ranges can include matching silence. The edit boundaries and order are unknown.</p>}
          {pair.artifacts.waveform && <EvidencePicture set={set} token={token} ready={mediaReady} chart
            artifact={pair.artifacts.waveform} alt={`${pair.id} · ${difference ? 'Channel difference waveform' : 'Waveform'} shared by both occurrences`} />}
          <div className={`${evidenceGrid} mt-3`}>{(['first', 'second'] as const).map((side) => {
            const span = pair[side]
            const chart = pair.artifacts[`${side}_waveform`]
            const recorded = pair.artifacts[`${side}_recorded_clip`]
            const original = 'recorded_excerpt' in span ? span.recorded_excerpt.source_timestamp_ranges[0] : undefined
            const occurrence = side === 'first' ? 'First occurrence' : 'Second occurrence'
            return <div key={side}>
              {chart && <EvidencePicture set={set} token={token} ready={mediaReady} chart
                artifact={chart} alt={`${pair.id} · ${side === 'first' ? 'First' : 'Second'} occurrence waveform`} />}
              {(difference || aligned) && <p className="mt-2 mb-1 text-caption">{aligned ? 'Sound after stereo alignment' : 'Channel difference'}</p>}
              <EvidenceAudio set={set} artifact={pair.artifacts[`${side}_clip`]} token={token} ready={mediaReady}
                alt={`${pair.id} · ${occurrence}${aligned ? ' after stereo alignment' : difference ? ' in the channel difference' : ''}`} />
              {recorded && <>
                <p className="mt-2 mb-1 text-caption">Original stereo recording</p>
                <EvidenceAudio set={set} artifact={recorded} token={token} ready={mediaReady}
                  alt={`${pair.id} · ${occurrence} in the original recording`} />
                {original && <p className="mt-1 mb-0 text-caption text-text-muted">
                  Original stereo excerpt PTS [{original.source_start_pts}, {original.source_end_pts}) at {original.time_base}.
                </p>}
              </>}
              <p className="mt-2 mb-0 text-caption">Analyzed audio {seconds(span.start_sample / context.sample_rate)}–{seconds(span.end_sample / context.sample_rate)}.</p>
              {span.source_timestamp_ranges.map((range) => {
                const start = rationalSeconds(range.source_start_pts, range.time_base)
                const end = rationalSeconds(range.source_end_pts, range.time_base)
                return <p key={range.start_sample} className="mt-1 mb-0 text-caption text-text-muted">
                  Source PTS [{range.source_start_pts}, {range.source_end_pts}) at {range.time_base}.
                  {videoOriginS != null && start != null && end != null && start >= videoOriginS
                    && ` Video playback ${seconds(start - videoOriginS)}–${seconds(end - videoOriginS)}.`}
                </p>
              })}
            </div>
          })}</div>
        </article>)}</div>}
  </section>
}

function StereoTimingCheck({ set, token, mediaReady, videoOriginS }: Props) {
  const check = set.stereo_timing_change
  if (!check || check.source_sha256 !== set.source.sha256) return null
  const { artifacts, measurement, sample_rate: rate } = check
  const offset = (samples: number) => `${samples >= 0 ? '+' : ''}${(samples * 1000 / rate).toFixed(3)} ms`
  return <section id="media-evidence-stereo-timing-change" aria-labelledby="stereo-timing-heading" className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id="stereo-timing-heading" className="m-0 text-body font-medium">Temporary change in stereo timing</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">Compared the timing of the left and right channels across {seconds(check.input_samples / rate)} of continuous 48 kHz stereo audio. Changing speakers, microphone positions, or ordinary edits can cause a timing change. This finding carries no AI confidence weight.</p>
    {measurement ? <>
      <p className="mt-2 mb-0 text-small">The delay shifted from {offset(measurement.before_offset_samples)} to {offset(measurement.inside_offset_samples)} between {seconds(measurement.start_sample / rate)} and {seconds(measurement.end_sample / rate)} of analyzed audio, then returned to {offset(measurement.after_offset_samples)}. A positive delay means the right channel arrived later.</p>
      <div className="mt-3 space-y-3">
        {artifacts.timing_plot && <EvidencePicture set={set} token={token} ready={mediaReady} chart artifact={artifacts.timing_plot} alt="Measured delay between the stereo channels" />}
        {artifacts.waveform && <EvidencePicture set={set} token={token} ready={mediaReady} chart artifact={artifacts.waveform} alt="Left-channel waveform with the flagged interval highlighted" />}
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3 max-[767px]:grid-cols-1" onPlayCapture={event => {
        if (event.target instanceof HTMLAudioElement) {
          event.currentTarget.querySelectorAll('audio').forEach(audio => { if (audio !== event.target && !audio.paused) audio.pause() })
        }
      }}>
        {check.clips.map(clip => {
          const asset = artifacts[clip.role]
          const range = clip.source_timestamp_ranges[0]
          const start = range && rationalSeconds(range.source_start_pts, range.time_base)
          const end = range && rationalSeconds(range.source_end_pts, range.time_base)
          const label = { before: 'Before the change', during: 'During the change', after: 'After the change' }[clip.role]
          return <div key={clip.role}>
            {asset && <EvidenceAudio set={set} artifact={asset} token={token} ready={mediaReady} alt={`${label} · Original stereo audio`} />}
            {range && <p className="mt-1 mb-0 text-caption text-text-muted">
              Source PTS [{range.source_start_pts}, {range.source_end_pts}) at {range.time_base}.
              {videoOriginS != null && start != null && end != null && start >= videoOriginS
                && ` Video playback ${seconds(start - videoOriginS)}–${seconds(end - videoOriginS)}.`}
            </p>}
          </div>
        })}
      </div>
    </> : <p className="mt-2 mb-0 text-small">{check.status === 'measured'
      ? 'No qualifying temporary delay change was found in the checked audio.'
      : 'Too little sound was shared clearly between the two channels to assess a temporary delay change.'}</p>}
  </section>
}

export default function AudioRepetitionContext(props: Props) {
  const stereo = props.set.stereo_repetition
  return <><RepeatCheck {...props} context={props.set.repetition_context} />
    <RepeatCheck {...props} context={props.set.similarity_context} />
    {stereo?.source_sha256 === props.set.source.sha256 && <>
      {stereo.channels.map(row => <div key={row.channel}><RepeatCheck {...props} context={row.measurement} channel={row.channel} />
        <RepeatCheck {...props} context={row.gain_measurement} channel={row.channel} /></div>)}
      <RepeatCheck {...props} context={stereo.difference_measurement} channel="difference" />
      <RepeatCheck {...props} context={stereo.aligned_measurement} channel="aligned" />
    </>}
    <StereoTimingCheck {...props} />
  </>
}
