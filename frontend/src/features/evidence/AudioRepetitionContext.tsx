import { EvidenceAudio, EvidencePicture, evidenceGrid } from './EvidenceMedia'
import { rationalSeconds, seconds, type AudioEvidenceSet, type AudioGainRepetitionContext, type AudioRepetitionContext as ExactContext, type AudioSimilarityContext } from './multimediaPackets'

type Props = {
  set: AudioEvidenceSet; token: string; mediaReady: boolean; videoOriginS?: number | null
}

function RepeatCheck({ set, token, mediaReady, videoOriginS, context, channel }: Props & {
  context?: ExactContext | AudioSimilarityContext | AudioGainRepetitionContext; channel?: 'left' | 'right'
}) {
  if (!context || context.source_sha256 !== set.source.sha256 || (!channel && context.pcm_sha256 !== set.source.pcm_sha256)
    || context.sample_rate !== set.source.sample_rate || context.input_samples !== set.source.samples) return null
  const gain = context.schema === 'chipmunk.audio-gain-repetition/1'
  const similar = !gain && context.schema !== 'chipmunk.audio-exact-repetition/1'
  const waveform = context.schema === 'chipmunk.audio-repeat-similarity/2'
  const method = gain
    ? 'This check compares the recorded channel for exact repeats after a fixed level or polarity change. Mixing new sound into that channel can hide repeats.'
    : waveform
      ? 'The exact and frequency-band searches yielded no reportable pair. A second similarity check compared waveform patterns in the same continuous 2.5–4 second mono input. Background noise can mask repeats.'
      : similar
        ? 'This check compares patterns across frequency bands in continuous 2.5–4 second mono inputs. Background noise can mask repeats.'
        : channel ? 'This check compares the recorded channel before it is mixed into mono.' : 'This check compares samples in continuous 2.5–4 second mono inputs.'
  const id = channel ? `media-evidence-${channel}-${gain ? 'gain-' : ''}repetition` : similar ? 'media-evidence-similar-audio' : 'media-evidence-repetition'
  const title = channel ? `${gain ? 'Gain-adjusted' : 'Exact'} repeats in the ${channel} channel`
    : similar ? 'Similar repeated audio' : 'Exact repeated audio'
  return <section id={id} aria-labelledby={`${id}-heading`} className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id={`${id}-heading`} className="m-0 text-body font-medium">{title}</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">{method} Loops, recurring alerts, and editing can all repeat audio. Matches carry no AI confidence weight.</p>
    {context.status !== 'measured'
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
            artifact={pair.artifacts.waveform} alt={`${pair.id} · Waveform shared by both occurrences`} />}
          <div className={`${evidenceGrid} mt-3`}>{(['first', 'second'] as const).map((side) => {
            const span = pair[side]
            const chart = 'scaled_occurrence' in pair ? pair.artifacts[`${side}_waveform`] : undefined
            return <div key={side}>
              {chart && <EvidencePicture set={set} token={token} ready={mediaReady} chart
                artifact={chart} alt={`${pair.id} · ${side === 'first' ? 'First' : 'Second'} occurrence waveform`} />}
              <EvidenceAudio set={set} artifact={pair.artifacts[`${side}_clip`]} token={token} ready={mediaReady}
                alt={`${pair.id} · ${side === 'first' ? 'First' : 'Second'} occurrence`} />
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

export default function AudioRepetitionContext(props: Props) {
  const stereo = props.set.stereo_repetition
  return <><RepeatCheck {...props} context={props.set.repetition_context} />
    <RepeatCheck {...props} context={props.set.similarity_context} />
    {stereo?.schema === 'chipmunk.stereo-exact-repetition/1' && stereo.source_sha256 === props.set.source.sha256
      && stereo.channels.map(row => <div key={row.channel}><RepeatCheck {...props} context={row.measurement} channel={row.channel} />
        <RepeatCheck {...props} context={row.gain_measurement} channel={row.channel} /></div>)}</>
}
