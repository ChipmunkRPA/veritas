import { EvidenceAudio, evidenceGrid } from './EvidenceMedia'
import { rationalSeconds, seconds, type AudioEvidenceSet } from './multimediaPackets'

export default function AudioRepetitionContext({ set, token, mediaReady, videoOriginS }: {
  set: AudioEvidenceSet; token: string; mediaReady: boolean; videoOriginS?: number | null
}) {
  const context = set.repetition_context
  if (!context || context.source_sha256 !== set.source.sha256 || context.pcm_sha256 !== set.source.pcm_sha256
    || context.sample_rate !== set.source.sample_rate || context.input_samples !== set.source.samples) return null
  return <section id="media-evidence-repetition" aria-labelledby="media-evidence-repetition-heading" className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id="media-evidence-repetition-heading" className="m-0 text-body font-medium">Exact repeated audio</h3>
    <p className="mt-2 mb-0 text-small text-text-muted">This check compares samples in continuous 2.5–4 second mono inputs. Loops, recurring alerts, and editing can all repeat audio. Matches carry no AI confidence weight.</p>
    {context.status !== 'measured'
      ? <p className="mt-2 mb-0 text-small">{context.status === 'ambiguous'
        ? 'The search reached its comparison limit before it could establish a result.'
        : 'This input could not be checked within the supported duration, source timing, and signal conditions.'}</p>
      : context.pairs.length === 0
        ? <p className="mt-2 mb-0 text-small">No exact active repeat of at least 0.15 seconds was found in the checked input.</p>
        : <div className="mt-3 space-y-4">{context.pairs.map((pair) => <article key={pair.id} id={`media-evidence-${pair.id}`} className="border-t border-border pt-3 scroll-mt-20">
          <h4 className="m-0 text-small font-medium">{pair.id} · {seconds(pair.sample_count / context.sample_rate)} of identical samples</h4>
          <div className={`${evidenceGrid} mt-3`}>{(['first', 'second'] as const).map((side) => {
            const span = pair[side]
            return <div key={side}>
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
