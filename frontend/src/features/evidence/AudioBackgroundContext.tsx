import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { EvidenceAudio, EvidenceLimits, EvidenceMeasurements, EvidencePicture, evidenceGrid } from './EvidenceMedia'
import { rationalSeconds, seconds, type AcousticContextEvent, type AudioEvidenceSet } from './multimediaPackets'

const labels: Record<AcousticContextEvent['category'], string> = {
  spectral_transition: 'Possible sound-content transition', energy_change: 'Sound level change', near_silence: 'Very quiet interval',
  clipping: 'Near-full-scale samples', upper_frequency_drop: 'Temporary upper-frequency drop',
}

export default function AudioBackgroundContext({ set, token, mediaReady, pauseAudio }: {
  set: AudioEvidenceSet; token: string; mediaReady: boolean; pauseAudio: () => void
}) {
  const { hash, search } = useLocation(), context = set.acoustic_context
  const [selectedId, setSelectedId] = useState(context?.events[0]?.id ?? null)
  const root = useRef<HTMLElement>(null)
  const events = context?.events ?? []
  useEffect(() => {
    const event = events.find((item) => hash === `#media-evidence-${item.id}`)
    if (event) { pauseAudio(); setSelectedId(event.id) }
  }, [hash, context]) // pauseAudio is a presentation callback; source/hash changes select the event.
  const selected = events.find((event) => event.id === selectedId) ?? events[0]
  const bound = context && ['chipmunk.acoustic-context/1', 'chipmunk.acoustic-context/2', 'chipmunk.acoustic-context/3'].includes(context.schema) && context.status === 'ready'
    && context.source.sha256 === set.source.sha256
    && context.source.pcm_sha256 === set.source.pcm_sha256
    && context.source.sample_rate === set.source.sample_rate && context.source.sample_rate > 0
    && context.source.samples === (set.coverage?.decoded_samples ?? set.source.samples)
  const select = (event: AcousticContextEvent) => {
    pauseAudio(); setSelectedId(event.id)
    requestAnimationFrame(() => {
      const article = Array.from(root.current?.querySelectorAll<HTMLElement>('article') ?? []).find((node) => node.id === `media-evidence-${event.id}`)
      article?.focus({ preventScroll: true }); article?.scrollIntoView?.({ block: 'start' })
    })
  }
  return <section ref={root} id="media-evidence-background" aria-label="Background and recording context" className="mt-5 rounded-md border border-border p-4 scroll-mt-20">
    <h3 className="m-0 text-body font-medium">Background &amp; recording context</h3>
    {!context ? <p className="mt-2 mb-0 text-small text-text-muted">Background and recording measurements were not captured for this saved result. Run a new analysis to include them; their absence does not clear the recording.</p>
      : !bound ? <p role="alert" className="mt-2 mb-0 text-small text-text-muted">The acoustic record does not match this analyzed audio input. Its charts and clips cannot be shown.</p>
        : <>
          <p className="mt-2 mb-0 text-small text-text-muted">Inspect changes in the mixed sound track, including possible sound-content transitions, quiet intervals and near-full-scale samples. Voice, music and environmental sound remain mixed together; this is not an isolated background track or a verdict about AI use.</p>
          <p className="mt-2 mb-0 font-mono text-caption text-text-muted">Measured input: 0.000 s–{seconds(context.source.samples / context.source.sample_rate)} · {context.bins.length} recorded acoustic blocks</p>
          {context.upper_frequency_drop?.status === 'unassessed' && <p className="mt-2 mb-0 text-small text-text-muted">The surrounding sound did not support a temporary upper-frequency comparison.</p>}
          {context.upper_frequency_drop?.status === 'no_supported_drop' && <p className="mt-2 mb-0 text-small text-text-muted">No temporary upper-frequency drop met the measurement rules in the comparable ranges.</p>}
          {context.summary && <EvidenceMeasurements values={[
            { label: 'Low-energy reference of the mixed track', value: context.summary.low_energy_reference_dbfs, unit: 'dBFS' },
            { label: 'Very quiet input', value: context.summary.near_silence_samples / context.source.sample_rate, unit: 's' },
            { label: 'Near-full-scale samples', value: context.summary.clipped_samples, unit: 'samples' },
          ]} />}
          <div className={`${evidenceGrid} mt-4`}>
            <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={context.overview.energy} token={token} ready={mediaReady} chart alt="Mixed-track sound level over the analyzed input" />
              <figcaption className="mt-2 text-caption text-text-muted">Sound level in the decoded input. A change can come from speech, music, distance, recording gain or editing.</figcaption></figure>
            <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={context.overview.spectrogram} token={token} ready={mediaReady} chart alt="Mixed-track frequency content over the analyzed input" />
              <figcaption className="mt-2 text-caption text-text-muted">Frequency content of the same sound track. This view does not identify or separate individual sound sources.</figcaption></figure>
          </div>
          {events.length ? <>
            <p className="mt-4 mb-2 text-small">Choose a recorded acoustic observation to compare its exact interval with the surrounding sound.</p>
            <div role="group" aria-label="Choose a background or recording observation" className="flex flex-wrap gap-2">
              {[...events].sort((a, b) => a.interval.start_sample - b.interval.start_sample).map((event) => <button key={event.id} type="button"
                aria-pressed={selected?.id === event.id} aria-controls={`media-evidence-${event.id}`}
                aria-label={`Inspect ${event.id}, ${labels[event.category]}, ${seconds(event.interval.start_sample / context.source.sample_rate)}–${seconds(event.interval.end_sample / context.source.sample_rate)}`}
                onClick={() => select(event)} className={`rounded-sm border px-3 py-2 text-left ${selected?.id === event.id ? 'border-accent bg-surface-2 ring-1 ring-accent' : 'border-border hover:border-accent'}`}>
                <span className="block text-small"><span className="font-mono mr-2">{event.id}</span>{labels[event.category]}</span>
                <span className="block mt-1 font-mono text-caption text-text-muted">{seconds(event.interval.start_sample / context.source.sample_rate)}–{seconds(event.interval.end_sample / context.source.sample_rate)}</span>
              </button>)}
            </div>
            {events.map((event) => <article key={event.id} id={`media-evidence-${event.id}`} hidden={event.id !== selected?.id} tabIndex={-1}
              aria-label={`Selected acoustic evidence ${event.id}`} className="mt-5 border-t border-border pt-4 scroll-mt-20">
              <h4 className="m-0 text-body font-medium"><span className="font-mono text-accent mr-3">{event.id}</span>{labels[event.category]}</h4>
              <p className="mt-2 mb-0 text-small">{event.observation}</p>
              <p className="mt-2 mb-0 font-mono text-caption text-text-muted">Analyzed input: {seconds(event.interval.start_sample / context.source.sample_rate)}–{seconds(event.interval.end_sample / context.source.sample_rate)} · samples [{event.interval.start_sample}, {event.interval.end_sample})</p>
              <div className={`${evidenceGrid} mt-4`}>
                <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={event.artifacts.waveform} token={token} ready={mediaReady} chart alt={`${event.id}: mixed-track waveform with the acoustic interval marked`} />
                  <figcaption className="mt-2 text-caption text-text-muted">The marked interval matches the selected clip. The surrounding waveform is listening context.</figcaption></figure>
                <figure className="m-0 min-w-0"><EvidencePicture set={set} artifact={event.artifacts.spectrogram} token={token} ready={mediaReady} chart alt={`${event.id}: frequency content around the acoustic interval`} />
                  <figcaption className="mt-2 text-caption text-text-muted">Recorded frequencies around this interval; no source separation is applied.</figcaption></figure>
              </div>
              <div className={`${evidenceGrid} mt-4`}>
                <EvidenceAudio set={set} artifact={event.artifacts.selected_clip} token={token} ready={mediaReady} alt={`${event.id}: exact acoustic interval`} />
                <EvidenceAudio set={set} artifact={event.artifacts.context_clip} token={token} ready={mediaReady} alt={`${event.id}: acoustic interval with surrounding sound`} />
              </div>
              <p className="mt-2 mb-0 text-caption text-text-muted">Context samples [{event.context_interval.start_sample}, {event.context_interval.end_sample}). Clip playback begins at zero; the source positions above remain unchanged.</p>
              <p className="mt-4 mb-0 text-small">{event.interpretation}</p>
              <EvidenceMeasurements values={event.observations} />
              <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">Original recording timestamps</summary>
                <ul className="pl-4 list-disc">{event.source_timestamp_ranges.map((range, index) => {
                  const a = rationalSeconds(range.source_start_pts, range.time_base), b = rationalSeconds(range.source_end_pts, range.time_base)
                  return <li key={index}>Samples [{range.start_sample}, {range.end_sample}): {a !== null && b !== null ? `${seconds(a)}–${seconds(b)}` : 'time conversion unavailable'} ({range.source_start_pts}–{range.source_end_pts} ticks, time base {range.time_base})</li>
                })}</ul>
              </details>
              {event.related_voice_packet_ids.length > 0 && <p className="mt-3 mb-0 text-caption text-text-muted">Overlapping voice intervals {event.related_voice_packet_ids.map((id, index) => <span key={id}>{index > 0 && ', '}<Link to={{ search, hash: `#media-evidence-${encodeURIComponent(id)}` }} className="underline underline-offset-2">{id}</Link></span>)}. Overlap does not establish a shared cause.</p>}
              <EvidenceLimits lines={event.limitations} />
            </article>)}
          </> : <p className="mt-4 mb-0 text-small text-text-muted">No acoustic interval met the saved selection rules. The overview remains available; this does not establish authenticity or a natural voice.</p>}
          {(context.omitted_events ?? 0) > 0 && <p className="mt-3 mb-0 text-caption text-text-muted">{context.omitted_events} further acoustic observations are omitted from this bounded presentation.</p>}
          {(context.omitted_continuity_windows ?? 0) > 0 && <p className="mt-3 mb-0 text-caption text-text-muted">{context.omitted_continuity_windows} overlapping or lower-ranked frequency comparisons are recorded in the export. Up to four distinct transition leads are shown.</p>}
          <EvidenceLimits lines={context.limitations} />
        </>}
  </section>
}
