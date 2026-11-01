import { EvidencePicture } from './EvidenceMedia'
import { seconds, type AudioEvidencePacket, type AudioEvidenceSet, type SampleInterval } from './multimediaPackets'

function withinInput(span: SampleInterval, samples: number) {
  return Number.isSafeInteger(span.start_sample) && Number.isSafeInteger(span.end_sample)
    && span.start_sample >= 0 && span.start_sample < span.end_sample && span.end_sample <= samples
}

export default function AudioInspectionTimeline({ set, token, mediaReady, selectedId, onSelect }: {
  set: AudioEvidenceSet; token: string; mediaReady: boolean; selectedId: string | null
  onSelect: (packet: AudioEvidencePacket) => void
}) {
  const rate = set.source.sample_rate
  const samples = set.coverage?.decoded_samples ?? set.source.samples
  const validClock = typeof rate === 'number' && Number.isFinite(rate) && rate > 0
    && typeof samples === 'number' && Number.isSafeInteger(samples) && samples > 0
  const packets = [...set.packets].sort((a, b) => a.interval.start_sample - b.interval.start_sample)
  const windows = validClock ? (set.coverage?.windows ?? []).filter((window) => withinInput(window, samples)) : []
  const position = (span: SampleInterval) => ({ left: `${100 * span.start_sample / samples!}%`,
    width: `${100 * (span.end_sample - span.start_sample) / samples!}%` })
  return <section aria-label="Audio interval navigator" className="mt-5 rounded-md border border-border bg-surface-2 p-4">
    <h3 className="m-0 text-body font-medium">Where to listen</h3>
    <p className="mt-2 mb-4 text-small text-text-muted">{packets.length} saved inspection interval(s). Choose a marked interval to see its waveform, hear the exact scored audio and read the basis for the flag.</p>
    {set.overview?.waveform && <figure className="m-0">
      <EvidencePicture set={set} artifact={set.overview.waveform} token={token} ready={mediaReady} chart alt="Waveform of the analyzed audio input" />
      <figcaption className="mt-2 text-caption text-text-muted">Recorded waveform of the analyzed input. Its shape is listening context, not proof of synthesis.</figcaption>
    </figure>}
    {validClock ? <>
      <div className="mt-5 flex justify-between text-caption font-mono text-text-muted" aria-label="Analyzed input time axis">
        <span>{seconds(0)}</span><span>{seconds(samples / rate)}</span>
      </div>
      <div className="mt-1 relative h-4 rounded-sm bg-border overflow-hidden" role="img" aria-label="Scored audio coverage; unfilled gaps have no score">
        {windows.map((window, index) => <span key={index} className="absolute h-full bg-text-muted/60" style={position(window)} />)}
      </div>
      <p className="mt-2 mb-3 text-caption text-text-muted">Gray marks recorded scoring windows. Gaps have no score and have not been cleared. Each row below marks one saved inspection interval on this same decoded-audio clock.</p>
    </> : <p className="mt-3 text-small text-text-muted">The analyzed sample clock is unavailable. Exact interval placement cannot be shown.</p>}
    <div className="space-y-2" role="group" aria-label="Choose an audio inspection interval">
      {packets.map((packet) => {
        const placed = validClock && withinInput(packet.interval, samples)
        const response = (set.coverage?.windows ?? []).filter((window) => window.start_sample === packet.interval.start_sample
          && window.end_sample === packet.interval.end_sample && Number.isFinite(window.score)).map((window) => window.score)
        const active = selectedId === packet.id
        const range = placed ? `${seconds(packet.interval.start_sample / rate)}–${seconds(packet.interval.end_sample / rate)}` : 'Timing unavailable'
        return <button key={packet.id} type="button" aria-pressed={active} aria-controls={`media-evidence-${packet.id}`}
          aria-label={`Inspect ${packet.id}, ${range}`} onClick={() => onSelect(packet)}
          className={`w-full text-left rounded-sm border px-3 py-2.5 transition-colors ${active ? 'border-accent bg-surface ring-1 ring-accent' : 'border-border bg-surface hover:border-accent'}`}>
          <span className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 text-small">
            <span><span className="font-mono font-medium mr-2">{packet.id}</span>{range}</span>
            <span className="text-caption text-text-muted">{response.length ? `Response ${Math.max(...response).toFixed(3)}` : 'Recorded inspection clue'}{active ? ' · Selected' : ''}</span>
          </span>
          {placed && <span aria-hidden="true" className="relative block h-2 mt-2 rounded-sm bg-border overflow-hidden">
            <span className={`absolute h-full rounded-sm ${active ? 'bg-accent' : 'bg-amber-500/70'}`} style={position(packet.interval)} />
          </span>}
        </button>
      })}
    </div>
    <p className="mt-3 mb-0 text-caption text-text-muted">Responses are uncalibrated inspection scores, not probabilities that a voice was replaced. Overlapping intervals are not independent confirmations.</p>
    {set.overview?.scores && <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">All recorded window responses</summary>
      <div className="mt-3"><EvidencePicture set={set} artifact={set.overview.scores} token={token} ready={mediaReady} chart alt="Recorded voice-check window responses over the analyzed input" /></div>
    </details>}
  </section>
}
