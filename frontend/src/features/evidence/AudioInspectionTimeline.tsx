import { EvidencePicture } from './EvidenceMedia'
import { COMBINED_SPEECH_REFERENCE } from '../analysis/speechReference'
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
  const calibrated = (window: (typeof windows)[number]) => set.coverage?.response_semantics === 'speech-window-responses/2'
    && window.response_reference?.kind === 'balanced-reference-ai-estimate'
    && window.response_reference.calibration_sha256 === COMBINED_SPEECH_REFERENCE
    && window.response_reference.reference_prior_ai === .5 && window.response_reference.scope === 'complete-4s-speech-window'
    && rate === 16000 && samples === 64000 && window.start_sample === 0 && window.end_sample === 64000
    && Number.isFinite(window.score) && window.score >= 0 && window.score <= 1
    && window.response_reference.direction === (window.score > .5 ? 'ai' : window.score < .5 ? 'original' : 'inconclusive')
  const calibratedCount = windows.filter(calibrated).length
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
        const responses = windows.filter((window) => window.start_sample === packet.interval.start_sample
          && window.end_sample === packet.interval.end_sample && Number.isFinite(window.score))
        const reference = responses.length === 1 && calibrated(responses[0]) ? responses[0] : null
        const response = responses.map((window) => window.score)
        const active = selectedId === packet.id
        const range = placed ? `${seconds(packet.interval.start_sample / rate)}–${seconds(packet.interval.end_sample / rate)}` : 'Timing unavailable'
        return <button key={packet.id} type="button" aria-pressed={active} aria-controls={`media-evidence-${packet.id}`}
          aria-label={`Inspect ${packet.id}, ${range}`} onClick={() => onSelect(packet)}
          className={`w-full text-left rounded-sm border px-3 py-2.5 transition-colors ${active ? 'border-accent bg-surface ring-1 ring-accent' : 'border-border bg-surface hover:border-accent'}`}>
          <span className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1 text-small">
            <span><span className="font-mono font-medium mr-2">{packet.id}</span>{range}</span>
            <span className="text-caption text-text-muted">{reference ? `AI reference estimate ${(100 * reference.score).toFixed(0)}%` : response.length ? `Response ${Math.max(...response).toFixed(3)}` : 'Recorded inspection clue'}{active ? ' · Selected' : ''}</span>
          </span>
          {placed && <span aria-hidden="true" className="relative block h-2 mt-2 rounded-sm bg-border overflow-hidden">
            <span className={`absolute h-full rounded-sm ${active ? 'bg-accent' : 'bg-amber-500/70'}`} style={position(packet.interval)} />
          </span>}
        </button>
      })}
    </div>
    <p className="mt-3 mb-0 text-caption text-text-muted">{calibratedCount
      ? `${calibratedCount === windows.length ? 'These speech estimates use' : 'Labelled reference estimates use'} a balanced AI/original reference population. They are not probabilities that this particular file was altered and do not assess background synthesis.${calibratedCount < windows.length ? ' Other responses are uncalibrated inspection scores on a different scale.' : ''}`
      : 'Responses are uncalibrated inspection scores, not probabilities that a voice was replaced.'} Overlapping intervals are not independent confirmations.</p>
    {set.overview?.scores && <details className="mt-4 text-small text-text-muted"><summary className="cursor-pointer">All recorded window responses</summary>
      <div className="mt-3"><EvidencePicture set={set} artifact={set.overview.scores} token={token} ready={mediaReady} chart alt="Recorded voice-check window responses over the analyzed input" /></div>
    </details>}
  </section>
}
