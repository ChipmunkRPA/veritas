import { ErrorState, SkeletonRows } from '../../components/ui'
import AudioEvidenceContent from './AudioEvidenceContent'
import DocumentEvidenceContent from './DocumentEvidenceContent'
import VideoEvidenceContent from './VideoEvidenceContent'
import VideoEvidenceWorkspace from './VideoEvidenceWorkspace'
import { EvidenceLimits } from './EvidenceMedia'
import { hasNestedVideoEvidence, useMultimediaEvidence, type EvidenceKind, type MultimediaEvidenceSet } from './multimediaPackets'

const labels = { video: 'Video evidence', audio: 'Audio evidence', document: 'Text and document evidence' }
export function MultimediaEvidenceContent({ set, token, mediaReady }: { set: MultimediaEvidenceSet; token: string; mediaReady: boolean }) {
  const omitted = 'omitted_candidates' in set ? set.omitted_candidates : set.omitted_windows
  const nestedVideo = set.schema === 'chipmunk.video-evidence/1' && hasNestedVideoEvidence(set.version)
  const nestedCandidates = nestedVideo && ((set.audio?.packets.length ?? 0) + (set.av?.packets.length ?? 0) + (set.audio?.acoustic_context?.events.length ?? 0)) > 0
  const stereo = nestedVideo ? set.audio?.stereo_context : set.schema === 'chipmunk.audio-evidence/1' ? set.stereo_context : undefined
  const monoLevelDrop = stereo?.windows.some((window) => window.downmix?.strong_attenuation)
  const audio = set.schema === 'chipmunk.video-evidence/1' ? set.audio : set.schema === 'chipmunk.audio-evidence/1' ? set : undefined
  const repeatedAudio = audio?.repetition_context?.pairs.length || audio?.similarity_context?.pairs.length
  const shortSpeechFlag = set.schema === 'chipmunk.audio-evidence/1' && set.short_voice_review?.status === 'flagged'
  const repeatedText = set.schema === 'chipmunk.document-evidence/1' && set.repeated_passages?.status === 'matched'
  return <div data-testid="multimedia-evidence-content" data-evidence-set={set.id ?? ''} data-evidence-hash={set.sha256 ?? ''}>
    {set.status !== 'ready' ? <p role="status" className="mt-3 text-small text-text-muted">
      Detailed evidence has not been prepared for this result. Run a new analysis to capture the exact inputs and their source positions. Unavailable evidence does not establish authenticity.
    </p> : <>
      <p className="mt-3 mb-0 text-body text-text-muted">Inspect the exact inputs behind the recorded responses, with their source context and limits. These observations do not prove that AI was used.</p>
      {set.packets.length === 0 && !nestedCandidates && !monoLevelDrop && !repeatedAudio && !shortSpeechFlag && !repeatedText && <p data-testid="multimedia-evidence-empty" className="mt-3 mb-0 text-small text-text-muted">
        No inspection candidate was selected from the recorded inputs. This does not establish authenticity; check the coverage and limitations below.
      </p>}
      {set.schema === 'chipmunk.video-evidence/1' ? nestedVideo
        ? <VideoEvidenceWorkspace set={set} token={token} mediaReady={mediaReady} />
        : <VideoEvidenceContent set={set} token={token} mediaReady={mediaReady} />
        : set.schema === 'chipmunk.audio-evidence/1' ? <AudioEvidenceContent set={set} token={token} mediaReady={mediaReady} />
          : <DocumentEvidenceContent set={set} token={token} mediaReady={mediaReady} />}
    </>}
    {omitted > 0 && <p className="mt-4 mb-0 text-small text-text-muted">{omitted} additional candidate(s) are not included in this bounded evidence set.</p>}
    <EvidenceLimits lines={set.limitations} />
    {set.status === 'ready' && set.sha256 && <details className="mt-5 text-small text-text-muted"><summary className="cursor-pointer">Evidence record</summary>
      <p className="mt-2 mb-0">The source, observations and assets are bound to this saved evidence version. JSON exports include the same evidence record.</p>
      <dl className="mt-2 mb-0 space-y-2 text-caption">
        <div><dt>Evidence SHA-256</dt><dd className="m-0 font-mono break-all">{set.sha256}</dd></div>
        <div><dt>Original file SHA-256</dt><dd className="m-0 font-mono break-all">{set.source.sha256}</dd></div>
      </dl>
    </details>}
  </div>
}
export default function MultimediaEvidenceCards({ analysisId, kind, token, mediaReady }: { analysisId: string; kind: EvidenceKind; token: string; mediaReady: boolean }) {
  const evidence = useMultimediaEvidence(analysisId, kind)
  return <section id="section-media-evidence" aria-labelledby="media-evidence-heading" className="scroll-mt-14" data-testid="media-evidence">
    <h2 id="media-evidence-heading" className="m-0 eyebrow">{labels[kind]}</h2>
    {evidence.isPending ? <SkeletonRows className="mt-4" />
      : evidence.error ? <ErrorState className="mt-3" onRetry={() => void evidence.refetch()}>Could not load the recorded evidence. Other analysis results remain available.</ErrorState>
        : evidence.data && <MultimediaEvidenceContent key={evidence.data.id ?? evidence.data.status} set={evidence.data} token={token} mediaReady={mediaReady} />}
  </section>
}
