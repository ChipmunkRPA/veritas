import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import type { AnalysisDetail, Finding, ModelCard, SignalV2 } from '../../lib/api/queries'
import { getImageEvidence } from '../evidence/imagePackets'
import { getMultimediaEvidence } from '../evidence/multimediaPackets'
import { anonymizeModelNames } from './checkNames'
import { findingSupport, type FindingSupportEvidence, type FindingSupportLink } from './findingSupport'

/** One query per findings list; it shares the evidence cards' query cache. */
export function useFindingEvidence(analysis: Pick<AnalysisDetail, 'id' | 'kind'>) {
  const supported = ['image', 'video', 'audio', 'document'].includes(analysis.kind)
  const result = useQuery<FindingSupportEvidence>({
    queryKey: ['analysis', analysis.id, 'evidence-packets'],
    queryFn: () => {
      if (analysis.kind === 'image') return getImageEvidence(analysis.id)
      if (analysis.kind === 'video' || analysis.kind === 'audio' || analysis.kind === 'document') return getMultimediaEvidence(analysis.id, analysis.kind)
      return Promise.reject(new Error('Source evidence is unavailable for this content type.'))
    },
    enabled: supported,
    staleTime: 30_000,
    retry: false,
  })
  return { evidence: result.data, pending: supported && result.isPending, failed: result.isError, retry: () => { void result.refetch() } }
}

export interface FindingSupportData {
  analysisId: string
  signals: readonly SignalV2[]
  evidence?: FindingSupportEvidence
  pending?: boolean
  failed?: boolean
  retry?: () => void
}

/** Read-only observations and links, never an examiner disposition or a new detection claim. */
export default function FindingSupportPanel({ finding, data, cards = [] }: {
  finding: Finding; data: FindingSupportData; cards?: readonly ModelCard[]
}) {
  const support = findingSupport({ ...data, finding })
  const describe = (text: string) => anonymizeModelNames(text, cards, data.signals)
  const packet = (item: FindingSupportLink) => (
    <li key={item.id} className="border-l-2 border-border pl-3">
      <Link to={item.href} data-testid="finding-source-support" className="inline-block min-h-8 text-small font-medium text-accent underline underline-offset-[3px] hover:text-accent-hover">
        {item.label}
      </Link>
      {item.summary && <p className="mt-0.5 mb-0 text-small text-text max-w-[76ch]">{describe(item.summary)}</p>}
      <p className="mt-1 mb-0 text-caption text-text-muted max-w-[84ch]">{item.association === 'shared_check' ? 'Related inspection from the same check. ' : ''}{item.scopeNote}</p>
    </li>
  )
  return (
    <div data-testid="finding-support" data-finding-id={finding.id} className="mt-3 w-full min-w-0">
      <p className="m-0 text-caption font-medium text-text-muted">Supporting evidence · <span className="font-mono">{finding.id}</span></p>
      {support.packets.length > 0 && <>
        <ul className="mt-2 mb-0 p-0 list-none space-y-3">{support.packets.slice(0, 2).map(packet)}</ul>
        {support.packets.length > 2 && <details className="mt-2 text-small">
          <summary className="cursor-pointer text-accent min-h-8">Show {support.packets.length - 2} more recorded locations</summary>
          <ul className="mt-2 mb-0 p-0 list-none space-y-3">{support.packets.slice(2).map(packet)}</ul>
        </details>}
      </>}
      {data.pending && <p role="status" className="mt-1 mb-0 text-caption text-text-muted">Loading linked source evidence…</p>}
      {data.failed && <p role="status" className="mt-1 mb-0 text-caption text-text-muted">
        Source evidence could not be loaded. Recorded check links remain available below.
        {data.retry && <button type="button" className="ml-2 min-h-8 text-accent underline underline-offset-[3px]" onClick={data.retry}>Retry evidence</button>}
      </p>}
      {!data.pending && !data.failed && support.packets.length === 0 && <p className="mt-1 mb-0 text-caption text-text-muted">
        {support.checks.length > 0 ? 'See the recorded checks below for the basis of this finding. No specific source region or interval is linked here.'
          : 'No linked source location or recorded check is available for this saved finding.'}
      </p>}
      {support.checks.length > 0 && <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
        {support.checks.map((item) => <Link key={item.id} to={item.href} data-testid="finding-check-support"
          className="inline-flex min-h-8 items-center text-caption text-accent underline underline-offset-[3px] hover:text-accent-hover">{item.label}</Link>)}
      </div>}
    </div>
  )
}
