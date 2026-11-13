import MediaAccessNotice from '../../components/MediaAccessNotice'
import { useEffect, useRef, useState } from 'react'
import { Link, Outlet, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { MoreHorizontal } from 'lucide-react'
import { api, hasApiKey } from '../../lib/api'
import { isTerminal, qk, useAnalysis, useCase, useDetectorRuns, useExhibits, useFindings, useMe, useModelCards, useSignals } from '../../lib/api/queries'
import { ApiError } from '../../lib/api/client'
import { useMediaQuery } from '../../lib/useMediaQuery'
import NotFound from '../../components/NotFound'
import { EXPERT_SECTIONS, defaultView, resolveView, storeView, storedView, type ResultView } from './view'
import { useAnalysisEvents } from '../../lib/api/sse'
import { fmtBytes, fmtDateLong, fmtSeconds, shortHash } from '../../lib/format'
import { cx } from '../../lib/cx'
import { Button, ButtonLink, CodeChip, ErrorState, Eyebrow, Menu, Skeleton, SkeletonRows } from '../../components/ui'
import LegacyResult from '../legacy/LegacyResult'
import DocumentShare from '../document/DocumentShare'
import ManuscriptView from '../document/ManuscriptView'
import SignalsLedger from '../signals/SignalsLedger'
import ExhibitPreview from '../viewer/ExhibitPreview'
import VideoExhibitPreview from '../video/VideoExhibitPreview'
import AudioExhibit from '../audio/AudioExhibit'
import { describeFacts, type AudioFacts } from '../audio/model'
import AnswerModule from './AnswerModule'
import { estimateOf } from './answer'
import { findingsFailed } from './vocabulary'
import HowReached from './ContributionWaterfall'
import { deviceWords } from '../../lib/recordWords'
import CoveragePanel from './CoveragePanel'
import FindingsList from './FindingsList'
import KeyFindings from './KeyFindings'
import ThumbnailRegionEvidence from './ThumbnailRegionEvidence'
import WholeImageReview from './WholeImageReview'
import ProgressRail from './ProgressRail'
import CancelAnalysis from './CancelAnalysis'
import CustodianHashNotice from '../cases/CustodianHashNotice'
import ProvenancePanel from './ProvenancePanel'
import TechnicalAppendix from './TechnicalAppendix'
import MethodsPanel from './MethodsPanel'
import ExportDialog from '../reports/ExportDialog'
import RerunDialog from './RerunDialog'
import { useSectionInView } from './sectionInView'
import { requestRunAgain, unpublishedRevision } from '../../lib/api/rerun'
import { deletable, mayDelete, mayExamine } from '../../lib/api/roles'
import RevisionDeltas from './RevisionDeltas'
import ImageEvidenceCards from '../evidence/ImageEvidenceCards'
import ImageReferenceComparison, { useImageReferenceAvailability } from '../evidence/ImageReferenceComparison'
import DocumentArchiveComparison from '../evidence/DocumentArchiveComparison'
import DocumentRevisionComparison from '../evidence/DocumentRevisionComparison'
import MediaReferenceComparison from '../evidence/MediaReferenceComparison'
import MultimediaEvidenceCards from '../evidence/MultimediaEvidenceCards'
import ExpertAnalysisInvitation from './ExpertAnalysisInvitation'
import AutomatedResultStatus from './AutomatedResultStatus'

/** Passed to the nested section routes, which scroll once the section they target has rendered. */
export interface WorkspaceContext {
  ready: boolean
}

const SECTIONS: { path: string; label: string }[] = [
  { path: '', label: 'Summary' },
  { path: 'findings', label: 'Findings' },
]
/** The self-service result exposes the automated assessment and its supporting evidence. */
const V2_SECTIONS = (exhibit: boolean, revision: boolean, image = false, media = false): { path: string; label: string }[] => [
  ...SECTIONS,
  ...(image ? [{ path: 'image-evidence', label: 'Image evidence' }] : []),
  ...(media ? [{ path: 'media-evidence', label: 'Source evidence' }] : []),
  ...(exhibit ? [{ path: 'exhibit', label: 'Exhibit' }] : []),
  { path: 'reached', label: 'How reached' },
  { path: 'ledger', label: 'Evidence' },
  { path: 'provenance', label: 'Provenance' },
  { path: 'methods', label: 'Methods' },
  { path: 'appendix', label: 'Appendix' },
  // Q-4c: a revision that supersedes another shows its deltas (CC-03).
  ...(revision ? [{ path: 'revision', label: 'Revision' }] : []),
]

/** Every section a result page can hold, in the order of the index; a page observes those it shows. */
const ALL_SECTIONS = ['summary', 'findings', 'image-evidence', 'media-evidence', 'exhibit', 'reached', 'ledger', 'provenance', 'methods', 'appendix', 'revision'] as const

/** Analysis workspace (ARCHITECTURE §7.1–7.3, UX §4.2): header with the review state line, a sticky section index, the progress
 *  rail while the analysis runs, then one scrolling document whose sections the nested routes deep-link to. Analyses with a v2
 *  decision get the answer module and key findings (E-2), the exhibit preview that opens the evidence viewer or the video
 *  workspace (E-4, E-5) or the manuscript, how the estimate was reached, coverage and the evidence ledger, provenance, methods and
 *  the technical appendix (E-3, E-7). Expert services are arranged separately through the contact invitation.
 *  v1 results render through `legacy/` only (E-8). */
export default function AnalysisWorkspace() {
  const { id = '' } = useParams()
  const nav = useNavigate()
  const qc = useQueryClient()
  const q = useAnalysis(id)
  const a = q.data
  const done = a?.status === 'done'
  const v2 = done && a?.decision != null
  const imageComparisonAvailable = useImageReferenceAvailability(id, done && a?.kind === 'image')
  const events = useAnalysisEvents(id, !!a && !isTerminal(a.status))
  const legacy = useQuery({ queryKey: qk.legacy(id), queryFn: () => api.get(id), enabled: done, staleTime: Infinity })
  const findings = useFindings(id, done)
  const signals = useSignals(id, v2)
  const exhibits = useExhibits(id, v2)
  const runs = useDetectorRuns(id, v2)
  const cards = useModelCards(v2)
  const tok = useQuery({ queryKey: qk.downloadToken(id), queryFn: () => api.downloadToken(id), enabled: done, staleTime: 5 * 60_000 })
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, staleTime: 15000, retry: 0 })
  const del = useMutation({
    mutationFn: api.remove,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.analyses() })
      void qc.invalidateQueries({ queryKey: qk.cases() })
      nav('/analyses')
    },
  })
  // N-19a ("Workflow"): a cancelled run is run again from its retained source, as the next revision of the same file
  const again = useMutation({
    mutationFn: () => requestRunAgain(id),
    onSuccess: (revision) => {
      void qc.invalidateQueries({ queryKey: qk.analysis(id), exact: true })
      void qc.invalidateQueries({ queryKey: qk.analyses() })
      void qc.invalidateQueries({ queryKey: qk.cases() })
      if (revision.case_id) void qc.invalidateQueries({ queryKey: qk.caseAnalyses(revision.case_id) })
      nav(`/analyses/${revision.id}`)
    },
  })
  const [confirming, setConfirming] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [rerunning, setRerunning] = useState(false)
  const [rerunRefused, setRerunRefused] = useState(false)
  const me = useMe()
  const toggleInHeader = useMediaQuery('(min-width: 640px) and (max-width: 1199px)')
  const linksRef = useRef<HTMLDivElement>(null)
  // PRD UX-01 Plain / Expert: the URL parameter carries the choice and the last choice is remembered (`view.ts` has the order).
  const location = useLocation()
  const [params] = useSearchParams()
  const caseQ = useCase(a?.case_id ?? '')
  const section = location.pathname.split('/')[3] ?? ''
  // N-19b (design review DS-11): the index marks the section the reader is in, which follows scrolling; until something is
  // observed it is the section of the address
  const inView = useSectionInView(ALL_SECTIONS, done)
  const currentSection = inView ?? (section || 'summary')
  // The current section's link stays in view where the links scroll sideways (375 px), without moving the page.
  useEffect(() => {
    const row = linksRef.current
    const link = row?.querySelector<HTMLElement>(`[data-section="${currentSection}"]`)
    if (!row || !link) return
    const left = link.offsetLeft - row.offsetLeft
    if (left < row.scrollLeft || left + link.offsetWidth > row.scrollLeft + row.clientWidth) row.scrollLeft = Math.max(0, left - 16)
  })
  const fallback = me.isFetched && (!a?.case_id || caseQ.isFetched) ? defaultView(me.data?.role, caseQ.data?.implicit) : null
  const view = v2 ? resolveView(section, params.get('view'), storedView(), fallback) : 'expert'
  const expert = view === 'expert'
  const chooseView = (v: ResultView, hash = '') => {
    storeView(v)
    const next = new URLSearchParams(params)
    next.set('view', v)
    const pathname = v === 'plain' && EXPERT_SECTIONS.has(section) ? `/analyses/${id}` : location.pathname
    nav({ pathname, search: `?${next.toString()}`, hash }, { replace: true })
  }
  // A key finding links to its ledger row (`#signal-…`), which only the expert view renders: open Expert, then scroll to the row.
  const hash = location.hash
  useEffect(() => {
    if (view === 'plain' && hash.startsWith('#signal-')) chooseView('expert', hash)
    // Only a new hash or view triggers the switch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, hash])
  const ledgerReady = signals.isSuccess && runs.isSuccess
  // The row is brought back to the centre when what is drawn above it arrives later (the exhibit, the findings, the summary):
  // on a loaded host the exhibits came after the signals, and the row had moved by the height of the exhibit section.
  const aboveReady = exhibits.isFetched && findings.isFetched && legacy.isSuccess
  useEffect(() => {
    if (!expert || !ledgerReady || !hash.startsWith('#signal-')) return
    document.getElementById(decodeURIComponent(hash.slice(1)))?.scrollIntoView({ block: 'center' })
  }, [expert, ledgerReady, aboveReady, hash])

  if (q.isPending) {
    return (
      <div aria-busy="true">
        <Skeleton w={120} h={12} /><div className="mt-3"><Skeleton w={420} h={32} /></div><div className="mt-3"><Skeleton w={320} h={12} /></div>
        <div className="mt-12 border-t border-border-strong pt-8"><Skeleton w={72} h={72} /></div>
      </div>
    )
  }
  if (q.error instanceof ApiError && q.error.status === 404) return <NotFound what="analysis" />
  if (q.error || !a) {
    return (
      <ErrorState onRetry={() => void q.refetch()}>
        {q.error?.message ?? 'Not found'} <span className="text-text-muted mx-1">·</span> <Link to="/analyses" className="text-accent underline underline-offset-[3px]">Back to analyses</Link>
      </ErrorState>
    )
  }

  // N-19a ("Actions by role"): a superseded revision offers neither a re-run nor an export (the server refuses both; its
  // state line leads to the revision that replaced it), and a "Re-run…" that is switched off says why
  const superseded = a.review?.state === 'superseded'
  const rerunBlocked = superseded || a.review?.state === 'withdrawn'
  const rerunWhy = a.review?.state === 'withdrawn' ? 'A withdrawn revision cannot be re-run. Take the file into the case again to analyse it anew.' : undefined
  // L-7: an action the caller's role cannot take is not offered (the server refuses it either way)
  const examines = mayExamine(me.data, caseQ.data?.my_role)
  const deletes = mayDelete(me.data) && deletable(a?.review?.state, a?.locked_at ?? a?.review?.locked_at)  // N-19a: not on a signed analysis
  // K-2: a revision cancelled or failed before it published supersedes nothing; it can be re-run from its stored pins
  const retryable = unpublishedRevision(a.status, a.supersedes_id) && !rerunBlocked && examines
  const retryRevision = retryable && (
    <div data-testid="retry-revision" className="mt-4">
      <p className="m-0 text-small text-text">
        Revision {a.revision} published nothing, so it replaced nothing: <Link to={`/analyses/${a.supersedes_id}`} className="text-accent underline underline-offset-[3px] decoration-1 hover:decoration-2">analysis <span className="font-mono">{a.supersedes_id!.slice(0, 8)}</span></Link> stays the current revision.
      </p>
      <Button variant="ghost" className="mt-4" onClick={() => setRerunning(true)}>Re-run…</Button>
      <RerunDialog open={rerunning} analysis={a} me={me.data} onClose={() => setRerunning(false)} />
    </div>
  )
  const r = legacy.data?.result ?? null
  const audioFacts = a.kind === 'audio' ? (r?.audio as AudioFacts | undefined) ?? null : null
  const dims = a.kind === 'image' && r?.image ? `${r.image.width}×${r.image.height} ${(r.image.format ?? '').toUpperCase()}`
    : a.kind === 'video' && r?.video?.width ? `${r.video.width}×${r.video.height}`
      : a.kind === 'audio' ? describeFacts(audioFacts) : null
  const elapsedRun = a.started_at && a.finished_at ? (Date.parse(a.finished_at) - Date.parse(a.started_at)) / 1000 : null
  const engine = health.data?.device && health.data.version ? `${deviceWords(health.data.device)} · Chipmunk Verify ${health.data.version}` : null
  const t = tok.data ?? ''
  const ready = done ? legacy.isSuccess && findings.isFetched && (!v2 || (signals.isFetched && runs.isFetched && exhibits.isFetched)) : true
  const rail = <ProgressRail analysis={a} events={events} />
  const evidenceFirst = !expert && (a.kind === 'video' || a.kind === 'audio' || a.kind === 'document')
  const mediaLabel = a.kind === 'video' ? 'Video & audio' : a.kind === 'audio' ? 'Audio intervals' : 'Text passages'
  const sections = v2 ? V2_SECTIONS(a.kind === 'image' || a.kind === 'video' || a.kind === 'audio', a.supersedes_id != null, a.kind === 'image', a.kind === 'video' || a.kind === 'audio' || a.kind === 'document')
    .filter((s) => expert || !EXPERT_SECTIONS.has(s.path)).map((s) => s.path === 'media-evidence' ? { ...s, label: mediaLabel } : s) : SECTIONS
  if (v2 && evidenceFirst) {
    const index = sections.findIndex((item) => item.path === 'media-evidence')
    if (index > 1) sections.splice(1, 0, ...sections.splice(index, 1))
  }
  // With an API key configured, media requests carry the signed token: wait for it before rendering stills, frames and thumbnails.
  const tokenReady = !hasApiKey || !!t
  const mediaNotice = <MediaAccessNotice ready={tokenReady} failed={!!tok.error} pending={tok.isFetching} onRetry={() => void tok.refetch()} />
  const hr = <hr className="border-0 border-t border-border" />

  const exhibitNotice = exhibits.error ? (
    exhibits.isFetching
      ? <p role="status" className="mt-3 mb-0 text-small text-text-muted">Reloading the exhibit… Other results remain available.</p>
      : <ErrorState className="mt-3" onRetry={() => void exhibits.refetch()}>Could not load the exhibit. Other results remain available.</ErrorState>
  ) : null

  const exhibit = a.kind === 'document'
    ? r?.sentences && r.text != null && r.paragraphs
      ? <ManuscriptView sentences={r.sentences} text={r.text} paragraphs={r.paragraphs} />
      : null
    : a.kind !== 'image' && a.kind !== 'video' && a.kind !== 'audio' ? null
      : !tokenReady
        ? <section id="section-exhibit" className="scroll-mt-14"><h2 className="m-0 eyebrow">Exhibit</h2>
          <p className="mt-3 mb-0 text-small text-text-muted">The preview will appear when secure media links are ready.</p></section>
      : exhibits.data !== undefined && findings.isFetched
        ? a.kind === 'image'
          ? <ExhibitPreview analysis={a} exhibits={exhibits.data} findings={findings.data ?? []} token={t} notice={exhibitNotice} />
          : a.kind === 'audio'
            ? <AudioExhibit analysis={a} facts={audioFacts} findings={findings.data ?? []} token={t} />
            : <VideoExhibitPreview analysis={a} findings={findings.data ?? []} token={t} />
        : <section id="section-exhibit" aria-labelledby="exhibit-heading" className="scroll-mt-14">
          <h2 id="exhibit-heading" className="m-0 eyebrow">Exhibit</h2>
          {exhibitNotice ?? <SkeletonRows className="mt-4" />}
        </section>

  const expertInvitation = done ? <ExpertAnalysisInvitation analysisId={a.id} /> : null
  const sourceComparison = !done ? null : a.kind === 'image' ? <ImageReferenceComparison key={a.id} analysisId={a.id} available={imageComparisonAvailable} />
    : a.kind === 'document' ? <>
        <DocumentRevisionComparison key={`draft-${a.id}`} analysisId={a.id} />
        {/\.(txt|docx)$/i.test(a.filename) && <DocumentArchiveComparison key={`archive-${a.id}`} analysisId={a.id} />}
      </>
      : (a.kind === 'video' || a.kind === 'audio') ? <MediaReferenceComparison key={a.id} analysisId={a.id} kind={a.kind} /> : null
  const body = !legacy.data ? null : !v2 ? (
    <LegacyResult a={legacy.data} token={t} engine={engine} mediaNotice={mediaNotice} afterVerdict={<>{expertInvitation}{sourceComparison}{hr}<FindingsList analysis={a} signals={signals.data ?? []} query={findings} failed={findingsFailed(a.caveats)} /></>} />
  ) : (
    <div className="mt-12 space-y-12" data-testid="v2-result">
      {!evidenceFirst && <KeyFindings analysis={a} query={findings} signals={signals.data ?? []} exhibits={exhibits.data ?? []} token={t} plain={view === 'plain'} />}
      {a.kind === 'image' && <ImageEvidenceCards analysisId={a.id} token={t} mediaReady={tokenReady} />}
      {a.kind === 'image' && <ThumbnailRegionEvidence exhibits={exhibits.data ?? []} token={t} mediaReady={tokenReady} />}
      {a.kind === 'image' && <WholeImageReview exhibits={exhibits.data ?? []} token={t} mediaReady={tokenReady} />}
      {sourceComparison}
      {(a.kind === 'video' || a.kind === 'audio' || a.kind === 'document') && <MultimediaEvidenceCards analysisId={a.id} kind={a.kind} token={t} mediaReady={tokenReady} />}
      {evidenceFirst && <KeyFindings analysis={a} query={findings} signals={signals.data ?? []} exhibits={exhibits.data ?? []} token={t} plain />}
      {findings.isFetched
        ? <AnswerModule analysis={a} findings={findings.data ?? []} signals={signals.data ?? []}
          onExpert={view === 'plain' ? () => chooseView('expert') : undefined} imageComparisonAvailable={imageComparisonAvailable}
          extra={a.kind === 'document' && r?.stats ? <DocumentShare stats={r.stats} estimate={a.decision && a.decision.code !== 'not_assessable' ? estimateOf(a.decision) : null} /> : undefined} />
        : <section id="section-summary" className="scroll-mt-14 border-b border-border pb-8"><SkeletonRows /></section>}
      {expertInvitation}
      {view === 'plain' && (
        <section data-testid="plain-more" aria-labelledby="plain-more-heading" className="border-t border-border pt-8">
          <h2 id="plain-more-heading" className="m-0 eyebrow">Full analysis record</h2>
          <p className="mt-3 mb-0 text-body text-text max-w-[70ch]">
            The source evidence above is available in this view. The full record adds the interactive exhibit, how the estimate was reached, every check with its reference, provenance and validation.
          </p>
          <Button variant="ghost" className="mt-4" onClick={() => chooseView('expert')}>Open full analysis record</Button>
        </section>
      )}
      {mediaNotice}
      {expert && (
        <>
          {exhibit}
          {hr}
          <HowReached analysis={a} signals={signals.data ?? []} findings={findings.data ?? []} />
          {hr}
          <section id="section-ledger" data-testid="evidence" aria-labelledby="evidence-heading" className="scroll-mt-14">
            <h2 id="evidence-heading" className="m-0 eyebrow">Evidence</h2>
            {(signals.isPending || runs.isPending) && <SkeletonRows className="mt-4" widths={['80%', '76%', '72%', '68%', '64%', '60%']} />}
            {signals.error && <ErrorState className="mt-4" onRetry={() => void signals.refetch()}>{signals.error.message}</ErrorState>}
            {runs.error && <ErrorState className="mt-4" onRetry={() => void runs.refetch()}>{runs.error.message}</ErrorState>}
            {signals.isSuccess && runs.isSuccess && (
              <>
                <div className="mt-4"><CoveragePanel analysis={a} runs={runs.data} findings={findings.data ?? []} /></div>
                <div className="mt-10"><SignalsLedger analysis={a} signals={signals.data} runs={runs.data} exhibits={exhibits.data ?? []} findings={findings.data ?? []} token={t} /></div>
              </>
            )}
          </section>
          {hr}
          <ProvenancePanel analysis={a} exhibits={exhibits.data ?? []} runs={runs.data ?? []} />
        </>
      )}
      {expert && (
        <>
          {hr}
          {/* PRD UX-01 order: the Methods & validation section precedes the L4 appendix. */}
          <MethodsPanel analysis={a} cards={cards.data} cardsPending={cards.isPending} cardsError={cards.error} onRetryCards={() => void cards.refetch()} />
          {hr}
          <TechnicalAppendix analysis={a} runs={runs.data ?? []} token={t} legacyMetadata={r?.metadata} engine={engine} cards={cards.data} />
          {a.supersedes_id && <>{hr}<RevisionDeltas analysis={a} /></>}
        </>
      )}
    </div>
  )

  // PRD UX-01: a two-button switch at the end of the section row; the pressed button carries `surface-3`, not the accent.
  const viewToggle = v2 && view && (
            <div role="group" aria-label="Result view" data-testid="view-toggle" data-view={view} className="ml-auto flex shrink-0 items-center border border-border rounded-sm">
              {(['plain', 'expert'] as const).map((v) => (
                <button key={v} type="button" aria-pressed={view === v} onClick={() => { if (view !== v) chooseView(v) }}
                  className={cx('h-7 px-2.5 text-small transition-colors duration-fast first:border-r first:border-border', view === v ? 'bg-surface-3 text-text' : 'text-text-muted hover:text-text')}>
                  {v === 'plain' ? 'Plain' : 'Detailed'}
                </button>
              ))}
            </div>
          )

  return (
    <article>
      <header>
        <div className="flex items-center justify-between gap-4 flex-wrap">
          {/* N-19a (design review DS-12): an exhibit of a case leads back to its case */}
          {a.case_reference && a.case_id
            ? <Link to={`/cases/${a.case_id}`} data-testid="back-link" className="text-small text-text-muted hover:text-text inline-flex items-center min-h-8">‹ <span className="font-mono text-mono ml-1">{a.case_reference}</span></Link>
            : <Link to="/analyses" data-testid="back-link" className="text-small text-text-muted hover:text-text inline-flex items-center min-h-8">‹ Analyses</Link>}
          <div className="flex items-center gap-2">
            {done && toggleInHeader && viewToggle}
            {done && (
              <>
                {/* Q-4c: re-run with the pinned manifest (CC-03); the dialog explains a refusal. */}
                {v2 && examines && !superseded && (
                  <span title={rerunWhy} className="inline-flex">
                    <Button variant="ghost" onClick={() => setRerunning(true)} disabled={rerunBlocked} aria-describedby={rerunWhy ? 'rerun-why' : undefined}>Re-run…</Button>
                    {rerunWhy && <span id="rerun-why" className="sr-only">{rerunWhy}</span>}
                  </span>
                )}
                {/* N-10b: a dialog whose request the server refused stays open with the refusal, although the page now knows
                    that the revision cannot be re-run and has switched the button off */}
                {v2 && examines && <RerunDialog open={rerunning && (!rerunBlocked || rerunRefused)} analysis={a} me={me.data}
                  onRefused={() => setRerunRefused(true)} onClose={() => { setRerunning(false); setRerunRefused(false) }} />}
                {/* Q-4a: a v2 analysis exports through the dialog (RP-03); v1 results keep the legacy ReportLab link. */}
                {superseded ? null : v2
                  ? <Button variant="ghost" onClick={() => setExporting(true)} data-tour="report">Export…</Button>
                  : tokenReady ? <ButtonLink href={api.reportUrl(a.id, t)} variant="ghost" target="_blank" rel="noreferrer" data-tour="report">PDF report</ButtonLink>
                    : <Button variant="ghost" disabled title="Secure media links are unavailable" data-tour="report">PDF report</Button>}
                {superseded ? null : tokenReady ? <ButtonLink href={api.exportUrl(a.id, t)} variant="ghost" target="_blank" rel="noreferrer">JSON</ButtonLink>
                  : <Button variant="ghost" disabled title="Secure media links are unavailable">JSON</Button>}
                {v2 && !superseded && <ExportDialog open={exporting} analysis={a} me={me.data} selfService onClose={() => setExporting(false)} />}
              </>
            )}
            <Menu align="right" ariaLabel="More actions" trigger={<MoreHorizontal size={16} strokeWidth={1.5} aria-hidden />} triggerClassName="h-8 w-8 justify-center"
              items={[{ id: 'copy', label: 'Copy analysis ID' },
                ...(deletes ? [{ id: 'delete', label: 'Delete analysis', danger: true }] : [])]}
              onSelect={(k) => {
                if (k === 'copy') navigator.clipboard?.writeText(a.id).catch(() => undefined)
                if (k === 'delete') setConfirming(true)
              }} />
          </div>
        </div>
        {confirming && (
          <p className="mt-3 flex items-center gap-3 text-small text-text bg-surface-2 border border-border rounded-sm px-3 h-10" role="alert">
            Delete <span className="font-medium">{a.filename}</span> and its files? This cannot be undone.
            <Button variant="danger" className="h-7 px-2.5 ml-1" onClick={() => del.mutate(a.id)} disabled={del.isPending}>{del.isPending ? 'Deleting…' : 'Delete'}</Button>
            <Button variant="text" onClick={() => setConfirming(false)}>Keep</Button>
            {del.error && <span className="text-danger">{(del.error as Error).message}</span>}
          </p>
        )}
        <Eyebrow className="mt-6" data-testid="analysis-eyebrow">
          {a.case_reference && <><span className="font-mono normal-case tracking-normal">{a.case_reference}</span>{a.exhibit_label ? ` · Ex. ${a.exhibit_label}` : ''} · </>}
          {(a.case_reference || (a.revision ?? 1) > 1) && <>revision {a.revision} · </>}
          {a.kind} · Analysis <span className="font-mono normal-case tracking-normal">{a.id.slice(0, 8)}</span>
        </Eyebrow>
        <h1 className="display text-display font-normal text-text mt-1 break-words">{a.filename}</h1>
        <div className="mt-3 flex items-center gap-x-5 gap-y-2 flex-wrap text-small text-text-muted num">
          <span className="inline-flex items-center gap-2">sha256 <CodeChip copy={a.sha256} title="Copy full sha256">{shortHash(a.sha256)}</CodeChip></span>
          <span>{fmtBytes(a.size_bytes)}{dims ? ` · ${dims}` : ''}</span>
          <span>submitted {fmtDateLong(a.created_at)}</span>
          {elapsedRun != null && <span>{fmtSeconds(elapsedRun)}</span>}
        </div>
        {done && <AutomatedResultStatus analysis={a} />}
        <CustodianHashNotice analysis={a} />
      </header>
      {done && (
        // UX §4.2: a 40 px sticky row on `surface` at 96 % with a hairline; the current section in `text` with a 2 px underline.
        // L-7: opaque, so the page does not read through it; between 640 and 1199 px the view switch sits in the header row and the
        // links get the whole width (at 900 px "Review" was clipped behind the switch).
        <nav aria-label="Sections" data-testid="section-index" className="sticky top-0 z-20 mt-8 flex min-w-0 items-center gap-3 h-10 border-b border-border bg-surface">
          <div ref={linksRef} data-testid="section-links" className="section-links flex min-w-0 flex-1 items-center gap-6 max-[1199px]:gap-4 overflow-x-auto">
            {/* N-19b (design review DS-11): the mark is on the section the reader is in, which follows scrolling; the address
                changes only with a click */}
            {sections.map((s) => {
              const isActive = (s.path || 'summary') === currentSection
              return (
                <Link key={s.label} to={s.path ? `/analyses/${a.id}/${s.path}` : `/analyses/${a.id}`} data-section={s.path || 'summary'}
                  aria-current={isActive ? 'location' : undefined}
                  className={cx('relative inline-flex items-center h-10 shrink-0 text-small transition-colors duration-fast', isActive ? 'text-text' : 'text-text-muted hover:text-text')}>
                  {s.label}
                  {isActive && <span aria-hidden className="absolute left-0 right-0 bottom-0 h-0.5 bg-text" />}
                </Link>
              )
            })}
          </div>
          {!toggleInHeader && viewToggle}
        </nav>
      )}

      {!done && rail}
      {!done && <CancelAnalysis key={a.id} analysis={a} role={me.data?.role} caseRole={caseQ.data?.my_role} />}
      {a.status === 'cancelled' && (
        <section data-testid="cancelled-analysis" role="status" className="mt-8 border-y border-border py-6">
          <Eyebrow>Analysis cancelled</Eyebrow>
          {a.cancellation && (
            // N-19a (design review DS-12): who cancelled the run, when, and the note
            <p data-testid="cancellation" className="mt-3 mb-0 text-small text-text max-w-[72ch] whitespace-pre-line [overflow-wrap:anywhere]">
              Cancelled by {a.cancellation.by}, <span className="num">{fmtDateLong(a.cancellation.at)}</span>.
              {a.cancellation.note ? <> <span className="text-text-muted">Note:</span> {a.cancellation.note}</> : ' No note was given.'}
            </p>
          )}
          <p className="mt-3 mb-0 text-small text-text-muted max-w-[72ch]">
            The source and completed checks are retained.
            {retryable ? '' : examines && !rerunBlocked
              ? ` Run again analyses the retained source as revision ${a.revision + 1} of this file, with the checks of this instance as they are now. This cancelled run stays in the record.`
              : ''}
          </p>
          {retryable ? retryRevision : examines && !rerunBlocked && (
            <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2" data-testid="run-again">
              <Button variant="ghost" onClick={() => again.mutate()} disabled={again.isPending}>{again.isPending ? 'Queuing…' : 'Run again'}</Button>
              {again.error && (
                <span role="alert" className="text-small text-danger">
                  {again.error instanceof ApiError ? <><span className="font-medium">{again.error.title}.</span> {again.error.message}</> : again.error.message}
                </span>
              )}
            </div>
          )}
        </section>
      )}

      {a.status === 'failed' && (
        <section className="mt-12 border-t border-border-strong border-b border-b-border py-8" role="alert">
          <Eyebrow className="text-danger">Analysis failed</Eyebrow>
          <pre className="mt-3 bg-surface-2 border border-border rounded-sm p-3 font-mono text-mono text-text whitespace-pre-wrap break-words max-w-[80ch]">{a.error ?? 'Unknown error'}</pre>
          {retryable ? retryRevision : <div className="mt-5"><Button variant="primary" onClick={() => nav('/upload')}>Retry with a new upload</Button></div>}
        </section>
      )}

      {done && legacy.error && <ErrorState className="mt-12" onRetry={() => void legacy.refetch()}>{(legacy.error as Error).message}</ErrorState>}
      {done && legacy.isPending && <SkeletonRows className="mt-12" />}
      {done && body}
      {done && <div className="mt-12">{rail}</div>}

      <Outlet context={{ ready } satisfies WorkspaceContext} />
    </article>
  )
}
