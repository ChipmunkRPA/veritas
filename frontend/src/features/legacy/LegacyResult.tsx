import { lazy, Suspense, useMemo, useState, type ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'
import { api, hasApiKey, type Analysis } from '../../lib/api'
import { fmtSeconds } from '../../lib/format'
import { cx } from '../../lib/cx'
import VerdictModule from './VerdictModule'
import EvidenceTable from './EvidenceTable'
import ManuscriptView from '../document/ManuscriptView'
import ImageExhibit from './ImageExhibit'
import { PlateGrid, captionFor, isFigure, titleFor, type PlateItem } from './Plates'

// recharts is only needed for video; code-split it.
const VideoExhibit = lazy(() => import('./VideoExhibit'))

function Footnote({ label, children }: { label: string; children: ReactNode }) {
  const [o, setO] = useState(false)
  return (
    <div>
      <button type="button" aria-expanded={o} onClick={() => setO(!o)} className="inline-flex items-center gap-1 text-small text-text-muted hover:text-text min-h-8 rounded-sm">
        {label} <ChevronRight size={14} strokeWidth={1.5} aria-hidden className={cx('transition-transform duration-base', o && 'rotate-90')} />
      </button>
      <div className="disclosure" data-open={o}><div>{children}</div></div>
    </div>
  )
}

/** The v1 result body (ARCHITECTURE §7.1 `legacy/`, E-8: kept for v1 results only, `result_schema_version = 1`): verdict module,
 *  exhibit, signal table, plates and footnotes, read from the v1 route. v2 analyses never render through here; the analysis
 *  workspace composes their layers itself. Retirement plan: `docs/v2/adr/0004-legacy-result-views.md`. */
export default function LegacyResult({ a, token, engine, afterVerdict, mediaNotice }: { a: Analysis; token: string; engine: string | null; afterVerdict?: ReactNode; mediaNotice?: ReactNode }) {
  const r = a.result ?? null
  // Figures are the artifacts that are not already rendered by the exhibit (poster, per-frame key frames) and, for
  // documents, not the sentence-score bar chart (the manuscript supersedes it; the PDF report keeps it).
  const figureNames = useMemo(() => (r?.artifacts ?? []).filter((n) => isFigure(n, a.kind)), [r?.artifacts, a.kind])
  const figIndex = useMemo(() => Object.fromEntries(figureNames.map((n, i) => [n, i + 1])), [figureNames])
  const plates: PlateItem[] = useMemo(() => figureNames.map((n, i) => ({ name: n, src: api.artifactUrl(a.id, n, token), title: titleFor(n), caption: captionFor(n), n: i + 1 })), [figureNames, a.id, token])
  if (!r) return null
  // With an API key configured, <img>/<video> requests cannot carry a header: wait for the signed token before
  // rendering media, otherwise the first paint fires unauthenticated requests (401) that only succeed on re-render.
  const tokenReady = !hasApiKey || !!token

  return (
    <div className="mt-12 space-y-12" data-testid="legacy-result">
      <section id="section-summary" className="scroll-mt-14">
        <VerdictModule v={r.verdict} signals={r.signals} stats={a.kind === 'document' ? r.stats : undefined} kind={a.kind} />
      </section>

      {afterVerdict}

      {!tokenReady && (mediaNotice ?? <p role="status" className="text-small text-text-muted py-6">Preparing secure media links…</p>)}
      <>
        {tokenReady && a.kind === 'image' && <ImageExhibit a={a} token={token} />}
        {tokenReady && a.kind === 'video' && (
          <Suspense fallback={<div className="mt-3 rounded-md bg-stage border border-stage-border h-[420px]" aria-busy="true" />}>
            <VideoExhibit a={a} token={token} figIndex={figIndex} />
          </Suspense>
        )}
        {a.kind === 'document' && r.sentences && r.text != null && r.paragraphs && <ManuscriptView sentences={r.sentences} text={r.text} paragraphs={r.paragraphs} />}

        <hr className="border-0 border-t border-border" />
        <EvidenceTable mediaReady={tokenReady} signals={r.signals} artifactUrl={(n) => api.artifactUrl(a.id, n, token)} figIndex={figIndex} />

        {tokenReady && plates.length > 0 && (<><hr className="border-0 border-t border-border" /><PlateGrid items={plates} /></>)}
      </>

      <hr className="border-0 border-t border-border" />
      <footer className="flex flex-wrap items-start gap-x-8 gap-y-2">
        <Footnote label="Timings">
          <dl className="mt-2 m-0 grid grid-cols-[max-content_1fr] gap-x-6 gap-y-0.5 text-small">
            {Object.entries(r.timings ?? {}).map(([k, v]) => (<div key={k} className="contents"><dt className="font-mono text-mono text-text-muted">{k}</dt><dd className="m-0 font-mono text-mono text-text num">{typeof v === 'number' ? fmtSeconds(v, 2) : String(v)}</dd></div>))}
          </dl>
        </Footnote>
        {r.metadata && (
          <Footnote label="Metadata & provenance">
            <pre className="mt-2 max-h-72 overflow-auto bg-surface-2 border border-border rounded-sm p-3 font-mono text-caption leading-[18px] text-text max-w-[80ch]">{JSON.stringify(r.metadata, null, 1)}</pre>
          </Footnote>
        )}
        <span className="text-small text-text-muted min-h-8 inline-flex items-center ml-auto">{engine ? `Engine: ${engine}` : ''}</span>
      </footer>
    </div>
  )
}
