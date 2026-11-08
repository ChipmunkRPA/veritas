import { useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { useExhibitPayloads, type AnalysisDetail, type DetectorRun, type ExhibitMeta, type TableExhibit } from '../../lib/api/queries'
import { cx } from '../../lib/cx'
import { Button, Disclosure, ErrorState, SeverityMark, SkeletonRows } from '../../components/ui'
import { statusWord } from '../signals/ledger'
import { ofTheServer, ON_THIS_SERVER } from './answer'
import {
  TITLES, c2paState, c2paWords, chainNodes, codecFacts, declaresAi, fields, findTable, severityCounts, tableIds, treeGroups,
  type C2paState, type ChainNode, type Fact, type TreeGroup,
} from './provenance'

/** Arrow-key navigation of a `role="tree"` (WAI-ARIA tree pattern, visible items only): Up / Down move, Home / End jump, Right opens
 *  a closed group or moves into it, Left closes an open group or moves to its parent. */
function useTreeKeys(onExpand: (key: string, open: boolean) => void) {
  const root = useRef<HTMLUListElement>(null)
  const onKeyDown = (e: KeyboardEvent<HTMLUListElement>) => {
    const items = Array.from(root.current?.querySelectorAll<HTMLElement>('[role="treeitem"]') ?? []).filter((x) => x.offsetParent !== null || x.getClientRects().length > 0 || !x.closest('[data-closed="true"]'))
    const current = document.activeElement as HTMLElement | null
    const i = current ? items.indexOf(current) : -1
    if (i < 0) return
    const focus = (el: HTMLElement | undefined) => { if (el) { e.preventDefault(); items.forEach((x) => { x.tabIndex = -1 }); el.tabIndex = 0; el.focus() } }
    const expanded = current?.getAttribute('aria-expanded')
    const key = current?.dataset.key ?? ''
    switch (e.key) {
      case 'ArrowDown': focus(items[i + 1]); break
      case 'ArrowUp': focus(items[i - 1]); break
      case 'Home': focus(items[0]); break
      case 'End': focus(items[items.length - 1]); break
      case 'ArrowRight':
        if (expanded === 'false') { e.preventDefault(); onExpand(key, true) } else if (expanded === 'true') focus(items[i + 1])
        break
      case 'ArrowLeft':
        if (expanded === 'true') { e.preventDefault(); onExpand(key, false) } else {
          const parent = current?.parentElement?.closest<HTMLElement>('[role="treeitem"]')
          focus(parent ?? undefined)
        }
        break
      default:
    }
  }
  return { root, onKeyDown }
}

/** Metadata tree (UX §5.8): groups with their fields, severity glyphs with a word at notable and above, rule codes and notes. Groups
 *  with a flagged field start open. */
export function MetadataTree({ groups, caption }: { groups: readonly TreeGroup[]; caption?: string }) {
  const initial = groups.some((g) => g.worst !== 'info') ? groups.filter((g) => g.worst !== 'info').map((g) => g.name) : groups.slice(0, 1).map((g) => g.name)
  const [open, setOpen] = useState<Set<string>>(new Set(initial))
  const { root, onKeyDown } = useTreeKeys((key, o) => setOpen((s) => { const n = new Set(s); if (o) n.add(key); else n.delete(key); return n }))
  const total = groups.reduce((n, g) => n + g.fields.length, 0)
  const allOpen = groups.every((g) => open.has(g.name))
  return (
    <div data-testid="metadata-tree">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h3 className="m-0 eyebrow">File & metadata</h3>
        <Button variant="text" onClick={() => setOpen(allOpen ? new Set(initial) : new Set(groups.map((g) => g.name)))}>{allOpen ? 'Show flagged groups only' : `Show all ${total} fields`}</Button>
      </div>
      {caption && <p className="mt-1 m-0 text-caption text-text-muted max-w-[80ch]">{caption}</p>}
      <ul ref={root} role="tree" aria-label="File and metadata" onKeyDown={onKeyDown} className="mt-2 m-0 p-0 list-none border-t border-border">
        {groups.map((g, gi) => {
          const o = open.has(g.name)
          return (
            <li key={g.name} role="treeitem" aria-level={1} aria-expanded={o} data-key={g.name} tabIndex={gi === 0 ? 0 : -1}
              className="border-b border-border outline-none focus-visible:bg-surface-3">
              <span className="flex items-baseline gap-2 min-h-8 py-1.5 cursor-pointer" onClick={() => setOpen((s) => { const n = new Set(s); if (o) n.delete(g.name); else n.add(g.name); return n })}>
                <span aria-hidden className="w-3 font-mono text-text-muted">{o ? '▾' : '▸'}</span>
                <span className="text-small font-medium text-text">{g.name}</span>
                <span className="text-caption text-text-muted num">{g.fields.length} {g.fields.length === 1 ? 'field' : 'fields'}</span>
                {g.worst !== 'info' && <SeverityMark severity={g.worst} className="ml-1" />}
              </span>
              <ul role="group" data-closed={!o} className={cx('m-0 p-0 list-none pb-2', !o && 'hidden')}>
                {g.fields.map((f, fi) => (
                  <li key={`${f.tag}-${fi}`} role="treeitem" aria-level={2} tabIndex={-1} data-severity={f.severity}
                    className="grid grid-cols-[minmax(0,200px)_minmax(0,1fr)_112px_minmax(0,1fr)] gap-x-4 py-1 pl-5 text-small outline-none focus-visible:bg-surface-3 max-[959px]:grid-cols-[minmax(0,160px)_minmax(0,1fr)_104px] max-[639px]:grid-cols-1">
                    <span className="text-text-muted truncate" title={f.tag}>{f.tag}</span>
                    <span className="font-mono text-mono text-text break-words">{f.value || '—'}</span>
                    <span>{f.severity === 'info'
                      ? <span className="inline-flex items-baseline gap-1.5"><span aria-hidden className="inline-block w-3 text-center text-sev-info">○</span><span className="sr-only">Info</span></span>
                      : <SeverityMark severity={f.severity} />}</span>
                    <span className="text-text-muted max-[959px]:col-start-2 max-[959px]:col-span-2 max-[639px]:col-start-1">{f.note}{f.rule && <span className="ml-2 font-mono text-micro">{f.rule}</span>}</span>
                  </li>
                ))}
              </ul>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

/** Content Credentials (UX §5.8, §7.6): neutral styling, validation state in words, the active manifest and its ingredient chain. */
export function C2paChain({ state, checked, credentials, chain }: { state: C2paState; checked: boolean; credentials?: Map<string, string>; chain: readonly ChainNode[] }) {
  const words = c2paWords(state, checked)
  // The active credential table carries the backend source-type tiers; prefer it to the unqualified chain rendering.
  // Only an active (depth 0) legacy chain row may supply a missing field. Ingredient declarations never classify this file.
  const sourceTypes = credentials?.get('digital source types') ?? chain.find((n) => n.depth === 0)?.sourceTypes ?? ''
  const ai = declaresAi(sourceTypes)
  const documentary = ai && checked && state === 'trusted'
  const selfDeclared = ai && checked && state === 'valid'
  const ingredients = chain.filter((n) => n.depth > 0)
  const { root, onKeyDown } = useTreeKeys(() => undefined)
  return (
    <div data-testid="c2pa" data-c2pa-state={words.state}>
      <h3 className="m-0 eyebrow">Content credentials</h3>
      <p className="mt-2 m-0 text-body text-text max-w-[72ch]">{words.line}</p>
      {credentials && state !== 'absent' && credentials.size > 0 && (
        <div className="mt-3">
          <p className="m-0 flex flex-wrap items-baseline gap-x-2 text-small text-text">
            {documentary && <span aria-hidden>◆</span>}
            <span className="font-medium">Active manifest</span>
            <span>{credentials.get('claim generator') ?? 'claim generator not stated'} · signed by {credentials.get('signer') ?? 'not stated'} · {credentials.get('signed at') ?? 'no timestamp'}</span>
            {ai && <span className="eyebrow">{documentary ? 'Documentary' : selfDeclared ? 'Self-declared' : 'Inspection only'}</span>}
          </p>
          <dl className="mt-2 m-0 grid grid-cols-[136px_minmax(0,1fr)] gap-x-4 gap-y-1 text-small max-[639px]:grid-cols-1">
            {['actions', 'digital source types', 'hash binding', 'issuer', 'legacy trust'].filter((k) => credentials.has(k)).map((k) => (
              <div key={k} className="contents"><dt className="text-text-muted">{k.charAt(0).toUpperCase() + k.slice(1)}</dt><dd className="m-0 text-text break-words">{credentials.get(k)}</dd></div>
            ))}
          </dl>
          {ai && <p className="mt-2 m-0 text-small text-text">
            {documentary
              ? <>The record includes an AI-generation action by {credentials.get('claim generator') ?? 'the signing software'}.</>
              : selfDeclared
                ? 'The record declares AI generation. The signer is unverified, so this counts as self-declared evidence.'
                : 'The record contains an AI declaration for inspection only; it is not relied on without validated credentials.'}
          </p>}
        </div>
      )}
      {ingredients.length > 0 && (
        <div className="mt-3">
          <p className="m-0 text-small font-medium text-text">Ingredients ({ingredients.length})</p>
          <p className="mt-1 m-0 text-caption text-text-muted">Ingredient trust is not established here. These declarations and validation failures are recorded by the active manifest.</p>
          <ul ref={root} role="tree" aria-label="Content Credentials ingredients" onKeyDown={onKeyDown} className="mt-1 m-0 p-0 list-none">
            {ingredients.map((n, i) => (
              <li key={i} role="treeitem" aria-level={n.depth} tabIndex={i === 0 ? 0 : -1} className="flex items-baseline gap-2 py-1 text-small outline-none focus-visible:bg-surface-3" style={{ paddingLeft: (n.depth - 1) * 20 }}>
                <span aria-hidden className="font-mono text-text-muted">{i === ingredients.length - 1 ? '└─' : '├─'}</span>
                <span aria-hidden className="text-text">○</span>
                <span className="text-text break-words">{n.title} · {n.relationship} · signed by {n.signer} · {n.sourceTypes}{n.failures && n.failures !== '0' ? ` · ${n.failures} recorded validation failures` : ''}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export function CodecFacts({ facts }: { facts: readonly Fact[] }) {
  return (
    <div data-testid="codec-facts">
      <h3 className="m-0 eyebrow">Codec facts</h3>
      <dl className="mt-2 m-0 border-t border-border">
        {facts.map((f) => (
          <div key={f.fact} className="grid grid-cols-[minmax(0,240px)_minmax(0,1fr)_minmax(0,1fr)] gap-x-4 py-1.5 border-b border-border text-small max-[959px]:grid-cols-[minmax(0,180px)_minmax(0,1fr)]">
            <dt className="text-text-muted">{f.fact}</dt>
            <dd className="m-0 font-mono text-mono text-text break-words">{f.value || '—'}</dd>
            <dd className="m-0 text-text-muted max-[959px]:col-start-2">{f.reference}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

export function RecordTable({ table }: { table: Pick<TableExhibit, 'columns' | 'rows' | 'caption'> }) {
  return (
    <div className="overflow-x-auto">
      {table.caption && <p className="m-0 mb-2 text-caption text-text-muted max-w-[80ch]">{table.caption}</p>}
      <table className="w-full border-collapse text-small">
        <thead><tr className="border-b border-border-strong">{table.columns.map((c) => <th key={c} scope="col" className="eyebrow text-left font-medium py-1.5 pr-4">{c}</th>)}</tr></thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i} className="border-b border-border">{row.map((c, j) => <td key={j} className="py-1.5 pr-4 align-top text-text break-words">{c == null ? '—' : String(c)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Block({ children }: { children: ReactNode }) {
  return <div className="mt-8 first:mt-4">{children}</div>
}

/** L3 Provenance & metadata (UX §5.8): Content Credentials, the metadata tree with its anomalies, codec facts for video, and the
 *  other provenance records. Each block says when its check did not run instead of rendering empty. */
export default function ProvenancePanel({ analysis, exhibits, runs }: { analysis: AnalysisDetail; exhibits: readonly ExhibitMeta[]; runs: readonly DetectorRun[] }) {
  const ids = tableIds(exhibits)
  const payloads = useExhibitPayloads(analysis.id, ids)
  const tables = payloads.data.filter((x): x is TableExhibit => x.kind === 'table')
  const run = (pred: (id: string) => boolean) => runs.find((r) => pred(r.detector_id))
  const c2paRun = run((id) => id.includes('c2pa'))
  const treeTable = findTable(tables, TITLES.tree)
  const fallbackTables = treeTable ? [] : [findTable(tables, TITLES.exif), findTable(tables, TITLES.jpeg)].filter((t): t is TableExhibit => !!t)
  const groups = treeTable ? treeGroups(treeTable) : fallbackTables.flatMap((t) => treeGroups(t))
  const counts = severityCounts(groups)
  const codec = findTable(tables, TITLES.codec)
  const used = new Set([treeTable, ...fallbackTables, codec, findTable(tables, TITLES.c2pa), findTable(tables, TITLES.chain)].filter(Boolean).map((t) => t!.id))
  const others = tables.filter((t) => !used.has(t.id))
  const notRunLine = (what: string, r: DetectorRun | undefined) =>
    r ? `${what}: ${ofTheServer(r.status, r.reason) ? ON_THIS_SERVER : statusWord(r.status)}${r.reason ? ` · ${r.reason}` : ''}.` : `${what} did not run for this analysis.`
  const metaRun = run((id) => id.startsWith('img.meta') || id === 'vid.meta.container')

  return (
    <section id="section-provenance" data-testid="provenance" aria-labelledby="provenance-heading" className="scroll-mt-14">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h2 id="provenance-heading" className="m-0 eyebrow">Provenance & metadata</h2>
        {groups.length > 0 && (
          <p data-testid="severity-counts" className="m-0 text-small text-text-muted num">
            ○ {counts.info} info · ◐ {counts.notable} notable · ● {counts.anomaly} {counts.anomaly === 1 ? 'anomaly' : 'anomalies'}{counts.documentary ? ` · ◆ ${counts.documentary} documentary` : ''}
          </p>
        )}
      </div>
      {payloads.isPending && ids.length > 0 && <SkeletonRows className="mt-4" />}
      {payloads.error && <ErrorState className="mt-4" onRetry={payloads.refetch}>{payloads.error.message}</ErrorState>}
      {!payloads.isPending && (
        <>
          <Block><C2paChain state={c2paState(analysis, fields(findTable(tables, TITLES.c2pa)))} checked={!!c2paRun && c2paRun.status === 'ok'} credentials={fields(findTable(tables, TITLES.c2pa))} chain={chainNodes(findTable(tables, TITLES.chain))} /></Block>
          <Block>
            {groups.length > 0
              ? <MetadataTree groups={groups} caption={treeTable?.caption} />
              : (
                <div data-testid="metadata-tree" data-state="partial">
                  <h3 className="m-0 eyebrow">File & metadata</h3>
                  <p className="mt-2 m-0 text-small text-text-muted">{notRunLine('The metadata checks', metaRun)} The metadata fields as recorded are in the technical appendix.</p>
                </div>
              )}
          </Block>
          {analysis.kind === 'video' && (
            <Block>
              {codec ? <CodecFacts facts={codecFacts(codec)} /> : (
                <div data-testid="codec-facts" data-state="partial">
                  <h3 className="m-0 eyebrow">Codec facts</h3>
                  <p className="mt-2 m-0 text-small text-text-muted">{notRunLine('The codec check', run((id) => id.startsWith('vid.codec')))}</p>
                </div>
              )}
            </Block>
          )}
          {others.length > 0 && (
            <Block>
              <h3 className="m-0 eyebrow">Other records</h3>
              <div className="mt-2">{others.map((t) => <Disclosure key={t.id} label={t.title} summary={`${t.rows.length} ${t.rows.length === 1 ? 'row' : 'rows'}`}><RecordTable table={t} /></Disclosure>)}</div>
            </Block>
          )}
        </>
      )}
    </section>
  )
}
