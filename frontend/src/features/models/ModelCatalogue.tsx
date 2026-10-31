import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Search } from 'lucide-react'
import { useModelCards } from '../../lib/api/queries'
import { cx } from '../../lib/cx'
import { Button, EmptyState, ErrorState, Menu, Skeleton } from '../../components/ui'
import { LICENCE_CLASS_WORD, MODALITY_WORD, admissionWord, catalogueFacts, clientOf, filterCards, installedWord, licenceClasses, loadedWord, type StateFilter } from './catalogue'

const STATE_LABEL: Record<StateFilter, string> = {
  all: 'Any state', enabled: 'Enabled', disabled: 'Disabled', installed: 'Weights installed', not_installed: 'Weights not installed', loaded: 'Loaded now',
}
const menuItems = <T extends string>(m: Record<T, string>) => (Object.keys(m) as T[]).map((id) => ({ id, label: m[id] }))

function Glyph({ on, none = false }: { on: boolean; none?: boolean }) {
  return <span aria-hidden className="inline-block w-3 mr-1.5 text-center font-mono">{none ? '·' : on ? '●' : '⊘'}</span>
}

/** Model catalogue (ARCHITECTURE §7.2 `/models`, VG-01, E-7): every detector of the manifest with its licence classes, the policy
 *  decision, whether its weights are on disk and in memory, and its admission state. Rows open the full card. */
export default function ModelCatalogue() {
  const q = useModelCards()
  const [search, setSearch] = useState('')
  const [modality, setModality] = useState('all')
  const [state, setState] = useState<StateFilter>('all')
  const cards = q.data ?? []
  const rows = filterCards(cards, search, modality, state)
  const modalities = [...new Set(cards.map((c) => c.modality))]
  const modalityLabel: Record<string, string> = { all: 'All modalities', ...Object.fromEntries(modalities.map((m) => [m, MODALITY_WORD[m] ?? m])) }
  const filtered = search.trim() !== '' || modality !== 'all' || state !== 'all'
  const clear = () => { setSearch(''); setModality('all'); setState('all') }

  return (
    <div>
      <h1 className="display text-h1 font-normal text-text">Models</h1>
      <p data-testid="catalogue-facts" className="text-small text-text-muted mt-1 num">{q.isPending ? <Skeleton w={320} /> : catalogueFacts(cards)}</p>
      <p className="mt-3 m-0 text-small text-text max-w-[80ch]">
        Each detector’s model card: what it was trained on, what it is known to get wrong, the weights it pins and what has been measured.
        A detector runs only when the licence policy enables it and its weights are installed.
        Licence classes include required model dependencies.
      </p>

      {q.error && <ErrorState className="mt-8" onRetry={() => void q.refetch()}>{q.error.message}</ErrorState>}
      {q.isSuccess && cards.length === 0 && <EmptyState className="mt-8">The models manifest lists no detectors.</EmptyState>}

      {(q.isPending || cards.length > 0) && (
        <>
          {/* N-19d (design review P2): one list idiom. The toolbar and the table are those of Home and Analyses: a boxed
              toolbar with the search field first, and the table in a card */}
          <div className="workspace-toolbar mt-8 flex flex-wrap items-center gap-x-5 gap-y-2">
            <div className="relative max-w-full grow min-w-0 sm:grow-0">
              <Search size={16} strokeWidth={1.5} aria-hidden className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
              <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="id, name or family" aria-label="Search id, name or family"
                data-testid="model-search" className="h-11 w-80 max-w-full pl-9 pr-2.5 rounded-sm border border-graphic bg-surface text-small placeholder:text-text-muted focus-visible:border-accent" />
            </div>
            <Menu label={modality === 'all' ? 'Modality' : `Modality: ${modalityLabel[modality]}`} onSelect={setModality} items={menuItems(modalityLabel)} />
            <Menu label={state === 'all' ? 'State' : `State: ${STATE_LABEL[state]}`} onSelect={setState} items={menuItems(STATE_LABEL)} />
            {filtered && <Button variant="text" onClick={clear}>Clear filters</Button>}
            {q.isSuccess && filtered && <span className="ml-auto text-small text-text-muted num" aria-live="polite">{rows.length} of {cards.length}</span>}
          </div>

          {q.isSuccess && rows.length === 0 ? (
            <EmptyState className="mt-6" action="Clear filters" onAction={clear}>No detector matches these filters.</EmptyState>
          ) : (
            <div className="relative mt-4 overflow-x-auto workspace-table">
              {/* N-19b (design review DS-13): below 640 px one stacked block for each detector */}
              <table role="table" className="stack-sm w-full border-collapse" data-testid="models">
                <thead role="rowgroup">
                  <tr role="row" className="border-b border-border-strong bg-surface-2">
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pl-4 pr-4">Detector</th>
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-24 max-[959px]:hidden">Modality</th>
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-36">Licence</th>
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-24">Policy</th>
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-32">Weights</th>
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-32">Memory</th>
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-28 max-[959px]:hidden">Admission</th>
                  </tr>
                </thead>
                <tbody role="rowgroup">
                  {q.isPending && Array.from({ length: 6 }).map((_, i) => (
                    <tr role="row" key={i} className="border-b border-border h-12"><td role="cell" colSpan={7} className="pl-4"><Skeleton w={240} /></td></tr>
                  ))}
                  {rows.map((c) => (
                    <tr role="row" key={c.id} data-testid="model-row" data-enabled={c.enabled} className="border-b border-border h-12 max-[639px]:h-auto transition-colors duration-fast hover:bg-surface-3">
                      {/* L-7 (design P2): a name wraps; it is never cut with an ellipsis */}
                      <td role="cell" className="py-2 pl-4 pr-4 min-w-[220px] max-w-[460px] max-[639px]:min-w-0">
                        <Link to={`/models/${encodeURIComponent(c.id)}`} className="block text-body font-medium text-text break-words hover:underline underline-offset-[3px]">{c.name}</Link>
                        <span className="block font-mono text-caption text-text-muted break-all">{c.id} {c.version}</span>
                      </td>
                      <td role="cell" className="pr-4 text-small text-text-muted max-[959px]:hidden">{MODALITY_WORD[c.modality] ?? c.modality}</td>
                      <td role="cell" data-th="licence" className="pr-4 text-small text-text">{licenceClasses(c).map((k) => LICENCE_CLASS_WORD[k] ?? k.replace(/_/g, ' ')).join(', ') || <span className="text-text-muted">model-free</span>}</td>
                      {/* state glyphs as on the coverage panel: ● yes, ⊘ no; the word carries the meaning */}
                      <td role="cell" data-th="policy" className={cx('pr-4 text-small whitespace-nowrap max-[639px]:whitespace-normal', c.enabled ? 'text-text' : 'text-text-muted')} title={(clientOf(c).enablement_reasons ?? []).join('; ') || undefined}><Glyph on={c.enabled} />{c.enabled ? 'enabled' : 'disabled'}</td>
                      <td role="cell" data-th="weights" className={cx('pr-4 text-small whitespace-nowrap max-[639px]:whitespace-normal', c.installed ? 'text-text' : 'text-text-muted')}><Glyph on={c.installed && (c.model_entries ?? []).length > 0} none={(c.model_entries ?? []).length === 0} />{installedWord(c)}</td>
                      <td role="cell" data-th="memory" className={cx('pr-4 text-small whitespace-nowrap max-[639px]:whitespace-normal', c.loaded ? 'text-text' : 'text-text-muted')}><Glyph on={c.loaded === true} none={c.loaded == null} />{loadedWord(c)}</td>
                      <td role="cell" className="pr-4 text-small text-text-muted max-[959px]:hidden">{admissionWord(c).split(' · ')[0]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
