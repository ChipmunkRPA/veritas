import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { FilePlus2, FolderPlus, Search } from 'lucide-react'
import { MEDIA_TYPES } from '../../components/MediaTypes'
import CaseOverview from './CaseOverview'
import CaseLibraryRows from './CaseLibraryRows'
import { useAwaitingSecondReview, useCases, useMe } from '../../lib/api/queries'
import { mayStartWork } from '../../lib/api/roles'
import { Button, EmptyState, ErrorState, Menu, Skeleton } from '../../components/ui'
import NewCaseDialog, { inputBase } from './NewCaseDialog'
import { filterCases, type ScopeFilter, type StatusFilter } from './labels'

const STATUS_LABEL: Record<StatusFilter, string> = { all: 'All statuses', open: 'Open', in_review: 'In review', signed_off: 'Signed off', closed: 'Closed' }
const SCOPE_LABEL: Record<ScopeFilter, string> = { all: 'Named and unfiled', named: 'Named cases', unfiled: 'Unfiled only' }
const menuItems = <T extends string>(m: Record<T, string>) => (Object.keys(m) as T[]).map((id) => ({ id, label: m[id] }))

/** Cases dashboard (ARCHITECTURE §7.2, UX §5.12): the cases the caller can see, most recent activity first, with search over case
 *  number, title and matter, a status filter and the named / unfiled scope. */
export default function CasesDashboard() {
  const nav = useNavigate()
  const q = useCases()
  const awaiting = useAwaitingSecondReview()  // N-19a: what waits for a second reviewer
  const starts = mayStartWork(useMe().data)
  const awaitingByCase = useMemo(() => {
    const m = new Map<string, number>()
    for (const a of awaiting.data ?? []) if (a.case_id) m.set(a.case_id, (m.get(a.case_id) ?? 0) + 1)
    return m
  }, [awaiting.data])
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState<StatusFilter>('all')
  const [scope, setScope] = useState<ScopeFilter>('all')
  const [creating, setCreating] = useState(false)
  const cases = [...(q.data ?? [])].sort((x, y) => Date.parse(y.updated_at) - Date.parse(x.updated_at))
  const rows = filterCases(cases, search, status, scope)
  const unfiled = cases.filter((c) => c.implicit).length
  const filtered = search.trim() !== '' || status !== 'all' || scope !== 'all'
  const clear = () => { setSearch(''); setStatus('all'); setScope('all') }

  return (
    <div>
      {/* L-7 (design P2): a working page head, not a hero: the title, what the page holds, the two actions and the four ways in. */}
      <header className="flex items-start justify-between gap-x-8 gap-y-4 flex-wrap">
        <div className="min-w-0">
          <h1 id="cases-heading" className="display text-h1 font-normal text-text m-0">Cases</h1>
          <p className="text-small text-text-muted mt-1 mb-0 max-w-[64ch]">Each case keeps its exhibits, the machine findings and the examiners’ review in one record with a custody log.</p>
        </div>
        {/* N-19a ("Actions by role"): a viewer or a reviewer is not offered what starts work */}
        {starts && <div className="flex items-center gap-3 flex-wrap">
          <Button variant="primary" onClick={() => nav('/upload')}><FilePlus2 size={16} strokeWidth={1.5} aria-hidden />New analysis</Button>
          <Button variant="ghost" onClick={() => setCreating(true)}><FolderPlus size={16} strokeWidth={1.5} aria-hidden />New case</Button>
        </div>}
      </header>
      {starts && <nav aria-label="Start an analysis" className="mt-6 flex items-center gap-x-6 gap-y-2 flex-wrap border-y border-border py-2.5 text-small">
        <span className="eyebrow">Analyse</span>
        {MEDIA_TYPES.map(({ kind, label, description, icon: Icon }) => (
          <Link key={kind} to={`/upload?kind=${kind}`} aria-label={`Analyse ${label.toLowerCase()}`} title={description}
            className="inline-flex items-center gap-2 min-h-8 text-text hover:underline underline-offset-[3px]">
            <Icon size={16} strokeWidth={1.5} aria-hidden className="text-text-muted" />{label}
          </Link>
        ))}
      </nav>}
      {q.isSuccess && cases.length > 0 && <CaseOverview cases={cases} awaiting={awaiting.data ?? []} />}
      <NewCaseDialog open={creating} onClose={() => setCreating(false)} onCreated={(c) => { setCreating(false); nav(`/cases/${c.id}`) }} />

      {q.error && <ErrorState className="mt-8" onRetry={() => void q.refetch()}>{q.error.message}</ErrorState>}
      {q.isSuccess && cases.length === 0 && (
        <div className="mt-8 p-6 border border-dashed border-border-strong rounded-md">
          {starts ? <>
            <h2 className="text-h3 font-medium text-text m-0">Your first source starts the record.</h2>
            <p className="text-small text-text-muted mt-2 mb-0 max-w-[64ch]">Start an analysis above, or create a named case to collect several exhibits. A quick analysis automatically opens an unfiled case.</p>
          </> : <p className="text-small text-text-muted m-0 max-w-[64ch]">No case has been shared with this account yet. An owner of a case adds its members.</p>}
        </div>
      )}

      {(q.isPending || cases.length > 0) && (
        <>
          <div className="mt-8 mb-4 flex items-baseline justify-between gap-3 flex-wrap">
            <h2 className="text-h3 font-medium text-text m-0">Case library</h2>
            <p className="text-caption text-text-muted m-0 num">{q.isPending ? <Skeleton w={180} /> : `${cases.length - unfiled} named · ${unfiled} unfiled`}</p>
          </div>
          <div className="workspace-toolbar flex flex-wrap items-center gap-x-5 gap-y-2">
            <div className="relative max-w-full grow min-w-0 sm:grow-0">
              <Search size={16} strokeWidth={1.5} aria-hidden className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" />
            <input type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search case number, title or matter" aria-label="Search case number, title or matter"
              data-testid="case-search" className={`${inputBase} min-h-11 pl-9 w-80 max-w-full`} />
            </div>
            <Menu label={status === 'all' ? 'Status' : `Status: ${STATUS_LABEL[status]}`} onSelect={setStatus} items={menuItems(STATUS_LABEL)} />
            <Menu label={scope === 'all' ? 'Scope' : `Scope: ${SCOPE_LABEL[scope]}`} onSelect={setScope} items={menuItems(SCOPE_LABEL)} />
            {filtered && <Button variant="text" onClick={clear}>Clear filters</Button>}
            {q.isSuccess && filtered && <span className="ml-auto text-small text-text-muted num" aria-live="polite">{rows.length} of {cases.length}</span>}
          </div>

          {q.isSuccess && rows.length === 0 ? (
            <EmptyState className="mt-6" action="Clear filters" onAction={clear}>No case matches these filters.</EmptyState>
          ) : (
            <div className="mt-4 overflow-x-auto workspace-table">
              {/* below 640 px each case is one stacked block, so its counts and review state stay on screen (L-7) */}
              {/* N-19b (design review DS-04): a fixed layout, so the title column takes what the three fixed columns leave and
                  the table is as wide as its card at every width: "Last activity" is on screen and nothing scrolls sideways */}
              <table role="table" className="stack-sm w-full table-fixed border-collapse" data-testid="cases">
                <thead role="rowgroup">
                  <tr role="row" className="border-b border-border-strong bg-surface-2">
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pl-4 pr-4">File number · title</th>
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-48 max-[959px]:w-44">Exhibits · analyses</th>
                    <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-3 pr-4 w-52 max-[959px]:w-44">Review</th>
                    <th role="columnheader" scope="col" className="eyebrow text-right font-medium py-3 pr-4 w-44">Last activity</th>
                  </tr>
                </thead>
                <tbody role="rowgroup">
                  {q.isPending && Array.from({ length: 4 }).map((_, i) => (
                    <tr role="row" key={i} className="border-b border-border h-12"><td role="cell" colSpan={4}><Skeleton w={240} /></td></tr>
                  ))}
                  <CaseLibraryRows cases={rows} awaiting={awaitingByCase} />
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
