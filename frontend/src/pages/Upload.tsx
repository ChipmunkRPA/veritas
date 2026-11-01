import { useEffect, useRef, useState, type DragEvent } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ArrowRight, File as FileIcon, UploadCloud, X } from 'lucide-react'
import { api, type Kind } from '../lib/api'
import { MAX_BATCH_FILES, droppedFiles, submitBatch, usable, withRelativePaths } from '../lib/api/batches'
import { intakeItem, type AnalysisProfile } from '../lib/api/intake'
import { qk, useCases, useMe, useModelCards, type Case } from '../lib/api/queries'
import { BillingError, billingApi, hasBillingIdentity, openCheckout, useBilling } from '../lib/api/billing'
import { fmtBytes, wordCount } from '../lib/format'
import { cx } from '../lib/cx'
import { Button, Disclosure, Eyebrow, Select, Tabs } from '../components/ui'
import { MediaIcon } from '../components/MediaTypes'
import SourcePreview from '../components/SourcePreview'
import { beginBlocked, intakeKind, PROFILES, TABS, validate } from './intakeOptions'
import IntakeChecks from '../features/intake/IntakeChecks'
import IntakeFields from '../features/cases/IntakeFields'
import NewCaseDialog from '../features/cases/NewCaseDialog'
import { EMPTY_DRAFT, draftProblem, recallDraft, rememberDraft, toIntakeFields, type CustodyDraft } from '../features/cases/intakeDraft'

const MIN_WORDS = 15
const LISTED = 200

/** Intake (UX §5.1; ARCHITECTURE §7.2 `/upload`, `/cases/:caseId/upload`). Without a case, the v1 quick route files the upload into
 *  an implicit case of its own; with a case, `POST /cases/{id}/items?analyze=true` records the acquisition fields and the browser
 *  pre-hash in the intake custody event. Several files, or a folder, make a batch (`POST /api/v2/batches`, PRD API-01): the server
 *  detects each file's kind and files every item into one case. */
export default function Upload() {
  const { caseId: routeCase = '' } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const kind = intakeKind(searchParams.get('kind'))
  const [file, setFile] = useState<File | null>(null)
  const [batch, setBatch] = useState<File[]>([])
  const [text, setText] = useState('')
  const [drag, setDrag] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const [hashing, setHashing] = useState(false)
  const [caseId, setCaseId] = useState(routeCase)
  const [draft, setDraft] = useState<CustodyDraft>(() => (routeCase ? recallDraft(routeCase) : EMPTY_DRAFT))
  const [creating, setCreating] = useState(false)
  const [created, setCreated] = useState<Case | null>(null)
  const [profile, setProfile] = useState<AnalysisProfile | ''>('')
  const inputRef = useRef<HTMLInputElement>(null)
  const attemptKey = useRef(crypto.randomUUID())
  const submitting = useRef(false)
  const billing = useBilling()
  const me = useMe()
  const paid = billing.data?.enabled === true
  const signedIn = hasBillingIdentity(me.data)
  const billingProblem = billing.isPending ? 'Checking analysis pricing…' : billing.isError ? 'Pricing is unavailable. Try again before starting an analysis.'
    : paid && !billing.data?.ready ? 'Checkout is not available yet. Please try again later.'
    : paid && !signedIn ? 'Sign in to keep your payment and analysis private.' : null
  const nav = useNavigate()
  const qc = useQueryClient()
  const cases = useCases()
  const catalogue = useModelCards()
  const tab = TABS.find((t) => t.kind === kind)!
  const words = wordCount(text)
  const isBatch = batch.length > 1
  const batchBytes = batch.reduce((n, f) => n + f.size, 0)
  const usingText = kind === 'document' && !file && !isBatch && text.trim().length > 0
  const problem = !paid && caseId && !isBatch ? draftProblem(draft) : null
  const sourceReady = isBatch ? !paid && batch.length <= MAX_BATCH_FILES : (!!file || (kind === 'document' && words >= MIN_WORDS)) && !problem
  const ready = sourceReady && !billingProblem
  const options = [...(cases.data ?? []), ...(created && !cases.data?.some((c) => c.id === created.id) ? [created] : [])]
    .filter((c) => !c.implicit || c.id === routeCase)
    .sort((x, y) => Date.parse(y.updated_at) - Date.parse(x.updated_at))
  const chosen = options.find((c) => c.id === caseId) ?? null

  const pickCase = (id: string) => { setCaseId(id); setDraft(id ? recallDraft(id) : EMPTY_DRAFT) }

  const up = useMutation({
    mutationFn: async (): Promise<{ path: string; checkout?: boolean }> => {
      if (billingProblem) throw new Error(billingProblem)
      if (paid) {
        if (isBatch) throw new Error('Choose one file per checkout. Each analysis costs $9.99 USD.')
        const blob = usingText ? new File([text], 'pasted-text.txt', { type: 'text/plain' }) : file!
        setProgress(null)
        const result = await billingApi.checkout(kind, blob, profile || 'full', attemptKey.current)
        return { path: result.checkout_url, checkout: true }
      }
      if (isBatch) {
        setProgress(0)
        const r = await submitBatch(batch, caseId || undefined, setProgress, profile || undefined)
        void qc.invalidateQueries({ queryKey: qk.cases() })
        return { path: `/batches/${r.id}` }
      }
      if (caseId) {
        const blob = usingText ? new File([text], 'pasted-text.txt', { type: 'text/plain' }) : file!
        setHashing(true)
        setProgress(null)
        const r = await intakeItem(caseId, kind, blob, blob.name, toIntakeFields(draft), {
          onHashed: () => { setHashing(false); setProgress(0) },
          onProgress: setProgress,
        }, profile || undefined)
        rememberDraft(caseId, draft)
        for (const key of [qk.cases(), qk.case(caseId), qk.caseItems(caseId), qk.caseAnalyses(caseId), qk.custody(caseId)]) void qc.invalidateQueries({ queryKey: key })
        return { path: r.analysis ? `/analyses/${r.analysis.id}` : `/cases/${caseId}` }
      }
      if (usingText) { setProgress(null); return { path: `/analyses/${(await api.text(text, undefined, profile || undefined)).id}` } }
      setProgress(0)
      return { path: `/analyses/${(await api.uploadWithProgress(kind, file!, (f) => setProgress(f), profile || undefined)).id}` }
    },
    onSuccess: ({ path, checkout }) => { if (checkout) openCheckout(path); else nav(path) },
    onError: () => { submitting.current = false },
    onSettled: () => { setProgress(null); setHashing(false) },
  })

  const choose = (f: File | null) => {
    if (submitting.current) return
    setBatch([])
    if (!f) { setFile(null); setError(null); return }
    const err = validate(f, tab)
    setError(err)
    setFile(err ? null : f)
  }
  const chooseMany = (files: File[]) => {
    if (submitting.current) return
    if (paid && files.length > 1) { setError('Choose one file per checkout. Each analysis costs $9.99 USD.'); return }
    const list = usable(files)
    if (list.length === 0) { setError(files.length ? 'Only hidden files were chosen' : null); return }
    if (list.length === 1) { choose(list[0]); return }
    setFile(null)
    setBatch(list)
    setError(list.length > MAX_BATCH_FILES ? `A batch holds at most ${MAX_BATCH_FILES} files; ${list.length} were chosen` : null)
  }
  const pickFolder = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.multiple = true
    input.setAttribute('webkitdirectory', '')
    input.onchange = () => chooseMany(withRelativePaths(Array.from(input.files ?? [])))
    input.click()
  }
  const onTab = (k: Kind) => { if (submitting.current) return; setSearchParams((params) => { params.set('kind', k); return params }, { replace: true }) }
  useEffect(() => { setFile(null); setBatch([]); setError(null); up.reset() }, [kind]) // clear source when its examination type changes
  useEffect(() => { attemptKey.current = crypto.randomUUID(); up.reset() }, [file, text, profile, kind]) // retry unchanged input with the same payment key
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    void droppedFiles(e.dataTransfer).then(chooseMany, () => setError('The dropped files could not be read'))
  }

  const pctLabel = progress == null ? null : `${Math.round(progress * 100)} %`  // N-19d: one percent format ("57%" against "70 %")
  const busy = up.isPending
  const idleLabel = paid ? `Continue to checkout · ${billing.data?.price_display}` : isBatch ? `Submit batch (${batch.length} files)` : 'Begin analysis'
  // N-19a: while the button is switched off, the line beside it says what is missing
  const missing = busy ? null : billingProblem ?? (paid && isBatch ? 'Choose one file per checkout.' : null) ?? beginBlocked({ kind, hasFile: !!file, words, minWords: MIN_WORDS, batch: batch.length, maxBatch: MAX_BATCH_FILES, problem })

  return (
    <div>
      {routeCase && (
        <Link to={`/cases/${routeCase}`} className="text-small text-text-muted hover:text-text inline-flex items-center min-h-8 mb-6">
          ‹ {chosen ? (chosen.implicit ? 'Unfiled case' : chosen.reference) : 'Case'}
        </Link>
      )}
      <h1 className="display text-h1 font-normal text-text m-0">New analysis</h1>
      <p className="text-small text-text-muted mt-1 mb-8 max-w-[62ch]">{paid ? 'Add a file or text, pay securely with Stripe, then your analysis begins.' : 'Add a file or text, choose the case and the depth of analysis, then read the findings with their limits.'}</p>

      {paid && <section aria-label="Analysis price" className="mb-8 p-5 rounded-md border border-border bg-surface-2">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1"><span className="text-h2 font-semibold text-text">{billing.data?.price_display}</span><span className="text-small text-text-muted">USD per analysis · one payment · no subscription</span></div>
        <p className="text-small font-medium text-text mt-3 mb-0">Payment is required before analysis. Your analysis starts only after Stripe confirms your payment.</p>
        <p className="text-small text-text mt-2 mb-0">Includes the automated analysis and available supporting evidence, PDF report and JSON export. Results may be inconclusive; payment does not guarantee an AI or original classification.</p>
        {billing.data?.mode === 'test' && <p role="status" className="text-caption text-accent mt-2 mb-0">Test mode — use Stripe test payment details. No real payment is collected.</p>}
        {!signedIn && <Link to={`/sign-in?next=${encodeURIComponent(`/upload?kind=${kind}`)}`} className="inline-flex mt-3 text-small text-accent underline underline-offset-4">Sign in to continue</Link>}
      </section>}
      {billing.isError && <p className="text-small text-danger mb-6" role="alert">Pricing could not be loaded. <Button variant="text" onClick={() => void billing.refetch()}>Try again</Button></p>}

      <div className="intake-layout">
      <div className="intake-form">
      <div className="flex items-center gap-3 mb-3"><span className="font-mono text-caption text-accent num">01</span><h2 className="text-h3 text-text font-medium m-0">Add your source</h2></div>
      <Tabs className="intake-tabs" label="Examination type" value={kind} onChange={onTab}
        items={TABS.map((t) => ({ id: t.kind, label: t.label, icon: <MediaIcon kind={t.kind} size={17} /> }))} />

      <div role="tabpanel" id={`panel-${kind}`} aria-labelledby={`tab-${kind}`} className="mt-6">
        {isBatch ? (
          <div data-testid="batch-list" className="rounded-md bg-surface-2 border border-border">
            <div className="h-12 flex items-center gap-3 px-4 border-b border-border">
              <span className="text-body font-medium text-text num">{batch.length} files</span>
              <span className="font-mono text-mono text-text-muted num">{fmtBytes(batchBytes)}</span>
              <Button variant="text" className="ml-auto" onClick={() => { setBatch([]); setError(null) }}>Clear</Button>
            </div>
            <ul className="max-h-64 overflow-y-auto m-0 p-0 list-none">
              {batch.slice(0, LISTED).map((f, i) => (
                <li key={`${i}-${f.name}`} className="h-9 flex items-center gap-3 px-4 border-b border-border last:border-b-0">
                  <span className="text-small text-text truncate min-w-0">{f.name}</span>
                  <span className="ml-auto font-mono text-mono text-text-muted num shrink-0">{fmtBytes(f.size)}</span>
                </li>
              ))}
              {batch.length > LISTED && <li className="h-9 flex items-center px-4 text-caption text-text-muted num">and {batch.length - LISTED} more</li>}
            </ul>
          </div>
        ) : !file ? (
          <div data-testid="drop-zone" onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={onDrop}
            className={cx('intake-dropzone rounded-md px-6 py-8 max-[639px]:px-4 border text-center transition-colors duration-base',
              error ? 'border-solid border-danger' : drag ? 'border-solid border-accent bg-surface-3' : 'border-dashed border-border-strong')}>
            <span className="intake-dropzone-icon mb-3" aria-hidden><UploadCloud size={26} strokeWidth={1.5} /></span>
            <p className="text-body font-medium text-text m-0">
              Drop {tab.noun} here, or{' '}
              <label className="text-accent underline underline-offset-[3px] decoration-1 hover:text-accent-hover hover:decoration-2 cursor-pointer focus-within:shadow-focus rounded-sm">
                browse
                <input ref={inputRef} type="file" multiple={!paid} disabled={busy} className="sr-only" accept={tab.accept} onChange={(e) => { chooseMany(Array.from(e.target.files ?? [])); e.target.value = '' }} />
              </label>
            </p>
            <p className="text-caption text-text-muted mt-1">
              {tab.formats}{paid ? ' · one file per checkout' : <> · several files or a{' '}<Button variant="text" className="text-caption" onClick={pickFolder}>folder</Button>{' '}make a batch</>}
            </p>
          </div>
        ) : (
          <div className="h-14 flex items-center gap-3 px-4 rounded-md bg-surface-2 border border-border">
            <FileIcon size={16} strokeWidth={1.5} className="text-text-muted shrink-0" aria-hidden />
            <span className="text-body font-medium text-text truncate min-w-0">{file.name}</span>
            <span className="font-mono text-mono text-text-muted num shrink-0">{fmtBytes(file.size)}</span>
            <label className="ml-auto text-accent underline underline-offset-[3px] decoration-1 hover:decoration-2 cursor-pointer text-small font-medium shrink-0">
              Replace
              <input type="file" className="sr-only" accept={tab.accept} onChange={(e) => { choose(e.target.files?.[0] ?? null); e.target.value = '' }} />
            </label>
            <Button variant="icon" aria-label="Remove file" onClick={() => choose(null)}><X size={16} strokeWidth={1.5} aria-hidden /></Button>
          </div>
        )}
        {file && <SourcePreview file={file} kind={kind} />}
        {error && <p className="mt-2 text-caption text-danger" role="alert">{error}</p>}
        {isBatch && (
          <p className="mt-2 mb-0 text-caption text-text-muted max-w-[60ch]">
            Each file&apos;s kind is detected from its contents, and every file is analysed in one case. Files that cannot be taken in are listed with the reason.
          </p>
        )}

        {kind === 'document' && !file && !isBatch && (
          <div className="mt-8">
            <div className="flex items-baseline justify-between">
              <Eyebrow as="span">Or paste text</Eyebrow>
              <span className={cx('font-mono text-mono num', words > 0 && words < MIN_WORDS ? 'text-danger' : 'text-text-muted')} aria-live="polite">
                {words} {words === 1 ? 'word' : 'words'}{words > 0 && words < MIN_WORDS ? <span className="text-caption font-sans"> · minimum {MIN_WORDS} words</span> : null}
              </span>
            </div>
            <textarea disabled={busy} value={text} onChange={(e) => setText(e.target.value)} aria-label="Text to analyse" placeholder="Paste the text to analyse"
              className="mt-2 w-full min-h-60 p-4 rounded-sm bg-surface border border-graphic display text-reader text-text placeholder:text-text-muted resize-y focus-visible:border-accent" />
          </div>
        )}
      </div>

      <div className="flex items-center gap-3 mt-8 mb-3 pt-6 border-t border-border"><span className="font-mono text-caption text-accent num">02</span><h2 className="text-h3 text-text font-medium m-0">Set up the examination</h2></div>
      {paid ? <p className="text-small text-text-muted mt-2">Your upload is prepared for checkout. One file or pasted text is included in each payment; uploading alone does not start an analysis.</p> : <Disclosure className="mt-3" testId="case-custody" label="Case & custody" defaultOpen={!!routeCase}
        summary={chosen ? (chosen.implicit ? 'unfiled case' : chosen.reference) : isBatch ? 'new case for the batch' : 'quick analysis'}>
        <div className="grid grid-cols-[140px_minmax(0,1fr)] gap-x-6 items-start max-[639px]:grid-cols-1 max-[639px]:gap-y-1">
          <label htmlFor="intake-case" className="text-small text-text-muted pt-2 max-[639px]:pt-0">Case</label>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <Select id="intake-case" value={caseId} onChange={(e) => pickCase(e.target.value)} className="w-[420px]">
              <option value="">{isBatch ? 'No case: the batch opens a case of its own' : 'No case: a quick analysis opens an unfiled case'}</option>
              {options.map((c) => <option key={c.id} value={c.id}>{c.implicit ? 'Unfiled' : c.reference} · {c.title}</option>)}
            </Select>
            <Button variant="text" onClick={() => setCreating(true)}>+ New case</Button>
          </div>
        </div>
        {isBatch
          ? <p className="mt-2 mb-0 text-caption text-text-muted max-w-[60ch]">Batch items are recorded with the batch as their acquisition note; add the custody details per item in the case afterwards.</p>
          : caseId
            ? <div className="mt-4"><IntakeFields draft={draft} onChange={setDraft} /></div>
            : <p className="mt-2 mb-0 text-caption text-text-muted max-w-[60ch]">Choose a case to record who supplied the file, how it was acquired and when it was received.</p>}
        {!isBatch && <p className="mt-4 mb-0 text-caption text-text-muted max-w-[60ch]">The file is hashed on this device before upload (SHA-256). The server refuses the upload if its own hash differs.</p>}
      </Disclosure>}
      <NewCaseDialog open={creating} onClose={() => setCreating(false)} onCreated={(c) => { setCreated(c); setCreating(false); pickCase(c.id) }} />

      <div className="mt-6 grid grid-cols-[140px_minmax(0,1fr)] gap-x-6 items-start max-[639px]:grid-cols-1 max-[639px]:gap-y-1" data-testid="intake-profile">
        <label htmlFor="intake-profile" className="text-small text-text-muted pt-2 max-[639px]:pt-0">Analysis profile</label>
        <div>
          <Select id="intake-profile" disabled={busy} value={profile} onChange={(e) => setProfile(e.target.value as AnalysisProfile | '')} className="w-56">
            {PROFILES.map((p) => <option key={p.id || 'default'} value={p.id}>{paid && !p.id ? 'Full (recommended)' : p.label}</option>)}
          </Select>
          <p className="mt-1 mb-0 text-caption text-text-muted max-w-[60ch]">{PROFILES.find((p) => p.id === (paid && !profile ? 'full' : profile))!.detail}{isBatch ? ' Applies to every file of the batch.' : ''}</p>
        </div>
      </div>

      <div className="mt-8 flex flex-wrap items-center gap-x-5 gap-y-3">
        <Button variant="primary" aria-disabled={!ready || busy} aria-busy={busy || undefined} disabled={!ready || busy} aria-describedby={missing ? 'begin-missing' : undefined}
          onClick={() => { if (!submitting.current && !busy && ready) { submitting.current = true; up.mutate() } }} className="relative overflow-hidden min-w-36">
          {busy ? (paid ? 'Preparing secure checkout…' : hashing ? 'Hashing…' : pctLabel ? `Uploading… ${pctLabel}` : 'Submitting…') : <>{idleLabel}<ArrowRight size={16} strokeWidth={1.5} aria-hidden /></>}
          {busy && (
            <span aria-hidden className="absolute left-0 bottom-0 h-0.5 w-full bg-surface/30 overflow-hidden">
              {progress != null
                ? <span className="block h-full bg-surface transition-[width] duration-fast" style={{ width: `${Math.round(progress * 100)}%` }} />
                : <span className="block h-full w-1/2 bg-surface animate-indeterminate" />}
            </span>
          )}
        </Button>
        <Link to={routeCase ? `/cases/${routeCase}` : '/analyses'} className="text-accent underline underline-offset-[3px] decoration-1 hover:text-accent-hover hover:decoration-2 text-small font-medium">Cancel</Link>
        {pctLabel && <span className="sr-only" aria-live="polite">Uploading {pctLabel}</span>}
        {missing && <span id="begin-missing" data-testid="begin-missing" className="basis-full text-small text-text-muted">{missing}</span>}
        {up.error && <span className="text-small text-danger" role="alert">{(up.error as Error).message}</span>}
        {up.error instanceof BillingError && up.error.orderId && <Link to={`/billing/return?order_id=${encodeURIComponent(up.error.orderId)}`} className="text-small text-accent underline underline-offset-4">Resume this order</Link>}
      </div>

      </div>
      <aside aria-label="Examination guide" className="min-w-0">
        {/* The checks available for the selected source and profile. */}
        <h2 className="m-0 eyebrow flex items-center gap-2"><MediaIcon kind={kind} size={16} />Checks for this {tab.label.toLowerCase()}</h2>
        <IntakeChecks kind={kind} profile={profile} cards={catalogue.data} pending={catalogue.isPending} error={catalogue.isError}
          onRetry={() => void catalogue.refetch()} batch={isBatch} />
        <p className="text-small text-text-muted mt-6 mb-0 pt-4 border-t border-border">
          Available checks depend on the source, analysis profile and this computer's resources.
        </p>
      </aside>
      </div>
    </div>
  )
}
