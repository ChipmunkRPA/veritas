import { useEffect, useId, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ApiError, apiClient, unwrap } from '../../lib/api/client'
import type { AnalysisDetail, Conclusion } from '../../lib/api/queries'
import { fmtDateLong } from '../../lib/format'
import { Button, EmptyState } from '../../components/ui'
import NamedAccountPrompt from '../auth/NamedAccountPrompt'
import { textareaClass } from './Dispositions'
import { JUSTIFY_FROM_STEPS, SCALE_POSITIONS, SCALE_WORD, comparedWith, deviationSteps, limitations, needsJustification, reviewReference, stepsFromOf, type ScalePosition } from './review'

function Recorded({ analysis, conclusion, names }: { analysis: AnalysisDetail; conclusion: Conclusion; names: ReadonlyMap<string, string> }) {
  const acknowledged = limitations(analysis).filter((l) => conclusion.limitations_acknowledged.includes(l.id))
  const unknown = conclusion.limitations_acknowledged.filter((id) => !acknowledged.some((l) => l.id === id))
  const steps = conclusion.deviation_steps
  // N-19n (D4-01): under a capture the steps are those from the step of the capture record, and the line says so
  const from = reviewReference(analysis, conclusion) ?? 'the machine decision'
  const rows: [string, React.ReactNode][] = [
    ['Rationale', conclusion.rationale],
    ...(acknowledged.length || unknown.length
      ? [['Limitations', <ul key="l" className="m-0 p-0 list-none">{acknowledged.map((l) => <li key={l.id}><span className="font-mono text-mono text-text-muted mr-2">{l.id}</span>{l.text}</li>)}{unknown.map((id) => <li key={id} className="font-mono text-mono">{id}</li>)}</ul>] as [string, React.ReactNode]]
      : []),
    ...(conclusion.next_steps ? [['Next steps', conclusion.next_steps] as [string, React.ReactNode]] : []),
    ...(conclusion.deviation_justification ? [['Justification', conclusion.deviation_justification] as [string, React.ReactNode]] : []),
  ]
  return (
    <div className="mt-3 border-l-2 border-annotation pl-4" data-testid="conclusion-recorded" data-position={conclusion.scale_position}>
      <p className="m-0 display text-reader text-text">{SCALE_WORD[conclusion.scale_position]}</p>
      <p className="mt-1 mb-0 text-small text-text-muted num">
        Recorded by {names.get(conclusion.author_id) ?? conclusion.author_id}, {fmtDateLong(conclusion.created_at)}
        {' · '}{conclusion.signature_valid ? 'record signature verifies' : <span className="text-danger">record signature does not verify</span>}
      </p>
      {steps != null && steps !== 0 && (
        <p className="mt-2 mb-0 text-small text-text">
          {Math.abs(steps)} {Math.abs(steps) === 1 ? 'step' : 'steps'} {steps > 0 ? 'towards' : 'away from'} AI generation relative to {from}
          {Math.abs(steps) >= JUSTIFY_FROM_STEPS && ': the report highlights the justification.'}
        </p>
      )}
      <dl className="mt-3 m-0 grid grid-cols-[120px_minmax(0,1fr)] gap-x-6 gap-y-2 text-small max-[639px]:grid-cols-1 max-[639px]:gap-y-0.5">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-text-muted">{k}</dt>
            <dd className="m-0 text-text whitespace-pre-line break-words max-w-[72ch] max-[639px]:mb-2">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

/** N-19j (browser QA Q2-09): what the form says when another conclusion was saved for the revision while this one was being
 *  written. The typed text stays in the form. */
export const SAVED_MEANWHILE = 'Another conclusion was saved for this revision while you were writing. What you typed is kept here: show the current conclusion to read it, then revise it.'

function ConclusionForm({ analysis, initial, current, onCancel, onSaved, onDraft }: {
  analysis: AnalysisDetail
  initial: Conclusion | null
  /** the conclusion on record now: it differs from `initial` once another tab or examiner has saved one */
  current?: Conclusion | null
  onCancel?: () => void
  onSaved: () => void
  /** whether the form holds text that is not saved, so that the block keeps it mounted */
  onDraft?: (dirty: boolean) => void
}) {
  const uid = useId()
  // The version this edit starts from (L-5): a conclusion recorded elsewhere since is refused (409), never silently replaced.
  const [base] = useState(initial?.version ?? 0)
  const [first] = useState(initial)
  const machine = analysis.decision?.code ?? null
  const caveats = limitations(analysis)
  const [position, setPosition] = useState<ScalePosition | null>(initial?.scale_position ?? null)
  const [rationale, setRationale] = useState(initial?.rationale ?? '')
  const [acknowledged, setAcknowledged] = useState<Set<string>>(new Set(initial?.limitations_acknowledged ?? []))
  const [nextSteps, setNextSteps] = useState(initial?.next_steps ?? '')
  const [justification, setJustification] = useState(initial?.deviation_justification ?? '')
  const [invalid, setInvalid] = useState<string | null>(null)
  // N-19n (D4-01): under a trusted capture and a trusted declaration the steps are counted from the record's step on the
  // scale (the first, the fifth), on both fusion paths, as the API counts them. Under a capture the form names that step;
  // under a declaration the words do not change
  const stepsFrom = stepsFromOf(analysis.decision)
  const own = comparedWith(machine, stepsFrom)
  const from = reviewReference(analysis) ?? 'the machine decision'
  const justify = position != null && needsJustification(machine, position, stepsFrom)
  const steps = position != null ? deviationSteps(machine, position, stepsFrom) : null
  const dirty = position !== (first?.scale_position ?? null) || rationale !== (first?.rationale ?? '') || nextSteps !== (first?.next_steps ?? '')
    || justification !== (first?.deviation_justification ?? '')
  useEffect(() => { onDraft?.(dirty) }, [dirty, onDraft])
  const overtaken = current != null && current.version !== base

  const put = useMutation({
    mutationFn: () => unwrap(apiClient.PUT('/api/v2/analyses/{aid}/conclusion', {
      params: { path: { aid: analysis.id } },
      body: {
        scale_position: position as ScalePosition, rationale: rationale.trim(), limitations_acknowledged: caveats.filter((l) => acknowledged.has(l.id)).map((l) => l.id),
        next_steps: nextSteps.trim() || undefined, deviation_justification: justify ? justification.trim() : undefined, base_version: base,
      },
    })),
    onSuccess: onSaved,
  })
  // the refusal of a save from a stale tab (409 "Conclusion changed"), said by `SAVED_MEANWHILE`
  const staleSave = put.error instanceof ApiError && put.error.status === 409 && put.error.title === 'Conclusion changed'

  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!position) { setInvalid('Choose a position on the scale.'); return }
    if (!rationale.trim()) { setInvalid('The rationale is required; it prints in the report.'); return }
    if (justify && !justification.trim()) { setInvalid(`This position is ${Math.abs(steps ?? 0)} steps from ${from}: add a justification.`); return }
    setInvalid(null)
    put.mutate()
  }
  const toggle = (id: string) => setAcknowledged((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n })

  return (
    <form onSubmit={submit} className="mt-3 border-l-2 border-annotation pl-4 max-w-[760px]" noValidate data-testid="conclusion-form">
      <fieldset className="m-0 p-0 border-0">
        <legend className="p-0 text-small text-text-muted">Your position</legend>
        <div className="mt-1 flex flex-col">
          {SCALE_POSITIONS.map((p) => (
            <label key={p} className="inline-flex items-baseline gap-2.5 min-h-8 py-1 text-small text-text">
              <input type="radio" name={`${uid}-position`} checked={position === p} onChange={() => setPosition(p)} className="accent-text translate-y-0.5" />
              <span>{SCALE_WORD[p]}{p === own && <span className="text-text-muted"> · {from}</span>}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <label className="mt-4 flex flex-col gap-1 text-small text-text-muted">
        Rationale <span className="sr-only">(required, printed)</span>
        <textarea value={rationale} onChange={(e) => setRationale(e.target.value)} rows={4} maxLength={10000} aria-required="true" className={textareaClass} />
      </label>
      {justify && (
        <label className="mt-4 flex flex-col gap-1 text-small text-text">
          Justification: this position is {Math.abs(steps ?? 0)} steps from {from}
          <textarea value={justification} onChange={(e) => setJustification(e.target.value)} rows={3} maxLength={10000} aria-required="true" className={textareaClass} />
        </label>
      )}
      <fieldset className="mt-4 m-0 p-0 border-0">
        <legend className="p-0 text-small text-text-muted">Limitations acknowledged</legend>
        {caveats.length === 0 ? <p className="mt-1 mb-0 text-small text-text-muted">This analysis records no caveats.</p> : (
          <div className="mt-1 flex flex-col">
            {caveats.map((l) => (
              <label key={l.id} className="inline-flex items-baseline gap-2.5 min-h-8 py-1 text-small text-text">
                <input type="checkbox" checked={acknowledged.has(l.id)} onChange={() => toggle(l.id)} className="accent-text translate-y-0.5" />
                <span><span className="font-mono text-mono text-text-muted mr-2">{l.id}</span>{l.text}</span>
              </label>
            ))}
          </div>
        )}
      </fieldset>
      <label className="mt-4 flex flex-col gap-1 text-small text-text-muted">
        Recommended next steps <span className="sr-only">(optional)</span>
        <textarea value={nextSteps} onChange={(e) => setNextSteps(e.target.value)} rows={2} maxLength={4000} className={textareaClass} />
      </label>
      {/* N-19o (browser QA Q5-04): a conclusion saved meanwhile is said in N-19j's sentence alone, below; the server's own
          ("The conclusion is now version 1, and this request was made from version 0. …") stood beside it */}
      {(invalid ?? (staleSave ? null : put.error?.message)) && <p role="alert" className="mt-3 mb-0 text-small text-danger">{invalid ?? put.error?.message}</p>}
      {(overtaken || (put.error instanceof ApiError && put.error.status === 409)) && (
        <p data-testid="conclusion-overtaken" role="status" className="mt-3 mb-0 text-small text-text">{SAVED_MEANWHILE}</p>
      )}
      <div className="mt-4 flex items-center gap-4">
        {(overtaken || (put.error instanceof ApiError && put.error.status === 409)) && <Button variant="ghost" onClick={onSaved}>Show the current conclusion</Button>}
        <Button variant="primary" type="submit" disabled={put.isPending}>{put.isPending ? 'Recording…' : initial ? 'Record revised conclusion' : 'Record conclusion'}</Button>
        {onCancel && <Button variant="text" onClick={onCancel}>Cancel</Button>}
      </div>
    </form>
  )
}

/** Examiner's conclusion (PRD RW-03; UX §5.14 "never merged into the machine assessment"): the verbal-scale position, rationale,
 *  limitations acknowledged and next steps. Each record is kept; the newest is current. `needsAccount`: the revision can take a
 *  conclusion but the caller is not a named account, so the form is replaced by the way to become one (N-9b). */
export default function ConclusionBlock({ analysis, conclusion, names, writable, needsAccount = false, onChanged }: {
  analysis: AnalysisDetail
  conclusion: Conclusion | null
  names: ReadonlyMap<string, string>
  writable: boolean
  needsAccount?: boolean
  onChanged: () => void
}) {
  const [revising, setRevising] = useState(false)
  // N-19j (browser QA Q2-09): a first conclusion saved from a stale tab was answered 409 and the form gave way to the other
  // tab's conclusion: the typed text was gone and no message said why. The form stays while it holds text that is not saved
  const [drafting, setDrafting] = useState(false)
  const composing = !conclusion || revising || drafting
  const done = () => { setRevising(false); setDrafting(false); onChanged() }
  return (
    <div className="mt-10" data-testid="conclusion">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h3 className="m-0 text-body font-medium text-text">Examiner’s conclusion</h3>
        {writable && conclusion && !composing && <Button variant="text" onClick={() => setRevising(true)}>Revise conclusion</Button>}
      </div>
      {conclusion && !(writable && composing) && <Recorded analysis={analysis} conclusion={conclusion} names={names} />}
      {writable && composing && (
        <ConclusionForm analysis={analysis} initial={conclusion} current={conclusion} onDraft={setDrafting}
          onCancel={conclusion ? () => { setRevising(false); setDrafting(false) } : undefined} onSaved={done} />
      )}
      {!writable && !conclusion && !needsAccount && <EmptyState className="mt-3">No conclusion was recorded for this revision.</EmptyState>}
      {needsAccount && (
        <NamedAccountPrompt>
          {conclusion ? 'A conclusion is revised under a named account, and a sign-off is recorded under one.' : 'No conclusion is recorded yet. A conclusion and a sign-off are recorded under a named account.'}
        </NamedAccountPrompt>
      )}
    </div>
  )
}
