import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useSearchParams } from 'react-router-dom'
import { ApiError, apiClient, unwrap } from '../../lib/api/client'
import { useCase, type AnalysisDetail, type Annotation, type Conclusion, type Finding, type Me, type Signoff } from '../../lib/api/queries'
import { fmtDateLong } from '../../lib/format'
import { Button } from '../../components/ui'
import { basisOf, headline, signerName } from '../analysis/answer'
import { numberPhrase, resultNumber } from '../analysis/resultNumber'
import { recordOf } from '../analysis/record'
import { namedAccount } from '../../lib/api/roles'
import { sentenceCase } from '../cases/labels'
import { textareaClass } from './Dispositions'
import { DECISION_WORD, KIND_WORD, SCALE_WORD, citesEvidence, isNote, needsComment, noteNumbers, recordDecision, reviewReference, signoffAction, stepsWord, type SignoffDecision, type SignoffKind } from './review'

/** How far the conclusion departs from the machine assessment, in words, or null when it agrees (ADR 0011). */
function departure(c: Conclusion): string | null {
  if (!c.departs_from_machine) return null
  const n = Math.abs(c.deviation_steps ?? 0)
  return n ? `by ${n} step${n === 1 ? '' : 's'}` : 'by recording no opinion'
}

const DECISIONS: readonly SignoffDecision[] = ['concur', 'concur_with_changes', 'non_concur']
/** N-19j (browser QA Q2-10): who may record the second review, as the API has it (`v2_review.post_signoff`): an examiner or
 *  reviewer who is an owner or a reviewer of the case, or an administrator, member or not; never the examiner who signed or
 *  the author of the conclusion. The page said "another examiner with the reviewer role on this case". */
export const PEER_REVIEW_BY = 'This case needs a peer review. An owner or reviewer of this case other than the examiner who signed records it, or an administrator.'
/** N-19j: what the comment of a decision that needs one is, in the words of the dialog. */
const COMMENT_ASKED: Partial<Record<SignoffDecision, string>> = {
  non_concur: 'Give the reason. It is printed in the report.', concur_with_changes: 'Say which changes. They are printed in the report.',
}

export function SignoffDialog({ open, kind, analysis, findings, conclusion, annotations, me, onClose, onSigned, onReload }: {
  open: boolean
  kind: SignoffKind
  analysis: AnalysisDetail
  findings: readonly Finding[]
  conclusion: Conclusion | null
  annotations: readonly Annotation[]
  me: Me | null
  onClose: () => void
  onSigned: () => void
  onReload: () => void
}) {
  const ref = useRef<HTMLDialogElement>(null)
  const uid = useId()
  // Mounted afresh for each opening, so nothing carries over from an earlier attempt. No default opinion (design review P0-3).
  // The conclusion is the one shown when the dialog opened; the request names it (id and version), so a revision made elsewhere
  // meanwhile is refused instead of being locked unseen (QA P1-6).
  const [shown] = useState<Conclusion | null>(conclusion)
  // N-19j: where the machine's result is not assessable the examiner's sign-off has no object: it records the opinion without a
  // comparison, and no decision is asked (the record is made as "concur", which nothing prints)
  const noObject = kind === 'examiner' && shown != null && shown.departs_from_machine == null
  const [decision, setDecision] = useState<SignoffDecision | null>(noObject ? 'concur' : null)
  const [comment, setComment] = useState('')
  const [missing, setMissing] = useState(false)
  const [uncommented, setUncommented] = useState(false)
  const post = useMutation({
    mutationFn: () => unwrap(apiClient.POST('/api/v2/analyses/{aid}/signoffs', {
      params: { path: { aid: analysis.id } },
      body: { kind, decision: decision as SignoffDecision, comment: comment.trim() || undefined, conclusion_id: shown?.id ?? '', conclusion_version: shown?.version ?? 1 },
    })),
    onSuccess: onSigned,
  })

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    else if (!open && d.open) d.close()
  }, [open])

  const d = analysis.decision
  const basis = d ? basisOf(d, findings) : null
  // D-68: the number beside the machine assessment is the one the result page shows, said as what it is; N-19n (D4-03):
  // whose it is under a record; (D4-04) in the line's case ("… · Estimate of the statistical checks alone 7–56" stood there)
  const number = numberPhrase(resultNumber(d, findings, analysis.condition_profile), recordOf(d) !== null)
  const machine = d && basis ? `${headline(d, basis)}${number ? ` · ${number.charAt(0).toLowerCase()}${number.slice(1)}` : ''}` : 'No machine decision'
  const numbers = noteNumbers(annotations)
  const inReport = annotations.filter((a) => isNote(a) && a.visibility === 'in_report').map((a) => numbers.get(a.id)).filter(Boolean)
  const warn = decision === 'non_concur' && !citesEvidence(`${comment} ${shown?.rationale ?? ''}`)
  const departs = kind === 'examiner' && shown ? departure(shown) : null
  // N-19n (D4-01): under a capture the dialog names the step of the capture record. What may be signed is unchanged, and
  // the decision keeps its object, the machine assessment: under a capture that is the record's result
  const reference = kind === 'examiner' && shown ? reviewReference(analysis, shown) : null
  const away = Math.abs(shown?.deviation_steps ?? 0)
  const where = !departs ? '' : !reference ? ` · departs from the machine assessment ${departs}` : away ? ` · ${stepsWord(away)} from ${reference}` : ` · not on ${reference}`
  // N-19 (browser QA P1-7): neither concurring decision beside a conclusion that departs. N-19j (Q2-03): and the mirror, no
  // "do not concur" beside a conclusion equal to the machine's
  const agrees = kind === 'examiner' && shown?.departs_from_machine === false
  const barred = (x: SignoffDecision) => (x === 'non_concur' ? agrees : departs != null)
  const commentNeeded = needsComment(decision)
  const submit = (e: FormEvent) => {
    e.preventDefault()
    setMissing(!decision)
    setUncommented(commentNeeded && !comment.trim())
    if (!decision || (commentNeeded && !comment.trim())) return
    post.mutate()
  }
  const changed = post.error instanceof ApiError && post.error.status === 409

  const rows: [string, React.ReactNode][] = [
    ['Machine assessment', machine],
    [kind === 'examiner' ? 'Your conclusion' : 'Examiner’s conclusion', shown ? (
      <>{SCALE_WORD[shown.scale_position]}<span className="block text-caption text-text-muted num">{`version ${shown.version}${where}`}</span></>
    ) : '—'],
  ]
  return (
    <dialog ref={ref} aria-labelledby={`${uid}-title`} data-testid="signoff-dialog" onCancel={(e) => { e.preventDefault(); onClose() }}
      className="m-auto w-[640px] max-w-[calc(100vw-32px)] p-0 rounded-md border border-border bg-surface text-text" style={{ boxShadow: 'var(--shadow-popover)' }}>
      <form onSubmit={submit} noValidate>
        <h2 id={`${uid}-title`} className="m-0 px-6 pt-6 pb-4 text-[17px] leading-6 font-medium border-b border-border">
          {kind === 'examiner' ? 'Sign off analysis' : 'Peer review of analysis'} <span className="font-mono text-mono">{analysis.id.slice(0, 8)}</span>
        </h2>
        <div className="px-6 py-5 grid grid-cols-[150px_minmax(0,1fr)] gap-x-6 gap-y-4 text-small max-[559px]:grid-cols-1 max-[559px]:gap-y-1">
          {rows.map(([k, v]) => (
            <div key={k} className="contents">
              <span className="text-text-muted">{k}</span>
              <span className="text-text">{v}</span>
            </div>
          ))}
          <span id={`${uid}-decision`} className="text-text-muted">Decision</span>
          {noObject ? (
            <p data-testid="signoff-no-object" className="m-0 text-text">The machine assessment gave no result, so there is nothing to concur with. This sign-off records your opinion without a comparison.</p>
          ) : (
            <div role="radiogroup" aria-labelledby={`${uid}-decision`} className="flex flex-col">
              {DECISIONS.map((x) => (
                <label key={x} className={`inline-flex items-center gap-2.5 min-h-8 ${barred(x) ? 'text-text-muted' : 'text-text'}`}>
                  <input type="radio" name={`${uid}-decision`} checked={decision === x} disabled={barred(x)} onChange={() => { setDecision(x); setUncommented(false) }} className="accent-text" /> {DECISION_WORD[kind][x]}
                </label>
              ))}
              {departs && <p className="mt-1 mb-0 text-caption text-text-muted">{!reference ? 'Your conclusion departs from the machine assessment, so it cannot be signed as concurring with it.'
                : away ? `Your conclusion is ${stepsWord(away)} from ${reference}, so it cannot be signed as concurring with the machine assessment.`
                : 'Your conclusion records no opinion on the scale, so it cannot be signed as concurring with the machine assessment.'}</p>}
              {agrees && <p className="mt-1 mb-0 text-caption text-text-muted">{reference ? `Your conclusion is on ${reference}, so it cannot be signed as not concurring with the machine assessment.`
                : 'Your conclusion is the machine assessment’s own, so it cannot be signed as not concurring with it.'}</p>}
              {missing && <p role="alert" className="mt-1 mb-0 text-caption text-danger">Choose a decision.</p>}
              {warn && <p className="mt-1 mb-0 text-caption text-text">A sign-off that does not concur should name the findings it disagrees with, for example F2.</p>}
            </div>
          )}
          <label htmlFor={`${uid}-comment`} className="text-text-muted">Comment <span className="block text-caption">{commentNeeded ? 'required, printed' : 'optional, printed'}</span></label>
          <div>
            <textarea id={`${uid}-comment`} value={comment} onChange={(e) => { setComment(e.target.value); setUncommented(false) }} rows={3} maxLength={4000} required={commentNeeded}
              aria-invalid={uncommented || undefined} aria-describedby={uncommented ? `${uid}-comment-error` : undefined} className={`${textareaClass} w-full`} />
            {uncommented && decision && <p id={`${uid}-comment-error`} role="alert" className="mt-1 mb-0 text-caption text-danger">{COMMENT_ASKED[decision]}</p>}
          </div>
          <span className="text-text-muted">Notes in report</span>
          <span className="text-text num">{inReport.length ? inReport.join(', ') : 'None'}<span className="text-text-muted"> · internal notes do not print</span></span>
          <span className="text-text-muted">Signer</span>
          <span className="text-text">{me ? `${me.display_name} · ${sentenceCase(me.role)}` : '—'}</span>
        </div>
        <div className="dialog-actions px-6 py-4 border-t border-border">
          <p className="m-0 text-small text-text">{kind === 'examiner' ? 'Signing locks this revision. Later changes are recorded as a new revision.' : 'The peer review is signed with your identity and recorded in the custody log.'}</p>
          {post.error && <p role="alert" className="mt-2 mb-0 text-small text-danger">{post.error.message}</p>}
          <div className="mt-4 flex items-center justify-end gap-4">
            <Button variant="text" onClick={onClose}>Cancel</Button>
            {changed ? <Button variant="primary" onClick={onReload}>Reload the analysis</Button> : <Button variant="primary" type="submit" disabled={post.isPending}>{post.isPending ? 'Signing…' : kind === 'examiner' ? 'Sign and lock' : 'Sign peer review'}</Button>}
          </div>
        </div>
      </form>
    </dialog>
  )
}

/** Sign-off (PRD RW-04; ARCHITECTURE §5.4; UX §5.14): the signed records with their HMAC check, and the sign-off the caller can
 *  make now. The examiner sign-off needs a conclusion by the signer and locks the revision; a peer review follows when the case
 *  requires one. */
export default function SignoffBlock({ analysis, findings, conclusion, records, annotations, me, names, onChanged }: {
  analysis: AnalysisDetail
  findings: readonly Finding[]
  conclusion: Conclusion | null
  records: readonly Signoff[]
  annotations: readonly Annotation[]
  me: Me | null
  names: ReadonlyMap<string, string>
  onChanged: () => void
}) {
  const [open, setOpen] = useState(false)
  const kase = useCase(analysis.case_id ?? '')
  const standing = me && kase.data ? { roles: me.roles, caseRole: kase.data.my_role } : undefined
  // N-9b: a sign-off is refused to a caller that is not a named account, so it is not offered to one (the conclusion block says
  // how to sign in)
  const named = namedAccount(me)
  const action = named ? signoffAction(analysis, conclusion, records, me?.id, standing) : null
  const examiner = records.find((s) => s.kind === 'examiner')
  const blocked = analysis.review?.state === 'withdrawn' || analysis.review?.state === 'superseded'
  const awaitsPeer = !!examiner && !!analysis.review?.requires_second_review && !records.some((s) => s.kind === 'peer_review')
  const hint = blocked ? null
    : analysis.review?.state === 'disputed' ? 'The peer reviewer does not concur, so this revision is not issued. The way forward is a new revision.'
    : me && !named && (!examiner || awaitsPeer) ? `${examiner ? 'A peer review' : 'A sign-off'} is recorded under a named account.`
    : !examiner && !conclusion ? 'Record a conclusion before signing off.'
      : !examiner && conclusion && conclusion.author_id !== me?.id ? 'The examiner who recorded the conclusion signs it off.'
        : awaitsPeer && !action ? PEER_REVIEW_BY
          : null

  // N-19a: "Record peer review…" in the page's header leads here and opens the dialog (`PeerReviewAction`)
  const [params, setParams] = useSearchParams()
  const asked = params.get('signoff') === 'peer'
  useEffect(() => {
    if (!asked || action === null) return
    if (action === 'peer_review') setOpen(true)
    setParams((p) => { const next = new URLSearchParams(p); next.delete('signoff'); return next }, { replace: true })
  }, [asked, action, setParams])

  return (
    <div className="mt-10" data-testid="signoffs">
      <h3 className="m-0 text-body font-medium text-text">Sign-off</h3>
      {records.length === 0 ? <p className="mt-2 mb-0 text-small text-text-muted">Not signed off.</p> : (
        <ul className="mt-3 m-0 p-0 list-none border-t border-border">
          {records.map((s) => (
            <li key={s.id} data-testid="signoff" data-kind={s.kind} className="border-b border-border py-2.5 text-small">
              <span className="flex items-baseline gap-2">
                <span aria-hidden className="inline-block w-3 text-center shrink-0">●</span>
                <span className="text-text">
                  <span className="font-medium">{KIND_WORD[s.kind]}</span>
                  {` · ${recordDecision(s, analysis.review?.examiner_departs, analysis.review?.examiner_reference ? reviewReference(analysis) : null)} · ${signerName(s, names.get(s.principal_id)) ?? s.principal_id}`}
                  <span className="text-text-muted num">{` · ${fmtDateLong(s.signed_at)} · revision ${s.analysis_revision} · `}</span>
                  <span className={s.signature_valid ? 'text-text-muted' : 'text-danger'}>{s.signature_valid ? 'signature verifies' : 'signature does not verify'}</span>
                </span>
              </span>
              {s.comment && <span className="block mt-1 pl-5 text-text whitespace-pre-line max-w-[72ch]">{s.comment}</span>}
            </li>
          ))}
        </ul>
      )}
      {hint && <p className="mt-3 mb-0 text-small text-text-muted">{hint}</p>}
      {action && (
        <Button variant="primary" className="mt-4" onClick={() => setOpen(true)}>{action === 'examiner' ? 'Sign off…' : 'Record peer review…'}</Button>
      )}
      {open && <SignoffDialog open={open} kind={action ?? 'examiner'} analysis={analysis} findings={findings} conclusion={conclusion} annotations={annotations} me={me}
        onClose={() => setOpen(false)} onSigned={() => { setOpen(false); onChanged() }} onReload={() => { setOpen(false); onChanged() }} />}
    </div>
  )
}
