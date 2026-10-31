/** The case & custody fields of the intake form (UX §5.1; PRD CC-01) and their mapping onto `POST /cases/{id}/items`. */
import type { AcquisitionMethod, IntakeFields } from '../../lib/api/intake'

export interface CustodyDraft {
  method: AcquisitionMethod
  /** Free text when the method is "other". */
  other: string
  custodian: string
  source: string
  /** A `datetime-local` value, in the reader's local time. */
  received: string
  statedHash: string
  notes: string
}

export const EMPTY_DRAFT: CustodyDraft = { method: 'not_stated', other: '', custodian: '', source: '', received: '', statedHash: '', notes: '' }

export const ACQUISITION_OPTIONS: readonly { id: AcquisitionMethod; label: string }[] = [
  { id: 'custodian_upload', label: 'Upload from custodian' },
  { id: 'url_download', label: 'Downloaded from URL' },
  { id: 'screen_capture', label: 'Screen capture' },
  { id: 'email_forward', label: 'Forwarded by email' },
  { id: 'other', label: 'Other' },
  { id: 'not_stated', label: 'Not stated' },
]

/** N-19j (browser QA Q2-06; `v2_cases.RECEIVED_CLOCK_SKEW`): a time received later than the intake is refused; a few minutes are
 *  left for the difference between this clock and the server's. It was accepted up to a day ahead. */
export const RECEIVED_SKEW_MS = 5 * 60_000
export const RECEIVED_AFTER_INTAKE = 'The time received is later than now. A file is received before it is taken in: correct the time, or leave it empty.'

/** A `datetime-local` value as ISO 8601 UTC; undefined when empty or unreadable. */
export function localToUtc(value: string): string | undefined {
  if (!value.trim()) return undefined
  const t = new Date(value)  // a date-time without an offset is local time
  return Number.isNaN(t.getTime()) ? undefined : t.toISOString()
}

/** A custodian's hash as the API reads it (`app/cases/intake.py` `stated_hash`): 64 hexadecimal characters (SHA-256) or 32
 *  (MD5), with an optional `sha256:` / `md5:` prefix that agrees with the length. */
const STATED_HASH = /^(?:(sha256|md5):)?([0-9a-f]{64}|[0-9a-f]{32})$/
export function statedHashReadable(value: string): boolean {
  const match = STATED_HASH.exec(value.trim().toLowerCase())
  return match !== null && (match[1] === undefined || match[1] === (match[2].length === 32 ? 'md5' : 'sha256'))
}

/** A source the form takes: a web address with http or https. */
export function sourceIsUrl(value: string): boolean {
  try {
    const url = new URL(value.trim())
    return (url.protocol === 'http:' || url.protocol === 'https:') && url.hostname !== '' && !/\s/.test(value.trim())
  } catch {
    return false
  }
}

/** The problem with a draft, in plain words and before anything is sent (N-19a; design review, P2 "screens" 10; browser QA
 *  "Workflow"): a mistyped custodian hash was refused only after the upload, with the name of the API's field, and a source
 *  that is no address drew no message. Null when the form can be sent. */
export function draftProblem(d: CustodyDraft, now = Date.now()): string | null {
  const problems = draftProblems(d, now)
  return problems.source ?? problems.received ?? problems.statedHash ?? null
}

export type DraftField = 'source' | 'received' | 'statedHash'

/** N-19k (design re-review D2-09): the problem of each field, all at once, so that each stands under its own field. The form
 *  showed the first one only, at the foot of the block: a bad hash drew no message until the source was corrected. */
export function draftProblems(d: CustodyDraft, now = Date.now()): Partial<Record<DraftField, string>> {
  const out: Partial<Record<DraftField, string>> = {}
  if (d.source.trim() && !sourceIsUrl(d.source)) {
    out.source = 'The source is not a web address. Write it with https:// or http://, or leave it empty and name the device in the notes.'
  }
  const utc = localToUtc(d.received)
  if (d.received.trim() && utc === undefined) out.received = 'Enter the received date and time, or leave it empty.'
  else if (utc && Date.parse(utc) > now + RECEIVED_SKEW_MS) out.received = RECEIVED_AFTER_INTAKE
  if (d.statedHash.trim() && !statedHashReadable(d.statedHash)) {
    out.statedHash = 'The custodian’s hash is not a SHA-256 (64 characters, 0 to 9 and a to f) or an MD5 (32 characters). Check it, or leave it empty.'
  }
  return out
}

/** The multipart fields. "Other" has no API field of its own, so its text opens the intake notes. */
export function toIntakeFields(d: CustodyDraft): IntakeFields {
  const other = d.method === 'other' && d.other.trim() ? `Acquired by: ${d.other.trim()}` : null
  const notes = [other, d.notes.trim() || null].filter(Boolean).join('\n')
  return {
    acquisition_method: d.method,
    supplied_by: d.custodian.trim() || undefined,
    source: d.source.trim() || undefined,
    date_acquired: localToUtc(d.received),
    custodian_stated_hash: d.statedHash.trim() || undefined,
    notes: notes || undefined,
  }
}

const key = (caseId: string) => `veritas_intake_${caseId}`

/** WCAG 3.3.7: the acquisition method and the custodian carry over to the next exhibit of the same case (this tab only). */
export function rememberDraft(caseId: string, d: CustodyDraft): void {
  try {
    sessionStorage.setItem(key(caseId), JSON.stringify({ method: d.method, other: d.other, custodian: d.custodian }))
  } catch { /* storage unavailable: nothing to carry over */ }
}

export function recallDraft(caseId: string): CustodyDraft {
  try {
    const raw = sessionStorage.getItem(key(caseId))
    if (!raw) return EMPTY_DRAFT
    const p = JSON.parse(raw) as Partial<Record<'method' | 'other' | 'custodian', unknown>>
    const method = ACQUISITION_OPTIONS.find((o) => o.id === p.method)?.id ?? 'not_stated'
    return { ...EMPTY_DRAFT, method, other: typeof p.other === 'string' ? p.other : '', custodian: typeof p.custodian === 'string' ? p.custodian : '' }
  } catch {
    return EMPTY_DRAFT
  }
}
