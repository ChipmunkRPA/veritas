/** Custody log wording (UX §5.12; PRD CC-01, CC-02; `app/audit/chain.py` actions and the details each call site records). Rows
 *  are never editable: this module only words the recorded events. */
import type { AuditEvent } from '../../lib/api/queries'
import { fmtDay, shortHash } from '../../lib/format'
import { STEP_OF_CAPTURE, STEP_OF_DECLARATION } from '../analysis/record'
import { DECISION_WORD, KIND_WORD, NO_COMPARISON, SCALE_WORD, STATUS_WORD, type ExaminerStatus, type ScalePosition, type SignoffDecision, type SignoffKind } from '../review/review'

export const ACQUISITION_WORD: Record<string, string> = {
  not_stated: 'acquisition not stated', custodian_upload: 'upload from custodian', url_download: 'downloaded from URL',
  screen_capture: 'screen capture', email_forward: 'forwarded by email', quick_upload: 'quick upload', other: 'other acquisition',
}

/** How the actor of a conclusion or sign-off authenticated (recorded since N-3). */
const AUTH_WORD: Record<string, string> = { session: 'browser session', api_key: 'API key' }

const WORD: Record<string, string> = {
  intake: 'received', original_viewed: 'original viewed', original_downloaded: 'original downloaded', hash_verified: 'hash verified',
  hash_mismatch: 'hash mismatch', analysis_started: 'analysis started', analysis_finished: 'analysed', rerun: 're-analysed',
  manifest_recorded: 'manifest recorded', annotation_created: 'annotation added', annotation_edited: 'annotation edited',
  annotation_deleted: 'annotation deleted', note_added: 'note added', status_changed: 'status changed',
  conclusion_recorded: 'conclusion recorded', signoff: 'signed off', report_generated: 'report generated',
  package_exported: 'exported', redaction: 'redacted', retention_purge: 'purged', deletion_requested: 'deletion requested',
  permission_changed: 'members changed', fuser_loaded: 'fusion model loaded', case_created: 'case opened',
  case_updated: 'case details changed', login: 'signed in', login_failed: 'sign-in failed', policy_changed: 'policy changed',
  webhook_configured: 'webhook configured', fuser_review_recorded: 'fusion model review recorded',
  fuser_review_withdrawn: 'fusion model review withdrawn', signing_key_adopted: 'signing key file written',
}

const FINISHED: Record<string, string> = { done: 'finished', failed: 'failed', cancelled: 'cancelled', deleted: 'deleted before it finished' }
const CUSTODIAN_HASH: Record<string, string> = { match: 'custodian hash agrees', mismatch: 'custodian hash differs' }
// Audit fields are machine keys. Translate the field names only; recorded note text and other user values stay verbatim.
const ANNOTATION_FIELD_WORD: Record<string, string> = {
  body: 'note text', examiner_status: 'examiner status', geometry: 'geometry', visibility: 'visibility',
  target_type: 'target type', target_id: 'target ID', shape: 'shape',
}


export const eventWord = (e: Pick<AuditEvent, 'action' | 'details'>): string =>
  e.action === 'signoff' && e.details.kind === 'peer_review' ? 'peer review signed' : WORD[e.action] ?? e.action.replace(/_/g, ' ')

/** N-19k (Q2-14): what an intake says of a file whose bytes this instance holds already; it read "bytes taken in before". */
export const SAME_BYTES = 'the same file as an earlier intake'

/** An event that needs attention: a failed re-hash or a failed analysis. */
export const isAlarm = (e: Pick<AuditEvent, 'action' | 'details'>): boolean =>
  e.action === 'hash_mismatch' || (e.action === 'analysis_finished' && e.details.status === 'failed')

/** The details column: what the event recorded, in words. Principals named in the details read by display name when known. */
export function eventDetail(e: Pick<AuditEvent, 'action' | 'details'>, names: ReadonlyMap<string, string> = new Map()): string {
  const raw = e.details
  // a member by the name the member list has, else by the name the event recorded (a removed member; since N-19a), else by id
  const named = typeof raw.principal_id === 'string' ? names.get(raw.principal_id) ?? (typeof raw.principal_name === 'string' && raw.principal_name ? raw.principal_name : undefined) : undefined
  const d = named ? { ...raw, principal_id: named } : raw
  const s = (k: string): string | null => (typeof d[k] === 'string' && d[k] !== '' ? (d[k] as string) : null)
  const join = (...parts: (string | null | undefined | false)[]) => parts.filter(Boolean).join(' · ')
  switch (e.action) {
    case 'intake':
      return join(s('filename'), ACQUISITION_WORD[s('acquisition_method') ?? 'not_stated'], CUSTODIAN_HASH[s('custodian_hash_match') ?? ''],
        d.new_media === false && SAME_BYTES)
    case 'analysis_started': return d.refusion ? 're-fusion of stored scores' : ''
    case 'analysis_finished': return FINISHED[s('status') ?? ''] ?? s('status') ?? ''
    case 'hash_verified':
    case 'hash_mismatch': return s('purpose') ? `before ${s('purpose')!.replace(/_/g, ' ')}` : ''
    case 'note_added':
    case 'annotation_created':
      return join(s('target_id'), d.examiner_status ? STATUS_WORD[d.examiner_status as ExaminerStatus] : null, d.visibility === 'in_report' && 'in report')
    case 'annotation_edited': return join(Array.isArray(d.fields) && d.fields.filter((f): f is string => typeof f === 'string')
        .map((f) => ANNOTATION_FIELD_WORD[f] ?? f.replace(/_/g, ' ')).join(', '),
      s('examiner_status') && STATUS_WORD[s('examiner_status') as ExaminerStatus])
    case 'conclusion_recorded':
      return join(SCALE_WORD[s('scale_position') as ScalePosition], typeof d.deviation_steps === 'number' && d.deviation_steps !== 0 && // N-19n (D4-01): an event recorded since says what its steps are counted from; one recorded before keeps its line
        `${Math.abs(d.deviation_steps)} ${Math.abs(d.deviation_steps) === 1 ? 'step' : 'steps'} from ${d.steps_from === 'capture_record' ? STEP_OF_CAPTURE
          // N-19o (D5-01): a declaration's step where the code is not that step (the fallback)
          : d.steps_from === 'declaration' && d.machine_code !== 'ai_supported_strong' ? STEP_OF_DECLARATION : 'the machine decision'}`,
        AUTH_WORD[s('auth') ?? ''])
    case 'signoff':
      // N-19j: the decision with its object; an event that says its sign-off has none (`object: null`) prints that
      return join(KIND_WORD[s('kind') as SignoffKind], d.kind === 'examiner' && d.object === null ? NO_COMPARISON : DECISION_WORD[s('kind') as SignoffKind]?.[s('decision') as SignoffDecision], s('signer'), d.locks_revision === true && 'revision locked',
        AUTH_WORD[s('auth') ?? ''])
    case 'status_changed': return join(s('from') && s('to') && `${s('from')!.replace(/_/g, ' ')} to ${s('to')!.replace(/_/g, ' ')}`, s('reason'))
    case 'rerun': return join(typeof d.revision === 'number' && `revision ${d.revision}`, s('mode'),
      d.supersedes_signed === true && 'supersedes a signed revision', s('reason'))
    case 'manifest_recorded': return s('pipeline_version') ?? ''
    case 'case_created': return join(s('reference'), s('title'))
    case 'case_updated': return caseChanges(d)
    case 'permission_changed':
      return d.change === 'case_member_set' ? join(s('principal_id'), s('case_role') && `as ${s('case_role')}`)
        : d.change === 'case_member_removed' ? join(s('principal_id'), 'removed') : s('change')?.replace(/_/g, ' ') ?? ''
    case 'report_generated': return s('profile') ?? ''
    default: return ''
  }
}

const CASE_FIELD_WORD: Record<string, string> = {
  title: 'title', matter_reference: 'client / matter', description: 'description', classification: 'classification',
  case_type: 'case type', requires_second_review: 'second-review rule', legal_hold: 'legal hold', retention_until: 'retention date',
  status: 'status',
}

/** A `case_updated` event in words (D-70: the custody log shows each change): a rule with the value it was given, when the event
 *  recorded one (since N-19a); a title, a reference and a description by name only. */
function caseChanges(d: Record<string, unknown>): string {
  const fields = Array.isArray(d.fields) ? d.fields.filter((f): f is string => typeof f === 'string') : []
  const values = d.values && typeof d.values === 'object' ? (d.values as Record<string, unknown>) : {}
  return fields.map((f) => {
    if (!(f in values)) return CASE_FIELD_WORD[f] ?? f.replace(/_/g, ' ')
    const v = values[f]
    switch (f) {
      case 'status': return v === 'closed' ? 'case closed' : v === 'open' ? 'case reopened' : `status ${String(v).replace(/_/g, ' ')}`
      case 'legal_hold': return v ? 'legal hold on' : 'legal hold lifted'
      case 'requires_second_review': return v ? 'second review required' : 'second review no longer required'
      case 'retention_until': return typeof v === 'string' ? `retained until ${fmtDay(v)}` : 'retention date removed'
      case 'classification': return `classification ${String(v)}`
      default: return CASE_FIELD_WORD[f] ?? f.replace(/_/g, ' ')
    }
  }).join(' · ')
}

/** The actor's display name: the name the log itself gives (N-19j, browser QA Q2-10: an administrator who is no member of
 *  the case stood there by id), else case members and the caller by id; `system` is the engine itself. */
export function actorName(id: string, names: ReadonlyMap<string, string>, given?: string | null): string {
  if (id === 'system') return 'Veritas'
  return given || names.get(id) || id
}

export interface ExhibitLabels { media: ReadonlyMap<string, string>; analyses: ReadonlyMap<string, string> }

/** "Ex. B" for events about one exhibit (by media object, else by analysis). */
export function exhibitOf(e: Pick<AuditEvent, 'media_id' | 'analysis_id'>, labels: ExhibitLabels): string | null {
  const label = (e.media_id && labels.media.get(e.media_id)) || (e.analysis_id && labels.analyses.get(e.analysis_id)) || null
  return label ? `Ex. ${label}` : null
}

export const eventHash = (e: Pick<AuditEvent, 'after_hash' | 'before_hash'>): string | null => {
  const h = e.after_hash ?? e.before_hash
  return h ? shortHash(h) : null
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const pad = (n: number) => String(n).padStart(2, '0')

/** "13 Sep 2026 11:34:33 UTC": custody timestamps are UTC, whatever the reader's time zone. The day has no leading zero
 *  ("4 Oct 2026 12:18:25 UTC"), as in `fmtDate` and in the reports: one form for a date with its time (N-19k). */
export function fmtUtc(iso: string): string {
  const t = new Date(iso)
  if (Number.isNaN(t.getTime())) return iso
  return `${t.getUTCDate()} ${MONTHS[t.getUTCMonth()]} ${t.getUTCFullYear()} ${pad(t.getUTCHours())}:${pad(t.getUTCMinutes())}:${pad(t.getUTCSeconds())} UTC`
}
