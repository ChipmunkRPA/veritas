/** Upload into a case (`POST /api/v2/cases/{case_id}/items?analyze=true`; UX §5.1; PRD CC-01). An XHR, because fetch reports no
 *  upload progress. The browser pre-hash travels as `client_sha256`; the server refuses the upload when its hash differs. */
import { sha256Blob } from '../hash'
import { authHeaders } from './client'
import type { components } from './schema'
import { refusedSession } from './sessionEnd'

export type IntakeResult = components['schemas']['IntakeResult']
export type AcquisitionMethod = components['schemas']['IntakeOut']['acquisition_method']
export type MediaKind = 'image' | 'video' | 'audio' | 'document'
export type AnalysisProfile = NonNullable<components['schemas']['BatchOut']['profile']>

export interface IntakeFields {
  acquisition_method: AcquisitionMethod
  supplied_by?: string
  source?: string
  /** ISO 8601 with a UTC offset. */
  date_acquired?: string
  custodian_stated_hash?: string
  notes?: string
}

export interface IntakeCallbacks {
  onHashed?: (sha256: string) => void
  onProgress?: (fraction: number) => void
}

/** The message of a failed intake: the problem+json `detail`, else the status text. */
export function intakeError(status: number, statusText: string, body: string): string {
  try {
    const detail = (JSON.parse(body) as { detail?: unknown }).detail
    if (typeof detail === 'string' && detail) return detail
  } catch { /* not JSON */ }
  return statusText || `HTTP ${status}`
}

export function intakeForm(kind: MediaKind, file: Blob, filename: string, sha256: string, fields: IntakeFields): FormData {
  const fd = new FormData()
  fd.append('kind', kind)
  fd.append('client_sha256', sha256)
  fd.append('acquisition_method', fields.acquisition_method)
  for (const key of ['supplied_by', 'source', 'date_acquired', 'custodian_stated_hash', 'notes'] as const) {
    const value = fields[key]?.trim()
    if (value) fd.append(key, value)
  }
  fd.append('file', file, filename)
  return fd
}

/** The intake URL: `profile` travels only when one is chosen, else the server default applies. */
export function intakeUrl(caseId: string, profile?: AnalysisProfile): string {
  return `/api/v2/cases/${encodeURIComponent(caseId)}/items?analyze=true${profile ? `&profile=${profile}` : ''}`
}

export async function intakeItem(caseId: string, kind: MediaKind, file: Blob, filename: string, fields: IntakeFields, cb: IntakeCallbacks = {},
  profile?: AnalysisProfile): Promise<IntakeResult> {
  const sha256 = await sha256Blob(file)
  cb.onHashed?.(sha256)
  return new Promise<IntakeResult>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', intakeUrl(caseId, profile))
    for (const [k, v] of Object.entries(authHeaders('POST'))) xhr.setRequestHeader(k, v)
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) cb.onProgress?.(e.loaded / e.total) }
    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { resolve(JSON.parse(xhr.responseText) as IntakeResult) } catch { reject(new Error('Malformed response')) }
      } else {
        const message = intakeError(xhr.status, xhr.statusText, xhr.responseText)
        reject(new Error(xhr.status === 401 ? refusedSession(message) : message))  // N-19a: the session ended during the intake
      }
    }
    xhr.send(intakeForm(kind, file, filename, sha256, fields))
  })
}
