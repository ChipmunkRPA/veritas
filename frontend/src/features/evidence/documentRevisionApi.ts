import { apiClient, authHeaders, unwrap } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'

export type DraftComparison = components['schemas']['DraftComparison']
export type DraftExcerpt = components['schemas']['DraftExcerpt']
export type DraftExtension = NonNullable<components['schemas']['DocumentRevisionAvailability']['extension']>
export type DocumentRevisionResponse = components['schemas']['DocumentRevisionResponse']

const route = '/api/v2/analyses/{aid}/document-revision-comparison'

export async function draftAvailability(analysisId: string, signal: AbortSignal) {
  const result = await unwrap(apiClient.GET(route, { params: { path: { aid: analysisId } }, signal }))
  return result.available ? result.extension ?? null : null
}

export async function compareDraft(analysisId: string, file: File, signal: AbortSignal): Promise<DocumentRevisionResponse> {
  const body = new FormData()
  body.append('earlier', file)
  body.append('include_pdf', 'true')
  const response = await fetch(route.replace('{aid}', encodeURIComponent(analysisId)), {
    method: 'POST', headers: authHeaders('POST'), body, signal,
  })
  const data: unknown = await response.json().catch(() => null)
  signal.throwIfAborted()
  const result = await unwrap(Promise.resolve({ response, data: response.ok ? data as DocumentRevisionResponse : undefined,
    error: response.ok ? undefined : data }))
  if (result?.comparison?.schema !== 'chipmunk.document-revision-comparison/1' || !Array.isArray(result.comparison.changes)) {
    throw new Error('The server returned an incomplete draft comparison.')
  }
  return result
}
