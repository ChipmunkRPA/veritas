import { apiClient, authHeaders, unwrap } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'

export type DraftComparison = components['schemas']['DraftComparison']
export type DraftExcerpt = components['schemas']['DraftExcerpt']
export type DraftExtension = NonNullable<components['schemas']['DocumentRevisionAvailability']['extension']>
export type DocumentRevisionResponse = components['schemas']['DocumentRevisionResponse']

const route = '/api/v2/analyses/{aid}/document-revision-comparison'

export async function draftAvailability(analysisId: string, signal: AbortSignal): Promise<{
  extension: DraftExtension; earlierExtensions?: DraftExtension[]; pairedCheckAvailable: boolean
} | null> {
  const result = await unwrap(apiClient.GET(route, { params: { path: { aid: analysisId } }, signal }))
  return result.available && result.extension
    ? { extension: result.extension, earlierExtensions: result.earlier_extensions ?? [result.extension],
      pairedCheckAvailable: result.paired_check_available === true } : null
}

export async function compareDraft(analysisId: string, file: File, signal: AbortSignal, includePairedCheck = false): Promise<DocumentRevisionResponse> {
  const body = new FormData()
  body.append('earlier', file)
  body.append('include_pdf', 'true')
  if (includePairedCheck) body.append('include_paired_check', 'true')
  const response = await fetch(route.replace('{aid}', encodeURIComponent(analysisId)), {
    method: 'POST', headers: authHeaders('POST'), body, signal,
  })
  const data: unknown = await response.json().catch(() => null)
  signal.throwIfAborted()
  const result = await unwrap(Promise.resolve({ response, data: response.ok ? data as DocumentRevisionResponse : undefined,
    error: response.ok ? undefined : data }))
  const schema = result?.comparison?.schema
  if ((schema !== 'chipmunk.document-revision-comparison/1' && schema !== 'chipmunk.document-revision-comparison/2'
      && schema !== 'chipmunk.document-revision-comparison/3' && schema !== 'chipmunk.document-revision-comparison/4'
      && schema !== 'chipmunk.document-revision-comparison/5')
      || !Array.isArray(result.comparison.changes)
      || ((schema === 'chipmunk.document-revision-comparison/4' || schema === 'chipmunk.document-revision-comparison/5')
        && !result.comparison.word_tables)) {
    throw new Error('The server returned an incomplete draft comparison.')
  }
  return result
}
