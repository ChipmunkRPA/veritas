import { apiClient, unwrap } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'

export type ArchiveComparison = components['schemas']['DocumentArchiveResponse']
export type ArchiveExcerpt = components['schemas']['DocumentArchiveExcerpt']

const route = '/api/v2/analyses/{aid}/document-archive-comparison'

export async function archiveAvailability(analysisId: string, signal: AbortSignal) {
  const result = await unwrap(apiClient.GET(route, { params: { path: { aid: analysisId } }, signal }))
  return result.available
}

export async function compareArchive(analysisId: string, signal: AbortSignal): Promise<ArchiveComparison> {
  const result = await unwrap(apiClient.POST(route, {
    params: { path: { aid: analysisId }, query: { include_pdf: true } }, signal,
  }))
  if (result?.comparison?.schema !== 'chipmunk.document-archive-comparison/1') {
    throw new Error('The server returned an incomplete archive comparison.')
  }
  return result as ArchiveComparison
}
