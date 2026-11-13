import { apiClient, unwrap } from '../../lib/api/client'
import type { components } from '../../lib/api/schema'

export type ArchiveComparison = components['schemas']['DocumentArchiveResponse']
export type ArchiveExcerpt = components['schemas']['DocumentArchiveExcerpt']
export type ArchiveCollection = 'frus' | 'gao'

const route = '/api/v2/analyses/{aid}/document-archive-comparison'

export async function archiveAvailability(analysisId: string, signal: AbortSignal) {
  const result = await unwrap(apiClient.GET(route, { params: { path: { aid: analysisId } }, signal }))
  return result.available ? { gaoAvailable: result.gao_available === true } : null
}

export async function compareArchive(analysisId: string, signal: AbortSignal, collection: ArchiveCollection = 'frus'): Promise<ArchiveComparison> {
  const result = await unwrap(apiClient.POST(route, {
    params: { path: { aid: analysisId }, query: { include_pdf: true, collection } }, signal,
  }))
  if (!result?.comparison || !['chipmunk.document-archive-comparison/1', 'chipmunk.document-archive-comparison/2'].includes(result.comparison.schema)) {
    throw new Error('The server returned an incomplete archive comparison.')
  }
  return result as ArchiveComparison
}
