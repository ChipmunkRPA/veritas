// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiClient } from '../../lib/api/client'
import { downloadPackage, downloadReport, generatePackage, reportFilename, type Report, type ReportPackage } from '../../lib/api/reports'

afterEach(() => vi.restoreAllMocks())
const publicPackage = { id: 'zip-1', status: 'done', error: null, schema_id: 'chipmunk.public-evidence/1', include_original: false } as ReportPackage
const response = (data: unknown) => ({ data, response: new Response(null, { status: 200 }) })

describe('public export API contracts', () => {
  it('defaults to excluding the source and verifies the returned package schema', async () => {
    const post = vi.spyOn(apiClient, 'POST').mockResolvedValue(response(publicPackage) as never)
    const get = vi.spyOn(apiClient, 'GET').mockResolvedValue(response(publicPackage) as never)
    expect(await generatePackage('analysis-1', 'report-1', 'draft')).toBe(publicPackage)
    expect(post).toHaveBeenCalledWith('/api/v2/analyses/{aid}/packages', {
      params: { path: { aid: 'analysis-1' } }, body: { report_id: 'report-1', kind: 'draft', include_original: false },
    })
    expect(get).toHaveBeenCalledWith('/api/v2/packages/{package_id}', { params: { path: { package_id: 'zip-1' } } })
  })
  it('refuses an internal or unknown package schema before polling or downloading', async () => {
    const internal = { ...publicPackage, schema_id: 'veritas.evidence/2.0' }
    vi.spyOn(apiClient, 'POST').mockResolvedValue(response(internal) as never)
    const get = vi.spyOn(apiClient, 'GET')
    const fetch = vi.spyOn(globalThis, 'fetch')
    await expect(generatePackage('analysis-1', 'report-1', 'draft')).rejects.toThrow('not a supported public evidence ZIP')
    await expect(downloadPackage(internal)).rejects.toThrow('not a supported public evidence ZIP')
    expect(get).not.toHaveBeenCalled()
    expect(fetch).not.toHaveBeenCalled()
  })
  it('does not request unavailable report bytes and uses Chipmunk filenames', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch')
    await expect(downloadReport({ id: 'old', downloadable: false } as Report)).rejects.toThrow('not available for download')
    expect(fetch).not.toHaveBeenCalled()
    expect(reportFilename({ display_id: 'R-1' })).toBe('chipmunk-verify-R-1.pdf')
  })
})
