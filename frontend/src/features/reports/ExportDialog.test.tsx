// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { AnalysisDetail, Me } from '../../lib/api/queries'
import type { Report, ReportPackage } from '../../lib/api/reports'
import ExportDialog from './ExportDialog'

const mocked = vi.hoisted(() => ({ generateReport: vi.fn(), generatePackage: vi.fn(), downloadPackage: vi.fn(), downloadReport: vi.fn(),
  reports: [] as Report[], key: vi.fn() }))
vi.mock('../../lib/api/reports', async (original) => ({ ...(await original<typeof import('../../lib/api/reports')>()),
  generateReport: mocked.generateReport, generatePackage: mocked.generatePackage, downloadPackage: mocked.downloadPackage, downloadReport: mocked.downloadReport,
  useReports: () => ({ data: mocked.reports, isPending: false }), useInstanceKey: () => ({ data: { key_id: 'ed25519:trusted', public_key_pem: 'public key' } }),
  saveInstanceKey: mocked.key,
}))
vi.mock('../review/ReviewStateLine', () => ({ default: () => <span>Recorded review state</span> }))

const report = (extra: Partial<Report> = {}): Report => ({ id: 'report-1', display_id: 'R-1-DRAFT', profile: 'forensic', status: 'done',
  downloadable: true, template_set: 'current-template-unrelated-to-r48', requested_at: '2026-10-31T12:00:00Z', signing_key_id: 'ed25519:trusted', page_count: 8,
  ...extra }) as Report
const bundle = (extra: Partial<ReportPackage> = {}): ReportPackage => ({ id: 'package-1', schema_id: 'chipmunk.public-evidence/1', status: 'done',
  include_original: false, file_count: 12, kind: 'draft', ...extra }) as ReportPackage
const analysis = (state = 'machine', valid = true): AnalysisDetail => ({ id: 'analysis-1', status: 'done', review: {
  state, requires_second_review: false, signatures_valid: valid,
} }) as AnalysisDetail
const examiner = { role: 'examiner' } as Me
const show = (value = analysis(), selfService = false, me = examiner) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const draw = (open: boolean) => <QueryClientProvider client={client}><ExportDialog open={open} analysis={value} me={me} selfService={selfService} onClose={vi.fn()} /></QueryClientProvider>
  return { ...render(draw(true)), draw }
}
beforeEach(() => {
  vi.clearAllMocks()
  mocked.reports = []
  mocked.generateReport.mockResolvedValue(report())
  mocked.generatePackage.mockResolvedValue(bundle())
  mocked.downloadPackage.mockResolvedValue(undefined)
  mocked.downloadReport.mockResolvedValue(undefined)
  HTMLDialogElement.prototype.showModal = function () { this.setAttribute('open', '') }
  HTMLDialogElement.prototype.close = function () { this.removeAttribute('open') }
})
afterEach(cleanup)

describe('self-service exports', () => {
  it.each(['PDF', 'ZIP'])('regenerates retired r68 reports through the existing %s action', async (format) => {
    const old = report({ id: 'old-report', display_id: 'R-old', template_set: 'veritas-report-2.0.r68', downloadable: false })
    const current = report({ id: 'new-report', display_id: 'R-new', template_set: 'veritas-report-2.0.r69', downloadable: true })
    mocked.reports = [old]
    mocked.generateReport.mockResolvedValue(current)
    show(analysis(), true)
    expect(screen.getByText('Generate a new report')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Download R-old' })).toBeNull()
    if (format === 'ZIP') fireEvent.click(screen.getByRole('radio', { name: /Evidence ZIP/ }))
    fireEvent.click(screen.getByRole('button', { name: format === 'ZIP' ? 'Download evidence ZIP' : 'Download report' }))
    await waitFor(() => expect(mocked.generateReport).toHaveBeenCalledWith('analysis-1', 'forensic', expect.any(Function), format === 'ZIP'))
    if (format === 'ZIP') await waitFor(() => expect(mocked.generatePackage).toHaveBeenCalledWith('analysis-1', 'new-report', 'draft', false, expect.any(Function)))
    else await waitFor(() => expect(mocked.downloadReport).toHaveBeenCalledWith(current))
    expect(mocked.downloadReport).not.toHaveBeenCalledWith(old)
    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('downloads a report without showing the examination workflow or signed column', async () => {
    mocked.reports = [report()]
    show(analysis(), true)
    expect(screen.queryByTestId('export-review-line')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Issue' })).toBeNull()
    expect(screen.queryByRole('columnheader', { name: 'Signed' })).toBeNull()
    expect(screen.getByRole('dialog').textContent).not.toMatch(/examiner|sign-off|signed off|declaration and signature/i)
    fireEvent.click(screen.getByRole('button', { name: 'Download report' }))
    await waitFor(() => expect(mocked.downloadReport).toHaveBeenCalledWith(report()))
    expect(mocked.generateReport).toHaveBeenCalledWith('analysis-1', 'forensic', expect.any(Function), false)
    expect(mocked.generatePackage).not.toHaveBeenCalled()
  })

  it.each([['machine', true], ['approved', true], ['disputed', true], ['approved', false]] as const)(
    'uses draft package mode for %s results with valid signatures %s', async (state, valid) => {
      show(analysis(state, valid), true)
      fireEvent.click(screen.getByRole('radio', { name: /Evidence ZIP/ }))
      expect(screen.queryByRole('button', { name: 'Issue' })).toBeNull()
      fireEvent.click(screen.getByRole('button', { name: 'Download evidence ZIP' }))
      await waitFor(() => expect(mocked.downloadPackage).toHaveBeenCalledWith(bundle()))
      expect(mocked.generatePackage).toHaveBeenCalledWith('analysis-1', 'report-1', 'draft', false, expect.any(Function))
    },
  )

  it.each(['superseded', 'withdrawn'])('keeps the export refusal for %s results', (state) => {
    show(analysis(state), true)
    expect((screen.getByRole('button', { name: 'Download report' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByRole('alert')).toBeTruthy()
    expect(mocked.generateReport).not.toHaveBeenCalled()
  })

  it('preserves account permissions using customer-facing wording', () => {
    show(analysis(), true, { role: 'viewer' } as Me)
    expect((screen.getByRole('button', { name: 'Download report' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByRole('alert').textContent).toBe('Your account cannot create new exports. Available reports can be downloaded below.')
    expect(mocked.generateReport).not.toHaveBeenCalled()
  })
})

describe('public evidence ZIP export', () => {
  it('exports a draft ZIP with the original excluded by default and reuses a current forensic report', async () => {
    show()
    fireEvent.click(screen.getByRole('radio', { name: /Evidence ZIP/ }))
    expect((screen.getByRole('checkbox', { name: 'Include original file (unredacted)' }) as HTMLInputElement).checked).toBe(false)
    expect((screen.getByRole('button', { name: 'Issue' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Download draft' }))
    await waitFor(() => expect(mocked.downloadPackage).toHaveBeenCalledWith(bundle()))
    expect(mocked.generateReport).toHaveBeenCalledWith('analysis-1', 'forensic', expect.any(Function), true)
    expect(mocked.generatePackage).toHaveBeenCalledWith('analysis-1', 'report-1', 'draft', false, expect.any(Function))
    expect(mocked.downloadReport).not.toHaveBeenCalled()
    expect(screen.getByText(/Downloaded evidence ZIP \(12 files\). Original file not included/)).toBeTruthy()
  })

  it('requires an explicit original selection and sends the issue kind only after review', async () => {
    mocked.generatePackage.mockResolvedValue(bundle({ include_original: true, kind: 'issue' }))
    show(analysis('approved'))
    fireEvent.click(screen.getByRole('radio', { name: /Evidence ZIP/ }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include original file (unredacted)' }))
    fireEvent.click(screen.getByRole('button', { name: 'Issue' }))
    await waitFor(() => expect(mocked.generatePackage).toHaveBeenCalledWith('analysis-1', 'report-1', 'issue', true, expect.any(Function)))
    await waitFor(() => expect(screen.getByText(/Original file included/)).toBeTruthy())
  })

  it('resets original inclusion whenever the dialog is opened again', () => {
    const { rerender, draw } = show()
    fireEvent.click(screen.getByRole('radio', { name: /Evidence ZIP/ }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Include original file (unredacted)' }))
    rerender(draw(false))
    rerender(draw(true))
    expect((screen.getByRole('checkbox', { name: 'Include original file (unredacted)' }) as HTMLInputElement).checked).toBe(false)
  })

  it('obeys the server download flag rather than a hardcoded template generation', async () => {
    const current = report({ id: 'new-report', display_id: 'R-new', downloadable: true })
    mocked.reports = [current, report({ id: 'old-report', display_id: 'R-old', template_set: 'veritas-report-2.0.r48', downloadable: false })]
    show()
    fireEvent.click(screen.getByRole('button', { name: 'Download R-new' }))
    await waitFor(() => expect(mocked.downloadReport).toHaveBeenCalledWith(current))
    expect(screen.queryByRole('button', { name: 'Download R-old' })).toBeNull()
    expect(screen.getByText('Generate a new report')).toBeTruthy()
  })

  it('keeps disputed or invalid review records draft-only and leaves PDF exports working', async () => {
    show(analysis('disputed', false))
    expect((screen.getByRole('button', { name: 'Issue' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Download draft' }))
    await waitFor(() => expect(mocked.downloadReport).toHaveBeenCalledWith(report()))
    expect(mocked.generateReport).toHaveBeenCalledWith('analysis-1', 'forensic', expect.any(Function), false)
    expect(mocked.generatePackage).not.toHaveBeenCalled()
  })

  it('shows a package failure without downloading a replacement or dropping the selected settings', async () => {
    mocked.generatePackage.mockRejectedValue(new Error('Evidence assets are unavailable.'))
    show()
    fireEvent.click(screen.getByRole('radio', { name: /Evidence ZIP/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Download draft' }))
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Evidence assets are unavailable.'))
    expect(mocked.downloadPackage).not.toHaveBeenCalled()
    expect(mocked.downloadReport).not.toHaveBeenCalled()
    expect((screen.getByRole('checkbox', { name: 'Include original file (unredacted)' }) as HTMLInputElement).checked).toBe(false)
  })
})
