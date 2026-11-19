// @vitest-environment jsdom
import { readFileSync } from 'node:fs'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import Upload from './Upload'
import { billingApi, openCheckout } from '../lib/api/billing'
import { api } from '../lib/api'
import { intakeItem } from '../lib/api/intake'
import { submitBatch } from '../lib/api/batches'

const fixtures = vi.hoisted(() => ({
  billing: { data: { purchase_notice: null as unknown, enabled: true, mode: 'test', ready: true, currency: 'usd', amount_minor: 999, price_display: 'US$9.99', requires_sign_in: true }, isPending: false, isError: false, refetch: vi.fn() },
  me: { auth: 'session', unauthenticated_instance: false },
}))
vi.mock('../lib/api/billing', async (original) => ({
  ...await original<object>(), useBilling: () => fixtures.billing,
  billingApi: { checkout: vi.fn() }, openCheckout: vi.fn(),
}))
vi.mock('../lib/api', async (original) => {
  const actual = await original<typeof import('../lib/api')>()
  return { ...actual, api: { ...actual.api, text: vi.fn(), uploadWithProgress: vi.fn() } }
})
vi.mock('../lib/api/intake', async (original) => ({ ...await original<object>(), intakeItem: vi.fn() }))
vi.mock('../lib/api/batches', async (original) => ({ ...await original<object>(), submitBatch: vi.fn() }))
vi.mock('../lib/api/queries', async (original) => ({
  ...await original<object>(),
  useCases: () => ({ data: [] }), useMe: () => ({ data: fixtures.me }),
  useModelCards: () => ({ data: [], isPending: false, isError: false }),
}))
vi.mock('../features/intake/IntakeChecks', () => ({ default: () => null }))
vi.mock('../features/cases/NewCaseDialog', () => ({ default: () => null }))
const source = 'This is a longer sample document with enough words to submit for automated authenticity analysis and evidence.'
const clients: QueryClient[] = []
function mount(path = '/upload?kind=document') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[path]}><Upload /></MemoryRouter></QueryClientProvider>)
}
beforeEach(() => {
  fixtures.billing.data.purchase_notice = JSON.parse(readFileSync('../backend/tests/fixtures/purchase_notice.json', 'utf8'))
  fixtures.billing.isPending = false; fixtures.billing.isError = false; fixtures.billing.data.ready = true; fixtures.billing.data.enabled = true
  fixtures.me.auth = 'session'; fixtures.me.unauthenticated_instance = false
})
afterEach(() => { cleanup(); clients.forEach((c) => c.clear()); clients.length = 0; vi.resetAllMocks() })

describe('paid analysis intake', () => {
  it.each([
    ['recording.WAVE', '.wave'], ['recording.oga', '.oga'], ['audiobook.M4B', '.m4b'],
  ])('accepts the supported %s alias without a browser MIME type before checkout', async (name, suffix) => {
    const file = new File(['recording'], name)
    const { container } = mount('/upload?kind=audio')
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!
    expect(file.type).toBe('')
    expect(input.accept.split(',')).toContain(suffix)
    fireEvent.change(input, { target: { files: [file] } })
    expect(await screen.findByText(name)).toBeTruthy()
    expect(screen.queryByRole('alert')).toBeNull()
    const button = screen.getByRole('button', { name: /Continue to checkout/ }) as HTMLButtonElement
    expect(button.disabled).toBe(true)
    expect(billingApi.checkout).not.toHaveBeenCalled()
    vi.mocked(billingApi.checkout).mockResolvedValue({ order_id: 'saved-order', checkout_url: 'https://checkout.stripe.com/c/pay/test', status: 'awaiting_payment' })
    fireEvent.click(screen.getByRole('checkbox', { name: /I understand and accept/ }))
    fireEvent.click(button)
    await waitFor(() => expect(billingApi.checkout).toHaveBeenCalledTimes(1))
    expect(vi.mocked(billingApi.checkout).mock.calls[0].slice(0, 2)).toEqual(['audio', file])
    expect(api.uploadWithProgress).not.toHaveBeenCalled()
    expect(intakeItem).not.toHaveBeenCalled()
    expect(submitBatch).not.toHaveBeenCalled()
  })

  it.each(['review.DOCX', 'article.pdf', 'deck.pptx', 'book.xlsx', 'message.eml'])('routes a dropped %s to Document before checkout', async (name) => {
    const file = new File(['sample'], name)
    mount('/upload')
    expect(screen.getByRole('tab', { name: 'Image' }).getAttribute('aria-selected')).toBe('true')
    fireEvent.drop(screen.getByTestId('drop-zone'), { dataTransfer: { files: [file] } })
    await screen.findByText(name)
    expect(screen.getByRole('tab', { name: 'Document' }).getAttribute('aria-selected')).toBe('true')
    expect((screen.getByRole('button', { name: /Continue to checkout/ }) as HTMLButtonElement).disabled).toBe(true)
    expect(billingApi.checkout).not.toHaveBeenCalled()
    vi.mocked(billingApi.checkout).mockResolvedValue({ order_id: 'saved-order', checkout_url: 'https://checkout.stripe.com/c/pay/test', status: 'awaiting_payment' })
    fireEvent.click(screen.getByRole('checkbox', { name: /I understand and accept/ }))
    fireEvent.click(screen.getByRole('button', { name: /Continue to checkout/ }))
    await waitFor(() => expect(billingApi.checkout).toHaveBeenCalledTimes(1))
    expect(vi.mocked(billingApi.checkout).mock.calls[0].slice(0, 2)).toEqual(['document', file])
  })

  it('shows the conversion guidance when an older Word file is dropped on a media tab', async () => {
    mount('/upload?kind=video')
    fireEvent.drop(screen.getByTestId('drop-zone'), { dataTransfer: { files: [new File(['older Word'], 'draft.doc')] } })
    expect((await screen.findByRole('alert')).textContent).toContain('Save the older Word file as DOCX or PDF')
    expect(screen.getByRole('tab', { name: 'Document' }).getAttribute('aria-selected')).toBe('true')
    expect(billingApi.checkout).not.toHaveBeenCalled()
  })

  it.each([
    ['draft.GDOC', 'Google Docs document as Word (DOCX)'],
    ['notes.gsheet', 'Google Sheets spreadsheet as Excel (XLSX)'],
    ['brief.gslides', 'Google Slides presentation as PowerPoint (PPTX)'],
    ['book.numbers', 'Apple Numbers spreadsheet as Excel (XLSX)'],
  ])('shows the export format for %s before checkout', async (filename, guidance) => {
    mount('/upload?kind=video')
    fireEvent.drop(screen.getByTestId('drop-zone'), { dataTransfer: { files: [new File(['export needed'], filename)] } })
    expect((await screen.findByRole('alert')).textContent).toContain(guidance)
    expect(screen.getByRole('tab', { name: 'Document' }).getAttribute('aria-selected')).toBe('true')
    expect(screen.getByRole('button', { name: /Continue to checkout/ }).hasAttribute('disabled')).toBe(true)
    expect(billingApi.checkout).not.toHaveBeenCalled()
  })

  it('accepts Word and saved XHTML, and guides users who choose an older Word file', () => {
    const { container } = mount()
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!
    expect(input.accept.split(',')).toEqual(expect.arrayContaining(['.docx', '.pdf', '.odt', '.pptx', '.odp', '.xlsx', '.ods', '.csv', '.tsv', '.epub', '.rtf', '.txt', '.md', '.html', '.xhtml', '.eml']))
    fireEvent.change(input, { target: { files: [new File(['old Word'], 'draft.DOC')] } })
    expect(screen.getByRole('alert').textContent).toContain('Save the older Word file as DOCX or PDF')
    fireEvent.change(input, { target: { files: [new File(['Word'], 'draft.DOCX')] } })
    expect(screen.getByText(/Word checks body paragraphs.*simple text boxes/)).toBeTruthy()
    const replacement = container.querySelector<HTMLInputElement>('input[type="file"]')!
    fireEvent.change(replacement, { target: { files: [new File(['saved article'], 'article.XHTML', { type: 'application/xhtml+xml' })] } })
    expect(screen.getByText('article.XHTML')).toBeTruthy()
    expect(screen.queryByText(/Word checks body paragraphs/)).toBeNull()
    expect(screen.queryByRole('alert')).toBeNull()
    expect(billingApi.checkout).not.toHaveBeenCalled()
  })
  it('explains the scope of Excel writing checks and how to upload older spreadsheets', () => {
    const { container } = mount()
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!
    fireEvent.change(input, { target: { files: [new File(['old workbook'], 'notes.xls')] } })
    expect(screen.getByRole('alert').textContent).toContain('macro-free XLSX, CSV or PDF')
    fireEvent.change(input, { target: { files: [new File(['workbook'], 'notes.XLSX')] } })
    expect(screen.queryByRole('alert')).toBeNull()
    expect(screen.getByText(/Excel checks saved writing on visible sheets, rows and columns/)).toBeTruthy()
  })
  it('shows one coverage notice per format in a document batch', () => {
    fixtures.billing.data.enabled = false
    const { container } = mount()
    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!
    fireEvent.change(input, { target: { files: [
      new File(['Word'], 'first.docx'), new File(['Word'], 'second.DOCX'), new File(['Slides'], 'brief.pptx'),
    ] } })
    const guidance = screen.getByText('What the document checks cover').parentElement!
    expect(guidance.querySelectorAll('li')).toHaveLength(2)
    expect(guidance.textContent).toContain('simple text boxes')
    expect(guidance.textContent).toContain('Hidden groups are excluded')
  })
  it('requires an unchecked limitations acknowledgement and resets it for a changed source', () => {
    mount()
    const sourceBox = screen.getByRole('textbox', { name: 'Text to analyse' })
    fireEvent.change(sourceBox, { target: { value: source } })
    const checkbox = screen.getByRole('checkbox', { name: /I understand and accept/ }) as HTMLInputElement
    const button = screen.getByRole('button', { name: /Continue to checkout/ }) as HTMLButtonElement
    expect(checkbox.checked).toBe(false)
    expect(button.disabled).toBe(true)
    expect(screen.getByRole('region', { name: 'Analysis limitations before payment' }).textContent).toContain('can miss')
    expect(screen.getByRole('region', { name: 'Analysis limitations before payment' }).textContent).toContain('no reliable conclusion')
    expect(screen.getByRole('region', { name: 'Analysis limitations before payment' }).textContent).toContain('without a calibrated whole-article authorship conclusion')
    expect(screen.getByRole('region', { name: 'Analysis limitations before payment' }).textContent).toContain('Automated writing checks currently cover English')
    fireEvent.click(button)
    expect(billingApi.checkout).not.toHaveBeenCalled()
    fireEvent.click(checkbox)
    expect(button.disabled).toBe(false)
    fireEvent.change(sourceBox, { target: { value: source + ' This is a changed source.' } })
    expect(checkbox.checked).toBe(false)
    expect(button.disabled).toBe(true)
  })
  it('shows the price and requires an authenticated identity before checkout', () => {
    fixtures.me.auth = 'single_user'; fixtures.me.unauthenticated_instance = true
    mount()
    fireEvent.change(screen.getByRole('textbox', { name: 'Text to analyse' }), { target: { value: source } })
    expect(screen.getByRole('region', { name: 'Analysis price' }).textContent).toContain('US$9.99')
    expect(screen.getByRole('region', { name: 'Analysis price' }).textContent).toContain('Payment is required before analysis.')
    expect(screen.getByRole('link', { name: 'Sign in to continue' }).getAttribute('href')).toContain('/sign-in?next=')
    expect((screen.getByRole('button', { name: /Continue to checkout/ }) as HTMLButtonElement).disabled).toBe(true)
    expect(billingApi.checkout).not.toHaveBeenCalled()
    expect(screen.queryByText('Case & custody')).toBeNull()
  })
  it('blocks submission when current pricing or Stripe readiness is unavailable', () => {
    fixtures.billing.data.ready = false
    const view = mount()
    fireEvent.change(screen.getByRole('textbox', { name: 'Text to analyse' }), { target: { value: source } })
    expect((screen.getByRole('button', { name: /Continue to checkout/ }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText('Checkout is not available yet. Please try again later.')).toBeTruthy()
    view.unmount()
    fixtures.billing.data.ready = true; fixtures.billing.isError = true
    mount()
    expect(screen.getByRole('alert').textContent).toContain('Pricing could not be loaded')
    expect((screen.getByRole('button', { name: /Continue to checkout/ }) as HTMLButtonElement).disabled).toBe(true)
  })
  it('makes one checkout for duplicate clicks, then reuses the same key after a network error', async () => {
    let rejectFirst!: (error: Error) => void
    vi.mocked(billingApi.checkout).mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectFirst = reject }))
    vi.mocked(billingApi.checkout).mockResolvedValue({ order_id: 'saved-order', checkout_url: 'https://checkout.stripe.com/c/pay/test', status: 'awaiting_payment' })
    mount()
    fireEvent.change(screen.getByRole('textbox', { name: 'Text to analyse' }), { target: { value: source } })
    fireEvent.click(screen.getByRole('checkbox', { name: /I understand and accept/ }))
    const first = screen.getByRole('button', { name: /Continue to checkout/ })
    fireEvent.click(first); fireEvent.click(first)
    await waitFor(() => expect(billingApi.checkout).toHaveBeenCalledTimes(1))
    const originalKey = vi.mocked(billingApi.checkout).mock.calls[0][3]
    rejectFirst(new Error('Network connection interrupted'))
    await screen.findByText('Network connection interrupted')
    fireEvent.click(screen.getByRole('button', { name: /Continue to checkout/ }))
    await waitFor(() => expect(openCheckout).toHaveBeenCalledOnce())
    expect(billingApi.checkout).toHaveBeenCalledTimes(2)
    expect(vi.mocked(billingApi.checkout).mock.calls[1][3]).toBe(originalKey)
    expect(vi.mocked(billingApi.checkout).mock.calls[1][0]).toBe('document')
    expect(vi.mocked(billingApi.checkout).mock.calls[1][2]).toBe('full')
    expect(vi.mocked(billingApi.checkout).mock.calls[1][4]).toMatchObject({ risk_accepted: true, risk_notice_version: 'purchase-risk/v3' })
    expect(api.text).not.toHaveBeenCalled()
    expect(api.uploadWithProgress).not.toHaveBeenCalled()
    expect(intakeItem).not.toHaveBeenCalled()
    expect(submitBatch).not.toHaveBeenCalled()
  })
})
