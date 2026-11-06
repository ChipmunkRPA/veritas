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
function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/upload?kind=document']}><Upload /></MemoryRouter></QueryClientProvider>)
}
beforeEach(() => {
  fixtures.billing.data.purchase_notice = JSON.parse(readFileSync('../backend/tests/fixtures/purchase_notice.json', 'utf8'))
  fixtures.billing.isPending = false; fixtures.billing.isError = false; fixtures.billing.data.ready = true
  fixtures.me.auth = 'session'; fixtures.me.unauthenticated_instance = false
})
afterEach(() => { cleanup(); clients.forEach((c) => c.clear()); clients.length = 0; vi.resetAllMocks() })

describe('paid analysis intake', () => {
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
    expect(vi.mocked(billingApi.checkout).mock.calls[1][4]).toMatchObject({ risk_accepted: true, risk_notice_version: 'purchase-risk/v1' })
    expect(api.text).not.toHaveBeenCalled()
    expect(api.uploadWithProgress).not.toHaveBeenCalled()
    expect(intakeItem).not.toHaveBeenCalled()
    expect(submitBatch).not.toHaveBeenCalled()
  })
})
