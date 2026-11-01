// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import BillingReturn from './BillingReturn'
import { billingApi, openCheckout, type BillingOrder } from '../lib/api/billing'

vi.mock('../lib/api/billing', async (original) => ({
  ...await original<object>(), billingApi: { order: vi.fn(), resume: vi.fn() }, openCheckout: vi.fn(),
}))
const id = '11111111-1111-4111-8111-111111111111'
const base: BillingOrder = { order_id: id, status: 'awaiting_payment', analysis_id: null, analysis_status: null, amount_minor: 999, currency: 'usd' }
const clients: QueryClient[] = []
function mount(search = '') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={[`/billing/return?order_id=${id}${search}`]}><BillingReturn /></MemoryRouter></QueryClientProvider>)
}
beforeEach(() => vi.mocked(billingApi.order).mockResolvedValue(base))
afterEach(() => { cleanup(); clients.forEach((c) => c.clear()); clients.length = 0; vi.resetAllMocks() })

describe('Stripe return page', () => {
  it('does not treat a success-looking return query as proof of payment', async () => {
    mount('&paid=true&session_id=cs_test_claim')
    await screen.findByText(/We are waiting for Stripe to confirm payment/)
    expect(screen.queryByRole('heading', { name: 'Payment confirmed' })).toBeNull()
    expect(screen.queryByRole('link', { name: 'View analysis' })).toBeNull()
    expect(billingApi.resume).not.toHaveBeenCalled()
  })
  it('uses the verified order even if the user returned through the cancel URL', async () => {
    vi.mocked(billingApi.order).mockResolvedValue({ ...base, status: 'fulfilled', analysis_id: 'analysis-1', analysis_status: 'done' })
    mount('&cancelled=1')
    await screen.findByRole('heading', { name: 'Payment confirmed' })
    expect(screen.getByRole('link', { name: 'View analysis' }).getAttribute('href')).toBe('/analyses/analysis-1')
    expect(screen.queryByRole('button', { name: 'Return to checkout' })).toBeNull()
  })
  it('reuses the saved order and ignores duplicate resume clicks', async () => {
    let finish!: (result: { order_id: string; checkout_url: string; status: 'awaiting_payment' }) => void
    vi.mocked(billingApi.resume).mockImplementation(() => new Promise((resolve) => { finish = resolve }))
    mount('&cancelled=1')
    const button = await screen.findByRole('button', { name: 'Return to checkout' })
    fireEvent.click(button); fireEvent.click(button)
    await waitFor(() => expect(billingApi.resume).toHaveBeenCalledTimes(1))
    expect(billingApi.resume).toHaveBeenCalledWith(id)
    finish({ order_id: id, checkout_url: 'https://checkout.stripe.com/c/pay/test', status: 'awaiting_payment' })
    await waitFor(() => expect(openCheckout).toHaveBeenCalledOnce())
  })
  it('offers support for a paid analysis failure without another checkout', async () => {
    vi.mocked(billingApi.order).mockResolvedValue({ ...base, status: 'fulfilled', analysis_id: 'analysis-1', analysis_status: 'failed' })
    mount()
    await screen.findByText(/Payment was confirmed, but the analysis could not finish/)
    expect(screen.getByRole('link', { name: 'sales@chipmunkrpa.com' }).getAttribute('href')).toContain(id)
    expect(screen.queryByRole('button', { name: 'Return to checkout' })).toBeNull()
  })
  it('keeps payment status uncertain after a failed order lookup', async () => {
    vi.mocked(billingApi.order).mockRejectedValue(new Error('Unable to read order'))
    mount()
    await screen.findByRole('alert')
    expect(screen.getByText(/check this order again before starting another checkout/)).toBeTruthy()
    expect(screen.queryByRole('heading', { name: 'Payment confirmed' })).toBeNull()
  })
  it('does not resume an asynchronously failed payment or imply an analysis started', async () => {
    vi.mocked(billingApi.order).mockResolvedValue({ ...base, status: 'payment_failed' })
    mount()
    await screen.findByRole('heading', { name: 'Payment did not complete' })
    expect(screen.getByText(/No analysis was started for this order/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Return to checkout' })).toBeNull()
    expect(screen.getByRole('link', { name: 'New analysis' }).getAttribute('href')).toBe('/upload')
  })
})
