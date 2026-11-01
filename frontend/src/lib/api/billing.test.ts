// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'
import { BillingError, billingApi, checkoutAddress, hasBillingIdentity } from './billing'

vi.mock('../hash', () => ({ sha256Blob: vi.fn(async () => 'a'.repeat(64)) }))
afterEach(() => { vi.unstubAllGlobals(); localStorage.clear() })

describe('payment client boundaries', () => {
  it('only opens the exact Stripe HTTPS checkout host', () => {
    expect(checkoutAddress('https://checkout.stripe.com/c/pay/test_123')).toBe('https://checkout.stripe.com/c/pay/test_123')
    for (const value of ['javascript:alert(1)', 'http://checkout.stripe.com/pay', 'https://checkout.stripe.com.attacker.test/pay', 'https://checkout.stripe.com@attacker.test/pay', 'https://evil@checkout.stripe.com/pay', 'https://checkout.stripe.com:4430/pay', '/billing/return']) {
      expect(() => checkoutAddress(value)).toThrow()
    }
  })
  it('does not treat the shared local account or download token as a payment identity', () => {
    expect(hasBillingIdentity({ auth: 'single_user', unauthenticated_instance: true })).toBe(false)
    expect(hasBillingIdentity({ auth: 'token' })).toBe(false)
    expect(hasBillingIdentity(null)).toBe(false)
    expect(hasBillingIdentity({ auth: 'session', unauthenticated_instance: false })).toBe(true)
    expect(hasBillingIdentity({ auth: 'bootstrap', unauthenticated_instance: false })).toBe(true)
  })
  it('sends source integrity and retry identity, leaving price entirely on the server', async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({ order_id: 'order', checkout_url: 'https://checkout.stripe.com/c/pay/test', status: 'awaiting_payment' }), { status: 200 }))
    vi.stubGlobal('fetch', fetcher)
    localStorage.setItem('veritas_api_key', 'fixture-key')
    await billingApi.checkout('document', new File(['Example'], 'source.txt'), 'full', 'same-attempt')
    const [path, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit]
    expect(path).toBe('/api/v2/billing/checkout')
    expect(init.headers).toMatchObject({ 'Idempotency-Key': 'same-attempt', 'X-API-Key': 'fixture-key' })
    const body = init.body as FormData
    expect(body.get('client_sha256')).toBe('a'.repeat(64))
    expect(body.get('kind')).toBe('document')
    expect(body.has('amount_minor')).toBe(false)
    expect(body.has('currency')).toBe(false)
  })
  it('keeps the saved order reference after a checkout provider error', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail: 'Stripe temporarily unavailable', order_id: 'saved-order' }), { status: 503 })))
    try { await billingApi.resume('saved-order'); expect.fail('Expected payment failure') } catch (error) {
      expect(error).toBeInstanceOf(BillingError)
      expect((error as BillingError).orderId).toBe('saved-order')
    }
  })
})
