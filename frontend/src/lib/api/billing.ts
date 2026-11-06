import { useQuery } from '@tanstack/react-query'
import type { Kind } from '../api'
import { sha256Blob } from '../hash'
import { ApiError, authHeaders } from './client'
import { refusedSession } from './sessionEnd'
import type { AnalysisProfile } from './intake'

export interface PurchaseNotice {
  version: string
  sha256: string
  title: string
  points: string[]
  acknowledgement_label: string
  document_notice: string
  stripe_message: string
}

export interface PurchaseAcceptance {
  risk_accepted: true
  risk_notice_version: string
  risk_notice_sha256: string
}

export function purchaseAcceptance(notice: PurchaseNotice): PurchaseAcceptance {
  return { risk_accepted: true, risk_notice_version: notice.version, risk_notice_sha256: notice.sha256 }
}

export interface BillingConfig {
  enabled: boolean
  mode: 'disabled' | 'test' | 'live'
  ready: boolean
  currency: string
  amount_minor: number
  price_display: string
  requires_sign_in: boolean
  purchase_notice: PurchaseNotice
}

export interface BillingOrder {
  order_id: string
  status: 'pending' | 'awaiting_payment' | 'fulfilled' | 'expired' | 'payment_failed'
  analysis_id: string | null
  analysis_status: string | null
  amount_minor: number
  currency: string
  kind: Kind
  purchase_acknowledged: boolean
  purchase_notice: PurchaseNotice
}

export interface CheckoutResult {
  order_id: string
  checkout_url: string
  status: BillingOrder['status']
}

export class BillingError extends ApiError {
  constructor(status: number, title: string, detail: string, readonly orderId?: string) {
    super(status, title, detail)
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/v2/billing${path}`, {
    ...init, headers: { ...authHeaders(init.method), ...init.headers },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = typeof body.detail === 'string' ? body.detail : 'The payment service could not complete this request.'
    throw new BillingError(response.status, typeof body.title === 'string' ? body.title : 'Payment request failed',
      response.status === 401 ? refusedSession(detail) : detail, typeof body.order_id === 'string' ? body.order_id : undefined)
  }
  return body as T
}

export const billingApi = {
  config: (): Promise<BillingConfig> => request('/config'),
  checkout: async (kind: Kind, file: File, profile: AnalysisProfile, key: string, acceptance: PurchaseAcceptance): Promise<CheckoutResult> => {
    const body = new FormData()
    body.append('file', file)
    body.append('kind', kind)
    body.append('profile', profile)
    body.append('risk_accepted', String(acceptance.risk_accepted))
    body.append('risk_notice_version', acceptance.risk_notice_version)
    body.append('risk_notice_sha256', acceptance.risk_notice_sha256)
    body.append('client_sha256', await sha256Blob(file))
    return request('/checkout', { method: 'POST', headers: { 'Idempotency-Key': key }, body })
  },
  order: (id: string): Promise<BillingOrder> => request(`/orders/${encodeURIComponent(id)}`),
  resume: (id: string, acceptance?: PurchaseAcceptance): Promise<CheckoutResult> => request(`/orders/${encodeURIComponent(id)}/checkout`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(acceptance ?? {}),
  }),
}

export const useBilling = () => useQuery({ queryKey: ['billing', 'config'], queryFn: billingApi.config, staleTime: 30_000, retry: 0 })

export function hasBillingIdentity(me: { auth?: string; unauthenticated_instance?: boolean } | null | undefined): boolean {
  return !!me && !!me.auth && !me.unauthenticated_instance && !['single_user', 'token'].includes(me.auth)
}

export function checkoutAddress(value: string): string {
  let url: URL
  try { url = new URL(value) } catch { throw new Error('The payment service returned an invalid checkout address.') }
  if (url.protocol !== 'https:' || url.hostname !== 'checkout.stripe.com' || url.username || url.password || url.port) {
    throw new Error('The payment service returned an unrecognised checkout address.')
  }
  return url.href
}

export function openCheckout(value: string): void {
  window.location.assign(checkoutAddress(value))
}

export function orderPrice(order: Pick<BillingOrder, 'amount_minor' | 'currency'>): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: order.currency.toUpperCase() }).format(order.amount_minor / 100)
}
