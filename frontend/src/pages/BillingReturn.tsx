import { useRef } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { CheckCircle2, CreditCard, LoaderCircle } from 'lucide-react'
import { Button } from '../components/ui'
import { billingApi, openCheckout, orderPrice } from '../lib/api/billing'

export default function BillingReturn() {
  const [params] = useSearchParams()
  const id = params.get('order_id') ?? ''
  const validId = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)
  const returnedFromCancel = params.get('cancelled') === '1'
  const opening = useRef(false)
  const order = useQuery({
    queryKey: ['billing', 'order', id], queryFn: () => billingApi.order(id), enabled: validId, retry: 0,
    refetchInterval: (query) => {
      const data = query.state.data
      return query.state.error || ['expired', 'payment_failed'].includes(data?.status ?? '') || (data?.status === 'fulfilled' && ['done', 'failed', 'cancelled'].includes(data.analysis_status ?? '')) ? false : 2000
    },
  })
  const resume = useMutation({
    mutationFn: async () => { const result = await billingApi.resume(id); openCheckout(result.checkout_url) },
    onError: () => { opening.current = false; void order.refetch() },
  })
  const data = order.data
  const fulfilled = data?.status === 'fulfilled'
  const failed = fulfilled && ['failed', 'cancelled'].includes(data.analysis_status ?? '')
  const expired = data?.status === 'expired'
  const paymentFailed = data?.status === 'payment_failed'
  const waiting = !!data && !fulfilled && !expired && !paymentFailed

  return <div className="max-w-[720px] mx-auto">
    <div className="flex items-center gap-3 text-accent mb-5" aria-hidden>
      {fulfilled ? <CheckCircle2 size={28} /> : <CreditCard size={28} />}
      <span className="eyebrow">One analysis · One payment</span>
    </div>
    <h1 className="display text-h1 font-normal text-text m-0">
      {fulfilled ? 'Payment confirmed' : paymentFailed ? 'Payment did not complete' : expired ? 'Checkout expired' : returnedFromCancel ? 'Checkout closed' : 'Confirming your payment'}
    </h1>
    {!validId ? <p role="alert" className="text-danger mt-4">This payment link is incomplete. Return to New analysis to start again.</p>
      : order.isPending ? <p role="status" className="flex items-center gap-2 text-text-muted mt-4"><LoaderCircle size={18} className="animate-spin" aria-hidden />Checking your order…</p>
      : order.isError ? <div className="mt-4">
        <p role="alert" className="text-danger">{order.error.message}</p>
        <Button variant="ghost" onClick={() => void order.refetch()}>Check again</Button>
        <p className="text-small text-text-muted mt-3">If you have already paid, check this order again before starting another checkout.</p>
      </div> : data && <>
        <div className="mt-6 p-6 rounded-md border border-border bg-surface-2">
          <p className="text-h3 text-text font-medium m-0">{orderPrice(data)} <span className="text-small font-normal text-text-muted">USD · one automated analysis</span></p>
          {fulfilled ? <>
            <p className="text-small text-text mt-3 mb-0">{failed
              ? 'Payment was confirmed, but the analysis could not finish. Contact support with the order reference below so we can resolve it.'
              : data.analysis_status === 'done' ? 'Your analysis is complete. Open it to read the conclusion and supporting evidence.'
              : 'Your analysis has been queued. You can open its progress now; refreshing this page will not start another analysis.'}</p>
            {data.analysis_id && <Link to={`/analyses/${encodeURIComponent(data.analysis_id)}`} className="inline-flex mt-5 text-accent underline underline-offset-4 font-medium">{data.analysis_status === 'done' ? 'View analysis' : 'Open analysis'}</Link>}
          </> : paymentFailed ? <p className="text-small text-text mt-3 mb-0">Stripe reported that this payment failed. No analysis was started for this order. You can start a new analysis with a different payment method.</p>
            : expired ? <p className="text-small text-text mt-3 mb-0">This checkout can no longer accept payment. Start a new analysis when you are ready.</p>
            : <p role="status" className="text-small text-text mt-3 mb-0">{returnedFromCancel
              ? 'Payment has not been confirmed for this order. You can return to the same checkout when you are ready.'
              : 'We are waiting for Stripe to confirm payment. Your analysis starts after that confirmation. This page checks automatically.'}</p>}
          {waiting && <div className="mt-5">
            <Button variant={returnedFromCancel ? 'primary' : 'ghost'} disabled={resume.isPending} onClick={() => {
              if (opening.current) return
              opening.current = true
              resume.mutate()
            }}>{resume.isPending ? 'Opening checkout…' : 'Return to checkout'}</Button>
            <p className="text-caption text-text-muted mt-2 mb-0">Continues this order; it does not create a second payment.</p>
          </div>}
          {resume.error && <p role="alert" className="text-small text-danger mt-3 mb-0">{resume.error.message}</p>}
        </div>
        <p className="text-caption text-text-muted mt-5 break-all">Order reference: <span className="font-mono">{data.order_id}</span></p>
        <p className="text-small text-text-muted mt-4">An automated result may be inconclusive. For payment support or a separate expert analysis engagement, contact <a href={`mailto:sales@chipmunkrpa.com?subject=${encodeURIComponent(`Chipmunk Verify order ${data.order_id}`)}`} className="text-accent underline underline-offset-4">sales@chipmunkrpa.com</a>.</p>
      </>}
    <Link to={expired || paymentFailed || !validId ? '/upload' : '/analyses'} className="inline-flex mt-6 text-small text-accent underline underline-offset-4">{expired || paymentFailed || !validId ? 'New analysis' : 'Back to analyses'}</Link>
  </div>
}
