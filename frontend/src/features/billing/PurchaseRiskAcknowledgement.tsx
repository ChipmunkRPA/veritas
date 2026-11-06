import type { PurchaseNotice } from '../../lib/api/billing'
import type { Kind } from '../../lib/api'

export default function PurchaseRiskAcknowledgement({ notice, accepted, onChange, kind, disabled = false }: {
  notice: PurchaseNotice; accepted: boolean; onChange: (accepted: boolean) => void; kind: Kind; disabled?: boolean
}) {
  return (
    <section aria-label="Analysis limitations before payment" className="mt-6 rounded-md border border-border bg-surface-2 p-5">
      <h2 className="m-0 text-h3 font-semibold text-text">{notice.title}</h2>
      <ul id="purchase-risk-points" className="mt-3 mb-0 list-disc space-y-2 pl-5 text-small text-text">
        {notice.points.map(point => <li key={point}>{point}</li>)}
      </ul>
      {kind === 'document' && <p id="purchase-document-note" className="mt-3 mb-0 text-small font-medium text-text">{notice.document_notice}</p>}
      <label className="mt-4 flex items-start gap-3 text-small font-medium text-text">
        <input type="checkbox" checked={accepted} onChange={event => onChange(event.target.checked)} disabled={disabled}
          aria-describedby={kind === 'document' ? 'purchase-risk-points purchase-document-note' : 'purchase-risk-points'} className="mt-1 accent-text" />
        <span>{notice.acknowledgement_label}</span>
      </label>
    </section>
  )
}
