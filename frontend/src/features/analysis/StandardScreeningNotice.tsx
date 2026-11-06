import { STANDARD_SERVICE_POLICY as policy } from './standardServicePolicy'

export default function StandardScreeningNotice() {
  return (
    <aside data-testid="standard-screening-notice" aria-label={policy.title}
      className="mt-5 max-w-[80ch] rounded-md border border-border bg-surface-2 px-4 py-4 text-small text-text">
      <h3 className="m-0 text-body font-semibold">{policy.title}</h3>
      <p className="mt-2 mb-0">{policy.confidence_notice}</p>
      <p className="mt-2 mb-0">{policy.court_notice}</p>
      <p className="mt-2 mb-0">{policy.private_engagement_notice} Contact{' '}
        <a className="underline text-accent" href={`mailto:${policy.expert_contact}`}>{policy.expert_contact}</a>.
      </p>
    </aside>
  )
}
