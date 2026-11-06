import { STANDARD_SERVICE_POLICY as policy } from './standardServicePolicy'
import { ArrowUpRight, MessagesSquare } from 'lucide-react'
import { ButtonLink, CodeChip } from '../../components/ui'

// Business enquiries for separately contracted expert services.
const EXPERT_CONTACT_EMAIL = 'sales@chipmunkrpa.com'

/** A separate service enquiry, outside the automated result and examination workflow. */
export default function ExpertAnalysisInvitation({ analysisId }: { analysisId: string }) {
  const subject = `Chipmunk Verify — expert analysis enquiry (${analysisId.slice(0, 8)})`
  const body = [
    'Hello Chipmunk team,',
    '',
    'I would like to discuss a separate expert analysis engagement.',
    '',
    `Chipmunk Verify analysis reference: ${analysisId}`,
    '',
    'Support requested (detailed analysis, written expert opinion, or expert witness support):',
    '',
    'Purpose and timeline:',
    '',
    'Preferred contact details:',
    '',
  ].join('\r\n')
  const href = `mailto:${EXPERT_CONTACT_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`

  return (
    <aside aria-labelledby="expert-analysis-heading" data-testid="expert-analysis-invitation"
      className="rounded-md border border-border bg-surface-2 p-5 sm:p-6 print:hidden">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between lg:gap-8">
        <div className="min-w-0 max-w-[68ch]">
          <p className="eyebrow m-0 flex items-center gap-2 text-accent">
            <MessagesSquare size={16} strokeWidth={1.5} aria-hidden /> Expert services
          </p>
          <h2 id="expert-analysis-heading" className="display mt-2 mb-0 text-h2 font-medium text-text">Need a privately researched expert report?</h2>
          <p className="mt-2 mb-0 text-body text-text">
            {policy.private_engagement_notice}
          </p>
          <p className="mt-2 mb-0 text-small text-text-muted">
            {policy.court_notice}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-start gap-2 lg:max-w-[260px]">
          <ButtonLink href={href} variant="primary" className="min-h-11" aria-describedby="expert-analysis-contact-note">
            Discuss expert analysis <ArrowUpRight size={16} strokeWidth={1.5} aria-hidden />
          </ButtonLink>
          <CodeChip copy={EXPERT_CONTACT_EMAIL} title="Copy contact email">{EXPERT_CONTACT_EMAIL}</CodeChip>
          <p id="expert-analysis-contact-note" className="m-0 max-w-[36ch] text-caption text-text-muted">
            Opens your email app with the analysis reference. Files and results are not attached.
          </p>
        </div>
      </div>
    </aside>
  )
}
