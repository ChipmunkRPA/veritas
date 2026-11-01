import type { Kind } from '../../lib/api'
import type { AnalysisProfile } from '../../lib/api/intake'
import type { ModelCard } from '../../lib/api/queries'
import { Button } from '../../components/ui'
import { intakeChecks, intakeGroups, PROFILE_NAMES } from './catalogue'

export default function IntakeChecks({ kind, profile, cards, pending, error, onRetry, batch = false }: {
  kind: Kind; profile: AnalysisProfile | ''; cards: readonly ModelCard[] | undefined
  pending: boolean; error: boolean; onRetry: () => void; batch?: boolean
}) {
  // Refetch errors must not present a stale count as current availability.
  if (error || (!pending && !cards)) return (
    <div className="mt-3 text-small text-text-muted" role="status">
      <p className="m-0">Available checks could not be confirmed.</p>
      <Button variant="text" className="mt-2" onClick={onRetry}>Retry available checks</Button>
    </div>
  )
  if (pending) return <p role="status" className="mt-3 mb-0 text-small text-text-muted">Loading available checks…</p>

  const checks = intakeChecks(cards!, kind, profile)
  const groups = intakeGroups(checks)
  return (
    <div data-testid="intake-checks">
      <p className="text-small text-text-muted mt-2 mb-4">
        {profile ? 'These checks are available for the selected profile.' : 'The instance chooses the profile when you submit. These checks are available in at least one profile; choose Quick, Full or Deep to see its checks.'}
        {' '}Some checks need suitable faces, speech or other source features. Every result reports what ran and what could not be assessed.
      </p>
      {batch && <p className="text-small text-text-muted mt-0 mb-4">This guide describes the selected source type. Each batch file uses the checks for its detected type.</p>}
      {checks.length === 0
        ? <p role="status" className="text-small text-text mt-0 mb-0">No checks are currently available for this source{profile ? ` in ${PROFILE_NAMES[profile]}` : ''}. An assessment cannot be made with the checks currently available.</p>
        : <>
          <p className="text-caption text-text-muted mt-0 mb-5 num">{checks.length} {checks.length === 1 ? 'check' : 'checks'} available{profile ? ` for ${PROFILE_NAMES[profile]}` : ' across profiles'}.</p>
          <dl className="m-0 space-y-5">
            {groups.map((group, index) => <div key={group.key} className="grid grid-cols-[16px_1fr] gap-x-3">
              <dt className="col-span-2 flex gap-3 text-small font-medium text-text">
                <span aria-hidden className="w-4 shrink-0 text-caption font-normal font-mono text-text-muted pt-0.5">{String(index + 1).padStart(2, '0')}</span>
                <span>{group.title}</span>
              </dt>
              <dd className="col-start-2 m-0 mt-1 text-small text-text-muted">{group.detail}</dd>
              <dd className="col-start-2 m-0 mt-1 text-caption text-text-muted num">
                  {group.count} {group.count === 1 ? 'check' : 'checks'}{!profile && ` across ${group.profiles.map((p) => PROFILE_NAMES[p]).join(', ')}`}
              </dd>
            </div>)}
          </dl>
        </>}
    </div>
  )
}
