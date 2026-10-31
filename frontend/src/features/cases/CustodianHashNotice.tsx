import type { AnalysisDetail } from '../../lib/api/queries'
import { fmtDateLong, shortHash } from '../../lib/format'

/** A custodian-stated hash that disagreed with the received file at intake (CC-01; browser QA P1-7): recorded, not blocked, so it
 *  is shown on the analysis itself and not only in the custody log that examiners may not be able to open. */
export default function CustodianHashNotice({ analysis }: { analysis: Pick<AnalysisDetail, 'custodian_hash' | 'sha256'> }) {
  const c = analysis.custodian_hash
  if (c?.state !== 'mismatch') return null
  const md5 = c.stated_hash != null && /^(md5:)?[0-9a-f]{32}$/i.test(c.stated_hash)
  return (
    <p role="status" data-testid="custodian-mismatch" className="mt-4 mb-0 border-y border-border py-3 text-small text-text max-w-[88ch]">
      <span className="font-medium text-danger">Custodian hash differs.</span>{' '}
      At intake on <span className="num">{fmtDateLong(c.intake_at)}</span> the custodian stated {md5 ? 'MD5' : 'SHA-256'}{' '}
      <span className="font-mono text-mono num">{shortHash(c.stated_hash?.replace(/^(sha256|md5):/i, ''))}</span>, but the file received hashes to SHA-256{' '}
      <span className="font-mono text-mono num">{shortHash(analysis.sha256)}</span>. This analysis is of the file received. Confirm with the
      custodian which file is the original before relying on it.
    </p>
  )
}
