/** What an exhibit row shows of the intake record (N-19; browser QA P1-8; design review DS-01): the date the custodian
 *  stated, the comparison of the stated hash with the file received, and the intake notes. */

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const pad = (n: number) => String(n).padStart(2, '0')

/** Notes up to this length stand in the row; longer ones open from a disclosure, so one exhibit does not push the others away. */
export const NOTES_INLINE = 160

/** `28 Sep 2026 23:45 UTC`: the stated date as the reports print it (`formatting.utc`). */
export function fmtUtcMinutes(iso: string): string {
  const t = new Date(iso)
  if (Number.isNaN(t.getTime())) return iso
  return `${t.getUTCDate()} ${MONTHS[t.getUTCMonth()]} ${t.getUTCFullYear()} ${pad(t.getUTCHours())}:${pad(t.getUTCMinutes())} UTC`
}

/** A mismatch at any intake of the exhibit stands (as the reports say it: `intake.custodian_check`); else a match, when a
 *  hash was stated; else nothing was stated. */
export function hashComparison(intakes: readonly { custodian_hash_match: string }[]): 'mismatch' | 'match' | null {
  if (intakes.some((r) => r.custodian_hash_match === 'mismatch')) return 'mismatch'
  return intakes.some((r) => r.custodian_hash_match === 'match') ? 'match' : null
}
