import type { Case } from '../../lib/api/queries'

/** Display words for case fields (backend `app/api/v2_cases.py` literals). */
export const CASE_STATUS_WORD: Record<string, string> = { open: 'Open', in_review: 'In review', signed_off: 'Signed off', closed: 'Closed' }

export const CASE_ROLE_WORD: Record<string, string> = { owner: 'Owner', examiner: 'Examiner', reviewer: 'Reviewer', viewer: 'Viewer' }
export const CASE_ROLES = ['owner', 'examiner', 'reviewer', 'viewer'] as const

export const CLASSIFICATIONS = ['internal', 'confidential', 'restricted', 'public'] as const
export const CLASSIFICATION_WORD: Record<string, string> = { internal: 'Internal', confidential: 'Confidential', restricted: 'Restricted', public: 'Public' }

export const sentenceCase = (value: string): string => {
  const words = value.replace(/_/g, ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

export type StatusFilter = 'all' | Case['status']
export type ScopeFilter = 'all' | 'named' | 'unfiled'

/** Cases dashboard filter (UX §5.12): the search reads the case number, title and client or matter reference, case-insensitively;
 *  every whitespace-separated term must appear. */
export function filterCases(cases: readonly Case[], query: string, status: StatusFilter, scope: ScopeFilter): Case[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
  return cases.filter((c) => {
    if (status !== 'all' && c.status !== status) return false
    if ((scope === 'named' && c.implicit) || (scope === 'unfiled' && !c.implicit)) return false
    const text = [c.reference, c.title, c.matter_reference ?? ''].join(' ').toLowerCase()
    return terms.every((t) => text.includes(t))
  })
}
