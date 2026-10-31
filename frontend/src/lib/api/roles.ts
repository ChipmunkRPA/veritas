/** `rbac.SATISFIES["examiner"]`: the principal roles that may take examiner actions (reports, packages, re-runs). */
export const EXAMINER_ROLES: ReadonlySet<string> = new Set(['administrator', 'examiner', 'api_client'])

/** `rbac.SATISFIES["reviewer"]`: the principal roles that may read the custody log and its export (`GET /audit`). */
export const REVIEWER_ROLES: ReadonlySet<string> = new Set(['administrator', 'reviewer'])

type Roles = { roles?: readonly string[] | null } | null | undefined
const holds = (me: Roles, allowed: ReadonlySet<string>): boolean => (me?.roles ?? []).some((r) => allowed.has(r))

/** Whether the principal may read the custody log of a case (`GET /cases/{id}/custody`): a reviewer or an administrator, and
 *  the owner of the case whatever the owner's instance role (D-67). The page does not request it otherwise: a refusal it can
 *  foresee is a state, not a failed request in the console (L-7). */
export const mayReadCustody = (me: Roles, caseRole?: string | null): boolean => holds(me, REVIEWER_ROLES) || (!!me && caseRole === 'owner')

/** The export of the log is the route of the whole audit log (`GET /audit/export`): reviewers and administrators. */
export const mayExportCustody = (me: Roles): boolean => holds(me, REVIEWER_ROLES)

/** The case roles a principal of this instance role can use (D-67: "case roles up to a member's instance role"; N-19a): the
 *  member form offers no role above them. An examiner owns, examines and peer-reviews (ADR 0011: the case grants the peer
 *  review); a reviewer reviews and reads; a viewer reads. The owner role manages members and intake, which are examiner routes. */
const CASE_ROLES_FOR: Record<string, readonly CaseRoleName[]> = {
  administrator: ['owner', 'examiner', 'reviewer', 'viewer'], examiner: ['owner', 'examiner', 'reviewer', 'viewer'],
  api_client: ['owner', 'examiner', 'reviewer', 'viewer'], reviewer: ['reviewer', 'viewer'], viewer: ['viewer'],
}
export type CaseRoleName = 'owner' | 'examiner' | 'reviewer' | 'viewer'
export const caseRolesFor = (instanceRole: string | null | undefined): readonly CaseRoleName[] => CASE_ROLES_FOR[instanceRole ?? ''] ?? ['viewer']

/** Whether the principal is offered what starts work: "New analysis", "New case", the "Analyse" links and "Demo now" (N-19a,
 *  "Actions by role"; `rbac.SATISFIES["examiner"]`: intake and case creation are examiner actions). A viewer or a reviewer
 *  reads and reviews; the API refuses them an upload, so the page does not offer one. Not offered before the account is known. */
export const mayStartWork = (me: Roles): boolean => holds(me, EXAMINER_ROLES)

/** Deleting an analysis is an administrator action (`DELETE /api/analyses/{aid}`). */
export const mayDelete = (me: Roles): boolean => (me?.roles ?? []).includes('administrator')

/** N-19a ("Actions by role"): delete is not offered on a signed analysis. The API refuses it (409: a signed revision is kept),
 *  so the menu does not hold it. Signed is an examiner's sign-off: the states reviewed, approved and disputed, or a lock. */
export const SIGNED_STATES: ReadonlySet<string> = new Set(['reviewed', 'approved', 'disputed'])
export const deletable = (state: string | null | undefined, lockedAt: string | null | undefined): boolean =>
  !lockedAt && !(state && SIGNED_STATES.has(state))

/** What the server cannot run (the engine block's capacity line, N-14a) is an administrator's to act on. */
export const maySeeServerCapacity = (me: Roles): boolean => (me?.roles ?? []).includes('administrator')

/** The facts of the engine in the sidebar (its device, the count of checks, the queue) are the administrator's (N-19a,
 *  "Actions by role"): they describe the server, not a case. Everyone else sees whether the engine is ready, and no more. */
export const maySeeEngineFacts = maySeeServerCapacity

/** The engineering record of a model card (the build's notes and the admission record) is for administrators: it is not product
 *  content (D-71; `rbac.py` gives the record route to administrators). */
export const mayReadEngineeringRecord = (me: Roles): boolean => (me?.roles ?? []).includes('administrator')

/** Whether the principal may take examiner actions on a case resource (statuses, notes, the conclusion, re-runs; `app/cases/access.py`):
 *  an administrator always; otherwise an examiner-capable principal who is an owner or examiner of the case. */
export function mayExamine(me: Roles, caseRole: string | null | undefined): boolean {
  if ((me?.roles ?? []).includes('administrator')) return true
  return holds(me, EXAMINER_ROLES) && (caseRole === 'owner' || caseRole === 'examiner')
}

/** Whether the principal is a named account (`app/api/v2_review.py` `named_examiner`). A conclusion, a sign-off, a withdrawal
 *  and the re-run of a signed revision carry a named person: the API refuses them to the local examiner of a single-user
 *  instance, to the bootstrap administrator key and to an API client, whatever their role. */
export function namedAccount(me: { auth?: string | null; unauthenticated_instance?: boolean | null; roles?: readonly string[] | null } | null | undefined): boolean {
  return !!me && !me.unauthenticated_instance && me.auth !== 'bootstrap' && !(me.roles ?? []).includes('api_client')
}
