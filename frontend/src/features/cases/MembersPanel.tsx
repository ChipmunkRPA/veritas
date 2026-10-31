import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { apiClient, unwrap } from '../../lib/api/client'
import { qk, useMemberCandidates, useMembers } from '../../lib/api/queries'
import { caseRolesFor, type CaseRoleName } from '../../lib/api/roles'
import { fmtDate } from '../../lib/format'
import { Button, EmptyState, ErrorState, Menu, Select, SkeletonRows } from '../../components/ui'
import { CASE_ROLE_WORD, sentenceCase } from './labels'

type CaseRole = CaseRoleName

/** Members of a case (PRD CM-02; ARCHITECTURE §5.2 members routes; D-67): the owner or an administrator adds and removes
 *  members and changes their roles. The form and the role menu offer no case role above the principal's instance role
 *  (`caseRolesFor`). Who can be added comes from `GET /cases/{id}/members/candidates`, which the owner can read. Every change
 *  is a `permission_changed` custody event. */
export default function MembersPanel({ caseId, canManage }: { caseId: string; canManage: boolean }) {
  const qc = useQueryClient()
  const members = useMembers(caseId)
  const candidates = useMemberCandidates(caseId, canManage)
  const [candidate, setCandidate] = useState('')
  const [role, setRole] = useState<CaseRole>('viewer')
  const done = () => {
    void qc.invalidateQueries({ queryKey: qk.members(caseId) })
    void qc.invalidateQueries({ queryKey: qk.memberCandidates(caseId) })
    void qc.invalidateQueries({ queryKey: qk.custody(caseId) })
  }
  const put = useMutation({
    mutationFn: ({ principalId, caseRole }: { principalId: string; caseRole: CaseRole }) =>
      unwrap(apiClient.PUT('/api/v2/cases/{case_id}/members/{principal_id}', { params: { path: { case_id: caseId, principal_id: principalId } }, body: { role: caseRole } })),
    onSuccess: () => { setCandidate(''); done() },
  })
  const remove = useMutation({
    mutationFn: (principalId: string) =>
      unwrap(apiClient.DELETE('/api/v2/cases/{case_id}/members/{principal_id}', { params: { path: { case_id: caseId, principal_id: principalId } } })),
    onSuccess: done,
  })

  const rows = members.data ?? []
  const outside = candidates.data ?? []
  const chosen = outside.find((p) => p.principal_id === candidate)
  // the roles the chosen principal can use, highest first; the form starts on the highest that is not the owner's
  const offered = caseRolesFor(chosen?.principal_role)
  const choose = (id: string) => {
    setCandidate(id)
    const roles = caseRolesFor(outside.find((p) => p.principal_id === id)?.principal_role)
    setRole(roles.find((r) => r !== 'owner') ?? 'viewer')
  }
  const failure = put.error ?? remove.error

  return (
    <section className="mt-12" aria-labelledby="members-heading" data-testid="members">
      <h2 id="members-heading" className="eyebrow m-0 border-b border-border-strong pb-2">Members</h2>
      {members.isPending && <SkeletonRows className="mt-4" widths={['50%', '44%']} />}
      {members.error && <ErrorState className="mt-4" onRetry={() => void members.refetch()}>{members.error.message}</ErrorState>}
      {members.isSuccess && rows.length === 0 && <EmptyState className="mt-4">No named members. Administrators see every case.</EmptyState>}
      {rows.length > 0 && (
        <div className="mt-1 overflow-x-auto">
          {/* below 640 px one stacked block per member (L-7): the role menu and Remove stay on screen */}
          <table role="table" className="stack-sm w-full border-collapse">
            <thead role="rowgroup">
              <tr role="row" className="border-b border-border">
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4">Name</th>
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-40">Case role</th>
                <th role="columnheader" scope="col" className="eyebrow text-left font-medium py-2 pr-4 w-40">Instance role</th>
                <th role="columnheader" scope="col" className="eyebrow text-right font-medium py-2 pr-4 w-28">Added</th>
                {canManage && <th role="columnheader" scope="col" className="w-20"><span className="sr-only">Actions</span></th>}
              </tr>
            </thead>
            <tbody role="rowgroup">
              {rows.map((m) => (
                <tr role="row" key={m.principal_id} data-testid="member-row" className="border-b border-border h-11 max-[639px]:h-auto">
                  <td role="cell" className="pr-4 text-small text-text">
                    {m.display_name} <span className="font-mono text-mono text-text-muted ml-1">{m.username}</span>
                  </td>
                  <td role="cell" data-th="case role" className="pr-4 text-small text-text">
                    {canManage
                      ? <Menu label={CASE_ROLE_WORD[m.case_role]} ariaLabel={`Case role of ${m.display_name}: ${CASE_ROLE_WORD[m.case_role]}`} triggerClassName="text-text -ml-1"
                          onSelect={(r) => { if (r !== m.case_role) put.mutate({ principalId: m.principal_id, caseRole: r }) }}
                          items={caseRolesFor(m.principal_role).map((r) => ({ id: r, label: CASE_ROLE_WORD[r] }))} />
                      : CASE_ROLE_WORD[m.case_role]}
                  </td>
                  <td role="cell" data-th="instance role" className="pr-4 text-small text-text-muted">{sentenceCase(m.principal_role)}</td>
                  <td role="cell" data-th="added" className="pr-4 text-right text-small text-text-muted num max-[639px]:text-left">{fmtDate(m.added_at)}</td>
                  {canManage && (
                    <td role="cell" className="text-right max-[639px]:text-left">
                      <Button variant="danger-text" onClick={() => remove.mutate(m.principal_id)} disabled={remove.isPending} aria-label={`Remove ${m.display_name}`}>Remove</Button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {failure && <p role="alert" className="mt-3 mb-0 text-small text-danger">{failure.message}</p>}
      {candidates.error && <p role="alert" className="mt-3 mb-0 text-small text-danger">{candidates.error.message}</p>}
      {canManage && candidates.isSuccess && outside.length > 0 && (
        <form className="mt-4 flex flex-wrap items-end gap-3" data-testid="add-member" onSubmit={(e) => { e.preventDefault(); if (candidate) put.mutate({ principalId: candidate, caseRole: role }) }}>
          <label className="flex flex-col gap-1 text-small text-text-muted w-72 max-w-full">
            Add a member
            <Select value={candidate} onChange={(e) => choose(e.target.value)} className="w-full">
              <option value="">Choose…</option>
              {outside.map((p) => <option key={p.principal_id} value={p.principal_id}>{p.display_name} ({p.username}, {p.principal_role.replace(/_/g, ' ')})</option>)}
            </Select>
          </label>
          <label className="flex flex-col gap-1 text-small text-text-muted w-40">
            Case role
            <Select value={role} onChange={(e) => setRole(e.target.value as CaseRole)} disabled={!candidate} className="w-full">
              {offered.map((r) => <option key={r} value={r}>{CASE_ROLE_WORD[r]}</option>)}
            </Select>
          </label>
          <Button variant="ghost" type="submit" disabled={!candidate || put.isPending}>{put.isPending ? 'Adding…' : 'Add member'}</Button>
          {chosen && offered.length < 4 && (
            <p className="m-0 basis-full text-caption text-text-muted" data-testid="role-bound">
              {chosen.display_name} is {/^[aeiou]/.test(chosen.principal_role) ? 'an' : 'a'} {chosen.principal_role.replace(/_/g, ' ')} on this instance, so the case role goes no higher than {CASE_ROLE_WORD[offered[0]].toLowerCase()}.
            </p>
          )}
        </form>
      )}
      {canManage && candidates.isSuccess && outside.length === 0 && <p className="mt-3 mb-0 text-caption text-text-muted">Every account of this instance is a member of the case.</p>}
    </section>
  )
}
