import type { ReactNode } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Eyebrow } from './ui'

const link = 'text-accent underline underline-offset-[3px] decoration-1 hover:decoration-2'

/** Not-found page (L-7; QA P2, design P2): an address that matches no page, or an analysis or case the caller cannot see. It says
 *  what was asked for and where to go; there is nothing to retry. A case resource of a case the caller is not a member of reads
 *  as not found by design, so the page names that possibility without confirming it. */
export default function NotFound({ what = 'page', children }: { what?: 'page' | 'analysis' | 'case'; children?: ReactNode }) {
  const { pathname } = useLocation()
  return (
    <section data-testid="not-found" data-what={what} aria-labelledby="not-found-title" className="max-w-[64ch]">
      <Eyebrow>Not found</Eyebrow>
      <h1 id="not-found-title" className="display text-h1 font-normal text-text mt-1">
        {what === 'page' ? 'This page does not exist' : `This ${what} is not available`}
      </h1>
      <p className="mt-3 m-0 text-body text-text">
        {children ?? (what === 'page'
          ? <>The address <span className="font-mono text-mono break-all">{pathname}</span> matches no page of this workspace. It may have been mistyped or copied incompletely.</>
          : <>No {what} with this address is visible to you. It may have been deleted, or it may belong to a case you are not a member of.</>)}
      </p>
      <p className="mt-6 m-0 flex flex-wrap gap-x-6 gap-y-2 text-small">
        <Link to="/" className={link}>Cases</Link>
        <Link to="/analyses" className={link}>Analyses</Link>
        <Link to="/upload" className={link}>New analysis</Link>
      </p>
    </section>
  )
}
