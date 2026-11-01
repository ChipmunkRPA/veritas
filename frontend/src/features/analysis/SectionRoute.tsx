import { useEffect } from 'react'
import { useLocation, useOutletContext } from 'react-router-dom'
import { prefersReducedMotion } from '../../lib/theme'
import type { WorkspaceContext } from './AnalysisWorkspace'

export type Section = 'summary' | 'findings' | 'image-evidence' | 'media-evidence' | 'exhibit' | 'reached' | 'ledger' | 'provenance' | 'methods' | 'appendix' | 'revision'

/** How long the page has to keep its height before a deep link counts as landed, and the longest it is followed. */
const SETTLE_MS = 1500
const FOLLOW_MS = 12_000

/** A nested workspace route (ARCHITECTURE §7.2). The workspace is one scrolling document (UX §2.3), so the route renders
 *  nothing and scrolls to its section once the workspace reports that the section has rendered.
 *
 *  N-19b (design review DS-11): a cold deep link landed 270 to 1,649 px from its section, because the sections above it gain
 *  height after that first scroll (an exhibit, the findings, the ledger arrive). The section is therefore brought back under
 *  the index each time the page changes its height, until the height has been still for `SETTLE_MS`, for `FOLLOW_MS` at most,
 *  and never after the reader has scrolled, pressed a key or touched the page. */
export default function SectionRoute({ section }: { section: Section }) {
  const { ready } = useOutletContext<WorkspaceContext>()
  const { hash } = useLocation()
  useEffect(() => {
    if (!ready) return
    if (section === 'summary') {
      // A link to a ledger row (`#signal-…`) is on the summary's route: the workspace scrolls to that row, and a page that
      // becomes ready after it must not take the reader back to the top.
      if (!/^#(?:signal-|finding-|image-evidence-|media-evidence-)/.test(hash)) window.scrollTo({ top: 0 })
      return
    }
    const target = () => (/^#(?:finding-F[1-9]\d*|image-evidence-[A-Z]\d+|media-evidence-(?:[A-Z]\d+|AV\d+|voice|timing|visual|all|background))$/.test(hash) ? document.getElementById(hash.slice(1)) : null)
      ?? document.getElementById(`section-${section}`)
    target()?.scrollIntoView({ block: 'start', behavior: prefersReducedMotion() ? 'auto' : 'smooth' })
    if (typeof ResizeObserver === 'undefined') return
    let stopped = false
    let quiet = 0
    let first = true
    const inputs = ['wheel', 'touchstart', 'keydown', 'mousedown'] as const
    const stop = () => {
      if (stopped) return
      stopped = true
      observer.disconnect()
      window.clearTimeout(quiet)
      window.clearTimeout(limit)
      for (const name of inputs) window.removeEventListener(name, stop)
    }
    const observer = new ResizeObserver(() => {
      if (first) { first = false; return }  // the observation of the size as it is: the first scroll is under way
      if (stopped) return
      target()?.scrollIntoView({ block: 'start', behavior: 'auto' })
      window.clearTimeout(quiet)
      quiet = window.setTimeout(stop, SETTLE_MS)
    })
    observer.observe(document.body)
    quiet = window.setTimeout(stop, SETTLE_MS)
    const limit = window.setTimeout(stop, FOLLOW_MS)
    for (const name of inputs) window.addEventListener(name, stop, { passive: true })
    return stop
  }, [ready, section, hash])
  return null
}
