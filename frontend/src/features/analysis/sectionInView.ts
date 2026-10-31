import { useEffect, useState } from 'react'

/** The line of the page that decides which section the reader is in: just under the sticky section index (40 px) and the
 *  space a section keeps above its heading (`scroll-mt-14`). */
export const SECTION_LINE = 96

/** The section the reader is in (N-19b; design review DS-11: the index never followed scrolling). An IntersectionObserver
 *  whose root is one line of the viewport, `SECTION_LINE` px from its top, reports the sections that span that line; the
 *  last of them in the order of the index is the current one. Above the first section it is the first; at the end of the
 *  page it is the last section, which may be too short to reach the line. Null until something was observed, and where the
 *  browser has no IntersectionObserver: the index then marks the section of the address. The address is never changed. */
export function useSectionInView(sections: readonly string[], enabled: boolean): string | null {
  const [current, setCurrent] = useState<string | null>(null)
  const key = sections.join(' ')
  useEffect(() => {
    if (!enabled || typeof IntersectionObserver === 'undefined') { setCurrent(null); return }
    const order = key.split(' ').filter(Boolean)
    const spanning = new Set<string>()
    let observer: IntersectionObserver | null = null
    const element = (id: string) => document.getElementById(`section-${id}`)
    const atEnd = () => window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2
    const decide = () => {
      const present = order.filter((id) => element(id))
      if (present.length === 0) return
      if (atEnd() && window.scrollY > 0) { setCurrent(present[present.length - 1]); return }
      const hit = [...present].reverse().find((id) => spanning.has(id))
      if (hit) { setCurrent(hit); return }
      // between two sections the reader is in the one whose heading passed the line last; above the first, in the first
      const passed = [...present].reverse().find((id) => element(id)!.getBoundingClientRect().top <= SECTION_LINE)
      setCurrent(passed ?? present[0])
    }
    const observe = () => {
      observer?.disconnect()
      spanning.clear()
      observer = new IntersectionObserver((entries) => {
        for (const e of entries) {
          const id = e.target.id.replace(/^section-/, '')
          if (e.isIntersecting) spanning.add(id)
          else spanning.delete(id)
        }
        decide()
      }, { rootMargin: `-${SECTION_LINE}px 0px -${Math.max(0, window.innerHeight - SECTION_LINE - 1)}px 0px`, threshold: 0 })
      for (const id of order) { const el = element(id); if (el) observer.observe(el) }
    }
    observe()
    // the end of the page has no section that crosses the line: a scroll that stops there is seen here
    let frame = 0
    const onScroll = () => { if (!frame) frame = window.requestAnimationFrame(() => { frame = 0; decide() }) }
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', observe)
    // a section that arrives later (its data loads) changes the page's height: the sections are then observed anew
    const grown = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(observe)
    grown?.observe(document.body)
    return () => {
      observer?.disconnect()
      grown?.disconnect()
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', observe)
      if (frame) window.cancelAnimationFrame(frame)
    }
  }, [enabled, key])
  return current
}
