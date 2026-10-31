import { useEffect, useState } from 'react'

/** Whether a CSS media query matches, kept in step with the window (false where `matchMedia` is unavailable). */
export function useMediaQuery(query: string): boolean {
  const read = () => typeof window !== 'undefined' && !!window.matchMedia?.(query).matches
  const [matches, setMatches] = useState(read)
  useEffect(() => {
    const mq = window.matchMedia?.(query)
    if (!mq) return
    const onChange = () => setMatches(mq.matches)
    onChange()
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [query])
  return matches
}
