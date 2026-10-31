import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'
const STORAGE_KEY = 'veritas_theme'

function readStored(): Theme | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'dark' || v === 'light' ? v : null
  } catch { return null }
}

function systemTheme(): Theme {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function initialTheme(): Theme { return readStored() ?? systemTheme() }

/** Apply to <html data-theme>. Light is the default; dark is opt-in via attribute. */
export function applyTheme(t: Theme) {
  document.documentElement.setAttribute('data-theme', t)
}

/** Reads/writes `data-theme`; persisted in localStorage under `veritas_theme`. */
export function useTheme(): { theme: Theme; toggle: () => void; set: (t: Theme) => void } {
  const [theme, setTheme] = useState<Theme>(() => initialTheme())
  useEffect(() => { applyTheme(theme) }, [theme])
  useEffect(() => {
    // Follow the OS only while the user has not chosen explicitly.
    if (readStored()) return
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)')
    if (!mq) return
    const on = (e: MediaQueryListEvent) => setTheme(e.matches ? 'dark' : 'light')
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  const set = useCallback((t: Theme) => {
    try { localStorage.setItem(STORAGE_KEY, t) } catch { /* ignore */ }
    setTheme(t)
  }, [])
  const toggle = useCallback(() => set(theme === 'dark' ? 'light' : 'dark'), [theme, set])
  return { theme, toggle, set }
}

/** Respect `prefers-reduced-motion` for JS-driven animation (count-up). */
export function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}
