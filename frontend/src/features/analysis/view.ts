/** Plain / Detailed self-service views. Both show the automated conclusion and source evidence.
 *  The existing "expert" URL value opens the detailed record, not a paid expert service or manual review workflow. */

export type ResultView = 'plain' | 'expert'

/** localStorage key of the last view the user chose; the URL parameter `view` wins over it. */
export const VIEW_KEY = 'veritas_result_view'

/** Workspace sections that exist only in the expert view: a deep link to one of them opens Expert. */
export const EXPERT_SECTIONS: ReadonlySet<string> = new Set(['exhibit', 'reached', 'ledger', 'provenance', 'methods', 'appendix', 'revision'])

const isView = (v: string | null | undefined): v is ResultView => v === 'plain' || v === 'expert'

/** UX-01: Plain is the default for the viewer role and for quick analyses (an implicit case); everyone else starts in Expert. */
export function defaultView(role: string | null | undefined, implicitCase: boolean | null | undefined): ResultView {
  return role === 'viewer' || implicitCase ? 'plain' : 'expert'
}

/** The view to render, or null while the default is not known yet (the caller renders L1–L2 only).
 *  Order: a deep-linked expert section, the `view` URL parameter, the stored choice, the role / quick-analysis default. */
export function resolveView(section: string, param: string | null, stored: string | null, fallback: ResultView | null): ResultView | null {
  if (EXPERT_SECTIONS.has(section)) return 'expert'
  if (isView(param)) return param
  if (isView(stored)) return stored
  return fallback
}

export function storedView(): string | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage.getItem(VIEW_KEY)
  } catch {
    return null
  }
}

export function storeView(view: ResultView): void {
  try {
    localStorage.setItem(VIEW_KEY, view)
  } catch {
    // Private mode or storage disabled: the URL still carries the choice.
  }
}
