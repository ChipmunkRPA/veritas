import type { ModelCard } from '../../lib/api/queries'

/** Catalogue availability, not a promise that an input passes applicability checks.
 * Loaded state is process-local and may be false for weights a separate worker can use. */
export const availableChecks = (cards: readonly ModelCard[]): ModelCard[] =>
  cards.filter((card) => card.enabled && card.installed)
