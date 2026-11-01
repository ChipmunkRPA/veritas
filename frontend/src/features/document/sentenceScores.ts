import type { SentenceScores } from '../../lib/api'

const SCORE_LABELS: Readonly<Record<string, string>> = {
  classification_response: 'Text response',
  classification_response_secondary: 'Second text response',
  pattern_ratio: 'Pattern ratio',
  pattern_response: 'Pattern response',
  style: 'Style response',
}

/** Labels describe measurements, never implementation identities or opaque IDs. */
export function sentenceScoreRows(scores: SentenceScores) {
  let additional = 0
  return Object.entries(scores).flatMap(([key, value]) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) return []
    const known = Object.prototype.hasOwnProperty.call(SCORE_LABELS, key)
    return [{ key, label: known ? SCORE_LABELS[key] : `Additional check ${++additional}`, value }]
  })
}
