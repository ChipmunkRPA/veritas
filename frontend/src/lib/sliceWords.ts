/** A validation slice in words (BUILD_SPEC N-19d; design review DS-19). The id of a slice is the build's key
 *  ("image/q>=90/side>1024/face<120"): the result keeps it, the page does not print it. Mirror of `slice_words` and
 *  `slices_in_text` of the report formatting; both read the vectors of `backend/tests/fixtures/slice_words_vectors.json`. */

import { CONTEXT } from './recordWords'

const FIXED: Record<string, string> = { lossless: 'lossless file', noface: 'no face', face: 'with a face', jpeg: 'JPEG', unknown: 'no validation slice' }
const NOUN: Record<string, [string, string]> = {
  q: ['JPEG quality ', ''], side: ['shorter side ', ' px'], face: ['face ', ' px'], res: ['', ' px high'], dur: ['duration ', ' s'], words: ['', ' words'],
}
const UNKNOWN: Record<string, string> = { side: 'size unknown', res: 'resolution unknown', dur: 'duration unknown', words: 'length unknown' }
const BOUND = /^(q|side|face|res|dur|words)(>=|<=|<|>)(\d+)s?$/
const RANGE = /^(q|side|face|res|dur|words)(\d+)-(\d+)s?$/
const ID = String.raw`(?:image|video|audio|document)/(?:[\w<>=.-]+/)*(?:(?:q|side|face|res|dur|words)(?:[<>]=?\d|\d+-\d|-unknown)|noface|lossless)[\w<>=/-]*`
const IN_TEXT = new RegExp(String.raw`(\(slice |\bslice )?\b(${ID})`, 'g')
const WHOLE = new RegExp(`^(?:${ID}|(?:image|video|audio|document)/unknown)$`)

const grouped = (digits: string) => Number(digits).toLocaleString('en-US')

/** `image/q>=90/side>1024/face<120` → "JPEG quality 90 or more, shorter side over 1,024 px, face under 120 px". The first part
 *  (the kind of media) is left out; a part outside the build's grammar keeps its own words. */
export function sliceWords(sliceId: string | null | undefined): string {
  if (!sliceId) return 'not recorded'
  const [kind, ...parts] = sliceId.split('/')
  const out = parts.map((part) => {
    const bound = BOUND.exec(part)
    if (bound) {
      const [before, unit] = NOUN[bound[1]]
      const n = `${grouped(bound[3])}${unit}`
      return before + ({ '>=': `${n} or more`, '<=': `${n} or less`, '<': `under ${n}`, '>': `over ${n}` } as Record<string, string>)[bound[2]]
    }
    const span = RANGE.exec(part)
    if (span) {
      const [before, unit] = NOUN[span[1]]
      return `${before}${grouped(span[2])} to ${grouped(span[3])}${unit}`
    }
    const unknown = part.endsWith('-unknown') ? UNKNOWN[part.slice(0, -'-unknown'.length)] : undefined
    return unknown ?? FIXED[part] ?? part.replace(/_/g, ' ')
  })
  return out.join(', ') || kind
}

/** A slice id inside a sentence the engine stored (a decision's reason, a limiting condition), in words: "(slice <id>)" reads
 *  "(<words>)", "slice <id>" reads "slice “<words>”", a bare id its words. */
export function slicesInText(text: string): string {
  return text.replace(IN_TEXT, (_match, lead: string | undefined, id: string) => {
    const words = sliceWords(id)
    return lead?.startsWith('(') ? `(${words}` : lead ? `slice “${words}”` : words
  })
}

const FALLBACK_REASON = /(no validation slice applies|weights each check by hand-set weights) \(uncalibrated fallback\)/g
const V1_NOTE = /\bv1: hypotheses not separated\b/g
// N-19k: a coverage finding stored with the key of its error-rate context ("video T2V inputs")
const CONTEXT_IN_TEXT = new RegExp(String.raw`\b(${Object.keys(CONTEXT).join('|')}) inputs\b`, 'g')
const QUESTIONS_TOGETHER = 'Not assessed separately: without a fusion model one score answers the questions together'
// N-19n (Q4-01; `effect.step_in_words`): the caveat of a failed step that a result stored with the stage's id in backticks.
// Only the step that writes the findings had no words; any other id reads as the step without words reads today
const RECORDED_AS = /The step that is recorded as `(\w+)` failed: this result lacks what that step gives\./g
const FINDINGS_STEP = 'The step that writes the findings failed: this result holds no findings from it.'
const UNNAMED_STEP = 'The step that is marked as failed in the analysis record failed: this result lacks what that step gives.'

/** A sentence the engine stored, as the page prints it (the report's `stored_words`): a slice id in words; the limiting
 *  condition of a result without a fusion model without the build's name for that path in brackets after it; and the note
 *  of such a result's one assessment in words; the error-rate context of a coverage finding in the words of `contextWords`. */
export const storedWords = (text: string): string =>
  FALLBACK_SCORE_WORDS.reduce((said, [from, to]) => said.split(from).join(to),
    slicesInText(text).replace(FALLBACK_REASON, '$1').replace(V1_NOTE, QUESTIONS_TOGETHER).replace(CONTEXT_IN_TEXT, (_match, key: string) => `${CONTEXT[key]} inputs`)
      .replace(RECORDED_AS, (_match, stage: string) => stage === 'synthesise' ? FINDINGS_STEP : UNNAMED_STEP))

/** N-19q (finding 9; `formatting.FALLBACK_SCORE_WORDS`): the fallback's two stored caveats call its number the score. */
export const FALLBACK_SCORE_WORDS: readonly (readonly [string, string])[] = [
  // N-19q (finding 10; `formatting.NO_PART_WORDS`): the stored note of a question the fusion model has no fitted part for
  ['this fuser artefact has no fitted H1 sub-fuser', 'the fusion model has no part fitted for this question'],
  ['this fuser artefact has no fitted H2 sub-fuser', 'the fusion model has no part fitted for this question'],
  // N-19q (finding 6; `formatting.EARLIER_SERIES_WORDS`): the stored title of the earlier method's per-frame series
  ['Per-frame scores of the earlier method (not in the estimate)', 'Per-frame values of the earlier method (the estimate does not read them)'],
  ['Uncalibrated fallback: probability is not a frequency estimate', 'Uncalibrated fallback: the score is not a frequency estimate'],
  ['Probabilities are statistical estimates from open-source detectors: evidence', 'Scores from open-source detectors are evidence'],
]

const PERCENT = /(?<=\d)%(?![0-9A-Za-z])/g
/** One percent format (the report's `percent_spaced`): the page writes "83 %", a sentence the engine stored "83%". */
export const percentSpaced = (text: string): string => text.replace(PERCENT, ' %')
// the fields of a response that hold a sentence of the engine; a file name, a detector's own observation and an examiner's
// text are quoted as recorded and keep their percent signs
const ENGINE_PROSE = new Set(['reason', 'verbal', 'caveats', 'limiting_conditions', 'note', 'text', 'statement', 'title', 'benchmark_note', 'conditions', 'label', 'dissent_note'])

/** Every sentence of a response as the page prints it (`storedWords`), the engine's sentences in the one percent format. A value that is a slice id and nothing else is a key the page
 *  reads (`slice_id`, the group a validation was measured at) and stays: the page says it with `sliceWords` where it prints it. */
export function sentencesInWords<T>(value: T, field?: string): T {
  if (typeof value === 'string') {
    if (WHOLE.test(value)) return value
    const said = storedWords(value)
    return (field && ENGINE_PROSE.has(field) ? percentSpaced(said) : said) as T
  }
  if (Array.isArray(value)) return value.map((v) => sentencesInWords(v, field)) as T
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, sentencesInWords(v, k)])) as T
  return value
}

/** The keys of the condition profile in words (the report's `CONDITION_WORDS`); another key is printed with spaces. */
const CONDITION_WORDS: Record<string, string> = {
  slice_id: 'Validation slice', jpeg_quality_est: 'Estimated JPEG quality', min_face_px: 'Smallest face (px)', median_face_px: 'Median face (px)',
  duration_s: 'Duration (s)', within_validated_envelope: 'Validated envelope', width: 'Width (px)', height: 'Height (px)',
  screenshot_likelihood: 'Screenshot rule (0 or 1)', validation_measured_at: 'Validation measured at', validation_support: 'Validation support',
  n_frames: 'Frames', speech_share: 'Speech share', music_share: 'Music share', double_compression: 'Double compression', language: 'Language',
}

export const conditionWord = (key: string): string => CONDITION_WORDS[key] ?? key.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase())
