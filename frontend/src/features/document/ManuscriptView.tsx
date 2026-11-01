import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import type { SentenceOut } from '../../lib/api'
import { prob } from '../../lib/format'
import { cx } from '../../lib/cx'
import { Button, Eyebrow } from '../../components/ui'
import { useDemoAction, type StepAction } from '../../demo/types'
import { sentenceScoreRows } from './sentenceScores'

type Dim = 'none' | 'non-ai' | 'non-mixed'
const isFlagged = (s: SentenceOut) => s.band === 'ai' || s.band === 'mixed'
const bandWord = (b: string) => (b === 'ai' ? 'AI-like' : b === 'mixed' ? 'mixed' : 'human-like')
const bandText = (b: string) => (b === 'ai' ? 'text-verdict-5' : b === 'mixed' ? 'text-verdict-3' : 'text-verdict-1')
const bandRule = (b: string) => (b === 'ai' ? 'border-hl-ai-rule' : b === 'mixed' ? 'border-hl-mixed-rule' : 'border-verdict-1')

/** The document manuscript (ARCHITECTURE §7.1 `document/`, evolves the v1 TextView): every sentence tinted and ruled by band, with the
 *  selected sentence's evidence in the rail. Shared by v2 document analyses and the v1 legacy result.
 *  `noted` and `notes` (N-19a): the note numbers of each sentence, shown after it, and the notes block of the rail for the
 *  selected sentence, which the v2 workspace supplies. */
export default function ManuscriptView({ sentences, text, paragraphs, noted, notes }: {
  sentences: SentenceOut[]; text: string; paragraphs: [number, number][]
  noted?: ReadonlyMap<number, readonly string[]>
  notes?: (sentenceIdx: number) => ReactNode
}) {
  const [selIdx, setSelIdx] = useState<number | null>(null)
  const [dim, setDim] = useState<Dim>('none')
  const [numbers, setNumbers] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  // Extraction anchors count Unicode codepoints; JS string.slice counts UTF-16
  // code units, which shifts every later sentence after an emoji/non-BMP letter.
  const sourceCharacters = useMemo(() => Array.from(text), [text])
  const sourceExcerpt = (s: SentenceOut) => sourceCharacters.slice(s.start, s.end).join('')
  const byPar = useMemo(() => {
    const m = new Map<number, SentenceOut[]>()
    sentences.forEach((s) => { if (!m.has(s.paragraph)) m.set(s.paragraph, []); m.get(s.paragraph)!.push(s) })
    return m
  }, [sentences])
  const flagged = useMemo(() => sentences.filter(isFlagged).map((s) => s.idx), [sentences])
  const sel = selIdx == null ? null : sentences.find((s) => s.idx === selIdx) ?? null
  const scoreRows = sel ? sentenceScoreRows(sel.scores) : []

  const focusSentence = (idx: number) => rootRef.current?.querySelector<HTMLElement>(`[data-idx="${idx}"]`)?.focus({ preventScroll: false })
  const step = useCallback((dir: 1 | -1) => {
    if (!flagged.length) return
    const pos = selIdx == null ? -1 : flagged.indexOf(selIdx)
    let next: number
    if (pos === -1) next = dir === 1 ? flagged[0] : flagged[flagged.length - 1]
    else next = flagged[(pos + dir + flagged.length) % flagged.length]
    setSelIdx(next)
    focusSentence(next)
  }, [flagged, selIdx])

  // `[` / `]` and J / K step through flagged sentences when focus is not in a form field.
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName) || t.isContentEditable || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === ']' || e.key === 'j' || e.key === 'J') { e.preventDefault(); step(1) }
      else if (e.key === '[' || e.key === 'k' || e.key === 'K') { e.preventDefault(); step(-1) }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [step])

  // Guided demo: select the first AI-highlighted sentence (else mixed, else the first scored one) so the rail fills.
  useDemoAction(useCallback((a: StepAction) => {
    if (a.type !== 'select-first-ai-sentence') return
    const pick = sentences.find((s) => s.band === 'ai') ?? sentences.find((s) => s.band === 'mixed') ?? sentences.find((s) => s.p_ai != null)
    if (pick) setSelIdx(pick.idx)
  }, [sentences]))

  const dimmed = (s: SentenceOut) => (dim === 'non-ai' && s.band !== 'ai') || (dim === 'non-mixed' && s.band !== 'mixed')
  const cls = (s: SentenceOut) => cx('sent', s.p_ai == null ? 'sent-unscored' : s.band === 'ai' ? 'sent-ai' : s.band === 'mixed' ? 'sent-mixed' : 'sent-human', selIdx === s.idx && 'sent-selected', dimmed(s) && 'sent-dim')
  const onSentKey = (e: KeyboardEvent<HTMLSpanElement>, s: SentenceOut) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelIdx(s.idx) }
  }
  const dimBtn = (d: Dim, label: string) => (
    <button key={d} type="button" aria-pressed={dim === d} onClick={() => setDim(d)} className={cx('text-small min-h-8 px-1 rounded-sm transition-colors duration-fast', dim === d ? 'text-text underline underline-offset-[3px]' : 'text-text-muted hover:text-text')}>{label}</button>
  )
  const flaggedPos = sel && isFlagged(sel) ? flagged.indexOf(sel.idx) + 1 : null

  return (
    <section aria-label="Manuscript" data-tour="manuscript">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <Eyebrow as="h2">Manuscript</Eyebrow>
        <div className="flex items-center gap-5 flex-wrap">
          <span className="inline-flex items-center gap-2 text-small text-text"><span aria-hidden className="inline-block w-3 h-3 bg-hl-ai border-b-2 border-hl-ai-rule" style={{ borderRadius: 2 }} />AI-like</span>
          <span className="inline-flex items-center gap-2 text-small text-text"><span aria-hidden className="inline-block w-3 h-3 bg-hl-mixed border-b border-dotted border-hl-mixed-rule" style={{ borderRadius: 2 }} />mixed</span>
          <span className="inline-flex items-center gap-2 text-small text-text"><span aria-hidden className="inline-block w-3 h-3 border border-border" style={{ borderRadius: 2 }} />human-like</span>
          <span className="text-small text-text-muted ml-2">Dim:</span>
          {dimBtn('none', 'none')}{dimBtn('non-ai', 'non-AI-like')}{dimBtn('non-mixed', 'non-mixed')}
          <button type="button" aria-pressed={numbers} onClick={() => setNumbers((n) => !n)} className={cx('text-small min-h-8 px-1 rounded-sm ml-2', numbers ? 'text-text underline underline-offset-[3px]' : 'text-text-muted hover:text-text')}>Numbers</button>
        </div>
      </div>

      <p className="mt-2 mb-0 text-small text-text-muted">Sentence scores are uncalibrated diagnostic values, not probabilities of authorship.</p>

      <div ref={rootRef} className="mt-4 grid gap-12 min-[960px]:grid-cols-[minmax(0,68ch)_320px]">
        <div className="min-w-0">
          {paragraphs.map((_, pi) => {
            const ss = byPar.get(pi) ?? []
            if (!ss.length) return null
            return (
              <div key={pi} className="grid grid-cols-[40px_1fr] mb-5">
                <span aria-hidden className="font-mono text-caption text-text-muted pt-2 select-none">¶{pi + 1}</span>
                <p className="m-0 reader text-reader text-text">
                  {ss.map((s) => (
                    <span key={s.idx}>
                      {numbers && <span aria-hidden className="font-mono text-micro text-text-muted align-super mr-0.5 select-none">{s.idx + 1}</span>}
                      <span data-idx={s.idx} role="button" tabIndex={s.p_ai == null ? -1 : 0} className={cls(s)} onClick={() => s.p_ai != null && setSelIdx(s.idx)} onKeyDown={(e) => onSentKey(e, s)}
                        aria-pressed={selIdx === s.idx} aria-label={s.p_ai == null ? `Sentence ${s.idx + 1}, not scored` : `Sentence ${s.idx + 1}, ${bandWord(s.band)} band, uncalibrated score ${prob(s.p_ai)}`}
                        title={s.p_ai == null ? 'not scored' : undefined}>
                        {sourceExcerpt(s)}
                      </span>
                      {noted?.get(s.idx)?.length ? (
                        <span data-testid="sentence-mark" className="font-mono text-micro text-annotation align-super ml-0.5 select-none"
                          title={`${noted.get(s.idx)!.length === 1 ? 'Note' : 'Notes'} ${noted.get(s.idx)!.join(', ')} on this sentence`}>{noted.get(s.idx)!.join(' ')}</span>
                      ) : null}{' '}
                    </span>
                  ))}
                </p>
              </div>
            )
          })}
        </div>

        {/* ≥960: sticky rail beside the manuscript. <960: the rail would stack below the whole text, so the selected
            sentence renders as a bottom sheet (40 vh, hairline top) and the empty prompt is not shown at all. */}
        <aside aria-live="polite" data-tour="rail" className={cx(
          'border-l border-border pl-6 min-[960px]:sticky min-[960px]:top-[72px] min-[960px]:self-start min-[960px]:max-h-[calc(100vh-96px)] min-[960px]:overflow-auto',
          sel
            ? 'max-[959px]:fixed max-[959px]:bottom-0 max-[959px]:left-12 max-[959px]:right-0 max-[959px]:z-30 max-[959px]:max-h-[40vh] max-[959px]:overflow-auto max-[959px]:border-l-0 max-[959px]:border-t max-[959px]:bg-surface max-[959px]:px-6 max-[959px]:py-4 max-[639px]:left-0'
            : 'max-[959px]:hidden',
        )}>
          {!sel ? (
            <p className="reader text-reader text-text-muted">Select a sentence to see its evidence.</p>
          ) : (
            <div key={sel.idx} className="animate-fade-in">
              <div className="flex items-center justify-between gap-4">
                <Eyebrow>Sentence {sel.idx + 1} · ¶{sel.paragraph + 1}</Eyebrow>
                <Button variant="text" className="hidden max-[959px]:inline-flex" onClick={() => setSelIdx(null)}>Close</Button>
              </div>
              <blockquote className={cx('m-0 mt-2 pl-3 border-l-[3px] reader text-reader italic text-text', bandRule(sel.band))}>{sourceExcerpt(sel)}</blockquote>
              <dl className="mt-4 m-0 grid grid-cols-[72px_1fr] gap-x-4 gap-y-1 text-small">
                <dt className="text-text-muted">Band</dt><dd className={cx('m-0 font-medium', bandText(sel.band))}>{bandWord(sel.band)}</dd>
                <dt className="text-text-muted">Score</dt><dd className="m-0 font-mono text-mono num text-text">{prob(sel.p_ai)}</dd>
                {sel.flags && sel.flags.length > 0 && (<><dt className="text-text-muted">Flags</dt><dd className="m-0 font-mono text-mono text-text">{sel.flags.join(', ')}</dd></>)}
              </dl>
              {sel.reasons.length > 0 && (
                <div className="mt-4 pt-4 border-t border-border">
                  <Eyebrow as="h3">Why</Eyebrow>
                  <ul className="mt-2 m-0 p-0 list-none space-y-1.5">
                    {sel.reasons.map((r, i) => <li key={i} className="grid grid-cols-[12px_1fr] text-small text-text"><span aria-hidden className="text-text-faint">·</span><span>{r}</span></li>)}
                  </ul>
                </div>
              )}
              {notes?.(sel.idx)}
              {scoreRows.length > 0 && (
                <div className="mt-4 pt-4 border-t border-border">
                  <Eyebrow as="h3">Scores</Eyebrow>
                  <table className="mt-2 w-full border-collapse"><tbody>
                    {scoreRows.map(({ key, label, value }) => (
                      <tr key={key} className="border-b border-border last:border-b-0"><td className="py-1 text-small text-text-muted">{label}</td><td className="py-1 text-right font-mono text-mono text-text num">{value.toFixed(3)}</td></tr>
                    ))}
                  </tbody></table>
                  <p className="mt-2 mb-0 text-caption text-text-muted">Raw diagnostic values; ratios and responses use different scales. None is a probability of authorship.</p>
                </div>
              )}
            </div>
          )}
          {flagged.length > 0 && (
            <div className="mt-6 pt-4 border-t border-border flex items-center justify-between text-small">
              <Button variant="text" onClick={() => step(-1)} aria-keyshortcuts="[ k">‹ prev flagged</Button>
              <span className="font-mono text-caption text-text-muted num">{flaggedPos ? `${flaggedPos} / ${flagged.length}` : `${flagged.length} flagged`}</span>
              <Button variant="text" onClick={() => step(1)} aria-keyshortcuts="] j">next flagged ›</Button>
            </div>
          )}
        </aside>
      </div>
    </section>
  )
}
