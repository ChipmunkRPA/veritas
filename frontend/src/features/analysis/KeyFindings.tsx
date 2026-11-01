import { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import type { UseQueryResult } from '@tanstack/react-query'
import { api, hasApiKey } from '../../lib/api'
import { useModelCards, type AnalysisDetail, type ExhibitMeta, type Finding, type ModelCard, type SignalV2 } from '../../lib/api/queries'
import { anonymizeFindingText } from './checkNames'
import { cx } from '../../lib/cx'
import { Button, DirectionMark, EmptyState, ErrorState, SkeletonRows, StrengthMeter, type StrengthLevel } from '../../components/ui'
import { FindingRows } from './FindingsList'
import { ASIDE_NOTE, ASIDE_TITLE, OUTSIDE_FINDINGS, contextFindings, contributionLine, isDocumentary, keyFindings, markOf, noWeight, notCounted, provenanceAside, reasonOf, recordPlate, type Decision, type KeyFinding } from './answer'
import { startingPointNote } from './contributions'
import { outsideValidated, recordOf } from './record'
import { NOT_IN_ESTIMATE, VOICE_STEP, findingsFailed, findingsNotWritten, measuredFromNeutral, outsideSentence, outsideWhy } from './vocabulary'
import FindingSupport, { useFindingEvidence, type FindingSupportData } from './FindingSupportPanel'

/** IF §6.2 bands onto the four-bar meter by count (D-16): weak 1, moderate 2, strong 3; decisive is the ◆ seal. The fourth bar
 *  stays empty until a band above "strong" exists, so the meter never says more than the finding's words. */
const METER: Record<Finding['strength'], StrengthLevel> = { none: 'none', weak: 'weak', moderate: 'moderate', strong: 'strong', decisive: 'documentary' }

/** Plate words for findings without a pictorial exhibit: a typographic label, not an illustration. */
const PLATE: Record<string, string> = {
  F1_provenance: 'Credentials', F2_self_declared_metadata: 'Metadata', F3_whole_image_generation: 'Whole image',
  F4_localised_manipulation: 'Region', F5_compression_history: 'Compression', F6_capture_pipeline: 'Capture', F7_duplication: 'Copy-move',
  F8_metadata_consistency: 'Metadata', F10_face: 'Face', F11_physical_scene: 'Scene', V1_face_track: 'Face track', V2_lip_sync: 'Lip-sync',
  V3_synthetic_video: 'Frames', V4_codec: 'Codec', A1_voice: 'Voice', T1_ai_text: 'Text',
}
/** Families about the picture as a whole: without a map of their own, the thumbnail is the source image. */
const WHOLE_IMAGE = new Set(['F3_whole_image_generation', 'F4_localised_manipulation', 'F6_capture_pipeline', 'F10_face', 'F11_physical_scene'])

const DIRECTION_MARK = { for: 'ai', against: 'authentic', neutral: 'neutral' } as const

function Thumb({ k, analysis, exhibits, token, cards, signals }: { k: KeyFinding; analysis: AnalysisDetail; exhibits: readonly ExhibitMeta[]; token: string; cards: readonly ModelCard[]; signals: readonly SignalV2[] }) {
  const f = k.finding
  const heat = exhibits.find((x) => f.evidence_ids?.includes(x.id) && (x.binaries ?? []).includes('preview'))
  const ready = !hasApiKey || token !== ''
  const src = heat
    ? `/api/v2/exhibits/${heat.id}/preview${token ? `?token=${encodeURIComponent(token)}` : ''}`
    : analysis.kind === 'image' && WHOLE_IMAGE.has(f.family) ? api.sourceUrl(analysis.id, token) : null
  if (!src || !ready) return <span data-testid="finding-thumb" data-thumb="plate" aria-hidden
    className="flex w-24 h-[72px] items-center justify-center text-center text-caption font-medium text-text-muted max-[959px]:w-[72px] max-[959px]:h-[54px]">{recordPlate(f) ?? PLATE[f.family] ?? 'Check'}</span>
  return (
    <span data-testid="finding-thumb" data-thumb={src ? (heat ? 'heatmap' : 'source') : 'plate'}
      className="block w-24 h-[72px] rounded-md border border-border bg-surface-2 overflow-hidden max-[959px]:w-[72px] max-[959px]:h-[54px]">
      {/* Maps upscale nearest-neighbour so boundaries never look more precise than the model (UX §8.3). */}
      <img src={src} alt={anonymizeFindingText(f, reasonOf(k)[0], cards, signals)} loading="lazy" className={cx('block w-full h-full object-cover', heat && '[image-rendering:pixelated]')} />
    </span>
  )
}

function Row({ k, position, analysis, signals, exhibits, token, plain, cards, support }: { k: KeyFinding; position: number; analysis: AnalysisDetail; signals: readonly SignalV2[]; exhibits: readonly ExhibitMeta[]; token: string; plain: boolean; cards: readonly ModelCard[]; support: FindingSupportData }) {
  const f = k.finding
  // N-19n (D3-09): outside what was validated no estimate is shown, and no finding says that it raised or lowered one
  const moved = analysis.decision && !outsideValidated(analysis.decision as Decision, analysis.condition_profile)
    ? contributionLine(f, analysis.decision as Decision, plain) : null
  // N-19n (D3-07): a record's mark stands in the meter's place: filled for a documentary record, in outline for a
  // self-declared marker, muted where the decision did not apply it
  const mark = markOf(f, analysis.decision as Decision | null)
  return (
    <li data-testid="key-finding" data-finding={f.id} data-direction={k.direction} data-counter={k.counter ? 'true' : undefined}
      className="grid grid-cols-[40px_96px_minmax(0,1fr)] gap-x-4 min-h-24 py-4 border-b border-border max-[959px]:grid-cols-[40px_72px_minmax(0,1fr)] max-[639px]:grid-cols-[40px_minmax(0,1fr)]">
      {/* The list position can change; the persisted reference below never does. */}
      <span data-testid="key-finding-position" className="pt-0.5 font-mono text-caption text-text-muted num">
        <span className="sr-only">Key finding </span>{position}
      </span>
      <span className="max-[639px]:hidden"><Thumb k={k} analysis={analysis} exhibits={exhibits} token={token} cards={cards} signals={signals} /></span>
      <div className="min-w-0">
        <div className="flex items-center gap-x-3 gap-y-1 flex-wrap">
          <DirectionMark direction={DIRECTION_MARK[k.direction]} showWord={false} />
          {mark
            ? <StrengthMeter level={mark.level} muted={mark.muted} label={mark.word} />
            : <StrengthMeter level={METER[f.strength]} direction={k.direction === 'neutral' ? undefined : k.direction} />}
          <h3 className="m-0 text-body font-medium text-text">
            {anonymizeFindingText(f, f.title, cards, signals)}{k.counter && <span className="font-normal text-text-muted"> — counter-evidence</span>}
          </h3>
        </div>
        {reasonOf(k).map((s) => <p key={s} className="mt-1 m-0 text-body text-text max-w-[72ch]">{anonymizeFindingText(f, s, cards, signals)}</p>)}
        {moved && <p data-testid="key-finding-contribution" className="mt-1 m-0 text-small text-text-muted max-w-[72ch]">{moved}</p>}
        <FindingSupport finding={f} data={support} cards={cards} />
      </div>
    </li>
  )
}

/** L2 key findings (UX §5.4): three to six ranked findings with a thumbnail, direction and strength glyphs, the plain reason and
 *  their evidence ids; the strongest counter-evidence last. The full list, with coverage and conditions, stays one click away. */
/** N-19k (Q2-12; `synthesis._outside_view`): the sentence of a finding outside the estimate whose check scored and whose
 *  signal carries the fusion's record of why it is not counted; null for any other finding. */
export function outsideLine(f: Finding, signals: readonly SignalV2[], kind: string): string | null {
  const signal = (f.evidence_ids ?? []).map((id) => signals.find((s) => s.id === id)).find((s) => s?.outside_estimate != null && s.raw_score != null)
  if (!signal || signal.raw_score == null) return null
  const sentence = outsideSentence(signal.raw_score.toFixed(2), outsideWhy(signal.outside_estimate, f.benchmark_note), f.strength !== 'none')
  return f.family === 'A1_voice' && kind === 'video' ? `${sentence} ${VOICE_STEP}` : sentence
}

export default function KeyFindings({ analysis, query, signals, exhibits, token, plain = false }: {
  analysis: AnalysisDetail
  query: UseQueryResult<Finding[], Error>
  signals: readonly SignalV2[]
  exhibits: readonly ExhibitMeta[]
  token: string
  /** Plain view (N-19d): the size of a check's effect in a word, and no architecture in a check's name */
  plain?: boolean
}) {
  const [all, setAll] = useState(false)
  const { hash } = useLocation()
  const cards = useModelCards().data ?? []
  const evidence = useFindingEvidence(analysis)
  const support: FindingSupportData = { analysisId: analysis.id, signals, ...evidence }
  const items = query.data ?? []
  const keys = analysis.decision ? keyFindings(items, analysis.decision) : []
  // Image evidence links to a precise finding without requiring Expert mode.
  useEffect(() => {
    if (!/^#finding-F[1-9]\d*$/.test(hash)) return
    if (!all) { setAll(true); return }
    document.getElementById(hash.slice(1))?.scrollIntoView({ block: 'center' })
  }, [hash, all, query.data])
  const toward = keys.filter((k) => k.direction === 'for').length
  const against = keys.filter((k) => k.direction === 'against').length
  // D-63: in a result measured from neutral points a check below ln 2 is neither; the header says how many there were
  const little = measuredFromNeutral(analysis.decision) ? keys.filter((k) => k.direction === 'neutral' && !isDocumentary(k.finding) && !notCounted(k.finding, analysis.decision as Decision)).length : 0
  // N-19d (browser QA "Copy"): "2 points toward · 1 point against" read as a score; the line counts findings and says so
  const pointing = `${toward} ${toward === 1 ? 'finding points' : 'findings point'} toward AI`
  const context = analysis.decision ? contextFindings(items, analysis.decision) : []
  const weightless = analysis.decision ? noWeight(analysis.decision as Decision) : false
  // N-19n (D3-07, 3): the records of the file that did not decide this result: no checks, and listed after them
  const aside = analysis.decision ? provenanceAside(items, analysis.decision as Decision) : []
  // D-63: the counts above say what the checks did from their neutral points; where the estimate starts is said beside them
  // N-19n (D3-09): not where the input is outside what was validated: the start's percent stood two blocks above a
  // breakdown that prints none, beside an estimate that is not shown. There the head of the block says, in the place of the
  // counts, that the findings are no reasons here (under a record that decides, the record's finding is one: the counts stay)
  const outside = analysis.decision ? outsideValidated(analysis.decision as Decision, analysis.condition_profile) : false
  const noReasons = outside && recordOf(analysis.decision as Decision) === null
  const startNote = analysis.decision && keys.length > 0 && !outside ? startingPointNote(analysis.decision as Decision) : null
  // D-63: a check outside the estimate carries its stand-alone word and "not in this estimate"; an earlier result keeps its words
  const outsideWords = measuredFromNeutral(analysis.decision) ? NOT_IN_ESTIMATE : 'not in this decision'

  return (
    <section id="section-findings" data-testid="findings" aria-labelledby="findings-heading" className="scroll-mt-14">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <h2 id="findings-heading" className="m-0 eyebrow">Findings and supporting evidence</h2>
        {keys.length > 0 && !noReasons && (
          <p data-testid="key-finding-counts" className="m-0 text-small text-text-muted num">
            {pointing} · {against} against{little > 0 && ` · ${little} had little effect`}
          </p>
        )}
      </div>
      <p className="mt-2 mb-0 text-small text-text-muted max-w-[76ch]">Read what each check found, then open its recorded source locations or check results. Findings are generated automatically.</p>
      {keys.length > 0 && noReasons && <p data-testid="key-findings-outside" className="mt-2 m-0 text-small text-text-muted max-w-[72ch]">{OUTSIDE_FINDINGS}</p>}
      {query.isPending && <SkeletonRows className="mt-4" />}
      {query.error && <ErrorState className="mt-4" onRetry={() => void query.refetch()}>{query.error.message}</ErrorState>}
      {/* N-19f: where the model gives the checks that scored no weight, their findings are no reasons; they stand below.
          N-19n (Q4-01): where the step that writes the findings failed, the place says so and nothing of the evidence;
          the Plain view holds no ledger and names the expert view */}
      {query.isSuccess && keys.length === 0 && <EmptyState className="mt-4">{findingsFailed(analysis.caveats)
        ? findingsNotWritten(plain ? 'expert' : 'ledger')
        : weightless
          ? 'The checks that scored carry no weight for an input like this one, so none of their findings is a reason here. They are listed below.'
          : 'No individual check produced more than weak support either way.'}</EmptyState>}
      {keys.length > 0 && (
        <ol className="mt-3 m-0 p-0 list-none border-t border-border-strong">
          {keys.map((k, index) => <Row key={k.finding.id} k={k} position={index + 1} analysis={analysis} signals={signals} exhibits={exhibits} token={token} plain={plain} cards={cards} support={support} />)}
        </ol>
      )}
      {startNote && <p data-testid="starting-point-note" className="mt-3 m-0 text-small text-text-muted max-w-[72ch]">{startNote}</p>}
      {context.length > 0 && (
        <div data-testid="context-findings" className="mt-4">
          <h3 className="m-0 eyebrow">Context: checks outside the fused estimate</h3>
          {/* N-19f, N-19h: no cause, so none that is untrue of a finding listed here: a check the model holds, with its gate open,
              can have no weight, and a record that was not applied (a marker under a trusted capture) is no check of the model's */}
          <p data-testid="context-findings-note" className="mt-1 m-0 text-small text-text-muted max-w-[72ch]">These findings provide additional context and carry no weight in the automated conclusion.</p>
          <ul className="mt-2 m-0 p-0 list-none">
            {context.map((f) => {
              // N-19k (Q2-12): a check that scored and is not counted says both halves in one sentence, its score and why,
              // from the fusion's record of that check for this input; a finding without that record keeps its two words
              const said = outsideLine(f, signals, analysis.kind)
              return (
                <li key={f.id} data-finding={f.id} className="py-1.5 border-b border-border text-small text-text">
                  <span className="font-mono text-caption text-text-muted num">{f.id}</span> {anonymizeFindingText(f, f.title, cards, signals)}{said == null && <span className="text-text-muted"> · {outsideWords}</span>}
                  {said != null && <p data-testid="outside-why" className="mt-0.5 m-0 text-small text-text-muted max-w-[72ch]">{said}</p>}
                  <FindingSupport finding={f} data={support} cards={cards} />
                </li>
              )
            })}
          </ul>
        </div>
      )}
      {aside.length > 0 && (
        <div data-testid="provenance-aside" className="mt-4">
          <h3 className="m-0 eyebrow">{ASIDE_TITLE}</h3>
          <p className="mt-1 m-0 text-small text-text-muted max-w-[72ch]">{ASIDE_NOTE}</p>
          <ul className="mt-2 m-0 p-0 list-none">
            {aside.map((f) => {
              // each row carries its diamond and no meter; a record without a mark of its own (credentials that do not
              // validate) takes the muted outline of a record that was not applied
              const mark = markOf(f, analysis.decision as Decision | null)
              return (
                <li key={f.id} data-finding={f.id} className="flex items-center gap-x-3 gap-y-1 flex-wrap py-1.5 border-b border-border text-small text-text">
                  <StrengthMeter level={mark?.level ?? 'self_declared'} muted={mark?.muted ?? true} label={mark?.word ?? 'not applied'} />
                  <span><span className="font-mono text-caption text-text-muted num">{f.id}</span> {anonymizeFindingText(f, f.title, cards, signals)}</span>
                  <FindingSupport finding={f} data={support} cards={cards} />
                </li>
              )
            })}
          </ul>
        </div>
      )}
      {(items.length > keys.length || all) && (
        <div className="mt-3">
          <Button variant="text" aria-expanded={all} aria-controls="all-findings" onClick={() => setAll(!all)}>
            {all ? 'Hide the full list' : `All ${items.length} findings, with coverage and conditions`}
          </Button>
          {all && <div id="all-findings" className="mt-3"><FindingRows items={items} decision={analysis.decision} support={support} /></div>}
        </div>
      )}
    </section>
  )
}
