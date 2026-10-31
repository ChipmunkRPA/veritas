import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Moon, Sun } from 'lucide-react'
import { useTheme } from '../lib/theme'
import { INFERNO_STOPS, SHOW_THRESHOLD_OVERLAY, buildLut, colourise } from '../lib/colormap'
import { cx } from '../lib/cx'
import {
  Button, ConfidenceMeter, DirectionMark, EmptyState, ErrorState, HatchPattern, HatchSwatch, HeatLegend, IconArray, IdChip, Lane,
  LaneAxis, NotRun, ReviewStateMark, ScaleGlyph, ScaleInterval, SeverityMark, Skeleton, SkeletonRows, StrengthMeter,
  strengthFromLikelihoodRatio, useSvgId, type StrengthLevel,
} from '../components/ui'
import type { UseQueryResult } from '@tanstack/react-query'
import HowReached from '../features/analysis/ContributionWaterfall'
import KeyFindings from '../features/analysis/KeyFindings'
import { SCALE_NOTE_MARKER } from '../features/analysis/resultNumber'
import { C2paChain, CodecFacts, MetadataTree } from '../features/analysis/ProvenancePanel'
import { chainNodes, codecFacts, fields, treeGroups } from '../features/analysis/provenance'
import {
  SPECIMEN_BROKEN_DECISION, SPECIMEN_CHAIN, SPECIMEN_CODEC, SPECIMEN_CREDENTIALS, SPECIMEN_DECISION, SPECIMEN_FINDINGS, SPECIMEN_MODEL_CARDS, SPECIMEN_RUNS,
  SPECIMEN_SIGNALS, SPECIMEN_TREE, SPECIMEN_VALIDATION, specimenAnalysis,
} from '../features/analysis/specimens'
import type { Finding } from '../lib/api/queries'
import MethodsPanel from '../features/analysis/MethodsPanel'
import SignalsLedger from '../features/signals/SignalsLedger'

/*
 * /design: the v2 design contract as a living gallery (BUILD_SPEC E-0, UX §8). Every new evidence primitive in the
 * states it can take, in the current theme. Specimen data is synthetic and labelled; nothing here is an analysis result.
 */

const SECTIONS = [
  ['tokens', 'Tokens'], ['scale-glyph', 'Verdict glyph'], ['scale-interval', 'Scale with interval'], ['strength', 'Strength'],
  ['direction', 'Direction'], ['severity', 'Severity'], ['review-state', 'Review state'], ['chips', 'Identity chips'],
  ['hatch', 'Hatch'], ['heat', 'Heat map'], ['lanes', 'Timeline lanes'], ['icon-array', 'Icon array'], ['evidence', 'Evidence panels'],
  ['states', 'Section states'],
] as const

const TOKEN_GROUPS: { group: string; stage?: boolean; tokens: readonly (readonly [string, string])[] }[] = [
  { group: 'Direction', tokens: [['--color-dir-ai', 'for AI generation or manipulation · filled dot'], ['--color-dir-authentic', 'against · ring'], ['--color-dir-neutral', 'little effect · small dot']] },
  { group: 'Strength', tokens: [['--color-strength-on', 'bar on'], ['--color-strength-off', 'bar off']] },
  { group: 'Graphics', tokens: [['--color-graphic', 'axes, connectors, interval ticks, lane frames'], ['--color-interval', 'plausible range inside the scale']] },
  { group: 'Severity', tokens: [['--color-sev-info', '○ info'], ['--color-sev-notable', '◐ notable'], ['--color-sev-anomaly', '● anomaly'], ['--color-sev-documentary', '◆ documentary']] },
  { group: 'Lanes', tokens: [['--color-lane', 'lane ground'], ['--color-lane-frame', 'bottom frame'], ['--color-lane-hatch', 'not analysed texture'], ['--color-lane-flag', 'flag rule at step 4–5'], ['--color-lane-tolerance', 'lip-sync tolerance band']] },
  { group: 'Examiner’s marks', tokens: [['--color-annotation', 'notes, N-chips'], ['--color-selection', 'selected segment outline']] },
  { group: 'Lenses', stage: true, tokens: [['--color-lens-ink', 'marks over media'], ['--color-lens-halo', 'halo under marks'], ['--color-lens-clone', 'copy-move pairs'], ['--color-lens-hatch', 'unreliable areas']] },
]
const TOKEN_NAMES = [...TOKEN_GROUPS.flatMap((g) => g.tokens.map(([n]) => n)), ...INFERNO_STOPS.map((_, i) => `--heat-${String(i).padStart(2, '0')}`)]

const LEVELS: StrengthLevel[] = ['none', 'weak', 'moderate', 'strong', 'very_strong', 'documentary', 'self_declared']
const SPECIMEN_LRS = [1.2, 2.4, 6, 40, 250, 1 / 40]
const LUT = buildLut()

function Section({ id, title, spec, children }: { id: string; title: string; spec: ReactNode; children: ReactNode }) {
  return (
    <section id={id} data-gallery={id} aria-labelledby={`${id}-title`} className="mt-16 scroll-mt-6">
      <h2 id={`${id}-title`} className="display text-h2 font-normal text-text">{title}</h2>
      <p className="mt-1 text-small text-text-muted max-w-(--container-reader)">{spec}</p>
      <div className="mt-5 border-t border-border-strong">{children}</div>
    </section>
  )
}

function Row({ state, label, note, children }: { state: string; label?: string; note?: ReactNode; children: ReactNode }) {
  return (
    <div data-row data-state={state} className="grid grid-cols-[160px_minmax(0,1fr)] gap-x-6 py-5 border-b border-border max-[639px]:grid-cols-1 max-[639px]:gap-y-2">
      <span className="eyebrow pt-0.5">{label ?? state}</span>
      <div className="min-w-0">
        {children}
        {note && <p className="mt-3 text-caption text-text-muted max-w-(--container-reader)">{note}</p>}
      </div>
    </div>
  )
}

function useTokenValues(theme: string) {
  const [values, setValues] = useState<Record<string, string>>({})
  useEffect(() => {
    const cs = getComputedStyle(document.documentElement)
    setValues(Object.fromEntries(TOKEN_NAMES.map((n) => [n, cs.getPropertyValue(n).trim()])))
  }, [theme])
  return values
}

function Swatch({ name, stage }: { name: string; stage?: boolean }) {
  return (
    <span aria-hidden className={cx('block w-8 h-4 rounded-xs border border-border overflow-hidden', stage && 'bg-stage')}>
      <span className="block w-full h-full" style={{ background: `var(${name})` }} />
    </span>
  )
}

/** Synthetic 8-bit map: two soft blobs over a smooth low field, upscaled nearest-neighbour (§8.3). No periodic texture: it would read as the reliability hatch. */
function specimenMap(w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h)
  const blobs = [[0.32, 0.4, 0.16, 1], [0.72, 0.62, 0.11, 0.82]] as const
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = 0.04 + (0.1 * y) / h
      for (const [bx, by, r, a] of blobs) v = Math.max(v, a * Math.exp(-((x / w - bx) ** 2 + (y / h - by) ** 2) / (r * r)))
      out[y * w + x] = Math.round(Math.min(1, v) * 255)
    }
  }
  return out
}

function HeatSpecimen({ show, opacity, mapOnly = false, hatchRight = false }: { show: number; opacity: number; mapOnly?: boolean; hatchRight?: boolean }) {
  const base = useRef<HTMLCanvasElement>(null)
  const heat = useRef<HTMLCanvasElement>(null)
  const hatchId = useSvgId('heat-hatch')
  useEffect(() => {
    const W = 80
    const H = 60
    const b = base.current?.getContext('2d')
    const h = heat.current?.getContext('2d')
    if (!b || !h) return
    // Stand-in for the source photograph: flat grey bands.
    for (let y = 0; y < H; y++) {
      const g = 72 + Math.round((64 * y) / H)
      b.fillStyle = `rgb(${g} ${g} ${g})`
      b.fillRect(0, y, W, 1)
    }
    const img = h.createImageData(W, H)
    colourise(specimenMap(W, H), LUT, { show, opacity }, img.data)
    h.putImageData(img, 0, 0)
  }, [show, opacity])
  const pixelated = { imageRendering: 'pixelated' as const }
  return (
    <div className={cx('relative w-[320px] h-[240px] max-w-full rounded-md overflow-hidden', mapOnly ? 'bg-heat-ground' : 'bg-stage')}>
      <canvas ref={base} width={80} height={60} aria-hidden className={cx('absolute inset-0 w-full h-full', mapOnly && 'invisible')} style={pixelated} />
      <canvas ref={heat} data-specimen="heat" width={80} height={60} aria-hidden className="absolute inset-0 w-full h-full" style={pixelated} />
      {hatchRight && (
        <svg aria-hidden className="absolute inset-0 w-full h-full">
          <defs><HatchPattern id={hatchId} tone="lens" /></defs>
          <rect x="72%" y={0} width="28%" height="100%" fill={`url(#${hatchId})`} />
        </svg>
      )}
    </div>
  )
}

export default function DesignGallery() {
  const { theme, toggle } = useTheme()
  const values = useTokenValues(theme)
  const [ready, setReady] = useState(false)
  const [status, setStatus] = useState('')
  useEffect(() => { setReady(true) }, [])
  const act = (what: string) => () => setStatus(`${what} pressed (specimen)`)

  const estimateSegments = [{ start: 9.8, end: 11.2, step: 3, id: 'a' }, { start: 12.4, end: 15.9, step: 5, id: 'b' }, { start: 15.9, end: 17.2, step: 4 }] as const
  const faceSegments = [{ start: 9.8, end: 11.2, step: 4, id: 'a' }, { start: 12.4, end: 15.9, step: 5, id: 'b' }] as const

  return (
    <div className="min-h-screen bg-surface text-text">
      <header className="border-b border-border">
        <div className="max-w-(--container-content) mx-auto h-14 px-12 flex items-center justify-between max-[959px]:px-6">
          <Link to="/" className="display text-[22px] leading-7 font-medium text-text">Veritas</Link>
          <Button variant="icon" data-testid="gallery-theme" onClick={toggle} aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} title="Theme">
            {theme === 'dark' ? <Sun size={16} strokeWidth={1.5} aria-hidden /> : <Moon size={16} strokeWidth={1.5} aria-hidden />}
          </Button>
        </div>
      </header>

      <main data-gallery-ready={ready ? 'true' : 'false'} className="max-w-(--container-content) mx-auto px-12 pt-12 pb-24 max-[959px]:px-6">
        <h1 className="display text-display font-normal text-text">Design contract</h1>
        <p className="mt-3 text-body text-text-muted max-w-(--container-reader)">
          Evidence primitives for the v2 workspace and reports, each in the states it can take (UX §8). Specimen values are synthetic.
          Current theme: <span className="font-mono text-mono">{theme}</span>.
        </p>
        <nav aria-label="Sections" className="mt-6 flex flex-wrap gap-x-4 gap-y-1 text-small">
          {SECTIONS.map(([id, title]) => <a key={id} href={`#${id}`} className="text-accent underline-offset-[3px] hover:underline">{title}</a>)}
        </nav>
        <p aria-live="polite" className="sr-only">{status}</p>

        <Section id="tokens" title="Tokens" spec="frontend/src/index.css @theme static; the light values are exported to the report CSS with make report-tokens. Values below are read from the page.">
          {TOKEN_GROUPS.map((g) => (
            <Row key={g.group} state="tokens" label={g.group}>
              <table className="w-full table-fixed border-collapse text-small">
                <colgroup><col className="w-12" /><col className="w-[232px]" /><col className="w-[200px]" /><col /></colgroup>
                <tbody>
                  {g.tokens.map(([name, use]) => (
                    <tr key={name} className="h-8">
                      <td className="w-12 pr-3"><Swatch name={name} stage={g.stage} /></td>
                      <td className="pr-4 font-mono text-mono text-text whitespace-nowrap">{name}</td>
                      <td className="pr-4 font-mono text-mono text-text-muted whitespace-nowrap num">{values[name] ?? ''}</td>
                      <td className="text-text-muted max-[1099px]:hidden">{use}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Row>
          ))}
          <Row state="tokens" label="Heat" note="Inferno, 11 stops, monotonic in L*; lib/colormap.ts interpolates the 256-entry LUT.">
            <div className="flex flex-wrap gap-0.5">
              {INFERNO_STOPS.map((hex, i) => {
                const n = String(i).padStart(2, '0')
                return (
                  <span key={hex} className="flex flex-col items-center gap-1" title={values[`--heat-${n}`]}>
                    <span aria-hidden className="block w-10 h-5" style={{ background: `var(--heat-${n})` }} />
                    <span className="font-mono text-micro text-text-muted">{n}</span>
                  </span>
                )
              })}
            </div>
          </Row>
        </Section>

        <Section id="scale-glyph" title="Verdict glyph" spec="Five cells filled to the band step; step 0 is insufficient evidence (UX §9).">
          <Row state="ready" label="steps 1–5">
            <div className="flex flex-wrap items-center gap-6">
              {([1, 2, 3, 4, 5] as const).map((s) => <span key={s} className="inline-flex items-center gap-2 text-small text-text-muted"><ScaleGlyph step={s} /><span className="font-mono text-mono">{s}</span></span>)}
            </div>
          </Row>
          <Row state="insufficient" label="step 0"><ScaleGlyph step={0} /></Row>
          <Row state="loading"><Skeleton w={68} h={6} /></Row>
        </Section>

        <Section id="scale-interval" title="Scale with interval" spec="6 px track, plausible range at 2 px with 10 px end ticks, 2×14 marker (UX §8.4).">
          <Row state="ready" note="Specimen: estimate 91, range 85–95. Under a calibrated result the caption says what the marker is, in place of the three segment labels (D-63).">
            <div className="max-w-[480px]"><ScaleInterval step={5} estimate={91} low={85} high={95} note={SCALE_NOTE_MARKER} /></div>
          </Row>
          <Row state="headline-by-level" label="headline by level" note="Specimen: estimate 92 under the headline “Indications”. The marker stands in the last segment; the headline follows the calibrated levels, not the segment (D-63).">
            <div className="max-w-[480px]"><ScaleInterval step={4} estimate={92} low={85} high={96} note={SCALE_NOTE_MARKER} /></div>
          </Row>
          <Row state="low-confidence" label="low agreement" note="The range is wide; the headline gains “— the checks disagree” and links to the explanation (§8.5).">
            <div className="max-w-[480px]"><ScaleInterval step={3} estimate={58} low={22} high={90} /></div>
          </Row>
          <Row state="partial" note="A stage did not run: the headline is unchanged, agreement is capped at medium (§8.5).">
            <div className="max-w-[480px]"><ScaleInterval step={4} estimate={74} low={60} high={84} /></div>
            <div className="mt-4"><NotRun what="Localisation" reason="model not installed" /></div>
          </Row>
          <Row state="insufficient" note="Empty track and no marker; findings are replaced by the reasons list.">
            <div className="max-w-[480px]"><ScaleInterval step={0} estimate={null} /></div>
          </Row>
          <Row state="band" label="uncalibrated score" note="A score that is not a calibrated likelihood has no point on this scale: the verdict’s segment is filled and no marker is drawn. The same form serves a result outside what was validated.">
            <div className="max-w-[480px]"><ScaleInterval step={4} band label="Score 85 / 100, not calibrated" /></div>
          </Row>
          <Row state="band" label="inconclusive" note="An inconclusive result, or one in which a single check scored: the range stays, the marker does not, and the estimate is body text.">
            <div className="max-w-[480px]"><ScaleInterval step={3} band low={53} high={78} label="Plausible range 53–78: inconclusive" /></div>
          </Row>
          <Row state="loading"><div className="max-w-[480px]"><ScaleInterval step={3} loading /></div></Row>
        </Section>

        <Section id="strength" title="Strength" spec="Achromatic and count-encoded: 0–4 bars from the likelihood ratio in either direction; documentary evidence is a ◆ seal (UX §8.1).">
          <Row state="ready" label="levels">
            <ul className="m-0 p-0 list-none flex flex-col gap-2">
              {LEVELS.map((l) => <li key={l}><StrengthMeter level={l} direction="for" showWord /></li>)}
            </ul>
          </Row>
          <Row state="ready" label="from ratio" note="Band edges < 1.5 · 3 · 10 · 100, to be confirmed (ENFSI), D-16.">
            <ul className="m-0 p-0 list-none grid grid-cols-[88px_1fr] gap-x-4 gap-y-2">
              {SPECIMEN_LRS.map((lr) => (
                <li key={lr} className="contents">
                  <span className="font-mono text-mono text-text-muted num">LR {lr >= 1 ? lr : `1/${Math.round(1 / lr)}`}</span>
                  <StrengthMeter level={strengthFromLikelihoodRatio(lr)} direction={lr >= 1 ? 'for' : 'against'} showWord />
                </li>
              ))}
            </ul>
          </Row>
          <Row state="ready" label="confidence" note="The confidence / agreement meter stays at three bars.">
            <div className="flex items-center gap-6 text-small text-text-muted">
              {(['low', 'medium', 'high'] as const).map((w, i) => <span key={w} className="inline-flex items-center gap-2"><ConfidenceMeter level={(i + 1) as 1 | 2 | 3} />{w}</span>)}
            </div>
          </Row>
          <Row state="loading"><Skeleton w={30} h={12} /></Row>
        </Section>

        <Section id="direction" title="Direction" spec="Hue is never the only channel: filled dot for, ring against, small dot for little effect; position carries it in charts (UX §8.2).">
          <Row state="ready" label="marks">
            <div className="flex flex-wrap items-center gap-6">
              <DirectionMark direction="ai" /><DirectionMark direction="authentic" /><DirectionMark direction="neutral" />
            </div>
          </Row>
          <Row state="ready" label="glyph only" note="Dense rows show the glyph; the words stay available to screen readers and in the title.">
            <div className="flex items-center gap-4">
              <DirectionMark direction="ai" showWord={false} /><DirectionMark direction="authentic" showWord={false} /><DirectionMark direction="neutral" showWord={false} />
            </div>
          </Row>
        </Section>

        <Section id="severity" title="Severity" spec="Metadata anomalies: glyph and word always (UX §8.1).">
          <Row state="ready" label="marks">
            <ul className="m-0 p-0 list-none flex flex-col gap-2">
              <li><SeverityMark severity="info">camera model recorded</SeverityMark></li>
              <li><SeverityMark severity="notable">created and modified dates differ</SeverityMark></li>
              <li><SeverityMark severity="anomaly">software field names an image editor</SeverityMark></li>
              <li><SeverityMark severity="documentary">Content Credentials declare AI generation</SeverityMark></li>
            </ul>
          </Row>
          <Row state="empty"><EmptyState action="Show all fields" onAction={act('Show all fields')}>No anomalies recorded.</EmptyState></Row>
        </Section>

        <Section id="review-state" title="Review state" spec="Visible on every surface and every PDF page; the record is append-only (UX §5.14, P7).">
          <Row state="ready" label="states">
            <ul className="m-0 p-0 list-none flex flex-col gap-2">
              <li><ReviewStateMark state="machine" /></li>
              <li><ReviewStateMark state="in_review">In review — R. Okafor</ReviewStateMark></li>
              <li><ReviewStateMark state="reviewed">Reviewed by R. Okafor, 11 Sep 2026 — opinion: concurs with the machine assessment</ReviewStateMark></li>
              <li><ReviewStateMark state="approved">Reviewed by D. Whitfield, 11 Sep 2026 · second review by M. Lee, 12 Sep 2026</ReviewStateMark></li>
              <li><ReviewStateMark state="disputed">Disputed · signed by D. Whitfield, 11 Sep 2026 · second review by M. Lee, 12 Sep 2026 — peer review: does not concur with the examiner’s opinion</ReviewStateMark></li>
              <li><ReviewStateMark state="superseded">Superseded by analysis 91c2e0a4 (Veritas 2.1.0)</ReviewStateMark></li>
              <li><ReviewStateMark state="withdrawn">Withdrawn — reason recorded in the custody log</ReviewStateMark></li>
            </ul>
          </Row>
        </Section>

        <Section id="chips" title="Identity chips" spec="Identity is a letter or number, identical in UI and PDF; the examiner’s marks use the accent, evidence never does (UX §8.1–§8.2).">
          <Row state="ready" label="evidence ids">
            <div className="flex flex-wrap items-center gap-2">
              {['F1', 'F2', 'S04', 'Fig. 3', 'R1', 'Seg. b', 'Face A'].map((c) => <IdChip key={c}>{c}</IdChip>)}
            </div>
          </Row>
          <Row state="ready" label="examiner’s marks">
            <div className="flex flex-wrap items-center gap-2"><IdChip analyst>N1</IdChip><IdChip analyst>N2</IdChip></div>
          </Row>
        </Section>

        <Section id="hatch" title="Hatch" spec="45°, 6 px pitch, 1 px lines: not analysed, not run and unreliable; always with a label (UX §8.3–§8.4).">
          <Row state="partial" label="not analysed"><HatchSwatch label={<span className="text-small text-text">not analysed · music, no speech</span>} /></Row>
          <Row state="ready" label="over media">
            <div className="inline-flex rounded-md bg-stage px-4 py-3"><HatchSwatch tone="lens" label={<span className="text-small text-stage-text">unreliable</span>} /></div>
          </Row>
          <Row state="ready" label="adjustment"><HatchSwatch tone="graphic" label={<span className="text-small text-text">adjustment (contribution waterfall)</span>} /></Row>
        </Section>

        <Section id="heat" title="Heat map" spec="Inferno LUT; transparent below Show ≥, alpha from 35 % to the chosen opacity above it; nearest-neighbour upscaling; hatched = unreliable (UX §8.3).">
          <Row state="ready" label="overlay" note={`Synthetic specimen map at Show ≥ ${SHOW_THRESHOLD_OVERLAY}, opacity 0.8, right strip hatched as unreliable.`}>
            <div className="flex flex-wrap items-end gap-6">
              <HeatSpecimen show={SHOW_THRESHOLD_OVERLAY} opacity={0.8} hatchRight />
              <div className="rounded-md bg-stage px-4 py-3"><HeatLegend onStage title="anomaly score (specimen)" integrity="0.62" /></div>
            </div>
          </Row>
          <Row state="ready" label="map only" note="Map on the black ground in both themes, Show ≥ 0.">
            <div className="flex flex-wrap items-end gap-6">
              <HeatSpecimen show={0} opacity={1} mapOnly />
              <HeatLegend title="anomaly score (specimen)" showUnreliable={false} />
            </div>
          </Row>
          <Row state="ready" label="attention" note="Same ramp at up to 60 % opacity, with the Attention badge; cannot be shown with localisation heat.">
            <HeatLegend attention title="where the classifier looked (specimen)" showUnreliable={false} />
          </Row>
          <Row state="loading"><Skeleton w={320} h={240} className="rounded-md max-w-full" /></Row>
          <Row state="error"><ErrorState onRetry={act('Retry map')}>The anomaly map could not be loaded. The Heat lens is disabled.</ErrorState></Row>
        </Section>

        <Section id="lanes" title="Timeline lanes" spec="Steps 1–3 tint, 4–5 solid with a 3 px flag rule; minimum drawn width 2 px; not analysed spans hatched with their label; sampled rug (UX §8.4).">
          <Row state="ready" label="lane stack" note="Specimen 40 s timeline; segment b selected, playhead at 13.1 s.">
            <div className="flex flex-col gap-(--lane-gap)">
              <LaneAxis duration={40} playhead={13.1} callouts={[{ t: 9.8, id: 'a' }, { t: 12.4, id: 'b' }]} />
              <Lane label="Estimate" duration={40} segments={estimateSegments} playhead={13.1} selected="b" />
              <Lane label="Face A" duration={40} segments={faceSegments} gaps={[{ start: 22, end: 31, label: 'not in frame' }]} playhead={13.1} selected="b" />
              <Lane label="Audio" duration={40} segments={[{ start: 12, end: 16, step: 4, id: 'b' }]} gaps={[{ start: 24, end: 36, label: 'music, no speech' }]} playhead={13.1} selected="b" />
              <Lane label="Sampled" duration={40} rug={Array.from({ length: 160 }, (_, i) => i * 0.25)} playhead={13.1} />
            </div>
          </Row>
          <Row state="loading"><Lane label="Synthesis" duration={40} state="loading" /></Row>
          <Row state="partial" label="not run"><Lane label="Lip-sync" duration={40} state="not_run" reason="model not installed" /></Row>
          <Row state="empty" label="no data"><Lane label="Face B" duration={40} state="no_data" /></Row>
        </Section>

        <Section id="icon-array" title="Icon array" spec="10×10 dots, filled in reading order; shared with the PDF natural-frequency line (UX §8.4).">
          <Row state="ready" label="filled" note="Specimen: 3 of 100 filled.">
            <IconArray filled={3} label="3 of 100 filled (specimen)" />
          </Row>
          <Row state="empty" label="none"><IconArray filled={0} label="0 of 100 filled (specimen)" /></Row>
        </Section>

        <Section id="evidence" title="Evidence panels" spec="How the estimate was reached, the evidence ledger, Content Credentials, the metadata tree and codec facts (UX §5.6–5.8), drawn from synthetic specimens (features/analysis/specimens.ts).">
          <Row state="key-findings" label="Key findings" note="Each finding says what its check did from its neutral point; the note below the list says where the estimate starts, and that the start is not evidence about the file (D-63).">
            <KeyFindings analysis={specimenAnalysis()} signals={SPECIMEN_SIGNALS} exhibits={[]} token=""
              query={{ data: SPECIMEN_FINDINGS, status: 'success', isSuccess: true, isPending: false, error: null } as UseQueryResult<Finding[], Error>} />
          </Row>
          <Row state="additive" label="Waterfall" note="The rows add up to the estimate within the integrity tolerance; four smaller checks are grouped as net; the provenance bound is a hatched adjustment.">
            <HowReached analysis={specimenAnalysis()} signals={SPECIMEN_SIGNALS} />
          </Row>
          <Row state="integrity-failure" label="Integrity gate" note="One row too many: no chart is drawn, and the reason is stated.">
            <HowReached analysis={specimenAnalysis(SPECIMEN_BROKEN_DECISION)} signals={[]} />
          </Row>
          <Row state="ledger" label="Evidence ledger" note="Contribution column at ±1.5 log-odds full scale; P(AI) muted when uncalibrated; checks that did not run grouped with their reason.">
            <SignalsLedger analysis={specimenAnalysis()} signals={SPECIMEN_SIGNALS} runs={SPECIMEN_RUNS} exhibits={[]} token="" />
          </Row>
          <Row state="metadata-tree" label="Metadata tree">
            <MetadataTree groups={treeGroups(SPECIMEN_TREE)} caption={SPECIMEN_TREE.caption} />
          </Row>
          <Row state="c2pa-documentary" label="Credentials, AI declared">
            <C2paChain state="trusted" checked credentials={fields(SPECIMEN_CREDENTIALS)} chain={chainNodes(SPECIMEN_CHAIN)} />
          </Row>
          <Row state="c2pa-invalid" label="Credentials invalid">
            <C2paChain state="invalid" checked chain={[]} />
          </Row>
          <Row state="codec-facts" label="Codec facts">
            <CodecFacts facts={codecFacts(SPECIMEN_CODEC)} />
          </Row>
          <Row state="methods-validated" label="Methods & validation" note="Inside a validated slice: dossier metrics with bootstrap intervals, fused error rates, the detector's row for this slice, reliability factors.">
            <MethodsPanel analysis={specimenAnalysis(SPECIMEN_DECISION, SPECIMEN_VALIDATION)} cards={SPECIMEN_MODEL_CARDS} />
          </Row>
          <Row state="methods-outside" label="Outside the envelope" note="No dossier, no fuser record: every absent figure reads not established; no cell is blank.">
            <MethodsPanel analysis={specimenAnalysis(SPECIMEN_DECISION, { calibration: [], condition_profile: { slice_id: 'image/lossless/side<512/face<120', within_validated_envelope: false, limiting_conditions: ['low resolution (shorter side 128 px)'] } })} cards={[]} />
          </Row>
        </Section>

        <Section id="states" title="Section states" spec="Loading, empty, error, partial and low agreement render in place of one section; other sections keep rendering (UX §8.5).">
          <Row state="loading" note="Skeletons shaped like the content; no spinners in the middle of the page."><div className="max-w-[480px]"><SkeletonRows /></div></Row>
          <Row state="empty"><EmptyState action="+ Add note" onAction={act('Add note')}>No notes yet.</EmptyState></Row>
          <Row state="error"><ErrorState onRetry={act('Retry notes')}>Notes could not be loaded.</ErrorState></Row>
          <Row state="partial" note="Affected lanes and lenses are hatched with the reason; the ledger gains a Not run group; the PDF adds limitation L-00.">
            <NotRun what="Voice analysis" reason="no audio stream" />
          </Row>
          <Row state="low-confidence" label="low agreement" note="Headline suffix “— the checks disagree”, wide interval, link “Why the checks disagree”.">
            <span className="inline-flex items-center gap-2 text-small text-text"><ConfidenceMeter level={1} />agreement low</span>
          </Row>
        </Section>
      </main>
    </div>
  )
}
