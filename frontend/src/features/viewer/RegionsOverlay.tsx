import type { ReactNode } from 'react'
import { boxToViewport, svgMatrix, toViewport, type Box, type Point, type Size, type View } from '../../lib/coords'
import type { AnnotationItem, RegionItem } from './lenses'

const INK = 'var(--color-lens-ink)'
const HALO = 'var(--color-lens-halo)'
const CLONE = 'var(--color-lens-clone)'
const NOTE = 'var(--color-annotation)'
const SELECTED = 'var(--color-selection)'

const chipWidth = (text: string) => Math.ceil(text.length * 6.7 + 8)

/** A label chip in viewport px (11 px mono on an opaque stage surface). */
function Chip({ at, text, colour }: { at: Point; text: string; colour: string }) {
  const w = chipWidth(text)
  return (
    <g transform={`translate(${Math.round(at.x)} ${Math.round(at.y)})`}>
      <rect width={w} height={16} rx={2} style={{ fill: 'var(--color-stage-2)' }} />
      <text x={4} y={11.5} className="font-mono" style={{ fill: colour, fontSize: 11 }}>{text}</text>
    </g>
  )
}

/** Halo then ink, both 1 CSS px regardless of zoom (`non-scaling-stroke`). */
function Outline({ box, polygon, colour, width, dashed = false }: { box: Box; polygon?: Point[] | null; colour: string; width: number; dashed?: boolean }) {
  const shape = (style: React.CSSProperties): ReactNode => polygon
    ? <polygon points={polygon.map((p) => `${p.x},${p.y}`).join(' ')} vectorEffect="non-scaling-stroke" style={style} />
    : <rect x={box.x} y={box.y} width={box.w} height={box.h} vectorEffect="non-scaling-stroke" style={style} />
  return (
    <>
      {shape({ fill: 'none', stroke: HALO, strokeWidth: width + 2 })}
      {shape({ fill: 'none', stroke: colour, strokeWidth: width, strokeDasharray: dashed ? '4 3' : undefined })}
    </>
  )
}

/** Vector marks over the media (UX §5.9; ARCHITECTURE §7.4): regions R1…, clone pairs C1 and C1′ with a dashed connector, faces A…,
 *  analyst annotations N1… and the box being drawn. Shapes are in original px under the view transform; labels sit in viewport px so
 *  they stay legible at any zoom. The overlay never takes pointer events: the stage handles pan, draw and selection. */
export default function RegionsOverlay({ view, viewport, items, notes, selected, draft }: {
  view: View
  viewport: Size
  items: readonly RegionItem[]
  notes: readonly AnnotationItem[]
  selected: string | null
  draft: Box | null
}) {
  const centre = (b: Box): Point => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 })
  const pairs = items.filter((r) => r.lens === 'clones' && r.pair && !r.label.endsWith('′'))
    .map((a) => [a, items.find((b) => b.lens === 'clones' && b.exhibitId === a.exhibitId && b.label === a.pair)] as const)
    .filter((p): p is readonly [RegionItem, RegionItem] => p[1] != null)
  return (
    <svg data-testid="regions-overlay" width={viewport.width} height={viewport.height} aria-hidden className="absolute left-0 top-0 pointer-events-none overflow-visible">
      <g transform={svgMatrix(view)}>
        {pairs.map(([a, b]) => {
          const p = centre(a.box)
          const q = centre(b.box)
          return <line key={`${a.key}-pair`} x1={p.x} y1={p.y} x2={q.x} y2={q.y} vectorEffect="non-scaling-stroke" style={{ stroke: CLONE, strokeWidth: 1, strokeDasharray: '4 4' }} />
        })}
        {items.map((r) => (
          <g key={r.key} data-region={r.label} data-lens={r.lens}>
            <Outline box={r.box} polygon={r.polygon} colour={selected === r.label ? SELECTED : r.lens === 'clones' ? CLONE : INK} width={selected === r.label ? 2 : 1.5} />
          </g>
        ))}
        {notes.map((n) => n.box && (
          <g key={n.annotation.id} data-note={n.label}>
            <Outline box={n.box} colour={selected === n.label ? SELECTED : NOTE} width={selected === n.label ? 2 : 1.5} />
          </g>
        ))}
        {notes.map((n) => n.line && (
          <line key={`${n.annotation.id}-line`} x1={n.line[0].x} y1={n.line[0].y} x2={n.line[1].x} y2={n.line[1].y} vectorEffect="non-scaling-stroke" style={{ stroke: NOTE, strokeWidth: 1.5 }} />
        ))}
        {draft && <Outline box={draft} colour={NOTE} width={1.5} dashed />}
      </g>
      {items.map((r) => {
        const b = boxToViewport(view, r.box)
        const text = r.lens === 'faces' && r.score != null ? `${r.label} · ${r.score.toFixed(2)}` : r.label
        // Chip text pairs with the stage material (light paper on the result page, dark in the viewer); outlines carry the lens colour.
        return <Chip key={`${r.key}-chip`} at={{ x: b.x + 2, y: b.y + 2 }} text={text} colour="var(--color-stage-text)" />
      })}
      {notes.map((n) => {
        // Notes label the top-right corner, so a note traced on a region never hides the region's label.
        const b = n.box ? boxToViewport(view, n.box) : null
        const at = b ? { x: b.x + b.w - chipWidth(n.label) - 2, y: b.y + 2 } : n.line ? toViewport(view, n.line[1]) : null
        return at && <Chip key={`${n.annotation.id}-chip`} at={at} text={n.label} colour={NOTE} />
      })}
    </svg>
  )
}
