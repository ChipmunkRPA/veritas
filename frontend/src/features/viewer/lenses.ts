/**
 * Lens catalogue of the evidence viewer (UX §5.9 "Lens families").
 *
 * - **Map lenses** (H heat, G attention, E ELA, N noise residual, J JPEG ghosts): one at a time. A heatmap exhibit's `semantics`
 *   picks the lens: `p_manipulated` → heat, `attribution` → attention, `error_level` → ELA, `residual` → JPEG ghosts when its
 *   title or detector names ghosts or double compression, else noise residual.
 * - **Marks** stack: reliability (R, the hatch over the active map's unreliable cells), regions from localisers (R1…), clone pairs
 *   (C, C1 and C1′), faces (F) and analyst annotations (A, N1…).
 * - **Preset** (UX §5.5): the map cited by the highest-ranked finding that cites one, with reliability and the localisation
 *   regions; otherwise the heat map when there is one. Faces and annotations stack whenever present.
 */
import type { Annotation, ExhibitPayload, Finding } from '../../lib/api/queries'
import { noteNumbers } from '../review/review'
import { polygonBounds, scaleBox, spaceFactors, type Box, type Point, type Size } from '../../lib/coords'
import type { Layers, MarkLayer } from './viewerState'
import { DEFAULT_OPACITY } from './viewerState'

export type HeatmapPayload = Extract<ExhibitPayload, { kind: 'heatmap' }>
export type RegionsPayload = Extract<ExhibitPayload, { kind: 'regions' }>
export type MapLensKey = 'heat' | 'attention' | 'ela' | 'noise' | 'ghosts'
export type MarkLensKey = 'reliability' | 'regions' | 'clones' | 'faces' | 'annotations'
export type LensKey = MapLensKey | MarkLensKey
export type Family = 'Localisation' | 'Faces' | 'Attention' | 'Filters' | 'Marks'

export interface LensDef { key: LensKey; letter: string | null; family: Family; name: string; meaning: string; kind: 'map' | 'mark' }

export const LENSES: readonly LensDef[] = [
  { key: 'heat', letter: 'H', family: 'Localisation', name: 'Heat', kind: 'map', meaning: 'Where pixel statistics differ from the rest of the image' },
  { key: 'reliability', letter: 'R', family: 'Localisation', name: 'Reliability', kind: 'mark', meaning: 'Where the map can’t be trusted' },
  // N-19d (design review P2): "Regions" was spelled out among single letters. R is the reliability hatch; O for the outlines
  { key: 'regions', letter: 'O', family: 'Localisation', name: 'Regions', kind: 'mark', meaning: 'Areas a localiser flagged for examination, outlined' },
  { key: 'clones', letter: 'C', family: 'Localisation', name: 'Clone matches', kind: 'mark', meaning: 'Areas that look copied from elsewhere in the image' },
  { key: 'faces', letter: 'F', family: 'Faces', name: 'Faces', kind: 'mark', meaning: 'Detected faces; swap detectors ran on each' },
  { key: 'attention', letter: 'G', family: 'Attention', name: 'Grad-CAM', kind: 'map', meaning: 'Where a classifier looked. Not a map of manipulation.' },
  { key: 'ela', letter: 'E', family: 'Filters', name: 'ELA', kind: 'map', meaning: 'How much each area changes on re-save. Bright edges and textures are normal.' },
  { key: 'noise', letter: 'N', family: 'Filters', name: 'Noise residual', kind: 'map', meaning: 'Local noise; sensor noise should look consistent' },
  { key: 'ghosts', letter: 'J', family: 'Filters', name: 'JPEG ghosts', kind: 'map', meaning: 'Areas saved at a different quality' },
  { key: 'annotations', letter: 'A', family: 'Marks', name: 'Annotations', kind: 'mark', meaning: 'Examiner’s regions and arrows' },
]
export const FAMILIES: readonly Family[] = ['Localisation', 'Faces', 'Attention', 'Filters', 'Marks']
export const lensDef = (key: LensKey): LensDef => LENSES.find((l) => l.key === key) as LensDef
export const lensByLetter = (letter: string): LensDef | undefined => LENSES.find((l) => l.letter === letter.toUpperCase())
const MAP_ORDER: readonly MapLensKey[] = ['heat', 'noise', 'ela', 'ghosts', 'attention']

export function heatmapLens(x: { semantics?: string | null; title?: string | null; created_by?: string | null }): MapLensKey {
  if (x.semantics === 'p_manipulated') return 'heat'
  if (x.semantics === 'attribution') return 'attention'
  if (x.semantics === 'error_level') return 'ela'
  return /ghost|double.?compress|jpeg.?(history|dq)/i.test(`${x.title ?? ''} ${x.created_by ?? ''}`) ? 'ghosts' : 'noise'
}

export function regionsLens(x: RegionsPayload): 'regions' | 'faces' | 'clones' {
  const labels = x.regions.map((r) => r.label ?? '')
  if (/copy.?move|clone/i.test(`${x.created_by} ${x.title}`) || labels.some((l) => /^C\d+′?:/.test(l))) return 'clones'
  if (/face/i.test(`${x.created_by} ${x.title}`) || (labels.length > 0 && labels.every((l) => /^face\b/i.test(l)))) return 'faces'
  return 'regions'
}

export interface Catalogue {
  maps: Record<MapLensKey, HeatmapPayload[]>
  regions: RegionsPayload[]
  faces: RegionsPayload[]
  clones: RegionsPayload[]
  /** The tile pyramid of the source image, carried by the heatmaps of images above 4096 px, and the exhibit that serves it. */
  tiles: { exhibitId: string; info: NonNullable<HeatmapPayload['tiles']> } | null
}

export function catalogue(payloads: readonly ExhibitPayload[]): Catalogue {
  const cat: Catalogue = { maps: { heat: [], attention: [], ela: [], noise: [], ghosts: [] }, regions: [], faces: [], clones: [], tiles: null }
  for (const x of payloads) {
    if (x.kind === 'heatmap' && x.frame_pts == null) {
      cat.maps[heatmapLens(x)].push(x)
      if (!cat.tiles && x.tiles) cat.tiles = { exhibitId: x.id, info: x.tiles }
    } else if (x.kind === 'regions' && x.regions.length > 0) {
      cat[regionsLens(x)].push(x)
    }
  }
  return cat
}

export const allMaps = (cat: Catalogue): HeatmapPayload[] => MAP_ORDER.flatMap((k) => cat.maps[k])
export const mapById = (cat: Catalogue, id: string | null | undefined): HeatmapPayload | null => allMaps(cat).find((m) => m.id === id) ?? null

/** Whether a lens has anything to show (the rail disables the others with the reason). */
export function available(cat: Catalogue, key: LensKey, activeMap: HeatmapPayload | null): boolean {
  if (key === 'annotations') return true
  if (key === 'reliability') return activeMap?.reliability_mask_artefact_id != null
  if (key === 'regions' || key === 'faces' || key === 'clones') return cat[key].length > 0
  return cat.maps[key].length > 0
}

/** The lenses a layer list turns on (map lens by its exhibit's semantics, marks by kind). */
export function activeLenses(cat: Catalogue, layers: Layers): Set<LensKey> {
  const map = mapById(cat, layers.map?.id)
  return new Set<LensKey>([...(map ? [heatmapLens(map)] : []), ...layers.marks.map((m) => m.kind)])
}

/** Toggle a lens (UX §5.9 "Combinations"): a map lens replaces the current map, keeping its opacity; a mark stacks, with every
 *  exhibit of its kind. */
export function toggleLayers(cat: Catalogue, layers: Layers, key: LensKey): Layers {
  if (lensDef(key).kind === 'map') {
    const exhibits = cat.maps[key as MapLensKey]
    const current = mapById(cat, layers.map?.id)
    if ((current && heatmapLens(current) === key) || !exhibits.length) return { ...layers, map: null }
    return { ...layers, map: { id: exhibits[0].id, opacity: layers.map?.opacity ?? DEFAULT_OPACITY } }
  }
  const kind = key as MarkLensKey
  if (layers.marks.some((m) => m.kind === kind)) return { ...layers, marks: layers.marks.filter((m) => m.kind !== kind) }
  const ids = kind === 'regions' || kind === 'faces' || kind === 'clones' ? cat[kind].map((x) => x.id) : [null]
  return { ...layers, marks: [...layers.marks, ...ids.map((id) => ({ kind, id }))] }
}

export function presetLayers(cat: Catalogue, findings: readonly Finding[], findingId: string | null = null): Layers {
  const ranked = [...findings].sort((a, b) => a.rank - b.rank)
  const chosen = findingId ? ranked.filter((f) => f.id === findingId) : ranked
  const maps = allMaps(cat)
  const cited = chosen.map((f) => maps.find((m) => f.evidence_ids?.includes(m.id))).find((m) => m != null) ?? null
  const map = cited ?? (findingId ? null : cat.maps.heat[0] ?? null)
  const marks: MarkLayer[] = []
  if (map?.reliability_mask_artefact_id) marks.push({ kind: 'reliability', id: null })
  if (map) for (const r of cat.regions) marks.push({ kind: 'regions', id: r.id })
  for (const r of cat.clones) if (!findingId || chosen.some((f) => f.evidence_ids?.includes(r.id))) marks.push({ kind: 'clones', id: r.id })
  for (const r of cat.faces) marks.push({ kind: 'faces', id: r.id })
  marks.push({ kind: 'annotations', id: null })
  return { map: map ? { id: map.id, opacity: DEFAULT_OPACITY } : null, marks }
}

export interface RegionItem {
  /** Visible label: R1…, A, B… for faces, C1 / C1′ for clone pairs. */
  label: string
  key: string
  exhibitId: string
  lens: 'regions' | 'faces' | 'clones'
  box: Box
  polygon: Point[] | null
  score: number | null
  probability: number | null
  text: string
  /** N-19d: the region's share of the image (0–1), measured as the reports measure it: the recorded box in the space it was
   *  recorded in (`coords.area_share`), not the box after it is mapped to original pixels. Null without a recorded box. */
  share: number | null
  /** The other side of a clone pair. */
  pair: string | null
}

const faceLetter = (i: number) => String.fromCharCode(65 + (i % 26)) + (i >= 26 ? String(Math.floor(i / 26)) : '')

/** Native-resolution crops a whole-image classifier scored (`Crops scored by …`): not localisation, so not `R…` (DR P0-1). */
export const isCrops = (x: RegionsPayload): boolean =>
  /^crops scored/i.test(x.title ?? '') || (x.regions.length > 0 && x.regions.every((r) => /\bcrop$/i.test(r.label ?? '')))

/** Every region of the catalogue in original px. Labels: R1… localisation, C-1… classifier crops, C1 / C1′ clone pairs, A, B… faces. */
export function regionItems(cat: Catalogue, source: Size | null): RegionItem[] {
  const out: RegionItem[] = []
  let cropIndex = 0
  for (const lens of ['regions', 'clones', 'faces'] as const) {
    let faceIndex = 0
    for (const x of cat[lens]) {
      const f = spaceFactors(x.coordinate_space, source)
      const crops = lens === 'regions' && isCrops(x)
      x.regions.forEach((r, i) => {
        const g = r.region
        const polygon = g.polygon && g.polygon.length >= 3 ? g.polygon.map(([px, py]) => ({ x: px * f.sx, y: py * f.sy })) : null
        const box = g.bbox ? scaleBox(g.bbox, f) : polygon ? polygonBounds(polygon.map((p) => [p.x, p.y])) : null
        if (!box) return
        const clone = /^(C\d+)(′?)/.exec(r.label ?? '')
        const label = lens === 'faces' ? faceLetter(faceIndex++) : lens === 'clones' && clone ? clone[1] + clone[2]
          : crops ? `C-${++cropIndex}` : r.display_id ?? `R${i + 1}`
        const space = x.coordinate_space
        const whole = space?.width && space?.height ? space.width * space.height : source ? source.width * source.height : 0
        out.push({
          label, key: `${x.id}#${i}`, exhibitId: x.id, lens, box, polygon, score: r.score ?? null, probability: r.probability ?? null,
          share: g.bbox && whole > 0 ? (g.bbox.w * g.bbox.h) / whole : null,
          text: r.label ?? '', pair: lens === 'clones' && clone ? clone[1] + (clone[2] ? '' : '′') : null,
        })
      })
    }
  }
  return out
}

const where = (b: Box, source: Size): string => {
  const cx = (b.x + b.w / 2) / source.width
  const cy = (b.y + b.h / 2) / source.height
  const v = cy < 1 / 3 ? 'upper' : cy > 2 / 3 ? 'lower' : 'middle'
  const h = cx < 1 / 3 ? 'left' : cx > 2 / 3 ? 'right' : 'centre'
  return v === 'middle' && h === 'centre' ? 'centre' : v === 'middle' ? `middle ${h}` : `${v} ${h}`
}

export const areaShare = (b: Box, source: Size): number => (100 * b.w * b.h) / (source.width * source.height)

// N-19d (design review P2: "R1 is 6.3 % in the viewer and 6.2 % in the PDF"): one number for a region's area. The reports
// and the stored finding sentences print the share with one decimal, a tie going to the even digit (6.25 reads "6.2"); a
// browser's own formatting sends a tie up ("6.3"). The viewer formats as the reports do. Both read the exact value of the
// number, so they differ only at an exact tie, and a tie at one decimal is an odd multiple of 0.25 (x.25, x.75).
export function oneDecimal(v: number): string {
  const quarters = v * 4  // exact: a power of two
  if (!Number.isInteger(quarters) || quarters % 2 === 0) return v.toFixed(1)
  const lower = Math.floor(v * 10)  // v × 10 ends in .5 exactly
  return ((lower % 2 === 0 ? lower : lower + 1) / 10).toFixed(1)
}
/** "6.2 %": the share of the image a region takes, as the PDF prints it. The region's own share when it has one, else the
 *  share of its bounds in original pixels (a polygon). */
export function shareText(r: Pick<RegionItem, 'share' | 'box'>, source: Size): string {
  return `${oneDecimal(r.share != null ? r.share * 100 : areaShare(r.box, source))} %`
}

/** Text alternative of the marks on the stage (UX §8.7, WCAG 1.1.1). */
export function describeRegions(mapTitle: string | null, items: readonly RegionItem[], source: Size): string {
  const head = mapTitle ? `${mapTitle}. ` : ''
  if (!items.length) return `${head}No regions marked.`
  const parts = items.map((r) => {
    const score = r.probability ?? r.score
    return `${r.label} ${where(r.box, source)}, ${shareText(r, source)} of the image${score != null ? `, score ${score.toFixed(2)}` : ''}`
  })
  return `${head}${items.length} ${items.length === 1 ? 'mark' : 'marks'}: ${parts.join('; ')}.`
}

export interface AnnotationItem { label: string; annotation: Annotation; box: Box | null; line: [Point, Point] | null }

/** Drawn annotations (box, polygon, arrow, text), labelled with the N-numbers the notes thread uses (`noteNumbers`). */
export function annotationItems(annotations: readonly Annotation[]): AnnotationItem[] {
  const numbers = noteNumbers(annotations)
  const drawn = annotations.filter((a) => a.geometry && ['box', 'polygon', 'arrow', 'text'].includes(a.shape) && a.geometry.frame_pts == null)
  return drawn.map((a, i) => {
    const g = a.geometry as Record<string, unknown>
    const n = (k: string) => Number(g[k] ?? 0)
    const label = numbers.get(a.id) ?? `N${i + 1}`
    if (a.shape === 'box') return { label, annotation: a, box: { x: n('x'), y: n('y'), w: n('w'), h: n('h') }, line: null }
    if (a.shape === 'polygon') return { label, annotation: a, box: polygonBounds((g.points as number[][]) ?? [[0, 0]]), line: null }
    if (a.shape === 'arrow') return { label, annotation: a, box: null, line: [{ x: n('x1'), y: n('y1') }, { x: n('x2'), y: n('y2') }] }
    return { label, annotation: a, box: { x: n('x'), y: n('y'), w: 0, h: 0 }, line: null }
  })
}
