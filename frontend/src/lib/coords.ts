/**
 * Original-pixel to viewport transforms (and back) for the evidence viewer (ARCHITECTURE §7.4).
 *
 * - **One transform.** The raster canvas and the SVG overlay share a uniform-scale affine map in CSS pixels:
 *   `viewport = original × scale + (tx, ty)`. Everything the viewer stores or posts (deep links, annotations) is in original px.
 * - **Exhibit spaces.** An exhibit's `coordinate_space` names the space its geometry or map grid is expressed in, and that space
 *   spans the whole image. The per-axis factor to original px is therefore `source / space`, which equals `scale_to_original`
 *   whenever the aspect ratio is kept; without the source size the scalar `scale_to_original` is used.
 * - **Orientation.** Exhibits are in EXIF-oriented pixels, as the decoded bitmap is. A source size recorded before the rotation
 *   (a result from before N-4 read by the report) is taken transposed when the space times its scale is the transposed size and
 *   not the recorded one; this is the rule of `backend/app/core/coords.py`, kept here so both pass the shared vectors.
 */

export interface Size { width: number; height: number }
export interface Point { x: number; y: number }
export interface Box { x: number; y: number; w: number; h: number }
export interface View { scale: number; tx: number; ty: number }
export interface CoordinateSpace { width: number; height: number; scale_to_original: number }
export interface Factors { sx: number; sy: number }

export const MIN_ZOOM = 0.1
export const MAX_ZOOM = 16
/** `−` / `+` steps (UX §5.9 lists 25–1600 %; ARCHITECTURE §7.4 reaches down to 10 % so very large images still fit). */
export const ZOOM_STEPS = [0.1, 0.15, 0.25, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 4, 6, 8, 12, 16] as const
/** Stage padding around a fitted image. */
export const FIT_PADDING = 24
/** At and above this zoom a 1 px pixel grid appears (UX §5.9: 800 %). */
export const PIXEL_GRID_ZOOM = 8

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

export const clampZoom = (scale: number): number => clamp(Number.isFinite(scale) && scale > 0 ? scale : 1, MIN_ZOOM, MAX_ZOOM)

export const toViewport = (v: View, p: Point): Point => ({ x: p.x * v.scale + v.tx, y: p.y * v.scale + v.ty })
export const toOriginal = (v: View, p: Point): Point => ({ x: (p.x - v.tx) / v.scale, y: (p.y - v.ty) / v.scale })
export const boxToViewport = (v: View, b: Box): Box => ({ x: b.x * v.scale + v.tx, y: b.y * v.scale + v.ty, w: b.w * v.scale, h: b.h * v.scale })
export const boxToOriginal = (v: View, b: Box): Box => ({ x: (b.x - v.tx) / v.scale, y: (b.y - v.ty) / v.scale, w: b.w / v.scale, h: b.h / v.scale })

/** The SVG `transform` of a view: shapes drawn in original px land on the canvas pixels beneath them. */
export const svgMatrix = (v: View): string => `matrix(${v.scale} 0 0 ${v.scale} ${v.tx} ${v.ty})`

/** A view at `scale` with the original point `centre` in the middle of the viewport. */
export function centredView(scale: number, centre: Point, viewport: Size): View {
  const s = clampZoom(scale)
  return { scale: s, tx: viewport.width / 2 - centre.x * s, ty: viewport.height / 2 - centre.y * s }
}

/** Fit: the whole image inside the viewport less the padding, centred. */
export function fitView(source: Size, viewport: Size, padding = FIT_PADDING): View {
  const aw = Math.max(1, viewport.width - 2 * padding)
  const ah = Math.max(1, viewport.height - 2 * padding)
  return centredView(Math.min(aw / source.width, ah / source.height), { x: source.width / 2, y: source.height / 2 }, viewport)
}

/** The original point in the middle of the viewport (the `cx`, `cy` of a deep link). */
export const viewCentre = (v: View, viewport: Size): Point => toOriginal(v, { x: viewport.width / 2, y: viewport.height / 2 })

/** Zoom to `scale` keeping the original point under the viewport `anchor` in place (wheel zoom at the cursor). */
export function zoomAt(v: View, scale: number, anchor: Point): View {
  const s = clampZoom(scale)
  const o = toOriginal(v, anchor)
  return { scale: s, tx: anchor.x - o.x * s, ty: anchor.y - o.y * s }
}

export const panBy = (v: View, dx: number, dy: number): View => ({ scale: v.scale, tx: v.tx + dx, ty: v.ty + dy })

/** The next `−` / `+` step from the current scale; a fitted scale between two steps moves to the nearer one in that direction. */
export function stepZoom(scale: number, direction: 1 | -1): number {
  const eps = 1e-6
  if (direction > 0) return ZOOM_STEPS.find((s) => s > scale + eps) ?? MAX_ZOOM
  return [...ZOOM_STEPS].reverse().find((s) => s < scale - eps) ?? MIN_ZOOM
}

/** Frame a box (selecting a region): the box with half its size again as margin, centred, at most `maxScale`. */
export function frameBox(box: Box, viewport: Size, maxScale = 8): View {
  const w = Math.max(1, box.w * 1.5)
  const h = Math.max(1, box.h * 1.5)
  const scale = Math.min(maxScale, Math.min(viewport.width / w, viewport.height / h))
  return centredView(scale, { x: box.x + box.w / 2, y: box.y + box.h / 2 }, viewport)
}

/** Per-axis factors from an exhibit's coordinate space to original px (identity when the exhibit names no space). */
export function spaceFactors(space: CoordinateSpace | null | undefined, source: Size | null): Factors {
  if (!space) return { sx: 1, sy: 1 }
  if (source) {
    const s = orientedSource(space, source)
    return { sx: s.width / space.width, sy: s.height / space.height }
  }
  return { sx: space.scale_to_original, sy: space.scale_to_original }
}

/** The alignment bound within which a space times its scale counts as a size (2 px, plus the six-decimal rounding of the scale). */
const fits = (space: CoordinateSpace, size: Size): boolean => {
  const scale = space.scale_to_original || 1
  const near = (n: number, target: number) => Math.abs(n * scale - target) <= 2 + n * 0.0000005 + 1e-9
  return near(space.width, size.width) && near(space.height, size.height)
}

/** The source size as the exhibit's space sees it: transposed when the space fits the transposed size and not this one. */
export function orientedSource(space: CoordinateSpace, source: Size): Size {
  const transposed = { width: source.height, height: source.width }
  return !fits(space, source) && fits(space, transposed) ? transposed : source
}

export const scaleBox = (b: Box, f: Factors): Box => ({ x: b.x * f.sx, y: b.y * f.sy, w: b.w * f.sx, h: b.h * f.sy })

export function polygonBounds(points: readonly (readonly number[])[]): Box {
  const xs = points.map((p) => p[0])
  const ys = points.map((p) => p[1])
  const x = Math.min(...xs)
  const y = Math.min(...ys)
  return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y }
}

export const containsPoint = (b: Box, p: Point): boolean => p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h

/** The source pixel under a viewport point (integer px), or null outside the image. */
export function pixelAt(v: View, p: Point, source: Size): Point | null {
  const o = toOriginal(v, p)
  const x = Math.floor(o.x)
  const y = Math.floor(o.y)
  return x >= 0 && y >= 0 && x < source.width && y < source.height ? { x, y } : null
}

/** The map cell holding an original pixel, for a map grid stretched over the whole image. */
export function gridCell(p: Point, source: Size, grid: readonly number[]): Point {
  return {
    x: clamp(Math.floor((p.x * grid[0]) / source.width), 0, grid[0] - 1),
    y: clamp(Math.floor((p.y * grid[1]) / source.height), 0, grid[1] - 1),
  }
}

/** A drawn box from two original points (any drag direction), clamped to the image and rounded to 0.01 px. */
export function normaliseBox(a: Point, b: Point, source: Size): Box {
  const r = (v: number) => Math.round(v * 100) / 100
  const x0 = clamp(Math.min(a.x, b.x), 0, source.width)
  const x1 = clamp(Math.max(a.x, b.x), 0, source.width)
  const y0 = clamp(Math.min(a.y, b.y), 0, source.height)
  const y1 = clamp(Math.max(a.y, b.y), 0, source.height)
  return { x: r(x0), y: r(y0), w: r(x1 - x0), h: r(y1 - y0) }
}
