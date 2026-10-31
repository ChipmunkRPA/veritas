/**
 * Raster scene of the evidence viewer (UX §5.9, §8.3; ARCHITECTURE §7.4): source (single image or tiles), dimming of unreliable
 * areas, the heat layer, the reliability hatch drawn over the heat, and the pixel grid at ≥ 800 %. One function draws the main
 * canvas, the loupe and the side-by-side pane, so every surface shows the same pixels under the same transform.
 *
 * - **Upscaling** is nearest-neighbour at ≥ 100 % and for map layers always: boundaries never look more precise than the model.
 * - **Compare.** `lens: false` draws the source only; `lensFromX` clips the lens layers to the right of a swipe divider.
 */
import { boxToViewport, PIXEL_GRID_ZOOM, toOriginal, type Box, type Size, type View } from '../../lib/coords'
import type { TileSource } from './TileSource'

export type Base = { kind: 'bitmap'; bitmap: ImageBitmap } | { kind: 'tiles'; tiles: TileSource }

export interface SceneLayers {
  heat: ImageBitmap | null
  /** 0–1, applied with `globalAlpha`. */
  opacity: number
  unreliable: ImageBitmap | null
  hatch: boolean
  dim: boolean
  mapOnly: boolean
}

export interface Scene { source: Size; base: Base | null; layers: SceneLayers }

export interface DrawOptions { dpr: number; lens: boolean; lensFromX?: number | null; grid?: boolean }

const cssVar = (name: string, fallback: string): string => {
  if (typeof document === 'undefined') return fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

let helper: HTMLCanvasElement | null = null
function helperContext(width: number, height: number): CanvasRenderingContext2D | null {
  helper = helper ?? document.createElement('canvas')
  if (helper.width !== width || helper.height !== height) {
    helper.width = width
    helper.height = height
  }
  return helper.getContext('2d')
}

const patterns = new WeakMap<CanvasRenderingContext2D, CanvasPattern | null>()
/** 45° hatch, 6 px pitch, 1 px lens-hatch lines (UX §8.1), like the SVG `HatchPattern`. */
function hatchPattern(ctx: CanvasRenderingContext2D): CanvasPattern | null {
  if (patterns.has(ctx)) return patterns.get(ctx) ?? null
  const tile = document.createElement('canvas')
  tile.width = 6
  tile.height = 6
  const t = tile.getContext('2d')
  let pattern: CanvasPattern | null = null
  if (t) {
    t.fillStyle = cssVar('--color-lens-hatch', 'rgb(243 243 241 / 0.55)')
    t.fillRect(0, 0, 1, 6)
    pattern = ctx.createPattern(tile, 'repeat')
    pattern?.setTransform(new DOMMatrix().rotate(45))
  }
  patterns.set(ctx, pattern)
  return pattern
}

/** Fill `img` with `fill` where the stencil is opaque, via the shared helper canvas (device px). */
function stencil(ctx: CanvasRenderingContext2D, mask: ImageBitmap, img: Box, viewport: Size, dpr: number, fill: string | CanvasPattern | null) {
  if (!fill) return
  const h = helperContext(Math.round(viewport.width * dpr), Math.round(viewport.height * dpr))
  if (!h) return
  h.setTransform(dpr, 0, 0, dpr, 0, 0)
  h.globalCompositeOperation = 'source-over'
  h.clearRect(0, 0, viewport.width, viewport.height)
  h.fillStyle = fill
  h.fillRect(img.x, img.y, img.w, img.h)
  h.globalCompositeOperation = 'destination-in'
  h.imageSmoothingEnabled = false
  h.drawImage(mask, img.x, img.y, img.w, img.h)
  h.globalCompositeOperation = 'source-over'
  ctx.save()
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.drawImage(h.canvas, 0, 0)
  ctx.restore()
}

function pixelGrid(ctx: CanvasRenderingContext2D, view: View, viewport: Size, source: Size) {
  const a = toOriginal(view, { x: 0, y: 0 })
  const b = toOriginal(view, { x: viewport.width, y: viewport.height })
  const x0 = Math.max(0, Math.floor(a.x))
  const x1 = Math.min(source.width, Math.ceil(b.x))
  const y0 = Math.max(0, Math.floor(a.y))
  const y1 = Math.min(source.height, Math.ceil(b.y))
  if (x1 <= x0 || y1 <= y0) return
  ctx.save()
  ctx.strokeStyle = 'rgb(243 243 241 / 0.12)'
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let x = x0; x <= x1; x++) {
    const vx = Math.round(x * view.scale + view.tx) + 0.5
    ctx.moveTo(vx, y0 * view.scale + view.ty)
    ctx.lineTo(vx, y1 * view.scale + view.ty)
  }
  for (let y = y0; y <= y1; y++) {
    const vy = Math.round(y * view.scale + view.ty) + 0.5
    ctx.moveTo(x0 * view.scale + view.tx, vy)
    ctx.lineTo(x1 * view.scale + view.tx, vy)
  }
  ctx.stroke()
  ctx.restore()
}

/** Draws the scene in viewport CSS px on a canvas of `viewport × dpr` device px. Returns whether the source has fully loaded. */
export function drawScene(ctx: CanvasRenderingContext2D, scene: Scene, view: View, viewport: Size, opts: DrawOptions): { baseComplete: boolean } {
  const { dpr } = opts
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, viewport.width, viewport.height)
  const img = boxToViewport(view, { x: 0, y: 0, w: scene.source.width, h: scene.source.height })
  // A closed ImageBitmap reports 0 × 0; a frame scheduled before a replacement rendered must never draw it.
  const open = (b: ImageBitmap | null): ImageBitmap | null => (b && b.width > 0 && b.height > 0 ? b : null)
  const L = { ...scene.layers, heat: open(scene.layers.heat), unreliable: open(scene.layers.unreliable) }
  const showLens = opts.lens && (L.heat != null || L.hatch || L.dim)
  let baseComplete = false
  ctx.imageSmoothingEnabled = view.scale < 1
  ctx.imageSmoothingQuality = 'high'
  if (showLens && L.mapOnly && L.heat) {
    // "Map only": the map on #000 (UX §8.3); the source still draws left of a swipe divider.
    if (opts.lensFromX != null) drawBase()
    else baseComplete = true
  } else {
    drawBase()
  }
  function drawBase() {
    if (scene.base?.kind === 'bitmap') {
      if (!open(scene.base.bitmap)) return
      ctx.drawImage(scene.base.bitmap, img.x, img.y, img.w, img.h)
      baseComplete = true
    } else if (scene.base?.kind === 'tiles') {
      baseComplete = scene.base.tiles.draw(ctx, view, viewport, dpr)
    }
  }
  if (showLens) {
    ctx.save()
    if (opts.lensFromX != null) {
      ctx.beginPath()
      ctx.rect(opts.lensFromX, 0, Math.max(0, viewport.width - opts.lensFromX), viewport.height)
      ctx.clip()
    }
    ctx.imageSmoothingEnabled = false
    if (L.mapOnly && L.heat) {
      ctx.fillStyle = cssVar('--color-heat-ground', '#000000')
      ctx.fillRect(img.x, img.y, img.w, img.h)
    }
    if (L.dim && L.unreliable) stencil(ctx, L.unreliable, img, viewport, dpr, 'rgb(0 0 0 / 0.5)')
    if (L.heat && L.opacity > 0) {
      ctx.globalAlpha = L.mapOnly ? 1 : L.opacity
      ctx.drawImage(L.heat, img.x, img.y, img.w, img.h)
      ctx.globalAlpha = 1
    }
    if (L.hatch && L.unreliable) stencil(ctx, L.unreliable, img, viewport, dpr, hatchPattern(ctx))
    ctx.restore()
  }
  if (opts.grid && view.scale >= PIXEL_GRID_ZOOM) pixelGrid(ctx, view, viewport, scene.source)
  return { baseComplete }
}
