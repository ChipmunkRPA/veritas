/**
 * Tile pyramid source for images above 4096 px on the long side (ARCHITECTURE §4.3 `TileSet`, §7.4).
 *
 * - **Levels.** `z = levels − 1` is the native image; each lower level halves both sides rounding up (`app/engine/tiles.py`),
 *   until the whole image fits one 512 px tile at `z = 0`.
 * - **Choice.** The lowest level with at least one level pixel per device pixel at the current zoom.
 * - **Drawing.** Only tiles inside the viewport are requested. Level 0 and every cached coarser level draw first, so the image is
 *   never blank while finer tiles load. An LRU cache holds up to 256 `ImageBitmap`s and closes the ones it evicts.
 */
import { boxToViewport, toOriginal, type Box, type Size, type View } from '../../lib/coords'

export interface TileSetInfo { tile_px: number; levels: number; width: number; height: number }
export interface TileRef { z: number; x: number; y: number; key: string; box: Box }

export const TILE_CACHE_SIZE = 256
const CONCURRENCY = 6

/** Level sizes by `z`. */
export function levelSizes(ts: TileSetInfo): Size[] {
  const sizes: Size[] = new Array(ts.levels)
  let w = ts.width
  let h = ts.height
  for (let z = ts.levels - 1; z >= 0; z--) {
    sizes[z] = { width: w, height: h }
    w = Math.ceil(w / 2)
    h = Math.ceil(h / 2)
  }
  return sizes
}

/** The lowest level whose resolution covers `scale × dpr` level px per original px. */
export function levelFor(ts: TileSetInfo, scale: number, dpr = 1): number {
  const sizes = levelSizes(ts)
  for (let z = 0; z < ts.levels; z++) if (sizes[z].width / ts.width >= scale * dpr - 1e-9) return z
  return ts.levels - 1
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** Tiles of level `z` that intersect the viewport, with their boxes in original px. */
export function visibleTiles(ts: TileSetInfo, z: number, view: View, viewport: Size): TileRef[] {
  const { width: lw, height: lh } = levelSizes(ts)[z]
  const fx = ts.width / lw
  const fy = ts.height / lh
  const a = toOriginal(view, { x: 0, y: 0 })
  const b = toOriginal(view, { x: viewport.width, y: viewport.height })
  const x0 = Math.max(0, a.x)
  const y0 = Math.max(0, a.y)
  const x1 = Math.min(ts.width, b.x)
  const y1 = Math.min(ts.height, b.y)
  if (x1 <= x0 || y1 <= y0) return []
  const span = ts.tile_px
  const cols = Math.ceil(lw / span)
  const rows = Math.ceil(lh / span)
  const c0 = clamp(Math.floor(x0 / fx / span), 0, cols - 1)
  const c1 = clamp(Math.ceil(x1 / fx / span) - 1, 0, cols - 1)
  const r0 = clamp(Math.floor(y0 / fy / span), 0, rows - 1)
  const r1 = clamp(Math.ceil(y1 / fy / span) - 1, 0, rows - 1)
  const out: TileRef[] = []
  for (let y = r0; y <= r1; y++) {
    for (let x = c0; x <= c1; x++) {
      out.push({ z, x, y, key: `${z}/${x}_${y}`, box: { x: x * span * fx, y: y * span * fy, w: Math.min(span, lw - x * span) * fx, h: Math.min(span, lh - y * span) * fy } })
    }
  }
  return out
}

/** Least-recently-used cache; `onEvict` releases evicted values. */
export class LruCache<V> {
  private readonly map = new Map<string, V>()

  constructor(readonly capacity: number, private readonly onEvict?: (v: V) => void) {}

  get(key: string): V | undefined {
    const v = this.map.get(key)
    if (v !== undefined) {
      this.map.delete(key)
      this.map.set(key, v)
    }
    return v
  }

  has(key: string): boolean {
    return this.map.has(key)
  }

  set(key: string, value: V): void {
    if (this.map.has(key)) this.map.delete(key)
    this.map.set(key, value)
    while (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value as string
      const v = this.map.get(oldest) as V
      this.map.delete(oldest)
      this.onEvict?.(v)
    }
  }

  get size(): number {
    return this.map.size
  }

  clear(): void {
    for (const v of this.map.values()) this.onEvict?.(v)
    this.map.clear()
  }
}

export type TileLoader = (url: string) => Promise<ImageBitmap>

const loadBitmap: TileLoader = async (url) => {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`tile ${url.split('?')[0]}: HTTP ${r.status}`)
  return createImageBitmap(await r.blob())
}

export class TileSource {
  private readonly cache: LruCache<ImageBitmap>
  private readonly inflight = new Set<string>()
  private readonly failed = new Set<string>()
  private queue: TileRef[] = []
  private active = 0
  private disposed = false

  constructor(
    readonly info: TileSetInfo,
    private readonly urlFor: (z: number, x: number, y: number) => string,
    private readonly onTile: () => void,
    private readonly onError: (message: string) => void = () => undefined,
    capacity = TILE_CACHE_SIZE,
    private readonly load: TileLoader = loadBitmap,
  ) {
    this.cache = new LruCache<ImageBitmap>(capacity, (b) => b.close())
  }

  get cached(): number {
    return this.cache.size
  }

  private request(tiles: readonly TileRef[]): void {
    const fresh = tiles.filter((t) => !this.cache.has(t.key) && !this.inflight.has(t.key) && !this.failed.has(t.key) && !this.queue.some((q) => q.key === t.key))
    if (!fresh.length) return
    this.queue.push(...fresh)
    this.pump()
  }

  private pump(): void {
    while (!this.disposed && this.active < CONCURRENCY && this.queue.length) {
      const t = this.queue.shift() as TileRef
      this.active++
      this.inflight.add(t.key)
      this.load(this.urlFor(t.z, t.x, t.y))
        .then((bitmap) => {
          if (this.disposed) bitmap.close()
          else {
            this.cache.set(t.key, bitmap)
            this.onTile()
          }
        })
        .catch((e: unknown) => {
          this.failed.add(t.key)
          if (!this.disposed) this.onError(e instanceof Error ? e.message : String(e))
        })
        .finally(() => {
          this.active--
          this.inflight.delete(t.key)
          this.pump()
        })
    }
  }

  /** Draws the visible tiles (coarse levels first) in viewport CSS px; returns whether the chosen level is complete. */
  draw(ctx: CanvasRenderingContext2D, view: View, viewport: Size, dpr = 1): boolean {
    const z = levelFor(this.info, view.scale, dpr)
    const needed = visibleTiles(this.info, z, view, viewport)
    // Drop queued tiles that left the viewport, then ask for the background level and the chosen one.
    const wanted = new Set(needed.map((t) => t.key))
    this.queue = this.queue.filter((t) => t.z === 0 || wanted.has(t.key))
    this.request(visibleTiles(this.info, 0, view, viewport))
    this.request(needed)
    for (let lz = 0; lz <= z; lz++) {
      for (const t of lz === z ? needed : visibleTiles(this.info, lz, view, viewport)) {
        const bitmap = this.cache.get(t.key)
        if (!bitmap) continue
        const b = boxToViewport(view, t.box)
        // Snap outwards to whole px so neighbouring tiles never leave a hairline seam.
        const x0 = Math.floor(b.x)
        const y0 = Math.floor(b.y)
        ctx.drawImage(bitmap, x0, y0, Math.ceil(b.x + b.w) - x0, Math.ceil(b.y + b.h) - y0)
      }
    }
    return needed.every((t) => this.cache.has(t.key) || this.failed.has(t.key))
  }

  dispose(): void {
    this.disposed = true
    this.queue = []
    this.cache.clear()
  }
}
