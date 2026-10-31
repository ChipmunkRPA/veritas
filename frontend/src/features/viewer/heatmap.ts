/**
 * Heat layer rendering (UX §8.3, ARCHITECTURE §7.4), shared by `heatmap.worker.ts` and its in-thread fallback.
 *
 * - **Payloads.** An 8-bit grayscale PNG over `value_range` and an optional reliability mask (8-bit, 255 = reliable), decoded once
 *   per exhibit.
 * - **Colour.** The inferno LUT with the `Show ≥` threshold and the alpha ramp at full opacity, alpha multiplied by mask/255, so an
 *   unreliable area never reads as evidence. The viewer applies the lens opacity with `globalAlpha` (the ramp is linear in
 *   opacity), so a toggle or an opacity change is a redraw and only a threshold change renders again: rendered layers are cached
 *   per (exhibit, threshold).
 * - **Unreliable areas** (mask < 128) become an alpha stencil for the reliability hatch and "Dim unreliable".
 */
import { buildLut, colourise } from '../../lib/colormap'

export interface DecodedMap { width: number; height: number; values: Uint8Array; mask: Uint8Array | null }
export interface ProbeValue { raw: number; value: number; reliable: boolean | null }

export const UNRELIABLE_BELOW = 128

/** The byte channel of decoded RGBA pixels (the payloads are grayscale, so R = G = B). */
export function grayChannel(rgba: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(rgba.length / 4)
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4]
  return out
}

/** RGBA of the heat layer at full opacity. */
export function heatPixels(map: DecodedMap, lut: Uint8ClampedArray, show: number): Uint8ClampedArray {
  return colourise(map.values, lut, { show, opacity: 1, mask: map.mask })
}

/** Opaque white where the mask marks a cell unreliable, transparent elsewhere; null without a mask or unreliable cells. */
export function unreliablePixels(map: DecodedMap): Uint8ClampedArray | null {
  if (!map.mask) return null
  const out = new Uint8ClampedArray(map.mask.length * 4)
  let any = false
  for (let i = 0; i < map.mask.length; i++) {
    if (map.mask[i] >= UNRELIABLE_BELOW) continue
    out.fill(255, i * 4, i * 4 + 4)
    any = true
  }
  return any ? out : null
}

/** The value of a map cell: the byte mapped linearly over `value_range`, and the cell's reliability (null without a mask). */
export function probeMap(map: DecodedMap, cell: { x: number; y: number }, valueRange: readonly number[]): ProbeValue {
  const i = cell.y * map.width + cell.x
  const raw = map.values[i]
  const [lo, hi] = valueRange
  return { raw, value: lo + (raw / 255) * (hi - lo), reliable: map.mask ? map.mask[i] >= UNRELIABLE_BELOW : null }
}

async function decodeGray(blob: Blob): Promise<{ width: number; height: number; values: Uint8Array }> {
  // No colour management: the bytes are values, not colours.
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' })
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new Error('2D canvas unavailable')
    ctx.drawImage(bitmap, 0, 0)
    return { width: bitmap.width, height: bitmap.height, values: grayChannel(ctx.getImageData(0, 0, bitmap.width, bitmap.height).data) }
  } finally {
    bitmap.close()
  }
}

export async function decodeMap(payload: Blob, mask: Blob | null): Promise<DecodedMap> {
  const p = await decodeGray(payload)
  if (!mask) return { ...p, mask: null }
  const m = await decodeGray(mask)
  if (m.width !== p.width || m.height !== p.height) {
    throw new Error(`The reliability mask (${m.width}×${m.height}) does not match the map (${p.width}×${p.height}).`)
  }
  return { ...p, mask: m.values }
}

export function pixelsToBitmap(pixels: Uint8ClampedArray, width: number, height: number): ImageBitmap {
  const canvas = new OffscreenCanvas(width, height)
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2D canvas unavailable')
  ctx.putImageData(new ImageData(pixels, width, height), 0, 0)
  return canvas.transferToImageBitmap()
}

/** Decoded maps by key; renders layers and stencils. Runs inside the worker, or on the main thread when workers are unavailable. */
export class HeatEngine {
  private readonly lut = buildLut()
  private readonly maps = new Map<string, DecodedMap>()

  async decode(key: string, payload: Blob, mask: Blob | null): Promise<{ map: DecodedMap; unreliable: ImageBitmap | null }> {
    const map = await decodeMap(payload, mask)
    this.maps.set(key, map)
    const u = unreliablePixels(map)
    return { map, unreliable: u ? pixelsToBitmap(u, map.width, map.height) : null }
  }

  render(key: string, show: number): ImageBitmap {
    const map = this.maps.get(key)
    if (!map) throw new Error(`map ${key} is not decoded`)
    return pixelsToBitmap(heatPixels(map, this.lut, show), map.width, map.height)
  }

  release(key: string): void {
    this.maps.delete(key)
  }
}
