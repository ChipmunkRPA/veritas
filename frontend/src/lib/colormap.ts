/**
 * Heat colormap and alpha ramp (UX §8.1, §8.3; ARCHITECTURE §7.4).
 *
 * Inferno at 11 stops (the `--heat-NN` tokens), linearly interpolated in sRGB to a 256-entry LUT. The ramp is
 * monotonic in CIE L*, so it survives grayscale print and colour-vision deficiency. Values below the show threshold
 * are transparent so the source stays readable; above it alpha rises from 35 % to 100 % of the chosen opacity.
 */

export const INFERNO_STOPS = [
  '#000004', '#160B39', '#420A68', '#6A176E', '#932667', '#BC3754', '#DD513A', '#F37819', '#FCA50A', '#F6D746', '#FCFFA4',
] as const

/** Default `Show ≥` on an overlay; "map only" shows everything. The slider runs 0–0.9 in steps of 0.05. */
export const SHOW_THRESHOLD_OVERLAY = 0.5
export const SHOW_THRESHOLD_MAP_ONLY = 0
export const SHOW_THRESHOLD_MAX = 0.9

export type Rgb = readonly [number, number, number]

export function hexToRgb(hex: string): Rgb {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) throw new Error(`not a #RRGGBB colour: ${hex}`)
  const n = parseInt(m[1], 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

/** 256 × RGB, entry i for the 8-bit map value i (t = i / 255). */
export function buildLut(stops: readonly string[] = INFERNO_STOPS): Uint8ClampedArray {
  if (stops.length < 2) throw new Error('a colormap needs at least two stops')
  const rgb = stops.map(hexToRgb)
  const segments = rgb.length - 1
  const lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const pos = (i * segments) / 255
    const k = Math.min(segments - 1, Math.floor(pos))
    const f = pos - k
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(rgb[k][c] + (rgb[k + 1][c] - rgb[k][c]) * f)
  }
  return lut
}

/** Alpha in [0, 1] for a map value t (§8.3): 0 below `show`, else opacity × (0.35 + 0.65 × (t − show)/(1 − show)). */
export function heatAlpha(t: number, show: number, opacity: number): number {
  const v = Math.min(1, Math.max(0, t))
  const s = Math.min(SHOW_THRESHOLD_MAX, Math.max(0, show))
  const o = Math.min(1, Math.max(0, opacity))
  if (v < s) return 0
  return o * (0.35 + (0.65 * (v - s)) / (1 - s))
}

/** 256 alpha bytes for one (show, opacity) pair; recomputed only when either changes. */
export function alphaTable(show: number, opacity: number): Uint8ClampedArray {
  const table = new Uint8ClampedArray(256)
  for (let i = 0; i < 256; i++) table[i] = Math.round(heatAlpha(i / 255, show, opacity) * 255)
  return table
}

/**
 * Colourise an 8-bit map into RGBA (ImageData layout). The optional reliability mask (0–255 per pixel) multiplies
 * alpha by mask/255, as the heatmap worker does.
 */
export function colourise(
  values: ArrayLike<number>,
  lut: Uint8ClampedArray,
  opts: { show: number; opacity: number; mask?: ArrayLike<number> | null },
  out: Uint8ClampedArray = new Uint8ClampedArray(values.length * 4),
): Uint8ClampedArray {
  if (lut.length !== 256 * 3) throw new Error('LUT must hold 256 RGB entries')
  if (out.length !== values.length * 4) throw new Error('output buffer must hold 4 bytes per map value')
  if (opts.mask && opts.mask.length !== values.length) throw new Error('mask and map sizes differ')
  const alpha = alphaTable(opts.show, opts.opacity)
  for (let i = 0; i < values.length; i++) {
    const v = Math.min(255, Math.max(0, values[i] | 0))
    const j = i * 4
    out[j] = lut[v * 3]
    out[j + 1] = lut[v * 3 + 1]
    out[j + 2] = lut[v * 3 + 2]
    out[j + 3] = opts.mask ? Math.round((alpha[v] * opts.mask[i]) / 255) : alpha[v]
  }
  return out
}

/** CIE L* (D65) of an sRGB colour, 0–100. */
export function lightness([r, g, b]: Rgb): number {
  const lin = (c: number) => {
    const s = c / 255
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  const y = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
  return y > 216 / 24389 ? 116 * Math.cbrt(y) - 16 : (24389 / 27) * y
}

/** CSS `rgb()` string for LUT entry i, used for vector legends. */
export function lutCss(lut: Uint8ClampedArray, i: number): string {
  const k = Math.min(255, Math.max(0, Math.round(i))) * 3
  return `rgb(${lut[k]} ${lut[k + 1]} ${lut[k + 2]})`
}
