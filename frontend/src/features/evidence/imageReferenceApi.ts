import { authHeaders, unwrap } from '../../lib/api/client'

interface ImageSource {
  sha256: string
  width: number
  height: number
}

export interface ReferencePng extends ImageSource {
  file: string
}

export interface NativeImageArea {
  x: number
  y: number
  w: number
  h: number
}

interface DifferenceRegion extends NativeImageArea {
  reference_region?: NativeImageArea
  reference_crop?: ReferencePng
  submitted_crop?: ReferencePng
  submitted_overlay?: ReferencePng
}

interface OrientedCropProof {
  transform: keyof typeof orientationLabels
  native_crop: NativeImageArea
  projected_crop: NativeImageArea
  matched_pixels: number
  matched_tiles: [number, number][]
  projected_rgb_sha256: string
  difference_mask_sha256: string
}

export interface ImageReferenceRecord {
  schema: string
  status: string
  reference: ImageSource
  submitted: ImageSource
  regions: DifferenceRegion[]
  mask?: ReferencePng
  alignment?: {
    submitted_overlap: NativeImageArea
    reference_overlap: NativeImageArea
  }
  submitted_over_reference_xy?: [[number, number], [number, number]]
  reference_crop_bounds?: NativeImageArea
  supported_resize_kernels?: string[]
  exact_projection?: {
    matched_pixels: number
    matched_tiles: [number, number][]
    rgb_sha256: string
  }
  exact_orientation?: {
    transform: keyof typeof orientationLabels
    matched_pixels: number
    matched_tiles: [number, number][]
    rgb_sha256: string
  }
  exact_oriented_crop?: OrientedCropProof
  exact_oriented_resize?: OrientedCropProof & { kernels: string[] }
  embedded_native_photo?: {
    submitted_bounds: NativeImageArea
    reference_bounds: NativeImageArea
    exact_pixels: number
    witness_cells: [number, number, number][]
    native_difference_sha256: string
  }
  additional_regions?: number
}

export const exactPngCropResizeSchema = 'chipmunk.image-reference-comparison/5'
export const exactPngOrientationSchema = 'chipmunk.image-reference-comparison/6'
export const exactOrientedCropSchema = 'chipmunk.image-reference-comparison/7'
export const exactOrientedResizeSchema = 'chipmunk.image-reference-comparison/8'
export const embeddedPhotoSchema = 'chipmunk.image-reference-comparison/9'
export const jpegPngEnvelopeSchema = 'chipmunk.image-reference-comparison/10'
export const orientationLabels = {
  rotate_90_clockwise: '90-degree clockwise rotation',
  rotate_180: '180-degree rotation',
  rotate_90_counterclockwise: '90-degree counterclockwise rotation',
  mirror_horizontal: 'left-to-right reflection',
  mirror_vertical: 'top-to-bottom reflection',
  reflect_main_diagonal: 'main-diagonal reflection',
  reflect_other_diagonal: 'opposite-diagonal reflection',
} as const
const swapsImageAxes = new Set(['rotate_90_clockwise', 'rotate_90_counterclockwise', 'reflect_main_diagonal', 'reflect_other_diagonal'])
const exactResizeKernels = new Set(['bicubic', 'bilinear', 'lanczos', 'opencv_area'])

export interface ImageReferenceResult {
  comparison: ImageReferenceRecord
  pngs: Record<string, string>
  pdf?: string
}

const comparisonPath = (analysisId: string) => `/api/v2/analyses/${encodeURIComponent(analysisId)}/image-reference-comparison`

async function request<T>(path: string, signal: AbortSignal, body?: FormData): Promise<T> {
  const method = body ? 'POST' : 'GET'
  const response = await fetch(path, { method, headers: authHeaders(method), signal, body })
  const data: unknown = await response.json().catch(() => null)
  signal.throwIfAborted()
  return unwrap(Promise.resolve({
    response,
    data: response.ok ? data as T : undefined,
    error: response.ok ? undefined : data,
  }))
}

export async function comparisonAvailability(analysisId: string, signal: AbortSignal): Promise<boolean> {
  const result = await request<{ available?: boolean } | null>(comparisonPath(analysisId), signal)
  return result?.available === true
}

function validArea(area?: NativeImageArea): area is NativeImageArea {
  return !!area && [area.x, area.y, area.w, area.h].every(Number.isSafeInteger)
    && area.x >= 0 && area.y >= 0 && area.w > 0 && area.h > 0
}

function validAreaWithin(area: NativeImageArea, width: number, height: number): boolean {
  return validArea(area) && area.x + area.w <= width && area.y + area.h <= height
}

export async function compareWithEarlierImage(analysisId: string, file: File, signal: AbortSignal): Promise<ImageReferenceResult> {
  const form = new FormData()
  form.append('reference', file)
  form.append('include_pdf', 'true')
  const result = await request<ImageReferenceResult | null>(comparisonPath(analysisId), signal, form)
  const comparison = result?.comparison
  if (typeof comparison?.schema !== 'string' || !comparison.schema.startsWith('chipmunk.image-reference-comparison/')
      || typeof comparison.status !== 'string' || !comparison.reference || !comparison.submitted
      || !Array.isArray(comparison.regions) || comparison.regions.some(region => !region)
      || !result?.pngs || typeof result.pngs !== 'object' || Array.isArray(result.pngs)) {
    throw new Error('The server returned an incomplete image comparison.')
  }
  const proof = comparison.exact_projection
  if (comparison.schema === exactPngCropResizeSchema && (!comparison.reference_crop_bounds
      || !proof || !Number.isSafeInteger(proof.matched_pixels) || !Array.isArray(proof.matched_tiles)
      || !Array.isArray(comparison.supported_resize_kernels)
      || !comparison.supported_resize_kernels.every(kernel => typeof kernel === 'string'))) {
    throw new Error('The server returned an incomplete exact resize comparison.')
  }
  const orientation = comparison.exact_orientation
  if (comparison.schema === exactPngOrientationSchema && (comparison.status !== 'comparable' || !orientation
      || !Object.prototype.hasOwnProperty.call(orientationLabels, orientation.transform)
      || !Number.isSafeInteger(orientation.matched_pixels) || !Array.isArray(orientation.matched_tiles))) {
    throw new Error('The server returned an incomplete orientation comparison.')
  }
  const orientedCrop = comparison.exact_oriented_crop
  if (comparison.schema === exactOrientedCropSchema && (comparison.status !== 'comparable' || !orientedCrop
      || !Object.prototype.hasOwnProperty.call(orientationLabels, orientedCrop.transform)
      || !validArea(orientedCrop.native_crop) || !validArea(orientedCrop.projected_crop)
      || orientedCrop.projected_crop.w !== comparison.submitted.width || orientedCrop.projected_crop.h !== comparison.submitted.height
      || orientedCrop.native_crop.x + orientedCrop.native_crop.w > comparison.reference.width
      || orientedCrop.native_crop.y + orientedCrop.native_crop.h > comparison.reference.height
      || !Number.isSafeInteger(orientedCrop.matched_pixels) || !Array.isArray(orientedCrop.matched_tiles))) {
    throw new Error('The server returned an incomplete oriented crop comparison.')
  }
  if (comparison.schema === exactOrientedResizeSchema) {
    const proof = comparison.exact_oriented_resize
    const swap = proof && swapsImageAxes.has(proof.transform)
    const { width, height } = comparison.reference
    if (comparison.status !== 'comparable' || !proof || !Object.prototype.hasOwnProperty.call(orientationLabels, proof.transform)
        || !validAreaWithin(proof.native_crop, width, height)
        || !validAreaWithin(proof.projected_crop, swap ? height : width, swap ? width : height)
        || !Number.isSafeInteger(proof.matched_pixels) || proof.matched_pixels < 1
        || proof.matched_pixels > comparison.submitted.width * comparison.submitted.height
        || !Array.isArray(proof.matched_tiles) || !Array.isArray(proof.kernels) || proof.kernels.length < 1
        || !proof.kernels.every(kernel => exactResizeKernels.has(kernel))) {
      throw new Error('The server returned an incomplete oriented resize comparison.')
    }
  }
  if (comparison.schema === embeddedPhotoSchema) {
    const proof = comparison.embedded_native_photo
    const earlier = proof?.reference_bounds
    const photo = proof?.submitted_bounds
    if (comparison.status !== 'comparable' || !proof || !earlier || !photo
        || !validAreaWithin(earlier, comparison.reference.width, comparison.reference.height)
        || !validAreaWithin(photo, comparison.submitted.width, comparison.submitted.height)
        || earlier.w !== photo.w || earlier.h !== photo.h
        || !Number.isSafeInteger(proof.exact_pixels) || proof.exact_pixels < 1 || proof.exact_pixels > photo.w * photo.h
        || !Array.isArray(proof.witness_cells) || proof.witness_cells.length < 12
        || typeof proof.native_difference_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(proof.native_difference_sha256)
        || comparison.regions.some(region => {
          const original = region.reference_region
          return !validArea(region) || !validArea(original) || original.w !== region.w || original.h !== region.h
            || original.x !== earlier.x + region.x - photo.x || original.y !== earlier.y + region.y - photo.y
            || region.x < photo.x || region.y < photo.y || region.x + region.w > photo.x + photo.w
            || region.y + region.h > photo.y + photo.h
        })) {
      throw new Error('The server returned an incomplete screenshot comparison.')
    }
  }
  if (comparison.schema === jpegPngEnvelopeSchema) {
    const crop = comparison.reference_crop_bounds
    if (comparison.status !== 'comparable' || !crop
        || !validAreaWithin(crop, comparison.reference.width, comparison.reference.height)
        || comparison.regions.some(region => !validAreaWithin(region, comparison.submitted.width, comparison.submitted.height)
          || !validArea(region.reference_region)
          || !validAreaWithin({ ...region.reference_region, x: region.reference_region.x - crop.x,
            y: region.reference_region.y - crop.y }, crop.w, crop.h))) {
      throw new Error('The server returned an incomplete JPEG to PNG comparison.')
    }
  }
  return result
}

function validBase64(value: unknown, prefix: string): value is string {
  return typeof value === 'string' && value.startsWith(prefix) && value.length % 4 === 0
    && /^[A-Za-z0-9+/]+={0,2}$/.test(value)
}

function pngData(pngs: ImageReferenceResult['pngs'], artifact?: ReferencePng): string | null {
  if (!artifact || !/^[a-z][a-z0-9-]{0,64}\.png$/.test(artifact.file)
      || !Object.prototype.hasOwnProperty.call(pngs, artifact.file)) return null
  const value = pngs[artifact.file]
  return validBase64(value, 'iVBORw0KGgo') ? value : null
}

export function referencePngSrc(pngs: ImageReferenceResult['pngs'], artifact?: ReferencePng): string | null {
  const base64 = pngData(pngs, artifact)
  return base64 ? `data:image/png;base64,${base64}` : null
}

export function imageReferenceExport(result: ImageReferenceResult): ImageReferenceResult {
  const { comparison, pngs } = result
  const artifacts = [comparison.mask, ...comparison.regions.flatMap(region => [region.reference_crop, region.submitted_crop, region.submitted_overlay])]
  const attached: ImageReferenceResult['pngs'] = {}
  for (const artifact of artifacts) {
    const base64 = pngData(pngs, artifact)
    if (artifact && base64) attached[artifact.file] = base64
  }
  return { comparison, pngs: attached }
}

export function comparisonPdfData(result: ImageReferenceResult): string | null {
  return validBase64(result.pdf, 'JVBERi0') ? result.pdf : null
}
