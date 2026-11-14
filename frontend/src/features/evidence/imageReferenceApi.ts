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
  selected_pixels?: number
  directly_measured_tiles?: number
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
    method?: string
    full_tiles?: number
    exact_tiles?: number
    anchor_count?: number
    supported_zones?: [number, number][]
    anchor_sha256?: string
    reference_decoded_rgb_sha256?: string
    jpeg_quality?: number
    reference_minus_submitted?: { x: number; y: number }
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
  same_framing_structure?: {
    detailed_tiles: number
    aligned_tiles: [number, number][]
    background_support: [number, number]
  }
  affects_overall_confidence?: boolean
  additional_regions?: number
}

export const exactPngCropResizeSchema = 'chipmunk.image-reference-comparison/5'
export const exactPngOrientationSchema = 'chipmunk.image-reference-comparison/6'
export const exactOrientedCropSchema = 'chipmunk.image-reference-comparison/7'
export const exactOrientedResizeSchema = 'chipmunk.image-reference-comparison/8'
export const embeddedPhotoSchema = 'chipmunk.image-reference-comparison/9'
export const jpegPngEnvelopeSchema = 'chipmunk.image-reference-comparison/10'
export const exactJpegIdentitySchema = 'chipmunk.image-reference-comparison/11'
export const shiftedJpegSchema = 'chipmunk.image-reference-comparison/12'
export const structureJpegSchema = 'chipmunk.image-reference-comparison/13'
export const sameJpegSchema = 'chipmunk.image-reference-comparison/14'
export const jpegDetailLossSchema = 'chipmunk.image-reference-comparison/15'
export const extendedJpegDetailLossSchema = 'chipmunk.image-reference-comparison/16'
export const isJpegDetailLoss = (schema: string) =>
  schema === jpegDetailLossSchema || schema === extendedJpegDetailLossSchema
export const IMAGE_REFERENCE_SECTION_ID = 'earlier-image-comparison'
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

function validSameJpegProof(comparison: ImageReferenceRecord): boolean {
  const { width, height } = comparison.submitted
  const proof = comparison.alignment
  const detailLoss = isJpegDetailLoss(comparison.schema)
  if (!proof || comparison.status !== 'comparable' || comparison.affects_overall_confidence !== false
      || proof.method !== (detailLoss ? 'exact_jpeg_pair_local_detail_loss' : 'exact_decoded_jpeg_pair_identity')
      || comparison.reference.width !== width || comparison.reference.height !== height
      || ![width, height].every(value => Number.isSafeInteger(value) && value >= 512 && value <= 768)
      || ![proof.submitted_overlap, proof.reference_overlap].every(area =>
        validArea(area) && area.x === 0 && area.y === 0 && area.w === width && area.h === height)
      || proof.reference_minus_submitted?.x !== 0 || proof.reference_minus_submitted.y !== 0
      || (comparison.schema === extendedJpegDetailLossSchema && proof.jpeg_quality !== 75 && proof.jpeg_quality !== 85)) return false
  const full = proof.full_tiles ?? 0
  const exact = proof.exact_tiles ?? 0
  const anchors = proof.anchor_count ?? 0
  const zones = proof.supported_zones
  if (detailLoss && (!comparison.regions.length || !comparison.mask || comparison.regions.some(region => {
    const pixels = region.selected_pixels ?? 0
    const direct = region.directly_measured_tiles ?? 0
    return ![pixels, direct].every(Number.isSafeInteger) || pixels % 256 !== 0 || pixels < 1024
      || pixels > region.w * region.h || direct < 1 || direct > pixels / 256
      || [region.x, region.y, region.w, region.h].some(value => value % 16 !== 0)
  }))) return false
  return [full, exact, anchors].every(Number.isSafeInteger)
    && full === Math.floor(width / 16) * Math.floor(height / 16) && full >= exact && exact >= .85 * full
    && exact >= anchors && anchors >= 24 && Array.isArray(zones)
    && zones.every(zone => Array.isArray(zone) && zone.length === 2 && zone.every(value => [0, 1, 2].includes(value)))
    && new Set(zones.map(zone => zone.join(','))).size === zones.length && zones.length >= 6
    && [0, 1].every(axis => new Set(zones.map(zone => zone[axis])).size === 3)
    && [proof.anchor_sha256, proof.reference_decoded_rgb_sha256].every(value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value))
    && comparison.regions.every(region => {
      const earlier = region.reference_region
      return validAreaWithin(region, width, height) && !!earlier
        && earlier.x === region.x && earlier.y === region.y && earlier.w === region.w && earlier.h === region.h
    })
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
  if (comparison.schema === exactJpegIdentitySchema) {
    const alignment = comparison.alignment
    const { width, height } = comparison.submitted
    if (comparison.status !== 'comparable' || alignment?.method !== 'exact_decoded_jpeg_delivery_identity'
        || comparison.reference.width !== width || comparison.reference.height !== height
        || ![alignment.submitted_overlap, alignment.reference_overlap].every(area =>
          validArea(area) && area.x === 0 && area.y === 0 && area.w === width && area.h === height)
        || !Number.isSafeInteger(alignment.full_tiles) || !Number.isSafeInteger(alignment.exact_tiles)
        || !Number.isSafeInteger(alignment.anchor_count) || !Array.isArray(alignment.supported_zones)) {
      throw new Error('The server returned an incomplete JPEG alignment comparison.')
    }
  }
  if (comparison.schema === shiftedJpegSchema) {
    const alignment = comparison.alignment
    const before = alignment?.reference_overlap
    const after = alignment?.submitted_overlap
    const shift = alignment?.reference_minus_submitted
    const full = alignment?.full_tiles ?? 0
    const exact = alignment?.exact_tiles ?? 0
    const anchors = alignment?.anchor_count ?? 0
    if (comparison.status !== 'comparable' || alignment?.method !== 'exact_decoded_jpeg_translated_overlap'
        || !before || !after || !shift || ![shift.x, shift.y].every(Number.isSafeInteger)
        || Math.max(Math.abs(shift.x), Math.abs(shift.y)) < 1 || Math.max(Math.abs(shift.x), Math.abs(shift.y)) > 24
        || comparison.reference.width !== comparison.submitted.width || comparison.reference.height !== comparison.submitted.height
        || !validAreaWithin(before, comparison.reference.width, comparison.reference.height)
        || !validAreaWithin(after, comparison.submitted.width, comparison.submitted.height)
        || before.w !== after.w || before.h !== after.h || before.x !== after.x + shift.x || before.y !== after.y + shift.y
        || [after.x, after.y, after.w, after.h].some(value => value % 16 !== 0)
        || ![full, exact, anchors].every(Number.isSafeInteger) || full !== after.w * after.h / 256
        || full < exact || exact < .65 * full || anchors < 24 || anchors > exact
        || comparison.regions.some(region => {
          const original = region.reference_region
          return !validArea(region) || !validArea(original)
            || !validAreaWithin({ ...region, x: region.x - after.x, y: region.y - after.y }, after.w, after.h)
            || original.w !== region.w || original.h !== region.h
            || original.x !== region.x + shift.x || original.y !== region.y + shift.y
        })) {
      throw new Error('The server returned an incomplete shifted JPEG comparison.')
    }
  }
  if (comparison.schema === structureJpegSchema) {
    const { width, height } = comparison.submitted
    const proof = comparison.same_framing_structure
    if (comparison.status !== 'comparable' || comparison.affects_overall_confidence !== false
        || comparison.reference.width !== width || comparison.reference.height !== height
        || ![width, height].every(value => Number.isSafeInteger(value) && value >= 512 && value <= 768)
        || !proof || !Number.isSafeInteger(proof.detailed_tiles) || !Array.isArray(proof.aligned_tiles)
        || proof.aligned_tiles.length < Math.max(24, .65 * proof.detailed_tiles) || proof.aligned_tiles.length > proof.detailed_tiles
        || !Array.isArray(proof.background_support) || proof.background_support.length !== 2
        || !proof.background_support.every(value => Number.isFinite(value) && value >= .80 && value <= 1)
        || comparison.regions.some(region => {
          const earlier = region.reference_region
          return !validAreaWithin(region, width, height) || !earlier
            || earlier.x !== region.x || earlier.y !== region.y || earlier.w !== region.w || earlier.h !== region.h
        })) {
      throw new Error('The server returned an incomplete JPEG structure comparison.')
    }
  }
  if ((comparison.schema === sameJpegSchema || isJpegDetailLoss(comparison.schema)) && !validSameJpegProof(comparison)) {
    throw new Error('The server returned an incomplete exact-block JPEG comparison.')
  }
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
