import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button } from '../../components/ui'
import { saveBlob } from '../../lib/api/reports'
import {
  compareWithEarlierImage, comparisonAvailability, comparisonPdfData, exactOrientedCropSchema, exactOrientedResizeSchema,
  exactPngCropResizeSchema, exactPngOrientationSchema, embeddedPhotoSchema, exactJpegIdentitySchema, jpegPngEnvelopeSchema, shiftedJpegSchema,
  IMAGE_REFERENCE_SECTION_ID, imageReferenceExport, isJpegDetailLoss, orientationLabels, referencePngSrc, sameJpegSchema, structureJpegSchema,
  type ImageReferenceResult, type NativeImageArea, type ReferencePng,
} from './imageReferenceApi'

type ComparisonState =
  | { status: 'idle' | 'pending' }
  | { status: 'error'; message: string }
  | { status: 'done'; result: ImageReferenceResult }

const unassessed: Record<string, string> = {
  different_dimensions: 'The copies have different dimensions and a common area could not be measured.',
  unsupported_dimensions: 'The image dimensions are outside the supported comparison range.',
  unaligned: 'The copies could not be aligned reliably.',
  integer_alignment_unresolved: 'The copies could not be aligned reliably on their pixel grids.',
  insufficient_common_background: 'Too little matching background was available for a reliable comparison.',
  partial_jpeg_delivery_unassessed: 'The submitted JPEG extends outside the aligned area, so its compression could not be compared reliably.',
  uncertain_delivery: 'The available compression comparisons did not provide enough support.',
  unsupported_resize_geometry: 'The copies are outside the supported size and proportion limits for a resize comparison.',
  same_framing_unresolved: 'The copies could not be reliably matched to the same framing after resizing.',
  resize_delivery_unresolved: 'Too few resize comparisons agreed on the image delivery.',
}

function NativeArea({ area }: { area: NativeImageArea }) {
  return <span className="font-mono">x {area.x}, y {area.y} · {area.w} × {area.h} px</span>
}

function ComparisonPicture({ result, artifact, label, overlay }: {
  result: ImageReferenceResult; artifact?: ReferencePng; label: string; overlay?: ReferencePng
}) {
  const [failed, setFailed] = useState(false)
  const [showOverlay, setShowOverlay] = useState(false)
  const [overlayFailed, setOverlayFailed] = useState(false)
  const src = referencePngSrc(result.pngs, artifact)
  const overlaySrc = overlay && overlay.width === artifact?.width && overlay.height === artifact.height
    ? referencePngSrc(result.pngs, overlay) : null
  return <figure className="m-0 min-w-0">
    <figcaption className="mb-2 text-small font-medium text-text">{label}</figcaption>
    <div className="flex min-h-24 items-center justify-center overflow-hidden rounded-md border border-border bg-surface-2">
      {src && artifact && !failed
        ? <div className="relative w-full">
            <img className="block h-auto max-h-80 w-full object-contain" src={src} alt={label}
              width={artifact.width} height={artifact.height} loading="lazy" onError={() => setFailed(true)} />
            {showOverlay && overlaySrc && !overlayFailed &&
              <img className="pointer-events-none absolute inset-0 h-full w-full object-contain" src={overlaySrc}
                alt="" onError={() => setOverlayFailed(true)} />}
          </div>
        : <p className="m-4 text-small text-text-muted">This evidence image is unavailable.</p>}
    </div>
    {src && !failed && overlaySrc && !overlayFailed && <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
      <Button variant="text" aria-pressed={showOverlay} onClick={() => setShowOverlay(value => !value)}>Highlight difference pixels</Button>
      {showOverlay && <p className="m-0 text-caption text-text-muted">Amber marks the selected pixels in this crop.</p>}
    </div>}
    {overlayFailed && <p role="status" className="mt-2 mb-0 text-caption text-text-muted">The difference overlay could not be loaded.</p>}
  </figure>
}

function ComparisonResult({ result }: { result: ImageReferenceResult }) {
  const { comparison } = result
  const { reference, submitted, regions, mask, alignment } = comparison
  const comparable = comparison.status === 'comparable'
  const exact = comparison.schema === exactPngCropResizeSchema ? comparison.exact_projection : undefined
  const orientedResize = comparison.schema === exactOrientedResizeSchema ? comparison.exact_oriented_resize : undefined
  const orientedCrop = comparison.schema === exactOrientedCropSchema ? comparison.exact_oriented_crop : orientedResize
  const orientation = comparison.schema === exactPngOrientationSchema ? comparison.exact_orientation : orientedCrop
  const embeddedPhoto = comparison.schema === embeddedPhotoSchema ? comparison.embedded_native_photo : undefined
  const delivery = comparison.schema === jpegPngEnvelopeSchema
  const sameJpegFraming = comparison.schema === exactJpegIdentitySchema ? alignment : undefined
  const detailLoss = isJpegDetailLoss(comparison.schema)
  const sameJpegBlocks = (comparison.schema === sameJpegSchema || detailLoss) ? alignment : undefined
  const shiftedJpeg = comparison.schema === shiftedJpegSchema ? alignment : undefined
  const sameStructureFraming = comparison.schema === structureJpegSchema ? comparison.same_framing_structure : undefined
  const cropped = comparison.schema === 'chipmunk.image-reference-comparison/4' || exact || delivery
    ? comparison.reference_crop_bounds : undefined
  const resized = comparable && (Boolean(cropped) || comparison.schema === 'chipmunk.image-reference-comparison/3')
  const completeMatch = (exact ?? orientation)?.matched_pixels === submitted.width * submitted.height
    || (embeddedPhoto && embeddedPhoto.exact_pixels === embeddedPhoto.submitted_bounds.w * embeddedPhoto.submitted_bounds.h)
  const kernels = (orientedResize?.kernels ?? comparison.supported_resize_kernels)?.map(kernel => kernel === 'opencv_area' ? 'area' : kernel).join(', ')
  const maskDescription = delivery ? 'White marks color differences outside the expected delivery range.'
    : (exact || orientedResize) ? 'White marks retained differences from the reproduced resize.'
    : resized ? 'White marks differences flagged by at least two tested resizes.' : 'White marks the selected differences.'
  const reason = Object.prototype.hasOwnProperty.call(unassessed, comparison.status)
    ? unassessed[comparison.status] : 'The selected copies could not be compared reliably.'
  const pdf = comparisonPdfData(result)
  const download = () => {
    const json = JSON.stringify(imageReferenceExport(result), null, 2) + '\n'
    saveBlob(new Blob([json], { type: 'application/json' }), 'chipmunk-verify-image-comparison.json')
  }
  return <div className="mt-5 border-t border-border pt-4">
    <div className="flex flex-wrap items-baseline justify-between gap-3">
      <p className="m-0 text-small font-medium text-text" role="status">
        {comparable ? regions.length === 0 ? 'No localized differences found' : `${regions.length} difference ${regions.length === 1 ? 'region' : 'regions'} found`
          : 'Comparison unassessed'}
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {pdf && <Button variant="text" onClick={() => {
          const decoded = atob(pdf)
          const bytes = new Uint8Array(decoded.length)
          for (let index = 0; index < decoded.length; index++) bytes[index] = decoded.charCodeAt(index)
          saveBlob(new Blob([bytes], { type: 'application/pdf' }), 'chipmunk-verify-image-comparison.pdf')
        }}>Download comparison PDF</Button>}
        <Button variant="text" onClick={download}>Download comparison JSON</Button>
      </div>
    </div>
    {!pdf && <p className="mt-2 mb-0 text-caption text-text-muted">The comparison PDF is unavailable. The JSON download includes the measurements and images.</p>}
    {comparable
      ? regions.length === 0 && !completeMatch && <p className="mt-2 mb-0 text-small text-text-muted">
          No localized differences met the comparison criteria in the measured area. Small changes may still be present.
        </p>
      : <p className="mt-2 mb-0 text-small text-text-muted">{reason}</p>}
    {comparable && cropped && <p className="mt-3 mb-0 text-small text-text-muted">
      The comparison matched the submitted image to an area of the earlier copy at <NativeArea area={cropped} />.
      {' '}Differences were measured after resizing that area. Pixels outside the earlier area were unassessed.
    </p>}
    {delivery && <p className="mt-2 mb-0 text-caption text-text-muted">
      We compared color differences across plausible resize, tone and compression variations.
      Ordinary processing can still cause highlights. This comparison does not assess AI authorship.
    </p>}
    {sameJpegFraming && <p className="mt-2 mb-0 text-small text-text-muted">
      Exact JPEG block matches support the same framing. After encoding the earlier PNG with the submitted JPEG settings,
      {' '}{sameJpegFraming.exact_tiles?.toLocaleString('en-US')} of {sameJpegFraming.full_tiles?.toLocaleString('en-US')} complete blocks matched.
      {' '}{sameJpegFraming.anchor_count?.toLocaleString('en-US')} unique detailed blocks supported the alignment.
      Ordinary edits can produce highlighted differences.
    </p>}
    {sameJpegBlocks && <p className="mt-2 mb-0 text-small text-text-muted">
      {sameJpegBlocks.exact_tiles?.toLocaleString('en-US')} of {sameJpegBlocks.full_tiles?.toLocaleString('en-US')} complete decoded JPEG blocks match exactly.
      {' '}{sameJpegBlocks.anchor_count?.toLocaleString('en-US')} unique detailed blocks support the same framing.
      Ordinary edits can produce highlighted differences.
    </p>}
    {detailLoss && <p className="mt-2 mb-0 text-small text-text-muted">
      Fine luminance detail is weaker in the highlighted areas while broader structure remains similar. Local blur or other ordinary processing can produce this pattern. This check carries no weight in the AI estimate.
    </p>}
    {shiftedJpeg && <p className="mt-2 mb-0 text-small text-text-muted">
      After whole-pixel alignment, {shiftedJpeg.exact_tiles?.toLocaleString('en-US')} of {shiftedJpeg.full_tiles?.toLocaleString('en-US')} complete JPEG blocks matched in the area compared. Ordinary edits can produce highlighted differences.
    </p>}
    {sameStructureFraming && <p className="mt-2 mb-0 text-small text-text-muted">
      {sameStructureFraming.aligned_tiles.length} of {sameStructureFraming.detailed_tiles} detailed areas matched across the two images.
      {' '}After accounting for overall tone and JPEG compression, highlights require a local change in both color and structure.
      {' '}Ordinary edits can produce these differences. AI origin remains undetermined.
    </p>}
    {comparable && exact && <p className="mt-2 mb-0 text-small text-text-muted">
      {completeMatch ? 'The resized area matches every submitted pixel.'
        : `The resized area matches ${exact.matched_pixels.toLocaleString('en-US')} submitted pixels exactly.`}
      {' '}Matching pixels cover {exact.matched_tiles.length} textured tiles using {kernels} resizing.
    </p>}
    {comparable && orientation && <p className="mt-2 mb-0 text-small text-text-muted">
      {orientedResize ? `An earlier crop matched after ${orientationLabels[orientation.transform]} and ${kernels} resizing.`
        : orientedCrop ? `An earlier crop aligned using ${orientationLabels[orientation.transform]}.`
        : `The earlier copy aligned by ${orientationLabels[orientation.transform]}.`}
      {' '}{completeMatch ? 'Every submitted pixel matches after alignment.'
        : `${orientation.matched_pixels.toLocaleString('en-US')} submitted pixels match exactly after alignment.`}
      {' '}Each crop keeps its source orientation.
    </p>}
    {comparable && orientedCrop && <p className="mt-2 mb-0 text-small text-text-muted">
      Earlier copy overlap <NativeArea area={orientedCrop.native_crop} />.
      {' '}Pixels outside this earlier area were unassessed.
    </p>}
    {resized && !cropped && <p className="mt-3 mb-0 text-small text-text-muted">
      The comparison found matching framing at different pixel dimensions. Differences were measured on the submitted image after resizing the earlier copy. The crops show pixels from each original file.
    </p>}
    {embeddedPhoto && <>
      <p className="mt-3 mb-0 text-small text-text-muted">
        The earlier photo matched at native scale inside the screenshot.
        {' '}{completeMatch ? 'Every pixel inside the photo matches the earlier copy.'
          : `${embeddedPhoto.exact_pixels.toLocaleString('en-US')} pixels inside the photo match the earlier copy exactly.`}
      </p>
      <dl className="mt-3 mb-0 grid grid-cols-1 gap-2 text-small text-text-muted sm:grid-cols-2">
        <div><dt className="font-medium text-text">Screenshot photo area</dt><dd className="m-0"><NativeArea area={embeddedPhoto.submitted_bounds} /></dd></div>
        <div><dt className="font-medium text-text">Earlier copy area</dt><dd className="m-0"><NativeArea area={embeddedPhoto.reference_bounds} /></dd></div>
      </dl>
      <p className="mt-2 mb-0 text-caption text-text-muted">
        Pixels outside those areas were unassessed. Local edits or interface overlays can produce differences. AI origin remains unassessed.
      </p>
    </>}
    {(alignment || embeddedPhoto || (comparable && regions.length > 0)) && <p className="mt-3 mb-0 text-caption text-text-muted">
      Coordinates use each image’s own pixels, measured from the top left after correcting its stored orientation.
    </p>}
    {alignment && <dl className="mt-3 mb-0 grid grid-cols-1 gap-2 text-small text-text-muted sm:grid-cols-2">
      <div><dt className="font-medium text-text">{shiftedJpeg ? 'Submitted image area compared' : 'Submitted image overlap'}</dt><dd className="m-0"><NativeArea area={alignment.submitted_overlap} /></dd></div>
      <div><dt className="font-medium text-text">{shiftedJpeg ? 'Earlier copy area compared' : 'Earlier copy overlap'}</dt><dd className="m-0"><NativeArea area={alignment.reference_overlap} /></dd></div>
      {!sameJpegFraming && !sameJpegBlocks && <div className="sm:col-span-2"><dt className="sr-only">Comparison coverage</dt><dd className="m-0">{shiftedJpeg ? 'Margins outside these bounds were unassessed.' : 'Pixels outside the overlap were unassessed.'}</dd></div>}
    </dl>}
    {comparable && regions.length > 0 && <>
      {mask && <div className="mt-4 max-w-xl">
        <ComparisonPicture result={result} artifact={mask} label="Difference mask in submitted image coordinates" />
        <p className="mt-2 mb-0 text-caption text-text-muted">{maskDescription} Black includes unselected and potentially unassessed pixels.</p>
      </div>}
      <ol className="mt-4 mb-0 list-none space-y-4 p-0">
        {regions.map((region, index) => <li key={index} className="rounded-md border border-border p-3">
          <h3 className="m-0 text-small font-medium">{detailLoss ? 'Fine detail loss' : 'Difference region'} {index + 1}</h3>
          {detailLoss && <p className="mt-1 mb-0 text-caption text-text-muted">
            {region.directly_measured_tiles} of {(region.selected_pixels ?? 0) / 256} tiles directly support the finding. Adjacent highlights can include small gaps.
          </p>}
          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="min-w-0">
              <ComparisonPicture result={result} artifact={region.reference_crop} label={`Earlier copy crop for region ${index + 1}`} />
              <p className="mt-2 mb-0 text-caption text-text-muted"><NativeArea area={region.reference_region ?? region} /></p>
            </div>
            <div className="min-w-0">
              <ComparisonPicture result={result} artifact={region.submitted_crop} overlay={region.submitted_overlay}
                label={`Submitted image crop for region ${index + 1}`} />
              <p className="mt-2 mb-0 text-caption text-text-muted"><NativeArea area={region} /></p>
            </div>
          </div>
        </li>)}
      </ol>
      {!!comparison.additional_regions && <p className="mt-3 mb-0 text-small text-text-muted">
        {comparison.additional_regions} more {comparison.additional_regions === 1 ? 'region was' : 'regions were'} omitted from this bounded view.
      </p>}
    </>}
    <details className="mt-4 text-small text-text-muted">
      <summary className="cursor-pointer">Source file records</summary>
      <dl className="mt-2 mb-0 space-y-2 text-caption">
        <div><dt>Earlier copy · {reference.width} × {reference.height} px · SHA-256</dt><dd className="m-0 break-all font-mono">{reference.sha256}</dd></div>
        <div><dt>Submitted image · {submitted.width} × {submitted.height} px · SHA-256</dt><dd className="m-0 break-all font-mono">{submitted.sha256}</dd></div>
      </dl>
    </details>
  </div>
}

function ComparisonForm({ analysisId }: { analysisId: string }) {
  const headingId = useId()
  const fileId = useId()
  const formatId = useId()
  const [file, setFile] = useState<File | null>(null)
  const [state, setState] = useState<ComparisonState>({ status: 'idle' })
  const active = useRef<AbortController | null>(null)
  useEffect(() => () => active.current?.abort(), [])

  const chooseFile = (next: File | null) => {
    active.current?.abort()
    active.current = null
    setFile(next)
    setState({ status: 'idle' })
  }

  const compare = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!file || active.current) return
    const request = new AbortController()
    active.current = request
    setState({ status: 'pending' })
    try {
      const result = await compareWithEarlierImage(analysisId, file, request.signal)
      if (active.current === request && !request.signal.aborted) setState({ status: 'done', result })
    } catch (error) {
      if (active.current === request && !request.signal.aborted) {
        setState({ status: 'error', message: error instanceof Error ? error.message : 'Please try again.' })
      }
    } finally {
      if (active.current === request) active.current = null
    }
  }

  return <section id={IMAGE_REFERENCE_SECTION_ID} aria-labelledby={headingId} tabIndex={-1}
    className="scroll-mt-14 rounded-md border border-border p-4">
    <h2 id={headingId} className="m-0 text-body font-medium text-text">Compare with an earlier copy</h2>
    <p className="mt-2 mb-0 max-w-[75ch] text-small text-text-muted">
      Choose an earlier copy to look for pixel differences. Ordinary processing can produce differences, and establishing how a change was made or which copy came first requires separate evidence.
    </p>
    <form onSubmit={event => void compare(event)} className="mt-4 flex flex-wrap items-end gap-3">
      <div className="min-w-0">
        <label htmlFor={fileId} className="mb-1 block text-small font-medium text-text">Earlier copy</label>
        <input id={fileId} type="file" accept="image/png,image/jpeg,.png,.jpg,.jpeg" aria-describedby={formatId}
          onChange={event => chooseFile(event.currentTarget.files?.[0] ?? null)}
          className="block max-w-full text-small text-text file:mr-2 file:cursor-pointer file:rounded-sm file:border file:border-border file:bg-surface-2 file:px-2 file:py-1 file:text-text" />
        <p id={formatId} className="mt-1 mb-0 text-caption text-text-muted">PNG or JPEG, up to 4 megapixels.</p>
      </div>
      <Button type="submit" variant="primary" disabled={!file || state.status === 'pending'}>
        {state.status === 'pending' ? 'Comparing…' : state.status === 'error' ? 'Retry comparison' : 'Compare images'}
      </Button>
    </form>
    {state.status === 'pending' && <p role="status" className="mt-3 mb-0 text-small text-text-muted">Measuring pixel differences…</p>}
    {state.status === 'error' && <p role="alert" className="mt-3 mb-0 text-small text-danger">Comparison failed. {state.message}</p>}
    {state.status === 'done' && <ComparisonResult result={state.result} />}
  </section>
}

export function useImageReferenceAvailability(analysisId: string, enabled = true): boolean {
  const availability = useQuery({
    queryKey: ['analysis', analysisId, 'image-reference-comparison'],
    queryFn: ({ signal }) => comparisonAvailability(analysisId, signal),
    refetchOnMount: 'always',
    refetchOnWindowFocus: false,
    retry: false,
    enabled,
  })
  return enabled && availability.isFetchedAfterMount && availability.data === true && !availability.isError
}

export default function ImageReferenceComparison({ analysisId, available }: { analysisId: string; available?: boolean }) {
  const checked = useImageReferenceAvailability(analysisId, available === undefined)
  return (available ?? checked) ? <ComparisonForm key={analysisId} analysisId={analysisId} /> : null
}
