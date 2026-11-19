import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Button, ErrorState, SkeletonRows } from '../../components/ui'
import type { Box } from '../../lib/coords'
import { cx } from '../../lib/cx'
import { evidenceImageUrl, evidenceSourceUrl, measurementValue, useImageEvidence, type EvidenceImage, type ImageCopyTransform, type ImageEvidencePacket, type ImageEvidenceSet } from './imagePackets'
import ImageInspectionChecks, { CandidateEligibilityNote } from './ImageInspectionChecks'
import { evidenceSourceMediaUrl } from './multimediaPackets'

function EvidencePicture({ set, artifact, token, ready, alt }: {
  set: ImageEvidenceSet; artifact: EvidenceImage; token: string; ready: boolean; alt: string
}) {
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const src = ready ? evidenceImageUrl(set, artifact, token) : null
  return (
    <div className="min-h-44 aspect-[4/3] rounded-md border border-border bg-surface-2 flex items-center justify-center overflow-hidden">
      {!ready ? <p className="m-4 text-small text-text-muted">Preparing secure media links…</p>
        : !src || failed ? <div role="alert" className="p-4 text-small text-text-muted">
          <p className="m-0">This evidence image could not be loaded. Its recorded position and explanation remain below.</p>
          {src && <Button variant="text" className="mt-2" onClick={() => { setFailed(false); setAttempt((x) => x + 1) }}>Retry image</Button>}
        </div>
          : <img key={`${artifact.sha256}:${attempt}`} src={src} alt={alt} width={artifact.width} height={artifact.height}
            loading="lazy" className="block w-full h-full object-contain" onError={() => setFailed(true)} />}
    </div>
  )
}

function RegionalComparison({ set, token, mediaReady }: { set: ImageEvidenceSet; token: string; mediaReady: boolean }) {
  const [showMasks, setShowMasks] = useState(false)
  const comparison = set.comparison
  if (!comparison) return null
  const wider = showMasks ? comparison.artifacts.wider_mask : comparison.artifacts.wider_overlay
  const shared = showMasks ? comparison.artifacts.shared_mask : comparison.artifacts.shared_overlay
  const statuses = { measured: 'Measured', withheld: 'Candidate regions withheld', missing: 'Not available',
    failed: 'Check failed', skipped: 'Check skipped', unavailable: 'Evidence unavailable' }
  return <section data-testid="image-regional-comparison" className="mt-5 rounded-md border border-border p-4" aria-labelledby="regional-comparison-heading">
    <div className="flex items-baseline justify-between gap-3 flex-wrap">
      <h3 id="regional-comparison-heading" className="m-0 text-body font-medium text-text">Compare local editing cues</h3>
      {comparison.status !== 'unavailable' && <button type="button" aria-pressed={showMasks}
        onClick={() => setShowMasks(v => !v)} className="min-h-8 px-2.5 rounded-sm border border-border text-small text-text-muted hover:text-text">
        {showMasks ? 'Show source overlays' : 'Show comparison masks'}
      </button>}
    </div>
    <p className="mt-2 mb-0 text-small text-text-muted">{comparison.interpretation}</p>
    {comparison.status === 'unavailable' ? <p className="mt-3 mb-0 text-small text-text" role="status">
      A shared regional view is unavailable for this result. This does not establish that the image is unaltered.
    </p> : <>
      {comparison.status !== 'ready' && <p className="mt-3 mb-0 text-small text-text">Some eligible regions are not shown. This comparison covers the retained evidence cards only.</p>}
      <div className="mt-4 grid grid-cols-2 gap-4 max-[639px]:grid-cols-1">
        {([['wider', 'Wider view: either check', wider], ['shared', 'Shared view: both checks', shared]] as const).map(([id, label, asset]) => asset &&
          <figure key={id} className="m-0 min-w-0">
            <figcaption className="mb-2 text-small font-medium text-text">{label}</figcaption>
            <EvidencePicture key={`${id}:${asset.sha256}:${token}`} set={set} artifact={asset} token={token} ready={mediaReady}
              alt={`${label}; ${showMasks ? 'white selected cells on the measured grid' : 'flagged cells over the submitted source image'}`} />
          </figure>)}
      </div>
      <p className="mt-2 mb-0 text-caption text-text-muted">{showMasks
        ? 'White marks retained cells; black means unselected, not verified original. A single shared cell is preserved.'
        : 'Orange marks cells selected by either check; purple marks cells selected by both. These overlays locate inspection cues, not proven AI edits.'}</p>
      {comparison.counts && <p className="mt-2 mb-0 text-small text-text-muted">
        Wider view: {measurementValue(comparison.counts.wider)} measured cells. Shared view: {measurementValue(comparison.counts.shared)} measured cells.
        {comparison.counts.shared === 0 && ' No shared cells were selected; disagreement does not clear the file.'}
      </p>}
    </>}
    <ul className="mt-3 mb-0 pl-4 space-y-1 text-small text-text-muted list-disc">
      {comparison.members.map(member => <li key={member.id}>{member.label}: {statuses[member.status]}.
        {member.packet_ids.length > 0 && <> Retained regions: {member.packet_ids.map((id, i) => <span key={id}>{i > 0 && ', '}
          <Link to={`#image-evidence-${encodeURIComponent(id)}`} className="underline underline-offset-[3px] hover:text-text">{id}</Link></span>)}.</>}
      </li>)}
      {comparison.limitations.map((line, i) => <li key={i}>{line}</li>)}
    </ul>
    {comparison.map_grid && <p className="mt-2 mb-0 text-caption text-text-muted">Measured grid {comparison.map_grid.join(' × ')}. Larger previews do not add pixel detail.</p>}
  </section>
}

const copyBasisText = {
  native_patch_alignment: 'The match aligns patches on the original pixel grid.',
  geometric_features: 'The match uses corresponding features in both areas.',
  jpeg_brightness_blocks: 'The complete encoded brightness blocks matched exactly in both areas.',
  jpeg_offgrid_texture: 'The aligned crops have closely matching color values inside and more different surroundings on at least three sides.',
}

const jpegCoverageText = {
  supported_pair: copyBasisText.jpeg_brightness_blocks,
  no_pair: 'The JPEG brightness-block check found no qualifying repeated area.',
  unsupported_profile: 'The brightness-block check requires an upright JPEG within its tested size and encoding scope.',
  unavailable: 'The JPEG brightness-block check could not read the encoded blocks.',
  unassessed_match_budget: 'The brightness-block check found too many competing offsets.',
}

const offgridCoverageText = {
  supported_pair: copyBasisText.jpeg_offgrid_texture,
  no_pair: 'The between-block JPEG check found no qualifying repeated area. The minimum crop side is 32 or 56 pixels, depending on the JPEG compression profile.',
  unsupported_profile: 'The between-block check covers upright color JPEGs with the tested standard compression tables.',
  unavailable: 'The between-block JPEG check could not complete.',
  unassessed_match_budget: 'Too many competing offsets prevented a distinct between-block comparison.',
}

const turnedJpegCoverageText = {
  supported_pair: 'After alignment, the crops have similar color inside and more different surroundings on at least three sides.',
  no_pair: 'The JPEG rotation check found no qualifying pair after a quarter turn or half turn.',
  unsupported_profile: 'The JPEG rotation check covers upright color JPEGs within the tested compression settings.',
  unavailable: 'The JPEG rotation check could not complete.',
  ambiguous: 'More than one pair fit the JPEG rotation check, so it could not select distinct areas.',
  unassessed_match_budget: 'Too many matching points prevented a distinct JPEG rotation comparison.',
}

const smallPngCoverageText = {
  supported_pair: 'Every RGB pixel in the two crops matches exactly. Pixels outside the crops differ on at least three sides.',
  no_pair: 'The small-copy check found no qualifying exact pair.',
  unsupported_profile: 'The small-copy check covers upright, static 8-bit RGB PNGs with sides from 384 to 768 pixels.',
  unavailable: 'The small-copy check could not complete.',
  ambiguous_pairs: 'Multiple exact pairs made this comparison ambiguous.',
  unassessed_match_budget: 'Too many competing offsets prevented a distinct small-copy comparison.',
}

const webpExactCoverageText = {
  ...smallPngCoverageText,
  no_pair: 'The lossless WebP check found no qualifying exact pair with or without right-angle rotation.',
  unsupported_profile: 'Exact comparison requires an opaque lossless WebP still with sides from 512 to 768 pixels. The image must be upright.',
}
const webpReflectionCoverageText = {
  ...webpExactCoverageText,
  no_pair: 'The reflection check found no qualifying exact pair after a horizontal or vertical flip.',
  unassessed_match_budget: 'Too many matching areas prevented the reflection check from isolating a pair.',
  ambiguous_pairs: 'Several reflected pairs met the evidence requirements, so no single pair is shown.',
  unavailable: 'The exact reflection check could not complete.',
}

const copyAreaLabels = { area_a: 'Area A', area_b: 'Area B' } as const
const copyAreas = ['area_a', 'area_b'] as const

function CopyLocationMap({ analysisId, source, regions, token, mediaReady }: {
  analysisId: string; source: ImageEvidenceSet['source'] & { width: number; height: number }
  regions: Record<(typeof copyAreas)[number], Box>; token: string; mediaReady: boolean
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [status, setStatus] = useState<'loading' | 'ready' | 'unavailable'>('loading')
  const { width, height, sha256 } = source
  const scale = Math.min(1, 760 / width, 520 / height)
  useEffect(() => {
    if (!mediaReady) return
    const controller = new AbortController()
    const load = async () => {
      setStatus('loading')
      const response = await fetch(evidenceSourceMediaUrl(analysisId, token), { signal: controller.signal })
      if (!response.ok) throw new Error('Original image unavailable')
      const blob = await response.blob()
      const digest = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())
      const actual = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('')
      if (actual !== sha256 || controller.signal.aborted) throw new Error('Original image hash differs')
      const image = await createImageBitmap(blob, { imageOrientation: 'from-image' })
      try {
        if (controller.signal.aborted) return
        const context = canvas.current?.getContext('2d')
        if (!context || image.width !== width || image.height !== height) throw new Error('Original image geometry differs')
        context.drawImage(image, 0, 0, context.canvas.width, context.canvas.height)
        setStatus('ready')
      } finally {
        image.close()
      }
    }
    void load().catch(() => { if (!controller.signal.aborted) setStatus('unavailable') })
    return () => controller.abort()
  }, [analysisId, height, mediaReady, sha256, token, width])

  return <div className="mt-2">
    {!mediaReady || status === 'loading' ? <p role="status" className="m-0 text-small text-text-muted">Loading the original image…</p>
      : status === 'unavailable' && <p role="status" className="m-0 text-small text-text-muted">The source map is unavailable. Both crops and their source coordinates are shown below.</p>}
    <div className="relative max-w-full" style={{ width: width * scale }} hidden={!mediaReady || status !== 'ready'}>
      <canvas ref={canvas} width={Math.max(1, Math.round(width * scale))} height={Math.max(1, Math.round(height * scale))}
        className="block w-full rounded-md" style={{ aspectRatio: `${width} / ${height}` }} role="img"
        aria-label="Original image with the two comparison areas marked" />
      {copyAreas.map((area, index) => {
        const { x, y, w, h } = regions[area]
        const label = copyAreaLabels[area]
        return <Link key={area} to={evidenceSourceUrl(analysisId, { region: regions[area] })}
          aria-label={`Inspect ${label} in the original image`}
          className={cx('absolute border-2 shadow-[0_0_0_1px_#111] focus-visible:outline-2 focus-visible:outline-white',
            index === 0 ? 'border-amber-400' : 'border-cyan-300')}
          style={{ left: `${x / width * 100}%`, top: `${y / height * 100}%`, width: `${w / width * 100}%`, height: `${h / height * 100}%` }}>
          <span aria-hidden="true" className={cx('absolute top-0 left-0 rounded-br px-1 text-caption leading-none text-black',
            index === 0 ? 'bg-amber-400' : 'bg-cyan-300')}>{label.at(-1)}</span>
        </Link>
      })}
    </div>
    {status === 'ready' && mediaReady && <p className="mt-2 mb-0 text-caption text-text-muted">The outlines locate Areas A and B. Select either to inspect the native pixels.</p>}
  </div>
}

function CopyLocations({ set, sourceHash, regions, token, mediaReady }: {
  set: ImageEvidenceSet; sourceHash: string; regions: Record<(typeof copyAreas)[number], Box>; token: string; mediaReady: boolean
}) {
  const [open, setOpen] = useState(false)
  const { width, height, coordinate_space, sha256 } = set.source
  if (width == null || height == null || ![width, height].every(value => Number.isSafeInteger(value) && value > 0)
    || coordinate_space !== 'exif-oriented-original' || sha256 !== sourceHash || !/^[a-f0-9]{64}$/.test(sha256)
    || !copyAreas.every(area => {
      const { x, y, w, h } = regions[area]
      return [x, y, w, h].every(Number.isSafeInteger) && x >= 0 && y >= 0 && w > 0 && h > 0 && x + w <= width && y + h <= height
    })) return null
  return <details className="mt-3" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary className="cursor-pointer text-small text-text-muted">Show both areas on the original image</summary>
    {open && <CopyLocationMap analysisId={set.analysis_id} source={{ ...set.source, width, height }} regions={regions}
      token={token} mediaReady={mediaReady} />}
  </details>
}

function CopySimilarity({ set, field, token, mediaReady }: {
  set: ImageEvidenceSet; field: 'copy_similarity' | 'copy_translation' | 'copy_affine'; token: string; mediaReady: boolean
}) {
  const check = set[field]
  if (!check || check.status === 'skipped_trusted_ai_credentials') return null
  const reflected = field === 'copy_similarity'
  const geometric = field === 'copy_affine'
  const sectionId = `image-${field.replace('_', '-')}`
  const baseVersion = set.prior_embedded_pair_evidence?.version ?? set.version
  const diagonalVersion = baseVersion === 'image-evidence-46' || baseVersion === 'image-evidence-47'
  const diagonalScope = diagonalVersion && [set.source.width, set.source.height].every(side => side != null && side >= 512 && side <= 672)
  const webpReflectionVersion = ['image-evidence-34', 'image-evidence-35', 'image-evidence-40', 'image-evidence-41', 'image-evidence-44', 'image-evidence-45', 'image-evidence-47'].includes(baseVersion)
  const pngReflectionVersion = ['image-evidence-36', 'image-evidence-37', 'image-evidence-38', 'image-evidence-39', 'image-evidence-42', 'image-evidence-43', 'image-evidence-46'].includes(baseVersion)
  const threePairScope = (diagonalVersion || ['image-evidence-42', 'image-evidence-43', 'image-evidence-44', 'image-evidence-45'].includes(baseVersion))
    && set.source.width === 512 && set.source.height === 512
  const diagonalJpeg = field === 'copy_translation' ? set.copy_translation?.jpeg_diagonal_review : undefined
  const heading = geometric ? 'Geometric copy comparison' : reflected ? 'Same-image copy similarity'
    : diagonalJpeg ? 'JPEG diagonal copy comparison'
    : webpReflectionVersion || pngReflectionVersion ? 'Exact and geometric copy comparisons' : 'Copy similarity without mirroring'
  const turnedReview = pngReflectionVersion || ['image-evidence-22', 'image-evidence-23'].includes(baseVersion)
  const diagonalJpegVersion = baseVersion === 'image-evidence-48' || baseVersion === 'image-evidence-49'
  const nativeJpegVersion = diagonalJpegVersion || ['image-evidence-28', 'image-evidence-29', 'image-evidence-30', 'image-evidence-31'].includes(baseVersion)
  const webpVersion = baseVersion === 'image-evidence-32' || baseVersion === 'image-evidence-33' || webpReflectionVersion
  const wide = (geometric && (diagonalVersion || diagonalJpegVersion || ['image-evidence-14', 'image-evidence-17', 'image-evidence-19', 'image-evidence-21', 'image-evidence-23', 'image-evidence-25', 'image-evidence-27', 'image-evidence-29', 'image-evidence-31', 'image-evidence-33', 'image-evidence-35', 'image-evidence-37', 'image-evidence-39', 'image-evidence-41', 'image-evidence-43', 'image-evidence-45'].includes(baseVersion)))
    || (reflected && (turnedReview || ['image-evidence-9', 'image-evidence-10', 'image-evidence-11', 'image-evidence-12',
      'image-evidence-13', 'image-evidence-14', 'image-evidence-16', 'image-evidence-17', 'image-evidence-18',
      'image-evidence-19', 'image-evidence-20', 'image-evidence-21', 'image-evidence-24', 'image-evidence-25',
      'image-evidence-26', 'image-evidence-27'].includes(baseVersion) || nativeJpegVersion || webpVersion))
  const basis = field === 'copy_translation' && set.copy_translation?.basis
  const jpegReview = field === 'copy_translation' && set.copy_translation?.jpeg_block_review
  const offgrid = field === 'copy_translation' ? set.copy_translation?.jpeg_offgrid_review : undefined
  const turnedJpeg = field === 'copy_translation' ? set.copy_translation?.jpeg_turned_review : undefined
  const nativeJpeg = Boolean(nativeJpegVersion && turnedJpeg && !offgrid)
  const separateMeasurements = turnedJpeg?.measurements
  const multi = field === 'copy_translation' ? set.copy_translation?.multi_exact_review : undefined
  const multiStatus = {
    supported_pair: multi?.measurements.length === 3
      ? 'Three separate pairs of textured areas match pixel for pixel after alignment.'
      : 'Two separate pairs of textured areas match pixel for pixel after alignment.',
    no_second_pair: 'The exact comparison did not find two distinct pairs.',
    ambiguous_pairs: threePairScope
      ? 'The matches did not meet the supported combination or spacing for a distinct exact comparison.'
      : 'Competing or nearby exact matches prevented a distinct two-pair comparison.',
    unsupported_profile: 'The source was outside the supported upright, opaque lossless image profile.',
    unassessed_match_budget: threePairScope ? 'Too many matching areas prevented a distinct exact comparison.'
      : 'Too many matching areas prevented a distinct two-pair comparison.',
    unavailable: threePairScope ? 'The exact copy-pair comparison could not complete.' : 'The exact two-pair comparison could not complete.',
  }
  const multiAlignments: Partial<Record<ImageCopyTransform, [string, string]>> = {
    same_orientation: ['Exact copy pair', 'The two crops match in their original orientation.'],
    horizontal_reflection: ['Exact reflected pair', 'Flip Area B horizontally to align with Area A.'],
    vertical_reflection: ['Exact reflected pair', 'Flip Area B vertically to align with Area A.'],
    rotate_90_ccw: ['Exact rotated pair', 'Rotate Area B 90° counterclockwise to align with Area A.'],
    rotate_180: ['Exact rotated pair', 'Rotate Area B 180° to align with Area A.'],
  }
  const smallPng = field === 'copy_translation' ? set.copy_translation?.small_png_copy_review : undefined
  const pngReflection = field === 'copy_translation' ? set.copy_translation?.png_exact_reflection_review : undefined
  const webpReflection = field === 'copy_translation' ? set.copy_translation?.webp_exact_reflection_review : undefined
  const reflection = pngReflection ?? webpReflection
  const exactReflection = basis === 'webp_exact_reflected_copy' || basis === 'png_exact_reflected_copy'
  const exactReview = reflection ?? (field === 'copy_translation' ? set.copy_translation?.webp_exact_copy_review : undefined)
  const transform = set.copy_translation?.pairs[0]?.transform
  const diagonalAxis = transform === 'top_left_to_bottom_right' ? 'top-left to bottom-right'
    : transform === 'top_right_to_bottom_left' ? 'top-right to bottom-left' : undefined
  const diagonal = exactReflection && diagonalScope && diagonalAxis
  const approximateDiagonal = diagonalJpeg?.status === 'supported_pair' && diagonalAxis
  const diagonalJpegStatus = {
    ...turnedJpegCoverageText,
    supported_pair: 'The aligned interiors have closely matching colors, with greater differences on at least three surrounding sides.',
    no_pair: 'The diagonal JPEG check found no qualifying pair.',
    same_axis: 'The match could be part of a larger mirror pattern, so this check withheld the local pair.',
    ambiguous: 'Several pairs fit the diagonal check, so this check withheld a distinct comparison.',
    unavailable: 'The diagonal JPEG check could not complete.',
    unassessed_match_budget: 'Too many matching areas prevented a distinct diagonal comparison.',
    unsupported_profile: baseVersion === 'image-evidence-49'
      ? 'The diagonal check covers upright color images with the tested JPEG 75 compression tables, or single-image baseline JPEGs with the tested JPEG 90 compression tables.'
      : 'The diagonal check covers upright color images with the tested JPEG 75 compression tables.',
  }
  const turned = (basis === 'png_exact_turned_copy' || basis === 'jpeg_turned_copy' || (basis === 'webp_exact_copy' && transform !== 'same_orientation')) && transform
  const rotation = turned === 'rotate_90_ccw' ? '90° counterclockwise' : '180°'
  let exactObservation: string | undefined
  let exactScope: string | undefined
  if (exactReview?.status === 'supported_pair') {
    exactObservation = diagonal
      ? `Reflecting Area B across the ${diagonal} diagonal aligns every RGB pixel with Area A in the ${pngReflection ? 'PNG' : 'lossless WebP'}.`
      : exactReflection
      ? `Flipping Area B ${transform === 'horizontal_reflection' ? 'horizontally' : 'vertically'} aligns every RGB pixel with Area A in the ${pngReflection ? 'PNG' : 'lossless WebP'}.`
      : turned
      ? `Rotating Area B ${rotation} aligns every RGB pixel with Area A in the lossless WebP.`
      : 'The two lossless WebP crops match exactly in their original orientation.'
  } else if (exactReview) {
    exactObservation = pngReflection && exactReview.status === 'unsupported_profile'
      ? smallPngCoverageText.unsupported_profile
      : (reflection ? webpReflectionCoverageText : webpExactCoverageText)[exactReview.status]
    if (exactReview.status === 'no_pair') {
      if (reflection && diagonalScope) exactObservation = 'The reflection check found no qualifying exact pair after a horizontal, vertical, or diagonal flip.'
      exactScope = 'This check covers separated, textured crops with sides from 16 to 56 pixels.'
    }
  } else if (!multi && field === 'copy_translation' && webpVersion) {
    const outsideExactSize = [set.source.width, set.source.height].some(side => side == null || side < 512 || side > 768)
    if (outsideExactSize && check.status === 'unsupported_geometry') {
      exactObservation = webpExactCoverageText.unsupported_profile
    } else {
      exactScope = outsideExactSize ? webpExactCoverageText.unsupported_profile : 'Exact lossless WebP comparison did not run for this result.'
    }
  } else if (!multi && field === 'copy_translation' && pngReflectionVersion) {
    const outsideExactSize = [set.source.width, set.source.height].some(side => side == null || side < 384 || side > 768)
    if (outsideExactSize && check.status === 'unsupported_geometry') {
      exactObservation = smallPngCoverageText.unsupported_profile
    } else {
      exactScope = outsideExactSize ? smallPngCoverageText.unsupported_profile : 'The exact PNG reflection check did not run for this result.'
    }
  }
  const exactMeasurement = exactReview?.measurement ?? smallPng?.measurement
  const interpretation = field === 'copy_translation' && (webpVersion || pngReflectionVersion) && !check.pairs.length
    ? 'Copy comparisons look for repeated image detail. The origin of the image and AI involvement remain undetermined.'
    : check.interpretation
  const colorMeasurement = diagonalJpeg?.measurement ?? turnedJpeg?.measurement ?? offgrid?.measurement
  const basisText = exactReview?.status === 'supported_pair' ? 'Pixels outside the aligned crops differ on at least three sides.' : smallPng && !reflection
    ? turned ? 'After alignment, pixels outside the crops differ on at least three sides.'
      : turnedReview && smallPng.status === 'no_pair' ? 'The small-copy check found no qualifying exact pair with or without right-angle rotation.'
        : smallPngCoverageText[smallPng.status]
    : diagonalJpeg ? diagonalJpegStatus[diagonalJpeg.status]
      : turnedJpeg ? turnedJpegCoverageText[turnedJpeg.status]
      : offgrid ? offgridCoverageText[offgrid.status] : jpegReview ? jpegCoverageText[jpegReview]
        : basis && basis in copyBasisText && check.pairs.length > 0 && copyBasisText[basis as keyof typeof copyBasisText]
  const proposalSearch = geometric ? set.copy_affine?.proposal_search : undefined
  const statusText = {
    supported_pair: approximateDiagonal ? `Area B closely matches Area A after reflection across its ${approximateDiagonal} diagonal.`
      : separateMeasurements ? 'Two separate pairs of rotated areas share closely matching detail.'
      : turned ? `Area B ${basis === 'jpeg_turned_copy' ? 'closely matches Area A' : 'matches Area A exactly'} after rotating it ${rotation}.`
      : geometric ? 'Matching detail was found after geometric alignment.'
      : reflected ? 'Similar areas were found after horizontal reflection.' : 'Similar image detail was found without mirroring.',
    no_pair: geometric ? 'None of the distinct candidates examined had enough shared detail after alignment.'
      : reflected ? 'This check found no qualifying reflected pair.' : 'This check found no qualifying non-mirrored pair.',
    unsupported_geometry: wide
      ? 'This check covers images with a long side of 513–1,024 px and a short side of at least 384 px, or a long side of 1,025–2,048 px and a short side of at least 768 px.'
      : 'This check requires native dimensions of 384 px or more on both sides and a long side of 513–1,024 px.',
    unassessed_match_budget: 'The check found too many candidate matches to show a distinct comparison.',
  }
  return <section className="mt-5 rounded-md border border-border p-4" aria-labelledby={`${sectionId}-heading`}>
    <h3 id={`${sectionId}-heading`} className="m-0 text-body font-medium text-text">{heading}</h3>
    <p className="mt-2 mb-0 text-small text-text">{multi?.status === 'supported_pair' ? multiStatus.supported_pair
      : exactObservation ?? ((diagonalJpeg || nativeJpeg) && check.status === 'no_pair' ? basisText : statusText[check.status])}</p>
    <p className="mt-2 mb-0 text-small text-text-muted">{interpretation}</p>
    {!reflected && <p className="mt-2 mb-0 text-caption text-text-muted">This check carries zero weight in AI authorship confidence.</p>}
    {basisText && (!(nativeJpeg || diagonalJpeg) || check.status !== 'no_pair') && <p className="mt-2 mb-0 text-caption text-text-muted">{basisText}</p>}
    {exactScope && <p className="mt-2 mb-0 text-caption text-text-muted">{exactScope}</p>}
    {multi && multi.status !== 'supported_pair' && <p className="mt-2 mb-0 text-caption text-text-muted">{multiStatus[multi.status]}</p>}
    {nativeJpeg && check.status === 'supported_pair' && <p className="mt-2 mb-0 text-caption text-text-muted">
      The rotation check used the original pixel grid at this image size.
    </p>}
    {turnedJpeg && offgrid?.status === 'unsupported_profile' && <p className="mt-2 mb-0 text-caption text-text-muted">
      The between-block translation check was outside this file’s compression profile.
    </p>}
    {exactMeasurement && <p className="mt-2 mb-0 text-caption text-text-muted">
      Compared {exactMeasurement.matching_pixels.toLocaleString('en-US')} RGB pixels in each crop.
      Mean surrounding differences (top, bottom, left, right) {exactMeasurement.perimeter_mean_absolute_difference.map(value => value.toFixed(2)).join(', ')} of 255 levels.
    </p>}
    {colorMeasurement && <p className="mt-2 mb-0 text-caption text-text-muted">
      {diagonalJpeg ? `Mean RGB difference in the smoothed interior was ${colorMeasurement.interior_mean.toFixed(2)} of 255 levels, excluding a six-pixel border.`
        : `Mean RGB difference inside ${colorMeasurement.interior_mean.toFixed(2)} of 255 levels.`}
      Surrounding differences (left, right, top, bottom) {colorMeasurement.perimeter_means.map(value => value?.toFixed(2) ?? 'unavailable').join(', ')}.
    </p>}
    {proposalSearch?.basis === 'reduced' && <p className="mt-2 mb-0 text-caption text-text-muted">The candidate search used a {proposalSearch.width} × {proposalSearch.height} pixel working image. The check measured any matching masks on the original pixel grid.</p>}
    {check.pairs.length > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">Coordinates use native pixels in the oriented original.</p>}
    {check.pairs.map((pair, pairIndex) => {
      const exactPair = multi?.measurements[pairIndex]
      const alignment = exactPair && 'transform' in pair ? multiAlignments[pair.transform] : undefined
      return <article key={pair.id} className="mt-4 border-t border-border pt-4">
        <h4 className="m-0 text-small font-medium text-text">{pair.id} · {alignment?.[0] ?? (diagonal ? 'Exact diagonally reflected pair' : exactReflection ? 'Exact reflected pair' : approximateDiagonal ? 'Diagonally aligned JPEG pair' : basis === 'jpeg_turned_copy' ? 'Rotated JPEG pair' : turned ? 'Exact rotated pair' : basis === 'webp_exact_copy' ? 'Exact copy pair' : geometric ? 'Matched source areas' : reflected ? 'Reflected pair' : 'Non-mirrored pair')}</h4>
        {exactPair && alignment && <p className="mt-2 mb-0 text-caption text-text-muted">
          {alignment[1]} Compared {exactPair.matching_pixels.toLocaleString('en-US')} RGB pixels in each crop.{' '}
          Mean surrounding differences (top, bottom, left, right) {exactPair.perimeter_mean_absolute_difference.map(value => value.toFixed(2)).join(', ')} of 255 levels.
        </p>}
        {separateMeasurements && 'transform' in pair && <p className="mt-2 mb-0 text-caption text-text-muted">
          Rotate Area B {pair.transform === 'rotate_90_ccw' ? '90° counterclockwise' : '180°'} to align with Area A.{' '}
          Mean RGB difference inside {separateMeasurements[pairIndex].interior_mean.toFixed(2)} of 255 levels.{' '}
          Surrounding differences (left, right, top, bottom) {separateMeasurements[pairIndex].perimeter_means.map(value => value?.toFixed(2) ?? 'unavailable').join(', ')}.
        </p>}
        <CopyLocations set={set} sourceHash={check.source_sha256} regions={pair.regions} token={token} mediaReady={mediaReady} />
        <div className="mt-3 grid grid-cols-2 gap-4 max-[639px]:grid-cols-1">
          {copyAreas.map((role, index) => {
            const region = pair.regions[role]
            const label = copyAreaLabels[role]
            const coordinatesId = `${sectionId}-${pair.id}-${role}-coordinates`
            const mask = 'selected_pixels' in pair ? pair.artifacts[index === 0 ? 'mask_a' : 'mask_b'] : undefined
            return <figure key={role} className="m-0 min-w-0">
              <EvidencePicture key={`${pair.artifacts[role].sha256}:${token}`} set={set} artifact={pair.artifacts[role]} token={token} ready={mediaReady}
                alt={`${pair.id}: ${label} cropped from the submitted image`} />
              <figcaption className="mt-2 text-small text-text-muted">
                <Link to={evidenceSourceUrl(set.analysis_id, { region })} aria-describedby={coordinatesId}
                  className="underline underline-offset-[3px]">{label} in the original image</Link>
                <span id={coordinatesId} className="block font-mono text-caption">x {region.x}, y {region.y}, {region.w} × {region.h} px</span>
              </figcaption>
              {mask && 'selected_pixels' in pair && <div className="mt-3">
                <EvidencePicture set={set} artifact={mask} token={token} ready={mediaReady} alt={`${pair.id}: ${label} ${index === 0 ? 'selected' : 'mapped'} mask`} />
                <p className="mt-2 mb-0 text-caption text-text-muted">{index === 0
                  ? `White marks ${pair.selected_pixels[role]} pixels selected from neighborhoods that matched after alignment.`
                  : `White shows the same selection mapped to ${pair.selected_pixels[role]} pixels in Area B.`}</p>
              </div>}
            </figure>
          })}
        </div>
      </article>
    })}
  </section>
}

const pairDifferenceObservations = {
  local_difference: 'A localized pixel difference was measured between the two displayed areas. The image may contain other matching areas, including resized copies.',
  matching_without_difference: 'The pixels matched throughout the two compared areas.',
  no_supported_pair: 'The check found no qualifying two-area comparison.',
  unassessed_ambiguous_match: 'Additional matching detail made the two-area comparison ambiguous.',
  outside_scope: 'The exact two-area comparison covers selected native sizes of upright RGB PNGs.',
  unassessed_runtime: 'The exact two-area comparison could not be completed for this file.',
}

function EmbeddedPairDifference({ set, token, mediaReady }: { set: ImageEvidenceSet; token: string; mediaReady: boolean }) {
  const [showCrops, setShowCrops] = useState(false)
  const check = set.embedded_pair_difference
  if (!check || check.status === 'skipped_trusted_ai_credentials') return null
  const { pair } = check
  return <section className="mt-5 rounded-md border border-border p-4" data-testid="image-embedded-pair-difference"
    aria-labelledby="image-embedded-pair-heading">
    <div className="flex flex-wrap items-baseline justify-between gap-3">
      <h3 id="image-embedded-pair-heading" className="m-0 text-body font-medium text-text">Pixel differences between matching areas</h3>
      {pair && <button type="button" aria-pressed={showCrops} onClick={() => setShowCrops(value => !value)}
        className="min-h-8 rounded-sm border border-border px-2.5 text-small text-text-muted hover:text-text">
        {showCrops ? 'Show highlighted differences' : 'Show original crops'}
      </button>}
    </div>
    <p className="mt-2 mb-0 text-small text-text">{pairDifferenceObservations[check.status]}</p>
    {pair && <>
      <p className="mt-2 mb-0 text-small text-text-muted">{check.interpretation}</p>
      <p className="mt-2 mb-0 text-small text-text-muted">
        {measurementValue(pair.different_pixels)} differing pixels and {measurementValue(pair.matching_pixels)} matching pixels in the aligned areas.
      </p>
      <div className="mt-4 grid grid-cols-2 gap-4 max-[639px]:grid-cols-1">
        {([['area_a', 'overlay_a', 'Area A'], ['area_b', 'overlay_b', 'Area B']] as const).map(([role, overlay, label]) => {
          const region = pair.difference_regions[role]
          const artifact = pair.artifacts[showCrops ? role : overlay]
          return <figure key={role} className="m-0 min-w-0">
            <EvidencePicture key={`${artifact.sha256}:${token}`} set={set} artifact={artifact} token={token} ready={mediaReady}
              alt={`${label} ${showCrops ? 'original crop' : 'with differing pixels highlighted in orange'}`} />
            <figcaption className="mt-2 text-small text-text-muted">
              <Link to={evidenceSourceUrl(set.analysis_id, { region })} className="underline underline-offset-[3px]">
                Inspect differences in {label}
              </Link>
              <span className="block font-mono text-caption">x {region.x}, y {region.y}, {region.w} × {region.h} px</span>
            </figcaption>
          </figure>
        })}
      </div>
      <p aria-live="polite" className="mt-2 mb-0 text-caption text-text-muted">
        {showCrops ? 'Both crops show pixels from the submitted image.' : 'Orange highlights differing pixels in each source crop.'}
      </p>
    </>}
  </section>
}

function EvidenceCard({ packet, set, token, mediaReady }: {
  packet: ImageEvidencePacket; set: ImageEvidenceSet; token: string; mediaReady: boolean
}) {
  const [overlay, setOverlay] = useState(false)
  const [mask, setMask] = useState(false)
  const { x, y, w, h } = packet.region
  const donor = packet.artifacts.copy_donor
  const coordinates = `x ${measurementValue(x)}, y ${measurementValue(y)}, ${measurementValue(w)} × ${measurementValue(h)} px`
  return (
    <article id={`image-evidence-${packet.id}`} tabIndex={-1} data-testid="image-evidence-card" data-evidence-id={packet.id}
      className="border-t border-border pt-6 scroll-mt-20 focus-visible:outline-2 focus-visible:outline-accent">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="font-mono text-small text-accent">{packet.id}</span>
        <h3 className="m-0 text-body font-medium text-text">{packet.title}</h3>
        <span className="text-caption text-text-muted">{packet.evidence_type === 'localisation' ? 'Localisation candidate' : 'Inspection region'}</span>
      </div>
      <p className="mt-2 mb-0 text-body text-text max-w-[80ch]">{packet.observation}</p>
      <div className={cx('mt-4 grid grid-cols-2 gap-4 max-[639px]:grid-cols-1', donor && 'min-[960px]:grid-cols-3')}>
        <figure className="m-0 min-w-0">
          <div className="mb-2 min-h-8 flex items-center text-small font-medium text-text">Position in the source</div>
          <EvidencePicture key={`context:${token}`} set={set} artifact={packet.artifacts.context} token={token} ready={mediaReady}
            alt={`${packet.id}: full source image with the inspection region outlined`} />
          <figcaption className="mt-2 text-caption text-text-muted">Outline marks this region. The surrounding image is retained for context.</figcaption>
        </figure>
        <figure className="m-0 min-w-0">
          <div className="mb-2 min-h-8 flex items-center justify-between gap-2 flex-wrap">
            <span className="text-small font-medium text-text">{mask ? 'Measured region mask' : overlay ? 'Diagnostic overlay' : 'Original crop'}</span>
            {packet.artifacts.mask && <button type="button" aria-pressed={mask} aria-label={`Show measured mask for ${packet.id}`}
              onClick={() => { setMask((v) => !v); setOverlay(false) }}
              className="min-h-8 px-2.5 rounded-sm border border-border text-small text-text-muted hover:text-text">
              {mask ? 'Show original crop' : 'Show mask'}
            </button>}
            <button type="button" aria-pressed={overlay} aria-label={`Show diagnostic overlay for ${packet.id}`} onClick={() => { setOverlay((v) => !v); setMask(false) }}
              className={cx('min-h-8 px-2.5 rounded-sm border text-small transition-colors duration-fast',
                overlay ? 'border-accent bg-surface-3 text-accent' : 'border-border text-text-muted hover:text-text hover:bg-surface-2')}>
              {overlay ? 'Hide overlay' : 'Show overlay'}
            </button>
          </div>
          <EvidencePicture key={`${mask ? 'mask' : overlay ? 'overlay' : 'crop'}:${token}`} set={set}
            artifact={mask && packet.artifacts.mask ? packet.artifacts.mask : overlay ? packet.artifacts.overlay : packet.artifacts.crop}
            token={token} ready={mediaReady} alt={`${packet.id}: ${mask ? 'selected connected region on the full measured grid' : overlay ? 'diagnostic overlay on the same source crop' : 'unaltered source crop with surrounding context'}`} />
          <figcaption aria-live="polite" className="mt-2 text-caption text-text-muted">
            {mask ? 'White marks this selected component; black means outside this component, not verified authentic. Boundaries follow the measured grid.'
              : overlay ? packet.overlay_legend : 'Cropped from the submitted image. This is not a reconstructed “before” image.'}
          </figcaption>
        </figure>
        {donor && <figure className="m-0 min-w-0">
          <div className="mb-2 min-h-8 flex items-center text-small font-medium text-text">Matching patch</div>
          <EvidencePicture key={`copy-donor:${token}`} set={set} artifact={donor} token={token} ready={mediaReady}
            alt={`${packet.id}: matching patch elsewhere in the submitted image`} />
          <figcaption className="mt-2 text-caption text-text-muted">Cropped from the submitted image for comparison with the outlined region.</figcaption>
        </figure>}
      </div>
      <div className="mt-5 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-x-8 gap-y-4 max-[639px]:grid-cols-1">
        <div>
          <h4 className="m-0 text-small font-medium text-text">What supports this region</h4>
          <p className="mt-2 mb-0 text-small text-text">{packet.interpretation}</p>
          {packet.localisation && <p className="mt-2 mb-0 text-small text-text-muted">
            Measured grid {packet.localisation.grid.join(' × ')} · selected area {measurementValue(packet.localisation.region_area_fraction * 100)}%
            {!packet.localisation.jpeg_reencode_consistency && <> · operating threshold {measurementValue(packet.localisation.threshold)} (uncalibrated)</>}.
          </p>}
          {packet.localisation?.candidate_eligibility && <CandidateEligibilityNote gate={packet.localisation.candidate_eligibility} />}
          {packet.measurements.length > 0 && <dl className="mt-3 mb-0 space-y-1.5">
            {packet.measurements.map((m, i) => <div key={`${m.label}:${i}`} className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 text-small">
              <dt className="text-text-muted">{m.label}</dt>
              <dd className="m-0 font-mono text-text num text-right">{measurementValue(m.value)}{m.unit ? ` ${m.unit}` : ''}</dd>
            </div>)}
          </dl>}
        </div>
        <div>
          <h4 className="m-0 text-small font-medium text-text">Other explanations and limits</h4>
          <ul className="mt-2 mb-0 pl-4 space-y-1 text-small text-text-muted list-disc">
            {packet.limitations.map((line, i) => <li key={i}>{line}</li>)}
          </ul>
        </div>
      </div>
      <div className="mt-5 flex flex-wrap items-baseline gap-x-5 gap-y-2">
        <Link to={evidenceSourceUrl(set.analysis_id, packet)} data-testid="evidence-source-link"
          className="text-small font-medium text-accent underline underline-offset-[3px] hover:text-accent-hover">Inspect this position in the original image</Link>
        <span className="text-caption font-mono text-text-muted num">{coordinates}</span>
      </div>
      {(packet.finding_ids.length > 0 || packet.signal_refs.length > 0) && <p className="mt-2 mb-0 text-caption text-text-muted">
        {packet.finding_ids.length > 0 && <span>Related findings: {packet.finding_ids.map((id, i) => <span key={id}>
          {i > 0 && ', '}<Link to={`#finding-${encodeURIComponent(id)}`} className="underline underline-offset-[3px] hover:text-text">{id}</Link>
        </span>)}. </span>}
        {packet.signal_refs.length > 0 && <span>Recorded checks: {packet.signal_refs.map((s) => s.display_id ?? 'Check').join(', ')}.</span>}
      </p>}
    </article>
  )
}

function JpegCompressionHistory({ set, token, mediaReady }: { set: ImageEvidenceSet; token: string; mediaReady: boolean }) {
  const review = set.jpeg_prior_grid_review
  if (!review || review.record.source_sha256 !== set.source.sha256) return null
  const { observation, basis, scope, limitation, region, measurement } = review.record
  const cells = measurement?.supported_cells
  return <section id="image-evidence-jpeg-prior-grid" aria-labelledby="jpeg-prior-grid-heading" className="mt-4 rounded-md border border-border p-4 scroll-mt-20">
    <h3 id="jpeg-prior-grid-heading" className="m-0 text-body font-medium">Local JPEG compression history</h3>
    <p className="mt-2 mb-0 text-small">{observation}</p>
    {basis && <p className="mt-2 mb-0 text-small">{basis}</p>}
    {cells && <p className="mt-2 mb-0 text-small">Compared {cells[0]} measured cells inside the rectangle and {cells[1]} around it.</p>}
    {region && <>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">{([
        ['context', 'Highlighted area in the submitted image'], ['crop', 'Unmarked crop from the submitted image'],
      ] as const).map(([role, label]) => {
        const asset = review.artifacts[role]
        return asset && <figure key={role} className="m-0 min-w-0">
          <EvidencePicture set={set} artifact={asset} token={token} ready={mediaReady} alt={label} />
          <figcaption className="mt-2 text-small text-text-muted">{label}</figcaption>
        </figure>
      })}</div>
      <p className="mt-2 mb-0 text-small">Source columns [{region[0]}, {region[2]}) and rows [{region[1]}, {region[3]}). Right and bottom edges are exclusive.</p>
      <Link to={evidenceSourceUrl(set.analysis_id, { region: {
        x: region[0], y: region[1], w: region[2] - region[0], h: region[3] - region[1],
      } })} className="text-small font-medium text-accent underline underline-offset-[3px]">Inspect this position in the original image</Link>
    </>}
    <p className="mt-2 mb-0 text-small text-text-muted">{scope} {limitation}</p>
  </section>
}

export function ImageEvidenceContent({ set, token, mediaReady }: { set: ImageEvidenceSet; token: string; mediaReady: boolean }) {
  const [showAll, setShowAll] = useState(false)
  const { hash, key } = useLocation()
  const root = useRef<HTMLDivElement>(null)
  const landed = useRef<string | null>(null)
  useEffect(() => {
    const navigation = `${key}:${hash}:${set.id}`
    if (landed.current === navigation) return
    const index = set.packets.findIndex((packet) => hash === `#image-evidence-${encodeURIComponent(packet.id)}`)
    if (index < 0 || set.status !== 'ready') return
    if (index >= 3 && !showAll) {
      setShowAll(true)
      return
    }
    const article = document.getElementById(`image-evidence-${set.packets[index].id}`)
    if (!article || !root.current?.contains(article)) return
    landed.current = navigation
    article.focus({ preventScroll: true })
    article.scrollIntoView?.({ block: 'start' })
  }, [hash, key, set.id, set.packets, set.status, showAll])
  const localised = set.packets.some((p) => p.evidence_type === 'localisation')
  const copySkipped = set.copy_similarity?.status === 'skipped_trusted_ai_credentials'
  const interpolation = set.native_interpolation_review
  const region = interpolation?.record.region
  const quantizedPattern = interpolation?.record.quantized_pattern
  const interpolationCells = set.source.width && set.source.height
    ? Math.floor(set.source.width / 32) * Math.floor(set.source.height / 32)
    : 0
  const hasLocalEvidence = set.copy_similarity?.pairs.length || set.copy_translation?.pairs.length || set.copy_affine?.pairs.length
    || set.embedded_pair_difference?.pair || region || set.jpeg_prior_grid_review?.record.region
  const shown = showAll ? set.packets : set.packets.slice(0, 3)
  return (
    <div ref={root} data-testid="image-evidence-content" data-evidence-set={set.id ?? ''} data-evidence-hash={set.sha256 ?? ''}>
      {copySkipped && <p className="mt-3 mb-0 text-small">Trusted Content Credentials cover this image and declare AI involvement. {set.copy_affine ? 'The image copy checks were skipped.' : 'The two copy checks were skipped.'}</p>}
      <CopySimilarity set={set} field="copy_similarity" token={token} mediaReady={mediaReady} />
      <CopySimilarity set={set} field="copy_translation" token={token} mediaReady={mediaReady} />
      <CopySimilarity set={set} field="copy_affine" token={token} mediaReady={mediaReady} />
      <EmbeddedPairDifference set={set} token={token} mediaReady={mediaReady} />
      <JpegCompressionHistory set={set} token={token} mediaReady={mediaReady} />
      {interpolation && <section aria-label="Local pixel interpolation" className="mt-4 rounded-md border border-border p-4">
        <h3 className="m-0 text-body font-medium">Local pixel interpolation</h3>
        <p className="mt-2 mb-0 text-small">{interpolation.record.observation}</p>
        {interpolation.record.basis && <p className="mt-2 mb-0 text-small">{interpolation.record.basis}</p>}
        {quantizedPattern && <p className="mt-2 mb-0 text-small">Stored pixel-value step {quantizedPattern.code_step}. At least {(100 * quantizedPattern.minimum_channel_fraction).toFixed(1)}% of the measured values in each color channel followed that step.</p>}
        {!!interpolation.record.supported_cells && <p className="mt-2 mb-0 text-small">{interpolation.record.supported_cells.toLocaleString('en-US')}{interpolationCells ? ` of ${interpolationCells.toLocaleString('en-US')}` : ''} native image cells supplied comparable measurements.</p>}
        {region && <p className="mt-2 mb-0 text-small">Source columns [{region[0]}, {region[2]}) and rows [{region[1]}, {region[3]}). Right and bottom edges are exclusive.</p>}
        <p className="mt-2 mb-0 text-small text-text-muted">{interpolation.record.scope} {interpolation.record.limitation}</p>
        {region && <div className="mt-3 grid gap-4 sm:grid-cols-2">{([
          ['context', 'Region in the submitted image'], ['crop', 'Original 16-bit source crop'],
        ] as const).map(([role, label]) => {
          const asset = interpolation.artifacts[role]
          return asset && <figure className="m-0 min-w-0" key={role}>
            <EvidencePicture set={set} artifact={asset} token={token} ready={mediaReady} alt={label} />
            <figcaption className="mt-2 text-small text-text-muted">{label}</figcaption>
          </figure>
        })}</div>}
      </section>}
      <RegionalComparison set={set} token={token} mediaReady={mediaReady} />
      {set.status === 'not_generated' ? <p role="status" className="mt-3 mb-0 text-small text-text-muted">
        Region evidence has not been prepared for this result. The analysis record is still available; re-run the image analysis to create crops and supporting observations.
      </p> : set.packets.length === 0 ? (!copySkipped && <p className="mt-3 mb-0 text-small text-text-muted" data-testid="image-evidence-empty">
        {set.coverage?.inspection_checks?.length ? 'No local editing candidates are shown. The coverage below explains what was measured and withheld.'
          : hasLocalEvidence ? 'No other local editing regions were recorded.' : 'No separate local editing regions were recorded.'}
        {' '}A whole-image score cannot identify which pixels were edited, and this does not establish that the image is unaltered.
      </p>) : <>
        <p className="mt-3 mb-0 text-body text-text-muted max-w-[80ch]">
          {localised ? 'Inspect the areas identified by local checks alongside their source crops and limitations.'
            : 'These areas contain diagnostic differences worth inspecting. They are not confirmed edits, and do not establish that AI was used.'}
        </p>
        {!localised && <p data-testid="no-edit-localisation" className="mt-2 mb-0 text-small text-text-muted">This evidence set contains diagnostic inspection regions, not confirmed edit locations.</p>}
        <div id="image-evidence-cards" className="mt-5 space-y-7">{shown.map((packet) => <EvidenceCard key={`${set.id}:${packet.id}`} packet={packet} set={set} token={token} mediaReady={mediaReady} />)}</div>
        {set.packets.length > 3 && <Button variant="text" className="mt-5" aria-expanded={showAll} aria-controls="image-evidence-cards" onClick={() => setShowAll((v) => !v)}>
          {showAll ? 'Show first 3 regions' : `Show all ${set.packets.length} recorded regions`}
        </Button>}
      </>}
      <ImageInspectionChecks checks={set.coverage?.inspection_checks ?? []} coverageMap={check => {
        const grid = check.map_grid
        if (!grid) return null
        return [
          { sha: check.reliability_mask_sha256, action: 'Show which areas were checked',
            label: 'Reliability mask on the measured image grid',
            caption: 'White marks cells that passed this check’s reliability filter. Black marks cells excluded from region selection.' },
          { sha: check.flip_consistency?.aligned_mask_png_sha256, action: 'Show the aligned repeat mask',
            label: 'Horizontal repeat mask aligned to the submitted image',
            caption: 'White marks cells selected in the repeated check after mapping back to the submitted image.' },
          { sha: check.jpeg_reencode_consistency?.aligned_mask_png_sha256, action: 'Show the JPEG repeat mask',
            label: 'JPEG repeat mask aligned to the submitted image',
            caption: 'White marks cells selected after re-encoding the same image. Coordinates match the uploaded file.' },
        ].map(({ sha, action, label, caption }) => {
          if (!sha) return null
          const asset = Object.values(set.overview ?? {}).find(item => item.sha256 === sha && item.media_type === 'image/png'
            && item.width === grid[0] && item.height === grid[1])
          if (!asset) return null
          return <details key={action} className="mt-3 max-w-sm text-small text-text-muted">
            <summary className="cursor-pointer">{action}</summary>
            <figure className="m-0 mt-2">
              <EvidencePicture key={`${sha}:${token}`} set={set} artifact={asset} token={token} ready={mediaReady} alt={label} />
              <figcaption className="mt-2 text-caption">{caption}</figcaption>
            </figure>
          </details>
        })
      }} />
      {set.omitted_regions > 0 && <p className="mt-4 mb-0 text-small text-text-muted">{set.omitted_regions} additional candidate {set.omitted_regions === 1 ? 'region is' : 'regions are'} not included in these evidence cards.</p>}
      {set.limitations.length > 0 && <ul className="mt-4 mb-0 pl-4 space-y-1 text-small text-text-muted list-disc">
        {set.limitations.map((line, i) => <li key={i}>{line}</li>)}
      </ul>}
      {set.status === 'ready' && set.sha256 && <details className="mt-5 text-small text-text-muted">
        <summary className="cursor-pointer">Evidence record</summary>
        <p className="mt-2 mb-0">The same recorded crops and observations are included in new PDFs and JSON exports. Later evidence versions do not change earlier reports.</p>
        <dl className="mt-2 mb-0 space-y-2 text-caption">
          <div><dt>Evidence SHA-256</dt><dd className="m-0 font-mono break-all">{set.sha256}</dd></div>
          <div><dt>Original image SHA-256</dt><dd className="m-0 font-mono break-all">{set.source.sha256}</dd></div>
        </dl>
      </details>}
    </div>
  )
}

export default function ImageEvidenceCards({ analysisId, token, mediaReady }: { analysisId: string; token: string; mediaReady: boolean }) {
  const evidence = useImageEvidence(analysisId)
  return (
    <section id="section-image-evidence" data-testid="image-evidence" aria-labelledby="image-evidence-heading" className="scroll-mt-14">
      <h2 id="image-evidence-heading" className="m-0 eyebrow">Image evidence</h2>
      {evidence.isPending ? <SkeletonRows className="mt-4" />
        : evidence.error ? <ErrorState className="mt-3" onRetry={() => void evidence.refetch()}>Could not load the image evidence: {evidence.error.message}</ErrorState>
          : evidence.data && <ImageEvidenceContent key={evidence.data.id ?? evidence.data.status} set={evidence.data} token={token} mediaReady={mediaReady} />}
    </section>
  )
}
