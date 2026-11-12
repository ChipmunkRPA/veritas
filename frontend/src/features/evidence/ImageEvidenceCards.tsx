import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Button, ErrorState, SkeletonRows } from '../../components/ui'
import { cx } from '../../lib/cx'
import { evidenceImageUrl, evidenceSourceUrl, measurementValue, useImageEvidence, type EvidenceImage, type ImageEvidencePacket, type ImageEvidenceSet } from './imagePackets'
import ImageInspectionChecks, { CandidateEligibilityNote } from './ImageInspectionChecks'

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
}

const jpegCoverageText = {
  supported_pair: copyBasisText.jpeg_brightness_blocks,
  no_pair: 'The JPEG brightness-block check found no qualifying repeated area.',
  unsupported_profile: 'The brightness-block check requires an upright JPEG within its tested size and encoding scope.',
  unavailable: 'The JPEG brightness-block check could not read the encoded blocks.',
  unassessed_match_budget: 'The brightness-block check found too many competing offsets.',
}

function CopySimilarity({ set, field, token, mediaReady }: {
  set: ImageEvidenceSet; field: 'copy_similarity' | 'copy_translation' | 'copy_affine'; token: string; mediaReady: boolean
}) {
  const check = set[field]
  if (!check || check.status === 'skipped_trusted_ai_credentials') return null
  const reflected = field === 'copy_similarity'
  const geometric = field === 'copy_affine'
  const heading = geometric ? 'Geometric copy comparison' : reflected ? 'Same-image copy similarity' : 'Copy similarity without mirroring'
  const sectionId = `image-${field.replace('_', '-')}`
  const baseVersion = set.prior_embedded_pair_evidence?.version ?? set.version
  const wide = (geometric && ['image-evidence-14', 'image-evidence-17'].includes(baseVersion)) || (reflected &&
    ['image-evidence-9', 'image-evidence-10', 'image-evidence-11', 'image-evidence-12', 'image-evidence-13',
      'image-evidence-14', 'image-evidence-16', 'image-evidence-17'].includes(baseVersion))
  const basis = field === 'copy_translation' && set.copy_translation?.basis
  const jpegReview = field === 'copy_translation' && set.copy_translation?.jpeg_block_review
  const basisText = jpegReview ? jpegCoverageText[jpegReview] : basis && check.pairs.length > 0 && copyBasisText[basis]
  const proposalSearch = geometric ? set.copy_affine?.proposal_search : undefined
  const statusText = {
    supported_pair: geometric ? 'Matching detail was found after geometric alignment.'
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
    <p className="mt-2 mb-0 text-small text-text">{statusText[check.status]}</p>
    <p className="mt-2 mb-0 text-small text-text-muted">{check.interpretation}</p>
    {!reflected && <p className="mt-2 mb-0 text-caption text-text-muted">This check carries zero weight in AI authorship confidence.</p>}
    {basisText && <p className="mt-2 mb-0 text-caption text-text-muted">{basisText}</p>}
    {proposalSearch?.basis === 'reduced' && <p className="mt-2 mb-0 text-caption text-text-muted">The candidate search used a {proposalSearch.width} × {proposalSearch.height} pixel working image. The check measured any matching masks on the original pixel grid.</p>}
    {check.pairs.length > 0 && <p className="mt-2 mb-0 text-caption text-text-muted">Coordinates use native pixels in the oriented original.</p>}
    {check.pairs.map(pair => <article key={pair.id} className="mt-4 border-t border-border pt-4">
      <h4 className="m-0 text-small font-medium text-text">{pair.id} · {geometric ? 'Matched source areas' : reflected ? 'Reflected pair' : 'Non-mirrored pair'}</h4>
      <div className="mt-3 grid grid-cols-2 gap-4 max-[639px]:grid-cols-1">
        {(['area_a', 'area_b'] as const).map((role, index) => {
          const region = pair.regions[role]
          const label = index === 0 ? 'Area A' : 'Area B'
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
    </article>)}
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
            Measured grid {packet.localisation.grid.join(' × ')} · selected area {measurementValue(packet.localisation.region_area_fraction * 100)}% · operating threshold {measurementValue(packet.localisation.threshold)} (uncalibrated).
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
  const hasCopyPair = set.copy_similarity?.pairs.length || set.copy_translation?.pairs.length || set.copy_affine?.pairs.length || set.embedded_pair_difference?.pair
  const shown = showAll ? set.packets : set.packets.slice(0, 3)
  return (
    <div ref={root} data-testid="image-evidence-content" data-evidence-set={set.id ?? ''} data-evidence-hash={set.sha256 ?? ''}>
      {copySkipped && <p className="mt-3 mb-0 text-small">Trusted Content Credentials cover this image and declare AI involvement. {set.copy_affine ? 'The image copy checks were skipped.' : 'The two copy checks were skipped.'}</p>}
      <CopySimilarity set={set} field="copy_similarity" token={token} mediaReady={mediaReady} />
      <CopySimilarity set={set} field="copy_translation" token={token} mediaReady={mediaReady} />
      <CopySimilarity set={set} field="copy_affine" token={token} mediaReady={mediaReady} />
      <EmbeddedPairDifference set={set} token={token} mediaReady={mediaReady} />
      <RegionalComparison set={set} token={token} mediaReady={mediaReady} />
      {set.status === 'not_generated' ? <p role="status" className="mt-3 mb-0 text-small text-text-muted">
        Region evidence has not been prepared for this result. The analysis record is still available; re-run the image analysis to create crops and supporting observations.
      </p> : set.packets.length === 0 ? (!copySkipped && <p className="mt-3 mb-0 text-small text-text-muted" data-testid="image-evidence-empty">
        {set.coverage?.inspection_checks?.length ? 'No local editing candidates are shown. The coverage below explains what was measured and withheld.'
          : hasCopyPair ? 'No other local editing regions were recorded.' : 'No separate local editing regions were recorded.'}
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
        const sha = check.reliability_mask_sha256
        if (!grid || !sha) return null
        const asset = Object.values(set.overview ?? {}).find(item => item.sha256 === sha && item.media_type === 'image/png'
          && item.width === grid[0] && item.height === grid[1])
        if (!asset) return null
        return <details className="mt-3 max-w-sm text-small text-text-muted">
          <summary className="cursor-pointer">Show which areas were checked</summary>
          <figure className="m-0 mt-2">
            <EvidencePicture key={`${sha}:${token}`} set={set} artifact={asset} token={token} ready={mediaReady}
              alt="Reliability mask on the measured image grid" />
            <figcaption className="mt-2 text-caption">White marks cells that passed this check’s reliability filter. Black marks cells excluded from region selection.</figcaption>
          </figure>
        </details>
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
