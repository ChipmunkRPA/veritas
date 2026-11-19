"""Optional native pixel comparison for a verified public image package."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageOps

from app.evidence import image_animation_binding as animation
from app.evidence import image_embedded_pair_binding as embedded_pair
from app.evidence import image_interpolation_binding as interpolation
from app.evidence import image_motion_photo_binding as motion_photo
from app.evidence import image_previous_grid_binding as prior_grid
from app.evidence import image_printed_ids_binding as printed_ids
from app.evidence import image_printed_totals_binding as printed_totals
from app.evidence import image_transparency_binding as transparency
from app.evidence.image_copy_similarity_binding import (
    DIAGONAL_EXACT_VERSIONS,
    DIAGONAL_TRANSFORMS,
    EXPANDED_JPEG_ROTATION_VERSIONS,
    JPEG_BLOCK_BASIS,
    JPEG_DIAGONAL_BASIS,
    JPEG_DIAGONAL_KEY,
    MULTI_EXACT_BASIS,
    NATIVE_JPEG_ROTATION_VERSIONS,
    OFFGRID_BASIS,
    PNG_EXACT_REFLECTION_BASIS,
    PNG_EXACT_REFLECTION_KEY,
    ROLES,
    SMALL_PNG_BASIS,
    TURNED_JPEG_BASIS,
    TURNED_PNG_BASIS,
    WEBP_EXACT_COPY_BASIS,
    WEBP_EXACT_COPY_KEY,
    WEBP_EXACT_REFLECTION_BASIS,
    WEBP_EXACT_REFLECTION_KEY,
    WEBP_MULTI_EXACT_VERSIONS,
    box_bounds,
    browser_diagonal_predecessors_unavailable,
    copy_records,
    native_rotation_size,
)
from app.evidence.image_eligibility import repeat_pass, stored_jpeg_orientation

MAX_SOURCE_BYTES = 60 * 1024 * 1024
MAX_SOURCE_PIXELS = 80_000_000
STRIPE_ROWS = 256


def _crops(evidence):
    for packet in evidence['packets']:
        yield f"{packet['id']} source crop", packet['crop_region'], packet['artifacts']['crop']
        if 'copy_donor' in packet['artifacts']:
            yield f"{packet['id']} matching crop", packet['copy_corroboration']['donor']['box'], packet['artifacts']['copy_donor']
    for record in copy_records(embedded_pair.base_document(evidence)):
        for pair in record['pairs']:
            for role in ROLES:
                yield f"{pair['id']} {role.replace('_', ' ')}", pair['regions'][role], pair['artifacts'][role]


def _matches(source, bounds, plate):
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    if plate.format != 'PNG' or plate.mode != 'RGB' or plate.size != (width, height):
        return False
    for top in range(0, height, STRIPE_ROWS):
        bottom = min(top + STRIPE_ROWS, height)
        with source.crop((x0, y0 + top, x1, y0 + bottom)) as expected, plate.crop((0, top, width, bottom)) as actual:
            if actual.tobytes() != expected.tobytes():
                return False
    return True


def _verify_geometric_masks(archive, evidence, source):
    pairs = (evidence.get('copy_affine') or {}).get('pairs', [])
    if not pairs:
        return 0
    import numpy as np
    from cv2 import error as TransformError

    from app.detectors.image import copy_affine

    try:
        gray = copy_affine.luma(np.asarray(source))
    except TransformError as error:
        raise ValueError('The local image decoder could not prepare the geometric comparison') from error
    for pair in pairs:
        matrix = np.array(pair['matrix'], np.float64)
        try:
            masks = copy_affine.dense_pair(gray, matrix, box_bounds(pair['seed'], *source.size))
        except TransformError as error:
            raise ValueError('The local image decoder could not replay the geometric masks') from error
        if masks is None:
            raise ValueError('The recorded geometric pair lacks matching source pixels')
        for role, expected in zip(ROLES, masks, strict=True):
            x0, y0, x1, y1 = box_bounds(pair['regions'][role], *source.size)
            ys, xs = np.nonzero(expected)
            bounds = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            if (x0, y0, x1, y1) != bounds or len(xs) != pair['selected_pixels'][role]:
                raise ValueError('Geometric copy mask differs from its exact source bounds')
            expected_crop = expected[y0:y1, x0:x1]
            asset = pair['artifacts']['mask_' + role[-1]]
            with archive.open(f"assets/{asset['sha256']}.png") as stream, Image.open(stream) as plate:
                if (plate.format != 'PNG' or plate.mode != 'L' or plate.size != (x1-x0, y1-y0)
                        or not np.array_equal(np.asarray(plate), expected_crop.astype(np.uint8) * 255)):
                    raise ValueError('Geometric copy mask differs from the aligned source pixels')
    return 2 * len(pairs)


def _verify_repeat_masks(archive, checks, source, data, orientation):
    if not checks:
        return 0
    import numpy as np

    from app.detectors.image.native_forensics import unorient
    from app.evidence.image_flip_mask import png_grid, verify_overlap
    from app.evidence.image_regional_jpeg import evaluated_jpeg, reencode, tagged_orientation

    regions = 0
    geometry = {"width": source.width, "height": source.height, "exif_orientation": orientation}
    for check, key, repeat in checks:
        if key == "jpeg_reencode_consistency":
            stored_orientation = stored_jpeg_orientation(check.get("resolution_policy", {}), geometry)
            if (stored_orientation != orientation or not evaluated_jpeg(data, quality=90)
                    or (stored_orientation != 1 and tagged_orientation(data) != stored_orientation)):
                raise ValueError("The JPEG repeat view differs from the supported original")
            stored = source if stored_orientation == 1 else Image.fromarray(unorient(np.asarray(source), stored_orientation))
            if hashlib.sha256(reencode(stored)).hexdigest() != repeat["reencoded_jpeg_sha256"]:
                raise ValueError("The JPEG repeat view differs from the committed re-encoding")
        scalar = archive.read(f"assets/{check['scalar_map_sha256']}.png")
        reliable = archive.read(f"assets/{check['reliability_mask_sha256']}.png")
        aligned = archive.read(f"assets/{repeat['aligned_mask_png_sha256']}.png")
        verify_overlap(png_grid(scalar, source.size).astype(float) * (1 / 255.0), png_grid(reliable, source.size) > 127,
                       check["selection_rule"]["threshold"], repeat, aligned)
        regions += len(repeat["candidates"])
    return regions


def _verify_source_quality(archive, checks, source, data, orientation):
    if not checks:
        return 0
    import numpy as np

    from app.detectors.image import jpeg_bitstream, localiser
    from app.detectors.image.native_forensics import orient, unorient
    from app.evidence.image_flip_mask import png_grid
    from app.evidence.image_regional_jpeg import evaluated_jpeg, tagged_orientation

    structure = jpeg_bitstream.read(data)
    geometry = {"width": source.width, "height": source.height, "exif_orientation": orientation}
    pixels = np.asarray(source)
    masks = {}
    for check in checks:
        stored_orientation = stored_jpeg_orientation(check.get("resolution_policy", {}), geometry)
        if stored_orientation not in masks:
            if stored_orientation != 1 and (not evaluated_jpeg(data, quality=90) or tagged_orientation(data) != stored_orientation):
                raise ValueError("The oriented quality mask requires a supported original JPEG")
            unusable, shares = localiser.conditions_mask(unorient(pixels, stored_orientation), structure.quality_luma if structure else None)
            masks[stored_orientation] = orient(unusable, stored_orientation), shares
        unusable, shares = masks[stored_orientation]
        grid = tuple(check["map_grid"])
        if len(grid) != 2 or any(type(value) is not int or not 0 < value <= limit
                                 for value, limit in zip(grid, source.size, strict=True)):
            raise ValueError("The image quality mask grid exceeds the supplied image")
        recorded = archive.read(f"assets/{check['reliability_mask_sha256']}.png")
        expected = localiser.reliable_grid(unusable, grid)
        if (shares != check["source_quality"]["native_pixel_shares"]
                or not np.array_equal(png_grid(recorded, grid), expected.astype(np.uint8) * 255)
                or round(1 - float(expected.mean()), 6) != round(check["excluded_area_fraction"], 6)):
            raise ValueError("The image quality mask or condition shares differ from the supplied source")
    return len(checks)


def check(archive: ZipFile, bundle: dict, original: Path) -> str:
    """Bind source bytes and EXIF coordinates, then compare every unannotated crop."""
    from app.reports.public_verify import IMAGE_EDITOR_PREVIEW, SCHEMA

    if bundle.get('schema') != SCHEMA or bundle.get('analysis', {}).get('kind') != 'image':
        raise ValueError('Original-image verification requires a public image evidence package')
    with original.open('rb') as stream:
        data = stream.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError('The original exceeds the image file size limit')
    if hashlib.sha256(data).hexdigest() != bundle['source_commitment']['sha256']:
        raise ValueError('The supplied image SHA-256 differs from the committed original')

    evidence = json.loads(archive.read('evidence.json'))
    native_review = evidence.get(interpolation.KEY)
    native_detail = ""
    if native_review is not None:
        from app.evidence.image_interpolation_source import verify

        interpolation.validate(evidence, evidence["source"].get(interpolation.COMMITMENT))
        if verify(data, native_review, lambda digest: archive.read(f"assets/{digest}.png")):
            native_detail = " Recomputed the native interpolation inspection and any exact 16-bit source crop."
    if review := evidence.get(prior_grid.KEY):
        from app.evidence.image_previous_grid_source import verify

        prior_grid.validate(evidence, evidence["source"].get(prior_grid.COMMITMENT))
        if verify(data, review, lambda digest: archive.read(f"assets/{digest}.png")):
            native_detail += " Replayed the local JPEG compression record from the original."
            if review["record"]["region"] is not None:
                native_detail += " Matched the marked image and exact source crop."
    if review := evidence.get(printed_ids.KEY):
        from app.evidence.image_printed_ids import verify_crops

        printed_ids.validate(evidence, evidence["source"].get(printed_ids.COMMITMENT))
        with Image.open(io.BytesIO(data), formats=["PNG", "JPEG"]) as raw:
            if (not printed_ids.size_supported(raw.size) or raw.getexif().get(274, 1) != 1
                    or raw.size != (evidence["source"]["width"], evidence["source"]["height"])):
                raise ValueError("Printed identifier original does not match its source dimensions")
            with raw.convert("RGB") as raster:
                verify_crops(raster, review, lambda digest: archive.read(f"assets/{digest}.png"))
        native_detail += " Matched both printed identifier crops to the submitted image."
    if review := evidence.get(printed_totals.KEY):
        from app.evidence.image_printed_totals import verify_crops

        printed_totals.validate(evidence, evidence["source"].get(printed_totals.COMMITMENT))
        with Image.open(io.BytesIO(data), formats=["PNG"]) as raw:
            if (raw.size != (evidence["source"]["width"], evidence["source"]["height"])
                    or raw.getexif().get(274, 1) != 1 or raw.mode not in {"RGB", "L"}
                    or "transparency" in raw.info or raw.n_frames != 1):
                raise ValueError("Printed total evidence requires the original upright opaque PNG")
            with raw.convert("RGB") as raster:
                verify_crops(raster, review, lambda digest: archive.read(f"assets/{digest}.png"))
        native_detail += " Matched all three printed total crops to the submitted image."
    projected = json.loads(archive.read('result.json')) if 'result.json' in archive.namelist() else {}
    preview = projected.get(IMAGE_EDITOR_PREVIEW)
    if transparency.KEY in projected:
        from app.evidence.image_transparency import verify

        record = projected[transparency.KEY]
        transparency.assets(record, evidence["source"])
        matched = verify(data, record, lambda digest: archive.read(f"assets/{digest}.png"))
        native_detail += " Recomputed the PNG transparency counts" + (" and both source crops." if matched else ".")
    if animation.KEY in projected:
        from app.evidence.image_animation import verify

        record = projected[animation.KEY]
        source = evidence["source"]
        animation.assets(record, source)
        verify(data, record, lambda digest: archive.read(f"assets/{digest}.png"),
               source_size=(source["width"], source["height"]))
        native_detail += " Recomputed the animated PNG opening comparison and verified any recorded previews."
    if motion_photo.KEY in projected:
        from app.evidence.image_motion_photo import verify

        source = evidence["source"]
        if verify(data, projected[motion_photo.KEY], lambda digest: archive.read(f"assets/{digest}.png"),
                  source_size=(source["width"], source["height"])):
            native_detail += " Replayed the Motion Photo times, comparison and recorded crops."
    crops = list(_crops(evidence))
    pair = (evidence.get(embedded_pair.KEY) or {}).get('pair')
    checks = (evidence.get("coverage") or {}).get("inspection_checks", [])
    repeat_checks = [(check, key, repeat) for check in checks for key, repeat in (repeat_pass(check),)
                     if repeat and repeat.get("aligned_mask_png_sha256")]
    quality_checks = [check for check in (evidence.get("coverage") or {}).get("inspection_checks", [])
                      if check.get("source_quality") is not None]
    if not crops and not repeat_checks and not quality_checks and pair is None and preview is None:
        return 'Original SHA-256 matched.' + (native_detail or ' No unannotated crops were recorded.')
    declared = evidence['source']
    if preview is not None:
        from app.evidence.image_editor_preview_source import verify as verify_preview

        preview_proof = verify_preview(data, preview, lambda digest: archive.read(f'assets/{digest}.png'),
                                     source_size=(declared['width'], declared['height']))
        if not crops and not repeat_checks and not quality_checks and pair is None:
            return 'Original SHA-256 matched.' + _preview_detail(preview_proof, preview["preview_carrier"]) + native_detail
    else:
        preview_proof = None
    if data[4:8] == b"ftyp":
        from pillow_heif import register_heif_opener

        register_heif_opener()
    with Image.open(io.BytesIO(data)) as raw:
        if raw.width * raw.height > MAX_SOURCE_PIXELS:
            raise ValueError('The original exceeds the image pixel limit')
        orientation = int(raw.getexif().get(274, 1) or 1)
        with ImageOps.exif_transpose(raw) as oriented:
            source = oriented.convert('RGB')
    translation = evidence.get('copy_translation') or {}
    offgrid = translation.get('basis') == OFFGRID_BASIS
    turned_jpeg = translation.get("basis") == TURNED_JPEG_BASIS
    diagonal_jpeg = translation.get("basis") == JPEG_DIAGONAL_BASIS
    with source:
        if (declared.get('coordinate_space') != 'exif-oriented-original'
                or source.size != (declared.get('width'), declared.get('height'))
                or declared.get('exif_orientation', orientation) != orientation):
            raise ValueError('The oriented image differs from the committed source geometry')
        for label, box, asset in crops:
            bounds = box_bounds(box, *source.size)
            with archive.open(f"assets/{asset['sha256']}.png") as stream, Image.open(stream) as plate:
                if not _matches(source, bounds, plate):
                    raise ValueError(f'{label} differs from pixels at its recorded source coordinates')
        masks = _verify_geometric_masks(archive, evidence, source)
        repeated = _verify_repeat_masks(archive, repeat_checks, source, data, orientation)
        quality = _verify_source_quality(archive, quality_checks, source, data, orientation)
        if offgrid:
            import numpy as np

            from app.evidence.image_jpeg_offgrid_source import verify

            verify(data, np.asarray(source), translation['pairs'][0], translation['jpeg_offgrid_review']['measurement'])
        if diagonal_jpeg:
            import numpy as np

            from app.evidence.image_jpeg_diagonal_source import verify

            version = embedded_pair.base_document(evidence)["version"]
            verify(data, np.asarray(source), translation["pairs"][0], translation[JPEG_DIAGONAL_KEY]["measurement"],
                   version=version, browser_predecessors_unavailable=browser_diagonal_predecessors_unavailable(translation))
        if turned_jpeg:
            import numpy as np

            from app.evidence.image_jpeg_turned_copy_source import verify_pairs

            version = embedded_pair.base_document(evidence)["version"]
            verify_pairs(data, np.asarray(source), translation, expanded=version in EXPANDED_JPEG_ROTATION_VERSIONS,
                         native=version in NATIVE_JPEG_ROTATION_VERSIONS and native_rotation_size(*source.size))
    if pair is not None:
        from app.evidence import image_embedded_pair_pixels as embedded_pixels

        original_png = embedded_pixels.source_image(data)
        if original_png is None:
            raise ValueError('Exact two-area evidence requires a PNG within the tested source scope')
        with original_png:
            embedded_pixels.verify(original_png, pair, lambda digest: archive.read(f'assets/{digest}.png'))
    basis = translation.get("basis")
    multi = basis == MULTI_EXACT_BASIS
    if multi:
        from app.evidence.image_exact_small_copy_source import verify_multiple

        webp = embedded_pair.base_document(evidence)["version"] in WEBP_MULTI_EXACT_VERSIONS
        verify_multiple(data, translation, webp=webp)
    small_png = basis in {SMALL_PNG_BASIS, TURNED_PNG_BASIS, PNG_EXACT_REFLECTION_BASIS}
    exact_webp = basis in {WEBP_EXACT_COPY_BASIS, WEBP_EXACT_REFLECTION_BASIS}
    reflected = basis in {PNG_EXACT_REFLECTION_BASIS, WEBP_EXACT_REFLECTION_BASIS}
    if small_png or exact_webp:
        from app.evidence.image_exact_small_copy_source import verify

        key = {WEBP_EXACT_REFLECTION_BASIS: WEBP_EXACT_REFLECTION_KEY, WEBP_EXACT_COPY_BASIS: WEBP_EXACT_COPY_KEY,
               PNG_EXACT_REFLECTION_BASIS: PNG_EXACT_REFLECTION_KEY}.get(basis, "small_png_copy_review")
        verify(data, translation["pairs"][0], translation[key]["measurement"], webp=exact_webp, reflected=reflected,
               include_diagonal=embedded_pair.base_document(evidence)["version"] in DIAGONAL_EXACT_VERSIONS)
    matched_blocks = 0
    if translation.get('basis') == JPEG_BLOCK_BASIS:
        from app.evidence.image_jpeg_copy_blocks_source import verify

        matched_blocks = verify(data, translation['pairs'])
    crop_count = len(crops) + (2 if pair is not None else 0)
    detail = (f'Original SHA-256 and {crop_count} unannotated crops matched at their recorded EXIF-oriented coordinates.'
              if crop_count else 'Original SHA-256 matched.')
    if preview_proof:
        detail += _preview_detail(preview_proof, preview["preview_carrier"])
    elif not crop_count and not native_detail:
        detail += ' No unannotated crops were recorded.'
    if pair is not None:
        detail += f' Recomputed {pair["different_pixels"]} differing pixels in the exact aligned comparison.'
    if masks:
        detail += f' Recomputed {masks} geometric copy masks from the source image.'
    if matched_blocks:
        detail += f' Reproduced {matched_blocks} matching encoded JPEG brightness blocks per paired area in total.'
    if small_png or exact_webp:
        transform = translation["pairs"][0]["transform"]
        alignment = f" after {transform.replace('_', ' ')}" if reflected else " after rotation" if transform != "same_orientation" else ""
        if transform in DIAGONAL_TRANSFORMS:
            alignment += " diagonal reflection"
        carrier = "lossless WebP" if exact_webp else "PNG"
        detail += f" Recomputed the exact matching RGB pixels and surrounding differences{alignment} on the submitted {carrier}."
    if multi:
        detail += " Recomputed exact RGB equality and surrounding differences for both copy pairs."
    if offgrid:
        detail += ' Recomputed the recorded color differences inside and around the two selected JPEG crops.'
    if diagonal_jpeg:
        detail += " Recomputed the color differences inside and around the two diagonally aligned JPEG crops."
    if turned_jpeg:
        number = "four" if len(translation["pairs"]) == 2 else "two"
        detail += f" Recomputed the color differences inside and around the {number} selected JPEG crops after rotation."
    if repeated:
        detail += f' Recomputed repeat-pass overlap for {repeated} source regions from their recorded masks.'
    if any(key == "jpeg_reencode_consistency" for _, key, _ in repeat_checks):
        detail += ' Reproduced the second JPEG view from the supplied image.'
    if quality:
        detail += f' Recomputed source quality masks for {quality} image {"check" if quality == 1 else "checks"}.'
    return detail + native_detail


def _preview_detail(proof, carrier):
    from app.evidence.image_editor_preview_binding import HEIF_CARRIERS, PNG_CARRIER

    count = proof["footprints"]
    label = "linked HEIC thumbnail" if carrier in HEIF_CARRIERS else "PNG Exif preview" if carrier == PNG_CARRIER else "editor-preview"
    return (f' Matched {proof["source_crops"]} main-image and stored thumbnail crops. '
            f'Recomputed {count} saved {label} {"footprint" if count == 1 else "footprints"}.')
