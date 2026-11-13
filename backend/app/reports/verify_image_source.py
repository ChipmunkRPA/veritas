"""Optional native pixel comparison for a verified public image package."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageOps

from app.evidence import image_embedded_pair_binding as embedded_pair
from app.evidence.image_copy_similarity_binding import (
    JPEG_BLOCK_BASIS,
    OFFGRID_BASIS,
    ROLES,
    box_bounds,
    copy_records,
)

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


def _verify_flip_masks(archive, checks, size):
    if not checks:
        return 0
    from app.evidence.image_flip_mask import png_grid, verify_overlap

    regions = 0
    for check in checks:
        repeat = check["flip_consistency"]
        scalar = archive.read(f"assets/{check['scalar_map_sha256']}.png")
        reliable = archive.read(f"assets/{check['reliability_mask_sha256']}.png")
        aligned = archive.read(f"assets/{repeat['aligned_mask_png_sha256']}.png")
        verify_overlap(png_grid(scalar, size).astype(float) * (1 / 255.0), png_grid(reliable, size) > 127,
                       check["selection_rule"]["threshold"], repeat, aligned)
        regions += len(repeat["candidates"])
    return regions


def _verify_source_quality(archive, checks, source, data):
    if not checks:
        return 0
    import numpy as np

    from app.detectors.image import jpeg_bitstream, localiser
    from app.evidence.image_flip_mask import png_grid

    structure = jpeg_bitstream.read(data)
    unusable, shares = localiser.conditions_mask(np.asarray(source), structure.quality_luma if structure else None)
    for check in checks:
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
    preview = json.loads(archive.read('result.json')).get(IMAGE_EDITOR_PREVIEW) if 'result.json' in archive.namelist() else None
    crops = list(_crops(evidence))
    pair = (evidence.get(embedded_pair.KEY) or {}).get('pair')
    repeat_checks = [check for check in (evidence.get("coverage") or {}).get("inspection_checks", [])
                     if (check.get("flip_consistency") or {}).get("aligned_mask_png_sha256")]
    quality_checks = [check for check in (evidence.get("coverage") or {}).get("inspection_checks", [])
                      if check.get("source_quality") is not None]
    if not crops and not repeat_checks and not quality_checks and pair is None and preview is None:
        return 'Original SHA-256 matched. No unannotated crops were recorded.'
    declared = evidence['source']
    if preview is not None:
        from app.evidence.image_editor_preview_source import verify as verify_preview

        preview_proof = verify_preview(data, preview, lambda digest: archive.read(f'assets/{digest}.png'),
                                     source_size=(declared['width'], declared['height']))
        if not crops and not repeat_checks and not quality_checks and pair is None:
            return 'Original SHA-256 matched.' + _preview_detail(preview_proof, preview["preview_carrier"])
    else:
        preview_proof = None
    with Image.open(io.BytesIO(data)) as raw:
        if raw.width * raw.height > MAX_SOURCE_PIXELS:
            raise ValueError('The original exceeds the image pixel limit')
        orientation = int(raw.getexif().get(274, 1) or 1)
        with ImageOps.exif_transpose(raw) as oriented:
            source = oriented.convert('RGB')
    translation = evidence.get('copy_translation') or {}
    offgrid = translation.get('basis') == OFFGRID_BASIS
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
        repeated = _verify_flip_masks(archive, repeat_checks, source.size)
        quality = _verify_source_quality(archive, quality_checks, source, data)
        if offgrid:
            import numpy as np

            from app.evidence.image_jpeg_offgrid_source import verify

            verify(data, np.asarray(source), translation['pairs'][0], translation['jpeg_offgrid_review']['measurement'])
    if pair is not None:
        from app.evidence import image_embedded_pair_pixels as embedded_pixels

        original_png = embedded_pixels.source_image(data)
        if original_png is None:
            raise ValueError('Exact two-area evidence requires a PNG within the tested source scope')
        with original_png:
            embedded_pixels.verify(original_png, pair, lambda digest: archive.read(f'assets/{digest}.png'))
    matched_blocks = 0
    if translation.get('basis') == JPEG_BLOCK_BASIS:
        from app.evidence.image_jpeg_copy_blocks_source import verify

        matched_blocks = verify(data, translation['pairs'])
    crop_count = len(crops) + (2 if pair is not None else 0)
    detail = (f'Original SHA-256 and {crop_count} unannotated crops matched at their recorded EXIF-oriented coordinates.'
              if crop_count else 'Original SHA-256 matched.')
    if preview_proof:
        detail += _preview_detail(preview_proof, preview["preview_carrier"])
    elif not crop_count:
        detail += ' No unannotated crops were recorded.'
    if pair is not None:
        detail += f' Recomputed {pair["different_pixels"]} differing pixels in the exact aligned comparison.'
    if masks:
        detail += f' Recomputed {masks} geometric copy masks from the source image.'
    if matched_blocks:
        detail += f' Reproduced {matched_blocks} matching encoded JPEG brightness blocks per paired area in total.'
    if offgrid:
        detail += ' Recomputed the recorded color differences inside and around the two selected JPEG crops.'
    if repeated:
        detail += f' Recomputed repeat-pass overlap for {repeated} source regions from their recorded masks.'
    if quality:
        detail += f' Recomputed source quality masks for {quality} image {"check" if quality == 1 else "checks"}.'
    return detail


def _preview_detail(proof, carrier):
    from app.evidence.image_editor_preview_binding import PNG_CARRIER

    count = proof["footprints"]
    label = "PNG Exif preview" if carrier == PNG_CARRIER else "editor-preview"
    return (f' Matched {proof["source_crops"]} main-image and stored thumbnail crops. '
            f'Recomputed {count} saved {label} {"footprint" if count == 1 else "footprints"}.')
