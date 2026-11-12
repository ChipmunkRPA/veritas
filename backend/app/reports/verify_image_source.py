"""Optional native pixel comparison for a verified public image package."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageOps

from app.evidence.image_copy_similarity_binding import ROLES, box_bounds, copy_records

MAX_SOURCE_BYTES = 60 * 1024 * 1024
MAX_SOURCE_PIXELS = 80_000_000
STRIPE_ROWS = 256


def _crops(evidence):
    for packet in evidence['packets']:
        yield f"{packet['id']} source crop", packet['crop_region'], packet['artifacts']['crop']
        if 'copy_donor' in packet['artifacts']:
            yield f"{packet['id']} matching crop", packet['copy_corroboration']['donor']['box'], packet['artifacts']['copy_donor']
    for record in copy_records(evidence):
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


def check(archive: ZipFile, bundle: dict, original: Path) -> str:
    """Bind source bytes and EXIF coordinates, then compare every unannotated crop."""
    from app.reports.public_verify import SCHEMA

    if bundle.get('schema') != SCHEMA or bundle.get('analysis', {}).get('kind') != 'image':
        raise ValueError('Original-image verification requires a public image evidence package')
    with original.open('rb') as stream:
        data = stream.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError('The original exceeds the image file size limit')
    if hashlib.sha256(data).hexdigest() != bundle['source_commitment']['sha256']:
        raise ValueError('The supplied image SHA-256 differs from the committed original')

    evidence = json.loads(archive.read('evidence.json'))
    crops = list(_crops(evidence))
    repeat_checks = [check for check in (evidence.get("coverage") or {}).get("inspection_checks", [])
                     if (check.get("flip_consistency") or {}).get("aligned_mask_png_sha256")]
    if not crops and not repeat_checks:
        return 'Original SHA-256 matched. No unannotated crops were recorded.'
    declared = evidence['source']
    with Image.open(io.BytesIO(data)) as raw:
        if raw.width * raw.height > MAX_SOURCE_PIXELS:
            raise ValueError('The original exceeds the image pixel limit')
        orientation = int(raw.getexif().get(274, 1) or 1)
        with ImageOps.exif_transpose(raw) as oriented:
            source = oriented.convert('RGB')
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
    detail = (f'Original SHA-256 and {len(crops)} unannotated crops matched at their recorded EXIF-oriented coordinates.'
              if crops else 'Original SHA-256 matched. No unannotated crops were recorded.')
    if masks:
        detail += f' Recomputed {masks} geometric copy masks from the source image.'
    if repeated:
        detail += f' Recomputed repeat-pass overlap for {repeated} source regions from their recorded masks.'
    return detail
