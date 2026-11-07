"""Synthetic native crop parity; no neural model/import or project media."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps
from PIL import __version__ as pillow_version

from scripts.image_smaug_views_v7 import views

SOURCE_SHA = 'ad8a17d78e610127d6e23fc7fb4af27a3e7a39da5734ea643dfaafdc667f282a'


def audit(reference, output):
    if output.exists() or hashlib.sha256(reference.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Exact inspected publisher source and a new audit receipt required')
    tree = ast.parse(reference.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'views']
    if len(functions) != 1:
        raise ValueError('Only the reviewed standalone crop function may be executed')
    namespace = {'math': math, 'Image': Image, 'ImageOps': ImageOps, 'SIDE': 576, 'CROP': 528}
    # Exact pinned, inspected pure PIL crop function only. No imports, class,
    # model loader, automatic device selection or other publisher module code.
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<reviewed-publisher-views>', 'exec'), namespace)
    records = []
    for width, height in ((400, 700), (700, 400), (528, 600), (576, 1001),
                          (577, 1003), (800, 800), (1003, 577), (531, 529)):
        pixels = bytes((x * 17 + y * 23 + c * 79) % 256
                       for y in range(height) for x in range(width) for c in range(3))
        for orientation in range(1, 9):
            image = Image.frombytes('RGB', (width, height), pixels)
            image.getexif()[274] = orientation
            for count in (1, 3):
                expected = namespace['views'](image, count)
                actual, traces = views(image, count)
                if (len(actual) != count or len(traces) != count
                        or any(a.size != (528, 528) or a.tobytes() != b.tobytes()
                               for a, b in zip(actual, expected, strict=True))):
                    raise ValueError('Complete synthetic native crop bytes differ')
                for trace in traces:
                    rw, rh = trace['resized_size']
                    ow, oh = trace['oriented_original_size']
                    box = trace['resized_crop_xyxy']
                    if (box[0] < 0 or box[1] < 0 or box[2] > rw or box[3] > rh
                            or any(n < 0 or n > maximum * denominator
                                   for (n, denominator), maximum in zip(trace['source_edge_bounds_rational'],
                                       (ow, oh, ow, oh), strict=True))):
                        raise ValueError('Crop/source correspondence leaves the oriented source')
                records.append({'input_size': [width, height], 'exif_orientation': orientation,
                    'crop_count': count, 'all_native_pixels_exact': True,
                    'crop_rgb_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in actual], 'traces': traces})
    result = {'schema': 'chipmunk.image-smaug-synthetic-views-audit/7', 'pillow': pillow_version,
        'reference_source_sha256': SOURCE_SHA, 'audit_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'local_views_code_sha256': hashlib.sha256(Path(__file__).with_name('image_smaug_views_v7.py').read_bytes()).hexdigest(),
        'passed': True, 'synthetic_cases': len(records), 'crops': sum(r['crop_count'] for r in records),
        'cases': records, 'reviewed_publisher_crop_function_executed': True,
        'publisher_full_module_imported': False, 'model_initialized': False, 'project_media_scored': False,
        'scope': 'Synthetic crop/orientation/source-coordinate parity only; not processor tokenization, neural inference, classifier accuracy, pixel edit localization or outside human review.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    audit(args.reference, args.output)
