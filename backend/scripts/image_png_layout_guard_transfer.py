# SPDX-License-Identifier: Apache-2.0
"""Compare a native PNG layout guard with the existing whole-image review."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'backend/data/independent-evaluation'
DOCS = ROOT / 'docs/evaluation-panels'
DEV = DOCS / 'image-whole-png-transfer-2026-11-12.json'
SCOPE = DOCS / 'image-whole-png-scope-probe-2026-11-12.json'
JPEG = DOCS / 'image-whole-fresh-sources-2026-11-12.json'
POINT = DOCS / 'image-v7-smaug-genuine-operating-point.json'
OCR = ROOT / 'models/ocr-manifest.json'
SCHEMA = 'chipmunk.image-png-layout-guard/1'
HELD_LAYOUTS = ('film', 'reader', 'cart', 'table', 'inspector', 'small')
SIZES = ((1280, 800), (1440, 900))
PUBLISHER_NAMES = ('s005_base', 's020_base', 's026_base', 's054_base', 's058_base', 's063_base', 's074_base', 's079_base')
PUBLISHER = 'https://huggingface.co/datasets/CompoSET/CompoSET/resolve/d428bf4f23f50ababb205185f179737cd9638db2'
METHOD = {'ocr_confidence': 55, 'minimum_alphanumeric': 3, 'minimum_word_width': 9,
          'minimum_word_height': 6, 'maximum_word_height': 96, 'perimeter_margin': 4,
          'minimum_ink_distance': 96, 'minimum_ink_share': .04, 'minimum_same_ink_words': 3,
          'minimum_edge_distance': 36, 'minimum_edge_pixels': 80, 'minimum_edge_share': .10,
          'flat_block_limit': .5346, 'flat_block_floor': .25}
GATE = {'maximum_new_photo_groups': 1, 'maximum_native_png_loss': 1, 'maximum_loss_per_jpeg_family': 1,
        'maximum_unexcluded_new_screen_groups': 1, 'maximum_genuine_flags': 1,
        'minimum_generated_indications_added': 8, 'minimum_generated_families_added': 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))


def capture_plan(run):
    from PIL import Image, ImageOps

    from scripts.image_whole_fresh_sources import ARCHIVE, ordinary_plan, rank

    used = {row.get('publisher_sha256') for row in read(DEV)['freeze']['items']}
    used.update(row['sha256'] for row in read(JPEG)['sources'])
    used.update(row['source_sha256'] for row in read(SCOPE)['freeze']['plan']['cases'])
    photos = []
    plan = []
    archive = ROOT / ARCHIVE
    for row in sorted(ordinary_plan(archive)['sources'], key=lambda row: rank('raster-layout/1/' + row['title'])):
        source = archive / row['file']
        if row['archive_sha256'] in used or not source.is_file():
            continue
        require(sha(source) == row['archive_sha256'], 'Prior photo source changed')
        with Image.open(source, formats=['JPEG', 'PNG']) as opened:
            photo = ImageOps.exif_transpose(opened).convert('RGB')
        if photo.width < 1440 or photo.height < 900:
            continue
        layout = HELD_LAYOUTS[len(photos)]
        photos.append(row['archive_sha256'])
        for width, height in SIZES:
            x, y = (photo.width - width) // 2, (photo.height - height) // 2
            name = f'{layout}-{width}'
            relative = f'{name}-native-photo.png'
            photo.crop((x, y, x+width, y+height)).save(run / relative, 'PNG')
            plan.append({'id': f'{name}-capture', 'layout': layout, 'size': [width, height], 'photo': relative,
                         'photo_sha256': sha(run / relative), 'crop': [x, y, x+width, y+height], 'source': row['page'],
                         'source_sha256': row['archive_sha256'], 'author': row['author'], 'license': row['license']})
        if len(photos) == len(HELD_LAYOUTS):
            break
    require(len(plan) == 12 and len(set(photos)) == 6, 'Incomplete source-disjoint capture plan')
    write(run / 'capture-plan.json', plan)


def inspected_words(binary, tessdata, path):
    # Keep recognized strings inside this subprocess result. The ledger stores only counts.
    result = subprocess.run([binary, str(path), 'stdout', '--tessdata-dir', str(tessdata), '-l', 'eng',
                             '--psm', '11', 'tsv'], capture_output=True, check=True, timeout=8,
                            env={**os.environ, 'OMP_THREAD_LIMIT': '1'})
    lines = defaultdict(list)
    for row in csv.DictReader(io.StringIO(result.stdout.decode('utf-8')), delimiter='\t'):
        if (row['level'] != '5' or float(row['conf']) < METHOD['ocr_confidence']
                or sum(char.isalnum() for char in row['text']) < METHOD['minimum_alphanumeric']):
            continue
        x, y, width, height = (int(row[key]) for key in ('left', 'top', 'width', 'height'))
        if (METHOD['minimum_word_height'] <= height <= METHOD['maximum_word_height']
                and width >= METHOD['minimum_word_width']):
            lines[tuple(row[key] for key in ('block_num', 'par_num', 'line_num'))].append((x, y, width, height))
    return lines.values()


def same_ink(rgb, lines):
    import numpy as np

    height, width = rgb.shape[:2]
    largest = 0
    margin = METHOD['perimeter_margin']
    for line in lines:
        tones = Counter()
        for x, y, w, h in line:
            require(0 <= x < x+w <= width and 0 <= y < y+h <= height, 'OCR word lies outside the source')
            pixels = rgb[y:y+h, x:x+w]
            around = rgb[max(0, y-margin):min(height, y+h+margin), max(0, x-margin):min(width, x+w+margin)]
            ring = np.concatenate((around[0], around[-1], around[:, 0], around[:, -1]))
            background = np.median(ring, axis=0)
            colors, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
            clear_ink = ((np.abs(colors-background).sum(axis=1) >= METHOD['minimum_ink_distance'])
                         & (counts >= w*h*METHOD['minimum_ink_share']))
            tones.update(map(tuple, colors[clear_ink].tolist()))
        largest = max(largest, max(tones.values(), default=0))
    return largest


def exact_edges(rgb):
    import numpy as np

    longest = []
    values = rgb.astype('int16')
    for plane in (values, values.transpose(1, 0, 2)):
        steps = plane[1:] - plane[:-1]
        aligned = ((np.abs(steps[:, 1:]).sum(axis=2) >= METHOD['minimum_edge_distance'])
                   & (steps[:, 1:] == steps[:, :-1]).all(axis=2))
        best = 0
        for row in aligned[aligned.sum(axis=1) >= METHOD['minimum_edge_pixels']-1]:
            cuts = np.diff(np.pad(row.astype('int8'), 1))
            runs = np.flatnonzero(cuts == -1) - np.flatnonzero(cuts == 1)
            best = max(best, int(runs.max(initial=0)) + 1)
        longest.append(best)
    return longest


def measures(path, binary, tessdata):
    import numpy as np
    from PIL import Image

    from app.engine.fusion.content_kinds import BLOCK, flat_block_share

    with Image.open(path, formats=['PNG']) as image:
        rgb = np.asarray(image.convert('RGB'))
    height, width = rgb.shape[:2]
    require(BLOCK == 8 and 768 <= min(width, height) <= max(width, height) <= 4920, 'Unsupported PNG grid')
    flat = round(flat_block_share(rgb) * (width // BLOCK) * (height // BLOCK))
    return {'size': [width, height], 'flat_blocks': flat, 'ink_words': same_ink(rgb, inspected_words(binary, tessdata, path)),
            'edges': exact_edges(rgb)}


def guards(record):
    width, height = record['size']
    horizontal, vertical = record['edges']
    flat = record['flat_blocks'] / ((width // 8) * (height // 8))
    layout = (record['ink_words'] >= METHOD['minimum_same_ink_words']
              or any(length >= max(METHOD['minimum_edge_pixels'], size*METHOD['minimum_edge_share'])
                     for length, size in ((horizontal, width), (vertical, height))))
    return {'flat_floor': flat > METHOD['flat_block_floor'], 'layout': flat > METHOD['flat_block_limit'] or layout}


def inputs(args):
    from PIL import Image, ImageOps

    from scripts.image_whole_png_transfer import png_identity

    rows = []
    refs = {}
    def add(panel, label, family, group, name, path, source_sha=None, expected=None, prior_score=None, credit=None):
        path = path.resolve()
        data = path.read_bytes()
        identity = digest(data)
        require(expected is None or identity == expected, 'Source file hash changed')
        size, rgb, near = png_identity(data)
        row = {'id': name, 'panel': panel, 'label': label, 'family': family, 'group': group, 'size': size,
               'sha256': identity, 'rgb_sha256': rgb, 'dhash64': near, 'bytes': len(data)}
        if source_sha:
            row['parent_sha256'] = source_sha
        if prior_score is not None:
            row['prior_score'] = prior_score
        if credit is not None:
            row['credit'] = credit
        rows.append(row)
        refs[name] = str(path)

    dev = read(DEV)
    scored = {row['id']: row['mean_logit'] for row in dev['cases']}
    for row in dev['freeze']['items']:
        label = 'screen' if row['family'] == 'browser' else row['label']
        add('development', label, row['family'], row['group'], row['id'], args.development / row['path'],
            row.get('publisher_sha256'), row['sha256'], scored[row['id']])

    scope = read(SCOPE)['freeze']
    for row in scope['cases']:
        matched = next(item for item in scope['plan']['cases'] if item['layout'] == row['layout']
                       and item['dimensions'] == row['dimensions'])
        add('retained-layout', 'screen' if row['role'] == 'screen' else 'ordinary', row['layout'],
            'retained-' + row['layout'], 'retained-' + row['id'].removesuffix('.png'), args.scope / row['id'],
            matched['source_sha256'], row['sha256'])

    for row in read(args.run / 'capture-plan.json'):
        name = row['id']
        credit = {key: row[key] for key in ('source', 'author', 'license', 'crop')}
        add('new-layout', 'ordinary', row['layout'], 'new-' + row['layout'], name + '-photo', args.run / row['photo'],
            row['source_sha256'], row['photo_sha256'], credit=credit)
        add('new-layout', 'screen', row['layout'], 'new-' + row['layout'], name, args.run / f'{name}.png',
            row['source_sha256'], credit=credit)

    for name in PUBLISHER_NAMES:
        credit = {'source': f'{PUBLISHER}/images/{name}.png', 'license': 'https://creativecommons.org/licenses/by/4.0/',
                  'license_source': f'{PUBLISHER}/README.md'}
        add('new-native', 'synthetic', 'gemini', name, name, args.publisher / f'{name}.png', credit=credit)

    native = [row for row in read(args.jpeg / 'freeze.json')['items'] if row['format'] == 'native' and row['label'] == 'synthetic']
    source_rows = read(JPEG)['sources']
    for row in native:
        source = args.jpeg / row['path']
        require(sha(source) == row['sha256'], 'Retained generated JPEG changed')
        item = source_rows[row['source_index']]
        require(item['sha256'] == row['sha256'] and item['family'] == row['family'], 'Generated JPEG source mapping changed')
        name = 'delivery-' + row['group'] if 'group' in row else 'delivery-' + item['group']
        path = args.run / f'{name}.png'
        with Image.open(source, formats=['JPEG']) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            if not path.exists():
                rgb.save(path, 'PNG')
            with Image.open(path, formats=['PNG']) as png:
                require(rgb.size == png.size and rgb.tobytes() == png.tobytes(), 'Lossless PNG pixels changed')
        add('png-from-jpeg', 'synthetic', row['family'], item['group'], name, path, row['sha256'],
            credit={'source': item['url'], 'license': item['license']})
    require(len(rows) == len(refs) == len({row['sha256'] for row in rows}) == 110, 'Incorrect PNG count or overlap')
    seen = [int(row['dhash64'], 16) for row in rows if row['panel'] == 'development' and row['label'] == 'synthetic']
    for row in (row for row in rows if row['panel'] in {'new-native', 'png-from-jpeg'}):
        value = int(row['dhash64'], 16)
        require(all((value ^ past).bit_count() > 4 for past in seen), 'Generated source near-duplicate')
        seen.append(value)
    return rows, refs


def ocr_pins(args):
    item = next(row for row in read(OCR)['files'] if row['file'] == 'eng.traineddata')
    require(sha(args.tessdata / item['file']) == item['sha256'], 'English OCR language data differs')
    return {'language': item['sha256'], 'binary': sha(args.tesseract), 'manifest': sha(OCR)}


def freeze(args):
    rows, refs = inputs(args)
    write(args.run / 'paths.json', refs)
    write(args.run / 'freeze.json', {'schema': SCHEMA, 'method': METHOD, 'gate': GATE, 'rows': rows,
                                     'point': read(POINT)['gate']['raw_mean_logit'],
                                     'pins': {path.name: sha(path) for path in (DEV, SCOPE, JPEG, POINT, Path(__file__))},
                                     'ocr': ocr_pins(args), 'browser': read(args.run / 'capture-browser.json'),
                                     'capture_plan_sha256': sha(args.run / 'capture-plan.json'),
                                     'capture_runner_sha256': sha(Path(__file__).with_name('capture_image_png_layout_controls.cjs'))})


def prepared(directory):
    lock = read(directory / 'freeze.json')
    refs = read(directory / 'paths.json')
    require(lock['schema'] == SCHEMA and lock['method'] == METHOD and lock['gate'] == GATE
            and lock['pins'][Path(__file__).name] == sha(Path(__file__))
            and lock['pins'][POINT.name] == sha(POINT), 'Frozen rule changed')
    rows = []
    for row in lock['rows']:
        path = Path(refs[row['id']])
        require(sha(path) == row['sha256'], 'Frozen PNG changed')
        rows.append({**row, 'input_path': str(path), 'input_sha256': row['sha256'], 'format': 'png'})
    return {**lock, 'operating_point': {'raw_mean_logit': lock['point']}}, rows


def measure(args):
    lock, rows = prepared(args.run)
    require(lock['ocr'] == ocr_pins(args), 'OCR runtime changed')
    results = []
    for row in rows:
        measured = measures(Path(row['input_path']), str(args.tesseract), args.tessdata)
        require(measured['size'] == row['size'], 'Native image size changed')
        results.append({'id': row['id'], 'sha256': row['sha256'], **measured, 'guards': guards(measured)})
        if len(results) % 10 == 0:
            print('measured', len(results), flush=True)
    write(args.run / 'metrics.json', {'freeze_sha256': sha(args.run / 'freeze.json'), 'rows': results})


def candidate(args):
    from scripts import run_image_whole_fresh

    def held(directory):
        lock, rows = prepared(directory)
        return lock, [row for row in rows if 'prior_score' not in row]
    run_image_whole_fresh.prepared = held
    run_image_whole_fresh.candidate(args.run)


def combine(args):
    lock, rows = prepared(args.run)
    metrics = read(args.run / 'metrics.json')
    complete = read(args.run / 'candidate-complete.json')
    journal_path = args.run / 'candidate.jsonl'
    require(metrics['freeze_sha256'] == complete['input_freeze_sha256'] == sha(args.run / 'freeze.json')
            and complete['point'] == lock['point'] and complete['candidate_journal_sha256'] == sha(journal_path),
            'Candidate or guard response differs from its freeze')
    journal = {row['id']: row for row in map(json.loads, journal_path.read_text().splitlines())}
    from scripts.review_image_png_layout_guard import audit
    outcomes = []
    for row, metrics_row in zip(rows, metrics['rows'], strict=True):
        require(metrics_row['id'] == row['id'], 'Guard evidence order changed')
        if row['panel'] == 'development':
            score = {'mean': row['prior_score'], 'status': 'reused'}
        else:
            inference = journal.pop(row['id'])
            require(inference['input_sha256'] == row['sha256'] and inference['status'] == 'ok', 'Candidate source differs')
            score = {'mean': inference['mean_logit'], 'crops': inference['crop_logits'], 'status': 'measured',
                     'crop_rgb_sha256': inference['crop_rgb_sha256']}
        outcomes.append({**metrics_row, 'candidate': score})
    require(not journal, 'Unexpected extra inference response')
    freeze_record = read(args.run / 'freeze.json')
    payload = {'freeze': freeze_record, 'freeze_sha256': sha(args.run / 'freeze.json'), 'outcomes': outcomes}
    payload['summary'] = audit(payload)
    write(args.run / 'receipt.json', payload)
    print(json.dumps(payload['summary'], sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('capture-plan', 'freeze', 'measure', 'candidate', 'combine'))
    parser.add_argument('run', type=Path)
    parser.add_argument('--scope', type=Path)
    parser.add_argument('--publisher', type=Path)
    parser.add_argument('--development', type=Path, default=DATA / 'image-whole-png-2026-11-12')
    parser.add_argument('--jpeg', type=Path, default=DATA / 'image-whole-fresh-2026-11-12')
    parser.add_argument('--tesseract', type=Path, default=Path('/opt/homebrew/opt/tesseract/bin/tesseract'))
    parser.add_argument('--tessdata', type=Path, default=ROOT / 'models/weights/ocr/tessdata')
    args = parser.parse_args()
    if args.phase == 'freeze':
        parser.error('--scope and --publisher are required for freezing') if args.scope is None or args.publisher is None else None
    if args.phase == 'capture-plan':
        capture_plan(args.run)
    else:
        {'freeze': freeze, 'measure': measure, 'candidate': candidate, 'combine': combine}[args.phase](args)


if __name__ == '__main__':
    main()
