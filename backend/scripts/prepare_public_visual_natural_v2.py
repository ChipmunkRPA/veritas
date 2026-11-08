"""Select and acquire licensed natural-media controls before watermark inference."""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import subprocess
import urllib.parse
import urllib.request
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts import public_visual_watermark_v1 as m

ROOT = Path(__file__).resolve().parents[2]
QUOTAS = {'photos-architecture-metadata.json': 10, 'photos-food-metadata.json': 10,
          'photos-plants-metadata.json': 6, 'photo-api-preflight.json': 6}
VIDEO_POOLS = {'video-rivers-metadata.json', 'video-waterfalls-metadata.json'}
BAD = re.compile(r'animation|computer.generated|diagram|illustration|screenshot|military|killing', re.I)


def canonical(url):
    p = urllib.parse.urlsplit(url)
    return urllib.parse.unquote(p.path).casefold()


def inventory(main, metadata, trial):
    # Search all JSON/JSONL source inventories, not a size-limited subset.
    result = subprocess.run(['rg', '-l', '--glob', '*.json', '--glob', '*.jsonl',
        r'(commons|upload|thumb)\.wikimedia\.org', str(main / 'backend/data')],
        capture_output=True, text=True, check=False)
    m.require(result.returncode in (0, 1), 'Prior source inventory search failed')
    excluded, manifests = set(), []
    for name in result.stdout.splitlines():
        path = Path(name).resolve()
        if path.is_relative_to(metadata) or path.is_relative_to(trial):
            continue
        matches = subprocess.run(['rg', '-o', r'https?://(?:commons|upload|thumb)\.wikimedia\.org[^"\s<>]*',
            str(path)], capture_output=True, text=True, check=False)
        m.require(matches.returncode in (0, 1), 'Prior source URL extraction failed')
        excluded.update(canonical(u) for u in matches.stdout.splitlines())
        manifests.append({'path': str(path.relative_to(main)), 'sha256': m.digest(path)})
    return excluded, manifests


def plan(main, metadata, trial):
    metadata = metadata.resolve()
    audit_path = metadata / 'metadata-eligibility-audit.json'
    audit = m.read(audit_path)
    pages = {}
    for row in audit['metadata_files']:
        path = metadata / row['name']
        m.require(m.digest(path) == row['sha256'], 'Preflight metadata changed')
        for page in m.read(path).get('query', {}).get('pages', {}).values():
            pages[page['pageid']] = page
    exclusions, manifests = inventory(main, metadata, trial)
    eligible, decisions = [], []
    for row in audit['records']:
        reasons = list(row['reasons'])
        page = pages[row['page_id']]
        info = (page.get('imageinfo') or page.get('videoinfo'))[0]
        description = info.get('extmetadata', {}).get('ImageDescription', {}).get('value', '')
        if BAD.search(html.unescape(page['title'] + ' ' + description)):
            reasons.append('non_photographic_or_out_of_scope_description')
        if canonical(info['url']) in exclusions:
            reasons.append('previous_project_source_url')
        if row['kind'] == 'image' and row['metadata_file'] not in QUOTAS:
            reasons.append('outside_declared_photo_strata')
        derivative = None
        if row['kind'] == 'video':
            derivative = next((d for d in info.get('derivatives', [])
                               if d.get('transcodekey') == '240p.vp9.webm'), None)
            if (row['metadata_file'] not in VIDEO_POOLS or not 5 <= info.get('duration', 0) <= 90
                    or derivative is None or derivative['bandwidth'] * info['duration'] / 8 > 4 * 1024**2):
                reasons.append('outside_declared_short_camera_clip_scope')
        decisions.append({'page_id': row['page_id'], 'metadata_file': row['metadata_file'], 'reasons': reasons})
        if not reasons:
            eligible.append({**row, 'info': info, 'download_url': info['thumburl'] if row['kind'] == 'image' else derivative['src']})
    def rank(row):
        return hashlib.sha256(('chipmunk-natural-v2/' + str(row['page_id'])).encode()).hexdigest()

    selected, creators = [], Counter()
    # The scarcer plant stratum is selected first, without any detector response.
    for pool in ('photos-plants-metadata.json', 'photo-api-preflight.json',
                 'photos-architecture-metadata.json', 'photos-food-metadata.json'):
        chosen = []
        for row in sorted((r for r in eligible if r['kind'] == 'image' and r['metadata_file'] == pool), key=rank):
            if creators[row['creator']] >= 2:
                continue
            chosen.append(row)
            creators[row['creator']] += 1
            if len(chosen) == QUOTAS[pool]:
                break
        m.require(len(chosen) == QUOTAS[pool], 'Insufficient fixed photo stratum; acquire more metadata before inference')
        selected += chosen
    video_creators, clips = set(), []
    for row in sorted((r for r in eligible if r['kind'] == 'video'), key=rank):
        if row['creator'] in video_creators:
            continue
        clips.append(row)
        video_creators.add(row['creator'])
        if len(clips) == 8:
            break
    m.require(len(clips) == 8, 'Eight short camera clips from distinct credited creators required')
    for i, row in enumerate(selected):
        row['id'] = f'I{i:02d}'
    for i, row in enumerate(clips):
        row['id'] = f'V{i:02d}'
    selected += clips
    m.require(len({r['page_id'] for r in selected}) == 40, 'Forty distinct source pages required')
    trial.mkdir(exist_ok=False)
    m.write(trial / 'source-plan.json', {'schema': 'chipmunk.natural-visual-source-plan/2',
        'selection_code_sha256': m.digest(Path(__file__)), 'metadata_root': str(metadata),
        'metadata_audit_sha256': m.digest(audit_path), 'metadata_files': audit['metadata_files'],
        'photo_quotas': QUOTAS, 'prior_source_inventories': manifests, 'excluded_prior_urls': sorted(exclusions),
        'eligibility_decisions': decisions, 'sources': selected, 'neural_forwards': 0,
        'scope': 'Fresh to inventoried project URLs; pretraining overlap and independent capture truth unknown. No-added-mark controls are not proof of original authorship.'})
    return {'selected_images': 32, 'selected_clips': 8, 'distinct_image_creators': len(creators),
            'distinct_video_creators': len(video_creators), 'prior_matching_manifests': len(manifests), 'neural_forwards': 0}


def check(trial):
    plan_ = m.read(trial / 'source-plan.json')
    m.require(plan_['selection_code_sha256'] == m.digest(Path(__file__)), 'Source selection code changed')
    root = Path(plan_['metadata_root'])
    m.require(m.digest(root / 'metadata-eligibility-audit.json') == plan_['metadata_audit_sha256'], 'Metadata audit changed')
    for row in plan_['metadata_files']:
        m.require(m.digest(root / row['name']) == row['sha256'], 'Per-file source metadata changed')
    for row in plan_['prior_source_inventories']:
        m.require(m.digest(ROOT / row['path']) == row['sha256'], 'Prior source inventory changed')
    return plan_


def download(url, path, limit):
    m.require(urllib.parse.urlsplit(url).scheme == 'https' and urllib.parse.urlsplit(url).hostname
              in ('thumb.wikimedia.org', 'upload.wikimedia.org'), 'Only declared public Wikimedia media routes allowed')
    request = urllib.request.Request(url, headers={'User-Agent': 'ChipmunkVerifyResearch/1.0 (https://github.com/ChipmunkRPA/veritas)'})
    with urllib.request.urlopen(request, timeout=45) as response:
        m.require(response.status == 200, 'A full media response is required')
        m.require(urllib.parse.urlsplit(response.url).hostname in ('thumb.wikimedia.org', 'upload.wikimedia.org'), 'Unexpected media redirect')
        data = response.read(limit + 1)
    m.require(len(data) <= limit, 'Media download exceeds the prospective bound; no silent replacement')
    with path.open('xb') as stream:
        stream.write(data)


def fit(rgb):
    image = Image.fromarray(rgb)
    w, h = image.size
    side = min(w, h)
    crop = [(w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2]
    return np.asarray(ImageOps.fit(image, (256, 256), method=Image.Resampling.BILINEAR)).copy(), {
        'decoded_width': w, 'decoded_height': h, 'center_crop_in_decoded_source': crop,
        'working_width': 256, 'working_height': 256, 'scope': 'Coordinates refer to the downloaded thumbnail/transcode, not an unexamined full-resolution original.'}


def video(path):
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_streams', '-show_entries', 'frame=best_effort_timestamp:stream=time_base,width,height',
        '-of', 'json', str(path)], timeout=40))
    stream, frames = probe['streams'][0], probe['frames']
    tb = Fraction(stream['time_base'])
    times = [int(f['best_effort_timestamp']) * tb for f in frames]
    m.require(len(times) >= 8 and all(a < b for a, b in zip(times, times[1:])), 'Strict source PTS order required')
    indices = [min(range(len(times)), key=lambda j: abs(times[j] - times[0] - Fraction(i, 2))) for i in range(8)]
    m.require(len(set(indices)) == 8, 'Eight distinct sampled source frames required')
    selection = 'select=' + '+'.join(f'eq(n\\,{i})' for i in indices)
    data = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-an',
        '-vf', selection, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], timeout=40)
    w, h = stream['width'], stream['height']
    m.require(len(data) == 8 * w * h * 3, 'Exact source-frame selection required')
    rgb = np.frombuffer(data, np.uint8).reshape(8, h, w, 3)
    fitted = [fit(frame) for frame in rgb]
    anchors = [{'sample_index': i, 'source_frame_index': j, 'source_pts': int(frames[j]['best_effort_timestamp']),
                'source_time_base': stream['time_base'], 'source_time_s': float(times[j]), 'derived_time_s': i / 2}
               for i, j in enumerate(indices)]
    return np.stack([v[0] for v in fitted]), anchors, fitted[0][1]


def acquire(trial):
    plan_ = check(trial)
    output = trial / 'sources'
    output.mkdir(exist_ok=False)
    records = []
    for row in plan_['sources']:
        path = output / (row['id'] + ('.media-image' if row['kind'] == 'image' else '.webm'))
        download(row['download_url'], path, (3 if row['kind'] == 'image' else 8) * 1024**2)
        if row['kind'] == 'image':
            with Image.open(io.BytesIO(path.read_bytes())) as image:
                pixels, geometry = fit(np.asarray(image.convert('RGB')))
            anchors = None
        else:
            pixels, anchors, geometry = video(path)
        array = output / (row['id'] + '.npy')
        with array.open('xb') as stream:
            np.save(stream, pixels, allow_pickle=False)
        records.append({'id': row['id'], 'kind': row['kind'], 'page_id': row['page_id'],
            'asset': str(path), 'asset_sha256': m.digest(path), 'working_array': str(array),
            'working_array_sha256': m.digest(array), 'rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
            'shape': list(pixels.shape), 'anchors': anchors, 'geometry': geometry,
            'license': row['license'], 'license_url': row['license_url'], 'creator': row['creator']})
        print(json.dumps({'acquired': row['id'], 'completed_sources': len(records), 'neural_forwards': 0}), flush=True)
    m.write(trial / 'acquired-sources.json', {'source_plan_sha256': m.digest(trial / 'source-plan.json'), 'sources': records, 'neural_forwards': 0})
    sheet = Image.new('RGB', (8 * 160, 7 * 180), 'white')
    draw = ImageDraw.Draw(sheet)
    previews = []
    for row in records:
        a = np.load(row['working_array'], allow_pickle=False)
        previews += [(row['id'], a)] if a.ndim == 3 else [(f"{row['id']} source {i}", a[i]) for i in (0, 3, 7)]
    for i, (label, pixels) in enumerate(previews):
        x, y = i % 8 * 160, i // 8 * 180
        sheet.paste(Image.fromarray(pixels).resize((160, 160)), (x, y))
        draw.text((x + 3, y + 163), label, fill='black')
    sheet.save(trial / 'source-review.png')
    return {'acquired_sources': len(records), 'review_previews': len(previews), 'neural_forwards': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'acquire'))
    p.add_argument('trial', type=Path)
    p.add_argument('--metadata', type=Path)
    a = p.parse_args()
    print(json.dumps(plan(ROOT, a.metadata, a.trial) if a.action == 'plan' else acquire(a.trial), indent=2))
