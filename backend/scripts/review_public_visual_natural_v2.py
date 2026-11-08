"""Model-free replay of natural source pixels, exact clocks and watermark evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import review_public_visual_watermark_v1 as readout
from scripts import run_public_visual_natural_v2 as frozen

m = frozen.m


def decode(path):
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_streams', '-show_entries', 'frame=best_effort_timestamp:stream=time_base,width,height',
        '-of', 'json', str(path)], timeout=40))
    data = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-an',
        '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], timeout=40)
    stream = probe['streams'][0]
    count, h, w = len(probe['frames']), stream['height'], stream['width']
    m.require(len(data) == count * h * w * 3, 'Exact decoded frame count/geometry required')
    return np.frombuffer(data, np.uint8).reshape(count, h, w, 3), probe


def source_replay(source):
    path = Path(source['asset'])
    geometry = source['geometry']
    box = geometry['center_crop_in_decoded_source']
    if source['kind'] == 'image':
        with Image.open(path) as im:
            image = im.convert('RGB')
            m.require(image.size == (geometry['decoded_width'], geometry['decoded_height']), 'Source image geometry differs')
            return np.asarray(image.resize((256, 256), Image.Resampling.BILINEAR, box=tuple(box))).copy()
    # Independently read all source timestamps and decode only the recorded indices.
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_streams', '-show_entries', 'frame=best_effort_timestamp:stream=time_base,width,height',
        '-of', 'json', str(path)], timeout=40))
    stream, frames = probe['streams'][0], probe['frames']
    times = [int(f['best_effort_timestamp']) * Fraction(stream['time_base']) for f in frames]
    indices = []
    for i, anchor in enumerate(source['anchors']):
        j = min(range(len(times)), key=lambda n: abs(times[n] - times[0] - Fraction(i, 2)))
        m.require(anchor == {'sample_index': i, 'source_frame_index': j,
            'source_pts': int(frames[j]['best_effort_timestamp']), 'source_time_base': stream['time_base'],
            'source_time_s': float(times[j]), 'derived_time_s': i / 2}, 'Original source sampling/PTS mapping differs')
        indices.append(j)
    selection = 'select=' + '+'.join(f'eq(n\\,{j})' for j in indices)
    data = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-an',
        '-vf', selection, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], timeout=40)
    h, w = stream['height'], stream['width']
    m.require(w == geometry['decoded_width'] and h == geometry['decoded_height'] and len(data) == 8 * w * h * 3,
              'Source clip decode geometry differs')
    originals = np.frombuffer(data, np.uint8).reshape(8, h, w, 3)
    return np.stack([np.asarray(Image.fromarray(f).resize((256, 256), Image.Resampling.BILINEAR,
                     box=tuple(box))) for f in originals])


def audit(trial, output):
    _, sources = frozen.check(trial)
    profile = m.read(trial / 'profile/profile.json')
    m.require(profile['child_exit_code'] == 0 and profile['guard'] is None and profile['sample_errors'] == 0,
              'Complete unguarded native study required')
    result = m.read(trial / 'result.json')
    m.require(result['freeze_sha256'] == m.digest(trial / 'freeze.json') and result['case_count'] == 640
        and result['generator_forwards'] == 64 and result['detector_forwards'] == 645,
        'Original prospective forward denominators required')
    journal = trial / 'case-journal-private.jsonl'
    m.require(result['journal_sha256'] == m.digest(journal), 'Native case journal changed')
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    expected = {f"{s['id']}-{state}-{transform}" for s in sources if s['kind'] == 'image'
                for state in ('unmarked', 'marked', 'partial', 'ordinary_edit') for transform in ('native', 'jpeg70')}
    expected |= {f"{s['id']}-{state}-{container}-{i}" for s in sources if s['kind'] == 'video'
                 for state in ('unmarked', 'partial', 'ordinary_edit') for container in ('raw', 'h264') for i in range(8)}
    m.require(len(rows) == len({r['id'] for r in rows}) == len(expected) == 640
        and {r['id'] for r in rows} == expected, 'Every fixed source/state/transform case required')
    output.mkdir(exist_ok=False)
    originals = {}
    for source in sources:
        pixels = source_replay(source)
        m.require(np.array_equal(pixels, np.load(source['working_array'], allow_pickle=False)), 'Independent source crop/clock replay differs')
        originals[source['id']] = pixels
    source_by_id = {r['id']: r for r in sources}
    bases, videos = {}, {}
    presence = bits_total = 0
    for row in rows:
        name = row['carrier']
        m.require(row['id'].startswith(name + '-'), 'Case/source carrier identity differs')
        payload = frozen.PAYLOADS[int(name[1:]) % 4]
        m.require(row['payload'] == payload, 'Predetermined source payload changed')
        base_path = Path(row['base_array'])
        m.require(m.digest(base_path) == row['base_sha256'], 'Base images/frames changed')
        if name not in bases:
            with np.load(base_path, allow_pickle=False) as data:
                base = {k: data[k] for k in data.files}
            m.require(np.array_equal(base['unmarked'], originals[name]), 'Retained base/source RGB differs')
            u, partial, ordinary = base['unmarked'], base['partial'], base['ordinary_edit']
            if u.ndim == 3:
                mask = np.zeros((256, 256), bool)
                mask[64:192, 64:192] = True
                m.require(np.array_equal(partial[~mask], u[~mask])
                    and np.array_equal(partial[mask], base['marked'][mask]), 'Partial mark source region changed')
                rotated = u.copy()
                rotated[64:192, 64:192] = np.rot90(u[64:192, 64:192])
            else:
                mask = np.zeros((8, 256, 256), bool)
                mask[2:6, 64:192, 64:192] = True
                m.require(np.array_equal(partial[~mask], u[~mask]), 'Partial clip changed outside intended sampled time/region')
                rotated = u.copy()
                for i in range(2, 6):
                    m.require(np.array_equal(partial[i, 64:192, 64:192], base['full_marked_frames'][i - 2, 64:192, 64:192]), 'Marked clip crop differs')
                    rotated[i, 64:192, 64:192] = np.rot90(u[i, 64:192, 64:192])
            m.require(np.array_equal(rotated, ordinary), 'Ordinary-edit control recipe changed')
            bases[name] = base
        base = bases[name]
        path = Path(row['private_array'])
        m.require(m.digest(path) == row['array_sha256'], 'Native evidence array changed')
        with np.load(path, allow_pickle=False) as data:
            rgb, planes, packed = data['pixels'], data['presence_byteplanes'], data['binary_bit_maps']
        m.require(planes.shape == (4, 65536) and planes.dtype == np.uint8
            and packed.shape == (32, 8192) and packed.dtype == np.uint8
            and rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8, 'Native storage geometry/type changed')
        logits = planes.T.copy().view('<f4').reshape(256, 256)
        m.require(np.isfinite(logits).all()
            and hashlib.sha256(logits.tobytes()).hexdigest() == row['native_presence_f32_sha256']
            and hashlib.sha256(rgb.tobytes()).hexdigest() == row['source_rgb_sha256'], 'Lossless native float/RGB identity differs')
        if row['kind'] == 'image':
            expected_rgb = frozen.authored.transform(base[row['state']], row['transform'])
        else:
            i = row['anchor']['sample_index']
            m.require(row['anchor'] == source_by_id[name]['anchors'][i], 'Original source clock lost')
            m.require(row['derived_anchor']['frame_index'] == i and row['derived_anchor']['source_time_s'] == i / 2
                and row['derived_anchor']['source_pts'] * Fraction(row['derived_anchor']['time_base']) == Fraction(i, 2),
                'Raw/encoded derived clock differs')
            state = row['clip_state'] if row['clip_state'] != 'unmarked' and 2 <= i < 6 else 'unmarked'
            m.require(state == row['state'], 'Sampled interval truth changed')
            if row['container'] == 'raw':
                expected_rgb = base[row['clip_state']][i]
            else:
                video = Path(row['video_path'])
                m.require(m.digest(video) == row['video_sha256'], 'Encoded evidence clip changed')
                if str(video) not in videos:
                    actual, probe = decode(video)
                    # Re-encode from retained pre-codec inputs, independently confirming the input-to-codec link.
                    replay_path = output / (name + '-' + row['clip_state'] + '.mp4')
                    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', '256x256', '-framerate', '2', '-i', 'pipe:0', '-c:v', 'libx264', '-crf', '23',
                        '-pix_fmt', 'yuv420p', '-threads', '1', str(replay_path)],
                        input=base[row['clip_state']].tobytes(), check=True, timeout=40)
                    replay, replay_probe = decode(replay_path)
                    m.require(actual.shape == (8, 256, 256, 3) and np.array_equal(actual, replay)
                        and probe == replay_probe, 'Retained inputs do not reproduce decoded H264 evidence')
                    videos[str(video)] = (actual, probe)
                values, probe = videos[str(video)]
                tb = probe['streams'][0]['time_base']
                pts = int(probe['frames'][i]['best_effort_timestamp'])
                m.require(row['derived_anchor'] == {'frame_index': i, 'source_pts': pts, 'time_base': tb,
                    'source_time_s': i / 2} and pts * Fraction(tb) == Fraction(i, 2), 'Derived clip clock differs')
                expected_rgb = values[i]
        m.require(np.array_equal(rgb, expected_rgb), 'Transformed input pixels differ from source/base/codec')
        mask, regions, decoded = readout.independent(logits, packed)
        e = row['evidence']
        def key(r):
            return (r['y'], r['x'], r['w'], r['h'], r['pixels'])

        m.require(sorted(regions, key=key) == sorted(e['regions'], key=key)
            and bool(regions) == e['flagged_for_inspection'] and int(mask.sum()) == e['selected_pixels']
            and float(mask.mean()) == e['selected_fraction'] and decoded == e['decoded_message']
            and e['decoded_bits_correct_of32'] == (None if decoded is None else sum(a == b for a, b in zip(decoded, payload, strict=True))),
            'Independent region/message replay differs')
        m.require(e['provider_authenticated'] is e['used_in_confidence'] is e['early_stop_eligible'] is False,
                  'Unsupported provider/AI/confidence claim acquired')
        truth = np.zeros((256, 256), bool)
        if row['state'] == 'marked':
            truth[:] = True
        elif row['state'] == 'partial':
            truth[64:192, 64:192] = True
        inter, union = int((mask & truth).sum()), int((mask | truth).sum())
        overlap = {'iou': inter / union if truth.any() else None,
            'coverage': inter / int(truth.sum()) if truth.any() else None,
            'pixels_selected_outside': int(mask.sum()) - inter}
        m.require(overlap == row['overlap'], 'Spatial truth/overlap differs')
        presence += 65536
        bits_total += 32 * 65536
    m.require(frozen.aggregate(rows) == result['groups'], 'Complete group totals differ')
    m.require({str(i): frozen.aggregate([r for r in rows if r['payload'] == p]) for i, p in enumerate(frozen.PAYLOADS)}
        == result['by_payload'], 'Payload group totals differ')
    m.require({k: frozen.aggregate([r for r in rows if r['kind'] == 'image' and r['source_stratum'] == k])
        for k in frozen.sources.QUOTAS} == result['by_photo_stratum'], 'Photo stratum totals differ')
    receipt = {'schema': 'chipmunk.natural-visual-source-audit/2', 'cases': len(rows), 'source_assets_redecoded': len(sources),
        'source_sampled_frames_replayed': 64, 'presence_pixels_replayed': presence,
        'binary_bit_predictions_replayed': bits_total, 'h264_clips_reencoded_and_redecoded': len(videos),
        'original_and_derived_source_clocks_verified': True, 'lossless_f32_replay_verified': True,
        'neural_forwards': 0, 'provider_requests': 0, 'outside_human_reviewers': 0,
        'serving_admitted': False, 'used_in_confidence': False}
    m.write(output / 'audit.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trial', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    audit(a.trial, a.output)
