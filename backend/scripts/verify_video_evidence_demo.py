"""Verify a completed local video analysis against its original and API assets.

Run from backend: .venv/bin/python scripts/verify_video_evidence_demo.py ANALYSIS_ID
This verifies engineering correspondence on a named fixture, not detection accuracy.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.canonical import canonical_hash
from app.detectors.video.decode import FrameTable, decode_frames
from app.evidence.mp4 import inspect_mp4, validate_clip_binding
from app.evidence.video_packets import packet_hash
from app.storage.artefacts import get_artefact
from app.storage.db import Analysis, AnalysisStage, get_session
from app.storage.objects import default_store


def main(aid: str) -> None:
    output = Path('data/video-evidence-demo') / aid
    output.mkdir(parents=True, exist_ok=True)
    store = default_store()
    with httpx.Client(base_url='http://127.0.0.1:8160', timeout=90) as client, get_session() as db:
        analysis = db.get(Analysis, aid)
        assert analysis is not None and analysis.kind == 'video' and analysis.status == 'done'
        store.verify(analysis.sha256)
        probe = db.get(AnalysisStage, (aid, 'probe'))
        row = get_artefact(db, probe.output_artefact_id, analysis_id=aid)
        record = json.loads(store.read_bytes(row.sha256))
        table = FrameTable.from_json(record['frame_table'])
        response = client.get(f'/api/v2/analyses/{aid}/evidence-packets')
        response.raise_for_status()
        evidence = response.json()
        assert evidence['status'] == 'ready' and evidence['version'] == 'video-evidence-2'
        assert evidence['sha256'] == packet_hash(evidence)
        assert evidence['original_result_hash'] == analysis.result_hash
        assert evidence['source']['frame_table_sha256'] == canonical_hash(record['frame_table'])
        exported = client.get(f'/api/analyses/{aid}/export.json')
        exported.raise_for_status()
        assert exported.json()['evidence'] == evidence
        (output / 'evidence.json').write_bytes(response.content)
        (output / 'export.json').write_bytes(exported.content)
        wanted = sorted({f['anchor']['frame_index'] for p in evidence['packets'] for f in p['frames']})
        originals = {f.index: f.rgb for f in decode_frames(store.path_for(analysis.sha256), table, wanted)}
        assets, frame_checks, clip_checks = {}, [], []
        for packet in evidence['packets']:
            validate_clip_binding(packet, evidence['source'])
            for role, asset in packet['artifacts'].items():
                if asset['sha256'] in assets:
                    continue
                response = client.get(asset['url'])
                response.raise_for_status()
                data = response.content
                assert hashlib.sha256(data).hexdigest() == asset['sha256']
                extension = 'mp4' if asset['media_type'] == 'video/mp4' else 'png'
                path = output / f"{packet['id']}-{role}.{extension}"
                path.write_bytes(data)
                assets[asset['sha256']] = (data, path)
                if extension == 'mp4':
                    metadata = inspect_mp4(data)
                    assert all(canonical_hash(value) == canonical_hash(asset.get(key)) for key, value in metadata.items())
                    ranged = client.get(asset['url'], headers={'Range': 'bytes=0-127'})
                    assert ranged.status_code == 206 and ranged.content == data[:128]
                    assert ranged.headers['content-range'] == f'bytes 0-127/{len(data)}'
                    ffprobe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'frame=pts:stream=codec_type',
                                              '-of', 'json', str(path)], capture_output=True, text=True, check=True, timeout=30)
                    probed = json.loads(ffprobe.stdout)
                    assert probed['streams'] == [{'codec_type': 'video'}]
                    assert [frame['pts'] for frame in probed['frames']] == metadata['pts']
            for frame in packet['frames']:
                anchor, refs = frame['anchor'], frame['artifact_refs']
                index = anchor['frame_index']
                image = Image.open(io.BytesIO(assets[packet['artifacts'][refs['original']]['sha256']][0])).convert('RGB')
                assert image.tobytes() == originals[index].tobytes()
                assert hashlib.sha256(image.tobytes()).hexdigest() == anchor['decoded_rgb_sha256']
                assert anchor['pts'] == table.pts[index]
                if refs.get('crop'):
                    region = frame['context_region']
                    crop = Image.open(io.BytesIO(assets[packet['artifacts'][refs['crop']]['sha256']][0])).convert('RGB')
                    expected = image.crop((region['x'], region['y'], region['x'] + region['w'], region['y'] + region['h']))
                    assert crop.tobytes() == expected.tobytes() and crop.size == expected.size
                frame_checks.append({'id': frame['id'], 'index': index, 'exact_source_pixels': True, 'exact_context_crop': bool(refs.get('crop'))})
            clip = packet.get('clip') or {}
            if clip.get('status') == 'ready':
                start, end = clip['start_frame_index'], clip['end_frame_index_exclusive']
                assert clip['source_pts'] == list(table.pts[start:end])
                clip_checks.append({'packet': packet['id'], 'sha256': packet['artifacts']['context_clip']['sha256'],
                                    'source_frame_range': [start, end], 'source_pts_match': True, 'range_download': True,
                                    'silent_reencoded': True, 'last_duration_basis': clip['last_frame_duration']['basis']})
            else:
                clip_checks.append({'packet': packet['id'], 'status': clip.get('status'), 'reason': clip.get('reason')})
        summary = {'analysis_id': aid, 'filename': analysis.filename, 'source_sha256': analysis.sha256,
                   'result_sha256': analysis.result_hash, 'evidence_sha256': evidence['sha256'],
                   'json_api_equal': True, 'asset_count': len(assets), 'packet_count': len(evidence['packets']),
                   'frames': frame_checks, 'clips': clip_checks, 'coverage': evidence['coverage'],
                   'scope': 'Engineering correspondence on this local fixture; no accuracy or pixel-localization claim.'}
        (output / 'verification.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps({key: summary[key] for key in ('analysis_id', 'evidence_sha256', 'asset_count', 'packet_count', 'clips')}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
