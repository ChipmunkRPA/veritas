"""Bounded, opt-in qualification; never an application/customer upload route."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from app.provenance import c2pa
from scripts import provider_provenance_v1 as method

ENDPOINTS = {'bria': 'https://engine.prod.bria-api.com/v2/image/verify',
             'suno': 'https://studio-api.prod.suno.com/api/c2pa/detect'}
MAX_INPUT, MAX_REPLY = 16 * 1024 * 1024, 65536
ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def png(pixels, label=False):
    output = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if label:
        info.add_text('AIGC', json.dumps({'Label': '1', 'ContentProducer': 'Bria'}))
    Image.fromarray(pixels).save(output, 'PNG', pnginfo=info)
    return output.getvalue()


def c2pa_chunks(data):
    chunks, offset = [], 8
    while offset + 12 <= len(data):
        size = int.from_bytes(data[offset:offset + 4], 'big')
        if offset + 12 + size > len(data):
            raise ValueError('Truncated source PNG')
        if data[offset + 4:offset + 8] == b'caBX':
            chunks.append(data[offset:offset + 12 + size])
        offset += 12 + size
    if not chunks:
        raise ValueError('Source fixture has no C2PA chunk')
    return b''.join(chunks)


def prepare(folder, source):
    folder.mkdir(parents=True, exist_ok=False)
    pixels = np.random.default_rng(20261107).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    authored = png(pixels)
    originals = [(source / f'bria-public-{i}.png').read_bytes() for i in range(2)]
    expected = ['403b9edfcc601993482ad04681652118e6e29e6984089b7c38cd3c5bf850b7e5',
                '1fb79b24aaa153a3d2b9c8c8a5afe6d8629dd305a16bb853f4cf5ed833d239da']
    if [sha(v) for v in originals] != expected:
        raise ValueError('The reviewed source fixtures changed')
    with Image.open(io.BytesIO(originals[0])) as image:
        original_pixels = np.array(image.convert('RGB'))
        webp = io.BytesIO()
        Image.fromarray(original_pixels).save(webp, 'WEBP', lossless=True)
    split = 8 + 12 + int.from_bytes(authored[8:12], 'big')
    copied = authored[:split] + c2pa_chunks(originals[0]) + authored[split:]
    np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(copied))), pixels)
    tone = io.BytesIO()
    with wave.open(tone, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((12000 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype('<i2').tobytes())
    base_audio = tone.getvalue()
    label = json.dumps({'Label': '1', 'ContentProducer': 'Suno'}).encode()
    chunk = b'AIGC' + len(label).to_bytes(4, 'little') + label + b'\0' * (len(label) % 2)
    labelled_audio = base_audio[:4] + (len(base_audio) + len(chunk) - 8).to_bytes(4, 'little') + base_audio[8:] + chunk
    cases = [
        ('authored', 'bria', 'image/png', authored, 'authored unmarked; expected no Bria match'),
        ('unsigned_label', 'bria', 'image/png', png(pixels, True), 'authored unsigned provider claim; expected no match'),
        ('provider_source_0', 'bria', 'image/png', originals[0], 'public provider showcase; current external result unknown'),
        ('provider_source_1', 'bria', 'image/png', originals[1], 'public provider showcase; current external result unknown'),
        ('stripped_0', 'bria', 'image/png', png(original_pixels), 'same source pixels, credentials removed'),
        ('transplanted', 'bria', 'image/png', copied, 'authored pixels with copied credential; expected no match'),
        ('webp_0', 'bria', 'image/webp', webp.getvalue(), 'provider pixels converted losslessly; no watermark check expected'),
        ('tone', 'suno', 'audio/wav', base_audio, 'authored unmarked; expected no Suno credentials'),
        ('tone_unsigned_label', 'suno', 'audio/wav', labelled_audio, 'authored unsigned claim; expected no credentials')]
    rows = []
    for case_id, provider, mime, data, scope in cases:
        path = folder / (case_id + '.media')
        path.write_bytes(data)
        rows.append({'id': case_id, 'provider': provider, 'mime': mime, 'path': path.name,
                     'sha256': sha(data), 'bytes': len(data), 'scope': scope})
    code = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in (
        Path(__file__), Path(method.__file__), ROOT / 'backend/app/provenance/c2pa.py')}
    plan = {'version': method.VERSION, 'cases': rows, 'code': code, 'max_requests': len(rows),
        'request_timeout_seconds': 25, 'max_response_bytes': MAX_REPLY, 'retries': 0,
        'stop_route_http': [401, 403, 404, 429], 'provider_endpoint_allowlist': ENDPOINTS,
        'admission': 'No serving or confidence admission. Provider response and explicit AI declaration/binding are separate.',
        'excluded_source': 'One public Suno playback object was not a decodable media file; do not treat it as an original download or bypass access controls.'}
    write_json(folder / 'plan.json', plan)
    print(json.dumps({'prepared': len(rows), 'plan_sha256': sha((folder / 'plan.json').read_bytes())}), flush=True)


def request_body(provider, mime, data):
    if provider == 'bria':
        return 'application/json', json.dumps({'image': base64.b64encode(data).decode()}, separators=(',', ':')).encode()
    boundary = 'chipmunk-qualification-' + sha(data)[:24]
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="authored.wav"\r\n'
            f'Content-Type: {mime}\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    return 'multipart/form-data; boundary=' + boundary, body


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def execute(folder, expected_plan_hash):
    plan_bytes = (folder / 'plan.json').read_bytes()
    if sha(plan_bytes) != expected_plan_hash:
        raise ValueError('Plan commitment does not match')
    plan = json.loads(plan_bytes)
    if (folder / 'requests.jsonl').exists():
        raise FileExistsError('This frozen qualification has already started; do not silently repeat it')
    for relative, digest in plan['code'].items():
        if sha((ROOT / relative).read_bytes()) != digest:
            raise ValueError('Frozen source changed: ' + relative)
    if plan['provider_endpoint_allowlist'] != ENDPOINTS or len(plan['cases']) != 9:
        raise ValueError('Unreviewed request plan')
    opener = urllib.request.build_opener(NoRedirect())
    blocked = set()
    start = time.monotonic()
    with (folder / 'requests.jsonl').open('x') as log:
        for case in plan['cases']:
            data = (folder / case['path']).read_bytes()
            if len(data) > MAX_INPUT or len(data) != case['bytes'] or sha(data) != case['sha256']:
                raise ValueError('Frozen input changed')
            provider = case['provider']
            local = c2pa.read(data, case['mime'], bundle=c2pa.load_bundle('c2pa'), thumbnails=False)
            content_type, body = request_body(provider, case['mime'], data)
            result = {'id': case['id'], 'provider': provider, 'source_sha256': sha(data), 'request_body_sha256': sha(body),
                      'http_status': None, 'response_sha256': None, 'response_bytes': 0, 'attempted': False, 'error': None,
                      'local': {'state': local.state, 'binding_status': local.binding_status,
                          'covers_bytes': local.covers_bytes, 'ai_declared': local.ai_declared,
                          'failure_codes': local.failure_codes}}
            response = b''
            before = time.monotonic()
            if provider in blocked:
                result['error'] = 'route_stopped_after_access_or_rate_limit_response'
            else:
                request = urllib.request.Request(ENDPOINTS[provider], data=body, method='POST', headers={
                    'Content-Type': content_type, 'Accept': 'application/json',
                    'User-Agent': 'ChipmunkVerify/limited-provenance-qualification'})
                result['attempted'] = True
                try:
                    try:
                        remote = opener.open(request, timeout=25)
                    except urllib.error.HTTPError as error:
                        remote = error
                    with remote:
                        result['http_status'] = remote.code
                        response = remote.read(MAX_REPLY + 1)
                        if len(response) > MAX_REPLY:
                            result['error'] = 'response_exceeds_bound'
                    if result['http_status'] in plan['stop_route_http']:
                        blocked.add(provider)
                except (OSError, TimeoutError, urllib.error.URLError) as error:
                    result['error'] = type(error).__name__
                (folder / (case['id'] + '.response')).write_bytes(response)
                result.update(response_sha256=sha(response), response_bytes=len(response))
            result['summary'] = method.summarize(provider, case['mime'], result['http_status'], response)
            if result['error']:
                result['summary'].update(status='unavailable', provider_reported_match=None)
            result['elapsed_seconds'] = time.monotonic() - before
            log.write(json.dumps(result, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps({'case': case['id'], 'http': result['http_status'], 'error': result['error'],
                              'summary': result['summary']}), flush=True)
            if time.monotonic() - start > 300:
                raise TimeoutError('Qualification deadline reached')
    write_json(folder / 'complete.json', {'version': method.VERSION, 'seconds': time.monotonic() - start,
        'cases_recorded': len(plan['cases']), 'request_attempts': sum(bool(json.loads(r)['attempted'])
            for r in (folder / 'requests.jsonl').read_text().splitlines()), 'model_forwards': 0,
        'customer_files_uploaded': 0, 'serving_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('folder', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--send-research-fixtures', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.source is None:
            parser.error('--source is required for preparation')
        prepare(args.folder, args.source)
    else:
        if not args.send_research_fixtures or not args.plan_sha256:
            parser.error('HTTP is disabled without --send-research-fixtures and --plan-sha256')
        execute(args.folder, args.plan_sha256)
