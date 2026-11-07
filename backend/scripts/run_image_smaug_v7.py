"""Freeze and run the current image pipeline versus a separate positive-only cue."""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import json
import mimetypes
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scripts import image_smaug_comparison_v7 as method

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ('backend/scripts/image_smaug_comparison_v7.py', 'backend/scripts/run_image_smaug_v7.py',
                'backend/scripts/review_image_smaug_v7.py',
                'backend/scripts/image_smaug_views_v7.py', 'backend/scripts/audit_image_smaug_runtime_v7.py',
                'backend/tests/test_image_smaug_comparison_v7.py', 'docs/image-smaug-v7-development-protocol.md')
PACKAGES = ('torch', 'transformers', 'tokenizers', 'numpy', 'Pillow', 'safetensors')


def runtime_files(main, candidate=False):
    site = main / ('tmp/image-smaug-v7-env/lib/python3.12/site-packages' if candidate else 'backend/.venv/lib/python3.12/site-packages')
    paths = set()
    directories = (('transformers/models/embedding_gemma2', 'transformers/models/gemma4',
                    'sentence_transformers/base', 'sentence_transformers/sentence_transformer') if candidate else
                   ('transformers/models/dinov2', 'transformers/models/vit', 'transformers/models/siglip'))
    for name in directories:
        paths.update((site / name).rglob('*.py'))
    for name in ('transformers/modeling_utils.py', 'transformers/processing_utils.py',
                 'transformers/image_utils.py', 'transformers/image_transforms.py',
                 'transformers/masking_utils.py', 'transformers/modeling_rope_utils.py'):
        paths.add(site / name)
    paths.update((site / 'tokenizers').glob('*.so'))
    if not candidate:
        paths.add(site / 'torch/nn/functional.py')
        paths.update((site / 'torch').glob('_C*.so'))
        paths.update((site / 'torch/lib').glob('libtorch_cpu*'))
        paths.update((site / 'cv2').rglob('*.so'))
        paths.update((site / 'PIL').glob('_imaging*.so'))
    return {str(p.relative_to(site)): method.digest(p) for p in sorted(paths)}


def sources():
    names = set(SOURCE_FILES)
    for p in (ROOT / 'backend/app').rglob('*'):
        if p.is_file() and p.suffix in {'.py', '.json', '.yaml', '.npy', '.j2', '.css'}:
            names.add(str(p.relative_to(ROOT)))
    return {name: method.digest(ROOT / name) for name in sorted(names)}


def configure(main, scratch):
    # Config is replaced before any other application import. Never print secrets.
    from app.core import config

    config.settings = config.Settings(_env_file=main / 'backend/.env', data_dir=scratch,
        database_url='sqlite:///' + str(scratch / 'state.db'), models_dir=main / 'models',
        hf_home=main / 'models/hf', offline=True)
    scratch.mkdir(parents=True, exist_ok=True)
    # Preserve the app's two-directory fuser search and their precedence.
    link = scratch / 'fusion'
    if not link.exists() and (main / 'backend/data/fusion').is_dir():
        link.symlink_to(main / 'backend/data/fusion', target_is_directory=True)
    return config.settings


def baseline_context(main, scratch):
    configure(main, scratch)
    from app.engine.build_info import config_sha256, sanitised_settings
    from app.engine.fetchers import ModelNotInstalled, ensure_verified
    from app.engine.fusion.registry import search_dirs, select_for_registry
    from app.engine.registry import get_registry

    registry = get_registry()
    detectors = registry.enabled('image', 'full')
    models = {}
    for mid in sorted({mid for did in detectors for mid in registry.required_models(registry.manifest.detector(did))}):
        model = registry.manifest.model(mid)
        if model.licence.class_ != 'commercial':
            raise ValueError('Enabled baseline weights lack commercial admission')
        try:
            files = {str(p.resolve().relative_to(main)): method.digest(p) for p in ensure_verified(model)}
            status = 'installed'
        except ModelNotInstalled:
            files, status = {}, 'not_installed'
        models[mid] = {'licence': model.licence.model_dump(mode='json'), 'status': status, 'files': files}
    selection = select_for_registry(registry, 'image', 'full')
    fusion = {}
    for directory in search_dirs():
        for path in sorted(directory.glob('fuser-image-*')):
            if path.is_file():
                fusion[str(path.resolve().relative_to(main)) if path.resolve().is_relative_to(main) else str(path.relative_to(ROOT))] = method.digest(path)
    with sqlite3.connect('file:' + str(main / 'backend/data/veritas.db') + '?mode=ro', uri=True) as db:
        row = db.execute('select id,version,thresholds,sha256 from policy_thresholds order by version desc limit 1').fetchone()
    return {'config_sha256': config_sha256(), 'settings': sanitised_settings(), 'profile': 'full',
        'detectors': detectors, 'models': models, 'fusion_files': fusion,
        'fusion_selection': {'mode': selection.mode, 'sha256': selection.sha256, 'version': selection.version, 'reason': selection.reason},
        'policy': {'id': row[0], 'version': row[1], 'thresholds': json.loads(row[2]), 'sha256': row[3]},
        'installed_implementation_sha256': runtime_files(main),
        'packages': {p: importlib.metadata.version(p) for p in PACKAGES}}


def freeze(main, trial):
    panel = json.loads((trial / 'panel.json').read_text())
    ctx = baseline_context(main, trial / 'freeze-context')
    for row in panel['items']:
        if method.digest(row['input_path']) != row['input_sha256']:
            raise ValueError('Prepared input changed')
    assets = json.loads((ROOT / 'docs/evaluation-panels/image-v7-smaug-acquisition.json').read_text())['files']
    assets += [json.loads((ROOT / 'docs/evaluation-panels/image-v7-smaug-processor-completion.json').read_text())['added_asset']]
    model = main / 'models/research/image-v7-smaug'
    for row in assets:
        if method.digest(model / row['name']) != row['sha256']:
            raise ValueError('Reviewed candidate assets changed')
    lock = {'schema': 'chipmunk.image-smaug-development-lock/7', 'created_at': datetime.now(UTC).isoformat(),
        'panel_sha256': method.digest(trial / 'panel.json'), 'implementation_sha256': sources(), 'baseline': ctx,
        'candidate_assets': {r['name']: r['sha256'] for r in assets},
        'candidate_packages': {'transformers': '5.22.2', 'sentence-transformers': '6.2.0', 'huggingface-hub': '1.32.0', 'tokenizers': '0.23.1'},
        'candidate_installed_implementation_sha256': runtime_files(main, candidate=True),
        'protocol_sha256': method.digest(ROOT / 'docs/image-smaug-v7-development-protocol.md'),
        'planned_counts': dict(Counter(r['role'] + ':' + r['format'] for r in panel['items'])),
        'model_response_count_before_freeze': 0, 'serving_changed': False}
    method.write(trial / 'lock.json', lock)
    print(json.dumps({'lock_sha256': method.digest(trial / 'lock.json'), 'planned': lock['planned_counts'],
                      'detectors': len(ctx['detectors']), 'fuser': ctx['fusion_selection']}), flush=True)


def load_lock(trial):
    lock = json.loads((trial / 'lock.json').read_text())
    if sources() != lock['implementation_sha256'] or method.digest(trial / 'panel.json') != lock['panel_sha256']:
        raise ValueError('Frozen source or prepared input packet changed')
    return lock, json.loads((trial / 'panel.json').read_text())


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def baseline(main, trial, *, synthetic=False):
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    scratch = trial / ('synthetic-baseline' if synthetic else 'baseline-scratch')
    if scratch.exists():
        raise ValueError('Refuse restart or overwrite; preserve original attempt')
    settings = configure(main, scratch)
    if synthetic:
        from PIL import Image

        source = trial / 'synthetic-authored.png'
        Image.new('RGB', (640, 640), (102, 153, 204)).save(source)
        items = [{'id': 'authored-runtime-only', 'input_path': str(source), 'input_sha256': method.digest(source)}]
        frozen = baseline_context(main, scratch)
    else:
        lock, panel = load_lock(trial)
        frozen = baseline_context(main, scratch)
        if frozen != lock['baseline']:
            raise ValueError('Actual baseline settings, weights, fuser, packages or policy differ')
        items = panel['items']
    from app.engine.build_info import config_sha256
    from app.engine.customer_assessment import customer_assessment
    from app.engine.models import get_model_manager
    from app.engine.pipelines import run_analysis
    from app.engine.registry import get_registry
    from app.policy.thresholds import Thresholds
    from app.policy.thresholds import digest as policy_digest
    from app.storage.db import Analysis, PolicyThresholdsRow, get_session, init_db
    from app.storage.objects import ObjectStore

    init_db()
    policy = frozen['policy']
    if policy_digest(Thresholds(**policy['thresholds'])) != policy['sha256']:
        raise ValueError('Policy digest does not match')
    with get_session() as db:
        db.add(PolicyThresholdsRow(id=policy['id'], version=policy['version'], thresholds=policy['thresholds'],
            sha256=policy['sha256'], reason='Frozen research comparison', created_by='system', created_at=datetime.now(UTC)))
        db.commit()
    registry, models = get_registry(), get_model_manager()
    result_dir = trial / ('synthetic-baseline-results' if synthetic else 'baseline-results')
    result_dir.mkdir()
    journal = trial / ('synthetic-baseline.jsonl' if synthetic else 'baseline.jsonl')
    for index, item in enumerate(items):
        started = time.monotonic()
        source = Path(item['input_path'])
        if method.digest(source) != item['input_sha256'] or config_sha256() != frozen['config_sha256']:
            raise ValueError('Frozen input or scoring settings changed')
        aid = str(uuid.uuid4())
        owned = scratch / 'cases' / aid
        store = ObjectStore(owned / 'objects')
        name = item['input_sha256'][:16] + source.suffix
        upload = settings.uploads_dir / aid
        upload.mkdir(parents=True)
        shutil.copyfile(source, upload / name)
        with get_session() as db:
            db.add(Analysis(id=aid, kind='image', filename=name, size_bytes=source.stat().st_size,
                content_type=mimetypes.guess_type(name)[0] or 'application/octet-stream', sha256=item['input_sha256'], status='queued'))
            db.commit()
        row = {'id': item['id'], 'analysis_id': aid, 'input_sha256': item['input_sha256']}
        try:
            outcome = run_analysis(aid, profile='full', registry=registry, models=models, store=store,
                                   policy_thresholds_id=policy['id'])
            with get_session() as db:
                analysis = db.get(Analysis, aid)
                result_hash = analysis.result_hash
            if outcome.status != 'done' or not result_hash:
                raise RuntimeError(f'Pipeline {outcome.status}: {outcome.error}')
            raw = store.read_bytes(result_hash)
            if __import__('hashlib').sha256(raw).hexdigest() != result_hash:
                raise ValueError('Canonical result bytes fail integrity check')
            result = json.loads(raw)
            assessment = customer_assessment({**result, 'status': 'done', 'kind': 'image'})
            packet = {'result': result, 'outputs': outcome.outputs,
                      'object_manifest': [{'sha256': p.name, 'bytes': p.stat().st_size}
                          for p in sorted((owned / 'objects/sha256').rglob('*')) if p.is_file()]}
            target = result_dir / (item['id'] + '.json')
            method.write(target, packet)
            if json.loads(target.read_text()) != packet:
                raise ValueError('Retained full stage/result packet does not round-trip')
            row.update(status='done', interpretation=method.direction(result, assessment),
                canonical_result_sha256=result_hash, retained_packet_sha256=method.digest(target),
                checks=[{'stage': stage, 'id': e['detector_id'], 'status': e['status'], 'reason': e.get('reason')}
                    for stage in ('provenance', 'faces', 'detect') for e in outcome.outputs.get(stage, {}).get('detectors', [])],
                assets_retained=False, asset_digest_manifest_retained=True)
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error), scratch_retained=True)
            traceback.print_exc()
        row['seconds'] = time.monotonic() - started
        append(journal, row)
        if row['status'] == 'done':
            for path in (owned, upload, settings.artifacts_dir / aid):
                if path.is_relative_to(scratch) and path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(index + 1, '/', len(items), item['id'], row['status'], round(row['seconds'], 2), flush=True)
    if not synthetic:
        load_lock(trial)
    print('baseline completed', flush=True)


def candidate(main, trial, role):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    from scripts.audit_image_smaug_runtime_v7 import check_tensor_file
    from scripts.image_smaug_views_v7 import views

    lock, panel = load_lock(trial)
    if runtime_files(main, candidate=True) != lock['candidate_installed_implementation_sha256']:
        raise ValueError('Pinned candidate installed implementation changed')
    for name, expected in lock['candidate_packages'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Candidate research package version changed')
    if role in {'development', 'audit'}:
        gate = json.loads((trial / 'gate.json').read_text())
        cal = [json.loads(line) for line in (trial / 'candidate-calibration.jsonl').read_text().splitlines()]
        if gate != method.select_gate(panel, cal):
            raise ValueError('Genuine-only gate changed or calibration incomplete')
    output = trial / ('candidate-' + role + '.jsonl')
    if output.exists():
        raise ValueError('Refuse overwrite or restart of scored attempt')
    model_dir = main / 'models/research/image-v7-smaug'
    if {name: method.digest(model_dir / name) for name in lock['candidate_assets']} != lock['candidate_assets']:
        raise ValueError('Pinned candidate assets changed')
    model = SentenceTransformer(str(model_dir), device='cpu', local_files_only=True, trust_remote_code=False,
        config_kwargs={'audio_config': None}, model_kwargs={'torch_dtype': torch.float32, 'attn_implementation': 'eager'})
    model.requires_grad_(False).eval()
    check_tensor_file(model[0].auto_model, model_dir / 'model.safetensors', 438760472)
    check_tensor_file(model[3], model_dir / '3_Dense/model.safetensors', 769)
    processor = model[0].processor.image_processor
    processor.max_soft_tokens = processor.image_seq_length = 140
    reference = {}
    if role == 'audit':
        for name in ('calibration', 'development'):
            reference.update({r['id']: r for r in map(json.loads, (trial / ('candidate-' + name + '.jsonl')).read_text().splitlines())})
        probes = {}
        for item in sorted(panel['items'], key=lambda x: method.order(x['id'])):
            probes.setdefault(tuple(item[k] for k in ('family', 'label', 'format', 'role')), item)
        items = list(probes.values())
    else:
        items = [x for x in panel['items'] if x['role'] == role]
    for index, item in enumerate(items):
        started = time.monotonic()
        row = {'id': item['id'], 'input_sha256': item['input_sha256']}
        try:
            if method.digest(item['input_path']) != item['input_sha256']:
                raise ValueError('Input bytes changed')
            with Image.open(item['input_path']) as image:
                crops, traces = views(image, 3)
            with torch.inference_mode():
                logits = model.encode([{'image': c} for c in crops], batch_size=1 if role == 'audit' else 3,
                    convert_to_tensor=True, show_progress_bar=False).float().reshape(-1)
            if logits.shape != (3,) or not torch.isfinite(logits).all():
                raise ValueError('Expected three finite logits')
            mean = float(logits.mean())
            raw = logits.cpu().tolist()
            independent = float(np.asarray(raw, np.float64).mean())
            row.update(status='ok', crop_logits=raw, mean_logit=mean, independent_mean_logit=independent,
                raw_sigmoid_response=float(torch.sigmoid(logits.mean())),
                crop_rgb_sha256=[__import__('hashlib').sha256(c.tobytes()).hexdigest() for c in crops], traces=traces)
            if abs(mean - independent) > 1e-4:
                raise ValueError('Mean-logit reduction exceeded fixed native tolerance')
            if role == 'audit':
                original = reference[item['id']]
                delta = max(abs(a - b) for a, b in zip(raw, original['crop_logits'], strict=True))
                probability_delta = abs(row['raw_sigmoid_response'] - original['raw_sigmoid_response'])
                if (original['status'] != 'ok' or delta > 1e-4 or probability_delta > 1e-5
                        or row['crop_rgb_sha256'] != original['crop_rgb_sha256'] or traces != original['traces']):
                    raise ValueError('Source/crop/coordinate or native single-batch replay failed')
                row.update(native_replay_passed=True, max_logit_difference=delta, sigmoid_difference=probability_delta)
        except Exception as error:
            row = {'id': item['id'], 'input_sha256': item['input_sha256'], 'status': 'failed',
                   'error_type': type(error).__name__, 'error': str(error)}
            traceback.print_exc()
        row['seconds'] = time.monotonic() - started
        append(output, row)
        print(index + 1, '/', len(items), item['id'], row['status'], round(row['seconds'], 2), flush=True)
    load_lock(trial)
    if role == 'calibration':
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        method.write(trial / 'gate.json', method.select_gate(panel, rows))
    print('candidate', role, 'completed', flush=True)


def profile(main, trial, arm):
    import psutil

    folder = trial / ('profile-' + arm)
    modules = {'scripts.run_image_smaug_v7', 'scripts.run_audio_separation_v14',
        'scripts.repeat_audio_separation_v14', 'scripts.run_writing_modernbert_v4',
        'scripts.run_video_demamba_v13', 'scripts.audit_image_smaug_runtime_v7'}
    for process in psutil.process_iter(['pid', 'cmdline']):
        cmd = process.info['cmdline'] or []
        if process.pid != os.getpid() and any(m in cmd for m in modules):
            raise ValueError('Heavy model jobs must run sequentially')
    folder.mkdir(exist_ok=False)
    child_args = [sys.executable, '-m', 'scripts.run_image_smaug_v7', arm, str(main), str(trial)]
    begin, max_rss, guard, samples, sample_errors = time.monotonic(), 0, None, 0, 0
    with (folder / 'worker.log').open('x') as log:
        child = subprocess.Popen(child_args, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                tree = [process, *process.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in tree if p.is_running())
                max_rss = max(max_rss, rss)
                samples += 1
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                sample_errors += 1
                rss = 0
            limit = (12 if arm in {'baseline', 'synthetic-baseline'} else 6) * 1024**3
            if (rss > limit or psutil.virtual_memory().available < 1536 * 1024**2
                    or shutil.disk_usage(trial).free < 1536 * 1024**2 or time.monotonic() - begin > 6 * 3600):
                guard = 'memory_disk_or_six_hour_timeout'
                for p in reversed(tree):
                    with contextlib.suppress(psutil.NoSuchProcess):
                        p.terminate()
                break
            time.sleep(.5)
        try:
            code = child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            code = child.wait()
    record = {'arm': arm, 'child_exit_code': code, 'guard': guard,
        'sampled_tree_rss_lower_bound_bytes': max_rss, 'samples': samples, 'sample_errors': sample_errors,
        'duration_s': time.monotonic() - begin, 'pid': child.pid, 'created_at': datetime.now(UTC).isoformat()}
    method.write(folder / 'profile.json', record)
    print(json.dumps(record), flush=True)
    if code or guard:
        raise RuntimeError('Incomplete original attempt retained; no automatic retry')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'freeze', 'synthetic-baseline', 'baseline',
        'candidate-calibration', 'candidate-development', 'candidate-audit', 'profile'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--arm', choices=('synthetic-baseline', 'baseline', 'candidate-calibration', 'candidate-development', 'candidate-audit'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        method.prepare(args.main, args.trial)
    elif args.mode == 'freeze':
        freeze(args.main, args.trial)
    elif args.mode == 'profile':
        if args.arm is None:
            parser.error('--arm required')
        profile(args.main, args.trial, args.arm)
    elif args.mode == 'baseline' or args.mode == 'synthetic-baseline':
        baseline(args.main, args.trial, synthetic=args.mode == 'synthetic-baseline')
    else:
        candidate(args.main, args.trial, args.mode.removeprefix('candidate-'))
