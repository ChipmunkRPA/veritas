"""Bounded, synthetic-only image compatibility test; no project-media scoring."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
ASSET_RECEIPT = ROOT / 'docs/evaluation-panels/image-v7-smaug-acquisition.json'
PROCESSOR_RECEIPT = ROOT / 'docs/evaluation-panels/image-v7-smaug-processor-completion.json'
ASSET_SHA = 'f8ef4c285ed4d344bce32e03fcdbd5127d68c7566fa53ca086780569723ff17b'
PROCESSOR_SHA = '636fa5240a41e5f18f783ba30e3f49b1af737b5d4716ac5fc37eba63b99ceeed'

def write(output, name, value):
    with (output / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_tensor_file(module, path, expected_values):
    import torch
    from safetensors import safe_open

    state = module.state_dict()
    with safe_open(path, framework='pt', device='cpu') as source:
        if set(state) != set(source.keys()):
            raise ValueError('Complete native state names differ from published tensors')
        values = 0
        for name, actual in state.items():
            stored = source.get_tensor(name)
            if (actual.device.type != 'cpu' or actual.dtype != torch.float32
                    or actual.shape != stored.shape or not torch.equal(actual, stored.float())):
                raise ValueError('Native CPUfloat32 state differs from exact released tensor')
            values += actual.numel()
        if values != expected_values:
            raise ValueError('Complete parameter/value count differs')
    return {'tensors': len(state), 'values': values, 'all_released_values_exact': True}


def worker(model_dir, output):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    from scripts.image_smaug_views_v7 import views

    if (hashlib.sha256(ASSET_RECEIPT.read_bytes()).hexdigest() != ASSET_SHA
            or hashlib.sha256(PROCESSOR_RECEIPT.read_bytes()).hexdigest() != PROCESSOR_SHA):
        raise ValueError('Exact reviewed asset and processor metadata required')
    acquired = json.loads(ASSET_RECEIPT.read_text())
    for item in acquired['files']:
        path = model_dir / item['name']
        if hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != item['sha256']:
            raise ValueError('Acquired source/model assets changed')
    extra = json.loads(PROCESSOR_RECEIPT.read_text())['added_asset']
    if hashlib.file_digest((model_dir / extra['name']).open('rb'), 'sha256').hexdigest() != extra['sha256']:
        raise ValueError('Complete pinned native processor template required')
    model = SentenceTransformer(str(model_dir), device='cpu', local_files_only=True,
        trust_remote_code=False, config_kwargs={'audio_config': None},
        model_kwargs={'torch_dtype': torch.float32, 'attn_implementation': 'eager'})
    model.requires_grad_(False).eval()
    if len(model) != 4 or model[0].config.model_type != 'embedding_gemma2' or model[0].config.audio_config is not None:
        raise ValueError('Exact image-only standard pipeline required')
    base = check_tensor_file(model[0].auto_model, model_dir / 'model.safetensors', 438760472)
    head = check_tensor_file(model[3], model_dir / '3_Dense/model.safetensors', 769)
    processor = model[0].processor.image_processor
    processor.max_soft_tokens = processor.image_seq_length = 140
    images = []
    for width, height in ((640, 640), (700, 400), (529, 711)):
        y, x = np.indices((height, width), dtype=np.uint16)
        pixels = np.stack([(x * 7 + y * 11) % 256, (x * 3 + y * 17) % 256,
                           (x * 23 + y * 5) % 256], axis=-1).astype(np.uint8)
        images.append(Image.fromarray(pixels))
    crop_sets = [views(image, 3) for image in images]
    flat = [{'image': crop} for crops, _ in crop_sets for crop in crops]
    with torch.inference_mode():
        a = model.encode(flat, batch_size=3, convert_to_tensor=True, show_progress_bar=False).float()
        b = model.encode(flat, batch_size=3, convert_to_tensor=True, show_progress_bar=False).float()
        single = model.encode(flat, batch_size=1, convert_to_tensor=True, show_progress_bar=False).float()
    if a.shape != (9, 1) or not torch.isfinite(a).all() or not torch.equal(a, b):
        raise ValueError('Complete finite exact repeated synthetic outputs required')
    max_logit_difference = float(torch.max(torch.abs(a - single)))
    probabilities = torch.sigmoid(a.reshape(3, 3).mean(1)).numpy()
    single_probabilities = torch.sigmoid(single.reshape(3, 3).mean(1)).numpy()
    raw = a.numpy().astype(np.float64).reshape(3, 3)
    manual = np.exp(-np.logaddexp(0, -raw.mean(1)))
    direct_difference = float(np.max(np.abs(probabilities - manual)))
    probability_difference = float(np.max(np.abs(probabilities - single_probabilities)))
    result = {'schema': 'chipmunk.image-smaug-synthetic-runtime/7', 'base_state': base, 'head_state': head,
        'packages': {name: importlib.metadata.version(name) for name in
                     ['torch', 'transformers', 'sentence-transformers', 'huggingface-hub', 'tokenizers', 'numpy', 'Pillow']},
        'input_rgb_sha256': [hashlib.sha256(i.tobytes()).hexdigest() for i in images],
        'crop_rgb_sha256': [[hashlib.sha256(c.tobytes()).hexdigest() for c in crops] for crops, _ in crop_sets],
        'crop_traces': [traces for _, traces in crop_sets], 'views': 9, 'repeat_exact': True,
        'max_single_batch_logit_difference': max_logit_difference,
        'max_single_batch_probability_difference': probability_difference,
        'max_independent_mean_logit_sigmoid_difference': direct_difference,
        'passed': max_logit_difference <= 1e-4 and max(probability_difference, direct_difference) <= 1e-5,
        'fixed_tolerances': {'single_batch_logits': 1e-4, 'single_batch_probabilities_and_manual_reduction': 1e-5},
        'project_images_scored': 0, 'publisher_python_executed': False, 'serving_admitted': False,
        'scope': 'Three authored synthetic RGB fixtures only. Full released state equality, standard native pipeline repeat/batch behavior and separate outer mean-logit/sigmoid arithmetic; not an alternate architecture implementation, detection accuracy, calibration or localized pixel evidence.'}
    write(output, 'worker-result.json', result)
    if not result['passed']:
        raise ValueError('Synthetic runtime differences exceed prospective tolerances')


def profile(model_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    for process in psutil.process_iter(['pid', 'cmdline']):
        cmd = process.info['cmdline'] or []
        if process.pid != os.getpid() and '-m' in cmd and cmd[cmd.index('-m') + 1] in {
                'scripts.run_audio_separation_v14', 'scripts.run_writing_modernbert_v4',
                'scripts.run_video_demamba_v13', 'scripts.repeat_audio_separation_v14',
                'scripts.audit_image_smaug_runtime_v7'}:
            raise ValueError('Run model studies sequentially')
    with (output / 'worker.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-m', 'scripts.audit_image_smaug_runtime_v7', 'worker',
                                  str(model_dir), str(output)], stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        started, max_rss, guard = time.monotonic(), 0, None
        while child.poll() is None:
            try:
                rss = process.memory_info().rss
            except psutil.NoSuchProcess:
                break
            max_rss = max(max_rss, rss)
            if rss > 6 * 1024**3 or psutil.virtual_memory().available < 1536 * 1024**2 or time.monotonic() - started > 180:
                guard = 'resource_or_timeout'
                child.terminate()
                break
            time.sleep(.1)
        code = child.wait(timeout=15)
    result = {'child_exit_code': code, 'guard': guard, 'pid': child.pid,
              'worker_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sampled_max_rss_lower_bound_bytes': max_rss, 'duration_s': time.monotonic() - started,
              'worker_result_available': (output / 'worker-result.json').exists(), 'project_images_scored': 0}
    write(output, 'profile.json', result)
    print(json.dumps(result), flush=True)
    if code or guard or not result['worker_result_available']:
        raise RuntimeError('Synthetic runtime incomplete; preserve logs, no project scoring')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('profile', 'worker'))
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.mode == 'worker':
        try:
            worker(args.model_dir, args.output)
        except Exception:
            traceback.print_exc()
            raise
    else:
        profile(args.model_dir, args.output)
