"""Bounded numerical parity against pinned, separately downloaded reference code.

Reference sources remain under ignored data/, are hash committed in the output,
and are never imported by the product. Only local pinned weights are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from unittest.mock import patch

import numpy as np
import torch
from fused_research_eval import DATA, SOURCE_REVISION, FusedResearch
from PIL import Image

REFERENCE = DATA / 'reference-fused'
SOURCES = ('model.py', 'blocks.py', 'decoder.py', 'moe.py', 'sparse_vit.py')
REFERENCE_HASHES = {
    'blocks.py': '4b0c83dc4cd354e9a29d195fdb24e15d7f3412b9bac677ea58fbe292aa179923',
    'decoder.py': 'c95529ec6ebedd7234251697849322b87308be9a5bb61501b5e83dadc7f88424',
    'model.py': '05201d9ee594266bb681c078bae103427c89197ff6743b9da4a9a41247a4dc40',
    'moe.py': '8286caad7d8940ad9743b75e7cd308223bf42fcb3505f895228ac5565c82296e',
    'sparse_vit.py': '92d60770c5760283a6fc54b0a239126fc1b5e457d3aa120d40ec9f82da28fa4d',
}
ATOL, RTOL = 0.0001, 0.0001


def reference_model(research):
    sys.path.insert(0, str(REFERENCE))
    from fused.model.model import FUSED

    parameters = inspect.signature(FUSED).parameters
    config = {key: value for key, value in research.config.items() if key in parameters}
    config['pretrained_path'] = None
    linspace = torch.linspace

    def cpu_linspace(*args, **kwargs):
        return linspace(*args, **{**kwargs, 'device': 'cpu'})

    def local_backbone(name, **kwargs):
        if name != 'convnext_xxlarge.clip_laion2b_soup' or kwargs != {'pretrained': True, 'num_classes': 0}:
            raise ValueError('Unexpected reference backbone construction')
        return research.backbone

    # Avoid allocating a second 3.2 GiB backbone or downloading weights. Trainable
    # layers are constructed on meta, then assigned the exact same saved tensors.
    with patch('timm.create_model', local_backbone), patch('torch.linspace', cpu_linspace), torch.device('meta'):
        model = FUSED(**config)
    loaded = model.load_state_dict(research.ops.tensors, strict=False, assign=True)
    missing = [key for key in loaded.missing_keys if not key.startswith('convnext.')]
    unexpected = [key for key in loaded.unexpected_keys if not key.startswith(('dinov2_', 'seg_bridge.'))]
    if missing or unexpected:
        raise ValueError(f'Reference state mismatch: missing={missing}, unexpected={unexpected}')
    model.in_mean = torch.tensor([0.485, 0.456, 0.406], device=research.device).view(1, 3, 1, 1)
    model.in_std = torch.tensor([0.229, 0.224, 0.225], device=research.device).view(1, 3, 1, 1)
    if any(t.device.type == 'meta' for t in (*model.parameters(), *model.buffers())):
        raise ValueError('Reference has unresolved meta parameters')
    return model.eval(), list(loaded.unexpected_keys)


@torch.inference_mode()
def run(device):
    source_hashes = {name: hashlib.sha256((REFERENCE / 'fused/model' / name).read_bytes()).hexdigest() for name in SOURCES}
    if source_hashes != REFERENCE_HASHES:
        raise ValueError('The separately reviewed reference source has changed')
    manifest = json.loads((DATA / 'manifest.json').read_text())
    selected, groups = [], set()
    for item in manifest['items']:
        group = (item['family'], item['label'])
        if group not in groups:
            selected.append(item)
            groups.add(group)
    research = FusedResearch(device)
    reference, unused = reference_model(research)
    report = {'purpose': 'local numerical implementation parity, not accuracy or product admission',
              'source_revision': SOURCE_REVISION, 'source_sha256': source_hashes,
              'device': device, 'torch_version': torch.__version__, 'dtype': 'float32', 'atol': ATOL, 'rtol': RTOL,
              'selection': 'first manifest item per family and label; no score selection',
              'comparison': 'same resized RGB pixels, native 512 mask probabilities before source projection; independent normalization paths',
              'reference_unused_checkpoint_keys': unused, 'rows': []}
    output = DATA / 'reference-parity.json'
    for item in selected:
        path = DATA / item['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['image_sha256']:
            raise ValueError('Source image differs from pinned manifest')
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        actual, actual_score = research.predict(image)
        pixels = torch.from_numpy(np.array(image.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255)
        pixels = pixels.permute(2, 0, 1)[None].to(device)
        result = reference((pixels - reference.in_mean) / reference.in_std)
        expected = result['logits'].sigmoid()[0, 0].cpu().numpy()
        expected_score = result['cls_logits'].softmax(-1)[0, 1].item()
        row = {'id': item['id'], 'image_sha256': item['image_sha256'],
               'max_mask_absolute_error': float(np.abs(actual - expected).max()),
               'mean_mask_absolute_error': float(np.abs(actual - expected).mean()),
               'classifier_absolute_error': abs(actual_score - expected_score),
               'mask_pixels_crossing_fixed_0_5': int(((actual >= 0.5) != (expected >= 0.5)).sum()),
               'mask_allclose': bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
               'classifier_allclose': bool(np.isclose(actual_score, expected_score, atol=ATOL, rtol=RTOL))}
        report['rows'].append(row)
        report['passed'] = all(r['mask_allclose'] and r['classifier_allclose'] for r in report['rows'])
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
    if not report['passed']:
        raise SystemExit('Numerical parity failed; see recorded errors')
    print(f'Parity passed for {len(selected)} inputs: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    run(parser.parse_args().device)
