"""Render private, source-matched examples from the rejected research result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from scripts import image_patch_head_v14 as method
from scripts import run_image_patch_head_v14 as study

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


def render(trial, output):
    plan = study.check(trial)
    diagnostic = study.read(trial / 'diagnostic-replay.json')
    method.require(diagnostic['cases'] == 1344 and diagnostic['native_features_and_maps_exact'],
                   'Complete actual native diagnostic replay required before illustration')
    outcomes = {r['id']: r for r in study.read(trial / 'outcomes-private.json')}
    rows = [json.loads(line) for line in (trial / 'native-private.jsonl').read_text().splitlines()]
    ai = [r for r in rows if r['role'] == 'diagnostic' and r['label'] == 1]
    controls = [r for r in rows if r['role'] == 'diagnostic' and r['label'] == 0 and r['variant'] == 'original']
    chosen = [
        ('Best overlap in this panel', max(ai, key=lambda r: (outcomes[r['id']]['metrics']['iou'], r['id']))),
        ('Flagged edit with weakest overlap', min((r for r in ai if outcomes[r['id']]['metrics']['flag']),
                                                key=lambda r: (outcomes[r['id']]['metrics']['iou'], r['id']))),
        ('Missed edit with largest changed area', max((r for r in ai if not outcomes[r['id']]['metrics']['flag']),
                                                     key=lambda r: (outcomes[r['id']]['metrics']['reference_positive_pixels'], r['id']))),
        ('Highest response on a genuine control', max(controls, key=lambda r: (r['maximum'], r['id']))),
    ]
    peers = {(e['cohort'], e['source']['group'], e['source']['label']): i
             for i, e in enumerate(plan['source_plan']['entries'])}
    point = study.read(trial / 'fitted-point-private.json')['point']['value']
    figure, axes = plt.subplots(4, 3, figsize=(12, 14))
    for index, (label, row) in enumerate(chosen):
        entry, pixels, truth = study.pixels_for(plan, row)
        peer = {'entry_index': peers[(entry['cohort'], entry['source']['group'], 0)], 'variant': 'original'}
        _, parent, _ = study.pixels_for(plan, peer)
        scores = np.load(trial / 'private' / row['scores_file'], allow_pickle=False)
        method.require(method.sha(scores) == row['scores_sha256'], 'Exact scored grid required')
        mask = method.selected(scores, point, row['width'], row['height'])
        method.require(method.sha(mask) == outcomes[row['id']]['mask_sha256'], 'Illustration must use the recorded source mask')
        region = mask if mask.any() else truth
        if region.any():
            yy, xx = np.where(region)
            x0, x1, y0, y1 = int(xx.min()), int(xx.max()) + 1, int(yy.min()), int(yy.max()) + 1
        else:
            y, x = np.unravel_index(np.argmax(scores), scores.shape)
            x0, x1 = int(np.ceil(x * row['width'] / 32)), int(np.ceil((x + 1) * row['width'] / 32))
            y0, y1 = int(np.ceil(y * row['height'] / 32)), int(np.ceil((y + 1) * row['height'] / 32))
        pad = max(3, int(.06 * max(x1 - x0, y1 - y0)))
        x0, x1 = max(0, x0 - pad), min(row['width'], x1 + pad)
        y0, y1 = max(0, y0 - pad), min(row['height'], y1 + pad)
        axes[index, 0].imshow(parent)
        axes[index, 0].set_title(label, fontsize=10)
        axes[index, 1].imshow(pixels)
        overlay = np.zeros((*mask.shape, 4), np.float32)
        overlay[mask] = (1, .05, .04, .45)
        axes[index, 1].imshow(overlay)
        if truth.any() and not truth.all():
            axes[index, 1].contour(truth, levels=[.5], colors=['#00bd4a'], linewidths=.8)
        axes[index, 1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#ffd323', linewidth=1))
        values = outcomes[row['id']]['metrics']
        overlap = 'n/a' if values['iou'] is None else f"{values['iou']:.3f}"
        axes[index, 1].set_title(f"Examined source | selected: {'yes' if mask.any() else 'no'} | IoU {overlap}", fontsize=9)
        axes[index, 2].imshow(np.concatenate([parent[y0:y1, x0:x1], pixels[y0:y1, x0:x1]], axis=1))
        axes[index, 2].set_title(f'Matched crop: original / examined\nx{x0}, y{y0}, {x1-x0}x{y1-y0}', fontsize=9)
        for ax in axes[index]:
            ax.set_xticks([])
            ax.set_yticks([])
    figure.suptitle('Chipmunk Verify — rejected image-region prototype\nSource-matched development evidence', fontsize=15, y=.985)
    figure.text(.025, .025, 'Green: known study edit. Red: prototype-selected pixels. Yellow: matched crop.\n'
                'The grid is a ranking response, not AI confidence or pixel-origin proof.\n'
                'Local TGIF/TGIF2 research sources; raw photos and this preview remain outside the public repository.', fontsize=9)
    figure.subplots_adjust(top=.91, bottom=.085, left=.025, right=.985, hspace=.32, wspace=.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    method.require(not output.exists(), 'Do not overwrite an earlier evidence preview')
    figure.savefig(output, dpi=140)
    plt.close(figure)
    print({'private_preview': str(output), 'examples': len(chosen), 'neural_forwards': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    render(args.trial, args.output)
