"""Private source-bound before/during/after illustration; never public media."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from scripts import background_context_v25 as method
from scripts import run_background_context_v25 as run


def plot(trial, output):
    result = run.read(trial / 'result.json')
    require = run.require
    require(result['diagnostic_files'] == 200 and result['outcomes'] == 600, 'Completed study required')
    outcomes = run.read(trial / 'outcomes-private.json')
    flags = {(r['id'], r['arm']): r for r in outcomes}
    rows = {r['id']: r for r in run.lines(trial / 'ordinary.jsonl')}
    plans = {r['id']: r for r in run.lines(run.ORDINARY / 'input-plan.jsonl')}
    ordinary = sorted(rows)
    corrected = [k for k in ordinary if flags[k, 'current-fixed']['flag'] is True and flags[k, 'context']['flag'] is False]
    retained = [k for k in ordinary if flags[k, 'current-fixed']['flag'] is True and flags[k, 'context']['flag'] is True]
    chosen = [corrected[0] if corrected else ordinary[0]]
    if retained:
        chosen.append(retained[0])
    elif len(corrected) > 1:
        chosen.append(corrected[1])
    fig, axes = plt.subplots(len(chosen), 3, figsize=(15, 4.1 * len(chosen) + 1), squeeze=False)
    fig.suptitle('Ordinary scene substitutions: face and background context', fontsize=18, fontweight='bold', y=.985)
    for index, identity in enumerate(chosen):
        row, plan = rows[identity], plans[identity]
        valid = [u for u in row['units'] if u['baseline'] is not None and u['context']['raw'] is not None]
        require(bool(valid), 'Illustration requires actually measured source context')
        unit = max(valid, key=lambda r: r['baseline'])
        boxes = {int(k): v for k, v in row['boxes'].items()}
        frames = sorted({i for key in ('sampled_frames', 'context_frames') for i in unit['location'][key]})
        keep = method.background_mask([boxes[i] for i in frames])
        a, b = unit['location']['start'] / 1e9, unit['location']['end'] / 1e9
        context = flags[identity, 'context']
        name = 'Background context removes this alert' if context['flag'] is False else 'An alert remains on this ordinary edit'
        note = f"Control {index + 1}: {name}\nMeasured support {a:.2f}–{b:.2f}s · source insertion 2.00–4.00s"
        axes[index, 0].text(0, 1.09, note, transform=axes[index, 0].transAxes, fontsize=11, linespacing=1.5)
        for col, frame in enumerate((36, 72, 108)):
            proof = plan['anchors'][str(frame)]
            path = run.ORDINARY / proof['path']
            require(run.sha(path) == proof['sha256'], 'Original anchor bytes changed')
            rgb = np.asarray(Image.open(path).convert('RGB'))
            require(rgb.shape == (480, 640, 3) and run.ordinary.digest(rgb) == proof['source_rgb_sha256'] == plan['rgb_sha256'][frame], 'Anchor differs from actual source pixels')
            axis = axes[index, col]
            axis.imshow(rgb, interpolation='nearest')
            for y, x in zip(*np.where(keep), strict=True):
                axis.add_patch(Rectangle((x * 40, y * 40), 40, 40, fill=False, edgecolor='#00a5b5', linewidth=.35, alpha=.5))
            if frame in boxes:
                x, y, w, h = boxes[frame]
                axis.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='#ffbc42', linewidth=1.7))
            axis.set_title(f"{('Before', 'During', 'After')[col]} · {frame / 24:.2f}s · frame {frame}", fontsize=11)
            axis.set_xlim(0, 640)
            axis.set_ylim(480, 0)
            axis.axis('off')
        axes[index, 0].text(0, -.11,
            f"Window appearance change {unit['baseline']:.4f} · background change {unit['context']['raw']:.4f} · adjusted response {unit['candidate']:.4f}",
            transform=axes[index, 0].transAxes, fontsize=10)
    fig.text(.04, .025, 'Amber: detected face. Cyan: common background cells used in the measured window.\nNo AI was introduced by this copy-only construction. Responses are inspection measurements, not AI confidence.', fontsize=10)
    fig.subplots_adjust(left=.035, right=.985, top=.84 if len(chosen) == 1 else .89, bottom=.15, hspace=.6, wspace=.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), 'Retain original private illustration')
    fig.savefig(output, dpi=140, facecolor='white')
    plt.close(fig)
    print({'private_control_rows': len(chosen), 'source_frames': len(chosen) * 3, 'new_model_forwards': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plot(args.trial, args.output)
