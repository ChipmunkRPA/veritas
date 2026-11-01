"""CPU-only amendment: match persisted writer precision without new inference.

The previous policy/output remains immutable. Thresholds, inputs, JPEG conditions,
Pillow projection, connected components, minimum area and caps are unchanged.
This amendment corrects float32-to-PNG rounding and uses float64 decoded values.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.image import jpeg_bitstream  # noqa: E402
from app.detectors.image import localiser as loc  # noqa: E402
from app.engine import exhibits  # noqa: E402
from scripts import audit_fused_production_grid as original  # noqa: E402
from scripts.audit_fused_production_grid import component_masks, to_source  # noqa: E402
from scripts.localization_benchmark import mask_metrics, sha, source  # noqa: E402


def writer_replay(native: np.ndarray, pixels: np.ndarray, truth: np.ndarray, *, jpeg_quality: int | None,
           source_probabilities: np.ndarray, score: float):
    height, width = truth.shape
    if native.shape != (512, 512) or native.dtype != np.float32:
        raise ValueError("Native probabilities must be the original float32 512-square map")
    if source_probabilities.shape != truth.shape or pixels.shape != (*truth.shape, 3):
        raise ValueError("Source coordinates disagree")
    if (not np.isfinite(native).all() or not np.isfinite(source_probabilities).all()
            or not np.isfinite(score) or not 0 <= score <= 1
            or native.min() < 0 or native.max() > 1
            or source_probabilities.min() < 0 or source_probabilities.max() > 1):
        raise ValueError("Invalid probabilities or classifier response")
    grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE["screening"])
    floating = cv2.resize(native, grid, interpolation=cv2.INTER_LINEAR)
    encoded = np.rint(np.clip(floating.astype(np.float64), 0, 1) * 255).astype(np.uint8)
    values = encoded.astype(np.float64) / 255
    excluded, conditions = loc.conditions_mask(pixels, jpeg_quality)
    reliable = loc.reliable_grid(excluded, grid)
    grid_masks, components = component_masks(values, reliable, width, height)
    grid_masks = {"screening_float": floating >= 0.5, "recorded_threshold": values >= 0.5, **grid_masks}
    projected = {"native_source_float": source_probabilities >= 0.5,
                 **{name: to_source(mask, width, height) for name, mask in grid_masks.items()}}
    projected_reliable = to_source(reliable, width, height)
    metrics = {"ungated": {name: mask_metrics(mask, truth) for name, mask in projected.items()},
               "image_gated": {name: mask_metrics(mask if score >= 0.5 else np.zeros_like(mask), truth)
                               for name, mask in projected.items()}}
    measurement = {"grid": list(grid), "score": score, "image_gate_positive": score >= 0.5,
                   "jpeg_quality_luma": jpeg_quality, "conditions": conditions,
                   "reliable_grid_cells": int(reliable.sum()), "grid_cells": reliable.size,
                   "reliable_grid_fraction": float(reliable.mean()), "fully_unmeasurable": not bool(reliable.any()),
                   "excluded_native_pixels": int(excluded.sum()),
                   "excluded_truth_pixels_native_conditions": int((excluded & truth).sum()),
                   "excluded_truth_pixels_projected_grid": int((~projected_reliable & truth).sum()),
                   "quantization_threshold_changed_cells": int(((floating >= .5) != (values >= .5)).sum()),
                   "uniform_response_on_reliable_cells": bool(reliable.any() and np.ptp(values[reliable]) == 0),
                   **components, "metrics": metrics}
    return measurement, encoded, reliable, grid_masks



def stored_png(directory: Path, record: dict) -> np.ndarray:
    path = directory / record['path']
    if path.parent != directory or sha(path.read_bytes()) != record['sha256']:
        raise ValueError('Prior audit asset differs from its frozen hash/path')
    with Image.open(path) as image:
        values = np.asarray(image).copy()
    if list(values.shape) != record['shape']:
        raise ValueError('Prior audit asset dimensions differ')
    return values


def amend(directory: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError('Amendment requires a new output directory')
    old_result_bytes=(directory/'results.json').read_bytes()
    old_audit_bytes=(directory/'production-grid-audit.json').read_bytes()
    result=json.loads(old_result_bytes)
    old_audit=json.loads(old_audit_bytes)
    root=Path(result['manifest_path']).parent
    manifest_bytes=(root/'manifest.json').read_bytes()
    policy_bytes=(root/'production-grid-policy.json').read_bytes()
    old_policy=json.loads(policy_bytes)
    if (sha(manifest_bytes)!=result['manifest_sha256'] or sha(policy_bytes)!=result['production_grid_policy_sha256']
            or old_audit['source_result_sha256']!=sha(old_result_bytes)
            or old_audit['manifest_sha256']!=sha(manifest_bytes) or old_audit['policy_sha256']!=sha(policy_bytes)):
        raise ValueError('Earlier manifest, policy, inference or audit commitments disagree')
    current_protocol=original.protocol()
    recorded_protocol={k:v for k,v in old_policy.items() if k not in {'declared_at','inference_has_started'}}
    for policy in (current_protocol, recorded_protocol):
        policy.pop('implementation_sha256')
    if current_protocol!=recorded_protocol:
        raise ValueError('Other operating policy changed; this is only a precision amendment')
    frozen_old_script=old_policy['implementation_sha256']['scripts/audit_fused_production_grid.py']
    if sha(Path(original.__file__).read_bytes())!=frozen_old_script:
        raise ValueError('The original audit helper changed')
    manifest=json.loads(manifest_bytes)
    items={item['id']:item for item in manifest['items']}
    ids=[row['id'] for row in result['rows']]
    if (len(items)!=len(manifest['items']) or ids!=result['selected_items'] or len(ids)!=len(set(ids))
            or ids!=[row['id'] for row in old_audit['rows']] or set(ids)!=set(items)):
        raise ValueError('Amendment must cover every original item in the same order')
    files={str(directory/'results.json'):sha(old_result_bytes), str(directory/'production-grid-audit.json'):sha(old_audit_bytes),
           str(root/'manifest.json'):sha(manifest_bytes), str(root/'production-grid-policy.json'):sha(policy_bytes)}
    output.mkdir()
    policy={
        'schema':'chipmunk.fused-writer-precision-amendment-policy/1','declared_at':datetime.now(UTC).isoformat(),
        'new_inference':False,'threshold_tuning':False,
        'reason':'Old adapter/audit used float32 quantization; persisted ExhibitWriter first uses float64, so half-rounding values and some region statistics differ.',
        'old_policy_sha256':sha(policy_bytes),'old_audit_sha256':sha(old_audit_bytes),'saved_inference_sha256':sha(old_result_bytes),
        'manifest_sha256':sha(manifest_bytes),'selection':'all prior 200 ordered saved inference rows; no selection by score',
        'recorded_quantization':'np.rint(np.clip(float_grid.astype(float64),0,1)*255).astype(uint8)',
        'recorded_decode':'encoded.astype(float64)/255; matches the corrected adapter and float64 evidence reader',
        'unchanged_policy':current_protocol,
        'implementation_sha256':{**original.implementation_hashes(),
            'scripts/audit_fused_writer_quantization.py':sha(Path(__file__).read_bytes()),
            'app/engine/exhibits.py':sha(Path(exhibits.__file__).read_bytes())},
        'interpretation':'Measurement correction only, not accuracy, representative validation or proof of edits.'}
    original.write_new(output/'amendment-policy.json',policy)
    assets=output/'masks'
    assets.mkdir()
    rows=[]
    old_assets=directory/'production-grid-masks'
    for position,(row,old_row) in enumerate(zip(result['rows'],old_audit['rows'],strict=True),1):
        item=items[row['id']]
        if (row['family'],row['label'])!=(item['family'],item['label']):
            raise ValueError('Saved result label differs from the manifest')
        data,pixels,truth=source(root,item)
        structure=jpeg_bitstream.read(data)
        native=original.probability(directory,row['probabilities']['native'],(512,512))
        projected=original.probability(directory,row['probabilities']['source'],truth.shape)
        measured,encoded,reliable,masks=writer_replay(native,pixels,truth,source_probabilities=projected,
            jpeg_quality=structure.quality_luma if structure else None,score=row['score'])
        for field in ('intersection','false_positive_pixels','false_negative_pixels','iou','pixel_f1'):
            if measured['metrics']['ungated']['native_source_float'][field]!=row[field]:
                raise ValueError('Saved source projection no longer reproduces original inference metrics')
        old_encoded=stored_png(old_assets,old_row['artifacts']['recorded_scalar'])
        old_reliable=stored_png(old_assets,old_row['artifacts']['reliability'])>127
        if not np.array_equal(reliable,old_reliable):
            raise ValueError('Conditions changed; this amendment must change quantization only')
        old_cap=stored_png(old_assets,old_row['artifacts']['ungated.per_map_display_cap'])>127
        changed={
            'encoded_changed_cells':int((encoded!=old_encoded).sum()),
            'threshold_changed_cells':int(((encoded>=128)!=(old_encoded>=128)).sum()),
            'component_count_changed':measured['qualifying_component_count']!=old_row['qualifying_component_count'],
            'component_geometry_changed':[(r['component_label'],r['cells'],r['bbox']) for r in measured['ranked_components']]!=[(r['component_label'],r['cells'],r['bbox']) for r in old_row['ranked_components']],
            'component_statistics_changed':measured['ranked_components']!=old_row['ranked_components'],
            'cap2_selection_changed_cells':int((masks['per_map_display_cap']!=old_cap).sum()),
            'cap2_selection_changed':not np.array_equal(masks['per_map_display_cap'],old_cap),
        }
        artifacts={'recorded_scalar':original.png(assets/(item['id']+'.scalar.png'),encoded),
                   'reliability':original.png(assets/(item['id']+'.reliable.png'),reliable.astype(np.uint8)*255),
                   'ungated.per_map_display_cap':original.png(assets/(item['id']+'.ungated.cap2.png'),masks['per_map_display_cap'].astype(np.uint8)*255),
                   'image_gated.per_map_display_cap':original.png(assets/(item['id']+'.gated.cap2.png'),(masks['per_map_display_cap'] if row['score']>=.5 else np.zeros_like(reliable)).astype(np.uint8)*255)}
        rows.append({**{k:old_row[k] for k in ('id','family','label','source_sha256','reference_sha256','source_dimensions',
                    'exif_orientation','production_comparable','production_comparable_exclusion_reasons','probabilities')},
                     'artifacts':artifacts,'amendment_changes':changed,**measured})
        if position%20==0:
            print(json.dumps({'processed':position,'total':len(ids)}),flush=True)
    families=['all',*sorted({r['family'] for r in rows})]
    changes={
        'images_with_quantization_changes':sum(r['amendment_changes']['encoded_changed_cells']>0 for r in rows),
        'encoded_changed_cells':sum(r['amendment_changes']['encoded_changed_cells'] for r in rows),
        'threshold_changed_cells':sum(r['amendment_changes']['threshold_changed_cells'] for r in rows),
        **{k:sum(bool(r['amendment_changes'][k]) for r in rows) for k in ('component_count_changed','component_geometry_changed','component_statistics_changed','cap2_selection_changed')},
        'cap2_selection_changed_cells':sum(r['amendment_changes']['cap2_selection_changed_cells'] for r in rows),
    }
    document={
        'schema':'chipmunk.fused-writer-precision-amendment/1','policy_sha256':sha((output/'amendment-policy.json').read_bytes()),
        'previous_audit_sha256':sha(old_audit_bytes),'source_result_sha256':sha(old_result_bytes),
        'manifest_sha256':sha(manifest_bytes),'protocol':policy,'change_summary':changes,
        'previous_all_summary':old_audit['production_comparable_summary']['all'],
        'all_stored_coordinate_summary':{f:original.summary([r for r in rows if f=='all' or r['family']==f]) for f in families},
        'production_comparable_summary':{f:original.summary([r for r in rows if r['production_comparable'] and (f=='all' or r['family']==f)]) for f in families},
        'prior_top_level_files_unchanged':all(sha(Path(path).read_bytes())==digest for path,digest in files.items()),
        'libraries':{'numpy':np.__version__,'opencv':cv2.__version__,'pillow':Image.__version__},'rows':rows}
    if not document['prior_top_level_files_unchanged']:
        raise ValueError('Earlier artifacts changed during amendment')
    original.write_new(output/'amendment-results.json',document)
    print(json.dumps({'change_summary':changes,'production_comparable':document['production_comparable_summary']['all']},indent=2),flush=True)
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    amend(args.directory,args.output)
