"""Select and acquire fresh image source pairs without loading any detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import prepare_image_supported_v11 as supported
from scripts import prepare_tgif_localization as tgif
from scripts import run_image_supported_v11 as previous

ROOT = Path(__file__).resolve().parents[2]
COUNT = 16  # Per generator, eight bbox and eight segmentation sources: 48 total.


def exclusion_inventory(main, trial):
    """Exclude all earlier selected COCO IDs, including unscored source drafts."""
    paths = sorted(p for p in (main / 'backend/data').rglob('tgif-plan.json')
        if not p.resolve().is_relative_to(trial.resolve()))
    ids, manifests = set(), []
    for path in paths:
        document = json.loads(path.read_text())
        supported.require(document['metadata_revision'] == tgif.REVISION, 'Unknown earlier TGIF metadata revision')
        found = {str(row['source_id']) for rows in document['groups'].values() for row in rows}
        ids.update(found)
        ids.update(map(str, document['excluded_coco_ids']))
        manifests.append({'path': str(path.relative_to(main)), 'sha256': supported.sha(path),
            'planned_source_ids': sorted(found)})
    supported.require(len(paths) >= 7 and len(ids) >= 288, 'All prior image selections must be inventoried')
    return ids, manifests


def candidate(main):
    trial = main / 'backend/data/independent-evaluation/image-supported-v11'
    previous.check(trial)
    complete = json.loads((trial / 'fit-complete.json').read_text())
    records = {}
    for arm in ('primary', 'primary-regions'):
        for suffix, kind in (('artifact', 'artifacts'), ('points', 'points')):
            path = trial / f'{arm}-{suffix}.json'
            supported.require(supported.sha(path) == complete[kind][arm], 'Frozen candidate or comparison parameters changed')
            records[path.name] = supported.sha(path)
    return {'development_trial': str(trial.relative_to(main)), 'fit_complete_sha256': supported.sha(trial / 'fit-complete.json'),
        'frozen_artifacts_and_points_sha256': records,
        'choice': 'JPEG primary-regions only, selected by the completed v11 development gates. Primary-only is a fixed comparator. No new fits or points.'}


def plan(main, trial):
    trial.mkdir(exist_ok=False)
    sources = trial / 'sources'
    sources.mkdir()
    exclusions, inventory = exclusion_inventory(main, trial)
    fixed = candidate(main)
    document = tgif.plan(sources, COUNT, exclusions)
    selected = [row for rows in document['groups'].values() for row in rows]
    supported.require(len(selected) == len({r['source_id'] for r in selected}) == 48
        and not ({r['source_id'] for r in selected} & exclusions), 'Forty-eight unique unconsumed source IDs required')
    # The older downloader's stated evaluation is not this experiment's rule.
    # Its source-selection method is reused; the actual rule is frozen here.
    document['evaluation'] = 'Fresh JPEG-only fixed v11 primary-regions and primary-only artifacts/points versus actual FUSED components. No refitting, PNG detector scoring or full-app verdict claim.'
    document['fresh_confirmation_v12'] = {
        'excluded_all_prior_selected_sources': True, 'excluded_source_ids': len(exclusions),
        'scope': 'Fresh to known project evaluations, same three benchmark generators. Pretraining overlap and independent upstream authorship are unknown.',
        'secondary_author_masks': 'Record unavailable author masks; never replace a selected original or edit. Exact RGB difference is required primary local truth.',
        'candidate': fixed, 'new_neural_forwards': 0, 'native_scoring_protocol_complete': False,
        'serving_admitted': False}
    (sources / 'tgif-plan.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    supported.write(trial / 'source-exclusions.json', {'schema': 'chipmunk.image-fresh-source-exclusions/12',
        'ids': sorted(exclusions), 'prior_plan_inventory': inventory})
    supported.write(trial / 'source-selection.json', {'schema': 'chipmunk.image-fresh-selection/12',
        'selection_code_sha256': supported.sha(Path(__file__)), 'source_helper_sha256': supported.sha(Path(tgif.__file__)),
        'source_plan_sha256': supported.sha(sources / 'tgif-plan.json'),
        'exclusions_sha256': supported.sha(trial / 'source-exclusions.json'), 'candidate': fixed,
        'planned_source_pairs': 48, 'planned_jpeg_inputs': 96, 'new_neural_forwards': 0,
        'native_scoring_protocol_complete': False, 'serving_admitted': False})
    print(json.dumps({'selected_source_pairs': 48, 'excluded_prior_ids': len(exclusions), 'new_neural_forwards': 0}), flush=True)


def check(main, trial):
    selection = json.loads((trial / 'source-selection.json').read_text())
    supported.require(selection['selection_code_sha256'] == supported.sha(Path(__file__))
        and selection['source_helper_sha256'] == supported.sha(Path(tgif.__file__))
        and selection['source_plan_sha256'] == supported.sha(trial / 'sources/tgif-plan.json')
        and selection['exclusions_sha256'] == supported.sha(trial / 'source-exclusions.json')
        and selection['candidate'] == candidate(main), 'Source/candidate selection changed')
    exclusions = json.loads((trial / 'source-exclusions.json').read_text())
    for row in exclusions['prior_plan_inventory']:
        supported.require(supported.sha(main / row['path']) == row['sha256'], 'An inventoried prior source plan changed')
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'acquire'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    if args.stage == 'plan':
        plan(args.main.resolve(), args.trial.resolve())
    else:
        check(args.main.resolve(), args.trial.resolve())
        tgif.acquire(args.trial / 'sources', allow_missing_author_masks=True)
        check(args.main.resolve(), args.trial.resolve())
