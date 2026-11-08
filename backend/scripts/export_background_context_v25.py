"""Publish complete video-context aggregates, never private source geometry."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from scripts import run_background_context_v25 as study

SCIENCE = 'cf62bee345c524862bdd962db87c7fafe1f5de9f'


def publish(trial, output):
    study.check(trial)
    study.check_points(trial)
    result = study.read(trial / 'result.json')
    freeze = study.read(study.ROOT / 'docs/evaluation-panels/video-background-context-v25-freeze.json')
    study.require(freeze['private_freeze_sha256'] == study.sha(trial / 'freeze.json'), 'Original published source freeze required')
    for path, digest in freeze['code_sha256'].items():
        original = subprocess.check_output(['git', 'show', f'{SCIENCE}:{path}'], cwd=study.ROOT)
        study.require(hashlib.sha256(original).hexdigest() == digest, 'Prospectively published source differs')
    study.require(result['point_files'] == 80 and result['diagnostic_files'] == 200 and result['outcomes'] == 600
        and result['diagnostic_connected_components'] == 20 and result['all_source_pixels_and_clocks_bound'] is True
        and result['all_saved_arrays_and_points_replayed'] is True and result['new_background_model_forwards'] == 0
        and result['counts']['sface_parity_forwards'] > 0
        and result['native_integral_max_difference'] <= 1e-12 and result['independent_temporal_max_difference'] <= 1e-12
        and result['serving_admitted'] is result['used_in_confidence'] is False, 'Complete fixed comparison and source arithmetic required')
    profiles = {}
    for phase in ('point', 'diagnostic', 'ordinary', 'review'):
        raw = study.read(trial / (phase + '-profile/profile.json'))
        study.require(raw['child_exit_code'] == 0 and raw['guard'] is None and raw['seconds_limit'] == 900,
                      'Every original guarded phase must have completed')
        profiles[phase] = {k: v for k, v in raw.items() if k != 'arguments'}
        profiles[phase]['private_arguments_withheld'] = True
    point_groups = len({r['group'] for r in study.lines(trial / 'point.jsonl')})
    assessed = result['points']['context']['assessed_groups']
    study.require(point_groups == 40 and assessed + result['points']['context']['unassessed_groups'] == point_groups
                  and result['points']['appearance-matched']['assessed_groups'] == assessed,
                  'All original point groups and the common assessed denominator must remain explicit')
    outcomes = study.read(trial / 'outcomes-private.json')
    coverage = {}
    for arm in study.POLICY['arms']:
        part = [r for r in outcomes if r['arm'] == arm]
        study.require(len(part) == 200 and all(r['originality_verdict'] is None for r in part),
                      'Every case must retain its actual observation scope without originality verdicts')
        coverage[arm] = {'planned_files': 200, 'wholly_unassessed': sum(r['scored_units'] == 0 for r in part),
            'partially_assessed': sum(0 < r['scored_units'] < r['planned_units'] for r in part),
            'all_planned_units_assessed': sum(r['scored_units'] == r['planned_units'] for r in part),
            'planned_units': sum(r['planned_units'] for r in part), 'assessed_units': sum(r['scored_units'] for r in part)}
    record = {'schema': 'chipmunk.video-background-context-public-development/25', 'science_commit': SCIENCE,
        'private_freeze_sha256': freeze['private_freeze_sha256'], 'aggregate_result': result,
        'resource_profiles': profiles, 'points_fit_complete_sha256': study.sha(trial / 'point-fit-complete.json'),
        'external_harness_history': study.read(trial / 'external-harness-events.json'),
        'point_coverage': {'planned_source_groups': point_groups, 'common_assessed_source_groups': assessed,
            'groups_not_common_assessed': point_groups - assessed,
            'shadow_artifact_scope': 'Its unassessed_groups field describes the prefiltered common set only. Use these full40-group coverage counts when reporting availability.'},
        'observation_coverage_by_arm': coverage,
        'raw_media_published': False, 'source_pairings_published': False, 'geometry_or_feature_arrays_published': False,
        'customer_inputs': 0, 'external_provider_requests': 0, 'outside_human_reviewers': 0,
        'serving_changed': False, 'used_in_confidence': False, 'fresh_confirmation': False,
        'limits': [
            'Consumed sources and known constructions; no fresh people, generator transfer or population accuracy claim.',
            'Face model forwards are repeated on the two prospectively selected clips only; other source-bound native responses are reused.',
            'Background pixel means and temporal arithmetic were independently checked inline and replayed from private arrays, not re-extracted in a second full native pass.',
            'No identity lookup, whole-video originality, lip-sync verdict or identified edited pixels.',
            'Inspection flags and internal comparison points are not customer AI-authorship confidence.',
            'Any passing development result still needs fresh source confirmation and full-app/export verification.',
            'Sampled process-tree RSS is a lower bound; retain sampling errors and original failed external harness attempts.'
        ], 'exporter_sha256': study.sha(Path(__file__))}
    study.write(output, record)
    print({'published_outcomes': 600, 'primary_pass': result['primary_pass'], 'serving_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    publish(args.trial, args.output)
