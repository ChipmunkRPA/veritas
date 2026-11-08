"""Publish full comparison aggregates, never audio traces or private fitted heads."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def publish(trial, output, science):
    require(len(science) == 40 and all(c in "0123456789abcdef" for c in science), "Full immutable science revision required")
    freeze = read(ROOT / 'docs/evaluation-panels/mouth-audio-context-v14-freeze.json')
    require(freeze['private_freeze_sha256'] == sha(trial / 'freeze.json'), 'Original prospective private freeze required')
    for name, expected in freeze['code_sha256'].items():
        require(sha(ROOT / name) == expected and hashlib.sha256(subprocess.check_output(
            ['git', 'show', science + ':' + name], cwd=ROOT)).hexdigest() == expected, 'Published prospective method differs')
    result, review = read(trial / 'result.json'), read(trial / 'complete-review.json')
    require(result['new_audio_decodes'] == review['audio_tracks_redecoded'] == 272
            and result['all_diagnostic_cases'] == 88 and review['new_window_feature_vectors_replayed'] == 6373
            and review['raw_new_feature_values_replayed'] == 127460 and review['all_native_mask_time_outcomes_exact']
            and review['both_complete_aggregates_exact'] and review['result_sha256'] == sha(trial / 'result.json')
            and review['reviewer_sha256'] == freeze['code_sha256']['backend/scripts/review_mouth_audio_context_v14.py'],
            'Complete native audio and independent source/feature/outcome review required')
    profiles = {}
    for name in ('fit-audio-profile', 'point-audio-profile', 'fit-profile', 'development-audio-profile', 'score-profile', 'review-profile'):
        profile = read(trial / name / 'profile.json')
        require(profile['child_exit_code'] == 0 and profile['guard'] is None and profile['seconds_limit'] == 900,
                'Every terminal bounded phase required')
        profiles[name] = {k: v for k, v in profile.items() if k != 'arguments'}
        profiles[name]['private_arguments_withheld'] = True
    stages = {name: read(trial / (name + '-complete.json')) for name in ('fit', 'calibration', 'development')}
    require(sum(r['clips'] for r in stages.values()) == 272 and sum(r['windows'] for r in stages.values()) == 6373,
            'All planned native cases/windows required')
    prefix = read(trial / 'prior-prefix-parity.json')
    require(prefix == {'fit_cases_exact': 160, 'point_cases_exact': 18, 'original_failure_unchanged': True},
            'Original complete persisted prefix must reproduce exactly')
    previous_folder = ROOT / 'backend/data/independent-evaluation/lipsync-audio-context-v13'
    previous_profiles = {}
    for name in ('fit-audio-profile', 'point-audio-profile'):
        value = read(previous_folder / name / 'profile.json')
        require(value['child_exit_code'] == (0 if name == 'fit-audio-profile' else 1) and value['guard'] is None,
                'Original native failure/status must remain')
        previous_profiles[name] = {k: v for k, v in value.items() if k != 'arguments'}
        previous_profiles[name]['private_arguments_withheld'] = True
    record = {'schema': 'chipmunk.mouth-audio-context-complete-result/14', 'science_commit': science,
              'public_freeze_sha256': sha(ROOT / 'docs/evaluation-panels/mouth-audio-context-v14-freeze.json'),
              'result': result, 'separate_complete_review': review, 'source_stages': stages,
              'resource_profiles': profiles, 'fit_point_correspondence': read(trial / 'head-fit-complete.json')['fit_point_native_export'],
              'earlier_source_preflight': freeze['historical_whole_app_guard'],
              'retained_original_native_attempt': freeze['prior_attempt'],
              'retained_original_resource_profiles': previous_profiles, 'original_prefix_parity': prefix,
              'small_head_fits': 4, 'fitting_only_head_reconstructions_included': 2,
              'new_visual_neural_forwards': 0, 'new_external_weights': 0, 'customer_inputs': 0,
              'outside_human_reviewers': 0, 'serving_changed': False,
              'raw_media_new_audio_traces_or_private_heads_published': False,
              'limitations': ['Consumed correlated development clips from28 actors,not fresh validation.',
                  'Native audio source/clock replay is new; previously audited visual measurements are reused.',
                  'Audio/mouth correlation is not AI provenance or achieved-sync ground truth.',
                  'Clean studio speech does not qualify strong noise,music or overlap.',
                  'A passing development comparison still requires fresh sources and full-app/export/payment verification.'],
              'exporter_sha256': sha(Path(__file__))}
    with output.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({'native_audio_tracks': 272, 'independent_audio_tracks': 272, 'windows_reviewed': 6373,
           'passes': result['passes'], 'serving_changed': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('science')
    args = parser.parse_args()
    publish(args.trial, args.output, args.science)
