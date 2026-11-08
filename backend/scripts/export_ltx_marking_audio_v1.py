"""Publish complete native semantics/audit aggregates and retain the failed run."""
import argparse
import hashlib
import subprocess
from pathlib import Path

from scripts import continue_ltx_marking_audio_v1 as study

ORIGINAL = '17002e2cc86d59b51aab32ac36024f6cc7db9902'
CONTINUATION = '55786dc3ac22cf41c0af9264abd80dd8cf639e32'


def publish(trial, output):
    study.check(trial)
    base, require = study.old.base, study.old.base.require
    freeze = base.read(study.old.ROOT / 'docs/evaluation-panels/ltx-marking-audio-v1-continuation-freeze.json')
    require(freeze['private_freeze_sha256'] == base.digest(trial / 'freeze.json'), 'Published continuation freeze required')
    for path, sha in freeze['code_sha256'].items():
        source = subprocess.check_output(['git', 'show', f'{CONTINUATION}:{path}'], cwd=study.old.ROOT)
        require(hashlib.sha256(source).hexdigest() == sha, 'Prospectively published source differs')
    original = base.read(study.PREVIOUS / 'freeze.json')
    for path, sha in original['code_sha256'].items():
        source = subprocess.check_output(['git', 'show', f'{ORIGINAL}:{path}'], cwd=study.old.ROOT)
        require(hashlib.sha256(source).hexdigest() == sha, 'Original failed science changed')
    native, review = base.read(trial / 'result.json'), base.read(trial / 'sample-review.json')
    require(native['cases'] == review['cases'] == 80 and native['generator_forwards'] == 17
        and native['detector_forwards'] == 81 and review['actual_file_decodes_and_clocks_replayed'] == 32
        and review['native_channel_sample_responses_replayed'] == 11520000
        and review['first_failed_attempt_prefix_cases_exact'] == 6 and review['complete_summary_exact']
        and native['serving_admitted'] is native['used_in_confidence'] is native['early_stop_eligible'] is False,
        'Complete original outcomes/repeats and independent replay required')
    profiles = {}
    for name, path in [('failed_original', study.PREVIOUS / 'profile/profile.json'),
                       ('complete_native', trial / 'profile/profile.json'), ('complete_review', trial / 'review-profile/profile.json')]:
        raw = base.read(path)
        require(raw['child_exit_code'] == (1 if name == 'failed_original' else 0)
                and raw['guard'] is None and raw['seconds_limit'] == 900, 'Original terminal attempts must remain')
        profiles[name] = {k: v for k, v in raw.items() if k != 'arguments'}
        profiles[name]['private_arguments_withheld'] = True
    record = {'schema': 'chipmunk.ltx-public-marking-contract-result/1', 'original_science_commit': ORIGINAL,
        'continuation_science_commit': CONTINUATION, 'private_freeze_sha256': freeze['private_freeze_sha256'],
        'native': native, 'independent_saved_sample_and_file_review': review, 'resource_profiles': profiles,
        'retained_original_failure': {'persisted_cases': 6, 'error': 'FileNotFoundError while writing first codec packet',
            'original_files_unchanged': True, 'final_model_call_counter': None,
            'scope': 'The failed attempt did not write final counters; the complete continuation counters do not include that earlier work.'},
        'source_lock': base.read(study.old.method.LOCK), 'outside_human_reviewers': 0,
        'customer_inputs': 0, 'provider_verification_calls': 0, 'new_weight_downloads': 0,
        'raw_audio_or_probability_arrays_published': False, 'serving_changed': False,
        'limits': ['Authored mathematical carriers, not natural speech, real provider exports or a population error rate.',
            'Compatible published audio marking arithmetic executed; full upstream package/loader and video PixelSeal were not executed.',
            'Public pattern presence is not authenticated provider involvement or AI authorship.',
            'Noise and partial-mark misses are retained; absent marks never establish originality.',
            'Sample-array and file replay is not another neural forward pass or outside human review.'],
        'exporter_sha256': base.digest(Path(__file__))}
    base.write(output, record)
    print({'published_cases': 80, 'sample_responses_reviewed': 11520000, 'original_failure_retained': True,
           'early_stop_eligible': False, 'serving_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    publish(args.trial, args.output)
