"""Technical resource successor: identical v87 mathematics, bounded numerical worker."""
from scripts import audio_common_prior_context_v87 as method
from scripts import audio_score_fusion_v83 as base

METHOD = method.METHOD
calibrate = method.calibrate
predict = method.predict


# The parallel study keeps v84's primary extraction, fitting and selection intact.
def study_paths():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    trial = root / 'backend/data/independent-evaluation/audio-common-reference-study-v88'
    return root, trial, root / 'docs/evaluation-panels/audio-common-reference-v88-freeze.json'


def study_bindings():
    from scripts import audio_combined_confidence_v84 as primary

    root, _, _ = study_paths()
    names = (
        'backend/scripts/audio_common_prior_context_v87.py', 'backend/scripts/audio_reference_comparison_v87.py',
        'backend/scripts/audio_common_prior_runtime_v87.py', 'backend/scripts/audio_parallel_dispatch_v87.py',
        'backend/tests/test_audio_common_prior_context_v87.py', 'backend/tests/test_audio_reference_comparison_v87.py',
        'docs/audio-common-reference-v87-protocol.md', 'backend/scripts/audio_common_prior_study_v88.py',
        'backend/scripts/audio_numeric_runtime_v88.py', 'backend/scripts/audio_parallel_dispatch_v88.py',
        'backend/tests/test_audio_numeric_runtime_v88.py', 'docs/audio-numeric-runtime-v88-protocol.md',
        'backend/scripts/audio_fusion_diagnostics_v86.py', 'backend/tests/test_audio_fusion_diagnostics_v86.py',
        'docs/audio-fusion-diagnostics-v86-protocol.md',
    )
    return {**primary.implementation(), **{name: primary.digest(root / name) for name in names}}


def prepare_study():
    from scripts import audio_combined_confidence_v84 as primary
    from scripts import audio_reference_comparison_v87 as comparison

    primary.admission(fresh=True)
    method.study_admission()
    prior_trial = primary.BASE / 'audio-common-reference-study-v87'
    stopped = primary.read(prior_trial / 'fit-profile/profile.json')
    base.require(stopped['child_started'] is False and stopped['child_exit_code'] is None
                 and stopped['guard'] == 'initial_headroom_unavailable' and not stopped['sample_errors']
                 and stopped['sampled_tree_rss_lower_bound_bytes'] == 0
                 and not (prior_trial / 'trained.json').exists(), 'Exact before-launch v87 resource stop required')
    primary.profile_pass(primary.TRIAL / 'train-profile/profile.json')
    root, trial, public = study_paths()
    parent_sha = primary.digest(primary.TRIAL / 'trained.json')
    base.require(parent_sha == '0aa21efdad351eebbc2390f241c49f1e40d55d188277db442f092c7b1ef668ae', 'Exact completed primary fitted/calibrated artifacts required')
    base.require(not any((primary.TRIAL / f'chunk-{index:02d}-profile').exists() for index in range(24, 48)),
                 'Parallel method must be fixed before any held-role extraction')
    base.require(not trial.exists() and not public.exists(), 'Never fitted parallel study required')
    code_bindings = study_bindings()
    trial.mkdir()
    lock = {'schema': 'chipmunk.audio-common-reference-freeze/88', 'parent_parallel_freeze_sha256': primary.digest(method.study_paths()[2]), 'parent_stopped_fit_profile_sha256': primary.digest(prior_trial / 'fit-profile/profile.json'), 'light_numeric_stage_only': True, 'initial_ram_headroom_bytes': 3221225472, 'rss_cap_bytes': 1610612736, 'neural_extraction_limits_unchanged': True, 'code_sha256': code_bindings,
            'primary_freeze_sha256': primary.digest(primary.PUBLIC), 'primary_trained_sha256': parent_sha,
            'primary_training_summary_sha256': primary.digest(primary.TRIAL / 'training-summary.json'),
            'method': METHOD, 'selection_rule': comparison.CHOICE,
            'development_calibration_reused': True, 'primary_method_and_selection_unchanged': True,
            'fixed_before_selection_and_confirmation_forwards': True, 'new_neural_calls_permitted': 0,
            'scope': 'Parallel calibration-reference ablation on the same neural responses/cohort. Added after primary fitting/calibration and before any held-role forward; not a second independent validation.'}
    primary.write(trial / 'freeze.json', lock)
    primary.write(public, lock)
    print({'parallel_method_fixed': True, 'held_roles_started': False, 'new_neural_calls': 0}, flush=True)


def study_admission():
    from scripts import audio_combined_confidence_v84 as primary

    primary.admission(fresh=True)
    _, trial, public = study_paths()
    lock, start = primary.read(trial / 'freeze.json'), primary.read(trial / 'execution-start.json')
    base.require(primary.digest(public) == primary.digest(trial / 'freeze.json') == start['freeze_sha256']
                 and start['remote_verified_before_parallel_calibration'] is True, 'Exact published parallel study freeze required')
    base.require(lock['code_sha256'] == study_bindings() and lock['method'] == METHOD
                 and primary.digest(primary.TRIAL / 'trained.json') == lock['primary_trained_sha256']
                 and primary.digest(primary.TRIAL / 'training-summary.json') == lock['primary_training_summary_sha256'],
                 'Parallel code, primary fits or reference method changed')
    prior_trial = primary.BASE / 'audio-common-reference-study-v87'
    base.require(primary.digest(method.study_paths()[2]) == lock['parent_parallel_freeze_sha256']
                 and primary.digest(prior_trial / 'fit-profile/profile.json') == lock['parent_stopped_fit_profile_sha256']
                 and not (prior_trial / 'trained.json').exists(), 'Original before-launch resource stop must remain intact')
    primary.profile_pass(trial.parent / (trial.name + '-preparation-profile-a03/profile.json'))
    return lock


def fit_study():
    from scripts import audio_combined_confidence_v84 as primary

    study_admission()
    _, trial, _ = study_paths()
    base.require(not any((primary.TRIAL / f'chunk-{index:02d}-profile').exists() for index in range(24, 48)),
                 'Parallel calibration must finish before held-role forwards')
    context = primary.read(primary.TRIAL / 'trained.json')['flatness_context']
    rows, bindings = primary.load_role('confidence_cal')
    artifact = calibrate(context, rows)
    primary.write(trial / 'trained.json', artifact)
    summary = {'schema': 'chipmunk.audio-common-reference-training/88', 'trained_sha256': primary.digest(trial / 'trained.json'),
               'source_stages': bindings, 'calibration_inputs_reused': len(rows), 'primary_fits_unchanged': True,
               'branches': {name: {key: value for key, value in detail.items()
                                   if key not in {'coefficients', 'calibration', 'conditional_ai_reference_prior'}}
                            for name, detail in artifact['branches'].items()},
               'held_role_inputs_used': 0, 'new_neural_calls': 0, 'defaults_changed': False}
    primary.write(trial / 'training-summary.json', summary)
    print({'branch_statuses': {name: detail['status'] for name, detail in artifact['branches'].items()},
           'calibration_inputs_reused': len(rows), 'held_role_inputs_used': 0}, flush=True)


def parallel_result(role):
    from scripts import audio_combined_confidence_v84 as primary
    from scripts import audio_reference_comparison_v87 as comparison

    study_admission()
    root, trial, _ = study_paths()
    primary.profile_pass(trial / 'fit-profile/profile.json')
    base.require(primary.digest(trial / 'trained.json') == primary.read(trial / 'training-summary.json')['trained_sha256'],
                 'Exact parallel calibrated artifact required')
    original_path = primary.TRIAL / ('selection.json' if role == 'selection' else 'result.json')
    primary_result = primary.read(original_path)
    if role == 'confirmation':
        receipt = primary.read(trial / 'confirmation-start.json')
        published = root / 'docs/evaluation-panels/audio-common-reference-v88-selection.json'
        base.require(receipt['remote_verified_before_confirmation_forwards'] is True
                     and primary.digest(trial / 'selection.json') == primary.digest(published) == receipt['selection_sha256'],
                     'Published parallel choice must precede final forwards')
    rows, bindings = primary.load_role(role)
    originals, added = primary.read(primary.TRIAL / 'trained.json'), primary.read(trial / 'trained.json')
    measured = comparison.evaluate(rows, originals, added)
    # The primary result is never overwritten or replaced by this additional comparison.
    expected_names = set(primary_result['evaluation']['totals'])
    base.require(all(measured['totals'][name] == primary_result['evaluation']['totals'][name] for name in expected_names),
                 'Parallel report must reproduce every original primary method count')
    choice = comparison.choose(measured, originals, added) if role == 'selection' else primary.read(trial / 'selection.json')['selected_arm']
    result = {'schema': f'chipmunk.audio-common-reference-{role}/88', 'freeze_sha256': primary.digest(trial / 'freeze.json'),
              'parallel_trained_sha256': primary.digest(trial / 'trained.json'), 'primary_report_sha256': primary.digest(original_path),
              'source_stages': bindings, 'primary_selected_arm': primary_result['selected_arm'], 'selected_arm': choice,
              'selection_rule': comparison.CHOICE, 'evaluation': measured, 'defaults_changed': False,
              'scope': 'Same fixed cohort and neural responses; parallel three-way calibration-reference comparison, not additional independent cases. Primary v84 selection remains intact. Class-only bins do not establish calibration. No population confidence or public threshold guarantee.'}
    name = 'selection' if role == 'selection' else 'result'
    primary.write(trial / (name + '.json'), result)
    primary.write(root / f'docs/evaluation-panels/audio-common-reference-v88-{name}.json', result)
    print({'selected_arm': choice, 'primary_selected_arm': primary_result['selected_arm'], 'role': role, 'cases': len(rows)}, flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'fit', 'select', 'confirm'))
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_study()
    elif args.command == 'fit':
        fit_study()
    else:
        parallel_result('selection' if args.command == 'select' else 'confirmation')
