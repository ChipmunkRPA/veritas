"""Parallel calibration-reference ablation; same fixed context logits, new map only."""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts import audio_score_fusion_v83 as base

METHOD = {
    "version": "audio-context-common-reference/87",
    "base_score_fits": "Unchanged v83 context score coefficients and fitting-only cutoff",
    "change": "Calibrate context maps with full-assessed-calibration-role reference weights conditioned on the branch; do not rebalance truth/family/group/source again within the branch",
    "reference_prior": "0.5 AI over the full assessed calibration role; branch-conditional prior follows that same weighted reference population",
    "input_predictors": "Same two component logits and measured input quality; no truth/provider/source/condition at prediction",
    "ridge": 0.005, "slope_bounds": [0.0, 5.0], "intercept_bounds": [-10.0, 10.0],
    "solver": {"method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000},
    "minimum_groups_per_truth_per_branch": 3,
    "scope": "Development/calibration reuse before held-role extraction, not independent recalibration data or a population confidence guarantee",
}


def solve(z, y, weights):
    matrix = np.column_stack([z, np.ones(len(z))])
    weights = weights / weights.sum()

    def objective(beta):
        value = matrix @ beta
        penalty = METHOD["ridge"] * beta[0]
        return (float(weights @ (np.logaddexp(0, value) - y * value) + 0.5 * METHOD["ridge"] * beta[0] ** 2),
                matrix.T @ (weights * (expit(value) - y)) + np.array([penalty, 0.0]))

    solver = METHOD["solver"]
    fitted = minimize(objective, [1.0, 0.0], jac=True, method=solver["method"],
                      bounds=[tuple(METHOD["slope_bounds"]), tuple(METHOD["intercept_bounds"])],
                      options={key: solver[key] for key in ("ftol", "gtol", "maxiter")})
    base.require(fitted.success and np.isfinite(fitted.x).all(), "Common-reference calibration failed to converge")
    return fitted.x.tolist()


def calibrate(context, rows):
    base.require(context["method"] == base.METHOD and context["arm"] == "flatness_context",
                 "Exact original context score fits required")
    base.require(bool(rows) and all(row["role"] == "confidence_cal" for row in rows),
                 "Only assigned calibration sources may fit the parallel map")
    base.require(not (set(context["fit_group_ids"]) & {row["group"] for row in rows})
                 and not (set(context["fit_source_ids"]) & {row["base_id"] for row in rows}),
                 "Calibration cannot reuse a fitting group or source")
    selected = [row for row in rows if base.predictors(row) is not None]
    base.require(bool(selected), "Assessed independent calibration inputs required")
    common_weights = base.weights(selected)
    result = {"method": copy.deepcopy(METHOD), "routing_cutoff": context["routing_cutoff"], "branches": {},
              "fit_group_ids": context["fit_group_ids"].copy(), "fit_source_ids": context["fit_source_ids"].copy(),
              "calibration_group_ids": sorted({row["group"] for row in rows}),
              "calibration_source_ids": sorted({row["base_id"] for row in rows})}
    for name in ("low", "high"):
        original = context["branches"][name]
        if "coefficients" not in original:
            result["branches"][name] = {"status": "unsupported_fit"}
            continue
        indices = [index for index, row in enumerate(selected) if base.branch(row, context) == name]
        subset = [selected[index] for index in indices]
        support = base.support(subset)
        detail = {"status": "unsupported_calibration", "coefficients": original["coefficients"].copy(),
                  "calibration_groups_by_truth": support, "calibration_windows": len(subset)}
        if min(support.values()) >= METHOD["minimum_groups_per_truth_per_branch"]:
            z = np.asarray([np.r_[base.predictors(row), 1.0] @ detail["coefficients"] for row in subset])
            labels = np.asarray([row["label"] for row in subset], np.float64)
            w = common_weights[indices]
            detail.update(status="calibrated", calibration=solve(z, labels, w),
                          conditional_ai_reference_prior=float(w @ labels / w.sum()))
        result["branches"][name] = detail
    return result


def predict(row, artifact):
    base.require(artifact["method"] == METHOD, "Exact common-reference artifact required")
    features = base.predictors(row)
    if features is None:
        return {"decision": "unassessed", "probability_ai": None, "status": "unassessed"}
    value = row["quality_flatness"]
    base.require(not isinstance(value, bool) and np.isfinite(value) and 0 <= value <= 1,
                 "Actual measured input flatness required")
    name = "low" if value <= artifact["routing_cutoff"] else "high"
    detail = artifact["branches"][name]
    if detail["status"] != "calibrated":
        return {"decision": "unassessed", "probability_ai": None, "status": "unassessed", "branch": name}
    z = float(np.r_[features, 1.0] @ detail["coefficients"])
    p = float(expit(detail["calibration"][0] * z + detail["calibration"][1]))
    return {"status": "ok", "probability_ai": p, "decision": "ai" if p > 0.5 else "original" if p < 0.5 else "inconclusive",
            "branch": name}


# The parallel study keeps v84's primary extraction, fitting and selection intact.
def study_paths():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    trial = root / 'backend/data/independent-evaluation/audio-common-reference-study-v87'
    return root, trial, root / 'docs/evaluation-panels/audio-common-reference-v87-freeze.json'


def study_bindings():
    from scripts import audio_combined_confidence_v84 as primary

    root, _, _ = study_paths()
    names = ('backend/scripts/audio_common_prior_context_v87.py', 'backend/scripts/audio_reference_comparison_v87.py',
             'backend/scripts/audio_common_prior_runtime_v87.py', 'backend/scripts/audio_parallel_dispatch_v87.py',
             'backend/tests/test_audio_common_prior_context_v87.py', 'backend/tests/test_audio_reference_comparison_v87.py',
             'docs/audio-common-reference-v87-protocol.md')
    return {**primary.implementation(), **{name: primary.digest(root / name) for name in names}}


def prepare_study():
    from scripts import audio_combined_confidence_v84 as primary
    from scripts import audio_reference_comparison_v87 as comparison

    primary.admission(fresh=True)
    primary.profile_pass(primary.TRIAL / 'train-profile/profile.json')
    root, trial, public = study_paths()
    parent_sha = primary.digest(primary.TRIAL / 'trained.json')
    base.require(parent_sha == '0aa21efdad351eebbc2390f241c49f1e40d55d188277db442f092c7b1ef668ae', 'Exact completed primary fitted/calibrated artifacts required')
    base.require(not any((primary.TRIAL / f'chunk-{index:02d}-profile').exists() for index in range(24, 48)),
                 'Parallel method must be fixed before any held-role extraction')
    base.require(not trial.exists() and not public.exists(), 'Never fitted parallel study required')
    trial.mkdir()
    lock = {'schema': 'chipmunk.audio-common-reference-freeze/87', 'code_sha256': study_bindings(),
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
    primary.profile_pass(trial.parent / (trial.name + '-preparation-profile/profile.json'))
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
    summary = {'schema': 'chipmunk.audio-common-reference-training/87', 'trained_sha256': primary.digest(trial / 'trained.json'),
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
        published = root / 'docs/evaluation-panels/audio-common-reference-v87-selection.json'
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
    result = {'schema': f'chipmunk.audio-common-reference-{role}/87', 'freeze_sha256': primary.digest(trial / 'freeze.json'),
              'parallel_trained_sha256': primary.digest(trial / 'trained.json'), 'primary_report_sha256': primary.digest(original_path),
              'source_stages': bindings, 'primary_selected_arm': primary_result['selected_arm'], 'selected_arm': choice,
              'selection_rule': comparison.CHOICE, 'evaluation': measured, 'defaults_changed': False,
              'scope': 'Same fixed cohort and neural responses; parallel three-way calibration-reference comparison, not additional independent cases. Primary v84 selection remains intact. Class-only bins do not establish calibration. No population confidence or public threshold guarantee.'}
    name = 'selection' if role == 'selection' else 'result'
    primary.write(trial / (name + '.json'), result)
    primary.write(root / f'docs/evaluation-panels/audio-common-reference-v87-{name}.json', result)
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
