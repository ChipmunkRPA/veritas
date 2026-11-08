"""Comparison receipts cannot bypass full gradients, geometry or reset agreement."""

import copy
import json

import pytest

from scripts.relayformer_checkpointed_probe_v24 import compare


def packet(arm):
    state = {f"authored-key-{i}": "a" * 64 for i in range(246)}
    one = {"loss": 1.25, "gradient_norm_before_clipping": 3.5, "binding": {"actual_patch_count": 4, "full_logit_float32_sha256": "b" * 64}, "gradient_sha256": dict(state), "updated_sha256": dict(state), "saved_activation_payload": {"packed_nonparameter_bytes": 1000 if arm == "baseline" else 100}}
    return {"schema": "chipmunk.authored-regional-checkpoint-comparison/24", "arm": arm, "completed_authored_forwards": 4, "completed_backward_passes": 2, "completed_optimizer_updates": 2, "project_neural_forwards": 0, "project_fitting_steps": 0, "serving_admitted": False, "source_geometry": {"source_size": [1024, 1024], "fit_size": [1024, 1024], "input_float32_sha256": "c" * 64}, "initial_selected_sha256": state, "frozen_state_sha256": {"authored-frozen": "d" * 64}, "passes": [copy.deepcopy(one), copy.deepcopy(one)], "exact_repeat": True, "starting_tensors_restored": True, "native_sigmoid_parity_exact": True, "checkpoint_trace": {"checkpointed_block_calls": 46, "body_entries": 92} if arm == "checkpointed" else None}


def files(tmp_path, a, b):
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    baseline.write_text(json.dumps(a))
    candidate.write_text(json.dumps(b))
    return baseline, candidate, tmp_path / "comparison.json"


def test_consistent_authored_receipt_schema_is_accepted_without_claiming_quality(tmp_path):
    a, b, output = files(tmp_path, packet("baseline"), packet("checkpointed"))
    compare(a, b, output)
    result = json.loads(output.read_text())
    assert result["complete_cross_arm_math_exact"] and result["packed_activation_payload_reduced_in_both_repeats"]
    assert result["project_neural_forwards"] == result["project_fitting_steps"] == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        compare(a, b, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize("fault", ["missing_gradient", "changed_gradient", "changed_update", "reduced_source", "missing_repeat", "no_recomputation", "unfinished_update", "negative_payload", "hidden_source_forward", "changed_frozen_base", "both_nonrepeat"])
def test_incomplete_or_resealed_math_receipts_are_refused(tmp_path, fault):
    a, b = packet("baseline"), packet("checkpointed")
    if fault == "missing_gradient":
        b["passes"][0]["gradient_sha256"].pop("authored-key-0")
    elif fault == "changed_gradient":
        for row in b["passes"]:
            row["gradient_sha256"]["authored-key-0"] = "f" * 64
    elif fault == "changed_update":
        for row in b["passes"]:
            row["updated_sha256"]["authored-key-0"] = "f" * 64
    elif fault == "reduced_source":
        b["source_geometry"]["fit_size"] = [512, 512]
    elif fault == "missing_repeat":
        b["passes"].pop()
    elif fault == "no_recomputation":
        b["checkpoint_trace"] = {"checkpointed_block_calls": 0, "body_entries": 0}
    elif fault == "unfinished_update":
        b["completed_optimizer_updates"] = 1
    elif fault == "negative_payload":
        b["passes"][0]["saved_activation_payload"]["packed_nonparameter_bytes"] = -1
    elif fault == "hidden_source_forward":
        b["project_neural_forwards"] = 1
    elif fault == "changed_frozen_base":
        b["frozen_state_sha256"]["authored-frozen"] = "f" * 64
    else:
        a["passes"][1]["loss"] = b["passes"][1]["loss"] = 2.0
    baseline, candidate, output = files(tmp_path, a, b)
    with pytest.raises(ValueError):
        compare(baseline, candidate, output)
    assert not output.exists()
