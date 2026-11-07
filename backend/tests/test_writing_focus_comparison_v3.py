"""Writing localization denominators, context coverage and frozen decisions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import run_writing_focus_v3 as runner
from scripts import writing_focus_v3 as method


def provenance():
    return {"total_ai_characters": 8, "total_human_characters": 7,
            "foci": [{"human_characters": 5, "ai_characters": 0},
                     {"human_characters": 0, "ai_characters": 5},
                     {"human_characters": 2, "ai_characters": 3}]}


def test_inclusive_reference_and_mixed_character_denominators():
    values = method.counts(provenance(), [0.8, 0.7, 0.9], 0.8)
    assert values["located_bot_characters"] == 3
    assert values["highlighted_human_characters"] == 7
    assert values["highlighted_pure_human_foci"] == 1
    assert values["recovered_pure_bot_foci"] == 0
    assert values["highlighted_mixed_foci"] == 1
    assert method.rates(values)["character_iou"] == 3 / 15
    assert method.rates(values)["bot_character_recall"] == 3 / 8


def test_no_highlights_are_zero_recall_not_perfect_precision():
    values = method.rates(method.counts(provenance(), [0.1, 0.1, 0.1], 0.8))
    assert values["bot_character_precision"] is None
    assert values["bot_character_recall"] == 0
    assert values["character_iou"] == 0


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1, "0.9", None])
def test_invalid_responses_do_not_become_human_calls(value):
    with pytest.raises(ValueError):
        method.counts(provenance(), [0.1, value, 0.8], 0.5)


def test_missing_position_blocks_all_counts():
    with pytest.raises(ValueError):
        method.counts(provenance(), [0.9], 0.8)


def test_more_abstention_or_higher_rank_cannot_substitute_for_recovery():
    def arm(recovered, highlighted, recall):
        return {str(p): {"counts": {"recovered_pure_bot_foci": recovered},
                         "rates": {"pure_human_origin_highlight_rate": highlighted,
                                   "bot_character_recall": recall}} for p in (0.5, 0.8)}

    gates = method.development_gates(arm(10, 0.1, 0.6), arm(9, 0., 0.5), {"0.5": 0.1, "0.8": 0.1}, True)
    assert not gates["0.5_more_pure_bot_recoveries"]
    assert not gates["0.8_bot_character_retention"]
    gates = method.development_gates(arm(10, 0.1, 0.6), arm(11, 0.12, 0.6), {"0.5": 0.0, "0.8": 0.1}, False)
    assert gates["0.5_human_origin_highlights"]
    assert not gates["0.5_author_iou_gain"]
    assert not gates["complete_no_failures"]


class OverflowTokenizer:
    def __init__(self, windows):
        self.windows = windows

    def __call__(self, text, **kwargs):
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": [4, 5, 6, 7, 8]}
        return {"input_ids": [[0, *w, 2] for w in self.windows],
                "special_tokens_mask": [[1, *[0 for _ in w], 1] for w in self.windows]}


def test_overflow_preserves_full_token_order_and_native_specials():
    record = runner.token_identity("synthetic", OverflowTokenizer([[4, 5, 6], [7, 8]]))
    assert record["tokens"] == 5 and record["token_windows"] == 2
    assert record["token_ids_sha256"] == method.text_sha("[4,5,6,7,8]")


@pytest.mark.parametrize("windows", [[[4, 5, 6]], [[4, 5, 6], [6, 7, 8]], [[4, 6, 5], [7, 8]]])
def test_omitted_duplicated_or_reordered_tokens_refused(windows):
    with pytest.raises(ValueError):
        runner.token_identity("synthetic", OverflowTokenizer(windows))


def response_pair():
    source = {"id": "source", "author_group": "author", "extracted_text_sha256": "doc",
              "foci": [{"sentence": 0, "start": 2, "end": 4, "focus_sha256": "f", "context_sha256": "c"}]}
    focus = {**source["foci"][0], **{view: {"score": 0.9, "coverage": {"tokens": 3, "token_windows": 1,
              "token_ids_sha256": "a" * 64}} for view in ("context", "focus")}}
    response = {**{k: source[k] for k in ("id", "author_group", "extracted_text_sha256")},
                "status": "ok", "foci": [focus]}
    return source, response


@pytest.mark.parametrize("change", ["position", "hash", "failed", "missing", "bad_score", "no_coverage"])
def test_wrong_positions_failures_and_missing_checks_cannot_be_reused(change):
    source, response = response_pair()
    runner.validate_response(response, source)
    if change == "position":
        response["foci"][0]["end"] = 5
    elif change == "hash":
        response["foci"][0]["context_sha256"] = "wrong"
    elif change == "failed":
        response["status"] = "failed"
    elif change == "missing":
        response["foci"] = []
    elif change == "bad_score":
        response["foci"][0]["focus"]["score"] = float("nan")
    else:
        response["foci"][0]["focus"]["coverage"]["tokens"] = 0
    with pytest.raises(ValueError):
        runner.validate_response(response, source)


def test_changed_freeze_is_rejected_before_model_loading(tmp_path, monkeypatch):
    historical = tmp_path / "history.json"
    historical.write_text("unchanged source")
    monkeypatch.setattr(runner, "HISTORICAL_PASSAGES_SHA", method.digest(historical))
    monkeypatch.setattr(runner, "implementation", lambda: {"code": "frozen"})
    monkeypatch.setattr(runner, "runtime", lambda: {"torch": "fixed"})
    monkeypatch.setattr(runner, "installed", lambda: {"transformers": "fixed"})
    monkeypatch.setattr(runner, "assets", lambda path, expected: expected)
    monkeypatch.setattr(runner, "read_inputs", lambda path: ["no inference"])
    lock = {"schema": runner.SCHEMA, "method": copy.deepcopy(method.METHOD),
            "inputs": str(tmp_path / "inputs"), "historical": str(historical),
            "baseline": str(tmp_path), "candidate": str(tmp_path),
            "input_sha256": method.INPUT_SHA, "historical_sha256": method.digest(historical),
            "implementation_sha256": {"code": "frozen"}, "runtime": {"torch": "fixed"},
            "installed_sha256": {"transformers": "fixed"},
            "baseline_assets": method.BASELINE_ASSETS, "candidate_assets": method.CANDIDATE_ASSETS,
            "baseline_revision": method.BASELINE_REVISION, "candidate_revision": method.CANDIDATE_REVISION,
            "new_classifier_forwards": 0, "fitting_started": False, "serving_adoption_permitted": False}
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    assert runner.check(path)[1] == ["no inference"]
    for key, value in [("method", {}), ("runtime", {}), ("implementation_sha256", {}),
                       ("baseline_revision", "changed"), ("candidate_assets", {}),
                       ("fitting_started", True), ("serving_adoption_permitted", True)]:
        altered = {**lock, key: value}
        path.write_text(json.dumps(altered))
        with pytest.raises(ValueError):
            runner.check(path)
    assert not Path(tmp_path / "inputs").exists()


def test_direct_audit_retains_rewriting_classes_and_content_weights():
    class NativeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[0, 4, 5, 2], [0, 6, 2, 1]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
                    "special_tokens_mask": torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]]),
                    "overflow_to_sample_mapping": torch.tensor([0, 0])}

    class BaselineModel:
        def __call__(self, input_ids, attention_mask):
            # Window1 is equal across all4classes; window2 has a pure editing response.
            logits = [0., 0., 0., 0.] if input_ids[0, 1] == 4 else [-100., -100., 100., -100.]
            return {"logits": torch.tensor([logits])}

    clf = SimpleNamespace(tok=NativeTokenizer(), model=BaselineModel())
    assert runner.direct_response(clf, "baseline", "synthetic") == (0.75 * 2 + 1) / 3


def test_installed_freeze_pins_concrete_runtime_instead_of_compatibility_shim():
    pins = runner.installed()
    assert set(pins) == set(runner.INSTALLED)
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in pins.values())
