import gzip
import json
import shutil
from pathlib import Path

import pytest

from scripts.review_mouth_paired_v9 import review

DOCS = Path(__file__).resolve().parents[2] / "docs"


def copy_public(tmp_path):
    target = tmp_path / "docs"
    panels = target / "evaluation-panels"
    panels.mkdir(parents=True)
    for name in ("mouth-learning-v8-evaluation.json", "mouth-paired-v9-evaluation.json"):
        shutil.copyfile(DOCS / name, target / name)
    for name in (
        "mouth-v8-review-responses.json.gz",
        "mouth-v8-head.json",
        "mouth-v8-training-bindings.json.gz",
        "mouth-v8-prefix-bindings.json.gz",
        "mouth-v9-review-responses.json.gz",
        "mouth-v9-head.json",
        "mouth-v9-fit-replay.json.gz",
    ):
        shutil.copyfile(DOCS / "evaluation-panels" / name, panels / name)
    return target


def test_three_arm_arithmetic_and_complete_source_pairs_are_publicly_reviewable():
    result = review(DOCS)
    assert result["source_matched_fit_pairs_checked"] == 768
    assert result["fitting_teacher_targets_checked"] == 4096
    assert result["counts"]["v8"]["localized_ai"] == 14
    assert result["counts"]["v9"]["planned"] == 88
    assert "Does not verify neural inference" in result["scope"]


def test_changed_teacher_float32_payload_is_detected(tmp_path):
    target = copy_public(tmp_path)
    path = target / "evaluation-panels/mouth-v9-fit-replay.json.gz"
    body = json.loads(gzip.decompress(path.read_bytes()))
    body["teacher_fit_targets"][0]["target"] += 0.2
    path.write_bytes(gzip.compress(json.dumps(body).encode()))
    with pytest.raises(ValueError, match="payload changed"):
        review(target)


def test_reserved_actor_teacher_offset_is_refused(tmp_path):
    target = copy_public(tmp_path)
    panels = target / "evaluation-panels"
    rows = json.loads(gzip.decompress((panels / "mouth-v8-training-bindings.json.gz").read_bytes()))["items"]
    reserved = next(r for r in rows if r["role"] == "point")
    bindings = json.loads(gzip.decompress((panels / "mouth-v8-prefix-bindings.json.gz").read_bytes()))[
        "bindings"
    ]
    offset = next(b for b in bindings if b["id"] == reserved["id"])["views"][0]["offset"]
    path = panels / "mouth-v9-fit-replay.json.gz"
    body = json.loads(gzip.decompress(path.read_bytes()))
    body["teacher_fit_targets"][0].update(id=reserved["id"], offset=offset, view=0)
    path.write_bytes(gzip.compress(json.dumps(body).encode()))
    with pytest.raises(ValueError, match="reserved"):
        review(target)
