import json

import pytest

from scripts import review_text_gohumanize_transfer_v198 as review


def response(case, primary, current, interval, *, shared=False):
    proposals = (primary, current, interval)
    return {
        "case": case,
        "status": "measured",
        "primary_current": primary,
        "canonical_current": current,
        "canonical_interval": interval,
        "canonical_shared_current": shared,
        "native_proofs": [{"present": True} if group is not None else None for group in proposals],
    }


def source(case, raw, ai, *, arm="short_ai", generator="gpt-4o-mini", group="fixture-group"):
    return {
        "case": case,
        "status": "constructed" if raw is not None else "unavailable",
        "text": raw,
        "text_sha256": review.text_digest(raw) if raw is not None else None,
        "ai_spans": ai,
        "generator": generator,
        "article_group": group,
        "arm": arm,
    }


def proposal(start, end, score=0.5):
    return {"start": start, "end": end, "score": score}


def test_primary_and_canonical_map_original_unicode_codepoints():
    raw = " \tAé🙂\r\n \t\r\nB\u00a0e\u0301\u2028Z "
    primary = review.view(raw, paragraphs=True)
    canonical = review.view(raw, paragraphs=False)
    assert primary == review.View("Aé🙂\n\nB e\u0301 Z", (2, 3, 4, None, None, 11, None, 13, 14, None, 16))
    assert canonical == review.View("Aé🙂 B e\u0301 Z", (2, 3, 4, None, 11, None, 13, 14, None, 16))
    assert review.view("A\rB\n \n\n\t\n\nC", paragraphs=True).text == "A B\n\nC"

    mask = review.project(primary, [proposal(2, 7), proposal(7, 9, 0.49)], 0.5)
    assert mask == review.Mask(frozenset({4, 11}), frozenset({4, 11, 13, 14}))
    with pytest.raises(ValueError, match="Unicode view"):
        review.project(primary, [proposal(0, 12)], 0.5)


def test_missing_view_is_unassessed_but_empty_response_is_assessed():
    row = source("a", "ab\n\ncd", [[0, 2]])
    saved = response("a", [proposal(0, 1)], [], None)
    metrics = review.measurements(row, saved, {"current": 0.5, "interval": 0.6})
    assert tuple(metrics) == review.METHODS
    assert metrics["baseline"]["character_iou"] == 0.5
    assert metrics["baseline"]["recovered_ai_intervals"] == 1
    assert metrics["baseline"]["unscored_ai_characters"] == 1
    assert metrics["canonical_current"]["ai_document_with_no_ai_located"] is True
    assert metrics["canonical_current"]["unscored_ai_characters"] == 2
    assert (
        metrics["canonical_union"]
        == metrics["baseline_plus_canonical_union"]
        == {
            "status": "unassessed",
            "ai_characters": 2,
            "human_characters": 2,
        }
    )


def test_shared_views_must_reuse_both_scores_and_binding():
    row = source("a", "é   x", [[0, 1]])
    saved = response("a", [proposal(0, 1)], [proposal(2, 3)], [], shared=True)
    with pytest.raises(ValueError, match="view sharing"):
        review.measurements(row, saved, {"current": 0.5, "interval": 0.5})


def test_partial_role_coverage_keeps_planned_truth_and_assessed_denominators_distinct():
    rows = [
        source("a", "ab\n\ncd", [[0, 2]]),
        source("b", "🙂h", [[0, 1]], generator="deepseek-chat"),
        source("c", None, None),
    ]
    saved = [
        response("a", [proposal(0, 1)], [], [proposal(1, 2, 0.6)]),
        {"case": "b", "status": "unassessed", "reason": "extraction_correspondence"},
        {"case": "c", "status": "preparation_unavailable"},
    ]
    all_metrics = review.aggregate(rows, saved, {"current": 0.5, "interval": 0.6})
    all_mixed = all_metrics["all"]["short_ai"]["baseline_plus_canonical_union"]
    assert all_mixed == {
        "ok": 1,
        "unassessed": 1,
        "preparation_unavailable": 1,
        "planned": 3,
        "ai_characters": 3,
        "human_characters": 3,
        "located_ai_characters": 2,
        "highlighted_human_characters": 0,
        "missed_ai_characters": 0,
        "unscored_ai_characters": 0,
        "unscored_human_characters": 2,
        "mean_character_iou_assessed": 1.0,
        "useful_ai_location": 1,
        "known_ai_intervals": 1,
        "recovered_ai_intervals": 1,
        "genuine_document_flagged": 0,
        "ai_document_with_no_ai_located": 0,
        "genuine_article_groups_flagged": 0,
    }
    assert all_metrics["gpt-4o-mini"]["short_ai"]["baseline"]["planned"] == 2
    assert all_metrics["deepseek-chat"]["short_ai"]["baseline"]["mean_character_iou_assessed"] is None
    with pytest.raises(ValueError, match="planned case"):
        review.aggregate(rows, saved[:2], {"current": 0.5, "interval": 0.6})


def test_two_genuine_flags_in_same_article_count_one_source_group():
    rows = [source(case, "xy", [], arm="human") for case in ("a", "b")]
    one = [proposal(0, 1)]
    saved = [response(row["case"], one, one, [], shared=True) for row in rows]
    baseline = review.aggregate(rows, saved, {"current": 0.5, "interval": 0.6})["all"]["human"]["baseline"]
    assert baseline["genuine_document_flagged"] == 2
    assert baseline["genuine_article_groups_flagged"] == 1


def test_guarded_batch_rejects_truncated_or_uncommitted_inputs(tmp_path, monkeypatch):
    stage = tmp_path / "extract-00"
    output = stage / "output"
    output.mkdir(parents=True)
    monkeypatch.setattr(review, "TRIAL", tmp_path)
    monkeypatch.setattr(review, "PAIRS_PER_BATCH", 1)
    rows = [{"case": f"test-{arm}"} for arm in review.ARMS]
    samples = [
        {
            "elapsed_s": 0.1,
            "tree_rss_bytes": 0,
            "ram_available_bytes": 7 * review.GIB,
            "disk_free_bytes": 4 * review.GIB,
            "archive_free_disk_bytes": 4 * review.GIB,
            "violations": [],
        },
        {
            "elapsed_s": 1.0,
            "tree_rss_bytes": review.GIB,
            "ram_available_bytes": 2 * review.GIB,
            "disk_free_bytes": 2 * review.GIB,
            "archive_free_disk_bytes": 2 * review.GIB,
            "violations": [],
        },
    ]

    def jsonl(records):
        return b"".join((json.dumps(row) + "\n").encode() for row in records)

    resource_journal = jsonl(samples)
    (stage / "resources.jsonl").write_bytes(resource_journal)
    profile = {
        "schema": "chipmunk.video-appearance-resource-profile/108",
        "command": "extract",
        "batch": 0,
        "child_started": True,
        "child_exit_code": 0,
        "guard": None,
        "sample_errors": 0,
        "error_types": [],
        "resource_limit_observations": [],
        "runtime_sha256": "script",
        "new_neural_calls": 3,
        "duration_s": 1.1,
        "minimum_available_ram_bytes": 2 * review.GIB,
        "sampled_tree_rss_lower_bound_bytes": review.GIB,
        "resource_journal_sha256": review.digest(resource_journal),
    }
    result = {
        "status": "complete",
        "batch": 0,
        "planned_cases": len(rows),
        "freeze_sha256": "freeze",
        "actual_forwards": {"encoder": 1, "current_classifier": 2},
        "new_neural_calls": 3,
    }

    def record(responses):
        journal = jsonl(responses)
        (output / "private-responses.jsonl").write_bytes(journal)
        result_data = json.dumps({**result, "journal_sha256": review.digest(journal)}).encode()
        (output / "result.json").write_bytes(result_data)
        profile_data = json.dumps({**profile, "worker_receipt_sha256": review.digest(result_data)}).encode()
        (stage / "profile.json").write_bytes(profile_data)
        return review.digest(profile_data)

    profile_hash = record(rows)
    assert review.verify_batch(0, rows, "freeze", "script") == (rows, result["actual_forwards"], profile_hash)
    record(rows[:-1])
    with pytest.raises(ValueError, match="omitted"):
        review.verify_batch(0, rows, "freeze", "script")
    record(rows)
    (output / "private-responses.jsonl").write_bytes(jsonl(reversed(rows)))
    with pytest.raises(ValueError, match="commitment changed"):
        review.verify_batch(0, rows, "freeze", "script")


def test_readout_comparison_ignores_formatting_and_keeps_json_types():
    assert review.exact_json({"b": 3, "a": 1}) == review.exact_json({"a": 1, "b": 3})
    assert review.exact_json({"a": True}) != review.exact_json({"a": 1})
