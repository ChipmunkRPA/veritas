"""Data-only review of v114 source states; no inference or accuracy estimate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import av_source_encoder_v114 as source
from scripts.review_video_native_inserts_v96 import Budget

read, write, sha, require = source.read, source.write, source.sha, source.require


def review_source(row, plan, expected_file):
    require(
        all(row[key] == plan[key] for key in ("asset_index", "source_group", "study_role", "mono_wav_sha256"))
        and row["encoder_routing_qualified"] is True
        and row["generator_conditioning_qualified"] is False
        and row["hidden_shape"] == [1500, 5, 384]
        and row["proof"]["layer_readout_matches"] is True,
        "Complete unchanged source/role/state qualification required",
    )
    require(
        Path(row["hidden_file"]).resolve() == expected_file.resolve()
        and row["hidden_file_sha256"] == sha(expected_file),
        "Exact exclusive cached-state file required",
    )
    hidden = np.load(expected_file, allow_pickle=False)
    require(
        hidden.shape == (1500, 5, 384)
        and hidden.dtype == np.float32
        and np.isfinite(hidden).all()
        and source.speech.fingerprint(hidden) == row["proof"]["full_hidden_sha256"],
        "Complete finite saved native states required",
    )
    audit = source.audit_chunks(hidden, plan["frames"])
    require(all(row[k] == v for k, v in audit.items()), "Complete native coordinate counts required")
    return audit


def main():
    source.verify()
    output = source.TRIAL / "review-v115"
    output.mkdir(exist_ok=False)
    budget = Budget(output)
    budget.observe(initial=True)
    records = source.source_records()
    checked, batches, roles = [], [], Counter()
    with (output / "reviewed-sources.jsonl").open("x") as journal:
        for batch in range(8):
            folder = source.TRIAL / f"extract-{batch:02d}"
            profile, result = read(folder / "profile.json"), read(folder / "output/result.json")
            require(
                profile["child_exit_code"] == 0
                and profile["guard"] is None
                and profile["sample_errors"] == 0
                and result["status"] == "complete"
                and result["batch"] == batch
                and result["sources"] == 6
                and result["new_neural_calls"] == 7
                and result["freeze_sha256"] == sha(source.TRIAL / "freeze.json")
                and profile["runtime_sha256"] == sha(Path(source.__file__))
                and profile["worker_receipt_sha256"] == sha(folder / "output/result.json")
                and profile["resource_journal_sha256"] == sha(folder / "resources.jsonl")
                and result["source_journal_sha256"] == sha(folder / "output/source-states.jsonl"),
                "Successful exact complete source stage required",
            )
            rows = source.lines(folder / "output/source-states.jsonl")
            require(len(rows) == 6, "No omitted or substituted sources")
            counts = []
            for offset, (row, (plan, _)) in enumerate(
                zip(rows, records[batch * 6 : (batch + 1) * 6], strict=True)
            ):
                audit = review_source(row, plan, source.CACHE / f"batch-{batch:02d}/source-{offset:02d}.npy")
                counts.append(audit)
                roles[plan["study_role"]] += 1
                checked.append(audit)
                journal.write(
                    json.dumps(
                        {"asset_index": plan["asset_index"], "study_role": plan["study_role"], **audit}
                    )
                    + "\n"
                )
                journal.flush()
                budget.observe()
            require(
                all(
                    sum(r[k] for r in counts) == result[k]
                    for k in ("routed_frames", "explicit_zero_tick_slots")
                ),
                "Exact batch routing aggregates required",
            )
            batches.append(
                {
                    "batch": batch,
                    "sources": 6,
                    "new_neural_calls": 7,
                    "result_sha256": sha(folder / "output/result.json"),
                    "profile_sha256": sha(folder / "profile.json"),
                    "source_journal_sha256": sha(folder / "output/source-states.jsonl"),
                    "duration_s": profile["duration_s"],
                    "sampled_tree_rss_lower_bound_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
                    "minimum_available_ram_bytes": profile["minimum_available_ram_bytes"],
                }
            )
    lock = read(source.PUBLIC)
    require(
        len(checked) == 48
        and dict(roles) == lock["role_counts"]
        and sum(r["routed_frames"] for r in checked) == 3642,
        "Complete original panel required",
    )
    budget.observe()
    budget.stream.close()
    result = {
        "schema": "chipmunk.av-source-encoder-review/115",
        "status": "complete",
        "sources": 48,
        "role_counts": dict(roles),
        "routed_frames": 3642,
        "explicit_zero_tick_slots": sum(r["explicit_zero_tick_slots"] for r in checked),
        "bitwise_source_repeat_pairs": 8,
        "new_neural_calls_in_extraction": 56,
        "new_neural_calls_in_review": 0,
        "generator_conditioning_qualified_sources": 0,
        "freeze_sha256": sha(source.PUBLIC),
        "batches": batches,
        "private_review_journal_sha256": sha(output / "reviewed-sources.jsonl"),
        "review_code_sha256": sha(Path(__file__)),
        "review_resource_journal_sha256": sha(output / "resources.jsonl"),
        "review_duration_s": budget.rows[-1]["elapsed_s"],
        "review_maximum_observed_self_rss_bytes": max(r["combined_rss_bytes"] for r in budget.rows),
        "defaults_changed": False,
        "scope": "Data-only saved-state/source/role/routing review. It does not reproduce neural inference, establish finite temporal locality, measure achieved mouth alignment, evaluate authenticity or train a detector. Private voice-derived features are not published.",
    }
    write(output / "result.json", result)
    write(source.PANEL / "av-source-encoder-result-v115.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
