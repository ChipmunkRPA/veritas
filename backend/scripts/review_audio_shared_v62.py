"""Independent set-algebra replay of the audio decision combinations; no neural imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()



def require(ok, message):
    if not ok:
        raise ValueError(message)


def review(trial: Path, result: Path) -> dict:
    frozen = json.loads((trial / "freeze.json").read_text())
    saved = json.loads(result.read_text())
    require(saved["freeze_sha256"] == sha(trial / "freeze.json"), "Changed prospective freeze")
    require(all(sha(ROOT / path) == value for path, value in frozen["code_sha256"].items()), "Frozen implementation changed")
    require(all(sha(ROOT / path) == value for path, value in frozen["artifacts_sha256"].items()), "Frozen source/fitting artifacts changed")
    manifest = BASE / "audio-v9/confirmation/variants.json"
    require(sha(manifest) == frozen["inputs_sha256"], "Changed reserve input manifest")
    sources = json.loads(manifest.read_text())["variants"]
    rows = []
    for index in range(12):
        path = trial / f"chunk-{index:02d}.jsonl"
        receipt = json.loads((trial / f"chunk-{index:02d}-complete.json").read_text())
        profile = json.loads((trial / f"chunk-{index:02d}-profile/profile.json").read_text())
        require(receipt == {"completed": 90, "journal_sha256": sha(path)}
                and saved["journal_sha256"][path.name] == sha(path), "Missing/altered chunk")
        require(profile["child_exit_code"] == 0 and profile["guard"] is None
                and profile["sample_errors"] == 0 and not profile["resource_limit_observations"], "Failed bounded stage")
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    require(len(rows) == len(sources) == 1080, "Missing original input")
    for row, source in zip(rows, sources, strict=True):
        require(all(row[key] == source[key] for key in ("id", "base_id", "group", "role", "label", "condition", "generator", "sha256", "pcm_sha256")),
                "Result source, truth, role or condition rebound")
        require(row["status"] in {"ok", "not_applicable"}, "Failed input cannot be dropped or treated as inconclusive")
        if row["status"] == "not_applicable":
            require(row["raw"] is None and row["wavlm"] is None
                    and row["original_decision"] == row["candidate_decision"] == "inconclusive"
                    and all(arm["decision"] == "inconclusive" for arm in row["candidates"].values()),
                    "Skipped check cannot carry directional evidence")
    ids = {r["id"] for r in rows}
    require(len(ids) == 1080, "Duplicate source input")
    assessed = {r["id"] for r in rows if r["status"] == "ok"}
    genuine = {r["id"] for r in rows if r["label"] == 0}
    synthetic = ids - genuine
    arms = {}
    for name, getter in (("current", lambda r: r["original_decision"]),
                         ("new_features_only", lambda r: r["candidates"]["standalone"]["decision"]),
                         ("features_plus_current", lambda r: r["candidates"]["complement"]["decision"]),
                         ("v7", lambda r: r["candidate_decision"])):
        arms[name] = ({r["id"] for r in rows if getter(r) == "ai"} & assessed,
                      {r["id"] for r in rows if getter(r) == "original"} & assessed)
    current, feature, complement = (arms[name] for name in ("current", "new_features_only", "features_plus_current"))
    ca, co = current
    ba, bo = complement
    fa, fo = feature
    ci, bi = assessed - ca - co, assessed - ba - bo
    pair_a, pair_o = ca | ba, co | bo
    all_a, all_o = pair_a | fa, pair_o | fo
    arms.update({
        "pair_conflict_abstain": (pair_a - pair_o, pair_o - pair_a),
        "current_then_complement": (ca | (ci & ba), co | (ci & bo)),
        "complement_then_current": (ba | (bi & ca), bo | (bi & co)),
        "pair_any_ai": (pair_a, pair_o - pair_a),
        "pair_agreement_only": (ca & ba, co & bo),
        "three_conflict_abstain": (all_a - all_o, all_o - all_a),
        "three_any_ai": (all_a, all_o - all_a),
    })
    require(all(not (a & o) and not ((a | o) - assessed) for a, o in arms.values()), "Invalid or unassessed directional assignment")

    def counts(indices, pair):
        a, o = pair
        part = [r for r in rows if r["id"] in indices]
        ai_found, genuine_correct = len(a & indices & synthetic), len(o & indices & genuine)
        false_ai, false_original = len(a & indices & genuine), len(o & indices & synthetic)
        missing = len(indices - assessed)
        withheld = len((indices & assessed) - a - o)
        return {"inputs": len(indices), "source_files": len({r["base_id"] for r in part}),
                "connected_groups": len({r["group"] for r in part}), "ai_inputs": len(indices & synthetic),
                "genuine_inputs": len(indices & genuine), "ai_found": ai_found, "genuine_correct": genuine_correct,
                "false_ai": false_ai, "false_original": false_original, "inconclusive_assessed": withheld,
                "unassessed": missing, "genuine_groups_flagged": len({r["group"] for r in part if r["id"] in a & genuine}),
                "without_direction": missing + withheld, "ai_without_ai_conclusion": len(indices & synthetic) - ai_found}

    for row in rows:
        expected = {name: "unassessed" if row["id"] not in assessed else "ai" if row["id"] in a
                    else "original" if row["id"] in o else "inconclusive" for name, (a, o) in arms.items()}
        require(row["decisions"] == expected, "Saved per-case combination differs from independent set algebra")
    require(saved["combined_probability"] is None and saved["serving_changed"] is False, "Unqualified service/probability claim")
    totals = {name: counts(ids, pair) for name, pair in arms.items()}
    require(totals == saved["totals"], "Standalone or combined operational counts differ")
    strata = {}
    for field in ("condition", "generator"):
        strata[field] = {value: {name: counts({r["id"] for r in rows if r[field] == value}, pair)
                                 for name, pair in arms.items()} for value in {r[field] for r in rows}}
    require(strata == saved["strata"], "Condition or generator outcomes differ")
    changes = {}
    old_correct = (ca & synthetic) | (co & genuine)
    for name, (a, o) in arms.items():
        correct = (a & synthetic) | (o & genuine)
        changes[name] = {"new_correct": len(correct - old_correct), "lost_correct": len(old_correct - correct),
                         "new_ai_hits": len((a - ca) & synthetic), "lost_ai_hits": len((ca - a) & synthetic),
                         "new_false_ai": len((a - ca) & genuine), "removed_false_ai": len((ca - a) & genuine),
                         "new_false_original": len((o - co) & synthetic), "removed_false_original": len((co - o) & synthetic)}
    require(changes == saved["changes_from_current"], "Actual newly gained/lost cases differ")
    return {"schema": "chipmunk.audio-shared-set-review/62", "status": "complete",
            "freeze_sha256": sha(trial / "freeze.json"), "result_sha256": sha(result),
            "inputs": len(ids), "methods": len(arms), "all_totals_strata_and_actual_gains_losses_exact": True,
            "unassessed": len(ids - assessed), "new_model_calls": 0,
            "reviewer": "Separate set algebra; no import of the original decision-combination implementation.",
            "scope": "Local independent arithmetic/source-binding review of fresh model outputs. Not an outside human review, additional neural replay or general accuracy guarantee."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = review(args.trial, args.result)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
