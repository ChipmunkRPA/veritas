"""Independent set-algebra replay of the audio decision combinations; no neural imports."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

PACKET_SHA256 = "84848472fef312c5fb5fd2ceddacd5fbe217caebadd09aacd85f28f5ee21cac3"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def review(packet: Path, result: Path) -> dict:
    payload = packet.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == PACKET_SHA256, "Changed original development source packet")
    body = json.loads(gzip.decompress(payload))
    rows = body["outcomes"]
    ids = {r["id"] for r in rows}
    require(len(ids) == len(rows) == 1449, "Missing or duplicated original input")
    assessed = {r["id"] for r in rows if r["status"] == "ok"}
    genuine = {r["id"] for r in rows if r["label"] == 0}
    synthetic = ids - genuine
    arms = {}
    for name, getter in (("current", lambda r: r["original_decision"]),
                         ("new_features_only", lambda r: r["candidates"]["standalone"]["decision"]),
                         ("features_plus_current", lambda r: r["candidates"]["complement"]["decision"])):
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

    saved = json.loads(result.read_text())
    require(saved["status"] == "complete" and saved["input_sha256"] == PACKET_SHA256, "Unbound or incomplete comparison")
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
    require(changes == saved["changes_vs_current"], "Actual newly gained/lost cases differ")
    return {"schema": "chipmunk.audio-combination-set-review/52", "status": "complete",
            "input_sha256": PACKET_SHA256, "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
            "inputs": len(ids), "methods": len(arms), "all_totals_strata_and_actual_gains_losses_exact": True,
            "unassessed": len(ids - assessed), "new_model_calls": 0,
            "reviewer": "Separate set algebra; no import of the original decision-combination implementation.",
            "scope": "Local cached-response arithmetic review; not outside review or fresh accuracy."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = review(args.packet, args.result)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
