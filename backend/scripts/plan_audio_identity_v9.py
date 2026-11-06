"""Repair source roles from publisher identities before any candidate fitting.

Keep the fixed fresh recordings. Retire connected consumed sources instead of
silently substituting fresh cases. Preserve exact recipes for unchanged roles;
changed-role sources must use the destination role's independent backgrounds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audio_source_identity_v9 import connected_groups, digest, role_violations, speaker_key
from scripts.plan_audio_multisource_v8 import roles


def repair(foreground, backgrounds):
    by_id = {r["id"]: r for r in foreground}
    if len(by_id) != len(foreground):
        raise ValueError("Every original foreground identity must be unique")
    fresh = [r for r in foreground if r["role"] == "confirmation"]
    if not fresh:
        raise ValueError("Keep an explicitly fixed confirmation source set")
    if any(r["foreground_background_shared"] for r in role_violations(foreground + backgrounds)):
        raise ValueError("Known foreground/background speaker connection needs a separate prospective repair")
    consumed, retired = [], []
    for group in connected_groups(foreground):
        members = [by_id[key] for key in group["sources"]]
        if any(r["role"] == "confirmation" for r in members):
            retired.extend(
                {
                    **r,
                    "retirement_reason": "Known source/voice/text component touches the fixed confirmation recordings",
                }
                for r in members
                if r["role"] != "confirmation"
            )
        else:
            consumed.extend(members)
    consumed_groups = connected_groups(consumed)
    assignment = roles(consumed_groups)
    for group in consumed_groups:
        group["role"] = assignment[group["group"]]
    fresh_groups = connected_groups(fresh)
    for group in fresh_groups:
        group["role"] = "confirmation"
    groups = consumed_groups + fresh_groups
    memberships = {source: (g["role"], g["group"]) for g in groups for source in g["sources"]}
    selected = []
    for original in consumed + fresh:
        role, group = memberships[original["id"]]
        selected.append(
            {
                **original,
                "prior_role": original["role"],
                "prior_group": original["group"],
                "role": role,
                "group": group,
                "canonical_speaker": speaker_key(original),
                "recipe_policy": "retain-exact-v8"
                if role == original["role"]
                else "regenerate-with-destination-role-backgrounds",
            }
        )
    if role_violations(selected + backgrounds):
        raise ValueError("Canonical speakers still cross roles or statistical groups after repair")
    if {r["sha256"] for r in fresh} != {r["sha256"] for r in selected if r["role"] == "confirmation"}:
        raise ValueError("Do not replace or drop fixed confirmation recordings")
    if {r["id"] for r in selected} & {r["id"] for r in retired} or len(selected) + len(retired) != len(
        foreground
    ):
        raise ValueError("Every original must remain accounted for exactly once")
    return selected, sorted(groups, key=lambda r: r["group"]), retired


def plan(old_root, output):
    old_path = old_root / "prepared-sources.json"
    audit_path = old_root / "source-planning/cross-corpus-identity-audit.json"
    body, audit = (json.loads(p.read_text()) for p in (old_path, audit_path))
    if (
        audit["prepared_sources_sha256"] != digest(old_path)
        or audit["source_decode_failures"]
        or not audit["reserved_recording_novelty_valid"]
    ):
        raise ValueError(
            "Complete historical source/text/audio audit must pass before reusing fixed recordings"
        )
    if not audit["v8_extraction_stopped"] or audit["v8_fitting_started"]:
        raise ValueError("Preserve the stopped pre-fitting v8 run")
    if audit["code_sha256"] != digest(Path(__file__).with_name("audio_source_identity_v9.py")):
        raise ValueError("Historical/canonical identity audit implementation changed")
    if any(old_root.rglob("fit.json")) or any(old_root.rglob("head.json")):
        raise ValueError("A candidate was already fitted; do not call the role repair prospective")
    selected, groups, retired = repair(body["sources"], body["backgrounds"])
    backgrounds = [{**r, "canonical_speaker": speaker_key(r)} for r in body["backgrounds"]]
    result = {
        "schema": "chipmunk.audio-canonical-identity-source-plan/9",
        "old_prepared_sources_sha256": digest(old_path),
        "identity_audit_sha256": digest(audit_path),
        "planning_code_sha256": digest(Path(__file__)),
        "identity_code_sha256": audit["code_sha256"],
        "original_source_plan_root": str(old_root.resolve()),
        "sources": selected,
        "backgrounds": backgrounds,
        "groups": groups,
        "retired_consumed_sources": retired,
        "role_counts": dict(Counter(r["role"] for r in selected)),
        "role_group_counts": dict(Counter(r["role"] for r in groups)),
        "role_generator_counts": {
            role: dict(Counter(r["generator"] for r in selected if r["role"] == role))
            for role in ("fit", "cal", "development", "confirmation")
        },
        "role_changes": [
            {"source_sha256": r["sha256"], "from": r["prior_role"], "to": r["role"]}
            for r in selected
            if r["role"] != r["prior_role"]
        ],
        "freshness": "Fixed new recording/script panel; 12 Kokoro profiles and two cloned/two genuine LibriSpeech identities occurred in prior project evaluations. Do not call every voice previously unseen. Corrected current fitting/calibration/development voices are separated from confirmation.",
        "recipe_policy": "Retain every exact v8 recipe when its source role is unchanged. For a changed source role only, use destination-role pools sorted by the original v8 stable hash, select each kind by SHA256(v9-background/source-id/kind) modulo pool size, and keep the v8 source/kind offset rule, mixer, four-second prefix and common gain. No detector responses select or regenerate cases.",
        "learner_policy": "Use the identical frozen v8.0 learner, PCA16/ridges, independent calibration, dual stability and actual service comparator. All16 advancement gates remain unchanged; denominator counts follow this source plan. No model or threshold sweep.",
        "fitting_started": False,
        "new_neural_inference_run": False,
        "deployment_eligible": False,
        "limitations": body["limitations"]
        + [
            "The earlier v8 literal dataset-key separation was invalid; the original run was stopped before fitting, with all1882 extraction records retained.",
            "Known cloned and genuine LibriSpeech IDs share a canonical parent. Publisher metadata is not biometric verification or proof of absence from pretrained weights.",
            "Prior voice exposure is disclosed separately from new recording/script eligibility. Known generic provider-name aliases are not independent biometric identity evidence.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "plan_sha256": digest(output),
                "role_counts": result["role_counts"],
                "role_groups": result["role_group_counts"],
                "retired_consumed_sources": len(retired),
                "role_changes": len(result["role_changes"]),
                "fresh_sources_retained": sum(r["role"] == "confirmation" for r in selected),
                "fitting_started": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan(args.old_root, args.output)
