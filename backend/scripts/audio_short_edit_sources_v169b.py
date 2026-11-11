"""Replace only the pending stock shortlist before any ordinary-edit scoring."""

from __future__ import annotations

import argparse
import json

from scripts import audio_short_edit_sources_v169 as prior

OUTPUT = prior.BASE / "audio-short-edit-sources-v169b"
PUBLIC = prior.ROOT / "docs/evaluation-panels/audio-short-edit-source-plan-v169b.json"
METHOD = prior.ROOT / "docs/audio-short-edit-source-plan-v169b.md"


def stock_shortlists(rows, claimed):
    claimed = set(claimed)
    voices = sorted({row["voice"] for row in rows}, key=prior.stable)
    groups = []
    for voice in voices:
        shortlist = []
        ordered = sorted(
            (row for row in rows if row["voice"] == voice), key=lambda row: prior.stable(row["id"])
        )
        for row in ordered:
            keys = prior.identities(row) - {f"voice:{voice}"}
            if keys & claimed:
                continue
            shortlist.append(row)
            claimed.update(keys)
            if len(shortlist) == 6:
                break
        if shortlist:
            index = len(groups)
            groups.append(
                {
                    "lane": "stock",
                    "group_id": f"stock-{index:02d}",
                    "role": prior.ROLES[index % len(prior.ROLES)],
                    "required": 1,
                    "ordered_candidates": shortlist,
                    "status": "pending_exact_production_decode",
                }
            )
    prior.require(
        len(prior.connected([row for group in groups for row in group["ordered_candidates"]])) == len(groups),
        "Stock candidate and fallback edges must form separate voice components",
    )
    return groups


def freeze():
    prior.require(not OUTPUT.exists() and not PUBLIC.exists(), "Preserve the initial successor freeze")
    previous = json.loads(prior.PUBLIC.read_text())
    roster_path = prior.OUTPUT / "private-roster.json"
    original = json.loads(roster_path.read_text())
    prior.require(
        previous["private_roster_sha256"] == prior.sha(roster_path)
        and previous["method_sha256"] == prior.sha(prior.METHOD)
        and previous["implementation_sha256"]
        == prior.sha(prior.ROOT / "backend/scripts/audio_short_edit_sources_v169.py")
        and previous["model_calls"] == 0
        and previous["holdout_scored"]
        is original["holdout_scored"]
        is original["exact_production_decode_complete"]
        is False,
        "The exact unscored prior roster and source method are required",
    )
    meta = prior.Metadata()
    history, ids, contexts = prior.exclude_history(meta)
    genuine = prior.genuine(meta, history, ids, contexts)
    lanes = {
        "genuine": genuine,
        **prior.synthetic(meta, history, ids, contexts, {row["voice"] for row in genuine}),
    }
    backgrounds = prior.noise(meta, history)
    groups = [group for group in original["groups"] if group["lane"] != "stock"]
    prior.require(
        meta.bindings == previous["input_metadata_sha256"] == original["input_metadata_sha256"]
        and prior.freeze_groups({**lanes, "stock": []}) == groups
        and backgrounds == original["noise_parents"],
        "Every non-stock role, candidate, background and source commitment must reproduce exactly",
    )
    claimed = set()
    for lane in ("genuine", "clone", "expressive"):
        same_prior_pool = [row for row in lanes[lane] if not prior.identities(row) & claimed]
        for row in same_prior_pool:
            claimed.update(prior.identities(row))
    stock = [row for row in lanes["stock"] if not prior.identities(row) & claimed]
    revised_stock = stock_shortlists(stock, claimed)
    roles = prior.aggregate([*groups, *revised_stock], backgrounds)
    prior.validate_roles([*groups, *revised_stock], backgrounds)
    prior.require(
        len(revised_stock) == 12
        and all(len(group["ordered_candidates"]) == 6 for group in revised_stock)
        and all(
            value["components"]["stock"] == value["seen_provider_voices_in_pending_groups"]["stock"] == 6
            for value in roles.values()
        ),
        "Expected twelve disjoint known stock profiles, six in each prospective role",
    )

    OUTPUT.mkdir(mode=0o700)
    target = OUTPUT / "private-roster.json"
    prior.write(
        target,
        {
            **original,
            "schema": "chipmunk.audio-short-edit-private-source-plan/169b",
            "groups": [*groups, *revised_stock],
            "prior_private_roster_sha256": previous["private_roster_sha256"],
        },
    )
    summary = {
        **previous,
        "schema": "chipmunk.audio-short-edit-source-plan/169b",
        "prior_public_plan_sha256": prior.sha(prior.PUBLIC),
        "prior_private_roster_sha256": previous["private_roster_sha256"],
        "private_roster_sha256": prior.sha(target),
        "implementation_sha256": prior.sha(prior.ROOT / "backend/scripts/audio_short_edit_sources_v169b.py"),
        "prior_implementation_sha256": previous["implementation_sha256"],
        "method_sha256": prior.sha(METHOD),
        "roles": roles,
        "stock_pending_candidates": sum(len(group["ordered_candidates"]) for group in revised_stock),
        "non_stock_roles_reproduced": True,
    }
    prior.write(PUBLIC, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze",))
    parser.parse_args()
    freeze()
