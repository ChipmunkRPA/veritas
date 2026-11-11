"""Prepare additional designed voices from the pinned Turkish source shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import audio_alania_fresh_v213 as prior
from scripts.video_appearance_corrected_v112 import scoped

TRIAL = prior.BASE / "audio-alania-followup-v236"
PUBLIC = prior.PANEL / "audio-alania-followup-freeze-v236.json"
PRIOR_RESULT = prior.PANEL / "audio-alania-fresh-complete-v213.json"
METHOD = prior.ROOT / "docs/audio-alania-followup-v236-method.md"
MODULE = "scripts.audio_alania_followup_v236"
PLANNED = 32
read, sha, require = prior.read, prior.sha, prior.require


def prior_material():
    rows, result = prior.verified_preparation()
    require(result == read(PRIOR_RESULT), "Exact published prior preparation required")
    return read(prior.TRIAL / "private-roster.json"), rows


def context():
    old_roster, _ = prior_material()

    def eligible_rows(rows, first_roster):
        return prior.eligible_rows(rows, [*first_roster, *old_roster])

    locked, selected = scoped(prior.context, {"PLANNED": PLANNED, "eligible_rows": eligible_rows})()
    locked.update(
        schema="chipmunk.audio-alania-followup-freeze/236",
        second_roster_freeze_sha256=sha(prior.PUBLIC),
        second_preparation_result_sha256=sha(PRIOR_RESULT),
        excluded_previous_slots=locked["excluded_previous_slots"] + len(old_roster),
    )
    for path in (Path(__file__), METHOD):
        locked["code_sha256"][str(path.relative_to(prior.ROOT))] = sha(path)
    return locked, selected


def previous_fingerprints():
    _, rows = prior_material()
    fingerprints = prior.previous_fingerprints()
    for row in rows:
        if row["status"] == "qualified":
            for key in prior.FINGERPRINTS:
                fingerprints[key].add(row[key])
    return fingerprints


IDENTITY = {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "context": context}
freeze = scoped(prior.freeze, IDENTITY)
verify = scoped(prior.verify, IDENTITY)
prepared_dir = scoped(prior.prepared_dir, {"TRIAL": TRIAL})
run = scoped(prior.run, {"TRIAL": TRIAL, "verify": verify, "MODULE": MODULE, "__file__": __file__})
PREPARATION = {
    "TRIAL": TRIAL,
    "PUBLIC": PUBLIC,
    "verify": verify,
    "prepared_dir": prepared_dir,
    "previous_fingerprints": previous_fingerprints,
}
worker = scoped(prior.worker, PREPARATION)
verified_preparation = scoped(prior.verified_preparation, {**PREPARATION, "__file__": __file__})


def readout():
    _, receipt = verified_preparation()
    return {**receipt, "schema": "chipmunk.audio-alania-followup-preparation/236"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "prepare", "prepare-worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require((args.command == "prepare-worker") == (args.output is not None), "Only the worker takes an output")
    result = {
        "preflight": lambda: context()[0],
        "freeze": freeze,
        "prepare": run,
        "prepare-worker": lambda: worker(args.output),
        "readout": readout,
    }[args.command]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
