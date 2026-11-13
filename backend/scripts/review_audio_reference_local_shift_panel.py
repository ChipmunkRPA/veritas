# SPDX-License-Identifier: Apache-2.0
"""Recompute standalone local waveform timing outcomes from their bounded ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

POSITIVES = ("local_later", "local_earlier")
ORDINARY = (
    "unchanged",
    "gain",
    "equalization",
    "whole_track_shift",
    "short_offset",
    "unrelated_insert",
    "silence_local",
    "silence_all",
    "preexisting_repeat",
)
STATES = {
    "local_shift",
    "no_local_shift",
    "unverified_clock",
    "different_clock",
    "too_short",
    "insufficient_support",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def tally(groups):
    results = {}
    for codec in ("pcm", "aac", "cross"):
        roles = POSITIVES + ORDINARY if codec != "cross" else ("codec_flac", "codec_aac")
        states = {role: Counter() for role in roles}
        correct = Counter()
        group_ids = set()
        incorrect, alerts, clocked, decoded = Counter(), Counter(), 0, 0
        minimum_iou = Fraction(1)
        errors = []
        for group in groups:
            rows = [c for c in group["cases"] if c["carrier"] == codec]
            require(
                len(rows) == len(roles) and {r["role"] for r in rows} == set(roles), "Incorrect role roster"
            )
            for row in rows:
                role, state, measured = row["role"], row["status"], row["range"]
                require(
                    state in STATES and (state == "local_shift") == bool(measured),
                    "Timing state does not match its range",
                )
                require((row["truth"] is not None) == (role in POSITIVES), "Truth must match the edit role")
                for key in ("earlier", "submitted"):
                    track = row[key]
                    if track["clock"]:
                        require(
                            track["clock"]["samples"] == track["native_readback_samples"]
                            and track["clock"]["pcm_sha256"] == track["native_readback_sha256"],
                            "Independent decode differs",
                        )
                        decoded += 1
                states[role][state] += 1
                clocked += state in ("local_shift", "no_local_shift")
                if not measured:
                    continue
                a, b = measured["submitted_samples"]
                c, d = measured["earlier_samples"]
                require(
                    a - c == b - d == measured["shift_samples"] and a < b and c < d and row["support"] >= 8,
                    "The reported audio coordinates or support differ",
                )
                if not row["truth"]:
                    alerts[role] += 1
                    continue
                coverage = iou(measured["submitted_samples"], row["truth"])
                error = abs(measured["shift_samples"] - row["offset"])
                if coverage >= Fraction(3, 5) and error <= 16:
                    correct[role] += 1
                    group_ids.add(group["group"])
                    errors.append(error)
                    minimum_iou = min(minimum_iou, coverage)
                else:
                    incorrect[role] += 1
        all_states = sum(states.values(), Counter())
        non_silent = (len(roles) - int(codec != "cross")) * len(groups)
        criteria = (
            {}
            if codec == "cross"
            else {
                "count": sum(correct.values()) >= 30,
                "signs": min(correct[r] for r in POSITIVES) >= 12,
                "groups": len(group_ids) >= 5,
                "ordinary": sum(alerts.values()) <= 1,
                "essential_controls": sum(
                    alerts[r] for r in ("unchanged", "preexisting_repeat", "whole_track_shift")
                )
                == 0,
                "coverage": 4 * clocked >= 3 * non_silent,
            }
        )
        results[codec] = {
            "by_role": states,
            "statuses": all_states,
            "correct": correct,
            "incorrect": incorrect,
            "control_alerts": alerts,
            "groups": len(group_ids),
            "independently_decoded_sides": decoded,
            "minimum_correct_iou": str(minimum_iou) if errors else None,
            "largest_correct_sample_error": max(errors, default=None),
            "gate": criteria,
            "advance": bool(criteria) and all(criteria.values()),
        }
    return results


def verify_application(app, rows, results):
    pcm = results["pcm"]["statuses"]
    require(
        app["native_wav_cases"] == sum(pcm.values())
        and app["native_wav_results"]
        == {
            "local_shift": pcm["local_shift"],
            "no_local_shift": pcm["no_local_shift"],
            "unassessed": sum(v for k, v in pcm.items() if k not in {"local_shift", "no_local_shift"}),
            "verified_wav_excerpts": 2 * pcm["local_shift"],
        }
        and app["wav_flac_unchanged_no_local_shift"]
        == results["cross"]["by_role"]["codec_flac"]["no_local_shift"],
        "App result does not match held pair or excerpt totals",
    )
    positive = app["wav_flac_positive"]
    require(
        positive["local_timing_status"] == "local_shift"
        and positive["exact_excerpts"] == 2
        and any(
            r["ordinal"] == positive["ordinal"]
            and any(c["status"] == "local_shift" and c["carrier"] == "pcm" for c in r["cases"])
            for r in rows
        ),
        "The playable mixed-carrier example does not identify a held positive",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--assemble", type=Path)
    a = p.parse_args()
    if a.assemble:
        lock = read(a.assemble / "freeze.json")
        development = read(a.assemble / "development.json")
        held = read(a.assemble / "held.json")
        app = read(a.assemble / "app-held-replay.json")
        require(
            held["freeze_sha256"] == sha(a.assemble / "freeze.json")
            and development["freeze_sha256"] == held["freeze_sha256"]
            and app["held_sha256"] == sha(a.assemble / "held.json")
            and len(development["rows"]) == 8
            and len({r["group"] for r in development["rows"]}) == 2
            and len(held["rows"]) == 24
            and len({r["group"] for r in held["rows"]}) == 6
            and {r["group"] for r in held["rows"]}.isdisjoint(r["group"] for r in development["rows"]),
            "Source roster or held application replay differs",
        )
        result = {
            "schema": "chipmunk.audio-reference-local-shift-panel/1",
            "freeze_sha256": held["freeze_sha256"],
            "method_sha256": lock["method_sha256"],
            "source_receipt_sha256": lock["source_receipt_sha256"],
            "protocol": lock["protocol"],
            "development_results": tally(development["rows"]),
            "rows": held["rows"],
            "results": tally(held["rows"]),
            "application": app,
        }
        a.receipt.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    receipt = read(a.receipt)
    require(
        receipt["schema"] == "chipmunk.audio-reference-local-shift-panel/1"
        and tally(receipt["rows"]) == receipt["results"],
        "Held result does not replay",
    )
    verify_application(receipt["application"], receipt["rows"], receipt["results"])
    print(json.dumps(receipt["results"], indent=2))


if __name__ == "__main__":
    main()
