# SPDX-License-Identifier: Apache-2.0
"""Publish aggregate native Lombard replay and detection tallies without source media."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

ARMS = ("current", "timing_candidate", "fine", "scale_adjusted")
ROLES = ("original", "resample_ordinary", "delayed_pixels_ordinary", "source_time_neural")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def clock_admission(fixture_rows):
    result = {}
    for row in fixture_rows:
        split = row["source"]["split"]
        time_base = Fraction(row["visual"]["time_base"])
        pts = row["visual"]["pts"]
        require(pts[0] == 0 and len(pts) > 1, "Exact native video origin required")
        picture_end = (pts[-1] + pts[1] - pts[0]) * time_base
        audio_end = Fraction(row["audio"]["samples"], 16000)
        ticks = min(picture_end, audio_end) * 25
        # The released adapter uses 25 ticks, a 12.5 tick hop, and a ±15 tick search.
        possible = sum(
            start >= 15 and start + 25 + 15 <= ticks for start in (j * 25 // 2 for j in range(int(ticks) + 1))
        )
        result.setdefault(split, Counter())[possible] += 1
    return {
        "note": "Exact source-clock upper bound before speaker attribution or mouth quality. Four windows begin at ticks 25, 37, 50, 62 and require 102 complete ticks.",
        "minimum_seconds_for_four_complete_windows": str(Fraction(102, 25)),
        "source_count_by_possible_windows": result,
    }


def public_rows(rows, scored, refs):
    sources = {k: i for i, k in enumerate(dict.fromkeys(r["ordinal"] for r in rows), 1)}
    groups = {k: i for i, k in enumerate(dict.fromkeys(r["group"] for r in rows), 1)}
    by_case = {(r["ordinal"], r["role"]): r for r in scored}
    by_ref = {r["ordinal"]: r for r in refs}
    output = []
    for row in rows:
        ordinal, role = row["ordinal"], row["role"]
        result = {
            "case": sources[ordinal],
            "group": groups[row["group"]],
            "role": role,
            "style": row["condition"],
            "available": row["status"] == "authored",
            "observation": row.get("observation", {}).get("status", row["status"]),
        }
        if result["available"]:
            result.update(
                native_counts=[row[k] for k in ("native_pictures", "float_pcm_samples", "changed_pictures")]
            )
        if score := by_case.get((ordinal, role)):
            require(
                score["observation"] == result["observation"],
                "Detector truth differs from the native observer",
            )
            current = score["current"]
            measured = bool(current["measured_windows"])
            arms = {
                "current": {
                    "measured": measured,
                    "alert": bool(current["ranges"]),
                    "localized": bool(current.get("right_interval")),
                    "windows": current["measured_windows"],
                    "explicitly_unassessed": not measured
                    or any(
                        flag in current["reason"].lower()
                        for flag in ("change unassessed", "changes unassessed")
                    ),
                }
            }
            for arm, outcome in {
                "timing_candidate": score["timing_candidate"],
                **score["visual_candidates"],
            }.items():
                arms[arm] = {
                    "measured": bool(outcome["measured"]),
                    "alert": bool(outcome["range"]),
                    "localized": bool(outcome.get("right_interval")),
                }
            result["arms"] = arms
        if role == "source_time_neural" and (reference := by_ref.get(ordinal)):
            result["earlier_copy"] = {
                "exact_range": True,
                "exhibits": reference["previews"],
                "event_profile_unassessed": reference["marked_event_timing"] == "profile",
            }
        output.append(result)
    return output


def tally(rows):
    result = {}
    for arm in ARMS:
        by_role = {}
        for row in rows:
            if "arms" not in row:
                continue
            counts = by_role.setdefault(row["role"], Counter())
            counts["cases"] += 1
            counts[f"observed_{row['observation']}"] += 1
            qualified = row["observation"] in {"source_event_retained", "observed_delayed_control"}
            counts["qualified"] += qualified
            outcome = row["arms"][arm]
            measured, alert, localized = (outcome[k] for k in ("measured", "alert", "localized"))
            if arm == "current":
                counts["measurement_windows"] += outcome["windows"]
                counts["explicitly_unassessed_local_timing"] += outcome["explicitly_unassessed"]
            require(not alert or measured, "Unassessed result has a reported range")
            require(not localized or alert, "Localization has no reported range")
            counts["measured"] += measured
            counts["alerts"] += alert
            counts["localizations"] += localized
            if qualified:
                counts["qualified_measured"] += measured
                counts["qualified_alerts"] += alert
                counts["qualified_localizations"] += localized
        result[arm] = by_role
    return result


def replay(receipt):
    require(receipt["schema"] == "chipmunk.lombard-neural-mouth-summary/1", "Unexpected public receipt")
    for split, value in receipt["results"].items():
        rows = value["rows"]
        sources = {r["case"] for r in rows}
        groups = {r["group"] for r in rows}
        require(
            len(sources) == value["frozen_sources"] == (2 if split == "development" else 11)
            and len(groups) == value["source_groups"] == (1 if split == "development" else 6)
            and len(rows) == value["planned_deliveries"] == len(sources) * len(ROLES)
            and {(r["case"], r["role"]) for r in rows} == {(i, k) for i in sources for k in ROLES},
            "Opaque source and role roster differs",
        )
        available = [r for r in rows if r["available"]]
        qualified = [
            r
            for r in available
            if r["role"] == "source_time_neural" and r["observation"] == "source_event_retained"
        ]
        counts = Counter((r["role"], r["observation"]) for r in rows)
        require(
            len(available) == value["verified_deliveries"]
            and len(qualified) == value["synchronized_neural_examples"]
            and len({r["group"] for r in qualified}) == value["synchronized_neural_source_groups"]
            and Counter(r["style"] for r in qualified) == value["synchronized_neural_by_style"]
            and [[*k, v] for k, v in sorted(counts.items())] == value["observations_by_role"]
            and [sum(r["native_counts"][i] for r in available) for i in range(3)]
            == [
                value[k]
                for k in (
                    "native_picture_timestamps_verified",
                    "unchanged_float_audio_samples_verified",
                    "exact_altered_picture_masks_verified",
                )
            ],
            "Published totals differ from sanitized native records",
        )
        if "detector_results" in value:
            require(
                all("arms" in r for r in available) and tally(rows) == value["detector_results"],
                "Published detector totals differ from sanitized rows",
            )
        if "earlier_copy_exact_ranges" in value:
            pairs = [r["earlier_copy"] for r in available if "earlier_copy" in r]
            require(
                sum(r["exact_range"] for r in pairs) == value["earlier_copy_exact_ranges"]
                and sum(r["exhibits"] for r in pairs) == value["earlier_copy_frame_exhibits"]
                and sum(r["event_profile_unassessed"] for r in pairs)
                == value["earlier_copy_marked_timing_unassessed_profile"],
                "Published reference comparison totals differ from sanitized rows",
            )


def aggregate(run, split):
    source = run / "freeze.json"
    detector = run / "detector-freeze.json"
    delivery = run / f"review-{split}.json"
    blind = run / f"blind-observations-{split}.json"
    construct = run / f"neural-{split}.json"
    scoring = run / f"detector-{split}.json"
    reference_path = run / f"reference-marked-{split}.json"
    content, neural = read(delivery), read(construct)
    require(
        content["freeze_sha256"] == neural["freeze_sha256"] == digest(source)
        and content["blind_observations_sha256"] == digest(blind),
        "Native review and construction do not refer to the same freeze",
    )
    rows = content["cases"]
    planned = 2 if split == "development" else 11
    role_counts = Counter(r["role"] for r in rows)
    require(
        len({r["ordinal"] for r in rows}) == planned
        and role_counts
        == dict.fromkeys(
            ("original", "resample_ordinary", "delayed_pixels_ordinary", "source_time_neural"), planned
        ),
        "Every predeclared role must appear",
    )
    available = [r for r in rows if r["status"] == "authored"]
    qualified = [
        r
        for r in available
        if r["role"] == "source_time_neural" and r["observation"]["status"] == "source_event_retained"
    ]
    example = None
    if qualified:
        case = qualified[0]
        obs = case["observation"]
        example = {
            "authored_picture_interval": case["authored_frames"],
            "authored_seconds": case["authored_seconds"],
            "observed_visual_release_seconds": obs["seconds"],
            "visual_picture_shift": obs["visual_picture_shift"],
            "signed_acoustic_midpoint_offset_s": obs["acoustic_bracket_midpoint_offset_s"],
        }
    result = {
        "frozen_sources": planned,
        "source_groups": len({r["group"] for r in rows}),
        "planned_deliveries": len(rows),
        "verified_deliveries": len(available),
        "native_picture_timestamps_verified": sum(r["native_pictures"] for r in available),
        "unchanged_float_audio_samples_verified": sum(r["float_pcm_samples"] for r in available),
        "exact_altered_picture_masks_verified": sum(r["changed_pictures"] for r in available),
        "synchronized_neural_examples": len(qualified),
        "synchronized_neural_source_groups": len({r["group"] for r in qualified}),
        "synchronized_neural_by_style": Counter(r["condition"] for r in qualified),
        "observations_by_role": content["counts"],
        "independent_native_replay_sha256": digest(delivery),
        "blind_picture_replay_sha256": digest(blind),
        "neural_construction_sha256": digest(construct),
        "checkpoint_exact_loads": neural["model_loads"],
        "renderer_root_forwards": neural["root_forwards"],
        "renderer_wall_seconds": neural["wall_seconds"],
        "renderer_peak_rss_bytes": neural["peak_rss_bytes"],
        "anonymous_example": example,
    }
    scored_rows, refs = [], []
    if scoring.is_file():
        scored = read(scoring)
        require(
            scored["native_review_sha256"] == digest(delivery)
            and scored["detector_freeze_sha256"] == digest(detector)
            and len(scored["cases"]) == len(available),
            "Detector readout differs from exact native fixtures",
        )
        scored_rows = scored["cases"]
        result["detector_result_sha256"] = digest(scoring)
    if reference_path.is_file():
        reference = read(reference_path)
        refs = reference["cases"]
        expected = {r["ordinal"] for r in available if r["role"] == "source_time_neural"}
        require(
            reference["native_review_sha256"] == digest(delivery)
            and len(refs) == len(expected)
            and {r["ordinal"] for r in refs} == expected
            and all(r["edited_frames"] == 16 and r["previews"] == 6 for r in refs),
            "Earlier-copy replay differs from exact native fixtures",
        )
        result.update(
            earlier_copy_result_sha256=digest(reference_path),
            earlier_copy_exact_ranges=len(refs),
            earlier_copy_frame_exhibits=sum(r["previews"] for r in refs),
            earlier_copy_marked_timing_unassessed_profile=sum(
                r["marked_event_timing"] == "profile" for r in refs
            ),
        )
    result["rows"] = public_rows(rows, scored_rows, refs)
    if scored_rows:
        result["detector_results"] = tally(result["rows"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--splits", choices=("development", "held"), nargs="+", default=("development", "held"))
    a = p.parse_args()
    if a.verify:
        replay(read(a.run))
        print("Public outcome counts replayed")
        return
    freeze = a.run / "freeze.json"
    protocol = read(freeze)
    result = {
        "schema": "chipmunk.lombard-neural-mouth-summary/1",
        "source_qualification_sha256": protocol["source_qualification_sha256"],
        "freeze_sha256": digest(freeze),
        "protocol": protocol["spec"],
        "released_timing_clock_admission": clock_admission(protocol["sources"]),
        "results": {split: aggregate(a.run, split) for split in a.splits},
    }
    replay(result)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
