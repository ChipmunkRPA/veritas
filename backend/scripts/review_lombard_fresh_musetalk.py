# SPDX-License-Identifier: Apache-2.0
"""Publish a redacted, standard-library replay of the native Lombard eligibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.lombard-fresh-musetalk/1"
ROLES = ("original", "same_time_resample", "same_time_neural", "earlier_sound_neural", "delayed_pixels")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def midpoint(pair):
    return sum(map(Fraction, pair)) / 2


def qualify(row):
    start, end = map(Fraction, row["edit"])
    step = Fraction(row["picture_step"])
    sound = midpoint(row["audio_bracket"])
    original = row["roles"]["original"].get("visual_seconds")
    flags = {}
    require(end - start == step * 16 and start <= sound < end, "Unexpected native edit extent")
    for role in ROLES:
        evidence = row["roles"][role]
        if pair := evidence.get("visual_seconds"):
            require(original is not None, "Measured redraw lacks verified original geometry")
            a, b = map(Fraction, pair)
            require(b - a == step, "Observed bracket is not two consecutive pictures")
            frames = (a - Fraction(original[0])) / step
            require(
                frames.denominator == 1 and frames == evidence["picture_shift"],
                "Delivered shift differs from native source",
            )
            if role == "delayed_pixels":
                accepted = abs(frames - 5) <= 1 and start <= a < b < end
                status = "observed_delayed_control" if accepted else "different_observed_motion"
            else:
                accepted = (
                    abs(frames) <= 1
                    and abs(midpoint(pair) - sound) <= Fraction(2, 25)
                    and start <= a < b < end
                )
                status = "source_event_retained" if accepted else "different_observed_motion"
            require(status == evidence["status"], "Published event decision does not replay")
        flags[role] = evidence["status"]
    return (
        flags["same_time_neural"] == flags["same_time_resample"] == "source_event_retained"
        and flags["delayed_pixels"] == "observed_delayed_control"
    )


def summary(rows):
    counts = Counter()
    for row in rows:
        require(set(row["roles"]) == set(ROLES), "Missing frozen study role")
        measured = qualify(row)
        require(measured == row["qualifies_local_event"], "Short event status differs")
        counts["qualified"] += measured
        counts["qualified_with_earlier_condition_rejected"] += (
            measured and row["roles"]["earlier_sound_neural"]["status"] != "source_event_retained"
        )
        if pair := row.get("earlier_sound_later_visual_seconds"):
            start, end = map(Fraction, row["edit"])
            early, late = map(Fraction, pair)
            reference = Fraction(row["roles"]["original"]["visual_seconds"][0])
            step = Fraction(row["picture_step"])
            require(late - early == step, "Later challenge lacks two consecutive pictures")
            shift = (early - reference) / step
            require(shift.denominator == 1, "Later challenge lies off the original picture clock")
            later = start <= early < late < end and abs(shift - 5) <= 1
            counts["earlier_sound_neural_later_visual_event"] += later
            counts["qualified_with_later_visual_event_in_earlier_condition"] += measured and later
        for role, evidence in row["roles"].items():
            counts[f"{role}:{evidence['status']}"] += 1
            counts["native_pictures"] += evidence.get("native_pictures", 0)
            counts["unchanged_float_samples"] += evidence.get("float_pcm_samples", 0)
            counts["changed_pictures"] += evidence.get("changed_pictures", 0)
    return dict(counts)


def render_scope(lock, render):
    expected = {(r["source"]["ordinal"], role): r for r in lock["sources"] for role in ROLES}
    by_case = {(r["ordinal"], r["role"]): r for r in render["cases"]}
    require(
        set(by_case) == set(expected) and len(by_case) == len(render["cases"]), "Frozen render scope differs"
    )
    counted = Counter(
        {"neural_frame_inputs": 0, "matched_latent_pairs": 0, "different_conditioned_crop_pairs": 0}
    )
    for key, row in expected.items():
        ordinal, role = key
        delivered = by_case[key]
        if delivered["status"] != "authored" or role not in ("same_time_neural", "earlier_sound_neural"):
            continue
        start, stop = row["span"]
        require(
            [r["index"] for r in delivered["edits"]] == list(range(start, stop)), "Neural frame inputs differ"
        )
        shift = -5 if role == "earlier_sound_neural" else 0
        routes = row["conditioning"][str(shift)]
        for actual, route in zip(delivered["edits"], routes, strict=True):
            require(
                all(
                    actual["conditioning"][k] == route[k]
                    for k in ("encoder_anchor", "tick_indices", "source_time_seconds")
                )
                and actual["pts"] == row["visual"]["pts"][actual["index"]]
                and actual["model"]["seed"]
                == int.from_bytes(
                    hashlib.sha256(
                        f"chipmunk.lombard-fresh-musetalk/1:{ordinal}:{actual['index']}".encode()
                    ).digest()[:8],
                    "big",
                )
                % 2**63,
                "Neural conditioning route or seed differs from source clock",
            )
            counted["neural_frame_inputs"] += 1
        if role != "earlier_sound_neural":
            continue
        original_condition = by_case[ordinal, "same_time_neural"]
        for before, current in zip(original_condition["edits"], delivered["edits"], strict=True):
            prior, after = before["model"], current["model"]
            require(
                all(
                    prior[k] == after[k]
                    for k in (
                        "seed",
                        "masked_input_f32_sha256",
                        "original_input_f32_sha256",
                        "sampled_latents_f32_sha256",
                    )
                )
                and prior["condition_with_position_f32_sha256"]
                != after["condition_with_position_f32_sha256"],
                "Paired neural branch changed more than its speech conditioning",
            )
            counted["matched_latent_pairs"] += 1
            counted["different_conditioned_crop_pairs"] += (
                prior["output_rgb_sha256"] != after["output_rgb_sha256"]
            )
    return dict(counted)


def assemble(folder):
    freeze, audit, render, observed = (
        folder / name for name in ("freeze.json", "audit.json", "render.json", "blind-observations-held.json")
    )
    lock, receipt = read(freeze), read(audit)
    require(
        receipt["freeze_sha256"] == sha(freeze)
        and receipt["render_sha256"] == sha(render)
        and receipt["observer_sha256"] == sha(observed),
        "Independent audit bindings differ",
    )
    private_render = read(render)
    observed_by_token = {r["token"]: r for r in read(observed)["rows"]}
    delayed_by_source = {
        r["ordinal"]: r
        for r in private_render["cases"]
        if r["role"] == "earlier_sound_neural" and r["status"] == "authored"
    }
    rows = []
    for i, row in enumerate(receipt["rows"], 1):
        views = {}
        for role, result in row["roles"].items():
            views[role] = {
                k: v
                for k, v in result.items()
                if k in ("status", "native_pictures", "float_pcm_samples", "changed_pictures")
            }
            observation = result.get("observation", {})
            if "frames" in observation:
                views[role].update(
                    visual_seconds=observation["seconds"], picture_shift=observation["visual_picture_shift"]
                )
        rows.append(
            {
                "clip": i,
                "speaker_group": i,
                "speaking_style": row["style"],
                "edit": row["edit"],
                "picture_step": row["picture_step"],
                "audio_bracket": row["audio_bracket"],
                "roles": views,
                "qualifies_local_event": row["qualifies_local_event"],
                "earlier_sound_later_visual_seconds": observed_by_token[
                    delayed_by_source[row["source_ordinal"]]["token"]
                ]["releases"]["later_neighborhood"].get("seconds")
                if row["source_ordinal"] in delayed_by_source
                else None,
            }
        )
    counts = summary(rows)
    require(counts["qualified"] == receipt["qualified"], "Published cohort result differs")
    return {
        "schema": SCHEMA,
        "freeze_sha256": sha(freeze),
        "render_sha256": sha(render),
        "audit_sha256": sha(audit),
        "native_observation_sha256": sha(observed),
        "native_render_scope": render_scope(lock, private_render),
        "protocol": lock["protocol"],
        "runtime_versions": lock["restored_runtime_package_versions"],
        "historical_asset_manifest_sha256": lock["renderer_asset_manifest_sha256"],
        "rows": rows,
        "summary": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    args = parser.parse_args()
    if args.assemble:
        with args.receipt.open("x") as stream:
            json.dump(assemble(args.assemble), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    published = read(args.receipt)
    rows = published["rows"]
    require(
        published["schema"] == SCHEMA and len(rows) == len({r["speaker_group"] for r in rows}) == 6,
        "Expected six held groups",
    )
    require(summary(rows) == published["summary"], "Public frozen summary differs")
    print(json.dumps(published["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
