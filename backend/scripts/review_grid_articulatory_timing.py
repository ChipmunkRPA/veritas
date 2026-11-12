# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen articulation timing totals and optionally the private native media."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median
from zipfile import ZipFile

TICK = 640
ROLES = ("recorded", "local_gain", "local_lowpass", "local_noise", "constant_early", "constant_late",
         "short_early", "short_late", "long_early", "long_late")
POINT_GROUPS, HELD_GROUPS = 2, 3
CASE_COUNT = {"point": 140, "held": 270, "neural": 8}


def read(path):
    return json.loads(path.read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    return digest(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def useful(result, span, lag):
    if result is None or span is None or lag is None or result["relative_lag_ticks"] is None:
        return False
    return abs(result["relative_lag_ticks"] - lag) <= 2 and location_overlaps(result, span)


def location_overlaps(result, span):
    a, b = result["extent_ticks"]
    c, d = span
    shared = max(0, min(b, d) - max(a, c))
    envelope = max(b, d) - min(a, c)
    return 5 * shared >= 2 * envelope


def rank_diagnostic(records):
    edited = [row for row in records if row["known_shift"] is not None and row["candidate"]]
    matched = [row for row in edited if useful(row["candidate"], row["known_extent"], row["known_shift"])]
    top = max((row["candidate"]["score"] for row in matched), default=None)
    controls = [row for row in records if top is not None and row["known_shift"] is None and row["candidate"]
                and row["candidate"]["score"] >= top]
    return {
        "edited_with_proposal": len(edited),
        "correct_sign": sum((r["candidate"]["relative_lag_ticks"] > 0) == (r["known_shift"] > 0) for r in edited),
        "signed_error_at_most_two_ticks": sum(abs(r["candidate"]["relative_lag_ticks"] - r["known_shift"]) <= 2
                                             for r in edited),
        "right_interval": sum(location_overlaps(r["candidate"], r["known_extent"]) for r in edited),
        "useful": len(matched),
        "top_useful_response": top,
        "controls_at_or_above_top_useful": len(controls),
        "control_groups_at_or_above_top_useful": len({row["group"] for row in controls}),
        "control_roles_at_or_above_top_useful": dict(Counter(row["role"] for row in controls)),
    }


def paired_processing_diagnostic(records):
    recorded = {row["recording"]: row["candidate"]["score"] for row in records
                if row["role"] == "recorded" and row["candidate"]}
    differences = defaultdict(list)
    for row in records:
        if row["role"] != "recorded" and row["candidate"] and row["recording"] in recorded:
            differences[row["role"]].append(row["candidate"]["score"] - recorded[row["recording"]])
    return {key: {"matched_recordings": len(values), "median_change": median(values)}
            for key, values in differences.items()}


def summarize(records, point):
    by_role = defaultdict(Counter)
    totals = defaultdict(Counter)
    added_groups = set()
    preserved = True
    for row in records:
        role = row["role"]
        truth = row["known_shift"] is not None
        cand = row["candidate"]
        incumbent = row["current"]
        supported = row["assessed"]
        flag = cand is not None and cand["score"] > point
        correctly = flag and useful(cand, row["known_extent"], row["known_shift"])
        original = any(useful(item, row["known_extent"], row["known_shift"]) for item in incumbent["ranges"])
        added = correctly and not original
        if added:
            added_groups.add(row["group"])
        preserved &= not original or correctly
        outcomes = {
            "cases": 1,
            "candidate_measured": int(supported),
            "candidate_alerts": int(flag),
            "candidate_useful": int(correctly),
            "current_measured": int(bool(incumbent["measured_windows"])),
            "current_alerts": int(bool(incumbent["ranges"])),
            "current_useful": int(original),
            "new_control_alerts": int(not truth and flag and not incumbent["ranges"]),
            "added_useful": int(added),
        }
        by_role[role].update(outcomes)
        totals["signed_edits" if truth else "controls"].update(outcomes)
    return {
        "by_role": {key: dict(values) for key, values in by_role.items()},
        "by_condition": {key: dict(values) for key, values in totals.items()},
        "groups_with_added_useful": len(added_groups),
        "preserved_every_current_location": preserved,
    }


def advancement(held):
    by_cond = held["by_condition"]
    edit, control = by_cond["signed_edits"], by_cond["controls"]
    result = {
        "new_locations": edit["added_useful"] >= 3 and held["groups_with_added_useful"] >= 2,
        "coverage": all(4 * row["candidate_measured"] >= 3 * row["cases"] for row in by_cond.values()),
        "control_alerts": control["new_control_alerts"] <= 1,
        "constant_offsets": sum(held["by_role"][f"constant_{label}"]["new_control_alerts"]
                                for label in ("early", "late")) == 0,
    }
    return {**result, "all_met": all(result.values())}


def expected_truth(ordinal, role):
    if role not in ROLES[-4:]:
        return None, None
    start = 26 + ordinal % 5
    end = start + 16
    span = [start, end] if role.startswith("short_") else [start-3, end+3]
    signed = (5 if role.startswith("short_") else 7) * (-1 if role.endswith("_early") else 1)
    return span, signed


def recreate(pcm, ordinal, role):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    if role == "recorded":
        return pcm
    span, shift = expected_truth(ordinal, role)
    if role.startswith("constant_"):
        shift = -5 if role.endswith("early") else 5
    if shift is not None:
        d = shift * TICK
        replacement = np.zeros_like(pcm)
        replacement[max(0, d):len(pcm) + min(0, d)] = pcm[max(0, -d):len(pcm) - max(0, d)]
        if span is None:
            return replacement
    else:
        start = 26 + ordinal % 5
        span = [start, start + 16]
        if role == "local_gain":
            replacement = pcm * .68
        elif role == "local_lowpass":
            replacement = sosfiltfilt(butter(4, 1900, fs=16000, output="sos"), pcm)
        else:
            assert role == "local_noise"
            a, b = (i * TICK for i in span)
            rms = np.sqrt(np.mean(pcm[a:b].astype(np.float64)**2))
            replacement = pcm + .12 * rms * np.random.default_rng(36120 + ordinal).normal(0, 1, len(pcm))
    a, b = (i * TICK for i in span)
    out = pcm.copy()
    for i in range(a, b):
        fade = min(160, i - a + 1, b - i) / 160
        out[i] = np.clip(round((1 - fade) * pcm[i] + fade * replacement[i]), -32768, 32767)
    assert np.array_equal(out[:a], pcm[:a]) and np.array_equal(out[b:], pcm[b:])
    return out


def evidence_native(row, frames, pts, clock, pcm):
    import numpy as np

    item = row["candidate"].get("evidence")
    if item is None:
        assert not row["flagged"]
        return 0
    assert row["flagged"]
    start, end = row["candidate"]["best"]["extent_ticks"]
    assert item["audio_samples"] == [start * TICK, end * TICK]
    assert digest(pcm[start*TICK:end*TICK].tobytes()) == item["audio_sha256"]
    for picture in item["pictures"]:
        index = picture["frame"]
        assert index in (start - 1, (start + end) // 2, end)
        assert pts[index] == picture["pts"] and clock == Fraction(picture["time_base"])
        assert digest(frames[index]) == picture["rgb_sha256"]
        x0, y0, x1, y1 = picture["crop_xyxy"]
        assert 0 <= x0 < x1 <= 360 and 0 <= y0 < y1 <= 288
        crop = np.frombuffer(frames[index], dtype=np.uint8).reshape(288, 360, 3)[y0:y1, x0:x1]
        assert digest(crop.tobytes()) == picture["crop_sha256"]
    return len(item["pictures"])


def native_readback(freeze, traces, fit):
    from scripts.review_grid_single_clip_release_timing import RGB_SIZE, native

    results = Counter()
    sources = {row["id"]: row for row in freeze["sources"]}
    for row in freeze["neural"]:
        sources[f"neural-{row['ordinal']}-{row['role']}"] = {**row, "file": row["path"]}
    assert {item["id"] for item in fit["sources"]} == {row["id"] for row in freeze["sources"] if row["split"] == "fit"}
    assert fit["counts"] == [sum(item["counts"][i] for item in fit["sources"]) for i in range(2)]
    for binding in fit["sources"]:
        source = sources[binding["id"]]
        path = Path(source["file"])
        assert file_digest(path) == source["sha256"]
        with ZipFile(source["archive"]) as archive:
            member = archive.getinfo(source["archive_member"])
            assert source["member_crc"] == member.CRC and digest(archive.read(member)) == source["sha256"]
        pts, _, raw, pcm = native(path)
        assert binding["pictures_sha256"] == digest(raw) and binding["pcm_sha256"] == digest(pcm.tobytes())
        results["fit_files"] += 1
        results["fit_pictures"] += len(pts)
    for split, trace in traces.items():
        cases = defaultdict(list)
        for row in trace["rows"]:
            cases[row["id"]].append(row)
        for binding in trace["source_audit"]:
            source = sources[binding["id"]]
            path = Path(source["file"])
            assert file_digest(path) == source["sha256"]
            if split != "neural":
                with ZipFile(source["archive"]) as archive:
                    member = archive.getinfo(source["archive_member"])
                    assert source["member_crc"] == member.CRC and digest(archive.read(member)) == source["sha256"]
            pts, clock, raw, pcm = native(path)
            frames = [raw[i:i+RGB_SIZE] for i in range(0, len(raw), RGB_SIZE)]
            assert binding["pts"] == pts and Fraction(binding["time_base"]) == clock
            assert binding["rgb_sha256"] == [digest(item) for item in frames]
            assert binding["pcm_sha256"] == digest(pcm.tobytes())
            results["native_files"] += 1
            results["native_pictures"] += len(frames)
            for row in cases[binding["id"]]:
                if split == "neural":
                    expected, expected_shift = source["picture_edit"], source["known_lag_ticks"]
                    edited = pcm
                else:
                    expected, expected_shift = expected_truth(source["ordinal"], row["role"])
                    edited = recreate(pcm, source["ordinal"], row["role"])
                assert row["known_extent"] == expected and row["known_shift"] == expected_shift
                assert row["pcm_sha256"] == digest(edited.tobytes())
                results["case_pcm_commitments"] += 1
                results["native_crop_commitments"] += evidence_native(row, frames, pts, clock, edited)
    return dict(results)


def public_row(row, local_groups, local_recordings):
    proposal = row["candidate"]
    best = proposal["best"]
    group = f"{local_groups.index(row['group']) + 1:02}"
    value = {
        "group": group, "recording": local_recordings[row["id"]], "role": row["role"],
        "known_extent": row["known_extent"], "known_shift": row["known_shift"],
        "assessed": bool(proposal["assessed"]),
        "candidate": {key: best[key] for key in ("extent_ticks", "relative_lag_ticks", "score")} if best else None,
    }
    if row["current"] is not None:
        value["current"] = row["current"]
    return value


def audit_receipt(receipt):
    assert receipt["schema"] == "chipmunk.grid-articulatory-timing-results/1"
    point_rows, held, neural = (receipt["rows"][key] for key in ("point", "held", "neural"))
    assert len(point_rows) == CASE_COUNT["point"] and len(held) == CASE_COUNT["held"] and len(neural) == CASE_COUNT["neural"]
    assert len({row["group"] for row in point_rows}) == POINT_GROUPS and len({row["group"] for row in held}) == HELD_GROUPS
    for subset, n in ((point_rows, 7), (held, 9)):
        for group in {row["group"] for row in subset}:
            assert Counter(row["role"] for row in subset if row["group"] == group) == dict.fromkeys(ROLES, n)
        for recording in {row["recording"] for row in subset}:
            matching = [row for row in subset if row["recording"] == recording]
            assert len({row["group"] for row in matching}) == 1 and Counter(row["role"] for row in matching) == dict.fromkeys(ROLES, 1)
    ordinary = [row["candidate"]["score"] for row in point_rows if row["known_shift"] is None and row["candidate"]]
    value = max(0., max(ordinary))
    assert receipt["point"] == {"value": value, "measured": len(ordinary), "total": 84}
    totals = {key: summarize(rows, value) for key, rows in (("held", held), ("neural", neural))}
    assert totals == receipt["totals"]
    assert receipt["ranking"] == {key: rank_diagnostic(rows) for key, rows in (("held", held), ("neural", neural))}
    assert receipt["paired_processing"] == paired_processing_diagnostic(held)
    assert advancement(totals["held"]) == receipt["advancement"]
    return totals


def build(run):
    freeze, point = read(run / "freeze.json"), read(run / "point.json")
    traces = {key: read(run / f"{key}-readout.json") for key in ("point", "held", "neural")}
    frozen_sha = file_digest(run / "freeze.json")
    fit_sha = file_digest(run / "fit.json")
    for key, value in traces.items():
        assert value["freeze_sha256"] == frozen_sha and value["fit_sha256"] == fit_sha
        assert value["point_sha256"] == (None if key == "point" else file_digest(run / "point.json"))
        assert len(value["rows"]) == CASE_COUNT[key]
    assert point["fit_sha256"] == fit_sha and point["cases_sha256"] == file_digest(run / "point-readout.json")
    public_rows = {}
    for key, trace in traces.items():
        groups = sorted({row["group"] for row in trace["rows"]})
        unique = dict.fromkeys(row["id"] for row in trace["rows"])
        recordings = {key: f"{i:02}" for i, key in enumerate(unique, 1)}
        public_rows[key] = [public_row(row, groups, recordings) for row in trace["rows"]]
        for row in trace["rows"]:
            if key != "point":
                best = row["candidate"]["best"]
                assert row["flagged"] == bool(best and best["score"] > point["value"])
    totals = {key: summarize(public_rows[key], point["value"]) for key in ("held", "neural")}
    receipt = {
        "schema": "chipmunk.grid-articulatory-timing-results/1",
        "method": freeze["spec"],
        "freeze_sha256": frozen_sha,
        "point": {key: point[key] for key in ("value", "measured", "total")},
        "open_close": {key: {name: value[f"open_close_{name}"] for name in ("counts", "auc")}
                       for key, value in traces.items() if key != "neural"},
        "rows": public_rows,
        "totals": totals,
        "ranking": {key: rank_diagnostic(public_rows[key]) for key in ("held", "neural")},
        "paired_processing": paired_processing_diagnostic(public_rows["held"]),
        "advancement": advancement(totals["held"]),
        "native_replay": native_readback(freeze, traces, read(run / "fit.json")),
    }
    audit_receipt(receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.run:
        receipt = build(args.run)
        write(args.receipt, receipt)
    else:
        receipt = read(args.receipt)
    audit_receipt(receipt)
    print(json.dumps({key: receipt[key] for key in ("point", "totals", "ranking", "paired_processing", "advancement",
                                                  "native_replay")}, sort_keys=True))


if __name__ == "__main__":
    main()
