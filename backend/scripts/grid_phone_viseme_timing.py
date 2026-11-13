# SPDX-License-Identifier: Apache-2.0
"""Evaluate blind speech-phone timing against native mouth shape on separate speakers."""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from contextlib import closing
from fractions import Fraction
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import probe_frames
from scripts import grid_articulatory_timing as old_states
from scripts import grid_paired_closure_panel as native
from scripts import grid_phone_alignment_v161 as phones
from scripts import grid_reserve_neural_mouth as neural
from scripts import grid_reserve_neural_mouth_readout as incumbent
from scripts import grid_reserve_word_clocks as grid
from scripts import grid_single_clip_release_timing as release
from scripts import video_lipsync_local_contrast_transfer as controls
from scripts.video_common_sources_v89 import read, require, sha, write

RATE, TICK, PICTURES = 16000, 640, 75
PHONE_STEP = 160
FAMILIES = {
    "closed_lips": ("B", "M", "P"),
    "rounded_lips": ("OW", "UH", "UW", "W"),
    "spread_vowel": ("EY", "IH", "IY"),
    "open_vowel": ("AA", "AE", "AH", "AO", "EH"),
}
INDEX = {phone: index for index, family in enumerate(FAMILIES.values()) for phone in family}
SPLIT = {"fit": ([10, 11, 12], 10), "point": ([13, 14], 6), "held": ([8, 9, 15], 8)}
ROLE_ORDER = controls.CASES
LAGS = np.arange(-11, 12)
FLANK = 12
WIDTHS = (14, 18, 22)
NEURAL = old_states.NEURAL
OLD_FREEZES = (
    (old_states.FRESH_PRIOR / "freeze.json", "roster", "member"),
    (
        old_states.FRESH_PRIOR.with_name("grid-marked-release-holdout-2026-11-12") / "freeze.json",
        "roster",
        "member",
    ),
    (
        old_states.FRESH_PRIOR.with_name("grid-articulatory-timing-release-2026-11-12") / "freeze.json",
        "sources",
        "archive_member",
    ),
)
SPEC = {
    "schema": "chipmunk.grid-phone-viseme-timing/1",
    "split": {key: [groups, count] for key, (groups, count) in SPLIT.items()},
    "selection": "Deterministic SHA-256 order over unused GRID MPEG members, excluding prior source plans, release and open/closed study rosters. Before fitting, reject files without 75 exact native picture timestamps. Speaker groups have earlier project exposure.",
    "phone": "Blind context-independent US English phone decoding at 100 Hz, full clip and 68% gain, with no supplied word labels, grammar or earlier video. Keep a native midpoint only when both copies decode the same admitted phone and it is at least 10 ms inside each interval.",
    "families": {name: list(group) for name, group in FAMILIES.items()},
    "shape": "Use three log-transformed inner-lip gaps normalized by eye distance, plus lip width divided by the clip's median width. Fit balanced regularized multiclass logistic regression (C=.1) on genuine aligned speakers only.",
    "schedule": {
        "flank_ticks": FLANK,
        "widths_ticks": list(WIDTHS),
        "lags": LAGS.tolist(),
        "reference_lag_limit": 5,
        "relative_lag_limits": [4, 8],
        "middle_comparisons": 3,
        "flank_comparisons": 2,
    },
    "rule": "Compare the same visual ticks with different supported phone families under two lags. The middle must favor the local lag while both flanks favor the same reference lag. Rank by the smallest of three mean log-likelihood gains.",
    "variants": list(ROLE_ORDER),
    "point": "Strict excess over zero and the greatest ordinary calibration response, with point set before held decoding.",
    "localization": {"minimum_iou": "2/5", "signed_tolerance_ticks": 2},
    "gate": {
        "new_locations": 4,
        "new_groups": 3,
        "maximum_new_ordinary_flags": 1,
        "minimum_coverage": 0.60,
        "maximum_constant_flags": 0,
        "preserve_current_correct": True,
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bindings():
    paths = (
        Path(__file__),
        Path(old_states.__file__),
        Path(native.__file__),
        Path(phones.__file__),
        Path(neural.__file__),
        Path(incumbent.__file__),
        Path(controls.__file__),
        Path(controls.baseline.__file__),
        Path(grid.__file__),
        Path(release.__file__),
        Path(sys.modules[probe_frames.__module__].__file__),
        old_states.GEOMETRY,
        native.geometry.WEIGHT,
    )
    return {path.name: sha(path) for path in paths}


def freeze(run):
    previous, archive_pin = grid.plan()
    publisher = grid.publisher_record()
    inputs = read(grid.planning.TRIAL / "bind-00/output/private-plan.json")["sources"]
    prior = {PurePosixPath(row["input"]["planned"]["member"]["name"]).stem for row in inputs}
    for path, section, member in OLD_FREEZES:
        prior.update(PurePosixPath(row[member]).stem for row in read(path)[section])
    run.mkdir(mode=0o700)
    sources = run / "sources"
    sources.mkdir(mode=0o700)
    roster, rejected = [], []
    for split, (groups, count) in SPLIT.items():
        for group in groups:
            source = next(row for row in previous.values() if row["component_index"] == group)
            publication = source["input"]["planned"]["source"]
            archive_name = publication["publisher_name"]
            package = (
                grid.planning.ARCHIVE
                / "grid-acquisition-continuation-v135"
                / f"source-{source['source_slot']:02}"
                / "archive.zip"
            )
            require(
                package.stat().st_size == publication["bytes"] == publisher[archive_name]["size"],
                "Publisher video archive size differs",
            )
            with ZipFile(package) as archive:
                members = {}
                for info in archive.infolist():
                    p = PurePosixPath(info.filename)
                    if (
                        len(p.parts) == 2
                        and p.parent.name == Path(archive_name).stem
                        and p.suffix == ".mpg"
                        and not p.name.startswith("._")
                    ):
                        require(p.stem not in members, "Duplicate archive member")
                        members[p.stem] = info
                require(len(members) == 1000, "Complete archive inventory required")
                order = sorted(
                    members.keys() - prior, key=lambda s: digest(f"phone-viseme-v1/{group}/{s}".encode())
                )
                admitted = 0
                for stem in order:
                    info = members[stem]
                    require(0 < info.file_size <= 1024**2, "Unexpected MPEG member extent")
                    body = archive.read(info)
                    case = f"{split}-{group}-{admitted:02}"
                    path = sources / f"{case}.mpg"
                    path.write_bytes(body)
                    table = probe_frames(path, timeout_s=15)
                    if (
                        table.rotation
                        or table.truncated
                        or table.interpolated
                        or [pts * table.tb for pts in table.pts] != [Fraction(i, 25) for i in range(PICTURES)]
                    ):
                        rejected.append(
                            {
                                "split": split,
                                "group": group,
                                "sha256": digest(body),
                                "member": info.filename,
                                "native_pictures": len(table.pts),
                                "reason": "native picture clock outside scope",
                            }
                        )
                        path.unlink()
                        continue
                    roster.append(
                        {
                            "id": case,
                            "group": group,
                            "split": split,
                            "ordinal": len(roster),
                            "sha256": digest(body),
                            "path": str(path),
                            "archive": str(package),
                            "publisher_name": archive_name,
                            "publisher_checksum": publisher[archive_name]["checksum"],
                            "member": info.filename,
                            "crc32": info.CRC,
                        }
                    )
                    admitted += 1
                    if admitted == count:
                        break
    require(
        len({row["sha256"] for row in roster}) == len(roster) == 66, "Incomplete or duplicate source roster"
    )
    shadow = [
        {
            **row,
            "split": "redraw",
            "id": f"redraw-{row['ordinal']}-{row['role']}",
            "group": previous[row["ordinal"]]["component_index"],
        }
        for row in incumbent.freeze(NEURAL)["sources"]
    ]
    require(
        not {row["group"] for row in shadow} & set(SPLIT["fit"][0] + SPLIT["point"][0]),
        "Redraw speakers must stay outside fit and point",
    )
    write(
        run / "freeze.json",
        {
            "spec": SPEC,
            "inputs": roster,
            "excluded_clocks": rejected,
            "redraws": shadow,
            "files": bindings(),
            "prior_exclusions": {path.parent.name: sha(path) for path, *_ in OLD_FREEZES},
            "plan_sha256": archive_pin,
            "phone_assets": phones.assets(),
            "host": [platform.system(), platform.machine(), platform.python_version()],
        },
    )


def sealed(run):
    freeze = read(run / "freeze.json")
    require(
        freeze["spec"] == SPEC
        and freeze["files"] == bindings()
        and freeze["phone_assets"] == phones.assets(),
        "Frozen method or assets differ",
    )
    return freeze


def speech_decoder():
    sys.path.insert(0, str(phones.SITE))
    from pocketsphinx import Decoder

    assets = phones.SITE / "pocketsphinx/model/en-us"
    return Decoder(
        hmm=str(assets / "en-us"),
        allphone=str(assets / "en-us-phone.lm.bin"),
        allphone_ci=True,
        samprate=RATE,
        frate=100,
        logfn="/dev/null",
    )


def phone_sequence(decoder, pcm):
    phones.decode(decoder, pcm.tobytes())
    found = [
        (seg.word, seg.start_frame * PHONE_STEP, (seg.end_frame + 1) * PHONE_STEP) for seg in decoder.seg()
    ]
    require(
        all(0 <= start < end <= len(pcm) for _, start, end in found)
        and all(previous[2] <= current[1] for previous, current in zip(found, found[1:])),
        "Phone times do not bind to the source clock",
    )
    return found


def stable_families(decoder, pcm):
    midpoints = np.arange(PICTURES) * TICK + TICK // 2
    scaled = np.rint(pcm.astype(np.float64) * 0.68).astype("<i2")
    sequences = [phone_sequence(decoder, source) for source in (pcm, scaled)]
    observed = []
    for sequence in sequences:
        view = np.full(PICTURES, "", dtype="U4")
        for phone, start, stop in sequence:
            if phone in INDEX:
                view[(start + PHONE_STEP <= midpoints) & (midpoints <= stop - PHONE_STEP)] = phone
        observed.append(view)
    stable = [str(a) if a == b else "" for a, b in zip(*observed, strict=True)]
    return np.array([INDEX.get(p, -1) for p in stable]), stable


def visual_features(observations):
    values = np.array([row if row is not None else [np.nan] * 4 for row in observations])
    present = np.isfinite(values).all(axis=1)
    if not present.any():
        return values
    widths = values[present, 3]
    values[present, 3] = widths / np.median(widths)
    values[present, :3] = np.log1p(values[present, :3] / 0.035)
    return values


def observe(row, module, task):
    path = Path(row["path"])
    require(sha(path) == row["sha256"], "Source video differs")
    table, pictures, pcm = native.video(path)
    observed = [neural.observed(module, task, frame) for frame in pictures]
    shape = visual_features([item for _, item in observed])
    boxes = [
        None if mesh is None else release.mouth_crop(mesh, picture)
        for (mesh, _), picture in zip(observed, pictures, strict=True)
    ]
    return table, pictures, pcm, shape, boxes


def fit(run):
    freeze = sealed(run)
    decoder = speech_decoder()
    module, task = neural.mp_task()
    features, truth, journal = [], [], []
    with closing(task):
        for row in freeze["inputs"]:
            if row["split"] != "fit":
                continue
            _, _, pcm, shape, _ = observe(row, module, task)
            labels, _ = stable_families(decoder, pcm)
            eligible = np.isfinite(shape).all(axis=1) & (labels >= 0)
            features.extend(shape[eligible].tolist())
            truth.extend(labels[eligible].tolist())
            journal.append(
                {
                    "id": row["id"],
                    "counts": np.bincount(labels[eligible], minlength=len(FAMILIES)).tolist(),
                    "pcm_sha256": digest(pcm.tobytes()),
                }
            )
    data, truth = np.array(features), np.array(truth)
    counts = np.bincount(truth, minlength=len(FAMILIES))
    require(np.min(counts) >= 15, f"Insufficient fitted frames per phone family: {counts}")
    mean, scale = data.mean(axis=0), np.maximum(data.std(axis=0), 0.05)
    model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=500, tol=1e-8)
    model.fit((data - mean) / scale, truth)
    require(
        max(model.n_iter_) < 500 and model.classes_.tolist() == list(range(len(FAMILIES))),
        "Shape fit did not converge over the expected classes",
    )
    write(
        run / "fit.json",
        {
            "freeze_sha256": sha(run / "freeze.json"),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
            "counts": counts.tolist(),
            "sources": journal,
        },
    )
    print("fit class counts", counts.tolist(), flush=True)


def log_shape(shape, model):
    logits = ((shape - model["mean"]) / model["scale"]) @ np.asarray(model["coefficients"]).T
    return log_softmax(logits + model["intercept"], axis=1)


def proposal(likelihood, speech):
    index = np.arange(PICTURES)
    source = index[:, None] + LAGS
    clipped = np.clip(source, 0, PICTURES - 1)
    classes = speech[clipped]
    valid = (source == clipped) & (classes >= 0) & np.isfinite(likelihood).all(axis=1)[:, None]
    compatible = likelihood[index[:, None], np.maximum(classes, 0)]
    competing = valid[:, :, None] & valid[:, None, :] & (classes[:, :, None] != classes[:, None, :])
    diff = np.where(competing, compatible[:, None, :] - compatible[:, :, None], 0.0)
    pad = np.zeros((1, len(LAGS), len(LAGS)))
    sums = np.concatenate([pad, np.cumsum(diff, axis=0)], axis=0)
    counts = np.concatenate([pad, np.cumsum(competing, axis=0)], axis=0)
    delta = LAGS[None, :] - LAGS[:, None]
    lag_mask = (abs(LAGS[:, None]) <= 5) & (abs(delta) >= 4) & (abs(delta) <= 8)
    best, measured = None, 0
    for start in range(FLANK, PICTURES - 2 * FLANK):
        for width in WIDTHS:
            end = start + width
            if end + FLANK > PICTURES:
                continue
            left, middle, right = (
                counts[start] - counts[start - FLANK],
                counts[end] - counts[start],
                counts[end + FLANK] - counts[end],
            )
            supported = lag_mask & (left >= 2) & (middle >= 3) & (right >= 2)
            if not supported.any():
                continue
            measured += 1
            gains = (
                (sums[start - FLANK] - sums[start]) / np.maximum(left, 1),
                (sums[end] - sums[start]) / np.maximum(middle, 1),
                (sums[end] - sums[end + FLANK]) / np.maximum(right, 1),
            )
            scores = np.where(supported, np.minimum.reduce(gains), -np.inf)
            i, j = np.unravel_index(scores.argmax(), scores.shape)
            value = float(scores[i, j])
            if best is None or value > best["score"]:
                best = {
                    "extent_ticks": [start, end],
                    "reference_lag_ticks": int(LAGS[i]),
                    "local_lag_ticks": int(LAGS[j]),
                    "relative_lag_ticks": int(delta[i, j]),
                    "score": value,
                    "gains": [float(gain[i, j]) for gain in gains],
                    "pair_counts": [int(count[i, j]) for count in (left, middle, right)],
                }
    return measured, best


def point_from(rows):
    measured = [row["best"]["score"] for row in rows if row["best"] is not None]
    require(len(rows) == 72, "All six ordinary controls for every point source required")
    return {"value": max(0.0, *measured), "controls": len(rows), "measured_controls": len(measured)}


def score(run, split):
    freeze = sealed(run)
    model = read(run / "fit.json")
    require(model["freeze_sha256"] == sha(run / "freeze.json"), "Fit freeze differs")
    point = None
    if split != "point":
        point = read(run / "point.json")
        require(
            point["fit_sha256"] == sha(run / "fit.json")
            and point["readout_sha256"] == sha(run / "point-readout.json"),
            "Calibration changed before held scoring",
        )
    sources = (
        freeze["redraws"] if split == "redraw" else [row for row in freeze["inputs"] if row["split"] == split]
    )
    cv2.setNumThreads(1)
    decoder = speech_decoder()
    module, task = neural.mp_task()
    detector = load_yunet(native.geometry.WEIGHT) if point is not None else None
    rows, native_records, labels, predictions = [], [], [], []
    with closing(task):
        for source in sources:
            table, pictures, pcm, shape, boxes = observe(source, module, task)
            likelihood = log_shape(shape, model)
            if detector is not None:
                observed, ticks, faces, patches, _, _ = incumbent.observe(
                    Path(source["path"]), source["sha256"], detector
                )
                require(np.array_equal(observed, pcm), "Current timing comparison used different PCM")
            if split == "redraw":
                variants = [(source["role"], pcm, source["picture_edit"], source["known_lag_ticks"])]
            else:
                variants = list(controls.variants(pcm, source["ordinal"]))
                require(tuple(item[0] for item in variants) == ROLE_ORDER, "Control schedule differs")
            native_records.append(
                {
                    "id": source["id"],
                    "pts": list(table.pts),
                    "time_base": table.time_base,
                    "rgb_sha256": [digest(frame.tobytes()) for frame in pictures],
                    "original_pcm_sha256": digest(pcm.tobytes()),
                }
            )
            for role, delivered, extent, lag in variants:
                if split == "point" and lag is not None:
                    continue
                families, stable = stable_families(decoder, delivered)
                measured, best = proposal(likelihood, families)
                if role == "recorded":
                    eligible = np.isfinite(likelihood).all(axis=1) & (families >= 0)
                    labels.extend(families[eligible].tolist())
                    predictions.extend(likelihood[eligible].argmax(axis=1).tolist())
                reported = bool(point and best and best["score"] > point["value"])
                old = (
                    controls.baseline.current(faces, patches, delivered.astype(np.float32) / 32768, ticks)
                    if detector
                    else None
                )
                rows.append(
                    {
                        "id": source["id"],
                        "group": source["group"],
                        "role": role,
                        "known_extent": extent,
                        "known_shift": lag,
                        "pcm_sha256": digest(delivered.tobytes()),
                        "stable_phone_ticks": sum(bool(phone) for phone in stable),
                        "measured": measured,
                        "best": best,
                        "reported": reported,
                        "current": old,
                        "native_evidence": old_states.exhibit(best, table, pictures, boxes, delivered)
                        if reported
                        else None,
                    }
                )
            print(split, source["id"], len(rows), flush=True)
    result = {
        "freeze_sha256": sha(run / "freeze.json"),
        "fit_sha256": sha(run / "fit.json"),
        "rows": rows,
        "point_sha256": sha(run / "point.json") if point else None,
        "native_records": native_records,
        "shape_class_accuracy": float(balanced_accuracy_score(labels, predictions)) if labels else None,
        "shape_class_counts": np.bincount(labels, minlength=len(FAMILIES)).tolist(),
    }
    write(run / f"{split}-readout.json", result)
    if split == "point":
        body = {
            **point_from(rows),
            "fit_sha256": sha(run / "fit.json"),
            "readout_sha256": sha(run / "point-readout.json"),
        }
        write(run / "point.json", body)
        print("point", body, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "point", "held", "redraw"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.run)
    elif args.phase == "fit":
        fit(args.run)
    else:
        score(args.run, args.phase)


if __name__ == "__main__":
    main()
