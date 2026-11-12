# SPDX-License-Identifier: Apache-2.0
"""Source-separated acoustic prediction of clearly open and closed mouth states."""

from __future__ import annotations

import argparse
import hashlib
from contextlib import closing
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from app.detectors.image.faces import load_yunet
from scripts import grid_paired_closure_panel as native
from scripts import grid_reserve_neural_mouth as neural
from scripts import grid_reserve_neural_mouth_readout as incumbent
from scripts import grid_reserve_word_clocks as grid
from scripts import grid_single_clip_release_timing as release
from scripts import video_lipsync_local_contrast_transfer as controls
from scripts import video_lipsync_short_bracket as existing
from scripts.video_common_sources_v89 import read, require, sha, write

RATE, TICK, PICTURES = 16000, 640, 75
FRESH_PRIOR = grid.planning.BASE / "grid-fresh-release-qualification-2026-11-12"
NEURAL = grid.planning.BASE / "grid-reserve-neural-mouth-2026-11-12-admitted"
GEOMETRY = neural.ASSETS / "permissions" / "face_landmarker.task"
ROLES = ("recorded", "local_gain", "local_lowpass", "local_noise", "constant_early", "constant_late",
         "short_early", "short_late", "long_early", "long_late")
SPLIT = {"fit": ([10, 11, 12], 12), "point": ([13, 14], 7), "held": ([8, 9, 15], 9)}
BANDS = (80, 150, 260, 400, 600, 850, 1200, 1700, 2300, 3100, 4200, 5600, 7600)
LAGS = np.arange(-10, 11)
SPEC = {
    "source": "Previously unplanned exact GRID utterances from fixed publisher archives. Groups are disjoint across fit, point and held. Speaker groups have prior project exposure.",
    "split": {key: [groups, size] for key, (groups, size) in SPLIT.items()},
    "roles": list(ROLES),
    "observation": "Closed is a median three-gap aperture at most .055 and open is at least .100, normalized by eye distance. Intermediate and unavailable measurements abstain.",
    "acoustic": "Seven 25 ms Hann windows centered at steps of 20 ms from -60 to +60 ms around each native picture. Log relative spectrum in twelve fixed bands and overall energy relative to the clip 95th percentile.",
    "bands_hz": list(BANDS),
    "fit": "Balanced L2 logistic regression (C=.05) predicts open versus closed only from original aligned speech, with point/held speakers excluded. The bounded input model sees no picture, raw speaker label or edit variants.",
    "compare": "Measure local compatibility change relative to one shared reference lag on both 400 ms flanks. Use only exactly shared frames with strong visual states and at least two states in the proposed range. Rank by the smallest of the three signed alignment gains.",
    "lags_ticks": LAGS.tolist(),
    "relative_lag_ticks": [4, 8],
    "extent_ticks": [14, 18, 22],
    "flank_ticks": 10,
    "local_minimum_states": 6,
    "flank_minimum_states": 4,
    "operating_point": "Strictly above the largest ordinary maximum among the two excluded point groups, with all controls retained.",
    "localization": "At least 0.40 temporal intersection over union and signed displacement error at most two native pictures.",
    "advancement": "At least three added correct locations across two held groups, at least 75% assessed coverage on edits and ordinary views, at most one added ordinary alert, and no new constant-offset alert.",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bindings():
    paths = (
        Path(__file__), Path(native.__file__), Path(neural.__file__), Path(incumbent.__file__),
        Path(controls.__file__), Path(existing.__file__), Path(grid.__file__), Path(release.__file__),
        GEOMETRY, native.geometry.WEIGHT,
    )
    return {path.name: sha(path) for path in paths}


def freeze(out):
    known_sources, archive_pin = grid.plan()
    prior_file = FRESH_PRIOR / "freeze.json"
    earlier = {PurePosixPath(row["member"]).stem for row in read(prior_file)["roster"]}
    earlier.update(PurePosixPath(row["input"]["planned"]["member"]["name"]).stem for row in known_sources.values())
    out.mkdir(mode=0o700)
    media = out / "sources"
    media.mkdir(mode=0o700)
    rows = []
    for split, (groups, n) in SPLIT.items():
        for group in groups:
            source = next(row for row in known_sources.values() if row["component_index"] == group)
            original = source["input"]["planned"]["source"]
            archive = (grid.planning.ARCHIVE / "grid-acquisition-continuation-v135"
                       / f"source-{source['source_slot']:02}" / "archive.zip")
            require(archive.stat().st_size == original["bytes"], "Retained publisher archive size differs")
            with ZipFile(archive) as package:
                members = {
                    PurePosixPath(item.filename).stem: item
                    for item in package.infolist()
                    if len(PurePosixPath(item.filename).parts) == 2
                    and PurePosixPath(item.filename).parent.name == Path(original["publisher_name"]).stem
                    and PurePosixPath(item.filename).suffix == ".mpg"
                    and not PurePosixPath(item.filename).name.startswith("._")
                }
                require(len(members) == 1000, "Complete publisher inventory required")
                order = sorted(members.keys() - earlier, key=lambda item: digest(f"articulation-v1/{group}/{item}".encode()))
                for i, name in enumerate(order[:n]):
                    member = members[name]
                    require(0 < member.file_size < 1024**2, "Unexpected source size")
                    raw = package.read(member)
                    case = f"{split}-{group}-{i:02}"
                    filename = media / f"{case}.mpg"
                    filename.write_bytes(raw)
                    rows.append({
                        "id": case, "split": split, "group": group, "ordinal": len(rows), "sha256": digest(raw),
                        "file": str(filename), "archive": str(archive), "archive_member": member.filename, "member_crc": member.CRC,
                    })
    require(len({row["sha256"] for row in rows}) == len(rows) == 77, "Incomplete or duplicate source roster")
    write(out / "freeze.json", {"spec": SPEC, "bindings": bindings(), "plan_sha256": archive_pin,
                                "prior_sha256": sha(prior_file), "sources": rows, "neural": incumbent.freeze(NEURAL)["sources"]})


def seal(out):
    frozen = read(out / "freeze.json")
    require(frozen["spec"] == SPEC and frozen["bindings"] == bindings(), "Frozen method differs")
    return frozen


def states(observations):
    aperture = np.array([np.nan if item is None else np.median(item[:3]) for item in observations])
    return np.where(aperture <= .055, 0, np.where(aperture >= .100, 1, -1))


def features(pcm):
    radius = 200
    starts = np.arange(PICTURES)[:, None] * TICK + np.arange(-3, 4)[None, :] * 320
    windows = starts[:, :, None] + np.arange(-radius, radius)[None, None, :]
    padded = np.pad(pcm.astype(np.float64) / 32768, (1600, 1600))
    audio = padded[windows + 1600]
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(radius * 2), axis=2)) ** 2
    frequencies = np.fft.rfftfreq(2 * radius, 1 / RATE)
    bands = np.stack([spectrum[..., (frequencies >= lo) & (frequencies < hi)].sum(axis=2)
                      for lo, hi in zip(BANDS[:-1], BANDS[1:], strict=True)], axis=2)
    power = np.log10(np.maximum(bands.sum(axis=2), 1e-8))
    relative = np.log10(np.maximum(bands, 1e-8)) - power[..., None]
    level = power - np.percentile(power, 95)
    values = np.concatenate([relative, np.clip(level, -7, .5)[..., None]], axis=2).reshape(PICTURES, -1)
    return values, np.max(level[:, 1:-1], axis=1) > -2.7


def observe(path, pin, module, task):
    require(sha(path) == pin, "Original media hash differs")
    table, pictures, pcm = native.video(path)
    observed = [neural.observed(module, task, picture) for picture in pictures]
    values = states([value for _, value in observed])
    boxes = [None if mesh is None else release.mouth_crop(mesh, picture)
             for (mesh, _), picture in zip(observed, pictures, strict=True)]
    return table, pictures, pcm, values, boxes


def fit(out):
    lock = seal(out)
    X, y, bindings_by_source = [], [], []
    module, task = neural.mp_task()
    with closing(task):
        for row in lock["sources"]:
            if row["split"] != "fit":
                continue
            _, pictures, pcm, state, _ = observe(Path(row["file"]), row["sha256"], module, task)
            values, active = features(pcm)
            selected = active & (state >= 0)
            X.extend(values[selected])
            y.extend(state[selected])
            bindings_by_source.append({"id": row["id"], "counts": np.bincount(state[selected], minlength=2).tolist(),
                                       "pcm_sha256": digest(pcm.tobytes()),
                                       "pictures_sha256": digest(b"".join(p.tobytes() for p in pictures))})
    X, y = np.array(X), np.array(y)
    counts = np.bincount(y, minlength=2)
    require(np.min(counts) >= 120, f"Too few supervised clean speech states: {counts}")
    mean, scale = X.mean(axis=0), np.maximum(X.std(axis=0), .05)
    lr = LogisticRegression(C=.05, class_weight="balanced", max_iter=500, tol=1e-8)
    lr.fit(np.clip((X - mean) / scale, -5, 5), y)
    require(lr.n_iter_[0] < 500, "Fit did not converge")
    write(out / "fit.json", {"freeze_sha256": sha(out / "freeze.json"), "mean": mean.tolist(), "scale": scale.tolist(),
                            "coef": lr.coef_[0].tolist(), "bias": float(lr.intercept_[0]), "counts": counts.tolist(),
                            "sources": bindings_by_source})


def predict(pcm, head):
    values, active = features(pcm)
    standardized = np.clip((values - head["mean"]) / head["scale"], -5, 5)
    return expit(standardized @ head["coef"] + head["bias"]), active


def proposal(state, pred, active):
    ticks = np.arange(PICTURES)
    indices = ticks[:, None] + LAGS
    in_range = (indices >= 0) & (indices < PICTURES)
    lookup = np.clip(indices, 0, PICTURES - 1)
    supported = in_range & active[lookup] & (state[:, None] >= 0)
    compat = (pred[lookup] - .5) * (2 * state[:, None] - 1)
    common = supported[:, :, None] & supported[:, None, :]
    diff = np.where(common, compat[:, None, :] - compat[:, :, None], 0.)
    zero = np.zeros((1, len(LAGS), len(LAGS)))
    sums = np.concatenate([zero, np.cumsum(diff, axis=0)], axis=0)
    count = np.concatenate([zero, np.cumsum(common, axis=0)], axis=0)
    opened = np.concatenate([zero, np.cumsum(common & (state[:, None, None] == 1), axis=0)], axis=0)
    relative = LAGS[None, :] - LAGS[:, None]
    valid_lag = (abs(relative) >= 4) & (abs(relative) <= 8)
    best, assessed = None, 0
    for start in range(10, 54):
        for length in (14, 18, 22):
            end = start + length
            if end + 10 > PICTURES:
                continue
            left_n, middle_n, right_n = count[start] - count[start-10], count[end] - count[start], count[end+10] - count[end]
            open_n = opened[end] - opened[start]
            available = valid_lag & (left_n >= 4) & (middle_n >= 6) & (right_n >= 4) & (open_n >= 2) & (middle_n-open_n >= 2)
            if not available.any():
                continue
            assessed += 1
            left = (sums[start-10] - sums[start]) / np.maximum(left_n, 1.)
            middle = (sums[end] - sums[start]) / np.maximum(middle_n, 1.)
            right = (sums[end] - sums[end+10]) / np.maximum(right_n, 1.)
            gain = np.minimum(np.minimum(left, middle), right)
            masked = np.where(available, gain, -np.inf)
            i, j = np.unravel_index(masked.argmax(), masked.shape)
            score = float(masked[i, j])
            if best is None or score > best["score"]:
                best = {"extent_ticks": [start, end], "reference_lag_ticks": int(LAGS[i]),
                        "local_lag_ticks": int(LAGS[j]), "relative_lag_ticks": int(relative[i, j]), "score": score,
                        "gain": [float(x[i, j]) for x in (left, middle, right)],
                        "measured_ticks": [int(x[i, j]) for x in (left_n, middle_n, right_n)]}
    return {"assessed": assessed, "best": best}


def exhibit(result, table, pictures, boxes, pcm):
    start, end = result["extent_ticks"]
    center = (start + end) // 2
    chosen = [i for i in (start-1, center, end) if boxes[i] is not None]
    rows = []
    for i in chosen:
        x0, y0, x1, y1 = boxes[i]
        rows.append({"frame": i, "pts": table.pts[i], "time_base": table.time_base,
                     "rgb_sha256": digest(pictures[i].tobytes()), "crop_xyxy": boxes[i],
                     "crop_sha256": digest(pictures[i][y0:y1, x0:x1].tobytes())})
    a, b = start * TICK, end * TICK
    return {"pictures": rows, "audio_samples": [a, b], "audio_sha256": digest(pcm[a:b].tobytes())}


def score(out, split):
    lock, head = seal(out), read(out / "fit.json")
    require(head["freeze_sha256"] == sha(out / "freeze.json"), "Fit intake differs")
    point = read(out / "point.json") if split != "point" else None
    if point:
        require(point["fit_sha256"] == sha(out / "fit.json") and point["cases_sha256"] == sha(out / "point-readout.json"),
                "Operating point provenance differs")
    cv2.setNumThreads(1)
    module, task = neural.mp_task()
    detector = load_yunet(native.geometry.WEIGHT) if point else None
    rows, source_audit, class_predictions, class_truth = [], [], [], []
    sources = lock["sources"] if split != "neural" else [
        {**row, "file": row["path"], "split": "neural", "id": f"neural-{row['ordinal']}-{row['role']}",
         "group": row["ordinal"] // 2} for row in lock["neural"]]
    with closing(task):
        for row in sources:
            if row["split"] != split:
                continue
            path = Path(row["file"])
            table, pictures, pcm, state, boxes = observe(path, row["sha256"], module, task)
            probability, activity = predict(pcm, head)
            chosen = activity & (state >= 0)
            if split != "neural":
                class_predictions.extend(probability[chosen].tolist())
                class_truth.extend(state[chosen].tolist())
            source_audit.append({"id": row["id"], "pts": list(table.pts), "time_base": table.time_base,
                                 "rgb_sha256": [digest(picture.tobytes()) for picture in pictures],
                                 "pcm_sha256": digest(pcm.tobytes()), "state_counts": np.bincount(state + 1, minlength=3).tolist()})
            if detector is not None:
                observed, ticks, faces, patches, _, _ = incumbent.observe(path, row["sha256"], detector)
                require(np.array_equal(pcm, observed), "Current and candidate use different decoded audio")
            if split == "neural":
                variations = [(row["role"], pcm, row["picture_edit"], row["known_lag_ticks"])]
            else:
                variations = list(controls.variants(pcm, row["ordinal"]))
                require(tuple(name for name, *_ in variations) == ROLES, "Known control roster differs")
            for role, delivered, extent, lag in variations:
                pred, active = predict(delivered, head)
                found = proposal(state, pred, active)
                best = found["best"]
                flagged = bool(point and best and best["score"] > point["value"])
                current = existing.current(faces, patches, delivered.astype(np.float32)/32768, ticks) if detector else None
                if flagged:
                    found["evidence"] = exhibit(best, table, pictures, boxes, delivered)
                rows.append({"id": row["id"], "group": row["group"], "role": role, "known_extent": extent,
                             "known_shift": lag, "pcm_sha256": digest(delivered.tobytes()),
                             "candidate": found, "flagged": flagged, "current": current})
            print(split, row["id"], len(rows), flush=True)
    payload = {"freeze_sha256": sha(out / "freeze.json"), "fit_sha256": sha(out / "fit.json"),
               "point_sha256": sha(out / "point.json") if point else None, "rows": rows, "source_audit": source_audit,
               "open_close_auc": roc_auc_score(class_truth, class_predictions) if class_truth else None,
               "open_close_counts": np.bincount(class_truth, minlength=2).tolist() if class_truth else []}
    write(out / f"{split}-readout.json", payload)
    if split == "point":
        ordinary = [row["candidate"]["best"] for row in rows if row["known_shift"] is None]
        measured = [rec["score"] for rec in ordinary if rec is not None]
        require(len(ordinary) == 84 and len(measured) >= 42, "Not enough measured calibration controls")
        write(out / "point.json", {"fit_sha256": sha(out / "fit.json"), "cases_sha256": sha(out / "point-readout.json"),
                                  "value": max(0., max(measured)), "measured": len(measured), "total": len(ordinary)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "point", "held", "neural"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.out)
    elif args.phase == "fit":
        fit(args.out)
    else:
        score(args.out, args.phase)


if __name__ == "__main__":
    main()
