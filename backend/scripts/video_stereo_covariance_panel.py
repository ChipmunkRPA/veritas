# SPDX-License-Identifier: Apache-2.0
"""Freeze and replay audio covariance comparisons on exact stereo video deliveries."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import partial
from pathlib import Path

from scripts import video_shared_sound_panel as prior

SCHEMA = "chipmunk.video-stereo-covariance/1"
ARMS = ("real", "complex")
MODES = ("audio_only", "with_picture")
BUDGETS = (0, 1, 2)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def save(path, document):
    content = json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("xb") as handle:
        handle.write(gzip.compress(content, mtime=0) if path.suffix == ".gz" else content)


def delivery_file(root, name):
    if Path(name).name != name or str(name).startswith("."):
        raise ValueError("Delivery name must be a direct child of the input directory")
    return root / name


def picture_readings(bank, deliveries, assets):
    from app.detectors.video.decode import probe_frames
    from app.detectors.video.viseme_worker import capture_mouth_frames

    model = assets / "face_landmarker.task"
    if file_hash(model) != prior.GEOMETRY_SHA:
        raise ValueError("Mouth landmark asset differs from the pinned prior observation")
    sys.path.insert(0, str(assets / "runtime"))
    result = {}
    for source in bank["sources"]:
        group = source["group"]
        pictures = delivery_file(deliveries, f"{group}-pictures.mp4")
        table = probe_frames(pictures)
        if (list(table.pts) != source["ticks"] or table.tb != Fraction(1, 30000)
                or prior.ffmpeg("-i", str(pictures), *prior.STREAM_HASH).decode().strip() != source["picture_stream"]):
            raise ValueError(f"{group} pictures differ from their publisher-bound delivery")
        frames, _, _ = capture_mouth_frames(pictures, table, model, motion=False, shots=[[0, len(table)]])
        result[group] = [{**frame, "sample": round(tick * prior.RATE / 30000)} for tick, frame in zip(table.pts, frames, strict=True)]
    return result


def validate_pictures(bank, pictures):
    if set(pictures) != {source["group"] for source in bank["sources"]}:
        raise ValueError("Picture observations must cover every source group")
    for source in bank["sources"]:
        frames = pictures[source["group"]]
        if (len(frames) != len(source["ticks"])
                or any(frame["sample"] != round(tick * prior.RATE / 30000)
                       for tick, frame in zip(source["ticks"], frames, strict=True))):
            raise ValueError("Picture observations do not use the original sample clock")


def score_delivery(row, deliveries, pictures):
    from app.bench.video_stereo_covariance import scan
    from app.detectors.audio.decode import decode_audio
    from app.evidence.video_viseme_binding import presented_samples

    path = delivery_file(deliveries, row["delivery"])
    digest = file_hash(path)
    if digest != row["input_sha256"]:
        raise ValueError(f"{path.name} differs from the frozen original bytes")
    decoded = decode_audio(path, source_sha256=digest, retain_stereo=True)
    extent = presented_samples(decoded.mapping)
    stereo = decoded.stereo_pcm
    if (extent != row["samples"] or stereo is None or len(stereo) < extent
            or hashlib.sha256(stereo[:extent].astype("<i2", copy=False).tobytes()).hexdigest() != row["stereo_pcm_sha256"]):
        raise ValueError(f"{path.name} differs from its measured stereo presentation")
    fields = ("group", "snr_db", "carrier", "role", "delivery", "input_sha256", "stereo_pcm_sha256", "samples", "truth",
              "mono_baseline", "audio_eligible")
    return {key: row[key] for key in fields} | {
        "incumbent": row["with_picture"], "candidate": scan(stereo[:extent], pictures[row["group"]]),
    }


def comparisons(rows, selected):
    counts = defaultdict(lambda: defaultdict(int))
    hit_groups, ordinary_groups = set(), set()
    for row in rows:
        proposal, assessed = selected(row)
        truth = row["truth"]
        hit = bool(truth is not None and proposal and prior.overlap(truth, proposal) >= .4)
        group = row["group"]
        kind = "edit" if truth is not None else "ordinary"
        tally = counts[row["carrier"], row["snr_db"]]
        tally[f"{kind}_files"] += 1
        if assessed is not None:
            tally[f"{kind}_assessed"] += assessed
        tally["correct"] += hit
        tally["misplaced"] += truth is not None and bool(proposal) and not hit
        tally["ordinary_flags"] += truth is None and bool(proposal)
        if hit:
            hit_groups.add((row["carrier"], row["snr_db"], group))
        if truth is None and proposal:
            ordinary_groups.add((row["carrier"], row["snr_db"], group))
    return {f"{carrier}/{snr}": dict(tally) | {
        "positive_groups": sum(key[:2] == (carrier, snr) for key in hit_groups),
        "ordinary_groups": sum(key[:2] == (carrier, snr) for key in ordinary_groups),
    } for (carrier, snr), tally in sorted(counts.items())}


def active(row, arm, mode, point):
    result = row["candidate"][arm]
    chosen = result[mode]
    assessed = result["picture_assessed" if mode == "with_picture" else "audio_assessed"]
    return (chosen["span"] if chosen is not None and chosen["margin_millidb"] > point else None), assessed


def calibrate(rows):
    by_carrier = {}
    for carrier in prior.CARRIERS:
        bank = [row for row in rows if row["carrier"] == carrier]
        options = []
        for arm in ARMS:
            for mode in MODES:
                maxima = {row["group"]: 0 for row in bank}
                for row in bank:
                    if row["truth"] is None and (option := row["candidate"][arm][mode]):
                        group = row["group"]
                        maxima[group] = max(maxima[group], option["margin_millidb"])
                points = sorted(maxima.values(), reverse=True)
                for budget in BUDGETS:
                    point = max(1000, points[budget])
                    counts = comparisons(bank, partial(active, arm=arm, mode=mode, point=point))
                    utility = sum((3 if key.endswith("/-5") else 2) * value["correct"]
                                  - value["misplaced"] - 2 * value["ordinary_flags"] - value["ordinary_groups"]
                                  for key, value in counts.items())
                    options.append({"arm": arm, "mode": mode, "budget": budget, "point": point, "utility": utility})
        selected = max((item for item in options if item["mode"] == "with_picture"),
                       key=lambda item: (item["utility"], -item["budget"], -ARMS.index(item["arm"])))
        by_carrier[carrier] = {"selected": selected, "options": options}
    return by_carrier


def summarize(rows, points, baseline):
    def pick(row):
        setting = points[row["carrier"]]["selected"]
        return active(row, setting["arm"], setting["mode"], setting["point"])

    def old(row):
        setting = baseline[row["carrier"]]
        choice = row["incumbent"][setting["arm"]]
        return (choice["span"] if choice and choice["margin_millidb"] > setting["point"] else None), row["audio_eligible"]

    alternate = {}
    for carrier, document in points.items():
        bank = [row for row in rows if row["carrier"] == carrier]
        alternate[carrier] = []
        for option in document["options"]:
            arm, mode, point = (option[key] for key in ("arm", "mode", "point"))
            counts = comparisons(bank, partial(active, arm=arm, mode=mode, point=point))
            alternate[carrier].append({"arm": arm, "mode": mode, "budget": option["budget"], "counts": counts})
    return {"selected": comparisons(rows, pick), "prior_stereo": comparisons(rows, old),
            "near_silence": comparisons(rows, lambda row: (row["mono_baseline"], None)), "alternates": alternate}


def measure(bank_path, deliveries, assets, output, *, freeze=None, frames=None):
    from app.bench import video_stereo_covariance as covariance
    from app.detectors.audio import decode
    from app.evidence import video_silent_mouth

    bank = read(bank_path)
    frozen = read(freeze) if freeze else None
    sources = bank["sources"]
    pins = {path.name: file_hash(path) for path in (Path(__file__), Path(covariance.__file__), Path(decode.__file__),
                                                    Path(video_silent_mouth.__file__))}
    if (bank["split"] == "held") != bool(frozen) or frozen and (
        frozen["code"] != pins or any({row[key] for row in sources} & {row[key] for row in frozen["sources"]}
                                      for key in ("speaker", "room_sha256", "sha256"))
    ):
        raise ValueError("Held scoring requires frozen code and disjoint speaker, room and file identities")
    pictures = read(frames) if frames else picture_readings(bank, deliveries, assets)
    validate_pictures(bank, pictures)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = pool.map(lambda row: score_delivery(row, deliveries, pictures), bank["rows"])
        for count, row in enumerate(tasks, 1):
            rows.append(row)
            if count % 28 == 0:
                print(f"Verified and measured {count}/{len(bank['rows'])} deliveries", flush=True)
    points = frozen["points"] if frozen else calibrate(rows)
    baseline = frozen["baseline"] if frozen else bank["choices"]
    summary = summarize(rows, points, baseline)
    receipt = {"schema": SCHEMA, "split": bank["split"], "code": pins, "source_receipt_sha256": file_hash(bank_path),
               "freeze_sha256": file_hash(freeze) if freeze else None, "sources": sources, "rows": rows,
               "points": points, "baseline": baseline, "summary": summary}
    save(output, receipt)
    print(json.dumps(summary["selected"], sort_keys=True), flush=True)


def verify(document, freeze=None):
    if document["schema"] != SCHEMA or (document["split"] == "held") != bool(freeze):
        raise ValueError("Held comparisons require their frozen development receipt")
    if freeze:
        frozen = read(freeze)
        verify(frozen)
        if (document["freeze_sha256"] != file_hash(freeze) or document["points"] != frozen["points"]
                or document["baseline"] != frozen["baseline"] or document["code"] != frozen["code"]
                or any({row[key] for row in document["sources"]} & {row[key] for row in frozen["sources"]}
                       for key in ("speaker", "room_sha256", "sha256"))):
            raise ValueError("Held data or decisions differ from their frozen scope")
    elif document["points"] != calibrate(document["rows"]):
        raise ValueError("Development operating points differ from their retained measurements")
    if document["summary"] != summarize(document["rows"], document["points"], document["baseline"]):
        raise ValueError("Comparison counts differ from retained intervals")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure")
    for key in ("bank_path", "deliveries", "assets", "output"):
        run.add_argument(key, type=Path)
    run.add_argument("--frames", type=Path)
    recount = commands.add_parser("verify")
    recount.add_argument("receipt", type=Path)
    for mode in (run, recount):
        mode.add_argument("--freeze", type=Path)
    args = vars(cli.parse_args())
    if args.pop("command") == "measure":
        measure(**args)
    else:
        result = read(args["receipt"])
        verify(result, args["freeze"])
        print(json.dumps({key: value for key, value in result["summary"].items() if key != "alternates"}, sort_keys=True))


if __name__ == "__main__":
    main()
