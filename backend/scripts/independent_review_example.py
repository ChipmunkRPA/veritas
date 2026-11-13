"""Create authored example receipts for the public review command."""

from __future__ import annotations

from app.bench.independent import LOCALIZATION_UNITS, SCHEMA, compare, digest, freeze
from app.bench.independent_locations import TRUTH_SCHEMA
from app.bench.independent_replication import replicate_locations

EXTENTS = {"audio": 48_000, "image": 1_000, "text": 500, "video": 100}
BASELINE = ((.2, 0, 0), (.8, 10, 0), (.2, 0, 0), (.8, 9, 7), (.3, 3, 1), (None, 0, 0))
CANDIDATE = ((.2, 0, 0), (.4, 0, 0), (.2, 0, 0), (.8, 9, 8), (.75, 9, 8), (None, 0, 0))


def _manifest():
    cases = []
    for modality, extent in EXTENTS.items():
        for split, labels in (("development", (0, 1)), ("confirmation", (0, 0, 0, 1, 1, 1))):
            for index, label in enumerate(labels):
                group = f"example/{modality}/{split}/{index}"
                cases.append({"id": group, "modality": modality, "split": split, "group": group,
                              "label": label, "family": "fictional label", "condition": "matched example",
                              "source_sha256": digest({"example_source": group}), "input_sha256": digest({"example_input": group}),
                              "lineage": [group], "license": {"id": "authored example, no source media",
                                                            "evaluation_permitted": True, "redistribution_permitted": True},
                              "localization_truth": {"unit": LOCALIZATION_UNITS[modality], "source_units": extent,
                                                     "target_units": extent // 10 if label else 0}})
    return {"schema": SCHEMA, "purpose": "Authored demonstration of receipt arithmetic. No detector was executed.", "cases": cases}


def _receipt(manifest, system, split, values):
    outcomes = []
    for case in manifest["cases"]:
        if case["split"] != split:
            continue
        index = int(case["id"].rsplit("/", 1)[-1])
        response, reported, overlap = values[index]
        outcome = {"id": case["id"], "input_sha256": case["input_sha256"], "response": response}
        if response is None:
            reason = "Example source mapping unavailable"
            outcome.update(status="inconclusive", reason=reason, localization={"status": "not_applicable", "reason": reason})
        else:
            scale = case["localization_truth"]["source_units"] // 100
            outcome.update(status="ok", localization={"status": "ok", "reported_units": reported * scale,
                                                     "overlap_units": overlap * scale})
        outcomes.append(outcome)
    return {"manifest_sha256": digest(manifest), "split": split, "system": system,
            "implementation_sha256": digest({"authored_method": system}), "method_kind": "model_free",
            "weight_sha256": [], "score_kind": "raw", "response_units": "fictional response units", "predictions": outcomes}


def _location_example(manifest, confirmation, *, moved=False):
    cases = {case["id"]: case for case in manifest["cases"] if case["split"] == "confirmation"}
    truth = {"schema": TRUTH_SCHEMA, "manifest_sha256": digest(manifest), "split": "confirmation", "cases": []}
    predictions = []
    for row in confirmation["predictions"]:
        case = cases[row["id"]]
        geometry = case["localization_truth"]
        source, target = geometry["source_units"], geometry["target_units"]
        image = case["modality"] == "image"
        extent = [source // 10, 10] if image else [source]
        def region(start, size, image=image):
            if size == 0:
                return []
            return [[start // 10, 0, (start + size) // 10, 10]] if image else [[start, start + size]]
        target_start = source // 2 if case["label"] else 0
        truth["cases"].append({"id": row["id"], "input_sha256": row["input_sha256"],
                               "extent": extent, "regions": region(target_start, target)})
        location = row["localization"]
        if location["status"] == "ok":
            reported, overlap = (location[k] for k in ("reported_units", "overlap_units"))
            start = target_start + target - overlap
            if moved and case["label"] and overlap < reported and overlap > 0:
                # Move an equal-size highlight across to the other side of the truth.
                start = target_start - (reported - overlap)
            location = {"status": "ok", "regions": region(start, reported)}
        predictions.append({**row, "localization": location})
    return truth, {**confirmation, "predictions": predictions}


def create(directory, write):
    directory.mkdir()
    manifest = _manifest()
    points = dict.fromkeys(EXTENTS, .5)
    write(directory / "manifest.json", manifest)
    write(directory / "points.json", points)
    systems = []
    for name, values in (("baseline", BASELINE), ("candidate", CANDIDATE)):
        system = f"authored example {name}"
        development = _receipt(manifest, system, "development", ((.2, 0, 0), (.8, 9, 7)))
        lock = freeze(manifest, development, thresholds=points)
        confirmation = _receipt(manifest, system, "confirmation", values)
        confirmation["freeze_sha256"] = digest(lock)
        for suffix, data in (("development", development), ("freeze", lock), ("confirmation", confirmation)):
            write(directory / f"{name}-{suffix}.json", data)
        if name == "candidate":
            truth, response = _location_example(manifest, confirmation)
            _, repeated = _location_example(manifest, confirmation, moved=True)
            write(directory / "location-truth.json", truth)
            write(directory / "candidate-locations.json", response)
            write(directory / "candidate-repeat-locations.json", repeated)
            write(directory / "location-replication.json", replicate_locations(manifest, truth, response, repeated, lock))
        systems.extend((confirmation, lock))
    write(directory / "comparison.json", compare(manifest, *systems, require_freeze_commitment=True))
