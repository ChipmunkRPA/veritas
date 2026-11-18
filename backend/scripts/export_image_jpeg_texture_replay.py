# SPDX-License-Identifier: Apache-2.0
"""Export audited JPEG masks and diagnose whether the signal exists at known edit sites."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from shutil import copyfile

import numpy as np
from safetensors.torch import load_file

from app.bench import image_jpeg_texture_prediction as detector
from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_locations
from scripts.audio_native_alignment_panel import digest, read, save
from scripts.export_image_jpeg_band_review import portable_mask
from scripts.export_image_region_rank_review import pack_truth
from scripts.image_cross_quality_fit import load_mask
from scripts.image_jpeg_texture_prediction_panel import _body, _inputs, _method, _runtimes, review
from scripts.image_region_rank_inputs import POSITIVE


def _coverage(truth, grid):
    n = detector.CELL
    h, w = grid
    return truth[:h*n, :w*n].reshape(h, n, w, n).sum(axis=(1, 3))


def _geometry_possible(coverage, truth_count):
    """Upper bound that allows any union of native cells, including disconnected ones."""
    ordered = np.sort(coverage.ravel())[::-1]
    covered = np.cumsum(ordered)
    area = np.arange(1, len(ordered) + 1) * detector.CELL ** 2
    union = truth_count + area - covered
    return bool(np.any((2 * covered >= truth_count) & (10 * covered >= 3 * union)))


def _diagnose(examples):
    by_case = {(group["tag"], item["quality"], item["role"]): (directory, item, group, maps)
               for directory, item, group, maps in examples}
    totals = defaultdict(lambda: defaultdict(Counter))
    for directory, item, group, maps in examples:
        if item["role"] not in POSITIVE:
            continue
        truth = load_mask(directory, item)
        first = next(iter(maps.values()))
        weights = _coverage(truth, first.shape)
        if not weights.any():
            raise ValueError("Known edit lies entirely outside the working JPEG grid")
        exact_geometry = _geometry_possible(weights, int(truth.sum()))
        original = by_case[group["tag"], item["quality"], "ordinary"][3]
        blurred = by_case[group["tag"], item["quality"], "local_blur"][3]
        key = f"{item['quality']}/{group['family']}"
        weights = weights / weights.sum()
        for arm, values in maps.items():
            score = float(np.sum(values * weights))
            ordinary = float(np.sum(original[arm] * weights))
            blur = float(np.sum(blurred[arm] * weights))
            totals[key][arm].update(edited=1, grid_can_localize=exact_geometry, higher_than_original=score > ordinary,
                                    higher_than_blur=score > blur, higher_than_both=score > max(ordinary, blur))
    return totals


def export(bank, ordinary_fills, intake, frozen, outcome, output):
    review(frozen / "freeze.json.gz", outcome)
    seal = read(frozen / "freeze.json.gz")
    evaluated = read(outcome)
    if (seal["code"] != _method() or seal["weight_sha256"] != digest(frozen / "texture.safetensors")
            or evaluated["input_manifests"] != [digest(p / "manifest.json") for p in (bank, ordinary_fills)]):
        raise ValueError("Source inventories or weights differ from the held evaluation")
    groups = {row["fingerprints"]["original"]["rgb_sha256"]: row for row in read(intake / "intake.json")["cases"]}
    detector_output = {(row["file"], row["sha256"]): row for row in evaluated["rows"]}
    run = _runtimes()
    net = detector.model()
    net.load_state_dict(load_file(frozen / "texture.safetensors"), strict=True)
    net.eval()
    output.mkdir(mode=0o700)
    masks = output / "masks"
    masks.mkdir(mode=0o700)
    manifest, input_index, location_truth, predictions = [], [], [], []
    expected, native_examples = {}, []
    photos_checked = set()
    measurement_cache = {}
    for path in (bank, ordinary_fills):
        for directory, item, group in _inputs(path):
            file, sha = item["file"], item["sha256"]
            recorded = detector_output.pop((file, sha))
            source_photo = groups[group["rgb_sha256"]]
            source_file = intake / f"{source_photo['tag']}.png"
            source_sha = source_photo["source"]["original"]["sha256"]
            if source_file not in photos_checked:
                if digest(source_file) != source_sha:
                    raise ValueError("The credited original photo differs from its published bytes")
                photos_checked.add(source_file)
            body = _body(directory, item)
            if recorded["mask_sha256"] != item["mask_sha256"] or recorded["source_group"] != group["rgb_sha256"]:
                raise ValueError("Held report does not point to the same JPEG and changed pixels")
            if sha not in measurement_cache:
                low, ring, target = detector.measure(body)
                with run.inference_mode():
                    values = net(run.from_numpy(low[None]), run.from_numpy(ring[None]))
                measurement_cache[sha] = detector.errors(*(row.numpy()[0] for row in values), target)
            fields = measurement_cache[sha]
            native_examples.append((directory, item, group, fields))
            q = str(item["quality"])
            choice = seal["choices"][q]["options"][seal["choices"][q]["preferred"]]
            arm = choice["arm"]
            score, selected = detector.locate(fields[arm])
            if score != recorded["checks"][arm]["score"]:
                raise ValueError("A held location score differs from the separately replayed JPEG")
            accepted = score is not None and score > choice["point"]
            width, height = group["shape"][1], group["shape"][0]
            identifier = f"{group['tag']}-{Path(file).stem}"
            case = {"id": identifier, "input_sha256": sha}
            is_edit = int(item["role"] in POSITIVE)
            manifest.append(case | {"split": "confirmation", "modality": "image", "label": is_edit,
                                    "group": source_sha, "family": group["family"], "condition": f"jpeg{q}/{item['role']}",
                                    "source_sha256": source_sha, "lineage": [source_sha],
                                    "license": {"id": "CC BY 2.0 photograph and CC BY 4.0 dataset subject mask",
                                                "evaluation_permitted": True, "redistribution_permitted": True},
                                    "localization_truth": {"unit": "pixels", "source_units": width * height,
                                                           "target_units": item["changed_pixels"]}})
            # Each case uses the exact inputs already retained by the reviewer at these paths.
            input_index.append({"id": identifier, "input": str(directory / file), "source": str(source_file)})
            cover = {"regions": []}
            if is_edit:
                cover = {"mask": pack_truth(directory / item["mask"], item["mask_sha256"],
                                            masks / f"{identifier}-truth.pbm", [width, height])}
            else:
                load_mask(directory, item)
            location_truth.append(case | {"extent": [width, height]} | cover)
            if accepted:
                grid_y, grid_x = selected.nonzero()
                box = [int(grid_x.min()) * detector.CELL, int(grid_y.min()) * detector.CELL,
                       (int(grid_x.max()) + 1) * detector.CELL, (int(grid_y.max()) + 1) * detector.CELL]
                if box != recorded["checks"][arm]["box"]:
                    raise ValueError("Native JPEG region bounds differ from the frozen proposal")
                native = selected.repeat(detector.CELL, 0).repeat(detector.CELL, 1)
                pixels = np.zeros((height, width), bool)
                pixels[:native.shape[0], :native.shape[1]] = native
                local = {"mask": portable_mask(pixels, masks / f"{identifier}-candidate.pbm")}
                counts = recorded["checks"][arm]
                expected[identifier] = (counts["selected"], counts["intersection"])
            else:
                local = {"regions": []}
                expected[identifier] = (0, 0)
            predictions.append(case | {"status": "ok", "response": score or 0., "localization": {"status": "ok"} | local})
    if detector_output:
        raise ValueError("Some held results have no delivered source photo")

    inventory = {"schema": independent.SCHEMA, "cases": manifest}
    audited = independent.audit_manifest(inventory)
    binding = {"manifest_sha256": audited["manifest_sha256"], "split": "confirmation"}
    truth = binding | {"schema": TRUTH_SCHEMA, "cases": location_truth}
    response = binding | {"system": "JPEG intact-photo texture prediction", "method_kind": "weighted",
                          "implementation_sha256": independent.digest(seal["code"]), "weight_sha256": [seal["weight_sha256"]],
                          "response_units": "Uncalibrated texture response", "score_kind": "raw", "predictions": predictions}
    areas = count_locations(inventory, truth, response, mask_directory=masks)
    for row in areas["predictions"]:
        loc = row["localization"]
        if expected[row["id"]] != (loc["reported_units"], loc["overlap_units"]):
            raise ValueError("Independent mask intersection differs from the source-bound region")
    # Keep the exported index portable even when the source archives live on separate volumes.
    source_root = output / "source-files"
    source_root.mkdir(mode=0o700)
    for row in input_index:
        for role in ("input", "source"):
            original = Path(row[role])
            relative = source_root / (digest(original) + original.suffix)
            if not relative.exists():
                copyfile(original, relative)
            row[role] = relative.relative_to(output).as_posix()
    file_index = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": input_index}
    checked = verify_inputs(inventory, file_index, output)
    for name, content in (("manifest", inventory), ("file-index", file_index), ("truth", truth), ("predictions", response),
                          ("recounted-areas", areas)):
        save(output / f"{name}.json", content)
    save(output / "summary.json", {"groups": audited["groups"], "processed_inputs": checked,
                                   "independently_recounted_regions": sum(count > 0 for count, _ in expected.values()),
                                   "source_mask_storage": "local PBM derived from exact pre-final-encode changes",
                                   "oracle_detail": _diagnose(native_examples),
                                   "evaluation_sha256": digest(outcome), "areas_sha256": independent.digest(areas)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("bank", "ordinary_fills", "intake", "frozen", "outcome", "output"):
        cli.add_argument(field, type=Path)
    export(**vars(cli.parse_args()))
