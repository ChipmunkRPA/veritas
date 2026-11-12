"""Independently replay audio phase features, source bindings, and veto arithmetic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measurements(data):
    samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768
    ticks = np.arange(512)
    tap = .5 - .5 * np.cos(2 * np.pi * ticks / 512)
    frame = np.stack([samples[i:i+512] * tap for i in range(0, len(samples)-511, 128)])
    strength = np.linalg.norm(frame, axis=1)**2
    active = frame[strength > max(np.percentile(strength, 40), 1e-10)]
    if len(active) < 12:
        return None
    transform = np.fft.rfft(active)
    frequencies = np.arange(transform.shape[1]) * 31.25
    magnitude = abs(transform)
    angles = np.angle(transform)
    phase, reference = [], []
    for low, high in [(180, 600), (600, 1250), (1250, 2500), (2500, 4500)]:
        bins = np.flatnonzero((low <= frequencies) & (high > frequencies))
        weight = (magnitude[:, bins-1] * magnitude[:, bins+1])**.5 * magnitude[:, bins]
        turns = np.exp(1j * (angles[:, bins-1] + angles[:, bins+1] - 2 * angles[:, bins]))
        local = np.abs(np.einsum("ij,ij->i", turns, weight)) / np.maximum(1e-12, weight.sum(axis=1))
        triplets = []
        for a in range(4, 64, 4):
            for b in range(5, 70, 6):
                if low <= frequencies[a+b] < high:
                    mix = transform[:, a] * transform[:, b]
                    summed = transform[:, a+b]
                    triplets.append(abs(np.vdot(summed, mix)) / max(1e-12, np.linalg.norm(mix) * np.linalg.norm(summed)))
        phase.extend([np.median(local), np.diff(np.percentile(local, [25, 75]))[0],
                      statistics.median(triplets), np.percentile(triplets, 90)])
        power = np.maximum(magnitude[:, bins]**2, 1e-15)
        flat = np.exp(np.log(power).sum(axis=1)/len(bins)) * len(bins) / power.sum(axis=1)
        center = ((power @ frequencies[bins]) / power.sum(axis=1) - low) / (high-low)
        for values in (flat, center):
            reference.extend([np.median(values), np.diff(np.percentile(values, [25, 75]))[0]])
    return {"phase": phase, "magnitude": reference}


def audit(run, parent):
    lock, models, scores, public = (read(run / name) for name in ("freeze.json", "models.json", "scores.json", "aggregate.json"))
    for name in ("freeze", "models", "scores"):
        require(public[f"{name}_sha256"] == digest(run / f"{name}.json"), "Phase study hash chain changed")
    for name in ("freeze", "roster", "scores", "aggregate"):
        require(lock["parent"][f"{name}_sha256"] == digest(parent / f"{name}.json"), "Parent study hash chain changed")
    rows = read(parent / "roster.json")
    prior = read(parent / "scores.json")["rows"]
    expected = {(i, "parent") for i in range(len(rows))}
    expected.update((i, view) for i in lock["delivery_rows"] for view in ("opus", "restored"))
    verified = {}
    for seen in scores["observed"]:
        i, view = seen["id"], seen["view"]
        require((i, view) in expected and (i, view) not in verified, "Extra or duplicate phase response")
        row = rows[i]
        if view == "parent":
            data = (parent / "mono" / f"{i:03}.pcm").read_bytes()
            require(row["id"] == i and seen["sha256"] == row["mono_sha256"]
                    and seen["baseline"] == prior[i]["baseline"], "An exact archived input or baseline changed")
        else:
            suffix = "opus" if view == "opus" else "wav"
            path = run / "delivery" / f"{i:03}-{view}.{suffix}"
            decoded = decode_audio(path, source_sha256=seen["delivery_sha256"], max_seconds=3)
            require(decoded.mapping["timeline"] == seen["timeline"], "Delivery timeline changed")
            data = np.rint(decoded.pcm * 32768).astype("<i2").tobytes()
        require(len(data) == 96000 and hashlib.sha256(data).hexdigest() == seen["sha256"], "An exact mono input changed")
        independently = measurements(data)
        require((independently is None) == (seen["features"] is None), "The measured activity disposition changed")
        if independently is not None:
            for arm, values in independently.items():
                require(np.allclose(values, seen["features"][arm], rtol=0, atol=1e-11), "Feature vector differs from replay")
        verified[i, view] = seen
    require(expected == set(verified), "The frozen response roster is incomplete")
    result = {}
    for arm, model in models["models"].items():
        group_inputs = defaultdict(list)
        for row in rows:
            if row["group"] in lock["fit"]:
                packet = verified[row["id"], "parent"]["features"]
                if packet is not None:
                    group_inputs[row["family"] != "genuine", row["group"]].append(packet[arm])
        groups = Counter(label for label, _ in group_inputs)
        matrix, target, weight = [], [], []
        for (label, _), values in group_inputs.items():
            matrix.extend(values)
            target.extend([float(label)-.5] * len(values))
            weight.extend([1/(2*groups[label]*len(values))] * len(values))
        matrix, weight = np.array(matrix), np.array(weight)
        mean = np.average(matrix, axis=0, weights=weight)
        spread = np.maximum(np.sqrt(np.average((matrix-mean)**2, axis=0, weights=weight)), 1e-4)
        scaled = (matrix - mean) / spread
        design = np.vstack((scaled * np.sqrt(weight[:, None]), np.eye(scaled.shape[1])*.5))
        wanted = np.r_[np.array(target)*np.sqrt(weight), np.zeros(scaled.shape[1])]
        fit, *_ = np.linalg.lstsq(design, wanted, rcond=None)
        head = model["head"]
        require(np.allclose(head["mean"], mean, rtol=0, atol=1e-12)
                and np.allclose(head["sd"], spread, rtol=0, atol=1e-12)
                and np.allclose(head["coef"], fit, rtol=0, atol=1e-10), "The independent regularized fit changed")
        cells = defaultdict(Counter)
        selection = defaultdict(list)
        cleared_groups, cleared_rooms = set(), set()
        for (i, view), item in verified.items():
            row, point = rows[i], item["response"][arm]
            base = item["baseline"]["decision"]
            if point is not None:
                independent = math.fsum(float(a*b) for a, b in zip((np.array(item["features"][arm])-mean)/spread, fit, strict=True))
                require(math.isclose(independent, point, rel_tol=0, abs_tol=1e-10), "A recorded head response changed")
            group, label = row["group"], "genuine" if row["family"] == "genuine" else "synthetic"
            role = ("fit" if group in lock["fit"] else "selection" if group in lock["selection"] else
                    "held" if row["role"] == "transfer" else "archived-native")
            treatment = row["placement"]
            if role == "selection" and label == "synthetic" and base == "ai":
                selection[treatment].append(point)
            veto = base == "ai" and point is not None and point < model["cutoff"]
            cells[f"{role}/{view}/{treatment}/{label}"].update(
                {"planned": 1, "assessed": int(point is not None), f"baseline_{base}": 1, "withdrawn": int(veto)})
            if (role, view, treatment, label, veto) == ("held", "parent", "center", "genuine", True):
                cleared_groups.add(group)
                cleared_rooms.add(row["room"])
        allowed_mixed_losses = math.floor(len(selection["center"]) * .1)
        require(model["cutoff"] == min(min(selection["clean"]), sorted(selection["center"])[allowed_mixed_losses]),
                "The frozen selection point changed")
        recorded = public["arms"][arm]
        require(dict(cells) == recorded["cells"] and len(cleared_groups) == recorded["corrected_groups"]
                and len(cleared_rooms) == recorded["corrected_rooms"], "The matched readout changed")
        relevant = [cells[f"held/{view}/center/synthetic"]["withdrawn"] for view in ("opus", "restored")]
        clean = sum(value["withdrawn"] for key, value in cells.items() if key.startswith("held/") and "/clean/" in key)
        eligible = (cells["held/parent/center/genuine"]["withdrawn"] >= 4
                    and len(cleared_groups) >= 3 and len(cleared_rooms) == 2 and not clean and max(relevant) <= 1
                    and cells["held/parent/center/synthetic"]["withdrawn"] <= 2
                    and cells["held/parent/offset/synthetic"]["withdrawn"] <= 1)
        require(eligible == recorded["advance"], "The frozen advancement decision changed")
        result[arm] = {"cells": len(cells), "advance": eligible}
    return {"verified_views": len(verified), "fresh_deliveries": len(expected) - len(rows),
            "phase_study_sha256": digest(run / "aggregate.json"), "arms": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.parent), indent=2, sort_keys=True))
