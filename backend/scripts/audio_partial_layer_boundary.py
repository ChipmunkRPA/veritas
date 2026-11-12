"""Compare frozen speech layers on a previously scored short-edit panel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_partial_residual_transfer as previous
from scripts import review_audio_partial_residual_transfer as review

SCHEMA = "chipmunk.audio-partial-layer-boundary/1"
LAYERS = {"weighted_local": None, "early_local": 4, "middle_local": 12, "deep_local": 24}
BASELINE = "incumbent_frames"
BOUNDARY = "layer_boundaries"
HEADS = (*LAYERS, BOUNDARY)
ARMS = (BASELINE, *HEADS)
SEED, DIMS = 20261112, 32
WIDTHS = dict.fromkeys(LAYERS, DIMS) | {BOUNDARY: 6 * len(LAYERS)}
FRAME_CELLS = (280 + np.arange(199) * 320) // previous.CELL
METHOD = {
    "scope": "Exactly the 316 consumed prior partial-edit inputs. New representations of already exposed clips.",
    "layers": LAYERS,
    "geometry": "Use the exact prior 20 cells and original sample timeline. Frame features may use wider context.",
    "flanks": "Cell means at offsets (-3,-2) and (+2,+3), clipping indices to the first or last cell.",
    "local": "Subtract the average of the two flanks, then use the same 32 fixed signed projection coordinates for every layer.",
    "boundary": "For each layer retain relative squared distances from the center to each flank, between flanks, the minimum center-to-flank distance, the center-to-mean-flank distance, and mean immediate-neighbor distance.",
    "fit": "Reuse the unpaired overlap-weighted L2 linear objective. Fit only the prior 72 mixed and dry splice-triple views from eight fitting hosts.",
    "point": "Strict maximum of every genuine score in the independent operating-point role. No fit or point updates after scoring diagnostic or held roles.",
    "location": "Prior strongest 0.6-second span, exact interval IoU >= 0.30 and edit coverage >= 0.50. Report threshold-free span and oracle ordering separately.",
    "primary": "Layer boundary head must add at least two useful -5 dB views from two families over the unrefitted incumbent frame baseline, lose no dry useful view, add at most one false pair and flag no background-only control.",
    "interpretation": "All arms, including a successful primary, are diagnostic. Confirmation on other sources is required.",
}


def project():
    signs = np.random.Generator(np.random.PCG64(SEED)).integers(0, 2, (previous.WIDTH, DIMS), dtype=np.int8)
    return ((signs.astype(np.float32) * 2 - 1) / np.float32(np.sqrt(previous.WIDTH))).astype("<f4")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bound_inputs(parent, weights):
    source, rows = previous.frozen(parent, weights)
    require(source == previous.read(parent / "model-freeze.json"), "Original inference source or runtime differs")
    result = previous.read(parent / "result.json")
    require(result["decisions_sha256"] == previous.sha(parent / "decisions.json")
            and result["fit_sha256"] == previous.sha(parent / "fit.json"), "Original fit or decision ledger differs")
    verified = previous.read(parent / "inputs-verified.json")
    require(verified["inputs_sha256"] == source["input_manifest_sha256"] and verified["rows"] == 316,
            "Expected the verified prior native inputs")
    require(len(rows) == len({r["pcm_sha256"] for r in rows}) == 316 and
            all(r["family"] != "expressive" or r["role"] == "confirmation" for r in rows),
            "Earlier role or expressive holdout changed")
    names = ["model-freeze.json", "inputs-verified.json", "inputs.json", "fit.json", "result.json", "decisions.json"]
    names.extend(f"{role}-features.{ext}" for role in previous.ROLES for ext in ("json", "npz"))
    lock = {"schema": SCHEMA, "method": METHOD, "code_sha256": previous.sha(Path(__file__)),
            "reviewer_sha256": previous.sha(Path(review.__file__)),
            "parent": {name: previous.sha(parent / name) for name in names},
            "projection": {"seed": SEED, "dimensions": DIMS, "sha256": digest(project().tobytes())},
            "feature_widths": WIDTHS, "roles": source["role_counts"],
            "runtime": {key: importlib.metadata.version(key) for key in ("numpy", "torch", "scipy")}}
    return lock, rows


def distance(left, right):
    power = np.mean(left ** 2 + right ** 2, axis=-1)
    return np.mean((left - right) ** 2, axis=-1) / np.maximum(power, 1e-12)


def measured_features(cells, projection):
    i = np.arange(previous.CELLS)
    left = (cells[np.maximum(i - 3, 0)] + cells[np.maximum(i - 2, 0)]) * .5
    right = (cells[np.minimum(i + 2, previous.CELLS - 1)] + cells[np.minimum(i + 3, previous.CELLS - 1)]) * .5
    flanks = (left + right) * .5
    dl, dr = distance(cells, left), distance(cells, right)
    near = (distance(cells, cells[np.maximum(i - 1, 0)]) +
            distance(cells, cells[np.minimum(i + 1, previous.CELLS - 1)])) * .5
    return (cells - flanks) @ projection, np.stack((dl, dr, distance(left, right), np.minimum(dl, dr),
                                                   distance(cells, flanks), near), axis=-1)


class LayerCapture:
    def __init__(self, model):
        self.model = model
        self.handle = model.backbone.register_forward_hook(self.capture, with_kwargs=True)
        self.record = None

    def capture(self, _module, _args, kwargs, output):
        import torch

        states = output.hidden_states
        mask = kwargs["attention_mask"]
        require(self.record is None and len(states) == 25 and mask.shape == (1, 199) and bool(torch.all(mask == 1)),
                "Expected one exact previous encoder clock")
        weighted = torch.stack(states).mul(self.model.layer_weights.softmax(dim=0)[:, None, None, None]).sum(0)[0]
        selected = [weighted if layer is None else states[layer][0] for layer in LAYERS.values()]
        self.record = np.stack([torch.stack([frames[i == FRAME_CELLS].mean(0) for i in range(previous.CELLS)])
                                .cpu().numpy() for frames in selected])


def encode(run, parent, weights, role, rows):
    import torch

    from app.detectors.audio.upgraded import prepare_voice_window
    from scripts.run_audio_specialists_v15 import load_voice

    output = run / f"{role}-features.npz"
    ledger = run / f"{role}-features.json"
    require(not output.exists() and not ledger.exists(), "Role was already measured")
    require(role in previous.ROLES[:2] or (run / "fit.json").is_file(), "Seal fit and points before held encoding")
    cases, saved_weighted, _, prior_clip = previous.load_role(parent, role, rows)
    old_inputs = previous.read(parent / f"{role}-features.json")["rows"]
    model = load_voice({"weights": str(weights), "assets": previous.ASSETS})
    hook = LayerCapture(model)
    projection = project()
    features, binding = defaultdict(list), []
    started = time.monotonic()
    try:
        for index, (row, old) in enumerate(zip(cases, old_inputs, strict=True)):
            raw = previous.pcm_bytes(parent, row)
            prepared = prepare_voice_window(np.frombuffer(raw, "<i2").astype(np.float32) / 32768)
            input_sha = digest(prepared.astype("<f4", copy=False).tobytes())
            require(old == {"id": row["id"], "pcm_sha256": row["pcm_sha256"], "model_input_sha256": input_sha},
                    "Native input differs from the parent forward")
            hook.record = None
            with torch.inference_mode():
                clip = float(model(torch.from_numpy(prepared)[None]).item())
            captured = hook.record
            require(captured is not None and captured.shape == (len(LAYERS), previous.CELLS, previous.WIDTH)
                    and np.isfinite(captured).all() and np.array_equal(captured[0], saved_weighted[index])
                    and clip == prior_clip[index], "Serving layer or original response parity failed")
            parts = []
            for name, cells in zip(LAYERS, captured, strict=True):
                projected, distances = measured_features(cells, projection)
                features[name].append(projected)
                parts.append(distances)
            features[BOUNDARY].append(np.concatenate(parts, axis=-1))
            binding.append(old)
            if len(binding) % 20 == 0:
                print(f"{role}: {len(binding)} exact views", flush=True)
    finally:
        hook.handle.remove()
    with output.open("xb") as stream:
        np.savez_compressed(stream, **{arm: np.stack(values) for arm, values in features.items()})
    previous.write(ledger, {"freeze_sha256": previous.sha(run / "freeze.json"), "feature_sha256": previous.sha(output),
                            "binding": binding, "forwards": len(binding), "seconds": round(time.monotonic() - started, 3)})


def load(run, parent, role, rows):
    cases, _, original_local, _ = previous.load_role(parent, role, rows)
    path = run / f"{role}-features.npz"
    binding = previous.read(run / f"{role}-features.json")
    require(binding["freeze_sha256"] == previous.sha(run / "freeze.json")
            and binding["feature_sha256"] == previous.sha(path)
            and binding["binding"] == previous.read(parent / f"{role}-features.json")["rows"]
            and binding["forwards"] == len(cases), "Feature, source or forward commitment differs")
    with np.load(path, allow_pickle=False) as values:
        features = {arm: values[arm].astype(np.float64) for arm in HEADS}
    require(all(x.shape == (len(cases), previous.CELLS, WIDTHS[arm]) and np.isfinite(x).all()
                for arm, x in features.items()), "Feature dimensions or finite checks failed")
    return cases, features, original_local


def apply(features, original, heads):
    return {BASELINE: original} | {arm: ((features[arm] - head["center"]) / head["scale"]) @ head["weights"] + head["bias"]
                                  for arm, head in heads.items()}


def seal(run, parent, rows):
    fitting, features, _ = load(run, parent, previous.ROLES[0], rows)
    heads = {arm: previous.train(fitting, features[arm], arm) for arm in HEADS}
    calibration, features, original = load(run, parent, previous.ROLES[1], rows)
    points = {}
    for arm, responses in apply(features, original, heads).items():
        value, witness = max((previous.select(responses[i])[0], r["id"]) for i, r in enumerate(calibration) if r["label"] == 0)
        points[arm] = {"value": value, "witness": witness}
    require(points[BASELINE]["value"] == previous.read(parent / "fit.json")["points"][BASELINE]["point"],
            "Parent affine point did not reproduce")
    previous.write(run / "fit.json", {"freeze_sha256": previous.sha(run / "freeze.json"), "heads": heads, "points": points,
                                     "ledgers": {r: previous.sha(run / f"{r}-features.json") for r in previous.ROLES[:2]}})


def decisions(cases, responses, points):
    matched = previous.matched_groups(cases)
    for i, row in enumerate(cases):
        checks = {}
        for arm, values in responses.items():
            score, start, end = previous.select(values[i])
            entry = {"score": score, "range": [start, end], "flag": score > points[arm]["value"],
                     "location": previous.useful_range(start, end, row["truth"]), "rank_both": False, "oracle_rank_both": False}
            if row["label"]:
                controls = [matched[row["pair_id"], row["condition"]][ordinary] for ordinary in previous.HUMAN]
                entry["rank_both"] = all(score > previous.select(values[j])[0] for j in controls)
                weights = previous.overlap_weights(row["truth"])
                oracle = np.average(values[i], weights=weights)
                entry["oracle_rank_both"] = all(oracle > np.average(values[j], weights=weights) for j in controls)
            checks[arm] = entry
        yield {key: row[key] for key in ("id", "role", "condition", "pair_id", "family", "arm", "label", "truth")} | {"checks": checks}


def summarize(rows):
    cells = defaultdict(Counter)
    for row in rows:
        label = row["label"]
        for arm, m in row["checks"].items():
            cells[row["role"], row["condition"], arm].update(positive=label, genuine=1-label,
                ai_flags=int(bool(label and m["flag"])), useful=int(m["flag"] and m["location"]),
                false_flags=int(not label and m["flag"]), threshold_free_locations=int(m["location"]),
                rank_both=int(m["rank_both"]), oracle_rank_both=int(m["oracle_rank_both"]))
    severe = [r for r in rows if r["role"] == "confirmation" and r["condition"] == "speech_minus5db"]
    dry = [r for r in rows if r["role"] == "confirmation" and r["condition"] == "dry"]

    def located(r, arm):
        return r["checks"][arm]["flag"] and r["checks"][arm]["location"]

    gain = [r for r in severe if located(r, BOUNDARY) and not located(r, BASELINE)]
    loss = [r for r in severe if located(r, BASELINE) and not located(r, BOUNDARY)]
    false_pairs = {arm: {r["pair_id"] for r in severe if r["pair_id"] is not None and not r["label"] and r["checks"][arm]["flag"]}
                   for arm in (BASELINE, BOUNDARY)}
    outcome = {"added_useful": len(gain), "lost_useful": len(loss), "added_families": len({r["family"] for r in gain}),
               "extra_false_pairs": len(false_pairs[BOUNDARY] - false_pairs[BASELINE]),
               "lost_dry_useful": sum(located(r, BASELINE) and not located(r, BOUNDARY) for r in dry),
               "background_flags": sum(r["arm"] == "background_only" and r["checks"][BOUNDARY]["flag"] for r in severe)}
    follow_up = (outcome["added_useful"] - outcome["lost_useful"] >= 2 and outcome["added_families"] >= 2
                 and outcome["extra_false_pairs"] <= 1 and outcome["lost_dry_useful"] == outcome["background_flags"] == 0)
    return {"counts": [{"role": r, "condition": c, "arm": a, **v} for (r, c, a), v in sorted(cells.items())],
            "primary": outcome, "follow_up": follow_up, "release_eligible": False}


def report(run, parent, rows):
    fit = previous.read(run / "fit.json")
    require(fit["freeze_sha256"] == previous.sha(run / "freeze.json") and fit["ledgers"] ==
            {r: previous.sha(run / f"{r}-features.json") for r in previous.ROLES[:2]}, "Fit and calibration commitment differs")
    measured = []
    for role in previous.ROLES[2:]:
        cases, features, original = load(run, parent, role, rows)
        measured.extend(decisions(cases, apply(features, original, fit["heads"]), fit["points"]))
    previous.write(run / "decisions.json", measured)
    result = summarize(measured)
    previous.write(run / "result.json", {"schema": SCHEMA, "fit_sha256": previous.sha(run / "fit.json"),
                                        "decisions_sha256": previous.sha(run / "decisions.json"), **result})
    print(result["primary"] | {"follow_up": result["follow_up"], "release_eligible": False}, flush=True)


def verify(run, parent, rows):
    fit = previous.read(run / "fit.json")
    result = previous.read(run / "result.json")
    measured = previous.read(run / "decisions.json")
    require(result["fit_sha256"] == previous.sha(run / "fit.json") and result["decisions_sha256"] ==
            previous.sha(run / "decisions.json"), "Terminal commitment differs")
    table = {r["id"]: r for r in measured}
    checks = {}
    for role in previous.ROLES:
        cases, features, original = load(run, parent, role, rows)
        if role == previous.ROLES[0]:
            for arm in HEADS:
                checks |= review.stationarity(features[arm], cases, {arm: fit["heads"][arm]})
            continue
        scores = {BASELINE: original}
        for arm, head in fit["heads"].items():
            values = (features[arm] - head["center"]) / head["scale"]
            scores[arm] = np.einsum("ijk,k->ij", values, head["weights"]) + head["bias"]
        if role == previous.ROLES[1]:
            for arm, values in scores.items():
                value, witness = max((review.select(values[i])[0], r["id"]) for i, r in enumerate(cases) if not r["label"])
                require(review.same(value, fit["points"][arm]["value"]) and witness == fit["points"][arm]["witness"],
                        "Frozen point or source witness differs")
            continue
        groups = review.groups(cases)
        for i, row in enumerate(cases):
            saved = table[row["id"]]
            require(all(saved[key] == row[key] for key in ("role", "condition", "pair_id", "family", "arm", "label", "truth")),
                    "Saved sample role, family, condition or truth differs")
            for arm, values in scores.items():
                value, bounds = review.select(values[i])
                saved_arm = saved["checks"][arm]
                require(review.same(value, saved_arm["score"]) and bounds == saved_arm["range"]
                        and (value > fit["points"][arm]["value"]) == saved_arm["flag"]
                        and review.useful(bounds, row["truth"]) == saved_arm["location"], "Score, range, or flag differs")
                if row["label"]:
                    controls = [groups[row["pair_id"], row["condition"]][key] for key in previous.HUMAN]
                    w = np.array(review.support(row["truth"]))
                    require(all(value > review.select(values[j])[0] for j in controls) == saved_arm["rank_both"]
                            and all(w @ values[i] > w @ values[j] for j in controls) == saved_arm["oracle_rank_both"],
                            "Pair-ordering or oracle result differs")
    require({k: result[k] for k in ("counts", "primary", "follow_up", "release_eligible")} == summarize(measured),
            "Aggregate differs")
    audit = {"schema": SCHEMA, "result_sha256": previous.sha(run / "result.json"), "fitted_heads": checks,
             "exact_parent_input_views": len(rows), "source_linked_decisions": len(measured) * len(ARMS)}
    previous.write(run / "audit.json", audit)
    print(audit, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "encode", "fit", "report", "verify"))
    parser.add_argument("run", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--role", choices=previous.ROLES)
    args = parser.parse_args()
    run, parent, weights = args.run, args.parent, args.weights
    lock, rows = bound_inputs(parent, weights)
    if args.phase == "freeze":
        run.mkdir(mode=0o700)
        previous.write(run / "freeze.json", lock)
        print({"freeze_sha256": previous.sha(run / "freeze.json"), "roles": lock["roles"], "forwards": 0}, flush=True)
        return
    require(lock == previous.read(run / "freeze.json"), "Exact frozen method, runtime and parent artifacts required")
    if args.phase == "encode":
        if args.role is None:
            parser.error("encode requires --role")
        encode(run, parent, weights, args.role, rows)
    elif args.phase == "fit":
        seal(run, parent, rows)
    elif args.phase == "report":
        report(run, parent, rows)
    else:
        verify(run, parent, rows)


if __name__ == "__main__":
    main()
