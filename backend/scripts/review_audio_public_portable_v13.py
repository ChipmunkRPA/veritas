"""Portable public audio-v13 replay with numerical libraries only.

No app, neural runtime, pretrained weights, media decoder or source audio is
required. Prior experiments are checksum-verified inputs; this command rebuilds
v13's receipt, normalization, own calibration, all outcomes/statistics/gates and
optional exact fitting-only refits. It does not re-execute native model audits.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BLOCKED = {"torch", "torchaudio", "transformers", "app", "fastapi", "cv2", "PIL", "matplotlib"}


class NoNeuralOrAppImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError("Portable numerical review forbids " + fullname)
        return None


sys.meta_path.insert(0, NoNeuralOrAppImports())
sys.path.insert(0, str(PROJECT / "backend"))

# Install the import policy before loading any third-party or project module.
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402

from scripts import audio_wavlm_codec_v13 as codec  # noqa: E402
from scripts.audio_calibration_v3_methods import clipped_logit  # noqa: E402
from scripts.audio_guard_v7 import initial  # noqa: E402
from scripts.audio_multisource_v8 import balanced_weights, final  # noqa: E402
from scripts.evaluate_audio_multisource_v8 import summarize  # noqa: E402
from scripts.review_audio_noise_v4 import valid  # noqa: E402

CAPSULE_SHA = "84848472fef312c5fb5fd2ceddacd5fbe217caebadd09aacd85f28f5ee21cac3"
SOURCE_SHA = "526a660d812c3afec959fb45946101d643181816e4b7c4248e93559280adbf83"
PRIOR_SHA = "0546940a656ce0f718318975491ee041f2a429be15ac57d316e8e5e3ea6345c3"
FREEZE_SHA = "34096f6cd001c27e217ba7ff8ee5f2ff6bae48d85c4d91b9424cb78e2ef3b4a1"
LINEAGE = {
    "lock_sha256": "5efc68aef52a5d45459da94321f85cefea77af8c5d3a22f090f2d2d9211ed54f",
    "receipt_sha256": "33ede5fb2efc5ecb5bffb406d024417ad37385ee9362db41ca7a8948e5c8dac2",
    "fit_sha256": "f29e122f7358b5850cfeed0ef3f705a38011cfcdfa6a172ec51fa2242c6056b8",
    "head_sha256": "23350d709c9dc4c968066e2ab36bb16792f5c18d2513408b5f5c3298a8f83ae9",
    "responses_sha256": "1ab0c272491d0b14a7718c02ca888d550bb2a30c5868defa808f503d7528b48f",
    "summary_sha256": "260e820dc2b008a5a9cf6b8893d25ce39ddc33fcf800a472dbf55adfac7997ce",
    "native_sha256": "622673e88eb3c7388fa9e6f440a7d405e59182897be8e77230f2f820a6b0e6dc",
    "numerical_sha256": "afecc10f91bd8dc9242a1397bbfa56c71beb923670c1e2828ee8b88ebb617a2f",
}
PROBES_SHA = "eeaa37622fae390f708490f1e8fdee16dba8e7fce92ef66db4eb595ac9476fc3"
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}
ARMS = ("standalone", "complement")
META = ("id", "base_id", "group", "role", "label", "generator", "dataset", "condition", "sha256", "pcm_sha256", "background_id")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError("Changed public artifact: " + path.name)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return json.load(stream)


def exact_object(value, expected):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Original artifact object commitment differs")


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, (int, float)):
        return math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-10)
    return type(a) is type(b) and a == b


def frozen_fitter(freeze):
    """Load only four verified numerical definitions, without neural imports.

The published fitter remains shared. Constants come from the checksum-bound
prospective method. No imports, classes, CLI, source loaders or checkpoint code
are executed from that file; its complete source checksum must still match.
"""
    name = "backend/scripts/audio_wavlm_response_v13.py"
    path = PROJECT / name
    if digest(path) != freeze["lock"]["implementation_sha256"][name]:
        raise ValueError("Original declared fitter source changed")
    selected = {"feature", "design", "fit", "select"}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in selected]
    if {n.name for n in nodes} != selected:
        raise ValueError("Declared numerical fitter definitions absent")
    namespace = {"np": np, "minimize": minimize, "expit": expit, "balanced_weights": balanced_weights,
                 "clipped_logit": clipped_logit, "valid": valid, "DIMENSIONS": 4608,
                 "ARMS": ARMS, "METHOD": freeze["lock"]["method"]}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)
    return namespace


def probability(row, fitted):
    if row["raw"] is None:
        if row["wavlm"] is not None:
            raise ValueError("Unassessed input cannot acquire a score")
        return None
    if not valid(row["raw"]):
        raise ValueError("Finite original raw response required")
    transform = fitted["transform"]
    vector = np.asarray(row["wavlm"], np.float64)
    center, scale, beta = [np.asarray(x, np.float64) for x in
                           (transform["center"], transform["scale"], fitted["coefficients"])]
    arm = fitted["arm"]
    width = 4608 + (2 if arm == "complement" else 1)
    if (arm not in ARMS or vector.shape != (4608,) or center.shape != vector.shape or scale.shape != vector.shape
            or beta.shape != (width,) or not np.isfinite(vector).all() or not np.isfinite(center).all()
            or not np.isfinite(scale).all() or not np.isfinite(beta).all() or (scale < 1e-6).any()
            or transform["variance_factor"] != math.sqrt(16 / 4608)
            or (arm == "complement" and beta[0] < 0)):
        raise ValueError("Complete finite original fitted mapping required")
    offset = int(arm == "complement")
    margin = ((vector - center) / scale * transform["variance_factor"]) @ beta[offset:-1] + beta[-1]
    if offset:
        margin += clipped_logit(row["raw"]) * beta[0]
    return float(expit(margin))


def original_inputs(folder):
    source = load(folder / "audio-v9-identity-plan.json.gz", SOURCE_SHA)
    prior = load(folder / "audio-v9-development-review.json.gz", PRIOR_SHA)
    expected = source["variants"]["development"]
    if len(expected) != 4248 or len(prior["features"]) != 4248:
        raise ValueError("Complete original source and raw-feature input matrix required")
    rows = []
    for variant, feature in zip(expected, prior["features"], strict=True):
        if any(variant[k] != feature[k] for k in ("id", "role", "sha256", "pcm_sha256")):
            raise ValueError("Public original source/feature correspondence differs")
        row = dict(variant)
        row["base_id"] = row.pop("base_id_sha256")
        row["background_id"] = row["background_sha256"]
        row.update(raw=feature["raw"], status=feature["status"])
        rows.append(row)
    baseline = {r["id"]: {k: r[k] for k in ("original_p", "original_decision", "status")} for r in prior["outcomes"]}
    return source, rows, baseline


def read_features(body, folder, expected):
    packets = body["feature_packets"]
    if set(packets) != set(COUNTS):
        raise ValueError("All original fitting/calibration/development roles required")
    packed = {}
    for role, count in COUNTS.items():
        if len(packets[role]) != math.ceil(count / 1000):
            raise ValueError("Missing or extra source role packet")
        for number, part in enumerate(packets[role]):
            name = f"audio-v13-features-{role}-{number:02d}.jsonl.gz"
            path = folder / name
            if (part["name"] != name or part["records"] != min(1000, count - number * 1000)
                    or digest(path) != part["sha256"] or path.stat().st_size != part["bytes"]):
                raise ValueError("Changed complete feature transfer")
            seen = 0
            with gzip.open(path, "rt") as stream:
                if json.loads(next(stream)) != {"schema": "chipmunk.wavlm-public-feature-part/13", "role": role, "part": number}:
                    raise ValueError("Changed role packet header")
                for line in stream:
                    value = json.loads(line)
                    index = value["index"]
                    if type(index) is not int or not 0 <= index < 4248 or index in packed or value["record"]["role"] != role:
                        raise ValueError("Complete unique global source indices required")
                    packed[index] = value["record"]
                    seen += 1
            if seen != part["records"]:
                raise ValueError("Feature packet denominator differs")
    if set(packed) != set(range(4248)):
        raise ValueError("Every original feature record required")
    header = body["feature_header"]
    if header != {"schema": "chipmunk.audio-wavlm-features/13", "lock_sha256": LINEAGE["lock_sha256"], "planned": 4248}:
        raise ValueError("Original feature header differs")
    checksum = hashlib.sha256((json.dumps(header, allow_nan=False) + "\n").encode())
    rows = []
    for index, source in enumerate(expected):
        record = codec.unpack(packed.pop(index))
        checksum.update((json.dumps(record, allow_nan=False) + "\n").encode())
        if any(record[k] != source[k] for k in ("id", "role", "sha256", "pcm_sha256", "status")):
            raise ValueError("Source/status/role changed in transported features")
        if record["status"] == "not_applicable":
            if any(record[k] is not None for k in ("wavlm", "wavlm_sha256", "feature_frames", "valid_feature_frames", "input_pcm_sha256")):
                raise ValueError("Unassessed input acquired candidate features")
        elif record["status"] == "ok":
            vector = np.asarray(record["wavlm"], np.float32)
            if (vector.shape != (4608,) or not np.isfinite(vector).all()
                    or hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest() != record["wavlm_sha256"]
                    or record["feature_frames"] != 199 or record["valid_feature_frames"] != 199
                    or record["input_pcm_sha256"] != source["pcm_sha256"]):
                raise ValueError("Exact original clock, PCM and vector bytes required")
            record["wavlm"] = vector
        else:
            raise ValueError("Failed source cannot disappear into the review")
        rows.append({**source, **record})
    if checksum.hexdigest() != LINEAGE["receipt_sha256"]:
        raise ValueError("Original full 413 MB feature receipt did not reproduce byte-for-byte")
    return rows


def comparable(value):
    # Public summaries change only the paired-interval explanatory sentence.
    result = {k: v for k, v in value.items() if k != "schema"}
    result["paired_differences"] = {name: {k: v for k, v in panel.items() if k != "interpretation"}
                                    for name, panel in value["paired_differences"].items()}
    return result


def review(capsule, folder, refit):
    body = load(capsule, CAPSULE_SHA)
    if body["schema"] != "chipmunk.wavlm-public-development/13" or body["original_lineage"] != LINEAGE or body["deployment_eligible"] is not False:
        raise ValueError("Exact original rejected result and non-admission required")
    for name, expected in body["references"].items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError("Original public reference changed: " + name)
    freeze = load(folder / "audio-v13-wavlm-freeze.json", FREEZE_SHA)
    for name, expected in freeze["lock"]["implementation_sha256"].items():
        path = PROJECT / name
        if not path.resolve().is_relative_to(PROJECT) or digest(path) != expected:
            raise ValueError("Original declared implementation changed: " + name)
    for field, key in (("fit_record", "fit_sha256"), ("head", "head_sha256"),
                       ("native_audit", "native_sha256"), ("numerical_audit", "numerical_sha256")):
        exact_object(body[field], LINEAGE[key])
    head = body["head"]
    if head["fitted"] != body["fit_record"]["fitted"] or head["fit_sha256"] != LINEAGE["fit_sha256"]:
        raise ValueError("Original fitting-before-calibration objects differ")
    source, expected, baseline = original_inputs(folder)
    if source["data_terms"] != body["data_terms"]:
        raise ValueError("Per-source terms must remain attached")
    rows = read_features(body, folder, expected)
    outcomes = codec.unpack(body["outcomes"])
    development = [r for r in rows if r["role"] == "development"]
    if len(outcomes) != 1449 or any(any(r[k] != s[k] for k in META) for r, s in zip(outcomes, development, strict=True)):
        raise ValueError("Complete source-bound 1,449 development outcomes required")
    commitments = [(r["id"], hashlib.sha256(json.dumps(r["probe"], allow_nan=False).encode()).hexdigest()) for r in outcomes]
    if hashlib.sha256(json.dumps(commitments, separators=(",", ":")).encode()).hexdigest() != PROBES_SHA:
        raise ValueError("Original filtered feature/response commitments changed")
    for row in outcomes:
        if row["probe"]["status"] == "computed":
            row["probe"]["wavlm"] = np.asarray(row["probe"]["wavlm"], np.float32)
    fitter = frozen_fitter(freeze)
    fitting = [r for r in rows if r["role"] == "fit"]
    calibration = [r for r in rows if r["role"] == "cal"]
    selected = [r for r in fitting if r["raw"] is not None]
    weight = balanced_weights(selected)
    matrix = np.asarray([r["wavlm"] for r in selected], np.float64)
    center = weight @ matrix
    transform = {"center": center.tolist(), "scale": np.maximum(np.sqrt(weight @ (matrix - center)**2), 1e-6).tolist(),
                 "variance_factor": math.sqrt(16 / 4608), "fit_sources": len({r["base_id"] for r in selected})}
    rebuilt, audit = {}, {}
    for arm in ARMS:
        fitted = head["fitted"][arm]
        if fitted["method"] != freeze["lock"]["method"] or fitted["arm"] != arm or not same(transform, fitted["transform"]):
            raise ValueError("Fitting-only normalization or fixed method differs")
        if refit and fitter["fit"](fitting, arm) != fitted:
            raise ValueError("Exact fitting-only object refit differs for " + arm)
        guards = {}
        for label in (0, 1):
            pool = []
            for group in sorted({r["group"] for r in calibration if r["label"] == label}):
                values = [probability(r, fitted) for r in calibration if r["label"] == label and r["group"] == group]
                maximum = None if None in values else max(v if label == 0 else 1 - v for v in values)
                pool.append({"group": group, "maximum": maximum, "rows": len(values)})
            guards[str(label)] = pool
        if not same(guards, head["guards"][arm]):
            raise ValueError("Own group/truth calibration differs")
        flat = []
        for record, original in zip(outcomes, development, strict=True):
            if record["status"] != original["status"] or any(record[k] != baseline[original["id"]][k] for k in ("original_p", "original_decision")):
                raise ValueError("Original applicability or service control differs")
            p = probability(original, fitted)
            probe = record["probe"]
            required = any(initial(probability(original, head["fitted"][a]), head["guards"][a]) == "original" for a in ARMS)
            if probe["status"] != ("computed" if required else "not_required"):
                raise ValueError("Missing or unexpected required stability probe")
            band = probability(probe, fitted) if required else None
            detail = final(p, band, original["raw"], probe["raw"], guards)
            actual = record["candidates"][arm]
            if not same(actual["p"], p) or actual["decision"] != detail["decision"] or not same(actual["detail"], detail):
                raise ValueError("Probability, independent guard or dual stability differs")
            flat.append({**record, "candidate_p": p, "candidate_decision": detail["decision"]})
        rebuilt[arm] = summarize(flat, 161)
        if not same(comparable(rebuilt[arm]), comparable(body["summaries"][arm])):
            raise ValueError("Complete statistics/strata/unchanged gates differ")
        audit[arm] = {"all_normalization_calibration_outcomes_statistics_rebuilt": True,
                      "refit_performed": refit, "exact_refit_matched": True if refit else None,
                      "failed_gates": [k for k, v in rebuilt[arm]["gates"].items() if not v]}
    native = body["native_audit"]
    plan = freeze["lock"]["native_ids"]
    probes = {r["id"] for r in outcomes if r["probe"]["status"] == "computed"}
    if (native["passed"] is not True or len(probes) != 511 or native["unique_cases"] != 593
            or [r["id"] for r in native["cases"]] != sorted(set(plan) | probes)
            or any(r["passed"] is not True for r in native["cases"])):
        raise ValueError("Complete native audit metadata differs")
    loaded = sorted({name.split(".")[0] for name in sys.modules} & BLOCKED)
    if loaded:
        raise ValueError("Unexpected application/neural import: " + ", ".join(loaded))
    return {"schema": "chipmunk.portable-audio-public-review/13", "passed": True,
        "capsule_sha256": CAPSULE_SHA, "reviewer_sha256": digest(Path(__file__)),
        "inputs": len(rows), "outcomes": len(outcomes),
        "original_feature_receipt_reproduced_byte_for_byte": True, "arms": audit,
        "selected_arm_for_possible_confirmation": fitter["select"](rebuilt), "native_metadata_cases": 593,
        "computed_probes": 511, "native_reexecution_performed": False, "serving_admitted": False,
        "blocked_neural_or_app_imports_loaded": loaded,
        "runtime": {"python": platform.python_version(), "platform": platform.system(),
                    "architecture": platform.machine(),
                    **{n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn", "joblib", "narwhals", "threadpoolctl", "cloudpickle")}},
        "scope": "Public v13 receipt/outcome/statistics replay and optional shared declared exact fitter. Earlier experiments are checksum-bound inputs. No source recordings, checkpoint downloads, model inference or outside human participation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--refit", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path; existing reviews are never overwritten")
    result = review(args.capsule, args.references or args.capsule.parent, args.refit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "inputs": result["inputs"], "outcomes": result["outcomes"],
                      "exact_refits": args.refit, "neural_app_imports_loaded": result["blocked_neural_or_app_imports_loaded"]}), flush=True)
