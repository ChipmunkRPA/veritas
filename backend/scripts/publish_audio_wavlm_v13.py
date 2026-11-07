"""Lossless public features and full rejected-result replay, without source media."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import audio_wavlm_codec_v13 as codec
from scripts import publish_audio_layer_results_v12 as previous
from scripts import review_audio_wavlm_v13 as independent
from scripts import run_audio_wavlm_v13 as execution
from scripts.audio_multisource_v8 import balanced_weights
from scripts.audio_source_identity_v9 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.publish_audio_conditional_results_v10 import exact_artifact
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_guard_v7 import close
from scripts.run_audio_guard_v7 import load_rows

original = previous.original
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}
REFERENCES = {**previous.REFERENCES,
              "audio-v13-wavlm-freeze.json": "34096f6cd001c27e217ba7ff8ee5f2ff6bae48d85c4d91b9424cb78e2ef3b4a1",
              "audio-v13-thread-ownership-supplement.json": "54daf46d8fb2ccb2320fe87f3c491f17950b57a7ae3f4d4327cbc23bf179337a"}
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
FIELDS = {"schema", "references", "original_lineage", "feature_packets", "feature_header", "fit_record", "head",
          "outcomes", "summaries", "native_audit", "numerical_audit", "data_terms", "deployment_eligible", "scope"}
PROBE_COMMITMENTS_SHA256 = "eeaa37622fae390f708490f1e8fdee16dba8e7fce92ef66db4eb595ac9476fc3"


def inputs(folder):
    if any(digest(folder / n) != h for n, h in REFERENCES.items()):
        raise ValueError("Exact original public source/vector/prospective references required")
    return previous.inputs(folder)


def export_features(root, destination):
    path = root / "features.jsonl"
    if digest(path) != LINEAGE["receipt_sha256"]:
        raise ValueError("Changed original complete feature receipt")
    buckets = {role: [] for role in COUNTS}
    with path.open() as stream:
        header = json.loads(next(stream))
        for index, line in enumerate(stream):
            row = json.loads(line)
            packed = codec.pack(row)
            if json.dumps(codec.unpack(packed), allow_nan=False) + "\n" != line:
                raise ValueError("Feature transport did not restore original JSON bytes")
            original.privacy_check(packed)
            buckets[row["role"]].append({"index": index, "record": packed})
    parts = {}
    for role, records in buckets.items():
        if len(records) != COUNTS[role]:
            raise ValueError("Every original role input must remain")
        parts[role] = []
        for part, start in enumerate(range(0, len(records), 1000)):
            selected = records[start:start + 1000]
            name = f"audio-v13-features-{role}-{part:02d}.jsonl.gz"
            meta = {"schema": "chipmunk.wavlm-public-feature-part/13", "role": role, "part": part}
            with (destination / name).open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
                for value in [meta, *selected]:
                    out.write((json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode())
            parts[role].append({"name": name, "sha256": digest(destination / name),
                                "bytes": (destination / name).stat().st_size, "records": len(selected)})
            print(json.dumps({"packed_role": role, "part": part, "records": len(selected)}), flush=True)
    return header, parts


def read_features(body, packet_folder, expected):
    header, packets = body["feature_header"], body["feature_packets"]
    if (header != {"schema": "chipmunk.audio-wavlm-features/13", "lock_sha256": LINEAGE["lock_sha256"], "planned": 4248}
            or set(packets) != set(COUNTS)):
        raise ValueError("Original complete feature header and three source roles required")
    values = {}
    for role, count in COUNTS.items():
        if len(packets[role]) != math.ceil(count / 1000):
            raise ValueError("Missing or extra role packet")
        for index, part in enumerate(packets[role]):
            name = f"audio-v13-features-{role}-{index:02d}.jsonl.gz"
            path = packet_folder / name
            if (part["name"] != name or part["records"] != min(1000, count - index * 1000)
                    or digest(path) != part["sha256"] or path.stat().st_size != part["bytes"]):
                raise ValueError("Changed role packet transfer or denominator")
            with gzip.open(path, "rt") as stream:
                if json.loads(next(stream)) != {"schema": "chipmunk.wavlm-public-feature-part/13", "role": role, "part": index}:
                    raise ValueError("Changed role packet header")
                n = 0
                for line in stream:
                    packed = json.loads(line)
                    original.privacy_check(packed)
                    position = packed["index"]
                    row = codec.unpack(packed["record"])
                    if (type(position) is not int or not 0 <= position < 4248 or position in values
                            or row["role"] != role):
                        raise ValueError("Source role/index changed during feature transport")
                    values[position] = row
                    n += 1
                if n != part["records"]:
                    raise ValueError("Incomplete feature packet")
    if set(values) != set(range(4248)):
        raise ValueError("Every original global feature index required")
    rows = [values[i] for i in range(4248)]
    if original.jsonl_digest(header, rows) != LINEAGE["receipt_sha256"]:
        raise ValueError("Public parts do not restore the original receipt byte-for-byte")
    result = []
    for row, old in zip(rows, expected, strict=True):
        execution.validate_feature(row, old)
        result.append({**old, **row})
    return result


def reconstruct(head, rows, baseline, outcomes, refit):
    fitting = [r for r in rows if r["role"] == "fit"]
    calibration = [r for r in rows if r["role"] == "cal"]
    development = [r for r in rows if r["role"] == "development"]
    if len(outcomes) != 1449 or any(any(r[k] != s[k] for k in execution.prior.META)
                                  for r, s in zip(outcomes, development, strict=True)):
        raise ValueError("Every original source/truth/group development outcome required")
    fit = [r for r in fitting if r["raw"] is not None]
    w = balanced_weights(fit)
    matrix = np.asarray([r["wavlm"] for r in fit], np.float64)
    center = w @ matrix
    transform = {"center": center.tolist(), "scale": np.maximum(np.sqrt(w @ (matrix-center)**2), 1e-6).tolist(),
                 "variance_factor": math.sqrt(16 / matrix.shape[1]), "fit_sources": len({r["base_id"] for r in fit})}
    summaries, audit = {}, {}
    for arm in execution.method.ARMS:
        fitted = head["fitted"][arm]
        if fitted["method"] != execution.method.METHOD or fitted["arm"] != arm or not same_tree(transform, fitted["transform"]):
            raise ValueError("Fitting-only normalization or method changed")
        if refit and execution.method.fit(fitting, arm) != fitted:
            raise ValueError("Complete original fitted object did not reproduce exactly")
        guards = {}
        for label in (0, 1):
            pools = []
            for group in sorted({r["group"] for r in calibration if r["label"] == label}):
                p = [independent.probability(r, fitted) for r in calibration if r["label"] == label and r["group"] == group]
                maximum = None if None in p else max(v if label == 0 else 1-v for v in p)
                pools.append({"group": group, "maximum": maximum, "rows": len(p)})
            guards[str(label)] = pools
        if not same_tree(guards, head["guards"][arm]):
            raise ValueError("Independent group/truth calibration differs")
        flat = []
        for row, source in zip(outcomes, development, strict=True):
            old = baseline[source["id"]]
            if row["status"] != source["status"] or any(row[k] != old[k] for k in ("original_p", "original_decision")):
                raise ValueError("Original assessment or actual service baseline changed")
            p = independent.probability(source, fitted)
            needed = any(execution.initial(independent.probability(source, head["fitted"][a]), head["guards"][a]) == "original"
                         for a in execution.method.ARMS)
            probe = row["probe"]
            if probe["status"] != ("computed" if needed else "not_required"):
                raise ValueError("Missing or unexpected stability probe")
            if needed:
                execution.features.validate({**probe, "status": "ok"}, {"status": "ok", "pcm_sha256": probe["input_pcm_sha256"]})
            band = independent.probability(probe, fitted) if needed else None
            detail = independent.decisions(p, band, source["raw"], probe["raw"], guards)
            actual = row["candidates"][arm]
            close(actual["p"], p)
            if actual["decision"] != detail["decision"] or not same_tree(actual["detail"], detail):
                raise ValueError("Independent probability, guard or dual stability differs")
            flat.append({**row, "candidate_p": p, "candidate_decision": detail["decision"]})
        summaries[arm] = execution.summarize(flat, 161)
        audit[arm] = {"normalization_calibration_prediction_stability_rebuilt": True,
                      "refit_performed": refit, "exact_refit_matched": True if refit else None}
    return summaries, audit


def review(body, folder, packet_folder, refit=False):
    original.privacy_check(body)
    if (set(body) != FIELDS or body["schema"] != "chipmunk.wavlm-public-development/13"
            or body["references"] != REFERENCES or body["original_lineage"] != LINEAGE or body["deployment_eligible"] is not False):
        raise ValueError("Exact public result fields, lineage and non-admission required")
    for field, key in (("fit_record", "fit_sha256"), ("head", "head_sha256"),
                       ("native_audit", "native_sha256"), ("numerical_audit", "numerical_sha256")):
        exact_artifact(body[field], LINEAGE[key])
    if body["head"]["fitted"] != body["fit_record"]["fitted"] or body["head"]["fit_sha256"] != LINEAGE["fit_sha256"]:
        raise ValueError("Original fitting-before-calibration artifact changed")
    source, prior, expected = inputs(folder)
    outcomes = codec.unpack(body["outcomes"])
    development = [r for r in expected if r["role"] == "development"]
    if len(outcomes) != 1449 or any(any(r[k] != s[k] for k in execution.prior.META)
                                  for r, s in zip(outcomes, development, strict=True)):
        raise ValueError("Complete original source/truth/role outcomes required")
    commitments = [(r["id"], hashlib.sha256(json.dumps(r["probe"], allow_nan=False).encode()).hexdigest()) for r in outcomes]
    if hashlib.sha256(json.dumps(commitments, separators=(",", ":")).encode()).hexdigest() != PROBE_COMMITMENTS_SHA256:
        raise ValueError("Original complete probe commitments changed")
    if any(r["status"] == "not_applicable" and any(c["p"] is not None or c["decision"] != "inconclusive"
            for c in r["candidates"].values()) for r in outcomes):
        raise ValueError("Unassessed inputs cannot acquire a directional conclusion")
    rows = read_features(body, packet_folder, expected)
    summaries, audit = reconstruct(body["head"], rows, {r["id"]: r for r in prior["outcomes"]}, outcomes, refit)
    if not same_tree({a: original.comparable(s) for a, s in summaries.items()},
                     {a: original.comparable(s) for a, s in body["summaries"].items()}):
        raise ValueError("Complete operational statistics or unchanged gates differ")
    native, numerical = body["native_audit"], body["numerical_audit"]
    plan = original.load(folder / "audio-v13-wavlm-freeze.json")["lock"]["native_ids"]
    probes = {r["id"] for r in outcomes if r["probe"]["status"] == "computed"}
    if (native["passed"] is not True or numerical["passed"] is not True or len(probes) != 511
            or native["predetermined_cells"] != 99 or native["computed_probes"] != 511 or native["unique_cases"] != 593
            or [r["id"] for r in native["cases"]] != sorted(set(plan) | probes)
            or any(r["passed"] is not True for r in native["cases"])):
        raise ValueError("Every original native cell and computed probe must remain")
    if body["data_terms"] != source["data_terms"]:
        raise ValueError("Original per-source data terms required")
    return {"schema": "chipmunk.wavlm-public-review/13", "passed": True, "inputs": len(rows),
            "outcomes": len(outcomes), "computed_probes": 511, "native_cases": 593, "arms": audit,
            "selected_arm_for_possible_confirmation": execution.method.select(summaries),
            "scope": "Public-only exact feature receipt/numerical/refit replay; shared declared statistics/bootstrap/gates. No pretrained model, source media, private manifest, new inference or outside human review."}


def export(root, folder, destination):
    source, _, expected = inputs(folder)
    files = {"lock_sha256": "lock.json", "receipt_sha256": "features.jsonl", "fit_sha256": "trained/fit.json",
             "head_sha256": "trained/head.json", "responses_sha256": "responses-callable.jsonl",
             "summary_sha256": "summary.json", "native_sha256": "native-review.json", "numerical_sha256": "numerical-review.json"}
    if any(digest(root / p) != LINEAGE[k] for k, p in files.items()):
        raise ValueError("Original complete private results changed")
    destination.mkdir(parents=True, exist_ok=False)
    header, parts = export_features(root, destination)
    _, responses = load_rows(root / "responses-callable.jsonl")
    development = [r for r in expected if r["role"] == "development"]
    if [r["id"] for r in responses] != [r["id"] for r in development]:
        raise ValueError("Complete original development order required")
    outcomes = [{**{k: old[k] for k in execution.prior.META},
                 **original.pick(r, {"status", "original_p", "original_decision", "candidates", "probe"})}
                for old, r in zip(development, responses, strict=True)]
    mapping = [{**r, "background_sha256": s["background_sha256"]} for r, s in zip(responses, development, strict=True)]
    summary = original.load(root / "summary.json")
    body = {"schema": "chipmunk.wavlm-public-development/13", "references": REFERENCES, "original_lineage": LINEAGE,
            "feature_packets": parts, "feature_header": header, "fit_record": original.load(root / "trained/fit.json"),
            "head": original.load(root / "trained/head.json"), "outcomes": codec.pack(outcomes),
            "summaries": {a: original.public_summary(s, mapping) for a, s in summary["arms"].items()},
            "native_audit": original.load(root / "native-review.json"), "numerical_audit": original.load(root / "numerical-review.json"),
            "data_terms": source["data_terms"], "deployment_eligible": False,
            "scope": "Complete rejected consumed-source result. Lossless float32 transport restores all original feature bytes; float64 fitted parameters untouched. Metadata use the public hashed catalogue. No source media, transcript, customer file, provider ID, private path or full pretrained weights. No reserved confirmation, serving change or outside human review."}
    result = review(body, folder, destination)
    path = destination / "audio-v13-development-review.json.gz"
    path.write_bytes(gzip.compress((json.dumps(body, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0))
    print(json.dumps({"passed": result["passed"], "result_bytes": path.stat().st_size,
                      "result_sha256": digest(path), "feature_packets": parts}), flush=True)


def tamper(body, folder, packets):
    review(body, folder, packets)
    first = next(i for i, row in enumerate(body["outcomes"]) if row["status"] == "ok")
    missing = next(i for i, row in enumerate(body["outcomes"]) if row["status"] == "not_applicable")
    mutations = {
        "fitted coefficient": lambda b: b["head"]["fitted"]["standalone"]["coefficients"].__setitem__(0, 99.),
        "calibration guard": lambda b: b["head"]["guards"]["complement"]["0"][0].__setitem__("maximum", .99),
        "mapped probability": lambda b: b["outcomes"][first]["candidates"]["complement"].__setitem__("p", .123),
        "unassessed called original": lambda b: b["outcomes"][missing]["candidates"]["complement"].__setitem__("decision", "original"),
        "source group": lambda b: b["outcomes"][first].__setitem__("group", "0"*64),
        "missing outcome": lambda b: b["outcomes"].pop(),
        "missing feature role": lambda b: b["feature_packets"].pop("cal"),
        "feature transfer": lambda b: b["feature_packets"]["fit"][0].__setitem__("sha256", "0"*64),
        "missing native case": lambda b: b["native_audit"]["cases"].pop(),
        "scope admission": lambda b: b.__setitem__("deployment_eligible", True),
    }
    def coherent_probe(b):
        row = next(r for r in b["outcomes"] if r["probe"]["status"] == "computed")
        probe = codec.unpack(row["probe"])
        probe["wavlm"][0] = float(np.float32(probe["wavlm"][0] + .125))
        probe = codec.pack(probe)
        probe["wavlm_sha256"] = probe["wavlm"]["sha256"]
        row["probe"] = probe
    mutations["coherently changed probe feature"] = coherent_probe
    cases = []
    for name, mutate in mutations.items():
        altered = copy.deepcopy(body)
        mutate(altered)
        try:
            review(altered, folder, packets)
        except (ValueError, KeyError, TypeError) as error:
            cases.append({"alteration": name, "rejected": True, "reason": str(error)})
        else:
            raise ValueError("Altered actual public result accepted: " + name)
    # An updated transfer checksum cannot authorize different original features.
    with TemporaryDirectory() as directory:
        target = Path(directory)
        for parts in body["feature_packets"].values():
            for part in parts:
                (target / part["name"]).symlink_to((packets / part["name"]).resolve())
        altered = copy.deepcopy(body)
        part = altered["feature_packets"]["fit"][0]
        path = target / part["name"]
        with gzip.open(path, "rt") as stream:
            lines = [json.loads(line) for line in stream]
        row = next(r for r in lines[1:] if r["record"]["status"] == "ok")
        value = codec.unpack(row["record"])
        value["wavlm"][0] = float(np.float32(value["wavlm"][0] + .125))
        row["record"] = codec.pack(value)
        row["record"]["wavlm_sha256"] = row["record"]["wavlm"]["sha256"]
        path.unlink()  # Remove the temporary symlink, never the original packet.
        path.write_bytes(gzip.compress(("\n".join(json.dumps(v, separators=(",", ":")) for v in lines) + "\n").encode(), mtime=0))
        part["sha256"], part["bytes"] = digest(path), path.stat().st_size
        try:
            review(altered, folder, target)
        except ValueError as error:
            cases.append({"alteration": "coherent feature and transfer digests", "rejected": True, "reason": str(error)})
        else:
            raise ValueError("Coherently changed original feature packet accepted")
    return {"schema": "chipmunk.wavlm-public-tamper/13", "passed": True, "unmodified_packet_passed": True,
            "cases": cases, "scope": "Actual packet alterations, no new inference or outside review"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("export", "review", "tamper"))
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--references", type=Path, default=Path("docs/evaluation-panels"))
    p.add_argument("--packets", type=Path)
    p.add_argument("--refit", action="store_true")
    a = p.parse_args()
    if a.stage == "export":
        export(a.input, a.references, a.output)
    else:
        body = original.load(a.input)
        result = (review(body, a.references, a.packets or a.input.parent, a.refit) if a.stage == "review"
                  else tamper(body, a.references, a.packets or a.input.parent))
        write(a.output, result)
        print(json.dumps(result if a.stage == "review" else {"passed": True, "alterations_rejected": len(result["cases"])}), flush=True)
