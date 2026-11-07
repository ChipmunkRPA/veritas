"""Lossless public layer vectors and complete rejected-result reproduction.

Three role-separated vector packets reference the existing public final vectors.
They reconstruct the original full layer receipt byte-for-byte. Outcome metadata
uses the published hashed source catalogue; no private background/provider IDs.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from scripts import audio_layer_public_codec_v12 as codec
from scripts import export_audio_layer_v12 as prospective
from scripts import publish_audio_full_results_v11 as previous
from scripts import publish_audio_identity_results_v9 as original
from scripts import run_audio_layer_v12 as execution
from scripts.audio_source_identity_v9 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.publish_audio_conditional_results_v10 import exact_artifact
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.review_audio_layer_v12 import reconstruct
from scripts.run_audio_guard_v7 import load_rows

REFERENCES = {**previous.REFERENCES,
              "audio-v11-development-review.json.gz": "5b2c2d06e785d60ed754adc175b77a8de57161a67165df164a497f0b3876ba30",
              "audio-v12-layer-freeze.json": "d6c76e7b9a67c661c88da3193831e2d787ce5431bfd78499c52904933f75a0dc"}
LINEAGE = {
    "lock_sha256": "2b0fe2f3f60f81f972b12eb756080b4272a0b72afddf106835864b27a5d9f179",
    "receipt_sha256": "c9298f6ce1e56f3b60b8f3b082853b7e84dd83e72dd38a61e58e40174bf1894e",
    "fit_sha256": "7ad7b09d8343f774f07540fd268b293f1217fe3752e26a25d69b7d7a02da5697",
    "head_sha256": "273bd3144dbdcad8e35658adbf8a2c68ff56037be8f3816fa60f9829f4d25c3a",
    "responses_sha256": "d2293c6157ec78159ca1b07ad6242e7c5271ea593bcf0978d47080c761c0b251",
    "summary_sha256": "6929fbc593c9e589444c9a331d979e01d6037f3f9765f44466c644c7cf4be407",
    "native_sha256": "d83c42009786f0b406af16acbcae974a0dc77c1f7cb249eeb509a6b59db80507",
    "numerical_sha256": "80bbc6c56f58545b541598412572fd67d30cb640d1f2c940d5607adf1fcdde3a",
}
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}
PROBE_COMMITMENTS_SHA256 = "ba4758947bd29d6873902f0d22d1d18b9e58b211414202c18e0fa4a38a9e6333"
FIELDS = {"schema", "references", "original_lineage", "feature_packets", "fit_record", "head",
          "response_header", "outcomes", "summaries", "native_audit", "numerical_audit",
          "deployment_eligible", "selected_arm_for_possible_confirmation", "data_terms", "scope"}


def inputs(folder):
    if any(digest(folder / n) != h for n, h in REFERENCES.items()):
        raise ValueError("Exact original public source/vector/freeze references required")
    source, prior, rows, _ = previous.inputs(folder)
    prospective.review(original.load(folder / "audio-v12-layer-freeze.json"), source,
                       original.load(folder / "audio-v9-execution-freeze.json"))
    return source, prior, rows


def features_export(root, folder, destination, prior):
    previous_vectors = {r["id"]: r for r in prior["features"]}
    source = root / "representations.jsonl"
    if digest(source) != LINEAGE["receipt_sha256"]:
        raise ValueError("Changed complete original layer receipt")
    names = {role: f"audio-v12-layer-{role}.jsonl.gz" for role in COUNTS}
    with source.open() as stream, ExitStack() as stack:
        header = json.loads(next(stream))
        writers = {}
        for role, name in names.items():
            raw = stack.enter_context((destination / name).open("xb"))
            out = stack.enter_context(gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0))
            meta = {"schema": "chipmunk.audio-layer-public-features/12", "role": role,
                    "records": COUNTS[role], "original_receipt_sha256": LINEAGE["receipt_sha256"],
                    "prior_features_sha256": REFERENCES["audio-v9-development-review.json.gz"],
                    "representation_header": header}
            original.privacy_check(meta)
            out.write((json.dumps(meta, separators=(",", ":")) + "\n").encode())
            writers[role] = out
        count = Counter()
        for index, line in enumerate(stream):
            record = json.loads(line)
            packed = codec.pack(record, previous_vectors)
            # Includes original key order, runtime scalar and signed float32 values.
            if json.dumps(codec.unpack(packed, previous_vectors), allow_nan=False) + "\n" != line:
                raise ValueError("Packed vectors did not restore original feature receipt bytes")
            original.privacy_check(packed)
            writers[record["role"]].write((json.dumps({"index": index, "record": packed}, separators=(",", ":")) + "\n").encode())
            count[record["role"]] += 1
            if (index + 1) % 1000 == 0:
                print(json.dumps({"packed_features": index + 1, "planned": 4248}), flush=True)
        if dict(count) != COUNTS:
            raise ValueError("Every fitting/calibration/development record must remain")
    return {role: {"name": name, "sha256": digest(destination / name),
                   "bytes": (destination / name).stat().st_size, "records": COUNTS[role]}
            for role, name in names.items()}


def features_read(packets, packet_folder, prior, expected_rows):
    if set(packets) != set(COUNTS):
        raise ValueError("All three original source roles required")
    previous_vectors = {r["id"]: r for r in prior["features"]}
    records, original_header = {}, None
    for role in COUNTS:
        part = packets[role]
        if part["name"] != f"audio-v12-layer-{role}.jsonl.gz" or part["records"] != COUNTS[role]:
            raise ValueError("Changed role packet identity or denominator")
        path = packet_folder / part["name"]
        if digest(path) != part["sha256"] or path.stat().st_size != part["bytes"]:
            raise ValueError("Changed public layer packet transfer")
        with gzip.open(path, "rt") as stream:
            header = json.loads(next(stream))
            if (header["schema"] != "chipmunk.audio-layer-public-features/12" or header["role"] != role
                    or header["records"] != COUNTS[role] or header["original_receipt_sha256"] != LINEAGE["receipt_sha256"]
                    or header["prior_features_sha256"] != REFERENCES["audio-v9-development-review.json.gz"]):
                raise ValueError("Changed layer packet source/header")
            if original_header is None:
                original_header = header["representation_header"]
            elif original_header != header["representation_header"]:
                raise ValueError("Original representation headers disagree")
            count = 0
            for line in stream:
                original.privacy_check(json.loads(line))
                row = json.loads(line)
                index = row["index"]
                if type(index) is not int or not 0 <= index < 4248 or index in records:
                    raise ValueError("Changed global layer record order")
                record = codec.unpack(row["record"], previous_vectors)
                if record["role"] != role:
                    raise ValueError("Feature role changed during transport")
                records[index] = record
                count += 1
            if count != COUNTS[role]:
                raise ValueError("Incomplete public role packet")
    if set(records) != set(range(4248)):
        raise ValueError("Every global original feature index is required")
    ordered = [records[i] for i in range(4248)]
    if original.jsonl_digest(original_header, ordered) != LINEAGE["receipt_sha256"]:
        raise ValueError("Layer packets do not restore the original full receipt byte-for-byte")
    rows = []
    for record, old in zip(ordered, expected_rows, strict=True):
        execution.validate_original(record, old, prior["representation_header"])
        rows.append({**old, **record})
    return rows


def native_review(native, rows, outcomes, folder):
    exact_artifact(native, LINEAGE["native_sha256"])
    plan = original.load(folder / "audio-v9-native-development-plan.json")
    probes = {r["id"] for r in outcomes if r["probe"]["status"] == "computed"}
    identifiers = sorted(set(plan["ids"]) | probes)
    if (native["passed"] is not True or native["predetermined_cells"] != 99
            or native["new_probe_cases"] != 422 or native["unique_cases"] != 508
            or [r["id"] for r in native["cases"]] != identifiers
            or any(r["passed"] is not True or r["original_passed"] is not True
                   or r["mapped_original_passed"] is not True or r["probe_passed"] is False for r in native["cases"])):
        raise ValueError("All original native cells and every computed layer probe required")
    by_id = {r["id"]: r for r in rows}
    for case in native["cases"]:
        if (case["sha256"] != by_id[case["id"]]["sha256"] or case["pcm_sha256"] != by_id[case["id"]]["pcm_sha256"]
                or case["predetermined_cell"] != (case["id"] in plan["ids"])
                or case["new_probe_case"] != (case["id"] in probes)):
            raise ValueError("Native source or case-role binding differs")


def review(body, folder, packet_folder, *, refit=False):
    original.privacy_check(body)
    if (set(body) != FIELDS or body["schema"] != "chipmunk.audio-layer-public-development/12"
            or body["references"] != REFERENCES or body["original_lineage"] != LINEAGE
            or body["deployment_eligible"] is not False):
        raise ValueError("Exact public layer-result allowlist and lineage required")
    head, fit_record = body["head"], body["fit_record"]
    exact_artifact(head, LINEAGE["head_sha256"])
    exact_artifact(fit_record, LINEAGE["fit_sha256"])
    if head["fitted"] != fit_record["fitted"] or head["fit_sha256"] != LINEAGE["fit_sha256"]:
        raise ValueError("Original fitting-before-calibration object differs")
    header = body["response_header"]
    if (set(header) != {"schema", "lock_sha256", "receipt_sha256", "head_sha256", "planned"}
            or any(header[k] != LINEAGE[k] for k in ("lock_sha256", "receipt_sha256", "head_sha256"))
            or header["planned"] != 1449):
        raise ValueError("Original complete response header binding differs")
    exact_artifact(body["native_audit"], LINEAGE["native_sha256"])
    exact_artifact(body["numerical_audit"], LINEAGE["numerical_sha256"])
    _, prior, expected = inputs(folder)
    outcomes = codec.unpack(body["outcomes"])
    development = [r for r in expected if r["role"] == "development"]
    if len(outcomes) != len(development) or any(
        any(r[k] != s[k] for k in execution.META) for r, s in zip(outcomes, development, strict=True)
    ):
        raise ValueError("Complete public outcome source/truth/role binding differs")
    commitments = [(r["id"], hashlib.sha256(json.dumps(r["probe"], allow_nan=False).encode()).hexdigest())
                   for r in outcomes]
    if hashlib.sha256(json.dumps(commitments, separators=(",", ":")).encode()).hexdigest() != PROBE_COMMITMENTS_SHA256:
        raise ValueError("Original complete neural probe commitments changed")
    for row in outcomes:
        if row["probe"]["status"] == "computed":
            probe = {**row["probe"], "status": "ok"}
            execution.validate_layers(probe)
            execution.prior.original.validate_feature(
                {**probe, "windows": [{"start_sample": 0, "end_sample": 64000}]}, prior["representation_header"])
    rows = features_read(body["feature_packets"], packet_folder, prior, expected)
    baseline = {r["id"]: r for r in prior["outcomes"]}
    summaries, audit = reconstruct(head, rows, baseline, outcomes, refit=refit)
    if (not same_tree({a: original.comparable(s) for a, s in summaries.items()},
                      {a: original.comparable(s) for a, s in body["summaries"].items()})
            or execution.method.select(summaries) != body["selected_arm_for_possible_confirmation"]):
        raise ValueError("Complete public operational outcomes, fixed gates or selection differ")
    native_review(body["native_audit"], rows, outcomes, folder)
    exact_artifact(body["numerical_audit"], LINEAGE["numerical_sha256"])
    if not body["numerical_audit"]["passed"]:
        raise ValueError("Original numerical review failed")
    return {"schema": "chipmunk.audio-layer-public-review/12", "passed": True, "inputs": len(rows),
            "outcomes": len(outcomes), "computed_probes": 422, "native_cases": 508, "arms": audit,
            "selected_arm_for_possible_confirmation": execution.method.select(summaries),
            "scope": "Public-only lossless feature receipt reconstruction and numerical/refit review; "
                     "no pretrained model, source waveforms, private manifest, new inference or outside reviewer required."}


def tamper(body, folder, packet_folder):
    # Rejection is meaningful only after this exact unmodified packet passes.
    review(body, folder, packet_folder)
    assessed = next(i for i, r in enumerate(body["outcomes"]) if r["status"] == "ok")
    unassessed = next(i for i, r in enumerate(body["outcomes"]) if r["status"] == "not_applicable")
    computed = next(i for i, r in enumerate(body["outcomes"]) if r["probe"]["status"] == "computed")
    names = ("fitted coefficient", "calibration guard", "source group", "mapped probability",
             "unassessed called original", "probe layer order", "probe float payload", "missing role packet",
             "missing outcome", "missing native case", "response header", "private path", "layer packet transfer")
    cases = []
    for name in names:
        changed = copy.deepcopy(body)
        if name == names[0]:
            changed["head"]["fitted"]["layers"]["coefficients"][0] += .1
        elif name == names[1]:
            changed["head"]["guards"]["layers"]["0"][0]["rows"] += 1
        elif name == names[2]:
            changed["outcomes"][0]["group"] = "altered"
        elif name == names[3]:
            changed["outcomes"][assessed]["candidates"]["layers"]["p"] = None
        elif name == names[4]:
            changed["outcomes"][unassessed]["candidates"]["layers"]["decision"] = "original"
        elif name == names[5]:
            changed["outcomes"][computed]["probe"]["layer_ids"] = [16, 8, 24]
        elif name == names[6]:
            changed["outcomes"][computed]["probe"]["layer_embeddings"]["sha256"] = "a" * 64
        elif name == names[7]:
            changed["feature_packets"].pop("cal")
        elif name == names[8]:
            changed["outcomes"].pop()
        elif name == names[9]:
            changed["native_audit"]["cases"].pop()
        elif name == names[10]:
            changed["response_header"]["planned"] -= 1
        elif name == names[11]:
            changed["notes"] = "/Users/placeholder/private.wav"
        else:
            changed["feature_packets"]["cal"]["sha256"] = "a" * 64
        try:
            review(changed, folder, packet_folder)
        except ValueError as error:
            cases.append({"alteration": name, "rejected": True, "reason": str(error)})
        else:
            raise ValueError("Altered actual capsule passed: " + name)
    return {"schema": "chipmunk.audio-layer-public-tamper/12", "passed": True, "cases": cases,
            "scope": "Deliberate alterations of the actual result packet; no new neural inference or outside review."}


def export(root, folder, destination):
    source, prior, expected = inputs(folder)
    destination.mkdir(parents=True, exist_ok=False)
    for key, path in {"lock_sha256": "lock.json", "receipt_sha256": "representations.jsonl",
                      "fit_sha256": "trained/fit.json", "head_sha256": "trained/head.json",
                      "responses_sha256": "responses.jsonl", "summary_sha256": "summary.json",
                      "native_sha256": "native-review.json", "numerical_sha256": "numerical-review.json"}.items():
        if digest(root / path) != LINEAGE[key]:
            raise ValueError("Changed original private artifact: " + key)
    packets = features_export(root, folder, destination, prior)
    header, responses = load_rows(root / "responses.jsonl")
    development = [r for r in expected if r["role"] == "development"]
    if [r["id"] for r in responses] != [r["id"] for r in development]:
        raise ValueError("Complete exact source/role outcomes required")
    allowed = {"status", "error_type", "original_p", "original_decision", "candidates", "probe"}
    outcomes = [{**{k: row[k] for k in execution.META}, **original.pick(r, allowed)}
                for row, r in zip(development, responses, strict=True)]
    mapping = [{**r, "background_sha256": s["background_sha256"]}
               for r, s in zip(responses, development, strict=True)]
    summary = original.load(root / "summary.json")
    body = {"schema": "chipmunk.audio-layer-public-development/12", "references": REFERENCES,
            "original_lineage": LINEAGE, "feature_packets": packets,
            "fit_record": original.load(root / "trained/fit.json"), "head": original.load(root / "trained/head.json"),
            "response_header": header, "outcomes": codec.pack(outcomes),
            "summaries": {a: original.public_summary(s, mapping) for a, s in summary["arms"].items()},
            "native_audit": original.load(root / "native-review.json"),
            "numerical_audit": original.load(root / "numerical-review.json"),
            "deployment_eligible": False,
            "selected_arm_for_possible_confirmation": summary["selected_arm_for_possible_confirmation"],
            "data_terms": source["data_terms"],
            "scope": "Complete rejected consumed-source result; source-layer vectors are lossless float32 transport, "
                     "referencing original public final vectors. Feature packets restore the original private receipt bytes. "
                     "Outcome/summary metadata use public hashed backgrounds and differ from private lineage bytes. "
                     "No waveform, customer file, transcript, provider ID, private path or full pretrained checkpoint. "
                     "No reserved confirmation, serving change or outside human review."}
    result = review(body, folder, destination)
    path = destination / "audio-v12-development-review.json.gz"
    path.write_bytes(gzip.compress((json.dumps(body, separators=(",", ":"), allow_nan=False) + "\n").encode(), mtime=0))
    print(json.dumps({"passed": result["passed"], "result_bytes": path.stat().st_size,
                      "result_sha256": digest(path), "feature_packets": packets}), flush=True)


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
    elif a.stage == "review":
        result = review(original.load(a.input), a.references, a.packets or a.input.parent, refit=a.refit)
        write(a.output, result)
        print(json.dumps(result), flush=True)
    else:
        result = tamper(original.load(a.input), a.references, a.packets or a.input.parent)
        write(a.output, result)
        print(json.dumps({"passed": result["passed"], "alterations_rejected": len(result["cases"])}), flush=True)
