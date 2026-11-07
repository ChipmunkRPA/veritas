"""Freeze and run the consumed four-arm writing comparison; research only."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from scripts import writing_modernbert_v4 as method
from scripts.prepare_writing_focus_v3 import HISTORICAL_PASSAGES_SHA

ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "backend/scripts/writing_focus_v3.py", "backend/scripts/writing_modernbert_v4.py",
    "backend/scripts/run_writing_modernbert_v4.py", "backend/scripts/review_writing_modernbert_v4.py",
    "backend/scripts/prepare_writing_focus_v3.py", "backend/scripts/audit_writing_sources_v2.py",
    "backend/tests/test_writing_modernbert_v4.py", "docs/writing-modernbert-v4-protocol.md",
    "backend/app/detectors/text/classifier.py", "backend/app/detectors/text/coverage.py",
    "backend/app/detectors/text/pipeline.py", "backend/app/detectors/text/extract.py",
    "backend/app/detectors/text/segment.py", "backend/app/detectors/text/adapters/learned.py",
)
INSTALLED = (
    "transformers.models.deberta_v2.modeling_deberta_v2", "transformers.models.modernbert.modeling_modernbert",
    "transformers.models.modernbert.configuration_modernbert", "transformers.modeling_rope_utils",
    "transformers.masking_utils", "transformers.activations",
    "transformers.modeling_utils", "transformers.tokenization_utils_tokenizers",
    "transformers.tokenization_utils_base", "tokenizers.tokenizers", "torch.nn.functional",
)
SCHEMA = "chipmunk.writing-modernbert-lock/4"
SYNTHETIC_TEXTS = (
    "This is a synthetic runtime check. It is not a detection-quality example.",
    "Unicode runtime probe: café, e\u0301 and 🙂. Preserve every native token.",
    "A repeated synthetic overflow probe. " * 180,
)


def require_sequential_evaluation():
    """Prospective full-audio replication runs alone; never compete with it."""
    import psutil

    modules = {"scripts.run_audio_separation_v14", "scripts.repeat_audio_separation_v14",
               "scripts.run_video_demamba_v13", "scripts.repeat_video_release_v14", "scripts.repeat_video_synthesis_v13",
               "scripts.verify_video_native_v14", "scripts.run_writing_modernbert_v4", "scripts.run_writing_focus_v3"}
    filenames = {name.rsplit('.', 1)[1] + '.py' for name in modules}
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = process.info["cmdline"] or []
        matched = any(m in command for m in modules) or any(Path(arg).name in filenames for arg in command)
        if process.pid != os.getpid() and matched:
            raise ValueError("Run heavy evaluations sequentially; no candidate or baseline model loaded")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_inputs(path):
    if method.digest(path) != method.INPUT_SHA:
        raise ValueError("Exact original consumed input packet required")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    method.validate_inputs(lines[0], lines[1:])
    return lines[1:]


def runtime():
    return {"python": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "safetensors", "numpy", "scipy", "scikit-learn")}}


def implementation():
    return {name: method.digest(ROOT / name) for name in CODE}


def installed():
    return {name: method.digest(Path(importlib.import_module(name).__file__)) for name in INSTALLED}


def assets(path, expected):
    actual = {name: method.digest(path / name) for name in expected}
    if actual != expected:
        raise ValueError("Pinned model/tokenizer/notice assets changed")
    return actual


def validate_smoke(path):
    value = json.loads(path.read_text())
    if (value['schema'] != 'chipmunk.writing-modernbert-synthetic-runtime/4'
            or value['candidate_assets'] != method.CANDIDATE_ASSETS
            or value['implementation_sha256'] != implementation() or value['runtime'] != runtime()
            or value['installed_sha256'] != installed()
            or value['synthetic_input_sha256'] != [method.text_sha(t) for t in SYNTHETIC_TEXTS]
            or value['strict_state_load'] is not True or value['repeat_exact'] is not True
            or value['project_text_forwards'] != 0 or value['serving_admitted'] is not False
            or len(value['responses']) != len(SYNTHETIC_TEXTS) or len(value['coverage']) != len(SYNTHETIC_TEXTS)
            or not 0 <= value['max_single_batch_difference'] <= 1e-5
            or not 0 <= value['max_direct_difference'] <= 1e-5):
        raise ValueError('Completed exact frozen synthetic runtime check required before project inference')
    for score, coverage in zip(value['responses'], value['coverage'], strict=True):
        method.finite_score(score)
        if (type(coverage['tokens']) is not int or coverage['tokens'] < 1
                or type(coverage['token_windows']) is not int or coverage['token_windows'] < 1
                or len(coverage['token_ids_sha256']) != 64):
            raise ValueError('Complete synthetic token coverage required')
    if value['coverage'][-1]['token_windows'] < 2:
        raise ValueError('Actual synthetic native overflow must be exercised')
    return value


def smoke(candidate, output):
    """Runtime only; no project text, new fitting or quality/accuracy claim."""
    require_sequential_evaluation()
    import torch

    if output.exists():
        raise ValueError('New synthetic receipt required')
    torch.set_num_threads(method.METHOD['threads'])
    assets(candidate, method.CANDIDATE_ASSETS)
    clf = load_model('candidate', {'candidate': candidate})
    identities = [token_identity(t, clf.tok) for t in SYNTHETIC_TEXTS]

    def predict(batch):
        scores = clf.predict(list(SYNTHETIC_TEXTS), max_len=512, batch=batch, check=lambda _: None)
        if (len(scores) != len(SYNTHETIC_TEXTS)
                or any(any(a[k] != b[k] for k in ('tokens', 'token_windows'))
                       for a, b in zip(clf.last_coverage, identities, strict=True))):
            raise ValueError('Synthetic complete token coverage differs')
        return [method.finite_score(v) for v in scores]

    batched, repeated, single = predict(4), predict(4), predict(1)
    direct = [direct_response(clf, 'candidate', t) for t in SYNTHETIC_TEXTS]
    body = {'schema': 'chipmunk.writing-modernbert-synthetic-runtime/4',
        'candidate_assets': method.CANDIDATE_ASSETS, 'implementation_sha256': implementation(),
        'installed_sha256': installed(), 'runtime': runtime(),
        'synthetic_input_sha256': [method.text_sha(t) for t in SYNTHETIC_TEXTS],
        'strict_state_load': True, 'repeat_exact': batched == repeated, 'responses': batched,
        'coverage': identities, 'max_single_batch_difference': max(abs(a - b) for a, b in zip(batched, single, strict=True)),
        'max_direct_difference': max(abs(a - b) for a, b in zip(batched, direct, strict=True)),
        'project_text_forwards': 0, 'serving_admitted': False,
        'scope': 'Only synthetic execution/token coverage and strict state correspondence; no project text, calibration, attribution truth or detection-quality validation.'}
    write(output, body)
    validate_smoke(output)
    print(json.dumps({'synthetic_runtime_passed': True, 'project_text_forwards': 0}), flush=True)


def freeze(inputs, historical, baseline, candidate, output, public, smoke_receipt):
    rows = read_inputs(inputs)
    synthetic = validate_smoke(smoke_receipt)
    if method.digest(historical) != HISTORICAL_PASSAGES_SHA:
        raise ValueError("Exact historical passage scores required")
    lock = {"schema": SCHEMA, "inputs": str(inputs.resolve()), "baseline": str(baseline.resolve()),
            "historical": str(historical.resolve()), "historical_sha256": HISTORICAL_PASSAGES_SHA,
            "candidate": str(candidate.resolve()), "input_sha256": method.INPUT_SHA,
            "method": method.METHOD, "implementation_sha256": implementation(), "runtime": runtime(),
            "installed_sha256": installed(),
            "baseline_assets": assets(baseline, method.BASELINE_ASSETS),
            "candidate_assets": assets(candidate, method.CANDIDATE_ASSETS),
            "baseline_revision": method.BASELINE_REVISION, "candidate_revision": method.CANDIDATE_REVISION,
            "synthetic_smoke": str(smoke_receipt.resolve()), "synthetic_smoke_sha256": method.digest(smoke_receipt),
            "synthetic_smoke_receipt": synthetic,
            "new_project_classifier_forwards": 0, "fitting_started": False, "serving_adoption_permitted": False}
    if public.exists() or output.exists():
        raise ValueError("New prospective output paths required")
    write(output, lock)
    body = {k: v for k, v in lock.items() if k not in ("inputs", "historical", "baseline", "candidate", "synthetic_smoke")}
    body.update(schema="chipmunk.writing-modernbert-public-freeze/4", private_lock_sha256=method.digest(output))
    if any(token in json.dumps(body) for token in ("/Users/", "/private/", "/home/")):
        raise ValueError("Private path cannot be published")
    write(public, body)
    print(json.dumps({"lock_sha256": method.digest(output), "documents": len(rows), "foci": 674,
                      "new_project_classifier_forwards": 0}), flush=True)


def check(lock_path):
    lock = json.loads(lock_path.read_text())
    if (lock["schema"] != SCHEMA or lock["method"] != method.METHOD
            or lock["implementation_sha256"] != implementation() or lock["runtime"] != runtime()
            or lock["installed_sha256"] != installed() or lock["input_sha256"] != method.INPUT_SHA
            or lock["baseline_assets"] != method.BASELINE_ASSETS or lock["candidate_assets"] != method.CANDIDATE_ASSETS
            or lock["baseline_revision"] != method.BASELINE_REVISION
            or lock["candidate_revision"] != method.CANDIDATE_REVISION
            or lock["historical_sha256"] != HISTORICAL_PASSAGES_SHA
            or method.digest(Path(lock["historical"])) != HISTORICAL_PASSAGES_SHA
            or lock["synthetic_smoke_sha256"] != method.digest(Path(lock['synthetic_smoke']))
            or lock['synthetic_smoke_receipt'] != validate_smoke(Path(lock['synthetic_smoke']))
            or lock["new_project_classifier_forwards"] != 0 or lock["fitting_started"] is not False
            or lock["serving_adoption_permitted"] is not False):
        raise ValueError("Frozen method/source/model/runtime/implementation changed")
    assets(Path(lock["baseline"]), method.BASELINE_ASSETS)
    assets(Path(lock["candidate"]), method.CANDIDATE_ASSETS)
    return lock, read_inputs(Path(lock["inputs"]))


def token_identity(text, tokenizer):
    """Independent no-truncation identity vs all native overflow content IDs."""
    expected = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    encoded = tokenizer(text, truncation=True, max_length=method.METHOD["max_len"], stride=0,
                        return_overflowing_tokens=True, return_special_tokens_mask=True)
    windows = [[token for token, special in zip(ids, mask, strict=True) if not special]
               for ids, mask in zip(encoded["input_ids"], encoded["special_tokens_mask"], strict=True)]
    if not expected or [token for window in windows for token in window] != expected:
        raise ValueError("Native tokenizer overflow must cover every original token exactly once")
    payload = json.dumps(expected, separators=(",", ":"))
    return {"tokens": len(expected), "token_windows": len(windows),
            "token_ids_sha256": method.text_sha(payload)}


def validate_candidate(model, tokenizer, info):
    """Exact local two-logit checkpoint; index1 follows the pinned publisher source."""
    import torch

    if (any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs", "unexpected_keys"))
            or not tokenizer.is_fast or model.config.model_type != "modernbert"
            or model.config.num_labels != 2 or model.config.classifier_pooling != "mean"
            or model.config.problem_type != "multi_label_classification"
            or model.config.id2label != {0: "LABEL_0", 1: "LABEL_1"}
            or sum(p.numel() for p in model.parameters()) != 149606402
            or any(p.is_meta or p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("Exact candidate tensors, architecture, CPUfloat32 and native tokenizer required")


def load_model(name, lock):
    require_sequential_evaluation()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from app.detectors.text.classifier import FourClassTextClassifier, _validated_logits
    from app.detectors.text.coverage import classify_complete

    if name == "baseline":
        clf = FourClassTextClassifier.from_snapshot(Path(lock[name]), "cpu")
        clf.model.requires_grad_(False)
        return clf
    path = Path(lock[name])
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        output_loading_info=True, dtype=torch.float32, attn_implementation="eager")
    validate_candidate(model, tok, info)
    model.requires_grad_(False).eval()
    clf = SimpleNamespace(tok=tok, model=model, last_coverage=None)

    def logits(encoded):
        output = model(**encoded).logits
        return _validated_logits(output, len(encoded["input_ids"]), width=2).softmax(dim=-1)[:, 1]

    @torch.inference_mode()
    def predict(texts, **kwargs):
        values, clf.last_coverage = classify_complete(texts, tok, "cpu", logits, **kwargs)
        return values

    clf.predict = predict
    return clf


def expected_header(lock_path, name):
    return {"schema": "chipmunk.writing-modernbert-responses/4", "lock_sha256": method.digest(lock_path),
            "model": name, "documents": 40, "foci": 674, "views": method.METHOD["views"]}


def validate_response(response, source):
    if (response["status"] != "ok" or any(response[k] != source[k] for k in
            ("id", "author_group", "extracted_text_sha256")) or len(response["foci"]) != len(source["foci"])):
        raise ValueError("All bound source documents, failures and foci must remain visible")
    for result, focus in zip(response["foci"], source["foci"], strict=True):
        if any(result[k] != focus[k] for k in ("sentence", "start", "end", "focus_sha256", "context_sha256")):
            raise ValueError("Response is bound to the wrong source position")
        for view in method.METHOD["views"]:
            method.finite_score(result[view]["score"])
            coverage = result[view]["coverage"]
            if (type(coverage["tokens"]) is not int or coverage["tokens"] < 1
                    or type(coverage["token_windows"]) is not int or coverage["token_windows"] < 1
                    or len(coverage["token_ids_sha256"]) != 64):
                raise ValueError("Complete native token coverage proof required")


def score(lock_path, name, output):
    import torch

    lock, rows = check(lock_path)
    header = expected_header(lock_path, name)
    if output.exists():
        existing = [json.loads(line) for line in output.read_text().splitlines()]
        if not existing or existing[0] != header or len(existing) > len(rows) + 1:
            raise ValueError("Journal does not match this exact prospective freeze")
        done = existing[1:]
        for response, source in zip(done, rows, strict=False):
            validate_response(response, source)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
        done = []
    if len(done) == len(rows):
        print(json.dumps({"model": name, "complete": 40, "new_forwards": 0}), flush=True)
        return
    torch.set_num_threads(method.METHOD["threads"])
    clf = load_model(name, lock)
    started = time.monotonic()
    with output.open("a") as stream:
        for index, source in enumerate(rows[len(done):], len(done)):
            result = {k: source[k] for k in ("id", "author_group", "extracted_text_sha256")}
            result.update(status="ok", foci=[{k: f[k] for k in
                          ("sentence", "start", "end", "focus_sha256", "context_sha256")} for f in source["foci"]])
            try:
                for view in method.METHOD["views"]:
                    texts = [f[view] for f in source["foci"]]
                    identities = [token_identity(text, clf.tok) for text in texts]
                    values = clf.predict(texts, max_len=512, batch=4, check=lambda _: None)
                    if len(values) != len(texts):
                        raise ValueError("One score per input is required")
                    for focus, value, actual, identity in zip(result["foci"], values, clf.last_coverage, identities, strict=True):
                        if any(actual[k] != identity[k] for k in ("tokens", "token_windows")):
                            raise ValueError("Scored tokens differ from independent complete-token audit")
                        focus[view] = {"score": method.finite_score(value), "coverage": identity}
                validate_response(result, source)
            except Exception as error:
                traceback.print_exc()
                result.update(status="failed", error_type=type(error).__name__)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({"model": name, "documents": index + 1, "planned": 40,
                              "status": result["status"], "elapsed_s": time.monotonic() - started}), flush=True)
            if result["status"] != "ok":
                raise ValueError("Failure retained; stop without fitting or omitting a document")


def read_responses(lock_path, name, path, sources):
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    if lines[0] != expected_header(lock_path, name) or len(lines) != 41:
        raise ValueError("Complete40-document journal for the correct model required")
    for response, source in zip(lines[1:], sources, strict=True):
        validate_response(response, source)
    return lines[1:]


def direct_response(clf, name, text):
    """Separate unbatched model forwards and aggregation, outside coverage helper."""
    import torch

    encoded = clf.tok(text, padding=True, truncation=True, max_length=512, stride=0,
                      return_overflowing_tokens=True, return_special_tokens_mask=True, return_tensors="pt")
    encoded.pop("overflow_to_sample_mapping")
    specials = encoded.pop("special_tokens_mask")
    weighted = denominator = 0
    with torch.inference_mode():
        for index in range(len(encoded["input_ids"])):
            mini = {k: value[index:index + 1] for k, value in encoded.items()}
            if name == "baseline":
                output = clf.model(input_ids=mini["input_ids"], attention_mask=mini["attention_mask"])["logits"]
            else:
                output = clf.model(**mini).logits
            if output.shape != (1, 4 if name == "baseline" else 2) or not torch.isfinite(output).all():
                raise ValueError("Native audit logits invalid")
            probability = output.float().softmax(dim=-1)[0]
            value = 1 - probability[0] if name == "baseline" else probability[1]
            tokens = int(((mini["attention_mask"][0] != 0) & (specials[index] == 0)).sum())
            if tokens < 1:
                raise ValueError("Every native writing window requires content")
            weighted += method.finite_score(float(value)) * tokens
            denominator += tokens
    return weighted / denominator


def native(lock_path, name, journal, output):
    import torch

    lock, sources = check(lock_path)
    responses = read_responses(lock_path, name, journal, sources)
    torch.set_num_threads(method.METHOD["threads"])
    clf = load_model(name, lock)
    audits = []
    for source, response in zip(sources, responses, strict=True):
        for focus, result in zip(source["foci"], response["foci"], strict=True):
            for view in method.METHOD["views"]:
                if focus["sentence"] != 0 and result[view]["coverage"]["token_windows"] == 1:
                    continue
                proof = token_identity(focus[view], clf.tok)
                actual = direct_response(clf, name, focus[view])
                difference = abs(actual - result[view]["score"])
                audits.append({"id": source["id"], "sentence": focus["sentence"], "view": view,
                               "input_sha256": focus[view + "_sha256"], "score_difference": difference,
                               "coverage_exact": proof == result[view]["coverage"],
                               "passed": proof == result[view]["coverage"] and difference <= 1e-5})
    value = {"schema": "chipmunk.writing-modernbert-native/4", "lock_sha256": method.digest(lock_path),
             "response_sha256": method.digest(journal), "model": name, "audits": audits,
             "probes": len(audits), "passed": all(r["passed"] for r in audits),
             "max_absolute_response_difference": max(r["score_difference"] for r in audits)}
    write(output, value)
    print(json.dumps({k: v for k, v in value.items() if k != "audits"}), flush=True)
    if not value["passed"]:
        raise ValueError("Native input/score correspondence audit failed")


def summarize(lock_path, baseline, candidate, output):
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    lock, sources = check(lock_path)
    journals = {"baseline": read_responses(lock_path, "baseline", baseline, sources),
                "candidate": read_responses(lock_path, "candidate", candidate, sources)}
    arms, author_iou, shifts, paired = {}, {}, {}, {}
    pure_truth = [f["ai_characters"] > 0 for s in sources for f in s["foci"] if f["focus_ai_fraction"] in (0, 1)]
    fractions = [f["focus_ai_fraction"] for s in sources for f in s["foci"]]
    for model, responses in journals.items():
        shifts[model] = {}
        for view in method.METHOD["views"]:
            name = model + "/" + view
            arm = arms[name] = {}
            flattened = [r[view]["score"] for response in responses for r in response["foci"]]
            pure_scores = [score for score, fraction in zip(flattened, fractions, strict=True) if fraction in (0, 1)]
            arm["pure_origin_rank_auc"] = float(roc_auc_score(pure_truth, pure_scores))
            arm["focus_provenance_spearman"] = float(spearmanr(fractions, flattened).statistic)
            arm["coverage"] = {"inputs": len(flattened), **{k: sum(r[view]["coverage"][k] for response in responses for r in response["foci"])
                                                             for k in ("tokens", "token_windows")}}
            for point in method.METHOD["reference_points"]:
                key = str(point)
                per_author = [method.counts(s, [r[view]["score"] for r in response["foci"]], point)
                              for s, response in zip(sources, responses, strict=True)]
                totals = {k: sum(c[k] for c in per_author) for k in per_author[0]}
                values = [method.rates(c)["character_iou"] for c in per_author]
                if any(v is None for v in values):
                    raise ValueError("Mixed-author IoU denominator must stay nonempty")
                author_iou[name, key] = np.array(values)
                arm[key] = {"counts": totals, "rates": method.rates(totals), "author_mean_iou": float(np.mean(values))}
        context = np.array([r["context"]["score"] for response in responses for r in response["foci"]])
        focus = np.array([r["focus"]["score"] for response in responses for r in response["foci"]])
        shifts[model] = {"mean_focus_minus_context": float(np.mean(focus - context)),
                         "mean_absolute_shift": float(np.mean(np.abs(focus - context)))}
        for point in method.METHOD["reference_points"]:
            shifts[model][str(point)] = {"context_only_highlights": int(np.sum((context >= point) & (focus < point))),
                                         "focus_only_highlights": int(np.sum((focus >= point) & (context < point)))}
    rng = np.random.default_rng(method.METHOD["author_bootstrap"]["seed"])
    indices = rng.integers(0, 40, size=(2000, 40))
    for point in method.METHOD["reference_points"]:
        key = str(point)
        difference = author_iou["candidate/focus", key] - author_iou["baseline/context", key]
        samples = difference[indices].mean(axis=1)
        paired[key] = {"author_mean_iou_gain": float(difference.mean()),
                       "percentile95": np.quantile(samples, [0.025, 0.975]).tolist()}
    gates = method.development_gates(arms["baseline/context"], arms["candidate/focus"],
                                     {key: value["percentile95"][0] for key, value in paired.items()}, True)
    historical = {(r["id"], r["sentence"]): r for r in json.loads(Path(lock["historical"]).read_text())["records"]}
    differences = []
    for response in journals["baseline"]:
        for focus in response["foci"]:
            previous = historical[response["id"], focus["sentence"]]
            if previous["context_sha256"] != focus["context_sha256"]:
                raise ValueError("Historical scores are bound to different contexts")
            differences.append(abs(focus["context"]["score"] - method.finite_score(previous["score"])))
    value = {"schema": "chipmunk.writing-modernbert-development/4", "lock_sha256": method.digest(lock_path),
             "input_sha256": method.INPUT_SHA, "baseline_response_sha256": method.digest(baseline),
             "candidate_response_sha256": method.digest(candidate), "documents": 40, "author_groups": 40,
             "foci": 674, "failures": 0, "uncovered_bot_characters": 0, "arms": arms,
             "focus_context_shifts": shifts, "primary_author_paired_comparison": paired,
             "historical_baseline_context_replay": {"inputs": len(differences),
                 "max_absolute_response_difference": max(differences),
                 "differences_above1e_minus5": sum(d > 1e-5 for d in differences),
                 "scope": "Current unchanged full-token baseline rerun; historical scores are comparison only, not reused in primary arms"},
             "development_gates": gates, "qualifies_for_further_development_validation": all(gates.values()),
             "serving_adoption_permitted": False, "fitting_performed": False, "fresh_confirmation": False,
             "outside_reviewers": 0, "scope": method.METHOD["scope"]}
    write(output, value)
    print(json.dumps({"complete": True, "passed_gates": sum(gates.values()), "gates": len(gates),
                      "serving_changed": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    p = commands.add_parser("freeze")
    for name in ("inputs", "historical", "baseline", "candidate", "output", "public"):
        p.add_argument(name, type=Path)
    p.add_argument('--smoke-receipt', type=Path, required=True)
    p = commands.add_parser('smoke')
    p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path)
    p = commands.add_parser("score")
    p.add_argument("lock", type=Path)
    p.add_argument("model", choices=("baseline", "candidate"))
    p.add_argument("output", type=Path)
    p = commands.add_parser("native")
    p.add_argument("lock", type=Path)
    p.add_argument("model", choices=("baseline", "candidate"))
    p.add_argument("journal", type=Path)
    p.add_argument("output", type=Path)
    p = commands.add_parser("summarize")
    for name in ("lock", "baseline", "candidate", "output"):
        p.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.inputs, args.historical, args.baseline, args.candidate, args.output, args.public, args.smoke_receipt)
    elif args.stage == 'smoke':
        smoke(args.candidate, args.output)
    elif args.stage == "score":
        score(args.lock, args.model, args.output)
    elif args.stage == "native":
        native(args.lock, args.model, args.journal, args.output)
    else:
        summarize(args.lock, args.baseline, args.candidate, args.output)
