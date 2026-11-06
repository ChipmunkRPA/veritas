"""Rebuild the email reference bundle from frozen detector receipts without activating it."""
from __future__ import annotations

import argparse
import json
import sys
from email.message import EmailMessage
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from calibrate_email_upgrade import read_cases
from calibrate_text_upgrade import features, quality, read_predictions
from prepare_text_upgrade import digest, file_hash
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.detectors.text.extract import extract_eml, from_string
from app.engine.registry import get_registry
from app.engine.stages.document_screening import DETECTOR, MODEL, email_reference_scope


def scope_rows(rows: list[dict]) -> list[dict]:
    accepted = []
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "message.eml"
        for row in rows:
            message = EmailMessage()
            message["From"] = "Writer <writer@example.test>"
            message["To"] = "Recipient <recipient@example.test>"
            message["Subject"] = "Comparison input"
            message.set_content(row["text"])
            path.write_bytes(message.as_bytes())
            doc = extract_eml(path)
            if email_reference_scope(doc) and doc.text == from_string(row["text"]).text:
                accepted.append(row)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("reference_scores", type=Path)
    parser.add_argument("transfer_manifest", type=Path)
    parser.add_argument("transfer_scores", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    reference = comparison["models"]["tropa"]
    manifest = get_registry().manifest
    data = {
        "version": "1.1.0-local-email", "bench_version": "email-authorship-local-v1",
        "manifest_sha256": comparison["reference_manifest_sha256"],
        "transfer_manifest_sha256": comparison["transfer_manifest_sha256"],
        "comparison_sha256": file_hash(args.comparison), "detector_id": DETECTOR,
        "detector_version": manifest.detector(DETECTOR).version,
        "checkpoint_revision": manifest.model(MODEL).source.revision,
        "reference_ai_prior": reference["calibration"]["reference_ai_prior"], "min_words": 50, "max_words": 220,
        "estimate": [reference["calibration"]["intercept"], reference["calibration"]["slopes"][0]],
        "bootstraps": [[row["intercept"], row["slopes"][0]] for row in reference["calibration_bootstrap"]],
    }
    for key, source, scores, expected in (
        ("holdout", args.reference_manifest, args.reference_scores, data["manifest_sha256"]),
        ("transfer", args.transfer_manifest, args.transfer_scores, data["transfer_manifest_sha256"]),
    ):
        if file_hash(source) != expected:
            raise ValueError("Source manifest differs from the frozen comparison")
        cases = read_cases(source)
        model, responses = read_predictions(scores, expected, cases)
        if model != "tropa":
            raise ValueError("Score receipts do not match the calibrated detector")
        rows = scope_rows([row for row in cases.values() if row["domain"] == "email" and row["split"] == "holdout"])
        x = features(np.array([[responses[row["id"]]] for row in rows]), ("tropa",))
        estimate = expit(data["estimate"][0] + data["estimate"][1] * x[:, 0])
        labels = np.array([int(row["label"] == "ai") for row in rows])
        data[key] = quality(labels, estimate)
        data[key + "_eligible_ids_sha256"] = digest("\n".join(row["id"] for row in rows))
        print(key, data[key])
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(file_hash(args.output))


if __name__ == "__main__":
    main()
