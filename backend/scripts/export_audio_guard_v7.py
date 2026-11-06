"""Publish licensed-source metadata and CPU fitting arithmetic, never waveforms."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

from app.detectors.audio import calibration as serving
from scripts.export_mouth_context_v10 import safe
from scripts.prepare_audio_guard_v7 import check
from scripts.run_audio_guard_v7 import digest, feature_rows, load_rows


def export(manifest, receipt, head_path, scores, summary_path, destination):
    m, header, rows = feature_rows(manifest, receipt)
    check(manifest)
    head = json.loads(head_path.read_text())
    score_header, results = load_rows(scores)
    summary = json.loads(summary_path.read_text())
    if (
        summary["scores_sha256"] != digest(scores)
        or summary["head_sha256"] != digest(head_path)
        or summary["receipt_sha256"] != digest(receipt)
    ):
        raise ValueError("Wrong source/fit/score summary bindings")
    # The numerical bootstrap helper is inherited; its old CV wording does not
    # describe this fixed fit/cal/development experiment. Preserve its original
    # file hash while giving the public copy an accurate nonnumeric scope.
    summary["original_summary_sha256"] = digest(summary_path)
    for difference in summary["differences"].values():
        difference["interpretation"] = (
            "Consumed-source development conditional on the fixed fitting/calibration and reused backgrounds; "
            "excludes fitting/calibration uncertainty. Not cross-validation, fresh testing or outside review."
        )
    summary["limitations"] = [
        "One fixed fit/cal/development split; development inputs were previously consumed.",
        "Fitting has eight known synthetic voice groups and one generator family; source/channel/duration confounding remains possible.",
        "Known metadata groups do not establish biometric cross-corpus identity or unknown pretrained overlap.",
        "Intervals condition on the fixed learner/guard and reused background pool, excluding fitting/calibration uncertainty.",
        "No new whole-recording, per-customer error-rate, overall-confidence or outside-validation claim.",
    ]
    fields = (
        "id",
        "base_id",
        "role",
        "group",
        "split",
        "label",
        "condition",
        "raw",
        "raw_status",
        "sha256",
        "pcm_sha256",
        "seconds",
        "samples",
        "sample_rate",
        "original_windows",
        "source_interval_samples",
        "background_sha256",
        "background_id",
        "background_kind",
        "background_start_sample",
        "snr_db",
        "common_peak_gain",
        "measured_snr_db",
        "dataset",
        "generator",
        "cluster_id",
        "speaker",
        "serving_p",
        "serving_decision",
        "embedding",
    )
    public_sources = []
    for row in m["sources"]:
        public = {
            k: row[k]
            for k in (
                "id",
                "split",
                "label",
                "cluster_id",
                "speaker",
                "source_group",
                "sha256",
                "decoded_pcm_sha256",
                "url",
                "license",
                "license_url",
                "generator",
                "seconds",
            )
            if k in row
        }
        text = " ".join(re.findall(r"\w+", row["transcript"].casefold())) if row.get("transcript") else None
        public["role_text_sha256"] = hashlib.sha256(text.encode()).hexdigest() if text else None
        public_sources.append(public)
    result_fields = tuple(k for k in fields if k != "embedding") + (
        "candidate_p",
        "candidate_decision",
        "candidate_detail",
        "probe",
        "original_p",
        "original_decision",
        "runtime_s",
    )
    body = {
        "schema": "chipmunk.audio-source-guard-public/7",
        "original_manifest_sha256": digest(manifest),
        "original_receipt_sha256": digest(receipt),
        "original_head_sha256": digest(head_path),
        "original_scores_sha256": digest(scores),
        "implementation_sha256": m["implementation_sha256"],
        "sources": public_sources,
        "groups": m["groups"],
        "representation_header": header,
        "rows": [{k: r[k] for k in fields if k in r} for r in rows],
        "head": head,
        "fit_record": json.loads((head_path.parent / "fit.json").read_text()),
        "score_header": score_header,
        "results": [{k: r[k] for k in result_fields if k in r} for r in results],
        "reference": m["reference"],
        "serving_bundle": serving.load(),
        "summary": summary,
        "scope": "Consumed licensed-source development and CPU reproduction; full neural checkpoints and waveforms excluded. Small fixed/fitted head and projection parameters included. Not outside validation.",
        "deployment_eligible": False,
    }
    body["statistical_runtime"] = json.loads((manifest.parent / "statistical-runtime.json").read_text())
    data = gzip.compress(safe(body), mtime=0)
    with (destination / "audio-v7-fitting-review.json.gz").open("xb") as stream:
        stream.write(data)
    with (destination.parent / "audio-guard-v7-evaluation.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "public_capsule_sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "input_rows": len(rows),
                "development_results": len(results),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export(
        args.root / "manifest.json",
        args.root / "representations.jsonl",
        args.root / "trained/head.json",
        args.root / "development-scores.jsonl",
        args.root / "summary.json",
        args.destination,
    )
