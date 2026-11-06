"""Publish fixed recipes or complete numerical outcomes, never source audio."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.prepare_audio_confirmation_v7 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import load_rows


def export(root, destination, full=False):
    from scripts.score_audio_confirmation_v7 import check

    lock_path = root / "lock.json"
    lock = check(lock_path)
    variants_path = Path(lock["variants_path"])
    variants = json.loads(variants_path.read_text())
    fields = (
        "schema",
        "variants_sha256",
        "prepared_sources_sha256",
        "head_sha256",
        "serving_calibration_sha256",
        "serving_bundle",
        "representation_header",
        "implementation_sha256",
        "runtime",
        "weights",
        "planned",
        "scope",
        "consumed_parity_sha256",
    )
    public_lock = {key: lock[key] for key in fields}
    head = json.loads(Path(lock["head_path"]).read_text())
    if (
        hashlib.sha256((json.dumps(head, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
        != lock["head_sha256"]
    ):
        raise ValueError("Exact JSON head serialization required for portable artifact verification")
    body = {
        "schema": "chipmunk.audio-fresh-confirmation-public/7",
        "original_lock_sha256": digest(lock_path),
        "lock": public_lock,
        "head": head,
        "variants": [
            {k: v for k, v in row.items() if k not in ("path", "production_input")}
            for row in variants["variants"]
        ],
        "source_catalogue": "audio-v7-fresh-sources.json",
        "source_catalogue_sha256": digest(destination / "audio-v7-fresh-sources.json"),
        "input_audit": json.loads((root / "input-audit.json").read_text()),
        "consumed_parity": json.loads((root / "consumed-parity.json").read_text()),
        "detector_inference_run": full,
        "deployment_eligible": False,
        "scope": "Fixed fresh source recipes and small fitted/fixed head/projection parameters; source waveforms, full transcripts, private paths and pretrained checkpoints excluded. No outside reviewer.",
    }
    if full:
        header, outcomes = load_rows(root / "scores.jsonl")
        summary = json.loads((root / "summary.json").read_text())
        native = json.loads((root / "native-audit.json").read_text())
        if (
            header["lock_sha256"] != digest(lock_path)
            or summary["scores_sha256"] != digest(root / "scores.jsonl")
            or len(outcomes) != 1080
        ):
            raise ValueError("All frozen outcomes and bound summary required")
        if native["lock_sha256"] != digest(lock_path) or native["scores_sha256"] != digest(
            root / "scores.jsonl"
        ):
            raise ValueError("Native audit must refer to these exact immutable outcomes")
        body.update(response_header=header, outcomes=outcomes, summary=summary, native_audit=native)
    raw = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
    if b"/Users/" in raw or b"/private/" in raw or b"/home/" in raw:
        raise ValueError("Private filesystem path survived explicit export field selection")
    compressed = gzip.compress(raw, mtime=0)
    name = "audio-v7-confirmation-review.json.gz" if full else "audio-v7-confirmation-inputs.json.gz"
    with (destination / name).open("xb") as stream:
        stream.write(compressed)
    receipt = {
        "schema": "chipmunk.audio-fresh-confirmation-public-receipt/7",
        "file": name,
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "bytes": len(compressed),
        "original_lock_sha256": digest(lock_path),
        "variants_sha256": lock["variants_sha256"],
        "head_sha256": lock["head_sha256"],
        "inputs": len(body["variants"]),
        "detector_inference_run": full,
        "deployment_eligible": False,
    }
    if full:
        write(destination.parent / "audio-guard-v7-confirmation-evaluation.json", summary)
    else:
        write(destination.parent / "audio-guard-v7-confirmation-execution.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    export(args.root, args.destination, args.full)
