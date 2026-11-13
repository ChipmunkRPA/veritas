"""Run public ledger checks and save a privacy-safe receipt for independent review."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = "docs/evaluation-panels"
SCRIPTS = "backend/scripts"
CHECKS = {
    "image": ("review_image_thumbnail_extent_transfer.py", ("image-thumbnail-extent-transfer-2026-11-12.json",), ()),
    "video": ("review_video_motion_late_transfer.py", ("video-motion-late-transfer-2026-11-12.json",),
              (f"{SCRIPTS}/review_video_lossy_sequence_transfer.py",)),
    "audio": ("audio_vocal_event_mixtures.py", ("audio-vocal-event-mixtures-2026-11-12.json",),
              (f"{PANELS}/audio-vocal-event-source-roster-2026-11-12.json",
               f"{PANELS}/audio-remote-neighbor-replication-2026-11-12.json")),
    "writing": ("review_writing_elife_exact_sentence_confirmation.py", ("writing-elife-exact-sentence-confirmation-2026-11-12.json",),
                (f"{PANELS}/writing-multiblock-independent-2026-11-12.json",)),
}
LOCALIZATION = {
    "image": ("review_image_boundary_multiple_transfer.py", "image-boundary-multiple-transfer-2026-11-12.json"),
    "video": ("review_video_reference_soundtrack_panel.py", "video-paired-soundtrack-transfer-2026-11-12.json"),
    "audio": ("review_audio_partialedit_dense_transfer.py", "audio-partialedit-dense-transfer-2026-11-12.json"),
    "writing": ("review_writing_source_conditioned.py", "writing-source-conditioned-2026-11-12.json"),
}
ISOLATED = (
    "import runpy,sys; "
    "script=sys.argv.pop(1); "
    "sys.argv[0]=script; "
    "sys.path[:0]=['backend','backend/scripts']; "
    "runpy.run_path(script,run_name='__main__')"
)


def hash_file(relative: str) -> str | None:
    path = ROOT / relative
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def revision() -> str | None:
    if not (ROOT / ".git").exists():
        return None
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   check=True, timeout=3, text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    head = completed.stdout.strip()
    return head if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) else None


def replay(modality: str, suite: str = "overview") -> dict:
    if suite == "localization":
        script, panel = LOCALIZATION[modality]
        panel_names, dependencies = (panel,), ()
    else:
        script, panel_names, dependencies = CHECKS[modality]
    entry = f"{SCRIPTS}/{script}"
    panels = [f"{PANELS}/{name}" for name in panel_names]
    fingerprints = {path: hash_file(path) for path in [entry, *panels, *dependencies]}
    arguments = ["review", *panels] if modality == "audio" and suite == "overview" else panels
    record = {"files": fingerprints, "entry": entry, "arguments": arguments}
    if None in fingerprints.values():
        return record | {"status": "missing_input"}
    try:
        completed = subprocess.run([sys.executable, "-B", "-I", "-S", "-c", ISOLATED, entry, *arguments],
                                   cwd=ROOT, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return record | {"status": "timed_out"}
    if fingerprints != {path: hash_file(path) for path in fingerprints}:
        return record | {"status": "inputs_changed"}
    return record | {"status": "matched" if completed.returncode == 0 else "rejected",
                     "exit_code": completed.returncode, "output_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                     "diagnostic_sha256": hashlib.sha256(completed.stderr).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="Write a new JSON receipt")
    parser.add_argument("--modality", action="append", choices=CHECKS, help="Select a modality (default: all four)")
    parser.add_argument("--suite", choices=("overview", "localization"), default="overview",
                        help="Select the published study set (default: overview)")
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("Choose a new receipt path")
    modalities = tuple(dict.fromkeys(args.modality or CHECKS))
    checks = {name: replay(name, args.suite) for name in modalities}
    receipt = {
        "schema": "chipmunk.public-evaluation-replay/1",
        "scope": "Arithmetic replay of published evaluation ledgers",
        "source_revision": revision(),
        "environment": {"python": platform.python_version(), "os": platform.system(), "architecture": platform.machine()},
        "runner_sha256": hash_file(f"{SCRIPTS}/replay_public_evaluations.py"),
        "checks": checks,
    }
    if args.suite != "overview":
        receipt["suite"] = args.suite
    with args.receipt.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")
    for name, result in checks.items():
        print(f"{name}: {result['status']}")
    if any(result["status"] != "matched" for result in checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
