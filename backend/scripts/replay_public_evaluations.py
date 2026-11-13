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
SCHEMA = "chipmunk.public-evaluation-replay/1"
SCOPE = "Arithmetic replay of published evaluation ledgers"
SHA256 = re.compile(r"[0-9a-f]{64}")
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
TRANSFERS = {
    "image": ("review_image_png_layout_guard.py", "image-png-layout-guard-2026-11-12.json", (), (
        f"{SCRIPTS}/image_png_layout_guard_transfer.py", f"{SCRIPTS}/capture_image_png_layout_controls.cjs",
        f"{PANELS}/image-v7-smaug-genuine-operating-point.json", f"{PANELS}/image-whole-png-scope-probe-2026-11-12.json",
        f"{PANELS}/image-whole-png-transfer-2026-11-12.json", f"{PANELS}/image-whole-fresh-sources-2026-11-12.json",
    )),
    "video": ("review_grid_phone_viseme_timing.py", "grid-phone-viseme-timing-2026-11-12.json", (), ()),
    "audio": ("audio_short_coarse_transfer.py", "audio-short-coarse-transfer-2026-11-12.json", ("review",), (
        f"{SCRIPTS}/review_audio_reduced_background_localization.py", f"{PANELS}/audio-noise-window-transfer-2026-11-12.json",
    )),
    "writing": ("writing_local_lexical_transfer.py", "writing-local-lexical-transfer-2026-11-12.json", ("review",), (
        f"{PANELS}/writing-elife-exact-sentence-confirmation-2026-11-12.json",
        f"{PANELS}/writing-multiblock-independent-2026-11-12.json",
    )),
}
SUITES = ("overview", "localization", "transfer")
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


def replay_inputs(modality: str, suite: str) -> tuple[str, list[str], tuple[str, ...]]:
    if suite == "localization":
        script, panel = LOCALIZATION[modality]
        panel_names, dependencies = (panel,), ()
        prefix = ()
    elif suite == "transfer":
        script, panel, prefix, dependencies = TRANSFERS[modality]
        panel_names = (panel,)
    else:
        script, panel_names, dependencies = CHECKS[modality]
        prefix = ("review",) if modality == "audio" else ()
    entry = f"{SCRIPTS}/{script}"
    panels = [f"{PANELS}/{name}" for name in panel_names]
    arguments = [*prefix, *panels]
    return entry, arguments, (entry, *panels, *dependencies)


def replay(modality: str, suite: str = "overview") -> dict:
    entry, arguments, inputs = replay_inputs(modality, suite)
    fingerprints = {path: hash_file(path) for path in inputs}
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


def load_receipt(path: Path) -> dict:
    with path.open("rb") as source:
        data = source.read(32_769)
    if len(data) > 32_768:
        raise ValueError("Replay receipt exceeds 32 KiB")
    receipt = json.loads(data)
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA or receipt.get("scope") != SCOPE:
        raise ValueError("Unknown replay receipt format")
    suite = receipt.get("suite", "overview")
    checks = receipt.get("checks")
    if (suite not in SUITES or not isinstance(checks, dict) or not checks
            or not checks.keys() <= CHECKS.keys() or not is_sha256(receipt.get("runner_sha256"))):
        raise ValueError("Invalid replay suite, checks or runner commitment")
    for modality, check in checks.items():
        entry, arguments, inputs = replay_inputs(modality, suite)
        if not isinstance(check, dict) or check.get("entry") != entry or check.get("arguments") != arguments:
            raise ValueError(f"Invalid {modality} replay command")
        files = check.get("files")
        if (not isinstance(files, dict) or files.keys() != set(inputs)
                or any(value is not None and not is_sha256(value) for value in files.values())):
            raise ValueError(f"Invalid {modality} input commitments")
        if check.get("status") == "matched":
            if (None in files.values() or type(check.get("exit_code")) is not int or check["exit_code"] != 0
                    or not is_sha256(check.get("output_sha256")) or not is_sha256(check.get("diagnostic_sha256"))):
                raise ValueError(f"Invalid successful {modality} result")
        elif check.get("status") not in ("missing_input", "timed_out", "inputs_changed", "rejected"):
            raise ValueError(f"Invalid {modality} result status")
    return receipt


def is_sha256(value) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def compare(left: dict, right: dict) -> bool:
    if left.get("suite", "overview") != right.get("suite", "overview"):
        print("Replay suites differ.")
        return False
    if left["runner_sha256"] != right["runner_sha256"]:
        print("Runner versions differ. Checking the recorded reader inputs and outputs.")
    first, second = left["checks"], right["checks"]
    all_matched = True
    for modality in sorted(first.keys() | second.keys()):
        a, b = first.get(modality), second.get(modality)
        if a is None or b is None:
            outcome = "missing from one receipt"
        elif a["status"] != "matched" or b["status"] != "matched":
            outcome = "one or both replays failed"
        elif a["files"] != b["files"]:
            outcome = "input commitments differ"
        elif a["output_sha256"] != b["output_sha256"]:
            outcome = "output commitments differ"
        elif a["diagnostic_sha256"] != b["diagnostic_sha256"]:
            outcome = "output matches, diagnostics differ"
        else:
            outcome = "matched"
        print(f"{modality}: {outcome}")
        all_matched &= outcome == "matched"
    return all_matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="New JSON receipt path, or the first receipt in a comparison")
    parser.add_argument("--compare", type=Path, help="Compare two saved receipts without running their commands")
    parser.add_argument("--modality", action="append", choices=CHECKS, help="Select a modality (default: all four)")
    parser.add_argument("--suite", choices=SUITES, default="overview",
                        help="Select the published study set (default: overview)")
    args = parser.parse_args()
    if args.compare is not None:
        if args.modality or args.suite != "overview":
            parser.error("A comparison uses the suites and modalities recorded in the receipts")
        try:
            matched = compare(load_receipt(args.receipt), load_receipt(args.compare))
        except (OSError, ValueError, RecursionError):
            parser.error("Could not read valid public replay receipts")
        raise SystemExit(0 if matched else 1)
    if args.receipt.exists():
        parser.error("Choose a new receipt path")
    modalities = tuple(dict.fromkeys(args.modality or CHECKS))
    checks = {name: replay(name, args.suite) for name in modalities}
    receipt = {
        "schema": SCHEMA,
        "scope": SCOPE,
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
