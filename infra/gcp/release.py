"""Create/verify a hash manifest for an explicitly staged release, never the live evidence tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath

MANIFEST = "runtime-release.json"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inventory(root: Path) -> dict[str, str]:
    fusion = root / "fusion"
    if fusion.is_symlink():
        raise ValueError("fusion directory must not be a symlink")
    files = sorted(fusion.rglob("*"))
    if not files:
        raise ValueError("stage reviewed fuser JSON and bootstrap arrays in data/fusion first")
    result = {}
    for path in files:
        if path.is_symlink():
            raise ValueError("release assets must not be symlinks")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = digest(path)
    for kind in ("image", "video", "audio"):
        if not any(re.fullmatch(rf"fusion/fuser-{kind}-[^/]+\.json", name) for name in result):
            raise ValueError(f"a reviewed {kind} fuser is required")
    return result


def verify(root: Path, commit: str) -> dict:
    document = json.loads((root / MANIFEST).read_text())
    if document.get("schema") != "veritas.runtime-release/1" or document.get("build_commit") != commit:
        raise ValueError("release manifest schema/build commit mismatch")
    if document.get("licence_policy") != "commercial":
        raise ValueError("release licence policy must be commercial")
    files = document.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("empty release manifest")
    for name, expected in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "fusion":
            raise ValueError("release asset must be a relative fusion path")
        if not re.fullmatch(r"[a-f0-9]{64}", str(expected)):
            raise ValueError("invalid release SHA-256")
    if inventory(root) != files:
        raise ValueError("release fusion files differ from the reviewed manifest")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.commit):
        parser.error("--commit requires a complete Git commit")
    if args.action == "create":
        target = args.data_dir / MANIFEST
        if target.exists():
            raise SystemExit("refusing to overwrite a release manifest; use a new staging directory")
        document = {"schema": "veritas.runtime-release/1", "build_commit": args.commit,
                    "licence_policy": "commercial", "files": inventory(args.data_dir)}
        with target.open("x") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
    verify(args.data_dir, args.commit)
    print("Release fusion hashes verified; weight availability and fuser compatibility are checked at VM startup.")


if __name__ == "__main__":
    main()
