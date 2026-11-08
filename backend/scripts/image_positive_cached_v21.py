"""Explicit cache-only successor entry point; frozen pair rules stay unchanged."""

import argparse
from pathlib import Path

from scripts import image_positive_pairs_v21 as frozen
from scripts.ranged_gzip_cache_v21 import CachedRangeFile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("step", "qualify"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", choices=tuple(r + "-" + k for r in ("training", "validation", "testing") for k in ("edits", "masks")))
    parser.add_argument("--number", type=int)
    args = parser.parse_args()
    # Only the transport cache changes; selection, body validation and pair math
    # still execute the original, independently hash-protected implementation.
    frozen.RangeFile = CachedRangeFile
    if args.command == "step":
        frozen.require(args.archive is not None and args.number is not None, "Explicit archive and phase required")
        frozen.step(args.folder.resolve(), args.archive, args.number)
    else:
        frozen.qualify(args.folder.resolve())
