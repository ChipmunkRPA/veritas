"""Separate900-second/6GiB local guard; preserve the attribution freeze."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

MODULES = {"scripts.review_image_pixel_lineage_v20"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=sorted(MODULES))
    parser.add_argument("folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    guard.MODULES = MODULES
    guard.run(args.module, args.arguments, args.folder, 900)
