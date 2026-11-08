"""900-second/6GiB guard for explicit local data-qualification phases."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

MODULES = {"scripts.image_positive_pairs_v21"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=sorted(MODULES))
    parser.add_argument("folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    guard.MODULES = MODULES
    guard.run(args.module, args.arguments, args.folder, 900)
