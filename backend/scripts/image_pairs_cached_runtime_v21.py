"""Keep the same900s/6GiB guard for the explicit cache-only entry point."""

import argparse
from pathlib import Path

from scripts import data_resource_guard_v21 as guard

MODULES = {"scripts.image_positive_cached_v21"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=sorted(MODULES))
    parser.add_argument("folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    guard.run(args.module, args.arguments, args.folder)
