"""Sequential900-second/6GiB local native guard; no hosted workflows."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = "scripts.image_imdet_native_v17"
    guard.MODULES = {module}
    guard.run(module, args.arguments, args.folder, 900)
