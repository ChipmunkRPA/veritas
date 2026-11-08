"""Guard the model-free image feature comparison without altering frozen runners."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    guard.MODULES = {"scripts.run_image_residual_v13"}
    guard.run("scripts.run_image_residual_v13", args.arguments, args.folder, 1800)
