"""Sequential native and source-grid audit under existing local resource limits."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module", choices=("scripts.run_image_iml_zoom_v16", "scripts.review_image_iml_zoom_v16")
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    guard.MODULES = {args.module}
    guard.run(args.module, args.arguments, args.folder, 900)
