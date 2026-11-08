"""Local900-second/6GiB resource guard for the new image localizer study."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module", choices=("scripts.inspect_relayformer_v18", "scripts.relayformer_native_v18")
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    guard.MODULES = {"scripts.inspect_relayformer_v18", "scripts.relayformer_native_v18"}
    guard.run(args.module, args.arguments, args.folder, 900)
