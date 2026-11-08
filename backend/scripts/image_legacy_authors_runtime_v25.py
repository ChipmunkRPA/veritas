"""Data-only explicit module on the unchanged900s/6GiB/1536MiB local monitor."""

import argparse
from pathlib import Path

from scripts import relayformer_learning_runtime_v22 as monitor

MODULE = "scripts.image_legacy_authors_v25"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    monitor.MODULES = {MODULE}
    monitor.run(MODULE, args.arguments, args.profile_folder)
