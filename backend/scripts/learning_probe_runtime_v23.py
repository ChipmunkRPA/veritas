"""Explicit new authored-probe module on the unchanged generic local monitor."""

import argparse
from pathlib import Path

from scripts import relayformer_learning_runtime_v22 as monitor

MODULE = "scripts.relayformer_learning_v23"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    monitor.MODULES = {MODULE}
    monitor.run(MODULE, args.arguments, args.profile_folder)
