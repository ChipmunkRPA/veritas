"""Explicit diagnostic-only extension of the unchanged generic local monitor."""

import argparse
from pathlib import Path

from scripts import relayformer_learning_runtime_v22 as monitor

MODULE = "scripts.diagnose_relayformer_learning_v22"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_folder", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # The generic run function accepts an explicit module argument. This wrapper
    # provides its own one-module allowlist; resource and failure rules are intact.
    monitor.MODULES = {MODULE}
    monitor.run(MODULE, args.arguments, args.profile_folder)
