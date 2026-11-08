"""Bound separate source preparation, native execution and model-free review."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

MODULES = {
    "prepare": "scripts.prepare_public_audioseal_speech_v2",
    "run": "scripts.run_public_audioseal_speech_v2",
    "review": "scripts.review_public_audioseal_speech_v2",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=MODULES)
    parser.add_argument("profile", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = MODULES[args.action]
    guard.MODULES = {module}
    guard.run(module, args.arguments, args.profile, 900)
