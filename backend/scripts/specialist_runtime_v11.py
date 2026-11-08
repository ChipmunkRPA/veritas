"""Bound ordinary-edit preparation/inference; keep prior runners immutable."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as previous

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.seconds <= 900:
        parser.error('Ordinary-edit job must be bounded by900seconds')
    previous.MODULES = {'scripts.run_genuine_edits_v24'}
    previous.run('scripts.run_genuine_edits_v24', args.arguments, args.folder, args.seconds)
