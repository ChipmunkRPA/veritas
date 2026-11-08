"""Bound source decoding/context comparison; retain original trials on failure."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as previous

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    previous.MODULES = {'scripts.run_background_context_v25'}
    previous.run('scripts.run_background_context_v25', args.arguments, args.folder, 900)
