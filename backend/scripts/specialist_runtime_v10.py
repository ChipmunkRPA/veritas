"""Bound the new model-free solver/combiner study; older runners stay frozen."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as previous

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.seconds <= 600:
        parser.error('Scalar job limit must be positive and at most600seconds')
    previous.MODULES = {'scripts.run_preserved_video_v23'}
    previous.run('scripts.run_preserved_video_v23', args.arguments, args.folder, args.seconds)
