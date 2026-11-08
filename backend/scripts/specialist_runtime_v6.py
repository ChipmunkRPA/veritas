"""Bounded video-face native runner; older frozen runtimes stay unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import specialist_runtime_v4 as previous

MODULE = 'scripts.score_video_specialists_v19'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--seconds', type=int, default=3600)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600:
        parser.error('Duration must be positive and at most3600seconds')
    if args.validate_only:
        print(json.dumps({'module': MODULE, 'arguments': args.arguments,
            'seconds_limit': args.seconds, 'child_started': False, 'output_created': False}))
    else:
        # Extend only this fresh process's allowlist; runtime4 source is immutable.
        previous.MODULES = {MODULE}
        previous.run(MODULE, args.arguments, args.folder, args.seconds)
