"""Isolated guard for the prospective sound/mouth experiment and its audit."""
import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as previous

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', choices=('scripts.run_mouth_audio_context_v14', 'scripts.review_mouth_audio_context_v14'))
    parser.add_argument('folder', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    previous.MODULES = {args.module}
    previous.run(args.module, args.arguments, args.folder, 900)
