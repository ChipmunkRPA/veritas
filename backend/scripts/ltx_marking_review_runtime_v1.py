"""Bound the separate sample/bit/codec replay without altering the native freeze."""
import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as previous

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    previous.MODULES = {'scripts.review_ltx_marking_audio_v1'}
    previous.run('scripts.review_ltx_marking_audio_v1', args.arguments, args.folder, 900)
