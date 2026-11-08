"""Sequential 900-second,6GiB RSS/1536MiB free RAM/disk guards for the study."""
import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('native', 'review'))
    parser.add_argument('folder', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    module = 'scripts.run_image_patch_head_v14' if args.mode == 'native' else 'scripts.review_image_patch_head_v14'
    guard.MODULES = {module}
    guard.run(module, args.arguments, args.folder, 900)
