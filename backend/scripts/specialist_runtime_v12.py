"""Guard fresh image acquisition separately from subsequent native scoring."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as previous

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--seconds', type=int, default=900)
    parser.add_argument('module', choices=('scripts.prepare_image_confirmation_v12', 'scripts.run_image_confirmation_v12',
        'scripts.review_image_confirmation_v12'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    limit = 3600 if args.module == 'scripts.prepare_image_confirmation_v12' else 900
    if not 0 < args.seconds <= limit:
        parser.error(f'Declared stage duration must be at most {limit} seconds')
    previous.MODULES = {'scripts.prepare_image_confirmation_v12', 'scripts.run_image_confirmation_v12',
        'scripts.review_image_confirmation_v12'}
    previous.run(args.module, args.arguments, args.folder, args.seconds)
