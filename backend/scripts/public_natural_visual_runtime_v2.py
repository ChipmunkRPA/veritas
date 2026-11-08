"""Bound source acquisition, native scoring and separate natural-media replay."""
import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('acquire', 'run', 'review'))
    p.add_argument('profile', type=Path)
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = p.parse_args()
    module = {'acquire': 'scripts.prepare_public_visual_natural_v2',
              'run': 'scripts.run_public_visual_natural_v2',
              'review': 'scripts.review_public_visual_natural_v2'}[args.action]
    guard.MODULES = {module}
    guard.run(module, args.arguments, args.profile, 900)
