"""Guard the separate post-failure investigation; preserve the original audit."""
import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    module = 'scripts.diagnose_image_patch_head_v14'
    guard.MODULES = {module}
    guard.run(module, ['run', str(args.trial)], args.folder, 900)
