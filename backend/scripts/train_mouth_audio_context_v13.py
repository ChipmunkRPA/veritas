"""External dispatch repair: invoke the already-frozen fitter without extraction.

The original CLI uses 'fit' for both extraction and training, so its training
function is unreachable through that CLI. No scientific function is changed.
Publish this launcher before fitting; retain original extraction and freeze.
"""
import argparse
from pathlib import Path

from scripts import run_mouth_audio_context_v13 as study
from scripts import specialist_runtime_v4 as guard

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('guarded', 'worker'))
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    if args.mode == 'worker':
        study.fit(args.trial)
    else:
        guard.MODULES = {'scripts.train_mouth_audio_context_v13'}
        guard.run('scripts.train_mouth_audio_context_v13', ['worker', str(args.trial)], args.trial / 'fit-profile', 900)
