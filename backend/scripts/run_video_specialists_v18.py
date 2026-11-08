"""Video specialist experiment stages; source audit precedes model inference."""
from __future__ import annotations

import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze-input-audit', 'audit-inputs'))
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    args = parser.parse_args()
    from scripts.audit_video_scoring_inputs_v18 import audit, freeze

    if args.stage == 'freeze-input-audit':
        freeze(args.trial.resolve())
    else:
        audit(args.main.resolve(), args.trial.resolve())
