"""Guard native public visual-watermark qualification and separate replay."""

import argparse
from pathlib import Path

from scripts import specialist_runtime_v4 as guard

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "review"])
    p.add_argument("profile", type=Path)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    module = (
        "scripts.run_public_visual_watermark_v1"
        if a.action == "run"
        else "scripts.review_public_visual_watermark_v1"
    )
    guard.MODULES = {module}
    guard.run(module, a.arguments, a.profile, 900)
