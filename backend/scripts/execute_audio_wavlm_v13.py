"""Call frozen research functions with one owner of inter-op initialization.

The legacy CLI and original-model loader both set PyTorch inter-op threads;
PyTorch permits that initialization once per process. This dedicated launcher
sets the declared intra-op threads and lets the unchanged loader own inter-op.
No frozen implementation, input, fitted object or acceptance rule is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    import torch

    from scripts import review_audio_wavlm_v13 as audit
    from scripts import run_audio_wavlm_v13 as frozen

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("compare", "native"))
    p.add_argument("lock", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--responses", type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    # Do not pre-initialize inter-op: the frozen original-model loader owns it.
    if args.stage == "compare":
        frozen.compare(args.lock, args.receipt, args.head, args.output)
    else:
        if args.responses is None:
            p.error("Native replay requires the complete original responses")
        audit.native(args.lock, args.receipt, args.head, args.responses, args.output)


if __name__ == "__main__":
    main()
