# SPDX-License-Identifier: Apache-2.0
"""Replay the production Full face stages on the same AVC and VP9 deliveries."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from scripts import replay_video_face_short_prefix_incumbent as incumbent
from scripts import video_face_short_carrier_transfer as study
from scripts.video_common_sources_v89 import read, require, sha, write


def run(folder, carrier, split):
    profile = study.FORMATS[carrier]
    source = folder / f"{carrier}-{split}.json"
    target = folder / f"{carrier}-{split}-incumbent.json"
    require(read(source)["freeze_sha256"] == sha(folder / "freeze.json"), "The scored source panel differs")

    def read_panel(path):
        return read(source if path == folder / f"{split}.json" else path)

    def panel_sha(path):
        return sha(source if path == folder / f"{split}.json" else path)

    def write_panel(path, data):
        require(path == folder / f"{split}-incumbent.json", "Unexpected incumbent receipt destination")
        write(target, data)

    baseline_scorer = incumbent.worker.scorer

    def scorer(state):
        from app.core.config import settings

        settings.video_face_short_window_review = False
        measure = baseline_scorer(state)

        def score(path, row):
            group = path.parent.name.removeprefix(f"{split}-")
            actual = folder / f"{carrier}-{split}-{group}" / path.with_suffix("." + profile["suffix"]).name
            return measure(actual, row)

        return score

    with (
        patch.object(incumbent, "read", read_panel),
        patch.object(incumbent, "sha", panel_sha),
        patch.object(incumbent, "write", write_panel),
        patch.object(incumbent.worker, "scorer", scorer),
    ):
        incumbent.run(folder, split)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("carrier", choices=study.FORMATS)
    p.add_argument("split", choices=("development", "held"))
    a = p.parse_args()
    run(a.folder, a.carrier, a.split)


if __name__ == "__main__":
    main()
