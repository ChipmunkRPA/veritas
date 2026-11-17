"""Replay whole-interval speech scoring on exact matched original utterances."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import audio_focused_context_transfer as files
from scripts import audio_interval_ranking_review as review
from scripts import audio_long_context_review as reference
from scripts.audio_long_context_fit import file_sha


def assess(bank, media, cache, lock, destination):
    import numpy as np
    import torch

    from app.detectors.audio import interval_features as intervals
    from scripts import audio_interval_ranking_fit as fitting
    from scripts import audio_long_context_transfer as previous

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fit = files.read(lock)
    if fit["features_sha256"] != file_sha(Path(intervals.__file__)) or fit["code_sha256"] != file_sha(Path(fitting.__file__)):
        raise ValueError("The comparison requires the same frozen interval model and features")
    _, heads = previous.fitted_heads(bank / "fit")
    prior_file = bank / "prospective-result.json.gz"
    prior = files.read(prior_file)["calibration"]["points"]["short"]
    plan_path = media / "freeze.json"
    plan = files.read(plan_path)
    plan_digest = file_sha(plan_path)
    if files.read(cache / "freeze.json")["source_sha256"] != plan_digest:
        raise ValueError("The cached features use a different matched source plan")
    journal = files.read(cache / "files.json")
    rows = []
    for case in plan["cases"]:
        digest = case["pcm_sha256"]
        directory, record = cache / digest, journal[digest]
        feature_file, speech_file = directory / "short.npy", directory / "speech.json"
        if file_sha(feature_file) != record["features"]["short"] or file_sha(speech_file) != record["speech_sha256"]:
            raise ValueError("A speech or feature input changed")
        previous.official_input(media, case)
        features = np.load(feature_file, allow_pickle=False)
        activity = files.read(speech_file)
        baseline = previous.common_reading(heads["short"], features, 0, activity)
        offset, width = record["views"]["short"]
        current = [list(span) for span in reference.recalled(baseline, prior, "wider", offset)]
        grid = intervals.features(features)
        readings = {name: intervals.readings(grid, head, activity) for name, head in fit["heads"].items()}
        rows.append(case | {"scope": [offset, offset + width], "current": current, "current_assessed": any(baseline),
                            "readings": readings, "feature_sha256": record["features"]["short"]})
    result = {"schema": review.SCHEMA, "design": fit["design"], "fit": fit, "rows": rows,
              "summary": review.summarize(rows, fit["points"]), "fit_sha256": file_sha(lock), "source_plan_sha256": plan_digest,
              "comparator": "previous calibrated six-second head alone, wider point fixed across both new points"}
    review.verify(result)
    files.save(destination, result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "media", "cache", "lock", "destination"):
        cli.add_argument(name, type=Path)
    assess(**vars(cli.parse_args()))
