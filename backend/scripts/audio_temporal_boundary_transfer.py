# SPDX-License-Identifier: MIT
"""Compare an explicit temporal boundary cue on short voice edits."""

import argparse
import json
from pathlib import Path

from scripts import audio_competing_speech_ranking as transfer
from scripts import audio_short_edit_detector_training as detector
from scripts import audio_short_edit_noise_projection as projection
from scripts import audio_short_edit_ssl_transfer as frozen

SOURCE = detector.SOURCE
SCHEMA = "chipmunk.audio-temporal-boundary/1"
TRANSFER_SCHEMA = "chipmunk.audio-temporal-boundary-transfer/1"
METHOD = frozen.EXPERIMENT | {
    "edge_input": "Reuse the content pathway and total parameter count. The boundary output reads the absolute "
                  "hidden feature difference between 40 ms before and 40 ms after each point, with edge hold. "
                  "The original one-channel content output still reads the undifferenced state.",
    "controlled_comparison": "Keep identical initialization, losses, fit/calibration/held roles, gates and "
                             "proposal scoring. Change only the input to the learned boundary output.",
}


def boundary_model():
    import torch
    from torch import nn
    from torch.nn import functional as F

    class BoundaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            network = detector.model()
            self.features = network[:-1]
            self.readout = network[-1]

        def forward(self, values):
            features = self.features(values)
            padded = F.pad(features, (4, 4), mode="replicate")
            transition = (padded[..., 8:] - padded[..., :-8]).abs()
            content = F.conv1d(features, self.readout.weight[:1], self.readout.bias[:1])
            edge = F.conv1d(transition, self.readout.weight[1:], self.readout.bias[1:])
            return torch.cat((content, edge), dim=1)

    return BoundaryModel()


def summary(baseline, prior, candidate):
    plan = baseline["sources"]
    prior_lock = prior["freeze"]
    lock = candidate["freeze"]
    SOURCE.check({key: lock[key] for key in ("source_sha256", "seed", "epochs", "speech_sha256")}
                 == {key: prior_lock[key] for key in ("source_sha256", "seed", "epochs", "speech_sha256")}
                 and all(lock["encoder"].get(key) == value for key, value in prior_lock["encoder"].items())
                 and candidate["feature_ledger_sha256"] == prior["feature_ledger_sha256"],
                 "The comparison must use the same frozen features and source roles")
    checks = {"prior": prior, "temporal_boundary": candidate}
    reports = {
        "prior": detector.result(baseline | {"candidate": prior}, schema=frozen.SCHEMA, method=frozen.EXPERIMENT),
        "temporal_boundary": detector.result(baseline | {"candidate": candidate}, schema=SCHEMA, method=METHOD),
    }
    clocks, budgets = projection.candidate_collisions(plan, checks)
    return {"counts": reports, "matched_edit_clocks": clocks, "equal_alarm_budgets": budgets}


def fit_specs(document):
    lock = document["freeze"]
    cases = document["cases"]
    SOURCE.check(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["speech_sha256"] == frozen.SPEECH
                 and len(cases) == len(detector.sources.FAMILIES)
                 and {(case["trained_family"], case["held_family"]) for case in cases}
                 == {detector.sources.FAMILIES, detector.sources.FAMILIES[::-1]}, "Fitted boundary method differs")
    return {case["trained_family"]: case["points"] for case in cases}


def score_transfer(args):
    import numpy as np
    from app.detectors.audio.upgraded import SpeechActivity

    torch = projection.runtime(3)
    trained = args.fit / "candidate.json"
    document = SOURCE.read(trained)
    specs = fit_specs(document)
    plan, digest = detector.load_plan(args.data)
    SOURCE.check(document["freeze"]["source_sha256"] == digest and SOURCE.path_sha(args.speech) == frozen.SPEECH,
                 "Transfer source or model assets differ")
    heads = {}
    for case in document["cases"]:
        family = case["trained_family"]
        rows = detector.relevant(plan, "calibration", family, calibration=True)
        SOURCE.check(specs[family] == detector.scores_at_points(rows, case["calibration_scores"]),
                     "Sealed boundary operating points differ")
        heads[family] = frozen.load_head(args.fit, case, model_factory=boundary_model)
    activity, encoder = SpeechActivity(args.speech), frozen.frozen_encoder(args.encoder)
    records = []
    with torch.inference_mode():
        for row, signal in transfer.transfer_cases(args.data, args.rooms, args.stock):
            audio = signal.astype(np.float32) / 32768.
            features, speech = encoder(audio), activity.spans(audio)
            readings = {}
            for family, (head, (mean, scale)) in heads.items():
                normalized = (features - mean[:, None]) / scale[:, None]
                local, edge = head(torch.from_numpy(normalized[None]))[0].numpy()
                readings[family] = detector.best_interval(local, edge, speech)
            records.append(row | {"heads": readings})
            if len(records) % 24 == 0:
                print(f"Verified and scored {len(records)} transfer recordings.", flush=True)
    SOURCE.write(args.output, {"schema": TRANSFER_SCHEMA, "fit_sha256": SOURCE.path_sha(trained), "points": specs,
                               "rows": records, "results": transfer.transfer_scores(records, specs)})


def review_transfer(record, candidate_path):
    specs = fit_specs(SOURCE.read(candidate_path))
    SOURCE.check(record["schema"] == TRANSFER_SCHEMA and record["fit_sha256"] == SOURCE.path_sha(candidate_path)
                 and specs == record["points"], "Transfer fit or operating points differ")
    checked = transfer.transfer_scores(record["rows"], specs)
    SOURCE.check(checked == record["results"], "Transfer counts differ")
    return checked


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="phase", required=True)
    for phase, fields in (("fit", ("data", "cache", "speech", "output")),
                          ("transfer", ("data", "rooms", "stock", "encoder", "speech", "fit", "output")),
                          ("review", ("baseline", "prior", "candidate")),
                          ("review-transfer", ("record", "candidate"))):
        command = commands.add_parser(phase)
        for field in fields:
            command.add_argument(field, type=Path)
    options = cli.parse_args()
    if options.phase == "fit":
        frozen.train(options, model_factory=boundary_model, schema=SCHEMA, method=METHOD)
    elif options.phase == "transfer":
        score_transfer(options)
    else:
        result = (summary(SOURCE.read(options.baseline), SOURCE.read(options.prior), SOURCE.read(options.candidate))
                  if options.phase == "review" else review_transfer(SOURCE.read(options.record), options.candidate))
        print(json.dumps(result, indent=2, sort_keys=True))
