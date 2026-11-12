# SPDX-License-Identifier: Apache-2.0
"""Run the exact production Full face stages without a database or billing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNOWN_PAID = Path("/private/tmp/veritas-similar-cut-app-2026-11-12/terminal-receipt.json")
KNOWN_PAID_SHA256 = "870bc1e5d0f40cfdf67b8c37897a7bc11bb707c0774ac8a84f32da2538808ab1"
KNOWN_FILES = ("g4-cross_cut_b.mp4", "g7-cross_cut_b.mp4")
PRIOR_FILES = ("pair-00-0-long_swap.mp4", "pair-00-0-original.mp4")
ALLOW = {"vid.faces.yunet", "vid.tracks.yunet_dinov2", "vid.review.face_appearance"}
PARITY_RESPONSE_TOLERANCE = 2e-6


def configure(folder):
    state = (folder / "direct-stage-state").resolve()
    state.mkdir(mode=0o700, exist_ok=True)
    os.environ.update(VERITAS_DATA_DIR=str(state), VERITAS_DATABASE_URL="sqlite:///:memory:",
                      VERITAS_MODELS_DIR=str(ROOT / "models"), VERITAS_HF_HOME=str(ROOT / "models/hf"),
                      VERITAS_OFFLINE="true", VERITAS_DEVICE="cpu", VERITAS_RUN_WORKER="false",
                      VERITAS_VIDEO_FACE_REVIEW_MODE="appearance", VERITAS_VIDEO_FACE_BOUNDARY_REFINEMENT="true",
                      VERITAS_VIDEO_WHOLE_FACE_REVIEW="false", VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW="true",
                      VERITAS_LOG_LEVEL="WARNING", MPLCONFIGDIR=str(state / "matplotlib"))
    return state


def scorer(state):
    import cv2
    import structlog
    import torch

    from app.core.config import settings
    from app.engine.boundary import Deadline, InMemoryRunStore
    from app.engine.detector import Profile
    from app.engine.exhibits import ExhibitWriter
    from app.engine.media import StoredMedia
    from app.engine.models import get_model_manager
    from app.engine.registry import get_registry
    from app.engine.runner import StageContext
    from app.engine.stages import video as stages
    from app.storage.objects import ObjectStore
    from scripts import replay_video_native_background_panel as prior

    if settings.data_dir.resolve() != state or settings.database_url != "sqlite:///:memory:" or settings.run_worker:
        raise RuntimeError("The direct comparator must be configured without a shared database or worker")

    class NoDbContext(StageContext):
        def emit(self, type, payload=None):
            return 0

    class MemoryRuns(InMemoryRunStore):
        def run_id(self, analysis_id, detector_id):
            return None

    def forbidden_db():
        raise RuntimeError("The direct stage replay must never request a DB session")

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    enabled = registry.enabled
    if not ALLOW.issubset(enabled("video", "full")):
        raise RuntimeError("The qualified production detector set is unavailable")
    registry.enabled = lambda modality, profile: [key for key in enabled(modality,profile) if key in ALLOW]
    models = get_model_manager()
    store = ObjectStore(state / "objects")

    def measure(path, receipt):
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hash != receipt["sha256"]:
            raise ValueError("Delivery differs from its source binding")
        analysis_id = source_hash[:32]
        context = {"registry":registry,"models":models,"run_store":MemoryRuns(),"kind":"video",
                   "media":StoredMedia(source_hash,"video",store,path=path)}
        outputs = {"ingest":{"sha256":source_hash,"filename":path.name}}
        writer = ExhibitWriter(analysis_id,store)
        deadline = Deadline(180)
        logger = structlog.get_logger().bind(scope="native-background-challenge")
        for name in ("probe","decode","shots","tracks","faces","detect"):
            ctx = NoDbContext(analysis_id,name,Profile.FULL,deadline,store,forbidden_db,outputs,context,writer,logger)
            outputs[name] = getattr(stages,name)(ctx)
        if outputs["probe"]["frame_table"] != receipt["clock"] or outputs["shots"]["frames_scanned"] != 120:
            raise ValueError("Production delivery timing does not match the verified native clock")
        [appearance] = [r for r in outputs["detect"]["detectors"] if r["detector_id"]=="vid.review.face_appearance"]
        [tracking] = [r for r in outputs["tracks"]["detectors"] if r["detector_id"]=="vid.tracks.yunet_dinov2"]
        details = appearance["details"]
        coarse = prior.compact(details.get("face_sequence"))
        for frame in coarse["frame_inputs"] if coarse else []:
            index = frame["frame_index"]
            if (frame["pts"] != receipt["clock"]["pts"][index] or frame["decoded_rgb_sha256"] != receipt["rgb_sha256"][index]):
                raise ValueError("The actual incumbent input is not bound to the exact delivered picture")
        return {"shot_boundaries":[r["frame_index"] for r in outputs["shots"]["boundaries"]],
                "track_status":tracking["status"],"tracks":len(outputs["tracks"]["tracks"]),
                "coarse_status":details.get("appearance_check_status"),"appearance_status":appearance["status"],"coarse":coarse,
                "boundary_refinement":prior.scope_range(details.get("face_boundary")),
                "short_range":prior.scope_range(details.get("face_return")),"short_coverage":details.get("face_return_check")}
    return measure


def parity(folder, known, measure, panel):
    from scripts import replay_video_native_background_panel as prior

    common,require = panel.common,panel.require
    require(common.digest(known)==KNOWN_PAID_SHA256,"The exact historical paid parity receipt is required")
    known_result=common.read(known)["rows"]
    known_root=panel.earlier.EARLIER
    known_manifest={r["file"]:r for r in common.read(known_root/"media.json")["media"]}
    previous={r["file"]:r for r in (json.loads(line) for line in (panel.PRIOR_PANEL/"private-incumbent.jsonl").read_text().splitlines())}
    prior_manifest={r["file"]:r for r in common.read(panel.PRIOR_PANEL/"media.json")["media"]}
    assertions=[]

    def check(actual, expected, role):
        got, saved = actual["coarse"], expected["coarse"]
        delta = abs(got["response"]-saved["response"]) if got and saved else 0.
        def native_binding(value):
            return {k:v for k,v in value.items() if k!="response"} if value else None
        differing = [k for k,v in expected.items() if k!="coarse" and actual[k]!=v]
        require(not differing and native_binding(got)==native_binding(saved) and delta <= PARITY_RESPONSE_TOLERANCE,
                f"Direct stages diverged for {role}: {differing}; raw response delta {delta}")
        return {"role":role,"same_decisions_and_native_bindings":True,"absolute_response_delta":delta}

    for name in KNOWN_FILES:
        row=known_manifest[name]
        [ref]=[r for r in known_result if r["input_sha256"]==row["sha256"]]
        result=measure(known_root/"media"/name,row)
        expected = {"coarse":prior.compact(ref["coarse_range"]),"shot_boundaries":ref["shot_boundary_indices"],
                    "track_status":ref["track_status"],"tracks":ref["tracks"],"appearance_status":ref["appearance_status"],
                    "short_coverage":ref["short_coverage"],"short_range":prior.scope_range(ref["dense_range"])}
        assertions.append(check(result,expected,"historical genuine cut with tracked false range"))
    for name in PRIOR_FILES:
        row=prior_manifest[name]
        ref=previous[name]
        result=measure(panel.PRIOR_PANEL/"media"/name,row)
        assertions.append(check(result,{k:ref[k] for k in result},"historical long swap" if row["truth"] else "historical genuine original"))
    return {"status":"verified","n":len(assertions),"rows":assertions,"historical_paid_receipt_sha256":KNOWN_PAID_SHA256,
            "maximum_allowed_raw_response_delta":PARITY_RESPONSE_TOLERANCE}


def run(folder,archive,known,parity_only):
    state=configure(folder)
    measure=scorer(state)
    from scripts import replay_video_native_background_panel as prior
    from scripts import video_native_background_challenge as panel
    from scripts import video_native_background_edges as guard

    common,require=panel.common,panel.require
    lock=panel.frozen(folder,archive or panel.ARCHIVE)
    paths={name:common.digest(prior.APP/name) for name in prior.BINDINGS}
    candidate=common.digest(Path(guard.__file__))
    require(candidate==lock["spec"]["candidate_sha256"],"Candidate no longer matches the prospective freeze")
    worker={"worker_sha256":common.digest(Path(__file__)),"source_freeze_sha256":common.digest(folder/"freeze.json"),
            "app_files":paths,"candidate_sha256":candidate,"stage_order":["probe","decode","shots","tracks","faces","detect"],
            "mode":"CPU Full with in-memory detector receipts and no database session"}
    gate=parity(folder,known,measure,panel)
    require(gate["n"]==4,"Four exact parity cases required before fresh scores")
    common.write(folder/"direct-stage-parity.json",gate)
    common.write(folder/"screen-freeze.json",worker)
    print("verified four exact Full stage parity cases",flush=True)
    if parity_only:
        return
    manifest=common.read(folder/"media.json")
    require(manifest["freeze_sha256"]==worker["source_freeze_sha256"] and len(manifest["media"])==108,"Held media roster differs")
    journal=folder/"private-incumbent.jsonl"
    start=time.monotonic()
    false_pairs=set()
    with journal.open("x") as stream:
        for number,row in enumerate(manifest["media"],1):
            path=folder/"media"/row["file"]
            result=measure(path,row)
            record={**{key:row[key] for key in ("file","sha256","condition","source_pair","direction","interval","truth","genuine_cut")},**result}
            if result["coarse"] and row["genuine_cut"]:
                false_pairs.add(row["source_pair"])
            stream.write(json.dumps(record,allow_nan=False,sort_keys=True)+"\n")
            stream.flush()
            if number%9==0:
                print("screened actual Full face stages",number,"of",len(manifest["media"]),flush=True)
    require(paths=={name:common.digest(prior.APP/name) for name in prior.BINDINGS},"Production code changed during the direct comparator")
    common.write(folder/"incumbent-seal.json",{"status":"verified","cases":len(manifest["media"]),
                 "journal_sha256":common.digest(journal),"candidate_challenge_source_pairs":sorted(false_pairs),
                 "media_sha256":common.digest(folder/"media.json"),"screen_freeze_sha256":common.digest(folder/"screen-freeze.json"),
                 "elapsed_s":time.monotonic()-start})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path)
    parser.add_argument("--known-paid",type=Path,default=KNOWN_PAID)
    parser.add_argument("--parity-only",action="store_true")
    args=parser.parse_args()
    run(args.folder,args.archive,args.known_paid,args.parity_only)


if __name__=="__main__":
    main()
