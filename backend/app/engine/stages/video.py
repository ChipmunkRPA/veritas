"""Video pipeline stages (ARCHITECTURE §3.1–3.2):
ingest → probe → provenance → decode → shots → tracks → faces → detect → fuse → synthesise → render → publish.

- **probe** runs ffprobe (`video/frames.probe`), builds the per-frame pts table (`video/decode.probe_frames`) and fixes
  the seeded sampling plan (`video/sampling.build_plan`, V-1). Its output is persisted, so resumed runs see identical frames.
- **decode** decodes the plan's legacy frame set (the v0.3 count: 4 per second within [16, 40], 16 in quick) pts-accurately
  through a bounded `FrameStore` and extracts the sound track (16 kHz mono). It is not persisted: frames and PCM are
  re-derived from the plan on resume.
- **shots** (full, deep) finds shot boundaries on the ≤ 640 px stream and rebuilds the plan with clips per shot; **tracks**
  (full, deep) runs the face-tracking plugin (YuNet on every 2nd low-resolution frame, DINOv2-small appearance, identity
  letters) and summarises the tracks. Both are persisted (V-2).
- **faces** runs the face-detection plugin on every sampled frame for the v0.3 face detectors; **detect** runs the video
  and audio plugins, with the per-track inputs (`TrackInputs`: tracks, face_track frames, the sparse sampler and the
  `FrameStore`) for the face-swap lane (V-3), the per-shot clips (`ClipInputs`) for the synthesis lane (V-4) and the
  tracks with the sound track (`AVInputs`) for the lip-sync lane (V-5).
- **fuse** is the legacy fuser: per-frame fusion (`image/scoring.fuse_frame_scores`) with the v0.3 video weight
  overrides, then `video/pipeline.assemble_video`, published as an uncalibrated fallback `Decision` (§4.4). Per-track
  detectors add one track-scoped signal per scored track, and synthesis-lane detectors one global signal (not averaged).
- **temporal** (V-7) turns the detect stage's window scores into smoothed lane series and `TimeSegment`s per lane and
  track (`video/segments.py`: median over 3, hysteresis 0.60 / 0.45, minimum 0.5 s), with a series exhibit per lane and
  track and a segments exhibit where something is flagged; it also lists the lip-sync lane's own segments.
- **second_pass** (deep) re-samples every frame of the track around the top-3 face-lane segments (padded by 0.5 s, at most
  64 frames per segment), re-runs the per-track face detectors on those frames only and segments the dense windows again.
  The V-2 tracker already covers every second low-resolution frame, so faces and tracks are reused, not re-run.
- **render** writes the v1 artifact files (timeline, contact sheet, poster, key frames); **publish** stores the v1 view.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import io
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.canonical import canonical_hash
from app.core.config import settings
from app.detectors.audio.coverage import coverage_seconds, scored_voice_spans
from app.detectors.audio.decode import VIDEO_AUDIO_MAX_SECONDS, decode_audio
from app.detectors.common.fusion import prob_to_signal_fields
from app.detectors.image.classifiers import specs as classifier_specs
from app.detectors.image.scoring import dfdc_frame_weight, face_frame_weight, frame_weight, fuse_frame_scores
from app.detectors.video import picture_hold, sampling, scene_return, scene_return_carrier
from app.detectors.video import segments as temporal_segments
from app.detectors.video.adapters import lipsync as lipsync_lane
from app.detectors.video.adapters import lipsync_xcorr as xcorr_lane
from app.detectors.video.adapters import swap as swap_lane
from app.detectors.video.adapters.codec import CodecInputs
from app.detectors.video.adapters.common import AudioTrack, SampledFrame
from app.detectors.video.adapters.face_appearance import DETECTOR as APPEARANCE_REVIEW
from app.detectors.video.adapters.face_sequence import DETECTOR as SEQUENCE_REVIEW
from app.detectors.video.adapters.face_sequence import FAMILY as SEQUENCE_FAMILY
from app.detectors.video.adapters.face_sequence import SequenceInputs
from app.detectors.video.adapters.lipsync import AVInputs
from app.detectors.video.adapters.swap import TrackInputs
from app.detectors.video.adapters.synthesis import ClipInputs
from app.detectors.video.adapters.tracks import LowResVideo
from app.detectors.video.adapters.whole_face import DETECTOR as WHOLE_FACE_REVIEW
from app.detectors.video.adapters.whole_face import FAMILY as WHOLE_FACE_FAMILY
from app.detectors.video.audio import SR, is_mostly_silence, limit_voice_signal, silent_signal, voice_signal
from app.detectors.video.audio import specs as audio_specs
from app.detectors.video.decode import DecodeError, FrameTable, decode_stream, probe_frames
from app.detectors.video.face_return import source_carrier
from app.detectors.video.frame_store import FrameStore
from app.detectors.video.frames import probe as ffprobe
from app.detectors.video.pipeline import (
    VideoScores,
    assemble_video,
    attach_artifacts,
    frame_count,
    video_artifacts,
)
from app.detectors.video.shots import METHOD as SHOT_METHOD
from app.detectors.video.shots import ShotDetector, detect_shots
from app.engine import card_copy, provenance_gate
from app.engine.boundary import DetectorTimeout
from app.engine.detector import Lane, Modality, Profile, StageOutputs
from app.engine.exhibits import exhibit_payload_hash
from app.engine.fusion import abstain, decisive
from app.engine.fusion.runtime import DECLARED_AI_VERBAL
from app.engine.profiles import SAMPLING
from app.engine.runner import StageContext
from app.engine.stages import common
from app.evidence import (
    face_return_binding,
    face_return_coverage,
    face_short_window_binding,
    face_spatial_cut_binding,
    picture_hold_binding,
    scene_return_binding,
)
from app.evidence.audio_binding import (
    VOICE_CLOCK_LIMIT,
    VOICE_CLOCK_POLICY,
    exclude_recording_voice,
    limit_voice_screening,
    source_clock_breaks,
)
from app.evidence.face_boundary_binding import is_paired
from app.evidence.face_boundary_binding import source_inputs as face_boundary_source_inputs
from app.evidence.face_interval_binding import sequence_observation
from app.schemas.evidence import AnalysisResultV2, Decision, SignalV2, TimeSegment
from app.schemas.legacy import BAND_CODE, FALLBACK_CAVEAT

# p37: freeze decoded voice PCM and actual AV correlation inputs for immutable nested evidence.
# p36: exact source-frame commitments and consistent legacy exhibit time units.
# Split voice windows at material source-clock gaps and overlaps.
# p51: measure continuous soundtrack around an HD mouth hold even when the rest of the clock jumps.
# p50: optional evidence-only bounded face proposal for qualified four-second H.264 MP4.
# p49: record source-aligned acoustic periodicity during a qualified HD mouth-region hold.
# p48: optional native timing refinement of short face appearance ranges.
PIPELINE_VERSION = "vid-2.0-p52"  # p52: source-bound timing window coverage; p35 (N-19q, finding 4; synthesis tpl-2.19): the finding on the checks that ran says of the Content Credentials check of the container that read the file and found no manifest that it ran and found none (it read "Not run: C2PA Content Credentials (video containers) (not applicable: …)" beside "No Content Credentials"); the earlier method's per-frame series under a fusion model is titled "Per-frame values of the earlier method (the estimate does not read them)" (finding 6); the finding of a check that entered the estimate stores its stand-alone words. p34 (N-19p, design review D6-05; synthesis tpl-2.18): the coverage finding counts the statistical checks only ("1 of 1 scored check flagged this file" beside "Only one statistical check scored"; it read "1 of 2", counting the codec lane). p33 (N-19p, security re-check SEC6-01, code review R10-02, R10-03): a v0.3 signal's `observed`, `reasoning` and the strings of its `details` are filtered as text from the file (a container tag held RLO, ESC and BEL in the signal of the capture-device tags, on both fusion paths), text from a file keeps its words apart, and the title and caption of every exhibit are filtered (`prov.c2pa.video` 0.4.3, `vid.meta.container` 0.3.1, `vid.codec.ffprobe` 0.4.1). p32 (N-19o, design review D5-03): outside what was validated, where no estimate is shown, the finding of a check of little effect is titled "Little effect" and a check that entered says "In this result …"; (D5-04): the finding of the container check, whose row the decision publishes and does not count, is titled by what it tests and says that the decision does not count it; its strength, direction and every count are unchanged (it read "Little effect on the estimate" and "In this estimate the check had little effect."). p31 (N-19n, what the third part of N-19m left; supervisor check 29): under a configured prior the reason of a result that D-77 took names the other answer's part of the scale where its published estimate prints outside 35 to 64. p30 (N-19n, the marker’s finding, one step for the part; design review D3-07): the finding of a self-declared marker is a record’s (its title "Self-declared: …", "Self-declared, not applied: …" under a record that outranks it); Content Credentials that do not validate are titled as such; the caveat of what was set aside names it without a reason; a signal copies what its check found and where (`details`). p29: the v1 summary of an inconclusive result outside what was validated says so also where the cap changed no code (N-19n). p28: a decisive code of the statistical checks needs the estimate as published in its answer’s part of the scale (D-77) and checks that together give it at least ln 2 (D-76 as amended); a result that fails one is inconclusive with that condition’s reason, and `fusion.rule` and `statistical_alone.rule` name it; the reason’s number is the printed whole points; a cap’s reason quotes the levels’ headline only where both conditions would have left it (N-19m, third part). p27: under a self-declared marker the confidence of a result is low where D-76 took the decisive code of the statistical estimate, as it is for a check that did not run, for disagreement and for a wide interval (it was medium) (N-19m; code review R7-03, D-76 as amended on 2026-10-05); p26: the decision of a result that a trusted capture decides is headed by the record ("The file's Content Credentials record a capture of these bytes" in `verbal`, on both fusion paths; it read the sentence of the decision code); its code, number, band and confidence are unchanged (N-19m; design review D3-01, D-15); p25: the row of a self-declared record in the breakdown names the kind of the record (Content Credentials by what they name or declare; it read "Generator named in unsigned metadata or a public watermark") (N-19l; code review R6-04); p24: the exhibit of the combined scores is titled "(hand-set weights)", not by the build's name "(legacy fuser)"; the checks that could not run on this server are one caveat with their names, in place of one caveat for each and of their ids in the caveat of a reduced profile; a caveat names a check as the page does, a step of the run and a provenance record in words, never a detector id, a stage name or the value of a kind (browser QA leads 3 and 4, design re-review D2-06); each check that scored and is not counted in the estimate carries the fusion's record of why (`outside_estimate`: not in the model, its gate closed, a weight of zero; browser QA Q2-12); a result of the fallback on an instance that holds a fusion model says that none was used for the analysis, not that none is loaded, and the v1 view of a result outside what was validated gives the reason of its cap (N-19k; browser QA Q2-14 and part 2); p23: a decisive result needs a check that supports it (D-76): a result that the statistical checks would decide inside the envelope without a check whose published row is at least ln 2 toward its answer is inconclusive, and its reason says so; each row of a check says whether the decision counts it (`counted`: not a row of zero, not the row of a check that declares H3: the container check, which the number line counted and the runtime did not); p22: a result that a provenance record decides names its record (`provenance_record`) and words a marker by its kind; the caveat of the statistical estimate before the override prints the clamped number, the caveats of an interval stand only with one, and outside what was validated that caveat says that the number is not shown; the fallback publishes its statistical score beside a record where a check of the content scored and none where records alone scored, and a fallback result that a record decides outside what was validated says so in its reason; the reason of a marker that decides alone ends with the sentence of its case (N-19h, E-19 … E-21, 2026-10-04); p21: a check counts for a strong result, for "k of n detectors" and for "no check" when its published row is not zero; where every check ran or was not applicable and none that scored counts, the result is not assessable in words of its own, and a provenance record over such a result decides alone (N-19f, D-64 as clarified, code review R4-05, 2026-10-04), and a fallback result whose condition stage failed is capped at inconclusive (N-19f, R4-03); p20: a strong code needs two checks that scored, and a fallback result in which only container or provenance records scored is not assessable (N-18, D-64, 2026-10-03); p19: a not-assessable result has no caveat or reason that speaks of an estimate or a probability, and a lane series is a "window score" (N-17, D-68); p18: a composite signal of one detector (the face-swap check, the voice check) is that detector's check and takes its id, so the fusion model's contribution reaches it; the per-frame series of the earlier method is titled as outside the estimate under a fusion model (N-17, D-63; browser QA P1-5); p17: a finding that entered the estimate is worded by its contribution (N-17, D-63, 2026-10-03); p16: contributions are measured from each check's neutral point (N-17); p15: distinguish unsupported C2PA containers from absent credentials (2026-09-20)
FUSER_ID = "legacy.video"
C2PA_DETECTOR = "prov.c2pa.video"
H3 = "H3_provenance_integrity"
ENTRY_STAGES = ("provenance", "tracks", "faces", "detect")
EXHIBIT_STAGES = ("provenance", "tracks", "faces", "detect", "temporal", "second_pass", "fuse", "render")
VIDEO_MAX_UNIT_SCORES = 5000  # window units feed the temporal stage: 240 crops per track × tracks, ≤ 60 voice windows
STILL_MAX_PX = 1920  # stills keep the scored pixels up to 1080p; larger frames are downscaled to bound the §4.3 budget
STILL_QUALITY = 90  # §4.3: JPEG q90
SIGNAL_STAGES = ("provenance", "detect")
METADATA_DETECTOR = "vid.meta.container"
GENERAL_DETECTORS = (("vid.frames.univfd", "univfd"), ("vid.frames.hf.haywoodsloan", "ai-image-detector-deploy"),
                     ("vid.frames.hf.ateeqq", "ai-vs-human-image-detector"))  # v0.3 score keys, in v0.3 order
FACE_CLASSIFIERS = (("vid.frames.hf.dima806", "deepfake_vs_real_image_detection"), ("vid.frames.hf.wvolf", "ViT_Deepfake_Detection"))
DFDC_DETECTOR, DFDC_KEY = "vid.frames.dfdc_b7", "dfdc_effnet_b7"
NOISE_DETECTOR = "vid.forensic.face_noise"
JITTER_DETECTOR, FLICKER_DETECTOR = "vid.temporal.landmark_jitter", "vid.temporal.face_flicker"
VOICE_DETECTORS = (("aud.gustking_xlsr", "wav2vec2-large-xlsr-deepfake-audio-classification"),
                   ("aud.mo_thecreator_w2v2", "Deepfake-audio-detection"))
SOURCES = {"video_device_tags": METADATA_DETECTOR, "video_encoder": METADATA_DETECTOR, "landmark_jitter": JITTER_DETECTOR,
           "face_flicker": FLICKER_DETECTOR, "noise_face_bg_video": NOISE_DETECTOR}  # v0.3 signals with a single source
HYPOTHESIS = {"video_device_tags": "H3_provenance_integrity", "video_encoder": "H3_provenance_integrity",
              "frames_general": "H1_whole_content_synthesis"}  # every other video signal: H2
MODALITY = {"audio_synthetic_voice": "audio", "audio_silent": "audio"}
FINDING_FAMILY = {"video_device_tags": "V4_codec", "video_encoder": "V4_codec", "frames_general": "V3_synthetic_video",
                  "audio_synthetic_voice": "A1_voice", "audio_silent": "A1_voice"}  # every other video signal: V1_face_track
SERIES_SIGNALS = ("frames_general", "frames_face", "score_volatility")
# N-19k: titled by what the scores are; the title and the caption named the build's "legacy fuser" in print
SERIES_CAPTION = "Per-frame fused p(AI) from hand-set weights: uncalibrated, not a frequency estimate."
# N-17 (browser QA P1-5 a): under a fusion model the headline does not come from these per-frame scores
# N-19q (finding 6): its values, not the check that the finding cites, are what the estimate does not read
SERIES_TITLE, SERIES_TITLE_OUTSIDE = "Per-frame scores (hand-set weights)", "Per-frame values of the earlier method (the estimate does not read them)"
SERIES_CAPTION_OUTSIDE = ("Per-frame p(AI) of the earlier per-frame method, uncalibrated. The fusion model that gives this "
                          "result's estimate does not read these values: they are shown for inspection only.")
# N-17 (browser QA P1-4, P1-5): the v0.3 signals that average several detectors, and the detectors each one averages
COMPOSITES = {"frames_face": (*(d for d, _ in FACE_CLASSIFIERS), DFDC_DETECTOR), "audio_synthetic_voice": tuple(d for d, _ in VOICE_DETECTORS)}


# --- ingest-side stages --------------------------------------------------------------------------------------

def probe(ctx: StageContext) -> dict[str, Any]:
    """Stream facts, the per-frame pts table and the seeded sampling plan. The output is persisted, so a resumed run decodes
    the same frames (§3.3). A file without decodable video frames gets no table, and `decode` fails the analysis."""
    path = common.media_path(ctx)
    codec_tag = (settings.video_face_paired_edge_refinement or settings.video_face_unrouted_short_review
                 or settings.video_face_spatial_cut_context or settings.video_face_short_window_review
                 or settings.video_face_short_carrier_review or settings.video_face_short_avc_review
                 or settings.video_scene_return_review)
    info = common.json_safe(ffprobe(path, include_codec_tag=True) if codec_tag else ffprobe(path))
    try:
        table = probe_frames(path, check=ctx.check)
    except DecodeError as e:
        ctx.log.warning("video.frame_table_unavailable", error=str(e)[:200])
        return {"info": info, "frame_table": None, "plan": None}
    legacy = frame_count(info.get("duration") or 0)
    cap = SAMPLING[ctx.profile].uniform_seek_frames
    plan = sampling.build_plan(table, ctx.profile, seed=sampling.plan_seed(ctx.outputs["ingest"]["sha256"]),
                               legacy_frames=min(legacy, cap) if cap else legacy)
    return {"info": info, "frame_table": table.to_json(), "plan": plan}


def current_provenance_output(output: Any, *, c2pa_expected: bool = True) -> bool:
    """Reuse this C2PA version only; absence is valid when the active registry does not schedule it.

    Status is deliberately ignored: version freshness does not change detector retry policy.
    Historical refusion bypasses this freshness predicate and keeps its existing integrity checks.
    """
    from app.detectors.image.adapters.provenance import C2PA_VERSION

    if not isinstance(output, dict) or not isinstance(output.get("detectors"), list):
        return False
    rows = output["detectors"]
    if any(not isinstance(row, dict) for row in rows):
        return False
    matches = [row for row in rows if row.get("detector_id") == "prov.c2pa.video"]
    if not matches:
        return not c2pa_expected
    return len(matches) == 1 and matches[0].get("version") == C2PA_VERSION


def provenance(ctx: StageContext) -> dict[str, Any]:
    return common.run_detectors(ctx, (Modality.VIDEO,), include={"provenance"},
                                outputs=StageOutputs(probe=ctx.outputs["probe"]["info"]))


def _audio(path: Path, has_audio: bool, *, source_sha256: str, capture: Callable | None = None) -> tuple[AudioTrack, dict[str, Any] | None]:
    if not has_audio:
        return AudioTrack("absent"), None
    try:
        decoded = decode_audio(path, source_sha256=source_sha256, max_seconds=VIDEO_AUDIO_MAX_SECONDS,
                               inspect_stereo=capture is not None, retain_stereo=capture is not None)
    except (ValueError, RuntimeError):
        return AudioTrack("unreadable"), None
    pcm, mapping = decoded.pcm, decoded.mapping
    if capture is not None:
        capture(decoded)
        # AV raw proofs commit this exact mono mapping. Freeze stereo separately.
        mapping = {key: value for key, value in mapping.items() if key != "stereo_context"}
    if len(pcm) < SR:
        return AudioTrack("short", None, len(pcm)), mapping
    if is_mostly_silence(pcm):
        return AudioTrack("silent", None, len(pcm)), mapping
    clock_breaks = tuple(item[0] for item in source_clock_breaks(mapping))
    return AudioTrack("analysed", pcm, len(pcm), clock_breaks=clock_breaks), mapping


def decode(ctx: StageContext) -> dict[str, Any]:
    """The plan's legacy frame set, pts-accurate, through a bounded `FrameStore`, and the sound track (16 kHz mono), into
    run state. Not persisted: frames are re-derived from the persisted plan on resume. The JSON output lists each frame's
    pts and pixel sha256, so a resumed run can be checked against the first."""
    path = common.media_path(ctx)
    probed = ctx.outputs["probe"]
    if not probed.get("frame_table") or not probed.get("plan"):
        raise ValueError("Video could not be decoded (no decodable video frames)")
    table = FrameTable.from_json(probed["frame_table"])
    store = FrameStore(path, table, check=ctx.check)
    wanted = probed["plan"]["legacy"]["frames"]
    decoded = store.get_many(wanted)
    frames = [SampledFrame(k, i, table.seconds(i), round(table.seconds(i) * 1000), decoded[i], pts=table.pts[i],
                           time_base=table.time_base) for k, i in enumerate(i for i in wanted if i in decoded)]
    if not frames:
        raise ValueError("No frames could be decoded")
    ctx.check("frame decoding")
    from app.evidence.video_audio import freeze_input

    audio_evidence = {}
    audio, audio_mapping = _audio(path, bool(probed["info"].get("has_audio")), source_sha256=ctx.outputs["ingest"]["sha256"],
                                  capture=lambda decoded: audio_evidence.update(freeze_input(ctx, decoded)))
    ctx.state["frames"], ctx.state["frame_store"], ctx.state["audio"] = frames, store, audio
    ctx.state["video_audio_mapping"] = audio_mapping
    height, width = frames[0].rgb.shape[:2]
    return {"method": "pts-accurate", "time_base": table.time_base,
            "frames": [{"index": f.index, "frame_index": f.frame_index, "time_s": f.time_s, "pts_ms": f.pts_ms, "pts": f.pts,
                        "sha256": hashlib.sha256(f.rgb.tobytes()).hexdigest()} for f in frames],
            "missing_frames": [i for i in wanted if i not in decoded], "width": int(width), "height": int(height),
            "audio": {"status": audio.status, "samples": audio.samples, "seconds": round(audio.samples / SR, 3),
                      "source_mapping": audio_mapping, "evidence_input": audio_evidence.get("input"),
                      "source_clock_break_count": len(audio.clock_breaks),
                      "stereo_context_sha256": audio_evidence.get("stereo_context_sha256")},
            "artefacts": audio_evidence.get("artefacts", [])}


def shots(ctx: StageContext) -> dict[str, Any]:
    """Shot boundaries on the ≤ 640 px stream (`video/shots.py`) and the sampling plan rebuilt with clips per detected shot
    (same seed and legacy set as the probe plan). Persisted."""
    probed = ctx.outputs["probe"]
    enabled = settings.environment == "dev" and settings.video_scene_return_review
    if not probed.get("frame_table") or not probed.get("plan"):
        outcome = {"method": SHOT_METHOD, "boundaries": [], "ranges": [], "plan": None}
        if enabled:
            outcome[scene_return_binding.COVERAGE] = scene_return_binding.coverage(
                "unavailable", reason="unsupported_video", webm_available=True)
        return outcome
    table = FrameTable.from_json(probed["frame_table"])
    path = common.media_path(ctx)
    info = probed["info"]
    profile = (scene_return_carrier.profile(table, (info.get("codec"), info.get("pix_fmt"), info.get("codec_tag")),
                                            source_carrier(path)) if enabled else None)
    stream = decode_stream(path, table, check=ctx.check)
    outcome = {}
    if profile:
        # Score the same low-resolution stream without touching the shot plan or face routing.
        detector = ShotDetector(table)
        verified = 0
        exhausted = False

        def observe():
            nonlocal verified, exhausted
            for frame in stream:
                detector.update(frame.index, frame.rgb)
                if (frame.index == verified < len(table) and frame.pts_verified and frame.pts == table.pts[verified]):
                    verified += 1
                yield frame
            exhausted = True

        measured = scene_return_carrier.detect(observe(), table, profile)
        for frame in stream:
            detector.update(frame.index, frame.rgb)
        first = table.pts[0] if profile == scene_return_binding.WEBM_SCHEMA else None
        if measured is not None:
            outcome[scene_return_binding.KEY] = {"start": measured.start, "end": measured.stop, "tiles": list(measured.tiles),
                                                  **({"schema": profile} if first is not None else {})}
        elif exhausted and verified == len(table):
            outcome[scene_return_binding.COVERAGE] = scene_return_binding.coverage("no_match", frames_scanned=verified, first_pts=first)
        else:
            outcome[scene_return_binding.COVERAGE] = scene_return_binding.coverage(
                "unavailable", reason="incomplete_decode", frames_scanned=verified, first_pts=first)
    else:
        detector = detect_shots(stream, table)
        if enabled:
            outcome[scene_return_binding.COVERAGE] = scene_return_binding.coverage(
                "unavailable", reason="unsupported_video", webm_available=True)
    base = probed["plan"]
    plan = sampling.build_plan(table, ctx.profile, seed=base["seed"], legacy_frames=len(base["legacy"]["frames"]),
                               shots=detector.ranges())
    return {**detector.as_json(), "plan": plan, **outcome}


def tracks(ctx: StageContext) -> dict[str, Any]:
    """Face tracks and appearance clusters from the face-tracking plugin, plus a per-track summary. Persisted."""
    probed = ctx.outputs["probe"]
    if not probed.get("frame_table"):
        return {"detectors": [], "exhibits": [], "artefacts": [], "tracks": []}
    table = FrameTable.from_json(probed["frame_table"])
    ranges = (ctx.outputs.get("shots") or {}).get("ranges") or [[0, len(table)]]
    video = LowResVideo(common.media_path(ctx), table, tuple((int(a), int(b)) for a, b in ranges))
    out = common.run_detectors(ctx, (Modality.VIDEO,), include={"face-tracking"},
                               outputs=StageOutputs(probe=probed["info"], frames=video))
    summary = []
    for exhibit in out["exhibits"]:
        if exhibit["kind"] != "face_track":
            continue
        frames = exhibit["frames"]
        # Face sizes from detected boxes; a track whose boxes carry no detection score falls back to all of its boxes.
        detected = ([min(f["bbox"]["w"], f["bbox"]["h"]) for f in frames if f.get("det_score") is not None]
                    or [min(f["bbox"]["w"], f["bbox"]["h"]) for f in frames])
        summary.append({**exhibit["track"], "exhibit_id": exhibit["id"], "time_base": table.time_base,
                        "start_s": table.seconds(frames[0]["frame_index"]), "end_s": table.seconds(frames[-1]["frame_index"]),
                        "detections": len(detected), "median_face_px": float(np.median(detected)),
                        "max_face_px": float(max(detected))})
    return {**out, "tracks": summary}


def track_inputs(ctx: StageContext) -> TrackInputs | None:
    """The per-track detectors' inputs (V-3): tracks summary, face_track frames, the plan's sparse sampler, the FrameStore."""
    tracked, store = ctx.outputs.get("tracks") or {}, ctx.state.get("frame_store")
    if not tracked.get("tracks") or store is None:
        return None
    plan = (ctx.outputs.get("shots") or {}).get("plan") or ctx.outputs["probe"]["plan"]
    sparse = plan.get("sparse") or {}
    exhibits = {e["id"]: e for e in tracked.get("exhibits", []) if e["kind"] == "face_track"}
    return TrackInputs(table=store.table, tracks=tuple(tracked["tracks"]),
                       frames={t["track_id"]: exhibits[t["exhibit_id"]]["frames"] for t in tracked["tracks"] if t["exhibit_id"] in exhibits},
                       sparse=tuple(sparse.get("frames") or ()), sparse_fps=sparse.get("rate_fps"), store=store)


def clip_inputs(ctx: StageContext) -> ClipInputs | None:
    """The synthesis lane's inputs (V-4): the plan's clips per shot (16 frames at 8 fps) and the FrameStore."""
    store = ctx.state.get("frame_store")
    plan = (ctx.outputs.get("shots") or {}).get("plan") or (ctx.outputs.get("probe") or {}).get("plan")
    if store is None or not plan or not plan.get("clips"):
        return None
    return ClipInputs(table=store.table, clips=tuple(plan["clips"]), shots=len(plan["shots"]["ranges"]),
                      shots_source=plan["shots"]["source"], store=store)


def av_inputs(ctx: StageContext, tracked: TrackInputs | None) -> AVInputs | None:
    """The lip-sync lane's inputs (V-5): the tracks and their face_track frames, the sound track and the FrameStore."""
    if tracked is None:
        store = ctx.state.get("frame_store")
        if store is None:
            return None
        return AVInputs(table=store.table, tracks=(), frames={}, audio=ctx.state.get("audio"),
                        store=store, audio_mapping=ctx.state.get("video_audio_mapping"))
    return AVInputs(table=tracked.table, tracks=tracked.tracks, frames=tracked.frames, audio=ctx.state.get("audio"),
                    store=tracked.store, audio_mapping=ctx.state.get("video_audio_mapping"))


def codec_inputs(ctx: StageContext) -> CodecInputs | None:
    """The codec lane's inputs (V-6): the original file, the frame table, ffprobe facts and the detected shot cuts."""
    probed = ctx.outputs.get("probe") or {}
    if not probed.get("frame_table"):
        return None
    ranges = (ctx.outputs.get("shots") or {}).get("ranges") or []
    return CodecInputs(path=common.media_path(ctx), table=FrameTable.from_json(probed["frame_table"]), info=probed.get("info") or {},
                       cuts=tuple(int(a) for a, _ in ranges[1:]), shots_source="detected" if ranges else "not detected")


def _outputs(ctx: StageContext) -> StageOutputs:
    tracked = track_inputs(ctx)
    return StageOutputs(probe=ctx.outputs["probe"]["info"], faces=(ctx.outputs.get("faces") or {}).get("faces"),
                        frames=ctx.state.get("frames"), audio=ctx.state.get("audio"), tracks=tracked,
                        clips=clip_inputs(ctx), av=av_inputs(ctx, tracked), codec=codec_inputs(ctx))


def faces(ctx: StageContext) -> dict[str, Any]:
    """Face-detection plugins → v0.3 face dicts per sampled frame (largest first)."""
    out = common.run_detectors(ctx, (Modality.VIDEO,), include={"face-detection"}, outputs=_outputs(ctx))
    table = ctx.outputs["decode"]["frames"]
    by_pts = {f["pts_ms"]: f["index"] for f in table}
    per_frame: list[list[dict[str, Any]]] = [[] for _ in table]
    for exhibit in out["exhibits"]:
        if exhibit["kind"] != "regions":
            continue
        for scored in exhibit["regions"]:
            region, index = scored["region"], by_pts.get(scored["region"].get("frame_pts"))
            box = region.get("bbox")
            if index is None or box is None:
                continue
            per_frame[index].append({**{k: int(round(box[k])) for k in ("x", "y", "w", "h")},
                                     "score": round(float(scored.get("score") or 0.0), 3),
                                     "landmarks": [list(p) for p in region.get("landmarks") or []]})
    return {**out, "faces": per_frame}


AUDIO_UNDECODABLE = "the sound track could not be decoded"


def detect(ctx: StageContext) -> dict[str, Any]:
    """D-49: a sound track the probe reports but extraction could not decode marks every detector that needs audio `failed`
    (voice and lip-sync), so an artefact that depends on them caps; absent, short and silent tracks stay `not_applicable`."""
    audio = ctx.state.get("audio")
    audio_failure = AUDIO_UNDECODABLE if audio is not None and audio.status == "unreadable" else None
    outputs = _outputs(ctx)
    out = common.run_detectors(ctx, (Modality.VIDEO, Modality.AUDIO),
                               exclude={"provenance", "face-detection", "face-tracking", SEQUENCE_FAMILY, WHOLE_FACE_FAMILY},
                               outputs=outputs, max_units=VIDEO_MAX_UNIT_SCORES, face_failure=common.face_stage_failure(ctx),
                               audio_failure=audio_failure)
    if not {SEQUENCE_REVIEW, APPEARANCE_REVIEW, WHOLE_FACE_REVIEW}.intersection(ctx.state["registry"].enabled(Modality.VIDEO, ctx.profile)):
        return {**out, "voice_clock_policy": VOICE_CLOCK_POLICY}
    store = ctx.state.get("frame_store")
    sampled = ctx.state.get("frames") or []
    # The frozen experiment used these largest-face positions. No classifier
    # response enters the sequence method, so a classifier resource skip must
    # not prevent this independent, much smaller inspection check from running.
    faces = outputs.faces or []
    centers = tuple(sorted(f.frame_index for f in sampled if f.index < len(faces) and faces[f.index]))
    probe = outputs.probe or {}
    encoding = probe.get("codec"), probe.get("pix_fmt")
    inputs = (SequenceInputs(store.table, store, centers, tuple(common.stage_exhibit_list(ctx, ("tracks",))), encoding,
                             probe.get("codec_tag"))
              if store is not None else None)
    review = common.run_detectors(ctx, (Modality.VIDEO,), include={SEQUENCE_FAMILY, WHOLE_FACE_FAMILY},
                                  outputs=dataclasses.replace(outputs, face_review=inputs),
                                  max_units=VIDEO_MAX_UNIT_SCORES, face_failure=common.face_stage_failure(ctx))
    merged = {key: [*out.get(key, []), *review.get(key, [])] for key in ("detectors", "exhibits", "artefacts")}
    return {**merged, "voice_clock_policy": VOICE_CLOCK_POLICY}


# --- legacy fuser ------------------------------------------------------------------------------------------

def _units(entry: dict[str, Any], prefix: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for u in entry["unit_scores"]:
        head, _, index = u["unit_ref"].partition(":")
        if head == prefix and index.isdigit():
            out[int(index)] = u["raw"]
    return out


def _measurements(entry: dict[str, Any] | None, key: str, int_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    if entry is None or entry["aggregate"] is None:
        return {key: None}
    return {k: int(v) if k in int_keys else v for k, v in entry["metrics"].items()}


def _entries_by_id(ctx: StageContext, *, ok_only: bool) -> dict[str, dict[str, Any]]:
    return {e["detector_id"]: e for e in common.fusion_entries(ctx, ENTRY_STAGES) if not ok_only or e["status"] == "ok"}


def _c2pa_signal(entry: dict[str, Any] | None, n: int) -> dict[str, Any] | None:
    """The video C2PA run as a v2 signal (I-2). On the fallback path it is not averaged (weight 0) and acts only through
    Stage 0 (`provenance_override`)."""
    if entry is None or entry["status"] != "ok" or entry["aggregate"] is None:
        return None
    aggregate, override = entry["aggregate"], entry["provenance_override"]
    prob = decisive.OVERRIDE_PROBABILITY[override] if override in decisive.ACTIVE_PRECEDENCE else None
    fields = prob_to_signal_fields(prob) if prob is not None else {"direction": "inconclusive", "strength": "none"}
    return SignalV2.model_validate({
        "id": common.signal_id(entry["detector_id"], aggregate), "display_id": f"S{n:02d}", "detector_id": entry["detector_id"],
        "detector_version": entry["version"], "model_revisions": entry["model_revisions"], "name": entry["name"],
        "family": entry["family"], "modality": entry["modality"], "category": "provenance", "status": "ok", "scope": "global",
        "scope_ref": aggregate["unit_ref"], "hypothesis": H3, "raw_score": aggregate["raw"], "raw_kind": aggregate["raw_kind"],
        "effective_weight": 0.0, "observed": entry["observed"], "reference": entry["reference"], "reasoning": entry["reasoning"],
        "metrics": entry["metrics"], "details": entry.get("details") or {}, "exhibit_ids": entry["exhibit_ids"],
        "provenance_override": override,
        "legacy": {"direction": fields["direction"], "strength": fields["strength"],
                   "decisive": override in ("decisive_ai", "trusted_capture")},
    }).model_dump(mode="json")


def provenance_override(decision: dict[str, Any], caveats: list[str], signal: dict[str, Any] | None, *,
                        content_scored: bool = True) -> tuple[dict[str, Any], list[str]]:
    """Stage 0 (D-15) on the video fallback decision from the C2PA signal, with the headlines of the artefact path
    (`runtime.fuse_with_artefact`). Fallback codes are never `*_strong` (§4.4), as on the image fallback path. The
    statistical estimate before the override is kept (D-33). N-19h (E-19, 5): `content_scored`, whether a check of the
    frames or the sound track scored (D-64). Without one the score the fallback holds is one of records alone (container
    tags), which is no statistical score: the record decides alone, and nothing is published beside it."""
    kind = (signal or {}).get("provenance_override")
    if not kind:
        return decision, caveats
    if kind == "self_declared_ai":
        return decision, [*caveats, decisive.ADVISORY_CAVEAT]
    statistical = decision.get("probability_ai_uncalibrated") if content_scored else None
    marker = decisive.marker_kind([signal])  # N-19h (E-21): the words of the marker's kind, under a video
    outcome = decisive.apply(0.5 if statistical is None else statistical, [kind], marker=marker, kind_of_file="video")
    if kind == "decisive_ai":
        band, confidence, verbal = "very_likely_ai", "high", DECLARED_AI_VERBAL
        reason = decisive.FALLBACK_DECLARED_AI_REASON
    elif kind == "trusted_capture":
        band, confidence = "very_likely_authentic", "high"
        verbal = decisive.CAPTURE_VERBAL  # N-19m (D3-01; D-15)
        reason = decisive.FALLBACK_TRUSTED_CAPTURE_REASON
    code = BAND_CODE[band]
    # D-68: the number before the override is the fallback's score, printed as one
    estimate_caveat = decisive.STATISTICAL_ESTIMATE_CAVEAT.split(":", 1)[0]
    kept = [decisive.FALLBACK_STATISTICAL_CAVEAT.format(statistical * 100) if c.startswith(estimate_caveat) else c
            for c in outcome.caveats if statistical is not None or not c.startswith(decisive.STATISTICAL_CAVEATS)]
    updated = Decision.model_validate({
        **decision, "code": code, "verbal": verbal, "band": band, "confidence": confidence, "reason": reason,
        "probability_ai_uncalibrated": round(outcome.probability, 4), "driving_hypothesis": H3,
        "provenance_record": decisive.record(kind, [signal], [signal["id"]] if signal.get("id") else []),  # N-19h (E-21)
        "statistical_estimate_before_provenance": None if statistical is None else round(statistical, 4),
    }).model_dump(mode="json")
    return updated, [*caveats, *kept]


def video_scores(ctx: StageContext) -> VideoScores:
    """Rebuild the v0.3 per-frame scores, temporal measurements and voice results from detector runs."""
    probed, decoded = ctx.outputs["probe"], ctx.outputs["decode"]
    table = decoded["frames"]
    faces_per_frame = (ctx.outputs.get("faces") or {}).get("faces") or [[] for _ in table]
    ok = _entries_by_id(ctx, ok_only=True)
    per_frame = [{"faces": faces_per_frame[f["index"]], "scores": {}, "forensics": {}} for f in table]
    base = {s.short: s.weight for s in classifier_specs()}

    def collect(detector_id: str, key: str) -> dict[str, Any] | None:
        entry = ok.get(detector_id)
        if entry is not None and entry["aggregate"] is not None:
            for i, raw in _units(entry, "frame").items():
                per_frame[i]["scores"][key] = round(raw, 4)  # v0.3 fuses the rounded per-frame scores
            return entry
        return None

    general: list[tuple[str, float]] = []
    face_models: list[tuple[str, float]] = []
    for detector_id, key in GENERAL_DETECTORS:
        if collect(detector_id, key) is not None:
            general.append((key, frame_weight(key, settings.univfd_weight if key == "univfd" else base.get(key, 1.0), video=True)))
    for detector_id, key in FACE_CLASSIFIERS:
        entry = collect(detector_id, key)
        if entry is not None:
            upscaled = entry["aggregate"]["quality"].get("upscaled_fraction", 0.0)
            face_models.append((key, face_frame_weight(key, base.get(key, 1.0), video=True, upscaled_frac=upscaled)))
    dfdc = collect(DFDC_DETECTOR, DFDC_KEY)
    if dfdc is not None:
        face_models.append((DFDC_KEY, dfdc_frame_weight(dfdc["aggregate"]["quality"].get("small_face_fraction", 0.0))))
    fuse_frame_scores(per_frame, general, face_models)
    noise = ok.get(NOISE_DETECTOR)
    if noise is not None:
        for i, ratio in _units(noise, "frame").items():
            per_frame[i]["forensics"]["noise_face_bg_ratio"] = ratio

    audio = decoded["audio"]
    audio_sigs: list = []
    if audio["status"] == "silent":
        sig, audio_info = silent_signal(audio["samples"])
        audio_sigs.append(sig)
    elif audio["status"] in ("absent", "unreadable"):
        audio_info = {"audio": audio["status"]}
    else:
        preds = {key: [raw for _, raw in sorted(_units(ok[d], "window").items())]
                 for d, key in VOICE_DETECTORS if d in ok and ok[d]["aggregate"] is not None}
        if preds:
            weights = {repo.split("/")[-1]: w for repo, _pos, w in audio_specs()}
            spans = scored_voice_spans([ok[d] for d, _ in VOICE_DETECTORS if d in ok])
            scored_samples = round(coverage_seconds(spans) * SR)
            sig, audio_info = voice_signal(preds, weights, scored_samples, windows_label="window(s) of ≤ 4 s")
            if audio.get("source_clock_break_count"):
                sig = limit_voice_signal(sig)
                audio_info.pop("p_synthetic")
            coverage = {"seconds_decoded": round(audio["samples"] / SR, 3), "scored_spans": [list(span) for span in spans]}
            sig.details.update(coverage)
            audio_info.update(coverage)
            audio_sigs.append(sig)
        else:
            audio_info = {"audio": "models_failed"}
    return VideoScores(info=probed["info"], idxs=[f["frame_index"] for f in table], times=[f["time_s"] for f in table],
                       per_frame=per_frame, jitter=_measurements(ok.get(JITTER_DETECTOR), "landmark_jitter", ("frames_tracked",)),
                       flicker=_measurements(ok.get(FLICKER_DETECTOR), "face_flicker_ratio"), audio_signals=audio_sigs,
                       audio_info=audio_info, metadata=METADATA_DETECTOR in ok, general_models=len(general),
                       face_models=len(face_models))


def track_signals(ctx: StageContext, first: int) -> list[dict[str, Any]]:
    """One track-scoped signal per face track a per-track detector scored (V-3 face-swap lane, V-5 lip-sync lane): the
    `track:<k>` unit as the raw score, lane, threat and fp_modes from the model card, the track's face_track exhibit plus the
    exhibits that cite the signal (window series, Grad-CAM maps, lip-sync segments), and the segments of the segments
    exhibits that cite it. Not averaged by the legacy fuser (effective weight 0)."""
    manifest = getattr(ctx.state.get("registry"), "manifest", None)
    tracks = {t["track_id"]: t for t in (ctx.outputs.get("tracks") or {}).get("tracks") or []}
    exhibits = common.stage_exhibit_list(ctx, ("detect",))
    out: list[dict[str, Any]] = []
    for entry in (ctx.outputs.get("detect") or {}).get("detectors", []):
        if entry["status"] != "ok" or entry["detector_id"] not in TRACK_TEXT:
            continue
        card = manifest.detector(entry["detector_id"]) if manifest is not None else None
        for unit in entry["unit_scores"]:
            track = tracks.get(unit["unit_ref"])
            if track is None:
                continue
            sid = f"{entry['detector_id']}:{unit['unit_ref']}"
            observed, reasoning = TRACK_TEXT[entry["detector_id"]](track, unit)
            citing = [x for x in exhibits if sid in (x.get("source_signal_ids") or [])]
            out.append(SignalV2.model_validate({
                "id": sid, "display_id": f"S{first + len(out):02d}", "detector_id": entry["detector_id"],
                "detector_version": entry["version"], "model_revisions": entry["model_revisions"],
                "name": f"{entry['name']}: {swap_lane.track_name(track)}", "family": entry["family"], "modality": entry["modality"],
                "category": TRACK_CATEGORY[entry["detector_id"]], "status": "ok", "scope": "track", "scope_ref": unit["unit_ref"],
                "hypothesis": entry["hypotheses"][0] if entry["hypotheses"] else None,
                "lane": card.lane if card is not None else None, "threat": list(card.threat) if card is not None else [],
                "track_id": unit["unit_ref"], "segments": [s for x in citing if x["kind"] == "segments" for s in x["segments"]],
                "fp_modes": card_copy.client_modes(card.fp_modes) if card is not None else [],
                "raw_score": unit["raw"], "raw_kind": unit["raw_kind"], "effective_weight": 0.0, "observed": observed,
                "reference": entry["reference"], "reasoning": reasoning, "metrics": unit["quality"],
                "exhibit_ids": [track["exhibit_id"], *(x["id"] for x in citing)],
                "legacy": {"direction": "inconclusive", "strength": "none", "decisive": False},
            }).model_dump(mode="json"))
    return out


# per-track detectors → (observed, reasoning) of a track unit, and the signal category
TRACK_TEXT = {"vid.swap.gend_clip_l14": swap_lane.track_text, "vid.lipsync.stable_syncnet": lipsync_lane.track_text,
              "vid.lipsync.landmark_xcorr": xcorr_lane.track_text}
TRACK_CATEGORY = {"vid.swap.gend_clip_l14": "face", "vid.lipsync.stable_syncnet": "temporal", "vid.lipsync.landmark_xcorr": "temporal"}
SYNTHESIS_DETECTORS = {"vid.synth.d3_xclip": "temporal", "vid.synth.frames.commfor384": "model",
                       "vid.synth.frames.dda": "model",
                       "vid.codec.ffprobe": "metadata"}  # whole-video v2 detectors publishing one global signal → signal category


AUDIO_INSPECTION_DETECTORS = {"aud.pella_v2": "model"}
REVIEW_DETECTORS = {SEQUENCE_REVIEW: "temporal", APPEARANCE_REVIEW: "temporal", WHOLE_FACE_REVIEW: "face"}


def synthesis_signals(ctx: StageContext, first: int) -> list[dict[str, Any]]:
    """One global signal per whole-video v2 detector run with an aggregate (V-4: D3 and the frame channel; V-6: the codec
    lane): the aggregate as the raw score, threat and fp_modes from the model card, and the exhibits that cite the signal.
    Not averaged by the legacy fuser (effective weight 0)."""
    manifest = getattr(ctx.state.get("registry"), "manifest", None)
    exhibits = common.stage_exhibit_list(ctx, ("detect",))
    decoded = ctx.outputs.get("decode") or {}
    clock_limited = bool((decoded.get("audio") or decoded).get("source_clock_break_count"))
    out: list[dict[str, Any]] = []
    for entry in (ctx.outputs.get("detect") or {}).get("detectors", []):
        aggregate = entry["aggregate"]
        if entry["status"] != "ok" or entry["detector_id"] not in {**SYNTHESIS_DETECTORS, **AUDIO_INSPECTION_DETECTORS, **REVIEW_DETECTORS} or aggregate is None:
            continue
        card = manifest.detector(entry["detector_id"]) if manifest is not None else None
        sid = common.signal_id(entry["detector_id"], aggregate)
        details = entry.get("details") or {}
        screening = details.get("audio_screening") or {}
        limit_screening = clock_limited and entry["detector_id"] in AUDIO_INSPECTION_DETECTORS and bool(screening)
        if limit_screening:
            screening = limit_voice_screening(screening)
            details = {**details, "audio_screening": screening}
        out.append(SignalV2.model_validate({
            "id": sid, "display_id": f"S{first + len(out):02d}", "detector_id": entry["detector_id"],
            "detector_version": entry["version"], "model_revisions": entry["model_revisions"], "name": entry["name"],
            "family": entry["family"], "modality": entry["modality"], "category": {**SYNTHESIS_DETECTORS, **AUDIO_INSPECTION_DETECTORS, **REVIEW_DETECTORS}[entry["detector_id"]],
            "status": "ok", "scope": "global",
            "scope_ref": aggregate["unit_ref"], "hypothesis": entry["hypotheses"][0] if entry["hypotheses"] else None,
            "lane": card.lane if card is not None else None, "threat": list(card.threat) if card is not None else [],
            "fp_modes": card_copy.client_modes(card.fp_modes) if card is not None else [], "raw_score": aggregate["raw"],
            "raw_kind": aggregate["raw_kind"], "effective_weight": 0.0, "observed": entry["observed"],
            "reference": entry["reference"], "reasoning": VOICE_CLOCK_LIMIT if limit_screening else entry["reasoning"],
            "metrics": entry["metrics"], "details": details, "calibrated_probability": screening.get("probability_ai"),
            "calibration_ref": screening.get("calibration_sha256"),
            "exhibit_ids": [x["id"] for x in exhibits if sid in (x.get("source_signal_ids") or [])],
            "legacy": {"direction": "inconclusive", "strength": "none", "decisive": False},
        }).model_dump(mode="json"))
    return out


def sole_sources(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The one detector behind a composite v0.3 signal, by the signal's v1 id, when exactly one of its detectors scored (N-17;
    browser QA P1-4, P1-5). The signal is then that check and takes the detector's id, so the contribution the fusion model
    gives the detector, the ledger row and the finding are one thing. With several detectors the signal is their weighted
    average and keeps the id of the legacy fuser: no fusion model holds such an average, so it stays outside the estimate.
    `entries` are the detector entries by id (the served set: `vid.frames.dfdc_b7` alone on faces, `aud.gustking_xlsr` alone
    on the sound track)."""
    out: dict[str, Any] = {}
    for v1_id, detectors in COMPOSITES.items():
        scored = [entries[d] for d in detectors if d in entries and entries[d]["status"] == "ok" and entries[d]["aggregate"] is not None]
        if len(scored) == 1:
            out[v1_id] = scored[0]
    return out


SOLE_FACE_NAME = "Face-swap detector on the tracked face (mean over face frames)"


def sole_words(signal: Mapping[str, Any], entry: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """N-19q (N-20b's read of the reports, finding 6): the face-swap signal that one detector carries scores that detector's
    mean over the face frames, which the fusion model reads; its v0.3 name and observation gave the top-30% frame mean
    ("(top-30% frame mean) returned a score of 0.83" beside "top-30% mean p(fake)=0.980"). The name and the observation lead
    with the check's own score; the v0.3 statistics follow it. The decision reads neither."""
    if signal["id"] != "frames_face" or entry is None or entry.get("aggregate") is None:
        return signal
    raw, frames = float(entry["aggregate"]["raw"]), int((entry.get("metrics") or {}).get("face_frames") or 0)
    return {**signal, "name": SOLE_FACE_NAME,
            "observed": f"mean p(fake)={raw:.3f} over {frames} face frames (the check's score); {signal['observed']}"}


def voice_signal_id(entries: Mapping[str, Mapping[str, Any]]) -> str:
    """The id of the sound track's voice signal, the same in the temporal stage (lane sources, exhibits) and the fuse stage."""
    sole = sole_sources(entries).get("audio_synthetic_voice")
    return common.signal_id(sole["detector_id"], sole["aggregate"]) if sole is not None else VOICE_SIGNAL_ID


def fuse(ctx: StageContext) -> dict[str, Any]:
    """Legacy fuser → uncalibrated fallback Decision (§4.4) plus the v1 video blocks. With a compatible fuser artefact the
    calibrated decision replaces the fallback decision (M-2); the per-frame view stays the legacy one until V-8."""
    selection = common.select_fusion(ctx, "video")
    legacy = assemble_video(video_scores(ctx))
    frames = legacy["frames"]
    outside = selection.mode == "artefact"
    series_id = ctx.exhibits.series([{"name": key, "points": [(f["time"], f[key]) for f in frames]} for key in ("p_frame", "p_face", "p_general")],
                                    x_unit="s", title=SERIES_TITLE_OUTSIDE if outside else SERIES_TITLE, media_ref="frame",
                                    caption=SERIES_CAPTION_OUTSIDE if outside else SERIES_CAPTION)
    by_id = _entries_by_id(ctx, ok_only=False)
    sole = sole_sources(by_id)
    signals = [common.legacy_signal_v2(dict(sole_words(s, sole.get(s["id"]))), n, entry=by_id.get(SOURCES.get(s["id"], "")) or sole.get(s["id"]), fallback_detector=FUSER_ID,
                                       modality=MODALITY.get(s["id"], "video"), version=PIPELINE_VERSION,
                                       hypothesis=HYPOTHESIS.get(s["id"], "H2_localised_manipulation"),
                                       exhibit_ids=[series_id] if s["id"] in SERIES_SIGNALS else None)
               for n, s in enumerate(legacy["signals"], 1)]
    c2pa_signal = _c2pa_signal(by_id.get(C2PA_DETECTOR), len(signals) + 1)
    if c2pa_signal is not None:
        signals.append(c2pa_signal)
    signals += track_signals(ctx, len(signals) + 1)
    signals += synthesis_signals(ctx, len(signals) + 1)
    attach_temporal(ctx, signals)
    informative = any(s["ai_probability"] is not None for s in legacy["signals"])
    decision = common.fallback_decision(legacy["verdict"], informative=informative, subject="video")
    caveats = [FALLBACK_CAVEAT, *legacy["verdict"]["caveats"]]
    clock_limited = bool(((ctx.outputs.get("decode") or {}).get("audio") or {}).get("source_clock_break_count"))
    if clock_limited:
        caveats.append(VOICE_CLOCK_LIMIT)
    if selection.mode != "artefact":  # the artefact path applies Stage 0 itself (runtime.fuse_with_artefact)
        decision, caveats = provenance_override(decision, caveats, c2pa_signal,
                                                content_scored=abstain.content_check_scored(common.fallback_checks(signals)))
    view = {**legacy, "verdict": common.legacy_verdict(legacy["verdict"], decision)}
    out = {"decision": decision, "assessments": common.LEGACY_ASSESSMENTS, "signals": signals,
           "caveats": caveats, "v1": common.json_safe(view), **common.stage_exhibits(ctx, {"fuse"})}
    if selection.mode == "artefact":
        entries = common.fusion_entries(ctx, ENTRY_STAGES)
        if clock_limited:
            entries = exclude_recording_voice(entries)
        return common.artefact_fusion(ctx, "video", selection, out, entries)
    return common.fallback_fusion(ctx, "video", selection, out)


# --- temporal stage and second pass (V-7) ------------------------------------------------------------------------------

H2 = "H2_localised_manipulation"
LEGACY_FRAME_DETECTORS = frozenset({*(d for d, _ in GENERAL_DETECTORS), *(d for d, _ in FACE_CLASSIFIERS), DFDC_DETECTOR, NOISE_DETECTOR})
FAMILY_LANE = {"learned-face": "face_swap", "temporal-face": "face_swap", "av-sync": "lip_sync", "voice": "voice",
               "synthetic-video": "synthesis"}  # the lane of a detector whose model card names none (fakes, overrides)
LANE_TITLES = {"face_swap": "Face-swap", "lip_sync": "Lip-sync", "voice": "Voice", "synthesis": "Synthesis", "partial_edit": "Partial-edit"}
VOICE_SIGNAL_ID = f"{FUSER_ID}:audio_synthetic_voice"
SECOND_PASS_FAMILIES = frozenset({"learned-face", "temporal-face"})
SECOND_PASS_TOP, SECOND_PASS_PAD_S, SECOND_PASS_MAX_FRAMES = 3, 0.5, 64
MAX_SERIES_POINTS = 5000
TEMPORAL_SERIES_CAPTION = (  # N-17 (D-68): a score, in the caption and in the title, for every result
    "Lane score per window: the mean over the lane's detectors of the classifier output (a share from 0 to 1, or the logistic of "
    "a logit), uncalibrated, and its median over 3 windows. A segment starts at 0.60 and ends below 0.45 (at least 0.5 s); the "
    "lane scores and segment thresholds remain uncalibrated; headline calibration does not apply to them.")
TEMPORAL_SEGMENTS_CAPTION = (
    "Spans where the smoothed lane score reached 0.60 and stayed at or above 0.45 for at least 0.5 s. The peak is the "
    "highest smoothed score inside the span, uncalibrated.")


def _frame_table(ctx: StageContext) -> FrameTable | None:
    probed = ctx.outputs.get("probe") or {}
    return FrameTable.from_json(probed["frame_table"]) if probed.get("frame_table") else None


def _lane_of(ctx: StageContext) -> Any:
    manifest = getattr(ctx.state.get("registry"), "manifest", None)

    def lane_of(entry: Mapping[str, Any]) -> str | None:
        if entry["detector_id"] in LEGACY_FRAME_DETECTORS:
            return None  # the v0.3 per-frame set is shown by the legacy per-frame series
        if entry["detector_id"] in SYNTHESIS_DETECTORS:
            return "synthesis"  # VF §9.1: the frame and temporal channels share the synthesis lane
        card = manifest.detector(entry["detector_id"]) if manifest is not None else None
        lane = getattr(card, "lane", None)
        return str(lane) if lane else FAMILY_LANE.get(entry["family"])

    return lane_of


def _channel_of(entry: Mapping[str, Any]) -> str:
    return "audio" if entry["modality"] == "audio" else "av" if entry["family"] == "av-sync" else "visual"


def _published_ids(entry: Mapping[str, Any], track_id: str | None, *, voice_id: str = VOICE_SIGNAL_ID) -> list[str]:
    """The published signal a lane series cites for one of its detectors: the voice signal (`voice_signal_id`), the track
    signal of a per-track detector, or the global signal of a whole-video detector. Other detectors publish no signal on
    this path."""
    detector_id = entry["detector_id"]
    if detector_id in AUDIO_INSPECTION_DETECTORS and entry["aggregate"] is not None:
        return [common.signal_id(detector_id, entry["aggregate"])]
    if detector_id in dict(VOICE_DETECTORS):
        return [voice_id]
    if track_id is not None and detector_id in TRACK_TEXT:
        return [f"{detector_id}:{track_id}"]
    if track_id is None and detector_id in SYNTHESIS_DETECTORS and entry["aggregate"] is not None:
        return [common.signal_id(detector_id, entry["aggregate"])]
    return []


def _thin(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= MAX_SERIES_POINTS:
        return points
    keep = np.unique(np.linspace(0, len(points) - 1, num=MAX_SERIES_POINTS).round().astype(int))
    return [points[int(k)] for k in keep]


def _time_segment(seg: temporal_segments.Segment, series: temporal_segments.LaneSeries, table: FrameTable | None,
                  times: np.ndarray | None, hypothesis: str, sources: list[str]) -> dict[str, Any]:
    """A `TimeSegment`: visual and AV spans take the pts of the nearest frames; audio spans count samples at 16 kHz."""
    if series.channel == "audio" or table is None or times is None or len(times) == 0:
        time_base, start_pts, end_pts = f"1/{SR}", round(seg.start_s * SR), round(seg.end_s * SR)
    else:
        first = int(table.pts[int(np.argmin(np.abs(times - seg.start_s)))])
        last = int(table.pts[int(np.argmin(np.abs(times - seg.end_s)))])
        time_base, start_pts, end_pts = table.time_base, first, max(first, last)
    lane = series.lane if series.lane in {m.value for m in Lane} else None
    return TimeSegment.model_validate({
        "start_pts": start_pts, "end_pts": end_pts, "time_base": time_base, "start_s": seg.start_s, "end_s": seg.end_s,
        "channel": series.channel, "track_id": series.track_id, "peak": seg.peak, "lane": lane, "hypothesis": hypothesis,
        "sources": sources}).model_dump(mode="json")


def _emit_lanes(ctx: StageContext, series_list: list[temporal_segments.LaneSeries], entries_by_id: Mapping[str, Mapping[str, Any]],
                table: FrameTable | None, *, prefix: str = "",
                published: Callable[[Mapping[str, Any], str | None], list[str]] = _published_ids,
                subject: str = "whole video") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A series exhibit per lane and track, a segments exhibit where the lane is flagged, and the lane summaries.
    `published(entry, track_id)` names the signals a lane cites (the audio pipeline publishes its own voice signal)."""
    times = table.times() if table is not None else None
    tracks = {t["track_id"]: t for t in (ctx.outputs.get("tracks") or {}).get("tracks") or []}
    lanes: list[dict[str, Any]] = []
    found: list[dict[str, Any]] = []
    for series in series_list:
        first = entries_by_id[series.detectors[0]]
        hypothesis = first["hypotheses"][0] if first["hypotheses"] else H2
        cited = list(dict.fromkeys(i for d in series.detectors for i in published(entries_by_id[d], series.track_id)))
        sources = cited or [f"detector:{d}" for d in series.detectors]
        name = swap_lane.track_name(tracks[series.track_id]) if series.track_id in tracks else (series.track_id or subject)
        label = f"{prefix}{LANE_TITLES.get(series.lane, series.lane)} lane, {name}"
        media_ref = "audio" if series.channel == "audio" else "frame"
        centres = [round(w.sample_s if w.sample_s is not None else (w.start_s + w.end_s) / 2.0, 6)
                   for w in series.windows]
        sample_note = (" Frame values are point samples; segment boundaries do not establish continuous scored coverage between frames."
                       if any(w.sample_s is not None for w in series.windows) else "")
        series_id = ctx.exhibits.series(
            [{"name": "p_window", "hypothesis": hypothesis, "track_id": series.track_id,
              "points": _thin([(t, round(w.p, 6)) for t, w in zip(centres, series.windows, strict=True)])},
             {"name": "p_smoothed", "hypothesis": hypothesis, "track_id": series.track_id,
              "points": _thin([(t, round(p, 6)) for t, p in zip(centres, series.smoothed, strict=True)])}],
            x_unit="s", title=f"{label}: window score", media_ref=media_ref,
            thresholds=[temporal_segments.EXIT, temporal_segments.ENTER], caption=TEMPORAL_SERIES_CAPTION + sample_note, source_signal_ids=cited)
        spans = [_time_segment(s, series, table, times, hypothesis, sources) for s in series.segments]
        segments_id = ctx.exhibits.segments(spans, lane=series.lane, title=f"{label}: segments", media_ref=media_ref,
                                            caption=TEMPORAL_SEGMENTS_CAPTION + sample_note, source_signal_ids=cited) if spans else None
        lanes.append({"lane": series.lane, "track_id": series.track_id, "channel": series.channel, "detectors": list(series.detectors),
                      "families": list(series.families), "features": series.features(), "segments": spans,
                      "windows": [{"kind": "point" if w.sample_s is not None else "interval",
                                   "start_s": w.sample_s if w.sample_s is not None else w.start_s,
                                   "end_s": w.sample_s if w.sample_s is not None else w.end_s, "p": round(p, 6)}
                                  for w, p in zip(series.windows, series.smoothed, strict=True)],
                      "series_exhibit_id": series_id, "segments_exhibit_id": segments_id})
        found += spans
    return lanes, found


def temporal(ctx: StageContext) -> dict[str, Any]:
    """The detect stage's window scores → smoothed lane series and `TimeSegment`s per lane and track (VF §8.1), listed with
    the lip-sync lane's own segments. Persisted."""
    detected = common.entries(ctx, ("detect",))
    series = temporal_segments.collect(detected, lane_of=_lane_of(ctx), channel_of=_channel_of)
    by_id = {e["detector_id"]: e for e in detected}
    lanes, found = _emit_lanes(ctx, series, by_id, _frame_table(ctx), published=functools.partial(_published_ids, voice_id=voice_signal_id(by_id)))
    listed = [s for x in common.stage_exhibit_list(ctx, ("detect",)) if x["kind"] == "segments" for s in x["segments"]]
    return {"method": temporal_segments.METHOD, "params": temporal_segments.PARAMS, "lanes": lanes,
            "segments": sorted([*found, *listed], key=lambda s: (s["start_s"], s["end_s"], s.get("track_id") or "")),
            **common.stage_exhibits(ctx, {"temporal"})}


def _second_pass_exhibit(exhibit: Any) -> dict[str, Any]:
    """A detector exhibit of the second pass, titled as such (JSON exhibits re-hash; binary hashes cover the payload only)."""
    if exhibit.created_by == "second_pass":
        return exhibit.model_dump(mode="json")
    updated = exhibit.model_copy(update={"title": f"Second pass: {exhibit.title}"})
    if updated.kind not in ("heatmap", "image"):
        updated = updated.model_copy(update={"sha256": exhibit_payload_hash(updated)})
    return updated.model_dump(mode="json")


def second_pass(ctx: StageContext) -> dict[str, Any]:
    """Deep (§3.1 `second_pass`): every frame of the track around the top-3 face-lane segments (padded by 0.5 s, at most 64
    frames per segment), the per-track face detectors re-run on those frames only, and the dense windows segmented again.
    The budget is the frame caps, the profile's `second_pass_cap_s` when set (a detector check past it ends the pass, not the
    analysis) and the analysis deadline. Persisted."""
    cap = SAMPLING[ctx.profile].second_pass_cap_s
    out: dict[str, Any] = {
        "method": temporal_segments.METHOD, "reason": None, "windows": [], "frames": 0, "budget_exhausted": False,
        "budget": {"top": SECOND_PASS_TOP, "pad_s": SECOND_PASS_PAD_S, "max_frames_per_segment": SECOND_PASS_MAX_FRAMES, "cap_s": cap},
        "detectors": [], "lanes": [], "segments": [], "exhibits": [], "artefacts": []}
    if cap == 0:
        return {**out, "reason": "no second pass in this profile"}
    tracked = track_inputs(ctx)
    if tracked is None:
        return {**out, "reason": "no face tracks"}
    picks = [s for lane in (ctx.outputs.get("temporal") or {}).get("lanes") or []
             if lane["track_id"] is not None and SECOND_PASS_FAMILIES & set(lane["families"]) for s in lane["segments"]]
    windows = temporal_segments.dense_windows(picks, tracked.frames, tracked.table.seconds, top=SECOND_PASS_TOP,
                                              pad_s=SECOND_PASS_PAD_S, max_frames=SECOND_PASS_MAX_FRAMES)
    if not windows:
        return {**out, "reason": "no face-lane segment to refine"}
    dense: dict[str, set[int]] = {}
    for w in windows:
        dense.setdefault(w["track_id"], set()).update(w["frames"])
    windowed = TrackInputs(table=tracked.table, tracks=tuple(t for t in tracked.tracks if t["track_id"] in dense),
                           frames={tid: [f for f in tracked.frames[tid] if int(f["frame_index"]) in frames] for tid, frames in dense.items()},
                           sparse=tuple(sorted(set().union(*dense.values()))), sparse_fps=None, store=tracked.store)
    writer = ctx.exhibits
    seen_exhibits, seen_artefacts = {e.id for e in writer.exhibits}, {a.id for a in writer.artefacts}
    budgeted = ctx if cap is None else dataclasses.replace(ctx, deadline=ctx.deadline.nested(cap))
    exhausted = False
    try:
        rerun = common.run_detectors(budgeted, (Modality.VIDEO,), include=set(SECOND_PASS_FAMILIES),
                                     outputs=StageOutputs(probe=(ctx.outputs.get("probe") or {}).get("info"), tracks=windowed),
                                     max_units=VIDEO_MAX_UNIT_SCORES)["detectors"]
    except DetectorTimeout:
        rerun, exhausted = [], True
    series = temporal_segments.collect(rerun, lane_of=_lane_of(ctx), channel_of=_channel_of)
    lanes, found = _emit_lanes(ctx, series, {e["detector_id"]: e for e in rerun}, tracked.table, prefix="Second pass: ")
    unused = common.not_stored(rerun)  # N-12b: what a detector wrote whose run could not be stored is not part of the result
    return {**out, "windows": windows, "frames": len(windowed.sparse), "budget_exhausted": exhausted, "detectors": rerun,
            "lanes": lanes, "segments": found,
            "exhibits": [_second_pass_exhibit(e) for e in writer.exhibits if e.id not in seen_exhibits and e.created_by not in unused],
            "artefacts": [dataclasses.asdict(a) for a in writer.artefacts if a.id not in seen_artefacts and a.created_by not in unused]}


def attach_temporal(ctx: StageContext, signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Temporal and second-pass exhibits citing a signal join its exhibit ids; the temporal stage's segments join its
    segments (the second pass refines inside its windows and stays in its own exhibits)."""
    first = common.stage_exhibit_list(ctx, ("temporal",))
    exhibits = [*first, *common.stage_exhibit_list(ctx, ("second_pass",))]
    for signal in signals:
        citing = [x for x in exhibits if signal["id"] in (x.get("source_signal_ids") or [])]
        if citing:
            signal["exhibit_ids"] = list(dict.fromkeys([*signal["exhibit_ids"], *(x["id"] for x in citing)]))
            signal["segments"] = [*signal["segments"], *(s for x in first if x["kind"] == "segments" and signal["id"] in x["source_signal_ids"]
                                                         for s in x["segments"])]
    return signals


def synthesise(ctx: StageContext) -> dict[str, Any]:
    out = common.synthesise_findings(ctx, kind="video", entry_stages=ENTRY_STAGES, exhibit_stages=EXHIBIT_STAGES,
                                     family_by_v1_id=FINDING_FAMILY)
    for signal in (ctx.outputs.get("fuse") or {}).get("signals", []):
        if signal.get("detector_id") not in REVIEW_DETECTORS or signal.get("status", "ok") != "ok":
            continue
        details = signal.get("details") or {}
        whole = details.get("whole_face_review")
        sequence = details.get("face_sequence")
        short_change = details.get("face_return")
        second_change = details.get("face_return_secondary")
        window_changes = details.get("face_short_windows") or []
        short_coverage = (details.get("face_return_check") or {}).get("status")
        detail_status = (details.get("face_detail_repeat") or {}).get("status")
        for finding in out["findings"]:
            if signal["id"] not in finding["evidence_ids"]:
                continue
            if whole:
                first = signal["observed"]
                second = "The six source frames show the sampled positions. This check carries no weight in the overall AI assessment."
                finding.update(title="Sampled faces to review" if whole["flagged"] else "Sampled face review",
                               statement=f"{first} {second}", probability=None, direction="neutral", strength="none",
                               review_priority="medium" if whole["flagged"] else "info", standalone_strength=None, standalone_direction=None,
                               sentences=[{"text": text, "evidence_ids": finding["evidence_ids"]} for text in (first, second)])
                continue
            if detail_status == "withheld":
                title = "Brief face-change range to review" if short_change else "Face change did not repeat at lower detail"
                first = signal["observed"]
                if not short_change and short_coverage in {"not_applicable", "unassessed"}:
                    first += " The separate brief face-change check could not assess this clip."
            elif sequence:
                title, first = "Possible face-change range", sequence_observation(sequence)
                if cut := details.get("face_spatial_cut"):
                    title, first = "Picture-cut context for a face change", face_spatial_cut_binding.observation(cut)
                elif is_paired(boundary := details.get("face_boundary")):
                    title = "Refined face-change range to review"
                    first = (f"Two nearby face transitions narrow the review to {boundary['start_stream_s']:.3f}–"
                             f"{boundary['end_stream_s']:.3f} s on the source clock. The broader face check supplies context.")
                if detail_status == "retained":
                    first += " The change remained when the same frames were checked at lower detail."
                if short_change:
                    first += " " + face_return_binding.observation(short_change)
                elif short_coverage in {"not_applicable", "unassessed"}:
                    first += " The separate brief face-change check could not assess this clip."
            elif short_change:
                title, first = "Brief face-change range to review", face_return_binding.observation(short_change)
            elif short_coverage in {"not_applicable", "unassessed"}:
                title = "Brief face changes unassessed"
                first = "The brief face-change check could not assess this clip. The other face samples supplied no additional range to review."
            else:
                title = "No additional face-change range"
                first = "The sampled face sequences supplied no additional change range to review. The result covers the measured windows."
            if second_change:
                title = "Brief face-change ranges to review"
                first += " " + face_return_binding.observation(second_change)
            if window_changes:
                title = "Brief face-change ranges to review" if len(window_changes) > 1 or short_change else "Brief face-change range to review"
                explanation = " ".join(face_return_binding.observation(item) for item in window_changes)
                first = f"{first} {explanation}" if sequence or short_change else explanation
            second = "Motion, expression, lighting and compression may cause similar changes. This inspection cue has no calibrated AI probability and does not change overall confidence."
            finding.update(title=title,
                           statement=first+" "+second, probability=None, direction="neutral", strength="none",
                           review_priority="medium" if sequence or short_change else "info", standalone_strength=None, standalone_direction=None,
                           sentences=[{"text": text, "evidence_ids": finding["evidence_ids"]} for text in (first, second)])
    return out


def _still_jpeg(rgb: np.ndarray) -> tuple[bytes, int, int]:
    image = Image.fromarray(rgb)
    if max(image.size) > STILL_MAX_PX:
        image.thumbnail((STILL_MAX_PX, STILL_MAX_PX), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=STILL_QUALITY)
    return buf.getvalue(), image.width, image.height


def _native_render_anchors(ctx: StageContext, frame_store: FrameStore) -> list[dict]:
    """Recover source commitments when rendering a stored appearance check."""
    expected = {}
    refined_indices = set()
    for entry in (ctx.outputs.get("detect") or {}).get("detectors", []):
        details = entry.get("details") or {}
        inputs = []
        if cut := details.get("face_spatial_cut"):
            inputs.extend(cut["frame_inputs"])
            refined_indices.update(row["frame_index"] for row in cut["frame_inputs"])
        if boundary := details.get("face_boundary"):
            boundary_inputs = face_boundary_source_inputs(boundary)
            refined_indices.update(item["frame_index"] for item in boundary_inputs)
            inputs.extend(boundary_inputs)
            for record in (details["face_sequence"], *details.get("face_sequence_windows", [])):
                inputs.extend(record["frame_inputs"])
        elif details.get("face_detail_repeat"):
            for record in details.get("face_sequence_windows", []):
                inputs.extend(record["frame_inputs"])
            if measurement := details["face_detail_repeat"].get("measurement"):
                refined_indices.update(item["frame_index"] for item in measurement["frame_inputs"])
        for window in details.get("face_short_windows") or []:
            inputs.extend(window["frame_inputs"])
            inputs.extend(window["scene_frame_inputs"])
            refined_indices.update(row["frame_index"] for row in window["scene_frame_inputs"])
        for key in ("face_return", "face_return_secondary"):
            if (face_return := details.get(key)) is None:
                continue
            supplemental = [*face_return["frame_inputs"], *face_return["scene_frame_inputs"]]
            refined_indices.update(item["frame_index"] for item in supplemental)
            inputs.extend(supplemental)
        for item in inputs:
            identity = item["pts"], item["decoded_rgb_sha256"]
            if expected.setdefault(item["frame_index"], identity) != identity:
                raise DecodeError("Stored face inputs disagree on the decoded source frame")
    anchors = frame_store.native_anchors()
    if not expected:
        return anchors
    missing = set(expected).difference(item["frame_index"] for item in anchors)
    if missing:
        for _ in frame_store.iter_many(missing):
            ctx.check("restore measured face evidence frames")
        anchors = frame_store.native_anchors()
    committed = {item["frame_index"]: (item["pts"], item["decoded_rgb_sha256"]) for item in anchors}
    if any(committed.get(index) != identity for index, identity in expected.items()):
        raise DecodeError("Stored face evidence differs from the re-decoded source frames")
    if any(not frame_store.native_pts_verified(index) for index in refined_indices):
        raise DecodeError("Native face evidence lacks a verified decoded timestamp")
    return anchors


def render(ctx: StageContext) -> dict[str, Any]:
    """Stills of the sampled frames (`/stills/{pts}.jpg`), artefact indexing, and the v1 artifact files."""
    frames = ctx.state.get("frames")
    if not frames:
        raise RuntimeError("the decoded frames are not available to render")
    writer = ctx.exhibits.bind("render")
    for f in frames:
        data, width, height = _still_jpeg(f.rgb)
        writer.image(data, role="still", media_type="image/jpeg", title=f"Sampled frame {f.index + 1} ({f.time_s:.2f} s)",
                     caption=f"The decoded frame the detectors scored, stored as JPEG q{STILL_QUALITY}.", media_ref="frame",
                     frame_pts=f.pts_ms, coordinate_space={"width": width, "height": height,
                                                           "scale_to_original": round(f.rgb.shape[1] / width, 6)})
    stills = common.stage_exhibits(ctx, {"render"})
    out = common.index_artefacts(ctx, EXHIBIT_STAGES, extra=stills["artefacts"])
    view = ctx.outputs["fuse"]["v1"]
    faces_per_frame = (ctx.outputs.get("faces") or {}).get("faces") or [[] for _ in frames]
    per_frame = [{"p_frame": fr["p_frame"], "p_face": fr["p_face"], "faces": faces_per_frame[k]} for k, fr in enumerate(view["frames"])]
    artifacts, key_times = video_artifacts(common.legacy_artifact_dir(ctx), [f.rgb for f in frames], per_frame, [f.time_s for f in frames])
    frame_store = ctx.state.get("frame_store")
    anchors = _native_render_anchors(ctx, frame_store) if frame_store is not None else []
    hold = None
    if frame_store is not None:
        table = frame_store.table
        if provenance_gate.stop(ctx):
            limit = picture_hold_binding.frame_limit(len(table), table.width, table.height)
            hold = {"coverage": picture_hold_binding.coverage("unavailable", picture_hold_binding.PROVENANCE, table.time_base,
                                                              limit=limit), "match": None}
        else:
            info = ctx.outputs["probe"]["info"]
            hold = picture_hold.inspect(frame_store.path, table, origin=info.get("timeline_origin_s"),
                                        check=lambda: ctx.check("repeated-picture inspection"), remaining_s=ctx.deadline.remaining(),
                                        sequence_review=True, motion_encoding=(info.get("codec"), info.get("pix_fmt")), mouth_review=True)
            mouth = hold["coverage"].get("mouth_local_hold")
            audio = ctx.state.get("audio")
            mapping = ctx.state.get("video_audio_mapping")
            if mouth and audio is not None and audio.pcm is not None and mapping:
                from app.detectors.video.mouth_hold_sound import measure

                if sound := measure(audio.pcm, mapping, mouth, table.time_base):
                    mouth["sound_context"] = sound
    scene = None
    scene_coverage = None
    shot_output = ctx.outputs.get("shots") or {}
    if settings.environment == "dev" and settings.video_scene_return_review:
        if provenance_gate.stop(ctx):
            scene_coverage = scene_return_binding.coverage("unavailable", reason="provenance")
        elif frame_store is None:
            scene_coverage = scene_return_binding.coverage("unavailable", reason="unsupported_video", webm_available=True)
        elif ctx.profile == Profile.QUICK:
            scene_coverage = scene_return_binding.coverage("unavailable", reason="profile")
        elif not shot_output:
            scene_coverage = scene_return_binding.coverage("unavailable", reason="scan_unavailable")
        else:
            scene_coverage = shot_output.get(scene_return_binding.COVERAGE)
    if frame_store is not None and not provenance_gate.stop(ctx):
        pending = shot_output.get(scene_return_binding.KEY)
        remaining = ctx.deadline.remaining()
        schema = pending.get("schema", scene_return_binding.SCHEMA) if pending is not None else None
        if pending is not None and (remaining is None or remaining > 2):
            found = scene_return.SceneReturn(pending["start"], pending["end"], tuple(pending["tiles"]))
            try:
                timeout = 8. if remaining is None else min(8., remaining - 1.)
                for _ in frame_store.iter_many(found.anchors, timeout_s=timeout):
                    ctx.check("read picture-change evidence frames")
                anchors = frame_store.native_anchors()
                by_index = {anchor["frame_index"]: anchor["decoded_rgb_sha256"] for anchor in anchors}
                if all(i in by_index and frame_store.native_pts_verified(i) for i in found.anchors):
                    scene = scene_return_binding.record(found, frame_store.table, [by_index[i] for i in found.anchors], schema=schema)
            except DecodeError:
                ctx.log.warning("video.scene_return_frames_unavailable")
        if pending is not None:
            first = frame_store.table.pts[0] if schema == scene_return_binding.WEBM_SCHEMA else None
            scene_coverage = (scene_return_binding.coverage("matched", frames_scanned=scene_return_binding.FRAME_COUNT, first_pts=first)
                              if scene is not None else scene_return_binding.coverage(
                                  "unavailable", reason="missing_anchors", frames_scanned=scene_return_binding.FRAME_COUNT, first_pts=first))
    decode_output = ctx.outputs.get("decode") or {}
    decoded_audio = decode_output.get("audio") or {}
    stereo_sha = decoded_audio.get("stereo_context_sha256")
    # A reused historical render output must not acquire evidence on publication.
    stereo_objects = [obj for obj in decode_output.get("artefacts", []) if obj["sha256"] == stereo_sha]
    return {**out, **stills, "legacy_artifacts": artifacts, "key_frame_times": key_times,
            "artefacts": stills["artefacts"] + stereo_objects,
            "frame_anchor_version": "stream-pts-rgb24/1", "frame_anchors": anchors,
            "picture_hold": hold,
            **({scene_return_binding.KEY: scene} if scene is not None else {}),
            **({scene_return_binding.COVERAGE: scene_coverage} if scene_coverage is not None else {}),
            "audio_mapping": ctx.state.get("video_audio_mapping"), "audio_status": decoded_audio.get("status"),
            "audio_stereo_context_sha256": stereo_sha}


def v1_view(ctx: StageContext, result: AnalysisResultV2) -> dict[str, Any]:
    """The v0.3 video result from the fuser, with the artifact files and key-frame times written by render."""
    rendered = ctx.outputs.get("render") or {}
    return attach_artifacts(copy.deepcopy(ctx.outputs["fuse"]["v1"]), rendered.get("legacy_artifacts") or [],
                            rendered.get("key_frame_times") or {})


def publish(ctx: StageContext) -> dict[str, Any]:
    result = common.build_result(ctx, kind="video", pipeline_version=PIPELINE_VERSION, entry_stages=ENTRY_STAGES,
                                 exhibit_stages=EXHIBIT_STAGES)
    rendered = ctx.outputs.get("render") or {}
    if rendered.get("frame_anchor_version") == "stream-pts-rgb24/1":
        result.manifest.sampling.update({"frame_anchor_version": "stream-pts-rgb24/1",
                                         "frame_anchor_sha256": canonical_hash(rendered["frame_anchors"]),
                                         "frame_table_sha256": canonical_hash(ctx.outputs["probe"]["frame_table"]),
                                         "timeline_origin_s": ctx.outputs["probe"]["info"].get("timeline_origin_s"),
                                         "exhibit_frame_pts_unit": "milliseconds"})
    if rendered.get("audio_mapping") is not None:
        result.manifest.sampling["video_audio_mapping_sha256"] = canonical_hash(rendered["audio_mapping"])
    if rendered.get("picture_hold") is not None:
        result.manifest.sampling["picture_hold_sha256"] = canonical_hash(rendered["picture_hold"])
    if rendered.get(scene_return_binding.KEY) is not None:
        result.manifest.sampling[scene_return_binding.COMMITMENT] = canonical_hash(rendered[scene_return_binding.KEY])
    if rendered.get(scene_return_binding.COVERAGE) is not None:
        result.manifest.sampling[scene_return_binding.COVERAGE_COMMITMENT] = canonical_hash(rendered[scene_return_binding.COVERAGE])
    for entry in (ctx.outputs.get("detect") or {}).get("detectors", []):
        if entry["detector_id"] != APPEARANCE_REVIEW:
            continue
        details = entry.get("details") or {}
        coverage = details.get("face_return_check")
        if coverage is not None and coverage.get("sparse_context") is None and {"dense_fallback", "fine_scene_fallback"} & coverage.keys():
            coverage = None
        bounded = details.get("face_short_proposal_check")
        if coverage is not None or bounded is not None:
            count = len(ctx.outputs["probe"]["frame_table"]["pts"])
            result.manifest.sampling[face_return_coverage.KEY] = face_return_coverage.from_worker(
                coverage, count, bounded=bounded, second_avc=details.get("face_second_avc_check"))
        if windows := details.get("face_short_windows_check"):
            coverage = face_short_window_binding.coverage(windows) | (details.get("face_short_windows_source") or {})
            result.manifest.sampling[face_short_window_binding.KEY] = face_short_window_binding.checked(coverage, len(ctx.outputs["probe"]["frame_table"]["pts"]))
        break
    from app.evidence.video_audio import published_record

    result.manifest.sampling["video_audio_evidence"] = published_record(ctx)
    return common.publish(ctx, result, v1_view(ctx, result), pipeline_version=PIPELINE_VERSION, run_stages=SIGNAL_STAGES)
