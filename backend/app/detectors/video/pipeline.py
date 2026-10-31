"""Video analysis: probe → sample frames → batched frame scoring → temporal consistency → fusion.

Three parts, so the v1 path and the v2 video pipeline share one fusion:
- `score_video` samples frames and runs every detector through the shared registry → `VideoScores`;
- `assemble_video` is the v0.3 signal construction, hypothesis separation and verdict over those scores;
- `video_artifacts` writes the v1 artifact files (timeline, contact sheet, poster, key frames).
The v2 legacy fuser (`engine/stages/video.py`) rebuilds `VideoScores` from detector runs and calls `assemble_video`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.core import deadline
from app.core.config import settings
from app.core.evidence import Signal
from app.core.logging import get_logger
from app.detectors.common.fusion import fuse, prob_to_signal_fields
from app.detectors.image.forensics import logistic
from app.detectors.image.scoring import score_frames
from app.detectors.video import temporal
from app.detectors.video.audio import audio_signals
from app.detectors.video.frames import probe, sample_frames

log = get_logger(__name__)

PHONE_TAG_KEYS = ("com.apple.quicktime.make", "com.apple.quicktime.model", "com.android.version", "com.android.manufacturer")
FRAME_SIGNAL_ARTIFACTS = ["timeline.png", "key_frames.jpg"]


def _save(art_dir: Path, name: str, rgb: np.ndarray) -> str:
    Image.fromarray(rgb).save(art_dir / name, quality=88)
    return name


def metadata_signals(info: dict) -> list[Signal]:
    sig: list[Signal] = []
    tags = info.get("tags", {})
    encoder = tags.get("encoder", "") or tags.get("handler_name", "")
    phone = {k: v for k, v in tags.items() if k in PHONE_TAG_KEYS}
    if phone:
        sig.append(
            Signal(
                id="video_device_tags", name="Capture-device container tags", category="metadata",
                observed="; ".join(f"{k.split('.')[-1]}={v}" for k, v in phone.items()),
                reference="Deepfake tools re-encode output and drop device tags; tags are forgeable",
                ai_probability=0.30, weight=0.4, direction="supports_authentic", strength="weak",
                reasoning="The container carries device-recorded metadata, consistent with an unmodified camera capture.",
                details=phone,
            )
        )
    else:
        sig.append(
            Signal(
                id="video_encoder", name="Container / encoder", category="metadata",
                observed=f"container={info.get('container')}, codec={info.get('codec')}, encoder='{encoder or 'n/a'}', {info.get('width')}x{info.get('height')} @ {info.get('fps')} fps, audio={'yes' if info.get('has_audio') else 'no'}",
                reference="Generic encoders (e.g. Lavf/ffmpeg) indicate re-encoding, which is common to both edited and re-shared videos",
                ai_probability=None, weight=0.0, direction="inconclusive", strength="none",
                reasoning="No capture-device metadata; the file has been re-encoded at least once. Uninformative on its own."
                + (" The absence of an audio track is common in research deepfake datasets and screen exports." if not info.get("has_audio") else ""),
                details={"encoder": encoder},
            )
        )
    return sig


def frame_count(duration: float) -> int:
    """v0.3 sampling plan: 4 frames per second, clipped to [video_min_frames, video_max_frames]."""
    return int(np.clip(round(duration * 4) if duration else settings.video_max_frames, settings.video_min_frames, settings.video_max_frames))


@dataclass
class VideoScores:
    """Detector outputs the legacy video fusion reads."""

    info: dict
    idxs: list[int]
    times: list[float]
    per_frame: list[dict]  # faces (v1 face dicts), scores, forensics, p_general, p_face, p_frame
    jitter: dict = field(default_factory=lambda: {"landmark_jitter": None})
    flicker: dict = field(default_factory=lambda: {"face_flicker_ratio": None})
    audio_signals: list[Signal] = field(default_factory=list)
    audio_info: dict = field(default_factory=dict)
    metadata: bool = True  # container metadata signal available
    general_models: int = 1  # whole-frame detectors that scored frames
    face_models: int = 1  # face detectors that scored face frames


def score_video(path: Path) -> tuple[VideoScores, list[np.ndarray], dict[str, float]]:
    """v1: sample frames and run every video detector through the shared registry."""
    timings: dict[str, float] = {}
    info = probe(path)
    n = frame_count(info.get("duration") or 0)

    t = time.perf_counter()
    sampled = sample_frames(path, n)
    idxs, times, frames = [s[0] for s in sampled], [s[1] for s in sampled], [s[2] for s in sampled]
    timings["decode"] = round(time.perf_counter() - t, 3)
    if not frames:
        raise ValueError("No frames could be decoded")
    deadline.check("frame decoding")

    t = time.perf_counter()
    per_frame = score_frames(frames, video=True)
    deadline.check("frame scoring")
    timings["scoring"] = round(time.perf_counter() - t, 3)

    frame_faces = [f["faces"] for f in per_frame]
    track = temporal.track_primary_face(frame_faces, settings.video_face_track_iou)
    jitter = temporal.landmark_jitter(frame_faces, track)
    flicker = temporal.face_flicker(frames, frame_faces, track)

    # --- sound track: synthetic-voice detectors ---
    try:
        t = time.perf_counter()
        a_sigs, audio_info = audio_signals(path, bool(info.get("has_audio")))
        timings["audio"] = round(time.perf_counter() - t, 3)
    except Exception as e:
        log.warning("audio.failed", error=str(e)[:160])
        a_sigs, audio_info = [], {"audio": "error"}
    deadline.check("audio")
    return VideoScores(info, idxs, times, per_frame, jitter, flicker, a_sigs, audio_info), frames, timings


def assemble_video(scores: VideoScores, *, timings: dict[str, float] | None = None) -> dict:
    """The v0.3 video signals, hypothesis separation and verdict. Artifact names are attached by the caller."""
    info, per_frame, times, idxs = scores.info, scores.per_frame, scores.times, scores.idxs
    dur = info.get("duration") or 0
    signals: list[Signal] = metadata_signals(info) if scores.metadata else []
    frame_faces = [f["faces"] for f in per_frame]
    n_face_frames = sum(1 for f in frame_faces if f)

    # --- learned detectors aggregated over frames ---
    if scores.general_models:
        pg = np.array([f["p_general"] for f in per_frame])
        pg_mean = float(pg.mean())
        signals.append(
            Signal(
                id="frames_general", name="Whole-frame AI-image detectors (UnivFD + classifiers), averaged", category="model",
                observed=f"mean p(AI)={pg_mean:.3f} over {len(per_frame)} frames; max={pg.max():.3f}",
                reference="Camera footage: mean < 0.35. Fully synthetic (text-to-video) footage scores high on most frames",
                weight=1.0, **prob_to_signal_fields(pg_mean),
                reasoning=("Frame-level detectors consistently classify the scene content as synthetic." if pg_mean >= 0.6 else "Frame-level detectors classify the overall scene as camera-captured; if manipulation exists it is localised (e.g. a swapped face) rather than whole-frame synthesis."),
                details={"per_frame": [f["p_general"] for f in per_frame]},
            )
        )
    if n_face_frames and scores.face_models:
        pf = np.array([f["p_face"] for f in per_frame if f["p_face"] is not None])
        topk = np.sort(pf)[-max(1, int(np.ceil(len(pf) * 0.3))):]
        p_top = float(topk.mean())
        flagged = int((pf > 0.5).sum())
        few = len(pf) < 6
        small_faces = float(np.mean([max((f["w"] for f in fr["faces"]), default=0) < settings.dfdc_min_face_px for fr in per_frame if fr["faces"]]))
        w_face = 1.6 * (0.5 if few else 1.0) * (0.3 if small_faces > 0.8 else 0.5 if small_faces > 0.5 else 1.0)
        signals.append(
            Signal(
                id="frames_face", name="Face-swap detectors on the tracked face (top-30% frame mean)", category="face",
                observed=f"top-30% mean p(fake)={p_top:.3f}; {flagged}/{len(pf)} face frames > 0.5; median={np.median(pf):.3f}" + (f"; faces small (<{settings.dfdc_min_face_px}px) in {small_faces:.0%} of face frames" if small_faces > 0.5 else "") + ("; few face frames" if few else ""),
                reference="Authentic faces: top-30% mean < 0.35 and < 20% of frames flagged; detectors are trained on faces ≥ ~150 px",
                weight=w_face, **prob_to_signal_fields(p_top),
                reasoning=(f"The face region is classified as synthetic in {flagged} of {len(pf)} sampled frames; manipulation need not be visible in every frame, so the upper tail is informative." if p_top >= 0.5 else "The face region scores below the face-swap threshold across sampled frames."),
                details={"per_frame": [f["p_face"] for f in per_frame]},
            )
        )
    elif not n_face_frames:
        signals.append(
            Signal(
                id="no_face_frames", name="No face detected in sampled frames", category="face",
                observed="0 frames with a detectable face", reference="Face-swap detectors require a visible face",
                ai_probability=None, weight=0.0, direction="inconclusive", strength="none",
                reasoning="Face-swap analysis is not applicable; the verdict rests on whole-frame synthesis detectors and forensics.",
            )
        )

    # --- temporal consistency ---
    jit, fl = scores.jitter, scores.flicker
    vol = temporal.score_volatility([f["p_face"] for f in per_frame])
    if jit.get("landmark_jitter") is not None:
        j = jit["landmark_jitter"]
        p = 0.40 + 0.35 * logistic(j, 0.035, 0.01)
        signals.append(
            Signal(
                id="landmark_jitter", name="Facial-landmark temporal jitter", category="temporal",
                observed=f"median box-normalised landmark shift={j} (p90 {jit['landmark_jitter_p90']}); tracked {jit['frames_tracked']} frames",
                reference="Natural motion at 2 fps sampling: median shift ≲ 0.03 of box size; swapped faces show erratic eye/mouth placement",
                weight=0.25, **prob_to_signal_fields(p),
                reasoning=("Landmark positions relative to the face box move erratically between frames, consistent with per-frame face synthesis." if p > 0.55 else "Landmark placement is stable relative to the face box, as expected for a real face under natural motion."),
                details=jit,
            )
        )
    if fl.get("face_flicker_ratio") is not None:
        r = fl["face_flicker_ratio"]
        p = 0.40 + 0.35 * logistic(r, 1.5, 0.2)
        signals.append(
            Signal(
                id="face_flicker", name="Face-region temporal flicker vs background", category="temporal",
                observed=f"face change / ring change = {r} (face {fl['face_change']}, ring {fl['background_change']})",
                reference="Real footage: face and surrounding ring change alike (ratio ≈ 0.7–1.4); re-synthesised faces change more than the ring (> 1.6)",
                weight=0.25, **prob_to_signal_fields(p),
                reasoning=("The face region changes far more between frames than its surroundings, a signature of frame-wise blending." if p > 0.55 else "Face-region change is proportional to scene change; no independent flicker."),
                details=fl,
            )
        )
    if vol.get("score_std") is not None:
        s_ = vol["score_std"]
        p = 0.40 + 0.35 * logistic(s_, 0.22, 0.06)
        signals.append(
            Signal(
                id="score_volatility", name="Frame-to-frame detector score volatility", category="temporal",
                observed=f"std={s_}, flagged fraction={vol['flagged_fraction']}",
                reference="Authentic faces: std < 0.15 (uniformly low scores). Manipulated faces alternate between detected/undetected frames",
                weight=0.2, **prob_to_signal_fields(p),
                reasoning=("Detector confidence swings widely across frames; intermittent detection is typical of face swaps whose quality varies with pose and lighting." if p > 0.55 else "Detector scores are stable across frames."),
                details=vol,
            )
        )

    # --- forensic: face vs background noise consistency (median over frames) ---
    ratios = [f["forensics"].get("noise_face_bg_ratio") for f in per_frame if f["forensics"].get("noise_face_bg_ratio")]
    if len(ratios) >= 3:
        med = float(np.median(ratios))
        p = 0.40 + 0.35 * logistic(abs(np.log(max(med, 1e-3))), 0.5, 0.15)
        signals.append(
            Signal(
                id="noise_face_bg_video", name="Face/background noise consistency (median over frames)", category="forensic",
                observed=f"median face/background noise ratio={med:.3f} over {len(ratios)} frames",
                reference="Single-source footage: face vs surrounding ring 0.6–1.6. Blended faces carry different noise statistics",
                weight=0.25, **prob_to_signal_fields(p),
                reasoning=("The face region's noise level is persistently inconsistent with the background across the clip." if p > 0.55 else "Noise statistics of face and background agree throughout the clip."),
                details={"ratios": [round(r, 3) for r in ratios]},
            )
        )

    # --- sound track: the voice tests a different hypothesis (voice clone / dub) from the face channel ---
    for a_sig in scores.audio_signals:
        # a face swap normally keeps the genuine audio, so a "bona fide voice" result is weak evidence about the visuals.
        if a_sig.ai_probability is not None and a_sig.ai_probability < 0.5:
            a_sig.weight = round(a_sig.weight * 0.25, 3)
            a_sig.reasoning += " A genuine-sounding voice does not rule out a visual face swap, so this negative carries little weight."
        elif a_sig.ai_probability is not None:
            a_sig.reasoning += " A synthetic voice indicates manipulation even when the faces appear genuine (voice clone or dub)."
    signals += scores.audio_signals

    # --- hypothesis separation: a face swap is localised, so whole-frame "this scene is real" evidence
    # does not refute a positive face-swap finding (it would in a fully synthetic video). ---
    face_sig = next((x for x in signals if x.id == "frames_face"), None)
    gen_sig = next((x for x in signals if x.id == "frames_general"), None)
    if face_sig and gen_sig and face_sig.ai_probability >= 0.5 and gen_sig.ai_probability < 0.5:
        gen_sig.weight = 0.3
        gen_sig.reasoning += " Down-weighted: a localised face manipulation is not refuted by an otherwise genuine scene."

    verdict = fuse(signals, subject="video")
    verdict.reasoning.insert(0, f"{len(per_frame)} frames sampled uniformly over {dur:.1f}s; a face was present in {n_face_frames} of them.")
    if dur and dur < 3:
        verdict.caveats.insert(0, "Very short clip: temporal signals have few observations.")
    small_share = float(np.mean([max((f["w"] for f in fr["faces"]), default=0) < settings.dfdc_min_face_px for fr in per_frame if fr["faces"]])) if n_face_frames else 0.0
    if small_share > 0.8:
        verdict.caveats.insert(0, f"The face is small in the frame (under {settings.dfdc_min_face_px} px in {small_share:.0%} of face frames); face-swap detectors are unreliable at this size, so their evidence is heavily down-weighted and the clip may remain inconclusive.")
        if verdict.confidence == "high":
            verdict.confidence = "medium"
    if 0 < n_face_frames < 6:
        verdict.caveats.insert(0, f"A face was detectable in only {n_face_frames} sampled frame(s); face-swap evidence is down-weighted and confidence is limited.")
        if verdict.confidence == "high":
            verdict.confidence = "medium"
    return {
        "verdict": verdict.model_dump(),
        "signals": [s.model_dump() for s in signals],
        "video": {**info, **{"audio_analysis": scores.audio_info, "key_frame_times": {}}},
        "frames": [
            {"index": i, "time": round(t, 3), "p_frame": f["p_frame"], "p_face": f["p_face"], "p_general": f["p_general"], "faces": len(f["faces"]), "scores": f["scores"]}
            for i, t, f in zip(idxs, times, per_frame)
        ],
        "artifacts": [],
        "timings": dict(timings or {}),
    }


def video_artifacts(art_dir: Path, frames: list[np.ndarray], per_frame: list[dict], times: list[float]) -> tuple[list[str], dict[str, float]]:
    """Timeline, composite sheet, poster (first sampled frame) and individual key frames → (file names, key-frame times)."""
    artifacts = [_save(art_dir, "timeline.png", temporal.timeline_plot(times, per_frame)), _save(art_dir, "key_frames.jpg", temporal.contact_sheet(frames, per_frame, times))]
    poster = frames[0]
    if poster.shape[1] > 960:
        poster = np.asarray(Image.fromarray(poster).resize((960, int(poster.shape[0] * 960 / poster.shape[1]))))
    artifacts.append(_save(art_dir, "poster.jpg", poster))
    key_times = {}
    for name, img, t_ in temporal.key_frame_crops(frames, per_frame, times):
        artifacts.append(_save(art_dir, name, img))
        key_times[name] = round(t_, 2)
    return artifacts, key_times


def attach_artifacts(result: dict, artifacts: list[str], key_times: dict[str, float]) -> dict:
    result["artifacts"] = list(artifacts)
    result["video"]["key_frame_times"] = dict(key_times)
    for s in result["signals"]:
        if s["id"] in ("frames_face", "frames_general"):
            s["artifacts"] = list(FRAME_SIGNAL_ARTIFACTS)
    return result


def analyze_video(path: Path, art_dir: Path) -> dict:
    t0 = time.perf_counter()
    scores, frames, timings = score_video(path)
    result = assemble_video(scores, timings=timings)
    attach_artifacts(result, *video_artifacts(art_dir, frames, scores.per_frame, scores.times))
    result["timings"]["total"] = round(time.perf_counter() - t0, 3)
    return result
