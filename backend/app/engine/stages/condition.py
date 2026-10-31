"""`condition` stage (ARCHITECTURE §3.1; PRD EA-02): condition profile, slice id and fusion context of one input.

Output: `{"profile": ConditionProfile, "context": {face_px, jpeg_q, min_side, has_camera_exif, flat_block_share,
greyscale}}`. The profile's `within_validated_envelope` stays false here; `fuse` sets it from the loaded artefact's slice
metrics, its size floor and its content rule (D-65: the last two context fields).

- **image.** Original size (probe); estimated JPEG quality from the luminance quantisation table read from the bitstream
  (`jpeg_bitstream`, I-1; the v0.3 estimate, so it equals the manifest's `jpeg_q_est`; null for non-JPEG files); face sizes from `faces` in original pixels (the longer side of each box × the analysis-copy scale);
  the screenshot rule. `has_camera_exif` is "any EXIF", the benchmark manifest's `has_exif` that `calibrate.py` uses.
- **video.** Frame size, duration and frame count from ffprobe; face sizes over the sampled frames; sound-track state.
- **document.** Word count.
- **Not established yet** (null): double compression (I-4), speech and music share (A-1), language, upscaling.
"""
from __future__ import annotations

import io
import statistics
from typing import Any

from PIL import Image

from app.core.logging import get_logger
from app.detectors.audio import content as audio_content
from app.detectors.image import jpeg_bitstream
from app.engine.fusion import abstain, content_kinds
from app.engine.runner import StageContext
from app.engine.stages import common
from app.schemas.evidence import ConditionProfile

log = get_logger(__name__)


def _sizes(boxes: list[dict[str, Any]], scale: float = 1.0) -> list[int]:
    return sorted(int(round(max(float(b["w"]), float(b["h"])) * scale)) for b in boxes if b.get("w") and b.get("h"))


def _face_fields(sizes: list[int]) -> dict[str, int | None]:
    if not sizes:
        return {"min_face_px": None, "median_face_px": None}
    return {"min_face_px": sizes[0], "median_face_px": int(round(statistics.median(sizes)))}


def _profile(**fields: Any) -> dict[str, Any]:
    return ConditionProfile.model_validate({"within_validated_envelope": False, **fields}).model_dump(mode="json")


def image(ctx: StageContext) -> dict[str, Any]:
    probed, decoded = ctx.outputs.get("probe") or {}, ctx.outputs.get("decode") or {}
    data = bytes(common.media(ctx).original_bytes())
    with Image.open(io.BytesIO(data)) as im:
        fmt = im.format
        width, height = int(probed.get("width") or im.width), int(probed.get("height") or im.height)
        has_exif = bool(probed["has_exif"]) if "has_exif" in probed else len(im.getexif()) > 0
        orientation = int(probed["exif_orientation"] if "exif_orientation" in probed else im.getexif().get(0x0112, 1) or 1)
    # N-4: the profile holds the size in EXIF-oriented pixels, the space of the detectors, the exhibits and the viewer. The
    # shorter side, the slice, the screenshot rule and the size floor read the same from either orientation.
    if decoded.get("width") and decoded.get("height"):
        width, height = int(decoded["width"]), int(decoded["height"])
    elif orientation in (5, 6, 7, 8):
        width, height = height, width
    scale = float(decoded.get("scale_to_original") or 1.0)
    sizes = _sizes((ctx.outputs.get("faces") or {}).get("faces") or [], scale)
    face_px = sizes[-1] if sizes else None
    jpeg = jpeg_bitstream.read(data) if fmt == "JPEG" else None
    q = jpeg.quality_luma if jpeg is not None else None
    screenshot = abstain.screenshot_likelihood(fmt, width, height, has_exif)
    kinds = _content_kinds(data)
    profile = _profile(width=width, height=height, jpeg_quality_est=q, screenshot_likelihood=screenshot, **_face_fields(sizes),
                       slice_id=abstain.image_slice(width, height, q, face_px),
                       limiting_conditions=abstain.image_conditions(width, height, q, face_px, screenshot))
    return {"profile": profile,
            "context": {"face_px": face_px, "jpeg_q": q, "min_side": min(width, height), "has_camera_exif": has_exif, **kinds}}


def _content_kinds(data: bytes) -> dict[str, Any]:
    """D-65 (N-18): the flat-block share and the greyscale test of the file as it decodes at full size, measured as the
    bench's `fit` and `cal` images were (`content_kinds.measure`). A file that cannot be decoded in full has neither,
    and the envelope then has no content rule to apply to it."""
    try:
        measured = content_kinds.measure(data)
    except Exception as e:  # noqa: BLE001  the decode stage reports a file that cannot be read; this stage measures what it can
        log.warning("condition.content_kinds_failed", error=f"{type(e).__name__}: {e}"[:200])
        return {"flat_block_share": None, "greyscale": None}
    return {"flat_block_share": measured["flat_block_share"], "greyscale": measured["greyscale"]}


def video(ctx: StageContext) -> dict[str, Any]:
    info = (ctx.outputs.get("probe") or {}).get("info") or {}
    decoded = ctx.outputs.get("decode") or {}
    width = int(info.get("width") or decoded.get("width") or 0) or None
    height = int(info.get("height") or decoded.get("height") or 0) or None
    duration = float(info["duration"]) if info.get("duration") else None
    n_frames = int(info["nb_frames"]) if info.get("nb_frames") else None
    faces_per_frame = (ctx.outputs.get("faces") or {}).get("faces") or []
    sizes = _sizes([box for frame in faces_per_frame for box in frame])
    face_px = sizes[-1] if sizes else None
    audio = (decoded.get("audio") or {}).get("status")
    profile = _profile(width=width, height=height, duration_s=duration, n_frames=n_frames, **_face_fields(sizes),
                       slice_id=abstain.video_slice(height, duration, face_px),
                       limiting_conditions=abstain.video_conditions(height, duration, face_px, audio,
                                                                   audio_decoded_s=(decoded.get("audio") or {}).get("seconds")))
    min_side = min(width, height) if width and height else None
    return {"profile": profile, "context": {"face_px": face_px, "jpeg_q": None, "min_side": min_side, "has_camera_exif": False}}


def document(ctx: StageContext) -> dict[str, Any]:
    words = (ctx.outputs.get("decode") or {}).get("words")
    words = int(words) if words is not None else None
    profile = _profile(slice_id=abstain.document_slice(words), limiting_conditions=abstain.document_conditions(words))
    return {"profile": profile, "context": {}}


def audio(ctx: StageContext) -> dict[str, Any]:
    """Speech / music / noise / silence per second (`detectors/audio/content.py`), the shares, the speech spans that gate
    the voice windows and the assessability of the recording. Critical in the audio pipeline: `detect` reads the gate."""
    facts = (ctx.outputs.get("probe") or {}).get("facts") or {}
    decoded = ctx.outputs.get("decode") or {}
    track = ctx.state.get("audio")
    content = audio_content.classify(track.pcm if track is not None else None).as_json()
    duration = float(facts["duration_s"]) if facts.get("duration_s") else content["duration_s"]
    shares = content["shares"]
    lossless = bool(facts.get("lossless"))
    profile = _profile(duration_s=duration, speech_share=shares["speech"], music_share=shares["music"],
                       slice_id=abstain.audio_slice(duration),
                       limiting_conditions=abstain.audio_conditions(duration, shares, codec=facts.get("codec"), lossless=lossless,
                                                                    analysed_s=decoded.get("seconds")))
    context = {"face_px": None, "jpeg_q": None, "min_side": None, "has_camera_exif": False, "duration_s": duration,
               "codec_lossy": not lossless, "speech_share": shares["speech"]}
    return {"profile": profile, "context": context, "content": content}
