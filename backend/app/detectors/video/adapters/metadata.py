"""Video container provenance as v2 plugins (model-free, CPU).

- **`vid.meta.container`**: the v0.3 container metadata rule (`video/pipeline.metadata_signals`). Capture-device container
  tags (QuickTime make/model, Android manufacturer/version) are weak authentic-leaning evidence (p 0.30, weight 0.4 in
  v0.3); a generic encoder without device tags is uninformative. Tags are forgeable.
- **`prov.c2pa.video`** (I-2): offline C2PA validation of MP4 / MOV (ISO BMFF) containers, with the same states, exhibits
  and D-15 strengths as the image detector (`image/adapters/provenance.c2pa_result`).
"""
from __future__ import annotations

from app.detectors.image.adapters.provenance import C2PA_PRODUCES, C2PA_VERSION, c2pa_result
from app.detectors.video.adapters.common import CPU_ONLY, H3, video_spec
from app.detectors.video.pipeline import metadata_signals
from app.engine.detector import Detector, DetectorResult, InputUnit, RunContext, UnitScore


class ContainerMetadataDetector(Detector):
    spec = video_spec("vid.meta.container", "Container / encoder and capture-device tags (ffprobe)", InputUnit.FILE, (H3,), ("table",),
                      family="provenance", needs=("probe",), lane="cpu", devices=CPU_ONLY, timeout_s=10.0,
                      version="0.3.1")  # 0.3.1 (N-19p, R10-03): its table keeps the words of a tag apart (tab and newline as a space)

    def run(self, ctx: RunContext) -> DetectorResult:
        info = ctx.outputs.probe or {}
        [signal] = metadata_signals(info)
        tags = info.get("tags") or {}
        exhibit_ids = []
        if tags:
            exhibit_ids.append(ctx.exhibits.table(["tag", "value"], [[k, str(v)] for k, v in sorted(tags.items())],
                                                  title="Container tags", media_ref="original"))
        device_tags = signal.id == "video_device_tags"
        aggregate = UnitScore("global", 0.30, "probability", quality={"outcome_weight": 0.4}) if device_tags else None
        return DetectorResult(
            status="ok", aggregate=aggregate, unit_scores=[aggregate] if aggregate else [], observed=signal.observed,
            reference=signal.reference, reasoning=signal.reasoning,
            metrics={"device_tags": float(len(signal.details)) if device_tags else 0.0, "has_audio": float(bool(info.get("has_audio")))},
            exhibit_ids=exhibit_ids)


class C2PAVideoDetector(Detector):
    spec = video_spec("prov.c2pa.video", "C2PA Content Credentials (video containers)", InputUnit.FILE, (H3,), C2PA_PRODUCES,
                      family="provenance", needs=(), lane="cpu", devices=CPU_ONLY, timeout_s=30.0, version=C2PA_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        return c2pa_result(ctx, ctx.media.original_bytes())
