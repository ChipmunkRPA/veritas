"""Optional source-aligned regional evidence for PNG and a bounded JPEG profile."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from PIL import Image

from app.core.config import settings
from app.detectors.image import jpeg_bitstream, regional
from app.detectors.image import localiser as loc
from app.detectors.image.adapters.common import H2, image_spec, original_bytes
from app.detectors.image.adapters.localisers import PNG_SIGNATURE, LocaliserDetector
from app.detectors.image.regional_support import readout
from app.engine.detector import Applicability, InputUnit, RunContext
from app.evidence.image_eligibility import FLIP_CONSISTENCY

JPEG_SIGNATURE = b"\xff\xd8"


def evaluated_jpeg(data: bytes) -> bool:
    structure = jpeg_bitstream.read(data)
    return bool(structure and structure.frame in {"baseline", "progressive"} and structure.precision == 8
                and max(structure.width, structure.height) == 768 and min(structure.width, structure.height) >= 512
                and [(c.id, c.h, c.v) for c in structure.components] == [(1, 2, 2), (2, 1, 1), (3, 1, 1)]
                and all(structure.table_bits.get(c.table_id) == 8 for c in structure.components)
                and structure.ijg_quality == 75 and structure.table(2) == structure.table(1)
                and (structure.frame == "baseline" or jpeg_bitstream.progressive_frame(data) is not None))


def flip_components(measured, aligned):
    """Keep original top-two component geometry when the repeat supports half its pixels."""
    rows, retained, stats = [], [], []
    for region, stat, component in zip(measured.regions[:2], measured.stats[:2], measured.component_masks, strict=True):
        pixels = int(np.count_nonzero(component))
        overlap = int(np.count_nonzero(component & aligned))
        supported = overlap * 2 >= pixels
        box = region["region"]["bbox"]
        rows.append({"region_id": region["display_id"], "bounds": [int(box[k]) for k in ("x", "y", "w", "h")],
                     "source_pixels": pixels, "shared_pixels": overlap, "retained": supported})
        if supported:
            retained.append(region)
            stats.append(stat)
    review = {"schema": FLIP_CONSISTENCY, "candidates": rows, "inspected_components": len(rows),
              "supported_components": len(retained), "withheld_components": len(rows) - len(retained),
              "aligned_mask_sha256": hashlib.sha256(aligned.tobytes()).hexdigest()}
    return retained, stats, review


class RegionalDetector(LocaliserDetector):
    MODEL = regional.MODEL_ID
    METHOD = "Sensitive regional screening"
    INPUT = 1024
    SUPPORTS_TILES = False
    REGION_THRESHOLD = regional.PIXEL_THRESHOLD
    REFERENCE = ("Local comparison found more edited regions and more false alarms on untouched images than the "
                 "previous check. Highlights are editing cues; AI authorship remains inconclusive from this check alone. "
                 "Unmarked areas are not confirmed original. The response is not an authorship confidence score.")
    spec = image_spec("loc.native_regional", METHOD, "localiser", InputUnit.IMAGE, (H2,), ("heatmap", "regions"),
                      model_ids=(MODEL,), devices=("cpu",), lane="cpu", timeout_s=180., version="0.1.1")

    def applicable(self, ctx: RunContext) -> Applicability:
        data = original_bytes(ctx)
        jpeg_review = settings.environment == "dev" and settings.image_region_jpeg_review
        if data.startswith(PNG_SIGNATURE):
            if jpeg_review and settings.image_region_mode == "legacy":
                from app.engine.registry import get_registry

                if self.spec.id not in get_registry().policy.allow_detectors:
                    return Applicability(False, "The optional regional review is enabled for eligible JPEG images")
            return super().applicable(ctx)
        if jpeg_review and evaluated_jpeg(data):
            info = ctx.media.info()
            if info["mode"] == "RGB" and info["exif_orientation"] == 1:
                return super().applicable(ctx)
        return Applicability(False, "This local editing check covers PNG and a limited JPEG review profile")

    def predict(self, model: Any, images: list[np.ndarray], device: Any) -> list[np.ndarray]:
        return [model.predict(Image.fromarray(pixels)).raw_mask for pixels in images]

    def localise(self, ctx: RunContext, pixels: np.ndarray) -> loc.LocaliserOutput:
        loaded = ctx.lease.get(self.MODEL)
        prediction = loaded.obj.predict(Image.fromarray(pixels))
        height, width = pixels.shape[:2]
        if prediction.raw_mask.shape != (height, width):
            raise ValueError("Regional response is not registered to this source")
        data = original_bytes(ctx)
        jpeg = data.startswith(JPEG_SIGNATURE)
        measured = readout(prediction.raw_mask, prediction.candidate_eligibility["eligible"],
                           self.REGION_THRESHOLD, data, pixels)
        regions, stats = measured.regions, measured.stats
        repeat = {}
        if (not jpeg and settings.environment == "dev" and settings.image_region_flip_review and regions
                and max(width, height) == 768 and min(width, height) >= 512):
            ctx.deadline.check("regional repeatability check")
            reflected = np.ascontiguousarray(pixels[:, ::-1])
            reflected_prediction = loaded.obj.predict(Image.fromarray(reflected))
            reflected_readout = readout(reflected_prediction.raw_mask, reflected_prediction.candidate_eligibility["eligible"],
                                        self.REGION_THRESHOLD, data, reflected)
            aligned = np.ascontiguousarray(reflected_readout.selected_grid_mask[:, ::-1])
            regions, stats, review = flip_components(measured, aligned)
            proof = ctx.exhibits.binary_mask(aligned)
            repeat = {
                "flip_consistency": {**review, "aligned_mask_png_sha256": proof.sha256,
                                     "reflected_input_sha256": reflected_prediction.geometry["input_float32_sha256"]},
            }
        return loc.LocaliserOutput(
            anomaly_map=measured.floating, reliable=measured.reliable, reliability_source="conditions",
            integrity_score=loc.integrity_score(measured.floating, measured.reliable), regions=regions,
            region_stats=stats, method=self.METHOD, weights_sha256=str(loaded.revision),
            conditions=measured.conditions, candidate_eligibility=prediction.candidate_eligibility,
            resolution_policy={"mode": "screening", "input_px": self.INPUT,
                               "grid": list(measured.floating.shape[::-1]), "tiles": 0, "overlap_px": 0,
                               "source_cropped": False, "fit_size": prediction.geometry["fit_size"],
                               "canvas_size": [1024, 1024], "padding_position": "top_left",
                               "source_rgb_sha256": prediction.geometry["source_rgb_sha256"],
                               "input_float32_sha256": prediction.geometry["input_float32_sha256"],
                               **({"source_format": "JPEG"} if jpeg else {}),
                               "native_binding": prediction.binding, **repeat})

    def run(self, ctx: RunContext):
        result = super().run(ctx)
        if result.status != "ok":
            return result
        count = int(result.metrics["regions"])
        result.observed = f"The local editing check marked {count} candidate region(s) for closer inspection."
        result.reasoning = (
            ("Local editing cues were found in the highlighted areas. " if count else
             "No region met this check's selection rules. This does not establish that the file is original. ")
            + "The complete image was fitted without stretching onto a padded 1024-pixel canvas. "
              "Highlights are aligned to the oriented source and exclude areas where measurement is less reliable. "
              "Ordinary edits and untouched images can produce similar cues. AI authorship is inconclusive from "
              "this check alone, and its uncalibrated response does not change the overall authorship confidence.")
        for exhibit in ctx.exhibits.exhibits:
            if exhibit.id not in result.exhibit_ids:
                continue
            exhibit.caption = ("Highlighted pixels are candidates for local editing, aligned to the oriented original. "
                               "The image was fitted without stretching and padded; padding is excluded. "
                               "These uncalibrated cues cannot determine who or what edited the image.")
        result.details["authorship_confidence_contribution"] = 0.
        result.details["localisation"]["validation_status"] = "limited_local_evaluation"
        policy = result.details["localisation"]["resolution_policy"]
        if policy.get("source_format") == "JPEG":
            result.reasoning += " JPEG compression can obscure small or softly blended edits."
        if (review := policy.get("flip_consistency")) is not None:
            result.reasoning += (f" A horizontally flipped repeat pass supported {review['supported_components']} "
                                 f"of {review['inspected_components']} inspected regions on the original pixel grid.")
        return result
