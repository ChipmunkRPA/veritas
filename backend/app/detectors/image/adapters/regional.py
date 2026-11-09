"""Sensitive lossless-image region evidence; optional, with legacy check retained."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from app.detectors.image import localiser as loc
from app.detectors.image import regional
from app.detectors.image.adapters.common import H2, image_spec, original_bytes
from app.detectors.image.adapters.localisers import PNG_SIGNATURE, LocaliserDetector
from app.detectors.image.regional_support import readout
from app.engine.detector import Applicability, InputUnit, RunContext


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
                      model_ids=(MODEL,), devices=("cpu",), lane="cpu", timeout_s=180., version="0.1.0")

    def applicable(self, ctx: RunContext) -> Applicability:
        if not original_bytes(ctx).startswith(PNG_SIGNATURE):
            return Applicability(False, "Sensitive regional screening is currently assessed on PNG images")
        return super().applicable(ctx)

    def predict(self, model: Any, images: list[np.ndarray], device: Any) -> list[np.ndarray]:
        return [model.predict(Image.fromarray(pixels)).raw_mask for pixels in images]

    def localise(self, ctx: RunContext, pixels: np.ndarray) -> loc.LocaliserOutput:
        loaded = ctx.lease.get(self.MODEL)
        prediction = loaded.obj.predict(Image.fromarray(pixels))
        height, width = pixels.shape[:2]
        if prediction.raw_mask.shape != (height, width):
            raise ValueError("Regional response is not registered to this source")
        measured = readout(prediction.raw_mask, prediction.candidate_eligibility["eligible"],
                           self.REGION_THRESHOLD, original_bytes(ctx), pixels)
        return loc.LocaliserOutput(
            anomaly_map=measured.floating, reliable=measured.reliable, reliability_source="conditions",
            integrity_score=loc.integrity_score(measured.floating, measured.reliable), regions=measured.regions,
            region_stats=measured.stats, method=self.METHOD, weights_sha256=str(loaded.revision),
            conditions=measured.conditions, candidate_eligibility=prediction.candidate_eligibility,
            resolution_policy={"mode": "screening", "input_px": self.INPUT,
                               "grid": list(measured.floating.shape[::-1]), "tiles": 0, "overlap_px": 0,
                               "source_cropped": False, "fit_size": prediction.geometry["fit_size"],
                               "canvas_size": [1024, 1024], "padding_position": "top_left",
                               "source_rgb_sha256": prediction.geometry["source_rgb_sha256"],
                               "input_float32_sha256": prediction.geometry["input_float32_sha256"],
                               "native_binding": prediction.binding})

    def run(self, ctx: RunContext):
        result = super().run(ctx)
        if result.status != "ok":
            return result
        count = int(result.metrics["regions"])
        result.observed = f"Sensitive regional screening marked {count} candidate region(s) for closer inspection."
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
        return result
