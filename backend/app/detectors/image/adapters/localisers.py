"""Localiser adapters (IF §3.2, §3.3; D-29): the `LocaliserDetector` base, `loc.mesorch` and `loc.radar`.

- **Run.** Native pixels (EXIF-oriented) → a screening pass over the whole image resized to the model input → in the deep
  profile, native tiles of the input size with 64 px overlap when both sides reach the input size → the conditions mask on
  native pixels → regions → exhibits. `app.detectors.image.localiser` holds the geometry, mask, regions and score.
- **Exhibits.** A `p_manipulated` heatmap (uncalibrated, value range 0–1, on an aspect-correct grid) whose reliability mask
  is the conditions mask, and a `regions` exhibit in native px when a region survives. The localise stage does not segment
  the map again.
- **Scores.** The aggregate is the integrity score (raw kind `score`, uncalibrated). Each region is a `region:<k>` unit
  with its mean map value.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import jpeg_bitstream, mesorch, radar
from app.detectors.image import localiser as loc
from app.detectors.image.adapters.common import H2, image_spec, original_bytes, revisions
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    Profile,
    RunContext,
    UnitScore,
)
from app.evidence.image_eligibility import validate_gate


class LocaliserDetector(Detector):
    """A localisation model behind the `Localiser` interface. Subclasses name the model and implement `predict`."""

    MODEL: ClassVar[str]
    METHOD: ClassVar[str]
    INPUT: ClassVar[int]
    OVERLAP: ClassVar[int] = 64
    TILE_BATCH: ClassVar[int] = 4
    MIN_SIDE: ClassVar[int] = 64
    REFERENCE: ClassVar[str] = ""
    SCREENING_RESIZE: ClassVar[str] = "resized"  # how the screening pass fits the whole image to the model input
    SUPPORTS_TILES: ClassVar[bool] = True

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        count = self.TILE_BATCH if self.SUPPORTS_TILES and ctx.profile == Profile.DEEP else 1
        description = (f"One screening input; deep tiles batch at most {self.TILE_BATCH}, each {self.INPUT} px"
                       if self.SUPPORTS_TILES else f"One {self.INPUT} × {self.INPUT} screening input; no tile pass")
        return (ActivationUse(self.MODEL, count, "localiser input",
                              description,
                              ("Whole-image decoding, retained tile maps and host-side blending are not a measured peak envelope.",)),)

    @abstractmethod
    def predict(self, model: Any, images: list[np.ndarray], device: Any) -> list[np.ndarray]:
        """RGB uint8 images → maps in [0, 1] registered to each whole image (tiles: `INPUT` × `INPUT` maps)."""

    def localise(self, ctx: RunContext, pixels: np.ndarray) -> loc.LocaliserOutput:
        model = ctx.lease.get(self.MODEL)
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        tiled = ctx.profile == Profile.DEEP and min(width, height) >= self.INPUT and max(width, height) > self.INPUT
        grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE["tiled" if tiled else "screening"])
        screening = cv2.resize(self.predict(model.obj, [pixels], ctx.device)[0].astype(np.float32), grid,
                               interpolation=cv2.INTER_LINEAR)
        values, n_tiles = screening, 0
        if tiled:
            origins = [(x, y) for y in loc.tile_origins(height, self.INPUT, self.OVERLAP)
                       for x in loc.tile_origins(width, self.INPUT, self.OVERLAP)]
            tiles = []
            for i in range(0, len(origins), self.TILE_BATCH):
                chunk = origins[i:i + self.TILE_BATCH]
                crops = [np.ascontiguousarray(pixels[y:y + self.INPUT, x:x + self.INPUT]) for x, y in chunk]
                tiles += [(x, y, m) for (x, y), m in zip(chunk, self.predict(model.obj, crops, ctx.device), strict=True)]
            values, n_tiles = loc.to_grid(loc.blend_tiles(tiles, width, height, self.OVERLAP), grid), len(tiles)
        structure = jpeg_bitstream.read(original_bytes(ctx))
        unreliable, shares = loc.conditions_mask(pixels, structure.quality_luma if structure else None)
        reliable = loc.reliable_grid(unreliable, grid)
        integrity = loc.integrity_score(values, reliable)
        if tiled:
            integrity = max(integrity, loc.integrity_score(screening, reliable))
        # The persisted heatmap is an 8-bit linear PNG. Segment that exact grid
        # so later evidence can reproduce the component boundaries without
        # rerunning a model or silently changing threshold membership.
        recorded_values = np.rint(np.clip(values.astype(np.float64), 0, 1) * 255) / 255.0
        regions, stats = loc.components(recorded_values, reliable, width, height)
        policy = {"mode": "tiled" if tiled else "screening", "input_px": self.INPUT, "grid": list(grid),
                  "scale_x": round(self.INPUT / width, 4), "scale_y": round(self.INPUT / height, 4), "tiles": n_tiles,
                  "overlap_px": self.OVERLAP if tiled else 0}
        return loc.LocaliserOutput(anomaly_map=values, reliable=reliable, reliability_source="conditions",
                                   integrity_score=integrity, regions=regions, region_stats=stats, method=self.METHOD,
                                   weights_sha256=str(model.revision), resolution_policy=policy, conditions=shares)

    def _small(self) -> str:
        return f"image below {self.MIN_SIDE} px"

    def applicable(self, ctx: RunContext) -> Applicability:
        """The size rule needs no model, so it is asked before the lease (BUILD_SPEC N-14c, E-10)."""
        height, width = ctx.media.native_pixels().shape[:2]
        return Applicability(False, self._small()) if min(int(width), int(height)) < self.MIN_SIDE else Applicability(True)

    def run(self, ctx: RunContext) -> DetectorResult:
        pixels = ctx.media.native_pixels()
        height, width = int(pixels.shape[0]), int(pixels.shape[1])
        if min(width, height) < self.MIN_SIDE:
            return DetectorResult.not_applicable(self._small())
        out = self.localise(ctx, pixels)
        grid_w, grid_h = out.resolution_policy["grid"]
        tiled = out.resolution_policy["mode"] == "tiled"
        pass_text = (f"native {self.INPUT} px tiles ({out.resolution_policy['tiles']}, {self.OVERLAP} px overlap) plus a "
                     f"screening pass" if tiled else
                     f"a screening pass on the whole {width}×{height} px image {self.SCREENING_RESIZE} to "
                     f"{self.INPUT}×{self.INPUT} px")
        selection_note = ("A separate whole-image selection rule can withhold all candidate regions; raw map values and "
                          "quality exclusions remain unchanged. " if out.candidate_eligibility is not None else "")
        exhibit_ids = [ctx.exhibits.heatmap(
            out.anomaly_map, semantics="p_manipulated", calibrated=False, value_range=(0.0, 1.0), mask=out.reliable,
            title=f"{self.METHOD} manipulation map",
            caption=f"Per-pixel output of {self.METHOD} in [0, 1], uncalibrated, from {pass_text}. Areas at or above "
                    f"{loc.THRESHOLD} outside the less reliable areas form candidate regions. "
                    f"{selection_note}Masked: {loc.CONDITIONS_LABEL.lower()}.",
            media_ref="original", coordinate_space={"width": width, "height": height, "scale_to_original": 1.0})]
        candidate_gate = None
        if out.candidate_eligibility is not None:
            heat = next(e for e in ctx.exhibits.exhibits if e.id == exhibit_ids[0])
            candidate_gate = {**out.candidate_eligibility, "source_sha256": ctx.media.sha256,
                              "scalar_map_sha256": heat.sha256,
                              "score_exact_decimal": repr(float.fromhex(out.candidate_eligibility["score_hex"])),
                              "threshold_exact_decimal": repr(float.fromhex(out.candidate_eligibility["threshold_hex"]))}
            validate_gate(candidate_gate, source_sha256=ctx.media.sha256, scalar_map_sha256=heat.sha256)
            if not candidate_gate["eligible"] and out.regions:
                raise ValueError("An ineligible image cannot publish selected localisation regions")
        if out.regions:
            exhibit_ids.append(ctx.exhibits.regions(
                out.regions, title=f"Regions flagged by {self.METHOD}", media_ref="original",
                caption=f"Connected areas where the {self.METHOD} map reaches {loc.THRESHOLD} outside the less reliable areas: "
                        "regions flagged for examination, not a mask of what was changed.",
                coordinate_space={"width": width, "height": height, "scale_to_original": 1.0}))
        n = len(out.regions)
        aggregate = UnitScore("global", round(out.integrity_score, 6), "score",
                              quality={"regions": float(n), "unreliable_share": out.conditions["unreliable"],
                                       "tiles": float(out.resolution_policy["tiles"]),
                                       "screening_scale": round(self.INPUT / max(width, height), 4)})
        units = [UnitScore(f"region:{k}", stat["mean"], "score", quality={"area_share": stat["area_share"], "max": stat["max"]})
                 for k, stat in enumerate(out.region_stats, 1)]
        flagged = float((out.anomaly_map >= loc.THRESHOLD).mean())
        largest = (f"; R1 mean {out.region_stats[0]['mean']:.2f} over {100 * out.region_stats[0]['area_share']:.1f} % of the "
                   "image" if n else "")
        screening_note = ("" if tiled else f" The map was computed at screening resolution (×{self.INPUT / max(width, height):.2f}"
                          " on the long side), which attenuates fine compression and noise traces; Deep analysis adds "
                          "native-resolution tiles." if self.SUPPORTS_TILES else
                          f" The whole image was inspected at {self.INPUT} × {self.INPUT}; native-resolution tiles are not available for this check.")
        reasoning = (f"The localisation model marks {n} area(s) for examination where its map reaches the {loc.THRESHOLD} "
                     "operating threshold outside the less reliable areas." if n else
                     f"The localisation model marks no area at its {loc.THRESHOLD} operating threshold outside the less "
                     "reliable areas. A fully generated image also gives an empty or diffuse map, so this says nothing "
                     "about generation.")
        if candidate_gate is not None:
            reasoning = (f"The whole-image response {candidate_gate['score_exact_decimal']} "
                         f"{'reached' if candidate_gate['eligible'] else 'did not reach'} the fixed selection threshold "
                         "0.5. This uncalibrated response selects candidates; it does not validate any pixel as edited. "
                         + (reasoning if candidate_gate["eligible"] else
                            "Local map regions were withheld by this gate. The original response and quality mask are retained; this is not a finding of authenticity."))
        return DetectorResult(
            status="ok", unit_scores=units or [aggregate], aggregate=aggregate,
            observed=f"integrity score {out.integrity_score:.3f} (mean of the top 1 % of reliable map values); {n} region(s) "
                     f"at map ≥ {loc.THRESHOLD}{largest}; {100 * out.conditions['unreliable']:.1f} % of pixels in less "
                     f"reliable areas; {pass_text}",
            reference=self.REFERENCE, reasoning=reasoning + screening_note,
            metrics={"integrity": out.integrity_score, "map_max": float(out.anomaly_map.max()), "flagged_share": flagged,
                     "regions": float(n), "tiles": float(out.resolution_policy["tiles"]), "grid_w": float(grid_w),
                     "grid_h": float(grid_h), **{f"{k}_share": v for k, v in out.conditions.items()}},
            exhibit_ids=exhibit_ids, model_revisions=revisions(ctx, self.MODEL),
            details={"localisation": {
                "schema": "recorded-mask/1", "threshold": loc.THRESHOLD,
                "comparison": "greater_or_equal", "scalar_encoding": "linear_uint8",
                "component_connectivity": 8, "minimum_region_fraction": loc.MIN_REGION_FRACTION,
                "grid": [grid_w, grid_h], "coordinate_space": "exif-oriented-original",
                "resolution_policy": out.resolution_policy, "conditions": out.conditions,
                "reliability_source": out.reliability_source,
                "capability": "ai_inpainting" if self.MODEL.startswith(("radar.", "fused.")) else "local_editing",
                "validation_status": "limited_local_evaluation" if candidate_gate else "local_evaluation_pending",
                **({"candidate_eligibility": candidate_gate} if candidate_gate is not None else {}),
            }})


class FusedDetector(LocaliserDetector):
    """Independently implemented single-screening map with explicit image selection."""

    MODEL = "fused.opensdi"
    METHOD = "FUSED"
    INPUT = 512
    TILE_BATCH = 1
    SUPPORTS_TILES = False
    REFERENCE = ("An uncalibrated local editing response from a 512-square screening input; fixed image selection "
                 "threshold 0.5 and recorded component threshold 0.5. Local independent-sample evaluation found "
                 "missed edits and authentic-image false alarms. Unmarked pixels are not confirmed authentic.")
    spec = image_spec("loc.fused", "FUSED AI-inpainting localisation", "localiser", InputUnit.IMAGE,
                      (H2,), ("heatmap", "regions"), model_ids=(MODEL,), devices=("mps", "cpu"),
                      timeout_s=180.0, version="0.1.0")

    def predict(self, model: Any, images: list[np.ndarray], device: Any) -> list[np.ndarray]:
        if len(images) != 1:
            raise ValueError("This local inspection check supports one screening image at a time")
        return [model.predict(Image.fromarray(images[0])).raw_mask]

    def localise(self, ctx: RunContext, pixels: np.ndarray) -> loc.LocaliserOutput:
        model = ctx.lease.get(self.MODEL)
        prediction = model.obj.predict(Image.fromarray(pixels))
        validate_gate(prediction.candidate_eligibility)
        height, width = pixels.shape[:2]
        grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE["screening"])
        native = np.asarray(prediction.raw_mask)
        if (native.shape != (self.INPUT, self.INPUT) or not np.isfinite(native).all()
                or native.min() < 0 or native.max() > 1):
            raise ValueError("The image localisation response has invalid dimensions or values")
        values = cv2.resize(native.astype(np.float32), grid, interpolation=cv2.INTER_LINEAR)
        structure = jpeg_bitstream.read(original_bytes(ctx))
        unreliable, shares = loc.conditions_mask(pixels, structure.quality_luma if structure else None)
        reliable = loc.reliable_grid(unreliable, grid)
        recorded = np.rint(np.clip(values.astype(np.float64), 0, 1) * 255) / 255.0
        regions, stats = loc.components(recorded, reliable, width, height)
        if not prediction.candidate_eligibility["eligible"]:
            regions, stats = [], []
        return loc.LocaliserOutput(
            anomaly_map=values, reliable=reliable, reliability_source="conditions",
            integrity_score=loc.integrity_score(values, reliable), regions=regions, region_stats=stats,
            method=self.METHOD, weights_sha256=str(model.revision), conditions=shares,
            resolution_policy={"mode": "screening", "input_px": self.INPUT, "grid": list(grid),
                               "scale_x": round(self.INPUT / width, 4), "scale_y": round(self.INPUT / height, 4),
                               "tiles": 0, "overlap_px": 0},
            candidate_eligibility=dict(prediction.candidate_eligibility))


class MesorchDetector(LocaliserDetector):
    """Mesorch (AAAI 2025) on the whole image at 512 px, plus native 512 px tiles in the deep profile."""

    MODEL = "mesorch.full_98"
    METHOD = "Mesorch"
    INPUT = mesorch.INPUT
    REFERENCE = ("map in [0, 1] from a ConvNeXt-T and SegFormer MiT-B3 hybrid over RGB and DCT frequency bands, trained on "
                 "spliced, copy-moved and removed regions (CASIAv2, FantasticReality, IMD2020, tampCOCO); operating threshold "
                 "0.5, the fixed threshold of the upstream pixel-F1 protocol")
    spec = image_spec("loc.mesorch", "Mesorch (ConvNeXt-T + SegFormer MiT-B3, DCT bands)", "localiser", InputUnit.IMAGE,
                      (H2,), ("heatmap", "regions"), model_ids=(MODEL,), timeout_s=120.0, version="0.4.2")

    def predict(self, model: Any, images: list[np.ndarray], device: Any) -> list[np.ndarray]:
        return list(mesorch.predict(model, mesorch.to_batch(images), device))


class RadarDetector(LocaliserDetector):
    """RADAR (NeurIPS 2025) on the whole image padded to a square at 896 px, plus native 896 px tiles in the deep profile."""

    MODEL = "radar.fusion_15k"
    METHOD = "RADAR"
    INPUT = radar.INPUT
    TILE_BATCH = 2
    SCREENING_RESIZE = "edge-padded to a square and resized"
    REFERENCE = ("map in [0, 1] from DINOv2 ViT-B/14 and Depth Anything V2 Base patch features fused by cross-attention, "
                 "trained on BBC-PAIR images inpainted by diffusion models; operating threshold 0.5, the paper's binarisation "
                 "threshold")
    spec = image_spec("loc.radar", "RADAR (DINOv2 ViT-B/14 + Depth Anything V2 Base, diffusion inpainting)", "localiser",
                      InputUnit.IMAGE, (H2,), ("heatmap", "regions"), model_ids=(MODEL,), timeout_s=180.0, version="0.4.2")

    def predict(self, model: Any, images: list[np.ndarray], device: Any) -> list[np.ndarray]:
        return radar.predict(model, images, device)
