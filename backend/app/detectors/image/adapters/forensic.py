"""Classical forensics (v0.3 heuristics) as v2 plugins: frequency spectrum, error level analysis, noise residual.

- **Inputs (I-1, D13).** Native pixels and the original bytes (`native_forensics`); never the analysis copy. Adapter
  version 0.4.0: v0.3 measured on a ≤ 1024 px resample, which the v1 pipeline keeps as the B-5 baseline.
- **Scores.** The v0.3 logistic mappings, capped to [0.40, 0.75]. They are reported as `raw_kind="probability"` and are
  uncalibrated.
- **Exhibits.** The ELA and noise maps are heatmap exhibits (`error_level`, `residual`) registered to the original. The
  spectrum figure is an image exhibit.
"""
from __future__ import annotations

from app.detectors.image import (
    copy_move,
    forensics,
    jpeg_dq,
    local_jpeg_grid,
    native_forensics,
    sharp_boundary,
)
from app.detectors.image.adapters.common import (
    ALL_PROFILES,
    CPU_ONLY,
    H1,
    H2,
    face_boxes,
    image_spec,
    original_bytes,
    png_bytes,
)
from app.engine.detector import Detector, DetectorResult, InputUnit, RunContext, UnitScore

NATIVE_VERSION = "0.4.0"


def _metrics(m: dict) -> dict[str, float]:
    return {k: float(v) for k, v in m.items() if isinstance(v, int | float)}


def _face_box(ctx: RunContext, shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    faces = face_boxes(ctx)
    return native_forensics.native_box(faces[0].box(), shape) if faces else None


class SpectrumDetector(Detector):
    spec = image_spec("img.forensic.spectrum", "Frequency-domain spectrum analysis", "forensic", InputUnit.IMAGE, (H1,),
                      ("image",), lane="cpu", devices=CPU_ONLY, timeout_s=30.0, version=NATIVE_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        pixels = ctx.media.native_pixels()
        if min(pixels.shape[:2]) < native_forensics.MIN_SPECTRUM_PX:
            return DetectorResult(status="not_applicable", reason=f"image smaller than {native_forensics.MIN_SPECTRUM_PX} px")
        m, figure = native_forensics.spectrum(pixels)
        assert figure is not None
        p, _p_peaks, p_slope = forensics.spectrum_score(m)
        exhibit = ctx.exhibits.image(png_bytes(figure), role="spectrum", media_type="image/png",
                                     title="Frequency spectrum and radial profile",
                                     caption=f"Mean over {m['spectrum_tiles']} native {m['spectrum_tile_px']} px tile(s), no resampling. "
                                             "Periodic peaks or an abnormal high-frequency tail can indicate generator up-sampling.")
        aggregate = UnitScore("global", p, "probability")
        return DetectorResult(
            status="ok", unit_scores=[aggregate], aggregate=aggregate,
            observed=f"slope={m['spectral_slope']}, periodic peaks={m['periodic_peaks']} (max {m['peak_strength_sigma']}σ), "
                     f"HF ratio={m['hf_ratio']}",
            reference="Camera images: smooth ~1/f spectrum (slope ≈ −0.8…−1.4), no peaks >3σ in the high band",
            reasoning=("Periodic spectral peaks consistent with generator up-sampling grids were detected" if m["periodic_peaks"] > 0
                       else "No periodic up-sampling peaks were detected")
                      + (" and the spectral slope departs from the natural-image range." if p_slope > 0.6
                         else "; the spectral slope is within the natural-image range."),
            metrics=_metrics(m), exhibit_ids=[exhibit])


class ELADetector(Detector):
    spec = image_spec("img.forensic.ela", "Error Level Analysis (JPEG recompression)", "forensic", InputUnit.IMAGE, (H2,),
                      ("heatmap",), needs=("faces",), lane="cpu", devices=CPU_ONLY, timeout_s=30.0, version=NATIVE_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        pixels = ctx.media.native_pixels()
        orientation = native_forensics.exif_orientation(original_bytes(ctx))
        result = native_forensics.ela(pixels, orientation=orientation, face_box=_face_box(ctx, pixels.shape))
        m = result.metrics
        p, raw = forensics.ela_score(m)
        ratio = m.get("ela_face_bg_ratio")
        if ratio is not None:
            observed = f"face/background error ratio={ratio}, block CV={m['ela_block_cv']}"
            reasoning = ("The face region shows a compression error level distinctly different from its surroundings, which is "
                         "characteristic of a pasted or regenerated face." if raw > 0.6 else
                         "Compression error is uniform between the face and the background, as expected for a single-generation image.")
        else:
            observed = f"mean error={m['ela_mean']}, block CV={m['ela_block_cv']}"
            reasoning = ("Error levels vary strongly between regions, suggesting composited content with different compression "
                         "histories." if raw > 0.6 else "Error levels are spatially uniform, consistent with a single compression history.")
        exhibit = ctx.exhibits.heatmap(result.display, semantics="error_level", title="Error level analysis",
                                       caption="Brighter means a larger change on a JPEG re-save at quality 90, measured on the "
                                               "original pixels in the file's stored orientation. A region that differs from its "
                                               "surroundings was compressed differently.",
                                       media_ref="original", coordinate_space=result.coordinate_space)
        aggregate = UnitScore("global", p, "probability", quality={"face_measured": float(ratio is not None)})
        return DetectorResult(
            status="ok", unit_scores=[aggregate], aggregate=aggregate, observed=observed,
            reference="Single-generation image: uniform error; face vs surrounding-ring ratio ≈ 1.0 (0.7–1.4); block CV < 0.9",
            reasoning=reasoning, metrics={**_metrics(m), "exif_orientation": float(orientation)}, exhibit_ids=[exhibit],
            details={"display_mapping_version": native_forensics.DISPLAY_MAPPING_VERSION})


class NoiseResidualDetector(Detector):
    spec = image_spec("img.forensic.noise", "Noise-residual consistency", "forensic", InputUnit.IMAGE, (H2, H1), ("heatmap",),
                      needs=("faces",), lane="cpu", devices=CPU_ONLY, timeout_s=30.0, version=NATIVE_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        pixels = ctx.media.native_pixels()
        result = native_forensics.noise(pixels, face_box=_face_box(ctx, pixels.shape))
        m = result.metrics
        p, p_flat, p_ratio = forensics.noise_score(m)
        ratio = m.get("noise_face_bg_ratio")
        parts = ["Residual noise is unusually flat, typical of rendered rather than sensor-captured pixels" if p_flat > 0.6
                 else "Sensor-like residual noise is present across the image"]
        if p_ratio is not None:
            parts.append("the face region's noise level is inconsistent with the background, a hallmark of face replacement"
                         if p_ratio > 0.6 else "face and background noise levels are consistent")
        exhibit = ctx.exhibits.heatmap(result.display, semantics="residual", title="Local noise residual",
                                       caption="Local standard deviation of the median-filter residual on the original pixels. "
                                               "Sensor noise should be spatially consistent.",
                                       media_ref="original", coordinate_space=result.coordinate_space)
        aggregate = UnitScore("global", p, "probability", quality={"face_measured": float(ratio is not None)})
        return DetectorResult(
            status="ok", unit_scores=[aggregate], aggregate=aggregate,
            observed=f"σ={m['noise_std']}, local CV={m['noise_local_cv']}, flat fraction={m['noise_flat_fraction']}"
                     + (f", face/bg ratio={ratio}" if ratio is not None else ""),
            reference="Camera: flat fraction < 0.6, face vs surrounding-ring noise ratio 0.6–1.6",
            reasoning="; ".join(parts) + ".", metrics=_metrics(m), exhibit_ids=[exhibit],
            details={"display_mapping_version": native_forensics.DISPLAY_MAPPING_VERSION})


DQ_VERSION = "0.4.0"
DQ_REGION_SCORE = 0.75  # IF §6.4 classical-heuristic cap; hand-set and uncalibrated (legacy_weight 0 on the fallback path)
DQ_REFERENCE = "A file compressed once shows no double-quantisation periodicity; a re-saved file shows it in every textured block"


def _grid_metrics(grid: jpeg_dq.GridOrigin) -> dict[str, float]:
    return {"grid_dx": float(grid.dx), "grid_dy": float(grid.dy), "grid_contrast": min(grid.contrast_x, grid.contrast_y)}


class JpegDoubleCompressionDetector(Detector):
    """I-4: aligned double-JPEG map and the global grid origin (`jpeg_dq`), from the original bytes only."""

    spec = image_spec("img.forensic.jpeg_dq", "Aligned double-JPEG map and grid origin", "forensic", InputUnit.IMAGE, (H2,),
                      ("heatmap",), lane="cpu", devices=CPU_ONLY, timeout_s=60.0, version=DQ_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        data = original_bytes(ctx)
        try:
            result = jpeg_dq.analyse(data)
        except ImportError:
            return DetectorResult(status="skipped", reason="jpeglib is not installed")
        if result is None:
            return self._grid_only(data)
        grid, supporting, tested = result.grid, result.supporting, len(result.fits)
        metrics = {"dq_frequencies_tested": float(tested), "dq_frequencies_supporting": float(len(supporting)),
                   "double_compression": float(result.double_compression),
                   "dq_gain_max": max((f.gain for f in result.fits), default=0.0), **_grid_metrics(grid)}
        crop = (f" The dominant 8 × 8 grid starts at ({grid.dx}, {grid.dy}) rather than the corner, so the image was cropped after "
                "an earlier compression." if grid.detected and (grid.dx, grid.dy) != (0, 0) else "")
        if not result.double_compression:
            return DetectorResult(
                status="ok", metrics=metrics, reference=DQ_REFERENCE,
                observed=f"no aligned double-compression trace ({len(supporting)} of {tested} tested frequencies)",
                reasoning="Consistent with a single compression, or with a re-save that leaves no trace (equal or finer earlier "
                          "steps, resizing between saves). This is not evidence that the image is unedited." + crop)
        if result.earlier_quality is not None:
            metrics["earlier_quality"] = float(result.earlier_quality)
        orientation = native_forensics.exif_orientation(data)
        posterior = native_forensics.orient(result.posterior(), orientation)
        reliable = native_forensics.orient(result.reliable, orientation)
        width = result.width if orientation < 5 else result.height
        exhibit = ctx.exhibits.heatmap(
            posterior, semantics="p_manipulated", title="Double-JPEG map", value_range=(0.0, 1.0), mask=reliable,
            caption="Per 8 × 8 block: how likely the block lacks the double-compression periodicity found in the rest of the file "
                    "(uncalibrated). Blocks too smooth to measure are masked.",
            media_ref="original",
            coordinate_space={"width": int(posterior.shape[1]), "height": int(posterior.shape[0]),
                              "scale_to_original": width / posterior.shape[1]})
        earlier = f", earlier quality ≈ {result.earlier_quality} (libjpeg scaling)" if result.earlier_quality is not None else ""
        region = result.localised
        if region is None:
            return DetectorResult(
                status="ok", metrics=metrics, exhibit_ids=[exhibit], reference=DQ_REFERENCE,
                observed=f"compressed at least twice{earlier}; {len(supporting)} of {tested} tested frequencies show the trace "
                         "and no region lacks it",
                reasoning="The file was saved as JPEG at least twice. Re-saving is common and is not deception, and no area "
                          "stands out from the rest." + crop)
        metrics |= {"dq_region_blocks": float(region.blocks), "dq_region_share": region.share}
        aggregate = UnitScore("global", DQ_REGION_SCORE, "probability", quality={})
        return DetectorResult(
            status="ok", unit_scores=[aggregate], aggregate=aggregate, metrics=metrics, exhibit_ids=[exhibit],
            reference=DQ_REFERENCE,
            observed=f"compressed at least twice{earlier}; a region of {region.blocks} blocks ({region.share:.1%} of the measurable "
                     "area) lacks the trace",
            reasoning="Most of the file carries the periodicity of an earlier compression but one region does not, so that region "
                      "has a different compression history, as when content is pasted or regenerated before the last save. Smooth "
                      "or saturated areas can also lose the trace." + crop)

    @staticmethod
    def _grid_only(data: bytes) -> DetectorResult:
        grid = jpeg_dq.grid_from_bytes(data)
        if grid is None or not grid.detected:
            return DetectorResult(status="not_applicable", reason="not a readable 8-bit JPEG and no JPEG block grid in the pixels")
        cropped = (grid.dx, grid.dy) != (0, 0)
        return DetectorResult(
            status="ok", metrics=_grid_metrics(grid), reference="Pixels never JPEG-compressed show no 8 × 8 block grid",
            observed=f"an 8 × 8 JPEG block grid in the pixels at offset ({grid.dx}, {grid.dy})",
            reasoning="The image was JPEG-compressed before it was saved in its current format"
                      + (", and cropped afterwards: the grid does not start at the corner." if cropped else "."))


class LocalJpegGridDetector(Detector):
    """Locate a different prior JPEG block alignment in a qualifying PNG."""

    spec = image_spec(local_jpeg_grid.DETECTOR_ID, "Local compression-grid inspection", "forensic", InputUnit.IMAGE,
                      (H2,), ("heatmap", "regions"), lane="cpu", devices=CPU_ONLY, profiles=ALL_PROFILES,
                      timeout_s=10.0, version="0.1.0")

    def run(self, ctx: RunContext) -> DetectorResult:
        probe = ctx.outputs.probe or {}
        width, height = probe.get("width", 0), probe.get("height", 0)
        if (probe.get("format") != "PNG" or probe.get("mode") != "RGB" or probe.get("n_frames") != 1
                or min(width, height) < 256 or width * height > 1_048_576):
            return DetectorResult(status="not_applicable", reason="requires a single-frame RGB PNG, at least 256 pixels per side and at most 1 megapixel")
        pixels = ctx.media.native_pixels()
        height, width = pixels.shape[:2]
        response = local_jpeg_grid.response(pixels)
        if response["global_phase"] is None:
            return DetectorResult(status="not_applicable", reason="no sufficiently clear prior JPEG grid for a local comparison")
        values, boxes = local_jpeg_grid.pixel_map(response, local_jpeg_grid.PNG_ARM, local_jpeg_grid.PNG_REVIEW_POINT,
                                                 (height, width))
        reference = "A single prior JPEG save generally leaves the same block alignment throughout measurable areas."
        if not boxes:
            return DetectorResult(status="ok", observed="A prior JPEG grid was measured; no region met the local review rule.",
                                  reference=reference, reasoning="Weak or erased compression traces can leave an edit undetected.")

        space = {"width": width, "height": height, "scale_to_original": 1.0}
        title = "Local compression-grid difference"
        source_ids = [f"{self.spec.id}:global"]
        heatmap = ctx.exhibits.heatmap(values, semantics="residual", title=title, value_range=(0.0, 1.0),
                                      caption="Highlighted blocks have a different earlier JPEG alignment from the dominant grid.",
                                      source_signal_ids=source_ids, coordinate_space=space, media_ref="original")
        ordered = sorted(boxes, key=lambda box: (-box[4], box[1], box[0]))
        regions = [{"region": {"kind": "bbox", "bbox": {"x": x, "y": y, "w": w, "h": h}}, "score": score}
                   for x, y, w, h, score in ordered]
        region_exhibit = ctx.exhibits.regions(regions, title=title, source_signal_ids=source_ids,
                                              coordinate_space=space, media_ref="original")
        aggregate = UnitScore("global", ordered[0][4], "score")
        count = len(boxes)
        return DetectorResult(
            status="ok", aggregate=aggregate, unit_scores=[aggregate], exhibit_ids=[heatmap, region_exhibit],
            observed=f"A different prior JPEG block alignment was measured in {count} local {'area' if count == 1 else 'areas'}.",
            reference=reference, metrics={"regions_measured": float(count)},
            reasoning="An area may have been saved or cropped separately before it was combined with the image. "
                      "This evidence cannot identify AI generation and carries zero weight in AI confidence.")


class SharpBoundaryDetector(Detector):
    """Record the native sides of an abrupt rectangular boundary."""

    spec = image_spec(sharp_boundary.DETECTOR_ID, "Sharp image boundary inspection", "forensic", InputUnit.IMAGE,
                      (H2,), ("heatmap", "regions"), lane="cpu", devices=CPU_ONLY, profiles=ALL_PROFILES,
                      timeout_s=10.0, version="0.1.0")

    def run(self, ctx: RunContext) -> DetectorResult:
        probe = ctx.outputs.probe or {}
        if (probe.get("format") != "PNG" or probe.get("mode") != "RGB" or probe.get("n_frames") != 1
                or not sharp_boundary.supported_size(probe.get("width", 0), probe.get("height", 0))):
            return DetectorResult.not_applicable("Requires a single-frame RGB PNG with both dimensions between 512 and 768 pixels.")
        pixels = ctx.media.native_pixels()
        height, width = pixels.shape[:2]
        boundary = sharp_boundary.measure(pixels)
        reference = "The check compares color steps along each side of a rectangle with parallel steps three pixels away."
        if boundary.score <= sharp_boundary.REVIEW_POINT:
            return DetectorResult(status="ok", observed="No abrupt rectangular boundary met the inspection rule.",
                                  reference=reference, reasoning="The check can miss smooth blends and other edits.")

        left, top, right, bottom = boundary.box
        space = {"width": width, "height": height, "scale_to_original": 1.0}
        title = "Abrupt rectangular boundary"
        sources = [f"{self.spec.id}:global"]
        heatmap = ctx.exhibits.heatmap(
            boundary.contour, semantics="residual", title=title, value_range=(0.0, sharp_boundary.MAX_STEP_RESPONSE),
            caption="Colored pixels mark measured color steps along the four selected sides.",
            source_signal_ids=sources, coordinate_space=space, media_ref="original")
        region = {"region": {"kind": "bbox", "bbox": {"x": left, "y": top, "w": right - left, "h": bottom - top}},
                  "score": boundary.score}
        regions = ctx.exhibits.regions([region], title=title, source_signal_ids=sources,
                                      coordinate_space=space, media_ref="original")
        aggregate = UnitScore("global", boundary.score, "score")
        return DetectorResult(
            status="ok", aggregate=aggregate, unit_scores=[aggregate], exhibit_ids=[heatmap, regions],
            observed="A sharp rectangular boundary was measured on four sides.", reference=reference,
            reasoning="Ordinary pasting or a local tone adjustment can create this boundary. It carries zero weight in AI confidence.",
            metrics={"regions_measured": 1.0},
            details={"sharp_boundary": {"schema": sharp_boundary.SCHEMA, "sides": boundary.sides}})


CM_VERSION = "0.4.0"
CM_SCORE = 0.75  # IF §6.4 classical-heuristic cap; hand-set and uncalibrated (legacy_weight 0 on the fallback path)
CM_REFERENCE = ("An unedited photograph seldom holds two areas whose keypoints are related by one affine transform, apart from "
                "repeated patterns and symmetric objects")
CM_COLUMNS = ["pair", "area C (x, y, w, h px)", "area C′ (x, y, w, h px)", "inlier keypoints", "share of the areas' keypoints",
              "rotation (°)", "scale x", "scale y", "shear", "translation (px)", "mirrored", "repeated structure"]


def _box_text(box: copy_move.Box) -> str:
    return ", ".join(f"{v:.0f}" for v in box)


def _clone_regions(pairs: list[copy_move.ClonePair]) -> list[dict]:
    """D-26 display ids C1 / C1′ … in native px; the score is the share of the areas' keypoints in the fit."""
    return [{"display_id": f"C{k}{suffix}", "region": {"kind": "bbox", "bbox": dict(zip("xywh", box, strict=True))},
             "score": p.coverage,
             "label": f"C{k}{suffix}: area {side} of pair {k}, {p.inliers} inlier keypoints" + (", on repeated structure" if p.repetitive else "")}
            for k, p in enumerate(pairs, 1) for suffix, side, box in (("", "a", p.box_a), ("′", "b", p.box_b))]


def _clone_rows(pairs: list[copy_move.ClonePair]) -> list[list]:
    return [[f"C{k} / C{k}′", _box_text(p.box_a), _box_text(p.box_b), float(p.inliers), p.coverage, p.affine.rotation_deg,
             p.affine.scale_x, p.affine.scale_y, p.affine.shear, f"{p.affine.translation[0]:.1f}, {p.affine.translation[1]:.1f}",
             "yes" if p.affine.reflected else "no", "yes" if p.repetitive else "no"] for k, p in enumerate(pairs, 1)]


class CopyMoveDetector(Detector):
    """I-12: SIFT keypoint copy-move (`copy_move`) on native pixels reduced to a long side of at most 2048 px."""

    spec = image_spec("img.forensic.copymove", "SIFT copy-move (keypoint g2NN, RANSAC affine)", "forensic", InputUnit.IMAGE, (H2,),
                      ("regions", "table"), lane="cpu", devices=CPU_ONLY, timeout_s=60.0, version=CM_VERSION)

    def run(self, ctx: RunContext) -> DetectorResult:
        pixels = ctx.media.native_pixels()
        if min(pixels.shape[:2]) < copy_move.MIN_INPUT_SIDE:
            return DetectorResult.not_applicable(f"image below {copy_move.MIN_INPUT_SIDE} px")
        result = copy_move.analyse(pixels)
        distinct = result.distinct
        ordered = [*distinct, *(p for p in result.pairs if p not in distinct)]
        repetitive = result.repetitive_image or any(p.repetitive for p in ordered)
        metrics = {"keypoints": float(result.keypoints), "g2nn_pairs": float(result.matches), "components": float(result.components),
                   "clone_pairs": float(len(ordered)), "clone_pairs_distinct": float(len(distinct)),
                   "clone_pairs_repetitive": float(sum(p.repetitive for p in ordered)), "crowded_share": round(result.crowded_share, 4),
                   "repetitive_structure": float(repetitive), "analysed_long_side": float(max(result.analysed)),
                   "max_inliers": float(max((p.inliers for p in ordered), default=0))}
        resolution = ("native pixels" if max(result.analysed) == max(result.width, result.height)
                      else f"luma reduced to {result.analysed[0]}×{result.analysed[1]} px")
        basis = f"{result.keypoints} keypoints, {result.matches} g2NN pairs; {resolution}"
        warning = ""
        if result.repetitive_image:
            warning = (f" Repeated structure: {100 * result.crowded_share:.1f} % of keypoints have ten or more look-alikes, so similar "
                       "areas are expected here without any editing, and pairs do not score.")
        elif repetitive:
            warning = (" Repeated structure: some pairs lie on areas with several look-alikes (tiles, windows, text), which form "
                       "similar areas without any editing; those pairs do not score.")
        if not ordered:
            return DetectorResult(
                status="ok", metrics=metrics, reference=CM_REFERENCE,
                observed=f"no pair of areas related by one affine transform ({basis})",
                reasoning="No copied area was found. Copies of smooth or small areas, and copies blurred or resampled strongly, leave "
                          "too few keypoints, so this is not evidence that nothing was copied." + warning)
        space = {"width": result.width, "height": result.height, "scale_to_original": 1.0}
        exhibit_ids = [
            ctx.exhibits.regions(_clone_regions(ordered), title="Clone pairs (SIFT copy-move)", media_ref="original",
                                 caption="Pairs of areas whose keypoints are related by one affine transform: C1 and C1′, C2 and C2′ …, "
                                         "boxes around the inlier keypoints in native pixels. Either area can be the copy; pairs "
                                         "on repeated structure are labelled.",
                                 coordinate_space=space),
            ctx.exhibits.table(CM_COLUMNS, _clone_rows(ordered), title="Clone pair transforms", media_ref="original",
                               caption="Affine transform from area C to area C′ (RANSAC fit): rotation, scales and shear of the "
                                       "linear part, translation in native pixels, and the inlier keypoints that support it.")]
        top = ordered[0]
        observed = (f"{len(ordered)} pair(s) of areas related by one affine transform, {len(distinct)} not on repeated structure; C1 "
                    f"({_box_text(top.box_a)}) and C1′ ({_box_text(top.box_b)}) px, {top.inliers} inlier keypoints, "
                    f"{100 * top.coverage:.0f} % of the areas' keypoints, rotation {top.affine.rotation_deg:.1f}°, scale "
                    f"{top.affine.scale_x:.2f} × {top.affine.scale_y:.2f}; {basis}")
        if not distinct:
            return DetectorResult(
                status="ok", metrics=metrics, exhibit_ids=exhibit_ids, reference=CM_REFERENCE, observed=observed,
                reasoning="Similar areas were found on repeated structure, where unedited photographs show the same pattern; they are "
                          "recorded for review only." + warning)
        aggregate = UnitScore("global", CM_SCORE, "probability", quality={"inliers": float(top.inliers), "coverage": top.coverage})
        return DetectorResult(
            status="ok", unit_scores=[aggregate], aggregate=aggregate, metrics=metrics, exhibit_ids=exhibit_ids,
            reference=CM_REFERENCE, observed=observed,
            reasoning="Two areas share many keypoints related by one affine transform, as when part of an image is copied and pasted "
                      "elsewhere, possibly rotated or scaled. Either area can be the copy, and symmetric objects or repeated items can "
                      "also form such pairs." + warning)
