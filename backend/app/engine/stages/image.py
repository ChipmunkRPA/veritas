"""Image pipeline stages (ARCHITECTURE §3.1–3.2). The default image path since P-5 (`VERITAS_PIPELINE=v1` keeps v0.3).

- **Resume-safe outputs.** Each stage returns JSON. Detector results, exhibit models and pending artefact records
  are part of that JSON, so a resumed run can rebuild the published result from persisted stage outputs alone.
- **Legacy fuser.** Reproduces the v0.3 fusion, and publishes an uncalibrated fallback `Decision` (§4.4) until M-3 ships
  a fuser artefact.
  - Log-odds average with the manifest `legacy_weight`s.
  - v0.3 rules, applied here rather than inside detectors (§2.2): quality multipliers, face-specialist asymmetry,
    hypothesis separation and the resolution gate.
  - D-15 overrides: `decisive_ai` floors s at 0.97, `self_declared_ai` at 0.85 (with the self-declared caveat), and
    `trusted_capture` caps it at 0.10. The statistical estimate before the override is kept in the caveats (D-33).
- **Synthesis.** Deliberately minimal: one finding per probability-scored signal and one coverage finding. M-4
  replaces it.
- **v1 view.** `render` writes the v1 artifact files (faces.jpg, ela.jpg, noise_map.jpg, spectrum.png, gradcam_*.jpg)
  for the Folio UI and the v1 PDF. `publish` stores `downgrade_v2(result)` plus the v1 image, faces and metadata blocks.
"""
from __future__ import annotations

import dataclasses
import io
from typing import Any

import cv2
import numpy as np
from PIL import Image
from pydantic import TypeAdapter

from app.core.config import settings
from app.core.evidence import Signal, band_for
from app.detectors.common.fusion import fuse as legacy_fuse
from app.detectors.common.fusion import prob_to_signal_fields
from app.detectors.image.local_jpeg_grid import DETECTOR_ID as LOCAL_GRID_DETECTOR_ID
from app.detectors.image.sharp_boundary import DETECTOR_ID as SHARP_BOUNDARY_DETECTOR_ID
from app.engine import provenance_gate
from app.engine import tiles as tile_pyramid
from app.engine.fusion import abstain
from app.engine.fusion.decisive import (
    CAPTURE_VERBAL,
    FALLBACK_DECLARED_AI_REASON,
    FALLBACK_STATISTICAL_CAVEAT,
    FALLBACK_TRUSTED_CAPTURE_REASON,
    OVERRIDE_PROBABILITY,
    STATISTICAL_CAVEATS,
    STATISTICAL_ESTIMATE_CAVEAT,
    marker_kind,
    watermark_only,
)
from app.engine.fusion.decisive import apply as apply_override
from app.engine.fusion.decisive import record as record_of_override
from app.engine.fusion.runtime import DECLARED_AI_VERBAL
from app.engine.media import MAX_PIXELS
from app.engine.runner import StageContext
from app.engine.stages import common
from app.evidence.image_copy_similarity_binding import credential_skip
from app.schemas.evidence import AnalysisResultV2, Decision, Exhibit, HeatmapExhibit, SignalV2
from app.schemas.legacy import (
    BAND_CODE,
    BAND_HEADLINE,
    FALLBACK_CAVEAT,
    H3_VALUE,
    V1_NOTE,
    downgrade_v2,
    fallback_cap,
    fallback_reason,
)
from app.storage.artefacts import index_object
from app.storage.objects import ObjectStore

# p38: full-extent forensic display maps; p37: immutable image evidence sets.
PIPELINE_VERSION = "img-2.0-p50"  # p50: opt-in sharp boundary inspection; p49: add native PNG prior-grid inspection; p48: signed provenance also stops image copy checks; p42: optional PNG splice screening and signed native area rule; p41: distinguish measured but unselected local regions from missing checks; p40: frozen candidate gates and writer-exact region quantization, evidence v3; p39: reproducible quantised component masks and frozen evidence v2; p36 (N-19q, finding 4; synthesis tpl-2.19): the finding on the checks that ran says of a Content Credentials check that read the file and found no manifest that it ran and found none ("C2PA Content Credentials ran and found no manifest in the file."; it read "Not run: C2PA Content Credentials (not applicable: “no C2PA manifest”)" beside the finding "No Content Credentials"); the finding of a check that entered the estimate stores its stand-alone words. p35 (N-19p, design review D6-05; synthesis tpl-2.18): the coverage finding counts the statistical checks only, so a check that tests the provenance and integrity of the file is no "scored check" among those that flagged the file. p34 (N-19p, security re-check SEC6-01, code review R10-02, R10-03): text from a file keeps its words apart (tab, newline, carriage return, Zl and Zp as a space; the other control and direction characters dropped), the title and caption of every exhibit are filtered, and the titles of C2PA thumbnails are filtered and cut to 200 characters (`prov.c2pa` 0.4.3, `img.meta.exif` 0.4.1, `img.meta.consistency` 0.4.5, `img.meta.generator` 0.4.7). p33 (N-19p, security re-check SEC6-02, code review R10-04): ExifTool reads a copy of a PNG through its IEND chunk (what follows is dropped; at a chunk that runs past the end the copy stops), with the first chunk keyed exactly "XML:com.adobe.xmp" and at most 64 other text chunks, and reads the thumbnail from that copy; the fallback without ExifTool reads PNG text through the bound of the checks (`img.meta.consistency` 0.4.4). p32 (N-19p, code review R10-01): the PNG keys that the metadata checks read by name are read wherever they stand in the file, past the bound on PNG text (a generator marker past the 64th text chunk or past 64 KB of text was dropped and the decision changed), and the generator check says how many text chunks the bound left unread (`img.meta.generator` 0.4.6, `img.meta.consistency` 0.4.3). p31 (N-19p, browser QA Q6-01): in the v1 view of a result a mention of AI in a text field has no `ai_probability` and the direction "neutral" (it gave the check's score of 0.65). p30 (N-19o, browser QA Q5-05): the finding of a mention of AI in a text field names the field and gives the example of "Not AI-generated" only for the phrase pattern; (design review D5-03): outside what was validated, where no estimate is shown, the finding of a check of little effect is titled "Little effect" (it read "Little effect on the estimate (AI generation)") and a check that entered says "In this result …" (it read "In this estimate …"). p29 (N-19n, what the third part of N-19m left; supervisor check 29): under a configured prior the reason of a result that D-77 took ends "…, but the estimate stands in the other answer's part of the scale, so the result is inconclusive." where its published estimate prints outside 35 to 64 (it said "the middle part of the scale" over a printed 21). p28 (N-19n, what the first two parts of N-19m left; the code reviewer's read of their draft): on the fallback the decision's own headline (`verbal`) under a declaration and under a marker whose floor decides is the record's sentence, as on the calibrated path and the video fallback (it was the sentence of the band). p27 (N-19n, SEC4-01; the security re-review of bc49948c): the XMP packet of a file is cut to 64 KB before the metadata checks scan it, and the reader of the IPTC source type is linear and bounded (it was cubic on a crafted packet); a stored reading changes only for a packet whose declaration stood past that limit, in an element of more than 300 characters, or under a fifth declared prefix. p26 (N-19n, the marker’s finding, one step for the part; design review D3-07): the finding of a self-declared marker is a record’s (its title "Self-declared: …", "Self-declared, not applied: …" where the decision did not apply it, and the decision of a result that the checks decide beside a marker is the checks’ in `verbal` and `reason`); Content Credentials that do not validate are titled as such; the caveat of what was set aside names it without a reason; unsigned metadata is worded by what the file holds (the kinds of a declaration and of generation parameters); and the signal of the metadata check holds what the check found and where (`details`), from which its finding says the place. p25: a picture whose metadata mentions AI in a text field has its findings (the step that writes them failed for every such picture since N-19h), the mention among them in sentences of its own, and the mention takes no part in the fallback's score (N-19n; browser QA Q4-01, P1). p24: the v1 summary of an inconclusive result outside what was validated says so, with why, also where the cap changed no code (it read "the checks don't point clearly either way"; N-19n). p23: a decisive code of the statistical checks needs the estimate as published in its answer’s part of the scale (D-77: 34 or less, 65 or more, after a configured prior) and checks that together give it at least ln 2 (D-76 as amended); a result that fails one is inconclusive with that condition’s reason, and `fusion.rule` and `statistical_alone.rule` name it; the reason’s number is the printed whole points; a cap’s reason quotes the levels’ headline only where both conditions would have left it (N-19m, third part). p22: under a self-declared marker the confidence of a result is low where D-76 took the decisive code of the statistical estimate (no check gives at least weak support for its answer), as it is for a check that did not run, for disagreement and for a wide interval; it was medium, and low for the same input without the marker; under a trusted declaration and a trusted capture the confidence stays the record's (N-19m; code review R7-03, D-76 as amended on 2026-10-05); p21: the decision of a result that a trusted capture decides is headed by the record ("The file's Content Credentials record a capture of these bytes" in `verbal`, which the v1 summary follows, on both fusion paths; it read the sentence of the decision code, "No indications of AI generation or manipulation found"); its code, number, band and confidence are unchanged (N-19m; design review D3-01, D-15); p20: the row of a self-declared record in the breakdown names the kind of the record (unsigned metadata, a public watermark message, Content Credentials by what they name or declare; one label stood for all) (N-19l; code review R6-04); p19: the reason of an image that is capped for its content says first what the image looks like and then the measurement ("it looks like a screenshot of text or an interface, or a flat graphic: 91 % of the image's 8 × 8 pixel blocks are flat …"; the measurement stood first) (N-19k; design re-review D2-12); the checks that could not run on this server are one caveat with their names, in place of one caveat for each and of their ids in the caveat of a reduced profile; a caveat names a check as the page does, a step of the run and a provenance record in words, never a detector id, a stage name or the value of a kind (browser QA leads 3 and 4, design re-review D2-06); each check that scored and is not counted in the estimate carries the fusion's record of why (`outside_estimate`: not in the model, its gate closed, a weight of zero; browser QA Q2-12); a result of the fallback on an instance that holds a fusion model says that none was used for the analysis, not that none is loaded, and the v1 view of a result outside what was validated gives the reason of its cap (N-19k; browser QA Q2-14 and part 2); p18: a decisive result needs a check that supports it (D-76): a result that the statistical checks would decide inside the envelope without a check whose published row is at least ln 2 toward its answer is inconclusive, and its reason says so; each row of a check says whether the decision counts it (`counted`: not a row of zero, not the row of a check that declares H3); p17: a result that a provenance record decides names its record (`provenance_record`) and words a marker by its kind; the caveat of the statistical estimate before the override prints the clamped number, the caveats of an interval stand only with one, and outside what was validated that caveat says that the number is not shown; the fallback publishes its statistical score beside a record where a check of the content scored and none where records alone scored, and a fallback result that a record decides outside what was validated says so in its reason; the reason of a marker that decides alone ends with the sentence of its case (N-19h, E-19 … E-21, 2026-10-04); p16: a check counts for a strong result, for "k of n detectors" and for "no check" when its published row is not zero; where every check ran or was not applicable and none that scored counts, the result is not assessable in words of its own, and a provenance record over such a result decides alone (N-19f, D-64 as clarified, code review R4-05, 2026-10-04), and a fallback result whose condition stage failed is capped at inconclusive (N-19f, R4-03); p15: an image of a kind the bench does not hold (flat blocks above its limit, greyscale, the screenshot rule) is outside the envelope and capped at inconclusive (N-18, D-65, 2026-10-03), a strong code needs two checks that scored, and a fallback result in which only metadata or provenance records scored is not assessable (N-18, D-64); p14: a not-assessable result has no caveat or reason that speaks of an estimate or a probability (N-17, D-68); p13: a finding that entered the estimate is worded by its contribution (N-17, D-63, 2026-10-03); p12: contributions are measured from each check's neutral point (N-17); p11: distinguish unsupported C2PA containers from absent credentials (2026-09-20)
EXHIBIT_STAGES = ("provenance", "faces", "detect", "localise")
SIGNAL_STAGES = ("provenance", "detect")
_EXHIBIT = TypeAdapter(Exhibit)

CATEGORY_BY_FAMILY = {"learned-whole-image": "model", "learned-face": "face", "temporal-face": "face", "forensic": "forensic",
                      "localiser": "forensic",
                      "codec": "forensic", "provenance": "provenance", "synthetic-video": "model", "av-sync": "temporal",
                      "voice": "model", "text": "stylometric"}
SCOPE_BY_PREFIX = {"face": "face", "track": "track", "window": "window", "sentence": "sentence", "segment": "segment",
                   "region": "region"}

# --- legacy fuser rules (v0.3 config.py and detectors/image/pipeline.py) ---
LEGACY_QUALITY_MULTIPLIERS = {
    "input_upscaled": 0.3,  # UnivFD on an image below 224 px
    "face_inputs_upscaled": 0.4,  # face_upscaled_weight: every face crop was upscaled to the model input
    "no_face": 0.4,  # a face-oriented classifier scored the whole image
    "small_face": 0.4,  # DFDC: largest face below dfdc_min_face_px
}
MIN_DETECTOR_SIDE = 224  # resolution gate: detectors are trained on ≥ 224 px inputs
LOW_RESOLUTION_CAVEAT = "Low-resolution input: pixel-level detector reliability is reduced."
NO_FACE_CAVEAT = ("No face was detected, so face-swap-specific detectors were down-weighted; general AI-image detectors carry "
                  "the verdict.")

# --- v1 artifact files ---
LEGACY_HEATMAP_ARTIFACTS = {"error_level": "ela.jpg", "residual": "noise_map.jpg"}
LEGACY_ARTIFACT_MAX_PX = 1024


# --- detector plumbing (shared with the document and video pipelines: stages/common.py) ------------------------

unit_score_json = common.unit_score_json
detector_entry = common.detector_entry
signal_id = common.signal_id
DbRunStore = common.DbRunStore
_media = common.media
_run_detectors = common.run_detectors
_stage_exhibits = common.stage_exhibits
_entries = common.entries
ingest = common.ingest


# --- stages ------------------------------------------------------------------------------------------------

def probe(ctx: StageContext) -> dict[str, Any]:
    """Header facts from the original bitstream (no pixel decode)."""
    data = bytes(_media(ctx).original_bytes())
    with Image.open(io.BytesIO(data)) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError(f"The image exceeds the {MAX_PIXELS // 1_000_000} MP pixel limit")
        exif = im.getexif()
        return {"format": im.format, "mode": im.mode, "width": int(im.width), "height": int(im.height),
                "n_frames": int(getattr(im, "n_frames", 1)), "has_exif": len(exif) > 0,
                "exif_orientation": int(exif.get(0x0112, 1) or 1), "has_icc_profile": bool(im.info.get("icc_profile")),
                "jpeg_quant_tables": len(getattr(im, "quantization", None) or {}),
                "progressive": bool(im.info.get("progressive") or im.info.get("progression")), "size_bytes": len(data)}


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
    matches = [row for row in rows if row.get("detector_id") == "prov.c2pa"]
    if not matches:
        return not c2pa_expected
    return len(matches) == 1 and matches[0].get("version") == C2PA_VERSION


def provenance(ctx: StageContext) -> dict[str, Any]:
    return _run_detectors(ctx, include={"provenance"})


def decode(ctx: StageContext) -> dict[str, Any]:
    """Native pixels and the ≤ 4096 px analysis copy (D13). Not persisted: recomputed deterministically on resume."""
    media = _media(ctx)
    native = media.native_pixels()
    copy, scale = media.analysis_copy()
    info = media.info()
    return {"width": int(native.shape[1]), "height": int(native.shape[0]), "analysis_width": int(copy.shape[1]),
            "analysis_height": int(copy.shape[0]), "scale_to_original": round(float(scale), 6),
            "exif_transposed": bool(info["exif_transposed"])}


def faces_from_exhibits(exhibits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Face-detection `regions` exhibits (JSON dumps) → face dicts in analysis-copy px: the box, the score and, when present,
    the five landmarks."""
    return [{**r["region"]["bbox"], "score": r.get("score"),
             **({"landmarks": r["region"]["landmarks"]} if r["region"].get("landmarks") else {})}
            for e in exhibits if e["kind"] == "regions" for r in e["regions"] if r["region"].get("bbox")]


def faces(ctx: StageContext) -> dict[str, Any]:
    out = _run_detectors(ctx, include={"face-detection"})
    return {**out, "faces": faces_from_exhibits(out["exhibits"])}


def detect(ctx: StageContext) -> dict[str, Any]:
    return _run_detectors(ctx, exclude={"provenance", "face-detection"}, face_failure=common.face_stage_failure(ctx))


def _gray(store: ObjectStore, sha256: str) -> np.ndarray:
    with Image.open(io.BytesIO(store.read_bytes(sha256))) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)


def _components(values: np.ndarray, reliable: np.ndarray, heatmap: HeatmapExhibit, max_regions: int = 10) -> list[dict[str, Any]]:
    valid = values[reliable]
    if valid.size == 0 or float(valid.max()) <= float(valid.min()):
        return []
    if heatmap.calibrated and heatmap.semantics == "p_manipulated":
        threshold, label = 0.5, "calibrated: p ≥ 0.5"
    else:
        threshold, label = float(np.percentile(valid, 95)), "relative: top 5 % of map values"
    binary = ((values >= threshold) & reliable).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    grid_h, grid_w = values.shape
    space = heatmap.coordinate_space
    sx, sy = (space.width / grid_w, space.height / grid_h) if space else (1.0, 1.0)
    min_area = max(1, int(0.0005 * grid_w * grid_h))
    regions = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        regions.append({"region": {"kind": "bbox", "bbox": {"x": round(x * sx, 3), "y": round(y * sy, 3),
                                                              "w": round(w * sx, 3), "h": round(h * sy, 3)}},
                        "score": round(float(values[labels == i].max()), 6), "label": label})
    regions.sort(key=lambda r: (-r["score"], r["region"]["bbox"]["y"], r["region"]["bbox"]["x"]))
    for k, region in enumerate(regions[:max_regions], 1):
        region["display_id"] = f"R{k}"
    return regions[:max_regions]


def localise(ctx: StageContext) -> dict[str, Any]:
    """Heatmaps → reliability-masked connected components → `RegionsExhibit`s (§3.1).

    Attribution maps (Grad-CAM) show where a classifier looked and are never turned into regions (IF §6.2). Localisers write
    their own regions at their operating threshold (IF §3.2), so their maps are not segmented again."""
    detected = ctx.outputs.get("detect") or {}
    shas = {a["id"]: a["sha256"] for a in detected.get("artefacts", [])}
    own_regions = {e["created_by"] for e in detected.get("exhibits", []) if e["kind"] == "regions"}
    for raw in detected.get("exhibits", []):
        if (raw["kind"] != "heatmap" or raw["semantics"] == "attribution" or raw["created_by"] in own_regions
                or raw["created_by"].startswith("loc.")):
            # Empty dedicated localisation is meaningful (including a failed
            # image gate). Do not replace it with relative top-5% candidates.
            continue
        heatmap = _EXHIBIT.validate_python(raw)
        ctx.check("localise")
        low, high = heatmap.value_range
        values = _gray(ctx.store, shas[heatmap.payload_artefact_id]) / 255.0 * (high - low) + low
        reliable = (np.ones(values.shape, bool) if heatmap.reliability_mask_artefact_id is None
                    else _gray(ctx.store, shas[heatmap.reliability_mask_artefact_id]) > 127)
        regions = _components(values, reliable, heatmap)
        if regions:  # regions cite the signal of the detector that wrote the map, so findings can scope to them (M-4)
            sources = list(heatmap.source_signal_ids) or [
                signal_id(e["detector_id"], e["aggregate"]) for e in detected.get("detectors", [])
                if e["detector_id"] == heatmap.created_by and e["status"] == "ok" and e["aggregate"] is not None]
            ctx.exhibits.regions(regions, title=f"Regions from {heatmap.title}"[:255], caption=regions[0]["label"],
                                 media_ref=heatmap.media_ref, source_signal_ids=sources, coordinate_space=heatmap.coordinate_space)
    return _stage_exhibits(ctx, {"localise"})


# --- legacy fuser ------------------------------------------------------------------------------------------

@dataclasses.dataclass
class LegacySignal:
    entry: dict[str, Any]
    category: str
    prob: float | None  # rounded to 4 decimals, as the v0.3 Signal stored it
    weight: float
    decisive: bool
    notes: list[str] = dataclasses.field(default_factory=list)


def legacy_weight(entry: dict[str, Any], manifest: Any) -> float:
    """Manifest `legacy_weight` (1.0 for detectors outside the manifest), or the detector's outcome weight, times the
    v0.3 multipliers for the quality flags the detector reported."""
    det = manifest.detector(entry["detector_id"]) if manifest is not None else None
    base = det.legacy_weight if det is not None and det.legacy_weight is not None else 1.0
    quality = entry["aggregate"]["quality"]
    weight = quality.get("outcome_weight", base)
    for flag, factor in LEGACY_QUALITY_MULTIPLIERS.items():
        if quality.get(flag, 0.0) >= 0.5:
            weight *= factor
    return weight


def legacy_signals(entries: list[dict[str, Any]], manifest: Any) -> list[LegacySignal]:
    out = []
    for e in entries:
        aggregate = e["aggregate"]
        if e["status"] != "ok" or aggregate is None:
            continue
        prob = aggregate["raw"] if aggregate["raw_kind"] == "probability" and 0.0 <= aggregate["raw"] <= 1.0 else None
        override = e["provenance_override"]
        if override == "self_declared_ai" or (e["detector_id"] in {"img.meta.generator", "img.watermark.dwtdct", "prov.c2pa"} and override is None):
            prob, weight, decisive = None, 0.0, False
        elif override is not None:
            floor = OVERRIDE_PROBABILITY[override]
            prob = min(prob if prob is not None else 1.0, floor) if override == "trusted_capture" \
                else max(prob if prob is not None else 0.0, floor)
            weight, decisive = 0.0, override != "self_declared_ai"  # overrides are not averaged (D-15, D-33)
        else:
            weight, decisive = (legacy_weight(e, manifest) if prob is not None else 0.0), False
        out.append(LegacySignal(e, CATEGORY_BY_FAMILY.get(e["family"], "statistical"),
                                None if prob is None else round(prob, 4), weight, decisive))
    return out


def apply_legacy_rules(signals: list[LegacySignal], *, faces_found: bool, small: bool) -> None:
    """The v0.3 cross-detector rules (detectors/image/pipeline.py), in their original order."""
    scored = [s for s in signals if s.prob is not None]
    general_strong = sum(1 for s in scored if s.category == "model" and s.prob >= 0.9)
    # specialist asymmetry: a face-swap detector saying "no swap" is weak evidence about a fully synthetic image
    for s in scored:
        if s.category == "face" and s.prob < 0.5:
            s.weight = round(s.weight * (0.25 if general_strong >= 2 else 0.5), 3)
            s.notes.append("Negative result from a swap-specialist detector; reduced weight: it does not rule out a fully "
                           "synthetic image" + (", and whole-image detectors agree the image is generated." if general_strong >= 2
                                                else "."))
    # hypothesis separation: a positive face-swap finding is not refuted by whole-image "real scene" evidence
    face_signals = [s for s in scored if s.category == "face"]
    if faces_found and face_signals:
        top = max(face_signals, key=lambda s: s.weight)
        if top.prob >= 0.6:
            for s in scored:
                if s.category == "model" and s.prob < 0.5:
                    s.weight = round(s.weight * 0.4, 3)
                    s.notes.append("Down-weighted: the scene may be genuine while the face is replaced.")
    if small:  # resolution gate
        for s in signals:
            if s.category in ("model", "face", "forensic"):
                s.weight = round(s.weight * 0.5, 3)


def _analysis_size(ctx: StageContext) -> tuple[int, int] | None:
    decoded, probed = ctx.outputs.get("decode") or {}, ctx.outputs.get("probe") or {}
    if decoded.get("analysis_width"):
        return int(decoded["analysis_width"]), int(decoded["analysis_height"])
    if probed.get("width"):
        return int(probed["width"]), int(probed["height"])
    return None


def fuse(ctx: StageContext) -> dict[str, Any]:
    """A compatible fuser artefact decides (M-2, §4.4); otherwise the legacy fuser publishes the fallback decision."""
    selection = common.select_fusion(ctx, "image")
    if selection.mode == "fallback":
        out = legacy_fusion(ctx)
        out.pop("descriptive_caveats")
        return common.fallback_fusion(ctx, "image", selection, out)
    legacy = legacy_fusion(ctx, apply_rules=False)
    legacy["caveats"] = legacy.pop("descriptive_caveats")
    return common.artefact_fusion(ctx, "image", selection, legacy, common.fusion_entries(ctx, SIGNAL_STAGES))


def legacy_fusion(ctx: StageContext, *, apply_rules: bool = True) -> dict[str, Any]:
    """Legacy fuser → uncalibrated fallback Decision (§4.4), with trusted documentary overrides.

    `apply_rules=False` keeps the v0.3 signals without the cross-detector weight rules: with a fuser artefact loaded those
    hand rules are removed (ML §4.2). `descriptive_caveats` are the input-condition caveats that stay true on either path."""
    manifest = getattr(ctx.state.get("registry"), "manifest", None)
    faces_output = ctx.outputs.get("faces")
    faces_found = bool((faces_output or {}).get("faces"))
    size = _analysis_size(ctx)
    small = size is not None and min(size) < MIN_DETECTOR_SIDE
    work = legacy_signals(common.fusion_entries(ctx, SIGNAL_STAGES), manifest)
    if apply_rules:
        apply_legacy_rules(work, faces_found=faces_found, small=small)

    v1_signals: list[Signal] = []
    v2_signals: list[dict[str, Any]] = []
    for w in work:
        e, aggregate = w.entry, w.entry["aggregate"]
        fields = prob_to_signal_fields(w.prob) if w.prob is not None else {"ai_probability": None, "direction": "inconclusive",
                                                                              "strength": "none"}
        sid = signal_id(e["detector_id"], aggregate)
        reasoning = " ".join([e["reasoning"], *w.notes]).strip()
        v1_signals.append(Signal(id=sid, name=e["name"], category=w.category, observed=e["observed"], reference=e["reference"],
                                 reasoning=reasoning, weight=w.weight, decisive=w.decisive, **fields))
        prefix = aggregate["unit_ref"].split(":", 1)[0]
        v2_signals.append(SignalV2.model_validate({
            "id": sid, "display_id": f"S{len(v2_signals) + 1:02d}", "detector_id": e["detector_id"],
            "detector_version": e["version"], "model_revisions": e["model_revisions"], "name": e["name"], "family": e["family"],
            "modality": e["modality"], "category": w.category, "status": "ok", "scope": SCOPE_BY_PREFIX.get(prefix, "global"),
            "scope_ref": aggregate["unit_ref"], "hypothesis": e["hypotheses"][0] if e["hypotheses"] else None,
            "raw_score": aggregate["raw"], "raw_kind": aggregate["raw_kind"], "effective_weight": w.weight,
            "observed": e["observed"], "reference": e["reference"], "reasoning": reasoning, "metrics": e["metrics"],
            "details": e.get("details") or {},  # N-19n: an entry stored before holds none
            "exhibit_ids": e["exhibit_ids"], "provenance_override": e["provenance_override"],
            "legacy": {"direction": fields["direction"], "strength": fields["strength"], "decisive": w.decisive},
        }).model_dump(mode="json"))

    verdict = legacy_fuse(v1_signals, subject="image")
    informative = any(s.ai_probability is not None for s in v1_signals)
    p, band, confidence = verdict.probability_ai, verdict.band, verdict.confidence
    caveats: list[str] = [FALLBACK_CAVEAT]
    # N-7: one override decides, by the precedence of the calibrated path and the video fallback (signed evidence outranks an
    # unsigned marker); the legacy fuser applied them in signal order. A result that documentary evidence decides is driven by
    # H3, so the v1 view keeps its band (`downgrade_v2`) and the page and the reports read their documentary wording.
    overrides = [w.entry["provenance_override"] for w in work if w.entry["provenance_override"]]
    documentary: str | None = None
    marker: str | None = None
    if overrides:
        statistical = legacy_fuse([s for s, w in zip(v1_signals, work, strict=True) if not w.entry["provenance_override"]],
                                  subject="image")
        watermark = watermark_only([w.entry for w in work])
        marker = marker_kind([w.entry for w in work])  # N-19h (E-21): a marker has the words of its kind
        outcome = apply_override(statistical.probability_ai, overrides, watermark=watermark, marker=marker, kind_of_file="image")
        estimate_caveat = STATISTICAL_ESTIMATE_CAVEAT.split(":", 1)[0]  # D-68: on the fallback the number is a score
        # N-19h (E-19, 5): a score of records alone (metadata, provenance) is no statistical score. Without a check of the
        # content the record decides alone, and no caveat prints the empty "50 / 100" or speaks of a statistical estimate
        content_scored = abstain.content_check_scored(common.fallback_checks(v2_signals))
        caveats += [FALLBACK_STATISTICAL_CAVEAT.format(statistical.probability_ai * 100) if c.startswith(estimate_caveat) else c
                    for c in outcome.caveats if content_scored or not c.startswith(STATISTICAL_CAVEATS)]
        if outcome.kind is not None:
            documentary, confidence = outcome.kind, "high"
        p, band = outcome.probability, band_for(outcome.probability)
    if small:
        caveats.append(LOW_RESOLUTION_CAVEAT)
    if faces_output is not None and not faces_found and common.face_stage_failure(ctx) is None:
        caveats.append(NO_FACE_CAVEAT)
    decoded = ctx.outputs.get("decode") or {}
    if float(decoded.get("scale_to_original") or 1.0) > 1.0:
        caveats.append(f"The image ({decoded['width']}×{decoded['height']}) was analysed at ≤ 4096 px on the long side; "
                       "face boxes are in analysed-image coordinates.")
    if documentary is None:  # L-3: a statistical fallback is never "very likely" or high
        band, confidence = fallback_cap(band, confidence)
    code, verbal = BAND_CODE[band], BAND_HEADLINE[band]
    if documentary == "trusted_capture":  # N-19m (D3-01; D-15): the record's own headline; the code and the band stay
        verbal = CAPTURE_VERBAL
    elif documentary == "decisive_ai":
        # N-19n (the code reviewer's read of N-19m's draft): the decision's own headline is the record's, as the calibrated
        # path and the video fallback give it. Under a declaration (FB, FD) and under a marker whose floor decides (FA, FE,
        # FW, FY) it was the sentence of the band ("Strong indications of AI generation or manipulation"), which the API
        # and the v1 summary read. A marker under a score above its number (FF) keeps the band's: the result is the checks'
        verbal = DECLARED_AI_VERBAL
    if documentary is not None:
        reason = FALLBACK_DECLARED_AI_REASON if documentary == "decisive_ai" else FALLBACK_TRUSTED_CAPTURE_REASON
    else:  # the capped band and confidence, never the fuser's own sentence
        reason = fallback_reason("image", band, confidence, {**verdict.model_dump(mode="json"), "probability_ai": p})
    driving = H3_VALUE if documentary is not None else None
    # N-19h (E-21): the record the decision applied (the documentary one, or the marker whose floor decides)
    deciding = None
    if driving is not None:
        kind = documentary
        applied = [(w.entry, s["id"]) for w, s in zip(work, v2_signals, strict=True) if w.entry["provenance_override"] == kind]
        deciding = record_of_override(kind, [e for e, _ in applied], [sid for _, sid in applied])
    if not informative:
        code, verbal, driving, deciding = "not_assessable", "Not assessable: no detector produced a usable result", None, None
        reason = "No enabled detector returned a score that can be used for this image."  # N-17 (D-68)
    decision = Decision(code=code, verbal=verbal, probability_ai_uncalibrated=round(p, 4) if informative else None, band=band,
                        confidence=confidence if informative else "low", reason=reason, fusion_mode="fallback",
                        driving_hypothesis=driving, provenance_record=deciding,
                        # N-19h (E-20, 3): the statistical score beside a record, as the video stage publishes it, where a
                        # check of the content scored (a score of metadata or provenance records alone is none: D-64)
                        statistical_estimate_before_provenance=(
                            round(statistical.probability_ai, 4) if deciding is not None and informative
                            and abstain.content_check_scored(common.fallback_checks(v2_signals)) else None))
    descriptive = [c for c in caveats if c == LOW_RESOLUTION_CAVEAT or c.startswith("The image (")]
    return {"decision": decision.model_dump(mode="json"),
            "assessments": [{"hypothesis": "H1_whole_content_synthesis", "applicable": False, "note": V1_NOTE}],
            "signals": v2_signals, "caveats": [*caveats, *verdict.caveats], "descriptive_caveats": descriptive}


def synthesise(ctx: StageContext) -> dict[str, Any]:
    """Findings synthesis v2 (M-4): ranked template findings with evidence ids, dissent notes and coverage."""
    return common.synthesise_findings(ctx, kind="image", entry_stages=("provenance", "faces", "detect"),
                                      exhibit_stages=EXHIBIT_STAGES)


def _fit(width: int, height: int, max_px: int) -> tuple[int, int]:
    scale = min(1.0, max_px / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _legacy_artifacts(ctx: StageContext) -> dict[str, str]:
    """Write the v1 artifact files the Folio UI and the v1 PDF read; returns {file name: creating detector}."""
    out_dir = settings.artifacts_dir / ctx.analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shas = {a["id"]: a["sha256"] for stage in EXHIBIT_STAGES for a in (ctx.outputs.get(stage) or {}).get("artefacts", [])}
    produced: dict[str, str] = {}
    boxes = (ctx.outputs.get("faces") or {}).get("faces") or []
    if boxes:
        rgb, _scale = _media(ctx).analysis_copy()
        vis = np.ascontiguousarray(rgb[:, :, ::-1])
        thickness, font_scale = max(2, rgb.shape[1] // 400), max(0.5, rgb.shape[1] / 1600)
        for i, b in enumerate(boxes, 1):
            x, y, w, h = (int(round(b[k])) for k in ("x", "y", "w", "h"))
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), thickness)
            cv2.putText(vis, f"face {i}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 200, 255), 2)
        Image.fromarray(vis[:, :, ::-1]).save(out_dir / "faces.jpg", quality=88)
        produced["faces.jpg"] = "faces"
    for stage in ("provenance", "detect"):
        for raw in (ctx.outputs.get(stage) or {}).get("exhibits", []):
            if (raw["kind"] == "heatmap" and raw.get("preview_artefact_id") in shas
                    and raw["created_by"] not in {LOCAL_GRID_DETECTOR_ID, SHARP_BOUNDARY_DETECTOR_ID}):
                semantics = raw["semantics"]
                name = (f"gradcam_{raw['created_by'].rsplit('.', 1)[-1]}.jpg" if semantics == "attribution"
                        else LEGACY_HEATMAP_ARTIFACTS.get(semantics))
                if name is None:
                    continue
                with Image.open(io.BytesIO(ctx.store.read_bytes(shas[raw["preview_artefact_id"]]))) as im:
                    preview = im.convert("RGB")
                space = raw.get("coordinate_space") or {}
                size = _fit(int(space.get("width") or preview.width), int(space.get("height") or preview.height),
                            LEGACY_ARTIFACT_MAX_PX)
                preview.resize(size, Image.Resampling.BICUBIC).save(out_dir / name, quality=88)
            elif raw["kind"] == "image" and raw.get("role") == "spectrum" and raw.get("payload_artefact_id") in shas:
                name = "spectrum.png"
                (out_dir / name).write_bytes(ctx.store.read_bytes(shas[raw["payload_artefact_id"]]))
            else:
                continue
            produced[name] = raw["created_by"]
    return produced


def render(ctx: StageContext) -> dict[str, Any]:
    """Index the artefacts written by earlier stages (idempotent on their deterministic ids), build the 512 px tile pyramid
    of a source image above 4096 px, and write the v1 artifact files."""
    records = [a for stage in EXHIBIT_STAGES for a in (ctx.outputs.get(stage) or {}).get("artefacts", [])]
    decoded = ctx.outputs.get("decode") or {}
    with ctx.sessions() as db:
        for a in records:
            index_object(db, a["sha256"], a["size_bytes"], media_type=a["media_type"], role=a["role"],
                         created_by=a["created_by"], analysis_id=ctx.analysis_id, artefact_id=a["id"])
        tiles = None
        if max(decoded.get("width", 0), decoded.get("height", 0)) > tile_pyramid.TILE_THRESHOLD_PX:
            tiles = tile_pyramid.build_tileset(db, ctx.store, _media(ctx).native_pixels(), analysis_id=ctx.analysis_id)
        db.commit()
    return {"artefacts_indexed": len(records), "bytes": sum(a["size_bytes"] for a in records), "tiles": tiles,
            "legacy_artifacts": _legacy_artifacts(ctx)}


def build_result(ctx: StageContext) -> AnalysisResultV2:
    exhibits = common.stage_exhibit_list(ctx, EXHIBIT_STAGES)
    tiles = (ctx.outputs.get("render") or {}).get("tiles")
    if tiles:  # heatmaps are registered to a source image the viewer loads as tiles (§4.3, §7.4)
        exhibits = [{**x, "tiles": tiles} if x["kind"] == "heatmap" else x for x in exhibits]
    return common.build_result(ctx, kind="image", pipeline_version=PIPELINE_VERSION,
                               entry_stages=("provenance", "faces", "detect"), exhibit_stages=EXHIBIT_STAGES,
                               exhibits=exhibits)


def v1_view(ctx: StageContext, result: AnalysisResultV2) -> dict[str, Any]:
    """`downgrade_v2(result)` plus the v1 image, faces, metadata and artifact blocks the Folio UI and the v1 PDF read."""
    v1 = downgrade_v2(result)
    if result.legacy is None and result.decision.fusion_mode == "artefact":
        # N-19n (the rest of Q2-14): a result outside what was validated says so in its v1 summary, as a video's does
        profile = result.condition_profile.model_dump(mode="json") if result.condition_profile is not None else None
        v1["verdict"]["summary"] = common.v1_summary(result.decision.model_dump(mode="json"), profile)
    elif result.legacy is None and result.decision.fusion_mode == "fallback":
        # N-19o (code review R9-03): on the fallback a result outside what was validated gives its reason, as the video
        # fallback's v1 summary does (`common.legacy_verdict`); it read "the checks don't point clearly either way"
        profile = result.condition_profile.model_dump(mode="json") if result.condition_profile is not None else None
        decision = result.decision.model_dump(mode="json")
        if abstain.outside_validated(decision, profile):
            v1["verdict"]["summary"] = str(decision["reason"])
    probed = ctx.outputs.get("probe") or {}
    v1["image"] = {k: probed.get(k) for k in ("width", "height", "mode", "format")}
    legacy = (ctx.outputs.get("render") or {}).get("legacy_artifacts") or {}
    v1["artifacts"] = sorted(legacy)
    for s in v1["signals"]:
        creator = s["id"].split(":", 1)[0]
        s["artifacts"] = sorted(name for name, by in legacy.items() if by == creator)
    boxes = (ctx.outputs.get("faces") or {}).get("faces") or []
    scores: list[dict[str, float]] = [{} for _ in boxes]
    for e in _entries(ctx, ("detect",)):
        if e["family"] != "learned-face" or e["status"] != "ok":
            continue
        for u in e["unit_scores"]:
            prefix, _, index = u["unit_ref"].partition(":")
            if prefix == "face" and index.isdigit() and int(index) < len(scores) and u["raw_kind"] == "probability":
                scores[int(index)][e["detector_id"]] = round(u["raw"], 4)
    v1["faces"] = [{"face": {k: int(round(b[k])) for k in ("x", "y", "w", "h")} | {"score": b.get("score")}, "scores": s,
                    **({"p_ai": round(float(np.mean(list(s.values()))), 4)} if s else {})}
                   for b, s in zip(boxes, scores, strict=True)]
    exif = next((x for x in (ctx.outputs.get("provenance") or {}).get("exhibits", [])
                 if x["kind"] == "table" and x["created_by"] == "img.meta.exif" and x.get("title") == "EXIF tags"), None)
    v1["metadata"] = {"exif": {str(r[0]): r[1] for r in exif["rows"]}} if exif else {}
    return v1


def publish(ctx: StageContext) -> dict[str, Any]:
    """Project the evidence tables, store the canonical result and the v1 view. Reports are separate jobs (B11)."""
    result = build_result(ctx)
    if provenance_gate.stop(ctx):
        result.manifest.sampling["image_copy_coverage"] = credential_skip(ctx.outputs["ingest"]["sha256"])
    return common.publish(ctx, result, v1_view(ctx, result), pipeline_version=PIPELINE_VERSION, run_stages=SIGNAL_STAGES)
