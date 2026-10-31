"""Findings synthesis engine (ARCHITECTURE §4.5–4.6; IF §6.1–6.4; UX §5.4, §7; EA-06, EA-07; D-16, D-23, D-26).

For one analysis:
1. **One finding per signal.** Family from the detector (provenance ids, detector family, or the legacy v1 id map for
   derived video signals). Strength from the signal's calibrated log LR, else from its raw probability (uncalibrated,
   capped at moderate); documentary overrides are decisive (trusted C2PA AI declaration) or strong (self-declared
   markers, trusted capture, invalid credentials). Family caps and corroboration follow IF §6.1/§6.4.
2. **Localised findings** carry their top region (`RegionsExhibit` linked to the signal) as scope, with its area and
   whether it overlaps a detected face.
3. **Dissent notes** (EA-06) for every finding whose direction opposes the decision: a different hypothesis is explained
   as a different question (UX §7.5); the same hypothesis names the weighting.
4. **Extra findings:** absent Content Credentials (UX §7.6, info), input conditions and validation status (F9, never
   votes), and **coverage**: checks that ran and did not, "k of n flagged" with the expected number of flags on genuine
   originals (not established until per-check false-positive rates exist), and the known error-rate row for the input's
   context (ML §5.1), or "not established".
5. **Ranking** (§4.5): decisive first, then (with contribution rows) the checks that entered the decision, then review priority, |contribution logit|, strength, and the internal key. Ids
   `F1…` follow the rank (D-26).
Every sentence comes from a template and carries evidence ids: signal ids, exhibit ids, assessment hypotheses, `decision`,
`condition_profile` or `detector:<id>`.
"""
from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core import coords, effect
from app.engine import card_copy
from app.engine.detector import ON_THIS_SERVER, of_the_server
from app.engine.fusion import abstain, decisive
from app.engine.fusion.decisive import OVERRIDE_PROBABILITY, WATERMARK_DETECTORS
from app.engine.synthesis import strength as st
from app.engine.synthesis.render import render, template_name
from app.schemas.evidence import Finding

TEMPLATE_SET = "tpl-2.19"  # 2.19 (N-19q, finding 1): the finding of a check that entered an estimate measured from neutral points stores its stand-alone words (`standalone_strength`, `standalone_direction`), which the sentence of D-76 reads; no sentence of a finding changes. 2.18 (N-19p, design review D6-05): the coverage line counts the statistical checks only ("1 of 1 scored check flagged this file" beside "Only one statistical check scored"; a check that tests the provenance and integrity of the file was counted among them). 2.17 (N-19o, browser QA Q5-05): the finding of a mention of AI in a text field names the field ("… mentions AI in the EXIF image description."; it read "in a text field"), and gives the example "“Not AI-generated” is found the same way" only where the hit is the phrase pattern; (design review D5-03): outside what was validated, where no estimate is shown, a check of little effect is titled "Little effect" (it read "Little effect on the estimate (AI generation)") and the sentence of a check that entered says "In this result …" (it read "In this estimate …"); (D5-04): the finding of a check whose row the decision does not count (a check that declares H3: the container check of a video) is titled by what it tests ("A check of the file’s provenance and integrity, not counted") and says that the decision does not count it; its strength, direction and every count are unchanged (it was titled "Little effect on the estimate (AI generation or manipulation)" and said "In this estimate the check had little effect."). 2.16 (N-19n, one step for the item): a mention of AI in a text field of the metadata has a finding of its own ("… mentions AI in a text field. Veritas found there: …"); a marker that is the IPTC digital source type says what the file declares, with its value; the finding of a marker that decides says "Veritas sets the result's number at no less than 85 out of 100 for it and reports the statistical checks beside it" (it called the record's number an estimate: "the estimate is floored at 85 …"); the note of a record that was not applied says only that it points the other way (the sentence says that it was not applied), and under a marker the note of a check names "the provenance record"; the finding on the checks that ran says which rate is not established, that of the checks' flags taken together (it said "no per-check false-positive rates are on record" beside a detector block that gives one; N-19n, D3-08); the finding of a marker in unsigned metadata says what the check found and where, in the check's words ("The generator is named in the file’s metadata (PNG Software chunk); Veritas lists that tool under “Stable Diffusion / ComfyUI”."; "… has the form that ComfyUI writes (the PNG text chunks “prompt” and “workflow”)."; it quoted Veritas's label as the file's words after "names an AI generator"; N-19n, D3-07, 5). 2.15: the finding of a provenance record that the decision did not apply says that it was not applied and which record outranks it ("… so this is self-declared evidence; it was not applied to this result: the signed capture record outranks it."; it said "the estimate is floored at 85 and the statistical estimate is still reported" under a capture's cap and under a declaration), and the note against it does not call it a check ("This record points the other way than the result. It was not applied: …"; it read "This check points the other way, but the documentary record decides the result") (N-19l; code review R6-03); 2.14: the coverage finding names the error-rate context of the input in words ("text-to-video inputs", "portrait inputs", "talking-head video inputs"; it printed the benchmark's key, "video T2V inputs", "portrait face inputs") (N-19k; the PDF words of both reviews); the coverage finding of a result that no check could be applied to says "No check was applied to this file." (it read "0 of 0 checks produced a result."); a check that scored and is not counted in the estimate says both in one sentence, its score and why, from the record of that check for the input (not in the fitted model; its gate closed, with the condition; no weight in the model, "at chance in testing" only where its own AUC is), in place of "This result provides no information about …" and of the general "it carries no weight in the fused model for this input"; the voice check of a video adds that the sound track can be examined as audio (N-19k; browser QA Q2-12); the finding on the input says once whether a validation slice applies (where a reason says that none applies, the conditions of the input are named as its conditions), and the note of a check against the estimate of a result without a fusion model says that none is loaded only where none is (N-19k; design re-review D2-02, browser QA Q2-14); 2.13: the findings of the record that decides rank before the checks; the finding of a marker that is Content Credentials is worded by its kind; the sentences on the statistical number beside a record (the declaration's, the floor's, the note of a check against the record) end by what stands beside it: reported, not shown outside what was validated, or none where the record decides alone, and say "score" on the fallback (N-19h; E-19 … E-21); 2.12: the coverage finding takes the singular for one check ("1 of 1 check produced a result"); the note of a check that answers another question than the estimate names both questions, and the note of a check against the estimate of a result without a fusion model says so without the build's name for that path (N-19d; browser QA "Copy"); 2.11: the coverage finding says when a step of the run failed and what the result lacks for it (N-18b; operations re-check O4-01); 2.10: a check that entered an estimate measured from neutral points is worded by its contribution (title, strength, direction, sentence), with its stand-alone likelihood ratio named as that; a check outside is said to be outside (N-17, D-63); 2.9: checks skipped for a reason of the server are counted and named as "could not run on this server" (N-14a); 2.8: on the fallback the metadata floor is said as an uncalibrated score (N-7, D-68); 2.7: regions of a rotated photo in oriented original px (N-4); 2.6: benign causes named in client words, never a measurement sentence (L-7); 2.5: regions in original px (L-4)
H1, H2, H3 = "H1_whole_content_synthesis", "H2_localised_manipulation", "H3_provenance_integrity"
FAMILY_BY_DETECTOR = {"prov.c2pa": "F1_provenance", "prov.c2pa.video": "F1_provenance", "img.meta.generator": "F2_self_declared_metadata",
                      "img.meta.exif": "F8_metadata_consistency", "img.meta.consistency": "F8_metadata_consistency",
                      "img.meta.qtables": "F5_compression_history", "img.forensic.jpeg_dq": "F5_compression_history",
                      "img.forensic.copymove": "F7_duplication", "vid.meta.container": "V4_codec", "vid.codec.ffprobe": "V4_codec",
                      "vid.swap.gend_clip_l14": "V1_face_track"}
FAMILY_BY_DETECTOR_FAMILY = {"learned-whole-image": "F3_whole_image_generation", "learned-face": "F10_face",
                             "temporal-face": "V1_face_track", "forensic": "F4_localised_manipulation",
                             "localiser": "F4_localised_manipulation",
                             "codec": "F5_compression_history", "provenance": "F1_provenance",
                             "synthetic-video": "V3_synthetic_video", "av-sync": "V2_lip_sync", "voice": "A1_voice",
                             "text": "T1_ai_text"}
DEFAULT_FAMILY = {"image": "F3_whole_image_generation", "video": "V1_face_track", "document": "T1_ai_text", "audio": "A1_voice"}
MEDIA = {"image": ("image", "images"), "video": ("video", "videos"), "document": ("document", "documents"),
         "audio": ("recording", "recordings")}
AI_SIDE = frozenset({"ai_supported_strong", "ai_supported"})
AUTHENTIC_SIDE = frozenset({"no_ai_indicators", "no_ai_indicators_strong"})
ENVELOPE_PREFIXES = ("no validation metrics", "the measured AUC", "no calibrated fusion model", "this slice lacks validation examples",
                     "too few validation examples", "invalid validation AUC", "the shorter side is", "the frame height is")  # L-3: every envelope reason
REGION_FAMILIES = frozenset({"F4_localised_manipulation", "F5_compression_history", "F7_duplication"})
IOU_CORROBORATION = 0.3
FACE_DETECTOR_FAMILY = "face-detection"
C2PA_DETECTORS = frozenset({"prov.c2pa", "prov.c2pa.video"})
LAST_KEYS = {"absent": "~absent", "conditions": "~conditions", "coverage": "~~coverage"}
STATUS_TEXT = {"not_applicable": "not applicable", "skipped": "skipped", "failed": "failed", "timeout": "timed out"}


@dataclass(frozen=True)
class SynthesisInput:
    kind: str
    decision: Mapping[str, Any]
    signals: Sequence[Mapping[str, Any]]
    entries: Sequence[Mapping[str, Any]] = ()
    exhibits: Sequence[Mapping[str, Any]] = ()
    condition_profile: Mapping[str, Any] | None = None
    calibration: Sequence[Mapping[str, Any]] = ()
    cards: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # detector id → {fp_modes, eval}
    family_by_v1_id: Mapping[str, str] = field(default_factory=dict)
    not_run: Sequence[Mapping[str, Any]] = ()  # the fuse output's `fusion.not_run` ({detector_id, reason}; D-46, D-49)
    failed_steps: Sequence[str] = ()  # N-18b: the caveats of the steps of the run that failed (`effect.step_failed`)


@dataclass
class _Work:
    key: str
    family: str
    variant: str  # score | decisive | trusted_capture | self_declared | invalid | untrusted | absent | conditions | coverage
    strength: str = "none"
    direction: str = "neutral"
    hypothesis: str | None = None
    lane: str | None = None
    template: str = ""
    view: dict[str, Any] = field(default_factory=dict)
    contribution: float | None = None
    probability: float | None = None
    scope: dict[str, Any] | None = None
    box: tuple[float, float, float, float] | None = None
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # corroboration: the top region, or both areas of a clone pair
    on_face: bool = False
    c2pa_invalid: bool = False
    weak_pair: bool = False
    corroborated: list[str] = field(default_factory=list)
    priority: str = "info"
    dissent: str | None = None
    fp_modes: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    benchmark: str | None = None
    reference: str = ""
    uncalibrated: bool = False
    log_lr: float | None = None
    # D-63: `strength` and `direction` are the check's words in this result. For a check that entered an estimate measured from
    # neutral points they come from its contribution, and the stand-alone words (its own likelihood ratio) are kept beside them
    in_estimate: bool = False
    outside_estimate: bool = False
    # N-19o (D5-04): the decision publishes the check's row and does not count it (`Contribution.counted`: a check that
    # declares H3, the container check of a video)
    not_counted: bool = False
    # N-19o (D5-03): the input is outside what was validated, so no estimate is shown (`abstain.outside_validated`)
    unshown: bool = False
    standalone_strength: str = "none"
    standalone_direction: str = "neutral"
    # N-19n (D3-07): the title of a self-declared marker's finding, by its kind and by whether the decision applied it
    title: str | None = None


# --- helpers -----------------------------------------------------------------------------------------------

def _phrase(hypothesis: str | None, family: str) -> str:
    if hypothesis == H1 or (hypothesis is None and family in ("F3_whole_image_generation", "V3_synthetic_video", "T1_ai_text")):
        return "AI generation"
    if hypothesis == H3:
        return "AI generation or manipulation"
    return "manipulation"


def _question(hypothesis: str | None, family: str) -> str:
    """The question a check or the estimate answers, as a clause after "asks" or "evidence about" (N-19d)."""
    if hypothesis == H1 or (hypothesis is None and family in ("F3_whole_image_generation", "V3_synthetic_video", "T1_ai_text")):
        return "whether the content as a whole was generated by AI"
    if hypothesis == H3:
        return "what the file's own records say about its origin"
    return "whether part of the content was altered or replaced"


def _lr_text(log_lr: float | None, uncalibrated: bool) -> str:
    if log_lr is None:
        return "no score"
    if uncalibrated:
        return "from the detector's raw score, not calibrated"
    lr = math.exp(min(abs(log_lr), 20.0))
    value = f"{lr:.0f}" if lr >= 10 else f"{lr:.1f}"
    return f"likelihood ratio about {value}" if log_lr > 0 else f"likelihood ratio about 1/{value}"


def _box(bbox: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return float(bbox["x"]), float(bbox["y"]), float(bbox["x"]) + float(bbox["w"]), float(bbox["y"]) + float(bbox["h"])


def _intersection(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    inter = _intersection(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _conditions(profile: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    all_conditions = list(profile.get("limiting_conditions") or [])
    envelope = [c for c in all_conditions if c.startswith(ENVELOPE_PREFIXES)]
    return [c for c in all_conditions if c not in envelope], envelope


def _source(inp: SynthesisInput) -> tuple[float, float] | None:
    profile = inp.condition_profile or {}
    return (profile["width"], profile["height"]) if profile.get("width") and profile.get("height") else None


def _face_boxes(inp: SynthesisInput) -> list[tuple[float, float, float, float]]:
    """Face boxes in original px (L-4), comparable with the region boxes."""
    face_detectors = {e["detector_id"] for e in inp.entries if e.get("family") == FACE_DETECTOR_FAMILY}
    return [_box(coords.box_to_original(r["region"]["bbox"], x.get("coordinate_space"), _source(inp)))
            for x in inp.exhibits if x.get("kind") == "regions" and x.get("created_by") in face_detectors
            for r in x.get("regions") or [] if r.get("region", {}).get("bbox")]


def _regions_exhibit(signal: Mapping[str, Any], exhibits: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The signal's own regions exhibit (localisers and copy-move write theirs), else the localise stage's regions of its map."""
    own = set(signal.get("exhibit_ids") or [])
    titles = {f"Regions from {x['title']}"[:255] for x in exhibits if x.get("kind") == "heatmap" and x.get("id") in own}
    candidates = [x for x in exhibits if x.get("kind") == "regions" and x.get("regions")]
    for x in candidates:
        if x.get("id") in own:
            return x
    for x in candidates:
        if x.get("created_by") == "localise" and (signal["id"] in (x.get("source_signal_ids") or []) or x.get("title") in titles):
            return x
    return None


def _benchmark(card: Mapping[str, Any] | None) -> str:
    rows = [r for r in (card or {}).get("eval") or [] if r.get("auc") is not None]
    if not rows:
        return "Benchmark performance for this check is not established: no evaluation is on record."
    r = rows[0]
    extra = ", ".join(v for v in (f"n = {r['n']}" if r.get("n") else "", r.get("date") or "") if v)
    return f"On {r['bench']} ({r['slice']}) this check reached AUC {float(r['auc']):.2f}" + (f" ({extra})." if extra else ".")


def _signal_view(s: Mapping[str, Any]) -> dict[str, Any]:
    raw = s.get("raw_score")
    metrics = s.get("metrics")
    legacy_trusted = bool(metrics.get("legacy_trusted")) if isinstance(metrics, Mapping) else False
    return {"id": s["id"], "name": s.get("name") or s["detector_id"], "observed": s.get("observed") or "",
            "score_text": "no score" if raw is None else f"{float(raw):.2f}",
            "exhibit_ids": list(s.get("exhibit_ids") or [])[:2],
            "legacy_trusted": legacy_trusted,
            "outside": s.get("outside_estimate")}  # N-19k: why a check that scored is not counted (`runtime.annotate`)


def _c2pa_invalid(metrics: Mapping[str, Any]) -> bool:
    """I-2 results carry an explicit `invalid` metric (a remote-only manifest is neither valid nor invalid); older
    results only `valid` and `trusted`."""
    if "invalid" in metrics:
        return bool(metrics["invalid"])
    return not metrics.get("valid") and not metrics.get("trusted")


# These identifiers share the explicitly recorded scope contract in image/adapters/learned.py.
_PORTRAIT_CLASSIFIERS = frozenset({"img.hf.dima806", "img.hf.wvolf"})


def _face_purpose(detector: str, hypothesis: str | None) -> str:
    if detector in _PORTRAIT_CLASSIFIERS:
        return "portrait" if hypothesis == H1 else "unknown"
    if detector == "img.dfdc_b7":
        return "swap" if hypothesis == H2 else "unknown"
    return "generation" if hypothesis == H1 else ("manipulation" if hypothesis == H2 else "unknown")


def _face_input_scope(detector: str, metrics: Any) -> str:
    """Interpret recorded adapter facts, never a heatmap canvas, free text or a hypothesis.

    A whole-image flag on a portrait classifier is an ANY aggregation over its face
    inputs. It does not establish that every input was the whole image. Missing or
    inconsistent historical facts remain unknown; the finding is not localisation.
    """
    if not isinstance(metrics, Mapping):
        return "unknown"
    values: dict[str, float | None] = {}
    for key in ("n_faces", "whole_image_input", "no_face"):
        if key not in metrics:
            values[key] = None
            continue
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            return "unknown"
        if (key == "n_faces" and (value < 0 or value != int(value))) or (key != "n_faces" and value not in (0, 1)):
            return "unknown"
        values[key] = value
    count, whole, no_face = (values[key] for key in ("n_faces", "whole_image_input", "no_face"))
    if no_face == 1:
        return ("no_face" if detector in _PORTRAIT_CLASSIFIERS and count in (None, 0) and whole in (None, 1)
                else "unknown")
    if count is None or count <= 0:
        return "unknown"
    if detector in _PORTRAIT_CLASSIFIERS:
        return "portrait" if whole == 1 else ("crop" if whole == 0 else "unknown")
    if detector == "img.dfdc_b7":
        return "crop" if whole in (None, 0) else "unknown"
    if detector == "face.gend_clip_l14":
        return "whole" if whole == 1 and count == 1 else ("crop" if whole == 0 else "unknown")
    return "unknown"


# --- per-signal findings -------------------------------------------------------------------------------------

def _signal_work(inp: SynthesisInput, s: Mapping[str, Any], descriptive: list[str],
                 faces: list[tuple[float, float, float, float]]) -> _Work:
    detector = s["detector_id"]
    v1_id = (s.get("legacy") or {}).get("v1_id") or ""
    # a v0.3 aggregate signal (it has a v1 id) keeps its v0.3 finding family, else the kind's default family
    family = (FAMILY_BY_DETECTOR.get(detector) or inp.family_by_v1_id.get(v1_id) or (DEFAULT_FAMILY.get(inp.kind) if v1_id else None)
              or FAMILY_BY_DETECTOR_FAMILY.get(s.get("family") or "") or DEFAULT_FAMILY.get(inp.kind, "F3_whole_image_generation"))
    override, metrics = s.get("provenance_override"), s.get("metrics") or {}
    card = inp.cards.get(detector)
    w = _Work(key=s["id"], family=family, variant="score", hypothesis=s.get("hypothesis"), lane=s.get("lane"),
              contribution=s.get("contribution_logit"), probability=s.get("calibrated_probability"),
              fp_modes=list((card or {}).get("fp_modes") or s.get("fp_modes") or []), conditions=list(descriptive),
              reference=s.get("reference") or "")
    if override == "decisive_ai":
        w.family, w.variant, w.strength, w.direction = "F1_provenance", "decisive", "decisive", "for"
    elif override == "trusted_capture":
        w.family, w.variant, w.strength, w.direction = "F1_provenance", "trusted_capture", "strong", "against"
    elif override == "self_declared_ai" and detector in WATERMARK_DETECTORS:
        w.family, w.variant, w.strength, w.direction = "F1_provenance", "self_declared_watermark", "strong", "for"
    elif override == "self_declared_ai":
        w.family, w.variant, w.strength, w.direction = "F2_self_declared_metadata", "self_declared", "strong", "for"
    elif detector in C2PA_DETECTORS and s.get("raw_kind") == "score" and _c2pa_invalid(metrics):
        w.variant, w.strength, w.direction, w.c2pa_invalid, w.hypothesis = "invalid", "strong", "for", True, H3
    else:
        if detector in C2PA_DETECTORS and metrics.get("valid"):
            w.variant = "trusted_other" if metrics.get("trusted") else "untrusted"
        log_lr = s.get("log_lr")
        if (log_lr is None and s.get("raw_kind") == "probability" and s.get("raw_score") is not None
                and 0.0 <= float(s["raw_score"]) <= 1.0):  # D-49: no log LR from an unusable probability
            p = min(max(float(s["raw_score"]), 1e-6), 1 - 1e-6)
            log_lr, w.uncalibrated = math.log(p / (1 - p)), True
        w.log_lr = log_lr
        raw_strength = "none" if log_lr is None else st.band(log_lr)
        w.strength = st.cap(raw_strength, family, uncalibrated=w.uncalibrated)
        w.direction = st.direction(log_lr, w.strength)
        w.benchmark = _benchmark(card)
    scope = "global"
    region_view = None
    regions = _regions_exhibit(s, inp.exhibits) if w.family in REGION_FAMILIES else None
    if regions is not None:
        top = regions["regions"][0]
        bbox = top["region"]["bbox"]
        space, source = regions.get("coordinate_space"), _source(inp)
        w.box = _box(coords.box_to_original(bbox, space, source))  # L-4: original px, the viewer's rule
        w.boxes = [w.box]
        partner = None
        if w.family == "F7_duplication" and len(regions["regions"]) > 1 and regions["regions"][1]["region"].get("bbox"):
            other = _box(coords.box_to_original(regions["regions"][1]["region"]["bbox"], space, source))  # C1′, the second area of the top clone pair
            w.boxes.append(other)
            partner = {"display_id": regions["regions"][1].get("display_id") or "C1′", "x0": round(other[0]), "x1": round(other[2]),
                       "y0": round(other[1]), "y1": round(other[3])}
        w.on_face = any(_intersection(box, face) > 0 for box in w.boxes for face in faces)
        share = coords.area_share(bbox, space, source)
        area = None if share is None else 100.0 * share
        w.scope = {"kind": "bbox", "bbox": {k: float(bbox[k]) for k in ("x", "y", "w", "h")}}
        region_view = {"exhibit_id": regions["id"], "count": len(regions["regions"]), "display_id": top.get("display_id") or "R1",
                       "area_text": "an unknown share" if area is None else f"{area:.1f} %", "x0": round(w.box[0]),
                       "x1": round(w.box[2]), "y0": round(w.box[1]), "y1": round(w.box[3]), "on_face": w.on_face,
                       "partner": partner, "pairs": max(1, len(regions["regions"]) // 2)}
        scope = "region"
    w.template = template_name(w.family, scope)
    w.view = {"s": _signal_view(s), "region": region_view}
    if w.variant == "self_declared":  # N-19h (E-21): the words of the marker's kind, from the signal's detector and metrics
        w.view["marker_carries"], w.view["marker_why"] = _marker_words(s, inp.kind)
        # N-19n (Q4-01): the IPTC source type has a sentence of its own, with the file's own value; where it is the marker
        # (no generator is named) it is the finding's first sentence, and no label is quoted as the file's words
        w.view["marker_source_type"] = _source_type_sentence(s, inp.kind)
        w.view["marker_named"] = _marker_kind(s) in NAMES_OR_SIGNS
        # N-19n (D3-07, 9): a declaration by the phrase, and parameters without a tool, say what the check found in its
        # own words: no label of the check is quoted as the file's words
        w.view["marker_holds"] = _holds_sentence(s, inp.kind)
        # N-19n (D3-07, 5): what the check found and where, from the field that the check's result has for it; under the
        # title "Self-declared: the file’s metadata names an AI generator" the sentence does not say the title again, and a
        # name of Veritas for a tool is given as Veritas's. None for a signal stored without the field, which keeps its words
        w.view["marker_found"] = _found_sentences(s)
    elif w.family == "F2_self_declared_metadata":
        # N-19n (browser QA Q4-01, P1): the metadata check's other outcome, a hit in free text, is a finding of its score and
        # no marker. Its template asked for the marker's words, which are set above only: the step failed for every such file
        w.view["mention"] = _mention(s)
        w.view["mention_where"], w.view["mention_phrase"] = _mention_where(s), _mention_phrase(s)  # N-19o (Q5-05)
    if w.family == "F10_face":
        w.view.update(face_purpose=_face_purpose(detector, w.hypothesis), face_input_scope=_face_input_scope(detector, metrics))
    return w


def _corroborate(work: list[_Work]) -> None:
    """IF §6.4: regions flagged by different families with IoU ≥ 0.3 upgrade one level; two weak ones raise priority. A clone
    pair (F7) takes part through either of its areas."""
    located = [w for w in work if w.boxes and w.direction == "for" and w.strength != "none"]
    for i, a in enumerate(located):
        for b in located[i + 1:]:
            if a.family != b.family and any(iou(x, y) >= IOU_CORROBORATION for x in a.boxes for y in b.boxes):
                a.corroborated.append(b.key)
                b.corroborated.append(a.key)
                if a.strength == b.strength == "weak":
                    a.weak_pair = b.weak_pair = True
    for w in located:
        if w.corroborated:
            w.strength = st.cap(st.upgrade(w.strength), w.family, corroborated=True, uncalibrated=w.uncalibrated)


SCORE_VARIANTS = frozenset({"score", "trusted_other", "untrusted"})  # findings worded by a likelihood ratio, never documentary


def _in_estimate(w: _Work, measured: bool, uncounted: Collection[str] = ()) -> None:
    """D-63 (`app.core.effect`): in a result whose contributions are measured from neutral points, a check that entered the
    estimate takes direction and strength from its contribution; its stand-alone words stay beside them. A check outside the
    estimate keeps its stand-alone words and is said to be outside. N-19o (D5-04): a check whose row the decision publishes
    and does not count (`uncounted`) is titled and said as not counted; its strength and direction stay those of its row,
    so no count changes (it was "Little effect on the estimate" beside a breakdown that says the row is not counted)."""
    w.standalone_strength, w.standalone_direction = w.strength, w.direction
    if not measured or w.variant not in SCORE_VARIANTS:
        return
    w.not_counted = w.key in uncounted
    if effect.entered(w.contribution):
        w.in_estimate = True
        w.strength, w.direction = effect.strength(w.contribution), effect.direction(w.contribution)
    else:
        w.outside_estimate = True


OUTSIDE_GENERAL = "This check is not in this estimate: it carries no weight in the fused model for this input."


def _outside_view(w: _Work, kind: str) -> dict[str, Any]:
    """N-19k (browser QA Q2-12): what a check outside the estimate says of that. A check that scored and whose signal
    carries the fusion's record of why (`outside_estimate`) gets one sentence with both halves, its score and why it is
    not counted, from that record and from what the check reached in testing (`effect.outside_why`): it replaces "This
    result provides no information about …" beside a score of 0.91, and the general sentence that gave no cause. The voice
    check of a video adds what the examiner can do. Any other check outside the estimate keeps the general sentence."""
    signal = w.view.get("s") or {}
    record, score = signal.get("outside"), signal.get("score_text")
    if not w.outside_estimate:
        return {"outside_both": None, "outside_line": None, "outside_step": None}
    if not record or score in (None, "no score"):
        return {"outside_both": None, "outside_line": OUTSIDE_GENERAL, "outside_step": None}
    why = effect.outside_why(record, w.benchmark)
    plain = w.strength == "none"
    return {"outside_both": effect.outside_sentence(score, why) if plain else None,
            "outside_line": None if plain else effect.outside_sentence(score, why, standalone=True),
            "outside_step": effect.VOICE_STEP if w.family == "A1_voice" and kind == "video" else None}


def _standalone_sentence(w: _Work) -> str:
    """The stand-alone likelihood ratio of a check that entered the estimate, named as stand-alone (D-63)."""
    phrase = _phrase(w.hypothesis, w.family)
    if w.standalone_strength == "none" or w.standalone_direction == "neutral":
        return f"Taken alone, the score is uninformative about {phrase}."
    side = "for" if w.standalone_direction == "for" else "against"
    return f"Taken alone, the score is {w.standalone_strength} support {side} {phrase} ({_lr_text(w.log_lr, w.uncalibrated)})."


def _dissent(w: _Work, decision_direction: str | None, driving: str | None, fallback: bool, beside: str = "reported",
             unused: bool = False, outranked: str | None = None, applied: str | None = None) -> str | None:
    """`applied`: the kind of the provenance record that the decision applied (`Decision.provenance_record`), where it names one."""
    if decision_direction is None or w.direction in ("neutral", decision_direction) or w.strength == "none":
        return None
    if outranked:
        # N-19l (R6-03, 2): a record that the decision did not apply is no check. N-19n (D3-07): the note says what the
        # finding's sentence does not; that it was not applied, and which record outranks it, is said there, once
        return "This record points the other way than the result."
    if driving == H3:
        # N-19n (D3-07): a marker is no documentary record; that word is a declaration's and a capture's
        record = "provenance" if applied == "self_declared_ai" else "documentary"
        return (f"This check points the other way, but the {record} record decides the result; "
                f"{_statistical_clause(beside, fallback, 'is reported beside it')}.")
    if w.hypothesis and driving and w.hypothesis != driving:
        # L-7 (QA P2): the note no longer says the result "does not contradict" an estimate it is listed against. N-19d (QA
        # "Copy"): each question is named, so the note never reads "a result against AI generation or manipulation does not
        # remove the evidence about AI generation"
        asked, led = _question(w.hypothesis, w.family), _question(driving, w.family)
        return (f"This check asks {asked}; the estimate rests on evidence about {led}. These are two questions: an answer to "
                "one does not settle the other.")
    if fallback:  # N-19d: without the build's name for this path in brackets ("hand-set weights (uncalibrated fallback)")
        # N-19k (Q2-14): "loaded" only where none is (`unused`: one is installed and this analysis did not use it)
        return ("This check points the other way from the estimate. "
                f"No fusion model {'was used for this analysis' if unused else 'is loaded'}: Veritas weights each check by "
                "hand-set weights, so one check can differ from the combined estimate.")
    return ("This check points the other way from the estimate. Veritas weights each check by its tested reliability, so one check "
            "can differ from the combined estimate.")


# --- extra findings -------------------------------------------------------------------------------------------

def _found_no_manifest(entry: Mapping[str, Any]) -> bool:
    """The entry of a Content Credentials check that read the file and found no manifest ("not applicable: no C2PA manifest"),
    not one whose container it cannot read: a check that ran, whose finding is "No Content Credentials" (N-19q, finding 4)."""
    return (entry.get("detector_id") in C2PA_DETECTORS and entry.get("status") == "not_applicable"
            and (entry.get("metrics") or {}).get("unsupported_container") != 1.0)


def _absent_credentials(inp: SynthesisInput) -> _Work | None:
    entry = next((e for e in inp.entries if _found_no_manifest(e)), None)
    if entry is None:
        return None
    return _Work(key=LAST_KEYS["absent"], family="F1_provenance", variant="absent", hypothesis=H3,
                 template="F1_provenance/global.j2", view={"detector_ref": f"detector:{entry['detector_id']}", "variant": "absent"})


NO_SLICE = "no validation slice applies"  # `runtime.FALLBACK_ENVELOPE` and its form for a model that was not used


def _conditions_work(profile: Mapping[str, Any] | None, descriptive: list[str], envelope: list[str]) -> _Work | None:
    if not profile or not (descriptive or envelope):
        return None
    return _Work(key=LAST_KEYS["conditions"], family="F9_distribution_conditions", variant="conditions",
                 template="F9_distribution_conditions/global.j2", conditions=list(descriptive),
                 view={"conditions": descriptive, "envelope": envelope, "slice_id": profile.get("slice_id") or "unknown",
                       "no_slice": any(NO_SLICE in reason for reason in envelope),
                       "measured": not str(profile.get("slice_id") or "unknown").endswith("unknown")})


def _contexts(kind: str, profile: Mapping[str, Any]) -> list[str]:
    """ML §5.1 error-rate contexts that apply to this input, most specific first."""
    slice_id = str(profile.get("slice_id") or "")
    if kind == "image":
        out = []
        if profile.get("screenshot_likelihood"):
            out.append("screenshot")
        if "/face<120" in slice_id:
            out.append("small face (< 120 px)")
        elif "/face" in slice_id and "/noface" not in slice_id:
            out.append("portrait face")
        if "/side<512" in slice_id:
            out.append("low-res (< 512)")
        if "/q<70" in slice_id:
            out.append("social-recompressed")
        return out
    if kind == "video":
        return ["video talking head"] if "/face" in slice_id and "/noface" not in slice_id else ["video T2V"]
    lossy = any(str(c).startswith("lossy compression") for c in (profile or {}).get("limiting_conditions") or [])
    return ["audio lossy"] if kind == "audio" and lossy else []  # L-3: a FLAC or WAV recording is not lossy


def _per_100(rate: float) -> str:
    value = 100.0 * rate
    return "fewer than 1" if 0 < value < 0.5 else f"{value:.0f}"


def _error_rate_sentence(inp: SynthesisInput) -> str:
    media = MEDIA.get(inp.kind, ("file", "files"))[1]
    contexts = _contexts(inp.kind, inp.condition_profile or {})
    rows = [r for c in inp.calibration if str(c.get("subject", "")).startswith("fuser-") for r in c.get("error_rates") or []]
    for context in contexts:
        row = next((r for r in rows if r.get("context") == context and r.get("threshold_kind") == "conformal_0.05"), None)
        if row is not None:
            return (f"In testing ({row['bench_version']}, {effect.context_words(context)} inputs), {_per_100(row['fpr'])} in 100 genuine {media} scored "
                    f"above the 5 % threshold (n = {row['n_neg']}), and {_per_100(row['tpr'])} in 100 AI-generated or "
                    f"manipulated {media} did (n = {row['n_pos']}).")
    if contexts:
        return f"Known error rates for {effect.context_words(contexts[0])} inputs are not established: no evaluation is on record."
    return f"No evaluation is on record for this kind of {MEDIA.get(inp.kind, ('file', 'files'))[0]}, so known error rates are not established."


def _coverage_work(inp: SynthesisInput, work: list[_Work]) -> _Work:
    entries = list(inp.entries)
    by_id = {e["detector_id"]: e for e in entries}
    capped = {str(n["detector_id"]): str(n.get("reason") or "did not run") for n in inp.not_run}
    refs = [f"detector:{e['detector_id']}" for e in entries] or ["decision"]
    # N-14a: a check skipped for a reason of the server is counted and named as one that could not run on this server
    server = [e for e in entries if of_the_server(e.get("status"), e.get("reason"))]
    not_run = [{"ref": f"detector:{e['detector_id']}", "name": e.get("name") or e["detector_id"],
                "status": ON_THIS_SERVER if e in server else STATUS_TEXT.get(str(e.get("status")), str(e.get("status"))),
                "reason": e.get("reason") or ""}
               for e in entries if e.get("status") != "ok" and not _found_no_manifest(e)]
    # N-19q (finding 4): a Content Credentials check that found no manifest ran, as its finding "No Content Credentials" says;
    # it read "Not run: C2PA Content Credentials (not applicable: “no C2PA manifest”)" beside that finding. It gives no result
    # that is counted (the count of checks that produced a result is the coverage's, as every other line counts it)
    no_manifest = [{"ref": f"detector:{e['detector_id']}", "name": e.get("name") or e["detector_id"]} for e in entries if _found_no_manifest(e)]
    # D-49: an `ok` entry whose score fusion could not use, and a gate-open artefact detector without an entry, which cites the
    # decision because the manifest lists only detectors with entries
    not_run += [{"ref": f"detector:{d}", "name": by_id[d].get("name") or d, "status": why, "reason": ""}
                for d, why in capped.items() if d in by_id and by_id[d].get("status") == "ok"]
    not_run += [{"ref": "decision", "name": d, "status": why, "reason": ""} for d, why in capped.items() if d not in by_id]
    # N-19p (design review D6-05): the statistical checks only, as "Only one statistical check scored" counts them: a check that
    # tests the file's provenance and integrity (H3) is not one (it read "1 of 2 scored checks flagged this file" there)
    scored = [w for w in work if w.variant in ("score", "untrusted") and w.log_lr is not None and w.key.split(":")[0] not in capped
              and w.hypothesis != H3]
    flagged = [w for w in scored if w.direction == "for"]
    rate_refs = ["condition_profile"] if inp.condition_profile else ["decision"]
    n_ran = sum(e.get("status") == "ok" and e["detector_id"] not in capped for e in entries)
    return _Work(key=LAST_KEYS["coverage"], family="coverage", variant="coverage", template="coverage/global.j2",
                 view={"detector_refs": refs, "n_total": len(entries) + sum(d not in by_id for d in capped), "n_ran": n_ran,
                       "n_server": len(server), "server_refs": [f"detector:{e['detector_id']}" for e in server],
                       "failed_steps": list(inp.failed_steps),
                       "not_run": not_run[:8], "no_manifest": no_manifest, "n_not_run_more": max(0, len(not_run) - 8), "n_scored": len(scored),
                       "n_flagged": len(flagged), "flag_refs": [w.key for w in flagged] or refs,
                       "error_rate_sentence": _error_rate_sentence(inp), "rate_refs": [*rate_refs, "decision"]})


def _marker_title(w: _Work, s: Mapping[str, Any], kind: str, unapplied: bool) -> str | None:
    """N-19n (D3-07): the finding of a self-declared marker is a record's, not a likelihood's: "Self-declared: " and what the
    file carries in the words of the marker's kind, as the watermark message's has; "Self-declared, not applied: " where the
    decision did not apply it. It read "Strong support for AI generation or manipulation", which no marker has."""
    if w.variant == "self_declared_watermark":
        carries = "a public AI-generator watermark message"
    elif w.variant == "self_declared":
        carries = decisive.marker_carries(_marker_kind(s), kind).replace("'", "’")
        carries = carries[0].lower() + carries[1:]
    else:
        return None
    return f"Self-declared{', not applied' if unapplied else ''}: {carries}"


NOT_COUNTED_TITLE = "A check of the file’s provenance and integrity, not counted"


def _title(w: _Work) -> str:
    if w.title is not None:
        return w.title
    fixed = {"decisive": "Documentary: Content Credentials declare AI generation",
             "trusted_capture": "Documentary: Content Credentials record a capture of these bytes",
             "self_declared_watermark": "Self-declared: a public AI-generator watermark message",
             "absent": "No Content Credentials", "conditions": "Input conditions and validation", "coverage": "Checks that ran",
             # N-19n (D3-07, 8): credentials that do not validate are no check and no marker: a title of their own (it read
             # "Strong support for modification after signing", with three bars)
             "invalid": "Content Credentials that do not validate"}
    if w.variant in fixed:
        return fixed[w.variant]
    if w.not_counted:  # N-19o (D5-04): what the check tests, as the breakdown says of its row
        return NOT_COUNTED_TITLE
    phrase = "modification after signing" if w.variant == "invalid" else _phrase(w.hypothesis, w.family)
    if w.strength == "none":
        # D-63: a check that entered the estimate is named for what it did there, not for its stand-alone likelihood ratio
        if w.in_estimate:  # N-19o (D5-03): outside what was validated no estimate is shown, so the title names none
            return "Little effect" if w.unshown else f"Little effect on the estimate ({phrase})"
        return f"Uninformative about {phrase}"
    title = f"{w.strength.capitalize()} support {'for' if w.direction == 'for' else 'against'} {phrase}"
    # N-19q (N-20b's read of the reports, finding 11): where the check's part in the estimate and its stand-alone reading give
    # two grades, the heading names its reading (F1 was headed "Strong support for AI generation" beside "Taken alone, the
    # score is weak support for AI generation (likelihood ratio about 9.5)")
    if w.in_estimate and (w.standalone_strength, w.standalone_direction) != (w.strength, w.direction):
        return f"{title} in {'this result' if w.unshown else 'this estimate'}"
    return title


def _rank_key(w: _Work, rows: bool = False, record: Collection[str] = ()) -> tuple:
    """`rows`: the decision has contribution rows (artefact fusion); then a check that entered it (non-zero contribution) ranks
    before every check that carries no weight in it (L-3), so F1 is never a check outside the fused estimate. `record`
    (N-19h, E-21): the signals of the record that decides; its findings entered the decision and rank with the documentary
    ones, before every check (a marker's finding and a capture's ranked after the checks, as findings without weight)."""
    of_record = w.key in record
    outside = rows and not of_record and not (w.contribution is not None and abs(w.contribution) > 1e-9)
    return (0 if w.strength == "decisive" or of_record else 1, 1 if outside else 0, st.PRIORITY_ORDER[w.priority],
            -abs(w.contribution or 0.0), -st.RANK[w.strength], w.key)


# N-19h (E-21): why a marker of each kind is self-declared evidence, for the sentence of its finding. Nothing says of Content
# Credentials that metadata can be edited, and nothing says of a signer on the trust list that anyone can sign
_MARKER_WHY = {
    decisive.METADATA: "Metadata can be added or edited by anyone, so this is self-declared evidence",
    decisive.CREDENTIALS_LISTED: "The credentials declare no AI generation, so the named generator counts as self-declared evidence",
    decisive.CREDENTIALS_UNLISTED_DECLARED: "Anyone can sign such credentials, so this is self-declared evidence",
    decisive.CREDENTIALS_UNLISTED_NAMED: "Anyone can sign such credentials, so this is self-declared evidence",
    decisive.METADATA_DECLARED: "Metadata can be added or edited by anyone, so this is self-declared evidence",
    decisive.METADATA_COMPOSITE: "Metadata can be added or edited by anyone, so this is self-declared evidence",
    decisive.METADATA_PARAMETERS: "Metadata can be added or edited by anyone, so this is self-declared evidence",
}


def _marker_kind(s: Mapping[str, Any]) -> str:
    marker = decisive.marker_kind([{**s, "provenance_override": "self_declared_ai"}])
    return marker if marker in _MARKER_WHY else decisive.METADATA


def _marker_words(s: Mapping[str, Any], kind: str) -> tuple[str, str]:
    """(what the file carries, why that is self-declared) for the finding of a marker that is no watermark message, by the
    kind of the marker (`decisive.marker_kind`): unsigned metadata keeps its sentences."""
    marker = _marker_kind(s)
    return decisive.marker_carries(marker, kind), _MARKER_WHY[marker]


# the kinds whose finding quotes what the record names (a generator, in metadata or in Content Credentials)
NAMES_OR_SIGNS = frozenset({decisive.METADATA, decisive.CREDENTIALS_LISTED, decisive.CREDENTIALS_UNLISTED_DECLARED,
                            decisive.CREDENTIALS_UNLISTED_NAMED})


def _joined(parts: Sequence[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _found(s: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """N-19n (D3-07, 5): what the metadata check found and where (`GeneratorMetadataDetector`: the tools, the labels that
    name no tool, the fingerprints with their fields, the labels of each group of places, the source type's value). None
    for another check and for a signal stored before the check's result had the field."""
    details = s.get("details")
    if s.get("detector_id") in decisive.CREDENTIAL_DETECTORS or s.get("detector_id") in WATERMARK_DETECTORS:
        return None
    return details if isinstance(details, Mapping) and isinstance(details.get("places"), list) else None


def _places(found: Mapping[str, Any], labels: Collection[str]) -> str:
    """The check's groups of places that hold one of these labels, in the check's words and joined: "PNG Software chunk and
    EXIF Software/Make/Model"."""
    groups = [str(row["place"]) for row in found["places"] if isinstance(row, Mapping) and set(row.get("labels") or ()) & set(labels)]
    return _joined(groups) if groups else "a structured field"


_PNG_TEXT = re.compile(r"PNG text '(.+)'")
STEALTH_FIELD = "alpha-channel least significant bits"  # `metadata_rules.generator_fingerprints`: no field of the metadata


def _form_fields(fields: Sequence[str]) -> str:
    """The fields of a fingerprint in a reader's words: "the PNG text chunks “prompt” and “workflow”", "the EXIF field Make"."""
    chunks = [m.group(1) for f in fields if (m := _PNG_TEXT.fullmatch(f))]
    words = [f"the PNG text chunk{'s' if len(chunks) > 1 else ''} {_joined([f'“{c}”' for c in chunks])}"] if chunks else []
    words += [f"the EXIF field {f.split(' ', 1)[1]}" if f.startswith("EXIF ") else f for f in fields if not _PNG_TEXT.fullmatch(f)]
    return _joined(words)


def _found_sentences(s: Mapping[str, Any]) -> list[str] | None:
    """N-19n (D3-07, 5): the first sentences of a marker in unsigned metadata that names a generator or has the form of one's
    files. A fingerprint: "The file’s metadata has the form that ComfyUI writes (the PNG text chunks “prompt” and “workflow”)."
    A tool in a structured field: "The generator is named in the file’s metadata (PNG Software chunk); Veritas lists that tool
    under “Stable Diffusion / ComfyUI”." (the label is Veritas's name for a family of tools, not the file's words). None for
    another kind of marker, whose sentence is `_holds_sentence` or the source type's, and without the check's field."""
    found = _found(s)
    if found is None or _marker_kind(s) != decisive.METADATA:
        return None
    out: list[str] = []
    prints = [fp for fp in found.get("fingerprints") or [] if isinstance(fp, Mapping) and fp.get("fields")]
    forms = [f"the form that {fp['generator']} writes ({_form_fields(fp['fields'])})" for fp in prints if STEALTH_FIELD not in fp["fields"]]
    if forms:
        out.append(f"The file’s metadata has {_joined(forms)}.")
    if len(forms) < len(prints):
        out.append("The picture’s alpha channel holds a stealth pnginfo payload in its least significant bits, as NovelAI and a "
                   "web UI extension write one.")
    tools = [str(t) for t in found.get("tools") or []]
    if tools:
        one, also = len(tools) == 1, " also" if out else ""
        out.append(f"{'The generator is' if one else 'Generators are'}{also} named in the file’s metadata ({_places(found, tools)}); "
                   f"Veritas lists {'that tool' if one else 'those tools'} under {_joined([f'“{t}”' for t in tools])}.")
    if found.get("parameters") and not prints:
        out.append(f"The file’s metadata also holds Stable-Diffusion style generation parameters ({found['parameters']}).")
    return out or None


def _holds_sentence(s: Mapping[str, Any], kind: str) -> str | None:
    """N-19n (D3-07, 9): the first sentence of a marker in unsigned metadata that names no generator and holds no AI
    source type: the phrase in a field that software writes, or generation parameters without a tool. None otherwise.
    The place stands in brackets, in the check's words (D3-07, 5), where the signal holds what the check found."""
    marker, found = _marker_kind(s), _found(s)
    metrics = s.get("metrics") if isinstance(s.get("metrics"), Mapping) else {}
    if marker == decisive.METADATA_DECLARED and metrics.get("phrase_declared") and not metrics.get("source_type_trained"):
        phrases = [str(p) for p in (found or {}).get("phrases") or []]
        held = _joined([MENTION_LABELS.get(p, p) for p in phrases]) if phrases else MENTION_LABELS[PHRASE_LABEL]
        place = f" ({_places(found, phrases)})" if found is not None else ""
        return f"{decisive.marker_carries(marker, kind)}: a field that software writes holds {held}{place}.".replace("'", "’")
    if marker == decisive.METADATA_PARAMETERS:
        place = f" ({found['parameters']})" if found is not None and found.get("parameters") else ""
        return f"The file’s metadata holds Stable-Diffusion style generation parameters{place}; no tool is named."
    return None


def _source_type_sentence(s: Mapping[str, Any], kind: str) -> str | None:
    """N-19n: "The file’s metadata declares AI generation: its IPTC digital source type is trainedAlgorithmicMedia.", or
    the sentence of a composite, with the file's own value (the metadata check's metrics); None without such a source type."""
    metrics = s.get("metrics") if isinstance(s.get("metrics"), Mapping) else {}
    if s.get("detector_id") in decisive.CREDENTIAL_DETECTORS or s.get("detector_id") in WATERMARK_DETECTORS:
        return None
    for metric, marker, value in (("source_type_trained", decisive.METADATA_DECLARED, "trainedAlgorithmicMedia"),
                                  ("source_type_composite", decisive.METADATA_COMPOSITE, "compositeWithTrainedAlgorithmicMedia")):
        if metrics.get(metric):
            return f"{decisive.marker_carries(marker, kind)}: its IPTC digital source type is {value}.".replace("'", "’")
    return None


# N-19n (Q4-01): in the finding of a mention the check's label "Generative-AI marker" (the pattern of "AI generated",
# "generative AI") is printed in these words: the sentence would else call the hint a marker
# and "IPTC DigitalSourceType = trainedAlgorithmicMedia" claims a field where the word stood in a caption
PHRASE_LABEL = "Generative-AI marker"
MENTION_LABELS = {PHRASE_LABEL: "a phrase such as “AI-generated”",
                  "IPTC DigitalSourceType = trainedAlgorithmicMedia": "the term trainedAlgorithmicMedia"}
MENTION_UNNAMED = "a term of its list of AI tools"


def _mention_where(s: Mapping[str, Any]) -> str:
    """N-19o (browser QA Q5-05): the fields that hold the mention, from the check's result (`details.mentions`): "the EXIF
    image description", "the PNG text chunk “Comment”"; "a text field" for a signal stored before the check named them."""
    details = s.get("details")
    rows = details.get("mentions") if isinstance(details, Mapping) else None
    fields = list(dict.fromkeys(str(r["field"]) for r in rows or [] if isinstance(r, Mapping) and r.get("field")))
    return _joined(fields) if fields else "a text field"


def _mention_phrase(s: Mapping[str, Any]) -> bool:
    """N-19o (Q5-05): whether the hit is the phrase pattern, where the example "“Not AI-generated” is found the same way"
    applies (it stood under "Made with Midjourney v7." too)."""
    return PHRASE_LABEL in [label.strip() for label in str(s.get("observed") or "").split(";")]


def _mention(s: Mapping[str, Any]) -> str:
    """What the metadata check matched in a text field: its labels for the patterns (`metadata.GENERATOR_PATTERNS`), joined
    as the check joins them. They are the check's words, not the file's, so they stand in no quotation marks."""
    labels = [label.strip() for label in str(s.get("observed") or "").split(";") if label.strip()]
    return "; ".join(MENTION_LABELS.get(label, label) for label in labels) or MENTION_UNNAMED


# N-19h (E-19, E-20): what a finding says of the statistical number that stands beside a record
REPORTED, WITHHELD, NO_NUMBER = "reported", "withheld", "none"


def _beside(decision: Mapping[str, Any], profile: Mapping[str, Any] | None) -> str:
    """Whether the statistical checks' own number stands beside the record of this result. `none`: the result holds none
    (a record decides alone: a check did not run, no check that scored has weight for the input, or on the fallback no
    check of the content scored). `withheld`: it holds one and the input is outside what was validated, where the number
    is not shown beside a record (E-20, 8). Otherwise `reported`."""
    if decision.get("statistical_estimate_before_provenance") is None:
        return NO_NUMBER if decision.get("driving_hypothesis") == H3 else REPORTED
    return WITHHELD if abstain.outside_validated(decision, profile) else REPORTED


def _statistical_clause(beside: str, fallback: bool, reported: str) -> str:
    """"the statistical estimate <reported>", or in its place that the checks gave none, or that it is not shown and why.
    On the fallback the number is a score, never an estimate (D-68)."""
    number = "score" if fallback else "estimate"
    if beside == NO_NUMBER:
        return f"the statistical checks gave no {number}"
    if beside == WITHHELD:
        return f"the statistical {number} is not shown, because this input is outside what was validated"
    return f"the statistical {number} {reported}"


# N-19l (code review R6-03, 2): the record that outranks a record which the decision did not apply, by the kind applied
OUTRANKS = {"trusted_capture": "the signed capture record outranks it", "decisive_ai": "the trusted declaration outranks it"}


def _outranked(signal: Mapping[str, Any], decision: Mapping[str, Any]) -> str | None:
    """N-19l (R6-03, 2): for the signal of a provenance record that the decision did not apply, which record outranks it;
    None for the record that decides, for a signal that is no record, and where no record decides."""
    own, applied = signal.get("provenance_override"), (decision.get("provenance_record") or {}).get("kind")
    return OUTRANKS.get(str(applied)) if own and applied and own != applied else None


def _checks_decide(signal: Mapping[str, Any], decision: Mapping[str, Any], fallback: bool) -> str | None:
    """N-19n (D3-07; Q, FF): for the signal of a self-declared marker beside a result that the statistical checks alone
    decide (the decision names no record, and no record drives it), why the marker was not applied, as what follows "it was
    not applied to this result". With a fusion model the checks alone give a strong code, which a marker's floor does not
    change; on the fallback the score stands at or above the number that the marker would set. None in every other case."""
    if signal.get("provenance_override") != "self_declared_ai" or decision.get("provenance_record") or decision.get("driving_hypothesis") == H3:
        return None
    code = decision.get("code")
    if code not in AI_SIDE:
        return None
    if fallback:
        return f": the score of the checks is no lower than the {OVERRIDE_PROBABILITY['self_declared_ai'] * 100:.0f} out of 100 that the marker would set"
    strong = "strong " if code == "ai_supported_strong" else ""
    return f", because the statistical checks alone already give {strong}indications of AI generation or manipulation"


def _floor_clause(fallback: bool, beside: str = REPORTED, outranked: str | None = None) -> str:
    """What the floor of self-declared evidence does (D-15, D-68), as the clause after "…, so this is self-declared
    evidence": N-19n (D3-07): the 85 is the record's number and no estimate, so the clause says that Veritas sets the
    result's number at no less than 85 out of 100, on both fusion paths ("the estimate is floored at 85 and the statistical
    estimate is still reported" called it one). N-19h: and what stands beside it: the statistical checks, or that they
    gave no number, or that it is not shown (`_statistical_clause`). N-19l (R6-03, 2): a marker that was not applied says
    that, and which record outranks it; it speaks of no floor that the result does not have."""
    if outranked:  # which record outranks it, or (N-19n) that the checks alone decide: the whole tail, with its punctuation
        return f"it was not applied to this result{outranked if outranked[0] in ':,' else ': ' + outranked}"
    sets = f"Veritas sets the result's number at no less than {OVERRIDE_PROBABILITY['self_declared_ai'] * 100:.0f} out of 100 for it"
    if beside == REPORTED:
        return f"{sets} and reports the statistical checks beside it"
    return f"{sets}, and {_statistical_clause(beside, fallback, 'is still reported')}"


# --- entry point -----------------------------------------------------------------------------------------------

def synthesise(inp: SynthesisInput) -> list[dict[str, Any]]:
    decision = inp.decision
    fallback = decision.get("fusion_mode") != "artefact"
    code = decision.get("code")
    decision_direction = "for" if code in AI_SIDE else ("against" if code in AUTHENTIC_SIDE else None)
    descriptive, envelope = _conditions(inp.condition_profile or {})
    faces = _face_boxes(inp)
    work = [_signal_work(inp, s, descriptive, faces) for s in inp.signals]
    _corroborate(work)
    media = MEDIA.get(inp.kind, ("file", "files"))[0]
    measured = not fallback and any(r.get("kind") == "neutral_start" for r in decision.get("contributions") or [])
    beside = _beside(decision, inp.condition_profile)  # N-19h: what the findings say of the statistical number
    signals_by_id = {s["id"]: s for s in inp.signals}
    unshown = abstain.outside_validated(decision, inp.condition_profile)
    uncounted = {str(r["signal_id"]) for r in decision.get("contributions") or []
                 if r.get("kind") == "detector" and r.get("counted") is False and r.get("signal_id") and effect.entered(r.get("logit"))}
    for w in work:
        signal = signals_by_id.get(w.key) or {}
        outranked = _outranked(signal, decision)
        # N-19n (D3-07): a marker that the decision did not apply: a record outranks it, or the checks alone decide (Q, FF)
        unapplied = outranked or _checks_decide(signal, decision, fallback)
        w.title = _marker_title(w, signal, inp.kind, unapplied is not None)
        _in_estimate(w, measured, uncounted)
        w.unshown = unshown
        w.priority = st.priority(w.family, w.strength, w.direction, c2pa_invalid=w.c2pa_invalid, region_on_face=w.on_face,
                                 weak_pair=w.weak_pair)
        w.dissent = _dissent(w, decision_direction, decision.get("driving_hypothesis"), fallback, beside,
                             any(str(c).startswith(abstain.MODEL_UNUSED) for c in (inp.condition_profile or {}).get("limiting_conditions") or []),
                             outranked, (decision.get("provenance_record") or {}).get("kind"))
        w.view.update({"strength": w.strength, "direction": w.direction, "variant": w.variant, "media": media,
                       "in_estimate": w.in_estimate, "outside_estimate": w.outside_estimate, "not_counted": w.not_counted,
                       "within": "this result" if unshown else "this estimate",
                       "standalone": _standalone_sentence(w),
                       **_outside_view(w, inp.kind),
                       "phrase": _phrase(w.hypothesis, w.family), "lr_text": _lr_text(w.log_lr, w.uncalibrated),
                       "conditions": w.conditions, "benchmark": w.benchmark or "", "dissent": w.dissent,
                       "fp_modes": card_copy.benign_causes(w.fp_modes),
                       "floor_clause": _floor_clause(fallback, beside, unapplied), "unapplied": unapplied is not None,
                       "statistical_clause": _statistical_clause(beside, fallback, "is still reported")})
    extras = [x for x in (_absent_credentials(inp), _conditions_work(inp.condition_profile, descriptive, envelope)) if x]
    rows = not fallback and bool(decision.get("contributions"))
    record = frozenset((decision.get("provenance_record") or {}).get("signal_ids") or [])
    ordered = sorted([*work, *extras], key=lambda w: _rank_key(w, rows, record))
    ordered.append(_coverage_work(inp, work))
    ids = {w.key: f"F{rank}" for rank, w in enumerate(ordered, 1)}
    findings = []
    for rank, w in enumerate(ordered, 1):
        sentences = render(w.template, w=w.view)
        evidence = list(dict.fromkeys(i for s in sentences for i in s["evidence_ids"]))
        findings.append(Finding.model_validate({
            "id": ids[w.key], "family": w.family, "hypothesis": w.hypothesis, "lane": w.lane, "strength": w.strength,
            "direction": w.direction, "review_priority": w.priority, "rank": rank, "title": _title(w), "statement": " ".join(s["text"] for s in sentences),
            "sentences": sentences, "evidence_ids": evidence, "scope": w.scope, "probability": w.probability,
            "contribution_logit": w.contribution, "corroborated_by": [ids[k] for k in w.corroborated if k in ids],
            "reference_range": w.reference, "benchmark_note": w.benchmark, "fp_modes": w.fp_modes, "conditions": w.conditions,
            "dissent_note": w.dissent,
            # N-19q (finding 1): the stand-alone words of a check that entered, which D-76's sentence reads
            "standalone_strength": w.standalone_strength if w.in_estimate else None,
            "standalone_direction": w.standalone_direction if w.in_estimate else None,
        }).model_dump(mode="json"))
    return findings
