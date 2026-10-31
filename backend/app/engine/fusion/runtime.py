"""The published decision from a fuser artefact (ARCHITECTURE §4.4; ML §4–§5; PRD EA-01, EA-02, EA-06, API-03;
D-15, D-17, D-31, D-33).

`fuse_with_artefact`, for one analysis:
1. **Scores.** Each artefact detector's aggregate (a probability, or a logit / LLR) → z (Stage 1). A detector that did
   not run, or reported no probability-scale score, is missing.
2. **Stages 2–4** through `Fuser.apply` with the input's context: s, exact contributions, CI90, disagreement D.
3. **Provenance bound.** Artefact detectors of hypothesis H3 may move s by at most the configured bound (DC-06). A clamp
   is its own `override` contribution row, so the rows still add up to logit s.
4. **Band** from the conformal sets on the bounded s before any deployment prior shift; decision code; confidence
   (high for very_likely_*, medium for likely_*, low for inconclusive, D > d95 or a CI90 wider than 0.3).
5. **EA-02.** Outside the validated envelope (including below the smallest validated size, L-1) the code is capped at
   `inconclusive`. **D-46 / D-49:** an artefact detector
   that failed, timed out or was skipped (inside its gate or not: a failed face stage reads as no face), returned an unusable
   score (a probability, logit or LLR aggregate that is non-finite, or a probability outside [0, 1]), or has no entry while
   inside its gate adds nothing (no missing term), caps the code at `inconclusive`, sets confidence low and adds a caveat
   naming it. `not_applicable`, and an `ok` entry without an aggregate or with another raw kind, abstain: the learned
   missing term acts, as the score cache records them (`score_v2.detector_row`). **D-49:** while such a check did not run
   and some statistical detector scored, no estimate, interval, statistical estimate or contribution rows are published
   (the remaining detectors' probability stays in `fusion.estimate_without_not_run`), and H1 / H2 are not applicable.
6. **Stage 0** on top: a trusted C2PA AI declaration and a trusted capture manifest decide and are not capped; a
   self-declared marker floors s at 0.85 and never produces a strong code on its own. While a check did not run an override
   publishes exactly its probability with one override row and no statistical-estimate caveats; self-declared is then low
   confidence and its reason names the checks. An unsigned marker also preserves low confidence forced by statistical
   disagreement or a wide interval; it does not resolve either uncertainty.
7. **Which checks count (D-64; N-19f).** `n_entered`: the detector rows of the published view that are not zero, outside H3
   (`abstain.checks_entered`). The rule for a strong code, "k of n detectors" and the marker's branch read it.
8. **Not assessable**, when no override applies, in this order: no artefact detector outside H3 produced a usable score;
   then the cap of D-49 answers while a check did not run and a check scored, whether one that scored entered or not; then,
   where every check ran or was not applicable and none that scored entered, the result is not assessable because the
   model gives the checks that scored no weight for the input (`abstain.NO_WEIGHT_REASON`): no row and no per-signal
   contribution. A provenance record over a result in which no check entered decides alone, as while a check did not run.

Also: sub-fuser assessments (H1; H2 lanes), the H3 assessment (C2PA state), per-signal display calibration
(`calibrated_probability`, `log_lr`, ML §5.4) and calibration records. `fallback_profile` marks the condition profile of a
fallback analysis as outside every validated slice (no artefact, no slice metrics); D-31 keeps the fallback's own caps.
"""
from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from app.core import effect
from app.core.evidence import Signal
from app.engine.detector import ON_THIS_SERVER, UNNAMED_CHECK, Hypothesis, Lane, of_the_server
from app.engine.fusion import abstain, calibration_evidence, decisive, derived
from app.engine.fusion.calibration import clamp_probability, prior_logit
from app.engine.fusion.conformal import alpha_key, calibrated
from app.engine.fusion.conformal import band as conformal_band
from app.engine.fusion.contributions import NeutralView
from app.engine.fusion.disagreement import calibrated_logit
from app.engine.fusion.features import SCREENSHOT_LIKE_MIN, Context, transform_score
from app.engine.fusion.registry import Selection
from app.schemas.evidence import (
    Assessment,
    CalibrationInfo,
    Contribution,
    Decision,
    LaneAssessment,
    StatisticalAlone,
)
from app.schemas.legacy import BAND_HEADLINE

H1, H2, H3 = (h.value for h in Hypothesis)
BAND_TO_CODE = {"very_likely_ai": "ai_supported_strong", "likely_ai": "ai_supported", "uncertain": "inconclusive",
                "likely_authentic": "no_ai_indicators", "very_likely_authentic": "no_ai_indicators_strong"}
CI_WIDTH_LOW_CONFIDENCE = 0.3  # ML §5.2
DEFAULT_ALPHA_STRONG, DEFAULT_ALPHA_MODERATE = 0.01, 0.05  # ML §4.5 levels; a workspace policy may choose others (API-03)
POLICY_LEVELS_CAVEAT = ("The workspace policy sets conformal levels α = {strong:g} and {moderate:g}, which this fusion model is not "
                        "calibrated at, so the calibrated levels α = 0.01 and 0.05 were used.")
SUB_FUSERS = {"h1_synthesis": (H1, None), "h2_face_swap": (H2, Lane.FACE_SWAP.value),
              "h2_partial_edit": (H2, Lane.PARTIAL_EDIT.value), "voice": (H2, Lane.VOICE.value)}
C2PA_DETECTORS = decisive.CREDENTIAL_DETECTORS
DECLARED_AI_VERBAL = "The file's Content Credentials declare it was created with AI"  # UX §7.1 basis `documentary`
SELF_DECLARED_VERBAL = "The file's metadata names an AI generator"  # UX §4.3 (D-33)
PENDING_REVIEW_CAVEAT = "Fusion model pending human review"  # D-17, D-57
NOT_RUN_CAVEAT = "A check that this analysis depends on did not run: {name} ({reason})."  # D-46
_NOT_RUN_WORDS = {"failed": "failed", "timeout": "timed out", "skipped": "skipped"}
SCORE_KINDS = frozenset({"probability", "logit", "llr"})
UNUSABLE_SCORE = "returned an unusable score"  # D-49
NOT_RUN_SENTENCE = "a check that this analysis depends on did not run"  # D-49
# N-8: the stage that measures the input's conditions (size, compression, faces) is a check the decision depends on, whatever
# the gates of the artefact: without it every context term reads as absent ("no face", "unknown quality")
CONDITION_CHECK = "input conditions"
CONDITION_FAILED = "the condition stage failed"
NO_ESTIMATE_SENTENCE = "No estimate is published while a check did not run."
NOT_ASSESSABLE_NOTE = "no artefact detector produced a score"  # D-49: H1 / H2 of a not-assessable result
NOT_RUN_NOTE = "no estimate: " + NOT_RUN_SENTENCE + " ({names})"  # D-49: H1 / H2 while a check did not run
# N-19f: H1 / H2 where every check ran or was not applicable and none that scored has a row that is not zero
NO_WEIGHT_NOTE = "for an input like this one the fusion model gives the checks that scored no weight"
D49_NOTES = (NOT_ASSESSABLE_NOTE, NOT_RUN_NOTE.split(" (", 1)[0], NO_WEIGHT_NOTE)  # prefixes `forensic.scope` recognises
FACE_SCOPE_CAVEAT = ("{name} is a face-oriented check and no face was detected, so it scored the whole image. Its stand-alone "
                     "evidence covers AI-generated faces; here it acts only through the fused model's weights, which were "
                     "fitted on images with and without faces.")  # L-1
DISAGREEMENT_CAVEAT = "Detectors strongly disagree: the disagreement index is above the 95th percentile of the calibration set."
WIDE_INTERVAL_CAVEAT = "The 90 % interval of the estimate is wider than 30 points, so the estimate is imprecise."
# N-19h (E-20, 6): where a record set the estimate no interval is published; the width that lowers a marker's confidence
# (D-33) is the statistical estimate's, and the caveat says so
WIDE_STATISTICAL_INTERVAL_CAVEAT = ("The 90 % interval of the statistical estimate is wider than 30 points, so the statistical "
                                    "estimate is imprecise.")
INTERVAL_SCOPE_CAVEAT = ("The 90 % interval reflects fuser weight and calibration uncertainty only, not detector error or "
                         "distribution shift.")
FALLBACK_ENVELOPE = "no calibrated fusion model is loaded, so no validation slice applies (uncalibrated fallback)"
# N-19k (Q2-14): a fusion model is installed and this analysis did not use it (the quick profile runs fewer checks than the
# model needs): "no … is loaded" is said only where none is
MODEL_UNUSED = abstain.MODEL_UNUSED
FALLBACK_ENVELOPE_UNUSED = MODEL_UNUSED + ", so no validation slice applies (uncalibrated fallback)"
_SET_TEXT = {("ai",): "{AI}", ("real",): "{non-AI}", ("ai", "real"): "{AI, non-AI}", (): "∅"}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _pp(total: float, phi: float) -> float:
    return _sigmoid(total) - _sigmoid(total - phi)


def _signal_id(entry: Mapping[str, Any]) -> str:
    aggregate = entry.get("aggregate")
    return f"{entry['detector_id']}:{aggregate['unit_ref'] if aggregate else 'global'}"


def score_of(entry: Mapping[str, Any] | None) -> tuple[float | None, float | None] | None:
    """(probability, logit) of a detector's aggregate, or None when it has no usable probability-scale score."""
    if entry is None or entry.get("status") != "ok" or entry.get("aggregate") is None or entry["aggregate"].get("raw") is None:
        return None
    raw, kind = float(entry["aggregate"]["raw"]), entry["aggregate"]["raw_kind"]
    if kind == "probability" and 0.0 <= raw <= 1.0:
        return raw, None
    if kind in ("logit", "llr") and math.isfinite(raw):
        return None, raw
    return None


def unusable_score(entry: Mapping[str, Any]) -> bool:
    """D-49: an `ok` entry whose probability, logit or LLR aggregate is present but unusable (non-finite, or a probability
    outside [0, 1]). An `ok` entry without an aggregate, or with another raw kind, is not unusable: it abstains."""
    aggregate = entry.get("aggregate")
    return aggregate is not None and aggregate.get("raw_kind") in SCORE_KINDS and score_of(entry) is None


def not_run(detectors: Iterable[str], by_id: Mapping[str, Mapping[str, Any]],
            in_gate: Collection[str] | None = None) -> dict[str, str]:
    """D-46 / D-49: {detector id: reason} for the artefact detectors whose result the decision depends on and lacks: failed,
    timed out or skipped (whether or not the gate is open), an unusable score, or no entry at all while inside their context
    gate (`in_gate`; None = every detector). These add nothing and cap the decision at inconclusive. `not_applicable` and an
    `ok` entry without a probability-scale aggregate abstain: they use the learned missing term."""
    out: dict[str, str] = {}
    for detector_id in detectors:
        entry = by_id.get(detector_id)
        if entry is None:
            if in_gate is None or detector_id in in_gate:
                out[detector_id] = "did not run"
        elif entry.get("status") == "not_applicable":
            continue
        elif entry.get("status") != "ok":
            # N-14a: a check skipped for the server's memory budget or for model files that are not installed is the server's
            word = (ON_THIS_SERVER if of_the_server(entry.get("status"), entry.get("reason"))
                    else _NOT_RUN_WORDS.get(str(entry.get("status")), "did not run"))
            out[detector_id] = f"{word}: {entry['reason']}" if entry.get("reason") else word
        elif unusable_score(entry):
            out[detector_id] = UNUSABLE_SCORE
    return out


def context_from(condition: Mapping[str, Any] | None, kind: str) -> Context:
    c = (condition or {}).get("context") or {}
    screenshot = ((condition or {}).get("profile") or {}).get("screenshot_likelihood")
    return Context(modality=kind, face_px=c.get("face_px"), jpeg_q=c.get("jpeg_q"), min_side=c.get("min_side"),
                   has_camera_exif=bool(c.get("has_camera_exif")),
                   screenshot_like=screenshot is not None and screenshot >= SCREENSHOT_LIKE_MIN)


START_LABEL = "every check at its neutral point"  # D-63: the starting row of a result published since N-17


def _rows(view: NeutralView, by_id: Mapping[str, Mapping[str, Any]], outside: Collection[str] = ()) -> list[Contribution]:
    """D-63: the published rows. The start is the estimate with every scoring check at its neutral point, in this context; a
    scoring check's row is the change when it moves from that point to its score and names what it is measured from.
    N-19i (R5-01): the row says whether its check is one the decision counts (`abstain.row_counts`; `outside`: the checks
    that declare H3), so that every count of the page and the reports is the runtime's."""
    rows = [Contribution(label=START_LABEL, kind="neutral_start", logit=view.start, pp=0.0)]
    for row in view.rows:
        entry = by_id.get(row.detector) or {}
        name = entry.get("name") or row.detector
        if row.kind == "missing":
            rows.append(Contribution(label=f"missing: {name}", kind="missing", logit=row.logit, pp=0.0))
        elif row.z is not None:
            rows.append(Contribution(signal_id=_signal_id(entry) if entry else None, label=name, kind="detector",
                                     logit=row.logit, pp=0.0, baseline="neutral_point" if row.standalone else "p50",
                                     counted=abstain.row_counts(row, outside)))
    return rows


def _with_pp(rows: Iterable[Contribution], total: float) -> list[Contribution]:
    return [r.model_copy(update={"pp": _pp(total, r.logit)}) for r in rows]


def c2pa_state(by_id: Mapping[str, Mapping[str, Any]]) -> str | None:
    """H3 `c2pa_state` from the C2PA detector run: trusted, valid (untrusted signer), invalid or absent. A remote-only
    manifest that was not fetched (offline), or an unsupported container, has no state."""
    entry = next((by_id[d] for d in C2PA_DETECTORS if d in by_id), None)
    if entry is None:
        return None
    metrics = entry.get("metrics") or {}
    if entry["status"] == "not_applicable":
        # Unsupported containers were never inspected for a manifest.
        return None if metrics.get("unsupported_container") == 1.0 else "absent"
    if entry["status"] != "ok" or metrics.get("remote_reference"):
        return None
    return "trusted" if metrics.get("trusted") else ("valid" if metrics.get("valid") else "invalid")


def watermark_readings(by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """H3 `watermarks[]`: the messages the public watermark decoders read (I-5), present or not."""
    from app.provenance import watermarks  # noqa: PLC0415 - cv2 and pywt load only when a decoder ran

    return [entry for d in sorted(decisive.WATERMARK_DETECTORS) if d in by_id and by_id[d].get("status") == "ok"
            for entry in watermarks.assessment_entries(by_id[d].get("metrics") or {})]


def _assessments(selection: Selection, scores: Mapping[str, Any], ctx: Context, prior_ai: float | None,
                 by_id: Mapping[str, Mapping[str, Any]], override: str | None, *,
                 inactive: Collection[str] = (), outside: Collection[str] = ()) -> tuple[list[Assessment], str | None]:
    fuser = selection.fuser
    assert fuser is not None
    h1: Assessment | None = None
    lanes: list[LaneAssessment] = []
    for name in fuser.sub_fusers():
        hypothesis, lane = SUB_FUSERS.get(name, (None, None))
        if hypothesis is None:
            continue
        sub = fuser.apply_sub(name, scores, ctx, prior_ai=prior_ai, inactive=inactive)
        rows = _with_pp(_rows(fuser.neutral_view(scores, ctx, prior_ai=prior_ai, inactive=inactive, sub=name), by_id, outside),
                        sub.logit)
        if hypothesis == H1:
            h1 = Assessment(hypothesis=H1, applicable=True, probability=sub.probability_display, contributions=rows)
        else:
            lanes.append(LaneAssessment(lane=lane, probability=sub.probability_display, band=sub.band, contributions=rows))
    out = [h1 or Assessment(hypothesis=H1, applicable=False, note="this fuser artefact has no fitted H1 sub-fuser"),
           Assessment(hypothesis=H2, applicable=bool(lanes), probability=max((x.probability for x in lanes), default=None),
                      lanes=lanes, note=None if lanes else "this fuser artefact has no fitted H2 sub-fuser")]
    state = c2pa_state(by_id)
    h3_ok = [e for e in by_id.values() if e.get("status") == "ok" and H3 in (e.get("hypotheses") or [])]
    out.append(Assessment(hypothesis=H3, applicable=state is not None or bool(h3_ok), c2pa_state=state,
                          watermarks=watermark_readings(by_id), note=f"provenance override: {override}" if override else None))
    candidates = [(h1.probability, H1)] if h1 is not None and h1.probability is not None else []
    candidates += [(x.probability, H2) for x in lanes]
    driving = max(candidates)[1] if candidates else None
    return out, driving


def signal_fields(selection: Selection, by_id: Mapping[str, Mapping[str, Any]], scores: Mapping[str, Any],
                  view: NeutralView, total: float) -> dict[str, dict[str, Any]]:
    """Per-signal contribution (from the check's neutral point, D-63), effective weight and the detector's univariate Platt
    probability and log LR (ML §5.4)."""
    fuser = selection.fuser
    assert fuser is not None
    doc = fuser.doc
    bench = (doc.get("bench") or {}).get("version") or "unknown"
    rows = {r.detector: r for r in view.rows if r.kind == "detector" and r.z is not None}
    out: dict[str, dict[str, Any]] = {}
    for detector_id in fuser.detectors:
        entry, score, row = by_id.get(detector_id), scores.get(detector_id), rows.get(detector_id)
        if entry is None or score is None or row is None:
            continue
        z = transform_score(score[0], score[1], fuser.transforms[detector_id])
        z_tilde = calibrated_logit((doc.get("signal_calibration") or {}).get(detector_id), z)
        out[_signal_id(entry)] = {
            "contribution_logit": row.logit, "contribution_pp": _pp(total, row.logit), "effective_weight": row.effective_weight,
            "calibrated_probability": None if z_tilde is None else _sigmoid(z_tilde), "log_lr": z_tilde,
            "calibration_ref": f"{detector_id}@{entry.get('version')}/{bench}"}
    return out


def outside_reasons(selection: Selection, by_id: Mapping[str, Mapping[str, Any]], scores: Mapping[str, Any],
                    view: NeutralView, terms: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """N-19k (browser QA Q2-12): for each check of the model that scored and is not counted in the estimate, why, as the
    record of that check for this input (`SignalV2.outside_estimate`): its context gate is closed (with the gate's
    literals), or it stands in the sum with a weight of zero. A check that is not among the model's is named by `annotate`."""
    fuser = selection.fuser
    assert fuser is not None
    rows = {r.detector: r for r in view.rows if r.kind == "detector" and r.z is not None}
    gates = dict(fuser.space.gates)
    out: dict[str, dict[str, Any]] = {}
    for j, detector_id in enumerate(fuser.detectors):
        entry = by_id.get(detector_id)
        if entry is None or scores.get(detector_id) is None:
            continue
        if not fuser.space.gate_open(j, terms):
            out[_signal_id(entry)] = {"reason": "gate", "gate": list(gates.get(detector_id) or [])}
        elif detector_id in rows and not effect.entered(rows[detector_id].logit):
            out[_signal_id(entry)] = {"reason": "zero_weight", "gate": []}
    return out


_V1_DIRECTION = {"for": "supports_ai", "against": "supports_authentic", "neutral": "neutral"}


def annotate(signals: Sequence[Mapping[str, Any]], fields: Mapping[str, Mapping[str, Any]], *,
             artefact_detectors: Iterable[str], outside: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Apply artefact fields to signals. Other signals do not enter the calibrated estimate: effective weight 0.

    `outside` (N-19k; `outside_reasons`): where the result publishes an estimate of its checks, each check that scored and
    is not counted in it carries why: the model's own checks from `outside`, and a check that is not among the model's as
    that. None where the result publishes no such estimate: there no check is counted, and none is singled out."""
    detectors = set(artefact_detectors)
    out = []
    for s in signals:
        extra = fields.get(s["id"])
        why = None if outside is None else outside.get(s["id"])
        if extra is None:
            if s.get("raw_score") is not None and s.get("detector_id") not in detectors:
                not_in_model = None if outside is None else {"reason": "not_in_model", "gate": []}
                out.append({**s, "effective_weight": 0.0, "contribution_logit": None, "contribution_pp": None,
                            **({"outside_estimate": not_in_model} if not_in_model else {})})
            else:
                out.append({**s, "outside_estimate": dict(why)} if why and s.get("raw_score") is not None else dict(s))
            continue
        merged = {**s, **extra, **({"outside_estimate": dict(why)} if why else {})}
        p, c = extra.get("calibrated_probability"), extra.get("contribution_logit")
        if merged.get("legacy") is not None and effect.entered(c):
            # D-63: a check that entered the estimate is labelled by its contribution, in the v1 mirror too
            merged["legacy"] = {**merged["legacy"], "direction": _V1_DIRECTION[effect.direction(c)], "strength": effect.strength(c)}
        elif p is not None and merged.get("legacy") is not None:  # outside the estimate: its stand-alone words
            merged["legacy"] = {**merged["legacy"], "direction": Signal.direction_from_prob(p),
                                "strength": Signal.strength_from_prob(p)}
        out.append(merged)
    return out


def fallback_profile(condition: Mapping[str, Any] | None, kind: str, *, installed: bool = False) -> dict[str, Any]:
    profile = dict((condition or {}).get("profile") or {"slice_id": f"{kind}/unknown"})
    profile["within_validated_envelope"] = False
    profile["validation_support"] = "unsupported"
    profile["validation_measured_at"] = None
    profile["limiting_conditions"] = [*profile.get("limiting_conditions", []),
                                      FALLBACK_ENVELOPE_UNUSED if installed else FALLBACK_ENVELOPE]
    return profile


def _calibration(selection: Selection, by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    fuser = selection.fuser
    assert fuser is not None and selection.sha256 is not None
    doc = fuser.doc
    bench = doc.get("bench") or {}
    test = (doc.get("metrics") or {}).get("test") or {}

    def number(value: Any) -> float | None:
        return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None

    review = getattr(selection, "review", None)
    common = {"method": "platt", "artefact_sha256": selection.sha256, "bench_version": bench.get("version"),
              "bench_manifest_sha256": bench.get("manifest_sha256"), "reviewed_by": selection.reviewed_by,
              # N-19d: the date of the review record that names the artefact, so that a result can say "reviewed by … on …"
              "review_date": review.date if review is not None else None}
    records = [CalibrationInfo(subject=f"fuser-{doc['modality']}-{doc['version']}", ece=number(test.get("ece")),
                               brier=number(test.get("brier")), test_view="size_matched" if test.get("view") else None,
                               reliability_diagram=calibration_evidence.diagram_for_artefact(selection.sha256, doc), **common)]
    for detector_id, entry in sorted((doc.get("signal_calibration") or {}).items()):
        if entry is not None:
            version = (by_id.get(detector_id) or {}).get("version") or "unknown"
            records.append(CalibrationInfo(subject=f"{detector_id}@{version}", **common))
    return [r.model_dump(mode="json") for r in records]


def fuse_with_artefact(selection: Selection, entries: Sequence[Mapping[str, Any]], condition: Mapping[str, Any] | None, *,
                       kind: str, prior_ai: float | None = None, h3_bound: float = decisive.H3_INFLUENCE_BOUND,
                       policy: Mapping[str, Any] | None = None, condition_failed: bool = False,
                       check_names: Mapping[str, str] | None = None) -> dict[str, Any]:
    """`policy`: the workspace policy thresholds version in force (`app.policy.thresholds.Policy.as_json`). None keeps the
    calibrated defaults: α 0.01 for the strong codes, 0.05 for the moderate codes, slice AUC ≥ 0.70. `condition_failed`: the
    condition stage of an image or video did not complete, which is a check that did not run (D-49; N-8): no estimate.
    `check_names` (N-19k; D2-06): the registry's name of each check of the model, for a caveat or a reason about a check
    that has no run entry; such a check was named by its id."""
    fuser = selection.fuser
    if selection.mode != "artefact" or fuser is None or selection.sha256 is None:
        raise ValueError("fuse_with_artefact needs a loaded fuser artefact")
    doc = fuser.doc
    bench = doc.get("bench") or {}
    by_id = derived.with_derived({e["detector_id"]: e for e in entries})  # H-6: partial-edit features from localiser statistics
    scores = {d: score_of(by_id.get(d)) for d in fuser.detectors}
    ctx = context_from(condition, kind)
    terms = ctx.terms()
    checks_not_run = not_run(fuser.detectors, by_id, {d for j, d in enumerate(fuser.detectors) if fuser.space.gate_open(j, terms)})
    if condition_failed:
        checks_not_run = {CONDITION_CHECK: CONDITION_FAILED, **checks_not_run}
    out = fuser.apply(scores, ctx, prior_ai=prior_ai, inactive=checks_not_run)
    shift = prior_logit(prior_ai)
    caveats: list[str] = []
    if selection.pending_review:  # D-57: no review sidecar names this artefact's sha256
        caveats.append(PENDING_REVIEW_CAVEAT)
    if prior_ai is not None:
        caveats.append(abstain.prior_caveat(prior_ai))  # N-19n: its fixed words are a constant that the reports and the page read
    def name_of(detector_id: str) -> str:
        return (by_id.get(detector_id) or {}).get("name") or (check_names or {}).get(detector_id) or (
            detector_id if detector_id == CONDITION_CHECK else UNNAMED_CHECK)

    names = {d: name_of(d) for d in checks_not_run}
    caveats += [NOT_RUN_CAVEAT.format(name=names[d], reason=why) for d, why in checks_not_run.items()]

    # 3. bound on non-override provenance influence
    declared = {det["id"]: det.get("hypotheses") or [] for det in doc["detectors"]}
    h3 = {d for d in fuser.detectors if H3 in ((by_id.get(d) or {}).get("hypotheses") or declared.get(d, []))}
    h3_logit = math.fsum(r.logit for r in out.decomposition.rows if r.detector in h3)
    bounded = decisive.bound_logit(out.logit, h3_logit, h3_bound)
    # D-63: the published rows are the same sum measured from each check's neutral point; the bound above, the face caveat
    # below and the decision read the fuser's own decomposition
    neutral = fuser.neutral_view(scores, ctx, prior_ai=prior_ai, inactive=checks_not_run)
    rows = _rows(neutral, by_id, h3)
    if bounded != out.logit:
        rows.append(Contribution(label=f"bound on provenance influence (|Δs| ≤ {h3_bound:g})", kind="override",
                                 logit=bounded - out.logit, pp=0.0))
    s_stat = _sigmoid(bounded)
    # L-1: a face-oriented check admitted without a gate that scored a whole image with no face is inside its admission
    # (fitted on images with and without faces) but outside its stand-alone evidence: disclosed, not capped
    caveats += [FACE_SCOPE_CAVEAT.format(name=name_of(d)) for d in fuser.detectors
                if (by_id.get(d) or {}).get("family") == "learned-face"
                and (((by_id[d].get("aggregate") or {}).get("quality") or {}).get("no_face"))
                and any(r.detector == d and r.logit != 0 for r in out.decomposition.rows)]

    # 4.–5. band, code, confidence, envelope, at the workspace policy levels (API-03)
    alpha_strong, alpha_moderate, min_auc = DEFAULT_ALPHA_STRONG, DEFAULT_ALPHA_MODERATE, abstain.MIN_SLICE_AUC
    if policy is not None:
        wanted = (float(policy["alpha_strong"]), float(policy["alpha_moderate"]))
        if all(calibrated(doc["conformal"], a) for a in wanted):
            alpha_strong, alpha_moderate = wanted
        else:
            caveats.append(POLICY_LEVELS_CAVEAT.format(strong=wanted[0], moderate=wanted[1]))
        min_auc = float(policy["min_slice_auc"])
    band_stat, sets = conformal_band(_sigmoid(bounded - shift), doc["conformal"], out.context, alpha_strong=alpha_strong,
                                     alpha_moderate=alpha_moderate)
    raw_ci = out.ci90 if bounded == out.logit else None
    # N-7: published clamped like the estimate, so a range never lies outside what the estimate can print ("2, range 1 to
    # 1"); the width that lowers the confidence is still the measured one, so no decision changes
    ci = None if raw_ci is None else (clamp_probability(raw_ci[0]), clamp_probability(raw_ci[1]))
    # N-18 (D-64): a strong code needs two checks that count; with one, the band and the code are those of the same side
    # one step weaker (the shared rule of `abstain`, which `bench/report.py` counts). The conformal sets are unchanged.
    # N-19f (R4-05): a check counts when its published row is not zero (`n_entered`), which is what the number line counts;
    # `n_scored`, the usable scores, still says whether any check scored at all and picks the note of H1 and H2
    statistical = [d for d in fuser.detectors if d not in h3]
    n_scored = sum(scores[d] is not None for d in statistical)
    n_entered = abstain.checks_entered(neutral.rows, h3)
    code = abstain.strong_needs_two(BAND_TO_CODE[band_stat], n_entered)
    one_check = code != BAND_TO_CODE[band_stat]
    band_policy = abstain.WEAKER_BAND[band_stat] if one_check else band_stat
    confidence = "high" if band_policy.startswith("very_likely") else ("medium" if band_policy.startswith("likely") else "low")
    d95 = (doc.get("disagreement") or {}).get("d95")
    strong_disagreement = out.disagreement is not None and d95 is not None and out.disagreement > d95
    if strong_disagreement:
        confidence = "low"
        caveats.append(DISAGREEMENT_CAVEAT)
    interval_computed, wide = raw_ci is not None, raw_ci is not None and raw_ci[1] - raw_ci[0] > CI_WIDTH_LOW_CONFIDENCE
    if wide:
        confidence = "low"
    interval_caveats_at = len(caveats)  # the interval caveats go here unless a check did not run (D-49), see below
    profile = dict((condition or {}).get("profile") or {"slice_id": f"{kind}/unknown", "limiting_conditions": []})
    capped, env = abstain.cap_for_slice(code, profile["slice_id"],
                                       ((doc.get("metrics") or {}).get("test") or {}).get("slices"),
                                       bench_version=bench.get("version"), min_auc=min_auc,
                                       below_floor=abstain.outside_reason(kind, profile, (condition or {}).get("context"), bench))
    profile["within_validated_envelope"] = env.within
    profile["validation_support"] = env.support
    profile["validation_measured_at"] = env.measured_at
    if env.within and env.support == "pooled":
        caveats.append(f"Validation uses the broader group {env.measured_at}; performance for this exact input slice "
                       "has not been established separately.")
    if not env.within:
        profile["limiting_conditions"] = [*profile.get("limiting_conditions", []), env.reason]
    set_text = _SET_TEXT[tuple(sets[alpha_key(alpha_moderate)])]
    # N-19m (D-77): the number of the reason is the estimate as published, in the whole points that the rule reads and a
    # result prints (it was `{:.0%}` of the value held here, a half to the even neighbour: "estimate 34%" over a printed 35)
    points = abstain.published_points(clamp_probability(s_stat))
    reason = (f"Calibrated fusion of {n_entered} of {len(statistical)} detectors (fuser {doc['modality']} {doc['version']}): "
              f"estimate {points}%, conformal set at α = {alpha_moderate:g} {set_text}.")
    if one_check:
        reason += f" {abstain.ONE_CHECK_SENTENCE}"
    band_final, code_final = band_policy, capped
    # N-19m, third part (D-77; D-76 as amended): the two conditions of a decisive code, D-77 first: the estimate as
    # published (the statistical checks' own, after a configured prior, clamped, six decimals) in whole points stands in
    # the part of the scale of the code's answer; then D-76: one check supports the code, and the checks together do. The
    # code is taken from the calibrated levels before the prior's shift, as before, and the bound is read on the number
    # that the reader sees after it. None where no check entered: such a result is not assessable
    rule = abstain.rule_for(code, clamp_probability(s_stat), neutral.rows, h3) if n_entered > 0 else None
    if capped != code:
        band_final, confidence = "uncertain", "low"
        conditions = [c for c in profile.get("limiting_conditions", []) if c != env.reason]
        # the reason's last sentence quotes the headline of the levels only where both conditions would have left that code
        # (it quoted a headline that a condition would not have given: code review, the read of the third part)
        reason = (f"Capped at inconclusive: {env.reason}"
                  + (f"; conditions: {', '.join(conditions)}" if conditions else "") + ". "
                  f"Before the cap the calibrated estimate was {points}%"
                  + (f" (\"{BAND_HEADLINE[band_policy]}\")" if rule is None else "") + ".")
        caveats.append(f"Not validated for inputs like this: {env.reason}.")
    # N-19i (D-76): a decisive code needs a check that supports it: a row of at least ln 2 toward its side, of a check that
    # counts. The rule changes a result that the statistical checks would decide inside the envelope, and nothing else: it
    # is asked after the envelope's cap, which keeps its reason and its caveat (asked before, it would leave that cap
    # nothing to change); a result with a check that did not run keeps the cap of D-49, a result without a check that
    # entered stays not assessable, and a provenance record decides below whatever the checks support
    # N-19m: `taken`, the condition that took the code; `unsupported`, D-76 in either form (no single check, or the checks
    # together): not D-77, whose result gives its own reason and nothing of D-76
    taken = rule if capped == code and not checks_not_run else None
    unsupported = taken not in (None, abstain.RULE_SCALE)
    if taken is not None:
        code_final, band_final, confidence = "inconclusive", "uncertain", "low"
        # N-19n: D-77's ending names the part in which the published estimate stands (the middle one, or under a prior the other answer's)
        reason += f" {abstain.rule_reason(code, taken, clamp_probability(s_stat))}"
    checks = "; ".join(f"{names[d]} ({why})" for d, why in checks_not_run.items())
    if checks_not_run:  # D-46 / D-49: a check that did not run never moves the verdict toward either class; no estimate is published
        reason = f"Capped at inconclusive: {NOT_RUN_SENTENCE}: {checks}. {NO_ESTIMATE_SENTENCE}"
        code_final, band_final, confidence = "inconclusive", "uncertain", "low"
    assessments, driving = _assessments(selection, scores, ctx, prior_ai, by_id, None, inactive=checks_not_run, outside=h3)
    verbal = BAND_HEADLINE[band_final]
    driving_final = driving if code_final not in ("inconclusive", "no_ai_indicators", "no_ai_indicators_strong") else None
    # N-19l (R6-05; D-68): what the statistical checks alone decided, which a record that decides below does not change
    # N-19m: it names the condition as the stage does, with the side of D-76 where D-76 took the code and none where D-77 did
    alone = StatisticalAlone(code=code_final, rule=taken, unsupported=(
        ("for" if abstain.SIDE_OF_CODE[code] > 0 else "against") if unsupported else None))

    # 6. Stage 0 overrides
    override_entries = [e for e in entries if e.get("status") == "ok" and e.get("provenance_override")]
    watermark = decisive.watermark_only(override_entries)
    marker = decisive.marker_kind(override_entries)  # N-19h (E-21): a marker has the words of its kind, in this kind of file
    outcome = decisive.apply(s_stat, [e["provenance_override"] for e in override_entries], watermark=watermark, marker=marker,
                             kind_of_file=kind)
    # D-49: while a check did not run an override publishes exactly its probability, without the statistical estimate.
    # N-19f: so does a record over a result in which no check entered (none scored at all, or none that scored has weight):
    # the model's start is not a statistical estimate to print beside the record
    not_run_override = (bool(checks_not_run) or n_entered == 0) and outcome.kind is not None
    override_caveats = [c for c in outcome.caveats if not (not_run_override and c.startswith(decisive.STATISTICAL_CAVEATS))]
    # N-19h (E-20, 8): outside what was validated the statistical estimate is printed beside no record: the caveat says that
    # it is not shown, and why, as a result without a record does; the field keeps it
    caveats += override_caveats if env.within else decisive.withhold_statistical(override_caveats)
    s_final: float | None = decisive.OVERRIDE_PROBABILITY[outcome.kind] if not_run_override else outcome.probability
    if outcome.kind == "decisive_ai":
        band_final, code_final, confidence, driving_final, verbal = "very_likely_ai", "ai_supported_strong", "high", H3, DECLARED_AI_VERBAL
        reason = ("A C2PA manifest that validates against the trust list declares generative AI. This documentary evidence "
                  "decides the result" + ("." if not_run_override else "; the statistical estimate is kept alongside it."))
    elif outcome.kind == "trusted_capture":
        band_final, code_final, confidence, driving_final = "very_likely_authentic", "no_ai_indicators_strong", "high", H3
        verbal = decisive.CAPTURE_VERBAL  # N-19m (D3-01; D-15): the record's own headline; the code and the band stay
        reason = ("A trusted C2PA capture manifest covers the current bytes, so the signed capture state caps the estimate at "
                  f"{decisive.OVERRIDE_PROBABILITY['trusted_capture']:.0%}.")
    elif outcome.kind == "self_declared_ai":
        # D-33: an unsigned marker alone cannot inherit a strong code from
        # context priors or learned missingness when no statistical check scored.
        if code_final != "ai_supported_strong" or n_entered == 0:
            band_final, code_final, driving_final = "likely_ai", "ai_supported", H3
        # Unsigned metadata does not resolve statistical uncertainty (ML 5.2/5.4).
        # Signed documentary overrides above retain their separate D-15 precedence.
        # N-19f: a marker that decides alone (no check entered) has confidence low, as while a check did not run
        # N-19m (R7-03; D-76 as amended on 2026-10-05): and low where D-76 took the decisive code of the statistical estimate
        # (it was medium beside "no check gave at least weak support", low for the same input without the marker); the third
        # part: whichever condition took the statistical code, D-77 as D-76
        confidence = "low" if checks_not_run or n_entered == 0 or strong_disagreement or wide or taken is not None else (
            "high" if code_final == "ai_supported_strong" else "medium")
        floor = f"{decisive.OVERRIDE_PROBABILITY['self_declared_ai']:.0%}"
        # N-19n (D3-07; Q): a marker beside a strong code of the statistical checks changes nothing, and the decision names
        # no record below. Its headline and its reason are then the checks' too: they read "The file's metadata names an
        # AI generator" and "… floors the estimate at 85%; …" over a result that the page heads by its strong indications,
        # and the API and the v1 summary read them
        if driving_final == H3:
            verbal = decisive.marker_verbal(marker, kind)
            reason = decisive.floor_reason(marker, kind, floor)
        if checks_not_run:
            reason += f" A check that this analysis depends on did not run: {checks}."
        elif n_entered == 0:  # N-19h (E-19): on N-19f's path the reason ends with the sentence of that case
            reason += " " + (abstain.NO_WEIGHT_REASON if n_scored else abstain.not_assessable_reason(
                [e for e in entries if H3 not in (e.get("hypotheses") or []) and e.get("hypotheses")]))
    elif n_scored == 0:
        s_final, band_final, code_final, confidence, driving_final = None, "uncertain", "not_assessable", "low", None
        verbal = abstain.NOT_ASSESSABLE_VERBAL
        reason = abstain.not_assessable_reason([e for e in entries if H3 not in (e.get("hypotheses") or [])
                                                and e.get("hypotheses")])
        ci = None
    elif checks_not_run:  # D-49: capped, no estimate, whether a check that scored entered or not
        s_final, ci = None, None
    elif n_entered == 0:  # N-19f: every check ran or was not applicable, checks scored, and the model gives them no weight
        s_final, band_final, code_final, confidence, driving_final = None, "uncertain", "not_assessable", "low", None
        verbal, reason, ci = abstain.NOT_ASSESSABLE_VERBAL, abstain.NO_WEIGHT_REASON, None
    override_signal = next((_signal_id(e) for e in override_entries if e["provenance_override"] == outcome.kind), None)
    # N-19l (R6-04, 1): the row of the record names the kind of the record that was applied
    override_label = (decisive.override_label(outcome.kind, decisive.marker_kind(
        [e for e in override_entries if e["provenance_override"] == outcome.kind])) if outcome.kind is not None else "")
    if not_run_override:  # one override row, so the rows still add up to logit s
        rows = [Contribution(signal_id=override_signal, label=override_label, kind="override",
                             logit=_logit(s_final), pp=0.0)]
        ci = None
    elif outcome.kind is not None and s_final is not None and s_final != s_stat:
        rows.append(Contribution(signal_id=override_signal, label=override_label, kind="override",
                                 logit=_logit(s_final) - bounded, pp=0.0))
        ci = None
    if interval_computed and not checks_not_run and n_entered:  # every check ran and one entered: the interval caveats
        # N-19h (E-20, 6): what the interval reflects is said only with an interval. Where a record set the estimate none is
        # published, and a wide one is said of the statistical estimate, which stands beside the record's value
        # (8): the caveat on the statistical estimate is published only where that estimate is: not outside what was validated
        interval_caveats = ([INTERVAL_SCOPE_CAVEAT, *([WIDE_INTERVAL_CAVEAT] if wide else [])] if ci is not None
                            else [WIDE_STATISTICAL_INTERVAL_CAVEAT] if wide and env.within else [])
        caveats[interval_caveats_at:interval_caveats_at] = interval_caveats
    if checks_not_run or n_entered == 0:  # D-49, N-19f: H1 and H2 publish no estimate either
        # N-19l (R6-05, 2): under a record, a check that did not run is the reason also where no other check scored (VE,
        # VF: the note said that no check produced a score, beside a reason and a caveat that name the check that failed)
        not_run_note = bool(checks_not_run) and (n_scored > 0 or outcome.kind is not None)
        note = (NOT_RUN_NOTE.format(names=", ".join(names[d] for d in checks_not_run)) if not_run_note
                else NOT_ASSESSABLE_NOTE if n_scored == 0 else NO_WEIGHT_NOTE)
        assessments[0] = Assessment(hypothesis=H1, applicable=False, note=note)
        assessments[1] = Assessment(hypothesis=H2, applicable=False, note=note)
    # N-19f (code review R4-01, P1): the total is the log-odds this function holds, never the logit of a probability it made
    # from them: from 36.74 log-odds the sigmoid is exactly 1.0, and the logit of it divided by zero (the packaged image fuser
    # reaches 36.755). Where an override changed the estimate the total is the override's own log-odds, which the override
    # rows above were measured to; otherwise it is the bounded log-odds, which the rows of the checks add up to.
    overridden = s_final is not None and (not_run_override or (outcome.kind is not None and s_final != s_stat))
    total = _logit(s_final) if overridden else bounded  # type: ignore[arg-type]
    # N-19h (E-21): the record the decision applied, where H3 drives (a marker beside a strong statistical result does not)
    applied = [e for e in override_entries if e["provenance_override"] == outcome.kind]
    deciding = decisive.record(outcome.kind, applied, [_signal_id(e) for e in applied]) if driving_final == H3 else None
    decision = Decision(
        code=code_final, verbal=verbal, probability_ai_calibrated=None if s_final is None else clamp_probability(s_final),
        ci90=ci, band=band_final, confidence=confidence, driving_hypothesis=driving_final, provenance_record=deciding,
        statistical_alone=alone if deciding is not None and not checks_not_run and n_entered else None, reason=reason,
        statistical_estimate_before_provenance=None if checks_not_run or n_entered == 0 else clamp_probability(s_stat),
        disagreement=out.disagreement, fusion_mode="artefact", fuser_sha256=selection.sha256,
        contributions=_with_pp(rows, total) if s_final is not None else [])
    assessments[2] = assessments[2].model_copy(update={"note": f"provenance override: {outcome.kind}" if outcome.kind else None})
    fields = signal_fields(selection, by_id, scores, neutral, total)
    outside: dict[str, dict[str, Any]] | None = outside_reasons(selection, by_id, scores, neutral, terms)
    if checks_not_run or n_entered == 0:  # D-49, N-19f: no additive contributions are published without an estimate of the checks
        fields = {sid: {**f, "contribution_logit": None, "contribution_pp": None} for sid, f in fields.items()}
        outside = None  # N-19k: nor is a check said to be outside an estimate that the result does not publish
    return {
        "decision": decision.model_dump(mode="json"),
        "assessments": [a.model_dump(mode="json") for a in assessments],
        "signal_fields": fields,
        "signal_outside": outside,
        "caveats": caveats,
        "calibration": _calibration(selection, by_id),
        "condition_profile": profile,
        "fusion": {**selection.as_json(), "context": out.context, "sets": sets, "statistical_band": band_stat,
                   "one_check": one_check,  # D-64: the strong code of `statistical_band` was not issued
                   # D-76: the decisive code of the estimate was not issued: no check supports it, or (N-19m) the checks
                   # together do not; false where D-77 took the code. `rule` names the condition that took it (`abstain.RULE_*`:
                   # the scale; no check; the checks together, less; the checks together, the other way)
                   "unsupported": unsupported, "rule": taken,
                   "h3_detectors": sorted(h3), "logit_before_bound": out.logit, "logit_bounded": bounded,
                   "not_run": [{"detector_id": d, "reason": why} for d, why in sorted(checks_not_run.items())],
                   # D-49: the remaining detectors' probability, kept only here while a check did not run
                   "estimate_without_not_run": clamp_probability(s_stat) if checks_not_run and n_entered else None},
    }
