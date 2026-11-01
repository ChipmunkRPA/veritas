"""View model of the standard and forensic report profiles (UX §6.3 R1–R12, §7; ARCHITECTURE §6.4; PRD RP-01 (1)–(6), RP-02,
RP-04, RP-06; D-10, D-25, D-37; BUILD_SPEC R-3).

Pure functions of a `ReportSnapshot` and the plates the engine decoded from the store. Headline, guidance, natural frequencies,
key findings and next step come from `view.py` (the answer module's wording); the estimate breakdown and the methods rows follow
`frontend/src/features/analysis/contributions.ts` and `methods.ts`, so paper says what the screen says (P8).

- **Sections.** Cover · Contents · 1 Summary and scope · 2 Key findings · 3 How the estimate was reached · 4 Exhibits ·
  5 Temporal analysis (video and audio) · 6 Provenance and metadata · 7 Detector results and error rates · 8 Methods and
  validation · 9 Limitations (never suppressed) · 10 Glossary · Appendix A Reproducibility · B Custody · C Notes and
  dispositions · D Sign-off · E Revision history.
- **RP-01 layers.** (1) lay summary: section 1; (2) findings and exhibits: sections 2, 4, 5, 6; (3) technical appendix: sections
  3, 7 and appendix A; (4) methods and validation: section 8; (5) integrity and custody: the cover hashes and appendix B;
  (6) examiner: appendices C and D.
- **Profiles** (D-10). `forensic` prints the declaration in the sign-off; `standard` is the same document without it.
- **Numbers.** Only snapshot values: the result document, frozen model-card extracts, custody events and review records. An
  absent value reads "not established" or "not recorded", never another slice's number.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any

from pydantic import ValidationError

from app.audit import chain as audit
from app.core import coords, effect
from app.core.canonical import canonical_hash
from app.core.evidence import band_for
from app.engine import card_copy
from app.engine.detector import ON_THIS_SERVER, UNNAMED_CHECK, of_the_server
from app.engine.fusion import disclosures
from app.engine.fusion import registry as fusion_registry
from app.engine.fusion.runtime import D49_NOTES
from app.engine.synthesis import strength as lr
from app.reports import figures, formatting, lint, prose, view
from app.reports.snapshot import ReportSnapshot, snapshot_hash
from app.schemas.evidence import ReliabilityDiagram
from app.schemas.legacy import V1_NOTE

NOT_ESTABLISHED = "not established"
NOT_RECORDED = "not recorded"
H1, H2, H3 = "H1_whole_content_synthesis", "H2_localised_manipulation", "H3_provenance_integrity"
PROB_FLOOR, PROB_CEILING = 0.02, 0.98  # app/engine/fusion/calibration.py
TOLERANCE, HALF_DP = 1e-6, 5e-7
DOMAIN = math.log(99)
PROB_TICKS = (1, 5, 20, 50, 80, 95, 99)
WATERFALL_TOP = 6
PLATE_BOX = (1004, 752)  # 85 × 64 mm at 300 dpi
STILL_BOX = (480, 360)
WIDE_BOX = (1600, 256)  # a picture over the time axis (the spectrogram of a recording, L-7): full text width, its own height
TABLE_ROWS = 60
DECLARATION_ID = "veritas.declaration.v2"
DECLARATIONS = {
    # Preserve the exact wording selected by historical conclusions; new defaults use v2.
    "veritas.declaration.v1": "I declare that I directed and reviewed the examination recorded in this report. The opinion stated is my own "
                              "and rests on the evidence, error rates and limitations recorded here. The automated estimates are probabilistic "
                              "evidence with known error rates, not a finding of fact.",
    DECLARATION_ID: "I declare that I directed and reviewed the examination recorded in this report. The opinion stated is my own "
                    "and rests on the evidence, error rates and limitations recorded here. The automated estimates are probabilistic "
                    "evidence subject to uncertainty and error, not a finding of fact.",
}
COVER_TITLE = {"image": "Examination of a digital image for indications of AI generation or manipulation",
               "video": "Examination of a video for indications of AI generation or manipulation",
               "audio": "Examination of an audio recording for indications of AI generation or manipulation",
               "document": "Examination of a document for indications of AI-written text"}
ARTICLE = {"image": "an image", "video": "a video", "audio": "a recording", "document": "a document"}
RUN_STATUS = {"not_applicable": "not applicable", "skipped": "skipped", "failed": "failed", "timeout": "timed out"}
EXAMINER_STATUS = view.EXAMINER_STATUS
SIGNOFF_KIND = {"examiner": "Examiner sign-off", "peer_review": "Peer review"}
SIGNOFF_DECISION = view.SIGNOFF_DECISION
REVIEW_STATE = {"machine": "machine assessment, not reviewed", "in_review": "in review", "reviewed": "reviewed by an examiner",
                "approved": "reviewed and peer reviewed", "disputed": "signed by an examiner; the peer reviewer does not concur",
                "withdrawn": "withdrawn"}
REVIEWED_WITH_CHANGES = "reviewed; the peer reviewer concurred with changes."  # N-19j (Q2-08)
# N-19j (design re-review D2-01, P1): each signer has a statement of his own with his own signature line. The forensic
# report printed one declaration ("I declare that I directed and reviewed the examination … The opinion stated is my
# own") over two signature lines: the reviewer did not direct the examination and stated no opinion of his own there
REVIEWER_STATEMENT_TITLE = "Statement of the peer reviewer"
REVIEWER_STATEMENT = {
    "concur": "I reviewed this report and the evidence it cites. I concur with the examiner’s opinion.",
    "concur_with_changes": "I reviewed this report and the evidence it cites. I concur with the examiner’s opinion, with the changes stated in my comment.",
    "non_concur": "I reviewed this report and the evidence it cites. I do not concur with the examiner’s opinion, for the reason stated in my comment."}
AUTH_WORD = {"session": "browser session", "api_key": "API key"}  # D-61: how the signer authenticated, from the custody event
SECOND_REVIEW_OUTSTANDING = "signed by an examiner; second review outstanding"  # D-56: reviewed, on a case that requires a second review
SEGMENT_LANE_ORDER = {lane: i for i, lane in enumerate(("face_swap", "partial_edit", "lip_sync", "voice", "synthesis"))}  # model.ts


def _letter(n: int) -> str:
    """0 → a, 25 → z, 26 → aa (the viewer's `letterFor`)."""
    out, k = "", n + 1
    while k > 0:
        out, k = chr(97 + (k - 1) % 26) + out, (k - 1) // 26
    return out


# N-19d (design review P2, as the page's `LANE_NAMES`): a lane is named as the check that drew it, not as a result
# N-19q (finding 10): the level of a lane's own estimate in the words of the scale, never "likely authentic"
LANE_BAND_WORDS = {"very_likely_ai": "strong indications", "likely_ai": "indications", "uncertain": "inconclusive",
                   "likely_authentic": "no significant indications", "very_likely_authentic": "no indications"}
LANES = (("face_swap", "Face-swap check"), ("lip_sync", "Lip-sync check"), ("partial_edit", "Partial-edit check"),
         ("voice", "Synthetic-speech check"), ("synthesis", "Synthesis check"), ("other", "Other checks"))
HEATMAP_TITLE = {"p_manipulated": "Anomaly map", "residual": "Noise residual map", "attribution": "Attention map",
                 "error_level": "Error level map"}
LOCALISATION_LEGEND = "Where pixel statistics differ from the rest of the image. Marked regions are candidates for editing, not confirmed edits."
LEGEND = {"p_manipulated": LOCALISATION_LEGEND, "residual": LOCALISATION_LEGEND,
          "attribution": "Where a classifier looked when scoring. Not a map of manipulation.",
          "error_level": "How much each area changes when re-saved. Bright edges and textures are normal; compare similar surfaces."}
C2PA_WORDS = {
    "absent": "No Content Credentials found. Most files carry none; this is not evidence either way.",
    "trusted": "Credentials valid · signer on the trust list.",
    "valid": "Credentials intact, but the signer isn’t on the trust list, so its identity is unverified.",
    "well-formed": "Credentials are present and well-formed, but they could not be validated, so what they state is unverified.",
    "invalid": "Credentials invalid — the file or its record changed after signing. Disregard what the record says.",
}
PROVENANCE_FAMILIES = view.RECORD_FAMILIES
LIPSYNC_LIMITATION = ("Lip-sync manipulation is checked only for audio–video misalignment; well-synchronised lip-sync deepfakes "
                      "are not detected.")  # D-37
QUESTIONS = {
    H1: {"image": "Was the whole image generated by AI?", "video": "Was the whole video generated by AI?",
         "audio": "Was the whole recording synthesised?", "document": "Was the text written by AI?"},
    H2: {"image": "Was part of the image edited, for example by inpainting, splicing or a face swap?",
         "video": "Were faces, lip movements or the voice in the video manipulated?", "audio": "Was the voice cloned or altered?",
         "document": "Were parts of the text written by AI?"},
    H3: {"image": "Does the file declare how it was made, in Content Credentials or metadata?",
         "video": "Does the file declare how it was made, in Content Credentials or metadata?",
         "audio": "Does the file declare how it was made, in its metadata?",
         "document": "Does the file declare how it was made, in its metadata?"},
}
NOT_ADDRESSED = ("Who made the file, or who directed its content.", "Whether the events, statements or scenes shown happened.",
                 "Which generator or editing tool was used, beyond what the file itself declares.")
STAGES = {
    "image": ("Intake and hashing: SHA-256 and MD5 of the original, which is stored write-once.",
              "Provenance and metadata: Content Credentials, watermarks, EXIF and XMP, JPEG structure.",
              "Condition profile: resolution, compression, faces and screenshot likelihood.",
              "Faces: detection, then face checks on each face.",
              "Localisation: anomaly maps and region segmentation.",
              "Whole-image classifiers: learned detectors of AI generation.",
              "Classical filters: noise, error level and copy-move checks.",
              "Fusion and calibration: the checks combined into one estimate.",
              "Findings: fixed sentence templates filled with the checks’ values.",
              "Report: rendered from a frozen snapshot of the result and the review records."),
    "video": ("Intake and hashing: SHA-256 and MD5 of the original, which is stored write-once.",
              "Probe and decoding: stream facts and a frame table by presentation time.",
              "Provenance and metadata: Content Credentials and container metadata.",
              "Shots and face tracks.",
              "Face-track checks: face swap per track.",
              "Lip-sync: mouth movement against the speech envelope.",
              "Voice windows: synthetic-speech checks on the sound track.",
              "Whole-frame synthesis checks.",
              "Codec and container checks.",
              "Temporal smoothing of window scores into segments.",
              "Fusion and calibration: the checks combined into one estimate.",
              "Findings: fixed sentence templates filled with the checks’ values.",
              "Report: rendered from a frozen snapshot of the result and the review records."),
    "audio": ("Intake and hashing: SHA-256 and MD5 of the original, which is stored write-once.",
              "Decoding to 16 kHz mono.",
              "Content gate: speech, music, noise and silence per second.",
              "Voice windows: synthetic-speech checks on speech windows.",
              "Temporal smoothing of window scores into segments.",
              "Fusion: the checks combined into one estimate.",
              "Findings: fixed sentence templates filled with the checks’ values.",
              "Report: rendered from a frozen snapshot of the result and the review records."),
    "document": ("Intake and hashing: SHA-256 and MD5 of the original, which is stored write-once.",
                 "Text extraction and sentence segmentation.",
                 "AI-text classifiers on sentence windows and document chunks.",
                 "Fusion: the checks combined into one estimate.",
                 "Findings: fixed sentence templates filled with the checks’ values.",
                 "Report: rendered from a frozen snapshot of the result and the review records."),
}
GENERAL_LIMITATIONS = {
    "all": ("Estimates are probabilistic. Every error rate applies to the test population named beside it; where AI content is "
            "rarer, the same estimate means a lower chance.",
            "Detectors generalise imperfectly to generators and editing tools released after their training data.",
            "Metadata and Content Credentials can be stripped or forged. Their absence is never treated as evidence either way.",
            "Automated checks do not identify who made a file, or whether what it shows happened.",
            "A qualified examiner should review any consequential conclusion."),
    "image": ("Heavy recompression, screenshots, resizing, cropping and social-media re-encoding weaken pixel-level checks.",
              "Face checks need a detectable, reasonably frontal face of at least 48 pixels.",
              "Classical filters (noise, error level, frequency) have known false alarms on HDR processing, heavy denoising and flat "
              "backgrounds, and their influence is capped."),
    "video": ("Frames are sampled, so very short or very long clips give fewer observations per second.",
              LIPSYNC_LIMITATION,
              "Voice-over, dubbing, occluded mouths and variable frame rates produce readings similar to desynchronised lip-sync.",
              "Fully generated video relies partly on whole-frame detectors trained on images."),
    "audio": ("Music, heavy compression, background noise and short speech reduce the reliability of voice checks.",
              "Recordings with less than 1 second of speech, or more music than speech, are not assessed."),
    "document": ("English only. Other languages, code, tables and lists reduce reliability.",
                 "Formulaic or templated human writing, non-native English, famous passages and grammar-tool editing can be flagged.",
                 "Paraphrased or lightly edited machine text, and models newer than the classifiers’ training data, can be missed."),
}
GENERAL_NOT_ASSESSABLE = ("No assessment was formed for this file. That says nothing about whether it is genuine: it is not a "
                          "finding either way.")
GLOSSARY = (
    ("Agreement", "How consistently the checks point the same way: high, medium or low. It is not the estimate and not a probability."),
    ("Anomaly map", "A picture of where pixel statistics differ from the rest of the image. Marked regions are candidates for editing, "
                    "not confirmed edits."),
    ("Attention map", "Where a classifier looked when scoring. It is not a map of manipulation."),
    ("Calibration", "Adjusting scores so that an estimate of 80 means that about 80 in 100 test items with that estimate were "
                    "AI-generated or manipulated, on the test set named."),
    ("Check", "One detector or forensic test applied to the file. Each check produces a signal."),
    ("Confidence interval", "The range of values consistent with the test results; the report gives 95 % intervals from resampling "
                            "the test set."),
    ("Content Credentials (C2PA)", "A signed record inside a file that states which software created or edited it. It can be removed, "
                                   "and its absence is not evidence either way."),
    ("Contribution", "How far one check moved the estimate, in log-odds. Positive values point toward indications, negative values "
                     "against."),
    ("Counter-evidence", "The strongest finding that points against the headline. It is always listed."),
    ("Documentary evidence", "What the file itself declares, such as validated Content Credentials. It records what the signing "
                             "software stated, not what the scene shows."),
    ("Error level analysis (ELA)", "How much each area of a JPEG changes when it is saved again. Bright edges and textures are normal."),
    ("Estimate", "A number from 2 to 98: the likelihood that the file was AI-generated or manipulated, on the test population named "
                 "in the report. The uncalibrated fallback gives a score instead."),
    ("Evidence ledger", "The table of every check with its observed value, reference, direction, strength and contribution."),
    ("False alarm", "A file without AI generation or manipulation that a check flagged. Its rate in testing is the false positive rate "
                    "(FPR)."),
    ("Finding", "A statement built from fixed templates and one or more signals, with a strength and a direction."),
    ("Fusion", "Combining the checks into one estimate by adding their evidence on a log-odds scale, weighted by how reliable each was "
               "in testing."),
    ("Hypothesis", "The question a check addresses: whole-content generation (H1), localised manipulation (H2) or declared provenance "
                   "(H3)."),
    # N-19n (D3-07, 8): the strength word of Content Credentials that do not validate
    ("Invalid", "Content Credentials that do not validate: the file or the record changed after signing, or the record is malformed (no ratio)."),
    ("Likelihood ratio", "How much more likely an observation is if the file is AI-generated or manipulated than if it is not. "
                         "Strength words follow its size."),
    ("Log-odds", "The logarithm of p / (1 − p). Adding log-odds is how independent evidence is combined."),
    ("Miss", "An AI-generated or manipulated file that scored below the flagging threshold. The miss rate is one minus the detection "
             "rate (TPR)."),
    ("Model starting point", "Where the fused model stands before it reads any check: its value when every check reports 50, for "
                             "inputs like this one. It comes from fitting the model and is not evidence about the file."),
    ("Natural frequency", "A rate stated as a count, such as 2 in 100 test items. Below 100 test items, the report states k of n."),
    ("Plausible range", "The 90 % interval around the estimate."),
    ("Reliability map", "Where an anomaly map can be trusted. Flat, dark or heavily compressed areas are often unreliable."),
    # N-19n (D3-07): the strength word of a marker's finding, beside "documentary"
    ("Self-declared", "A record in the file that anyone can write (no ratio)."),
    ("Signal", "The measured output of one check: observed value, reference, direction and strength, numbered S01, S02 and so on."),
    ("Validated envelope", "The input conditions (media type, resolution, compression, faces) under which error rates were measured."),
)


# D-63: since N-17 the starting row is the estimate with every scoring check at its neutral point; a result published earlier
# (a `context_prior` row) keeps the term and the definition it was published with. One entry of the same length replaces the
# other, so the glossary still ends on its page
LEGACY_START_TERM = "Model starting point"
NEUTRAL_START_ENTRY = ("Starting point", "The fused model’s estimate for inputs like this one with every check at its neutral point: the "
                                         "score at which that check alone favours neither answer. It comes from fitting the model and "
                                         "is not evidence about the file.")


# N-19q (N-20b's read of the reports, finding 2): the range of the estimate is the range that Methods and the code give
# (`calibration.clamp_probability`: 2 to 98); it read "A number from 1 to 99" beside "the estimate is clamped to 2–98". Under
# the fallback the entry gives the score's own range (`settings.prob_floor`, `prob_ceiling`, 2 to 98 by default)
FALLBACK_ESTIMATE_ENTRY = ("Estimate", "The uncalibrated fallback gives a score instead, a number from {floor} to {ceiling}; it is not a "
                                       "calibrated likelihood.")


def _range_entry(decision: dict[str, Any], entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if decision.get("fusion_mode") != "fallback":
        return entries
    from app.core.config import settings

    entry = (FALLBACK_ESTIMATE_ENTRY[0], FALLBACK_ESTIMATE_ENTRY[1].format(floor=_points(settings.prob_floor), ceiling=_points(settings.prob_ceiling)))
    return [entry if g[0] == "Estimate" else g for g in entries]


def _points(p: float) -> str:
    return f"{p * 100:.0f}" if abs(p * 100 - round(p * 100)) < 1e-9 else f"{p * 100:g}"


def glossary(decision: dict[str, Any]) -> list[tuple[str, str]]:
    if view.start_kind(decision) == "context_prior":
        return _range_entry(decision, list(GLOSSARY))
    entries = sorted([*(g for g in GLOSSARY if g[0] != LEGACY_START_TERM), NEUTRAL_START_ENTRY], key=lambda g: g[0].lower())
    if view.not_assessable(decision):  # N-17 (D-68): no term of an estimate under a result that has none
        return [g for g in entries if not view.speaks_of_estimate(f"{g[0]} {g[1]}")]
    return _range_entry(decision, entries)


# what a report prints for a term of the glossary besides the term itself (lower case, as `used_glossary` reads the text)
GLOSSARY_ALSO = {"Confidence interval": r"\bci\b", "Reliability map": r"\breliability\b", "Natural frequency": r"\b\d+ in 100\b",
                 "Miss": r"\bmissed\b|\btpr\b", "False alarm": r"\bfpr\b|false-alarm", "Likelihood ratio": r"\blr\b",
                 "Hypothesis": r"\bh[123]\b", "Anomaly map": r"\banomaly\b", "Attention map": r"\battention\b"}


def used_glossary(entries: list[dict[str, str]], text: str) -> list[dict[str, str]]:
    """N-19c (design review, P2 "PDFs": the glossary explained anomaly maps in an audio report): the entries whose term stands
    in the text of the report, as a word or in its plural; a term with an abbreviation in brackets counts by either, and a
    term the report prints in another form by that form (`GLOSSARY_ALSO`: "CI", "FPR", "2 in 100")."""
    flat = " ".join(text.split()).lower()

    def used(term: str) -> bool:
        names = [term.split(" (")[0], *re.findall(r"\(([^)]+)\)", term)]
        if term in GLOSSARY_ALSO and re.search(GLOSSARY_ALSO[term], flat):
            return True
        return any(re.search(rf"(?<![\w-]){re.escape(name.lower())}(?:s|es)?(?![\w-])", flat) for name in names)

    return [entry for entry in entries if used(entry["term"])]


def _listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def signed(x: float, dp: int = 2) -> str:
    return f"{'+' if x >= 0 else '−'}{abs(x):.{dp}f}"


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def anchor(prefix: str, ident: str) -> str:
    return f"{prefix}-" + re.sub(r"[^A-Za-z0-9_-]", "-", str(ident))


def standalone_p(signal: dict[str, Any]) -> float | None:
    if view.is_mention(signal):  # N-19o (Q5-01): a mention sets no number; its row has no P(AI) and the figure no dot
        return None
    if signal.get("calibrated_probability") is not None:
        return float(signal["calibrated_probability"])
    return float(signal["raw_score"]) if signal.get("raw_kind") == "probability" and signal.get("raw_score") is not None else None


HEURISTIC_FAMILIES = frozenset({"forensic", "filter"})  # the pixel-statistics checks of v0.3 (error level, noise, copy-move, spectrum)


def p_ai(signal: dict[str, Any]) -> float | None:
    """The P(AI) that the ledger and the figure "Detector estimates" print (N-19q, N-20b's read of the reports, finding 8): none
    for a forensic heuristic that is not in the fitted model and has no calibration of its own, whose map says "Values are
    scores, not probabilities" (the error-level and noise checks printed 0.41 and 0.60 under P(AI) and as dots on the axis)."""
    if (signal.get("calibrated_probability") is None and signal.get("family") in HEURISTIC_FAMILIES
            and (signal.get("outside_estimate") or {}).get("reason") == "not_in_model"):
        return None
    return standalone_p(signal)


def value_column(exhibit: dict[str, Any]) -> str:
    """N-19q (finding 8): the regions table names what its values are (it was headed "Probability or score" over a classifier's
    logits and over map scores): probabilities out of 100, a logit with its unit, or a score."""
    regions = exhibit.get("regions") or []
    probabilities = any(r.get("probability") is not None for r in regions)
    scores = any(r.get("probability") is None and r.get("score") is not None for r in regions)
    if probabilities and scores:
        return "Probability or score"
    if probabilities:
        return "Probability (out of 100)"
    said = f"{exhibit.get('title') or ''} {exhibit.get('caption') or ''}".lower()
    return "Logit (log-odds)" if "logit" in said else "Score"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else f"{text}."


def _capital(text: str | None) -> str:
    return (text or "")[:1].upper() + (text or "")[1:]


def _row(key: str, value: Any, missing: str = NOT_ESTABLISHED) -> dict[str, Any]:
    established = value is not None and value != ""
    return {"key": key, "value": value if established else missing, "established": established}


def _metric(x: float | None, dp: int = 3) -> str:
    return NOT_ESTABLISHED if x is None else f"{x:.{dp}f}"


def _with_ci(x: float | None, ci: Any) -> str:
    if x is None:
        return NOT_ESTABLISHED
    return f"{x:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else f"{x:.3f}"


# --- estimate breakdown (contributions.ts) -----------------------------------------------------------------------------------

def integrity(decision: dict[str, Any]) -> dict[str, Any]:
    """The integrity gate: the rows add up to logit(estimate) within 1e-6 plus the 6 dp rounding bound, or to the 2 / 98 clamp."""
    rows = decision.get("contributions") or []
    p = view.estimate_of(decision)
    if view.record_alone(decision):  # N-19h (E-19, 3): one override row is no breakdown of an estimate
        return {"ok": False, "reason": "record"}
    if decision.get("fusion_mode") == "fallback":
        return {"ok": False, "reason": "fallback"}
    if p is None and view.no_estimate_capped(decision):  # D-49
        return {"ok": False, "reason": "capped"}
    if p is None or p <= 0 or p >= 1:
        return {"ok": False, "reason": "no_estimate"}
    target = logit(p)
    if not rows:
        return {"ok": False, "reason": "no_rows", "target": target}
    total = math.fsum(float(r["logit"]) for r in rows)
    tolerance = TOLERANCE + HALF_DP * len(rows) + HALF_DP / (p * (1 - p))
    if abs(total - target) <= tolerance:
        return {"ok": True, "sum": total, "target": target, "tolerance": tolerance, "clamped": False}
    clamped = ((p >= PROB_CEILING - HALF_DP and total >= logit(PROB_CEILING) - tolerance)
               or (p <= PROB_FLOOR + HALF_DP and total <= logit(PROB_FLOOR) + tolerance))
    return {"ok": clamped, "reason": None if clamped else "mismatch", "sum": total, "target": target, "tolerance": tolerance,
            "clamped": clamped}


NOT_FUSED = "no check gave a score that can be used"  # N-17: Methods and the stages of a not-assessable result
NOT_ASSESSABLE_RULE = (f"The headline “{view.HEADLINE['insufficient']}” is given when no check that tests for AI generation or "
                       "manipulation gives a score that can be used, or when the content is of a kind the checks were not built "
                       "for. It is not a finding either way.")


# N-19f (code review R4-05): the checks scored and the fusion was reached; the model gives them no weight for the input
NO_WEIGHT_FUSED = "for an input like this one the fusion model gives the checks that scored no weight"
NO_WEIGHT_RULE = (f"The headline “{view.HEADLINE['insufficient']}” is given here because, for an input like this one, the fusion "
                  "model gives the checks that scored no weight. It is not a finding either way.")


def no_estimate_formed(decision: dict[str, Any]) -> str:
    """N-17 (D-68; design review DS-10): what "How the result was reached" says of a not-assessable result, in place of the
    narrative, the strip, the breakdown and the calibration of an estimate that does not exist."""
    reason = _sentence(str(decision.get("reason") or "the file could not be assessed"))
    return f"{view.NO_ESTIMATE_FORMED}: {reason[:1].lower()}{reason[1:]}" if reason[1:2].islower() else f"{view.NO_ESTIMATE_FORMED}: {reason}"


def fallback_score_word(decision: dict[str, Any]) -> str | None:
    """N-19h (E-20, 1): what the fallback's hand-set weights gave, in a word. "score" where the number of the result is the
    fallback's own; "statistical score" beside a number that a record set (a floor or a cap does not come from hand-set
    weights); None where a record set the number and no statistical score stands beside it."""
    if not view.record_set_number(decision):
        return "score"
    return None if decision.get("statistical_estimate_before_provenance") is None else "statistical score"


def record_alone_text(decision: dict[str, Any], why: str | None = None) -> str:
    """N-19h (E-19, 3): what stands in place of the breakdown where a record decides alone (it drew "Model starting point
    (before any check) 50", the record's row and "Estimate 85"): there is no statistical estimate to break down, why, and
    what the number is. A declaration shows no number of its own."""
    number = "score" if decision.get("fusion_mode") == "fallback" else "estimate"
    none = f"There is no statistical {number} to break down" + (f": {why}." if why else ".")
    if (view.record_of(decision) or {}).get("kind") == "decisive_ai":
        return f"{none} The provenance record decides the result."
    return f"{none} The number is the value Chipmunk Verify gives the provenance record."


def fallback_text(check: dict[str, Any], decision: dict[str, Any] | None = None, why: str | None = None,
                  profile: dict[str, Any] | None = None) -> str:
    """`why` (N-19h): for a record that decides alone, why no statistical number stands beside it (`view.alone_why`)."""
    reason = check.get("reason")
    if reason == "record":
        return record_alone_text(decision or {}, why)
    if reason == "fallback":
        # N-19c (design review DS-17): it went on "Each check’s weight is listed in the evidence ledger", and the ledger of a
        # report has no weight column
        # N-19n (D3-04; D-68): the fallback's number is never an estimate ("How the score was reached" stood over "… so the
        # estimate came from the uncalibrated fallback")
        came = fallback_score_word(decision or {}) or "score"
        return (f"{view.no_model(profile, 'is loaded')}, so the {came} came from the uncalibrated fallback with hand-set weights. That "
                "path records no additive contributions, and Chipmunk Verify does not draw a breakdown that does not add up.")
    if reason == "capped":
        return "There is no estimate to break down: a check that this analysis depends on did not run."
    if reason == "no_estimate":
        return "There is no estimate to break down: the file could not be assessed."
    if reason == "no_rows":
        return "The fusion model recorded no contribution rows for this analysis."
    return (f"The recorded contributions add up to {signed(check['sum'], 4)} log-odds, but the estimate corresponds to "
            f"{signed(check['target'], 4)}. The waterfall is not drawn.")


LOG_ODDS_TICKS = (-4, -2, 0, 2, 4)  # N-19k (D2-03): the axis of a capped result's breakdown (contributions.ts)


CAPPED_BARS = "Each bar below starts at zero and shows what one check returned, in log-odds."


def capped_combined(kind: str, bars: bool = True, record: bool = False, record_row: bool = False) -> str:
    """N-19k (D2-03; contributions.ts `cappedCombined`): the one sentence over the breakdown of a result outside what was
    validated, in place of the sentences on what pushed the estimate and where the bars start. `bars`: rows are drawn.
    N-19l (R6-02): under a provenance record (`record`) the result states the record's value, so what is not stated is the
    estimate of the statistical checks; where the record has a row among the bars (`record_row`), why it carries no amount."""
    one, _ = view.media(kind)
    what = "the estimate of the statistical checks is not stated" if record else "no estimate is stated"
    return (f"The checks were combined as for any {one}. That combination is not validated for this kind of {one}, so {what}."
            + (f" {CAPPED_BARS}" if bars else "") + (f" {RECORD_ROW_NO_AMOUNT}" if bars and record_row else ""))


# N-19l (R6-02): the row of the record in the breakdown of a result outside what was validated. Its amount would be the
# distance from the combination of the checks, which is not shown, to the record's value: O printed "+5.38" above "Estimate 97"
# N-19n (Q3-05): it ended "… and this one would be the distance from a combination that is not shown.", which only its author
# can read
RECORD_ROW_NO_AMOUNT = "The provenance record is listed without an amount: it is not what a check returned."


def breakdown_flat(decision: dict[str, Any], profile: dict[str, Any] | None, number: view.ResultNumber) -> bool:
    """N-19k (D2-03), N-19l (R6-02; ContributionWaterfall.tsx `flat`): the breakdown of a result outside what was validated
    lets no estimate be read off it, with a record or without one: the result is capped, or the statistical estimate that
    stands beside its record is withheld (U printed "Estimate 98" as its last row under "statistical checks alone: not
    shown")."""
    return number.kind == "capped" or view.statistical_withheld(decision, profile)
RECORD_ROW = "Value set by the record"  # N-19h (E-20, 1): the last row where a record set the number (contributions.ts)


def waterfall(decision: dict[str, Any], signals: list[dict[str, Any]], top: int = WATERFALL_TOP,
              number: view.ResultNumber | None = None, flat: bool | None = None) -> list[dict[str, Any]]:
    """Starting point, the top rows by |Δ|, the rest as "n smaller checks (net)", adjustments, estimate (UX §8.4).

    N-19n (D3-09; ContributionWaterfall.tsx `flatRows`): the breakdown of a result outside what was validated (`flat`)
    lists what the checks returned and the record's row, and nothing else. Its rows are not to be added, so it holds no
    row of the starting point (the constant that completes the sum), none of the fitted term of a check that could not
    score, no row of several checks together ("4 smaller checks (net) +0.19"), and no last row that says "Not shown";
    under a record that set the number the last row stays the record's value (N-19l).

    `number` (N-17; the number rule of D-68, left open by N-7): what the result shows. The last row carries a marker and the
    estimate only for a result that shows its number; an inconclusive or one-check result gets its range without a marker, a
    capped result no number."""
    rows = decision.get("contributions") or []
    by_id = {s["id"]: s for s in signals}
    start = math.fsum(float(r["logit"]) for r in rows if r["kind"] in view.START_KINDS)
    checks = [r for _, r in sorted(((i, r) for i, r in enumerate(rows) if r["kind"] in ("detector", "missing")),
                                   key=lambda x: (-abs(float(x[1]["logit"])), x[0]))]
    # N-19l (R6-02): `flat` is the caller's (`breakdown_flat`: outside what was validated, with a record or without one)
    flat = (number is not None and number.kind == "capped") if flat is None else flat
    if flat:
        checks = [r for r in checks if r["kind"] == "detector"]
    shown = checks if flat else checks[:top] if len(checks) > top + 1 else checks
    rest = checks[len(shown):]
    recorded = view.record_of(decision) is not None
    little = view.little_of(decision)

    def bar(kind: str, label: str, display: str, start_: float, end: float, delta: float, p: float | None = None,
            value: str | None = None, uncounted: bool = False) -> dict[str, Any]:
        value = value or (view.numeral(sigmoid(end)) if kind in ("start", "estimate") else signed(delta))
        # N-19c (browser QA "PDF rendering"): one unit to a column. A check's row holds its change in log-odds; the starting
        # point and the estimate hold a probability out of 100, in a column of its own (`point`)
        if flat and kind == "adjustment" and recorded:
            # N-19l (R6-02): the record's row carries no amount there, and no bar: its amount is no check's (the bound on a
            # provenance check's influence in a result that no record decides is what that check returned, and keeps it)
            return {"kind": kind, "label": label, "display_id": display, "delta": 0.0, "from": 0.0, "to": 0.0, "value": "–", "point": False,
                    "p": None, "svg": figures.waterfall_row_svg("blank", 0.0, 0.0, DOMAIN)}
        # N-19n (D3-10; the page since N-19k): a check that had little effect is drawn thin and in the neutral ink
        slight = kind == "detector" and abs(delta) < little
        if uncounted:
            # N-19p (D6-05): a row that the decision does not count reads "not counted", with no bar (it read "−0.00")
            return {"kind": kind, "label": label, "display_id": display, "delta": delta, "from": start_ if not flat else 0.0,
                    "to": end if not flat else delta, "value": view.NOT_COUNTED, "point": False, "slight": False,
                    "p": None if p is None else view.numeral(p), "svg": figures.waterfall_row_svg("blank", 0.0, 0.0, DOMAIN)}
        if flat and kind != "estimate":
            # N-19k (D2-03): outside what was validated every row is a bar from zero with its own change in log-odds;
            # nothing is drawn on a running total
            return {"kind": kind, "label": label, "display_id": display, "delta": delta, "from": 0.0, "to": delta, "value": signed(delta),
                    "point": False, "slight": slight, "p": None if p is None else view.numeral(p),
                    "svg": figures.waterfall_row_svg("flat", 0.0, delta, DOMAIN, slight)}
        return {"kind": kind, "label": label, "display_id": display, "delta": delta, "from": start_, "to": end, "value": value,
                "point": kind in ("start", "estimate"), "slight": slight,
                "p": None if p is None else view.numeral(p), "svg": figures.waterfall_row_svg(kind, start_, end, DOMAIN, slight)}

    out = [] if flat else [bar("start", NEUTRAL_START_LABEL if view.start_kind(decision) == "neutral_start" else STARTING_POINT_LABEL, "",
                               start, start, start)]
    running = start
    for r in shown:
        s = by_id.get(r.get("signal_id") or "")
        label = ((s or {}).get("name") or r["label"]) + (FROM_FIFTY if r.get("baseline") == "p50" else "")
        out.append(bar("missing" if r["kind"] == "missing" else "detector", label, (s or {}).get("display_id") or "", running,
                       running + float(r["logit"]), float(r["logit"]), standalone_p(s) if s else None,
                       uncounted=r["kind"] == "detector" and r.get("counted") is False))
        running += float(r["logit"])
    if rest:
        begin = running
        running = begin + math.fsum(float(r["logit"]) for r in rest)
        out.append(bar("group", f"{len(rest)} smaller checks (net)", "", begin, running, running - begin))
    for r in rows:
        if r["kind"] == "override":
            s = by_id.get(r.get("signal_id") or "")
            out.append(bar("adjustment", view.row_label(r, decision), (s or {}).get("display_id") or "", running, running + float(r["logit"]),
                           float(r["logit"])))
            running += float(r["logit"])
    estimate = view.estimate_of(decision)  # the reported estimate (clamped 2–98), so the row reads as the summary does (DR P0-2)
    last = bar("estimate", "Estimate", "", running, running, running, value=view.numeral(estimate) if estimate is not None else None)
    if number is not None and number.kind == "text":
        # N-19q (finding 11): a row that holds the plausible range is named so (it read "Estimate 24–85"; the page's `lastRow`)
        last.update(svg=figures.waterfall_row_svg("blank", running, running, DOMAIN), value=number.span or last["value"],
                    label=view.RANGE_WORD if number.span else last["label"])
    # N-19h (E-20, 1): the last row is the record's value, not an estimate. N-19l (R6-04, 2): under a declaration that set
    # the number too (it read "Estimate 97" where a marker's and a capture's row read "Value set by the record"); where
    # the estimate stands above the declaration's value the last row is the statistical estimate, as before
    if record_last(decision, number):
        last.update(svg=figures.waterfall_row_svg("blank", running, running, DOMAIN), label=RECORD_ROW)
    if flat:
        # N-19l (R6-02): under a record outside what was validated the last row is the record's value where the record set
        # the number. N-19n (D3-09): otherwise there is no last row (it read "Not shown: outside what was validated", the
        # row of a total in a list that is not to be added; U printed "Estimate 98" there before N-19l)
        if not (view.record_set_number(decision) and estimate is not None):
            return out
        last.update(svg=figures.waterfall_row_svg("blank", 0.0, 0.0, DOMAIN), point=False, **{"from": 0.0, "to": 0.0, "delta": 0.0},
                    label=f"{RECORD_ROW}: {view.numeral(estimate)} out of 100", value="–")
    out.append(last)
    return out


def record_last(decision: dict[str, Any], number: view.ResultNumber | None) -> bool:
    """The last row of the breakdown is the value that a provenance record set (ContributionWaterfall.tsx `lastRow`): the
    number of the kind `record` (a marker's floor, a capture's cap), or a declaration's value, whose number is of the kind
    `none`."""
    if number is None:
        return False
    return number.kind == "record" or (number.kind == "none" and view.record_set_number(decision) and view.estimate_of(decision) is not None)


WORDS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve")


def _spelled(n: int) -> str:
    return WORDS[n] if n < len(WORDS) else str(n)


def _counted(n: int) -> str:
    """A count as the subject of a clause: `none`, `one`, `two` … (DR P0-2 "no had little effect")."""
    return "none" if n == 0 else _spelled(n).lower()


STARTING_POINT_LABEL = "Model starting point (before any check)"  # L-2: a result published before N-17 (`context_prior`)
NEUTRAL_START_LABEL = "Starting point (every check at its neutral point)"  # D-63, as the UI (contributions.ts)
FROM_FIFTY = " (measured from 50)"  # D-63: a check without a neutral point of its own keeps the 50 baseline and its row says so


def starting_point_line(decision: dict[str, Any]) -> str | None:
    """The start bar in plain words: a property of the fitted model, never a finding about the file. D-63: since N-17 the bars
    start where every scoring check sits at its neutral point; a result published earlier keeps its rows and its wording (L-2)."""
    kind = view.start_kind(decision)
    if kind is None:
        return None
    where = view.start_where(decision)
    if kind == "context_prior":
        return (f"The bars start at {where} out of 100, where the fused model stands before it reads any check (every check at 50). "
                "That starting point comes from fitting the model; it is not evidence about this file.")
    return (f"The bars start at {where} out of 100: the model’s estimate for an input like this one with every check that scored at "
            "its neutral point, the score at which that check alone favours neither answer. That starting point comes from fitting "
            "the model; it is not evidence about this file.")


def from_fifty_line(decision: dict[str, Any]) -> str | None:
    """D-63: the checks whose rows keep the 50 baseline, named."""
    names = [str(r["label"]) for r in decision.get("contributions") or [] if r.get("baseline") == "p50"]
    if not names:
        return None
    one = len(names) == 1
    return (f"{_listed(names)} {'has' if one else 'have'} no neutral point of {'its' if one else 'their'} own, so "
            f"{'its bar is' if one else 'their bars are'} measured from a score of 50.")


def narrative(decision: dict[str, Any]) -> list[str]:
    """N-19i (R5-01, 4): the sentence counts the checks the decision counts (`view.counted_rows`: what the reason's "k of n
    detectors" and the number line count), and its three counts are of those checks. It counted every row of a check: "Two
    checks entered the fusion model" stood over "Calibrated fusion of 1 of 4 detectors" and "only one check scored". Of a
    row that is published and not counted it says so."""
    rows = decision.get("contributions") or []
    detectors = view.counted_rows(decision)
    floor = view.little_of(decision)  # D-63: ln 2; 0.05 for a result published before N-17
    up = sum(1 for r in detectors if float(r["logit"]) >= floor)
    down = sum(1 for r in detectors if float(r["logit"]) <= -floor)
    little = len(detectors) - up - down
    missing = sum(1 for r in rows if r["kind"] == "missing")
    adjustments = [view.row_label(r, decision) for r in rows if r["kind"] == "override"]
    out = [f"{_spelled(len(detectors))} {'check' if len(detectors) == 1 else 'checks'} entered the fusion model. Chipmunk Verify combines them "
           "by adding each check’s evidence on a log-odds scale, weighted by how reliable the check was in testing.",
           f"{_counted(up).capitalize()} pushed the estimate up, {_counted(down)} pushed it down, and {_counted(little)} had little effect.",
           *view.not_counted_lines(decision)]
    if (start := starting_point_line(decision)) is not None:
        out.append(start)
    if (fifty := from_fifty_line(decision)) is not None:
        out.append(fifty)
    if missing:
        out.append(f"{_plural(missing, 'check')} produced no score and {'adds its' if missing == 1 else 'add their'} fitted “missing” term instead.")
    if adjustments:
        out.append(f"The result was then adjusted: {'; '.join(adjustments)}.")
    return out


LABEL_ROOM = 60  # thousandths of the strip between two dots of one row: room for an S-id beside a dot


def agreement(signals: list[dict[str, Any]], little: float = effect.LN2, decision: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Scored checks at their standalone p, lowest first; the direction is the ledger's for the check (`_direction_word`): from
    the contribution (neutral below `little`, D-63), else from the check's own likelihood ratio or recorded direction."""
    scored = sorted(((s, p_ai(s)) for s in signals
                     if s.get("status", "ok") == "ok" and s.get("hypothesis") and s.get("provenance_override") is None),
                    key=lambda x: (x[1] if x[1] is not None else 2.0, x[0]["id"]))
    last = [-math.inf, -math.inf, -math.inf]
    out = []
    for s, p in scored:
        if p is None:
            continue
        # DS-17: the dot says what the check's ledger row says (`_direction_word`), never "p above 50"
        word = _direction_word(s, little, decision)
        direction = {"toward indications": "ai", "toward no indications": "authentic"}.get(word, "neutral")
        x = p * 1000
        row = next((i for i, v in enumerate(last) if x - v >= LABEL_ROOM), last.index(min(last)))
        last[row] = x
        # N-17 (design review DS-17): the dot carries its S-id, printed beside it (on its left near the right-hand end); `at` is
        # the place of the label on the 12 mm strip of `agreement_svg` (the dot's row at 40 − 12 × row of 64 units)
        beside = {"side": "right", "offset": f"{(1 - p) * 100:.2f}%"} if p > 0.9 else {"side": "left", "offset": f"{p * 100:.2f}%"}
        out.append({"display_id": s.get("display_id") or s["id"], "name": s.get("name", ""), "p": view.numeral(p), "p_raw": p,
                    "at": {**beside, "top": f"{(40 - 12 * row) * 12 / 64 - 1.1:.2f}mm"},
                    "direction": word, "key": direction, "row": row})
    return out


# --- sections ------------------------------------------------------------------------------------------------------------------

def _names(snapshot: ReportSnapshot, principal_id: str) -> str:
    return snapshot.actor_names.get(principal_id, principal_id)


def _fuser(result: dict[str, Any]) -> dict[str, Any] | None:
    return next((c for c in result.get("calibration") or [] if str(c.get("subject", "")).startswith("fuser-")), None)


# N-19c (design review, P2 "PDFs": four limitations stated one fact): the facts that a caveat, a limiting condition, the
# report's own sentence and a disclosure can each state. What states a fact that is listed already joins that limitation.
LIMITATION_FACTS = (("fallback", re.compile(r"uncalibrated fallback|no calibrated fusion model", re.IGNORECASE)),
                    ("low_resolution", re.compile(r"low[- ]resolution", re.IGNORECASE)),
                    ("checks_missing", re.compile(r"omitted or incomplete checks|failed or timed out", re.IGNORECASE)))


def _limitation_fact(text: str) -> str | None:
    return next((fact for fact, pattern in LIMITATION_FACTS if pattern.search(text)), None)


def applied_limitations(snapshot: ReportSnapshot, decision: dict[str, Any]) -> list[dict[str, Any]]:
    """UX §6.3 R10 "applied to this exhibit": the result's caveats first (`L-01…` as on screen), then limiting conditions, the
    envelope, failed checks, the fallback fusion, the D-37 lip-sync sentence for video, and the manifest's disclosures.
    N-19c: one limitation for one fact. Each caveat keeps its line and its number (an examiner acknowledges them by number);
    a later sentence that states a fact already listed is printed with that limitation, the fullest one of them only."""
    result, kind = snapshot.result, snapshot.analysis.kind
    profile = result.get("condition_profile") or {}
    items: list[str] = []
    facts: dict[str, int] = {}  # fact → the index of the limitation that states it
    joined: dict[int, list[str]] = {}

    def add(text: str | None, *, caveat: bool = False) -> None:
        if not text or " ".join(text.split()).lower() in {" ".join(i.split()).lower() for i in [*items, *(x for xs in joined.values() for x in xs)]}:
            return
        fact = _limitation_fact(text)
        if fact is not None and fact in facts and not caveat:
            joined.setdefault(facts[fact], []).append(text)
            return
        if fact is not None:
            facts.setdefault(fact, len(items))
        items.append(text)

    for caveat in view.shown_caveats(result, decision):  # N-19h (E-20, 8): a stored statistical number, outside, is not printed
        add(caveat, caveat=True)
    offered = len(items)  # N-19j (Q2-04): the result's own limitations, which the conclusion form puts to the examiner
    for condition in profile.get("limiting_conditions") or []:
        add(f"Limiting condition: {condition}.")
    if profile and not profile.get("within_validated_envelope", True):
        # N-19q (finding 9): the slice in brackets is the input's, not the envelope's ("outside the validated envelope (150 to
        # 600 words)" stood beside "Conditions of this input: 150 to 600 words")
        add(f"The input ({formatting.slice_words(profile.get('slice_id'))}) is outside the validated envelope: no measured error rate describes inputs like it.")
    by_status = snapshot.analysis.coverage.by_status if snapshot.analysis.coverage is not None else {}
    failed = int(by_status.get("failed", 0)) + int(by_status.get("timeout", 0))
    if failed:
        add(f"{_plural(failed, 'check')} failed or timed out, so fewer checks informed the result.")
    server = view.server_sentence(snapshot.analysis.coverage.model_dump() if snapshot.analysis.coverage is not None else None)
    # N-14a: the server's memory budget or missing model files, not the file. N-19k (browser QA lead 3): a result composed
    # since then holds one caveat that names these checks, so the report adds no sentence of its own beside it (n checks
    # gave n + 1 limitations); a stored result without it keeps this one
    if server and not any(ON_THIS_SERVER in c and view.SERVER_FEWER in c for c in result.get("caveats") or []):
        add(f"{server} {view.SERVER_FEWER}")
    if decision.get("fusion_mode") == "fallback":
        none = view.no_model(result.get("condition_profile"))  # N-19k (Q2-14): "loaded" only where none is
        add(f"{none}: nothing in this result is calibrated." if view.not_assessable(decision)  # N-17 (D-68)
            else f"{none}: nothing in this result is calibrated." if fallback_score_word(decision) is None
            else f"{none}: the {fallback_score_word(decision)} comes from hand-set weights and is not a "
                 "calibrated probability.")
    for item in disclosures.for_result({**result, "kind": kind}):  # L-3: the `not_validated.yaml` disclosures that affect this result
        add(item["text"])
    learned_lipsync = any(s.get("detector_id") == "vid.lipsync.stable_syncnet" and s.get("status") == "ok" for s in result.get("signals") or [])
    if kind == "video" and not learned_lipsync:
        add(LIPSYNC_LIMITATION)
    manifest = result.get("manifest") or {}
    for disclosure in manifest.get("disclosures") or []:
        add(disclosure)
    if "research" in str(manifest.get("licence_posture") or ""):
        add("Research-licensed components were used.")
    # N-19 (browser QA P1-8): a recorded custodian hash mismatch is a limitation too. Last, so the numbers of the limitations
    # an examiner acknowledged on the page (the list above, in this order) do not move
    if view.custodian_mismatch(snapshot):
        add(view.CUSTODIAN_MISMATCH_LIMITATION)
    acknowledged = set(snapshot.review.conclusion.limitations_acknowledged) if snapshot.review.conclusion is not None else set()
    texts = [f"{_sentence(text)} {_sentence(max(joined[n], key=len))}" if n in joined else text for n, text in enumerate(items)]
    return [{"id": f"L-{i:02d}", "text": text, "acknowledged": f"L-{i:02d}" in acknowledged, "offered": i <= offered}
            for i, text in enumerate(texts, 1)]


def estimate_sentence(decision: dict[str, Any], basis: str, coverage: dict[str, Any] | None,
                      findings: list[dict[str, Any]] | None = None, profile: dict[str, Any] | None = None, *, failed: bool = False) -> str:
    """The sentence under the headline of the standard and forensic reports. The number follows `view.result_number` (D-68).
    `failed` (N-19n): the step that writes the findings failed (`view.agreement`)."""
    n = view.result_number(decision, findings, profile)
    ran, agree = view.checks(coverage), view.agreement(decision, coverage, findings, failed=failed)
    # "Estimate 93–98: only one check scored." already says why no agreement is reported
    # N-19m (D3-01): none under a trusted capture either, as none is printed under a declaration: the result is the record's
    agreed = ("" if view.agreement_beside(n, agree) != agree or basis == "documentary_capture"
              else f" Agreement among the {ran} that ran is {agree}.")
    if basis == "insufficient":
        return view.no_estimate_line(coverage, decision)
    if basis == "documentary_declared":
        # N-19m (D3-05; R7-01): one sentence for the statistical number beside every record, with its unit ("The statistical
        # checks alone estimate 90." stood here) and with what the checks alone decided
        return ("The file’s signed credentials declare AI generation, which Chipmunk Verify treats as documentary evidence. "
                f"{view.statistical_sentence(decision, profile)}")
    if view.no_estimate_capped(decision):
        return view.NO_ESTIMATE_NOT_RUN
    if n.kind == "none":
        return "No estimate was produced."
    if n.kind == "capped":
        return view.NO_ESTIMATE_OUTSIDE
    if n.kind == "record":  # N-19h (E-20, 1): the record's value in words, and what the statistical checks alone give
        return f"{view.record_sentence(decision)} {view.statistical_sentence(decision, profile)}{agreed}"
    if n.kind == "uncalibrated":
        return f"{_capital(n.text)}: it comes from hand-set weights.{agreed}"  # short words: the lay summary's reading grade
    if n.kind == "text":
        return f"{n.text}.{agreed}"
    rng = f", with a plausible range of {n.span.replace('–', ' to ')}" if n.span else ""
    return f"The estimated likelihood is {view.numeral(n.estimate)} out of 100{rng}.{agreed}"


def lay_summary(snapshot: ReportSnapshot, decision: dict[str, Any], basis: str, limitations: list[dict[str, Any]]) -> dict[str, Any]:
    """RP-01 (1): what was examined, the decision in words, main reasons, main limitation, what this does not mean, next step."""
    kind, result = snapshot.analysis.kind, snapshot.result
    coverage = snapshot.analysis.coverage.model_dump() if snapshot.analysis.coverage is not None else None
    received = snapshot.media.received_at
    examined = f"Chipmunk Verify examined {ARTICLE.get(kind, 'a file')}" + (f" taken in on {formatting.utc_date(received)}" if received else "") + "."
    means, does_not_mean = view.guidance(kind, decision, basis, coverage, list(result.get("findings") or []),
                                         list(result.get("signals") or []), result.get("condition_profile"),
                                         caveats=result.get("caveats"))
    # N-10b (code re-verification R2-08): a capped result has no "Main reasons". Its findings carry strengths that were
    # measured on inputs unlike this one ("Strong support against AI generation" under a result that shows no estimate);
    # the reason for the result is the cap, which is the main limitation below. The findings stay in the fuller reports
    # N-19h: keyed on the input, not on the number (a capture's result outside what was validated shows the record's number)
    capped = view.outside_validated(decision, result.get("condition_profile"))
    failed = effect.findings_failed(result.get("caveats"))  # N-19n (Q4-01)
    reasons = []
    statuses = view.finding_statuses(snapshot)  # D-69: the examiner's status stands beside the finding wherever it is listed
    # N-19h (E-21): where a record decides such a result, its finding is the reason that is given
    listed = list(result.get("findings") or [])
    for f, counter in (view.record_findings(listed, decision) if capped else view.key_findings(listed, decision)):
        words = view.strength_words(f, decision)
        reason = view.reason_of(f, counter)
        # Documentary reasons can quote arbitrary observed values. Keep those verbatim,
        # including weaker F1/F2 findings whose titles do not say "Documentary".
        if f.get("family") not in {"F1_provenance", "F2_self_declared_metadata"} and not view.is_documentary(f):
            # Split the synthesis fallback's ledger pointer without dropping a claim.
            reason = prose.system_prose(reason).replace(
                "; what it measured is listed in the evidence ledger.",
                ". See the evidence ledger for what this check measures.",
            )
        reasons.append({"id": f.get("id", ""), "words": words, "title": f.get("title", ""), "counter": counter, "text": reason,
                        "status": statuses.get(str(f.get("id")))})
    # N-7: the main limitation of a capped result is its cap (why no measured error rate describes the input), not the first caveat
    cap = next((x for x in limitations if x["text"].startswith(view.CAP_CAVEAT)), None)
    limitation = _sentence((cap or limitations[0])["text"]) if limitations else GENERAL_LIMITATIONS["all"][1]
    testing = view.in_testing(kind, decision, result.get("condition_profile"), list(result.get("calibration") or []))["line"]
    # L-3: findings that carry no weight in this decision: listed after the reasons, and named in one short sentence of the text
    context = view.context_findings(list(result.get("findings") or []), decision)
    context_line = ("Checks outside the fused estimate (no weight in this decision, shown for the examiner): "
                    + "; ".join(f"{f.get('id', '')} {f.get('title', '')}" for f in context) + ".") if context else ""
    ids = [str(f.get("id", "")) for f in context]
    context_short = (f"{', '.join(ids[:-1]) + ' and ' + ids[-1] if len(ids) > 1 else ids[0]} "
                     f"{'are' if len(ids) > 1 else 'is'} shown as context. {'They carry' if len(ids) > 1 else 'It carries'} "
                     "no weight here.") if ids else ""
    if capped:
        context_line, context_short = view.CAPPED_CHECKS_BELOW, view.CAPPED_CHECKS_BELOW
    # N-19n (D3-07, 3): the provenance findings that did not decide this result, after the checks outside the estimate
    aside = view.provenance_aside(list(result.get("findings") or []), decision)
    aside_line = (f"{view.ASIDE_TITLE}: " + "; ".join(f"{f.get('id', '')} {f.get('title', '')}" for f in aside) + f". {view.ASIDE_NOTE}") if aside else ""
    aside_ids = [str(f.get("id", "")) for f in aside]
    aside_short = (f"{', '.join(aside_ids[:-1]) + ' and ' + aside_ids[-1] if len(aside_ids) > 1 else aside_ids[0]} "
                   f"{'are records' if len(aside_ids) > 1 else 'is a record'} of the file that did not decide this result.") if aside_ids else ""
    out = {"examined": examined, "headline": view.headline(decision, basis),
           "estimate": estimate_sentence(decision, basis, coverage, list(result.get("findings") or []), result.get("condition_profile"), failed=failed),
           "means": means, "reasons": reasons, "context": context_line, "aside": aside_line, "insufficient": view.insufficient_reasons(decision, result.get("condition_profile"), coverage) if basis == "insufficient" else [],
           "does_not_mean": does_not_mean, "limitation": f"Main limitation: {main_limitation(limitation)}", "testing": testing,
           "next_step": view.next_step(decision, basis, coverage, failed=failed)}
    parts = [examined, _sentence(out["headline"]), out["estimate"], *means, *(_sentence(r) for r in out["insufficient"]),
             *(f"{_sentence(r['title'])} {_sentence(r['text'])}" for r in reasons), context_short, aside_short, *does_not_mean, out["limitation"], testing, out["next_step"]]
    out["text"] = " ".join(p for p in parts if p)
    out["lint"] = lint.lay_summary(out["text"])
    return out


RAN_WITHOUT_SCORE = ("Not addressed: checks for this question ran, but none produced a score that can be used. Their values "
                     "are shown for inspection only.")
# N-19n (D3-06): under a not-assessable result without a fusion model, which does not separate the questions: no sentence
# claims a check "for this question", and none speaks of one score that answers the questions together (there is no score)
RAN_TOGETHER = "Not addressed: the checks that ran produced no score that can be used. Their values are shown for inspection only."
NO_CHECK_APPLIED = "Not addressed: no check could be applied to this file."
NO_CHECK_FOR_QUESTION = "Not addressed: no check for this question produced a result."


def main_limitation(text: str) -> str:
    """N-19k: the first limitation as the one-line "Main limitation: …" gives it. A limiting condition of the input stands in
    the list as "Limiting condition: <condition>."; the line read "Main limitation: Limiting condition: …"."""
    rest = text.removeprefix("Limiting condition: ")
    return rest[:1].upper() + rest[1:] if rest != text else text


def scope(snapshot: ReportSnapshot) -> dict[str, Any]:
    """Questions addressed and not addressed (UX §6.3 R2, §7.3 rule 9).

    N-17 (browser QA P1-5 f): in a result whose rows are measured from neutral points, a question of content (H1, H2) is
    addressed by the checks that are in the estimate. The row counts them among the checks that ran; a question whose checks
    all stayed outside the estimate is not addressed, whatever ran for it. Other results keep their wording."""
    result, kind = snapshot.result, snapshot.analysis.kind
    signals, findings = result.get("signals") or [], result.get("findings") or []
    assessments = {a["hypothesis"]: a for a in result.get("assessments") or []}
    by_weight = view.measured_from_neutral(result.get("decision"))
    # N-19k (D2-02): under a result that is not assessable a question of content whose checks ran is not "not addressed:
    # no check … produced a result" over a finding that counts the checks that did. A result without a fusion model does
    # not separate the questions (its checks name none), so there the checks that ran are the result's
    not_assessable = (result.get("decision") or {}).get("code") == "not_assessable"
    separated = any(s.get("hypothesis") for s in signals)
    coverage = getattr(snapshot.analysis, "coverage", None)
    ran_at_all = int((getattr(coverage, "by_status", None) or {}).get("ok", 0)) > 0
    # N-19n (D3-06): a result without a fusion model answers the questions of content together, with one score. Where it
    # is not assessable there is no score: its rows said "Not assessed separately: … one score answers the questions
    # together" (a recording of silence, to which no check was applied) and "checks for this question ran" of a question
    # for which no check exists (the fallback's checks name no question)
    together = (result.get("decision") or {}).get("fusion_mode") == "fallback"
    no_check_at_all = coverage is not None and int(getattr(coverage, "detectors_total", 0) or 0) == 0
    listed = {str(d.get("id")) for d in (result.get("manifest") or {}).get("detectors") or [] if d.get("id")}
    rows = []
    for hypothesis, label in ((H1, "H1"), (H2, "H2"), (H3, "H3")):
        question = QUESTIONS[hypothesis].get(kind, QUESTIONS[hypothesis]["image"])
        assessment = assessments.get(hypothesis)
        ran = [s for s in signals if s.get("hypothesis") == hypothesis and s.get("status", "ok") == "ok"]
        # N-19q (N-20b's read of the reports, finding 9): the row counts the checks that ran, as "the 3 checks that ran" does,
        # not their signals: a derived signal (the sentence-level consensus of the text checks) read "Addressed by 4 checks"
        checks = _checks_of(ran, listed)
        addressed = bool(ran) or (assessment is not None and assessment.get("applicable") and
                                  (hypothesis != H3 or assessment.get("c2pa_state") is not None))
        if not signals and assessment is None:
            addressed = any(f.get("hypothesis") == hypothesis for f in findings)
        note = (assessment or {}).get("note")
        if hypothesis != H3 and note and str(note).startswith(D49_NOTES):  # D-49: no estimate, so not addressed whatever ran
            addressed = False
            how = _sentence(note[:1].upper() + note[1:]) + (f" {_plural(len(checks), 'check')} ran." if checks else "")
        elif by_weight and hypothesis != H3 and ran:
            inside = len(_checks_of([s for s in ran if effect.entered(s.get("contribution_logit"))], listed))
            addressed = inside > 0
            how = f"Addressed by {_plural(len(checks), 'check')}. In the estimate: {inside} of {len(checks)}." if addressed else (
                f"Not addressed by the estimate: {_plural(len(checks), 'check')} ran for this question and "
                f"{'is' if len(checks) == 1 else 'are'} shown for inspection; none carries weight in this estimate.")
        elif not_assessable and hypothesis != H3 and together and (not note or str(note).startswith(V1_NOTE)):
            # (a note that gives a reason of its own stays: "the document is too short for the text checks")
            addressed = False
            how = RAN_TOGETHER if ran_at_all else NO_CHECK_APPLIED if no_check_at_all else NO_CHECK_FOR_QUESTION
        elif not_assessable and hypothesis != H3 and (ran or (not separated and ran_at_all)):
            addressed, how = False, RAN_WITHOUT_SCORE
        else:
            how = (f"Addressed by {_plural(len(checks), 'check')}." if checks else "Addressed.") if addressed \
                else _sentence(_capital(formatting.stored_words(str(note)))) if note else NO_CHECK_FOR_QUESTION  # N-19q (finding 10)
        rows.append({"id": label, "anchor": anchor("hyp", hypothesis), "question": question, "addressed": addressed, "how": how})
    return {"questions": rows, "not_addressed": list(NOT_ADDRESSED)}


def _checks_of(signals: list[dict[str, Any]], listed: set[str]) -> set[str]:
    """The checks of these signals: their detectors, those of the run's manifest where it lists any (a derived signal of the
    v0.3 text pipeline names no detector of the run)."""
    ids = {str(s.get("detector_id") or s.get("id") or f"#{i}") for i, s in enumerate(signals)}
    return (ids & listed) or ids


def xref(ident: str, signals: dict[str, dict[str, Any]], exhibits: dict[str, dict[str, Any]]) -> dict[str, str] | None:
    if ident in signals:
        s = signals[ident]
        return {"label": s.get("display_id") or s.get("name") or ident, "href": "#" + anchor("sig", ident)}
    if ident in exhibits:
        e = exhibits[ident]  # N-19c: under the label, and at the anchor, of the row that prints it (`exhibit_labels`)
        return {"label": e.get("display_id") or e.get("title") or ident, "href": "#" + (e.get("anchor") or anchor("ex", ident))}
    if ident in (H1, H2, H3):
        return {"label": {H1: "H1", H2: "H2", H3: "H3"}[ident], "href": "#" + anchor("hyp", ident)}
    if ident == "decision":
        return {"label": "Summary", "href": "#section-summary"}
    if ident == "condition_profile":
        return {"label": "Condition profile", "href": "#conditions"}
    if ident.startswith("detector:"):
        return {"label": ident.removeprefix("detector:"), "href": "#" + anchor("det", ident.removeprefix("detector:"))}
    return None


def dispositions(snapshot: ReportSnapshot, signals: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest examiner status per finding or signal (RW-01; dispositions are annotations, latest wins)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for a in sorted(snapshot.annotations, key=lambda a: (a.edited_at or a.created_at, a.id)):
        if a.examiner_status and a.target_type in ("finding", "signal") and a.target_id:
            target = a.target_id if a.target_type == "finding" else (signals.get(a.target_id, {}).get("display_id") or a.target_id)
            out[(a.target_type, a.target_id)] = {
                "target": target, "status": EXAMINER_STATUS.get(a.examiner_status, a.examiner_status),
                "note": a.body if a.visibility == "in_report" else ("Internal note, not reproduced in reports." if a.body else None),
                "author": _names(snapshot, a.author_id), "at": formatting.utc(a.edited_at or a.created_at)}
    return out


def finding_rows(snapshot: ReportSnapshot, images: dict[str, str], signals: dict[str, dict[str, Any]],
                 exhibits: dict[str, dict[str, Any]], marks: dict[tuple[str, str], dict[str, Any]],
                 thumbnail_details: dict[str, dict[str, str]] | None = None) -> list[dict[str, Any]]:
    rows, printed = [], set()
    decision = view.decision_of(snapshot.result)
    for f in sorted(snapshot.result.get("findings") or [], key=lambda f: (int(f.get("rank", 0)), str(f.get("id")))):
        strength, direction = str(f.get("strength", "none")), view.finding_direction(f)
        documentary = view.is_documentary(f)
        marker = view.record_word(f, decision)  # N-19n (D3-07): "self-declared", or "not applied"; no strength, no direction
        words = "documentary" if documentary else marker or view.STRENGTH_WORDS.get(strength, "little effect")
        if not documentary and marker is None and strength != "none" and direction in ("for", "against"):
            words = f"{words} {direction}"
        cited = dict.fromkeys([*(f.get("evidence_ids") or []), *(i for s in f.get("sentences") or [] for i in s.get("evidence_ids") or [])])
        refs = [r for r in (xref(i, signals, exhibits) for i in cited) if r is not None]
        captions = {n for n in (f.get("benchmark_note"), f.get("dissent_note")) if n}
        sentences = [s["text"] for s in f.get("sentences") or []] or [f.get("statement", "")]
        sentences = [text for text in sentences if text not in captions]
        # These findings may quote source metadata, including weaker documentary variants.
        if f.get("family") not in {"F1_provenance", "F2_self_declared_metadata"} and not documentary:
            sentences = [prose.system_prose(text) for text in sentences]
        benchmark = f.get("benchmark_note") if f.get("benchmark_note") not in printed else None  # a shared AUC caption prints once (DR P1-7)
        printed.add(f.get("benchmark_note"))
        notes = [n for n in (benchmark, f.get("dissent_note")) if n]
        if f.get("conditions"):
            notes.append("Conditions: " + "; ".join(f["conditions"]) + ".")
        modes = card_copy.client_modes(f.get("fp_modes") or [])  # N-19k: a stored mode in the build's shorthand is not printed
        if modes:
            notes.append("Known false alarms: " + "; ".join(modes) + ".")
        detail = (thumbnail_details or {}).get(str(f.get("id")), {})
        aside = view.set_aside(f, decision)  # N-19k (Q2-14; D-64): a metadata block under a result that is not assessable
        if aside:
            strength, words, notes = "none", view.SET_ASIDE_WORDS, [view.SET_ASIDE_NOTE, *notes]
        rows.append({"id": f.get("id"), "anchor": anchor("fnd", str(f.get("id"))), "documentary": documentary, "bars": view.BARS.get(strength, 0),
                     "mark": None if aside else view.finding_mark(f, decision), "words": words, "title": view.SET_ASIDE_TITLE if aside else f.get("title", ""), "sentences": sentences,
                     "notes": [prose.system_prose(n) for n in notes], "refs": refs, "thumbnail": images.get(f"thumb:{f.get('id')}"),
                     "thumbnail_caption": detail.get("caption"), "thumbnail_alt": detail.get("alt", "Evidence map"),
                     "disposition": marks.get(("finding", str(f.get("id"))))})
    return rows


TEST_VIEWS = {"size_matched": "size-matched test set"}  # J-3d: the artefact's `metrics.test.view` in words


def test_view_words(view: str | None) -> str:
    return TEST_VIEWS.get(str(view), str(view).replace("_", " ")) if view else "test view not recorded"


def historical_calibration(fuser: dict[str, Any] | None, decision: dict[str, Any], kind: str) -> dict[str, Any]:
    """Validate frozen evidence and its local identity bindings; never fetch a newer benchmark for an old report."""
    raw = (fuser or {}).get("reliability_diagram")
    if raw is None:
        return {"status": "absent", "note": "Historical calibration diagram not recorded for this result."}
    unavailable = {"status": "unavailable", "note": "Historical calibration diagram unavailable: stored data or identity binding is invalid."}
    try:
        record = ReliabilityDiagram.model_validate(raw)
    except ValidationError:
        return unavailable
    assert fuser is not None
    if (decision.get("fusion_mode") != "artefact"
            or record.population != f"{kind}/test"
            or not str(fuser.get("subject", "")).startswith(f"fuser-{kind}-")
            or record.published_artefact_sha256 != fuser.get("artefact_sha256")
            or record.bench_version != fuser.get("bench_version")
            or record.manifest_sha256 != fuser.get("bench_manifest_sha256")
            or record.published_artefact_sha256 != decision.get("fuser_sha256")):
        return unavailable
    ece = fuser.get("ece")
    if isinstance(ece, bool) or not isinstance(ece, int | float) or not math.isfinite(ece) or ece != record.ece:
        return unavailable
    parent_view = fuser.get("test_view")
    if parent_view not in (None, "size_matched") or (record.view == "size_matched") != (parent_view == "size_matched"):
        return unavailable
    return {**record.model_dump(mode="json"), "status": "recorded",
            "view_label": "size-matched test set" if record.view == "size_matched" else "full test set",
            "ece_label": _metric(record.ece), "ece_ci_label": f"[{record.ece_ci95[0]:.3f}, {record.ece_ci95[1]:.3f}]",
            "svg": figures.reliability_svg([(b.bin, b.mean_s, b.mean_y) for b in record.bins]),
            "support_svg": figures.calibration_support_svg([b.n for b in record.bins])}


def estimate_section(snapshot: ReportSnapshot, decision: dict[str, Any]) -> dict[str, Any]:
    result, kind = snapshot.result, snapshot.analysis.kind
    signals = list(result.get("signals") or [])
    check = integrity(decision)
    fuser = _fuser(result)
    if decision.get("fusion_mode") == "fallback":
        word = fallback_score_word(decision)  # N-19h (E-20, 1): a floor or a cap does not come from hand-set weights
        none = view.no_model(result.get("condition_profile"))  # N-19k (Q2-14)
        statement = (f"{none}, and this result publishes no statistical score." if word is None
                     else f"{none}. The {word} comes from the uncalibrated fallback with hand-set weights: "
                          "it is a score, not a calibrated probability.")
    elif fuser is not None:
        ece = f", expected calibration error {fuser['ece']:.3f} ({test_view_words(fuser.get('test_view'))})" if fuser.get("ece") is not None else ""
        # N-19h (E-20, 5): the sentence names what is calibrated. Beside a record's value that is the statistical estimate; in
        # a result without a statistical estimate it is the fusion model, of which this result publishes no estimate
        how = f"calibrated by {formatting.calibration_method(fuser.get('method'))} on {fuser.get('bench_version') or 'an unnamed test set'}{ece}"
        statistical = decision.get("statistical_estimate_before_provenance")
        # N-19n (Q3-06): also where a record sets the result's number or decides without one, and the statistical estimate
        # beside it is not shown because the input is outside what was validated ("The statistical estimate is calibrated
        # by …" stood directly under "the estimate of the statistical checks is not shown")
        outside = (view.result_number(decision, list(result.get("findings") or []), result.get("condition_profile")).kind == "capped"
                   or view.statistical_withheld(decision, result.get("condition_profile")))
        if outside:  # N-19k (D2-03): the calibration is the model's; it does not reach an input outside what was validated
            subject = f"The fusion model is {how}. That calibration does not cover this kind of input, so this result states no estimate"
        elif view.estimate_of(decision) is None or (view.record_set_number(decision) and statistical is None):
            subject = f"The fusion model is {how}; this result publishes no estimate of it"
        elif view.record_set_number(decision):
            subject = f"The statistical estimate is {how}"
        else:
            subject = f"The estimate is {how}"
        statement = (f"{subject}. "
                     f"Fusion artefact {formatting.short_hash(fuser.get('artefact_sha256'))}"
                     f", {formatting.review_phrase(fuser.get('reviewed_by'), fuser.get('review_date'))}.")
    else:
        statement = (f"The estimate came from fusion artefact {formatting.short_hash(decision.get('fuser_sha256'))}, but no calibration "
                     "record is attached to the result, so its calibration is not established.")
    _, many = view.media(kind)
    rates = [{"context": r["context"], "threshold": f"{formatting.threshold_word(r['threshold_kind'])} · {float(r['threshold']):.2f}",
              "false_alarms": f"{view.frequency(float(r['fpr']), int(r['n_neg']))} genuine {many}",
              "detections": f"{view.frequency(float(r['tpr']), int(r['n_pos']))} AI-generated or manipulated {many}",
              "fpr": _with_ci(float(r["fpr"]), r.get("fpr_ci95")), "tpr": _with_ci(float(r["tpr"]), r.get("tpr_ci95")),
              "n": f"{r['n_neg']} / {r['n_pos']}", "bench": r["bench_version"]} for r in (fuser or {}).get("error_rates") or []]
    dots = agreement(signals, view.little_of(decision), decision)
    p = view.estimate_of(decision)
    # N-17 (the number rule of D-68): the strip's bar and the last row of the breakdown show what the summary shows
    number = view.result_number(decision, list(result.get("findings") or []), result.get("condition_profile"))
    marker = number.kind == "calibrated"
    if view.not_assessable(decision):  # N-17 (D-68; DS-10): no narrative, strip, breakdown or calibration of an estimate that was not formed
        return {"none": no_estimate_formed(decision), "narrative": [], "integrity": check, "bars": [], "fallback": None, "ticks": [],
                "axis": None, "dots": [], "agreement_svg": None, "statement": None, "rates": [], "reliability_diagram": None,
                "calibration": []}
    ticks = [{"label": str(t), "left": f"{_x_pct(logit(t / 100)):.2f}%"} for t in PROB_TICKS]
    # N-19l (R6-02): the breakdown of N-19k is the breakdown of every result outside what was validated, under a record too
    capped = breakdown_flat(decision, result.get("condition_profile"), number)
    recorded = view.record_of(decision) is not None
    if capped:  # N-19k (D2-03): the axis of a result outside what was validated is in log-odds; it has no tick of a probability
        ticks = [{"label": "0" if t == 0 else signed(t, 0), "left": f"{_x_pct(t):.2f}%"} for t in LOG_ODDS_TICKS]
    # N-19h (E-19, 2): where a record decides alone no sentence of the fusion is printed (how many checks entered the model,
    # what pushed the estimate, "then adjusted"); the breakdown is a sentence
    alone = view.record_alone(decision)
    record_row = recorded and any(r.get("kind") == "override" for r in decision.get("contributions") or [])
    bars = waterfall(decision, signals, number=number, flat=capped) if check["ok"] else []
    told = ([capped_combined(kind, bool(bars), recorded, record_row)] if capped
            else narrative(decision) if decision.get("contributions") and not alone else [])
    return {"none": None, "title": reached_title(result), "narrative": told, "integrity": check, "flat": capped,
            "record_last": record_last(decision, number),
            "bars": bars,
            "fallback": None if check["ok"] else fallback_text(check, decision, view.alone_why(decision, list(result.get("assessments") or []), kind),
                                                                      result.get("condition_profile")),
            "marker": marker, "number": number.kind,
            "number_text": _capital(number.text) if number.kind == "record" else number.text if number.kind in ("text", "uncalibrated") else None,
            "ticks": ticks, "axis": figures.axis_svg([float(t) for t in LOG_ODDS_TICKS] if capped else [logit(t / 100) for t in PROB_TICKS], DOMAIN),
            "dots": dots, "agreement_svg": figures.agreement_svg([(d["p_raw"], d["key"], d["row"]) for d in dots], p if marker else None) if dots else None,
            "statement": statement, "rates": rates,
            "reliability_diagram": historical_calibration(fuser, decision, kind),
            "calibration": [{"subject": formatting.calibration_subject(c.get("subject")), "method": formatting.calibration_method(c.get("method")),
                             "bench": c.get("bench_version") or NOT_RECORDED,
                             "ece": _metric(c.get("ece")), "brier": _metric(c.get("brier")), "rows": len(c.get("error_rates") or []),
                             "reviewed_by": formatting.reviewer(c.get("reviewed_by"), c.get("review_date"))} for c in result.get("calibration") or []]}


def _x_pct(v: float) -> float:
    return (max(-DOMAIN, min(DOMAIN, v)) + DOMAIN) / (2 * DOMAIN) * 100


def _regions_of_map(heatmap: dict[str, Any], e: dict[str, Any]) -> bool:
    """Whether `e` is the table of the regions marked on `heatmap`: joined by source signal, or by the creator or the legacy
    localise title where one of the two names no signal."""
    if e.get("kind") != "regions" or regions_lens(e) != "regions":
        return False
    source_signals, region_signals = set(heatmap.get("source_signal_ids") or []), set(e.get("source_signal_ids") or [])
    same_creator = bool(heatmap.get("created_by")) and e.get("created_by") == heatmap.get("created_by")
    localise_title = e.get("created_by") == "localise" and e.get("title") == f"Regions from {heatmap.get('title', '')}"[:255]
    return bool(source_signals & region_signals) or ((same_creator or localise_title) and not (source_signals and region_signals))


def print_order(exhibits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """N-19c (design review DS-18; browser QA "PDF rendering"): the exhibits in the order the report prints them. That is the
    stored order of the result, which is the order of the viewer's catalogue and the same for every analysis of one file;
    the order used to fall back to the exhibit id, which differs from analysis to analysis, so a re-run renumbered the
    figures. The table of a map's regions follows its map directly."""
    maps = [e for e in exhibits if e.get("kind") == "heatmap"]
    after: dict[str, list[dict[str, Any]]] = {}
    moved: set[str] = set()
    for e in exhibits:
        owner = next((m for m in maps if _regions_of_map(m, e)), None)
        if owner is not None:
            after.setdefault(str(owner["id"]), []).append(e)
            moved.add(str(e["id"]))
    return [x for e in exhibits if str(e["id"]) not in moved for x in (e, *after.get(str(e["id"]), ()))]


def number_exhibits(rows: list[dict[str, Any]]) -> None:
    """N-19c (DS-18): what is drawn is a figure and what is a table is a table, each numbered from 1 in the order of print, so
    the numbers rise by one. An exhibit that prints neither (segments that the temporal chapter lists, a face track without
    stills, series that are stored apart) carries its title alone. A map or a picture is a figure also where its image
    cannot be shown, so its number does not depend on what a report could load."""
    figure = table = 0
    for row in rows:
        drawn = (row["kind"] in ("heatmap", "image", "contact_sheet") or bool(row["series"] and row["series"]["strips"])
                 or bool(row["stills"]))
        if drawn:
            figure += 1
            row["label"] = f"Fig. {figure}"
        elif row["table"] is not None:
            table += 1
            row["label"] = f"Table {table}"
        else:
            row["label"] = ""


def _track_reference(printed: dict[str, str] | None, e: dict[str, Any]) -> str:
    """How the temporal chapter points at a face track's exhibit: by its figure number, or in words where it has none (its
    title is the face's name, which the line already says)."""
    label = printed["label"] if printed else e.get("display_id") or e.get("title")
    return "listed under Exhibits" if label == e.get("title") else str(label)


def exhibit_labels(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Exhibit id → the label and the anchor of the row that prints it (a sampled frame: the contact sheet's)."""
    return {i: {"label": row["label"] or row["title"], "anchor": row["anchor"]} for row in rows for i in row["ids"]}


def _box_text(bbox: dict[str, Any]) -> str:
    return f"x {bbox['x']:.0f}, y {bbox['y']:.0f}, {bbox['w']:.0f} × {bbox['h']:.0f} px"


_CLONE_LABEL = re.compile(r"^(C\d+)(′?)")
_FRAME_BOX = re.compile(r"^frame (\d+) face (\d+)$")  # the label of a video's per-frame face box (`FrameFacesDetector`)


def regions_lens(exhibit: dict[str, Any]) -> str:
    """faces, clones, crops or regions, by the viewer's rule (`regionsLens` and `isCrops` in the viewer's lenses module), so the PDF
    names a region as the screen does (DR P0-1; D-26)."""
    labels = [str(r.get("label") or "") for r in exhibit.get("regions") or []]
    source = f"{exhibit.get('created_by', '')} {exhibit.get('title', '')}"
    if re.search(r"copy.?move|clone", source, re.IGNORECASE) or any(re.match(r"^C\d+′?:", label) for label in labels):
        return "clones"
    if re.search(r"face", source, re.IGNORECASE) or (labels and all(re.match(r"^face\b", label, re.IGNORECASE) for label in labels)):
        return "faces"
    if re.match(r"^crops scored", str(exhibit.get("title") or ""), re.IGNORECASE) or (labels and all(re.search(r"\bcrop$", label, re.IGNORECASE) for label in labels)):
        return "crops"
    return "regions"


def region_label(lens: str, region: dict[str, Any], index: int) -> str:
    """Face A, B … for faces, C1 / C1′ for clone pairs, C-n for classifier crops (numbered across crop exhibits by the caller), else
    the region's display id or R-n (the viewer's `regionItems`). N-19q (N-20b's read of the reports, finding 6): a video's
    per-frame face box is named by its frame and box ("Frame 35, box 1"), not as a face of its own: one person's boxes were
    "Face A" … "Face J1" beside the one track "Face A", which names the face (the page shows a video's faces by their tracks
    only, so the image viewer's rule is unchanged)."""
    if lens == "faces" and (region.get("region") or {}).get("frame_pts") is not None:
        frame = _FRAME_BOX.match(str(region.get("label") or ""))
        return f"Frame {frame.group(1)}, box {frame.group(2)}" if frame else f"Box {index}"
    if lens == "faces":
        n = index - 1
        return "Face " + chr(65 + n % 26) + (str(n // 26) if n >= 26 else "")
    if lens == "crops":
        return f"C-{index}"
    clone = _CLONE_LABEL.match(str(region.get("label") or "")) if lens == "clones" else None
    if clone:
        return clone.group(1) + clone.group(2)
    return str(region.get("display_id") or f"R{index}")


def live_annotations(snapshot: ReportSnapshot) -> list[Any]:
    """The annotations appendix C lists and numbers N1, N2 …: those with a body or a shape."""
    return [a for a in snapshot.annotations if a.body or a.shape != "none"]


def overlay_marks(snapshot: ReportSnapshot, heatmap: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    """Fig. 4 marks in original pixels, (label, bbox, kind): the map's localisation regions, joined by source signal or the legacy localise title and named as the region table
    names them, and the in-report box annotations on the map, numbered as appendix C numbers them. A region box is mapped from
    its exhibit's coordinate space as the region table maps it (`coords.box_to_original`; N-4); a note's box is stored in
    original pixels by the viewer."""
    marks: list[tuple[str, dict[str, Any], str]] = []
    source = _source_size(snapshot.result.get("condition_profile") or {})
    for e in snapshot.result.get("exhibits") or []:
        if _regions_of_map(heatmap, e):
            for i, r in enumerate(e.get("regions") or [], 1):
                bbox = (r.get("region") or {}).get("bbox")
                if bbox:
                    marks.append((region_label("regions", r, i), coords.box_to_original(bbox, e.get("coordinate_space"), source), "region"))
    for n, a in enumerate(live_annotations(snapshot), 1):
        g = a.geometry or {}
        on_map = a.target_type in ("exhibit", "region") and a.target_id == heatmap.get("id")
        on_image = a.target_type == "region" and a.target_id is None
        if (a.visibility == "in_report" and (on_map or on_image) and a.shape == "box" and g.get("frame_pts") is None
                and all(k in g for k in ("x", "y", "w", "h"))):
            marks.append((f"N{n}", g, "note"))
    return marks


def _source_size(profile: dict[str, Any]) -> tuple[Any, Any] | None:
    """The image size the condition profile records, or None. `coords` reads it per exhibit: a profile from before N-4 holds
    the stored size of a rotated photo, which the exhibit's space shows to be transposed."""
    return (profile["width"], profile["height"]) if profile.get("width") and profile.get("height") else None


# N-19c (design review, P2 "PDFs"): the spectrogram of a silent recording is as bright as that of a loud one, and said nothing of it
# N-19q (finding 10): and what the figure of a recording of silence shows (the caption said that it is drawn as bright as a loud
# one over a black box)
SPECTROGRAM_SCALE = ("Brightness is relative to the loudest point of this recording, not to a fixed level: a quiet recording is "
                     "drawn as bright as a loud one, and a recording of silence is drawn black.")
INSPECTION_CAPTION = "Not assessable: for inspection only."  # N-17 (D-68): under every per-window figure of such a result


# N-19k (the PDF words of both reviews): the step of the run that drew a figure, by the name the analysis record gives it
# (`ProgressRail.tsx STAGE_LABEL`); it read "From temporal", "From the render stage"
STEP_WORDS = {"ingest": "Intake & hashing", "probe": "Container probe", "provenance": "Provenance & metadata", "decode": "Decode",
              "shots": "Shot boundaries", "tracks": "Face tracks", "faces": "Faces", "condition": "Recording conditions",
              "detect": "Detectors", "temporal": "Temporal checks", "second_pass": "Second pass", "localise": "Localisation",
              "fuse": "Fusion & calibration", "synthesise": "Findings", "render": "Figures", "publish": "Publish result"}
_BUILD_NAME = re.compile(r"\(legacy fuser\)|\bfrom the legacy fuser\b")


def creator_words(created_by: str | None, names: dict[str, str]) -> str:
    """Who drew a figure, in words: a step of the run by its name in the analysis record ("the step “Temporal checks”"), a
    check by the name the ledger gives it; never a stage name or a detector id."""
    creator = str(created_by or "render")
    if creator in STEP_WORDS:
        return f"the step “{STEP_WORDS[creator]}”"
    return names.get(creator) or UNNAMED_CHECK


def without_build_name(text: str) -> str:
    """A stored title or caption without the build's name for the fallback's combination: "Sentence scores (legacy fuser)"
    reads "Sentence scores (hand-set weights)" (results made since N-19k are stored so)."""
    said = _BUILD_NAME.sub(lambda m: "(hand-set weights)" if m.group(0).startswith("(") else "from hand-set weights", text)
    for old, new in formatting.EARLIER_SERIES_WORDS:  # N-19q (finding 6): a stored title of the earlier method's series
        said = said.replace(old, new)
    return said


def check_names(snapshot: ReportSnapshot, signals: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Each check by the name the ledger gives it: its signals' name where they share one, else the snapshot's."""
    by_detector: dict[str, set[str]] = {}
    for s in signals.values():
        if s.get("detector_id") and s.get("name"):
            by_detector.setdefault(str(s["detector_id"]), set()).add(str(s["name"]))
    return {**dict(snapshot.detector_names), **{d: next(iter(n)) for d, n in by_detector.items() if len(n) == 1}}


def exhibit_rows(snapshot: ReportSnapshot, images: dict[str, str], signals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result, kind = snapshot.result, snapshot.analysis.kind
    names = check_names(snapshot, signals)
    inspection = view.not_assessable(result.get("decision"))
    profile = result.get("condition_profile") or {}
    recorded = _source_size(profile)
    findings = result.get("findings") or []
    out, crop_start, crops = [], {}, 0
    for e in result.get("exhibits") or []:  # crops are numbered in stored order, as the viewer's catalogue numbers them
        if e["kind"] == "regions" and regions_lens(e) == "crops":
            crop_start[str(e["id"])], crops = crops, crops + len(e.get("regions") or [])
    stills: list[dict[str, Any]] = []
    sheet_at: int | None = None
    for e in print_order(list(result.get("exhibits") or [])):
        eid = str(e["id"])
        if e["kind"] == "image" and e.get("role") == "still":  # L-4: sampled video frames go on one contact sheet
            if sheet_at is None:
                sheet_at = len(out)  # N-19c: the sheet stands where the first frame is stored, as one figure
            stills.append(e)
            continue
        via = set(e.get("source_signal_ids") or []) | {s["id"] for s in signals.values() if eid in (s.get("exhibit_ids") or [])}
        cited_by = [f["id"] for f in findings
                    if eid in (f.get("evidence_ids") or []) or via & set(f.get("evidence_ids") or [])
                    or any(eid in (s.get("evidence_ids") or []) or via & set(s.get("evidence_ids") or []) for s in f.get("sentences") or [])]
        title, caption = without_build_name(e.get("title", "")), without_build_name(prose.system_prose(e.get("caption", "")))
        if inspection:  # N-17 (D-68): a per-window figure is a "window score", captioned as for inspection only
            title = title.replace("window probability", "window score")
            caption = " ".join(x for x in (INSPECTION_CAPTION if e["kind"] in ("series", "segments") else "", view.without_estimate(caption)) if x)
        row: dict[str, Any] = {"anchor": anchor("ex", eid), "ids": [eid], "label": "", "title": title,
                               "caption": caption, "kind": e["kind"], "created_by": e.get("created_by", ""),
                               "creator": creator_words(e.get("created_by"), names),  # N-19k: what the figure prints after "From"
                               "sources": sorted(signals[i].get("display_id") or signals[i]["name"] for i in via if i in signals),
                               "findings": cited_by, "plates": [], "legend": None, "colour_bar": None, "scale_labels": None, "reading": None,
                               "table": None, "series": None, "track": None, "stills": [], "note": None, "sha256": e.get("sha256"),
                               "contact": None, "picture": None}
        if e["kind"] == "heatmap":
            semantics = e.get("semantics", "p_manipulated")
            plates = []
            if kind == "image":
                plates.append({"title": "Source image", "uri": images.get("source")})
            plates.append({"title": HEATMAP_TITLE.get(semantics, "Map"), "uri": images.get(eid)})
            if e.get("reliability_mask_artefact_id"):
                plates.append({"title": "Reliability (hatched: unreliable)", "uri": images.get(f"{eid}#reliability")})
            if kind == "image":
                marks = overlay_marks(snapshot, e)
                image_size = coords.oriented_source(e.get("coordinate_space"), recorded)  # the oriented image the overlay shows
                plates.append({"title": "Overlay with regions and annotations (maximum opacity 70 %)", "uri": images.get(f"{eid}#overlay"),
                               "unavailable": "not drawn, because it needs verified scalar data and whole-image alignment.",
                               "marks": figures.marks_svg(image_size[0], image_size[1], marks) if marks and image_size else None})
            lo, hi = e.get("value_range") or (0.0, 1.0)
            row.update(plates=plates, legend=LEGEND.get(semantics, LOCALISATION_LEGEND), colour_bar=figures.heat_legend_svg(),
                       scale_labels=(f"{lo:.3g}", f"{(lo + hi) / 2:.3g}", f"{hi:.3g}"),  # DR P1-8: no "4.16667"
                       reading=("Values are scores." if inspection else "Values are calibrated probabilities." if e.get("calibrated")
                                else "Values are scores, not probabilities.")
                       + f" Map grid {e['grid'][0]} × {e['grid'][1]}, 8-bit. "
                       "Overlay: normalized values at least 0.5, up to 70 % opacity, scaled by reliability when recorded. "
                       "These are display settings, not decision thresholds. Map only: full range on black, reliability applied.")
        elif is_spectrogram(e):
            # L-7: over the time axis of the recording, with its frequency and time ticks; the seconds come from the waveform exhibit
            seconds = next((w[1] for w in map(_waveform, result.get("exhibits") or []) if w is not None), None)
            row["picture"] = {"uri": images.get(eid), "alt": str(e.get("title", "Spectrogram")), "y_ticks": ["8 kHz", "4 kHz", "0"],
                              "x_ticks": _time_ticks(seconds) if seconds else [], "x_label": X_UNITS["s"]}
            row["note"] = SPECTROGRAM_SCALE
        elif e["kind"] == "image":
            row["plates"] = [{"title": str(e.get("role", "image")).replace("_", " ").capitalize(), "uri": images.get(eid)}]
        elif e["kind"] == "regions":
            cells, lens = [], regions_lens(e)
            for i, r in enumerate(e.get("regions") or [], 1):
                region = r.get("region") or {}
                bbox = region.get("bbox")
                # L-4: map-grid or analysis-copy geometry → original px per axis, the viewer's rule (`app/core/coords.py`)
                space, source = e.get("coordinate_space"), recorded
                where = _box_text(coords.box_to_original(bbox, space, source)) if bbox else (
                    f"polygon of {len(region.get('polygon') or [])} points" if region.get("polygon") else "mask")
                share_of = coords.area_share(bbox, space, source) if bbox else None
                share = f"{share_of * 100:.1f} %" if share_of is not None else NOT_RECORDED
                value = view.numeral(r["probability"]) if r.get("probability") is not None else (f"{r['score']:.2f}" if r.get("score") is not None else "–")
                cells.append([region_label(lens, r, crop_start.get(eid, 0) + i), where, share, value, r.get("label") or "–"])
            row["table"] = {"columns": ["Region", "Where (original pixels)", "Area", "Score" if inspection else value_column(e), "Label"],
                            "names": [], "rows": cells[:TABLE_ROWS],
                            "more": max(0, len(cells) - TABLE_ROWS)}
        elif e["kind"] == "table":
            cells = [[("–" if c is None else f"{c:g}" if isinstance(c, float) else str(c)) for c in r] for r in e.get("rows") or []]
            if e.get("created_by") == "img.meta.qtables":
                for cell in cells:
                    if len(cell) == 2 and cell[0] == "outcome":
                        cell[1] = prose.quantisation_outcome(cell[1])
            if e.get("created_by") == "img.meta.consistency" and e.get("columns") == ["group", "tag", "value", "severity", "rule", "note"]:
                # Only the generated rule column is presentation text. Captured values stay verbatim.
                for cell in cells:
                    if len(cell) == 6:
                        cell[4] = prose.metadata_rule(cell[4])
            lowered = [str(c).lower() for c in e.get("columns") or []]
            if "tag" in lowered:  # N-19k: a tag without a name is an unnamed tag with its number
                at = lowered.index("tag")
                for cell in cells:
                    if at < len(cell):
                        cell[at] = formatting.tag_words(cell[at])
            row["table"] = {"columns": [COLUMN_WORDS.get(str(c), str(c).replace("_", " ")) for c in e.get("columns") or []],
                            # N-19k: the columns that hold a name (a tag, a field), which is never broken over lines
                            "names": [i for i, c in enumerate(e.get("columns") or []) if str(c).lower() in NAME_COLUMNS],
                            "rows": cells[:TABLE_ROWS], "more": max(0, len(cells) - TABLE_ROWS)}
        elif e["kind"] == "text_spans":
            # Document spans retain legacy diagnostic scores under the schema's
            # probability field; whole-document calibration does not calibrate them.
            document_scores = kind == "document"
            cells = []
            for sp in e.get("spans") or []:
                score = sp.get("probability")
                value = "–" if score is None else f"{score:.2f}" if document_scores else view.numeral(score)
                cells.append([str(sp["sentence_idx"] + 1), f"{sp['start']}–{sp['end']}", value,
                              sp.get("label") or "–", "; ".join(sp.get("reasons") or []) or "–"])
            row["table"] = {"columns": ["Sentence", "Characters",
                                        "Uncalibrated score" if document_scores else "Score" if inspection else "Probability", "Label", "Reasons"],
                            "names": [], "rows": cells[:TABLE_ROWS], "more": max(0, len(cells) - TABLE_ROWS)}
            if document_scores:
                row["note"] = ("Sentence scores are uncalibrated diagnostic values." if inspection
                               else "Sentence scores are uncalibrated diagnostic values, not probabilities of authorship.")
        elif e["kind"] == "series":
            row["series"] = series_view(e)
        elif e["kind"] == "face_track":
            track = e.get("track") or {}
            frames = e.get("frames") or []
            row["track"] = {"label": f"Face {track['identity']}" if track.get("identity") else str(track.get("track_id")),
                            "frames": track.get("n_frames", len(frames)),
                            "range": f"frames {frames[0]['frame_index']}–{frames[-1]['frame_index']}" if frames else NOT_RECORDED}
            row["stills"] = [uri for key, uri in sorted(images.items()) if key.startswith(f"{eid}#still-")]
        elif e["kind"] == "segments":
            # N-19n (D3-08): of a lane whose check is not counted the line says what the section says ("1 segment, listed under
            # Temporal analysis" stood where that section says "Not listed: 1 segment on the lane …")
            mine = list(e.get("segments") or [])
            left = sum(segment_uncounted(result, signals, _lane_key(seg.get("lane") or e.get("lane"), seg.get("hypothesis")), seg) for seg in mine)
            listed = len(mine) - left
            row["note"] = (f"{_plural(len(mine), 'segment')}, listed under Temporal analysis." if not left
                           else f"{_plural(left, 'segment')}, {UNCOUNTED_LANE}." if not listed
                           else f"{_plural(listed, 'segment')} listed under Temporal analysis; {_plural(left, 'segment')} {UNCOUNTED_LANE}.")
        out.append(row)
    if stills and sheet_at is not None:
        out.insert(sheet_at, contact_sheet(stills, images, names))
    number_exhibits(out)
    return out


_FRAME_NUMBER = re.compile(r"Sampled frame (\d+)")


def contact_sheet(stills: list[dict[str, Any]], images: dict[str, str], names: dict[str, str] | None = None) -> dict[str, Any]:
    """L-4 (design P1-5): the sampled frames as one contact sheet (four across, time order, timecode and sample number) instead of
    one half-page figure per frame. N-19c: one figure with one number (its label read "Fig. 35–13")."""
    ordered = sorted(stills, key=lambda e: (float(e.get("frame_pts") or 0), str(e["id"])))
    cells = []
    for e in ordered:
        m = _FRAME_NUMBER.search(str(e.get("title", "")))
        pts = e.get("frame_pts")
        cells.append({"uri": images.get(str(e["id"])), "alt": str(e.get("title", "Sampled frame")), "sha256": e.get("sha256"),
                      "label": " · ".join(x for x in (formatting.mmss(float(pts) / 1000.0) if pts is not None else None,
                                                     # N-19k (design re-review): the number is the sample's place among
                                                     # the sampled frames, not a frame of the video (the viewer's "frame 24")
                                                     f"sample {m.group(1)}" if m else None) if x) or str(e.get("title", ""))})
    return {"anchor": anchor("ex", str(ordered[0]["id"])), "ids": [str(e["id"]) for e in ordered], "label": "",
            "title": f"Sampled frames ({len(cells)})",
            "caption": "The decoded frames the detectors scored, stored as JPEG, in time order, each with its payload SHA-256.",
            "kind": "contact_sheet",
            "created_by": " and ".join(sorted({str(e.get("created_by") or "render") for e in ordered})),
            "creator": " and ".join(sorted({creator_words(e.get("created_by"), names or {}) for e in ordered})), "sources": [], "findings": [], "plates": [], "legend": None, "colour_bar": None, "scale_labels": None,
            "reading": None, "table": None, "series": None, "track": None, "stills": [], "note": None, "sha256": None, "contact": cells,
            "picture": None}


SERIES_NAMES = {"offset_frames": ("Sync offset", "frames"), "p_window": ("Window score", "0–1"), "p_smoothed": ("Smoothed window score", "0–1"),
                "p_frame": ("Frame score", "0–1"), "p_face": ("Face score", "0–1"), "p_general": ("Whole-frame score", "0–1"),
                "p_synthetic": ("Synthetic-speech score", "0–1"),
                # L-7 waveform envelope: one quantity, so the three lines share one labelled strip
                "level_low": ("Lowest sample", "share of full scale"), "level_high": ("highest sample", "share of full scale"),
                "level_rms": ("RMS level", "share of full scale")}
X_UNITS = {"s": "time (s)", "pts": "presentation time (pts)", "sentence": "sentence", "frequency": "frequency"}


def _tick(v: float) -> str:
    return f"{v:.3g}"


def nice_axis(lo: float, hi: float, steps: int) -> tuple[float, float]:
    """N-19c (design review, P2 "PDFs": ticks at 0.25, 2.38, 4.5, 6.62, 8.75): the smallest range that holds `lo` to `hi` in
    `steps` steps of a round size (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5 or 8 times a power of ten), from a multiple of the step, so
    that every tick is a round number and the data fills most of the axis."""
    if not hi > lo:
        return lo, hi
    magnitude = 10.0 ** math.floor(math.log10((hi - lo) / steps))
    for multiple in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0):
        step = multiple * magnitude
        start = math.floor(lo / step + 1e-9) * step
        if start + steps * step >= hi - 1e-9 * step:
            return start, start + steps * step
    return lo, hi


WAVEFORM_SERIES = ("level_low", "level_high", "level_rms")
# table columns a detector names by key, in words (L-7: the content table printed START_S and END_S)
COLUMN_WORDS = {"start_s": "start (s)", "end_s": "end (s)"}
NAME_COLUMNS = ("tag", "field", "group", "key")


def is_spectrogram(e: dict[str, Any]) -> bool:
    return e.get("kind") == "image" and e.get("role") == "spectrum" and e.get("media_ref") == "audio"


def _waveform(e: dict[str, Any]) -> tuple[list[tuple[float, float, float, float]], float] | None:
    """The bins (time, lowest, highest, RMS) of a waveform exhibit and the seconds they span, else None."""
    by_name = {str(s.get("name")): s.get("points") or [] for s in e.get("series") or []}
    if e.get("kind") != "series" or not all(by_name.get(name) for name in WAVEFORM_SERIES):
        return None
    low, high, rms = (by_name[name] for name in WAVEFORM_SERIES)
    bins = [(float(a[0]), float(a[1]), float(b[1]), float(c[1])) for a, b, c in zip(low, high, rms, strict=False)
            if a[1] is not None and b[1] is not None and c[1] is not None]
    if not bins:
        return None
    step = bins[1][0] - bins[0][0] if len(bins) > 1 else 2 * bins[0][0]
    return bins, bins[-1][0] + step / 2


def _time_ticks(seconds: float) -> list[str]:
    """The ticks of a strip that spans the recording exactly (a waveform, a spectrogram): at its quarters, to a tenth of a
    second (whole seconds from 100 s), never "2.38" (N-19c)."""
    return [f"{seconds * q:.0f}" if seconds >= 100 else f"{seconds * q:.1f}".removesuffix(".0") for q in (0, 0.25, 0.5, 0.75, 1)]


def waveform_view(e: dict[str, Any]) -> dict[str, Any] | None:
    """L-7: a waveform exhibit as one envelope strip from 0 s, symmetric about zero, with its peak as the labelled range."""
    made = _waveform(e)
    if made is None:
        return None
    bins, seconds = made
    extreme = max(max(abs(b[1]), abs(b[2])) for b in bins)
    peak = next((step for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if extreme <= step), 1.0)
    return {"strips": [{"svg": figures.waveform_svg(bins, (0.0, seconds), peak),
                        "label": "Lowest to highest sample (grey) and RMS level (dark), share of full scale",
                        "y_ticks": [f"+{peak:g}", "0", f"−{peak:g}"], "dashed": False}],
            "x_ticks": _time_ticks(seconds), "x_label": X_UNITS["s"], "caption": "", "stored": 0}


def series_view(e: dict[str, Any]) -> dict[str, Any]:
    """L-4 (design P1-4): one strip per quantity (probability-like 0–1 scores apart from offsets and other units), each with its
    own labelled y ticks and a human name; a shared x axis with ticks and its unit. No two quantities share an unlabelled range."""
    waveform = waveform_view(e)
    if waveform is not None:
        return waveform
    inline = [s for s in e.get("series") or [] if s.get("points")]
    thresholds = [float(v) for v in e.get("thresholds") or []]
    lines = [(s, [(float(x), float(y)) for x, y in s["points"] if y is not None]) for s in inline]
    xs = [x for _, pts in lines for x, _ in pts]
    x_range = nice_axis(min(xs), max(xs), 4) if xs else (0.0, 1.0)  # N-19c: round ticks; the lines keep their own x
    # N-19q (N-20b's read of the reports, finding 6): a series of one x (the GOP strip of a clip with one key frame) printed the
    # five ticks "0 0 0 0 0"; it is drawn in the middle of a unit range and its one value is the middle tick
    single = x_range[0] == x_range[1]
    if single:
        x_range = (x_range[0] - 0.5, x_range[0] + 0.5)
    groups: dict[str, list[tuple[dict[str, Any], list[tuple[float, float]]]]] = {}
    for s, pts in lines:
        unit = SERIES_NAMES.get(str(s.get("name")), ("", ""))[1]
        unit_key = unit if unit and unit != "0–1" else ("0–1" if pts and all(0 <= y <= 1 for _, y in pts) else str(s.get("name")))
        groups.setdefault(unit_key, []).append((s, pts))
    strips = []
    for unit_key, members in groups.items():
        ys = [y for _, pts in members for _, y in pts]
        probability = unit_key == "0–1"
        levels = [v for v in thresholds if (0 <= v <= 1) == probability]
        y_range = (0.0, 1.0) if probability else ((min(ys + levels), max(ys + levels)) if ys else (0.0, 1.0))
        if y_range[0] == y_range[1]:
            y_range = (y_range[0] - 1.0, y_range[1] + 1.0)
        if not probability:
            y_range = nice_axis(y_range[0], y_range[1], 2)
        names = [SERIES_NAMES.get(str(s.get("name")), (str(s.get("name", "")).replace("_", " "), ""))[0] for s, _ in members]
        strips.append({"svg": figures.series_svg([pts for _, pts in members], x_range, y_range, levels),
                       "label": ", ".join(dict.fromkeys(names)) + (" (uncalibrated, 0–1)" if probability else
                                                                   f" ({SERIES_NAMES.get(str(members[0][0].get('name')), ('', unit_key))[1]})"),
                       "y_ticks": [_tick(y_range[1]), _tick((y_range[0] + y_range[1]) / 2), _tick(y_range[0])],
                       "dashed": bool(levels)})
    span = x_range[1] - x_range[0]
    ticks = ["", "", _tick(x_range[0] + span / 2), "", ""] if single else [_tick(x_range[0] + span * q) for q in (0, 0.25, 0.5, 0.75, 1)]
    return {"strips": strips, "x_ticks": ticks,
            "x_label": X_UNITS.get(str(e.get("x_unit")), str(e.get("x_unit") or "")),
            "caption": "Dashed: threshold." if any(s["dashed"] for s in strips) else "", "stored": len(e.get("series") or []) - len(inline)}


UNCOUNTED_LANE = "not counted in this estimate: drawn for inspection only"


def segment_uncounted(result: dict[str, Any], signals: dict[str, dict[str, Any]], lane: str, seg: dict[str, Any]) -> bool:
    """N-19k (browser QA Q2-12; model.ts `segmentUncounted`): under a result with an estimate, a segment of a check that
    scored and is not counted in it (the fusion's record on its signal). Never the lane of a face track, as on the page.
    N-19n: one rule for the list of segments and for the line under the figure of the lanes."""
    decision = result.get("decision") or {}
    inspection = decision.get("code") == "not_assessable" or view.estimate_of(decision) is None
    sources = [i for i in seg.get("sources") or [] if i in signals]
    return (not inspection and lane not in ("face_swap", "partial_edit") and bool(sources)
            and all(signals[i].get("outside_estimate") for i in sources))


def uncounted_note(n: int, labels: list[str]) -> str | None:
    """N-19k (Q2-12; model.ts `uncountedNote`): what the list of segments says of the ones it leaves out."""
    if not n or not labels:
        return None
    one, single = n == 1, len(labels) == 1
    where = f"the lane “{labels[0]}”" if single else "the lanes " + ", ".join(f"“{x}”" for x in labels)
    return (f"Not listed: {_plural(n, 'segment')} on {where}. {'Its check is' if single else 'Their checks are'} not counted in this "
            f"estimate; the {'lane shows' if single else 'lanes show'} {'it' if one else 'them'} for inspection.")


def _lane_key(lane: str | None, hypothesis: str | None) -> str:
    return lane if lane in dict(LANES) else "synthesis" if hypothesis == H1 else "other"


def temporal(snapshot: ReportSnapshot, signals: dict[str, dict[str, Any]], images: dict[str, str],
             printed: dict[str, dict[str, str]] | None = None) -> dict[str, Any] | None:
    result, kind = snapshot.result, snapshot.analysis.kind
    if kind not in ("video", "audio"):
        return None
    seen: dict[tuple[Any, ...], tuple[str, dict[str, Any]]] = {}
    lane_p: dict[str, dict[str, Any]] = {}
    for a in result.get("assessments") or []:
        for la in a.get("lanes") or []:
            lane_p[la["lane"]] = la
            for seg in la.get("segments") or []:
                seen.setdefault((la["lane"], seg["start_pts"], seg["end_pts"], seg.get("track_id"), seg["channel"]), (la["lane"], seg))
    for s in signals.values():
        for seg in s.get("segments") or []:
            key = _lane_key(seg.get("lane") or s.get("lane"), seg.get("hypothesis"))
            seen.setdefault((key, seg["start_pts"], seg["end_pts"], seg.get("track_id"), seg["channel"]), (key, seg))
    for e in result.get("exhibits") or []:
        if e["kind"] == "segments":
            for seg in e.get("segments") or []:
                key = _lane_key(seg.get("lane") or e.get("lane"), seg.get("hypothesis"))
                seen.setdefault((key, seg["start_pts"], seg["end_pts"], seg.get("track_id"), seg["channel"]), (key, seg))
    faces = {e["track"]["track_id"]: e["track"].get("identity") for e in result.get("exhibits") or [] if e["kind"] == "face_track"}
    # the viewer's order and letters (`video/model.ts`): start, lane order, track; "Seg. a", "Seg. b" … unless a display id is stored
    segments = sorted(seen.values(), key=lambda x: (float(x[1]["start_s"]), SEGMENT_LANE_ORDER.get(x[0], len(SEGMENT_LANE_ORDER)),
                                                    str(x[1].get("track_id") or ""), int(x[1]["start_pts"])))
    letters = [s.get("display_id") or f"Seg. {_letter(n)}" for n, (_, s) in enumerate(segments)]
    # L-4 (design P0-2): a result that could not be assessed, or published no estimate, draws tints only, without the flag rule or peaks
    decision = result.get("decision") or {}
    inspection = decision.get("code") == "not_assessable" or view.estimate_of(decision) is None
    profile = result.get("condition_profile") or {}
    duration = float(profile.get("duration_s") or max((float(s["end_s"]) for _, s in segments), default=0.0))

    # N-19k (browser QA Q2-12; model.ts `segmentUncounted`): under a result with an estimate, the segments of a check that
    # scored and is not counted in it (the fusion's record on its signal). Their lane is drawn as an inspection lane is, in
    # tints without the flag rule, and they are in no list of segments: "Voice lane, whole video: … 1 segment" stood under
    # the figure of a result that counted 0 flags. Never the lane of a face track, as on the page
    left = [segment_uncounted(result, signals, lane, s) for lane, s in segments]
    lanes, left_lanes = [], []
    for key, label in LANES:
        mine = [(s, out) for (lane, s), out in zip(segments, left, strict=True) if lane == key]
        if not mine and key not in lane_p:
            continue
        la = lane_p.get(key)
        aside = bool(mine) and all(out for _, out in mine)
        steps = [(float(s["start_s"]), float(s["end_s"]), min(3, view.BAND_STEP[band_for(float(s["peak"]))]) if inspection or aside
                  else view.BAND_STEP[band_for(float(s["peak"]))]) for s, _ in mine]
        if inspection:
            estimate = f"{_plural(len(mine), 'segment')}, values shown for inspection only"
        elif aside:
            estimate = f"{_plural(len(mine), 'segment')}, {UNCOUNTED_LANE}"
            left_lanes.append(label)
        else:
            # N-19q (finding 10): in the report's words (it read "lane estimate 23 (calibrated sub-fuser; likely authentic)",
            # in a report that says it does not confirm that a recording is authentic)
            estimate = (f"its own calibrated estimate {view.numeral(la['probability'])} out of 100 ({LANE_BAND_WORDS.get(str(la.get('band')), 'level not recorded')})"
                        if la and la.get("probability") is not None else _plural(len(mine), "segment"))
        lanes.append({"label": label, "estimate": estimate,
                      "svg": figures.lane_svg(steps, duration)})
    rows = [{"id": letters[n], "lane": dict(LANES)[lane], "channel": s["channel"],
             "track": (f"Face {faces[s['track_id']]}" if faces.get(s.get("track_id")) else s.get("track_id")) or "–",
             "time": f"{formatting.mmss(s['start_s'])}–{formatting.mmss(s['end_s'])}",
             "peak": "–" if inspection else view.numeral(float(s["peak"])),
             "sources": ", ".join(signals[i].get("display_id") or i for i in s.get("sources") or [] if i in signals) or "–"}
            for n, (lane, s) in enumerate(segments) if not left[n]]
    tracks = [{"label": f"Face {e['track']['identity']}" if e["track"].get("identity") else e["track"]["track_id"], "anchor": anchor("ex", e["id"]),
               "fig": _track_reference((printed or {}).get(str(e["id"])), e), "frames": e["track"].get("n_frames", 0),
               "stills": [uri for key, uri in sorted(images.items()) if key.startswith(f"{e['id']}#still-")]}
              for e in result.get("exhibits") or [] if e["kind"] == "face_track"]
    return {"lanes": lanes, "segments": rows, "tracks": tracks, "duration": formatting.mmss(duration), "inspection": inspection,
            "uncounted": uncounted_note(sum(left), left_lanes),
            "inspection_caption": ("Not assessable" if decision.get("code") == "not_assessable" else "No estimate")
            + ": values shown for inspection only. Segments are drawn as tints, without a flag rule or peak values.",
            "ticks": [formatting.mmss(0), formatting.mmss(duration / 2), formatting.mmss(duration)]}


def watermark_words(record: dict[str, Any]) -> str:
    """N-19k (the PDF words of both reviews: "present False; scheme dwtDct"): one reading of a public watermark message, as
    a sentence: whose message it is, whether it was found, and the count of bits behind that. The codec's name and the
    record's keys are not printed; a record of another shape says only what it holds, a truth value as yes or no."""
    name = record.get("name") or "a generator that the record does not name"
    equal, bits, threshold = (record.get(k) for k in ("equal_bits", "bits", "threshold"))
    if isinstance(record.get("present"), bool) and all(isinstance(n, int) and not isinstance(n, bool) for n in (equal, bits)):
        needs = f"; at least {threshold} are needed" if isinstance(threshold, int) and not isinstance(threshold, bool) else ""
        return (f"the public watermark message of {name} was {'found' if record['present'] else 'not found'} "
                f"({equal} of {bits} bits match{needs}).")
    said = [f"{k.replace('_', ' ')}: {('yes' if v else 'no') if isinstance(v, bool) else v}" for k, v in sorted(record.items())
            if k not in ("scheme", "message") and isinstance(v, str | int | float | bool)]
    return "; ".join(said) + "." if said else "a reading without recorded values."


def provenance(snapshot: ReportSnapshot, signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = snapshot.result
    h3 = next((a for a in result.get("assessments") or [] if a.get("hypothesis") == H3), None)
    state = (h3 or {}).get("c2pa_state")
    credentials = C2PA_WORDS.get(state, "Content Credentials were not checked for this analysis.") if state else \
        "Content Credentials were not checked for this analysis."
    watermarks = [watermark_words(w) for w in (h3 or {}).get("watermarks") or []]
    rows = []
    for f in sorted((f for f in result.get("findings") or [] if f.get("family") in PROVENANCE_FAMILIES), key=lambda f: int(f.get("rank", 0))):
        direction = view.finding_direction(f)
        marker = view.record_word(f, view.decision_of(result))  # N-19n (D3-07): the column "Strength" says "self-declared"
        words = "documentary" if view.is_documentary(f) else marker or view.STRENGTH_WORDS.get(str(f.get("strength")), "little effect")
        if not view.is_documentary(f) and marker is None and f.get("strength") != "none" and direction in ("for", "against"):
            words = f"{words} {direction}"
        statement = f.get("statement", "")
        if f.get("family") not in {"F1_provenance", "F2_self_declared_metadata"} and not view.is_documentary(f):
            statement = prose.system_prose(statement)
        aside = view.set_aside(f, view.decision_of(result))  # N-19k (Q2-14; D-64), as in the findings chapter
        rows.append({"id": f["id"], "href": "#" + anchor("fnd", f["id"]), "words": view.SET_ASIDE_WORDS if aside else words,
                     "title": view.SET_ASIDE_TITLE if aside else f.get("title", ""), "statement": statement})
    metadata = [{"id": s.get("display_id") or "–", "href": "#" + anchor("sig", s["id"]), "name": s.get("name", ""), "observed": view.signal_observed(s) or "–",
                 "reference": s.get("reference") or "–"}
                for s in sorted(signals.values(), key=lambda s: (s.get("display_id") or "~", s["id"])) if s.get("category") in ("metadata", "provenance")]
    profile = result.get("condition_profile") or {}
    compression = []
    if profile.get("jpeg_quality_est") is not None:
        compression.append(f"Estimated JPEG quality of the current file: {profile['jpeg_quality_est']}.")
    if profile.get("double_compression") is not None:
        compression.append("Saved at least twice (double compression detected)." if profile["double_compression"]
                           else "No double compression was detected.")
    # N-19n (the reads of the first part): where the step that writes the findings failed, the section says that, and not
    # that no finding was recorded, which describes the evidence
    none = (effect.findings_not_written("ledger") if effect.findings_failed(snapshot.result.get("caveats"))
            else "No metadata or provenance finding was recorded.")
    return {"credentials": credentials, "state": state, "watermarks": watermarks, "findings": rows, "findings_none": none, "metadata": metadata,
            "compression": compression or ["No compression history was recorded for this file."]}


# D-63 "Headline rule in words": the headline follows the prediction sets at the two calibrated levels, and no table of fixed
# ranges is printed (browser QA P1-2: 68 of 91 calibrated results contradicted the table). N-19m (D-77; D-76 as amended): a
# headline of one answer has two further conditions, one of them a part of the scale, so no report says any longer that
# the headline follows "not fixed ranges of the estimate"
HEADLINE_RULE = (
    "The headline follows two calibrated levels, and a headline of one answer must meet two further conditions.",
    "At each level Chipmunk Verify forms the set of answers (AI, not AI) that the estimate is compatible with for inputs of this context. "
    "The moderate level is 5 % and the strong level 1 % by default; a workspace policy can set both to the same value.",
    "A set of one answer at the moderate level gives “{likely_ai}” or “{likely_authentic}”. When the set at the strong level "
    "holds that one answer too and at least two checks scored, the headline is “{very_likely_ai}” or “{very_likely_authentic}”: "
    "one check alone does not give either.",
    "A set at the moderate level that holds both answers, or neither, gives “{uncertain}”. A cap (an input outside what was "
    "validated, a check that did not run) and documentary evidence act after this rule.",
    # N-19m (D-77): the estimate in the answer's part of the scale, read on the number as a result prints it
    "A headline of one answer also needs the estimate itself in that answer’s part of the scale: 34 or less for no indications, "
    "65 or more for indications.",
    # N-19i (D-76): a check that supports it; N-19m (D-76 as amended): and the checks together
    "It also needs one check that gives at least weak support for that answer on its own (a factor of two on the odds), and "
    "the checks together must give it at least weak support. Where either condition is not met the headline is “{uncertain}”.",
    "So one estimate can carry different headlines for different inputs, and the marker on the scale can stand in a segment "
    "that the headline does not name.",
)
FALLBACK_HEADLINE_RULE = (
    "This result comes from the uncalibrated fallback, which has no calibrated levels.",
    "Its headline follows fixed ranges of the score: “{likely_ai}” from 65, “{uncertain}” from 35 to 64 and “{likely_authentic}” "
    "below 35. It never issues “{very_likely_ai}” or “{very_likely_authentic}”.",
    "A cap, such as an input outside what was validated, makes the headline “{uncertain}”. A result in which no check of the "
    "content scored (metadata or provenance records only) is not assessable. A provenance record (a trusted declaration, a "
    "trusted capture record, a self-declared marker) acts after this rule.",  # N-19h (E-20, 4): as the calibrated rule says
)
# N-19o (design review D5-01): under a record that decides on the fallback the record sets the headline, and the ranges are
# those of the score; the rule above said that the fallback "never issues" the strongest headline beside an examiner whose
# conclusion on that headline is on the record's step
FALLBACK_RECORD_RULE = (
    "This result comes from the uncalibrated fallback, which has no calibrated levels, and a provenance record decides it.",
    "The record sets the headline. The ranges are those of the score: “{likely_ai}” from 65, “{uncertain}” from 35 to 64 and "
    "“{likely_authentic}” below 35; the score alone gives neither “{very_likely_ai}” nor “{very_likely_authentic}”.",
)
# N-19i (D-76): the first sentence of the rule under a result whose estimate reached a level and that no check supports
# N-19q (finding 1): the rule reads each check's part in the estimate (it read "no check gives at least weak support")
UNSUPPORTED_RULE = ("The headline of this result is “{uncertain}”: its estimate reaches a calibrated level, and no check's part "
                    "in that estimate gives at least weak support for the answer of that level. The rule in full:")
# N-19m: the first sentence under a result that D-77 took, and under one that the second condition of D-76 took
OFF_SCALE_RULE = ("The headline of this result is “{uncertain}”: its estimate reaches a calibrated level and stands in the middle "
                  "part of the scale (35 to 64). The rule in full:")
# N-19n (supervisor check 29): under a configured prior the published estimate of such a result can stand in the part of
# the other answer; the sentence names the answer of the levels and that part
OFF_SCALE_OTHER_RULE = ("The headline of this result is “{uncertain}”: the calibrated levels give {answer}, and its estimate stands "
                        "in {part}. The rule in full:")
TOGETHER_RULE = ("The headline of this result is “{uncertain}”: its estimate reaches a calibrated level, and the checks together "
                 "do not give at least weak support for the answer of that level. The rule in full:")
RATES_NOT_ON_RECORD = "The measured rates of the two levels are not on record for this fusion model."
LEVEL_WORDS = {"0.05": "5 %", "0.01": "1 %"}


def _share(rate: float, n: int) -> str:
    """A measured share with its count: "7.6 % (73 of 958)"; the count is left out when it is not a whole number of items."""
    k = rate * n
    count = f" ({round(k)} of {n:,})" if abs(k - round(k)) < 1e-6 else f" of {n:,}"
    return f"{rate * 100:.1f} %{count}"


def headline_rule(kind: str, decision: dict[str, Any]) -> dict[str, Any]:
    """Methods, "Headline rule": the rule in words, and for a result of the served fusion model what each level measured on
    the test split (`registry.release_level_rates`: the artefact's own record, found by the result's fuser sha256)."""
    _, many = view.media(kind)
    if decision.get("fusion_mode") != "artefact":
        rule = FALLBACK_RECORD_RULE if view.record_of(decision) is not None else FALLBACK_HEADLINE_RULE
        return {"sentences": [s.format(**view.HEADLINE) for s in rule], "rates": [], "scope": None, "note": None}
    sentences = [s.format(**view.HEADLINE) for s in HEADLINE_RULE]
    if (rule := view.rule_taken(decision)) is not None:  # N-19i (D-76), N-19m (D-77): a condition took the levels' answer
        first = (OFF_SCALE_RULE if view.off_scale(decision) is not None
                 else TOGETHER_RULE if rule in view.TOGETHER_RULES else UNSUPPORTED_RULE)
        words = dict(view.HEADLINE)
        if (part := view.off_scale_part(decision)) is not None:  # N-19n: from where the published estimate prints
            first = OFF_SCALE_OTHER_RULE
            words.update(answer=view.OFF_SCALE_ANSWER[str(view.off_scale(decision))], part=view.ANSWERS_PART[part])
        sentences = [first.format(**words), *sentences[1:]]
    measured = fusion_registry.release_level_rates(kind, decision.get("fuser_sha256"))
    if measured is None:
        return {"sentences": sentences, "rates": [], "scope": None, "note": RATES_NOT_ON_RECORD}
    rows = []
    for alpha, level in measured["levels"].items():
        if level["threshold"] is None or level["fpr"] is None:
            rows.append({"level": LEVEL_WORDS[alpha], "false_alarms": "no threshold exists at this level for this model", "detections": "–",
                         "note": "The two strongest headlines are not issued."})
            continue
        rows.append({"level": LEVEL_WORDS[alpha], "false_alarms": _share(float(level["fpr"]), measured["n_real"]),
                     "detections": _share(float(level["tpr"]), measured["n_fake"]) if level["tpr"] is not None and measured["n_fake"] else "–",
                     "note": f"target {LEVEL_WORDS[alpha]}"})
    sized = ", size-matched view" if measured["view"] == "size_matched" else ""
    scope = (f"Test split of {measured['bench_version']}{sized}: authentic and AI-generated or manipulated {many} that scored above the "
             "threshold of each level. The thresholds are those of the model as a whole, set on the calibration split; the headline "
             "uses the threshold of the input’s context.")
    return {"sentences": sentences, "rates": rows, "scope": scope, "note": None}


def _direction_word(s: dict[str, Any], little: float = effect.LN2, decision: dict[str, Any] | None = None) -> str:
    if view.is_mention(s):  # N-19o (Q5-01): a mention is no marker and points neither way
        return view.MENTION_DIRECTION
    if view.row_not_counted(s.get("id"), decision):  # N-19p (R10-05): it read "little effect" beside a row not counted
        return "neutral"
    override = s.get("provenance_override")
    if override == "decisive_ai":
        return "documentary"
    if override == "self_declared_ai":
        return "toward indications (declared)"
    if override == "trusted_capture":
        return "toward no indications (capture record)"
    c = s.get("contribution_logit")
    if c is not None:
        return "toward indications" if c >= little else "toward no indications" if c <= -little else "little effect"
    if s.get("log_lr") is not None and lr.band(float(s["log_lr"])) != "none":
        return "toward indications" if float(s["log_lr"]) > 0 else "toward no indications"
    legacy = (s.get("legacy") or {}).get("direction")
    return {"supports_ai": "toward indications", "supports_authentic": "toward no indications"}.get(legacy, "neutral")


NOT_IN_ESTIMATE = "not in this estimate"  # D-63: a check outside the estimate carries its stand-alone word and this


def _strength_word(s: dict[str, Any], findings: list[dict[str, Any]], decision: dict[str, Any] | None = None) -> str:
    """L-4 (design P1-7): one strength per signal on every surface. D-63: in a result measured from neutral points, a check that
    entered the estimate takes the strength of its contribution; a scored check outside it keeps its stand-alone word and says
    "not in this estimate". Otherwise a signal with its own finding takes that finding's strength (family caps, corroboration)
    in the findings' words, else its likelihood-ratio band (the UI ledger's `signalStrength`)."""
    measured, c = view.measured_from_neutral(decision), s.get("contribution_logit")
    if s.get("provenance_override") == "decisive_ai":
        return "documentary"
    if view.row_not_counted(s.get("id"), decision):  # N-19p (R10-05): it read "little effect" beside a row not counted
        return view.NOT_COUNTED
    if measured and effect.entered(c):
        return view.STRENGTH_WORDS[effect.strength(float(c))]
    own = view.own_finding(s["id"], findings)
    if own is not None and (marker := view.record_word(own, decision)) is not None:  # N-19n (D3-07): a marker has no strength
        return marker
    level = (view.finding_strength(own) if own is not None
             else lr.band(float(s["log_lr"])) if s.get("log_lr") is not None else (s.get("legacy") or {}).get("strength"))
    if level == "documentary":
        return "documentary"
    word = view.STRENGTH_WORDS.get(str(level), NOT_RECORDED)
    outside = measured and s.get("hypothesis") and s.get("provenance_override") is None and standalone_p(s) is not None
    return f"{word}, {NOT_IN_ESTIMATE}" if outside else word


def detectors(snapshot: ReportSnapshot, signals: dict[str, dict[str, Any]], marks: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    result, kind = snapshot.result, snapshot.analysis.kind
    _, many = view.media(kind)
    # N-19c (design review, P2 "PDFs": the rows ran S04, S05, S01, S02, S03): in the order of the S-ids, as the references read
    def s_id(s: dict[str, Any]) -> tuple[int, str]:
        m = re.search(r"\d+", str(s.get("display_id") or ""))
        return (int(m.group()) if m else 10**9, str(s["id"]))

    ordered = sorted(signals.values(), key=s_id)
    findings = [f for f in result.get("findings") or [] if isinstance(f, dict)]
    decision = result.get("decision") or {}
    little = view.little_of(decision)
    ledger = [{"id": s.get("display_id") or "–", "anchor": anchor("sig", s["id"]), "name": s.get("name", ""), "family": s.get("family", ""),
               "observed": view.signal_observed(s) or "–", "reference": s.get("reference") or "–", "direction": _direction_word(s, little, decision),
               "strength": _strength_word(s, findings, decision),
               "contribution": view.NOT_COUNTED if view.row_not_counted(s["id"], decision) and s.get("contribution_logit") is not None
               else signed(float(s["contribution_logit"])) if s.get("contribution_logit") is not None else "–",
               # L-7 (design P2): two decimals, as the findings and the screen ledger print a signal's probability ("0.90", not "90")
               "p": f"{p_ai(s):.2f}" if p_ai(s) is not None else "–", "disposition": marks.get(("signal", s["id"]))}
              for s in ordered if s.get("status", "ok") == "ok"]
    # N-14a: a check skipped for the server's memory budget or for missing model files is said as the server's
    not_run = [{"anchor": anchor("sig", s["id"]), "name": s.get("name", ""),
                "status": ON_THIS_SERVER if of_the_server(s.get("status"), s.get("status_reason")) else RUN_STATUS.get(s.get("status"), s.get("status")),
                "reason": s.get("status_reason") or NOT_RECORDED}
               for s in ordered if s.get("status", "ok") != "ok"]
    cards = {c["id"]: c for c in snapshot.model_cards}
    slice_id = (result.get("condition_profile") or {}).get("slice_id")
    calibration = result.get("calibration") or []
    blocks = []
    for ref in sorted((result.get("manifest") or {}).get("detectors") or [], key=lambda d: d["id"]):
        card = cards.get(ref["id"])
        records = [c for c in calibration if str(c.get("subject", "")).split("@")[0] == ref["id"]]
        testing = [f"{formatting.context_words(r['context'])} ({r['bench_version']}): {view.frequency(float(r['fpr']), int(r['n_neg']))} genuine {many} scored above the "
                   f"flagging threshold, and {view.frequency(float(r['tpr']), int(r['n_pos']))} AI-generated or manipulated {many} did."
                   for c in records for r in c.get("error_rates") or []]
        evals = (card or {}).get("eval") or []
        hit = next((row for row in evals if row.get("slice") == slice_id), None)
        if not testing and hit is not None:
            testing = [f"{hit['bench']}, slice {hit['slice']}: AUC {_metric(hit.get('auc'))} · TPR at 1 % FPR {_metric(hit.get('tpr_at_fpr01'))} · "
                       f"ECE {_metric(hit.get('ece'))}" + (f" · n {hit['n']}" if hit.get("n") is not None else "") + (f" · {hit['date']}" if hit.get("date") else "")]
        if not testing:
            testing = [f"{NOT_ESTABLISHED} for this slice ({_plural(len(evals), 'benchmark row')} on the model card)" if evals
                       else f"{NOT_ESTABLISHED}: no measured error rate is on record for this detector"]
        # N-19k (the PDF words of both reviews): the card's modes in client wording, as a result publishes them; a mode that
        # speaks the build's shorthand ("CFE 10 of 30 in both") stays on the model card
        recorded = sorted(set((card or {}).get("fp_modes") or []) | {m for s in signals.values() if s.get("detector_id") == ref["id"] for m in s.get("fp_modes") or []})
        # N-19n (D3-08): and the measured row of a card in the reader's words, where the card holds it in the benchmark's
        # shorthand; no block points to "the engineering record of the model card"
        false_alarms, measured = card_copy.block_modes(recorded, [ref["id"]])
        causes = prose.system_prose("; ".join(false_alarms))
        said = " ".join(x for x in (f"{causes}." if causes and measured else causes, *measured) if x)
        licences = ref.get("licence") or "; ".join(f"{x['spdx']} ({x['class']})" for x in (card or {}).get("licences") or []) or NOT_RECORDED
        weights = ref.get("weights_sha256") or ref.get("revision")
        blocks.append({"id": ref["id"], "anchor": anchor("det", ref["id"]), "name": (card or {}).get("name") or ref["id"], "version": ref.get("version", ""),
                       "weights": weights or "model-free", "licence": licences, "device": formatting.device_words(ref.get("device"), ref.get("precision")) or NOT_RECORDED,
                       "citation": prose.system_prose((card or {}).get("citation") or NOT_ESTABLISHED), "admission": admission_basis(ref, card)[0],
                       "testing": testing, "false_alarms": said or NOT_ESTABLISHED,
                       "signals": ", ".join(sorted(s.get("display_id") or s["id"] for s in signals.values() if s.get("detector_id") == ref["id"])) or "–"})
    # N-19q (N-20b's read of the reports, finding 9): a check whose score is not a probability has no P(AI); its finding quotes
    # the score ("returned a score of 1.00" beside a dash), so the ledger says why the two differ
    scored_only = any(p_ai(s) is None and s.get("raw_score") is not None and not view.is_mention(s)
                      for s in ordered if s.get("status", "ok") == "ok")
    # N-19q (finding 6): a check in the fusion model prints its calibrated probability under P(AI) and its finding the detector's
    # own score ("returned a score of 0.83" beside P(AI) 0.63): the ledger names which value each is
    calibrated = any(s.get("calibrated_probability") is not None and s.get("raw_score") is not None and p_ai(s) is not None
                     and f"{p_ai(s):.2f}" != f"{float(s['raw_score']):.2f}" for s in ordered if s.get("status", "ok") == "ok")
    return {"ledger": ledger, "not_run": not_run, "detectors": blocks, "p_note": P_NOTE if scored_only else None,
            "p_calibrated_note": P_CALIBRATED_NOTE if calibrated else None}


P_NOTE = "A dash under P(AI): the check gives a score that is not a probability; its finding quotes that score."
P_CALIBRATED_NOTE = ("Where P(AI) and a finding’s score differ, P(AI) is the check’s score calibrated on the benchmark and taken "
                     "alone; the finding quotes the detector’s own score, which the fusion model reads.")


def admission_basis(ref: dict[str, Any], card: dict[str, Any] | None) -> tuple[str, str | None]:
    """RP-04: the admission the result's manifest recorded when the analysis ran, with a dated addendum once the card has changed;
    a result without one prints the current card, marked. Returns the text and the decision the Acceptance factor counts."""
    recorded, now = ref.get("admission"), (card or {}).get("admission_decision") or (card or {}).get("admission")
    if recorded:
        changed = card is not None and (now, card.get("admission_date")) != (recorded["decision"], recorded.get("date"))
        addendum = f" (card now: {', '.join(x for x in ((card or {}).get('admission') or NOT_RECORDED, (card or {}).get('admission_date')) if x)})"
        return recorded["decision"] + (addendum if changed else ""), recorded["decision"]
    if now:
        return f"{card.get('admission') or now} (not recorded at analysis time)", now
    return NOT_RECORDED, None


def methods(snapshot: ReportSnapshot, decision: dict[str, Any]) -> dict[str, Any]:
    """RP-04 and UX §6.3 R9 (methods.ts): envelope, slice metrics, reliability factors, per-detector validation, method rows."""
    result, kind = snapshot.result, snapshot.analysis.kind
    profile = result.get("condition_profile")
    if not profile:
        envelope = "No condition profile is recorded for this analysis, so no validation slice applies to it."
    elif profile.get("within_validated_envelope") and profile.get("validation_support") == "pooled":
        envelope = (f"Validation uses the broader group {profile.get('validation_measured_at') or 'recorded in the model'}. "
                    f"Performance for this exact input slice ({formatting.slice_words(profile['slice_id'])}) has not been established separately.")
    elif profile.get("within_validated_envelope") and profile.get("validation_support") in (None, "legacy"):
        envelope = (f"This input meets the recorded validation AUC threshold ({formatting.slice_words(profile['slice_id'])}). "
                    "The legacy record does not identify whether validation was direct or pooled.")
    elif profile.get("within_validated_envelope"):
        envelope = (f"This input meets the recorded AUC support criterion for its slice ({formatting.slice_words(profile['slice_id'])}). "
                    "This does not establish a false-positive or detection rate for this exact input slice.")
    else:
        why = [c for c in profile.get("limiting_conditions") or [] if c]
        envelope = (f"This input ({formatting.slice_words(profile['slice_id'])}) is outside the validated envelope" + (f": {'; '.join(why)}" if why else "")
                    + ". No measured error rate describes inputs like it.")
    ref = result.get("validation_ref") or {}
    m = ref.get("slice_metrics") or {}
    fuser = _fuser(result)
    n = f"{m['n_pos']} AI-generated or manipulated · {m['n_neg']} genuine" if m and (m.get("n_pos", 0) + m.get("n_neg", 0)) > 0 else None
    slice_rows = [_row("Slice", formatting.slice_words(profile["slice_id"]) if profile and profile.get("slice_id") else None, NOT_RECORDED),
                  _row("Validation record", ref.get("dossier_version"), f"{NOT_ESTABLISHED}: no validation record is in force"),
                  _row("TPR", _with_ci(m["tpr"], m.get("tpr_ci95")) if m.get("tpr") is not None else None),
                  _row("FPR", _with_ci(m["fpr"], m.get("fpr_ci95")) if m.get("fpr") is not None else None),
                  _row("AUC", _metric(m["auc"]) if m.get("auc") is not None else None),
                  _row("Samples", n),
                  _row("Calibration ECE", f"{fuser['ece']:.3f} ("
                       + "; ".join(x for x in (fuser.get("bench_version"), test_view_words(fuser.get("test_view"))) if x) + ")"
                       if fuser and fuser.get("ece") is not None else None)]
    manifest = result.get("manifest") or {}
    refs = manifest.get("detectors") or []
    cards = {c["id"]: c for c in snapshot.model_cards}
    ran = [cards.get(d["id"]) for d in refs]
    cited = sum(1 for c in ran if c and c.get("citation") and not own_rule(c["citation"]))
    own = sum(1 for c in ran if c and c.get("citation") and own_rule(c["citation"]))
    admitted = sum(1 for d in refs if admission_basis(d, cards.get(d["id"]))[1] == "admitted")  # recorded at analysis time
    rates = (fuser or {}).get("error_rates") or []
    tested = ((f"Fusion calibrated and tested on {fuser['bench_version']}" + (f" (manifest {formatting.short_hash(fuser['bench_manifest_sha256'])})" if fuser.get("bench_manifest_sha256") else "")
               + (f"; validation record {ref['dossier_version']}" if ref.get("dossier_version") else "")) if fuser and fuser.get("bench_version")
              else f"Validation record {ref['dossier_version']}" if ref.get("dossier_version") else None)
    publication = publication_line(cited, own, len(refs))
    error_rate = (f"This slice ({formatting.slice_words((profile or {}).get('slice_id'))}): TPR {_with_ci(m.get('tpr'), m.get('tpr_ci95'))}, FPR {_with_ci(m.get('fpr'), m.get('fpr_ci95'))}"
                  if m and (m.get("tpr") is not None or m.get("fpr") is not None)
                  else f"{_plural(len(rates), 'measured error-rate row')} for the fusion ({', '.join(sorted({r['bench_version'] for r in rates}))})" if rates else None)
    controls = "; ".join([
        "reproducibility manifest recorded" + (f" (config {formatting.short_hash(manifest['config_sha256'])})" if manifest.get("config_sha256") else ""),
        (f"fusion artefact {formatting.short_hash(decision['fuser_sha256'])}, {formatting.review_phrase(fuser.get('reviewed_by') if fuser else None, fuser.get('review_date') if fuser else None)}")
        if decision.get("fusion_mode") == "artefact" and decision.get("fuser_sha256") else "uncalibrated fallback fusion with hand-set weights",
        f"release check {'against validation record ' + ref['dossier_version'] if ref.get('dossier_version') else NOT_ESTABLISHED}"])
    acceptance = (f"{of_checks(admitted, len(refs), 'is', 'are')} admitted on the internal benchmark; independent benchmark results {NOT_ESTABLISHED}"
                  if refs and admitted else None)
    factors = [_row("Tested (harness and dataset)", tested), _row("Peer review and publication", publication), _row("Known error rate", error_rate),
               {"key": "Standards and controls", "value": controls, "established": True},
               _row("Acceptance (published methods, independent benchmarks)", acceptance,
                    f"{NOT_ESTABLISHED}: no detector that ran is admitted on the internal benchmark, and no independent benchmark result is recorded")]
    slice_id = (profile or {}).get("slice_id")
    validation = []
    for d in sorted(refs, key=lambda d: d["id"]):
        card = cards.get(d["id"])
        evals = (card or {}).get("eval") or []
        hit = next((r for r in evals if r.get("slice") == slice_id), None)
        measured = (" · ".join(x for x in (f"AUC {_metric(hit.get('auc'))}", f"TPR at 1 % FPR {_metric(hit.get('tpr_at_fpr01'))}", f"ECE {_metric(hit.get('ece'))}",
                                           f"n {hit['n']}" if hit.get("n") is not None else None, ", ".join(x for x in (hit.get("bench"), hit.get("date")) if x)) if x)
                    if hit else f"{NOT_ESTABLISHED} for this slice ({_plural(len(evals), 'benchmark row')} on the card)" if evals
                    else f"{NOT_ESTABLISHED}: no benchmark results on the card" if card else f"{NOT_ESTABLISHED}: no model card on record")
        validation.append({"id": d["id"], "version": d.get("version", ""), "admission": admission_basis(d, card)[0], "measured": measured})
    fusion = ("Uncalibrated fallback: hand-set weights combine the detector scores. The result is a score, not a calibrated probability."
              if decision.get("fusion_mode") == "fallback"
              else f"Calibrated fusion ({(fuser or {}).get('subject', 'fuser artefact')}): a monotone logistic stacker over the detector scores, "
                   "Platt calibration per input context, and conformal bands for the headline.")
    start_term = "model starting point" if view.start_kind(decision) == "context_prior" else "starting point (every check at its neutral point)"
    formula = (f"logit(estimate) = {start_term} + sum of the checks' contributions + missing-check terms + adjustments; the estimate is "
               "clamped to 2–98." if decision.get("fusion_mode") != "fallback"
               else "score = sigmoid(sum of w × logit(p) / sum of w) over the checks that produced a probability p, with a hand-set weight "
                    "w per check.")
    method_rows = [_row("Pipeline", f"{manifest.get('pipeline_version')} · profile {manifest.get('effective_profile') or manifest.get('profile')}" if manifest else None, NOT_RECORDED),
                   _row("Fusion", fusion, NOT_RECORDED),
                   _row("Calibration", formatting.calibration_method(fuser["method"]) + (f" on {fuser['bench_version']}" if fuser.get("bench_version") else "") if fuser
                        else "none (uncalibrated)" if decision.get("fusion_mode") == "fallback" else None),
                   _row("Licence posture", formatting.policy_words(manifest.get("licence_posture")), NOT_RECORDED)]
    # D-63: in a result measured from neutral points the words of a check in the estimate come from its contribution, which is
    # a factor on the odds (ln 2, ln 10, ln 100 in log-odds); every other result keeps the likelihood-ratio rows
    measured = view.measured_from_neutral(decision)
    ratio = "factor on the odds" if measured else "likelihood ratio"
    strengths = [{"word": "strong", "rule": f"{ratio} of at least 100"}, {"word": "moderate", "rule": f"{ratio} from 10 to 100"},
                 {"word": "weak", "rule": f"{ratio} from 2 to 10"}, {"word": "little effect", "rule": f"{ratio} below 2"},
                 {"word": "documentary", "rule": "Content Credentials validated against the trust list (no ratio)"},
                 # N-19n (D3-07): the word of a marker's finding, which no row of ratios defines; in a report whose result
                 # holds such a finding (the glossary follows the words that a report prints)
                 *([{"word": "self-declared", "rule": "a record in the file that anyone can write (no ratio)"}]
                   if any(view.is_marker(f) for f in snapshot.result.get("findings") or []) else []),
                 # (8): the word of Content Credentials that do not validate, the product's own word for the state
                 *([{"word": "invalid", "rule": "Content Credentials that do not validate: the file or the record changed after signing, "
                                                "or the record is malformed (no ratio)"}]
                   if any(view.is_invalid_credentials(f) for f in snapshot.result.get("findings") or []) else [])]
    strengths_note = ("For a check in the estimate the factor is its contribution: how far it moved the odds from its neutral point, "
                      "toward or against. For a check outside the estimate it is the check’s stand-alone likelihood ratio."
                      if measured else None)
    stages = list(STAGES.get(kind, STAGES["image"]))
    rule = headline_rule(kind, decision)
    if view.not_assessable(decision):  # N-17 (D-68): nothing was fused, so no formula, calibration or headline level is described
        weightless = view.no_weight(decision)  # N-19f: the fusion was reached and gave the checks that scored no weight
        stage = f"Fusion: reached, with no result, because {NO_WEIGHT_FUSED}." if weightless else f"Fusion: not reached, because {NOT_FUSED}."
        none = f"None: {NO_WEIGHT_FUSED}." if weightless else f"None: {NOT_FUSED}."
        stages = [stage if x.startswith("Fusion") else x for x in stages]
        method_rows = [r if r["key"] not in ("Fusion", "Calibration") else _row(r["key"], none) for r in method_rows]
        formula = None
        rule = {"sentences": [NO_WEIGHT_RULE if weightless else NOT_ASSESSABLE_RULE], "rates": [], "scope": None, "note": None}
    return {"stages": stages, "envelope": envelope, "within": bool((profile or {}).get("within_validated_envelope")),
            "slice": slice_rows, "factors": factors, "validation": validation, "method": method_rows, "formula": formula,
            "headline_rule": rule, "strengths_note": strengths_note,
            "strengths": strengths}


# The manifest's sampling keys in words (L-7: the row printed `voice_window_s 4.0 · max_windows 300`)
SAMPLING_WORDS = {"voice_window_s": "voice window (s)", "max_windows": "windows at most", "max_seconds": "seconds analysed at most",
                  "sample_rate": "sample rate (Hz)", "plan_version": "sampling plan", "legacy_frames": "frames sampled",
                  "tracked_frames": "frames tracked", "tracked_fps": "tracking rate (frames per second)",
                  "frame_indices_sha256": "frame indices SHA-256"}


# N-19q (N-20b's read of the reports, finding 7): a check whose model card cites Veritas's own rules (a heuristic of the
# code, a rule of a research note) cites no publication. It read "15 of the 15 checks in this analysis cite a publication"
# beside citations of "Veritas v0.3"
OWN_RULE = re.compile(r"\bVeritas v\d|\bbackend/|\bdocs/v2/")


def own_rule(citation: str) -> bool:
    return bool(OWN_RULE.search(citation))


def publication_line(cited: int, own: int, n: int) -> str | None:
    """The row "Peer review and publication": the checks that cite a publication other than Chipmunk Verify's rules, and the
    others named as Chipmunk Verify's rules."""
    if not n or not (cited or own):
        return None
    rules = f"{own} {'is' if own == 1 else 'are'} Chipmunk Verify’s own {'rule' if own == 1 else 'rules'}, with no publication"
    if not cited:
        return f"No check in this analysis cites a publication; {rules} (listed on the model cards)"
    line = f"{of_checks(cited, n, 'cites', 'cite')} a publication"
    return f"{line}; {rules} (listed on the model cards)" if own else f"{line} (listed on the model cards)"


def of_checks(k: int, n: int, singular: str, plural: str) -> str:
    """"3 of the 4 checks in this analysis cite", "1 of the 4 … cites", "The only check in this analysis cites" (L-7: the count
    names its population, so it does not read against "three checks entered the fusion model"; no "1 of 1 checks")."""
    if n == 1 and k == 1:
        return f"The only check in this analysis {singular}"
    return f"{k} of the {n} checks in this analysis {singular if k == 1 else plural}"


def reproducibility(snapshot: ReportSnapshot, engine_version: str, template_set: str) -> dict[str, Any]:
    result = snapshot.result
    m = result.get("manifest") or {}
    profile = result.get("condition_profile") or {}
    decision = result.get("decision") or {}
    created, finished = snapshot.analysis.created_at, snapshot.analysis.finished_at
    duration = f"{(finished - created).total_seconds():.1f} s" if created and finished else NOT_RECORDED
    rows = [_row("Chipmunk Verify", m.get("veritas_version"), NOT_RECORDED), _row("Pipeline", m.get("pipeline_version"), NOT_RECORDED),
            _row("Profile", (f"{m.get('profile')} (effective: {m['effective_profile']})" if m.get("effective_profile") and m.get("effective_profile") != m.get("profile") else m.get("profile")), NOT_RECORDED),
            _row("Licence posture", formatting.policy_words(m.get("licence_posture")), NOT_RECORDED), _row("Git commit", m.get("git_commit"), NOT_RECORDED),
            _row("Configuration SHA-256", m.get("config_sha256"), NOT_RECORDED),
            _row("Device", formatting.device_words(m.get("device"), m.get("precision")), NOT_RECORDED),
            _row("Analysis template set", formatting.version_words(m.get("template_set")), NOT_RECORDED),
            _row("Fusion", (formatting.fusion_words(decision.get("fusion_mode")) or "")
                 + (f" · artefact {decision['fuser_sha256']}" if decision.get("fuser_sha256") else "") or None, NOT_RECORDED),
            _row("Calibration id", m.get("calibration_id") or decision.get("fuser_sha256"), NOT_RECORDED),
            _row("Policy thresholds", m.get("policy_thresholds_id"), NOT_RECORDED),
            _row("Libraries", " · ".join(f"{k} {v}" for k, v in sorted((m.get("library_versions") or {}).items())) or None, NOT_RECORDED),
            _row("Sampling and seeds", " · ".join(f"{SAMPLING_WORDS.get(k, k.replace('_', ' '))} {formatting.version_words(v) if k == 'plan_version' else v}"
                                                   for k, v in sorted((m.get("sampling") or {}).items())) or None, "not applicable"),
            _row("Timings", f"analysis created {formatting.utc_seconds(created)}, finished {formatting.utc_seconds(finished)} ({duration})", NOT_RECORDED),
            _row("Result document SHA-256", canonical_hash(result), NOT_RECORDED),
            _row("Report snapshot SHA-256", snapshot_hash(snapshot), NOT_RECORDED),
            _row("Report engine", f"{engine_version} · template set {template_set}", NOT_RECORDED)]
    registry = [{"id": d["id"], "version": d.get("version", ""), "weights": d.get("weights_sha256") or d.get("revision") or "model-free",
                 "licence": d.get("licence") or NOT_RECORDED, "coverage": prose.coverage(d.get("coverage_date") or NOT_RECORDED),
                 "device": formatting.device_words(d.get("device"), d.get("precision")) or NOT_RECORDED}
                for d in sorted(m.get("detectors") or [], key=lambda d: d["id"])]
    # L-3: "inside" never stands alone beside error rates that are not established
    rates_known = "not established" not in view.in_testing(snapshot.analysis.kind, result.get("decision") or {}, profile,
                                                           list(result.get("calibration") or []))["line"]
    inside = "inside" if rates_known else "inside a measured slice; error rates for inputs like this are not established"
    # N-19d (DS-19): the keys and the slice in words (the table read "SLICE ID image/lossless/side<512/face<120", "DURATION S")
    conditions = [{"key": formatting.CONDITION_WORDS.get(k, k.replace("_", " ")),
                   "value": ("; ".join(v) or "none") if isinstance(v, list) else (inside if v else "outside") if k == "within_validated_envelope"
                   else formatting.slice_words(v) if k == "slice_id" else str(v)}
                  for k, v in profile.items() if v is not None]
    # N-19c (design review DS-17, DS-18): the last row of the manifest table, in words a reader can act on. It means the re-run
    # with the pinned manifest of this analysis (it named the route of a new analysis under the current configuration, as an
    # API path), and it no longer starts a page of its own.
    again = (f"run analysis {snapshot.analysis.id} again with the manifest pinned above (in Chipmunk Verify, “Re-run…” on the analysis), on an "
             "instance that holds the weights listed under Model registry")
    number = "score" if (result.get("decision") or {}).get("fusion_mode") == "fallback" else "estimate"
    rerun = (f"To reproduce the result, {again}. On the same host and device the file should again be not assessable, for the same reason."
             if view.not_assessable(result.get("decision"))  # N-17 (D-68)
             # N-19q (finding 9): the fallback's number is the score (N-19n's rule)
             else f"To reproduce the {number}, {again}. On the same host and device the {number} should agree to 3 decimals.")
    rows.append({"key": "Re-run", "value": rerun, "established": True, "prose": True})
    return {"rows": rows, "registry": registry, "conditions": conditions, "rerun": rerun}


def manifest_attachment(snapshot: ReportSnapshot, engine_version: str, template_set: str) -> dict[str, Any]:
    """The reproducibility manifest attached to the PDF (ARCHITECTURE §6.4, §6.6): the result's manifest plus input and output ids."""
    result = snapshot.result
    return {"schema": "veritas.reproducibility-manifest/1", "analysis_id": snapshot.analysis.id, "analysis_revision": snapshot.analysis.revision,
            "manifest": result.get("manifest") or {}, "input": {"sha256": snapshot.media.sha256, "md5": snapshot.media.md5},
            "result_sha256": canonical_hash(result), "fuser_sha256": (result.get("decision") or {}).get("fuser_sha256"),
            "dossier_version": (result.get("validation_ref") or {}).get("dossier_version"), "report": {
                "id": snapshot.report.id, "display_id": snapshot.report.display_id, "revision": snapshot.report.revision,
                "profile": snapshot.report.profile, "snapshot_sha256": snapshot_hash(snapshot), "engine": engine_version, "template_set": template_set}}


NOT_STATED = "not stated"
TAKEN_IN = "taken in by Chipmunk Verify"


def custodian_hash_row(snapshot: ReportSnapshot) -> str | None:
    """The cover row of a hash the custodian stated (N-19; design review DS-01): the hash and how it compares with the file."""
    media = snapshot.media
    if media.custodian_hash_match not in ("match", "mismatch") or not media.custodian_stated_hash:
        return None
    compared = "matches the file received" if media.custodian_hash_match == "match" else "DIFFERS from the file received"
    return f"stated {media.custodian_stated_hash}, {compared}"


def acquired_row(snapshot: ReportSnapshot) -> str | None:
    """The cover row "Acquired (as stated)" (N-19; browser QA P1-8): what the custodian stated of the acquisition, then the
    time at which Chipmunk Verify took the file in. The cover printed that system time as "Received", under the custody label."""
    media = snapshot.media
    if media.received_at is None:
        return None
    stated = (formatting.utc(media.date_acquired) if media.date_acquired else "date not stated",
              formatting.ACQUISITION_WORD.get(media.acquisition_method or "", media.acquisition_method),
              f"supplied by {media.supplied_by}" if media.supplied_by else None, f"source {media.source}" if media.source else None)
    return " · ".join([*(x for x in stated if x), f"{TAKEN_IN} {formatting.utc_seconds(media.received_at)}"])


def intake_record(snapshot: ReportSnapshot) -> list[dict[str, Any]] | None:
    """The intake record of the exhibit, for the custody appendix (N-19; browser QA P1-8): method, custodian, source, date
    acquired and notes as they were stated at intake, and the time Chipmunk Verify took the file in. None without an intake record
    (a file that was not taken in through a case intake). The stated hash and its comparison stand beside the hashes."""
    media = snapshot.media
    if media.received_at is None:
        return None
    return [_row("Method", formatting.ACQUISITION_WORD.get(media.acquisition_method or "", media.acquisition_method), NOT_STATED),
            _row("Custodian", media.supplied_by, NOT_STATED), _row("Source", media.source, NOT_STATED),
            _row("Date acquired", formatting.utc(media.date_acquired) if media.date_acquired else None, NOT_STATED),
            _row("Taken in", f"{formatting.utc_seconds(media.received_at)} (the time of this instance)"),
            _row("Notes", media.intake_notes, "none recorded")]


CUSTODY_DETAIL_KEYS = ("display_id", "profile", "status", "decision")


def custody_detail(event: dict[str, Any]) -> str:
    """N-19c (design review, P2 "PDFs"): what tells two events of one kind apart, from the fields the event recorded that are
    not free text: a report's id and profile (two "report generated" rows read alike), a status, a decision."""
    details = event.get("details") or {}
    # N-19k: a sign-off decision in the words the reports use for it ("does not concur"; it read "non concur")
    parts = [(SIGNOFF_DECISION.get(str(details[k])) if k == "decision" else None) or str(details[k]).replace("_", " ")
             for k in CUSTODY_DETAIL_KEYS if isinstance(details.get(k), str) and details[k]]
    if details.get("redacted") is True:
        parts.append("redacted")
    return " · ".join(parts) or "–"


def custody(snapshot: ReportSnapshot) -> dict[str, Any]:
    rows, consistent, previous = [], True, 0
    for e in snapshot.custody:
        ok = audit.event_hash(str(e.get("prev_hash")), e) == e.get("event_hash") and int(e["seq"]) > previous
        consistent &= ok
        previous = int(e["seq"])
        rows.append({"seq": e["seq"], "ts": formatting.utc_seconds(e["ts"]), "action": formatting.action_word(str(e["action"]), e.get("details")),
                     "detail": custody_detail(e),
                     "actor": _names(snapshot, str(e["actor_id"])), "object": (str(e["after_hash"])[:12] + "…") if e.get("after_hash") else "–",
                     "hash": str(e["event_hash"])[:12] + "…", "ok": ok})
    extract = (f"Each of the {_plural(len(rows), 'event')} in this extract was checked: its hash was recomputed from its fields and the "
               "sequence numbers increase." + ("" if consistent else " At least one event does not match its recorded hash.")) if rows \
        else "No custody events are recorded for this analysis."
    v = snapshot.custody_verification
    chain = (f"The whole audit chain ({_plural(int(v['checked']), 'event')}) was verified when this report was requested: intact." if v and v.get("valid")
             else f"The whole audit chain was verified when this report was requested: broken at event {v.get('first_broken_seq')} ({v.get('reason')})." if v
             else "The whole audit chain was not verified for this report.")
    media = snapshot.media
    custodian = ({"text": "The hash stated by the custodian at intake differs from the received file. The mismatch is recorded, not resolved: "
                          "this report describes the file that was received, whose hashes are above.",
                  "stated": media.custodian_stated_hash or "not recorded"} if media.custodian_hash_match == "mismatch"
                 else {"text": "The hash stated by the custodian at intake matches the file received.", "stated": media.custodian_stated_hash}
                 if media.custodian_hash_match == "match" and media.custodian_stated_hash else None)
    return {"rows": rows, "consistent": consistent, "extract": extract, "chain": chain, "custodian": custodian,
            "intake": intake_record(snapshot),
            "report_hash": (f"A PDF cannot contain its own hash. The SHA-256 of this PDF is recorded in the custody log as a “report generated” event for "
                            f"report {snapshot.report.id}.")}


def _geometry(a: Any) -> str:
    g = a.geometry or {}
    if a.shape == "box" and g:
        return _box_text(g) + (f" · frame pts {g['frame_pts']}" if g.get("frame_pts") is not None else "")
    if a.shape == "polygon" and g:
        return f"polygon of {len(g.get('points') or [])} points (original pixels)"
    if a.shape == "arrow" and g:
        return f"arrow from ({g['x1']:.0f}, {g['y1']:.0f}) to ({g['x2']:.0f}, {g['y2']:.0f}) px"
    if a.shape == "text" and g:
        return f"pin at ({g['x']:.0f}, {g['y']:.0f}) px"
    if a.shape == "interval" and g:
        base = Fraction(g["time_base"])
        return f"{formatting.mmss(float(g['start_pts'] * base))}–{formatting.mmss(float(g['end_pts'] * base))}"
    if a.shape == "sentence" and g:
        return f"sentence {g['sentence_idx'] + 1}" if g.get("sentence_idx") is not None else f"characters {g.get('start')}–{g.get('end')}"
    return "–"


def notes(snapshot: ReportSnapshot, signals: dict[str, dict[str, Any]], exhibits: dict[str, dict[str, Any]],
          marks: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    rows = []
    live = live_annotations(snapshot)
    for n, a in enumerate(live, 1):
        target = {"finding": a.target_id, "signal": signals.get(a.target_id or "", {}).get("display_id") or a.target_id,
                  "exhibit": exhibits.get(a.target_id or "", {}).get("display_id") or a.target_id}.get(a.target_type) or a.target_type
        rows.append({"id": f"N{n}", "target": f"{a.target_type} {target}" if a.target_type in ("finding", "signal", "exhibit") else target,
                     "where": _geometry(a), "status": EXAMINER_STATUS.get(a.examiner_status or "", "–"),
                     "body": a.body if a.visibility == "in_report" else ("Internal note, not reproduced in reports." if a.body else "–"),
                     "author": _names(snapshot, a.author_id), "at": formatting.utc(a.edited_at or a.created_at) + (f" · revision {a.revision}" if a.revision > 1 else "")})
    return {"rows": rows, "dispositions": [marks[k] for k in sorted(marks)], "internal": sum(1 for a in live if a.visibility != "in_report" and a.body)}


def review_state_words(snapshot: ReportSnapshot, status: view.ReviewStatus) -> str:
    """The review state of this report in words: a signed report that is not issued never reads as reviewed (D-56).
    N-19j (browser QA Q2-08): a revision that the reviewer passed with changes is not printed as plainly "peer reviewed":
    the state says that the reviewer concurred with changes, with the comment as the changes asked for."""
    state = snapshot.review.state
    if state == "reviewed" and status.signed and not status.issued and "second review outstanding" in status.line:
        return SECOND_REVIEW_OUTSTANDING
    words = REVIEW_STATE.get(state, state)
    peer = next((s for s in snapshot.review.signoffs if s.kind == "peer_review"), None)
    if state == "approved" and peer is not None and peer.decision == "concur_with_changes":
        words = REVIEWED_WITH_CHANGES + (f" Changes asked for: {_sentence(peer.comment)}" if peer.comment else " No comment states them.")
    return words + "; a review record does not verify" if "does not verify" in status.line else words


def _span(ids: list[str]) -> str:
    return ids[0] if len(ids) == 1 else f"{ids[0]} and {ids[1]}" if len(ids) == 2 else f"{ids[0]} to {ids[-1]}"


def acknowledged_line(ids: list[str], listed: int | list[dict[str, Any]] | None) -> str:
    """Which limitations the examiner acknowledged. N-19j (browser QA Q2-04): the report says which of the limitations it
    lists were put to the examiner. The conclusion form offers the result's own (`offered`: their numbers are their places
    among the result's caveats, which a conclusion stores); the report adds the limitations of the method after them when
    it is printed. It printed "L-01, L-02, L-03 (3 of the 10 listed)", as if seven had been declined.
    `listed`: the limitations as `applied_limitations` returns them. A count (N-19c) keeps the earlier words."""
    if listed is None:
        return ", ".join(ids) or "none"
    if isinstance(listed, int):
        if not ids:
            return f"none of the {listed} listed" if listed else "none: no limitation is listed"
        return f"{', '.join(ids)} ({len(ids)} of the {listed} listed)"
    if not listed:
        return "none: no limitation is listed"
    offered = [x["id"] for x in listed if x.get("offered")]
    added = [x["id"] for x in listed if not x.get("offered")]
    not_put = (f" {_span(added)} {'is a limitation' if len(added) == 1 else 'are limitations'} of the method that this report adds; "
               f"{'it was' if len(added) == 1 else 'they were'} not put to the examiner." if added else "")
    if not offered:
        return ("No limitation of this result was put to the examiner." + not_put) if added else "none: no limitation is listed"
    kept = [i for i in ids if i in offered]
    other = [i for i in ids if i not in offered]  # a number the form of an earlier build stored: printed as it was acknowledged
    put = f"the {len(offered)} put to the examiner" if len(offered) > 1 else "the one put to the examiner"
    said = (f"{', '.join(kept)}: all of {put}." if len(kept) == len(offered) and len(offered) > 1
            else f"{kept[0]}: {put}." if len(kept) == len(offered)
            else f"{', '.join(kept)} of {put} ({_span(offered)})." if kept
            else f"none of {put} ({_span(offered)}).")
    return said + (f" Also acknowledged: {', '.join(other)}." if other else "") + not_put


def signoff(snapshot: ReportSnapshot, decision: dict[str, Any], basis: str, status: view.ReviewStatus,
            listed: int | list[dict[str, Any]] | None = None) -> dict[str, Any]:
    coverage = snapshot.analysis.coverage.model_dump() if snapshot.analysis.coverage is not None else None
    findings = list(snapshot.result.get("findings") or [])
    n = view.result_number(decision, findings, snapshot.result.get("condition_profile"))
    # D-68: the number as the result page and the sign-off dialog say it; N-19n (D4-03): whose it is under a record; (D4-04)
    # in the line's case, as the line of the one-page reports sets it ("… · Estimate of the statistical checks alone 7–56")
    phrase = view.number_phrase(n, own=view.record_of(decision) is not None)
    machine = view.headline(decision, basis)
    if basis not in ("insufficient", "documentary_declared") and phrase is not None:
        machine += f" · {view._lower_first(phrase)}" + (f" (plausible range {n.span} %)" if n.kind == "calibrated" and n.span else "")
        if n.kind != "capped" and basis != "documentary_capture":  # N-19m (D3-01): a capture's result holds no agreement
            failed = effect.findings_failed(snapshot.result.get("caveats"))  # N-19n (Q4-01)
            machine += f" · agreement {view.agreement_beside(n, view.agreement(decision, coverage, findings, failed=failed))}"
    c = snapshot.review.conclusion
    conclusion = None
    if c is not None:
        position = "No opinion on the scale" if c.scale_position == "no_opinion" else view.HEADLINE[view.CODE_BAND.get(c.scale_position, "uncertain")]
        steps = c.deviation_steps
        toward = "indications" if (steps or 0) > 0 else "no indications"
        reference = view.review_reference(decision, c)  # N-19n (D4-01): under a capture, the step of the capture record
        relation = ("No comparison with the machine assessment is recorded." if steps is None
                    else f"On {reference}." if steps == 0 and reference is not None
                    else f"{_plural(abs(steps), 'step')} from {reference}, toward {toward}." if reference is not None
                    else ("Concurs with the machine assessment, with changes." if view.with_changes(snapshot)
                          else "Concurs with the machine assessment.") if steps == 0
                    else f"Departs from the machine assessment by {_plural(abs(steps), 'step')} toward {toward}.")
        # N-19p (D6-02): "Deviation" only where the conclusion departs; a justification typed on the step is "Justification"
        from app.cases import review as review_rules

        conclusion = {"position": position, "relation": relation, "rationale": c.rationale, "next_steps": c.next_steps,
                      "justification": c.deviation_justification,
                      "justification_label": "Deviation" if review_rules.departs_of(c) else view.JUSTIFICATION_LABEL, "acknowledged": acknowledged_line(list(c.limitations_acknowledged), listed),
                      "author": c.author_name, "at": formatting.utc(c.created_at),
                      "record": "The record of this conclusion does not verify: its signature does not match its content."
                                if c.signature_valid is False else None}
    # D-61: the authentication method is in the sign-off's custody event (its object ids name the sign-off record)
    auth = {str(i): (e.get("details") or {}).get("auth") for e in snapshot.custody if e.get("action") == "signoff" for i in e.get("object_ids") or []}
    declaration_id = (c.declaration_text_id if c is not None else None) or DECLARATION_ID
    declaration = (DECLARATIONS.get(declaration_id) or f"The declaration {declaration_id} is not part of this template set; its text is not reproduced."
                   if c is not None else "No conclusion is recorded, so no declaration is made.")

    def statement(s: Any) -> tuple[str, str] | None:
        """The statement a signer signs under: the examiner's declaration, or what the reviewer did and decided."""
        if s.kind == "examiner":
            return "Declaration", declaration
        if s.kind == "peer_review" and s.decision in REVIEWER_STATEMENT:
            return REVIEWER_STATEMENT_TITLE, REVIEWER_STATEMENT[s.decision]
        return None

    ordered = sorted(snapshot.review.signoffs, key=lambda s: s.kind != "examiner")  # the examiner first, then the reviewer
    signoffs = [{"kind": SIGNOFF_KIND.get(s.kind, s.kind),
                 "decision": view.signoff_decision(s, c, view.review_reference(decision, c) if c is not None else None)
                 + (" (the record does not verify)" if s.signature_valid is False else ""),
                 "name": s.principal_name, "at": formatting.utc(s.signed_at), "comment": s.comment,
                 "statement_title": (statement(s) or (None, None))[0], "statement": (statement(s) or (None, None))[1],
                 "auth": AUTH_WORD.get(str(auth.get(s.id)), "not recorded")} for s in ordered]
    return {"machine": machine, "machine_meta": f"Chipmunk Verify {snapshot.report.veritas_version} · analysis {snapshot.analysis.id[:8]} · "
                                                f"{formatting.utc(snapshot.analysis.finished_at)}",
            "conclusion": conclusion, "signoffs": signoffs, "issued": status.issued, "status_line": status.line,
            # D-69: a conclusion without an examiner sign-off is a draft opinion, with no declaration and no signature lines
            "signed": status.signed,
            "state": review_state_words(snapshot, status), "declaration": declaration}


def revision_rows(snapshot: ReportSnapshot, status: view.ReviewStatus) -> list[dict[str, Any]]:
    rows = [{"display_id": r["display_id"], "id": r["id"], "revision": r["revision"], "at": formatting.utc(r["requested_at"]),
             "state": (SECOND_REVIEW_OUTSTANDING if r["review_state"] == "reviewed" and r.get("issuable") is False
                       else REVIEW_STATE.get(r["review_state"], r["review_state"])), "reason": r.get("revision_reason") or "–",
             "sha256": formatting.short_hash(r["sha256"]) if r.get("sha256") else r["status"]} for r in snapshot.revisions]
    report = snapshot.report
    rows.append({"display_id": report.display_id, "id": report.id, "revision": report.revision, "at": formatting.utc(report.requested_at),
                 "state": review_state_words(snapshot, status), "reason": report.revision_reason or "–",
                 "sha256": "this revision (recorded in the custody log)"})
    return rows


def quoted_values(snapshot: ReportSnapshot) -> list[str]:
    """Values the report quotes rather than writes (RP-02 exemption): names, file and case fields, metadata and detector text,
    analyst text."""
    out: list[str] = [snapshot.media.filename, snapshot.media.content_type, snapshot.report.requested_by_name,
                      *(x for x in (snapshot.media.supplied_by, snapshot.media.source, snapshot.media.acquisition_method,
                                    snapshot.media.intake_notes) if x)]
    if snapshot.case is not None:
        out += [snapshot.case.reference, snapshot.case.title, snapshot.case.classification, snapshot.case.exhibit_label or ""]
    c = snapshot.review.conclusion
    if c is not None:
        out += [c.rationale, c.next_steps or "", c.deviation_justification or "", c.author_name]
    out += [x for s in snapshot.review.signoffs for x in (s.principal_name, s.comment or "")]
    out += [a.body or "" for a in snapshot.annotations] + list(snapshot.actor_names.values()) + list(snapshot.detector_names.values())
    out += [r.get("revision_reason") or "" for r in snapshot.revisions] + [snapshot.report.revision_reason or ""]
    result = snapshot.result
    for s in result.get("signals") or []:
        out += [s.get("name", ""), s.get("observed", ""), s.get("reference", ""), s.get("reasoning", ""), s.get("status_reason") or "",
                s.get("detector_id", ""), s.get("family", ""), *(s.get("fp_modes") or [])]
    for e in result.get("exhibits") or []:
        out += [e.get("title", ""), e.get("caption", ""), e.get("created_by", ""), *(str(c) for r in e.get("rows") or [] for c in r if isinstance(c, str))]
        out += [x for sp in e.get("spans") or [] for x in [sp.get("label", ""), *(sp.get("reasons") or [])]]
    for card in snapshot.model_cards:
        out += [card.get("name") or "", card.get("citation") or "", *(card.get("fp_modes") or []), *(card.get("known_limitations") or [])]
    m = result.get("manifest") or {}
    out += [d["id"] for d in m.get("detectors") or []] + [str(v) for v in (m.get("library_versions") or {}).values()]
    return [x for x in out if x]


def identifier_values(snapshot: ReportSnapshot) -> list[str]:
    """What the internal-vocabulary lint may quote (DR P0-3): the user's data (file, case title and exhibit, analyst and annotation
    text, names, metadata values) and technical identifiers (detector ids and names, library versions). Unlike `quoted_values`,
    model-card prose, system sentences and enum fields are scanned, since that is where build vocabulary leaks."""
    out: list[str] = [snapshot.media.filename, snapshot.media.content_type, snapshot.report.requested_by_name,
                      *(x for x in (snapshot.media.supplied_by, snapshot.media.source, snapshot.media.intake_notes) if x)]
    if snapshot.case is not None:
        out += [snapshot.case.title, snapshot.case.exhibit_label or ""]
    c = snapshot.review.conclusion
    if c is not None:
        out += [c.rationale, c.next_steps or "", c.deviation_justification or "", c.author_name]
    out += [x for s in snapshot.review.signoffs for x in (s.principal_name, s.comment or "")]
    out += [a.body or "" for a in snapshot.annotations] + list(snapshot.actor_names.values())
    out += list(snapshot.detector_names.values()) + list(snapshot.detector_names)
    out += [r.get("revision_reason") or "" for r in snapshot.revisions] + [snapshot.report.revision_reason or ""]
    result = snapshot.result
    for s in result.get("signals") or []:
        out += [s.get("name", ""), s.get("observed", ""), s.get("reference", ""), s.get("detector_id", "")]
    for card in snapshot.model_cards:
        out += [card.get("id") or "", card.get("name") or "", *prose.citation_identifiers(card.get("citation") or "")]
    m = result.get("manifest") or {}
    out += [d["id"] for d in m.get("detectors") or []] + [str(v) for v in (m.get("library_versions") or {}).values()]
    out += list((m.get("library_versions") or {}).keys())
    return [x for x in out if x]


REVISION_MODE_WORDS = {"rerun": "re-run", "reanalyse": "re-analysed", "refusion": "re-fused"}


def supersedes_row(snapshot: ReportSnapshot) -> list[dict[str, Any]]:
    """The cover row of a later analysis revision (J-3e): the revision it supersedes and how this one was made."""
    analysis = snapshot.analysis
    if not analysis.supersedes_id:
        return []
    made = f"; this revision was {REVISION_MODE_WORDS[analysis.mode]}" if analysis.mode in REVISION_MODE_WORDS else ""
    number = analysis.supersedes_revision if analysis.supersedes_revision is not None else analysis.revision - 1
    return [_row("Supersedes analysis", f"Revision {number} ({analysis.supersedes_id}){made}")]


COVER_ESTIMATES = "Estimates are probabilistic evidence subject to uncertainty and error, not a finding of fact."
COVER_NOT_ASSESSABLE = "The automated checks could not assess this exhibit, which is not a finding either way."  # N-17 (D-68)
REACHED_TITLE, REACHED_TITLE_NONE = "How the estimate was reached", "How the result was reached"
# N-19k (D2-03; ContributionWaterfall.tsx `reachedTitle`): the chapter is headed by what it shows. Outside what was validated
# that is what the checks returned, and under a score of hand-set weights it is not an estimate
REACHED_TITLE_CAPPED, REACHED_TITLE_SCORE = "What the checks returned", "How the score was reached"


def reached_title(result: dict[str, Any]) -> str:
    """N-19n (D3-04; contributions.ts `reachedTitle`): where a record decides alone there is no estimate and no score to
    reach ("How the estimate was reached" stood over "There is no statistical estimate to break down"), on both fusion
    paths; every result of the fallback that has a score is headed by the score, with a record or without one."""
    decision = view.decision_of(result)
    if view.not_assessable(decision) or view.record_alone(decision):
        return REACHED_TITLE_NONE
    number = view.result_number(decision, list(result.get("findings") or []), result.get("condition_profile"))
    if breakdown_flat(decision, result.get("condition_profile"), number):  # N-19l (R6-02): under a record too
        return REACHED_TITLE_CAPPED
    if number.kind == "capped":
        return REACHED_TITLE_CAPPED
    return REACHED_TITLE_SCORE if decision.get("fusion_mode") == "fallback" else REACHED_TITLE


def forensic_view(snapshot: ReportSnapshot, images: dict[str, str], *, engine_version: str, template_set: str,
                  thumbnail_details: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    result, report, kind = snapshot.result, snapshot.report, snapshot.analysis.kind
    decision = view.decision_of(result)  # N-19h: with the record that decides, also for a stored result
    findings = list(result.get("findings") or [])
    basis = view.basis_of(decision, findings)
    step = view.step_of(decision, basis)
    status = view.review_status(snapshot)
    signals = {s["id"]: s for s in result.get("signals") or []}
    # N-19c: every reference to an exhibit (a finding's evidence, a note, a face track) names it as the exhibits chapter does
    exhibit_view = exhibit_rows(snapshot, images, signals)
    printed = exhibit_labels(exhibit_view)
    exhibits = {e["id"]: {**e, "display_id": printed[str(e["id"])]["label"], "anchor": printed[str(e["id"])]["anchor"]}
                if str(e["id"]) in printed else e for e in result.get("exhibits") or []}
    marks = dispositions(snapshot, signals)
    limitations = applied_limitations(snapshot, decision)
    summary = lay_summary(snapshot, decision, basis, limitations)
    case = snapshot.case
    title = view.PROFILE_TITLES[report.profile]
    reference = formatting.case_reference(case.reference) if case is not None else None
    centre = view.case_centre(case)
    temporal_section = temporal(snapshot, signals, images, printed)
    numbered = [("summary", "Summary and scope"), ("findings", "Key findings"),
                ("estimate", reached_title(result)),
                ("exhibits", "Exhibits"), *([("temporal", "Temporal analysis")] if temporal_section else []),
                ("provenance", "Provenance and metadata"), ("detectors", "Detector results and error rates"),
                ("methods", "Methods and validation"), ("limitations", "Limitations"), ("glossary", "Glossary")]
    appendices = [("appendix-a", "Reproducibility"), ("appendix-b", "Custody"), ("appendix-c", "Notes and dispositions"),
                  ("signoff", "Sign-off"), ("revisions", "Revision history")]
    sections = [{"key": k, "number": str(i), "title": t, "href": f"#section-{k}"} for i, (k, t) in enumerate(numbered, 1)]
    sections += [{"key": k, "number": chr(ord("A") + i), "title": t, "href": f"#section-{k}"} for i, (k, t) in enumerate(appendices)]
    number = {s["key"]: s["number"] for s in sections}
    profile = result.get("condition_profile") or {}
    dims = f"{profile['width']}×{profile['height']}" if profile.get("width") and profile.get("height") else None
    media = snapshot.media
    stated_hash = custodian_hash_row(snapshot)
    examiner = view.examiner_opinion(snapshot, status.signed)
    cover = [_row("Case", (reference if reference == "Unfiled" else f"{case.reference} · {case.title}") if case else None, "No case"),
             _row("Exhibit", f"{case.exhibit_label} · {media.filename}" if case and case.exhibit_label else media.filename),
             {"key": "SHA-256", "value": media.sha256, "established": True, "mono": True},
             _row("MD5", media.md5, NOT_RECORDED) | {"mono": True},
             _row("File", " · ".join(x for x in (media.content_type, formatting.size(media.size_bytes), dims,
                                                 f"{float(profile['duration_s']):.1f} s" if profile.get("duration_s") else None) if x)),
             *([_row("Custodian hash", stated_hash)] if stated_hash else []),
             _row("Acquired (as stated)", acquired_row(snapshot), NOT_RECORDED),
             *([_row("Legal hold", "The case is under legal hold.")] if case is not None and case.legal_hold else []),
             _row("Analysis", f"{snapshot.analysis.id} · revision {snapshot.analysis.revision} · analysed {formatting.utc(snapshot.analysis.finished_at)}"),
             *supersedes_row(snapshot),
             _row("Report", f"{report.display_id} · {report.id} · {'issued' if status.issued else 'generated'} {formatting.utc(report.requested_at)} "
                            f"· by {report.requested_by_name}"),
             *([_row("Supersedes", f"{report.supersedes_report_id}" + (f" · {report.revision_reason}" if report.revision_reason else ""))]
               if report.supersedes_report_id else []),
             # N-19 (D-69; browser QA P1-6): the review row no longer ends with the first sentence of the rationale; on a signed
             # record both positions stand under the facts of the cover, at the size of a headline (`cover.html.j2`)
             _row("Review", status.line),
             _row("Classification", case.classification.capitalize() if case and case.classification else None, "Not classified")]
    return {
        "title": f"Chipmunk Verify {title.lower()} {report.display_id}",
        "subject": f"{centre} · {media.filename}",
        "keywords": f"Chipmunk Verify, {report.profile}, {report.display_id}, analysis {snapshot.analysis.id}",
        "header_left": f"Chipmunk Verify · {title}",
        "header_centre": view.header_centre(centre, view.draft_mark(report.profile, status)),
        "issued": status.issued,
        "status_line": status.line,
        "declaration": report.profile == "forensic",
        "cover_title": COVER_TITLE.get(kind, COVER_TITLE["image"]),
        "cover": cover,
        "cover_headline": view.headline(decision, basis),
        "scope_statement": (f"This report records an automated examination by Chipmunk Verify {report.veritas_version} and, where stated, an examiner’s "
                            f"review. {COVER_NOT_ASSESSABLE if view.not_assessable(decision) else COVER_ESTIMATES} Section "
                            f"{number['limitations']} lists the limitations that applied to this exhibit."),
        "sections": sections,
        "number": number,
        "step": step,
        "summary": summary,
        "examiner": examiner,
        "scale": view.scale(decision, basis, step, view.result_number(decision, findings, result.get("condition_profile"))),
        "scale_note": view.scale_note(decision, view.result_number(decision, findings, result.get("condition_profile"))),
        "scope": scope(snapshot),
        "findings": finding_rows(snapshot, images, signals, exhibits, marks, thumbnail_details),
        "thumbnail_note": view.THUMBNAIL_NOTE,
        # N-19n (Q4-01): where no finding stands; a result whose step for the findings failed says that, not that none was recorded
        "findings_none": (effect.findings_not_written("ledger") if effect.findings_failed(result.get("caveats"))
                          else "No findings were recorded for this analysis."),
        "estimate": estimate_section(snapshot, decision),
        "exhibits": exhibit_view,
        "temporal": temporal_section,
        "provenance": provenance(snapshot, signals),
        "detectors": detectors(snapshot, signals, marks),
        "methods": methods(snapshot, decision),
        "limitations": limitations,
        # N-17 (D-68): the first general limitation is about estimates; a not-assessable result has its own first sentence
        "general_limitations": [*((GENERAL_NOT_ASSESSABLE, *GENERAL_LIMITATIONS["all"][1:]) if view.not_assessable(decision)
                                  else GENERAL_LIMITATIONS["all"]), *GENERAL_LIMITATIONS.get(kind, ())],
        "glossary": [{"term": t, "definition": d} for t, d in glossary(decision)],
        "appendix": reproducibility(snapshot, engine_version, template_set),
        "custody": custody(snapshot),
        "notes": notes(snapshot, signals, exhibits, marks),
        "signoff": signoff(snapshot, decision, basis, status, limitations),
        "revisions": revision_rows(snapshot, status),
        "seal": figures.seal_svg(),
        "marks": figures.marks(),
    }
