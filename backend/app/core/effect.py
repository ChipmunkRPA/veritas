"""What a check did in a decision, in one vocabulary (D-63; BUILD_SPEC N-17). The page's mirror is
`frontend/src/features/analysis/vocabulary.ts`.

For a check that entered the estimate, direction and strength come from its contribution, the change in log-odds when the
check moves from its neutral point to its score:
- **direction** by sign: `for` (toward AI generation or manipulation) or `against`;
- **strength** by the D-16 bands on its size: weak from ln 2, moderate from ln 10, strong from ln 100;
- below ln 2 the check **had little effect**: strength `none`, direction `neutral`, whatever its sign.

A check outside the estimate (no contribution, or one of exactly 0) keeps its stand-alone words, from its own likelihood
ratio, and is said to be not in the estimate. The key findings and their counts, the agreement label, the ledger, the strip,
the findings templates and the reports read these functions, so that one check has one word on every surface.

A result published before N-17 (a `context_prior` starting row) was measured against the 50 % baseline and labelled with a
threshold of 0.05 log-odds for "little effect" and with stand-alone strength words: it keeps both (`LEGACY_LITTLE`).
"""
from __future__ import annotations

import math
import re

LN2, LN10, LN100 = math.log(2.0), math.log(10.0), math.log(100.0)
ENTERED = 1e-9  # a contribution of at most this size is a check with no weight in the estimate
LEGACY_LITTLE = 0.05  # UX §5.6 of results published before N-17
LITTLE_EFFECT = "little effect"
# N-19d, N-19k (the PDF words of both reviews): an error-rate context is a key of the benchmark ("video T2V": the test videos
# generated from text, the context of a video without a face; "portrait face": a face of 120 px or more). A sentence says
# its words before "inputs"; a key outside the table is words already ("screenshot").
CONTEXT_WORDS = {"video T2V": "text-to-video", "portrait face": "portrait", "video talking head": "talking-head video", "audio lossy": "lossy audio"}


def context_words(context: str) -> str:
    return CONTEXT_WORDS.get(context, context)


def entered(contribution: float | None) -> bool:
    return contribution is not None and abs(float(contribution)) > ENTERED


def strength(contribution: float) -> str:
    """`none` (little effect), `weak`, `moderate` or `strong` by the size of the contribution (D-16 bands)."""
    size = abs(float(contribution))
    if size >= LN100:
        return "strong"
    if size >= LN10:
        return "moderate"
    return "weak" if size >= LN2 else "none"


def direction(contribution: float, little: float = LN2) -> str:
    """`for`, `against`, or `neutral` when the check had little effect. `little`: the size below which it had little effect
    (`LEGACY_LITTLE` for a result published before N-17)."""
    c = float(contribution)
    return "for" if c >= little else "against" if c <= -little else "neutral"


def words(contribution: float) -> str:
    """"little effect", or "weak support for" … "strong support against": the label of a check that entered the estimate."""
    level = strength(contribution)
    return LITTLE_EFFECT if level == "none" else f"{level} support {'for' if float(contribution) > 0 else 'against'}"


# N-17 (D-68; design review DS-10): a result that is not assessable does not speak of "the estimate" or "the probability".
_ESTIMATE_WORDS = re.compile(r"estimat|probabil", re.IGNORECASE)


def speaks_of_estimate(text: str | None) -> bool:
    """Whether a text names an estimate or a probability."""
    return bool(text) and _ESTIMATE_WORDS.search(str(text)) is not None


def without_estimate(text: str | None) -> str:
    """The sentences of a text that name neither (a stored caption under a result that is not assessable)."""
    return " ".join(x for x in re.split(r"(?<=[.!?])\s+", text or "") if x and not speaks_of_estimate(x))



# N-18b (operations re-check O4-01): a step of the run that failed, in the words of the result. A stage that is not critical
# can fail and the run goes on; the result then lacks what that step gives, and says so: one caveat, the same sentence on the
# page, in the four report profiles and in the coverage finding. `vocabulary.ts` recognises the sentence by its form.
# stage id -> (what the step does, what the result lacks for it)
STEPS: dict[str, tuple[str, str]] = {
    "probe": ("reads the file's technical properties", "this result holds no file properties from it"),
    "provenance": ("reads Content Credentials and provenance records", "this result holds no provenance record from it"),
    "faces": ("finds faces", "this result holds no face regions from it, and no check ran on a face it would have found"),
    "condition": ("measures the conditions of the input", "this result was not compared with the conditions it was validated for"),
    "detect": ("runs the checks", "this result holds no check's score from it"),
    "localise": ("marks regions", "this result shows no regions from it"),
    "shots": ("finds the shots of the video", "this result holds no shot boundaries from it"),
    "tracks": ("follows faces through the video", "this result holds no face tracks from it"),
    "temporal": ("lines the checks' scores up over time", "this result shows no timeline lanes from it"),
    "second_pass": ("examines the flagged segments again", "this result holds no second pass over them"),
    # N-19n (browser QA Q4-01): the step that writes the findings is not critical either. A result is published without
    # them, and its caveat fell to the sentence of a step without words, with the stage's name in backticks
    "synthesise": ("writes the findings", "this result holds no findings from it"),
    "render": ("prepares the exhibits", "this result shows no exhibits from it"),
}
# a stage this table does not hold is never named by its id (N-19n): the analysis record lists the stages by their titles
UNNAMED_STEP = ("is marked as failed in the analysis record", "this result lacks what that step gives")
_STEP_FAILED = re.compile(r"^The step that .+ failed: .+\.$")
# the caveat of a step without words as results stored before N-19n hold it
_RECORDED_AS = re.compile(r"The step that is recorded as `(\w+)` failed: this result lacks what that step gives\.")


def step_failed(stage: str) -> str:
    """The caveat of a step that failed: "The step that marks regions failed: this result shows no regions from it." """
    does, lacks = STEPS.get(stage, UNNAMED_STEP)
    return f"The step that {does} failed: {lacks}."


def step_in_words(text: str) -> str:
    """N-19n: a caveat stored with a stage's id in backticks ("The step that is recorded as `synthesise` failed: this
    result lacks what that step gives."), as it is shown: the sentence of that step in words. The stored caveat stays."""
    return _RECORDED_AS.sub(lambda m: step_failed(m.group(1)), text)


def failed_steps(caveats: object) -> list[str]:
    """The caveats of a result that say a step of its run failed, in their order."""
    return [c for c in caveats or [] if isinstance(c, str) and _STEP_FAILED.match(c)] if isinstance(caveats, (list, tuple)) else []


# --- a result whose step for the findings failed (N-19n; browser QA Q4-01) ---------------------------------------------------
# Such a result is done, with its decision and no finding. Where the findings would stand it says that they could not be
# written, and nothing there describes the evidence: "No finding gave at least weak support either way." is true of a
# result with findings only. The page's mirror: `vocabulary.ts` (`findingsFailed`, `findingsNotWritten`).
FINDINGS_NOT_WRITTEN = "The findings of this result could not be written: a step of the analysis failed."
SCORES_IN = {"ledger": "the evidence ledger", "expert": "the expert view", "full": "the full report"}
AGREEMENT_NOT_WRITTEN = "not reported (the findings could not be written)"
RUN_AGAIN = "Run the analysis again before relying on this result: its findings could not be written. If they fail again, tell the administrator of {server}."
THIS_SERVER, REPORT_SERVER = "this server", "the Veritas server that made this report"


def findings_failed(caveats: object) -> bool:
    """Whether the step that writes the findings failed in the run of this result (its caveat, as stored now or before)."""
    return step_failed("synthesise") in [step_in_words(c) for c in failed_steps(caveats)]


def findings_not_written(where: str) -> str:
    """The sentence in the findings' place, with where the checks and their scores stand (`SCORES_IN`)."""
    return f"{FINDINGS_NOT_WRITTEN} The checks and their scores are in {SCORES_IN[where]}."


def run_again(server: str = REPORT_SERVER) -> str:
    """The next step of such a result: on the page "… of this server.", in a report the server that made it."""
    return RUN_AGAIN.format(server=server)


# --- a check that scored and carries no weight (N-19k; browser QA Q2-12) -----------------------------------------------------
# One sentence carries both halves: the score, and why it is not counted, from the record of that check for this input
# (`SignalV2.outside_estimate`, written by the fusion) and from what the check measured in testing (its benchmark note).
# The page's mirror: `vocabulary.ts outsideWhy`.

AT_CHANCE = 0.05  # an AUC within this distance of 0.50 is said to be at chance
NO_RECORD_WHY = "it carries no weight in the fused model for this input"  # a result stored before N-19k names no reason
NOT_IN_MODEL_WHY = "it is not in the fitted model for inputs like this one"
ZERO_WEIGHT_WHY = "the model gives it no weight for inputs like this one"
VOICE_STEP = "To assess the voice, examine the sound track as audio."
_AUC_NOTE = re.compile(r"reached AUC (\d\.\d\d)(?: \(n = (\d+))?")
# a gate's term, said of the inputs it lets in and of the ones it keeps out
GATE_TERMS = {
    "has_face": ("inputs with a face", "inputs without a face"),
    "face_px:120-256": ("inputs with a face of 120 to 256 px", "inputs without a face of 120 to 256 px"),
    "face_px>=256": ("inputs with a face of 256 px or more", "inputs without a face of 256 px or more"),
    "jpeg_q>=90": ("JPEG quality of 90 or more", "JPEG quality under 90"),
    "jpeg_q:70-89": ("JPEG quality of 70 to 89", "JPEG quality outside 70 to 89"),
    "jpeg_q<70": ("JPEG quality under 70", "JPEG quality of 70 or more"),
    "min_side<512": ("a shorter side under 512 px", "a shorter side of 512 px or more"),
    "min_side>1024": ("a shorter side over 1,024 px", "a shorter side of 1,024 px or less"),
    "upscaled": ("upscaled inputs", "inputs that were not upscaled"),
    "has_camera_exif": ("inputs with a camera's EXIF block", "inputs without a camera's EXIF block"),
    "video_frame": ("frames of a video", "still images"),
    "screenshot_like": ("inputs that look like a screenshot", "inputs that do not look like a screenshot"),
}


def gate_words(literals: list[str] | tuple[str, ...]) -> str:
    """The inputs a gate lets in, in words: `["!has_face"]` reads "inputs without a face". A term this table does not hold
    is given as a condition of the input, never by its name."""
    said = []
    for literal in literals:
        term, wanted = (literal[1:], False) if literal.startswith("!") else (literal, True)
        said.append(GATE_TERMS[term][0 if wanted else 1] if term in GATE_TERMS else "inputs under another condition")
    return " and ".join(said) or "other inputs"


def measured_auc(benchmark_note: str | None) -> tuple[float, str | None] | None:
    """(AUC, n) of a check's benchmark note ("On vb-1.1 (video/test) this check reached AUC 0.52 (n = 29, 2026-09-17)."), or None."""
    m = _AUC_NOTE.search(benchmark_note or "")
    return (float(m.group(1)), m.group(2)) if m else None


def outside_why(outside: dict | None, benchmark_note: str | None = None, *, figures: bool = True) -> str:
    """Why a check that scored is not counted in this estimate: not in the fitted model, its gate closed and by which
    condition, or no weight in the model, with what the check reached in testing: "at chance" only where its AUC is.
    `figures`: with the AUC and the number of test items (the finding and the page's list; the summary leaves them out)."""
    reason = (outside or {}).get("reason")
    if reason == "not_in_model":
        return NOT_IN_MODEL_WHY
    if reason == "gate":
        return f"its gate is closed: the model uses this check only for {gate_words(list((outside or {}).get('gate') or []))}"
    if reason != "zero_weight":
        return NO_RECORD_WHY
    measured = measured_auc(benchmark_note)
    if measured is None:
        return ZERO_WEIGHT_WHY
    auc, n = measured
    reached = f"AUC {auc:.2f}" + (f", n = {n}" if n else "")
    if abs(auc - 0.5) <= AT_CHANCE + 1e-9:
        return "in testing it was at chance on inputs like this one" + (f" ({reached})" if figures else "")
    return ZERO_WEIGHT_WHY + (f" (in testing it reached {reached})" if figures else "")


def outside_sentence(score_text: str, why: str, *, standalone: bool = False) -> str:
    """The sentence of a check that scored and is not counted. `standalone`: the check's own words stand before it."""
    if standalone:
        return f"Its score of {score_text} is not counted in this estimate: {why}."
    return f"This check scored {score_text} and is not counted in this estimate: {why}."
