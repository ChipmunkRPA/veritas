"""Report engine (ARCHITECTURE §6.1–6.4; D-10, D-25; BUILD_SPEC R-1, R-2, R-3).

- **Engine.** WeasyPrint, pinned exactly (`weasyprint==70.0`, `pydyf==0.12.1`), on Pango (macOS `brew install pango`; Debian
  `libpango-1.0-0 libpangoft2-1.0-0`). `VERITAS_REPORT_ENGINE`: `auto` and `weasyprint` use it when it loads; `reportlab` keeps
  only the v1 PDF (`builder.py`). Without WeasyPrint, v2 reports are refused (503).
- **Templates.** Jinja2 over `templates/` with autoescaping and `StrictUndefined`: `base.html.j2` is the common page system
  (UX §6.1), `profiles/<profile>.html.j2` a profile (D-10), `sections/` the forensic sections. A profile is available when its
  template exists. The CSS (`fonts.css`, generated `tokens.css`, `print.css`) is inlined into the document; figures are inline
  SVG (`figures.py`) and plates and thumbnails are re-encoded store objects as data URIs.
- **Views.** `summary` and `newsroom` render `view.page_view`; `standard` and `forensic` render `forensic.forensic_view` and attach
  a redacted hash commitment as `manifest.json`. Exact reproduction requires the internal record.
- **Determinism** (§6.3): render only from a `ReportSnapshot`; PDF dates, including the attachment's, come from `requested_at`;
  the PDF identifier is `sha256(snapshot_hash + template_set + engine_version)[:32]`; fonts are the bundled static instances.
- **Fetching** (§6.3 rule 4). `RestrictedFetcher` opens `file:` URLs inside `templates/`, `fonts/` and the artefact store, and
  `data:` URLs; everything else is refused and listed in `RenderResult.refused`.
- **Lints** (§6.3 rule 7; RP-01, RP-02). Every render scans the extracted PDF text for banned terms outside quoted values; the
  standard and forensic profiles also measure the lay summary. `RenderResult.lint` carries the outcome; it never changes bytes.
"""
from __future__ import annotations

import hashlib
import html
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import jinja2
import markupsafe

from app.core.canonical import canonical_json
from app.core.config import settings
from app.core.logging import get_logger
from app.reports import figures, forensic, formatting, heatmaps, image_evidence, lint, multimedia_evidence
from app.reports.public_view import public_manifest, public_view
from app.reports.snapshot import FULL_PROFILES, ReportSnapshot, snapshot_hash
from app.reports.view import page_view

log = get_logger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = REPORTS_DIR / "templates"
FONTS_DIR = REPORTS_DIR / "fonts"
TEMPLATE_SET = "veritas-report-2.0.r81"  # r81: recorded image reliability masks and email body coverage; r72: preserve inconclusive decisions and declare check coverage; r71: face review intervals; r70: scoped email and video soundtrack estimates; r69: source-bound sound-content transitions and scoped local article testing; r68: separate local speech calibration and private retired calibration components; r67: consistent customer decision-rule and report identity throughout; r66: scoped visual conclusions and frozen sound-category evidence; r65: customer conclusions, source-linked inferences and service attribution; r64: neutral architecture and fusion identifiers in all generated report configuration fields; r63: self-identifying media plate captions retain IDs and source coordinates across page breaks; r62: source-linked acoustic context and cross-format evidence locators; r61: neutral generated contribution narrative; r60: parent-bound video voice and audio/visual correlation evidence; r59: explicit image candidate eligibility, suppressed regions and raw map commitments; r58: distinguish document forward context from ratio reduction tokens; r57: explicit video audio/channel gaps and document context omissions; r56: video context clip commitments and direction-aware document input evidence; r55: neutral prefixed primary limitation on partial document extraction; r54: neutral historical document ledger and sentence prose; r53: frozen video, audio and document evidence plates and public asset commitments; r52: public localisation limitations; r51: model-free localisation references; r50: frozen measured component masks and 2x2 localisation plates; r49: frozen image evidence plates; r48: neutral check labels and redacted manifest; r47: Chipmunk Verify branding; r46: N-19q (N-20b's read of the reports; supervisor check 48) the sentence of D-76 says what the rule reads: "no check's part in this estimate reached weak support (a factor of 2) toward that answer" (it read "no check gave at least weak support for that on its own" beside a check that taken alone gives weak support and a check outside the estimate at moderate support), with one sentence more where a check taken alone or a check outside the estimate gives that answer weak support or more, saying why that reading does not decide the result; the note of the scale and the headline rule of such a result say the same (finding 1). r45: N-19p (the re-checks of N-19o as built) beside a record the sentence on a configured prior says that it adjusts the statistical checks' estimate (Q6-02); a justification typed on the machine's step is printed after the rationale as "Justification", and "Justification of the departure" and "Deviation" stand only where the conclusion departs (D6-02); a row that the decision does not count reads "not counted", with no bar and no "−0.00", in the breakdown and the ledger, whose direction is "neutral" (D6-05, R10-05). r44: N-20b (the field test under the served rules) the disclosure of the kinds of genuine photograph gives the field test's camera originals called AI as 5 of 52, with its Wilson interval and Fisher's exact test against the benchmark (it read "8 of 52", without either); r43: N-19o (design review D5-01) on the fallback under a trusted declaration the examiner's line and the sign-off section name the step of the declaration ("on the step of the declaration", "1 step from the step of the declaration, toward no indications"; it read "agrees with the machine assessment" on the fifth step and "departs from the machine assessment by 1 step" on the code's own), as under a capture they name the capture record's; the headline rule of a fallback result that a provenance record decides says that the record sets the headline and that the ranges are those of the score (it said that the fallback never issues the strongest headline); (browser QA Q5-01) the ledger row of a mention of AI in a text field says what the finding says ("the term trainedAlgorithmicMedia", "a phrase such as “AI-generated”", the tool named; it read the check's label, "IPTC DigitalSourceType = trainedAlgorithmicMedia"), points neither way and has no P(AI), and the figure "Detector estimates" draws no dot for it; (browser QA Q5-02) under a configured prior "What this means" gives the share once wherever the number of the result, or the statistical checks' number beside a record, is printed ("The estimate is adjusted for the share of AI-generated or manipulated inputs that this server is set to assume, 90 %."; it stood only beside the sentences of D-76 and D-77 and in the limitations); (code review R9-04) on the fallback "it points the other way" reads the printed whole points (a score of 0.645 printed 65 beside a capture and said nothing more); (design review D5-04) the reason label of a check whose row the decision does not count reads "not counted" (it read "little effect" beside a breakdown that says the row is not counted), and a contribution under 0.005 reads "less than 0.01 log-odds", never "0.00"; (code review R9-02) "What this means" knows a provenance record by its family, so that a marker that was not applied is never "a check of part of the file … not counted in this estimate"; r42: N-19n (design review D3-02, D3-03 d, D3-04) under a record that decides, the one disclosure of a fallback result says that the statistical checks are not validated ("Nothing in this result is validated" stood under a validated record), and the foot of the one-page reports says that a provenance record decides the result (it called it "an estimate"); every result of the fallback that has a score is headed "How the score was reached" and says "the score" (never "the estimate"), and a record that decides alone "How the result was reached"; "Draft opinion, not signed" stands over a draft opinion only; and the rest of N-19n under the same set (code review R9-06: the note named its first slice only): (D3-08, Q3-05, Q3-06, the browser QA's lead 3, the rest of Q2-14) words beside a record or a score: a caveat, a reason and a step said in a reader's words, "the score" on the fallback; (Q4-01) a result whose step for the findings failed says so in their place, and a mention of AI in a text field has its own finding; (D3-07) a marker's finding has a record's mark and word ("Self-declared: …", "Self-declared, not applied: …"), the provenance findings that did not decide a result are listed as records and not as checks, Content Credentials that do not validate have a title of their own, and unsigned metadata is worded by what the file holds; (D3-06) the scope table of a result that is not assessable without a fusion model claims no check and no score for a question; (D3-09, D3-10, D3-11, Q3-02) outside what was validated the breakdown lists what the checks returned and nothing else, a row of little effect is thin, the scale has no note, the headline no "the checks disagree", and the summary says why the input is outside, also where the cap changed no code; (the code reviewer's read of N-19m's draft) the record's headline on the image fallback, one check beside a record, and the unit in the appendix; (R8-02; check 29) the note under the scale names the answer's part, D-76's three sentences beside a record open alike, and under a configured prior a result that D-77 took names the part of the scale and the configured share; (D4-01) under a trusted capture and a trusted declaration the examiner's line and the sign-off section name the step of the record; (D4-02 … D4-05) beside a record the statistical checks' number says whose it is and whether they point the other way, the caveat with that number is shown in one form, and the sentence on what an unsupported estimate comes from counts the checks of little effect in either direction. r41: N-19m (code review R7-01 … R7-03; design review D3-05; D-76 beside a record) the statistical number beside a record that decides says what the statistical checks alone decided, with its unit: "statistical checks alone: 90 out of 100, they alone do not decide" and "The statistical checks alone do not decide: their estimate is 7–56 out of 100." where they alone are inconclusive (the estimate stood bare beside a capture's cap, a declaration and a marker's floor, and as "estimate 10 out of 100" under a capture that did not set the number), "98 out of 100" where they decide (it read "98" beside "10 out of 100", and "estimate 90." in the declaration's sentence); D-76's sentence in its form beside a record is said wherever their number is shown, also where the record set the result's number; the limitation with the statistical number reads "Statistical estimate without the provenance record: 83–96 out of 100; the statistical checks alone do not decide." where they do not decide (it read "… before the provenance override: 90%."); r40: N-19m (design review D3-01; D-15) a result that a trusted capture decides is headed as the record's: the headline is "The file’s Content Credentials record a capture of these bytes" wherever the headline of a result is printed (it was the headline of the decision code, so the result read as a clean statistical one), on the first step of the scale on both fusion paths; the line of the one-page reports begins with the basis ("Documentary · Content Credentials (capture record) · …") and holds neither an agreement nor a count of checks, and the full reports print no sentence on the agreement among the checks, as under a declaration; what the result does not mean is said of the record (what the capture device declared; a screen or a print can be captured; a signing key can be misused) and the next step is to check who signed the capture record, with a sentence more where the statistical checks alone point the other way; r39: N-19l (code review R6-01) a result stored before its decision named the record that decides is read as a fresh one: the record, its kind and its signals are worked out from the stored signals by the functions the runtime calls (the driving hypothesis asked first), so the record's finding is among the reasons (it stood in none of the four reports under a declaration outside what was validated, a capture and a watermark message), stored Content Credentials that are no declaration read in the words of their signer, and a watermark message is no check that scored; the findings of the record that decides stand first among the reasons and the key findings, with the ids they were stored with (a stored marker's or capture's finding ranked behind the checks that entered); r38: N-19k (browser QA Q2-05) a one-page report that runs over its page is set again in up to three tighter steps until it fits, with the same content (the issued summary of a departing opinion with its justification, the custodian's line and a miniature ran to a second page of two lines); (the PDF words of both reviews) the rest of the listed words: an error-rate context is said in words where a stored finding or "In testing" names it ("portrait inputs", "text-to-video inputs"); Methods says "validation record" and "release check" (it read "dossier" and "release gate"); the watermark record of the provenance section is a sentence ("the public watermark message of … was not found (30 of 48 bits match; at least 44 are needed)"; it read "present False; scheme dwtDct"); the false alarms of a check print the model card's modes in client wording only (a mode in the benchmark's shorthand, "CFE 10 of 30 in both", stays on the card); a metadata tag without a name is "unnamed tag 41733"; a number, a date and a tag name are not broken over lines in a table (the custody sequence "111 / 8"); the contact sheet numbers its samples as samples ("00:01.0 · sample 5"; it read "frame 5"); a figure names who drew it in words: a step of the run by its name in the analysis record ("From the step “Temporal checks”"; it read "From temporal", "From the render stage") and a check by the name the ledger gives it, never its id; a stored title or caption reads "(hand-set weights)" where it named the build's "legacy fuser"; (browser QA Q2-14, "PDF rendering") words of the build: the custody log says a sign-off decision in words ("does not concur"; it read "non concur"), the one-page reports say "taken in by Veritas on …", and the summary's "Main limitation" is not headed "Limiting condition" a second time; (browser QA lead 3) checks that could not run on this server are said once under the headline and once with their names: the reasons of a result that is not assessable do not repeat the count where the decision's reason names the checks, and the limitations hold the result's one caveat for them without a sentence of the report's own beside it (n checks gave n + 1 limitations); (design re-review D2-02, P1; browser QA Q2-14) the scope table of a result that is not assessable says of a question whose checks ran without a score that can be used that they ran and that their values are shown for inspection only (it read "no check for this question produced a result" over the count of the checks that did); a result whose instance holds a fusion model that the analysis did not use says that none was used, not that none is loaded; a finding that points away from AI generation is no key finding of a result that is not assessable; (D2-03) the breakdown of a result outside what was validated lets no estimate be read off it: the chapter is headed "What the checks returned" with one sentence on why no estimate is stated, every row is a bar from zero on an axis in log-odds, the starting row holds its log-odds and no percent, and no column or caption gives a running total (the rows stood on the running total under an axis labelled in probability); the chapter of a score of hand-set weights is headed "How the score was reached"; the calibration sentence of a result outside what was validated says that the calibration does not cover the input; (browser QA Q2-12) under a clean headline "What this means" gives a check that scored and is not counted one sentence with its score and why, from the record of that check for the input ("A check of the sound track (F4) scored 0.91 and is not counted in this estimate: in testing it was at chance on inputs like this one. …"; it read "The sound track was flagged by a check that carries no weight in this estimate"), and in the temporal chapter the lane of such a check is drawn as an inspection lane is and its segments are not listed, with one line that says so; r37: N-19j (design re-review D2-01, D2-10, D2-11; browser QA Q2-02, Q2-03, Q2-08) the record of a review: each signer has a statement of his own followed by his own signature line, on one page (the forensic report printed the examiner’s declaration over both lines; the standard report prints neither statement); the running header of a report that is not issued says why and keeps the word Draft ("Draft · second review outstanding", "Draft · second review: does not concur"), and a refusal stands under the examiner’s opinion wherever that is printed at headline size; a sign-off decision is printed with its object ("concurs with the machine assessment", "does not concur with the examiner’s opinion"), and a sign-off beside a result that is not assessable as an opinion recorded without a comparison; a revision that the reviewer passed with changes says so in its review state, with the changes asked for; the review status line says "not reviewed by an examiner"; r36: N-19i (D-76; code review R5-01) a result that is inconclusive because no check gives at least weak support for the answer of its estimate says so where "What this means" stands, with where the estimate comes from, and neither the note of the scale nor the headline rule says of it that the headline follows the calibrated levels; the headline rule of every calibrated result names the check that a headline of one answer needs; the sentence on how many checks entered the fusion model counts the checks the decision counts, as the number line and the count "in the estimate" do (it counted every row of a check, a row of zero and the row of a check that declares H3 among them), and says of a row that is published and not counted that it is shown and not counted; r35: N-19h (E-19, E-20, E-21) a result that a provenance record decides is presented as that record's: the headline and "What this means" come from the record the decision applied (a capture beside a marker is the capture's; a marker is worded by its kind), the record's finding stands under the reasons, also outside what was validated beside the note on the checks; a number that the record set is said in words with what set it and the statistical checks' own beside it, never as an estimated likelihood, a score of hand-set weights or a marker on the scale; outside what was validated the statistical number is printed beside no record; where a record decides alone section 3 prints no sentence of a fusion, the breakdown is a sentence and the agreement reads "no statistical estimate"; the calibration sentence names what is calibrated, and the fallback's headline rule says that a provenance record acts after it; r34: N-19f (code review R4-05) a result that is not assessable because the fusion model gives the checks that scored no weight for the input has words of its own: the line under the headline, the stage and the rows "Fusion" and "Calibration" of Methods, the headline rule and the scope table say that, not that no check gave a score or that the fusion was not reached; the findings of the checks that scored are listed as checks outside the fused estimate and never as reasons, and the one-page profiles say where they are listed; r33: N-19d one word for the role ("Examiner’s opinion", "acknowledged by the examiner", "a qualified examiner", "an examiner’s review"; the review status line of D-25 keeps its words) and the review of a fusion model as its effective state ("pending human review", or "reviewed by <name> on <date>" from the review record; it read "automated fit — pending human review"), "1 internal note is listed without its text", and on the one-page profiles the sentence of a check's score points at the full report, not at a ledger the page does not have, and a validation slice is said in words ("JPEG quality 90 or more, shorter side over 1,024 px, face under 120 px"), as are the keys of the condition profile; r32: N-19c (design review DS-18; browser QA "PDF rendering") the structure of the reports: what is drawn is a figure and a table is a table, each numbered from 1 in the order of print, which is the stored order of the result (it fell back to the exhibit id, so a re-run renumbered the figures; a video report ran Fig. 2, 4, 9 … and its contact sheet was "Fig. 35–13"); the table of a map's regions follows its map; every reference to an exhibit carries the label the exhibits chapter prints; the one-page profiles carry the header centre of the forensic profile, with the classification; r31: N-19 (D-69; design review DS-03, DS-05) an unsigned report does not read as cleared: a redacted report is headed "Redacted copy", never "For publication"; until an examiner sign-off exists the header of every profile carries "Draft · not reviewed" (the newsroom profile "… · not for publication", once more under its title), and the sign-off block is "Draft opinion, not signed" without the declaration and the signature lines; r30: N-18 (browser QA P1-1; D-64) under a clean headline "What this means" names a check outside the estimate that flagged the file ("The sound track was flagged by a check that carries no weight in this estimate …"); Methods, "Headline rule", says that the two strongest headlines need two checks that scored, and for the fallback that a result without a check of the content is not assessable; r29: N-17 (D-68, design review DS-10) a result that is not assessable speaks of no estimate and no probability: its one sentence is "No estimate was formed: …"; the chapter is "How the result was reached" with that sentence alone; Methods, the limitations, the glossary, the cover, the closing line and the reproduction note have their form for such a result; a lane figure is a "window score", captioned as for inspection only; r28: N-17 (browser QA P1-5) the scope table counts the checks that are in the estimate and says "Not addressed by the estimate" of a question whose checks all carry no weight; the disclosure of the video lanes names the lanes whose checks are not fitted (lip-sync, the face-swap check of a whole track) and the unvalidated segments; r27: N-17 (D-63) the headline rule in words: Methods says that the headline follows the two calibrated levels, with what each level measured on the test split, in place of the table of fixed ranges; the scale of a calibrated result says what its marker is; the strength words of a check in the estimate are factors on the odds; r26: N-17 (D-63) one vocabulary: direction and strength of a check that entered the estimate come from its contribution (little effect below ln 2), in the counts, the agreement label, the key findings, the ledger and the strip; an AI headline that no check supports at least weakly says where the estimate comes from; r25: N-17 (D-63) the starting row is the estimate with every check at its neutral point, named as that in "How the estimate was reached", the glossary, Methods and on the one-page reports; a result published earlier keeps "Model starting point"; r24: N-14a checks skipped for a reason of the server (its memory budget, model files not installed) are said as the server's: in "What this means", the reasons, the next step, the limitations and the detector list; r23: N-12b the icon arrays of "In testing" count a half upwards, as their captions do; r22: N-10b the summary of a capped result gives no "Main reasons" (its findings were not validated for the input); a half is rounded upwards; r21: N-7 the number a result shows (no number or marker for a capped result, a score for the fallback, body text for an inconclusive or one-check result), the cap reason, the fallback's one disclosure; r20: N-4 regions and overlay marks in oriented original px; r19: N-3 invalid review records, authentication method, custodian hash mismatch; r18: N-2 status line
PROFILES = ("summary", "standard", "forensic", "newsroom")
STYLESHEETS = ("fonts.css", "tokens.css", "print.css")
INSTALL_HINT = "install Pango (macOS: brew install pango; Debian: apt install libpango-1.0-0 libpangoft2-1.0-0)"


class EngineUnavailable(RuntimeError):
    """WeasyPrint cannot be loaded, or the configured engine cannot produce v2 reports."""


class ProfileUnavailable(LookupError):
    """This build has no template for the requested profile."""


@dataclass(frozen=True)
class RenderResult:
    pdf: bytes
    page_count: int
    engine_version: str
    refused: tuple[str, ...]
    lint: dict[str, Any] = field(default_factory=dict)


_lock = threading.Lock()
_weasyprint: Any = None
_load_error: str | None = None
_env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)), autoescape=jinja2.select_autoescape(("html", "j2")),
                          undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True)
_env.filters.update(formatting.FILTERS)


def _dlopen_with_fallback(original: Any, directories: list[Path]) -> Any:
    """`FFI.dlopen` that retries a bare library name under the configured directories (dyld does not search Homebrew)."""

    def dlopen(self: Any, name: Any, flags: int = 0) -> Any:
        try:
            return original(self, name, flags)
        except OSError:
            if not isinstance(name, str) or os.sep in name:
                raise
            leaf = name if name.endswith(".dylib") else f"{name}.dylib"
            for directory in directories:
                candidate = Path(directory) / leaf
                if candidate.is_file():
                    return original(self, str(candidate), flags)
            raise

    return dlopen


def load_weasyprint() -> Any:
    """The WeasyPrint module. Raises EngineUnavailable (with the install hint) when it cannot load or is not configured."""
    global _weasyprint, _load_error
    if settings.report_engine == "reportlab":
        raise EngineUnavailable("VERITAS_REPORT_ENGINE=reportlab produces the v1 PDF only; v2 reports need WeasyPrint")
    with _lock:
        if _weasyprint is not None:
            return _weasyprint
        if _load_error is not None:
            raise EngineUnavailable(_load_error)
        try:
            import cffi

            original = cffi.FFI.dlopen
            if sys.platform == "darwin":
                cffi.FFI.dlopen = _dlopen_with_fallback(original, list(settings.report_library_dirs))
            try:
                import weasyprint
            finally:
                cffi.FFI.dlopen = original
            from app.engine.media import refuse_truncated_images

            refuse_truncated_images()  # weasyprint.images turns Pillow's LOAD_TRUNCATED_IMAGES on at import (L-1)
        except (ImportError, OSError) as e:
            _load_error = f"WeasyPrint could not be loaded ({type(e).__name__}); {INSTALL_HINT}"
            log.warning("reports.engine_unavailable", error=str(e))
            raise EngineUnavailable(_load_error) from None
        _weasyprint = weasyprint
        return weasyprint


def engine_version() -> str:
    return f"weasyprint {load_weasyprint().__version__}"


def available_profiles() -> tuple[str, ...]:
    return tuple(p for p in PROFILES if (TEMPLATES_DIR / "profiles" / f"{p}.html.j2").is_file())


def require_profile(profile: str) -> None:
    if profile not in available_profiles():
        raise ProfileUnavailable(f"This build has no {profile} report template.")


def pdf_identifier(snapshot_sha256: str, version: str) -> bytes:
    return hashlib.sha256(f"{snapshot_sha256}{TEMPLATE_SET}{version}".encode()).hexdigest()[:32].encode("ascii")


def plates(snapshot: ReportSnapshot, *, thumbnails: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    """Plate key → data URI: `thumb:<finding id>` thumbnails, exhibit plates, face-track stills and the source image."""
    thumbnails = heatmaps.finding_thumbnails(snapshot) if thumbnails is None else thumbnails
    images = {f"thumb:{finding_id}": detail["uri"] for finding_id, detail in thumbnails.items()}
    exhibits = [e for e in snapshot.result.get("exhibits") or [] if isinstance(e, dict)]
    heat_ids = {str(e.get("id")) for e in exhibits if e.get("kind") == "heatmap"}
    still_ids = {str(e.get("id")) for e in exhibits if e.get("kind") == "image" and e.get("role") == "still"}  # contact sheet (L-4)
    wide_ids = {str(e.get("id")) for e in exhibits if forensic.is_spectrogram(e)}  # over the time axis (L-7)
    for key, sha256 in sorted(snapshot.exhibit_images.items()):
        # Scalar and mask bytes are data, never standalone RGB previews or an unbound overlay.
        if key in heat_ids or key.endswith(("#scalar", "#reliability")):
            continue
        if key == "source" and (sha256 != snapshot.media.sha256 or snapshot.analysis.kind != "image"
                                or (snapshot.result.get("input") or {}).get("sha256") != sha256):
            continue
        box = forensic.STILL_BOX if "#still-" in key or key in still_ids else forensic.WIDE_BOX if key in wide_ids else forensic.PLATE_BOX
        uri = figures.plate(sha256, box, photo=key == "source")
        if uri is not None:
            images[key] = uri
    for exhibit in exhibits:
        if exhibit.get("kind") != "heatmap":
            continue
        bound = heatmaps.scalar_binding(snapshot, exhibit)
        if bound is None:
            continue
        eid = exhibit["id"]
        uri = figures.scalar_map_plate(box=forensic.PLATE_BOX, **bound)
        if uri is not None:
            images[eid] = uri
        size = heatmaps.overlay_source_size(snapshot, exhibit)
        if size is None or uri is None:
            continue
        source = snapshot.media.sha256
        overlay = figures.scalar_overlay_plate(source_sha256=source, source_size=size, box=forensic.PLATE_BOX, **bound)
        if overlay is not None:
            images[f"{eid}#overlay"] = overlay
            if bound["reliability_sha256"]:
                reliability = figures.reliability_plate(bound["reliability_sha256"], source, forensic.PLATE_BOX)
                if reliability is not None:
                    images[f"{eid}#reliability"] = reliability
    return images


SOURCE_THUMB_BOX = (472, 354)  # 40 × 30 mm at 300 dpi: the examined picture on a one-page report (N-19d)


def source_plate(snapshot: ReportSnapshot, box: tuple[int, int]) -> str | None:
    """The examined picture as a plate, under the rule of `plates`: only an image whose stored bytes are those of the media
    and of the result's input. None for another kind of media, a missing or altered object."""
    sha256 = snapshot.exhibit_images.get("source")
    if not sha256 or sha256 != snapshot.media.sha256 or snapshot.analysis.kind != "image" \
            or (snapshot.result.get("input") or {}).get("sha256") != sha256:
        return None
    return figures.plate(sha256, box, photo=True)


def view_of(snapshot: ReportSnapshot, *, version: str) -> dict[str, Any]:
    withheld = snapshot.report.profile == "newsroom" or snapshot.report.redacted
    thumbnails = {} if withheld else heatmaps.finding_thumbnails(snapshot)
    if snapshot.report.profile in FULL_PROFILES:
        out = forensic.forensic_view(snapshot, {} if withheld else plates(snapshot, thumbnails=thumbnails), engine_version=version,
                                    template_set=TEMPLATE_SET, thumbnail_details=thumbnails)
    else:
        images = {key: value["uri"] for key, value in thumbnails.items()}
        # Redacted reports withhold source pictures, including the new source crops.
        if not (snapshot.report.profile == "newsroom" or snapshot.report.redacted):
            source = source_plate(snapshot, SOURCE_THUMB_BOX)
            if source is not None:
                images["source"] = source
        out = page_view(snapshot, images=images, thumbnail_details=thumbnails)
    out["image_evidence"] = image_evidence.view(snapshot)
    out["evidence"] = multimedia_evidence.view(snapshot)
    return out


def _slices_in_words(value: Any, quoted: tuple[str, ...] | None = None) -> Any:
    """Every string of a report's view as a report prints a stored sentence (`formatting.stored_words`): the slice ids of
    the build in words. Drawn markup (figures) is left as it is.

    With `quoted` (the values of the snapshot that hold a percent sign and are quoted, not written: a file name, a detector's
    observation, an examiner's text), a sentence also gets the one percent format ("83 %"). A string that holds one of those
    values keeps its percent signs as recorded, and so does a string that is no sentence (a number with its sign positions a
    mark on a figure)."""
    if isinstance(value, markupsafe.Markup):
        return value
    if isinstance(value, str):
        said = formatting.stored_words(value)
        if quoted is not None and " " in said and any(ch.isalpha() for ch in said) and not any(q in said for q in quoted):
            said = formatting.percent_spaced(said)
        return said
    if isinstance(value, dict):
        return {key: _slices_in_words(item, quoted) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_slices_in_words(item, quoted) for item in value)
    return value


def _version_or_note() -> str:
    try:
        return engine_version()
    except EngineUnavailable:
        return "report engine not loaded"


def render_html(snapshot: ReportSnapshot, *, embed_fonts: bool = True, v: dict[str, Any] | None = None,
                _public: bool = False) -> str:
    """The report document. `embed_fonts=False` (the browser preview) leaves out the file-URL font faces."""
    require_profile(snapshot.report.profile)
    css = "\n".join((TEMPLATES_DIR / name).read_text(encoding="utf-8") for name in STYLESHEETS if embed_fonts or name != "fonts.css")
    template = _env.get_template(f"profiles/{snapshot.report.profile}.html.j2")
    # N-19d: no slice id of the build in a report, also where it stands inside a sentence the engine stored
    # (and one percent format in its sentences; what the report quotes keeps its own)
    quoted = tuple(q for q in dict.fromkeys(forensic.quoted_values(snapshot)) if q and formatting.holds_percent(q))
    raw = v if v is not None else view_of(snapshot, version=_version_or_note())
    presentation = raw if _public else public_view(snapshot, raw)
    v = _slices_in_words(presentation, quoted)
    # New source excerpts and sample/PTS anchors are verbatim frozen evidence,
    # not generated prose to rewrite for house style or internal vocabulary.
    if "evidence" in presentation:
        v["evidence"] = presentation["evidence"]
    if snapshot.report.profile in FULL_PROFILES and v.get("glossary"):
        # N-19c: the glossary holds the terms this report uses. The document is rendered once without it, and its text read
        # (tags and what they carry left out, so no image data is taken for a word)
        body = re.sub(r"<[^>]*>", " ", template.render(s=snapshot, v={**v, "glossary": []}, css="", template_set=TEMPLATE_SET))
        v = {**v, "glossary": forensic.used_glossary(v["glossary"], html.unescape(body))}
    return template.render(s=snapshot, v=v, css=css, template_set=TEMPLATE_SET)


def _restricted_fetcher(roots: tuple[Path, ...]) -> Any:
    from weasyprint.urls import URLFetcher

    class RestrictedFetcher(URLFetcher):
        def __init__(self) -> None:
            super().__init__(allowed_protocols=("file", "data"), allow_redirects=False)
            self.roots = tuple(root.resolve() for root in roots)
            self.refused: list[str] = []

        def fetch(self, url: str, headers: Any = None) -> Any:
            parsed = urlparse(url)
            if parsed.scheme == "data":
                return super().fetch(url, headers)
            path = Path(url2pathname(parsed.path)).resolve() if parsed.scheme == "file" and parsed.netloc in ("", "localhost") else None
            if path is None or not any(path.is_relative_to(root) for root in self.roots):
                self.refused.append(url)
                log.warning("reports.fetch_refused", url=url[:200])
                raise ValueError("report resources are limited to the templates, the bundled fonts and the artefact store")
            return super().fetch(url, headers)

    return RestrictedFetcher()


def pdf_text(pdf: bytes) -> str:
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def report_lint(snapshot: ReportSnapshot, pdf: bytes, v: dict[str, Any]) -> dict[str, Any]:
    text = pdf_text(pdf)
    source_quotes = multimedia_evidence.quoted_values(snapshot)
    from app.engine.service_policy import standard_service_policy

    # Exact trusted disclosure wording includes explicit negations such as
    # "not as proof". Exempt the fixed copy, never arbitrary nearby text.
    policy = standard_service_policy()
    disclosures = [policy[k] for k in ('confidence_notice', 'court_notice', 'private_engagement_notice')]
    out: dict[str, Any] = {"banned_terms": lint.vocabulary(text, [*forensic.quoted_values(snapshot), *source_quotes, *disclosures]),
                           "internal_terms": lint.internal_terms(text, [*forensic.identifier_values(snapshot), *source_quotes])}
    if snapshot.report.profile in FULL_PROFILES:
        summary = v["summary"]["lint"]
        out.update(lay_summary_words=summary.words, lay_summary_grade=summary.grade, lay_summary_ok=summary.ok)
    return out


ONE_PAGE_FITS = 3  # the tighter settings of a one-page report (`print.css`)


def render_pdf(snapshot: ReportSnapshot) -> RenderResult:
    weasyprint = load_weasyprint()
    from weasyprint.text.fonts import FontConfiguration

    version = f"weasyprint {weasyprint.__version__}"
    v = public_view(snapshot, view_of(snapshot, version=version))
    fetcher = _restricted_fetcher((TEMPLATES_DIR, FONTS_DIR, settings.objects_dir))
    html = weasyprint.HTML(string=render_html(snapshot, v=v, _public=True), base_url=TEMPLATES_DIR.as_uri() + "/", url_fetcher=fetcher)
    document = html.render(font_config=FontConfiguration())
    if (snapshot.report.profile not in FULL_PROFILES and not (v.get("image_evidence") or {}).get("packets")
            and not (v.get("image_evidence") or {}).get("inspection_checks")
            and not ((v.get("image_evidence") or {}).get("copy_similarity") or {}).get("pairs")
            and not ((v.get("image_evidence") or {}).get("copy_translation") or {}).get("pairs")
            and not (v.get("evidence") or {}).get("packets")):
        # N-19k (browser QA Q2-05): a one-page report fits its page. One that runs over (an issued revision with a departing
        # opinion, its justification, the custodian's line and a miniature ran to a second page of two lines) is set again,
        # one step tighter at a time (`print.css`, `.fit-1` … `.fit-3`); the content is the same at every step
        for fit in range(1, ONE_PAGE_FITS + 1):
            if len(document.pages) <= 1:
                break
            html = weasyprint.HTML(string=render_html(snapshot, v={**v, "fit": fit}, _public=True), base_url=TEMPLATES_DIR.as_uri() + "/", url_fetcher=fetcher)
            document = html.render(font_config=FontConfiguration())
    requested = snapshot.report.requested_at.astimezone(UTC)
    stamp = requested.strftime("%Y-%m-%dT%H:%M:%SZ")
    document.metadata.created = document.metadata.modified = stamp
    options: dict[str, Any] = {"pdf_identifier": pdf_identifier(snapshot_hash(snapshot), version)}
    if snapshot.report.profile in FULL_PROFILES:
        manifest = canonical_json(public_manifest(snapshot, version, TEMPLATE_SET)).decode("utf-8")
        options["attachments"] = [weasyprint.Attachment(string=manifest, name="manifest.json", description="Redacted record hash commitments",
                                                        created=requested, modified=requested)]
    pdf = document.write_pdf(**options)
    return RenderResult(pdf=pdf, page_count=len(document.pages), engine_version=version, refused=tuple(fetcher.refused),
                        lint=report_lint(snapshot, pdf, v))
