"""Document pipeline stages (ARCHITECTURE §3.1–3.2): ingest → decode → detect → fuse → synthesise → render → publish.

- **decode** extracts the text (PDF, DOCX, TXT, MD, HTML) and segments sentences exactly as v0.3, each in a child process
  that is killed at `document_decode_timeout_s` (N-10, N-14b). The output is persisted, so a resumed run reuses the same
  sentence offsets. Fewer than 15 words fails the analysis with the v0.3 message.
- **detect** runs the enabled text detectors over the sentences.
- **fuse** is the legacy fuser. It rebuilds `DocumentScores` from the detector runs, applies the v0.3 per-sentence fusion,
  smoothing and document verdict (`text/pipeline.assemble_document`), and publishes an uncalibrated fallback `Decision`
  (§4.4). The `condition` stage arrives with M-2.
- **render** writes `sentence_scores.png` for the Folio UI and the v1 PDF; **publish** stores the v1 document view.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import uuid
from typing import Any

from app.core.canonical import canonical_hash
from app.detectors.text.adapters.stylometry import LEXICON_TITLE, SPANS_TITLE
from app.detectors.text.extract import EXTRACTION_VERSION, Document
from app.detectors.text.pipeline import (
    DocumentScores,
    assemble_document,
    check_document_length,
    check_document_size,
    decode_document,
    sentence_plot,
)
from app.detectors.text.segment import Sentence
from app.engine.detector import ON_THIS_SERVER, Modality, StageOutputs, of_the_server
from app.engine.registry import TEXT_PRIMARY_DETECTORS
from app.engine.runner import StageContext
from app.engine.stages import common
from app.schemas.evidence import AnalysisResultV2, Decision
from app.schemas.legacy import FALLBACK_CAVEAT

PIPELINE_VERSION = "doc-2.0-p20"  # p20: frozen paired-token contexts and opposing raw-ratio evidence; p19: local OCR/source mapping; p17: the caveat of a failed step that writes the findings is said in words ("The step that writes the findings failed: this result holds no findings from it."; it named the stage in backticks) (N-19n; browser QA Q4-01); p16: the exhibit of the combined scores is titled "(hand-set weights)", not by the build's name "(legacy fuser)"; the checks that could not run on this server are one caveat with their names, in place of one caveat for each and of their ids in the caveat of a reduced profile; a caveat names a check as the page does, a step of the run and a provenance record in words, never a detector id, a stage name or the value of a kind (browser QA leads 3 and 4, design re-review D2-06); each check that scored and is not counted in the estimate carries the fusion's record of why (`outside_estimate`: not in the model, its gate closed, a weight of zero; browser QA Q2-12); a result of the fallback on an instance that holds a fusion model says that none was used for the analysis, not that none is loaded, and the v1 view of a result outside what was validated gives the reason of its cap (N-19k; browser QA Q2-14 and part 2); p15: a not-assessable result has no caveat or reason that speaks of an estimate or a probability (N-17, D-68); p14: the result names the text classifier that scored the document (N-5, 2026-09-30)
DOCUMENT_MAX_UNITS = 4000  # 1 500 sentence windows (segment.MAX_UNITS) plus chunks
PRIMARY_DETECTORS = TEXT_PRIMARY_DETECTORS  # v0.3 used the fine-tune whenever it was installed
SECONDARY_DETECTOR = "txt.hf.hello_simpleai"
BINOCULARS_DETECTOR = "txt.binoculars"
STYLOMETRY_DETECTOR = "txt.stylometry"
FUSER_ID = "legacy.document"
EXHIBIT_STAGES = ("detect", "fuse")
STYLE_METRIC_KEYS = ("words", "sentences", "mean_sentence_len", "sentence_len_cv", "mattr", "lexicon_weighted_per_1000",
                     "contrast_per_1000", "hedges_per_1000", "transition_opener_pct", "triads_per_1000", "emdash_per_1000")
# D-62 (QA P1-10): the result names the primary classifier that scored the document
CLASSIFIER_NOTE = "Text classifier that scored this document: {name}."
FINE_TUNE_ABSENT_NOTE = ("The fine-tuned text classifier ({fine_tune}) is not installed on this server, so the base classifier "
                         "({name}) scored this document.")
NO_LEARNED_EVIDENCE = ("No learned text detector produced a usable document result. Stylometric patterns and derived sentence "
                       "consensus are diagnostic only and cannot establish an AI-authorship verdict.")
SPANS_CAPTION = ("Per-sentence p(AI) from the legacy fuser after contextual smoothing: uncalibrated, not a frequency "
                 "estimate. Sentences under three words are not scored.")


def decode(ctx: StageContext) -> dict[str, Any]:
    path = common.media_path(ctx)
    doc, sents = decode_document(path, path.suffix)
    words = check_document_length(doc, sents)
    artifacts = []
    for digest, encoded in doc.source_rasters.items():
        content = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Frozen OCR input does not match its source commitment")
        obj = ctx.store.put_bytes(content)
        artifacts.append({"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"ocr:{ctx.analysis_id}:{digest}")),
                          "sha256": obj.sha256, "size_bytes": obj.size_bytes, "media_type": "image/png",
                          "role": "preview", "created_by": "document-ocr/1"})
    return {"text": doc.text, "paragraphs": [[int(a), int(b)] for a, b in doc.paragraphs], "meta": common.json_safe(doc.meta),
            "sentences": [dataclasses.asdict(s) for s in sents], "words": words, "source_map": doc.source_map,
            "artefacts": artifacts}


def current_decode_output(output: Any) -> bool:
    """Do not reuse pre-OCR extraction; its absent page mapping cannot be invented."""
    if not isinstance(output, dict) or output.get("meta", {}).get("extraction_version") != EXTRACTION_VERSION:
        return False
    hashes = {artifact.get("sha256") for artifact in output.get("artefacts", [])}
    return all(page.get("raster", {}).get("sha256") in hashes for page in output.get("meta", {}).get("ocr", {}).get("pages", [])
               if page.get("raster"))


def document_input(ctx: StageContext) -> tuple[Document, list[Sentence]]:
    d = ctx.outputs["decode"]
    doc = Document(d["text"], [(int(a), int(b)) for a, b in d["paragraphs"]], dict(d.get("meta") or {}), list(d.get("source_map") or []))
    return doc, [Sentence(**s) for s in d["sentences"]]


def condition(ctx: StageContext) -> dict[str, Any]:
    """Revalidate cached extraction before inference or reuse of downstream results."""
    from app.engine.stages import condition as input_condition

    doc, _ = document_input(ctx)
    check_document_size(doc)
    return input_condition.document(ctx)


def detect(ctx: StageContext) -> dict[str, Any]:
    return common.run_detectors(ctx, (Modality.TEXT,), outputs=StageOutputs(document=ctx.outputs["decode"]),
                                max_units=DOCUMENT_MAX_UNITS)


# --- legacy fuser ------------------------------------------------------------------------------------------

def _units(entry: dict[str, Any], prefix: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for u in entry["unit_scores"]:
        head, _, index = u["unit_ref"].partition(":")
        if head == prefix and index.isdigit():
            out[int(index)] = u["raw"]
    return out


def _creator_exhibit(detected: dict[str, Any], creator: str, kind: str, title: str) -> dict[str, Any] | None:
    return next((x for x in detected.get("exhibits", []) if x["created_by"] == creator and x["kind"] == kind and x["title"] == title),
                None)


def _classifier_note(primary: dict[str, Any], detected: dict[str, Any]) -> str:
    """Which primary classifier served (D-62): the base classifier is named as standing in when the fine-tune was enabled and
    its files are not installed."""
    fine_tune_id = PRIMARY_DETECTORS[0]
    fine_tune = next((e for e in detected.get("detectors", []) if e["detector_id"] == fine_tune_id), None)
    if (primary["detector_id"] != fine_tune_id and fine_tune is not None and fine_tune["status"] == "skipped"
            and "not installed" in (fine_tune.get("reason") or "")):
        return FINE_TUNE_ABSENT_NOTE.format(fine_tune=fine_tune["name"], name=primary["name"])
    return CLASSIFIER_NOTE.format(name=primary["name"])


def document_scores(detected: dict[str, Any], n_sentences: int) -> tuple[DocumentScores, dict[str, str], list[str]]:
    """Rebuild the v0.3 detector outputs from detector runs → (scores, {v1 signal id: detector id}, caveats)."""
    ok = {e["detector_id"]: e for e in detected.get("detectors", []) if e["status"] == "ok" and e["aggregate"] is not None}
    scores = DocumentScores(n_sentences=n_sentences)
    sources: dict[str, str] = {}
    notes: list[str] = []

    primary = next((ok[d] for d in PRIMARY_DETECTORS if d in ok), None)
    if primary is not None:
        windows = _units(primary, "sentence")
        chunks = _units(primary, "chunk")
        scores.p_win = [windows.get(i) for i in range(n_sentences)]
        scores.p_chunks = [chunks[k] for k in sorted(chunks)]
        scores.p_doc_primary = primary["aggregate"]["raw"]
        scores.primary_source = next(iter(primary["model_revisions"]), primary["detector_id"])
        scores.primary_fine_tuned = primary["detector_id"] == PRIMARY_DETECTORS[0]
        sources["classifier_document"] = primary["detector_id"]
        for other in PRIMARY_DETECTORS:
            if other in ok and other != primary["detector_id"]:
                notes.append(f"{ok[other]['name']} also ran; v0.3 fuses one primary classifier, so its result is recorded in "
                             "the detector runs but not fused.")
        notes.append(_classifier_note(primary, detected))
    secondary = ok.get(SECONDARY_DETECTOR)
    if secondary is not None:
        windows = _units(secondary, "sentence")
        scores.p_win2 = [windows.get(i) for i in range(n_sentences)]
        scores.p_doc_sec = secondary["aggregate"]["raw"]
        scores.secondary_source = next(iter(secondary["model_revisions"]), secondary["detector_id"])
        sources["classifier_secondary"] = SECONDARY_DETECTOR
    binoculars = ok.get(BINOCULARS_DETECTOR)
    if binoculars is not None:
        per_sentence = _units(binoculars, "sentence")
        scores.bino_doc = binoculars["aggregate"]["raw"]
        scores.bino_tokens = int(binoculars["aggregate"]["quality"].get("tokens", 0))
        scores.bino_sent = [per_sentence.get(i) for i in range(n_sentences)]
        sources["binoculars_document"] = BINOCULARS_DETECTOR
    style = ok.get(STYLOMETRY_DETECTOR)
    if style is not None:
        metrics: dict[str, Any] = {k: style["metrics"].get(k) for k in STYLE_METRIC_KEYS}
        for key in ("words", "sentences"):
            metrics[key] = int(metrics[key] or 0)
        table = _creator_exhibit(detected, STYLOMETRY_DETECTOR, "table", LEXICON_TITLE)
        metrics["lexicon_hits"] = {str(row[0]): int(row[1]) for row in (table or {}).get("rows", [])}
        scores.metrics = metrics
        scores.p_style_doc = style["aggregate"]["raw"]
        scores.style_reasoning = style["reasoning"]
        spans = _creator_exhibit(detected, STYLOMETRY_DETECTOR, "text_spans", SPANS_TITLE)
        for span in (spans or {}).get("spans", []):
            if 0 <= span["sentence_idx"] < n_sentences:
                scores.flags[span["sentence_idx"]] = list(span["reasons"])
        sources["stylometry_document"] = STYLOMETRY_DETECTOR
    return scores, sources, notes


def not_run_on_server(detected: dict[str, Any]) -> str | None:
    """"2 checks could not run on this server: A (memory budget); B (model not installed)."; None when none was skipped for a
    reason of the server (N-14a)."""
    server = [e for e in detected.get("detectors", []) if of_the_server(e.get("status"), e.get("reason"))]
    if not server:
        return None
    names = "; ".join(f"{e.get('name') or e['detector_id']} ({e['reason']})" for e in server[:6])
    return f"{len(server)} {'check' if len(server) == 1 else 'checks'} {ON_THIS_SERVER}: {names}."


def enforce_extraction_scope(output: dict[str, Any], doc: Document) -> dict[str, Any]:
    """Retain measured excerpt results but withhold a whole-file verdict for unread pages."""
    unchecked = doc.meta.get("unmapped_pages") or []
    if not unchecked:
        return output
    scope = (f"Pages {', '.join(map(str, unchecked))} were not reliably extracted and were not checked. "
             "The recorded scores and excerpts apply only to extracted text. No whole-document authorship conclusion is available.")
    before = output["decision"]
    decision = Decision.model_validate({**before, "code": "not_assessable",
        "verbal": "Partially assessed: some document pages remain unchecked", "reason": scope,
        "probability_ai_calibrated": None, "probability_ai_uncalibrated": None, "ci90": None,
        "band": "uncertain", "confidence": "low", "driving_hypothesis": None, "provenance_record": None,
        "statistical_alone": None, "statistical_estimate_before_provenance": None, "contributions": []}).model_dump(mode="json")
    output = {**output, "decision": decision, "caveats": [*output.get("caveats", []), scope]}
    output["assessments"] = [{**assessment, "applicable": False, "probability": None, "ci90": None,
                              "contributions": [], "lanes": [], "note": scope}
                             if assessment.get("hypothesis") == "H1_whole_content_synthesis" else assessment
                             for assessment in output.get("assessments", [])]
    if isinstance(output.get("v1"), dict) and "verdict" in output["v1"]:
        verdict = common.legacy_verdict(output["v1"]["verdict"], decision)
        verdict.update(summary=decision["verbal"], reasoning=[scope], caveats=[*verdict.get("caveats", []), scope])
        output["v1"] = {**output["v1"], "verdict": verdict}
    return output


def fuse(ctx: StageContext) -> dict[str, Any]:
    """Legacy fuser → uncalibrated fallback Decision (§4.4) plus the v1 document blocks. With a compatible fuser artefact
    the calibrated decision replaces the fallback decision (M-2)."""
    selection = common.select_fusion(ctx, "document")
    doc, sents = document_input(ctx)
    detected = ctx.outputs.get("detect") or {}
    scores, sources, notes = document_scores(detected, len(sents))
    legacy = assemble_document(doc, sents, scores, None)
    spans_id = ctx.exhibits.add(
        "text_spans", title="Sentence scores (hand-set weights)", media_ref="document", caption=SPANS_CAPTION,
        spans=[{"start": x["start"], "end": x["end"], "sentence_idx": x["idx"], "probability": x["p_ai"], "label": x["band"],
                "reasons": x["reasons"]} for x in legacy["sentences"]])
    by_id = {e["detector_id"]: e for e in detected.get("detectors", [])}
    signals = [common.legacy_signal_v2(s, n, entry=by_id.get(sources.get(s["id"], "")), fallback_detector=FUSER_ID,
                                       modality="text", version=PIPELINE_VERSION, family="text",
                                       exhibit_ids=[spans_id] if s["id"] == "sentence_consensus" else None)
               for n, s in enumerate(legacy["signals"], 1)]
    for signal in signals:
        entry = by_id.get(signal["detector_id"], {})
        if source_details := entry.get("details", {}).get("text_evidence"):
            signal["details"]["text_evidence"] = source_details
    # Stylometry also creates sentence_consensus, so counting legacy signals
    # mistakes two heuristic views for independent learned evidence. Sources are
    # present only for successful detector runs with an aggregate. Learned
    # adapters validate their numeric scores before those runs are fused.
    learned_available = any(source in sources for source in (
        "classifier_document", "classifier_secondary", "binoculars_document"))
    abstain = selection.mode == "fallback" and not learned_available
    informative = not abstain and any(s["ai_probability"] is not None for s in legacy["signals"])
    decision = common.fallback_decision(legacy["verdict"], informative=informative, subject="document")
    if abstain:
        # N-14a: checks that were skipped for the server's memory budget or for model files that are not installed are named
        # in the reason as the server's, so that nobody reads the result as a fault of the file
        decision.update(verbal="Not assessable: no learned text detector produced a usable result",
                        band="uncertain", reason=" ".join(filter(None, [NO_LEARNED_EVIDENCE, not_run_on_server(detected)])))
        notes.append(NO_LEARNED_EVIDENCE)
    verdict = common.legacy_verdict({**legacy["verdict"], "caveats": [*notes, *legacy["verdict"]["caveats"]]}, decision)
    if abstain:
        # The legacy adapter otherwise keeps its original AI label and summary,
        # even when the canonical decision publishes no estimate.
        verdict.update(label="Not assessable", summary=decision["verbal"], reasoning=[NO_LEARNED_EVIDENCE])
    view = {"verdict": verdict, "signals": legacy["signals"], "sentences": legacy["sentences"], "stats": legacy["stats"],
            "document": legacy["document"]}
    out = {"decision": decision, "assessments": common.LEGACY_ASSESSMENTS, "signals": signals,
           "caveats": [FALLBACK_CAVEAT, *notes, *legacy["verdict"]["caveats"]], "v1": common.json_safe(view),
           **common.stage_exhibits(ctx, {"fuse"})}
    if selection.mode == "artefact":
        fused = common.artefact_fusion(ctx, "document", selection, out, detected.get("detectors", []))
    else:
        fused = common.fallback_fusion(ctx, "document", selection, out)
    return enforce_extraction_scope(fused, doc)


def synthesise(ctx: StageContext) -> dict[str, Any]:
    return common.synthesise_findings(ctx, kind="document", entry_stages=("detect",), exhibit_stages=EXHIBIT_STAGES)


def render(ctx: StageContext) -> dict[str, Any]:
    out = common.index_artefacts(ctx, EXHIBIT_STAGES)
    sentences = ((ctx.outputs.get("fuse") or {}).get("v1") or {}).get("sentences", [])
    produced = sentence_plot([x["p_ai"] for x in sentences if x["p_ai"] is not None], common.legacy_artifact_dir(ctx))
    return {**out, "legacy_artifacts": {name: "fuse" for name in produced}}


def v1_view(ctx: StageContext, result: AnalysisResultV2) -> dict[str, Any]:
    """The v0.3 document result: verdict, signals, sentences, stats and document metadata from the fuser, text and
    paragraphs from decode, and the artifact files written by render."""
    view = dict(ctx.outputs["fuse"]["v1"])
    decoded = ctx.outputs["decode"]
    legacy_artifacts = (ctx.outputs.get("render") or {}).get("legacy_artifacts") or {}
    view.update(text=decoded["text"], paragraphs=decoded["paragraphs"], artifacts=sorted(legacy_artifacts), timings={},
                source_map=decoded.get("source_map", []))
    return view


def publish(ctx: StageContext) -> dict[str, Any]:
    result = common.build_result(ctx, kind="document", pipeline_version=PIPELINE_VERSION, entry_stages=("detect",),
                                 exhibit_stages=EXHIBIT_STAGES)
    decoded = ctx.outputs["decode"]
    result.manifest.sampling["document_source"] = {
        "version": decoded.get("meta", {}).get("source_map_version"),
        "source_sha256": decoded.get("meta", {}).get("original_sha256"),
        "extracted_text_sha256": decoded.get("meta", {}).get("extracted_text_sha256"),
        "source_map_sha256": canonical_hash(decoded.get("source_map", [])),
        "decode_sha256": canonical_hash(decoded),
        "offset_unit": "unicode_codepoint",
    }
    return common.publish(ctx, result, v1_view(ctx, result), pipeline_version=PIPELINE_VERSION, run_stages=("detect",))
