"""A bounded second reading of existing positive writing contexts."""

from __future__ import annotations

from dataclasses import replace

from app.core.config import settings
from app.detectors.text.adapters.common import CPU_ONLY, FULL_DEEP, text_spec
from app.detectors.text.coverage import classify_complete
from app.engine.boundary import Cancelled
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    RunContext,
    StageOutputs,
)
from app.evidence import document_matched_context, document_scope

ID = "txt.review.agreement"
FAMILY = "text_review"
MODEL = "hf.rasbt.writing_review"
KEY = document_scope.CHECKPOINT_AGREEMENT_KEY
BATCH = 4
REVIEW_SECONDS = 20.0
FINISH_RESERVE_SECONDS = 30.0


def enabled() -> bool:
    return (settings.environment == "dev" and settings.text_detector_mode == "candidate"
            and settings.text_window_sentences == 3 and settings.text_checkpoint_agreement_review)


def _supported_source(decoded: dict) -> bool:
    meta = decoded["meta"]
    method = meta.get("method")
    if method == "plain":
        return True
    if method == "pymupdf":
        try:
            if document_scope.pdf_excluded_pages(meta):
                return False
        except ValueError:
            return False
    sources = decoded.get("source_map") or ()
    paragraphs = decoded["paragraphs"]
    if not sources or len(sources) != len(paragraphs) or any(
        not isinstance(source, dict) or (source.get("start"), source.get("end")) != (start, end)
        for source, (start, end) in zip(sources, paragraphs, strict=True)
    ):
        return False
    if method == "python-docx":
        return all(source.get("kind") == "docx" and len(source.get("body_path", ())) == 1
                   and source.get("run_mapping_complete") for source in sources)
    if method == "pymupdf":
        return all(source.get("kind") == "pdf" and source.get("extraction") == "text_layer"
                   and source.get("fragments") for source in sources)
    if method == "email":
        mail = meta.get("email") or {}
        return mail.get("body_content_type") == "text/plain" and all(
            source.get("kind") == "email" and source.get("quote_role") == "body"
            and source.get("mime_part") == mail.get("body_part") for source in sources
        )
    return False


def _email_reply_body(decoded: dict) -> list[dict] | None:
    meta = decoded.get("meta") or {}
    if meta.get("method") != "email":
        return None
    return document_scope.email_agreement_body_spans(
        decoded.get("source_map") or [], meta.get("email") or {}, len(decoded["text"]),
    )


def _pdf_text_pages(decoded: dict) -> list[dict] | None:
    meta = decoded.get("meta") or {}
    if meta.get("method") != "pymupdf":
        return None
    return document_scope.pdf_agreement_text_spans(decoded.get("source_map") or [], meta, decoded["text"])


def candidates(decoded: dict, primary: dict) -> list[dict] | None:
    from app.detectors.text.extract import _ocr_language

    text = decoded["text"]
    body_spans = _email_reply_body(decoded)
    pdf_spans = _pdf_text_pages(decoded)
    if pdf_spans is not None:
        text = " ".join(text[span["start"]:span["end"]] for span in pdf_spans)
        paragraphs = len(pdf_spans)
        sentences = sum(any(span["start"] <= s["start"] < s["end"] <= span["end"] for span in pdf_spans)
                        for s in decoded["sentences"])
    elif body_spans is None:
        if not _supported_source(decoded):
            return None
        paragraphs, sentences = len(decoded["paragraphs"]), len(decoded["sentences"])
    else:
        end = body_spans[-1]["end"]
        text = text[:end]
        paragraphs = len(body_spans)
        sentences = sum(sentence["end"] <= end for sentence in decoded["sentences"])
    if ((body_spans is None and paragraphs not in {1, 4}) or not 68 <= len(text.split(None, 420)) <= 420
            or paragraphs == 1 and sentences != 4):
        return None
    language = _ocr_language(text)
    if language.get("code") != "en" or language.get("status") != "supported":
        return None
    if not document_matched_context.matches(primary["detector_id"], primary["version"], primary["model_revisions"]):
        return None
    source = primary["details"]["text_evidence"]
    windows = [window for window in source["windows"] if document_scope.checkpoint_window_in_scope(window, body_spans)
               and (pdf_spans is None or document_scope.pdf_agreement_window_page(window, pdf_spans) is not None)
               and window["score"] >= .5]
    return windows[:document_scope.CHECKPOINT_AGREEMENT_MAX]


def _review(ctx, text: str, windows: list[dict]) -> dict:
    from app.engine.stages import common

    remaining = ctx.deadline.remaining()
    budget = REVIEW_SECONDS if remaining is None else min(REVIEW_SECONDS, remaining - FINISH_RESERVE_SECONDS)
    if not windows or budget <= 0:
        return {}
    optional = replace(ctx, deadline=ctx.deadline.nested(budget))
    try:
        output = common.run_detectors(optional, (WritingAgreementReview.spec.modality,), include={FAMILY},
                                      outputs=StageOutputs(document={"text": text, "review_windows": windows}))
        reviewed = next((entry for entry in output["detectors"] if entry["detector_id"] == ID
                         and entry["status"] == "ok"), None)
        return reviewed["details"]["checkpoint_responses"] if reviewed else {}
    except Cancelled:
        raise
    except Exception as error:
        ctx.log.warning("writing_review.unavailable", error_type=type(error).__name__)
        return {}
    finally:
        ctx.check("after the optional second writing check")


def record(ctx, detected: dict, decoded: dict) -> None:
    """Store the second response with the exact primary window it revisited."""
    primary = next((entry for entry in detected["detectors"] if entry["detector_id"] == document_matched_context.DETECTOR
                    and entry["status"] == "ok"), None)
    if primary is None or (windows := candidates(decoded, primary)) is None:
        return
    evidence = primary["details"]["text_evidence"]
    text = decoded["text"]
    recorded = [{**window["focus"], "spans": window["spans"], "input_sha256": window["input_sha256"], "state": "unavailable"}
                for window in windows]
    value = {"schema": document_scope.CHECKPOINT_AGREEMENT_SCHEMA, "ranges": recorded,
             "authorship_confidence_contribution": 0}
    document = {"coverage": {KEY: value}, "source": {"chars": len(text)}}
    if (body_spans := _email_reply_body(decoded)) is not None:
        email = decoded["meta"]["email"]
        legacy = document_scope.CHECKPOINT_AGREEMENT_EMAIL_SCHEMA
        if document_scope.mapped_email_html(email):
            extended = document_scope.CHECKPOINT_AGREEMENT_HTML_SHORT_LAYOUT_SCHEMA
            schema = (extended if len(body_spans) in document_scope.CHECKPOINT_AGREEMENT_EMAIL_PARAGRAPHS[extended]
                      else document_scope.CHECKPOINT_AGREEMENT_HTML_SCHEMA)
        else:
            schema = (legacy if len(body_spans) in document_scope.CHECKPOINT_AGREEMENT_EMAIL_PARAGRAPHS[legacy]
                      else document_scope.CHECKPOINT_AGREEMENT_EMAIL_LAYOUT_SCHEMA)
        value.update(schema=schema, body_spans=body_spans)
        document["source"]["method"] = "email"
        document["coverage"]["email"] = email
    pdf_spans = _pdf_text_pages(decoded)
    if pdf_spans is not None:
        value.update(schema=document_scope.CHECKPOINT_AGREEMENT_PDF_SCHEMA, pdf_spans=pdf_spans)
        document["source"]["method"] = "pymupdf"
        document["coverage"].update(decoded["meta"])
    mapping = decoded.get("source_map") if body_spans is not None or pdf_spans is not None else None
    document_scope.validate_checkpoint_agreement(document, text, mapping=mapping)
    eligible = [window for window in windows if document_matched_context.one_complete_window(window["token_coverage"])]
    responses = _review(ctx, text, eligible)
    if responses:
        try:
            measured = []
            for window, row in zip(windows, recorded, strict=True):
                response = responses.get(window["unit_ref"])
                if response and response["input_sha256"] == row["input_sha256"]:
                    score = float.fromhex(response["response_exact_hex"])
                    row = {**row, "response_exact_hex": response["response_exact_hex"], "tokens": response["tokens"],
                           "state": "corroborated" if score >= .5 else "different"}
                measured.append(row)
            value["ranges"] = measured
            document_scope.validate_checkpoint_agreement(document, text, mapping=mapping)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
            value["ranges"] = recorded
            ctx.log.warning("writing_review.invalid_response", error_type=type(error).__name__)
    for window in evidence["windows"]:
        window["score_exact_hex"] = float(window["score"]).hex()
    evidence[KEY] = value


class WritingAgreementReview(Detector):
    spec = replace(text_spec(ID, "Second writing check", InputUnit.SENTENCES, model_ids=(MODEL,), devices=CPU_ONLY,
                             timeout_s=45.0, profiles=FULL_DEEP, version="1.0.0"), family=FAMILY)

    def applicable(self, ctx: RunContext) -> Applicability:
        return Applicability(bool(enabled() and (ctx.outputs.document or {}).get("review_windows")),
                             "no eligible primary writing contexts")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(MODEL, BATCH, "token window", "At most 512 tokens per input including specials"),)

    def run(self, ctx: RunContext) -> DetectorResult:
        from app.detectors.text.extract import text_hash

        document = ctx.outputs.document
        text = document["text"]
        windows = document["review_windows"]
        inputs = [" ".join(text[span["start"]:span["end"]] for span in window["spans"]) for window in windows]
        if not 0 < len(windows) <= document_scope.CHECKPOINT_AGREEMENT_MAX or any(
            text_hash(value) != window["input_sha256"] for value, window in zip(inputs, windows, strict=True)
        ):
            raise ValueError("Writing review input differs from the recorded context")
        scores, coverage = score_inputs(inputs, ctx.lease.get(MODEL).obj, ctx.device, check=ctx.deadline.check)
        measured = {
            window["unit_ref"]: {"input_sha256": window["input_sha256"], "response_exact_hex": float(score).hex(),
                                 "tokens": proof["tokens"]}
            for window, score, proof in zip(windows, scores, coverage, strict=True)
            if document_matched_context.one_complete_window(proof)
        }
        return DetectorResult(status="ok", details={"checkpoint_responses": measured})


def score_inputs(inputs, model_parts, device, *, check):
    import torch

    tokenizer, model, temperature = model_parts

    def evaluate(encoded):
        logits = model(encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits.float()
        if logits.shape[-1] != 2 or not torch.isfinite(logits).all():
            raise ValueError("Invalid writing review response")
        return (logits / temperature).softmax(dim=-1)[:, 1]

    with torch.inference_mode():
        return classify_complete(inputs, tokenizer, device, evaluate, max_len=512, batch=BATCH, check=check)
