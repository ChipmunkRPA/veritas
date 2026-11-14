"""Shared helpers for the document adapters: detector specs and the extracted document from the decode stage."""
from __future__ import annotations

from app.detectors.text import language_scope
from app.detectors.text.extract import Document
from app.detectors.text.segment import Sentence
from app.engine.detector import (
    Applicability,
    DetectorSpec,
    Hypothesis,
    InputUnit,
    Modality,
    Profile,
    Requirements,
    RunContext,
)

ALL_PROFILES = frozenset(Profile)
FULL_DEEP = frozenset({Profile.FULL, Profile.DEEP})
DEEP_ONLY = frozenset({Profile.DEEP})
ANY_DEVICE = ("cuda", "mps", "cpu")
CPU_ONLY = ("cpu",)


def text_spec(det_id: str, name: str, unit: InputUnit, produces: tuple[str, ...] = (), *, model_ids: tuple[str, ...] = (),
              lane: str = "accelerator", devices: tuple[str, ...] = ANY_DEVICE, timeout_s: float = 60.0,
              profiles: frozenset[Profile] = ALL_PROFILES, version: str = "0.3.0") -> DetectorSpec:
    """Historical adapters default to 0.3.0; changed pre/post-processing must explicitly supply a new version."""
    return DetectorSpec(id=det_id, version=version, name=name, modality=Modality.TEXT, input_unit=unit, family="text",
                        hypotheses=(Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS,), produces=produces,
                        requirements=Requirements(model_ids=model_ids, needs=("document",), devices=devices, timeout_s=timeout_s,
                                                  lane=lane),  # type: ignore[arg-type]
                        profiles=profiles)


def document_input(ctx: RunContext) -> tuple[Document, list[Sentence]]:
    """The decode stage output: text, paragraph offsets, metadata and the v0.3 sentence segmentation."""
    d = ctx.outputs.document or {}
    doc = Document(d.get("text", ""), [(int(a), int(b)) for a, b in d.get("paragraphs", [])], dict(d.get("meta") or {}),
                   list(d.get("source_map") or []))
    return doc, [Sentence(**s) for s in d.get("sentences", [])]


def has_sentences(ctx: RunContext) -> Applicability:
    d = ctx.outputs.document
    if not d or not d.get("sentences"):
        return Applicability(False, "no extracted text")
    if message := language_scope.MESSAGES.get(d.get(language_scope.KEY)):
        return Applicability(False, message[1])
    return Applicability(True)


def revisions(ctx: RunContext, *model_ids: str) -> dict[str, str]:
    return {m: str(ctx.lease.get(m).revision) for m in model_ids}
