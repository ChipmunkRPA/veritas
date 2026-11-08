"""Display-only wording for system prose; frozen evidence and manifests remain verbatim."""
from __future__ import annotations

import re

# Exclude URL path segments and leave a sentence's terminal punctuation intact.
_LOCAL_PATH = re.compile(r"(?<![\w:/.-])(?:docs|backend|frontend|tests)/[A-Za-z0-9_./<>-]*[A-Za-z0-9_<>-]")
_TICKET = re.compile(r"(?<![\w-])(?:H-3|I-1|I-4|I-5|I-12|I-14|M-3|D-15|N-\d+[a-z]?)(?![\w-])")
_PARENTHESES = re.compile(r"\([^()]*\)")
_OUTCOME = re.compile(r" \([A-Z]\d+\.[a-z][a-z0-9_]*\)(?=:)")
_EXTERNAL = re.compile(r"(?:https?://)?(?:github\.com|huggingface\.co|arxiv\.org)/[^\s)]+")
_NO_SCORE = re.compile(r"^(.+?) produced a result without a score(?=[;.])")
_MODEL_CARD = re.compile(r"^([A-Za-z0-9._-]+/([A-Za-z0-9._-]+)), Hugging Face model card")


def system_prose(value: str) -> str:
    """Remove build references only from known system prose, keeping measured caveats.

    Callers must not use this on observed/reference evidence, examiner text or metadata.
    Only parentheses containing a removed token have their separators repaired.
    """
    def group(match: re.Match[str]) -> str:
        original = match.group(0)[1:-1]
        cleaned = _TICKET.sub("", _LOCAL_PATH.sub("", original))
        if cleaned == original:
            return match.group(0)
        cleaned = re.sub(r"([,;])\s*(?:[,;]\s*)+", r"\1 ", cleaned).strip(" ,;")
        return f"({cleaned})" if cleaned else ""

    text = _PARENTHESES.sub(group, value)
    text = _TICKET.sub("", _LOCAL_PATH.sub("", text))
    text = text.replace("has_face gate", "face-detection gate")
    # N-19c (design review, P2 "PDFs"): the generated sentence of a check without a score read as a claim about the file where
    # the check's name is a state ("No face detected in sampled frames produced a result without a score")
    text = _NO_SCORE.sub(r"\1: no score was produced", text)
    text = re.sub(r"[ \t]+([:;,.])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def metadata_rule(value: str) -> str:
    """Label known generated rule keys, preserving separators and unknown tokens verbatim."""
    # MetadataAnalysis.tree_rows joins multiple hits in this generated column.
    # New/unknown codes remain exact until their presentation contract is reviewed.
    known = {
        "M1.original_after_digitised", "M1.modified_before_capture", "M1.modified_after_capture", "M1.xmp_differs",
        "M1.subseconds_missing", "M2.edit_history", "M2.editor_named", "M2.trail_removed", "M3.not_assessed",
        "M3.aspect_differs", "M3.thumbnail_differs", "M4.maker_notes_mismatch", "M4.maker_notes_absent",
        "M4.dimensions_differ", "M5.position_out_of_range", "M5.null_island", "M5.altitude", "M5.datum", "M5.gps_clock",
        "M6.generator_declared", "M7.platform_pattern", "M7.no_metadata",
    }
    return "".join(part.partition(".")[2].replace("_", " ").capitalize() if part in known else part
                   for part in re.split(r"(,\s*)", value))


def quantisation_outcome(value: str) -> str:
    """The generated outcome cell already explains the enum in words after the colon."""
    return _OUTCOME.sub("", value)


def coverage(value: str) -> str:
    """Describe known implementation references; the attached manifest retains exact paths."""
    text = value.replace("generator names in backend/app/detectors/image/metadata.py GENERATOR_PATTERNS", "generator names")
    text = text.replace("the IF §4.3 M6 fingerprints", "structured metadata fingerprints")
    text = text.replace(" in backend/app/provenance/metadata_rules.py", "")
    text = text.replace("diffusers stable_diffusion_xl/watermark.py", "diffusers SDXL watermark implementation")
    text = text.replace("black-forest-labs/flux src/flux/util.py", "black-forest-labs FLUX reference utilities")
    text = text.replace("CompVis/stable-diffusion scripts/txt2img.py", "CompVis Stable Diffusion reference generation script")
    return system_prose(text)


def citation_identifiers(value: str) -> list[str]:
    """Exact external identifiers, not whole citations or arbitrary snake-case prose."""
    out = [m.group(0).rstrip(".,;") for m in _EXTERNAL.finditer(value)]
    model = _MODEL_CARD.match(value)
    if model:
        out += [model.group(1), model.group(2)]
    return out
