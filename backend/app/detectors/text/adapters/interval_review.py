"""Read isolated sentences for optional source-mapped passage review."""
from __future__ import annotations

import math

from app.core.canonical import FLOAT_DP
from app.core.config import settings
from app.detectors.text import language_scope
from app.detectors.text.adapters.common import FULL_DEEP
from app.engine.boundary import DetectorTimeout
from app.evidence import (
    document_interval_locations,
    document_matched_context,
    document_passage_intervals,
    document_tracked_changes,
)

KEY = document_passage_intervals.KEY
EXTRA_SECONDS = 18.0


def enabled() -> bool:
    return (settings.environment == "dev" and settings.text_passage_intervals_review
            and settings.text_detector_mode == "candidate" and settings.text_window_sentences == 3)


def policy() -> str | bool:
    if not enabled():
        return False
    if settings.text_passage_document_baseline_review:
        schema = (document_passage_intervals.REPORT_SCHEMA if settings.text_passage_report_review
                  else document_passage_intervals.DOCUMENT_BASELINE_SCHEMA)
    else:
        schema = document_passage_intervals.NATIVE_SCHEMA
    return f"{schema}+short-scope/1"


def measure(classifier, doc, sentences, context, ctx, spec, revisions):
    if (not enabled() or ctx.profile not in FULL_DEEP
            or ctx.outputs.document.get(language_scope.KEY) != language_scope.ENGLISH
            or not document_matched_context.matches(spec.id, spec.version, revisions)):
        return None
    method = doc.meta.get("method")
    end = len(doc.text) if method == "plain" else document_interval_locations.checked_end(
        method, doc.text, doc.source_map, doc.meta, ctx.outputs.document.get(document_tracked_changes.KEY),
    )
    if end is None:
        return None
    scoped = [sentence for sentence in sentences if sentence.end <= end]
    if len(scoped) < document_passage_intervals.MIN_SENTENCES:
        return document_passage_intervals.short_scope(method, end, len(scoped))
    if len(scoped) > document_passage_intervals.MAX_SENTENCES:
        return None
    geometry = [[sentence.start, sentence.end, sentence.n_words] for sentence in scoped]
    remaining = ctx.deadline.remaining()
    readings = None
    if remaining is None or remaining >= EXTRA_SECONDS + 2:
        deadline = ctx.deadline.nested(EXTRA_SECONDS)
        try:
            inputs = [sentence.text for sentence in scoped]
            contexts = context[:len(scoped)]
            shortened = len(scoped) < len(sentences)
            if shortened:
                # Replace the last primary window, which included the first sentence of the excluded email tail.
                inputs.append(" ".join(inputs[-2:]))
            values = list(classifier.predict(inputs, batch=12, check=deadline.check))
            if shortened:
                contexts[-1] = values.pop()
            deadline.check("separate writing passage responses")
            if (len(values) == len(contexts) == len(scoped)
                    and all(math.isfinite(value) and 0 <= value <= 1 for value in (*values, *contexts))):
                # Select from exactly the values the canonical evidence record will retain.
                readings = {"context": [round(float(value), FLOAT_DP) for value in contexts],
                            "isolated": [round(float(value), FLOAT_DP) for value in values]}
        except (DetectorTimeout, IndexError, OSError, RuntimeError, ValueError):
            # The existing context and document scores are already available.
            pass
    return document_passage_intervals.capture(
        doc.text, doc.source_map, geometry, readings, method=method, checked_end=end,
        document_baseline=settings.text_passage_document_baseline_review,
        report_review=settings.text_passage_report_review,
    )
