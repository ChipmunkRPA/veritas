"""Source-bound comparisons of already recorded classifier contexts.

A response belongs to its complete input. This comparison does not run a model,
measure formatting-only effects or assign authorship confidence to a passage.
"""

from __future__ import annotations

import copy
import math

from app.core.canonical import canonical_hash

SCHEMA = "chipmunk.document-context-comparison/1"
MAX_CHECKS = 6
NOTE = (
    "These saved classifier inputs all contain the selected passage but include different surrounding text. "
    "Their responses apply to the complete inputs, not the selected words. This is context information, "
    "not a formatting-only test, independent confirmation or a calibrated authorship probability. "
    "It does not change the overall confidence."
)


def build(result: dict, text: str, focus: dict) -> dict:
    from app.api.public_projection import alias_signal
    from app.detectors.text.extract import text_hash
    from app.evidence.document_packets import _range, verify_context_window

    a, b = _range(focus, len(text))
    checks, seen = [], set()
    for signal in result.get("signals", []):
        evidence = (signal.get("details") or {}).get("text_evidence") or {}
        if signal.get("status") != "ok" or evidence.get("version") not in {
            "scored-context/1",
            "scored-context/2",
        }:
            continue
        identity = (signal.get("detector_id", signal["id"]), canonical_hash(evidence))
        if identity in seen:
            continue
        seen.add(identity)
        if (
            evidence.get("extracted_text_sha256") != text_hash(text)
            or evidence.get("offset_unit") != "unicode_codepoint"
        ):
            raise ValueError("Context comparison refers to a different extracted source")
        unique, scores = {}, {}
        for window in evidence.get("windows", []):
            if window.get("score_kind", "classifier_probability") != "classifier_probability":
                continue  # Ratio reductions answer a different question; never pool their values.
            verify_context_window(window, text)
            spans = window["spans"]
            if not any(span["start"] <= a < b <= span["end"] for span in spans):
                continue
            digest, score = window["input_sha256"], window["score"]
            if digest in scores and not math.isclose(scores[digest], score, abs_tol=1.1e-6, rel_tol=0):
                raise ValueError("The same recorded classifier input has inconsistent responses")
            scores[digest] = score
            unique.setdefault(
                digest,
                {
                    "spans": copy.deepcopy(spans),
                    "text": " ".join(text[s["start"] : s["end"]] for s in spans),
                    "input_sha256": digest,
                    "score": score,
                    "token_coverage": {
                        key: value
                        for key, value in window.get("token_coverage", {}).items()
                        if key
                        in {
                            "tokens",
                            "covered_tokens",
                            "token_windows",
                            "context_limit",
                            "omitted_tokens",
                            "coverage_fraction",
                        }
                        and type(value) in (int, float)
                        and math.isfinite(value)
                        and value >= 0
                    },
                },
            )
        if len(unique) < 2:
            continue
        ordered = sorted(unique.values(), key=lambda w: (w["score"], w["input_sha256"]))
        low, high = ordered[0], ordered[-1]
        checks.append(
            {
                "signal_ref": {"id": alias_signal(signal["id"]), "display_id": signal.get("display_id")},
                "score_kind": "classifier_probability",
                "distinct_inputs": len(unique),
                "lowest_response": low["score"],
                "highest_response": high["score"],
                "response_spread": round(high["score"] - low["score"], 6),
                "inputs": [low, high],
                "omitted_inputs": len(unique) - 2,
            }
        )
    checks.sort(
        key=lambda c: (-c["response_spread"], c["signal_ref"]["display_id"] or "", c["signal_ref"]["id"])
    )
    return {
        "schema": SCHEMA,
        "status": "recorded" if checks else "not_assessed",
        "evidence_type": "context",
        "authorship_confidence_pct": None,
        "affects_overall_confidence": False,
        "focus": {"start": a, "end": b},
        "checks": checks[:MAX_CHECKS],
        "omitted_checks": max(0, len(checks) - MAX_CHECKS),
        "note": NOTE,
        "reason": None
        if checks
        else "Fewer than two distinct compatible classifier inputs containing this passage were recorded.",
    }


def validate(block: dict, *, focus: dict, chars: int, text: str | None = None) -> None:
    """Validate public structure and measurements without an application runtime."""
    import hashlib

    def require(value, detail):
        if not value:
            raise ValueError(detail)

    def interval(span):
        a, b = span["start"], span["end"]
        require(
            type(a) is int and type(b) is int and 0 <= a < b <= chars,
            "Invalid context-comparison source range",
        )
        return a, b

    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1

    a, b = interval(focus)
    require(
        set(block)
        == {
            "schema",
            "status",
            "evidence_type",
            "authorship_confidence_pct",
            "affects_overall_confidence",
            "focus",
            "checks",
            "omitted_checks",
            "note",
            "reason",
        }
        and block["schema"] == SCHEMA
        and block["evidence_type"] == "context"
        and block["authorship_confidence_pct"] is None
        and block["affects_overall_confidence"] is False
        and block["focus"] == {"start": a, "end": b}
        and block["note"] == NOTE,
        "Context comparison claims unsupported authority or confidence",
    )
    require(
        block["status"] in {"recorded", "not_assessed"}
        and (block["status"] == "recorded") == bool(block["checks"])
        and len(block["checks"]) <= MAX_CHECKS
        and type(block["omitted_checks"]) is int
        and block["omitted_checks"] >= 0,
        "Context-comparison availability or bounds differ",
    )
    require(
        (block["reason"] is None) == bool(block["checks"]), "Context-comparison availability is unexplained"
    )
    refs = set()
    for check in block["checks"]:
        require(
            set(check)
            == {
                "signal_ref",
                "score_kind",
                "distinct_inputs",
                "lowest_response",
                "highest_response",
                "response_spread",
                "inputs",
                "omitted_inputs",
            },
            "Context comparison contains undeclared fields",
        )
        ref = check["signal_ref"]["id"]
        require(
            ref not in refs and check["score_kind"] == "classifier_probability",
            "Duplicate or incomparable context check",
        )
        refs.add(ref)
        low, high = check["lowest_response"], check["highest_response"]
        require(
            number(low)
            and number(high)
            and low <= high
            and number(check["response_spread"])
            and abs(check["response_spread"] - (high - low)) <= 1.1e-6,
            "Context response range or spread differs",
        )
        require(
            type(check["distinct_inputs"]) is int
            and check["distinct_inputs"] >= 2
            and type(check["omitted_inputs"]) is int
            and check["omitted_inputs"] == check["distinct_inputs"] - 2
            and len(check["inputs"]) == 2,
            "Context-input counts differ",
        )
        require(
            [w["score"] for w in check["inputs"]] == [low, high]
            and check["inputs"][0]["input_sha256"] != check["inputs"][1]["input_sha256"],
            "Selected context inputs do not match distinct range endpoints",
        )
        for window in check["inputs"]:
            require(
                set(window) == {"spans", "text", "input_sha256", "score", "token_coverage"}
                and number(window["score"]),
                "Invalid context input fields or response",
            )
            spans = [interval(s) for s in window["spans"]]
            require(
                spans
                and all(x[1] <= y[0] for x, y in zip(spans, spans[1:]))
                and any(start <= a < b <= end for start, end in spans),
                "A comparison input omits its selected passage",
            )
            actual = " ".join(text[start:end] for start, end in spans) if text is not None else window["text"]
            require(
                len(actual) == sum(end - start for start, end in spans) + len(spans) - 1
                and window["text"] == actual
                and hashlib.sha256(actual.encode("utf-8", "surrogatepass")).hexdigest()
                == window["input_sha256"],
                "Context-comparison input differs from its recorded source",
            )
