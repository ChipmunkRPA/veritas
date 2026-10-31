"""Admission records from measured evidence (BUILD_SPEC H-4; D-41, D-44; ML §3.2 A1–A6).

- **Inputs per detector.**
  - `ablation-<id>.json` (`calibrate --ablate`): A3a, A3b, A4.
  - `admission-evidence-<id>.json` (`admission_evidence.py`): A2, A5, A6.
  - The H-3 stand-alone camera-original check (`h3_real_fp.json`).
  - The card licence classes of the detector's models.
  - An inputs file with the A1 device-parity verdict and the H-3 decision.
  A missing file leaves its criteria null ("not established"); nothing is inferred. Available new-policy ablation,
  gated and standalone inputs must carry the same known policy and alias-equivalent exclusion sets. All-legacy replay is
  labelled legacy. The separate historical H-3 check does not inherit this exclusion-policy claim.
- **A1.** Every model of the detector has licence class `commercial`, and CPU vs MPS parity passed. A detector without
  models runs on the CPU only, so parity does not apply. A non-commercial class fails A1 for the commercial profile.
- **Criteria.** A3 and A4 are recomputed with the current rule (`ablation.criteria`, D-44) from the ablation's stored LGO
  and cross-fit tables, so a rule change needs no refit; a file without the tables keeps its stored criteria.
- **Decision.** `ablation.admission_decision` (D-44); the A5 terms follow the transforms whose A5 verdict fails.
- **Scope.** A record decides union-fuser membership. A detector with its own lane (card `lane: partial_edit`, or the
  `provenance` family whose documentary signals act in Stage 0) is admitted to that lane separately (H-6 G8; the
  Stage 0 rules), so a union-fuser rejection leaves its card decision `pending` for the lane, unless A1 fails or H-3 retired it.
- **Card line.** `card_admission` summarises a record for the card's `admission` field (admitted with or without context terms
  → `admitted`, not established → `pending`, an H-3 retirement → `retired`); `update_card_lines` rewrites the one-line
  `admission:` entry of each named detector in `app/models.yaml` and leaves every other line untouched.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from app.bench.ablation import AblationError, admission_decision, criteria, gate_words
from app.bench.contamination import POLICY_VERSION, canonical_exclusions

_EMPTY_CRITERIA = {"A3a": {"pass": None}, "A3b": {"pass": None}, "A3": {"pass": None}, "A4": {"pass": None}}


def a1_verdict(licence_classes: Sequence[str], parity: Mapping | None) -> dict:
    """A1 from the card licence classes (empty = no model) and the parity verdict ({pass, evidence} or None)."""
    classes = list(licence_classes)
    restricted = [c for c in classes if c != "commercial"]
    if restricted:
        return {"pass": False, "licence_classes": classes, "parity": parity,
                "reason": f"licence class {', '.join(sorted(set(restricted)))} is not commercial (D-04)"}
    if not classes:
        return {"pass": True, "licence_classes": classes, "parity": None, "reason": "no model: CPU only, parity not applicable"}
    verdict = None if parity is None else parity.get("pass")
    reason = ("parity not established" if verdict is None else ("parity passed" if verdict else "parity failed"))
    return {"pass": verdict, "licence_classes": classes, "parity": parity, "reason": reason}


def _criteria(ablation: Mapping | None) -> dict:
    if not ablation:
        return _EMPTY_CRITERIA
    if ablation.get("lgo") is not None and ablation.get("cross_fit") is not None:
        return criteria(ablation["lgo"], ablation["cross_fit"])
    return ablation["criteria"]


def contamination_provenance(ablation: Mapping | None, evidence: Mapping | None,
                             gated_ablation: Mapping | None) -> dict:
    """Match scientific inputs, keeping legacy replay distinct from current-policy evidence.

    H-3 is a separate historical population check and is not covered by this policy.
    Different dropped counts are valid across ablation scope and standalone populations.
    """
    if evidence is not None and evidence.get("schema") not in (None, "veritas.admission-evidence/1", "veritas.admission-evidence/2"):
        raise AblationError("evidence: unknown admission evidence schema")
    inputs = {name: doc for name, doc in (("ablation", ablation), ("gated_ablation", gated_ablation), ("evidence", evidence))
              if doc is not None}
    strict = any("contamination_policy" in (doc.get("items") or {})
                 or doc.get("schema") == "veritas.admission-evidence/2" for doc in inputs.values())
    if not strict:
        return {"status": "legacy" if inputs else "not_established", "policy": None, "inputs": list(inputs),
                "contamination_sets": None,
                "note": "Legacy contamination provenance; current exclusion policy was not established." if inputs else
                        "No ablation or standalone evidence available."}
    effective = None
    declared = {}
    for name, doc in inputs.items():
        items = doc.get("items") or {}
        if items.get("contamination_policy") != POLICY_VERSION:
            raise AblationError(f"{name}: missing or unknown contamination policy; expected {POLICY_VERSION}")
        if name == "evidence" and doc.get("schema") != "veritas.admission-evidence/2":
            raise AblationError("evidence: current contamination policy requires veritas.admission-evidence/2")
        try:
            keys = canonical_exclusions(items.get("contamination_sets"))
        except ValueError as exc:
            raise AblationError(f"{name}: {exc}") from exc
        if effective is not None and keys != effective:
            raise AblationError(f"{name}: contamination exclusions differ between available admission inputs")
        effective = keys
        declared[name] = list(items["contamination_sets"])
    return {"status": "current", "policy": POLICY_VERSION, "inputs": list(inputs),
            "contamination_sets": next(iter(declared.values())), "exclusions_by_input": declared}


def build(detector_id: str, *, ablation: Mapping | None, evidence: Mapping | None, camera: Mapping | None, a1: Mapping,
          camera_decision: str | None = None, camera_features: Sequence[str] = (), lane: str | None = None,
          gated_ablation: Mapping | None = None, carried_gate: Sequence[str] | None = None) -> dict:
    """One record. `gated_ablation` is the D-51 retry (`calibrate --ablate <id>` with `<id>=!<term>`); its criteria decide a
    qualifying candidate, and its `method.gates[<id>]` must be exactly the negated stratum term. `carried_gate` (D-53) is a
    member's negated gate carried in the base: the main ablation must measure exactly that gate, and its criteria decide."""
    contamination = contamination_provenance(ablation, evidence, gated_ablation)
    own = (((ablation or {}).get("method") or {}).get("gates") or {}).get(detector_id) or []
    carried = None if carried_gate is None else list(carried_gate)
    if any(lit.startswith("!") for lit in own) and own != carried:
        raise AblationError(f"{detector_id}: the ablation gates the candidate with {own}; a negated gate belongs to the gated retry "
                            "(the main ablation starts from the ungated configuration) unless it is the carried gate (D-53)")
    if carried is not None and ablation is not None and own != carried:
        raise AblationError(f"{detector_id}: the carried gate {carried} is not the gate the main ablation measured ({own or 'none'})")
    crit = _criteria(ablation)
    ev = evidence or {}
    a2, a5, a6 = ((ev.get(k) or {}).get("pass") for k in ("A2", "A5", "A6"))
    a5_failing = [t for t, e in sorted(((ev.get("A5") or {}).get("transforms") or {}).items()) if e.get("pass") is False]
    gated = None if gated_ablation is None else {
        "criteria": _criteria(gated_ablation), "gates": ((gated_ablation.get("method") or {}).get("gates") or {}).get(detector_id),
        "term_support": (gated_ablation.get("method") or {}).get("term_support"), "command": gated_ablation.get("command")}
    decision = admission_decision(crit, a1=a1["pass"], a2=a2, a5=a5, a6=a6, a5_failing=a5_failing, camera=camera,
                                  camera_decision=camera_decision, camera_features=camera_features,
                                  term_support=((ablation or {}).get("method") or {}).get("term_support"),
                                  bench=((ablation or {}).get("bench") or {}).get("version"), candidate=detector_id, gated=gated,
                                  carried=carried)
    return {"detector_id": detector_id, "contamination_provenance": contamination,
            "camera_contamination_provenance": "not_established" if camera is not None else "not_available",
            "A1": dict(a1), "A2": ev.get("A2"), "A3a": crit["A3a"], "A3b": crit["A3b"], "A4": crit["A4"],
            "A5": ev.get("A5"), "A6": ev.get("A6"), "camera_original": camera, "h3_decision": camera_decision,
            "h3_features": list(camera_features), "ablation_command": (ablation or {}).get("command"), "carried_gate": carried,
            "gated_ablation": None if gated is None else {"command": gated["command"], "gates": gated["gates"], "A3": gated["criteria"]["A3"],
                                                          "A3a": gated["criteria"]["A3a"], "A4": gated["criteria"]["A4"]},
            "baseline_detectors": (ablation or {}).get("baseline_detectors"), "items": (ablation or {}).get("items"), "lane": lane,
            **decision}


def _verdict(value: bool | None) -> str:
    return "pass" if value is True else ("**fail**" if value is False else "not established")


def _pts(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:+.1f}"


def _a2(entry: Mapping | None) -> str:
    if not entry or entry.get("auc") is None:
        return "not established"
    ci = entry.get("ci95") or {}
    lo, hi = ci.get("lo"), ci.get("hi")
    band = "" if lo is None or hi is None else f" [{lo:.4f}, {hi:.4f}]"
    return f"{_verdict(entry.get('pass'))}: {entry['auc']:.4f}{band}"


def _a4(entry: Mapping) -> str:
    if entry.get("pass") is None and entry.get("pooled_delta") is None:
        return "not established"
    lo, hi = entry.get("pooled_ci95_lo"), entry.get("pooled_ci95_hi")
    lower, upper = ("–" if v is None else f"{100 * v:+.2f}" for v in (lo, hi))  # two decimals: the 1.0 pt bound decides
    parts = [f"{_verdict(entry.get('pass'))}: pooled {_pts(entry.get('pooled_delta'))} pts [lower {lower}, upper {upper}]"]
    if entry.get("pooled_harm_established"):
        parts.append("established pooled harm")
    if entry.get("blocking_strata"):
        parts.append("blocking " + ", ".join(entry["blocking_strata"]))
    if entry.get("interaction_strata"):
        parts.append("point above +2.0 without established harm " + ", ".join(entry["interaction_strata"]))
    if entry.get("small_point_strata"):
        parts.append("point above +2.0 on fewer than 150 reals (no term) " + ", ".join(entry["small_point_strata"]))
    return "; ".join(parts)


def _a5(entry: Mapping | None) -> str:
    if not entry:
        return "not established"
    cells = [f"{name} {_pts(t.get('delta_tpr'))} pts (n {t.get('n')})" for name, t in sorted((entry.get("transforms") or {}).items())]
    return f"{_verdict(entry.get('pass'))}" + (": " + ", ".join(cells) if cells else "")


def _a6(entry: Mapping | None) -> str:
    if not entry or entry.get("p95_ms") is None:
        return "not established"
    return f"{_verdict(entry.get('pass'))}: p95 {entry['p95_ms']:.0f} ms (n {entry.get('n')})"


def _camera(record: Mapping) -> str:
    camera = record.get("camera_original") or {}
    if camera.get("fpr") is None:
        cell = "no score on reals"
    else:
        cell = f"{100 * camera['fpr']:.1f} % (n {camera['n']})" + (", triggered" if camera.get("triggered") else "")
    if record.get("h3_decision"):
        cell += f"; H-3 {record['h3_decision']}" + (f" ({', '.join(record['h3_features'])})" if record["h3_features"] else "")
    return cell


def render_markdown(records: Sequence[Mapping], *, bench: str, manifest_sha256: str, generated: str, command: str,
                    ablations: str | None = None, gated_ablations: str | None = None) -> str:
    """`ablations` / `gated_ablations`: the `--ablations` / `--gated-ablations` directory labels, named in the header."""
    where = f"`{ablations}/ablation-<id>.md`" if ablations else f"`{bench}/ablation-<id>.md`"
    lines = [f"# Admission records on {bench} fit ∪ cal (D-41, D-44, D-51, D-53)", "",
             f"Generated {generated} by `backend/scripts/admission_record.py`; manifest sha256 `{manifest_sha256}`. "
             "Automated record pending human review. Criteria ML §3.2 A1–A6 with the D-44 A4 / A5 rules (amending D-41) and the "
             f"D-51 amendments; per-detector ablation tables (criteria recomputed with the current rule) are in {where}"
             + (f", gated retries in `{gated_ablations}/ablation-<id>.md`" if gated_ablations else "")
             + ", the stand-alone camera check in `../h3_real_fp_tables.md`.", "",
             f"- Command: `{command}`",
             "- Scope: union-fuser membership. Detectors with their own lane (`partial_edit`, `provenance` Stage 0) are admitted "
             "to it separately; the localisers (`loc.*`) are partial-edit lane members only, without a union-fuser ablation (H-6).",
             "- A4 (D-44, D-51, D-53): τ_0.05 on the weighting strata of the cal reals (R6 → R3 on vb-1.1), reals at τ weighted to "
             "the rank rule's share; pooled reals block with a paired lower bound above +1.0 pt (D-53); a stratum blocks only with "
             "≥ 150 reals and a paired lower bound above +2.0 pts; a point ΔFPR above +2.0 pts without established harm on ≥ 150 "
             "reals asks for the stratum's runtime term (R3 `jpeg_q<70`; R5 `has_face`; R1, R2, R4, R6 none), a smaller stratum "
             "is not established without a term. A5 terms: JPEG and social recompression `jpeg_q<70`, screenshots `screenshot_like`.",
             "- Terms (D-51): a record keeps a term only when the fuser represents it (`features.GROUPS`) and ≥ 20 fit and ≥ 20 "
             "cal items of each class carry it; otherwise it is not learnable and the detector runs ungated. A rejected candidate "
             "whose blocking strata all have a learnable term (both sides) is ablated again gated off where the term holds "
             "(`<id>=!<term>`) and admitted context-gated when that ablation passes.", "",
             "| Detector | Decision | Context terms | Gates | A1 | A2 AUC | A3a Δ LGO macro | A3b | A4 | A5 | A6 | Camera check (H-3) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        a3a = r["A3a"]
        a3a_cell = _verdict(a3a.get("pass")) + ("" if a3a.get("lgo_macro_delta") is None else
                                                f": {_pts(a3a['lgo_macro_delta'])} pts (lower {_pts(a3a.get('ci95_lo'))})")
        a3b = r["A3b"]
        a3b_cell = _verdict(a3b.get("pass")) + (f" ({len(a3b['qualifying'])} slices)" if a3b.get("qualifying") else "")
        gates = "; ".join(f"`{d}={','.join(v)}`" for d, v in (r.get("gates") or {}).items()) or "–"
        lines.append(f"| `{r['detector_id']}` | {r['decision']} | {', '.join(r['context_terms']) or '–'} | {gates} | "
                     f"{_verdict(r['A1']['pass'])}: {r['A1']['reason']} | {_a2(r['A2'])} | {a3a_cell} | {a3b_cell} | {_a4(r['A4'])} | "
                     f"{_a5(r['A5'])} | {_a6(r['A6'])} | {_camera(r)} |")
    lines += ["", "## Contamination provenance", "",
              "The matched inputs are the main ablation, any gated retry and standalone evidence. "
              "H-3 remains a separate historical camera check; its contamination policy is not established."]
    for r in records:
        provenance = r.get("contamination_provenance") or {"status": "legacy", "policy": None}
        label = provenance["policy"] if provenance["status"] == "current" else provenance["status"]
        lines.append(f"- `{r['detector_id']}`: {label}; current exclusion policy "
                     + ("matched across available inputs." if provenance["status"] == "current" else "not established."))
    lines += ["", "## Reasons", ""]
    lines += [f"- `{r['detector_id']}`: " + ("; ".join(r["reasons"]) or "every criterion passes")
              + (f"; union-fuser record only: the {r['lane']} lane is admitted separately ({LANE_RECORDS[r['lane']]})"
                 if r.get("lane") else "") for r in records]
    return "\n".join(lines) + "\n"


CARD_DECISIONS = {"admitted": "admitted", "admitted_context_gated": "admitted", "rejected": "rejected", "not_established": "pending"}
LANE_RECORDS = {"partial_edit": "H-6, G8 on vb-1.1 edits", "provenance": "Stage 0 documentary rules"}


def _short(value: bool | None) -> str:
    return "pass" if value is True else ("fail" if value is False else "not established")


def card_admission(record: Mapping, *, bench: str, date: str, evidence: str, item: str = "H-4",
                   served_until: str | None = None) -> dict:
    """The card `admission` entry for one record: {decision, date (not for pending), served_until, evidence}; `item` names the
    BUILD_SPEC item whose admission run produced the record (H-4 on the H-4 set, J-3 on the shipped set in the D-45 feature
    space). `served_until` (D-51) is kept only on a rejected or retired card: the served artefact that still holds the detector."""
    decision = "retired" if record.get("h3_decision") == "retire" else CARD_DECISIONS[record["decision"]]
    lane = record.get("lane")
    if lane and decision == "rejected" and record["A1"]["pass"] is not False:
        decision = "pending"
    a2, a3a, a4 = record.get("A2") or {}, record["A3a"], record["A4"]
    parts = [f"A1 {_short(record['A1']['pass'])} ({record['A1']['reason']})",
             f"A2 AUC {a2['auc']:.4f}" if a2.get("auc") is not None else "A2 not established",
             f"A3a {_pts(a3a['lgo_macro_delta'])} pts" if a3a.get("lgo_macro_delta") is not None else "A3a not established",
             f"A3b {_short(record['A3b'].get('pass'))}",
             f"A4 {_short(a4.get('pass'))}" + (f" (blocking {', '.join(a4['blocking_strata'])})" if a4.get("blocking_strata") else ""),
             "A5 " + _a5(record.get("A5")).replace("**", ""), f"A6 {_short((record.get('A6') or {}).get('pass'))}"]
    terms = f" with context terms {', '.join(record['context_terms'])}" if record.get("context_terms") else ""
    scope = f"union-fuser admission; the {lane} lane is admitted separately ({LANE_RECORDS[lane]})" if lane else ""
    gate = (record.get("gates") or {}).get(record["detector_id"])
    outcome = f"admitted; used only where {gate_words(gate)}" if gate else record["decision"]  # D-51: the gate in words
    text = (f"{item} D-44 admission on {bench} fit ∪ cal ({evidence}{'; ' + scope if scope else ''}): {outcome}{terms}; "
            + "; ".join(parts) + "; automated record pending human review")
    served = {"served_until": served_until} if served_until and decision in ("rejected", "retired") else {}
    return {"decision": decision, **({} if decision == "pending" else {"date": date}), **served, "evidence": text}


_ID = re.compile(r"^  - id: (\S+)\s*$")
_ADMISSION = re.compile(r"^(\s+)admission: \{.*\}\s*$")


def update_card_lines(text: str, cards: Mapping[str, Mapping]) -> str:
    """Replace the one-line `admission:` entry of every detector in `cards`; refuses an id without exactly one such line."""
    lines, current, seen = text.splitlines(keepends=True), None, {d: 0 for d in cards}
    for i, line in enumerate(lines):
        match = _ID.match(line)
        if match:
            current = match.group(1)
            continue
        found = _ADMISSION.match(line)
        if found and current in cards:
            card = cards[current]
            fields = [f"decision: {card['decision']}"] + ([f"date: {json.dumps(card['date'])}"] if card.get("date") else [])
            fields += [f"served_until: {json.dumps(card['served_until'])}"] if card.get("served_until") else []
            fields.append(f"evidence: {json.dumps(card['evidence'], ensure_ascii=False)}")
            lines[i] = f"{found.group(1)}admission: {{{', '.join(fields)}}}\n"
            seen[current] += 1
    wrong = sorted(d for d, n in seen.items() if n != 1)
    if wrong:
        raise ValueError(f"no single-line `admission:` entry for {', '.join(wrong)}")
    return "".join(lines)
