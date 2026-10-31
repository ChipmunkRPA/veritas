"""D-53 backward elimination for J-5's image set (BUILD_SPEC J-3c1). Pure functions over admission records; the CLI
`scripts/j3c1_eliminate.py` keeps the ledger and prints the bg.sh commands.

- **Pool** (`pool`): from `v2.1.1c-pass1-admissions.json`, since A1, A2, A6, H-3 retirement and lane do not depend on the
  base. A member is a union-fuser candidate (no lane of its own) whose A1, A2 and A6 pass and that H-3 did not retire,
  with its H-3 gate. Every other detector is excluded with its reason; `img.watermark.dwtdct` is refused under D-50.
- **Member and base.** A member is (id, sorted gate literals); a base is a frozenset of members (`base_env` gives the
  `j3_ablate.sh` environment, members ablated as they stand: `J3_OWN_GATES=1`).
- **Step** (`next_step`, D-53 (a)–(c)), for a base of two or more members and its step record (`admission_record.py
  --detectors <members> --carried-gates <D-53 gates> [--gated-ablations]`):
  - a member is admitted as it stands when its decision is admitted (with or without terms or gate) from its main
    ablation (no gated retry);
  - (a) every member admitted: stop;
  - a not-admitted member without an A3a lower bound refuses the run (execution failure);
  - (b) the not-admitted members without a gate whose gated retry qualifies are retried; while one has no retry, the step
    asks for the retry job. If any retry admits, the only change is the gate of the admitted retry with the highest gated
    A3a lower bound (ties: the higher gated point estimate, then the id); the gate stays in every later base;
  - (c) otherwise exactly one not-admitted member is removed: members whose main A4 lists a blocking stratum go first; then
    the lowest A3a lower bound (ties: the lower point estimate, then the id).
  - A member already carrying a gate (H-3 or D-53) does not qualify for a retry. A removed member never returns.
- **Refusals** (the run stops and J-5 is blocked): a repeated base, a 10th ablated step (the lone-member stop is not one),
  a not-admitted member without an A3a lower bound, a removed member in the base, a member without a record.
- **Lone member** (`lone_member_decision`): never ablated and never dropped; admitted (with its gate) only when the joint
  check passes and its main ablation in the two-member step lists no blocking stratum; otherwise not established.
- **Joint check** (`joint_verdict`): S_end ("with") against the served set ("without"), refitted in the same harness. It
  passes when A3 (A3a or A3b) and A4 pass and S_end's served-rule weighted out-of-fold FPR is ≤ 6.0 %. No joint check
  runs when S_end equals the served set with the same gates.
- **Add-back diagnostic** (`addback_diagnostic`): each removed member ablated added back to S_end; any that would be
  admitted is disclosed as order dependence, and S_end does not change.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

Member = tuple[str, tuple[str, ...]]
Base = frozenset[Member]

SERVED_SET: Base = frozenset({("img.commfor384", ()), ("img.dda", ()), ("img.dfdc_b7", ("has_face",))})
MAX_ABLATED_STEPS = 9
JOINT_FPR_MAX = 0.06
ADMITTED = ("admitted", "admitted_context_gated")
REFUSED_UNDER_D50 = {"img.watermark.dwtdct": "refused under D-50"}


class EliminationError(ValueError):
    """The elimination cannot continue on these inputs."""


def member(detector_id: str, gate: Iterable[str] = ()) -> Member:
    return (detector_id, tuple(sorted(gate)))


def base_words(base: Iterable[Member]) -> str:
    return "{" + ", ".join(d + (f" {','.join(g)}" if g else "") for d, g in sorted(base)) + "}"


def base_json(base: Iterable[Member]) -> list[dict]:
    return [{"detector_id": d, "gate": list(g)} for d, g in sorted(base)]


def base_from_json(entries: Sequence[Mapping]) -> Base:
    return frozenset(member(e["detector_id"], e.get("gate") or ()) for e in entries)


def gate_string(gates: Mapping[str, Sequence[str]]) -> str:
    return ";".join(f"{d}={'|'.join(g)}" for d, g in sorted(gates.items()) if g)


def pool(record: Mapping) -> dict:
    """{members: [Member], excluded: {id: reason}} from the pass-1 admission record."""
    members: list[Member] = []
    excluded: dict[str, str] = {}
    for r in record["records"]:
        detector_id, reasons = r["detector_id"], []
        if detector_id in REFUSED_UNDER_D50:
            reasons.append(REFUSED_UNDER_D50[detector_id])
        if r.get("lane"):
            reasons.append(f"own lane ({r['lane']})")
        for key in ("A1", "A2", "A6"):
            verdict = (r.get(key) or {}).get("pass")
            if verdict is not True:
                reasons.append(f"{key} {'fails' if verdict is False else 'not established'}"
                               + (f" ({r['A1']['reason']})" if key == "A1" else ""))
        if r.get("h3_decision") == "retire":
            reasons.append("retired by H-3")
        if reasons:
            excluded[detector_id] = "; ".join(reasons)
        else:
            members.append(member(detector_id, r.get("h3_features") or () if r.get("h3_decision") == "context_gate" else ()))
    return {"members": sorted(members), "excluded": dict(sorted(excluded.items()))}


def base_env(base: Base) -> dict[str, str]:
    """`j3_ablate.sh` environment for a step's main job: every member as it stands, gate included."""
    return {"J3_BASE": ",".join(sorted(d for d, _ in base)), "J3_GATES": gate_string(dict(base)), "J3_OWN_GATES": "1",
            "candidates": " ".join(sorted(d for d, _ in base))}


def retry_env(base: Base, detector_id: str, literals: Sequence[str]) -> dict[str, str]:
    """The gated retry of one member: the base's gates plus the member's negated term (`J3_RETRY=1`)."""
    gates = {**dict(base), detector_id: tuple(literals)}
    return {"J3_BASE": ",".join(sorted(d for d, _ in base)), "J3_GATES": gate_string(gates), "J3_RETRY": "1",
            "candidates": detector_id}


def _records(doc: Mapping | None) -> dict[str, Mapping]:
    return {r["detector_id"]: r for r in (doc or {}).get("records") or []}


def admitted_as_it_stands(record: Mapping) -> bool:
    return record["decision"] in ADMITTED and not record.get("gated_ablation")


def _a3a(record: Mapping, *, gated: bool = False) -> tuple[float | None, float | None]:
    entry = ((record.get("gated_ablation") or {}).get("A3a") if gated else record.get("A3a")) or {}
    return entry.get("ci95_lo"), entry.get("lgo_macro_delta")


def _tie_text(pick: str, ties_lo: Sequence[str], ties_point: Sequence[str], point_words: str) -> str:
    if len(ties_lo) < 2:
        return "none"
    others = ", ".join(d for d in ties_lo if d != pick)
    if len(ties_point) < 2:
        return f"tied A3a lower bound with {others}: {point_words}"
    return f"tied A3a lower bound and point estimate with {', '.join(d for d in ties_point if d != pick)}: detector id"


def next_step(base: Iterable[Member], record: Mapping, gated_record: Mapping | None = None, *, removed: Iterable[str] = (),
              history: Iterable[Iterable[Member]] = (), step: int) -> dict:
    """The D-53 change after ablated step `step` of `base` (see the module docstring). `record` is the step record; the retry
    outcomes come from `gated_record` when given, else from `record` (a record built with `--gated-ablations`)."""
    base = frozenset(base)
    out: dict = {"step": step, "base": base_json(base)}
    removed = set(removed)
    if removed & {d for d, _ in base}:
        return {**out, "action": "refuse", "clause": "refusal", "reason": f"removed members in the base: {sorted(removed & {d for d, _ in base})}"}
    if len(base) < 2:
        return {**out, "action": "stop", "clause": "lone member",
                "reason": "a one-member base is never ablated and never dropped; the joint check stands in for its leave-one-out"}
    recs = _records(record)
    missing = sorted(d for d, _ in base if d not in recs)
    if missing:
        return {**out, "action": "refuse", "clause": "refusal", "reason": f"no step record for {missing}"}
    gates = dict(base)
    not_admitted = sorted(d for d in gates if not admitted_as_it_stands(recs[d]))
    out["not_admitted"] = not_admitted
    if not not_admitted:
        return {**out, "action": "stop", "clause": "(a)", "reason": "every member is admitted as it stands"}
    no_lower = [d for d in not_admitted if _a3a(recs[d])[0] is None]
    if no_lower:
        return {**out, "action": "refuse", "clause": "refusal",
                "reason": f"not-admitted members without an A3a lower bound (execution failure): {no_lower}"}
    retries = _records(gated_record) if gated_record is not None else recs
    qualifying = [d for d in not_admitted if not gates[d] and (recs[d].get("gated_retry") or {}).get("qualifies") is True]
    pending = [d for d in qualifying if not (retries.get(d) or {}).get("gated_ablation")]
    if pending:
        return {**out, "action": "retry", "clause": "(b)", "reason": "gated retries qualify and have not run",
                "retries": {d: list(recs[d]["gated_retry"]["gate"]) for d in pending}}
    admitted = [d for d in qualifying if retries[d]["decision"] == "admitted_context_gated" and (retries[d].get("gates") or {}).get(d)]
    history = [frozenset(h) for h in history]
    if admitted:
        def gated_key(d: str) -> tuple:
            lo, point = _a3a(retries[d], gated=True)
            return (float("inf") if lo is None else -lo, float("inf") if point is None else -point, d)

        pick = min(admitted, key=gated_key)
        ties_lo = [d for d in admitted if gated_key(d)[0] == gated_key(pick)[0]]
        ties_point = [d for d in ties_lo if gated_key(d)[1] == gated_key(pick)[1]]
        literals = tuple(retries[pick]["gates"][pick])
        next_base = (base - {(pick, ())}) | {member(pick, literals)}
        change = {"action": "gate", "clause": "(b)", "detector_id": pick, "gate": list(literals),
                  "gated_a3a_lower": _a3a(retries[pick], gated=True)[0], "admitted_retries": admitted,
                  "tie_break": _tie_text(pick, ties_lo, ties_point, "higher gated point estimate"),
                  "reason": f"gated retry admits {pick}={','.join(literals)} with the highest gated A3a lower bound"}
    else:
        blocked = [d for d in not_admitted if (recs[d].get("A4") or {}).get("blocking_strata")]
        candidates = blocked or not_admitted

        def removal_key(d: str) -> tuple:
            lo, point = _a3a(recs[d])
            return (lo, float("inf") if point is None else point, d)

        pick = min(candidates, key=removal_key)
        ties_lo = [d for d in candidates if removal_key(d)[0] == removal_key(pick)[0]]
        ties_point = [d for d in ties_lo if removal_key(d)[1] == removal_key(pick)[1]]
        next_base = frozenset(m for m in base if m[0] != pick)
        change = {"action": "remove", "clause": "(c)", "detector_id": pick, "a3a_lower": _a3a(recs[pick])[0],
                  "blocking_first": bool(blocked), "blocking_strata": list(recs[pick]["A4"].get("blocking_strata") or []),
                  "tie_break": _tie_text(pick, ties_lo, ties_point, "lower point estimate"),
                  "reason": (f"{pick} lists blocking strata {', '.join(recs[pick]['A4']['blocking_strata'])} and has the lowest A3a "
                             "lower bound among the members with established harm" if blocked else
                             f"{pick} has the lowest A3a lower bound among the not-admitted members")}
    if next_base in history or next_base == base:
        return {**out, **change, "action": "refuse", "clause": "refusal", "change": change["action"],
                "reason": f"repeated base {base_words(next_base)}", "next_base": base_json(next_base)}
    if len(next_base) >= 2 and step + 1 > MAX_ABLATED_STEPS:
        return {**out, **change, "action": "refuse", "clause": "refusal", "change": change["action"],
                "reason": f"a {step + 1}th ablated step exceeds {MAX_ABLATED_STEPS}", "next_base": base_json(next_base)}
    return {**out, **change, "next_base": base_json(next_base)}


def needs_joint_check(end_base: Iterable[Member]) -> bool:
    return frozenset(end_base) != SERVED_SET


def joint_verdict(comparison: Mapping) -> dict:
    """The joint check from a `calibrate --reference` comparison ("with" = S_end, "without" = the served set)."""
    crit, cross, lgo = comparison["criteria"], comparison["cross_fit"], comparison["lgo"]
    fpr = ((cross.get("operating_point") or {}).get("with") or {}).get("served_rule_weighted_fpr")
    checks = {"A3": crit["A3"]["pass"], "A4": crit["A4"]["pass"], "served_rule_fpr": None if fpr is None else bool(fpr <= JOINT_FPR_MAX)}
    passed = False if False in checks.values() else (None if None in checks.values() else True)
    return {"passed": passed, "checks": checks, "served_rule_weighted_fpr": {
                name: ((cross.get("operating_point") or {}).get(name) or {}).get("served_rule_weighted_fpr") for name in ("without", "with")},
            "lgo_macro_delta": lgo["macro"]["delta"], "lgo_macro_ci95": lgo["macro"].get("ci95"),
            "A3a": crit["A3a"], "A3b": {k: crit["A3b"].get(k) for k in ("pass", "qualifying")},
            "A4": {k: crit["A4"].get(k) for k in ("pass", "pooled_delta", "pooled_ci95_lo", "pooled_ci95_hi", "blocking_strata")},
            "rule": "A3 (A3a or A3b) and A4 pass, and S_end's served-rule weighted out-of-fold FPR ≤ 6.0 % (D-53)"}


def lone_member_decision(step_record: Mapping, joint: Mapping | None, detector_id: str, gate: Sequence[str] = ()) -> dict:
    """The last member's record decision from its two-member-step main ablation and the joint check."""
    r = _records(step_record).get(detector_id)
    if r is None:
        raise EliminationError(f"no two-member step record for {detector_id}")
    blocking = list((r.get("A4") or {}).get("blocking_strata") or [])
    verdict = None if joint is None else joint.get("passed")
    words = "passes" if verdict is True else ("fails" if verdict is False else "not run")
    admitted = verdict is True and not blocking
    decision = ("admitted_context_gated" if gate else "admitted") if admitted else "not_established"
    return {"decision": decision, "gates": {detector_id: list(gate)} if admitted and gate and any(g.startswith("!") for g in gate) else {},
            "reason": f"last member (D-53): no leave-one-out baseline; joint check {words}; blocking strata "
                      f"{', '.join(blocking) or 'none'}", "blocks_j5": not admitted}


def addback_diagnostic(addback_record: Mapping | None, removed: Iterable[str]) -> dict:
    """Removed members that would be admitted when added back to S_end (order dependence, disclosed only)."""
    recs = _records(addback_record)
    rows = {d: (recs[d]["decision"] if d in recs else None) for d in sorted(removed)}
    return {"decisions": rows, "order_dependent": [d for d, v in rows.items() if v in ADMITTED], "end_set_changes": False}
