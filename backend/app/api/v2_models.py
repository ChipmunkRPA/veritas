"""API v2 models catalogue (ARCHITECTURE §5.2 `GET /models`, VG-01).

Every manifest detector's model card, its complete model dependency closure with licences and pinned revisions or checksums,
and whether the registry enables it under the licence policy in force (with the reasons when it does not). Benchmark fields (`eval`,
`validated_slices`) are whatever `vbench_report.py` wrote into the manifest; the API never fills them.

E-7 adds the runtime state of each card: whether every required model is on disk (`installed`) and whether the model manager
holds its direct runtime bundles (`loaded`). Per-component state reports only independently tracked model entries.
Engineering records are not product content (D-71, N-10). The one document of the repository the API serves is a detector's
admission record (`docs/v2/bench/admissions/<id>.md`, ML §3.2), to administrators only: their card lists it, and every other
role's card says only whether one is on file. Review documents and QA records are never served, whatever they cite. The
record is served as text only when it is this detector's, so a record id never reaches the filesystem as a path.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from app.api.security import CheckedRoute, authorize, rate_limit
from app.api.v2 import DEFAULT_LIMIT, NO_STORE, Cursor, Limit, Page, ProblemError, paginate, problem_responses
from app.engine import card_copy
from app.engine.manifest import DetectorEntry, ModelEntry
from app.engine.registry import DetectorRegistry, get_registry

router = APIRouter(prefix="/api/v2", tags=["v2"], dependencies=[Depends(rate_limit), Depends(authorize)], route_class=CheckedRoute)

# The documents tree that holds the admission records; absent in images built without it (no records then).
DOCS_ROOT = Path(__file__).resolve().parents[3] / "docs" / "v2"
# the shape of a record id. `qa:` ids were served until N-10 and are still well-formed: they name nothing (404)
RECORD_ID = r"^(admission|qa):[A-Za-z0-9_.\-]+$"


class ModelState(ModelEntry):
    installed: bool = Field(description="Every pinned file of the model is on disk")
    loaded: bool = Field(description="The model manager independently tracks this model ID in memory; false does not rule out its use inside another model's runtime bundle")
    loaded_devices: list[str] = Field(default_factory=list)


class RecordRef(BaseModel):
    id: str = Field(description="`admission:<detector id>`")
    kind: Literal["admission"]
    path: str = Field(description="Repository path of the record")
    title: str


class EngineeringNote(BaseModel):
    field: str = Field(description="What the string is, in words")
    text: str = Field(description="The manifest string, verbatim")
    withheld: bool = Field(default=True, description="False when the client wording prints the string without its references")


class CardCopy(BaseModel):
    """The card as a client page prints it (L-7): no ticket or decision ids, repository paths or raw enums."""
    family: str
    hypotheses: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)
    lane: str | None = None
    citation: str | None = None
    training_data: str | None = None
    coverage_date: str | None = None
    input_constraints: list[str] = Field(default_factory=list)
    contamination_sets: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list, description="False-positive modes, then known limitations")
    enablement_reasons: list[str] = Field(default_factory=list)
    licence_policy: str
    source: str | None = Field(default=None, description="Where the check comes from: a download address, or that it is part of Veritas")
    disclosures: list[str] = Field(default_factory=list)
    licence_notices: dict[str, str] = Field(default_factory=dict, description="Model id → licence notice")
    record_titles: dict[str, str] = Field(default_factory=dict, description="Record id → title")
    truncated: list[str] = Field(default_factory=list, description="One-value fields whose wording is the lead of the recorded value")
    engineering: list[EngineeringNote] = Field(default_factory=list, description="Recorded strings the client wording shortens or leaves out")


class ModelCard(DetectorEntry):
    enabled: bool
    enablement_reasons: list[str] = Field(default_factory=list)
    licence_policy: str
    disclosures: list[str] = Field(default_factory=list)  # D-34
    model_entries: list[ModelState] = Field(default_factory=list)
    installed: bool = Field(description="Every direct and transitive model the detector needs is installed (true for a model-free detector)")
    loaded: bool | None = Field(description="Every direct model has a runtime bundle tracked in memory; dependencies may be held inside those bundles without separate entries; null for a model-free detector")
    admission_record: bool = Field(default=False, description="An admission record of this detector is on file")
    records: list[RecordRef] = Field(default_factory=list, description="The admission record, on an administrator's card; empty for every other role")
    client: CardCopy | None = None


@lru_cache(maxsize=256)
def _text(path: Path, mtime_ns: int) -> str:  # keyed on mtime so an edited record is re-read
    return path.read_text(encoding="utf-8", errors="replace")


def _read(path: Path) -> str:
    return _text(path, path.stat().st_mtime_ns)


def _title(text: str, fallback: str) -> str:
    return next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")), fallback)


def _records(detector_id: str) -> list[tuple[RecordRef, Path]]:
    """The detector's admission record, when one is on file: the only document served (D-71). `detector_id` is an id of the
    manifest, never the request's text."""
    admission = DOCS_ROOT / "bench" / "admissions" / f"{detector_id}.md"
    if not admission.is_file():
        return []
    return [(RecordRef(id=f"admission:{detector_id}", kind="admission", path=f"docs/v2/bench/admissions/{admission.name}",
                       title=_title(_read(admission), f"Admission record: {detector_id}")), admission)]


def _loaded_models(registry: DetectorRegistry) -> dict[str, list[str]]:
    """Manifest model id → devices it is loaded on. v1 registry names map to the manifest models behind them."""
    from app.engine.models import get_model_manager, v1_model_ids

    out: dict[str, list[str]] = {}
    for info in get_model_manager().snapshot():
        ids = v1_model_ids(info.id[3:], registry.manifest) if info.id.startswith("v1:") else (info.id,)
        for model_id in ids:
            out.setdefault(model_id, [])
            if info.device not in out[model_id]:
                out[model_id].append(info.device)
    return out


def _card(registry: DetectorRegistry, det: DetectorEntry, loaded: dict[str, list[str]], *, administrator: bool = False) -> dict[str, Any]:
    """The card of one detector. Only an administrator's card lists the admission record (D-71)."""
    from app.engine.fetchers import installed

    decision = registry.decision(det.id)
    records = [ref for ref, _ in _records(det.id)]
    required = [m for m in (registry.manifest.model(i) for i in registry.required_models(det)) if m is not None]
    states = {m.id: ModelState(**m.model_dump(by_alias=True), installed=installed(m), loaded=m.id in loaded,
                               loaded_devices=loaded.get(m.id, [])) for m in required}
    card = ModelCard(**det.model_dump(), enabled=decision.enabled, enablement_reasons=list(decision.reasons),
                     licence_policy=registry.policy.label, disclosures=registry.disclosures([det.id]),
                     model_entries=list(states.values()),  # required_models preserves direct order, then breadth-first dependencies
                     installed=all(s.installed for s in states.values()),
                     loaded=all(i in states and states[i].loaded for i in det.models) if det.models else None,
                     admission_record=bool(records), records=records if administrator else [])
    out = card.model_dump(mode="json", by_alias=True)
    out["client"] = _client(registry, out).model_dump(mode="json")
    return out


def _quoted(registry: DetectorRegistry, card: dict[str, Any]) -> list[str]:
    """Identifiers the card page prints as identifiers: detector and model ids, upstream repository and weight-file names."""
    out = [d.id for d in registry.manifest.detectors] + [m.id for m in registry.manifest.models]
    for m in card.get("model_entries") or []:
        source = m["source"]
        for value in (source.get("repo"), source.get("entrypoint")):
            out += [value, *value.split("/")] if value else []
        for f in source.get("files") or []:
            name = f["path"].rsplit("/", 1)[-1]
            stem = name.rsplit(".", 1)[0]
            out += [f["path"], name, *([stem] if "_" in stem else [])]
    out += [part for part in (card.get("source_url") or "").split("/") if "_" in part]
    return out


def _client(registry: DetectorRegistry, card: dict[str, Any]) -> CardCopy:
    """The card in client words (L-7, design review P1-3): enums as words, prose without its references, and verbatim under
    `engineering` every string the client wording shortens or leaves out."""
    quoted = _quoted(registry, card)
    engineering: list[EngineeringNote] = []
    truncated: list[str] = []

    def one(label: str, field: str) -> str | None:
        text, complete = card_copy.lead(card.get(field), quoted)
        if card.get(field) and text != card[field]:
            engineering.append(EngineeringNote(field=label, text=card[field], withheld=text is None))
        if text is not None and not complete:
            truncated.append(field)
        return text

    def many(label: str, values: list[str]) -> list[str]:
        out: list[str] = []
        for value in values:
            text = card_copy.client_text(value, quoted)
            if text != value:
                engineering.append(EngineeringNote(field=label, text=value, withheld=text is None))
            if text is not None and text not in out:
                out.append(text)
        return out

    citation, training, coverage = one("Citation", "citation"), one("Training data", "training_data"), one("Coverage date", "coverage_date")
    inputs = many("Intended input", card.get("input_constraints") or [])
    contamination = many("Contamination set", card.get("contamination_sets") or [])
    failure_modes = many("False-positive mode", card.get("fp_modes") or []) + [
        m for m in many("Known limitation", card.get("known_limitations") or []) if m not in (card.get("fp_modes") or [])]
    admission = card.get("admission") or {}
    if admission.get("evidence"):
        engineering.append(EngineeringNote(field="Admission evidence", text=admission["evidence"]))
    notices: dict[str, str] = {}
    for m in card.get("model_entries") or []:
        notice = (m.get("licence") or {}).get("notice")
        text = card_copy.client_text(notice, quoted)
        if notice and text != notice:
            engineering.append(EngineeringNote(field=f"Licence notice of {m['id']}", text=notice, withheld=text is None))
        if text:
            notices[m["id"]] = text
    if card.get("adapter"):
        engineering.append(EngineeringNote(field="Adapter", text=card["adapter"]))
    source = card_copy.source_words(card.get("source_url"))  # N-19d: no repository path in the Source row
    if card.get("source_url") and source != card["source_url"]:
        engineering.append(EngineeringNote(field="Source", text=card["source_url"], withheld=False))
    reasons = list(card.get("enablement_reasons") or [])
    if admission.get("criteria"):
        engineering.append(EngineeringNote(field="Admission criteria", text=json.dumps(admission["criteria"], sort_keys=True)))
    engineering += [EngineeringNote(field="Enablement reason", text=r, withheld=False) for r in reasons if card_copy.reason_words(r) != r]
    titles: dict[str, str] = {}
    for record in card.get("records") or []:
        title = card_copy.record_title(record["title"], card["name"], quoted)
        taken = sum(1 for t in titles.values() if t == title or t.startswith(f"{title} "))
        titles[record["id"]] = f"{title} {taken + 1}" if taken else title
    return CardCopy(
        family=card_copy.words(card["family"]),
        hypotheses=[card_copy.HYPOTHESIS_WORDS.get(h, card_copy.words(h)) for h in card.get("hypotheses") or []],
        threats=[card_copy.THREAT_WORDS.get(t, card_copy.words(t)) for t in card.get("threat") or []],
        lane=card_copy.THREAT_WORDS.get(card["lane"], card_copy.words(card["lane"])) if card.get("lane") else None,
        citation=citation, training_data=training, coverage_date=coverage, input_constraints=inputs,
        contamination_sets=contamination, failure_modes=failure_modes,
        enablement_reasons=[card_copy.reason_words(r) for r in reasons], licence_policy=card_copy.policy_words(card["licence_policy"]),
        source=source,
        disclosures=many("Disclosure", card.get("disclosures") or []), licence_notices=notices, record_titles=titles,
        truncated=truncated, engineering=engineering)


def card_prose(card: dict[str, Any]) -> list[tuple[str, str]]:
    """The strings the model-card page prints as prose outside its engineering record: (field, text) pairs of `client`."""
    client = card["client"]
    out = [(f, client[f]) for f in ("family", "lane", "citation", "training_data", "coverage_date", "licence_policy", "source") if client.get(f)]
    out += [(f, v) for f in ("hypotheses", "threats", "input_constraints", "contamination_sets", "failure_modes",
                             "enablement_reasons", "disclosures") for v in client[f]]
    out += [("licence_notices", v) for v in client["licence_notices"].values()]
    out += [("record_titles", v) for v in client["record_titles"].values()]
    return [("name", card["name"]), *out]


def raw_prose(card: dict[str, Any]) -> list[tuple[str, str]]:
    """The manifest strings the page printed as prose before L-7 (the probe's `--raw` view)."""
    scalars = ("name", "family", "citation", "training_data", "coverage_date", "lane", "licence_policy")
    lists = ("enablement_reasons", "disclosures", "hypotheses", "threat", "input_constraints", "contamination_sets", "fp_modes",
             "known_limitations")
    out: list[tuple[str, str]] = [(f, card[f]) for f in scalars if isinstance(card.get(f), str) and card[f]]
    out += [(f, v) for f in lists for v in card.get(f) or [] if isinstance(v, str)]
    admission = card.get("admission") or {}
    if admission.get("evidence"):
        out.append(("admission.evidence", admission["evidence"]))
    out += [("records.title", r["title"]) for r in card.get("records") or []]
    out += [("model_entries.licence.notice", m["licence"]["notice"]) for m in card.get("model_entries") or []
            if (m.get("licence") or {}).get("notice")]
    return out


@router.get("/models", response_model=Page[ModelCard], responses=problem_responses(400, 401, 422))
def list_models(request: Request, cursor: Cursor = None, limit: Limit = DEFAULT_LIMIT, modality: str | None = None, enabled: bool | None = None):
    """Model cards by detector id."""
    registry = get_registry()
    loaded = _loaded_models(registry)
    detectors = sorted(registry.manifest.detectors, key=lambda d: d.id)
    administrator = request.state.principal.satisfies("administrator")
    cards = [_card(registry, d, loaded, administrator=administrator) for d in detectors
             if modality is None or str(getattr(d.modality, "value", d.modality)) == modality]
    if enabled is not None:
        cards = [c for c in cards if c["enabled"] is enabled]
    page, next_cursor = paginate(cards, lambda c: (c["id"],), cursor, limit)
    return JSONResponse(content={"items": page, "next_cursor": next_cursor}, headers=NO_STORE)


@router.get("/models/{detector_id}", response_model=ModelCard, responses=problem_responses(401, 404))
def get_model(detector_id: str, request: Request):
    registry = get_registry()
    det = registry.manifest.detector(detector_id)
    if det is None:
        raise ProblemError(404, "Detector not found", "No detector with this id is in the models manifest.")
    administrator = request.state.principal.satisfies("administrator")
    return JSONResponse(content=_card(registry, det, _loaded_models(registry), administrator=administrator), headers=NO_STORE)


@router.get("/models/{detector_id}/records/{record_id}", response_class=Response,
            responses={200: {"content": {"text/markdown": {"schema": {"type": "string"}}}}, **problem_responses(401, 403, 404, 422, 500)})
def get_model_record(detector_id: str, record_id: str = PathParam(pattern=RECORD_ID)):
    """The admission record of the detector, as Markdown text. Administrators only."""
    det = get_registry().manifest.detector(detector_id)
    if det is None:
        raise ProblemError(404, "Detector not found", "No detector with this id is in the models manifest.")
    path = next((p for ref, p in _records(det.id) if ref.id == record_id), None)
    if path is None:
        raise ProblemError(404, "Record not found", "This detector has no record with this id.")
    try:
        text = _read(path)
    except OSError as exc:
        raise ProblemError(500, "Record unreadable", "The record could not be read.") from exc
    return Response(content=text, media_type="text/markdown; charset=utf-8", headers=NO_STORE)
