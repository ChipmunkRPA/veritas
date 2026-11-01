"""Summarize all 38 frozen web samples using the public customer assessment.

This reads saved API responses only. It never runs inference, changes thresholds,
or copies source media/text into the report. Refuse a final report if any planned
item is absent, unfinished, changed, or inconsistent between analysis and export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BACKEND / "data/independent-evaluation/multimodal"
DEFAULT_OUTPUT = BACKEND.parent / "output/evaluation"
CATEGORIES = {
    "ai": "Leaning AI", "original": "Leaning original", "record": "Provenance record",
    "inconclusive": "Inconclusive", "failed": "Failed",
}
KINDS = {"image": "Image", "video": "Video", "document": "Article"}
COHORTS = {"main": "Main sample set", "challenge": "Newer video challenges"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def text(value: object) -> str:
    return html.escape(str(value), quote=True)


def safe_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Unsupported link: {value}")
    return text(value)


def classify(detail: dict) -> str:
    """Only the public conclusion determines whether a tendency was published."""
    if detail.get("status") != "done":
        return "failed"
    assessment = detail.get("customer_assessment") or {}
    if assessment.get("direction") == "record":
        return "record"
    direction = assessment.get("direction")
    confidence = assessment.get("confidence_pct")
    if direction in {"ai", "original"}:
        if (assessment.get("applies") is not True
                or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not assessment.get("threshold_pct", 80) <= confidence <= 100):
            raise ValueError("Published tendency lacks a valid public confidence estimate")
        return direction
    return "inconclusive"


def document_observations(exported: dict) -> dict:
    """Inspect exact local positions without reproducing copyright source prose."""
    result = exported.get("result") or {}
    evidence = exported.get("evidence") or {}
    packets = evidence.get("packets") or []
    extracted = result.get("text") or ""
    positions = []
    directions = Counter()
    for packet in packets:
        focus = packet.get("focus") or {}
        start, end = focus.get("start"), focus.get("end")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(extracted)):
            raise ValueError(f"Invalid document focus position: {packet.get('id')}")
        if extracted[start:end] != focus.get("text"):
            raise ValueError(f"Document focus does not match extracted text: {packet.get('id')}")
        responses = {w.get("direction") for w in packet.get("supporting_windows", [])}
        if {"supports_ai", "opposes_ai"} <= responses:
            direction = "conflicting"
        elif responses == {"supports_ai"}:
            direction = "ai_side_only"
        elif responses == {"opposes_ai"}:
            direction = "opposing_side_only"
        else:
            direction = "other"
        directions[direction] += 1
        positions.append({"id": packet.get("id"), "start": start, "end": end, "responses": direction})
    return {
        "displayed_passages": len(packets), "verified_positions": len(positions),
        "displayed_passage_responses": dict(directions), "positions": positions,
        "candidate_spans": evidence.get("coverage", {}).get("candidate_spans"),
        "omitted_passages": evidence.get("omitted_windows", 0),
        "legacy_ai_sentence_flags": result.get("stats", {}).get("ai_sentences", 0),
        "legacy_uncalibrated_score": result.get("verdict", {}).get("probability_ai"),
        "legacy_label": result.get("verdict", {}).get("label"),
    }


def reason_for(detail: dict, category: str) -> str:
    assessment = detail.get("customer_assessment") or {}
    if category == "failed":
        return "Analysis did not complete successfully. This item remains in the sample count."
    if category == "record":
        record = (detail.get("decision") or {}).get("provenance_record") or {}
        if record.get("kind") == "self_declared_ai":
            return "Embedded provenance declares AI generation, but the signer is not on the trust list. This is an attachable self-declaration, not independent certification or detection from content alone."
        return "The conclusion comes from an embedded provenance record. It is not converted into confidence or counted as a content-based tendency."
    if category != "inconclusive":
        return "—"
    headline = assessment.get("headline", "")
    if "no calibrated estimate" in headline:
        return "Text checks provide responses and locations, but no reliable document-level probability has been calibrated. Check scores are not confidence estimates."
    if "outside the verified scope" in headline:
        condition = detail.get("condition_profile") or {}
        if detail.get("kind") == "image" and condition.get("min_face_px"):
            short_side = min(condition.get("width", 0), condition.get("height", 0))
            return f"Short edge: {short_side} pixels; face size: approximately {condition['min_face_px']:g} pixels. These input conditions lack sufficient validation samples from both classes."
        duration = condition.get("duration_s")
        if duration is not None and duration < 4:
            return f"Duration: {duration:g} seconds, below four seconds. Validation coverage for short clips is insufficient."
        if detail.get("kind") == "video":
            return "This combination of resolution, duration and face size lacks sufficient validation samples from both classes. Percentages from broader conditions cannot be carried over."
        return "Validation coverage for these input conditions is insufficient to publish a confidence estimate."
    caveats = " ".join(detail.get("caveats") or []).lower()
    if "strongly disagree" in caveats:
        return "Checks strongly disagree. Even with a high saved fusion estimate, the final result publishes neither an AI nor an original-file tendency."
    if "reliable tendency" in headline:
        return "Checks conflict or do not provide enough joint support. The saved fusion estimate cannot serve as the final conclusion on its own."
    return "Available evidence does not support a reliable tendency. See the limitations in the local analysis."


def scope_for(source: dict, assessment: dict) -> str:
    if source["kind"] == "document":
        return "English news articles and local passages. Chinese text, email, scanned documents and other domains were not validated."
    if source["kind"] == "image":
        return "Combined checks of the image as a whole. This does not establish reliable localization for every region."
    has_audio = any(s.get("codec_type") == "audio" for s in source.get("media", {}).get("streams", []))
    if assessment.get("scope") in {"visual_content", "visual_only"}:
        return "Visual content only. Audio is inspected separately and excluded from this percentage; it cannot establish that speech is genuine."
    if not has_audio:
        return "Visual checks only. The original has no audio track, so it cannot assess speech or background-audio detection."
    return "Combined file checks. Any audio conclusion is limited to the coverage stated in the analysis."


def load_rows(base: Path) -> tuple[list[dict], list[dict], dict]:
    main_path, challenge_path = base / "manifest.json", base / "challenge-manifest.json"
    manifests = {"main": read_json(main_path), "challenge": read_json(challenge_path)}
    expected = Counter((cohort, item["kind"], item["label"])
                       for cohort, manifest in manifests.items() for item in manifest["items"])
    target = Counter({("main", "image", "ai"): 8, ("main", "image", "real"): 8,
                      ("main", "video", "ai"): 4, ("main", "video", "real"): 4,
                      ("main", "document", "ai"): 6, ("main", "document", "real"): 6,
                      ("challenge", "video", "ai"): 2})
    if expected != target:
        raise ValueError("This report requires the frozen 36 main samples and 2 separate video challenges")
    rows, diagnostics, missing = [], [], []
    ids, hashes = set(), set()
    for cohort, manifest in manifests.items():
        for source in manifest["items"]:
            sid = source["id"]
            if sid in ids or source["sha256"] in hashes:
                raise ValueError(f"Duplicate sample or content hash: {sid}")
            ids.add(sid)
            hashes.add(source["sha256"])
            run_name = "challenge" if cohort == "challenge" else source["kind"]
            folder = base / f"{run_name}-run/items" / sid
            paths = [folder / "analysis.json", folder / "outcome.json"]
            if any(not p.exists() for p in paths):
                missing.append(sid)
                continue
            detail, outcome = [read_json(p) for p in paths]
            if detail.get("status") not in {"done", "failed", "cancelled"}:
                missing.append(sid)
                continue
            exported = {}
            if detail["status"] == "done":
                export_path = folder / "export.json"
                if not export_path.exists():
                    missing.append(sid)
                    continue
                exported = read_json(export_path)
                if detail.get("customer_assessment") != exported.get("customer_assessment"):
                    raise ValueError(f"Analysis/export customer conclusions differ: {sid}")
            if detail.get("sha256") != source["sha256"]:
                raise ValueError(f"Analysed file does not match the planned sample: {sid}")
            original = Path(source["path"])
            if not original.is_absolute():
                original = BACKEND / original
            if digest(original.read_bytes()) != source["sha256"]:
                raise ValueError(f"Original file bytes changed: {sid}")
            category = classify(detail)
            expected_prediction = {"original": "real", "record": "record_led"}.get(category, category)
            if outcome.get("prediction") != expected_prediction:
                raise ValueError(f"Saved outcome differs from the public conclusion: {sid}")
            assessment = detail.get("customer_assessment") or {}
            analysis_id = str(UUID(detail["id"]))
            url = f"http://127.0.0.1:5230/analyses/{analysis_id}?view=plain"
            headline = assessment.get("headline") or ""
            conclusion = headline
            if category == "record":
                conclusion = "Provenance declares AI generation; signer is not trusted"
            elif category == "failed":
                conclusion = "Analysis failed or was cancelled"
            elif not conclusion:
                conclusion = CATEGORIES[category]
            available = category in {"ai", "original"}
            confidence = assessment.get("confidence_pct") if available else None
            interval = assessment.get("interval_pct") if available else None
            doc = document_observations(exported) if source["kind"] == "document" and exported else None
            evidence_note = ""
            evidence_url = f"http://127.0.0.1:5230/analyses/{analysis_id}/{'image-evidence' if source['kind'] == 'image' else 'media-evidence'}?view=plain"
            if doc:
                conflicting = doc["displayed_passage_responses"].get("conflicting", 0)
                evidence_note = f"{doc['displayed_passages']} text passages displayed; {conflicting} contain opposing check responses. All displayed positions match the extracted text."
                if doc["omitted_passages"]:
                    evidence_note += f" Another {doc['omitted_passages']} candidates are omitted from the limited display."
            rows.append({
                "cohort": cohort, "sample_id": sid, "kind": source["kind"], "label": source["label"],
                "generator": source.get("generator") or "—", "category": category,
                "conclusion": conclusion, "customer_headline_en": headline,
                "percentage_available": available, "confidence_pct": confidence,
                "interval_pct": interval, "scope": scope_for(source, assessment),
                "reason": reason_for(detail, category), "evidence_note": evidence_note,
                "local_url": url, "evidence_url": evidence_url, "source_url": source["source_url"],
                "source_group": source["source_group"], "sha256": source["sha256"],
                "content_direction_matches_label": category in {"ai", "original"}
                    and ("ai" if category == "ai" else "real") == source["label"],
            })
            diagnostics.append({
                "sample_id": sid, "analysis_id": analysis_id, "category": category,
                "decision_code_not_used_for_counting": (detail.get("decision") or {}).get("code"),
                "saved_fusion_estimate_not_used_for_counting": (detail.get("decision") or {}).get("probability_ai_calibrated"),
                "customer_assessment": assessment, "condition_profile": detail.get("condition_profile"),
                "legacy_document_fields_not_final_conclusion": doc,
                "input_artifacts_sha256": {p.name: digest(p.read_bytes()) for p in
                                           [*paths, folder / "export.json"] if p.exists()},
            })
    if missing:
        raise ValueError("Final report not written: unfinished planned samples: " + ", ".join(missing))
    if len(rows) != 38:
        raise ValueError("A final report must preserve all 38 planned samples")
    return rows, diagnostics, {"main": digest(main_path.read_bytes()), "challenge": digest(challenge_path.read_bytes())}


def count_rows(rows: list[dict]) -> dict:
    return {name: sum(r["category"] == name for r in rows) for name in CATEGORIES}


def summary_table(rows: list[dict], cohort: str) -> str:
    parts = []
    for kind in KINDS:
        for label in ["ai", "real"]:
            subset = [r for r in rows if r["cohort"] == cohort and r["kind"] == kind and r["label"] == label]
            if not subset:
                continue
            counts = count_rows(subset)
            parts.append(f"<tr><th>{KINDS[kind]}</th><td>{'AI-generated / continued' if label == 'ai' else 'Real / human-written'}</td>"
                         f"<td>{len(subset)}</td>" + "".join(f"<td>{counts[c]}</td>" for c in CATEGORIES) + "</tr>")
    return ('<div class="table-wrap"><table class="summary"><thead><tr><th>Type</th><th>Source label</th><th>Count</th>'
            + "".join(f"<th>{label}</th>" for label in CATEGORIES.values())
            + "</tr></thead><tbody>" + "".join(parts) + "</tbody></table></div>")


def sample_table(rows: list[dict]) -> str:
    result = []
    for row in rows:
        scope = text(row["scope"])
        percentage = f"{row['confidence_pct']:g}% · automated estimate" if row["percentage_available"] else "Percentage unavailable"
        interval = row.get("interval_pct")
        interval_text = ""
        if interval and len(interval) == 2:
            interval_text = f'<small>Recorded 90% interval: {interval[0]:g}–{interval[1]:g}%</small>'
        note = f'<p class="detail">{text(row["evidence_note"])}</p>' if row["evidence_note"] else ""
        headline = f'<details class="source-heading"><summary>Exact application wording</summary><span>{text(row["customer_headline_en"])}</span></details>' if row["customer_headline_en"] and row["customer_headline_en"] != row["conclusion"] else ""
        source_name = "Official publisher example" if row["cohort"] == "challenge" else {"image": "DiTFake", "video": "DeepAction", "document": "RAID"}[row["kind"]]
        label = "AI continuation" if row["label"] == "ai" and row["kind"] == "document" else "AI-generated" if row["label"] == "ai" else "Human-written" if row["kind"] == "document" else "Real"
        result.append(f'''<tr data-cohort="{row['cohort']}" data-kind="{row['kind']}" data-label="{row['label']}" data-category="{row['category']}">
<td><strong>{text(row['sample_id'])}</strong><small>{COHORTS[row['cohort']]} · {KINDS[row['kind']]}</small></td>
<td><span class="label">{label}</span><small>{source_name}</small></td>
<td class="generator">{text(row['generator'])}</td>
<td class="result"><span class="badge {row['category']}">{CATEGORIES[row['category']]}</span><p>{text(row['conclusion'])}</p><strong class="percentage">{percentage}</strong>{interval_text}{headline}</td>
<td>{scope}</td><td class="reason">{text(row['reason'])}{note}</td>
<td class="links"><a href="{safe_url(row['local_url'])}" target="_blank" rel="noopener">Open local analysis ↗</a><a href="{safe_url(row['evidence_url'])}" target="_blank" rel="noopener">View supporting evidence ↗</a><a href="{safe_url(row['source_url'])}" target="_blank" rel="noopener">Original source ↗</a></td></tr>''')
    return "".join(result)


CSS = """
:root{--bg:#F8F7F1;--ink:#183C34;--muted:#52675E;--line:#D7DED5;--accent:#285F50;--tint:#F0F2EB}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 system-ui,-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
header{border-bottom:1px solid var(--line);padding:20px max(28px,calc((100vw - 1520px)/2));display:flex;justify-content:space-between;gap:20px;align-items:center}
.brand{font-size:21px;font-weight:750;letter-spacing:-.5px}.tag{font-size:12px;border:1px solid var(--line);border-radius:20px;padding:4px 12px;color:var(--muted)}
main{max-width:1580px;padding:42px 30px 70px;margin:auto}h1{font-size:38px;line-height:1.2;margin:0 0 20px;letter-spacing:-1px}h2{font-size:23px;margin:0 0 14px}h3{font-size:17px;margin:0 0 10px}p{margin:0 0 12px}a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px}a:focus-visible,input:focus-visible,select:focus-visible,button:focus-visible{outline:3px solid #91b8a1;outline-offset:3px}
.hero{max-width:1050px}.eyebrow{font-size:12px;letter-spacing:1.3px;color:var(--muted);text-transform:uppercase;margin-bottom:12px}.lead{font-size:18px}.muted,small{color:var(--muted)}small{display:block;font-size:12px;line-height:1.55;margin-top:5px}.hero .muted{font-size:14px}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:28px 0}
.stat{border:1px solid var(--line);background:#fffefa;border-radius:12px;padding:20px}.stat strong{font-size:36px;line-height:1.15;display:block;font-variant-numeric:tabular-nums}.stat span{font-size:13px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:30px 0}.callout{background:var(--tint);border:1px solid var(--line);border-radius:12px;padding:22px}.callout p{font-size:14px;margin:0}.callout strong{font-weight:700}.callout.warning{border-left:4px solid #AE4617}
section{margin-top:38px}.section-head{display:flex;justify-content:space-between;gap:20px;align-items:baseline}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px;background:#fffefa}table{border-collapse:collapse;width:100%;text-align:left}th,td{border-bottom:1px solid var(--line);padding:14px 15px;vertical-align:top}thead th{font-size:12px;font-weight:700;background:var(--tint);white-space:nowrap}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}.summary{min-width:800px}.summary td:nth-child(n+3){font-variant-numeric:tabular-nums}.summary tbody th{font-size:14px;font-weight:600}#samples{min-width:1450px;font-size:13px}#samples th{position:sticky;top:0}#samples td:first-child{min-width:165px}#samples .generator{max-width:165px;overflow-wrap:anywhere}#samples .result{min-width:250px;width:23%}#samples .reason{min-width:250px;max-width:320px}#samples td:nth-child(5){min-width:205px;max-width:260px}.result p{margin:8px 0 5px}.percentage{font-size:13px;font-weight:600}.badge,.label{display:inline-block;border-radius:6px;padding:3px 8px;font-size:12px;white-space:nowrap}.badge.ai{background:#F9E4DA;color:#9B2915}.badge.original{background:#E2F0EA;color:#0B6B54}.badge.record{background:#E9E7F1;color:#594B75}.badge.inconclusive{background:#EEEBE4;color:#625C50}.badge.failed{background:#F7DCD7;color:#9B2915}.label{background:var(--tint)}.detail{font-size:12px;color:var(--muted);margin-top:10px}.links{min-width:155px}.links a{display:block;margin-bottom:10px;white-space:nowrap}.source-heading{font-size:11px;color:var(--muted);margin-top:8px}.source-heading summary{cursor:pointer}.source-heading span{display:block;line-height:1.6;padding:5px 0}
.filters{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin:18px 0}.filters label{display:flex;flex-direction:column;font-size:12px;color:var(--muted);gap:5px}.filters select,.filters input,.filters button{font:inherit;color:var(--ink);background:#fffefa;border:1px solid #AAB8AE;border-radius:7px;padding:9px 12px;min-height:42px}.filters input{width:230px}.filters button{cursor:pointer}.filters button:hover{background:var(--tint)}.count-note{font-size:13px;color:var(--muted);margin:10px 0}.empty{display:none;padding:28px;text-align:center}.protocol{max-width:1100px}.protocol p{font-size:14px}.downloads{display:flex;gap:20px;font-size:14px}.hashes{font:11px/1.8 ui-monospace,monospace;overflow-wrap:anywhere;border-top:1px solid var(--line);padding-top:18px;margin-top:28px;color:var(--muted)}.doc-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.doc-note{border:1px solid var(--line);border-radius:10px;padding:22px;background:#fffefa}.doc-note p:last-child{margin-bottom:0}noscript{display:block;padding:12px}
@media(max-width:950px){main{padding:30px 20px}.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,1fr)}.doc-grid{grid-template-columns:1fr}h1{font-size:31px}.section-head{display:block}header{padding:18px 20px}.tag{display:none}}
@media(max-width:560px){.stats{grid-template-columns:repeat(2,1fr)}.filters label{flex:1 0 40%}.filters input{width:100%}.stat{padding:16px}.hero .lead{font-size:16px}}
@media print{header,main{padding:16px}.filters,.source-heading,.downloads{display:none}body{font-size:10px}.table-wrap{overflow:visible;border:0}#samples{min-width:0;font-size:8px}#samples td{min-width:0!important;max-width:none!important;padding:7px}thead{display:table-header-group}.stat strong{font-size:24px}.grid,.doc-grid{display:block}.callout,.doc-note{margin-bottom:12px}.badge{white-space:normal}.links a{white-space:normal}a{color:inherit}.stats{margin:16px 0}.hashes{font-size:8px}}
"""


JS = """
const controls=['cohort','kind','label','category'].map(id=>document.getElementById(id));
const search=document.getElementById('search');
const rows=[...document.querySelectorAll('#samples tbody tr')];
function update(){const term=search.value.trim().toLocaleLowerCase();let visible=0;
for(const row of rows){const match=controls.every(control=>!control.value||row.dataset[control.id]===control.value)&&(!term||row.textContent.toLocaleLowerCase().includes(term));row.hidden=!match;if(match)visible++;}
document.getElementById('visible-count').textContent=`Showing ${visible} / ${rows.length} samples · Filters do not change the fixed totals above`;
document.getElementById('empty').style.display=visible?'none':'block';}
controls.forEach(control=>control.addEventListener('change',update));search.addEventListener('input',update);
document.getElementById('reset').addEventListener('click',()=>{controls.forEach(control=>control.value='');search.value='';update();});update();
"""


def render(rows: list[dict], diagnostics: list[dict], manifest_hashes: dict) -> str:
    main = [r for r in rows if r["cohort"] == "main"]
    challenge = [r for r in rows if r["cohort"] == "challenge"]
    counts = count_rows(main)
    challenge_counts = count_rows(challenge)
    matches = sum(r["content_direction_matches_label"] for r in main)
    doc_diagnostics = [d["legacy_document_fields_not_final_conclusion"] for d in diagnostics
                       if d["legacy_document_fields_not_final_conclusion"] is not None]
    position_count = sum(d["verified_positions"] for d in doc_diagnostics)
    conflicts = sum(d["displayed_passage_responses"].get("conflicting", 0) for d in doc_diagnostics)
    ai_high_disagreement = [d for d in diagnostics if d["sample_id"].startswith("image-ai-")
                           and d["category"] == "inconclusive"
                           and d["saved_fusion_estimate_not_used_for_counting"] == 0.98]
    low_face_scope = [r for r in rows if r["sample_id"].startswith("image-real-")
                      and r["category"] == "inconclusive" and "face" in r["reason"]]
    doc_rows = [r for r in main if r["kind"] == "document"]
    doc_unavailable = sum(r["category"] == "inconclusive" for r in doc_rows)
    stat_html = "".join(f'<div class="stat"><strong>{counts[key]}</strong><span>{label}</span></div>' for key, label in CATEGORIES.items())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chipmunk Verify · Public Sample Evaluation</title><style>{CSS}</style></head><body>
<header><div class="brand">Chipmunk Verify</div><span class="tag">Local evaluation · All 38 samples retained</span></header><main>
<div class="hero"><p class="eyebrow">Local evaluation / Public source samples</p><h1>What the checks can conclude.</h1><p class="lead">Of 36 main samples, {counts['ai'] + counts['original']} received a content-based tendency, {counts['record']} relied on a provenance record, and {counts['inconclusive']} remained inconclusive.</p><p class="muted">Two newer video challenges are reported separately. These are actual customer conclusions and their limitations. The small sample size and limited sources do not establish overall accuracy.</p><p><a href="audio-comparison.html">View the independent audio comparison</a></p></div>
<div class="stats" aria-label="Results for the 36 main samples">{stat_html}</div>
<div class="grid"><article class="callout warning"><h3>Articles: {doc_unavailable} / {len(doc_rows)} inconclusive</h3><p>Text locations, context and check responses are available, but reliable document-level confidence calibration is missing. High check scores cannot replace a supported conclusion.</p></article><article class="callout"><h3>{len(ai_high_disagreement)} AI images: high scores, inconclusive results</h3><p>The saved fusion estimates are all 0.98, but the checks strongly disagree. Customers receive an inconclusive result, so these {len(ai_high_disagreement)} images are not counted as successful detections.</p></article><article class="callout"><h3>{len(low_face_scope)} real images: insufficient validation coverage</h3><p>The image dimensions and small faces lack sufficient validation coverage. A saved estimate favouring original content did not result in a published original-file tendency.</p></article></div>
<section><div class="section-head"><h2>Main sample set · 36 files</h2><span class="muted">16 images · 8 videos · 12 articles</span></div><p class="muted">{matches} published content-based tendencies agree with the source labels. Provenance records and inconclusive results remain in the totals; they are not counted as successful content-based detections.</p>{summary_table(rows, 'main')}</section>
<section><h2>Newer video challenges · 2 files</h2><p class="muted">Official HunyuanVideo 1.5 and LTX-2.5 generated examples: {challenge_counts['ai']} leaning AI and {challenge_counts['inconclusive']} inconclusive. Both are silent and have no paired real-video controls. Their performance is not pooled with the main sample set.</p>{summary_table(rows, 'challenge')}</section>
<section><h2>What the article evidence provides</h2><div class="doc-grid"><div class="doc-note"><h3>Traceable locations that still need review</h3><p>The 12 articles display {position_count} text passages. Every location was checked against the extracted text. {conflicts} passages retain conflicting check responses. Open the corresponding analysis to inspect the text, context and supporting or opposing responses.</p><p class="muted">These are selected candidate passages, not verification of every word. Real news articles also trigger AI-style responses; a highlight does not prove that the author used AI.</p></div><div class="doc-note"><h3>Issue found: conflicting legacy JSON wording</h3><p>The top-level customer conclusion says there is no calibrated estimate, while legacy article fields still contain labels such as “Likely AI-generated,” uncalibrated scores and estimated AI-written shares.</p><p class="muted">This product issue remains to be corrected. This report counts the final customer conclusion; legacy fields are retained only for diagnostics and are not treated as confidence or reliable detections.</p></div></div></section>
<section><div class="section-head"><h2>Inspect every sample · 38 files</h2><div class="downloads"><a href="web-sample-results.csv" download>Download CSV</a></div></div><p class="muted">Automated percentages are not independently measured accuracy rates. Recorded 90% intervals do not cover every detection error; a narrow interval does not guarantee a correct conclusion.</p>
<div class="filters"><label>Sample set<select id="cohort"><option value="">All</option><option value="main">Main sample set</option><option value="challenge">Newer video challenges</option></select></label><label>Type<select id="kind"><option value="">All</option><option value="image">Image</option><option value="video">Video</option><option value="document">Article</option></select></label><label>Source label<select id="label"><option value="">All</option><option value="ai">AI-generated / continued</option><option value="real">Real / human-written</option></select></label><label>Actual result<select id="category"><option value="">All</option>{''.join(f'<option value="{key}">{label}</option>' for key, label in CATEGORIES.items())}</select></label><label>Search<input id="search" type="search" placeholder="Sample ID, generator, reason"></label><button id="reset" type="button">Reset filters</button></div><p class="count-note" id="visible-count" aria-live="polite">Showing 38 / 38 samples</p><noscript>JavaScript is disabled. All 38 samples remain visible below; filters are unavailable.</noscript>
<div class="table-wrap"><table id="samples"><thead><tr><th>Sample</th><th>Source label</th><th>Generator (source information)</th><th>Actual customer conclusion / percentage</th><th>Scope</th><th>Reason / evidence notes</th><th>Analysis and source</th></tr></thead><tbody>{sample_table(rows)}</tbody></table><div class="empty" id="empty">No samples match the current filters.</div></div></section>
<section class="protocol"><h2>How to interpret this evaluation</h2><p>Labels come from public research annotations or explicitly generated examples published by their creators, not from appearance. Images come from DiTFake, main videos from DeepAction, and articles from the labelled training portion of RAID. AI articles include continuations of human-written prompts; not every word was generated from scratch, and word-level ground truth is unavailable. Overlap with detector or base-model training data is unknown.</p><p>Samples were fixed before evaluation. Original bytes and SHA-256 hashes are preserved, and upload filenames contain no truth labels. Real and generated media differ in resolution, encoding and compression, so source characteristics may affect the results. Main videos use earlier generators; the two newer challenges are supplemental examples and do not represent all modern generators.</p><p>Most videos are silent. The only main video with audio has a published percentage limited to visual content. This report does not evaluate speech or background-audio detection; see the <a href="audio-comparison.html">separate audio evaluation</a>.</p><p>Provenance records are counted separately. A file credential declaring AI generation is a clue; an untrusted signature is not independent authenticity certification or successful detection from content alone. Uncalibrated scores, legacy decision codes and legacy article labels do not override the final customer conclusion.</p><p>This report links to original sources and local analyses without reproducing images, videos, article text or complete source files. Samples are used for local evaluation only. Third-party content rights are retained; model licences were not treated as permission to redistribute example media.</p></section>
<footer class="hashes">Generated: {now}<br>Main manifest SHA-256: {manifest_hashes['main']}<br>Challenge manifest SHA-256: {manifest_hashes['challenge']}<br>Data: locally saved analysis.json / export.json / outcome.json. No inference was rerun and no decision rules were changed.</footer>
</main><script>{JS}</script></body></html>'''


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = {"cohort": "Sample set", "sample_id": "Sample ID", "kind": "Type", "label": "Source label",
              "generator": "Generator", "category": "Result category", "conclusion": "Actual customer conclusion",
              "customer_headline_en": "Exact application wording", "percentage_available": "Percentage availability",
              "confidence_pct": "Automated estimate (%)", "interval_pct": "Recorded 90% interval (%)",
              "scope": "Scope", "reason": "Inconclusive reason or provenance note", "evidence_note": "Evidence notes",
              "local_url": "Local analysis URL", "evidence_url": "Supporting evidence URL", "source_url": "Original source URL",
              "source_group": "Source group", "sha256": "Original file SHA-256"}
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields.values()))
        writer.writeheader()
        for row in rows:
            values = dict(row)
            values.update(cohort=COHORTS[row["cohort"]], kind=KINDS[row["kind"]],
                          label="AI-generated / continued" if row["label"] == "ai" else "Real / human-written",
                          category=CATEGORIES[row["category"]], percentage_available="Available (automated estimate)" if row["percentage_available"] else "Unavailable")
            values["interval_pct"] = "–".join(f"{n:g}" for n in row["interval_pct"]) if row.get("interval_pct") else ""
            writer.writerow({label: values.get(key) for key, label in fields.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, diagnostics, manifest_hashes = load_rows(args.input.resolve())
    page = render(rows, diagnostics, manifest_hashes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "web-sample-results.html").write_text(page, encoding="utf-8")
    write_csv(args.output / "web-sample-results.csv", rows)
    summary = {cohort: {"count": sum(r["cohort"] == cohort for r in rows),
                       "outcomes": count_rows([r for r in rows if r["cohort"] == cohort])}
               for cohort in COHORTS}
    (args.output / "web-sample-diagnostics.json").write_text(json.dumps({
        "purpose": "Local diagnostic data; not a source-media or source-text redistribution",
        "manifest_sha256": manifest_hashes, "summary": summary, "items": diagnostics,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(rows), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
