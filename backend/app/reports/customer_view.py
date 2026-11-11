"""Readable customer conclusions derived from frozen evidence, never a change to it.

An inspection score, a check's contribution and the combined estimate answer
different questions. Keep those scopes explicit instead of repeating conflicting
stand-alone and weighted likelihood-ratio prose in the customer finding cards.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from app.engine.customer_assessment import EMAIL_REFERENCE_SLICE, customer_assessment, text_reference_rate
from app.engine.finding_assessment import finding_assessment
from app.engine.service_policy import standard_service_policy
from app.reports import formatting, view
from app.reports.snapshot import ReportSnapshot

RATE_EXPLANATION = (
    "The saved evaluation does not tell us how often this combined result would wrongly flag an original file, "
    "or miss an AI-generated file, under comparable conditions. The displayed estimate is therefore not a "
    "verified accuracy rate for files like this one."
)
TENDENCY_RATE_EXPLANATION = (
    "Reference testing does not establish a universal accuracy rate for these findings. Errors and inconclusive results "
    "remain possible. The combined percentage reflects only eligible checks; inspection-only findings are not averaged into it."
)

FOCUS = {
    "F3_whole_image_generation": "Visual-generation check",
    "F4_localised_manipulation": "Local editing check",
    "F5_compression_history": "Compression patterns",
    "F6_capture_pipeline": "Capture and noise patterns",
    "F7_duplication": "Repeated image regions",
    "F10_face": "Face-manipulation check",
    "F11_physical_scene": "Scene consistency",
    "V1_face_track": "Faces across video frames",
    "V2_lip_sync": "Voice and mouth timing",
    "V3_synthetic_video": "Visual-generation check",
    "V4_codec": "Video container and encoding",
    "A1_voice": "Synthetic-voice check",
    "T1_ai_text": "AI-writing check",
}
CONTEXT_FAMILIES = {"coverage", "F9_distribution_conditions"}
SOURCE_FAMILIES = {"F1_provenance", "F2_self_declared_metadata", "F8_metadata_consistency"}


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _signals(finding: dict, result: dict) -> list[dict]:
    refs = set(finding.get("evidence_ids") or [])
    refs.update(ref for sentence in finding.get("sentences") or [] for ref in sentence.get("evidence_ids") or [])
    return [s for s in result.get("signals") or [] if s.get("id") in refs
            or (s.get("legacy") or {}).get("v1_id") in refs or f"detector:{s.get('detector_id')}" in refs]


def _observation(signal: dict, public_ledger: dict[str, dict], redact: Callable[[str], str]) -> str:
    text = str(signal.get("observed") or "")
    # Explain signed logits through the recorded test boundary, not as negative probabilities.
    whole = re.search(r"(\d+) of (\d+) frame\(s\) at logit > 0", text)
    face = re.search(r"(\d+)/(\d+) face frames > ([\d.]+)", text)
    if whole:
        flagged, count = whole.groups()
        return (f"{flagged} of {count} sampled frames crossed this check's AI flagging boundary. "
                "This tests the sampled visual content; it does not assess every frame or the sound track.")
    if face:
        flagged, count, _ = face.groups()
        return (f"{flagged} of {count} sampled face frames crossed this check's inspection boundary. "
                "A face flag does not identify edited pixels or establish that a face was replaced.")
    row = public_ledger.get(str(signal.get("display_id")), {})
    text = str(row.get("observed") or text)
    # Even old snapshots must use the allowlisted audio observation projection.
    text = re.split(r"; per (?:model|check):", text, maxsplit=1)[0]
    if "mean p(synthetic)=" in text:
        text = text.replace("mean p(synthetic)=", "Raw speech inspection response ")
        text += ". This response is not a calibrated confidence percentage."
    text = text.replace("fused p(synthetic)=", "Combined voice-check response ").replace("p(fake)=", "inspection response ")
    return redact(text) if text else "No separate observation was saved for this check."


def _packet_inference(packet: dict) -> str:
    kind = packet.get("kind")
    if kind in {"av", "acoustic", "sound"}:
        return "No AI/original conclusion from this measurement. It describes timing or recording conditions, which can also occur in ordinary material."
    if kind == "document":
        directions = {w.get("supports_ai") for w in packet.get("windows") or []}
        if directions == {True, False}:
            return "Mixed evidence for this passage: the recorded checks disagree. No reliable AI/original conclusion can be assigned to these words."
        if directions == {True}:
            return "Leaning towards AI-written text in the scored context. This is an inspection clue, not a calibrated probability or proof of authorship for the highlighted words."
        if directions == {False}:
            return "The recorded checks did not flag this scored context as AI-written. Human authorship is not established by a low response."
    if kind == "audio" and "reaches the check's 0.5 inspection threshold" in str(packet.get("interpretation")):
        return "Leaning towards a synthetic-voice pattern in this complete window. This is an uncalibrated inspection clue; it does not establish a replaced voice or an exact edit boundary."
    if kind == "video":
        measures = {m.get("label"): _number(m.get("value")) for m in packet.get("measurements") or []}
        boundary = measures.get("Recorded decision boundary")
        response = next((v for k, v in measures.items() if k in {"Frame detection score", "Window detection score"} and v is not None), None)
        if response is not None and boundary is not None and response >= boundary:
            return "Leaning towards an AI/deepfake pattern in this sampled input. This is an inspection clue, not the probability that these pixels were altered."
    if packet.get("evidence_type") == "anomaly":
        if packet.get("semantics") == "error_level":
            return "This region responds differently to JPEG recompression. Texture, edges and earlier saves can cause the difference."
        if packet.get("semantics") == "residual":
            return "The measured noise varies across this region. Lighting, texture and ordinary processing can cause the variation."
    if packet.get("region"):
        return "Possible editing trace in this region. The recorded pattern warrants inspection, but does not by itself distinguish AI editing from ordinary processing."
    return "Insufficient evidence for an AI/original conclusion at this source location. The recorded response and its limits are shown below."


def _links(finding: dict, packets: list[dict], image_packets: list[dict]) -> list[dict]:
    if finding.get("family") in CONTEXT_FAMILIES:
        return []
    refs = set(finding.get("evidence_ids") or [])
    refs.update(ref for sentence in finding.get("sentences") or [] for ref in sentence.get("evidence_ids") or [])
    out = []
    for packet, prefix in [*((p, "multimedia-evidence") for p in packets), *((p, "image-evidence") for p in image_packets)]:
        direct = finding.get("id") in (packet.get("finding_ids") or [])
        shared = any(ref.get("id") in refs for ref in packet.get("signal_refs") or [])
        if not direct and not shared:
            continue
        region = packet.get("region") or {}
        locator = packet.get("locator") or (f"x {region.get('x')}, y {region.get('y')}; {region.get('w')} x {region.get('h')} pixels" if region else "recorded source location")
        out.append({"id": packet["id"], "href": f"#{prefix}-{packet['id']}", "label": f"{packet['id']} - {locator}",
                    "observation": packet.get("observation") or "", "direct": direct})
    return out


def _finding(finding: dict, row: dict, snapshot: ReportSnapshot, packets: list[dict], image_packets: list[dict],
             ledger: dict[str, dict], redact: Callable[[str], str]) -> dict:
    result = snapshot.result
    signals = _signals(finding, result)
    links = _links(finding, packets, image_packets)
    family = finding.get("family")
    focus = FOCUS.get(family, row.get("title") or "Recorded check")
    check_ids = [s["display_id"] for s in signals if s.get("display_id")]
    if family not in SOURCE_FAMILIES | CONTEXT_FAMILIES and check_ids:
        focus += f" ({', '.join(check_ids)})"
    observations = [_observation(s, ledger, redact) for s in signals if s.get("status", "ok") == "ok"]
    effect = _number(finding.get("contribution_logit"))
    outside = [s.get("outside_estimate") for s in signals if s.get("outside_estimate")]
    if family in SOURCE_FAMILIES:
        inference = "Provenance context. A missing or unsigned origin record does not establish whether content is AI-generated."
        if view.is_documentary(finding):
            inference = "Documentary evidence: use the recorded credential statement and verification outcome below."
        if any(s.get("provenance_override") == "self_declared_ai" for s in signals):
            inference = "Unverified origin marker. It does not authenticate AI generation, change the content estimate or justify skipping content checks."
        else:
            observations = list(row.get("sentences") or [row.get("reason", "")])
    elif family in CONTEXT_FAMILIES:
        inference = "Coverage information only. Checks that did not run, and unmeasured regions, provide no evidence of originality."
        observations = list(row.get("sentences") or [row.get("reason", "")])
    elif family == "A1_voice" and any((s.get("details") or {}).get("audio_screening") for s in signals) and not finding.get("scope"):
        screening = next(s['details']['audio_screening'] for s in signals if (s.get('details') or {}).get('audio_screening'))
        inference = screening['reason'] + ' ' + screening['limitation']
    elif view.outside_validated(result.get("decision") or {}, result.get("condition_profile")):
        inference = "Insufficient validated evidence for an AI/original conclusion from this check on this input."
    elif effect is not None and effect != 0 and not outside:
        target = "AI generation or manipulation" if effect > 0 else "original content"
        if abs(effect) < math.log(2):
            inference = ("No reliable AI/original conclusion from this check alone. "
                         f"Its small contribution moved the combined estimate towards {target}.")
        else:
            size = "limited" if abs(effect) < math.log(10) else "material"
            inference = f"Leaning towards {target}. This check had {size} influence on the combined estimate."
    elif family == "A1_voice" and any("reaches the check's 0.5 inspection threshold" in str(p.get("interpretation"))
                                     for p in packets if p.get("id") in {x["id"] for x in links}):
        inference = "Inspection clues lean towards a synthetic-voice pattern. This check is excluded from the combined estimate, so it does not increase the headline confidence."
    else:
        inference = "Insufficient evidence for a reliable AI/original conclusion from this check alone."
        if outside:
            inference += " It is included as inspection context and does not change the combined estimate."
    assessment = finding_assessment(finding, {**result, "kind": snapshot.analysis.kind})
    if family not in SOURCE_FAMILIES:
        inference = assessment["interpretation"]
    if not observations:
        observations = list(row.get("sentences") or [row.get("reason") or "No scored observation was recorded."])
    if family not in SOURCE_FAMILIES | CONTEXT_FAMILIES and not links:
        scope = "No specific source region or interval is linked to this finding. The check record and any referenced figures provide its recorded basis."
    elif links:
        scope = "The source links below identify the saved inputs and observations. Selected samples do not cover every pixel, frame, word or sound."
    else:
        scope = ""
    return {**row, "assessment": assessment, "title": focus, "inference": inference, "observations": observations, "support": links, "notes": [],
            **({"sentences": [inference, *observations]} if family in SOURCE_FAMILIES and any(s.get("provenance_override") == "self_declared_ai" for s in signals) else {}),
            "scope_note": scope, "status": None, "disposition": None,
            "reason": " ".join([inference, *observations[:1]])}


def customer_view(snapshot: ReportSnapshot, out: dict, redact: Callable[[str], str]) -> dict:
    """Called on a complete, already-redacted presentation; partial projection helpers stay unchanged."""
    if "header_left" not in out:
        return out
    out["customer_report"] = True
    out["public_report_id"] = snapshot.report.id
    out["cover_title"] = f"{snapshot.analysis.kind.capitalize()} authenticity analysis"
    out["title"] = f"Chipmunk Verify {out['cover_title'].lower()} {snapshot.report.id}"
    out["keywords"] = f"Chipmunk Verify, {snapshot.report.profile}, {snapshot.report.id}"
    out["analysis_line"] = f"Analysis {snapshot.analysis.id} · revision {snapshot.analysis.revision} · report {snapshot.report.id}"
    out["examiner"] = None
    out["header_left"] = "Chipmunk Verify | Analysis report"
    out["header_centre"] = snapshot.analysis.kind.capitalize()
    out["status_line"] = "Prepared by Chipmunk Verify | Automated analysis"
    out["draft_note"] = None
    labels = ((snapshot.result.get('manifest') or {}).get('sampling') or {}).get('source_labels')
    out['source_labels'] = labels if (isinstance(labels, dict) and labels.get('version') in ('source-labels/1', 'source-labels/2')
        and labels.get('source_sha256') == snapshot.media.sha256) else None
    # Older frozen document evidence called missing exact inputs "proof".
    # Describe the missing records without changing that original evidence.
    evidence = out.get('evidence') or {}
    for key in ('coverage_details', 'limitations'):
        if key in evidence:
            evidence[key] = [line.replace('exact local input proof', 'exact local input records') for line in evidence[key]]
    policy = standard_service_policy()
    out["service_policy"] = policy
    out["expert_contact"] = policy["expert_contact"]
    out["expert_note"] = policy["private_engagement_notice"] + " Contact " + policy["expert_contact"] + "."
    out["scope_statement"] = policy["court_notice"]
    # The service-scope block already contains the complete engagement notice.
    # Repeating it below findings can create an otherwise empty extra page.
    out["closing"] = ""
    assessment = customer_assessment({**snapshot.result, "kind": snapshot.analysis.kind})
    if (snapshot.result.get("condition_profile") or {}).get("slice_id") == EMAIL_REFERENCE_SLICE:
        # Do not print the internal reference operating point or historically
        # mislabeled transfer counts in a customer report.
        estimate_view = out.get("estimate")
        if isinstance(estimate_view, dict):
            estimate_view["rates"] = []
            decision = snapshot.result.get("decision") or {}
            p = _number(decision.get("probability_ai_calibrated"))
            if assessment["direction"] == "inconclusive" and decision.get("code") == "ai_supported" and p is not None:
                estimate_view["none"] = "The saved check does not support a reliable email authorship tendency."
    text_rate = text_reference_rate(snapshot.analysis.kind, snapshot.result.get("condition_profile"),
                                          snapshot.result.get("calibration") or [])
    out["customer_assessment"] = assessment
    out["customer_reference_test"] = text_rate
    out["customer_historical_note"] = (
        "The reference evaluation applies only to its recorded sample, scope and decision rules."
        if text_rate else
        "Historical test figures in this appendix describe earlier check configurations and test thresholds. They do not establish a universal customer accuracy rate or validate newly added inspection-only checks."
    )
    summary = out.get("summary") or {}
    if assessment["applies"]:
        estimate = (f"{assessment['confidence_label']}: {assessment['confidence_pct']:g}% in this direction."
                    if assessment["confidence_pct"] is not None else "No confidence percentage is assigned to a directional conclusion.")
        out["customer_estimate_first"] = estimate
        out["customer_interval_value"] = None
        if assessment["interval_pct"] is not None:
            lo, hi = assessment["interval_pct"]
            estimate += f" {assessment['interval_label']}: {lo:g}-{hi:g}%."
            out["customer_interval_value"] = f"{lo:g}–{hi:g}%"
        out.update(cover_headline=assessment["headline"], headline=assessment["headline"], customer_estimate=estimate,
                   estimate_line=estimate, means=[assessment["meaning"]], scale=None, scale_note=None)
        summary.update(headline=assessment["headline"], estimate=estimate, means=[assessment["meaning"]])
        out["customer_uncertainty"] = assessment["limitation"]
        if "Fusion model pending human review" in str(summary.get("limitation")):
            summary["limitation"] = "The combined scoring method has not received a documented independent validation review. Its accuracy for comparable files remains uncertain."
        out["customer_interval_note"] = assessment["interval_note"] if assessment["interval_pct"] is not None else ""
    if "cover" in out:
        out["cover"] = [r for r in out["cover"] if r["key"] not in {"MD5", "Review", "Classification", "Acquired (as stated)", "Report"}]
        out["cover"].extend([{"key": "Report", "value": f"{snapshot.report.id} | Generated {formatting.utc(snapshot.report.requested_at)}", "established": True},
                             {"key": "Prepared by", "value": "Chipmunk Verify", "established": True}])
    for row in (out.get("custody") or {}).get("rows") or []:
        row["actor"] = "Chipmunk Verify"
    for row in out.get("revisions") or []:
        row["state"] = "Automated analysis report"
    if isinstance(out.get("custody"), dict):
        out["custody"]["actor_note"] = "This customer log identifies the service that recorded each event. Original account identities and signed event bytes remain in the internal audit record."
    # Empty human-review workflows do not belong in automated customer reports.
    out["sections"] = [s for s in out.get("sections", []) if s.get("key") not in {"appendix-c", "signoff"}]
    if out["sections"] and (out.get("evidence") or out.get("image_evidence")):
        position = next((i + 1 for i, s in enumerate(out["sections"]) if s.get("key") == "findings"), 0)
        out["sections"].insert(position, {"key": "source-evidence", "number": "2A", "title": "Source evidence",
                                         "href": "#image-evidence-heading" if out.get("image_evidence") else "#multimedia-evidence-heading"})
        out.setdefault("number", {})["source-evidence"] = "2A"
    for section in out["sections"]:
        if section.get("key") == "detectors":
            section["title"] = "Technical check records"
        if section.get("key") == "revisions":
            section["number"] = "C"
            if "number" in out:
                out["number"]["revisions"] = "C"
    packets = (out.get("evidence") or {}).get("packets") or []
    image_packets = (out.get("image_evidence") or {}).get("packets") or []
    for packet in [*packets, *image_packets]:
        packet["customer_inference"] = _packet_inference(packet)
    ledger = {str(r.get("id")): r for r in (out.get("detectors") or {}).get("ledger") or []}
    unverified_ids = {str(s.get("display_id")) for s in snapshot.result.get("signals") or []
                      if s.get("provenance_override") == "self_declared_ai" and s.get("display_id")}
    for row in ledger.values():
        row["disposition"] = None
        if str(row.get("id")) in unverified_ids:
            row.update(direction="Context only", strength="Unverified", contribution="Not included", p="-")
        if "per check:" in str(row.get("observed")):
            row["observed"] = str(row["observed"]).split("; per check:", 1)[0].replace("fused p(synthetic)=", "Combined voice-check response ")
    for limitation in out.get("limitations") or []:
        limitation["acknowledged"] = False
    stored = {str(f.get("id")): f for f in snapshot.result.get("findings") or []}
    out["findings"] = [_finding(stored[str(row["id"])], row, snapshot, packets, image_packets, ledger, redact)
                       if str(row.get("id")) in stored else row for row in out.get("findings") or []]
    by_id = {str(f["id"]): f for f in out["findings"]}
    for row in (out.get("provenance") or {}).get("findings") or []:
        finding = by_id.get(str(row.get("id")))
        if finding and "Unverified origin marker" in finding.get("inference", ""):
            row["statement"] = " ".join([finding["inference"], *finding.get("observations", [])])
            row["words"] = "Unverified"
    out["channel_findings"] = []
    if snapshot.analysis.kind == "video":
        soundtrack = assessment.get("soundtrack")
        voice_packet_ids = {p["id"] for p in packets if p.get("kind") == "audio"}
        if soundtrack and any(image.get("anchor") == "multimedia-evidence-voice-waveform" and image.get("uri")
                              for image in (out.get("evidence") or {}).get("overview") or []):
            out["customer_soundtrack_link"] = {"href": "#multimedia-evidence-voice-waveform", "label": "Voice waveform"}
        for label, families in [("Visual content", {"F3_whole_image_generation", "V3_synthetic_video"}),
                                ("Faces", {"V1_face_track", "F10_face"}),
                                ("Voice", {"A1_voice"}), ("Voice / mouth timing", {"V2_lip_sync"})]:
            members = [f for f in stored.values() if f.get("family") in families]
            if not members:
                continue
            if label == "Voice" and soundtrack:
                packet_link = next((link for f in members
                    for link in by_id.get(str(f.get("id")), {}).get("support", []) if link["id"] in voice_packet_ids), None)
                if packet_link:
                    out["customer_soundtrack_link"] = {"href": packet_link["href"], "label": f"Voice interval {packet_link['id']}"}
                continue
            included = [f for f in members if _number(f.get("contribution_logit")) is not None
                        and not any(s.get("outside_estimate") for s in _signals(f, snapshot.result))
                        and (snapshot.result.get("condition_profile") or {}).get("within_validated_envelope") is True]
            total = sum(float(f["contribution_logit"]) for f in included)
            if included and total != 0:
                direction = "AI generation / manipulation" if total > 0 else "original content"
                note = f"Checks in the estimate lean towards {direction}."
                if abs(total) < math.log(2):
                    note = f"No reliable conclusion from these checks alone. They slightly moved the estimate towards {direction}."
            elif label == "Voice" and any("Inspection clues lean" in by_id.get(str(f.get("id")), {}).get("inference", "") for f in members):
                note = "Synthetic-voice inspection clues are present, but this check is excluded from the headline confidence."
            else:
                note = "No reliable AI/original direction established; inspection context only."
            out["channel_findings"].append({"label": label, "text": note, "ids": [f.get("id") for f in members]})
    for row in summary.get("reasons") or []:
        f = by_id.get(str(row.get("id")))
        if f:
            row.update(title=f["title"], text=f["reason"], status=None)
    summary["context"] = "Some checks are inspection-only and do not change the combined estimate. Their findings below say so explicitly." if view.context_findings(list(stored.values()), snapshot.result.get("decision") or {}) else ""
    for packet in packets:
        if snapshot.analysis.kind == "video" and packet.get("kind") == "audio" and any(s.get("outside_estimate") for f in stored.values() if f.get("family") == "A1_voice" for s in _signals(f, snapshot.result)):
            summary["context"] += (" Read the separate speech assessment for the voice evidence."
                                   if assessment.get("soundtrack") else
                                   " In this video, the voice evidence is separate from the visual estimate; a visual leaning towards original content does not clear the voice as original.")
            break
    if assessment["applies"]:
        summary["testing"] = text_rate or TENDENCY_RATE_EXPLANATION
        if isinstance(out.get("testing"), dict):
            out["testing"].update(arrays=[], line=summary["testing"])
    elif "not established" in str(summary.get("testing")):
        summary["testing"] = RATE_EXPLANATION
    if not assessment["applies"] and "not established" in str((out.get("testing") or {}).get("line")):
        out["testing"]["line"] = RATE_EXPLANATION
    if isinstance(out.get("methods"), dict):
        if assessment["applies"]:
            for method in out["methods"].get("method") or []:
                if method.get("key") == "Fusion":
                    method["value"] = str(method.get("value") or "").replace(
                        "conformal bands for the headline.",
                        "uncertainty bands retained from the saved calculation. The customer assessment preserves evidence scope and the recorded uncertainty safeguards.")
        rule = out["methods"].get("headline_rule") or {}
        if assessment["applies"]:
            if (snapshot.result.get("condition_profile") or {}).get("slice_id") == EMAIL_REFERENCE_SLICE:
                rule["sentences"] = ["The selected email body can receive an AI-writing tendency only when its calibrated check and recorded uncertainty safeguards support that interpretation.",
                                     "The point estimate controls this wording. The interval reflects uncertainty in the reference fit."]
            else:
                rule["sentences"] = ["The overall assessment uses applicable, calibrated evidence and the recorded uncertainty safeguards. Missing calibration or unsupported coverage leaves the result inconclusive.",
                                     "Confidence is scoped to the assessed evidence; it is not an average of finding percentages or an uncertainty-interval endpoint. Missing calibration, unsupported inputs or conflicting evidence can prevent a directional conclusion."]
            rule.update(rates=[], scope=None, note=text_rate or TENDENCY_RATE_EXPLANATION)
        elif rule.get("note") and not rule.get("rates"):
            rule["note"] = RATE_EXPLANATION
    if snapshot.analysis.kind == "video":
        effects = [(_number(f.get("contribution_logit")), f) for f in stored.values()
                   if not any(s.get("outside_estimate") for s in _signals(f, snapshot.result))]
        positive = sum(value for value, _ in effects if value is not None and value > 0)
        negative = -sum(value for value, _ in effects if value is not None and value < 0)
        if positive > 0 and negative > 0 and assessment["direction"] in {"original", "ai"}:
            greater = "original content" if negative > positive else "AI generation or manipulation"
            summary["means"].append(f"The checks point in different directions. In the saved calculation, the contributions towards {greater} were larger in total than the opposing contributions. A high raw score from one check does not automatically outweigh the other checks.")
    return out
