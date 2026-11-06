"""Summarize the frozen 80% rule, preserving failures and abstentions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / "data/independent-evaluation/rule80-v1"
OUTPUT = ROOT.parent / "output/evaluation/rule80-fresh"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def metric(k, n):
    return {"count": k, "denominator": n, "rate": k/n if n else None, "wilson95": wilson(k,n)}


def rates(rows):
    real = [r for r in rows if r["label"] == 0]
    ai = [r for r in rows if r["label"] == 1]
    ai_verdicts = [r for r in rows if r["prediction"] == "ai"]
    original_verdicts = [r for r in rows if r["prediction"] == "original"]
    return {
        "samples": len(rows), "genuine": len(real), "ai": len(ai), "source_groups": len({r["group"] for r in rows}),
        "false_alarms": metric(sum(r["prediction"] == "ai" for r in real), len(real)),
        "false_original": metric(sum(r["prediction"] == "original" for r in ai), len(ai)),
        "missed_ai": metric(sum(r["prediction"] != "ai" for r in ai), len(ai)),
        "detected_ai": metric(sum(r["prediction"] == "ai" for r in ai), len(ai)),
        "correct_original": metric(sum(r["prediction"] == "original" for r in real), len(real)),
        "inconclusive": metric(sum(r["prediction"] == "inconclusive" for r in rows), len(rows)),
        "genuine_inconclusive": metric(sum(r["prediction"] == "inconclusive" for r in real), len(real)),
        "ai_inconclusive": metric(sum(r["prediction"] == "inconclusive" for r in ai), len(ai)),
        "failures": metric(sum(r["prediction"] == "failed" for r in rows), len(rows)),
        "provenance_led": metric(sum(r["prediction"] == "record" for r in rows), len(rows)),
        "ai_provenance_declarations": metric(sum(r["prediction"] == "record" and r.get("decision_code") == "ai_supported"
                                                for r in ai), len(ai)),
        "ai_without_any_ai_alert": metric(sum(r["prediction"] != "ai" and not
                                              (r["prediction"] == "record" and r.get("decision_code") == "ai_supported")
                                              for r in ai), len(ai)),
        "ai_verdict_precision": metric(sum(r["label"] == 1 for r in ai_verdicts), len(ai_verdicts)),
        "original_verdict_precision": metric(sum(r["label"] == 0 for r in original_verdicts), len(original_verdicts)),
        "confusion_counts": {str(label): dict(Counter(r["prediction"] for r in rows if r["label"] == label))
                             for label in (0, 1)},
        "assessment_scope_counts": dict(Counter((r.get("assessment") or {}).get("scope_label", "Unavailable") for r in rows)),
    }


def verify_published_result(kind, row):
    """Recompute each headline from the immutable published bytes, not the score log."""
    if row["status"] != "done":
        return
    from app.engine.customer_assessment import customer_assessment

    sha = row["result_sha256"]
    path = TRIAL / f"runs-{kind}/objects/sha256" / sha[:2] / sha[2:4] / sha
    if digest(path) != sha:
        raise ValueError("Published result bytes do not match the retained hash")
    result = json.loads(path.read_bytes())
    assessment = customer_assessment({**result, "status": "done", "kind": kind})
    if assessment != row["assessment"] or result["decision"]["code"] != row["decision_code"]:
        raise ValueError("Published result does not reproduce the recorded customer assessment")
    direction = assessment["direction"]
    if row["prediction"] != (direction if direction in {"ai", "original", "record"} else "inconclusive"):
        raise ValueError("Outcome prediction differs from the published headline")


def audit_freeze(lock):
    from scripts.run_fresh_rule_validation import source_hashes

    if source_hashes() != lock["source_sha256"]:
        raise ValueError("Production scoring or reporting code changed during validation")
    for name, field in (("run_fresh_rule_validation.py", "runner_sha256"),
                        ("prepare_fresh_rule_validation.py", "preparation_sha256")):
        if digest(Path(__file__).with_name(name)) != lock[field]:
            raise ValueError("Frozen experiment implementation changed during validation")
    files = {}
    for model in lock["models"].values():
        if model["license"]["class_"] != "commercial":
            raise ValueError("Checkpoint lacks declared commercial permission")
        for name, sha in model["files"].items():
            if sha is None or digest(Path(name)) != sha:
                raise ValueError("Installed checkpoint bytes differ from the frozen rule")
            files[name] = sha
    excluded = json.loads((TRIAL/"exclusions.json").read_text())
    for prior in excluded["manifests"]:
        if digest(ROOT/prior["path"]) != prior["sha256"]:
            raise ValueError("A prior exclusion manifest changed after sample selection")
    prior_hashes = set(excluded["sha256"])
    for kind in ("image", "video", "audio", "document"):
        selected = json.loads((TRIAL/f"{kind}-manifest.json").read_text())["items"]
        if any(r.get(field) in prior_hashes for r in selected for field in ("sha256", "source_sha256")):
            raise ValueError("Selected input or raw audio bytes repeat a prior manifest hash")
    return {"production_source_files": len(lock["source_sha256"]),
            "commercial_model_entries": len(lock["models"]), "verified_checkpoint_files": len(files),
            "declared_model_licenses": dict(Counter(m["license"]["spdx"] for m in lock["models"].values())),
            "prior_exclusion_manifests_verified": len(excluded["manifests"]),
            "exclusion_receipt_sha256": digest(TRIAL/"exclusions.json"), "prior_input_byte_overlap": 0,
            "published_result_bytes_replayed": True, "threshold_or_calibration_changes": False}


def read_complete(kind):
    manifest = TRIAL / f"{kind}-manifest.json"
    receipt = TRIAL / f"{kind}-receipt.json"
    output = TRIAL / f"{kind}-outcomes.jsonl"
    expected = json.loads(manifest.read_text())["items"]
    if json.loads(manifest.read_text())["plan_sha256"] != digest(TRIAL/f"{kind}-plan.json"):
        raise ValueError("Selected source plan changed after the manifest was frozen")
    actual = [json.loads(line) for line in output.read_text().splitlines()]
    ref = json.loads(receipt.read_text())
    if (ref["rule_sha256"] != digest(TRIAL / "frozen-rule.json") or ref["manifest_sha256"] != digest(manifest)
            or [r["id"] for r in expected] != [r["id"] for r in actual]):
        raise ValueError("Incomplete or mismatched frozen evaluation receipt")
    for source, row in zip(expected, actual, strict=True):
        # The runner's ``seconds`` is wall-clock runtime; the audio manifest's
        # identically named field is the source duration, retained separately here.
        if any(row.get(k) != v for k,v in source.items() if k not in {"status", "seconds"}):
            raise ValueError("Outcome differs from the selected source metadata")
        if row["rule_sha256"] != ref["rule_sha256"] or row["manifest_sha256"] != ref["manifest_sha256"]:
            raise ValueError("Outcome receipt differs from the frozen rule or sample manifest")
        if source["status"] == "ready" and digest(TRIAL/source["path"]) != source["sha256"]:
            raise ValueError("Input bytes no longer match the selected sample")
        verify_published_result(kind, row)
        if "seconds" in source:
            row["source_duration_seconds"] = source["seconds"]
        if row["prediction"] in {"ai", "original"}:
            a = row["assessment"]
            if a["threshold_pct"] != 80 or a["confidence_pct"] < 80 or a["direction"] != row["prediction"]:
                raise ValueError("Tendency does not meet the frozen public rule")
    return actual, {"manifest": digest(manifest), "receipt": digest(receipt), "outcomes": digest(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows, receipts = {}, {}
    for kind in ("image", "video", "audio", "document"):
        rows[kind], receipts[kind] = read_complete(kind)
    lock = json.loads((TRIAL/"frozen-rule.json").read_text())
    summary = {"rule": lock["customer_rule"], "threshold": lock["customer_threshold"],
               "frozen_rule_sha256": digest(TRIAL/"frozen-rule.json"), "receipts": receipts,
               "audit": audit_freeze(lock), "media": {}}
    display = []
    for kind, outcomes in rows.items():
        if kind == "audio":
            conditions = {c: rates([r for r in outcomes if r["condition"] == c])
                          for c in ("clean", "noise_0", "music_5")}
            for condition, result in conditions.items():
                selected = [r for r in outcomes if r["condition"] == condition]
                result["by_family"] = {family: rates([r for r in selected if r["family"] == family])
                                       for family in sorted({r["family"] for r in selected})}
                result["unsupported_duration_with_tendency"] = sum(
                    r["source_duration_seconds"] < 2.5 and r["prediction"] in {"ai", "original"} for r in selected)
            summary["media"][kind] = {"conditions": conditions, "total_variants": len(outcomes),
                                     "unique_source_recordings": len({r["source_group"] for r in outcomes}),
                                     "note": "Three variants of each source. Voices overlap earlier calibration/evaluation; generator and prompt limits are in the frozen plan."}
            for condition, result in conditions.items():
                label = {"clean": "Audio · clean", "noise_0": "Audio · strong noise (0 dB)",
                         "music_5": "Audio · music (5 dB)"}[condition]
                display.append((label, result))
        else:
            result = rates(outcomes)
            result["by_family"] = {family: rates([r for r in outcomes if r["family"] == family])
                                   for family in sorted({r["family"] for r in outcomes})}
            result["check_status_counts"] = dict(Counter(c["status"] for r in outcomes for c in r.get("checks", [])))
            if kind == "image":
                result["cohorts"] = {"generation_sources": rates([r for r in outcomes if not r["id"].startswith("tgif-")]),
                                     "local_edit_sources": rates([r for r in outcomes if r["id"].startswith("tgif-")])}
            elif kind == "video":
                result["cohorts"] = {"generation_sources": rates([r for r in outcomes if r["group"].startswith("DeepAction:")]),
                                     "face_swap_sources": rates([r for r in outcomes if r["group"].startswith("RTFS:")])}
                zero_weight = []
                for row in outcomes:
                    if row["label"] != 1 or not row["group"].startswith("RTFS:") or row["status"] != "done":
                        continue
                    sha = row["result_sha256"]
                    published = json.loads((TRIAL/f"runs-video/objects/sha256/{sha[:2]}/{sha[2:4]}/{sha}").read_bytes())
                    if any(s.get("detector_id") == "vid.frames.dfdc_b7" and s.get("status") == "ok"
                           and s.get("effective_weight") == 0 for s in published.get("signals", [])):
                        zero_weight.append({"id": row["id"], "prediction": row["prediction"],
                                            "confidence_pct": row["assessment"]["confidence_pct"]})
                result["face_check_zero_weight_on_swaps"] = zero_weight
            summary["media"][kind] = result
            display.append(({"document": "Articles", "video": "Videos · visual content", "image": "Images"}[kind], result))
    summary["limitations"] = [
        "Freshness means excluded local source files, source groups and article topics. Training overlap with detection weights and backbones is unverified.",
        "Audio speech identities overlap prior studies. Twelve synthetic clips use synthesis families absent from the calibration.",
        "Audio inputs use the first four seconds or the shorter native duration, decoded and re-encoded to mono 16 kHz PCM. Eight source recordings are below the calibration's 2.5-second minimum. Long conversations were not evaluated.",
        "Audio variants share sources and a fixed background pool. Per-condition rates avoid treating repeated variants as independent originals.",
        "Every video headline concerns visual content only. No video supplied usable speech; lip-sync checks completed on just two clips. This does not validate general lip-sync detection or combined video/speech confidence.",
        "Native image/video formats and resolutions differ across publishers and classes. These results can include source/codec shortcuts.",
        "Publisher labels are retained; genuine source provenance has not been independently authenticated.",
        "Inconclusive results remain in the total sample counts. Missed AI includes AI called original, AI left inconclusive, provenance-led outcomes, and technical failures.",
        "Wilson intervals are descriptive case-level intervals assuming independent outcomes; they do not adjust for matched pairs, shared sources or shared audio backgrounds.",
        "The observed rates describe these selected sources, not the proportion of future customer alerts that will be correct."
    ]
    audio_sources = json.loads((TRIAL/"audio-manifest.json").read_text())["items"]
    summary["media"]["audio"]["source_duration_seconds"] = {
        "minimum": min(r["seconds"] for r in audio_sources), "maximum": max(r["seconds"] for r in audio_sources),
        "recordings_below_2_5_seconds": len({r["source_group"] for r in audio_sources if r["seconds"] < 2.5}),
    }
    summary["sources"] = {
        "image": ["https://huggingface.co/datasets/Jouesmak/DiTFake", "https://github.com/IDLabMedia/tgif-dataset"],
        "video": ["https://huggingface.co/datasets/faridlab/deepaction_v1", "https://huggingface.co/datasets/kontur-ai/rtfs-10k"],
        "audio": ["https://github.com/isjwdu/DFADD"],
        "document": ["https://huggingface.co/datasets/iitolstykh/LLMTrace_classification",
                     "https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Text_Dataset"],
    }
    setup = TRIAL / "setup-no-model-video-outcomes.jsonl"
    summary["setup_exclusions"] = {
        "attempts_without_bound_model_manager": len(setup.read_text().splitlines()) if setup.exists() else 0,
        "reason": "Runner setup error before any learned-model inference; same selected inputs rerun after binding the installed model manager.",
        "scoring_configuration_unchanged": all(
            lock[k] == json.loads((TRIAL/"setup-no-model-frozen-rule.json").read_text())[k]
            for k in ("source_sha256", "models", "settings", "customer_rule", "customer_threshold", "policy")),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (args.output/"outcomes.csv").open("w",newline="") as out:
        fields = ["id","kind","label","family","group","source_group","condition","status","prediction","confidence_pct",
                  "scope_label","headline","source_url","source_duration_seconds","seconds","result_sha256"]
        writer = csv.DictWriter(out,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, **{k: (r.get("assessment") or {}).get(k) for k in
                                 ("confidence_pct", "scope_label", "headline")}}
                         for group in rows.values() for r in group)
    headers = ["Media","Samples","False alarms on genuine","AI called original","AI without AI tendency","Inconclusive","Technical failures"]
    def rate(value):
        return f"{value['count']}/{value['denominator']} ({100*value['rate']:.1f}%)" if value["denominator"] else "—"
    table_rows = [[label,str(result["samples"]),*(rate(result[k]) for k in
                  ("false_alarms","false_original","missed_ai","inconclusive","failures"))] for label,result in display]
    escape = html.escape
    table = "<table><thead><tr>"+"".join("<th>"+escape(x)+"</th>" for x in headers)+"</tr></thead><tbody>"
    table += "".join("<tr>"+"".join("<td>"+escape(x)+"</td>" for x in row)+"</tr>" for row in table_rows)+"</tbody></table>"
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Chipmunk Verify — Frozen 80% rule evaluation</title>
<style>body{font:16px system-ui,sans-serif;max-width:1250px;margin:45px auto;padding:0 24px;color:#213b34;background:#f8faf8}
h1{font-size:28px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:13px;text-align:left;border-bottom:1px solid #ccd8d0}
th{background:#e6efe9}p,li{line-height:1.6}small{overflow-wrap:anywhere}</style>
<h1>Chipmunk Verify · Frozen 80% rule evaluation</h1><p>The current local production pipelines and customer headline rule were run without fitting or threshold changes.
All enabled checkpoints passed the commercial license class and installed file verification. The detailed license and code freeze stays in the local trial receipt.</p>
<p>80% is the frozen estimate threshold. The rule also retains its current coverage and inconclusive-result safeguards.
The actual observed correctness and abstention rates are measured below. AI inputs include fully generated content and AI edits.</p>
<p><strong>Finding:</strong> This test does not support broad original-content claims for localized edits or face swaps.
Six AI-edited images and five face swaps were wrongly called original. Articles remained entirely inconclusive.
Audio issued no wrong-original verdicts, with one genuine recording falsely flagged under strong noise and another under music.</p>"""
    document += table+"<p>False alarms use all genuine inputs as the denominator. AI called original uses all AI inputs. AI without an AI tendency includes inconclusive results. Inconclusive and technical failure rates use all inputs in each row.</p>"
    document += "<p>At 0 dB the added noise and source speech have equal average power. At 5 dB the speech is louder than the added music. Each audio row uses the same 48 source recordings.</p>"
    document += "<h2>Observed correctness of issued tendencies</h2><p>These proportions use only the tendencies issued in this balanced test set. Small counts and the artificial 50% AI prevalence limit what they establish about calibration or future customer alerts.</p>"
    def precision(value):
        if not value["denominator"]:
            return "No tendencies issued"
        lo, hi = value["wilson95"]
        return rate(value)+f"; 95% interval {100*lo:.1f}–{100*hi:.1f}%"
    document += "<table><thead><tr><th>Media</th><th>Correct AI tendencies</th><th>Correct original tendencies</th></tr></thead><tbody>"
    document += "".join("<tr><td>"+escape(label)+"</td><td>"+escape(precision(result["ai_verdict_precision"]))+
                        "</td><td>"+escape(precision(result["original_verdict_precision"]))+"</td></tr>"
                        for label,result in display)+"</tbody></table>"
    provenance = [(kind, sum(r["label"] == 1 and r["prediction"] == "record" and r.get("decision_code") == "ai_supported"
                             for r in outcomes)) for kind,outcomes in rows.items()]
    if any(count for _,count in provenance):
        document += "<p>AI-origin declarations in file credentials: "+", ".join(escape(kind)+" "+str(count)
                    for kind,count in provenance if count)+". These received provenance-led AI conclusions and are reported separately from statistical AI tendencies. They remain in the frozen ‘AI without an AI tendency’ metric, so that metric describes absence of a statistical flag rather than absence of every service alert.</p>"
        document += "<p>After counting those declarations as service alerts, AI inputs receiving no AI alert at all: "+", ".join(
            escape(kind)+" "+rate(summary["media"][kind]["ai_without_any_ai_alert"])
            for kind,count in provenance if count)+". This supplementary service-level count preserves the original frozen rule-only metric above.</p>"
    errors = []
    for kind,outcomes in rows.items():
        selected = [r for r in outcomes if (r["label"] == 0 and r["prediction"] == "ai")
                    or (r["label"] == 1 and r["prediction"] == "original")]
        errors.extend((kind,r) for r in selected[:3])
    if errors:
        document += "<h2>Examples of wrong tendencies</h2><p>Up to three per media type. Every case, including inconclusive outcomes, is retained in the CSV.</p><table><thead><tr><th>Media</th><th>Case</th><th>Ground truth</th><th>Issued tendency</th><th>Reported confidence</th></tr></thead><tbody>"
        for kind,row in errors:
            values = [kind,row["id"],"Genuine" if row["label"] == 0 else "AI-generated or edited",
                      row["prediction"],str(row["assessment"]["confidence_pct"])+"%"]
            document += "<tr>"+"".join("<td>"+escape(x)+"</td>" for x in values)+"</tr>"
        document += "</tbody></table>"
    face = summary["media"]["video"]["cohorts"]["face_swap_sources"]
    zero_weight = summary["media"]["video"]["face_check_zero_weight_on_swaps"]
    document += "<p>Face-swap cohort: "+str(face["detected_ai"]["count"])+"/"+str(face["ai"])+" swaps received a statistical AI tendency; "+str(face["false_original"]["count"])+" were called original. The face-specific check completed with zero effective weight in the combined estimate on "+str(len(zero_weight))+" swap clips, including "+str(sum(r["prediction"] == "original" for r in zero_weight))+" of the wrong-original cases. This is an observed integration/coverage issue requiring further investigation; an uncalibrated inspection score alone does not resolve the verdict.</p>"
    edits = summary["media"]["image"]["cohorts"]["local_edit_sources"]
    document += "<p>Localized image edits: "+str(edits["detected_ai"]["count"])+"/"+str(edits["ai"])+" received a statistical AI tendency; "+str(edits["false_original"]["count"])+" were called original, "+str(edits["ai_inconclusive"]["count"])+" were inconclusive and "+str(edits["ai_provenance_declarations"]["count"])+" were identified through AI declarations in file credentials. Fully generated images received 9/12 statistical AI tendencies.</p>"
    document += "<ul>"+"".join("<li>"+escape(note)+"</li>" for note in summary["limitations"])+"</ul>"
    document += "<p>"+str(summary["setup_exclusions"]["attempts_without_bound_model_manager"])+" setup attempts ran without a bound model manager and were excluded. The same selected inputs were rerun with the installed models loaded; scoring code, weights and thresholds remained unchanged.</p>"
    document += "<p>Dataset sources: "+"; ".join(
        escape(kind)+" ("+", ".join('<a href="'+escape(url)+'">'+escape(url.rsplit('/',1)[-1])+"</a>" for url in urls)+")"
        for kind,urls in summary["sources"].items())+".</p>"
    document += '<p><a href="outcomes.csv">Download all case outcomes (CSV)</a> · <a href="summary.json">Download counts, intervals and audit (JSON)</a></p><small>Frozen rule SHA-256 '+summary["frozen_rule_sha256"]+"</small></html>"
    (args.output/"results.html").write_text(document)
    for row in table_rows:
        print(" | ".join(row))


if __name__ == "__main__":
    main()
