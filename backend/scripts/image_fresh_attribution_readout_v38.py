"""Explicit successor readout for v37's already-normalized creator-key schema."""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import psutil

from scripts import image_fresh_attribution_v37 as collection
from scripts import local_probe_limits_v24 as limits
from scripts.data_resource_guard_v21 import sample
from scripts.image_source_groups_v22 import author_key
from scripts.image_source_rights_v19 import require, sha, write


def known_creator_keys(sources):
    """v22 source_record stores author_key(profile), not the original URL."""
    keys, unknown = set(), 0
    for source in sources:
        value = source["author"]
        if value is None:
            unknown += 1
            continue
        require(isinstance(value, str) and bool(value) and not any(c.isspace() for c in value), "Explicit normalized creator key or None required")
        require(author_key(f"https://www.flickr.com/photos/{value}/") == value, "Prior creator key must already be canonical and casefolded")
        keys.add(value)
    return keys, unknown


def creator_counts(records, sources):
    previous, unknown = known_creator_keys(sources)
    matched = [row for row in records if row["status"] == "matched_current_provider_metadata"]
    linked, unlinked = [], []
    for row in matched:
        key = author_key(row["attribution"]["author_url"])
        (linked if key in previous else unlinked).append(key)
    return {"matched_linked_to_known_creator_keys": len(linked), "matched_without_known_creator_link": len(unlinked), "distinct_unlinked_creator_keys": len(set(unlinked)), "prior_sources_with_unknown_creator": unknown}


def report(folder):
    protocol = collection.admission(folder)
    failed_path = folder / "summary-profile/profile.json"
    failed = json.loads(failed_path.read_text())
    require(failed["command"] == "summarize" and failed["arguments"] == [str(folder)] and failed["child_exit_code"] == 1, "Retained failed v37 summary required for this explicit successor")
    records, profiles = [], []
    for number in range(8):
        profile_path, profile = collection.verified_profile(folder, number)
        rows = collection.review_phase(folder, protocol, number)
        completed = json.loads((folder / f"phase-{number:03d}/completed.json").read_text())
        require(completed["journal_sha256"] == sha(folder / f"phase-{number:03d}/responses.jsonl") and completed["recorded"] == len(rows) == collection.CHUNK, "Complete original metadata journals required")
        records.extend(rows)
        profiles.append({"phase": number, "profile_sha256": sha(profile_path), "sampled_tree_rss_lower_bound_bytes": profile["sampled_tree_rss_lower_bound_bytes"], "duration_s": profile["duration_s"]})
    require(len(records) == collection.COUNT, "Complete256-source denominator required")
    graph = json.loads(collection.GRAPH.read_text())
    matched = [row for row in records if row["status"] == "matched_current_provider_metadata"]
    result = {
        "schema": "chipmunk.fresh-image-attribution-readout/38", "planned": collection.COUNT, "recorded": len(records),
        "statuses": dict(Counter(row["status"] for row in records)),
        "http_statuses": dict(Counter(str(row.get("http_status", "network_error")) for row in records)),
        "unresolved_reasons": dict(Counter(row["error"]["message"] if row["error"]["type"] == "ValueError" else row["error"]["type"] for row in records if row["status"] == "unresolved")),
        "matched_photo_licences": dict(Counter(str(row["attribution"]["license_id"]) for row in matched)),
        **creator_counts(records, graph["sources"]),
        "private_plan_sha256": sha(folder / "plan.json"), "phase_profiles": profiles,
        "retained_failed_v37_summary_profile_sha256": sha(failed_path), "readout_program_sha256": sha(__file__),
        "correction": "Read stored v22/v25/v35 author keys directly after canonical-key validation; apply author_key only to newly received profile URLs. No repeated requests, changed selections or response edits. Original v37 summary remains failed.",
        "images_downloaded": 0, "neural_calls": 0, "roles_assigned": False, "evaluation_admitted": False,
        "limitations": "Current photo/server/secret/owner/CC-BY or BY-SA2.0 correspondence only. No source pixels, cross-pool exact/near duplicate grouping, creator-alias resolution, pretraining independence, camera authenticity or complete generation rights established. No external person contacted. Unresolved requests retained without replacement; no accuracy result.",
    }
    write(folder / "summary-v38.json", result)
    print(json.dumps(result), flush=True)


def monitor(folder):
    output = folder / "summary-v38-profile"
    output.mkdir(exist_ok=False)
    started, child, code, guard = time.monotonic(), None, None, None
    observations, errors, transitions = [], [], []
    with (output / "worker.log").open("x") as log, (output / "resource-samples.jsonl").open("x") as journal:
        def observe(rss, initial=False):
            row = limits.observe(output, started, rss)
            row["violations"] = limits.violations(row, initial=initial)
            observations.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            return row["violations"]

        try:
            if observe(0, initial=True):
                guard = "initial_headroom_unavailable"
            else:
                child = subprocess.Popen([sys.executable, "-m", "scripts.image_fresh_attribution_readout_v38", "report", str(folder)], stdout=log, stderr=subprocess.STDOUT)
                try:
                    process = psutil.Process(child.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    code = child.wait(timeout=0.25)
                    transitions.append({"classification": "verified_terminal_before_first_sample", "exit_code": code})
                    process = None
                while process is not None and child.poll() is None:
                    try:
                        rss, event = sample(process, child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
                        code = child.wait(timeout=0.25)
                        transitions.append({"classification": "authoritative_terminal_after_missing_process", "exit_code": code, "error_type": type(error).__name__})
                        break
                    if observe(rss):
                        guard = "recorded_resource_limit"
                        break
                    if event:
                        transitions.append(event)
                        break
                    time.sleep(0.25)
        except BaseException as error:
            errors.append({"type": type(error).__name__, "detail": str(error)})
            guard = "monitor_or_unresolved_sampling_error"
        finally:
            if child is not None:
                if child.poll() is None:
                    child.terminate()
                try:
                    code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
                if observe(0) and guard is None:
                    guard = "final_resource_limit"
    result = {
        "schema": "chipmunk.fresh-attribution-readout-resource-profile/38", "arguments": [str(folder)],
        "child_exit_code": code, "guard": guard, "sample_errors": len(errors), "errors": errors,
        "terminal_observations": transitions, "duration_s": time.monotonic() - started,
        "sampled_tree_rss_lower_bound_bytes": max(row["tree_rss_bytes"] for row in observations),
        "minimum_observed_available_ram_bytes": min(row["ram_available_bytes"] for row in observations),
        "minimum_observed_free_disk_bytes": min(row["disk_free_bytes"] for row in observations),
        "resource_limit_observations": [row for row in observations if row["violations"]],
        "resource_journal_sha256": sha(output / "resource-samples.jsonl"), "worker_sha256": sha(__file__),
        "limits": {"seconds": limits.SECONDS, "sampled_tree_rss_bytes": limits.RSS_LIMIT, "ram_disk_reserve_bytes": limits.RESERVE, "initial_ram_headroom_bytes": limits.START_RAM, "initial_disk_headroom_bytes": limits.START_DISK},
        "scope": "One explicit post-collection readout successor. No network, media, model, repeat or overwrite. Failed predecessor retained. RSS is sampled, not a true peak guarantee.",
    }
    write(output / "profile.json", result)
    print(json.dumps(result), flush=True)
    if code != 0 or guard or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "monitor"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    globals()[args.command](args.folder.resolve())
