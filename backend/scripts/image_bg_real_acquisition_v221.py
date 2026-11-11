"""Acquire and independently verify only the frozen BG-REAL PNG pairs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import re
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

from scripts import image_bg_real_roster_v218 as roster
from scripts.image_source_groups_v22 import near, sketch

ROOT, DATA, PUBLIC = roster.ROOT, roster.DATA, roster.PUBLIC
PUBLIC_ROSTER = PUBLIC / "image-bg-real-metadata-roster-v218.json"
PUBLIC_SHA = "4a3665f23821adcec8f48eece8f4a8b36f34cd8d066c3b5183cc02f0222a350d"
PAIRS = DATA / "image-positive-pairs-v21-cache/qualification/pairs.jsonl"
ASSETS = ("original", "edited", "semantic_mask")
MAX_BYTES, MAX_REDIRECTS, TOTAL_SECONDS = 4_000_000, 3, 900
HUB = "huggingface.co"
CDN = {
    "cas-bridge.xethub.hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
    "cdn-lfs-eu-1.huggingface.co",
}
REDIRECT = {301, 302, 303, 307, 308}
HEX = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
CODE = (
    "backend/scripts/image_bg_real_acquisition_v221.py",
    "backend/tests/test_image_bg_real_acquisition_v221.py",
    "docs/image-bg-real-acquisition-v221-method.md",
    "backend/scripts/image_bg_real_roster_v218.py",
    "backend/scripts/image_source_groups_v22.py",
)
read, require, sha, write = roster.read, roster.require, roster.sha, roster.write


class Rejected(ValueError):
    """An anticipated individual asset or pair failure with a fixed public reason."""


def check(condition, reason):
    if not condition:
        raise Rejected(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def commitment(headers, *, allow_etag=False):
    # Linked ETag describes the actual asset rather than a resolver response.
    linked = headers.get("X-Linked-Etag")
    value = (linked or (headers.get("ETag") if allow_etag else None) or "").strip('"')
    if linked and HEX.fullmatch(value) is None:
        raise Rejected("unrecognized_publisher_asset_commitment")
    return value if HEX.fullmatch(value) else None


def verify_asset_bytes(body, expected_sha, linked):
    actual = digest(body)
    check(expected_sha is None or actual == expected_sha, "publisher_image_sha256_mismatch")
    if linked is not None:
        if len(linked) == 40:
            # Git uses the object header as part of the blob digest.
            actual_linked = hashlib.sha1(
                f"blob {len(body)}\0".encode() + body, usedforsecurity=False
            ).hexdigest()
        else:
            actual_linked = actual
        check(actual_linked == linked, "publisher_asset_commitment_mismatch")
    return actual


def download_url(path):
    return f"https://{HUB}/datasets/{roster.REPO}/resolve/{roster.REVISION}/{path}"


def redirect_to(current, location, asset):
    check(bool(location), "missing_asset_redirect")
    try:
        url = urljoin(current, location)
        target = urlsplit(url)
        port = target.port
    except ValueError:
        raise Rejected("untrusted_asset_redirect") from None
    check(
        target.scheme == "https"
        and target.username is None
        and target.password is None
        and port in (None, 443)
        and not target.fragment,
        "untrusted_asset_redirect",
    )
    if target.hostname == HUB:
        expected = f"/api/resolve-cache/datasets/{roster.REPO}/{roster.REVISION}/{asset}"
        check(unquote(target.path) == expected, "redirect_changed_pinned_asset")
    else:
        check(target.hostname in CDN, "untrusted_asset_redirect")
    return url


def download(session, asset, expected_sha, deadline):
    url, linked, length = download_url(asset), None, None
    for hop in range(MAX_REDIRECTS + 1):
        check(time.monotonic() < deadline, "acquisition_time_budget_exhausted")
        with session.get(
            url,
            stream=True,
            allow_redirects=False,
            timeout=(5, 10),
            headers={"Accept-Encoding": "identity", "User-Agent": "ChipmunkVerify-Research/1.0"},
        ) as response:
            status = response.status_code
            check(status == 200 or status in REDIRECT, f"asset_http_{status}")
            if hop == 0:
                check(
                    response.headers.get("X-Repo-Commit") == roster.REVISION,
                    "asset_resolver_revision_mismatch",
                )
                linked = commitment(response.headers, allow_etag=status == 200)
                length = response.headers.get("X-Linked-Size")
                check(
                    length is None or (length.isdecimal() and 0 < int(length) <= MAX_BYTES),
                    "asset_size_limit",
                )
            if status in REDIRECT:
                check(hop < MAX_REDIRECTS, "asset_redirect_limit")
                url = redirect_to(url, response.headers.get("Location"), asset)
                continue
            if linked is None and urlsplit(url).hostname == HUB:
                linked = commitment(response.headers, allow_etag=True)
            check(
                response.headers.get("Content-Encoding", "identity") == "identity",
                "unexpected_asset_encoding",
            )
            declared = response.headers.get("Content-Length")
            check(
                declared is None or (declared.isdecimal() and 0 < int(declared) <= MAX_BYTES),
                "asset_size_limit",
            )
            body = bytearray()
            for chunk in response.iter_content(65_536):
                check(time.monotonic() < deadline, "acquisition_time_budget_exhausted")
                check(len(body) + len(chunk) <= MAX_BYTES, "asset_size_limit")
                body.extend(chunk)
            check(
                body and all(n is None or len(body) == int(n) for n in (length, declared)),
                "asset_body_extent",
            )
            actual = verify_asset_bytes(body, expected_sha, linked)
            return bytes(body), {
                "sha256": actual,
                "bytes": len(body),
                "resolver_revision": roster.REVISION,
                "publisher_asset_commitment": linked,
                "final_host": urlsplit(url).hostname,
            }
    raise Rejected("asset_redirect_limit")


def png_pixels(body, size):
    try:
        with Image.open(io.BytesIO(body)) as source:
            check(
                source.format == "PNG" and source.size == tuple(size),
                "png_geometry_or_orientation_mismatch",
            )
            source.verify()
        with Image.open(io.BytesIO(body)) as source:
            check(
                getattr(source, "n_frames", 1) == 1 and source.getexif().get(274, 1) == 1,
                "png_geometry_or_orientation_mismatch",
            )
            check(source.mode in {"1", "L", "LA", "P", "RGB", "RGBA"}, "unsupported_png_sample_mode")
            rgba = np.asarray(source.convert("RGBA"))
            check(np.all(rgba[:, :, 3] == 255), "transparent_png")
            return np.ascontiguousarray(rgba[:, :, :3])
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError):
        raise Rejected("invalid_png") from None


def fingerprint(body, pixels):
    height, width = pixels.shape[:2]
    return {
        "encoded_sha256": digest(body),
        "rgb_sha256": digest(pixels.tobytes()),
        "size": [width, height],
        "sketch": f"{sketch(Image.fromarray(pixels)):064x}",
    }


def edit_truth(original, edited, mask, reported_area):
    check(original.shape == edited.shape == mask.shape, "png_geometry_or_orientation_mismatch")
    values = mask[:, :, 0]
    check(
        np.array_equal(values, mask[:, :, 1])
        and np.array_equal(values, mask[:, :, 2])
        and np.isin(values, (0, 255)).all(),
        "nonbinary_semantic_mask",
    )
    support = values > 0
    support_count = int(support.sum())
    check(
        0 < support_count < support.size and abs(support_count - reported_area * support.size) <= 1,
        "semantic_mask_area_mismatch",
    )
    exact = np.any(original != edited, axis=2)
    changed = int(exact.sum())
    check(changed > 0, "empty_exact_edit")
    check(not np.any(exact & ~support), "exact_changes_outside_semantic_mask")
    return exact, support_count


def stable_path(case, role):
    source = case["recipient"]
    require(
        roster.SOURCE.fullmatch(source) is not None and role in ASSETS, "Invalid frozen PNG path identity"
    )
    folder = PurePosixPath("v2/masks/edit" if role == "semantic_mask" else "v2/images") / source[3]
    suffix = {"original": "_A0.png", "edited": "_F5.png", "semantic_mask": "_F5_edit.png"}[role]
    return roster.named_png(case[role]["path"], folder, source + suffix)


def frozen_cases(roster_dir):
    require(sha(PUBLIC_ROSTER) == PUBLIC_SHA, "Published v218 roster changed")
    public = read(PUBLIC_ROSTER)
    private_path = roster_dir / "private-roster.json"
    require(
        sha(roster_dir / "summary.json") == PUBLIC_SHA
        and sha(private_path) == public["private_roster_sha256"],
        "Exact private v218 roster required",
    )
    private = read(private_path)
    require(
        public["publisher_repo"] == roster.REPO
        and public["publisher_revision"] == roster.REVISION
        and public["publisher_generator_record"] == roster.MODEL
        and all(
            private[key] == public[key] for key in private if key not in {"ordered_candidates", "excluded"}
        )
        and all(sha(ROOT / name) == value for name, value in public["code_commitments"].items())
        and all(
            sha(PUBLIC / name) == value for name, value in public["exposure_reference_commitments"].items()
        ),
        "Frozen v218 release, licence, code or exposure identity changed",
    )
    rows = private["ordered_candidates"]
    exposed, exposure_hashes = roster.exposure_keys()
    require(
        len(rows) == 36
        and len({r["recipient"] for r in rows}) == 36
        and public["known_exposure_commitments"] == exposure_hashes
        and sorted(rows, key=roster.rank) == rows,
        "Exact ordered 24-plus-12 roster required",
    )
    for n, row in enumerate(rows):
        require(
            row["ordinal"] == n
            and row["tier"] == ("primary" if n < 24 else "backup")
            and row["binary_qualification"] == "pending"
            and roster.attribution(row["photo_attribution"], row["recipient"])
            == (row["flickr_owner"], row["flickr_photo_id"])
            and not roster.source_keys(row) & exposed,
            "Source attribution, eligibility or order changed",
        )
        for role in ASSETS:
            stable_path(row, role)
    return rows, exposed


def exposure_data():
    graph = read(DATA / roster.EXPOSURE_FILES["v39_graph"])["sources"]
    v189 = read(DATA / roster.EXPOSURE_FILES["v189_inputs"])["sources"]
    expected = read(PUBLIC / "image-jpeg-diagnostic-freeze-v189.json")["qualified_pair_index_sha256"]
    require(sha(PAIRS) == expected, "Exact prior v26 edit-image index required")
    edit_hashes = {json.loads(line)["edited_image"]["sha256"] for line in PAIRS.read_text().splitlines()}
    views = [view for source in v189 for quality in source["inputs"].values() for view in quality.values()]
    known_bytes = edit_hashes | {v["input_sha256"] for v in views}
    known_bytes |= {r.get("original_sha256") or r["source_jpeg_sha256"] for r in graph}
    known_rgb = {r["rgb_sha256"] for r in graph} | {v["decoded_rgb_sha256"] for v in views}
    require(
        len(graph) == 866 and len(edit_hashes) == 397 and len(views) == 590,
        "Prior original, edited and JPEG exposure support changed",
    )
    return graph, known_bytes, known_rgb


def prior_matches(case, fingerprints, source_keys, exposure):
    graph, known_bytes, known_rgb = exposure
    found = set()
    if roster.source_keys(case) & source_keys:
        found.add("prior_source_identity")
    for view in fingerprints.values():
        if view["encoded_sha256"] in known_bytes:
            found.add("prior_exact_encoded")
        if view["rgb_sha256"] in known_rgb:
            found.add("prior_exact_rgb")
        if any(near(view, old) for old in graph):
            found.add("prior_near_thumbnail")
    return sorted(found)


def overlaps(first, second):
    if set(first["identity_keys"]) & set(second["identity_keys"]):
        return True
    return any(
        x["encoded_sha256"] == y["encoded_sha256"] or x["rgb_sha256"] == y["rgb_sha256"] or near(x, y)
        for x in first["fingerprints"].values()
        for y in second["fingerprints"].values()
    )


def ordered_decisions(attempts):
    components = []
    for row in attempts:
        connected = [i for i, group in enumerate(components) if any(overlaps(row, other) for other in group)]
        merged = [row]
        for i in reversed(connected):
            merged.extend(components.pop(i))
        components.append(merged)
    result = {}
    for group in components:
        prior_link = any(row["prior_matches"] for row in group)
        retained = False
        for row in sorted(group, key=lambda r: r["ordinal"]):
            if row["pair_failure"]:
                status = row["pair_failure"]
            elif prior_link:
                status = "prior_exposure_component"
            elif retained:
                status = "new_duplicate_component"
            else:
                status, retained = "qualified", True
            result[row["ordinal"]] = status
    return result


def fetch_case(session, cache, case, deadline):
    folder = cache / f"{case['ordinal']:02d}"
    folder.mkdir(mode=0o700)
    files = {}
    for role in ASSETS:
        path = stable_path(case, role)
        try:
            body, receipt = download(session, path, case[role]["sha256"], deadline)
        except Rejected as error:
            files[role] = {"status": "rejected", "reason": str(error), "repo_path": path}
        except requests.RequestException:
            files[role] = {"status": "rejected", "reason": "asset_transport_error", "repo_path": path}
        else:
            target = folder / f"{role}.png"
            with target.open("xb") as handle:
                handle.write(body)
            files[role] = {
                "status": "saved",
                "repo_path": path,
                "file": str(target.relative_to(cache)),
                **receipt,
            }
    return files


def verify_file(cache, case, role, recorded):
    require(recorded["repo_path"] == stable_path(case, role), "Saved response belongs to another asset")
    rel = f"{case['ordinal']:02d}/{role}.png"
    path = cache / rel
    if recorded["status"] == "rejected":
        require(not path.exists() and isinstance(recorded["reason"], str), "Rejected asset receipt differs")
        raise Rejected(recorded["reason"])
    require(
        recorded["status"] == "saved"
        and recorded["file"] == rel
        and path.is_file()
        and path.resolve() == path
        and recorded["resolver_revision"] == roster.REVISION
        and recorded["final_host"] in CDN | {HUB}
        and 0 < path.stat().st_size == recorded["bytes"] <= MAX_BYTES,
        "Recorded asset path, scope or size differs",
    )
    body = path.read_bytes()
    require(
        verify_asset_bytes(body, case[role]["sha256"], recorded["publisher_asset_commitment"])
        == recorded["sha256"],
        "Retained asset digest differs",
    )
    return body


def inspect_case(cache, case, files, source_keys, exposure):
    failure = None
    pixels, fingerprints = {}, {}
    for role in ASSETS:
        try:
            body = verify_file(cache, case, role, files[role])
        except Rejected as error:
            if files[role]["status"] == "saved":
                raise ValueError("A verified saved asset changed after acquisition") from error
            failure = failure or str(error)
            continue
        try:
            pixels[role] = png_pixels(body, case["size"])
        except Rejected as error:
            failure = failure or str(error)
        else:
            if role != "semantic_mask":
                fingerprints[role] = fingerprint(body, pixels[role])
    exact = None
    measured = None
    if failure is None:
        try:
            exact, support = edit_truth(*(pixels[role] for role in ASSETS), case["semantic_edit_area"])
        except Rejected as error:
            failure = str(error)
        else:
            measured = {
                "changed_pixels": int(exact.sum()),
                "support_pixels": support,
                "exact_mask_pixel_sha256": digest(exact.astype(np.uint8).tobytes()),
                "semantic_mask_sha256": files["semantic_mask"]["sha256"],
                "semantic_mask_binding": "publisher_asset_commitment"
                if files["semantic_mask"]["publisher_asset_commitment"]
                else "exact_acquired_bytes",
            }
    return {
        "ordinal": case["ordinal"],
        "tier": case["tier"],
        "publisher_split": case["publisher_split"],
        "files": files,
        "fingerprints": fingerprints,
        "identity_keys": sorted(":".join(key) for key in roster.source_keys(case)),
        "prior_matches": prior_matches(case, fingerprints, source_keys, exposure),
        "pair_failure": failure,
        "measured": measured,
    }, exact


def bindings(roster_dir, cache):
    cases, source_keys = frozen_cases(roster_dir)
    exposure = exposure_data()
    plan = {
        "schema": "chipmunk.image-bg-real-acquisition-plan/221",
        "published_v218_sha256": PUBLIC_SHA,
        "private_roster_sha256": sha(roster_dir / "private-roster.json"),
        "v26_edit_commitment_sha256": sha(PAIRS),
        "code_commitments": {name: sha(ROOT / name) for name in CODE},
        "runtime_versions": {
            name: importlib.metadata.version(name) for name in ("requests", "numpy", "Pillow")
        },
        "roster_dir": str(roster_dir),
        "private_cache": str(cache),
        "primary": roster.PRIMARY,
        "backup": roster.BACKUP,
        "requested_asset_roles": list(ASSETS),
        "max_bytes_per_file": MAX_BYTES,
        "max_redirects_per_file": MAX_REDIRECTS,
        "time_budget_s": TOTAL_SECONDS,
        "scoring_permitted": False,
    }
    return plan, cases, source_keys, exposure


def prepare(roster_dir, trial, cache):
    require(
        not trial.exists()
        and not cache.exists()
        and cache.parent.is_dir()
        and not trial.is_relative_to(cache)
        and not cache.is_relative_to(trial),
        "Fresh separate private study and cache destinations required",
    )
    plan, _, _, _ = bindings(roster_dir, cache)
    trial.mkdir(mode=0o700)
    path = trial / "plan.json"
    write(path, plan)
    print(json.dumps({"plan_sha256": sha(path), "primary": roster.PRIMARY, "backup": roster.BACKUP}))


def lease(trial, expected):
    path = trial / "plan.json"
    require(sha(path) == expected, "Exact prospective acquisition plan SHA-256 required")
    planned = read(path)
    cache = Path(planned["private_cache"])
    require(cache == cache.resolve(), "Private cache path changed")
    plan, cases, source_keys, exposure = bindings(Path(planned["roster_dir"]), cache)
    require(plan == planned, "Acquisition code, runtime or roster changed")
    return cache, cases, source_keys, exposure


def enough(attempts):
    return len(attempts) >= roster.PRIMARY and (
        sum(value == "qualified" for value in ordered_decisions(attempts).values()) >= roster.PRIMARY
        or len(attempts) == roster.PRIMARY + roster.BACKUP
    )


def acquire(trial, expected):
    cache, cases, source_keys, exposure = lease(trial, expected)
    cache.mkdir(mode=0o700)
    deadline = time.monotonic() + TOTAL_SECONDS
    attempts = []
    path = trial / "attempts.jsonl"
    with path.open("x") as journal, requests.Session() as session:
        session.trust_env = False
        for case in cases:
            require(time.monotonic() < deadline, "Acquisition time budget exhausted before complete roster")
            files = fetch_case(session, cache, case, deadline)
            result, _ = inspect_case(cache, case, files, source_keys, exposure)
            attempts.append(result)
            journal.write(json.dumps(result, allow_nan=False) + "\n")
            journal.flush()
            require(time.monotonic() < deadline, "Acquisition time budget exhausted before complete roster")
            if enough(attempts):
                break
    require(enough(attempts), "Complete ordered bounded attempt journal required")
    write(
        trial / "acquired.json",
        {"plan_sha256": expected, "journal_sha256": sha(path), "attempted": len(attempts)},
    )
    print(json.dumps({"attempted": len(attempts), "journal_sha256": sha(path)}))


def write_truth(path, mask):
    with path.open("xb") as handle:
        Image.fromarray(mask.astype(np.uint8) * 255).save(handle, format="PNG")
    with Image.open(path) as saved:
        require(np.array_equal(np.asarray(saved) > 0, mask), "Saved exact edit truth failed readback")
    return sha(path)


def seal(trial, expected):
    cache, cases, source_keys, exposure = lease(trial, expected)
    receipt, path = read(trial / "acquired.json"), trial / "attempts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(
        receipt == {"plan_sha256": expected, "journal_sha256": sha(path), "attempted": len(rows)}
        and roster.PRIMARY <= len(rows) <= roster.PRIMARY + roster.BACKUP,
        "Incomplete acquisition receipt",
    )
    attempts, masks = [], {}
    for case, row in zip(cases[: len(rows)], rows, strict=True):
        require(not enough(attempts), "Acquisition used a backup after the pool already qualified")
        result, exact = inspect_case(cache, case, row["files"], source_keys, exposure)
        require(result == row, "Independent source and pixel readback differs")
        attempts.append(result)
        if exact is not None:
            masks[row["ordinal"]] = exact
    require(enough(attempts), "Incomplete bounded backup schedule")
    decisions = ordered_decisions(attempts)
    accepted = [row for row in attempts if decisions[row["ordinal"]] == "qualified"]
    require(len(accepted) <= roster.PRIMARY, "Unexpected unbounded accepted roster")
    destination = trial / "qualified-private.json"
    require(
        not destination.exists() and not (trial / "summary.json").exists(), "Preserve earlier qualification"
    )
    retained = []
    for row in accepted:
        n = row["ordinal"]
        truth_path = cache / f"{n:02d}/exact_truth.png"
        mask_sha = write_truth(truth_path, masks[n])
        retained.append(
            {
                "source": cases[n],
                "measurement": row,
                "truth_file": str(truth_path.relative_to(cache)),
                "truth_png_sha256": mask_sha,
            }
        )
    write(
        destination,
        {
            "schema": "chipmunk.image-bg-real-pixel-qualified-private/221",
            "plan_sha256": expected,
            "acquisition_journal_sha256": sha(path),
            "cache": str(cache),
            "qualified": retained,
            "attempt_statuses": [{"ordinal": n, "status": decisions[n]} for n in sorted(decisions)],
        },
    )
    summary = {
        "schema": "chipmunk.image-bg-real-pixel-qualification/221",
        "plan_sha256": expected,
        "published_v218_sha256": PUBLIC_SHA,
        "acquisition_journal_sha256": sha(path),
        "qualified_private_sha256": sha(destination),
        "attempted": len(attempts),
        "qualified": len(accepted),
        "backup_attempted": len(attempts) - roster.PRIMARY,
        "unattempted_backups": roster.PRIMARY + roster.BACKUP - len(attempts),
        "target_shortfall": roster.PRIMARY - len(accepted),
        "status_counts": dict(Counter(decisions.values())),
        "qualified_by_publisher_split": dict(Counter(row["publisher_split"] for row in accepted)),
        "accepted_mask_bindings": dict(Counter(row["measured"]["semantic_mask_binding"] for row in accepted)),
        "neural_calls": 0,
        "serving_admitted": False,
    }
    write(trial / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("prepare")
    preflight.add_argument("roster_dir", type=Path)
    preflight.add_argument("trial", type=Path)
    preflight.add_argument("cache", type=Path)
    for name in ("acquire", "seal"):
        command = sub.add_parser(name)
        command.add_argument("trial", type=Path)
        command.add_argument("plan_sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.roster_dir.resolve(), args.trial.resolve(), args.cache.resolve())
    else:
        (acquire if args.command == "acquire" else seal)(args.trial.resolve(), args.plan_sha256)


if __name__ == "__main__":
    main()
