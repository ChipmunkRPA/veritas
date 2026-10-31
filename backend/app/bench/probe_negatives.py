"""Score-blind OpenVid frame preparation for isolated, exploratory research.

Planning touches metadata only. No benchmark split or commercial/training admission
is changed; a separate, explicit visual review precedes exploratory role assignment.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from PIL import Image

from app.bench import identity

KNOWN_ANIMATION_PARENT = "1gCMSJw0v0w"
_SHA = re.compile(r"[0-9a-f]{64}")
_PARENT = re.compile(r"[A-Za-z0-9_-]{11}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MIN_GROUPS = {"fit": 40, "calibration": 20, "verification": 20}
LIMITATIONS = (
    "Exploratory research only; no automatic training or production admission.",
    "Original fit membership is retained; derived calibration/verification are exploratory, not independent benchmark holdouts.",
    "Source audit labels do not establish camera originality; every extracted frame needs explicit visual screening.",
    "Video-parent and known identity grouping do not prove camera, person, scene, or upstream-source independence.",
    "Recorded CC-BY-4.0 metadata does not resolve publisher research/non-commercial intent or upstream rights; commercial admission remains unresolved.",
    "D-12 permits OpenVid frames in principle; this preparation remains an isolated research proxy, not authenticated R1 camera data.",
)


class NegativePanelError(ValueError):
    """Preparation or assignment would violate the frozen research contract."""


def _json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _relative_path(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and all(ord(c) >= 32 and ord(c) != 127 for c in value) and
            not Path(value).is_absolute() and ".." not in Path(value).parts)


def _metadata_errors(row: dict) -> list[str]:
    errors = []
    for key, expected in (("source_id", "openvid-1m"), ("source_dataset", "nkp37/OpenVid-1M"),
                          ("modality", "video"), ("label", "real"), ("role", "core"), ("split", "fit")):
        if row.get(key) != expected:
            errors.append(f"ineligible_{key}")
    if row.get("audit_ok") is not True:
        errors.append("source_not_audited")
    if identity.source_video_id(row) is None:
        errors.append("unrecognised_parent")
    if not _SHA.fullmatch(str(row.get("item_id", ""))):
        errors.append("invalid_native_sha256")
    if not _REVISION.fullmatch(str(row.get("source_revision", ""))):
        errors.append("missing_source_revision")
    if str(row.get("licence", "")).lower() != "cc-by-4.0":
        errors.append("unexpected_licence")
    if row.get("commercial_ok") is not True:
        errors.append("source_commercial_metadata_not_true")
    if not _relative_path(row.get("native_path")) or Path(str(row.get("native_path", ""))).suffix.lower() != ".mp4":
        errors.append("invalid_native_path")
    for key in ("native_bytes", "width", "height"):
        if not _positive_int(row.get(key)):
            errors.append(f"invalid_{key}")
    if not _finite(row.get("duration_s")) or row["duration_s"] <= 0:
        errors.append("invalid_duration_s")
    return errors


def plan_openvid_negatives(rows: list[dict], *, excluded_parent_ids: Iterable[str] = (),
                           excluded_item_ids: Iterable[str] = ()) -> dict:
    """Select one clip per parent, without opening any media or reading scores.

    All manifest rows participate in identity components, including test/wild and
    excluded rows. Connected non-fit/non-core/non-real/non-audited identities and
    supplied prior-pilot identities are quarantined before deterministic selection.
    Unresolved lineage elsewhere is disclosed, never called a complete identity audit.
    """
    excluded_parents = set(excluded_parent_ids) | {KNOWN_ANIMATION_PARENT}
    excluded_items = set(excluded_item_ids)
    if any(not isinstance(p, str) or not _PARENT.fullmatch(p) for p in excluded_parents):
        raise NegativePanelError("invalid excluded parent ID")
    if any(not isinstance(i, str) or not _SHA.fullmatch(i) for i in excluded_items):
        raise NegativePanelError("invalid excluded item SHA256")
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k]
                                            for k in ("item_id", "group_key")):
            raise NegativePanelError("full metadata requires nonempty item_id and group_key")
    try:
        components = identity.components(rows)
        grouped = identity.grouped_rows(rows)
    except (TypeError, ValueError) as exc:
        raise NegativePanelError(f"invalid identity metadata: {exc}") from exc
    blocked: dict[int, set[str]] = {}
    for indices in components:
        members = [rows[i] for i in indices]
        reasons = set()
        if any(r.get("split") != "fit" or r.get("role") != "core" for r in members):
            reasons.add("connected_non_fit_or_non_core_identity")
        if any(r.get("label") != "real" or r.get("audit_ok") is not True for r in members):
            reasons.add("connected_non_real_or_unaudited_identity")
        parents = {identity.source_video_id(r) for r in members}
        parents.update(lineage["youtube_source_video"] for r in members
                       if (lineage := identity.talkinghead_lineage(r)))
        if parents & excluded_parents or any(r["item_id"] in excluded_items for r in members):
            reasons.add("connected_excluded_parent_or_item")
        for i in indices:
            blocked[i] = reasons
    candidates, excluded = [], []
    seen_parents: set[str] = set()
    # Item order is stable even if input parquet order changes or a parent has several clips.
    indices = sorted(range(len(rows)), key=lambda i: (rows[i]["item_id"],
                                                     str(rows[i].get("source_row_key", "")),
                                                     str(rows[i].get("native_path", ""))))
    for i in indices:
        row = rows[i]
        if row.get("source_id") != "openvid-1m":
            continue
        parent = identity.source_video_id(row)
        reasons = set(_metadata_errors(row)) | blocked[i]
        if parent in seen_parents:
            reasons.add("repeat_parent_stable_first_item")
        if reasons:
            excluded.append({"item_id": row["item_id"], "parent_id": parent, "reasons": sorted(reasons)})
            continue
        seen_parents.add(parent)
        candidate = {key: row[key] for key in (
            "item_id", "source_id", "source_dataset", "source_revision", "source_row_key", "licence",
            "native_path", "native_bytes", "width", "height", "duration_s")}
        candidate.update(original_item_id=row["item_id"], parent_id=parent,
                         source_native_sha256=row["item_id"], source_group_key=row["group_key"],
                         group_key=grouped[i]["group_key"], original_split="fit", source_split="fit",
                         source_audit_ok=True, source_commercial_ok=row["commercial_ok"],
                         source_modality="video", label="real", role="research_candidate",
                         audit_ok=False, split="unassigned", training_admitted=False,
                         limitations=list(LIMITATIONS))
        candidate["selection_metadata_sha256"] = _json_sha(candidate)
        candidates.append(candidate)
    parents = sorted(seen_parents)
    return {"schema_version": 1, "candidates": candidates, "excluded": excluded,
            "input_rows": len(rows), "candidate_count": len(candidates),
            "source_group_count": len({r["group_key"] for r in candidates}),
            "native_bytes": sum(r["native_bytes"] for r in candidates),
            "parent_set_sha256": hashlib.sha256("\n".join(parents).encode()).hexdigest(),
            "parent_set_hash_scheme": "SHA256(UTF8(newline.join(sorted(unique parent IDs)))); no trailing newline",
            "excluded_parent_ids": sorted(excluded_parents), "excluded_item_ids": sorted(excluded_items),
            "metadata_identity_limitations": {
                "unrecognised_openvid_item_ids": sorted(r["item_id"] for r in rows
                    if r.get("source_id") == "openvid-1m" and identity.source_video_id(r) is None),
                "unrecognised_talkinghead_item_ids": sorted(r["item_id"] for r in rows
                    if r.get("source_id") == "talkingheadbench" and identity.talkinghead_lineage(r) is None),
                "media_rehashed": False, "complete_identity_audit": False},
            "exploratory_only": True, "training_admitted": False, "limitations": list(LIMITATIONS)}


def _source_path(root: Path, relative: str) -> Path:
    if not _relative_path(relative):
        raise NegativePanelError("invalid native relative path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise NegativePanelError("native path escapes benchmark or is not a file")
    return path


def _sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_source(path: Path, candidate: dict) -> None:
    if path.stat().st_size != candidate["native_bytes"] or _sha_file(path) != candidate["source_native_sha256"]:
        raise NegativePanelError("native bytes differ from the planned SHA256/size")


def _run(command: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                              timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NegativePanelError(f"decoder failed or timed out: {type(exc).__name__}") from exc
    if proc.returncode != 0:
        raise NegativePanelError(f"decoder returned {proc.returncode}: {proc.stderr.decode(errors='replace')[-500:]}")
    return proc


def _validate_candidate(candidate: dict) -> None:
    expected = candidate.get("selection_metadata_sha256")
    payload = {k: v for k, v in candidate.items() if k != "selection_metadata_sha256"}
    if not isinstance(expected, str) or expected != _json_sha(payload):
        raise NegativePanelError("selection metadata differs from the plan")
    if (candidate.get("source_split") != "fit" or candidate.get("original_split") != "fit"
            or candidate.get("source_audit_ok") is not True or candidate.get("audit_ok") is not False
            or candidate.get("training_admitted") is not False
            or candidate.get("parent_id") == KNOWN_ANIMATION_PARENT
            or identity.source_video_id(candidate) != candidate.get("parent_id")):
        raise NegativePanelError("candidate is not an unaudited fit-source research plan")


def extract_midpoint(candidate: dict, *, benchmark_root: Path, candidate_dir: Path,
                     timeout_s: float = 30, max_side: int = 2048) -> dict:
    """Decode the first frame at/after the video midpoint, retaining original PTS.

    FFmpeg runs locally with only the file input protocol, no autorotation or
    frame interpolation. Selection precedes optional downscaling. Each subprocess
    has a timeout; an exclusive-create PNG is never overwritten. The caller owns
    immutable plan/result persistence and subsequent manual visual review.
    """
    _validate_candidate(candidate)
    if not _finite(timeout_s) or not 0 < timeout_s <= 300:
        raise NegativePanelError("timeout_s must be finite and in (0, 300]")
    if not _positive_int(max_side) or not 1 <= max_side <= 4096:
        raise NegativePanelError("max_side must be an integer in [1, 4096]")
    deadline = time.monotonic() + timeout_s

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise NegativePanelError("per-item extraction deadline expired")
        return seconds

    root = Path(benchmark_root).resolve(strict=True)
    output = Path(candidate_dir).absolute()
    # Existing ancestor symlinks are forbidden: writes must stay in the named candidate tree.
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise NegativePanelError("candidate directory must not contain symlink ancestors")
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise NegativePanelError("candidate directory must be separate from benchmark")
    target = output / f"{candidate['parent_id']}.png"
    if target.exists() or target.is_symlink():
        raise NegativePanelError("refusing to overwrite existing frame")
    source = _source_path(root, candidate["native_path"])
    _verify_source(source, candidate)  # No decoder sees bytes before verification.
    probe = _run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-f", "mov", "-select_streams", "v:0",
                  "-show_entries", "stream=width,height,start_time,duration,time_base:format=duration",
                  "-of", "json", "file:" + str(source)], remaining())
    try:
        info = json.loads(probe.stdout)
        stream = info["streams"][0]
        duration = float(stream.get("duration", info["format"]["duration"]))
        start = float(stream["start_time"])
        stream_time_base = stream["time_base"]
        numerator, denominator = (int(x) for x in stream_time_base.split("/"))
        if (not math.isfinite(duration) or duration <= 0 or not math.isfinite(start)
                or numerator <= 0 or denominator <= 0):
            raise ValueError("invalid timing")
        if abs(float(info["format"]["duration"]) - candidate["duration_s"]) > 0.01:
            raise ValueError("duration differs from native metadata")
        width, height = stream["width"], stream["height"]
        if (width, height) != (candidate["width"], candidate["height"]):
            raise ValueError("dimensions differ from native metadata")
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise NegativePanelError(f"invalid or inconsistent video timing/dimensions: {exc}") from exc
    midpoint = duration / 2
    requested_pts_s = start + midpoint
    # select runs against original input timestamps; showinfo logs the selected native PTS.
    scale = min(1.0, max_side / max(width, height))
    out_width, out_height = max(1, math.floor(width * scale)), max(1, math.floor(height * scale))
    filters = (f"select=gte(t\\,{requested_pts_s:.12f}),showinfo,"
               f"scale={out_width}:{out_height}:flags=lanczos,format=rgb24")
    proc = _run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "info", "-threads", "1",
                 "-protocol_whitelist", "file", "-copyts", "-noautorotate", "-f", "mov", "-i", "file:" + str(source),
                 "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filters, "-frames:v", "1",
                 "-fps_mode", "passthrough", "-threads", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1"],
                remaining())
    log = proc.stderr.decode(errors="replace")
    match = re.search(r"\[Parsed_showinfo_\d+[^\]]*\]\s+n:\s*0\s+pts:\s*(-?\d+)\s+pts_time:\s*([-+\deE.]+)", log)
    if match is None:
        raise NegativePanelError("decoder did not report the selected frame PTS")
    configured = re.search(r"\[Parsed_showinfo_\d+[^\]]*\]\s+config in time_base:\s*(\d+)/(\d+)", log)
    if configured is None or int(configured[1]) <= 0 or int(configured[2]) <= 0:
        raise NegativePanelError("decoder did not report a valid frame time base")
    numerator, denominator = int(configured[1]), int(configured[2])
    time_base = f"{numerator}/{denominator}"
    pts = int(match[1])
    pts_s = pts * numerator / denominator
    logged_pts_s = float(match[2])
    if not math.isfinite(logged_pts_s) or abs(logged_pts_s - pts_s) > max(1e-4, abs(pts_s) * 1e-5):
        raise NegativePanelError("decoder PTS disagrees with stream time base")
    if pts_s + 1e-8 < requested_pts_s or pts_s >= start + duration + 1e-8:
        raise NegativePanelError("decoded PTS lies outside the requested midpoint interval")
    try:
        with Image.open(io.BytesIO(proc.stdout)) as image:
            if image.format != "PNG" or image.size != (out_width, out_height) or image.mode != "RGB":
                raise NegativePanelError("decoder produced an unexpected PNG grid or mode")
            image.load()
            pixels = image.tobytes()
    except (OSError, ValueError) as exc:
        raise NegativePanelError(f"invalid decoded PNG: {exc}") from exc
    _verify_source(source, candidate)  # Refuse a source altered during decoding.
    remaining()  # An over-budget extraction is never published as a complete frame.
    content_sha = hashlib.sha256(proc.stdout).hexdigest()
    pixel_sha = hashlib.sha256(f"RGB:{out_width}x{out_height}:".encode() + pixels).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{candidate['parent_id']}.", suffix=".part", dir=output,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(proc.stdout)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, target)  # Atomic publication; fails if another writer already created the target.
    except FileExistsError as exc:
        raise NegativePanelError("refusing to overwrite existing frame") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    result = copy.deepcopy(candidate)
    result.update(item_id=content_sha, content_sha256=content_sha, pixel_sha256=pixel_sha,
                  pixel_hash_scheme="SHA256(UTF8('RGB:<width>x<height>:') + row-major RGB8 bytes)",
                  path=str(target), native_path=str(target), original_native_path=candidate["native_path"],
                  native_bytes=len(proc.stdout), source_native_bytes=candidate["native_bytes"],
                  width=out_width, height=out_height, modality="image", camera_real_audit=None,
                  audit_ok=False, commercial_ok=False, split="unassigned", exploratory_role=None,
                  real_stratum="video-frame-derived/R4-research-proxy", training_admitted=False,
                  extraction={"policy": "first decoded frame at or after stream midpoint; no interpolation",
                              "requested_midpoint_s": midpoint, "stream_start_time_s": start,
                              "stream_duration_s": duration, "requested_pts_s": requested_pts_s,
                              "decoded_pts": pts, "decoded_pts_s": pts_s, "time_base": time_base,
                              "stream_time_base": stream_time_base,
                              "source_width": width, "source_height": height, "max_side": max_side,
                              "resized": (out_width, out_height) != (width, height),
                              "scale_filter": "lanczos", "autorotate": False})
    result["extraction_metadata_sha256"] = _json_sha(result)
    return result


def assign_exploratory_roles(rows: list[dict], *, group_counts: dict[str, int], seed: str) -> dict:
    """Freeze disjoint research roles for explicitly accepted, hash-bound frames.

    Minimum 40/20/20 source groups for fit/calibration/verification is a feasibility
    safeguard, not a power calculation. Roles never change original benchmark fit
    membership or set audit_ok/training_admitted; callers retain all rejected rows.
    """
    if not isinstance(seed, str) or not seed.strip():
        raise NegativePanelError("role assignment needs a nonempty prespecified seed")
    if (set(group_counts) != set(_MIN_GROUPS) or any(type(group_counts[k]) is not int
            or group_counts[k] < minimum for k, minimum in _MIN_GROUPS.items())):
        raise NegativePanelError("unsafe group counts: require at least 40 fit, 20 calibration, 20 verification")
    ids, parents = set(), set()
    for row in rows:
        item = row.get("item_id")
        parent = row.get("parent_id")
        if (not isinstance(item, str) or not _SHA.fullmatch(item) or item in ids
                or not isinstance(parent, str) or not _PARENT.fullmatch(parent) or parent in parents
                or row.get("content_sha256") != item or not _SHA.fullmatch(str(row.get("pixel_sha256", "")))):
            raise NegativePanelError("invalid or duplicate frame/parent identity")
        ids.add(item)
        parents.add(parent)
        audit = row.get("camera_real_audit")
        if (not isinstance(audit, dict) or audit.get("decision") != "accepted"
                or audit.get("image_sha256") != item or any(not isinstance(audit.get(k), str) or not audit[k].strip()
                    for k in ("reviewer", "reviewed_at", "basis", "comment"))):
            raise NegativePanelError("each frame needs explicit hash-bound manual camera-like visual acceptance")
        try:
            if datetime.fromisoformat(audit["reviewed_at"].replace("Z", "+00:00")).utcoffset() is None:
                raise ValueError("missing timezone")
        except ValueError as exc:
            raise NegativePanelError("manual review timestamp must be ISO-8601 with timezone") from exc
        if (row.get("source_split") != "fit" or row.get("original_split") != "fit"
                or row.get("audit_ok") is not False or row.get("training_admitted") is not False
                or row.get("split") != "unassigned" or row.get("exploratory_role") is not None
                or not isinstance(row.get("group_key"), str) or not row["group_key"]
                or row.get("source_id") != "openvid-1m" or identity.source_video_id(row) != parent
                or row.get("source_dataset") != "nkp37/OpenVid-1M" or row.get("modality") != "image"
                or row.get("real_stratum") != "video-frame-derived/R4-research-proxy"
                or row.get("commercial_ok") is not False):
            raise NegativePanelError("frame must remain an unassigned fit-derived research candidate")
        # The only permitted change after extraction is the explicit, content-bound review.
        extraction = {k: v for k, v in row.items() if k != "extraction_metadata_sha256"}
        extraction["camera_real_audit"] = None
        if row.get("extraction_metadata_sha256") != _json_sha(extraction):
            raise NegativePanelError("frame metadata differs from the immutable extraction record")
    # Include exact pixel duplicates as well as all preserved full-manifest identity groups.
    identity_rows = [{**row, "norm_sha256": row["pixel_sha256"]} for row in rows]
    grouped = identity.grouped_rows(identity_rows)
    groups = sorted({r["group_key"] for r in grouped},
                    key=lambda group: (hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest(), group))
    if sum(group_counts.values()) != len(groups):
        raise NegativePanelError("role counts must exactly exhaust disjoint identity groups")
    assignments = {}
    offset = 0
    for role in _MIN_GROUPS:
        for group in groups[offset:offset + group_counts[role]]:
            assignments[group] = role
        offset += group_counts[role]
    result_rows = []
    for row, identity_row in zip(rows, grouped, strict=True):
        result = copy.deepcopy(row)
        result["group_key"] = identity_row["group_key"]
        result["exploratory_role"] = assignments[result["group_key"]]
        result_rows.append(result)
    result_rows.sort(key=lambda r: r["item_id"])
    role_records = [{k: row[k] for k in ("item_id", "parent_id", "group_key", "exploratory_role")}
                    for row in result_rows]
    return {"schema_version": 1, "rows": result_rows, "seed": seed,
            "group_counts": dict(group_counts), "row_counts": dict(Counter(r["exploratory_role"] for r in result_rows)),
            "role_assignment_sha256": _json_sha(role_records), "role_assignment_records": role_records,
            "exploratory_only": True, "training_admitted": False, "limitations": list(LIMITATIONS)}
