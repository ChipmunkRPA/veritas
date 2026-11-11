"""Offline evidence-package verifier (ARCHITECTURE §6.5; PRD RP-05, RP-06; D-11, D-30; BUILD_SPEC R-4).

`python -m app.cli verify <package.zip> (--public-key PEM | --expect-key-id ID) [--json]` needs no server, database or network:
this module imports the standard library, `cryptography` (through `app.reports.signing`) and `app.core.canonical` only. Checks,
in order:
1. **archive**: a readable ZIP within size limits, with safe entry names (no absolute paths, `..`, backslashes or duplicates),
   holding the D-30 files;
2. **hashes**: every file listed in `manifest.sha256` (`<sha256>  <path>`, sorted by path) hashes to its line;
3. **completeness**: no listed file is missing and no file is unlisted (the manifest and its signature excepted);
4. **manifest signature**: `manifest.sha256.sig` is an Ed25519 signature over `manifest.sha256` by `instance_public_key.pem`, which
   must be the key given with `--public-key` and carry the id given with `--expect-key-id` where they are given;
5. **report signatures**: each report listed in `bundle.json` hashes to its recorded SHA-256 and its signature verifies with that key;
6. **custody**: in the extract of `bundle.json`, each event hash recomputes from the event's fields and `prev_hash`, seq numbers
   strictly increase, adjacent seq numbers are linked by `prev_hash`, the export anchor is not behind the last event (and names
   its hash when it is the last), and each report has its `report_generated` event with the report's hash.

**The signer (BUILD_SPEC N-11; SEC-06).** A package carries its own public key, and whoever alters a package can replace that
key, sign every file again and rewrite the README. The six checks therefore say that the files agree with each other; who signed
them is known only against a key that reached the reader apart from the package (the pin). Three outcomes:
- `verified` (exit 0): every check passes and the signing key is the pinned one;
- `self_consistent` (exit 3, "SELF-CONSISTENT: signer not authenticated"): every check passes and no key was pinned;
- `not_verified` (exit 1): a check fails, a pinned key that did not sign among them.
`Verification.ok` is true for `verified` only. The pin is the instance's public key (`--public-key`, the PEM of
`GET /api/v2/instance/public-key`) or its key id (`--expect-key-id`, shown in the export dialog). A key id is the first 64 bits of
the SHA-256 of the key: enough against a key made without a search, weaker than the whole key, which `--public-key` compares.

The extract is the analysis's projection of the global chain, so seq gaps between events of other analyses are expected; the
whole chain is verified on the instance (`app.cli audit verify`).

The optional `--source-image IMAGE` check uses Pillow to compare unannotated crops with the submitted original locally.
The optional `--source-audio MEDIA` check uses FFmpeg to compare exact samples and source timestamps from an audio or video original.
The optional `--source-video VIDEO` check compares native frames, crops, and measurement inputs with the video original.
"""
from __future__ import annotations

import hashlib
import json
import re
import wave
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from subprocess import SubprocessError
from typing import Any

from app.core.canonical import canonical_json
from app.reports import signing

SCHEMA = "veritas.evidence/2.0"
REQUIRED = ("README.txt", "bundle.json", "instance_public_key.pem", "manifest.sha256", "manifest.sha256.sig", "reproduce.md",
            "report.pdf")
UNLISTED = frozenset({"manifest.sha256", "manifest.sha256.sig"})
BODY_FIELDS = ("seq", "ts", "actor_id", "action", "case_id", "media_id", "analysis_id", "report_id", "object_ids", "before_hash",
               "after_hash", "details")  # app.audit.chain.BODY_FIELDS
MAX_ENTRIES = 50_000
MAX_UNCOMPRESSED = 16 * 1024 ** 3
CHUNK = 1024 * 1024
_LINE = re.compile(r"([0-9a-f]{64})  (\S[^\n]*)")
_KEY_ID = re.compile(r"ed25519:[0-9a-f]{16}")  # app.reports.signing.key_id_of
SIGNER_NOT_AUTHENTICATED = "SELF-CONSISTENT: signer not authenticated"
# N-19k (browser QA Q2-14 "the verifier"): a draft package is named as one on the line of the verdict. It read "VERIFIED"
# alone: the signature of a draft verifies, and the report in it is not issued
DRAFT_PACKAGE = "DRAFT package: the report in it is not issued"


def draft_note(bundle: dict | None) -> str:
    """" · DRAFT package: …" for the bundle of a draft export, else ""."""
    return f" · {DRAFT_PACKAGE}" if (bundle or {}).get("kind") == "draft" else ""
EXIT_VERIFIED, EXIT_NOT_VERIFIED, EXIT_ERROR, EXIT_SELF_CONSISTENT = 0, 1, 2, 3
EXIT = {"verified": EXIT_VERIFIED, "not_verified": EXIT_NOT_VERIFIED, "self_consistent": EXIT_SELF_CONSISTENT}
HOW_TO_PIN = ("The files agree with each other and with the key inside the package, which whoever altered a package could have "
              "replaced.\nGive the instance key, obtained from the instance or its operator and not from this package: "
              "--public-key <PEM file> or --expect-key-id <key id>.")


class PackageUnreadable(ValueError):
    """The path is not a readable ZIP archive."""


class PinInvalid(ValueError):
    """The expected key id is not a key id: nothing could match it, so it is refused before anything is checked."""


def key_id_pin(text: str) -> str:
    """`ed25519:<16 hex>` as the export dialog prints it; the prefix and the case are optional (an id read aloud or typed)."""
    value = text.strip().lower()
    value = value if value.startswith("ed25519:") else f"ed25519:{value}"
    if _KEY_ID.fullmatch(value) is None:
        raise PinInvalid(f"{text!r} is not an instance key id (ed25519: and 16 hexadecimal characters).")
    return value


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class Verification:
    checks: list[Check] = field(default_factory=list)
    key_id: str | None = None
    bundle: dict[str, Any] = field(default_factory=dict)
    pinned: str | None = None  # what the caller gave to name the signer: "public key", "key id", or both
    signer_authenticated: bool = False  # the manifest signature verifies with the pinned key

    @property
    def consistent(self) -> bool:
        """Every check passes: the files agree with each other and with the key the package carries."""
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def outcome(self) -> str:
        if not self.consistent:
            return "not_verified"
        return "verified" if self.signer_authenticated else "self_consistent"

    @property
    def ok(self) -> bool:
        """Verified: consistent, and signed by the key the caller pinned. A consistent package with no pin is not `ok`."""
        return self.outcome == "verified"

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "result": self.outcome, "signer_authenticated": self.signer_authenticated, "pinned": self.pinned,
                "key_id": self.key_id, "bundle_id": self.bundle.get("bundle_id"), "kind": self.bundle.get("kind"),
                "analysis_id": (self.bundle.get("analysis") or {}).get("id"),
                "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks]}


def event_hash(prev_hash: str, event: Mapping[str, Any]) -> str:
    return hashlib.sha256(prev_hash.encode("ascii", "replace") + canonical_json({n: event[n] for n in BODY_FIELDS})).hexdigest()


def verify_package(path: Path, *, public_key_pem: bytes | None = None, expect_key_id: str | None = None,
                   source_image: Path | None = None, source_audio: Path | None = None, source_video: Path | None = None) -> Verification:
    """Check the package at `path`. `public_key_pem` and `expect_key_id` pin the signer; without either the best outcome is
    `self_consistent` (`ok` false). Raises `PinInvalid` for a key id of the wrong form and `PackageUnreadable` for a non-archive."""
    expected = None if expect_key_id is None else key_id_pin(expect_key_id)
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        raise PackageUnreadable(f"{path} is not a readable ZIP archive ({type(e).__name__}).") from None
    with archive as zf:
        out = Verification(pinned=" and ".join(word for word, given in (("public key", public_key_pem), ("key id", expected))
                                               if given is not None) or None)
        try:
            out = _verify(zf, public_key_pem, out, expected)
            if out.consistent and source_image is not None:
                out.checks.append(_image_source(zf, out.bundle, source_image))
            if out.consistent and source_audio is not None:
                out.checks.append(_audio_source(zf, out.bundle, source_audio))
            if out.consistent and source_video is not None:
                out.checks.append(_video_source(zf, out.bundle, source_video))
            return out
        except (zipfile.BadZipFile, zlib.error, EOFError, OSError, NotImplementedError) as e:  # a CRC failure or a damaged entry
            out.checks.append(Check("readable", False, f"an entry cannot be read ({type(e).__name__}: {e})"))
            return out


def _image_source(archive: zipfile.ZipFile, bundle: dict, original: Path) -> Check:
    from PIL import Image

    from app.reports import verify_image_source

    name = "original image crops"
    try:
        return Check(name, True, verify_image_source.check(archive, bundle, original))
    except (ValueError, TypeError, KeyError, OSError, Image.DecompressionBombError) as error:
        return Check(name, False, str(error))


def _audio_source(archive: zipfile.ZipFile, bundle: dict, original: Path) -> Check:
    name = "original audio samples"
    try:
        from app.reports import verify_audio_source

        return Check(name, True, verify_audio_source.check(archive, bundle, original))
    except ImportError:
        return Check(name, False, "Install NumPy and the project dependencies for local audio comparison.")
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, wave.Error, EOFError) as error:
        return Check(name, False, str(error))


def _video_source(archive: zipfile.ZipFile, bundle: dict, original: Path) -> Check:
    name = "original video frames"
    try:
        from cv2 import error as ImageTransformError
        from PIL import Image

        from app.reports import verify_video_source
    except ImportError:
        return Check(name, False, "Install the project dependencies for local video comparison.")
    try:
        return Check(name, True, verify_video_source.check(archive, bundle, original))
    except (ImageTransformError, SubprocessError):
        return Check(name, False, "The local decoder could not finish the video source comparison.")
    except (ValueError, TypeError, KeyError, IndexError, OSError, RuntimeError, Image.DecompressionBombError) as error:
        return Check(name, False, str(error))


def _verify(zf: zipfile.ZipFile, public_key_pem: bytes | None, out: Verification | None = None,
            expect_key_id: str | None = None) -> Verification:
    out = out if out is not None else Verification()
    names, problem = _archive(zf)
    out.checks.append(Check("archive", problem is None, problem or f"{len(names)} files with safe names; the D-30 files are present"))
    if problem is not None:
        return out
    manifest = zf.read("manifest.sha256")
    listed, problem = _parse_manifest(manifest)
    if problem is not None:
        out.checks.append(Check("hashes", False, problem))
        return out
    try:
        altered = sorted(n for n, sha in listed.items() if n in names and _sha256(zf, n) != sha)
    except (zipfile.BadZipFile, OSError) as e:  # a CRC failure is an altered byte too
        altered, problem = [], f"a file cannot be read ({e})"
    present = sum(1 for n in listed if n in names)
    out.checks.append(Check("hashes", not altered and problem is None,
                            problem or (f"altered: {', '.join(altered)}" if altered else f"all {present} listed files match")))
    missing, unlisted = sorted(set(listed) - names), sorted(names - set(listed) - UNLISTED)
    out.checks.append(Check("completeness", not missing and not unlisted,
                            "; ".join(x for x in (f"missing: {', '.join(missing)}" if missing else "",
                                                  f"not listed: {', '.join(unlisted)}" if unlisted else "") if x)
                            or f"every file is listed ({len(listed)}) and present"))

    embedded = zf.read("instance_public_key.pem")
    try:
        key = signing.load_public_key(public_key_pem if public_key_pem is not None else embedded)
    except signing.SigningKeyError as e:
        out.checks.append(Check("manifest signature", False, str(e)))
        return out
    out.key_id = signing.key_id_of(key)
    key_problem = None
    if public_key_pem is not None:
        try:
            same = signing.key_id_of(signing.load_public_key(embedded)) == out.key_id
        except signing.SigningKeyError:
            same = False
        key_problem = None if same else "the package embeds a different public key than the one given"
    if key_problem is None and expect_key_id is not None and expect_key_id != out.key_id:
        key_problem = f"the signing key is {out.key_id}, not the expected {expect_key_id}"
    signed = signing.verify(key, manifest, zf.read("manifest.sha256.sig").decode("ascii", "replace"))
    out.signer_authenticated = out.pinned is not None and signed and key_problem is None
    by = f"{out.key_id}, the {out.pinned} given" if out.pinned is not None else f"{out.key_id}, the key inside the package (no key was given)"
    out.checks.append(Check("manifest signature", signed and key_problem is None,
                            key_problem or (f"signed by {by}" if signed else f"the signature does not verify with {out.key_id}")))

    try:
        bundle = json.loads(zf.read("bundle.json"))
    except (ValueError, UnicodeDecodeError):
        out.checks.append(Check("bundle", False, "bundle.json is not valid JSON"))
        return out
    from app.reports import public_verify

    if not isinstance(bundle, dict) or bundle.get("schema") not in {SCHEMA, public_verify.SCHEMA}:
        out.checks.append(Check("bundle", False, "bundle.json has an unsupported evidence schema"))
        return out
    out.bundle = bundle
    out.checks.append(_reports(zf, names, bundle, key, out.key_id))
    if bundle['schema'] == public_verify.SCHEMA:
        out.checks.extend(Check(*item) for item in public_verify.checks(zf, names, bundle))
    else:
        out.checks.append(_custody(bundle))
    return out


def _archive(zf: zipfile.ZipFile) -> tuple[set[str], str | None]:
    infos = zf.infolist()
    if len(infos) > MAX_ENTRIES:
        return set(), f"the archive holds {len(infos)} entries (limit {MAX_ENTRIES})"
    if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED:
        return set(), "the archive expands beyond the size limit"
    names: set[str] = set()
    for info in infos:
        name = info.filename
        limit = 32 * 1024 * 1024 if name.endswith(('.json', '.sha256')) else 16 * 1024 if name.endswith(('.pem', '.sig')) else None
        if limit is not None and info.file_size > limit:
            return names, f"metadata file {name!r} exceeds the size limit"
        if info.is_dir():
            continue
        parts = PurePosixPath(name).parts
        if not name or "\\" in name or name.startswith("/") or ".." in parts or ":" in parts[0]:
            return names, f"unsafe entry name {name!r}"
        if name in names:
            return names, f"duplicate entry {name!r}"
        names.add(name)
    missing = [r for r in REQUIRED if r not in names]
    return names, (f"required files missing: {', '.join(missing)}" if missing else None)


def _parse_manifest(data: bytes) -> tuple[dict[str, str], str | None]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {}, "manifest.sha256 is not UTF-8 text"
    if not text.endswith("\n"):
        return {}, "manifest.sha256 does not end with a newline"
    listed: dict[str, str] = {}
    for n, line in enumerate(text[:-1].split("\n"), 1):
        match = _LINE.fullmatch(line)
        if match is None:
            return {}, f"manifest.sha256 line {n} is not '<sha256>  <path>'"
        sha, name = match.groups()
        if name in listed or name in UNLISTED:
            return {}, f"manifest.sha256 lists {name!r} twice or lists itself"
        listed[name] = sha
    if list(listed) != sorted(listed):
        return {}, "manifest.sha256 is not sorted by path"
    return listed, None


def _sha256(zf: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with zf.open(name) as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _reports(zf: zipfile.ZipFile, names: set[str], bundle: Mapping[str, Any], key: Any, key_id: str) -> Check:
    reports = bundle.get("reports")
    if not isinstance(reports, list) or not reports:
        return Check("report signatures", False, "bundle.json lists no report")
    problems, checked = [], []
    for r in reports:
        if not isinstance(r, Mapping):
            problems.append("a report entry is not an object")
            continue
        file = r.get("file")
        if file not in names:
            problems.append(f"{r.get('id')}: {file!r} is not in the package")
            continue
        data = zf.read(file)
        if hashlib.sha256(data).hexdigest() != r.get("sha256"):
            problems.append(f"{file}: its SHA-256 differs from the recorded report hash")
        elif r.get("signing_key_id") != key_id:
            problems.append(f"{file}: recorded as signed by {r.get('signing_key_id')}, not {key_id}")
        elif not signing.verify(key, data, str(r.get("signature") or "")):
            problems.append(f"{file}: the report signature does not verify")
        else:
            checked.append(file)
    return Check("report signatures", not problems, "; ".join(problems) or f"{', '.join(checked)} signed by {key_id}")


def _custody(bundle: Mapping[str, Any]) -> Check:
    name = "custody"
    events, anchor = bundle.get("custody"), bundle.get("custody_anchor")
    if not isinstance(events, list) or not events:
        return Check(name, False, "bundle.json holds no custody extract")
    previous: tuple[int, str] | None = None
    for event in events:
        if not isinstance(event, Mapping) or any(k not in event for k in (*BODY_FIELDS, "prev_hash", "event_hash")):
            return Check(name, False, "a custody event lacks fields")
        try:
            seq, recomputed = int(event["seq"]), event_hash(str(event["prev_hash"]), event)
        except (TypeError, ValueError):
            return Check(name, False, "a custody event has fields of the wrong type")
        if recomputed != event["event_hash"]:
            return Check(name, False, f"event {seq}: its hash does not match its fields")
        if previous is not None:
            if seq <= previous[0]:
                return Check(name, False, f"event {seq} follows event {previous[0]}: the extract is out of order or repeats an event")
            if seq == previous[0] + 1 and event["prev_hash"] != previous[1]:
                return Check(name, False, f"event {seq}: prev_hash does not name event {previous[0]}")
        previous = (seq, str(event["event_hash"]))
    assert previous is not None
    if not isinstance(anchor, Mapping) or not isinstance(anchor.get("head_seq"), int):
        return Check(name, False, "bundle.json has no custody anchor")
    if anchor["head_seq"] < previous[0] or (anchor["head_seq"] == previous[0] and anchor.get("head_hash") != previous[1]):
        return Check(name, False, "the custody anchor does not match the last event of the extract")
    for r in bundle.get("reports") or []:
        if isinstance(r, Mapping) and not any(e["action"] == "report_generated" and e["report_id"] == r.get("id")
                                              and e["after_hash"] == r.get("sha256") for e in events):
            return Check(name, False, f"report {r.get('id')} has no report_generated event with its hash")
    return Check(name, True, f"{len(events)} events recompute and increase; anchored at chain event {anchor['head_seq']}")
