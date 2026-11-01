"""Fuser loader (ARCHITECTURE §4.4 "Loader"; ML §4.8 "Loading"; D-04, D-17, D-57).

- **Directories** (D-57 as amended 2026-09-30). `VERITAS_FUSER_DIR` when set, and only that directory. Otherwise the artefacts
  of `data_dir/fusion` and of the release directory packaged with the source (`RELEASE_DIR`, hashes in its `MANIFEST.json`)
  are listed together, and the release rule below chooses among them: an instance that kept an older artefact in its data
  directory serves the packaged one after an upgrade. Where both directories hold one version, the packaged file is served;
  when the other file has different bytes it is shadowed, and the start-up log and `fusers status` name it (`shadowed`). A
  caller that names a directory (the benchmark scorer, tests) sees only that directory. A sha256 pin is looked up in every
  file of both directories, shadowed ones included.
- **Candidates.** `fuser-<modality>-<semver>.json` files, newest semver first (a release sorts above its pre-releases).
- **Compatibility.** An artefact is used only when:
  - its schema is `veritas.fuser/1` and its modality is the analysis kind;
  - `reviewed_by` is set (D-17; the loop writes its own flag);
  - every detector it was fitted on is enabled for this kind and profile, with the same weights hash
    (`weights_sha12`, the score-cache key of the benchmark). A detector the licence policy denies is not enabled, so
    it can never feed the default fuser (D-04);
  - every such detector has its weight files installed; otherwise every analysis would run with that check skipped, so
    the fallback with its caps is used instead (D-46);
  - a face-detection detector is enabled for this kind and profile when the artefact holds a detector that needs faces, or
    a gate literal on a face term; otherwise every such check would read "no face detected" (D-49). Re-fusion asks
    instead whether the source's runs hold one (`app.engine.refusion.select_for_source`);
  - its bootstrap matrix loads and matches its recorded sha256.
- **Fallback.** Otherwise the v0.3 hand fusion runs (`fusion_mode = fallback`), and the reasons each candidate was
  rejected are kept for `/api/ready`, logs and the stage output.
- **Pinning.** A refusion job may pin an artefact by sha256; only that artefact is considered.
- **Review** (D-57). The artefact bytes are never edited. A reviewer is recorded in a sidecar
  `fuser-<modality>-<version>.review.json` beside the artefact or in `data_dir/fusion` (`python -m app.cli fusers review`). The
  effective `reviewed_by` is the sidecar's reviewer when the sidecar names the artefact's sha256, and the loop's pending flag
  otherwise (`Selection.reviewed_by`): results, reports, the `fusers` row, `/api/ready` and the cloud preflight read that.

The artefact sha256 is the sha256 of the JSON bytes as read. It goes on the analysis row and into the `fusers` table.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.engine.detector import Modality, Profile
from app.engine.fusion import derived
from app.engine.fusion.artefact import LOOP_REVIEW_FLAG, SCHEMA, Fuser
from app.engine.fusion.features import face_gated

log = get_logger(__name__)

NOWEIGHTS = "noweights"  # score-cache key suffix of detectors without pinned weights
NOT_INSTALLED = "not-installed"  # D-46: the detector's pinned weight files are missing locally
NO_FACE_DETECTOR = "no face detector is enabled"  # D-49
KINDS = ("image", "video", "document", "audio")
KIND_MODALITIES: dict[str, tuple[Modality, ...]] = {
    "image": (Modality.IMAGE,), "video": (Modality.VIDEO, Modality.AUDIO), "document": (Modality.TEXT,),
    "audio": (Modality.AUDIO,)}
_NAME = re.compile(r"fuser-(?P<modality>[a-z]+)-(?P<version>\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)\.json")
RELEASE_DIR = Path(__file__).resolve().parent / "release"  # D-57: the served artefacts, tracked with the source
RELEASE_MANIFEST = "MANIFEST.json"
REVIEW_SCHEMA = "veritas.fuser-review/1"
REVIEW_SUFFIX = ".review.json"


def semver_key(version: str) -> tuple:
    """Semantic-version precedence: numeric core, then a release above its pre-releases, then pre-release identifiers
    (numeric below alphanumeric, a longer set above its prefix)."""
    core, _, pre = version.partition("-")
    major, minor, patch = (int(part) for part in core.split("."))
    if not pre:
        return (major, minor, patch, 1, ())
    ids = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return (major, minor, patch, 0, ids)


def loaded_detector_hashes(registry: Any, kind: str, profile: Profile | str) -> dict[str, str]:
    """{detector id: weights_sha12} for the detectors enabled for this kind and profile."""
    from app.bench.score import weights_sha12

    out: dict[str, str] = {}
    for modality in KIND_MODALITIES.get(kind, ()):
        for detector_id in registry.enabled(modality, profile):
            entry = registry.manifest.detector(detector_id)
            if entry is None or not entry.models:
                out[detector_id] = NOWEIGHTS
                continue
            try:
                sha12 = weights_sha12(registry.manifest, detector_id)
            except KeyError as e:  # an unresolvable model: no artefact can match it
                log.warning("fusion.weights_hash_unavailable", detector_id=detector_id, error=str(e))
                continue
            out[detector_id] = sha12 if models_installed(registry.manifest, entry.models) else NOT_INSTALLED
    for feature in derived.FEATURES:  # H-6: a derived feature is available with its source detector's weights
        if feature.source in out:
            out.setdefault(feature.id, out[feature.source])
    return out


def models_installed(manifest: Any, model_ids: Any) -> bool:
    """Whether every pinned file of the models and their dependencies is present locally (no hashing)."""
    from app.engine import fetchers

    seen: set[str] = set()
    stack = list(model_ids)
    while stack:
        model_id = stack.pop()
        if model_id in seen:
            continue
        seen.add(model_id)
        entry = manifest.model(model_id)
        if entry is None or not fetchers.installed(entry):
            return False
        stack += list(getattr(entry, "depends_on", None) or [])
    return True


def face_detector_enabled(registry: Any, kind: str, profile: Profile | str) -> bool:
    return any(registry.detector_family(d) == "face-detection" for m in KIND_MODALITIES.get(kind, ()) for d in registry.enabled(m, profile))


def needs_faces(registry: Any) -> Callable[[str], bool]:
    """Whether a detector's adapter declares `needs: faces`, read without the enablement check (`DetectorRegistry.adapter_spec`)."""
    def check(detector_id: str) -> bool:
        spec = registry.adapter_spec(detector_id)
        return spec is not None and "faces" in spec.requirements.needs
    return check


def compatibility(doc: Mapping[str, Any], kind: str, loaded: Mapping[str, str], *, face_detector: bool = True,
                  needs_faces: Callable[[str], bool] | None = None, permitted_missing: Mapping[str, str] | None = None) -> str | None:
    """None when the artefact may be used; otherwise the first reason it may not. `face_detector`: whether a face-detection
    detector is available; without one, an artefact holding a detector that needs faces (`needs_faces`) or a gate literal on a
    face term is refused (D-49)."""
    if doc.get("schema") != SCHEMA:
        return f"not a {SCHEMA} artefact"
    if doc.get("modality") != kind:
        return f"modality {doc.get('modality')!r} is not {kind!r}"
    if not doc.get("reviewed_by"):
        return "reviewed_by is not set (D-17)"
    for det in doc.get("detectors") or []:
        detector_id = det.get("id")
        expected = det.get("weights_sha12") or str(det.get("score_key") or "").partition("@")[2]
        if detector_id not in loaded:
            if ((permitted_missing or {}).get(detector_id) == expected
                    and f"miss:{detector_id}" in (doc.get("coef") or {})):
                continue  # Use the fitted missing-input feature; never substitute a new detector's score.
            return f"detector {detector_id} is not enabled (disabled, denied by the licence policy or not in this profile)"
        if loaded[detector_id] == NOT_INSTALLED:
            return f"detector {detector_id} has no installed weights"
        if loaded[detector_id] != expected:
            return f"detector {detector_id} has weights {loaded[detector_id]}, the artefact was fitted on {expected or 'unknown'}"
    if not face_detector:
        gates = doc.get("gates") or {}
        for det in doc.get("detectors") or []:
            detector_id = det.get("id")
            if face_gated(gates.get(detector_id)) or (needs_faces is not None and needs_faces(detector_id)):
                return f"detector {detector_id} depends on faces and {NO_FACE_DETECTOR}"
    return None


@dataclass(frozen=True)
class Review:
    """A review sidecar that names its artefact's sha256 (D-57)."""
    reviewer: str
    date: str
    statement: str
    path: Path


@dataclass(frozen=True)
class Selection:
    kind: str
    mode: Literal["artefact", "fallback"]
    fuser: Fuser | None = None
    sha256: str | None = None
    path: Path | None = None
    version: str | None = None
    reason: str = ""
    rejected: tuple[tuple[str, str], ...] = ()  # (file name, why it was not used)
    review: Review | None = None  # D-57: the matching review sidecar, when one exists
    # N-19k (Q2-14): a fallback beside an artefact of this kind that is installed and was not used (not usable with the
    # checks of this analysis, another version than the configured one, a re-run of a fallback analysis): such a result
    # does not say that no fusion model is loaded
    installed: bool = False
    intentionally_missing: tuple[str, ...] = ()

    @property
    def reviewed_by(self) -> str | None:
        """The effective reviewer (D-57): the sidecar's, else the pending flag; None on the fallback."""
        if self.mode != "artefact":
            return None
        return self.review.reviewer if self.review is not None else LOOP_REVIEW_FLAG

    @property
    def pending_review(self) -> bool:
        return self.mode == "artefact" and self.review is None

    def as_json(self) -> dict[str, Any]:
        return {"mode": self.mode, "sha256": self.sha256, "file": self.path.name if self.path else None,
                "version": self.version, "reason": self.reason, "intentionally_missing": list(self.intentionally_missing),
                "rejected": [{"file": name, "reason": why} for name, why in self.rejected]}


@dataclass(frozen=True)
class _Cached:
    key: tuple[int, int]
    sha256: str
    doc: dict[str, Any]
    fuser: Fuser | None = None
    boot_key: tuple[int, int] | None = None  # the bootstrap matrix as it was when `fuser` was loaded


_cache: dict[Path, _Cached] = {}
_lock = threading.Lock()


def _read(path: Path) -> _Cached:
    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    with _lock:
        hit = _cache.get(path)
    if hit is not None and hit.key == key:
        return hit
    data = path.read_bytes()
    entry = _Cached(key=key, sha256=hashlib.sha256(data).hexdigest(), doc=json.loads(data.decode("utf-8")))
    with _lock:
        _cache[path] = entry
    return entry


def _boot_key(path: Path, doc: Mapping[str, Any]) -> tuple[int, int] | None:
    name = (doc.get("bootstrap") or {}).get("file")
    if not name:
        return None
    stat = (path.parent / Path(name).name).stat()
    return (stat.st_mtime_ns, stat.st_size)


def _fuser(path: Path, entry: _Cached) -> Fuser:
    """The loaded artefact, reloaded and re-verified when its bootstrap matrix changed on disk."""
    boot_key = _boot_key(path, entry.doc)
    if entry.fuser is not None and entry.boot_key == boot_key:
        return entry.fuser
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.sha256:
        raise ValueError("the artefact changed while it was being loaded")
    fuser = Fuser.from_bytes(data, path.parent)
    with _lock:
        _cache[path] = _Cached(key=entry.key, sha256=entry.sha256, doc=entry.doc, fuser=fuser, boot_key=boot_key)
    return fuser


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def search_dirs(fusion_dir: Path | None = None) -> list[Path]:
    """The directories artefacts and review sidecars are read from, in order (D-57)."""
    if fusion_dir is not None:
        return [Path(fusion_dir)]
    if settings.fuser_dir is not None:
        return [Path(settings.fuser_dir)]
    return [settings.data_dir / "fusion", RELEASE_DIR]


def _listed(kind: str, directory: Path) -> list[tuple[str, Path]]:
    if not directory.is_dir():
        return []
    found = [(semver_key(m["version"]), m["version"], p) for p in directory.iterdir()
             if (m := _NAME.fullmatch(p.name)) and m["modality"] == kind and p.is_file()]
    return [(version, path) for _, version, path in sorted(found, key=lambda t: (t[0], t[2].name), reverse=True)]


def _packaged(directory: Path) -> bool:
    return directory.resolve() == RELEASE_DIR.resolve()


def candidates(kind: str, fusion_dir: Path | None = None, *, every: bool = False) -> list[tuple[str, Path]]:
    """The artefact files of a kind in the directories of the order together (D-57), newest version first and, within a
    version, the packaged file first. A file of another directory with the version of a packaged file is left out: the packaged
    artefact is the one served. `every` (a sha256 pin) keeps those files too."""
    found = [(semver_key(version), _packaged(directory), version, path)
             for directory in search_dirs(fusion_dir) for version, path in _listed(kind, directory)]
    found.sort(key=lambda t: (t[0], t[1], t[3].name), reverse=True)
    out: list[tuple[str, Path]] = []
    packaged_versions: set[str] = set()
    for _, packaged, version, path in found:
        if packaged:
            packaged_versions.add(version)
        elif version in packaged_versions and not every:
            continue
        out.append((version, path))
    return out


def shadowed(kind: str) -> list[dict[str, str | None]]:
    """The artefact files outside the release directory that a packaged artefact of the same version and other bytes is served
    in place of (D-57): the file, its sha256 (None when unreadable) and the packaged file's name. An identical copy is not
    listed: nothing differs."""
    directories = search_dirs()
    packaged = {version: path for directory in directories if _packaged(directory) for version, path in _listed(kind, directory)}
    out: list[dict[str, str | None]] = []
    for directory in directories:
        if _packaged(directory):
            continue
        for version, path in _listed(kind, directory):
            if version not in packaged:
                continue
            try:
                sha256 = _read(path).sha256
            except (OSError, ValueError):
                sha256 = None
            if sha256 is None or sha256 != _read(packaged[version]).sha256:
                out.append({"file": str(path), "sha256": sha256, "served": packaged[version].name})
    return out


def source_of(path: Path) -> str:
    """Where an artefact was read from, for `/api/ready` and `fusers status`."""
    parent = path.resolve().parent
    if parent == RELEASE_DIR.resolve():
        return "release"
    return "configured" if settings.fuser_dir is not None and parent == Path(settings.fuser_dir).resolve() else "data"


# --- review sidecar (D-57) ---------------------------------------------------------------------------------

def review_name(artefact_name: str) -> str:
    return artefact_name.removesuffix(".json") + REVIEW_SUFFIX


def review_problem(doc: Any, sha256: str) -> str | None:
    """None when a sidecar document is a review of the artefact with this sha256; otherwise why it is not."""
    if not isinstance(doc, dict) or doc.get("schema") != REVIEW_SCHEMA:
        return f"not a {REVIEW_SCHEMA} record"
    if doc.get("artefact_sha256") != sha256:
        return f"it names artefact {str(doc.get('artefact_sha256'))[:12]}, not {sha256[:12]}"
    reviewer, statement = doc.get("reviewer"), doc.get("statement")
    if not isinstance(reviewer, str) or not reviewer.strip() or reviewer.strip() == LOOP_REVIEW_FLAG:
        return "it names no reviewer"
    if not isinstance(statement, str) or not statement.strip():
        return "it holds no statement"
    try:
        date.fromisoformat(str(doc.get("date")))
    except ValueError:
        return "its date is not an ISO date"
    return None


def read_review(artefact: Path, sha256: str, fusion_dir: Path | None = None) -> tuple[Review | None, str | None]:
    """The review of an artefact and, when a sidecar exists but does not count, why (the first such reason)."""
    problem: str | None = None
    for directory in dict.fromkeys([*search_dirs(fusion_dir), artefact.parent]):
        path = directory / review_name(artefact.name)
        if not path.is_file():
            continue
        try:
            doc = _read(path).doc
        except (OSError, ValueError) as e:
            problem = problem or f"{path.name} is unreadable: {e}"
            continue
        why = review_problem(doc, sha256)
        if why is None:
            return Review(reviewer=doc["reviewer"].strip(), date=str(doc["date"]), statement=doc["statement"].strip(), path=path), None
        problem = problem or f"{path.name} does not count: {why}"
    return None, problem


def write_review(artefact: Path, sha256: str, *, reviewer: str, statement: str, directory: Path | None = None,
                 today: date | None = None) -> Path:
    """Write the review sidecar of an artefact into `directory` (default `settings.fusion_dir`). The artefact is not touched."""
    doc = {"schema": REVIEW_SCHEMA, "artefact": artefact.name, "artefact_sha256": sha256, "reviewer": reviewer.strip(),
           "date": (today or date.today()).isoformat(), "statement": statement.strip()}
    why = review_problem(doc, sha256)
    if why is not None:
        raise ValueError(f"the review record is not valid: {why}")
    directory = Path(directory or settings.fusion_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / review_name(artefact.name)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def release_candidates(kind: str, listed: list[tuple[str, Path]]) -> tuple[list[tuple[str, Path]], str | None]:
    """The release artefact of a kind (K-3; D-46): `settings.fuser_versions[kind]` when set, else the newest version installed.
    Older artefacts are never served in its place; they stay loadable through an explicit sha256 pin. Returns the candidate list
    and, when the configured version is not installed, the fallback reason."""
    wanted = (settings.fuser_versions or {}).get(kind)
    if wanted is None:
        return listed[:1], None
    release = [(version, path) for version, path in listed if version == wanted]
    return release, None if release else f"the release fuser version {kind} {wanted} (VERITAS_FUSER_VERSIONS) is not installed"


def select_fuser(kind: str, loaded: Mapping[str, str], *, fusion_dir: Path | None = None, pin: str | None = None,
                 face_detector: bool = True, needs_faces: Callable[[str], bool] | None = None,
                 permitted_missing: Mapping[str, str] | None = None) -> Selection:
    """The release artefact for this kind when it is compatible, else the fallback with the reason (`compatibility`). With `pin`
    (a re-run, re-fusion or reproduction), the installed artefact with that sha256, whatever its version."""
    rejected: list[tuple[str, str]] = []
    pinned_why: str | None = None
    listed = candidates(kind, fusion_dir, every=pin is not None)
    if pin is None and listed:
        listed, missing = release_candidates(kind, listed)
        if missing is not None:
            return Selection(kind=kind, mode="fallback", reason=missing, installed=True)
    for version, path in listed:
        try:
            entry = _read(path)
        except (OSError, ValueError) as e:
            rejected.append((path.name, f"unreadable: {e}"))
            continue
        if pin is not None and entry.sha256 != pin:
            continue
        why = compatibility(entry.doc, kind, loaded, face_detector=face_detector, needs_faces=needs_faces,
                            permitted_missing=permitted_missing)
        if why is None:
            try:
                fuser = _fuser(path, entry)
            except (OSError, ValueError, KeyError) as e:
                why = f"does not load: {e}"
            else:
                return Selection(kind=kind, mode="artefact", fuser=fuser, sha256=entry.sha256, path=path, version=version,
                                 reason=f"{path.name} matches the enabled detectors", rejected=tuple(rejected),
                                 intentionally_missing=tuple(d for d in fuser.detectors if d not in loaded and d in (permitted_missing or {})),
                                 review=read_review(path, entry.sha256, fusion_dir)[0])
        rejected.append((path.name, why))
        if pin is not None:
            pinned_why = why
    if pin is not None:
        reason = f"no compatible fuser artefact has sha256 {pin}" + (f": {pinned_why}" if pinned_why else "")
    elif not listed:
        reason = f"no fuser artefact for {kind}"
    else:
        name, why = rejected[0]
        reason = f"the release fuser artefact {name} is not usable: {why}"
    return Selection(kind=kind, mode="fallback", reason=reason, rejected=tuple(rejected), installed=bool(listed))


def select_for_registry(registry: Any, kind: str, profile: Profile | str, *, pin: str | None = None) -> Selection:
    """A fresh run's selection: the detectors and face detectors enabled for this kind and profile."""
    permitted_missing = {}
    if kind == "video" and settings.audio_detector_mode == "candidate" and pin is None:
        from app.bench.score import weights_sha12

        old = registry.manifest.detector("aud.gustking_xlsr")
        if old is not None and registry.decision(old.id).reasons == ("retained for audio rollback; local candidate is selected",):
            permitted_missing[old.id] = weights_sha12(registry.manifest, old.id)
    return select_fuser(kind, loaded_detector_hashes(registry, kind, profile), pin=pin, permitted_missing=permitted_missing,
                        face_detector=face_detector_enabled(registry, kind, profile), needs_faces=needs_faces(registry))


STATUS_KINDS = ("image", "video", "audio", "document")


def review_state(selection: Selection) -> str | None:
    return None if selection.mode != "artefact" else "pending" if selection.pending_review else "reviewed"


def fusion_status(registry: Any, profile: Profile | str) -> dict[str, str]:
    """`fusion: artefact | fallback` per analysis kind with a v2 pipeline, for `/api/ready` (§5.2)."""
    status = {kind: select_for_registry(registry, kind, profile).mode for kind in STATUS_KINDS}
    if settings.audio_detector_mode == "candidate":
        from app.detectors.audio import calibration
        from app.engine.fetchers import installed

        try:
            calibration.load()
            detector = registry.manifest.detector("aud.pella_v2")
            ready = registry.decision(detector.id).enabled and all(
                installed(registry.manifest.model(mid)) for mid in registry.required_models(detector))
            status["audio"] = "artefact" if ready else "fallback"
        except (OSError, ValueError):
            status["audio"] = "fallback"
    return status


def release_artefact(kind: str) -> dict[str, Any] | None:
    """The artefact file the loader serves for a kind when it is compatible (`release_candidates`), whatever is enabled or
    installed: file, version, sha256, directory, source and review state. None when the kind has no artefact file, or the
    configured release version is not installed."""
    listed, _ = release_candidates(kind, candidates(kind))
    if not listed:
        return None
    version, path = listed[0]
    try:
        sha256 = _read(path).sha256
    except (OSError, ValueError) as e:
        return {"file": path.name, "version": version, "sha256": None, "directory": str(path.parent), "source": source_of(path),
                "review": None, "reviewed_by": None, "review_problem": f"unreadable: {e}"}
    review, problem = read_review(path, sha256)
    return {"file": path.name, "version": version, "sha256": sha256, "directory": str(path.parent), "source": source_of(path),
            "review": "reviewed" if review is not None else "pending",
            "reviewed_by": review.reviewer if review is not None else LOOP_REVIEW_FLAG,
            "review_date": review.date if review is not None else None, "review_problem": problem}


def release_bench(kind: str) -> Mapping[str, Any] | None:
    """The bench block of the release artefact of a kind (version and manifest sha256), whatever is enabled or installed;
    None when the kind has no readable artefact file. The fallback's size floor is that bench's (N-7)."""
    listed, _ = release_candidates(kind, candidates(kind))
    if not listed:
        return None
    try:
        return _read(listed[0][1]).doc.get("bench") or None
    except (OSError, ValueError):
        return None


def release_level_rates(kind: str, sha256: str | None) -> dict[str, Any] | None:
    """What an installed artefact of a kind records about its two calibrated levels on the test split (N-17, D-63), for
    the artefact `sha256` names: per level the pooled threshold set on the calibration split, the share of authentic test
    items above it and the share of AI test items above it, with the counts of both classes, the bench version and the view
    of the test set. None for an artefact that is not installed (an earlier model, a fixture), an unreadable file or
    an artefact without such a record: a report then says that the rates are not on record, and prints none."""
    if not sha256:
        return None
    entry = None
    for _, path in candidates(kind, every=True):  # every installed artefact of the kind: a result names its own by sha256
        try:
            read = _read(path)
        except (OSError, ValueError):
            continue
        if read.sha256 == sha256:
            entry = read
            break
    if entry is None:
        return None
    test = (entry.doc.get("metrics") or {}).get("test") or {}
    if not test.get("fpr_at_tau") or not test.get("n0"):
        return None
    levels = {alpha: {"threshold": (test.get("thresholds") or {}).get(alpha), "fpr": test["fpr_at_tau"].get(alpha),
                      "tpr": (test.get("tpr_at_fpr") or {}).get(alpha)}  # `tpr_at_fpr` holds the TPR at the same threshold (bench/report.py)
              for alpha in ("0.05", "0.01")}
    return {"bench_version": test.get("bench_version"), "n_real": int(test["n0"]), "n_fake": int(test.get("n1") or 0),
            "view": (test.get("view") or {}).get("primary"), "levels": levels}


def fuser_status(registry: Any, profile: Profile | str) -> dict[str, dict[str, Any]]:
    """Per kind, what serves under this profile (D-57): `mode`; for an artefact its `version`, `sha256` and `review` state
    (`reviewed` or `pending`); `reason` (why the fallback runs, or which file matched); `artefact`, the kind's release
    artefact file whether or not it can serve here (`release_artefact`); and `shadowed`, the data-directory files a packaged
    artefact is served in place of."""
    out: dict[str, dict[str, Any]] = {}
    for kind in STATUS_KINDS:
        selection = select_for_registry(registry, kind, profile)
        out[kind] = {"mode": selection.mode, "version": selection.version, "sha256": selection.sha256,
                     "review": review_state(selection), "reviewed_by": selection.reviewed_by, "reason": selection.reason,
                     "artefact": release_artefact(kind), "shadowed": shadowed(kind)}
    return out


def shadow_sentence(row: Mapping[str, str | None]) -> str:
    other = f"sha256 {row['sha256'][:12]}" if row["sha256"] else "unreadable"
    return f"the packaged {row['served']} is served in place of {row['file']}, which has other bytes ({other})"


def warn_shadowed() -> list[str]:
    """One warning per artefact file a packaged artefact of the same version is served in place of (every start: D-57).
    Returns the shadowed files."""
    files: list[str] = []
    try:
        for kind in STATUS_KINDS:
            for row in shadowed(kind):
                log.warning("fusion.shadowed_artefact", kind=kind, file=row["file"], sha256=row["sha256"], served=row["served"])
                files.append(str(row["file"]))
    except Exception as e:  # a start is never stopped by this check
        log.error("fusion.status_unavailable", error=str(e))
    return files


def warn_fallbacks(registry: Any, profile: Profile | str) -> list[str]:
    """One warning per kind that has an artefact file and still runs on the uncalibrated fallback, with the reason (a prod
    start calls this: OPS-03). Returns the kinds warned about."""
    warned: list[str] = []
    try:
        status = fuser_status(registry, profile)
    except Exception as e:  # a start is never stopped by this check
        log.error("fusion.status_unavailable", error=str(e))
        return warned
    for kind, row in status.items():
        if row["mode"] == "fallback" and row["artefact"] is not None:
            log.warning("fusion.fallback_at_start", kind=kind, profile=str(getattr(profile, "value", profile)), reason=row["reason"])
            warned.append(kind)
    return warned


def verify_release(directory: Path | None = None) -> list[str]:
    """The packaged release directory against its hash manifest (D-57): one line per difference, empty when it matches."""
    directory = directory or RELEASE_DIR
    try:
        listed = json.loads((directory / RELEASE_MANIFEST).read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError) as e:
        return [f"{RELEASE_MANIFEST} is unreadable: {e}"]
    problems = []
    for row in listed:
        path = directory / Path(str(row["file"])).name
        if not path.is_file():
            problems.append(f"{row['file']} is missing")
            continue
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            problems.append(f"{row['file']} differs from the manifest")
    named = {Path(str(row["file"])).name for row in listed}
    problems += [f"{p.name} is not in the manifest" for p in sorted(directory.iterdir())
                 if p.is_file() and p.name not in named | {RELEASE_MANIFEST} and not p.name.endswith(REVIEW_SUFFIX)]
    return problems


def record_fuser(db: Session, selection: Selection) -> bool:
    """Audit row for an artefact in use (`fusers`, ARCHITECTURE §4.7). The caller commits. True when the row is new (the
    caller then records `fuser_loaded` after its commit)."""
    from app.storage.db import FuserRow, utcnow

    if selection.mode != "artefact" or selection.fuser is None or selection.sha256 is None:
        return False
    row = db.get(FuserRow, selection.sha256)
    if row is not None:
        if row.reviewed_by != selection.reviewed_by:  # D-57: a review recorded (or withdrawn) since the row was written
            row.reviewed_by = selection.reviewed_by
            db.flush()
        return False
    doc = selection.fuser.doc
    db.add(FuserRow(sha256=selection.sha256, modality=str(doc["modality"]), version=str(doc["version"]),
                    bench_version=(doc.get("bench") or {}).get("version"), reviewed_by=selection.reviewed_by,
                    loaded_at=utcnow()))
    db.flush()
    return True
