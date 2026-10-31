"""Veritas probe head (ML §7; D-12; BUILD_SPEC I-13): permissive training pool, format-parity views, feature cache and a
nested leave-one-generator-out logistic probe.

- **Pool (D-12, ML §7.3).** Image fakes from the permissive datasets (Rapidata sets under CDLA-Permissive-2.0,
  bitmind/nano-banana, Qwen-Image-Bench, MLLM-Generated, Co-Spy-Bench, GPT-Image-Edit-1.5M, midjourney-v6-recap,
  OpenSDI_train) and reals from Commons and OpenVid-1M only, each row `commercial_ok` with no NC term in its licence. Only
  `audit_ok` items of the training splits (`fit` by default: `cal` is the fuser's calibration split, `test` the lockbox)
  and without a C2PA manifest (provenance must not leak into pixel weights, ML §4.7).
- **Exclusion (ML §7.3).** Fakes of the held-out LGO fold and every item sharing a `group_key` with one are dropped;
  `check_disjoint` then refuses the pool when a row sits outside the training splits, belongs to the held-out fold, or
  shares a group with an item outside the training splits or in that fold (a manifest whose groups span splits).
- **Views (ML §7.4 item 2).** Both classes get the same augmentation, seeded per (seed, item, view): resize by U[0.5, 1],
  a random square crop of `crop_px` (the short side when smaller), JPEG re-encode at q ∈ [60, 100] 4:2:0. The shortcut
  classifier on the views' (JPEG q, width, height, has_exif) must stay at AUC ≤ 0.65 (`shortcut.MAX_AUC`).
- **Feature cache.** `features/<backbone_id>@<weights_sha12>.parquet`, keyed by (item_id, view), float16 or float32
  bytes plus the view parameters; resumable like the score cache.
- **Probe (ML §7.4 items 1, 4).** Standardised features, L2 logistic regression, C chosen by the lowest mean item-level
  log-loss over inner folds. With ≥ 2 LGO families left in the pool, each inner fold validates on one family's fakes plus
  a group-disjoint share of the other items (nested LGO); otherwise `StratifiedGroupKFold(5)`. Classes weigh the same and,
  within a class, every source weighs the same; an item's views share its weight; an item's logit is its views' mean.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from app.bench import media
from app.bench import shortcut as sc
from app.engine.fusion.features import fp_guarded

POOL_FAKE_DATASETS = {
    "bitmind/nano-banana": "MIT",
    "Qwen/Qwen-Image-Bench": "Apache-2.0",
    "zr-zhang/MLLM-Generated-Image-Detection-Dataset": "Apache-2.0",
    "ruojiruoli/Co-Spy-Bench": "MIT",
    "UCSC-VLAA/GPT-Image-Edit-1.5M": "CC-BY-4.0",
    "Photoroom/midjourney-v6-recap": "MIT",
    "nebula/OpenSDI_train": "CC-BY-SA-4.0",
}
RAPIDATA_PREFIX, RAPIDATA_LICENCE = "Rapidata/", "CDLA-Permissive-2.0"
POOL_REAL_DATASETS = {"wikimedia/commons": "per-file CC0 / CC BY / CC BY-SA / public domain", "nkp37/OpenVid-1M": "CC-BY-4.0"}
TRAIN_SPLITS = ("fit", "cal")
REQUIRED_COLUMNS = ("item_id", "modality", "label", "source_dataset", "source_id", "licence", "commercial_ok", "group_key",
                    "lgo_fold", "split", "audit_ok", "has_c2pa", "native_path")
SCALE_RANGE = (0.5, 1.0)
JPEG_Q_RANGE = (60, 100)
DEFAULT_CROP_PX = 448
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
GROUPED_FOLDS = 5
_NC = re.compile(r"(\bNC\b|-nc\b|-nc-|non-?commercial)", re.IGNORECASE)
_KEY = re.compile(r"[a-z0-9][a-z0-9._-]*@[0-9a-z]+")


class ProbeDataError(ValueError):
    """The training data cannot be used without breaking D-12, the lockbox or the held-out fold (ML §7.3)."""


# --- pool ---------------------------------------------------------------------------------------------------

def pool_licence(row: dict) -> str | None:
    """The D-12 licence under which the row may train the probe, or None."""
    if row.get("modality") != "image" or not row.get("commercial_ok") or _NC.search(str(row.get("licence") or "")):
        return None
    dataset = str(row.get("source_dataset") or "")
    if row["label"] == "real":
        return POOL_REAL_DATASETS.get(dataset)
    if dataset.startswith(RAPIDATA_PREFIX):
        return RAPIDATA_LICENCE if row.get("licence") == RAPIDATA_LICENCE else None
    return POOL_FAKE_DATASETS.get(dataset)


@dataclass
class Pool:
    rows: list[dict]
    lgo_exclude: str | None
    splits: tuple[str, ...]
    dropped: Counter = field(default_factory=Counter)


def select_pool(rows: list[dict], *, lgo_exclude: str | None, splits: Sequence[str] = ("fit",)) -> Pool:
    splits = tuple(splits)
    if not splits or any(s not in TRAIN_SPLITS for s in splits):
        raise ProbeDataError(f"training splits must come from {' / '.join(TRAIN_SPLITS)} (ML §2.6 lockbox), got {list(splits)}")
    if not rows:
        raise ProbeDataError("the manifest is empty")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ProbeDataError(f"manifest columns missing: {', '.join(missing)}")
    if not any(r.get("split") for r in rows):
        raise ProbeDataError("the manifest has no splits: run `make vbench-split` first")
    folds = sorted({r["lgo_fold"] for r in rows if r.get("lgo_fold")})
    if lgo_exclude is not None and lgo_exclude not in folds:
        raise ProbeDataError(f"LGO fold {lgo_exclude!r} is not in the manifest (folds: {', '.join(folds) or 'none'})")
    held_groups = {r["group_key"] for r in rows if lgo_exclude and r.get("lgo_fold") == lgo_exclude}
    kept, dropped = [], Counter()
    for row in sorted(rows, key=lambda r: r["item_id"]):
        if row.get("modality") != "image":
            reason = "modality"
        elif pool_licence(row) is None:
            reason = "not in the D-12 pool"
        elif row.get("split") not in splits:
            reason = f"split {row.get('split')}"
        elif row.get("audit_ok") is not True:
            reason = "label audit"
        elif row.get("has_c2pa"):
            reason = "C2PA manifest"
        elif lgo_exclude and row.get("lgo_fold") == lgo_exclude:
            reason = f"LGO fold {lgo_exclude}"
        elif row["group_key"] in held_groups:
            reason = f"group shared with LGO fold {lgo_exclude}"
        else:
            kept.append(row)
            continue
        dropped[reason] += 1
    check_disjoint(kept, rows, lgo_exclude=lgo_exclude, splits=splits)
    if len({r["label"] for r in kept}) < 2:
        raise ProbeDataError(f"the pool needs real and fake items ({len(kept)} rows: {dict(Counter(r['label'] for r in kept))})")
    return Pool(kept, lgo_exclude, splits, dropped)


def check_disjoint(train: list[dict], manifest: list[dict], *, lgo_exclude: str | None, splits: Sequence[str]) -> None:
    """Refuse training rows that are, or share a group with, items outside the training splits or in the held-out fold."""
    forbidden: dict[str, str] = {}
    for row in manifest:
        if row.get("split") not in splits:
            forbidden.setdefault(row["group_key"], f"split {row.get('split')}")
        elif lgo_exclude and row.get("lgo_fold") == lgo_exclude:
            forbidden.setdefault(row["group_key"], f"LGO fold {lgo_exclude}")
    problems = []
    for row in train:
        item = row["item_id"][:12]
        if row.get("split") not in splits:
            problems.append(f"{item} is in split {row.get('split')}")
        elif lgo_exclude and row.get("lgo_fold") == lgo_exclude:
            problems.append(f"{item} belongs to the held-out fold {lgo_exclude}")
        elif row["group_key"] in forbidden:
            problems.append(f"{item} shares group {row['group_key']} with a {forbidden[row['group_key']]} item")
    if problems:
        raise ProbeDataError(f"{len(problems)} training rows overlap the lockbox or the held-out fold (ML §7.3): "
                             + "; ".join(problems[:5]))


# --- views --------------------------------------------------------------------------------------------------

def view_rng(seed: int, item_id: str, view: int) -> np.random.Generator:
    return np.random.default_rng([seed, int(hashlib.sha256(item_id.encode()).hexdigest()[:15], 16), view])


def view_name(view: int) -> str:
    return f"aug{view}"


def decode_rgb(data: bytes) -> Image.Image:
    """ML N4 decoder: PIL, EXIF orientation applied, 8-bit RGB."""
    return media.to_rgb8(ImageOps.exif_transpose(media.open_image(data)))


def load_rgb(path: Path) -> Image.Image:
    return decode_rgb(path.read_bytes())


def augment_view(im: Image.Image, rng: np.random.Generator, *, crop_px: int) -> tuple[Image.Image, dict]:
    """One class-blind training view: resize, random square crop, JPEG re-encode (draw order: scale, x, y, q)."""
    scale = float(rng.uniform(*SCALE_RANGE))
    w, h = max(1, round(im.width * scale)), max(1, round(im.height * scale))
    resized = im.resize((w, h), Image.Resampling.BICUBIC) if (w, h) != im.size else im
    side = min(crop_px, w, h)
    x, y = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
    q = int(rng.integers(JPEG_Q_RANGE[0], JPEG_Q_RANGE[1] + 1))
    buf = io.BytesIO()
    resized.crop((x, y, x + side, y + side)).save(buf, format="JPEG", quality=q, subsampling=2)
    view = Image.open(io.BytesIO(buf.getvalue()))
    view.load()
    return view.convert("RGB"), {"scale": round(scale, 6), "crop": [x, y, side, side], "jpeg_q": q, "width": side, "height": side}


# --- backbones and the feature cache --------------------------------------------------------------------------

class Backbone(Protocol):
    key: str  # <backbone_id>@<weights_sha12>
    dim: int

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray: ...
    def close(self) -> None: ...


class FakeBackbone:
    """Model-free stand-in for tests and dry runs: the 4 × 4 mean RGB of the view (48 dims)."""

    key = "fake.probe_backbone@noweights"
    dim = 48
    licence = {"spdx": "none", "class": "commercial"}

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        return np.stack([np.asarray(im.convert("RGB").resize((4, 4), Image.Resampling.BOX), dtype=np.float32).reshape(-1) / 255.0
                         for im in images])

    def close(self) -> None:
        pass


def feature_schema():
    import pyarrow as pa

    s, i = pa.string(), pa.int32()
    return pa.schema([("item_id", s), ("view", s), ("status", s), ("reason", s), ("dtype", s), ("dim", i), ("feature", pa.binary()),
                      ("jpeg_q", i), ("width", i), ("height", i), ("params", s), ("runtime_ms", pa.float64()), ("extracted_at", s)])


PE_CORE_KEY_PREFIX = "pe_core.large_p14_336@"


def require_feature_precision(key: str, row: dict) -> None:
    """Old PE-Core fp16/unknown caches cannot inherit the new fp32 parity evidence."""
    if not key.startswith(PE_CORE_KEY_PREFIX) or row.get("status") != "ok":
        return
    try:
        params = json.loads(row.get("params") or "{}")
    except (TypeError, ValueError) as exc:
        raise ProbeDataError(f"{key}: invalid feature precision provenance") from exc
    if row.get("dtype") != "float32" or not isinstance(params, dict) or params.get("inference_dtype") != "float32":
        raise ProbeDataError(f"{key}: PE-Core requires float32 inference and storage with explicit inference_dtype "
                             "provenance; legacy fp16/unknown features are not admitted. Preserve this cache and "
                             "extract into a new --features directory without --fp16; refit and readmit the probe.")


class FeatureCache:
    """Per-backbone feature tables held in memory and rewritten atomically on flush."""

    FINAL = frozenset({"ok", "not_applicable"})

    def __init__(self, directory: Path):
        self.dir = directory
        self._rows: dict[str, dict[tuple[str, str], dict]] = {}
        self._dirty: set[str] = set()

    def path(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise ValueError(f"invalid backbone key {key!r} (expected <backbone_id>@<weights_sha12>)")
        return self.dir / f"{key}.parquet"

    def table(self, key: str) -> dict[tuple[str, str], dict]:
        if key not in self._rows:
            import pyarrow.parquet as pq

            path = self.path(key)
            rows = pq.read_table(path).to_pylist() if path.is_file() else []
            for row in rows:
                require_feature_precision(key, row)
            self._rows[key] = {(r["item_id"], r["view"]): r for r in rows}
        return self._rows[key]

    def done(self, key: str, item_view: tuple[str, str]) -> bool:
        row = self.table(key).get(item_view)
        return bool(row) and row["status"] in self.FINAL

    def put(self, key: str, item_view: tuple[str, str], *, status: str, extracted_at: str, feature: np.ndarray | None = None,
            params: dict | None = None, reason: str | None = None, runtime_ms: float | None = None, fp16: bool = False) -> None:
        params = params or {}
        vec = None if feature is None else np.asarray(feature, dtype=np.float16 if fp16 else np.float32)
        row = {
            "item_id": item_view[0], "view": item_view[1], "status": status, "reason": reason,
            "dtype": None if vec is None else str(vec.dtype), "dim": None if vec is None else int(vec.size),
            "feature": None if vec is None else vec.tobytes(), "jpeg_q": params.get("jpeg_q"), "width": params.get("width"),
            "height": params.get("height"), "params": json.dumps(params, sort_keys=True), "runtime_ms": runtime_ms,
            "extracted_at": extracted_at,
        }
        require_feature_precision(key, row)
        self.table(key)[item_view] = row
        self._dirty.add(key)

    def flush(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for key in sorted(self._dirty):
            path = self.path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            rows = sorted(self.table(key).values(), key=lambda r: (r["item_id"], r["view"]))
            pq.write_table(pa.Table.from_pylist(rows, schema=feature_schema()), tmp)
            os.replace(tmp, path)
        self._dirty.clear()


def feature_vector(row: dict) -> np.ndarray:
    return np.frombuffer(row["feature"], dtype=np.dtype(row["dtype"])).astype(np.float32)


@dataclass
class ExtractSummary:
    extracted: int = 0
    skipped: int = 0
    not_applicable: int = 0


def extract_features(rows: Sequence[dict], vbench_dir: Path, backbone: Backbone, cache: FeatureCache, *, n_views: int,
                     crop_px: int, seed: int, now: str, fp16: bool = False, batch_size: int = 16,
                     checkpoint_every: int = 256, path_key: str = "native_path", content_seeded: bool = False) -> ExtractSummary:
    """Embed `n_views` augmented views per row; resumable (final rows are skipped); undecodable media → not_applicable.

    `path_key` names the row field holding the file (relative to `vbench_dir`, or absolute). Training views are seeded by
    (seed, item_id, view); `content_seeded` seeds them by the sha256 of the file bytes instead (scored views, H-5), so the
    same bytes get the same views whatever the item id, and records it as `content_sha256` in the view parameters."""
    inference_dtype = str(getattr(backbone, "dtype", "unknown")).removeprefix("torch.")
    if backbone.key.startswith(PE_CORE_KEY_PREFIX) and (fp16 or inference_dtype != "float32"):
        raise ProbeDataError("PE-Core extraction requires float32 inference and storage; --fp16 failed device parity")
    summary, batch, since_flush = ExtractSummary(), [], 0

    def run_batch() -> None:
        started = time.perf_counter()
        feats = np.asarray(backbone.embed([view for _, view, _ in batch]), dtype=np.float32)
        if feats.shape != (len(batch), backbone.dim) or not np.isfinite(feats).all():
            raise RuntimeError(f"{backbone.key} returned features of shape {feats.shape} (finite: {bool(np.isfinite(feats).all())})")
        per_view = (time.perf_counter() - started) * 1000.0 / len(batch)
        for (item_view, _, params), vec in zip(batch, feats, strict=True):
            cache.put(backbone.key, item_view, status="ok", feature=vec, params=params, runtime_ms=per_view, fp16=fp16,
                      extracted_at=now)
        summary.extracted += len(batch)
        batch.clear()

    try:
        for row in rows:
            todo = [v for v in range(n_views) if not cache.done(backbone.key, (row["item_id"], view_name(v)))]
            summary.skipped += n_views - len(todo)
            if not todo:
                continue
            try:
                data = (vbench_dir / row[path_key]).read_bytes()
                image = decode_rgb(data)
            except (media.MediaError, OSError) as e:
                for v in todo:
                    cache.put(backbone.key, (row["item_id"], view_name(v)), status="not_applicable", reason=str(e), extracted_at=now)
                summary.not_applicable += len(todo)
                continue
            seed_key = hashlib.sha256(data).hexdigest() if content_seeded else row["item_id"]
            extra = {"content_sha256": seed_key} if content_seeded else {}
            for v in todo:
                view, params = augment_view(image, view_rng(seed, seed_key, v), crop_px=crop_px)
                batch.append(((row["item_id"], view_name(v)), view, {**params, "seed": seed, "crop_px": crop_px, "inference_dtype": inference_dtype, **extra}))
                since_flush += 1
                if len(batch) >= batch_size:
                    run_batch()
            if since_flush >= checkpoint_every:
                cache.flush()
                since_flush = 0
        if batch:
            run_batch()
    finally:
        cache.flush()
    return summary


@dataclass
class Views:
    x: np.ndarray            # (n_views, dim) float32
    item: np.ndarray         # view → index into `items`
    items: list[dict]
    rows: list[dict]         # cache rows, aligned with x
    dropped: Counter


def load_views(pool: Pool, cache: FeatureCache, key: str, *, n_views: int, crop_px: int, seed: int) -> Views:
    """The pool's cached views; items without any usable view are dropped with the reason, missing views refuse."""
    table = cache.table(key)
    xs, item_idx, used_rows, items, dropped, missing = [], [], [], [], Counter(), []
    for row in pool.rows:
        cached = [table.get((row["item_id"], view_name(v))) for v in range(n_views)]
        if any(c is None or c["status"] not in FeatureCache.FINAL for c in cached):
            missing.append(row["item_id"][:12])
            continue
        ok = [c for c in cached if c["status"] == "ok"]
        if not ok:
            dropped["no decodable view"] += 1
            continue
        for c in ok:
            params = json.loads(c["params"])
            if params.get("seed") != seed or params.get("crop_px") != crop_px:
                raise ProbeDataError(f"{key}: views of {row['item_id'][:12]} were built with seed {params.get('seed')} and crop "
                                     f"{params.get('crop_px')}, not seed {seed} and crop {crop_px}")
            xs.append(feature_vector(c))
            item_idx.append(len(items))
            used_rows.append(c)
        items.append(row)
    if missing:
        raise ProbeDataError(f"{len(missing)} pool items have no cached features under {key} (first: {', '.join(missing[:3])}): "
                             "run scripts/probe_features.py first")
    if len({r["label"] for r in items}) < 2:
        raise ProbeDataError("after dropping undecodable items the pool has one class only")
    return Views(np.stack(xs), np.asarray(item_idx), items, used_rows, dropped)


def view_shortcut(views: Views, *, seed: int) -> sc.ShortcutResult:
    """The format-parity shortcut test (ML §7.4 item 2) on what the backbone sees: the views' JPEG q and size."""
    x = np.array([[float(r["jpeg_q"]), float(r["width"]), float(r["height"]), 0.0] for r in views.rows])
    y = [1 if views.items[i]["label"] == "fake" else 0 for i in views.item]
    return sc.shortcut_test(x, y, [views.items[i]["group_key"] for i in views.item], variant="views", seed=seed)


# --- nested LGO probe -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class InnerFold:
    name: str
    train: np.ndarray  # item indices
    val: np.ndarray


def _labels(items: Sequence[dict]) -> np.ndarray:
    return np.array([1 if r["label"] == "fake" else 0 for r in items], dtype=int)


INNER_MODES = ("auto", "grouped")


def inner_folds(items: Sequence[dict], *, seed: int, inner: str = "auto") -> tuple[str, list[InnerFold]]:
    """Nested-LGO inner folds (one per remaining LGO family) or grouped folds; train and validation never share a group.

    `inner="grouped"` always uses `StratifiedGroupKFold(5)` (the D-39 minimal probe: grouped-CV L2, no nested LGO)."""
    from sklearn.model_selection import StratifiedGroupKFold

    if inner not in INNER_MODES:
        raise ValueError(f"inner must be one of {INNER_MODES}, not {inner!r}")
    y, groups = _labels(items), np.array([r["group_key"] for r in items])
    families = sorted({r["lgo_fold"] for r in items if r["label"] == "fake" and r.get("lgo_fold")})
    others = np.array([i for i, r in enumerate(items) if not (r["label"] == "fake" and r.get("lgo_fold"))], dtype=int)
    candidates: list[tuple[str, np.ndarray]] = []
    mode = "grouped"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if (inner == "auto" and len(families) >= 2 and len(set(y[others].tolist())) == 2
                and len(set(groups[others].tolist())) >= len(families)):
            mode = "nested_lgo"
            splitter = StratifiedGroupKFold(len(families), shuffle=True, random_state=seed)
            shares = [others[m] for _, m in splitter.split(others, y[others], groups[others])]
            for family, share in zip(families, shares, strict=True):
                fakes = np.array([i for i, r in enumerate(items) if r["label"] == "fake" and r.get("lgo_fold") == family], dtype=int)
                candidates.append((f"LGO {family}", np.concatenate([fakes, share])))
        elif len(set(groups.tolist())) >= GROUPED_FOLDS:
            splitter = StratifiedGroupKFold(GROUPED_FOLDS, shuffle=True, random_state=seed)
            candidates = [(f"group fold {k}", m) for k, (_, m) in enumerate(splitter.split(np.zeros(len(y)), y, groups))]
    folds = []
    for name, val in candidates:
        val_groups = set(groups[val].tolist())
        train = np.array([i for i in range(len(items)) if groups[i] not in val_groups], dtype=int)
        if len(set(y[train].tolist())) == 2 and len(set(y[val].tolist())) == 2:
            folds.append(InnerFold(name, train, np.sort(val)))
    if not folds:
        raise ProbeDataError("no inner fold has both classes on both sides: the pool is too small for model selection")
    return mode, folds


def sample_weights(items: Sequence[dict]) -> np.ndarray:
    """Item weights: the classes weigh the same and, within a class, every source weighs the same (mean 1)."""
    counts = Counter((r["label"], r["source_id"]) for r in items)
    sources = Counter(label for label, _ in counts)
    w = np.array([1.0 / (sources[r["label"]] * counts[(r["label"], r["source_id"])]) for r in items])
    return w * len(items) / w.sum()


@dataclass
class ProbeFit:
    c: float
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    mode: str
    grid: list[dict]
    folds: list[dict]

    def logits(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):  # Accelerate matmul raises spurious FP flags on finite inputs
            out = ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept
        if not np.isfinite(out).all():
            raise RuntimeError("the probe produced non-finite logits")
        return out


def _item_logits(logit_views: np.ndarray, view_item: np.ndarray, n_items: int) -> np.ndarray:
    sums = np.bincount(view_item, weights=logit_views, minlength=n_items)
    return sums / np.maximum(np.bincount(view_item, minlength=n_items), 1)


@fp_guarded  # Accelerate matmul raises spurious FP flags on finite inputs; the outputs are checked below
def fit_probe(views: Views, *, seed: int, c_grid: Sequence[float] = C_GRID, inner: str = "auto") -> ProbeFit:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    x, view_item, items = views.x.astype(np.float64), views.item, views.items
    y_item, w_item = _labels(items), sample_weights(items)
    per_item = np.bincount(view_item, minlength=len(items))
    y_view, w_view = y_item[view_item], w_item[view_item] / per_item[view_item]
    mode, folds = inner_folds(items, seed=seed, inner=inner)

    def fit(mask: np.ndarray, c: float):
        scaler = StandardScaler().fit(x[mask])
        model = LogisticRegression(C=c, max_iter=5000).fit(scaler.transform(x[mask]), y_view[mask], sample_weight=w_view[mask])
        return scaler, model

    grid, per_fold = [], {}
    for c in c_grid:
        fold_stats = []
        for f in folds:
            scaler, model = fit(np.isin(view_item, f.train), c)
            val_views = np.isin(view_item, f.val)
            logits = _item_logits(np.where(val_views, model.decision_function(scaler.transform(x)), 0.0), view_item, len(items))[f.val]
            p = 1.0 / (1.0 + np.exp(-logits))
            fold_stats.append({"name": f.name, "n_train": int(len(f.train)), "n_val": int(len(f.val)),
                               "n_val_fake": int(y_item[f.val].sum()), "n_val_real": int(len(f.val) - y_item[f.val].sum()),
                               "log_loss": float(log_loss(y_item[f.val], p, sample_weight=w_item[f.val], labels=[0, 1])),
                               "auc": float(roc_auc_score(y_item[f.val], p))})
        per_fold[c] = fold_stats
        grid.append({"c": float(c), "log_loss": float(np.mean([s["log_loss"] for s in fold_stats])),
                     "auc": float(np.mean([s["auc"] for s in fold_stats]))})
    best = min(grid, key=lambda g: (g["log_loss"], g["c"]))
    scaler, model = fit(np.ones(len(x), dtype=bool), best["c"])
    probe = ProbeFit(c=best["c"], coef=model.coef_[0].astype(np.float64), intercept=float(model.intercept_[0]),
                     mean=scaler.mean_.astype(np.float64), scale=scaler.scale_.astype(np.float64), mode=mode, grid=grid,
                     folds=per_fold[best["c"]])
    if not (np.isfinite(probe.coef).all() and np.isfinite(probe.intercept)):
        raise RuntimeError("the probe fit produced non-finite weights")
    return probe


# --- held-out evaluation (D-39 minimal: train on `fit`, evaluate on the group-disjoint `cal` pool) -----------------

def subset_views(views: Views, keep: Sequence[bool]) -> Views:
    """The views of the items where `keep` is true, re-indexed."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(views.items),):
        raise ValueError("keep must have one entry per item")
    new_index = np.cumsum(keep) - 1
    mask = keep[views.item]
    return Views(views.x[mask], new_index[views.item[mask]], [r for r, k in zip(views.items, keep, strict=True) if k],
                 [r for r, m in zip(views.rows, mask, strict=True) if m], Counter(views.dropped))


def item_scores(fit: ProbeFit, views: Views) -> np.ndarray:
    """Item probabilities: the sigmoid of the mean of the item's view logits."""
    return 1.0 / (1.0 + np.exp(-_item_logits(fit.logits(views.x), views.item, len(views.items))))


def evaluate_head(fit: ProbeFit, train: Views, held_out: Views, *, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """Held-out metrics of a head: AUC with a grouped bootstrap interval, TPR / FPR at τ_α from the held-out reals (None below
    the conformal minimum) and TPR at the smallest threshold that flags no held-out real (labelled, not τ_α).

    Refuses held-out items that are, or share a group with, a training item."""
    from app.bench import metrics as mt

    train_ids, train_groups = {r["item_id"] for r in train.items}, {r["group_key"] for r in train.items}
    overlap = [r["item_id"][:12] for r in held_out.items if r["item_id"] in train_ids or r["group_key"] in train_groups]
    if overlap:
        raise ProbeDataError(f"{len(overlap)} held-out items overlap the training items or their groups (first: {', '.join(overlap[:3])})")
    s, y = item_scores(fit, held_out), _labels(held_out.items)
    if len(set(y.tolist())) < 2:
        raise ProbeDataError("the held-out set needs real and fake items")
    boot = mt.GroupedBootstrap([r["group_key"] for r in held_out.items], n_boot=n_boot, seed=seed)
    reals = s[y == 0]
    tau = mt.operational_threshold(reals, alpha)
    zero_fp = float(np.nextafter(reals.max(), np.inf))
    return {"n_real": int((y == 0).sum()), "n_fake": int(y.sum()), "auc": mt.auc(s, y),
            "auc_ci95": boot.ci(lambda idx: mt.auc(s[idx], y[idx])), "alpha": alpha, "tau_alpha": tau,
            "tpr_at_tau": mt.tpr_at(s, y, tau).as_json(), "fpr_at_tau": mt.fpr_at(s, y, tau).as_json(),
            "tau_zero_fp": zero_fp, "tpr_at_zero_fp": mt.tpr_at(s, y, zero_fp).as_json()}


# --- outputs --------------------------------------------------------------------------------------------------

HEAD_FILES = ("model.safetensors", "card.yaml", "training_manifest.jsonl")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def training_manifest(views: Views) -> list[dict]:
    w_item = sample_weights(views.items)
    per_item = np.bincount(views.item, minlength=len(views.items))
    out = []
    for vi, row in zip(views.item, views.rows, strict=True):
        item = views.items[vi]
        out.append({"item_id": item["item_id"], "view": row["view"], "label": item["label"], "source_id": item["source_id"],
                    "source_dataset": item["source_dataset"], "licence": item["licence"], "pool_licence": pool_licence(item),
                    "group_key": item["group_key"], "lgo_fold": item.get("lgo_fold"), "split": item["split"],
                    "weight": round(float(w_item[vi] / per_item[vi]), 9), "params": json.loads(row["params"])})
    return sorted(out, key=lambda r: (r["item_id"], r["view"]))


def build_card(*, backbone_key: str, backbone_licence: dict, pool: Pool, views: Views, fit: ProbeFit,
               shortcut_views: sc.ShortcutResult, shortcut_native: sc.ShortcutResult | None, n_views: int, crop_px: int,
               seed: int, git_commit: str, created_at: str) -> dict:
    sources: dict[str, dict] = defaultdict(lambda: {"dataset": None, "licence": None, "real": 0, "fake": 0})
    for item in views.items:
        entry = sources[item["source_id"]]
        entry["dataset"], entry["licence"] = item["source_dataset"], pool_licence(item)
        entry[item["label"]] += 1
    labels = Counter(r["label"] for r in views.items)
    commercial = backbone_licence.get("class") == "commercial"
    return {
        "id": "veritas.probe_v1",
        "display_name": f"Veritas probe v1 on {backbone_key}",
        "modality": "image",
        "hypothesis": "full_synthesis",
        "backbone": {"key": backbone_key, "licence": backbone_licence,
                     "feature_inference_dtype": "float32" if backbone_key.startswith(PE_CORE_KEY_PREFIX) else "not_enforced"},
        "head": {"kind": "logistic_l2", "c": fit.c, "standardised": True, "aggregation": "mean logit over views",
                 "views": {"n": n_views, "crop_px": crop_px, "scale": list(SCALE_RANGE), "jpeg_q": list(JPEG_Q_RANGE)}},
        "licence": {"class": "commercial" if commercial else "non_commercial", "pool": "permissive (D-12, ML §7.3)",
                    "notice": "trained only on D-12 permissive datasets; CC BY-SA sources carry share-alike terms whose effect "
                              "on weights is undetermined (ML §7.3, U)"},
        "tos_review": "accepted_by_owner_2026-09-12",
        "training_data": {
            "splits": list(pool.splits), "lgo_exclude": pool.lgo_exclude, "n_items": len(views.items), "n_real": labels["real"],
            "n_fake": labels["fake"], "n_groups": len({r["group_key"] for r in views.items}), "n_views": int(len(views.x)),
            "sources": {k: sources[k] for k in sorted(sources)},
            "dropped": dict(sorted((pool.dropped + views.dropped).items())),
            "training_manifest": "training_manifest.jsonl", "training_manifest_sha256": None,
        },
        "controls": {
            "inner_cv": fit.mode, "c_grid": fit.grid, "folds": fit.folds,
            "shortcut": {"max_auc": sc.MAX_AUC, "views_auc": round(shortcut_views.auc, 6), "passed": shortcut_views.passed,
                         "native_auc": None if shortcut_native is None else round(shortcut_native.auc, 6)},
        },
        "weights": {"file": "model.safetensors", "sha256": None},
        "seed": seed, "git_commit": git_commit, "created_at": created_at,
        "admission": {"decision": "pending", "evidence": "docs/v2/bench/admissions/veritas.probe_v1.md"},
        "bench": {"version": None, "filled_by": "vbench_report"},
    }


def write_head(out_dir: Path, fit: ProbeFit, views: Views, card: dict, *, force: bool = False) -> dict:
    """Write model.safetensors, training_manifest.jsonl and card.yaml (with both sha256 values); refuses to overwrite."""
    import yaml
    from safetensors.numpy import save_file

    existing = [name for name in HEAD_FILES if (out_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(f"{out_dir} already holds {', '.join(existing)} (use --force)")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({"linear.weight": fit.coef.astype(np.float32)[None, :], "linear.bias": np.array([fit.intercept], dtype=np.float32),
               "scaler.mean": fit.mean.astype(np.float32), "scaler.scale": fit.scale.astype(np.float32)},
              str(out_dir / "model.safetensors"), metadata={"format": "veritas-probe-v1", "backbone": card["backbone"]["key"]})
    with (out_dir / "training_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in training_manifest(views):
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    card = {**card, "weights": {**card["weights"], "sha256": _sha256(out_dir / "model.safetensors")},
            "training_data": {**card["training_data"], "training_manifest_sha256": _sha256(out_dir / "training_manifest.jsonl")}}
    (out_dir / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return card
