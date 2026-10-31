"""Veritas probe head scores in the benchmark score cache (BUILD_SPEC H-5; D-41; ADR 0006).

The probe must pass the H-4 admission like any detector, and the ablations read the score cache. A head scored on its own
training items, or on a family it was trained on inside an LGO refit, would overstate its fused value, so the rows are
cross-fitted:

- **Scored bytes.** The head scores what the detectors scored: the `score.build_tasks` files (`norm` for image items, `native`
  plus transform for the robustness subset). Each file gets `n_views` class-blind views (`probe.augment_view`) seeded by
  (seed, sha256 of the file bytes, view), cached in `<features>/scored/<variant>-<transform>/<backbone key>.parquet`.
- **Heads.** On the D-12 pool of the training splits (`probe.select_pool`), from the cached training views: the production
  head; one LGO head per family in the pool (the pool without that family and every item sharing its groups); `n_oof`
  out-of-fold heads (`StratifiedGroupKFold` over the pool's groups). Each passes the format-parity shortcut test and is fitted
  by `probe.fit_probe` (nested LGO inner folds with `inner="auto"`).
- **Assignment.** A fake of LGO family F → the LGO-F head, so no fused LGO refit sees a probe trained on F; else an item of
  the production pool → the out-of-fold head of its fold, so no in-sample score reaches the fuser; else → the production
  head. Families without a head (none in the pool) use the production head, which never saw them.
- **Rows.** Detector `img.veritas_probe.<backbone>`, weights sha12 over the backbone key and every head's weights sha256 (a
  refit can change the last bits of near-separable heads through multithreaded BLAS, and with them the key).
  `prob` = sigmoid of the mean view logit; `aux.head`, `aux.views`, `aux.view_logits`; `runtime_ms` = the views' embedding
  time (the logistic head is not timed). A file with no decodable view → `not_applicable`.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.bench import probe as pb
from app.bench import shortcut as sc
from app.bench.score import ScoreCache, ScoreTask

DETECTOR_PREFIX = "img.veritas_probe"
N_OOF = 5


def detector_id(backbone: str) -> str:
    return f"{DETECTOR_PREFIX}.{backbone}"


def scored_cache(features_dir: Path, variant: str, transform: str) -> pb.FeatureCache:
    return pb.FeatureCache(Path(features_dir) / "scored" / f"{variant}-{transform}")


def extract_scored(tasks: Sequence[ScoreTask], backbone: pb.Backbone, features_dir: Path, *, n_views: int, crop_px: int,
                   seed: int, now: str, fp16: bool = False, batch_size: int = 16,
                   checkpoint_every: int = 1024) -> dict[tuple[str, str], pb.ExtractSummary]:
    """Embed the content-seeded views of every task's file, one feature cache per (variant, transform); resumable."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for task in tasks:
        groups.setdefault((task.variant, task.transform), []).append({"item_id": task.item_id, "path": str(task.path)})
    out = {}
    for (variant, transform), rows in sorted(groups.items()):
        out[(variant, transform)] = pb.extract_features(
            rows, Path("."), backbone, scored_cache(features_dir, variant, transform), n_views=n_views, crop_px=crop_px, seed=seed,
            now=now, fp16=fp16, batch_size=batch_size, checkpoint_every=checkpoint_every, path_key="path", content_seeded=True)
    return out


@dataclass
class Head:
    name: str
    pool: pb.Pool
    views: pb.Views
    fit: pb.ProbeFit
    shortcut: sc.ShortcutResult


@dataclass
class Heads:
    heads: dict[str, Head]
    oof_fold: dict[str, int]  # production-pool item → out-of-fold head index
    families: tuple[str, ...]

    def assign(self, row: Mapping) -> str:
        family = row.get("lgo_fold")
        if row.get("label") == "fake" and family in self.families:
            return f"lgo-{family}"
        if row["item_id"] in self.oof_fold:
            return f"oof-{self.oof_fold[row['item_id']]}"
        return "production"


def oof_folds(items: Sequence[dict], *, n_splits: int, seed: int) -> dict[str, int]:
    """Group-disjoint, label-stratified out-of-fold assignment of the production pool's items."""
    import warnings

    from sklearn.model_selection import StratifiedGroupKFold

    groups = np.array([r["group_key"] for r in items])
    if len(set(groups.tolist())) < n_splits:
        raise pb.ProbeDataError(f"the pool has {len(set(groups.tolist()))} groups, fewer than {n_splits} out-of-fold heads")
    y = np.array([1 if r["label"] == "fake" else 0 for r in items])
    folds: dict[str, int] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for k, (_, held) in enumerate(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)):
            folds.update({items[i]["item_id"]: k for i in held})
    return folds


def fit_heads(rows: list[dict], cache: pb.FeatureCache, key: str, *, n_views: int, crop_px: int, seed: int,
              splits: Sequence[str] = ("fit", "cal"), inner: str = "auto", n_oof: int = N_OOF, log=print) -> Heads:
    """The production, LGO and out-of-fold heads; refuses (ProbeDataError) a head whose training views leak the format."""
    load = {"n_views": n_views, "crop_px": crop_px, "seed": seed}
    production_pool = pb.select_pool(rows, lgo_exclude=None, splits=splits)
    production_views = pb.load_views(production_pool, cache, key, **load)
    families = tuple(sorted({r["lgo_fold"] for r in production_views.items if r["label"] == "fake" and r.get("lgo_fold")}))
    oof = oof_folds(production_views.items, n_splits=n_oof, seed=seed)
    plans: list[tuple[str, pb.Pool, pb.Views]] = [("production", production_pool, production_views)]
    for family in families:
        pool = pb.select_pool(rows, lgo_exclude=family, splits=splits)
        plans.append((f"lgo-{family}", pool, pb.load_views(pool, cache, key, **load)))
    for k in range(n_oof):
        keep = [oof[r["item_id"]] != k for r in production_views.items]
        plans.append((f"oof-{k}", production_pool, pb.subset_views(production_views, keep)))
    heads: dict[str, Head] = {}
    for name, pool, views in plans:
        shortcut = pb.view_shortcut(views, seed=seed)
        if not shortcut.passed:
            raise pb.ProbeDataError(f"head {name}: the format-parity shortcut test failed on its training views "
                                    f"(AUC {shortcut.auc:.4f} > {sc.MAX_AUC})")
        fit = pb.fit_probe(views, seed=seed, inner=inner)
        heads[name] = Head(name, pool, views, fit, shortcut)
        log(f"head {name}: {len(views.items)} items, {len(views.x)} views, inner {fit.mode}, C {fit.c:g}, "
            f"inner log-loss {min(g['log_loss'] for g in fit.grid):.4f}")
    return Heads(heads, oof, families)


def weights_sha12(backbone_key: str, head_sha256: Mapping[str, str]) -> str:
    h = hashlib.sha256(backbone_key.encode())
    for name in sorted(head_sha256):
        h.update(f"\n{name}:{head_sha256[name]}".encode())
    return h.hexdigest()[:12]


def missing_views(tasks: Iterable[ScoreTask], features_dir: Path, backbone_key: str, *, n_views: int) -> list[ScoreTask]:
    """The tasks some of whose views are not cached with a final status (run `extract` first)."""
    tables: dict[tuple[str, str], dict] = {}
    missing = []
    for task in tasks:
        group = (task.variant, task.transform)
        if group not in tables:
            tables[group] = scored_cache(features_dir, *group).table(backbone_key)
        cached = [tables[group].get((task.item_id, pb.view_name(v))) for v in range(n_views)]
        if any(c is None or c["status"] not in pb.FeatureCache.FINAL for c in cached):
            missing.append(task)
    return missing


@dataclass
class ScoreSummary:
    ok: int = 0
    not_applicable: int = 0
    missing: int = 0  # tasks without cached views (extract first); no row written

    def as_json(self) -> dict:
        return {"ok": self.ok, "not_applicable": self.not_applicable, "missing": self.missing}


def score_tasks(tasks: Iterable[ScoreTask], rows_by_id: Mapping[str, dict], heads: Heads, features_dir: Path, backbone_key: str,
                cache: ScoreCache, detector_key: str, *, n_views: int, crop_px: int, seed: int,
                now: str) -> tuple[ScoreSummary, dict[str, int]]:
    """Write one score row per task whose views are all cached; returns the summary and the rows per head."""
    caches: dict[tuple[str, str], pb.FeatureCache] = {}
    summary, per_head = ScoreSummary(), {}
    for task in tasks:
        group = (task.variant, task.transform)
        table = caches.setdefault(group, scored_cache(features_dir, *group)).table(backbone_key)
        cached = [table.get((task.item_id, pb.view_name(v))) for v in range(n_views)]
        if any(c is None or c["status"] not in pb.FeatureCache.FINAL for c in cached):
            summary.missing += 1
            continue
        ok = [c for c in cached if c["status"] == "ok"]
        if not ok:
            reason = next((c["reason"] for c in cached if c["reason"]), "no decodable view")
            cache.put(detector_key, task.key, {"status": "not_applicable", "reason": reason}, scored_at=now)
            summary.not_applicable += 1
            continue
        for c in ok:
            params = json.loads(c["params"])
            if params.get("seed") != seed or params.get("crop_px") != crop_px or "content_sha256" not in params:
                raise pb.ProbeDataError(f"scored views of {task.item_id[:12]} ({task.variant}/{task.transform}) were not built with "
                                        f"content seeding, seed {seed} and crop {crop_px}")
        name = heads.assign(rows_by_id[task.item_id])
        logits = heads.heads[name].fit.logits(np.stack([pb.feature_vector(c) for c in ok]))
        mean = float(np.mean(logits))
        runtime = [c["runtime_ms"] for c in ok if c["runtime_ms"] is not None]
        cache.put(detector_key, task.key, {
            "status": "ok", "prob": float(1.0 / (1.0 + np.exp(-mean))),
            "aux": {"head": name, "views": len(ok),
                    "feature_inference_dtype": "float32" if backbone_key.startswith(pb.PE_CORE_KEY_PREFIX) else "not_enforced",
                    "view_logits": [round(float(v), 6) for v in logits]},
            "runtime_ms": float(sum(runtime)) if runtime else None}, scored_at=now)
        summary.ok += 1
        per_head[name] = per_head.get(name, 0) + 1
    return summary, dict(sorted(per_head.items()))
