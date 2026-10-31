"""`make vbench-scope` (H-2, D-40): the vb-1.1 scoring scope, a subset manifest for `vbench_score.py --manifest`.

- **Pool.** Rows with role `core` whose label audit did not fail (`audit_ok` not false). The split does not matter: scores are
  cached per item, and the vb-1.1 splits are recomputed by `vbench-split`.
- **Images.** Every real, plus at most 5,000 fakes spread over the families in proportion to availability (largest remainders,
  at least one per family).
- **Video.** At most 350 reals spread evenly over the sources (D-43: every source represented, small sources whole), plus at
  most 300 fakes spread evenly over the families (vendors). The other video reals stay in the manifest, unscored, and are
  listed in the summary. **Audio.** Every row.
- **Robustness.** 600 images of the scope from the `test` split only (J-4, ML audit P1-3: the H-2 subset also held fit, cal
  and wild items), 300 real from R1–R5 and 300 fake, spread evenly over their strata (`perturb.allocate`), flagged in the
  `robustness` column that `perturb.robustness_subset` honours; scored under the 11 D-40 transforms.
- **Order inside a stratum.** sha256("vb-1.1-scope:<item_id>"); robustness sha256("vb-1.1-robustness:<item_id>"). A superset
  manifest whose additions are reals keeps the same fakes, so a scope built on vb-1.0 before the top-up finished scores a prefix
  of the final scope.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

from app.bench import perturb as pt
from app.bench.smoke import allocate_proportional

SCOPE_BENCH = "vb-1.1"
IMAGE_FAKE_CAP = 5000
VIDEO_FAKE_CAP = 300
VIDEO_REAL_CAP = 350  # D-43
ROBUSTNESS_TARGETS = {"real": 300, "fake": 300}
ROBUSTNESS_SPLIT = "test"  # J-4
ROBUSTNESS_TRANSFORMS = ("jpeg75", "jpeg50", "webp80", "resize0.5", "crop70", "social_ig", "social_x", "screenshot_jpeg",
                         "noise3", "upscale2x", "exif_forge")  # D-40
SALT, ROBUSTNESS_SALT = "vb-1.1-scope", "vb-1.1-robustness"


def pool(rows: Iterable[Mapping]) -> list[dict]:
    return [dict(r) for r in rows if (r.get("role") or "core") == "core" and r.get("audit_ok") is not False]


def _ordered(rows: Iterable[dict], salt: str) -> list[dict]:
    return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}:{r['item_id']}".encode()).hexdigest())


def _groups(rows: Iterable[dict], key: Callable[[dict], object]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(key(row) or "unknown")].append(row)
    return out


def _take(groups: Mapping[str, list[dict]], alloc: Mapping[str, int], salt: str) -> list[dict]:
    chosen: list[dict] = []
    for key in sorted(groups):
        chosen += _ordered(groups[key], salt)[:alloc.get(key, 0)]
    return chosen


def robustness_items(rows: Sequence[Mapping], targets: Mapping[str, int] = ROBUSTNESS_TARGETS) -> list[dict]:
    images = [dict(r) for r in rows if r["modality"] == "image" and r.get("split") == ROBUSTNESS_SPLIT]
    reals = _groups((r for r in images if r["label"] == "real" and r.get("real_stratum") in pt.IMAGE_REAL_STRATA),
                    lambda r: r.get("real_stratum"))
    fakes = _groups((r for r in images if r["label"] == "fake"), lambda r: r.get("family"))
    return (_take(reals, pt.allocate({k: len(v) for k, v in reals.items()}, targets["real"]), ROBUSTNESS_SALT)
            + _take(fakes, pt.allocate({k: len(v) for k, v in fakes.items()}, targets["fake"]), ROBUSTNESS_SALT))


def select(rows: Iterable[Mapping], *, image_fake_cap: int = IMAGE_FAKE_CAP, video_fake_cap: int = VIDEO_FAKE_CAP,
           video_real_cap: int = VIDEO_REAL_CAP, robustness_targets: Mapping[str, int] = ROBUSTNESS_TARGETS) -> list[dict]:
    items = pool(rows)
    chosen: list[dict] = []
    for modality in ("image", "video", "audio"):
        kind = [r for r in items if r["modality"] == modality]
        if modality == "audio":
            chosen += kind
            continue
        reals = [r for r in kind if r["label"] == "real"]
        if modality == "video":
            sources = _groups(reals, lambda r: r.get("source_id"))
            reals = _take(sources, pt.allocate({k: len(v) for k, v in sources.items()}, video_real_cap), SALT)
        chosen += reals
        families = _groups((r for r in kind if r["label"] == "fake"), lambda r: r.get("family"))
        counts = {k: len(v) for k, v in families.items()}
        alloc = allocate_proportional(counts, image_fake_cap) if modality == "image" else pt.allocate(counts, video_fake_cap)
        chosen += _take(families, alloc, SALT)
    robust = {r["item_id"] for r in robustness_items(chosen, robustness_targets)}
    return sorted(({**r, "robustness": r["item_id"] in robust} for r in chosen), key=lambda r: (r["modality"], r["item_id"]))


def counts(rows: Sequence[Mapping]) -> dict[str, Counter]:
    return {
        "modality_label": Counter((r["modality"], r["label"]) for r in rows),
        "image_family": Counter(r.get("family") for r in rows if r["modality"] == "image" and r["label"] == "fake"),
        "video_family": Counter(r.get("family") for r in rows if r["modality"] == "video" and r["label"] == "fake"),
        "real_stratum": Counter((r["modality"], r.get("real_stratum")) for r in rows if r["label"] == "real"),
        "robustness": Counter((r["label"], r.get("real_stratum") if r["label"] == "real" else r.get("family"))
                              for r in rows if r.get("robustness")),
    }


def unscored_video_reals(scope: Sequence[Mapping], source: Sequence[Mapping]) -> list[dict]:
    """Pool video reals left out by the D-43 cap: they stay in the manifest, unscored."""
    chosen = {r["item_id"] for r in scope}
    return sorted((r for r in pool(source) if r["modality"] == "video" and r["label"] == "real" and r["item_id"] not in chosen),
                  key=lambda r: (str(r.get("source_id")), r["item_id"]))


def summary_markdown(scope: Sequence[Mapping], source: Sequence[Mapping], *, source_label: str, source_sha256: str,
                     scope_label: str, scope_sha256: str, generated: str) -> str:
    c, s = counts(scope), counts(pool(source))
    lines = [f"# {SCOPE_BENCH} scoring scope", "",
             f"Generated {generated} by `backend/scripts/vbench_scope.py` (H-2, D-40, D-43).", "",
             f"- **Source manifest:** `{source_label}`, sha256 `{source_sha256}`, {len(source)} rows ({len(pool(source))} in the pool).",
             f"- **Scope:** `{scope_label}`, sha256 `{scope_sha256}`, {len(scope)} rows.",
             f"- **Rule:** every image real, ≤ {IMAGE_FAKE_CAP} image fakes proportional over families (≥ 1 each), ≤ "
             f"{VIDEO_REAL_CAP} video reals even over sources (D-43), ≤ {VIDEO_FAKE_CAP} video fakes even over vendors, every "
             f"audio row; robustness {ROBUSTNESS_TARGETS['real']} real (R1–R5) + {ROBUSTNESS_TARGETS['fake']} fake images from "
             f"`{ROBUSTNESS_SPLIT}` only (J-4), even over strata, under {', '.join(ROBUSTNESS_TRANSFORMS)}.", "",
             "## Items", "", "| Modality | Label | Pool | Scope |", "|---|---|---|---|"]
    lines += [f"| {m} | {label} | {s['modality_label'][(m, label)]} | {n} |" for (m, label), n in sorted(c["modality_label"].items())]
    lines += ["", "## Image fake families", "", "| Family | Pool | Scope |", "|---|---|---|"]
    lines += [f"| {f} | {s['image_family'][f]} | {n} |" for f, n in sorted(c["image_family"].items(), key=lambda kv: str(kv[0]))]
    lines += ["", "## Video fake families", "", "| Family | Pool | Scope |", "|---|---|---|"]
    lines += [f"| {f} | {s['video_family'][f]} | {n} |" for f, n in sorted(c["video_family"].items(), key=lambda kv: str(kv[0]))]
    lines += ["", "## Real strata", "", "| Modality | Stratum | Scope |", "|---|---|---|"]
    lines += [f"| {m} | {st} | {n} |" for (m, st), n in sorted(c["real_stratum"].items(), key=lambda kv: tuple(map(str, kv[0])))]
    lines += ["", "## Robustness images", "", "| Label | Stratum or family | Items |", "|---|---|---|"]
    lines += [f"| {label} | {st} | {n} |" for (label, st), n in sorted(c["robustness"].items(), key=lambda kv: tuple(map(str, kv[0])))]
    unscored = unscored_video_reals(scope, source)
    pool_sources = Counter(str(r.get("source_id")) for r in pool(source) if r["modality"] == "video" and r["label"] == "real")
    scope_sources = Counter(str(r.get("source_id")) for r in scope if r["modality"] == "video" and r["label"] == "real")
    lines += ["", "## Unscored video reals (D-43)", "",
              f"{len(unscored)} pool video reals are outside the scope; they stay in the manifest, unscored.", "",
              "| Source | Pool | Scope | Unscored |", "|---|---|---|---|"]
    lines += [f"| {src} | {n} | {scope_sources[src]} | {n - scope_sources[src]} |" for src, n in sorted(pool_sources.items())]
    for src in sorted({str(r.get("source_id")) for r in unscored}):
        ids = [r["item_id"] for r in unscored if str(r.get("source_id")) == src]
        lines += ["", f"- **{src}:** " + ", ".join(f"`{i}`" for i in ids)]
    return "\n".join(lines) + "\n"
