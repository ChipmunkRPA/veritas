# Admission record: veritas.probe_v1 (L2 logistic probe on CLIP ViT-L/14 features)

BUILD_SPEC I-13 at D-39 minimal scope; criteria ML §3.2 A1–A6. Bench `vb-1.0-core` (D-39), manifest sha256
`d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`. 2026-09-14. Automated record (ralph loop), pending human review.

**Decision: not admitted on this record.** The head separates the held-out permissive `cal` items well (AUC 0.9903), but A3a,
A3b, A4 and A5 are not established: the head has no detector adapter, and features exist only for the permissive pool, so it
cannot be scored on the other `fit` ∪ `cal` images that `calibrate --ablate` needs. The held-out set is also drawn from the same
sources as the training pool, so the AUC does not show generalisation to other real sources (see "Reading the numbers"). The
head is not in `models.yaml` and does not feed the image fuser (I-14).

## What was built

| Step | Command and output |
|---|---|
| Features | `probe_features.py --backbone clip_l14 --manifest data/vbench/core/manifest.parquet --splits fit,cal --views 3 --seed 7` (job `i13-features`): 330 pool items × 3 class-blind views (scale 0.5–1.0, 448 px crop, JPEG q 60–100), 990 extracted, 0 not applicable → `clip.vit_l14_openai@5d438d2337cf.parquet` |
| Production head | `train_probe.py --backbone clip_l14 --manifest data/vbench/core/manifest.parquet --lgo-exclude none --splits fit --inner grouped --views 3 --seed 7` → `data/weights/veritas-probe-v1/`: 240 items (100 real, 140 fake, 131 groups), 720 views; C 1 by grouped 5-fold CV (inner log-loss 0.2392, inner AUC 0.9958); weights sha256 `0543444ad288…`, training manifest sha256 `45b1a113f0c2…`; card `tos_review: accepted_by_owner_2026-09-12`, admission pending |
| Held-out evaluation | `eval_probe.py --backbone clip_l14 --manifest data/vbench/core/manifest.parquet --views 3 --seed 7 --bootstrap 2000 --scores data/vbench/scores --reference img.hf.ateeqq --reference img.commfor384 --reference img.dda` (job `i13-eval`) → `probe-eval-clip_l14.json` in this directory |

D-39 minimal: CLIP ViT-L/14 only; L2 chosen by grouped CV (`--inner grouped`), no nested LGO; heads train on `fit` only and are
evaluated on the permissive pool of `cal`, which the split assignment keeps group-disjoint from `fit` (`probe.evaluate_head`
refuses any shared item or group). `test` was not touched.

## Criteria

| Criterion | Result | Evidence |
|---|---|---|
| A1 licence and parity | licence pass; parity not established | backbone `clip.vit_l14_openai` class commercial, SPDX Apache-2.0 (timm card; weights from openai/CLIP, MIT); training pool D-12 permissive only (Rapidata CDLA-Permissive-2.0, Qwen-Image-Bench Apache-2.0, nano-banana MIT, midjourney-v6-recap MIT, MLLM-Generated Apache-2.0; reals Wikimedia Commons per-file CC0 / CC BY / CC BY-SA / public domain). CC BY-SA share-alike effect on weights undetermined (ML §7.3). No CPU vs MPS feature parity run |
| A2 stand-alone AUC | **pass on the permissive cal pool**: 0.9903, 95 % CI [0.9751, 1.0] | 17 reals, 73 fakes held out; grouped bootstrap B 2000. References on the same 90 items: img.hf.ateeqq 0.7607, img.commfor384 0.7284, img.dda 0.7929 |
| A3a marginal fused value | not established | no detector adapter: the head is not in the score cache for the non-pool `fit` ∪ `cal` images |
| A3b coverage gap | not established | as A3a |
| A4 no harm to reals | not established | as A3a; τ_0.05 is not established from 17 held-out reals (the conformal rank needs n ≥ 19) |
| A5 robustness | not established | the robustness transforms were not embedded |
| A6 latency | per view p50 30.7 ms, p95 32.9 ms, max 35.0 ms | cached `runtime_ms` of the 990 feature rows (MPS, float32, batches of 16, warm); 3 views per image; the logistic head runs on CPU and was not timed |

## Leave-generator-out refits

Each refit drops that family and every item sharing its groups from the `fit` pool (`select_pool(lgo_exclude=F)`), and is
evaluated on the 17 `cal` reals plus that family's `cal` fakes.

| Fold | Train items | Held-out fakes | AUC [95 % CI] | Flagged at zero held-out FP | Ateeqq / CommFor / DDA AUC on the same items |
|---|---|---|---|---|---|
| F1 midjourney | 222 | 5 | 0.9647 [0.8442, 1.0] | 4 of 5 | 0.6588 / 1.0 / 1.0 |
| F2 flux2 | 227 | 4 | 1.0 [1.0, 1.0] | 4 of 4 | 0.6324 / 0.6618 / 0.5 |
| F3 gpt-image-1.5-2 | 232 | 6 | 0.9412 [0.8333, 1.0] | 3 of 6 | 0.6961 / 0.5294 / 0.6176 |
| F4 nano-banana | 225 | 4 | 1.0 [1.0, 1.0] | 4 of 4 | 0.7647 / 0.3971 / 0.7353 |
| F5 seedream4 | 222 | 5 | 0.9765 [0.8810, 1.0] | 4 of 5 | 0.7882 / 1.0 / 0.7882 |
| F6 qwen-image | 218 | 7 | 1.0 [1.0, 1.0] | 7 of 7 | 0.7647 / 0.7563 / 0.6975 |
| F7 imagen4 | 233 | 7 | 1.0 [1.0, 1.0] | 7 of 7 | 0.7479 / 0.6555 / 0.8571 |

LGO macro over F1–F7: AUC 0.9832; share flagged at zero held-out FP 0.8714. Each fold holds 4–7 fakes, so a single item moves
the fold's recall by 14–25 pts.

## Reading the numbers

- **Same-source held-out set.** Held-out reals are 17 Commons photos (R1 camera 9, R2 smartphone 7, R6 hard negatives 1) from the
  sources the head trains on, and held-out fakes come from the same permissive datasets. The core `cal` split's other real strata
  (R3, R4, R5, screenshots) are not in the D-12 pool, so they were not evaluated.
- **Format shortcut.** On the training views the shortcut classifier (JPEG q, width, height, has_exif) stays at AUC 0.4574
  (limit 0.65), so the augmentation removes the format cue from what the backbone sees. On the native files of the same
  training items it reaches AUC 0.9975 (diagnostic), so the two classes differ strongly in format and source. Content-level
  source cues that survive augmentation are not ruled out.
- **Reference detectors** were measured on the same items but are not trained on this pool; the comparison shows the head
  fits this pool, not that it outperforms them on other sources.

## To admit later

A detector adapter for the head (features at inference, 3 views or a single native crop), scoring through `vbench-score` on
core, then `calibrate --ablate veritas.probe_v1`, `admission_evidence.py` and A1 parity.
