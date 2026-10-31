# H-6 partial-edit lane on vb-1.1 (BUILD_SPEC H-6)

Bench `vb-1.1`, manifest sha256 `8314119e487a…`, scope `data/vbench/scope/vb-1.1.parquet` (sha256 `5059434a74cf…`),
2026-09-15. Automated record (ralph loop), pending human review.

**Summary.**
- **G8 on vb-1.1 test** (partial edits TPR@5%FPR, target ≥ 50 %): **86.3 %** (132 / 153, Wilson 79.9–90.8 %) for an interim
  fuser over the H-4 admitted set. With the four derived partial-edit features it is 87.6 % (134 / 153). Their gain of two
  items is inside the interval, and the same fuser's test FPR at τ_0.05 rises from 5.7 % to 6.5 %.
- **Derived features rejected as union-fuser features:** A2 0.50–0.65, A3a ≤ +1.2 pts, A3b fail (D-44 records in
  `admissions/vb-1.1-admissions.md`). They stay computable at runtime (`derived.with_derived`) and are not fused in v2.1.
- **Localiser false positives:** on the fuser's scored bytes (`norm`), `loc.mesorch` reports ≥ 1 region on 35.7 % of fit ∪ cal
  reals and on 41.6 % of edits. Its integrity score does not separate edits from reals (AUC 0.533).
- **Reading.** The v2.0 fuser measured G8 12.5 % (5 / 40) on vb-1.0-core. The v2.1 number differs in detector set (the
  admitted learned detectors `img.commfor384`, `img.dda`) and in test population (153 edits); no localiser feature contributes.

## What was run

| Step | Command | Output |
|---|---|---|
| Metrics on old rows | job `h6-rescore` (H-2 queue, 2026-09-15 01:30–03:43Z): `vbench_score.py --manifest data/vbench/scope/vb-1.1.parquet --variant norm --subset all --rescore-missing-metrics loc.mesorch,img.forensic.jpeg_dq,img.forensic.copymove`, then `--variant native --subset robustness` | norm: 9,891 tasks, 1,505 rescored, 0 failed; native: 600 skipped (metrics present) |
| Derived features | `derive_features.py --scores data/vbench/scores` | `pe.mesorch.integrity@c9c8c6fde826` and `pe.mesorch.region_area@…` 17,050 ok; `pe.jpeg_dq.support@noweights` 13,645 ok; `pe.copymove.pairs@noweights` 17,047 ok |
| A2, A5, A6 | `admission_evidence.py --manifest data/vbench/manifest.parquet --scope data/vbench/scope/vb-1.1.parquet --scores data/vbench/scores --detector pe.mesorch.integrity --detector pe.mesorch.region_area --detector pe.jpeg_dq.support --detector pe.copymove.pairs --threats partial_edit --out data/fusion/h6-evidence` | `admissions/vb-1.1/admission-evidence-pe.*.json` |
| A3a, A3b, A4 | job `h6-ablate2`: `bash scripts/h6_ablate.sh` (baseline `img.commfor384,img.dda,img.dfdc_b7`, gate `img.dfdc_b7=has_face`, `--context scored --scope data/vbench/scope/vb-1.1.parquet`, B 2,000) | `admissions/vb-1.1/ablation-pe.*.md` |
| G8 | job `h6-g8`: `bash scripts/h6_g8.sh` (two interim fits `2.1.0-h6.base` and `2.1.0-h6.pe`, then `vbench_report.py --eval-split test --scope …`) | `data/fusion/h6-report-{base,pe}/results.json`; artefact sha256 `1d0537f040e4…` and `79a3aa5a8eac…`, both logged in `test_access.log` |
| Localiser rates | `localiser_rates.py --manifest data/vbench/manifest.parquet --scores data/vbench/scores --detector loc.mesorch --scope data/vbench/scope/vb-1.1.parquet --out data/fusion/h6-localiser` | table below (fit ∪ cal; `test` is read only through the report) |

The interim artefacts live in `data/fusion/h6-fusers/`, which the API loader does not read. H-7 fits the v2.1 fuser.

## G8 on vb-1.1 test (scope items; τ from the fuser's `cal` reals, equal stratum weights)

Edits are So-Fake-OOD TAMPERED, OpenSDI (SD1.5, SD2, SD3, SDXL, FLUX) and Semi-Truths p2p; the report pools them.

| System | Detectors | Edit AUC [95 % CI] | TPR@1%FPR | **TPR@5%FPR** (k / n; Wilson) | Test FPR at τ_0.05 (k / n) |
|---|---|---|---|---|---|
| base | commfor384, dda, dfdc_b7 (has_face) | 0.9664 [0.9489, 0.9812] | 75.8 % (116 / 153) | **86.3 %** (132 / 153; 79.9–90.8 %) | 5.7 % (122 / 2,152) |
| pe | base + 4 derived features | 0.9681 [0.9515, 0.9824] | 73.9 % (113 / 153) | **87.6 %** (134 / 153; 81.4–91.9 %) | 6.5 % (139 / 2,152) |

Test FPR at τ_0.05 per real stratum (k / n), base → pe: R1 42 / 387 → 42 / 387; R2 0 / 409 → 0 / 409; R3 61 / 479 → 68 / 479;
R4 2 / 405 → 3 / 405; R5 15 / 450 → 24 / 450; R6 2 / 22 → 2 / 22. R1 and R3 above 5 % are G3 findings for H-7.

## Derived features as union-fuser candidates (vb-1.1 fit ∪ cal, scope items: 5,001 items, 2,697 real, 2,304 fake)

| Feature | A2 AUC edits vs reals [95 % CI] | A3a Δ LGO macro [CI] | Edit recall out-of-fold, without → with | A4 pooled upper bound | Decision |
|---|---|---|---|---|---|
| `pe.mesorch.integrity` | 0.5334 [0.4849, 0.5837] | +1.2 pts [−0.1, +2.5] | 84.5 % → 86.3 % (+1.9 [+0.0, +4.3]) | +0.41 | rejected (A2, A3) |
| `pe.mesorch.region_area` | 0.5421 [0.5005, 0.5869] | +0.8 pts [−0.1, +1.9] | 84.5 % → 85.1 % (+0.6 [−1.2, +3.1]) | +0.42 | rejected (A2, A3) |
| `pe.jpeg_dq.support` | 0.6476 [0.5975, 0.6951] | +0.0 pts [+0.0, +0.0] | 84.5 % → 84.5 % | +0.00 | rejected (A2, A3) |
| `pe.copymove.pairs` | 0.4955 [0.4929, 0.4978] | +0.1 pts [+0.0, +0.4] | 84.5 % → 84.5 % | +0.11 | rejected (A2, A3) |

- **Context gating (D-41).** No gate was applied. A feature whose source does not apply (no JPEG quantisation, no map) is
  missing (m_j = 1), and the fuser learns that column. None of the features passed, so no gate was fitted.
- **A5** is not established for these features: the robustness subset holds 3 edits.
- **A6** is the source detector's call: p95 1,004 ms (Mesorch), 418 ms (double JPEG), 493 ms (copy-move).
- **Missingness leak (corrected).** The first ablation (`h6-ablate`, history `data/fusion/h6-ablation-unscoped/`) read every
  scored fit ∪ cal item. That included 200–250 fakes of the interim scoring pass, outside the vb-1.1 scope, whose rows predate
  the metric capture. Every real had the statistic, so the missing indicator encoded the label: A3a came out +2.2 to +3.0 pts
  for features with A2 ≈ 0.5. Since then `calibrate --scope` and `vbench_report --scope` restrict items to the scope.

## Localiser behaviour: `loc.mesorch` 0.4.0 on norm bytes, fit ∪ cal scope items

Readings as I-10: ≥ 1 region; integrity read as a probability ≥ 2/3 (weak or stronger) and ≥ 10/11 (≥ moderate). Counts with
95 % Wilson intervals.

| Reals | n | ≥ 1 region | integrity ≥ 2/3 | integrity ≥ 10/11 |
|---|---|---|---|---|
| all | 2,697 | 963 (35.7 %; 33.9–37.5 %) | 578 (21.4 %; 19.9–23.0 %) | 249 (9.2 %; 8.2–10.4 %) |
| R1 camera originals | 424 | 148 (34.9 %; 30.5–39.6 %) | 105 (24.8 %; 20.9–29.1 %) | 41 (9.7 %; 7.2–12.9 %) |
| R2 smartphone | 612 | 194 (31.7 %; 28.1–35.5 %) | 110 (18.0 %; 15.1–21.2 %) | 49 (8.0 %; 6.1–10.4 %) |
| R3 social | 589 | 176 (29.9 %; 26.3–33.7 %) | 105 (17.8 %; 14.9–21.1 %) | 50 (8.5 %; 6.5–11.0 %) |
| R4 web / stock | 495 | 206 (41.6 %; 37.4–46.0 %) | 117 (23.6 %; 20.1–27.6 %) | 51 (10.3 %; 7.9–13.3 %) |
| R5 faces | 549 | 230 (41.9 %; 37.8–46.1 %) | 136 (24.8 %; 21.3–28.6 %) | 57 (10.4 %; 8.1–13.2 %) |
| R6 hard negatives | 28 | 9 (32.1 %; 17.9–50.7 %) | 5 (17.9 %; 7.9–35.6 %) | 1 (3.6 %; 0.6–17.7 %) |

| Edits | n | ≥ 1 region | integrity ≥ 2/3 | integrity ≥ 10/11 |
|---|---|---|---|---|
| all | 161 | 67 (41.6 %; 34.3–49.3 %) | 50 (31.1 %; 24.4–38.6 %) | 21 (13.0 %; 8.7–19.1 %) |
| OpenSDI FLUX | 22 | 6 (27.3 %) | 4 (18.2 %) | 1 (4.5 %) |
| OpenSDI SD1.5 | 19 | 11 (57.9 %) | 8 (42.1 %) | 2 (10.5 %) |
| OpenSDI SD2 | 20 | 4 (20.0 %) | 3 (15.0 %) | 0 (0.0 %) |
| OpenSDI SD3 | 23 | 11 (47.8 %) | 7 (30.4 %) | 2 (8.7 %) |
| OpenSDI SDXL | 18 | 9 (50.0 %) | 6 (33.3 %) | 4 (22.2 %) |
| Semi-Truths p2p | 39 | 20 (51.3 %) | 17 (43.6 %) | 10 (25.6 %) |
| So-Fake-OOD TAMPERED | 20 | 6 (30.0 %) | 5 (25.0 %) | 2 (10.0 %) |

Integrity AUC, edits vs reals: 0.5334. For comparison, I-10 measured on native camera originals of partial vb-1.0 (427 items):
27.9 % with ≥ 1 region, 7.0 % integrity ≥ 10/11 (`docs/v2/qa/i10_localiser_fpr.md`). The norm re-encode (JPEG, no EXIF)
is not the input the localiser was trained for (card: classical splicing, copy-move and removal edits); native bytes of the
edits were not measured here.

## Consequences

- **H-7** fits the v2.1 image fuser without derived features. The partial-edit threat is covered by the admitted detectors
  (G8 recorded above), and `h2_partial_edit` stays a sub-fuser over the same feature space.
- `loc.mesorch` stays default off (card; profiles `full`, `deep`) and outside the whole-image verdict; its maps remain an
  explanation layer. A
  region finding must not be read as evidence of an edit on its own: the method flags a third of real photographs on these
  bytes.
- The derived-feature statistics are captured in the score cache and at runtime. They can be re-evaluated on native bytes
  when the robustness subset holds enough edits (3 today).
