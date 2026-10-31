# Image fuser `fuser-image-2.1.1` on vb-1.1 (BUILD_SPEC J-5)

Bench `vb-1.1`, manifest sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`, scope
`data/vbench/scope/vb-1.1.parquet` (sha256 `2041bf678721…`, J-4). 2026-09-17. Automated record (ralph loop),
`reviewed_by: ralph-loop (automated; pending human review)` (D-17). Measured on `test` with 95 % intervals. The goal table FAILS
(`../gate-v2.1.1.txt`: 26 PASS, 26 FAIL, 6 MISSING of 58 checks); every FAIL or MISSING goal is in `../shortfalls.md`. It supersedes
`fuser-image-2.1.0.md`.

## Decisions

- **Fitted set:** the J-3c1 end set under D-53 (`v2.1.1-admissions.json`, `elimination.detectors` / `elimination.gates`), read by
  `backend/scripts/j5_fit.sh` from the record:
  - `img.dda`, ungated;
  - `img.dfdc_b7`, gated `has_face` (its H-3 gate);
  - `img.hf.ateeqq`, gated `!has_face` (D-53 step 2 retry; both ablations in `v2.1.1c1-step2/` and `v2.1.1c1-step2-gated/`);
  - `img.hf.dima806`, ungated (admitted on A3b only).
- **Fit policy:** D-45 with the D-50 residuals: no additive size or face terms, one effective Platt intercept, resolution-balanced
  weights, `img.dda` size-driven missing fixed at 0, `img.dfdc_b7` face-driven missing behind its `has_face` gate. The fit dropped
  `has_face`, `face_px:120-256`, `face_px>=256`, `min_side<512`, `min_side>1024` (`size_face`) and `jpeg_q>=90` (`constant_on_fit`);
  `has_face` stays an interaction group. No additive context term remains.
- **Default profile** (`backend/app/models.yaml`): `img.hf.ateeqq` and `img.hf.dima806` are now `default_enabled: true` in `full`
  and `deep`; `img.commfor384` (rejected at D-53 step 3) is off and its `served_until` is cleared (D-51). The loader serves
  `fuser-image-2.1.1.json` under `full` and `deep` ("matches the enabled detectors"); 2.1.0 is refused ("detector img.commfor384 is
  not enabled"). `quick` falls back, because `img.dda` does not run in `quick`.
- **Video and audio** (gate 2b check): the served `fuser-video-2.1.0` read a context-only AUC of 0.8160 on `cal` (limit 0.55), so video
  fits use `policy.VIDEO_2B` (no additive size or face terms, one effective intercept; no resolution balance, which is defined on
  image min-side buckets). `fuser-audio-2.1.0` read 0.5000 and was refitted without a policy change, because both 2.1.0 artefacts
  record the H-2 scope sha that the gate refuses against the J-4 scope. `policy.VIDEO_2B` is a post-hoc rule change under D-45 (a)
  (`../shortfalls.md`).
- **Context of the scored bytes** (ADR 0005, `--context scored`) and the scope items only (`--scope`).

## Commands

| Step | Job and command |
|---|---|
| Diagnostic gate before the refit | `j5-gate-pre2`: `shortcut_gate.py … --scope data/vbench/history/vb-1.1/scope-h2-5059434a74cf.parquet --record data/fusion/j5-pre/shortcut_gate.json` (served 2.1.0: structure FAIL, 2b image 0.7897, 2b video 0.8160, 2b audio 0.5000) |
| Fit, LGO refits, two reproducibility fits | `j5-chain` at ac9ca5c: `bash scripts/j5_fit.sh`, i.e. `calibrate.py --manifest data/vbench/manifest.parquet --scores data/vbench/scores --modality image --seed 7 --bench vb-1.1 --context scored --scope data/vbench/scope/vb-1.1.parquet --detectors img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 --gates "img.dfdc_b7=has_face;img.hf.ateeqq=!has_face"` with `--version 2.1.1` (one `--exclude-folds F<k>` refit per fold) |
| Video and audio refits | `j5-chain`: `bash scripts/j5_refit_av.sh` (video detectors of H-8, one `--exclude-folds` refit per vendor; audio `aud.gustking_xlsr`) |
| dima806 robustness rows | `j5-score-dima806`: `bash scripts/j5_score_dima806.sh` (`vbench-score` on the robustness subset, none and 11 transforms, with every other image detector denied) |
| Report and goal table | `j5-report2`: `bash scripts/j5_report.sh`, i.e. `vbench_report.py … --artefact data/fusion/fuser-image-2.1.1.json --lgo-artefacts data/fusion/lgo-image-2.1.1 --video-artefact data/fusion/fuser-video-2.1.1.json --video-lgo-artefacts data/fusion/lgo-video-2.1.1 --audio-artefact data/fusion/fuser-audio-2.1.1.json --eval-split test --bench vb-1.1 --scope data/vbench/scope/vb-1.1.parquet --repro <repro> <repro2> --latency ../docs/v2/bench/perf.json --write-artefact --update-cards app/models.yaml`, then `vbench_gate.py` |
| Shortcut gate | `j5-shortcut-gate`: `shortcut_gate.py --manifest data/vbench/manifest.parquet --scores data/vbench/scores --scope data/vbench/scope/vb-1.1.parquet` → `../shortcut_gate.json` |

Two report runs are logged in `test_access.log`. The first (`j5-chain`, 11:23Z) left every G5, G6c and G11 row MISSING: the J-4
robustness pass ran without `img.hf.dima806` ("no cached rows for img.hf.dima806"). After `j5-score-dima806` the second run
(`j5-report2`, 11:54Z) is the record. The fitted coefficients did not change between the runs (each run rewrites only
`metrics.test` and `metrics.lgo`); nothing was tuned on `test`.

**Artefact sha256.** `results.json` names the sha256 the report read (`c4303453c7bb…`, after the first run's metrics write). After
the second write, job `j5-report2` logged: image `746ada66f73d77c3ca948eb86d38e3f422ac3e6843ccfbab64f807b065ad153a`, video
`fbbccc434f1af6a9e1fb64ccfddbbc1dc70a01f1f4fab9457b49e699f7ffcb45`, audio
`44b63c7c8029103cf1b049c0e3548e6c2a39a027d95e39119145b83bc71990d3` (pre-write before the first run: image `0f52c35aa8c1…`, video
`dbfc655961eb…`, audio `f5185662545d…`). `metrics.test.view` and `metrics.test.full` live with `metrics.test`; the runtime reads the
view marker from there (`runtime._calibration`, `test_view: size_matched`).

## Fit

- `fit_n` 3,300, `cal_n` 1,695 (fit ∪ cal scope items; C2PA items and 6 items where a selected detector did not run excluded); 4
  detectors, 0 skipped. git `ac9ca5c`.
- λ 3.0 (5-fold grouped CV log-loss 0.3794); cal AUC 0.9206 (in-sample); d95 2.446.
- Sub-fusers `h1_synthesis`, `h2_face_swap`, `h2_partial_edit`; `voice` skipped (fewer than 10 items of a class).
- LGO refits in `data/fusion/lgo-image-2.1.1/` (fit_n 3,209–3,263). Reproducibility: the two refits are identical except
  `created_at` (G12 PASS).

## Results on `test`

Size-matched view (primary, D-45): 958 reals, 820 fakes; full view (secondary): 2,152 reals, 1,843 fakes. τ_0.05 0.7150, τ_0.01 0.9607
from 850 `cal` reals.

| Set | AUC (95 % CI) | TPR@5 %FPR (k / n; Wilson) | FPR at τ_0.05 | ECE | Brier |
|---|---|---|---|---|---|
| `image/test` | 0.9054 (0.8905–0.9192) | 71.5 % (586 / 820; 68.3–74.4 %) | 7.6 % (73 / 958) | 0.0537 | 0.1270 |
| `image/test/full` | 0.9014 (0.8900–0.9122) | 72.2 % (1,330 / 1,843; 70.1–74.2 %) | 7.5 % (162 / 2,152) | 0.0519 | 0.1281 |
| t2i | 0.9426 (0.9281–0.9549) | 78.4 % (337 / 430; 74.2–82.0 %) | 7.6 % (56 / 733) | 0.0398 | 0.0982 |
| edit | 0.9168 (0.8642–0.9616) | 78.4 % (40 / 51; 65.4–87.5 %) | 6.5 % (47 / 724) | 0.0837 | 0.1139 |
| face swap vs R5 | 0.7819 (0.7384–0.8217) | 51.4 % (95 / 185; 44.2–58.5 %) | 6.6 % (26 / 395) | 0.1482 | 0.2075 |
| GAN face vs R5 | 0.8662 (0.8226–0.9073) | 63.0 % (58 / 92; 52.8–72.2 %) | 7.1 % (24 / 338) | 0.1232 | 0.1632 |

- **LGO** (refits without F, size-matched folds): F1 81.2 % (32 fakes), F2 40.0 % (15), F3 25.0 % (28), F4 64.7 % (34), F5 83.3 %
  (66), F6 42.1 % (19), F7 72.2 % (18). Macro 0.5837 (G1a FAIL), minimum 0.2500 (G1b FAIL), macro ECE 0.1651 (G4b FAIL).
- **Operating point:** τ_0.05 flags 7.5 % of test reals (162 / 2,152; Wilson 6.5–8.7 %), 7.3 % stratum-weighted with R6 in R3; τ_0.01
  flags 1.3 % (29 / 2,152; Wilson 0.9–1.9 %). 2.1.0 read 5.0 % stratum-weighted at τ_0.05.
- **Real strata** (τ_0.05): R1 14 / 387 (3.6 %), R2 25 / 409 (6.1 %), R3 56 / 479 (11.7 %), R4 37 / 405 (9.1 %), R5 26 / 450 (5.8 %),
  R6 4 / 22.
- **Coverage and abstention:** conformal coverage reals 91.5 %, AI 93.9 % (G6a FAIL). Inconclusive 28.3 % (504 / 1,778; G6b FAIL);
  32.4 % under social_ig (G6c PASS).
- **Robustness** (562 items; τ fixed at the clean τ_0.05): G5a fails on jpeg75 (16.7 pts), webp80 (12.5) and resize0.5 (23.5); G5b on
  jpeg50 (28.0); G5c on resize0.5 (+3.0), crop70 (+2.0), screenshot_jpeg (+21.1) and noise3 (+22.8).
- **Shortcut gate** (D-52): structure PASS; 2a weighted 0.8582, unweighted 0.9526, disclosed by the `D-45 2a` row of
  `../shortfalls.md`; 2b 0.5000 (2.1.0: 0.7897); **2c: no remaining additive context term**, so no removal refit exists to run and
  `../shortcut_context_terms.json` is not needed.

## Video and audio

- **`fuser-video-2.1.1`** (`policy.VIDEO_2B`, git ac9ca5c): 5 detectors (`vid.lipsync.landmark_xcorr` skipped: no usable scores on
  fit items); `fit_n` 214, `cal_n` 139; λ 3.0 (CV log-loss 0.2328); cal AUC 0.9599 (in-sample). The fit dropped `has_face`,
  `face_px:*` and `min_side:*` (`size_face`) and `video_frame` (`constant_on_fit`). Gate 2b 0.5000. On `test`: AUC 0.9613
  (0.9374–0.9806), recall at τ_0.05 80.7 % (109 / 135), real videos flagged 2.5 % (4 / 162; G9b PASS); vendor-LGO macro 0.5953
  (G9a FAIL; 2.1.0: 0.6392).
- **`fuser-audio-2.1.1`**: `aud.gustking_xlsr`; `fit_n` 561, `cal_n` 313; λ 3.0 (CV log-loss 0.5046); cal AUC 0.7942. Gate 2b 0.5000.
  On `test`: AUC 0.8474 (0.8070–0.8852); EER on ASVspoof 5 17.4 % (G10a FAIL); in-the-wild bona fide FPR 1.1 % (1 / 94; G10b PASS).

## Field test

`docs/field_test.md` (job `j5-field`, 2026-09-17 05:11, `scripts/field_test.py run-v21 --record v211` against the dev API on `full`;
in-distribution sample). Images: original 16/23 decided, 13 correct; v2.1 set 42/80 decided, 35 correct (H-9: 22/22 and 66/71). Wrong
image decisions: 8 real camera originals called AI (short side 1,080–4,160 px), 2 FLUX.2 stills called authentic. Videos: 7/8 and
18/32 decided, all correct. Images by min-side bucket are in the record; no finished analysis published without an estimate (D-49).
