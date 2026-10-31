# Image fuser `fuser-image-2.1.0` on vb-1.1 (BUILD_SPEC H-7)

Bench `vb-1.1`, manifest sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`, scope
`data/vbench/scope/vb-1.1.parquet` (sha256 `5059434a74cf…`). 2026-09-15. Automated record (ralph loop),
`reviewed_by: ralph-loop (automated; pending human review)` (D-17). Measured on `test` with 95 % intervals. The gate FAILS
(`../gate-image-2.1.0.txt`: 36 PASS, 10 FAIL, 12 MISSING of 58 checks). Video, audio and latency rows are MISSING by design in
this image run. Every FAIL or MISSING image goal is in `../shortfalls.md`.

## Decisions

- **Fitted set:** the detectors admitted on vb-1.1 under D-44 (`vb-1.1-admissions.md`); superseded by v2.1.1-admissions (J-3c).
  J-5 gives the numbers of the refit.
  - `img.commfor384` and `img.dda`, admitted with the `jpeg_q<70` A5 term;
  - `img.dfdc_b7`, admitted with its H-3 gate `has_face`.
  - Nothing else passed: the probe head was rejected on established R4 harm (`probe-v1.md`), and the derived partial-edit
    features added nothing (`../partial_edit_h6.md`).
- **Default profile** (`backend/app/models.yaml`):
  - `img.commfor384` and `img.dda` are now `default_enabled: true`: the loader only uses an artefact whose detectors are
    enabled.
  - Following the I-14 precedent for `img.univfd`, the rejected detectors without a lane of their own are off:
    `img.hf.ateeqq` (A4), `img.hf.dima806` (A3), `img.forensic.spectrum` (A2, A3).
  - The partial-edit forensic checks and the metadata / watermark rules stay on for their lanes and Stage 0.
- **Loader:** `full` and `deep` select `fuser-image-2.1.0.json` ("matches the enabled detectors"). `quick` falls back because
  `img.dda` does not run in `quick`, as v2.0 did for the forensic set.
- **Context of the scored bytes** (ADR 0005, `--context scored`) and the scope items only (`--scope`, PROGRESS spec question
  "Scope vs manifest population").
- **D-44 runtime terms:** `screenshot_like` (ConditionProfile `screenshot_likelihood` ≥ 0.5) is a context term and an interaction
  group since this item. The fit keeps only the terms present in fit rows: `has_face`, `face_px:120-256`, `face_px>=256`,
  `jpeg_q>=90`, `min_side<512`, `min_side>1024`; interaction group `has_face`.
  - No `norm` fit row has JPEG q < 90 or a screenshot context, so the `jpeg_q<70` interaction the admission names cannot be
    learned from fit rows.
  - The cost shows under recompression: G5c social_ig ΔFPR +16.9 pts, jpeg50 +5.4 pts.

## Commands

| Step | Job and command |
|---|---|
| Fit, LGO refits, reproducibility fit | `h7-fit`: `bash scripts/h7_fit.sh`, i.e. `calibrate.py --manifest data/vbench/manifest.parquet --scores data/vbench/scores --modality image --seed 7 --bench vb-1.1 --context scored --scope data/vbench/scope/vb-1.1.parquet --detectors img.commfor384,img.dda,img.dfdc_b7 --gates "img.dfdc_b7=has_face"` with `--version 2.1.0` (and one `--exclude-folds F<k>` refit per fold) |
| Report and gate | `h7-report3`: `bash scripts/h7_report.sh`. Two reproducibility fits at one commit, then `vbench_report.py … --artefact data/fusion/fuser-image-2.1.0.json --lgo-artefacts data/fusion/lgo-image-2.1.0 --eval-split test --bench vb-1.1 --scope data/vbench/scope/vb-1.1.parquet --repro <repro> <repro2> --write-artefact --update-cards app/models.yaml`, then `vbench_gate.py` |
| Envelope | `envelope_record.py --results ../docs/v2/bench/results.json --field ../docs/field_test.json --json ../docs/v2/bench/h7_envelope.json --markdown ../docs/v2/bench/h7_envelope.md` |

Three report runs are logged in `test_access.log`. The first (`h7-report`) had no robustness rows: the vb-1.1 manifest has no
`robustness` column, so the subset fell back to the stratified rule, and its two reproducibility fits came from different
commits (G12 "differing git_commit"). The second (`h7-report2`) left the "face present" field condition empty: the manifest
never fills `face_px_max`. The report now takes the robustness flags from the scope and `face_px` from the scored context. The
third run is the record. The fitted coefficients did not change between runs (each run rewrites only `metrics.test` and
`metrics.lgo`); nothing was tuned on `test`.

## Fit

- `fit_n` 3,305, `cal_n` 1,696 (fit ∪ cal scope items; C2PA items excluded); 3 detectors, 0 skipped.
- λ 0.1 (5-fold grouped CV log-loss 0.2089); cal AUC 0.9716 (in-sample); d95 2.173.
- Sub-fusers `h1_synthesis`, `h2_face_swap`, `h2_partial_edit`.
- LGO refits in `data/fusion/lgo-image-2.1.0/`. Artefact sha256 after `--write-artefact`: `734e9759314c…`.

## Results on `test` (lockbox; 2,152 reals, 1,843 fakes scored)

| Set | AUC (95 % CI) | TPR@5 %FPR (k / n; Wilson) | TPR@1 %FPR | ECE | Brier |
|---|---|---|---|---|---|
| `image/test` | 0.9632 (0.9571–0.9688) | 84.4 % (1,556 / 1,843; 82.7–86.0 %) | 64.8 % | 0.0221 | 0.0752 |
| t2i (1,264 fakes) | 0.9630 (0.9557–0.9697) | 86.7 % (1,096 / 1,264; 84.7–88.5 %) | 66.0 % | 0.0321 | 0.0731 |
| edit (153) | 0.9664 (0.9489–0.9812) | 86.3 % (132 / 153; 79.9–90.8 %) | 75.8 % | 0.0334 | 0.0688 |
| personalised (92) | 0.9989 (0.9979–0.9996) | 100.0 % (92 / 92; 96.0–100 %) | 94.6 % | 0.0659 | 0.0328 |
| face swap vs R5 (211) | 0.9380 (0.9173–0.9565) | 71.6 % (151 / 211; 65.1–77.2 %) | 46.0 % | 0.0794 | 0.1084 |
| GAN face vs R5 (123) | 0.9274 (0.9010–0.9516) | 69.1 % (85 / 123; 60.5–76.6 %) | 48.8 % | 0.0818 | 0.1146 |
| legacy (107) | 0.9721 (0.9580–0.9840) | 86.0 % (92 / 107; 78.2–91.3 %) | 85.0 % | 0.0520 | 0.0589 |

- **LGO** (refits without F, TPR@5 %FPR on F's test fakes): F1 Midjourney 98.2 % (277), F2 FLUX.2 59.0 % (134), F3 GPT-image
  50.5 % (107), F4 nano-banana 84.3 % (198), F5 Seedream 4 75.3 % (146), F6 Qwen-Image 86.8 % (190), F7 Imagen 4 94.4 % (178).
  Macro 0.7836 (G1a PASS), minimum 0.5047 (G1b PASS), LGO macro ECE 0.0822 (G4b FAIL). v2.0 on core: macro 0.5145, F1 21.1 %.
  Corrected (J-5): these are the H-7 run's values. Under the ADR 0007 threshold rule (H-8, `../gate-v2.1-final.txt`) the macro is
  0.7716 (G1a PASS) and the minimum 0.4766 (F3 47.7 %; G1b FAIL). Both are superseded by `fuser-image-2.1.1.md` (J-5).
- **Real strata** (FPR at τ_0.05): R1 42 / 387 (10.9 %), R2 0 / 409, R3 61 / 479 (12.7 %), R4 2 / 405, R5 15 / 450 (3.3 %),
  R6 2 / 22. Pooled 122 / 2,152 (5.7 %; Wilson 4.8–6.7 %).
- **Coverage and abstention:** conformal coverage reals 95.2 %, AI 95.2 % (G6a PASS). Inconclusive 18.2 % (729 / 3,995; G6b PASS).
  Selective risk 5.9 % (192 / 3,266).
- **Robustness** (557 robustness items scored native: 296 reals, 261 fakes; τ fixed at the clean τ_0.05). ΔTPR = clean − transformed
  (positive: detection falls); ΔFPR = transformed − clean (positive: more false positives):

| Transform | ΔTPR pts | ΔFPR pts | | Transform | ΔTPR pts | ΔFPR pts |
|---|---|---|---|---|---|---|
| jpeg75 | +4.2 | +0.0 | | social_ig | +2.7 | **+16.9** |
| jpeg50 | +0.8 | **+5.4** | | social_x | +3.1 | −0.3 |
| webp80 | +2.7 | +1.4 | | screenshot_jpeg | **+86.6** | −6.4 |
| resize0.5 | **+17.2** | +0.0 | | noise3 | +2.3 | +1.4 |
| crop70 | +2.7 | +1.0 | | upscale2x (276 reals) | **+31.0** | −6.2 |
| exif_forge | +0.0 | +0.0 | | not scored | jpeg95, screenshot, launder, rephoto_proxy | |

- **G12:** the two reproducibility fits are identical except `created_at`. **G14** legacy AUC 0.9721 PASS. **G15a/b** invariants
  PASS.

## Slice validation map (PRD EA-02; `results.json` `field_conditions`, n ≥ 100 per class)

| Field condition | Measured at | Reals | Fakes | AUC |
|---|---|---|---|---|
| JPEG q ≥ 90, ≥ 1 MP, no face | own slice | 489 | 205 | 0.9863 |
| JPEG q 70–89 | own slice | 751 | 298 | 0.9803 |
| PNG / lossless (no JPEG quantisation tables) | own slice | 365 | 1,116 | 0.9162 |
| face present | own slice | 867 | 891 | 0.9525 |
| social recompressed (social_ig, social_x) | own slice | 592 | 522 | 0.9399 |
| screenshot (screenshot_jpeg; the PNG screenshot was not scored) | own slice | 296 | 261 | 0.8615 |

- **Clean-condition predicates:** JPEG quality and size come from the manifest (the native file). "face present" uses the face
  stage's `face_px` of the scored bytes.
- **Leaf slices:** `results.json` holds 42 runtime leaf slices. Two are measured at their own level
  (`image/lossless/side<512/face120-256` 0.8312, `image/q>=90/side512-1024/noface` 0.9299); the others carry their nearest
  measured parent. The lowest AUC, 0.7824, is carried from `image/lossless/side<512` (193 reals, 180 fakes). Every leaf is
  ≥ 0.70.
- **Envelope record** (`../h7_envelope.md`): **23 of 23** field-test images (`docs/field_test.json`, v2.0 run) recorded a slice id
  inside the validated envelope. Their decisions in that run were inconclusive for other reasons (conformal set); the v2.1 field
  run is H-9.

## Known issues carried to later items

- **G3 R1 / R3:** the equal-stratum τ puts false positives on camera originals and social-platform bytes while R2 / R4 sit near 0.
  The lossless no-EXIF RAISE reals (H-3) remain in R1.
- **Scoring scope:** the recompression interaction terms need fit rows with JPEG q < 90 (native social or JPEG-50 rows are
  robustness rows, never fit rows). jpeg95, screenshot, launder and rephoto_proxy were not scored (D-40 scope).
- **Duplicates:** the two OpenFake ideogram test items with byte-identical cal / fit copies (PROGRESS "vb-1.1 duplicates")
  are in `test`; they can move the pooled numbers by at most 2 items.

## H-8 re-measurement (2026-09-15)

- **Run:** job `h8-report` (`scripts/h8_report.sh`) re-ran this artefact with the video and audio fusers under the ADR 0007
  threshold rule. R6 (11 `cal` reals) is pooled into R3, so τ_0.05 = 0.5790 from 850 `cal` reals.
- **Numbers that moved:** image/test TPR at τ_0.05 84.4 % → 83.6 %; G1b minimum family (F3) 50.5 % → 47.7 % (FAIL); G7b
  71.6 % → 69.7 % (FAIL); G3 R1 10.9 % → 9.3 %, R3 12.7 % → 12.3 %.
- **Artefact:** `--write-artefact` refilled `metrics.test`; sha256 now `5d586e34f0ab…`.
- **Records:** current numbers are in `../results.md` and `../shortfalls.md`.
