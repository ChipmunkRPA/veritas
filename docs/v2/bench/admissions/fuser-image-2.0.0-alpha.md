# Image fuser `fuser-image-2.0.0-alpha` on vb-1.0-core (I-14; M-3 merged, D-39)

Bench `vb-1.0-core`, manifest sha256 `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`. 2026-09-14. Automated record
(ralph loop), `reviewed_by: ralph-loop (automated; pending human review)` (D-17). Measured on core with 95 % intervals; the gate
FAILS, and every FAIL or MISSING image goal is in `../shortfalls.md`.

## Decisions

- **Fitted set.** The commercial default image set minus `img.univfd`: `img.hf.ateeqq`, `img.hf.dima806`, `img.dfdc_b7`,
  `img.forensic.{spectrum,ela,noise,jpeg_dq,copymove}`, `img.meta.{generator,exif,consistency,qtables}`, `img.watermark.dwtdct`.
  No candidate passed every admission criterion as written (`baseline-image.md`, `img.commfor384.md`, `img.dda.md`,
  `face.gend_clip_l14.md`, `veritas-probe-v1.md`), so the rejected candidates stay off and the fuser is fitted on the baseline.
- **UnivFD removed, decided on fit ∪ cal before the lockbox run.** A2 AUC 0.5065 [0.4382, 0.5875] on its declared threats; the
  baseline without it is 4.1 pts better on LGO macro TPR (CI [-9.4, +0.8] for with − without). `models.yaml`: `img.univfd`
  `default_enabled: false`, admission `rejected`. `vid.frames.univfd` is unchanged (V-8 decides the video lanes).
- **dima806 and DFDC-B7 stay enabled** and are fitted (A3a +3.5 pts; DFDC-B7 is the only face-swap signal). Their admission stays
  `pending` because A4 fails as written (spec question in PROGRESS: A4 cannot pass on strata of 49–85 reals unless ΔFPR = 0).
- **Skipped by the fit** ("no usable scores on fit items"): the four metadata rules and the watermark decoder. On all 642 fit
  items their cached `norm` rows have status `not_applicable` and no probability (checked in the score cache); they still act
  through Stage 0 and findings, not through the fuser.
- **Profiles.** `full` and `deep` load this artefact (loader check: "fuser-image-2.0.0-alpha.json matches the enabled detectors",
  sha256 `3486cde0d4bf…`). `quick` does not enable the forensic detectors, so quick analyses keep the uncalibrated fallback
  ("detector img.forensic.copymove is not enabled"). Profiles are unchanged; a quick-profile artefact is not built.

## Fit

`fit_n` 642, `cal_n` 288 (fit ∪ cal, C2PA items excluded: fit 55, cal 39); 8 detectors fitted, 5 skipped; λ 0.3 (CV log-loss
0.2985); cal AUC 0.8928 (in-sample); d95 0.7690. LGO refits exclude one fold each (F1–F7) in `data/fusion/lgo-image-2.0.0-alpha/`.

## Results on `test` (lockbox), v2 vs the v0.3 baseline (`baseline-v0.3/results.json`)

| Set | v2 AUC (95 % CI) | v2 TPR@5 %FPR | v2 TPR@1 %FPR | v2 ECE / Brier | v0.3 AUC | v0.3 TPR@5 % | v0.3 TPR@1 % |
|---|---|---|---|---|---|---|---|
| `image/test` (344 real, 544 fake) | 0.8944 (0.8703–0.9156) | 47.2 % (Wilson 43.1–51.4) | 39.7 % | 0.0626 / 0.1287 | 0.8116 | 34.4 % | 7.0 % |
| t2i (381 fakes) | 0.9159 (0.8939–0.9351) | 53.8 % | 46.2 % | 0.0586 / 0.1128 | 0.8155 | 35.2 % | 5.5 % |
| edit (40) | 0.7918 (0.7219–0.8513) | 12.5 % | 7.5 % | 0.2008 / 0.1892 | 0.7123 | 15.0 % | 5.0 % |
| face swap vs R5 (60) | 0.9042 (0.8546–0.9456) | 48.3 % | 41.7 % | 0.1044 / 0.1268 | 0.8067 | 38.3 % | 8.3 % |
| GAN face vs R5 (36) | 0.7573 (0.6600–0.8452) | 25.0 % | 8.3 % | 0.1680 / 0.2148 | 0.7436 | 22.2 % | 11.1 % |
| legacy (32) | 0.9865 (0.9649–0.9989) | 43.8 % | 40.6 % | 0.1434 / 0.0725 | 0.9359 | 31.3 % | 0.0 % |

- **Thresholds** from 110 `cal` reals: τ_0.05 = 0.8186, τ_0.01 = 0.8636.
- **Real strata, FPR at τ_0.05:** R1 17.5 % (11/63), R2 0/26, R3 2.6 % (2/76), R4 0/88, R5 6.9 % (6/87), R6 0/4. v0.3: R1 17.5 %,
  R3 13.2 %, R5 6.9 %, R6 25.0 %.
- **LGO (refits, τ from each refit's cal reals), TPR at τ_0.05:** F1 midjourney 21.1 %, F2 flux2 51.3 %, F3 gpt-image-1.5-2 50.0 %,
  F4 nano-banana 58.0 %, F5 seedream4 28.3 %, F6 qwen-image 64.0 %, F7 imagen4 87.5 %; macro 0.5145 (v0.3 0.3620), minimum 0.2113
  (v0.3 0.2000), macro ECE 0.1223.
- **Conformal coverage at α = 0.05:** reals 94.5 % (325/344), AI 96.5 % (525/544); inconclusive 41.4 %.
- **Robustness (fixed τ_0.05, 148 reals / 130 fakes):** ΔTPR / ΔFPR jpeg75 -0.1000 / +0.0338, resize0.5 -0.0846 / +0.0068,
  screenshot_jpeg -0.1231 / +0.0676, social_ig -0.0923 / +0.0270 (ΔTPR_t = TPR_clean − TPR_t, so the fused output flags more
  items under these transforms). Other transforms are not reported: the forensic detectors have no cached rows for them.
- **Invariants (G15):** max |Σφ′ + b′ − logit s| 8.882e-16; 0 of 888 monotonicity trials violated.
- **G12:** two `calibrate` runs identical except `created_at` (`repro-` vs `repro2-image-2.0.0-alpha/`).

## Gate

`gate-image-2.0.0-alpha.txt`: 14 PASS, 18 FAIL, 26 MISSING of 58 checks (exit 1). PASS: G3 R4, G5a jpeg75 and resize0.5, G5b social_ig
and screenshot_jpeg, G5c resize0.5, G6a real and ai, G6c, G7a, G12, G14 (legacy AUC 0.9865), G15a, G15b. Video, audio (G9, G10) and
latency (G13) are MISSING in this run: V-8, A-1 and X-4 own them.

## Model cards

`vbench_report.py --update-cards app/models.yaml` wrote one `eval` row (`image/test`) to each fitted detector card: Ateeqq AUC 0.7535,
dima806 0.5131, DFDC-B7 0.5374 (403 face items), spectrum 0.6274, noise 0.5433, ELA 0.4967, jpeg_dq 0.5 (4 items), copymove 0.5 (16
items). These are single-detector AUCs over all test threats, not on each card's declared threats (A2 in the admission records).

## Report fix during this item

The first report run fused the 11 transforms without forensic rows as if those detectors had not run (partial fusion: G5a / G5b
PASS rows on jpeg95, webp80, crop70 and others). `report._robustness` now skips a transform when a fitted detector has no cached
row on an item another detector has one for, and lists the reason (`not_scored_reasons`); test
`test_robustness_needs_every_artefact_detectors_rows`. A second run passed one repro artefact twice to `--repro`; the third run
compared two separate `calibrate` outputs. `test_access.log` holds all three lockbox accesses.

## Reproduce

```bash
B=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct
make calibrate BENCH=vb-1.0-core DETECTORS=$B
cd backend
for f in F1 F2 F3 F4 F5 F6 F7; do .venv/bin/python scripts/calibrate.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --modality image --seed 7 --out data/fusion/lgo-image-2.0.0-alpha/ --bench vb-1.0-core --detectors $B --exclude-folds $f --version 2.0.0-alpha.lgo-$f; done
for d in repro repro2; do .venv/bin/python scripts/calibrate.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --modality image --seed 7 --out data/fusion/$d-image-2.0.0-alpha/ --bench vb-1.0-core --detectors $B; done
.venv/bin/python scripts/vbench_report.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --artefact data/fusion/fuser-image-2.0.0-alpha.json --lgo-artefacts data/fusion/lgo-image-2.0.0-alpha --eval-split test --bench vb-1.0-core \
  --repro data/fusion/repro-image-2.0.0-alpha/fuser-image-2.0.0-alpha.json data/fusion/repro2-image-2.0.0-alpha/fuser-image-2.0.0-alpha.json \
  --write-artefact --update-cards app/models.yaml
.venv/bin/python scripts/vbench_gate.py --results ../docs/v2/bench/results.json --goals ../docs/v2/bench/goals.yaml \
  --manifest data/vbench/core/manifest.parquet > ../docs/v2/bench/gate-image-2.0.0-alpha.txt
```

Job logs: `i14-calibrate`, `i14-report` (run 1), `i14-report2` (self-compared repro, superseded), `i14-report3` (recorded).
