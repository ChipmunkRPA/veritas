# Veritas-Bench results: vb-mini · `cal`

Generated 2026-09-15T09:43:46Z by `backend/scripts/vbench_report.py` (git 6af1e933009f235875a23c5957cd31eb5a5039b8).

> **Not a goal evaluation.** evaluated on `cal`, not `test`: τ_α, ECE and coverage are in-sample.

- **System:** artefact `fuser-image-0.0.0-smoke.json` (sha256 `5fef786fdb71a2ce…`, fitted on vb-mini, reviewed_by: ralph-loop (automated; pending human review)).
- **Manifest:** sha256 `013a3f7ba2b6c5de7d1dfe358c8931a489d715ee52337bfb30d8fc7ea9333d80`; variant `norm`.
- **Thresholds from `cal` reals** (equal stratum weights): τ_0.05 = 0.5635 (n = 43); τ_0.01 = – (n = 43).
- **Counts:** 43 reals and 63 fakes scored; 0 items without a score; excluded {}.

## Sets

| Set | Reals | Fakes | AUC (95 % CI) | TPR@5 %FPR | TPR@1 %FPR | FPR at τ_0.05 | ECE (CI) | Brier | Inconclusive |
|---|---|---|---|---|---|---|---|---|---|
| `image/cal` | 43 | 63 | 0.9775 (0.9519–0.9947) | 87.3 % (55/63; Wilson 76.9 %–93.4 %) | 0.0 % (0/63; Wilson 0.0 %–5.7 %; rule of three ≤ 4.8 %) | 4.7 % (2/43; Wilson 1.3 %–15.5 %) | 0.0406 (0.0382–0.1023) | 0.0610 | 12.3 % |
| `image/cal/t2i` | 43 | 45 | 0.9793 (0.9505–0.9979) | 88.9 % (40/45; Wilson 76.5 %–95.2 %) | 0.0 % (0/45; Wilson 0.0 %–7.9 %; rule of three ≤ 6.7 %) | 4.7 % (2/43; Wilson 1.3 %–15.5 %) | 0.0502 (0.0402–0.1125) | 0.0566 | 11.4 % |
| `image/cal/edit` | 43 | 4 | 0.9942 (0.9667–1.0000) | 100.0 % (4/4; Wilson 51.0 %–100.0 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) | 4.7 % (2/43; Wilson 1.3 %–15.5 %) | 0.1349 (0.0597–0.2644) | 0.0504 | 14.9 % |
| `image/cal/personalised` | 43 | 3 | 1.0000 (1.0000–1.0000) | 100.0 % (3/3; Wilson 43.9 %–100.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) | 4.7 % (2/43; Wilson 1.3 %–15.5 %) | 0.1081 (0.0561–0.1783) | 0.0340 | 15.2 % |
| `image/cal/faceswap_vs_R5` | 10 | 7 | 1.0000 (1.0000–1.0000) | 100.0 % (7/7; Wilson 64.6 %–100.0 %) | 0.0 % (0/7; Wilson 0.0 %–35.4 %; rule of three ≤ 42.9 %) | 10.0 % (1/10; Wilson 1.8 %–40.4 %) | 0.1959 (0.1224–0.2526) | 0.0621 | 23.5 % |
| `image/cal/gan_face_vs_R5` | 10 | 4 | 0.7250 (0.3846–1.0000) | 25.0 % (1/4; Wilson 4.6 %–69.9 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) | 10.0 % (1/10; Wilson 1.8 %–40.4 %) | 0.3850 (0.2289–0.4949) | 0.2047 | 50.0 % |
| `image/cal/legacy` | 3 | 4 | 1.0000 (1.0000–1.0000) | 75.0 % (3/4; Wilson 30.1 %–95.4 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) | 0.0812 (0.0116–0.2238) | 0.0357 | 14.3 % |

Conformal coverage at α = 0.05: reals 95.3 % (41/43; Wilson 84.5 %–98.7 %), AI 96.8 % (61/63; Wilson 89.1 %–99.1 %).

## Real strata (FPR at τ)

| Stratum | n | FPR at τ_0.05 | FPR at τ_0.01 |
|---|---|---|---|
| R1 | 4 | 25.0 % (1/4; Wilson 4.6 %–69.9 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| R2 | 8 | 0.0 % (0/8; Wilson 0.0 %–32.4 %; rule of three ≤ 37.5 %) | 0.0 % (0/8; Wilson 0.0 %–32.4 %; rule of three ≤ 37.5 %) |
| R3 | 11 | 0.0 % (0/11; Wilson 0.0 %–25.9 %; rule of three ≤ 27.3 %) | 0.0 % (0/11; Wilson 0.0 %–25.9 %; rule of three ≤ 27.3 %) |
| R4 | 9 | 0.0 % (0/9; Wilson 0.0 %–29.9 %; rule of three ≤ 33.3 %) | 0.0 % (0/9; Wilson 0.0 %–29.9 %; rule of three ≤ 33.3 %) |
| R5 | 10 | 10.0 % (1/10; Wilson 1.8 %–40.4 %) | 0.0 % (0/10; Wilson 0.0 %–27.8 %; rule of three ≤ 30.0 %) |
| R6 | 1 | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |

## Fake families (recall at τ)

| Family | Threat | LGO fold | n | TPR at τ_0.05 | TPR at τ_0.01 |
|---|---|---|---|---|---|
| faceswap-stills | faceswap | – | 7 | 100.0 % (7/7; Wilson 64.6 %–100.0 %) | 0.0 % (0/7; Wilson 0.0 %–35.4 %; rule of three ≤ 42.9 %) |
| flux1 | t2i | – | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| flux2 | t2i | F2 | 2 | 50.0 % (1/2; Wilson 9.5 %–90.5 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| gan-faces | gan_face | – | 4 | 25.0 % (1/4; Wilson 4.6 %–69.9 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| gpt-4o-image-1 | t2i | – | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| gpt-image-1.5-2 | t2i | F3 | 1 | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| hidream | t2i | – | 1 | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| hunyuanimage | t2i | – | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| ideogram | t2i | – | 1 | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| imagen3 | t2i | – | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| imagen4 | t2i | F7 | 3 | 100.0 % (3/3; Wilson 43.9 %–100.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| legacy | t2i | – | 4 | 75.0 % (3/4; Wilson 30.1 %–95.4 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| long-tail | t2i | – | 5 | 100.0 % (5/5; Wilson 56.6 %–100.0 %) | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) |
| midjourney | t2i | F1 | 4 | 100.0 % (4/4; Wilson 51.0 %–100.0 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| modern-t2i-faces | t2i | – | 3 | 66.7 % (2/3; Wilson 20.8 %–93.9 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| nano-banana | t2i | F4 | 3 | 100.0 % (3/3; Wilson 43.9 %–100.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| partial-edits | edit | – | 4 | 100.0 % (4/4; Wilson 51.0 %–100.0 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| personalised | personalised | – | 3 | 100.0 % (3/3; Wilson 43.9 %–100.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| qwen-image | t2i | F6 | 3 | 100.0 % (3/3; Wilson 43.9 %–100.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| recraft | t2i | – | 1 | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| sd3 | t2i | – | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| seedream3 | t2i | – | 2 | 50.0 % (1/2; Wilson 9.5 %–90.5 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| seedream4 | t2i | F5 | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |

## Leave-generator-out folds

| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC | ECE |
|---|---|---|---|---|---|---|

Macro TPR at τ_0.05 over F1–F7: –; minimum –; macro ECE –.
Missing folds: F1: no refit artefact without F1; F2: no refit artefact without F2; F3: no refit artefact without F3; F4: no refit artefact without F4; F5: no refit artefact without F5; F6: no refit artefact without F6; F7: no refit artefact without F7.

## Robustness (fixed τ_0.05)

Reference `native`/`none` on 57 of 297 robustness items.

| Transform | Reals | Fakes | ΔTPR | ΔFPR | ΔAUC | Inconclusive under t |
|---|---|---|---|---|---|---|
| exif_forge | 8 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0 % |

Not scored: jpeg95, jpeg75, jpeg50, webp80, resize0.5, crop70, screenshot, screenshot_jpeg, social_ig, social_x, launder, noise3, upscale2x, rephoto_proxy.
- jpeg95: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- jpeg75: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- jpeg50: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- webp80: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- resize0.5: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- crop70: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- screenshot: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- screenshot_jpeg: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- social_ig: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- social_x: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- launder: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- noise3: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- upscale2x: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support
- rephoto_proxy: no cached rows for img.commfor384, img.dda, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, pe.copymove.pairs, pe.jpeg_dq.support

## Invariants

On 106 items: max |Σφ′ + b′ − logit s| = 1.776e-15; 0 of 106 random +Δz moves lowered s.

## Slices

9 slices; 0 measured (≥ 100 items per class), 0 merged with a measured parent slice, 9 not measured.

## Field conditions (EA-02 slice validation map)

| Condition | Reals | Fakes | AUC | Measured at |
|---|---|---|---|---|
| JPEG q ≥ 90, ≥ 1 MP, no face | 10 | 13 | – | not measured |
| JPEG q 70–89 | 19 | 8 | – | not measured |
| PNG / lossless (no JPEG quantisation tables) | 8 | 41 | – | not measured |
| face present | 0 | 0 | – | not measured |
| social recompressed (social_ig, social_x) | 0 | 0 | – | not measured |
| screenshot (screenshot, screenshot_jpeg) | 0 | 0 | – | not measured |

## Reproducibility

Two `calibrate` runs identical except `created_at`: **True** (fuser-image-0.0.0-smoke.json vs fuser-image-0.0.0-smoke.json).

## Not measured

- video (no video system: `--video-score-key`)
- audio (no audio system: `--audio-score-key`)
