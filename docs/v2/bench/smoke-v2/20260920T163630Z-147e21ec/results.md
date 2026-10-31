# Veritas-Bench results: vb-mini · `cal`

Generated 2026-09-20T16:36:40Z by `backend/scripts/vbench_report.py` (git 21702066f6e1bf0073629617e35bd3b5fe4d6983).

> **Not a goal evaluation.** evaluated on `cal`, not `test`: τ_α, ECE and coverage are in-sample.

- **System:** artefact `fuser-image-0.0.0-smoke.json` (sha256 `293f7f719a012f23…`, fitted on vb-mini, reviewed_by: ralph-loop (automated; pending human review)).
- **Manifest:** sha256 `3e2e117b50c6e75476db84a3dca948ce93e98511e7ade985ab0591acd2d0b461`; variant `norm`; items of the scope sha256 `2041bf6787219d73…` only.
- **Thresholds from `cal` reals** (equal stratum weights): τ_0.05 = 0.7489 (n = 52); τ_0.01 = – (n = 52).
- **Counts:** 52 reals and 50 fakes scored; 0 items without a score; 1 capped without an estimate because a check did not run (cal reals 0); excluded {}.
- **View (D-45):** `image/cal`, its threat subsets and the LGO folds are size-matched in min-side buckets (every item of the rarer class plus a salted sha256 sample of the other at the pooled class ratio); `image/cal/full` and `/full/<subset>` hold the full view.

- **image selective policy:** loaded artefact `293f7f719a012f23131207a5ac7ff9a816607b192d050276609c3c3a2fa8d403`; envelope missing (sha256 `not present`). Default α = 0.05, slice AUC ≥ 0.70; excludes documentary provenance overrides, workspace policy overrides, configured prior changes, provenance contribution bound, confidence adjustments.

## Sets

| Set | Reals | Fakes | AUC (95 % CI) | TPR@5 %FPR | TPR@1 %FPR | FPR at τ_0.05 | ECE (CI) | Brier | Inconclusive |
|---|---|---|---|---|---|---|---|---|---|
| `image/cal` | 25 | 25 | 0.9360 (0.8649–0.9852) | 68.0 % (17/25; Wilson 48.4 %–82.8 %) | 0.0 % (0/25; Wilson 0.0 %–13.3 %; rule of three ≤ 12.0 %) | 8.0 % (2/25; Wilson 2.2 %–25.0 %) | 0.1603 (0.1035–0.2453) | 0.1137 | 100.0 % |
| `image/cal/full` | 52 | 50 | 0.9246 (0.8700–0.9677) | 62.0 % (31/50; Wilson 48.2 %–74.1 %) | 0.0 % (0/50; Wilson 0.0 %–7.1 %; rule of three ≤ 6.0 %) | 3.8 % (2/52; Wilson 1.1 %–13.0 %) | 0.1073 (0.0877–0.1876) | 0.1156 | 100.0 % |
| `image/cal/t2i` | 16 | 11 | 0.9716 (0.8977–1.0000) | 63.6 % (7/11; Wilson 35.4 %–84.8 %) | 0.0 % (0/11; Wilson 0.0 %–25.9 %; rule of three ≤ 27.3 %) | 0.0 % (0/16; Wilson 0.0 %–19.4 %; rule of three ≤ 18.8 %) | 0.1817 (0.0922–0.2726) | 0.0875 | 100.0 % |
| `image/cal/full/t2i` | 52 | 36 | 0.9060 (0.8346–0.9612) | 58.3 % (21/36; Wilson 42.2 %–72.9 %) | 0.0 % (0/36; Wilson 0.0 %–9.6 %; rule of three ≤ 8.3 %) | 3.8 % (2/52; Wilson 1.1 %–13.0 %) | 0.1198 (0.0870–0.2118) | 0.1287 | 100.0 % |
| `image/cal/edit` | 18 | 1 | 1.0000 (1.0000–1.0000) | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) | 0.0 % (0/18; Wilson 0.0 %–17.6 %; rule of three ≤ 16.7 %) | 0.1951 (0.1708–0.2249) | 0.0458 | 100.0 % |
| `image/cal/full/edit` | 52 | 3 | 1.0000 (1.0000–1.0000) | 100.0 % (3/3; Wilson 43.9 %–100.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) | 3.8 % (2/52; Wilson 1.1 %–13.0 %) | 0.1601 (0.1029–0.2401) | 0.0595 | 100.0 % |
| `image/cal/personalised` | 13 | 0 | – (–) | – | – | 0.0 % (0/13; Wilson 0.0 %–22.8 %; rule of three ≤ 23.1 %) | – (–) | – | 100.0 % |
| `image/cal/full/personalised` | 52 | 3 | 0.9231 (0.8113–1.0000) | 33.3 % (1/3; Wilson 6.1 %–79.2 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) | 3.8 % (2/52; Wilson 1.1 %–13.0 %) | 0.2563 (0.1131–0.3670) | 0.1092 | 100.0 % |
| `image/cal/faceswap_vs_R5` | 6 | 2 | 1.0000 (1.0000–1.0000) | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) | 33.3 % (2/6; Wilson 9.7 %–70.0 %) | 0.4290 (0.3121–0.4900) | 0.2558 | 100.0 % |
| `image/cal/full/faceswap_vs_R5` | 12 | 5 | 1.0000 (1.0000–1.0000) | 100.0 % (5/5; Wilson 56.6 %–100.0 %) | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) | 16.7 % (2/12; Wilson 4.7 %–44.8 %) | 0.3134 (0.2207–0.3983) | 0.1506 | 100.0 % |
| `image/cal/gan_face_vs_R5` | 6 | 2 | 0.7500 (0.2857–1.0000) | 50.0 % (1/2; Wilson 9.5 %–90.5 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) | 33.3 % (2/6; Wilson 9.7 %–70.0 %) | 0.4376 (0.3206–0.5044) | 0.2601 | 100.0 % |
| `image/cal/full/gan_face_vs_R5` | 12 | 3 | 0.7778 (0.4615–1.0000) | 33.3 % (1/3; Wilson 6.1 %–79.2 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) | 16.7 % (2/12; Wilson 4.7 %–44.8 %) | 0.3256 (0.2437–0.4122) | 0.1567 | 100.0 % |
| `image/cal/legacy` | 3 | 1 | 1.0000 (1.0000–1.0000) | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) | 0.0751 (0.0319–0.1266) | 0.0132 | 100.0 % |
| `image/cal/full/legacy` | 5 | 2 | 1.0000 (1.0000–1.0000) | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) | 0.0638 (0.0273–0.1029) | 0.0107 | 100.0 % |

Conformal coverage at α = 0.05: reals 92.0 % (23/25; Wilson 75.0 %–97.8 %), AI 100.0 % (25/25; Wilson 86.7 %–100.0 %).

## Real strata (FPR at τ)

| Stratum | n | FPR at τ_0.05 | FPR at τ_0.01 |
|---|---|---|---|
| R1 | 4 | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| R2 | 10 | 0.0 % (0/10; Wilson 0.0 %–27.8 %; rule of three ≤ 30.0 %) | 0.0 % (0/10; Wilson 0.0 %–27.8 %; rule of three ≤ 30.0 %) |
| R3 | 13 | 0.0 % (0/13; Wilson 0.0 %–22.8 %; rule of three ≤ 23.1 %) | 0.0 % (0/13; Wilson 0.0 %–22.8 %; rule of three ≤ 23.1 %) |
| R4 | 11 | 0.0 % (0/11; Wilson 0.0 %–25.9 %; rule of three ≤ 27.3 %) | 0.0 % (0/11; Wilson 0.0 %–25.9 %; rule of three ≤ 27.3 %) |
| R5 | 12 | 16.7 % (2/12; Wilson 4.7 %–44.8 %) | 0.0 % (0/12; Wilson 0.0 %–24.2 %; rule of three ≤ 25.0 %) |
| R6 | 2 | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |

## Real strata by min-side bucket (FPR at τ_0.05)

| Stratum | Min side (px) | n | FPR at τ_0.05 |
|---|---|---|---|
| R1 | 384-767 | 3 | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| R1 | >1536 | 1 | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| R2 | >1536 | 10 | 0.0 % (0/10; Wilson 0.0 %–27.8 %; rule of three ≤ 30.0 %) |
| R3 | 1025-1536 | 3 | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| R3 | 384-767 | 4 | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| R3 | 768-1024 | 2 | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| R3 | >1536 | 4 | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| R4 | 1025-1536 | 5 | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) |
| R4 | 384-767 | 3 | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| R4 | <384 | 3 | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| R5 | 384-767 | 3 | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| R5 | 768-1024 | 3 | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| R5 | <384 | 6 | 33.3 % (2/6; Wilson 9.7 %–70.0 %) |
| R6 | >1536 | 2 | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |

## Fake families (recall at τ)

| Family | Threat | LGO fold | n | TPR at τ_0.05 | TPR at τ_0.01 |
|---|---|---|---|---|---|
| faceswap-stills | faceswap | – | 5 | 100.0 % (5/5; Wilson 56.6 %–100.0 %) | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) |
| flux1 | t2i | – | 2 | 50.0 % (1/2; Wilson 9.5 %–90.5 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| flux2 | t2i | F2 | 2 | 50.0 % (1/2; Wilson 9.5 %–90.5 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| gan-faces | gan_face | – | 3 | 33.3 % (1/3; Wilson 6.1 %–79.2 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| gpt-4o-image-1 | t2i | – | 1 | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| gpt-image-1.5-2 | t2i | F3 | 1 | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| hidream | t2i | – | 1 | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| hunyuanimage | t2i | – | 1 | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| ideogram | t2i | – | 1 | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| imagen3 | t2i | – | 1 | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| imagen4 | t2i | F7 | 2 | 50.0 % (1/2; Wilson 9.5 %–90.5 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| legacy | t2i | – | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| long-tail | t2i | – | 4 | 75.0 % (3/4; Wilson 30.1 %–95.4 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| midjourney | t2i | F1 | 4 | 100.0 % (4/4; Wilson 51.0 %–100.0 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| modern-t2i-faces | t2i | – | 2 | 50.0 % (1/2; Wilson 9.5 %–90.5 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| nano-banana | t2i | F4 | 3 | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| partial-edits | edit | – | 3 | 100.0 % (3/3; Wilson 43.9 %–100.0 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| personalised | personalised | – | 3 | 33.3 % (1/3; Wilson 6.1 %–79.2 %) | 0.0 % (0/3; Wilson 0.0 %–56.1 %; rule of three ≤ 100.0 %) |
| qwen-image | t2i | F6 | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| recraft | t2i | – | 1 | 100.0 % (1/1; Wilson 20.7 %–100.0 %) | 0.0 % (0/1; Wilson 0.0 %–79.3 %; rule of three ≤ 300.0 %) |
| sd3 | t2i | – | 2 | 100.0 % (2/2; Wilson 34.2 %–100.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| seedream3 | t2i | – | 2 | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |
| seedream4 | t2i | F5 | 2 | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) | 0.0 % (0/2; Wilson 0.0 %–65.8 %; rule of three ≤ 150.0 %) |

## Leave-generator-out folds

| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC | ECE |
|---|---|---|---|---|---|---|

Macro TPR at τ_0.05 over F1–F7: –; minimum –; macro ECE –.
Missing folds: F1: no refit artefact without F1; F2: no refit artefact without F2; F3: no refit artefact without F3; F4: no refit artefact without F4; F5: no refit artefact without F5; F6: no refit artefact without F6; F7: no refit artefact without F7.

## Robustness (fixed τ_0.05)

Reference `native`/`none` on 0 of 0 robustness items.


Not scored: jpeg95, jpeg75, jpeg50, webp80, resize0.5, crop70, screenshot, screenshot_jpeg, social_ig, social_x, launder, noise3, upscale2x, rephoto_proxy, exif_forge.

## Invariants

On 102 items: max |Σφ′ + b′ − logit s| = 8.882e-16; 0 of 102 random +Δz moves lowered s.

## Slices

11 slices; 0 measured (≥ 100 items per class), 0 merged with a measured parent slice, 11 not measured.

## Field conditions (EA-02 slice validation map)

| Condition | Reals | Fakes | AUC | Measured at |
|---|---|---|---|---|
| JPEG q ≥ 90, ≥ 1 MP, no face | 14 | 8 | – | not measured |
| JPEG q 70–89 | 22 | 8 | – | not measured |
| PNG / lossless (no JPEG quantisation tables) | 8 | 29 | – | not measured |
| face present | 14 | 25 | – | not measured |
| social recompressed (social_ig, social_x) | 0 | 0 | – | not measured |
| screenshot (screenshot, screenshot_jpeg) | 0 | 0 | – | not measured |

## Reproducibility

Two `calibrate` runs identical except `created_at`: **True** (fuser-image-0.0.0-smoke.json vs fuser-image-0.0.0-smoke.json).

## Not measured

- video (no video system: `--video-score-key`)
- audio (no audio system: `--audio-score-key`)
