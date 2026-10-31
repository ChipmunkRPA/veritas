# Veritas-Bench results: vb-1.1 · `test`

Generated 2026-10-10T19:07:41Z by `backend/scripts/vbench_report.py` (git ffcca70b400209c63b95df635834f880273ddda7).

- **System:** artefact `fuser-image-2.1.1.json` (sha256 `746ada66f73d77c3…`, fitted on vb-1.1, reviewed_by: ralph-loop (automated; pending human review)).
- **Manifest:** sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`; variant `norm`; items of the scope sha256 `2041bf6787219d73…` only.
- **Thresholds from `cal` reals** (equal stratum weights): τ_0.05 = 0.7150 (n = 850); τ_0.01 = 0.9607 (n = 850).
- **Counts:** 2152 reals and 1843 fakes scored; 0 items without a score; 0 capped without an estimate because a check did not run (cal reals 0); excluded {"cal": {"c2pa_manifest": 155, "outside_scope:image": 420, "outside_scope:video": 221}, "diagnostic": {"outside_scope:image": 260}, "excluded": {"outside_scope:video": 15}, "fit": {"c2pa_manifest": 219, "outside_scope:image": 751, "outside_scope:video": 391}, "test": {"c2pa_manifest": 314, "outside_scope:image": 982, "outside_scope:video": 493}, "wild": {"outside_scope:image": 93}}.
- **View (D-45):** `image/test`, its threat subsets and the LGO folds are size-matched in min-side buckets (every item of the rarer class plus a salted sha256 sample of the other at the pooled class ratio); `image/test/full` and `/full/<subset>` hold the full view.

- **image selective policy:** loaded artefact `746ada66f73d77c3ca948eb86d38e3f422ac3e6843ccfbab64f807b065ad153a`; envelope present (sha256 `ba33e71490a0b4c5815028fa116bf355f302f29d62b34e1d9b3509b3d3e58501`). Default α = 0.05, slice AUC ≥ 0.70; excludes documentary provenance overrides, workspace policy overrides, configured prior changes, provenance contribution bound, confidence adjustments.

- **video selective policy:** loaded artefact `fbbccc434f1af6a9e1fb64ccfddbbc1dc70a01f1f4fab9457b49e699f7ffcb45`; envelope present (sha256 `3a1dbaa4eaa8a2343f16d565dd74c770379fb5bbb2357e79d25201536b624a3d`). Default α = 0.05, slice AUC ≥ 0.70; excludes documentary provenance overrides, workspace policy overrides, configured prior changes, provenance contribution bound, confidence adjustments.

- **audio selective policy:** loaded artefact `44b63c7c8029103cf1b049c0e3548e6c2a39a027d95e39119145b83bc71990d3`; envelope present (sha256 `610b3ee21edf1717f0059aa0ea0c4b4eea3fe132d13b93e6574f7a8bb4d924c6`). Default α = 0.05, slice AUC ≥ 0.70; excludes documentary provenance overrides, workspace policy overrides, configured prior changes, provenance contribution bound, confidence adjustments.

## Sets

| Set | Reals | Fakes | AUC (95 % CI) | TPR@5 %FPR | TPR@1 %FPR | FPR at τ_0.05 | ECE (CI) | Brier | Inconclusive |
|---|---|---|---|---|---|---|---|---|---|
| `image/test` | 958 | 820 | 0.9054 (0.8905–0.9192) | 71.5 % (586/820; Wilson 68.3 %–74.4 %) | 46.2 % (379/820; Wilson 42.8 %–49.6 %) | 7.6 % (73/958; Wilson 6.1 %–9.5 %) | 0.0537 (0.0432–0.0724) | 0.1270 | 39.7 % |
| `image/test/full` | 2152 | 1843 | 0.9014 (0.8900–0.9122) | 72.2 % (1330/1843; Wilson 70.1 %–74.2 %) | 46.2 % (852/1843; Wilson 44.0 %–48.5 %) | 7.5 % (162/2152; Wilson 6.5 %–8.7 %) | 0.0519 (0.0421–0.0661) | 0.1281 | 41.5 % |
| `image/test/t2i` | 733 | 430 | 0.9426 (0.9281–0.9549) | 78.4 % (337/430; Wilson 74.2 %–82.0 %) | 60.7 % (261/430; Wilson 56.0 %–65.2 %) | 7.6 % (56/733; Wilson 5.9 %–9.8 %) | 0.0398 (0.0319–0.0615) | 0.0982 | 35.7 % |
| `image/test/full/t2i` | 2152 | 1264 | 0.9051 (0.8916–0.9177) | 73.7 % (932/1264; Wilson 71.2 %–76.1 %) | 51.0 % (645/1264; Wilson 48.3 %–53.8 %) | 7.5 % (162/2152; Wilson 6.5 %–8.7 %) | 0.0556 (0.0471–0.0719) | 0.1262 | 42.9 % |
| `image/test/edit` | 724 | 51 | 0.9168 (0.8642–0.9616) | 78.4 % (40/51; Wilson 65.4 %–87.5 %) | 64.7 % (33/51; Wilson 51.0 %–76.4 %) | 6.5 % (47/724; Wilson 4.9 %–8.5 %) | 0.0837 (0.0556–0.1454) | 0.1139 | 38.2 % |
| `image/test/full/edit` | 2152 | 153 | 0.8915 (0.8575–0.9239) | 75.2 % (115/153; Wilson 67.8 %–81.3 %) | 62.7 % (96/153; Wilson 54.9 %–70.0 %) | 7.5 % (162/2152; Wilson 6.5 %–8.7 %) | 0.1037 (0.0850–0.1396) | 0.1365 | 51.9 % |
| `image/test/personalised` | 468 | 20 | 0.9784 (0.9543–0.9961) | 90.0 % (18/20; Wilson 69.9 %–97.2 %) | 80.0 % (16/20; Wilson 58.4 %–91.9 %) | 10.5 % (49/468; Wilson 8.0 %–13.6 %) | 0.1198 (0.1076–0.1627) | 0.0843 | 49.4 % |
| `image/test/full/personalised` | 2152 | 92 | 0.9834 (0.9741–0.9911) | 93.5 % (86/92; Wilson 86.5 %–97.0 %) | 80.4 % (74/92; Wilson 71.2 %–87.3 %) | 7.5 % (162/2152; Wilson 6.5 %–8.7 %) | 0.1111 (0.0976–0.1240) | 0.0767 | 52.0 % |
| `image/test/faceswap_vs_R5` | 395 | 185 | 0.7819 (0.7384–0.8217) | 51.4 % (95/185; Wilson 44.2 %–58.5 %) | 9.2 % (17/185; Wilson 5.8 %–14.2 %) | 6.6 % (26/395; Wilson 4.5 %–9.5 %) | 0.1482 (0.1230–0.1945) | 0.2075 | 64.0 % |
| `image/test/full/faceswap_vs_R5` | 450 | 211 | 0.8139 (0.7774–0.8486) | 54.5 % (115/211; Wilson 47.8 %–61.1 %) | 11.4 % (24/211; Wilson 7.8 %–16.4 %) | 5.8 % (26/450; Wilson 4.0 %–8.3 %) | 0.1445 (0.1091–0.1787) | 0.1904 | 61.6 % |
| `image/test/gan_face_vs_R5` | 338 | 92 | 0.8662 (0.8226–0.9073) | 63.0 % (58/92; Wilson 52.8 %–72.2 %) | 8.7 % (8/92; Wilson 4.5 %–16.2 %) | 7.1 % (24/338; Wilson 4.8 %–10.3 %) | 0.1232 (0.1052–0.1661) | 0.1632 | 71.6 % |
| `image/test/full/gan_face_vs_R5` | 450 | 123 | 0.8964 (0.8636–0.9256) | 66.7 % (82/123; Wilson 57.9 %–74.4 %) | 10.6 % (13/123; Wilson 6.3 %–17.2 %) | 5.8 % (26/450; Wilson 4.0 %–8.3 %) | 0.0861 (0.0758–0.1273) | 0.1398 | 64.7 % |
| `image/test/legacy` | 154 | 72 | 0.9915 (0.9823–0.9984) | 94.4 % (68/72; Wilson 86.6 %–97.8 %) | 86.1 % (62/72; Wilson 76.3 %–92.3 %) | 1.9 % (3/154; Wilson 0.7 %–5.6 %) | 0.0202 (0.0106–0.0494) | 0.0330 | 15.9 % |
| `image/test/full/legacy` | 229 | 107 | 0.9892 (0.9801–0.9966) | 96.3 % (103/107; Wilson 90.8 %–98.5 %) | 88.8 % (95/107; Wilson 81.4 %–93.5 %) | 5.2 % (12/229; Wilson 3.0 %–8.9 %) | 0.0462 (0.0283–0.0660) | 0.0446 | 16.1 % |
| `video/test` | 162 | 135 | 0.9613 (0.9374–0.9806) | 80.7 % (109/135; Wilson 73.3 %–86.5 %) | 0.0 % (0/135; Wilson 0.0 %–2.8 %; rule of three ≤ 2.2 %) | 2.5 % (4/162; Wilson 1.0 %–6.2 %) | 0.0495 (0.0441–0.0881) | 0.0762 | 50.2 % |
| `video/test/real` | 162 | 0 | – (–) | – | – | 2.5 % (4/162; Wilson 1.0 %–6.2 %) | – (–) | – | 45.7 % |
| `audio/test` | 262 | 444 | 0.8474 (0.8070–0.8852) | 18.9 % (84/444; Wilson 15.5 %–22.8 %) | 3.6 % (16/444; Wilson 2.2 %–5.8 %) | 2.3 % (6/262; Wilson 1.1 %–4.9 %) | 0.0775 (0.0702–0.1260) | 0.1662 | 72.8 % |
| `audio/test/real` | 262 | 0 | – (–) | – | – | 2.3 % (6/262; Wilson 1.1 %–4.9 %) | – (–) | – | 62.0 % |
| `audio/test/asvspoof5` | 86 | 90 | 0.8786 (0.8221–0.9283) | 38.9 % (35/90; Wilson 29.5 %–49.2 %) | 2.2 % (2/90; Wilson 0.6 %–7.7 %) | 4.7 % (4/86; Wilson 1.8 %–11.4 %) | 0.2047 (0.1640–0.2412) | 0.1760 | 73.3 % |
| `audio/test/in_the_wild` | 94 | 89 | 0.9081 (0.8126–0.9782) | 1.1 % (1/89; Wilson 0.2 %–6.1 %) | 1.1 % (1/89; Wilson 0.2 %–6.1 %) | 1.1 % (1/94; Wilson 0.2 %–5.8 %) | 0.1566 (0.1206–0.2467) | 0.1412 | 81.6 % |

Conformal coverage at α = 0.05: reals 91.5 % (877/958; Wilson 89.6 %–93.1 %), AI 93.9 % (770/820; Wilson 92.1 %–95.3 %).

## Real strata (FPR at τ)

| Stratum | n | FPR at τ_0.05 | FPR at τ_0.01 |
|---|---|---|---|
| R1 | 387 | 3.6 % (14/387; Wilson 2.2 %–6.0 %) | 0.5 % (2/387; Wilson 0.1 %–1.9 %) |
| R2 | 409 | 6.1 % (25/409; Wilson 4.2 %–8.9 %) | 0.2 % (1/409; Wilson 0.0 %–1.4 %) |
| R3 | 479 | 11.7 % (56/479; Wilson 9.1 %–14.9 %) | 1.7 % (8/479; Wilson 0.8 %–3.3 %) |
| R4 | 405 | 9.1 % (37/405; Wilson 6.7 %–12.3 %) | 3.7 % (15/405; Wilson 2.3 %–6.0 %) |
| R5 | 450 | 5.8 % (26/450; Wilson 4.0 %–8.3 %) | 0.7 % (3/450; Wilson 0.2 %–1.9 %) |
| R6 | 22 | 18.2 % (4/22; Wilson 7.3 %–38.5 %) | 0.0 % (0/22; Wilson 0.0 %–14.9 %; rule of three ≤ 13.6 %) |

## Real strata by min-side bucket (FPR at τ_0.05)

| Stratum | Min side (px) | n | FPR at τ_0.05 |
|---|---|---|---|
| R1 | 1025-1536 | 22 | 0.0 % (0/22; Wilson 0.0 %–14.9 %; rule of three ≤ 13.6 %) |
| R1 | 384-767 | 93 | 3.2 % (3/93; Wilson 1.1 %–9.1 %) |
| R1 | 768-1024 | 44 | 11.4 % (5/44; Wilson 5.0 %–24.0 %) |
| R1 | >1536 | 228 | 2.6 % (6/228; Wilson 1.2 %–5.6 %) |
| R2 | 1025-1536 | 15 | 6.7 % (1/15; Wilson 1.2 %–29.8 %) |
| R2 | 384-767 | 11 | 9.1 % (1/11; Wilson 1.6 %–37.7 %) |
| R2 | >1536 | 383 | 6.0 % (23/383; Wilson 4.0 %–8.8 %) |
| R3 | 1025-1536 | 135 | 9.6 % (13/135; Wilson 5.7 %–15.8 %) |
| R3 | 384-767 | 123 | 10.6 % (13/123; Wilson 6.3 %–17.2 %) |
| R3 | 768-1024 | 95 | 15.8 % (15/95; Wilson 9.8 %–24.4 %) |
| R3 | <384 | 6 | 33.3 % (2/6; Wilson 9.7 %–70.0 %) |
| R3 | >1536 | 120 | 10.8 % (13/120; Wilson 6.4 %–17.7 %) |
| R4 | 1025-1536 | 79 | 15.2 % (12/79; Wilson 8.9 %–24.7 %) |
| R4 | 384-767 | 145 | 2.8 % (4/145; Wilson 1.1 %–6.9 %) |
| R4 | <384 | 172 | 11.6 % (20/172; Wilson 7.7 %–17.3 %) |
| R4 | >1536 | 9 | 11.1 % (1/9; Wilson 2.0 %–43.5 %) |
| R5 | 384-767 | 130 | 2.3 % (3/130; Wilson 0.8 %–6.6 %) |
| R5 | 768-1024 | 75 | 2.7 % (2/75; Wilson 0.7 %–9.2 %) |
| R5 | <384 | 245 | 8.6 % (21/245; Wilson 5.7 %–12.7 %) |
| R6 | 384-767 | 5 | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) |
| R6 | 768-1024 | 3 | 33.3 % (1/3; Wilson 6.1 %–79.2 %) |
| R6 | >1536 | 14 | 21.4 % (3/14; Wilson 7.6 %–47.6 %) |

## Fake families (recall at τ)

| Family | Threat | LGO fold | n | TPR at τ_0.05 | TPR at τ_0.01 |
|---|---|---|---|---|---|
| faceswap-stills | faceswap | – | 211 | 54.5 % (115/211; Wilson 47.8 %–61.1 %) | 11.4 % (24/211; Wilson 7.8 %–16.4 %) |
| flux1 | t2i | – | 60 | 66.7 % (40/60; Wilson 54.1 %–77.3 %) | 48.3 % (29/60; Wilson 36.2 %–60.7 %) |
| flux2 | t2i | F2 | 64 | 35.9 % (23/64; Wilson 25.3 %–48.2 %) | 15.6 % (10/64; Wilson 8.7 %–26.4 %) |
| gan-faces | gan_face | – | 123 | 66.7 % (82/123; Wilson 57.9 %–74.4 %) | 10.6 % (13/123; Wilson 6.3 %–17.2 %) |
| gpt-4o-image-1 | t2i | – | 30 | 40.0 % (12/30; Wilson 24.6 %–57.7 %) | 6.7 % (2/30; Wilson 1.8 %–21.3 %) |
| gpt-image-1.5-2 | t2i | F3 | 50 | 32.0 % (16/50; Wilson 20.8 %–45.8 %) | 8.0 % (4/50; Wilson 3.2 %–18.8 %) |
| hidream | t2i | – | 47 | 80.9 % (38/47; Wilson 67.5 %–89.6 %) | 55.3 % (26/47; Wilson 41.2 %–68.6 %) |
| hunyuanimage | t2i | – | 20 | 95.0 % (19/20; Wilson 76.4 %–99.1 %) | 90.0 % (18/20; Wilson 69.9 %–97.2 %) |
| ideogram | t2i | – | 38 | 76.3 % (29/38; Wilson 60.8 %–87.0 %) | 65.8 % (25/38; Wilson 49.9 %–78.8 %) |
| imagen3 | t2i | – | 49 | 75.5 % (37/49; Wilson 61.9 %–85.4 %) | 28.6 % (14/49; Wilson 17.8 %–42.4 %) |
| imagen4 | t2i | F7 | 72 | 79.2 % (57/72; Wilson 68.4 %–86.9 %) | 47.2 % (34/72; Wilson 36.1 %–58.6 %) |
| legacy | t2i | – | 107 | 96.3 % (103/107; Wilson 90.8 %–98.5 %) | 88.8 % (95/107; Wilson 81.4 %–93.5 %) |
| long-tail | t2i | – | 159 | 77.4 % (123/159; Wilson 70.3 %–83.2 %) | 58.5 % (93/159; Wilson 50.7 %–65.9 %) |
| midjourney | t2i | F1 | 124 | 91.9 % (114/124; Wilson 85.8 %–95.6 %) | 80.6 % (100/124; Wilson 72.8 %–86.6 %) |
| modern-t2i-faces | t2i | – | 89 | 69.7 % (62/89; Wilson 59.5 %–78.2 %) | 40.4 % (36/89; Wilson 30.9 %–50.8 %) |
| nano-banana | t2i | F4 | 80 | 62.5 % (50/80; Wilson 51.5 %–72.3 %) | 27.5 % (22/80; Wilson 18.9 %–38.1 %) |
| partial-edits | edit | – | 153 | 75.2 % (115/153; Wilson 67.8 %–81.3 %) | 62.7 % (96/153; Wilson 54.9 %–70.0 %) |
| personalised | personalised | – | 92 | 93.5 % (86/92; Wilson 86.5 %–97.0 %) | 80.4 % (74/92; Wilson 71.2 %–87.3 %) |
| qwen-image | t2i | F6 | 84 | 51.2 % (43/84; Wilson 40.7 %–61.6 %) | 17.9 % (15/84; Wilson 11.1 %–27.4 %) |
| recraft | t2i | – | 36 | 88.9 % (32/36; Wilson 74.7 %–95.6 %) | 50.0 % (18/36; Wilson 34.5 %–65.5 %) |
| sd3 | t2i | – | 63 | 98.4 % (62/63; Wilson 91.5 %–99.7 %) | 95.2 % (60/63; Wilson 86.9 %–98.4 %) |
| seedream3 | t2i | – | 26 | 92.3 % (24/26; Wilson 75.9 %–97.9 %) | 73.1 % (19/26; Wilson 53.9 %–86.3 %) |
| seedream4 | t2i | F5 | 66 | 72.7 % (48/66; Wilson 61.0 %–82.0 %) | 37.9 % (25/66; Wilson 27.1 %–49.9 %) |

## Leave-generator-out folds

| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC | ECE |
|---|---|---|---|---|---|---|
| F1 | midjourney | 32 | 248 | 81.2 % (26/32; Wilson 64.7 %–91.1 %) | 0.9292 | 0.1066 |
| F2 | flux2 | 15 | 233 | 40.0 % (6/15; Wilson 19.8 %–64.3 %) | 0.6652 | 0.2914 |
| F3 | gpt-image-1.5-2 | 28 | 569 | 25.0 % (7/28; Wilson 12.7 %–43.4 %) | 0.7224 | 0.2090 |
| F4 | nano-banana | 34 | 369 | 64.7 % (22/34; Wilson 47.9 %–78.5 %) | 0.8738 | 0.0907 |
| F5 | seedream4 | 66 | 969 | 83.3 % (55/66; Wilson 72.6 %–90.4 %) | 0.9643 | 0.0794 |
| F6 | qwen-image | 19 | 217 | 42.1 % (8/19; Wilson 23.1 %–63.7 %) | 0.7507 | 0.2254 |
| F7 | imagen4 | 18 | 217 | 72.2 % (13/18; Wilson 49.1 %–87.5 %) | 0.9017 | 0.1532 |

Macro TPR at τ_0.05 over F1–F7: 0.5837; minimum 0.2500; macro ECE 0.1651.

## Robustness (fixed τ_0.05)

Reference `native`/`none` on 562 of 562 robustness items.

| Transform | Reals | Fakes | ΔTPR | ΔFPR | ΔAUC | Inconclusive under t |
|---|---|---|---|---|---|---|
| jpeg75 | 298 | 264 | 0.1667 | -0.0134 | 0.0630 | 51.4 % |
| jpeg50 | 298 | 264 | 0.2803 | -0.0201 | 0.1172 | 100.0 % |
| webp80 | 298 | 264 | 0.1250 | 0.0101 | 0.0395 | 41.8 % |
| resize0.5 | 298 | 264 | 0.2348 | 0.0302 | 0.1542 | 38.4 % |
| crop70 | 298 | 264 | 0.0417 | 0.0201 | 0.0299 | 36.3 % |
| screenshot_jpeg | 298 | 264 | 0.1326 | 0.2114 | 0.1478 | 94.1 % |
| social_ig | 298 | 264 | 0.1667 | -0.0067 | 0.0680 | 39.1 % |
| social_x | 298 | 264 | 0.1212 | -0.0101 | 0.0405 | 48.8 % |
| noise3 | 298 | 264 | -0.0076 | 0.2282 | 0.0668 | 33.6 % |
| upscale2x | 289 | 263 | 0.0000 | 0.0104 | -0.0082 | 37.0 % |
| exif_forge | 298 | 240 | 0.0000 | -0.0034 | -0.0004 | 39.0 % |

Not scored: jpeg95, screenshot, launder, rephoto_proxy.
- jpeg95: no cached rows for img.dda
- screenshot: no cached rows for img.dda
- launder: no cached rows for img.dda
- rephoto_proxy: no cached rows for img.dda

## Decision rules (D-64, D-65, D-76, D-77)

Content the bench does not hold (D-65), image system: applied: flat-block limit 0.5346 of vb-1.1, measured on the scored bytes. A capped decision is one answer in the set at α = 0.05, inside the slice envelope, that the condition makes inconclusive; an item under two conditions is counted under each. A strong code needs two checks that scored (D-64): the decisions of this report are at α = 0.05, so that rule changes none of them; the columns “Fewer than two checks” and “Strong code not issued” count the items it concerns. A decision needs a check that supports it (D-76): an item whose set holds one answer while no check gives at least weak support for that answer is not decided, and neither is an item in which no check entered. A decision also needs the estimate itself in its answer’s part of the scale (D-77: 34 or less for the clean answer, 65 or more for the AI answer, in whole points), and checks that together give the answer at least weak support (D-76 as amended). The conditions are asked in that order, D-77 first, and an item is counted under the one that is its result’s reason; the last four columns count them as a content condition is counted.

| Set | Scored | Flat blocks | Screenshot rule | Greyscale | Any content condition | Not measured | Fewer than two checks | Strong code not issued | No check supports the answer | No check entered | Estimate outside its answer’s part of the scale | Checks together do not support the answer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `image/test` | 1778 | 9 (7 real, 2 AI); 7 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 29 (28 real, 1 AI); 17 decisions capped | 37 (34 real, 3 AI); 23 decisions capped | 0 | 184 (88 real, 96 AI) | 0 (0 real, 0 AI) | 43 (16 real, 27 AI); 39 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 49 (41 real, 8 AI); 46 decisions capped | 21 (8 real, 13 AI); 19 decisions capped |
| `image/test/full` | 3995 | 24 (13 real, 11 AI); 19 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 45 (44 real, 1 AI); 23 decisions capped | 68 (56 real, 12 AI); 41 decisions capped | 0 | 236 (140 real, 96 AI) | 0 (0 real, 0 AI) | 72 (36 real, 36 AI); 62 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 78 (64 real, 14 AI); 75 decisions capped | 30 (13 real, 17 AI); 28 decisions capped |
| `image/test/t2i` | 1163 | 7 (6 real, 1 AI); 5 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 26 (26 real, 0 AI); 14 decisions capped | 33 (32 real, 1 AI); 19 decisions capped | 0 | 9 (9 real, 0 AI) | 0 (0 real, 0 AI) | 9 (8 real, 1 AI); 8 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 12 (10 real, 2 AI); 11 decisions capped | 6 (5 real, 1 AI); 5 decisions capped |
| `image/test/full/t2i` | 3416 | 21 (13 real, 8 AI); 17 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 44 (44 real, 0 AI); 22 decisions capped | 64 (56 real, 8 AI); 38 decisions capped | 0 | 140 (140 real, 0 AI) | 0 (0 real, 0 AI) | 43 (36 real, 7 AI); 33 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 69 (64 real, 5 AI); 66 decisions capped | 16 (13 real, 3 AI); 15 decisions capped |
| `image/test/edit` | 775 | 6 (5 real, 1 AI); 4 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 30 (30 real, 0 AI); 19 decisions capped | 35 (34 real, 1 AI); 22 decisions capped | 0 | 0 (0 real, 0 AI) | 0 (0 real, 0 AI) | 10 (9 real, 1 AI); 8 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 2 (2 real, 0 AI); 2 decisions capped | 4 (4 real, 0 AI); 4 decisions capped |
| `image/test/full/edit` | 2305 | 16 (13 real, 3 AI); 12 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 44 (44 real, 0 AI); 22 decisions capped | 59 (56 real, 3 AI); 33 decisions capped | 0 | 140 (140 real, 0 AI) | 0 (0 real, 0 AI) | 37 (36 real, 1 AI); 27 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 65 (64 real, 1 AI); 62 decisions capped | 13 (13 real, 0 AI); 12 decisions capped |
| `image/test/personalised` | 488 | 4 (4 real, 0 AI); 3 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 7 (7 real, 0 AI); 3 decisions capped | 11 (11 real, 0 AI); 6 decisions capped | 0 | 0 (0 real, 0 AI) | 0 (0 real, 0 AI) | 6 (6 real, 0 AI); 6 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 2 (2 real, 0 AI); 2 decisions capped | 2 (1 real, 1 AI); 2 decisions capped |
| `image/test/full/personalised` | 2244 | 13 (13 real, 0 AI); 10 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 44 (44 real, 0 AI); 22 decisions capped | 56 (56 real, 0 AI); 31 decisions capped | 0 | 140 (140 real, 0 AI) | 0 (0 real, 0 AI) | 37 (36 real, 1 AI); 27 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 64 (64 real, 0 AI); 61 decisions capped | 14 (13 real, 1 AI); 13 decisions capped |
| `image/test/faceswap_vs_R5` | 580 | 1 (1 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 16 (16 real, 0 AI); 7 decisions capped | 17 (17 real, 0 AI); 7 decisions capped | 0 | 210 (132 real, 78 AI) | 0 (0 real, 0 AI) | 33 (14 real, 19 AI); 28 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 43 (41 real, 2 AI); 42 decisions capped | 5 (2 real, 3 AI); 4 decisions capped |
| `image/test/full/faceswap_vs_R5` | 661 | 1 (1 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 21 (21 real, 0 AI); 9 decisions capped | 22 (22 real, 0 AI); 9 decisions capped | 0 | 212 (132 real, 80 AI) | 0 (0 real, 0 AI) | 35 (16 real, 19 AI); 29 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 43 (41 real, 2 AI); 42 decisions capped | 7 (3 real, 4 AI); 6 decisions capped |
| `image/test/gan_face_vs_R5` | 430 | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 10 (9 real, 1 AI); 5 decisions capped | 10 (9 real, 1 AI); 5 decisions capped | 0 | 145 (129 real, 16 AI) | 0 (0 real, 0 AI) | 20 (13 real, 7 AI); 16 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 45 (40 real, 5 AI); 44 decisions capped | 11 (2 real, 9 AI); 9 decisions capped |
| `image/test/full/gan_face_vs_R5` | 573 | 1 (1 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 22 (21 real, 1 AI); 10 decisions capped | 23 (22 real, 1 AI); 10 decisions capped | 0 | 148 (132 real, 16 AI) | 0 (0 real, 0 AI) | 24 (16 real, 8 AI); 18 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 47 (41 real, 6 AI); 46 decisions capped | 12 (3 real, 9 AI); 10 decisions capped |
| `image/test/legacy` | 226 | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 3 (3 real, 0 AI); 3 decisions capped | 3 (3 real, 0 AI); 3 decisions capped | 0 | 3 (3 real, 0 AI) | 0 (0 real, 0 AI) | 2 (2 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 5 (5 real, 0 AI); 3 decisions capped | 2 (2 real, 0 AI); 2 decisions capped |
| `image/test/full/legacy` | 336 | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 3 (3 real, 0 AI); 3 decisions capped | 3 (3 real, 0 AI); 3 decisions capped | 0 | 8 (8 real, 0 AI) | 0 (0 real, 0 AI) | 5 (5 real, 0 AI); 1 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 5 (5 real, 0 AI); 3 decisions capped | 4 (4 real, 0 AI); 4 decisions capped |
| `video/test` | 297 | – | – | – | – | – | 1 (1 real, 0 AI) | 0 (0 real, 0 AI) | 5 (0 real, 5 AI); 4 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 2 (0 real, 2 AI); 1 decisions capped |
| `video/test/real` | 162 | – | – | – | – | – | 1 (1 real, 0 AI) | 0 (0 real, 0 AI) | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped |
| `audio/test` | 706 | – | – | – | – | – | 706 (262 real, 444 AI) | 69 (53 real, 16 AI) | 0 (0 real, 0 AI); 0 decisions capped | 1 (1 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped |
| `audio/test/real` | 262 | – | – | – | – | – | 262 (262 real, 0 AI) | 53 (53 real, 0 AI) | 0 (0 real, 0 AI); 0 decisions capped | 1 (1 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped |
| `audio/test/asvspoof5` | 176 | – | – | – | – | – | 176 (86 real, 90 AI) | 5 (3 real, 2 AI) | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped |
| `audio/test/in_the_wild` | 183 | – | – | – | – | – | 183 (94 real, 89 AI) | 13 (12 real, 1 AI) | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped |
| robustness: jpeg75 | 562 | 5 (3 real, 2 AI); 2 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 5 (5 real, 0 AI); 5 decisions capped | 10 (8 real, 2 AI); 7 decisions capped | 0 | 25 (17 real, 8 AI) | 0 (0 real, 0 AI) | 12 (5 real, 7 AI); 10 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 9 (9 real, 0 AI); 9 decisions capped | 1 (1 real, 0 AI); 1 decisions capped |
| robustness: jpeg50 | 562 | 7 (2 real, 5 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 8 (6 real, 2 AI); 0 decisions capped | 13 (7 real, 6 AI); 0 decisions capped | 0 | 26 (18 real, 8 AI) | 0 (0 real, 0 AI) | 27 (14 real, 13 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 23 (14 real, 9 AI); 0 decisions capped | 27 (12 real, 15 AI); 0 decisions capped |
| robustness: webp80 | 562 | 3 (2 real, 1 AI); 3 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 3 (2 real, 1 AI); 3 decisions capped | 0 | 24 (17 real, 7 AI) | 0 (0 real, 0 AI) | 14 (7 real, 7 AI); 14 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 10 (9 real, 1 AI); 10 decisions capped | 2 (2 real, 0 AI); 2 decisions capped |
| robustness: resize0.5 | 562 | 2 (2 real, 0 AI); 2 decisions capped | 2 (0 real, 2 AI); 2 decisions capped | 5 (5 real, 0 AI); 3 decisions capped | 9 (7 real, 2 AI); 7 decisions capped | 0 | 42 (33 real, 9 AI) | 0 (0 real, 0 AI) | 9 (7 real, 2 AI); 9 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 29 (26 real, 3 AI); 29 decisions capped | 7 (2 real, 5 AI); 7 decisions capped |
| robustness: crop70 | 562 | 2 (2 real, 0 AI); 0 decisions capped | 1 (1 real, 0 AI); 0 decisions capped | 6 (6 real, 0 AI); 2 decisions capped | 8 (8 real, 0 AI); 2 decisions capped | 0 | 24 (19 real, 5 AI) | 0 (0 real, 0 AI) | 16 (14 real, 2 AI); 16 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 22 (21 real, 1 AI); 22 decisions capped | 6 (2 real, 4 AI); 6 decisions capped |
| robustness: screenshot_jpeg | 562 | 302 (189 real, 113 AI); 44 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 5 (5 real, 0 AI); 0 decisions capped | 305 (192 real, 113 AI); 44 decisions capped | 0 | 0 (0 real, 0 AI) | 0 (0 real, 0 AI) | 16 (13 real, 3 AI); 5 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 3 (3 real, 0 AI); 0 decisions capped | 35 (27 real, 8 AI); 2 decisions capped |
| robustness: social_ig | 562 | 5 (3 real, 2 AI); 3 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 5 (5 real, 0 AI); 5 decisions capped | 10 (8 real, 2 AI); 8 decisions capped | 0 | 24 (16 real, 8 AI) | 0 (0 real, 0 AI) | 9 (3 real, 6 AI); 9 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 8 (8 real, 0 AI); 8 decisions capped | 2 (2 real, 0 AI); 2 decisions capped |
| robustness: social_x | 562 | 4 (3 real, 1 AI); 1 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 5 (5 real, 0 AI); 5 decisions capped | 9 (8 real, 1 AI); 6 decisions capped | 0 | 25 (17 real, 8 AI) | 0 (0 real, 0 AI) | 11 (6 real, 5 AI); 10 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 11 (11 real, 0 AI); 11 decisions capped | 3 (1 real, 2 AI); 3 decisions capped |
| robustness: noise3 | 562 | 0 (0 real, 0 AI); 0 decisions capped | 8 (1 real, 7 AI); 8 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 8 (1 real, 7 AI); 8 decisions capped | 0 | 26 (18 real, 8 AI) | 0 (0 real, 0 AI) | 7 (7 real, 0 AI); 7 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 14 (13 real, 1 AI); 14 decisions capped | 6 (4 real, 2 AI); 6 decisions capped |
| robustness: upscale2x | 552 | 3 (3 real, 0 AI); 2 decisions capped | 8 (1 real, 7 AI); 7 decisions capped | 5 (5 real, 0 AI); 0 decisions capped | 16 (9 real, 7 AI); 9 decisions capped | 0 | 0 (0 real, 0 AI) | 0 (0 real, 0 AI) | 6 (3 real, 3 AI); 6 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 1 (1 real, 0 AI); 1 decisions capped | 1 (0 real, 1 AI); 1 decisions capped |
| robustness: exif_forge | 538 | 3 (3 real, 0 AI); 1 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 5 (5 real, 0 AI); 5 decisions capped | 8 (8 real, 0 AI); 6 decisions capped | 0 | 26 (18 real, 8 AI) | 0 (0 real, 0 AI) | 5 (2 real, 3 AI); 3 decisions capped | 0 (0 real, 0 AI); 0 decisions capped | 13 (13 real, 0 AI); 13 decisions capped | 1 (1 real, 0 AI); 1 decisions capped |

## Invariants

On 1000 items: max |Σφ′ + b′ − logit s| = 7.105e-15; 0 of 1000 random +Δz moves lowered s.

## Slices

12 slices; 2 measured (≥ 100 items per class), 10 merged with a measured parent slice, 0 not measured.

| Slice | Reals | Fakes | AUC | Measured at |
|---|---|---|---|---|
| `image/q>=90/side512-1024/noface` | 305 | 831 | 0.9233 | `image/q>=90/side512-1024/noface` |
| `image/q>=90/side<512/face120-256` | 282 | 184 | 0.8711 | `image/q>=90/side<512/face120-256` |
| `image/q>=90/side512-1024/face120-256` | 36 | 211 | 0.9069 | `image/q>=90/side512-1024` |
| `image/q>=90/side512-1024/face<120` | 40 | 135 | 0.9069 | `image/q>=90/side512-1024` |
| `image/q>=90/side512-1024/face>=256` | 76 | 306 | 0.9069 | `image/q>=90/side512-1024` |
| `image/q>=90/side<512/face<120` | 92 | 14 | 0.9179 | `image/q>=90/side<512` |
| `image/q>=90/side<512/face>=256` | 17 | 0 | 0.9179 | `image/q>=90/side<512` |
| `image/q>=90/side<512/noface` | 299 | 62 | 0.9179 | `image/q>=90/side<512` |
| `image/q>=90/side>1024/face120-256` | 100 | 13 | 0.9210 | `image/q>=90/side>1024` |
| `image/q>=90/side>1024/face<120` | 44 | 5 | 0.9210 | `image/q>=90/side>1024` |
| `image/q>=90/side>1024/face>=256` | 180 | 23 | 0.9210 | `image/q>=90/side>1024` |
| `image/q>=90/side>1024/noface` | 681 | 59 | 0.9210 | `image/q>=90/side>1024` |

## Field conditions (EA-02 slice validation map)

| Condition | Reals | Fakes | AUC | Measured at |
|---|---|---|---|---|
| JPEG q ≥ 90, ≥ 1 MP, no face | 489 | 205 | 0.9137 | `jpeg_q>=90_1mp_noface` |
| JPEG q 70–89 | 751 | 298 | 0.9051 | `jpeg_q70-89` |
| PNG / lossless (no JPEG quantisation tables) | 365 | 1116 | 0.8705 | `lossless` |
| face present | 867 | 891 | 0.8742 | `face` |
| social recompressed (social_ig, social_x) | 596 | 528 | 0.8886 | `social` |
| screenshot (screenshot, screenshot_jpeg) | 298 | 264 | 0.7950 | `screenshot` |

## Reproducibility

Two `calibrate` runs identical except `created_at`: **True** (fuser-image-2.1.1.json vs fuser-image-2.1.1.json).

## Video and audio

- **video:** system `fuser-video-2.1.1.json`; variant `native`; τ_0.05 = 0.7330 (n = 74); τ_0.01 = – (n = 74); 162 reals and 135 fakes scored, 0 without a score, 0 capped without an estimate because a check did not run (cal reals 0); excluded {}.
  Runtime slices: 19 leaves, 5 inside the envelope (AUC ≥ 0.70 at ≥ 100 items per class directly or pooled; both classes present in the leaf): `video/res480-719/dur4-60s/face120-256` 0/5 → 0.9613 at `video`; `video/res480-719/dur4-60s/face<120` 0/2 → 0.9613 at `video`; `video/res480-719/dur4-60s/face>=256` 0/9 → 0.9613 at `video`; `video/res480-719/dur4-60s/noface` 0/25 → 0.9613 at `video`; `video/res480-719/dur<4s/face120-256` 0/3 → 0.9613 at `video`; `video/res480-719/dur<4s/face>=256` 0/1 → 0.9613 at `video`; `video/res480-719/dur<4s/noface` 0/13 → 0.9613 at `video`; `video/res<480/dur4-60s/face>=256` 1/0 → 0.9613 at `video`; `video/res<480/dur4-60s/noface` 1/0 → 0.9613 at `video`; `video/res<480/dur>60s/face120-256` 1/0 → 0.9613 at `video`; `video/res>=720/dur4-60s/face120-256` 18/12 → 0.9613 at `video`; `video/res>=720/dur4-60s/face<120` 8/5 → 0.9613 at `video`; `video/res>=720/dur4-60s/face>=256` 88/16 → 0.9613 at `video`; `video/res>=720/dur4-60s/noface` 21/43 → 0.9613 at `video`; `video/res>=720/dur<4s/face120-256` 6/0 → 0.9613 at `video`; `video/res>=720/dur<4s/face>=256` 7/0 → 0.9613 at `video`; `video/res>=720/dur<4s/noface` 7/0 → 0.9613 at `video`; `video/res>=720/dur>60s/face120-256` 3/0 → 0.9613 at `video`; `video/res>=720/dur>60s/face>=256` 1/1 → 0.9613 at `video`.
- **audio:** system `fuser-audio-2.1.1.json`; variant `native`; τ_0.05 = 0.8038 (n = 113); τ_0.01 = 0.8085 (n = 113); 262 reals and 444 fakes scored, 3 without a score, 6 capped without an estimate because a check did not run (cal reals 5); excluded {}.
  Runtime slices: 2 leaves, 2 inside the envelope (AUC ≥ 0.70 at ≥ 100 items per class directly or pooled; both classes present in the leaf): `audio/dur4-60s` 150/275 → 0.7840 at `audio/dur4-60s`; `audio/dur<4s` 112/169 → 0.9233 at `audio/dur<4s`.

| Audio set | EER |
|---|---|
| `audio/test` | 24.4 % |
| `audio/test/asvspoof5` | 17.4 % |
| `audio/test/in_the_wild` | 16.9 % |

`video/lgo`:

| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC |
|---|---|---|---|---|---|
| V-Kling | kling | 19 | 162 | 84.2 % (16/19; Wilson 62.4 %–94.5 %) | 0.9789 |
| V-Seedance | seedance | 19 | 162 | 36.8 % (7/19; Wilson 19.1 %–59.0 %) | 0.8268 |
| V-Sora | sora | 19 | 162 | 21.1 % (4/19; Wilson 8.5 %–43.3 %) | 0.8450 |
| V-Veo | veo | 18 | 162 | 55.6 % (10/18; Wilson 33.7 %–75.4 %) | 0.9530 |
| V-WanHunyuan | wan-hunyuan | 18 | 162 | 100.0 % (18/18; Wilson 82.4 %–100.0 %) | 0.9983 |

Macro TPR at τ_0.05 over V-Veo, V-Sora, V-Kling, V-Seedance, V-WanHunyuan: 0.5953; minimum 0.2105.

## Not measured

