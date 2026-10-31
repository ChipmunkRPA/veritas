# Veritas-Bench results: vb-1.0-core · `test`

Generated 2026-09-14T02:18:42Z by `backend/scripts/vbench_report.py` (git d46efee3c655d35c6000cb7e065021914890ee70).

- **System:** score key `v1.image_fused@107574c6b0fa`.
- **Manifest:** sha256 `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`; variant `norm`.
- **Thresholds from `cal` reals** (equal stratum weights): τ_0.05 = 0.6701 (n = 110); τ_0.01 = 0.8522 (n = 110).
- **Counts:** 344 reals and 544 fakes scored; 0 items without a score; excluded {"cal": {"c2pa_manifest": 39}, "fit": {"c2pa_manifest": 55}, "test": {"c2pa_manifest": 88}}.

## Sets

| Set | Reals | Fakes | AUC (95 % CI) | TPR@5 %FPR | TPR@1 %FPR | FPR at τ_0.05 | ECE (CI) | Brier | Inconclusive |
|---|---|---|---|---|---|---|---|---|---|
| `image/test` | 344 | 544 | 0.8116 (0.7731–0.8465) | 34.4 % (187/544; Wilson 30.5 %–38.5 %) | 7.0 % (38/544; Wilson 5.1 %–9.4 %) | 8.1 % (28/344; Wilson 5.7 %–11.5 %) | 0.1072 (0.0982–0.1338) | 0.1909 | – |
| `image/test/t2i` | 344 | 381 | 0.8155 (0.7768–0.8514) | 35.2 % (134/381; Wilson 30.5 %–40.1 %) | 5.5 % (21/381; Wilson 3.6 %–8.3 %) | 8.1 % (28/344; Wilson 5.7 %–11.5 %) | 0.1051 (0.0968–0.1371) | 0.1883 | – |
| `image/test/edit` | 344 | 40 | 0.7123 (0.6269–0.7991) | 15.0 % (6/40; Wilson 7.1 %–29.1 %) | 5.0 % (2/40; Wilson 1.4 %–16.5 %) | 8.1 % (28/344; Wilson 5.7 %–11.5 %) | 0.1932 (0.1608–0.2730) | 0.2501 | – |
| `image/test/personalised` | 344 | 27 | 0.8892 (0.8195–0.9446) | 59.3 % (16/27; Wilson 40.7 %–75.5 %) | 22.2 % (6/27; Wilson 10.6 %–40.8 %) | 8.1 % (28/344; Wilson 5.7 %–11.5 %) | 0.1091 (0.1007–0.1977) | 0.1359 | – |
| `image/test/faceswap_vs_R5` | 87 | 60 | 0.8067 (0.7313–0.8756) | 38.3 % (23/60; Wilson 27.1 %–51.0 %) | 8.3 % (5/60; Wilson 3.6 %–18.1 %) | 6.9 % (6/87; Wilson 3.2 %–14.2 %) | 0.1151 (0.1021–0.1998) | 0.1851 | – |
| `image/test/gan_face_vs_R5` | 87 | 36 | 0.7436 (0.6479–0.8307) | 22.2 % (8/36; Wilson 11.7 %–38.1 %) | 11.1 % (4/36; Wilson 4.4 %–25.3 %) | 6.9 % (6/87; Wilson 3.2 %–14.2 %) | 0.1903 (0.1443–0.2722) | 0.2279 | – |
| `image/test/legacy` | 58 | 32 | 0.9359 (0.8810–0.9747) | 31.2 % (10/32; Wilson 18.0 %–48.6 %) | 0.0 % (0/32; Wilson 0.0 %–10.7 %; rule of three ≤ 9.4 %) | 0.0 % (0/58; Wilson 0.0 %–6.2 %; rule of three ≤ 5.2 %) | 0.2555 (0.2150–0.3063) | 0.1863 | – |
| `video/test` | 38 | 32 | 0.6933 (0.5663–0.8167) | 0.0 % (0/32; Wilson 0.0 %–10.7 %; rule of three ≤ 9.4 %) | 0.0 % (0/32; Wilson 0.0 %–10.7 %; rule of three ≤ 9.4 %) | 0.0 % (0/38; Wilson 0.0 %–9.2 %; rule of three ≤ 7.9 %) | 0.1964 (0.1826–0.3226) | 0.2445 | – |
| `video/test/real` | 38 | 0 | – (–) | – | – | 0.0 % (0/38; Wilson 0.0 %–9.2 %; rule of three ≤ 7.9 %) | – (–) | – | – |
| `audio/test` | 74 | 76 | 0.8637 (0.7908–0.9247) | 5.3 % (4/76; Wilson 2.1 %–12.8 %) | 0.0 % (0/76; Wilson 0.0 %–4.8 %; rule of three ≤ 3.9 %) | 1.4 % (1/74; Wilson 0.2 %–7.3 %) | 0.1278 (0.0975–0.2064) | 0.1607 | – |
| `audio/test/real` | 74 | 0 | – (–) | – | – | 1.4 % (1/74; Wilson 0.2 %–7.3 %) | – (–) | – | – |
| `audio/test/asvspoof5` | 17 | 19 | 0.8204 (0.6505–0.9487) | 5.3 % (1/19; Wilson 0.9 %–24.6 %) | 0.0 % (0/19; Wilson 0.0 %–16.8 %; rule of three ≤ 15.8 %) | 0.0 % (0/17; Wilson 0.0 %–18.4 %; rule of three ≤ 17.6 %) | 0.3208 (0.2563–0.4273) | 0.2512 | – |
| `audio/test/in_the_wild` | 37 | 19 | 0.8634 (0.7160–0.9833) | 0.0 % (0/19; Wilson 0.0 %–16.8 %; rule of three ≤ 15.8 %) | 0.0 % (0/19; Wilson 0.0 %–16.8 %; rule of three ≤ 15.8 %) | 2.7 % (1/37; Wilson 0.5 %–13.8 %) | 0.1370 (0.1075–0.2894) | 0.1451 | – |

## Real strata (FPR at τ)

| Stratum | n | FPR at τ_0.05 | FPR at τ_0.01 |
|---|---|---|---|
| R1 | 63 | 17.5 % (11/63; Wilson 10.0 %–28.6 %) | 4.8 % (3/63; Wilson 1.6 %–13.1 %) |
| R2 | 26 | 0.0 % (0/26; Wilson 0.0 %–12.9 %; rule of three ≤ 11.5 %) | 0.0 % (0/26; Wilson 0.0 %–12.9 %; rule of three ≤ 11.5 %) |
| R3 | 76 | 13.2 % (10/76; Wilson 7.3 %–22.6 %) | 0.0 % (0/76; Wilson 0.0 %–4.8 %; rule of three ≤ 3.9 %) |
| R4 | 88 | 0.0 % (0/88; Wilson 0.0 %–4.2 %; rule of three ≤ 3.4 %) | 0.0 % (0/88; Wilson 0.0 %–4.2 %; rule of three ≤ 3.4 %) |
| R5 | 87 | 6.9 % (6/87; Wilson 3.2 %–14.2 %) | 1.1 % (1/87; Wilson 0.2 %–6.2 %) |
| R6 | 4 | 25.0 % (1/4; Wilson 4.6 %–69.9 %) | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) |
| screenshot | 58 | 10.3 % (6/58; Wilson 4.8 %–20.8 %) | 3.4 % (2/58; Wilson 1.0 %–11.7 %) |

## Fake families (recall at τ)

| Family | Threat | LGO fold | n | TPR at τ_0.05 | TPR at τ_0.01 |
|---|---|---|---|---|---|
| faceswap-stills | faceswap | – | 60 | 38.3 % (23/60; Wilson 27.1 %–51.0 %) | 8.3 % (5/60; Wilson 3.6 %–18.1 %) |
| flux1 | t2i | – | 19 | 26.3 % (5/19; Wilson 11.8 %–48.8 %) | 5.3 % (1/19; Wilson 0.9 %–24.6 %) |
| flux2 | t2i | F2 | 20 | 15.0 % (3/20; Wilson 5.2 %–36.0 %) | 0.0 % (0/20; Wilson 0.0 %–16.1 %; rule of three ≤ 15.0 %) |
| gan-faces | gan_face | – | 36 | 22.2 % (8/36; Wilson 11.7 %–38.1 %) | 11.1 % (4/36; Wilson 4.4 %–25.3 %) |
| gpt-4o-image-1 | t2i | – | 4 | 25.0 % (1/4; Wilson 4.6 %–69.9 %) | 25.0 % (1/4; Wilson 4.6 %–69.9 %) |
| gpt-image-1.5-2 | t2i | F3 | 16 | 18.8 % (3/16; Wilson 6.6 %–43.0 %) | 0.0 % (0/16; Wilson 0.0 %–19.4 %; rule of three ≤ 18.8 %) |
| hidream | t2i | – | 15 | 60.0 % (9/15; Wilson 35.7 %–80.2 %) | 6.7 % (1/15; Wilson 1.2 %–29.8 %) |
| hunyuanimage | t2i | – | 7 | 28.6 % (2/7; Wilson 8.2 %–64.1 %) | 0.0 % (0/7; Wilson 0.0 %–35.4 %; rule of three ≤ 42.9 %) |
| ideogram | t2i | – | 12 | 50.0 % (6/12; Wilson 25.4 %–74.6 %) | 8.3 % (1/12; Wilson 1.5 %–35.4 %) |
| imagen3 | t2i | – | 14 | 7.1 % (1/14; Wilson 1.3 %–31.5 %) | 0.0 % (0/14; Wilson 0.0 %–21.5 %; rule of three ≤ 21.4 %) |
| imagen4 | t2i | F7 | 23 | 60.9 % (14/23; Wilson 40.8 %–77.8 %) | 13.0 % (3/23; Wilson 4.5 %–32.1 %) |
| legacy | t2i | – | 32 | 31.2 % (10/32; Wilson 18.0 %–48.6 %) | 0.0 % (0/32; Wilson 0.0 %–10.7 %; rule of three ≤ 9.4 %) |
| long-tail | t2i | – | 46 | 39.1 % (18/46; Wilson 26.4 %–53.5 %) | 4.3 % (2/46; Wilson 1.2 %–14.5 %) |
| midjourney | t2i | F1 | 33 | 60.6 % (20/33; Wilson 43.7 %–75.3 %) | 6.1 % (2/33; Wilson 1.7 %–19.6 %) |
| modern-t2i-faces | t2i | – | 24 | 45.8 % (11/24; Wilson 27.9 %–64.9 %) | 25.0 % (6/24; Wilson 12.0 %–44.9 %) |
| nano-banana | t2i | F4 | 28 | 14.3 % (4/28; Wilson 5.7 %–31.5 %) | 7.1 % (2/28; Wilson 2.0 %–22.6 %) |
| partial-edits | edit | – | 40 | 15.0 % (6/40; Wilson 7.1 %–29.1 %) | 5.0 % (2/40; Wilson 1.4 %–16.5 %) |
| personalised | personalised | – | 27 | 59.3 % (16/27; Wilson 40.7 %–75.5 %) | 22.2 % (6/27; Wilson 10.6 %–40.8 %) |
| qwen-image | t2i | F6 | 25 | 40.0 % (10/25; Wilson 23.4 %–59.3 %) | 4.0 % (1/25; Wilson 0.7 %–19.5 %) |
| recraft | t2i | – | 13 | 23.1 % (3/13; Wilson 8.2 %–50.3 %) | 0.0 % (0/13; Wilson 0.0 %–22.8 %; rule of three ≤ 23.1 %) |
| sd3 | t2i | – | 19 | 31.6 % (6/19; Wilson 15.4 %–54.0 %) | 0.0 % (0/19; Wilson 0.0 %–16.8 %; rule of three ≤ 15.8 %) |
| seedream3 | t2i | – | 10 | 30.0 % (3/10; Wilson 10.8 %–60.3 %) | 0.0 % (0/10; Wilson 0.0 %–27.8 %; rule of three ≤ 30.0 %) |
| seedream4 | t2i | F5 | 21 | 23.8 % (5/21; Wilson 10.6 %–45.1 %) | 4.8 % (1/21; Wilson 0.8 %–22.7 %) |

## Leave-generator-out folds

| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC | ECE |
|---|---|---|---|---|---|---|
| F1 | midjourney | 71 | 344 | 62.0 % (44/71; Wilson 50.3 %–72.4 %) | 0.8950 | 0.0859 |
| F2 | flux2 | 39 | 344 | 20.5 % (8/39; Wilson 10.8 %–35.5 %) | 0.7458 | 0.2211 |
| F3 | gpt-image-1.5-2 | 32 | 344 | 21.9 % (7/32; Wilson 11.0 %–38.8 %) | 0.7384 | 0.1964 |
| F4 | nano-banana | 50 | 344 | 20.0 % (10/50; Wilson 11.2 %–33.0 %) | 0.7507 | 0.1544 |
| F5 | seedream4 | 46 | 344 | 37.0 % (17/46; Wilson 24.5 %–51.4 %) | 0.8384 | 0.1250 |
| F6 | qwen-image | 50 | 344 | 40.0 % (20/50; Wilson 27.6 %–53.8 %) | 0.8414 | 0.1190 |
| F7 | imagen4 | 48 | 344 | 52.1 % (25/48; Wilson 38.3 %–65.5 %) | 0.8735 | 0.0873 |

Macro TPR at τ_0.05 over F1–F7: 0.3620; minimum 0.2000; macro ECE 0.1413.

## Robustness (fixed τ_0.05)

Reference `native`/`none` on 278 of 278 robustness items.

| Transform | Reals | Fakes | ΔTPR | ΔFPR | ΔAUC | Inconclusive under t |
|---|---|---|---|---|---|---|
| jpeg95 | 148 | 130 | 0.0308 | 0.0000 | 0.0143 | – |
| jpeg75 | 148 | 130 | 0.0308 | -0.0068 | 0.0301 | – |
| jpeg50 | 148 | 130 | 0.0538 | -0.0203 | 0.0465 | – |
| webp80 | 148 | 130 | 0.0846 | 0.0000 | 0.0421 | – |
| resize0.5 | 148 | 130 | 0.1769 | -0.0270 | 0.0069 | – |
| crop70 | 148 | 130 | 0.1077 | -0.0473 | 0.0044 | – |
| screenshot | 148 | 130 | -0.0462 | 0.0405 | 0.0378 | – |
| screenshot_jpeg | 148 | 130 | -0.0385 | 0.0405 | 0.0436 | – |
| social_ig | 148 | 130 | 0.0308 | -0.0338 | 0.0109 | – |
| social_x | 148 | 130 | 0.0462 | -0.0068 | 0.0314 | – |
| launder | 148 | 130 | 0.2846 | -0.0676 | 0.0905 | – |
| noise3 | 148 | 130 | -0.0923 | -0.0068 | -0.0046 | – |
| upscale2x | 139 | 130 | -0.0692 | 0.0072 | 0.0081 | – |
| rephoto_proxy | 94 | 0 | – | -0.0213 | – | – |

Not scored: exif_forge.

## Slices

10 slices; 2 with an AUC (≥ 20 items per class).

| Slice | Reals | Fakes | AUC |
|---|---|---|---|
| `image/lossless/side512-1024/noface` | 41 | 253 | 0.8544 |
| `image/lossless/side<512/noface` | 29 | 59 | 0.7767 |

## Video and audio

- **video:** system `v1.video_fused@a6030abaa903`; variant `native`; τ_0.05 = – (n = 16); τ_0.01 = – (n = 16); 38 reals and 32 fakes scored, 0 without a score; excluded {}.
- **audio:** system `v1.audio_fused@4eea2f720e5f`; variant `native`; τ_0.05 = 0.9486 (n = 25); τ_0.01 = – (n = 25); 74 reals and 76 fakes scored, 1 without a score; excluded {}.

| Audio set | EER |
|---|---|
| `audio/test` | 19.7 % |
| `audio/test/asvspoof5` | 23.5 % |
| `audio/test/in_the_wild` | 16.2 % |

`video/lgo`:

| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC |
|---|---|---|---|---|---|
| V-Kling | kling | 5 | 38 | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) | 0.5579 |
| V-Seedance | seedance | 5 | 38 | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) | 0.4316 |
| V-Sora | sora | 5 | 38 | 0.0 % (0/5; Wilson 0.0 %–43.4 %; rule of three ≤ 60.0 %) | 0.5737 |
| V-Veo | veo | 4 | 38 | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) | 0.8750 |
| V-WanHunyuan | wan-hunyuan | 4 | 38 | 0.0 % (0/4; Wilson 0.0 %–49.0 %; rule of three ≤ 75.0 %) | 0.5395 |

Macro TPR at τ_0.05 over V-Veo, V-Sora, V-Kling, V-Seedance, V-WanHunyuan: 0.0000; minimum 0.0000.

## Not measured

