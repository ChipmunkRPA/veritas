# Evaluation: goals, results, interpretation

## v2.1.1 on Veritas-Bench vb-1.1

- **Source.** `docs/v2/bench/results.json`, generated 2026-10-10T19:07:41Z at git `ffcca70` by
  `bash backend/scripts/j5_report.sh --figures` (job `n20a-report`: the served code with the decision rules of section N, D-64,
  D-65, D-76 and D-77; artefact shas unchanged), manifest sha256 `8314119e487a…`, scored scope sha256 `2041bf678721…` (J-4:
  robustness items from `test` only). The artefacts and the scores: `bash backend/scripts/j5_fit.sh`,
  `bash backend/scripts/j5_refit_av.sh` (job `j5-chain` at ac9ca5c), `bash backend/scripts/j5_score_dima806.sh` (job
  `j5-score-dima806`), `bash backend/scripts/j5_report.sh` (job `j5-report2`, which wrote the artefacts' metrics),
  `scripts/shortcut_gate.py` (job `j5-shortcut-gate`). Record: `docs/v2/bench/admissions/fuser-image-2.1.1.md`; the decision
  figures: `docs/v2/qa/n20a-record/README.md`.
- **Status.** Automated records, pending human review (D-17). AUC and ECE carry grouped bootstrap 95 % intervals; rates carry
  Wilson 95 % intervals.
- **Re-used test, post-selection.** Before these figures, vb-1.1 `test` had been read by 8 v2.1 report runs and the v2.1 ML
  audit's refit, and the v2.1.1 design was chosen after those test-derived numbers had been seen; the J-5a probe, the J-11
  report and the N-20a report read it again (count: `docs/v2/bench/shortfalls.md`, "Post-hoc rule changes", which also says
  what each decision rule of section N changed). Nothing was fitted on `test`, but these are not fresh-lockbox figures.
- **Superseded.** Every v2.1 image figure below this section. The v2.1 image fuser learned an image-size prior (D-45): its size
  terms alone reached a context-only AUC of 0.7897 on `cal`. `fuser-image-2.1.1` carries no additive size or face term (gate 2b
  0.5000) and is trained with resolution-balanced weights.
- **Views (D-45).** `image/test`, its threat subsets and the LGO folds are size-matched in min-side buckets (every item of the rarer
  class plus a salted sample of the other at the pooled class ratio): 958 reals, 820 fakes. `image/test/full` (2,152 reals, 1,843
  fakes) is the secondary view. The real strata below use every test real.
- **Thresholds.** τ_α from `cal` reals (rank rule, equal stratum weights; ADR 0007 pools strata with fewer than 19 `cal` reals).
  Image τ_0.05 = 0.7150 and τ_0.01 = 0.9607 (850 reals); video τ_0.05 = 0.7330 (74 reals; τ_0.01 not established); audio
  τ_0.05 = 0.8038 and τ_0.01 = 0.8085 (113 reals).
- **Operating point.** "TPR@5 %FPR" is recall at τ_0.05, the threshold whose calibration target is 5 % FPR; it is not the realised
  test rate. On image test reals τ_0.05 flags 7.5 % (162/2,152; Wilson 6.5 %–8.7 %; size-matched view 7.6 %, 73/958) and 7.3 %
  stratum-weighted (R6 with R3), against 5 %; τ_0.01 flags 1.3 % (29/2,152; Wilson 0.9 %–1.9 %; size-matched 1.8 %, 17/958;
  stratum-weighted 1.4 %), against 1 %. Like for like with `fuser-image-2.1.0` at its τ_0.05 (probe
  `scripts/v2/probes/j5a_operating_point.py`, `docs/v2/qa/j5a_operating_point_probe.md`): full view 7.5 % vs 5.3 %, size-matched
  view 7.6 % vs 10.1 %, stratum-weighted 7.3 % vs 5.0 %. 2.1.0's lower full-view rate came with the size prior (D-45); on the
  size-matched view its rate is higher. Cause as measured: the `cal` reals hold unusually few high-scoring reals, mostly
  saturated `img.hf.ateeqq` scores (`cal` 4.93 % weighted, `fit` reals 9.44 %; groups reshuffled between `cal` and `test` reals
  within strata, B = 2,000: null median 4.83 %, 95 % [3.25 %, 6.73 %], P(≥ 7.32 %) = 0.0070; 2.1.0 p = 0.3973); per-context Platt
  slopes add about 0.5 pt; no code, context or composition cause was found. τ is not re-tuned on `test`.
- **Shortcut gate** (D-45 as amended by D-52, `docs/v2/bench/shortcut_gate.json`): served artefacts PASS; structure PASS; 2a weighted
  AUC 0.8582, unweighted 0.9526 (container geometry of the bench), disclosed by the `D-45 2a` row of `shortfalls.md`; 2b image,
  video and audio 0.5000 PASS; 2c no remaining additive context term.

### Systems

| Kind | Artefact | Fitted detectors | Profiles that load it | Record |
|---|---|---|---|---|
| Image | `fuser-image-2.1.1` | `img.dda`, `img.dfdc_b7` (only where a face is detected), `img.hf.ateeqq` (only where no face is detected), `img.hf.dima806`; no additive context terms | `full`, `deep` | `docs/v2/bench/admissions/fuser-image-2.1.1.md`; admissions `v2.1.1-admissions.md` (D-53) |
| Video | `fuser-video-2.1.1` | `vid.synth.frames.commfor384`, `vid.synth.frames.dda`, `vid.frames.dfdc_b7`, `vid.codec.ffprobe`, `aud.gustking_xlsr`; no additive size or face terms (`policy.VIDEO_2B`), but the DFDC-B7 missing term (+0.37 logit) is a face-presence prior (see "Video" below); none of the five is admitted (records pending) | `full`, `deep` | `docs/v2/bench/admissions/fuser-image-2.1.1.md` (Video and audio) |
| Audio | `fuser-audio-2.1.1` | `aud.gustking_xlsr` | `quick`, `full`, `deep` | `docs/v2/bench/admissions/fuser-image-2.1.1.md` (Video and audio) |

`quick` image and video analyses and documents use the uncalibrated fallback fuser. An artefact's sha256 changes whenever the
report fills its `metrics` block; `results.json` names the sha256 the report read, and job `j5-report2` logged the sha256 after the
write (image `746ada66f73d…`, video `fbbccc434f1a…`, audio `44b63c7c8029…`).

### Results per set (`test`)

| Set | Reals | Fakes | AUC (95 % CI) | TPR@5 %FPR | FPR at τ_0.05 | FPR at τ_0.01 | ECE (CI) | Brier | Inconclusive |
|---|---|---|---|---|---|---|---|---|---|
| `image/test` | 958 | 820 | 0.9054 (0.8905–0.9192) | 71.5 % (586/820; Wilson 68.3 %–74.4 %) | 7.6 % (73/958; Wilson 6.1 %–9.5 %) | 1.8 % (17/958; Wilson 1.1 %–2.8 %) | 0.0537 (0.0432–0.0724) | 0.1270 | 39.7 % |
| `image/test/full` | 2152 | 1843 | 0.9014 (0.8900–0.9122) | 72.2 % (1330/1843; Wilson 70.1 %–74.2 %) | 7.5 % (162/2152; Wilson 6.5 %–8.7 %) | 1.3 % (29/2152; Wilson 0.9 %–1.9 %) | 0.0519 (0.0421–0.0661) | 0.1281 | 41.5 % |
| `image/test/t2i` | 733 | 430 | 0.9426 (0.9281–0.9549) | 78.4 % (337/430; Wilson 74.2 %–82.0 %) | 7.6 % (56/733; Wilson 5.9 %–9.8 %) | – | 0.0398 (0.0319–0.0615) | 0.0982 | 35.7 % |
| `image/test/edit` | 724 | 51 | 0.9168 (0.8642–0.9616) | 78.4 % (40/51; Wilson 65.4 %–87.5 %) | 6.5 % (47/724; Wilson 4.9 %–8.5 %) | – | 0.0837 (0.0556–0.1454) | 0.1139 | 38.2 % |
| `image/test/personalised` | 468 | 20 | 0.9784 (0.9543–0.9961) | 90.0 % (18/20; Wilson 69.9 %–97.2 %) | 10.5 % (49/468; Wilson 8.0 %–13.6 %) | – | 0.1198 (0.1076–0.1627) | 0.0843 | 49.4 % |
| `image/test/faceswap_vs_R5` | 395 | 185 | 0.7819 (0.7384–0.8217) | 51.4 % (95/185; Wilson 44.2 %–58.5 %) | 6.6 % (26/395; Wilson 4.5 %–9.5 %) | – | 0.1482 (0.1230–0.1945) | 0.2075 | 64.0 % |
| `image/test/gan_face_vs_R5` | 338 | 92 | 0.8662 (0.8226–0.9073) | 63.0 % (58/92; Wilson 52.8 %–72.2 %) | 7.1 % (24/338; Wilson 4.8 %–10.3 %) | – | 0.1232 (0.1052–0.1661) | 0.1632 | 71.6 % |
| `image/test/legacy` | 154 | 72 | 0.9915 (0.9823–0.9984) | 94.4 % (68/72; Wilson 86.6 %–97.8 %) | 1.9 % (3/154; Wilson 0.7 %–5.6 %) | – | 0.0202 (0.0106–0.0494) | 0.0330 | 15.9 % |
| `video/test` | 162 | 135 | 0.9613 (0.9374–0.9806) | 80.7 % (109/135; Wilson 73.3 %–86.5 %) | 2.5 % (4/162; Wilson 1.0 %–6.2 %) | – | 0.0495 (0.0441–0.0881) | 0.0762 | 50.2 % |
| `audio/test` | 262 | 444 | 0.8474 (0.8070–0.8852) | 18.9 % (84/444; Wilson 15.5 %–22.8 %) | 2.3 % (6/262; Wilson 1.1 %–4.9 %) | – | 0.0775 (0.0702–0.1260) | 0.1662 | 72.8 % |
| `audio/test/asvspoof5` | 86 | 90 | 0.8786 (0.8221–0.9283) | 38.9 % (35/90; Wilson 29.5 %–49.2 %) | 4.7 % (4/86; Wilson 1.8 %–11.4 %) | – | 0.2047 (0.1640–0.2412) | 0.1760 | 73.3 % |
| `audio/test/in_the_wild` | 94 | 89 | 0.9081 (0.8126–0.9782) | 1.1 % (1/89; Wilson 0.2 %–6.1 %) | 1.1 % (1/94; Wilson 0.2 %–5.8 %) | – | 0.1566 (0.1206–0.2467) | 0.1412 | 81.6 % |

**Decision error rate (image, size-matched `image/test`).** Of the images that receive an AI or authentic decision under the served
policy with the decision rules of section N (D-64, D-65, D-76, D-77), 9.2 % are wrong (selective risk 99/1,073; Wilson
7.6 %–11.1 %); before those rules (K-1) 10.4 % (125/1,197; Wilson 8.8 %–12.3 %), and under the J-5 policy, before ad26e7a,
10.3 % (131/1,274; Wilson 8.7 %–12.1 %). Video: 4.7 % (7/148; Wilson 2.3 %–9.4 %), before the rules 7/153; audio unchanged,
8.2 % (16/194). The inconclusive column above is under the same rules: 705 of 1,778 `image/test` items (K-1: 581), 149 of 297
videos (K-1: 144). This is the benchmark anchor for the field decision accuracy below.

Audio EER: `audio/test` 24.4 %, `audio/test/asvspoof5` 17.4 %, `audio/test/in_the_wild` 16.9 %. The threat subsets of the full image
view and the per-subset τ_0.01 rates are in `results.md`.

- **Image.** Without the size prior the full-view AUC is 0.9014 against 0.9632 for v2.1, and recall at τ_0.05 72.2 % against
  83.6 %; the v2.1 figures rested partly on image size.
- **Video.** τ_0.05 flags 2.5 % of real videos; 50.2 % of test videos (149/297) are inconclusive (K-1, before the decision
  rules of section N: 48.5 %; v2.1: 18.9 %). Every 480–719 px video slice has no real videos in its validation data, so all 58
  test AI videos at that resolution are capped. The
  fitted missing term of DFDC-B7 (`miss:vid.frames.dfdc_b7` +0.3695 logit) is a face-presence prior: "no face found" alone
  separates the video classes at AUC 0.718 on `fit`, 0.701 on `cal` (`docs/v2/qa/k4-docs/video_face_prior.md`) and 0.711 on
  `test` (release audit). Gate 2b tests additive terms only and reads 0.5000; the vb-1.2 lever is in `shortfalls.md`.
- **Audio.** Recall at τ_0.05 is 18.9 %: the voice fuser holds one detector, and in-the-wild clones are rarely flagged (1/89).

### Image, per generator family (full view, test fakes at the image τ)

| Family | Threat | LGO fold | n | TPR at τ_0.05 | TPR at τ_0.01 |
|---|---|---|---|---|---|
| sd3 | t2i | – | 63 | 98.4 % (62/63; Wilson 91.5 %–99.7 %) | 95.2 % (60/63; Wilson 86.9 %–98.4 %) |
| legacy | t2i | – | 107 | 96.3 % (103/107; Wilson 90.8 %–98.5 %) | 88.8 % (95/107; Wilson 81.4 %–93.5 %) |
| hunyuanimage | t2i | – | 20 | 95.0 % (19/20; Wilson 76.4 %–99.1 %) | 90.0 % (18/20; Wilson 69.9 %–97.2 %) |
| personalised | personalised | – | 92 | 93.5 % (86/92; Wilson 86.5 %–97.0 %) | 80.4 % (74/92; Wilson 71.2 %–87.3 %) |
| seedream3 | t2i | – | 26 | 92.3 % (24/26; Wilson 75.9 %–97.9 %) | 73.1 % (19/26; Wilson 53.9 %–86.3 %) |
| midjourney | t2i | F1 | 124 | 91.9 % (114/124; Wilson 85.8 %–95.6 %) | 80.6 % (100/124; Wilson 72.8 %–86.6 %) |
| recraft | t2i | – | 36 | 88.9 % (32/36; Wilson 74.7 %–95.6 %) | 50.0 % (18/36; Wilson 34.5 %–65.5 %) |
| hidream | t2i | – | 47 | 80.9 % (38/47; Wilson 67.5 %–89.6 %) | 55.3 % (26/47; Wilson 41.2 %–68.6 %) |
| imagen4 | t2i | F7 | 72 | 79.2 % (57/72; Wilson 68.4 %–86.9 %) | 47.2 % (34/72; Wilson 36.1 %–58.6 %) |
| long-tail | t2i | – | 159 | 77.4 % (123/159; Wilson 70.3 %–83.2 %) | 58.5 % (93/159; Wilson 50.7 %–65.9 %) |
| ideogram | t2i | – | 38 | 76.3 % (29/38; Wilson 60.8 %–87.0 %) | 65.8 % (25/38; Wilson 49.9 %–78.8 %) |
| imagen3 | t2i | – | 49 | 75.5 % (37/49; Wilson 61.9 %–85.4 %) | 28.6 % (14/49; Wilson 17.8 %–42.4 %) |
| partial-edits | edit | – | 153 | 75.2 % (115/153; Wilson 67.8 %–81.3 %) | 62.7 % (96/153; Wilson 54.9 %–70.0 %) |
| seedream4 | t2i | F5 | 66 | 72.7 % (48/66; Wilson 61.0 %–82.0 %) | 37.9 % (25/66; Wilson 27.1 %–49.9 %) |
| modern-t2i-faces | t2i | – | 89 | 69.7 % (62/89; Wilson 59.5 %–78.2 %) | 40.4 % (36/89; Wilson 30.9 %–50.8 %) |
| flux1 | t2i | – | 60 | 66.7 % (40/60; Wilson 54.1 %–77.3 %) | 48.3 % (29/60; Wilson 36.2 %–60.7 %) |
| gan-faces | gan_face | – | 123 | 66.7 % (82/123; Wilson 57.9 %–74.4 %) | 10.6 % (13/123; Wilson 6.3 %–17.2 %) |
| nano-banana | t2i | F4 | 80 | 62.5 % (50/80; Wilson 51.5 %–72.3 %) | 27.5 % (22/80; Wilson 18.9 %–38.1 %) |
| faceswap-stills | faceswap | – | 211 | 54.5 % (115/211; Wilson 47.8 %–61.1 %) | 11.4 % (24/211; Wilson 7.8 %–16.4 %) |
| qwen-image | t2i | F6 | 84 | 51.2 % (43/84; Wilson 40.7 %–61.6 %) | 17.9 % (15/84; Wilson 11.1 %–27.4 %) |
| gpt-4o-image-1 | t2i | – | 30 | 40.0 % (12/30; Wilson 24.6 %–57.7 %) | 6.7 % (2/30; Wilson 1.8 %–21.3 %) |
| flux2 | t2i | F2 | 64 | 35.9 % (23/64; Wilson 25.3 %–48.2 %) | 15.6 % (10/64; Wilson 8.7 %–26.4 %) |
| gpt-image-1.5-2 | t2i | F3 | 50 | 32.0 % (16/50; Wilson 20.8 %–45.8 %) | 8.0 % (4/50; Wilson 3.2 %–18.8 %) |

Rows with few fakes carry wide Wilson intervals; read them as "not validated", not as a ranking.

### Leave-one-generator-out (image refits without the family, size-matched folds)

| Fold | Families | Fakes | Reals | TPR at τ_0.05 | AUC | ECE |
|---|---|---|---|---|---|---|
| F1 | midjourney | 32 | 248 | 81.2 % (26/32; Wilson 64.7 %–91.1 %) | 0.9292 | 0.1066 |
| F2 | flux2 | 15 | 233 | 40.0 % (6/15; Wilson 19.8 %–64.3 %) | 0.6652 | 0.2914 |
| F3 | gpt-image-1.5-2 | 28 | 569 | 25.0 % (7/28; Wilson 12.7 %–43.4 %) | 0.7224 | 0.2090 |
| F4 | nano-banana | 34 | 369 | 64.7 % (22/34; Wilson 47.9 %–78.5 %) | 0.8738 | 0.0907 |
| F5 | seedream4 | 66 | 969 | 83.3 % (55/66; Wilson 72.6 %–90.4 %) | 0.9643 | 0.0794 |
| F6 | qwen-image | 19 | 217 | 42.1 % (8/19; Wilson 23.1 %–63.7 %) | 0.7507 | 0.2254 |
| F7 | imagen4 | 18 | 217 | 72.2 % (13/18; Wilson 49.1 %–87.5 %) | 0.9017 | 0.1532 |

Macro TPR at τ_0.05 over F1–F7 is 0.5837 against ≥ 0.75 (G1a FAIL). Like for like with v2.1, whose 0.7716 is a full-view figure,
the v2.1.1 full-view macro is 0.5908. A family is left out of the fuser's training only: the training data of the ateeqq and
dima806 classifiers is not established. The weakest fold, F3 GPT-image-1.5 / 2, reaches
0.2500 against ≥ 0.50 (G1b FAIL). Macro ECE is 0.1651 against ≤ 0.08 (G4b FAIL).

### Video, per vendor (refits without the vendor)

| Fold | Fakes | TPR at τ_0.05 | AUC |
|---|---|---|---|
| V-Kling | 19 | 84.2 % (16/19; Wilson 62.4 %–94.5 %) | 0.9789 |
| V-Seedance | 19 | 36.8 % (7/19; Wilson 19.1 %–59.0 %) | 0.8268 |
| V-Sora | 19 | 21.1 % (4/19; Wilson 8.5 %–43.3 %) | 0.8450 |
| V-Veo | 18 | 55.6 % (10/18; Wilson 33.7 %–75.4 %) | 0.9530 |
| V-WanHunyuan | 18 | 100.0 % (18/18; Wilson 82.4 %–100.0 %) | 0.9983 |

Macro TPR at τ_0.05 is 0.5953 against ≥ 0.80 (G9a FAIL); the minimum is Sora, 0.2105.

### Real strata (image false positives, every test real)

| Stratum | n | FPR at τ_0.05 (target 5 %) | FPR at τ_0.01 (target 1 %) |
|---|---|---|---|
| R1 | 387 | 3.6 % (14/387; Wilson 2.2 %–6.0 %) | 0.5 % (2/387; Wilson 0.1 %–1.9 %) |
| R2 | 409 | 6.1 % (25/409; Wilson 4.2 %–8.9 %) | 0.2 % (1/409; Wilson 0.0 %–1.4 %) |
| R3 | 479 | 11.7 % (56/479; Wilson 9.1 %–14.9 %) | 1.7 % (8/479; Wilson 0.8 %–3.3 %) |
| R4 | 405 | 9.1 % (37/405; Wilson 6.7 %–12.3 %) | 3.7 % (15/405; Wilson 2.3 %–6.0 %) |
| R5 | 450 | 5.8 % (26/450; Wilson 4.0 %–8.3 %) | 0.7 % (3/450; Wilson 0.2 %–1.9 %) |
| R6 | 22 | 18.2 % (4/22; Wilson 7.3 %–38.5 %) | 0.0 % (0/22; Wilson 0.0 %–14.9 %; rule of three ≤ 13.6 %) |
| pooled | 2152 | 7.5 % (162/2152; Wilson 6.5 %–8.7 %); stratum-weighted 7.3 % | 1.3 % (29/2152; Wilson 0.9 %–1.9 %); stratum-weighted 1.4 % |

G3 requires ≤ 5 % per stratum with a Wilson upper bound ≤ 7.5 %: R1 passes; R2, R3, R4, R5 and R6 fail. Per stratum × min-side
bucket rates are in `results.md`.

### Robustness (image, fixed τ_0.05, reference: native files, 562 robustness items)

| Transform | Reals | Fakes | ΔTPR | ΔFPR | ΔAUC | Inconclusive under t |
|---|---|---|---|---|---|---|
| jpeg75 | 298 | 264 | 0.1667 | -0.0134 | 0.0630 | 46.6 % |
| jpeg50 | 298 | 264 | 0.2803 | -0.0201 | 0.1172 | 100.0 % |
| webp80 | 298 | 264 | 0.1250 | 0.0101 | 0.0395 | 36.7 % |
| resize0.5 | 298 | 264 | 0.2348 | 0.0302 | 0.1542 | 29.4 % |
| crop70 | 298 | 264 | 0.0417 | 0.0201 | 0.0299 | 28.3 % |
| screenshot_jpeg | 298 | 264 | 0.1326 | 0.2114 | 0.1478 | 85.9 % |
| social_ig | 298 | 264 | 0.1667 | -0.0067 | 0.0680 | 34.3 % |
| social_x | 298 | 264 | 0.1212 | -0.0101 | 0.0405 | 43.4 % |
| noise3 | 298 | 264 | -0.0076 | 0.2282 | 0.0668 | 27.4 % |
| upscale2x | 289 | 263 | 0.0000 | 0.0104 | -0.0082 | 33.9 % |
| exif_forge | 298 | 240 | 0.0000 | -0.0034 | -0.0004 | 34.8 % |

ΔTPR is clean − transformed, ΔFPR transformed − clean. G5a fails on `jpeg75`, `webp80` and `resize0.5`; G5b on `jpeg50`; G5c on
`resize0.5`, `crop70`, `screenshot_jpeg` and `noise3`. `exif_forge` is trivially satisfied: the fuser reads no metadata feature.
Not scored for the admitted detectors (D-40 scope): `jpeg95`, `screenshot`, `launder`, `rephoto_proxy`.

### Field conditions (EA-02 slice validation map, image `test`)

| Condition | Reals | Fakes | AUC | Measured at |
|---|---|---|---|---|
| JPEG q ≥ 90, ≥ 1 MP, no face | 489 | 205 | 0.9137 | `jpeg_q>=90_1mp_noface` |
| JPEG q 70–89 | 751 | 298 | 0.9051 | `jpeg_q70-89` |
| PNG / lossless (no JPEG quantisation tables) | 365 | 1116 | 0.8705 | `lossless` |
| face present | 867 | 891 | 0.8742 | `face` |
| social recompressed (social_ig, social_x) | 596 | 528 | 0.8886 | `social` |
| screenshot (screenshot, screenshot_jpeg) | 298 | 264 | 0.7950 | `screenshot` |

The runtime slice maps are in `results.md`. The runtime envelope is read from each artefact's `test`-split slice metrics, so the
`test` decision figures are evaluated with an envelope derived from the same data. Test images are scored on the `norm` re-encode (ADR 0004/0005), so every test image
falls in one of 12 `q>=90` image leaves (2 measured with ≥ 100 items per class, 10 carrying a measured parent's AUC). Of the
19 video leaves, 5 contain both classes and lie inside the envelope; the 14 others (every 480–719 px leaf among them) cap their
videos at inconclusive. Both audio leaves are inside.

### Calibration, conformal sets, invariants

- **Conformal coverage at α = 0.05 (image, size-matched view).** Reals 91.5 % (877/958; Wilson 89.6 %–93.1 %), AI 93.9 % (770/820;
  Wilson 92.1 %–95.3 %) (G6a FAIL). Inconclusive under the served policy with the decision rules of section N: 39.7 % of
  `image/test` (705/1,778; G6b FAIL) and 39.1 % under `social_ig` (220/562; G6c PASS). The K-1 figures (32.7 % and 34.3 %)
  predate D-64, D-65, D-76 and D-77, and the J-5 figures (28.3 % and 32.4 %) predate ad26e7a, which makes a slice without both
  classes abstain; AUC, TPR, FPR, ECE and the coverage are unchanged by these rules, which change which items are decided.
- **Additivity (G15a).** Contributions add up to the fused logit within 7.105 × 10⁻¹⁵ on 1,000 items.
- **Monotonicity (G15b).** 0 of 1,000 random positive detector moves lowered the score.
- **Reproducibility (G12).** Two `calibrate` runs at one commit give identical artefacts except `created_at`.
- **Latency (J-11, `docs/v2/qa/j11-latency/`).** On `full` with the served v2.1.1 detector sets: G13a 4.697 s (p95 over the 256 px
  face 1.061 s, 800 px 1.575 s and 24 MP camera 4.697 s workloads) and G13b 11.877 s (9 s 1080p video, 2 valid runs). G13 specifies
  an idle host; these were measured on a loaded host (1-minute load 14–16), so they are loaded-host figures. The J-4 figures below
  (4.12 s, 9.22 s) used the v2.1 detector set and are superseded.

### Field test v2.1.1 under the served rules (N-20b; D-64, D-65, D-76, D-77; `docs/field_test.md`)

Run 2026-10-11 00:02–00:12 UTC through the live v2 API on `full` (job `n20b-field`, `scripts/field_test.py run-v21 --record
n20b`), on the same v2.1.1 artefacts and items as the K-1 run below (`docs/field_test_n20b.json`). Every item's p(AI) is
that of K-1; the served rules only take decisions away. Each cell: this run / K-1.

| Kind | Set | n | Decided | Correct | Capped | Not assessable (D-64) | D-76 | D-77 |
|---|---|---|---|---|---|---|---|---|
| image | original | 23 | 12 / 15 | 11 / 12 | 1 / 1 | 0 / 0 | 3 / 0 | 0 / 0 |
| image | v2.1 | 80 | 36 / 38 | 30 / 31 | 5 / 4 | 0 / 0 | 1 / 0 | 0 / 0 |
| video | original | 8 | 5 / 5 | 5 / 5 | 2 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| video | v2.1 | 32 | 10 / 13 | 10 / 13 | 5 / 5 | 0 / 0 | 3 / 0 | 0 / 0 |
| document | original | 20 | 20 / 20 | 20 / 20 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

- **D-76** took 7 decisions: three real camera originals that K-1 called AI at p(AI) 0.683–0.706 (below τ_0.05), one AI still,
  and three AI clips that K-1 decided correctly. **Camera originals:** 5 of 52 received an AI decision (9.6 %, Wilson
  4.2–20.6 %; K-1: 8 of 52), against the bench R1 FPR of 3.6 % (14/387; Wilson 2.2–6.0 %); Fisher exact p = 0.061, so the field
  no longer shows camera originals flagged more often than the benchmark does. Small samples; an in-distribution sample.

### Field test v2.1.1 under the served abstention policy (K-1, D-42, D-49; the commit history of `docs/field_test.md`)

Run 2026-09-22 00:19 through the live v2 API on `full` (job `k1-field`, `scripts/field_test.py run-v21 --record k1`), on the same
v2.1.1 artefacts as the J-5 run of 2026-09-17 (`docs/field_test_v211.json`). ad26e7a, after J-5, made a runtime slice lacking
either class abstain; every item's p(AI) is unchanged, and the only differences are decisions capped by that rule. The field set
is an in-distribution sample (same sources as vb-1.1): held out by group key and file hash, but its AI items come from generators
and datasets vb-1.1 trains on.

| Kind | Set | n | Decision rate | Decision accuracy | p(AI) side | p(AI) AUC | Inconclusive causes | J-5: rate / accuracy | v2.1 (H-9): rate / accuracy |
|---|---|---|---|---|---|---|---|---|---|
| image | original | 23 | 15/23 (65 %) | 12/15 (80 %) | 17/23 (74 %) | 0.871 | envelope cap 1, conformal band 7 | 16/23 / 13/16 | 22/23 / 22/22 |
| image | v2.1 | 80 | 38/80 (48 %) | 31/38 (82 %) | 57/80 (71 %) | 0.806 | envelope cap 4, conformal band 38 | 42/80 / 35/42 | 71/80 / 66/71 |
| video | original | 8 | 5/8 (62 %) | 5/5 (100 %) | 7/8 (88 %) | 1.000 | envelope cap 2, conformal band 1 | 7/8 / 7/7 | 7/8 / 7/7 |
| video | v2.1 | 32 | 13/32 (41 %) | 13/13 (100 %) | 30/32 (94 %) | 0.984 | envelope cap 5, conformal band 14 | 18/32 / 18/18 | 19/32 / 19/19 |
| document | original | 20 | 20/20 (100 %) | 20/20 (100 %) | 20/20 (100 %) | 1.000 | – | 20/20 / 20/20 | 20/20 / 20/20 |

- **Envelope caps (12):** 5 real camera originals in `image/q70-89/side>1024` leaves (1 original, 4 v2.1 set), 6 AI clips at
  480–719 px (2 original, 4 v2.1 set) and 1 real clip in `video/res>=720/dur<4s/face>=256`. Each was a correct decision in J-5.
- **Wrong image decisions (10, unchanged from J-5):** 8 real camera originals called AI (3 of the original set, 5 of the 40 v2.1
  Commons originals; p(AI) 0.68–0.96, short side 1,080–4,160 px) and 2 FLUX.2 stills called authentic (p(AI) 0.10 and 0.13,
  1,024 px). v2.1 made 5 wrong decisions on the v2.1 stills, all AI stills called authentic.
- **Images by min-side bucket** (v2.1 set): 768–1,024 px 19 correct, 2 wrong, 12 inconclusive of 33 (32 AI); above 1,536 px 5
  correct, 3 wrong, 27 inconclusive of 35 (2 AI). Large camera originals are where v2.1's size prior used to decide.
- **No estimate (D-49):** every finished analysis published a probability; no check failed to run and no item was not assessable.
- **D-42:** the v2.1-set decision rates are below 50 % for images (48 %) and videos (41 %); the causes and next levers are in
  `docs/v2/bench/shortfalls.md` ("Field decision rate below 50 %"). The envelope rule is not relaxed. Small samples without
  intervals.
- **Bench–runtime slicing gap:** the bench report slices test images by the scored `norm` re-encode (always JPEG q ≥ 90), while a
  live upload is sliced by its own bytes, so the bench inconclusive rates miss the field caps on JPEG q 70–89 camera originals.
- **Camera originals in this run:** real camera originals received an AI decision in 8 of 52 cases (15.4 %; Wilson
  8.0 %–27.5 %), against the bench R1 FPR of 3.6 % at τ_0.05 (14/387; Wilson 2.2 %–6.0 %; Fisher exact p = 0.002). Three of the
  eight were decided at p(AI) 0.683–0.706, below τ_0.05 (0.715): decisions follow the conformal set, so the two are not the same
  measure. Under D-76 those three are inconclusive, and the N-20b run (above) no longer shows the difference.
- **Size-confounded:** in the v2.1 set, 32 of the 33 stills at 768–1,024 px are AI and 33 of the 35 above 1,536 px are camera
  originals, so the field set cannot check the benchmark's container-geometry cues (D-52). Read the decision accuracy here against
  the bench decision error rate of the same policy (10.4 %, before the decision rules of section N; under them it is 9.2 %, and
  the field test under them is N-20b's).

### Goals

`docs/v2/bench/gate-v2.1.1.txt`: 26 PASS, 26 FAIL and 6 MISSING of 58 checks. `check_shortfalls.py` PASS: 16 of 28 goals are below
target (G1a, G1b, G3, G4a, G4b, G4c, G5a, G5b, G5c, G6a, G6b, G7a, G7b, G7c, G9a, G10a). Each has its values, mitigations and
consequence in `docs/v2/bench/shortfalls.md` and its statement in `docs/limitations.md`. The admission rules changed after fit
and calibration results were seen (D-44, D-51, D-53, `policy.VIDEO_2B`): `shortfalls.md`, "Post-hoc rule changes".

---

## v2.1 on Veritas-Bench vb-1.1 (superseded by v2.1.1 above)

- **Source.** `docs/v2/bench/results.json`, generated 2026-09-15T10:43:20Z at git `5b491fa`, manifest sha256 `8314119e487a…`, scored
  scope sha256 `5059434a74cf…` (D-40, D-43: every image real, up to 5,000 image fakes, 600 robustness items × 11 transforms, 650
  videos, 1,600 recordings). Commands: `GATE=gate-v2.1-final.txt bash backend/scripts/h8_report.sh --latency ../docs/v2/bench/perf.json`
  (job `h10-report`), `make bench` (job `h10-bench`), `scripts/field_test.py run-v21` (H-9).
- **Status.** Automated records, pending human review (D-17). AUC and ECE carry grouped bootstrap 95 % intervals; rates carry
  Wilson 95 % intervals.
- **Thresholds.** τ_α from `cal` reals (rank rule, equal stratum weights; ADR 0007 pools strata with fewer than 19 `cal` reals).
  Image τ_0.05 = 0.5790 and τ_0.01 = 0.9371 (850 reals); video τ_0.05 = 0.7163 (74 reals; τ_0.01 not established); audio
  τ_0.05 = 0.8041 and τ_0.01 = 0.8089 (118 reals). TPR@5 %FPR is recall at τ_0.05.

### Systems

| Kind | Artefact | Fitted detectors | Profiles that load it | Record |
|---|---|---|---|---|
| Image | `fuser-image-2.1.0` | `img.commfor384`, `img.dda`, `img.dfdc_b7` (gated on `has_face`), with context terms | `full`, `deep` | `docs/v2/bench/admissions/fuser-image-2.1.0.md`; its admissions are superseded by v2.1.1-admissions (J-3c), and J-5 gives the numbers of the refit |
| Video | `fuser-video-2.1.0` | `vid.synth.frames.commfor384`, `vid.synth.frames.dda`, `vid.frames.dfdc_b7`, `vid.codec.ffprobe`, `aud.gustking_xlsr` | `full`, `deep` | `docs/v2/bench/admissions/fuser-video-2.1.0.md` |
| Audio | `fuser-audio-2.1.0` | `aud.gustking_xlsr` | `quick`, `full`, `deep` | `docs/v2/bench/admissions/fuser-video-2.1.0.md` (Audio) |

`quick` image and video analyses and documents use the uncalibrated fallback fuser. An artefact's sha256 changes whenever the
report fills its `metrics` block; `results.json` names the sha256 the report read.

### Results per set (`test`)

| Set | Reals | Fakes | AUC (95 % CI) | TPR@5 %FPR | FPR at τ_0.05 | ECE (CI) | Inconclusive |
|---|---|---|---|---|---|---|---|
| `image/test` | 2152 | 1843 | 0.9632 (0.9571–0.9688) | 83.6 % (1541/1843; Wilson 81.9 %–85.2 %) | 5.3 % (113/2152; Wilson 4.4 %–6.3 %) | 0.0221 (0.0162–0.0322) | 18.2 % |
| `image/test/t2i` | 2152 | 1264 | 0.9630 (0.9557–0.9697) | 86.0 % (1087/1264; Wilson 84.0 %–87.8 %) | 5.3 % (113/2152; Wilson 4.4 %–6.3 %) | 0.0321 (0.0247–0.0455) | 18.6 % |
| `image/test/edit` | 2152 | 153 | 0.9664 (0.9489–0.9812) | 86.3 % (132/153; Wilson 79.9 %–90.8 %) | 5.3 % (113/2152; Wilson 4.4 %–6.3 %) | 0.0334 (0.0198–0.0589) | 22.1 % |
| `image/test/personalised` | 2152 | 92 | 0.9989 (0.9979–0.9996) | 100.0 % (92/92; Wilson 96.0 %–100.0 %) | 5.3 % (113/2152; Wilson 4.4 %–6.3 %) | 0.0659 (0.0585–0.0751) | 22.1 % |
| `image/test/faceswap_vs_R5` | 450 | 211 | 0.9380 (0.9173–0.9565) | 69.7 % (147/211; Wilson 63.2 %–75.5 %) | 3.1 % (14/450; Wilson 1.9 %–5.2 %) | 0.0794 (0.0626–0.1104) | 42.2 % |
| `image/test/gan_face_vs_R5` | 450 | 123 | 0.9274 (0.9010–0.9516) | 67.5 % (83/123; Wilson 58.8 %–75.1 %) | 3.1 % (14/450; Wilson 1.9 %–5.2 %) | 0.0818 (0.0616–0.1156) | 45.0 % |
| `image/test/legacy` | 229 | 107 | 0.9721 (0.9580–0.9840) | 86.0 % (92/107; Wilson 78.2 %–91.3 %) | 0.9 % (2/229; Wilson 0.2 %–3.1 %) | 0.0520 (0.0234–0.0785) | 22.9 % |
| `video/test` | 162 | 135 | 0.9668 (0.9454–0.9842) | 87.4 % (118/135; Wilson 80.8 %–92.0 %) | 3.1 % (5/162; Wilson 1.3 %–7.0 %) | 0.0595 (0.0461–0.0926) | 18.9 % |
| `audio/test` | 263 | 449 | 0.8461 (0.8049–0.8833) | 18.7 % (84/449; Wilson 15.4 %–22.6 %) | 2.3 % (6/263; Wilson 1.0 %–4.9 %) | 0.0820 (0.0705–0.1264) | 72.8 % |
| `audio/test/asvspoof5` | 86 | 90 | 0.8786 (0.8221–0.9283) | 38.9 % (35/90; Wilson 29.5 %–49.2 %) | 4.7 % (4/86; Wilson 1.8 %–11.4 %) | 0.2027 (0.1620–0.2393) | 73.3 % |
| `audio/test/in_the_wild` | 95 | 90 | 0.9065 (0.8083–0.9753) | 1.1 % (1/90; Wilson 0.2 %–6.0 %) | 1.1 % (1/95; Wilson 0.2 %–5.7 %) | 0.1534 (0.1214–0.2463) | 81.6 % |

| Audio set | EER |
|---|---|
| `audio/test` | 24.3 % |
| `audio/test/asvspoof5` | 17.4 % |
| `audio/test/in_the_wild` | 16.8 % |

- **Image.** The v2.0 fuser had AUC 0.8944 and recall 47.2 % at τ_0.05 on the core lockbox (section below). The populations differ,
  so this is not a paired comparison.
- **Video.** τ_0.05 is established, 3.1 % of real videos are flagged and 18.9 % of test videos are inconclusive (v2.0 on core: no
  threshold, 70 of 70 inconclusive). No τ_0.01 exists at 74 `cal` reals, so the TPR@1 %FPR column of `results.md` is 0.
- **Audio.** Recall at τ_0.05 is 18.7 %: the voice fuser holds one detector, and in-the-wild clones are rarely flagged (1/90).

### Image, per generator family (test fakes at the image τ)

| Family | Threat | LGO fold | n | TPR at τ_0.05 | TPR at τ_0.01 |
|---|---|---|---|---|---|
| hidream | t2i | – | 47 | 100.0 % (47/47; Wilson 92.4 %–100.0 %) | 78.7 % (37/47; Wilson 65.1 %–88.0 %) |
| ideogram | t2i | – | 38 | 100.0 % (38/38; Wilson 90.8 %–100.0 %) | 84.2 % (32/38; Wilson 69.6 %–92.6 %) |
| personalised | personalised | – | 92 | 100.0 % (92/92; Wilson 96.0 %–100.0 %) | 92.4 % (85/92; Wilson 85.1 %–96.3 %) |
| sd3 | t2i | – | 63 | 100.0 % (63/63; Wilson 94.3 %–100.0 %) | 100.0 % (63/63; Wilson 94.3 %–100.0 %) |
| seedream3 | t2i | – | 26 | 100.0 % (26/26; Wilson 87.1 %–100.0 %) | 80.8 % (21/26; Wilson 62.1 %–91.5 %) |
| midjourney | t2i | F1 | 124 | 97.6 % (121/124; Wilson 93.1 %–99.2 %) | 91.1 % (113/124; Wilson 84.8 %–95.0 %) |
| imagen3 | t2i | – | 49 | 95.9 % (47/49; Wilson 86.3 %–98.9 %) | 51.0 % (25/49; Wilson 37.5 %–64.4 %) |
| recraft | t2i | – | 36 | 94.4 % (34/36; Wilson 81.9 %–98.5 %) | 83.3 % (30/36; Wilson 68.1 %–92.1 %) |
| imagen4 | t2i | F7 | 72 | 91.7 % (66/72; Wilson 83.0 %–96.1 %) | 59.7 % (43/72; Wilson 48.2 %–70.3 %) |
| modern-t2i-faces | t2i | – | 89 | 91.0 % (81/89; Wilson 83.3 %–95.4 %) | 61.8 % (55/89; Wilson 51.4 %–71.2 %) |
| hunyuanimage | t2i | – | 20 | 90.0 % (18/20; Wilson 69.9 %–97.2 %) | 75.0 % (15/20; Wilson 53.1 %–88.8 %) |
| qwen-image | t2i | F6 | 84 | 89.3 % (75/84; Wilson 80.9 %–94.3 %) | 42.9 % (36/84; Wilson 32.8 %–53.5 %) |
| partial-edits | edit | – | 153 | 86.3 % (132/153; Wilson 79.9 %–90.8 %) | 75.2 % (115/153; Wilson 67.8 %–81.3 %) |
| legacy | t2i | – | 107 | 86.0 % (92/107; Wilson 78.2 %–91.3 %) | 84.1 % (90/107; Wilson 76.0 %–89.8 %) |
| nano-banana | t2i | F4 | 80 | 83.8 % (67/80; Wilson 74.2 %–90.3 %) | 41.2 % (33/80; Wilson 31.1 %–52.2 %) |
| flux1 | t2i | – | 60 | 83.3 % (50/60; Wilson 72.0 %–90.7 %) | 60.0 % (36/60; Wilson 47.4 %–71.4 %) |
| seedream4 | t2i | F5 | 66 | 81.8 % (54/66; Wilson 70.9 %–89.3 %) | 54.5 % (36/66; Wilson 42.6 %–66.0 %) |
| long-tail | t2i | – | 159 | 81.1 % (129/159; Wilson 74.3 %–86.5 %) | 64.8 % (103/159; Wilson 57.1 %–71.8 %) |
| faceswap-stills | faceswap | – | 211 | 69.7 % (147/211; Wilson 63.2 %–75.5 %) | 43.6 % (92/211; Wilson 37.1 %–50.3 %) |
| gan-faces | gan_face | – | 123 | 67.5 % (83/123; Wilson 58.8 %–75.1 %) | 48.0 % (59/123; Wilson 39.3 %–56.7 %) |
| gpt-4o-image-1 | t2i | – | 30 | 60.0 % (18/30; Wilson 42.3 %–75.4 %) | 16.7 % (5/30; Wilson 7.3 %–33.6 %) |
| flux2 | t2i | F2 | 64 | 57.8 % (37/64; Wilson 45.6 %–69.1 %) | 25.0 % (16/64; Wilson 16.0 %–36.8 %) |
| gpt-image-1.5-2 | t2i | F3 | 50 | 48.0 % (24/50; Wilson 34.8 %–61.5 %) | 8.0 % (4/50; Wilson 3.2 %–18.8 %) |

Rows with few fakes carry wide Wilson intervals (`gpt-4o-image-1`, 30 fakes: 42.3 %–75.4 %); read them as "not validated", not as
a ranking.

### Leave-one-generator-out (image refits without the family)

| Fold | Families | Fakes | TPR at τ_0.05 | AUC | ECE |
|---|---|---|---|---|---|
| F1 | midjourney | 277 | 98.2 % (272/277; Wilson 95.8 %–99.2 %) | 0.9932 | 0.0562 |
| F2 | flux2 | 134 | 58.2 % (78/134; Wilson 49.7 %–66.2 %) | 0.9098 | 0.1131 |
| F3 | gpt-image-1.5-2 | 107 | 47.7 % (51/107; Wilson 38.4 %–57.0 %) | 0.8179 | 0.1754 |
| F4 | nano-banana | 198 | 83.3 % (165/198; Wilson 77.5 %–87.9 %) | 0.9648 | 0.0470 |
| F5 | seedream4 | 146 | 75.3 % (110/146; Wilson 67.8 %–81.6 %) | 0.9311 | 0.0814 |
| F6 | qwen-image | 190 | 85.3 % (162/190; Wilson 79.5 %–89.6 %) | 0.9707 | 0.0488 |
| F7 | imagen4 | 178 | 92.1 % (164/178; Wilson 87.2 %–95.3 %) | 0.9859 | 0.0535 |

Macro TPR at τ_0.05 over F1–F7 is 0.7716 against ≥ 0.75 (G1a PASS). The weakest fold, F3 GPT-image-1.5 / 2, reaches 0.4766
against ≥ 0.50 (G1b FAIL). Macro ECE is 0.0822 against ≤ 0.08 (G4b FAIL).

### Video, per vendor (refits without the vendor)

| Fold | Fakes | TPR at τ_0.05 | AUC |
|---|---|---|---|
| V-Kling | 19 | 78.9 % (15/19; Wilson 56.7 %–91.5 %) | 0.9740 |
| V-Seedance | 19 | 26.3 % (5/19; Wilson 11.8 %–48.8 %) | 0.7654 |
| V-Sora | 19 | 42.1 % (8/19; Wilson 23.1 %–63.7 %) | 0.8616 |
| V-Veo | 18 | 72.2 % (13/18; Wilson 49.1 %–87.5 %) | 0.9688 |
| V-WanHunyuan | 18 | 100.0 % (18/18; Wilson 82.4 %–100.0 %) | 1.0000 |

Macro TPR at τ_0.05 is 0.6392 against ≥ 0.80 (G9a FAIL); the minimum is Seedance, 0.2632.

### Real strata (image false positives)

| Stratum | n | FPR at τ_0.05 | FPR at τ_0.01 |
|---|---|---|---|
| R1 | 387 | 9.3 % (36/387; Wilson 6.8 %–12.6 %) | 1.0 % (4/387; Wilson 0.4 %–2.6 %) |
| R2 | 409 | 0.0 % (0/409; Wilson 0.0 %–0.9 %; rule of three ≤ 0.7 %) | 0.0 % (0/409; Wilson 0.0 %–0.9 %; rule of three ≤ 0.7 %) |
| R3 | 479 | 12.3 % (59/479; Wilson 9.7 %–15.6 %) | 1.3 % (6/479; Wilson 0.6 %–2.7 %) |
| R4 | 405 | 0.5 % (2/405; Wilson 0.1 %–1.8 %) | 0.0 % (0/405; Wilson 0.0 %–0.9 %; rule of three ≤ 0.7 %) |
| R5 | 450 | 3.1 % (14/450; Wilson 1.9 %–5.2 %) | 0.2 % (1/450; Wilson 0.0 %–1.2 %) |
| R6 | 22 | 9.1 % (2/22; Wilson 2.5 %–27.8 %) | 0.0 % (0/22; Wilson 0.0 %–14.9 %; rule of three ≤ 13.6 %) |

G3 requires ≤ 5 % per stratum with a Wilson upper bound ≤ 7.5 %: R2, R4 and R5 pass; R1, R3 and R6 fail (R6 on 22 reals).

### Robustness (image, fixed τ_0.05, reference: native files)

| Transform | Reals | Fakes | ΔTPR | ΔFPR | ΔAUC | Inconclusive under t |
|---|---|---|---|---|---|---|
| jpeg75 | 296 | 261 | 0.0383 | 0.0034 | 0.0132 | 13.5 % |
| jpeg50 | 296 | 261 | 0.0077 | 0.0473 | 0.0289 | 6.1 % |
| webp80 | 296 | 261 | 0.0268 | 0.0068 | 0.0181 | 12.4 % |
| resize0.5 | 296 | 261 | 0.1839 | 0.0000 | 0.0995 | 14.4 % |
| crop70 | 296 | 261 | 0.0230 | -0.0034 | 0.0247 | 10.1 % |
| screenshot_jpeg | 296 | 261 | 0.8621 | -0.0608 | 0.1141 | 8.4 % |
| social_ig | 296 | 261 | 0.0230 | 0.1588 | 0.0636 | 24.6 % |
| social_x | 296 | 261 | 0.0307 | -0.0034 | 0.0085 | 12.9 % |
| noise3 | 296 | 261 | 0.0153 | 0.0135 | 0.0136 | 16.9 % |
| upscale2x | 276 | 261 | 0.3065 | -0.0580 | 0.1478 | 8.4 % |
| exif_forge | 296 | 245 | 0.0000 | 0.0000 | 0.0000 | 13.3 % |

ΔTPR is clean − transformed, ΔFPR transformed − clean. G5a fails on `resize0.5`, G5b on `screenshot_jpeg` and `upscale2x`, G5c on
`jpeg50` and `social_ig`. Not scored for the admitted detectors (D-40 scope): `jpeg95`, `screenshot`, `launder`, `rephoto_proxy`.

### Field conditions (EA-02 slice validation map, image `test`)

| Condition | Reals | Fakes | AUC | Measured at |
|---|---|---|---|---|
| JPEG q ≥ 90, ≥ 1 MP, no face | 489 | 205 | 0.9863 | `jpeg_q>=90_1mp_noface` |
| JPEG q 70–89 | 751 | 298 | 0.9803 | `jpeg_q70-89` |
| PNG / lossless (no JPEG quantisation tables) | 365 | 1116 | 0.9162 | `lossless` |
| face present | 867 | 891 | 0.9525 | `face` |
| social recompressed (social_ig, social_x) | 592 | 522 | 0.9399 | `social` |
| screenshot (screenshot, screenshot_jpeg) | 296 | 261 | 0.8615 | `screenshot` |

The runtime slice maps are in `results.md`: 42 image leaves (2 measured with ≥ 100 items per class, 40 carrying a measured
parent's AUC), 19 video leaves and 2 audio leaves. The H-7 envelope record (`docs/v2/bench/h7_envelope.md`) finds 23 of 23
original field images inside the validated envelope.

### Calibration, conformal sets, invariants

- **Conformal coverage at α = 0.05 (image).** Reals 95.2 % (2048/2152; Wilson 94.2 %–96.0 %), AI 95.2 % (1755/1843; Wilson
  94.2 %–96.1 %) (G6a PASS). Inconclusive: 18.2 % of `image/test` (729/3995; G6b PASS) and 24.6 % under `social_ig` (137/557; G6c PASS).
- **Additivity (G15a).** Contributions add up to the fused logit within 3.553 × 10⁻¹⁵ on 1,000 items.
- **Monotonicity (G15b).** 0 of 977 random positive detector moves lowered the score.
- **Reproducibility (G12).** Two `calibrate` runs give identical artefacts except `created_at`.

### Latency (`docs/v2/bench/perf.md`, J-4)

| Profile | image 256 px face | image 800 px | image 12 MP | image 24 MP camera | video 9 s 1080p | video 60 s 1080p | audio 5 min | text 1,000 words | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| `quick` | 0.53 s | 0.54 s | 0.54 s | 2.15 s | 2.62 s | 8.32 s | 0.62 s | 2.07 s | 3.32 GB |
| `full` | 0.55 s | 1.06 s | 3.08 s | 4.12 s | 9.22 s | 50.79 s | 5.18 s | 3.11 s | 7.40 GB |
| `deep` | 0.56 s | 1.07 s | 3.12 s | 4.14 s | 9.75 s | 60.86 s | 5.21 s | 2.60 s | 4.18 GB |

p95 is nearest-rank over 1–5 warm runs per workload (at this n, the maximum): client wall time from upload to `done`. The 24 MP
workload is an R1 Commons camera original (6000 × 4000, EXIF), copied unmodified (J-4). G13a (image p95 ≤ 25 s, maximum over the
256 px, 800 px and 24 MP workloads) passes at 4.12 s and G13b (9 s 1080p video ≤ 120 s) at 9.22 s on `full`. The host was shared
(1-minute load 6.2–7.0 on 18 CPUs), so these figures bound an idle host from above. The H-10 run (v2.1) measured `full` 800 px
1.07 s and 9 s video 10.78 s.

### Field test v2.1 (H-9, D-42; `docs/field_test.md`)

| Kind | Set | n | Decision rate | Decision accuracy | p(AI) side | p(AI) AUC | Inconclusive causes | Not finished |
|---|---|---|---|---|---|---|---|---|
| image | original | 23 | 22/23 (96 %) | 22/22 (100 %) | 22/23 (96 %) | 1.000 | conformal band 1 | 0 |
| image | v2.1 | 80 | 71/80 (89 %) | 66/71 (93 %) | 72/80 (90 %) | 0.993 | conformal band 8 | 1 |
| video | original | 8 | 7/8 (88 %) | 7/7 (100 %) | 7/8 (88 %) | 1.000 | conformal band 1 | 0 |
| video | v2.1 | 32 | 19/32 (59 %) | 19/19 (100 %) | 30/32 (94 %) | 0.988 | conformal band 13 | 0 |
| document | original | 20 | 20/20 (100 %) | 20/20 (100 %) | 20/20 (100 %) | 1.000 | – | 0 |

The v2.1 set holds 112 items outside vb-1.1 by `group_key`: 40 AI stills (FLUX.2, GPT-image-1.5 / 2, Midjourney, nano-banana,
Qwen-Image, Seedream 4), 40 Commons camera originals, 16 AI clips (Kling, Sora 2, Veo 3.1, Wan / Hunyuan) and 16 OpenVid-1M real
clips. These are small samples without intervals. On the original manifest v2.0 published no image or video decision (0/23, 0/8);
v2.1 decides 22/23 and 7/8, all correct. The per-family table and every item are in `docs/field_test.md`.

### Goals

`docs/v2/bench/gate-v2.1-final.txt`: 38 PASS, 14 FAIL and 6 MISSING of 58 checks. `check_shortfalls.py` PASS: 10 of 28 goals are
below target (G1b, G3, G4b, G5a, G5b, G5c, G7b, G7c, G9a, G10a). Each has its values, mitigations and consequence in
`docs/v2/bench/shortfalls.md` and its statement in `docs/limitations.md`.

---

## v2.0 on Veritas-Bench vb-1.0-core (superseded by v2.1 above)

- **Source.** `docs/v2/bench/results.json`, generated 2026-09-14T04:55:20Z at git `7b2a82a`, manifest sha256 `d0224343a5c0…`.
  It covers the `test` lockbox of `vb-1.0-core` (D-39: 2,000 images, 150 videos, 300 recordings, split per stratum into test 50 %,
  fit 33 %, cal 17 %).
- **Status.** Automated records, pending human review (D-17).
- **Intervals.** AUC, ECE and Brier carry grouped bootstrap 95 % intervals (B = 2000). Rates carry Wilson 95 % intervals.
- **Thresholds.** τ_α comes from `cal` reals only (rank rule, equal stratum weights). "At τ_0.05" means at the threshold that flags
  5 % of the calibration reals.
- **Commands.** Each admission record below lists the commands that reproduce its numbers.

### Systems

| Kind | Artefact | Fitted detectors | Profiles that load it | Record |
|---|---|---|---|---|
| Image | `fuser-image-2.0.0-alpha` (sha256 `3486cde0d4bf…`) | `img.hf.ateeqq`, `img.hf.dima806`, `img.dfdc_b7` and the forensic and metadata detectors listed in the record | `full`, `deep` | `docs/v2/bench/admissions/fuser-image-2.0.0-alpha.md` |
| Video | `fuser-video-2.0.0-alpha` (`b86c4746e184…`) | frame detectors (ateeqq, dima806, DFDC-B7), `vid.codec.ffprobe`, `aud.gustking_xlsr` | `full`, `deep` | `docs/v2/bench/admissions/fuser-video-2.0.0-alpha.md` |
| Audio | `fuser-audio-2.0.0-alpha` (`2ea7921de5b4…`) | `aud.gustking_xlsr` | `quick`, `full`, `deep` | `docs/v2/bench/admissions/fuser-audio-2.0.0-alpha.md` |

`quick` image and video analyses use the uncalibrated fallback fuser.

### Results per set (`test`)

| Set | Reals | Fakes | AUC [95 % CI] | TPR at τ_0.05 | FPR at τ_0.05 | ECE [95 % CI] | Brier [95 % CI] | EER | Inconclusive |
|---|---|---|---|---|---|---|---|---|---|
| `image/test` | 344 | 544 | 0.8944 [0.8703, 0.9156] | 47.2 % (257/544; 43.1 %–51.4 %) | 5.5 % (19/344; 3.6 %–8.5 %) | 0.0626 [0.0538, 0.0991] | 0.1287 [0.1142, 0.1445] | – | 41.4 % (368/888) |
| `image/test/t2i` | 344 | 381 | 0.9159 [0.8939, 0.9351] | 53.8 % (205/381; 48.8 %–58.7 %) | 5.5 % | 0.0586 [0.0487, 0.0904] | 0.1128 [0.0983, 0.1290] | – | 36.8 % (267/725) |
| `image/test/personalised` | 344 | 27 | 0.8591 [0.8081, 0.9044] | 33.3 % (9/27; 18.6 %–52.2 %) | 5.5 % | 0.1721 [0.1322, 0.2446] | 0.1612 [0.1258, 0.1996] | – | 31.8 % (118/371) |
| `image/test/edit` | 344 | 40 | 0.7918 [0.7219, 0.8513] | 12.5 % (5/40; 5.5 %–26.1 %) | 5.5 % | 0.2008 [0.1579, 0.2550] | 0.1892 [0.1497, 0.2331] | – | 33.9 % (130/384) |
| `image/test/faceswap_vs_R5` | 87 | 60 | 0.9042 [0.8546, 0.9456] | 48.3 % (29/60; 36.2 %–60.7 %) | 6.9 % (6/87; 3.2 %–14.2 %) | 0.1044 [0.0778, 0.1646] | 0.1268 [0.0986, 0.1570] | – | 45.6 % (67/147) |
| `image/test/gan_face_vs_R5` | 87 | 36 | 0.7573 [0.6600, 0.8452] | 25.0 % (9/36; 13.8 %–41.1 %) | 6.9 % (6/87; 3.2 %–14.2 %) | 0.1680 [0.1424, 0.2838] | 0.2148 [0.1627, 0.2689] | – | 47.2 % (58/123) |
| `image/test/legacy` | 58 | 32 | 0.9865 [0.9649, 0.9989] | 43.8 % (14/32; 28.2 %–60.7 %) | 0.0 % (0/58; 0.0 %–6.2 %) | 0.1434 [0.1205, 0.2057] | 0.0725 [0.0446, 0.1028] | – | 30.0 % (27/90) |
| `video/test` | 38 | 32 | 0.7599 [0.6301, 0.8754] | 0.0 % (0/32; 0.0 %–10.7 %) | 0.0 % (0/38; 0.0 %–9.2 %) | 0.1471 [0.1407, 0.2869] | 0.1963 [0.1473, 0.2474] | – | 100.0 % (70/70) |
| `audio/test` | 75 | 76 | 0.8670 [0.7993, 0.9283] | 5.3 % (4/76; 2.1 %–12.8 %) | 1.3 % (1/75; 0.2 %–7.2 %) | 0.0890 [0.0870, 0.1791] | 0.1555 [0.1117, 0.1992] | 22.4 % | 61.6 % (93/151) |
| `audio/test/asvspoof5` | 17 | 19 | 0.8545 [0.7023, 0.9606] | 5.3 % (1/19; 0.9 %–24.6 %) | 0.0 % (0/17; 0.0 %–18.4 %) | 0.2369 [0.1824, 0.3595] | 0.1805 [0.1046, 0.2713] | 23.5 % | 80.6 % (29/36) |
| `audio/test/in_the_wild` | 37 | 19 | 0.8485 [0.7000, 0.9786] | 0.0 % (0/19; 0.0 %–16.8 %) | 2.7 % (1/37; 0.5 %–13.8 %) | 0.1458 [0.1258, 0.2927] | 0.1635 [0.0802, 0.2487] | 26.3 % | 55.4 % (31/56) |

- **Image.** On the same items, the v0.3 baseline had AUC 0.8116 (`docs/v2/bench/baseline-v0.3.md`). Image τ_0.05 = 0.8186 and
  τ_0.01 = 0.8636, both from 110 `cal` reals.
- **Video.** τ is not established: the rank rule needs 19 `cal` reals and core has 16. At τ_0.05 no fake is flagged, and every test
  video's conformal set holds both classes (70 of 70 inconclusive). The video fuser ranks items but does not yet support a decision
  at a known error rate.
- **Audio.** τ_0.05 = 0.8336 (25 `cal` reals); τ_0.01 is not established. The v0.3 fused score on the same items had AUC 0.8637
  and ECE 0.1278, and the same G10 values: the v2 fuser is a monotone transform of one detector.

### Image, per generator family (test fakes at the image τ)

| Family | Threat | LGO fold | Fakes | TPR at τ_0.05 | TPR at τ_0.01 |
|---|---|---|---|---|---|
| `flux1` | t2i | – | 19 | 94.7 % (18/19; 75.4 %–99.1 %) | 89.5 % (17/19; 68.6 %–97.1 %) |
| `imagen4` | t2i | F7 | 23 | 87.0 % (20/23; 67.9 %–95.5 %) | 82.6 % (19/23; 62.9 %–93.0 %) |
| `hunyuanimage` | t2i | – | 7 | 85.7 % (6/7; 48.7 %–97.4 %) | 71.4 % (5/7; 35.9 %–91.8 %) |
| `qwen-image` | t2i | F6 | 25 | 76.0 % (19/25; 56.6 %–88.5 %) | 60.0 % (15/25; 40.7 %–76.6 %) |
| `gpt-4o-image-1` | t2i | – | 4 | 75.0 % (3/4; 30.1 %–95.4 %) | 75.0 % (3/4; 30.1 %–95.4 %) |
| `flux2` | t2i | F2 | 20 | 70.0 % (14/20; 48.1 %–85.5 %) | 50.0 % (10/20; 29.9 %–70.1 %) |
| `hidream` | t2i | – | 15 | 66.7 % (10/15; 41.7 %–84.8 %) | 60.0 % (9/15; 35.7 %–80.2 %) |
| `modern-t2i-faces` | t2i | – | 24 | 66.7 % (16/24; 46.7 %–82.0 %) | 50.0 % (12/24; 31.4 %–68.6 %) |
| `ideogram` | t2i | – | 12 | 58.3 % (7/12; 32.0 %–80.7 %) | 50.0 % (6/12; 25.4 %–74.6 %) |
| `nano-banana` | t2i | F4 | 28 | 53.6 % (15/28; 35.8 %–70.5 %) | 46.4 % (13/28; 29.5 %–64.2 %) |
| `recraft` | t2i | – | 13 | 53.8 % (7/13; 29.1 %–76.8 %) | 38.5 % (5/13; 17.7 %–64.5 %) |
| `long-tail` | t2i | – | 46 | 50.0 % (23/46; 36.1 %–63.9 %) | 45.7 % (21/46; 32.2 %–59.8 %) |
| `faceswap-stills` | faceswap | – | 60 | 48.3 % (29/60; 36.2 %–60.7 %) | 41.7 % (25/60; 30.1 %–54.3 %) |
| `gpt-image-1.5-2` | t2i | F3 | 16 | 43.8 % (7/16; 23.1 %–66.8 %) | 25.0 % (4/16; 10.2 %–49.5 %) |
| `legacy` | t2i | – | 32 | 43.8 % (14/32; 28.2 %–60.7 %) | 40.6 % (13/32; 25.5 %–57.7 %) |
| `imagen3` | t2i | – | 14 | 42.9 % (6/14; 21.4 %–67.4 %) | 42.9 % (6/14; 21.4 %–67.4 %) |
| `sd3` | t2i | – | 19 | 36.8 % (7/19; 19.1 %–59.0 %) | 36.8 % (7/19; 19.1 %–59.0 %) |
| `personalised` | personalised | – | 27 | 33.3 % (9/27; 18.6 %–52.2 %) | 33.3 % (9/27; 18.6 %–52.2 %) |
| `seedream4` | t2i | F5 | 21 | 33.3 % (7/21; 17.2 %–54.6 %) | 23.8 % (5/21; 10.6 %–45.1 %) |
| `gan-faces` | gan_face | – | 36 | 25.0 % (9/36; 13.8 %–41.1 %) | 8.3 % (3/36; 2.9 %–21.8 %) |
| `midjourney` | t2i | F1 | 33 | 18.2 % (6/33; 8.6 %–34.4 %) | 18.2 % (6/33; 8.6 %–34.4 %) |
| `partial-edits` | edit | – | 40 | 12.5 % (5/40; 5.5 %–26.1 %) | 7.5 % (3/40; 2.6 %–19.9 %) |
| `seedream3` | t2i | – | 10 | 0.0 % (0/10; 0.0 %–27.8 %) | 0.0 % (0/10; 0.0 %–27.8 %) |

Families with fewer than about 20 fakes have intervals wider than 40 points. Read those rows as "not validated", not as a ranking.

### Leave-one-generator-out (image refits without the family)

| Fold | Family | Fakes (all splits) | AUC | TPR at τ_0.05 | ECE |
|---|---|---|---|---|---|
| F1 | midjourney | 71 | 0.8548 | 21.1 % (15/71; 13.2 %–32.0 %) | 0.1873 |
| F2 | flux2 | 39 | 0.9242 | 51.3 % (20/39; 36.2 %–66.1 %) | 0.0864 |
| F3 | gpt-image-1.5-2 | 32 | 0.8851 | 50.0 % (16/32; 33.6 %–66.4 %) | 0.1080 |
| F4 | nano-banana | 50 | 0.9420 | 58.0 % (29/50; 44.2 %–70.6 %) | 0.0818 |
| F5 | seedream4 | 46 | 0.7564 | 28.3 % (13/46; 17.3 %–42.5 %) | 0.1924 |
| F6 | qwen-image | 50 | 0.9515 | 64.0 % (32/50; 50.1 %–75.9 %) | 0.0881 |
| F7 | imagen4 | 48 | 0.9744 | 87.5 % (42/48; 75.3 %–94.1 %) | 0.1119 |

Macro TPR at τ_0.05 is 0.5145 against a target of ≥ 0.75 (G1a, FAIL). The weakest family reaches 0.2113 against ≥ 0.50 (G1b, FAIL).
Macro ECE is 0.1223 (G4b ≤ 0.08, FAIL).

### Video, per vendor (refits without the vendor)

| Fold | Fakes | AUC | TPR at τ_0.05 | ECE |
|---|---|---|---|---|
| V-Kling | 5 | 0.7000 | 0.0 % (0/5; 0.0 %–43.4 %) | 0.4247 |
| V-Seedance | 5 | 0.5368 | 0.0 % (0/5; 0.0 %–43.4 %) | 0.4549 |
| V-Sora | 5 | 0.9632 | 0.0 % (0/5; 0.0 %–43.4 %) | 0.3176 |
| V-Veo | 4 | 0.7171 | 0.0 % (0/4; 0.0 %–49.0 %) | 0.4210 |
| V-WanHunyuan | 4 | 1.0000 | 0.0 % (0/4; 0.0 %–49.0 %) | 0.2359 |

Macro TPR at τ_0.05 is 0.0 (G9a, FAIL): no refit can set τ from 15–16 `cal` reals.

### Real strata (image false positives)

Strata as assigned in `docs/v2/bench/sources.yaml`; item counts per split are in `docs/v2/bench/vb-1.0-core_summary.md`.

| Stratum | Sources | Reals | FPR at τ_0.05 | FPR at τ_0.01 |
|---|---|---|---|---|
| R1 | camera originals: Commons Canon EOS R5 and Sony α7 IV, RAISE-1k | 63 | 17.5 % (11/63; 10.0 %–28.6 %) | 3.2 % (2/63; 0.9 %–10.9 %) |
| R2 | smartphone photos: Samsung, Google Pixel, Apple iPhone | 26 | 0.0 % (0/26; 0.0 %–12.9 %) | 0.0 % (0/26; 0.0 %–12.9 %) |
| R3 | social-platform reals: Reddit, Tumblr, Bluesky | 76 | 2.6 % (2/76; 0.7 %–9.1 %) | 2.6 % (2/76; 0.7 %–9.1 %) |
| R4 | photo datasets: COCO val, Flickr30k, ImageNet, DOCCI | 88 | 0.0 % (0/88; 0.0 %–4.2 %) | 0.0 % (0/88; 0.0 %–4.2 %) |
| R5 | real faces: FFHQ-256, DeepFakeFace (wiki) | 87 | 6.9 % (6/87; 3.2 %–14.2 %) | 3.4 % (3/87; 1.2 %–9.7 %) |
| R6 | hard captures: HDR, night, long exposure, light trails, selfie, shallow depth of field | 4 | 0.0 % (0/4; 0.0 %–49.0 %) | 0.0 % (0/4; 0.0 %–49.0 %) |

G3 requires ≤ 5 % per stratum with a Wilson upper bound ≤ 7.5 %. Only R4 passes; the other strata fail on the point estimate or
the upper bound.

### Robustness (image, 278-item subset, reference: native files)

| Transform | ΔAUC | ΔTPR at τ_0.05, clean − transformed (pts) | FPR clean → transformed | Inconclusive |
|---|---|---|---|---|
| `jpeg75` | −0.0004 | −10.0 | 3.4 % (5/148) → 6.8 % (10/148; 3.7 %–12.0 %) | 31.3 % (87/278) |
| `resize0.5` | −0.0083 | −8.5 | 3.4 % → 4.1 % (6/148; 1.9 %–8.6 %) | 34.2 % (95/278) |
| `screenshot_jpeg` | +0.0096 | −12.3 | 3.4 % → 10.1 % (15/148; 6.2 %–16.0 %) | 31.7 % (88/278) |
| `social_ig` | −0.0001 | −9.2 | 3.4 % → 6.1 % (9/148; 3.2 %–11.2 %) | 31.3 % (87/278) |

A negative ΔTPR means the transformed images were flagged more often than the clean ones. On all four transforms, more reals were
flagged too, so these transforms do not degrade recall; they push scores upward overall.
Not scored: `jpeg95`, `jpeg50`, `webp80`, `crop70`, `screenshot`, `social_x`, `launder`, `noise3`, `upscale2x`, `rephoto_proxy`,
`exif_forge`. The fitted forensic detectors have cached rows only for the four transforms above (D-39), so those G5 and G11 rows
are MISSING.

### Condition slices (image `test`)

| Slice | Reals | Fakes | AUC |
|---|---|---|---|
| lossless, shorter side < 512 px, no face | 29 | 59 | 0.8930 |
| lossless, shorter side 512–1024 px, no face | 41 | 253 | 0.7229 |

The other eight slices have fewer than 20 items of a class, and no AUC is reported for them.

### Calibration, conformal sets, invariants

- **Conformal coverage at α = 0.05 (image).** Reals 94.5 % (325/344; 91.5 %–96.4 %), AI 96.5 % (525/544; 94.6 %–97.8 %) (G6a PASS).
- **Additivity (G15a).** Contributions add up to the fused logit within 8.9 × 10⁻¹⁶.
- **Monotonicity (G15b).** A random positive detector change never lowered the score: 0 violations in 888 trials.
- **Reproducibility (G12).** Two `make calibrate` runs give identical artefacts except `created_at`.

### Latency

`docs/v2/bench/perf.md` (X-4) holds per-profile figures, stage and detector breakdowns, and memory. G13a (image p95 ≤ 25 s) passes
at 1.10 s and G13b (9 s 1080p video ≤ 120 s) passes at 8.23 s, measured on the `full` profile on a shared host.

### Goals

The latest gate run (`docs/v2/bench/gate-x4-perf.txt`) passes 18 of 58 checks, fails 20 and has 20 MISSING. Each failing goal is
listed with its measured value, the mitigations attempted and the product consequence in `docs/v2/bench/shortfalls.md`.

---

## v0.3 reference (public datasets and v0.3 pipelines; not v2 results)

The sections below predate v2. Their numbers come from `scripts/eval_image.py` and `scripts/eval_text.py`, run on 2026-08-22 on an
Apple M5 Pro (MPS). Raw tables are in `docs/eval_image.md` and `docs/eval_text.md`; per-sample scores are in `docs/eval_image.json`.

## Goals set for v0.1 and outcome

| Area | Goal | Result | Status |
|---|---|---|---|
| AI-generated photos vs real photos | fused AUC ≥ 0.90, FPR ≤ 5 % at the 0.5 threshold | **AUC 0.972, FPR 2.0 %, TPR 76 %** (n = 330: 180 ELSA_D3 fakes from SD 1.4 / SD 2.1 / SDXL, 150 real COCO + Flickr) | met |
| Face deepfakes (GAN faces + in-the-wild face swaps) vs real faces | face-fused AUC ≥ 0.80 | **AUC 0.939, FPR 10 %, TPR 77 %** (n = 150: 50 StyleGAN faces, 50 WildDeepfake swap frames, 50 FFHQ-style real faces); was 0.886 before the whole-image portrait fix | met |
| Video face swaps | correct band on the held-out sample clips | 2/2 real clips → authentic (0.18, 0.13); 2/3 fake clips → likely AI (0.74, 0.67, medium confidence); **1 miss**: a 1.4 s clip scored 0.40 (uncertain, low confidence) | partially met |
| AI-written text (HC3) | fused AUC ≥ 0.95, FPR ≤ 5 % | **AUC 1.000, acc 0.99, FPR 2.0 %, TPR 100 %** (n = 300) | met |
| Sentence-level highlighting | on spliced human+AI documents the estimated AI share is within ±10 points of truth | 43 % estimated vs ≈ 35–40 % true on the smoke-test splice; sentence bands separate at the splice boundary | met (small sample) |
| Zero-shot text detector | Binoculars calibrated for the chosen model pair | threshold 0.9505, scale 26 (balanced acc 0.79, AUC 0.84 with the Qwen2.5-1.5B pair) | local HC3 tuning only; not a released-reference reproduction (see method audit below) |
| Out-of-distribution text (8 frontier-model paragraphs vs 8 Wikipedia intros) | flag ≥ 50 % of modern LLM text at 0 % FPR | v0.1: 0 / 8 flagged (mean p 0.16). **v0.2 (fine-tuned classifier + re-weighted fusion): 8 / 8 flagged (mean p 0.81), 0 / 8 human false positives, human sentence-level AI share 2 %** | met (see caveats in the fine-tune section) |
| Service | model-free tests, PDF report for every kind, API validation | 14/14 tests pass; reports render for image / video / document; invalid uploads → 415, oversize → 413 | met |

## Per-model findings that shaped the ensemble weights

| Model | Stills (photos AUC / FPR) | Faces AUC | Decision |
|---|---|---|---|
| Ateeqq/ai-vs-human-image-detector (SigLIP) | 0.940 / 10 % | 0.60 | weight 1.0 for stills; **0.3 on video frames** (fires ≈ 1.0 on compressed video) |
| haywoodsloan/ai-image-detector-deploy (Swin-v2) | 0.876 / 8.7 % | 0.45 | weight 1.0 |
| UnivFD CLIP ViT-L/14 probe | 0.852 / 6.7 % | 0.53 | weight 1.4 (generator-agnostic; native-resolution crops) |
| Organika/sdxl-detector (Swin) | 0.834 / 17 % | 0.58 | weight 0.5 (high FPR) |
| DFDC EfficientNet-B7 (2-checkpoint ensemble) | n/a (0 % FPR on photos) | 0.812; 0.84 mean on WildDeepfake swaps | weight 2.0 — primary face-swap detector, leads on video |
| dima806/deepfake_vs_real (ViT) | n/a | **1.000 on whole 256 px portraits**; 0.867 when re-cropped | weight 1.0 on stills (portraits fed whole; upscaled crops ×0.4), 0.25 on video frames |
| Wvolf/ViT_Deepfake_Detection (ViT) | n/a | **1.000 on whole portraits**; 0.849 re-cropped | weight 0.8 on stills, 0.25 on video |
| prithivMLmods/Deep-Fake-Detector-v2 | 0.06–0.12 AUC (inverted) | inverted | **disabled** |

Text: DeBERTa-v3-large (desklib) AUC 1.000 on HC3 — weight 1.5. RoBERTa (Hello-SimpleAI) AUC 1.000 but it was
*trained on HC3*, so this is optimistic — weight 0.6. Binoculars AUC 0.84 — weight 1.0 (historical local ratio variant;
this HC3 result does not establish robustness to classifier distribution shift). Stylometry AUC 0.70 — weight 0.5 (explanatory).

## Field test — realistic AI media from public sources (2026-08-23)

Not a benchmark: 51 items pulled from the open web to mimic what users upload. Full per-item table: `docs/field_test.md`.

| Category | Sources | Result |
|---|---|---|
| Images (23) | AI: Midjourney v6 photoreal prompts, FLUX.1-dev, FLUX.1-schnell · Real: Wikimedia Commons camera originals with EXIF (Canon, Nikon, Sony, Panasonic, vivo, Samsung) | **20/23** — real passed 12/12, AI caught 8/11 (misses: one Midjourney v6 and two FLUX-schnell photoreal images) |
| Videos (8) | AI: Veo 3 / 3.1, Seedance 1 Pro, Pika 2.2, Wan 2.1 (Rapidata T2V sets) · Real: Commons footage (WebM) | **7/8** — AI caught 5/5, real passed 2/3; the third real clip (distant table-tennis players, faces < 120 px) ends *inconclusive* at 0.54 with an explicit small-face caveat |
| Documents (20) | AI: Cosmopedia (Mixtral-written WikiHow/stories) · Real: BBC News, XSum articles | **20/20** |

What the first run exposed and what changed (each fix is principled, then re-checked against the 480-image benchmark — fused AUC 0.972 photos / 0.951 overall, unchanged):
- Bogus sample data: the first Commons pull returned 2 KB HTML throttle pages that the app correctly rejected as "unrecognised image type"; collection now paces requests and verifies magic bytes.
- UnivFD misread 10 MP DSLR photos (0.97 "fake") because 224 px native crops of a 3872 px image are featureless texture → multi-scale crops (native + 1024 px copy) for large images.
- Two classifiers fire on modern smartphone computational photography → Ateeqq weight 1.0 → 0.7; coherent camera EXIF (make, model, exposure, ISO, no editor) now weighs 0.9 as moderate authenticity evidence.
- A fully synthetic FLUX portrait was dragged to 0.47 by face-swap specialists saying "no swap" (true but irrelevant) → their negatives count ×0.25 when ≥ 2 whole-image detectors agree at ≥ 0.9.
- WebM with a reported 1000 fps yielded one decoded frame → fps sanity check, sequential fallback decoding, few-frame caveat.
- DFDC flags small compressed faces → face-swap evidence ×0.3 below 120 px with a caveat; such clips land inconclusive rather than "AI".

Open: Midjourney v6 and FLUX-schnell photoreal stills remain under-detected by the current open-source detector set (3/11 missed here; consistent with the 0.80 TPR on ELSA_D3 SDXL). A detector trained on 2025-era generators is the next lever.

## Audio channel (v0.3): synthetic-voice detection on video sound tracks

`scripts/eval_audio.py`, n = 320 clips: 160 ASVspoof 2021-DF (bona fide vs spoof, codec-compressed) + 160 in-the-wild
(`garystafford/deepfake-audio-detection`).

| Model | ASVspoof-DF AUC | in-the-wild AUC | overall AUC / FPR | Decision |
|---|---|---|---|---|
| Gustking wav2vec2-XLSR | 0.974 | 0.854 | 0.917 / 10.6 % | primary, weight 1.0 |
| mo-thecreator wav2vec2-base | 0.966 | 0.721 | 0.855 / 25.6 % | corroborating, weight 0.5 |
| MattyB95 AST-ASVspoof5 | 0.629 | 0.427 | 0.550 / **76.9 %** | **disabled** — fires on ordinary phone recordings |
| **Configured fusion** | **0.974** | 0.827 | **0.910 / 13.8 %** | channel weight 0.8 |

Fusion rule: a face swap normally keeps genuine audio, so a "bona fide voice" result is weak evidence about the visuals
(weight ×0.25); a synthetic-voice positive counts as manipulation even when faces look genuine (voice clone / dub).
Sample clips after the change: real iPhone clip 0.14 (was 0.24 with the AST model on), face-swap clip 0.65.

## Lessons that changed the pipeline

- **Do not resample what a low-level detector will inspect.** Re-cropping a 256 px portrait and rescaling it took
  the GAN-face detectors from AUC 1.00 to 0.85–0.87 and gave them ~50 % false positives. Portraits are now passed
  whole; small faces that must be upscaled are down-weighted. UnivFD likewise uses native-resolution crops.
- **Specialists' negatives are weak evidence.** A face-*swap* detector saying "no swap" says little about a fully
  synthetic GAN face; whole-frame "real scene" detectors say little about a localised swap. Both asymmetries are
  encoded in the fusion.
- **Heuristics corroborate, they do not decide.** Forensic/temporal heuristics are capped to [0.40, 0.75] and
  weighted 0.25–0.45 so that the absence of an anomaly cannot overturn a strong learned detector.
- **Never drop a weighted signal for being near 0.5.** An early version excluded "strength = none" signals from
  fusion, which silently removed a 0.5999 face signal with weight 1.6 and flipped a verdict.

## Text detector fine-tune (v0.2) — what closed the frontier-model gap

Corpus (`backend/data/corpus/`): human = 98 Wikipedia intros + 200 CNN/DailyMail articles + 250 HC3 human answers;
machine = 151 Qwen2.5-3B-Instruct generations across 20 registers, 24 frontier-model paragraphs written for training,
250 HC3 ChatGPT answers. Fine-tune from `desklib/ai-text-detector-v1.01`: embeddings + lower 12 of 24 DeBERTa layers
frozen (152M trainable), lr 2e-5, 2 epochs, batch 4, max_len 320 (`scripts/finetune_text_detector.py`). Output:
`data/weights/desklib-ft-v2`, picked up automatically when present.

| Set | base | fine-tuned |
|---|---|---|
| Held-out 8 frontier paragraphs + 8 Wikipedia (classifier alone) | TPR 0.25, FPR 0 | TPR 1.00, FPR 0 |
| Same, full pipeline after fusion re-weighting | 0/8 flagged | 8/8 flagged (mean 0.81), 0/8 FP |
| HC3 n=300 (classifier) | AUC 1.000 | AUC 1.000, FPR 2.7 % |
| OOD human: Enron emails / Yelp / IMDB (n=240) FPR | 28 % / 4 % / 0 % | **14 %** / 7 % / 0 % |
| OOD machine: RAID GPT-4 / Cohere / Llama-2 / Mistral / MPT (n=200) TPR | 0.97 / 0.97 / 1.00 / 1.00 / 1.00 | 0.85 / 0.82 / 1.00 / 1.00 / 0.97 |
| OOD overall (n=440) | AUC 0.993, FPR 10.4 % | AUC 0.986, FPR 7.1 % |

Fusion weights changed with the adoption: classifier 1.5 → 2.0 (the only component validated on 2025-26 prose),
RoBERTa 0.6 → 0.3 and Binoculars 1.0 → 0.6 (both vote "human" on modern text; see the pair experiment below).

Caveats, stated plainly: the held-out frontier paragraphs share an author and session with the 24 training
paragraphs, so 8/8 demonstrates transfer within that author's range, not across all current models; no API
access was available to generate a multi-model 2025-26 corpus (the script supports it: `build_ai_corpus.py
--claude`). The ~12-point recall loss on RAID GPT-4/Cohere is real; RAID is in-distribution for the base model's
own training data, so its 0.97 there is optimistic. Enron emails remain the hardest human register (14 % FPR).

## Sentence-level false positives on encyclopedic human text

On 8 random Wikipedia intros, 26 % of human sentences were initially flagged AI-written (all three sentence signals
agree on formulaic, low-perplexity factual sentences — LLMs are trained on Wikipedia). Median-of-3 smoothing alone
did not help (24 %) because the runs are contiguous. Adding a **document prior** — a half-strength log-odds shift of
every sentence toward the document-level classifier/Binoculars estimate — cut it to **11 %** (mean per-document AI
share 7 %, max 34 %) while leaving the spliced human+ChatGPT test unchanged (AI run localised exactly, share 0.43 vs
≈0.50 true) and a pure ChatGPT answer at 100 %. Every adjusted sentence carries a reason explaining the adjustment.

## Binoculars model-pair experiment (negative result)

`scripts/experiment_binoculars_pair.py`, HC3 n = 200, OOD set = 8 frontier-model paragraphs + 8 Wikipedia intros.

| Pair | HC3 AUC | OOD AUC | frontier-model paragraphs flagged |
|---|---|---|---|
| Qwen2.5-1.5B / 1.5B-Instruct (shipped) | 0.841 | 0.27 | 0 / 8 |
| Qwen2.5-3B / 3B-Instruct | 0.775 | 0.47 | 0 / 8 |

These historical numbers evaluate the **Veritas local ratio variant**, not the released Binoculars implementation.
The local code uses base-model next-token loss divided by H(instruction-tuned model, base model), with both
terms excluding final-position logits. The released implementation reverses those model roles and includes
all token-ID-nonpadding positions in the denominator. Tokenization, context coverage and precision also differ.
The orientation existed in the initial local implementation; the HC3 threshold and sigmoid slope were tuned
for that variant. Swapping the formula while keeping those settings would invalidate its calibration.

The larger pair reduced HC3 AUC while increasing OOD AUC on this tiny reused panel; neither pair flagged
any of its eight frontier-model paragraphs. These observations do not establish that zero-shot ratios fail
on current models, that larger pairs cannot help, or that supervised training is the only remaining option.
The serving 1.5B variant and all historical scores remain unchanged. A separately versioned reference-oriented
candidate needs fresh source-disjoint calibration and validation before any replacement or accuracy claim.
See the [20 September formula audit](v2/qa/binoculars-formula-audit-2026-09-20/README.md) for pinned primary
sources, reproducible synthetic comparisons and the distinction between arithmetic conformity and accuracy.

## Dataset notes (important for anyone re-running)

- `Hemg/AI-Generated-vs-Real-Images-Datasets` was removed: visual inspection showed its "AI art" split is
  dominated by ordinary photographs; its labels cannot be trusted.
- `poloclub/diffusiondb`, `nlphuji/flickr30k`, `Hello-SimpleAI/HC3` use legacy loading scripts; the harness reads
  raw files / parquet directly instead.
- `xingjunm/WildDeepfake` streaming only yielded fake shards within the time budget; real faces come from the
  FFHQ-style real split of `JamieWithofs/Deepfake-and-real-images`.

## Known misses and next steps (v0.2 goals)

0. **Small GAN portraits** (≤ 256 px StyleGAN faces) are ranked well (AUC 0.886) but under-flagged at 0.5; add a GAN-face specialist with low FPR (see `limitations.md`).

1. **Short / low-quality video swaps**: the 1.4 s DFDC-style clip was missed. Next: add a second DFDC checkpoint
   (ensemble of 2–3 of the author's five), denser sampling for clips < 3 s, and an audio-visual sync signal.
2. **Newer generators**: ELSA_D3 covers SD 1.4–SDXL; add FLUX / Midjourney v6 / Imagen samples as they become
   available in public datasets and re-evaluate UnivFD and the classifiers.
3. **Binoculars strength**: run `make eval-text` with the Falcon-7B pair on a GPU host (expected AUC > 0.95) and
   ship both calibrations.
4. **Out-of-distribution text — mitigated in v0.2** (0/8 → 8/8 on the held-out frontier paragraphs; see the fine-tune section).
   Remaining: (a) ~~stronger Binoculars pair~~ — tested, does not help; (b) extend the fine-tune corpus with multi-model 2025-26 output (`build_ai_corpus.py --claude` once credentials exist) and re-check RAID GPT-4 recall; the original plan was to fine-tune the DeBERTa head
   on a 2025–2026 corpus (Claude / GPT / Gemini / Llama outputs across registers, plus paraphrases); (c) report
   the "human-like" band for polished text as *uninformative* until (a)/(b) reach AUC ≥ 0.90 on a held-out
   frontier-model set.
5. **Calibration**: fit a single Platt/isotonic layer on the fused log-odds per kind so that "70 %" means 70 %
   empirically; today the probability is a calibrated-weight ensemble, not a frequency guarantee.
