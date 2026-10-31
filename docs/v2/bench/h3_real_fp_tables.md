# H-3 tables: real-photo false positives on vb-1.1 fit ∪ cal

Generated 2026-09-14T19:52:35+00:00 by `backend/scripts/real_fp_analysis.py`. Manifest sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`; fuser `fuser-image-2.0.0-alpha` (sha256 `3486cde0d4bf1134fe3616299cf27fc12fe07d4bc4c5a5b96d0999302e6a8f33`). τ_0.05 with equal stratum weights; rates with Wilson 95 % intervals. Numbers only: causes and decisions are in `h3_real_fp_analysis.md`.

## Stand-alone detectors (τ from the detector's own fit ∪ cal reals)

| Detector | n reals | τ | R1 ∪ R2 FPR | R1 ∪ R2 < 8 MP | R1 ∪ R2 ≥ 8 MP | R1 ∪ R2, abstentions not flagged | D-41 check | R1 | R2 | R3 | R4 | R5 | R6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `face.gend_clip_l14` | 889 | 0.9321 | 11.4 % (23/201; 7.7–16.6) | 3.4 % (2/58; 1.0–11.7) | 14.7 % (21/143; 9.8–21.4) | 2.2 % (23/1036; 1.5–3.3) | n < 300 | 12.4 % (22/177; 8.4–18.1) | 4.2 % (1/24; 0.7–20.2) | 5.3 % (8/152; 2.7–10.0) | 6.8 % (3/44; 2.3–18.2) | 1.2 % (6/485; 0.6–2.7) | 0.0 % (0/7; 0.0–35.4) |
| `img.commfor384` | 2202 | 0.4717 | 5.2 % (54/1035; 4.0–6.7) | 0.9 % (4/450; 0.3–2.3) | 8.5 % (50/585; 6.5–11.1) | 5.2 % (54/1036; 4.0–6.7) | clear | 5.4 % (23/424; 3.6–8.0) | 5.1 % (31/611; 3.6–7.1) | 4.7 % (27/580; 3.2–6.7) | 3.7 % (11/298; 2.1–6.5) | 0.0 % (0/262; 0.0–1.4) | 11.1 % (3/27; 3.9–28.1) |
| `img.dda` | 2320 | 0.5554 | 4.8 % (50/1035; 3.7–6.3) | 0.9 % (4/450; 0.3–2.3) | 7.9 % (46/585; 5.9–10.3) | 4.8 % (50/1036; 3.7–6.3) | clear | 1.4 % (6/424; 0.7–3.1) | 7.2 % (44/611; 5.4–9.5) | 3.9 % (23/584; 2.6–5.8) | 4.7 % (19/401; 3.1–7.3) | 1.5 % (4/273; 0.6–3.7) | 11.1 % (3/27; 3.9–28.1) |
| `img.dfdc_b7` | 889 | 0.7526 | 8.5 % (17/201; 5.3–13.1) | 6.9 % (4/58; 2.7–16.4) | 9.1 % (13/143; 5.4–14.9) | 1.6 % (17/1036; 1.0–2.6) | n < 300 | 7.9 % (14/177; 4.8–12.8) | 12.5 % (3/24; 4.3–31.0) | 7.9 % (12/152; 4.6–13.3) | 0.0 % (0/44; 0.0–8.0) | 0.8 % (4/485; 0.3–2.1) | 0.0 % (0/7; 0.0–35.4) |
| `img.faces.yunet` | 0 | – | – | – | – | 0.0 % (0/1036; 0.0–0.4) | n < 300 | – | – | – | – | – | – |
| `img.forensic.copymove` | 24 | 0.7500 (max, ties) | 100.0 % (13/13; 77.2–100.0) | 100.0 % (1/1; 20.7–100.0) | 100.0 % (12/12; 75.8–100.0) | 1.3 % (13/1036; 0.7–2.1) | n < 300 | 100.0 % (12/12; 75.8–100.0) | 100.0 % (1/1; 20.7–100.0) | 100.0 % (5/5; 56.6–100.0) | 100.0 % (5/5; 56.6–100.0) | 100.0 % (1/1; 20.7–100.0) | – |
| `img.forensic.ela` | 2694 | 0.6843 | 0.6 % (6/1035; 0.3–1.3) | 0.4 % (2/450; 0.1–1.6) | 0.7 % (4/585; 0.3–1.7) | 0.6 % (6/1036; 0.3–1.3) | clear | 1.4 % (6/424; 0.7–3.1) | 0.0 % (0/611; 0.0–0.6) | 12.9 % (76/587; 10.5–15.9) | 3.0 % (15/495; 1.8–4.9) | 1.6 % (9/549; 0.9–3.1) | 10.7 % (3/28; 3.7–27.2) |
| `img.forensic.jpeg_dq` | 61 | 0.7500 (max, ties) | 100.0 % (12/12; 75.8–100.0) | 100.0 % (4/4; 51.0–100.0) | 100.0 % (8/8; 67.6–100.0) | 1.2 % (12/1036; 0.7–2.0) | n < 300 | 100.0 % (8/8; 67.6–100.0) | 100.0 % (4/4; 51.0–100.0) | 100.0 % (15/15; 79.6–100.0) | 100.0 % (4/4; 51.0–100.0) | 100.0 % (30/30; 88.6–100.0) | – |
| `img.forensic.noise` | 2694 | 0.6713 | 1.7 % (18/1035; 1.1–2.7) | 1.6 % (7/450; 0.8–3.2) | 1.9 % (11/585; 1.1–3.3) | 1.7 % (18/1036; 1.1–2.7) | clear | 2.6 % (11/424; 1.5–4.6) | 1.1 % (7/611; 0.6–2.3) | 12.3 % (72/587; 9.9–15.2) | 2.4 % (12/495; 1.4–4.2) | 0.7 % (4/549; 0.3–1.9) | 10.7 % (3/28; 3.7–27.2) |
| `img.forensic.spectrum` | 2694 | 0.4995 | 7.6 % (79/1035; 6.2–9.4) | 4.2 % (19/450; 2.7–6.5) | 10.3 % (60/585; 8.1–13.0) | 7.6 % (79/1036; 6.2–9.4) | clear | 8.0 % (34/424; 5.8–11.0) | 7.4 % (45/611; 5.5–9.7) | 7.0 % (41/587; 5.2–9.3) | 2.4 % (12/495; 1.4–4.2) | 2.4 % (13/549; 1.4–4.0) | 3.6 % (1/28; 0.6–17.7) |
| `img.hf.ateeqq` | 2694 | 0.9997 | 4.6 % (48/1035; 3.5–6.1) | 4.2 % (19/450; 2.7–6.5) | 5.0 % (29/585; 3.5–7.0) | 4.6 % (48/1036; 3.5–6.1) | clear | 10.6 % (45/424; 8.0–13.9) | 0.5 % (3/611; 0.2–1.4) | 8.0 % (47/587; 6.1–10.5) | 2.4 % (12/495; 1.4–4.2) | 4.6 % (25/549; 3.1–6.6) | 3.6 % (1/28; 0.6–17.7) |
| `img.hf.dima806` | 2694 | 0.9935 | 2.0 % (21/1035; 1.3–3.1) | 1.8 % (8/450; 0.9–3.5) | 2.2 % (13/585; 1.3–3.8) | 2.0 % (21/1036; 1.3–3.1) | clear | 4.7 % (20/424; 3.1–7.2) | 0.2 % (1/611; 0.0–0.9) | 4.6 % (27/587; 3.2–6.6) | 1.6 % (8/495; 0.8–3.2) | 15.1 % (83/549; 12.4–18.4) | 3.6 % (1/28; 0.6–17.7) |
| `img.hf.haywoodsloan` | 2694 | 0.9790 | 3.2 % (33/1035; 2.3–4.4) | 1.3 % (6/450; 0.6–2.9) | 4.6 % (27/585; 3.2–6.6) | 3.2 % (33/1036; 2.3–4.4) | clear | 4.2 % (18/424; 2.7–6.6) | 2.5 % (15/611; 1.5–4.0) | 3.2 % (19/587; 2.1–5.0) | 1.6 % (8/495; 0.8–3.2) | 8.2 % (45/549; 6.2–10.8) | 10.7 % (3/28; 3.7–27.2) |
| `img.meta.consistency` | 0 | – | – | – | – | 0.0 % (0/1036; 0.0–0.4) | n < 300 | – | – | – | – | – | – |
| `img.meta.exif` | 0 | – | – | – | – | 0.0 % (0/1036; 0.0–0.4) | n < 300 | – | – | – | – | – | – |
| `img.meta.generator` | 0 | – | – | – | – | 0.0 % (0/1036; 0.0–0.4) | n < 300 | – | – | – | – | – | – |
| `img.meta.qtables` | 0 | – | – | – | – | 0.0 % (0/1036; 0.0–0.4) | n < 300 | – | – | – | – | – | – |
| `img.watermark.dwtdct` | 0 | – | – | – | – | 0.0 % (0/1036; 0.0–0.4) | n < 300 | – | – | – | – | – | – |
| `loc.mesorch` | 0 | – | – | – | – | 0.0 % (0/1036; 0.0–0.4) | n < 300 | – | – | – | – | – | – |
| `prov.c2pa` | 0 | – | – | – | – | 0.0 % (0/1036; 0.0–0.4) | n < 300 | – | – | – | – | – | – |

## Stand-alone R1 ∪ R2 flags per source (k / n at the detector's τ)

| Detector | commons-r1-camera | commons-r2-smartphone | synthbuster-plus-test | vision-natives |
|---|---|---|---|---|
| `face.gend_clip_l14` | 22 / 172 | 0 / 16 | 0 / 5 | 1 / 8 |
| `img.commfor384` | 23 / 310 | 20 / 272 | 0 / 114 | 11 / 339 |
| `img.dda` | 3 / 310 | 43 / 272 | 3 / 114 | 1 / 339 |
| `img.dfdc_b7` | 14 / 172 | 2 / 16 | 0 / 5 | 1 / 8 |
| `img.forensic.copymove` | 12 / 12 | – | – | 1 / 1 |
| `img.forensic.ela` | 4 / 310 | 0 / 272 | 2 / 114 | 0 / 339 |
| `img.forensic.jpeg_dq` | 8 / 8 | 4 / 4 | – | – |
| `img.forensic.noise` | 6 / 310 | 7 / 272 | 5 / 114 | 0 / 339 |
| `img.forensic.spectrum` | 31 / 310 | 26 / 272 | 3 / 114 | 19 / 339 |
| `img.hf.ateeqq` | 44 / 310 | 1 / 272 | 1 / 114 | 2 / 339 |
| `img.hf.dima806` | 19 / 310 | 1 / 272 | 1 / 114 | 0 / 339 |
| `img.hf.haywoodsloan` | 18 / 310 | 3 / 272 | 0 / 114 | 12 / 339 |

## Fused (fuser-image-2.0.0-alpha; τ 0.8252 from 850 cal reals)

| Slice | FPR |
|---|---|
| R1 ∪ R2 | 4.2 % (44/1036; 3.2–5.7) |
| R1 | 10.4 % (44/424; 7.8–13.6) |
| R2 | 0.0 % (0/612; 0.0–0.6) |
| R3 | 2.5 % (15/589; 1.5–4.2) |
| R4 | 0.0 % (0/495; 0.0–0.8) |
| R5 | 6.2 % (34/549; 4.5–8.5) |
| R6 | 0.0 % (0/28; 0.0–12.1) |

## Contributions on the 44 flagged R1 ∪ R2 reals

| Detector | Mean contribution (pp) | Largest positive contribution (items) |
|---|---|---|
| `img.dfdc_b7` | +0.9 | 2 |
| `img.hf.dima806` | +0.4 | 4 |
| `img.hf.ateeqq` | +0.1 | 12 |
| `img.forensic.noise` | +0.0 | 24 |
| `img.forensic.copymove` | +0.0 | 0 |
| `img.forensic.jpeg_dq` | +0.0 | 0 |
| `img.forensic.spectrum` | +0.0 | 0 |
| `img.forensic.ela` | -0.1 | 0 |

## Flagged R1 ∪ R2 reals per source (44 flagged)

| Source | Flagged | FPR | Median s | Median context prior | Fakes of the source: n, median s |
|---|---|---|---|---|---|
| synthbuster-plus-test | 42 / 114 | 36.8 % | 0.813 | 0.886 | 386, 0.833 |
| commons-r1-camera | 2 / 310 | 0.6 % | 0.085 | 0.090 | – |
| commons-r2-smartphone | 0 / 272 | 0.0 % | 0.047 | 0.090 | – |
| vision-natives | 0 / 340 | 0.0 % | 0.045 | 0.090 | – |

## Worst camera-original false positives (contact sheet order, 60 items; 44 at or above τ)

| # | Item | s | Above τ | Stratum | Source | MP | JPEG q |
|---|---|---|---|---|---|---|---|
| 1 | `08c4d92ca878` | 0.975 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 2 | `7a4cc76be130` | 0.947 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 3 | `186224e7f7c4` | 0.946 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 4 | `0b88de624ecc` | 0.944 | yes | R1 | commons-r1-camera | 1.3 | 94 |
| 5 | `c25e0c4d1634` | 0.943 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 6 | `961513e9a917` | 0.939 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 7 | `8ee03c2396fd` | 0.937 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 8 | `8e5b0c16e003` | 0.935 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 9 | `d858fcfc9f79` | 0.920 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 10 | `a6d94dec398e` | 0.916 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 11 | `b5490f0e2db3` | 0.912 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 12 | `1b2e7c1d197c` | 0.900 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 13 | `ce7c4ed70ece` | 0.894 | yes | R1 | commons-r1-camera | 1.2 | 94 |
| 14 | `b99ac38aa2ce` | 0.873 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 15 | `45d28a2a0a3e` | 0.873 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 16 | `848c03a587b4` | 0.870 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 17 | `855da49a6099` | 0.864 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 18 | `bc35b6d88df4` | 0.859 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 19 | `9adcbf61ce34` | 0.853 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 20 | `6183cc5e7571` | 0.845 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 21 | `86b9e3b069bb` | 0.843 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 22 | `8317b2550d60` | 0.842 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 23 | `dedd7de78c97` | 0.839 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 24 | `dbac5c331a40` | 0.838 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 25 | `e48198377fd0` | 0.836 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 26 | `7a252964d854` | 0.836 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 27 | `d28ceab39052` | 0.836 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 28 | `0db1f3250b12` | 0.836 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 29 | `b4c6af167fa9` | 0.833 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 30 | `4d8b7574b330` | 0.832 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 31 | `2102569bb148` | 0.832 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 32 | `afd0d39f20ae` | 0.832 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 33 | `9274a1ed1bca` | 0.831 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 34 | `2b2bc10d9691` | 0.830 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 35 | `147d32239a89` | 0.829 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 36 | `7c41626a304c` | 0.829 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 37 | `422f6bb896f0` | 0.828 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 38 | `60b2541d5cc6` | 0.828 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 39 | `5e7a421b8cbd` | 0.827 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 40 | `04b4a1ad2423` | 0.827 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 41 | `aa4cd70e5ddf` | 0.826 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 42 | `200684722755` | 0.826 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 43 | `f46472aea8e0` | 0.826 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 44 | `b6f4344fc5ca` | 0.825 | yes | R1 | synthbuster-plus-test | 0.7 | – |
| 45 | `1e6696965d2a` | 0.825 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 46 | `b0f11f9ebecd` | 0.825 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 47 | `eeb8f5d97589` | 0.824 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 48 | `13130c0b26ea` | 0.824 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 49 | `6de403a45fc9` | 0.824 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 50 | `da07bd50ad0d` | 0.822 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 51 | `bbf2a50a2536` | 0.819 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 52 | `b2668cea3045` | 0.819 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 53 | `dab6de4b5634` | 0.818 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 54 | `bed49c3ab257` | 0.818 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 55 | `d293ecb88bfe` | 0.818 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 56 | `9e22963803a3` | 0.817 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 57 | `e9f424f6fbf1` | 0.815 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 58 | `804497688118` | 0.814 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 59 | `272a04eb6e80` | 0.814 | no | R1 | synthbuster-plus-test | 0.7 | – |
| 60 | `f29a177bcd3b` | 0.813 | no | R1 | synthbuster-plus-test | 0.7 | – |

Contact sheet `backend/data/vbench/audit/h3_camera_fp.png` (sha256 `b5bbcfdd45017f58a0b2380622254d09748dd85f992f225a17d4b1feb96af5b8`, gitignored).
