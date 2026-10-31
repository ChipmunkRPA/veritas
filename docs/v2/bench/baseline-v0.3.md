# Veritas v0.3 baseline on vb-1.0-core (B-5)

Bench `vb-1.0-core` (D-39), manifest sha256 `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`, `test` split (the
lockbox; access appended to `test_access.log`). Generated 2026-09-14 from git `d46efee`. Automated record (ralph loop), pending
human review.

**Result: FAIL, as expected.** 24 PASS, 23 FAIL, 11 MISSING of 58 checks. This is the v0.3 reference the v2 system is compared
with (I-14, V-8, A-1, Q-5); it is not a v2 result.

## Systems measured

| Modality | System (score-cache key) | Variant | Items scored |
|---|---|---|---|
| image | `v1.image_fused@107574c6b0fa` (v0.3 fused image score) | `norm` | 344 reals, 544 fakes; 182 C2PA items excluded across the splits |
| video | `v1.video_fused@a6030abaa903` (v0.3 video pipeline) | `native` | 38 reals, 32 fakes |
| audio | `v1.audio_fused@4eea2f720e5f` (v0.3 voice detectors) | `native` | 74 reals, 76 fakes, 1 without a score |

Scoring jobs (bg.sh): `b5-core-image` (2,000 images, 0 failed), `b5-core-robust`, `b5-core-audio` (v1 300 of 300),
`b5-core-video` (v1 150 of 150, 0 failed). Thresholds come from `cal` reals with equal stratum weights: image τ_0.05 = 0.6701 and
τ_0.01 = 0.8522 (n = 110); audio τ_0.05 = 0.9486 (n = 25), τ_0.01 not established; video τ not established (n = 16).

## Headline numbers (95 % intervals)

| Set | AUC (grouped bootstrap CI) | TPR at τ_0.05 (Wilson) | FPR at τ_0.05 (Wilson) | ECE |
|---|---|---|---|---|
| image test | 0.8116 (0.7731–0.8465) | 34.4 % (30.5–38.5 %) | 8.1 % (5.7–11.5 %) | 0.1072 |
| image test, t2i | 0.8155 (0.7768–0.8514) | 35.2 % (30.5–40.1 %) | 8.1 % | 0.1051 |
| image test, edit | 0.7123 (0.6269–0.7991) | 15.0 % (7.1–29.1 %) | 8.1 % | 0.1932 |
| image LGO F1–F7 | per fold 0.7384–0.8950 | macro 36.2 %, minimum 20.0 % | – | macro 0.1413 |
| video test | 0.6933 (0.5663–0.8167) | 0.0 % (0.0–10.7 %) | 0.0 % (0.0–9.2 %) | 0.1964 |
| audio test | 0.8637 (0.7908–0.9247) | 5.3 % (2.1–12.8 %) | 1.4 % (0.2–7.3 %) | 0.1278 |

Audio EER: test 19.7 %, `asvspoof5` 23.5 %, `in_the_wild` 16.2 %. Full tables (families, strata, robustness, slices, video LGO) in
`baseline-v0.3/results.md`; raw metrics in `baseline-v0.3/results.json` and `error_rates.json`.

## Gate table

`vbench_gate.py` output, verbatim (`baseline-v0.3/gate.txt`, exit 1):

```text
goal   part                                      value  op     target  status   detail
--------------------------------------------------------------------------------------
G1a    macro                                    0.3620  >=     0.7500  FAIL     
G1b    min_family                               0.2000  >=     0.5000  FAIL     
G2     image/test/t2i                           0.0551  >=     0.5500  FAIL     21/381, Wilson 0.036–0.083
G3     R1                                       0.1746  <=     0.0500  FAIL     11/63, Wilson 0.100–0.286; Wilson upper ≤ 0.075
G3     R2                                       0.0000  <=     0.0500  FAIL     0/26, Wilson 0.000–0.129; Wilson upper ≤ 0.075
G3     R3                                       0.1316  <=     0.0500  FAIL     10/76, Wilson 0.073–0.226; Wilson upper ≤ 0.075
G3     R4                                       0.0000  <=     0.0500  PASS     0/88, Wilson 0.000–0.042; Wilson upper ≤ 0.075
G3     R5                                       0.0690  <=     0.0500  FAIL     6/87, Wilson 0.032–0.142; Wilson upper ≤ 0.075
G3     R6                                       0.2500  <=     0.0500  FAIL     1/4, Wilson 0.046–0.699; Wilson upper ≤ 0.075
G3     screenshot                               0.1034  <=     0.0500  FAIL     6/58, Wilson 0.048–0.208; Wilson upper ≤ 0.075
G4a    image/test                               0.1072  <=     0.0500  FAIL     
G4b    macro                                    0.1413  <=     0.0800  FAIL     
G4c    image/test                               0.1909  <=     0.1200  FAIL     
G5a    jpeg95                                   0.0308  <=     0.1000  PASS     
G5a    jpeg75                                   0.0308  <=     0.1000  PASS     
G5a    webp80                                   0.0846  <=     0.1000  PASS     
G5a    resize0.5                                0.1769  <=     0.1000  FAIL     
G5a    crop70                                   0.1077  <=     0.1000  FAIL     
G5b    jpeg50                                   0.0538  <=     0.2000  PASS     
G5b    social_ig                                0.0308  <=     0.2000  PASS     
G5b    social_x                                 0.0462  <=     0.2000  PASS     
G5b    screenshot_jpeg                         -0.0385  <=     0.2000  PASS     
G5b    noise3                                  -0.0923  <=     0.2000  PASS     
G5b    upscale2x                               -0.0692  <=     0.2000  PASS     
G5c    jpeg95                                   0.0000  <=     0.0200  PASS     
G5c    jpeg75                                  -0.0068  <=     0.0200  PASS     
G5c    jpeg50                                  -0.0203  <=     0.0200  PASS     
G5c    webp80                                   0.0000  <=     0.0200  PASS     
G5c    resize0.5                               -0.0270  <=     0.0200  PASS     
G5c    crop70                                  -0.0473  <=     0.0200  PASS     
G5c    screenshot                               0.0405  <=     0.0200  FAIL     
G5c    screenshot_jpeg                          0.0405  <=     0.0200  FAIL     
G5c    social_ig                               -0.0338  <=     0.0200  PASS     
G5c    social_x                                -0.0068  <=     0.0200  PASS     
G5c    launder                                 -0.0676  <=     0.0200  PASS     
G5c    noise3                                  -0.0068  <=     0.0200  PASS     
G5c    upscale2x                                0.0072  <=     0.0200  PASS     
G5c    rephoto_proxy                           -0.0213  <=     0.0200  PASS     
G5c    exif_forge                                    –  <=     0.0200  MISSING  metric not in results
G6a    real                                          –  >=     0.9400  MISSING  metric not in results
G6a    ai                                            –  >=     0.9400  MISSING  metric not in results
G6b    image/test                                    –  <=     0.2000  MISSING  metric not in results
G6c    image/robustness/social_ig                    –  <=     0.4000  MISSING  metric not in results
G7a    image/test/faceswap_vs_R5                0.8067  >=     0.9000  FAIL     
G7b    image/test/faceswap_vs_R5                0.3833  >=     0.7000  FAIL     23/60, Wilson 0.271–0.510
G7c    image/test/gan_face_vs_R5                0.2222  >=     0.8500  FAIL     8/36, Wilson 0.117–0.381
G8     image/test/edit                          0.1500  >=     0.5000  FAIL     6/40, Wilson 0.071–0.291
G9a    macro                                    0.0000  >=     0.8000  FAIL     
G9b    video/test/real                          0.0000  <=     0.0500  PASS     0/38, Wilson 0.000–0.092
G10a   audio/test/asvspoof5                     0.2353  <=     0.1000  FAIL     
G10b   audio/test/in_the_wild                   0.0270  <=     0.1000  PASS     1/37, Wilson 0.005–0.138
G11    exif_forge                                    –  <=     0.0300  MISSING  metric not in results
G12    calibrate_x2                                  –  ==       true  MISSING  metric not in results
G13a   image_256px_face, image_800px                 –  <=         25  MISSING  metric not in results
G13b   video_9s_1080p                                –  <=        120  MISSING  metric not in results
G14    image/test/legacy                        0.9359  >=     0.9500  FAIL     
G15a   image/test                                    –  <=  1.000e-09  MISSING  metric not in results
G15b   image/test                                    –  ==          0  MISSING  metric not in results
24 PASS, 23 FAIL, 11 MISSING of 58 checks → FAIL
```

## Reading the table

- **Robustness rows (G5).** The baseline's robustness table covers 13 transforms on 278 robustness items because the v0.3
  image scorer's cached rows from the stopped full-vb-1.0 job (`b5-score`) include them; the v2 core robustness job scores only
  `jpeg75`, `social_ig`, `screenshot_jpeg` and `resize0.5` (D-39). Negative ΔTPR values mean more fakes flagged after the
  transform.
- **G9b PASS is not evidence of a working video threshold.** τ_α is the conformal rank ⌈(n + 1)(1 − α)⌉ of the `cal` reals
  (`app/bench/metrics.py`), which exists only for n ≥ 19 at α = 0.05 and n ≥ 99 at α = 0.01. Video has 16 `cal` reals, so no
  video is flagged at τ: TPR 0 of 32 and FPR 0 of 38 both follow from that. The video AUC is 0.6933 (0.5663–0.8167).
- **G3 on R2 and R6 cannot pass on core.** A Wilson upper bound ≤ 7.5 % needs n ≥ 48 even with 0 false positives; R2 has 26
  test reals and R6 has 4 (`shortfalls.md`).
- **MISSING goals.** G5c `exif_forge` and G11: no `exif_forge` rows (51 perturbation errors on WebP inputs, and not a D-39 core
  transform). G6a–c: the v0.3 score has no conformal sets. G12 and G15a/b concern a fitted v2 fuser artefact, which a score key
  does not have. G13a/b: latency comes from `make bench` (X-4).
- **Compared with vb-mini.** None; goals are judged on core only (D-39).

## Reproduce

```bash
cd backend
.venv/bin/python scripts/vbench_report.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --score-key v1.image_fused@107574c6b0fa --audio-score-key v1.audio_fused@4eea2f720e5f \
  --video-score-key v1.video_fused@a6030abaa903 --eval-split test --bench vb-1.0-core --out ../docs/v2/bench/baseline-v0.3/
.venv/bin/python scripts/vbench_gate.py --results ../docs/v2/bench/baseline-v0.3/results.json --goals ../docs/v2/bench/goals.yaml \
  --manifest data/vbench/core/manifest.parquet
```
