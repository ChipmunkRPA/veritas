# vb-1.0-core summary

Generated 2026-09-13T22:20:44+00:00 by `backend/scripts/vbench_core.py` from `backend/data/vbench/manifest.parquet` (B-4c, D-39).

- **Source:** vb-1.0, 14468 items, sha256 `fc17fe098b69c8f87526c9bb855eeb3784ca5b9e8e8536ca32c7bc5e55dffbf4`.
- **Core manifest:** `backend/data/vbench/core/manifest.parquet`, 2450 items, sha256 `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`, pinned in `docs/v2/bench/goals.yaml` as `bench: vb-1.0-core`.
- **Pool:** rows with role `core` in `fit`, `cal` or `test` and `audit_ok` not false; in-the-wild, diagnostic and excluded rows are left out (no goal set reads them).
- **Images ≤ 2000:** 300 robustness images first (150 real, 150 fake of the vb-1.0 robustness subset, even over strata), then the rest by largest remainders over (modality · threat · family · real stratum) with at least one item per stratum.
- **Video ≤ 150:** 75 fake even over families (vendors), 75 real over the real strata. **Audio ≤ 300:** 150 real, 150 fake, even over strata.
- **Splits:** each stratum's quota goes to test 50 %, fit 33 %, cal 17 %, capped by availability.
- **Order:** sha256(`core:<item_id>`) inside a (stratum, split) cell, sha256(`core-robustness:<item_id>`) for the robustness images. The same vb-1.0 manifest gives the same core.
- **Robustness transforms on core:** jpeg75, social_ig, screenshot_jpeg, resize0.5 (D-39). The other vb-1.0 transforms are not measured on core.
- **Records:** every gate record measured on core states it and reports confidence intervals (D-39). The counts are small, so the intervals are wide.

## Checks

| Check | Result | Detail |
|---|---|---|
| images ≤ 2000 | PASS | 2000 images |
| video ≤ 150 (75 real / 75 fake) | PASS | 75 real, 75 fake |
| audio ≤ 300 (150 real / 150 fake) | PASS | 150 real, 150 fake |
| image test share ≥ 45 % | PASS | 48.8 % |
| every real stratum and fake family represented | PASS | 52 strata |
| robustness ≤ 300 images | PASS | 150 real, 150 fake |
| no duplicate items | PASS | 2450 items |

## Modalities

| Modality | Items | Real | Fake | fit | cal | test | Test share |
|---|---|---|---|---|---|---|---|
| image | 2000 | 737 | 1263 | 697 | 327 | 976 | 48.8 % |
| video | 150 | 75 | 75 | 48 | 32 | 70 | 46.7 % |
| audio | 300 | 150 | 150 | 100 | 49 | 151 | 50.3 % |

## Strata

Every stratum of the pool with its pool size and the core counts per split.

| Modality | Label | Threat | Stratum | Pool | fit | cal | test | Core |
|---|---|---|---|---|---|---|---|---|
| audio | fake | tts | asvspoof2021-df | 200 | 13 | 6 | 19 | 38 |
| audio | fake | tts | asvspoof5 | 200 | 13 | 6 | 19 | 38 |
| audio | fake | tts | in-the-wild-audio | 200 | 12 | 6 | 19 | 37 |
| audio | fake | tts | mlaad | 400 | 12 | 6 | 19 | 37 |
| audio | real | real | A-bonafide | 600 | 50 | 25 | 75 | 150 |
| image | fake | edit | partial-edits | 600 | 32 | 18 | 53 | 103 |
| image | fake | faceswap | faceswap-stills | 700 | 41 | 20 | 60 | 121 |
| image | fake | gan_face | gan-faces | 400 | 23 | 12 | 36 | 71 |
| image | fake | personalised | personalised | 300 | 19 | 8 | 27 | 54 |
| image | fake | t2i | flux1 | 400 | 22 | 13 | 36 | 71 |
| image | fake | t2i | flux2 | 256 | 17 | 8 | 23 | 48 |
| image | fake | t2i | gpt-4o-image-1 | 244 | 15 | 9 | 22 | 46 |
| image | fake | t2i | gpt-image-1.5-2 | 300 | 18 | 10 | 27 | 55 |
| image | fake | t2i | hidream | 140 | 9 | 5 | 15 | 29 |
| image | fake | t2i | hunyuanimage | 64 | 7 | 2 | 7 | 16 |
| image | fake | t2i | ideogram | 154 | 10 | 5 | 16 | 31 |
| image | fake | t2i | imagen3 | 150 | 12 | 4 | 14 | 30 |
| image | fake | t2i | imagen4 | 258 | 16 | 9 | 23 | 48 |
| image | fake | t2i | legacy | 400 | 25 | 11 | 35 | 71 |
| image | fake | t2i | long-tail | 500 | 27 | 14 | 46 | 87 |
| image | fake | t2i | midjourney | 402 | 26 | 12 | 33 | 71 |
| image | fake | t2i | modern-t2i-faces | 295 | 19 | 10 | 24 | 53 |
| image | fake | t2i | nano-banana | 400 | 24 | 13 | 34 | 71 |
| image | fake | t2i | qwen-image | 275 | 18 | 7 | 25 | 50 |
| image | fake | t2i | recraft | 130 | 10 | 3 | 13 | 26 |
| image | fake | t2i | sd3 | 200 | 14 | 5 | 19 | 38 |
| image | fake | t2i | seedream3 | 88 | 7 | 3 | 10 | 20 |
| image | fake | t2i | seedream4 | 290 | 17 | 9 | 27 | 53 |
| image | real | real | R1 | 800 | 63 | 24 | 69 | 156 |
| image | real | real | R2 | 324 | 42 | 9 | 27 | 78 |
| image | real | real | R3 | 768 | 49 | 26 | 76 | 151 |
| image | real | real | R4 | 900 | 56 | 28 | 88 | 172 |
| image | real | real | R5 | 900 | 56 | 29 | 87 | 172 |
| image | real | real | R6 | 50 | 3 | 1 | 4 | 8 |
| video | fake | t2v | ai-talking-humans | 100 | 2 | 1 | 2 | 5 |
| video | fake | t2v | animatediff | 30 | 2 | 1 | 2 | 5 |
| video | fake | t2v | cogvideox | 30 | 2 | 1 | 2 | 5 |
| video | fake | t2v | kling | 112 | 2 | 1 | 2 | 5 |
| video | fake | t2v | luma-ray | 30 | 2 | 1 | 2 | 5 |
| video | fake | t2v | marey | 28 | 2 | 1 | 2 | 5 |
| video | fake | t2v | mochi | 20 | 2 | 1 | 2 | 5 |
| video | fake | t2v | pika | 20 | 2 | 1 | 2 | 5 |
| video | fake | t2v | runway | 60 | 2 | 1 | 2 | 5 |
| video | fake | t2v | seedance | 40 | 2 | 1 | 2 | 5 |
| video | fake | t2v | sora | 140 | 2 | 1 | 2 | 5 |
| video | fake | t2v | stable-video | 30 | 1 | 1 | 2 | 4 |
| video | fake | t2v | veo | 200 | 1 | 1 | 2 | 4 |
| video | fake | t2v | videopoet | 30 | 1 | 1 | 2 | 4 |
| video | fake | t2v | wan-hunyuan | 160 | 1 | 1 | 2 | 4 |
| video | fake | talking_head | talking-head-animation | 290 | 1 | 1 | 2 | 4 |
| video | real | real | V-general | 270 | 20 | 10 | 30 | 60 |
| video | real | real | V-talking-head | 15 | 1 | 6 | 8 | 15 |

## Robustness subset

Drawn from the vb-1.0 robustness subset (1500 images), whose perturbed copies and native scores are cached. Transforms on core: jpeg75, social_ig, screenshot_jpeg, resize0.5.

| Label | fit | cal | test | Total |
|---|---|---|---|---|
| real | 74 | 19 | 57 | 150 |
| fake | 55 | 25 | 70 | 150 |

## Test reals per stratum

Strata with fewer than 360 `test` reals are flagged: the gate reports their false-positive rates with Wilson intervals, and `shortfalls.md` names the sample size as the cause (D-38).

| Modality | Stratum | Test reals | Flag |
|---|---|---|---|
| audio | A-bonafide | 75 | below 360 |
| image | R1 | 69 | below 360 |
| image | R2 | 27 | below 360 |
| image | R3 | 76 | below 360 |
| image | R4 | 88 | below 360 |
| image | R5 | 87 | below 360 |
| image | R6 | 4 | below 360 |
| video | V-general | 30 | below 360 |
| video | V-talking-head | 8 | below 360 |
