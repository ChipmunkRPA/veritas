# vb-1.0 summary

Generated 2026-09-13T16:05:37+00:00 by `backend/scripts/vbench_split.py` from `backend/data/vbench/manifest.parquet`.

- **Manifest:** 14468 items, sha256 `fc17fe098b69c8f87526c9bb855eeb3784ca5b9e8e8536ca32c7bc5e55dffbf4` (after the split and audit columns were written).
- **Splits:** `StratifiedGroupKFold(20, shuffle=True, random_state=7)` by (modality, threat, family, real stratum) over `group_key`; folds 0–6 fit, 7–10 cal, 11–19 test (ML §2.6). In-the-wild fakes → `wild`, diagnostic sources → `diagnostic`, sources failing the label audit → `excluded`.
- **Group disjointness:** 0 group keys span more than one of fit / cal / test.
- **Label audit (D-06, agent):** 58 sources, 57 passed, 1 failed (`docs/v2/bench/audit.md`).

## Splits

| Split | Items | Real | Fake | Groups | Share of fit ∪ cal ∪ test |
|---|---|---|---|---|---|
| `fit` | 5185 | 1966 | 3219 | 3566 | 37.3 % |
| `cal` | 2681 | 818 | 1863 | 2000 | 19.3 % |
| `test` | 6027 | 1843 | 4184 | 4535 | 43.4 % |
| `wild` | 300 | 0 | 300 | 300 | – |
| `diagnostic` | 260 | 105 | 155 | 260 | – |
| `excluded` | 15 | 15 | 0 | 14 | – |

## Real strata

| Modality | Stratum | fit | cal | test | wild | diagnostic | excluded | total |
|---|---|---|---|---|---|---|---|---|
| audio | A-bonafide | 215 | 119 | 266 | 0 | 0 | 0 | 600 |
| image | CFE | 0 | 0 | 0 | 0 | 105 | 0 | 105 |
| image | R1 | 503 | 93 | 204 | 0 | 0 | 0 | 800 |
| image | R2 | 243 | 21 | 60 | 0 | 0 | 0 | 324 |
| image | R3 | 268 | 153 | 347 | 0 | 0 | 0 | 768 |
| image | R4 | 315 | 180 | 405 | 0 | 0 | 0 | 900 |
| image | R5 | 315 | 180 | 405 | 0 | 0 | 0 | 900 |
| image | R6 | 14 | 11 | 25 | 0 | 0 | 0 | 50 |
| video | V-general | 92 | 55 | 123 | 0 | 0 | 15 | 285 |
| video | V-talking-head | 1 | 6 | 8 | 0 | 0 | 0 | 15 |

## Test reals per stratum

Strata with fewer than 360 `test` reals are flagged: the gate reports their false-positive rates with Wilson intervals, and `shortfalls.md` names the sample size as the cause (D-38).

| Modality | Stratum | Test reals | Flag |
|---|---|---|---|
| audio | A-bonafide | 266 | below 360 |
| image | R1 | 204 | below 360 |
| image | R2 | 60 | below 360 |
| image | R3 | 347 | below 360 |
| image | R4 | 405 | – |
| image | R5 | 405 | – |
| image | R6 | 25 | below 360 |
| video | V-general | 123 | below 360 |
| video | V-talking-head | 8 | below 360 |

## Fake families

| Modality | Threat | Family | LGO fold | fit | cal | test | wild | diagnostic | excluded | total |
|---|---|---|---|---|---|---|---|---|---|---|
| audio | tts | asvspoof2021-df | – | 70 | 40 | 90 | 0 | 0 | 0 | 200 |
| audio | tts | asvspoof5 | – | 70 | 40 | 90 | 0 | 0 | 0 | 200 |
| audio | tts | in-the-wild-audio | – | 70 | 40 | 90 | 0 | 0 | 0 | 200 |
| audio | tts | mlaad | – | 140 | 80 | 180 | 0 | 0 | 0 | 400 |
| image | edit | partial-edits | – | 210 | 120 | 270 | 0 | 0 | 0 | 600 |
| image | faceswap | faceswap-stills | – | 245 | 140 | 315 | 0 | 0 | 0 | 700 |
| image | gan_face | gan-faces | – | 140 | 80 | 180 | 0 | 0 | 0 | 400 |
| image | personalised | personalised | – | 105 | 60 | 135 | 0 | 0 | 0 | 300 |
| image | t2i | communityforensics-eval | – | 0 | 0 | 0 | 0 | 155 | 0 | 155 |
| image | t2i | flux1 | – | 140 | 80 | 180 | 0 | 0 | 0 | 400 |
| image | t2i | flux2 | F2 | 87 | 52 | 117 | 0 | 0 | 0 | 256 |
| image | t2i | gpt-4o-image-1 | – | 86 | 50 | 108 | 0 | 0 | 0 | 244 |
| image | t2i | gpt-image-1.5-2 | F3 | 105 | 60 | 135 | 0 | 0 | 0 | 300 |
| image | t2i | hidream | – | 49 | 28 | 63 | 0 | 0 | 0 | 140 |
| image | t2i | hunyuanimage | – | 21 | 12 | 31 | 0 | 0 | 0 | 64 |
| image | t2i | ideogram | – | 51 | 32 | 71 | 0 | 0 | 0 | 154 |
| image | t2i | imagen3 | – | 51 | 32 | 67 | 0 | 0 | 0 | 150 |
| image | t2i | imagen4 | F7 | 89 | 52 | 117 | 0 | 0 | 0 | 258 |
| image | t2i | legacy | – | 138 | 80 | 182 | 0 | 0 | 0 | 400 |
| image | t2i | long-tail | – | 175 | 100 | 225 | 0 | 0 | 0 | 500 |
| image | t2i | midjourney | F1 | 141 | 81 | 180 | 0 | 0 | 0 | 402 |
| image | t2i | modern-t2i-faces | – | 112 | 54 | 129 | 0 | 0 | 0 | 295 |
| image | t2i | nano-banana | F4 | 140 | 80 | 180 | 0 | 0 | 0 | 400 |
| image | t2i | qwen-image | F6 | 93 | 56 | 126 | 0 | 0 | 0 | 275 |
| image | t2i | recraft | – | 44 | 27 | 59 | 0 | 0 | 0 | 130 |
| image | t2i | sd3 | – | 70 | 40 | 90 | 0 | 0 | 0 | 200 |
| image | t2i | seedream3 | – | 30 | 19 | 39 | 0 | 0 | 0 | 88 |
| image | t2i | seedream4 | F5 | 100 | 58 | 132 | 0 | 0 | 0 | 290 |
| image | t2i | wild | – | 0 | 0 | 0 | 300 | 0 | 0 | 300 |
| video | t2v | ai-talking-humans | – | 35 | 20 | 45 | 0 | 0 | 0 | 100 |
| video | t2v | animatediff | – | 10 | 7 | 13 | 0 | 0 | 0 | 30 |
| video | t2v | cogvideox | – | 9 | 6 | 15 | 0 | 0 | 0 | 30 |
| video | t2v | kling | V-Kling | 37 | 23 | 52 | 0 | 0 | 0 | 112 |
| video | t2v | luma-ray | – | 8 | 7 | 15 | 0 | 0 | 0 | 30 |
| video | t2v | marey | – | 8 | 6 | 14 | 0 | 0 | 0 | 28 |
| video | t2v | mochi | – | 7 | 4 | 9 | 0 | 0 | 0 | 20 |
| video | t2v | pika | – | 6 | 4 | 10 | 0 | 0 | 0 | 20 |
| video | t2v | runway | – | 21 | 12 | 27 | 0 | 0 | 0 | 60 |
| video | t2v | seedance | V-Seedance | 14 | 8 | 18 | 0 | 0 | 0 | 40 |
| video | t2v | sora | V-Sora | 49 | 28 | 63 | 0 | 0 | 0 | 140 |
| video | t2v | stable-video | – | 9 | 8 | 13 | 0 | 0 | 0 | 30 |
| video | t2v | veo | V-Veo | 70 | 40 | 90 | 0 | 0 | 0 | 200 |
| video | t2v | videopoet | – | 8 | 6 | 16 | 0 | 0 | 0 | 30 |
| video | t2v | wan-hunyuan | V-WanHunyuan | 56 | 32 | 72 | 0 | 0 | 0 | 160 |
| video | talking_head | talking-head-animation | – | 100 | 59 | 131 | 0 | 0 | 0 | 290 |

## LGO folds

| Fold | Items |
|---|---|
| F1 | 402 |
| F2 | 256 |
| F3 | 300 |
| F4 | 400 |
| F5 | 290 |
| F6 | 275 |
| F7 | 258 |
| V-Kling | 112 |
| V-Seedance | 40 |
| V-Sora | 140 |
| V-Veo | 200 |
| V-WanHunyuan | 160 |

## Sources

| Source | Modality | Audit | Commercial use | fit | cal | test | wild | diagnostic | excluded | total |
|---|---|---|---|---|---|---|---|---|---|---|
| `34data-fastwan2` | video | pass | no | 12 | 3 | 15 | 0 | 0 | 0 | 30 |
| `34data-hunyuanvideo` | video | pass | no | 11 | 10 | 9 | 0 | 0 | 0 | 30 |
| `34data-kling` | video | pass | no | 6 | 7 | 21 | 0 | 0 | 0 | 34 |
| `34data-sora2` | video | pass | no | 20 | 6 | 24 | 0 | 0 | 0 | 50 |
| `34data-veo3` | video | pass | no | 14 | 11 | 25 | 0 | 0 | 0 | 50 |
| `34data-wan2` | video | pass | no | 11 | 7 | 22 | 0 | 0 | 0 | 40 |
| `aigc-lipsync-benchmark` | video | pass | no | 35 | 20 | 45 | 0 | 0 | 0 | 100 |
| `aigibench-test` | image | pass | no | 485 | 224 | 641 | 100 | 0 | 0 | 1450 |
| `asvspoof2021-df` | audio | pass | no | 136 | 86 | 178 | 0 | 0 | 0 | 400 |
| `asvspoof5-test` | audio | pass | yes | 145 | 90 | 165 | 0 | 0 | 0 | 400 |
| `bitmind-nano-banana` | image | pass | yes | 32 | 20 | 48 | 0 | 0 | 0 | 100 |
| `coco-val` | image | pass | no | 93 | 51 | 156 | 0 | 0 | 0 | 300 |
| `commons-r1-camera` | image | pass | yes | 432 | 52 | 116 | 0 | 0 | 0 | 600 |
| `commons-r2-smartphone` | image | pass | yes | 243 | 21 | 60 | 0 | 0 | 0 | 324 |
| `commons-r6-hard-negatives` | image | pass | yes | 14 | 11 | 25 | 0 | 0 | 0 | 50 |
| `commons-video-general` | video | fail | yes | 0 | 0 | 0 | 0 | 0 | 15 | 15 |
| `commons-video-speeches` | video | pass | yes | 1 | 6 | 8 | 0 | 0 | 0 | 15 |
| `communityforensics-eval` | image | pass | no | 0 | 0 | 0 | 0 | 260 | 0 | 260 |
| `deepaction-v1` | video | pass | no | 76 | 51 | 123 | 0 | 0 | 0 | 250 |
| `deepfakeface` | image | pass | no | 183 | 80 | 207 | 0 | 0 | 0 | 470 |
| `elsa-d3-validation` | image | pass | no | 44 | 24 | 62 | 0 | 0 | 0 | 130 |
| `ffhq-256` | image | pass | no | 107 | 64 | 129 | 0 | 0 | 0 | 300 |
| `flickr30k` | image | pass | no | 76 | 41 | 83 | 0 | 0 | 0 | 200 |
| `genvidbench-6.7m` | video | pass | no | 35 | 26 | 39 | 0 | 0 | 0 | 100 |
| `in-the-wild-audio` | audio | pass | yes | 144 | 63 | 193 | 0 | 0 | 0 | 400 |
| `jamiewithofs-test` | image | pass | no | 46 | 25 | 59 | 0 | 0 | 0 | 130 |
| `midjourney-v6-recap` | image | pass | yes | 90 | 49 | 113 | 0 | 0 | 0 | 252 |
| `mlaad-tiny` | audio | pass | no | 140 | 80 | 180 | 0 | 0 | 0 | 400 |
| `mllm-generated` | image | pass | yes | 36 | 25 | 59 | 0 | 0 | 0 | 120 |
| `openfake-core-test` | image | pass | no | 334 | 198 | 413 | 0 | 0 | 0 | 945 |
| `openfake-reddit-test` | image | pass | no | 0 | 0 | 0 | 200 | 0 | 0 | 200 |
| `opensdi-flux` | image | pass | yes | 22 | 12 | 26 | 0 | 0 | 0 | 60 |
| `opensdi-sd15` | image | pass | yes | 19 | 13 | 28 | 0 | 0 | 0 | 60 |
| `opensdi-sd2` | image | pass | yes | 26 | 8 | 26 | 0 | 0 | 0 | 60 |
| `opensdi-sd3` | image | pass | yes | 15 | 16 | 29 | 0 | 0 | 0 | 60 |
| `opensdi-sdxl` | image | pass | yes | 25 | 11 | 24 | 0 | 0 | 0 | 60 |
| `openvid-1m` | video | pass | yes | 69 | 42 | 89 | 0 | 0 | 0 | 200 |
| `qwen-image-bench` | image | pass | yes | 355 | 206 | 399 | 0 | 0 | 0 | 960 |
| `rapidata-bananamark` | image | pass | yes | 61 | 33 | 48 | 0 | 0 | 0 | 142 |
| `rapidata-face-generation` | image | pass | yes | 126 | 60 | 144 | 0 | 0 | 0 | 330 |
| `rapidata-flux2-pro` | image | pass | yes | 1 | 0 | 15 | 0 | 0 | 0 | 16 |
| `rapidata-hidream` | image | pass | yes | 6 | 9 | 25 | 0 | 0 | 0 | 40 |
| `rapidata-hunyuanimage-2.1` | image | pass | yes | 0 | 1 | 12 | 0 | 0 | 0 | 13 |
| `rapidata-i2v-seedance` | video | pass | yes | 14 | 8 | 18 | 0 | 0 | 0 | 40 |
| `rapidata-ideogram-v2` | image | pass | yes | 3 | 3 | 12 | 0 | 0 | 0 | 18 |
| `rapidata-imagen4-and-opponents` | image | pass | yes | 107 | 63 | 190 | 0 | 0 | 0 | 360 |
| `rapidata-imagen4-ultra` | image | pass | yes | 7 | 0 | 1 | 0 | 0 | 0 | 8 |
| `rapidata-openai-4o` | image | pass | yes | 4 | 8 | 32 | 0 | 0 | 0 | 44 |
| `rapidata-recraft-v3` | image | pass | yes | 4 | 0 | 6 | 0 | 0 | 0 | 10 |
| `rapidata-seedream3` | image | pass | yes | 1 | 8 | 4 | 0 | 0 | 0 | 13 |
| `rapidata-t2v-marey` | video | pass | yes | 25 | 16 | 35 | 0 | 0 | 0 | 76 |
| `rapidata-t2v-veo3` | video | pass | yes | 89 | 48 | 113 | 0 | 0 | 0 | 250 |
| `rapidata-t2v-veo3.1` | video | pass | yes | 22 | 11 | 17 | 0 | 0 | 0 | 50 |
| `semi-truths-p2p` | image | pass | no | 31 | 26 | 43 | 0 | 0 | 0 | 100 |
| `sofake-ood-test` | image | pass | no | 740 | 407 | 921 | 0 | 0 | 0 | 2068 |
| `synthbuster-plus-test` | image | pass | no | 312 | 187 | 396 | 0 | 0 | 0 | 895 |
| `talkingheadbench` | video | pass | yes | 100 | 59 | 131 | 0 | 0 | 0 | 290 |
| `wilddeepfake-test` | image | pass | yes | 0 | 75 | 25 | 0 | 0 | 0 | 100 |

## Shortcut test (ML §2.1 P3)

Standardised logistic regression on (JPEG quality estimate, width, height, has_exif), out-of-fold predictions under `StratifiedGroupKFold(5, shuffle=True, random_state=7)` over the image items in fit ∪ cal ∪ test. A lossless file's quality is encoded as 100. The gate applies to `norm`; `native` is a diagnostic, because metadata and container signals are evaluated on native bytes and are expected to differ by source.

| Variant | Real | Fake | AUC | Gate (≤ 0.65) | Coefficients (standardised: q, width, height, has_exif) |
|---|---|---|---|---|---|
| `norm` | 3742 | 6946 | 0.5372 | pass | +0.000, -0.820, -0.084, +0.000 |
| `native` | 3742 | 6946 | 0.7033 | diagnostic | +0.599, -0.737, +0.110, -0.376 |
