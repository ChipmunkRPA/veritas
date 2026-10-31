# vb-1.1 summary

Generated 2026-09-14T18:01:43+00:00 by `backend/scripts/vbench_split.py` from `backend/data/vbench/manifest.parquet`.

- **Manifest:** 15767 items, sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3` (after the split and audit columns were written).
- **Splits:** `StratifiedGroupKFold(20, shuffle=True, random_state=7)` by (modality, threat, family, real stratum) over `group_key`; folds 0–6 fit, 7–10 cal, 11–19 test (ML §2.6). In-the-wild fakes → `wild`, diagnostic sources → `diagnostic`, sources failing the label audit → `excluded`.
- **Group disjointness:** 0 group keys span more than one of fit / cal / test.
- **Label audit (D-06, agent):** 60 sources, 59 passed, 1 failed (`docs/v2/bench/audit.md`).

## Splits

| Split | Items | Real | Fake | Groups | Share of fit ∪ cal ∪ test |
|---|---|---|---|---|---|
| `fit` | 5445 | 2219 | 3226 | 3845 | 35.8 % |
| `cal` | 2951 | 1073 | 1878 | 2194 | 19.4 % |
| `test` | 6796 | 2634 | 4162 | 5021 | 44.7 % |
| `wild` | 300 | 0 | 300 | 300 | – |
| `diagnostic` | 260 | 105 | 155 | 260 | – |
| `excluded` | 15 | 15 | 0 | 14 | – |

## Real strata

| Modality | Stratum | fit | cal | test | wild | diagnostic | excluded | total |
|---|---|---|---|---|---|---|---|---|
| audio | A-bonafide | 215 | 120 | 265 | 0 | 0 | 0 | 600 |
| image | CFE | 0 | 0 | 0 | 0 | 105 | 0 | 105 |
| image | R1 | 357 | 89 | 403 | 0 | 0 | 0 | 849 |
| image | R2 | 443 | 169 | 412 | 0 | 0 | 0 | 1024 |
| image | R3 | 373 | 216 | 479 | 0 | 0 | 0 | 1068 |
| image | R4 | 315 | 180 | 405 | 0 | 0 | 0 | 900 |
| image | R5 | 350 | 200 | 450 | 0 | 0 | 0 | 1000 |
| image | R6 | 17 | 11 | 22 | 0 | 0 | 0 | 50 |
| video | V-general | 147 | 84 | 189 | 0 | 0 | 15 | 435 |
| video | V-talking-head | 2 | 4 | 9 | 0 | 0 | 0 | 15 |

## Test reals per stratum

Strata with fewer than 300 `test` reals are flagged: the gate reports their false-positive rates with Wilson intervals, and `shortfalls.md` names the sample size as the cause (D-38).

| Modality | Stratum | Test reals | Flag |
|---|---|---|---|
| audio | A-bonafide | 265 | below 300 |
| image | R1 | 403 | – |
| image | R2 | 412 | – |
| image | R3 | 479 | – |
| image | R4 | 405 | – |
| image | R5 | 450 | – |
| image | R6 | 22 | below 300 |
| video | V-general | 189 | below 300 |
| video | V-talking-head | 9 | below 300 |

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
| image | t2i | flux2 | F2 | 88 | 52 | 116 | 0 | 0 | 0 | 256 |
| image | t2i | gpt-4o-image-1 | – | 85 | 51 | 108 | 0 | 0 | 0 | 244 |
| image | t2i | gpt-image-1.5-2 | F3 | 105 | 60 | 135 | 0 | 0 | 0 | 300 |
| image | t2i | hidream | – | 49 | 28 | 63 | 0 | 0 | 0 | 140 |
| image | t2i | hunyuanimage | – | 21 | 13 | 30 | 0 | 0 | 0 | 64 |
| image | t2i | ideogram | – | 52 | 32 | 70 | 0 | 0 | 0 | 154 |
| image | t2i | imagen3 | – | 51 | 32 | 67 | 0 | 0 | 0 | 150 |
| image | t2i | imagen4 | F7 | 90 | 52 | 116 | 0 | 0 | 0 | 258 |
| image | t2i | legacy | – | 140 | 82 | 178 | 0 | 0 | 0 | 400 |
| image | t2i | long-tail | – | 175 | 100 | 225 | 0 | 0 | 0 | 500 |
| image | t2i | midjourney | F1 | 140 | 82 | 180 | 0 | 0 | 0 | 402 |
| image | t2i | modern-t2i-faces | – | 107 | 59 | 129 | 0 | 0 | 0 | 295 |
| image | t2i | nano-banana | F4 | 140 | 80 | 180 | 0 | 0 | 0 | 400 |
| image | t2i | qwen-image | F6 | 94 | 56 | 125 | 0 | 0 | 0 | 275 |
| image | t2i | recraft | – | 44 | 28 | 58 | 0 | 0 | 0 | 130 |
| image | t2i | sd3 | – | 70 | 40 | 90 | 0 | 0 | 0 | 200 |
| image | t2i | seedream3 | – | 30 | 19 | 39 | 0 | 0 | 0 | 88 |
| image | t2i | seedream4 | F5 | 99 | 60 | 131 | 0 | 0 | 0 | 290 |
| image | t2i | wild | – | 0 | 0 | 0 | 300 | 0 | 0 | 300 |
| video | t2v | ai-talking-humans | – | 35 | 20 | 45 | 0 | 0 | 0 | 100 |
| video | t2v | animatediff | – | 9 | 7 | 14 | 0 | 0 | 0 | 30 |
| video | t2v | cogvideox | – | 10 | 7 | 13 | 0 | 0 | 0 | 30 |
| video | t2v | kling | V-Kling | 37 | 24 | 51 | 0 | 0 | 0 | 112 |
| video | t2v | luma-ray | – | 10 | 6 | 14 | 0 | 0 | 0 | 30 |
| video | t2v | marey | – | 9 | 8 | 11 | 0 | 0 | 0 | 28 |
| video | t2v | mochi | – | 7 | 4 | 9 | 0 | 0 | 0 | 20 |
| video | t2v | pika | – | 8 | 4 | 8 | 0 | 0 | 0 | 20 |
| video | t2v | runway | – | 21 | 12 | 27 | 0 | 0 | 0 | 60 |
| video | t2v | seedance | V-Seedance | 14 | 8 | 18 | 0 | 0 | 0 | 40 |
| video | t2v | sora | V-Sora | 49 | 28 | 63 | 0 | 0 | 0 | 140 |
| video | t2v | stable-video | – | 10 | 7 | 13 | 0 | 0 | 0 | 30 |
| video | t2v | veo | V-Veo | 70 | 40 | 90 | 0 | 0 | 0 | 200 |
| video | t2v | videopoet | – | 11 | 5 | 14 | 0 | 0 | 0 | 30 |
| video | t2v | wan-hunyuan | V-WanHunyuan | 56 | 32 | 72 | 0 | 0 | 0 | 160 |
| video | talking_head | talking-head-animation | – | 100 | 60 | 130 | 0 | 0 | 0 | 290 |

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
| `34data-fastwan2` | video | pass | no | 9 | 6 | 15 | 0 | 0 | 0 | 30 |
| `34data-hunyuanvideo` | video | pass | no | 14 | 4 | 12 | 0 | 0 | 0 | 30 |
| `34data-kling` | video | pass | no | 10 | 8 | 16 | 0 | 0 | 0 | 34 |
| `34data-sora2` | video | pass | no | 20 | 8 | 22 | 0 | 0 | 0 | 50 |
| `34data-veo3` | video | pass | no | 16 | 10 | 24 | 0 | 0 | 0 | 50 |
| `34data-wan2` | video | pass | no | 13 | 8 | 19 | 0 | 0 | 0 | 40 |
| `aigc-lipsync-benchmark` | video | pass | no | 35 | 20 | 45 | 0 | 0 | 0 | 100 |
| `aigibench-test` | image | pass | no | 486 | 216 | 648 | 100 | 0 | 0 | 1450 |
| `asvspoof2021-df` | audio | pass | no | 137 | 89 | 174 | 0 | 0 | 0 | 400 |
| `asvspoof5-test` | audio | pass | yes | 139 | 85 | 176 | 0 | 0 | 0 | 400 |
| `bitmind-nano-banana` | image | pass | yes | 35 | 19 | 46 | 0 | 0 | 0 | 100 |
| `coco-val` | image | pass | no | 104 | 56 | 140 | 0 | 0 | 0 | 300 |
| `commons-r1-camera` | image | pass | yes | 285 | 47 | 317 | 0 | 0 | 0 | 649 |
| `commons-r2-smartphone` | image | pass | yes | 249 | 23 | 52 | 0 | 0 | 0 | 324 |
| `commons-r6-hard-negatives` | image | pass | yes | 17 | 11 | 22 | 0 | 0 | 0 | 50 |
| `commons-video-general` | video | fail | yes | 0 | 0 | 0 | 0 | 0 | 15 | 15 |
| `commons-video-speeches` | video | pass | yes | 2 | 4 | 9 | 0 | 0 | 0 | 15 |
| `communityforensics-eval` | image | pass | no | 0 | 0 | 0 | 0 | 260 | 0 | 260 |
| `deepaction-v1` | video | pass | no | 85 | 55 | 110 | 0 | 0 | 0 | 250 |
| `deepfakeface` | image | pass | no | 175 | 77 | 218 | 0 | 0 | 0 | 470 |
| `elsa-d3-validation` | image | pass | no | 40 | 24 | 66 | 0 | 0 | 0 | 130 |
| `ffhq-256` | image | pass | no | 147 | 73 | 180 | 0 | 0 | 0 | 400 |
| `flickr30k` | image | pass | no | 69 | 42 | 89 | 0 | 0 | 0 | 200 |
| `genvidbench-6.7m` | video | pass | no | 32 | 23 | 45 | 0 | 0 | 0 | 100 |
| `in-the-wild-audio` | audio | pass | yes | 149 | 66 | 185 | 0 | 0 | 0 | 400 |
| `jamiewithofs-test` | image | pass | no | 48 | 27 | 55 | 0 | 0 | 0 | 130 |
| `midjourney-v6-recap` | image | pass | yes | 87 | 48 | 117 | 0 | 0 | 0 | 252 |
| `mlaad-tiny` | audio | pass | no | 140 | 80 | 180 | 0 | 0 | 0 | 400 |
| `mllm-generated` | image | pass | yes | 52 | 20 | 48 | 0 | 0 | 0 | 120 |
| `openfake-core-test` | image | pass | no | 325 | 203 | 417 | 0 | 0 | 0 | 945 |
| `openfake-reddit-test` | image | pass | no | 0 | 0 | 0 | 200 | 0 | 0 | 200 |
| `opensdi-flux` | image | pass | yes | 23 | 10 | 27 | 0 | 0 | 0 | 60 |
| `opensdi-sd15` | image | pass | yes | 21 | 10 | 29 | 0 | 0 | 0 | 60 |
| `opensdi-sd2` | image | pass | yes | 17 | 10 | 33 | 0 | 0 | 0 | 60 |
| `opensdi-sd3` | image | pass | yes | 23 | 12 | 25 | 0 | 0 | 0 | 60 |
| `opensdi-sdxl` | image | pass | yes | 17 | 8 | 35 | 0 | 0 | 0 | 60 |
| `openvid-1m` | video | pass | yes | 122 | 67 | 161 | 0 | 0 | 0 | 350 |
| `qwen-image-bench` | image | pass | yes | 358 | 187 | 415 | 0 | 0 | 0 | 960 |
| `rapidata-bananamark` | image | pass | yes | 53 | 37 | 52 | 0 | 0 | 0 | 142 |
| `rapidata-face-generation` | image | pass | yes | 120 | 66 | 144 | 0 | 0 | 0 | 330 |
| `rapidata-flux2-pro` | image | pass | yes | 3 | 1 | 12 | 0 | 0 | 0 | 16 |
| `rapidata-hidream` | image | pass | yes | 6 | 9 | 25 | 0 | 0 | 0 | 40 |
| `rapidata-hunyuanimage-2.1` | image | pass | yes | 0 | 1 | 12 | 0 | 0 | 0 | 13 |
| `rapidata-i2v-seedance` | video | pass | yes | 14 | 8 | 18 | 0 | 0 | 0 | 40 |
| `rapidata-ideogram-v2` | image | pass | yes | 3 | 6 | 9 | 0 | 0 | 0 | 18 |
| `rapidata-imagen4-and-opponents` | image | pass | yes | 103 | 64 | 193 | 0 | 0 | 0 | 360 |
| `rapidata-imagen4-ultra` | image | pass | yes | 5 | 0 | 3 | 0 | 0 | 0 | 8 |
| `rapidata-openai-4o` | image | pass | yes | 4 | 12 | 28 | 0 | 0 | 0 | 44 |
| `rapidata-recraft-v3` | image | pass | yes | 3 | 0 | 7 | 0 | 0 | 0 | 10 |
| `rapidata-seedream3` | image | pass | yes | 1 | 4 | 8 | 0 | 0 | 0 | 13 |
| `rapidata-t2v-marey` | video | pass | yes | 26 | 16 | 34 | 0 | 0 | 0 | 76 |
| `rapidata-t2v-veo3` | video | pass | yes | 88 | 55 | 107 | 0 | 0 | 0 | 250 |
| `rapidata-t2v-veo3.1` | video | pass | yes | 19 | 8 | 23 | 0 | 0 | 0 | 50 |
| `semi-truths-p2p` | image | pass | no | 41 | 18 | 41 | 0 | 0 | 0 | 100 |
| `sofake-ood-test` | image | pass | no | 732 | 434 | 902 | 0 | 0 | 0 | 2068 |
| `synthbuster-plus-test` | image | pass | no | 324 | 190 | 381 | 0 | 0 | 0 | 895 |
| `talkingheadbench` | video | pass | yes | 100 | 60 | 130 | 0 | 0 | 0 | 290 |
| `vision-natives` | image | pass | yes | 194 | 146 | 360 | 0 | 0 | 0 | 700 |
| `vision-social` | image | pass | yes | 105 | 60 | 135 | 0 | 0 | 0 | 300 |
| `wilddeepfake-test` | image | pass | yes | 0 | 100 | 0 | 0 | 0 | 0 | 100 |

## Shortcut test (ML §2.1 P3)

Standardised logistic regression on (JPEG quality estimate, width, height, has_exif), out-of-fold predictions under `StratifiedGroupKFold(5, shuffle=True, random_state=7)` over the image items in fit ∪ cal ∪ test. A lossless file's quality is encoded as 100. The gate applies to `norm`; `native` is a diagnostic, because metadata and container signals are evaluated on native bytes and are expected to differ by source.

| Variant | Real | Fake | AUC | Gate (≤ 0.65) | Coefficients (standardised: q, width, height, has_exif) |
|---|---|---|---|---|---|
| `norm` | 4891 | 6946 | 0.6281 | pass | +0.000, -1.429, +0.211, +0.000 |
| `native` | 4891 | 6946 | 0.7431 | diagnostic | +0.620, -1.188, +0.371, -0.481 |
