# vb-1.0:mini summary

Generated 2026-09-12T14:58:28+00:00 by `backend/scripts/vbench_split.py` from `backend/data/vbench/manifest.parquet`.

- **Manifest:** 1661 items, sha256 `ee0d16216a6b1471f2bf27e58064b0ac7028162e622a3f67d149a42e6bf497a3` (after the split and audit columns were written).
- **Splits:** `StratifiedGroupKFold(20, shuffle=True, random_state=7)` by (modality, threat, family, real stratum) over `group_key`; folds 0–6 fit, 7–10 cal, 11–19 test (ML §2.6). In-the-wild fakes → `wild`, diagnostic sources → `diagnostic`, sources failing the label audit → `excluded`.
- **Group disjointness:** 0 group keys span more than one of fit / cal / test.
- **Label audit (D-06, agent):** 58 sources, 57 passed, 1 failed (`docs/v2/bench/audit.md`).

## Splits

| Split | Items | Real | Fake | Groups | Share of fit ∪ cal ∪ test |
|---|---|---|---|---|---|
| `fit` | 546 | 231 | 315 | 450 | 35.1 % |
| `cal` | 312 | 100 | 212 | 260 | 20.1 % |
| `test` | 698 | 250 | 448 | 590 | 44.9 % |
| `wild` | 30 | 0 | 30 | 30 | – |
| `diagnostic` | 60 | 30 | 30 | 60 | – |
| `excluded` | 15 | 15 | 0 | 14 | – |

## Real strata

| Modality | Stratum | fit | cal | test | wild | diagnostic | excluded | total |
|---|---|---|---|---|---|---|---|---|
| audio | A-bonafide | 21 | 12 | 27 | 0 | 0 | 0 | 60 |
| image | CFE | 0 | 0 | 0 | 0 | 30 | 0 | 30 |
| image | R1 | 52 | 9 | 19 | 0 | 0 | 0 | 80 |
| image | R2 | 33 | 6 | 41 | 0 | 0 | 0 | 80 |
| image | R3 | 30 | 19 | 40 | 0 | 0 | 0 | 89 |
| image | R4 | 32 | 17 | 41 | 0 | 0 | 0 | 90 |
| image | R5 | 30 | 17 | 43 | 0 | 0 | 0 | 90 |
| image | R6 | 17 | 11 | 22 | 0 | 0 | 0 | 50 |
| video | V-general | 10 | 4 | 13 | 0 | 0 | 15 | 42 |
| video | V-talking-head | 6 | 5 | 4 | 0 | 0 | 0 | 15 |

## Fake families

| Modality | Threat | Family | LGO fold | fit | cal | test | wild | diagnostic | excluded | total |
|---|---|---|---|---|---|---|---|---|---|---|
| audio | tts | asvspoof2021-df | – | 7 | 4 | 9 | 0 | 0 | 0 | 20 |
| audio | tts | asvspoof5 | – | 7 | 4 | 9 | 0 | 0 | 0 | 20 |
| audio | tts | in-the-wild-audio | – | 4 | 7 | 9 | 0 | 0 | 0 | 20 |
| audio | tts | mlaad | – | 14 | 8 | 18 | 0 | 0 | 0 | 40 |
| image | edit | partial-edits | – | 21 | 12 | 27 | 0 | 0 | 0 | 60 |
| image | faceswap | faceswap-stills | – | 22 | 12 | 34 | 0 | 0 | 0 | 68 |
| image | gan_face | gan-faces | – | 14 | 8 | 18 | 0 | 0 | 0 | 40 |
| image | personalised | personalised | – | 11 | 8 | 13 | 0 | 0 | 0 | 32 |
| image | t2i | communityforensics-eval | – | 0 | 0 | 0 | 0 | 30 | 0 | 30 |
| image | t2i | flux1 | – | 13 | 9 | 18 | 0 | 0 | 0 | 40 |
| image | t2i | flux2 | F2 | 11 | 7 | 12 | 0 | 0 | 0 | 30 |
| image | t2i | gpt-4o-image-1 | – | 10 | 7 | 13 | 0 | 0 | 0 | 30 |
| image | t2i | gpt-image-1.5-2 | F3 | 11 | 7 | 12 | 0 | 0 | 0 | 30 |
| image | t2i | hidream | – | 6 | 4 | 10 | 0 | 0 | 0 | 20 |
| image | t2i | hunyuanimage | – | 4 | 0 | 12 | 0 | 0 | 0 | 16 |
| image | t2i | ideogram | – | 5 | 4 | 10 | 0 | 0 | 0 | 19 |
| image | t2i | imagen3 | – | 5 | 3 | 8 | 0 | 0 | 0 | 16 |
| image | t2i | imagen4 | F7 | 10 | 7 | 13 | 0 | 0 | 0 | 30 |
| image | t2i | legacy | – | 8 | 8 | 15 | 0 | 0 | 0 | 31 |
| image | t2i | long-tail | – | 15 | 17 | 20 | 0 | 0 | 0 | 52 |
| image | t2i | midjourney | F1 | 12 | 8 | 18 | 0 | 0 | 0 | 38 |
| image | t2i | modern-t2i-faces | – | 10 | 5 | 15 | 0 | 0 | 0 | 30 |
| image | t2i | nano-banana | F4 | 14 | 8 | 18 | 0 | 0 | 0 | 40 |
| image | t2i | qwen-image | F6 | 11 | 6 | 13 | 0 | 0 | 0 | 30 |
| image | t2i | recraft | – | 3 | 4 | 13 | 0 | 0 | 0 | 20 |
| image | t2i | sd3 | – | 5 | 6 | 9 | 0 | 0 | 0 | 20 |
| image | t2i | seedream3 | – | 5 | 1 | 10 | 0 | 0 | 0 | 16 |
| image | t2i | seedream4 | F5 | 10 | 5 | 15 | 0 | 0 | 0 | 30 |
| image | t2i | wild | – | 0 | 0 | 0 | 30 | 0 | 0 | 30 |
| video | t2v | ai-talking-humans | – | 4 | 2 | 4 | 0 | 0 | 0 | 10 |
| video | t2v | animatediff | – | 1 | 0 | 2 | 0 | 0 | 0 | 3 |
| video | t2v | cogvideox | – | 1 | 1 | 1 | 0 | 0 | 0 | 3 |
| video | t2v | kling | V-Kling | 5 | 3 | 7 | 0 | 0 | 0 | 15 |
| video | t2v | luma-ray | – | 0 | 2 | 1 | 0 | 0 | 0 | 3 |
| video | t2v | marey | – | 3 | 0 | 1 | 0 | 0 | 0 | 4 |
| video | t2v | mochi | – | 1 | 0 | 1 | 0 | 0 | 0 | 2 |
| video | t2v | pika | – | 0 | 1 | 1 | 0 | 0 | 0 | 2 |
| video | t2v | runway | – | 0 | 3 | 3 | 0 | 0 | 0 | 6 |
| video | t2v | seedance | V-Seedance | 1 | 2 | 1 | 0 | 0 | 0 | 4 |
| video | t2v | sora | V-Sora | 4 | 4 | 6 | 0 | 0 | 0 | 14 |
| video | t2v | stable-video | – | 1 | 1 | 1 | 0 | 0 | 0 | 3 |
| video | t2v | veo | V-Veo | 7 | 4 | 9 | 0 | 0 | 0 | 20 |
| video | t2v | videopoet | – | 2 | 0 | 1 | 0 | 0 | 0 | 3 |
| video | t2v | wan-hunyuan | V-WanHunyuan | 5 | 4 | 7 | 0 | 0 | 0 | 16 |
| video | talking_head | talking-head-animation | – | 12 | 6 | 11 | 0 | 0 | 0 | 29 |

## LGO folds

| Fold | Items |
|---|---|
| F1 | 38 |
| F2 | 30 |
| F3 | 30 |
| F4 | 40 |
| F5 | 30 |
| F6 | 30 |
| F7 | 30 |
| V-Kling | 15 |
| V-Seedance | 4 |
| V-Sora | 14 |
| V-Veo | 20 |
| V-WanHunyuan | 16 |

## Sources

| Source | Modality | Audit | Commercial use | fit | cal | test | wild | diagnostic | excluded | total |
|---|---|---|---|---|---|---|---|---|---|---|
| `34data-fastwan2` | video | pass | no | 2 | 0 | 1 | 0 | 0 | 0 | 3 |
| `34data-hunyuanvideo` | video | pass | no | 0 | 2 | 1 | 0 | 0 | 0 | 3 |
| `34data-kling` | video | pass | no | 0 | 2 | 3 | 0 | 0 | 0 | 5 |
| `34data-sora2` | video | pass | no | 0 | 2 | 3 | 0 | 0 | 0 | 5 |
| `34data-veo3` | video | pass | no | 1 | 1 | 3 | 0 | 0 | 0 | 5 |
| `34data-wan2` | video | pass | no | 1 | 1 | 2 | 0 | 0 | 0 | 4 |
| `aigc-lipsync-benchmark` | video | pass | no | 4 | 2 | 4 | 0 | 0 | 0 | 10 |
| `aigibench-test` | image | pass | no | 51 | 29 | 55 | 10 | 0 | 0 | 145 |
| `asvspoof2021-df` | audio | pass | no | 12 | 10 | 18 | 0 | 0 | 0 | 40 |
| `asvspoof5-test` | audio | pass | yes | 14 | 6 | 20 | 0 | 0 | 0 | 40 |
| `bitmind-nano-banana` | image | pass | yes | 3 | 2 | 5 | 0 | 0 | 0 | 10 |
| `coco-val` | image | pass | no | 8 | 8 | 14 | 0 | 0 | 0 | 30 |
| `commons-r1-camera` | image | pass | yes | 45 | 5 | 10 | 0 | 0 | 0 | 60 |
| `commons-r2-smartphone` | image | pass | yes | 33 | 6 | 41 | 0 | 0 | 0 | 80 |
| `commons-r6-hard-negatives` | image | pass | yes | 17 | 11 | 22 | 0 | 0 | 0 | 50 |
| `commons-video-general` | video | fail | yes | 0 | 0 | 0 | 0 | 0 | 15 | 15 |
| `commons-video-speeches` | video | pass | yes | 6 | 5 | 4 | 0 | 0 | 0 | 15 |
| `communityforensics-eval` | image | pass | no | 0 | 0 | 0 | 0 | 60 | 0 | 60 |
| `deepaction-v1` | video | pass | no | 6 | 4 | 15 | 0 | 0 | 0 | 25 |
| `deepfakeface` | image | pass | no | 16 | 10 | 21 | 0 | 0 | 0 | 47 |
| `elsa-d3-validation` | image | pass | no | 1 | 4 | 8 | 0 | 0 | 0 | 13 |
| `ffhq-256` | image | pass | no | 9 | 4 | 17 | 0 | 0 | 0 | 30 |
| `flickr30k` | image | pass | no | 11 | 3 | 6 | 0 | 0 | 0 | 20 |
| `genvidbench-6.7m` | video | pass | no | 4 | 1 | 5 | 0 | 0 | 0 | 10 |
| `in-the-wild-audio` | audio | pass | yes | 13 | 11 | 16 | 0 | 0 | 0 | 40 |
| `jamiewithofs-test` | image | pass | no | 4 | 3 | 6 | 0 | 0 | 0 | 13 |
| `midjourney-v6-recap` | image | pass | yes | 5 | 5 | 13 | 0 | 0 | 0 | 23 |
| `mlaad-tiny` | audio | pass | no | 14 | 8 | 18 | 0 | 0 | 0 | 40 |
| `mllm-generated` | image | pass | yes | 4 | 3 | 5 | 0 | 0 | 0 | 12 |
| `openfake-core-test` | image | pass | no | 28 | 19 | 48 | 0 | 0 | 0 | 95 |
| `openfake-reddit-test` | image | pass | no | 0 | 0 | 0 | 20 | 0 | 0 | 20 |
| `opensdi-flux` | image | pass | yes | 1 | 0 | 5 | 0 | 0 | 0 | 6 |
| `opensdi-sd15` | image | pass | yes | 3 | 2 | 1 | 0 | 0 | 0 | 6 |
| `opensdi-sd2` | image | pass | yes | 1 | 2 | 3 | 0 | 0 | 0 | 6 |
| `opensdi-sd3` | image | pass | yes | 3 | 1 | 2 | 0 | 0 | 0 | 6 |
| `opensdi-sdxl` | image | pass | yes | 4 | 1 | 1 | 0 | 0 | 0 | 6 |
| `openvid-1m` | video | pass | yes | 9 | 4 | 7 | 0 | 0 | 0 | 20 |
| `qwen-image-bench` | image | pass | yes | 40 | 16 | 40 | 0 | 0 | 0 | 96 |
| `rapidata-bananamark` | image | pass | yes | 4 | 7 | 6 | 0 | 0 | 0 | 17 |
| `rapidata-face-generation` | image | pass | yes | 12 | 5 | 19 | 0 | 0 | 0 | 36 |
| `rapidata-flux2-pro` | image | pass | yes | 5 | 1 | 0 | 0 | 0 | 0 | 6 |
| `rapidata-hidream` | image | pass | yes | 1 | 4 | 5 | 0 | 0 | 0 | 10 |
| `rapidata-hunyuanimage-2.1` | image | pass | yes | 0 | 0 | 10 | 0 | 0 | 0 | 10 |
| `rapidata-i2v-seedance` | video | pass | yes | 1 | 2 | 1 | 0 | 0 | 0 | 4 |
| `rapidata-ideogram-v2` | image | pass | yes | 1 | 1 | 4 | 0 | 0 | 0 | 6 |
| `rapidata-imagen4-and-opponents` | image | pass | yes | 3 | 24 | 9 | 0 | 0 | 0 | 36 |
| `rapidata-imagen4-ultra` | image | pass | yes | 0 | 1 | 4 | 0 | 0 | 0 | 5 |
| `rapidata-openai-4o` | image | pass | yes | 4 | 3 | 3 | 0 | 0 | 0 | 10 |
| `rapidata-recraft-v3` | image | pass | yes | 0 | 0 | 8 | 0 | 0 | 0 | 8 |
| `rapidata-seedream3` | image | pass | yes | 2 | 0 | 6 | 0 | 0 | 0 | 8 |
| `rapidata-t2v-marey` | video | pass | yes | 8 | 1 | 2 | 0 | 0 | 0 | 11 |
| `rapidata-t2v-veo3` | video | pass | yes | 6 | 9 | 10 | 0 | 0 | 0 | 25 |
| `rapidata-t2v-veo3.1` | video | pass | yes | 3 | 0 | 2 | 0 | 0 | 0 | 5 |
| `semi-truths-p2p` | image | pass | no | 2 | 1 | 7 | 0 | 0 | 0 | 10 |
| `sofake-ood-test` | image | pass | no | 82 | 37 | 101 | 0 | 0 | 0 | 220 |
| `synthbuster-plus-test` | image | pass | no | 27 | 17 | 37 | 0 | 0 | 0 | 81 |
| `talkingheadbench` | video | pass | yes | 12 | 6 | 11 | 0 | 0 | 0 | 29 |
| `wilddeepfake-test` | image | pass | yes | 0 | 0 | 10 | 0 | 0 | 0 | 10 |

## Shortcut test (ML §2.1 P3)

Standardised logistic regression on (JPEG quality estimate, width, height, has_exif), out-of-fold predictions under `StratifiedGroupKFold(5, shuffle=True, random_state=7)` over the image items in fit ∪ cal ∪ test. A lossless file's quality is encoded as 100. The gate applies to `norm`; `native` is a diagnostic, because metadata and container signals are evaluated on native bytes and are expected to differ by source.

| Variant | Real | Fake | AUC | Gate (≤ 0.65) | Coefficients (standardised: q, width, height, has_exif) |
|---|---|---|---|---|---|
| `norm` | 479 | 738 | 0.6079 | pass | +0.000, -1.192, -0.075, +0.000 |
| `native` | 479 | 738 | 0.7414 | diagnostic | +0.578, -1.069, +0.201, -0.541 |
