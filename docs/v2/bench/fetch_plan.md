# vb-1.0 fetch plan (dry run)

Generated 2026-09-12T08:01:09+00:00 by `make vbench-fetch DRY_RUN=1` from `docs/v2/bench/sources.yaml (sha256 e12dd7920eae)`. No media was downloaded: the dry run reads HF dataset info and tree listings only. `api` sources are not contacted.

- **Sources:** 64 (58 fetchable, 6 skipped, 0 errors)
- **Planned items:** 16790
- **Projected bytes transferred:** 38.01 GB; sum of source byte caps 86.10 GB. The fetcher stops at 40.00 GB of cache in total and never downloads an object above 5.00 GB whole.

| Source | Dataset | Mode | Modality | Licence | Commercial | Items | Files (selected / matched) | Projected | Cap | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| `commons-r1-camera` | wikimedia/commons | api | image | per-file Commons licence | yes | 600 | n/a | 5.86 GB | 7.00 GB | ok |
| `synthbuster-plus-test` | marco-willi/synthbuster-plus | parquet_stream | image | none declared (RAISE is research-use; U) | no | 895 | 9 / 9 | 1.14 GB | 3.50 GB | ok |
| `commons-r2-smartphone` | wikimedia/commons | api | image | per-file Commons licence | yes | 800 | n/a | 3.91 GB | 5.00 GB | ok |
| `sofake-ood-test` | saberzl/So-Fake-OOD | parquet_stream | image | cc-by-nc-4.0 | no | 2200 | 8 / 46 | 3.01 GB | 6.00 GB | ok |
| `coco-val` | detection-datasets/coco | parquet_stream | image | none declared (COCO image licences vary; U) | no | 300 | 2 / 2 | 45.41 MB | 1.00 GB | ok |
| `flickr30k` | nlphuji/flickr30k | zip_range | image | none declared (Flickr image licences vary; U) | no | 200 | 1 / 1 | 27.34 MB | 1.00 GB | ok |
| `openfake-core-test` | ComplexDataLab/OpenFake | parquet_stream | image | cc-by-nc-4.0 | no | 945 | 4 / 13 | 636.77 MB | 6.00 GB | ok |
| `ffhq-256` | merkol/ffhq-256 | parquet_stream | image | none declared (FFHQ is CC BY-NC-SA 4.0; U) | no | 300 | 1 / 15 | 29.30 MB | 1.00 GB | ok |
| `deepfakeface` | OpenRL/DeepFakeFace | zip_range | image | openrail | no | 470 | 2 / 2 | 18.36 MB | 1.50 GB | ok |
| `commons-r6-hard-negatives` | wikimedia/commons | api | image | per-file Commons licence | yes | 500 | n/a | 2.44 GB | 3.00 GB | ok |
| `midjourney-v6-recap` | Photoroom/midjourney-v6-recap | parquet_stream | image | mit (Midjourney terms of service U) | yes | 250 | 2 / 124 | 35.40 MB | 1.00 GB | ok |
| `aigibench-test` | HorizonTEL/AIGIBench | zip_range | image | cc-by-nc-sa-4.0 | no | 1450 | 16 / 16 | 1.42 GB | 4.00 GB | ok |
| `rapidata-flux2-pro` | Rapidata/Flux-2-pro_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 60 | 1 / 69 | 18.87 MB | 600.00 MB | ok |
| `rapidata-bananamark` | Rapidata/bananamark-dataset | parquet_stream | image | CDLA-Permissive-2.0 | yes | 205 | 4 / 4 | 88.89 MB | 1.00 GB | ok |
| `qwen-image-bench` | Qwen/Qwen-Image-Bench | per_file | image | apache-2.0 (generator terms of service U) | yes | 960 | 18000 / 18000 | 947.42 MB | 2.00 GB | ok |
| `mllm-generated` | zr-zhang/MLLM-Generated-Image-Detection-Dataset | per_file | image | apache-2.0 (generator terms of service U) | yes | 120 | 2178 / 2178 | 248.15 MB | 1.00 GB | ok |
| `rapidata-openai-4o` | Rapidata/OpenAI-4o_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 100 | 1 / 22 | 19.53 MB | 600.00 MB | ok |
| `bitmind-nano-banana` | bitmind/nano-banana | parquet_stream | image | mit (Google terms of service U) | yes | 100 | 1 / 31 | 150.00 MB | 1.00 GB | ok |
| `rapidata-seedream3` | Rapidata/Seedream-3_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 75 | 1 / 101 | 16.26 MB | 500.00 MB | ok |
| `rapidata-face-generation` | Rapidata/Face_Generation_Benchmark | parquet_stream | image | CDLA-Permissive-2.0 | yes | 360 | 4 / 4 | 114.96 MB | 1.00 GB | ok |
| `rapidata-imagen4-and-opponents` | Rapidata/Imagen4_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 360 | 10 / 45 | 99.14 MB | 1.50 GB | ok |
| `rapidata-imagen4-ultra` | Rapidata/Imagen-4-ultra-24-7-25_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 50 | 1 / 130 | 11.33 MB | 500.00 MB | ok |
| `rapidata-hidream` | Rapidata/Hidream_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 100 | 1 / 23 | 23.14 MB | 600.00 MB | ok |
| `rapidata-ideogram-v2` | Rapidata/Ideogram-V2_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 60 | 1 / 59 | 15.59 MB | 500.00 MB | ok |
| `rapidata-recraft-v3` | Rapidata/Recraft-v3-24-7-25_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 80 | 1 / 122 | 18.20 MB | 500.00 MB | ok |
| `rapidata-hunyuanimage-2.1` | Rapidata/HunyuanImage-2.1_t2i_human_preference | parquet_stream | image | CDLA-Permissive-2.0 | yes | 100 | 1 / 91 | 38.38 MB | 600.00 MB | ok |
| `elsa-d3-validation` | elsaEU/ELSA_D3 | parquet_stream | image | none declared (U) | no | 130 | 2 / 11 | 33.01 MB | 1.00 GB | ok |
| `jamiewithofs-test` | JamieWithofs/Deepfake-and-real-images | parquet_stream | image | none declared (U) | no | 130 | 1 / 1 | 1.27 MB | 500.00 MB | ok |
| `wilddeepfake-test` | xingjunm/WildDeepfake | per_file | image | apache-2.0 | yes | 100 | 6 / 115 | 200.00 MB | 1.00 GB | ok |
| `opensdi-flux` | nebula/OpenSDI_test | parquet_stream | image | cc-by-sa-4.0 | yes | 60 | 1 / 7 | 9.96 MB | 300.00 MB | ok |
| `opensdi-sd15` | nebula/OpenSDI_test | parquet_stream | image | cc-by-sa-4.0 | yes | 60 | 1 / 19 | 9.96 MB | 300.00 MB | ok |
| `opensdi-sd2` | nebula/OpenSDI_test | parquet_stream | image | cc-by-sa-4.0 | yes | 60 | 1 / 7 | 9.96 MB | 300.00 MB | ok |
| `opensdi-sd3` | nebula/OpenSDI_test | parquet_stream | image | cc-by-sa-4.0 | yes | 60 | 1 / 10 | 9.96 MB | 300.00 MB | ok |
| `opensdi-sdxl` | nebula/OpenSDI_test | parquet_stream | image | cc-by-sa-4.0 | yes | 60 | 1 / 9 | 9.96 MB | 300.00 MB | ok |
| `semi-truths-p2p` | semi-truths/Semi-Truths-Evalset | per_file | image | cc (variant undeclared; U) | no | 100 | 2 / 6 | 200.00 MB | 500.00 MB | ok |
| `openfake-reddit-test` | ComplexDataLab/OpenFake | parquet_stream | image | cc-by-nc-4.0 | no | 200 | 2 / 5 | 123.05 MB | 1.50 GB | ok |
| `communityforensics-eval` | OwensLab/CommunityForensics-Eval | parquet_stream | image | cc-by-nc-sa-4.0 | no | 600 | 6 / 413 | 2.00 GB | 2.00 GB | ok |
| `commons-video-general` | wikimedia/commons | api | video | per-file Commons licence | yes | 200 | n/a | 3.91 GB | 5.00 GB | ok |
| `openvid-1m` | nkp37/OpenVid-1M | zip_range | video | cc-by-4.0 | yes | 200 | 2 / 170 | 1.37 GB | 2.00 GB | ok |
| `commons-video-speeches` | wikimedia/commons | api | video | per-file Commons licence | yes | 150 | n/a | 4.39 GB | 5.00 GB | ok |
| `rapidata-t2v-veo3` | Rapidata/text-2-video-human-preferences-veo3 | per_file | video | apache-2.0 | yes | 250 | 1184 / 1184 | 695.94 MB | 1.00 GB | ok |
| `rapidata-t2v-veo3.1` | Rapidata/text-2-video-human-preferences-veo3.1 | per_file | video | apache-2.0 | yes | 50 | 79 / 79 | 315.18 MB | 500.00 MB | ok |
| `rapidata-t2v-marey` | Rapidata/text-2-video-human-preferences-moonvalley-marey | per_file | video | apache-2.0 | yes | 110 | 1413 / 1413 | 437.65 MB | 700.00 MB | ok |
| `rapidata-i2v-seedance` | Rapidata/image-to-video-human-preference-seedance-1-pro | per_file | video | apache-2.0 | yes | 40 | 297 / 297 | 336.79 MB | 500.00 MB | ok |
| `34data-veo3` | 34data/gen-videos-veo3 | zip_range | video | none declared (U) | no | 50 | 1 / 1 | 145.00 MB | 500.00 MB | ok |
| `34data-sora2` | 34data/gen-videos-sora2 | zip_range | video | none declared (U) | no | 50 | 1 / 1 | 335.00 MB | 500.00 MB | ok |
| `34data-kling` | 34data/gen-videos-kling | zip_range | video | none declared (U) | no | 50 | 1 / 1 | 250.00 MB | 500.00 MB | ok |
| `34data-wan2` | 34data/gen-videos-wan2 | zip_range | video | none declared (U) | no | 40 | 1 / 1 | 200.00 MB | 500.00 MB | ok |
| `34data-hunyuanvideo` | 34data/gen-videos-hunyuanvideo | zip_range | video | none declared (U) | no | 30 | 2 / 2 | 150.00 MB | 400.00 MB | ok |
| `34data-fastwan2` | 34data/gen-videos-fastwan2 | zip_range | video | none declared (U) | no | 30 | 2 / 2 | 150.00 MB | 400.00 MB | ok |
| `genvidbench-6.7m` | jian-0/GenVidBench | zip_range | video | none declared (U) | no | 100 | 2 / 2 | 500.00 MB | 1.00 GB | ok |
| `aigc-lipsync-benchmark` | ZiqiaoPeng/AIGC_LipSync_Benchmark | per_file | video | none declared (U) | no | 100 | 615 / 615 | 680.80 MB | 1.00 GB | ok |
| `talkingheadbench` | luchaoqi/TalkingHeadBench | per_file | video | mit | yes | 290 | 2994 / 2994 | 538.88 MB | 2.00 GB | ok |
| `deepaction-v1` | faridlab/deepaction_v1 | per_file | video | none declared (U) | no | 250 | 2605 / 2605 | 554.56 MB | 2.50 GB | ok |
| `faceforensics-c23` | bitmind/FaceForensicsC23 | excluded_bulk | video | FF++ terms (access form) | no | 0 | n/a | n/a | 1.00 GB | skipped: excluded_bulk: excluded by D-05 (re-upload bypassing the FF++ access form); ML's 200 compressed face-swap clips are not fetched |
| `genbuster-200k` | l8cv/GenBuster-200K | excluded_bulk | video | mit | yes | 0 | n/a | n/a | 1.00 GB | skipped: excluded_bulk: 29 × 4.3 GB multi-part 7z (113.66 GB) without member-level random access (ML F1, D-05) |
| `asvspoof5-test` | SpeechAntiSpoofingBenchmarks/ASVspoof5 | parquet_stream | audio | odc-by | yes | 400 | 2 / 200 | 48.05 MB | 500.00 MB | ok |
| `asvspoof2021-df` | SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF | parquet_stream | audio | other (ASVspoof 2021 terms; U) | no | 400 | 3 / 80 | 21.88 MB | 1.00 GB | ok |
| `in-the-wild-audio` | garystafford/deepfake-audio-detection | per_file | audio | cc-by-4.0 | yes | 400 | 1866 / 1866 | 120.63 MB | 500.00 MB | ok |
| `mlaad-tiny` | mueller91/MLAAD-tiny | per_file | audio | cc-by-nc-4.0 | no | 400 | 7900 / 7900 | 86.74 MB | 500.00 MB | ok |
| `synthbuster-zenodo` | zenodo/10066460 | excluded_bulk | image | cc-by-nc-sa-4.0 | no | 0 | n/a | n/a | 1.00 GB | skipped: excluded_bulk: single 12.37 GB zip on Zenodo; its content comes through marco-willi/synthbuster-plus (ML F1) |
| `co-spy-bench` | ruojiruoli/Co-Spy-Bench | excluded_bulk | image | mit | yes | 0 | n/a | n/a | 1.00 GB | skipped: excluded_bulk: 110 non-seekable tar.gz files (SD 3 medium 7.6–8.7 GB each, above the object cap); SD 3 comes from synthbuster-plus and Rapidata |
| `communityforensics-full` | OwensLab/CommunityForensics | excluded_bulk | image | cc-by-nc-sa-4.0 | no | 0 | n/a | n/a | 1.00 GB | skipped: excluded_bulk: 1.1 TB training set (ML F1); only the Eval set is sampled |
| `gpt-image-edit-1.5m` | UCSC-VLAA/GPT-Image-Edit-1.5M | excluded_bulk | image | cc-by-4.0 | yes | 0 | n/a | n/a | 1.00 GB | skipped: excluded_bulk: multi-part tar archives of about 20 GB per part (4.1 TB, V); partial edits come from So-Fake-OOD, OpenSDI and Semi-Truths |

## Exclusions (D-05)

- `bitmind/FaceForensicsC23`: re-upload that bypasses the FaceForensics++ access form (D-05)
- `l8cv/GenBuster-200K`: bulk only (29 × 4.3 GB multi-part 7z, no member-level random access); GenBuster-200K-mini is allowed (D-05)
- `faridlab/deepspeak_v2`: manual gating; the loop does not accept dataset terms (D-03)
- `nuriachandra/Deepfake-Eval-2024`: manual gating (D-03)
- `ControlNet/LAV-DF`: click-through gating (D-03)
- `ControlNet/AV-Deepfake1M-PlusPlus`: manual gating (D-03)

## Sources

### `commons-r1-camera`

- **Dataset:** `wikimedia/commons` @ `n/a`; mode `api`; role `core`
- **Licence:** per-file Commons licence; commercial_ok true
- **Caps:** 600 items, 7.00 GB; projected 5.86 GB (declared mean 10.00 MB × 600 items)
- **group_key:** `artist`, fallback `pageid`, namespace `wikimedia/commons`
- **Notes:** Category sizes 38,623 / 6,235 files (V). Candidates are the first 900 listed files per category, shuffled; files without a camera Make EXIF are dropped. Wikimedia throttled 1.1 s pacing during the probe, so requests are 5 s apart.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R1 [camera=canon-eos-r5] | 300 | n/a |
| real · real · R1 [camera=sony-ilce-7m4] | 300 | n/a |

### `synthbuster-plus-test`

- **Dataset:** `marco-willi/synthbuster-plus` @ `dbfb72f1ee96e953ee5cff80c58832fe89e1d2b5`; mode `parquet_stream`; role `core`
- **Licence:** none declared (RAISE is research-use; U); commercial_ok false
- **Caps:** 895 items, 3.50 GB; projected 1.14 GB (declared mean 1.30 MB × 895 items)
- **group_key:** `image_id`, namespace `marco-willi/synthbuster-plus`
- **Selected files:** `data/test-00000-of-00009.parquet` (797.30 MB), `data/test-00001-of-00009.parquet` (474.49 MB), `data/test-00002-of-00009.parquet` (267.81 MB), `data/test-00003-of-00009.parquet` (241.09 MB), `data/test-00004-of-00009.parquet` (371.59 MB), `data/test-00005-of-00009.parquet` (471.90 MB) and 3 more
- **Notes:** Test split 2,800 rows, 9 shards (3.7 GB); 14 sources × 200 rows; image_id repeats across the 14 sources, so it is the group key (V).

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R1 [generator=raise1k] | 200 | n/a |
| fake · t2i · flux1 [generator=FLUX.1-dev] | 100 | n/a |
| fake · t2i · flux1 [generator=FLUX.1-schnell] | 100 | n/a |
| fake · t2i · imagen3 [generator=imagen3] | 75 | n/a |
| fake · t2i · sd3 [generator=SD3-medium] | 150 | n/a |
| fake · t2i · legacy [generator=stable-diffusion-xl] | 30 | n/a |
| fake · t2i · legacy [generator=stable-diffusion-1-3] | 30 | n/a |
| fake · t2i · legacy [generator=stable-diffusion-1-4] | 30 | n/a |
| fake · t2i · legacy [generator=stable-diffusion-2] | 30 | n/a |
| fake · t2i · legacy [generator=dalle2] | 30 | n/a |
| fake · t2i · legacy [generator=dalle3] | 30 | n/a |
| fake · t2i · legacy [generator=firefly] | 30 | n/a |
| fake · t2i · legacy [generator=midjourney-v5] | 30 | n/a |
| fake · t2i · legacy [generator=glide] | 30 | n/a |

### `commons-r2-smartphone`

- **Dataset:** `wikimedia/commons` @ `n/a`; mode `api`; role `core`
- **Licence:** per-file Commons licence; commercial_ok true
- **Caps:** 800 items, 5.00 GB; projected 3.91 GB (declared mean 5.00 MB × 800 items)
- **group_key:** `artist`, fallback `pageid`, namespace `wikimedia/commons`
- **Notes:** ≥ 150 per brand (Samsung 250, Google 250, Apple 300). The Apple categories are named "Taken with iPhone 15 Pro" (763), "… 15 Pro Max" (1,336) and "… 16 Pro" (1,712) (V), so they replace ML's random + Make=Apple filter.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R2 [brand=samsung] | 250 | n/a |
| real · real · R2 [brand=google-pixel9pro] | 150 | n/a |
| real · real · R2 [brand=google-pixel8pro] | 100 | n/a |
| real · real · R2 [brand=apple-15pro] | 100 | n/a |
| real · real · R2 [brand=apple-15promax] | 100 | n/a |
| real · real · R2 [brand=apple-16pro] | 100 | n/a |

### `sofake-ood-test`

- **Dataset:** `saberzl/So-Fake-OOD` @ `b4b34d16fda2020c2923a21d431d1ab3acbdbaff`; mode `parquet_stream`; role `core`
- **Licence:** cc-by-nc-4.0; commercial_ok false
- **Caps:** 2200 items, 6.00 GB; projected 3.01 GB (declared mean 1.40 MB × 2200 items)
- **group_key:** `post_id`, fallback `filename`, namespace `saberzl/So-Fake-OOD`
- **Selected files:** `data/test_image-00008-of-00046.parquet` (2.76 GB), `data/test_image-00012-of-00046.parquet` (2.67 GB), `data/test_image-00018-of-00046.parquet` (2.72 GB), `data/test_image-00025-of-00046.parquet` (2.77 GB), `data/test_image-00029-of-00046.parquet` (2.74 GB), `data/test_image-00034-of-00046.parquet` (2.77 GB) and 2 more
- **Notes:** 46 shards × about 2.7 GB (V); 8 shards sampled (ML F3 says 6) because Bluesky reals are 1,449 of 91,370 rows. post_id was empty in the probe sample, so the group key falls back to the file name. X and Instagram reals are link-only and never match. Masks of TAMPERED rows are not stored yet.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R3 [label=REAL, platform=Reddit] | 500 | n/a |
| real · real · R3 [label=REAL, platform=Tumblr] | 200 | n/a |
| real · real · R3 [label=REAL, platform=Bluesky] | 200 | n/a |
| fake · t2i · F2 · flux2 [generator=FLUX_2, label=FULL_SYNTHETIC] | 60 | n/a |
| fake · t2i · flux1 [generator=Flux.1_pro, label=FULL_SYNTHETIC] | 100 | n/a |
| fake · t2i · F3 · gpt-image-1.5-2 [generator=GPT-image-1.5, label=FULL_SYNTHETIC] | 60 | n/a |
| fake · t2i · F3 · gpt-image-1.5-2 [generator=GPT-image-2, label=FULL_SYNTHETIC] | 60 | n/a |
| fake · t2i · gpt-4o-image-1 [generator=GPT4o, label=FULL_SYNTHETIC] | 100 | n/a |
| fake · t2i · F4 · nano-banana [generator=nano_banana, label=FULL_SYNTHETIC] | 60 | n/a |
| fake · t2i · F4 · nano-banana [generator=nano_banana_2, label=FULL_SYNTHETIC] | 60 | n/a |
| fake · t2i · F5 · seedream4 [generator=seedream4.5, label=FULL_SYNTHETIC] | 70 | n/a |
| fake · t2i · seedream3 [generator=Seedream3.0, label=FULL_SYNTHETIC] | 75 | n/a |
| fake · t2i · F7 · imagen4 [generator=Imagen4, label=FULL_SYNTHETIC] | 100 | n/a |
| fake · t2i · imagen3 [generator=imagen3, label=FULL_SYNTHETIC] | 75 | n/a |
| fake · t2i · hidream [generator=Hidream, label=FULL_SYNTHETIC] | 100 | n/a |
| fake · t2i · ideogram [generator=Ideogram3, label=FULL_SYNTHETIC] | 60 | n/a |
| fake · t2i · ideogram [generator=Ideogram2, label=FULL_SYNTHETIC] | 40 | n/a |
| fake · t2i · recraft [generator=Recraftv3, label=FULL_SYNTHETIC] | 80 | n/a |
| fake · edit · partial-edits [editor=openai, label=TAMPERED] | 150 | n/a |
| fake · edit · partial-edits [editor=ideogram, label=TAMPERED] | 50 | n/a |

### `coco-val`

- **Dataset:** `detection-datasets/coco` @ `cf0b22332314a937e9dc8a1957b21725430bb41d`; mode `parquet_stream`; role `core`
- **Licence:** none declared (COCO image licences vary; U); commercial_ok false
- **Caps:** 300 items, 1.00 GB; projected 45.41 MB (declared mean 155.00 KB × 300 items)
- **group_key:** `image_id`, namespace `detection-datasets/coco`
- **Selected files:** `data/val-00000-of-00002-c4f2e391ee4aba11.parquet` (385.00 MB), `data/val-00001-of-00002-7af5414a3b178949.parquet` (385.03 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R4 | 300 | n/a |

### `flickr30k`

- **Dataset:** `nlphuji/flickr30k` @ `2b239befc81b6e3f035ce6bd52f5f4d60f5625f7`; mode `zip_range`; role `core`
- **Licence:** none declared (Flickr image licences vary; U); commercial_ok false
- **Caps:** 200 items, 1.00 GB; projected 27.34 MB (declared mean 140.00 KB × 200 items)
- **group_key:** `img`, namespace `nlphuji/flickr30k`
- **Selected files:** `flickr30k-images.zip` (4.09 GB)
- **Notes:** The repo ships one 4.39 GB zip plus a CSV; the train/val/test split lives only in the CSV, so members are sampled from the whole zip (all real Flickr photographs).

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R4 | 200 | n/a |

### `openfake-core-test`

- **Dataset:** `ComplexDataLab/OpenFake` @ `3fd1109dc3258874243fa31c5bda9ee24260163b`; mode `parquet_stream`; role `core`
- **Licence:** cc-by-nc-4.0; commercial_ok false
- **Caps:** 945 items, 6.00 GB; projected 636.77 MB (declared mean 690.00 KB × 945 items)
- **group_key:** `prompt`, namespace `ComplexDataLab/OpenFake`
- **Selected files:** `core/test-00001-of-00013.parquet` (4.84 GB), `core/test-00002-of-00013.parquet` (4.78 GB), `core/test-00004-of-00013.parquet` (4.86 GB), `core/test-00006-of-00013.parquet` (4.89 GB)
- **Notes:** core/test: 13 shards × 4.7–5.0 GB (V); generator strings confirmed on 7,031 partial-statistics rows, so rare ones (nano-banana-pro 19, ideogram-2.0 19, halfmoon 14 per 7,031) may fall short. sd-3.5 was not seen in the data (U), so SD 3.5 is not requested here.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R4 [generator=imagenet, label=real] | 200 | n/a |
| real · real · R4 [generator=docci, label=real] | 200 | n/a |
| fake · t2i · F1 · midjourney [generator=midjourney-7, label=fake] | 100 | n/a |
| fake · t2i · F2 · flux2 [generator=flux.2-klein-9b, label=fake] | 60 | n/a |
| fake · t2i · F3 · gpt-image-1.5-2 [generator=gpt-image-1.5, label=fake] | 40 | n/a |
| fake · t2i · F3 · gpt-image-1.5-2 [generator=gpt-image-2, label=fake] | 20 | n/a |
| fake · t2i · F4 · nano-banana [generator=nano-banana-pro, label=fake] | 20 | n/a |
| fake · t2i · F5 · seedream4 [generator=seedream-v5.0, label=fake] | 30 | n/a |
| fake · t2i · ideogram [generator=ideogram-2.0, label=fake] | 20 | n/a |
| fake · t2i · recraft [generator=recraft-v3, label=fake] | 40 | n/a |
| fake · t2i · long-tail [generator=z-image-turbo, label=fake] | 50 | n/a |
| fake · t2i · long-tail [generator=illustrious, label=fake] | 50 | n/a |
| fake · t2i · long-tail [generator=ernie-image-turbo, label=fake] | 25 | n/a |
| fake · t2i · long-tail [generator=ernie-image, label=fake] | 25 | n/a |
| fake · t2i · long-tail [generator=lumina-17-2-25, label=fake] | 20 | n/a |
| fake · t2i · long-tail [generator=aurora-20-1-25, label=fake] | 15 | n/a |
| fake · t2i · long-tail [generator=frames-23-1-25, label=fake] | 15 | n/a |
| fake · t2i · long-tail [generator=halfmoon-4-4-25, label=fake] | 15 | n/a |

### `ffhq-256`

- **Dataset:** `merkol/ffhq-256` @ `f23c0e21f6fe222b4429272a86be00e4583b0298`; mode `parquet_stream`; role `core`
- **Licence:** none declared (FFHQ is CC BY-NC-SA 4.0; U); commercial_ok false
- **Caps:** 300 items, 1.00 GB; projected 29.30 MB (declared mean 100.00 KB × 300 items)
- **group_key:** `row`, namespace `merkol/ffhq-256`
- **Selected files:** `data/train-00000-of-00015-991c06ed507bdb85.parquet` (469.68 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R5 | 300 | n/a |

### `deepfakeface`

- **Dataset:** `OpenRL/DeepFakeFace` @ `eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c`; mode `zip_range`; role `core`
- **Licence:** openrail; commercial_ok false
- **Caps:** 470 items, 1.50 GB; projected 18.36 MB (declared mean 40.00 KB × 470 items)
- **group_key:** `person`, namespace `OpenRL/DeepFakeFace`
- **Selected files:** `insight.zip` (1.01 GB), `wiki.zip` (1.65 GB)
- **Notes:** insight.zip: 30,101 entries, `insight/<dir>/<person>_<dob>_<year>.jpg` (V, central directory read by range). The wiki.zip layout is assumed to match (U). The IMDB-WIKI person id groups swaps with their source identities (ML P1). DFF's "openrail" card is not a clear commercial grant, so commercial_ok is false.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R5 [archive=wiki] | 360 | n/a |
| fake · faceswap · faceswap-stills · insightface-swap [archive=insight] | 110 | n/a |

### `commons-r6-hard-negatives`

- **Dataset:** `wikimedia/commons` @ `n/a`; mode `api`; role `core`
- **Licence:** per-file Commons licence; commercial_ok true
- **Caps:** 500 items, 3.00 GB; projected 2.44 GB (declared mean 5.00 MB × 500 items)
- **group_key:** `artist`, fallback `pageid`, namespace `wikimedia/commons`
- **Notes:** Category names and file counts found by the probe (HDR 5,764 / 1,422; night 1,446; light trails 296; long exposure 224; selfies 1,259; depth of field 495; film 53 / 49; Civil War glass negatives 9,208). No "portrait mode", "beauty filter" or "Night Sight" category exists. Only JPEGs count, so the small film categories may fall short.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · R6 [kind=hdr-tonemapped] | 100 | n/a |
| real · real · R6 [kind=hdr] | 50 | n/a |
| real · real · R6 [kind=night] | 100 | n/a |
| real · real · R6 [kind=light-trails] | 25 | n/a |
| real · real · R6 [kind=long-exposure] | 25 | n/a |
| real · real · R6 [kind=selfie] | 70 | n/a |
| real · real · R6 [kind=depth-of-field] | 30 | n/a |
| real · real · R6 [kind=film] | 40 | n/a |
| real · real · R6 [kind=expired-film] | 30 | n/a |
| real · real · R6 [kind=glass-negative] | 30 | n/a |

### `midjourney-v6-recap`

- **Dataset:** `Photoroom/midjourney-v6-recap` @ `21c628db81401da88c5b33507230528cf3fe4a12`; mode `parquet_stream`; role `core`
- **Licence:** mit (Midjourney terms of service U); commercial_ok true
- **Caps:** 250 items, 1.00 GB; projected 35.40 MB (declared mean 145.00 KB × 250 items)
- **group_key:** `prompt`, namespace `Photoroom/midjourney-v6-recap`
- **Selected files:** `train_034.parquet` (1.45 GB), `train_092.parquet` (1.44 GB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F1 · midjourney | 250 | n/a |

### `aigibench-test`

- **Dataset:** `HorizonTEL/AIGIBench` @ `e44ec40efe5117a5ccdaa6ff0e89ed934d03d310`; mode `zip_range`; role `core`
- **Licence:** cc-by-nc-sa-4.0; commercial_ok false
- **Caps:** 1450 items, 4.00 GB; projected 1.42 GB (declared mean 1.00 MB × 1450 items)
- **group_key:** `path`, namespace `HorizonTEL/AIGIBench`
- **Selected files:** `test/BlendFace.zip` (4.19 GB), `test/CommunityAI.zip` (5.75 GB), `test/E4S.zip` (3.80 GB), `test/FaceSwap.zip` (2.14 GB), `test/IP-Adapter.zip` (7.85 GB), `test/InSwap.zip` (2.12 GB) and 10 more
- **Notes:** 25 test zips, 0.8–23.7 GB (V); members are `<Name>/0_real/*` and `<Name>/1_fake/*` (V on WFIR and SDXL central directories). ≤ 100 items per zip (ML F3). max_items is 1,450, not ML's 1,400, because this source also carries 240 R5 reals. Candidates per archive are capped at 600 after shuffling.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · faceswap · faceswap-stills [cls=1_fake, generator=BlendFace] | 70 | n/a |
| fake · faceswap · faceswap-stills [cls=1_fake, generator=E4S] | 70 | n/a |
| fake · faceswap · faceswap-stills [cls=1_fake, generator=FaceSwap] | 70 | n/a |
| fake · faceswap · faceswap-stills [cls=1_fake, generator=InSwap] | 70 | n/a |
| fake · faceswap · faceswap-stills [cls=1_fake, generator=SimSwap] | 70 | n/a |
| fake · faceswap · faceswap-stills [cls=1_fake, generator=StyleSwim] | 70 | n/a |
| fake · faceswap · faceswap-stills [cls=1_fake, generator=WFIR] | 70 | n/a |
| real · real · R5 [cls=0_real, generator=BlendFace\|E4S\|FaceSwap\|InSwap\|SimSwap\|StyleSwim\|WFIR] | 210 | n/a |
| fake · gan_face · gan-faces [cls=1_fake, generator=StyleGAN3] | 90 | n/a |
| fake · gan_face · gan-faces [cls=1_fake, generator=StyleGAN-XL] | 90 | n/a |
| fake · gan_face · gan-faces [cls=1_fake, generator=R3GAN] | 90 | n/a |
| real · real · R5 [cls=0_real, generator=StyleGAN3\|StyleGAN-XL\|R3GAN] | 30 | n/a |
| fake · personalised · personalised [cls=1_fake, generator=InstantID] | 75 | n/a |
| fake · personalised · personalised [cls=1_fake, generator=PhotoMaker] | 75 | n/a |
| fake · personalised · personalised [cls=1_fake, generator=Infinite-ID] | 75 | n/a |
| fake · personalised · personalised [cls=1_fake, generator=IP-Adapter] | 75 | n/a |
| fake · t2i · F1 · midjourney [cls=1_fake, generator=Midjourney] | 50 | n/a |
| fake · t2i · wild [cls=1_fake, generator=CommunityAI] | 100 | n/a |

### `rapidata-flux2-pro`

- **Dataset:** `Rapidata/Flux-2-pro_t2i_human_preference` @ `ef2e83f43b41897a8e1a40dd8a5dac209d164c79`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 60 items, 600.00 MB; projected 18.87 MB (declared mean 322.00 KB × 60 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00011-of-00069.parquet` (336.98 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F2 · flux2 [generator=flux-2-pro-25-11-25] | 60 | n/a |

### `rapidata-bananamark`

- **Dataset:** `Rapidata/bananamark-dataset` @ `cc951c4448660075b6d0491a1bf551799f4887bd`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 205 items, 1.00 GB; projected 88.89 MB (declared mean 444.00 KB × 205 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00000-of-00004.parquet` (375.66 MB), `data/train-00001-of-00004.parquet` (416.49 MB), `data/train-00002-of-00004.parquet` (279.42 MB), `data/train-00003-of-00004.parquet` (756.96 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F2 · flux2 [generator=black-forest-labs/flux-2-flex\|black-forest-labs/flux-2-pro\|black-forest-labs/flux-2-dev] | 60 | n/a |
| fake · t2i · F5 · seedream4 [generator=bytedance/seedream-4] | 50 | n/a |
| fake · t2i · hunyuanimage [generator=tencent/hunyuan-image-3] | 50 | n/a |
| fake · t2i · ideogram [generator=ideogram-ai/ideogram-v3-quality] | 20 | n/a |
| fake · t2i · long-tail [generator=prunaai/wan-2.2-image] | 25 | n/a |

### `qwen-image-bench`

- **Dataset:** `Qwen/Qwen-Image-Bench` @ `d2493deb153b020cf169c7e3f57d15e4dd697038`; mode `per_file`; role `core`
- **Licence:** apache-2.0 (generator terms of service U); commercial_ok true
- **Caps:** 960 items, 2.00 GB; projected 947.42 MB (18000 parsed files, mean 1010.59 KB × 960 items)
- **group_key:** `pid`, namespace `Qwen/Qwen-Image-Bench`
- **Selected files:** `images/FLUX.2-pro/000001_9abeb34a.png` (1.09 MB), `images/FLUX.2-pro/000002_bd1f0d9c.png` (1.18 MB), `images/FLUX.2-pro/000003_084eb030.png` (1.12 MB), `images/FLUX.2-pro/000004_47a50d10.png` (1.26 MB), `images/FLUX.2-pro/000005_f83b4579.png` (1.25 MB), `images/FLUX.2-pro/000006_d7573ce9.png` (1.54 MB) and 17994 more
- **Notes:** 18 model folders × 1,000 images, 18.6 GB (V). One prompt is rendered by every model, so the 6-digit prompt id is the group key. Downloads are paced 0.7 s (HF API limit 500 requests per 5 min, probe).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F2 · flux2 [generator=FLUX.2-pro] | 30 | 1000 |
| fake · t2i · F2 · flux2 [generator=FLUX.2_max] | 30 | 1000 |
| fake · t2i · F3 · gpt-image-1.5-2 [generator=GPT-Image-1.5] | 30 | 1000 |
| fake · t2i · F3 · gpt-image-1.5-2 [generator=gpt-image-2] | 30 | 1000 |
| fake · t2i · gpt-4o-image-1 [generator=GPT-Image-1] | 100 | 1000 |
| fake · t2i · F4 · nano-banana [generator=nano-banana-pro] | 50 | 1000 |
| fake · t2i · F4 · nano-banana [generator=nano-banana-2.0] | 50 | 1000 |
| fake · t2i · F5 · seedream4 [generator=Seedream-4.0] | 50 | 1000 |
| fake · t2i · F5 · seedream4 [generator=Seedream-4.5] | 50 | 1000 |
| fake · t2i · F5 · seedream4 [generator=Seedream-5.0] | 50 | 1000 |
| fake · t2i · F6 · qwen-image [generator=Qwen-Image] | 80 | 1000 |
| fake · t2i · F6 · qwen-image [generator=Qwen-Image-2512] | 80 | 1000 |
| fake · t2i · F6 · qwen-image [generator=Qwen-Image-2.0-pro] | 80 | 1000 |
| fake · t2i · F7 · imagen4 [generator=Imagen-4.0] | 50 | 1000 |
| fake · t2i · F7 · imagen4 [generator=Imagen-4.0-Ultra] | 50 | 1000 |
| fake · t2i · hunyuanimage [generator=HunyuanImage-3.0] | 50 | 1000 |
| fake · t2i · long-tail [generator=GLM-Image] | 50 | 1000 |
| fake · t2i · long-tail [generator=kling_v2_1] | 50 | 1000 |

### `mllm-generated`

- **Dataset:** `zr-zhang/MLLM-Generated-Image-Detection-Dataset` @ `1498eead24292a9a2d134476b2c559193b68b9de`; mode `per_file`; role `core`
- **Licence:** apache-2.0 (generator terms of service U); commercial_ok true
- **Caps:** 120 items, 1.00 GB; projected 248.15 MB (1452 parsed files, mean 2.07 MB × 120 items)
- **group_key:** `path` (regex `images/Raw/([^/]+) Images/[^/]+/(\d+)\.`), namespace `zr-zhang/MLLM-Generated-Image-Detection-Dataset`
- **Selected files:** `images/Raw/Hybrid Images/GPT-Image2-fake/1.png` (1.37 MB), `images/Raw/Hybrid Images/GPT-Image2-fake/10.png` (1.48 MB), `images/Raw/Hybrid Images/GPT-Image2-fake/100.png` (1.07 MB), `images/Raw/Hybrid Images/GPT-Image2-fake/101.png` (1.03 MB), `images/Raw/Hybrid Images/GPT-Image2-fake/102.png` (1.68 MB), `images/Raw/Hybrid Images/GPT-Image2-fake/103.png` (1.65 MB) and 2172 more
- **Notes:** Fakes only: the real-image source is undocumented (ML). Raw variant only (Preprocessing holds resized copies). The same index appears under each class folder, so (subset, index) groups the matched triplets (U).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F3 · gpt-image-1.5-2 · gpt-image-2 [generator=GPT-Image2-fake] | 60 | 726 |
| fake · t2i · F4 · nano-banana · nano-banana-2 [generator=Nano-Banana2-fake] | 60 | 726 |

### `rapidata-openai-4o`

- **Dataset:** `Rapidata/OpenAI-4o_t2i_human_preference` @ `9fafb39b4bb3bac6e2fbabd13503fa1199fde400`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 100 items, 600.00 MB; projected 19.53 MB (declared mean 200.00 KB × 100 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00004-of-00022.parquet` (199.94 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · gpt-4o-image-1 [generator=4o-26-3-25] | 100 | n/a |

### `bitmind-nano-banana`

- **Dataset:** `bitmind/nano-banana` @ `9ea8da32a5be03f4946e6cb10c2d2f8e90f0a0a4`; mode `parquet_stream`; role `core`
- **Licence:** mit (Google terms of service U); commercial_ok true
- **Caps:** 100 items, 1.00 GB; projected 150.00 MB (declared mean 1.50 MB × 100 items)
- **group_key:** `row`, namespace `bitmind/nano-banana`
- **Selected files:** `data/train-00003-of-00031.parquet` (459.88 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F4 · nano-banana | 100 | n/a |

### `rapidata-seedream3`

- **Dataset:** `Rapidata/Seedream-3_t2i_human_preference` @ `a9cd84d2a93e82954fe69bd3097b53c48f51a486`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 75 items, 500.00 MB; projected 16.26 MB (declared mean 222.00 KB × 75 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00063-of-00101.parquet` (230.30 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · seedream3 [generator=seedream-3-24-7-25] | 75 | n/a |

### `rapidata-face-generation`

- **Dataset:** `Rapidata/Face_Generation_Benchmark` @ `c6ed7eab14620db5daa77aec0feda5992091f5b3`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 360 items, 1.00 GB; projected 114.96 MB (declared mean 327.00 KB × 360 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00000-of-00004.parquet` (333.68 MB), `data/train-00001-of-00004.parquet` (267.97 MB), `data/train-00002-of-00004.parquet` (396.94 MB), `data/train-00003-of-00004.parquet` (444.79 MB)
- **Notes:** 2,310 pairs, 12 generator strings (complete statistics, V); 4 shards, 1.5 GB.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F6 · qwen-image [generator=qwen/qwen-image] | 60 | n/a |
| fake · t2i · modern-t2i-faces [generator=google/imagen-4-fast] | 30 | n/a |
| fake · t2i · modern-t2i-faces [generator=ideogram-ai/ideogram-v3-balanced] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=bytedance/seedream-3] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=luma/photon-flash] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=prunaai/wan-2.2-image] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=recraft-ai/recraft-v3] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=black-forest-labs/flux-1.1-pro] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=minimax/image-01] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=black-forest-labs/flux-schnell] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=luma/photon] | 27 | n/a |
| fake · t2i · modern-t2i-faces [generator=bria/image-3.2] | 27 | n/a |

### `rapidata-imagen4-and-opponents`

- **Dataset:** `Rapidata/Imagen4_t2i_human_preference` @ `1e6215e1a0394ee61d6059dd9ef0ae29531abd83`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 360 items, 1.50 GB; projected 99.14 MB (declared mean 282.00 KB × 360 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00000-of-00045.parquet` (141.69 MB), `data/train-00006-of-00045.parquet` (163.36 MB), `data/train-00008-of-00045.parquet` (156.63 MB), `data/train-00012-of-00045.parquet` (124.41 MB), `data/train-00013-of-00045.parquet` (142.36 MB), `data/train-00014-of-00045.parquet` (142.75 MB) and 4 more
- **Notes:** model2 is Imagen 4 Ultra (exp-05-20); model1 is one of 15 opponents (V, partial statistics), which supply the FLUX.1.1-pro, SD 3 and long-tail buckets. 45 shards × 163 MB; 10 sampled (the cap is 1.5 GB because the source serves four families).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F7 · imagen4 [generator=imagen-4.0-ultra-generate-exp-05-20] | 50 | n/a |
| fake · t2i · flux1 [generator=flux-1.1-pro] | 100 | n/a |
| fake · t2i · sd3 [generator=stable-diffusion-3] | 50 | n/a |
| fake · t2i · long-tail [generator=janus-7b] | 40 | n/a |
| fake · t2i · long-tail [generator=lumina-17-2-25] | 30 | n/a |
| fake · t2i · long-tail [generator=aurora-20-1-25] | 30 | n/a |
| fake · t2i · long-tail [generator=frames-23-1-25] | 30 | n/a |
| fake · t2i · long-tail [generator=halfmoon-4-4-25] | 30 | n/a |

### `rapidata-imagen4-ultra`

- **Dataset:** `Rapidata/Imagen-4-ultra-24-7-25_t2i_human_preference` @ `0946337d152e63845a7afeb29efcd7b93791aa5e`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 50 items, 500.00 MB; projected 11.33 MB (declared mean 232.00 KB × 50 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00039-of-00130.parquet` (199.31 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · F7 · imagen4 [generator=imagen-4-ultra-24-7-25] | 50 | n/a |

### `rapidata-hidream`

- **Dataset:** `Rapidata/Hidream_t2i_human_preference` @ `49f0a03eaffcf4f2b6d9fabd24d0ef2435e7706a`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 100 items, 600.00 MB; projected 23.14 MB (declared mean 237.00 KB × 100 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00004-of-00023.parquet` (228.74 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · hidream [generator=hidream-I1-full] | 100 | n/a |

### `rapidata-ideogram-v2`

- **Dataset:** `Rapidata/Ideogram-V2_t2i_human_preference` @ `9d9bb0aa365e9fbc77e865731ec96655a10e0990`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 60 items, 500.00 MB; projected 15.59 MB (declared mean 266.00 KB × 60 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00023-of-00059.parquet` (118.39 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · ideogram · ideogram-v2 [generator=ideogram] | 60 | n/a |

### `rapidata-recraft-v3`

- **Dataset:** `Rapidata/Recraft-v3-24-7-25_t2i_human_preference` @ `d7d47aab05cf8fcb5427e278d8817b9dbd65eafb`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 80 items, 500.00 MB; projected 18.20 MB (declared mean 233.00 KB × 80 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00041-of-00122.parquet` (201.14 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · recraft [generator=recraft-v3-24-7-25] | 80 | n/a |

### `rapidata-hunyuanimage-2.1`

- **Dataset:** `Rapidata/HunyuanImage-2.1_t2i_human_preference` @ `c8a2675e752d9e4c0036f4be4e4d933e07f4c1c6`; mode `parquet_stream`; role `core`
- **Licence:** CDLA-Permissive-2.0; commercial_ok true
- **Caps:** 100 items, 600.00 MB; projected 38.38 MB (declared mean 393.00 KB × 100 items)
- **group_key:** `prompt`, namespace `rapidata-t2i-prompts`
- **Selected files:** `data/train-00000-of-00091.parquet` (366.14 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · hunyuanimage [generator=hunyuan-image-2.1] | 100 | n/a |

### `elsa-d3-validation`

- **Dataset:** `elsaEU/ELSA_D3` @ `d980a1702f9cfe0f5a367a83ccf377eb6d95a88f`; mode `parquet_stream`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 130 items, 1.00 GB; projected 33.01 MB (declared mean 260.00 KB × 130 items)
- **group_key:** `prompt`, namespace `elsaEU/ELSA_D3`
- **Selected files:** `data/validation-00002-of-00011-bce2527afe096c22.parquet` (461.87 MB), `data/validation-00008-of-00011-bae91b22d2a936a0.parquet` (471.14 MB)
- **Notes:** One caption rendered by 4 generators per row (DeepFloyd IF, SD 1.4, SD 2.1, SDXL in the first rows; distinct values U), so the caption is the group key.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · legacy | 130 | n/a |

### `jamiewithofs-test`

- **Dataset:** `JamieWithofs/Deepfake-and-real-images` @ `6a492449fba897ccf8460731d618dded5b5e9c4d`; mode `parquet_stream`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 130 items, 500.00 MB; projected 1.27 MB (declared mean 10.00 KB × 130 items)
- **group_key:** `row`, namespace `JamieWithofs/Deepfake-and-real-images`
- **Selected files:** `data/test-00000-of-00001.parquet` (110.30 MB)
- **Notes:** ClassLabel [Fake, Real], so Fake is 0 (V); 256 px; the generator is undocumented.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · gan_face · gan-faces [label=0] | 130 | n/a |

### `wilddeepfake-test`

- **Dataset:** `xingjunm/WildDeepfake` @ `f3835aaf281dd9f8d79b51c4e02f050d3f7af0b4`; mode `per_file`; role `core`
- **Licence:** apache-2.0; commercial_ok true
- **Caps:** 100 items, 1.00 GB; projected 200.00 MB (declared mean 2.00 MB × 100 items)
- **group_key:** `video`, namespace `xingjunm/WildDeepfake`
- **Selected files:** `deepfake_in_the_wild/fake_test/11.tar.gz` (78.77 MB), `deepfake_in_the_wild/fake_test/16.tar.gz` (20.90 MB), `deepfake_in_the_wild/fake_test/27.tar.gz` (86.17 MB), `deepfake_in_the_wild/fake_test/41.tar.gz` (31.46 MB), `deepfake_in_the_wild/fake_test/67.tar.gz` (47.56 MB), `deepfake_in_the_wild/fake_test/81.tar.gz` (69.13 MB)
- **Notes:** 115 fake_test archives, median 45 MB (V); tar.gz has no random access, so small archives are downloaded whole and at most 25 frames kept per archive. The archive (one video) is the group key.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · faceswap · faceswap-stills · in-the-wild-swap | 100 | n/a |

### `opensdi-flux`

- **Dataset:** `nebula/OpenSDI_test` @ `7e233eaf98fcfee4c74c788f0e34d06feb7ad0df`; mode `parquet_stream`; role `core`
- **Licence:** cc-by-sa-4.0; commercial_ok true
- **Caps:** 60 items, 300.00 MB; projected 9.96 MB (declared mean 170.00 KB × 60 items)
- **group_key:** `key` (regex `([^/]+)\.[A-Za-z]+$`), namespace `nebula/OpenSDI_test`
- **Selected files:** `data/flux-00002-of-00007.parquet` (438.29 MB)
- **Notes:** 20,000 rows per split, real 10,000 / fake 10,000, masks on 5,000 (V). ClassLabel [real, fake] → fake is 1.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · edit · partial-edits [label=1] | 60 | n/a |

### `opensdi-sd15`

- **Dataset:** `nebula/OpenSDI_test` @ `7e233eaf98fcfee4c74c788f0e34d06feb7ad0df`; mode `parquet_stream`; role `core`
- **Licence:** cc-by-sa-4.0; commercial_ok true
- **Caps:** 60 items, 300.00 MB; projected 9.96 MB (declared mean 170.00 KB × 60 items)
- **group_key:** `key` (regex `([^/]+)\.[A-Za-z]+$`), namespace `nebula/OpenSDI_test`
- **Selected files:** `data/sd15-00003-of-00019.parquet` (76.20 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · edit · partial-edits [label=1] | 60 | n/a |

### `opensdi-sd2`

- **Dataset:** `nebula/OpenSDI_test` @ `7e233eaf98fcfee4c74c788f0e34d06feb7ad0df`; mode `parquet_stream`; role `core`
- **Licence:** cc-by-sa-4.0; commercial_ok true
- **Caps:** 60 items, 300.00 MB; projected 9.96 MB (declared mean 170.00 KB × 60 items)
- **group_key:** `key` (regex `([^/]+)\.[A-Za-z]+$`), namespace `nebula/OpenSDI_test`
- **Selected files:** `data/sd2-00004-of-00007.parquet` (465.39 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · edit · partial-edits [label=1] | 60 | n/a |

### `opensdi-sd3`

- **Dataset:** `nebula/OpenSDI_test` @ `7e233eaf98fcfee4c74c788f0e34d06feb7ad0df`; mode `parquet_stream`; role `core`
- **Licence:** cc-by-sa-4.0; commercial_ok true
- **Caps:** 60 items, 300.00 MB; projected 9.96 MB (declared mean 170.00 KB × 60 items)
- **group_key:** `key` (regex `([^/]+)\.[A-Za-z]+$`), namespace `nebula/OpenSDI_test`
- **Selected files:** `data/sd3-00007-of-00010.parquet` (339.32 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · edit · partial-edits [label=1] | 60 | n/a |

### `opensdi-sdxl`

- **Dataset:** `nebula/OpenSDI_test` @ `7e233eaf98fcfee4c74c788f0e34d06feb7ad0df`; mode `parquet_stream`; role `core`
- **Licence:** cc-by-sa-4.0; commercial_ok true
- **Caps:** 60 items, 300.00 MB; projected 9.96 MB (declared mean 170.00 KB × 60 items)
- **group_key:** `key` (regex `([^/]+)\.[A-Za-z]+$`), namespace `nebula/OpenSDI_test`
- **Selected files:** `data/sdxl-00001-of-00009.parquet` (461.05 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · edit · partial-edits [label=1] | 60 | n/a |

### `semi-truths-p2p`

- **Dataset:** `semi-truths/Semi-Truths-Evalset` @ `610748cddab90c48f2b27b17d77d701812a40138`; mode `per_file`; role `core`
- **Licence:** cc (variant undeclared; U); commercial_ok false
- **Caps:** 100 items, 500.00 MB; projected 200.00 MB (declared mean 2.00 MB × 100 items)
- **group_key:** `name` (regex `instance_\d+_(.+?)_(?:ADE20K|CelebAHQ|CityScapes|HumanParsing|OpenImages|SUN_RGBD)`), fallback `path`, namespace `semi-truths/Semi-Truths-Evalset`
- **Selected files:** `p2p/CelebAHQ.tar.bz2` (84.25 MB), `p2p/HumanParsing.tar.bz2` (81.18 MB)
- **Notes:** 6 p2p archives, 596 MB in total (V); member naming is assumed to follow the inpainting keys (U), so the group key falls back to the member path. Labels are in metadata.csv, which is not read; p2p members are edited images.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · edit · partial-edits · prompt-to-prompt | 100 | n/a |

### `openfake-reddit-test`

- **Dataset:** `ComplexDataLab/OpenFake` @ `3fd1109dc3258874243fa31c5bda9ee24260163b`; mode `parquet_stream`; role `core`
- **Licence:** cc-by-nc-4.0; commercial_ok false
- **Caps:** 200 items, 1.50 GB; projected 123.05 MB (declared mean 630.00 KB × 200 items)
- **group_key:** `row`, namespace `ComplexDataLab/OpenFake`
- **Selected files:** `reddit/test-00000-of-00005.parquet` (4.52 GB), `reddit/test-00003-of-00005.parquet` (4.65 GB)
- **Notes:** reddit/test: 5 shards × 4.5–4.7 GB; partial statistics show fake 11,670 / real 2,826 and type video 10,926 / image 3,570 (V); only image rows are kept. Highest label-noise risk (ML P6): the agent audit decides admission (D-06).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · wild [label=fake, type=image] | 200 | n/a |

### `communityforensics-eval`

- **Dataset:** `OwensLab/CommunityForensics-Eval` @ `7d4a74a88d2cac93b513c0853bf92c260eaceea0`; mode `parquet_stream`; role `diagnostic`
- **Licence:** cc-by-nc-sa-4.0; commercial_ok false
- **Caps:** 600 items, 2.00 GB; projected 2.00 GB (declared mean 3.70 MB × 600 items)
- **group_key:** `image_name`, namespace `OwensLab/CommunityForensics-Eval`
- **Selected files:** `data/CompEval-00112-of-00413.parquet` (2.53 GB), `data/CompEval-00120-of-00413.parquet` (32.28 MB), `data/CompEval-00129-of-00413.parquet` (7.19 MB), `data/CompEval-00274-of-00413.parquet` (144.48 MB), `data/CompEval-00303-of-00413.parquet` (166.46 MB), `data/CompEval-00334-of-00413.parquet` (55.22 MB)
- **Notes:** Diagnostic and contamination check only (ML F3): in-distribution for the Community Forensics detector. 413 shards (median 69 MB); image_data is raw bytes (V); label 1 = fake per the card.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2i · communityforensics-eval [label=1] | 300 | n/a |
| real · real · CFE [label=0] | 300 | n/a |

### `commons-video-general`

- **Dataset:** `wikimedia/commons` @ `n/a`; mode `api`; role `core`
- **Licence:** per-file Commons licence; commercial_ok true
- **Caps:** 200 items, 5.00 GB; projected 3.91 GB (declared mean 20.00 MB × 200 items)
- **group_key:** `artist`, fallback `pageid`, namespace `wikimedia/commons`
- **Notes:** CirrusSearch yield and content (animations, screen captures) are U; the agent audit screens them (D-06).

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · V-general [topic=street] | 50 | n/a |
| real · real · V-general [topic=nature] | 50 | n/a |
| real · real · V-general [topic=people] | 50 | n/a |
| real · real · V-general [topic=city] | 50 | n/a |

### `openvid-1m`

- **Dataset:** `nkp37/OpenVid-1M` @ `d8a63bd22989c80b5734ec2bb989f4e1b61a5807`; mode `zip_range`; role `core`
- **Licence:** cc-by-4.0; commercial_ok true
- **Caps:** 200 items, 2.00 GB; projected 1.37 GB (declared mean 7.00 MB × 200 items)
- **group_key:** `row`, namespace `nkp37/OpenVid-1M`
- **Selected files:** `OpenVid_part30.zip` (42.38 GB), `OpenVid_part40.zip` (43.56 GB)
- **Notes:** 170 whole zips of 19–50 GB (V), read by range only; split `_partaa/_partab` pieces never match. Member layout U.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · V-general | 200 | n/a |

### `commons-video-speeches`

- **Dataset:** `wikimedia/commons` @ `n/a`; mode `api`; role `core`
- **Licence:** per-file Commons licence; commercial_ok true
- **Caps:** 150 items, 5.00 GB; projected 4.39 GB (declared mean 30.00 MB × 150 items)
- **group_key:** `artist`, fallback `pageid`, namespace `wikimedia/commons`
- **Notes:** Real talking heads: ML's 100 plus 50 of TalkingHeadBench's 150 reals, because that repo holds no real clips (only CelebV-HQ / FF++ references; V). Category sizes 120 / 129 / 68 / 52 files (V); formats U.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · V-talking-head [set=speeches] | 45 | n/a |
| real · real · V-talking-head [set=slovenia] | 45 | n/a |
| real · real · V-talking-head [set=us] | 30 | n/a |
| real · real · V-talking-head [set=white-house] | 30 | n/a |

### `rapidata-t2v-veo3`

- **Dataset:** `Rapidata/text-2-video-human-preferences-veo3` @ `22878d6d9c692e09a3183e81f0428cc4c03a3f98`; mode `per_file`; role `core`
- **Licence:** apache-2.0; commercial_ok true
- **Caps:** 250 items, 1.00 GB; projected 695.94 MB (1104 parsed files, mean 2.78 MB × 250 items)
- **group_key:** `prompt`, namespace `rapidata-t2v-prompts`
- **Selected files:** `videos/0000_alpha_357741999.mp4` (3.80 MB), `videos/0000_hunyuan_1724.mp4` (1.12 MB), `videos/0000_hunyuan_421.mp4` (1.31 MB), `videos/0000_pika2.2_0.mp4` (2.60 MB), `videos/0000_pika2.2_1.mp4` (2.36 MB), `videos/0000_pika_1657819738.mp4` (2.23 MB) and 1178 more
- **Notes:** 1,184 mp4 (3.5 GB) named `<prompt>_<generator>_<seed>.mp4` (V). The kling-v2.1-master, sora-2, luma-ray2 and runway-alpha repos hold no video files (V), so those generators come from this repo, the marey repo and 34data.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-Veo · veo [generator=veo3] | 50 | 80 |
| fake · t2v · V-Veo · veo [generator=veo2] | 20 | 76 |
| fake · t2v · V-Sora · sora [generator=sora] | 40 | 158 |
| fake · t2v · V-WanHunyuan · wan-hunyuan [generator=hunyuan] | 30 | 158 |
| fake · t2v · V-WanHunyuan · wan-hunyuan [generator=wan2.1] | 30 | 79 |
| fake · t2v · luma-ray [generator=ray2] | 30 | 158 |
| fake · t2v · runway [generator=alpha] | 30 | 79 |
| fake · t2v · pika [generator=pika2.2] | 20 | 158 |

### `rapidata-t2v-veo3.1`

- **Dataset:** `Rapidata/text-2-video-human-preferences-veo3.1` @ `5d9fb3e419ba3e5c8b98130abcde9a44978fa189`; mode `per_file`; role `core`
- **Licence:** apache-2.0; commercial_ok true
- **Caps:** 50 items, 500.00 MB; projected 315.18 MB (79 parsed files, mean 6.30 MB × 50 items)
- **group_key:** `prompt`, namespace `rapidata-t2v-prompts`
- **Selected files:** `videos/veo31_00_0.mp4` (8.71 MB), `videos/veo31_01_0.mp4` (2.38 MB), `videos/veo31_02_0.mp4` (5.93 MB), `videos/veo31_03_0.mp4` (1.54 MB), `videos/veo31_04_0.mp4` (3.35 MB), `videos/veo31_05_0.mp4` (6.49 MB) and 73 more

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-Veo · veo | 50 | 79 |

### `rapidata-t2v-marey`

- **Dataset:** `Rapidata/text-2-video-human-preferences-moonvalley-marey` @ `cdbf62f223abf5864d3d7cf9b6afeb3391fcc771`; mode `per_file`; role `core`
- **Licence:** apache-2.0; commercial_ok true
- **Caps:** 110 items, 700.00 MB; projected 437.65 MB (1413 parsed files, mean 3.98 MB × 110 items)
- **group_key:** `prompt`, namespace `rapidata-t2v-prompts`
- **Selected files:** `Videos/alpha_0000_357741999.mp4` (3.80 MB), `Videos/alpha_0001_219242721.mp4` (2.77 MB), `Videos/alpha_0002_3590243511.mp4` (4.52 MB), `Videos/alpha_0003_2649066252.mp4` (2.64 MB), `Videos/alpha_0004_293654896.mp4` (2.30 MB), `Videos/alpha_0005_1440689931.mp4` (3.31 MB) and 1407 more
- **Notes:** 1,413 mp4 (5.9 GB); generator tokens kling-v2.1-master-25-7-25 77, marey-11-8-25 78, mochi-1-24-7-25 79 (V).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-Kling · kling [generator=kling-v2.1-master-25-7-25] | 50 | 77 |
| fake · t2v · marey [generator=marey-11-8-25] | 40 | 78 |
| fake · t2v · mochi [generator=mochi-1-24-7-25] | 20 | 79 |

### `rapidata-i2v-seedance`

- **Dataset:** `Rapidata/image-to-video-human-preference-seedance-1-pro` @ `987399c7d471f15d556598f79108c4c4f8c0b74a`; mode `per_file`; role `core`
- **Licence:** apache-2.0; commercial_ok true
- **Caps:** 40 items, 500.00 MB; projected 336.79 MB (199 parsed files, mean 8.42 MB × 40 items)
- **group_key:** `prompt`, namespace `rapidata-i2v-prompts`
- **Selected files:** `Videos/hailuo-02/hailuo-02_camera-motion_0001_ui.mp4` (8.25 MB), `Videos/hailuo-02/hailuo-02_camera-motion_0004.mp4` (10.29 MB), `Videos/hailuo-02/hailuo-02_camera-motion_0009.mp4` (2.83 MB), `Videos/hailuo-02/hailuo-02_camera-motion_0011.mp4` (2.46 MB), `Videos/hailuo-02/hailuo-02_camera-motion_0014.mp4` (6.42 MB), `Videos/hailuo-02/hailuo-02_camera-motion_0015.mp4` (9.36 MB) and 291 more
- **Notes:** 297 mp4 (2.7 GB); seedance-1-pro 100 clips (V).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-Seedance · seedance [generator=seedance-1-pro] | 40 | 100 |

### `34data-veo3`

- **Dataset:** `34data/gen-videos-veo3` @ `dad55b648cf7da90584e69245b8eb22da0929ba4`; mode `zip_range`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 50 items, 500.00 MB; projected 145.00 MB (declared mean 2.90 MB × 50 items)
- **group_key:** `row`, namespace `34data/gen-videos-veo3`
- **Selected files:** `data_001.zip` (861.27 MB)
- **Notes:** data_001.zip 903 MB, 314 flat `<sha1>.mp4` members (V).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-Veo · veo | 50 | n/a |

### `34data-sora2`

- **Dataset:** `34data/gen-videos-sora2` @ `82a7b46ee58f417a233b7e89e705b5065039df3a`; mode `zip_range`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 50 items, 500.00 MB; projected 335.00 MB (declared mean 6.70 MB × 50 items)
- **group_key:** `row`, namespace `34data/gen-videos-sora2`
- **Selected files:** `data_001.zip` (1.89 GB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-Sora · sora | 50 | n/a |

### `34data-kling`

- **Dataset:** `34data/gen-videos-kling` @ `07d2e8ff55f94b745749bd94e9af80b6546d272a`; mode `zip_range`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 50 items, 500.00 MB; projected 250.00 MB (declared mean 5.00 MB × 50 items)
- **group_key:** `row`, namespace `34data/gen-videos-kling`
- **Selected files:** `data_001.zip` (4.43 GB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-Kling · kling | 50 | n/a |

### `34data-wan2`

- **Dataset:** `34data/gen-videos-wan2` @ `660cd80d16ac7cc962d892461e6b1e506809a3c7`; mode `zip_range`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 40 items, 500.00 MB; projected 200.00 MB (declared mean 5.00 MB × 40 items)
- **group_key:** `row`, namespace `34data/gen-videos-wan2`
- **Selected files:** `data_001.zip` (1.68 GB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-WanHunyuan · wan-hunyuan | 40 | n/a |

### `34data-hunyuanvideo`

- **Dataset:** `34data/gen-videos-hunyuanvideo` @ `063249f0fab03017fcbe012997afab36d53886ef`; mode `zip_range`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 30 items, 400.00 MB; projected 150.00 MB (declared mean 5.00 MB × 30 items)
- **group_key:** `row`, namespace `34data/gen-videos-hunyuanvideo`
- **Selected files:** `data_001.zip` (642.38 MB), `data_002.zip` (334.12 MB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-WanHunyuan · wan-hunyuan | 30 | n/a |

### `34data-fastwan2`

- **Dataset:** `34data/gen-videos-fastwan2` @ `93f76b0892cfb2f61ef9585f4a37d775ee18c12b`; mode `zip_range`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 30 items, 400.00 MB; projected 150.00 MB (declared mean 5.00 MB × 30 items)
- **group_key:** `row`, namespace `34data/gen-videos-fastwan2`
- **Selected files:** `data_001.zip` (2.74 GB), `data_002.zip` (1.43 GB)

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-WanHunyuan · wan-hunyuan | 30 | n/a |

### `genvidbench-6.7m`

- **Dataset:** `jian-0/GenVidBench` @ `701cafb6f999d7ea0cbf3c354df6177311a4d824`; mode `zip_range`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 100 items, 1.00 GB; projected 500.00 MB (declared mean 5.00 MB × 100 items)
- **group_key:** `row`, namespace `jian-0/GenVidBench`
- **Selected files:** `GenVidBench/6.7m/OpenAI_Sora.zip` (467.69 MB), `GenVidBench/6.7m/keling.zip` (1.28 GB)
- **Notes:** Regression set of older clips. ML named the Pair1/Pair2 archives, but they are rar and multi-part 7z, and cogvideo_mora_svd_ms_fix.tar is 16.7 GB (above the 5 GB object cap), so the two zips under 6.7m/ are used instead (0.49 GB and 1.38 GB, V). Members are assumed to be generated clips (U; audit).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · V-Sora · sora · sora-2024 [archive=OpenAI_Sora] | 50 | n/a |
| fake · t2v · V-Kling · kling · kling-1.x [archive=keling] | 50 | n/a |

### `aigc-lipsync-benchmark`

- **Dataset:** `ZiqiaoPeng/AIGC_LipSync_Benchmark` @ `19f92a49ba3f29e34b8f1109f082555770da3fa0`; mode `per_file`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 100 items, 1.00 GB; projected 680.80 MB (615 parsed files, mean 6.81 MB × 100 items)
- **group_key:** `row`, namespace `ZiqiaoPeng/AIGC_LipSync_Benchmark`
- **Selected files:** `videos/clip_1.mp4` (10.06 MB), `videos/clip_10.mp4` (2.52 MB), `videos/clip_100.mp4` (854.39 KB), `videos/clip_101.mp4` (1.36 MB), `videos/clip_102.mp4` (1.70 MB), `videos/clip_103.mp4` (1.93 MB) and 609 more
- **Notes:** 615 clips (4.39 GB); the card attributes them to Kling, JiMeng, Wan and Hunyuan, with no per-clip mapping (V).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · ai-talking-humans | 100 | 615 |

### `talkingheadbench`

- **Dataset:** `luchaoqi/TalkingHeadBench` @ `cd5f4672a13c61efe0df6aa79c61bbc9602ff625`; mode `per_file`; role `core`
- **Licence:** mit; commercial_ok true
- **Caps:** 290 items, 2.00 GB; projected 538.88 MB (2994 parsed files, mean 1.86 MB × 290 items)
- **group_key:** `name` (regex `^([^/]+?)--`), fallback `path`, namespace `luchaoqi/TalkingHeadBench`
- **Selected files:** `fake/AniPortraitAudio/test/00683--xqtbz-pVp2g_1--AniPortraitAudio.mp4` (1.11 MB), `fake/AniPortraitAudio/test/01045--eSHy8q1MP00_2--AniPortraitAudio.mp4` (1.62 MB), `fake/AniPortraitAudio/test/01160--GLiEItdhO5A_1--AniPortraitAudio.mp4` (1.73 MB), `fake/AniPortraitAudio/test/03167--SOIdHcybQOY_1--AniPortraitAudio.mp4` (1.47 MB), `fake/AniPortraitAudio/test/03921--nRLP98lG1YA_0--AniPortraitAudio.mp4` (969.09 KB), `fake/AniPortraitAudio/test/04183--DF2ou1RxM6I_0--AniPortraitAudio.mp4` (1.49 MB) and 2988 more
- **Notes:** `fake/<gen>/<split>/<ffhq_img>--<driver>--<gen>.mp4`; Hallo3 has 10 files and MAGI-1 66 (V). The FFHQ identity image is the group key. No real clips exist in the repo, so ML's 150 reals are not available here.

| Bucket | Quota | Candidates |
|---|---|---|
| fake · talking_head · talking-head-animation [generator=LivePortrait] | 40 | 529 |
| fake · talking_head · talking-head-animation [generator=AniPortraitAudio] | 40 | 542 |
| fake · talking_head · talking-head-animation [generator=AniPortraitVideo] | 40 | 422 |
| fake · talking_head · talking-head-animation [generator=Hallo] | 40 | 420 |
| fake · talking_head · talking-head-animation [generator=Hallo2] | 40 | 432 |
| fake · talking_head · talking-head-animation [generator=EmoPortrait] | 40 | 573 |
| fake · talking_head · talking-head-animation [generator=MAGI-1] | 40 | 66 |
| fake · talking_head · talking-head-animation [generator=Hallo3] | 10 | 10 |

### `deepaction-v1`

- **Dataset:** `faridlab/deepaction_v1` @ `476932c3369424ca59dc1f8fb758cb3c13fe63fe`; mode `per_file`; role `core`
- **Licence:** none declared (U); commercial_ok false
- **Caps:** 250 items, 2.50 GB; projected 554.56 MB (2605 parsed files, mean 2.22 MB × 250 items)
- **group_key:** `prompt`, namespace `faridlab/deepaction_v1`
- **Selected files:** `BDAnimateDiffLightning/0/a.mp4` (587.20 KB), `BDAnimateDiffLightning/0/b.mp4` (481.01 KB), `BDAnimateDiffLightning/0/c.mp4` (477.23 KB), `BDAnimateDiffLightning/0/d.mp4` (478.04 KB), `BDAnimateDiffLightning/0/e.mp4` (427.37 KB), `BDAnimateDiffLightning/1/a.mp4` (161.31 KB) and 2599 more
- **Notes:** `<Generator>/<prompt_id>/<a..e>.mp4`, Pexels reals average 26 MB, so the cap is 2.5 GB (ML F3 says 2 GB) (V).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · t2v · animatediff [generator=BDAnimateDiffLightning] | 30 | 500 |
| fake · t2v · cogvideox [generator=CogVideoX5B] | 30 | 500 |
| fake · t2v · runway [generator=RunwayML] | 30 | 505 |
| fake · t2v · stable-video [generator=StableDiffusion] | 30 | 500 |
| fake · t2v · V-Veo · veo [generator=Veo] | 30 | 100 |
| fake · t2v · videopoet [generator=VideoPoet] | 30 | 400 |
| real · real · V-general [generator=Pexels] | 70 | 100 |

### `faceforensics-c23`

- **Dataset:** `bitmind/FaceForensicsC23` @ `n/a`; mode `excluded_bulk`; role `core`
- **Licence:** FF++ terms (access form); commercial_ok false
- **Caps:** 0 items, 1.00 GB; projected n/a ()
- **Status:** skipped: excluded_bulk: excluded by D-05 (re-upload bypassing the FF++ access form); ML's 200 compressed face-swap clips are not fetched
- **Notes:** excluded by D-05 (re-upload bypassing the FF++ access form); ML's 200 compressed face-swap clips are not fetched

### `genbuster-200k`

- **Dataset:** `l8cv/GenBuster-200K` @ `n/a`; mode `excluded_bulk`; role `core`
- **Licence:** mit; commercial_ok true
- **Caps:** 0 items, 1.00 GB; projected n/a ()
- **Status:** skipped: excluded_bulk: 29 × 4.3 GB multi-part 7z (113.66 GB) without member-level random access (ML F1, D-05)
- **Notes:** 29 × 4.3 GB multi-part 7z (113.66 GB) without member-level random access (ML F1, D-05)

### `asvspoof5-test`

- **Dataset:** `SpeechAntiSpoofingBenchmarks/ASVspoof5` @ `6aeb859f64c582dd1e54772e30483b0234819cfb`; mode `parquet_stream`; role `core`
- **Licence:** odc-by; commercial_ok true
- **Caps:** 400 items, 500.00 MB; projected 48.05 MB (declared mean 123.00 KB × 400 items)
- **group_key:** `notes` (regex `"speaker_id":\s*"([^"]+)"`), fallback `path`, namespace `SpeechAntiSpoofingBenchmarks/ASVspoof5`
- **Selected files:** `data/test-00101-of-00200.parquet` (401.57 MB), `data/test-00179-of-00200.parquet` (399.63 MB)
- **Notes:** 200 shards × about 420 MB (V); ClassLabel [bonafide, spoof]; notes is a JSON string holding speaker_id and attack_id (V; exact formatting U, so the group key falls back to the path).

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · A-bonafide [label=0] | 200 | n/a |
| fake · tts · asvspoof5 [label=1] | 200 | n/a |

### `asvspoof2021-df`

- **Dataset:** `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF` @ `89e8ad0bd2819f09b65aee839c58885b1414c0f2`; mode `parquet_stream`; role `core`
- **Licence:** other (ASVspoof 2021 terms; U); commercial_ok false
- **Caps:** 400 items, 1.00 GB; projected 21.88 MB (declared mean 56.00 KB × 400 items)
- **group_key:** `path`, namespace `SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF`
- **Selected files:** `data/test-00007-of-00080.parquet` (408.89 MB), `data/test-00038-of-00080.parquet` (408.68 MB), `data/test-00040-of-00080.parquet` (412.19 MB)
- **Notes:** 80 shards × about 430 MB (V); bona fide rows are a small share of DF eval, hence 3 shards.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · A-bonafide [label=0] | 200 | n/a |
| fake · tts · asvspoof2021-df [label=1] | 200 | n/a |

### `in-the-wild-audio`

- **Dataset:** `garystafford/deepfake-audio-detection` @ `fcf5344bb7f82b54b6b932291326d29750ef1e82`; mode `per_file`; role `core`
- **Licence:** cc-by-4.0; commercial_ok true
- **Caps:** 400 items, 500.00 MB; projected 120.63 MB (1866 parsed files, mean 308.81 KB × 400 items)
- **group_key:** `name` (regex `^(.+?)_part_\d+$`), fallback `path`, namespace `garystafford/deepfake-audio-detection`
- **Selected files:** `fake/el_0001_c_part_002.flac` (128.39 KB), `fake/el_0001_c_part_003.flac` (116.77 KB), `fake/el_0001_c_part_004.flac` (130.65 KB), `fake/el_0001_c_part_005.flac` (123.54 KB), `fake/el_0001_part_001.flac` (143.10 KB), `fake/el_0002_c_part_001.flac` (96.37 KB) and 1860 more
- **Notes:** 933 real / 933 fake flac files, 2.5–12.5 s (V); parts of one clip share a group.

| Bucket | Quota | Candidates |
|---|---|---|
| real · real · A-bonafide [cls=real] | 200 | 933 |
| fake · tts · in-the-wild-audio [cls=fake] | 200 | 933 |

### `mlaad-tiny`

- **Dataset:** `mueller91/MLAAD-tiny` @ `9143e5ea709575ebab6bec52840a1043aada7bb1`; mode `per_file`; role `core`
- **Licence:** cc-by-nc-4.0; commercial_ok false
- **Caps:** 400 items, 500.00 MB; projected 86.74 MB (7900 parsed files, mean 222.05 KB × 400 items)
- **group_key:** `path`, namespace `mueller91/MLAAD-tiny`
- **Selected files:** `fake/de/CaroTTS/Kanzlerin_03_17_f000019.wav` (186.56 KB), `fake/de/CaroTTS/Kanzlerin_07_14_f000040.wav` (206.52 KB), `fake/de/CaroTTS/Kanzlerin_08_13_f000031.wav` (289.24 KB), `fake/de/CaroTTS/Kanzlerin_08_17_f000019.wav` (106.38 KB), `fake/de/CaroTTS/Kanzlerin_12_15_f000021.wav` (136.86 KB), `fake/de/CaroTTS/Kanzlerin_17_16_f000049.wav` (154.27 KB) and 7894 more
- **Notes:** 7,900 TTS clips under `fake/<lang>/<model>/` (79 models × 100, V); paired with ASVspoof5 bona fide (ML §2.5).

| Bucket | Quota | Candidates |
|---|---|---|
| fake · tts · mlaad | 400 | 7900 |

### `synthbuster-zenodo`

- **Dataset:** `zenodo/10066460` @ `n/a`; mode `excluded_bulk`; role `core`
- **Licence:** cc-by-nc-sa-4.0; commercial_ok false
- **Caps:** 0 items, 1.00 GB; projected n/a ()
- **Status:** skipped: excluded_bulk: single 12.37 GB zip on Zenodo; its content comes through marco-willi/synthbuster-plus (ML F1)
- **Notes:** single 12.37 GB zip on Zenodo; its content comes through marco-willi/synthbuster-plus (ML F1)

### `co-spy-bench`

- **Dataset:** `ruojiruoli/Co-Spy-Bench` @ `n/a`; mode `excluded_bulk`; role `core`
- **Licence:** mit; commercial_ok true
- **Caps:** 0 items, 1.00 GB; projected n/a ()
- **Status:** skipped: excluded_bulk: 110 non-seekable tar.gz files (SD 3 medium 7.6–8.7 GB each, above the object cap); SD 3 comes from synthbuster-plus and Rapidata
- **Notes:** 110 non-seekable tar.gz files (SD 3 medium 7.6–8.7 GB each, above the object cap); SD 3 comes from synthbuster-plus and Rapidata

### `communityforensics-full`

- **Dataset:** `OwensLab/CommunityForensics` @ `n/a`; mode `excluded_bulk`; role `core`
- **Licence:** cc-by-nc-sa-4.0; commercial_ok false
- **Caps:** 0 items, 1.00 GB; projected n/a ()
- **Status:** skipped: excluded_bulk: 1.1 TB training set (ML F1); only the Eval set is sampled
- **Notes:** 1.1 TB training set (ML F1); only the Eval set is sampled

### `gpt-image-edit-1.5m`

- **Dataset:** `UCSC-VLAA/GPT-Image-Edit-1.5M` @ `n/a`; mode `excluded_bulk`; role `core`
- **Licence:** cc-by-4.0; commercial_ok true
- **Caps:** 0 items, 1.00 GB; projected n/a ()
- **Status:** skipped: excluded_bulk: multi-part tar archives of about 20 GB per part (4.1 TB, V); partial edits come from So-Fake-OOD, OpenSDI and Semi-Truths
- **Notes:** multi-part tar archives of about 20 GB per part (4.1 TB, V); partial edits come from So-Fake-OOD, OpenSDI and Semi-Truths

