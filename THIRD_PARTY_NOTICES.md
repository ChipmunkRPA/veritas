# Third-party notices

The original notices reference `backend/scripts/notices_doc.py` (`make notices-doc`) and
`backend/tests/test_notices_doc.py`; both are absent from this checkout. The 2026-10-31 local-inspection weight
additions and model counts below are synchronized directly with `backend/app/models.yaml`. Existing code and font
notices retain their recorded sources.

## What this page covers

- **Covered:** the code of others that this repository carries, the font files it carries, and the model weights the
  software downloads (`python -m app.cli models fetch`). No weights are in the repository.
- **Not covered:** the Python packages of `backend/requirements/*.lock` and the npm packages of
  `frontend/package-lock.json` other than the fonts. The repository does not carry them, but a container image and the
  built web application do, and neither comes with their licence texts. An image built with `BAKE_MODELS` also carries
  the weights below without their licence texts. Whoever distributes an image or a build adds them.
- **Veritas itself** has no licence file here: the choice of a licence for the repository is its owner's.

## Code in the repository

Under `backend/app/third_party`, each package with its unchanged licence file.

| Package | From | Version | Licence | Copyright | What is here, and what was changed |
|---|---|---|---|---|---|
| `cospy` | <https://github.com/Megum1/CO-SPY> | commit 310d048387edaf11f236ed1c7180d0e1e823250f | MIT ([text](backend/app/third_party/cospy/LICENSE)) | Copyright (c) 2025 Siyuan Cheng | `model.py`: adapted: the artefact CNN and the fusion arithmetic of the released `sd-v1_4` model; training, augmentation, remote constructors and permissive weight loading removed (`NOTICE.md`). A research candidate: no detector of the service uses it |
| `imwatermark` | <https://github.com/ShieldMnt/invisible-watermark> | 0.2.0 (the PyPI wheel) | MIT ([text](backend/app/third_party/imwatermark/LICENSE)) | Copyright (c) 2021 ShieldMnt | `max_dct.py`: `imwatermark/maxDct.py` unchanged, with a `ruff: noqa` line added at the top (`__init__.py`) |
| `latentsync` | <https://github.com/bytedance/LatentSync> | commit a229c3948406bc2cf6eaf4873e662e70c6a04746 | Apache-2.0 ([text](backend/app/third_party/latentsync/LICENSE)) | Copyright (c) 2024 Bytedance Ltd. and/or its affiliates | `stable_syncnet.py`: StableSyncNet and its attention class; diffusers' feed-forward block re-implemented with the same parameter names, gradient checkpointing, the attention-mask path, einops and the OmegaConf file removed (the file's docstring) |

## Fonts

The reports use the families in `backend/app/reports/fonts`. The web application also bundles Manrope and DM Sans;
their licence texts are in `frontend/public/licenses`. All are under the SIL Open Font License 1.1.

| Family | Files | Copyright | Licence text |
|---|---|---|---|
| Inter | 6 | Copyright 2016 The Inter Project Authors (https://github.com/rsms/inter) | [LICENSE-Inter.txt](backend/app/reports/fonts/LICENSE-Inter.txt) |
| JetBrainsMono | 2 | Copyright 2020 The JetBrains Mono Project Authors (https://github.com/JetBrains/JetBrainsMono) | [LICENSE-JetBrainsMono.txt](backend/app/reports/fonts/LICENSE-JetBrainsMono.txt) |
| Newsreader | 4 | Copyright 2020 The Newsreader Project Authors (http://github.com/productiontype/Newsreader) | [LICENSE-Newsreader.txt](backend/app/reports/fonts/LICENSE-Newsreader.txt) |

| npm package | Version | Licence (as `frontend/package-lock.json` records it) |
|---|---|---|
| `@fontsource-variable/inter` | 5.3.0 | OFL-1.1 |
| `@fontsource-variable/dm-sans` | 5.3.0 | OFL-1.1 ([text](frontend/public/licenses/DM-Sans.txt)) |
| `@fontsource-variable/jetbrains-mono` | 5.3.0 | OFL-1.1 |
| `@fontsource-variable/manrope` | 5.3.0 | OFL-1.1 ([text](frontend/public/licenses/Manrope.txt)) |
| `@fontsource-variable/newsreader` | 5.3.0 | OFL-1.1 |

## Model weights

The 36 models of `backend/app/models.yaml`, each pinned to a revision or to file hashes. The class decides
what may run. The default licence policy (`VERITAS_LICENCE_POLICY`) is the `commercial` preset: a detector runs only
when every model it needs is of the class `commercial`. No unknown or non-commercial checkpoint is allowlisted by default. The local `loc.fused` opt-in enables an otherwise default-disabled detector whose weights and dependency are already classified commercial; it is not an exception for unknown weights.
The note of each row is the manifest's record of where the licence was read.

- `commercial` (30): the recorded weight licence allows commercial use; this does not grant permission for a separate source repository without a licence.
- `non_commercial` (1): the licence forbids commercial use.
- `unknown` (5): no licence was found at the pinned revision, or its terms are not a plain grant.

| Model | Weights from | Licence (SPDX) | Class | Note |
|---|---|---|---|---|
| `clip.vit_l14_openai` | Hugging Face `timm/vit_large_patch14_clip_224.openai` at `18d0535469bb561bf468d76c1d73aa35156c922b` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0 (timm); weights originate from openai/CLIP (MIT) |
| `univfd.fc_probe` | <https://github.com/WisconsinAIVision/UniversalFakeDetect/raw/763391eff3284f6950ffb323599c1a7a819f2ecd/pretrained_weights/fc_weights.pth> | MIT | `commercial` | github.com/WisconsinAIVision/UniversalFakeDetect (GitHub licence API: MIT) |
| `commfor.vit_s16_384` | Hugging Face `OwensLab/commfor-model-384` at `6076002bf0d9dd37537f965ee2f06f826c333b61` | MIT | `commercial` | HF model-card licence mit; github.com/JeongsooP/Community-Forensics LICENSE (MIT, Jeongsoo Park 2025; GitHub licence API: MIT) |
| `dda.dinov2_vitl14_lora` | Hugging Face `Junwei-Xi/Dual-Data-Alignment` at `4390d9023899196b437480bb6a441915ef5d816c` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0; github.com/roy-ch/Dual-Data-Alignment LICENSE (GitHub licence API: Apache-2.0) |
| `dinov2.vitl14` | torch.hub `facebookresearch/dinov2` at `b8931f7bf91576930313be2c6d6af376033b35f0` | Apache-2.0 | `commercial` | github.com/facebookresearch/dinov2 LICENSE (GitHub licence API: Apache-2.0) |
| `spai.vitb16_mfm` | Google Drive file `1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI` | Apache-2.0 | `commercial` | github.com/mever-team/spai LICENSE (Apache-2.0); README: 'The source code and model weights of this project are released under the Apache 2 License' |
| `fused.opensdi` | Verified local mirror of Hugging Face `aonuzhdin/FUSED` at `b7dfe0cb2d2f8fc8fd9e6ebea5f7185cfa8e1af7`; SHA-256 `7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c` | MIT | `commercial` | The HF checkpoint card declares MIT for the weights. The separate `AntonNuzhdin/FUSED` source repository has no established licence at the inspected revision. Its code is neither vendored nor imported by the application; the runtime uses our independent tensor implementation. This row does not assert source redistribution permission or service admission. See `docs/fused-local-research.md`. |
| `fused.convnext_xxlarge` | Verified local mirror of Hugging Face `timm/convnext_xxlarge.clip_laion2b_soup`; SHA-256 `1ddf6dbc7f0753fc4dfec5db5fa474acfd1ac6f67f466d6a976780c30b3a1b23` | Apache-2.0 | `commercial` | The HF backbone card declares Apache-2.0; the runtime uses the installed Apache-2.0 `timm` implementation and local safetensors. No remote source or weight loading. This dependency is loaded inside the composite runtime. |
| `mesorch.full_98` | Google Drive file `1PJxKteinMyaAYokKy0JhuzBnBc6bGsau` | MIT | `commercial` | github.com/scu-zjz/Mesorch LICENSE (MIT, Copyright (c) 2024 K-Squad; GitHub licence API: MIT). The checkpoint is linked from the upstream README, which names no separate weights terms; the repository licence is taken to cover it. Training images (CASIAv2, FantasticReality, IMD2020, tampCOCO per upstream balanced_dataset.json) carry their own terms, not reviewed (IF §2) |
| `radar.fusion_15k` | Google Drive file `1oPuC8nHkZex2VQBAPtmNEtxdSF-xylWM` | Apache-2.0 | `commercial` | github.com/alex-costanzino/RADAR-code LICENSE.txt (Apache-2.0, Copyright (c) 2025 BBC R&D, based on DINOv2); the README links the checkpoints with no separate terms. Training images: BBC-PAIR (github.com/bbc/PAIR, BBC terms, not reviewed). The assembled detector also needs depthanything.v2_base, which is CC-BY-NC-4.0 |
| `radar.projector_15k` | Google Drive file `1OL-PTnNzYKK1JmLa67S6cRYl-R3Jt_43` | Apache-2.0 | `commercial` | github.com/alex-costanzino/RADAR-code LICENSE.txt (Apache-2.0, Copyright (c) 2025 BBC R&D) |
| `dinov2.vitb14_reg4` | <https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth> | Apache-2.0 | `commercial` | github.com/facebookresearch/dinov2 LICENSE (GitHub licence API: Apache-2.0); the README releases code and model weights under Apache-2.0 |
| `dinov2.small` | Hugging Face `facebook/dinov2-small` at `ed25f3a31f01632728cabb09d1542f84ab7b0056` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0 (facebook/dinov2-small, not gated; checked 2026-09-13); github.com/facebookresearch/dinov2 LICENSE (GitHub licence API: Apache-2.0) |
| `xclip.base_p16` | Hugging Face `microsoft/xclip-base-patch16` at `d6184e3fd8780d04c85d0f1eabe5f94bf44d98f6` | MIT | `commercial` | HF model-card licence mit (microsoft/xclip-base-patch16, not gated; checked 2026-09-13); github.com/microsoft/VideoX LICENSE grants the MIT License, Copyright (c) Microsoft Corporation (the file opens with a project title, so the GitHub licence API reports NOASSERTION) |
| `syncnet.latentsync_stable` | Hugging Face `ByteDance/LatentSync-1.5` at `32a20d29aead0498e3e885e90dbbe8027da1b61b` | LicenseRef-OpenRAIL++-M | `unknown` | HF model-card licence openrail++ (ByteDance/LatentSync-1.5, not gated; checked 2026-09-13). CreativeML Open RAIL++-M Attachment A bans use for the administration of justice, law enforcement, immigration or asylum processes (#11) and fully automated decision making that adversely impacts an individual's legal rights (#6); Veritas has a litigation journey (PRD J2) and a forensic / court report profile (RP-03), so under D-04 (OpenRAIL counts as commercial only without field-of-use conflicts) the class is unknown and off by default pending a supervisor decision. Model code github.com/bytedance/LatentSync is Apache-2.0 (vendored in app/third_party/latentsync with its LICENSE and attribution) |
| `depthanything.v2_base` | Hugging Face `depth-anything/Depth-Anything-V2-Base-hf` at `b1958afc87fb45a9e3746cb387596094de553ed8` | CC-BY-NC-4.0 | `non_commercial` | HF model-card licence cc-by-nc-4.0 (Depth Anything V2 Base, Small is Apache-2.0 but has another width: RADAR's fusion block was trained on the 768-dimensional Base features). Non-commercial: under the default commercial policy loc.radar runs only when allow-listed (D-04) |
| `gend.clip_l14` | Hugging Face `yermandy/GenD_CLIP_L_14` at `891ce014a0308386c4d7d25b3dcf436a22db5504` | MIT | `commercial` | HF model-card licence mit; github.com/yermandy/GenD LICENSE (GitHub licence API: MIT); base encoder openai/clip-vit-large-patch14 from github.com/openai/CLIP (GitHub licence API: MIT) |
| `siglip2.so400m_p14_384` | Hugging Face `google/siglip2-so400m-patch14-384` at `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0 (google/siglip2-so400m-patch14-384, not gated; checked 2026-09-12) |
| `pe_core.large_p14_336` | Hugging Face `timm/vit_pe_core_large_patch14_336.fb` at `e63206c8e3a0e9b699e40f31080eebd78fd2258e` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0 (timm export, not gated; checked 2026-09-12); timm 1.0.28 pretrained_cfg licence apache-2.0; weights originate from facebook/PE-Core-L14-336 (apache-2.0). The export's config.json says 'custom'; the card and timm registry say apache-2.0 |
| `yunet.2023mar` | <https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/face_detection_yunet_2023mar.onnx> | MIT | `commercial` | opencv_zoo models/face_detection_yunet/LICENSE (MIT, Shiqi Yu) |
| `dfdc.effnet_b7_555` | <https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19> | MIT | `commercial` | github.com/selimsef/dfdc_deepfake_challenge (GitHub licence API: MIT) |
| `dfdc.effnet_b7_999` | <https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/0.0.1/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23> | MIT | `commercial` | github.com/selimsef/dfdc_deepfake_challenge (GitHub licence API: MIT) |
| `hf.haywoodsloan.ai_image_detector_deploy` | Hugging Face `haywoodsloan/ai-image-detector-deploy` at `d6a3aca2287455efa16f82946e0c5169929c27de` | NOASSERTION | `unknown` | no licence in the model-card metadata or README at the pinned revision (checked 2026-09-11) |
| `hf.ateeqq.ai_vs_human_image_detector` | Hugging Face `Ateeqq/ai-vs-human-image-detector` at `60e82406916921b823616bee33397baab38af3f0` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0 |
| `hf.dima806.deepfake_vs_real_image_detection` | Hugging Face `dima806/deepfake_vs_real_image_detection` at `29e4cf9efc543845610045f6ba7e88e5cf9d9301` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0 (base model google/vit-base-patch16-224-in21k) |
| `hf.wvolf.vit_deepfake_detection` | Hugging Face `Wvolf/ViT_Deepfake_Detection` at `f7f34ca885b39dfa712057fa3cc34c9372003003` | NOASSERTION | `unknown` | no licence in the model-card metadata or README at the pinned revision (checked 2026-09-11) |
| `hf.gustking.wav2vec2_xlsr_deepfake` | Hugging Face `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` at `f7050b586236dc910d1157f430def2d0647b02b4` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0 |
| `hf.mo_thecreator.deepfake_audio_detection` | Hugging Face `mo-thecreator/Deepfake-audio-detection` at `e4d9874b493362149cec96ced85f00b00b1a04c0` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0 |
| `hf.desklib.ai_text_detector_v1_01` | Hugging Face `desklib/ai-text-detector-v1.01` at `5fdea974cd4287c61674951ec78803aa274e2fb7` | MIT | `commercial` | HF model-card licence mit (base model microsoft/deberta-v3-large) |
| `local.desklib_ft_v2` | made here: `weights/desklib-ft-v2` (scripts/finetune_text_detector.py) | NOASSERTION | `unknown` | Veritas fine-tune of desklib/ai-text-detector-v1.01 (MIT) on a corpus mixing Wikipedia, CNN/DailyMail, HC3 and model outputs including Claude-written paragraphs (docs/finetune_text.md); training-data terms not reviewed (D-12) |
| `hf.hello_simpleai.chatgpt_detector_roberta` | Hugging Face `Hello-SimpleAI/chatgpt-detector-roberta` at `d2b342c61775d5dd0221808a79983ed3b86ffd86` | NOASSERTION | `unknown` | no licence in the model-card metadata or README at the pinned revision (checked 2026-09-11) |
| `hf.qwen.qwen2_5_1_5b` | Hugging Face `Qwen/Qwen2.5-1.5B` at `8faed761d45a263340a0528343f099c05c9a4323` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0; LICENSE file in repository |
| `hf.qwen.qwen2_5_1_5b_instruct` | Hugging Face `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Apache-2.0 | `commercial` | HF model-card licence apache-2.0; LICENSE file in repository |

## Audio additions reviewed 2026-10-31

The registry now also pins Silero speech activity (MIT), Pella v2 plus its W2v-BERT 2.0 configuration (publisher MIT declarations), and PANNs Cnn10 (weights CC-BY-4.0; inference source MIT). Full attribution and copied notices are in [the public legal notice](frontend/public/licenses/audio-model-notices.txt). The code under `app/third_party/panns` retains its MIT notice; `app/third_party/kaldi` retains the TorchAudio BSD-2-Clause notice. The latter is the exact filterbank arithmetic with unused MFCC/native imports removed. No sound-category output is treated as authenticity evidence. See [commercial audit](docs/commercial-model-audit.md) for licensing layers, unresolved provenance, pins and measured limitations.
