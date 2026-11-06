# Image localization checkpoint comparison

This comparison evaluates the currently selected OpenSDI checkpoint against the multi-generator So-Fake checkpoint from the same publisher. Both use the existing independent inference implementation and the same frozen semantic backbone. The publisher declares MIT for both checkpoints. The backbone has an Apache 2.0 grant. The benchmark assets remain in the ignored local evaluation directory and retain their source terms. [Checkpoint release](https://huggingface.co/aonuzhdin/FUSED/tree/b7dfe0cb2d2f8fc8fd9e6ebea5f7185cfa8e1af7), [backbone release](https://huggingface.co/timm/convnext_xxlarge.clip_laion2b_soup).

| Checkpoint | SHA-256 | Size |
| --- | --- | ---: |
| OpenSDI baseline | `7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c` | 319,859,039 bytes |
| So-Fake candidate | `0f9273ba9d48b1a6de0d2b602c7a26cdd3624967f465ff0683668ef9c4359ad3` | 319,861,822 bytes |

The candidate has the same 879 tensor keys, shapes and dtypes. The loader validates the pinned file hash and the exact legacy configuration before applying the shared inference operations. The candidate config has `attention_branches: 8` where the current configuration has `num_latents: 8`. The publisher's evaluator filters obsolete constructor fields and supplies eight queries from its current config. The matching learned tensor shapes and fixed pooling and segmentation operations allow reuse of the independent runtime and existing backbone. [Publisher's evaluator](https://github.com/AntonNuzhdin/FUSED/blob/e8580ebad33decf0787a96133af4fafee640e024/eval/opensdi.py).

## Test design

The fixed external set uses 72 distinct COCO source IDs from the TGIF and TGIF2 authors’ test splits. It pairs each authentic photo with one spliced edit. The three groups each have 24 pairs, split equally between object segmentation and bounding-box edit shapes. The generation families are Adobe Firefly, Stable Diffusion 2 and FLUX.1 Fill. The script orders the author metadata by salted SHA-256 and selects generation variation zero before looking at inference. COCO IDs from the prior local web sample are excluded. The authors' test split and the distinct COCO source IDs limit overlap with earlier local tests. The candidate's training dataset also draws from COCO, so the comparison cannot rule out training-source overlap. [Benchmark and CC BY-SA 4.0 release](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea), [benchmark paper](https://doi.org/10.1186/s13635-026-00235-9), [training dataset paper](https://openreview.net/pdf?id=frqg27qMkd).

The full published splice and uncropped authentic image share native pixel coordinates. The ground truth comprises exact RGB changes between that pair. The separate author mask is preserved and checked for matching geometry and overlap. This distinguishes the pixels that changed in the delivered splice from the prompt mask, which can have different boundaries because of feathering. Each test image, author mask and derived reference carries a byte hash. The archive reader streams only requested named PNG members to content-addressed files, with a verified cache index for retries. It never extracts a remote filesystem tree.

The completed input set is in `backend/data/localization-eval/tgif-pairs-v2/`. Its plan SHA-256 is `afdf1caccb11b0a6a674aab42c1d019b6224a87281c8aa42c980135cb99d4ac4`. Its manifest SHA-256 is `4ec157d240f88fce43e554d5dee0bd2da9ac388345d3f0a188c93826629c8230`. The median changed area is 4.5% for Firefly, 2.8% for SD2 and 3.8% for FLUX Fill. Median shares of changed pixels inside the authors' masks are 94.7%, 90.9% and 92.7%, respectively. The nearest 63-bit DCT perceptual-hash distance is 16 bits within the 72 authentic controls and between those controls and the 200 earlier OpenSDI authentic controls.

The same application preprocessing, image gate of 0.5, recorded map threshold of 0.5, quality exclusions, connected components and two-region per-map presentation cap apply to both checkpoints. The paired outcomes include authentic photos with any displayed region, edited images with no selected overlap, mean edited IoU and pixel F1. Missed edits count as zero overlap. Bootstrap intervals use the same source images for both checkpoints. The comparison measures this inspection layer and does not claim a calibrated probability of AI generation.

Replacement depends on a meaningful improvement in overlap or false alerts and on the tradeoff across edit families. Preserve the baseline for rollback. The already consumed OpenSDI evaluation can check for regressions on whole-image regeneration artifacts, with the fresh TGIF pairs answering a different question about splices.

## Fixed TGIF comparison result

Both checkpoints scored all 144 files on the same Mac in float32. The results SHA-256 is `ef51af293d6b019cb2c872d0d17d4cf0df833d7ee173940b759005373fae733a` under `backend/data/localization-eval/tgif-pairs-v2-results/`.

| Checkpoint | Authentic files with displayed regions | Edited files with any overlap | Edited mean IoU | Edited mean pixel F1 |
| --- | ---: | ---: | ---: | ---: |
| OpenSDI baseline | 3/72 | 9/72 | 0.04878 | 0.06601 |
| So-Fake candidate | 0/72 | 5/72 | 0.03315 | 0.04411 |

The candidate reduced false highlights but also suppressed four additional edits. The paired IoU change was −0.01563 with a 95% bootstrap interval of −0.06284 to +0.03067. Firefly showed better overlap on four edits with the candidate versus three with the baseline. The candidate missed all 24 FLUX Fill edits at this operating point, while the baseline overlapped five. Both models handled splices much less effectively than the earlier OpenSDI comparisons.

Removing the image gate retained regions in every authentic image for both checkpoints. On the edited files, ungated mean IoU was 0.20274 for the baseline and 0.18153 for the candidate. Its image-level scores ranked edits somewhat better overall (AUROC 0.661 vs 0.622). The weaker maps and additional misses led us to defer replacement.

A follow-up confirmation cohort excludes all 72 source IDs above, as well as the earlier web controls. The source plan is frozen before any score is read. Alternate splicing localizers are compared under their published settings with the same geometry and display constraints. The existing test results remain recorded separately.

## Alternate splicing checkpoint screening

TIDAL vR.P.30.1 has a published Apache 2.0 checkpoint and a compact ResNet-34 U-Net. It measures local differences when an image is recompressed at three qualities. Its default `>0.70` map threshold and 400-pixel native area rule show regions with overlap on **50/72** edited images (mean IoU **0.3311**) in the first cohort. At those settings it also shows regions on **23/72** authentic images. That default is unsuitable despite the stronger edit masks. [Weight release](https://huggingface.co/the-harsh-vardhan/TIDAL/tree/6bc4c3fd066f13d614949db64eb7f1abdc912ada), [publisher's reference inference](https://huggingface.co/spaces/the-harsh-vardhan/tidal-api/tree/ff9cb49fa77a80b430a603a396cd10610e7e4298).

The alternate checkpoint script also compares the standalone MADL dual-stream response using its released ConvNeXt-Tiny classifier and proposed manipulation map. The local run checks reviewed source and checkpoint hashes, uses `weights_only=True`, requires an exact checkpoint state match and runs offline on CPU. [Weight release and Apache 2.0 terms](https://huggingface.co/moindy/MADL/tree/529e9ddd19586c18f9b47685240adcff5fa87007), [standalone component source](https://github.com/shannima/MADL/tree/af174679d88c7b605c56de34878e5ab2528f6794).

The follow-up TIDAL procedure uses 24 separate development originals and their edits to compare the native map with the minimum of the native and inverse-mirrored maps. Each rule uses the smallest integer area threshold at or above the published minimum that satisfies the development false alert budget. The candidate with the higher development mean IoU advances to 48 new, untouched source pairs. The procedure freezes the source split and selected rule before processing confirmation files. The 5% development target is comparable to the baseline's 3/72 rate in the first cohort. Actual confirmation counts determine the tradeoff.

The independent application-side checkpoint loader and image transforms reproduced the publisher's CPU float32 maps bit-for-bit on six inputs, one authentic and one edited example from each generator. This check covers numerical behavior before the evidence display policy.

The development split selected the native map with an area gate of 7,993 out of 147,456 screening cells above 0.70. The frozen rule file SHA-256 is `8eac15298b2b856ad9d906208c7dfa5095084677e640bb4618db02e24d95e5d7`. In development, the rule overlapped 7/24 edits and highlighted 1/24 authentic inputs. The corresponding mirror-consensus rule overlapped 6/24 edits at the same false alert count. The untouched PNG pairs gave 9/48 overlapping edits, 2/48 authentic highlights, mean IoU 0.11434 and mean pixel F1 0.13742. Overlap concentrated in larger edits: 6/12 at or above 10% of the image, and 0/21 below 3%. Re-encoding the same confirmation files at JPEG quality 75 reduced overlap to 0/48. This supports a PNG inspection scope and shows that prior compression can hide the signal.

At its fixed published settings the standalone MADL map overlapped 1/72 edits with mean IoU 0.00826 and marked 1/72 authentic originals. The other generators' edits went unmarked. It was not selected for local rollout.

## Matched confirmation and local selection

| Check on the same 48 untouched PNG pairs | Authentic photos with displayed regions | Edits with any overlap | Mean edited IoU |
| --- | ---: | ---: | ---: |
| Prior OpenSDI localizer | 4/48 | 5/48 | 0.04900 |
| Frozen TIDAL rule | 2/48 | 9/48 | 0.11434 |

Mean edited IoU improved in each generator group: Firefly from 0.03361 to 0.06712, SD2 from 0.05885 to 0.12740, and FLUX Fill from 0.05454 to 0.14849. The paired mean IoU improvement was 0.06534. Its bootstrap interval spans −0.01179 to +0.14657, reflecting the small confirmation cohort. The observed gains in both region overlap and false alerts support a scoped local replacement. The comparison receipt SHA-256 is `fd1ca0d26cc81890fdfa96f959399a2fd0db2e5d6b028dc0e768a69c89db1504`.

Install the optional `splice` Python dependency and place the pinned checkpoint at `models/weights/splice/tidal-v30.1.pt`. Add `loc.splice` to `VERITAS_LICENCE_POLICY.allow_detectors` in the local backend configuration. Keep `loc.fused` in that allowlist for other formats. The newer check selects full and deep PNG analyses when its local files and dependency are available. It writes the original screening response, quality exclusions and exact active-cell count to the evidence record. It carries zero weight in the whole-file confidence estimate. To roll back, remove `loc.splice` from the allowlist and restart the backend.

The paid-flow verification used the normal API, job queue and actual CPU checkpoint with an isolated database and test Stripe transport. The paired edited and authentic PNGs matched the measured native counts. Checkout created zero inference jobs before the signed paid webhook. The edited example produced one localization region and four exact evidence images. The authentic example withheld the localization region. API and JSON agreed with the frozen PDF and ZIP. Both reports and packages verified with the instance key, and customer exports passed a private model name scan. The rendered report sections and verification receipt are under `output/image-splice-rollout/isolated-final/`.

## Local reproduction

From the repository root, use a new ignored output directory. The `plan` command fixes source IDs and member paths before scoring. Repeat `--exclude-coco-id` for prior local source IDs. `acquire` can resume validated archive members after a network failure.

```sh
backend/.venv/bin/python backend/scripts/prepare_tgif_localization.py plan backend/data/localization-eval/tgif-check --count 24 \
  --exclude-coco-id 15440 --exclude-coco-id 15810 --exclude-coco-id 24260 --exclude-coco-id 10393 \
  --exclude-coco-id 6954 --exclude-coco-id 26501 --exclude-coco-id 7938 --exclude-coco-id 3983
backend/.venv/bin/python backend/scripts/prepare_tgif_localization.py acquire backend/data/localization-eval/tgif-check
backend/.venv/bin/python backend/scripts/compare_image_localizers.py backend/data/localization-eval/tgif-check backend/data/localization-eval/tgif-check-results --device mps
```

The comparison reads the two named, digest-pinned files in `models/weights/fused/` and the existing `convnext.safetensors`. Run it outside other heavyweight model jobs on the Mac. The alternative `--device cpu` uses the same float32 inputs and region policy. The script records each exact native model map and paired result under the selected results directory.

For the consumed OpenSDI regression set, pass `--baseline-cache backend/data/localization-eval/fused-holdout-offset20-count20/fused-research-f32/results.json` and use its parent dataset directory as `manifest_root`. That option verifies the exact earlier model output file and native arrays, requires its matching manifest and device, and replays the current writer quantization on those arrays. The new checkpoint still runs fresh inference.

A CPU replay matched all gated and ungated per-map display metrics exactly for all 200 previously audited OpenSDI baseline maps. The existing localization runtime and image report test selection passed 53 tests.

The final application-side TIDAL implementation reproduced the exact native map bytes, active cell counts and displayed-mask metrics on all 96 confirmation files. Replaying the 144 initial maps through the final component rank also reproduced every stored metric.
