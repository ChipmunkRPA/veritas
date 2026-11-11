# Whole-clip face-swap comparison

The previous appearance check looks for a change between neighboring time groups. A consistent face swap can occupy the entire clip and leave no original flank. This experiment compares two face-crop classifiers on that separate case. It uses the installed, checksum-pinned weights and the application's crop and scoring code, with one frame per forward. [GenD paper](https://openaccess.thecvf.com/content/WACV2026/papers/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.pdf) · [Pinned model card](https://huggingface.co/yermandy/GenD_CLIP_L_14/blob/891ce014a0308386c4d7d25b3dcf436a22db5504/README.md) · [Earlier bounded-swap comparison](video-face-specialists-v20-result.md)

## Method

The first ten groups in SHA-256 order of `continuous-swap-2026-11-11` concatenated with the source-group ID were fixed for each role. Development and confirmation use the previously retained, disjoint v6 and v17 RTFS rosters respectively. Each group contributed its original and two full-clip swaps from separate generators. The test used the saved matched delivery clips. It also kept the unchanged videos from qualified camera slots 1, 4, 5, and 8, plus one publisher's synthetic-base swap pair as transfer checks. The unswapped member of that pair is synthetic-origin and is labeled separately from the genuine controls. [RTFS source](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md) · [Camera qualification](video-vcd2-matched-controls-v290-result.md) · [Publisher pair](video-faceswapai-source-v185.md)

Six native source frames were selected at equally spaced interior times. Both arms used the same decoded RGB, source timestamps and largest-face boxes. The classifiers used their own production crop path and float32 on the local GPU. The clip statistic was the median, with at least four scored frames required. Each arm's operating point came from the 90th percentile of the ten development originals, with equality excluded. Both flagged one development original. The confirmation responses and transfer cases had no role in selecting those points.

## Results

Both arms scored every planned clip and verified identical source inputs on all 396 sampled frame positions per arm. The separate arithmetic review reproduced every clip median, operating point and aggregate flag count. [Exact aggregate](evaluation-panels/video-full-clip-face-swap-transfer.json)

| Confirmation or transfer input | Current face-frame classifier | GenD |
| --- | ---: | ---: |
| Continuous swaps flagged | 1/20 | 3/20 |
| Continuous swaps missed | 19/20 | 17/20 |
| Original paired clips falsely flagged | 1/10 | 0/10 |
| Genuine camera clips falsely flagged | 0/4 | 0/4 |
| Publisher full-clip swap flagged | 0/1 | 0/1 |
| Publisher unswapped synthetic base flagged | 0/1 | 0/1 |

The four correct flags came from different clips. An exploratory union would have flagged 4/20 swaps and 1/10 paired originals. The union was calculated after the result and needs a separate evaluation. Both checks miss too many full-clip swaps for a default change. This stage measures presence at sampled faces. It leaves clip-level confidence, frame-level localization, and complete app behavior outside the result.

Each guarded arm completed on the Mac. Elapsed times were 128 seconds for GenD and 138 seconds for the reference. Peak sampled process-tree RSS was 1.58 and 0.97 GiB respectively. Maximum observed GPU driver allocations were 2.04 and 1.58 GiB. The timings include native decoding, source verification and model loading.
