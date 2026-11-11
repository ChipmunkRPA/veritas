# Matched video inputs for a known face-swap interval

The exact publisher input, production output and donor portrait are the
three verified v185 FaceSwapAI study assets. Both adult portraits were
created synthetically. The publisher releases the study under CC BY 4.0.
The prepared input and all constructed cases represent one source component.
[Verified source and rights](video-faceswapai-source-v185.md) · [Publisher study](https://faceswapai.com/resources/video-face-swap-continuity-study)

The original declares full-range `yuvj420p` with the BT.470 BG matrix
(Rec.601 coefficients). The publisher swap uses `yuv420p` with no range
or matrix tag. Decode the source with its declared range and matrix.
For the publisher output, keep all four combinations of limited/full
range and Rec.601/Rec.709 matrix, and normalize each to full-range native
RGB. These are fixed alternative interpretations of the same bytes.
An external color calibration or authoritative output tags would be
needed to identify the intended combination.

All cases use the same 1280 × 720 geometry, 240 native picture positions
and common lossless H.264 RGB QP0 final encoder. The exact time base is
1/12288 with 512 ticks per picture. Every output packet, including the
last, receives a 512-tick duration. No case contains audio.

| Case | Frames from index 72 through 143 |
| --- | --- |
| `unchanged` | Normalized original |
| `recompression_only` | Same original through ordinary CRF24 H.264, then back to full RGB |
| `swap_tv_bt601` | Publisher swap, limited range with Rec.601 matrix |
| `swap_pc_bt601` | Publisher swap, full range with Rec.601 matrix |
| `swap_tv_bt709` | Publisher swap, limited range with Rec.709 matrix |
| `swap_pc_bt709` | Publisher swap, full range with Rec.709 matrix |

The ordinary intermediate explicitly encodes limited-range Rec.601 YUV.
Every intervention preserves the same original source pictures outside
the half-open interval `[72,144)`, equal to `[36864,73728)` on the native
clock and `[3,6)` seconds. For each normalization, independently hash
the source RGB at every observed native timestamp. Decode every final
picture through the production video decoder and require exact agreement
with the selected source hash. Verify every packet PTS, DTS and display
duration, and save the actual changed-frame indices against the
`unchanged` baseline. Save native PNGs for frames 71, 107 and 144 and
decode them back to the same RGB hashes. The publisher swap can have
background changes within the inserted interval, including color and
compression differences. The timing label records which source frames
were substituted. It supplies no face-pixel ground truth.

Run in a fresh private work context from the repository root. The v108
data-only monitor applies its existing 900-second limit, 1.5 GiB sampled
process-tree RSS limit, 3 GiB initial available RAM and disk minimum and
1.5 GiB ongoing reserves. It checks the mounted Seagate volume. Retain
partial attempts without overwriting the source files. Use the private
monitor, native hashes and archived byte commitments to verify the
readout before publishing an aggregate. Media and PNGs stay local.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_matched_v186 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_matched_v186 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_matched_v186 readout
```

The original itself is a synthetic still-state sequence with crossfades.
This preparation supports a small diagnostic of responses to a
publisher-labeled face-swap intervention. It provides no genuine-camera
false-alarm estimate, lip-sync test or population sensitivity estimate.
