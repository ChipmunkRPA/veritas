# Measured picture change in paired video

Qualified earlier-copy comparisons can show a frame-by-frame interval of
strong local picture difference, with its earlier and analyzed native crops
from before, during and after. The browser and PDF also show the initial
comparison window. Both intervals and their separate source timestamps are
recorded in JSON. Gradual changes can extend past the displayed measured
interval. Conventional edits can produce the same evidence.

The option is `VERITAS_VIDEO_REFERENCE_MEASURED_INTERVAL=true`, used with
`VERITAS_VIDEO_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_VIDEO_REFERENCE_LOCAL_AREA=true`. The additional interval requires
matching native timelines on upright 720 × 480 H.264 4:2:0 clips at 23–25 fps,
with 40–100 pictures. It uses the existing localized difference, which compares
luminance after accounting for whole-picture brightness and contrast. A
contiguous run on the original seed rectangle must stand above the surrounding
frames and show a substantial change at both ends. The app keeps the crop
rectangle from the existing comparison and emits the second interval only
when it differs from the initial window. The interval has zero AI-confidence
weight. A separate earlier-copy alignment route handles trimmed videos.

The design draws on the distinction between coarse proposals and refined
temporal boundaries in Cao, Guo and Chi, and the consecutive-frame approach
in Umur, Güşlü and Delibasoglu. The MARS preprint also evaluates how local
residuals respond to motion and texture. This implementation measures paired
pixels, uses deterministic code, and imports no research model weights.
[CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_Structural-Semantic_Perception_for_Diffusion-Guided_Temporal_Forgery_Localization_CVPR_2026_paper.pdf)
[Consecutive-frame study](https://arxiv.org/abs/2606.07090)
[MARS](https://arxiv.org/abs/2611.12508)

## Local results

Each round authored 16 abrupt local brightness or texture edits, eight
gradual local edits, eight neutral re-encodes, eight global grades and eight
wrong-source pairs. The two rounds used distinct CREMA-D actor groups, also
excluded from the three earlier paired-area studies. The recordings had prior
exposure in other project studies. All authored boundaries were derived from
the source pictures before H.264 encoding. The archive receipts bind decoded
RGB to source hashes. CREMA-D publishes its individual recordings under
DbCL 1.0 and the database under ODbL 1.0.
[Pinned publisher license](https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

| Outcome | Initial trial | Source-disjoint confirmation |
| --- | ---: | ---: |
| Exact abrupt intervals, serving → candidate with serving fallback | 0/16 → 16/16 | 0/16 → 15/16 |
| Actors with an exact gain | 8 | 8 |
| Gradual edits with a refined interval | 4/8 | 0/8 |
| Added intervals on neutral delivery or global grade | 0/16 | 0/16 |
| Wrong-source pairs with an interval | 0/8 | 0/8 |
| Exact interval on previously exposed neural redraws | 4/17 → 9/17 | 4/17 → 5/17 |
| Native crop commitments independently replayed | 390 | 339 |

The first round failed its frozen gradual-edit criterion. The confirmation
raised the required edge step before scoring a second group of sources. Its
only missed abrupt interval was a texture change for which the existing
selector produced no usable proposal. A global grade produced an existing
local finding, with no additional interval. No previously qualifying crop
fell below 0.10 spatial overlap when the selected frame changed.

The neural diagnostic reused eleven previously exposed LivePortrait and six
MuseTalk redraws. It included eleven independently timing-qualified cases.
The confirmation reduced boundary error on one timing-qualified case and
preserved median error for both generators and the timing-qualified subset.
It missed the frozen requirement of a gain in two neural source groups. Both
full advancement decisions remain false. The serving option uses the narrower
factual scope supported by the source-disjoint conventional tests. Single-file
lip redraw detection and other codecs require separate evaluation.

The authenticated local app reproduced the earlier [11, 19) and measured
[13, 20) intervals on a held example. The measured interval shows 0.542 to
0.833 seconds and native frame pairs 12, 16 and 20. PDF and JSON used these
same timestamps and source crops. The separate neutral re-encode, gradual
change and wrong-source controls produced no additional interval. Turning
the option off reproduced the original comparison response.

## Replay

```sh
python3 -B -S backend/scripts/review_video_reference_boundary_transfer.py \
  docs/evaluation-panels/video-reference-measured-interval-development-2026-11-13.json
python3 -B -S backend/scripts/review_video_reference_boundary_transfer.py \
  docs/evaluation-panels/video-reference-measured-interval-confirmation-2026-11-13.json
```

Use `--run` with the retained run directory and `--previous` with the earlier
archived paired-mouth study to verify the native crop hashes. The offline
`compare_video_references.py --verify` command replays an API export against
its two source files. `verify_native_video_interval.py --earlier` checks both
native clocks and all twelve crop references when the additional interval is
present. The latter checks the exhibit geometry and source bytes. Rerun the
full comparison to reproduce the selected interval itself. The public receipts
contain no raw media or local source paths.
