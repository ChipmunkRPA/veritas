# Focus changes can coincide with a face swap

On 352 held video deliveries, an experimental rule that hid face-transition
findings during broad sharpness changes removed two ordinary alerts. It also
hid 20 of 32 correctly located face swaps that coincided with a staged
sharpness change. Both delivery clocks failed the prespecified requirements.
The app keeps the recorded transition, its source frames, and its zero
confidence weight.

## Comparison

The eight groups supply a genuine recording and publisher-provided
face-swap material. We held these groups out of the focus-rule development
set and excluded known matching captures, donor file references, and assets
from the earlier 25 and 30 fps sustained and sparse transition studies.
Some sources have broader project-level exposure.
The comparison staged leading and trailing face swaps at known native
frames. Its ordinary controls were the original, a whole-picture softening
or sharpening, blur limited to the center or background, a gain ramp, a
same-recording jump, and a cross-recording cut. A further positive combined
a face swap with a simultaneous whole-picture sharpness change.
[Publisher data and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)

Each source picture sequence used 120 native frames from 30 fps material.
We tested one H.264 encode and a more compressed second H.264 encode at
30 fps and at a delivery rate of 30000/1001 fps. The latter changes the
delivery clock to 29.97 fps and preserves the authored frame order. Its
results describe an offline replay of the same method on a changed delivery
clock. The serving check admits the qualified exact 30 fps clock. The sharpness
changes were digitally staged, so this panel leaves optical focus behavior
in camera recordings unmeasured.

For each localized transition, the experimental filter compared six frames
around the boundary. It measured changes in fine relative to coarse picture
detail in textured tiles outside the face, where coarse picture structure
remained comparable. It required a consistent change across multiple image
quadrants. The rule and points were fixed before the confirmation set ran.
The comparison held the existing face-transition reading fixed, and applied
the experimental filter only to that reading.
[Current 30 fps face-transition check](video-face-single-boundary-30fps-2026-11-18.md)

| Delivery clock and encoder | Standalone swaps at exact frame / 16 | Swap plus focus at exact frame / 8 | All ordinary alerts / 64 | Whole-picture focus alerts / 16 |
| --- | ---: | ---: | ---: | ---: |
| 30 fps, direct | 16 → 16 | 8 → 3 | 3 → 2 | 1 → 0 |
| 30 fps, second encode | 15 → 15 | 8 → 3 | 3 → 3 | 1 → 1 |
| 29.97 fps, direct | 16 → 16 | 8 → 3 | 4 → 4 | 2 → 2 |
| 29.97 fps, second encode | 16 → 16 | 8 → 3 | 4 → 3 | 2 → 1 |

In each row, the first value is the existing face-transition reading and
the second applies the experimental filter. The two removed ordinary
deliveries came from the same staged focus recovery. The hidden compound
edits came from the same five groups in every delivery arm. Ordinary
central-only blur produced two alerts in each arm, all retained by the
filter. There were no wrong edit locations and no alerts on the untouched,
gain, jump, cut, or background-only blur controls. A broad sharpness change
can occur alongside a local face edit and does not determine the cause of
the face transition.

The browser and standard PDF now explain that focus can produce a similar
pattern and can coincide with a swap. Local replay checked a known edit, an
ordinary focus change, and an unchanged recording. All three preserved the
same JSON evidence and verified the regenerated signed PDF against their
source videos. The two localized cases kept all 12 displayed original
frames and face crops. No new analysis or billing jobs were created.

## Reproduce the counts

The native-input review replayed 42,240 decoded RGB checks and timestamps.
It verified 327 source-frame records used for face crops and 654
surrounding-picture frame references used by the adjudication. Those source
materials stay local. The downloadable research record contains anonymous group labels,
edit and transition frames, the eligibility outcomes, and the exact
scalar readings needed to recalculate the fixed focus decision. Its
release commitments refer to closed private records. The public checker
recreates the counts and frozen decision requirements from the logged
readings with the Python standard library.

```sh
python3 -I -S backend/scripts/review_video_face_focus_guard.py \
  docs/evaluation-panels/video-face-focus-context-transfer-2026-11-18.json.gz
```

[Anonymized paired comparison](evaluation-panels/video-face-focus-context-transfer-2026-11-18.json.gz)
