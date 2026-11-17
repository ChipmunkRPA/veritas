# Comparing changed words with native mouth image sequences

The short lip-sync check can miss a word replacement when speech continues
at the original cadence. This trial adds a small visual model that estimates
visible speech groups from the mouth pixels and their motion over neighboring
native frames. The model uses the publisher's phone alignments in genuine
recordings. It looks for local disagreement with the independently identified
speech sounds in the delivered soundtrack. We use the distinction between
aligned cadence and sound-specific articulation from recent research.
[Articulation and timing study](https://arxiv.org/abs/2610.09223) ·
[Frame-level speech and mouth correspondence study](https://arxiv.org/abs/2610.30872)

The fit uses 336 untouched recordings from 14 speakers. A crop aligned on
the eyes and nose preserves a fixed scale relative to the face. A small
network uses both normalized brightness and the movement visible against
each clip's own median crop. It learns seven classes, including quiet and
sounds with less distinctive movement. The comparison uses five groups with
visible cues. The previous phone asset reads the sound separately. The
training, calibration and test speakers are separate. Landmark and speech
assets use the same pins as the local articulation check. No third-party
lip-reading model is used. [Lombard Grid source and CC BY 4.0 terms](https://zenodo.org/records/3736465) ·
[Existing checkpoint and landmark audit](video-lip-articulation-2026-11-15.md)

## First held comparison

Thirteen recordings from seven calibration speakers fix the points for PCM
and AAC separately. The comparison chooses between looking at each mouth
group's support relative to the clip and comparing it with a different
visible group at the same time. A candidate needs better agreement on both
sides at a shared offset. The original timing and articulation check takes
priority when it already reports a range. A useful location overlaps at
least 40% of the temporal union with the annotated replacement.

The two held panels reuse published source clips from earlier project
studies. The speakers are disjoint, with seven clips in the first panel
and 36 in the second. Unchanged clips and gain adjustments form ordinary
controls. Repeated same-word splices and whole-track delay controls stay
separate. Each PCM and AAC delivery preserves the original picture track.

| Held delivery | Word replacements located, existing → combined | Ordinary flags, existing → combined | Whole-track delay flags, existing → combined |
| --- | ---: | ---: | ---: |
| 7 clips, PCM | 1 → 1 of 7 | 0 → 2 of 14 | 0 → 1 of 14 |
| 7 clips, 48 kHz AAC | 1 → 1 of 7 | 0 → 2 of 14 | 0 → 1 of 14 |
| 36 clips, PCM | 5 → 7 of 36 | 3 → 7 of 72 | 4 → 12 of 72 |
| 36 clips, 48 kHz AAC | 6 → 7 of 36 | 3 → 7 of 72 | 3 → 11 of 72 |

The fit added two correct PCM locations and one AAC location in the larger
panel. The original narrow search covered up to 140 ms in either direction.
The whole-track controls move the soundtrack by 200 ms. Some plausible
constant alignments were outside that search. This version stays in
evaluation. [Frozen points](evaluation-panels/video-mouth-patch-selection-2026-11-17.json.gz) ·
[Held locations and controls](evaluation-panels/video-mouth-patch-held-2026-11-17.json.gz)

## Wider alignment and unused source recordings

The wider comparison searches offsets through 360 ms in either direction
and checks whether the same whole-track offset explains both halves of the
clip. We selected both variants on the same 13 calibration recordings before
measuring a new panel of 24 single-digit replacements in 24 separate readers.
All 24 readers are outside the fit and calibration groups. Eight are also
outside the earlier held groups. The target and donor utterances are absent
from the prior source rosters. Each staged change replaces one digit with
another rendition by the same speaker in the same recording condition.
The panel also contains a same-digit replacement, unchanged and gain-only
tracks, and both directions of a whole-track 200 ms delay.
[Source positions and digests](evaluation-panels/video-mouth-patch-fresh-sources-2026-11-17.json.gz)

| Fresh delivery | Correct locations, existing / narrow / wide | Ordinary flags, existing / narrow / wide | Whole-track delay flags, existing / narrow / wide |
| --- | ---: | ---: | ---: |
| PCM | 0 / 0 / 0 of 24 | 4 / 6 / 4 of 48 | 10 / 11 / 10 of 48 |
| 48 kHz AAC | 0 / 0 / 0 of 24 | 4 / 6 / 4 of 48 | 8 / 9 / 8 of 48 |

Both variants leave the two same-digit flags in each carrier unchanged.
The narrow variant adds one misplaced region per carrier. The wider variant
adds zero regions on these fresh deliveries, and adds zero useful locations
on the two earlier held panels. The single-upload app keeps its existing
check. A separate comparison with an earlier video recovers brief changes
on these same native clocks. [Narrow points](evaluation-panels/video-mouth-patch-narrow-selection-2026-11-17.json.gz) ·
[Wide points](evaluation-panels/video-mouth-patch-wide-selection-2026-11-17.json.gz) ·
[Fresh narrow outcomes](evaluation-panels/video-mouth-patch-fresh-narrow-2026-11-17.json.gz) ·
[Fresh wide outcomes](evaluation-panels/video-mouth-patch-fresh-wide-2026-11-17.json.gz) ·
[Earlier wide outcomes](evaluation-panels/video-mouth-patch-wide-held-2026-11-17.json.gz) ·
[Earlier video comparison](video-short-phrase-comparison-2026-11-17.md)

## Reproduce

Use the exact [source archives](video-neural-phone-clock-2026-11-14.md#independent-replay)
for the publisher PCM, pictures and phoneme positions. The source plan
contains each original member and its digest. The separate numeric receipts
are sufficient to verify split totals, frozen points and the count of
locations without opening the videos. Crops and intermediate audio features
stay in the local workspace.
[Source plan](evaluation-panels/video-mouth-patch-sources-2026-11-17.json.gz) ·
[Fit receipt](evaluation-panels/video-mouth-patch-fit-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
module=scripts.video_visual_speech_evaluation
python3 -B -S -m "$module" review "$panels/video-mouth-patch-selection-2026-11-17.json.gz"
python3 -B -S -m "$module" review "$panels/video-mouth-patch-held-2026-11-17.json.gz"
plan="$panels/video-mouth-patch-sources-2026-11-17.json.gz"
"$runner" -B -m scripts.video_visual_speech_train capture "$plan" "$sources" "$assets" "$run/fit-pictures" --phase fit
"$runner" -B -m scripts.video_visual_speech_train train "$run/fit-pictures" "$run/fit"
```

The initial narrow fit and comparison can be reproduced from
[the recorded version](https://github.com/ChipmunkRPA/veritas/commit/8fa5d385784b81026885352692c4c8e04a6eb73f).
The new source recipe rebuilds 288 playable files with the original picture
bitstream and frame ticks. It verifies 576 source and submission bindings,
including the exact PCM and AAC presentation samples. The reader can recount
the predicted ranges on the frame grid with the same overlap rule. Both the
existing and combined checks locate zero of the 48 different-digit deliveries
on that grid. [Playable source and event commitments](evaluation-panels/video-mouth-patch-fresh-playable-2026-11-17.json.gz)

```sh
fresh="$panels/video-mouth-patch-fresh-sources-2026-11-17.json.gz"
wide="$panels/video-mouth-patch-wide-selection-2026-11-17.json.gz"
result="$panels/video-mouth-patch-fresh-wide-2026-11-17.json.gz"
python3 -B -S -m "$module" review "$wide"
python3 -B -S -m "$module" review "$result"
"$runner" -B -m scripts.export_video_visual_speech_fresh "$fresh" "$wide" "$result" "$sources" "$run/visual-fresh"
python3 -B -S -m scripts.independent_review count-events \
  "$run/visual-fresh/manifest.json" "$run/visual-fresh/truth.json" \
  "$run/visual-fresh/combined.json" "$run/visual-fresh-events.json" --match .40 .40
```
