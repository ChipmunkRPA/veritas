# Can a lip check find two replaced words from the same speaker?

The existing short-clip lip check looks for changes in the amount of delay
between mouth motion and speech. Replacing words can change the relationship
between the visible mouth and the soundtrack while leaving speech at roughly
the same point in the clip. This comparison assesses that separate question
with real recordings and exact splice locations.
[Timing and articulation research](https://arxiv.org/html/2610.09223v1)
· [Same-speaker replacement research](https://arxiv.org/html/2610.30872v1)

## Sources and controls

The Grid Lombard archive supplies both the original pictures and the
target and donor speech. Publisher alignment labels identify the fourth
and fifth words of each six-word utterance. The two donors come from the
same speaker and recording condition as the target. One repeats the same
letter and digit. The other speaks a different letter and digit. Both donor
regions have sample counts within 15% of the target and within 60 ms of
each other. Both undergo the same resampling, constrained gain adjustment,
and 8 ms edge blends. All cases use the same original pictures. The replacement
changes only the samples inside the two-word interval. The alignment
boundaries are supplied by the publisher and can have timing error.
[Publisher archives and CC BY 4.0 terms](https://zenodo.org/records/3736465)

The source roster excludes the 32 speakers in the preceding primary and
carrier panels. The selected 16 speakers each contribute one plain and
one Lombard delivery. Six speakers set operating points, and ten separate
speakers supply the held evaluation. Each held target has an untouched
copy, a local gain change, a matched same-word splice, a different-word
splice, and two constant-delay controls. Lombard speech records how people
talk in noisy conditions. The benchmark does not add background sound to
the delivered recordings. [Earlier source panels](video-lip-mouth-width-transfer-2026-11-15.md)

## Comparison

Both new variants use the existing predicted mouth opening and measured
mouth opening. They search 400, 560, and 720 ms intervals and require more
disagreement inside the proposed interval than on either side. One
compares sound and pictures at their native positions. The other chooses
one lag that best fits both flanks. Only candidates with before, during,
and after pictures at verified original timestamps can be returned.
Evidence describes a mismatch between observed mouth motion and predicted
opening for the speech, with no inferred local delay from either new variant.
[Existing aperture check](video-lip-aperture-scale-2026-11-15.md)
· [Pinned speech checkpoint](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)

Each new variant can supply a range only when the existing checks give
none. The top and second-highest ordinary-control scores across
calibration speakers define the two research points. A useful location
must reach 0.40 temporal intersection over union against the exact
different-word splice. Same-word edits and constant delays are separate
controls. Decisions do not receive the donor recording or word labels.

## Held results

The two operating points produced identical decisions on these 120 held
deliveries. Each pair of plain and Lombard recordings belongs to the same
source speaker.

| Comparison | Correct ranges on different words | Ordinary original or gain flags | Same-word splice flags | Constant-delay flags |
| --- | ---: | ---: | ---: | ---: |
| Existing timing checks | 3/20 | 1/40 | 3/20 | 3/40 |
| Also compare mouth opening at native alignment | 3/20 | 3/40 | 3/20 | 3/40 |
| Also compare at one shared flank lag | 3/20 | 1/40 | 3/20 | 3/40 |

The current check's three correct ranges came from three speakers. One
also had an overlapping flag on its same-word control. In one correct
result, the replacement was **0.670–1.330 s** and the timing check flagged
**0.800–1.520 s** on the soundtrack. Original picture indices 18, 27 and
36 anchor the before, during and after mouth crops at approximately
**0.753 s**, **1.129 s**, and **1.505 s**. The check also records the
picture interval separately from the audio interval.

Both candidates reduced unassessed different-word cases from two to zero,
and neither found an additional correct range. The native-alignment
variant added two ordinary alerts. Both variants stay in evaluation.
The recordings and splices test visible speech compatibility under this
controlled setup. They supply no evidence about AI authorship, generated
mouths, or noisy soundtracks.

## Replay

The published evaluation includes the qualified source hashes, exact
audio intervals, measured ranges and source picture commitments. The
reviewer replays the control counts and recalculates the witness timestamps
on the original picture clock. Audio and picture bytes, identity labels,
landmark tracks, and speech posterior tensors remain local.
[Numeric receipt](evaluation-panels/video-lip-content-transfer-2026-11-15.json.gz)

Two complete native runs reproduced the frozen source plan and receipt
byte for byte. The numeric results agree on Python 3.12 and 3.14. A
separate decode verified 32 full source clocks and 781 unique original
frames and mouth crops, covering all 1,785 recorded picture references.

```sh
export PYTHONPATH=backend
module=scripts.video_lip_content_transfer
receipt=docs/evaluation-panels/video-lip-content-transfer-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

Use [the pinned publisher inputs](video-neural-phone-clock-2026-11-14.md#replay)
and [the timing assets](video-lip-viseme-transfer-2026-11-15.md#independent-replay).
The capture uses the same Mac GPU profile as the preceding native replay.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$sources" "$run/roster.json"
"$runner" -B -m "$module" run "$run/roster.json" "$sources" "$assets" "$run/content.json.gz"
cmp "$receipt" "$run/content.json.gz"
"$runner" -B -m "$module" audit-pictures "$receipt" "$run/roster.json" "$sources"
```
