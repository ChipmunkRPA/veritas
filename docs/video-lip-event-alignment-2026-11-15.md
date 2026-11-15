# Ordered lip and speech events under local timing changes

This experimental check matches audible B and P events to independently
observed lip releases on the source clock. It compares two explanations
for the same sequence of events. The first uses one constant offset. The
second permits one short change with distinct timing support before and
after it. A useful observation identifies a range and the direction of a
local audio offset. It does not assign an AI origin.
[Recent timing and articulation research](https://arxiv.org/abs/2610.09223) ·
[Earlier phone and lip measurements](video-bilabial-mouth-transfer-2026-11-14.md)

## Comparison and results

The comparison enumerates ordered one-to-one associations of audible
events and visible releases. It penalizes omissions and the extra timing
change, and leaves competing nearly equal associations ambiguous when
they point to a different range or timing direction. The core range spans
the matched events with 80 ms of context. The wider readout extends that
range toward the surrounding timing anchors by up to 180 ms from either
outer event. This expansion only changes the extent of the review window.
Four separate calibration speakers supplied both decision points before
held scoring. Both thresholds coincided with the zero cost-improvement
floor. Useful localization requires at least 0.40 temporal intersection
over union and a signed timing estimate within 80 ms of the authored shift.

The two held panels reuse observed source pictures and recorded phone
timings from the earlier Lombard Grid studies. Their 128 authored local
shifts and 320 controls come from 16 held speakers. The audio edits reuse
recorded speech. [Lombard Grid source and license](https://zenodo.org/records/3736465) ·
[Phoneme recognition checkpoint](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)

| Timing check | Useful local locations | Ordinary flags | Constant-delay flags |
| --- | ---: | ---: | ---: |
| Current app timing arithmetic | 0/128 | 0/192 | 0/128 |
| Prior matched-lip-release candidate | 5/128 | 0/192 | 0/128 |
| Ordered events, core span | 3/128 | 0/192 | 0/128 |
| Ordered events, wider span | 4/128 | 0/192 | 0/128 |
| Both experimental methods combined | 6/128 | 0/192 | 0/128 |

Only 27 of 128 authored shifts had at least three audio events and three
visible releases. The wider readout added one correct location beyond the
prior candidate, in which the audio was 225 ms late against a known 200 ms
delay. Its source review range spans **1.092–1.613 s**. The ordered method
also flagged an unrelated range with the wrong timing direction in another
edited clip. [Ordered timing measurements](evaluation-panels/video-lip-event-alignment-2026-11-15.json)

## AAC soundtrack transfer

A fixed hash selected one clip per held speaker. All seven soundtrack
variants per source were encoded as 48 kbps AAC in M4A and decoded to
16 kHz mono. The replay checked the decoded duration and tested sample
origin within one sample. Visual observations stayed on their original
picture clocks. The neural acoustic reader scored the decoded soundtrack
without labels. The frozen timing readouts located **0/32 local changes**
both before and after compression on that subset. The ordered check chose
an incorrect range once in each condition. There were zero flags on the
remaining 48 ordinary deliveries and 32 constant-delay deliveries. These
results retain the cue in evaluation, with sparse visible and audible
event coverage as the main limitation.
[Compressed soundtrack measurements](evaluation-panels/video-lip-event-alignment-aac-2026-11-15.json)

## Replay

The portable recount reads only committed clocks, measurements and source
hashes. The optional native AAC rerun uses the publisher archives and
pinned acoustic checkpoint from the [parent source replay](video-neural-phone-clock-2026-11-14.md#independent-replay).

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panel=docs/evaluation-panels/video-lip-event-alignment-2026-11-15.json
codec=docs/evaluation-panels/video-lip-event-alignment-aac-2026-11-15.json
python3 -B -S -m scripts.video_lip_event_alignment review "$panel"
python3 -B -S -m scripts.video_lip_event_alignment review-compression "$codec"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.video_lip_event_alignment compression "$run/aac.json" \
  --receipt "$panel" --sources "$archives" --model "$checkpoint"
```
