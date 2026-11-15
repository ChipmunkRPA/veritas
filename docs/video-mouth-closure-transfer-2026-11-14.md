# Can the video show the lip closure heard in the soundtrack?

The current timing check found none of 20 staged replacements where
the same speaker’s voice changes from a vowel to an M sound near the
same point in the clip. The 100 ms lip-closure check located four of
those replacements at its less restrictive operating point. It also
flagged six controls. The candidate stays in evaluation.

## What was measured

Timing and articulation answer different questions. Mouth movement can
remain close in time to speech while the shapes correspond to different
sounds. This comparison measures one specific articulation. An audible
`/m/` normally involves both lips meeting. The acoustic reader takes the
full recording and returns a timestamp for `m` or `mʲ` without access to
the expected words, pictures, edit range, or publisher labels. Its
emission must reach 0.55 and agree within 40 ms with a reading from an
independently filtered copy of the same sound. This repeat tests
stability under filtering. Both reads use the same acoustic checkpoint.
[Timing and articulation](https://arxiv.org/html/2610.09223v1)
· [Local speech-content comparison](https://arxiv.org/html/2610.30872v1)

The six development speakers established a median acoustic timestamp
bias of 127.5 ms from nine independently labeled events. After adjusting
for that delay, two variants inspect 100 or 160 ms on either side of
the expected closure. Each requires at least four consecutive native
pictures with complete mouth geometry and no adjacent gap above 50 ms.
The score is the smallest inner-lip distance, normalized by eye spacing,
across the included pictures. A larger minimum means no near-closure
was measured in that window. The strongest observation is retained per
file. The tighter and broader points use the largest and second-largest
ordinary development speaker maxima, respectively, with the same floor.
These settings measure an articulation cue and have no AI-confidence
weight. [Pinned Apache-2.0 acoustic checkpoint](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)
· [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
· [MediaPipe license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)

## Held comparison

Six development and ten held speakers each provided one plain and one
Lombard condition. The new roster excludes speaker codes in the two
preceding neural phone cohorts and the earlier exclusion list. Earlier
exposure outside those documented groups is unknown.
Each comparison uses archive-matched recordings of the same speaker.
The selected recording shows the speaker saying A or I. Its soundtrack
receives a 420 ms excerpt centered on a labeled M from another take.
The pictures and sound outside the authored interval stay exact. The
study also tests original recordings, filtering, gain, alignment of
same-letter re-recordings, and 120 ms constant shifts. Lombard refers
to the speaker’s recording condition. [Lombard Grid and CC BY 4.0 terms](https://zenodo.org/records/3736465)

A useful location requires at least 0.40 temporal intersection over
union with the edited interval and alignment with the inserted phone
within the declared 40 ms label tolerance. The baseline reads the app’s
current timing arithmetic on the same face pictures and soundtrack.
The check’s 20 positives and 160 controls are repeated views of ten
held speakers. They are a paired diagnostic rather than independent
population samples.

| Check | Dubs measured | Dubs located | Ordinary control flags | Constant-shift flags |
| --- | ---: | ---: | ---: | ---: |
| Current timing | 3/20 | 0/20 | 0/120 | 0/40 |
| 100 ms, tighter point | 20/20 | 3/20 | 2/120 | 4/40 |
| 100 ms, broader point | 20/20 | 4/20 | 2/120 | 4/40 |
| 160 ms, tighter point | 20/20 | 2/20 | 0/120 | 3/40 |
| 160 ms, broader point | 20/20 | 5/20 | 5/120 | 4/40 |

The four 100 ms broader detections came from three speakers. Its two
ordinary flags were on same-letter dubs. The 160 ms broader rule also
flagged an untouched M clip and its filtered and level-adjusted copies.
All constant-shift flags came from sound moved earlier. A consistent
timestamp error or differences between legitimate takes can therefore
produce a missing-closure lead. Even with the wider measurement the
check missed 15 of 20 changes. This comparison uses ordinary recorded
speech and authored cuts. Generated mouth motion and speech, spontaneous
speech, and recordings with camera cuts need separate validation.

One correct 100 ms range covers **1.065–1.265 s**, with native pictures
**26–30**. The staged edit spans **0.940–1.360 s**, and the inserted M
label spans **1.055–1.245 s**. The replay decoded and verified the exact
mouth crops used in the observation.
[Numeric results](evaluation-panels/video-mouth-closure-transfer-2026-11-14.json)

## Local replay

Standard-library Python recounts all outcomes, the disjoint speaker
split, each operating point, and the measured picture timestamps.
Native source replay verified 128 publisher sources, 288 waveforms,
3,681 picture timestamps, and 84 selected frames with mouth crops.
It uses the saved acoustic and aperture measurements and reruns the
media reconstruction without either recognition checkpoint. Independent
reviewers can obtain the original media directly from the publisher.
[Source replay receipt](evaluation-panels/video-mouth-closure-source-replay-2026-11-14.json)

```sh
export PYTHONPATH=backend
first=docs/evaluation-panels/video-neural-phone-clock-2026-11-14.json
second=docs/evaluation-panels/video-neural-phone-context-2026-11-14.json
receipt=docs/evaluation-panels/video-mouth-closure-transfer-2026-11-14.json
python3 -B -S -m scripts.video_mouth_closure_transfer review "$receipt" --parents "$first" "$second"
backend/.venv/bin/python -B -m scripts.video_mouth_closure_transfer replay "$receipt" \
  --parents "$first" "$second" --sources "$sources" --output "$replay"
```

For a native measurement rerun, prepare the full two media archives,
publisher JSON and alignments, and the pinned acoustic checkpoint as
described in the [phone benchmark](video-neural-phone-clock-2026-11-14.md#independent-replay).
Use the same face landmarker and reviewed geometry runtime as in the
[lip-release benchmark](video-bilabial-mouth-transfer-2026-11-14.md#replay).
The measured run used Apple Metal for acoustic inference and MediaPipe
0.10.32 for geometry. The measurement script checks exact model hashes.
Different devices or precision can change neural output near a cutoff.

```sh
backend/.venv/bin/python -B -m scripts.video_mouth_closure_transfer freeze "$run/freeze.json" \
  --parents "$first" "$second" --sources "$sources"
backend/.venv/bin/python -B -m scripts.video_mouth_closure_transfer development "$run/freeze.json" \
  --sources "$sources" --runtime "$runtime" --face "$geometry" --model "$checkpoint" --output "$run/fit.json"
backend/.venv/bin/python -B -m scripts.video_mouth_closure_transfer held "$run/freeze.json" \
  --sources "$sources" --runtime "$runtime" --face "$geometry" --model "$checkpoint" \
  --fit "$run/fit.json" --output "$run/result.json"
```
