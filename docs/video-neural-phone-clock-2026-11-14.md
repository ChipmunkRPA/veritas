# Qualifying speech timestamps for short lip-sync checks

A short lip-sync check needs an accurate time for the audible speech sound
before it can compare that sound with the visible mouth. The neural
speech check found more publisher-labeled /b/ and /p/ sounds than the
earlier decoder. It also reported more sounds without a matching label.
Reading overlapping audio windows yielded the same held original and
shift counts as reading the full track. The candidate stays in local
evaluation.

## Source and method

The Grid Audio-Visual Lombard corpus supplies matched frontal video,
16 kHz mono audio and publisher phone alignments under CC BY 4.0.
We ran two separate source cohorts. Each cohort has four development
and eight held speakers with two plain utterances and two Lombard
utterances per person. We excluded known speakers from earlier project
Lombard studies and excluded all 12 first-cohort speakers from the
second cohort. Both publisher media archives passed their original
size and checksum verification. Each selected audio, video and
alignment member carries its own SHA-256. The automated publisher
alignments supply comparison intervals and have their own timing error.
[Dataset and license](https://zenodo.org/records/3736465)

The evaluation compares the previously reviewed PocketSphinx phone
decoder with a pinned Apache-2.0 neural phone checkpoint. It uses
the same 16-bit representation of every waveform on the original
16 kHz clock. The neural reading keeps an unprompted token emission
using the verified 20 ms output hop. It receives no words or labels.
One comparison requires a repeat reading within 40 ms after low-pass
filtering. The separate local-context comparison requires agreement
across two inputs, using the full track and overlapping 1.2-second
crops. The focus on bilabial sounds follows visual speech work on
lip closure. Timing and mouth articulation remain distinct measurements.
[Pinned checkpoint and license](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)
· [Earlier decoder license](https://github.com/cmusphinx/pocketsphinx/blob/v5.0.4/LICENSE)
· [Phoneme–viseme study](https://openaccess.thecvf.com/content_CVPRW_2020/html/w39/Agarwal_Detecting_Deep-Fake_Videos_From_Phoneme-Viseme_Mismatches_CVPRW_2020_paper.html)
· [Timing and articulation research](https://arxiv.org/abs/2610.09223)

Each /b/ or /p/ timestamp can match one published interval with up to
40 ms padding. Unmatched means the publisher supplied no eligible
interval for that reading. Each staged delivery shifts the target
sound by 200 ms in one direction within a 700 ms window. A tracked
shift requires a match in the original and edited versions and a
change in measured time within 60 ms of the known displacement.

## Held results

| Held cohort and check | Original label matches | Unmatched original readings | Target matches in shifted clips | Correctly tracked shifts |
| --- | ---: | ---: | ---: | ---: |
| First, earlier decoder | 13/64 | 16 | 23/64 | 12/64 |
| First, full-track neural reading | 30/64 | 33 | 24/64 | 24/64 |
| Second, earlier decoder | 16/59 | 23 | 21/64 | 16/64 |
| Second, full-track neural reading | 21/59 | 38 | 19/64 | 18/64 |
| Second, overlapping neural readings | 21/59 | 38 | 19/64 | 18/64 |

The earlier decoder found more eligible shifted targets on the second
cohort. More of the neural matches tracked the 200 ms displacement
from their own original reading, but the absolute coverage remains
low. The overlapping comparison provides no observed improvement on
these clips. The primary assessment here is the acoustic time anchor.
Both cohorts use short scripted English speech with a visible speaker.
They do not establish reliable lip-sync detection in general clips.

## Independent replay

The two numeric receipts store timing measurements, source hashes,
authored waveform digests and the operating rules. The second also
records the individual window evidence. Source media and model weights
stay local. Standard-library Python recounts the matches, unmatched
reads, shift tracking and second-cohort window agreement.

```sh
export PYTHONPATH=backend
first=docs/evaluation-panels/video-neural-phone-clock-2026-11-14.json
second=docs/evaluation-panels/video-neural-phone-context-2026-11-14.json
python3 -B -S -m scripts.video_neural_phone_clock review "$first"
python3 -B -S -m scripts.video_neural_phone_clock review "$second" --previous "$first"
```

The [first native replay](evaluation-panels/video-neural-phone-clock-source-replay-2026-11-14.json)
and [second native replay](evaluation-panels/video-neural-phone-context-source-replay-2026-11-14.json)
verify the original labels, recreate the staged waveforms, reopen the
original picture clock and bind neighboring source frames by their
decoded RGB hashes. These frames supply source context for a measured
audio event. The replay measures no mouth deformation or authorship.

For a local rerun obtain `lombardgrid_audio.zip`,
`lombardgrid_front.zip`, the alignments as `alignment.zip`, the publisher
JSON archive as `publisher-json.zip`, and record metadata as
`record.json` from Zenodo record 3736465. Run `freeze --sources` with
the directory and add `--previous "$first"` for the second cohort.
Run `measure` using that roster, the same source directory, the pinned
checkpoint directory as `--model`, and the verified 5.0.4 macOS wheel
as `--wheel`. Use a separate `--output` directory per cohort. The measured
run used Apple Metal. With both source archives available, run
`replay --sources` on a receipt to verify the native source bindings.
