# Lip and speech timing on 48 kHz AAC and native 29.97 fps video

The local Full analysis can use its speech-to-mouth check on a verified
48 kHz mono AAC soundtrack. The soundtrack must preserve one continuous
clock with a zero start and a verifiable presentation length. The check
reads the 16 kHz analysis samples within that length. The range, original
pictures, mouth crops, waveform and audio excerpt share the original
presentation clock. The measurement carries zero weight in AI confidence.
[Prior local speech-to-mouth comparison and asset terms](video-lip-articulation-2026-11-15.md)

The comparison looks at short speech-to-mouth disagreement and native
picture clocks, two areas explored in recent audiovisual localization
research. It reuses the app's pinned assets and fitted rules.
[Speech and mouth alignment study](https://arxiv.org/html/2610.30872v1)
· [Frame-rate study](https://arxiv.org/html/2604.16808v5)

## 48 kHz AAC on the retained speech replacement panel

The 34 held Lombard Grid recordings come from 17 actors who did not fit
the articulation check or its operating point. Reuse the native pictures
and exact same-speaker audio edits from the earlier PCM and 16 kHz AAC
comparison. Encode each soundtrack as 48 kHz mono AAC at 96 kbps. The
production decoder recovered the authored sample count on all 204
deliveries. The audio starts matched within one analysis sample. The test
changes the delivery of these clips. Their speakers, audio source and
known edit ranges are shared with the earlier held test.
[Lombard Grid sources and CC BY 4.0 terms](https://zenodo.org/records/3736465)

| Check at the existing points | Useful ranges /34 | Misplaced ranges /34 | Ordinary flags /68 | Same-word splice flags /34 | Uniform-delay flags /68 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic timing | 0 | 1 | 2 | 1 | 2 |
| Basic timing plus relative mouth opening | 0 | 1 | 2 | 1 | 2 |
| Basic timing plus speech-to-mouth matching | 6 | 3 | 3 | 3 | 3 |
| All three checks | 6 | 3 | 3 | 3 | 3 |

A useful range must reach 0.40 temporal intersection over union with the
known edit. The six useful ranges avoid the same-word alerts on their
matched control deliveries. The adopted fallback also reduced unresolved
edited deliveries from 12 to six. The relative mouth-opening fallback
added no witnesses in this carrier. Soundtrack substitutions, dubbing and
ordinary timing differences can produce local speech-to-mouth disagreement.
The observed gain applies to these controlled replacements.
[48 kHz measurements](evaluation-panels/video-lip-delivery-aac48-2026-11-16.json.gz)

The checkout replay tested the same edited file with the extension off
and on, plus the unchanged and same-word controls. The trial measured
audio at **0.640–1.040 s** and picture at **0.680–1.080 s**. The browser
loaded all three original mouth crops and the 0.400-second sound excerpt.
The Standard PDF and JSON match those times and the signed evidence
replays the exact source positions. Four simulated USD 9.99 purchases
queued zero new analyses before payment. The whole-file assessments
agreed on the two runs of the edited file.
[Full local replay receipt](evaluation-panels/video-lip-delivery-aac48-app-2026-11-16.json)

## Separate 29.97 fps camera comparison

The 29.97 fps files are acted speech from a different source collection.
The original source videos have a millisecond picture clock. The retained
24 actors are from the previous frame-clock study. The 21 additional actors
did not contribute to that study or to the earlier Lombard Grid fit. Each
has two staged timing edits, an untouched recording, gain and filter
controls, and two uniform whole-track delays. We measured every original
frame at its native timestamp and anchored decoded audio to the audited
start. Temporal overlap uses the same 0.40 target. A separate count
requires the timing displacement to have the right direction and be
within 80 ms of the introduced shift. A speech-to-mouth mismatch does
not by itself estimate a displacement.
[Prior source audit](video-lip-frame-clock-2026-11-15.md)
· [CREMA-D source terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

| Additional 21 actors | Useful ranges /42 | Useful with correct shift /42 | Ordinary flags /63 | Uniform-delay flags /42 |
| --- | ---: | ---: | ---: | ---: |
| Basic timing | 3 | 2 | 0 | 0 |
| Basic timing plus relative mouth opening | 5 | 4 | 0 | 1 |
| Basic timing plus speech-to-mouth matching | 8 | 2 | 8 | 7 |
| All three checks | 8 | 4 | 8 | 8 |

On the earlier 24 actors, relative mouth opening found six of 48 local
edits versus two with basic timing. It added five ordinary flags out of
72 and one uniform-delay flag out of 48. The speech-to-mouth fallback
found 11 of those edits along with ten ordinary and eight uniform-delay
flags. The difference between cohorts and the ordinary controls shows
why the speech-to-mouth extension uses its tested 24–25 fps picture
profile. [Native picture-clock measurements](evaluation-panels/video-lip-delivery-native-2026-11-16.json.gz)

### MP4 replay at 29.97 fps

We encoded lossless H.264 pictures, verified all decoded RGB frames
against the publisher files, and checked that each picture kept the
exact source timestamp. We then muxed seven mono 16 kHz AAC soundtracks
per actor. The app's decoder checked the continuous audio clock on all
315 MP4 deliveries. Clip length, source selection and the timing edits
match the native comparison. The 21 additional actors below are the
same 21 actors in both carriers.

| Additional 21 actors, MP4 | Useful ranges /42 | Useful with correct shift /42 | Ordinary flags /63 | Uniform-delay flags /42 |
| --- | ---: | ---: | ---: | ---: |
| Basic timing | 3 | 2 | 0 | 0 |
| Basic timing plus relative mouth opening | 3 | 2 | 1 | 0 |
| Basic timing plus speech-to-mouth matching | 8 | 2 | 8 | 7 |
| All three checks | 8 | 2 | 9 | 7 |

On the earlier 24 actors, the MP4 replay found one useful range with
basic timing and five with relative mouth opening. Ordinary flags rose
from 2/72 to 7/72. Speech-to-mouth matching raised useful ranges from
one to 12, with 12/72 ordinary and 8/48 uniform-delay alerts. The
relative mouth-opening gain on the 21 additional actors did not carry
over from native audio to AAC. The app keeps its existing picture
cadence limits. [MP4 measurements](evaluation-panels/video-lip-delivery-native-mp4-2026-11-16.json.gz)

## Independent replay

For the local 48 kHz option, set `VERITAS_VIDEO_LIP_AAC48_ARTICULATION_REVIEW=true`
with the existing local lip timing and articulation options enabled. The
new option defaults off. The same source clock checks apply to analysis,
evidence construction and signed verification.

The portable reader verifies the source clocks, exact frame brackets,
measured intervals and dispositions, then recounts every group. It uses
the standard library. To recreate the 48 kHz result, use the previous
study's pinned private plan and held picture capture. The source archives
and assets are described in that study. The native run needs the audited
publisher originals and the local landmark runtime. The receipts contain
source commitments and measured intervals. Faces and audio stay local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.video_lip_delivery_transfer
panels=docs/evaluation-panels
runner=backend/.venv/bin/python
python3 -B -S -m "$module" review "$panels/video-lip-delivery-aac48-2026-11-16.json.gz"
python3 -B -S -m "$module" review "$panels/video-lip-delivery-native-2026-11-16.json.gz"
python3 -B -S -m "$module" review "$panels/video-lip-delivery-native-mp4-2026-11-16.json.gz"
"$runner" -B -m "$module" aac48 "$plan" "$held_picture_capture" "$archives/lombardgrid_audio.zip" "$assets" "$run/aac48.json.gz"
"$runner" -B -m "$module" native "$audit" "$publisher_media" "$assets" "$run/native.json.gz"
"$runner" -B -m "$module" native "$audit" "$publisher_media" "$assets" "$run/native-mp4.json.gz" --repack-aac16
"$runner" -B -m scripts.video_lip_articulation_app "$run/app48" "$archives" "$plan" "$run/aac48.json.gz" "$assets" \
  --aac48 --standard-checks --browser
```
