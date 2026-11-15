# Using vowel and lip-closure shapes to locate short timing shifts

The earlier lip-release check needed enough audible B and P sounds and
visible releases before, during and after a change. This comparison adds
speech sounds that correspond to different mouth openings. It compares a
constant offset with a short change in offset. The model receives one
soundtrack at a time and the observed aperture from the file's own pictures.
Recent research distinguishes the timing of mouth motion from agreement
with the spoken sound. This trial uses the observed mouth opening as its
only shape feature. [PVSync research](https://arxiv.org/abs/2610.09223) ·
[Earlier ordered-release study](video-lip-event-alignment-2026-11-15.md)

## What the check measures

The pinned acoustic checkpoint supplies timestamps and responses for
closures, open vowels, rounded sounds and spread sounds. The video
measurement uses native inner-lip apertures, normalized within each
clip by its 10th-to-90th percentile range. Sixteen untouched fit recordings
from four speakers supply class medians on the original clock. Three
classes qualified. They contributed 37 closure events, 23 rounded
events and 43 spread events. The open-vowel class lacked the required
support. The inference uses the original assets from the previous tests.
[Pinned Apache-2.0 acoustic checkpoint](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)
· [Picture measurements and checked model terms](video-bilabial-mouth-transfer-2026-11-14.md)

Use one 40 ms lag grid from −240 ms to +240 ms and compare windows
lasting 400, 560 and 720 ms. A candidate interval needs at least two
inside events and two outside, one on either side. The alternate lag
must improve the fit inside, and both flanks must independently prefer
the same outer lag. Each compared lag uses the same acoustic events.
The readout compares fit quality over all those observations against
the best constant lag. It withholds near-ties that favor substantially
different locations or opposite directions. It compares closure-only
timing against all supported classes.

Four separate speakers provide ordinary, gain-only, filtered and
constant-delay cases to set the tighter and wider points. The points
are the two highest speaker maxima. The recorded point file precedes
held scoring. Useful localization requires at least 0.40 temporal
intersection over union and a signed shift within 80 ms of the known
edit. The comparison's source windows and lags describe a timing
observation. Such shifts can also arise from ordinary audio editing.

## Primary held comparison

The 128 shifts and 320 controls use the same 64 recordings from 16
held speakers in the earlier test. The new acoustic extraction reused
verified original pictures and decoded native audio. All voice
segments derive from ordinary recorded speech. [Lombard Grid source
and CC BY 4.0 terms](https://zenodo.org/records/3736465)

| Check | Useful local shifts /128 | Incorrect local positions or signs | Ordinary flags /192 | Constant-delay flags /128 |
| --- | ---: | ---: | ---: | ---: |
| Current app timing | 0 | 0 | 0 | 0 |
| Both previous release-based methods | 6 | — | 0 | 0 |
| Closures only, tighter | 0 | 0 | 0 | 0 |
| All supported classes, tighter | 9 | 9 | 2 | 5 |
| All supported classes, wider | 16 | 20 | 27 | 18 |

The nine tighter-level locations are distinct from the six in the
earlier methods. The new check had four or more eligible events on
113 of 128 local edits, compared with four for closures alone. The
wider point offers more detections with more ordinary and constant-delay
alerts. These settings apply only to this experiment.

## Separate speakers and AAC soundtracks

The transfer selects eight further speakers by the fixed publisher ID
order, excluding all 24 speakers from the primary cohorts and the
earlier source exclusions. Each speaker supplies one plain and one
Lombard recording. The 16 recordings each supply two local shifts,
two constant delays and three ordinary comparisons. The fit, decision
points, frame geometry and roughly 24 fps native picture clock carry
over without adjustment. The selection has no exposure to the new
scores. These speakers are new to this comparison. Cue coverage on
unrestricted recordings remains open.

Each of the 112 deliveries is measured as PCM and after a 48 kbps AAC
encode and decode at 16 kHz. The measurement checks audio sample origin
within one sample. The native picture frames remain identical across
both carriers. Both methods read every soundtrack separately.

| Check and carrier | Useful local shifts /32 | Incorrect local positions or signs | Ordinary flags /48 | Constant-delay flags /32 |
| --- | ---: | ---: | ---: | ---: |
| Current app timing, PCM | 0 | 0 | 0 | 0 |
| Current app timing, AAC | 0 | 0 | 0 | 0 |
| All supported classes, tighter PCM | 4 | 1 | 0 | 3 |
| All supported classes, tighter AAC | 3 | 1 | 0 | 1 |
| All supported classes, wider PCM | 6 | 4 | 11 | 7 |
| All supported classes, wider AAC | 5 | 4 | 10 | 5 |

The tighter result preserved three of the four PCM locations after
AAC. The positive cases came from three speakers. For each positive
case the receipt distinguishes the audio range, its projected picture
range, the phoneme-matched picture anchors and the native before,
during and after frames. The two scopes can differ because each stream
has its own time coordinates. Source replay verifies both original
frame and mouth-crop pixels at the recorded presentation timestamp.
One AAC case placed the sound at **0.680–1.400 s** and the corresponding
picture interval at **0.520–1.240 s**. Its three surrounding source frames
are at **0.500, 0.875, and 1.250 s**. The measured delay change was
160 ms against the known 200 ms.

These results support a local trial using the tighter point with
source evidence and zero authorship-confidence weight. Current evidence
measures short shifts in speech with known faces. Synchronized
redrawing, offscreen speakers, cuts, heavy occlusion and arbitrary
video clocks remain outside this transfer. This receipt documents a
research runner and no app setting is enabled by it.

## Independent replay

The two compressed receipts store source clocks, aperture values,
recognized phone times, before/during/after frame commitments, and
counts. They include no audio or picture bytes. The portable reader
recomputes calibration and both transfer comparisons without a model.
Per-event score arithmetic sums integer units at seven decimal places.
Python 3.11 and 3.12 produce the same decisions and numeric receipts.
Two local transfer runs reproduced the timing inputs and counts.
The separate source audit verified 112 PCM and 112 AAC waveforms,
920 picture timestamps, and 31 source frames with their exact mouth crops.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panel=docs/evaluation-panels/video-lip-viseme-transfer-2026-11-15.json.gz
holdout=docs/evaluation-panels/video-lip-viseme-carrier-2026-11-15.json.gz
module=scripts.video_lip_viseme_transfer
python3 -B -S -m "$module" review "$panel"
python3 -B -S -m "$module" review-transfer "$holdout" "$panel"
```

Use the publisher archives and pinned assets from the
[parent native replay](video-bilabial-mouth-transfer-2026-11-14.md#replay).
The source audit rebuilds the PCM, AAC sample clocks and native witness
pictures without running the phone recognizer. The transfer model
runner uses the primary receipt as its sealed comparison.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" capture "$archives" "$checkpoint" "$run/acoustic.json"
python3 -B -S -m "$module" measure "$run/acoustic.json" "$run/primary.json"
"$runner" -B -m "$module" transfer "$panel" "$archives" "$checkpoint" \
  "$face_landmarker" "$mediapipe_runtime" "$run/new-transfer.json"
"$runner" -B -m "$module" audit-source "$holdout" "$panel" "$archives"
```
