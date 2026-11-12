# Mouth and speech-content agreement on matched local dubs

The paired mouth and speech-spectrum check measured all 60 held views. It correctly localized zero of 12 authored speech replacements and proposed four review ranges on the 48 controls. Both positives it flagged fell below the interval overlap requirement. The current timing check had no eligible measurement on this short-clip panel. The new check remains experimental. [Aggregate and source replay](evaluation-panels/video-mouth-content-transfer-2026-11-11.json)

## Method

Recent work separates audio and picture delay from whether a mouth appears to articulate the speech being heard. Two October 2026 papers report models that compare representations at the articulation or frame level. This trial tests an independently implemented spectral agreement cue using the existing commercially licensed face detector. [PVSync](https://arxiv.org/abs/2610.09223) · [Lip-focused audiovisual consistency](https://arxiv.org/abs/2610.30872) · [Face weight license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The official pinned CREMA-D release supplied 72 native recordings from 24 known actors. Each actor supplied a neutral recording of one sentence, a separate expressive recording of the same sentence, and a neutral recording of a different sentence. The publisher's recording names identify the scripted utterances. Every recording matched its Git LFS pointer and byte length. The roster excluded all 15 actors in the preceding local-delay study. The same speaker occurs only within a source group. Eight groups fit the relation, four different groups set the candidate points, and 12 held groups set neither. Some CREMA-D speakers or source material may have been used in other project experiments. [Publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Prior delay test](video-lipsync-local-delay-transfer-2026-11-11.md)

Each held recording supplied one unchanged view, one local gain reduction, one mild local low-pass filter, one same-words re-recording and one different-words replacement. All four interventions used the same 24-tick, 0.96-second half-open interval and preserved the target's pictures and every PCM sample outside the interval. The two dubs use same-speaker donor audio stretched to the target duration and matched approximately to the original 40 ms RMS envelope, with a 10 ms mix at each edge. The median of the held recordings' median absolute active RMS errors was 0.644 dB for same words and 0.624 dB for different words. The original and alternate whole-utterance identities are known. The trial does not independently align each substituted phoneme. A same-words re-recording can differ in articulation, timing and emotion.

The paired arm fits a ridge regression from the current and adjacent mouth patches to the relative energy in ten speech-frequency bands. Each acoustic frame removes its common log energy. At inference it compares the prediction residual inside a range against both measured flanks. The audio-only comparison uses deviation from the learned spectral average over the same sample positions. Both arms use 0.48, 0.80 and 1.12-second candidate windows, the same valid mouth samples and the same audio activity gate. Each arm keeps its largest two-sided contrast. Its internal point is the larger of 0.025 and the largest measured ordinary calibration response plus 0.025. The existing timing comparator uses the production aperture and lag calculation on the study's directly observed single face. A correctly located range needs at least 0.50 temporal intersection-over-union. The frozen advancement rule requires at least six correct ranges, at least three more than either comparator, and at most two added ordinary ranges relative to the current check.

Two preliminary runs stopped in development. The first planned edit and flanks exceeded one calibration recording's native duration. A source-clock preflight established a fixed shorter edit. The second had too few measured control windows to seal a calibration point. A separate freeze lowered the readout RMS gate and allowed fewer measured positions per flank while keeping identical support for both new arms. All held detector scores were opened only after that final fit and calibration were sealed.

## Held results

| Check | Measured views | Correct different-words intervals | Wrong positive intervals | Ordinary review ranges |
| --- | ---: | ---: | ---: | ---: |
| Current one-second timing | 0/60 | 0/12 | 0/12 | 0/48 |
| Audio-only spectrum | 60/60 | 0/12 | 0/12 | 2/48 |
| Paired mouth and spectrum | 60/60 | 0/12 | 2/12 | 4/48 |

The paired arm flagged 0.960–1.440 s for an authored edit at 0.400–1.360 s and 1.040–1.520 s for an edit at 0.480–1.440 s. Both overlaps were 5/13. The paired arm's four ordinary ranges came from one low-pass filter and three same-words re-recordings. Both audio-only control ranges came from same-words re-recordings. Neither arm flagged an untouched recording or gain control.

The run committed 647 native source pictures. A separate reviewer checked the 72 licensed file hashes, recomputed the held decision arithmetic, rebuilt all 60 exact PCM inputs with the frozen authoring method, and verified 360 source-audio slices. It decoded 132 original pictures and matched all 132 measured mouth crops at their recorded presentation timestamps. The retained files contain source identities, local excerpts and per-case records. The public receipt contains aggregate measurements and anonymous range examples.

The trial tests short fixed-mouth dubbing on two recorded sentences. Individual phoneme recognition, neural retalking, spontaneous speech and additional languages remain untested. The candidate stays outside the serving confidence and findings.

## Reproduce

The intake module requires the official pinned publisher tree and the retained earlier study intake freeze. The runner and reviewer use a fresh working folder under local storage.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
backend/.venv/bin/python -B -m scripts.prepare_video_mouth_content_transfer freeze "$run_dir" \
  --tree "$publisher_tree" --prior-freeze "$earlier_intake_freeze"
backend/.venv/bin/python -B -m scripts.prepare_video_mouth_content_transfer acquire "$run_dir"
for phase in freeze fit calibration calibrate held readout; do
  backend/.venv/bin/python -B -m scripts.video_mouth_content_transfer "$phase" "$run_dir" || exit
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_content_transfer "$run_dir"
```
