# Repeated speech against two mouth sequences in one video

The candidate checked 104 held videos per delivery format and correctly located **3 of 32** known soundtrack edits in each format. It flagged **0 of 72** controls. The existing timing calculation could not measure a window on these 2.52-second deliveries. This narrow check missed most known edits and remains experimental. [Frozen case arithmetic and source commitments](evaluation-panels/video-lipsync-repeat-audio-2026-11-12.json)

## What the test measured

A local soundtrack move can leave the same speech fragment at two positions. This comparison looks for active acoustic similarity, then checks the corresponding original pictures for two moving, different mouth sequences. It measures both source intervals and their distance. A repeat can be intentional and the source of the change remains unknown. The test used conventional soundtrack edits, with no neural mouth redraws. Recent primary research separates timing from articulation, and treats short, perturbed edits as a distinct validation question. [PVSync](https://arxiv.org/abs/2610.09223) · [AV-SyncBench](https://arxiv.org/abs/2607.00726) · [AV-Deepfake1M++](https://arxiv.org/abs/2507.20579)

The source roster uses actor codes outside the project's documented acquisition plans and both later recorded-speech intake rosters. It preorders six neutral phrases for each of 15 CREMA-D actors, with four development actors and eleven held actors. The 90 source files matched the official Git LFS SHA-256 and byte length. Qualification selected the first recording per actor with an uninterrupted common 63-picture and 40,320-sample extent. Four development and eight held actors qualified. The initial three-second screen proved too long. A subsequent 2.52-second freeze preserved the same detector points and gate, and excluded three held actors by media length before any held detector scored. Each prepared picture came from a distinct verified publisher frame within 25 ms of its scheduled time. The submitted files use 25 fps H.264 with either PCM in MOV or AAC in MP4. [Publisher database and content terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Existing commercial-use face localization asset terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Each source supplies two 480 ms soundtrack copies, two 640 ms localized shifts of 320 ms, and nine controls. The controls include unmodified speech, local gain and filtering, a constant offset, repeated pictures, matched picture and sound, a matched moving mouth patch and sound, speech inserted from a different actor, and a repeated quiet background. Each local shift has a known repeat witness where copied and untouched samples coexist. The authored transition uses a 5 ms fade. The audio-only comparator looks for equal decoded samples or similarity in two active frequency bands. The mouth comparison requires at least 240 ms of acoustic support and tests moving, eye-aligned mouth crops with surrounding-picture and pose checks. A follow-up guard suppresses a match if at least two thirds of the observed mouth-pair crops remain similar when measured directly on the native pixel grid over small translations. Both rules were fixed after development and before held scoring. The comparison needs at least half temporal overlap for each of the two ranges. The advancement rule requires 16 correct pairs across seven or more actors, at least 75% measured coverage, at most two ordinary alerts, and zero alerts on synchronized copies or a whole-track offset.

## Held results

The outcomes were identical for PCM and AAC except for the exact-sample comparator. The two carriers share the eight held actors and all authored operations.

| Check | Useful or nominated edit pairs, 32 edits | Ordinary files with a pair or flag, 72 controls | Measured, 104 videos |
| --- | ---: | ---: | ---: |
| Existing timing calculation | 0 | 0 | 0 |
| Two-band audio alone, pairs nominated | 22 | 16 | 104 |
| Eye-aligned mouth check | 4 | 1 | 104 |
| With the native mouth guard | 3 | 0 | 104 |

The guarded result found two copied speech fragments and one local shift across three actors. It found no earlier local shift and produced no wrong pair on an edited clip. One flagged local shift had matching speech at **0.780–1.100 s** and **1.100–1.420 s**, compared with the authored repeat witness at **0.800–1.120 s** and **1.120–1.440 s**. The other two guarded results used **0.870–1.110 s** with **1.910–2.150 s**, and **0.630–1.130 s** with **1.670–2.170 s**. The native mouth guard suppressed the eye-aligned check's synchronized mouth-and-sound false alert and also removed one useful edit pair. Two-band audio alone also nominated 14 synchronized copies and two repeated backgrounds per carrier. Those nominations describe acoustic resemblance. The video measurement needs other evidence to distinguish them reliably from a timing mismatch.

All three actor result files reproduced byte for byte when their source deliveries were rebuilt. Independent FFprobe and FFmpeg readback on 62 original and rebuilt deliveries verified 3,906 native picture timestamps, 2,519,680 decoded samples, 12 exact audio excerpts, and 96 source frame observations. Every guarded positive was checked. The audit reviewed exact source pictures and audio. The mouth matching itself uses the original computation.

## Local reproduction

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.prepare_video_lipsync_repeat_audio freeze "$sources"
backend/.venv/bin/python -B -m scripts.prepare_video_lipsync_repeat_audio acquire "$sources"
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_lipsync_repeat_audio "$phase" "$run" "$sources" || exit 1
done
python3 -B -S -m scripts.review_video_lipsync_repeat_audio "$receipt" --assemble "$run"
python3 -B -S -m scripts.review_video_lipsync_repeat_audio docs/evaluation-panels/video-lipsync-repeat-audio-2026-11-12.json
```
