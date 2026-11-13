# A new source bank for synchronized mouth-redraw tests

We qualified 17 genuine recordings with independently observed acoustic and visible mouth-release brackets. A frozen selection schedules 13 recordings for later redraws, including 11 from all six held speaker groups. These are original recordings ready for a rendering trial. The study supplies no newly rendered AI examples or detector accuracy results. [Public source qualification and replay](evaluation-panels/lombard-original-mouth-event-2026-11-12.json)

## Source and rights

The University of Sheffield's Lombard corpus contains a separately recorded group of 54 speakers. Its prompted sentences are unique relative to original GRID. It includes a matched plain and Lombard style for the same speakers. The release record declares CC BY 4.0. The paper describes a simultaneous camera and microphone setup and subsequent alignment of the separately released audio and video. This study uses the publisher's paired frontal MOV and external WAV. It uses anonymous publication codes as source groups and makes no cross-corpus identity claim. [Official publisher record](https://zenodo.org/records/3736465) · [Original recording and alignment methods](https://eprints.whiterose.ac.uk/id/eprint/131924/1/jasa-el-slash-8.pdf)

The bounded intake verified the publisher MD5 values for both metadata ZIPs, then fetched only the selected video and audio archive members. Each member matched the frozen ZIP size and CRC32 and was recorded by SHA-256. The 64 source files use 9.9 MiB. The full media archives were unnecessary. The initial download exposed a ZIP member-reader interface error. A subsequent attempt reached the publisher rate limit and resumed using the same roster and CRC-verified local files. No recording was substituted after looking at its signals.

## Independent observations

Before observing events, the roster assigned two source groups to development and six to held use. Each speaker contributed two plain and two Lombard utterances. The fixed target was the earliest eligible internal /b/ or /p/ with enough surrounding time in the publisher's forced alignment. That metadata provides a search neighborhood. A waveform-only pass required a distinct acoustic transient with subsequent speech. A separate muted-picture pass required near closure followed by a clear reopening. Their timestamps use the released source clocks. A qualifying case needed at most 80 ms between bracket midpoints, both brackets inside a proposed 16-picture interval, and genuine source pictures on both sides.

| Source qualification | Count |
| --- | ---: |
| Planned genuine recordings | 32 |
| Distinct acoustic bursts | 23 |
| Unique visible reopenings | 26 |
| Both observations available | 19 |
| Qualified original pairs | 17 |
| Scheduled sources after the one-per-style, per-speaker rule | 13 |

For one held recording, the 15 ms acoustic bracket is **0.925–0.940 s** and the visible bracket spans approximately **0.918–0.959 s**. Independent readback matched all **1,924** source picture timestamps, **1,263,200** original float PCM samples, **23** burst excerpts and **52** original mouth crops. The front clips are H.264 and preserve their individual rational cadence near 24 fps. The separate mono WAVs use 16 kHz float32. The observed source events support a local time correspondence. The publisher labels are forced alignments and no independent human performed phoneme or listener judgment on this cohort.

## What makes a valid neural counterpart

A rendering trial should reuse the existing rights-reviewed expression renderer or a separately reviewed audio-conditioned pipeline. The stock MuseTalk repository permits commercial use of its trained model and explicitly restricts its demonstration media to noncommercial research. The prior isolated project path already checked its dependencies and uses locally admitted geometry. The recent DeepSpeak data includes consenting subjects and modern neural redraws, but commercial access requires a separate application and grant. LipSync-A, LipSyncTIMIT and AV-Deepfake1M++ have research or noncommercial release terms. Their accuracy results cannot substitute for a local trial. [Prior expression asset audit](grid-reserve-neural-mouth-2026-11-12.md) · [MuseTalk publisher terms](https://github.com/TMElyralab/MuseTalk#disclaimerlicense) · [DeepSpeak v2 terms](https://huggingface.co/datasets/faridlab/deepspeak_v2) · [LipSync-A terms](https://huggingface.co/datasets/AnsonShe/LipSync-A) · [LipSyncTIMIT terms](https://github.com/skrantidatta/LIPINC-V2) · [AV-Deepfake1M++ terms](https://arxiv.org/abs/2507.20579)

For each rendered and conventional control clip, the next native check should reopen the actual output, verify the exact rational picture clock and half-open changed-pixel interval, and compare the complete source PCM. The muted-picture observer should mark its event again without reading the rendered condition or the acoustic measurement. A redraw can enter the synchronized-event subset only if its observed reopening independently corresponds to the original burst and stays within one native picture of the genuine release. Include matched mouth resampling, blur and a shifted-picture control using the same compositing geometry. This tests local event timing and still needs broader listener or phone-level validation. No renderer loaded during this source qualification.

## Local replay

Place the [publisher record JSON](https://zenodo.org/api/records/3736465), its alignment ZIP and its metadata ZIP in the private run directory as `record.json`, `alignment.zip` and `publisher-json.zip`. The native audit requires FFmpeg. The muted visual observer uses the existing reviewed local Face Landmarker task and its isolated runtime. Keep source media and per-source identities local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze acquire; do
  backend/.venv/bin/python -B -m scripts.lombard_mouth_source_intake "$phase" "$run" || exit 1
done
for phase in freeze audio visual compare; do
  backend/.venv/bin/python -B -m scripts.qualify_lombard_mouth_events "$phase" "$run" || exit 1
done
python3 -B -S -m scripts.review_lombard_mouth_events "$receipt" --assemble "$run"
python3 -B -S -m scripts.review_lombard_mouth_events docs/evaluation-panels/lombard-original-mouth-event-2026-11-12.json
```
