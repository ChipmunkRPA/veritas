# Local face-swap and lip-sync follow-up

The follow-up separates face swaps, soundtrack timing shifts, and AI mouth
redrawing. The last two can occur independently. All source videos, candidate
weights, and raw measurements stay in ignored local storage under
`backend/data/independent-evaluation/video-v3/`.

## Face swaps

The [RTFS-10K release](https://huggingface.co/datasets/godmodes/rtfs-10k/tree/bca865e675852666a426368b2f5614abf1c1926a)
identifies paired authentic footage and two different swap generators. The
source plan excludes all 24 original YouTube IDs and all donors from the
previous evaluation. It selects 24 new groups with one original, one InSwapper
clip, and one UniFace clip each. All 72 planned clips were acquired. Identical
H.264 encoding and duration caps reduce codec shortcuts. The fit uses only
the original study's 12 calibration groups. Prior training overlap is unknown.
The new manifest SHA-256 is
`b732512832ea90eb36db1c7255c2e6c05a78bf5d4b7f8fb1d78ccf87f32bf8b5`.

The comparison includes the current frame ensemble, the disabled GenD
track candidate, and a pinned [EfficientNet-B2 candidate](https://huggingface.co/ThinothW/DFDS-EfficientNet-B2-Deepfake-Detection-Engine/tree/d7887b76ac0a12ef6a5109515cc1ee435caf8804).
The standalone runner verifies the last candidate's weight digest, loads a
strict state dict with `weights_only=True`, and uses the published resolution,
normalization, crop padding, and class order. It substitutes the project's
face detector for the publisher's face proposal and samples 16 fixed source
frames. The checkpoint page declares Apache 2.0. The [training dataset card](https://huggingface.co/datasets/ThinothW/Deepfake-Identity-Isolated-Dataset-PreP#license)
limits the dataset to academic and research use and refers to its upstream
licenses. The checkpoint was evaluated offline, with its commercial-use
provenance unresolved.

| Frozen fit on the 72 fresh clips | Raw AUC | Brier | Correct AI calls, 48 swaps | Incorrect original calls, 48 swaps | False AI calls, 24 originals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current face check | 0.624 | 0.243 | 1 | 0 | 0 |
| GenD track candidate | 0.705 | 0.223 | 0 | 1 | 0 |
| EfficientNet-B2 candidate | 0.581 | 0.250 | 0 | 0 | 0 |

The GenD ranking gain appeared again, but its paired Brier change of -0.020
had a 95% source-group interval of -0.051 to +0.011. It produced no correct
swap call at 80% and one wrong original tendency. The B2 fit flattened to
approximately 50% because the old calibration subset did not show a positive
ordering. Neither replaced the current face check. These calibrated counts
are for the offline comparison on the face lane. The whole-video conclusion
uses other evidence and has different decision rules.

The separate localization spot check used one consumed test source with
matched 30 fps inputs. Three recoded versions inserted the original, InSwapper
or UniFace frames between frames 60 and 119, covering 2.000 to 4.000 s. The
current check's highest frame response landed inside the edit on the UniFace
copy at 3.600 s and just after the interval on the InSwapper copy at 4.133 s.
The GenD peak was inside the InSwapper edit at 2.900 s and outside the UniFace
edit at 0.100 s. The genuine matched control also returned a high raw response
at 4.133 s. Each method localized one of two edits in this example. A peak
provides an inspection target without proving an edited interval.

## Audio and picture timing

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D/tree/1658cd342dff90010aa843eaeebd53610a08b1dc)
licenses the database under ODbL and the contents under DbCL. The test froze
one neutral speech clip per actor, with ten actors for development and 18 for
held-out test. Each source produced an untouched timing control and copies
with audio 320 ms earlier and 320 ms later. Independent decoding verified
identical picture frames and timestamps within each group, and exactly 5,120
shifted 16 kHz PCM samples across the common overlap. The tests measure the
known added offset relative to the recorded source timing, which can itself
contain an offset. The manifest SHA-256 is
`55936efe0be93b3fbd80dc3ac824158273ab869c98883b430686483d12f74b40`.

The trial candidate compares inner-lip geometry from the pinned [face landmark bundle](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
with the sound level and its change over time. The runtime was installed into
the trial folder to avoid replacing the app's OpenCV dependency. The frozen
rule uses the change in sound level, requires a peak of at least 0.55 and a
margin above plausible competing lags, and reports a shift only from 160 ms.
An added shift is correctly recovered within two 40 ms ticks of its known
direction and size. The active baseline uses a stable track offset; weak or
missing measurements abstain.

| Held-out timing result | Current | Candidate |
| --- | ---: | ---: |
| Correct recovered shifts among 36 edited copies | 0 | 4 |
| Incorrect reported shifts among 36 edited copies | 0 | 2 |
| Flags among 18 untouched timing controls | 0 | 1 |
| Clips with a measured window among 54 | 20 | 54 |

The two filmed [LatentSync demo pairs](https://github.com/bytedance/LatentSync/tree/a229c3948406bc2cf6eaf4873e662e70c6a04746)
are a separate small transfer check for AI mouth redrawing. No ground-truth
audio offset or frame edit mask is provided. The current check flagged one
redrawn video and could not measure the other redrawn clip or either original.
The candidate flagged both originals and both redrawn clips. Its extra coverage
did not justify a replacement. The current check remains an inspection clue for
timing changes, and timing agreement cannot identify a redrawn mouth.

## Evidence replay

The ordinary video evidence view opens a recorded face peak when one has a
verified face context region. The timing view links each displayed flagged
source interval to its frame crops, original audio, and lag plot. It describes
weak correlations and distinguishes the measured window from the timing
change inside it.

Two illustrative cases ran through the normal local worker, using a simulated
test Stripe transport and the relevant commercial-eligible checks. No job
existed before the paid webhook. Both signed PDFs, JSON exports, and offline
evidence packages matched their source records, and the customer artifacts
passed the implementation-name scan. The full swap report shows the frame
peak at 5.166667 s and neighbors at 4.900000 and 5.433333 s. Its score is
uncalibrated. The timing example records a step from 6.000 to 7.480 s. Packet
AV02 covers 6.000 to 7.000 s and selects a lag of -320 ms. It shows source
frames at 6.000, 6.480, and 6.960 s and the matching waveform.

## Reproduction

Run from `backend` using fresh receipt paths. `trial/av/mesh-runtime` below
refers to the ignored local target directory where `mediapipe==0.10.32` was
installed with `--no-deps`. The timing runner verifies the official
`face_landmarker/float16/1` asset. The separate face runner verifies its exact
checkpoint before loading it. The evidence replays use
`validate_local_audio_rollout.py --mode candidate --cases video --video-evidence-only`
with a selected fixture and simulated test checkout.

```sh
prior=data/independent-evaluation/video-v2
trial=data/independent-evaluation/video-v3
.venv/bin/python scripts/prepare_face_swap_upgrade_eval.py "$trial/face-swap" \
  --groups 24 --exclude-manifest "$prior/face-swap/manifest.json"
PYTHONPATH=. VERITAS_DEVICE=mps .venv/bin/python scripts/run_face_swap_upgrade_eval.py \
  "$trial/face-swap/manifest.json" "$trial/face-swap/scores-mps.jsonl"
.venv/bin/python scripts/prepare_video_timing_upgrade.py "$trial/av"
PYTHONPATH=. VERITAS_DEVICE=mps .venv/bin/python scripts/run_video_av_controls.py \
  "$trial/av/manifest.json" "$trial/av/current.jsonl" --timing-only
PYTHONPATH=".:$trial/av/mesh-runtime" .venv/bin/python scripts/run_video_timing_candidate.py \
  "$trial/av/manifest.json" "$trial/av/face_landmarker.task" \
  "$trial/av/mesh-development-cpu.jsonl" --split development
PYTHONPATH=. .venv/bin/python scripts/run_dfds_face_eval.py \
  "$prior/face-swap/manifest.json" "$trial/dfds/pytorch_model.bin" "$trial/dfds/old.jsonl" --device mps
PYTHONPATH=. .venv/bin/python scripts/run_dfds_face_eval.py \
  "$trial/face-swap/manifest.json" "$trial/dfds/pytorch_model.bin" "$trial/dfds/fresh.jsonl" --device mps
.venv/bin/python scripts/summarize_video_forensic_followup.py "$prior" "$trial" "$trial/summary.json"
```

The timing summary also uses `mesh-test.jsonl` and `mesh-publisher.jsonl`.
Generate them with the timing runner's `test` and `publisher-example` splits.
The frozen rule in `mesh-rule.json` uses a 12-tick search in each direction,
a 4-tick minimum offset, a 0.55 minimum peak, a margin of 0.18 over the best
near-zero offset and a margin of 0.10 over peaks outside the best lag's
three-tick neighborhood. Recovery tolerance is two ticks. The file commits to
the development score SHA-256 before held-out scoring. The combined summary
SHA-256 is `39572d558126c1e6c90cd11460f69880e881fcbd418ae1b90465449bc6ccecd5`.
