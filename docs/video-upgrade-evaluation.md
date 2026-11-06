# Local video detector comparison

The comparison separates fully generated video, face swaps and lip-sync edits.
Each needs its own labels and controls. All trial scores and input media are
kept in ignored local storage under `backend/data/independent-evaluation/video-v2/`.
The trial scripts use isolated analysis storage.

## Fully generated video

[DeepAction](https://huggingface.co/datasets/faridlab/deepaction_v1/blob/476932c3369424ca59dc1f8fb758cb3c13fe63fe/README.md)
matches camera and generated footage by action. The publisher identifies real
videos as Pexels material and licenses its generated videos under CC BY 4.0.
The acquisition script picks 36 action groups by a fixed hash order and excludes
the four actions used in the earlier web sample. Twelve action groups are for
calibration and 24 for test. The test generator families are disjoint from the
calibration families. DeepAction also contributed to the release video
benchmark and its fitted combiner. The release summary records 70 DeepAction
items in the scored scope. The checked-in records do not identify those exact
clips, so this cohort is a matched engineering comparison rather than a fresh
test of the release combiner. Exact pretrained overlap of the component
detectors is also unestablished.

The manifest contains 70 available native videos and two unavailable files
from action 97. Both missing files returned 404 and remain in the test
denominator. Each available video also has a copy encoded with the same H.264
settings, with audio and metadata removed, an at-most-six-second window and
bounded resolution. This reduces container and encoding differences while
retaining source aspect ratio and frame rate. The original and encoded copy
of each clip stay in the same action group.

Manifest SHA-256
`3b5688815c31dab4111e00c33cc3de7f6928ca0576bbf321a52db8bba8a08030`.
The test covers the selected action and generator families. It provides
limited evidence about more recent generators, face edits or voice swaps.

### Candidate and comparison

The [AEGIS checkpoint](https://huggingface.co/MusapYildiz/aegis-video-detector/tree/95b71346cec650165e6ad3fb20ed9e80f4b6702a)
and [inference code](https://github.com/MusapYildiz/ai_video_detection_benchmark/tree/d86a774fd971954a023e1cd00ed7ff5b2575e0d1)
carry MIT declarations. The fitted checkpoint includes its DINOv2 encoder.
Meta publishes [DINOv2 under Apache 2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).
The [publisher documents](https://github.com/MusapYildiz/ai_video_detection_benchmark#veri-seti-faz-2)
some training sources with separate terms, including GenVidBench, and reports
undocumented provenance for some locally collected training material. The
checkpoint declaration and training-source terms are recorded separately.

The offline evaluator checks the weights and all five inference source files
against fixed SHA-256 digests before importing the reviewed implementation.
It constructs the encoder architecture locally and loads the complete model
with strict state-dict matching and `weights_only=True`. The OpenCV path uses
the midpoint four-second sampling window. The publisher's original CLI uses
a random start, so the deterministic comparison records this difference.

The first full pipeline CPU cases completed with the installed calibration.
The benchmark stopped after the same real source's native and matched copies
because one CPU case took about five minutes. The matched comparison instead
uses both current synthesis adapters with their normal production sampling
and the mean of whichever raw responses are available. Some encoded copies
fall below an existing native-crop size requirement. Individual applicability
and all-source coverage are reported separately. That evaluates the
synthesis lane. The whole-file result also uses face and container checks,
along with scope and provenance rules.

The summary reports paired discrimination, Brier and log loss, plus
all-source tendency counts at the existing 80% threshold. Both systems get
the same calibration sample. Score-difference intervals resample action
groups. Review misses, false alarms, coverage and compute cost alongside
average accuracy when deciding whether to replace the current synthesis
component.

Alternatives reviewed include [CakeLens v5](https://huggingface.co/fangpenlin/cakelens-v5)
with MIT terms and a 3.25 GB checkpoint, and [STALL](https://github.com/OmerBenHayun/STALL/blob/main/LICENSE.txt)
with CC BY-NC 4.0 terms. The smaller AEGIS checkpoint is the first matched
comparison. The trial does not enable a new live detector.

CakeLens subsequently passed a strict-load CPU smoke using three
nine-frame groups on a real matched-encoding clip. Its raw score was
0.787 and inference took 63 seconds. This establishes local execution
and motivates measuring GPU throughput before using the full cohort.
The checkpoint has 270.6 million model parameters and includes optimizer
state. The evaluator loads only the model state for inference.

### First candidate result

AEGIS completed 140 input variants. Both test conditions contain 23 camera
and 23 generated sources. The unavailable pair remains accounted for
separately. Candidate inference took a median 6.14 seconds per native test
video and 4.53 seconds per encoded copy on the two-thread CPU run.

| Variant | Raw AUC on calibration | Raw AUC on test | Real test clips scored at least 0.80 raw | Generated test clips scored at least 0.80 raw |
| --- | ---: | ---: | ---: | ---: |
| Native | 0.444 | 0.845 | 8/23 | 19/23 |
| Matched encoding | 0.493 | 0.841 | 6/23 | 18/23 |

The preselected monotone calibration learned slope zero across the 48
calibration variants. It returned about 50% on every held-out input and no
80% tendency. This is a material calibration failure despite some ranking
signal on the test generators. Using the raw output as confidence would
label several camera files AI. Candidate scores and the standalone summary are in
`aegis/candidate.jsonl` and `aegis/standalone-summary.json` under the trial
directory. The score receipt SHA-256 is
`fa000b1b4fb5fd5dea3497720bc9fd7f371867f30afc3bd24e2745d925a311d4`.

The same-input comparison used 19 scored calibration inputs common to
both systems and 33 common test inputs. The current synthesis lane had
AUC 0.911 and recalibrated Brier 0.155. AEGIS had AUC 0.941 and Brier 0.175.
Its mean paired difference in Brier, weighted equally by source group, was
+0.011 with a 95% bootstrap interval of -0.043 to +0.057. The test denominator
includes one unavailable file per label.

| Method | Camera, 24 planned | Generated, 24 planned |
| --- | --- | --- |
| Current synthesis lane at 80% after trial calibration | 0 AI, 18 inconclusive, 5 with no score, 1 unavailable | 12 AI, 3 inconclusive, 8 with no score, 1 unavailable |
| AEGIS at 80% after trial calibration | 0 AI, 23 inconclusive, 1 unavailable | 0 AI, 23 inconclusive, 1 unavailable |

These are lane-level tendencies under the trial calibration. The two
full-pipeline real-source spot checks both returned no reliable automated
estimate. The measured release adapters stay in place. One of the two
existing synthesis adapters had no scored real clips in this matched test
condition. The other scored 18 real and 15 generated clips and reached AUC
0.981 on those clips. This makes a separate native-size paired test useful
before changing their relative roles. The current lane score receipt
SHA-256 is `5cbae3189c7975b5502e47a5f4557fe47b4d661bc4c17fad87157ab1b8cc42b1`.

The separate [GenBuster-Bench++ release](https://huggingface.co/datasets/l8cv/GenBuster-Bench-plusplus/tree/7f184bef604882227bf2a585608d3e47e4f0b60b)
has 1,000 videos per label and an MIT tag. The preparation script selects
12 real and 12 generated sources for calibration and 24 of each for test
by fixed filename hash order. It checks the archive digest and each member's
digest. One of the original 2,000 videos lacks a hash-form filename and
is excluded before selection. Per-video generators and shared source scenes
are not given in the archive, so disjointness here is by exact file bytes.
The AEGIS publisher has reported results on this benchmark. It serves as
another source for comparison and cannot be claimed as a blind test of
that publisher's development. The newer archive is separate from the
historical release's scored DeepAction data.

On the native GenBuster test after calibration, AEGIS flagged 11 of 24
generated clips. One leaned original and 12 were inconclusive. No camera
clip leaned AI, 15 of 24 leaned original and nine were inconclusive. AUC
was 0.915, Brier was 0.132, and CPU inference took 9.45 seconds median.
Transferring that fit unchanged to DeepAction flagged 5 of 23 native
camera clips and 3 of 23 matched-encoding camera clips as AI. It flagged
18 of 23 and 16 of 23 generated clips, respectively. AEGIS remains offline.
The native
GenBuster score receipt SHA-256 is
`d8e365d1ab8d421de9a556b4c7a465910fba3b812a7d69708b289b7ba47e377c`.

## Face swaps

[RTFS-10K](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)
declares CC BY-SA 4.0 and identifies the Creative Commons source videos. The
manifest links every original target and its InSwapper and UniFace versions.
It keeps a single target per YouTube source and prevents reuse of donor photo
filenames. It selects successful generations without an optional face enhancer
and excludes records tagged as children. Unrecorded cross-video subject
identity remains possible. Original source links and terms are retained.

All 72 planned videos were acquired, 24 real and 48 swapped. The 12 calibration
and 12 test groups are separate, with one real and both swaps in each group.
Manifest SHA-256
`a86cc49c7b0a40815e0e5921fdf1ccb7260f174d6d937d4e6ed2d684223977eb`.
The face comparison runs the current check and the existing disabled GenD
candidate on the same matched copies. It uses the commercial-safe local
face preprocessing described in the [model audit](commercial-model-audit.md).
Missing faces, failed model calls and paired sample counts are recorded.

The [VideoMAE candidate](https://huggingface.co/Vansh180/VideoMae-ffc23-deepfake-detector/tree/ea26518b527d938d66e568c52bcf1c19a86cf295)
declares MIT weights. Its card names the
[Kinetics-finetuned base checkpoint](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics),
which is labeled CC BY-NC 4.0, and fine-tuning on FaceForensics++, whose
source terms are separate. The derivative MIT tag does not resolve the
base restriction. This candidate remains an offline research comparison.
The offline runner follows the model
publisher's 16 uniformly spaced full-frame sample and verified class
order. Its pinned checkpoint stores the original Transformers v5.2
query and value biases. The runner assigns those values and a zero
key bias to the installed Transformers implementation, then requires
strict loading without missing tensors.

VideoMAE scored all 72 RTFS videos. On the 36 held-out sources it had
AUC 0.698 and a calibrated Brier score of 0.224, with no 80% tendency
after calibration. The median two-thread CPU time was 0.96 seconds.
The score receipt is `face-swap/scores-videomae.jsonl` and its
SHA-256 is `55c430202ab40e08f12d6663f7c52de906f09cee21777eb0ce01cbed6f39c600`.

Both face checks scored all 36 calibration and 36 test videos through their
production adapters on Metal. Their calibration was fitted and recorded before test
rows were complete. The test compares 12 unseen source groups, each with
one original, one InSwapper and one UniFace video.

| Check | Test AUC | Calibrated Brier | Originals at 80% (12) | Swaps at 80% (24) |
| --- | ---: | ---: | --- | --- |
| Current face check | 0.632 | 0.247 | 12 inconclusive | 1 AI, 23 inconclusive |
| GenD | 0.740 | 0.217 | 2 original, 10 inconclusive | 1 original, 23 inconclusive |
| VideoMAE (offline research) | 0.698 | 0.224 | 12 inconclusive | 24 inconclusive |

GenD improved ranking on both swap families. The paired Brier difference
was -0.031 with a 95% source-group interval of -0.075 to +0.020. Using
GenD's raw score at 0.80 would flag 20 swaps and seven genuine originals.
After calibration it correctly flagged no swaps and leaned original on one.
The current face check stays in place. Its one decisive swap and GenD's
decision counts are trial lane results, separate from whole-video fusion.
The score receipt is `face-swap/scores-mps.jsonl` with SHA-256
`ec8a17f8d5b08a3348ede7954c39f6203133bad311193fcf5964f3ec3c7431b5`.

## Lip sync and voice fusion

RTFS originals have no audio track. DeepAction encoded copies also omit audio.
The separate audiovisual control set uses all ten examples from the
[DigiFakeAV publisher's example folder](https://huggingface.co/datasets/cambrain/DigiFakeAV/tree/395b1fa4d494e7474ccaccd638d9ed2df7701d59/few_examples).
The dataset card carries an MIT tag and separates real video and audio,
generated visuals with real audio, and generated visuals with generated
audio. Two real source clips each produce a 3.8-second natural, 320 ms delayed
and cross-source voice-over control. The first source had a silent soundtrack,
which makes one cross-source case a silence control. The 16-row manifest SHA-256
is `8e178406a2b4df843e54ac75801355a174e5ba7c4b298179e8df52e18351e969`.

All 16 completed. The timing check reported no flagged tracks on any of the
eight publisher-labelled synthetic visual examples. It had no stable offset
for the real 3.8-second natural and delayed clips with audible speech, so it
did not recover the known 320 ms shift in that source. The other real source
was silent. The real speech and delayed copy got separate calibrated
original-voice tendencies. A separate correlation of decoded audio verified
the shift at 5,120 samples at 16 kHz, with correlation 0.990 after alignment.
All eight publisher-generated originals exceeded
the voice calibration's four-second duration limit. Their speech windows
were inspected without a whole-track percentage. The recorded outcome hash
is `d7420ce01cdfe02beed264ad5cfc05ad4eb36c1adbce07a59ef41e9dbef8842f`.

The eight synthetic-visual originals were also checked using their first
3.8 seconds, with the same H.264/AAC encoding as the human controls. All
four generated-video/real-audio excerpts leaned original speech
(6.1–6.8% estimated likelihood of generated speech). All four
generated-video/generated-audio excerpts leaned AI speech (95.6–95.7%).
The timing check still flagged none of the eight. The duration-limited
voice estimate applies to the audio only and does not contradict the
publisher's visual label. This small engineering set does not establish
general accuracy for video soundtracks. The source and result commitments
are in `av-controls/short-synthetic/`. The score receipt SHA-256 is
`eee150438dfee7970189e7e876cf6ba39573649b2b098a294ab70668fb9d5f47`.

The video result displays the saved short speech assessment as a separate
section, with a link to voice evidence. The same conclusion appears in the
JSON export and PDF. The short English-speech calibration covers supported
2.5–4 second tracks. Inconclusive or out-of-scope tracks carry no speech
confidence percentage. The visual estimate and timing observations retain
their separate scopes.

### Joint visual and voice calibration audit

The DigiFakeAV release provides original picture/original speech, generated
picture/original speech and generated picture/generated speech. Its examples
lack original picture/generated speech and source lineage for splitting by
speaker. In the local short controls only one independent original video had
audible original speech. DeepAction matched copies and RTFS originals have
no audio. These sets cannot estimate the error rate when one channel is
authentic and the other is synthesized across held-out source groups.

[DeepSpeak v2](https://huggingface.co/datasets/faridlab/deepspeak_v2/blob/main/README.md)
has separate labels, source identity, multiple voice generators and
application-based licensing. [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb#license)
also requires approval to download the dataset, separate from the code's MIT
grant. [LAV-DF](https://github.com/ControlNet/LAV-DF/blob/master/TERMS_AND_CONDITIONS.md)
restricts media use to noncommercial research and education, and
[AV-Deepfake1M](https://github.com/ControlNet/AV-Deepfake1M#license) requires a
signed dataset agreement. [HiFi-AVDF](https://arxiv.org/html/2603.23960v1)
is presented as original/generated pairs and describes research-only media
terms in the paper despite the host's MIT tag. The reviewed sources did not
offer a promptly obtainable, adequately labeled four-condition cohort for
this commercial calibration.

The current evidence supports separate speech and visual tendencies. The
short clips with generated picture and original speech showed why the
distinction matters. The known timing offset control showed why the timing
measurement remains an inspection clue. A joint fit needs the same checks
on all four conditions with independent held-out source groups, including
original picture with synthetic speech and ordinary audio edits.

Potential lip-sync replacements have separate evidence or terms gaps.
[LipFD](https://github.com/AaronComo/LipFD) publishes checkpoint and data
links but has no commercial grant in its source repository. The
[LipSyncTIMIT](https://github.com/skrantidatta/LIPINC-V2#license-and-usage-terms)
data are explicitly limited to non-commercial research. The project's
existing StableSyncNet checkpoint has OpenRAIL field-of-use restrictions
that conflict with expert witness and legal workflows. The
[LADDER study](https://arxiv.org/abs/2604.16808) gives a promising
lightweight design based on mouth landmarks, with no published checkpoint
identified in its paper. These remain separate from the active timing check.

## Reproduction

Run from `backend`. Save trial outputs to fresh paths.

```sh
trial=data/independent-evaluation/video-v2
.venv/bin/python scripts/prepare_video_upgrade_eval.py "$trial/deepaction"
PYTHONPATH=. .venv/bin/python scripts/run_video_upgrade_eval.py candidate \
  "$trial/deepaction/manifest.json" "$trial/aegis/candidate.jsonl" \
  --upstream "$trial/aegis/upstream" --checkpoint "$trial/aegis/checkpoint_best.pt"
PYTHONPATH=. .venv/bin/python scripts/run_video_synthesis_lane_eval.py \
  "$trial/deepaction/manifest.json" "$trial/current-lane-matched.jsonl"
.venv/bin/python scripts/summarize_video_upgrade_eval.py \
  "$trial/deepaction/manifest.json" "$trial/current-lane-matched.jsonl" \
  "$trial/aegis/candidate.jsonl" "$trial/synthesis-summary.json"
.venv/bin/python scripts/prepare_genbuster_video_eval.py \
  "$trial/genbuster/benchmark.zip.part" "$trial/genbuster/locked"
PYTHONPATH=. .venv/bin/python scripts/run_video_upgrade_eval.py candidate \
  "$trial/genbuster/locked/manifest.json" "$trial/genbuster/aegis-native.jsonl" \
  --variant original --upstream "$trial/aegis/upstream" --checkpoint "$trial/aegis/checkpoint_best.pt"

.venv/bin/python scripts/prepare_face_swap_upgrade_eval.py "$trial/face-swap"
PYTHONPATH=. .venv/bin/python scripts/run_face_swap_upgrade_eval.py \
  "$trial/face-swap/manifest.json" "$trial/face-swap/scores.jsonl"
.venv/bin/python scripts/run_videomae_face_eval.py \
  "$trial/face-swap/manifest.json" "$trial/videomae" "$trial/face-swap/scores-videomae.jsonl"
.venv/bin/python scripts/summarize_face_swap_upgrade_eval.py \
  "$trial/face-swap/manifest.json" "$trial/face-swap/scores.jsonl" \
  "$trial/face-swap/summary.json" --temporal "$trial/face-swap/scores-videomae.jsonl"

.venv/bin/python scripts/prepare_video_av_controls.py "$trial/av-controls"
PYTHONPATH=. .venv/bin/python scripts/run_video_av_controls.py \
  "$trial/av-controls/manifest.json" "$trial/av-controls/current.jsonl"
.venv/bin/python scripts/validate_local_audio_rollout.py --mode candidate --cases video \
  --output ../output/soundtrack-paid-check \
  --video-control "$trial/av-controls/short-synthetic/FVFA_20805-short.mp4"
```

For AEGIS, download `checkpoint_best.pt` from the linked checkpoint revision.
Use the four branch Python files and `src/utils/video_io.py` from the linked
source revision in the `upstream` directory. The runner verifies each input
hash before loading the candidate.

For VideoMAE, download `config.json`, `preprocessor_config.json` and
`model.safetensors` into the trial's `videomae` directory from the linked
revision. For CakeLens, the offline runner supports `--candidate cakelens`
with the four pinned model source files and its checkpoint. This uses the
publisher's nine-frame crop and model, but selects up to three source
windows per video to bound runtime, rather than scoring every window.
