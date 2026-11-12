# Overlapping voice windows for short synthetic inserts

Overlapping two-second windows located three of six held synthetic inserts in clean audio and falsely flagged one of six genuine controls. Under louder competing speech, all assessed edits stayed inconclusive across every arm. The experiment did not qualify for serving. [Aggregate and independent replay](evaluation-panels/audio-overlap-window-transfer-2026-11-12.json)

## Fixed comparison

Chipmunk Verify's current voice route scores windows of up to four seconds. Its file-level voice calibration accepts one window covering an entire 2.5–4-second recording. Six-second clips remain inconclusive if any voice window is admitted, with raw window observations still available. This study tests local evidence from those longer inputs. Short partial voice edits can occupy only a small portion of a whole-file score. [Long-form noisy speech benchmark](https://arxiv.org/abs/2507.16220) · [Partial speech localization research](https://arxiv.org/abs/2604.02913)

Compare the strongest production-length window, a two-second grid starting at 0, 2 and 4 seconds, and a second two-second grid starting at 1 and 3 seconds. All views use the installed voice model and the same activity spans from one pass over the exact track. Retain a short window only when its detected speech covers at least half its duration. The overlap score is the lower raw log-odds score of two adjacent admitted windows. Require detected speech in at least half of their one-second intersection. Retain the highest overlap score and its intersection as the proposed support region. The views share samples, so their scores do not supply independent confirmation or a calibrated probability. [Checkpoint terms](commercial-model-audit.md)

Each experimental arm uses a separate point, set to the maximum admitted genuine track response in the clean and noisy selection controls. The screening rule adds a local AI flag to an inconclusive current result only when the overlap score strictly exceeds its point. Unassessed inputs stay unassessed. A useful location requires at least 0.30 intersection over union and 0.50 recall against the exact insert support. Because the target occupies 0.80 seconds, the four-second comparator's maximum possible IoU is 0.20. Its results are a detection comparison, and the short-window arms test the finer support interval.

## Source separation and exact edits

Each role contains six genuine host groups and six synthetic source groups, with two sources per clone, expressive and stock family. Take one earlier qualified matched pair of 0.80-second ordinary and AI voice inserts with 20 ms fades from each comparison. Place each exact three-second clip into a second utterance of its own host, with a fixed offset from 0.75 to 2.25 seconds. Both arms share the same outer host context and artificial joins. Their PCM is identical outside the known insert. This creates controlled long clips rather than untouched six-second field recordings. The earlier source bank, exact short inserts and generator families have prior project exposure. Selection and confirmation share no declared foreground group, and model pretraining overlap remains unknown. [Qualified short edits and controls](audio-short-edit-cases-v172-result.md) · [Source rights](audio-short-edit-source-plan-v169.md)

Each matched pair contributes clean and noisy copies with the same gain. The noisy copies mix continuous human speech from official AMI microphones at −5 dB relative to the ordinary foreground. Use one meeting for selection and two different meetings for confirmation. Manual word times put at least 5.52 seconds of background speech and 1.03–3.22 seconds of simultaneous background voices in the selected cuts. These six-second intervals are outside the cuts from the two prior room studies. There are 48 tracks across both roles, with 24 in confirmation. Repeated deliveries share foreground and room sources. [Native room lineage](audio-stream-presence-transfer-2026-11-11.md) · [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Local results

| Held comparison | Clean AI edits flagged /6 | Clean ordinary false flags /6 | Clean useful locations /6 | Noisy AI edits flagged /6 | Noisy ordinary false flags /6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current whole-file voice direction | 0 | 0 | 0 | 0 | 0 |
| Four-second window comparison | 2 | 0 | 0 | 0 | 0 |
| Two-second grid starting at 0 s | 3 | 2 | 3 | 0 | 0 |
| Two-second grid starting at 1 s | 4 | 1 | 3 | 0 | 0 |
| One-second overlap and combined screening | 3 | 1 | 3 | 0 | 0 |

Every track passed speech admission and all 48 current whole-file voice results were inconclusive. The overlap rule located one cloned and two stock inserts in clean audio. Its three correct locations and one genuine false flag were shared with the grid starting at one second. That grid also added one true flag with weaker location accuracy. The sample provides no evidence that overlap corroboration improved clean discrimination over that individual grid.

In both held rooms, every experimental arm missed all three synthetic edits. Without applying the point, the strongest overlap interval still located only one of six noisy edits. Each separate short grid located three. Across all four experimental arms, noisy edited tracks scored above their own matched genuine controls in only two of six pairs. A lower point alone would leave the location and ordering problems. The combined rule passed one of four advancement checks.

The initial launch stopped at an incorrect benchmark assertion that expected the incumbent's file-level calibration to accept six seconds. No calibration or held results were produced by that attempt. The completed freeze preserves the actual out-of-scope direction and compares the current window length under the same experimental calibration rule as the short arms. The two CPU scoring stages made 96 production-length and 240 short-window forwards, plus 9,024 speech-activity frames. They took 216.7 seconds and peaked at 2.97 GB process RSS. The independent readout rebuilt 48 tracks, checked every scored window hash and activity span rule, and reproduced the four points, 24 result cells, and all advancement checks without re-running inference. The measured model scores, recordings and exact source intervals stay local.

## Local app

A standalone voice finding now links to recorded window coverage when it has saved scores and no selected review clip. The evidence view shows exact decoded times and sample offsets. The PDF links to the same coverage and the JSON export retains the full window list. A six-second paid-flow test verified the two saved intervals at 0.000–4.000 and 2.000–6.000 seconds. Its whole-file assessment remained inconclusive. A second recording kept its existing selected-clip links.

## Reproduce locally

Use the retained v169c and v172 source files, the three AMI microphone recordings, the pinned manual annotation ZIP and the existing local audio checkpoint. Use a fresh run directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
args=("$run" "$annotations" "$rooms" "$weights")
backend/.venv/bin/python -B -m scripts.audio_overlap_window_transfer freeze "${args[@]}"
backend/.venv/bin/python -B -m scripts.audio_overlap_window_transfer score "${args[@]}" --panel selection
backend/.venv/bin/python -B -m scripts.audio_overlap_window_transfer fit "${args[@]}"
backend/.venv/bin/python -B -m scripts.audio_overlap_window_transfer score "${args[@]}" --panel confirmation
backend/.venv/bin/python -B -m scripts.audio_overlap_window_transfer report "${args[@]}"
backend/.venv/bin/python -B -m scripts.review_audio_overlap_window_transfer "$run" "$annotations" "$rooms"
```
