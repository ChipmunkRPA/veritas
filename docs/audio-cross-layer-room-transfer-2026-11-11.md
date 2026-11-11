# Cross-layer voice scores under competing speech

A fixed relation head improved ranking on two newly mixed room scenes. The locked operating points recovered none of the incumbent’s six missed synthetic cases. The experiment established a ranking gain on these inputs and no operating-point gain. [Exact aggregate](evaluation-panels/audio-cross-layer-room-transfer-2026-11-11.json)

## Fixed comparison

Compare the serving raw response with compact heads over two masked means from layers 8 and 24 of the same frozen voice encoder. The layer pair was fixed from previously published captures. A shared seeded projection reduces each layer to 64 coordinates. Direct concatenation uses 128 features. The relation head uses the absolute differences and products of those same two sets of standardized coordinates, also 128 features. Both heads get the ordinary score and the same regularization. A third head removes the same-source clean/noisy consistency penalty to measure its effect. The implementation uses neither the papers’ code nor weights. [Cross-layer relation method](https://arxiv.org/abs/2611.05678) · [Speech consistency under distorted conditions](https://aclanthology.org/2026.findings-acl.285/) · [Earlier public feature packets](audio-layer-v12-development-evaluation.md)

The heads learned on the existing fitting partition, using equal total weights by truth class and equal weights by generator, connected group and source within each class. All three fitted heads used the same 1,901 assessed inputs from 169 connected groups and 1,678 matched clean/transformed pairs. No acoustic metadata enters the learned score. The separate calibration partition supplied a fixed 95% rank of the genuine connected-group maxima for each arm. These are internal operating points for comparison, and the task allows only a new AI flag where the incumbent is assessed and inconclusive. Neither the historical development rows nor the new room clips set the points. The older reserved confirmation set was excluded.

## Native transfer

Reuse 11 genuine RUEG foregrounds and eight synthetic foregrounds from Alania, Okur and Drifting that were previously scored with the ordinary classifier. Their exact source commitments and PCM do not overlap the original fitting and calibration roles. An unrecognized common speaker or pretraining overlap across corpora remains possible. The native panel supplies clean controls, two fresh exact −5 dB mixes per foreground, and both background cuts played alone. The two new three-second cuts come from previously used AMI recordings, with zero and 2.58 seconds of annotated simultaneous words. The fresh cases are new mixtures of retained speakers and utterances. [Foreground source and rights](audio-temporal-head-transfer-2026-11-11.md) · [AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Condition | Incumbent synthetic AI flags | Ordinary raw AUC | Concatenated AUC | Cross-layer relation AUC | Relation without pairing AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 8/8 | 1.000 | 1.000 | 1.000 | 1.000 |
| Mostly one interfering voice | 4/8 | 0.917 | 0.865 | 0.958 | 0.938 |
| Overlapping interfering voices | 6/8 | 0.948 | 0.906 | 0.979 | 0.948 |

All 59 inputs were assessed. The noisy rows each have 11 genuine foregrounds and one background-only control. The incumbent flagged zero genuine cases. No arm added any flag on the room panel or the 1,449 previously scored development views. On historical 0 dB competing speech, the paired relation head's AUC was 0.917 versus 0.928 for the original score. The stronger rank on the two new rooms therefore did not generalize uniformly across datasets. The fixed noisy-room result leaves the relation head in evaluation. A new calibration experiment would need separate natural rooms and speakers for setting operating points and testing false alarms.

The CPU run performed 5,546 speech-gate forwards and 72 voice forwards, including the incumbent’s existing stability probes. Both layer features came from the ordinary voice forward. The run took 44.9 seconds and peaked at 2.98 GB process memory. A separate local calculation reproduced every native outcome count and AUC from the saved features and verified all 59 PCM bindings. The held aggregate includes whole-artifact commitments and deliberately excludes individual scores, voice representations, raw recordings and speaker IDs.

## Public numerical replay

From the project’s backend directory, run the two commands in a new empty local directory. The exact role packets are already published. The commands require numerical dependencies and reuse public features without model inference.

```sh
review_dir=$(mktemp -d)
python -m scripts.audio_cross_layer_transfer fit "$review_dir"
python -m scripts.audio_cross_layer_transfer readout "$review_dir"
```

The separate native runner can verify a locally prepared, rights-reviewed panel of exact three-second mono signed 16-bit PCM. It requires a `native-manifest.json` in the fitted directory with ordered integer IDs, case group and truth, condition, source and PCM SHA-256 commitments, and the pinned assets and calibration hashes. Store each waveform at `pcm/<id>.pcm`. The native runner records its freeze before using the local voice weights. The internal acoustic-source manifests and clips used here stay private.

```sh
python -m scripts.audio_cross_layer_native freeze "$review_dir"
python -m scripts.audio_cross_layer_native score "$review_dir"
python -m scripts.audio_cross_layer_native readout "$review_dir"
```

## Natural-room point selection

The 59 cases above then supplied point selection for a distinct confirmation trial. The frozen rule uses the second highest genuine-source maximum across noisy views where the incumbent was inconclusive. It uses 12 eligible genuine groups and permits a strict excess for at most one point-setting group. The same rule applies independently to all four scores. This second study reuses the fitted heads and prepares different speakers and natural rooms for confirmation. [Room point tool](../backend/scripts/audio_cross_layer_room_points.py)

### Confirmation with separate speakers and rooms

The confirmation used 11 other genuine RUEG reader groups, seven Alania voice groups and one Drifting voice group. Each foreground had a clean control and two exact −5 dB mixes. The new AMI sessions provided two three-second cuts with 0 and 2.04 seconds of annotated overlap, plus an isolated control per cut. All eight participants had distinct publisher speaker IDs from those of the four previously used natural-room sessions. The declared foreground groups and exact source hashes were disjoint from the point-setting panel. The source banks have previous project exposure. [Natural-room source receipts and terms](audio-ami-native-stereo-v291-method.md) · [Confirmed aggregate](evaluation-panels/audio-cross-layer-room-confirmation-2026-11-11.json)

| Rule | Mostly one voice, synthetic AI flags | Mostly one voice, genuine false flags | Overlapping voices, synthetic AI flags | Overlapping voices, genuine false flags |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 1/8 | 0/9 | 1/8 | 1/12 |
| Direct raw score, room point | 6/8 | 0/9 | 8/8 | 9/12 |
| Concatenated layers, room point | 6/8 | 1/9 | 8/8 | 11/12 |
| Layer relation, paired penalty | 8/8 | 2/9 | 7/8 | 12/12 |
| Layer relation, no paired penalty | 8/8 | 1/9 | 5/8 | 10/12 |

Three genuine inputs in the mostly-single-voice scene and one genuine clean control were unassessed. All 24 synthetic inputs were assessed. The 19 clean controls kept their incumbent outcomes under the scoped policy, including one synthetic control incorrectly leaning original. The run completed all 59 cases, with 5,546 gate forwards and 66 voice forwards in 39.1 seconds. The early/late relation ranked better than the raw score in the mostly-single-voice scene, AUC 0.972 versus 0.917. Both relation arms ranked worse in overlapping speech, with AUC 0.635 and 0.552 versus 0.771 for the raw score.

The two-scene rule failed its false-alarm condition. The useful next comparison is a predeclared single-interferer trial with additional meetings and synthetic families. It also needs a measured way to identify that acoustic condition without using benchmark labels. The current test’s condition comes from source annotations.

From `backend`, create room points from the completed initial panel and bind their digest into the prepared confirmation manifest as `room_points_sha256`. After running the native `freeze` and `score` commands on the confirmation folder, read the fixed comparison.

```sh
python -m scripts.audio_cross_layer_room_points freeze "$point_selection_dir" "$points_file"
python -m scripts.audio_cross_layer_room_points readout "$point_selection_dir" "$confirmation_dir" "$result_file" "$points_file"
```
