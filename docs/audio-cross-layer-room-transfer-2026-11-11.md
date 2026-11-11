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
