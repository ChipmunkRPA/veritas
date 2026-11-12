# Environmental sound origin under overlapping speech

A small spectral check and a separate acoustic-texture check failed transfer to two previously unseen text-to-audio generator families. The texture check flagged 17 of 18 ordinary backgrounds when mixed with genuine speech. The two experimental responses carry no weight in Chipmunk Verify.

## Frozen inputs and method

Use the pinned EnvSDD test release and its authors’ original-source crosswalk. Before opening audio, join the first 1,100 rows to Clotho’s official per-recording license table. Require an unambiguous exact original filename and a CC0 license, and reject repeated Freesound creators and recording IDs. Fifty-six groups qualified in the inspected rows. A fixed hash order assigned 40 distinct groups to fitting, selection, and held testing in counts of 12, 10, and 18. An ambiguous filename in the license inventory led to explicit exclusion before the freeze. The 120 acquired clips include genuine recordings and publisher-released text-to-audio outputs. No audio-to-audio inputs or generator checkpoints were loaded. [Pinned dataset release and terms](https://huggingface.co/datasets/EnvSDD/EnvSDD/blob/b0832726deb55484dbce869c6f09c41f7e9e5fbb/README.md) · [Clotho per-recording license metadata](https://zenodo.org/records/4783391)

Each development group supplies its original plus AudioLDM 1 and 2 outputs. The held groups instead supply AudioGen and AudioLCM outputs. All roles use exact source-matched publisher groups and distinct authors. The split measures transfer across selected generator families and source groups within one dataset and one original sound corpus. No manual sound-category or incidental-speech screening was applied.

Decode the center three seconds to signed 16-bit mono at 16 kHz. The first check measures per-band energy and flatness. The second uses multiband envelope fluctuation, Teager energy, and short-lag consistency. Train each standardized linear head on fitting sources, with equal weight by origin and source within origin. Fix the lowest point above the second-highest genuine selection response. Test the same points on exact native deliveries, 32 kb/s Opus, and versions mixed with seven independent genuine readers at −6 dB environment-to-speech ratio. The mixture measures whether the background can still be classified while the foreground speaks. It does not isolate either source from an unknown upload. The study uses local NumPy and SciPy implementations and no third-party detector checkpoint. [Mixed-audio texture motivation](https://arxiv.org/abs/2605.23201) · [Environmental component evaluation](https://arxiv.org/abs/2606.10791)

## Held results

Every cell has 18 clips. The two synthetic columns contain the unseen generators. Each delivery repeats the same source groups.

| Delivery | Magnitude false flags, genuine | Magnitude correct, AudioGen / AudioLCM | Texture false flags, genuine | Texture correct, AudioGen / AudioLCM |
| --- | ---: | ---: | ---: | ---: |
| Native | 2 | 1 / 10 | 4 | 6 / 7 |
| Opus | 2 | 1 / 10 | 4 | 7 / 7 |
| Mixed with genuine speech | 0 | 0 / 1 | 17 | 16 / 16 |

A delivery qualifies only with at least 11 hits from each of its two generators and at most two genuine false alarms. Both features failed in all three deliveries. The magnitude check flagged almost no backgrounds through the speech. The texture check responded to most mixtures of either origin. The isolated arms also missed most examples from at least one unseen generator. These results support keeping environmental origin separate from the current speech assessment. [Aggregate and frozen commitments](evaluation-panels/environmental-sound-transfer-2026-11-11.json)

An independent offline replay rebuilt the roster and confirmed all 120 native publisher WAVs and their exact three-second cuts. It then regenerated all 162 held inputs, including fresh Opus round trips and genuine-speech mixtures, and reproduced both cutoffs and all fixed responses without refitting. [Independent reviewer](../backend/scripts/review_environmental_sound_transfer.py)

## Replay

Supply the public metadata named in `metadata_paths` and the exact private parent speech panel. Read only the non-audio Parquet columns `label, attack_type, generative_model, dataset, filename, scene, event, caption` from the pinned `test-00013-of-00014` shard to produce the metadata rows JSON. The freeze commits that file, the EnvSDD test protocol revision and the three exact Clotho license tables. The acquisition stage reads only the two row groups needed for the 120 admitted files. Retain original audio, per-recording provenance, sample digests and scores locally. [Runner](../backend/scripts/environmental_sound_transfer.py) · [Pinned EnvSDD source protocol](https://github.com/apple-yinhan/EnvSDD/blob/d6179d1c773ca78e4939be5933c9914f97815920/metadata/test_metadata.csv) · [Parent speech panel](audio-spatial-interference-2026-11-11.md)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in prepare acquire fit score; do
  backend/.venv/bin/python -B -m scripts.environmental_sound_transfer "$stage" "$run" --speech-parent "$spatial_run"
done
backend/.venv/bin/python -B -m scripts.review_environmental_sound_transfer "$run" "$spatial_run" "$run/audit.json"
```
