# Independent speaker counting for noisy voice screening

An independent speaker counter closed two of six incorrect three-voice routes and preserved all seven recovered synthetic-voice flags in the separate room test. A level-normalized variant withheld five of the six incorrect routes and preserved five recovered flags on the same material. Both variants need further testing before they can route voice confidence. [Exact aggregate and freezes](evaluation-panels/audio-independent-overlap-count-2026-11-11.json)

## Fixed comparisons

Use the authors' MIT-licensed CountNet convolutional-recurrent weights and frequency scaler from commit `ae15e1ac096862667a7bfdedf0b67a70a7543edd`. The model estimates the maximum concurrent speaker count in a five-second mono input. The local runner reads named HDF5 arrays and implements the original Theano convolution direction and hard-sigmoid recurrence on CPU. It loads neither serialized Python objects nor the original Keras runtime. The authors' example returned five voices. An independently implemented SciPy and NumPy forward reproduced its class and its outputs within 0.000000015. Authored digital silence returned zero in both implementations. The repository distributes the runner and summary. The author's checkpoint, scaler and example remain in the local experiment directory with the MIT notice. [Author release and model terms](https://github.com/faroit/CountNet/tree/ae15e1ac096862667a7bfdedf0b67a70a7543edd) · [Paper and dataset](https://www.audiolabs-erlangen.de/resources/2017-CountNet)

Replay all 146 prior exact three-second inputs. Include the 118 point-selection and separate-room clips and the 28 known-count controls. Prepare another 36 waveforms from nine disjoint triplets of 27 genuine English and Turkish readers from older project source banks. Their declared reader groups and exact clean PCM are disjoint from the prior routing study. Use the center three seconds of the retained gain-matched track and require at least 400 ms of joint speech on the three separate source signals under the fixed speech activity check. Each triplet supplies one voice, two voices, three with equally loud interferers, and three with the third voice 6 dB quieter than the other interferer. Both three-voice conditions use the same −5 dB total foreground-to-background ratio. These exact mixtures are new. Model pretraining overlap is unknown. [Original routing panel and AMI source caveats](audio-measured-speaker-routing-2026-11-11.md) · [Reader corpus and rights](audio-temporal-corroboration-2026-11-11.md)

Center the three-second mono waveform in a five-second buffer with digital silence. Also shift it half a second earlier and later within separate buffers. The count uses every source sample exactly once per view. Compare a center count of at least three with a count of at least three in two of the three views. Either rule may veto a two-voice route already admitted by both diarization operating points. It never supplies a route on its own. The voice detector, calibration and room points stay fixed. Advancement requires no known three-voice leak, retention of at least half of the new genuine two-voice controls, and at least five of seven recovered synthetic-voice flags with zero added false flags in the separate room test.

## Results

| Condition | Diarization alone | With independent count veto |
| --- | ---: | ---: |
| Retained three-voice mixtures routed incorrectly | 6/14 | 4/14 |
| Retained two-voice mixtures routed | 5/7 | 4/7 |
| New three-voice mixtures routed incorrectly | 0/18 | 0/18 |
| New two-voice mixtures routed | 9/9 | 9/9 |
| Additional correct synthetic-voice flags in the separate room test | 7 | 7 |
| Additional false flags in that room test | 0 | 0 |

The center-only and two-of-three policies reached identical routing totals. The independent count estimated zero voices in 16 of 19 separate-room clips with one interfering voice plus foreground. It estimated three voices on 14 of 19 clips in the overlapping scene. The nine new triplets provided a disjoint-reader check, but the original diarizer already handled all their three-voice mixtures. The remaining four retained leaks matter for the proposed use.

Freeze two gain targets and replay the consumed panel as a diagnostic. Multiply the entire three-second waveform by one scalar to target −24 or −18 dBFS RMS, with a sample peak ceiling of 0.95. Compare the ordinary explicit-overlap veto with one that also withholds the route on a zero count. At −24 dBFS, the zero-or-overlap rule reduced the retained three-voice leaks from six to one, kept all nine new two-voice routes, and retained five of seven recovered synthetic flags without adding a false flag in the separate room test. It kept two of the five retained two-voice routes. A zero count conveys disagreement with the diarizer and remains an uncertain measurement. The result comes from a consumed diagnostic panel and leaves one known unsafe route. The proposed supplement remains experimental.

The comparison ran 546 count-model forwards, one separate publisher-example qualification, and 72 diarization forwards for the disjoint-reader inputs. The level replay added 364 count-model forwards. A separate standard-library replay verified all 182 PCM commitments, 364 retained and new diarization interval grids, all count outputs, and every route and source-group total. Public results include condition counts and artifact commitments. Audio samples, source identifiers and per-clip results stay with the evaluator.

## Reproduce locally

From the `backend` directory, use the [count runner](../backend/scripts/audio_independent_overlap_count.py) with the usual numerical and PyTorch dependencies plus `h5py`. In a fresh working folder place the exact author files listed in its `ASSETS` dictionary. Use the [prior routing panel](audio-measured-speaker-routing-2026-11-11.md) files as `previous-freeze.json`, `previous-manifest.json` and `previous-measured.json`. Supply the exact signed 16-bit PCM in `pcm/NNN.pcm` at the ordered indices. The `prepare` command takes the retained temporal audio `roster.json` and `pcm/` folder, plus the exact local speech activity weights. It constructs the disjoint-reader mixtures and records their hashes.

```sh
python -m scripts.audio_independent_overlap_count prepare RUN_DIR --source LOCAL_READER_PANEL --vad LOCAL_VAD
python -m scripts.audio_independent_overlap_count freeze RUN_DIR
python -m scripts.audio_independent_overlap_count run RUN_DIR
```

For the new diarization replay, copy the fresh manifest into `RUN_DIR/fresh-diarization/manifest.json` with its row IDs renumbered from zero. Use those same local IDs for the corresponding byte-identical PCM. Add the pinned runtime archive and model from the earlier study, then run its `freeze` and `run` commands in that subfolder. The independent readout binds the two rosters and rechecks each measured diarization interval.

```sh
python -m scripts.audio_independent_overlap_count readout RUN_DIR
python -m scripts.audio_overlap_count_level_replay freeze RUN_DIR
python -m scripts.audio_overlap_count_level_replay run RUN_DIR
```
