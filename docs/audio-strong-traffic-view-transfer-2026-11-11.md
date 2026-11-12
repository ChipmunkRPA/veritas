# Corroborating voice scores under heavy traffic

The new two-view rule added **five synthetic-voice flags and five false flags on genuine readers**. The correct additions came from four foreground groups in one of three traffic recordings. The rule failed its frozen advancement criteria and remains experimental. [Audited aggregate](evaluation-panels/audio-strong-traffic-view-transfer-2026-11-11.json)

## Source-separated test

Ten distinct LibriSpeech test-clean readers supplied exact three-second center excerpts. Their known reader IDs were excluded from the prior audio source and calibration catalogue and the held short-edit manifest. The archive matched the publisher's checksum. Eight previously qualified synthetic voices supplied three clone, three expressive, and two stock controls. Their exact sources and declared groups or voices were outside the preceding WavLM transfer. Some synthetic inputs had previously been used in other project studies. Model training overlap remains unknown. [OpenSLR publication and CC BY 4.0 license](https://www.openslr.org/12/) · [Synthetic source qualifications and licenses](audio-short-edit-source-plan-v169.md) · [Prior WavLM transfer](audio-wavlm-two-background-transfer-2026-11-11.md)

All 18 voices supplied a clean view and six noisy views. Three independently recorded roadside backgrounds provided exact, machine-screened provisional excerpts at nominal −5 and −10 dB full-window SNR. Both levels use the same background excerpt per foreground and recording. The mixed signal receives common attenuation when needed to avoid clipping. Two isolated excerpts from each background complete the 132 planned views. The three tapes were new to the earlier WavLM traffic and two-background comparisons. They had previous exposure to the current model in another panel. The earlier screening is mechanical and has no human listening confirmation. [Background source rights and recorded coverage](audio-commons-background-v234-result.md) · [Provisional slice evidence](audio-commons-slice-screen-v247-result.md)

## Frozen comparison

Keep the current three-second classifier, calibration, and full-profile speech gate on the original signed 16 kHz samples. For an originally admitted inconclusive recording, calculate the previously fitted second opinion on both the original waveform and a reduced view. The fixed transform subtracts 1.5 times the frequency bin's 20th-percentile short-time power estimate, using 512-sample Hann windows and a 128-sample hop. It retains at least 16% of power in each bin and reconstructs with the original phase and the same sample count. The changed waveform is a diagnostic view. It does not establish the speech or noise contribution.

Use the same paired head and exact published AI point. The primary rule changes an admitted inconclusive result only when the reduced view reaches the point and the ordinary paired score is within 0.05 of the point or higher. Compare the ordinary and reduced scores separately with the same point. The recorded responses never landed on the point itself. The study requires at least six added noisy synthetic flags across three foreground groups and two traffic sources, at most one added false flag, zero clean changes, and at least five genuine admissions per −5 dB scene. No parameters were fitted in this test. [Frozen head and input contract](audio-paired-transfer-v230-method.md) · [Current checkpoint terms](commercial-model-audit.md) · [Pinned second-opinion checkpoint and publisher terms](https://huggingface.co/microsoft/wavlm-base-plus/blob/4c66d4806a428f2e922ccfa1a962776e232d487b/README.md)

## Results

| Condition | Genuine assessed / planned | Synthetic assessed / planned | Synthetic AI flags, incumbent → primary | Genuine false AI flags, incumbent → primary |
| --- | ---: | ---: | ---: | ---: |
| Clean | 10/10 | 6/8 | 5 → 5 | 0 → 0 |
| Traffic −5 dB | 21/30 | 10/24 | 1 → 5 | 1 → 3 |
| Traffic −10 dB | 9/30 | 3/24 | 1 → 2 | 1 → 4 |

The primary rule added five genuine false flags across four reader groups and two traffic sources. The ordinary second opinion alone added four synthetic flags and four genuine false flags. The reduced view alone added six of each. The 73 unassessed cases include all six isolated backgrounds and 35 of 48 noisy synthetic views. No expressive voice passed the original speech gate under traffic. These measured limits separate two gaps. The original gate rejects many heavily masked speech windows, and adding a second score can incorrectly flag readers even where the gate admits the recording. None of the experimental arms enters the overall confidence score.

The run completed 12,408 speech-gate frame forwards, 109 current-model forwards including stability and reduced views, and 76 paired encoder evaluations. The independent standard-library readout checked all 132 exact PCM hashes, 108 mixture levels, all 18 foreground source bindings, and all four decision paths. The worst measured noise-level quantization deviation was 0.000289 dB. One preparation attempt stopped before inference when the historical background helper rejected its old implementation fingerprint. The completed attempt independently verified the saved publisher acquisition, exact PCM qualification, and provisional window commitments before freezing.

## Local replay

Use a new local directory with the publisher's test-clean archive, the retained v169c synthetic preparation, the v247 qualified source excerpts, the earlier WavLM transfer receipts, and the pinned model assets. The worker uses the existing offline model runtime. The [reviewer](../backend/scripts/review_audio_strong_traffic_view.py) recomputes the arithmetic separately from the model runtime. Only aggregate counts and whole-artifact commitments are published.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m scripts.audio_strong_traffic_view freeze "$run_dir" \
  --librispeech "$test_clean_archive" --earlier "$previous_wavlm_run"
python -m scripts.audio_strong_traffic_view baseline "$run_dir"
python -m scripts.audio_strong_traffic_view candidate "$run_dir"
python -I backend/scripts/review_audio_strong_traffic_view.py "$run_dir" \
  --published docs/evaluation-panels/audio-strong-traffic-view-transfer-2026-11-11.json
```
