# Comparing repeated speech context in noisy original audio

The new cue located two of four fresh edits at its frozen point, with zero false genuine groups. The current native comparison located none at the same realized false-group count. The cue located no edits in either earlier panel and falsely flagged six genuine groups on the earlier expressive panel. The result supports replication on more fresh sources before product use. [Aggregate results](evaluation-panels/audio-remote-neighbor-transfer-2026-11-12.json)

The later 12-edit replication found zero useful locations with the same cue and cutoff. [Larger replication](audio-remote-neighbor-replication-2026-11-12.md)

## Frozen cue

Measure one original six-second mono waveform. A separate locally pinned speech encoder supplies an early and a late representation. For each admitted speech frame, choose four late-layer neighbors at least 0.80 seconds away, one per distinct 0.20-second bucket, with a minimum cosine similarity of 0.20. Compare the query's early-layer distance to those neighbors with the early-layer differences among the neighbors themselves. The same-frame late-layer search matches acoustically similar contexts. It does not establish which phones or speakers they contain. The model sees the full original waveform. No earlier recording, separated signal, exact edit interval, or stem enters scoring. [Pinned weight card](https://huggingface.co/microsoft/wavlm-base-plus/blob/4c66d4806a428f2e922ccfa1a962776e232d487b/README.md) · [Model-card-linked terms](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)

Native frame centers are 200 samples from the beginning, then every 320 samples at 16 kHz. A proposed 0.80-second range needs at least 24 supported centers and begins on the fixed 0.20-second grid. Its score is the median supported response. The strongest range represents the file. The late-layer matching uses remote recurrence, unlike adjacent trajectory changes or a boundary model. In strong mixtures, similar embeddings can come from foreground speech, a competing voice, or their combination. The matched ordinary splices and isolated rooms test the resulting specificity directly. [Adjacent trajectory research](https://arxiv.org/abs/2604.01083) · [Recent intra-segment study](https://arxiv.org/html/2609.25822v1) · [Earlier project layer and boundary comparison](audio-partial-layer-boundary-2026-11-12.md)

Use 12 genuine previously measured point controls and lock the new cue at their maximum eligible response before held scoring. Use the existing current threshold on exact two-second native windows beginning at 1 and 3 seconds. Compare the cue and current check separately. A fixed union flags if either arm exceeds its fixed point and preserves both reported ranges. The retrospective equal-false-group comparisons use budgets zero, one and two, counting a genuine group once if either ordinary view flags. Useful localization reaches 0.30 temporal intersection over union and covers at least half of the known edit. Unassessed results and false isolated-room flags remain visible.

## Separate source cohorts

The earlier expressive and two-family panels supply 46 exact mixtures, including 14 edited recordings. They were previously evaluated with the incumbent, and their scores are reused only after exact source and window verification. They had no earlier exposure to this cue. The fresh cohort contains four new 0.64- or 0.96-second inserts, two expressive and two stock, with eight matched genuine splices and two isolated-room controls. The new records have four distinct host readers, four distinct other-reader donors and four distinct generated voice profiles. Each exact generated asset, script and decoded fragment was screened against the known archive history and prior cohorts. Banks, some profiles, and rooms had unrelated project exposure. [Expressive source and terms](https://huggingface.co/datasets/laion/elevenlabs-v4-five-language-voice-acting) · [Stock source qualification](audio-short-edit-source-plan-v169.md) · [Earlier expressive transfer](audio-variable-boundary-transfer-2026-11-12.md) · [Earlier two-family transfer](audio-short-native-response-transfer-2026-11-12.md)

The fresh six-second mixes have matched 20 ms fades, a −5 dB whole-clip foreground-to-background level, and identical PCM outside each exact edit. At least two recorded background speakers overlap each fresh edit for 320–760 ms. The two room recordings have distinct participant groups, and the selected physical capture intervals exclude exact prior cuts in the bound room receipts. Model pretraining overlap is unknown. [Genuine source terms](https://www.openslr.org/12/) · [AMI room terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held results

| Panel | AI edits | Current fixed useful / false groups | Neighbor fixed useful / false groups | Fixed union useful / false groups |
| --- | ---: | ---: | ---: | ---: |
| Fresh two-family cohort | 4 | 0 / 0 | 2 / 0 | 2 / 0 |
| Earlier expressive cohort | 8 | 1 / 2 | 0 / 6 | 1 / 6 |
| Earlier two-family cohort | 6 | 0 / 1 | 0 / 1 | 0 / 2 |

| Fresh retrospective false groups | Current useful | Neighbor useful |
| --- | ---: | ---: |
| 0 | 0 | 3 |
| 1 | 2 | 3 |
| 2 | 2 | 3 |

The two fresh frozen-point findings were expressive voices in different rooms. The first known edit spans 0.24–0.88 seconds and the cue reported 0.20–1.00 seconds, with 500 ms of simultaneous annotated competing speech inside the true edit. This edit lies before the incumbent's first two-second window. The second true edit spans 3.60–4.56 seconds, within the incumbent grid. The cue selected 3.60–4.40 seconds, with 680 ms of recorded overlap in the edit. Neither exact room-only control flagged. At the retrospective zero-false-group point the cue also localized a stock edit, and that response was below the frozen point. The other fresh stock edit remained poorly localized.

The earlier expressive panel's six false genuine groups include one isolated room. Two flagged other-reader controls overlapped their known ordinary splices, showing that the cue can respond to a genuine speaker change. The earlier two-family panel supplied an additional false group under the fixed union and no additional useful range. All fresh cases were assessed. One earlier expressive input was unassessed by both arms. The four fresh edits cannot establish a reliable cross-generator or room-specific operating point. Earlier results show that the cue cannot currently be interpreted as an authenticity finding by itself.

## Replay

The local run made 72 original full-clip encoder forwards and 28 current-detector forwards for the fresh comparison. The first setup attempt stopped before any forward because the genuine point path used a cohort label as the source locator. The corrected freeze reproduced the exact roster. The native audit rebuilt the 14 fresh inputs and their original PCM, and the standard-library verifier replayed source window hashes, exact neighbor clocks, annotated overlaps, and all fixed and equal-false-group results. Neither audit reruns the models.

```sh
base=backend/data/independent-evaluation
run="$base/audio-remote-neighbor-final-2026-11-12"
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_remote_neighbor_transfer audit "$run"
python3 -B -S backend/scripts/review_audio_remote_neighbor_transfer.py "$run" \
  "$base/audio-original-pcm-boundary-2026-11-12" \
  "$base/audio-same-speaker-enrollment-2026-11-12" \
  "$base/audio-variable-boundary-2026-11-12" \
  "$base/audio-short-native-response-2026-11-12" \
  "$base/spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip"
```

To reproduce from the retained archives, use a new local directory and run `freeze`, `point`, `held`, `current`, and `report` before both audits. Keep raw media, individual frame relationships, embeddings, and scores local. [Runner](../backend/scripts/audio_remote_neighbor_transfer.py) · [Independent reviewer](../backend/scripts/review_audio_remote_neighbor_transfer.py)
