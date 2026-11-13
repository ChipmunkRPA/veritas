# Keep the current audio decision and test a positive-only second opinion

A frozen pooled combination added ten AI flags and three ordinary-view flags in the archived confirmation role. Only one connected genuine group was newly flagged. A separate 32-recording RADAR transfer reproduced none of the gains. The existing audio decision remains in service. [Archived replay](evaluation-panels/audio-selective-fusion-transfer-2026-11-12.json) · [Prospective RADAR replay](evaluation-panels/audio-selective-radar-transfer-2026-11-12.json)

## Archived comparison

The previous whole-file combination assigned many new original directions to generated recordings. This test retains every existing AI, original and unassessed result. For an assessed but inconclusive recording, it compares a new AI flag from the original raw score, the fixed pooled combination, two previously fitted context maps, and a rule requiring both the raw and common-reference responses to clear their points. These share an underlying voice check. Agreement carries correlated evidence. Environmental-audio studies motivate separating the target voice response from the condition around it. [Earlier confidence comparison](audio-combined-confidence-v84-final.md) · [ESDD2](https://arxiv.org/html/2606.10791v1) · [Cross-view residual research](https://arxiv.org/html/2609.13842v1)

The operating point for each arm and measured spectral-flatness branch is the third-highest genuine connected-group maximum from the earlier calibration role. Use strict excess and require eight genuine groups. The existing fit-only flatness split measures the submitted waveform. It receives no reference SNR, file label or generator. The new policy was designed after the old comparison. It reuses two source-separated roles of 1,044 already scored recordings each. Each role has 116 exact sources with repeated clean and mixed variants. The two role results are comparisons on consumed inputs. There are no new model forwards in this part.

| Archived confirmation arm | Additional AI views | Additional genuine views flagged | Newly flagged genuine source groups | Extra AI views with competing speech |
| --- | ---: | ---: | ---: | ---: |
| Raw score alone | 5 | 3 | 1 | 1 |
| Pooled combination | 10 | 3 | 1 | 0 |
| Flatness branch | 21 | 10 | 3 | 2 |
| Common-reference branch | 21 | 10 | 3 | 2 |
| Common-reference and raw agreement | 4 | 2 | 1 | 1 |

The existing confirmation result found 366 of 540 generated variants. It flagged 19 of 504 genuine variants across seven connected groups and gave 17 generated variants an original direction. The pooled policy raised those counts to 376 AI detections and 22 genuine false flags across eight groups. Its ten gains cover all three declared generator families. Seven came from noise and three from music. Two of the three extra genuine-view flags also came from the strongest tested noise condition. Both flatness branches found more edits and added false flags to three previously unflagged ordinary groups. Their directions were identical in this comparison. The positive-only rule cannot increase false-original directions. It leaves the existing 17 in place.

## Transfer to unused published recordings

Select eight unused genuine and eight unused generated English FLAC files from each of two official RADAR archive prefixes. Require published whole-file labels, 5–12 second originals, native mono signed 16-bit PCM at 16 kHz, and a continuous clock. Exclude project-recorded RADAR selections and byte aliases. Freeze the original first 64,000 samples and both historical pooled points before scoring. The recordings had no earlier detector scores in the tracked RADAR studies. Generator, speaker and transformation identities remain unavailable. The 32 files are a bounded transfer sample and do not estimate separate rates by generator or background. [Publisher release and commercial-compatible CC BY 4.0 terms](https://zenodo.org/records/22110815) · [Voice assets](commercial-model-audit.md) · [Complementary encoder terms and pins](audio-wavlm-v13-development-evaluation.md)

The current check, the raw-only rule and the pooled second opinion gave identical directions. They found **12 of 16** synthetic inputs and flagged **zero of 16** genuine inputs. Three synthetic and three genuine inputs were inconclusive. One per class was unassessed. The extra check ran on the six assessed inconclusive inputs, and none cleared the frozen point. The prospective criterion required four extra AI flags across both archive splits and at most one extra genuine flag. The controlled gains did not transfer.

## Reproduce

The public reader checks both archived ledgers, frozen group points, all prospective directions and the unchanged abstentions. The archived ledger uses role-local integer aliases. The prospective receipt includes only exact sampled-window commitments and score records. No source audio or learned parameters are published. The native reader independently decoded and verified all 32 complete FLAC clocks and four-second prefixes.

```sh
python3 -B -IS backend/scripts/review_audio_selective_fusion_transfer.py \
  docs/evaluation-panels/audio-selective-fusion-transfer-2026-11-12.json \
  docs/evaluation-panels/audio-selective-radar-transfer-2026-11-12.json

export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_OFFLINE=true
for phase in freeze fit report review; do
  backend/.venv/bin/python -B -m scripts.audio_selective_fusion_transfer "$phase" "$operating" || exit 1
done
for phase in freeze score audit; do
  backend/.venv/bin/python -B -m scripts.audio_selective_radar_transfer "$phase" "$source" "$run" "$operating" || exit 1
done
```

For native replay, use the original compact calibration, selection and confirmation journals and the separately fitted artifacts linked from the [previous comparison](audio-combined-confidence-v84-final.md). For RADAR, use the bounded publisher archives and previous intake receipts described in the [RADAR source procedure](audio-radar-transfer-2026-11-12.md). Inference requires the existing pinned voice, activity and complementary encoder weights plus the original [inference-only bundle](audio-combined-serving-v97-result.md).
