# Overlapping speech windows on previously unused files

The local Full speech review reads two or three disjoint windows. A short voice edit can fall between them or account for only a small share of an inspected window. This comparison tests an additional grid of three-second windows spaced 1.5 seconds apart. Both neighboring windows must return an AI-leaning direction. The method reports their shared 1.5 seconds and retains all existing sparse evidence. The voice weights and their calibration stay fixed. The overlap is a region to inspect. Known edit times come from the publisher's segment labels.

## Source and procedure

The study selects 18 partially edited and 12 genuine clips from the licensed LlamaPartialSpoof release. The edits cover six generator families, with three distinct source speaker groups in each. No selected file ID or native PCM digest appears in the three linked prior source ledgers. Eight of 30 speaker IDs are new to those ledgers, including one of 18 edited source IDs. This is a transfer to new exact files, with substantial previous speaker exposure. No participant's identity or model training exposure was independently verified.
[Publisher release and CC BY 4.0 terms](https://zenodo.org/records/14214149) · [Earlier publisher localization](audio-partial-publisher-transfer-2026-11-12.md) · [Background-reduced localization](audio-reduced-background-localization-2026-11-12.md) · [Staggered window study](audio-noise-window-transfer-2026-11-12.md)

Each 6–11 second native mono source contributes a gain-only control and a mixture with recorded traffic. The full foreground is 8 dB louder than the added recording by RMS. Matched deliveries share one peak-safe gain. The four starting offsets come from a single 24.4-second traffic recording used in earlier experiments. The experiment covers neither speech buried under stronger background sound nor independent recording environments. It uses the pinned candidate voice calibration and compares the current long-window check, the dense grid alone, and their union. A useful location covers at least half the labeled edit and reaches 0.30 temporal intersection over union. The fixed comparison allows at most one additional genuine group and four percentage points of additional flagged genuine time, with useful additions in at least five edit groups and three generator families. This measures time spent on genuine speech inside both untouched and partly edited recordings.
[Traffic source and credit](https://commons.wikimedia.org/wiki/File:Audio_sound_of_a_busy_traffic_in_Porthacourt.ogg) · [Current Full speech-window review and scope](audio-radar-transfer-2026-11-12.md#local-evidence)

## Results

| Condition | Sparse review | Dense review | Union |
| --- | ---: | ---: | ---: |
| Gain-only edits localized | 7/18 | 13/18 | 15/18 |
| Edits localized with traffic | 6/18 | 8/18 | 12/18 |
| Genuine gain-only files flagged | 0/12 | 0/12 | 0/12 |
| Genuine files flagged with traffic | 0/12 | 1/12 | 1/12 |
| Genuine time flagged, gain-only | 23.35% | 17.58% | 31.74% |
| Genuine time flagged, traffic | 24.85% | 9.53% | 29.93% |

All 60 files had a usable response on at least one sampled window. The union added useful locations for nine distinct edited source groups from five families and one 1.5-second flag on a genuine traffic file. Across both conditions, flagged genuine time rose from 24.10% to 30.83%. That 6.73-point increase exceeded the frozen four-point limit, so the union failed the original gate. The dense grid alone reduced flagged genuine time to 13.55% across both conditions. It added eight useful gain-only locations and six traffic locations, while losing two useful gain-only locations and four with traffic that sparse had found. File and time rates reflect this deliberately enriched panel. Most flagged genuine time came from speech outside the annotated edit in partly edited files.

Keep the current review in the app. The dense grid is a candidate for shorter, separately labeled inspection ranges. A fresh comparison should test whether it can retain both methods' useful locations without showing long overlapping amounts of genuine speech. Include independently sourced noisy recordings and more held source speaker groups before using the result to revise confidence. The offline reader verified 300 native window hashes against the 60 delivery commitments.
[Frozen result and portable replay](evaluation-panels/audio-dense-evidence-confirmation-2026-11-12.json)

## Reproduce

Use the same archive byte ranges and publisher labels as the [prior source preparation](audio-reduced-background-localization-2026-11-12.md#reproduce-the-frozen-inputs). The archive folder contains the 192 MiB prefix, its 160 MiB extension and the three 96 MiB slices. The traffic recording is the exact Ogg in the [earlier download instructions](audio-radar-transfer-2026-11-12.md#reproduce-the-source-selection). Set `run` to a new private scratch path. The runner records the archive hashes, source and delivery commitments, and procedure before collecting any voice results. Original speech and derived mixes stay local.

```sh
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_BILLING_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_dense_evidence_confirmation run "$run" \
  --archive "$archive" --labels "$labels" --background "$traffic"
python3 -B -S -m scripts.audio_dense_evidence_confirmation review "$run/receipt.json" --native "$run"
```

The portable reader also recomputes the comparison from the published result without loading model weights or decoded media. Native replay additionally checks each delivered PCM commitment and each scored source window.
