# Confirming voice stability after simulated call processing

The frozen paired rule reduced false genuine groups from three to two and kept both useful locations at their original intervals. It missed useful localization on ten of 12 generated inserts. The result supports a follow-up for this exact research window and delivery scope. [Anonymous windows and independent replay](evaluation-panels/audio-communication-replication-2026-11-12.json)

## Source separation and fixed comparison

Use 12 new six-second speech groups, with four generated inserts per family from DuplexGen PLN, ElevenLabs v4, and Kokoro. Each group has a genuine same-reader splice, a genuine other-reader splice, and an untouched host. Four isolated room captures bring the total to 52 original and processed pairs. The selection verified no shared original source bytes or declared foreground speakers across discovery and confirmation. The four confirmation AMI meetings have 16 recorded participants, with no declared participant in the discovery meetings. The independent input audit checked 440–960 ms of overlapping annotated competing words within every edit. All source archives and eight held synthesis voice profiles had earlier unrelated project exposure. No claim is made about source presence in pretrained data. [Source licenses and discovery panel](audio-remote-neighbor-replication-2026-11-12.md)

The first two intake attempts stopped before freezing or measuring results. The retained pools could fill only one entirely unused profile in each of the expressive and stock families. The final source freeze excluded all discovery identities, known prior exact inputs and repeated held identities, while allowing expressive and stock profiles used in unrelated studies. It selected new exact originals and captured room intervals before scoring. Raw source material stays local.

At −5 dB whole-clip foreground-to-background ratio, match all four original variants in each group outside the known edit. Each meeting supplies one 0.64-second edge edit and two interior edits of 0.80 and 0.96 seconds. The edge edits fall wholly outside the fixed model windows. Use the unchanged bandpass, compressor and 24 kb/s Opus voice chain. The app decoder verifies that every processed delivery retains the original mixture's continuous, zero-start, 96,000-sample clock. [Fixed discovery method](audio-communication-transfer-2026-11-12.md)

Compare the strongest admitted two-second window in each delivery with the strongest shared window after taking the lower response at each identical sample interval. The two windows cover 1–3 and 3–5 seconds. Both views use the same installed voice checkpoint and speech gate. The three arms share the frozen discovery point and use strict excess. A useful location needs at least 0.30 temporal intersection over union and 0.50 edit coverage. Advance only with a useful paired location, retention of every useful original interval, and removal of at least one genuine false group without adding another. The test does not fit calibration or provide AI-confidence weight. [Model terms](commercial-model-audit.md)

## Local results

| 12 generated inserts and 40 genuine files | Original | Processed | Both views |
| --- | ---: | ---: | ---: |
| Generated flags at the frozen point | 3 | 3 | 3 |
| Useful generated locations at the frozen point | 2 | 2 | 2 |
| False genuine groups at the frozen point | 3 | 5 | 2 |
| Useful locations at 0 / 1 / 2 matched false groups | 0 / 2 / 2 | 0 / 1 / 1 | 0 / 2 / 2 |
| Unassessed files | 0 | 1 | 1 |

Both retained locations used the 3–5 second window. They contained a clone edit at 3.760–4.560 seconds and an expressive edit at 3.840–4.640 seconds. Each 0.80-second edit had 0.40 temporal overlap by the study's intersection-over-union rule. The paired arm kept the one generated flag outside its useful edit range. It found zero of four stock edits and zero of four edge edits. Four room-only controls produced zero false flags across all arms, with one losing speech admission after processing. At equal false-group counts, the original and paired arms found the same number of useful edits.

The two-second windows used here differ from the Full app's up-to-four-second windows. A six-second upload in that profile is planned at 0–4 and 2–6 seconds before speech admission. The measured benefit applies to the research windows and this processing chain. The paired responses share input samples and classifier weights, so the result supports a stability observation with zero overall confidence weight. [Full window scope](audio-measured-spatial-transfer-2026-11-12.md#voice-comparison)

## Replay

The held run made 103 original and 102 processed classifier forwards. The native audit rerendered all 52 original mixtures and re-decoded all 52 Opus deliveries. The separate standard-library reviewer verified 156 native source components, all 208 model-window commitments including skipped windows, the annotated competing words, and 72 outcome cells. Both audits use the saved model responses. Run these steps from the repository root with the pinned model and source archives available locally.

```sh
base=backend/data/independent-evaluation
run="$base/audio-communication-replication-2026-11-12"
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in prepare score report audit; do
  backend/.venv/bin/python -B -m scripts.audio_communication_replication "$phase" "$run" || exit 1
done
python3 -B -S backend/scripts/review_audio_communication_replication.py "$run" \
  "$base/audio-remote-neighbor-replication-2026-11-12" \
  "$base/audio-communication-transfer-2026-11-12" \
  "$base/spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip"
```

The public receipt contains anonymous source-group ordinals, edit intervals, and the four scalar window responses per matched pair. A public reviewer can reconstruct the reported arithmetic from that receipt alone.

```sh
python3 -B -S backend/scripts/review_audio_communication_replication.py --public \
  docs/evaluation-panels/audio-communication-replication-2026-11-12.json
```
