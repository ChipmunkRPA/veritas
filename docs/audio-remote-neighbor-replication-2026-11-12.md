# Fresh replication of the remote-context audio cue

The discovery gain did not repeat on 12 new edits. At the unchanged points, the current native check localized one edit and the remote-context cue localized zero. The result was the same when the cue's eligible ranges were restricted to the current check's admitted time coverage. [Aggregate results](evaluation-panels/audio-remote-neighbor-replication-2026-11-12.json)

## Held sources and frozen comparison

This replication uses 12 new edited clips, with four independent donor recordings each from DuplexGen PLN, ElevenLabs v4, and Kokoro. Each edited group also contains a genuine same-reader splice, a genuine other-reader splice, and an unedited host delivery. Four genuine AMI room-only controls bring the panel to 52 clips. The 24 distinct genuine utterances came from the pinned LibriSpeech clean-100 archive. The source qualification rejects previously recorded byte, decoded PCM, script or fragment matches and excludes speakers assigned to the recent trial roles. The wider banks and voice profiles have prior project exposure. For PLN, the qualifier binds the exact publisher sidecar and uses the declared user channel, excluding any other channel's activity. [LibriSpeech terms](https://www.openslr.org/12/) · [PLN source terms](https://huggingface.co/datasets/DuplexGen/duplexgen-spoken) · [Expressive source terms](https://huggingface.co/datasets/laion/elevenlabs-v4-five-language-voice-acting) · [Stock source qualification](audio-short-edit-source-plan-v169.md)

Four archived AMI rooms contribute distinct participant groups and 16 previously unused exact six-second capture intervals. They differ from the two rooms in the discovery cohort. Each room contributes one 0.64-second edit and two interior edits of 0.80 and 0.96 seconds. The eight interior edits fit entirely inside one native two-second incumbent window before speech admission. Two other edits end before the first current window and two begin after the last one. Every edit contains 430–960 ms of simultaneous annotated words from at least two background speakers. The matched clips use the same −5 dB whole-clip background, levels, and 20 ms fades. The samples outside the exact edit are identical within each group. [AMI publisher terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The replication keeps the earlier cue implementation, model bytes, evidence geometry and fixed cutoffs. The WavLM cue runs once on the full original clip and uses remote frame matches to nominate a supported 0.80-second range. The current native comparison scores two-second windows starting at 1 and 3 seconds under the unchanged speech gate. A common-time readout restricts eligible cue ranges to the union of the two-second windows that actually passed speech admission. It keeps the same cue cutoff and the same full-clip feature calculation. This isolates coverage of the reported range while preserving each method's original inference context. [Discovery method and model rights](audio-remote-neighbor-transfer-2026-11-12.md)

A useful location has at least 0.30 temporal intersection over union and covers at least half of the known edit. The fixed cutoffs come directly from the discovery receipts. Retrospective results compare ranking at zero, one, and two false genuine groups. A genuine group counts once if any of its controls flags. Room-only clips form their own false groups. All unavailable outcomes remain visible.

## Held results

| All 12 edits and 40 genuine views | Current | Remote-context cue | Cue within shared time coverage |
| --- | ---: | ---: | ---: |
| AI flags at the frozen point | 3 | 2 | 2 |
| Useful AI locations at the frozen point | 1 | 0 | 0 |
| False genuine groups at the frozen point | 3 | 2 | 2 |
| Useful AI locations at 0 / 1 / 2 equal false groups | 0 / 0 / 1 | 0 / 0 / 0 | 0 / 0 / 0 |
| Useful ranges before applying a point | 5 | 3 | 3 |
| Unassessed files | 0 | 1 | 1 |

On the eight interior edits, the current comparison located one edit at the frozen point and at one or two equal false groups. Both cue readouts located none. All three readouts placed their strongest ranges outside useful support on the four early or late edits even before thresholds were applied. Restricting the cue to the admitted native time coverage changed its best response or range in 13 files and gave the same aggregate fixed and equal-alarm outcomes.

The cue's two fixed AI flags were both on clones and selected ranges outside the useful edit support. Its two false genuine groups included four ordinary splices and one unedited host. No room-only clip flagged, and one room-only clip was unassessed by the cue. Neither the expressive nor stock cases reproduced the earlier fixed gain. The fixed points and implementation were preserved and this replication supplies no support for serving the cue.

## Replay

The CPU run made 52 independent full-clip encoder forwards and 102 admitted incumbent two-second forwards. Native replay reconstructed all 52 exact PCM mixtures and component files. The independent verifier checked actual competing-speech overlap, source intervals, the unchanged thresholds, and both full and restricted time comparisons. Neither verifier reruns model inference.

```sh
base=backend/data/independent-evaluation
run="$base/audio-remote-neighbor-replication-2026-11-12"
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_remote_neighbor_replication audit "$run"
python3 -B -S backend/scripts/review_audio_remote_neighbor_replication.py "$run" \
  "$base/audio-remote-neighbor-final-2026-11-12" \
  "$base/audio-original-pcm-boundary-2026-11-12" \
  "$base/spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip"
```

To reconstruct the run in a new directory, run `freeze`, `neighbor`, `current`, and `report` before both audits. Raw audio, timestamps associated with individual donors, private identities, and scores remain local. [Runner](../backend/scripts/audio_remote_neighbor_replication.py) · [Independent verifier](../backend/scripts/review_audio_remote_neighbor_replication.py)
