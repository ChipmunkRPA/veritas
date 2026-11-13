# Reproduce stereo gain evidence from public recordings

This replay runs the app's fixed-gain and exact stereo repeat checks on original
microphone WAVs from the University of Edinburgh AMI Corpus. It reconstructs
the published comparison for two meetings and adds two earlier unscored
meeting recordings. Reviewers can download all inputs and run the checks on
CPU without private development files, model weights, or app credentials.
[AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) ·
[Original comparison](audio-gain-repetition-transfer-2026-11-13.md)

The source schedule was fixed before the new downloads. Each meeting uses
channels 1 and 5 of Array 1, from the same synchronized tabletop recording.
Twelve three-second windows per meeting start at 75 through 405 seconds in
30-second steps. Each window supplies four known gain or polarity edits, one
exact copy, and six control deliveries. The added meetings use the same
geometry, ratios, and sample tolerance as the original comparison. Recordings
stay in the reviewer's scratch directory. No speech is synthesized. The AMI
Consortium releases the source recordings under CC BY 4.0. The replay derives
stereo excerpts and local audio edits from those original recordings.
[AMI capture notes](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml) ·
[Pinned publisher files and source schedule](evaluation-panels/audio-gain-public-replay-2026-11-13.json)

| Meetings | Gain-adjusted edits localized | Exact-copy controls located by exact channel check | Gain findings on other controls |
| --- | ---: | ---: | ---: |
| ES2004a and IS1009a, original comparison | 96/96 | 24/24 | 0/144 |
| ES2005a and ES2008a, additional recordings | 96/96 | 24/24 | 0/144 |

Neither exact channel nor exact mono located any gain-adjusted target. The
largest boundary error in either cohort was one sample. All 528 deliveries
were assessed, and 88 scheduled deliveries matched the production native stereo
and mono decodes byte for byte. These are constructed copies across four meeting
recordings. Different edits from the same recording are correlated. The replay
measures repetition evidence and adds no AI confidence. A second download into
an empty directory reproduced the full 528-delivery readout digest.

## Reproduce from a clean checkout

Use the repository revision that contains this receipt, the backend Python
dependencies and FFmpeg. The eight publisher WAVs total about 208 MiB. Run the
three stages from the repository root. Keep the scratch directory for audit.

```sh
review_dir=$(mktemp -d)
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze acquire replay; do
  backend/.venv/bin/python -B -m scripts.replay_audio_gain_public "$stage" "$review_dir" || exit 1
done
```

Acquisition checks the size and SHA-256 of every first-party WAV. If the
publisher is temporarily unavailable, start a new scratch directory and use
the original URL and filename pinned in the receipt. The replay checks the
recorded decision totals and complete outcomes. It also publishes a full
readout digest to expose sample-level differences in the FFmpeg mono downmix.
The original environment used FFmpeg 7.0.2 and NumPy 2.2.6.
