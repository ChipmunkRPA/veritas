# Public synthetic-speech transfer panel from Alania

Use the first `cc-by` shard of Alania Turkish Synthetic Speech by
PatientDesk AI, pinned to release revision
`277dfc8a2322014099ff72688bcaee12480c4605`. The publisher's public Hub
metadata and the full downloaded file agree on byte length and SHA-256.
The publisher identifies text-designed synthetic voices and the
commercially permitted CC BY 4.0 domain-text and audio subset. The
generator checkpoint is released under Apache 2.0. The local archive
retains the publisher card, attribution and terms.
[Pinned publisher files](https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr/tree/277dfc8a2322014099ff72688bcaee12480c4605) · [Domain text](https://huggingface.co/datasets/cloud0day3/alania-domain-text-tr) · [Generator card](https://huggingface.co/openbmb/VoxCPM2)

Read metadata columns only. Select native durations from 2.5 through
4.0 seconds, designed voice identities, the publisher's domain-text
source, exact CC BY 4.0 tags and its documented generator and postfilter.
Exclude other text sources and audio-channel variants. Order rows by
a fixed salted hash of the immutable shard digest, row group and ordinal.
Keep 24 rows with distinct voice groups and text documents. No score or
decoded audio enters selection. The roster stores only the minimum private
metadata needed to recover each selected source from the original shard.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_roster_v178 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_alania_roster_v178 verify
```

The panel is a positive-only test on clean, publisher-filtered Turkish
speech from one generator. A later run can report speech-gate coverage,
AI flags and misses for transfer from the English reference. Calibrating
Turkish confidence or measuring false alarms requires a matched genuine
comparison. This stage opens no audio column and makes zero model calls.
