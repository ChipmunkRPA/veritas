# Confirm the short speech review on a separate MP3 panel

The stricter two-second MP3 check found **one of 12** known edits in a separate confirmation panel, compared with **zero** for the four-second check. It flagged one of 16 genuine groups, compared with two for the four-second check. The flagged group was already flagged by the four-second check. [Audited confirmation](evaluation-panels/audio-short-mp3-confirmation-2026-11-12.json)

## Frozen comparison

The 52 recordings come from the earlier call-processing confirmation panel. Exact audio and genuine host identities are disjoint from the codec comparison's calibration and held recordings. Two generated voice profiles appear in earlier studies with separate exact clips. The 12 inserts cover three generator families, and the 40 genuine files include unchanged hosts, ordinary same-reader and other-reader splices, and four isolated room recordings. Each recording was newly encoded using the same 16 kHz 48 kb/s MP3 recipe as the earlier comparison. [Source and rights record](audio-communication-replication-2026-11-12.md)

Each arm uses its stricter MP3 point from the [codec comparison](audio-short-codec-transfer-2026-11-12.md), unchanged. A useful range covers at least half of the edit and reaches 0.30 temporal IoU. The complete four-second and two-second grids were assessed on all 52 files. Neither arm flagged a room-only recording. The two-second arm also gave one generated-file flag that missed its known edit.

| Held comparison | Four-second check | Two-second check |
| --- | ---: | ---: |
| Confirmation edits usefully located | 0/12 | 1/12 |
| Confirmation genuine groups flagged | 2/16 | 1/16 |
| Both panels, edits usefully located at the stricter points | 0/24 | 3/24 |
| Both panels, genuine groups flagged at the stricter points | 2/30 | 2/30 |

The confirmation fell short of the prespecified two-location criterion. Together, the two panels recover three useful locations across three source groups and generator families at the same observed number of genuine-group flags. Adding the review alongside the four-second results would produce one new genuine group over both panels. These are two small authored panels with strong competing speech, and the comparison measures exact ranges under the tested delivery.

## Narrow local scope

The local Full candidate mode uses the stricter point as a separate evidence-only review for the measured MP3 profile. It requires six decoded seconds, one 16 kHz mono stream, a verified 48 kb/s MP3 carrier, and the continuous source clock that maps decoded sample zero to tick 1,105 on the 16 kHz evidence timeline. The browser and export distinguish decoded listening time from source timestamps. The check leaves the main assessment and overall confidence unchanged.

Four local Full worker runs used simulated checkout. The two known positives showed 3–5 and 4–6 seconds, the genuine control showed no flag, and a 64 kb/s control was skipped. Chrome verified the playable two-second excerpt and source ticks `[49105, 81105)` for the 3–5 second case. The Standard PDF embeds the exact marked waveform from the JSON evidence. Signed source verification regenerated the waveform and matched the clip bytes against the uploaded MP3's decoded PCM.

## Replay

The public standard-library reader compares the points to the earlier receipt and recomputes coverage, useful locations and genuine-group outcomes. Native replay reopens all 52 carriers and checks 364 exact window bindings.

```sh
python3 -B -IS backend/scripts/review_audio_short_codec_transfer.py \
  docs/evaluation-panels/audio-short-mp3-confirmation-2026-11-12.json --confirmation
```

With the retained audio sources and pinned installed checkpoints, use a new private run directory.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_short_mp3_confirmation prepare --run "$run"
backend/.venv/bin/python -B -m scripts.audio_short_mp3_confirmation score --run "$run"
backend/.venv/bin/python -B -m scripts.review_audio_short_codec_transfer "$receipt" --confirmation-native "$run"
```
