# Finding a local move between two recordings

Chipmunk Verify can compare a submitted recording with an earlier copy and show where the same waveform plays earlier or later. The check requires matching decoded clocks and enough unchanged surrounding sound. It reports each clip’s own sample positions and supplies playable excerpts. A matching shift describes the relationship between two files. A conventional edit, source mix or other workflow can cause it. The result does not change automated confidence or establish either recording’s origin. [Held comparison receipt](evaluation-panels/audio-reference-local-shift-2026-11-12.json)

## Test design

The existing soundtrack comparison was frozen before scoring. The study used 32 original 16 kHz mono speech recordings from the University of Sheffield Lombard corpus. Eight recordings from two speakers went to development, and 24 recordings from six other speakers went to held testing. Both plain and Lombard speech were included. The source recordings are released under CC BY 4.0. Original files and derived audio remain local. [Publisher’s release](https://zenodo.org/records/3736465) · [Source qualification and file commitments](lombard-original-mouth-event-2026-11-12.md)

For each original, the test created two 640 ms local moves with 5 ms edge crossfades and an unaffected flank on either side. The signed offset was 160, 200, 240 or 280 ms. A correct result had to overlap at least 60% of the authored interval by temporal intersection over union and recover the signed offset to within 16 samples. Controls included unchanged recordings, local gain and equalization, a constant whole-track offset, a short 40 ms offset outside the target range, an unrelated inserted voice, local silence, silence in both recordings, and identical copies of a recording with a pre-existing repeat. The whole-track and 40 ms controls measure the narrower target of a supported local shift.

The held advancement gate required at least 30 of 48 correct edits, at least 12 correct in each direction, coverage across at least five speaker groups, at most one ordinary control alert, zero alerts on unchanged, repeated or whole-track shifted controls, and assessment of at least 75% of non-silent pairs. There was no threshold fitting. A separate OpenCV calculation confirmed the timing decisions and support counts. Independent FFmpeg readback matched the decoded PCM commitments for all 1,152 pair sides.

## Held results

| Pair delivery | Exact signed offset and sufficient interval overlap | Ordinary controls with a local range | Assessed non-silent pairs | Unassessed both-silent pairs |
| --- | ---: | ---: | ---: | ---: |
| Native PCM WAV to PCM WAV | 34/48 | 0/216 | 240/240 | 24/24 |
| AAC to AAC, research only | 34/48 | 0/216 | 240/240 | 24/24 |

Each arm recovered 18 of 24 later shifts and 16 of 24 earlier shifts across all six held speaker groups. The 34 matched edits had a minimum temporal overlap of 0.75 by intersection over union and zero signed sample error. Fourteen of 48 known moves per arm produced no local range despite sufficient aggregate signal to assess the pair. The development arms both recovered 12 of 16. The same-source WAV-to-FLAC comparisons produced no local range in all 24 held pairs. Cross-carrier WAV-to-AAC had different decoded lengths in 23 of 24 pairs, which were unassessed, and the remaining pair produced no local range.

The app’s narrower timing path accepts native mono 16 kHz signed-16 WAV or FLAC. The app comparison was verified against every held WAV pair, including all 68 playable excerpts from localized results, and every unchanged WAV-to-FLAC pair. A matched WAV-to-FLAC positive reproduced the exact same ranges and two playable excerpts. In that held positive, the submitted excerpt covers **0.560–1.200 s** and the corresponding earlier excerpt covers **0.360–1.000 s**, a **200 ms later** match. This study uses prompted speech with local moves authored against known earlier copies. It gives no detection-rate estimate for unpaired recordings, different sound sources, wider audio formats, or synthesized speech.

## Use it in the local app

Enable `VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED=true` and `VERITAS_AUDIO_REFERENCE_LOCAL_TIMING=true` in the local backend. In the completed analysis, open Media evidence and choose **Compare with an earlier audio recording**. Both files need continuous matching decoded clocks and a duration of 1.5 to 12 seconds. The panel reports when there is insufficient distinctive sound to assess timing.

Local Chrome loaded both playable excerpts and showed their waveforms in the 200 ms example. Its PDF and JSON downloads matched the comparison API response. The PDF shows the two times and sample ranges, and the JSON includes the clips and supporting window positions. Offline replay against the two files confirmed the downloaded JSON. The supplemental comparison leaves the analysis record and billing state unchanged.

## Local replay

Use the source directory from the [Lombard intake and qualification](lombard-original-mouth-event-2026-11-12.md) as `sources` and an unused private directory as `run`. The panel uses FFmpeg and the backend environment, with OpenCV as an independent timing comparator. The receipt’s arithmetic replay uses only Python’s standard library.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.audio_reference_local_shift_panel "$phase" "$run" "$sources" || exit 1
done
backend/.venv/bin/python -B -m scripts.verify_audio_local_shift_panel_app "$run"
python3 -B -S -m scripts.review_audio_reference_local_shift_panel "$receipt" --assemble "$run"
python3 -B -S -m scripts.review_audio_reference_local_shift_panel docs/evaluation-panels/audio-reference-local-shift-2026-11-12.json
```
