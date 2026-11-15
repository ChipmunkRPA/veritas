# Temporary phase changes in background sound

The local trial checks a single recording for a temporary change shared
by two narrowband tones. It marks an approximate time range, shows the
waveform and spectrogram, and plays the exact submitted audio for that
range. The evidence lists both frequencies and the measured phase
offsets. The PDF and JSON use the same source coordinates. The check
carries zero AI-confidence weight. Changes in recording speed or the
background itself can produce the same pattern.

## What the check measures

The check projects audio onto the first four harmonics of nominal 50 Hz
and 60 Hz families in overlapping 320 ms Hann windows, spaced 40 ms
apart. Samples on both sides of each proposed range provide a continuous
phase reference. The two tones must be distinct from nearby frequencies,
show consistent interior offsets, and return to the surrounding phase
relationship. This borrows ideas from phase-based electrical-network
frequency forensics. The implementation uses neither a grid reference
nor model weights. It measures tones present in the mixed audio.
[Phase-discontinuity study](https://aquarius.ime.eb.br/~apolin/papers/IEEETIFS2010Daniel.pdf)
· [Harmonic analysis study](https://eurasip.org/Proceedings/Eusipco/Eusipco2013/papers/1569728299.pdf)

Enable `VERITAS_AUDIO_TONAL_PHASE_REVIEW=true` for a local trial. The check
accepts complete 3–12 second 16 kHz mono PCM16 WAV or FLAC files on a
continuous zero-origin timeline. It reports separately when the tones
could not be measured, when measured ranges had no supported change, and
when the check selected a range. The range follows the 40 ms analysis
grid. The underlying projection windows span a wider interval.

## Held comparisons

The initial study used 228 development clips with speech from eight SINE
readers. The first held set used four different readers and 120 trials.
The broader single-tone rule located 6/24 staged changes and flagged 1/96
controls. The two-tone rule selected no location. Both operating points
were frozen for transfer. Recorded or constructed tone sources are pinned
by SHA-256. The first panel used two development sounds and two held
sounds by different contributors. [SINE source and replay](audio-sine-resynthesis-transfer-2026-11-14.md)
· [Omegatron, CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Mains_hum_60_Hz.ogg)
· [Rayo v.2.0, CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:60-hz.wav)
· [Julo, public domain](https://commons.wikimedia.org/wiki/File:MainsBrum50Hz.ogg)
· [BPK, public domain](https://commons.wikimedia.org/wiki/File:60Hz_hum_square.ogg)

The transfer used 24 different LibriSpeech reader IDs and two previously
unused field recordings of hum. Rain, train sound or competing speech
was mixed 5 dB above the main voice. The added tone was 7 or 13 dB below
the main voice. The 144
positive clips include 72 shifts in the tone alone and 72 shifts in the
whole mixture. Inserted intervals last 560–1,040 ms. The 612 controls
include continuous tones, unmodified natural speech, isolated backgrounds,
gain, low-pass filtering, a speech-only splice, whole-track delay, and
gradual variation in tone timing. Useful localization covers half the
known interval and has at least 0.30 temporal intersection over union.
[LibriSpeech, CC BY 4.0](https://www.openslr.org/12)
· [Pinned LibriSpeech shards](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/all/train.clean.100)
· [DAAyer, CC BY 4.0](https://freesound.org/people/DAAyer/sounds/513490/)
· [namikiri, CC BY 4.0](https://freesound.org/people/namikiri/sounds/785990/)

| Transfer check | Useful staged locations | Both boundaries within 120 ms | Control flags |
| --- | ---: | ---: | ---: |
| Existing mixed-spectrum transition check, using paired edges | 0/144 | 0 | 0/612 |
| Single tone | 2/144 | 1 | 40/612 |
| Single tone with outer persistence | 2/144 | 1 | 15/612 |
| Two related tones | 12/144 | 5 | 7/612 |

The two-tone check could evaluate 53 of 144 staged changes. The other 91
lacked a usable comparison. All 12 useful ranges came from the same
transformer recording, across rain and competing speech. None came from
the second hum recording or train mixtures. The seven control flags came
from the 72 deliberately smooth timing changes. The other 540 controls
produced zero flags. This supports a narrow, descriptive trial. It leaves
short voice edits under noise and recordings with no measurable tone unresolved.
The staged positives test acoustic localization, with no AI-generated label.
[Initial study receipt](evaluation-panels/audio-tonal-phase-transfer-2026-11-14.json)
· [Transfer receipt](evaluation-panels/audio-persistent-tone-transfer-2026-11-14.json)

## Local application verification

Six USD 9.99 simulated checkouts ran three inputs with the trial off
and on. No job started before simulated payment confirmation. All
three pairs had identical overall assessments. The three enabled runs
exercised a measured change, no supported change, and unassessed tones.
The measured range was 1.560–2.080 seconds. Its playback matched the
submitted samples, and exact waveform and spectrogram assets appeared
in the signed report. The evidence API, JSON and report snapshot agreed.
The signed packages also replayed against the original recordings.
[Local checkout receipt](evaluation-panels/audio-tonal-phase-worker-2026-11-14.json)

## Replay

The receipts carry hashes, case definitions and numeric measurements.
Use `review` with standard-library Python to reconstruct the counts.
Native replay decodes the pinned sources, rebuilds the mixtures and
compares measurements byte for byte. Use the bank preparation in the
[SINE study](audio-sine-resynthesis-transfer-2026-11-14.md) and the pinned
background sources in the [focused speech study](audio-focused-window-transfer-2026-11-12.md).
Put the four Commons originals and two exact Freesound previews in
`$hums`. Put the first two train-clean-100 parquet shards in `$shards`
with the filenames and hashes in the transfer receipt. `$sounds` holds
the pinned `rain.ogg`, `train.ogg` and `competing_speech.ogg`.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
initial=docs/evaluation-panels/audio-tonal-phase-transfer-2026-11-14.json
transfer=docs/evaluation-panels/audio-persistent-tone-transfer-2026-11-14.json
python3 -B -S -m scripts.audio_tonal_phase_transfer review "$initial"
python3 -B -S -m scripts.audio_persistent_tone_transfer review "$transfer" --parent "$initial"
backend/.venv/bin/python -B -m scripts.audio_tonal_phase_transfer freeze "$run/initial.json"
backend/.venv/bin/python -B -m scripts.audio_tonal_phase_transfer develop "$run/initial.json" \
  --bank "$bank" --hums "$hums" --sounds "$sounds" --output "$run/development.json"
backend/.venv/bin/python -B -m scripts.audio_tonal_phase_transfer held "$run/development.json" \
  --bank "$bank" --hums "$hums" --sounds "$sounds" --output "$run/initial-replay.json"
cmp "$run/initial-replay.json" "$initial"
backend/.venv/bin/python -B -m scripts.audio_persistent_tone_transfer freeze "$run/transfer.json" \
  --parent "$initial" --shards "$shards"
backend/.venv/bin/python -B -m scripts.audio_persistent_tone_transfer run "$run/transfer.json" \
  --parent "$initial" --shards "$shards" --hums "$hums" --sounds "$sounds" --output "$run/transfer-replay.json"
cmp "$run/transfer-replay.json" "$transfer"
backend/.venv/bin/python -B -m scripts.audio_tonal_phase_app \
  "$run/serving" "$transfer" "$initial" "$shards" "$hums" "$sounds"
```
