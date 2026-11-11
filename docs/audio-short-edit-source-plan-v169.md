# Fresh source roster for ordinary short speech edits

The local GRID run found false AI tendencies after copying genuine audio
within the same recording. This roster supports a new short-speech comparison
with separate development and holdout roles. It sets aside genuine speech,
cloned speech, expressive synthesis and stock synthesis before any score is
available. The same source and background remain paired across ordinary edit
and noise conditions. [GRID result](grid-serving-completion-result.md)

The builder uses recorded local publisher receipts and the following source
banks. It checks each original archive's presence and saved size before
selection. Selected archive bytes and production PCM must pass the next
qualification step.

| Lane | Source and retained rights declaration | Identity binding |
| --- | --- | --- |
| Genuine | LibriSpeech clean 100-hour Parquet and the four archived train.360 shards, CC BY 4.0 | Declared reader, utterance and normalized text |
| Clone | DuplexGen PLN publisher-verified tar files, publisher-declared MIT PLN terms | Declared LibriSpeech donor, original dialogue, archived utterance and script |
| Expressive | LAION ElevenLabs generation archive, CC BY 4.0 | Published generation script, challenge record, encoded bytes and declared stock voice |
| Stock | Publisher-pinned English Kokoro directed-test Parquet, CC BY-SA 4.0 | Both text fields, encoded bytes, original script and declared stock voice |
| Background | MUSAN noise/free-sound inventory, recorded public-domain declaration for this subset | Original bytes and recorded decoded PCM |

The source-rights declarations apply to the recorded assets and their stated
terms. The source selection establishes recorded corpus ancestry and
comparisons with known prior inputs. Model pretraining overlap and biometric
identity remain unknown. The expressive and Kokoro banks retain previously
seen voice profiles. Their new encoded bytes and text are checked separately
from voice exposure. Keep recordings, transcripts, archive members and
individual sample hashes private. [Source terms and history](audio-calibration-data-preparation-v68.md)

Before role assignment, exclude known byte, text and speaker links in the
historical v58 catalogue and later consumed v81 bank. The serving calibration
binds the historical v3 fit. Exclude the source and WAV byte hashes and the
decoded PCM hashes for all 32 GRID originals. Bind each GRID publisher source
and member identity and exclude the eight exact publisher-matched reference
prompts. Exact decoded duplicate checks apply during qualification, when the
selected source bytes and samples become available.

Within each lane, group all known voice, donor, dialogue, source and normalized
text or eight-word-prefix connections. Allocate each entire group to one role
using a fixed salted metadata order. Start with holdout and alternate. A
candidate connected to an earlier lane is excluded before assignment.
The fixed roster asks for up to two genuine recordings and one AI recording
per connected group. It records at most six pending candidates per group in
deterministic order. A candidate can be skipped only for pre-score provenance,
duration, integrity or exclusion checks. A missing source or failed window
stays visible. Holdout source outcomes stay unused during development.

The planned source prefix is 47,648 samples at 16 kHz, matching the GRID
short-audio duration. Use 0 and +5 dB noise conditions with four distinct
noise parents per role. Exact source prefix, background assignment and
case transformation bytes need their own prospective run receipt before
scoring.

Run from the repository root.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_sources_v169 freeze
```

The private roster goes under ignored independent-evaluation data. The public
summary contains only counts and whole-artifact commitments. The roster freeze
performs no audio decoding, generation, detector inference or fitting.
