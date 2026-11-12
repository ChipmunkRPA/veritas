# Same-speaker source qualification for short voice edits

The earlier speaker-enrollment test improved separation on genuine speech but attenuated generated inserts spoken in a different voice from the enrolled speaker. This source preparation checks whether the retained archives can support a separate comparison in which the clone targets the enrolled reader. It freezes the comparison sources before any authenticity or acoustic score. [Earlier test](audio-enrolled-target-transfer-2026-11-12.md)

## Rights and source identity

The publisher assigns MIT terms to generated dialogue audio in the DuplexGen `PLN` planning split. The release uses Chatterbox for isolated utterances and includes the prompt speaker ID in a publisher sidecar. The renderer derives that ID from LibriSpeech directories. Only user turns are eligible. The qualifier verifies the exact sidecar, speaker number, turn transcript, and native stereo channel assignment. Some per-turn archives contain other-speaker activity on the other channel, so the donor uses only the exact sidecar-designated user channel. The retained prompt path was stripped from the archive. Speaker identity here is based on publisher provenance. [DuplexGen audio card](https://huggingface.co/datasets/DuplexGen/duplexgen-spoken) · [Publisher data licenses](https://github.com/duplexgen/duplexgen-code/blob/main/docs/DATA_LICENSES.md) · [Renderer and speaker provenance](https://github.com/duplexgen/duplexgen-code/blob/main/tts_render/convert_spoken.py)

Both genuine utterances use the same LibriSpeech speaker number, with different chapter and utterance IDs. The original enrollment utterance lasts four to five seconds and the derived reference is exactly four seconds. The published renderer requires a synthesis prompt longer than five seconds, giving a separate-source condition for the enrollment under the documented pipeline. The exact original prompt utterance is unavailable, so overlap between it and the longer genuine host remains unknown. LibriSpeech is CC BY 4.0. Media, transcripts, utterance hashes, and speaker pairings stay in the local run. [LibriSpeech publisher and terms](https://www.openslr.org/12/) · [Publisher-pinned acquisition and prior source history](audio-calibration-data-preparation-v68.md)

## Frozen source procedure

The plan fixes the source order by a salted hash before decoding audio. It assigns ten metadata candidate speakers to selection and sixteen to confirmation, each with a distinct selected clone dialogue and script. Each speaker belongs to a single role. For a qualified group the first eligible genuine host and enrollment come from different chapters. A clone matches its host and enrollment through the same publisher-declared LibriSpeech speaker ID. Eligibility uses declared metadata, integrity, source duration, and simple signal energy.

The exact overlap screen binds the project's v58 history, v81 qualified source inventory, GRID source receipts, v72 and v82 manifests, the later qualified v169c foreground receipts, and the two preceding long-clip and enrollment rosters. It rejects recorded source IDs, exact text or eight-word prefix, exact original or derived WAV bytes, and decoded PCM or recorded prefix matches. The banks and some speaker identities or dialogue contexts have appeared earlier in the project. This comparison qualifies new source excerpts within those banks. It cannot establish whether the recordings entered an upstream training corpus.

For every accepted group, the qualifier reads the exact Parquet or isolated archive member under the publisher-pinned whole-archive hash. It checks the FLAC sample count or WAV header, verifies the genuine mono clock and the clone's isolated user channel, and decodes with the app's ordinary PCM path. It retains eight seconds of continuous genuine host material, four seconds of enrollment, and three seconds of cloned donor. Each retains a zero-based exact sample range at 16 kHz, and a separately decoded WAV reproduces its samples and clock. A future trial can reserve the first two seconds of the host for an ordinary same-host control, then use the remaining six seconds as the host. It must still freeze exact splices and background captures.

## Qualified groups

Fourteen distinct speaker groups qualified from 17 attempted, producing 42 exact WAVs. Three groups did not produce a duration- and chapter-qualified enrollment and host pair. Nine additional metadata alternates were unnecessary. A separate receipt review rehashed all source WAVs and their decoded PCM, checked every clock and role, checked the saved source histories, and verified that the four reserved room cohorts contain 16 distinct annotated participants. [Aggregate qualification receipt](evaluation-panels/audio-same-speaker-clone-sources-2026-11-12.json)

| Role | Qualified speakers | Reserved AMI sessions |
| --- | ---: | --- |
| Selection | 6 | ES2004a (3), IS1009a (3) |
| Confirmation | 8 | ES2010d (4), IS1007a (4) |

All 14 speaker identities had recorded project exposure. Eight selected donor dialogue contexts were previously known. The exact selected media and scripts passed the frozen exclusions. Two generated archive WAVs contained activity in their other native channel, which was excluded from the qualified donor. The source preparation covers one generator family. No room mixture, detector score, or change to the app follows from this qualification alone.

## Local replay

```sh
run=backend/data/independent-evaluation/audio-same-speaker-clone-sources-final-2026-11-12
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_same_speaker_clone_sources verify "$run"
```

Use a new directory and run `freeze` followed by `qualify` to reconstruct a separate receipt. Replay reads the original retained archives, requalifies the selected slots and checks every generated source WAV byte for byte. [Qualification runner](../backend/scripts/audio_same_speaker_clone_sources.py)
