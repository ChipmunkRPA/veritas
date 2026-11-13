# Named-generator source census for short English edits

The PhonemeDF authors pair LibriSpeech audio with independently synthesized
versions of the same text and publish a distinct MeloTTS archive. The paper
states that MeloTTS used its built-in voices. The WAV and alignment member
identifiers identify the LibriSpeech prompt. They do not identify which
built-in MeloTTS voice was used. The corpus release declares CC BY 4.0,
LibriSpeech declares CC BY 4.0, and the MeloTTS publisher declares MIT.
This audit uses released samples and no new generator checkpoint.
[PhonemeDF release](https://zenodo.org/records/19522979) ·
[Construction and voice selection](https://arxiv.org/html/2603.15037v1#S3.SS1) ·
[LibriSpeech terms](https://www.openslr.org/12/) ·
[MeloTTS terms](https://huggingface.co/myshell-ai/MeloTTS-English)

The first 96 MiB of the MeloTTS archive had already been used for
whole-utterance testing. This census extended the bound to 320 MiB and
matched each publisher member to the existing 320 MiB genuine prefix.
The complete TextGrid archive replayed against the publisher MD5. It
required matching full word tiers on each native clock and the same
two- or three-word short-edit geometry, duration ratio and speech activity
conditions as the [earlier short-edit intake](audio-phoneme-short-edit-intake-2026-11-13.md).
The census records geometric eligibility only. No splices were scored.

| Source qualification | Count |
| --- | ---: |
| Complete native MeloTTS WAVs | 1,225 |
| Genuine/MeloTTS prompt pairs | 1,068 |
| Pairs with complete matching word tiers | 1,067 |
| Pairs with an eligible short-edit location | 914 |
| Declared LibriSpeech readers across the paired inputs | 11 |
| Eligible readers outside both checked earlier source plans | 0 |

The archive prefix contains 12 declared LibriSpeech readers overall and
11 among the paired inputs. All but one paired reader occur in the saved
v9 source plan. The remaining reader occurs in the older v172 short-edit
development source plan. The census checks both exact local plans against
their published SHA-256 values. A second native replay produced the same
receipt byte for byte. The [scalar receipt](evaluation-panels/audio-melo-short-source-census-2026-11-13.json)
contains the byte-range commitments and counts per anonymous source reader.
It contains no native media or text. [V9 source commitment](evaluation-panels/audio-v9-execution-freeze.json) ·
[V172 source commitment](evaluation-panels/audio-short-edit-case-freeze-v172.json)

## Separate confirmation

Keep these MeloTTS inputs as a known English source for controlled
diagnostics. For a fresh generator-held English bank, select lawful
genuine speech from a separately released corpus, such as the CC BY 4.0
[Multilingual LibriSpeech](https://www.openslr.org/94/), and first exclude
recorded speaker, audiobook, chapter, text and original PCM matches
against previous roles. A different corpus name alone does not establish
independence.

Generate matched short phrases locally using at least two model families
with an explicit stock-voice identity in the output manifest. For example,
the [KittenTTS Nano 0.8 checkpoint card](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32)
lists Apache-2.0 for the weights, separate from the project's code license.
Before use, pin the checkpoint revision and its exact assets and reserve
its generator family and chosen voice IDs from all development and
calibration roles. KittenTTS and Kokoro both use StyleTTS2, so use a
different architecture if the claim is architecture-held rather than
checkpoint-held. A second separately pinned stock-voice family is still
needed.

Reserve [AMI](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
meeting groups with no known overlapping participants or recording
sessions. Freeze matched unedited hosts, genuine resampling or splice
controls, dry generated inserts and isolated background controls. Give
the same matched foregrounds the same background samples and gain at 0
and −5 dB. Store exact native positions, generator configuration and
voice IDs, room and capture identities and source commitments before
scoring. Keep independent group-level holdout counts separate from
multiple renders of each source. No new model, room or English
confirmation bank was acquired beyond the bounded PhonemeDF source check.

## Native replay

Download the first 40 consecutive 8 MiB MeloTTS ranges using the publisher
URL below and keep their response headers. Each range is named
`OFFSET.range` and its header `OFFSET.headers`. The genuine and TextGrid
cache layout follows the [prior public replay](audio-phoneme-short-edit-intake-2026-11-13.md#replay).
Native ancestry comparison also requires the two local prior plans named
in the reviewer. The public summary preserves their hashes. The local
plans can contain source paths and are not part of this publication.

```sh
end=$((offset + 8388607))
curl -fSsL --max-filesize 8388608 -H 'Accept-Encoding: identity' \
  --range "$offset-$end" -D "$melo/$offset.headers" \
  "https://zenodo.org/records/19522979/files/PhonemeDF_MeloTTS.tar.gz?download=1&chipmunk-part=melo_confirmation_v1_$offset" \
  -o "$melo/$offset.range"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_audio_melo_short_sources \
  --melo "$melo" --cache "$new_cache" --old-cache "$old_cache" \
  --compare docs/evaluation-panels/audio-melo-short-source-census-2026-11-13.json
```
