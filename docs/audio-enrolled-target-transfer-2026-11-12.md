# Speaker-enrolled separation under overlapping room speech

A separate clean utterance improved extraction of the enrolled voice on held genuine speech. It also attenuated the short generated inserts by a median of 5.07 and 6.73 dB in the two held rooms. The approach failed its frozen preservation requirement. The audio authenticity check was not run on these new mixtures, so no fixed-point or equal-false-alarm comparison is available. [Aggregate results](evaluation-panels/audio-enrolled-target-transfer-2026-11-12.json)

## Sources and scope

The earlier long-clip panel supplies six genuine selection episodes and nine held edit episodes. Each held episode contains a 0.8-second synthetic splice, a matched same-host ordinary splice, and a matched other-voice ordinary splice. The nine edited hosts and nine donor groups are distinct. There are three donors each from the clone, expressive, and stock families. This study uses the existing qualified source assignments and does not claim fresh foreground speakers or generators. It renders all 43 six- or eight-second mixtures anew, with 16 distinct background participants in four AMI meetings and exact captures distinct from the archived tested intervals. The two held rooms differ from both the selection rooms and the earlier long-clip held rooms. [Earlier long-clip test](audio-spectra-long-room-holdout-2026-11-12.md) · [Donor source qualification](audio-short-edit-source-plan-v169.md) · [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The recorded room speech is 5 dB louder than the unedited foreground across the whole clip. AMI word alignments put at least two simultaneous recorded background voices inside every synthetic edit. Overlap lasts 490–800 ms per 800 ms edit and totals 6.13 of 7.20 edited seconds. The −5 dB mix setting does not assert the same instantaneous ratio throughout each splice. [AMI annotations](https://groups.inf.ed.ac.uk/ami/download/)

True enrollment uses four seconds from a different utterance with the same canonical LibriSpeech speaker and a different transcript. Sham enrollment uses a different speaker from the same role, also distinct from the ordinary donor. Four genuine speakers outside both test roles supply a common generic speech dictionary. The selection hosts, held hosts, and generic roles share no declared speaker identity. Twenty-one exact enrollment cuts serve the full panel. The four isolated-room controls have no target reference and are recorded as unavailable for this check. LibriSpeech and AMI source recordings, evaluation donor media, and exact enrollment samples remain local. [LibriSpeech terms](https://www.openslr.org/12/) · [Source qualification](audio-complete-sources-v81-result.md)

## Frozen acoustic comparison

Speaker-conditioned source extraction typically supplies separate enrollment speech in addition to the mixture. Here a small deterministic spectral dictionary replaces a learned extractor. This differs from the previous onset-conditioned spatial check because it uses a separate clean utterance and only the mono mixture at inference. Twelve prototypes describe the enrolled speaker and 24 describe generic speech. A 512-sample short-time transform, 128-sample hop, and 28 fixed nonnegative fitting iterations produce a target mask applied to the original mixture phase. The true and sham arms use identical generic prototypes. Ground-truth stems only measure how much foreground and background the frozen mask passed. They never enter mask estimation. [VoiceFilter reference-conditioning paper and project](https://google.github.io/speaker-id/publications/VoiceFilter/) · [Prior onset-conditioned test](audio-target-onset-transfer-2026-11-12.md)

Qualification requires true enrollment to beat sham in two-thirds of held genuine same-host edits, a median exact-edit SNR gain of at least +0.5 dB in each held room on those genuine edits, and median generated foreground retention of at least −3 dB during the edits in both rooms. The original PCM and exact edit ranges are retained throughout. The two controls and the generated arm for each held episode use identical background PCM and match sample-for-sample outside the splice.

| Held room | Edits | Same-host SNR gain, true / sham | Other-voice SNR gain, true | Generated SNR gain, true | Generated foreground retained, true |
| --- | ---: | ---: | ---: | ---: | ---: |
| ES2010d | 5 | +4.16 / −0.29 dB | +3.49 dB | +3.54 dB | −5.07 dB |
| IS1007a | 4 | +0.91 / −1.88 dB | +1.12 dB | −0.94 dB | −6.73 dB |

True enrollment beat sham in eight of nine same-host controls. The genuine SNR gate passed in both rooms and the generated retention gate failed in both. True enrollment also attenuated the other genuine donor by a median of 2.89 and 4.84 dB in the two rooms. A change in extraction response can follow an ordinary speaker change, so the response alone has no established interpretation as an authenticity cue. The tested donors have different declared identity from the host. Exact edits made by cloning the enrolled host remain untested. Even a useful target-isolated listening view would require the original available alongside it to avoid hiding another voice during the edit.

A metadata-only inventory of the 14 retained LibriSpeech `train.clean.100` shards found 83 declared LibriSpeech speaker IDs with at least two genuine utterances of 14 words or more and at least one archived PLN clone candidate. These matches account for 416 clone candidates of at least three seconds whose archive members had no recorded prior use in the saved index. A same-person trial would still need exact payload, decoded-PCM, script, and prior-exposure checks, then role assignment by shared speaker identity before scoring. This candidate inventory covers one generator family. [Publisher-pinned archive inventory and terms](audio-calibration-data-preparation-v68.md) · [Prior source selection policy](audio-short-edit-source-plan-v169.md)

## Reproduction

The native audit rebuilds the 43 mixtures and 21 enrollments and recomputes every acoustic cell. The independent standard-library review checks input and reference receipts, identity and role separation, original PCM and component reconstruction, sample identity outside the known edit, two-person background overlap, reported medians, and the frozen gate. It does not reimplement spectral estimation. No neural forwards or production changes were made. The existing source banks and meeting recordings have prior project exposure. The exact cuts and mixtures in this trial are new, and model pretraining exposure is unknown.

```sh
base=backend/data/independent-evaluation
run="$base/audio-enrolled-target-2026-11-12"
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_enrolled_target_transfer audit "$run"
python3 -B -S backend/scripts/review_audio_enrolled_target_transfer.py "$run" \
  "$base/spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip" \
  "$base/audio-complete-sources-v81/manifest.json" \
  "$base/audio-spectra-long-room-final-2026-11-12/roster.json"
```

To reconstruct a separate receipt from the retained sources, run the native module with `freeze` and then `measure` on a new local directory before running both audits. The public JSON contains aggregates and run commitments without utterance- or source-level entries.
