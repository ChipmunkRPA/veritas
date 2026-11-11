# Matched foregrounds with six competing speakers

This development comparison follows the false alarms observed on genuine
speech with competing speakers in PACC-P. Freeze 16 matched English
utterance pairs from the v81 fitting source bank. Each pair has a genuine
LibriSpeech recording and stock synthetic speech generated from the same
intended text. Both retain their first 48,000 production mono signed-16-bit
samples at 16 kHz. Speaking rate may change which words fall inside those
three seconds. The v81 foregrounds have prior development exposure. Four
connected groups of four pairs each retain one fixed babble pool per group.

Use four separate pools of six declared VCTK speakers. Candidate recording
identities come from a pinned DFADD bona fide archive index and a pinned
VCTK transcript mirror. The archive publisher does not document which
microphone supplied its WAVs. The freeze commits to the original cached
byte identities, source and license records, and deterministic talker
ranking. VCTK recordings are available under CC BY 4.0, with attribution
to Christophe Veaux, Junichi Yamagishi and Kirsten MacDonald at the
University of Edinburgh. The DFADD repository carries an MIT license.
LibriSpeech material remains attributed under CC BY 4.0. The pinned stock
generator and voice profiles carry Apache 2.0 terms, and their input text
retains source attribution. [VCTK publisher and license](https://datashare.ed.ac.uk/handle/10283/3443) · [Pinned DFADD archive index](https://huggingface.co/datasets/isjwdu/DFADD/tree/dfc1eeab3cb0068db8e87a2b89a1ebd103665b1f) · [LibriSpeech](https://www.openslr.org/12/) · [v81 qualification](audio-complete-sources-v81-result.md)

Exclude declared voice and available raw byte and exact PCM identities
from the v172 holdout source receipts and the v177 PACC-P observation
inputs. Production full and 3-second PCM are checked before use. VCTK
babble parents also exclude known v58 original byte and PCM hits. The
PACC-P publisher identifies its babble source as MUSAN LibriVox, a separate
source corpus. These checks cover the available recorded identities.
Voice aliases, broader corpus exposure and training overlap can remain
unknown. Holdout metadata is read solely to construct the exclusion lists.
No holdout recording, model, calibration or production setting is opened
by this preparation. [PACC-P sample and attribution](audio-pacc-sample-v176-method.md)

Bring the six verified three-second talker prefixes to equal whole-prefix
RMS, sum them at sample zero, then scale and round the sum to signed 16-bit
PCM with fixed headroom. No room simulation, replacement or padding is
used. For each foreground pair, apply that pool at nominal 0 and +5 dB
relative to the genuine foreground over the full three seconds. Retain both
clean arms and use a single gain across all six cases. Quantize foreground
and babble separately so each condition adds byte-identical sample values
to both arms. Record each arm’s measured SNR because foreground levels can
differ. Genuine background speech remains a separate component from the
foreground origin label. Verify every WAV’s exact length and PCM byte
round-trip.

The denominator is 96 cases, with 32 per condition and 16 per condition
and foreground class. Preserve unavailable sources, pools and cases in
private receipts. The worker opens only direct paths in admitted
development source banks. Plan and readout publish aggregate counts and
artifact commitments. Source identities, voices, transcripts, WAVs and
per-case receipts stay private. Shared foreground groups and babble pools
make the cases dependent. This panel prepares one English synthetic family
for controlled development comparisons and does not estimate population
error rates.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_matched_babble_v181 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_matched_babble_v181 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_matched_babble_v181 readout
```
