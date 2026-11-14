# Short stock-voice splices under competing speech

This local test examines how often the existing five-window speech review
finds a 960 ms synthetic insert in a six-second recording. It repeats
matched foreground versions dry, with recorded competing speech at 0 dB,
and with louder competing speech at −5 dB. The two-second review range is
an inspection lead. The test records a range as useful when it covers at
least half the known edit and its temporal intersection over union is at
least 0.30. No point was fitted on these inputs. [Earlier same-word
speech test](audio-phoneme-short-edit-score-2026-11-13.md)

## Inputs and separation

The English Multilingual LibriSpeech test shard yielded 3,769 metadata
rows from 42 declared readers across 58 books. Bounded Parquet ranges
provided source audio from four row groups, decoded to a continuous 16 kHz
mono clock. Eight held readers contribute
one untouched recording, one ordinary splice using another read from the
same speaker, one ordinary splice using a different genuine reader, and
two local stock-voice versions. Four additional readers supply the
ordinary different-speaker fragments. The 12 host and donor readers use
12 distinct books within this panel. The selected 24 source waveforms
have distinct native PCM fingerprints and no matching source fingerprint
or complete transcript in the two audited earlier source plans. The MLS
paper describes its English test split as disjoint from the earlier
LibriSpeech development and test splits. The selected numeric reader IDs
also have zero matches among the 322 and 12 known LibriSpeech reader IDs
in those two local source plans. These checks establish evaluation
ancestry within the inspected materials. Model pretraining exposure and
aliases in other corpora remain unknown. [MLS release, CC BY 4.0](https://www.openslr.org/94/)
· [English release construction, §4.1 and Figure 4](https://arxiv.org/pdf/2012.03411)
· [Bounded English mirror](https://huggingface.co/datasets/parler-tts/mls_eng)
· [Earlier source-plan commitments](audio-melo-short-source-census-2026-11-13.md)

The positive edits use Piper LJSpeech Medium, a VITS checkpoint trained
from scratch on public-domain speech, and KittenTTS Nano 0.8 INT8, a
StyleTTS2 checkpoint with Apache-2.0 weights. Each generator uses one
stock voice and newly authored short prompts. No host voice was cloned.
Piper and Kitten are distinct generator checkpoints and use different
architectures from each other. The prior source plan includes StyleTTS2
and Kokoro, so Kitten shares architecture lineage with prior work. The
Piper GPL runtime was used only as a local generator tool. The six
synthesis artifacts were hash-pinned. The frozen Kitten outputs are one
stochastic draw that was saved before inference. A separate seed probe
reproduced the exact bytes of a new prompt across three sessions.
[Piper checkpoint and training grant](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ljspeech/medium/MODEL_CARD)
· [Piper runtime and voice rights](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/COMMERCIAL_USE.md)
· [KittenTTS checkpoint and grant](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8)

Two declared-participant-disjoint AMI sessions provide the recorded
maskers. Exact 16 kHz PCM byte ranges were checked against the native WAV
headers. The session participant sets had no overlap with the 18 named
AMI sessions used for the prior room and voice source checks. Manual
annotations show at least 340 ms of overlapping background speech at
both candidate edit clocks in all selected masker excerpts. A further
four isolated background excerpts check false alarms. The same PCM
background samples and gain are applied across a host's matched
foreground versions. The foreground and background are independently
rounded to signed 16-bit values, then added without clipping. No modeled
room channel was added. [AMI corpus and usage terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The fixed destinations are **1.450–2.410 s** and **3.450–4.410 s**. They
alternate between hosts. Every localized ordinary or synthetic version
uses the same 20 ms fades and matches the source outside its authored
interval to the sample, including after mixing. There are 48 positive
and 72 matched ordinary deliveries from eight host groups. Four
background-only controls bring the total to **124**. Repeated conditions
from one host remain a single source group when counting distinct
findings.

## Comparison

The existing optional review uses five overlapping two-second windows,
starting at 0 through 4 seconds, and its existing PCM/FLAC point. The
earlier sparse review uses windows **1.000–3.000 s** and
**3.000–5.000 s** at its prior point. A second sparse readout uses the
existing five-window point to isolate the effect of sampling. All use the
same existing classifier and speech admission logic and select the
strongest admitted response. The paired ranking test compares the highest
responses on jointly admitted windows covering enough of the edit, with
the corresponding ordinary different-reader splice as the control.

Both authored edit positions lie near the centers of the earlier sparse
windows. The sampling ablation therefore favors that geometry. Its
result cannot establish how sparse windows would behave when edits fall
between them. Separate random-clock clips are needed to measure that
tradeoff. The synthetic clips also change speaker and words. The
ordinary different-reader splice measures some of that confound, while
same-speaker generated replacements remain a separate harder task.

## Results

Each cell gives useful locations out of 16 positive deliveries and false
flags out of 24 ordinary deliveries. All 124 deliveries had admitted
speech, and none of the four isolated background excerpts flagged.

| Foreground to background | Existing five-window review | Earlier sparse point | Sparse at existing review point |
| --- | ---: | ---: | ---: |
| Dry | 11/16 · 1/24 | 16/16 · 3/24 | 11/16 · 1/24 |
| 0 dB | 5/16 · 7/24 | 10/16 · 9/24 | 7/16 · 2/24 |
| −5 dB | 3/16 · 4/24 | 8/16 · 7/24 | 3/16 · 0/24 |

Across all three conditions, the five-window review located 19/48
synthetic deliveries, selected the wrong interval on seven, and gave no
flag on 22. The earlier sparse point located 34, selected the wrong
interval on three, and gave no flag on 11. It flagged 19 ordinary
deliveries across six host groups, compared with 12 across five groups
for the existing review. At −5 dB, it found four Piper and four Kitten
edits. The five-window review found one and two, respectively.

At the existing point, the sparse grid located 21/48 and produced zero
wrong-window flags. It flagged three ordinary deliveries across three
host groups. Under −5 dB, its three correct windows were **3.000–5.000 s**
for edits at **3.450–4.410 s**. All three also appeared in the existing
review. At this noise level the positive edit outscored its matched
ordinary different-reader splice in 13/16 jointly admitted comparisons.
The analysis could use better placement and score discrimination under
noise. The present result favors testing a second, prospectively frozen
placement rule with edit locations across the full clip and new
speakers, capture groups and voice generator families. This panel did
not fit a new operating point or alter serving behavior.

## Verification

The run completed 620 classifier forwards offline. A separate native
check reread all 124 frozen waveform commitments. It verified the exact
outside-interval samples and shared integer maskers on 96 paired edits.
The anonymous scalar receipt retains the five scheduled logits, exact
sample clocks, source commitments and all derived outcomes. The local
Seagate evaluation archive retains the concrete frozen waveforms and
replay scripts. The receipt contains no audio or transcripts.
[Scalar receipt](evaluation-panels/audio-stock-voice-transfer-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_audio_stock_voice_transfer.py \
  docs/evaluation-panels/audio-stock-voice-transfer-2026-11-13.json
```
