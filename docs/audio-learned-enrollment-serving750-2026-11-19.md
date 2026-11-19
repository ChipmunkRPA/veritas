# Learned speech extraction adds zero useful detector locations

The unchanged 750 ms serving check gained **zero useful synthetic-speech
locations** from the learned extraction view. Using the processed copy alone lost
one of two useful clean held detections. Keeping original and processed results
preserved that detection, added two development genuine false-flag files across
two reader lineages, and expanded highlighted genuine time. The frozen research
gate fails. The extractor stays in research and customer assessment is unchanged.

## Acoustic success and detector question

The [genuine reconstruction preflight](audio-learned-enrollment-genuine-preflight-2026-11-19.md)
passed its small noisy acoustic gate. A separately frozen one-family follow-up
retained all four held noisy cloned intervals within the −3 dB bound. That panel
also contained one held clean clone attenuated by more than 30 dB and one
development noisy clone below −3 dB. Acoustic reconstruction therefore leaves
synthetic preservation and detector utility as separate questions.

This detector comparison uses all preceding native inputs: twelve genuine,
twelve synthetic and eight isolated-background deliveries. Four physical
Chatterbox edits repeat across clean, −5 dB and −10 dB conditions. Two reader/meeting
lineages are development and two are held. Each supplied genuine enrollment is
from a different chapter of its declared target reader. The wrong enrollment
comes from another reader in the same role.

The inputs, references, backgrounds and exact extractor are fixed before detector
responses. Native sources and raw extraction outputs produce 96 measured views,
with original, true-reference processed and wrong-reference processed copies for
each input. Processing retains all 192,000 samples on the original 16 kHz clock.
Raw output is rounded to PCM16 without peak normalization, time alignment or
stretching. No sample clipped during this conversion.

## Frozen check and complete result

Run the actual serving speech admission, preprocessing and 23-window schedule
in an isolated offline comparison without the customer request deadline.
Each window lasts 750 ms with 500 ms steps. The existing raw operating point stays
**6.642364579351768**, using strict excess and two adjacent admitted windows.
The final 250 ms remains outside this schedule. This internal evaluation point
provides no calibrated probability or customer screening promise.

The additive arm keeps the union of original and true-reference windows. It
preserves every original range, including misplaced and ordinary alerts. Wrong
reference is diagnostic only. A useful selected window covers at least half the
known edit and has temporal intersection over union of at least 0.30.

Each table cell has two synthetic and two genuine deliveries. Noisy cells also
have two isolated backgrounds. Values compare original with additive review.

| Role and condition | Useful synthetic | Missed synthetic | Genuine false-flag files | Background flags |
| --- | ---: | ---: | ---: | ---: |
| Development, clean | 0 → 0 /2 | 2 → 2 /2 | 0 → 0 /2 | — |
| Development, −5 dB | 0 → 0 /2 | 2 → 2 /2 | 0 → 1 /2 | 0 → 0 /2 |
| Development, −10 dB | 0 → 0 /2 | 2 → 2 /2 | 0 → 1 /2 | 0 → 0 /2 |
| Held, clean | 2 → 2 /2 | 0 → 0 /2 | 0 → 0 /2 | — |
| Held, −5 dB | 1 → 1 /2 | 1 → 1 /2 | 2 → 2 /2 | 0 → 0 /2 |
| Held, −10 dB | 0 → 0 /2 | 2 → 2 /2 | 1 → 1 /2 | 0 → 0 /2 |

Across all twelve synthetic deliveries, useful results stay at 3/12 and misses
at 9/12. Genuine false-flag files rise from 3/12 to 5/12. The two added files belong
to different development reader lineages, one at each noisy level. Two additional
synthetic deliveries gain only incorrectly timed alerts. They supply no useful
edit location. Background flags remain 0/8.

Held genuine file counts stay unchanged, while their highlighted union duration
expands by three seconds. Across the whole panel, highlighted genuine time rises
from 3.75 to 9.75 seconds. Counting flagged files alone would hide this cost.
Selected windows failing the localization rule include partially overlapping
context windows. The receipt separately counts those wholly outside the edit.

The processed copy alone locates only one of the two clean held edits. One
processed synthetic view loses assessed coverage at its edit. Wrong-reference
clean views are unassessed in all four synthetic and four genuine deliveries.
One true-reference background view and two wrong-reference background views are
also unassessed. Original and additive synthetic/genuine target coverage are fully
assessed, so the unchanged noisy misses are not explained by those admission gaps.
An unassessed view supports no original-file verdict.

The frozen research gate required a new held noisy physical event with a bounded
ordinary cost, clear matched controls and assessed coverage. Its cost and coverage
limits pass. The gain requirement fails because there are zero new held noisy
events. These results support no deployment or confidence change.

## Reuse, controls and verification

This is one generated family and two held lineages, with correlated condition
copies. Genuine, cloned, enrollment and background recordings have prior project
exposure. Genuine sources are from train-clean-100, also named by the extractor's
training recipe. Exact model-training and original cloning-prompt overlap remain
unresolved. Current-role separation establishes no unseen-reader validation.
Genuine controls use untouched foregrounds with matched noise. Ordinary local
splices and resampling edits remain outside this panel.

The source run completed 96 views and 1,255 classifier calls on 1,254 distinct
inputs, including one exact repeat, in about seven minutes on two CPU threads.
Independent replay reconstructed 18,432,000 view samples and checked 2,208 scheduled
windows, 1,670 admitted preprocessing bindings, original-clock intervals, matched
controls and the frozen gate. Eight authored selector controls passed.

The [anonymous receipt](evaluation-panels/audio-learned-enrollment-serving750-2026-11-19.json)
contains grouped counts and durations. It excludes source identities, individual
positions, recording hashes, private paths, per-case responses, raw features and
media. A separate private recount confirmed the two new genuine lineages directly
from the closed hexadecimal readings. The portable reviewer checks the pinned
aggregate arithmetic and rejects thirteen authored corruptions. Exact source/model
replay requires retained private custody.

```sh
python3 -I -B -S backend/scripts/review_audio_learned_enrollment_serving750.py \
  docs/evaluation-panels/audio-learned-enrollment-serving750-2026-11-19.json \
  --self-test
```

The study reuses the [existing detector](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector)
and [reviewed extraction code and checkpoint](https://github.com/REAL-TSE/wesep-real-tse/blob/d9ed77d788e9bc8ca08d15854defe404e4593773/README.md).
The [DuplexGen source card](https://huggingface.co/datasets/DuplexGen/duplexgen-spoken)
and [license table](https://github.com/duplexgen/duplexgen-code/blob/main/docs/DATA_LICENSES.md)
assign MIT terms to the selected PLN recordings. Genuine recordings and meeting
data retain [LibriSpeech CC BY 4.0](https://www.openslr.org/12/) and
[AMI CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) terms.
Generation, detector points, overall confidence and production behavior are unchanged.
