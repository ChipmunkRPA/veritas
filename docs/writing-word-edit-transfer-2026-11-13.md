# Changed-word supervision for short writing edits

We tested whether training on the exact words added or replaced by an AI
editor would help locate an edited sentence from a single submission. The
existing three-sentence check found **8 of 40** fresh report edits. Adding
the word-trained check kept that at **8 of 40** and added one ordinary
source group each in the GAO, USWDS and eLife panels. The word-trained check
stays experimental. [Frozen scalar receipt](evaluation-panels/writing-word-edit-transfer-2026-11-13.json)

## Method and source split

MixDetect motivates separate supervision for the words that changed. In
this experiment, a word alignment supplies that supervision during
training. The detector reads only the submitted excerpt at inference.
This implementation uses the already pinned Apache-2.0 writing checkpoint
and its existing token representations. It uses no MixDetect code or
weights. [MixDetect](https://arxiv.org/abs/2609.32625) ·
[Pinned checkpoint and terms](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

- **Fit.** Forty-eight prior publisher polishes and their 24 unchanged
  excerpts supplied 124 changed and 10,531 ordinary destination words.
  Words unchanged within a known edited sentence received no training
  label. The fit gives each work and class equal weight. One ridge logistic
  head uses context-level features. The other adds each word's local class
  margins and their differences from the surrounding sentence medians.
- **Calibration.** A sentence takes the mean of its highest quarter of
  word responses. Each head must exceed its own second-highest maximum
  across 21 separate genuine journal works. The existing comparator keeps
  its first six qualifying three-sentence windows. All points were sealed
  before held inference. They have no public confidence interpretation.
- **Confirmation.** The deterministic source intake examined 48 of 198
  eligible GAO work IDs and admitted 20 new works. Twenty-one lacked a
  common isolated polish across both generators, six failed publisher
  geometry, and one shared a normalized 12-word source sequence with a
  previously seen or already selected work. Each admitted work supplied
  an untouched excerpt and one Gemini 2.5 Flash and GPT-5.4 Nano polish
  of the same sentence. The exact neighboring sentences are identical.
  The external held panels contain 16 genuine forum versions, 12 genuine
  USWDS versions and 36 eLife excerpts. The latter include untouched,
  constructed same-paper revisions and rewrites from Granite and OLMo.
  External texts had previous project exposure under other methods.
  Unseen report works share the fit's publisher and domain. Pretraining
  exposure remains unknown.

[Publisher data and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9) ·
[Report dataset source terms](https://gov-report-data.github.io/#license) ·
[Genuine journal controls](writing-journal-body-transfer-2026-11-11.md) ·
[Forum and USWDS controls](writing-temperature-transfer-2026-11-13.md) ·
[eLife sources and attribution](writing-elife-exact-sentence-confirmation-2026-11-12.md)

A useful location covers at least half of the known non-whitespace edit
and reaches 0.30 intersection over union. All 124 held excerpts were
assessed. The candidate would need four additional report locations from
both generators, at least two more than the context arm adds, and at most
one extra ordinary group per held cohort. The contextual token projection
also reproduced native primary scores in a direct sampled check.

| Held outcome | Existing | Context head | Word head | Existing plus context | Existing plus word |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini report locations | 4/20 | 1/20 | 1/20 | 5/20 | 4/20 |
| Nano report locations | 4/20 | 2/20 | 2/20 | 5/20 | 4/20 |
| Granite and OLMo eLife locations | 0/16 | 0/16 | 0/16 | 0/16 | 0/16 |
| Untouched GAO files flagged | 10/20 | 11/20 | 9/20 | 11/20 | 11/20 |
| Genuine forum versions flagged | 7/16 | 8/16 | 6/16 | 8/16 | 7/16 |
| Genuine USWDS versions flagged | 3/12 | 3/12 | 4/12 | 3/12 | 4/12 |
| Ordinary eLife files flagged | 0/20 | 0/20 | 1/20 | 0/20 | 1/20 |

The word head alone flagged 19 of 40 fresh rewrites, but only three
locations overlapped the known edit usefully. All three were already
covered by the existing check. More detailed supervision alone did not
make the contextual token responses more transferable. This comparison
concerns short English polishes and the existing feature extractor.
It leaves training a distinct word-localization encoder for later work.

## Replay

The public reader recounts every held score, selected sentence and
non-whitespace overlap using the standard library. It contains no prose,
model tensors or private paths.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_word_edit_transfer review \
  docs/evaluation-panels/writing-word-edit-transfer-2026-11-13.json
```

For a native run, rebuild the short-report roster described in the
[temperature test](writing-temperature-transfer-2026-11-13.md) and the eLife
short-excerpt roster described in the [typography test](writing-typography-transfer-2026-11-13.md).
Use the same locally retained publisher cache, pinned model and Python
runtime. The freeze binds the case identities, source hashes, checkpoint
and runner. The native run computes 446 unique development contexts and
447 held contexts before producing the portable receipt.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
backend/.venv/bin/python -B -m scripts.writing_word_edit_transfer freeze "$run" \
  --cache "$publisher_cache" --short-roster "$temperature_roster" \
  --elife-roster "$typography_roster" --snapshot "$snapshot"
for phase in score-development cutoff score-held; do
  backend/.venv/bin/python -B -m scripts.writing_word_edit_transfer "$phase" "$run" \
    --snapshot "$snapshot" || exit 1
done
backend/.venv/bin/python -B -m scripts.writing_word_edit_transfer publish "$run" \
  --output "$run/receipt.json"
```
