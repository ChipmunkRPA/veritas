# Passage evidence on separately published article rewrites

Use every planned arm from the verified v193 intake of the GoHumanize
CC BY 4.0 test split. The 54 paired excerpts come from 30 federal
article groups. Two generators supplied the published rewrites. The
unchanged text, full rewrite, constructed short and longer replacements,
and ordinary formatting controls preserve source and generator ancestry.
Unavailable constructed arms stay in the planned denominator.
[Dataset and rights audit](https://huggingface.co/datasets/gohumanize/gohumanize-open-humanizer-dataset/tree/cc066df21227f5fdb73b513c330181f946ce82ef)

Compare the four fixed v101 methods on these fresh sources. Run the
current passage check on the app-extracted layout and on the whitespace
canonical view. Apply the sealed interval head to the canonical view and
combine highlights as in v101. Reuse the separate v7 calibration maps
and internal research points. Models see text and positions only. Layout
normalization preserves every non-whitespace source character and
token boundary. If an effective text is identical across both views,
share its current-classifier response. [Earlier comparison](text-layout-methods-v101-result.md)

Project each proposal to the original Unicode character positions.
Measure useful localization, known span recovery, human characters
highlighted, genuine documents and groups flagged, and assessed and
missing coverage. Report each arm and generator separately. The two
views share the same document. Combined highlights carry no combined
probability. The ordinary formatting controls carry zero AI truth.
The generated phrases and injected boundaries reflect the construction
procedure, with no claim about naturally co-written articles or emails.

The six fixed batches contain nine source pairs each. Run each in an
offline CPU float32 process under the existing 900-second, 6 GiB
initial RAM and sampled descendant RSS guard, 3 GiB initial disk and
1.5 GiB running reserves. Record real model forwards and retained input
proofs. Keep case texts, proposal scores and individual hashes local.
Publish aggregate coverage, errors and whole-artifact commitments.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.text_gohumanize_transfer_v196 freeze
for batch in 0 1 2 3 4 5; do
  PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.text_gohumanize_transfer_v196 run "$batch"
done
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.text_gohumanize_transfer_v196 readout
```
