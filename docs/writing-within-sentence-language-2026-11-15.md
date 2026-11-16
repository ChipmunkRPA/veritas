# Find a language switch inside a sentence

The previous English language gate reads complete sentences or groups
of short sentences. When the full text still reads primarily as English,
the new check samples shorter parts of individual sentences. A 12-word
sample must agree with a containing 20-word sample on a language other
than English. Samples stay within one extracted sentence. The app may
use up to 96 evenly spaced short samples across a long document. It
reports the observed range that changed the coverage decision without
assigning an authorship score. Both samples reuse the bundled language
identifier.
[Earlier language-window test](writing-language-window-transfer-2026-11-15.md)
· [Independent research on identifying language switches](https://aclanthology.org/2024.eacl-long.38/)

## Test results

The earlier 112-work panel supplied development inputs. The held panel
uses 84 different English original-and-rewrite pairs and 84 native
foreign sentences from four other authors. The intake checks that the
two English panels have no work hashes or normalized 12-word sequences
in common. English versions in each pair receive the same inserted
prose. A substantial clause has 19–31 words. A short clause has 9–12.
The controls include clean English and a short foreign book title
inserted within the sentence. Each non-English passage and its original
position are fixed before either comparison runs. This test measures
language eligibility on staged edits. It covers four foreign languages
and two English writing genres.
[English source plan](writing-structured-range-transfer-2026-11-15.md#independent-replay)
· [Spanish](https://www.gutenberg.org/ebooks/64465)
· [French](https://www.gutenberg.org/ebooks/70891)
· [German](https://www.gutenberg.org/ebooks/74008)
· [Italian](https://www.gutenberg.org/ebooks/36379)
· [Archive reuse terms](https://www.gutenberg.org/policy/license.html)

| Held delivery | Files | Allowed by old gate | Added stops with one sample | Added stops with paired samples |
| --- | ---: | ---: | ---: | ---: |
| Clean English | 168 | 168 | 2 | 0 |
| Short book title | 168 | 168 | 13 | 0 |
| Substantial foreign clause | 168 | 61 | 61 | 60 |
| Short foreign clause | 168 | 133 | 127 | 99 |
| Foreign sentence between English sentences | 168 | 2 | 2 | 0 |

The paired check left one previously admitted substantial-clause
delivery and 34 previously admitted short-clause deliveries eligible.
Its marked range identified the known language, covered enough of the
insertion and kept the overlap required by the replay in 37 of 60
additional substantial stops and 74 of 99 additional short stops.
The evidence shows the samples that influenced the gate, with limited
precision for individual word boundaries. On development inputs, the
paired check added 53 stops on 53 missed clauses with zero stops across
448 controls.
[Held receipt](evaluation-panels/writing-language-clause-held-2026-11-15.json.gz)
· [Development receipt](evaluation-panels/writing-language-clause-development-2026-11-15.json.gz)

## Local app

The development setting `VERITAS_TEXT_WITHIN_SENTENCE_LANGUAGE_REVIEW`
enables the paired check. It runs when the existing language gate would
admit the extracted text as English. When the pair agrees on another
language, the writing section shows an unassessed result and identifies
the extracted sentence and both character ranges. The two source
digests are included in the signed coverage record. The original-file
verifier checks the text and sentence against the upload. A valid
trusted Content Credentials decision still takes precedence.

The local fixture ran five simulated USD 9.99 checkouts with zero jobs
before payment. TXT, Word, EML and selectable PDF inputs returned
unassessed authorship. Their API, JSON, Standard PDF and signed source
replay agreed on both ranges. One matched English control completed the
existing writing classifier.
[Local app receipt](evaluation-panels/writing-language-clause-app-2026-11-15.json)

## Replay

The portable reader recomputes the gate counts and useful locations
from committed coordinates without downloading any source. Python 3.12
and 3.14 produced identical review counts, and a second native run
reproduced the two input plans and measurement receipts byte for byte.
Native replay needs the private freeze from the linked earlier language
and English-writing studies. Save the four plain-text editions using
their book numbers in the filename. The runner verifies their digests
before building the staged deliveries.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.writing_language_clause_transfer
held=docs/evaluation-panels/writing-language-clause-held-2026-11-15.json.gz
python3 -B -S -m "$module" review "$held"
"$runner" -B -m "$module" freeze held "$earlier_freeze" "$english_plan" "$books" "$run/held-inputs.json.gz"
"$runner" -B -m "$module" capture "$run/held-inputs.json.gz" "$run/held.json.gz"
cmp "$held" "$run/held.json.gz"
"$runner" -B -m scripts.writing_language_clause_app "$run/app" "$run/held-inputs.json.gz" "$held"
```
