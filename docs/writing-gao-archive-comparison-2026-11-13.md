# Local comparison with published GAO report highlights

Chipmunk Verify can compare an authenticated user's completed TXT analysis
with a locally installed collection of 12,228 U.S. Government Accountability
Office report highlights. Choose **U.S. GAO report highlights** under
**Compare with public archive** in the analysis results. The source record names the GAO publication, date and pinned
GovReport transcription. A match shows the compared boundaries in both
versions, exact differing wording and the submitted sentences that contain
differences. PDF and JSON preserve the same coordinates. The check carries
zero AI-authorship weight. Ordinary editing and transcription can produce
the same differences. [GovReport](https://gov-report-data.github.io/) ·
[GAO attribution and use terms](https://www.gao.gov/copyright)

## Held test

The new test used 20 works excluded from both earlier report studies.
Each had an untouched excerpt and six publisher-annotated variants.
The untouched excerpts matched the separately published GovReport highlights
after case, character and whitespace normalization. The source check reads
the GovReport transcription of each GAO report. It does not compare against a
fresh download of the individual GAO page.

| Held result | Matched to correct report | Wording differences returned | Strict locations |
| --- | ---: | ---: | ---: |
| Untouched excerpts | 18/20 | 0/20 | — |
| Gemini edited versions | 35/60 | 35/60 | 12/60 |
| GPT-5.4 Nano edited versions | 38/60 | 38/60 | 11/60 |

Forty-seven edited files and two untouched excerpts received no source
attribution. No held input matched the wrong source. Strict means one
reported group of submitted sentences overlaps at least half of all
publisher-annotated non-whitespace edit characters and reaches 0.30 IoU.
These counts describe factual source comparison for indexed reports. No
reference existed for other types of writing. [Benchmark](https://arxiv.org/html/2606.06481v1) ·
[Prior statistical comparison](writing-interval-transfer-2026-11-13.md)

A model-free check requires unique matching eight-token sequences from
one source, at least 40 distinct matching tokens covering 55% of the
submitted tokens, and consistent order in the two transcriptions. It
excludes quoted blocks and declines a comparison when a second source
has any unique matching sequence. Differences must lie between verified
anchors, and no individual change can exceed 96 tokens. Only exact
wording differences within the matched passage are shown. Broader sentence
ranges identify where to review those differences. Capitalization,
typographic quote differences and Unicode compatibility variants are
normalized. This comparison may abstain on long rewrites, mixed sources,
changed paragraph edges or text outside its local collection.

The hard controls included six kinds of mixed, reversed, quoted or
unrelated inputs on 36 source rotations. None of the 216 controls was
attributed to an archive source. The same 36 works separately supplied
plain and formatting-only copies. Each set returned 33 correct matches
without a reported difference and three abstentions. These controls do
not estimate how often unmatched uploads occur in practice.

## Install and replay locally

Download the three pinned GAO JSONL files from the GovReport release into
a local folder. The builder validates all three publisher SHA-256 values
and record counts. It saves only the public highlights, publication dates,
record identifiers and hashes. The full source files include report
bodies that the app does not read. For limited disk space, the builder
can receive one source file on standard input. Keep the archive pack out
of Git. GovReport credits Huang et al. and licenses its compilation
under CC BY 4.0. The app attributes wording to GAO and links to its
publication. [Pinned GovReport files](https://huggingface.co/datasets/launch/gov_report/tree/32feeaede49fed993aef070bc4da09263fd0429a/data) ·
[GovReport citation and terms](https://gov-report-data.github.io/#license)

```sh
python3 -B backend/scripts/build_gao_archive_pack.py "$pack" \
  --train "$sources/gao_train.jsonl" \
  --valid "$sources/gao_valid.jsonl" \
  --test "$sources/gao_test.jsonl"
```

Configure the development backend with these settings and restart.
The first GAO comparison builds the local index. The existing diplomatic
collection remains the default for TXT and Word. The GAO collection is
an explicit TXT option when the pack is configured. On the development
Mac the 41 MB pack took about 11 seconds to index, and the measured
process footprint rose by about 320 MiB. The process reuses that index.

```sh
VERITAS_DOCUMENT_ARCHIVE_COMPARISON_ENABLED=true
VERITAS_DOCUMENT_GAO_ARCHIVE_PATH=/absolute/path/to/gao-pack.json
VERITAS_DOCUMENT_GAO_ARCHIVE_SHA256=671287085feaa971c501be2e2db7d50a36ca2a9adba0642c9540f4add35b341e
```

In payment modes the normal paid analysis entitlement still applies. The
comparison starts no additional analysis or checkout and sends no submitted
content to an outside service. Use the local verifier with the original
TXT and the archive JSON to reconstruct the exact comparison.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_gao_archive_export \
  "$export" "$submitted" "$pack" 671287085feaa971c501be2e2db7d50a36ca2a9adba0642c9540f4add35b341e
```

The [held ledger](evaluation-panels/writing-gao-archive-transfer-2026-11-13.json)
includes positions, overlap counts, attribution results and abstentions.
It carries no report prose. The scalar reader independently recounts
all 140 test cases. The production route also reproduced the same
measurements and returned verifiable PDF and JSON for an original and
one edited delivery from each generator.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_gao_archive_transfer \
  docs/evaluation-panels/writing-gao-archive-transfer-2026-11-13.json
```
