# Check printed chi-square probabilities

This local experiment recomputes a printed chi-square probability from
the stated statistic and integer degrees of freedom. It follows the
reported-value consistency approach from statcheck and recent scientific
writing verification research. A disagreement means the three printed
values do not agree under the ordinary chi-square distribution. It
does not identify which value is wrong or who wrote the text.
[Statcheck methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540394/) ·
[Scientific writing verification research](https://arxiv.org/abs/2604.08501)

The probe examines an explicit `χ²(df) = value, p … value` statement
within one native text block, with an optional sample size. It allows
one unit in the last displayed digit when interpreting a rounded or
truncated statistic or an equality. It leaves other tests, separated
table cells, OCR, and known adjusted or alternate probability methods
outside this experiment. A second version also required the words
*Pearson chi-square* or a comparable explicit test name nearby. Both
versions used the same source text and exact PDF positions. No model or
publishing service was involved. [Chi-square test and simulated probability methods](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/chisq.test.html)

## Results on published files

The first group uses a prospectively selected sample of full PLOS One
research articles published from 2018 through 2021. Six article IDs
per year were selected by hashing `chipmunk-chi-square-2026-11-18/`
followed by the DOI. The source list was the first 32 matches for
`body:"χ2 p"~6 OR body:"χ 2 p"~6`, restricted by year and journal. A
second group uses 15 distinct articles returned by queries targeting
printed statistical notation. The latter is a deliberately enriched
sample and does not estimate how often this syntax occurs in general
uploads. The three older development articles were separate. Sources
and the expected comparison were frozen before the held readings.
The publisher's article XML supplied its license and corroborated
the printed values. [Publisher search documentation](https://api.plos.org/solr/search-fields/) ·
[PLOS article reuse and attribution terms](https://journals.plos.org/plosone/s/licenses-and-copyright)

| Group | Selected articles | Readable native PDFs with XML | Matched notation | Checked statements | With an explicit nearby test name | Printed disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 3 | 1 | 0 | 0 | 0 | 0 |
| General search | 24 | 15 | 0 | 0 | 0 | 0 |
| Notation search | 15 | 10 | 25 | 18 | 0 | 2 |

Seven other files required OCR or a separate map for an image region.
Nine more could not be obtained within the 2.7 MB PDF or 0.5 MB XML download caps.
The 18 checked statements came from three articles. Seven more
statements had alternate probability language in the document or near
the result. The two observed disagreements came from one article by
Tupper and colleagues. Both positions were confirmed against the
source JATS and visually on the original PDF. Page 10 gives
`χ²(3) = 17.50, p = .002`, while the implied range is approximately
0.000555–0.000560. Page 16 gives `χ²(3) = 11.33, p = .050`, while
the implied range is approximately 0.01002–0.01012. The same printed
statistic and degrees of freedom appear with `p = .010` in another
result on that page. This does not establish the underlying test result
without the original analysis. [Tupper et al., source PDF](https://journals.plos.org/plosone/article/file?id=10.1371%2Fjournal.pone.0208403&type=printable)

The probe found all 16 staged probabilities deliberately changed to
conflict with their statistic. It flagged zero of 16 equivalent
formatting controls, zero of 16 adjusted probability controls, and
zero of 16 permutation controls. The staged set is based on admitted
source statements. It is too small to estimate false alarms for tests
with undocumented corrections. The candidate stays out of the app
because broad coverage was weak and the observed discrepancies came
from only one independent work. AI confidence and passage localization
are outside the scope of these measurements.

## Replay the measurements

The public receipt retains publisher identifiers, licenses, original
file SHA-256 values, numeric readings, and exact text ranges and PDF
boxes. It contains no full documents or source prose. This offline
command recomputes the probabilities, decisions and aggregate counts
from the receipt using only Python's standard library. The evaluation
also compared the independent log-gamma calculation with SciPy on
9,216 input combinations. [Numeric receipt](evaluation-panels/document-chi-square-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_document_chi_square \
  docs/evaluation-panels/document-chi-square-2026-11-18.json.gz
```
