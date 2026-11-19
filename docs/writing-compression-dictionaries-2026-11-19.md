# Can compression locate a short writing insertion?

Neither compression comparison found a known insertion. The primary masked
reading added alerts in two of nine ordinary technical work groups. The
existing passage check flagged none of those groups. All three methods missed
the sole admitted fiction insertion. The trial supports keeping the existing
passage check. [Count-only results](evaluation-panels/writing-compression-dictionaries-2026-11-19.json)

## Comparison

A window is one to three sentences of standalone English prose with at least
three surrounding sentences on either side. The candidate compresses that
window twice with standard-library raw DEFLATE. One dictionary contains
ordinary text and the other contains known generated rewrites. Both get the
same byte budget from each matched source to reduce differences in length and
topic. The primary encoding retains common grammatical words, punctuation and
limited word-shape information. A diagnostic encoding keeps lexical content.
This differs from the previous study in which an editor was asked to shorten
prose. [Earlier compression edit study](writing-compression-2026-11-18.md) ·
[Published compression-based detection](https://doi.org/10.1016/j.neucom.2026.132731) ·
[Reference implementation of seeded compression detection](https://github.com/thinkst/zippy)

Both arms use the same predeclared selection rule. Measure the ordinary minus
generated compressed byte count divided by input byte length. Rank every
eligible window, then select at most one uniquely best location above the
calibrated research point. Seven known meaning-preserving paired source works
fed the dictionaries. Eight other archived source groups fixed the operating
points before held scoring. The technical and fiction panels supplied no fit
inputs. An overlap match covers at least half of the true non-whitespace
characters and reaches 0.30 intersection over union. The experiment used our
own implementation without outside detector code, model weights or seed text.
[Prior source and exact-position study](writing-positional-attribution-2026-11-19.md)

## Source lineage

The technical panel reused nine independently fixed archive works, eight
Programming Historian lessons and one Data Carpentry course, under CC BY 4.0.
It contains seven known edits from two local generators and 18 genuinely
published earlier and later excerpts. These sources are a holdout for the
compression method and were measured previously by the current passage check.
Their editorial chain establishes real version differences. It leaves the
historical authors' editing tools unknown. [Archive scope and licensing](writing-publisher-insertions-2026-11-19.md)

Five new works by five historical authors qualified from a fixed intake of
seven U.S. public-domain works. The official Project Gutenberg catalogue
identifies their rights. Input bodies came from its official mirror, and
header and footer wrappers were removed. Each work contributed the genuine
published excerpt and two harder controls, a constructed sentence omission
and a constructed same-work sentence substitution. These last two controls
reuse the author's prose and are experimentally assembled versions.
Known sources and their text fingerprints were checked for overlap with fit
inputs and retained comparison panels. Model pretraining exposure is unknown.
[Project Gutenberg licensing](https://www.gutenberg.org/policy/license.html) ·
[Official mirroring](https://www.gutenberg.org/help/mirroring.html)

Two locally installed Apache-2.0 generator families supplied 20 fiction
rewrite attempts. Twelve passed exact text geometry. One of those 12 retained
the story facts closely enough for the held comparison. Nine changed a fact
or qualifier and two were unclear. The single admitted result comes from
SmolLM2 and makes this an exploratory fiction comparison. The study cannot
estimate literary-text sensitivity or cross-generator transfer. All 16
fiction excerpts round-tripped through 32 TXT and DOCX files with the same
extracted text and positions. [Qwen checkpoint terms](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) ·
[SmolLM2 checkpoint terms](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/blob/31b70e2e869a7173562077fd711b654946d38674/LICENSE)

| Check | Technical insertions located / 7 | Technical ordinary works flagged / 9 | Fiction insertion located / 1 | Fiction ordinary works flagged / 5 |
| --- | ---: | ---: | ---: | ---: |
| Existing passage check | 0 | 0 | 0 | 0 |
| Masked compression | 0 | 2 | 0 | 0 |
| Lexical compression | 0 | 1 | 0 | 0 |

All 41 excerpts were assessed. The masked method flagged three ordinary
technical files and the lexical method flagged two. The lexical arm ranked
only two technical targets uniquely first even before selection, and the
masked arm ranked zero. Neither method flagged a fiction control in 15
files. The two compression arms together took 0.067 seconds locally with
Python 3.11.9 and zlib 1.2.12. Their dictionaries were small, with 723 bytes
per side for the masked arm and 1,161 bytes per side for the lexical arm.
The experiment tests this frozen method and data scale.
[Incumbent checkpoint and terms](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Recount

Public materials contain only aggregate counts and commitments. The reader
checks the reported arithmetic and omits source prose, draft text and model
predictions. Running it requires the Python standard library.

```sh
python3 -B -I -S backend/scripts/review_writing_compression_dictionaries.py \
  docs/evaluation-panels/writing-compression-dictionaries-2026-11-19.json
```
