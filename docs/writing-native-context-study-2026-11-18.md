# Testing context and token scores on short writing edits

The previous audit found that most missed passages already fit the
sentence grid. This study tests different measurements of the same short
passages. It keeps their exact original character positions and compares
each candidate with the current pooled, within-document and report review
on identical excerpts. [Earlier localization audit](writing-interval-localization-audit-2026-11-18.md)

## What the scorer measures

- **Predicting the focus.** Compare how well the immediately preceding
  two sentences help predict the focus, against a distant two-sentence
  excerpt from the same document. Choose the distant excerpt deterministically
  by token length and lexical overlap with the original context. Both
  prefixes have matching token counts.
- **Predicting the next sentence.** Compare how the focus helps predict
  its successor with how a distant replacement helps predict it. Both
  runs retain the same preceding sentence. Match the replacement's token
  count by taking an equal-length suffix when necessary.
- **Two-sided comparison.** An unusual focus needs lower-than-usual support
  on both comparisons relative to other eligible sentences in the file.
  The single-sided results are reported separately.
- **Token losses.** The same forward pass over the native focus measures
  ordinary next-token loss and how strongly the model would repeat the
  token it has just read. Both are reduced on the same target token IDs.
  The latter is a sentence-scale test of the published repetition idea.

Adjacent here means consecutive in the submitted text. A comparison can
span a paragraph break. Prefixes are limited to 96 tokens. The minimum
twenty-token prefix on the successor side includes the shared preceding
sentence, so a substituted suffix can be shorter than twenty tokens.
The candidate sees no earlier version, edit labels, author records or
generator identifier. It records intervals on the extracted source instead
of matching a rewritten reconstruction. The local scorer uses an already
installed Apache-2.0 model, quantized offline for the Mac test. We used our
own reductions and none of the authors' code, weights, or datasets from the
papers that motivated the comparison. [Inter-sentence modeling](https://arxiv.org/html/2606.18946v1) ·
[Token repetition study](https://arxiv.org/html/2607.04061v1) ·
[Pinned local checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

## Calibration and control roles

The exact plan was locked before scoring. Twelve genuine calibration
works supplied six historical and six economic excerpts, with no shared
source group or normalized twelve-word sequence in the retrospective
cohort. Ten works met the requirement for at least five complete scored
sentences, including four historical and six economic. Each arm's operating
point is the second-highest genuine work maximum. Above that point, one
top sentence can be selected. Ordinary document-level alerts are counted
separately from useful known locations. A location must cover at least half
of the edit and meet 0.30 intersection over union on non-whitespace positions.
The operating points were bound before any assessment inference.

The retrospective cohort includes five historical handbook rewrites and
three NASA or NOAA rewrites. They preserve at least three genuine flanking
sentences on both sides. Eight corresponding untouched documents and four
ordinary edits provide controls. Two ordinary edits are archived changes
and two are same-article swaps made from genuine text. These source works
have been examined previously in this project. [Earlier historical archive work](writing-rtf-draft-2026-11-18.md) ·
[NASA and NOAA source audit](writing-alignment-preference-2026-11-18.md)

A separate source qualification checked earlier input material for
normalized twelve-word overlap against 2,381 earlier excerpts. It found
zero matches. The court cohort has three official U.S. Supreme Court
opinions, three matched genuine substitutions and one locally generated
rewrite that passed the structural checks. The GovInfo cohort has nine
official prepared speech excerpts from four administrations, nine matched
genuine substitutions and three accepted rewrites. Every substitution uses
genuine text from the same source. A separate target-only prompt for a
second local generator admitted four further rewrites. That comparison
reuses the eight ordinary controls for the corresponding four GovInfo
sources. The first two generation passes admitted four of 24 attempts.
The second generator's target-only pass admitted four of nine.
[Supreme Court opinion archive](https://www.supremecourt.gov/opinions/slipopinion/19) ·
[Official GovInfo transcripts](https://www.govinfo.gov/features/state-of-the-union)

An independent reviewer recorded four content caveats without consulting
the scorer results. The court rewrite changed a legal referent. One earlier GovInfo
rewrite copied its successor sentence, which confounds the successor test.
Two of the target-only rewrites shifted meaning. The primary tables retain
all eight structurally admitted rewrites. A separate calculation excludes
those four diagnosed cases. This leaves two prepared-speech rewrites per
generator and ordinary-only legal controls. The new source pool is too
small to establish detection rates across either register.

## Results

The current app located zero of the eight retrospective rewrites. The
two-sided comparison located the one SmolLM2 NOAA rewrite, with two
additional alerts on eight untouched excerpts and none on the four
ordinary edits. It found none of the five historical rewrites. Every
other added score found zero locations there. Seven of the eight known
edits could be expressed on the candidate's eligible sentence grid.
Among the seven misses with the two-sided check, one lacked eligible
geometry, two were ranked first at a score below the fixed point, and
four ranked a different sentence first below the point.

In the court sample, the app and all candidate scores missed the one known
rewrite. It was in scope. Each candidate ranked a different sentence first
below its point. The two-sided comparison, token repetition and ordinary
next-token score gave zero alerts on the six ordinary files. The left and
right scores added one and two ordinary alerts respectively.

| Assessment panel | Labeled edits found, app → two-sided | Ordinary files alerted by app | Left | Right | Two-sided | Token repetition | Next-token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retrospective, 12 ordinary files | 0 → 1 of 8 | 0 | 3 | 1 | 2 | 0 | 2 |
| Court, 6 ordinary files | 0 → 0 of 1 | 0 | 1 | 2 | 0 | 0 | 0 |
| Prepared speech, 18 ordinary files | 0 → 0 of 3 | 1 | 3 | 2 | 2 | 0 | 5 |
| Prepared speech, second generator, 8 shared ordinary files | 0 → 0 of 4 | 0 | 0 | 0 | 1 | 0 | 3 |

The table compares each candidate on its own at its fixed point. Every
candidate found zero new-source edits. The four semantically clean edits
gave the same result. One of those four had no eligible matching sentence,
and none of the three eligible sentences ranked first under any of the five scores.
Repeated-token loss caused zero ordinary alerts across the assessed panels
and yielded zero useful locations. The one retrospective gain from the
two-sided score came with two alerts on genuine originals from other
groups. These scores stay in evaluation.

## Independent replay

The local source audit verified 32 new-source file hashes and their
sentence-level offsets, including exact donor text in the 12 genuine
substitutions. An independent scoring comparison reproduced the five
calibration points and checked exact retrospective location decisions.
The numeric capsule binds each assessment to its recorded pre-inference
operating points and pairs it with the exact current app readings. Its
standard-library reviewer recounts all cases and the separate content
adjudication. It needs no model weights or source prose.
[Portable numerical receipt](evaluation-panels/writing-native-context-study-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_writing_native_context_study \
  docs/evaluation-panels/writing-native-context-study-2026-11-18.json.gz
```
