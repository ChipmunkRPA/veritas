# Shared sentence ranges in adjacent writing checks

When two positive writing contexts covered the same two sentences, their shared range was usually too short to localize a whole-paragraph rewrite. On eight known paragraph edits, the current check localized five. The new check identified part of the edited text in four cases and met the full localization criterion in zero. Adding it to the current check recovered zero locations. [Public receipt](evaluation-panels/writing-shared-sentence-transfer-2026-11-12.json)

## Frozen comparison

Use two adjacent three-sentence inputs from one interior paragraph. They must share two sentences and have distinct outside sentences. Rank each pair by the lower of its two raw responses and select the unique highest pair at the current raw review point of 0.5. The selected two sentences are the intersection of the measured inputs. Each score still applies to its complete three-sentence context. A separate notice arm also requires a whole-paragraph response at the same raw point. Both readings use the existing checkpoint and its saved responses. The current comparator displays the first six positive contexts in source order.

The held comparison uses two previously scored, source-separated English panels. The Federal Register panel has six agency groups, 18 ordinary files and eight admitted generated paragraph rewrites, four per generator. Its controls include original prose, one ordinary full-paragraph edit and one ordinary two-sentence edit per agency. Each original source has an official issue and exact text witness. Its generated and genuine files share a verified Word body mapping. The Kubernetes panel contains ten different bylined works, 17 ordinary files and 14 admitted one-sentence rewrites. It measures specificity and shorter-edit spill. Earlier studies scored the two panels for different rules. This replay used no new fitting, generation, inference or downloads. [Notice qualification](writing-practical-notice-transfer-2026-11-11.md) · [Article qualification](writing-multiblock-independent-2026-11-12.md)

A useful location must cover at least 50% of the edited non-whitespace characters and reach 0.30 intersection over union. The notice criterion requires useful additions in two distinct work groups across both generators, preservation of useful current locations and at most one added ordinary false-flag group.

## Results

| Outcome | Current contexts | Shared two sentences | With paragraph response | Current plus shared |
| --- | ---: | ---: | ---: | ---: |
| Federal Register paragraph rewrites usefully localized | 5/8 | 0/8 | 0/8 | 5/8 |
| Federal Register paragraph rewrites with any edit overlap | 6/8 | 4/8 | 4/8 | 6/8 |
| Federal Register ordinary files flagged | 5/18 | 1/18 | 0/18 | 5/18 |
| Federal Register ordinary groups flagged | 2/6 | 1/6 | 0/6 | 2/6 |
| Kubernetes sentence rewrites usefully localized | 0/14 | 0/14 | — | 0/14 |
| Kubernetes ordinary files flagged | 7/17 | 0/17 | — | 7/17 |
| Kubernetes ordinary groups flagged | 5/10 | 0/10 | — | 5/10 |

The narrow ranges covered 780 edited characters and zero unrelated characters across the four notice hits. Two of eight notice rewrites had an eligible pair large enough to meet both overlap requirements before applying the score rule. The candidate had at least one eligible pair in 25 of 26 notice files and 25 of 31 article files. The earlier notice study counted six useful current locations using 0.30 intersection over union alone. Applying the same minimum recall to all arms reduces that count to five. The candidate remains experimental.

## Replay

The runner verifies the committed notice source, file and response hashes. It checks the original DOCX file hash for every admitted notice and requires exactly one actual input window per stored focus. The article replay verifies the retained case and response hashes, then rebuilds the exact text and digest for each three-sentence input. It requires full native window coverage. The held readout binds 57 case outcomes and contains no source text or generated excerpts. A separate character-set calculation reproduced the candidate outcomes in all 57 cases. A changed operating point was rejected.

An initial replay stopped at an audit guard before producing results. The saved native records include one model input per focus and use additional direct-response comparison on a subset. The corrected guard checks those two quantities separately. A later terminology correction distinguished partial overlap from a flag with zero edit overlap. All 57 case outcomes were byte-identical before and after that correction.

```sh
python3 -B backend/scripts/writing_shared_sentence_transfer.py freeze "$run/freeze.json"
python3 -B backend/scripts/writing_shared_sentence_transfer.py verify "$run/freeze.json" \
  --notice "$notices" --article-parent "$long_articles" --article "$reviewed_articles"
```

Use an empty `run` directory or reuse its saved freeze. The notice directory is the retained Federal Register transfer on the development archive. The other inputs are the retained multi-paragraph Kubernetes source run and its reviewed independent-sentence run. The comparison can be replayed without model weights. [Runner](../backend/scripts/writing_shared_sentence_transfer.py)
