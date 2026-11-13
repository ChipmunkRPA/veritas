# Semantic continuity in short article edits

A sentence similarity check located **0 of 17** known AI rewrites in a fresh PLOS Biology panel. The existing passage check located **2 of 17**. A simpler comparison with only the adjacent sentences added one useful location and five ordinary work groups with flags. Both candidates stayed outside the app. [Text-free measurements and source commitments](evaluation-panels/writing-semantic-continuity-transfer-2026-11-12.json)

## Sources and controls

The frozen source intake ranked 18 PLOS Biology records by DOI hash after checking Crossref's CC BY 4.0 entries. All 18 official JATS files were retrieved. Ten research works dated May or June 2016 were admitted after article-level license, DOI, byline, geometry and overlap checks. The other works included four outside the research scope, one with unqualified byline separation, two without qualifying geometry or overlap separation, and one after the admission cap. The admitted excerpts were separated from 13 retained source packets by normalized 12-word fingerprints. The checkpoint's pretraining exposure remains unknown. The receipt identifies each work, author, license, exact publisher XML checksum and source excerpt positions. [PLOS terms](https://journals.plos.org/plosbiology/s/licenses-and-copyright) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Each work supplied one unchanged eight-sentence excerpt and one constructed ordinary replacement using a genuine sentence from elsewhere in the same article. Two installed Apache-2.0 models independently attempted three one-sentence rewrites per work. The earliest qualifying response preserved the other seven sentences, sentence count and matched source geometry and met frozen length and word-distance constraints. Of 60 attempts, 10 Granite and 7 OLMo variants qualified. Complete factual equivalence and historical authoring tools were unverified. The model received one local excerpt sentence at a time. [Granite terms](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [OLMo terms](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)

## Frozen comparison

The candidate uses an Apache-2.0 sentence encoder to measure cosine similarity in one paragraph. At each eligible interior position, it compares the central sentence with its two immediate neighbors. The main candidate subtracts that mean from the mean agreement across the two opposing pairs of neighbors. The comparison arm uses the central sentence's average dissimilarity alone. Both pick one unique strongest focus above their own second-highest genuine work maximum. Only five of 12 exposed Kubernetes fit works had an eligible five-sentence neighborhood. The PLOS originals and generated edits stayed outside point selection. The cutoff is an internal study operating point. A similarity reading describes local content continuity and adds zero authorship confidence. [Pinned encoder and license](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/ce0dbbfefc15629ec53a6bf960220076679cef83) · [Sentence embedding research](https://arxiv.org/abs/1908.10084) · [Structure-aware writing research](https://arxiv.org/abs/2510.00890)

The incumbent keeps its first six native three-sentence contexts that meet its existing raw response point. To count as useful, a displayed location had to cover at least half the known edited non-whitespace characters and reach at least 0.30 intersection over union. This separates source localization from a score that merely flags some part of the document.

## Fresh held results

| Outcome | Current passage check | Flank contrast | Adjacent-sentence comparison |
| --- | ---: | ---: | ---: |
| Known AI edits usefully located | 2/17 | 0/17 | 1/17 |
| Additional useful edits over current | — | 0 | 1 |
| AI targets with a measured eligible focus | — | 17/17 | 17/17 |
| AI edits flagged only at the wrong location | — | 2/17 | 2/17 |
| AI edits with no candidate location shown | — | 15/17 | 14/17 |
| Untouched originals flagged | 0/10 | 1/10 | 1/10 |
| Constructed ordinary replacements flagged | 0/10 | 3/10 | 5/10 |
| Ordinary work groups with either view flagged | 0/10 | 3/10 | 5/10 |

The current check's two useful locations were Granite edits. The adjacent-sentence arm located one previously missed OLMo edit. Adding it to the current results would locate 3 of 17 edits and flag 5 of 10 ordinary work groups. The main candidate missed every known target even though all 17 had exact measured sentence ranges. Neither result meets the frozen benefit and false-alarm criterion. On the two source-exposed comparison panels, the main candidate found none of 14 Kubernetes or 24 eLife edits. The adjacent-sentence arm found no Kubernetes edits and two eLife edits. Their four-neighbor eligibility excluded 12 Kubernetes and 18 eLife targets. The result covers short, controlled scientific excerpts and uses a small fitting cohort.

## Local verification and replay

The CPU run measured 896 distinct exact sentences in 56 embedding forwards. It measured 661 incumbent contexts in 166 additional forwards and reused 461 hash-identical saved incumbent readings from the exposed panels. A separate local source audit reopened the ten official JATS files, reconstructed all 37 held deliveries and verified 296 sentence positions, 296 context inputs and 17 generated edit positions.

The public reader recomputes rankings, points, location overlap and outcomes from the saved response values and source commitments. It runs without model weights, source text or application dependencies. Recreating the native experiment requires the pinned publisher XMLs, the privately retained exact generation attempts and the recorded local model snapshots and runtime. The public receipt omits excerpts, generated responses, embeddings and model inputs.

```sh
python3 -B -I -S backend/scripts/review_writing_semantic_continuity_transfer.py \
  docs/evaluation-panels/writing-semantic-continuity-transfer-2026-11-12.json
```
