# Locating substantial rewrites against a verified earlier version

The source-aligned check located **14 of 14** controlled AI rewrites on a new publisher panel. The existing passage check located **2 of 14**. Adding the source-aligned check raised the number of flagged genuine work groups from **1 of 10** to **3 of 10**, exceeding the frozen allowance of one additional group. The study supports locating substantial changes against an earlier version. The matched controls leave AI authorship unqualified, so the candidate remains in research. [Public receipt and source positions](evaluation-panels/writing-source-aligned-transfer-2026-11-12.json)

## Source and comparison

The retained F1000Research catalog supplied 98 pairs of versioned publisher XML files. Nine already scored works supplied the separate selection set. From the remainder, ten works qualified with the same recorded authors in both versions, distinct dated publication versions from 2016–2017, explicit article-level CC BY 4.0 terms, and one bounded revised sentence between two uniquely matching neighbors. The intake screened exact normalized 12-word overlap against retained source packets and across admitted groups. The combined fingerprint inventory contained 21,565 entries. Publisher records establish the versions and attribution. Historical editing tools and checkpoint training exposure remain unknown. [Earlier catalog and source procedure](writing-reference-revision-transfer-2026-11-11.md) · [F1000Research](https://f1000research.com/about) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Each held work supplied the earlier text compared with itself and the genuine revised text compared with the earlier version. Two pinned Apache 2.0 generators rephrased the genuine later sentence. Both unchanged surrounding sentences stayed in place. The first valid completion per generator and work had to be a single complete sentence of 12–40 words, between 0.75 and 1.30 times the later sentence's length, with normalized word edit distance of at least 0.25 and the same multiset of visible numeric strings. All 60 attempts completed. Eight Granite and six OLMo variants qualified. The numeric check preserves visible numbers and leaves full semantic fidelity unverified. [Granite model and terms](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [OLMo model and terms](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)

The magnitude arm uses normalized word Levenshtein distance between the two focal sentences. The distributed arm multiplies it by the fraction of destination thirds containing edits, one minus the longest exactly retained run as a share of destination words, and the smaller of one and the retained content-word share divided by 0.45. The earlier version supplies the lexical alignment and exact bounds of the submitted edit. Both arms require strict excess over their own second-highest score on the nine genuine selection works. Recent work likewise separates how much language changed from where it changed. This comparison uses its own rules and licensed publisher sources. [EditLens](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c7d936e696f57468b6b80a7b1463178c-Abstract-Conference.html) · [MixDetect](https://arxiv.org/abs/2609.32625)

The incumbent uses its first six source-ordered context responses at the fixed experimental point. A useful mark covers at least half the known edited non-whitespace characters and has intersection over union of at least 0.30. The additive gate requires at least six admitted cases per generator, gains on two or more works across both generators, preservation of the incumbent locations, and at most one additional genuine work flagged.

## Held results

| Outcome | Current passage check | Edit magnitude | Distributed edits | Current plus distributed |
| --- | ---: | ---: | ---: | ---: |
| Generated rewrites usefully located | 2/14 | 14/14 | 14/14 | 14/14 |
| Genuine earlier texts flagged | 1/10 | 0/10 | 0/10 | 1/10 |
| Genuine later revisions flagged | 1/10 | 2/10 | 2/10 | 3/10 |
| Genuine work groups flagged in either version | 1/10 | 2/10 | 2/10 | 3/10 |

The additive arm gained 12 locations across seven works and both generator families. The structural arms made identical decisions. In the only pair within 0.15 of the same reference-relative edit magnitude, both rules flagged the genuine revision at 0.515 and the generated rewrite at 0.516. A later diagnostic scan of 60 rights-qualified unused groups found two additional genuine revisions with a magnitude of at least 0.45 and exact source anchors. Both rules flagged both revisions. This diagnostic was deliberately enriched for substantial changes.

## Replay

The [portable reviewer](../backend/scripts/review_writing_source_aligned_transfer.py) recalculates the selection points, overlap rules, six arm decisions for each of 34 cases, matched-magnitude count and advancement gate from the public scalars. Exact source prose, generated sentences and full native response records remain local.

~~~sh
python3 -I -S backend/scripts/review_writing_source_aligned_transfer.py \
  docs/evaluation-panels/writing-source-aligned-transfer-2026-11-12.json
~~~

To recreate the source inputs, use the version-specific XML URLs and hashes in the receipt. The [original source reader](../backend/scripts/writing_reference_revision_sources.py) verifies the publisher metadata and removes only JATS bibliographic references before assembling prose blocks. The receipt records the block hashes, excerpt ranges and focus ranges in Unicode code points. Apply the reader's author-year pattern before feature tokenization. Use the [earlier rewrite study's lexical pattern and one-sentence validator](../backend/scripts/writing_blind_rewrite_probe.py), word-level SequenceMatcher with autojunk disabled, and divide Levenshtein edits by the larger word count. The conserved content share is the multiset overlap divided by the smaller content-word count. Content words have at least four characters and exclude these common terms.

~~~text
about after again also among and any are because been before between both but could
each else even from have however into just like may most much must need once only
over same shall should since some such than that them then there these they this
those thus upon very want were what when where which while will with would your
~~~

For model recreation, the receipt includes the exact prompt, quantized revisions, model hashes, runtime versions and sampling settings. Sort the held DOIs by SHA-256 of the literal prefix `chipmunk-source-aligned-writing-v1:` followed by the DOI. For each family and zero-based work index, the three seeds start at the family base plus three times that index. Use the model's chat template. The initial launch detected a local runtime version mismatch before inference. The completed run recorded the installed versions in a fresh lock and used the same weight and generation settings. To recreate the incumbent inputs, use the [native context and CPU scoring helpers](../backend/scripts/writing_paired_email_revision.py) with the pinned classifier revision.

The incumbent ran 102 complete inputs in 26 CPU forwards. The separate model-free native replay checked 178 publisher XML byte commitments, 20 selected publisher versions, 34 case intervals, 102 native context positions, 60 generation attempt bindings and 204 arm decisions. It rejected a moved destination span and a changed false-flag count. A separate diagnostic replay verified all four source positions of the two extra genuine revisions.
