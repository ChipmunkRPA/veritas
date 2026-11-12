# Exact sentence evidence in multi-paragraph writing

The new paragraph-relative check reduced genuine highlights on the fresh sources from 11 of 20 files to 2 of 20. The incumbent and the candidate both missed all 28 known AI-edited locations across the paired plain-text and email files. On previously measured sources, the incumbent localized six edits and the candidate localized zero. Adding the candidate to the incumbent recovered zero locations. The candidate remains a research check.

| Fresh outcome | Current native windows | Paragraph-relative sentence | With independent confirmation | Current plus confirmed |
|---|---:|---:|---:|---:|
| Useful location, plain-text article | 0/14 | 0/14 | 0/14 | 0/14 |
| Useful location, multipart email | 0/14 | 0/14 | 0/14 | 0/14 |
| Genuine plain-text articles flagged | 5/10 | 1/10 | 1/10 | 5/10 |
| Genuine emails flagged | 6/10 | 1/10 | 1/10 | 6/10 |
| Genuine work groups flagged in either format | 6/10 | 1/10 | 1/10 | 6/10 |
| Mixed files with highlights only at incorrect locations | 12/28 | 4/28 | 4/28 | 12/28 |
| Mixed files with a highlight touching another paragraph | 10/28 | 2/28 | 2/28 | 10/28 |
| Email files with any highlight touching the genuine quote | 3/24 | 0/24 | 0/24 | 3/24 |
| Previously exposed mixed files usefully localized | 6/38 | 0/38 | 0/38 | 6/38 |

## Sources and known edits

Ten fresh archived English Kubernetes blog works supplied six genuine body paragraphs each in source order. The work groups had distinct credited writers and were held beyond the earlier excerpt panel's cap. Twelve separate works supplied a source-exposed transfer using the earlier exact generated replacements in their longer original article context. Unsupported markup and non-prose blocks were omitted. A frozen source receipt binds each paragraph to its exact archive character range and records the historical CC BY 4.0 terms. None of the fresh exact model inputs matched the previously measured excerpt inputs. Exposure in checkpoint pretraining is unknown. [Kubernetes website license](https://raw.githubusercontent.com/kubernetes/website/main/LICENSE) · [Earlier archived excerpt study](writing-archived-passage-transfer-2026-11-12.md)

Each body was tested as a plain-text article and as a constructed multipart EML with genuine archived quoted prose, a different alternate HTML part and a fixture attachment. The extractor selected the UTF-8 plain-text body and preserved the quote role and Unicode character offsets. These email fixtures measure MIME placement on technical writing. A separate corpus of genuine mailbox language would be needed to estimate performance on workplace correspondence.

Each positive file replaced exactly one known sentence. The two local Apache 2.0 generators produced eight and six admissible fresh rewrites. Ten and nine exact rewrites transferred from the previous work panel. The two formats are paired observations of the same edit, and the genuine work-group comparison counts each underlying work once.

## Comparison and coverage

The incumbent comparator used the first six native context windows with a raw response at least 0.50. The candidate considered exact focus sentences with two neighbors in the same body paragraph. It required that the isolated focus score exceed both isolated neighbor scores, that the native contextual score clear the fixed PEP operating point, and that the native score reach its paragraph's median eligible response. The independent classifier read only the highest-ranked candidate sentence. Any evidence highlight needed at least 0.30 nonwhitespace character intersection over union and 0.50 recall with the exact edit to count as useful. The method takes the value of neighboring sentences from hybrid-text research and uses its own fixed contrast and checkpoint confirmation. [SenFlow](https://arxiv.org/abs/2606.18946) · [SenDetEX](https://aclanthology.org/2025.emnlp-main.268/) · [Pinned independent classifier](https://huggingface.co/rasbt/ai-text-detector-modernbert/tree/a6efd4bf6ecd852f332b2569ae0f98cae13b7985)

All native windows and the selected isolation reads had complete token coverage. The candidate covered an eligible focus in 154 of 288 fresh body paragraph occurrences. Six of 14 fresh edited sentences met the same-paragraph geometry in each format. The shortlist contained 16 file occurrences of four unique exact sentences across the full run. The held check cleared five genuine work groups and added zero useful locations. The preregistered rule recorded a nominal false-highlight-only pass because there were zero incumbent useful locations to preserve on the fresh set. This exposed a flaw in that criterion. The earlier-source sensitivity loss and the zero fresh hits keep the candidate in research.

## Reproducibility

The audit verified 132 archived source paragraphs, 110 fixtures and 2,255 exact native context-to-span bindings. Primary inference used 906 unique inputs in 227 CPU forwards. The independent checkpoint used four unique sentences in one CPU forward. The replay checked the per-format outcome counts and rejected a changed quote flag. An initial attempt stopped before scoring when paragraph coordinates changed from tuples to lists during JSON replay. The corrected run used the same sources, settings and rule. All 20 generated responses, 110 fixture hashes and 906 exact primary inputs reproduced.

The [public JSON receipt](evaluation-panels/writing-multiparagraph-localization-transfer-2026-11-12.json) includes the frozen comparison, per-file outcome ledger and hashes. The [runner](../backend/scripts/writing_body_localization_transfer.py) uses the previous source archive and private generator fixtures. The [reviewer](../backend/scripts/review_writing_body_localization_transfer.py) independently reconstructs text, source locations and decisions from the local fixture and response packets.
