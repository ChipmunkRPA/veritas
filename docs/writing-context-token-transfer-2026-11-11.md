# Contextual token contributions in short work email

The prior short-email study improved recall by scoring sentences in isolation, but falsely highlighted seven of twelve human examples. This test asks whether an exact sentence can be located from the contextual representations the current classifier already reads. MixDetect separates the location of editing from the amount of editing, and OpAI-Bench measures how edit type and history change detection. Their results motivate testing localized features and known source edits. This study uses the existing Apache-2.0 classifier and its original weights. It uses none of the papers’ code, weights or datasets. [Prior short-email study](writing-short-email-contrast-transfer-2026-11-11.md) · [MixDetect](https://arxiv.org/abs/2609.32625) · [OpAI-Bench](https://arxiv.org/abs/2606.06481) · [Checkpoint and license](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

## Method and sources

The current model averages contextual token vectors before applying a linear four-class head. Applying the same head before averaging reproduces the same whole-window logits. For a sentence, the experimental reduction includes only the token positions that cover its exact non-whitespace source characters. Those token representations still encode surrounding text, so the reduced response describes how the model reacts to the measured focus within its context. The local measurements are neither separate neural decisions for individual words nor calibrated authorship probabilities.

Two small regularized heads were fitted using six previously consumed workplace email groups. Both used the current window’s combined AI margin. The token head additionally received the exact focus token margins for its combined response and the three separate model classes for generated, edited and humanized text. The heads used the same feature normalization and balanced weights across source groups and labels. The fit used 64 eligible sentences, including nine known rewrites. Two review cutoffs were fixed on a separate four previously consumed human source groups. The stricter cutoff exceeded the highest group maximum. The more permissive cutoff exceeded the second highest. These empirical cutoffs have four calibration groups and make no false-alarm guarantee. No fresh input contributed to fitting or cutoff selection.

The holdout contains four fresh 2010 work correspondence groups from one public Department of Justice release. It includes unrelated messages concerning a legislative schedule, a records response, interagency review and meeting records. Each group has a 55–88-word excerpt and a genuine human sentence splice. Three donors were by the same writer and one was by another documented federal official in the same workplace thread. Only attributable federal-duty text was used. The original and donor controls shared no normalized 12-token fragment with 45,281 previously retained project entries or other admitted groups. The test uses PDF text-layer excerpts with exact archival ranges and locally constructed multipart EML. It checks the decoded plain-text body and source positions. It leaves mail authentication, other MIME parts and attachments outside scope. [Justice Department archive](https://www.justice.gov/oip/media/1440671/dl?inline=) · [U.S. government works](https://www.copyright.gov/title17/92chap1.html)

The two pinned Apache-2.0 generator families each attempted one rewrite of the same exact sentence in each group. Five rewrites met the frozen completion, length and copy rules, with three from Qwen and two from SmolLM. Two outputs failed the single-sentence rule and one copied too much of the original. Admitted cases retained all original words outside the replacement. Useful localization required at least 0.30 non-whitespace character intersection over union with the known edit. The current selection marks its scored context. The two calibrated selections mark the exact focus as the place to inspect while retaining the surrounding input scope. [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

## Results

The three CPU passes completed 50 actual model calls, including 14 on the held-out cases. All 13 held-out EML bodies, 35 three-sentence context inputs, 26 eligible sentence ranges and five edit spans passed the source-binding checks. Across all passes, the largest absolute difference between native logits and the reconstructed token average was about 0.0000081. An independent readout reproduced every selected source range, 80 metric counts and all four focus AUC values. [Aggregate and frozen fit](evaluation-panels/writing-context-token-transfer-2026-11-11.json)

| Outcome | Current | Window, stricter | Window, more permissive | Token, stricter | Token, more permissive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Authentic originals flagged | 1/4 | 1/4 | 1/4 | 1/4 | 1/4 |
| Genuine human sentence edits flagged | 1/4 | 1/4 | 1/4 | 0/4 | 3/4 |
| All human files flagged | 2/8 | 2/8 | 2/8 | 1/8 | 4/8 |
| Human source groups with a flag | 2/4 | 2/4 | 2/4 | 1/4 | 3/4 |
| Qwen edits usefully localized | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| SmolLM edits usefully localized | 0/2 | 2/2 | 2/2 | 0/2 | 1/2 |
| All edits usefully localized | 0/5 | 2/5 | 2/5 | 0/5 | 1/5 |
| Edited file flagged only at a wrong location | 0/5 | 1/5 | 1/5 | 1/5 | 1/5 |

The simple context head retained the exact same two false-positive files as the current check and found two new useful locations, both from the same generator family and on separate source groups. Its held-out sentence AUC was 0.629. The token head’s AUC fell to 0.552. The raw exact-focus margin had an AUC of 0.648, but this study did not set or test a deployment cutoff for that standalone margin.

The token head offers no promotion case. The simpler window selection merits a fresh replication across more writers, topics and generators. The estimates are based on only four groups in one official correspondence release and the method selects a review location from surrounding context. Production passage selection and confidence are unchanged.
