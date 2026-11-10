# Recent detection research: methods and next tests

Reviewed 9 November 2026. This targeted refresh includes October papers,
3–4 November preprints/revisions, a 6–10 November conference paper, and two
relevant June papers. It is not an exhaustive bibliography. Relevant method
and evaluation sections were read; the BabelFake entry is based on its primary
abstract. Repository inspection is identified separately from paper review.
Published benchmark scores are not Chipmunk Verify accuracy or customer confidence.

## Images: learn editing evidence separately from image content

[PAL, 6 October, preprint](https://arxiv.org/html/2610.07916v1)
uses source-related image pairs to learn artifact representations before
localization. Its shared encoder compares features through absolute differences
and pooling; a relation classifier uses paired supervision. Matched negative
distributions are intended to discourage content shortcuts. The learned encoder
then initializes a localization model.

**Proposed local test:** independently implement an edit-relation variant using
permitted components and our source-linked edits. Compare it with the native
regional candidate on new generators, ordinary edits and compression. Keep all
related sources together across roles. Measure actual retained pixels and false
highlighted area. A reconstruction difference is not automatically an AI mask.
No PAL code, checkpoint or dataset has been adopted or cleared by this review.

[Inpainting Exchange, revision 4 November](https://arxiv.org/html/2602.00192v2)
separates synthesized content from an autoencoder's image-wide processing
artifacts. It restores source pixels outside known edit masks and uses controls
that vary generated content, background processing and compositing seams.
**Proposed local test:** source-linked controls that distinguish local editing
from global processing, with matched compression. Evaluate overlap with the
actual edit and false highlighted area, not just image-level detection. This
paper's controlled findings do not establish that every detector or generator
has the same shortcut. No author dataset, code or weights were adopted.

[FOX, 3 November](https://arxiv.org/html/2611.04111v1)
trains and locks its visual detector before training a separate explanatory
pathway. A predictor maps locked features into a frozen text-embedding space;
a decoder verbalizes guidance without changing the detector.
**Design lesson:** report generation should not alter measured detection results.
Explanations must distinguish measurements from interpretation; fluent language
does not establish pixel-level truth or independently validate a verdict. This
is a potential report-design comparison, not a qualified replacement for our
detectors. The published implementation's code and checkpoint permissions were
not established here, and no assets were adopted.

## Video: articulation consistency and temporal localization

[C-MIND, 22 October, preprint](https://arxiv.org/html/2610.30872v1)
aligns audio and mouth representations before temporal aggregation. Its local
interaction includes concatenation, products and absolute differences. The
paper controls speaker replacement and distinguishes audio-only replacement
from regenerated lips; well-synchronized retalking can still be difficult.

**Proposed local test:** compare an independently trained small interaction head
over permitted frame/audio features with the existing RMS/aperture method.
Use real timestamps, aligned neural retalking, soundtrack offsets and ordinary
controls on the same panel. This targets articulation consistency rather than
assuming that coarse synchronization proves originality. Code/weight permission
and local performance remain unestablished for the published implementation.

[LoCC, 22 June, preprint](https://arxiv.org/html/2606.22772v1)
uses neighboring mouth frames to reconstruct a counterfactual middle frame,
then learns segment and frame decisions with teacher/student supervision.
**Proposed local test:** a separate neighborhood-consistency cue, with genuine
head turns, cuts and compression controls. Any independent lightweight variant
must be described as an adaptation, not a reproduction of the diffusion model.
No published weights or code are incorporated.

[BabelFake, 5 October, preprint](https://arxiv.org/abs/2610.06339)
separates visual-only from joint audio/visual generation and examines language
and generator pairings with consented participants. **Evaluation lesson:** keep
component labels and genuine-audio/fake-video conditions distinct. The prepared
144-group local panel is not equivalent to that multilingual benchmark. No
BabelFake media was downloaded or its data terms cleared here.

## Audio: separate speech, background and watermark shortcuts

[CLRM, 3 November](https://arxiv.org/html/2611.05678v1)
keeps an existing speech encoder frozen and trains compact heads over selected
layers' standardized mean/standard-deviation summaries, differences, products
and cosine similarity. Source-fitted logistic fusion combines its score with
the base detector. The paper reports both a primary protocol with shared genuine
utterances and additional utterance/speaker-disjoint protocols.
**Proposed local test:** independently compare explicit layer relations against
parameter-matched concatenation using permitted encoders. Fit normalization and
layer selection only on development sources, then freeze before fresh grouped
testing. This requires a new study; our completed audio confirmation cannot
become a fitting set and still serve as independent confirmation. No CLRM
implementation or weights were inspected or adopted.

[What Do Deepfake Audio Detectors Hear?, 13 October](https://arxiv.org/html/2610.17617v1)
tests independently real/fake speech and environmental events. It uses all-real
mixtures as negatives and tests fake speech, fake events and both separately.
Inference-time band-stop probes examine frequency dependence without refitting.

**Proposed local test:** add a two-by-two speech/event truth matrix and balanced
mixing controls, then probe the fixed speech models by frequency band. Current
sound-category tags are context, not a trained synthetic-background detector.
Our present genuine-background speech study does not establish that capability.
These are new tests, not additions to the already frozen audio confirmation.

The authors' [protocol generator](https://github.com/nii-yamagishilab/AntiDeepfake-AudioLLM/blob/main/protocols/generate_compspoofv2_protocol.py)
and encoder configuration were inspected. They support different label targets;
an independent local harness must choose its target explicitly and retain
unknown cases rather than silently dropping them. The code advertises BSD-3;
the actual [checkpoint license](https://github.com/nii-yamagishilab/AntiDeepfake-AudioLLM/blob/main/LICENSE-CHECKPOINT)
is **CC-BY-NC-4.0**. Those released checkpoints are excluded from commercial use.
No author code was executed or copied into the application.

[The Watermark Shortcut, 22 June](https://arxiv.org/html/2606.23335v1)
finds that detectors can learn watermark presence instead of synthesis evidence.
Its paired clean/marked examples and augmentation of both truth classes test
and reduce that shortcut. **Proposed local test:** matched marked/unmarked
genuine and generated speech, with separately recorded provenance observations.
Keep trusted exact-file-bound explicit-AI credentials distinct from editable or
transferable public marks. A detectable watermark alone is not a reason to
promote a statistical classifier's score to proof. No watermark-removal method,
dataset or published detector weights were adopted.

## Text: audit formatting and learn complementary frozen-model evidence

[Linguistic features across domains and models, 6–10 November](https://aclanthology.org/2026.findings-aacl.10.pdf)
uses a balanced linear SVM over 284 document-level features. It fits preprocessing
on training data and tests unseen model families and domains, with feature-group
ablations. Lexical richness transfers more consistently in this study than
several other feature groups, but results remain context-dependent.
**Proposed local test:** a small interpretable feature comparator, length-matched
and tested separately on articles, emails and extracted documents. Compare its
added value over the passage and interval checks. Vocabulary diversity cannot
justify identifying a particular sentence or writer as AI-generated without
separate localization and error validation. No author code, data or weights
were adopted or their asset terms cleared here.

[Measuring Artifact Reliance, 24 October](https://aclanthology.org/2026.docinsights-1.13.pdf)
shows how extraction formatting can separate benchmark classes. Its format-only
interventions include normalization and profile-matched rewrapping; a placebo
helps distinguish specific artifact reliance from general instability.

**Proposed local test:** begin with a stricter whitespace-only audit preserving
all non-whitespace characters and source span correspondence. Compare the
current passage and interval checks at their fixed settings. Then test natural
articles, email and extracted documents; a formatting effect is not evidence of
AI authorship. This would be an independent audit, not reproduction of its
complete punctuation/profile protocol or a new detector accuracy result.

The [whitespace-only input implementation](text-format-audit-v93-preparation.md)
has since passed preparation on 288 consumed documents: 1,440 requested views,
800 distinct effective texts and exact original-character correspondence.
The [fixed-model response comparison](text-format-comparison-v94-result.md) has
now completed all 800 effective inputs. The paragraph intervention changed
current highlights in 91/288 documents; horizontal spacing changed none. A
separate standard-library review reproduced the source-character and aggregate
arithmetic. This measures controlled formatting sensitivity, not fresh-author
accuracy, an exact reproduction of the paper or a qualified serving replacement.

[PAWN++ in the Wild, 29 October](https://aclanthology.org/2026.wac-1.25.pdf)
learns a small head over two frozen language models' token likelihood features
and hidden representations. Its source-paired web tests expose weak transfer to
frontier rewrites despite strong curated-benchmark performance.

**Proposed local test:** retain our permitted Qwen pair as a comparator and test
an independently written learned combination against the fixed ratio and
passage/interval methods. Use source-matched rewrites and continuations with
held domains and generators. The [repository README and license](https://github.com/OlegHSE/PAWN-in-the-Wild)
were inspected: code is AGPL-3.0 and the Llama probes are gated. AGPL is not a
non-commercial license; integration obligations and all checkpoint terms still
need separate review. No source code or checkpoint was incorporated.

## What changes now

The review defines concrete follow-up comparisons. It does not retune the v84/v88
audio models, reference maps, selected arms or completed final cases. New ideas
need their own fitting/calibration and separate selection/final sources before
any confidence contribution. Keep useful candidates and report gains alongside
both wrong directions, coverage and cost; an added error is not an automatic
rejection. No new method from this refresh is yet qualified in the app.

The [known-boundary video pilot](video-native-insert-pilot-v96-result.md) has
also completed input construction and a second command-interface review:
six clips, 900 native frames and 18 timestamp-bound before/during/after PNGs.
Two swap conditions share time support with three non-AI edit controls.
This supports a controlled comparison; it does not implement C-MIND or LoCC,
establish exact AI-pixel truth, or supply a new detector accuracy result.

The [full known-boundary construction](video-native-construction-v103-progress.md)
is now running sequentially on Seagate under that unchanged plan. At its start
snapshot, three groups (18 clips, 2,880 native frames and 54 PNGs) passed both
construction and the second command-interface review. The complete target is
90 groups and 540 clips. Actual detector comparisons remain separate.

The [recorded-context display](document-context-comparison-v100.md) now makes
the lowest/highest saved responses and exact source inputs visible for a
selected passage in ordinary evidence, PDF and JSON. Its helper passed a
replay on all 800 retained inputs and 6,286 passage locations, with four
unassessed locations and no new neural calls. This is a qualified software
display improvement, not a formatting-invariant detector, independent
authorship confirmation or an adoption of a paper's weights. The key image,
lip-interaction and speech/background method sections and the audio checkpoint's
noncommercial terms were rechecked against primary sources during this update.

The [layout-consistent method comparison](text-layout-methods-v101-result.md)
has now completed on those same consumed responses. Normalization alone loses
useful localization. Keeping the original context and adding both canonical
checks increases useful mixed locations from 45/128 to 58/128, while genuine
flags rise from 5/128 to 10/128 and false highlighted human area grows. A separate
standard-library review reproduces all 5,760 observations and aggregates. This
advances a complementary candidate for separate calibration and fresh-data
testing; short-insert detection and customer-confidence qualification remain
unfinished. It is not an additional fresh accuracy experiment or serving change.

The existing commercial checkpoint restrictions, customer model-name filtering,
prepayment and preliminary-report limits continue to apply. No restricted
weights, raw customer data, outreach, workflow, real payment or GCP deployment
is part of this reading/code-inspection refresh.
