# Real-text context and combination calibration pilot

This is the first real-data adapter for the shared image/video/audio/passage
specialist framework. It does not complete calibration of the other three media.
The study is development only and cannot enable a serving detector automatically.

## Sources and scope

The publisher's [TXD-22 version1](https://data.mendeley.com/datasets/prcjcggtjf/1)
declares CC BY4.0. The acquired CSV is121,656,267 bytes, SHA256
`232c260f8caf7ea61a331f39fa730956862afbe0c5fc554720daa07d215a2f7d`.
Actual parsing confirms66,000 records and22 classes of3,000 each. The initial
automated page/API requests returned403 and a browser download timed out; the
exact public file link displayed on the page subsequently downloaded successfully.
No access gate was bypassed. Raw text remains private.

UTF-8 and Windows1252 decoding both failed. Latin1 is used only as a lossless byte
container for parsing CSV; the pilot selects unchanged ASCII strings, making no
encoding claim about excluded non-ASCII records. Mixed-process source classes
are excluded as local authorship truth. The human/AI labels remain publisher
claims, not independently examined provenance.

Connect every original normalized question and duplicate normalized answer across
the full corpus before selection. Select128 eligible components, each containing
one human and one AI answer of at least160 words. Prefer17 generators in a fixed
round-robin sequence, with hash ordering as the fallback, before seeing any score.
Assign32 components to each of specialist calibration, combination fitting,
combination calibration and confirmation. The dataset omits participant IDs;
these splits cannot establish independence between human writers. External
training overlap for the fixed pretrained detectors is unknown. Neither model's
declared evaluation/training corpus is treated as unseen data here.

## Local truth, models and comparisons

Copy exactly the first60 words and next100 words from each source. Construct
human/human, AI/AI, human/AI and AI/human two-passage documents. Store each original
source slice and the corresponding Unicode-codepoint interval in the constructed
document. The inserted blank line is outside scored passage scope. No sentence
rewriting, whole-file-label propagation or substring-origin inference is used.

Use the existing pinned Apache2 tropa-mini and ModernBERT weights, their unchanged
native response definitions, CPUfloat32 and complete512-token overflow coverage.
No remote Python is executed. All assets and inference code are hash-bound.
Each detector forms one dependence family; correlated model outputs are inputs
to a supervised nonnegative combiner, not independent votes.

Freeze four arms: existing detector with pooled calibration; existing detector
with separate short/long calibration; two detectors with pooled calibration;
two detectors with separate short/long calibration. Route solely by measured
passage length (<=75 words short, otherwise long). Use the shared25-group/class
support floor, L2=.001, positive-family maximum, and declared50:50 reference prior.
Fit every arm and its separate combination calibration before any confirmation
inference. Include all planned files and missing observations in denominators.

Independently replay one training input per role/context/class and every native
overflow, using direct unbatched model calls (absolute agreement<=1e-5). Repeat
the same audit on confirmation. Bound each inference run to30minutes,6GiB sampled
process-tree memory, and at least1.5GiB free disk/available memory. Retain partial
journals and guard failures; do not silently restart or drop failed inputs.

Report file and local Brier loss, false-AI calls, missed AI, ties/unassessed results
and all denominators. An internal0.5 tendency readout under the stated reference
prior is descriptive, not a public screening threshold or customer correctness
claim. Constructed document behavior is not full article/email/document validation.
No serving admission is permitted from this pilot, including if numerical results
improve. Image, video and noisy-voice adapters and independent-source confirmation
remain required before the cross-media rollout is complete.
