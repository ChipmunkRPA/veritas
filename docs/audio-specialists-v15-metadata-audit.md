# Independent local-coordinate audit of the noisy-voice plan

The separate metadata reviewer checked all 4,004 planned files and 12,012 local
windows from the frozen 143-pair, 48-background study. It called no model and did
not use the rendering/scoring helper's truth or ancestry functions. The actual
[aggregate receipt](evaluation-panels/audio-specialists-v15-independent-metadata-audit.json)
binds the frozen input plan and reviewer implementation.

It verified all four human/AI patterns under all seven background conditions,
the original 16 kHz sample clock and half-open window boundaries, explicit speech
eligibility, source labels and source-byte assignments, the same peak-safe gain
across each pair, role-disjoint known speaker/transcript/background ancestry,
and label/SNR-independent background parents and offsets. Context routing agrees
with the recorded original-window flatness and frozen 0.65 context split. This
split is a waveform routing condition, not a customer screening threshold.

The mixed-file human-only window remains genuine. A window that merely touches
the AI boundary has no AI overlap. Missing speech eligibility is retained as
unassessed evidence. Sixteen model-free tests include those boundary cases and
refuse copied whole-file truth, shifted/missing windows, altered contexts/gain,
label-dependent background offsets, missing recipes, source-reserve use and
known speaker leakage. Source rows retain their historical development roles;
the new four-role assignment belongs to the connected foreground pair.

This reviewer verifies metadata commitments and logical consistency. It does
not independently decode waveform bytes, reproduce the speech gate or establish
publisher labels. Source preparation already recorded its actual decoder probes
and byte/PCM commitments; native authenticity scoring is a separate sequential
study. No new waveform, accuracy or improvement result is claimed by this audit.

The post-inference reviewer is implemented separately. Once all four native role
receipts and artifact/point hashes are complete, it will check every local output
against its source window and report condition, mixture pattern, context, view,
generator and time-range errors. Eight authored tests reject moved or missing
evidence and preserve source-balanced error counts. Actual full-result review
has not run while native inference remains active.

The version4 scalar replay helper also handles unavailable scores explicitly.
Three authored regression tests preserve missing confirmation units, test result
withholding on a previously assessed file and refuse to claim that adverse check
passed when every file was already inconclusive. Earlier image/passage replay
code remains pinned. This replay is prepared for the audio release; no actual
audio Linux replay or outside human review is claimed yet.
