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
