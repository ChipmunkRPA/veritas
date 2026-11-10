# Complete native mouth-clip construction and mask replay

All eight construction batches and all eight separate replay batches completed.
The bank retains **48 source groups and 288 planned conditions**. There are
**283 ready complete clips**: 94 with known neural mouth-edit provenance and 189
original/ordinary-processing clips. Five edits for the short source remain
explicitly unavailable; its original remains available. No source was replaced.

| Condition | Ready | Unavailable |
| --- | ---: | ---: |
| Original rewrite |48|0|
| Ordinary crop/resampling |47|1|
| Local blur, sigma 2 |47|1|
| Local blur, sigma 4 |47|1|
| Neural aligned conditioning |47|1|
| Neural conditioning 320 ms earlier |47|1|

The source roles remain 18 fitting and ten each confidence-calibration, selection
and confirmation. No detector scored these clips during construction/replay.
The [full construction aggregate](evaluation-panels/native-mouth-clips-complete-v120.json)
includes per-role availability and all completed-stage commitments. The
[earlier partial snapshot](native-mouth-clips-v120-progress.md) is historical,
not a claim that generation is still running.

## Verified evidence positions and pixels

The separate data-only replay checked **21,767 output pictures**, all native
PTS/time bases/dimensions, and **4,700 edited-picture masks**. Every mask exactly
equals the measured source/output RGB differences, totalling 10,797,766 changed
pixels across neural and ordinary edits. Outside each known edit, including
the retained context pictures, decoded pixels are byte-identical to source.
All mask indices/timestamps, before/during/after indices and recorded crop
pixel-edge transforms match. This confirms construction evidence, not a detector's
ability to locate an unknown edit.

All **11,808,637 decoded audio samples** reproduce the qualified original
mono16k signed-16 waveform bytes, original audio origin and continuous mapping.
Requested audio conditioning and soundtrack equality do not establish achieved
mouth/phoneme synchronization. Mouth alpha and missing-face causes were not
independently reconstructed, and neural forwards were not repeated by the audit.
See the [complete replay aggregate](evaluation-panels/native-mouth-clips-audit-complete-v121.json).

## Actual execution

Construction made 8,680 neural inputs, including eight exact seeded render repeats,
and took 1,351.64 seconds across the eight bounded stages. Replay made zero neural
calls and took 238.24 seconds. Every stage exited zero, with no guard or sampling
error. Actual sampled resource bounds and whole-result/profile hashes remain
in the aggregates; they are not true memory peaks. The ready media total
1,857,724,200 bytes and remain private on Seagate along with masks, source pairings,
face/voice states and per-source fingerprints. No raw media/features were published.

## Next detection work and limits

This bank is ready for a new source-connected comparison, not serving promotion.
The [small reference-interaction core](mouth-articulation-core-v122-design.md)
implements interpolation using actual neighbor times, ordered speech summaries and
common projections learned only from genuine fitting sources. Its 13 authored tests pass, including a
closed-form shared-latent recovery check and rejection of reserved/edited fitting
rows. No real-media head or calibration has been fitted by that implementation.

Next extraction must cover full native clips without using condition labels,
true masks or edit boundaries as features or sampling cues. Compare the current
check, visual-only, concatenation and interaction arms; retain misses, false
alarms and unassessed scope, with exact time/region evidence. All current edits
are centred 20-native-picture constructions from one retalking generator. A
different-position/generator transfer test and achieved quality/alignment review
are still needed before broad reliability claims. Publisher labels do not prove
identity, original authenticity or unknown pretraining separation. This is an
internal code replay, not outside human review.

Image, voice/background-noise and writing improvements and full customer-flow
qualification remain open in the broader program. Serving models, confidence
and rollback remain unchanged. No outreach, GitHub Actions, real charge or GCP
deployment occurred.
