# Complete audio source qualification v81

Qualify every one of the 64 v80 matched native outputs after all four bounded
generation stages pass. Retain the original v79 dependency failure. Require all
ordered source indices, unchanged published code/source/phoneme bindings, no
network or detector calls, unchanged generator state and the exact first-source
repeat. The repeat is not an additional study case. No source substitution,
prompt editing or generation retry is allowed.

Read every native float32 array without pickle, verify recorded bytes, sample
hash and native 24 kHz clock. Wrap the complete samples in an IEEE float WAV
container with the existing backend audio library and require exact float32
readback. Do not clip, quantize, pad, change speed or resample at this wrapping
step. Decode the WAV through the existing production decoder: its explicit
16 kHz signed-16-bit conversion is part of the recorded production input and
timeline, not represented as unchanged native samples. Report native amplitudes
outside the signed-16-bit range and source durations.

Keep all generated sources in the complete manifest. Apply the existing v72
four-second, finite, nonsilent fixed-prefix and selected/history byte/PCM
eligibility checks without detector or VAD selection. An ineligible source stays
recorded with its reason; never pad or replace it. Rehash all 448 previously
qualified originals. Join the 464 foreground and 48 background nodes on actual
voice, source context, full/prefix text, raw bytes and full/prefix PCM aliases.
Require the original 157 planned connected groups to remain whole and confined
to their assigned role, including matched genuine/synthetic texts. A new alias
that merges separate declared groups or roles fails qualification.

Record all input, code, decoder, protocol, generation stage and output manifest
hashes. Run under the existing corrected child-tree observer and unchanged
900-second/6-GiB limits, initial 6-GiB RAM and 3-GiB disk headroom, and 1.5-GiB
running reserves. Retain a partial journal or failed profile without overwrite
or automatic retry. No neural models or fitting run in this stage.

This establishes input integrity and known-source role separation. It does not
verify intelligibility, exact spoken words, independence from model pretraining,
detector quality or calibrated confidence. Historical expressive stock voices
and music remain historically seen and explicitly disclosed in the source-role
documentation. Only aggregate counts and whole-artifact hashes may be published.
Audio, transcripts, record identifiers/pairings, voice features and learned
artifacts remain private. Defaults and hosting remain unchanged.
