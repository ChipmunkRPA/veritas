# Whole original and local-delay speech states for the fixed pilot

This new context requires the actual completed v143 plan and its unchanged
metadata/code. Only its sixteen ordered pilot clips are admitted. Retain all
unavailable specifications and reserve membership; do not replace a source from
quality, geometry or model responses. Group the clips into eight fixed batches
of two. The current plan contains an original whole waveform and one fixed
middle-interval audio control per clip: 32 distinct planned encoding inputs.

Before freezing, bind only metadata/code/package and actual qualification
receipts. Recheck the original v110 authored/owned-descendant qualification,
the v114 source contract and all eight successful retained source executions.
Require exact installed package versions and the three recorded installed
encoder/feature-source file hashes, plus the three publisher-pinned Apache-2.0
speech asset manifest entries. Do not read pretrained binary weights or source
media at admission. The unchanged `av_speech_native_v110.NativeEncoder` verifies
actual asset bytes and its original native/library contract inside each later
guarded worker; no model, preprocessing, graph or precision modification is made.

Inside a worker, independently verify each admitted original WAV/NPY/file hash,
exact mono16k signed16 conversion, complete finite float32 PCM, original source
clock mapping and origins through the v142 validator. Keep canonical signed16
float32 bit patterns, including correct range without clipping or gain changes.
For the control, copy only `pcm[a:b] = original[a−5120:b−5120]` using the actual
v143 integer ranges. Preserve waveform length, observed origins and every sample
outside that interval. No padding, wraparound, replacement or hidden-state shift
is permitted. A zero-effect copy remains a recorded control with zero changed
samples; it is not replaced or claimed to be a nonzero perturbation.

Save both complete original and delayed derived WAVs/NPYs privately on Seagate.
Independently round-trip their headers, signed16 samples and float32 bits. Save
the complete original mapping, the unchanged **output clock mapping**, and
explicit output-to-original sample/value provenance. Modified values come from
earlier input samples; unchanged output timestamps must not be presented as
proof that those values were the original soundtrack at those positions.

Encode each whole waveform independently with the exact qualified CPUfloat32
native encoder, eager deterministic single-thread execution and original fixed
log-mel padding. Never shift, slice or reuse an earlier whole-input state array
as the changed waveform's encoder response. The fixed first original waveform
of each batch is encoded again immediately, requiring byte-identical full states
and native proof. If that original is predeclared unavailable, retain an
unavailable repeat; do not repeat another source. A fully available batch has
four whole inputs plus one repeat: exactly five forwards, forty across eight
batches. Cryptographic/round-trip/model failures abort and remain failed; they
are not relabeled as missing input.

Flush a private journal before/after loading and every encode. Observational
hooks record entry/completion of the actual encoder root. Require the exact
protocol original0 → repeat0 → delay0 → original1 → delay1, skipping only planned
unavailable inputs. Bind each begin's source, waveform, repeat flag, full PCM hash
and sample count; bind each end's full native proof and monotonically increasing
forward count. No extra encoder root call or unmatched begin/end may pass.

Validate all full states as finite float32 `(1500,5,384)`, with native layer-input/
final-readout parity proof. Save every unique response and the fixed repeat as
complete private NPYs, reloading without pickle and requiring exact readback.
For each unique waveform, audit all forty v109 source-time/earlier contexts
(twenty output pictures × two views) against a separate explicit ten-tick ×
five-layer copy in tick-major order. Keep unavailable contexts explicit without
invented zeros or an original-file verdict. The original states supply both
MuseTalk driver choices. Delayed whole-input states remain distinct records for
the control/head evaluation. These routes and nominal source intervals do not
establish finite encoder dependency: padding, global spectral normalization and
unmasked attention remain. No local AI evidence, achieved synchronization,
calibration, accuracy or customer-confidence result follows from these states.

Freeze and manually publish the tested context before any forward. Run one
exclusive `batch-XX/authored` stage at a time through the unchanged v108 owned
monitor: 900 seconds, 6 GiB sampled descendant RSS and initial available RAM,
3 GiB initial project/archive disk, 1.5 GiB running reserves. Require actual
successful preceding batch completion; stop on failure or admission denial.
Preserve failures and partial artifacts without retry, overwrite or relaxed
limits. Sampled memory is a lower bound rather than a guaranteed peak.

Publish code, aggregate observed availability/routing/forward counts and whole
artifact commitments only. Waveforms, learned speech states, per-source paths,
hashes, clocks, pairings and vectors remain private; derived speech features may
retain voice information. No new dependency, outside outreach, Actions, payment,
deployment, serving/default or confidence change is included.

## Consumer API

Import `scripts.grid_source_speech_v147`. Trial is
`backend/data/independent-evaluation/grid-source-speech-v147`; private cache is
the same context name on the authorized Seagate archive. Public freeze is
`docs/evaluation-panels/grid-source-speech-freeze-v147.json`.

`load_completed(batch)` requires the actual successful
`batch-XX/authored/{profile.json,output/result.json}` and matching whole-journal
hashes. It validates the complete protocol, fixed repeat, source/role/waveform,
owned output paths, original/output mappings, all forty routing contexts and
observed aggregate counts. It returns four ordered input records: the two
clips' `original_pcm`, then `delayed_audio_control`, retaining unavailable records.
This API reads metadata only; the consuming guarded worker must verify actual
NPY/WAV bytes before loading. `source_record(source_ordinal,
waveform_id="original_pcm")` resolves the fixed batch and rejects reserve or
unknown sources.

An encoded record has `source_ordinal` (the original v143 clip ordinal),
`waveform_id`, `role`, `batch`, `status="encoded"`, `whole_input_encoded`,
`hidden_file`, `hidden_file_sha256` (whole NPY), `hidden_shape=[1500,5,384]`,
`hidden_dtype="float32"`, and `proof` with `mel_sha256`, `full_hidden_sha256`
(bare contiguous float32 bytes), `layer_readout_matches`. `waveform` contains
private PCM/WAV paths and file hashes, sample count, bare float32 PCM hash,
changed-sample count, original mapping, output clock and explicit value mapping.
`routing` contains forty ordered entries with `frame_index`, `view`
(`source_time_conditioning` / `earlier_conditioning`), availability, exact v143
conditioning dictionary and bare `(50,384)` float32 chunk hash when verified.

MuseTalk must select **`original_pcm` only**, verify the whole hidden NPY hash,
shape/dtype/finiteness and bare-state proof within its own guarded worker, then
derive its exact v109 chunk and compare that context's chunk commitment. It
executes zero new speech-encoder forwards. A changed audio control requires its
separate `delayed_audio_control` record; original hidden states cannot substitute.
