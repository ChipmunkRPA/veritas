# Native matched-source generation: technical successor v80

The v79 phoneme preflight completed all 64 fixed sources and all 64 complete
chunks, with no generator or detector call. Its first generation batch then
stopped at `import soundfile`: the isolated generator runtime does not contain
that optional package. The import precedes model construction, output-directory
creation and every forward call. The original failure log, profile, code and
freeze remain intact. The prior preparation assumption that soundfile was
installed was incorrect.

This successor changes the native **storage container only**. It writes the
complete generated mono float32 array with installed NumPy, `allow_pickle=False`,
and exclusive file creation. It reads every array back and requires exact sample
bytes, including signed zero. Its separately recorded native clock is 24,000 Hz.
There is no clipping, quantization, resampling, padding, repeated speech, changed
prompt or regenerated phoneme plan. A later production-decoding qualification
will explicitly create a float WAV container and bind that file to these samples.

The v79 source plan, complete phoneme plan, order, seeds, 16 publisher-verified
profiles, model/config bytes, 90-package runtime and CPU float32 math remain
unchanged. The original 64-source plan still has four fixed 16-source batches.
The first generated source must repeat exactly at the same seed; that repeat
does not become an additional research case. Strict saved-state coverage,
unchanged model/profile checks and offline enforcement remain mandatory.

Freeze this successor code, tests, protocol and complete private source bindings;
publish and verify the freeze remotely **before its first generation call**.
Retain all failed or partial stages. Stop on any failed stage without automatic
retry or replacement. All original bounded resource limits remain: 900 seconds
per stage, sampled child-tree RSS at most 6 GiB, at least 6 GiB initial available
RAM and 3 GiB initial free disk, and 1.5 GiB running RAM/disk reserves. The Seagate
volume must remain mounted with the same disk reserves. Samples do not prove a
true peak-memory bound.

Source roles are the corrected v75 connected ancestry roles. Native outputs
shorter than four seconds remain recorded and ineligible for a four-second
experiment; never pad or replace them to improve eligibility. Permission and
known-source independence remain prerequisites, not claims of no pretraining
overlap. Generation does not establish intelligibility, accurate spoken words,
detector performance or calibrated confidence. No detector, VAD, confidence
fitting, customer-default change, deployment or outreach is authorized by this
source-generation stage.

Publish only code, aggregate counts and whole-artifact hashes. Source texts,
source identities/pairings, generated and original audio, per-source hashes,
voice features, profile files and learned artifacts remain private.
