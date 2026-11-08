# LTX public audio marking: executable local contract and retained misses

The published LTX audio marking implementation **runs locally without a
provider account**, using exactly compatible cached weights. All 80 planned
native/file cases and the separate complete sample/codec audit finished. This
adds a concrete public-verification route; it does not authenticate AI origin
or enable a customer shortcut. Wrong-payload presence and noisy/partial misses
remain explicit.

The [original protocol](ltx-marking-audio-v1-protocol.md) and
[source lock](../backend/scripts/ltx_marking_sources_v1.json) were published in
[17002e2](https://github.com/ChipmunkRPA/veritas/commit/17002e2cc86d59b51aab32ac36024f6cc7db9902)
before media inference. Its first attempt failed while saving a codec packet.
The [storage-only continuation](ltx-marking-audio-v1-storage-amendment.md) and
[second freeze](evaluation-panels/ltx-marking-audio-v1-continuation-freeze.json)
were published and remotely verified as
[55786dc](https://github.com/ChipmunkRPA/veritas/commit/55786dc3ac22cf41c0af9264abd80dd8cf639e32)
before a separate complete run. All original files remain unchanged. No result
selected a model, coefficient, payload, point, case or replacement carrier.

## Complete outcomes

There are eight rate/carrier groups: four mathematical stereo carriers at each
of 24 and 48 kHz, two seconds long. These are authored waveforms, not human voices,
genuine provider exports, independent source groups or a customer accuracy set.
The same groups supply the codec and intervention cases; do not count their
derivatives as independent recordings.

| Condition | Cases | General watermark presence | Exact public payload plus presence |
|---|---|---|---|
| Native unmarked |8|0|0|
| Native expected payload |8|8|8|
| Native opposite payload |8|8|0|
| Different payloads across channels |8|8|0|
| Mark only in the middle second |8|1|1|
| Expected mark plus 5 dB noise |8|0|0|
| Unmarked PCM16 WAV |8|0|0|
| Expected mark, PCM16 WAV |8|8|8|
| Unmarked 128 kbps AAC/M4A |8|0|0|
| Expected mark, 128 kbps AAC/M4A |8|8|8|

All 24 clean expected-pattern versions recovered all 16 bits and passed presence;
all 24 unmarked versions had no presence response. These are small authored
contract results, not a population false-positive/false-negative estimate.
All eight opposite-payload cases still returned family presence despite matching
zero of 16 expected bits. All eight mixed-channel cases also returned presence,
with 7–12 matching bits. A general presence boolean therefore cannot establish
LTX attribution.

All eight 5 dB noise cases missed presence and recovered only 7–11 expected bits.
Seven of eight half-length marked cases missed the file presence rule; three
recovered all 16 bits but only one also passed presence. Preserve both results
rather than silently replacing the publisher's rule. A failure to find the mark
does not clear AI involvement, and this experiment supplies no reliable boundary
claim for a marked interval or an AI voice edit.

## Source scope and commercial component permissions

The official [marking pack](https://huggingface.co/Lightricks/LTX-2-Marking/tree/718f58d124e0acf58da85752692d93b0bd54c38a)
publishes Meta VideoSeal/PixelSeal and AudioSeal weights/configs under MIT;
the LTX loader code has separate Apache-2.0 terms. The reviewed pack is opt-in.
HDR skips video marking, source audio in audio-to-video is not marked, and
text-to-audio uses sound marking only. Do not extend this to every LTX Studio,
hosted API, version or generation. The LTX generative weights' separate license
was not needed because no generation model was acquired or run.

We executed only the inspected
[audio marker methods](https://github.com/Lightricks/LTX-2/blob/4f521619ae0029e8fbb64d555030c2b121a33b4f/packages/ltx-pipelines/src/ltx_pipelines/marking/audio_seal.py),
channel conversion, payload and
[bit-test definitions](https://github.com/Lightricks/LTX-2/blob/4f521619ae0029e8fbb64d555030c2b121a33b4f/packages/ltx-pipelines/src/ltx_pipelines/marking/score.py).
The full package, `create` loader and upstream file decoder were not executed.
The exact existing generator/detector bytes match the published pack checksums;
all loaded tensors are checked through the restricted prior loader. No model
download, unrestricted pickle, provider-verification call or customer upload
occurred. This is numerical compatibility with the released audio method,
not qualification of an original LTX export or its full production pipeline.

The code's audio `detected` field is sample-level family presence aggregated
across both channels. Its `payload` field returns the expected baked pattern,
not the decoded message. The new research adapter records actual decoded-bit
agreement separately. Even a perfect match is public post-hoc marking that can
be added to arbitrary audio. The minimum 16-bit one-tail p-value is 1/65,536;
it is not AI-authorship confidence, identity, a signature or an early-stop basis.

Video PixelSeal code and its declared 256-bit configuration were read, but no
video checkpoint was loaded or video detection executed in this experiment.
Published source availability is stronger than a generic policy, but remains
different from measured production compatibility and commercial SaaS admission.

## Complete local audit and retained failed attempt

The complete continuation made 17 generator and 81 detector forwards, including
the declared exact repeats. The generator has 14,675,873 parameters and the
detector 8,647,298. Every loaded tensor matches its source state or prescribed
weight-normalization conversion. The original six-case prefix reproduces every
waveform, sample probability, message response and interpreted result bitwise.

The separate audit reconstructs all 80 cases, 11,520,000 native channel/sample
probabilities, 80 independent bit tests, all condition counts and 32 actual file
decodes/native stream clocks. It enumerates the 16-bit null space independently
of the upstream SciPy test and primary combinatorial sum. File checks retain
stereo/native rate and the first exact two decoded seconds; WAV quantization
and AAC loss are included. No neural forward ran in this audit.

| Attempt / phase | Seconds | Sampled RSS lower bound, bytes | Sampling errors | Exit / guard |
|---|---|---|---|---|
| Original failed packet export |19.172|1,123,631,104|0|1 / none|
| Complete continuation |31.380|1,090,846,720|1|0 / none|
| Complete sample/file audit |3.155|64,618,496|0|0 / none|

Each phase used its 900-second/6 GiB/minimum-available-RAM/disk guard. Sampling
errors are retained; RSS is a lower bound. The failed run did not write final
model-call counters, so an exact recorded total is not invented. Its six saved
cases and error are included in the
[complete public aggregate](evaluation-panels/ltx-marking-audio-v1-result.json),
alongside the two prospective commits, successful continuation and full audit.
Raw audio/probability arrays remain private. The storage correction changes
only flat packet identifiers; it does not rescue or retune any watermark result.

All 20 interpretation/storage tests and six catalog tests pass locally. The
[actual Linux run](https://github.com/ChipmunkRPA/veritas/actions/runs/42484273870)
also passes all 20 interpretation/storage tests; its
[receipt](evaluation-panels/ltx-marking-audio-v1-linux-array-review.json)
records the exact source and workflow revisions, job and decoded-log hash.
It downloads no models or media and performs no neural inference. Outside
human reviewers remain zero. Software-contract checks are not independent source-truth or
provider accuracy testing.

## Admission decision

Keep this as an isolated public-pattern research adapter. Do not enable the
customer fast path, inject AI confidence or treat an absent mark as originality.
Next qualification needs authentic original LTX exports, fresh natural speech,
music/background controls, real transformations and clearly scoped file/track
evidence. Native video marking and full-app/PDF/JSON/prepaid-flow integration
remain incomplete. The separate lip-sync methodology and commercial checkpoint
review also remain open; this is not a lip-sync detector replacement. Serving
voice calibration, billing and GCP resources were unchanged.
