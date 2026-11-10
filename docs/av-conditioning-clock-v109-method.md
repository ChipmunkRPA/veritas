# Native audio-to-picture conditioning clock

Prepare source-time routing for the 48 qualified
[native waveforms](av-waveforms-v105-result.md). This is a data-only stage.
It supplies indices for a future generator comparison; no speech encoder,
generator or detector is loaded. Sentence-connected source roles, old model
parameters and all earlier results stay unchanged.

## Why a new clock path is needed

The retained v7 benchmark intentionally accepts normalized 25-Hz pictures and
mono-16-kHz audio at the same declared origin. Its audio chunk anchor is twice
the picture index. The new source panel has actual native picture timestamps
and a measured 25–27 ms audio lead. Applying that old index equation directly
would conflate source time, picture cadence and audio origin. Do not change
or reinterpret the old benchmark to make the new panel appear qualified.

For each common native picture, compute the exact rational audio-relative
time `source_picture_time - source_audio_origin`. Choose the 50-Hz encoder
anchor by floor, matching the retained adapter's anchor convention. Preserve
the residual time, not a rounded timestamp. Keep the possible anchor range
under the existing native-clock quantization allowance; an index crossing that
boundary is explicitly ambiguous. Verify the old v105 sample coordinate again.

Select ten ordered speech ticks, anchor minus four through anchor plus five,
then the five encoder states at each tick, yielding 50 by 384 values. Preserve
tick-major, then layer-major order. Retain the old prefix convention of
`floor(audio_samples / 320)` ticks, including its omission of a partial final
tick. Out-of-prefix slots are explicit zero slots, never clamped copies of
the first/last token. A picture outside the observed waveform is unavailable;
it cannot supply invented zero conditioning or an originality verdict.

The pure array helper is tested on authored values only. It rejects missing,
nonfinite, malformed or changed input/clock arrays. Exact normalized 25-Hz
fixtures must recover the retained index equation and ordering. Native/Fraction
fixtures test other cadences, the audio lead, boundary uncertainty and padding.
This does not qualify actual pretrained hidden states or generator output.

## Temporal indexing is not local evidence

Hash and inspect the installed speech preprocessing/encoder source without
importing the neural libraries. Check the cached publisher-bound Apache-2.0
speech configuration files against their retained asset plan. The reviewed
geometry is 16 kHz audio, 400-sample centered STFT, 160-sample hop, a kernel-3
stride-1 convolution and a kernel-3 stride-2 convolution, both with padding 1.
There are 3,000 mel steps and 1,500 encoder positions in a 30-second input;
the configured four-layer tiny encoder is expected to supply five recorded
states, which still needs native qualification in this new path.

For encoder index j, record the conservative, half-open virtual pre-attention
STFT/convolution hull `[320*j-520, 320*j+520)`. The 65-ms hull includes boundary
operations; it is not a claim that every endpoint contributes a nonzero value.
Ten neighboring ticks span a 245-ms nominal hull. Record start-boundary and
right-padding dependence separately from retained-prefix membership.

The reviewed implementation clips log-mel values against a maximum taken over
the whole input, and encoder layers are called without an attention mask.
The encoder's compatibility mask argument is unused. Therefore these indexed
states can depend on the whole padded input. Even an in-range nominal
convolution hull does not establish finite local support of the hidden states,
measured padding sensitivity or phonetic alignment. A token at a time is not
proof of synthetic speech at that time. Do not turn these metadata checks into
an AI confidence score or an automatic feature-locality claim.

## Complete source preparation and remaining neural work

Keep all 48 source records and all common picture coordinates, without
response-based selection. Rehash each cached mono WAV and verify rate,
channels, width and sample count. Bind the exact v105 journal/profile, v104
sentence/role plan, source identities and recorded native 20-frame edit
coordinates. Retain short/partial/ambiguous support and all previous roles.
Keep individual frame/token positions, source IDs and waveform commitments
private; publish aggregates and whole-artifact hashes only.

`generator_conditioning_qualified` remains false for every source. Next neural
qualification must verify actual five-state outputs, native routing/repeats,
padding effects and render response to changed conditioning. Generation must
retain source clocks and known edit boundaries. Achieved speech/mouth alignment
needs separate measurement; requested conditioning alone does not prove it.
Only then can a new articulation-consistency detector be trained and compared
on the sentence/actor-connected development and reserved sources.

## Resource and preservation rules

Freeze code, authored tests, this procedure, two cached speech config records,
installed library source hashes and source/role/waveform commitments before
preparing these records. The data-only stage uses the unchanged 900-second,
1.5-GiB observed self/active-command budget, 3-GiB initial RAM/disk and 1.5-GiB
running reserves. Resource observations occur between sources, not as a true
peak measurement. This stage spawns no decoder or neural child. Config/text/WAV
reads are not model deserialization. No weights or new packages are acquired.

Use exclusive paths and preserve failed/partial attempts without retry or
overwrite. No source substitution, old-model removal, serving/UI/API/PDF/JSON
change, real charge, external contact, workflow or GCP action is part of it.
Neural tests retain their separate 6-GiB initial-RAM admission and all existing
runtime, RSS and running-reserve limits.
