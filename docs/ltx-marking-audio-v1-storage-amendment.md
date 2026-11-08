# LTX marking contract: explicit storage-only continuation

The original run at science commit17002e2 stopped after six saved native cases
with `FileNotFoundError` while saving the first PCM16 file observation. A codec
case label contained a slash, which incorrectly became a directory in its NPZ
packet path. The original log, profile, six records/arrays and absent final
result remain unchanged. It exited1 after19.172seconds with no resource guard;
sampled RSS was at least1,123,631,104bytes, sampling errors0. Final model-call
counters were not written and are not reported as an exact recorded total.

This separate continuation encodes the one slash in a codec identifier as a
double underscore before creating its packet filename. It compiles the original
inspected run definition with exactly that one AST expression wrapped. No model,
payload, waveform, rate, noise, codec, point, decision or sample count changes.
Tests cover all80 unique flat names and refusal of ambiguous/path-like names.
The original eight science files remain frozen. This amendment adds four more
code/test/protocol pins and commits the original failed-attempt bytes.

Publish and verify the new freeze before rerunning in a different empty trial.
Repeat all80 cases under the unchanged900-second/6GiB/minimum-RAM/disk guard.
After completion require bitwise equality of every saved waveform, probability,
message response and interpreted result in the original six-case prefix.
Keep any second failure; do not overwrite/restart either directory. The failure
does not qualify an external provider detector or change customer confidence.
