# Native video and speech contract results

With available RAM above the unchanged 6-GiB admission requirement, both
previously prepared authored CPU contracts executed successfully. The
[actual receipts](evaluation-panels/native-authored-contract-results-v113.json)
include result/profile hashes and resource measurements. Earlier readiness
snapshots remain historical; they are not overwritten with these outcomes.

| Contract | Actual result | Successful authored forwards |
| --- | --- | --- |
| Video alignment and repeat | Manual/native alignment agrees; 128-value float32 features repeat bitwise; blank-pattern face detection repeats | 4 |
| Speech state packing and routing | Five native states match separately captured layer/final outputs; repeated input is bitwise equal; all 50 canonical picture chunks match the retained adapter | 3 |

The speech experiment also kept its two-second tone prefix byte-identical and
added a quiet suffix starting at 2.4 seconds. Earlier convolution-state values
were unchanged, while earlier transformer-state values changed. Maximum
absolute differences for the four later states were approximately 0.0746,
0.1107, 0.4717 and 2.1181. This demonstrates broader context dependence on
this authored input. A correctly indexed speech token is therefore not proof
that synthetic speech occurred within a narrow local time interval. This is
not an AI-versus-real accuracy experiment or a phonetic alignment measurement.

Both stages exited successfully without a resource guard or sampling error.
Elapsed times were 0.78 and 4.94 seconds; sampled descendant RSS lower bounds
were 92,078,080 and 755,646,464 bytes. Sampling does not establish true peaks.
The three speech forwards and four video forwards exclude any later source
work, including the failed source extraction below.

The first source-video extraction then failed on a saved-record/scalar
comparison error after unsaved neural work. Its journal contains zero complete
case responses and its partial forward count is unknown. Its code, profile,
log and empty journal remain unchanged. The separately frozen
[v112 correction](video-appearance-corrected-v112-method.md) uses the exact
saved numeric cutoff and new exclusive output paths; it changes neither the
model nor calibration. This failure is not an inconclusive detector result.

These contracts establish execution and routing properties only. Fresh source
errors, mouth-generation response, achieved audio/mouth alignment, detector
calibration and the full customer flow need separate checks. No serving default,
customer confidence, source-authenticity claim or rollback asset changed.
