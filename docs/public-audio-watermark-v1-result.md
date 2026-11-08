# Public audio watermark runs locally; real-speech qualification remains

The public AudioSeal generator and detector now run locally without a provider
account, API call or customer upload. This corrects the earlier blanket access
claim with actual execution. The implementation is still isolated research,
not an enabled app check or proof of any provider's current production mark.

The [protocol](public-audio-watermark-v1-protocol.md) and
[freeze](evaluation-panels/public-audio-watermark-v1-freeze.json) were published
at `f791a5b` before model construction/inference. All32 declared authored cases
completed:4 carriers across8 mark/noise/codec variants, with4 generator and37
detector forwards. All state tensors, four direct-versus-convenience API
readouts, exact source sample counts and the bit-exact native repeat passed.
Seven model-free interval/invalid-input tests pass.

## Actual detections, misses and false responses

The carriers are harmonic pulses, pulsed band-limited noise, chirps and silence.
**They are not human speech, real provider output or a population accuracy set.**
The mark is deliberately added with the public generator. Each partial mark
occupies the known2–6second source interval in an8second input. The noise is
white noise at10dB relative to carrier RMS, not a measured real background scene.

| Variant,4 carriers each | Cases with selected interval | Mean interval IoU for marked cases | Mean selected time outside mark |
|---|---:|---:|---:|
| Unmarked | 3/4 false responses | Not applicable | 1.750s |
| Fully marked | 4/4 | 1.0000 | 0s |
| Partially marked | 4/4 | .8814 | .625s |
| Unmarked with noise | 1/4 false response | Not applicable | 1.875s |
| Fully marked with noise | 3/4;1miss | .6250 | 0s |
| Partially marked with noise | 3/4;1miss | .5083 | .875s |
| Unmarked MP3 | 2/4 false responses | Not applicable | 2.000s |
| Partially marked MP3 | 4/4 | .8704 | .8125s |

These unexpected unmarked-carrier responses remain in the result. They cannot
be converted to real-speech false-alarm rates, suppressed from the denominator,
or rescued by choosing a new point after reading them. They also prevent an
unqualified claim that a mark check supplies definitive AI authorship.

The native convenience function's clip score is the **fraction of sample
responses above its watermark cutoff**. It is not calibrated AI-authorship
confidence. A quarter-second bin is selected when more than half its samples
exceed that same native cutoff. Native responses, unmerged bin measurements and
joined source-sample/time intervals are preserved in the private evidence.
Per-sample output does not establish single-sample boundary accuracy; the
selected interval readout has250ms bins and can include time outside the mark.

## Separate source and arithmetic replay

The [separate model-free replay](evaluation-panels/public-audio-watermark-v1-source-audit.json)
verified all32 case identities/waveforms,1,024 quarter-second bins, all joined
intervals, exact unmarked samples outside partial spans and matched-noise
transformations. Eight retained MP3s were decoded again to the identical
PCM supplied to the detector, with no trimming/padding to hide sample-count
changes. Sample count and repeat decoding do not establish immunity to codec
transients or a sample-accurate forensic edit boundary. Independent sample-mean
arithmetic agrees within the fixed1e-12 limit.

The [execution receipt](evaluation-panels/public-audio-watermark-v1-execution.json)
records native completion in30.58seconds, with1,165,361,152 bytes sampled
process-tree RSS lower bound, and separate replay in1.29seconds, with125,714,432
bytes. Both guards completed without activation or sampling errors. These are
sampled process measurements, not guaranteed peak system/GPU memory totals.

## Licensing, loading and next work

The publisher's [source](https://github.com/facebookresearch/audioseal) and
[checkpoint release](https://huggingface.co/facebook/audioseal) explicitly permit
commercial use under MIT. The reviewed source/model revisions and file SHA256
identities are frozen. An initial download check stopped because the final HTTP
ETag was incorrectly treated as a file SHA256; the retained download was then
verified against the pinned publisher's actual LFS byte identity. That is a
download-identity correction, not an inference rerun or changed outcome rule.

No upstream unrestricted pickle loader was used. Restricted Torch data loading
replaces only an unused training-metadata default factory with an inert ordinary
dictionary, preserving its entries; checkpoint training configuration is neither
resolved nor used. Every one of the40 final detector tensors and73 generator
tensors matches its source tensor or exact native weight-normalization result.
Private dependencies did not alter the serving environment.

Next: fixed, rights-cleared real-speech/background/codec tests of unmarked and
deliberately marked originals, with partial-interval truth and genuine controls,
then app/PDF/API/JSON integration of a qualified scoped check. A genuine recording
can receive this public mark; detecting it alone neither authenticates a provider
nor proves AI generation. No response enters overall confidence or the trusted
AI-declaration early stop. Real Muse/Gemini/Grok compatibility remains unverified.

All authored arrays, model checkpoints and per-case journals stay local; public
outputs are code, methods, hashes and aggregates. No customer analysis, payment,
serving model, provider upload, GCP action or outside human review occurred.
