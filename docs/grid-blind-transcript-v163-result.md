# Local blind transcript comparison

Both decoders were compared on the eight genuine pilot recordings with a
publisher word reference matched to both the utterance and the speaker. The
other eight roster entries remain unavailable for this comparison. All eight
matched inputs completed under the local CPU guard.
[Method](grid-blind-transcript-v163-method.md) · [Actual result](evaluation-panels/grid-blind-transcript-complete-v163.json)

The new decoder made 23 word edits against 48 reference words. The previous
decoder made 42 on the same recordings. Seven clips had fewer edits and one
tied. Neither recovered an exact six-word utterance. The predeclared
digit-normalized comparison produced the same new-decoder result.
[Previous result](grid-phone-alignment-v162-result.md)

The worker made eight encoder and 64 decoder forwards, took 5.54 seconds,
and reached 891,305,984 bytes of sampled descendant RSS. The improvement
supports further transcript testing. Independent phone timing and accurate
visual closures are still needed to qualify an automatic timing check.
