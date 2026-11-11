# Original and delayed pilot speech inputs complete

All eight guarded CPU batches completed: **16 original and 16 locally delayed
whole waveforms**. The delayed cases copied the fixed earlier sample interval
into the requested edit interval, preserving length, origins and the exact
samples outside it. Every changed waveform was encoded separately. None used
a shifted slice of the original hidden states as a substitute.

The workers observed **40 encoder forwards**, including eight fixed original
repeats that matched bitwise. Independent value copies reproduced all **1,280
planned speech-context routes**. Every retained waveform and state array passed
its saved-value check. No guard violation or sampling error occurred. Summed
worker-stage time was 63.85 seconds and maximum sampled descendant RSS was
0.72 GiB. See the [actual aggregate receipt](evaluation-panels/grid-source-speech-complete-v147.json)
and [frozen method](grid-source-speech-v147-method.md).

The speech-conditioned generator uses only the original waveform's states.
The later detector comparison uses each clip's actual soundtrack, including the
independently encoded changed waveform for its audio-delay control. An indexed
context is not proof that its state depends only on that interval. This stage
does not establish voice authenticity, achieved lip motion, detector accuracy or
customer confidence. Waveforms, states, individual clocks and fingerprints remain
private; no reserve clip was encoded.
