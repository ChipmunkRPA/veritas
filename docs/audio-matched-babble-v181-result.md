# Matched genuine and synthetic babble controls prepared

All 56 source recordings qualified through the production decoder. The
guarded data-only worker built four separate six-talker pools and all
96 matched three-second WAVs. It completed in 5.49 seconds with
168.8 MiB maximum sampled process-tree RSS and zero model calls.
[Frozen method](audio-matched-babble-v181.md) · [Verified preparation](evaluation-panels/audio-matched-babble-result-v181.json)

| Treatment | Genuine foreground | Synthetic foreground |
| --- | ---: | ---: |
| Clean | 16 | 16 |
| Six competing speakers, 0 dB | 16 | 16 |
| Six competing speakers, +5 dB | 16 | 16 |

The saved WAVs round-tripped with exact PCM hashes. A separate local
readback verified shared gain and recomputed the measured SNR for all
64 noisy variants. Within each matched pair, the two foregrounds
received exactly the same babble samples and gain.

The foregrounds come from four connected source components and share
one pool per component. All babble voices have prior project exposure.
The available declared voice and PCM exclusions matched neither the
v172 holdout nor the PACC-P observation inputs. The comparison is for
development, and preparation establishes no detector performance.
