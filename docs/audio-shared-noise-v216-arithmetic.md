# Three-arm shared background arithmetic

The constructor accepts caller-qualified mono 16 kHz signed 16-bit
little-endian PCM for any available genuine, Alania, and Drifting arm.
Use the first 48,000 samples from each original, covering the half-open
interval `[0, 48000)` on its native sample clock. Kitchen and traffic
excerpts are supplied by the caller and each already spans exactly
48,000 samples. The caller records the tape offsets, source identity,
missing arms, and applicable rights.

Measure RMS over the entire selected window. Attenuate each available
foreground to the quietest foreground's RMS and quantize to signed 16-bit
PCM. The quietest foreground's samples are unchanged at this step.
The paired mixer uses that foreground as the reference for 0 and +5 dB
noise. Compute each tape's non-clipping gain across its full set of arms
and conditions. Use the lesser gain for both tapes and the single clean
control per arm. Quantize the clean and background samples separately
before adding them. A single clean view therefore serves all four noisy
conditions in each arm.

Independently subtract the clean PCM from every resulting mixture using
wide integers. Require exact equality with that condition's shared
quantized residual. Recompute whole-window SNR from the clean PCM and
actual residual, within 0.1 dB of nominal for each available arm.
Degenerate inputs fail explicitly. This step produces no speech-gate or
authenticity decision.

`mix_shared_noise(foregrounds, tapes)` returns `sample_rate`,
`sample_range`, `reference_arm`, `common_gain`, the shared `noise_s16le`
by `(tape, dB)`, and `variants` keyed by available arm. Each arm records
its source RMS, pre-mix level-match gain, clean PCM, noisy PCM, and
measured SNR. The function performs no I/O.
