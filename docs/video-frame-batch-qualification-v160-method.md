# Same-device frame check qualification

Use the first 15 of 16 sampled faces in the original clip from the first
completed app batch. The order is fixed before either adapter runs. The
reference forwards eight faces and then seven. The experimental adapter
forwards one face at a time. Both read the same verified source file, exact
frame timestamps, and stored largest-face boxes. Decode checks the original
RGB commitments before inference. [Serving receipt](grid-serving-sound-cache-v156-result.md)

Run each adapter in a fresh process with both production checkpoints on the
app batch's observed device. Verify both weight files against their pinned
hashes before loading. Record each member's actual batch sizes, float32 tensor
commitments and logits privately. Compare frame order, per-frame probability,
mean, evidence timestamps, highest frame, and the raw 0.5 decision. Accept
absolute deviations up to 0.001 for logits and 0.0001 for probabilities on
this probe. The separate process monitor records sampled RSS and elapsed time.

Use `python -m scripts.video_frame_batch_qualification_v160 freeze` before
either run, then `run reference` and `run batch-one` sequentially. Use `readout`
after both guarded runs finish. If a stage stops, retain its resource record.
Numerical parity on one clip is an adapter check. Full app memory and evidence
delivery, and accuracy on independent face swaps, need their own trials.
