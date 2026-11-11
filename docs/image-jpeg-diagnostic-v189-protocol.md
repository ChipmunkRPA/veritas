# Reserved JPEG localization diagnostic

This prospective local diagnostic compares three fixed ways of presenting
regional editing cues on **59 photos in 48 source groups**. It uses the entire
previously unused diagnostic role of the v26 plan. Its source-paired AI edits
are all from the previously studied SD2 editing family. Known source groups
are disjoint from fitting, calibration, method selection and native
confirmation. Creator aliases, pretrained-model overlap and unknown
camera-origin information limit generalization. See the
[v26 protocol](image-regional-supervised-v26-protocol.md) and the
[paired-source review](image-positive-pairs-v21-protocol.md).

## Inputs and comparisons

Each source supplies five matched views at both JPEG quality 75 and quality
90. JPEG75 uses the exact historical v42 Pillow encoding and JPEG90 changes
only the quality argument. Each view retains its original dimensions. Both
qualities use the same sources, so they represent 48 source groups total.

| View | AI truth | Reference region |
| --- | --- | --- |
| Publisher's SD2 edit | Positive | Exact precompression changed RGB pixels already verified in v26 |
| Unmodified original | Zero | Empty |
| Local tone adjustment | Zero | Exact precompression pixels changed by the v26 tone control |
| Local Gaussian blur | Zero | Exact precompression pixels changed by the v26 blur control |
| Conventional copy-move | Zero | Exact precompression pixels changed by compositing the horizontally mirrored source into the existing author alpha |

The copy-move uses only pixels from the same photograph. The alpha defines
where it can change pixels. The footprint records where pixels actually
changed, including any partial-alpha boundaries. Retain zero-change controls
as planned observations and count their coverage. The ordinary footprints
measure whether highlighted pixels correspond to conventional edits. They
carry no positive AI truth, and any highlight on these views counts in the
non-AI highlight rates. JPEG encoding effects never enter the truth masks.

The baseline is the v42 JPEG path with its actual whole-image gate and 0.50
component threshold. The sensitive check uses the same fixed final adapted
model as v42 and v46, with the v46 readout above 0.90 and a minimum 1% native
response area. Both use the existing quantized production component selection,
JPEG quality conditions and source projection. The shared view intersects
the selected pixels of both paths without another filter. These are the only
three readouts. No fit, gate search, selection by JPEG quality or use of this
panel for a serving default follows from this diagnostic. The sensitive
check's PNG serving scope and the JPEG production fallback are separate.

Report JPEG75 and JPEG90 results individually. Include edited photographs with
any selected pixels, with overlap, with IoU at least 0.30 and recall at least
0.50, and with no region. Report mean edited IoU with misses in the denominator
and useful locations gained and lost versus the current check. For each control
report flagged photographs and groups, highlighted area, exact overlap and
IoU/recall for nonempty ordinary footprints, and the nonempty-footprint count.
Use the fixed v27 seed and 10,000 paired component draws for 95% intervals on
differences in group-balanced useful localization, IoU, per-control false
group rates and mean highlighted area. The intervals are pointwise and
conditional on these sources and the fixed models.

## Local execution

Run from `backend` with the existing Python environment. Use a fresh ignored
trial directory and an existing empty private directory for raw maps. This
uses the historical Apple Silicon runtime, the current JPEG baseline on Metal
and the candidate on CPU with two PyTorch threads, both float32. Run the
phases sequentially. A successful data-only preparation verifies model bytes,
the training-chain and source qualification receipts, source/alpha/truth
geometry, the complete JPEG input set and ordinary masks. It writes the exact
private input bindings and a freeze with hashes and aggregate counts. Review
and record that freeze before scoring. Keep its private inputs, raw arrays and
reference masks local. The expected freeze digest is the SHA-256 printed in
the preparation worker log.

Check Metal availability from the same shell that will run the workers.
The restricted tool shell can report MPS unavailable even on the qualified
Mac. The saved raw float32 maps require approximately 2.25 GiB before
journals, ordinary masks and disk reserves.

```sh
trial="$PWD/data/independent-evaluation/image-jpeg-diagnostic-v189"
raw="/absolute/private/empty/v189-raw-maps"
./.venv/bin/python -c 'import torch; assert torch.backends.mps.is_available(), "Metal is unavailable in this shell"'
./.venv/bin/python -m scripts.image_jpeg_diagnostic_runtime_v189 prepare "$trial" "$raw"
shasum -a 256 "$trial/freeze.json"
```

Set `freeze_sha` to the verified digest, then run the exact phases below in
a shell configured to stop on the first failure. Each quality has 12 batches
of 25 views except the final batch of 20. The two arms total 1,180 planned
forwards and four first-input repeats. Raw maps must read back exactly. A
bounded worker verifies every phase, keeps successful and failed logs, and
refuses existing outputs. It samples process-tree RSS and checks time, RAM
and free space on both volumes. The 900-second and 6 GiB RSS limits apply per
phase. Sampled process RSS does not measure total Metal memory.

```sh
set -e
for quality in 75 90; do
  for phase in $(seq 0 11); do
    for arm in baseline candidate; do
      ./.venv/bin/python -m scripts.image_jpeg_diagnostic_runtime_v189 score "$trial" "$freeze_sha" "$arm" "$quality" "$phase"
    done
    ./.venv/bin/python -m scripts.image_jpeg_diagnostic_runtime_v189 readout "$trial" "$freeze_sha" "$quality" "$phase"
  done
done
./.venv/bin/python -m scripts.image_jpeg_diagnostic_runtime_v189 seal "$trial" "$freeze_sha"
```

For an incomplete run, inspect the original worker log and use the model-free
coverage command. It counts the full planned denominator and separates
verified batches from unverified batches. The seal requires every planned
result and rejects partial results. It writes the aggregate to
`summary.json`. Publish only aggregate outcomes and source commitments.

```sh
./.venv/bin/python -m scripts.image_jpeg_diagnostic_v189 coverage "$trial" "$freeze_sha"
```
