# BG-REAL JPEG localization comparison v229

This study compares the frozen v189 operating points on a different editing family. The public v225 qualification and its private commitments bind 24 BG-REAL v2 photo pairs from the pinned LaMa object-removal release. Each pair supplies the original, the edited PNG, and the publisher's semantic edit mask. Qualification verified 24 deduplicated source groups, independently attributed originals, and exact RGB edit truth inside the semantic mask. The roster includes 19 publisher test sources and five validation sources. Both JPEG qualities reuse the same groups.

Each source has five matched views at JPEG qualities 75 and 90. The AI view uses the v225 exact changed pixels in source coordinates as its positive truth. The original has zero AI truth. Conventional tone adjustment, Gaussian blur, and mirrored copy-move use the v189 transformations within the publisher semantic mask. Each conventional view has zero AI truth and separately records the pixels that the transformation changed before encoding. JPEG compression preserves coordinates and can introduce changes beyond the pre-encoding edit footprint.

The three evaluated mask policies are v189 current, sensitive, and their exact source-pixel intersection. The current map uses the existing image eligibility gate, a pixel cutoff of 0.5, and the production region selection. The sensitive map uses the fixed v42/v46 `p0.90-a0.01000` map gate and the same region selection. A useful region has pixel IoU of at least 0.30 and recall of at least 0.50. Per-quality results include useful hits, empty masks, mean edited IoU, false flags and highlighted area for each control, and overlap with genuine retouches. Against-current differences use the unchanged v189 paired group bootstrap with 10,000 draws. The intervals are pointwise and conditional on these 24 groups. They characterize cue localization on this editing family. Whole-file authorship confidence and serving policy remain outside this comparison.

## Local execution

Run from the repository root. Use an empty private map directory on the local archive and a fresh trial under the local evaluation directory. The baseline runs on Apple Metal with its frozen float32 implementation. The candidate uses CPU float32 with two threads. Use one monitored worker at a time. The inherited limits are 900 seconds and 6 GiB worker-tree RSS per phase, 6 GiB available memory and 3 GiB free on both relevant disks at phase start, and 1.5 GiB reserves while running. The first input repeats exactly once per arm and quality. The plan has 480 unique forwards and four repeats.

```sh
trial="$PWD/backend/data/independent-evaluation/image-bg-real-jpeg-v229"
raw="/absolute/private/existing-empty-v229-raw-maps"
run_v229() {
  PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 backend/.venv/bin/python \
    -m scripts.image_bg_real_jpeg_runtime_v229 "$@"
}
run_v229 prepare "$trial" "$raw"
```

Record and verify the frozen SHA-256 from the successful prepare worker log. Preparation reads the qualified source files and frozen model assets to verify identity. It creates source-coordinate masks and bindings for both JPEG encodings without performing a model forward.

```sh
freeze_sha="recorded-prepare-freeze-sha256"
for quality in 75 90; do
  for phase in 0 1 2 3 4; do
    run_v229 score "$trial" "$freeze_sha" baseline "$quality" "$phase" || exit 1
    run_v229 score "$trial" "$freeze_sha" candidate "$quality" "$phase" || exit 1
    run_v229 readout "$trial" "$freeze_sha" "$quality" "$phase" || exit 1
  done
done
run_v229 seal "$trial" "$freeze_sha"
```

Each arm-quality combination has four 25-view phases and one 20-view phase. Each readout checks the matched producer phases and their resource profiles, input bytes, raw arrays and source truth. The seal verifies all 20 producer and 10 readout receipts plus the prepare profile before writing the aggregate `summary.json`. Keep the inputs, masks, raw maps, profiles and row journals private. The aggregate contains source counts, split counts, results and hashes for the resource profiles, completions and row journals. It omits per-source identifiers and file paths.
