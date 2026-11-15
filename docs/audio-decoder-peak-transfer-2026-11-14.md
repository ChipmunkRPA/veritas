# Narrow spectral peaks on speech edits under competing voices

Some speech decoders can leave peaks at frequencies dictated by their
upsampling and waveform reconstruction. The published approach measures
these peaks on four-second recordings. This trial asks whether a narrower
measurement of the submitted recording can locate a partial edit under
competing speech. It uses independently written signal processing and
requires no new model weights or clean reference at inference.
[Decoder-artifact study](https://arxiv.org/html/2609.32050v1)
· [Partial-splice boundary and interior study](https://arxiv.org/html/2609.25822v1)

## Method

Measure 12 fixed frequencies in the 1–7.35 kHz range on the native
16 kHz source clock. In each 200 ms Hann block, subtract the median of
nearby bins from the peak at a registered frequency. Use four consecutive
blocks per range and keep the second-smallest prominence. This requires
support in at least three blocks. The local contrast variant subtracts
the stronger available flank estimate. The paired variant uses the
weaker of two peaks from the same registered frequency pattern. Apply
the recorded level floors and advance every 200 ms. The development
controls set a separate point for each method, before held measurements.
[Frozen frequencies, rule and scores](evaluation-panels/audio-decoder-peak-transfer-2026-11-14.json)

The trial uses 195 six-second files from four previously used project
panels. The first stock-voice panel is for point selection. The three
held panels include randomized stock-voice replacements, three other
generator families, and SINE publisher edits. Ordinary controls include
untouched clips, genuine splices and isolated recordings of competing
speech. Same-word resynthesis is counted separately. Parent records bind
all decoded PCM, edit clocks and existing short-voice comparison scores.
Several deliveries share a host or background recording. The held
panels test transfer from the point-selection bank, with prior project
exposure disclosed. The historical comparisons use the short-review
schedule frozen for each panel.
[Stock point-selection panel](audio-stock-voice-transfer-2026-11-13.md)
· [Randomized stock voices](audio-random-clock-stock-2026-11-13.md)
· [Three-generator panel](audio-remote-neighbor-replication-2026-11-12.md)
· [SINE source and license](audio-sine-resynthesis-transfer-2026-11-14.md)

## Results

A useful position covers at least half of the annotated edit and at
least 30% of the union of those intervals. Each held cell reports useful
locations and ordinary-file review flags with the existing comparison
retained.

| Rule | Random stock, 16 edits / 24 ordinary | Other generators, 12 / 36 | SINE, 12 / 24 |
| --- | ---: | ---: | ---: |
| Existing short review | 1 / 3 | 0 / 5 | 0 / 0 |
| Plus absolute prominence | 1 / 6 | 0 / 10 | 0 / 0 |
| Plus local contrast | 1 / 5 | 0 / 11 | 0 / 1 |
| Plus paired contrast | 1 / 3 | 0 / 6 | 0 / 2 |

Local contrast added two correct ranges on the point-selection panel,
using 1.5 kHz in the same host and same source interval. This effect did
not transfer. The local contrast added nine ordinary flags on held
deliveries and four new ordinary source groups. All 15 held isolated
backgrounds remained unflagged. Among 12 held same-word resyntheses,
local contrast added two review flags to the existing one.

Before applying any points, the strongest range met the overlap rule on
9/40 held edits for absolute prominence, 2/40 for local contrast, and
4/40 for paired contrast. Every held edit had an acoustically measurable
candidate overlapping its label. The main failure is ranking the right
interval among measurable candidates. Simply lowering these points would
recover at most those strongest ranges while adding other alerts. All
three rules remain experimental.

## Reproduce the comparison

The public ledger includes numeric readings at the 12 measured frequencies,
exact sample labels and whole-input hashes. It excludes source audio and
model weights. Python's standard library recomputes the points, coverage
and outcomes using the committed parent receipts. Native replay reads the
exact inputs from the parent procedures. The alternate native calculation
projects directly onto the measured frequencies instead of using an FFT.
Both calculations also passed the two analytic controls in the test file.
Across all 195 clips the native comparison preserved measurement admission,
decision points and outcomes. Six of 73,220 numeric readings differed by
about 0.0000001 dB after rounding. [Native comparison receipt](evaluation-panels/audio-decoder-peak-projection-2026-11-14.json)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/audio-decoder-peak-transfer-2026-11-14.json
python3 -B -S -m scripts.audio_decoder_peak_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_decoder_peak_transfer measure "$run/fft" \
  --archive "$stock_archive" --sine "$sine_panel"
backend/.venv/bin/python -B -m scripts.audio_decoder_peak_transfer project "$run/projection" \
  --archive "$stock_archive" --sine "$sine_panel"
python3 -B -S -m scripts.audio_decoder_peak_transfer compare "$run/fft/receipt.json" \
  --projection "$run/projection/receipt.json"
```

The archive arguments use the same layouts described in the
[shared Mel comparison](audio-shared-mel-context-2026-11-14.md#replay).
Independent reports can cite the panel, input SHA-256, selected sample
range, and which result or location they reproduced.
