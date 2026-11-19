# Matched-genuine segment correction under strong noise

The small acoustic correction increased false alerts and added no useful
noisy locations. The original shorter-window control found more clean
and washing-machine edits with unchanged ordinary-file alerts. Both
findings come from a completed 496-delivery comparison with frozen
methods and points. Application behavior and overall confidence remain
unchanged.

## Fitting, calibration and transfer

The previous MLS panel supplies four foreground-reader groups for fitting
and four separate groups for calibration. Each includes a distinct genuine
donor reader, giving eight declared readers on each side and zero declared
reader overlap. Each split contains 184 deliveries. The fitting pool
provides 2,530 admitted windows, 47 positive windows and 216 matched
synthetic-versus-genuine comparisons.

Two experimental heads use the installed commercial-permission voice
classifier's cached 750 ms readings. The score-only head has two parameters.
The acoustic head has nineteen and adds source-derived spectral, level,
periodicity, speech-coverage and neighboring-score measurements. Both use
group-balanced logistic fitting, matched-clock ranking and fixed ridge
regularization. Their normalization uses fitting data alone. Separate
ordinary calibration groups set a strict maximum point and a wider point
allowing at most one foreground group. Both points respect the highest admitted
isolated-background response and were frozen before transfer inference.
These scores are ranking units with zero weight in customer confidence.
[Asset review](commercial-model-audit.md)

Transfer uses sixteen native twelve-second LibriSpeech hosts from eight
readers in **four connected reader/donor components**. Each host has one
960 ms insert from each of two retained stock generators, plus five
ordinary counterparts covering untouched speech, same- and different-reader
splices and two resampling controls. All appear clean and with three
recorded background sessions at a target-to-background ratio of −10 dB.
Background-only excerpts enter separately. The 496 deliveries contain
**32 synthetic events repeated across four conditions**, producing 128
synthetic, 320 ordinary and 48 background-only observations.

The transfer is fresh to the fitted correction. All inputs had prior
earlier-copy comparison use, and 124 deliveries had prior standalone
recurrent-view testing. The other 372 are additional standalone deliveries.
Every reused baseline reading reproduced exactly. Generator families and
voices appear in both cohorts. Transfer backgrounds differ from fitting
captures. Cross-corpus physical reader identity and pretraining overlap
remain unresolved. Counts represent four connected transfer components
and three background sessions, with repeated prompts and conditions.
[MLS, CC BY 4.0](https://www.openslr.org/94/) ·
[LibriSpeech, CC BY 4.0](https://www.openslr.org/12/) ·
[DEMAND, CC BY-SA 3.0](https://zenodo.org/records/1227121) ·
[AMI, CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Frozen results

Every cell gives **located synthetic edits / 32 · ordinary alerts / 80**.
All three alternatives retain the current confirmed two-second intervals.
The raw short arm uses its earlier frozen point. The two learned columns
use their respective wider calibration points. A useful interval covers
half the planted edit and reaches 0.30 temporal IoU.

| Condition | Current two-second | Add raw 750 ms | Add score-only head | Add acoustic head |
| --- | ---: | ---: | ---: | ---: |
| Clean | 16 · 0 | 28 · 0 | 30 · 0 | 28 · 2 |
| Washing machine | 5 · 0 | 8 · 0 | 13 · 0 | 5 · 0 |
| River | 0 · 0 | 0 · 0 | 0 · 0 | 0 · 5 |
| Restaurant | 0 · 5 | 0 · 5 | 0 · 10 | 0 · 20 |

The raw supplement adds twelve clean locations across all four components
and three washing-machine locations across three components. Both
generators contribute gains, and every addition clears its matched
ordinary edit clocks. Its misses are 4/32 clean, 24/32 washing-machine,
32/32 river and 32/32 restaurant. Standalone raw windows locate 27 clean
and four washing-machine edits, demonstrating why the existing ranges
remain valuable.

The score-only wider supplement adds eight washing-machine locations
while adding five restaurant ordinary alerts from one component. River
and restaurant locations remain zero. This secondary result needs separate
confirmation before its changed point could support a serving decision.

The acoustic wider supplement adds zero useful noisy locations. It adds
fifteen restaurant ordinary alerts across three components. Its stricter
point also yields twenty restaurant ordinary alerts and zero restaurant
locations. The predeclared noisy-gain and ordinary-alarm requirements fail.
The acoustic correction stays in development.

Unassessed synthetic counts remain 13/32 under river noise and 11/32 under
restaurant noise in the combined arms. Ordinary unassessed counts remain
31/80 and 29/80, respectively. Misses include those unassessed files.
Every arm flags zero isolated backgrounds, and 46/48 remain unassessed in
the combined arms. The wider acoustic arm produces off-target-only alerts
on two river and eight restaurant synthetic files. The original scan has
two off-target-only restaurant alerts.

## Reproduction and limits

Separate local reviews checked all 864 fitting, calibration and transfer
PCM deliveries, 11,408 transfer short-window clocks and 5,903 admitted
two-second bindings. An independent FFT implementation reconstructed all
11,683 admitted feature vectors to within 1.78 × 10⁻¹⁵. A separate SLSQP
fit reproduced both heads within 5.18 × 10⁻⁷ per coefficient, including
gradient checks and frozen points. A second recount reproduced all 496
outcomes and the gate. Nine authored timing and admission checks passed.
Transfer used 3,028 classifier forwards on two CPU threads in 21.6 minutes.
The pretrained backbone and product weights stayed unchanged.

The public receipt contains aggregate outcomes, source-group denominators,
method settings and one opaque whole-study commitment. Media, individual
identifiers, recording fingerprints, exact case clocks, response vectors
and fitted head weights stay private. The public reviewer verifies
aggregate arithmetic. Full native, fitting and case-level replay was
completed locally.
[Aggregate receipt](evaluation-panels/audio-noise-matched-segment-transfer-2026-11-18.json)

```sh
python3 -B -S backend/scripts/review_audio_noise_matched_segment_transfer.py \
  docs/evaluation-panels/audio-noise-matched-segment-transfer-2026-11-18.json
```

The raw shorter-window finding supports examining its source-bound
application seam. Exact extra off-target ranges and paid-flow evidence
need review before customer activation. The comparison leaves the
severe river and restaurant localization gap unresolved.
