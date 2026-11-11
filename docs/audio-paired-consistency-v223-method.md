# Train a speech score to hold up across matched background noise

The fresh two-generator comparison found large losses at equal speech and
background power. Mild spectral conditioning and the rollback classifier
failed on the shared river excerpts. This experiment changes the fitted
scoring objective using existing, guarded feature packets.
[Fresh noise results](audio-fresh-shared-noise-score-v219-result.md) · [Conditioning result](audio-noise-conditioning-v211-result.md) · [Rollback result](audio-rollback-crossover-v220-result.md)

Use the earlier v84 English panel's fixed fitting, calibration and
selection roles. Keep the clean and environmental noise conditions at
0 and +5 dB. Each role contains 348 planned variants of 116 four-second
foregrounds. Verify exact packed feature hashes, source correspondence,
and disjoint known source groups and noise recording identities across
roles. Keep missing speech windows in their original denominators.
[Earlier feature panel](audio-combined-confidence-v84-protocol.md)

Fit three small weighted logistic heads with the same row weighting and
fixed ridge penalties. The first uses the current detector's raw logit.
The second adds the fixed 4,608 pooled speech features, normalized using
fitting rows. The third uses those same inputs and adds a penalty on score
drift. For each assessed clean/noisy pair, subtract the clean logit from
the noisy logit, square the difference, and weight by truth, generator
family, connected group, source and assessed noisy variant. The fixed
penalty strength is 0.15. The pairing and noise condition enter fitting
only. At prediction, each head receives features from one clip.

Set one pair of diagnostic direction points per method on the separate
calibration role. Use the number of genuine groups flagged AI and synthetic
groups flagged original by the recorded current route across both noisy
conditions. For each method use strict score tails that meet those group
budgets. Tied group extrema stay at the boundary. If the two tails overlap,
give the AI boundary precedence and shorten the original tail. Use the
same points for all three selection conditions. These are score-only
directions for a fixed comparison. The recorded current result also
includes its original stability and applicability checks.

Compare both noise levels separately and together, and retain clean
performance. Report AI detections, false AI, wrong-original calls,
inconclusive and unavailable cases, connected-group errors, raw ranking,
and paired score drift. Include the frozen current, earlier feature
combination and global fusion. The comparison uses previously assessed
English sources. Its four-second input extent and fixed environments bound
its scope. A promising candidate needs native replay and a separately
frozen fresh comparison before any customer confidence or evidence change.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_paired_consistency_v223 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_paired_consistency_v223 run
```

Publish aggregate counts and whole-artifact commitments. Saved speech
features, individual scores and fitted parameters stay private.
