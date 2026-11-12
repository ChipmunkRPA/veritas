# Pitch-aligned excitation checks on short voice edits

The new acoustic check missed all ten held short AI edits under louder competing speech. It also missed all six held edits mixed with measured AMI meeting speech. At equal false-alarm counts it located fewer edits than the existing voice check. [Aggregate results and commitments](evaluation-panels/audio-periodic-excitation-transfer-2026-11-12.json)

## Method and source roles

The earlier phase comparison measured aggregate coupling among fixed frequency bins. This test estimates a voiced period in each 64 ms frame and measures phase and periodicity at harmonics of that period. A twelfth-order linear predictor estimates the short-term spectral envelope. The residual gives an approximation of the excitation signal. Features measure residual cycle repetition, phase curvature, crest factor, and changes between adjacent frames. A 200 ms cell requires repeated supported frames. An L2-regularized head uses eight features per cell and their differences from nearby supported cells. This signal can be influenced by whichever periodic component dominates the frame, including competing speech. It cannot identify the target speaker. The implementation adds no checkpoint or external algorithm code. [Prior whole-file phase experiment](audio-phase-corroboration-2026-11-11.md) · [Period estimation research](https://www.ee.columbia.edu/~dpwe/e6820/papers/deChevK02-yin.pdf) · [Glottal inverse filtering research](https://www.isca-archive.org/interspeech_2024/freixes24_interspeech.html) · [Implementation](../backend/scripts/audio_periodic_excitation_transfer.py)

Reuse the earlier exact qualified 316-view panel, with separate source groups for fitting, operating-point selection, diagnostic and held evaluation. Train on 72 views from eight matched fitting hosts. Each matched comparison includes a same-host genuine splice, another genuine speaker, and an AI insert, with matched fades, gain and background. The fitting objective uses the fraction of each cell occupied by the insert. Expressive synthesis stays out of both fitting and point selection. Freeze the point at the maximum across 46 measured genuine point-selection files. The held strong-interference condition has ten AI edits and 34 genuine controls, including unedited and isolated-background recordings. Apply the fixed head and point to the 48 six-second AMI-derived files, including six held AI and six matched genuine inserts under strong room speech. The exact panel and generator families had prior project exposure. Source roles, provenance and PCM commitments remain fixed. [Matched-source qualification and rights](audio-partial-residual-transfer-2026-11-12.md) · [AMI source conditions](audio-overlap-window-transfer-2026-11-12.md) · [AMI Corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

For each file, select a 0.6-second range from the highest-scoring three successive cells. At least two of those cells must have qualified periodic measurements. Retain their exact sample-clock subranges. A useful location requires at least 0.30 intersection over union and 0.50 coverage of the exact edit. Compare the frozen points first. Retrospective comparisons use identical numbers of genuine source groups falsely flagged, counting a repeated genuine group once. The AMI fixed-point comparator is the earlier 0.6-second incumbent frame check. Its raw scores were not retained. The AMI score-ranked comparison therefore uses the earlier measured two-second check and preserves the width difference in the result. [Partial-edit localization research](https://arxiv.org/abs/2604.02913)

## Local results

| Held condition | Incumbent useful / genuine false | Excitation useful / genuine false | Best range overlaps edit before applying a point, incumbent / excitation |
| --- | ---: | ---: | ---: |
| Dry, 10 edits and 20 genuine views | 2 / 0 | 0 / 2 | 8 / 1 |
| Competing speech at −5 dB, 10 edits and 34 genuine views | 0 / 0 | 0 / 0 | 5 / 1 |
| Competing speech at +15 dB, 10 edits and 20 genuine views | 1 / 0 | 0 / 3 | 9 / 0 |
| Clean AMI-panel hosts, 6 edits and 6 genuine views | 2 / 1 | 0 / 0 | 5 / 0 |
| Recorded AMI room speech, 6 edits and 6 genuine views | 0 / 0 | 0 / 0 | 1 / 0 |

In the −5 dB panel, the incumbent localized two edits with one genuine false group and still localized two with two false groups. The excitation check localized zero at both counts. The edited clip scored above both matched genuine splices in 7 of 10 comparisons for the incumbent and 2 of 10 for excitation. The equal-false-alarm AMI comparison gave one useful location for the prior two-second check and zero for excitation at each of those counts. Stronger relaxation of the point would not correct the candidate’s poor ordering or location. Every held clip had at least one eligible range, so the result was driven by discrimination and localization. The excitation cue stays experimental and contributes no customer-facing score or finding.

## Verification

The CPU report stage scored 216 diagnostic and held inputs in 5.3 seconds after fitting, without a new model forward. A deterministic sine-harmonic control returned the exact period on every frame. Digital silence produced no qualified cells. The runner replay recalculated the features, subranges and counts. A separate standard-library reviewer reopened the exact PCM for all 216 observations and reconstructed the operating point, source-clock overlap, and false-alarm arithmetic. It rejected a one-sample range alteration after the outer observation hash was updated. The aggregate contains no audio, per-file responses, or fitted weights.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit report audit; do
  backend/.venv/bin/python -B -m scripts.audio_periodic_excitation_transfer "$phase" "$run"
done
python3 -I -S backend/scripts/review_audio_periodic_excitation_transfer.py "$run" "$parent" "$meeting"
```

Use the retained four-second partial-residual and final six-second overlap-window directories as `parent` and `meeting`, respectively. Supply `--parent` and `--meeting` to the runner if those directories are at a different location.
