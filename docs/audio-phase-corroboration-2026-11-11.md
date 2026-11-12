# Phase and magnitude checks under recorded voice interference

Both checks failed the frozen qualification. The phase check cleared none of the six genuine false AI flags under centered interference. The magnitude check cleared five, while withdrawing 13 of 18 correct synthetic flags in those scenes. No customer decision uses either check.

## Frozen design

The candidate measures phase curvature between adjacent frequency bins and normalized three-bin coupling within a three-second mono clip. Both use the same 512-sample Hann frames and focus on the higher-energy 60% of frames. A matched comparison uses spectral flatness and centroid statistics from the same frames and four bands. Fit one small, regularized linear head per measurement family. Balance both origin classes and weight foreground source groups equally within each class. The phase methods were implemented locally with NumPy and SciPy. The study imports no external method code and adds no new pretrained weights. [Bispectral speech study](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/AlBadawy_Detecting_AI-Synthesized_Speech_Using_Bispectral_Analysis_CVPRW_2019_paper.pdf) · [Modified group-delay study](https://xplorestaging.ieee.org/document/11443246/)

Ten archival foreground groups fit the two heads. Seven different groups select the highest point preserving all clean and at least 90% of mixed true AI calls. Each head may only withhold an existing AI flag and return inconclusive. The 16 held foreground groups and two held room recordings are disjoint from both roles. The held material was previously evaluated by the project in the parent stereo study. Synthetic generator families recur across roles, and model pretraining overlap is unknown. The four native held room recordings are genuine controls. [Exact parent study and rights](audio-spatial-interference-2026-11-11.md) · [Existing voice checkpoint terms](commercial-model-audit.md)

After freezing both points, make 96 new delivery variants from the 16 held clean clips and 32 matched simultaneous room mixtures. One variant uses 24 kb/s Opus. The other uses a fixed frequency-domain noise reducer and a conventional dynamics compressor. The file hashes, decoded signed PCM and three-second source clocks are recorded. Processing preserves the original source label. This stress test was motivated by research that separates original speech provenance from later processing. [Benign voice restoration study](https://arxiv.org/abs/2603.14033) · [RADAR media transformation evaluation](https://arxiv.org/abs/2605.09568)

## Results

| Held condition | Incumbent genuine false flags | Incumbent correct AI flags | Phase flags withheld, genuine / AI | Magnitude flags withheld, genuine / AI |
| --- | ---: | ---: | ---: | ---: |
| Centered native-delivery mixtures | 6/28 | 18/36 | 0 / 0 | 5 / 13 |
| Offset native-delivery mixtures | 1/14 | 3/18 | 0 / 0 | 1 / 3 |
| Opus simultaneous mixtures | 3/14 | 9/18 | 0 / 0 | 3 / 9 |
| Restored simultaneous mixtures | 6/14 | 14/18 | 0 / 0 | 6 / 14 |

The Opus and restored rows contain the same 14 genuine and 18 synthetic simultaneous mixtures. In their unmodified delivery, the current voice check flagged 5/14 genuine and 11/18 synthetic. Opus moved two genuine and three synthetic flags to inconclusive and added one synthetic flag. Restoration introduced one false flag on a previously inconclusive genuine recording and recovered three synthetic flags. These are transformations of the same recordings, so they measure delivery sensitivity rather than additional independent subjects. The phase check also withdrew one correct AI flag on the nine unmodified clean synthetic controls.

## Verification

The runner measured 313 exact views, including 96 new deliveries with 110 voice classifier forwards. An independent replay reopened every delivery, verified its clock and PCM digest, recalculated phase and magnitude features through separate arithmetic, solved both regularized fits independently, and recomputed all 48 decision cells. It made no classifier calls. Source audio, private row responses and the small fitted heads stay local. [Frozen aggregate](evaluation-panels/audio-phase-corroboration-2026-11-11.json) · [Runner](../backend/scripts/audio_phase_corroboration.py) · [Independent replay](../backend/scripts/review_audio_phase_corroboration.py)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze train score aggregate; do
  backend/.venv/bin/python -B -m scripts.audio_phase_corroboration "$stage" "$run" --parent "$spatial_run"
done
backend/.venv/bin/python -B -m scripts.review_audio_phase_corroboration "$run" --parent "$spatial_run"
```
