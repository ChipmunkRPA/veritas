# Acoustic prediction of visible mouth states in single-video timing

The new check localized **0 of 108** known timing shifts and added **4 alerts across 162 ordinary controls**. The current check also localized none and flagged no controls. All four new alerts came from local low-pass filtering. The new check remains experimental. [Held results and independent replay](evaluation-panels/grid-articulatory-timing-2026-11-12.json)

The comparison learns two visible states from clean speech and tests whether the states and soundtrack agree at a different lag in the middle of a clip than on both sides. It looks for local changes in timing. The two visual states group many articulations together and supply no phoneme identity or AI authorship score. Recent research separates timing alignment from finer articulation and motivates testing both. This implementation uses the existing source-bound landmark runtime and no code or checkpoint from those studies. [PVSync](https://arxiv.org/abs/2610.09223) · [Phoneme and geometry comparison](https://arxiv.org/abs/2510.14241) · [Earlier local timing trial](video-lipsync-local-contrast-transfer-2026-11-12.md)

## Frozen panel and method

The 77 genuine original GRID recordings came from eight retained publisher archives. Deterministic hash order excluded previously planned reserve utterances and exact members of the prior independently labeled release panel. Speaker groups had prior project exposure. The new study separated the speakers into 36 fit clips across three groups, 14 ordinary calibration clips across two groups, and 27 held clips across three groups. Each recording retains 75 native pictures at 25 fps and 47,648 continuous mono samples at 16 kHz. Source media stays in the private run. [GRID publisher and terms](https://zenodo.org/records/3625687) · [Earlier source and clock qualification](grid-reserve-word-clocks-2026-11-12.md)

Eye-normalized median aperture across three inner-lip gaps defines clearly closed and open pictures. Intermediate geometry and missed observations abstain. Fit a balanced regularized acoustic head only on genuine aligned fit pictures. Its inputs are twelve relative spectral bands and relative energy across seven Hann windows around each picture. It uses no transcripts, earlier video or recordings from the calibration or held groups. The source-bound timing comparison searches for a 14-, 18-, or 22-picture range in which a different lag better matches the visible states. Both ten-picture flanks must favor the same reference lag. Each comparison uses identical active observations under both lags, and the interior must contain both open and closed pictures. Rank by the smallest of the three signed gains.

Four signed audio edits per original copy waveform samples 200 or 280 ms earlier or later in an authored local interval. The picture sequence and sound outside the authored interval stay exact. Each transition within the interval uses a ten-millisecond blend. Six control waveforms use the original recording, local gain, local low-pass filtering, local noise, and two constant whole-track offsets. The controlled waveforms enter the timing measures on the same native clock as the source pictures. The fixed gate is strict excess over the largest ordinary calibration response. A useful range requires at least 0.40 temporal intersection over union and signed displacement within 80 ms. Advancement requires three additional correct ranges across at least two held groups, at least 75% coverage in both main conditions, at most one new ordinary alert and no new alerts on constant offsets.

The separate neural panel reuses the eight verified original, neural-redraw and ordinary-retime files from two held speaker groups. Four views contain an independently observed signed shift and use the known native picture edit. The two synchronized redraws test whether the timing check raises a range on that condition. Neither their material nor the corresponding speaker groups entered fit or point selection. [Native neural panel and exact evidence](grid-reserve-neural-mouth-2026-11-12.md)

## Results

| Held outcome | Current timing | Mouth-state timing |
| --- | ---: | ---: |
| Signed edits measured | 100/108 | 103/108 |
| Correct signed and localized ranges | 0/108 | 0/108 |
| Ordinary controls measured | 155/162 | 155/162 |
| Ordinary alerts | 0/162 | 4/162 |
| Alerts on constant offsets | 0/54 | 0/54 |
| Correct locations for known neural and ordinary picture shifts | 0/4 | 0/4 |

The state prediction reached ROC AUC **0.806** on 1,171 eligible pictures in the 27 held genuine originals. The training set had 232 closed and 1,268 open observations. The held set had 279 closed and 892 open observations. These are repeated pictures within three groups and measure state prediction on this panel. They give no estimate of detector accuracy on independent uploads.

Before applying the frozen point, the strongest candidate had the correct sign on **58 of 103** measured signed edits, a signed error within 80 ms on **46 of 103**, and sufficient interval overlap on **50 of 103**. **23 of 103** met both the signed error and overlap criteria. The highest useful response was 0.302816, below the frozen point of 0.410857. Twelve ordinary views across all three held groups equaled or exceeded that useful response. They include eight low-pass controls, an untouched original, a gain change and both constant-offset directions. Across 26 paired observations, local low-pass filtering increased the median strongest response by 0.114. The four authored shift variants changed it by medians of 0.002–0.006 against their measured originals. The ranking leaves no operating point with a useful location and a similar observed control alert count.

The eight redraw panel files all had measurement support. None alerted at the frozen point. All four measured signed picture shifts missed both interval overlap and the signed tolerance even before thresholding. In the two source groups, the best neural and ordinary shifted-picture candidates had identical range and lag outputs. The current visual states can discriminate opening from closing while losing the identity and order of the underlying speech events. This is evidence to test narrower articulation identities and processing stability separately in future work. The existing app timing detector is unchanged.

The replay independently checked all **85 original and redraw files**, **6,375 native pictures**, **418 waveform commitments** and **12 exact native mouth crops**. A repeat run reproduced every fit coefficient, measured response, case disposition and source evidence commitment. The public reviewer replays the frozen point, paired processing effects and signed location counts with standard Python.

## Reproduce

Use the retained rights-qualified GRID archives, the native landmark runtime and existing timing fixtures. The local macOS landmark runtime needs a usable graphics context to initialize its CPU delegate. The public receipt retains anonymized timing and outcome records. The optional native audit independently reopens the selected archive members, decodes original picture and sound and rebuilds every controlled waveform. [Existing native runtime and asset qualification](grid-paired-closure-timing-2026-11-12.md)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze fit point held neural; do
  backend/.venv/bin/python -B -m scripts.grid_articulatory_timing "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_grid_articulatory_timing "$receipt" --run "$run"
python3 -B -S backend/scripts/review_grid_articulatory_timing.py "$receipt"
```
