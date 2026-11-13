# Blind speech-phone and mouth-shape timing on short video

Earlier short-video experiments found that both broadband correlation and a learned open/closed mouth score missed known offsets. This experiment tests whether particular speech sounds provide a stronger time reference. It independently implements a small check using retained local speech and face tools. It uses no code or weights from the cited lip-sync research. [Open/closed study](grid-articulatory-timing-2026-11-12.md) · [Phoneme-temporal research](https://arxiv.org/abs/2510.14241) · [Timing and articulation research](https://arxiv.org/abs/2610.09223)

## Frozen inputs and comparison

Select 66 exact, unused MPEG members in deterministic hash order from eight previously retained GRID speaker archives. Exclude exact utterances from the previous source plans, release qualification and open/closed study. These speaker groups had earlier project exposure. Fit on 30 recordings from three speakers, set the point on 12 recordings from two other speakers and test 24 recordings from three further speakers. Each input must retain 75 exact native pictures at 25 fps and 47,648 continuous mono samples at 16 kHz. The first attempt found one point source with only 74 native pictures and stopped before selecting a point or scoring any held views. The corrected freeze admits the next native-clock-qualified source in the original order and keeps the same fit, method and held roster. [GRID publisher and CC BY 4.0 terms](https://zenodo.org/records/3625687)

Decode English phones on the original soundtrack and again at 68% gain without supplied words, a grammar or an earlier video. Keep only exact phone agreement at a native picture midpoint, at least 10 ms inside both decoded intervals. Group selected phones into closed lips, rounded lips, spread vowels and open vowels. A balanced logistic fit predicts those four acoustic labels using only the three eye-normalized mouth gaps and the mouth width relative to the clip median. The recorded decoder timing is approximate and is subject to recognition errors and natural variation in articulation. [Decoder documentation](https://cmusphinx.github.io/wiki/phonemerecognition/) · [Pinned decoder terms](https://github.com/cmusphinx/pocketsphinx/blob/v5.0.4/LICENSE) · [Acoustic model notice](https://github.com/cmusphinx/pocketsphinx/blob/v5.0.4/model/en-us/en-us/README) · [Reviewed face task and source terms](grid-reserve-neural-mouth-2026-11-12.md)

Search for a 560, 720 or 880 ms middle interval where the mouth shapes favor a different lag from both 480 ms flanks. Compare identical native pictures with usable and different phone families under both lags. A proposal requires three informative middle pictures and two per flank. The proposal's score is the smallest of the middle and two flank gains. The strict point was fixed from the strongest response across 72 ordinary calibration views before held decoding. A correct result needs at least 0.40 interval intersection over union and signed error at most two native pictures. The current comparator uses the serving arithmetic with a directly observed single face. [Existing timing logic and controlled waveform recipe](video-lipsync-local-contrast-transfer-2026-11-12.md)

The 24 held originals contribute 96 local audio shifts of ±200 or ±280 ms and 144 controls. The controls cover original speech, local gain, low-pass filtering, mild local noise and constant whole-track shifts. All local audio treatments preserve native pictures and samples outside their declared range. The frozen promotion criteria require four new useful locations across three source groups, at least 60% assessed coverage, no more than one added control flag, zero constant-offset flags and preservation of current useful locations.

The separate eight-view redraw comparison tests two previously verified speakers. It includes four signed timing edits and four original or synchronized views. Neither speaker enters this experiment's fit or point roles. Two edited views are neural redraws with a visible closure displaced 320 ms relative to the unchanged soundtrack. The other two are ordinary picture retimes with the same known offset. A shift can come from ordinary editing and carries no AI authorship weight. [Redraw source and achieved timing](grid-reserve-neural-mouth-2026-11-12.md)

## Held results

| Outcome | Current timing | Phone-conditioned shape |
| --- | ---: | ---: |
| Correct signed audio edits | 0/96 | 1/96 |
| Assessed audio edits | 92/96 | 24/96 |
| Ordinary control alerts | 0/144 | 3/144 |
| Assessed controls | 137/144 | 29/144 |
| Correct signed neural and ordinary picture retimes | 0/4 | 0/4 |
| Alerts on originals and synchronized redraws | 0/4 | 2/4 |

The fit used 480 native pictures, with 69 closed, 82 rounded, 208 spread and 121 open labels. Across 447 eligible original pictures in the held set, balanced agreement with the decoded phone families was 54.6%. The same measure was 43.5% on 240 point pictures. These are comparisons with decoder labels on correlated pictures, without independently labeled phones. Only 25 of 72 point controls supported a range proposal. The largest of those responses set the internal point at 1.3079590563.

The candidate's useful result estimates the sound moved 160 ms later over **1.040–1.760 s**, against an authored +200 ms shift over **1.120–1.760 s**. Its reported source pictures occur at **1.000, 1.400 and 1.760 s**. The measured range was 0.89 temporal IoU with 40 ms signed error. The three ordinary alerts came from a gain change, a low-pass filter and a constant early offset. No unchanged recording in the 24-source held set alerted.

The redraw panel demonstrates a different failure. Four views from one speaker had a reportable score and the other speaker had insufficient support on all four views. Both shifted pictures from the first speaker yielded a +160 ms estimate despite a known −320 ms audio-minus-picture shift. The original and synchronized neural redraw of that speaker also flagged. Both genuinely shifted pictures from the second speaker remained unassessed.

The strongest unthresholded candidate was useful on nine of the 24 assessed audio edits. The retrospective ranking gave zero useful locations at up to one ordinary flag, one at two flags, three at five flags from one speaker group, and four at eight flags across two groups. This trade-off cannot establish a new decision point. Even allowing more ordinary alerts, the gains and coverage provide insufficient support to add the check to the app. A future test should first verify blind phone clocks against independent phone labels, then measure whether a wider set of mouth features gives enough stable, distinctive timing observations across speakers and ordinary processing. [Portable held receipt](evaluation-panels/grid-phone-viseme-timing-2026-11-12.json)

Independent native readback reopened 66 published archive members and all eight redraw-panel files. It checked 5,550 native pictures, 3,525,952 decoded samples, 320 test-view PCM commitments, 24 evidence crops and the one excluded picture clock. The portable receipt contains no source media or decoded phone sequences.

## Reproduce

Use the retained reviewed GRID archives, PocketSphinx 5.0.4 wheel and stateless local face task. The native check uses FFmpeg and the optional Mac graphics context for the CPU face task. The portable reviewer verifies the point, signed interval arithmetic, paired grouping and held error trade-off. The native replay reopens the source members, re-decodes the original frames and PCM, reconstructs every waveform, and checks saved evidence. It does not rerun phone or face inference. Source media, speech labels, mouth crops and private paths stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze fit point held redraw; do
  backend/.venv/bin/python -B -m scripts.grid_phone_viseme_timing "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_grid_phone_viseme_timing "$receipt" --run "$run"
python3 -B -I -S backend/scripts/review_grid_phone_viseme_timing.py \
  docs/evaluation-panels/grid-phone-viseme-timing-2026-11-12.json
```
