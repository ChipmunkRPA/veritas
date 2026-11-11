# Audio and picture timing engineering fixture

The local fixture runs the existing timing grid, face and speech window
eligibility, full lag search, Pearson correlation and track change rules.
It supplies deterministic mouth-aperture and audio-level arrays and declares
synthetic face and speech support. It measures the behavior of those rules
when support is present. The numeric arrays require no media decoding, model
inference, or customer file. [Fixture](../backend/scripts/av_timing_engineering_fixture.py)

Run from the repository root. Add `--windows` for the measured lags and all 31
correlation values at each window.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.av_timing_engineering_fixture
```

## Exact window support

The 190-tick arms represent 7.6 seconds at 25 Hz and 121,600 samples on an
implied 16-kHz PCM clock. They admit start ticks
`25, 37, 50, 62, 75, 87, 100, 112, 125, 137, 150`. Each start `s` uses mouth
ticks `[s,s+25)` and audio search support `[s-15,s+40)`.

The local copy replaces output ticks `[77,120)`, or samples
`[49,280,76,800)`. An eight-tick delay uses the earlier reference interval
`[69,112)`. An eight-tick advance uses `[85,128)`. Both preserve the
authored level sequence outside the intervention. The full search lies wholly
before the intervention for starts 25 and 37 and wholly after it for 137 and
150. The other seven searches cross an edit boundary. Even with that mixed
search context, the +8 arm has exact copied support at its intended lag for
starts 75 and 87, and the −8 arm for 87 and 100.

The flat correlation arm has 227 ticks. Its authored audio level is constant
and positive during `[77,152)`. Starts 100 and 112 have their complete search
inside that interval. Starts 25 and 37 have clean lead-in, and starts 175 and
187 have clean trailing support. Speech availability is supplied to this arm
as an engineering input. The unchanged relative-collapse rules make the
measurement.

## Local result

| Authored input | Measured windows | Observed reading | Returned timing range |
| --- | ---: | --- | --- |
| Same time | 11 | In sync, lag 0 | — |
| Global +320 ms | 11 | Constant offset, lag +8 ticks | — |
| Local +320 ms | 11 | Offset step, +8 ticks from baseline | 2.74–4.48 s |
| Local −320 ms | 11 | Offset step, −8 ticks from baseline | 3.24–4.74 s |
| Local flat correlation | 14 | Correlation collapse | 3.48–6.00 s |
| Speech unavailable | 0 | Unassessed | — |

The exact copied audio interval was `[3.08, 4.80)` seconds. Comparison with
each raw flagged span uses exact rational half-open intervals. Signed errors
subtract the authored boundary from the flagged boundary.

| Raw timing flag | Temporal IoU with copied audio | Signed onset error | Signed end error |
| --- | ---: | ---: | ---: |
| Local +320 ms | 70/103 | −340 ms | −320 ms |
| Local −320 ms | 75/86 | +160 ms | −60 ms |

Mixed windows selected intermediate or other lags.
The two wholly flat windows measured zero correlation at every lag. The
collapse also included two neighboring windows with partly flat support.

The separate articulation input uses six authored `/m/` closures with aligned
audio sample intervals. The same-time case measured a local-change interval of
−80 to +80 ms. Delaying the two middle events by 320 ms produced a supported
change interval of +240 to +400 ms. The wider-gap variant still matched all
six events and abstained on local change because the event separation broke
the comparison. These are clock and support checks on supplied events.
[Articulation core](../backend/scripts/mouth_phone_timing_core.py)

Seven fixture tests, sixteen articulation tests, and scoped Ruff checks passed.
The results describe synchronization arithmetic, including its boundary error
on designed inputs. Applying the method to genuine speech requires measured
face support, independent speech and speaker attribution, and timed phone
qualification. The fixture provides no AI attribution or serving calibration.

## Why a stitched GRID pair is a cut control

The completed pilot metadata records sixteen originals, two per source for
eight verified source groups. Each has 75 usable dense YuNet observations,
2.96 seconds between the first and last picture, and 47,648 audio samples.
The unchanged timing adapter had one possible full-search start per clip,
at tick 25. It measured that window in fourteen originals. The other two
lacked a supported speaking-face attribution. [Pilot receipt](grid-native-mouth-scores-v155-result.md)

Two same-source utterances can be trimmed to 74 pictures and 47,360 samples
each, then joined on the exact 40 ms grid. With the join at tick 74, only
starts 25 and 100 have their entire audio searches within their respective
utterances. The others mix across the cut in the picture window or the audio
search. Two clean windows cannot exercise the two stable runs or the relative
collapse rule. A paired join can test cut handling and abstention. A
continuous single-shot recording with independently established support is
needed to test natural timing changes with clean lead-in and trailing context.
