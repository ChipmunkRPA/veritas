# Longer video timing and unsupported gaps

The app now stops an offset comparison at any unmeasured window. Missing
picture timestamps, insufficient speech, and unavailable mouth measurements
can all break that sequence. A finding with four measured windows split into
two runs of two says that offset changes were unassessed and gives the longest
measured run. Each measured interval retains its local timing evidence.

## Matched longer video

Two three-second GRID utterances from each of eight retained source components
formed 104 six-second views. Every condition used the same join at three seconds.
The 32 neural views used existing, pixel-verified edits from two branches.
The known mechanical interventions copied source pictures or audio samples,
including a 320 ms displacement inside a 2.28-second interval. Four types of
ordinary resampling or brightness changes, genuine joined originals, and
constant audio delay served as controls. The worker bound all 15,600 output
pictures and all 9,984,000 decoded audio samples to their native clocks. It
ran the actual timing adapter with one observed face per frame and tested an
explicit track split at the join separately. [GRID license and source](https://zenodo.org/records/3625687) ·
[Prior construction and generator bindings](grid-native-clips-v149-method.md)

| Condition | Views | Pooled views with four or more measured windows | Prior timing flags | Prior localization at IoU ≥ 0.20 |
| --- | ---: | ---: | ---: | ---: |
| Known neural mouth redraw | 32 | 29 | 4 | 3 |
| Known mechanical timing intervention | 24 | 21 | 0 | 0 |
| Ordinary controls | 48 | 43 | 3 | — |

Both candidate measurements, contrast-normalized mouth motion and agreement
between two lag estimates, localized zero interventions. Motion flagged one
ordinary control and agreement flagged zero. The explicit split-track route
flagged zero cases. The edit masks establish local neural redrawing. An
earlier independent movement audit measured whether the redraws achieved
their requested timing. [Achieved movement audit](grid-mouth-motion-v158-result.md)

After the gap fix, a retrospective replay reran all seven prior flags,
all joined originals, and all long local soundtrack displacements. All 22
views retained the same number of measured windows. The fix removed three
ordinary flags, including one original and two resampling controls. It also
removed four neural-view flags whose comparison crossed unsupported windows.
The three formerly qualifying redraw overlaps had union IoUs of **0.388350,
0.283688, and 0.312500**. The fourth was 0.136054. All 15 previously
unflagged views in the replay remained unflagged.

For the 0.388350 historical result, the offline verifier matched the old
3.740–5.000 s flag with native PNGs at 3.720, 4.360, and 5.000 s. It
separately verified the overlapping 4.080–4.880 s known edit with PNGs
at 4.040, 4.480, and 4.880 s. The overlap does not validate an offset
comparison across the missing measurements. [Offline frame verifier](../backend/scripts/verify_native_video_interval.py)

## Continuous camera speech

The AMI publisher's camera/headset correspondence and speech annotations
selected four 16-second cuts from two sessions before scoring. Each cut
provided a recorded reference, ordinary recompression, a 320 ms local audio
displacement within the 5.000–10.000 s interval, and a constant 320 ms delay.
All 16 matched views verified 6,400 decoded pictures and 4,096,000 audio
samples. The source camera times and corresponding headset sample ranges were
rechecked. The publisher's common-clock description supplies the alignment
assumption. [AMI release and license](https://groups.inf.ed.ac.uk/ami/download/) ·
[Camera and microphone mappings](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml) ·
[Research consent](https://groups.inf.ed.ac.uk/ami/corpus/ethicsandconsent.shtml)

The current route retained two or three windows in four views and had no
timing-change opportunity on this panel. The most speech-active clean camera
cut supplied 23 full-search windows before the automatic speaking-face gate.
A separate diagnosis used its published camera/headset correspondence to
compare those windows without the automatic gate. The aperture rule flagged
the original and the constant-delay control. Its union IoU on the local
displacement was **0.178**, below the frozen localization point. The
mouth-motion rule gave zero flags. No attribution bypass or new timing
candidate was adopted.

These tests used one Mac, eight known GRID components and two AMI sessions.
They establish a concrete failure mode and the behavior of the local fix.
The continuous panel still supplies no validated sensitivity estimate for
the production timing check. [Aggregate counts and receipt commitments](evaluation-panels/video-longer-timing-transfer-2026-11-11.json)
