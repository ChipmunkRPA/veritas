# Finding short soundtrack moves at a marked mouth event

The earlier-video comparison found **40 of 40** known local soundtrack moves after adding the marked check. The automatic check found **36 of 40**. The four additional locations came from two held source groups and included moves in both directions. Neither check flagged any of the 120 ordinary comparisons in this panel. [Local source replay](evaluation-panels/video-marked-sound-fallback-2026-11-12.json)

## Comparison and scope

Use 20 source events from nine held Lombard speaker groups. The source qualification separately observed an acoustic burst and a visible mouth reopening before the timing measurements ran. Each event supplies earlier and later audio moves of five or seven native picture periods, a 640 ms edit interval, and six controls. Controls preserve the original, change local gain, apply local filtering or background sound, or move the entire soundtrack in either direction. Native pictures are identical in every comparison. The 120 controls are correlated views of 20 recordings. [Publisher data and terms](https://zenodo.org/records/3736465) · [Source qualification and unmarked trial](short-lip-release-timing-2026-11-12.md)

Keep the automatic timing result when it finds a local move. For a matching upright 720 × 480 pair with 48–75 native pictures at a constant cadence between 23.5 and 25 fps, use the marked mouth closure to select a distinctive excerpt. Both copies must have the same zero-origin picture and continuous sound clocks. Require an exact decoded sound match at a displaced location, a bounded span of different samples, and identical samples before and after that span. The check tries the established 160 ms and 60 ms excerpts. An unsupported match yields no fallback finding. This measurement describes local timing and contributes zero weight to authorship confidence.

All four additional localizations reproduced the signed offsets exactly. Each is roughly 208 ms. Two included a 160 ms excerpt and two included a 60 ms excerpt. The displayed change boundaries use the first and last differing decoded samples. At a fade edge, 16-bit decoding can remove a one-sample difference.

A separate fresh visual study tested mouth-only picture changes with identical sound. A direct extension matched 24 of 34 independently observed visual shifts within one native picture. It gave zero timing flags on 40 unchanged or ordinary resampled pairs. Four matches fell on clips without an independently established reopening, including a move in the opposite direction from the authored edit. The visual extension remains experimental. The existing earlier-copy picture comparison found all six exact rendered picture-edit intervals and produced 36 before, during and after exhibits with zero changed audio samples. [Visual challenge builder](../backend/scripts/build_lombard_paired_visual_challenge.py)

## Local use and replay

In **Compare with an earlier video**, attach the earlier copy and enter the mouth closure time just before reopening. The plain view and PDF show the changed interval, both mouth crops and matched waveforms. JSON adds playable excerpts and exact sample and native picture positions. Local checks with lossless video deliveries verified all four recovered offsets against the sources, exported their PDF and JSON, and checked ordinary gain and whole-track moves. The standalone verifier accepts the new format and can still replay earlier formats.

Use the held inputs and observations from the unmarked timing study to reproduce the 160 case results.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_video_marked_sound_fallback \
  "$sources" "$run/observations-held.json" \
  docs/evaluation-panels/video-marked-sound-fallback-2026-11-12.json --verify
```
