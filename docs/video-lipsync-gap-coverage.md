# Lip-sync checks now preserve missing-picture coverage

The production 25-Hz lip-sync grid previously assigned the nearest source frame
to every tick between the video endpoints, including large timestamp gaps.
An authored two-second fixture with a 1.24-second picture gap admitted two
one-second scoring windows. Some assigned pictures were 600 ms from the audio
tick. These repeated pictures could affect mouth-motion/correlation readings
despite no observed source picture at those times.

Each tick now requires a source picture within one 25-Hz grid period, 40 ms.
Uncovered ticks retain the existing unavailable-frame marker, and any window
containing one is excluded before crop reads or model access. This is an input
coverage allowance, not a manipulation score or public screening threshold.
A long held picture may be valid video playback, but does not supply observed
mouth motion at the missing picture times. Low-cadence material can therefore
have fewer assessable windows; it is not called original or AI for that reason.

Both adapters now explain that a full speech window needs timestamp-matched
video frames. Their implementation/manifest versions advance together so new
runs identify the changed coverage policy. The default commercial model-free
lane stays enabled; the restricted optional embedding lane stays disabled.
No model, confidence map, decision point or license policy changed.

The timing/evidence suite passes **39 checks**, including the new gap cases,
unchanged nearest-frame routing at 24/25/30/60 fps, partial-window preservation,
single-picture support, nonzero audio/video origins and exact evidence bindings.
Both adapters skip the long-gap fixture with no frame store, model lease or
exhibit sink available, demonstrating that the skip precedes those operations.
The separate customer-assessment/export suite passes **79 checks**, including
skipped/inconclusive scope handling. Manifest and adapter versions match.

```sh
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_video_lipsync_frame_gaps.py backend/tests/test_video_evidence_timing.py backend/tests/test_video_av_evidence.py
PYTHONPATH=backend backend/.venv/bin/pytest -q -o addopts='' backend/tests/test_customer_assessment.py backend/tests/test_customer_assessment_exports.py
```

These are software/controlled-media checks, not fresh deepfake error-rate
measurements. Historical evaluations remain at their recorded commits and
parameters. New native generation/articulation evaluation is still required;
this coverage fix does not detect well-synchronized retalking by itself.
