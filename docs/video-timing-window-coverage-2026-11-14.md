# Speech and mouth timing coverage in video reports

New Full and Deep analyses can report why each scheduled one-second
window did or did not reach a mouth/audio timing measurement. The
window can lack sufficient speech, picture timestamps, audio search
margin, a usable face track, varying mouth motion, or an attributable
speaker. The coverage records the furthest stage reached across face
tracks for each window. A measured window may still be inconclusive.
The existing timing search, attribution rule, and scores are unchanged.

The evidence page lists unassessed windows on the original source clock.
The PDF shows up to twelve unassessed ranges, and the JSON preserves
the complete schedule within the decoded audio limit of 120 seconds.
Each JSON row carries exact sample bounds, source timestamp ticks, and
the number of speech ticks. Silence, an absent or too-short soundtrack,
and discontinuous audio clocks still use the channel-level coverage
message where this schedule cannot run.

## Local comparison

Three deliveries use one archived CREMA-D recording. The moving and held
deliveries preserve the same soundtrack. The off-camera control replaces
the whole picture with black while preserving the sound. [Source
qualification and license](video-lipsync-mouth-coverage-2026-11-11.md#native-test)

| Delivery | Measured | Insufficient speech | Incomplete audio margin | Unchanged mouth crops | No tracked face |
| --- | ---: | ---: | ---: | ---: | ---: |
| Moving face | 3 | 2 | 1 | 0 | 0 |
| Held mouth | 2 | 2 | 1 | 1 | 0 |
| Off-camera speech | 0 | 2 | 1 | 0 | 3 |

The held-mouth delivery identifies **1.480–2.480 s** as unassessed because
the mouth crops were identical. All three deliveries ran through the local
worker after simulated Stripe payment. No analysis job was created on the
unpaid event. The evidence API, JSON, PDF, and signed package agreed.
These are coverage checks on controlled deliveries from one recording.
They give no estimate of lip-sync deepfake detection accuracy.

## Independent replay

The public package verifier checks the coverage against the signed result
commitment. With a separately supplied original video, it can also
recompute speech activity and picture timestamp coverage. Mouth motion
eligibility and speaker attribution are recorded observations from the
analysis. Those decisions are not rerun by the source verifier.

```sh
PYTHONPATH=backend python -m app.cli verify package.zip --public-key trusted.pem \
  --source-audio original.mov --source-video original.mov --json
```

The reproducible encoded fixture at
`backend/tests/test_video_nested_evidence.py::test_unmeasured_timing_windows_reach_the_pdf_and_signed_export`
exercises a shifted source clock, the report, and the signed replay
without private media or model weights.
