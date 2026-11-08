# Sound/mouth correspondence: prospective input and dispatch correction

This separate attempt preserves the complete [v13 protocol](mouth-audio-context-v13-protocol.md),
its frozen code and all results. The earlier attempt completed 160 fitting
tracks/3,560 windows and 18 ordinary point tracks. It stopped on the intact
nineteenth point clip, which has 50 frames: the envelope helper incorrectly
required the 51-frame minimum of the fitting sources. No learned head or
diagnostic audio score exists from that attempt. Both native resource receipts,
partial journals and the original failure remain unchanged and are hash-bound.

Three engineering corrections apply prospectively:

1. Accept at least 20 exact 640-sample blocks in the envelope helper, while
   retaining the identical finite float32, sample-block and native-clock checks.
   Actual source roles remain 160/24/88 with no replacements. Window features
   still require the same complete geometry, common lag anchors and minimum four
   observations per part. Explicitly reject window centers outside the source.
2. The CLI uses `fit` for source-audio extraction and `train` for the existing
   fitting function. No ambiguous dispatch or implicit extraction restart.
3. Keep `fit-complete.json` for fitting-source extraction and use the distinct
   `head-fit-complete.json` for trained heads/points before diagnostic processing.

No feature equation, lag, aperture, input selection, source role, hyperparameter,
label, ordinary point rule or advancement requirement changes. All twenty audio
features remain middle/complete-support correlations with the five fixed lags.
Both measurement groups are dependent. The original 49-variable visual-only
control, 69-variable candidate and retained-v8 priority remain as specified.

Publish this code and complete new freeze before new extraction. Re-extract
all 160 fitting and 24 point tracks, then require the original 160/18 saved
records to reproduce exactly before any fitting. Bind complete heads/points
before extracting all 88 diagnostic tracks. The original failed directory is
read-only to this attempt. No retry, head selection, new source or failure
omission is permitted. Every original comparison, control, source audit,
resource guard, independent full audio/feature/outcome replay, privacy restriction
and subsequent fresh-validation requirement in v13 remains mandatory.

The earlier app-wide source-preflight failure and separate v13 dispatch note
also remain historical records. This experiment does not claim those old
commands passed. It does not change serving confidence, model defaults, Stripe,
reports, rollback, customer files or GCP resources.
