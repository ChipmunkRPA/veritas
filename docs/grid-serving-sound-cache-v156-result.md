# Background sound check verified in local serving

All eleven matched analyses completed again with the sound dependency's cache
inside the isolated APFS working directory. The background sound route ran
successfully in **11 of 11** cases; the [previous batch](grid-serving-baseline-v154-partial-coverage.md)
recorded a cache failure in all eleven. The route supplies background sound
observations and remains separate from the speech authenticity estimate.

The serving replay verified sampled picture bytes, native picture clocks,
decoded PCM, sample counts and audio origins for every case. The separate
[readout](grid-serving-readout-v159-method.md) replay verified visual input,
timing input and raw timing bindings in all eleven cases. The frame authenticity
route still skipped on the unchanged activation budget. All eleven clips were
outside the saved validation envelope, so the app did not issue an overall AI
or original direction.

The worker ran for 151.69 seconds, reaching 5.13 GiB maximum sampled descendant
RSS, with no monitor guard stop or sampling error. The [actual receipt](evaluation-panels/grid-serving-sound-cache-verified-v156.json)
records aggregate coverage and private artifact commitments. This single-source
batch verifies the serving fix. It does not validate sound classifications,
video authenticity or a change to overall confidence. App defaults stayed fixed.
