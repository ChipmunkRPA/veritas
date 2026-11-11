# A separate Turkish synthesis source ran offline

The frozen Okur graphs loaded on the Mac with the existing isolated ONNX Runtime. Four of eight authored sentences produced qualified three-to-four-second waveforms. The other four stayed unavailable because their predicted duration fell outside the fixed interval. The first qualified waveform and its prepared PCM repeated byte-for-byte. All four qualified clips use one stock voice. [Method](audio-okur-turkish-v240-method.md) · [Verified aggregate](evaluation-panels/audio-okur-turkish-complete-v240.json)

The guarded worker completed 19 graph calls in 2.34 seconds, including the repeat, with 193.4 MiB maximum sampled descendant memory. It made zero network attempts and zero authenticity detector calls. The clips supply an independent synthesis architecture for a small matched-noise challenge. They are source preparation results, so detection performance and calibration remain unmeasured.

## Larger stock-voice batch

Twenty of 24 additional authored sentences qualified under the same frozen duration and decode rules. Four were outside the duration interval. The first qualified follow-up repeated exactly, and every qualified waveform differed from the earlier batch by its exact native and prepared fingerprints. The two batches now supply 24 qualified utterances from the same stock voice. The guarded follow-up completed 67 graph calls in 7.62 seconds, with zero network attempts and zero detector calls. [Follow-up method](audio-okur-turkish-followup-v245-method.md) · [Verified follow-up aggregate](evaluation-panels/audio-okur-turkish-followup-complete-v245.json)
