# Native audio/picture routing: clock preparation completed

The [frozen method](av-conditioning-clock-v109-method.md) completed on all
**48 waveform sources and 3,642 native picture coordinates**. It prepares
timestamp-aware speech-token routing for future mouth generation. It does not
execute a speech encoder, render a mouth edit or measure detector accuracy.

## Timing assumptions now explicit

The old generator benchmark intentionally uses normalized 25-Hz pictures and
a common declared origin. Applying its twice-picture-index equation directly
to these native source timestamps would differ at **3,498 of 3,642 positions**,
by up to **44 speech ticks, or 880 ms**. This is a comparison of hypothetical
index equations, not a measured lip-sync error in generated media or a revision
of the old normalized benchmark.

Ignoring the recorded 25–27 ms audio lead also changes the nominal anchor at
every picture. The new path uses actual rational picture times minus the audio
origin, and retains the residual after selecting a 50-Hz anchor. It does not
assume native picture cadence is 25 Hz or substitute video zero for sample zero.

| Recorded support | Count |
| --- | ---: |
| Sources / common picture coordinates | 48 / 3,642 |
| Picture/sample coordinates within their WAV | 3,642 |
| Clock positions with more than one possible anchor | 767 |
| Picture chunks requiring explicit out-of-prefix zero slots | 209 |
| Pictures with nominal convolution windows inside raw source extent | 3,364 |
| Sources retaining their planned 20-native-frame edit | 47 |
| Edit-frame centers with full retained tick prefix | 940 / 940 |
| Edit-frame centers with nominal convolution windows inside source | 940 / 940 |
| Sources qualified for actual generator conditioning | **0** |
| New neural calls | **0** |

Ambiguity follows the existing timestamp-quantization allowance. Boundary zero
slots and the omitted partial final speech tick remain explicit; the helper
does not clamp them to repeated first/last features. All sources and roles stay
in the plan, including the shortest source without a supported 20-frame edit.
The 18 fitting, 10 calibration, 10 selection and 10 confirmation roles are
unchanged. No reserved detector responses were consumed.

## Token time does not establish local synthesis evidence

The installed, hash-bound source has a centered STFT and two convolution stages,
giving a **65-ms nominal convolution hull** per speech tick and a **245-ms hull**
for ten neighboring ticks. These are conservative virtual geometric bounds
before boundary operations, not measured local influence of the final features.

The same source clips spectral values against a whole-input maximum and invokes
encoder layers without a padding mask. Indexed states can therefore depend on
the whole padded input. Neither an in-range nominal hull nor the displayed
token time proves that a particular short audio interval is synthetic. Actual
feature output, padding sensitivity, generator response and achieved phonetic
synchronization still need neural qualification.

## Verification and artifacts

All **64 local tests** passed: 30 new routing/array-contract tests and 34 retained
waveform/sentence-role checks. Authored arrays verify native fractional clocks,
the retained 25-Hz equation, layer order, audio origin, uncertainty, zero slots,
invalid inputs and preservation of unavailable source positions. They are not
pretrained features. Ruff and whitespace checks passed.

Preparation rehashed all 48 mono WAVs and checked exact headers, original sample
coordinates, source/role identities and native edit positions. It exited zero
in **0.384 seconds**, with maximum observed self RSS **54,706,176 bytes** and
minimum observed available RAM **5,699,600,384 bytes**. Between-source resource
observations are not peak measurements. Existing data-only admission/reserve
limits and the separate 6-GiB neural admission stayed unchanged.

A separate standard-library readout recomputed **3,642 frame mappings and
36,420 ordered tick slots** from the saved sample fractions. It enumerated the
two convolution neighborhoods independently and reproduced all reported counts.
This is recorded-data arithmetic verification, not an outside human review,
new encoder execution or achieved-sync measurement.

- [Aggregate result](evaluation-panels/av-conditioning-clock-result-v109.json)
- [Recorded-data arithmetic review](evaluation-panels/av-conditioning-clock-review-v109.json)
- [Actual preparation profile](evaluation-panels/av-conditioning-clock-profile-v109.json)
- [Pre-run freeze](evaluation-panels/av-conditioning-clock-freeze-v109.json),
  published in `88d6d08` before preparation
- Private plan SHA-256:
  `559dbd7f391017862901a2bc1b88aa70a1d56cba1f7e1d4af467cb9daa42887b`

Source identities, individual WAV/frame/token commitments, raw media and all
model weights remain private. The next stage must qualify actual five-state
outputs and routing before generation and the new articulation comparison.
Serving confidence, rollback, customer reports and payments are unchanged.
No checkpoint was loaded or downloaded; no outreach, GitHub Actions, real
charge or GCP operation occurred.
