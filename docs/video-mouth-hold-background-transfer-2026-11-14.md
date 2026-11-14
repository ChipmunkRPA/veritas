# Soundtrack evidence during a mouth hold under loud background sound

The app now reports the measured level during a held mouth region. A
source interval can be digitally silent, contain sustained voice-like
periodicity, or contain sound without sustained periodicity. When sound
is present without a reliable periodic pattern, speech presence remains
undetermined. This measurement carries zero weight in AI confidence.
[How the picture and soundtrack are bound](video-mouth-hold-sound-2026-11-13.md)

## Frozen noise comparison

Use one original and one published alternate from each of 26 SINE
readers, selecting the first eligible pair per reader from the earlier
licensed intake. Eight readers supply development and the other 18
supply held data. The alternate can be a resynthesis, short splice, or
infill. This trial measures acoustic activity, without using the
authorship labels. Each side supplies a six-second recording with four
fixed 400 ms windows, plus four −5 dB foreground-to-background mixtures
and a matched background-only version per mixture. This gives 468
deliveries and 1,872 measured windows including 1,296 held windows.
[SINE and AMI source qualification](audio-sine-resynthesis-transfer-2026-11-14.md)
· [Rain, train, and traffic sources](audio-background-safeguard-2026-11-13.md#confirmation-inputs)

The environmental recordings are reused across both folds, using
nonoverlapping excerpts for each fold. The competing-voice recordings
come from different rooms in the two folds. Windows from the same clip
and controls from the same excerpt are correlated. The mixtures are
authored on retained recordings. These counts give no direct estimate
of accuracy on natural video, new generators, or visible-speaker
attribution.

The baseline uses raw 40 ms autocorrelation. One candidate balances
four frequency bands before measuring periodicity. The other uses
nearby sound on both sides of the window to estimate and subtract
background spectral power, then measures periodicity. Research on
residual harmonics motivates reducing the effect of dominant spectra.
This trial uses its own filters and no third-party research code or
weights. [Residual-harmonic voicing research](https://arxiv.org/abs/2001.00459)

The dev grid required at least one recovered clean-positive interval,
at most one newly flagged distinct background window and one extra
mixture without corresponding clean support, and no purported recovery
where the background alone also flagged. The band-balanced view never
qualified. The other view recovered one development interval at its
selected point, then added zero recoveries in the held fold. Only 33
of the 144 held clean windows met the baseline's sustained-periodicity
rule. Across their 132 noisy counterparts, the baseline retained two
periodicity measurements. Both were competing-voice mixtures. Neither
new view enters the app.

## Speech-activity comparison

The pinned speech-activity asset already used elsewhere in the app was
tested separately at its existing inference settings. A window counts
when at least 200 ms overlaps a reported speech span. The model reads
the full six seconds, then the replay measures overlap on the same
fixed source windows as the periodicity comparisons.
[Pinned MIT license](https://github.com/snakers4/silero-vad/blob/066409f3d9db45a30d21dfcbf9147826ffdd7052/LICENSE)

| Held sound | Speech activity in speech deliveries | Speech activity with background alone |
| --- | ---: | ---: |
| Clean | 134/144 | — |
| Rain | 118/144 | 0/144 |
| Train | 94/144 | 0/144 |
| Traffic | 111/144 | 0/144 |
| Competing voices | 142/144 | 134/144 |

Voice activity offers better coverage of speech mixed with these
environmental backgrounds than the existing periodicity response. The
competing-voice controls already contain speech. They show why this
response cannot tell who was speaking during a visible mouth hold.
It remains an offline comparison for this video check.

## Local app and independent replay

Two simulated paid runs reused a qualified two-second picture hold at
**0.520–0.840 s**. The first used a digitally silent subsecond audio
stream that still covered the held interval. The second used rain,
which measured **−26.291 dBFS RMS** and showed periodicity in **0 of
14** blocks. Both had the same 250 ms audio origin and exact before,
during and after frames **12, 17 and 21**. The app, PDF, JSON and signed
package verified the same waveform, audio samples and pictures. No
analysis was queued before simulated payment.

The two public receipts contain hashes, scalar statistics, speech-sample
counts and paired outcomes. They include no recordings or feature
vectors. The portable commands recalculate decisions with standard-library
Python. The native commands additionally rebuild each exact input from
the pinned SINE/AMI cache and the three publisher sound files.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
noise=docs/evaluation-panels/video-mouth-hold-noise-transfer-2026-11-14.json
activity=docs/evaluation-panels/video-mouth-hold-activity-transfer-2026-11-14.json
python3 -B -S -m scripts.review_video_mouth_hold_noise_transfer "$noise"
python3 -B -S -m scripts.video_mouth_hold_activity_transfer "$noise" "$activity"
for phase in freeze measure; do
  backend/.venv/bin/python -B -m scripts.video_mouth_hold_noise_transfer "$phase" "$sine_panel" "$sounds" "$new_run" || exit 1
done
backend/.venv/bin/python -B -m scripts.video_mouth_hold_activity_transfer "$new_run/result.json" "$new_run/activity.json" \
  --source "$sine_panel" --sounds "$sounds" --checkpoint models/weights/audio-upgrade/silero_vad.jit
```
