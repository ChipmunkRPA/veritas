# Exact digital silence in audio and video soundtracks

Chipmunk Verify now shows short runs of exact digital silence in the decoded
mono track. The measurement needs at least 100 ms of zero samples and 100 ms
of measurable sound on either side, all on one continuous source clock. It
shows up to three qualifying intervals, with sample positions, source times,
the marked waveform, and listening clips. The PDF and JSON record the same
observations. Muting, a noise gate, or ordinary editing can cause this. The
check carries zero weight in AI confidence.

The recording may contain audible background sound while its foreground voice
pauses. Applying a lossy codec, resampling, mixing, or fading can also leave
nonzero samples in a muted interval. This comparison uses the exact decoded
input and does not infer what caused the silence or recover missing speech.
[Timing and muted audio discussion in ITU-T P.862.3](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-P.862.3-200711-W%21%21PDF-E&lang=e&type=items)

## Matched local test

The study used the 36 original SINE excerpts from the earlier evaluation and
eight separate recordings of concurrent speech from AMI. It constructed 296
continuous six-second mono PCM deliveries with known changes. The two mutes
were 120 and 240 ms and started away from the half-second energy grid.
The same corpus supplied a dry and a competing-speech view. A separate
control zeroed the foreground while preserving the background; another
attenuated the mixture without muting it. The ordinary files and source
rooms received no edits. The old comparator uses the existing half-second
low-energy rule. It serves a broader purpose of highlighting very quiet
portions, so its ordinary observations are not mislabeled as false mutes.
[Source commitments and terms](audio-sine-resynthesis-transfer-2026-11-14.md#sources-and-controls)

| Condition | Exact intervals from half-second bins | Exact intervals from sample check |
| --- | ---: | ---: |
| Dry, 120 ms whole-track mute | 0/36 | 31/36 |
| Dry, 240 ms whole-track mute | 0/36 | 26/36 |
| Competing speech, 120 ms whole-track mute | 0/36 | 36/36 |
| Competing speech, 240 ms whole-track mute | 0/36 | 36/36 |

The new comparison reported no intervals on the 152 controls. Some inserted
dry mutes had quiet adjacent speech or pauses and did not meet the measured
flank requirement. Both noisy cases use the same sources. These counts
measure an authored recording operation on decoded lossless inputs. They
give no accuracy estimate for generated voices or original recordings in
the field.
[Scalar receipt](evaluation-panels/audio-digital-silence-2026-11-14.json)

## Customer flow and replay

New audio or video analyses with a decoded audio timeline show the check
under **Background & recording context**. It reports how much decoded
audio was eligible for comparison and how many intervals were shown.
Each event plays the exact selected interval and a clip with both
measured 100 ms flanks. Replaying a signed export from the original
recalculates the zero runs and flank levels and verifies the playback
against the original samples. The normal voice assessment retains its
own inputs and rules.

Two isolated, simulated prepaid audio analyses used a noisy matched pair.
The whole-track mute reproduced samples **[27200, 31040)**,
**1.700–1.940 s**. The clip with only its foreground voice muted produced
no exact silence. The API, PDF, JSON, and signed original replay agreed on
both records. The test used the USD 9.99 payment gate.

The scalar reader uses standard Python. It can verify the counts without
source recordings. The native runner takes a panel directory and its
published SINE receipt. Both local measurements reproduced the same
296 case records and counts.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_digital_silence_transfer \
  docs/evaluation-panels/audio-digital-silence-2026-11-14.json
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_digital_silence_transfer \
  "$new_receipt" --sine-parent docs/evaluation-panels/audio-sine-resynthesis-transfer-2026-11-14.json \
  --panel "$sine_panel"
```

Prepare the native panel through the [SINE replay instructions](audio-sine-resynthesis-transfer-2026-11-14.md#replay).
