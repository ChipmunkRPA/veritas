# Matched ordinary and synthetic speech edits

Compare the pinned short-speech route and one-way repetition abstention on the
development role from [v172](audio-short-edit-cases-v172.md). All 23 donor
comparisons contribute nine matched pairs each, for 414 three-second cases.
Each pair has a common genuine host, gain, join length, noise parent and
quantized additive noise. The ordinary edit uses a different genuine voice.
The synthetic edit uses a stock, cloned or expressive AI voice. Compare hard
joins and 20 ms and 50 ms joins at clean, nominal 0 dB and nominal +5 dB
conditions. Nominal SNR uses the genuine host as the reference. Retain the
measured SNR of each arm privately.

Verify the published v172 commitments against the private development plan,
case journal and successful guarded resource profile. Open only constructed
matched WAVs in the exact development directory. Check the file hash and the
production reader's 16 kHz mono, zero-origin, three-second sample extent
against the prepared float32 and signed16 PCM commitments. The scorer receives
the verified waveform alone. The full-profile speech gate and pinned
production calibration govern the baseline. Perform one raw classifier
prediction for each accepted waveform. The original-side stability check
already required by serving runs only when serving requests it. Track those
forwards separately. The optional symmetric band experiment from v173 is
omitted. Both policies reuse the baseline prediction and its calibration.

The comparator changes an AI tendency to inconclusive when either exact
repetition or approximate waveform similarity reports a measured repeated
span. Evaluate cues on the complete waveform without edit timing or origin
labels. Cue ambiguity and unavailable cues leave the baseline direction in
place. Repetition supplies inspection evidence and can occur in speech of
either origin. It provides no probability, AI vote or verified edit boundary.

Report planned, scored and decided coverage with all four directions, plus
ordinary false AI calls, synthetic misses and AI-to-inconclusive changes.
A synthetic miss includes every planned synthetic case without an AI
tendency. Ordinary and synthetic counts are reported separately for all
combinations of donor family, noise and join. Keep the paired AI contrast
and both-decided coverage, the equal-host means and a descriptive sensitivity
check omitting one noise parent at a time. The 23 comparisons are linked by
reused source clips and share four noise recordings. The strata and
sensitivity summaries describe one development stress panel. Fit and
independently verify any proposed change using separate source ancestry and
noise parents.

Run only after the completed v172 development preparation. These commands
freeze a new local study, run its offline v108 guarded worker, and verify the
private observations against the emitted aggregate.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_matched_edit_diagnostic_v174 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_matched_edit_diagnostic_v174 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_matched_edit_diagnostic_v174 readout
```

The study performs no fitting, does not access holdout files, and leaves
customer-facing decisions unchanged.
