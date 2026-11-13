# Short upper-frequency drops in native audio

Chipmunk Verify can now show a supported temporary loss and return of
upper-frequency sound on a single six-second audio upload. The optional
check reports a physical observation in the mixed recording. It carries
zero weight in the AI estimate. It can also respond to ordinary sound
changes and has limited coverage on some real recordings.

## Evaluation

The initial frozen method searched for at least a 12 dB drop in two
upper bands and demanded stable relative band energy before and after
the drop. It placed 7 of 25 known 3.4 kHz low-pass interventions and 5
of 25 at 3.8 kHz, with no selected interval on 120 controls. The earlier
sound-transition check placed neither edge pair on any of the 50
interventions. The new result missed the 80% research gate. Inspection
showed the original rule often lacked two comparable, stable flanks.
[Initial source commitments and outcomes](evaluation-panels/audio-bandwidth-return-development-2026-11-13.json)
· [Existing transition method](../backend/app/evidence/audio_continuity.py)

The revised rule uses 80 ms cells, 400 ms flanks and supported ranges
of 560 to 1,200 ms. It measures the 4.1–5.5 kHz and 5.5–7.4 kHz bands
relative to 0.5–1.9 kHz. Both flanks must carry measurable upper-band
energy with a similar balance. The middle band from 1.9 to 2.7 kHz
limits changes that simply reshape much of the signal. The minimum
drop must strictly exceed 16 dB in both upper bands against both
flanks. On the consumed development panel, this version located 24 of
25 per cutoff with no flags on the inspected unfiltered, cut, gain,
or equalization cases. Those reused sources were used to select the
rules. The comparison adapts the forensic distinction between frequency
measurement and authenticity inference. It uses no external detector,
separation weights, or AI model. [Recent research on codec and bandwidth effects](https://arxiv.org/html/2611.13258v1)

The separate confirmation froze 32 six-second excerpts from four
different AMI meetings, with eight time-stratified, content-independent
selections per meeting. These meetings were excluded from the original
bandwidth panel. The captures had earlier project exposure for other
checks. Each source contributes matched untouched, global low-pass,
short low-pass at each cutoff, short gain reduction, and local
same-meeting natural-cut variants. The 25 ms crossfades and intervention
clock are the same across matched local variants. Useful means covering
at least half the intervention with a time IoU of at least 0.30.
[AMI source and terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
· [Confirmation measurements](evaluation-panels/audio-bandwidth-return-confirmation-2026-11-13.json)

| Confirmation condition | Planned | Ranges shown | Useful | Unassessed |
| --- | ---: | ---: | ---: | ---: |
| Short low-pass at 3.4 kHz | 32 | 17 | 17 | 12 |
| Short low-pass at 3.8 kHz | 32 | 17 | 17 | 12 |
| Untouched | 32 | 0 | — | 10 |
| Whole-file low-pass | 32 | 0 | — | 32 |
| Short gain change | 32 | 0 | — | 11 |
| Same-meeting ordinary cut | 32 | 1 | — | 12 |

The prior general transition check put two edges near none of the 64
known interventions. The revised method placed 34, including 8 of 8
in two meetings and 1 of 8 in a third at each cutoff. The fourth
meeting yielded no useful locations. The frozen 80% all-delivery
and four-meeting criteria failed. Among comparable filtered cases the
check placed 17 of 20 per cutoff, and the matched negative panel produced
one observation on an ordinary cut. The observation describes energy
change even if a filter was never applied. The measured coverage and
the 34-location gain justify a limited opt-in for listening evidence.
They do not support an AI, splice, or broad audio-detection claim.

## Local app and exact replay

Set `VERITAS_AUDIO_BANDWIDTH_RETURN_REVIEW=true` for new standalone,
native six-second mono 16 kHz PCM16 WAV analyses. The strongest qualifying
interval shows source sample positions, waveform, spectrogram, selected
and surrounding WAV excerpts, and measured drops in each band. The
PDF shows the same observation and plots. The JSON stores the exact
span and supporting measurements. A file without two measurable
surrounding frequency profiles shows unassessed. Files with measurable
profiles that fail to support a drop have their own outcome. Other
formats and durations are outside the tested serving scope. The option
is off by default and saved evidence continues to use its recorded
method version.

An isolated paid worker rechecked a native confirmation upload and its
untouched match. The measured **1.200–2.080 s** range, samples
**[19200, 33280)**, showed a **31.379338 dB** minimum relative upper-band
drop. The untouched upload produced no observation. The same edited
upload with the option off returned the previous acoustic schema and
the same overall decision. The API, PDF and JSON matched, including the
exact WAV excerpt and the waveform and spectrogram pixels. Signed
verification from the original audio reran the band comparison and
regenerated the plots. Checkout was simulated at USD 9.99. No analysis
was queued until simulated payment was confirmed, and a repeated
confirmation created no duplicate job.
[Local application receipt](evaluation-panels/audio-bandwidth-return-app-2026-11-13.json)

## Transfer to eight and twelve seconds

The same measurement now runs on continuous native mono PCM16 WAV or
FLAC at 16 kHz, for exactly six, eight or twelve seconds. The option
remains local and off by default. A six-second WAV keeps its original
evidence version. New format and duration records bind the source codec,
bit depth and exact clock to the saved acoustic evidence. The replay
verifier checks the original file and recomputes the measurements and
plots. Other durations, stereo sources and higher-depth FLAC stay outside
this optional check.

Two different AMI meetings supplied 32 source positions, selected by a
fixed digest rule before scoring. The earlier bandwidth study used
different meetings. These meetings had prior project exposure in other
work. For each source the six- and
eight-second deliveries are exact crops of the twelve-second delivery.
Each receives the same intervention and its unchanged, gain, conventional
cut and whole-file low-pass controls. The check uses the existing 16 dB
point throughout. The denominator is the same at every duration.

| Source duration | Useful at 3.4 kHz | Useful at 3.8 kHz | Ordinary observations | Edited files unassessed |
| --- | ---: | ---: | ---: | ---: |
| 6 seconds | 18/32 | 17/32 | 3/128 | 14/64 |
| 8 seconds | 18/32 | 18/32 | 3/128 | 8/64 |
| 12 seconds | 18/32 | 18/32 | 3/128 | 0/64 |

At each duration, one unchanged, one gain and one ordinary cut yielded a
physical frequency-drop observation. The 32 whole-file low-pass controls
were unassessed at each duration. Useful locations at either new duration
were 2 of 16 per cutoff in one meeting and 16 of 16 in the other.
This is evidence about finding a temporary frequency change in the
recorded mixture. It carries zero AI-confidence weight.
[Source and outcome receipt](evaluation-panels/audio-bandwidth-duration-transfer-2026-11-13.json)

An isolated Full analysis on a held twelve-second FLAC measured a drop of
**31.360893 dB** at **5.360–6.240 s**, or samples **[85760, 99840)**.
The untouched match showed no supported drop. The report, evidence JSON
and original WAV excerpt reproduced the same range. The signed export
recomputed the source clock, audio, waveform and spectrogram from the FLAC.
Analysis waited for simulated USD 9.99 payment, and disabling the optional
check preserved the paid result's overall decision.

The scalar reader requires only standard-library Python. To reproduce all
576 rows from the same native files, use the two publisher WAVs below and
a fresh output directory. The script verifies the complete parent hashes.
The ES2010d WAV has two channels. The replay reads only its first channel
to construct the mono deliveries.
[ES2010d source](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2010d/audio/ES2010d.Array1-01.wav)
· [IS1007a source](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1007a/audio/IS1007a.Array1-01.wav)
· [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze score; do
  backend/.venv/bin/python -B -m scripts.review_audio_bandwidth_duration "$phase" \
    "$run" "$es2010d" "$is1007a" || exit 1
done
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_bandwidth_duration review \
  docs/evaluation-panels/audio-bandwidth-duration-transfer-2026-11-13.json
cmp "$run/receipt.json" docs/evaluation-panels/audio-bandwidth-duration-transfer-2026-11-13.json
```

## Original six-second replay

The confirmation reader can run without scientific dependencies. To
reproduce the PCM measurements from native WAVs, use the four frozen
AMI channel 01 files. The initial test reuses the source caches linked
from the [formant comparison](audio-formant-local-transfer-2026-11-13.md#replay).

```sh
python3 -B -S backend/scripts/audio_bandwidth_confirmation.py review \
  docs/evaluation-panels/audio-bandwidth-return-confirmation-2026-11-13.json

export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.audio_bandwidth_confirmation "$phase" \
    "$run" "$es2007c" "$is1004c" "$es2012b" "$is1001b" || exit 1
done
```

## Replay from public source ranges

A clean checkout can acquire the exact source intervals directly from the
AMI project’s first-party mirror. The command fetches four pinned WAV headers
and 32 contiguous twelve-second windows, then verifies the two six-second PCM
digests in each window. It recreates each source clock position from the parent
PCM extent and the frozen selection rule. The download uses 12,288,176 bytes
in 36 cache files. Keep this cache outside the repository. The parent WAV
hashes still identify the archived inputs. This bounded path verifies the
selected PCM rather than reading the entire parent WAVs.
[AMI downloads and attribution](https://groups.inf.ed.ac.uk/ami/download/)
· [AMI CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

From the repository root, run `acquire` with an empty or previously verified
cache. `verify-sources` checks the same source intervals offline using the
Python standard library. `replay` uses the backend numerical dependencies to
regenerate the 192 submitted waveforms and both checks’ measurements, then
compares every row directly with the published receipt.

```sh
review_cache=$(mktemp -d)
python3 -B -IS backend/scripts/replay_audio_bandwidth_public.py acquire "$review_cache"
python3 -B -IS backend/scripts/replay_audio_bandwidth_public.py verify-sources "$review_cache"
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  backend/.venv/bin/python -B -m scripts.replay_audio_bandwidth_public replay "$review_cache"
```

Local replay reproduced all 192 rows from the downloaded ranges. An
independent cache cut from the four SHA-256-verified full WAVs produced
identical bytes and replay output. The source-only check matched on Python
3.11 and 3.12 and rejected a one-bit change in a neighboring control
excerpt. The replay reproduces the prior comparison. It adds no new
capture, detector fit, or accuracy estimate.
