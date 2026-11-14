# Safeguards for speech flags after background reduction

The [matched traffic diagnosis](audio-background-rate-diagnostic-2026-11-13.md)
reproduced five genuine false flags on native 48 kHz and equivalent 16 kHz
files. This study tests whether corroborating each supplemental flag with an
additional cheap view can remove those kinds of errors while preserving the
synthetic flags that motivated the optional review. These probes use the same
classifier and the same recorded samples. The original voice assessment never
changes. The experiments contain no estimate of independent probability.

## Fixed comparison

Compare three routes through the same inputs:

- Current review. The normal voice result must be inconclusive or unassessed.
  Process the submitted waveform with the existing speech enhancement. Require
  speech activity on the result and apply the saved secondary operating point.
- Source band. For a current supplemental flag, require an additional response
  from the exact original waveform after the existing 100–3,800 Hz voice
  stability filter. This preserves the original sample clock and can show
  whether the flag only materializes after enhancement.
- Polarity-inverted enhancement. For a current supplemental flag, require its
  response to persist after multiplying the exact processed waveform by −1.
  The acoustic content is unchanged. This checks the stability of the
  classifier response with one more inference and one enhancement pass total.

Freeze both probe points before scoring the new cohort. Use the second-lowest
new response among 19 previously recovered noisy synthetic clips so each
point retains 18 of 19 on this limited historical sample. The two earlier
enhancement panels included two genuine false flags. Require at least six
recoverable noisy synthetic flags in the fresh panel to judge retention, at
least 85% of those overall and 75% in each tested scene. Retain every clean
synthetic flag. A candidate must remove at least two additional genuine flags
across at least two of the three background recordings. This uses the same
locally pinned voice and enhancement assets and adds no weights. Controlled
noise testing and time-frequency consistency before and after audio processing
motivate the comparison. The TFCL paper learns consistency across paired
clean and distorted speech during training. This experiment probes responses
at inference time using the incumbent and no third-party research code.
The TFCL repository labels its code MIT, while the distribution containing
its weights lists `other` without an explicit commercial grant. The experiment
uses the project’s existing verified model assets. [Noise-aware evaluation research](https://arxiv.org/abs/2512.13744) · [Audio processing consistency research](https://arxiv.org/abs/2607.17761) · [TFCL code](https://github.com/JunXue-tech/TFCL) · [TFCL checkpoint distribution](https://huggingface.co/datasets/JunXueTech/TFCL) · [Earlier assets and terms](audio-nemotron-noise-transfer-2026-11-12.md)

## Confirmation inputs

The confirmation used 27 genuine and 21 synthetic source files. Eighteen of
each class came from a bounded 96 MiB prefix of the public RADAR evaluation
archive. The additional nine genuine clips and three synthetic clips came
from a 320 MiB prefix of LlamaPartialSpoof with one file per distinct speaker
ID. Two of those synthetic sources use XTTSv2 and one uses CosyVoice. RADAR
does not identify the voices, generators, or original transformations in the
label table and the broader overlap between corpora and model training is
unknown. The selection excludes already used source IDs and available speaker
IDs from the relevant prior public panels, including both enhancement panels.
No VCTK recordings enter the confirmation. Both dataset publishers declare
CC BY 4.0. [RADAR publisher](https://zenodo.org/records/22110815) · [LlamaPartialSpoof publisher](https://zenodo.org/records/14214149)

Take the central three seconds of each recording. Save one clean clip and
three clips mixed with rain, an arriving train, and Port Harcourt traffic at
−5 dB relative to the added background by RMS. Use the same peak-safe
foreground gain across a source's four clips. Twelve isolated controls sample
the same background excerpts. The files are 16 kHz PCM16 WAV uploads with
continuous clocks. The decoder reconstructed all 204 committed files and
analysis inputs in a second preparation pass. The three background sources
are distinct and absent from the probe point selection, with earlier use in
other project evaluations. [Rain and train recordings and terms](audio-focused-window-transfer-2026-11-12.md) · [Traffic recording and terms](audio-radar-transfer-2026-11-12.md)

## Results

| Condition | Primary AI, synthetic / genuine | Additional AI-like flags on synthetic, current / source band / polarity | Additional false flags on genuine, current / source band / polarity |
| --- | ---: | ---: | ---: |
| Clean | 14/21 · 1/27 | 3 / 1 / 3 | 1 / 0 / 1 |
| Rain | 1/21 · 2/27 | 9 / 6 / 8 | 2 / 2 / 2 |
| Arriving train | 2/21 · 2/27 | 8 / 7 / 8 | 4 / 3 / 4 |
| Street traffic | 7/21 · 5/27 | 7 / 7 / 7 | 5 / 5 / 5 |
| All noisy combinations | 10/63 · 9/81 | 24 / 20 / 23 | 11 / 10 / 11 |
| Isolated backgrounds | 0/12 | 0 | 0 |

The current optional route flagged 24 noisy synthetic clips from 14 source
files. Source-band corroboration retained 20 clips from 13 sources and
withheld one genuine train flag. It also withheld two of the three clean
synthetic flags. The polarity probe retained 23 noisy flags across the same
14 synthetic files and all 11 genuine false flags. Its front end uses the
magnitude of spectral coefficients, which explains why inverting the waveform
gave no new discrimination. The slightly stricter operating point alone
can withhold near-boundary flags. In both cases the primary results stayed
the same. Neither probe met the frozen advancement criteria, and both stay
outside the app. The current supplementary findings stay out of overall
confidence. [Frozen study and measurements](evaluation-panels/audio-background-safeguard-2026-11-13.json)

Under noisy conditions, the primary was inconclusive on 21 synthetic and 27
genuine deliveries and unassessed on 32 synthetic and 41 genuine deliveries.
Enhancement admitted 50 of 53 eligible synthetic inputs and 65 of 68 eligible
genuine inputs. It admitted five isolated sound excerpts and none crossed the
flag point. The outcome pertains to 48 source files and their three matched
noisy versions. It cannot estimate how either probe behaves on naturally noisy
speech or unseen language and generator families. A separate recount checked
all 204 WAV hashes and clocks, recalculated the two points from the historical
responses, and reproduced the saved condition counts.

## Replay

Prepare the previously pinned enhancement panels as `old`, their existing
assets, the official RADAR metadata and traffic sample as `radar_dir`,
the published Llama label table as `part-a.labels.txt` under `llama_dir`, and
the pinned rain and train recordings under `sounds`. Download these bounded
prefixes under `inputs`. Keep recordings and model files local.

```sh
curl -fLsS --range 0-100663295 --max-filesize 100663296 \
  'https://zenodo.org/records/22110815/files/RADAR2026-eval.tar.gz?download=1' \
  -o "$inputs/radar-96.gz"
curl -fLsS --range 0-335544319 --max-filesize 335544320 \
  'https://zenodo.org/records/14214149/files/R01TTS.0.a.tgz?download=1' \
  -o "$inputs/llama-320.tar"
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze verify-inputs; do
  backend/.venv/bin/python -B -m scripts.audio_background_confirmation "$phase" "$run" \
    --source "$inputs" --radar "$radar_dir" --llama "$llama_dir" --sounds "$sounds" || exit 1
done
backend/.venv/bin/python -B -m scripts.audio_background_confirmation fit "$run" --old "$old"
backend/.venv/bin/python -B -m scripts.audio_background_confirmation score "$run"
python3 -B -S -m scripts.audio_background_confirmation review \
  docs/evaluation-panels/audio-background-safeguard-2026-11-13.json
```
