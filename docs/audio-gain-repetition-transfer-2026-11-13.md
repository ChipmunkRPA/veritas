# Exact gain-adjusted audio repeats

This check compares two nonoverlapping intervals within one native stereo channel. It reports an exact signed 16-bit sample relationship under one supported constant gain and nearest-integer quantization. An amplitude difference or reversed polarity can conceal the match from an exact-equality check. The relationship describes repeated samples. A repeated sound effect or routine production edit can satisfy it, and the math does not establish which occurrence came first in the editing process.

## Fixed comparison

The development set used all 16 retained room excerpts from the earlier stereo repetition study. The transfer set used 24 three-second excerpts from two AMI Corpus meeting sessions, ES2004a and IS1009a, at 75, 105, …, 405 seconds. These time ranges are disjoint from the 60, 90, …, 390 second native intake windows. The two meeting identities were previously used. All scheduled transfer excerpts were included without filtering for silence or speech activity. We verified the four original native 16 kHz microphone WAV files against the first-party acquisition hashes and preserved the sample clock of both microphone channels. The AMI Consortium releases the corpus under CC BY 4.0. [AMI Corpus and license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) · [Publisher source and consent intake](audio-ami-native-stereo-v291-method.md) · [Development room excerpts](stereo-repetition-2026-11-13.md#local-evidence)

From each excerpt we authored four gain-adjusted repeats in one channel, one exact copy and six comparison deliveries. The four gain conditions were 3/4 attenuation, −1 polarity reversal, 3/2 amplification, and 1/2 attenuation of the earlier occurrence. The selected channel alternated. The unchanged other channel retained the competing sound captured by the second microphone. The controls were unmodified input, all-input gain, a gain outside the fixed table, a competing microphone mixed into the target interval, a repeat interrupted by clipped samples, and a 74.9 ms repeat below the 150 ms scope. We used separate authored silence, periodic tone and an ordinary alert repeated at a lower level as additional stresses. No speech synthesis, inference weights or calibration were used.

The frozen advancement gate required at least 80% of the 96 gain-adjusted cases with correct channel, gain and both half-open intervals within eight samples, at least one correct case per meeting, and zero extra or wrongly localized relationships across the transfer set. We independently checked each emitted relationship against the exact source samples. The incumbents used the exact-equality check on each native channel and on the normal FFmpeg downmix. A separate local test decoded all 11 variants from six preselected sources through the production decoder. All 66 comparisons reproduced both native stereo and mono bytes. A separate numeric smoke test reproduced an exact off-grid pair for each of the eleven ratios in the support table.

## Transfer results

| Condition | Cases | Correct gain-related ranges | Correct incumbent per-channel ranges | Correct incumbent mono ranges |
| --- | ---: | ---: | ---: | ---: |
| Later attenuated to 3/4 | 24 | 24 | 0 | 0 |
| Polarity reversed | 24 | 24 | 0 | 0 |
| Later amplified to 3/2 | 24 | 24 | 0 | 0 |
| Earlier attenuated to 1/2 | 24 | 24 | 0 | 0 |
| Exact-equality copy | 24 | 0 | 24 | 0 |
| Six controls | 144 | 0 flags | 0 flags | 0 flags |

All 528 channel searches on the 264 transfer deliveries completed with measured status. The gain check recovered 94 of the 96 authored ranges to the sample. On two deliveries a neighboring sample also satisfied the relation and extended both stops by one sample, 62.5 microseconds. It recovered all 64 gain-adjusted cases and added no other match on the 176 development deliveries. The existing per-channel check still found all 16 exact copies on development and all 24 on transfer. The complete first run assessed 443 deliveries in 45.052 seconds. The gain search took 26.47 ms median and 47.363 ms maximum per delivery for both channels. The incumbent's two exact channel searches plus its exact mono search took 21.039 ms median and 42.355 ms maximum. These timings exclude decode and report rendering.

The silent file returned unavailable on both channels. The periodic tone exceeded the candidate-pair budget and returned ambiguous on both. The ordinary alert satisfied the exact gain relation. Its roughly 191 ms audible excerpts were surrounded by equal digital silence, which extended the detector's mathematically matching intervals to about 1.45 seconds each. Matching ranges can therefore overstate the audible event and provide no evidence of edit intent or AI authorship. The test covers exact fixed gains in short native 16 kHz lossless material. Lossy codecs, arbitrary or slowly varying gains, and naturally occurring repeat rates beyond the two meetings remain unmeasured.

## Local replay

The replay re-read the original samples and reproduced all 443 private records, including input hashes, interval boundaries, statuses and exact rational witnesses. The public receipt retains aggregate counts and source commitments. Recordings and per-file records stay local. The runner takes the retained development fixture directory, the original AMI archive root, and a fresh scratch directory. It also expects the earlier private room roster and native PCM at their recorded paths. [Frozen aggregate and replay receipt](evaluation-panels/audio-gain-repetition-transfer-2026-11-13.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.audio_gain_repetition_transfer "$phase" \
    --development "$development" --ami "$ami" --ffmpeg "$ffmpeg" \
    --output "$scratch" --receipt "$receipt" || exit 1
done
```
