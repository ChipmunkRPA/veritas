# Exact text-matched speech under competing voices

Two recent room trials mixed genuine and generated speech from different
utterances. The publisher's ODSS archive provides natural, VITS and
FastPitch–HiFiGAN recordings of the same prompts. This follow-up tests whether
the voice results change when content is held constant and the room and
competing speakers change between development and assessment. The score
points belong to this comparison. [Earlier matched-room trial](audio-wpe-fresh-room-transfer-2026-11-13.md)
· [ODSS release and paper](https://zenodo.org/records/8370669)

Recent work tests the speech and background as separate sources and reports
that detectors can respond to either. We used matched background-only inputs
to measure that failure independently of missed synthetic speech. The test
uses genuine competing speech throughout. [Component-wise detector study](https://arxiv.org/abs/2610.17617)
· [Mixed-audio benchmark](https://arxiv.org/abs/2605.23201)

## Matched sources

The earlier inventory verified 3,778 Hi-Fi TTS utterance triplets across eight
publisher reader groups using the ZIP directory. This run retrieved and opened
48 small WAV members from the publisher, representing two triplets per group.
Each member matched its local and central ZIP names, uncompressed length and
CRC. FFmpeg independently decoded all 48 complete PCM sequences and matched
their lengths and SHA-256 hashes. The utterances run 2.055–2.531 seconds.
The publisher's transcript manifest supplies a single unique text for each
selected member stem. No normalized prompt or exact PCM is shared across
groups. Exact prompt matching comes from the publisher's manifest. We did
not measure speech-recognition error or verify that speaker identity is
perceptually identical after synthesis. [Earlier inventory](evaluation-panels/audio-odss-inventory-v297.json)
· [ODSS CC BY-SA 4.0 license](https://zenodo.org/records/8370669)
· [Hi-Fi TTS source and CC BY 4.0 license](https://www.openslr.org/109/)

The development and assessment sets each contain four distinct publisher
reader groups. The development room uses the measured University of Surrey
responses. The assessment room uses the measured BBC R&D responses. The
corresponding competing speech is from different AMI parent recordings with
zero shared annotated participants. Those tapes had prior project use for
separate work. All three versions of a prompt use the same interfering
excerpt at the same level and direction. Each excerpt contains 1.04–3.00
seconds of annotated simultaneous words from at least two speakers within the
three-second inspection interval. The short natural and synthetic WAVs keep
their full sample ranges on the rendered clock. There is no local voice
generation, partial splice, or speed adjustment. [Surrey measurement and license](https://openresearch.surrey.ac.uk/esploro/outputs/dataset/IoSR-Listening-Room-Multichannel-BRIR-Dataset/99512517702346)
· [BBC response source and license qualification](audio-bbc-room-assets-v303-method.md)
· [AMI recording and manual annotation license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Comparison

The six-second stereo mixtures place the entire foreground utterance within
**1.5–4.5 seconds** and normalize the average foreground energy across each
natural/generated trio. The same background level is used for all three.
The three delivered conditions are room-only, 0 dB, and −5 dB measured over
the full six-second stereo mean. Within the held three-second scoring
window, SNR ranges from 2.08 to 8.44 dB in the 0 dB condition and from −2.92
to 3.44 dB in the −5 dB condition. Each noisy condition has a matched
background-only control. The 176 delivered views
comprise 88 development and 88 held. Both ears use the publisher's measured
impulse responses, with their relative delays intact. This controlled room
rendering remains separate from a native recording of the combined scene.

Keep the existing mono response and measure two alternative views separately.
Weighted prediction error targets late reverberation. Frontal mid-side
cancellation removes the share of the stereo mean predicted by the stereo
difference. It assumes the wanted speaker is centered, and fits its gain
from the submitted mixture alone. Speech admission and the commercial voice
classifier are unchanged. Each primary point is the highest admitted human
response in its own four-speaker noisy development set, frozen before held
scoring. The separate one-speaker point shows sensitivity to the small point
set. The receipt reader also reports a diagnostic union with mono. Count
prompts and speaker groups separately because the same foreground appears at
both noise levels. This study assesses a complete known speech interval.
Partial edit localization needs its own test. Transfer to the held role
changes acoustics, readers and background conversation together. The two
synthesis families remain the same across roles. [WPE method](https://www.rd.ntt/cs/team_project/media/signal/wpe/)
· [Spatial cue motivation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3246289/)
· [Voice asset and commercial terms](commercial-model-audit.md)

## Results

| Held view at its primary point | VITS detected | FastPitch–HiFiGAN detected | Genuine flags | Genuine reader groups flagged | Background-only flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mono | 7/16 | 10/16 | 2/16 | 1/4 | 0/16 |
| WPE | 11/16 | 11/16 | 9/16 | 4/4 | 0/16 |
| Frontal mid-side | 12/16 | 16/16 | 9/16 | 4/4 | 0/16 |

Frontal cancellation preserved all 17 mono synthetic detections and added
11, including both generator families. It also preserved both mono genuine
flags and added seven, reaching every held genuine reader group. At the
stronger mixture level it detected 13/16 synthetic views versus 6/16 for
mono, with 5/8 versus 1/8 genuine flags. WPE added six synthetic flags and
lost one of mono's, with seven additional genuine flags. A union with mono
recovers that lost WPE detection, reaching 23/32 with the same nine genuine
flags. The separately fitted one-reader points did not resolve the held
trade-off. All foreground scoring windows were admitted. The speech gate
declined one frontal background-only view. The other 47 held background-only
readings across three views made no AI call. Room-only mono detected 11/16
synthetic views with 0/8 genuine flags at its frozen noisy point.

Frontal cancellation improved source-window scale-invariant
signal-to-distortion ratio in all 48 held noisy target mixtures. Median
improvement was 1.58 dB at nominal 0 dB and 2.15 dB at nominal −5 dB. WPE
improved 43/48, with corresponding medians of 0.50 and 1.15 dB. This acoustic
gain did not yield better separation of synthetic and genuine responses on
the same prompts. The frontal and mono views each ranked synthetic speech
above its genuine counterpart in 13/16 FastPitch–HiFiGAN pairs. For VITS
they did so in 11/16 and 12/16, respectively. WPE ranked 10/16 and 12/16.

Both alternate routes remain experimental. The frontal arm's gain in
synthetic flags came with a large rise in genuine flags and nearly unchanged
matched ranking. The measured room, four held speakers, two synthesis
families and assumption of a centered target limit transfer. This run does
not change the overall AI confidence. The local fit used 216 model forwards
and the held run used 215. [Anonymous scalar receipt](evaluation-panels/audio-odss-matched-room-2026-11-13.json)

An additional transfer could use a larger panel of exact pairs with newer
generators and aligned partial speech. PhonemeDF includes paired text-to-speech
and voice-conversion renditions with phoneme alignments. Its publisher API
lists CC BY 4.0. Only the release metadata and documentation were inspected;
no recording from that corpus was opened or scored in this study. Its audio
comes in separate multi-gigabyte compressed archives, and source-specific
rights still need checking before use. [PhonemeDF paper](https://aclanthology.org/2026.lrec-1.469/)
· [Release terms and file inventory](https://zenodo.org/api/records/19522979)

## Reproduce

The source cache holds bounded byte ranges from the publisher ZIP and is
approximately 8 MB. The public receipt omits speaker IDs, prompts and media.
The offline replay reconstructs source commitments using the publisher cache,
qualified impulse-response files and AMI inputs. All 176 delivered waveform
hashes are distinct. An alternate FFT implementation reproduced every one of
128 noisy frontal views bit for bit. The maximum difference in WPE views was
one signed 16-bit unit. The standalone reader recomputes the points and
outcome arithmetic from anonymous scalar scores.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_odss_matched_room download "$cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
for phase in freeze fit score replay; do
  backend/.venv/bin/python -B -m scripts.audio_odss_matched_room "$phase" "$cache" \
    --run "$run" --surrey "$surrey_room" --bbc "$bbc_room" --ami "$ami_sources" || exit 1
done
python3 -B -IS backend/scripts/review_audio_odss_matched_room.py "$receipt"
```
