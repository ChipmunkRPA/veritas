# Temporary changes in low-power bands under competing speech

The low-power spectral check found no useful speech edit or room replacement
at its frozen point. It stays in evaluation. The strongest proposed interval
also rarely covered the right location before any cutoff. Neither profile
added a correctly located speech edit to the current standalone check.

## Method and source roles

The proposed check uses only the submitted six-second mono recording. In
40 ms Fourier windows it measures power in 16 bands between 140 Hz and
7.6 kHz. It takes each band's 20th-percentile level over 300 ms and removes
the shared overall gain. The proposed location has a similar spectral shape
before and after and a second similar shape at the inner edges. Its score is
the weaker jump over those boundaries minus the greater within-shape
difference. Proposals range from 400 to 1,560 ms. A separate median profile
uses the same geometry. The method draws on the use of lower spectral
quantiles in speech noise estimation. On a mixture of multiple speakers,
even low-energy windows can still reflect speech.
[Quantile noise estimation study](https://doi.org/10.1109/ICASSP.2000.862122)
· [Frozen procedure](evaluation-panels/audio-local-floor-return-plan-2026-11-18.json)

One hundred ordinary calibration submissions from 12 SINE readers and four
AMI rooms set the highest observed points at 11.538841 dB for the lower
quantile and 10.029041 dB for the median. The held comparison uses 16
separate readers and four separate AMI rooms. It reproduces the exact
standalone speech-edit, unchanged, cross-faded ordinary-splice and
isolated-room inputs from the previous comparison. The new matched controls
are a local 3 dB level dip and local low-pass processing. The separate
background experiment replaces only the room within the annotated interval,
at the same whole-excerpt level and with cross-faded boundaries. This
creates a known change in the acoustic scene on genuine speech.
[Exact source reservation](evaluation-panels/audio-standalone-dual-view-sources-2026-11-18.json)
· [Prior standalone comparison](audio-standalone-dual-view-2026-11-18.md)

SINE releases its processed material under Apache 2.0. The AMI room
recordings use CC BY 4.0. These recordings have prior project exposure.
The new method and points were frozen before held scoring. No earlier
voice copy, source stem, learned weight or edit annotation enters the
scoring of a submitted file. Annotations define how the evaluation files
were made and whether a proposed range located the right change.
[SINE release](https://huggingface.co/datasets/PeacefulData/SINE#license)
· [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held comparison

The table counts a location when it covers at least half the annotated
interval and at least 0.30 temporal intersection over union. Each level
has 16 speech edits, 16 room replacements and 64 ordinary submissions
across the four matched control types. A further four isolated AMI
recordings were measured without a foreground. Both new profiles assessed
all 196 submissions.

| Method | Speech edits at −5 dB | Speech edits at +3 dB | Room replacements at −5 / +3 dB | Ordinary alerts at −5 / +3 dB | Isolated-room alerts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current two-second check | 0/16 | 3/16 | Outside the current comparison | 1/32 and 3/32 | 0/4 |
| Lower spectral quantile | 0/16 | 0/16 | 0/16 and 0/16 | 0/64 and 0/64 | 0/4 |
| Median spectral level | 0/16 | 0/16 | 0/16 and 0/16 | 1/64 and 1/64 | 0/4 |

The incumbent ordinary counts cover the original and same-recording
splice roles, for which exact prior scores are available. It assessed
two of the four isolated recordings. Both median
alerts came from local equalization. Neither new check as a fallback
added a useful location. The proposed clock geometry can represent
all 16 known edit ranges before energetic admission and scoring. Before
the cutoff, the strongest lower-quantile proposal usefully overlapped
1 of 16 and 3 of 16 speech edits at −5 and +3 dB, respectively. It
overlapped only 1 of 16 and 2 of 16 background replacements. The median
top proposal overlapped two speech edits at each level and one background
replacement in total. Even a retrospective budget of eight flags per
64 ordinary clips at each level with zero isolated-room flags would
locate one higher-SNR speech edit and zero lower-SNR speech
edits or room replacements with the low-power scores alone. The frozen
points remain unchanged.

[Calibration receipt](evaluation-panels/audio-local-floor-return-calibration-2026-11-18.json.gz)
· [Held measurements](evaluation-panels/audio-local-floor-return-held-2026-11-18.json.gz)

## Replay

Review the numeric evidence with Python's standard library only. To
measure the panel again, restore the publisher's pinned recordings using
the earlier study's source-replay instructions. Re-rendered controls
are bound to the exact unchanged mixture and the same clip gain. The
public receipts contain input hashes, scores, intervals and distances.
A separate local measurement regenerated both receipts byte for byte.
[Source replay](audio-standalone-dual-view-2026-11-18.md#replay)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module=scripts.audio_local_floor_return
base=docs/evaluation-panels/audio-local-floor-return
python3 -B -S -m "$module" review "$base-calibration-2026-11-18.json.gz" \
  "$base-held-2026-11-18.json.gz"
backend/.venv/bin/python -B -m "$module" calibrate "$new_calibration" "$run_sources" "$ami_backgrounds"
backend/.venv/bin/python -B -m "$module" held "$new_calibration" "$run_sources" "$ami_backgrounds" "$new_held"
```
