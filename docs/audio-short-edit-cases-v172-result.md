# Matched short speech edit panel prepared locally

Both guarded construction runs completed with zero unavailable cases and zero
model calls. The readouts verified all 1,347 private WAV commitments. Each
case stores exactly three seconds of 16 kHz mono PCM. Preparation completed
in 4.51 seconds for development and 5.05 seconds for holdout, with maximum
sampled process-tree RSS below 92 MiB in both runs.
[Frozen construction method](audio-short-edit-cases-v172.md) · [Verified aggregates](evaluation-panels/audio-short-edit-cases-result-v172.json)

| Speech-origin truth | Development | Holdout |
| --- | ---: | ---: |
| Genuine throughout | 375 | 408 |
| Genuine host with an AI insert | 207 | 216 |
| Fully synthetic speech | 69 | 72 |
| Total WAV cases | 651 | 696 |
| Source recordings | 37 | 40 |

Each origin condition includes clean, 0 dB and +5 dB cases. Matched ordinary
and AI inserts use the same host, noise and gain, with hard and faded joins.
An independent PCM replay verified shared noise residuals in all 898 noisy
cases and confirmed the recorded edit containment on all 1,116 splices.
Each ordinary and AI pair had identical samples outside its splice. The
sample records retain inserted support, blend support and observed changes.

Source and noise reuse make cases within a role dependent. These counts
describe preparation and provide no detection or calibration estimate.
Holdout scoring stays withheld until the comparison is frozen from
development.
