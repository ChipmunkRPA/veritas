# Repeated-audio evidence under recorded noise and codec changes

The app's exact check requires identical decoded samples. A prior comparison on GRID clips found that a waveform-similarity check recovered more known copies after lossy encoding. This comparison takes both methods to different recordings and crosses delivery codecs with two recorded backgrounds. It also measures a separate frequency-band candidate at fixed settings. [Existing codec comparison](grid-audio-repetition-codecs-v170-result.md)

## Fixed comparison

The roster includes all 24 qualified three-second windows from two AMI meeting sessions and all nine qualified RUEG crops from nine distinct publisher participant codes. The windows are disjoint from the GRID and short-edit repetition panels. Both source sets were previously used for other local preparation or evaluation tasks. AMI windows from the same meeting share a source group. The nine RUEG crops represent the previously qualified subset of 24 planned speakers. Every submitted source WAV and its decoded sample commitment were checked against the earlier qualification record. [AMI qualification and CC BY 4.0](audio-ami-native-stereo-v291-result.md) · [RUEG qualification and CC0](audio-rueg-genuine-v246-method.md)

Each original is paired with a within-recording copy. The fixed recipe selects the most energetic centered 650 ms interval among six evenly spaced candidate positions starting at 0.100 seconds. The positions are 200 ms apart. It replaces 1.750–2.400 seconds with the copy and applies 20 ms raised-cosine blends at both edges. Comparison uses the interior after excluding those blends. Source selection, copy placement, comparison methods, and the matching rule were committed before scoring.

The five conditions are clean and the DEMAND washroom and river backgrounds at +5 dB and −5 dB whole-clip SNR. The noise comes from channel one in each checksum-pinned publisher archive. Both versions of a source use the same tape excerpt, background gain, and common attenuation for each level. The original defines nominal SNR. The realized whole-clip SNR of the copy differs from the assigned level by up to 3.83 dB. Tape windows begin at 2 seconds and advance by 8 seconds in fixed source order. The noise tapes were used in earlier calibration studies with other foregrounds. [DEMAND publication and CC BY-SA 3.0](https://zenodo.org/records/1227121) · [Prior matched preparation](audio-demand-controls-v204-method.md)

Each condition passes through 16 kHz mono PCM WAV, AAC at 64 kb/s, MP3 at 64 kb/s, and Opus at 48 kb/s. All inputs use the production decoder with checked codec clock and exact sample count. For AAC, actual decoder padding remains in the input and its length is recorded. A correct match requires both reported locations to overlap the known donor and target by at least 2,400 samples each. The exact method requires the authored lag. Both similarity methods allow 32 samples of lag error. Every reported pair on an untouched file counts as an alarm. A copy with reported pairs and no correct pair is recorded separately.

The first similarity method uses the unmodified v170 pre-emphasized waveform implementation. The second filters the waveform into 180–900 Hz, 900–2,500 Hz, and 2,500–6,000 Hz bands with fourth-order Butterworth sections in both directions. Each band nominates up to eight autocorrelation lags. Nominations within four samples of a stronger lag are merged. Local verification uses consecutive 40 ms windows, stepping 10 ms. At least two bands must reach correlation 0.65 and RMS amplitude 16 on both input windows. Two nonoverlapping ranges must have at least 150 ms of supported samples. A case that produces more than eight pairs is recorded as ambiguous. The candidate settings were fixed without looking at these test responses.

## Local results

All 1,320 planned views decoded and reached all three methods. Each condition includes 132 known-copy views and 132 untouched views from the same 33 foregrounds across four codecs. A separate arithmetic audit checked every location against the authored donor, target, and lag. It also verified the decoded WAV identities, codec sample counts, and case totals. The study ran zero neural forwards. [Results by codec, noise, and anonymized source group](evaluation-panels/audio-repeat-robustness-transfer-2026-11-11.json)

| Condition | Exact samples | Waveform similarity | Two-band similarity |
| --- | ---: | ---: | ---: |
| Clean | 33/132 | 112/132 | 90/132 |
| Washroom +5 dB | 0/132 | 115/132 | 90/132 |
| Washroom −5 dB | 0/132 | 63/132 | 92/132 |
| River +5 dB | 0/132 | 34/132 | 55/132 |
| River −5 dB | 0/132 | 4/132 | 4/132 |
| All known copies localized | 33/660 | 328/660 | 331/660 |
| Untouched views with a reported pair | 0/660 | 2/660 | 0/660 |
| Copy views with only a wrong location | 0/660 | 11/660 | 1/660 |
| Copy views unavailable or ambiguous | 0/660 | 49/660 | 132/660 |

The exact method localized all 33 clean WAV copies and none of the 99 clean lossy copies. On those 99 lossy copies, waveform similarity localized 84 and two-band similarity localized 67. The two waveform-similarity alarms appeared in AAC and Opus versions of the same untouched RUEG recording under washroom noise at nominal −5 dB. The two-band method localized 198/480 AMI and 133/180 RUEG copies. Waveform similarity localized 231/480 AMI and 97/180 RUEG copies. Repeated conditions and codecs share source material, so the 660 views per arm are not independent recordings.

An exact-or-two-band comparison localized 341/660 copies with zero untouched views carrying a pair. An exploratory union of all three localized 398/660 and retained both untouched alarms. The additional bands help in the washroom at −5 dB and river at +5 dB, and their stricter activity requirement leaves more AMI inputs unassessed. Both similarity methods miss nearly all copies in the river at −5 dB. This is an ordinary-edit localization experiment on 11 source groups and two tapes. It leaves synthetic-voice detection, different-source splices, and broader noise and recording conditions unmeasured. The similarity methods remain experimental.

Two published approaches motivate a separate time-frequency keypoint comparison. The Mel-spectrogram study reports white-noise tests at +20 and +30 dB, which are milder than the recorded strong-noise condition here. The September 2026 keyframe study evaluates longer audio and music with distinctive local matches. A follow-up should freeze its method on separate development material, then compare it on disjoint real recordings with repeated words, music, and low-SNR edits. [Mel-spectrogram keypoint study](https://doi.org/10.3390/app152111845) · [Keyframe-matching study](https://doi.org/10.1016/j.specom.2026.103450)

The local freeze SHA-256 is `a7a2bc4d0797d1cc7224e91c3a469de9282f003db697bdf57068fd35071db325`. The executed runner SHA-256 is `f419e63342b1bbf72600d3803cc1ada725aeb3212b251aa60275b5b416f69e4f`. The result carries the whole private journal commitment. Recordings and per-view sample offsets remain local.
