# Lip-sync coverage when the visible mouth stays still

The default audio–video timing check now excludes a speech window when its measured mouth-opening proxy is flat or its native crop input is missing. The earlier correlation code returned zero for a flat input. On a track attributed from moving windows, that zero could enter the correlation-collapse calculation. The updated check distinguishes an unchanged mouth crop from a varying crop with a flat proxy. Both cases are unassessed for timing. The report identifies how many speech windows lacked measurable mouth input, and offset comparisons stop at those omitted windows.

## Native test

Three source-separated CREMA-D recordings supplied moving faces and recorded speech. We selected the three longest qualified recordings before scoring. Each supplied a 25 fps native RGB reference, an ordinary CRF23 recompression, and a lossless copy with its picture held from 1.000 up to 2.520 seconds. The first recording also supplied one fully held control. All deliveries within each recording used identical decoded PCM. We verified the 979 output picture timestamps, the lossless picture commitments, and the moving-window evidence through the serving adapter. The test supplied one observed face track per clip. [CREMA-D source license](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Prior source qualification](video-native-source-v92-result.md)

| Condition | Planned clips | Scored windows before | Scored windows after | Unchanged-mouth windows omitted |
| --- | ---: | ---: | ---: | ---: |
| Moving reference | 3 | 5 | 5 | 0 |
| Ordinary recompression | 3 | 5 | 5 | 0 |
| Authored picture hold | 3 | 3 | 1 | 4 |
| Fully held control | 1 | 0 | 0 | 0 |

The authored hold excluded two windows in each of two source groups, at **1.000–2.000 seconds** and **1.480–2.480 seconds**. One held clip retained its moving window and the other reported that timing could not be measured. The offline native verifier matched before, during and after PNGs at 0.960, 1.760 and 2.520 seconds for the mixed clip. The third recording supplied no complete speech window for the existing lag search under any delivery. A deterministic regression with a held mouth and continuing speech reproduced the old correlation collapse and confirmed that the updated route omitted all five flat windows. An additional test verified that offset changes across omitted windows do not become measured steps. [Exact aggregate](evaluation-panels/video-lipsync-mouth-coverage-2026-11-11.json)

This change improves which intervals qualify for a timing comparison. It leaves detection of subtle articulation changes, well-synchronized generated speech and actual face swaps outside this test.
