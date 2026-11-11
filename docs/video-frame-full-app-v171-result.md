# Local full-app result for one-face video batching

Both guarded arms completed on the same genuine six-second 1080p video.
The one-face adapter executed and published source-bound scores for all
24 sampled face frames. The reference skipped this route under the
existing model-manager activation budget. Both arms matched the
independent frame clock, sampled RGB and mono PCM.
[Method](video-frame-full-app-v171-method.md) · [Actual result](evaluation-panels/video-frame-full-app-complete-v171.json)

| Outcome | Reference | One-face adapter |
| --- | ---: | ---: |
| Declared activation allowance | 6,400 MiB | 800 MiB |
| Successful detector routes | 13 | 14 |
| Scored frames from the target route | 0 | 24 |
| Maximum sampled process-tree RSS | 4,858,970,112 bytes | 5,441,732,608 bytes |
| Guarded worker elapsed | 40.60 s | 46.69 s |
| App overall tendency for this genuine control | Unavailable | Leaning original for visual content |

The complete app used more sampled process memory with the additional
route active. Both arms completed under the same resource monitor.
The candidate's committed detector output, linked signal and frame
series agreed. The other route statuses and saved fuser identity
matched, and both evidence sets passed their asset checks.
An independent replay verified both committed result hashes and found that
the public projection exposed checkpoint pins. The corrective branch
withholds those fields in the API, JSON exports and technical appendix.
Replaying the same two results through it found zero matches among 77
registered model identifiers and recorded checkpoint pins.
[Public projection fix](https://github.com/ChipmunkRPA/veritas/tree/codex/public-model-revision-redaction)

The first v168 reference stopped before inference when application
imports added five known dependency defaults to the environment. The
corrected v171 worker binds the exact values before entry and preserves
the earlier attempt. This establishes app admission and evidence delivery
on one genuine control. Accuracy on independent face swaps and genuine
sources remains to be measured before adopting the adapter.
