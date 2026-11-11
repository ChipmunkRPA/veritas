# One-face video batching completes on a genuine masked speaker

The normal full-analysis pipeline completed in both arms on a genuine
1080p USGS interview excerpt. Both arms detected a face in all 24 sampled
frames, with identical sampled positions and face geometry. The one-face
adapter scored all 24 and published a matching frame series. The reference
skipped the target route under its existing activation budget.
[Frozen method](video-frame-full-app-v175-method.md) · [Verified result](evaluation-panels/video-frame-full-app-complete-v175.json)

| Outcome | Reference | One-face adapter |
| --- | ---: | ---: |
| Declared route activation allowance | 6,400 MiB | 800 MiB |
| Target face frames scored and published | 0 | 24 |
| Maximum sampled process-tree RSS | 4,752,424,960 bytes | 4,951,113,728 bytes |
| Guarded worker time | 44.17 s | 48.26 s |
| Overall app tendency | Unavailable | Leaning original for assessed visual content |

The saved result and evidence passed the exact picture-clock, RGB and mono
PCM checks. Both arms published the same appearance review interval at
1.268–2.836 seconds on this genuine clip. A read-only replay verified the
before, during and after frame roles, all 33 native RGB inputs behind the
interval and 133 saved evidence assets in each arm. The appearance finding
was present when the target route skipped, and it describes a visual change
that warrants review. The genuine source makes this a false alarm if the
appearance finding is interpreted as AI origin.

This is a second successful admission and evidence test on a genuine
recording. Independent face-swap sensitivity and interval localization on
confirmed edits still require rights-reviewed positive footage.
[First genuine control](video-frame-full-app-v171-result.md) · [USGS source](https://www.usgs.gov/media/videos/working-usgs-md-de-dc-water-science-center)
