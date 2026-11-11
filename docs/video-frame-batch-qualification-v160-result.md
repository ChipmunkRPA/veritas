# Local frame check parity

Both guarded Apple GPU runs completed on the same 15 stored face crops. The
one-face batch produced exactly the same member logits, frame probabilities,
aggregate, highest-scoring frame, raw decisions and evidence timestamps as
the reference batches of eight and seven. Both used the same two verified
checkpoints and byte-identical float32 model inputs.
[Method](video-frame-batch-qualification-v160-method.md) · [Actual result](evaluation-panels/video-frame-batch-qualification-complete-v160.json)

| Arm | Largest forward batch | Max sampled descendant RSS | Worker elapsed |
| --- | ---: | ---: | ---: |
| Reference | 8 | 1,349,664,768 bytes | 9.22 s |
| One face per batch | 1 | 1,271,463,936 bytes | 8.71 s |

The one-face arm used about 75 MiB less sampled process memory. GPU driver
allocation at exit was about 40 MiB higher. Peak GPU memory was outside the
probe. This trial establishes numerical parity on one clip. App admission,
end-to-end resource use, and false alarms and misses on independent videos
need a separate comparison before any default changes.
