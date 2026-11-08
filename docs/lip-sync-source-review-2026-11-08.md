# Direct audio-visual timing candidates: current permission and resource review

The existing timing and mouth-edit results remain insufficient for a new
lip-sync authenticity claim. A fresh primary-source review did not admit a
replacement checkpoint or run another lip-sync model. Do not present the
separate public-watermark contract as lip-sync validation.

The [Synchformer owner response](https://github.com/v-iashin/Synchformer/issues/21)
explicitly distinguishes its MIT source from checkpoint permission: the owner
could not provide an assurance that the repository grant covers pretrained
checkpoint files or clears their commercial use. A third-party project's MIT
redistribution notice does not resolve that original-source gap. The model was
not downloaded, relicensed or executed for Chipmunk Verify. Methods may inform
an independent implementation, but restricted/unresolved assets are not imported.

The new [AV-Sync Evaluator card](https://huggingface.co/qianyijie/avsync-evaluator/blob/main/README.md)
describes a reference-free alignment scalar built on Qwen2.5-Omni-3B. The head
card declares MIT and separately defers base weights to their provider's terms.
Its source describes evaluation of sound-picture alignment, not AI authorship.
Complete dependency/weight/license lineage, exact decoding/timing and runtime
qualification remain necessary. The roughly [8.8 GB checkpoint](https://huggingface.co/qianyijie/avsync-evaluator/commit/81111e1db38122cd8494c9c16aa9e2b9e45772ca) also exceeds the
current local free-disk allowance before base dependencies; it was not acquired
or quietly quantized into a different comparison. No accuracy gain follows
from its paper's or card's claims.

The earlier [DAVID-Net dependency exclusion](audio-visual-v12-source-review.md)
also remains: its MIT head depends on a noncommercial video backbone. Neither
its random-weight nor missing-backbone fallback is an acceptable substitute.

The source search did reveal the official
[LTX public marking pack](https://huggingface.co/Lightricks/LTX-2-Marking), which
can be tested with exact already-cached permitted audio weights. The
[separate native contract](ltx-marking-audio-v1-result.md) is now complete and
preserves wrong-payload presence plus noisy/partial misses. It adds an
executable provider-pattern route while leaving direct synchronization,
natural-media qualification, independent human testing and the full improvement
goal open. No outreach, customer data transfer, cloud resource or service
confidence change was made.
