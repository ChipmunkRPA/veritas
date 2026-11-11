# Serving benchmark stopped at external object storage

The corrected isolated settings loaded successfully and the first serving
worker recorded all eleven planned cases. **All eleven failed during ingest**
when the unchanged object store tried to create a hard link on the external
archive volume. The filesystem returned `OSError` errno 45, “Operation not
supported.” No detector stage was reached, and every case remained unavailable.

The recorder exited zero because it saved all failures; that is not a successful
analysis result. The process monitor reported no guard or sampling error. The
[actual failure receipt](evaluation-panels/grid-serving-baseline-storage-stop-v153.json)
binds the unchanged attempt, private log, case journal and resource profile.
It contains no detector coverage or accuracy measurement.

Preserve the attempted v153 stage and its artifacts. Do not run additional v153
cases on that storage. A new isolated context must use a hard-link-capable
internal working volume and prove the production object-store operation before
the first analysis. Keep original evaluation media on the external archive and
leave app code, inference settings and resource limits unchanged. No live app
data, billing, defaults or deployment changed.
