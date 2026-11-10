# Formatting response comparison: fixed method, before-launch stop

The [protocol](text-format-comparison-v94-protocol.md), worker, strict neural
runtime, tests and [freeze](evaluation-panels/text-format-response-freeze-v94.json)
were published by manual push in `5194c98`. The exact remote freeze blob,
`becabb4f736f538558dc32df7f6e7ba753479ecf`, was verified before queueing any
new model execution. All 24 focused formatting/comparison checks and lint passed
locally. Input/freeze inspection imported no neural library.

The full-video decoder completed and its finalization passed before this queue
started. The text admission controller then waited for two observations of at
least 6 GiB available RAM, ten seconds apart, using the unchanged neural startup
requirement. It reached its 900-second wait limit **before attempting batch 0**.
Dispatcher exit status 1 was observed; the stop is retained in the
[aggregate receipt](evaluation-panels/text-format-response-admission-stop-v94.json).

**Zero model workers and zero new neural calls started.** No scientific worker
profile, response journal or model comparison result exists. This is a
before-launch resource stop, not a failed model execution or evidence about
accuracy/robustness. All eight fixed batches remain unattempted. No map, point,
head, source, runtime limit or serving default changed; no failed attempt was
silently overwritten or retried. The owner has been asked to free about 1 GB
so the actual frozen startup check can pass.

The [v93 preparation](text-format-audit-v93-preparation.md) remains complete:
288 consumed controlled documents, 1,440 requested views and 800 distinct
effective classifier texts. Original character correspondence and labels are
verified. Detector stability, fresh natural documents and serving qualification
remain unmeasured by this new study. Code checks and input equivalence cannot
substitute for those missing response results. No workflow, outreach, real
payment or GCP work occurred.
