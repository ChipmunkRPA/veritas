# Formatting response comparison: historical stop, later completion

**Update:** the [complete model comparison and separate review](text-format-comparison-v94-result.md)
have now passed. All eight previously unattempted batches completed, covering
800 effective inputs and preserving every unchanged reference response. The
paragraph-layout intervention changed current highlights in 91/288 documents.
This is a controlled formatting result, not fresh-author accuracy or a serving
change. The earlier before-launch resource stop below remains historical evidence.

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

**At that earlier stop, zero model workers and zero new neural calls started.**
No scientific worker profile, response journal or model comparison result then
existed. This was a before-launch resource stop, not a failed model execution or
evidence about accuracy/robustness. All eight fixed batches were then unattempted. No map, point,
head, source, runtime limit or serving default changed; no failed attempt was
silently overwritten or retried. The later successful launch kept the same
startup check and all original limits, with process permissions checked first.

The [v93 preparation](text-format-audit-v93-preparation.md) remains complete:
288 consumed controlled documents, 1,440 requested views and 800 distinct
effective classifier texts. Original character correspondence and labels are
verified. The completed comparison now measures controlled formatting stability;
fresh natural documents and serving qualification remain unfinished. Code
checks and input equivalence alone did not establish that stability. No workflow,
outreach, real payment or GCP work occurred.
