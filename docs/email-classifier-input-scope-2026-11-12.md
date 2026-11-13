# What the email writing checks read

For new email analyses, open **Media evidence** and look for **What the email writing checks read**. The table reports complete retained inputs for each recorded classifier, including inputs that read message text only, quoted text only, a signature only, or several of those roles together. It identifies the selected MIME part. A zero count describes the saved input scope. It gives no authorship verdict.

Styled or otherwise unsupported HTML may retain scored text inputs without a verified role map. Those rows appear under **Role unavailable**. An input with incomplete token coverage or beyond the listing limit appears under **Unlisted**. Attachments, alternate MIME bodies, and sender identity are outside these counts. The previous optional corroboration check keeps its existing input requirements.

The Standard PDF shows the same counts. JSON retains exact extracted Unicode ranges and SHA-256 hashes under `evidence.coverage.email_classifier_input_scope`. The signed package commits to the scope record. With the original EML, source replay reconstructs the selected part, role mapping, and each saved input hash. It can verify extracted hashes with roles unassigned for unsupported HTML. The coverage record carries no additional text excerpts or detector names.

The local paid worker recorded two message-only and three mixed inputs for an authored three-sentence plain reply followed by a quote and signature. An authored styled HTML reply recorded four inputs whose roles were unavailable. Simulated unpaid events queued zero analyses. The browser, PDF, JSON and signed source replay matched the stored evidence. These are software verification cases. [Local verification receipt](evaluation-panels/email-classifier-input-scope-app-2026-11-12.json)

## Archive intake

The Apache Software Foundation maintains public mailing-list archives and states that messages sent to its public forums are made public without conditions. Its Apache 2.0 contribution language addresses material intentionally submitted for inclusion in a work. Public thread pages alone did not establish a source-separated corpus of matched genuine revisions of the same email body. The candidate threads examined included patch diffs, automatically generated notices, and separate replies. No archive email bodies entered this test. [ASF archive policy](https://www.apache.org/foundation/public-archives.html) · [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
