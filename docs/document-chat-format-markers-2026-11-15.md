# Chat formatting markers in document uploads

In **Text and document evidence**, new analyses show matching opening
and closing markers when the source contains an intact assistant-style
message. The browser and Standard PDF show the literal markers and their
locations. The JSON retains the same source record, and an offline
verifier can reproduce it from the original upload.

These delimiters are public formatting syntax. Anyone can type them,
quote an example, or copy them from a transcript. The check assigns zero
AI-authorship weight. Statistical writing checks continue when this
observation is present. Verified credentials use their own provenance
evaluation. [Provider provenance coverage](provider-provenance-coverage.md)

## Source and format coverage

The check requires an eligible opening and its corresponding closing
marker with text between them. It declines incompatible closers,
uncompleted messages, and pairs that cross recognized turn or tool
boundaries. It leaves fenced examples out when those boundaries are
preserved by extraction. It reports up to eight located messages and
keeps the total count. Each pair spans up to 16,384 extracted characters.

| Documented formatting | Primary source |
| --- | --- |
| OpenAI Harmony final channel | [OpenAI format specification](https://github.com/openai/harmony/blob/main/docs/format.md) |
| ChatML-style role, including the published Qwen template | [Qwen template](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/chat_template.json) |
| Meta Llama 3 role header | [Meta prompt format](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/llama3_1) |
| Google Gemma 1–3 turn markup | [Google formatting guide](https://ai.google.dev/gemma/docs/core/prompt-structure) |
| Google Gemma 4 turn markup | [Google formatting guide](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4) |
| DeepSeek V3 assistant delimiters | [DeepSeek tokenizer template](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json) |
| MiniMax M2 assistant framing | [Pinned MiniMax template](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/04312f0cfbdeefb912575595b7c045512a29293c/chat_template.jinja) |
| Moonshot Kimi K2.5 assistant framing | [Pinned Moonshot template](https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/chat_template.jinja) |

Evidence uses the original decoded character positions for plain text,
Word paragraphs and supported email bodies. HTML markers need a verified
mapping back to the decoded MIME part, including escaped entities.
Selectable PDF text is eligible when both markers have source block
locations on the same page. The PDF location includes the page, mapped
block and extracted character offsets. OCR-only text, complex unmapped
HTML, email quotations and attachments are outside this check.

These sources document formatter behavior for the named releases.
Normal copied answers may contain none of these delimiters. The check
has no access to private provider marks or keys. Gemma coverage applies
to the documented open Gemma formats.

## Local verification and replay

The format tests cover eight opening and closing pairs, Unicode and
line-end normalization, wrong closers, unsupported turns, fenced samples,
an original Word source, an escaped HTML message and text on a selectable
PDF page. An additional panel uses 72 frozen ordinary government-report
excerpts from the earlier writing study. All 72 had usable source
mapping and none contained a qualifying pair.
[Ordinary report receipt](evaluation-panels/document-chat-format-markers-controls-2026-11-15.json)

The paid local replay verifies marked plain text, Word, HTML email and PDF,
and an ordinary plain-text control. It uses simulated USD 9.99 checkout.
No analysis job starts before payment. Each eligible file publishes its
source-position commitment. The replay confirms the evidence API and JSON
agree, checks the marker text and offsets in the reports, and verifies the
signed packages against the original documents.
[Local app receipt](evaluation-panels/document-chat-format-markers-app-2026-11-15.json)

The initial Metal worker stopped inside the native device-memory probe.
The CPU replay completed all five passage classifiers. Available memory
prevented the optional writing-pattern check from running in that replay.
The reports record that coverage alongside the marker result.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
backend/.venv/bin/python -B -m scripts.document_chat_markers_app "$run/app"
backend/.venv/bin/python -B -m scripts.document_chat_markers_app "$run/controls.json" \
  --ordinary-roster "$report_roster"
```

Use the exact extended source roster from the [matched report study](writing-matched-encoder-2026-11-14.md#replay)
for the ordinary controls. The local app replay uses only authored example
text. The public receipts carry hashes and marker positions.
