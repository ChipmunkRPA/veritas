# Three-sentence writing check

The optional check marks an exact middle sentence when its AI-like writing response is sufficiently stronger than those of both neighbors. Plain Media Evidence, the Standard PDF, and the JSON show the same original-text range. The check contributes zero weight to the overall assessment. An unflagged or unassessed passage leaves authorship undetermined.

## Local setup

For an instance using CPU or Apple GPU (MPS) with the existing pinned candidate writing checkpoint installed, set the following and restart the worker. The check is off by default. It runs on new analyses of a mapped, single English paragraph with exactly three sentences and complete tokenizer coverage. It accepts plain TXT, a simple Word body paragraph, and a single paragraph in a selected plain-text email body. The Word check requires a complete ordinary paragraph without tables or revision structures in the paragraph. The email check requires a single mapped message paragraph in the selected MIME part. Its coverage excludes quoted or signed tails, HTML bodies, attachments, and headers. PDFs, multi-paragraph files, and other languages are outside this three-sentence check's scope. An eligible analysis retains its normal result when the optional responses are unavailable.

```dotenv
VERITAS_TEXT_DETECTOR_MODE=candidate
VERITAS_TEXT_WINDOW_SENTENCES=3
VERITAS_TEXT_PASSAGE_POSITION_REVIEW=true
```

## Method and supporting evidence

The worker reuses the loaded encoder to read three complete inputs. The middle response uses all three sentences. Each outer response uses that sentence and its one immediate neighbor. The saved local contrast is the middle response minus the stronger outer response. The check reuses the frozen fitted coefficients and operating point from the evaluation. These are relative raw responses. Customer reports use neutral names for the checks. [Fitted head and evaluation](writing-sentence-context-head-2026-11-12.md) · [Frozen rule](writing-two-context-transfer-2026-11-12.md)

Across three frozen historical publisher evaluations, the check located six additional known edits from two generator families and added zero flagged ordinary work groups. In the most recent test, it found two of 13 edits missed by the current check and flagged zero of 14 originals and zero of 14 genuine publisher revisions. Its low recovery rate and the misses on other tested generators limit what the response can establish. [Earlier confirmation](writing-position-confirmation-2026-11-12.md) · [Second-generator confirmation](writing-position-qwen-confirmation-2026-11-12.md)

The signed package binds the coverage record to the projected result. With the original text, the verifier reconstructs the exact excerpt, all three positions, and input hashes. It also recalculates the direction from the recorded readings. The source replay uses the saved model responses. It does not repeat inference. [Local application receipt](evaluation-panels/writing-position-app-2026-11-12.json)

The Apple GPU replay reused 68 eligible passages from the two retained publisher panels. It reproduced all 68 CPU decisions at the existing point. The largest change in the raw contrast was 0.000029. Four separate worker runs with simulated prepaid checkout exercised the Apple GPU path. Both known rewrites flagged, the ordinary revision did not flag, and the disabled run omitted the check while preserving the same assessment and findings. No job ran before payment. Chrome showed the exact sentence and all three source ranges, the JSON and PDF downloads matched the API, and signed source verification reproduced the selected text and positions.

## Word and email coverage

The format comparison reused 56 excerpts from the earlier writing agreement format bank. For each eligible case, it took the first single sentence fully inside the annotated span whose two neighbors were outside the span and met the three-sentence check's input rules. Each plain-text, Word, and EML version ran separately on CPU with the same weights. All 56 passages produced identical input hashes, raw responses, sentence ranges, and flag decisions across the three formats. The check flagged 8 of 23 AI rewrites and 5 of 33 ordinary passages, including 3 of 16 originals and 2 of 17 human revisions. That measures format parity for the same words across controlled files. It offers no accuracy evidence for natural correspondence or complex document layouts. [Earlier format bank](writing-agreement-document-format-transfer-2026-11-12.md)

Word results identify the original body paragraph and its character positions. EML results identify the selected MIME part and positions in its decoded text. The signed replay checks the source locator and exact input hashes. Repeated spaces preserve their full original character range when collapsed for analysis.

Six local analyses exercised a known rewrite and an ordinary control across the three formats with simulated prepaid checkout. The known rewrite selected the same middle sentence at characters 173–320 in every format. The ordinary control returned no flag in all three. The API, JSON, Standard PDF, and signed source replay agreed. The browser component tests checked both native position labels. An unrelated optional writing check was skipped by the worker's memory budget during these runs.

## Assets and attribution

Chipmunk Robotics fitted and distributes the 1,025-parameter projection and its standardization statistics in `backend/app/detectors/text/calibrations/passage-position-head-v1.json` under the repository's MIT license. The asset contains numeric coefficients only. The fit derives from 12 historical works by the Kubernetes website contributors, whose archived prose is licensed CC BY 4.0. The experiment extracted excerpts and prepared single-sentence substitutions with pinned Apache-2.0 Qwen and SmolLM2 generators. The source prose, model weights and raw generated text remain outside the distributed asset. [Kubernetes license](https://github.com/kubernetes/website/blob/main/LICENSE) · [Archived fit sources](writing-multiblock-independent-2026-11-12.md) · [Generator terms](writing-archived-passage-transfer-2026-11-12.md)

The existing Apache-2.0 encoder is attributed to WasItAIGenerated. Its publisher NOTICE attributes Desklib and Microsoft Corporation for upstream MIT components. The app loads the pinned checkpoint and ships the numeric projection separately. [Publisher card](https://huggingface.co/wasitaigeneratedcom/tropa-mini-ai-text-detector) · [Repository notices](../THIRD_PARTY_NOTICES.md)
