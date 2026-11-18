# Public Content Credentials reference files

On November 17, 2026 (Pacific time), Chipmunk Verify read the published bytes
of 21 image, video, audio, and PDF files from 10 provider families. The check
used the serving adapters, C2PA SDK 0.90.14, and the bundled C2PA trust lists.
Six files passed the trusted AI-declaration gate. The replay verified only
embedded credentials. Provider watermark verifiers and remote manifests
received zero requests.

## Files with signed records

| Provider and download | Local result | Consequence |
| --- | --- | --- |
| Google Gemini [JPEG 1](https://huggingface.co/datasets/bigdatamark/synthid-research/resolve/af707a1b8823b277f260a1ebfbae3c9877d3d897/gemini-3.1-flash-image/001.jpg), [JPEG 2](https://huggingface.co/datasets/bigdatamark/synthid-research/resolve/af707a1b8823b277f260a1ebfbae3c9877d3d897/gemini-3.1-flash-image/002.jpg) | Google Media Processing Services signed an AI source declaration bound to both files. The recorded actions are opening and resizing. | Skip deeper checks. |
| OpenAI [PNG 1](https://huggingface.co/datasets/bigdatamark/synthid-research/resolve/af707a1b8823b277f260a1ebfbae3c9877d3d897/gpt-image-2/001.png), [PNG 2](https://huggingface.co/datasets/bigdatamark/synthid-research/resolve/af707a1b8823b277f260a1ebfbae3c9877d3d897/gpt-image-2.5-flare/1.png), [PNG 3](https://huggingface.co/datasets/bigdatamark/synthid-research/resolve/af707a1b8823b277f260a1ebfbae3c9877d3d897/gpt-image-2.5-sunburst/1.png) | OpenAI Media Service signed an AI source declaration bound to all three images. | Skip deeper checks. |
| OpenAI [PDF](https://raw.githubusercontent.com/richardwooding/c2pa/1119ea0c86b3ede12461b477b9069cefc3ee9e95/testdata/c2pa_chatgpt.pdf) | OpenAI Media Service signed an AI source declaration with verified whole-document binding. | Skip writing checks when the [opt-in PDF check](pdf-content-credentials-2026-11-13.md) is enabled. |
| Adobe [Firefly JPEG](https://raw.githubusercontent.com/contentauth/example-assets/c37f93e115289ac1ef00f899a860e6e08ab9886f/images/Firefly_tabby_cat.jpg), [Photoshop JPEG](https://raw.githubusercontent.com/contentauth/example-assets/c37f93e115289ac1ef00f899a860e6e08ab9886f/images/car-es-Ps-Cr.jpg) | Both signatures are intact and match the files. The legacy list recognizes the signers. The Firefly sample declares AI involvement. The Photoshop sample records opening, cropping, and resizing. | Continue. The current official trust list is required. |
| Older OpenAI [ChatGPT PNG](https://raw.githubusercontent.com/contentauth/example-assets/c37f93e115289ac1ef00f899a860e6e08ab9886f/images/ChatGPT_Image.png), [Sora MP4](https://raw.githubusercontent.com/contentauth/example-assets/c37f93e115289ac1ef00f899a860e6e08ab9886f/videos/sora.MP4) | The file hashes match. The certificates are expired and untrusted under the configured validator. The Sora record contains an AI declaration. | Continue. |
| Anthropic Claude [PNG](https://c2paviewer.com/samples/claude-api-c2pa-sample.png), [JPEG](https://c2paviewer.com/samples/claude-api-c2pa-sample.jpg) | Signed, file-bound records of delivery through Claude. Both lack an explicit AI source declaration. The signer is untrusted under the configured lists. | Continue. |

OpenAI and Anthropic both describe signing in workflows that can handle an
existing file. Chipmunk Verify shows the signer and reads what the signed
record actually declares. The gate requires a trusted signature, matching
file hash, explicit AI source declaration, and zero validation failures.
A verified declaration records AI involvement. It carries no statistical
confidence score and makes no claim about the authorship of every part.
[OpenAI provenance documentation](https://developers.openai.com/api/docs/guides/content-provenance) ·
[Claude file-signing documentation](https://www.anthropic.com/news/claude-text-watermark#08-other-questions) ·
[Google hosted image and video credentials](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/content-credentials)

## Additional providers and products

| Provider | Documented signal and observed coverage | Provider source |
| --- | --- | --- |
| Meta Muse | Meta documents invisible Content Seal on supported Muse Image outputs. No embedded C2PA appeared in either blog image preview. The previews were tested only for C2PA. Meta has announced video expansion. | [Muse documentation](https://ai.meta.com/blog/introducing-muse-image-muse-video-msl/), [Meta identification portal](https://www.meta.ai/identification/) |
| xAI Grok | The checked Imagine MP4 previews lack embedded C2PA. Separately, xAI documents opt-in SynthID Text and an application-based verifier, plus AudioSeal on supported voice APIs. A matching AudioSeal family signal alone cannot identify xAI as the producer. | [Imagine demos](https://x.ai/api/imagine), [Grok watermark documentation](https://docs.x.ai/developers/advanced-api-usage/watermarking) |
| Runway | Runway Dev documents signed AI declarations for native output. Runway says the sample web videos were re-encoded. Both previews lacked C2PA. | [Runway Dev export guidance](https://dev.runwayml.com/learn/generate-ugc-videos) |
| Suno and ElevenLabs | Suno documents C2PA for new song downloads. Its public M4A showcase lacked a manifest. ElevenLabs documents a watermark-first audio verifier with a classifier fallback. | [Suno Credentials](https://suno.com/suno-credentials), [ElevenLabs Audio Detector](https://elevenlabs.io/docs/eleven-creative/audio-tools/audio-detector) |
| Microsoft and Amazon | Selected Microsoft Designer and Foundry outputs carry C2PA. Amazon documents C2PA on historical Nova Canvas images and Nova Reel 1.1 videos, and a separate speech watermark for Nova Sonic. | [Microsoft Designer](https://support.microsoft.com/en-us/windows/ai/ai-apps/edit-images-with-designer), [Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-safety/provenance-disclosure), [Amazon Nova Canvas](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-canvas.html), [Amazon Nova Reel](https://docs.aws.amazon.com/ai/responsible-ai/nova-reel/overview.html), [Bedrock watermarking](https://docs.aws.amazon.com/bedrock/latest/userguide/watermarking.html) |
| ByteDance, Kuaishou, and Alibaba | CapCut describes credentials and watermarks for named Seedance and Seedream tools. Kling offers an image/video portal for its digital watermark. Alibaba documents cloud detection of implicit AIGC labels. The researcher-labeled Qwen image lacked embedded C2PA. | [CapCut announcement](https://www.capcut.com/newsroom/giving-creators-more-control-with-dreamina-seedance-2-5-and-dola-seedream-5-0-pro), [Kling detection](https://kling.ai/docs/ai-content-detection), [Alibaba Guardrails](https://help.aliyun.com/en/document_detail/2979119.html) |
| MiniMax, DeepSeek, Baidu, and Tencent | Published signals include MiniMax export notices, visible DeepSeek disclosures, Baidu image watermarking, and Tencent's scoped labels and metadata tools. The review found no verified general-purpose local decoder for these providers. | [MiniMax export labels](https://code.minimax.io/doc/en/download-watermark-rules.html), [DeepSeek disclosure](https://cdn.deepseek.com/policies/en-US/model-algorithm-disclosure.html), [Baidu hidden image mark](https://cloud.baidu.com/doc/NLP/s/clxim3bfq), [Tencent video notice](https://cloud.tencent.com/document/api/1616/119001) |
| Stability AI, FLUX, Bria, Midjourney, and Synthesia | These represent managed C2PA, own-source watermark verification, and metadata-backed app verification. The researcher-labeled FLUX PNG lacked embedded C2PA. Product and export scope are recorded individually. | [Stability AI credentials](https://kb.stability.ai/knowledge-base/content-credentials-c2pa-faq), [FLUX service terms](https://bfl.ai/legal/developer-terms-of-service), [Bria verification](https://docs.bria.ai/visual-verification-overview), [Midjourney verifier](https://www.midjourney.com/verify), [Synthesia video labels](https://help.synthesia.io/en/articles/17634791-how-do-i-add-an-ai-label-to-my-videos) |

The [four-modality matrix](provider-signal-matrix.md) and
[primary-source catalog](provider-signals-catalog.json) cover 82 families
and 110 scoped signal records. They include independent review gaps for
providers such as Cohere and Luma's signed exports. Text marks from OpenAI,
Anthropic, Google, and Grok use statistical token-choice methods for specific
products. Whitespace and pasted provider names carry no authenticated origin.

When a download has no embedded credential, the normal scan continues.
This also applies to an unsupported container or invalid record. Published
previews can differ from original exports, and a separate watermark may
exist in the pixels, audio, or text.

## Replay on the downloaded file

The [verification receipt](evaluation-panels/public-content-credentials-2026-11-17.json)
names each exact URL, SHA-256, certificate identity, validation outcome and
gate disposition. Save a chosen sample as `<id>.asset` in a scratch directory.
Then run the offline replay from the project root with its backend environment.
Certificate dates and trust lists can change the outcome on a later replay.

```sh
scratch=$(mktemp -d)
receipt=docs/evaluation-panels/public-content-credentials-2026-11-17.json
sample=gemini_synthid_001
curl -fsSL "$(jq -er --arg sample "$sample" '.samples[] | select(.id == $sample) | .url' "$receipt")" \
  -o "$scratch/$sample.asset"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_public_content_credentials \
  "$receipt" "$scratch" --sample "$sample"
```

The ordinary result shows the verified signer and the source declaration
when the analysis has a saved credentials record. The detailed record also
keeps the validation codes and ingredient chain. A local browser replay of
one Gemini JPEG, one OpenAI PNG, and the OpenAI PDF showed all three signers
after simulated payment. It recorded zero jobs before payment and zero
statistical model leases during those analyses. The repository distributes
the receipt and replay code. The source media remains with its publishers.
