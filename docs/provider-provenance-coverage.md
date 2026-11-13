# Provider signals and the provenance-first gate

Updated 8 November 2026. The [machine-readable catalog](provider-signals-catalog.json)
covers **82 provider families**, all four modality fields, **105 scoped signal
records**, primary-source URLs, verification routes and explicit gaps.
Fifty-four families have scoped documentation, including announcements or
policies with unresolved export scope; twenty-eight remain review gaps.
These numbers count research records, not working detectors.
This is a broader inventory, not a claim to decode every provider or every export.
The [four-modality matrix](provider-signal-matrix.md) and
[latest broader search](provider-signal-followup-2026-11-08.md) distinguish current
marking documentation, future commitments and missing verification routes.
Individual provider rows retain their actual review dates.

## What can stop processing now

Before models are acquired, supported image, video and audio pipelines validate
local Content Credentials. Further statistical checks stop only when all of
these hold: a trusted signer, valid credentials, an explicit AI declaration,
a hard binding to the current file, no validation failures and no unresolved
remote reference. The result records the declaration and skipped checks,
without inventing a statistical confidence score.

Image evidence also skips its reflected and non-mirrored copy searches when
that verified gate closes. The ordinary evidence view and PDF explain the
skipped coverage, and the signed export binds both statuses to the result and
the source hash. Local tests signed an eligible 768 × 512 PNG and verified
the full Quick, Full and Deep paths, including zero calls to either matcher.
The Quick test also rendered the PDF and verified the signed ZIP offline.

The audio adapter now checks **WAV, M4A, MP3 and FLAC**. Dispatch recognizes FLAC
with a leading ID3 credential tag as well as ordinary FLAC and MP3 with/without
ID3. This is format routing; actual SDK signature and content-binding validation
still decides the gate. There is no provider-name allowlist that bypasses trust.

A signed statement of AI involvement describes the signer's declaration. It does
not locate every edit, establish complete authorship, or verify truth or intent.
Capture-only credentials do not trigger this AI-positive shortcut. Missing,
stripped, invalid, untrusted, unsupported or inconclusive signals continue normal
detection and never establish originality. Offline trust uses the bundled
snapshot; revocation and remote manifests are not checked.

An optional [whole-document PDF credential check](pdf-content-credentials-2026-11-13.md)
also stops statistical writing checks when a trusted embedded declaration covers
the submitted PDF. This was verified offline on a pinned public provider PDF.
The accepted layout requires a single catalog-associated store and a signed byte
hash covering the file outside that store. Credentials on embedded objects,
Office documents, ordinary article spacing and production text watermarks do
not use this shortcut.

The gate remains inside the prepaid analysis pipeline. The initial inventory
review made no provider-verification calls. A [separate frozen qualification](provider-provenance-v1-result.md)
subsequently made nine real requests using authored controls and public provider
samples, with no customer uploads. Neither external route is enabled for customers.

The [marker policy correction](unverified-marker-policy-v1-result.md) also removes
an old fusion-stage rule that could force an AI outcome from copyable metadata
or an unkeyed public message after the fast gate correctly continued processing.
Those observations remain visible, but supply no AI-origin confidence. Historical
marker-led results receive an explicit retired-rule interpretation without rewriting
their saved calculations.

## Broader provider findings

| Provider family | Collected signals and verification route | Important qualification |
|---|---|---|
| OpenAI / ChatGPT / Sora | [Image C2PA; image/audio SynthID; Content Provenance API; restricted text access](https://developers.openai.com/api/docs/guides/content-provenance); [historical Sora video credentials and marks](https://openai.com/index/creating-with-sora-safely/) | The documented endpoint covers images/audio, not a universal Sora video detector. Historical Sora credentials remain worth inspecting on retained files; internal tracing is not a public decoder. |
| Anthropic / Claude | [Keyed token-choice watermark](https://www.anthropic.com/news/claude-text-watermark); [supported image-file credentials](https://www.anthropic.com/news/claude-text-watermark?gsid=c8c05ab2-f266-425f-8453-12ae96c34905) | No hidden characters. Private-preview text access differs from ordinary local credential validation. |
| Google / Gemini / Imagen / Veo / Lyria / NotebookLM | [SynthID for supported image, video, audio and text; public verification portal](https://deepmind.google/models/synthid/) | Public text reference code does not contain the production Gemini key. |
| Meta / Muse | [Custom Muse Image marking and public Content Seal research schemes](https://github.com/facebookresearch/content-seal) | Public AudioSeal/image/video/text research implementations do not automatically match Muse production. |
| xAI / Grok | [Supported API text watermark and access-controlled verifier](https://docs.x.ai/developers/advanced-api-usage/watermarking); [visible image/video marks](https://docs.x.ai/grok/faq) | A copied response header or logo is not file authentication. |
| Microsoft / Azure / Copilot | [Selected hosted C2PA and invisible marks](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-safety/provenance-disclosure); [verification API](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/how-to-provenance-detection) | Resource/Entra access is required; product availability varies. Indexed GPT request controls remain unresolved by the retrieved page. |
| Amazon / Nova / Bedrock | [Sonic speech marks and selected Claude text marks](https://docs.aws.amazon.com/en_en/bedrock/latest/userguide/watermarking.html); [Reel 1.1 video credentials and watermark](https://docs.aws.amazon.com/ai/responsible-ai/nova-reel/overview.html) | Sonic verification requires AWS contact. Actual provider fixtures/verifiers remain unqualified. Transcripts are not marked by Sonic. |
| Adobe / Firefly | [Image credentials](https://helpx.adobe.com/firefly/web/get-started/learn-the-basics/content-credentials-overview.html); [Translate Audio/Video credentials](https://helpx.adobe.com/firefly/web/work-with-audio-and-video/work-with-audio/about-translate-audio-and-translate-video.html) | The audio/video workflow describes AI voices, translation and lip edits. This does not establish whole-file synthesis or every partner product. |
| Stability AI | [Current hosted image/video/audio C2PA](https://kb.stability.ai/knowledge-base/content-credentials-c2pa-faq); legacy SDXL reference marker | Current hosted FAQ says C2PA rather than invisible marks. Legacy pipelines and self-hosted open weights have different behavior. |
| Black Forest Labs / FLUX | [Service credential terms](https://bfl.ai/legal/developer-terms-of-service) and selected Azure-hosted FLUX provenance | Does not establish every checkpoint/export. |
| Bria | [Image/video credentials and invisible watermark checks](https://docs.bria.ai/visual-verification-overview) | Public image/video verification endpoints require no API token. GIF/WebP image watermark checks are omitted, not negative. |
| Midjourney | [Image/video verifier](https://www.midjourney.com/verify) and [metadata-ID help](https://docs.midjourney.com/hc/en-us/articles/48207461163661-Content-Authenticity) | Help says copied tags fail; the verifier warns marks can be copied/fabricated. No automatic shortcut until this discrepancy and binding are resolved. |
| ElevenLabs | [Own-audio watermark detector](https://elevenlabs.io/docs/eleven-creative/audio-tools/audio-detector) | A positive may be a watermark OR fallback ML classification. Those outcomes must be distinguished. |
| Resemble / Chatterbox | [Public Perth audio code](https://github.com/resemble-ai/Perth); [versioned image/video/audio verification API](https://docs.resemble.ai/detect/watermark/detect) | Public v1 is not universal v2/SynthID compatibility. User-applied marks alone do not prove AI generation. |
| Suno | [C2PA on new song downloads and free verification API](https://suno.com/suno-credentials) | No token in documented examples; old downloads are not retroactively marked. |
| MiniMax / Hailuo | [Official image CLI watermark option](https://github.com/MiniMax-AI/cli/blob/main/src/commands/image/generate.ts); [MiniMax Code visible/implicit labeling rules](https://code.minimax.io/doc/en/download-watermark-rules.html) | Neither source establishes a universal hidden decoder across all MiniMax products/modalities. |
| DeepSeek | [Platform labeling terms](https://cdn.deepseek.com/policies/zh-CN/deepseek-terms-of-use.html) | No compatible public production key or copied-text whitespace decoder established. |
| Alibaba / Qwen / Wan | [Optional image notice](https://help.aliyun.cn/zh/model-studio/text-to-image-v2-api-reference); [video workflow](https://www.alibabacloud.com/help/en/model-studio/video-generate-edit-model/) | Visible image mark defaults off in the reviewed API; does not establish hidden/text/audio coverage. |
| ByteDance / Doubao / Seedance / Seedream | [Video watermark option](https://docs.volcengine.com/docs/lake-ai-service/video-generation-enhanced-edition-basic-edition?lang=en), [speech metadata/rhythm options](https://docs.volcengine.com/docs/doubao-voice/podcast-api-websocket-v3protocol?lang=en), [structured media labels](https://docs.volcengine.com/docs/intelligent-processing/add-aigc-metadata-guide?lang=zh) | Speech marking defaults off; producer/IDs are caller-configurable. A separate MediaKit custom-mark extractor is not a universal Seedance verifier. |
| Tencent / Hunyuan | [Video AI notice controls](https://cloud.tencent.com/document/api/1616/119001); [customer-applied AIGC metadata queries](https://cloud.tencent.com/document/product/460/122025) | Storage request authentication does not authenticate labels. Cloud Infinite capabilities do not establish default Hunyuan/Yuanbao marking. |
| Baidu / ERNIE | [Image frequency-domain mark and payload](https://ai.baidu.com/ai-doc/NLP/Ml9i5amtk) | Official extraction is not publicly open; Baidu support is needed. |
| Kuaishou / Kling | [Invisible image/video marks and detection portal](https://kling.ai/docs/ai-content-detection) | No compatible local public decoder or tested API contract established. |
| Synthesia | [Automatic C2PA and separate visible video notices](https://help.synthesia.io/en/articles/17634791-how-do-i-add-an-ai-label-to-my-videos) | Editor, API and dubbed visible-label behavior differs; actual credentials still need validation. |
| HeyGen | [Default/removable video logos](https://help.heygen.com/en/articles/11057301-how-to-remove-the-heygen-watermark) | Paid workflow may omit the logo. |
| Runway | [Free-plan video watermark](https://help.runwayml.com/hc/en-us/articles/50404627334547-Free-plan-details) | Paid exports may be unmarked. |
| Pika | [Current support says exports have no watermark](https://pika.art/support) | Older FAQ differs; do not assume marks survive across plans/dates. |
| Moonshot / Kimi | [Explicit notices and implicit metadata rules](https://www.kimi.com/user/agreement/modelUse?version=v2) | Platform terms are not a public decoder/key or proof of all self-hosted output. |
| NVIDIA | [Hosted Cosmos3 Nano visual SynthID](https://build.nvidia.com/nvidia/cosmos3-nano/modelcard) | Named hosted image/video generation; no automatic audio, text, numeric-action or downloadable-checkpoint coverage. |
| Kakao | [SynthID Detector partner listing](https://deepmind.google/models/synthid/) | Exact models/modalities are not established by the listing alone. |
| Apple | [Version-limited image SynthID documentation](https://support.apple.com/en-us/149520) | Named tools from OS 27.2; rollout and actual verification were not tested. Google's overview still says “soon”; do not infer other modalities. |

The broader follow-up adds **Cartesia, Inworld, Fish Audio, Hume, PlayAI/PlayHT,
Lightricks/LTX, D-ID, WRITER/Palmyra, Perplexity, IBM/Granite, AI21/Jamba,
StepFun and Baichuan**. It finds a concrete Cartesia watermark API, distinguishes
Inworld's classifier from watermark proof, fills Zhipu's hosted image/video
marking-documentation gap and records MiniMax's optional audible rhythm notice.
Luma's visible marks are plan-dependent. Mistral Vibe Code and WRITER describe
future December 2 commitments; these are not current working verifiers.

The [latest follow-up](provider-signal-expansion-2026-11-08.md) adds Samsung,
SenseTime, iFLYTEK, Murf, Canva, Leonardo and Descript. It also records Google's
scoped hosted credentials and Genie preview, Qwen audio tags, Muse's linked
portal, Hailuo and LTX export marks. Google's separate AI Content Detection API
is a statistical image classifier, not a watermark verifier or early-stop route.

Focused primary-source searches still leave **Cohere, Udio, Ideogram, Recraft,
Fish Audio, Hume, PlayAI/PlayHT, Perplexity, IBM/Granite, AI21/Jamba, Canva,
Leonardo, Descript, PixVerse, Freepik/Magnific, WellSaid, Voicemod,
OpenMOSS and Mureka** as review gaps. Queries and all four explicit modality states remain
in the catalog. This does not mean those providers never mark content.
Customer discussions, transparency pledges and a shared watermark algorithm
are insufficient to fill a missing production-verification specification.

The [broader follow-up](provider-signal-broad-search-2026-11-08.md) adds the
CapCut-family public media verifier, scoped Cosmos visual marking, Vidu labels,
Speechify Studio marking, conditional Higgsfield policy and Krea partner-model
credentials. It preserves the difference between a documented route and an
integration qualified on actual files.

## What the app collects, and what it does not infer

The [structured-label reader](structured-source-labels.md) recognizes supported
AIGC fields, locators and checksums. Known-name normalization now covers the
82 families (190 exact aliases), including `volcengine`, `seedream`, Deepgram and Sarvam. Recognizing a known string
does not show the provider actually uses that field, authenticate it, decode a
hidden watermark or add confidence. Unknown names/codes and private identifiers
remain withheld. Version 1 historical and version 2 current records render in
the normal view and reports; both retain source binding.

Public implementations exist and can be run without provider accounts. Our
[actual AudioSeal evaluation](public-audio-watermark-v1-result.md) ran locally,
including retained false responses and misses. It is not yet a serving detector
or evidence of Muse compatibility. The separate [fresh speech/background study](public-audioseal-speech-v2-result.md)
completed576 cases with retained false responses and strong-noise misses; the
fixed public-family check remains outside serving, confidence and early stopping.
The separate [visual native contract](public-visual-watermark-v1-result.md) also
completed72 image and96 frame cases with exact mask/source-time replay.
Compression-related misses and false responses remain; real-media/provider
qualification and app admission are still open. The subsequent
[natural-media study](public-visual-natural-v2-result.md) completed 640 cases
with retained control flags and compressed timing/message failures; it remains
outside serving, confidence and the early-stop gate.
Matching key/payload, scheme, generation
configuration, tokenizer and license remain necessary. Public detection-key
schemes exist; “every detector needs a secret provider key” would be incorrect.

For future external integrations, preserve exact file checksums, route/version,
applicable checks, provider response, outcome and scope. Admit only authentic
responses tied to the submitted bytes, tested with real positive and negative
fixtures, and with clear errors/unavailable states and customer data controls.
No uploaded JSON, provider-name string or undifferentiated “positive” may trigger
the shortcut. The catalog itself never becomes per-file evidence.

## Local qualification and next integrations

Actual local SDK-signed PNG, MP4, WAV, M4A, MP3 and FLAC fixtures exercise the
shipping adapters. Thirty full-pipeline cases cover image/video and both audio
modes across quick/full/deep profiles, with every model lease forbidden.
Changed MP3/FLAC audio under retained credentials, changed image pixels and
stripped credentials cannot activate the gate. These are locally authored
declarations under a test authority, not genuine provider fixtures or a detector
accuracy benchmark. The test authority is never added to production trust.

Next priority is qualifying real-provider fixtures for Suno/Bria/Cartesia and the
available public watermark schemes, then integrating supported routes with
clear scope and data handling. Restricted production text verification, Muse
compatibility and the twenty-eight review gaps remain open. No new weights,
statistical calibration, public confidence threshold or GCP deployment was
introduced by this inventory.

The [LTX public marking contract](ltx-marking-audio-v1-result.md) adds two
scoped audio/video records. At that stage, the inventory had 67 families/98 records;
the initial [follow-up search](provider-signal-followup-2026-11-08.md) expanded it
to 71 families/101 records. The additional speech/mobile/regional sweep now
records 82 families/105 records, including Deepgram's public speech checker,
Sarvam's video-only overlay, vivo notices and Tencent metadata-query scopes.
Its audio code runs locally with exact compatible cached weights; generic
presence, public-payload recovery and origin authentication remain separate.
Noise and partial-mark misses are retained. Video implementation is inspected,
not executed. Neither path is admitted to serving or the fast gate.
