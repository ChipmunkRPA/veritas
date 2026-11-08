# Provider signals: expanded review and fast-path limits

Reviewed 8 November 2026. The [catalog](provider-signals-catalog.json) now contains
**57 provider families and 83 scoped records**, with an explicit image, video,
audio and writing state for every family. **Thirteen families still have no
established scoped marking specification.** These are documentation counts,
not working-detector counts or an exhaustive census. Earlier research,
historical products, planned features, classifiers and production claims remain
distinguishable in the [matrix](provider-signal-matrix.md).

This follow-up searched first-party technical documentation, product help,
source repositories and provider policies. The catalog retains exact focused
queries, source URLs, product/version limits, generation controls, verification
routes and runtime status. Eight consequential source pages were also fetched
directly and hashed after browser retrieval limits/errors; their receipt is in
the catalog. Full provider documents remain local rather than being republished.

## Findings that extend or correct the earlier inventory

| Provider / route | Established by the retrieved primary source | What remains unestablished |
|---|---|---|
| Apple Intelligence | [Apple's support page](https://support.apple.com/en-us/149520) names SynthID on selected image tools from OS version 27.2. Earlier images need re-editing by a named tool to acquire that mark. | Actual rollout and genuine-file verification were not tested. Google's overview still says Apple support is coming soon. Keep this discrepancy and the software-version condition; do not infer audio, video or writing coverage. |
| Google hosted credentials | [The technical model list](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/content-credentials) describes Google LLC signatures for selected Gemini/Omni image and Veo outputs. | A model list and an issuer name do not validate an uploaded asset. Audio/document packaging is not inferred from the page's general introduction or Lyria entries. Genuine provider fixtures remain to qualify. |
| Google Genie 4 | [The product page](https://deepmind.google/models/genie/) states generated video and audio will carry SynthID in its trusted-tester preview. | This does not release a local decoder or establish which exports the portal successfully verifies. |
| Google AI Content Detection API | [The technical contract](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/ai-content-detection?hl=en) describes an image classifier in private preview, operating on pixel/noise/spectral patterns; it excludes C2PA. | This is distinct from the SynthID portal. Its location in a transparency FAQ does not turn its statistical score into watermark recovery or a documentary shortcut. |
| Meta / Muse | [The Muse launch page](https://ai.meta.com/blog/introducing-muse-image-muse-video-msl/) links the [identification portal](https://www.meta.ai/identification/). Muse Image marking is documented; the same page describes video extension as planned. | The portal's image/video/audio upload UI does not prove all three modalities are verified. No file was submitted. Released WAM/VideoSeal/AudioSeal research weights are not automatically compatible with Muse production. |
| Alibaba / Qwen audio | [The reviewed SDK](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-tts-python-sdk) documents optional AIGC tags in WAV/MP3/Opus, disabled by default, plus caller-configurable propagation fields. | No exact serializer, cryptographic authentication or robust hidden-waveform reader was established. A configurable label is not provider proof. |
| iFLYTEK / Spark audio | [The concrete TTS API](https://static.xfyun.cn/doc/spark/super%20smart-tts.html) describes optional MP3 AIGC metadata and a separate beginning/end perceptible notice. Both default off. | Its printed metadata example is ambiguous/malformed and does not identify an ID3 frame/encoding. Numeric producer codes remain unmapped and redacted. Current policy text did not reproduce all indexed excerpts, so those excerpts do not fill other modalities. |
| SenseTime / SenseNova | [The provider announcement](https://www.sensetime.com/hk/news/51169933) explicitly lists text, image, audio and video notices/implicit file metadata. A [specific image API example](https://www.sensetime.com/cn/news/sensenova-u1-5-lite-token-plan-20260911-1741) permits disabling its export mark. | Broad policy and a particular API are different scopes. No authenticated hidden-marker verifier or exact serializer was established; disabling one export mark does not prove other metadata is absent. |
| MiniMax / Hailuo | [Video subscription terms](https://hailuoai.video/doc/payment-policy.html) distinguish watermarked free downloads from paid watermark-free downloads. | No compatible hidden video decoder was established. Absence of the export mark cannot exclude Hailuo. Earlier image and audible-speech records remain separately scoped. |
| Lightricks / LTX | [LTX Studio help](https://help.ltx.io/hc/en-us/p/ltx-studio) likewise describes marks on free exports and unmarked paid exports. | This does not establish open-weight, Desktop or API marking behavior, nor a hidden decoder. The earlier total documentation gap is narrowed to a verification gap. |
| Samsung / Galaxy AI | [Photo assist help](https://www.samsung.com/us/support/answer/ANS10000934/) documents an AI notice on named image-editing/generation workflows. | Device/software scope matters; a marked partial edit does not make the entire photograph synthetic. Copied or cropped symbols cannot authenticate origin. |
| Murf dubbing | [The API limits table](https://murf.ai/api/docs/capabilities/dubbing) lists a mark on free dubbing exports and none on paid tiers. | Mechanism/placement, a standalone speech watermark and a detection endpoint are unspecified. The inventory calls this an export mark. |

OpenAI, Anthropic and Grok were rechecked against their current official API
documentation. Their previous distinctions remain: supported image/audio
verification differs from restricted text access; keyed token-choice marking
does not rely on hidden spaces; live response headers are not signatures on a
copied passage. The existing inventory also retains Microsoft, Amazon, Adobe,
Stability, FLUX, Bria, Suno, ElevenLabs, Resemble, DeepSeek, ByteDance, Tencent,
Baidu, Kling, Zhipu, Moonshot and the other reviewed providers.

## Explicit gaps and excluded evidence

Focused queries still did not establish scoped production marking specifications
for Cohere, Udio, Ideogram, Recraft, Fish Audio, Hume, PlayAI/PlayHT, Perplexity,
IBM/Granite and AI21/Jamba. Canva, Leonardo and Descript are now explicit rows
with four unestablished modality states rather than implied coverage. These are
search gaps, not evidence that their outputs contain no marks.

Canva's [developer terms](https://www.canva.com/policies/api-services-sch1/)
require preserving applied provenance, but do not specify every export or a
verifier. Stock-photo license watermarks, website branding, user gallery prompts,
feature requests and general educational articles were excluded. IBM's model
checkpoint signatures authenticate model downloads, not generated articles.
Common corporate ownership also does not establish identical marking behavior.

## What can save analysis work now

The prepaid pipeline already checks supported Content Credentials before
acquiring statistical models. Only trusted, valid, exact-file-bound credentials
that explicitly declare AI involvement, with no validation failures or unresolved
remote reference, can stop those checks. The result explains the declaration,
the skipped checks and its limits; it does not invent a statistical probability,
every edit location, or an original-file verdict.

Unsigned metadata, visible/audible notices, public fixed-message hits and
provider classifier scores cannot activate that shortcut. A negative,
unsupported, unavailable or inconclusive signal continues applicable analysis.
An audio mark in a video only addresses the sound, and a marked passage only
addresses that passage. No current collection promises universal provider
recognition or whole-file authorship.

Public compatible detectors can run without a provider account. Public-key
verification schemes also exist. Matching production configuration and evidence
still matter: the [fresh local visual study](public-visual-natural-v2-result.md)
found control flags, degraded message recovery and compressed responses outside
the marked interval, despite flagging all intentionally marked cases. It remains
outside serving/confidence/early stopping.

Next qualification should prioritize genuine positive and negative fixtures for
Suno, Bria and Cartesia, followed by accessible OpenAI/text/provider verification
routes and exact Muse compatibility. A future external fast path needs qualified
responses bound to the submitted file, scoped outcomes, documented errors and
customer data controls. No customer uploads or verification requests were made
in this follow-up; external customer routes remain disabled. Reading Google
documentation did not create resources or deploy the app to GCP.

## Local validation

All **267 selected backend tests passed**, covering the inventory, actual
structured media/document containers, source binding, report disclosures and
the signed-file fast path. The full-pipeline fixtures forbid model acquisition.
Static checks, matrix consistency and whitespace checks also pass.

The bounded structured-label reader now normalizes 134 exact aliases for all 57
catalog families, including the seven added families. This does not establish
that each provider uses the supported field layout. Existing per-alias tests
verify that copied claims remain unauthenticated, outside confidence and unable
to activate the gate in any modality. Private numeric identifiers stay redacted.
Inventory checks additionally distinguish classifiers, version conditions,
future features, generic policies and unspecified export marks. No serving
weights, speech calibration, public threshold or USD9.99 prepayment rule changed.
