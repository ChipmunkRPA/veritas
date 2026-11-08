# Provider signals: broader search and verification priorities

Initial broader review, 8 November 2026. This sweep recorded
**67 provider families and 96 scoped signal records**, with explicit image,
video, audio and writing states for each. **Nineteen families remain search
gaps**. This is a curated, sourced inventory, not a guarantee to recognize every
provider, generation date, model, export or surviving mark. Policies, planned
features and public research are counted as records, not working detectors.
The [follow-up](provider-signal-followup-2026-11-08.md) now expands the catalog
to 71 families/101 records, including the public LTX contract. The current
[matrix](provider-signal-matrix.md) shows the full scope; each source row
keeps its actual review date and focused search queries.

This review searched provider technical documentation, help pages, source
repositories and product policies. It adds ten families: Higgsfield, Vidu,
Speechify, Krea, PixVerse, Freepik/Magnific, WellSaid, Voicemod, OpenMOSS/MOSI.AI
and Mureka. Only four of those have a scoped marking record established here.
CapCut, Dreamina, Pippit and Hypic are additional product scopes within the
existing ByteDance family, not independent new detector counts.

## Findings that affect the fast-path queue

| Provider / product | Primary-source finding | Implementation consequence |
|---|---|---|
| CapCut family | The [public preview](https://ai-generated-detect.capcut.com/) accepts images, video and audio. [Dreamina's safety guide](https://dreamina.capcut.com/clause/dreamina-user-safety-guide) describes visible marks, invisible marks and provenance metadata. | A concrete additional provider verification route, separate from the previously recorded Volcengine custom-watermark API. No supported automation API or genuine response qualification was established. |
| NVIDIA Cosmos3 Nano | The [hosted model card](https://build.nvidia.com/nvidia/cosmos3-nano/modelcard) and [system card](https://build.nvidia.com/nvidia/cosmos3-nano/systemcard) specify mandatory SynthID on returned images and videos. Numeric action arrays are not visual media marks. | Replaces a partner-listing-only gap for these hosted visual outputs. Do not infer marking in PersonaPlex audio, generated text, every downloadable checkpoint or every deployment. |
| Amazon Nova Reel 1.1 | The [service card](https://docs.aws.amazon.com/ai/responsible-ai/nova-reel/overview.html) explicitly describes video Content Credentials, alongside the invisible watermark. | Adds a version-scoped local credential candidate. A genuine sample must still pass trust, binding and AI-declaration validation. The general product statement is not file evidence. |
| Vidu Q3 | The [reference-to-video API](https://platform.vidu.com/docs/api-reference/video-models/vidu-q3/reference-to-video) documents optional visible marks and customizable `meta_data`, with default Vidu metadata when that field is empty. | The existing structured-label reader can inspect compatible fields as unverified claims. Actual provider bytes/serialization remain unqualified. A callback signature authenticates the callback, not an arbitrary uploaded video. |
| Speechify Studio | The [October 9 Studio terms](https://speechify.com/studio-terms/) declare an imperceptible machine-readable audio mark. | Adds an audio marking claim, with verifier/key/format/legacy scope still unresolved. Studio terms distinguish reader products and API voice cloning; do not extend the claim to those. |
| Higgsfield | [Help](https://higgsfield.ai/creator-hub/help-center/credits/watermark-and-how-to-remove) distinguishes visible free-account branding from conditional invisible marks. [Terms §6.4](https://higgsfield.ai/terms-of-use-agreement) allow secure metadata, watermarks or C2PA, without promising them on every output. | Visible branding is advisory. The conditional policy does not establish a hidden decoder or fill every modality with operational coverage. Paid logo-free downloads can still have provenance marks. |
| Krea / Gemini Omni Flash | The [named model page](https://www.krea.ai/models/gemini-omni-flash) documents SynthID and C2PA on its video outputs. | Records this partner-model scope without attributing every Krea model or export to the same mechanism. Actual exported credentials were not tested here. |
| ElevenLabs | The [current Audio Detector guide](https://elevenlabs.io/docs/eleven-creative/audio-tools/audio-detector) describes watermark-first detection followed by a legacy classifier. [Third-party outputs](https://elevenlabs.io/docs/help-center/legal/audio-detector/is-content-generated-by-third-party-models-available-through-eleven-labs-watermarked) are generally excluded from its watermarking. | Keep the branches separate. A combined positive result alone does not establish that a watermark was recovered. The inventory now records the fallback classifier separately. |

The directly retrieved CapCut portal lists JPEG/PNG/JPG/WebP, MP4/MOV/WebM/AVI
and WAV/MP3/FLAC/OGG/OPUS/M4A. It says **20 MB per image and 100 MB per video or
audio**, and that uploaded content is not stored. Exact byte enforcement,
processing location, detailed privacy terms and result semantics were not
independently verified. Its heading names CapCut, Dreamina, Pippit and Hypic;
the discovery index also mentions TikTok. That difference remains explicit.
No file was uploaded, and the portal does not establish universal Seedance,
TikTok-upload or ByteDance API coverage.

Vidu's `watermark` defaults off; `wm_position` and `wm_url` control the overlay.
Its metadata example includes label, producer, propagation, identifiers and
reserved fields that callers can replace. Neither that example nor an embedded
provider name authenticates authorship. Private identifiers remain redacted in
the app's extracted records.

## Named major providers rechecked

- **OpenAI:** [Official API documentation](https://developers.openai.com/api/docs/guides/content-provenance)
  describes supported image C2PA and image/audio SynthID checks, with separate
  results. Text verification is application-based. This is not a universal
  video or other-provider verifier. Omitted checks are not negative checks.
- **Google / Gemini:** [SynthID documentation](https://deepmind.google/models/synthid/)
  covers supported media and token-choice text marking. Its public media portal
  names Google and partners, but a partnership does not establish every
  product/modality. Published text reference code needs compatible generation
  configuration; a demonstration key is not a Gemini production key.
- **Anthropic / Claude:** The [canonical explanation](https://www.anthropic.com/news/claude-text-watermark)
  describes keyed token choices and private-preview verification. It explicitly
  excludes hidden characters. Short passages, proofreading and constrained
  answers can supply little detectable signal. Supported file credentials are
  separate from text watermarking.
- **Meta / Muse:** [Content Seal](https://facebookresearch.github.io/content-seal/)
  expressly distinguishes Muse Image's proprietary production implementation
  from its released research models. Public compatible detectors can run
  locally; compatibility with production Muse must be demonstrated, not assumed.
- **xAI / Grok:** [Current API documentation](https://docs.x.ai/developers/advanced-api-usage/watermarking)
  specifies model/team/date scope and application-based detection, including an
  uncertain result. Token marking adds no hidden text. A live response header
  does not authenticate a copied passage.
- **DeepSeek, MiniMax/Hailuo, Qwen/Wan, Doubao/Seedance/Seedream, Hunyuan,
  ERNIE, Kling, GLM, Kimi, StepFun, Baichuan, SenseNova and iFLYTEK:** retain
  their individually sourced catalog records. Visible notices, optional rhythm
  notices, unsigned AIGC fields, a generic labeling policy and a private verifier
  are different mechanisms. No universal Chinese-provider whitespace signature
  or shared production decoder was established.

The catalog also retains Adobe, Microsoft, Amazon, Stability, FLUX, Bria,
Midjourney, Suno, Resemble, Cartesia, Inworld, Runway, Luma, Pika, Synthesia,
HeyGen, D-ID, Apple, Samsung, Kakao and the remaining reviewed families. The
four-modality matrix is the complete family list; this paragraph is not a
second, narrower coverage claim.

One useful source conflict remains unresolved: a Microsoft search result
describes GPT text-watermark request controls, but opening the
[specific URL](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-safety/enable-text-watermarking-gpt-models)
redirected to the general provenance page. The retrieved page does not establish
the indexed control's current deployment/version availability. The catalog
retains that discrepancy rather than treating the snippet as operational proof.

## Gaps, exclusions and what actually saves work

No compatible scoped specification was established here for **Cohere, Udio,
Ideogram, Recraft, Fish Audio, Hume, PlayAI/PlayHT, Perplexity, IBM/Granite,
AI21/Jamba, Canva, Leonardo, Descript, PixVerse, Freepik/Magnific, WellSaid,
Voicemod, OpenMOSS or Mureka**. That is a search result, not proof of absent
marks. User-created voices and soundboard clips named “Watermark”, stock-photo
license marks, feature requests, model-checkpoint signatures, branding in a web
interface and generic preservation policies were excluded as file-authorship
evidence. A video's “watermark-free” marketing description does not establish
that its invisible marks or credentials are absent.

The implemented order remains:

1. Check supported local credentials before acquiring statistical models.
2. Stop those checks only for a trusted, valid, current-file-bound signed record
   that explicitly declares AI involvement, without validation failures or an
   unresolved remote reference. Explain the declaration and skipped checks.
3. Preserve unsigned labels and public marking observations as supporting
   evidence. They cannot activate the fast path or add AI-origin confidence.
4. When evidence is absent, stripped, unsupported, unavailable, failed or
   inconclusive, continue applicable analysis. Never convert that state into
   an original-file verdict.

For external qualification, first obtain authentic marked/unmarked fixtures and
separate applicable outcomes for **Suno, Bria, Cartesia and the new CapCut
route**. Then qualify accessible Google/OpenAI/Microsoft and text routes with
their exact access, data handling and response contracts. Muse production
compatibility remains a separate task. Qualification must bind the response to
the exact submitted file, preserve errors and unsupported states, distinguish
watermarks from classifier fallbacks, and specify whether it covers visuals,
audio, a passage or only part of a composite. A detected audio mark in a video
does not establish that its pictures are synthetic.

This review made **zero provider-verification requests and zero customer
uploads**. All external customer verification routes remain disabled. It did
not create cloud resources, change statistical calibration or alter prepaid
USD9.99 analysis. The expanded known-name normalization is only for supported
structured claims; it does not imply that each provider emits that layout.

## Local validation

All **262 selected backend tests passed**, covering the catalog/matrix,
authored structured media and document containers, source binding,
private-identifier redaction, API/JSON/report disclosures and the signed-file
fast path before model acquisition. All **155 exact aliases across 67 families**
remain unverified claims; each alias is checked against all four modality gate
decisions. Catalog/matrix consistency, selected static checks and whitespace
checks pass. Statistical weights and the local speech calibration remain
unchanged. These are implementation-contract tests, not a provider-watermark
accuracy benchmark or qualification of any newly discovered external route.
