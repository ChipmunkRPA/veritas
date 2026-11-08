# Broader provider-signal review

Reviewed 8 November 2026. The [catalog](provider-signals-catalog.json) expands
from 37 to **50 provider families**, with **69 scoped signal records** and a
[four-modality matrix](provider-signal-matrix.md). Thirty-nine families have
scoped documentation, including announcements or policies whose export scope
is unresolved; eleven remain verification gaps. These are research counts,
not 50 working detectors or a claim to cover every provider/export.
Each family retains its own review date; this supplement does not silently
re-date the earlier source reviews.

## New findings that change the implementation queue

| Family | Primary-source finding | Implication |
|---|---|---|
| Cartesia | [Watermarking documentation](https://docs.cartesia.ai/build-with-cartesia/capability-guides/watermarking) describes marks on supported Sonic models, a public portal and a paid-plan detection API. | A concrete new verification candidate. The documented response is a statistical watermark detection, with possible false positives. It needs genuine marked/unmarked fixtures and response qualification before admission. |
| Inworld | The [content-detection API](https://docs.inworld.ai/api-reference/detectionAPI/contentdetection/detect-content) scores supported TTS speech, including sound tracks from video containers. | It returns a classifier score; it does not separately expose authenticated watermark recovery. Other synthetic speech can score high. Missing `probability` means unscored, not human. Never use this score as a documentary shortcut or a visual-video verdict. |
| Zhipu / Z.ai | The official [image API](https://docs.bigmodel.cn/api-reference/模型-api/图像生成.md) and [video API](https://docs.bigmodel.cn/api-reference/模型-api/视频生成异步.md) document default visible/implicit marks through `watermark_enabled`. | A previous documentation gap is filled. No compatible public production decoder/key/verifier was established. The video field reuses image wording; do not infer separate audio marking or self-hosted checkpoint behavior. |
| MiniMax | The reviewed [T2A reference](https://platform.minimaxi.com/document/T2A%20V2?key=66719005a427f0c8a5701643) and [official speech console](https://solutions.minimax.cn/debug/speech) describe an optional rhythm notice appended to non-streaming audio. | Record it as an audible label, not a recovered hidden signature. The option defaults off. The response's input-character statistic is unrelated to watermark verification. |
| Luma | [Dream Machine plan documentation](https://lumalabs.ai/learning-hub/payments-subscriptions) describes plan-dependent export watermarks. | A visible-label source is now recorded. It does not establish a hidden Ray/Photon verifier, every partner model, or marks on paid unwatermarked exports. |
| D-ID | [Current help](https://help.d-id.com/hc/en-us/articles/31350550554641-How-do-I-change-or-remove-the-watermark-on-my-videos) describes avatar-video notices and plan-dependent replacement/removal. | The [FAQ](https://www.d-id.com/faqs/) differs on removal. Preserve this conflict; a visible logo is advisory evidence. |
| WRITER | The [October update](https://writer.com/blog/oct-roundup-new-at-writer/) announces customer-keyed, configurable Palmyra text marking and detection, available by December 2. | Record as planned as of this review. Neither customer keys nor a working current production detector were established. Third-party models in WRITER have their own policies. |
| Mistral | The [Vibe Code page](https://legal.mistral.ai/ai-governance/ai-systems/vibe-code/) promises output marking and detection by December 2. | A future delivery commitment cannot become current detector coverage. Scope is the named product. |
| StepFun | [Platform terms, section 2.2.4](https://platform.stepfun.com/legal/user-agreement.html) declare visible notices and implicit file metadata. | Record the policy, keeping format/modalities unresolved. No public serializer, decoder/key or authenticated verification contract was established. |
| Baichuan | [Baixiaoyi terms](https://policy.baichuan-ai.com/cooperation/agreement) describe possible implicit output labels with generation attributes, producer details and identifiers. | This is a scoped product policy, not proof of deployment across every format/model. Unsigned fields remain unverified claims. |
| Adobe / Firefly | The [Translate Audio/Video FAQ](https://helpx.adobe.com/firefly/web/work-with-audio-and-video/work-with-audio/about-translate-audio-and-translate-video.html) explicitly describes credentials for generated speaker voices, translation and changed lip movements. | Expands the earlier image-only documentation scope to these audio/video workflows. A signed AI-edit declaration still does not mean the entire recording was generated. |

The Cartesia search index reported a 3-second minimum, while the directly fetched
documentation and portal specify **5 seconds to 5 minutes, up to 32 MB**. The
stricter retrieved limit is recorded; no endpoint behavior was tested. The API
example uses `Cartesia-Version: 2026-08-14` and a `detected` boolean. Its portal
states server processing worldwide and deletion shortly after use; an exact
retention interval was not established.

The Inworld API documents 25 MiB / five minutes, Basic authentication and ordered
multipart `config` then `file`. It names TTS-2 / TTS-2 Flash and explicitly warns
that noise, music, mixing, narrowband sound and editing reduce scores. The
provider says it does not store/log the file or filename. These are documented
terms, not an independent privacy or accuracy audit.

## Rechecked named providers and public implementations

- [OpenAI's Content Provenance API](https://developers.openai.com/api/docs/guides/content-provenance)
  documents image credentials and image/audio watermark results. Applicable
  results are separate; absent checks are not negative checks. Text access is
  application-based. This endpoint does not establish a universal Sora-video
  or other-provider verifier.
  Separately, the [Sora system card](https://deploymentsafety.openai.com/sora-2/provenance-and-transparency-initiatives)
  documents video credentials, moving visible marks and internal tracing tools.
  These are recorded as historical export signals: the
  [current Sora safety page](https://openai.com/index/creating-with-sora-safely/)
  reports that the product became unavailable on April 26. Retained files can
  still contain credentials; internal search is not a released local decoder.
- [Google SynthID](https://deepmind.google/models/synthid/) documents image,
  video, audio and token-choice text marks, plus a public media verification
  portal and named partners. A partner listing does not specify every model,
  modality or export. Public text code does not supply every production key.
- [Anthropic's current canonical article](https://www.anthropic.com/news/claude-text-watermark)
  describes keyed token-choice marking, private-preview detection access,
  supported-file credentials and legacy-model rollout. It explicitly excludes
  hidden characters. The canonical page now exposes the credential section
  previously visible in a different retrieved snapshot.
- [Grok's current API documentation](https://docs.x.ai/developers/advanced-api-usage/watermarking)
  gives a November 7 start for listed text models, team/model controls and
  application-based detection with an uncertain state. The response header is
  an indicator on a live provider response, not a signature on a copied passage.
- [Meta Content Seal](https://facebookresearch.github.io/content-seal/)
  distinguishes custom Muse Image production marking from public research
  implementations. Public AudioSeal, WAM, VideoSeal/PixelSeal and TextSeal /
  TextSeal2 are useful compatible-family candidates, not automatic evidence of
  Muse compatibility. The existing authored image/video/audio studies retain
  their false responses/misses and remain outside serving/confidence.

The current official FLUX reference, CompVis SD1 script and diffusers SDXL
watermarker were also read directly through GitHub. Parsing their constants
without executing external code confirmed all three existing local message
readers: **46, 136 and 48 bits**, respectively. The catalog now includes the
previously omitted FLUX and SD1 reference records, with source blob hashes.
FLUX's integer-to-bits expression drops leading zeros despite its source comment
calling the message 48-bit. Public fixed-message hits remain copyable marking
observations, outside AI confidence and early stopping. This source parity
check is separate from a file-level accuracy benchmark.

The existing catalog also retains scoped sources for Microsoft/Azure,
Amazon/Nova, Adobe/Firefly, Stability, FLUX, Bria, Midjourney, ElevenLabs,
Resemble/Chatterbox, Suno, DeepSeek, Alibaba/Qwen/Wan, ByteDance/Doubao/Seedance,
Tencent/Hunyuan, Baidu/ERNIE, Kuaishou/Kling, Synthesia, HeyGen, Runway, Pika,
Moonshot/Kimi and SynthID partner listings. Generic requirements to preserve
labels do not establish compatible production keys or prove every export is marked.

## Explicit remaining search gaps

Fresh primary-domain searches did not establish a compatible production
verification route for **Cohere, Udio, Ideogram, Recraft, Fish Audio, Hume,
PlayAI/PlayHT, Lightricks/LTX, Perplexity, IBM/Granite or AI21/Jamba**. Exact
queries are attached to their catalog rows. This means the route is unestablished
by this review; it does not mean the output must be unmarked.

Excluded results include user-created Fish Audio voices named “Watermark”,
shared chatbot conversations, unrelated copyright-mark tools, general research
or educational articles, feature requests and terms merely forbidding removal.
For example, Perplexity's acceptable-use restriction is not a decoder contract.
No characters, whitespace patterns, generic logos, timestamps, filenames or
provider-name mentions authenticate an article/email/document.

## Gate behavior and concrete next qualification

The app already checks supported Content Credentials before acquiring models.
Only a trusted, valid, current-file-bound, explicitly AI-declaring record with
no validation failures or unresolved remote reference can stop statistical
processing. Such a result reports the declaration and skipped checks, without
inventing an AI probability or locating every edit. Missing/inconclusive signals
continue analysis and cannot establish originality.

The expanded exact-name normalization recognizes the 50 catalog families only
inside supported structured-label fields. It remains outside the gate and
confidence. It does not mean those providers actually use our supported field
format. Existing source checksums, identifier redaction and unverified notices
remain in place. A stale source comment referring to the retired public-mark
confidence floor was corrected; no numeric policy was changed.

Prioritize qualified positive/negative fixtures for Suno, Bria and now Cartesia,
then production-key/configuration compatibility for supported public families
and accessible text verification. Each new route needs exact input binding,
an authentic response, explicit applicable/unsupported/unavailable/error states,
negative controls and declared customer data handling. Preserve its scope:
a watermark may identify provider involvement in part of a file; it does not
authenticate the entire file or every visual/audio/text component.

This supplement made **zero provider-verification calls and uploaded zero
customer files**. All external customer routes remain disabled. No generation
requests, paid API calls, model downloads, calibration changes or cloud
deployment were made by this review. Retrieval checksums for five directly
fetched reference documents are retained locally under
`output/provider-signal-review-2026-11-08/retrieval.json`; no full provider
documents or customer material are included in the public change.

## Local validation

- 279 selected tests passed across actual structured containers, source binding,
  signed-file shortcuts before model acquisition, report disclosures and the
  policy excluding copyable markers from AI confidence.
- The 50 normalization families contain 115 exact aliases. For each authored
  alias claim, all four modality gate decisions continue detection.
- The inventory checker rejects stale counts, duplicated families, a future
  feature presented as current, a generic policy presented as audio coverage,
  and an unqualified provider classifier promising an automatic shortcut.
- Matrix generation/check, the selected static checks and whitespace checks pass.
  Serving weights and the local speech calibration were unchanged.
