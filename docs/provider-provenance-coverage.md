# Provider provenance and the first detection gate

Reviewed 7 November 2026. Provider capabilities change; product-specific
documentation takes precedence over general announcements. A provider name,
logo, unusual spacing or an unsigned metadata field is not authenticated origin.

## Current local implementation

Image, video and supported audio pipelines now validate Content Credentials
before any statistical detector can acquire model weights. The gate requires
all of: a trusted signer, valid credentials, an explicit AI declaration, a hard
binding to the uploaded bytes, no validation failures and no unresolved remote
reference. It is provider-agnostic: it does not trust a company name in a field.
WAV and M4A have a new model-free audio credential adapter. Other audio
containers explicitly remain unassessed by that adapter.

A qualifying declaration stops subsequent detection, with a recorded reason.
The result describes AI involvement **declared by the signed record**, not the
origin of every pixel, sound or sentence. Statistical checks remain marked
skipped. Audio forms no voice confidence estimate on this path. A trusted
capture record does not trigger this AI-positive shortcut.

Absent, stripped, invalid, untrusted or unsupported credentials continue through
normal detection. They never clear a file as original. Offline verification uses
the bundled C2PA Trust List snapshot, without fetching remote manifests or
checking revocation. The report identifies those limits. Historical refusion
retains earlier detector evidence instead of erasing it or requesting new models.

The gate is inside the analysis pipeline; it does not bypass the USD 9.99
prepayment requirement. No uploaded file is sent to an outside provider by this
implementation. It does **not** decode every private watermark below.

## Named providers

| Provider | Documented marking and verification | What is available here |
|---|---|---|
| OpenAI | The official [Content Provenance API](https://developers.openai.com/api/docs/guides/content-provenance) checks supported image C2PA and image/audio SynthID signals. Text verification requires approved access. A negative result does not rule out OpenAI or other AI content. | Local signed credentials where present and supported. The external API is not configured or called; private text verification is not available here. |
| Anthropic / Claude | [Provider explanation](https://www.anthropic.com/news/claude-text-watermark) describes statistical word-choice watermarking, with no hidden characters, and a detection API in private preview. Supported generated files can carry C2PA notes indicating Claude involvement. | Local signed credentials where present. No claimed Claude text-key access or automatic private-preview verification. |
| Google / Gemini, Imagen, Veo and Lyria | [SynthID](https://deepmind.google/models/synthid/) covers provider-supported media; the public detector portal checks images, video and audio from Google and supported partners. [Gemini verification](https://support.google.com/gemini/answer/16722517?hl=en) describes the supported app workflow. | Generic local credentials where present. The provider portal is a separate verification workflow, not an integrated local decoder or an inferred permission to upload customer files. |
| Meta / Muse | Meta's [Muse Image and Muse Video announcement](https://ai.meta.com/blog/introducing-muse-image-muse-video-msl/) describes Content Seal for Muse Image and a preview verification tool. That announcement says video expansion is planned; it does not establish current video coverage. | Generic local credentials where present. Content Seal verification is not integrated. No universal Muse text/audio watermark claim. |
| xAI / Grok | [Output watermarking documentation](https://docs.x.ai/developers/advanced-api-usage/watermarking) describes supported API text marking and a detection API available by application. The [consumer FAQ](https://docs.x.ai/grok/faq) documents visible Grok marks on generated images/video. An API response's watermark header is not an authenticated property of a subsequently copied article. | Generic local credentials where present. A visible logo or copied header cannot stop detection. Access to the provider's text verification API is not configured here. |
| MiniMax / Hailuo | The official [image API](https://platform.minimax.cn/docs/api-reference/image-generation-t2i) documents optional `aigc_watermark`, default false. This does not establish a universal hidden watermark or independently available decoder across all products and formats. | Generic local credentials where present. No invented MiniMax audio/video/text decoder; optional or removed marks cannot establish originality. |
| DeepSeek | Its [terms, section 5.3](https://cdn.deepseek.com/policies/zh-CN/deepseek-terms-of-use.html) state that platform-generated content receives labels. They do not publish a general detector key or establish that all copied text has a detectable whitespace pattern. | Generic local credentials where present. Provider-specific private text verification is not implemented. |

Other major providers, including Alibaba/Qwen/Wan, ByteDance/Doubao/Seedance,
Tencent/Hunyuan, Baidu and Kuaishou/Kling, belong in the ongoing review. Do not
infer their output format or detector availability from another provider's
implementation. The [Chinese labeling rules](https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm)
describe explicit labels and implicit metadata. A labeling obligation does not
give this application a private verification key or make unsigned fields
tamper-proof. A bounded structured-metadata reader and provider-code review
remain required follow-up; this release does not claim those are implemented.

## Text watermark source review

The [official SynthID Text repository](https://github.com/google-deepmind/synthid-text)
and its [frequentist detector source](https://github.com/google-deepmind/synthid-text/blob/main/src/synthid_text/detector_frequentist.py)
were inspected. Its [code license](https://github.com/google-deepmind/synthid-text/blob/main/LICENSE)
is Apache-2.0. The detector uses the generation configuration/key to compute
masked statistics over tokens. Its null-test p-value is not an AI-authorship
probability. The reference implementation's demonstration configuration is not
Google's deployed Gemini key; the repository also warns against treating its
reference generation code as production-ready. Nothing was installed, imported
or presented as a universal decoder.

## Verification scope and remaining work

Authored PNG, WAV and MP4 records passed actual SDK verification and the shipping
adapters. Local full-pipeline tests reached published results while model leases
were forbidden. Stripped PNG metadata continued to detection, and changed pixels
with old credentials failed validation and continued. These use an explicitly
local test trust root, not impersonated provider credentials or a provider
accuracy benchmark. An SDK signing-profile issue concerning missing certificate
Organization fields was identified while constructing the test fixtures; see
the [upstream issue](https://github.com/contentauth/c2pa-rs/issues/2262).

External verification needs real API access, documented response semantics,
account permissions, privacy controls and end-to-end tests before activation.
Successful provider verification still needs its exact scope recorded. A mark
can indicate processing or editing without establishing full AI authorship.
No arbitrary uploaded JSON or caller-supplied `detected` flag may activate the
trusted gate. Do not mark unavailable verifiers as having checked a file.
