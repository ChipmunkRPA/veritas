# Provider-signal follow-up: executable routes and remaining gaps

Reviewed 8 November 2026. The source-linked [catalog](provider-signals-catalog.json)
and [four-modality matrix](provider-signal-matrix.md) now cover **71 provider
families and 101 signal records**. Twenty families remain explicit search gaps.
These records include public research, visible notices, policies and unconfirmed
announcements. They are not 101 working detectors or a promise of universal
provider coverage. Every family has a separate image, video, audio and writing
state, including unsupported or unresolved scope.

This follow-up rechecked the named major providers, searched 18 existing-family
gaps, and added Meitu, Kunlun/Tiangong/Skywork, Shutterstock and Getty/iStock.
Queries and primary references are retained per family. A domain search with no
matching result means this review did not establish a compatible signal; it
does not establish that a provider never marks its outputs.

## New findings and limits

| Source | Finding | Consequence for Chipmunk Verify |
|---|---|---|
| [OpenAI announcement](https://openai.com/index/eu-text-provenance/) and [technical report](https://cdn.openai.com/pdf/e9508624-d767-41b6-a26d-e34ca798ada6/textgrain-entropy-calibrated-watermarking-for-language-model-text.pdf) | textGrain uses keyed token choices and an entropy budget. The report describes verification using text and the secret key. Selected API generation is opt-in; eligible EU ChatGPT/Codex rollout is separate. | Publishing the method does not provide a production key. Text detector access remains application-based. Planned open-source release is not evidence of an available decoder. The image/audio API remains a separate route. |
| [Official LTX marking pack](https://huggingface.co/Lightricks/LTX-2-Marking/tree/718f58d124e0acf58da85752692d93b0bd54c38a) and [pinned audio implementation](https://github.com/Lightricks/LTX-2/blob/4f521619ae0029e8fbb64d555030c2b121a33b4f/packages/ltx-pipelines/src/ltx_pipelines/marking/audio_seal.py) | Public opt-in audio/video marking with available detector methods and public patterns. | The separate [80-case native audio contract](ltx-marking-audio-v1-result.md) is complete, including wrong payloads, stereo, WAV/AAC and noisy/partial misses. Video code was inspected only. Public patterns do not authenticate provider origin. |
| [Meitu Xiuxiu service terms](https://pro.meitu.com/xiuxiu/harmonyos-agreements/common/service.html?lang=en) | The reviewed version describes lower-left image notices and text/interface disclosures. | Adds scoped visible-label documentation. No hidden decoder, signed credential or audio/video compatibility is established. An interface notice may not survive export. |
| [Tiangong terms, clause 3.10](https://static.tiangong.cn/fe/skywork-site-assets/html/agreement/web-20260514/serve.html) | Its document-generation service describes visible disclosures and implicit file-metadata labels. | Adds a document labeling-policy record, not a verified signature or byte layout. It does not establish a watermark in copied plain text or cover every Skywork deployment. |
| [Shutterstock announcement](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-joins-content-authenticity-initiative) | A July 2023 intention to integrate C2PA in its AI Image Generator. | Retain it as an unconfirmed announcement. Current product help does not qualify a shipped provenance format. The matrix's Planned cell is not a new release promise. |
| [Getty generation FAQ](https://www.gettyimages.co.uk/ai/generation/faqs) | The current FAQ names Bria as its generation provider. | Add Getty/iStock as a review gap. A partner relationship does not prove that Bria marks survive Getty exports or that Bria's verifier accepts them. Third-party metadata assertions are insufficient. |

The [Recraft developer terms](https://www.recraft.ai/legal/developer-terms)
reserve possible marking; its [feature discussion](https://feedback.recraft.ai/feature-requests/p/incorporate-content-credentials-into-image-metadata)
records work in progress. Neither qualifies a deployed format. Similarly,
[Canva's preservation requirement](https://www.canva.com/policies/ai-product-terms/?frame=0)
does not establish marking of every export. Ideogram's possible safeguards are
not a released verifier. These distinctions remain explicit gaps.

The audio search again excluded Fish Audio voices merely named “Watermark”,
user-added Descript branding, and unrelated removal tools. IBM explanations of
other providers and synthetic-table research do not establish marking in
Granite passages. No compatible new production decoder was established for
Udio, Cohere, Leonardo, Fish, Hume, PlayAI, WellSaid, Voicemod, Mureka, PixVerse,
Freepik, Perplexity or AI21. Existing OpenMOSS and other gaps remain visible.

## Major-provider coverage is scoped by modality

- **Google/Gemini, Imagen, Veo, Lyria and NotebookLM:** retain the documented
  [SynthID](https://deepmind.google/models/synthid/) routes. Public media checking
  and published text methods are different interfaces. Partner listings do not
  establish every partner product, export date or modality.
- **Anthropic/Claude:** [keyed token-choice text marking](https://www.anthropic.com/news/claude-text-watermark)
  is distinct from supported-file Content Credentials. Hidden spaces are not
  its documented mechanism. Private-preview verification is not a local key.
- **Meta/Muse:** [Content Seal](https://facebookresearch.github.io/content-seal/)
  distinguishes production Muse Image from released research schemes. Public
  detector weights do not automatically establish production compatibility.
- **Grok/xAI:** [text watermarking](https://docs.x.ai/developers/advanced-api-usage/watermarking)
  has model/date/team controls, statistical detection and an uncertain state.
  Detector access is application-based. A copied response header cannot verify
  an uploaded passage; visual branding is a separate claim.
- **DeepSeek, MiniMax/Hailuo, Qwen/Wan, ByteDance/Doubao/Seedream/Seedance/CapCut,
  Hunyuan, ERNIE, Kling, GLM, Kimi, StepFun, SenseNova and iFLYTEK:** retain their
  individual primary references and product controls in the catalog. A generic
  Chinese labeling policy is not a shared decoder. Optional rhythm notices,
  logos, unsigned AIGC metadata and private watermark checks are separate kinds
  of evidence. The new Meitu/Tiangong rows do not imply live integration.
- **Adobe, Microsoft, Amazon, Stability, FLUX, Bria, Suno, ElevenLabs, Resemble,
  Cartesia, Runway and the remaining families:** the matrix retains every scoped
  route, gap and qualification state. A provider's classifier fallback must not
  be reported as recovery of a watermark.

## When processing may stop

The implemented local gate runs before acquiring statistical models. It stops
those checks only for a trusted, valid, exact-file-bound signed declaration of
AI involvement, without validation failures or unresolved remote references.
It preserves what was declared and which checks were skipped. Capture-only
credentials do not supply an AI-positive shortcut.

Visible labels, unsigned metadata, fixed public payloads and generic watermark
presence remain supporting observations. No negative, stripped, unsupported,
failed or inconclusive result becomes an original-file verdict. A mark in a
video's soundtrack cannot establish the origin of its pictures. Future
early-stopping decisions must preserve component scope rather than suppressing
unrelated checks.

The next integrations should qualify authentic original positive and negative
files for accessible routes, including CapCut, Suno, Bria and Cartesia. OpenAI,
Google and restricted text services need their exact supported access and
response contracts. Qualify exact input binding, watermark-versus-classifier
outcomes, errors, unsupported states and data handling before enabling a route.
Local public code needs compatible payloads/keys, licenses and real export
tests; a published method alone is not enough.

This follow-up made **zero provider-verification requests and zero customer
uploads**. External customer routes remain disabled. The catalog is research
documentation and never becomes uploaded-file evidence. Existing model-name
redaction, USD9.99 prepayment, voice calibration and GCP deferral are unchanged.
The expanded inventory does not change the existing 155-alias structured-claim
reader into 71 authenticated provider detectors.
