# Provider marking coverage matrix

Inventory updated 2026-11-08: **82 families, 105 scoped records**, with **28 families remaining explicit review gaps**.
Records include planned features, public research and visible notices; these counts are not working detectors.

Each cell records documentation scope, not measured accuracy or verified production compatibility. “Gap” means not established by this review, including where the provider does not offer that modality. “Planned” means an announcement only; “Historical” is not current export coverage. “Research” identifies a released scheme, without claiming production-provider compatibility. “Export mark” leaves the marking mechanism unresolved; “version-limited” is conditional on the documented software version. “Metadata service” identifies customer-applied tags, without establishing default model-output marking. A generic policy with unresolved export scope does not fill the four cells.

All external verification routes are disabled for customer files. Only the existing trusted, valid, current-file-bound, explicitly AI-declaring Content Credentials gate can skip statistical checks. A video-container audio check covers sound only. A provider classifier score is not a verified watermark hit.

See the [source catalog](provider-signals-catalog.json), [coverage and gate](provider-provenance-coverage.md) and [latest search findings](provider-signal-followup-2026-11-08.md) for exact products, controls, dates, limits, contradictions and qualification requirements.

| Provider family | Image | Video | Audio | Writing | Reviewed |
|---|---|---|---|---|---|
| [OpenAI / ChatGPT / Sora](https://developers.openai.com/api/docs/guides/content-provenance) | Credentials, Watermark | Historical | Watermark | Keyed text | 2026-11-08 |
| [Anthropic / Claude](https://www.anthropic.com/news/claude-text-watermark) | Credentials | Gap | Gap | Keyed text | 2026-11-08 |
| [Google / Gemini / Imagen / Veo / Lyria / NotebookLM / Genie](https://deepmind.google/models/synthid/) | Watermark, Credentials, Classifier | Watermark, Credentials | Watermark | Keyed text | 2026-11-08 |
| [Meta / Muse](https://github.com/facebookresearch/content-seal) | Watermark, Research | Research | Research | Research | 2026-11-08 |
| [xAI / Grok](https://docs.x.ai/developers/advanced-api-usage/watermarking) | Visible label | Visible label | Gap | Keyed text | 2026-11-08 |
| [Microsoft / Azure / Copilot](https://learn.microsoft.com/en-us/azure/foundry/responsible-ai/content-safety/provenance-disclosure) | Credentials, Watermark | Credentials, Watermark | Credentials, Watermark | Keyed text | 2026-11-08 |
| [Amazon / Nova / Bedrock](https://docs.aws.amazon.com/en_en/bedrock/latest/userguide/watermarking.html) | Watermark | Watermark, Credentials | Watermark | Keyed text | 2026-11-08 |
| [Adobe / Firefly](https://helpx.adobe.com/firefly/web/get-started/learn-the-basics/content-credentials-overview.html) | Credentials | Credentials | Credentials | Gap | 2026-11-08 |
| [Stability AI](https://kb.stability.ai/knowledge-base/content-credentials-c2pa-faq) | Credentials, Research | Credentials | Credentials | Gap | 2026-11-08 |
| [Black Forest Labs / FLUX](https://bfl.ai/legal/developer-terms-of-service) | Credentials, Research | Gap | Gap | Gap | 2026-11-08 |
| [Bria](https://docs.bria.ai/visual-verification-overview) | Credentials, Watermark | Credentials, Watermark | Gap | Gap | 2026-11-07 |
| [Midjourney](https://docs.midjourney.com/hc/en-us/articles/48207461163661-Content-Authenticity) | Metadata | Metadata | Gap | Gap | 2026-11-07 |
| [ElevenLabs](https://elevenlabs.io/docs/eleven-creative/audio-tools/audio-detector) | Gap | Gap | Watermark, Classifier | Gap | 2026-11-08 |
| [Resemble AI / Chatterbox](https://github.com/resemble-ai/Perth) | Watermark | Watermark | Research, Watermark | Watermark | 2026-11-07 |
| [Suno](https://suno.com/suno-credentials) | Gap | Gap | Credentials | Gap | 2026-11-07 |
| [MiniMax / Hailuo](https://github.com/MiniMax-AI/cli/blob/main/src/commands/image/generate.ts) | Visible label | Export mark | Audible label | Gap | 2026-11-08 |
| [DeepSeek](https://cdn.deepseek.com/policies/zh-CN/deepseek-terms-of-use.html) | Gap | Gap | Gap | Label policy | 2026-11-08 |
| [Alibaba / Qwen / Wan](https://help.aliyun.cn/zh/model-studio/text-to-image-v2-api-reference) | Visible label | Visible label | Label policy | Gap | 2026-11-08 |
| [ByteDance / Doubao / Seedance / Seedream / CapCut / Dreamina](https://docs.volcengine.com/docs/lake-ai-service/video-generation-enhanced-edition-basic-edition?lang=en) | Label policy, Watermark, Credentials, Visible label | Visible label, Custom mark, Label policy, Watermark, Credentials | Audible label, Label policy, Watermark, Credentials | Gap | 2026-11-08 |
| [Tencent / Hunyuan](https://cloud.tencent.com/document/api/1616/119001) | Metadata service | Visible label, Metadata service | Metadata service | Metadata service | 2026-11-08 |
| [Baidu / ERNIE](https://ai.baidu.com/ai-doc/NLP/Ml9i5amtk) | Watermark | Gap | Gap | Gap | 2026-11-07 |
| [Kuaishou / Kling](https://kling.ai/docs/ai-content-detection) | Watermark | Watermark | Gap | Gap | 2026-11-07 |
| [Synthesia](https://help.synthesia.io/en/articles/17634791-how-do-i-add-an-ai-label-to-my-videos) | Gap | Credentials, Visible label | Gap | Gap | 2026-11-07 |
| [HeyGen](https://help.heygen.com/en/articles/11057301-how-to-remove-the-heygen-watermark) | Gap | Visible label | Gap | Gap | 2026-11-07 |
| [Runway](https://help.runwayml.com/hc/en-us/articles/50404627334547-Free-plan-details) | Gap | Visible label | Gap | Gap | 2026-11-07 |
| [Pika](https://pika.art/support) | Gap | Historical | Gap | Gap | 2026-11-07 |
| [Moonshot / Kimi](https://www.kimi.com/user/agreement/modelUse?version=v2) | Label policy | Label policy | Label policy | Label policy | 2026-11-07 |
| [NVIDIA](https://deepmind.google/models/synthid/) | Watermark | Watermark | Gap | Gap | 2026-11-08 |
| [Kakao](https://deepmind.google/models/synthid/) | Gap | Gap | Gap | Gap | 2026-11-07 |
| [Apple](https://deepmind.google/models/synthid/) | Watermark (version-limited) | Gap | Gap | Gap | 2026-11-08 |
| [Zhipu / Z.ai / GLM](https://docs.bigmodel.cn/api-reference/模型-api/图像生成.md) | Visible label, Watermark | Visible label, Watermark | Gap | Gap | 2026-11-08 |
| [Mistral](https://legal.mistral.ai/ai-governance/ai-systems/vibe-code/) | Gap | Gap | Gap | Planned | 2026-11-08 |
| Cohere | Gap | Gap | Gap | Gap | 2026-11-08 |
| [Luma](https://lumalabs.ai/learning-hub/payments-subscriptions) | Visible label | Visible label | Gap | Gap | 2026-11-08 |
| Udio | Gap | Gap | Gap | Gap | 2026-11-08 |
| Ideogram | Gap | Gap | Gap | Gap | 2026-11-08 |
| Recraft | Gap | Gap | Gap | Gap | 2026-11-08 |
| [Cartesia](https://docs.cartesia.ai/build-with-cartesia/capability-guides/watermarking) | Gap | Gap | Watermark | Gap | 2026-11-08 |
| [Inworld](https://inworld.ai/blog/introducing-inworld-tts) | Gap | Gap | Watermark, Classifier | Gap | 2026-11-08 |
| [D-ID](https://help.d-id.com/hc/en-us/articles/31350550554641-How-do-I-change-or-remove-the-watermark-on-my-videos) | Gap | Visible label | Gap | Gap | 2026-11-08 |
| [WRITER / Palmyra](https://writer.com/blog/oct-roundup-new-at-writer/) | Gap | Gap | Gap | Planned | 2026-11-08 |
| [StepFun](https://platform.stepfun.com/legal/user-agreement.html) | Gap | Gap | Gap | Gap | 2026-11-08 |
| [Baichuan](https://policy.baichuan-ai.com/cooperation/agreement) | Gap | Gap | Gap | Gap | 2026-11-08 |
| Fish Audio | Gap | Gap | Gap | Gap | 2026-11-08 |
| Hume | Gap | Gap | Gap | Gap | 2026-11-08 |
| PlayAI / PlayHT | Gap | Gap | Gap | Gap | 2026-11-08 |
| [Lightricks / LTX](https://help.ltx.io/hc/en-us/p/ltx-studio) | Gap | Export mark, Watermark | Watermark | Gap | 2026-11-08 |
| Perplexity | Gap | Gap | Gap | Gap | 2026-11-08 |
| IBM / Granite | Gap | Gap | Gap | Gap | 2026-11-08 |
| AI21 / Jamba | Gap | Gap | Gap | Gap | 2026-11-08 |
| [Samsung / Galaxy AI](https://www.samsung.com/us/support/answer/ANS10000934/) | Visible label | Gap | Gap | Gap | 2026-11-08 |
| [SenseTime / SenseNova](https://www.sensetime.com/hk/news/51169933) | Label policy, Export mark | Label policy | Label policy | Label policy | 2026-11-08 |
| [iFLYTEK / Spark](https://static.xfyun.cn/doc/spark/super%20smart-tts.html) | Gap | Gap | Label policy, Audible label | Gap | 2026-11-08 |
| [Murf](https://murf.ai/api/docs/capabilities/dubbing) | Gap | Export mark | Gap | Gap | 2026-11-08 |
| Canva | Gap | Gap | Gap | Gap | 2026-11-08 |
| Leonardo AI | Gap | Gap | Gap | Gap | 2026-11-08 |
| Descript | Gap | Gap | Gap | Gap | 2026-11-08 |
| [Higgsfield](https://higgsfield.ai/creator-hub/help-center/credits/watermark-and-how-to-remove) | Visible label | Visible label | Gap | Gap | 2026-11-08 |
| [Vidu](https://platform.vidu.com/docs/api-reference/video-models/vidu-q3/reference-to-video) | Gap | Visible label, Label policy | Gap | Gap | 2026-11-08 |
| [Speechify](https://speechify.com/studio-terms/) | Gap | Gap | Watermark | Gap | 2026-11-08 |
| [Krea](https://www.krea.ai/models/gemini-omni-flash) | Gap | Watermark, Credentials | Gap | Gap | 2026-11-08 |
| PixVerse | Gap | Gap | Gap | Gap | 2026-11-08 |
| Freepik / Magnific | Gap | Gap | Gap | Gap | 2026-11-08 |
| WellSaid | Gap | Gap | Gap | Gap | 2026-11-08 |
| Voicemod | Gap | Gap | Gap | Gap | 2026-11-08 |
| OpenMOSS / MOSI.AI | Gap | Gap | Gap | Gap | 2026-11-08 |
| Mureka | Gap | Gap | Gap | Gap | 2026-11-08 |
| [Meitu / Xiuxiu / Meiyan](https://pro.meitu.com/xiuxiu/harmonyos-agreements/common/service.html?lang=en) | Visible label | Gap | Gap | Visible label | 2026-11-08 |
| [Kunlun / Tiangong / Skywork](https://static.tiangong.cn/fe/skywork-site-assets/html/agreement/web-20260514/serve.html) | Gap | Gap | Gap | Label policy | 2026-11-08 |
| [Shutterstock](https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-joins-content-authenticity-initiative) | Planned | Gap | Gap | Gap | 2026-11-08 |
| Getty Images / iStock | Gap | Gap | Gap | Gap | 2026-11-08 |
| [Deepgram](https://deepgram.com/watermark) | Gap | Gap | Watermark | Gap | 2026-11-08 |
| [Sarvam](https://docs.sarvam.ai/api/api-guides-tutorials/dubbing/how-to/choose-export-formats) | Gap | Visible label | Gap | Gap | 2026-11-08 |
| [vivo / BlueLM](https://bbs.vivo.com.cn/newbbs/thread/39000231) | Visible label | Visible label | Gap | Gap | 2026-11-08 |
| Xiaomi / MiMo | Gap | Gap | Gap | Gap | 2026-11-08 |
| HONOR | Gap | Gap | Gap | Gap | 2026-11-08 |
| OPPO / Xiaobu | Gap | Gap | Gap | Gap | 2026-11-08 |
| Moonvalley / Marey | Gap | Gap | Gap | Gap | 2026-11-08 |
| Rime | Gap | Gap | Gap | Gap | 2026-11-08 |
| Smallest AI / Waves | Gap | Gap | Gap | Gap | 2026-11-08 |
| NAVER / HyperCLOVA | Gap | Gap | Gap | Gap | 2026-11-08 |
| LG AI Research / EXAONE | Gap | Gap | Gap | Gap | 2026-11-08 |

Generated from the catalog with `python backend/scripts/provider_signal_inventory.py --write`. Use `--check` to detect stale counts or a matrix that no longer matches the research records.
