# Public watermark detectors and production-provider coverage

Reviewed 7 November 2026 after the user's challenge to the earlier blanket
statement about provider access. **We can run released compatible watermark
detectors locally.** API access is only one verification route, not a general
prerequisite. Reading an algorithm alone does not establish which weights,
configuration and key a deployed product uses; those facts need checking.

## Concrete public implementations

| Public system | Released resources and requirements | What a successful local test would establish |
|---|---|---|
| [AudioSeal](https://github.com/facebookresearch/audioseal) | The publisher explicitly licenses code and model weights under MIT. Public generator and detector checkpoints, including localized detection, are available. The optional 16-bit message does not affect watermark detection. | Compatibility with the released AudioSeal watermark family. The detector does not need to know an embedded message to recognize that family. It does not establish which provider or person applied it. |
| [VideoSeal / PixelSeal / ChunkySeal](https://github.com/facebookresearch/videoseal) | The publisher releases image/video model code, checkpoints and evaluation tools under MIT. Versions/checkpoint identities and preprocessing still need pinning and execution tests. | Compatibility with the particular released image/video watermark models. It is not automatically verification of every current Muse Image/Video output. |
| [Watermark Anything](https://github.com/facebookresearch/watermark-anything) | The newer SA-1B-based `wam_mit.pth` is explicitly MIT. The original `wam_coco.pth` is separately noncommercial and is excluded from commercial admission. | Detection/localization of compatible research watermarks; no general provider attribution from a recovered bit pattern alone. |
| [TextSeal](https://github.com/facebookresearch/textseal) | Toolkit code is Apache-2.0. It includes public detection using one key and fuller detection using both keys. Matching tokenizer/configuration and the appropriate verification key are needed; generator checkpoints have separate licenses. | Detection under an actually supplied compatible configuration/public key. A demonstration key is not a deployed-provider key. Do not assert that every scheme needs a private secret. |
| [SynthID Text](https://github.com/google-deepmind/synthid-text) | Apache-2.0 reference detector code. Its documented detector must use the same watermark configuration as generation. The Bayesian detector additionally trains per key; the frequentist detector does not need that training. | Detection for the configured key/tokenizer/scheme, not automatic recognition of all Gemini or Claude output. A null-test p-value is not AI-authorship confidence. |

The [Meta Content Seal overview](https://github.com/facebookresearch/content-seal)
expressly distinguishes the proprietary custom Muse Image deployment from the
downloadable open research versions. The [official project site](https://facebookresearch.github.io/content-seal/)
links those local research implementations. That distinction is stronger evidence
than either assuming universal compatibility or assuming every detector is
unavailable without a provider account.

## Work to execute locally

Start with the public AudioSeal generator/detector and public image/video
detectors. Audit pinned source, weights and transitive dependencies; run isolated
authored marked/unmarked controls before touching serving dependencies. Check
partial marked intervals/regions, compression, resizing/resampling and added
noise, preserving source coordinates and incomplete coverage. Then evaluate
rights-cleared marked and unmarked real media with points fixed before new
confirmation cases. Positive authored controls establish implementation behavior,
not production-provider accuracy.

Integrate passing checks as explicitly scoped watermark-family evidence with
matching UI/PDF/API/JSON records. A mark may be added to genuine material, so
watermark presence alone is not a generic proof of AI generation. Provider/AI
claims require the corresponding authenticated declaration or independently
validated compatible product signal. Missing marks do not establish originality.
Do not use public demonstration keys to pretend a private product was checked.

The [first actual AudioSeal native contract](public-audio-watermark-v1-result.md)
is now complete: pinned public source/weights, an isolated local adapter,
32authored cases and a separate waveform/time/readout replay. It found full and
partial marks but also false responses on unmarked synthetic carriers and noisy
misses. This is implementation evidence, not real-speech/provider accuracy or
app admission. No serving dependency or confidence changed. Public image/video
and text family adapters, rights-cleared real-media qualification and app exports
remain required. External private verifiers are unconfigured; their absence does
not block this public-family work. No customer media was sent to a provider.
