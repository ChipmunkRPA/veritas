# Fresh audio comparison: generator and source preparation

The [ten-method development comparison](audio-combinations-v52-result.md)
found complementary gains worth fresh testing. This preparation records
publisher metadata for potential new test generators. It is **not a completed
license-chain approval, runtime qualification, source-role freeze or accuracy
test**. No generator weights or source audio were downloaded and no generation
or detector call was made in this step.

| Publisher repository | Pinned revision | Publisher weight terms | Main weight bytes |
| --- | --- | --- | ---: |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M/tree/f3ff3571791e39611d31c381e3a41a3af07b4987) | `f3ff3571791e39611d31c381e3a41a3af07b4987` | Apache-2.0 | 327,212,226 |
| [MeloTTS-English](https://huggingface.co/myshell-ai/MeloTTS-English/tree/bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef) | `bb4fb7346d566d277ba8c8c7dbfdf6786139b8ef` | MIT | 207,860,748 |
| [OuteTTS-0.3-500M](https://huggingface.co/OuteAI/OuteTTS-0.3-500M/tree/7e7c98867a077fca82ede8aa649b3afc698dd8c8) | `7e7c98867a077fca82ede8aa649b3afc698dd8c8` | CC-BY-SA-4.0 | 998,419,744 |
| [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo/tree/749d1c1a46eb10492095d68fbcf55691ccf137cd) | `749d1c1a46eb10492095d68fbcf55691ccf137cd` | MIT | Multiple components; not yet selected |

These byte counts and LFS digests are publisher metadata, not verification of
downloaded weight bytes or a total runtime footprint. Auxiliary text encoders,
tokenizers, vocoders, speaker assets and their notices require separate review
and a pinned offline runtime before use. Preserve attribution/share-alike
conditions where applicable. The Oute 500M card explicitly distinguishes its
commercially licensed variant from the **noncommercial 1B variant**; its example
code defaults to 1B. Do not copy that default or substitute a larger model.

Kokoro uses StyleTTS2 architecture, already represented in the earlier source
pool; Chatterbox Turbo also shares a family with earlier material. Different
checkpoints are not proof of unseen architecture, voice identity or absence from
pretraining. Melo and the Qwen-based Oute candidate offer other synthesis paths,
subject to the remaining asset/runtime review. Record family overlap instead of
describing every new generated file as an independent new generator.

For genuine speech, [OpenSLR's LibriSpeech page](https://www.openslr.org/12/)
states CC-BY-4.0 and links original metadata. Audit speaker/chapter/text identities
against all prior project cohorts before choosing recordings. A new file offset
does not make an already used speaker independent. Fixed TTS voice packs are
also repeated speakers: many prompts through one voice are not many independent
source groups. Keep prompt/reference/voice-connected components together and
assign source roles only after inspecting the resulting graph and support.

Next prepare a small deterministic, resource-bounded local runtime smoke using
authored non-speech/test prompts, then review actual generation and decoding
cost before choosing phase sizes. Freeze generator revisions, allowed assets,
prompts, seeds, foreground/background identities, roles and every candidate
operator before source inference. Keep model selection and final confirmation
separate. Include strong noise, music, genuine voice overlap, ordinary processing
and unavailable/speech-gate controls. Route using measured conditions only,
never the known truth or generator. Retain failures and all false-call types.

The priority detector arms are current, the reopened v7 decision layer, the
standalone complement, pair conflict-abstention, sensitive three-head union and
three-head conflict-abstention. A future combined probability needs its own
fitting/calibration; raw agreement does not supply it. The full parameter/source
freeze is pending the audits above. No serving default, calibration, customer
confidence, workflow, outreach, payment or deployment changed. Large assets
belong on the authorized external development drive with verified hashes.
