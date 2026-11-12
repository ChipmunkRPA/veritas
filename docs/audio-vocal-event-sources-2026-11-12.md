# Vocal-event sources for strong-background audio tests

The source roster contains three synthetic vocal events and six ordinary recordings. All nine files decoded locally. The roster can support a small test of foreground laughter and breath-like sounds against stronger backgrounds once listening review and matched mixtures are complete. The two ordinary coughs are additional false-alarm controls. No mixture or detector was run at this stage. [Source receipt](evaluation-panels/audio-vocal-event-source-roster-2026-11-12.json)

| Source | Event | Duration | Commitment |
| --- | --- | ---: | --- |
| LAION `0323`, publisher item `421` | Adult-labelled snort-laugh | 11.00 s | Raw WAV and JSON member byte ranges and hashes |
| LAION `0376`, publisher item `729` | Adult-labelled nasal sniff | 9.50 s | Raw WAV and JSON member byte ranges and hashes |
| LAION `0715`, publisher item `1441` | Adult-labelled giggle | 8.10 s | Raw WAV and JSON member byte ranges and hashes |
| Freesound `529818`, dastudiospr | Recorded laughter | 3.66 s | CC0 and exact official MP3 preview hash |
| Freesound `823548`, beautifuldaymonster1968 | Performed chuckle with documented EQ | 6.97 s | CC0 and exact official MP3 preview hash |
| Freesound `516473`, padpad | Three recorded nasal sniff variations | 6.22 s | CC0 and exact official MP3 preview hash |
| Freesound `569568`, ValentinPetiteau | Recorded inhale and exhale | 7.73 s | CC0 and exact official MP3 preview hash |
| Freesound `466874`, jmbland94 | Self-recorded adult cough | 3.97 s | CC0 and exact official MP3 preview hash |
| Freesound `393733`, jackthemurray | Incidental cough recorded during Foley | 3.47 s | CC0 and exact official MP3 preview hash |

LAION declares CC BY 4.0 for the pinned dataset. OpenMOSS separately declares Apache 2.0 on its model card. The archive attributes all three synthetic files to MOSS SoundEffect v2.0 and gives their generation prompts. The dataset card leaves the exact checkpoint hash unspecified. Any distribution of the source audio must retain LAION attribution and the license. The ordinary recordings have separate per-item CC0 declarations from six contributors. Their cited descriptions and uploads support the source classification. [Pinned LAION dataset card](https://huggingface.co/datasets/laion/more-synthetic-vocalbursts-raw/blob/e2c0c2a7b322d8b89cd01c9982062f87eaa7d742/README.md) · [Pinned OpenMOSS model card](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0/blob/e35df4d82fbe87fcd5d14e5d100e349c0c3c076d/README.md) · [Individual item links and licenses](evaluation-panels/audio-vocal-event-source-roster-2026-11-12.json)

The bounded TAR header walk inspected 156 sidecars, including 139 with eligible adult prompt labels. Selection retained the specified laughter and breath-like categories only. The raw `moss/` archive also contains unrelated prompts. The retained set excludes those prompts and all enhanced, separate NSFW and other generator routes. The available synthetic cough label describes respiratory illness and was outside the neutral selection. Adult labels are properties of the generation prompts. Listener review of the retained sound, absence of speech and extra layers remains pending. [Pinned vocal-burst taxonomy](https://github.com/LAION-AI/Voice-Acting-Pipeline/blob/9ed7cd95b3e48d09d53749f2ef9eb85229b138ca/data/vocal_bursts_taxonomy_sfw.json)

## Replay

From the repository root, with Python 3.10 or newer, FFmpeg and FFprobe installed:

```sh
run=backend/data/independent-evaluation/audio-moss-vocal-event-sources-2026-11-12
python3 backend/scripts/audio_vocal_event_sources.py fetch "$run"
python3 backend/scripts/audio_vocal_event_sources.py verify "$run"
```

The fetch checks the revision and publisher LFS SHA-256 at each Hugging Face redirect, then requires HTTP 206 for the exact byte envelope containing the two named TAR members. It validates both header checksums and member hashes without retrieving the complete shard. For a control preview it checks the public item page for the CC0 link and the exact preview URL before verifying the preview hash. The offline check verifies every committed media file and fully decodes it. The media path is Git-ignored. [Replay script](../backend/scripts/audio_vocal_event_sources.py)

A clean fetch into a fresh directory reproduced all nine media files and the three sidecars byte for byte. The three archive envelopes transferred 5,496,048 bytes. The six official preview files transferred 736,594 bytes, plus the public item pages. The offline check rejected a single-byte WAV change.

All synthetic clips share a generator family and native 48 kHz PCM WAV. The controls are 44.1 kHz MP3 playback files. A detector could learn codec history, channel count or duration instead of voice origin. A follow-up needs both classes subjected to the same rendering conditions, separate original-codec checks and additional generator and human recording families before estimating error rates. The source roster supplies no evidence about ambient-environment generation or detector performance under strong background noise.
