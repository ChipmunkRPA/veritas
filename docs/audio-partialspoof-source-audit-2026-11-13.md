# Fresh source audit for supervised short voice localization

PartialSpoof v1.2 supplies a new source for training a temporal head on
annotated speech, with publisher partitions whose declared speakers do not
overlap. Its public protocol supplies no per-edit generator lineage. Use
it for candidate development or a corpus-specific comparison. A
generator-held confirmation needs separate, explicitly identified source
families. No detector ran for this audit.
[Publisher data and attribution](https://zenodo.org/records/5766198)
· [Native label documentation](https://zenodo.org/records/5766198/files/README_v1.2)
· [Construction paper](https://arxiv.org/abs/2204.05177)

## Verified publisher metadata

The data publisher declares CC BY 4.0. The parent ASVspoof2019 data
release includes the ODC Attribution license, whose grant covers its
database rights and permits commercial use. Individual content rights
have separate scope. Retain attribution to both releases. This audit
did not download pretrained models or adopt any new data for commercial
training.
[Parent release and its license file](https://datashare.ed.ac.uk/handle/10283/3336)

The complete README, CM protocols and timestamp archives match their
published MD5s and the pinned SHA-256s. All 121,461 expected time-label
files are present. Every positive is labeled `CON`, despite source
ASVspoof protocols listing individual attack IDs and the PartialSpoof
paper describing inputs with segments from several attacks. Original
genuine IDs link across the two CM protocols. The public metadata has no
mapping from a `CON` ID or interval to those parent attacks or donor
recordings, so the original speaker label also cannot establish every
input's complete source ancestry.

| Publisher partition | Declared speakers | Positive protocol rows | Score-blind short-edit candidates | Speakers among candidates |
| --- | ---: | ---: | ---: | ---: |
| Train | 20 | 22,800 | 87 | 19 |
| Development | 20 | 22,296 | 113 | 10 |
| Evaluation | 67 | 63,882 | 411 | 48 |

The census requires one contiguous 0.50–1.20 s spoof interval in 4–9 s of
annotation, with at least 0.40 s of annotated genuine speech on each side.
It merges directly adjacent same-label intervals only. The 63,882 positive
evaluation annotations each contain two identical sequences starting over
at the same time. The reader discards the second only when both halves
are byte-equivalent line sequences. Before accepting any native row,
independently decode the original audio and verify its channel format,
sample rate and final label position on the exact source clock. Separate
positive protocol rows lack any spoof interval in the VAD metadata. Those
28,463 rows are ineligible for supervised local truth. Do not label them
genuine. [Machine-readable census](evaluation-panels/audio-partialspoof-source-audit-2026-11-13.json)

The train and development tar.gz endpoints both returned valid 206
responses for their first 64 KiB. This supports bounded native preflight
without caching the complete multi-gigabyte archives. A gzip stream still
needs its prefix; a random compressed offset does not provide individual
WAV access. The probes acquired no complete audio and no eval archive.
Larger prefixes would need to contain enough suitable genuine and
edited files before any waveform group could be frozen.

## Separate confirmation sources

- **PhonemeDF v1.0.1** supplies explicit per-generator native archives and
  aligned genuine/synthetic words. The existing exact same-word experiment
  admitted four reader roles from bounded prefixes and used only the two
  Chatterbox branches. A new run can expand the pinned prefixes and use
  MeloTTS as one separately identified generator, checking the published
  model terms and retaining an additional qualified generator for held
  confirmation. Keep speaker IDs and exact utterance and sample digests
  disjoint from earlier study roles. Match natural controls through the
  same resampling/fades, and reserve different original AMI meetings and
  disjoint manual-annotation participant groups across roles. Those
  new clips must qualify before use. Publisher-known group separation
  cannot establish undocumented pretrained-model exposure.
  [Existing aligned-input procedure](audio-phoneme-short-edit-intake-2026-11-13.md)
  · [PhonemeDF archives](https://zenodo.org/records/19522979)
  · [MeloTTS publisher terms](https://huggingface.co/myshell-ai/MeloTTS-English-v3)
  · [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
- **ToolDF and LENS-DF** provide useful rendering designs. ToolDF's MIT
  repository says MAADD audio is composed locally from several
  independently licensed collections. LENS-DF's MIT code and sample
  record do not resolve all underlying speech and MUSAN component terms
  for our use. Its paper applies background at 0–10 dB and concatenates
  complete segments, a different scope from sub-second inserts under
  louder competing voices.
  [ToolDF source and composition](https://github.com/rlataewoo/tooldf)
  · [LENS-DF sample](https://zenodo.org/records/15948624)
  · [LENS-DF source design](https://arxiv.org/html/2507.16220v1)
- **WeDefense and PartialEdit** expose independent methods. The WeDefense
  toolkit lists PartialSpoof as supported and other partial corpora as
  planned. PartialEdit v1.1 gives corrected timestamps and speaker splits.
  Although its release metadata declares CC BY 4.0, VoiceCraft's
  upstream model license expressly restricts use of its outputs for
  commercial training. Keep the E1 archive out of that pipeline.
  E2's exact checkpoint terms would need separate verification.
  [WeDefense support](https://github.com/zlin0/wedefense)
  · [PartialEdit v1.1](https://zenodo.org/records/18829689)
  · [VoiceCraft model terms](https://github.com/jasonppy/VoiceCraft/blob/master/LICENSE-MODEL)

## Replay

Run this standard-library audit from the repository root. The files use
official publisher URLs. The two range requests are limited to 64 KiB
each, and the reviewer checks the returned range and byte digest.

```sh
set -e
scratch=$(mktemp -d)
base=https://zenodo.org/records/5766198/files
curl -fLsS https://zenodo.org/api/records/5766198 -o "$scratch/5766198.json"
for name in README_v1.2 database_protocols.tar.gz database_vad.tar.gz; do
  curl -fLsS "$base/$name" -o "$scratch/$name"
done
for split in train dev; do
  curl -fLsS --range 0-65535 --max-filesize 65536 \
    -D "$scratch/$split-range.headers" \
    "$base/database_$split.tar.gz?download=1" -o "$scratch/$split-prefix"
done
python3 -B -IS backend/scripts/review_audio_partialspoof_source.py "$scratch" \
  --compare docs/evaluation-panels/audio-partialspoof-source-audit-2026-11-13.json
```
