# Learned speech extraction preserves four noisy clones but suppresses a clean clone

The frozen noisy-only preservation guard passes all four held deliveries, which
repeat two underlying reader/meeting lineages at two interference levels. The full
panel also contains a held clean clone attenuated by more than 30 dB and a
development noisy clone below the −3 dB retention bound. These results support
further diagnostic comparison while leaving general synthetic preservation
unqualified. The original waveform stays essential.

This is the separately frozen follow-up to the genuine reconstruction preflight.
It uses the same exact causal BSRNN checkpoint, configuration, genuine references,
background components and original sample clocks. The genuine acoustic result
alone could not establish that extraction preserves synthetic edits.

## Source and native construction

Four retained publisher-linked Chatterbox PLN clones supply two development and
two held targets. The publisher's sidecars identify each cloned reader and assign
user and assistant audio channels. Rehash the three complete retained TAR files,
verify each native 24 kHz stereo original and sidecar, and select only the exact
user channel. Its independently decoded 16 kHz PCM16 waveform must match the
previously qualified derivative. The assistant channel contributes no samples.

The declared clone, host and enrollment reader match. Each genuine enrollment is
from a different chapter than its host. Development and held roles share no
declared reader or normalized source-script key. Publisher identity assignments
are provenance claims. The stripped original cloning-prompt audio prevents a
complete prompt-overlap audit. These sources and rooms have prior project
exposure, and same-corpus overlap with the extractor's train-clean-100 recipe
remains possible.

A fixed hash order selects a 960 ms, level-qualified fragment from each retained
three-second clone prefix before learned responses. Fit its RMS to the genuine
host interval, apply the original shared peak-safe gain and ten-millisecond fades,
and round the weighted clone and surviving host boundary separately. Add the exact
existing noise. All samples outside the authored interval remain byte-identical to
their matched genuine control. Twelve native twelve-second inputs cover clean,
−5 dB and −10 dB conditions across four targets. Two reference choices give 24
outputs plus one exact repeat.

## Frozen measurement and complete outcome

At the original fixed interval, fit the raw output against the exact weighted
clone, remaining genuine boundary and noise components. Clone coefficient magnitude
gives acoustic retention. This measurement applies no time realignment, peak
normalization, score fitting or rescaling of model outputs. Its known components
exist for this controlled audit and are unavailable from an ordinary user upload.

The predeclared guard requires every held noisy true-reference clone to retain at
least −3 dB. It contains four condition deliveries from two lineages, with one
Chatterbox family. The clean findings are reported diagnostics outside that
predeclared noisy-only gate.

| Scope, true reference | Deliveries | Retained at least −3 dB | Strongly attenuated finding |
| --- | ---: | ---: | --- |
| Held, −5 dB | 2 | 2 | None below the bound |
| Held, −10 dB | 2 | 2 | None below the bound |
| Held, clean | 2 | 1 | One below −30 dB |
| Development, both noisy levels | 4 | 3 | One below −3 dB |
| Development, clean | 2 | 2 | None below the bound |

All four held noisy gains lie in [−3, 0) dB. This necessary noisy guard passes.
The clean failure shows why a cleaner view cannot replace the original for
analysis. A reference-conditioned extractor can remove the material an
investigation needs to examine. This study has no authenticity detector outcome,
false-alarm rate, calibrated confidence, voice attribution or production activation.

## Second-family boundary and rights

The [DuplexGen publisher](https://huggingface.co/datasets/DuplexGen/duplexgen-spoken)
assigns MIT terms to the PLN scenario. The
[publisher license table](https://github.com/duplexgen/duplexgen-code/blob/main/docs/DATA_LICENSES.md)
is the applicable scenario-specific source grant. The genuine recordings and
background annotations retain [LibriSpeech CC BY 4.0](https://www.openslr.org/12/)
and [AMI CC BY 4.0](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) terms.
The previously reviewed extractor and permitted active components stay unchanged.

Retained PhonemeDF MeloTTS and other rendered families offer separate licensed
recordings, but they do not provide another clone of these LibriSpeech hosts. The
[primary paper](https://aclanthology.org/2026.lrec-1.469.pdf), section 3, explains that
generation uses built-in or VCTK reference voices and explicitly excludes cloning
the original LibriSpeech readers. Such audio can support a separate distinct-voice
stress test with its own scope and freeze. It cannot increase this same-identity
family denominator.

## Independent review and privacy

A separate native implementation verified all twelve inputs, 2,304,000 sample
values, 2,119,680 outside-edit sample identities and eight authored component-grader
controls. The source extraction run closed in about 78 seconds on two CPU threads.
Independent arithmetic checked 4,608,000 raw output samples and all 120 recorded
scalars, agreeing with the frozen noisy guard and every reported diagnostic.

The companion receipt contains grouped broad retention bands and counts. It omits
individual identities, source clocks, paths, source fingerprints, response vectors,
features and media. The portable reviewer uses only the standard library, checks
the pinned aggregate and rejects eight authored alterations. Exact source/model
replay requires retained private custody.

```sh
python3 -I -B -S backend/scripts/review_audio_learned_enrollment_conditional_retention.py \
  docs/evaluation-panels/audio-learned-enrollment-conditional-retention-2026-11-19.json --self-test
```

The next comparison will hold the existing serving detector points fixed and test
original and extracted copies separately on synthetic, genuine and background
controls. Acoustic retention supplies no automatic detection improvement.
