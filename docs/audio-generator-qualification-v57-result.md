# Offline CPU speech generation qualified on authored text

The separately frozen v57 smoke passed full English preprocessing and three
generation calls on two authored sentences. It qualifies this local generation
path for preparing test material. It is **not fresh detector validation**, a
human intelligibility review or qualification of the other planned generators.
No research cohort or customer upload was used or scored.

All **548 saved checkpoint tensors**, containing 81,763,410 elements, match their
actual loaded key/shape/dtype/value. The upstream graph's 140 additional AdaIN
normalization parameters were verified as exact one scales and zero biases,
with no running statistics. Every parameter was frozen for inference and the
complete model state fingerprint stayed unchanged after generation. Independent
functional tests verify the identity normalization operation and reject missing
learned tensors, nonneutral state and mixed/unexpected checkpoint keys.

The generator and preprocessing ran offline from explicit publisher-bound local
assets. G2P produced usable, in-vocabulary phonemes for every spoken token, with
no dropped word or silent truncation. The outputs were native mono float32 at
24 kHz, finite and nonzero, with no samples beyond the PCM clipping range:

| Authored call | Native samples | Duration | Generation time in this local smoke |
| --- | ---: | ---: | ---: |
| Prompt 0 | 134,400 | 5.600 seconds | 0.870 seconds |
| Prompt 1 | 111,000 | 4.625 seconds | 0.663 seconds |
| Prompt 0 repeat | 134,400 | 5.600 seconds | 0.764 seconds |

The fixed-seed repeat's complete float32 waveform matches exactly. These three
timings on one Mac are observations, not a latency promise. Production decoding,
source/voice-role assignment and multi-generator detector comparison remain
separate work. The native waveforms stay private.

Both bounded preparation and smoke profiles passed with no guard, sampling
error, resource violation or network attempt. The smoke lasted 11.924 seconds
including validation/import/load/generation, with sampled worker-tree RSS
1,521,188,864 bytes. This is a sampled lower bound, not a certified memory peak.
Afterward all 25,302 installed files, 90 package versions and frozen project
code bindings were reverified. Thirteen focused tests and lint passed. The
[result](evaluation-panels/audio-generator-qualification-v57-result.json) and
[execution receipt](evaluation-panels/audio-generator-qualification-v57-execution.json)
bind the remotely published prospective freeze and both profiles.

The earlier v54 preparation and v55 smoke failures remain unchanged. Their
separate diagnostics led to explicit inventory-scope and identity-state checks;
they were not rewritten into passing runs. Package extraction and first language
asset download failures also remain in the setup records. The generation method
and weights were not modified to obtain the successful result.

The history review already finds 12 declared Kokoro voice profiles in earlier
source material. A different prompt through a previously tested voice is not
an independent new speaker. Complete the expanded source/voice/text overlap
audit before selecting the fresh cohort, and keep role-connected sources
together. Other generator runtime/asset reviews and the actual common-source
audio comparison remain pending. Serving defaults, protected audio calibration,
customer confidence, workflows, outreach, charges and GCP are unchanged.
