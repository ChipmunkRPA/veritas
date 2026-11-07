"""Model-only alternative using the unchanged consumed writing design and gates."""
from __future__ import annotations

import copy

from scripts import writing_focus_v3 as previous

INPUT_SHA = previous.INPUT_SHA
BASELINE_REVISION = previous.BASELINE_REVISION
BASELINE_ASSETS = previous.BASELINE_ASSETS
CANDIDATE_REVISION = 'e047d9c56f51a7e5c5ef5b0dd556f519bb9dd624'
CANDIDATE_ASSETS = {
    'README.md': '328d6508b1a135e8d1559e25efdf0cb7dc9f00ba78663fc6904003ff0ab73891',
    'BASE_README.md': '40539466fa1117be4693b8f1fdfdac81fac3c5db472ff57bc05d013d48eafa82',
    'config.json': 'a6225f6c55e0fb92ccb6a53832714993a5eeceff0a4ac89e90f803de8e06a742',
    'model.safetensors': '2ae8059c621795fc490353bbd920bc3672e6eaf8ad66ddb7c015b4ecb0129218',
    'tokenizer.json': '6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30',
    'tokenizer_config.json': 'c16f1b07f93d05be753ec0acf8cb511d895a8bdb09669f6b156df81e3f96c2eb',
    'special_tokens_map.json': 'ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049',
}
METHOD = copy.deepcopy(previous.METHOD)
METHOD.update(schema='chipmunk.writing-modernbert-method/4',
    candidate_response='Standard ModernBERT CPUfloat32/eager, softmax index1 from pinned publisher inference. Stored multilabel problem_type retained; no labels supplied, temperature, fitted calibration or publisher operating point.',
    candidate_repository='GeorgeDrayson/modernbert-ai-detection-raid-mage',
    native_context_limit='Use the same512-token complete-overflow processing as baseline; native8192 position capacity is not a long-document validation claim.',
    scope='Consumed mixed-author development, model/context2x2 and retained-character provenance. No unused authors, unassisted-human false-alarm estimate, word-origin proof, calibrated confidence or serving admission.')

# Keep the original input/denominator/arithmetic/continuation functions unchanged.
digest = previous.digest
text_sha = previous.text_sha
finite_score = previous.finite_score
validate_inputs = previous.validate_inputs
counts = previous.counts
rates = previous.rates
development_gates = previous.development_gates
