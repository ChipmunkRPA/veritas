"""Exercise the portable reviewer in isolated processes, without app imports."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "backend/scripts/review_audio_public_portable_v13.py"


def isolated(body):
    setup = f"import runpy\nm = runpy.run_path({str(SCRIPT)!r}, run_name='portable_test')\n"
    result = subprocess.run(
        [sys.executable, "-c", setup + body], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", ["torch", "torchaudio", "transformers", "app", "fastapi", "cv2", "PIL", "matplotlib"])
def test_app_and_neural_imports_are_blocked(module):
    isolated(f"""
import importlib, sys
assert not (m['BLOCKED'] & {{name.split('.')[0] for name in sys.modules}})
try:
    importlib.import_module({module!r})
except ImportError as exc:
    assert 'Portable numerical review forbids' in str(exc)
else:
    raise AssertionError('Forbidden dependency loaded')
""")


@pytest.mark.parametrize("arm", ["standalone", "complement"])
def test_probability_matches_hand_computed_margin(arm):
    isolated(f"""
import math
width, arm = 4608, {arm!r}
factor = math.sqrt(16 / width)
vector, center, scale = [0.] * width, [0.] * width, [1.] * width
vector[0], vector[1], center[0], scale[0] = 3., 4., 1., 2.
coefficients = [2., -.25] + [0.] * (width - 2) + [-.4]
expected_margin = factor * (2. * (3. - 1.) / 2. - .25 * 4.) - .4
if arm == 'complement':
    coefficients = [.7] + coefficients
    expected_margin += .7 * math.log(3.)
fitted = {{'arm': arm, 'coefficients': coefficients,
          'transform': {{'center': center, 'scale': scale, 'variance_factor': factor}}}}
actual = m['probability']({{'raw': .75, 'wavlm': vector}}, fitted)
assert math.isclose(actual, 1 / (1 + math.exp(-expected_margin)), abs_tol=1e-14)
""")


def test_unassessed_and_invalid_features_cannot_acquire_a_probability():
    isolated("""
import copy, math
assert m['probability']({'raw': None, 'wavlm': None}, {}) is None
fitted = {'arm': 'complement', 'coefficients': [1.] + [0.] * 4609,
          'transform': {'center': [0.] * 4608, 'scale': [1.] * 4608,
                        'variance_factor': math.sqrt(16 / 4608)}}
row = {'raw': .8, 'wavlm': [0.] * 4608}
cases = [({'raw': None, 'wavlm': [0.] * 4608}, fitted),
         ({'raw': float('nan'), 'wavlm': [0.] * 4608}, fitted),
         ({'raw': .8, 'wavlm': [0.] * 4607}, fitted),
         ({'raw': .8, 'wavlm': [float('inf')] * 4608}, fitted)]
for field, value in [('scale', [0.] * 4608), ('center', [float('nan')] * 4608),
                     ('variance_factor', 1.)]:
    bad = copy.deepcopy(fitted)
    bad['transform'][field] = value
    cases.append((row, bad))
bad = copy.deepcopy(fitted)
bad['coefficients'][0] = -1.
cases.append((row, bad))
for value, mapping in cases:
    try:
        m['probability'](value, mapping)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete or invalid mapping accepted')
""")


def test_missing_or_unstable_band_cannot_become_an_original_verdict():
    isolated("""
guards = {str(c): [{'group': str(i), 'maximum': .1, 'rows': 1} for i in range(19)] for c in (0, 1)}
assert m['initial'](.01, guards) == 'original'
assert m['final'](.01, .01, .01, .01, guards)['decision'] == 'original'
assert m['final'](None, None, None, None, guards)['decision'] == 'inconclusive'
for band_p, band_raw in [(None, .01), (.01, None), (.5, .01), (.01, .5)]:
    result = m['final'](.01, band_p, .01, band_raw, guards)
    assert result['decision'] == 'inconclusive' and result['original_stability_veto']
""")


def test_transport_or_object_change_is_rejected_before_replay():
    isolated("""
import hashlib, json, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / 'packet.json'
    path.write_text('{"value": 1}\\n')
    original = m['digest'](path)
    assert m['load'](path, original) == {'value': 1}
    path.write_text('{"value": 2}\\n')
    try:
        m['load'](path, original)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed transfer accepted')
value = {'signed': -0., 'kept': True}
commitment = hashlib.sha256((json.dumps(value, indent=2) + '\\n').encode()).hexdigest()
m['exact_object'](value, commitment)
for changed in [{'signed': 0., 'kept': True}, {'signed': -0.}, {'signed': -0., 'kept': 1}]:
    try:
        m['exact_object'](changed, commitment)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed object commitment accepted')
assert not m['same']({'flag': True}, {'flag': 1})
assert not m['same']({'denominator': 10}, {})
""")


def test_frozen_fitter_has_no_neural_execution_and_refuses_changed_source():
    isolated("""
import copy, sys
freeze = m['load'](m['PROJECT'] / 'docs/evaluation-panels/audio-v13-wavlm-freeze.json', m['FREEZE_SHA'])
fitter = m['frozen_fitter'](freeze)
assert all(callable(fitter[key]) for key in ('feature', 'design', 'fit', 'select'))
assert not (m['BLOCKED'] & {name.split('.')[0] for name in sys.modules})
changed = copy.deepcopy(freeze)
changed['lock']['implementation_sha256']['backend/scripts/audio_wavlm_response_v13.py'] = '0' * 64
try:
    m['frozen_fitter'](changed)
except ValueError:
    pass
else:
    raise AssertionError('Changed declared fitter accepted')
""")
