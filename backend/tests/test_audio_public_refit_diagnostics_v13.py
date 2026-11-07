"""Diagnostics stay separate from exact-refit acceptance and missing evidence."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_diagnostic_difference_and_missing_calibration_in_isolated_process():
    code = """
from scripts.diagnose_audio_public_refit_v13 import maximum_difference, own_guards
assert maximum_difference([0., 1.], [0., 1.125]) == .125
assert maximum_difference([], []) == 0.
for a, b in [([1.], [1., 2.]), ([float('nan')], [1.])]:
    try:
        maximum_difference(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid difference accepted')
rows = [{'label': label, 'group': 'g', 'raw': None, 'wavlm': None} for label in (0, 1)]
guards = own_guards(rows, {})
assert guards == {'0': [{'group': 'g', 'maximum': None, 'rows': 1}],
                  '1': [{'group': 'g', 'maximum': None, 'rows': 1}]}
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "backend", capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
