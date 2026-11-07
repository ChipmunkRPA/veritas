"""Error evidence must retain original bytes without persisting decoded audio."""
import hashlib
import json
import subprocess

from scripts.repeat_audio_separation_v14 import failure_diagnostics


def test_decoder_cause_retains_complete_stderr_but_never_stdout(tmp_path):
    stderr = b'decoder failed\xff\n' * 10000
    decoder = subprocess.CalledProcessError(252, ['ffmpeg'], output=b'RAW_AUDIO_DO_NOT_SAVE', stderr=stderr)
    wrapped = ValueError('verified timeline failed')
    wrapped.__cause__ = decoder
    failure_diagnostics(wrapped, tmp_path)
    receipt = json.loads((tmp_path / 'failure-diagnostics.json').read_text())
    assert [r['error_type'] for r in receipt['exception_chain']] == ['ValueError', 'CalledProcessError']
    row = receipt['exception_chain'][1]
    assert row['returncode'] == 252 and row['stderr_bytes'] == len(stderr)
    assert row['stderr_sha256'] == hashlib.sha256(stderr).hexdigest()
    assert (tmp_path / row['stderr_file']).read_bytes() == stderr
    assert not receipt['raw_audio_or_stdout_saved']
    assert all(b'RAW_AUDIO_DO_NOT_SAVE' not in p.read_bytes() for p in tmp_path.iterdir())


def test_nondecode_error_does_not_invent_decoder_evidence(tmp_path):
    failure_diagnostics(RuntimeError('model failed'), tmp_path)
    receipt = json.loads((tmp_path / 'failure-diagnostics.json').read_text())
    assert receipt['exception_chain'] == [{'error_type': 'RuntimeError'}]
    assert sorted(p.name for p in tmp_path.iterdir()) == ['failure-diagnostics.json']
