"""Exercise the exact queue CLI before long-running study scheduling."""
import json
import os
import subprocess
import sys
from pathlib import Path


def command(module, output):
    return [sys.executable, '-m', 'scripts.specialist_runtime_v4', '--validate-only', module,
        str(output), '/private/tmp/authored-project', '/private/tmp/authored-video-trial']


def test_actual_video_queue_command_preflights_without_launch_or_output(tmp_path):
    output = tmp_path / 'not-created'
    process = subprocess.run(command('scripts.prepare_video_specialists_v18', output),
        capture_output=True, text=True, check=True, env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])})
    receipt = json.loads(process.stdout)
    assert receipt['supported'] is True and receipt['module'] == 'scripts.prepare_video_specialists_v18'
    assert receipt['child_started'] is False and receipt['output_created'] is False
    assert receipt['arguments'] == ['/private/tmp/authored-project', '/private/tmp/authored-video-trial']
    assert not output.exists()


def test_audio_module_is_not_accidentally_accepted_by_video_runner(tmp_path):
    process = subprocess.run(command('scripts.run_audio_specialists_v15', tmp_path / 'not-created'),
        capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])})
    assert process.returncode == 2 and 'invalid choice' in process.stderr
    assert not (tmp_path / 'not-created').exists()
