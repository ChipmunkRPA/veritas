"""Preflight the exact queued native image command without starting a worker."""
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.continue_image_supported_v11 import command


def test_image_queue_command_is_accepted_with_complete_nested_arguments(tmp_path):
    main, trial = tmp_path / 'project', tmp_path / 'trial'
    actual = command(main, trial, 'fusion_cal', 'jpeg')
    result = subprocess.run([*actual[:3], '--validate-only', *actual[3:]], check=True,
        capture_output=True, text=True, env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])})
    receipt = json.loads(result.stdout)
    assert receipt['module'] == 'scripts.run_image_supported_v11' and receipt['seconds_limit'] == 900
    assert receipt['arguments'] == ['score', str(main), str(trial), '--role', 'fusion_cal', '--format', 'jpeg']
    assert receipt['supported'] and not receipt['child_started'] and not receipt['output_created']
    assert not list(tmp_path.iterdir())


def test_image_runner_does_not_accept_video_or_excess_duration(tmp_path):
    base = [sys.executable, '-m', 'scripts.specialist_runtime_v5', '--validate-only']
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    for arguments in (['scripts.prepare_video_specialists_v18', str(tmp_path / 'profile')],
            ['--seconds', '901', 'scripts.run_image_supported_v11', str(tmp_path / 'profile')]):
        result = subprocess.run([*base, *arguments], capture_output=True, text=True, env=env)
        assert result.returncode == 2
    assert not list(tmp_path.iterdir())
