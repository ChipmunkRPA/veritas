"""Wait for the identified live role, then run the remaining frozen stages once."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts import prepare_audio_specialists_v15 as source
from scripts import run_audio_specialists_v15 as scoring


def same_process(pid, created):
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - created) < 1e-5 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def continue_once(main, trial, worker_pid, worker_created, monitor_pid, monitor_created):
    status_path = trial / 'pipeline-state.json'
    if status_path.exists():
        raise FileExistsError('Use the existing pipeline handle; no restart or duplicated stage')
    scoring.check(trial)
    if not same_process(worker_pid, worker_created) or not same_process(monitor_pid, monitor_created):
        raise ValueError('Both specific prior worker and resource monitor must be confirmed live at launch')
    command = psutil.Process(worker_pid).cmdline()
    if ('scripts.run_audio_specialists_v15' not in command or 'score-role' not in command
            or '--role' not in command or command[command.index('--role') + 1] != 'specialist_cal'
            or str(trial) not in command or psutil.Process(worker_pid).ppid() != monitor_pid):
        raise ValueError('The live process is not this exact first-role study')
    state = {'schema': 'chipmunk.audio-specialists-pipeline/15', 'status': 'running',
        'stage': 'waiting_for_identified_specialist_cal', 'freeze_sha256': source.sha(trial / 'freeze.json'),
        'orchestrator_sha256': source.sha(Path(__file__)),
        'prior_worker': {'pid': worker_pid, 'created': worker_created},
        'prior_monitor': {'pid': monitor_pid, 'created': monitor_created},
        'completed': [], 'serving_admitted': False}

    def save():
        state['updated_unix_seconds'] = time.time()
        temporary = trial / 'pipeline-state-next.json'
        with temporary.open('w') as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(status_path)
        print(json.dumps({'stage': state['stage'], 'status': state['status'], 'completed': state['completed']}), flush=True)

    def role_receipt(role, folder):
        profile = json.loads((trial / folder / 'profile.json').read_text())
        if profile['child_exit_code'] != 0 or profile['guard'] is not None:
            raise ValueError('Failed/guarded prior role is retained; do not advance or restart')
        scoring.responses(trial, role)
        state['completed'].append(role)
        save()

    def execute(stage, role=None, folder=None):
        state['stage'] = stage if role is None else role
        save()
        arguments = [stage, str(main), str(trial)]
        if role:
            arguments += ['--role', role]
        command = [sys.executable, '-m', 'scripts.run_audio_specialists_v15', *arguments]
        if folder:
            command = [sys.executable, '-m', 'scripts.specialist_runtime_v3',
                'scripts.run_audio_specialists_v15', str(trial / folder), *arguments]
        subprocess.run(command, check=True)

    save()
    try:
        while same_process(worker_pid, worker_created) or same_process(monitor_pid, monitor_created):
            if time.time() - monitor_created > 3660:
                raise TimeoutError('Identified prior role exceeded its own bounded lifetime; inspect that same job')
            time.sleep(1)
        role_receipt('specialist_cal', 'specialist-cal-profile')
        for role in ('fusion_fit', 'fusion_cal'):
            folder = role + '-profile'
            execute('score-role', role, folder)
            role_receipt(role, folder)
        execute('fit')
        complete = json.loads((trial / 'fit-complete.json').read_text())
        for arm in scoring.ARMS:
            for suffix, key in (('artifact', 'artifacts'), ('operating-point', 'operating_points')):
                if source.sha(trial / (arm + '-' + suffix + '.json')) != complete[key][arm]:
                    raise ValueError('Every calibration and operating-point hash must be frozen before confirmation')
        state['completed'].append('all_calibration_artifacts_and_points_frozen')
        state['fit_complete_sha256_before_confirmation'] = source.sha(trial / 'fit-complete.json')
        save()
        execute('score-role', 'confirmation', 'confirmation-profile')
        role_receipt('confirmation', 'confirmation-profile')
        execute('review')
        state.update(status='completed', stage='four_arm_result_written', result_sha256=source.sha(trial / 'result.json'))
        save()
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('trial', type=Path)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--worker-created', type=float, required=True)
    parser.add_argument('--monitor-pid', type=int, required=True)
    parser.add_argument('--monitor-created', type=float, required=True)
    args = parser.parse_args()
    continue_once(args.main, args.trial, args.worker_pid, args.worker_created, args.monitor_pid, args.monitor_created)
