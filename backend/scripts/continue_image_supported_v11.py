"""One sequential image experiment after the exact existing review/video queue.

Never restart native attempts, refit a frozen study, or admit a serving change.
An existing output/profile or changed byte commitment stops the queue.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts.continue_audio_specialists_v15 import same_process
from scripts.prepare_image_supported_v11 import sha
from scripts.run_image_supported_v11 import check

ROOT = Path(__file__).resolve().parents[2]


def command(main, trial, role, branch):
    return [sys.executable, '-m', 'scripts.specialist_runtime_v5', 'scripts.run_image_supported_v11',
        str(trial / (role + '-' + branch + '-profile')), 'score', str(main), str(trial),
        '--role', role, '--format', branch]


def run(main, audio, video, trial, pid, created):
    if not same_process(pid, created):
        raise ValueError('Exact identified review/video queue must still be live at launch')
    prior_command = psutil.Process(pid).cmdline()
    if (len(prior_command) < 3 or prior_command[1:3] != ['-m', 'scripts.continue_specialist_reviews_v3']
            or str(audio) not in prior_command or str(video) not in prior_command):
        raise ValueError('Not the identified previous audio-review/video-preparation job')
    check(trial)
    if (trial / 'pipeline-state.json').exists():
        raise FileExistsError('Preserve the existing image queue; no implicit retry')
    pins = {name: sha(ROOT / name) for name in ('backend/scripts/specialist_runtime_v5.py',
        'backend/scripts/continue_image_supported_v11.py')}
    preflight = subprocess.run([sys.executable, '-m', 'scripts.specialist_runtime_v5', '--validate-only',
        *command(main, trial, 'specialist_cal', 'png')[3:]], check=True, capture_output=True, text=True)
    validated = json.loads(preflight.stdout)
    if not validated['supported'] or validated['child_started'] or validated['output_created']:
        raise ValueError('Actual image native command failed no-worker preflight')
    state = {'schema': 'chipmunk.image-supported-pipeline/11', 'status': 'running',
        'stage': 'waiting_for_identified_review_and_video_preparation', 'completed': [],
        'freeze_sha256': sha(trial / 'freeze.json'), 'prior_process': {'pid': pid, 'created': created},
        'queue_code_sha256': pins, 'actual_command_preflight': validated, 'serving_admitted': False}
    started = time.monotonic()

    def save():
        state['updated_unix_seconds'] = time.time()
        temporary = trial / 'pipeline-state-next.json'
        with temporary.open('w') as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(trial / 'pipeline-state.json')
        print(json.dumps({'stage': state['stage'], 'status': state['status'], 'completed': state['completed']}), flush=True)

    def unchanged():
        check(trial)
        if ({name: sha(ROOT / name) for name in pins} != pins
                or sha(trial / 'freeze.json') != state['freeze_sha256']):
            raise ValueError('Queued image source/freeze changed; no automatic repair')

    def score_role(role):
        for branch in ('png', 'jpeg'):
            unchanged()
            state['stage'] = role + '-' + branch
            save()
            subprocess.run(command(main, trial, role, branch), check=True)
            state['completed'].append(state['stage'])
            save()

    save()
    try:
        while same_process(pid, created):
            if time.monotonic() - started > 7200:
                raise TimeoutError('Prior identified job remains live beyond this queue limit; inspect that same job')
            time.sleep(1)
        prior = json.loads((audio / 'postprocessing-v3-state.json').read_text())
        if (prior['status'] != 'completed'
                or sha(video / 'input-preparation.json') != prior['video_preparation_sha256']):
            raise ValueError('Prior review/video preparation did not complete with bound output')
        for role in ('specialist_cal', 'fusion_fit', 'fusion_cal'):
            score_role(role)
        unchanged()
        state['stage'] = 'fit_all_artifacts_and_points'
        save()
        subprocess.run([sys.executable, '-m', 'scripts.run_image_supported_v11', 'fit', str(main), str(trial)], check=True)
        state['completed'].append('all_artifacts_and_points_frozen')
        state['fit_complete_sha256_before_confirmation'] = sha(trial / 'fit-complete.json')
        score_role('confirmation')
        unchanged()
        if sha(trial / 'fit-complete.json') != state['fit_complete_sha256_before_confirmation']:
            raise ValueError('Calibrated parameters changed after confirmation')
        state['stage'] = 'review'
        save()
        subprocess.run([sys.executable, '-m', 'scripts.run_image_supported_v11', 'review', str(main), str(trial)], check=True)
        state.update(status='completed', stage='full_development_comparison', result_sha256=sha(trial / 'result.json'))
        save()
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('main', 'audio', 'video', 'trial'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--prior-pid', type=int, required=True)
    parser.add_argument('--prior-created', type=float, required=True)
    args = parser.parse_args()
    run(args.main.resolve(), args.audio.resolve(), args.video.resolve(), args.trial.resolve(), args.prior_pid, args.prior_created)
