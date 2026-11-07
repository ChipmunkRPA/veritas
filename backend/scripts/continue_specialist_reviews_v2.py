"""Run source review and video input preparation after one identified audio job.

This is one-shot research orchestration. It neither restarts inference nor
admits a model. Video decoding starts only after the identified audio pipeline
has terminated with complete, source-bound results.
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
from scripts.inventory_video_specialists_v18 import sha

ROOT = Path(__file__).resolve().parents[2]
PINNED = ('backend/scripts/review_audio_specialists_v15.py', 'backend/scripts/audit_audio_specialists_v15.py',
    'backend/scripts/prepare_video_specialists_v18.py', 'backend/scripts/inventory_video_specialists_v18.py',
    'backend/scripts/prepare_face_interval_eval.py', 'backend/scripts/prepare_video_upgrade_eval.py',
    'backend/scripts/specialist_runtime_v3.py', 'backend/app/detectors/video/decode.py')


def run(main, audio, video, pid, created):
    state_path = audio / 'postprocessing-state.json'
    if state_path.exists():
        raise FileExistsError('Use the existing postprocessing handle; no silent retry')
    if not same_process(pid, created):
        raise ValueError('Exact prior audio continuation process must be live at launch')
    command = psutil.Process(pid).cmdline()
    if (len(command) < 3 or command[1:3] != ['-m', 'scripts.continue_audio_specialists_v15']
            or str(main) not in command or str(audio) not in command):
        raise ValueError('Not this exact previously launched audio pipeline')
    state = {'schema': 'chipmunk.specialist-postprocessing-pipeline/2', 'status': 'running',
        'stage': 'waiting_for_identified_audio_pipeline', 'prior_process': {'pid': pid, 'created': created},
        'implementation_sha256': sha(Path(__file__)), 'source_sha256': {name: sha(ROOT / name) for name in PINNED},
        'video_inventory_sha256': sha(video / 'source-inventory.json'), 'completed': [], 'serving_admitted': False}

    def save():
        state['updated_unix_seconds'] = time.time()
        temporary = audio / 'postprocessing-state-next.json'
        with temporary.open('w') as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write('\n')
        temporary.replace(state_path)
        print(json.dumps({'stage': state['stage'], 'status': state['status'], 'completed': state['completed']}), flush=True)

    def check_code():
        if ({name: sha(ROOT / name) for name in PINNED} != state['source_sha256']
                or sha(video / 'source-inventory.json') != state['video_inventory_sha256']):
            raise ValueError('Pinned source/review/input preparation changed while waiting')

    save()
    try:
        while same_process(pid, created):
            if time.time() - created > 18000:
                raise TimeoutError('Identified prior job exceeded bounded lifetime; inspect that same job')
            time.sleep(1)
        pipeline = json.loads((audio / 'pipeline-state.json').read_text())
        if pipeline['status'] != 'completed' or sha(audio / 'result.json') != pipeline['result_sha256']:
            raise ValueError('Audio did not terminate with the expected full result; preserve failed attempt')
        check_code()
        state['stage'] = 'audio_local_source_review'
        save()
        subprocess.run([sys.executable, '-m', 'scripts.review_audio_specialists_v15', str(audio),
            str(audio / 'independent-local-source-review.json')], check=True)
        state['completed'].append('audio_local_source_review')
        state['audio_review_sha256'] = sha(audio / 'independent-local-source-review.json')
        check_code()
        state['stage'] = 'video_source_input_preparation'
        save()
        subprocess.run([sys.executable, '-m', 'scripts.specialist_runtime_v3', 'scripts.prepare_video_specialists_v18',
            str(video / 'input-preparation-profile'), str(main), str(video)], check=True)
        state['completed'].append('video_source_input_preparation')
        state.update(status='completed', stage='reviewed_audio_and_prepared_video_inputs',
            video_preparation_sha256=sha(video / 'input-preparation.json'))
        save()
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__)
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main', type=Path)
    parser.add_argument('audio', type=Path)
    parser.add_argument('video', type=Path)
    parser.add_argument('--prior-pid', type=int, required=True)
    parser.add_argument('--prior-created', type=float, required=True)
    args = parser.parse_args()
    run(args.main, args.audio, args.video, args.prior_pid, args.prior_created)
