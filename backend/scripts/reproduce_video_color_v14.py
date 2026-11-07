"""Public synthetic colour-path experiment; no private media or neural weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from scripts.diagnose_video_color_v14 import cv_frame, ffmpeg_branches, file_sha


def reproduce(output):
    if output.exists():
        raise ValueError('New reproduction receipt required')
    colors = np.array([[230, 30, 40], [30, 220, 40], [40, 30, 230], [220, 190, 30],
                       [30, 190, 210], [190, 40, 190], [90, 140, 180], [200, 120, 60]], dtype=np.uint8)
    chart = np.repeat(np.repeat(colors[None], 64, axis=0), 16, axis=1)
    source = np.stack([np.roll(chart, i * 2, axis=1) for i in range(12)])
    width, height, index = 128, 64, 3
    with tempfile.TemporaryDirectory(prefix='verify-colour-source-') as scratch:
        path = Path(scratch) / 'synthetic-bt709.mkv'
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}', '-r', '8', '-i', 'pipe:0', '-vf',
            'scale=in_range=pc:out_color_matrix=bt709:out_range=tv,format=yuv420p',
            '-filter_threads', '1', '-frames:v', '12', '-c:v', 'ffv1', '-level', '3', '-threads', '1',
            '-colorspace', 'bt709', '-color_range', 'tv', '-color_primaries', 'bt709', '-color_trc', 'bt709', str(path)]
        subprocess.run(command, input=source.tobytes(), capture_output=True, check=True, timeout=60)
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_streams',
            '-show_entries', 'stream=color_space,color_range,pix_fmt:frame=color_space,color_range',
            '-of', 'json', str(path)], capture_output=True, check=True, timeout=60)
        meta = json.loads(probe.stdout)
        if (len(meta['frames']) != 12 or meta['streams'][0]['color_space'] != 'bt709'
                or meta['streams'][0]['color_range'] != 'tv'
                or any(f['color_space'] != 'bt709' or f['color_range'] != 'tv' for f in meta['frames'])):
            raise ValueError('Complete synthetic BT709 limited-range declaration required')
        cv = cv_frame(path, index)
        images = ffmpeg_branches(path, index, width, height)
        digests = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
        cv_sha = hashlib.sha256(cv.tobytes()).hexdigest()
        conditions = {'metadata_default_equals_explicit_bt709': digests[0] == digests[2],
                      'opencv_equals_explicit_bt601': cv_sha == digests[1],
                      'opencv_differs_from_metadata_default': cv_sha != digests[0]}
        result = {'schema': 'chipmunk.synthetic-video-colour-reproduction/14',
            'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
            'opencv': cv2.__version__, 'opencv_ffmpeg_build': [line.strip() for line in cv2.getBuildInformation().splitlines()
                if any(name in line for name in ('FFMPEG:', 'avcodec:', 'avformat:', 'avutil:', 'swscale:'))],
            'ffmpeg_version': subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0],
            'ffmpeg_binary_sha256': file_sha(Path(shutil.which('ffmpeg'))),
            'experiment_code_sha256': file_sha(Path(__file__)),
            'renderer_code_sha256': file_sha(Path(__file__).with_name('diagnose_video_color_v14.py')),
            'synthetic_rgb_input_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
            'synthetic_file_sha256': file_sha(path), 'frames_created': 12, 'audited_frame': index,
            'stream_metadata': meta['streams'][0], 'opencv_rgb_sha256': cv_sha,
            'ffmpeg_rgb_sha256': dict(zip(('metadata_default', 'bt601', 'bt709'), digests, strict=True)),
            'conditions': conditions, 'hypothesis_reproduced': all(conditions.values()),
            'neural_forwards': 0, 'private_inputs_used': False, 'serving_changed': False,
            'scope': 'Synthetic controlled renderer behaviour on this actual environment; not real-clip replay, a global OpenCV claim, detector accuracy, neural/native model verification or outside human review. Generated container hashes may differ across runs/platforms; installed binary hashes are environment-specific.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))
    if not result['hypothesis_reproduced']:
        raise ValueError('Specified colour-path hypothesis did not reproduce; receipt retained, no requirement relaxed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    reproduce(parser.parse_args().output)
