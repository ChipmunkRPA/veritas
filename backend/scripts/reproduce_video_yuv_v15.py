"""Fixed native-YUV experiment, separate from the completed negative RGB study."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from scripts.diagnose_video_color_v14 import cv_frame, ffmpeg_branches, file_sha

WIDTH, HEIGHT, FRAMES, FPS, INDEX = 128, 64, 12, 8, 3
Y = (80, 145, 41, 210, 170, 110, 125, 190)
U = (90, 54, 220, 40, 160, 180, 110, 70)
V = (220, 34, 110, 145, 40, 190, 150, 175)
SOURCE_SHA = '2025fc4531fbd91e0148b7b26b17181157c0e80ede8b1361d9c5a561f9a02cf4'


def native_samples():
    """Self-authored legal-range planar samples; no RGB-to-YUV conversion."""
    frames = []
    for index in range(FRAMES):
        planes = []
        for values, width, height, shift in ((Y, WIDTH, HEIGHT, index * 2),
                (U, WIDTH // 2, HEIGHT // 2, index), (V, WIDTH // 2, HEIGHT // 2, index)):
            row = bytes(value for value in values for _ in range(width // len(values)))
            row = row[-shift:] + row[:-shift] if shift else row
            planes.append(row * height)
        frames.append(b''.join(planes))
    return b''.join(frames)


def validate_source(expected, recovered, metadata):
    if len(expected) != WIDTH * HEIGHT * 3 // 2 * FRAMES or recovered != expected:
        raise ValueError('All authored native YUV samples must recover exactly')
    streams, frames = metadata.get('streams', []), metadata.get('frames', [])
    if len(streams) != 1 or len(frames) != FRAMES:
        raise ValueError('Exactly one video stream and all12 frames required')
    stream = streams[0]
    for item in [stream, *frames]:
        if any(item.get(k) != v for k, v in {'width': WIDTH, 'height': HEIGHT,
                'pix_fmt': 'yuv420p', 'color_space': 'bt709', 'color_range': 'tv'}.items()):
            raise ValueError('Native dimensions, format and limited BT709 tags must agree')
    time_base = Fraction(stream['time_base'])
    if any(Fraction(frame['best_effort_timestamp']) * time_base != Fraction(i, FPS)
           for i, frame in enumerate(frames)):
        raise ValueError('Every original frame clock must equal its authored timestamp')


def compare_renderers(cv_rgb, images):
    expected = (HEIGHT, WIDTH, 3)
    if (cv_rgb.shape != expected or cv_rgb.dtype != np.uint8 or len(images) != 3
            or any(i.shape != expected or i.dtype != np.uint8 for i in images)):
        raise ValueError('Complete original-size renderer outputs required')
    def sha(a):
        return hashlib.sha256(a.tobytes()).hexdigest()
    hashes = dict(zip(('metadata_default', 'bt601', 'bt709'), map(sha, images), strict=True))
    cv_hash = sha(cv_rgb)
    return {'opencv_rgb_sha256': cv_hash, 'ffmpeg_rgb_sha256': hashes,
        'opencv_matches': [name for name, digest in hashes.items() if digest == cv_hash],
        'metadata_default_equals_bt709': hashes['metadata_default'] == hashes['bt709'],
        'explicit_matrices_differ': hashes['bt601'] != hashes['bt709']}


def run(output):
    if output.exists():
        raise ValueError('New receipt required; preserve all prior experiments')
    source = native_samples()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError('Frozen authored native source commitment changed')
    with tempfile.TemporaryDirectory(prefix='verify-native-yuv-') as scratch:
        path = Path(scratch) / 'authored-yuv.mkv'
        encode = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
            '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', 'pipe:0', '-map', '0:v:0',
            '-filter_threads', '1', '-frames:v', str(FRAMES), '-c:v', 'ffv1', '-level', '3',
            '-threads', '1', '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_range', 'tv',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', str(path)]
        subprocess.run(encode, input=source, capture_output=True, check=True, timeout=60)
        decoded = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(path),
            '-map', '0:v:0', '-fps_mode', 'passthrough', '-threads', '1', '-pix_fmt', 'yuv420p',
            '-f', 'rawvideo', 'pipe:1'], capture_output=True, check=True, timeout=60).stdout
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
            '-show_streams', '-show_entries',
            'stream=width,height,pix_fmt,color_space,color_range,time_base:frame=width,height,pix_fmt,color_space,color_range,best_effort_timestamp',
            '-of', 'json', str(path)], capture_output=True, check=True, timeout=60)
        metadata = json.loads(probe.stdout)
        validate_source(source, decoded, metadata)
        compared = compare_renderers(cv_frame(path, INDEX), ffmpeg_branches(path, INDEX, WIDTH, HEIGHT))
        result = {'schema': 'chipmunk.fixed-native-yuv-experiment/15',
            'python': platform.python_version(), 'platform': platform.platform(), 'numpy': np.__version__,
            'opencv': cv2.__version__, 'opencv_ffmpeg_build': [line.strip() for line in cv2.getBuildInformation().splitlines()
                if any(name in line for name in ('FFMPEG:', 'avcodec:', 'avformat:', 'avutil:', 'swscale:'))],
            'ffmpeg_version': subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0],
            'ffmpeg_binary_sha256': file_sha(Path(shutil.which('ffmpeg'))),
            'experiment_code_sha256': file_sha(Path(__file__)),
            'renderer_code_sha256': file_sha(Path(__file__).with_name('diagnose_video_color_v14.py')),
            'authored_yuv_sha256': hashlib.sha256(source).hexdigest(), 'native_samples_bytes': len(source),
            'recovered_yuv_sha256': hashlib.sha256(decoded).hexdigest(), 'native_samples_exact': True,
            'container_sha256': file_sha(path), 'frames': FRAMES, 'audited_frame': INDEX,
            'source_timestamps_exact': True, 'source_stream': metadata['streams'][0], **compared,
            'neural_forwards': 0, 'private_inputs_used': False, 'original_experiment_changed': False,
            'scope': 'Fixed authored YUV and exact native recovery isolate encoding conversion. Renderer equality and mismatch are observations, not pass conditions, detector validation, a single-library causal claim or original native-gate repair.'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
