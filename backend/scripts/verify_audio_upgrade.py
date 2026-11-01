"""Engineering parity and bounded controls; this is not a detection-accuracy benchmark."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.detectors.audio.upgraded import load_voice, load_sound_events, load_speech_activity, prepare_voice_window
from app.third_party.kaldi.fbank import fbank

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'models/weights/audio-upgrade'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sound_reference_parity(model):
    import torch.nn as nn
    import torch.nn.functional as F
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank
    from torchlibrosa.augmentation import SpecAugmentation

    path = WEIGHTS / 'panns-models.py'
    original = ast.parse(path.read_text())
    classes = ast.Module(body=[node for node in original.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {'init_layer', 'init_bn', 'ConvBlock', 'Cnn10'}], type_ignores=[])
    scope = dict(torch=torch, nn=nn, F=F, np=np, Spectrogram=Spectrogram,
                 LogmelFilterBank=LogmelFilterBank, SpecAugmentation=SpecAugmentation)
    exec(compile(classes, str(path), 'exec'), scope)
    oracle = scope['Cnn10'](32000, 1024, 320, 64, 50, 14000, 527).eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    x = torch.from_numpy(np.random.default_rng(41).normal(0, .01, (1, 128000)).astype(np.float32))
    with torch.inference_mode():
        delta = (model(x)['clipwise_output'] - oracle(x)['clipwise_output']).abs().max().item()
    assert delta == 0
    return {'input': 'Fixed-seed 4-second 32kHz numerical control, identical tensor to both implementations',
            'max_abs_delta': delta,
            'method': 'Original publisher Cnn10 AST unchanged vs inference-only local extraction, same weights, eval mode'}


def main():
    model = load_voice('pella.voice_v2', 'cpu', 'float32').obj
    # Only the fbank path is used. Its official source has an unused torchaudio
    # import for MFCC; this stub avoids requiring an incompatible native wheel.
    fake = types.ModuleType('torchaudio')
    sys.modules['torchaudio'] = fake
    reference = module('official_kaldi', WEIGHTS / 'kaldi.py')
    compliance = types.ModuleType('torchaudio.compliance')
    fake.compliance = compliance
    compliance.kaldi = reference
    sys.modules['torchaudio.compliance'] = compliance
    sys.modules['torchaudio.compliance.kaldi'] = reference
    published = module('official_pella', WEIGHTS / 'pellav2_infer.py')
    sample_path = ROOT / 'backend/data/multimedia-background-demo/322b73cd-98f1-4551-b464-eedac9fddafa/assets/5da1e58f0952f646345fe2011baf94e520ee4fc8ba4bbf72ce48db255794b027.wav'
    assert hashlib.sha256(sample_path.read_bytes()).hexdigest() == sample_path.stem
    recording, rate = sf.read(sample_path, dtype='float32')
    assert rate == 16000 and recording.ndim == 1
    samples = {'silence_control': np.zeros(64000, np.float32),
               'noise_control': np.random.default_rng(304).normal(0, .025, 64000).astype(np.float32),
               'existing_speech_fixture': recording[:64000]}
    oracle = published.Detector.__new__(published.Detector)
    torch.nn.Module.__init__(oracle)
    oracle.backbone, oracle.layer_weights, oracle.head = model.backbone, model.layer_weights, model.head
    oracle.eval()
    sound = load_sound_events('panns.cnn10', 'cpu', 'float32').obj
    vad = load_speech_activity('silero.speech_activity', 'cpu', 'float32').obj
    folder = ROOT / 'backend/data/model-upgrade-review/audio'
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for name, raw in samples.items():
        path = folder / f'{name}.wav'
        sf.write(path, raw, 16000, subtype='FLOAT')
        ours = prepare_voice_window(raw)
        theirs = published.load_clip(path)
        assert np.array_equal(ours, theirs)
        x = torch.from_numpy(ours)[None]
        opts = dict(num_mel_bins=80, sample_frequency=16000, dither=0.0, window_type='povey')
        delta_fbank = (fbank(x * 32768, **opts) - reference.fbank(x * 32768, **opts)).abs().max().item()
        with torch.inference_mode():
            got, expected = model(x), oracle(x)
        delta_logit = (got - expected).abs().max().item()
        assert delta_fbank == 0 and delta_logit == 0
        results.append({'control': name, 'source_pcm_sha256': hashlib.sha256(raw.astype('<f4').tobytes()).hexdigest(),
            'source_fixture': str(sample_path.relative_to(ROOT)) if name == 'existing_speech_fixture' else 'generated integrity control',
            'samples': len(raw), 'preprocessing_exact': True, 'fbank_max_delta': delta_fbank,
            'logit_max_delta': delta_logit, 'raw_voice_score': torch.sigmoid(got).item(),
            'speech_spans': vad.spans(raw), 'possible_sounds': sound.predict(raw)})
    assert vad.spans(samples['silence_control']) == []
    assert vad.spans(samples['silence_control']) == []  # state reset after a different source
    result = {'purpose': 'Preprocessing/reference-forward parity and integration controls only; no accuracy or calibration estimate.',
        'device': 'cpu', 'torch': torch.__version__, 'controls': results,
        'sound_reference_forward': sound_reference_parity(sound.model),
        'weights': {name: hashlib.sha256((WEIGHTS/name).read_bytes()).hexdigest() for name in
                    ('pellav2_detector.pt', 'panns-Cnn10.pth', 'silero_vad.jit', 'kaldi.py', 'pellav2_infer.py')}}
    (folder / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
