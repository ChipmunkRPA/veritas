"""Six prespecified inputs replayed with production PCM on CPU; diagnostic only, no refitting."""
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

DATA=ROOT/'data/independent-evaluation/audio'


def main():
    from app.detectors.audio import content
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_speech_activity, load_voice
    from app.detectors.video.adapters.audio import GustkingVoiceDetector, PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.detectors.video.adapters.loaders import load_audio_classifier
    from app.engine.detector import RunContext, StageOutputs
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    selected=json.loads((DATA/'api-parity-expected.json').read_text())
    locked=json.loads((DATA/'run-lock.json').read_text())
    output={'purpose':'Post-evaluation engineering diagnostic on six prespecified inputs, consumed holdout; no new accuracy estimate or calibration fit.',
            'device':'cpu','dtype':'float32','cpu_threads':4,'decoder_source_sha256':hashlib.sha256((ROOT/'app/detectors/audio/decode.py').read_bytes()).hexdigest(),
            'original_scores_sha256':hashlib.sha256((DATA/'scores.jsonl').read_bytes()).hexdigest(),
            'calibration_unchanged_sha256':hashlib.sha256((DATA/'offline-calibration.json').read_bytes()).hexdigest(), 'inputs':[]}
    for item in selected:
        path=Path(item['path'])
        source_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        decoded=decode_audio(path,source_sha256=source_sha)
        import soundfile as sf
        previous,_=sf.read(path,dtype='float32')
        record=dict(id=item['id'],label=item['label'],condition=item['condition'],path=item['path'],mapping=decoded.mapping,
                    max_abs_pcm_delta=float(np.max(np.abs(decoded.pcm-previous))),old_float_pcm_expected=item['scores'],production_pcm_scores={})
        output['inputs'].append(record)
    for key,detector in [('old',GustkingVoiceDetector()),('new',PellaVoiceDetector())]:
        ids=[detector.MODEL_ID]+(['silero.speech_activity'] if key=='new' else [])
        for model_id in ids:
            paths=ensure_verified(get_registry().manifest.model(model_id),force=True)
            hashes=[]
            for path in paths:
                with path.open('rb') as stream:
                    sha=hashlib.file_digest(stream,'sha256').hexdigest()
                hashes.append({'path':str(path),'sha256':sha})
            assert hashes==locked['verified_local_weight_files'][model_id]
        models={detector.MODEL_ID:load_audio_classifier(detector.MODEL_ID,'cpu','float32')} if key=='old' else {
            detector.MODEL_ID:load_voice(detector.MODEL_ID,'cpu','float32'),
            'silero.speech_activity':load_speech_activity('silero.speech_activity','cpu','float32')}
        lease=SimpleNamespace(get=models.__getitem__)
        for item in output['inputs']:
            decoded=decode_audio(Path(item['path']),source_sha256=item['mapping']['source_sha256'])
            gate=content.classify(decoded.pcm).as_json()
            ctx=RunContext(analysis_id='production-pcm-diagnostic',media=None,profile='full',device='cpu',dtype='float32',lease=None,
                outputs=StageOutputs(audio=AudioTrack('analysed',decoded.pcm,len(decoded.pcm),gate=tuple(tuple(s) for s in gate['speech_spans']))),
                deadline=SimpleNamespace(check=lambda label='':None),exhibits=SimpleNamespace(series=lambda *a,**k:'diagnostic-series'),log=None)
            applicable=detector.applicable(ctx)
            if not applicable.ok:
                item['production_pcm_scores'][key]={'status':'not_applicable','reason':applicable.reason}
                continue
            ctx.lease=lease
            result=detector.run(ctx)
            item['production_pcm_scores'][key]={'status':result.status,'raw_unrounded':None if result.aggregate is None else float(result.aggregate.raw),
                'score':None if result.aggregate is None else round(float(result.aggregate.raw),6),
                'windows':[{'start_sample':round(u.quality['start_sample']),'end_sample':round(u.quality['end_sample']),
                            'raw_unrounded':float(u.raw),'score':round(float(u.raw),6)} for u in result.unit_scores]}
        del ctx,lease,models
        gc.collect()
    target=DATA/'production-pcm-diagnostic.json'
    assert not target.exists(), 'Never overwrite a completed diagnostic'
    target.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['label','condition','max_abs_pcm_delta','production_pcm_scores']} for r in output['inputs']],indent=2))


if __name__=='__main__':
    main()
