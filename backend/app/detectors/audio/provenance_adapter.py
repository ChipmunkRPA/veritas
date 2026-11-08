"""Offline Content Credentials on supported original audio bitstreams."""
from dataclasses import replace

from app.detectors.image.adapters.provenance import C2PADetector, c2pa_result
from app.engine.detector import Detector, DetectorResult, Modality
from app.provenance.c2pa import sniff_mime


class AudioCredentialsDetector(Detector):
    spec = replace(C2PADetector.spec, id='prov.c2pa.audio', version='0.4.5', modality=Modality.AUDIO,
        name='Content Credentials on the audio file')

    def run(self, ctx):
        data = ctx.media.original_bytes()
        mime = sniff_mime(data)
        if mime not in ('audio/wav', 'audio/mp4', 'audio/mpeg', 'audio/flac'):
            return DetectorResult.not_applicable('Local Content Credentials validation is not supported for this audio container; no provider watermark was checked')
        return c2pa_result(ctx, data)
