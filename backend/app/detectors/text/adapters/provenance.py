"""Offline whole-document Content Credentials for qualified PDF files."""

from dataclasses import replace

from app.detectors.image.adapters.provenance import C2PADetector, c2pa_result
from app.engine.detector import Detector, DetectorResult, Modality
from app.provenance.pdf_credentials import MIME

ID = "prov.c2pa.pdf"


class PdfCredentialsDetector(Detector):
    spec = replace(C2PADetector.spec, id=ID, version="1.0.0", modality=Modality.TEXT,
                   name="Content Credentials on the PDF")

    def run(self, ctx):
        data = ctx.media.original_bytes()
        if bytes(data[:5]) != b"%PDF-":
            return DetectorResult(status="not_applicable", reason="Whole-document Content Credentials are checked on supported PDFs",
                                  metrics={"unsupported_container": 1.0})
        return c2pa_result(ctx, data, mime_override=MIME, require_pdf_document=True)
