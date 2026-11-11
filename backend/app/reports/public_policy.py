"""Public-safe report generations; older signed bytes are never rewritten."""
from app.reports import engine

# This generation already used the public-safe headline and check labels.
_PREVIOUS_PUBLIC_TEMPLATE = "veritas-report-2.0.r83"


def downloadable_template(template: str, kind: str | None = None) -> bool:
    # Earlier headlines could promote inconclusive results or overstate video
    # coverage. Generate a new public copy using the corrected presentation.
    return template in {engine.TEMPLATE_SET, _PREVIOUS_PUBLIC_TEMPLATE}


def package_template(template: str, kind: str | None = None) -> bool:
    return downloadable_template(template, kind)
