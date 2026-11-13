"""Public-safe report generations; older signed bytes are never rewritten."""
from app.reports import engine

# These generations already used the public-safe headline and check labels.
_FACE_SCOPE_LEGACY_TEMPLATES = {
    "veritas-report-2.0.r83", "veritas-report-2.0.r84", "veritas-report-2.0.r85", "veritas-report-2.0.r87", "veritas-report-2.0.r88",
    "veritas-report-2.0.r89", "veritas-report-2.0.r90", "veritas-report-2.0.r91", "veritas-report-2.0.r92", "veritas-report-2.0.r94",
    "veritas-report-2.0.r95", "veritas-report-2.0.r96", "veritas-report-2.0.r97", "veritas-report-2.0.r98", "veritas-report-2.0.r99",
    "veritas-report-2.0.r100",
}
_PREVIOUS_PUBLIC_TEMPLATES = _FACE_SCOPE_LEGACY_TEMPLATES | {
    "veritas-report-2.0.r101", "veritas-report-2.0.r102", "veritas-report-2.0.r103", "veritas-report-2.0.r104",
    "veritas-report-2.0.r105", "veritas-report-2.0.r106", "veritas-report-2.0.r107", "veritas-report-2.0.r108", "veritas-report-2.0.r109",
    "veritas-report-2.0.r110", "veritas-report-2.0.r111", "veritas-report-2.0.r112", "veritas-report-2.0.r113",
    "veritas-report-2.0.r114", "veritas-report-2.0.r115", "veritas-report-2.0.r116",
}


def downloadable_template(template: str, kind: str | None = None) -> bool:
    # Earlier headlines could promote inconclusive results or overstate video
    # coverage. Generate a new public copy using the corrected presentation.
    return template == engine.TEMPLATE_SET or template in _PREVIOUS_PUBLIC_TEMPLATES


def package_template(template: str, kind: str | None = None) -> bool:
    return downloadable_template(template, kind)


def needs_legacy_face_scope_check(template: str, kind: str) -> bool:
    return kind == "image" and template in _FACE_SCOPE_LEGACY_TEMPLATES


def legacy_face_scope_safe(frozen) -> bool:
    from app.reports.public_view import is_named_face_scope_limitation

    if not needs_legacy_face_scope_check(frozen.report.template_set, frozen.analysis.kind):
        return True
    return not any(is_named_face_scope_limitation(caveat) for caveat in frozen.result.get("caveats") or [])
