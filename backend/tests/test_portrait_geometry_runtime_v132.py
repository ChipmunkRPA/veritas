import zipfile

import pytest

from scripts import portrait_geometry_runtime_v132 as runtime


def test_installed_manifest_checks_native_libraries_and_sources_except_regenerated_record(tmp_path):
    wheel = tmp_path / "authored.whl"
    site = tmp_path / "site"
    files = {
        "pkg/__init__.py": b"authored source",
        "pkg/native.dylib": b"authored binary",
        "pkg-1.dist-info/METADATA": b"Version: 1",
        "pkg-1.dist-info/RECORD": b"original record",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
            target = site / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    (site / "pkg-1.dist-info/RECORD").write_text("pip generated record")
    assert len(runtime.installed_manifest([wheel], site)) == 3
    (site / "pkg/native.dylib").write_bytes(b"changed binary")
    with pytest.raises(ValueError, match="differs"):
        runtime.installed_manifest([wheel], site)


@pytest.mark.parametrize("name", ["../escape.py", "/escape.py", "pkg\\escape.py", "pkg.data/platlib/file.py"])
def test_unsafe_or_unplanned_wheel_install_layout_is_rejected(tmp_path, name):
    wheel = tmp_path / "authored.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(name, b"x")
    with pytest.raises(ValueError):
        runtime.installed_manifest([wheel], tmp_path / "site")


def test_frozen_runtime_dispatch_cannot_launch_a_neural_stage(monkeypatch):
    calls = []
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    runtime.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", "out", "0"])
    assert calls[0][3] == "scripts.portrait_geometry_runtime_v132"
    with pytest.raises(ValueError):
        runtime.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored"])
