from types import SimpleNamespace

import pytest

from scripts import audio_generator_qualification_v55 as qualification


def test_runtime_inventory_excludes_project_metadata_on_pythonpath(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    site = root / "lib/site-packages"
    site.mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    for directory, package in ((site / "owned-1.dist-info", "owned"), (project / "project.egg-info", "project")):
        directory.mkdir()
        filename = "METADATA" if directory.name.endswith("dist-info") else "PKG-INFO"
        (directory / filename).write_text(f"Metadata-Version: 2.1\nName: {package}\nVersion: 1.0\n")
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.setattr(qualification, "RUNTIME", root)
    monkeypatch.setattr(qualification.sysconfig, "get_path", lambda name: str(site))
    assert {d.metadata["Name"] for d in qualification.runtime_distributions()} == {"owned"}


def test_runtime_inventory_does_not_relax_its_prefix_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(qualification.sysconfig, "get_path", lambda name: str(tmp_path))
    with pytest.raises(ValueError, match="site-packages"):
        qualification.runtime_distributions()


def test_imported_model_origin_must_match_the_qualified_runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    owned = root / "module.py"
    owned.write_text("")
    project = tmp_path / "shadow.py"
    project.write_text("")
    monkeypatch.setattr(qualification, "RUNTIME", root)
    assert qualification.checked_module_origin(SimpleNamespace(__file__=str(owned))) == "module.py"
    with pytest.raises(ValueError, match="escaped"):
        qualification.checked_module_origin(SimpleNamespace(__file__=str(project)))
