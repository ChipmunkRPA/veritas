"""Preflight refuses changed sources and avoids whole-package execution."""
import ast
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import review_av_candidate_v12 as review


def test_changed_source_bytes_are_refused(tmp_path, monkeypatch):
    path = tmp_path / "source.py"
    path.write_bytes(b"original\n")
    monkeypatch.setattr(review, "FILES", {"source.py": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert review.verified(tmp_path, "source.py") == b"original\n"
    path.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="Changed inspected source"):
        review.verified(tmp_path, "source.py")


@pytest.mark.parametrize("fault", ["truncated", "length", "changed_payload"])
def test_bounded_header_tampering_is_refused(tmp_path, monkeypatch, fault):
    header = json.dumps({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode().ljust(17472)
    monkeypatch.setattr(review, "HEADER_SHA", hashlib.sha256(header).hexdigest())
    data = struct.pack("<Q", 17472) + header
    path = tmp_path / "weight-header.bin"
    path.write_bytes(data)
    assert review.checked_header(tmp_path)["a"]["shape"] == [1]
    if fault == "truncated":
        changed = data[:-1]
    elif fault == "length":
        changed = struct.pack("<Q", 17471) + data[8:]
    else:
        changed = data.replace(b'"F32"', b'"I32"')
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="Exact original tensor header"):
        review.checked_header(tmp_path)


def test_selected_definition_loading_does_not_execute_top_level_code():
    source = b"import nonexistent_remote_package\nraise RuntimeError('top level ran')\ndef selected():\n return 7\n"
    namespace = review.definitions(source, {"selected"}, {})
    assert namespace["selected"]() == 7
    assert "nonexistent_remote_package" not in sys.modules
    with pytest.raises(ValueError, match="Complete inspected definitions"):
        review.definitions(source, {"selected", "absent"}, {})


def test_pinned_acquisition_urls_cannot_follow_a_symbolic_revision():
    for name in review.FILES:
        url = review.source_url(name)
        assert url.startswith("https://") and "main" not in url and "?" not in url
        assert review.GITHUB in url or review.HUB in url
    assert review.source_url("hf/config.json").endswith("/models/config.json")


def test_existing_receipt_is_refused_before_source_or_neural_work(tmp_path):
    receipt = tmp_path / "existing.json"
    receipt.write_text("unchanged\n")
    result = subprocess.run([sys.executable, str(Path(review.__file__)), "review", str(tmp_path / "absent"), str(receipt)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2 and "new output path" in result.stderr
    assert receipt.read_text() == "unchanged\n"


def test_network_access_is_confined_to_explicit_acquisition():
    source = ast.parse(Path(review.__file__).read_text())
    functions = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef)}
    def calls(node):
        return [n for n in ast.walk(node) if isinstance(n, ast.Attribute) and n.attr == "urlopen"]

    assert calls(functions["acquire"])
    assert all(not calls(node) for name, node in functions.items() if name != "acquire")
