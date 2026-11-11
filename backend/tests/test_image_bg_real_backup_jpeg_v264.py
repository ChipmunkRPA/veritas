import pytest

from scripts import image_bg_real_backup_jpeg_v264 as backup


def test_exact_fresh_jpeg_batch_plan():
    batches = [backup.tasks(phase) for phase in range(backup.PHASES)]
    assert [len(rows) for rows in batches] == [25, 25, 10]
    assert [row for rows in batches for row in rows] == [
        (source, kind) for source in range(12) for kind in backup.KINDS
    ]
    assert backup.QUALITIES == (75, 90)
    with pytest.raises(ValueError, match="One planned diagnostic batch required"):
        backup.tasks(3)


def test_scoring_requires_published_preparation(monkeypatch, tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("unrelated public result")
    monkeypatch.setattr(backup, "PUBLIC", freeze)
    with pytest.raises(ValueError, match="Prospective v264 freeze required"):
        backup.lease(tmp_path, "unpublished-preparation")


def test_worker_preserves_inherited_arguments(monkeypatch):
    command = ["python", "-m", backup.MODULE, "score", "trial", "freeze", "baseline", "75", "0"]
    monitor_process = backup.run.__globals__["subprocess"]
    monkeypatch.setattr(backup.subprocess, "Popen", lambda argv, **kwargs: (argv, kwargs))
    assert monitor_process.Popen(command, stdout="log", stderr=monitor_process.STDOUT) == (
        [*command[:3], "_worker", *command[3:]],
        {"stdout": "log", "stderr": backup.subprocess.STDOUT},
    )
