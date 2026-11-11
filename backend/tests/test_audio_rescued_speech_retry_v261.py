import ast
import inspect
from types import SimpleNamespace

from scripts import audio_rescued_speech_retry_v261 as retry


def test_worker_delegates_thread_initialization_and_model_loading_once():
    syntax = ast.parse(inspect.getsource(retry.worker) + "\n" + inspect.getsource(retry.classify))
    assert not {"set_num_threads", "set_num_interop_threads"} & {
        node.attr for node in ast.walk(syntax) if isinstance(node, ast.Attribute)
    }
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "scripts.run_audio_specialists_v15"
        and [name.name for name in node.names] == ["load_voice"]
        for node in ast.walk(syntax)
    )

    events = []

    def loader(lock):
        assert not events, "A second initialization would repeat the interop-thread failure"
        assert lock["assets"] and lock["weights"]
        events.append("load")
        return SimpleNamespace(
            register_forward_pre_hook=lambda _: SimpleNamespace(remove=lambda: events.append("remove"))
        )

    assert retry.classify([], loader, None, None, None) == []
    assert events == ["load", "remove"]
