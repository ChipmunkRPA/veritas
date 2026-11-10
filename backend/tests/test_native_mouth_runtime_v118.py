from scripts import native_mouth_runtime_v118 as runtime


def test_same_measurement_code_objects_with_exclusive_new_context():
    old = runtime.core.TRIAL
    scope = runtime.context()
    for name in ("freeze", "verify", "run", "qualify"):
        assert scope[name].__code__ is getattr(runtime.core, name).__code__
        assert scope[name].__globals__["TRIAL"] == runtime.TRIAL
    assert old == runtime.core.TRIAL and old != runtime.TRIAL
    assert scope["qualify"].__globals__["CACHE"] == runtime.CACHE
    assert scope["response_counts"] is runtime.core.response_counts
    assert scope["earlier_plan"] is runtime.core.earlier_plan


def test_retained_overlay_sources_and_versions_are_bound():
    rows = runtime.overlay_identity()
    assert len(rows) == 4
    assert {r["name"] for r in rows} == {"diffusers", "importlib-metadata", "zipp", "huggingface-hub"}
    assert all(r["source_files"] > 0 and len(r["source_manifest_sha256"]) == 64 for r in rows)


def test_exact_runtime_worker_routing_without_original_argv_mutation(monkeypatch):
    args = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", "private"]
    calls = []
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    runtime.context()["routed_popen"](args)
    assert args[3] == "scripts.video_appearance_native_v108"
    assert calls == [[*args[:3], "scripts.native_mouth_runtime_v118", *args[4:]]]
