from __future__ import annotations

import dataclasses
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_hash
from app.detectors.text import binoculars
from app.detectors.text.adapters.learned import BinocularsDetector
from app.detectors.text.extract import from_string, text_hash
from app.detectors.text.segment import segment
from app.engine.boundary import Deadline, DetectorTimeout, InMemoryRunStore, run_detector
from app.engine.detector import Profile, RunContext
from app.engine.manifest import load_manifest
from app.engine.models import FootprintStore, ManagedModel, ModelManager
from app.engine.registry import DetectorRegistry


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def get_vocab(self):
        return {chr(i): i + 1 for i in range(128)} | {"😀": 200}

    def __call__(self, text, **kwargs):
        # A surrogate-pair character remains one codepoint; like byte-level BPE,
        # two token pieces may refer to the same non-ASCII source character.
        offsets = [(i, i + 1) for i, c in enumerate(text) for _ in range(2 if c == "😀" else 1)]
        return {"input_ids": [self.get_vocab().get(text[a:b], 201) for a, b in offsets],
                "offset_mapping": offsets}


def fake_losses(self, tokens, check=None):
    ids = tokens[0].tolist()
    return np.array([1 + value / 50 for value in ids[1:]], dtype=np.float32), np.full(len(ids) - 1, 2, dtype=np.float32)


def test_production_manifest_can_import_changed_adapter():
    entry = load_manifest().detector("txt.binoculars")
    assert entry is not None
    assert DetectorRegistry._adapter_class(entry) is BinocularsDetector
    assert entry.version == "0.4.1"


def context(doc, first=None, other=None):
    first, other = first or CharacterTokenizer(), other or CharacterTokenizer()
    sources = {BinocularsDetector.OBSERVER: first, BinocularsDetector.PERFORMER: other}
    sents = segment(doc)
    return SimpleNamespace(outputs=SimpleNamespace(document={"text": doc.text, "paragraphs": doc.paragraphs,
                           "meta": doc.meta, "source_map": doc.source_map,
                           "sentences": [dataclasses.asdict(s) for s in sents]}),
                           lease=SimpleNamespace(get=lambda key: SimpleNamespace(obj=(sources[key], object()), revision="fixture")),
                           device="cpu", deadline=SimpleNamespace(check=lambda _: None))


def test_recorded_rolling_inputs_equal_actual_forward_ids_and_unicode_offsets(monkeypatch):
    monkeypatch.setattr(binoculars, "CTX", 24)
    monkeypatch.setattr(binoculars, "STRIDE", 16)
    calls = []

    def losses(self, tokens, check=None):
        calls.append(tokens[0].tolist())
        return fake_losses(self, tokens, check)

    monkeypatch.setattr(binoculars.Binoculars, "_chunk_scores", losses)
    text = "😀 First sentence has words.\n\nYes we can. The final sentence has useful words."
    tokenizer = CharacterTokenizer()
    scores = binoculars.Binoculars.from_parts(tokenizer, object(), object(), "cpu").token_scores(text)
    assert len(calls) > 2
    assert [c["input_token_ids"] for c in scores.input_contexts] == calls
    assert scores.offsets[:2] == [(0, 1), (0, 1)]
    covered = []
    for c in scores.input_contexts:
        assert canonical_hash(c["input_token_ids"]) == c["token_ids_sha256"]
        a, b = c["spans"][0].values()
        assert c["text"] == text[a:b] and c["input_sha256"] == text_hash(text[a:b])
        assert len(c["input_token_ids"]) <= 24 and c["anchor_token_count"] == 1
        lo, hi = c["source_token_range"]
        assert c["token_offsets"] == [list(o) for o in scores.offsets[lo:hi]]
        covered.extend(range(*c["scored_token_range"]))
    assert covered == list(range(len(scores.offsets)))
    # A short sentence's reduction includes neighbours, but its scored tokens
    # still came from several real rolling forward inputs.
    doc = from_string(text)
    sents = segment(doc)
    actual = binoculars.Binoculars.from_parts(tokenizer, object(), object(), "cpu").token_scores(doc.text)
    records = binoculars.sentence_score_records(actual, sents)
    record = records[1]
    assert record["neighbor_expansion"] is True
    assert len(record["context_ids"]) > 1
    idx = record["scored_token_indices"]
    assert record["score"] == float(np.mean(actual.obs_nll[idx])) / float(np.mean(actual.xent[idx]))
    assert binoculars.sentence_scores(actual, sents) == [r["score"] for r in records]


def test_without_anchor_first_token_is_explicitly_unscored(monkeypatch):
    monkeypatch.setattr(binoculars.Binoculars, "_chunk_scores", fake_losses)
    tokenizer = CharacterTokenizer()
    tokenizer.eos_token_id = None
    doc = from_string("A sufficiently long sentence has enough characters.")
    scores = binoculars.Binoculars.from_parts(tokenizer, object(), object(), "cpu").token_scores(doc.text)
    assert scores.unscored_first_token and not scores.valid_mask()[0]
    assert scores.input_contexts[0]["anchor_token_count"] == 0
    assert scores.input_contexts[0]["scored_token_range"][0] == 1
    assert 0 not in binoculars.sentence_score_records(scores, segment(doc))[0]["scored_token_indices"]


def test_refused_dual_lease_replays_identical_cpu_ranges_with_one_model_pinned(monkeypatch, tmp_path):
    import torch

    from app.core.config import settings

    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(binoculars, "CTX", 24)
    monkeypatch.setattr(binoculars, "STRIDE", 16)
    monkeypatch.setattr(BinocularsDetector, "VOCABULARY_SIZE", 256)
    detector = BinocularsDetector()
    calls = []
    doc = from_string("A measured writing response has enough text. 😀 The next sentence crosses several contexts.")

    def execute(budget, label, *, time_out=False):
        def build(model_id, device, dtype):
            assert device == "cpu" and dtype == "float32"

            def forward(ids):
                pinned = sorted(row.id for row in models.snapshot() if row.pins)
                calls.append((label, model_id, pinned, ids[0].tolist()))
                if time_out and model_id == detector.PERFORMER:
                    raise DetectorTimeout("fixture deadline during the second forward")
                center = ids.unsqueeze(-1).float()
                vocab = torch.arange(256, dtype=torch.float32)
                gap = 1. if model_id == detector.OBSERVER else 3.
                return SimpleNamespace(logits=-(vocab - center - gap).abs() / 64.)

            return SimpleNamespace(obj=(CharacterTokenizer(), forward), revision="fixture")

        entries = [ManagedModel(model_id, fp32_bytes=50_000, activation_bytes_per_item=8_000)
                   for model_id in (detector.OBSERVER, detector.PERFORMER)]
        models = ModelManager(budget_bytes=budget, models=entries, loader_for=lambda _: build,
                              store=FootprintStore(tmp_path / f"{label}.json"))
        store = InMemoryRunStore()
        run_ctx = RunContext(label, None, Profile.FULL, "cpu", "float32", None, context(doc).outputs, Deadline(), None, None)
        result = run_detector(detector, run_ctx, store=store, models=models)
        assert len(store.for_analysis(label)) == 1
        assert all(row.pins == 0 for row in models.snapshot())
        assert models.free_bytes() == budget - sum(row.bytes_declared for row in models.snapshot())
        return result

    baseline = execute(150_000, "simultaneous")
    sequential = execute(85_000, "sequential")
    assert baseline.status == sequential.status == "ok"
    assert baseline.aggregate == sequential.aggregate and baseline.unit_scores == sequential.unit_scores
    assert baseline.details == sequential.details and baseline.model_revisions == sequential.model_revisions
    forward_inputs = {
        label: [(model, ids) for mode, model, _, ids in calls if mode == label]
        for label in ("simultaneous", "sequential")
    }
    assert len(forward_inputs["sequential"]) > 4 and forward_inputs["sequential"] == forward_inputs["simultaneous"]
    assert all(pinned == [model] for mode, model, pinned, _ in calls if mode == "sequential")
    timed_out = execute(85_000, "timeout", time_out=True)
    assert timed_out.status == "timeout" and not timed_out.details and timed_out.aggregate is None
    monkeypatch.setattr(settings, "environment", "production")
    refused = execute(85_000, "production")
    assert refused.status == "skipped" and refused.reason == "memory budget"


@pytest.mark.parametrize("exact_head", [True, False])
def test_cpu_groups_keep_exact_losses_and_release_reservations(monkeypatch, tmp_path, exact_head):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from app.core.config import settings
    from app.detectors.text import compact_cpu

    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(binoculars, "CTX", 24)
    monkeypatch.setattr(binoculars, "STRIDE", 16)
    monkeypatch.setattr(compact_cpu, "CTX", 24)
    monkeypatch.setattr(BinocularsDetector, "VOCABULARY_SIZE", 256)
    monkeypatch.setattr(BinocularsDetector, "HIDDEN_SIZE", 32)
    detector = BinocularsDetector()
    keys = (detector.OBSERVER, detector.PERFORMER)
    doc = from_string("An exact writing comparison uses several contexts. 😀 The last context has separate offsets.")
    config = Qwen2Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(17)
        checkpoints = {key: Qwen2ForCausalLM(config).bfloat16().float().eval() for key in keys}
        if not exact_head:
            checkpoints[detector.PERFORMER].lm_head.weight[0, 0] = 0.1234567

    def execute(grouped, *, status="ok"):
        loads = []

        def build(key, device, dtype):
            assert (device, dtype) == ("cpu", "float32")
            loads.append(key)
            return SimpleNamespace(obj=(CharacterTokenizer(), checkpoints[key]), revision="fixture")

        monkeypatch.setattr(settings, "text_ratio_compact_cpu", grouped)
        entries = [ManagedModel(key, fp32_bytes=200_000, activation_bytes_per_item=45_000) for key in keys]
        models = ModelManager(budget_bytes=420_000, models=entries, loader_for=lambda _: build,
                              store=FootprintStore(tmp_path / f"grouped-{grouped}.json"))
        ctx = RunContext("sample", None, Profile.FULL, "cpu", "float32", None, context(doc).outputs, Deadline(), None, None)
        result = run_detector(detector, ctx, store=InMemoryRunStore(), models=models)
        assert result.status == status, result.reason
        assert all(row.pins == 0 for row in models.snapshot())
        assert models.free_bytes() == models.budget_bytes - sum(row.bytes_declared for row in models.snapshot())
        return result, loads

    reference, per_window = execute(False)
    result, grouped = execute(True)
    assert result.details == reference.details
    assert result.unit_scores == reference.unit_scores and result.aggregate == reference.aggregate
    if exact_head:
        assert grouped == list(keys) and len(per_window) > 4

        def stopped(*args):
            raise DetectorTimeout("test stopped during the reserved projection")

        monkeypatch.setattr(compact_cpu, "_project", stopped)
        cancelled, _ = execute(True, status="timeout")
        assert cancelled.aggregate is None and not cancelled.details
    else:
        assert grouped == [*keys, *per_window]


def test_adapter_freezes_raw_ratio_threshold_and_matching_paired_tokenization(monkeypatch):
    monkeypatch.setattr(binoculars.Binoculars, "_chunk_scores", fake_losses)
    doc = from_string("😀 This sentence contains enough characters for the language check. Yes we can.")
    result = BinocularsDetector().run(context(doc))
    evidence = result.details["text_evidence"]
    assert evidence["version"] == "scored-context/2" and evidence["shared_tokenization_verified"]
    assert len(evidence["windows"]) == 2
    assert all(w["score_kind"] == "perplexity_ratio" and w["ai_when"] == "lt" and not w["is_calibrated"] for w in evidence["windows"])
    assert evidence["windows"][0]["score"] > 1  # A valid ratio, never clamped to a probability.
    assert evidence["windows"][1]["neighbor_expansion"]
    assert float.fromhex(evidence["windows"][0]["score_exact_hex"]) == evidence["windows"][0]["score"]


def test_different_default_eos_keeps_actual_shared_explicit_anchor(monkeypatch):
    monkeypatch.setattr(binoculars.Binoculars, "_chunk_scores", fake_losses)
    other = CharacterTokenizer()
    other.eos_token_id = 5
    result = BinocularsDetector().run(context(from_string("This source has enough tokens for a measured response."), other=other))
    evidence = result.details["text_evidence"]
    assert evidence["shared_tokenization_verified"]
    assert evidence["windows"][0]["paired_tokenizer_default_eos_ids"] == [0, 5]
    assert evidence["windows"][0]["shared_input_anchor_id"] == 0
    assert all(c["input_token_ids"][0] == 0 and c["anchor_token_count"] == 1 for c in evidence["contexts"])


@pytest.mark.parametrize("difference", ["vocabulary", "source_offsets", "source_ids"])
def test_paired_tokenizer_disagreement_cannot_generate_evidence(monkeypatch, difference):
    monkeypatch.setattr(binoculars.Binoculars, "_chunk_scores", fake_losses)

    class Different(CharacterTokenizer):
        def get_vocab(self):
            vocab = super().get_vocab()
            if difference == "vocabulary":
                vocab["a"] = 999
            return vocab

        def __call__(self, text, **kwargs):
            encoded = deepcopy(super().__call__(text, **kwargs))
            if difference == "source_offsets":
                encoded["offset_mapping"][0] = (1, 2)
            elif difference == "source_ids":
                encoded["input_ids"][0] = 999
            return encoded

    with pytest.raises(ValueError, match="paired text check"):
        BinocularsDetector().run(context(from_string("This text has more than twelve source tokens."), other=Different()))


def test_recording_bound_is_explicit_and_does_not_change_measured_scores(monkeypatch):
    monkeypatch.setattr(binoculars.Binoculars, "_chunk_scores", fake_losses)
    doc = from_string("A sufficiently long sentence has enough characters for this measurement.")
    original = BinocularsDetector().run(context(doc))
    monkeypatch.setattr(binoculars, "MAX_EVIDENCE_TOKENS", 8)
    bounded = BinocularsDetector().run(context(doc))
    assert bounded.aggregate == original.aggregate and bounded.unit_scores == original.unit_scores
    evidence = bounded.details["text_evidence"]
    assert evidence["windows"] == [] and evidence["contexts"] == []
    assert evidence["omitted_windows"] and "bound" in evidence["omitted_windows"][0]["reason"]
