"""Keep intermediate CPU states while moving between the two writing models."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import torch

from app.detectors.text.binoculars import CTX, LOSS_BLOCK_TOKENS, _losses_from_logits
from app.engine.models import MemoryBudgetExceeded

HEAD_CHECK_ROWS = 512
FP32_BYTES = 4
BF16_BYTES = 2


class UnsupportedModel(ValueError):
    """The exact compact representation is unavailable for this model."""


@dataclass(frozen=True)
class PairedLayout:
    vocab: int
    width: int

    @property
    def head_elements(self):
        return self.vocab * self.width

    @property
    def capture_bytes(self):
        # Hold two original-precision heads and one temporary for bitwise verification.
        return 2 * self.head_elements * BF16_BYTES + HEAD_CHECK_ROWS * self.width * FP32_BYTES

    def hidden_bytes(self, tokens):
        return 2 * tokens * self.width * FP32_BYTES

    def projection_bytes(self, tokens):
        # Two full F32 heads and two F32 logit rows per token, plus loss-block temporaries.
        return (2 * (self.head_elements + tokens * self.vocab)
                + 5 * LOSS_BLOCK_TOKENS * self.vocab) * FP32_BYTES

    def window_limit(self, available):
        return max(0, (available - self.capture_bytes) // self.hidden_bytes(CTX))


def _head(model, layout, check):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    head = weight = compact = widened = None
    try:
        if type(model) is not Qwen2ForCausalLM or model.training:
            raise UnsupportedModel("CPU reuse requires the qualified evaluation-mode writing model")
        head = model.lm_head
        if (type(head) is not torch.nn.Linear or head.bias is not None or head.weight.device.type != "cpu"
                or head.weight.dtype != torch.float32 or head.weight.shape != (layout.vocab, layout.width)
                or not head.weight.is_contiguous()):
            raise UnsupportedModel("The writing model's CPU output head cannot be replayed")
        weight = head.weight.detach()
        compact = weight.to(torch.bfloat16)
        for start in range(0, layout.vocab, HEAD_CHECK_ROWS):
            stop = start + HEAD_CHECK_ROWS
            check("paired compact head verification")
            widened = compact[start:stop].float()
            if not torch.equal(widened.view(torch.int32), weight[start:stop].view(torch.int32)):
                raise UnsupportedModel("The writing model's output weights require full precision")
            widened = None
        return compact
    finally:
        model = head = weight = compact = widened = None


@torch.inference_mode()
def _score_group(windows, *, layout, allowances, borrow, model_for, check):
    """Release both leased backbones before projecting the original shape of each window."""
    captured, heads, hidden = [], [], None
    sequences = [window.numel() for window in windows]
    reserve = layout.capture_bytes + layout.hidden_bytes(sum(sequences))
    try:
        with borrow((), reserve):
            for model_id, activation in allowances.items():
                check("paired grouped writing lease")
                model = None
                try:
                    with borrow((model_id,), activation) as member:
                        check("paired grouped writing forward")
                        model = model_for(member, model_id)
                        heads.append(_head(model, layout, check))
                        states = []
                        captured.append(states)
                        for tokens in windows:
                            check("paired grouped writing window")
                            hidden = model.model(input_ids=tokens).last_hidden_state
                            if (hidden.device.type != "cpu" or hidden.dtype != torch.float32
                                    or hidden.shape != (*tokens.shape, layout.width)
                                    or not hidden.is_contiguous()
                                    or hidden.untyped_storage().nbytes() != hidden.numel() * FP32_BYTES):
                                raise UnsupportedModel("The writing model returned an unsupported CPU hidden state")
                            states.append(hidden)
                            hidden = None
                        check("completed paired grouped writing forward")
                finally:
                    model = hidden = None

            check("paired grouped writing projection lease")
            with borrow((), layout.projection_bytes(max(sequences))):
                return _project(windows, captured, heads, check)
    finally:
        # Clear references retained in tracebacks on cancellation or failed leases.
        for states in captured:
            states.clear()
        heads.clear()


def _project(windows, captured, heads, check):
    observer_head = performer_head = observed = performed = None
    try:
        observer_head, performer_head = (head.float() for head in heads)
        scores = []
        for tokens, observer, performer in zip(windows, *captured, strict=True):
            check("paired grouped writing projection")
            observed = torch.nn.functional.linear(observer, observer_head)
            performed = torch.nn.functional.linear(performer, performer_head)
            scores.append(_losses_from_logits(observed, performed, tokens))
            observed = performed = None
        check("completed paired grouped writing loss reduction")
        return scores
    finally:
        observer = performer = observer_head = performer_head = observed = performed = None


def grouped_losses(windows, *, layout, limit, allowances, borrow, model_for, check, fallback):
    """Yield identical per-window losses. Each group releases its reservations before yielding."""
    remaining = iter(windows)
    while batch := list(islice(remaining, limit)):
        if len(batch) == 1:
            yield fallback(batch[0])
            return
        try:
            scores = _score_group(batch, layout=layout, allowances=allowances, borrow=borrow,
                                  model_for=model_for, check=check)
        except (UnsupportedModel, MemoryBudgetExceeded):
            # Finish on the existing per-window path if the exact representation or a reservation is unavailable.
            check("paired writing fallback")
            yield from map(fallback, batch)
            yield from map(fallback, remaining)
            return
        yield from scores
