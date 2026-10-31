"""Binoculars (Hans et al., ICML 2024): zero-shot LLM-text detection with two related language models.

Score = log-perplexity of the text under the *observer* model, divided by the cross-perplexity between
the *performer* and *observer* (how surprising the observer finds the performer's next-token
predictions). Machine text is unusually predictable *relative to* what a model would itself produce,
so its score is low. No training data is required; the threshold is calibrated per model pair
(scripts/calibrate_binoculars.py). Per-token quantities let us score individual sentences in the
context of the whole document.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core import deadline
from app.core.config import settings
from app.core.device import DEVICE, llm_dtype
from app.detectors.common.registry import registry

CTX = 1024
STRIDE = 768
LOSS_BLOCK_TOKENS = 32


@dataclass
class TokenScores:
    offsets: list[tuple[int, int]]  # char span per token
    obs_nll: np.ndarray  # observer negative log-likelihood per token
    xent: np.ndarray  # performer→observer cross-entropy per token
    unscored_first_token: bool = False  # only when the tokenizer has no EOS anchor; both first losses must be NaN

    def valid_mask(self) -> np.ndarray:
        """Validate complete expected coverage, never infer missing coverage from NaNs."""
        n = len(self.offsets)
        for values in (self.obs_nll, self.xent):
            if not isinstance(values, np.ndarray) or values.shape != (n,):
                raise ValueError("Binoculars token loss shape/offset alignment is invalid")
            if not np.issubdtype(values.dtype, np.floating):
                raise ValueError("Binoculars token losses must be floating point")
        if type(self.unscored_first_token) is not bool or (self.unscored_first_token and not n):
            raise ValueError("Binoculars unscored first-token marker is invalid")
        valid = np.ones(n, dtype=bool)
        if self.unscored_first_token:
            if not np.isnan(self.obs_nll[0]) or not np.isnan(self.xent[0]):
                raise ValueError("Binoculars unscored first token requires paired NaNs")
            valid[0] = False
        _check_losses(self.obs_nll[valid], self.xent[valid], int(valid.sum()))
        return valid


def _check_losses(nll: np.ndarray, xent: np.ndarray, count: int) -> None:
    for values in (nll, xent):
        if not isinstance(values, np.ndarray) or values.shape != (count,):
            raise ValueError("Binoculars chunk loss shape does not cover every expected token")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("Binoculars token losses must be floating point")
        if not np.isfinite(values).all():
            raise ValueError("Binoculars returned non-finite token losses or incomplete token coverage")
        if (values < 0).any():
            raise ValueError("Binoculars returned negative token losses")


def _ratio(nll: np.ndarray, xent: np.ndarray, *, numpy_division: bool = False) -> float:
    # Preserve the existing mean arithmetic for valid inputs. Validate the reduction
    # too: a finite input array can overflow its floating-point accumulator.
    with np.errstate(over="ignore", invalid="ignore"):
        numerator, denominator = np.mean(nll), np.mean(xent)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Binoculars mean loss/cross-entropy must be finite with positive cross-entropy")
    # The document path historically divides NumPy scalars before converting to
    # Python; sentence spans convert each mean first. Keep both rounding paths.
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(numerator / denominator) if numpy_division else float(numerator) / float(denominator)
    if not np.isfinite(value):
        raise ValueError("Binoculars loss ratio is non-finite")
    return value


def _check_logits_shape(raw: torch.Tensor, ids: torch.Tensor) -> None:
    if (not isinstance(raw, torch.Tensor) or raw.ndim != 3 or raw.shape[:2] != ids.shape
            or raw.shape[2] < 1 or not raw.is_floating_point()):
        raise ValueError("Binoculars language-model logits have an invalid shape or dtype")


def _full_shape_probabilities(raw: torch.Tensor, ids: torch.Tensor, *, log: bool,
                              expected_vocab: int | None = None, reuse_owned_cast: bool = False) -> torch.Tensor:
    """Preserve full shapes and borrowed inputs. MPS may reuse a fresh F16-to-F32 cast, never a borrowed F32 input."""
    _check_logits_shape(raw, ids)
    owned_cast = reuse_owned_cast and raw.dtype == torch.float16 and raw.shape[1] > 1
    source_storage = raw.untyped_storage().data_ptr() if owned_cast else None
    raw = raw.float()
    if not torch.isfinite(raw).all():  # includes the unused final row and F64-to-F32 overflow
        raise ValueError("Binoculars language-model logits are non-finite")
    if expected_vocab is not None and raw.shape[2] != expected_vocab:
        raise ValueError("Binoculars language-model logits have mismatched vocabularies")
    values = raw[:, :-1]
    if owned_cast:
        if raw.untyped_storage().data_ptr() == source_storage:
            raise ValueError("Binoculars probability buffer is not independently owned")
        # No retry after a failed out= operation: the owned buffer may already have been modified.
        return torch.log_softmax(values, dim=-1, out=values) if log else torch.softmax(values, dim=-1, out=values)
    return torch.log_softmax(values, dim=-1) if log else torch.softmax(values, dim=-1)


def _full_shape_losses(logp_obs: torch.Tensor, p_perf: torch.Tensor,
                       ids: torch.Tensor, *, consume_performer: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Nonmutating by default. Opt-in callers must exclusively own their fresh performer probability buffer."""
    if (consume_performer and p_perf.numel()
            and p_perf.untyped_storage().data_ptr() == logp_obs.untyped_storage().data_ptr()):
        raise ValueError("Binoculars performer probability buffer aliases observer evidence")
    nll = -logp_obs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    product = p_perf.mul_(logp_obs) if consume_performer else p_perf * logp_obs
    xent = -product.sum(-1)
    del product
    return nll[0].cpu().numpy(), xent[0].cpu().numpy()


@torch.inference_mode()
def _losses_from_logits(observer_logits: torch.Tensor, performer_logits: torch.Tensor,
                        ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """CPU reductions use 32-row F32 temporaries; accelerators retain original probability-operation shapes.

    Native model logits remain caller-owned here. The accelerator driver releases each model's logits before the next forward;
    this standalone arithmetic helper cannot release caller references. Neither path bounds model-forward memory.
    """
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
        raise ValueError("Binoculars expects one nonempty token sequence")
    for raw in (observer_logits, performer_logits):
        _check_logits_shape(raw, ids)
    if observer_logits.shape != performer_logits.shape:
        raise ValueError("Binoculars language-model logits have mismatched vocabularies")
    if ids.device.type != "cpu":
        reuse = ids.device.type == "mps"
        logp_obs = _full_shape_probabilities(observer_logits, ids, log=True, reuse_owned_cast=reuse)
        p_perf = _full_shape_probabilities(performer_logits, ids, log=False, reuse_owned_cast=reuse)
        return _full_shape_losses(logp_obs, p_perf, ids, consume_performer=reuse)
    count = ids.shape[1] - 1
    nll, xent = np.empty(count, dtype=np.float32), np.empty(count, dtype=np.float32)
    for start in range(0, ids.shape[1], LOSS_BLOCK_TOKENS):
        stop = min(start + LOSS_BLOCK_TOKENS, ids.shape[1])
        lo, lp = observer_logits[:, start:stop].float(), performer_logits[:, start:stop].float()
        if not torch.isfinite(lo).all() or not torch.isfinite(lp).all():
            raise ValueError("Binoculars language-model logits are non-finite")
        score_stop = min(stop, count)
        if start < score_stop:
            rows = score_stop - start
            tgt = ids[:, start + 1:score_stop + 1]
            logp_obs = torch.log_softmax(lo[:, :rows], dim=-1)
            nll[start:score_stop] = (-logp_obs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1))[0].cpu().numpy()
            p_perf = torch.softmax(lp[:, :rows], dim=-1)
            xent[start:score_stop] = (-(p_perf * logp_obs).sum(-1))[0].cpu().numpy()
            del logp_obs, p_perf
        # Do not keep the previous block's F32 casts or probability matrices while allocating the next block.
        del lo, lp
    return nll, xent


class Binoculars:
    def __init__(self) -> None:
        self.device: Any = DEVICE
        self.tok = AutoTokenizer.from_pretrained(settings.binoculars_observer_id)
        dt = llm_dtype()
        self.observer = AutoModelForCausalLM.from_pretrained(settings.binoculars_observer_id, dtype=dt).to(DEVICE).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(settings.binoculars_performer_id, dtype=dt).to(DEVICE).eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    @classmethod
    def from_parts(cls, tokenizer: Any, observer: Any, performer: Any, device: Any) -> Binoculars:
        """A view over already-loaded models (v2 adapters pass the leased observer, performer and their device)."""
        b = cls.__new__(cls)
        b.device, b.tok, b.observer, b.performer = device, tokenizer, observer, performer
        if b.tok.pad_token_id is None:
            b.tok.pad_token = b.tok.eos_token
        return b

    @torch.inference_mode()
    def _chunk_scores(self, ids: torch.Tensor, *, check: Callable[[str], None] | None = None,
                      ) -> tuple[np.ndarray, np.ndarray]:
        checkpoint = deadline.check if check is None else check
        checkpoint("Binoculars input transfer")
        ids = ids.to(self.device)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Binoculars expects one nonempty token sequence")
        if ids.device.type != "cpu":
            return self._full_shape_chunk_scores(ids) if check is None else self._full_shape_chunk_scores(ids, check=check)
        observer_logits = performer_logits = None
        try:
            checkpoint("Binoculars observer forward")
            observer_logits = self.observer(ids).logits
            checkpoint("completed Binoculars observer forward")
            performer_logits = self.performer(ids).logits
            checkpoint("completed Binoculars performer forward")
            result = _losses_from_logits(observer_logits, performer_logits, ids)
            checkpoint("completed Binoculars loss reduction")
            return result
        finally:
            # A retained timeout traceback must not retain the vocabulary logits.
            observer_logits = performer_logits = None

    @torch.inference_mode()
    def _full_shape_chunk_scores(self, ids: torch.Tensor, *, check: Callable[[str], None] | None = None,
                                 ) -> tuple[np.ndarray, np.ndarray]:
        # Full shapes preserve accelerator rounding. Only MPS reuses verified owned buffers; CUDA is unchanged.
        checkpoint = deadline.check if check is None else check
        reuse = ids.device.type == "mps"
        raw = logp_obs = p_perf = None
        try:
            checkpoint("Binoculars observer forward")
            raw = self.observer(ids).logits
            checkpoint("completed Binoculars observer forward")
            logp_obs = _full_shape_probabilities(raw, ids, log=True, reuse_owned_cast=reuse)
            raw = None
            checkpoint("Binoculars performer forward")
            raw = self.performer(ids).logits
            checkpoint("completed Binoculars performer forward")
            p_perf = _full_shape_probabilities(raw, ids, log=False,
                                               expected_vocab=logp_obs.shape[2], reuse_owned_cast=reuse)
            raw = None
            checkpoint("Binoculars loss reduction")
            result = _full_shape_losses(logp_obs, p_perf, ids, consume_performer=reuse)
            checkpoint("completed Binoculars loss reduction")
            return result
        except BaseException:
            # A forward can enqueue device work before a cooperative check raises.
            # Keep the caller's lease until that work completes. Sync failures propagate.
            if ids.device.type == "mps":
                torch.mps.synchronize()
            elif ids.device.type == "cuda":
                torch.cuda.synchronize(ids.device)
            raise
        finally:
            raw = logp_obs = p_perf = None

    def token_scores(self, text: str, *, check: Callable[[str], None] | None = None) -> TokenScores:
        """Score complete rolling coverage, checking the caller's deadline between expensive operations.

        Omitted callbacks retain the legacy context deadline. The v2 adapter passes
        its nested deadline; no check can pre-empt a native forward already running.
        """
        checkpoint = deadline.check if check is None else check
        checkpoint("Binoculars tokenization")
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        checkpoint("completed Binoculars tokenization")
        ids = enc["input_ids"]
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        n = len(ids)
        if len(offsets) != n:
            raise ValueError("Binoculars tokenizer ID/offset alignment is invalid")
        nll = np.full(n, np.nan, dtype=np.float32)
        xent = np.full(n, np.nan, dtype=np.float32)
        # leading BOS-like anchor so the first real token is scored too
        anchor = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []
        body = CTX - len(anchor)
        start = 0
        while start < n:
            checkpoint("Binoculars token window")
            end = min(start + STRIDE, n)
            ctx_start = max(0, end - body)  # include up to (body - STRIDE) tokens of left context
            chunk = anchor + ids[ctx_start:end]
            tokens = torch.tensor([chunk], dtype=torch.long)
            a, b = self._chunk_scores(tokens) if check is None else self._chunk_scores(tokens, check=check)
            checkpoint("completed Binoculars token window")
            _check_losses(a, b, len(chunk) - 1)
            # a[k] scores chunk[k+1]. With an anchor, chunk[k+1] == ids[ctx_start + k]; without, ids[ctx_start + k + 1].
            shift = 0 if anchor else 1
            for k in range(len(a)):
                di = ctx_start + k + shift
                if start <= di < end:
                    nll[di], xent[di] = a[k], b[k]
            start = end
        checkpoint("perplexity scoring")
        scores = TokenScores(offsets, nll, xent, unscored_first_token=bool(n and not anchor))
        scores.valid_mask()
        checkpoint("completed Binoculars token coverage")
        return scores

    @staticmethod
    def score_span(ts: TokenScores, start: int, end: int, min_tokens: int = 12) -> tuple[float | None, int]:
        """Binoculars score over tokens whose char spans overlap [start, end). Returns (score, n_tokens)."""
        valid = ts.valid_mask()
        idx = [i for i, (a, b) in enumerate(ts.offsets) if b > start and a < end and valid[i]]
        if len(idx) < min_tokens:
            return None, len(idx)
        return _ratio(ts.obs_nll[idx], ts.xent[idx]), len(idx)

    @staticmethod
    def to_probability(score: float) -> float:
        """Logistic mapping around the calibrated threshold (lower score → more likely machine)."""
        z = (settings.binoculars_threshold - score) * settings.binoculars_scale
        return float(1 / (1 + np.exp(-z)))


MIN_DOCUMENT_TOKENS = 12


def document_score(ts: TokenScores) -> tuple[float | None, int]:
    """(score over every scored token, number of scored tokens); None below 12 tokens."""
    valid = ts.valid_mask()
    n = int(valid.sum())
    if n < MIN_DOCUMENT_TOKENS:
        return None, n
    return _ratio(ts.obs_nll[valid], ts.xent[valid], numpy_division=True), n


def sentence_scores(ts: TokenScores, sents: list) -> list[float | None]:
    """Per-sentence scores; a sentence with too few tokens (and at least 3 words) is scored with its neighbours."""
    out: list[float | None] = [None] * len(sents)
    for s in sents:
        sc, _n = Binoculars.score_span(ts, s.start, s.end)
        if sc is None and s.n_words >= 3:  # short sentence: include neighbours for a stable estimate
            lo = sents[max(0, s.idx - 1)].start
            hi = sents[min(len(sents) - 1, s.idx + 1)].end
            sc, _n = Binoculars.score_span(ts, lo, hi)
        out[s.idx] = sc
    return out


def get_binoculars() -> Binoculars:
    """Load-only compatibility getter; inference must use the scoped borrow."""
    return registry.get("text:binoculars", Binoculars)


def borrow_binoculars() -> AbstractContextManager[Binoculars]:
    """Keep both legacy language models leased through token scoring."""
    return registry.borrow("text:binoculars", Binoculars)
