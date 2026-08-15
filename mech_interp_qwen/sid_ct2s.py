#!/usr/bin/env python3
"""Faithful SID (CT2S) amateur branch for Qwen2.5-VL.

SID -- Self-Introspective Decoding, arXiv 2408.02032 -- builds its amateur by keeping only the
*least important* vision tokens after an early decoder layer, where importance is the model's own
attention. Its "Context and Text-aware Token Selection" (CT2S) is, per Eq. 5:

    Score_i(v) = mean over heads of  A_i[..., -1, v]        # last query row, layer i
    keep       = the lowest-scoring `k` vision tokens
    all later decoder layers attend to `keep` only, the rest of the vision band is masked

Two things this module deliberately does NOT inherit
----------------------------------------------------
1. **The repo's `use_sid` path is not SID.** `llava/.../custom_modeling_llama.py` selects the
   retained vision tokens with `torch.randperm` -- random visual dropout, not attention-ranked
   selection. `mech_interp/sid_correct.py` fixed that for LLaVA by swapping the whole
   `LlamaModel.forward`. That approach does not port to Qwen (mRoPE, `get_rope_index`, the vision
   merge), so here CT2S is implemented with **hooks only**, in a single forward pass, touching no
   model source.

2. **The count is a ratio, not 100.** The SID paper's stated recipe is
   *"we set Layer i=3 and preserve top 10% least important vision tokens for Shikra, LLaVA-1.5,
   and LLaVA-NeXT"* -- the same 10% across models encoding 256 / 576 / 2304 vision tokens. The
   released code's `--fast-v-attention-rank 100` (= 17.4% of LLaVA's 576) is a `fast-v` default,
   not the paper's number. Qwen2.5-VL's dynamic resolution makes any absolute count meaningless,
   so `keep_ratio=0.10` is the default here. `keep_tokens` overrides it with an absolute count if
   you want released-code parity.

Layer indexing, stated once
---------------------------
`rank_layer` is the **0-indexed** decoder layer whose attention does the ranking; the mask applies
from `rank_layer + 1` onward. The paper's 1-indexed "Layer i=3" is `rank_layer=2` (the default).
The released code's `fast_v_agg_layer=2` ranks with the layer at 0-index 1, i.e. `rank_layer=1`.
Both are reachable; the off-by-one between paper and code is resolved by naming, not inherited.

Getting the attention weights out
---------------------------------
Two things are needed, and both are handled here:
  * `attn_implementation="eager"` (the `qwen_runtime.load_model` default) -- sdpa and flash never
    materialise the attention matrix;
  * `output_attentions=True` on the ranking layer's attention module. transformers 4.51's
    `Qwen2_5_VLAttention` explicitly does `if not output_attentions: attn_weights = None` before
    returning, so eager alone is not enough. A pre-hook sets it for that one module only, which
    also makes the code work on newer versions where the flag was dropped and weights come back
    unconditionally (the flag is then simply not in the signature and nothing is set).
Cost is negligible at POPE sequence lengths (~450 tokens => ~10 MB transient for one layer).
"""
import contextlib
import inspect
from typing import Optional

import torch

from qwen_runtime import decoder_layers

# SID paper defaults (Sec. 5.1 "Implementation Details" + Sec. 4.2).
SID_RANK_LAYER = 2      # 0-indexed; paper's 1-indexed Layer i=3
SID_KEEP_RATIO = 0.10   # keep the 10% least-attended vision tokens


class SidCT2S:
    """Installs CT2S for the duration of one forward pass via `session(vision_pos)`."""

    def __init__(self, model, rank_layer: int = SID_RANK_LAYER,
                 keep_ratio: float = SID_KEEP_RATIO, keep_tokens: Optional[int] = None):
        self.layers = decoder_layers(model)
        if not 0 <= rank_layer < len(self.layers) - 1:
            raise ValueError(
                f"rank_layer={rank_layer} out of range for a {len(self.layers)}-layer decoder "
                f"(must leave at least one later layer to mask)"
            )
        self.rank_layer = int(rank_layer)
        self.keep_ratio = float(keep_ratio)
        self.keep_tokens = None if keep_tokens is None else int(keep_tokens)
        # Where `output_attentions` sits in this version's attention signature (None if the flag
        # was dropped and weights are returned unconditionally).
        attn_params = list(inspect.signature(self.layers[self.rank_layer].self_attn.forward).parameters)
        self._oa_index = attn_params.index("output_attentions") if "output_attentions" in attn_params else None
        self._reset(None)

    # ------------------------------------------------------------------ state
    def _reset(self, vision_pos):
        self.vision_pos = vision_pos
        self.attn_weights = None
        self.drop_idx = None
        self._mask = None
        self.n_vision = 0
        self.n_keep = 0

    @property
    def n_dropped(self) -> int:
        return 0 if self.drop_idx is None else int(self.drop_idx.numel())

    # ------------------------------------------------------------------ hooks
    def _request_attention(self, _module, args, kwargs):
        """Pre-hook on the ranking layer's attention: ask it to return its weights."""
        if self._oa_index is None:
            return None  # this version returns weights unconditionally
        if len(args) > self._oa_index:
            args = args[: self._oa_index] + (True,) + args[self._oa_index + 1:]
        else:
            kwargs["output_attentions"] = True
        return args, kwargs

    def _capture_attention(self, _module, _inputs, output):
        """Forward hook on layers[rank_layer].self_attn -> stash the attention weights."""
        self.attn_weights = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None

    def _select_tokens(self):
        """SID Eq. 5 -> the vision positions to mask out."""
        aw = self.attn_weights
        if not torch.is_tensor(aw):
            raise RuntimeError(
                f"SID: no attention weights captured at decoder layer {self.rank_layer}. "
                f"Load the model with attn_implementation='eager' -- sdpa and flash never "
                f"materialise the attention matrix."
            )
        # aw: [batch, heads, q, k]; mean over heads, take the last query row
        score = aw.float().mean(dim=1)[0, -1, :]
        vis = self.vision_pos
        n_vis = int(vis.numel())
        if n_vis == 0:
            raise RuntimeError("SID: no vision tokens found in input_ids")
        n_keep = (self.keep_tokens if self.keep_tokens is not None
                  else int(round(self.keep_ratio * n_vis)))
        n_keep = max(1, min(n_keep, n_vis))
        keep_local = torch.topk(score[vis], n_keep, largest=False).indices
        keep_mask = torch.zeros(n_vis, dtype=torch.bool, device=vis.device)
        keep_mask[keep_local] = True
        self.drop_idx = vis[~keep_mask]
        self.n_vision, self.n_keep = n_vis, n_keep

    def _build_mask(self, hidden, incoming):
        """A [1,1,q,k] float additive mask: causal structure + dropped vision columns at -inf.

        Float-additive (rather than boolean) because it is the one form both the eager and the
        sdpa attention paths accept. Column-masking applies to every query row, matching SID's
        own mask construction. Cached: identical for every layer after `rank_layer`.
        """
        if self._mask is not None:
            return self._mask
        if isinstance(incoming, dict):
            raise RuntimeError(
                "SID: this transformers version passes a mask mapping to decoder layers; "
                "update _build_mask() in sid_ct2s.py"
            )
        dtype, device = hidden.dtype, hidden.device
        neg = torch.finfo(dtype).min
        if torch.is_tensor(incoming) and incoming.dim() == 4:
            if incoming.dtype == torch.bool:  # True = attend
                base = torch.zeros(incoming.shape, dtype=dtype, device=device)
                base.masked_fill_(~incoming, neg)
            else:
                base = incoming.to(dtype).clone()
        else:
            # sdpa's is_causal shortcut passes None -- rebuild the causal structure ourselves
            q = hidden.shape[1]
            causal = torch.triu(torch.ones(q, q, dtype=torch.bool, device=device), diagonal=1)
            base = torch.zeros((1, 1, q, q), dtype=dtype, device=device).masked_fill(causal, neg)
        base[..., self.drop_idx] = neg
        self._mask = base
        return base

    def _patch_mask(self, _module, args, kwargs):
        """Pre-hook on every layer after rank_layer -> swap in the CT2S attention mask."""
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None:
            raise RuntimeError("SID: decoder layer called without hidden states")
        if self.drop_idx is None:
            self._select_tokens()
        incoming = kwargs["attention_mask"] if "attention_mask" in kwargs else (
            args[1] if len(args) > 1 else None
        )
        mask = self._build_mask(hidden, incoming)
        if "attention_mask" in kwargs or len(args) < 2:
            kwargs["attention_mask"] = mask
        else:
            args = (args[0], mask) + tuple(args[2:])
        return args, kwargs

    # ------------------------------------------------------------------ session
    @contextlib.contextmanager
    def session(self, vision_pos: torch.Tensor):
        """Arm CT2S for one forward pass over an input whose vision band is `vision_pos`."""
        self._reset(vision_pos)
        rank_attn = self.layers[self.rank_layer].self_attn
        handles = [
            rank_attn.register_forward_pre_hook(self._request_attention, with_kwargs=True),
            rank_attn.register_forward_hook(self._capture_attention),
        ]
        for i in range(self.rank_layer + 1, len(self.layers)):
            handles.append(
                self.layers[i].register_forward_pre_hook(self._patch_mask, with_kwargs=True)
            )
        try:
            yield self
        finally:
            for h in handles:
                h.remove()

    def describe(self) -> str:
        return (f"CT2S rank_layer={self.rank_layer} (paper Layer i={self.rank_layer + 1}) "
                f"keep={'%d tokens' % self.keep_tokens if self.keep_tokens is not None else '%.0f%%' % (100 * self.keep_ratio)}"
                f"  last: {self.n_keep}/{self.n_vision} kept, {self.n_dropped} masked")
