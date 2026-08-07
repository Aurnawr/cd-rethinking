#!/usr/bin/env python3
"""Faithful SID (Self-Introspective Decoding) amateur for the logit-lens battery.

The repo's `use_sid` path in `custom_modeling_llama.py` selects the retained vision
tokens with `torch.randperm` -- i.e. a RANDOM 72-of-576 subset. That is not SID. SID's
CT2S ("Context and Text-aware Token Selection", arXiv 2408.02032) uses the model's OWN
attention to rank vision tokens and keeps only the LEAST-important ones, so the amateur
is a *targeted* hallucination-inducer, not random visual dropout.

Original selection (huofushuo/SID, transformers/.../modeling_llama.py):

    last_layer_attention      = layer_outputs[1]                      # attn @ AGG_LAYER-1
    last_layer_attention_avg  = torch.mean(last_layer_attention, 1)[0]        # mean heads
    last_tok                  = last_layer_attention_avg[-1]                  # last-token row
    img                       = last_tok[SYS : SYS + IMAGE]                   # image band
    keep                      = img.topk(ATTENTION_RANK, largest=False).indices + SYS

We reproduce this WITHOUT editing any repo file: `install_correct_sid(model)` swaps the
LlamaModel instance's bound `forward` for a verbatim copy of the repo's forward whose
ONLY change is the `use_sid` selection block (random -> attention-ranked least-important).
This mirrors the repo's own monkey-patch convention (`evolve_sid_greedy_search`).

Config held at SID's official defaults (`pope_eval.py`): agg_layer=2, attention_rank=100.
The image span is located dynamically per sample from IMAGE_TOKEN_INDEX (more correct than
the repo's hardcoded SYS_LENGTH=35). Set it each sample via `set_sid_key_position`.
"""
import types
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

# SID official defaults (huofushuo/SID pope_eval.py): rank=100 kept, aggregation at layer 2.
# (The repo used rank=72; kept configurable so both can be compared.)
SID_AGG_LAYER = 2
SID_ATTENTION_RANK = 100
IMAGE_TOKEN_LENGTH = 576
IMAGE_TOKEN_INDEX = -200  # LLaVA placeholder id, pre multimodal expansion
DEFAULT_SYS_LENGTH = 35   # repo fallback if the placeholder cannot be located


def set_sid_key_position(model, input_ids):
    """Record where the 576 image embeddings begin, from the pre-expansion input_ids.

    LLaVA expands the single IMAGE_TOKEN_INDEX placeholder into 576 embeddings in place,
    so the placeholder's index in `input_ids` is the image-band start after the merge.
    """
    ids = input_ids[0]
    pos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    model.model._sid_image_start = int(pos[0].item()) if pos.numel() else DEFAULT_SYS_LENGTH


def set_sid_config(model, agg_layer=SID_AGG_LAYER, attention_rank=SID_ATTENTION_RANK):
    model.model._sid_agg_layer = int(agg_layer)
    model.model._sid_attention_rank = int(attention_rank)


def _correct_llama_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    use_sid: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """Verbatim copy of the repo LlamaModel.forward, SID selection made faithful.

    Differences from `custom_modeling_llama.LlamaModel.forward`:
      * the `use_sid` block ranks image tokens by the model's own attention at
        AGG_LAYER-1 and keeps the LEAST-attended `attention_rank` of them
        (original SID), instead of a random subset;
      * attention weights are requested only at AGG_LAYER-1 (memory-frugal);
      * `all_self_attns` is never accumulated (we do not return attentions).
    Everything else -- pre/post RMSNorm, the post-norm final hidden append at index 32,
    hidden-state bookkeeping -- is identical, so the logit lens stays exact.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    # SID config (defaults = SID official; may be overridden per model)
    AGG_LAYER = getattr(self, "_sid_agg_layer", SID_AGG_LAYER)
    ATTENTION_RANK = getattr(self, "_sid_attention_rank", SID_ATTENTION_RANK)
    SYS_LENGTH = getattr(self, "_sid_image_start", DEFAULT_SYS_LENGTH)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = None  # we never return attentions
    next_decoder_cache = () if use_cache else None
    layer_outputs = None
    gen_attention_mask = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        # ---- SID (faithful CT2S): attention-ranked least-important vision tokens ----
        if use_sid:
            if idx < AGG_LAYER:
                # full causal attention (identical to the standard mask)
                new_attention_mask = attention_mask
            elif idx == AGG_LAYER:
                # layer_outputs is from idx-1 == AGG_LAYER-1; [1] is its attention weights
                last_layer_attention = layer_outputs[1]                       # [b, heads, q, k]
                last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]  # [q, k]
                last_tok = last_layer_attention_avg[-1]                        # [k]
                img_att = last_tok[SYS_LENGTH : SYS_LENGTH + IMAGE_TOKEN_LENGTH]
                keep = img_att.topk(ATTENTION_RANK, largest=False).indices + SYS_LENGTH
                gen_attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
                gen_attention_mask[:, SYS_LENGTH : SYS_LENGTH + IMAGE_TOKEN_LENGTH] = False
                gen_attention_mask[:, keep] = True
                gen_attention_mask = self._prepare_decoder_attention_mask(
                    gen_attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
                new_attention_mask = gen_attention_mask
            else:
                new_attention_mask = gen_attention_mask
        else:
            new_attention_mask = attention_mask

        # need this layer's attention only to rank at AGG_LAYER (read next iteration)
        need_attn = bool(use_sid) and (idx == AGG_LAYER - 1)
        layer_out_attn = bool(output_attentions) or need_attn

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=layer_out_attn,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if layer_out_attn else 1],)

    hidden_states = self.norm(hidden_states)

    # append final (post-norm) hidden state -> index 32, matching the repo's lens contract
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def install_correct_sid(model, agg_layer=SID_AGG_LAYER, attention_rank=SID_ATTENTION_RANK):
    """Swap the LlamaModel instance's forward for the faithful-SID version (reversible).

    Returns the original bound forward so callers can restore it if needed.
    """
    inner = model.model  # LlavaLlamaForCausalLM -> LlamaModel
    if getattr(inner, "_sid_orig_forward", None) is None:
        inner._sid_orig_forward = inner.forward
    set_sid_config(model, agg_layer, attention_rank)
    inner.forward = types.MethodType(_correct_llama_forward, inner)
    return inner._sid_orig_forward


def uninstall_correct_sid(model):
    inner = model.model
    if getattr(inner, "_sid_orig_forward", None) is not None:
        inner.forward = inner._sid_orig_forward
        inner._sid_orig_forward = None
