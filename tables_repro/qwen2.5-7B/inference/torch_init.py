"""Minimal standalone copy of LLaVA's disable_torch_init.

The Qwen pipeline only needed this one helper from the LLaVA package, so we
inline it here to keep this folder free of any LLaVA dependency. It skips the
default weight initialisation of Linear / LayerNorm layers, which is redundant
when the weights are about to be overwritten by a pretrained checkpoint and
noticeably speeds up model construction.
"""
import torch


def disable_torch_init():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
