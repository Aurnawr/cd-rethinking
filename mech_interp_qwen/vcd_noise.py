#!/usr/bin/env python3
"""VCD's diffusion-noise image corruption.

Ported verbatim from this repo's `inference/cd_utils/vcd_utils.py:add_diffusion_noise` (itself
from DAMO-NLP-SG/VCD, arXiv 2311.16922). Copied rather than imported because that module also
imports transformers-4.31-only generation symbols (`GreedySearchOutput`, ...) at module scope and
is unimportable under the transformers version Qwen2.5-VL requires.

Applied to Qwen's `pixel_values`: the Qwen image processor's output is a *linear rearrange* of
the normalized image into flattened 14x14 patches (see `image_processing_qwen2_vl.py`:
rescale/normalize, then reshape+permute+reshape only). Elementwise Gaussian noise on that tensor
is therefore identical to noising the normalized image, which is exactly where VCD applies it for
LLaVA.
"""
import torch

NUM_DIFFUSION_STEPS = 1000


def add_diffusion_noise(image_tensor: torch.Tensor, noise_step: int) -> torch.Tensor:
    """q(x_t | x_0) for a single forward-diffusion step t = noise_step (0..999)."""
    num_steps = NUM_DIFFUSION_STEPS  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t].to(x_0.device, x_0.dtype)
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t].to(x_0.device, x_0.dtype)
        return alphas_t * x_0 + alphas_1_m_t * noise

    return q_x(image_tensor.clone(), int(noise_step))
