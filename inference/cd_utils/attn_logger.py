import torch

def compute_attention_masses(attentions, img_start, img_end, query_position=-1):
    """
    Returns (visual_mass, total_mass) averaged over all layers and heads.
    visual_mass: mean attention mass on image tokens
    total_mass:  mean attention mass on full sequence (always ~1.0 by softmax)
    """
    visual_masses = []
    total_masses  = []

    for layer_attn in attentions:
        attn_at_pos = layer_attn[0, :, query_position, :]  # (heads, seq_k)
        visual_mass = attn_at_pos[:, img_start:img_end].sum(dim=-1)  # (heads,)
        total_mass  = attn_at_pos.sum(dim=-1)                        # (heads,) always ~1.0

        visual_masses.append(visual_mass.mean().item())
        total_masses.append(total_mass.mean().item())

    return (
        sum(visual_masses) / len(visual_masses),
        sum(total_masses)  / len(total_masses),
    )


def get_expanded_image_span(input_ids_1d, image_token_index, num_patches):
    positions = (input_ids_1d == image_token_index).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return None, None
    pos = positions[0].item()
    return pos, pos + num_patches


def get_expanded_image_span_dynamic(input_ids_1d, image_token_index, total_seq_len):
    """AnyRes-aware image-token span.

    LLaVA-1.6 expands the single <image> placeholder into a dynamic number of
    tokens (base thumbnail + spatially-unpadded hi-res crops + newline tokens),
    so the count cannot be read from a static num_patches. It is recovered from
    the post-forward sequence length: each placeholder is replaced by N image
    tokens, hence total_seq_len = (len(input_ids) - num_placeholders) + N_total.
    """
    positions = (input_ids_1d == image_token_index).nonzero(as_tuple=True)[0]
    if len(positions) == 0:
        return None, None
    pos = positions[0].item()
    num_placeholders = len(positions)
    num_image_tokens = total_seq_len - (input_ids_1d.shape[0] - num_placeholders)
    return pos, pos + num_image_tokens