#!/usr/bin/env python3
"""Model-side contract test on a randomly-initialised, tiny Qwen2.5-VL - CPU, no weights.

Everything this pipeline relies on that is *not* a matter of numerics -- module layout, the
hidden-state/norm contract the logit lens assumes, the CT2S hook mechanics, and the fact that
each amateur actually differs from the expert -- is architecture, not weights. So it can be
verified in seconds against a 4-layer random model instead of a 16 GB checkpoint, and it will
catch a transformers upgrade that breaks the port before any GPU time is spent.

Only the real processor is downloaded (tokenizer + image processor + chat template, a few MB);
the model itself is constructed from a small config with random weights.

    python mech_interp_qwen/test_model_tiny.py

Complements:
  test_analysis_synthetic.py  everything downstream of the forward pass (no torch needed)
  selfcheck.py                the same contracts on the real checkpoint, plus scientific sanity
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from amateur_branches import amateur_context, build_amateur_inputs  # noqa: E402
from qwen_runtime import (  # noqa: E402
    build_inputs, decoder_layers, final_norm, forward_hiddens, image_token_id, lens_margins,
    lm_head_weight, n_hidden_layers, text_model, vision_positions, yes_no_token_ids,
)
from sid_ct2s import SidCT2S  # noqa: E402

PROCESSOR_ID = os.environ.get("QWEN_PROCESSOR_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
FAILURES = []


def check(ok, name, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def tiny_model():
    """A 4-layer Qwen2.5-VL with the real tokenizer geometry and the real patch/merge sizes."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

    hidden = 64
    # patch_size / spatial_merge_size / temporal_patch_size must match the real processor, or the
    # number of vision embeddings the tower emits will not match the placeholders it inserted.
    # Passed as a dict: Qwen2_5_VLConfig only accepts dict or None for vision_config.
    vision = dict(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
        in_channels=3, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
        window_size=112, out_hidden_size=hidden, fullatt_block_indexes=[1],
    )
    cfg = Qwen2_5_VLConfig(
        vocab_size=152064, hidden_size=hidden, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, tie_word_embeddings=False,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        vision_config=vision,
        image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg)
    model.eval()
    return model


def main():
    from transformers import AutoProcessor

    print(f"loading processor: {PROCESSOR_ID}")
    processor = AutoProcessor.from_pretrained(
        PROCESSOR_ID, min_pixels=16 * 28 * 28, max_pixels=64 * 28 * 28
    )
    model = tiny_model()
    device = "cpu"
    print(f"tiny model: {n_hidden_layers(model)} layers, hidden={model.config.hidden_size}")

    print("\nmodule resolution:")
    tm = text_model(model)
    layers = decoder_layers(model)
    check(isinstance(layers, torch.nn.ModuleList) and len(layers) == n_hidden_layers(model),
          "decoder_layers", f"{type(tm).__name__} with {len(layers)} layers")
    check(hasattr(final_norm(model), "weight"), "final_norm", type(final_norm(model)).__name__)
    W = lm_head_weight(model)
    check(W.shape == (model.config.vocab_size, model.config.hidden_size), "lm_head weight",
          str(tuple(W.shape)))
    check(all(hasattr(l, "self_attn") for l in layers), "layers expose self_attn")

    img_tok = image_token_id(model, processor)
    check(img_tok == 151655, "image token id", str(img_tok))
    yes_ids, no_ids = yes_no_token_ids(processor.tokenizer)
    check(bool(yes_ids) and bool(no_ids) and not set(yes_ids) & set(no_ids),
          "yes/no token ids", f"yes={yes_ids} no={no_ids}")
    yes_t, no_t = torch.tensor(yes_ids), torch.tensor(no_ids)

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    question = "Is there a bicycle in the image?"

    print("\nexpert pass:")
    inputs = build_inputs(processor, image, question, device=device)
    vis = vision_positions(inputs["input_ids"], img_tok)
    n_vis = int(vis.numel())
    check(n_vis > 4, "vision band located", f"{n_vis} vision tokens in input_ids")
    check(bool((vis[1:] - vis[:-1] == 1).all()), "vision band is contiguous")

    hs_e, logits_e = forward_hiddens(model, inputs)
    L = n_hidden_layers(model)
    check(hs_e.shape == (L + 1, model.config.hidden_size), "hidden states shape",
          f"{tuple(hs_e.shape)} (expected {(L + 1, model.config.hidden_size)})")

    m_e = lens_margins(model, hs_e, yes_t, no_t)
    true_margin = float(logits_e[yes_t].max() - logits_e[no_t].max())
    lens_err = abs(float(m_e[-1]) - true_margin)
    check(lens_err < 1e-3, "logit-lens fidelity at the final layer",
          f"|lens - true| = {lens_err:.2e}  (lens={m_e[-1]:+.4f} true={true_margin:+.4f})")
    check(not np.allclose(m_e[:-1], m_e[-1]), "lens varies across layers",
          f"range {m_e.min():+.3f}..{m_e.max():+.3f}")

    print("\nSID / CT2S:")
    sid_full = SidCT2S(model, rank_layer=1, keep_ratio=1.0)
    with amateur_context("sid", inputs, sid=sid_full, img_token_id=img_tok):
        hs_full, _ = forward_hiddens(model, inputs)
    check(sid_full.n_dropped == 0, "keep_ratio=1.0 drops nothing",
          f"{sid_full.n_keep}/{sid_full.n_vision} kept")
    check(torch.allclose(hs_full, hs_e, atol=1e-4), "keep_ratio=1.0 is a no-op",
          f"max|dh| = {float((hs_full - hs_e).abs().max()):.2e}  "
          f"(proves the substituted mask reproduces the model's own causal mask)")

    sid = SidCT2S(model, rank_layer=1, keep_ratio=0.10)
    with amateur_context("sid", inputs, sid=sid, img_token_id=img_tok):
        hs_sid, _ = forward_hiddens(model, inputs)
    expected_keep = max(1, min(int(round(0.10 * n_vis)), n_vis))
    check(torch.is_tensor(sid.attn_weights), "attention weights captured (eager)",
          str(tuple(sid.attn_weights.shape)) if torch.is_tensor(sid.attn_weights) else "None")
    check(sid.n_vision == n_vis and sid.n_keep == expected_keep
          and sid.n_dropped == n_vis - expected_keep,
          "CT2S token budget",
          f"{sid.n_keep}/{sid.n_vision} kept, {sid.n_dropped} masked (expected keep {expected_keep})")
    check(torch.is_tensor(sid.drop_idx) and bool(((sid.drop_idx >= vis.min())
                                                  & (sid.drop_idx <= vis.max())).all()),
          "masked columns are all inside the vision band")
    m_sid = lens_margins(model, hs_sid, yes_t, no_t)
    check(abs(float(m_e[-1] - m_sid[-1])) > 1e-6, "SID amateur differs from expert",
          f"Delta at readout = {float(m_e[-1] - m_sid[-1]):+.4f}")

    # the ranked scores must come from the ranking layer, and the ranking must pick the *least*
    # attended tokens -- verify the kept set is exactly the bottom-k of the captured scores
    scores = sid.attn_weights.float().mean(dim=1)[0, -1, :]
    kept = sorted(set(vis.tolist()) - set(sid.drop_idx.tolist()))
    bottom_k = set(vis[torch.topk(scores[vis], expected_keep, largest=False).indices].tolist())
    check(set(kept) == bottom_k, "kept set == least-attended bottom-k of SID Eq.5 scores")

    print("\nhooks are removed after the session:")
    hs_after, _ = forward_hiddens(model, inputs)
    check(torch.allclose(hs_after, hs_e, atol=1e-5), "expert pass unaffected after SID session",
          f"max|dh| = {float((hs_after - hs_e).abs().max()):.2e}")

    print("\nVCD / ICD amateurs:")
    torch.manual_seed(1234)
    vcd_inputs = build_amateur_inputs("vcd", processor, inputs, image, question, 900, device)
    check(vcd_inputs["input_ids"].equal(inputs["input_ids"]), "VCD keeps the expert prompt")
    check(not torch.allclose(vcd_inputs["pixel_values"], inputs["pixel_values"]),
          "VCD perturbs pixel_values",
          f"mean|dx| = {float((vcd_inputs['pixel_values'] - inputs['pixel_values']).abs().mean()):.4f}")
    hs_vcd, _ = forward_hiddens(model, vcd_inputs)
    m_vcd = lens_margins(model, hs_vcd, yes_t, no_t)
    check(abs(float(m_e[-1] - m_vcd[-1])) > 1e-6, "VCD amateur differs from expert",
          f"Delta at readout = {float(m_e[-1] - m_vcd[-1]):+.4f}")

    icd_inputs = build_amateur_inputs("icd", processor, inputs, image, question, 900, device)
    check(not icd_inputs["input_ids"].equal(inputs["input_ids"]),
          "ICD changes the prompt (adversarial system message)",
          f"{inputs['input_ids'].shape[1]} -> {icd_inputs['input_ids'].shape[1]} tokens")
    check(torch.allclose(icd_inputs["pixel_values"], inputs["pixel_values"]),
          "ICD keeps the clean image")
    hs_icd, _ = forward_hiddens(model, icd_inputs)
    m_icd = lens_margins(model, hs_icd, yes_t, no_t)
    check(abs(float(m_e[-1] - m_icd[-1])) > 1e-6, "ICD amateur differs from expert",
          f"Delta at readout = {float(m_e[-1] - m_icd[-1]):+.4f}")

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("PASSED: all model-side contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
