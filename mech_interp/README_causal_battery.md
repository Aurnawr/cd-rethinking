# CD mechanistic battery — causal + cross-method generalization

Extends the original POPE-COCO logit-lens experiment (see `README.md`) from a single
method (VCD, correlational) to a **causal** battery generalized across the repo's three
contrastive-decoding methods — **VCD, ICD, SID** — on LLaVA-1.5-7B, run on **Modal**.

Thesis under test: *contrastive decoding does not correct hallucinations at the point of
decision; it applies a uniform, one-sided shift of the Yes/No margin that is blind to
whether an answer is a hallucination.* We show this holds for every CD variant and back it
with causal (activation-patching) and constructive (steering) evidence.

## Why it generalizes cheaply

Every method forms the same contrastive logit algebra
`z_CD = (1+alpha) z_expert - alpha z_amateur` (+ plausibility mask); only the **amateur
forward pass** differs. Verified against the repo's inference code:

| method | amateur branch | where |
|---|---|---|
| VCD | diffusion-noised image, same prompt | `add_diffusion_noise(img, 900)` |
| ICD | clean image, adversarial system prompt | `conv.system = <negative>` |
| SID | clean image, `use_sid=True` (random 72/576 vision-token attention mask from layer 2) | `custom_modeling_llama.py` |

So the whole methodology is method-agnostic: swap the amateur (`amateur_branches.py`) and
rerun one battery. No existing repo model/inference file is modified.

## Files (all new, under `mech_interp/`)

| file | role |
|---|---|
| `amateur_branches.py` | per-method amateur constructor (VCD/ICD/SID) |
| `extract_cd_generic.py` | expert+amateur passes; per-layer logit-lens Yes/No margins for both; saves `logit_lens_per_sample_<method>.npz` + H∪T hiddens |
| `fit_probe_dirs.py` | per-layer hallucination-direction probes -> `probe_dirs.npz` |
| `subspace_align.py` | orthogonality test: cos(perturbation, hallucination direction) |
| `patch_causal.py` | activation patching: causal per-layer Yes/No effect on H |
| `steer_probe.py` | constructive counterfactual: probe-direction steering at the formation layer |
| `analyze_all.py` | cross-method figures + `summary_all.{json,csv}` |
| `modal_app.py` | Modal image (torch 2.0.1 / transformers 4.31.0, cu117) + volume + functions |
| `run_mech_modal.sh` | driver: `setup / smoke / extract / analyze / pull` |

## Experiments -> evidence

- **A. Decision-margin battery ×{VCD,ICD,SID}** — selectivity AUC of `-Delta`, fraction of
  H with `Delta>0`, R² by hallucination status. All non-selective + one-sided => general.
- **B. Subspace alignment ×3** — `cos ≈ 0` while `||Delta h||` large => perturbation is
  orthogonal to the hallucination subspace ("wrong subspace", not just "wrong layer").
- **C. Activation patching (causal), VCD** — causal per-layer effect, no linearity
  assumption; locates the true causal depth.
- **D. Steering counterfactual** — subtracting the probe direction at the formation layer
  flips hallucinations to No with low collateral: the targeted correction CD isn't.

## Run (Modal; CLI via uvx, auth from ~/.modal.toml)

```bash
bash mech_interp/run_mech_modal.sh setup     # image build + model + POPE-COCO data (one-time)
bash mech_interp/run_mech_modal.sh smoke      # 8-sample env validation (checks lens MAE ~ 0)
bash mech_interp/run_mech_modal.sh extract    # vcd/icd/sid full battery (parallel L4s, ~70 min)
bash mech_interp/run_mech_modal.sh analyze    # probes + align + patch + steer + figures
bash mech_interp/run_mech_modal.sh pull       # -> ~/Downloads/cd_pope_mech_interp_results/
```

Notes: L4 GPU (~$0.80/hr); model+data+hiddens live on the `cd-mech` Volume; only small
result files (npz/csv/png/json) are pulled. The extractor's `lens_final_layer_MAE` must be
~0 — this repo's custom model appends the final hidden state post-RMSNorm, so the lens
norms layers 0–31 and uses layer 32 as-is.
