# CD mechanistic battery on Qwen2.5-VL-7B-Instruct (POPE-COCO)

A self-contained port of the `mech_interp/` logit-lens experiment from **LLaVA-1.5-7B** to
**Qwen2.5-VL-7B-Instruct**, covering the same three contrastive-decoding methods - **VCD, ICD,
SID** - with the corrected (paper-faithful) SID, and one master script that runs the whole thing
on a Lightning.ai Studio from a bare environment.

Thesis under test, unchanged: *contrastive decoding does not correct hallucinations at the point
of decision; it applies a uniform, one-sided shift of the Yes/No margin that is blind to whether
an answer is a hallucination.* If that holds on a second model family with a different visual
tokenizer, a different LLM backbone and a different depth, it is a property of the CD algebra
rather than of LLaVA.

```bash
# smoke test first - ALWAYS (~10 min, 40 questions/split)
MAX_SAMPLES=40 bash mech_interp_qwen/run_all_lightning.sh

# full run
bash mech_interp_qwen/run_all_lightning.sh
```

---

## What it produces

The 12 figures in `$ROOT/results/regen/` - the same set as
`cd_pope_mech_interp_results/regen/` from the LLaVA run, same filenames:

| figure | question it answers |
|---|---|
| `halluc_auc_vs_cd_effect_<m>.png` | where is the hallucination decided, vs where does CD's effect land? |
| `layerwise_battery_<m>.png` | 4-panel: is the information available, is it used, is the shift one-sided, is any of it "about" hallucination? |
| `logit_lens_curves_<m>.png` | directional Yes/No margins on hallucinated samples + where Delta lands |
| `cd_normalized_curves_<m>.png` | raw vs de-confounded residual-magnitude curves |

for `<m>` in `vcd`, `icd`, `sid`. Plus, in `$ROOT/results/`:

```
logit_lens_per_sample_{vcd,icd,sid}.npz   per-layer expert/amateur Yes-No margins, gt, pred, split
extract_summary_{vcd,icd,sid}.json        n, n_H, n_T, lens_final_layer_MAE, config, splits_done
per_layer.csv                             probe A/B accuracy+AUC+balanced acc, CD residual change
per_layer_normalized.csv                  the probe curves + de-confounded residual curves
per_layer_resid_{vcd,icd,sid}.csv         per-method residual magnitudes on H
summary.{txt,json}                        l*_truth, l*_halluc, l*_CD and the coincide verdict
summary_all.{json,csv}                    cross-method selectivity / one-sidedness / R^2
layerwise_curves.png                      probe curves + primary-method CD curve
cross_method_battery.png                  all three methods on the same four panels
```

## The three quantities

- **Probe A (truth)** `hidden_l -> object-present?` over all samples → `l*_truth`, the depth where
  the true answer is most decodable.
- **Probe B (hallucination)** restricted to ground-truth-No samples, `hidden_l -> did the model
  emit a false-positive "Yes"?` → `l*_halluc`, the hallucination-specific representation.
- **Delta_l = m_expert_l - m_amateur_l**, the per-unit-alpha margin shift each CD method applies at
  layer `l`. Since every method forms `z_CD = (1+a) z_expert - a z_amateur` and the margin is
  linear in logits, Delta is recovered offline from the two saved margin arrays.

If Delta cannot separate hallucinated Yes-answers (H) from correct ones (T) at *any* depth - while
Probe A shows the model plainly represents the difference - CD is not correcting; it is shifting.

---

## Running on Lightning.ai

### 0. Studio setup

```bash
cd /teamspace/studios/this_studio
git clone <your-fork-url> cd-rethinking
cd cd-rethinking
git checkout qwen2.5-vl-mech-interp
nvidia-smi                        # attach a GPU first
```

An **L4 (24 GB)** is enough. A10G / L40S / A100 are proportionally faster. bf16 7B weights are
~16 GB, and POPE sequences are short (~430 tokens: ~390 vision + ~40 text), so memory is not the
constraint - wall-clock is.

Do **not** `pip install -e .` on this branch. That installs the repo's LLaVA stack, which pins
`transformers==4.31.0` and cannot coexist with the version Qwen2.5-VL needs. The master script
builds an isolated venv instead.

### 1. Run

```bash
MAX_SAMPLES=40 bash mech_interp_qwen/run_all_lightning.sh   # smoke, ~10 min
bash mech_interp_qwen/run_all_lightning.sh                   # full, see timings below
```

Everything lands under `$ROOT` (default
`/teamspace/studios/this_studio/cd_pope_mech_interp_qwen`), which is persistent storage - weights,
images and completed splits all survive a Studio restart and are skipped on re-run.

### 2. What the master script does

| stage | what | where it runs |
|---|---|---|
| `env` | `python -m venv --system-site-packages` + `pip install -r requirements.txt` | CPU |
| `model` | `snapshot_download` of the ~16 GB checkpoint into `$ROOT/models` | CPU |
| `data` | POPE-COCO question files + only the referenced COCO val2014 images | CPU |
| `selfcheck` | 8-sample fail-fast validation (see below) | GPU |
| `extract` | the single pass over POPE: expert + 3 amateurs per question | GPU |
| `probes` | per-layer Probe A / Probe B | CPU (the slow analysis stage) |
| `resid` | per-method residual-magnitude CSVs | CPU |
| `figures` | the 12 regen figures + the cross-method battery | CPU |
| `package` | verifies all 12 figures exist, tars the small artifacts | CPU |

Run a subset with `STAGES`:

```bash
STAGES="figures package" bash mech_interp_qwen/run_all_lightning.sh
```

### 3. Timings (order of magnitude, L4)

| stage | full run (~9000 questions) |
|---|---|
| model download | 10-25 min, once |
| image download | 5-15 min, once (POPE reuses a small image pool; a few hundred to ~2k images) |
| extract | ~2-4 h - 4 forward passes per question |
| probes | 20-60 min - 29 layers x 2 probes x 5-fold CV on 3584-dim features |
| resid + figures | < 2 min |

Extraction checkpoints after every split and `--resume` (on by default) skips completed splits, so
a disconnect costs at most the split in flight.

### 4. Configuration

Every knob is an environment variable:

| var | default | meaning |
|---|---|---|
| `ROOT` | `/teamspace/studios/this_studio/cd_pope_mech_interp_qwen` | persistent working root |
| `MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | hub id, or a local checkpoint directory |
| `MAX_SAMPLES` | `0` (all) | cap questions per split - set `40` for a smoke test |
| `SPLITS` | `random popular adversarial` | POPE-COCO splits |
| `METHODS` | `vcd icd sid` | which amateurs to extract |
| `PRIMARY_METHOD` | `vcd` | whose CD curve goes in `per_layer.csv` / `layerwise_curves.png` |
| `DTYPE` | `bfloat16` | `float16` risks overflow on Qwen; `float32` doubles memory |
| `ATTN_IMPL` | `eager` | **keep this** - SID needs real attention weights |
| `NOISE_STEP` | `900` | VCD diffusion step, this repo's value |
| `SID_RANK_LAYER` | `2` | 0-indexed ranking layer (SID paper's 1-indexed "Layer i=3") |
| `SID_KEEP_RATIO` | `0.10` | fraction of least-attended vision tokens kept (SID paper) |
| `SID_KEEP_TOKENS` | *(unset)* | absolute override; set `100` for released-code parity |
| `MAX_PIXELS` | *(unset)* | cap vision tokens; e.g. `451584` = `576*28*28` → ≤576 tokens |
| `PROBE_JOBS` | `-1` (all cores) | CV parallelism in the probe stage; lower it if that stage runs out of RAM |
| `STAGES` | all | subset of stages to run |
| `RESUME` | `1` | skip already-completed splits |
| `USE_VENV` | `1` | isolate deps from the repo's LLaVA environment |

---

## SID: what "corrected" means here

The repo's shipped `use_sid` path (`llava/.../custom_modeling_llama.py`) selects the retained
vision tokens with `torch.randperm` - **random visual dropout, not SID**. `mech_interp/sid_correct.py`
fixed that for LLaVA. This branch reimplements the fix for Qwen from the paper.

**Config comes from the SID paper, not the released code.** The official repo
([huofushuo/SID](https://github.com/huofushuo/SID)) has no Qwen implementation at all - it covers
LLaVA-1.5, InstructBLIP, MiniGPT-4 and Shikra, and cites Qwen-VL only in the bibliography. But the
paper states the recipe as a **ratio**, which is exactly what makes it portable:

> *"we set Layer i=3 and preserve top 10% least important vision tokens for Shikra, LLaVA-1.5, and
> LLaVA-NeXT"* - arXiv 2408.02032, Sec. 5.1

The same 10% across models encoding 256 / 576 / 2304 vision tokens. So for Qwen2.5-VL's dynamic
resolution the faithful choice is the ratio, applied to whatever the image yields (~390 tokens for
COCO val2014 at native resolution - no capping needed, and `MAX_PIXELS` is there if you want it).

Two discrepancies in the upstream material, resolved explicitly rather than inherited:

- **count**: the released code's `--fast-v-attention-rank 100` is 17.4% of LLaVA's 576, not the
  paper's 10%. It is a `fast-v` default. `mech_interp/sid_correct.py` used it; this port defaults
  to the paper's 0.10 and exposes `SID_KEEP_TOKENS=100` for parity.
- **layer**: the paper says "Layer i=3" (1-indexed); the released code says `fast_v_agg_layer=2`,
  which ranks with the layer at 0-index 1. The port names the parameter unambiguously -
  `SID_RANK_LAYER` is the **0-indexed** decoder layer whose attention does the ranking, and the
  mask applies from `SID_RANK_LAYER + 1` onward. Paper → `2` (the default); released code → `1`.

### Implementation

`sid_ct2s.py` installs CT2S with **hooks only**, in a single forward pass, touching no model source:

1. a forward hook on `layers[rank_layer].self_attn` captures the attention weights (plus a
   pre-hook forcing `output_attentions=True`, because transformers 4.51's `Qwen2_5_VLAttention`
   nulls them out otherwise - eager alone is not sufficient);
2. SID Eq. 5: mean over heads of the **last query row**, restricted to the vision columns, which
   are located from `input_ids == config.image_token_id` (no hardcoded system-prompt length or
   fixed 576 offset - the LLaVA version had to guess `SYS_LENGTH=35`);
3. the lowest-scoring `round(keep_ratio * n_vision)` tokens are kept;
4. a pre-hook on every later layer swaps in a `[1,1,q,k]` **float additive** mask - the causal
   structure plus the dropped vision columns at `-inf`, applied to all query rows, exactly as SID's
   own mask does.

The float-additive form is the one both eager and sdpa accept, and `test_model_tiny.py` asserts
the strongest possible correctness property: with `keep_ratio=1.0` the substituted mask reproduces
the model's own forward **bit for bit** (`max|dh| = 0`), so the only thing SID changes is the
intended masking.

## The other two amateurs

| method | amateur |
|---|---|
| VCD | `add_diffusion_noise(pixel_values, 900)`. Qwen's `pixel_values` is a linear rearrange of the normalized image into flattened 14x14 patches, so elementwise Gaussian noise there is identical to noising the normalized image - the same operation VCD applies for LLaVA. |
| ICD | clean image, system message replaced with the canonical adversarial negative *"You are a confused objects detector to provide a fuzzy overview or impression of the image."* Fixed rather than sampled from the repo's five-prompt list, so Delta carries no per-sample sampling noise. |

---

## Design notes / deviations from the LLaVA pipeline

- **One extraction pass, not four.** The LLaVA pipeline ran `extract_activations.py` (2 forwards)
  and then `extract_cd_generic.py` once per method (2 forwards each) = 8 forwards/sample,
  recomputing the identical expert pass four times. `extract_cd.py` computes the expert pass once
  and shares it: **4 forwards/sample**, one resumable job. `test_pipeline_tiny.py` asserts the
  three methods' `clean_margin` arrays are bit-identical, which is what "shared" has to mean.
- **No `hiddens_ht_*.npz`.** The LLaVA run dumped multi-GB fp16 hidden tensors for the H∪T
  population purely so `resid_stats.py` could take norms of them. Those norms are now recorded in
  fp32 during the forward pass (`clean_norm`, `cd_change_<m>` in each shard), so the stats are more
  accurate *and* ~5 GB smaller. Nothing else consumed those dumps once the causal battery was out
  of scope.
- **No causal battery.** Activation patching, probe-direction steering and subspace alignment are
  deliberately not ported - they produce none of the target figures and would multiply GPU time.
  `mech_interp/{patch_causal,steer_probe,subspace_align,fit_probe_dirs}.py` remain on the LLaVA
  branch if they are ever wanted here.
- **Fewer CV fits.** `mech_interp/train_probes.py` ran three separate cross-validation passes per
  probe per layer (`cross_val_score` + `cross_val_predict(proba)` + `cross_val_predict`). This one
  runs a single `predict_proba` pass and derives accuracy and balanced accuracy from the same
  out-of-fold probabilities - identical numbers for logistic regression, a third of the fits.
- **Layer count from the data.** Qwen2.5-VL-7B is 28 blocks → 29 hidden states, d=3584 (LLaVA-1.5
  is 32 → 33, d=4096). Nothing is hardcoded; the same figure code renders both.
- **The logit lens is unchanged in spirit.** HuggingFace's Qwen2 decoder appends the final hidden
  state *after* `self.norm`, the same contract the LLaVA port relies on: layers `0..L-1` go through
  the frozen final RMSNorm, layer `L` is used as-is. `lens_final_layer_MAE` in each
  `extract_summary_*.json` must be ~0 - it is the guard that the contract still holds.
- **No existing repo file is modified.** `mech_interp/download_pope_data.py` is reused as-is (it is
  pure requests/json with no model dependency); `add_diffusion_noise` is copied into
  `vcd_noise.py` with attribution only because its home module imports transformers-4.31-only
  symbols and cannot be imported here.

## Version pinning

`transformers==4.51.3` is pinned. Qwen2.5-VL landed in 4.49 and its internals have churned since
(module layout moved to `model.model.language_model` in 4.52; the attention classes were replaced
by the `ALL_ATTENTION_FUNCTIONS` interface; mask construction moved to `masking_utils`). The code
resolves submodules by *shape* rather than by path and probes the attention signature at runtime,
so a bump is survivable - but **run the tests below after any bump** rather than trusting it.

torch is deliberately *not* pinned: Studio images ship a CUDA-matched build, and reinstalling torch
from PyPI is the most common way to end up CPU-only. The venv is created with
`--system-site-packages` to reuse it, and the `env` stage errors out if torch is missing or older
than 2.1.

---

## Tests

Three layers, cheapest first. The first two need no GPU, no checkpoint and no dataset.

```bash
python mech_interp_qwen/test_analysis_synthetic.py   # ~30 s, needs only numpy/pandas/sklearn/matplotlib
python mech_interp_qwen/test_model_tiny.py           # ~30 s, needs torch+transformers (CPU)
python mech_interp_qwen/test_pipeline_tiny.py        # ~2-3 min, CPU
```

| test | covers |
|---|---|
| `test_analysis_synthetic.py` | everything downstream of the forward pass: shard contract, both probe CSVs, all 12 regen figures, the cross-method battery |
| `test_model_tiny.py` | model-side contracts on a 4-layer random Qwen2.5-VL: module resolution, logit-lens fidelity, CT2S token selection and mask equivalence, hook cleanup, all three amateurs differing from the expert |
| `test_pipeline_tiny.py` | the real stage sequence end to end, including `extract_cd.py`, per-split checkpointing and `--resume` |

`selfcheck.py` is the fourth layer and runs on the real checkpoint as part of the master script.
It repeats the structural checks and adds the ones that need real weights:

1. **layer count** - `L+1` hidden states.
2. **lens fidelity** - `|lens_margin[L] - true_margin| < 1e-2`. The single most important check: a
   broken norm/head contract corrupts every figure without erroring.
3. **generate vs logit-argmax** - `generate(max_new_tokens=5)`'s first token matches the forward
   argmax used as `pred` (the same check the LLaVA pipeline runs).
4. **Yes/No coverage** - the greedy first token really is Yes-ish or No-ish; if Qwen answers with
   prose, the prompt is wrong and the whole margin is meaningless.
5. **attention capture** - the CT2S hook receives real attention weights.
6. **SID token budget** - exactly `n_vision - round(ratio*n_vision)` vision columns are masked.
7. **amateurs differ** - all three amateur margins differ from the expert.

### The scientific sanity gate

`selfcheck` and the extractor both print **baseline greedy accuracy per split**. Qwen2.5-VL-7B
should land well above chance on `coco-random`, lower on `adversarial`. If it is near 50%, the
prompt or preprocessing is wrong - stop and fix it before trusting any probe, because every
downstream number is conditioned on `pred`.

Also watch, in `train_probes`' output:

- **class balance and hallucination rate.** If the gt-No hallucination rate is under ~10%, `n_H` is
  small and the selectivity AUC has wide error bars. Prefer `bal_acc` / `auc` over raw accuracy -
  the peak selection already uses AUC for exactly this reason.
- **`lens_final_layer_MAE`** in each `extract_summary_*.json`: must be ~0.

## Where the results live, and what was run on top of them

The completed Lightning run is committed at `results/qwen2.5-vl-7b/`: 9,000 POPE-COCO questions
across all three splits, three methods, `lens_final_layer_MAE = 0.0`, plus the 12 `regen/`
figures. The matching LLaVA-1.5-7B run is at `results/llava1.5-7b/`. See `results/README.md` for
the npz contract and the model-vs-model caveats (33 vs 29 hidden states, 257 vs 141
hallucinations).

Two further experiments from the write-up run on those stored margins, for both models, with no
GPU: oracle calibration of the selectivity metric and the scalar-bias sweep. See
`mech_interp_calib/README.md`.
