# Stored mechanistic-interpretability results

Per-sample logit-lens margins and derived figures for the discriminative CD battery, for both
model families, plus the calibration experiments run on top of them.
Nothing here needs a GPU to re-analyse: every downstream script reads only the small
`logit_lens_per_sample_<method>.npz` files.

## Layout

| path | contents |
|---|---|
| `llava1.5-7b/` | LLaVA-1.5-7B run (33 hidden states, `mech_interp/`) |
| `qwen2.5-vl-7b/` | Qwen2.5-VL-7B-Instruct run (29 hidden states, `mech_interp_qwen/`) |
| `calibration/` | outputs of `mech_interp_calib/` (oracle calibration + scalar bias, both models) |

Both model directories carry the same core artifacts: `logit_lens_per_sample_{vcd,icd,sid}.npz`,
`extract_summary_*.json`, `per_layer*.csv`, `summary_all.{json,csv}`, `cross_method_battery.png`
and a `regen/` directory with the 12 per-method figures
(`{halluc_auc_vs_cd_effect,layerwise_battery,logit_lens_curves,cd_normalized_curves}_{vcd,icd,sid}.png`).

## The npz contract

```
clean_margin    [n, L]  float32   expert logit-lens Yes-No margin per layer
amateur_margin  [n, L]  float32   amateur logit-lens Yes-No margin per layer
gt              [n]     int       1 = object present, 0 = absent
pred            [n]     int       1 = model answered Yes, 0 = No
split           [n]     str       random | popular | adversarial
```

Layer `-1` is the readout, after the final norm.
The sign of `clean_margin[:, -1]` reproduces the recorded greedy prediction on 100% of samples
for all five full runs and 99.98% for the sixth, so the readout margin is the real decision and
not an approximation of it.

## Provenance and one conversion

The Qwen directory is the Lightning.ai run of `mech_interp_qwen/run_all_lightning.sh`, verbatim:
9,000 POPE-COCO questions across all three splits, `lens_final_layer_MAE = 0.0`,
SID at `rank_layer = 2`, `keep_ratio = 0.10`, VCD at `noise_step = 900`.

The LLaVA directory is the earlier run.
Its VCD margins came from the first-generation extractor, which stored the amateur pass under
the key `noisy_margin` and covered only VCD.
That file was rewritten once into `logit_lens_per_sample_vcd.npz` under the shared key names
above, with no change to any value, so that all six (model, method) pairs read through one code
path.
Its ICD and SID margins come from the later per-method extractor and already used the shared
contract.

## Model differences worth knowing before comparing numbers

Depth differs, so absolute layer indices are not comparable: LLaVA-1.5-7B has 33 hidden states
(32 blocks), Qwen2.5-VL-7B has 29 (28 blocks).
Cross-model figures plot fractional depth instead.

Hallucination counts differ a lot.
On the same 9,000 questions LLaVA answers Yes on 3,710 and hallucinates on 257 of them (6.9% of
its Yes answers); Qwen answers Yes on 3,693 and hallucinates on 141 (3.8%).
Qwen is the stronger model on this benchmark (87.90% vs 85.49% greedy accuracy), and the smaller
`|H|` widens every confidence interval computed on it.
