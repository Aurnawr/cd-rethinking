# cd-rethinking — `master` branch

A reproducibility study on the effectiveness of **Contrastive Decoding (CD)** for
mitigating hallucination in vision-language models. This folder contains the code,
experiment scripts, and results for the `master` line of work.

> Note: This branch's contents live entirely inside this `master/` folder so that
> multiple branches can later be merged into a single repository without collisions.

## Contents

| Path | Description |
| --- | --- |
| `inference/` | Inference code for the decoding methods (baseline, CD, and variants) on the MME and POPE benchmarks. |
| `eval/` | Evaluation utilities for scoring model outputs. |
| `scripts/` | Shell scripts to run inference and evaluation experiments end to end. |
| `outputs/` | Generated model outputs and result files (e.g. MME `.jsonl` and CSV results). |
| `plots/` | Figures and plotting artifacts produced from the results. |
| `R_score.py` | Computes the semantic flip ratio (R) between decoding methods. |
| `R_score_eda.py` | Exploratory analysis of the R-score results. |
| `transitions.py` | EDA on answer transitions across MME results. |
| `pyproject.toml` | Project configuration and dependencies. |
| `README_MME.md` | Notes on running the MME experiments. |
| `llava.egg-info/` | Package metadata for the LLaVA dependency. |

## Getting started

1. Install dependencies from `pyproject.toml`.
2. Use the scripts in `scripts/` to run inference (see `README_MME.md` for MME-specific steps).
3. Score outputs with the utilities in `eval/` and analyze with `R_score.py` / `transitions.py`.
