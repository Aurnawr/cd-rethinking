#!/usr/bin/env python3
"""Modal orchestration for the CD mechanistic battery (LLaVA-1.5-7B, POPE-COCO).

Runs the portable mech_interp scripts on a Modal L4 GPU against a persistent Volume
holding the model, POPE data, and results. The repo is live-mounted (not baked), so
iterating on the mech scripts needs no image rebuild.

Env note: this repo pins torch==2.0.1 / transformers==4.31.0 with a custom LLaVA; the
image reproduces that exact stack (cu118). The scripts self-manage sys.path to import
the repo's own `llava` + `cd_utils`, so no editable install is required.

Usage (CLI is not on PATH -> invoke via uvx):
  uvx --from modal modal run mech_interp/modal_app.py::smoke
  uvx --from modal modal run mech_interp/modal_app.py::download_model
  uvx --from modal modal run mech_interp/modal_app.py::download_data
  uvx --from modal modal run mech_interp/modal_app.py::extract --method vcd --max-samples 40
  uvx --from modal modal run mech_interp/modal_app.py::extract --method vcd
  uvx --from modal modal run mech_interp/modal_app.py::patch  --method vcd
  uvx --from modal modal run mech_interp/modal_app.py::steer
  uvx --from modal modal run mech_interp/modal_app.py::fit_probes
Then pull results:
  uvx --from modal modal volume get cd-mech results ~/Downloads/cd_pope_mech_interp_results --force
"""
import os
import subprocess

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root on the Mac
REMOTE_REPO = "/root/repo"
MODEL_DIR = "/vol/models/llava-v1.5-7b"
DATA_DIR = "/vol/data"
RESULTS_DIR = "/vol/results"
VISION_TOWER = "openai/clip-vit-large-patch14-336"
MODEL_REPO = "liuhaotian/llava-v1.5-7b"

_IGNORE = [
    ".git", ".git/**", "outputs", "outputs/**", "repro_outputs", "repro_outputs/**",
    "data", "data/**", "repro_aokqva", "repro_aokqva/**", "*.egg-info", "**/__pycache__",
    "**/*.pyc", ".venv*", "**/*.jsonl", "**/*.npz", "**/*.png", "**/*.pdf",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git")
    .pip_install("setuptools", "wheel")
    .pip_install("torch==2.0.1", "torchvision==0.15.2")  # PyPI wheel bundles CUDA runtime
    .pip_install(
        "transformers==4.31.0", "tokenizers>=0.12.1,<0.14", "sentencepiece==0.1.99",
        "accelerate==0.21.0", "peft==0.4.0",
        "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13",
        "numpy==1.26.4", "scikit-learn==1.2.2", "pydantic<2,>=1", "shortuuid",
        "pandas", "matplotlib", "requests", "tqdm",
        "hf_transfer", "huggingface_hub", "protobuf",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/vol/hf_home",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
        }
    )
    .add_local_dir(REPO, REMOTE_REPO, ignore=_IGNORE)
)

app = modal.App("cd-mech")
vol = modal.Volume.from_name("cd-mech", create_if_missing=True)
VOLUMES = {"/vol": vol}
GPU = "L4"


def _run(script: str, *cli_args: str):
    """Run a mech_interp script from the mounted repo; stream output; raise on failure."""
    cmd = ["python", os.path.join(REMOTE_REPO, "mech_interp", script), *map(str, cli_args)]
    print("[modal] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REMOTE_REPO, check=True)
    vol.commit()


# --------------------------------------------------------------------------- setup

@app.function(image=image, volumes=VOLUMES, timeout=2400)
def download_model():
    """Fetch LLaVA-1.5-7B weights + the CLIP vision tower into the Volume."""
    from huggingface_hub import snapshot_download

    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(MODEL_REPO, local_dir=MODEL_DIR)
    snapshot_download(VISION_TOWER)  # into HF_HOME=/vol/hf_home
    vol.commit()
    print("[download_model] done:", sorted(os.listdir(MODEL_DIR)))


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def download_data():
    """POPE-COCO question files + only the referenced COCO val2014 images."""
    _run("download_pope_data.py", "--data-dir", DATA_DIR)
    pope = os.path.join(DATA_DIR, "pope", "coco")
    print("[download_data] contents:", sorted(os.listdir(pope)))


# --------------------------------------------------------------------------- smoke

@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=1200)
def smoke(method: str = "vcd"):
    """Validate the pinned env end-to-end via a tiny extract (loads the model ONCE)."""
    import torch  # noqa
    import transformers  # noqa
    print(f"[smoke] torch {torch.__version__} cuda {torch.cuda.is_available()} "
          f"transformers {transformers.__version__}")
    # The extract subprocess loads the model + runs both branches through the full
    # pipeline (logit lens, amateur). Loading here too would double GPU memory -> OOM.
    _run("extract_cd_generic.py", "--method", method, "--model-path", MODEL_DIR,
         "--data-dir", DATA_DIR, "--out-dir", RESULTS_DIR, "--max-samples", "8",
         "--splits", "random")
    print("[smoke] OK")


# --------------------------------------------------------------------------- experiments

@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=7200, memory=32768)
def extract(method: str, max_samples: int = 0, splits: str = "random popular adversarial"):
    """Logit-lens battery for one CD method over POPE-COCO."""
    _run("extract_cd_generic.py", "--method", method, "--model-path", MODEL_DIR,
         "--data-dir", DATA_DIR, "--out-dir", RESULTS_DIR,
         "--max-samples", max_samples, "--splits", *splits.split())


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def fit_probes():
    """Per-layer probe directions (presence + hallucination) from saved H u T hiddens."""
    _run("fit_probe_dirs.py", "--results-dir", RESULTS_DIR)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=7200)
def patch(method: str = "vcd", max_samples: int = 0):
    """Activation patching: causal per-layer effect of the amateur perturbation on H."""
    _run("patch_causal.py", "--method", method, "--model-path", MODEL_DIR,
         "--data-dir", DATA_DIR, "--out-dir", RESULTS_DIR, "--max-samples", max_samples)


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=7200)
def steer(max_samples: int = 0):
    """Constructive counterfactual: probe-direction steering at the formation layer."""
    _run("steer_probe.py", "--model-path", MODEL_DIR, "--data-dir", DATA_DIR,
         "--out-dir", RESULTS_DIR, "--probe-dirs", os.path.join(RESULTS_DIR, "probe_dirs.npz"),
         "--max-samples", max_samples)


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def align():
    """Subspace-alignment summary (reads the big H u T hiddens on the Volume in place)."""
    _run("subspace_align.py", "--results-dir", RESULTS_DIR)


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def resid_stats(method: str):
    """Per-layer residual-magnitude CSV from hiddens_ht_<method>.npz (small output)."""
    _run("resid_stats.py", "--results-dir", RESULTS_DIR, "--method", method)


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def analyze():
    """Cross-method figures + summary tables, written to the Volume's results dir."""
    _run("analyze_all.py", "--results-dir", RESULTS_DIR)


@app.function(image=image, volumes=VOLUMES, timeout=600)
def ls():
    """List what's on the Volume (debug helper)."""
    for root in ("/vol/models", DATA_DIR, RESULTS_DIR):
        print(f"== {root} ==")
        for dp, _dn, fn in os.walk(root):
            for f in sorted(fn)[:50]:
                print(os.path.join(dp, f))
