"""
modal_app.py -- runs CHAIR_sampling+ICD/common/generate_llava.py and
generate_llava_icd.py on Modal (L4 GPU), detached and resumable, so it
survives your laptop closing.

This app is completely self-contained: it uses its own app name and volume
name, distinct from any other project's Modal app, so it is safe to deploy
under a fresh Modal account with no other setup.

One-time setup (downloads LLaVA-1.5-7B weights + the 500 COCO images +
annotations into a persistent Modal Volume -- only needs to run once, ever):

    modal run modal_app.py --action setup

Deploy the persistent app (do this once; re-run after any code edit):

    modal deploy modal_app.py

Then submit jobs asynchronously via spawn_jobs.py (recommended -- see
README.md), or directly:

    python -c "
    import modal
    f = modal.Function.from_name('cd-rethink-sampling-icd', 'generate')
    f.spawn(mode='baseline', method=None, decode='sample', seed=0,
            out_name='llava_baseline_seed0.jsonl', n_images=3)   # smoke test
    "

`modal deploy` + `.spawn()` keeps the job running on Modal's infrastructure
even if this terminal disconnects or the laptop closes -- your local process
only needs to stay connected for the few seconds it takes to submit the job.
Progress is committed to the Volume periodically (every ~60s), so a
killed/restarted run resumes from wherever it left off (both generation
scripts already skip images present in --out).
"""
import subprocess
import threading
from pathlib import Path

import modal

APP_NAME = "cd-rethink-sampling-icd"
VOL_NAME = "cd-rethink-sampling-icd-vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

HERE = Path(__file__).resolve().parent          # CHAIR_sampling+ICD/
VOL_MOUNT = "/vol"
REMOTE_ROOT = "/root/CHAIR_sampling_ICD"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers==5.12.1",
        "tokenizers==0.22.2",
        "accelerate==1.14.0",
        "sentencepiece==0.2.1",
        "einops==0.6.1",
        "numpy==1.26.4",
        "pillow==12.2.0",
        "matplotlib",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(HERE / "common"), remote_path=f"{REMOTE_ROOT}/common")
    .add_local_file(str(HERE / "image_ids_500.json"), remote_path=f"{REMOTE_ROOT}/image_ids_500.json")
)


@app.function(image=image, volumes={VOL_MOUNT: volume}, timeout=7200)
def setup_assets():
    """One-time, idempotent: LLaVA-1.5-7B weights + the 500 COCO val2017
    images + annotations, into the persistent volume."""
    import json, os, shutil, tempfile, urllib.request, zipfile
    from huggingface_hub import snapshot_download

    models_dir = f"{VOL_MOUNT}/models/llava-v1.5-7b"
    if not os.path.exists(f"{models_dir}/config.json"):
        print("[setup] downloading LLaVA-1.5-7B weights...", flush=True)
        snapshot_download("liuhaotian/llava-v1.5-7b", local_dir=models_dir)
        volume.commit()
    else:
        print("[setup] weights already present, skipping", flush=True)

    ann_dir = f"{VOL_MOUNT}/data/coco/annotations"
    os.makedirs(ann_dir, exist_ok=True)
    if not os.path.exists(f"{ann_dir}/instances_val2017.json"):
        print("[setup] downloading COCO val2017 annotations...", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            zpath = f"{tmp}/a.zip"
            urllib.request.urlretrieve(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip", zpath
            )
            with zipfile.ZipFile(zpath) as z:
                for name in ("annotations/instances_val2017.json", "annotations/captions_val2017.json"):
                    with z.open(name) as s, open(f"{ann_dir}/{os.path.basename(name)}", "wb") as d:
                        shutil.copyfileobj(s, d)
        volume.commit()
    else:
        print("[setup] annotations already present, skipping", flush=True)

    img_dir = f"{VOL_MOUNT}/data/coco/val2017"
    os.makedirs(img_dir, exist_ok=True)
    ids = json.load(open(f"{REMOTE_ROOT}/image_ids_500.json"))
    base = "http://images.cocodataset.org/val2017/"
    got = 0
    for i in ids:
        dst = f"{img_dir}/{i:012d}.jpg"
        if os.path.exists(dst):
            continue
        urllib.request.urlretrieve(base + f"{i:012d}.jpg", dst)
        got += 1
        if got % 100 == 0:
            print(f"[setup] {got} images downloaded...", flush=True)
            volume.commit()
    volume.commit()
    print(f"[setup] assets ready: {len(ids)} images total", flush=True)


def _symlink_models_and_data():
    import os
    for name in ("models", "data"):
        link = f"{REMOTE_ROOT}/{name}"
        target = f"{VOL_MOUNT}/{name}"
        if not os.path.exists(link):
            os.symlink(target, link)


def _run_with_periodic_commit(cmd):
    """Runs cmd, streaming stdout, while committing the volume every ~60s
    on a dedicated thread independent of subprocess output (the generation
    scripts only print once per 25 images, which can be minutes -- gating
    commits on stdout activity would leave long windows with no durable
    checkpoint)."""
    proc = subprocess.Popen(cmd, cwd=REMOTE_ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    stop_committing = threading.Event()

    def commit_loop():
        while not stop_committing.wait(60):
            volume.commit()

    committer = threading.Thread(target=commit_loop, daemon=True)
    committer.start()
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
    finally:
        stop_committing.set()
        committer.join()

    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[1]} exited with code {proc.returncode}")


@app.function(image=image, gpu="L4", volumes={VOL_MOUNT: volume}, timeout=4 * 3600)
def generate(mode: str, decode: str, seed: int, out_name: str,
             method: str = "", n_images: int = 500):
    """Runs generate_llava.py: --mode {baseline,capture} --decode {greedy,sample}."""
    _symlink_models_and_data()
    out_path = f"{VOL_MOUNT}/outputs/{out_name}"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "python", f"{REMOTE_ROOT}/common/generate_llava.py",
        "--mode", mode,
        "--decode", decode,
        "--seed", str(seed),
        "--n-images", str(n_images),
        "--out", out_path,
    ]
    if method:
        cmd += ["--method", method]
    print(f"[generate] {' '.join(cmd)}", flush=True)
    _run_with_periodic_commit(cmd)
    print(f"[generate] DONE -> {out_path}", flush=True)


@app.function(image=image, gpu="L4", volumes={VOL_MOUNT: volume}, timeout=4 * 3600)
def generate_icd(prompt_key: str, decode: str, seed: int, out_name: str, n_images: int = 500):
    """Runs generate_llava_icd.py: one disturbance prompt per call (the
    official ICD methodology: 5 separate full passes, not a randomly-mixed
    single run)."""
    _symlink_models_and_data()
    out_path = f"{VOL_MOUNT}/outputs/{out_name}"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "python", f"{REMOTE_ROOT}/common/generate_llava_icd.py",
        "--prompt-key", prompt_key,
        "--decode", decode,
        "--seed", str(seed),
        "--n-images", str(n_images),
        "--out", out_path,
    ]
    print(f"[generate_icd] {' '.join(cmd)}", flush=True)
    _run_with_periodic_commit(cmd)
    print(f"[generate_icd] DONE -> {out_path}", flush=True)


@app.local_entrypoint()
def main(action: str = "generate", mode: str = "baseline", method: str = "",
         decode: str = "sample", seed: int = 0, n_images: int = 500,
         out_name: str = ""):
    if action == "setup":
        setup_assets.remote()
        return
    if not out_name:
        out_name = f"llava_{method or mode}_seed{seed}.jsonl"
    generate.remote(mode=mode, decode=decode, seed=seed, out_name=out_name,
                     method=method, n_images=n_images)
