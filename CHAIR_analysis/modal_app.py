"""
modal_app.py -- runs CHAIR_analysis/common/generate_llava.py on Modal (L4 GPU),
detached and resumable, so it survives your laptop closing.

One-time setup (downloads LLaVA-1.5-7B weights + the 500 COCO images into a
persistent Modal Volume -- only needs to run once, ever):

    modal run --detach modal_app.py --action setup

Smoke test (do this before any full run -- cheap, few minutes, catches
environment problems before they cost real GPU time):

    modal run --detach modal_app.py --action generate --mode capture --method vcd \
        --seed 0 --n-images 3 --out-name smoke_vcd_seed0.jsonl

Full run, one example (repeat per seed/method/stats-source per the plan):

    modal run --detach modal_app.py --action generate --mode capture --method sid \
        --seed 0 --n-images 500 --out-name vcd_capture/sid_seed0.jsonl

`--detach` means the job keeps running on Modal's infrastructure even if this
terminal disconnects or the laptop closes. Progress is committed to the
Volume periodically (every ~60s), so a killed/restarted run resumes from
wherever it left off (generate_llava.py already skips images already present
in --out).
"""
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "cd-rethink-chair"
VOL_NAME = "cd-rethink-chair-vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

HERE = Path(__file__).resolve().parent          # CHAIR_analysis/
VOL_MOUNT = "/vol"
REMOTE_ROOT = "/root/CHAIR_analysis"

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
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(HERE / "common"), remote_path=f"{REMOTE_ROOT}/common")
    .add_local_file(str(HERE / "image_ids_500.json"), remote_path=f"{REMOTE_ROOT}/image_ids_500.json")
    .add_local_file(str(HERE / "proxy_stats_llava.json"), remote_path=f"{REMOTE_ROOT}/proxy_stats_llava.json")
)


@app.function(image=image, volumes={VOL_MOUNT: volume}, timeout=3600)
def setup_assets():
    """One-time, idempotent: LLaVA-1.5-7B weights + the 500 COCO val2017
    images + annotations, into the persistent volume. Mirrors
    CHAIR_analysis/download_assets.sh but writes to /vol and only fetches
    LLaVA (this plan is LLaVA-only)."""
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


@app.function(image=image, gpu="L4", volumes={VOL_MOUNT: volume}, timeout=6 * 3600)
def generate(mode: str, seed: int, out_name: str, method: str = "",
             stats_source: str = "vcd", n_images: int = 500):
    """Runs generate_llava.py inside the container, with models/ and data/
    symlinked from the persistent volume so the script's existing
    relative-path logic works completely unmodified. Commits the volume
    every ~60s so partial progress survives a container restart."""
    import os

    for name in ("models", "data"):
        link = f"{REMOTE_ROOT}/{name}"
        target = f"{VOL_MOUNT}/{name}"
        if not os.path.exists(link):
            os.symlink(target, link)

    out_path = f"{VOL_MOUNT}/outputs/{out_name}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "python", f"{REMOTE_ROOT}/common/generate_llava.py",
        "--mode", mode,
        "--seed", str(seed),
        "--n-images", str(n_images),
        "--out", out_path,
    ]
    if method:
        cmd += ["--method", method]
    if mode == "proxy":
        cmd += ["--stats-source", stats_source]

    print(f"[generate] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=REMOTE_ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Commit on a fixed wall-clock interval, independent of subprocess
    # output -- generate_llava.py only prints once per 25 images, so
    # gating commits on stdout activity (as before) left multi-minute
    # windows with no durable checkpoint. A dedicated thread guarantees
    # partial progress is actually persisted even during silent stretches.
    import threading
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
        raise RuntimeError(f"generate_llava.py exited with code {proc.returncode}")
    print(f"[generate] DONE -> {out_path}", flush=True)


@app.local_entrypoint()
def main(action: str = "generate", mode: str = "capture", method: str = "",
         stats_source: str = "vcd", seed: int = 0, n_images: int = 500,
         out_name: str = ""):
    if action == "setup":
        setup_assets.remote()
        return
    if not out_name:
        out_name = f"{mode}_{method or stats_source}_seed{seed}.jsonl"
    generate.remote(mode=mode, seed=seed, out_name=out_name,
                     method=method, stats_source=stats_source, n_images=n_images)
