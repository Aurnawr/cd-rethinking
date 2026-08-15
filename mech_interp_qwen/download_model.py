#!/usr/bin/env python3
"""Fetch the Qwen2.5-VL weights into persistent storage.

`--local-dir` gives a plain directory of files that `from_pretrained(<path>)` reads directly, so
the weights survive Studio restarts instead of living in an ephemeral HF cache. Idempotent:
already-present files are skipped, so a killed download resumes cheaply.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--target", required=True, help="local directory to materialize the repo into")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    os.makedirs(args.target, exist_ok=True)
    config = os.path.join(args.target, "config.json")
    verb = "verifying" if os.path.exists(config) else "downloading"
    print(f"[model] {verb} {args.repo_id} -> {args.target}")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.target,
        max_workers=args.max_workers,
        # skip the duplicate formats: safetensors is what transformers loads
        ignore_patterns=["*.pth", "*.bin", "*.msgpack", "*.h5", "original/*"],
    )
    if not os.path.exists(config):
        print(f"[model] ERROR: no config.json under {args.target} after download", file=sys.stderr)
        return 1
    print(f"[model] done: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
