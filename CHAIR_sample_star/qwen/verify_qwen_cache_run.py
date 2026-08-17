"""
verify_qwen_cache_run.py -- convenience wrapper that submits verify_qwen_cache
on Modal and BLOCKS until it finishes, printing the result. Run this before
any real Qwen job (spawn_qwen_jobs.py --full) -- see README.md.

Usage:
  python verify_qwen_cache_run.py                       # default: 3 images, 20 steps, vcd+sid
  python verify_qwen_cache_run.py --n-images 5 --n-steps 30
"""
import argparse

import modal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--methods", default="vcd,sid")
    args = ap.parse_args()

    f = modal.Function.from_name("cd-rethink-sample-star", "verify_qwen_cache")
    print(f"Submitting verify_qwen_cache(n_images={args.n_images}, n_steps={args.n_steps}, "
          f"methods={args.methods!r}) -- this blocks until done, watch the Modal dashboard "
          f"or `modal app logs cd-rethink-sample-star` in another terminal for live progress.")
    try:
        f.remote(n_images=args.n_images, n_steps=args.n_steps, methods=args.methods)
        print("\nPASS -- generate_qwen.py's cached implementation is verified correct. "
              "Safe to run spawn_qwen_jobs.py --full.")
    except Exception as e:
        print(f"\nFAILED: {e}")
        print("Do NOT run spawn_qwen_jobs.py --full until this is resolved.")
        raise


if __name__ == "__main__":
    main()
