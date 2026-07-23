"""
view_logits.py -- render the top-30 logit captures as HUMAN-READABLE text
(words, not token ids). Decodes each token id to its string and lays the
expert / amateur / contrastive(2E-A) top-30 side by side for every step.

Usage:
  python view_logits.py --model LLaVA --method vcd                 # full dump -> readable_LLaVA_vcd.txt
  python view_logits.py --model LLaVA --method vcd --image 724     # one image to stdout
  python view_logits.py --model LLaVA --method vcd --n 20          # first 20 images only
  python view_logits.py --model Qwen  --method vcd

'*' marks the token CD actually emitted; '=' after a branch's #0 marks it matching
the expert's #0. 'APC' column shows which contrastive candidates survive the
plausibility cutoff.
"""
import argparse, json, sys
from pathlib import Path

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent

CFG = {
    "LLaVA": {
        "topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
        "tok":  LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b",
        "slow": True,
    },
    "Qwen": {
        "topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
        "tok":  REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct",
        "slow": False,
    },
}


def w(tok, tid):
    """decode one id to a display string with visible spaces/newlines."""
    s = tok.decode([tid])
    return s.replace("\n", "\\n").replace("\t", "\\t") or "∅"


def render_step(tok, s, buf):
    E = list(zip(s["expert_top_ids"], s["expert_top_logits"]))
    A = list(zip(s["amateur_top_ids"], s["amateur_top_logits"]))
    C = list(zip(s["cd_top_ids"], s["cd_top_pre_apc_logits"]))
    apc = s["cd_top_survives_apc"]
    chosen = s["chosen_id"]
    e0 = s["expert_top_ids"][0]

    buf.append(f"  step {s['step']:>3} | emitted: {w(tok, chosen)!r:>14}  "
               f"| APC cutoff {s['apc_cutoff']}")
    buf.append(f"      {'#':>2}  {'EXPERT E':>26}  {'AMATEUR A':>26}  {'CONTRASTIVE 2E-A':>26}  APC")
    for i in range(len(E)):
        def cell(pair, mark_match=False):
            tid, lg = pair
            star = "*" if tid == chosen else " "
            eqm = "=" if (mark_match and i == 0 and tid == e0) else " "
            return f"{star}{w(tok, tid)!r:>18} {lg:8.3f}{eqm}"
        surv = "Y" if (i < len(apc) and apc[i]) else "."
        buf.append(f"      {i:>2}  {cell(E[i])}  {cell(A[i], True)}  {cell(C[i], True)}   {surv}")
    buf.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["LLaVA", "Qwen"], default="LLaVA")
    ap.add_argument("--method", choices=["vcd", "sid"], default="vcd")
    ap.add_argument("--image", type=int, default=None, help="single image_id -> stdout")
    ap.add_argument("--n", type=int, default=None, help="first N images only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = CFG[args.model]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(cfg["tok"]), use_fast=not cfg["slow"])
    path = cfg["topk"] / f"captions_topk_{args.method}.jsonl"

    if args.image is not None:
        for line in open(path):
            r = json.loads(line)
            if r["image_id"] == args.image:
                buf = [f"=== image {r['image_id']} | {args.model} {args.method.upper()} "
                       f"| {len(r['steps'])} steps ===",
                       f"caption: {r['caption']}", ""]
                for s in r["steps"]:
                    render_step(tok, s, buf)
                print("\n".join(buf))
                return
        sys.exit(f"image {args.image} not found in {path}")

    out = Path(args.out) if args.out else HERE / f"readable_{args.model}_{args.method}.txt"
    n = 0
    with open(out, "w") as f:
        for line in open(path):
            r = json.loads(line)
            buf = [f"{'='*100}",
                   f"IMAGE {r['image_id']} | {args.model} {args.method.upper()} | {len(r['steps'])} steps",
                   f"caption: {r['caption']}", ""]
            for s in r["steps"]:
                render_step(tok, s, buf)
            f.write("\n".join(buf) + "\n")
            n += 1
            if args.n and n >= args.n:
                break
            if n % 100 == 0:
                print(f"  {n} images written...", flush=True)
    mb = out.stat().st_size / 1048576
    print(f"wrote {out}  ({n} images, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
