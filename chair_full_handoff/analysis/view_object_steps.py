"""
view_object_steps.py -- markdown dump of ONLY the decoding steps that emit a
CHAIR object mention (real or hallucinated), with each step tagged REAL /
HALLUCINATED. Words are decoded (not ids); top-30 of expert / amateur /
contrastive(2E-A) shown side by side per step.

Object attribution is caption-level (the corrected CHAIR procedure), and each
mention is mapped to the decoding step that produced its first token.

Usage:
  python view_object_steps.py --model LLaVA --method vcd --n 60
"""
import argparse, json, sys, math
from pathlib import Path

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LLAVA_ROOT = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
HERE = Path(__file__).resolve().parent
COCO = LLAVA_ROOT / "data" / "coco" / "annotations"

CFG = {
    "LLaVA": {"topk": LLAVA_ROOT / "llava_logit_eda" / "outputs_full_topk",
              "tok": LLAVA_ROOT / "llava_logit_eda" / "models" / "llava-v1.5-7b", "slow": True},
    "Qwen":  {"topk": REPO / "CHAIR-qwen-detailed" / "outputs_full_topk",
              "tok": REPO / "CHAIR-qwen-detailed" / "models" / "Qwen2.5-VL-7B-Instruct", "slow": False},
}


def wd(tok, tid):
    s = tok.decode([tid]).replace("\n", "\\n").replace("|", "\\|")
    return s if s.strip() else "␣"  # visible marker for whitespace-only tokens


def softmax(logits):
    """softmax over the SHOWN top-30 logits (tail mass excluded -> slight
    over-estimate; exact full-vocab softmax needs re-running the model)."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [e / z for e in exps]


def step_table(tok, s):
    """markdown table of top-30 E / A / CD for one step, with logits and
    (top-30) softmax probabilities per branch."""
    chosen = s["chosen_id"]
    E = list(zip(s["expert_top_ids"], s["expert_top_logits"]))
    A = list(zip(s["amateur_top_ids"], s["amateur_top_logits"]))
    C = list(zip(s["cd_top_ids"], s["cd_top_pre_apc_logits"]))
    pE = softmax(s["expert_top_logits"])
    pA = softmax(s["amateur_top_logits"])
    pC = softmax(s["cd_top_pre_apc_logits"])
    apc = s["cd_top_survives_apc"]
    out = ["| # | Expert | E | P(E) | Amateur | A | P(A) | Contrastive 2E−A | C | P(C) | APC |",
           "|--:|:--|--:|--:|:--|--:|--:|:--|--:|--:|:-:|"]
    for i in range(len(E)):
        def cell(pair, p):
            tid, lg = pair
            mark = " ⬅" if tid == chosen else ""
            return f"`{wd(tok, tid)}`{mark}", f"{lg:.2f}", f"{p*100:.1f}%"
        ew, el, ep = cell(E[i], pE[i])
        aw, al, ap = cell(A[i], pA[i])
        cw, cl, cp = cell(C[i], pC[i])
        surv = "✓" if (i < len(apc) and apc[i]) else ""
        out.append(f"| {i} | {ew} | {el} | {ep} | {aw} | {al} | {ap} | {cw} | {cl} | {cp} | {surv} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["LLaVA", "Qwen"], default="LLaVA")
    ap.add_argument("--method", choices=["vcd", "sid"], default="vcd")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = CFG[args.model]
    sys.path.insert(0, str(LLAVA_ROOT))
    from llava_logit_eda.object_mentions import all_mentions, load_chair
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(cfg["tok"]), use_fast=not cfg["slow"])

    path = cfg["topk"] / f"captions_topk_{args.method}.jsonl"
    recs = [json.loads(l) for l in open(path)][: args.n]
    chair = load_chair([r["image_id"] for r in recs], str(COCO))

    out = Path(args.out) if args.out else HERE / f"object_steps_{args.model}_{args.method}_{args.n}img.md"
    doc = [f"# Object-emitting decoding steps — {args.model} / {args.method.upper()} "
           f"(first {len(recs)} images)",
           "",
           "Only steps that emit a CHAIR object mention are shown. Each is tagged "
           "**REAL** or **HALLUCINATED** (a mention is hallucinated iff its "
           "canonical MSCOCO-80 category is absent from the image's ground truth). "
           "`⬅` marks the token CD actually emitted; ✓ in the APC column marks "
           "contrastive candidates that survive the plausibility cutoff.",
           "",
           "**Columns:** `E`/`A`/`C` are the raw logits of the expert, amateur and "
           "contrastive (2E−A, pre-APC) branches; `P(E)`/`P(A)`/`P(C)` are their "
           "softmax probabilities. _Note:_ the capture stored only each branch's "
           "top-30 logits, so these probabilities are a softmax **over the shown "
           "top-30**, not the full vocabulary — they slightly over-estimate the "
           "true probability (the excluded tail carries a little mass), but for the "
           "top candidates the difference is negligible.",
           ""]
    n_real = n_hall = 0
    for r in recs:
        steps = r["steps"]
        chosen = [s["chosen_id"] for s in steps]
        gt = sorted(chair.imid_to_objects.get(r["image_id"], set()))
        ms = [m for m in all_mentions(chair, tok, r["image_id"], chosen)
              if m["category"] == "object"]
        doc += [f"\n---\n\n## Image {r['image_id']}", "",
                f"**Caption:** {r['caption'].strip()}", "",
                f"**Ground-truth objects ({len(gt)}):** {', '.join(gt) if gt else '(none)'}", ""]
        if not ms:
            doc += ["_No CHAIR object mentions in this caption._", ""]
            continue
        for m in ms:
            si = m["step_indices"][0]
            if si >= len(steps):
                continue
            tag = "🔴 **HALLUCINATED**" if m["hallucinated"] else "🟢 **REAL**"
            if m["hallucinated"]: n_hall += 1
            else: n_real += 1
            s = steps[si]
            doc += [f"### step {si} — mention `{m['word']}` → category "
                    f"`{m['node_word']}` — {tag}",
                    f"emitted token: `{wd(tok, s['chosen_id'])}`  |  "
                    f"APC cutoff: {s['apc_cutoff']}", "",
                    step_table(tok, s), ""]
    Path(out).write_text("\n".join(doc))
    mb = out.stat().st_size / 1048576
    print(f"wrote {out}  ({len(recs)} images, {n_real} real + {n_hall} hallucinated "
          f"object steps, {mb:.1f} MB)")


if __name__ == "__main__":
    main()
