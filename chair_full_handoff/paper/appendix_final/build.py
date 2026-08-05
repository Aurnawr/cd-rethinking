"""
Builds the appendix deliverable for the two chosen examples
(hallucinated 'bear', image 47010; hallucinated 'car', image 481582):

  photo_<id>.jpg  : the COCO image on its own
  bars_<id>.pdf   : the three logit panels (Expert E, Amateur A, Contrastive 2E-A)
  appendix.tex    : ONE self-contained LaTeX file -- caption, the hallucinated
                    object, and the top-10 logit table (native LaTeX, matching the
                    bar plots). Compiles on its own and can also be pasted into a paper.

4 images + 1 .tex, all in this folder.
"""
import json, sys, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

REPO = Path("/teamspace/studios/this_studio/cd-rethinking")
LR = REPO / "llava_logit_eda" / "CHAIR-llava-detailed"
COCO = LR / "data" / "coco" / "annotations"
IMGDIR = LR / "data" / "coco" / "val2017"
OUT = REPO / "paper" / "appendix_final"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(LR))
from llava_logit_eda.object_mentions import all_mentions, load_chair
from transformers import AutoTokenizer

C_NORM = "#4C78A8"; C_HALL = "#E45756"; C_MASK = "#B0A9A0"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#888888", "axes.linewidth": 0.8})

EXAMPLES = [(47010, 98), (481582, 73)]   # bear, car

tok = AutoTokenizer.from_pretrained(str(LR / "llava_logit_eda" / "models" / "llava-v1.5-7b"), use_fast=False)
recs = {r["image_id"]: r for r in
        (json.loads(l) for l in open(LR / "llava_logit_eda" / "outputs_full_topk" / "captions_topk_vcd.jsonl"))}
chair = load_chair(list(recs), str(COCO))


def wd(tid):
    s = tok.decode([tid]); return "␣" if not s.strip() else s.strip()


def tex_escape(s):
    for a, b in [("\\", r"\textbackslash "), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde "), ("^", r"\textasciicircum "),
                 ("␣", r"\textvisiblespace ")]:
        s = s.replace(a, b)
    return s


def gather(image_id, step, K=12):
    r = recs[image_id]; s = r["steps"][step]; chosen = s["chosen_id"]
    emap = dict(zip(s["expert_top_ids"], s["expert_top_logits"]))
    amap = dict(zip(s["amateur_top_ids"], s["amateur_top_logits"]))
    cmap = dict(zip(s["cd_top_ids"], s["cd_top_pre_apc_logits"]))
    rows = []
    for tid in s["expert_top_ids"][:K]:
        E = emap[tid]; A = amap.get(tid)
        C = (2 * E - A) if A is not None else cmap.get(tid)
        rows.append({"tid": tid, "word": wd(tid), "E": E, "A": A, "C": C, "chosen": tid == chosen})
    chosen_seq = [st["chosen_id"] for st in r["steps"]]
    word = node = None
    for m in all_mentions(chair, tok, image_id, chosen_seq):
        if m["category"] == "object" and m["hallucinated"] and m["step_indices"][0] == step:
            word, node = m["word"], m["node_word"]; break
    return r, s, rows, chosen, word, node


def panel(ax, rows, key, title, cutoff=None):
    xs = np.arange(len(rows))
    vals = [(x[key] if x[key] is not None else np.nan) for x in rows]
    colors, hatches = [], []
    for x in rows:
        if key == "C" and cutoff is not None and x["C"] is not None and x["C"] < cutoff:
            colors.append(C_MASK); hatches.append("///")
        elif x["chosen"]:
            colors.append(C_HALL); hatches.append("")
        else:
            colors.append(C_NORM); hatches.append("")
    bars = ax.bar(xs, vals, color=colors, edgecolor="white", linewidth=0.6, width=0.72)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    if cutoff is not None:
        ax.axhline(cutoff, ls="--", lw=1.3, color="#333333", zorder=5)
        ax.text(len(rows) - 0.4, cutoff, " APC cutoff", va="bottom", ha="right",
                fontsize=8.5, color="#333333")
    ax.set_xticks(xs)
    ax.set_xticklabels([x["word"] for x in rows], rotation=45, ha="right", fontsize=8.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3)
    for xi, x in enumerate(rows):
        if x["chosen"] and not np.isnan(vals[xi]):
            ax.annotate("emitted", (xi, vals[xi]), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8, color=C_HALL, fontweight="bold")
    return vals


def make_photo(image_id):
    dst = OUT / f"photo_{image_id}.jpg"
    shutil.copyfile(IMGDIR / f"{image_id:012d}.jpg", dst)
    return dst


def make_bars(image_id, step):
    r, s, rows, chosen, word, node = gather(image_id, step)
    cutoff = s["apc_cutoff"]
    fig, (axE, axA, axC) = plt.subplots(1, 3, figsize=(11.4, 3.6))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.24, wspace=0.24)
    vE = panel(axE, rows, "E", "Expert  (E)")
    vA = panel(axA, rows, "A", "Amateur  (A)")
    vC = panel(axC, rows, "C", "Contrastive  (2E$-$A)", cutoff=cutoff)
    allv = [v for v in vE + vA + vC if not np.isnan(v)] + [cutoff]
    lo, hi = min(allv), max(allv); pad = 0.06 * (hi - lo)
    for ax in (axE, axA, axC):
        ax.set_ylim(lo - pad, hi + pad + 1.2)
    axE.set_ylabel("logit", fontsize=10.5)
    legend = [Patch(facecolor=C_HALL, label="hallucinated token (emitted)"),
              Patch(facecolor=C_NORM, label="other top-12 candidates"),
              Patch(facecolor=C_MASK, hatch="///", label="below APC cutoff (masked)"),
              Line2D([0], [0], ls="--", color="#333333", label="APC plausibility cutoff")]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9.0,
               bbox_to_anchor=(0.5, -0.02))
    out = OUT / f"bars_{image_id}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300); plt.close(fig)
    return out


def table_tex(image_id, step):
    r, s, rows, chosen, word, node = gather(image_id, step, K=10)
    cutoff = s["apc_cutoff"]
    lines = [r"\begin{tabular}{r l r r r c}", r"\toprule",
             r"Rank & Token & $E$ & $A$ & $2E-A$ & APC \\", r"\midrule"]
    for rk, x in enumerate(rows):
        A = "--" if x["A"] is None else f"{x['A']:.2f}"
        C = "--" if x["C"] is None else f"{x['C']:.2f}"
        surv = "" if x["C"] is None else ("kept" if x["C"] >= cutoff else "masked")
        cells = [str(rk), tex_escape(x["word"]), f"{x['E']:.2f}", A, C, surv]
        if x["chosen"]:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def info(image_id, step):
    r, s, rows, chosen, word, node = gather(image_id, step)
    E = next(x["E"] for x in rows if x["chosen"])
    A = next(x["A"] for x in rows if x["chosen"])
    C = next(x["C"] for x in rows if x["chosen"])
    aids = s["amateur_top_ids"]
    a_rank = aids.index(chosen) if chosen in aids else None
    a_rank_str = f"rank {a_rank + 1}" if a_rank is not None else "below its top 30"
    cap = " ".join(r["caption"].split())   # collapse newlines/blank lines -> single spaces
    return dict(cap=tex_escape(cap), word=tex_escape(word),
                node=tex_escape(node), tok=tex_escape(wd(chosen)),
                E=E, A=A, C=C, d=E - A, cutoff=s["apc_cutoff"], a_rank_str=a_rank_str)


def block(idx, image_id, step):
    d = info(image_id, step)
    return rf"""
\subsection*{{Example {idx}: LLaVA-1.5-7B hallucinates ``{d['word']}''}}

\noindent\textbf{{Hallucinated object:}} ``{d['word']}'' (COCO category
\emph{{{d['node']}}}), which is \emph{{not present}} in the image.

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.55\linewidth]{{photo_{image_id}.jpg}}
  \caption{{COCO image {image_id}. VCD caption generated by LLaVA-1.5-7B:
  ``{d['cap']}'' The object ``{d['word']}'' in this caption is a hallucination:
  it is not present in the image.}}
  \label{{fig:photo-{image_id}}}
\end{{figure}}

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=\linewidth]{{bars_{image_id}.png}}
  \caption{{Per-branch logits at the step that emits ``{d['word']}''. The $x$-axis
  is the expert's top-12 tokens in a fixed order, identical across the three panels:
  the expert logit $E$, the amateur logit $A$, and the contrastive score $2E-A$.
  The dashed line is the APC plausibility cutoff; hatched bars fall below it. The
  emitted (hallucinated) token is red. The expert ranks ``{d['word']}'' first and
  the amateur ranks it {d['a_rank_str']}; since $d=E-A={d['d']:+.2f}>0$, the
  contrastive score raises it rather than suppressing it, so it stays rank~1 and is
  emitted.}}
  \label{{fig:bars-{image_id}}}
\end{{figure}}

\begin{{table}}[htbp]
  \centering
  {table_tex(image_id, step)}
  \caption{{Expert top-10 tokens at the same step, matching
  Figure~\ref{{fig:bars-{image_id}}}, with the amateur logit $A$, the contrastive
  score $2E-A$, and whether each token survives the APC cutoff (${d['cutoff']:.2f}$).
  The hallucinated token (bold) has both the highest expert logit and the highest
  contrastive score, so contrastive decoding leaves it at rank~1.}}
  \label{{tab:top10-{image_id}}}
\end{{table}}
"""


doc = (
    "% ============================================================\n"
    "%  Appendix: two worked examples of contrastive decoding\n"
    "%  failing to remove object hallucinations.\n"
    "%  Compiles on its own. To drop into a paper, copy the two\n"
    "%  \\subsection* blocks between \\begin{document} and \\end{document},\n"
    "%  and make sure the preamble has graphicx, booktabs, amsmath.\n"
    "%  Keep the 4 image files in the same folder as this .tex.\n"
    "% ============================================================\n"
    r"\documentclass[11pt]{article}" "\n"
    r"\usepackage[margin=1in]{geometry}" "\n"
    r"\usepackage{graphicx}" "\n"
    r"\usepackage{booktabs}" "\n"
    r"\usepackage{amsmath}" "\n"
    r"\graphicspath{{./}}" "\n\n"
    r"\begin{document}" "\n\n"
    r"\section*{Worked examples: contrastive decoding does not remove object hallucinations}" "\n\n"
    "We show two decoding steps where VCD emits a hallucinated object on "
    "LLaVA-1.5-7B. In each, the contrastive adjustment $d=E-A$ is positive, so the "
    "contrastive score $2E-A$ amplifies the hallucinated token rather than "
    "suppressing it, and the adaptive plausibility constraint (APC) does not remove it.\n"
    + block(1, *EXAMPLES[0])
    + block(2, *EXAMPLES[1])
    + "\n" r"\end{document}" "\n"
)

for image_id, step in EXAMPLES:
    make_photo(image_id)
    make_bars(image_id, step)
(OUT / "appendix.tex").write_text(doc)
print("wrote:", *sorted(p.name for p in OUT.glob("*") if p.suffix in (".jpg", ".pdf", ".tex")))
