"""
generate_report.py -- builds a PDF EDA document from run_eda.py outputs.

For each of the 10 CHAIR images, shows:
  - the image, the question, and the three generated captions (greedy/vcd/sid),
    with every CHAIR object mention (real and hallucinated) plus every
    number-word and colour-word mention listed
  - for each method, one small 4-column table per mention (token | expert
    before | amateur | after subtraction | after APC), 4 tables per page,
    covering EVERY object/number/colour mention found -- not a sample.
    GREEDY tables are 2-column (token | expert) since it has no amateur/CD
    stage.
  - a closing summary page with aggregate statistics across all images.

USAGE: python generate_report.py --out-dir outputs --report outputs/EDA_report.pdf \
           --model-path ./models/llava-v1.5-7b --coco-path ../data/coco/annotations
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from object_mentions import load_tokenizer, load_chair, all_mentions

TOP_N_ROWS = 10
TABLES_PER_PAGE = 4


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _mention_title(mention):
    if mention["category"] == "object":
        if mention["hallucinated"]:
            return f"HALLUCINATED object: '{mention['node_word']}'", "#C00000"
        return f"real object: '{mention['node_word']}'", "#1F7A1F"
    label = "NUMBER" if mention["category"] == "number" else "COLOUR"
    return f"{label} word: '{mention['word']}'", "#1F4E79"


def _table_rows(step, chosen_id, is_greedy):
    recs = sorted(step["top_k"], key=lambda r: -r["expert_logit"])[:TOP_N_ROWS]
    if not any(r["token_id"] == chosen_id for r in recs):
        chosen_rec = next((r for r in step["top_k"] if r["token_id"] == chosen_id), None)
        if chosen_rec is not None:
            recs = recs[: TOP_N_ROWS - 1] + [chosen_rec]

    if is_greedy:
        col_labels = ["token", "expert (before)"]
        rows = [[r["token_str"], f"{r['expert_logit']:.2f}"] for r in recs]
    else:
        col_labels = ["token", "expert\n(before)", "amateur", "after\nsubtraction", "after\nAPC"]
        rows = []
        for r in recs:
            post = f"{r['cd_logit_post_apc']:.2f}" if r.get("cd_logit_post_apc") is not None else "MASKED"
            rows.append([
                r["token_str"],
                f"{r['expert_logit']:.2f}",
                f"{r['amateur_logit']:.2f}",
                f"{r['cd_logit_pre_apc']:.2f}",
                post,
            ])
    chosen_row_idx = next((i for i, r in enumerate(recs) if r["token_id"] == chosen_id), None)
    return col_labels, rows, chosen_row_idx


def _draw_mention_table(ax, image_id, method, step, mention, is_greedy):
    chosen_id = step["chosen_token_id"]
    col_labels, rows, chosen_row_idx = _table_rows(step, chosen_id, is_greedy)

    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    n_cols = len(col_labels)
    for j in range(n_cols):
        table[0, j].set_facecolor("#D9E1F2")
        table[0, j].set_text_props(weight="bold")
    if chosen_row_idx is not None:
        for j in range(n_cols):
            table[chosen_row_idx + 1, j].set_facecolor("#F4B6B6")

    label, color = _mention_title(mention)
    ax.set_title(
        f"img {image_id} | {method.upper()} | step {step['step_idx']}  chosen='{step['chosen_token_str']}'\n[{label}]",
        fontsize=8.5, color=color,
    )


def _tables_page(image_id, method, entries, is_greedy):
    """entries: list of up to TABLES_PER_PAGE (step, mention) tuples."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.flatten()
    for ax, (step, mention) in zip(axes, entries):
        _draw_mention_table(ax, image_id, method, step, mention, is_greedy)
    for ax in axes[len(entries):]:
        ax.axis("off")
    fig.text(0.5, 0.98, f"Image {image_id} -- {method.upper()}", ha="center", fontsize=11, weight="bold")
    fig.text(0.5, 0.005, "pink row = token actually chosen at this step", ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def _intro_page():
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.96, "LLaVA-1.5-7B Contrastive Decoding: Per-Token Logit EDA",
              ha="center", fontsize=16, weight="bold")
    body = (
        "This document traces, token by token, exactly how VCD and SID transform LLaVA-1.5-7B's\n"
        "raw next-token logits during greedy decoding on 10 CHAIR-benchmark images.\n\n"
        "Definitions (per decoding step, over the full 32000-token vocabulary V):\n\n"
        "  expert (before)  = raw logit from the real image + full attention (the normal forward pass)\n"
        "  amateur          = raw logit from a deliberately degraded forward pass\n"
        "                       VCD:  same model, image replaced by diffusion-noised image (t=500)\n"
        "                       SID:  same model, real image, but attention beyond layer 2 is\n"
        "                             restricted to 72 of the 576 image tokens (visual starvation)\n"
        "  after subtraction = (1 + alpha) * expert - alpha * amateur,  alpha=1.0\n"
        "                       i.e. amplify whatever the expert says MORE than the amateur says,\n"
        "                       on the theory that the amateur's opinion reflects language-prior\n"
        "                       bias rather than genuine visual evidence.\n"
        "  after APC         = the value above, or 'MASKED' if expert_logit fell below\n"
        "                       log(beta) + max(expert_logit), beta=0.2 -- the Adaptive Plausibility\n"
        "                       Constraint that stops the amplification step from resurrecting tokens\n"
        "                       the expert itself considered implausible. argmax of this column\n"
        "                       (ignoring MASKED rows) is the token actually emitted at that step.\n\n"
        "GREEDY has no amateur model or CD stage, so its tables only show the token and its expert logit.\n\n"
        "Step selection: this report does NOT sample generic decoding steps (function words like\n"
        "'the'/'in'/'at' are never tabulated). Instead it runs this repo's CHAIR implementation\n"
        "(eval/chair.py) against real COCO ground-truth annotations to find every COCO-object word\n"
        "mentioned in each caption, classify each as REAL (present in the image's ground truth) or\n"
        "HALLUCINATED (not), and locate the exact decoding step(s) whose generated characters produced\n"
        "it (via incremental per-token decode offset tracking). It additionally flags every mention of\n"
        "a number-word (one, two, several, ...) or a colour-word (red, blue, ...) -- these have no\n"
        "ground truth to check against, so they are shown but not classified real/hallucinated.\n\n"
        "EVERY mention found gets its own table (nothing sampled or capped): each table shows the top-10\n"
        "vocabulary tokens by expert logit at that exact step, with the token actually chosen highlighted\n"
        "in pink. Four tables per page. Table titles are red for hallucinated objects, green for real\n"
        "objects, blue for number/colour words. Each image's overview page lists every mention found.\n"
    )
    fig.text(0.06, 0.90, body, fontsize=9, va="top", family="monospace")
    return fig


def _image_overview_page(image_id, img_path, question, captions, mentions_by_method, gt_objects):
    # Page height is computed from actual caption length so long captions
    # (VCD/SID can run past 100+ tokens) are never pushed off the bottom of
    # a fixed-size page -- all positioning below is in inches-from-top,
    # converted to figure-fraction only at draw time.
    line_h, label_h, section_gap = 0.20, 0.24, 0.14
    mention_line_h = 0.20
    title_h, img_top, img_h_in, bottom_margin = 0.35, 0.55, 3.0, 0.3
    question_y = img_top + img_h_in + 0.4    # question line starts just below the image
    top_margin = question_y + 0.65           # captions start below the question + GT line

    wrapped_per_method = {m: _wrap(t, 95) for m, t in captions.items()}
    mention_lines_per_method = {}
    for m, mentions in mentions_by_method.items():
        real_words = sorted({mm["node_word"] for mm in mentions if mm["category"] == "object" and not mm["hallucinated"]})
        hall_words = sorted({mm["node_word"] for mm in mentions if mm["category"] == "object" and mm["hallucinated"]})
        attr_words = sorted({mm["word"] for mm in mentions if mm["category"] in ("number", "color")})
        mention_lines_per_method[m] = (
            _wrap(f"real objects mentioned: {', '.join(real_words) or '(none)'}", 100),
            _wrap(f"HALLUCINATED objects mentioned: {', '.join(hall_words) or '(none)'}", 100),
            _wrap(f"number/colour words mentioned: {', '.join(attr_words) or '(none)'}", 100),
        )
    captions_height = sum(
        label_h + len(wrapped_per_method[m]) * line_h
        + sum(len(block) for block in mention_lines_per_method.get(m, ([], [], []))) * mention_line_h
        + section_gap
        for m in wrapped_per_method
    )
    fig_height = max(8.5, top_margin + captions_height + bottom_margin)

    fig = plt.figure(figsize=(11, fig_height))

    def y_frac(y_inches_from_top):
        return 1 - y_inches_from_top / fig_height

    ax_img = fig.add_axes([0.05, y_frac(img_top + img_h_in), 0.4, img_h_in / fig_height])
    try:
        img = Image.open(img_path).convert("RGB")
        ax_img.imshow(img)
    except Exception as e:
        ax_img.text(0.5, 0.5, f"(image not found: {e})", ha="center")
    ax_img.axis("off")
    ax_img.set_title(f"COCO image {image_id}", fontsize=10)

    fig.text(0.5, y_frac(title_h), f"Image {image_id}", ha="center", fontsize=14, weight="bold")
    fig.text(0.05, y_frac(question_y), f"Question: {question}", fontsize=10, weight="bold")
    fig.text(0.05, y_frac(question_y + 0.3), f"Ground-truth COCO objects: {', '.join(sorted(gt_objects)) or '(none)'}",
             fontsize=9, style="italic")

    y = top_margin
    for method, lines in wrapped_per_method.items():
        fig.text(0.05, y_frac(y), f"{method.upper()}:", fontsize=10, weight="bold", color="#1F4E79")
        y += label_h
        for line in lines:
            fig.text(0.07, y_frac(y), line, fontsize=9)
            y += line_h

        real_block, hall_block, attr_block = mention_lines_per_method.get(method, ([], [], []))
        for line in real_block:
            fig.text(0.07, y_frac(y), line, fontsize=8.5, color="#1F7A1F")
            y += mention_line_h
        for line in hall_block:
            fig.text(0.07, y_frac(y), line, fontsize=8.5, weight="bold", color="#C00000")
            y += mention_line_h
        for line in attr_block:
            fig.text(0.07, y_frac(y), line, fontsize=8.5, color="#1F4E79")
            y += mention_line_h
        y += section_gap
    return fig


def _wrap(text, width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def _summary_page(csv_path):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.95, "Aggregate statistics across all 10 images", ha="center", fontsize=14, weight="bold")

    rows = list(csv.DictReader(open(csv_path)))
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    lines = []
    for method, rs in by_method.items():
        def avg(key):
            vals = [float(r[key]) for r in rs if r[key] not in (None, "", "None")]
            return sum(vals) / len(vals) if vals else float("nan")
        lines.append(f"{method.upper():8s}  n_images={len(rs):2d}  "
                      f"avg_n_steps={avg('n_steps'):6.1f}  "
                      f"avg_expert_entropy={avg('avg_expert_entropy'):6.3f}  "
                      f"avg_amateur_entropy={avg('avg_amateur_entropy'):6.3f}")
        lines.append(f"{'':8s}  "
                      f"avg_n_above_apc={avg('avg_n_above_apc'):7.1f}  "
                      f"avg_cd_prob_shift={avg('avg_cd_prob_shift'):6.3f}  "
                      f"avg_logit_diff_top1={avg('avg_logit_diff_top1'):6.3f}")
        lines.append("")

    fig.text(0.05, 0.85, "\n".join(lines), fontsize=9, family="monospace", va="top")

    interp = (
        "Reading these numbers:\n\n"
        "  avg_n_above_apc: how many of the 32000 vocabulary tokens survive the APC mask per step, on\n"
        "    average. If this is very large, APC is barely constraining anything and the contrastive\n"
        "    amplification in Stage 3 is essentially unconstrained; if very small, APC is doing almost\n"
        "    all the work and the contrastive term barely matters.\n\n"
        "  avg_cd_prob_shift: total probability mass that moved between the expert distribution and the\n"
        "    final post-APC CD distribution, averaged per step. Large values mean CD is aggressively\n"
        "    reweighting the vocabulary; small values mean it is a near no-op most of the time.\n\n"
        "  avg_logit_diff_top1 (expert_logit - amateur_logit for the CHOSEN token): if this is small or\n"
        "    negative, it means the amateur model assigned a similar-or-higher logit to the token that\n"
        "    was ultimately chosen as the expert did -- i.e. the 'hallucination signal' CD is supposed\n"
        "    to suppress was not actually concentrated on that token. This is the key generative-vs-\n"
        "    discriminative asymmetry: in POPE-style yes/no discrimination the amateur's bias shows up\n"
        "    as a clean shift on a single binary choice, but in open-ended generation the vocabulary is\n"
        "    32000-wide and the amateur's 'blindness' spreads its probability mass diffusely across many\n"
        "    plausible continuations rather than concentrating it on the specific hallucinated object\n"
        "    token -- so subtracting the amateur logit amplifies whatever the expert already favoured\n"
        "    at least as much as it suppresses genuine hallucination.\n"
    )
    fig.text(0.05, 0.55, interp, fontsize=9, va="top", family="monospace")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--report", default=None)
    ap.add_argument("--image-ids-file", default="image_ids.json")
    ap.add_argument("--model-path", default="./models/llava-v1.5-7b",
                     help="Only used to load the tokenizer for object-mention alignment (no GPU/model weights needed)")
    ap.add_argument("--coco-path", default="../data/coco/annotations",
                     help="Directory holding instances_val2017.json and captions_val2017.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    report_path = Path(args.report) if args.report else out_dir / "EDA_report.pdf"
    image_ids = json.load(open(args.image_ids_file))

    print("Loading tokenizer and CHAIR ground truth ...")
    tokenizer = load_tokenizer(args.model_path)
    chair = load_chair(image_ids, args.coco_path)

    with PdfPages(report_path) as pdf:
        pdf.savefig(_intro_page())
        plt.close("all")

        for image_id in image_ids:
            print(f"Rendering image {image_id} ...")
            per_method = {}
            for method in ["greedy", "vcd", "sid"]:
                fp = out_dir / method / f"img_{image_id}_eda.json"
                if fp.exists():
                    per_method[method] = _load_json(fp)

            mentions_by_method = {}
            for method, d in per_method.items():
                chosen_ids = [s["chosen_token_id"] for s in d["steps"]]
                mentions_by_method[method] = all_mentions(chair, tokenizer, image_id, chosen_ids)

            captions = {m: d["generated_text"] for m, d in per_method.items()}
            img_path = Path(args.data_dir) / f"{image_id:012d}.jpg"
            question = "Describe this image in detail."
            gt_objects = chair.imid_to_objects.get(image_id, set())
            pdf.savefig(_image_overview_page(image_id, img_path, question, captions, mentions_by_method, gt_objects))
            plt.close("all")

            for method in ["greedy", "vcd", "sid"]:
                if method not in per_method:
                    continue
                d = per_method[method]
                steps = d["steps"]
                mentions = mentions_by_method[method]
                entries = [(steps[m["step_indices"][0]], m) for m in mentions]
                is_greedy = method == "greedy"
                for i in range(0, len(entries), TABLES_PER_PAGE):
                    batch = entries[i:i + TABLES_PER_PAGE]
                    fig = _tables_page(image_id, method, batch, is_greedy)
                    pdf.savefig(fig)
                    plt.close(fig)

        csv_path = out_dir / "summary.csv"
        if csv_path.exists():
            pdf.savefig(_summary_page(csv_path))
            plt.close("all")

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
