"""Compute CHAIR-S / CHAIR-I for a file of generated COCO captions.

Usage:

    python eval/chair_eval.py \\
        --cap-file   outputs/chair/llava-7b-greedy/captions.jsonl \\
        --coco-path  data/coco/annotations

The caption file may be either:
  * a JSONL file with one ``{"image_id": int, "caption": str}`` per line, or
  * a JSON list of the same dicts, or
  * a THRONE-style ``responses.json`` (``{"responses": [[p_idx, image_id, text], ...]}``).

Results (CHAIR-S, CHAIR-I, average objects / caption length) are printed and,
unless ``--no-save``, written next to the caption file as ``<cap>_chair.json``.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chair import CHAIR


def load_captions(cap_file):
    """Load captions from jsonl / json-list / THRONE-responses formats."""
    with open(cap_file, "r") as f:
        content = f.read().strip()

    caps = []
    # Try whole-file JSON first (list or THRONE responses dict).
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "responses" in data:
            for entry in data["responses"]:
                # THRONE: [prompt_idx, image_id, text]
                caps.append({"image_id": int(entry[1]), "caption": entry[2]})
            return caps
        if isinstance(data, list):
            for d in data:
                caps.append({"image_id": int(d["image_id"]), "caption": d["caption"]})
            return caps
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL.
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        caps.append({"image_id": int(d["image_id"]), "caption": d["caption"]})
    return caps


def main():
    parser = argparse.ArgumentParser(description="Compute CHAIR metrics.")
    parser.add_argument("--cap-file", required=True,
                        help="Generated captions (jsonl / json list / THRONE responses).")
    parser.add_argument("--coco-path", default="data/coco/annotations",
                        help="Directory with instances_val2017.json and captions_val2017.json.")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the detailed <cap>_chair.json output file.")
    args = parser.parse_args()

    caps = load_captions(args.cap_file)
    if not caps:
        raise SystemExit(f"No captions loaded from {args.cap_file}")

    imids = {c["image_id"] for c in caps}
    evaluator = CHAIR(imids, args.coco_path)
    evaluator.get_annotations()
    results = evaluator.compute_chair(caps)

    print("=" * 50)
    print(f"Caption file : {args.cap_file}")
    print(f"# captions   : {results['num_captions']}")
    print("-" * 50)
    print(f"CHAIR-S (per sentence) : {results['CHAIR_s'] * 100:.1f}")
    print(f"CHAIR-I (per instance) : {results['CHAIR_i'] * 100:.1f}")
    print(f"avg objects / caption  : {results['avg_objects_per_caption']:.2f}")
    print(f"avg caption length     : {results['avg_caption_length']:.1f}")
    print("=" * 50)

    if not args.no_save:
        base = os.path.splitext(args.cap_file)[0]
        out_file = base + "_chair.json"
        summary = {k: v for k, v in results.items() if k != "sentences"}
        summary["CHAIR_s_pct"] = results["CHAIR_s"] * 100
        summary["CHAIR_i_pct"] = results["CHAIR_i"] * 100
        with open(out_file, "w") as f:
            json.dump({"summary": summary, "sentences": results["sentences"]}, f, indent=2)
        print(f"Detailed results written to {out_file}")


if __name__ == "__main__":
    main()
