"""
verify_outputs.py -- structural sanity checks on capture files BEFORE
trusting any analysis result. Run this after downloading capture files from
the Modal volume and before running agreement_analysis.py /
contrastive_analysis.py / plot_d_histograms.py.

Checks, per file:
  1. Line count matches --n-images (or a clear warning if a run is partial /
     still in progress -- this is expected mid-run, not necessarily a bug).
  2. image_id set has no duplicates.
  3. image_id set is a subset of image_ids_500.json (the canonical set) --
     catches a run accidentally pointed at the wrong dataset.
  4. Every record has the required schema fields for its method
     (two-branch methods need amateur/cd fields; baseline does not).
  5. No empty/whitespace-only captions.
  6. Math consistency: for two-branch methods, wherever a token id appears
     in BOTH expert_top_ids and amateur_top_ids AND cd_top_ids, checks
     cd_top_pre_apc_logit == (1+cd_alpha)*E - cd_alpha*A within float
     tolerance (cd_alpha=1.0 here, so this is 2*E - A) -- this is the same
     kind of check used earlier in this project to catch capture bugs that
     exit code 0 would not reveal.

Usage:
  python verify_outputs.py                      # checks everything found under outputs/captures/
  python verify_outputs.py --n-images 500        # also flags partial runs
"""
import argparse, glob, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CAPTURES = REPO / "outputs" / "captures"

TWO_BRANCH_FIELDS = {"expert_top_ids", "expert_top_logits", "amateur_top_ids", "amateur_top_logits",
                      "cd_top_ids", "cd_top_pre_apc_logits", "cd_top_survives_apc", "apc_cutoff", "chosen_id", "step"}
BASELINE_FIELDS = {"expert_top_ids", "expert_top_logits", "chosen_id", "step"}
CD_ALPHA = 1.0


def check_file(path, canonical_ids, n_images_expected):
    problems = []
    recs = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                problems.append(f"line {lineno}: invalid JSON ({e})")

    if not recs:
        return ["EMPTY FILE"], 0

    n = len(recs)
    if n_images_expected and n != n_images_expected:
        problems.append(f"line count {n} != expected {n_images_expected} "
                         f"(fine if the run is still in progress / resumable)")

    ids = [r["image_id"] for r in recs]
    if len(set(ids)) != len(ids):
        dupes = [i for i in set(ids) if ids.count(i) > 1]
        problems.append(f"DUPLICATE image_ids: {dupes[:10]}{'...' if len(dupes) > 10 else ''}")

    bad_ids = [i for i in ids if i not in canonical_ids]
    if bad_ids:
        problems.append(f"image_ids NOT in canonical image_ids_500.json: {bad_ids[:10]}"
                         f"{'...' if len(bad_ids) > 10 else ''}")

    two_branch = "amateur_top_ids" in (recs[0].get("steps") or [{}])[0] if recs[0].get("steps") else False
    required = TWO_BRANCH_FIELDS if two_branch else BASELINE_FIELDS

    empty_captions = 0
    schema_errors = 0
    math_checked = 0
    math_errors = 0

    for r in recs:
        if not r.get("caption", "").strip():
            empty_captions += 1
        for s in r.get("steps", []):
            missing = required - set(s.keys())
            if missing:
                schema_errors += 1
                continue
            if two_branch:
                emap = dict(zip(s["expert_top_ids"], s["expert_top_logits"]))
                amap = dict(zip(s["amateur_top_ids"], s["amateur_top_logits"]))
                for tid, cd_logit in zip(s["cd_top_ids"], s["cd_top_pre_apc_logits"]):
                    if tid in emap and tid in amap:
                        expected = (1 + CD_ALPHA) * emap[tid] - CD_ALPHA * amap[tid]
                        math_checked += 1
                        if abs(expected - cd_logit) > 0.01:
                            math_errors += 1

    if empty_captions:
        problems.append(f"{empty_captions}/{n} records have an empty caption")
    if schema_errors:
        problems.append(f"{schema_errors} step records missing required fields")
    if math_errors:
        problems.append(f"{math_errors}/{math_checked} cd-formula consistency checks FAILED "
                         f"(cd_top_pre_apc_logit != (1+a)E - aA)")

    return problems, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=None, help="expected image count per file (omit to skip this check)")
    ap.add_argument("--glob", default="*_*.jsonl", help="filename glob under outputs/captures/, e.g. 'qwen_*.jsonl'")
    args = ap.parse_args()

    canonical_ids = set(json.load(open(REPO / "image_ids_500.json")))
    files = sorted(glob.glob(str(CAPTURES / args.glob)))
    if not files:
        print(f"No files found matching {CAPTURES / args.glob}")
        sys.exit(1)

    any_problems = False
    print(f"Canonical dataset: {len(canonical_ids)} images\n")
    for f in files:
        problems, n = check_file(Path(f), canonical_ids, args.n_images)
        status = "OK" if not problems else "PROBLEMS"
        print(f"[{status}] {Path(f).name}  (n={n})")
        for p in problems:
            print(f"    - {p}")
            any_problems = True

    print()
    if any_problems:
        print("Some files have problems -- see above. Partial-run line-count warnings are")
        print("expected mid-run; anything else should be investigated before trusting the analysis.")
        sys.exit(1)
    else:
        print("All files passed structural verification.")


if __name__ == "__main__":
    main()
