"""
aggregate_seeds.py -- collect the per-seed CHAIR scores and report mean +/- std
across seeds for each (model, method). Reads outputs/caps/*_chair.json produced
by chair_eval.py and writes outputs/aggregate.json plus a printed table.

Caption files are named  {model}_{method}[_{stats}]_seed{S}.jsonl , so the
grouping key is everything before "_seed" (e.g. llava_vcd, llava_proxy_vcd,
qwen_sid, llava_greedy).
"""
import json, glob, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAPS = ROOT / "outputs" / "caps"


def main():
    files = sorted(glob.glob(str(CAPS / "*_chair.json")))
    groups = {}
    for f in files:
        name = Path(f).name[: -len("_chair.json")]
        m = re.match(r"(.+)_seed(\d+)$", name)
        if not m:
            continue
        key, seed = m.group(1), int(m.group(2))
        s = json.load(open(f))["summary"]
        groups.setdefault(key, []).append((seed, s["CHAIR_s_pct"], s["CHAIR_i_pct"]))

    def ms(xs):
        mean = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return mean, sd

    agg = {}
    print("\n" + "=" * 78)
    print(f"{'setting':28} {'seeds':>5}  {'CHAIR-S mean+/-std':>22}  {'CHAIR-I mean+/-std':>22}")
    print("=" * 78)
    for key in sorted(groups):
        rows = sorted(groups[key])
        seeds = [r[0] for r in rows]
        cs = [r[1] for r in rows]; ci = [r[2] for r in rows]
        cs_m, cs_s = ms(cs); ci_m, ci_s = ms(ci)
        agg[key] = {"seeds": seeds, "n": len(seeds),
                    "CHAIR_s": {"mean": cs_m, "std": cs_s, "values": cs},
                    "CHAIR_i": {"mean": ci_m, "std": ci_s, "values": ci}}
        print(f"{key:28} {len(seeds):>5}  {cs_m:>7.1f} +/- {cs_s:<9.2f}  {ci_m:>7.1f} +/- {ci_s:<9.2f}")
    print("=" * 78)

    out = ROOT / "outputs" / "aggregate.json"
    json.dump(agg, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    print("Note: greedy is deterministic (std should be ~0); vcd/sid/proxy vary with seed.")


if __name__ == "__main__":
    main()
