# import json
# import numpy as np

# datasets = ["coco", "gqa", "aokvqa"]
# splits = ["random", "popular", "adversarial"]

# for dataset in datasets:
#     for split in splits:
#         print(f"{dataset},{split}")
#         records = [json.loads(l) for l in open(f"/teamspace/studios/this_studio/cd-rethinking/attn/eval/{dataset}/vcd/{dataset}-{split}.jsonl")]
#         records = [r for r in records if r["fraction_expert"] is not None 
#                    and r["fraction_cd"] is not None]

#         total = len(records)

#         # how often does CD improve visual grounding?
#         cd_better  = [r for r in records if r["fraction_cd"] > r["fraction_expert"]]
#         exp_better = [r for r in records if r["fraction_expert"] > r["fraction_cd"]]
#         equal      = [r for r in records if r["fraction_cd"] == r["fraction_expert"]]

#         print(f"Total samples: {total}")
#         print(f"CD has higher visual grounding:     {len(cd_better)}/{total} = {100*len(cd_better)/total:.1f}%")
#         print(f"Expert has higher visual grounding: {len(exp_better)}/{total} = {100*len(exp_better)/total:.1f}%")

#         # same breakdown but only on flipped samples — the ones CD claims to be correcting
#         flipped = [r for r in records if r["cd_flipped_yes_to_no"] == True]
#         print(f"\n--- On CD-flipped samples only (N={len(flipped)}) ---")
#         cd_better_f  = [r for r in flipped if r["fraction_cd"] > r["fraction_expert"]]
#         exp_better_f = [r for r in flipped if r["fraction_expert"] > r["fraction_cd"]]
#         print(f"CD has higher visual grounding:     {len(cd_better_f)}/{len(flipped)} = {100*len(cd_better_f)/len(flipped):.1f}%")
#         print(f"Expert has higher visual grounding: {len(exp_better_f)}/{len(flipped)} = {100*len(exp_better_f)/len(flipped):.1f}%")
#         print ("---------------------------------------------")


import json
import numpy as np

for dataset in ["aokvqa-random", "aokvqa-popular", "aokvqa-adversarial"]:
    records = [json.loads(l) for l in open(f"/teamspace/studios/this_studio/cd-rethinking/attn/eval/aokvqa/vcd/{dataset}.jsonl")]
    records = [r for r in records if r["fraction_expert"] is not None 
               and r["fraction_cd"] is not None]
    
    diffs = [r["fraction_cd"] - r["fraction_expert"] for r in records]
    
    print(f"\n{dataset}")
    print(f"Mean fraction_expert: {np.mean([r['fraction_expert'] for r in records]):.6f}")
    print(f"Mean fraction_cd:     {np.mean([r['fraction_cd'] for r in records]):.6f}")
    print(f"Mean diff (cd-exp):   {np.mean(diffs):.6f}")
    print(f"Median diff:          {np.median(diffs):.6f}")
    print(f"Std diff:             {np.std(diffs):.6f}")
    
    # flipped only
    flipped = [r for r in records if r["cd_flipped_yes_to_no"] == True]
    if flipped:
        diffs_f = [r["fraction_cd"] - r["fraction_expert"] for r in flipped]
        print(f"Mean diff flipped:    {np.mean(diffs_f):.6f}")