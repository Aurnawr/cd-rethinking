# LLaVA-Bench reproduction — *The Mirage of Performance Gains* (2504.10020v4)

Self-contained folder to reproduce the **LLaVA-Bench-in-the-Wild** column of the
paper's **Table 4** (greedy base) and **Table 6** (direct-sampling base).

## Paper target numbers (LLaVA-1.5-7B)

| Table 4 (greedy) | LLaVA-Bench | | Table 6 (sampling) | LLaVA-Bench |
|---|---|---|---|---|
| Greedy | **65.7** | | Sample | **64.3** |
| VCD | **64.6** | | Sample† (=APC) | **65.3** |
| SID | **64.9** | | VCD | **64.2** |
| | | | SID | **64.5** |

Metric = LLaVA-Bench-in-the-Wild judge relative score: judge sees image caption
(`context.jsonl`) + question + GPT-4 reference answer + model answer, emits two
1–10 scores; final = mean(model)/mean(ref) × 100 over the 60 questions.

## Repo → paper mapping (what we already had vs. ran)

| Paper cell | decode flags | file | status |
|---|---|---|---|
| T4 Greedy | `do_sample=False` | `outputs/llava/greedy.jsonl` | ✅ staged |
| T4 VCD | greedy + VCD | `outputs/llava/vcd_greedy.jsonl` | ⏳ run on GPU |
| T4 SID | greedy + SID | `outputs/llava/sid_greedy.jsonl` | ⏳ run on GPU |
| T6 Sample | `do_sample=True,t=1` | `outputs/llava/sample.jsonl` | ✅ staged |
| T6 Sample† | sampling + APC β=0.1 | `outputs/llava/sampledagger_apc_b0.100.jsonl` | ✅ staged |
| T6 VCD | sampling + VCD | `outputs/llava/vcd_sample.jsonl` | ✅ staged |
| T6 SID | sampling + SID | `outputs/llava/sid_sample.jsonl` | ⏳ run on GPU |

**Key facts established during setup**
- The old `vcd/` run used `do_sample=True, temp=1.0` → it is the **Table 6** VCD, not Table 4.
- "**Sample†**" in the paper = the adaptive plausibility constraint applied alone = the
  `apc` runs. β=0.1 is the standard value → `sampledagger_apc_b0.100.jsonl`.
- `questions.jsonl` is byte-identical to the official llava-bench-in-the-wild set, so
  `answers_gpt4.jsonl` (reference / "Assistant 1") and `context.jsonl` align by line order.
- Qwen `vcd_sample.jsonl` is **incomplete (42/60)** — must be re-run.

## Layout
```
data/llava_bench/   questions.jsonl, context.jsonl, answers_gpt4.jsonl, rule.json, images/
models/             llava-v1.5-7b -> HF cache snapshot (qwen2.5-7b to be added)
inference/          llava_bench_infer_cd.py (unified VCD/SID) + originals + cd_utils/
eval/               llava_bench_gpt_review.py, llava_bench_summarize.py
outputs/llava/ outputs/qwen/
scripts/            run_missing_llava.sh
```

## How to finish (needs GPU, e.g. L4)
1. `bash scripts/run_missing_llava.sh`  → produces vcd_greedy / sid_greedy / sid_sample.
2. Judge with Claude Opus 4.8 (in-session): build per-item judge prompts from
   context+question+answers_gpt4(ans1)+model(ans2)+rule, score each 1–10 pair.
3. `python eval/llava_bench_summarize.py --review <review.jsonl>` per method → relative score.

### Caveats for the write-up
- A Claude judge ≠ gpt-4-0314, so absolute numbers will differ from 65.7 etc.; what
  reproduces is the **pattern** (VCD/SID ≈ or slightly below base; Sample† gives the +1 bump).
- These are ~1-pt deltas on 60 questions → within judge noise; report ≥3 judge passes ± std.
