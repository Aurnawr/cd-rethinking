# Calibration experiments: LLaVA-1.5-7B vs Qwen2.5-VL-7B

All numbers at the readout layer, beta = 1, POPE-COCO pooled over the random, popular and adversarial splits.

## Experiment 3: oracle calibration of the selectivity metric

| model | method | n_H | n_T | sel AUC | sel 95% CI | sel AUC resid. | above chance | oracle ceiling | mean abs D | s* | mean abs D / s* | mean D on H | mean D on T | frac D>0 on H |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | vcd | 257 | 3453 | 0.523 | [0.49, 0.55] | 0.420 | False | 0.885 | 3.640 | 0.150 | 24.27 | 3.316 | 3.600 | 0.965 |
| LLaVA-1.5-7B | icd | 261 | 3453 | 0.639 | [0.60, 0.67] | 0.467 | True | 0.889 | 0.181 | 0.020 | 9.074 | -0.168 | -0.056 | 0.257 |
| LLaVA-1.5-7B | sid | 261 | 3453 | 0.691 | [0.66, 0.72] | 0.492 | True | 0.889 | 3.079 | 0.170 | 18.11 | 1.607 | 3.111 | 0.797 |
| Qwen2.5-VL-7B | vcd | 141 | 3552 | 0.805 | [0.77, 0.84] | 0.558 | True | 0.901 | 4.935 | 0.450 | 10.97 | 2.137 | 4.996 | 0.844 |
| Qwen2.5-VL-7B | icd | 141 | 3552 | 0.494 | [0.45, 0.54] | 0.303 | False | 0.901 | 0.564 | 0.060 | 9.402 | 0.033 | 0.052 | 0.496 |
| Qwen2.5-VL-7B | sid | 141 | 3552 | 0.726 | [0.68, 0.77] | 0.552 | True | 0.901 | 5.726 | 0.440 | 13.01 | 3.050 | 5.775 | 0.872 |

### Per-pair verdict

- **LLaVA-1.5-7B / VCD**: non-selective: the shift's ranking of H over T is indistinguishable from chance (AUC 0.52, CI lower bound 0.49).
- **LLaVA-1.5-7B / ICD**: selectivity 0.64 above chance and correctly signed on H (-0.17 log-odds), but the H-vs-T differential is only +0.11 log-odds against a margin of median magnitude 3.2 (5.3% of it) - a nudge, not a correction.
- **LLaVA-1.5-7B / SID**: selectivity 0.69 is above chance but is the boost-correct-positives artifact: the shift is positive on both classes (H +1.61, T +3.11 log-odds), so on a hallucination it still reinforces the false Yes, just less than it reinforces a correct one.
- **Qwen2.5-VL-7B / VCD**: selectivity 0.81 is above chance but is the boost-correct-positives artifact: the shift is positive on both classes (H +2.14, T +5.00 log-odds), so on a hallucination it still reinforces the false Yes, just less than it reinforces a correct one.
- **Qwen2.5-VL-7B / ICD**: non-selective: the shift's ranking of H over T is indistinguishable from chance (AUC 0.49, CI lower bound 0.45).
- **Qwen2.5-VL-7B / SID**: selectivity 0.73 is above chance but is the boost-correct-positives artifact: the shift is positive on both classes (H +3.05, T +5.78 log-odds), so on a hallucination it still reinforces the false Yes, just less than it reinforces a correct one.

## Scalar-bias experiment (POPE-COCO pooled, 9,000 questions)

Accuracies in percent. `b_equiv` is CD's own mean shift flattened to a constant; `b*` is the accuracy-maximising constant.

| model | method | expert acc | CD acc | CD - expert | b_equiv | acc(b_equiv) | b* | acc(b*) | b* - expert | expert yes-rate | CD yes-rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | vcd | 85.49 | 85.04 | -0.444 | 1.234 | 86.48 | 0.840 | 86.86 | 1.367 | 41.24 | 48.16 |
| LLaVA-1.5-7B | icd | 85.47 | 85.04 | -0.422 | -0.254 | 84.93 | 0.860 | 86.89 | 1.422 | 41.27 | 40.02 |
| LLaVA-1.5-7B | sid | 85.47 | 85.89 | 0.422 | 0.627 | 86.43 | 0.860 | 86.89 | 1.422 | 41.27 | 44.31 |
| Qwen2.5-VL-7B | vcd | 87.90 | 88.30 | 0.400 | -0.499 | 87.34 | 2.380 | 90.02 | 2.122 | 41.03 | 41.90 |
| Qwen2.5-VL-7B | icd | 87.90 | 87.16 | -0.744 | -0.239 | 87.68 | 2.380 | 90.02 | 2.122 | 41.03 | 40.24 |
| Qwen2.5-VL-7B | sid | 87.90 | 89.10 | 1.200 | 1.285 | 89.22 | 2.380 | 90.02 | 2.122 | 41.03 | 43.97 |
