# Table 5 - llava-13b (sample strategy)

Accuracy and Yes(%) averaged across coco/aokvqa/gqa; deltas vs `sample` baseline.

| Category | Method | Accuracy | ΔAcc | Yes (%) | ΔYes |
|---|---|---|---|---|---|
| Random | sample | 84.9 |  | 40.8 |  |
| Random | VCD | 87.2 | +2.3 | 42.0 | +1.2 |
| Random | ICD | 86.1 | +1.2 | 40.4 | -0.5 |
| Random | SID | 86.0 | +1.1 | 40.2 | -0.6 |
| Random | sample† | 85.8 | +0.9 | 40.0 | -0.8 |
| Popular | sample | 81.7 |  | 42.9 |  |
| Popular | VCD | 85.0 | +3.3 | 44.5 | +1.6 |
| Popular | ICD | 83.3 | +1.6 | 42.7 | -0.2 |
| Popular | SID | 83.4 | +1.7 | 43.6 | +0.7 |
| Popular | sample† | 82.9 | +1.2 | 42.3 | -0.6 |
| Adversarial | sample | 78.6 |  | 46.5 |  |
| Adversarial | VCD | 81.3 | +2.7 | 48.0 | +1.5 |
| Adversarial | ICD | 80.3 | +1.7 | 46.5 | +0.0 |
| Adversarial | SID | 80.0 | +1.5 | 46.5 | +0.0 |
| Adversarial | sample† | 80.0 | +1.4 | 46.2 | -0.3 |
