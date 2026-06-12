# Table 5 — LLaVA-v1.5-7B (sample strategy)

Accuracy and Yes(%) averaged across coco/aokvqa/gqa; deltas vs the `sample` baseline.

| Category | Method | Accuracy | ΔAcc | Yes (%) | ΔYes |
|---|---|---|---|---|---|
| Random | sample | 84.4 |  | 43.9 |  |
| Random | VCD | 87.9 | +3.5 | 48.5 | +4.6 |
| Random | ICD | 86.8 | +2.4 | 43.1 | -0.8 |
| Random | SID | 87.5 | +3.1 | 46.3 | +2.4 |
| Random | sample† | 86.8 | +2.4 | 44.0 | +0.2 |
| Popular | sample | 80.5 |  | 47.3 |  |
| Popular | VCD | 83.0 | +2.5 | 52.9 | +5.5 |
| Popular | ICD | 82.8 | +2.3 | 46.2 | -1.1 |
| Popular | SID | 83.3 | +2.8 | 50.3 | +2.9 |
| Popular | sample† | 83.1 | +2.6 | 47.6 | +0.2 |
| Adversarial | sample | 77.2 |  | 51.4 |  |
| Adversarial | VCD | 78.6 | +1.4 | 57.7 | +6.3 |
| Adversarial | ICD | 79.4 | +2.2 | 50.2 | -1.1 |
| Adversarial | SID | 78.8 | +1.7 | 55.1 | +3.8 |
| Adversarial | sample† | 79.0 | +1.8 | 51.5 | +0.1 |
