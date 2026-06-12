# Table 5 — LLaVA-v1.5-7B (sample strategy), per dataset

Accuracy and Yes(%) per dataset; deltas vs the `sample` baseline.


## dataset = coco

| Category | Method | Accuracy | ΔAcc | Yes (%) | ΔYes |
|---|---|---|---|---|---|
| Random | sample | 83.6 |  | 39.3 |  |
| Random | VCD | 88.2 | +4.6 | 44.0 | +4.7 |
| Random | ICD | 86.2 | +2.5 | 38.8 | -0.5 |
| Random | SID | 87.7 | +4.0 | 42.1 | +2.8 |
| Random | sample† | 86.6 | +2.9 | 40.5 | +1.2 |
| Popular | sample | 81.7 |  | 39.9 |  |
| Popular | VCD | 85.6 | +3.9 | 46.5 | +6.6 |
| Popular | ICD | 84.1 | +2.3 | 39.9 | +0.0 |
| Popular | SID | 85.7 | +3.9 | 44.0 | +4.1 |
| Popular | sample† | 84.5 | +2.7 | 41.7 | +1.7 |
| Adversarial | sample | 79.8 |  | 44.4 |  |
| Adversarial | VCD | 82.0 | +2.2 | 50.3 | +5.9 |
| Adversarial | ICD | 81.9 | +2.0 | 42.5 | -1.9 |
| Adversarial | SID | 82.0 | +2.1 | 47.8 | +3.3 |
| Adversarial | sample† | 81.9 | +2.1 | 43.7 | -0.7 |

## dataset = aokvqa

| Category | Method | Accuracy | ΔAcc | Yes (%) | ΔYes |
|---|---|---|---|---|---|
| Random | sample | 84.1 |  | 46.1 |  |
| Random | VCD | 87.4 | +3.3 | 50.3 | +4.2 |
| Random | ICD | 86.8 | +2.7 | 45.3 | -0.9 |
| Random | SID | 87.4 | +3.3 | 48.6 | +2.5 |
| Random | sample† | 86.7 | +2.7 | 45.7 | -0.5 |
| Popular | sample | 80.6 |  | 49.4 |  |
| Popular | VCD | 82.6 | +2.0 | 54.5 | +5.1 |
| Popular | ICD | 83.9 | +3.2 | 47.9 | -1.6 |
| Popular | SID | 82.9 | +2.3 | 52.3 | +2.8 |
| Popular | sample† | 83.5 | +2.9 | 49.2 | -0.2 |
| Adversarial | sample | 76.5 |  | 54.0 |  |
| Adversarial | VCD | 76.9 | +0.4 | 61.2 | +7.2 |
| Adversarial | ICD | 78.0 | +1.6 | 54.7 | +0.7 |
| Adversarial | SID | 76.8 | +0.3 | 58.6 | +4.6 |
| Adversarial | sample† | 76.8 | +0.3 | 55.6 | +1.6 |

## dataset = gqa

| Category | Method | Accuracy | ΔAcc | Yes (%) | ΔYes |
|---|---|---|---|---|---|
| Random | sample | 85.5 |  | 46.2 |  |
| Random | VCD | 88.0 | +2.5 | 51.1 | +4.9 |
| Random | ICD | 87.5 | +2.0 | 45.2 | -1.0 |
| Random | SID | 87.5 | +1.9 | 48.1 | +1.9 |
| Random | sample† | 87.1 | +1.6 | 45.9 | -0.3 |
| Popular | sample | 79.2 |  | 52.6 |  |
| Popular | VCD | 80.8 | +1.6 | 57.6 | +5.0 |
| Popular | ICD | 80.5 | +1.3 | 50.8 | -1.9 |
| Popular | SID | 81.2 | +2.1 | 54.6 | +1.9 |
| Popular | sample† | 81.3 | +2.1 | 51.8 | -0.8 |
| Adversarial | sample | 75.2 |  | 55.7 |  |
| Adversarial | VCD | 77.0 | +1.8 | 61.5 | +5.8 |
| Adversarial | ICD | 78.3 | +3.0 | 53.5 | -2.2 |
| Adversarial | SID | 77.8 | +2.5 | 59.0 | +3.3 |
| Adversarial | sample† | 78.2 | +3.0 | 55.1 | -0.6 |
