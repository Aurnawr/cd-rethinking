# Table 5 - llava-13b (sample strategy), per dataset

Accuracy and Yes(%) per dataset; deltas vs `sample` baseline.


## dataset = coco

| Category | Method | Accuracy | ΔAcc | Yes (%) | ΔYes |
|---|---|---|---|---|---|
| Random | sample | 85.0 |  | 37.4 |  |
| Random | VCD | 86.8 | +1.8 | 39.1 | +1.7 |
| Random | ICD | 85.6 | +0.6 | 37.7 | +0.3 |
| Random | SID | 85.9 | +0.8 | 37.3 | -0.1 |
| Random | sample† | 85.5 | +0.5 | 37.3 | -0.1 |
| Popular | sample | 83.5 |  | 38.4 |  |
| Popular | VCD | 86.1 | +2.6 | 40.0 | +1.6 |
| Popular | ICD | 85.0 | +1.5 | 38.5 | +0.1 |
| Popular | SID | 84.8 | +1.3 | 38.6 | +0.1 |
| Popular | sample† | 84.4 | +0.9 | 38.1 | -0.3 |
| Adversarial | sample | 81.3 |  | 40.6 |  |
| Adversarial | VCD | 84.6 | +3.2 | 42.4 | +1.8 |
| Adversarial | ICD | 83.7 | +2.4 | 39.7 | -0.9 |
| Adversarial | SID | 83.1 | +1.8 | 39.8 | -0.8 |
| Adversarial | sample† | 83.1 | +1.8 | 39.9 | -0.7 |

## dataset = aokvqa

| Category | Method | Accuracy | ΔAcc | Yes (%) | ΔYes |
|---|---|---|---|---|---|
| Random | sample | 85.5 |  | 42.7 |  |
| Random | VCD | 87.4 | +2.0 | 44.0 | +1.3 |
| Random | ICD | 86.7 | +1.2 | 41.8 | -0.8 |
| Random | SID | 86.0 | +0.5 | 41.9 | -0.8 |
| Random | sample† | 86.3 | +0.8 | 41.6 | -1.1 |
| Popular | sample | 82.6 |  | 43.8 |  |
| Popular | VCD | 85.9 | +3.3 | 45.8 | +2.0 |
| Popular | ICD | 85.0 | +2.4 | 43.4 | -0.5 |
| Popular | SID | 84.9 | +2.3 | 45.1 | +1.3 |
| Popular | sample† | 83.9 | +1.2 | 43.7 | -0.1 |
| Adversarial | sample | 77.4 |  | 49.7 |  |
| Adversarial | VCD | 79.9 | +2.5 | 51.6 | +1.9 |
| Adversarial | ICD | 78.4 | +1.0 | 50.9 | +1.3 |
| Adversarial | SID | 78.1 | +0.7 | 50.4 | +0.7 |
| Adversarial | sample† | 78.8 | +1.4 | 51.0 | +1.4 |

## dataset = gqa

| Category | Method | Accuracy | ΔAcc | Yes (%) | ΔYes |
|---|---|---|---|---|---|
| Random | sample | 84.3 |  | 42.4 |  |
| Random | VCD | 87.4 | +3.1 | 43.0 | +0.6 |
| Random | ICD | 86.0 | +1.8 | 41.6 | -0.8 |
| Random | SID | 86.1 | +1.9 | 41.4 | -1.0 |
| Random | sample† | 85.7 | +1.5 | 41.1 | -1.3 |
| Popular | sample | 78.9 |  | 46.4 |  |
| Popular | VCD | 83.0 | +4.1 | 47.8 | +1.3 |
| Popular | ICD | 79.9 | +1.0 | 46.2 | -0.3 |
| Popular | SID | 80.5 | +1.6 | 47.0 | +0.5 |
| Popular | sample† | 80.4 | +1.5 | 45.0 | -1.4 |
| Adversarial | sample | 77.0 |  | 49.2 |  |
| Adversarial | VCD | 79.4 | +2.4 | 50.1 | +0.9 |
| Adversarial | ICD | 78.7 | +1.7 | 48.9 | -0.3 |
| Adversarial | SID | 78.9 | +1.9 | 49.4 | +0.2 |
| Adversarial | sample† | 78.0 | +1.0 | 47.8 | -1.4 |
