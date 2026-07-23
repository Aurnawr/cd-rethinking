# Top-10 Candidate Logit Tables — Every Real & Hallucinated Object Word Chosen (10-image pilot)

For every decoding step where the model actually output a real-object or hallucinated-object word, this shows the top 10 candidates **ranked by raw expert logit** (the plain, real-image-only score) at that exact step, with each candidate's expert logit, amateur logit, their difference d=E-A, the combined contrastive score before and after the APC survival filter, and which one was actually chosen.


---
# VCD


## VCD — HALLUCINATED OBJECTS


### image 776 — word "child" (node: person) — step 86

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` child` | 22.328 | 22.281 | +0.047 | 22.375 | yes | 22.375 | **YES** |
| 2 | ` bed` | 21.219 | 21.078 | +0.141 | 21.359 | yes | 21.359 |  |
| 3 | ` nur` | 20.016 | 20.078 | -0.062 | 19.953 | no | —(rejected) |  |
| 4 | ` children` | 19.219 | 19.297 | -0.078 | 19.141 | no | —(rejected) |  |
| 5 | ` play` | 18.484 | 18.516 | -0.031 | 18.453 | no | —(rejected) |  |
| 6 | ` home` | 17.891 | 17.641 | +0.250 | 18.141 | no | —(rejected) |  |
| 7 | ` room` | 17.797 | 17.703 | +0.094 | 17.891 | no | —(rejected) |  |
| 8 | ` family` | 17.016 | 16.656 | +0.359 | 17.375 | no | —(rejected) |  |
| 9 | ` living` | 16.984 | 16.453 | +0.531 | 17.516 | no | —(rejected) |  |
| 10 | ` kid` | 16.438 | 16.625 | -0.188 | 16.250 | no | —(rejected) |  |


### image 724 — word "handbag" (node: handbag) — step 67

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` hand` | 23.000 | 23.016 | -0.016 | 22.984 | yes | 22.984 | **YES** |
| 2 | ` back` | 22.484 | 22.453 | +0.031 | 22.516 | yes | 22.516 |  |
| 3 | ` cell` | 18.672 | 18.781 | -0.109 | 18.562 | no | —(rejected) |  |
| 4 | ` suit` | 18.641 | 18.594 | +0.047 | 18.688 | no | —(rejected) |  |
| 5 | ` bag` | 17.562 | 17.703 | -0.141 | 17.422 | no | —(rejected) |  |
| 6 | ` bott` | 17.016 | 17.266 | -0.250 | 16.766 | no | —(rejected) |  |
| 7 | ` book` | 16.953 | 16.875 | +0.078 | 17.031 | no | —(rejected) |  |
| 8 | ` sk` | 16.625 | 16.484 | +0.141 | 16.766 | no | —(rejected) |  |
| 9 | ` large` | 16.312 | 16.312 | +0.000 | 16.312 | no | —(rejected) |  |
| 10 | ` small` | 16.125 | 16.266 | -0.141 | 15.984 | no | —(rejected) |  |


### image 724 — word "handbag" (node: handbag) — step 68

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `bag` | 24.453 | 24.312 | +0.141 | 24.594 | yes | 24.594 | **YES** |
| 2 | ` bag` | 16.625 | 16.500 | +0.125 | 16.750 | no | —(rejected) |  |
| 3 | `b` | 14.930 | 14.781 | +0.148 | 15.078 | no | —(rejected) |  |
| 4 | `h` | 13.875 | 13.734 | +0.141 | 14.016 | no | —(rejected) |  |
| 5 | `ker` | 13.477 | 13.383 | +0.094 | 13.570 | no | —(rejected) |  |
| 6 | `-` | 13.430 | 13.352 | +0.078 | 13.508 | no | —(rejected) |  |
| 7 | `ful` | 12.992 | 12.938 | +0.055 | 13.047 | no | —(rejected) |  |
| 8 | `ic` | 12.758 | 12.680 | +0.078 | 12.836 | no | —(rejected) |  |
| 9 | `gun` | 12.758 | 12.828 | -0.070 | 12.688 | no | —(rejected) |  |
| 10 | ` lug` | 11.758 | 11.594 | +0.164 | 11.922 | no | —(rejected) |  |


### image 724 — word "fire hydrant" (node: fire hydrant) — step 104

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` fire` | 16.625 | 17.031 | -0.406 | 16.219 | yes | 16.219 | **YES** |
| 2 | ` ben` | 16.281 | 16.797 | -0.516 | 15.766 | yes | 15.766 |  |
| 3 | ` traffic` | 15.641 | 15.547 | +0.094 | 15.734 | yes | 15.734 |  |
| 4 | ` par` | 14.570 | 15.172 | -0.602 | 13.969 | no | —(rejected) |  |
| 5 | ` person` | 14.352 | 14.719 | -0.367 | 13.984 | no | —(rejected) |  |
| 6 | ` few` | 13.938 | 14.266 | -0.328 | 13.609 | no | —(rejected) |  |
| 7 | ` bus` | 13.938 | 14.055 | -0.117 | 13.820 | no | —(rejected) |  |
| 8 | ` street` | 13.836 | 13.414 | +0.422 | 14.258 | no | —(rejected) |  |
| 9 | ` park` | 13.797 | 13.922 | -0.125 | 13.672 | no | —(rejected) |  |
| 10 | ` small` | 13.617 | 13.680 | -0.062 | 13.555 | no | —(rejected) |  |


### image 724 — word "fire hydrant" (node: fire hydrant) — step 105

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` hyd` | 26.609 | 26.250 | +0.359 | 26.969 | yes | 26.969 | **YES** |
| 2 | ` tr` | 16.891 | 15.852 | +1.039 | 17.930 | no | —(rejected) |  |
| 3 | ` h` | 16.609 | 16.234 | +0.375 | 16.984 | no | —(rejected) |  |
| 4 | `tr` | 16.250 | 15.352 | +0.898 | 17.148 | no | —(rejected) |  |
| 5 | `h` | 16.000 | 15.773 | +0.227 | 16.227 | no | —(rejected) |  |
| 6 | ` escape` | 15.945 | 16.516 | -0.570 | 15.375 | no | —(rejected) |  |
| 7 | ` l` | 15.523 | 15.266 | +0.258 | 15.781 | no | —(rejected) |  |
| 8 | ` ex` | 15.312 | 15.422 | -0.109 | 15.203 | no | —(rejected) |  |
| 9 | ` station` | 15.117 | 14.977 | +0.141 | 15.258 | no | —(rejected) |  |
| 10 | `pl` | 15.062 | 15.031 | +0.031 | 15.094 | no | —(rejected) |  |


### image 724 — word "fire hydrant" (node: fire hydrant) — step 106

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `rant` | 32.094 | 31.656 | +0.438 | 32.531 | yes | 32.531 | **YES** |
| 2 | `r` | 18.094 | 18.016 | +0.078 | 18.172 | no | —(rejected) |  |
| 3 | `rate` | 17.453 | 17.234 | +0.219 | 17.672 | no | —(rejected) |  |
| 4 | `rent` | 15.469 | 15.398 | +0.070 | 15.539 | no | —(rejected) |  |
| 5 | `ant` | 15.070 | 15.188 | -0.117 | 14.953 | no | —(rejected) |  |
| 6 | `ra` | 14.844 | 14.672 | +0.172 | 15.016 | no | —(rejected) |  |
| 7 | `ron` | 13.414 | 13.516 | -0.102 | 13.312 | no | —(rejected) |  |
| 8 | `ran` | 13.117 | 13.219 | -0.102 | 13.016 | no | —(rejected) |  |
| 9 | `rane` | 12.938 | 12.977 | -0.039 | 12.898 | no | —(rejected) |  |
| 10 | `range` | 12.742 | 12.617 | +0.125 | 12.867 | no | —(rejected) |  |


### image 2473 — word "backpack" (node: backpack) — step 91

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` back` | 18.125 | 17.656 | +0.469 | 18.594 | yes | 18.594 | **YES** |
| 2 | ` few` | 17.406 | 17.047 | +0.359 | 17.766 | yes | 17.766 |  |
| 3 | ` couple` | 17.406 | 17.078 | +0.328 | 17.734 | yes | 17.734 |  |
| 4 | ` sk` | 16.906 | 16.281 | +0.625 | 17.531 | yes | 17.531 |  |
| 5 | ` person` | 16.844 | 16.750 | +0.094 | 16.938 | yes | 16.938 |  |
| 6 | ` pair` | 16.656 | 15.805 | +0.852 | 17.508 | yes | 17.508 |  |
| 7 | ` snow` | 16.359 | 15.352 | +1.008 | 17.367 | no | —(rejected) |  |
| 8 | ` ben` | 16.312 | 15.695 | +0.617 | 16.930 | no | —(rejected) |  |
| 9 | ` hand` | 16.062 | 16.078 | -0.016 | 16.047 | no | —(rejected) |  |
| 10 | `part` | 15.484 | 15.352 | +0.133 | 15.617 | no | —(rejected) |  |


### image 2473 — word "backpack" (node: backpack) — step 92

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `pack` | 25.281 | 25.281 | +0.000 | 25.281 | yes | 25.281 | **YES** |
| 2 | ` pack` | 17.109 | 17.219 | -0.109 | 17.000 | no | —(rejected) |  |
| 3 | `drop` | 17.000 | 17.422 | -0.422 | 16.578 | no | —(rejected) |  |
| 4 | `ward` | 16.141 | 16.125 | +0.016 | 16.156 | no | —(rejected) |  |
| 5 | `back` | 15.531 | 15.477 | +0.055 | 15.586 | no | —(rejected) |  |
| 6 | `yard` | 15.211 | 15.250 | -0.039 | 15.172 | no | —(rejected) |  |
| 7 | `-` | 14.336 | 14.516 | -0.180 | 14.156 | no | —(rejected) |  |
| 8 | `ho` | 13.938 | 13.930 | +0.008 | 13.945 | no | —(rejected) |  |
| 9 | `country` | 13.812 | 13.539 | +0.273 | 14.086 | no | —(rejected) |  |
| 10 | ` back` | 13.805 | 13.734 | +0.070 | 13.875 | no | —(rejected) |  |


### image 1584 — word "backpack" (node: backpack) — step 85

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` back` | 23.562 | 23.234 | +0.328 | 23.891 | yes | 23.891 | **YES** |
| 2 | ` hand` | 23.109 | 23.203 | -0.094 | 23.016 | yes | 23.016 |  |
| 3 | ` um` | 19.797 | 19.922 | -0.125 | 19.672 | no | —(rejected) |  |
| 4 | ` personal` | 19.562 | 19.484 | +0.078 | 19.641 | no | —(rejected) |  |
| 5 | ` b` | 18.953 | 18.734 | +0.219 | 19.172 | no | —(rejected) |  |
| 6 | ` items` | 18.641 | 18.641 | +0.000 | 18.641 | no | —(rejected) |  |
| 7 | ` lug` | 17.203 | 17.203 | +0.000 | 17.203 | no | —(rejected) |  |
| 8 | ` their` | 17.188 | 16.984 | +0.203 | 17.391 | no | —(rejected) |  |
| 9 | ` various` | 17.141 | 17.078 | +0.062 | 17.203 | no | —(rejected) |  |
| 10 | ` belong` | 17.094 | 17.031 | +0.062 | 17.156 | no | —(rejected) |  |


### image 1584 — word "backpack" (node: backpack) — step 86

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `pack` | 26.234 | 26.094 | +0.141 | 26.375 | yes | 26.375 | **YES** |
| 2 | ` pack` | 17.141 | 17.016 | +0.125 | 17.266 | no | —(rejected) |  |
| 3 | `back` | 15.875 | 15.656 | +0.219 | 16.094 | no | —(rejected) |  |
| 4 | `b` | 13.922 | 13.922 | +0.000 | 13.922 | no | —(rejected) |  |
| 5 | `s` | 13.570 | 13.523 | +0.047 | 13.617 | no | —(rejected) |  |
| 6 | ` back` | 13.250 | 13.102 | +0.148 | 13.398 | no | —(rejected) |  |
| 7 | `-` | 13.156 | 13.148 | +0.008 | 13.164 | no | —(rejected) |  |
| 8 | ` b` | 12.664 | 12.781 | -0.117 | 12.547 | no | —(rejected) |  |
| 9 | `p` | 12.422 | 12.328 | +0.094 | 12.516 | no | —(rejected) |  |
| 10 | `Pack` | 11.922 | 11.805 | +0.117 | 12.039 | no | —(rejected) |  |


### image 1584 — word "backpack" (node: backpack) — step 87

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `s` | 28.484 | 28.266 | +0.219 | 28.703 | yes | 28.703 | **YES** |
| 2 | ` and` | 17.375 | 17.219 | +0.156 | 17.531 | no | —(rejected) |  |
| 3 | ` or` | 15.617 | 15.609 | +0.008 | 15.625 | no | —(rejected) |  |
| 4 | ` b` | 15.508 | 15.688 | -0.180 | 15.328 | no | —(rejected) |  |
| 5 | ` stra` | 15.422 | 15.344 | +0.078 | 15.500 | no | —(rejected) |  |
| 6 | ` pack` | 15.289 | 15.312 | -0.023 | 15.266 | no | —(rejected) |  |
| 7 | `,` | 15.008 | 14.914 | +0.094 | 15.102 | no | —(rejected) |  |
| 8 | `pack` | 14.469 | 14.422 | +0.047 | 14.516 | no | —(rejected) |  |
| 9 | `.` | 14.055 | 14.062 | -0.008 | 14.047 | no | —(rejected) |  |
| 10 | `es` | 13.953 | 13.844 | +0.109 | 14.062 | no | —(rejected) |  |


### image 1584 — word "traffic light" (node: traffic light) — step 90

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` few` | 18.781 | 18.875 | -0.094 | 18.688 | yes | 18.688 |  |
| 2 | ` car` | 18.188 | 19.297 | -1.109 | 17.078 | yes | 17.078 |  |
| 3 | ` traffic` | 18.000 | 17.156 | +0.844 | 18.844 | yes | 18.844 | **YES** |
| 4 | ` person` | 17.766 | 17.922 | -0.156 | 17.609 | yes | 17.609 |  |
| 5 | ` couple` | 17.547 | 17.750 | -0.203 | 17.344 | yes | 17.344 |  |
| 6 | ` b` | 17.516 | 16.984 | +0.531 | 18.047 | yes | 18.047 |  |
| 7 | ` ben` | 17.266 | 17.391 | -0.125 | 17.141 | yes | 17.141 |  |
| 8 | `part` | 16.719 | 16.859 | -0.141 | 16.578 | no | —(rejected) |  |
| 9 | ` tr` | 16.516 | 17.484 | -0.969 | 15.547 | no | —(rejected) |  |
| 10 | ` total` | 16.297 | 16.438 | -0.141 | 16.156 | no | —(rejected) |  |


### image 1584 — word "traffic light" (node: traffic light) — step 91

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` light` | 24.766 | 23.562 | +1.203 | 25.969 | yes | 25.969 | **YES** |
| 2 | ` signal` | 19.219 | 17.797 | +1.422 | 20.641 | no | —(rejected) |  |
| 3 | ` sign` | 18.188 | 17.844 | +0.344 | 18.531 | no | —(rejected) |  |
| 4 | ` lights` | 16.844 | 15.688 | +1.156 | 18.000 | no | —(rejected) |  |
| 5 | ` jam` | 16.562 | 16.766 | -0.203 | 16.359 | no | —(rejected) |  |
| 6 | ` camera` | 16.328 | 16.469 | -0.141 | 16.188 | no | —(rejected) |  |
| 7 | ` stop` | 16.250 | 15.758 | +0.492 | 16.742 | no | —(rejected) |  |
| 8 | ` lamp` | 16.016 | 14.695 | +1.320 | 17.336 | no | —(rejected) |  |
| 9 | ` control` | 15.156 | 14.492 | +0.664 | 15.820 | no | —(rejected) |  |
| 10 | `-` | 15.078 | 15.164 | -0.086 | 14.992 | no | —(rejected) |  |


### image 6763 — word "car" (node: car) — step 71

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` few` | 17.703 | 17.578 | +0.125 | 17.828 | yes | 17.828 |  |
| 2 | ` d` | 17.641 | 17.219 | +0.422 | 18.062 | yes | 18.062 |  |
| 3 | ` couple` | 17.375 | 17.375 | +0.000 | 17.375 | yes | 17.375 |  |
| 4 | ` car` | 17.031 | 15.227 | +1.805 | 18.836 | yes | 18.836 | **YES** |
| 5 | ` bott` | 16.812 | 16.438 | +0.375 | 17.188 | yes | 17.188 |  |
| 6 | ` clock` | 16.641 | 17.453 | -0.812 | 15.828 | yes | 15.828 |  |
| 7 | ` chair` | 16.484 | 16.672 | -0.188 | 16.297 | yes | 16.297 |  |
| 8 | ` person` | 16.094 | 15.883 | +0.211 | 16.305 | yes | 16.305 |  |
| 9 | ` cell` | 15.875 | 15.516 | +0.359 | 16.234 | no | —(rejected) |  |
| 10 | ` c` | 15.602 | 16.078 | -0.477 | 15.125 | no | —(rejected) |  |


### image 6763 — word "bottle" (node: bottle) — step 84

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` other` | 20.156 | 20.234 | -0.078 | 20.078 | yes | 20.078 |  |
| 2 | ` bott` | 19.781 | 17.844 | +1.938 | 21.719 | yes | 21.719 | **YES** |
| 3 | ` ch` | 19.172 | 18.234 | +0.938 | 20.109 | yes | 20.109 |  |
| 4 | ` more` | 18.297 | 18.562 | -0.266 | 18.031 | no | —(rejected) |  |
| 5 | ` people` | 18.203 | 17.797 | +0.406 | 18.609 | no | —(rejected) |  |
| 6 | ` books` | 18.109 | 18.531 | -0.422 | 17.688 | no | —(rejected) |  |
| 7 | ` additional` | 17.047 | 17.047 | +0.000 | 17.047 | no | —(rejected) |  |
| 8 | ` cu` | 17.031 | 16.375 | +0.656 | 17.688 | no | —(rejected) |  |
| 9 | ` d` | 16.656 | 15.266 | +1.391 | 18.047 | no | —(rejected) |  |
| 10 | ` wine` | 16.531 | 15.320 | +1.211 | 17.742 | no | —(rejected) |  |


### image 6763 — word "bottle" (node: bottle) — step 85

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `les` | 29.219 | 29.125 | +0.094 | 29.312 | yes | 29.312 | **YES** |
| 2 | `le` | 20.688 | 20.906 | -0.219 | 20.469 | no | —(rejected) |  |
| 3 | `led` | 19.188 | 19.156 | +0.031 | 19.219 | no | —(rejected) |  |
| 4 | `LES` | 16.312 | 16.188 | +0.125 | 16.438 | no | —(rejected) |  |
| 5 | `l` | 16.062 | 16.125 | -0.062 | 16.000 | no | —(rejected) |  |
| 6 | `ling` | 15.500 | 15.578 | -0.078 | 15.422 | no | —(rejected) |  |
| 7 | `es` | 14.961 | 14.695 | +0.266 | 15.227 | no | —(rejected) |  |
| 8 | `lles` | 14.539 | 14.664 | -0.125 | 14.414 | no | —(rejected) |  |
| 9 | `lers` | 14.500 | 14.688 | -0.188 | 14.312 | no | —(rejected) |  |
| 10 | `lets` | 14.305 | 14.406 | -0.102 | 14.203 | no | —(rejected) |  |


### image 6763 — word "chair" (node: chair) — step 108

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` chair` | 19.328 | 18.469 | +0.859 | 20.188 | yes | 20.188 | **YES** |
| 2 | ` d` | 17.812 | 16.781 | +1.031 | 18.844 | yes | 18.844 |  |
| 3 | ` person` | 17.500 | 17.266 | +0.234 | 17.734 | no | —(rejected) |  |
| 4 | ` hand` | 17.328 | 17.516 | -0.188 | 17.141 | no | —(rejected) |  |
| 5 | ` cell` | 17.109 | 15.891 | +1.219 | 18.328 | no | —(rejected) |  |
| 6 | ` couple` | 16.812 | 15.820 | +0.992 | 17.805 | no | —(rejected) |  |
| 7 | ` clock` | 16.812 | 17.141 | -0.328 | 16.484 | no | —(rejected) |  |
| 8 | ` cup` | 16.516 | 16.375 | +0.141 | 16.656 | no | —(rejected) |  |
| 9 | ` third` | 16.234 | 15.969 | +0.266 | 16.500 | no | —(rejected) |  |
| 10 | ` book` | 16.000 | 16.078 | -0.078 | 15.922 | no | —(rejected) |  |


### image 1425 — word "doughnut" (node: donut) — step 16

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` d` | 16.891 | 16.656 | +0.234 | 17.125 | yes | 17.125 | **YES** |
| 2 | ` don` | 15.516 | 15.438 | +0.078 | 15.594 | yes | 15.594 |  |
| 3 | ` m` | 14.875 | 14.102 | +0.773 | 15.648 | no | —(rejected) |  |
| 4 | ` sand` | 14.578 | 15.117 | -0.539 | 14.039 | no | —(rejected) |  |
| 5 | ` cre` | 14.008 | 13.812 | +0.195 | 14.203 | no | —(rejected) |  |
| 6 | ` pas` | 13.945 | 14.062 | -0.117 | 13.828 | no | —(rejected) |  |
| 7 | ` c` | 13.703 | 13.828 | -0.125 | 13.578 | no | —(rejected) |  |
| 8 | ` cro` | 13.570 | 13.445 | +0.125 | 13.695 | no | —(rejected) |  |
| 9 | ` dess` | 13.555 | 13.891 | -0.336 | 13.219 | no | —(rejected) |  |
| 10 | ` or` | 13.508 | 13.617 | -0.109 | 13.398 | no | —(rejected) |  |


### image 1425 — word "doughnut" (node: donut) — step 17

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ough` | 25.938 | 26.125 | -0.188 | 25.750 | yes | 25.750 | **YES** |
| 2 | `ought` | 18.625 | 18.203 | +0.422 | 19.047 | no | —(rejected) |  |
| 3 | `oug` | 17.953 | 18.047 | -0.094 | 17.859 | no | —(rejected) |  |
| 4 | `ome` | 16.281 | 16.781 | -0.500 | 15.781 | no | —(rejected) |  |
| 5 | `um` | 15.750 | 15.031 | +0.719 | 16.469 | no | —(rejected) |  |
| 6 | `ish` | 15.609 | 15.766 | -0.156 | 15.453 | no | —(rejected) |  |
| 7 | `oun` | 15.477 | 15.477 | +0.000 | 15.477 | no | —(rejected) |  |
| 8 | `ess` | 15.430 | 15.227 | +0.203 | 15.633 | no | —(rejected) |  |
| 9 | `unk` | 15.047 | 14.922 | +0.125 | 15.172 | no | —(rejected) |  |
| 10 | `une` | 14.633 | 14.742 | -0.109 | 14.523 | no | —(rejected) |  |


### image 1425 — word "doughnut" (node: donut) — step 18

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `nut` | 20.625 | 20.484 | +0.141 | 20.766 | yes | 20.766 | **YES** |
| 2 | ` nut` | 13.688 | 13.531 | +0.156 | 13.844 | no | —(rejected) |  |
| 3 | `,` | 12.164 | 12.383 | -0.219 | 11.945 | no | —(rejected) |  |
| 4 | ` pas` | 12.016 | 12.039 | -0.023 | 11.992 | no | —(rejected) |  |
| 5 | `ut` | 11.812 | 11.281 | +0.531 | 12.344 | no | —(rejected) |  |
| 6 | ` or` | 11.594 | 11.609 | -0.016 | 11.578 | no | —(rejected) |  |
| 7 | ` filled` | 11.352 | 11.289 | +0.062 | 11.414 | no | —(rejected) |  |
| 8 | `n` | 11.195 | 10.977 | +0.219 | 11.414 | no | —(rejected) |  |
| 9 | ` b` | 10.797 | 10.883 | -0.086 | 10.711 | no | —(rejected) |  |
| 10 | `-` | 10.641 | 10.578 | +0.062 | 10.703 | no | —(rejected) |  |


### image 1425 — word "fork" (node: fork) — step 67

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 17.797 | 17.547 | +0.250 | 18.047 | yes | 18.047 |  |
| 2 | ` fork` | 17.609 | 15.680 | +1.930 | 19.539 | yes | 19.539 | **YES** |
| 3 | ` sp` | 17.344 | 16.766 | +0.578 | 17.922 | yes | 17.922 |  |
| 4 | ` cup` | 15.664 | 16.016 | -0.352 | 15.312 | no | —(rejected) |  |
| 5 | ` kn` | 15.266 | 14.969 | +0.297 | 15.562 | no | —(rejected) |  |
| 6 | ` wine` | 14.820 | 10.609 | +4.211 | 19.031 | no | —(rejected) |  |
| 7 | ` glass` | 14.805 | 11.812 | +2.992 | 17.797 | no | —(rejected) |  |
| 8 | ` bott` | 14.500 | 12.781 | +1.719 | 16.219 | no | —(rejected) |  |
| 9 | ` pair` | 14.344 | 12.617 | +1.727 | 16.070 | no | —(rejected) |  |
| 10 | ` ut` | 14.195 | 12.336 | +1.859 | 16.055 | no | —(rejected) |  |


## VCD — REAL OBJECTS


### image 776 — word "teddy bear" (node: teddy bear) — step 7

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` ted` | 17.094 | 17.219 | -0.125 | 16.969 | yes | 16.969 | **YES** |
| 2 | ` brown` | 16.469 | 16.453 | +0.016 | 16.484 | yes | 16.484 |  |
| 3 | ` stuff` | 15.977 | 15.719 | +0.258 | 16.234 | yes | 16.234 |  |
| 4 | ` large` | 14.992 | 15.094 | -0.102 | 14.891 | no | —(rejected) |  |
| 5 | ` c` | 14.828 | 14.492 | +0.336 | 15.164 | no | —(rejected) |  |
| 6 | ` pl` | 13.977 | 14.242 | -0.266 | 13.711 | no | —(rejected) |  |
| 7 | ` fl` | 13.836 | 14.000 | -0.164 | 13.672 | no | —(rejected) |  |
| 8 | ` small` | 13.805 | 13.625 | +0.180 | 13.984 | no | —(rejected) |  |
| 9 | ` different` | 13.633 | 13.008 | +0.625 | 14.258 | no | —(rejected) |  |
| 10 | ` old` | 13.352 | 11.484 | +1.867 | 15.219 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 8

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `dy` | 27.062 | 26.688 | +0.375 | 27.438 | yes | 27.438 | **YES** |
| 2 | `d` | 17.391 | 17.266 | +0.125 | 17.516 | no | —(rejected) |  |
| 3 | ` be` | 13.688 | 13.250 | +0.438 | 14.125 | no | —(rejected) |  |
| 4 | `di` | 13.469 | 13.336 | +0.133 | 13.602 | no | —(rejected) |  |
| 5 | `b` | 13.336 | 13.062 | +0.273 | 13.609 | no | —(rejected) |  |
| 6 | `die` | 13.055 | 13.008 | +0.047 | 13.102 | no | —(rejected) |  |
| 7 | ` bear` | 12.500 | 12.359 | +0.141 | 12.641 | no | —(rejected) |  |
| 8 | ` ted` | 12.297 | 12.164 | +0.133 | 12.430 | no | —(rejected) |  |
| 9 | `der` | 11.312 | 11.227 | +0.086 | 11.398 | no | —(rejected) |  |
| 10 | `ious` | 11.172 | 11.211 | -0.039 | 11.133 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 9

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` be` | 22.969 | 22.422 | +0.547 | 23.516 | yes | 23.516 | **YES** |
| 2 | ` bear` | 18.312 | 18.531 | -0.219 | 18.094 | no | —(rejected) |  |
| 3 | ` ted` | 14.039 | 13.977 | +0.062 | 14.102 | no | —(rejected) |  |
| 4 | ` brown` | 13.922 | 13.531 | +0.391 | 14.312 | no | —(rejected) |  |
| 5 | `b` | 13.914 | 13.633 | +0.281 | 14.195 | no | —(rejected) |  |
| 6 | `-` | 12.648 | 12.633 | +0.016 | 12.664 | no | —(rejected) |  |
| 7 | ` stuff` | 12.562 | 12.273 | +0.289 | 12.852 | no | —(rejected) |  |
| 8 | ` animals` | 12.234 | 11.617 | +0.617 | 12.852 | no | —(rejected) |  |
| 9 | ` b` | 11.555 | 11.484 | +0.070 | 11.625 | no | —(rejected) |  |
| 10 | ` and` | 11.266 | 10.742 | +0.523 | 11.789 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 10

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ars` | 32.938 | 32.375 | +0.562 | 33.500 | yes | 33.500 | **YES** |
| 2 | `ats` | 17.047 | 16.609 | +0.438 | 17.484 | no | —(rejected) |  |
| 3 | `ams` | 16.688 | 16.469 | +0.219 | 16.906 | no | —(rejected) |  |
| 4 | `ers` | 16.672 | 16.203 | +0.469 | 17.141 | no | —(rejected) |  |
| 5 | `es` | 16.391 | 16.156 | +0.234 | 16.625 | no | —(rejected) |  |
| 6 | `ans` | 15.484 | 15.750 | -0.266 | 15.219 | no | —(rejected) |  |
| 7 | `er` | 14.469 | 14.227 | +0.242 | 14.711 | no | —(rejected) |  |
| 8 | `a` | 14.344 | 14.211 | +0.133 | 14.477 | no | —(rejected) |  |
| 9 | `ads` | 14.289 | 14.039 | +0.250 | 14.539 | no | —(rejected) |  |
| 10 | `ar` | 14.078 | 14.422 | -0.344 | 13.734 | no | —(rejected) |  |


### image 776 — word "bed" (node: bed) — step 21

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bed` | 20.031 | 18.125 | +1.906 | 21.938 | yes | 21.938 | **YES** |
| 2 | ` blue` | 18.859 | 16.328 | +2.531 | 21.391 | yes | 21.391 |  |
| 3 | ` surface` | 17.781 | 17.844 | -0.062 | 17.719 | no | —(rejected) |  |
| 4 | ` c` | 17.422 | 15.727 | +1.695 | 19.117 | no | —(rejected) |  |
| 5 | ` p` | 16.828 | 17.547 | -0.719 | 16.109 | no | —(rejected) |  |
| 6 | ` blank` | 16.641 | 14.766 | +1.875 | 18.516 | no | —(rejected) |  |
| 7 | ` soft` | 16.516 | 15.008 | +1.508 | 18.023 | no | —(rejected) |  |
| 8 | ` light` | 15.891 | 12.930 | +2.961 | 18.852 | no | —(rejected) |  |
| 9 | ` table` | 15.875 | 16.375 | -0.500 | 15.375 | no | —(rejected) |  |
| 10 | ` white` | 15.875 | 14.758 | +1.117 | 16.992 | no | —(rejected) |  |


### image 776 — word "bear" (node: bear) — step 34

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` be` | 18.469 | 18.188 | +0.281 | 18.750 | yes | 18.750 | **YES** |
| 2 | ` of` | 18.234 | 17.969 | +0.266 | 18.500 | yes | 18.500 |  |
| 3 | ` ted` | 17.391 | 17.344 | +0.047 | 17.438 | yes | 17.438 |  |
| 4 | ` sitting` | 15.555 | 14.797 | +0.758 | 16.312 | no | —(rejected) |  |
| 5 | ` position` | 15.359 | 15.047 | +0.312 | 15.672 | no | —(rejected) |  |
| 6 | ` larger` | 14.859 | 14.445 | +0.414 | 15.273 | no | —(rejected) |  |
| 7 | ` over` | 14.695 | 15.109 | -0.414 | 14.281 | no | —(rejected) |  |
| 8 | ` bear` | 14.656 | 14.719 | -0.062 | 14.594 | no | —(rejected) |  |
| 9 | ` stuff` | 14.352 | 14.148 | +0.203 | 14.555 | no | —(rejected) |  |
| 10 | ` placed` | 14.250 | 14.352 | -0.102 | 14.148 | no | —(rejected) |  |


### image 776 — word "bear" (node: bear) — step 35

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ars` | 27.031 | 26.891 | +0.141 | 27.172 | yes | 27.172 | **YES** |
| 2 | `ams` | 17.109 | 16.750 | +0.359 | 17.469 | no | —(rejected) |  |
| 3 | `ans` | 15.344 | 15.445 | -0.102 | 15.242 | no | —(rejected) |  |
| 4 | `es` | 15.305 | 14.914 | +0.391 | 15.695 | no | —(rejected) |  |
| 5 | `ads` | 14.953 | 14.883 | +0.070 | 15.023 | no | —(rejected) |  |
| 6 | `ats` | 14.680 | 14.492 | +0.188 | 14.867 | no | —(rejected) |  |
| 7 | `ers` | 14.625 | 14.422 | +0.203 | 14.828 | no | —(rejected) |  |
| 8 | `a` | 13.984 | 13.891 | +0.094 | 14.078 | no | —(rejected) |  |
| 9 | `eds` | 13.688 | 13.742 | -0.055 | 13.633 | no | —(rejected) |  |
| 10 | `ings` | 13.547 | 13.211 | +0.336 | 13.883 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 49

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` ted` | 16.328 | 16.328 | +0.000 | 16.328 | yes | 16.328 | **YES** |
| 2 | ` be` | 15.867 | 15.594 | +0.273 | 16.141 | yes | 16.141 |  |
| 3 | ` arrangement` | 14.789 | 14.234 | +0.555 | 15.344 | yes | 15.344 |  |
| 4 | ` scene` | 13.703 | 13.023 | +0.680 | 14.383 | no | —(rejected) |  |
| 5 | ` bed` | 13.492 | 12.805 | +0.688 | 14.180 | no | —(rejected) |  |
| 6 | ` largest` | 13.062 | 12.867 | +0.195 | 13.258 | no | —(rejected) |  |
| 7 | ` colors` | 13.031 | 12.602 | +0.430 | 13.461 | no | —(rejected) |  |
| 8 | ` group` | 12.898 | 12.305 | +0.594 | 13.492 | no | —(rejected) |  |
| 9 | ` ass` | 12.461 | 11.992 | +0.469 | 12.930 | no | —(rejected) |  |
| 10 | ` collection` | 12.375 | 12.211 | +0.164 | 12.539 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 50

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `dy` | 26.234 | 25.969 | +0.266 | 26.500 | yes | 26.500 | **YES** |
| 2 | `d` | 19.297 | 18.906 | +0.391 | 19.688 | no | —(rejected) |  |
| 3 | `b` | 14.922 | 14.484 | +0.438 | 15.359 | no | —(rejected) |  |
| 4 | `der` | 13.156 | 12.820 | +0.336 | 13.492 | no | —(rejected) |  |
| 5 | `die` | 12.953 | 12.641 | +0.312 | 13.266 | no | —(rejected) |  |
| 6 | `di` | 12.594 | 12.367 | +0.227 | 12.820 | no | —(rejected) |  |
| 7 | ` bear` | 12.094 | 12.102 | -0.008 | 12.086 | no | —(rejected) |  |
| 8 | ` be` | 12.039 | 11.680 | +0.359 | 12.398 | no | —(rejected) |  |
| 9 | `ds` | 11.320 | 11.008 | +0.312 | 11.633 | no | —(rejected) |  |
| 10 | `ders` | 11.281 | 10.695 | +0.586 | 11.867 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 51

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` be` | 25.688 | 25.547 | +0.141 | 25.828 | yes | 25.828 | **YES** |
| 2 | ` bear` | 19.859 | 19.875 | -0.016 | 19.844 | no | —(rejected) |  |
| 3 | `-` | 14.555 | 14.500 | +0.055 | 14.609 | no | —(rejected) |  |
| 4 | ` collection` | 14.336 | 13.727 | +0.609 | 14.945 | no | —(rejected) |  |
| 5 | ` b` | 14.172 | 14.156 | +0.016 | 14.188 | no | —(rejected) |  |
| 6 | ` animals` | 13.633 | 13.352 | +0.281 | 13.914 | no | —(rejected) |  |
| 7 | ` bunch` | 13.312 | 13.039 | +0.273 | 13.586 | no | —(rejected) |  |
| 8 | ` to` | 13.188 | 13.164 | +0.023 | 13.211 | no | —(rejected) |  |
| 9 | ` balls` | 13.188 | 13.211 | -0.023 | 13.164 | no | —(rejected) |  |
| 10 | `'` | 13.156 | 13.250 | -0.094 | 13.062 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 52

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ars` | 31.234 | 31.281 | -0.047 | 31.188 | yes | 31.188 | **YES** |
| 2 | `ans` | 18.062 | 18.375 | -0.312 | 17.750 | no | —(rejected) |  |
| 3 | `ers` | 17.359 | 17.547 | -0.188 | 17.172 | no | —(rejected) |  |
| 4 | `ams` | 16.672 | 16.922 | -0.250 | 16.422 | no | —(rejected) |  |
| 5 | `es` | 16.281 | 16.625 | -0.344 | 15.938 | no | —(rejected) |  |
| 6 | `ats` | 15.320 | 15.414 | -0.094 | 15.227 | no | —(rejected) |  |
| 7 | `ar` | 14.281 | 14.781 | -0.500 | 13.781 | no | —(rejected) |  |
| 8 | `a` | 13.547 | 13.562 | -0.016 | 13.531 | no | —(rejected) |  |
| 9 | `as` | 13.359 | 13.820 | -0.461 | 12.898 | no | —(rejected) |  |
| 10 | `ared` | 13.312 | 13.344 | -0.031 | 13.281 | no | —(rejected) |  |


### image 4765 — word "man" (node: person) — step 5

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 18.859 | 18.312 | +0.547 | 19.406 | yes | 19.406 | **YES** |
| 2 | ` person` | 18.453 | 17.859 | +0.594 | 19.047 | yes | 19.047 |  |
| 3 | ` sur` | 18.266 | 17.734 | +0.531 | 18.797 | yes | 18.797 |  |
| 4 | ` young` | 17.938 | 16.922 | +1.016 | 18.953 | yes | 18.953 |  |
| 5 | ` thr` | 17.641 | 16.875 | +0.766 | 18.406 | yes | 18.406 |  |
| 6 | ` male` | 17.219 | 16.500 | +0.719 | 17.938 | no | —(rejected) |  |
| 7 | ` woman` | 16.938 | 14.023 | +2.914 | 19.852 | no | —(rejected) |  |
| 8 | ` dynamic` | 16.875 | 17.125 | -0.250 | 16.625 | no | —(rejected) |  |
| 9 | ` l` | 16.031 | 15.867 | +0.164 | 16.195 | no | —(rejected) |  |
| 10 | ` sk` | 15.906 | 15.172 | +0.734 | 16.641 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 14

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sur` | 23.266 | 23.141 | +0.125 | 23.391 | yes | 23.391 | **YES** |
| 2 | ` white` | 20.953 | 20.984 | -0.031 | 20.922 | no | —(rejected) |  |
| 3 | ` large` | 18.359 | 18.453 | -0.094 | 18.266 | no | —(rejected) |  |
| 4 | ` yellow` | 17.984 | 17.266 | +0.719 | 18.703 | no | —(rejected) |  |
| 5 | ` long` | 17.859 | 17.266 | +0.594 | 18.453 | no | —(rejected) |  |
| 6 | ` wave` | 17.703 | 18.000 | -0.297 | 17.406 | no | —(rejected) |  |
| 7 | ` small` | 17.188 | 16.672 | +0.516 | 17.703 | no | —(rejected) |  |
| 8 | ` bo` | 16.906 | 16.484 | +0.422 | 17.328 | no | —(rejected) |  |
| 9 | ` board` | 16.484 | 16.141 | +0.344 | 16.828 | no | —(rejected) |  |
| 10 | ` bright` | 16.172 | 15.859 | +0.312 | 16.484 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 15

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `f` | 27.750 | 27.438 | +0.312 | 28.062 | yes | 28.062 | **YES** |
| 2 | `board` | 17.594 | 17.406 | +0.188 | 17.781 | no | —(rejected) |  |
| 3 | `fer` | 16.672 | 16.516 | +0.156 | 16.828 | no | —(rejected) |  |
| 4 | `ft` | 14.047 | 13.867 | +0.180 | 14.227 | no | —(rejected) |  |
| 5 | `fers` | 13.422 | 13.203 | +0.219 | 13.641 | no | —(rejected) |  |
| 6 | `fo` | 12.898 | 12.703 | +0.195 | 13.094 | no | —(rejected) |  |
| 7 | `fc` | 11.773 | 11.688 | +0.086 | 11.859 | no | —(rejected) |  |
| 8 | `fin` | 11.609 | 11.570 | +0.039 | 11.648 | no | —(rejected) |  |
| 9 | `face` | 11.578 | 11.586 | -0.008 | 11.570 | no | —(rejected) |  |
| 10 | `ge` | 11.547 | 11.641 | -0.094 | 11.453 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 16

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 25.594 | 25.547 | +0.047 | 25.641 | yes | 25.641 | **YES** |
| 2 | ` board` | 16.891 | 16.906 | -0.016 | 16.875 | no | —(rejected) |  |
| 3 | `ing` | 16.047 | 15.812 | +0.234 | 16.281 | no | —(rejected) |  |
| 4 | `boards` | 14.367 | 14.336 | +0.031 | 14.398 | no | —(rejected) |  |
| 5 | ` sur` | 13.102 | 12.812 | +0.289 | 13.391 | no | —(rejected) |  |
| 6 | `ba` | 12.289 | 12.297 | -0.008 | 12.281 | no | —(rejected) |  |
| 7 | `-` | 12.164 | 12.062 | +0.102 | 12.266 | no | —(rejected) |  |
| 8 | `able` | 12.117 | 11.992 | +0.125 | 12.242 | no | —(rejected) |  |
| 9 | `bo` | 11.898 | 11.906 | -0.008 | 11.891 | no | —(rejected) |  |
| 10 | `acing` | 10.875 | 10.805 | +0.070 | 10.945 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 63

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sur` | 18.750 | 18.703 | +0.047 | 18.797 | yes | 18.797 | **YES** |
| 2 | ` wave` | 17.906 | 17.875 | +0.031 | 17.938 | yes | 17.938 |  |
| 3 | ` ocean` | 15.922 | 15.859 | +0.062 | 15.984 | no | —(rejected) |  |
| 4 | ` waves` | 15.188 | 15.234 | -0.047 | 15.141 | no | —(rejected) |  |
| 5 | ` focus` | 15.102 | 15.422 | -0.320 | 14.781 | no | —(rejected) |  |
| 6 | ` white` | 14.664 | 14.867 | -0.203 | 14.461 | no | —(rejected) |  |
| 7 | ` board` | 14.305 | 14.180 | +0.125 | 14.430 | no | —(rejected) |  |
| 8 | ` water` | 14.062 | 13.891 | +0.172 | 14.234 | no | —(rejected) |  |
| 9 | ` cr` | 14.047 | 13.938 | +0.109 | 14.156 | no | —(rejected) |  |
| 10 | ` main` | 13.922 | 13.969 | -0.047 | 13.875 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 64

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `f` | 27.594 | 27.500 | +0.094 | 27.688 | yes | 27.688 | **YES** |
| 2 | `board` | 17.531 | 17.344 | +0.188 | 17.719 | no | —(rejected) |  |
| 3 | `fer` | 16.781 | 16.719 | +0.062 | 16.844 | no | —(rejected) |  |
| 4 | `ft` | 14.180 | 13.961 | +0.219 | 14.398 | no | —(rejected) |  |
| 5 | `fers` | 13.367 | 13.328 | +0.039 | 13.406 | no | —(rejected) |  |
| 6 | `fc` | 13.352 | 13.266 | +0.086 | 13.438 | no | —(rejected) |  |
| 7 | `fo` | 13.156 | 13.094 | +0.062 | 13.219 | no | —(rejected) |  |
| 8 | `ge` | 12.117 | 12.055 | +0.062 | 12.180 | no | —(rejected) |  |
| 9 | `fly` | 11.984 | 11.914 | +0.070 | 12.055 | no | —(rejected) |  |
| 10 | `face` | 11.867 | 11.805 | +0.062 | 11.930 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 65

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 26.391 | 26.312 | +0.078 | 26.469 | yes | 26.469 | **YES** |
| 2 | `boards` | 16.547 | 16.656 | -0.109 | 16.438 | no | —(rejected) |  |
| 3 | `ing` | 15.961 | 15.883 | +0.078 | 16.039 | no | —(rejected) |  |
| 4 | ` board` | 15.844 | 15.867 | -0.023 | 15.820 | no | —(rejected) |  |
| 5 | ` breaking` | 13.047 | 13.133 | -0.086 | 12.961 | no | —(rejected) |  |
| 6 | `able` | 12.672 | 12.570 | +0.102 | 12.773 | no | —(rejected) |  |
| 7 | ` and` | 12.609 | 12.867 | -0.258 | 12.352 | no | —(rejected) |  |
| 8 | `acing` | 12.555 | 12.430 | +0.125 | 12.680 | no | —(rejected) |  |
| 9 | ` wave` | 12.445 | 12.258 | +0.188 | 12.633 | no | —(rejected) |  |
| 10 | ` in` | 12.281 | 12.414 | -0.133 | 12.148 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 5

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` stop` | 20.609 | 20.141 | +0.469 | 21.078 | yes | 21.078 | **YES** |
| 2 | ` and` | 17.906 | 18.203 | -0.297 | 17.609 | no | —(rejected) |  |
| 3 | ` oct` | 16.484 | 16.781 | -0.297 | 16.188 | no | —(rejected) |  |
| 4 | ` traffic` | 16.094 | 15.227 | +0.867 | 16.961 | no | —(rejected) |  |
| 5 | `,` | 15.461 | 15.602 | -0.141 | 15.320 | no | —(rejected) |  |
| 6 | ` ST` | 15.391 | 15.281 | +0.109 | 15.500 | no | —(rejected) |  |
| 7 | ` street` | 15.086 | 14.398 | +0.688 | 15.773 | no | —(rejected) |  |
| 8 | ` Stop` | 14.867 | 15.312 | -0.445 | 14.422 | no | —(rejected) |  |
| 9 | ` "` | 14.688 | 14.742 | -0.055 | 14.633 | no | —(rejected) |  |
| 10 | ` four` | 14.383 | 13.719 | +0.664 | 15.047 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 6

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sign` | 25.000 | 24.906 | +0.094 | 25.094 | yes | 25.094 | **YES** |
| 2 | ` or` | 16.922 | 17.469 | -0.547 | 16.375 | no | —(rejected) |  |
| 3 | `-` | 16.859 | 17.016 | -0.156 | 16.703 | no | —(rejected) |  |
| 4 | ` and` | 16.297 | 16.703 | -0.406 | 15.891 | no | —(rejected) |  |
| 5 | `light` | 15.781 | 15.281 | +0.500 | 16.281 | no | —(rejected) |  |
| 6 | ` street` | 15.711 | 15.438 | +0.273 | 15.984 | no | —(rejected) |  |
| 7 | ` signs` | 15.312 | 15.219 | +0.094 | 15.406 | no | —(rejected) |  |
| 8 | ` signal` | 14.938 | 15.055 | -0.117 | 14.820 | no | —(rejected) |  |
| 9 | ` light` | 14.344 | 14.203 | +0.141 | 14.484 | no | —(rejected) |  |
| 10 | ` s` | 13.953 | 14.008 | -0.055 | 13.898 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 16

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` stop` | 19.078 | 19.156 | -0.078 | 19.000 | yes | 19.000 | **YES** |
| 2 | ` sign` | 17.984 | 17.844 | +0.141 | 18.125 | yes | 18.125 |  |
| 3 | ` pole` | 16.016 | 16.203 | -0.188 | 15.828 | no | —(rejected) |  |
| 4 | ` street` | 15.484 | 15.094 | +0.391 | 15.875 | no | —(rejected) |  |
| 5 | ` red` | 15.203 | 15.273 | -0.070 | 15.133 | no | —(rejected) |  |
| 6 | ` scene` | 14.945 | 14.859 | +0.086 | 15.031 | no | —(rejected) |  |
| 7 | ` traffic` | 14.727 | 14.242 | +0.484 | 15.211 | no | —(rejected) |  |
| 8 | ` top` | 14.031 | 13.773 | +0.258 | 14.289 | no | —(rejected) |  |
| 9 | ` road` | 13.812 | 13.617 | +0.195 | 14.008 | no | —(rejected) |  |
| 10 | ` intersection` | 13.664 | 13.039 | +0.625 | 14.289 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 17

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sign` | 26.172 | 26.172 | +0.000 | 26.172 | yes | 26.172 | **YES** |
| 2 | ` signs` | 17.766 | 17.766 | +0.000 | 17.766 | no | —(rejected) |  |
| 3 | ` signal` | 16.234 | 16.344 | -0.109 | 16.125 | no | —(rejected) |  |
| 4 | `-` | 15.797 | 15.883 | -0.086 | 15.711 | no | —(rejected) |  |
| 5 | ` is` | 15.484 | 15.391 | +0.094 | 15.578 | no | —(rejected) |  |
| 6 | ` light` | 14.414 | 14.344 | +0.070 | 14.484 | no | —(rejected) |  |
| 7 | `light` | 14.227 | 14.086 | +0.141 | 14.367 | no | —(rejected) |  |
| 8 | ` and` | 14.133 | 14.148 | -0.016 | 14.117 | no | —(rejected) |  |
| 9 | ` s` | 14.055 | 14.094 | -0.039 | 14.016 | no | —(rejected) |  |
| 10 | ` has` | 13.039 | 12.922 | +0.117 | 13.156 | no | —(rejected) |  |


### image 724 — word "truck" (node: truck) — step 42

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` tr` | 20.547 | 21.172 | -0.625 | 19.922 | yes | 19.922 | **YES** |
| 2 | ` car` | 19.109 | 19.969 | -0.859 | 18.250 | yes | 18.250 |  |
| 3 | ` bus` | 18.328 | 19.219 | -0.891 | 17.438 | no | —(rejected) |  |
| 4 | ` couple` | 17.812 | 18.484 | -0.672 | 17.141 | no | —(rejected) |  |
| 5 | ` few` | 17.047 | 17.922 | -0.875 | 16.172 | no | —(rejected) |  |
| 6 | ` large` | 16.172 | 16.969 | -0.797 | 15.375 | no | —(rejected) |  |
| 7 | ` white` | 15.828 | 15.234 | +0.594 | 16.422 | no | —(rejected) |  |
| 8 | ` small` | 15.578 | 15.984 | -0.406 | 15.172 | no | —(rejected) |  |
| 9 | ` mix` | 15.406 | 16.344 | -0.938 | 14.469 | no | —(rejected) |  |
| 10 | ` motor` | 15.258 | 15.727 | -0.469 | 14.789 | no | —(rejected) |  |


### image 724 — word "truck" (node: truck) — step 43

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `uck` | 26.578 | 26.797 | -0.219 | 26.359 | yes | 26.359 | **YES** |
| 2 | `ash` | 17.797 | 17.250 | +0.547 | 18.344 | no | —(rejected) |  |
| 3 | `unk` | 15.000 | 15.414 | -0.414 | 14.586 | no | —(rejected) |  |
| 4 | `ucker` | 14.797 | 14.836 | -0.039 | 14.758 | no | —(rejected) |  |
| 5 | `ump` | 14.602 | 14.977 | -0.375 | 14.227 | no | —(rejected) |  |
| 6 | `uc` | 14.570 | 14.523 | +0.047 | 14.617 | no | —(rejected) |  |
| 7 | `icy` | 14.547 | 13.766 | +0.781 | 15.328 | no | —(rejected) |  |
| 8 | `oupe` | 13.688 | 13.688 | +0.000 | 13.688 | no | —(rejected) |  |
| 9 | `amp` | 13.320 | 13.203 | +0.117 | 13.438 | no | —(rejected) |  |
| 10 | `uce` | 13.094 | 12.820 | +0.273 | 13.367 | no | —(rejected) |  |


### image 724 — word "car" (node: car) — step 46

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` car` | 20.719 | 21.703 | -0.984 | 19.734 | yes | 19.734 | **YES** |
| 2 | ` bus` | 19.906 | 20.734 | -0.828 | 19.078 | yes | 19.078 |  |
| 3 | ` couple` | 18.953 | 20.062 | -1.109 | 17.844 | no | —(rejected) |  |
| 4 | ` few` | 18.641 | 20.031 | -1.391 | 17.250 | no | —(rejected) |  |
| 5 | ` motor` | 17.078 | 17.703 | -0.625 | 16.453 | no | —(rejected) |  |
| 6 | ` small` | 16.641 | 17.234 | -0.594 | 16.047 | no | —(rejected) |  |
| 7 | ` smaller` | 16.547 | 17.672 | -1.125 | 15.422 | no | —(rejected) |  |
| 8 | ` van` | 16.484 | 16.984 | -0.500 | 15.984 | no | —(rejected) |  |
| 9 | ` tr` | 16.156 | 16.500 | -0.344 | 15.812 | no | —(rejected) |  |
| 10 | ` tra` | 15.141 | 15.047 | +0.094 | 15.234 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 92

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` stop` | 14.367 | 14.484 | -0.117 | 14.250 | yes | 14.250 | **YES** |
| 2 | ` ups` | 14.133 | 14.047 | +0.086 | 14.219 | yes | 14.219 |  |
| 3 | ` presence` | 13.984 | 13.773 | +0.211 | 14.195 | yes | 14.195 |  |
| 4 | ` overall` | 13.852 | 13.859 | -0.008 | 13.844 | yes | 13.844 |  |
| 5 | ` combination` | 12.969 | 13.188 | -0.219 | 12.750 | yes | 12.750 |  |
| 6 | ` unusual` | 12.961 | 12.805 | +0.156 | 13.117 | yes | 13.117 |  |
| 7 | ` street` | 12.203 | 11.812 | +0.391 | 12.594 | no | —(rejected) |  |
| 8 | ` image` | 12.000 | 12.367 | -0.367 | 11.633 | no | —(rejected) |  |
| 9 | ` unique` | 11.930 | 11.930 | +0.000 | 11.930 | no | —(rejected) |  |
| 10 | ` mix` | 11.883 | 11.711 | +0.172 | 12.055 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 93

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sign` | 25.500 | 25.469 | +0.031 | 25.531 | yes | 25.531 | **YES** |
| 2 | ` signs` | 18.219 | 18.188 | +0.031 | 18.250 | no | —(rejected) |  |
| 3 | `-` | 15.953 | 16.062 | -0.109 | 15.844 | no | —(rejected) |  |
| 4 | ` signal` | 15.781 | 15.852 | -0.070 | 15.711 | no | —(rejected) |  |
| 5 | `light` | 15.586 | 15.320 | +0.266 | 15.852 | no | —(rejected) |  |
| 6 | ` light` | 15.266 | 15.148 | +0.117 | 15.383 | no | —(rejected) |  |
| 7 | ` and` | 14.609 | 14.711 | -0.102 | 14.508 | no | —(rejected) |  |
| 8 | ` s` | 14.602 | 14.609 | -0.008 | 14.594 | no | —(rejected) |  |
| 9 | ` ups` | 14.320 | 13.812 | +0.508 | 14.828 | no | —(rejected) |  |
| 10 | ` pole` | 13.562 | 13.750 | -0.188 | 13.375 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 10

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 21.422 | 20.406 | +1.016 | 22.438 | yes | 22.438 |  |
| 2 | ` sk` | 21.391 | 20.516 | +0.875 | 22.266 | yes | 22.266 |  |
| 3 | ` snow` | 20.828 | 18.719 | +2.109 | 22.938 | yes | 22.938 | **YES** |
| 4 | ` man` | 20.781 | 19.656 | +1.125 | 21.906 | yes | 21.906 |  |
| 5 | ` ski` | 18.797 | 18.328 | +0.469 | 19.266 | no | —(rejected) |  |
| 6 | ` young` | 18.297 | 15.664 | +2.633 | 20.930 | no | —(rejected) |  |
| 7 | ` male` | 17.922 | 16.859 | +1.062 | 18.984 | no | —(rejected) |  |
| 8 | ` professional` | 16.594 | 16.016 | +0.578 | 17.172 | no | —(rejected) |  |
| 9 | ` winter` | 16.531 | 15.484 | +1.047 | 17.578 | no | —(rejected) |  |
| 10 | ` cross` | 16.281 | 17.422 | -1.141 | 15.141 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 11

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 22.125 | 20.172 | +1.953 | 24.078 | yes | 24.078 | **YES** |
| 2 | ` sk` | 19.516 | 19.375 | +0.141 | 19.656 | no | —(rejected) |  |
| 3 | ` sports` | 16.359 | 15.234 | +1.125 | 17.484 | no | —(rejected) |  |
| 4 | `-` | 16.219 | 16.156 | +0.062 | 16.281 | no | —(rejected) |  |
| 5 | ` ski` | 16.062 | 16.188 | -0.125 | 15.938 | no | —(rejected) |  |
| 6 | `y` | 15.055 | 14.383 | +0.672 | 15.727 | no | —(rejected) |  |
| 7 | ` board` | 14.969 | 13.734 | +1.234 | 16.203 | no | —(rejected) |  |
| 8 | ` sport` | 14.750 | 14.102 | +0.648 | 15.398 | no | —(rejected) |  |
| 9 | ` ath` | 14.633 | 14.367 | +0.266 | 14.898 | no | —(rejected) |  |
| 10 | `sk` | 14.148 | 14.133 | +0.016 | 14.164 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 12

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `er` | 24.859 | 24.641 | +0.219 | 25.078 | yes | 25.078 | **YES** |
| 2 | `ing` | 19.312 | 19.062 | +0.250 | 19.562 | no | —(rejected) |  |
| 3 | ` j` | 18.047 | 17.734 | +0.312 | 18.359 | no | —(rejected) |  |
| 4 | ` jump` | 17.281 | 16.984 | +0.297 | 17.578 | no | —(rejected) |  |
| 5 | ` ath` | 15.711 | 15.789 | -0.078 | 15.633 | no | —(rejected) |  |
| 6 | `ers` | 15.242 | 15.273 | -0.031 | 15.211 | no | —(rejected) |  |
| 7 | ` r` | 15.219 | 15.055 | +0.164 | 15.383 | no | —(rejected) |  |
| 8 | ` enthus` | 14.664 | 14.016 | +0.648 | 15.312 | no | —(rejected) |  |
| 9 | ` f` | 14.523 | 14.656 | -0.133 | 14.391 | no | —(rejected) |  |
| 10 | ` trick` | 13.914 | 13.750 | +0.164 | 14.078 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 27

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` snow` | 20.484 | 20.156 | +0.328 | 20.812 | yes | 20.812 | **YES** |
| 2 | ` person` | 18.219 | 18.109 | +0.109 | 18.328 | no | —(rejected) |  |
| 3 | ` sk` | 17.391 | 17.281 | +0.109 | 17.500 | no | —(rejected) |  |
| 4 | ` main` | 17.109 | 16.922 | +0.188 | 17.297 | no | —(rejected) |  |
| 5 | ` man` | 16.953 | 16.453 | +0.500 | 17.453 | no | —(rejected) |  |
| 6 | ` individual` | 15.609 | 15.438 | +0.172 | 15.781 | no | —(rejected) |  |
| 7 | ` air` | 15.602 | 15.344 | +0.258 | 15.859 | no | —(rejected) |  |
| 8 | ` young` | 14.930 | 13.562 | +1.367 | 16.297 | no | —(rejected) |  |
| 9 | ` action` | 14.828 | 14.680 | +0.148 | 14.977 | no | —(rejected) |  |
| 10 | ` ath` | 14.789 | 14.711 | +0.078 | 14.867 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 28

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 22.516 | 22.250 | +0.266 | 22.781 | yes | 22.781 | **YES** |
| 2 | `-` | 14.820 | 14.938 | -0.117 | 14.703 | no | —(rejected) |  |
| 3 | `y` | 14.250 | 14.305 | -0.055 | 14.195 | no | —(rejected) |  |
| 4 | ` sports` | 13.656 | 13.695 | -0.039 | 13.617 | no | —(rejected) |  |
| 5 | ` board` | 13.562 | 13.555 | +0.008 | 13.570 | no | —(rejected) |  |
| 6 | `boards` | 13.312 | 13.125 | +0.188 | 13.500 | no | —(rejected) |  |
| 7 | ` sk` | 12.680 | 13.148 | -0.469 | 12.211 | no | —(rejected) |  |
| 8 | ` is` | 12.352 | 12.336 | +0.016 | 12.367 | no | —(rejected) |  |
| 9 | ` r` | 12.016 | 11.930 | +0.086 | 12.102 | no | —(rejected) |  |
| 10 | ` sport` | 11.797 | 11.883 | -0.086 | 11.711 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 29

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `er` | 24.453 | 24.438 | +0.016 | 24.469 | yes | 24.469 | **YES** |
| 2 | ` is` | 18.562 | 18.422 | +0.141 | 18.703 | no | —(rejected) |  |
| 3 | `ing` | 16.859 | 16.906 | -0.047 | 16.812 | no | —(rejected) |  |
| 4 | ` can` | 16.562 | 16.625 | -0.062 | 16.500 | no | —(rejected) |  |
| 5 | `,` | 16.016 | 15.977 | +0.039 | 16.055 | no | —(rejected) |  |
| 6 | ` f` | 15.805 | 15.422 | +0.383 | 16.188 | no | —(rejected) |  |
| 7 | ` j` | 15.562 | 15.219 | +0.344 | 15.906 | no | —(rejected) |  |
| 8 | `ers` | 15.219 | 15.250 | -0.031 | 15.188 | no | —(rejected) |  |
| 9 | ` and` | 14.961 | 15.000 | -0.039 | 14.922 | no | —(rejected) |  |
| 10 | ` r` | 14.844 | 14.875 | -0.031 | 14.812 | no | —(rejected) |  |


### image 2473 — word "people" (node: person) — step 53

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 24.844 | 24.281 | +0.562 | 25.406 | yes | 25.406 | **YES** |
| 2 | ` individuals` | 21.578 | 21.016 | +0.562 | 22.141 | no | —(rejected) |  |
| 3 | ` snow` | 19.469 | 19.125 | +0.344 | 19.812 | no | —(rejected) |  |
| 4 | ` ski` | 18.453 | 18.391 | +0.062 | 18.516 | no | —(rejected) |  |
| 5 | ` persons` | 18.422 | 18.219 | +0.203 | 18.625 | no | —(rejected) |  |
| 6 | ` figures` | 17.938 | 18.219 | -0.281 | 17.656 | no | —(rejected) |  |
| 7 | ` on` | 17.531 | 16.547 | +0.984 | 18.516 | no | —(rejected) |  |
| 8 | ` objects` | 16.766 | 17.156 | -0.391 | 16.375 | no | —(rejected) |  |
| 9 | ` spect` | 16.719 | 15.562 | +1.156 | 17.875 | no | —(rejected) |  |
| 10 | ` elements` | 16.453 | 16.922 | -0.469 | 15.984 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 63

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` snow` | 23.266 | 22.750 | +0.516 | 23.781 | yes | 23.781 | **YES** |
| 2 | ` ski` | 17.484 | 17.453 | +0.031 | 17.516 | no | —(rejected) |  |
| 3 | ` compet` | 17.438 | 17.328 | +0.109 | 17.547 | no | —(rejected) |  |
| 4 | ` participants` | 17.297 | 17.250 | +0.047 | 17.344 | no | —(rejected) |  |
| 5 | ` athlet` | 17.156 | 17.359 | -0.203 | 16.953 | no | —(rejected) |  |
| 6 | ` rid` | 16.797 | 16.438 | +0.359 | 17.156 | no | —(rejected) |  |
| 7 | ` winter` | 16.719 | 16.688 | +0.031 | 16.750 | no | —(rejected) |  |
| 8 | ` enthus` | 16.359 | 16.266 | +0.094 | 16.453 | no | —(rejected) |  |
| 9 | ` sports` | 16.078 | 16.156 | -0.078 | 16.000 | no | —(rejected) |  |
| 10 | ` board` | 14.430 | 14.359 | +0.070 | 14.500 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 64

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 22.000 | 21.781 | +0.219 | 22.219 | yes | 22.219 | **YES** |
| 2 | ` sports` | 17.688 | 17.781 | -0.094 | 17.594 | no | —(rejected) |  |
| 3 | ` enthus` | 16.312 | 16.531 | -0.219 | 16.094 | no | —(rejected) |  |
| 4 | ` sport` | 14.930 | 15.055 | -0.125 | 14.805 | no | —(rejected) |  |
| 5 | ` athlet` | 14.141 | 14.344 | -0.203 | 13.938 | no | —(rejected) |  |
| 6 | ` board` | 13.898 | 13.930 | -0.031 | 13.867 | no | —(rejected) |  |
| 7 | `-` | 13.891 | 13.992 | -0.102 | 13.789 | no | —(rejected) |  |
| 8 | `boards` | 13.438 | 13.281 | +0.156 | 13.594 | no | —(rejected) |  |
| 9 | ` rid` | 13.008 | 12.992 | +0.016 | 13.023 | no | —(rejected) |  |
| 10 | ` snow` | 12.578 | 12.453 | +0.125 | 12.703 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 65

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ers` | 25.812 | 25.781 | +0.031 | 25.844 | yes | 25.844 | **YES** |
| 2 | `ing` | 21.875 | 21.891 | -0.016 | 21.859 | no | —(rejected) |  |
| 3 | ` enthus` | 19.922 | 19.922 | +0.000 | 19.922 | no | —(rejected) |  |
| 4 | `er` | 19.062 | 19.016 | +0.047 | 19.109 | no | —(rejected) |  |
| 5 | ` or` | 16.609 | 16.688 | -0.078 | 16.531 | no | —(rejected) |  |
| 6 | ` and` | 16.438 | 16.422 | +0.016 | 16.453 | no | —(rejected) |  |
| 7 | ` rid` | 15.773 | 15.742 | +0.031 | 15.805 | no | —(rejected) |  |
| 8 | `/` | 15.383 | 15.492 | -0.109 | 15.273 | no | —(rejected) |  |
| 9 | ` participants` | 15.320 | 15.352 | -0.031 | 15.289 | no | —(rejected) |  |
| 10 | ` friends` | 14.977 | 14.883 | +0.094 | 15.070 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 119

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` snow` | 19.812 | 19.469 | +0.344 | 20.156 | yes | 20.156 | **YES** |
| 2 | ` main` | 15.859 | 15.750 | +0.109 | 15.969 | no | —(rejected) |  |
| 3 | ` sk` | 15.516 | 15.211 | +0.305 | 15.820 | no | —(rejected) |  |
| 4 | ` people` | 15.469 | 15.484 | -0.016 | 15.453 | no | —(rejected) |  |
| 5 | ` crowd` | 15.352 | 15.359 | -0.008 | 15.344 | no | —(rejected) |  |
| 6 | ` audience` | 15.086 | 15.008 | +0.078 | 15.164 | no | —(rejected) |  |
| 7 | ` person` | 15.078 | 14.930 | +0.148 | 15.227 | no | —(rejected) |  |
| 8 | ` group` | 14.961 | 14.773 | +0.188 | 15.148 | no | —(rejected) |  |
| 9 | ` on` | 14.867 | 14.836 | +0.031 | 14.898 | no | —(rejected) |  |
| 10 | ` spect` | 14.305 | 14.211 | +0.094 | 14.398 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 120

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 22.188 | 21.984 | +0.203 | 22.391 | yes | 22.391 | **YES** |
| 2 | ` sports` | 13.398 | 13.602 | -0.203 | 13.195 | no | —(rejected) |  |
| 3 | `boards` | 13.320 | 13.273 | +0.047 | 13.367 | no | —(rejected) |  |
| 4 | `-` | 12.688 | 12.812 | -0.125 | 12.562 | no | —(rejected) |  |
| 5 | `y` | 12.109 | 12.258 | -0.148 | 11.961 | no | —(rejected) |  |
| 6 | ` board` | 11.867 | 11.945 | -0.078 | 11.789 | no | —(rejected) |  |
| 7 | ` sport` | 11.719 | 11.961 | -0.242 | 11.477 | no | —(rejected) |  |
| 8 | `bound` | 11.242 | 11.328 | -0.086 | 11.156 | no | —(rejected) |  |
| 9 | `bo` | 10.742 | 10.633 | +0.109 | 10.852 | no | —(rejected) |  |
| 10 | ` enthus` | 10.641 | 10.836 | -0.195 | 10.445 | no | —(rejected) |  |


### image 2473 — word "snowboarder" (node: person) — step 121

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `er` | 24.469 | 24.344 | +0.125 | 24.594 | yes | 24.594 | **YES** |
| 2 | `ing` | 20.109 | 20.031 | +0.078 | 20.188 | no | —(rejected) |  |
| 3 | `ers` | 19.672 | 19.688 | -0.016 | 19.656 | no | —(rejected) |  |
| 4 | ` j` | 16.016 | 15.938 | +0.078 | 16.094 | no | —(rejected) |  |
| 5 | ` enthus` | 15.078 | 14.977 | +0.102 | 15.180 | no | —(rejected) |  |
| 6 | ` jump` | 14.414 | 14.289 | +0.125 | 14.539 | no | —(rejected) |  |
| 7 | `ed` | 13.594 | 13.539 | +0.055 | 13.648 | no | —(rejected) |  |
| 8 | ` r` | 13.445 | 13.539 | -0.094 | 13.352 | no | —(rejected) |  |
| 9 | ` ath` | 12.930 | 12.938 | -0.008 | 12.922 | no | —(rejected) |  |
| 10 | ` is` | 12.922 | 13.125 | -0.203 | 12.719 | no | —(rejected) |  |


### image 5529 — word "man" (node: person) — step 5

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 19.906 | 19.750 | +0.156 | 20.062 | yes | 20.062 |  |
| 2 | ` man` | 19.891 | 19.359 | +0.531 | 20.422 | yes | 20.422 | **YES** |
| 3 | ` snow` | 19.516 | 19.172 | +0.344 | 19.859 | yes | 19.859 |  |
| 4 | ` sk` | 18.234 | 17.750 | +0.484 | 18.719 | no | —(rejected) |  |
| 5 | ` ski` | 17.156 | 16.750 | +0.406 | 17.562 | no | —(rejected) |  |
| 6 | ` winter` | 17.031 | 17.000 | +0.031 | 17.062 | no | —(rejected) |  |
| 7 | ` l` | 16.703 | 16.359 | +0.344 | 17.047 | no | —(rejected) |  |
| 8 | ` male` | 16.453 | 15.914 | +0.539 | 16.992 | no | —(rejected) |  |
| 9 | ` scene` | 16.156 | 16.219 | -0.062 | 16.094 | no | —(rejected) |  |
| 10 | ` thr` | 16.000 | 14.328 | +1.672 | 17.672 | no | —(rejected) |  |


### image 5529 — word "man" (node: person) — step 51

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 20.234 | 20.328 | -0.094 | 20.141 | yes | 20.141 | **YES** |
| 2 | ` sk` | 20.031 | 20.141 | -0.109 | 19.922 | yes | 19.922 |  |
| 3 | ` snow` | 17.266 | 17.078 | +0.188 | 17.453 | no | —(rejected) |  |
| 4 | ` ski` | 17.219 | 17.328 | -0.109 | 17.109 | no | —(rejected) |  |
| 5 | ` person` | 16.547 | 16.594 | -0.047 | 16.500 | no | —(rejected) |  |
| 6 | ` scene` | 16.297 | 16.375 | -0.078 | 16.219 | no | —(rejected) |  |
| 7 | ` slope` | 16.109 | 15.797 | +0.312 | 16.422 | no | —(rejected) |  |
| 8 | ` image` | 14.477 | 14.922 | -0.445 | 14.031 | no | —(rejected) |  |
| 9 | ` mountain` | 14.469 | 14.203 | +0.266 | 14.734 | no | —(rejected) |  |
| 10 | ` main` | 14.289 | 14.492 | -0.203 | 14.086 | no | —(rejected) |  |


### image 5529 — word "skier" (node: person) — step 96

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sk` | 16.906 | 17.094 | -0.188 | 16.719 | yes | 16.719 | **YES** |
| 2 | ` man` | 16.781 | 16.906 | -0.125 | 16.656 | yes | 16.656 |  |
| 3 | ` snow` | 16.062 | 15.812 | +0.250 | 16.312 | yes | 16.312 |  |
| 4 | ` combination` | 14.531 | 14.406 | +0.125 | 14.656 | no | —(rejected) |  |
| 5 | ` ski` | 14.516 | 14.539 | -0.023 | 14.492 | no | —(rejected) |  |
| 6 | ` overall` | 14.398 | 14.328 | +0.070 | 14.469 | no | —(rejected) |  |
| 7 | ` presence` | 14.219 | 14.125 | +0.094 | 14.312 | no | —(rejected) |  |
| 8 | ` image` | 14.125 | 14.406 | -0.281 | 13.844 | no | —(rejected) |  |
| 9 | ` slope` | 14.039 | 13.781 | +0.258 | 14.297 | no | —(rejected) |  |
| 10 | ` focus` | 13.695 | 13.742 | -0.047 | 13.648 | no | —(rejected) |  |


### image 5529 — word "skier" (node: person) — step 97

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ier` | 26.047 | 26.172 | -0.125 | 25.922 | yes | 25.922 | **YES** |
| 2 | `is` | 24.266 | 24.297 | -0.031 | 24.234 | no | —(rejected) |  |
| 3 | `ies` | 20.531 | 20.453 | +0.078 | 20.609 | no | —(rejected) |  |
| 4 | `illed` | 17.031 | 16.906 | +0.125 | 17.156 | no | —(rejected) |  |
| 5 | `ied` | 16.469 | 16.438 | +0.031 | 16.500 | no | —(rejected) |  |
| 6 | `ate` | 16.312 | 16.406 | -0.094 | 16.219 | no | —(rejected) |  |
| 7 | `ater` | 16.031 | 16.188 | -0.156 | 15.875 | no | —(rejected) |  |
| 8 | `id` | 15.984 | 15.828 | +0.156 | 16.141 | no | —(rejected) |  |
| 9 | `ir` | 15.695 | 15.617 | +0.078 | 15.773 | no | —(rejected) |  |
| 10 | `ib` | 15.609 | 15.609 | +0.000 | 15.609 | no | —(rejected) |  |


### image 5529 — word "skis" (node: skis) — step 100

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sk` | 15.430 | 15.555 | -0.125 | 15.305 | yes | 15.305 | **YES** |
| 2 | ` path` | 14.727 | 14.516 | +0.211 | 14.938 | yes | 14.938 |  |
| 3 | ` ski` | 14.523 | 14.930 | -0.406 | 14.117 | yes | 14.117 |  |
| 4 | ` movements` | 14.242 | 14.062 | +0.180 | 14.422 | yes | 14.422 |  |
| 5 | ` focus` | 14.148 | 14.219 | -0.070 | 14.078 | yes | 14.078 |  |
| 6 | ` presence` | 14.039 | 14.266 | -0.227 | 13.812 | yes | 13.812 |  |
| 7 | ` movement` | 13.828 | 13.648 | +0.180 | 14.008 | yes | 14.008 |  |
| 8 | ` equipment` | 13.648 | 14.086 | -0.438 | 13.211 | no | —(rejected) |  |
| 9 | ` tracks` | 13.500 | 13.273 | +0.227 | 13.727 | no | —(rejected) |  |
| 10 | ` position` | 13.438 | 13.445 | -0.008 | 13.430 | no | —(rejected) |  |


### image 5529 — word "skis" (node: skis) — step 101

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `is` | 27.891 | 27.812 | +0.078 | 27.969 | yes | 27.969 | **YES** |
| 2 | `ies` | 19.406 | 19.250 | +0.156 | 19.562 | no | —(rejected) |  |
| 3 | `ate` | 17.484 | 17.500 | -0.016 | 17.469 | no | —(rejected) |  |
| 4 | `illed` | 17.312 | 16.828 | +0.484 | 17.797 | no | —(rejected) |  |
| 5 | `ates` | 17.047 | 17.312 | -0.266 | 16.781 | no | —(rejected) |  |
| 6 | `ier` | 16.562 | 16.375 | +0.188 | 16.750 | no | —(rejected) |  |
| 7 | `il` | 16.469 | 16.047 | +0.422 | 16.891 | no | —(rejected) |  |
| 8 | `id` | 16.438 | 16.031 | +0.406 | 16.844 | no | —(rejected) |  |
| 9 | `idd` | 15.938 | 15.500 | +0.438 | 16.375 | no | —(rejected) |  |
| 10 | `ating` | 15.867 | 16.156 | -0.289 | 15.578 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 14

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 23.562 | 23.578 | -0.016 | 23.547 | yes | 23.547 | **YES** |
| 2 | ` tour` | 19.844 | 19.312 | +0.531 | 20.375 | no | —(rejected) |  |
| 3 | ` passenger` | 17.953 | 17.547 | +0.406 | 18.359 | no | —(rejected) |  |
| 4 | ` city` | 17.578 | 17.469 | +0.109 | 17.688 | no | —(rejected) |  |
| 5 | ` double` | 16.969 | 16.797 | +0.172 | 17.141 | no | —(rejected) |  |
| 6 | ` sight` | 16.938 | 16.141 | +0.797 | 17.734 | no | —(rejected) |  |
| 7 | ` b` | 16.844 | 16.578 | +0.266 | 17.109 | no | —(rejected) |  |
| 8 | ` London` | 16.594 | 16.938 | -0.344 | 16.250 | no | —(rejected) |  |
| 9 | ` public` | 16.391 | 16.109 | +0.281 | 16.672 | no | —(rejected) |  |
| 10 | ` trans` | 15.633 | 15.609 | +0.023 | 15.656 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 15

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `es` | 26.312 | 26.438 | -0.125 | 26.188 | yes | 26.188 | **YES** |
| 2 | ` tour` | 14.656 | 14.047 | +0.609 | 15.266 | no | —(rejected) |  |
| 3 | ` drivers` | 14.023 | 13.844 | +0.180 | 14.203 | no | —(rejected) |  |
| 4 | ` stops` | 13.938 | 14.203 | -0.266 | 13.672 | no | —(rejected) |  |
| 5 | ` routes` | 13.781 | 13.281 | +0.500 | 14.281 | no | —(rejected) |  |
| 6 | ` t` | 13.719 | 13.297 | +0.422 | 14.141 | no | —(rejected) |  |
| 7 | ` lines` | 13.305 | 12.953 | +0.352 | 13.656 | no | —(rejected) |  |
| 8 | ` types` | 12.898 | 12.742 | +0.156 | 13.055 | no | —(rejected) |  |
| 9 | ` driving` | 12.891 | 13.188 | -0.297 | 12.594 | no | —(rejected) |  |
| 10 | ` cars` | 12.719 | 12.961 | -0.242 | 12.477 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 24

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 24.734 | 24.750 | -0.016 | 24.719 | yes | 24.719 | **YES** |
| 2 | ` red` | 20.453 | 20.391 | +0.062 | 20.516 | no | —(rejected) |  |
| 3 | ` double` | 20.109 | 19.969 | +0.141 | 20.250 | no | —(rejected) |  |
| 4 | ` large` | 18.422 | 18.656 | -0.234 | 18.188 | no | —(rejected) |  |
| 5 | ` larger` | 17.484 | 17.688 | -0.203 | 17.281 | no | —(rejected) |  |
| 6 | ` prominent` | 16.203 | 16.172 | +0.031 | 16.234 | no | —(rejected) |  |
| 7 | ` vehicles` | 15.914 | 16.047 | -0.133 | 15.781 | no | —(rejected) |  |
| 8 | ` big` | 15.852 | 16.109 | -0.258 | 15.594 | no | —(rejected) |  |
| 9 | ` two` | 15.836 | 15.844 | -0.008 | 15.828 | no | —(rejected) |  |
| 10 | ` v` | 15.602 | 15.953 | -0.352 | 15.250 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 25

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `es` | 26.688 | 26.781 | -0.094 | 26.594 | yes | 26.594 | **YES** |
| 2 | ` drivers` | 16.125 | 15.977 | +0.148 | 16.273 | no | —(rejected) |  |
| 3 | ` stops` | 15.047 | 15.008 | +0.039 | 15.086 | no | —(rejected) |  |
| 4 | ` routes` | 14.961 | 14.742 | +0.219 | 15.180 | no | —(rejected) |  |
| 5 | `iest` | 13.789 | 13.789 | +0.000 | 13.789 | no | —(rejected) |  |
| 6 | ` t` | 13.648 | 13.594 | +0.055 | 13.703 | no | —(rejected) |  |
| 7 | `ier` | 13.531 | 13.742 | -0.211 | 13.320 | no | —(rejected) |  |
| 8 | ` lines` | 13.523 | 13.414 | +0.109 | 13.633 | no | —(rejected) |  |
| 9 | ` front` | 13.492 | 13.633 | -0.141 | 13.352 | no | —(rejected) |  |
| 10 | ` types` | 13.344 | 13.102 | +0.242 | 13.586 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 34

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 18.016 | 18.531 | -0.516 | 17.500 | yes | 17.500 | **YES** |
| 2 | `,` | 17.766 | 18.188 | -0.422 | 17.344 | yes | 17.344 |  |
| 3 | ` tour` | 16.125 | 16.000 | +0.125 | 16.250 | no | —(rejected) |  |
| 4 | ` with` | 15.414 | 15.539 | -0.125 | 15.289 | no | —(rejected) |  |
| 5 | ` and` | 14.891 | 15.117 | -0.227 | 14.664 | no | —(rejected) |  |
| 6 | ` that` | 14.727 | 14.953 | -0.227 | 14.500 | no | —(rejected) |  |
| 7 | ` style` | 14.602 | 14.875 | -0.273 | 14.328 | no | —(rejected) |  |
| 8 | ` model` | 14.336 | 14.852 | -0.516 | 13.820 | no | —(rejected) |  |
| 9 | ` while` | 14.195 | 14.438 | -0.242 | 13.953 | no | —(rejected) |  |
| 10 | ` design` | 13.562 | 14.008 | -0.445 | 13.117 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 46

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 20.422 | 20.484 | -0.062 | 20.359 | yes | 20.359 | **YES** |
| 2 | `.` | 16.828 | 16.891 | -0.062 | 16.766 | no | —(rejected) |  |
| 3 | ` red` | 15.844 | 15.906 | -0.062 | 15.781 | no | —(rejected) |  |
| 4 | ` vehicle` | 15.789 | 16.016 | -0.227 | 15.562 | no | —(rejected) |  |
| 5 | ` tour` | 15.633 | 15.219 | +0.414 | 16.047 | no | —(rejected) |  |
| 6 | ` city` | 15.297 | 15.047 | +0.250 | 15.547 | no | —(rejected) |  |
| 7 | `,` | 15.219 | 15.250 | -0.031 | 15.188 | no | —(rejected) |  |
| 8 | ` model` | 15.055 | 15.266 | -0.211 | 14.844 | no | —(rejected) |  |
| 9 | ` design` | 15.031 | 15.172 | -0.141 | 14.891 | no | —(rejected) |  |
| 10 | ` with` | 14.953 | 14.977 | -0.023 | 14.930 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 49

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 23.125 | 23.125 | +0.000 | 23.125 | yes | 23.125 | **YES** |
| 2 | ` are` | 19.641 | 19.609 | +0.031 | 19.672 | no | —(rejected) |  |
| 3 | ` of` | 19.453 | 19.484 | -0.031 | 19.422 | no | —(rejected) |  |
| 4 | ` vehicles` | 18.516 | 18.688 | -0.172 | 18.344 | no | —(rejected) |  |
| 5 | ` appear` | 16.594 | 16.625 | -0.031 | 16.562 | no | —(rejected) |  |
| 6 | ` types` | 16.500 | 16.141 | +0.359 | 16.859 | no | —(rejected) |  |
| 7 | ` have` | 16.266 | 16.203 | +0.062 | 16.328 | no | —(rejected) |  |
| 8 | ` the` | 16.250 | 16.359 | -0.109 | 16.141 | no | —(rejected) |  |
| 9 | ` red` | 16.250 | 16.297 | -0.047 | 16.203 | no | —(rejected) |  |
| 10 | ` tour` | 15.695 | 15.258 | +0.438 | 16.133 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 50

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `es` | 26.531 | 26.641 | -0.109 | 26.422 | yes | 26.422 | **YES** |
| 2 | ` types` | 19.281 | 18.906 | +0.375 | 19.656 | no | —(rejected) |  |
| 3 | ` drivers` | 15.773 | 15.641 | +0.133 | 15.906 | no | —(rejected) |  |
| 4 | ` models` | 15.477 | 15.406 | +0.070 | 15.547 | no | —(rejected) |  |
| 5 | ` routes` | 15.234 | 15.000 | +0.234 | 15.469 | no | —(rejected) |  |
| 6 | ` lines` | 15.180 | 14.875 | +0.305 | 15.484 | no | —(rejected) |  |
| 7 | ` vehicles` | 14.891 | 15.055 | -0.164 | 14.727 | no | —(rejected) |  |
| 8 | ` sizes` | 14.391 | 14.391 | +0.000 | 14.391 | no | —(rejected) |  |
| 9 | ` services` | 13.961 | 13.766 | +0.195 | 14.156 | no | —(rejected) |  |
| 10 | ` stops` | 13.875 | 13.867 | +0.008 | 13.883 | no | —(rejected) |  |


### image 1584 — word "passenger" (node: person) — step 53

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` passengers` | 21.469 | 21.453 | +0.016 | 21.484 | yes | 21.484 | **YES** |
| 2 | ` a` | 20.109 | 19.969 | +0.141 | 20.250 | yes | 20.250 |  |
| 3 | ` numerous` | 19.312 | 19.109 | +0.203 | 19.516 | no | —(rejected) |  |
| 4 | ` many` | 19.062 | 18.953 | +0.109 | 19.172 | no | —(rejected) |  |
| 5 | ` several` | 18.703 | 18.656 | +0.047 | 18.750 | no | —(rejected) |  |
| 6 | ` multiple` | 18.156 | 18.047 | +0.109 | 18.266 | no | —(rejected) |  |
| 7 | ` people` | 18.109 | 18.062 | +0.047 | 18.156 | no | —(rejected) |  |
| 8 | ` their` | 16.859 | 17.094 | -0.234 | 16.625 | no | —(rejected) |  |
| 9 | ` large` | 16.719 | 16.406 | +0.312 | 17.031 | no | —(rejected) |  |
| 10 | ` tour` | 16.266 | 15.648 | +0.617 | 16.883 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 57

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 18.297 | 18.391 | -0.094 | 18.203 | yes | 18.203 | **YES** |
| 2 | ` of` | 17.312 | 17.297 | +0.016 | 17.328 | yes | 17.328 |  |
| 3 | ` having` | 16.312 | 16.391 | -0.078 | 16.234 | no | —(rejected) |  |
| 4 | ` being` | 15.969 | 15.969 | +0.000 | 15.969 | no | —(rejected) |  |
| 5 | ` filled` | 15.492 | 15.562 | -0.070 | 15.422 | no | —(rejected) |  |
| 6 | ` appearing` | 15.422 | 15.484 | -0.062 | 15.359 | no | —(rejected) |  |
| 7 | ` occup` | 15.406 | 15.492 | -0.086 | 15.320 | no | —(rejected) |  |
| 8 | ` carrying` | 14.742 | 14.773 | -0.031 | 14.711 | no | —(rejected) |  |
| 9 | ` more` | 14.078 | 14.148 | -0.070 | 14.008 | no | —(rejected) |  |
| 10 | ` closer` | 14.078 | 14.180 | -0.102 | 13.977 | no | —(rejected) |  |


### image 1584 — word "people" (node: person) — step 60

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 17.391 | 17.328 | +0.062 | 17.453 | yes | 17.453 | **YES** |
| 2 | ` passengers` | 17.328 | 17.438 | -0.109 | 17.219 | yes | 17.219 |  |
| 3 | ` visible` | 15.633 | 15.734 | -0.102 | 15.531 | no | —(rejected) |  |
| 4 | ` occup` | 13.633 | 13.633 | +0.000 | 13.633 | no | —(rejected) |  |
| 5 | ` rid` | 13.211 | 13.141 | +0.070 | 13.281 | no | —(rejected) |  |
| 6 | ` than` | 13.109 | 13.148 | -0.039 | 13.070 | no | —(rejected) |  |
| 7 | ` prominent` | 13.078 | 13.031 | +0.047 | 13.125 | no | —(rejected) |  |
| 8 | ` seats` | 12.703 | 12.703 | +0.000 | 12.703 | no | —(rejected) |  |
| 9 | ` cars` | 12.430 | 12.719 | -0.289 | 12.141 | no | —(rejected) |  |
| 10 | ` individuals` | 12.398 | 12.312 | +0.086 | 12.484 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 102

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 21.266 | 21.344 | -0.078 | 21.188 | yes | 21.188 | **YES** |
| 2 | ` street` | 20.906 | 20.922 | -0.016 | 20.891 | yes | 20.891 |  |
| 3 | ` area` | 19.688 | 19.594 | +0.094 | 19.781 | yes | 19.781 |  |
| 4 | ` traffic` | 19.203 | 19.000 | +0.203 | 19.406 | no | —(rejected) |  |
| 5 | ` city` | 18.844 | 18.875 | -0.031 | 18.812 | no | —(rejected) |  |
| 6 | ` scene` | 18.594 | 18.578 | +0.016 | 18.609 | no | —(rejected) |  |
| 7 | ` vehicles` | 18.312 | 18.281 | +0.031 | 18.344 | no | —(rejected) |  |
| 8 | ` road` | 18.188 | 18.016 | +0.172 | 18.359 | no | —(rejected) |  |
| 9 | ` intersection` | 17.484 | 17.422 | +0.062 | 17.547 | no | —(rejected) |  |
| 10 | ` ped` | 17.203 | 17.125 | +0.078 | 17.281 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 103

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `es` | 25.766 | 25.719 | +0.047 | 25.812 | yes | 25.812 | **YES** |
| 2 | ` is` | 19.328 | 19.703 | -0.375 | 18.953 | no | —(rejected) |  |
| 3 | ` drivers` | 18.578 | 18.359 | +0.219 | 18.797 | no | —(rejected) |  |
| 4 | ` and` | 17.656 | 17.891 | -0.234 | 17.422 | no | —(rejected) |  |
| 5 | ` traffic` | 16.500 | 16.219 | +0.281 | 16.781 | no | —(rejected) |  |
| 6 | ` are` | 16.484 | 16.297 | +0.188 | 16.672 | no | —(rejected) |  |
| 7 | ` has` | 16.047 | 16.312 | -0.266 | 15.781 | no | —(rejected) |  |
| 8 | ` route` | 16.000 | 15.875 | +0.125 | 16.125 | no | —(rejected) |  |
| 9 | ` may` | 15.398 | 15.648 | -0.250 | 15.148 | no | —(rejected) |  |
| 10 | ` must` | 15.359 | 15.547 | -0.188 | 15.172 | no | —(rejected) |  |


### image 1584 — word "people" (node: person) — step 127

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 20.844 | 20.719 | +0.125 | 20.969 | yes | 20.969 | **YES** |
| 2 | ` various` | 20.188 | 20.109 | +0.078 | 20.266 | yes | 20.266 |  |
| 3 | ` both` | 19.406 | 19.406 | +0.000 | 19.406 | yes | 19.406 |  |
| 4 | ` the` | 19.219 | 19.500 | -0.281 | 18.938 | no | —(rejected) |  |
| 5 | ` vehicles` | 19.062 | 19.062 | +0.000 | 19.062 | no | —(rejected) |  |
| 6 | ` public` | 19.031 | 18.766 | +0.266 | 19.297 | no | —(rejected) |  |
| 7 | ` multiple` | 18.734 | 18.656 | +0.078 | 18.812 | no | —(rejected) |  |
| 8 | ` bus` | 18.484 | 18.547 | -0.062 | 18.422 | no | —(rejected) |  |
| 9 | ` a` | 18.375 | 18.391 | -0.016 | 18.359 | no | —(rejected) |  |
| 10 | ` different` | 17.719 | 17.594 | +0.125 | 17.844 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 7

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 17.422 | 17.891 | -0.469 | 16.953 | yes | 16.953 | **YES** |
| 2 | ` passenger` | 16.672 | 17.328 | -0.656 | 16.016 | yes | 16.016 |  |
| 3 | ` comm` | 16.203 | 16.469 | -0.266 | 15.938 | yes | 15.938 |  |
| 4 | ` sub` | 15.039 | 15.188 | -0.148 | 14.891 | no | —(rejected) |  |
| 5 | ` electric` | 14.867 | 14.688 | +0.180 | 15.047 | no | —(rejected) |  |
| 6 | ` met` | 14.820 | 15.109 | -0.289 | 14.531 | no | —(rejected) |  |
| 7 | ` light` | 13.984 | 14.180 | -0.195 | 13.789 | no | —(rejected) |  |
| 8 | ` city` | 13.656 | 14.297 | -0.641 | 13.016 | no | —(rejected) |  |
| 9 | ` public` | 13.648 | 14.289 | -0.641 | 13.008 | no | —(rejected) |  |
| 10 | ` tram` | 13.016 | 13.250 | -0.234 | 12.781 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 24

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 20.391 | 20.672 | -0.281 | 20.109 | yes | 20.109 | **YES** |
| 2 | ` passenger` | 14.641 | 14.930 | -0.289 | 14.352 | no | —(rejected) |  |
| 3 | ` large` | 14.406 | 14.984 | -0.578 | 13.828 | no | —(rejected) |  |
| 4 | ` long` | 14.336 | 15.336 | -1.000 | 13.336 | no | —(rejected) |  |
| 5 | ` sub` | 13.727 | 13.703 | +0.023 | 13.750 | no | —(rejected) |  |
| 6 | ` electric` | 13.367 | 13.031 | +0.336 | 13.703 | no | —(rejected) |  |
| 7 | ` light` | 13.359 | 13.484 | -0.125 | 13.234 | no | —(rejected) |  |
| 8 | ` view` | 13.344 | 13.531 | -0.188 | 13.156 | no | —(rejected) |  |
| 9 | ` front` | 13.344 | 13.344 | +0.000 | 13.344 | no | —(rejected) |  |
| 10 | ` comm` | 13.336 | 13.531 | -0.195 | 13.141 | no | —(rejected) |  |


### image 6040 — word "passenger" (node: person) — step 28

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` passengers` | 20.594 | 20.047 | +0.547 | 21.141 | yes | 21.141 | **YES** |
| 2 | ` people` | 18.203 | 17.844 | +0.359 | 18.562 | no | —(rejected) |  |
| 3 | ` commut` | 15.773 | 14.984 | +0.789 | 16.562 | no | —(rejected) |  |
| 4 | ` c` | 14.797 | 15.203 | -0.406 | 14.391 | no | —(rejected) |  |
| 5 | ` passenger` | 14.539 | 14.336 | +0.203 | 14.742 | no | —(rejected) |  |
| 6 | ` rid` | 14.078 | 13.750 | +0.328 | 14.406 | no | —(rejected) |  |
| 7 | ` cars` | 13.977 | 13.922 | +0.055 | 14.031 | no | —(rejected) |  |
| 8 | ` comm` | 13.016 | 12.297 | +0.719 | 13.734 | no | —(rejected) |  |
| 9 | ` train` | 12.805 | 12.625 | +0.180 | 12.984 | no | —(rejected) |  |
| 10 | ` individuals` | 12.766 | 12.477 | +0.289 | 13.055 | no | —(rejected) |  |


### image 6040 — word "people" (node: person) — step 36

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 22.500 | 21.625 | +0.875 | 23.375 | yes | 23.375 | **YES** |
| 2 | ` visible` | 20.438 | 19.812 | +0.625 | 21.062 | no | —(rejected) |  |
| 3 | ` individuals` | 19.781 | 18.906 | +0.875 | 20.656 | no | —(rejected) |  |
| 4 | ` passengers` | 17.734 | 17.281 | +0.453 | 18.188 | no | —(rejected) |  |
| 5 | ` persons` | 15.297 | 14.648 | +0.648 | 15.945 | no | —(rejected) |  |
| 6 | ` distinct` | 14.969 | 14.391 | +0.578 | 15.547 | no | —(rejected) |  |
| 7 | ` different` | 14.836 | 14.094 | +0.742 | 15.578 | no | —(rejected) |  |
| 8 | ` rid` | 14.594 | 14.312 | +0.281 | 14.875 | no | —(rejected) |  |
| 9 | ` individual` | 14.477 | 13.984 | +0.492 | 14.969 | no | —(rejected) |  |
| 10 | ` of` | 14.430 | 13.977 | +0.453 | 14.883 | no | —(rejected) |  |


### image 6040 — word "passenger" (node: person) — step 45

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` passengers` | 19.844 | 19.578 | +0.266 | 20.109 | yes | 20.109 | **YES** |
| 2 | ` people` | 16.859 | 16.547 | +0.312 | 17.172 | no | —(rejected) |  |
| 3 | ` individuals` | 16.281 | 15.945 | +0.336 | 16.617 | no | —(rejected) |  |
| 4 | ` windows` | 15.078 | 15.617 | -0.539 | 14.539 | no | —(rejected) |  |
| 5 | ` rid` | 14.688 | 14.539 | +0.148 | 14.836 | no | —(rejected) |  |
| 6 | ` train` | 14.273 | 14.406 | -0.133 | 14.141 | no | —(rejected) |  |
| 7 | ` commut` | 13.883 | 13.445 | +0.438 | 14.320 | no | —(rejected) |  |
| 8 | ` travel` | 13.328 | 12.953 | +0.375 | 13.703 | no | —(rejected) |  |
| 9 | ` seats` | 13.133 | 13.234 | -0.102 | 13.031 | no | —(rejected) |  |
| 10 | ` occup` | 12.539 | 12.359 | +0.180 | 12.719 | no | —(rejected) |  |


### image 6040 — word "car" (node: car) — step 71

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` cars` | 22.750 | 18.266 | +4.484 | 27.234 | yes | 27.234 | **YES** |
| 2 | ` vehicles` | 20.656 | 16.047 | +4.609 | 25.266 | no | —(rejected) |  |
| 3 | ` park` | 19.312 | 15.305 | +4.008 | 23.320 | no | —(rejected) |  |
| 4 | ` other` | 17.672 | 16.875 | +0.797 | 18.469 | no | —(rejected) |  |
| 5 | ` people` | 17.141 | 17.312 | -0.172 | 16.969 | no | —(rejected) |  |
| 6 | ` objects` | 17.062 | 15.812 | +1.250 | 18.312 | no | —(rejected) |  |
| 7 | ` personal` | 16.938 | 13.211 | +3.727 | 20.664 | no | —(rejected) |  |
| 8 | ` small` | 16.562 | 14.711 | +1.852 | 18.414 | no | —(rejected) |  |
| 9 | ` autom` | 15.875 | 10.938 | +4.938 | 20.812 | no | —(rejected) |  |
| 10 | ` ben` | 15.875 | 16.812 | -0.938 | 14.938 | no | —(rejected) |  |


### image 6040 — word "car" (node: car) — step 80

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` car` | 21.094 | 21.000 | +0.094 | 21.188 | yes | 21.188 | **YES** |
| 2 | ` on` | 18.500 | 18.672 | -0.172 | 18.328 | no | —(rejected) |  |
| 3 | ` close` | 17.656 | 17.609 | +0.047 | 17.703 | no | —(rejected) |  |
| 4 | ` position` | 17.609 | 17.734 | -0.125 | 17.484 | no | —(rejected) |  |
| 5 | ` located` | 17.562 | 17.766 | -0.203 | 17.359 | no | —(rejected) |  |
| 6 | ` closer` | 17.406 | 17.391 | +0.016 | 17.422 | no | —(rejected) |  |
| 7 | ` visible` | 16.844 | 16.812 | +0.031 | 16.875 | no | —(rejected) |  |
| 8 | ` near` | 16.812 | 16.734 | +0.078 | 16.891 | no | —(rejected) |  |
| 9 | ` vehicle` | 16.750 | 16.750 | +0.000 | 16.750 | no | —(rejected) |  |
| 10 | ` in` | 16.609 | 16.734 | -0.125 | 16.484 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 88

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 21.656 | 21.703 | -0.047 | 21.609 | yes | 21.609 | **YES** |
| 2 | ` image` | 20.375 | 20.516 | -0.141 | 20.234 | yes | 20.234 |  |
| 3 | ` scene` | 19.594 | 19.609 | -0.016 | 19.578 | no | —(rejected) |  |
| 4 | ` tracks` | 18.500 | 18.250 | +0.250 | 18.750 | no | —(rejected) |  |
| 5 | ` track` | 17.812 | 17.703 | +0.109 | 17.922 | no | —(rejected) |  |
| 6 | ` frame` | 17.250 | 17.328 | -0.078 | 17.172 | no | —(rejected) |  |
| 7 | ` picture` | 16.172 | 16.344 | -0.172 | 16.000 | no | —(rejected) |  |
| 8 | ` street` | 16.094 | 16.422 | -0.328 | 15.766 | no | —(rejected) |  |
| 9 | ` rail` | 15.898 | 15.914 | -0.016 | 15.883 | no | —(rejected) |  |
| 10 | ` road` | 14.930 | 15.414 | -0.484 | 14.445 | no | —(rejected) |  |


### image 6040 — word "car" (node: car) — step 91

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` car` | 19.953 | 19.891 | +0.062 | 20.016 | yes | 20.016 | **YES** |
| 2 | ` on` | 19.891 | 19.859 | +0.031 | 19.922 | yes | 19.922 |  |
| 3 | ` further` | 19.203 | 19.094 | +0.109 | 19.312 | yes | 19.312 |  |
| 4 | ` one` | 18.938 | 18.906 | +0.031 | 18.969 | yes | 18.969 |  |
| 5 | ` closer` | 18.312 | 18.172 | +0.141 | 18.453 | no | —(rejected) |  |
| 6 | ` two` | 18.156 | 17.938 | +0.219 | 18.375 | no | —(rejected) |  |
| 7 | ` towards` | 17.891 | 17.781 | +0.109 | 18.000 | no | —(rejected) |  |
| 8 | ` slightly` | 17.406 | 17.453 | -0.047 | 17.359 | no | —(rejected) |  |
| 9 | ` in` | 17.281 | 17.156 | +0.125 | 17.406 | no | —(rejected) |  |
| 10 | ` park` | 17.016 | 17.156 | -0.141 | 16.875 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 110

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 22.562 | 22.266 | +0.297 | 22.859 | yes | 22.859 | **YES** |
| 2 | ` rail` | 20.047 | 19.828 | +0.219 | 20.266 | no | —(rejected) |  |
| 3 | ` line` | 18.188 | 18.141 | +0.047 | 18.234 | no | —(rejected) |  |
| 4 | ` or` | 18.000 | 17.766 | +0.234 | 18.234 | no | —(rejected) |  |
| 5 | ` railway` | 17.656 | 17.547 | +0.109 | 17.766 | no | —(rejected) |  |
| 6 | ` route` | 16.125 | 16.391 | -0.266 | 15.859 | no | —(rejected) |  |
| 7 | `'` | 15.445 | 15.359 | +0.086 | 15.531 | no | —(rejected) |  |
| 8 | `-` | 15.070 | 14.953 | +0.117 | 15.188 | no | —(rejected) |  |
| 9 | ` trans` | 15.000 | 15.016 | -0.016 | 14.984 | no | —(rejected) |  |
| 10 | ` transport` | 14.719 | 14.547 | +0.172 | 14.891 | no | —(rejected) |  |


### image 6040 — word "people" (node: person) — step 113

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 21.000 | 20.875 | +0.125 | 21.125 | yes | 21.125 | **YES** |
| 2 | ` passengers` | 20.391 | 20.406 | -0.016 | 20.375 | yes | 20.375 |  |
| 3 | ` many` | 17.984 | 18.219 | -0.234 | 17.750 | no | —(rejected) |  |
| 4 | ` a` | 17.891 | 17.984 | -0.094 | 17.797 | no | —(rejected) |  |
| 5 | ` the` | 17.453 | 17.500 | -0.047 | 17.406 | no | —(rejected) |  |
| 6 | ` travel` | 17.172 | 17.031 | +0.141 | 17.312 | no | —(rejected) |  |
| 7 | ` numerous` | 16.812 | 17.000 | -0.188 | 16.625 | no | —(rejected) |  |
| 8 | ` various` | 16.625 | 16.641 | -0.016 | 16.609 | no | —(rejected) |  |
| 9 | ` both` | 16.484 | 16.516 | -0.031 | 16.453 | no | —(rejected) |  |
| 10 | ` individuals` | 16.375 | 16.250 | +0.125 | 16.500 | no | —(rejected) |  |


### image 6763 — word "man" (node: person) — step 4

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 21.328 | 21.250 | +0.078 | 21.406 | yes | 21.406 | **YES** |
| 2 | ` couple` | 18.094 | 17.766 | +0.328 | 18.422 | no | —(rejected) |  |
| 3 | ` young` | 18.000 | 18.500 | -0.500 | 17.500 | no | —(rejected) |  |
| 4 | ` happy` | 17.672 | 17.141 | +0.531 | 18.203 | no | —(rejected) |  |
| 5 | ` sm` | 17.516 | 17.188 | +0.328 | 17.844 | no | —(rejected) |  |
| 6 | ` woman` | 15.250 | 15.570 | -0.320 | 14.930 | no | —(rejected) |  |
| 7 | ` well` | 15.000 | 14.891 | +0.109 | 15.109 | no | —(rejected) |  |
| 8 | ` beautiful` | 14.906 | 14.984 | -0.078 | 14.828 | no | —(rejected) |  |
| 9 | ` middle` | 14.805 | 14.438 | +0.367 | 15.172 | no | —(rejected) |  |
| 10 | ` hand` | 14.625 | 14.680 | -0.055 | 14.570 | no | —(rejected) |  |


### image 6763 — word "woman" (node: person) — step 7

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` woman` | 25.938 | 25.484 | +0.453 | 26.391 | yes | 26.391 | **YES** |
| 2 | ` beautiful` | 21.125 | 20.219 | +0.906 | 22.031 | no | —(rejected) |  |
| 3 | ` sm` | 20.453 | 19.859 | +0.594 | 21.047 | no | —(rejected) |  |
| 4 | ` young` | 19.812 | 21.438 | -1.625 | 18.188 | no | —(rejected) |  |
| 5 | ` lady` | 19.531 | 19.422 | +0.109 | 19.641 | no | —(rejected) |  |
| 6 | ` pretty` | 18.578 | 18.438 | +0.141 | 18.719 | no | —(rejected) |  |
| 7 | ` girl` | 18.562 | 20.469 | -1.906 | 16.656 | no | —(rejected) |  |
| 8 | ` women` | 18.484 | 18.031 | +0.453 | 18.938 | no | —(rejected) |  |
| 9 | ` pre` | 17.000 | 17.219 | -0.219 | 16.781 | no | —(rejected) |  |
| 10 | ` female` | 16.406 | 16.891 | -0.484 | 15.922 | no | —(rejected) |  |


### image 6763 — word "man" (node: person) — step 21

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 23.359 | 23.172 | +0.188 | 23.547 | yes | 23.547 | **YES** |
| 2 | ` woman` | 21.797 | 21.719 | +0.078 | 21.875 | yes | 21.875 |  |
| 3 | ` couple` | 19.500 | 19.547 | -0.047 | 19.453 | no | —(rejected) |  |
| 4 | ` young` | 18.531 | 18.969 | -0.438 | 18.094 | no | —(rejected) |  |
| 5 | ` sm` | 17.812 | 17.031 | +0.781 | 18.594 | no | —(rejected) |  |
| 6 | ` lady` | 17.406 | 17.344 | +0.062 | 17.469 | no | —(rejected) |  |
| 7 | ` two` | 16.656 | 17.500 | -0.844 | 15.812 | no | —(rejected) |  |
| 8 | ` gentleman` | 16.484 | 16.188 | +0.297 | 16.781 | no | —(rejected) |  |
| 9 | ` pair` | 16.469 | 16.922 | -0.453 | 16.016 | no | —(rejected) |  |
| 10 | ` happy` | 16.219 | 15.562 | +0.656 | 16.875 | no | —(rejected) |  |


### image 6763 — word "tie" (node: tie) — step 26

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` tie` | 17.141 | 17.797 | -0.656 | 16.484 | yes | 16.484 | **YES** |
| 2 | ` blue` | 16.000 | 16.953 | -0.953 | 15.047 | yes | 15.047 |  |
| 3 | ` pur` | 15.469 | 15.719 | -0.250 | 15.219 | no | —(rejected) |  |
| 4 | ` strip` | 15.109 | 13.023 | +2.086 | 17.195 | no | —(rejected) |  |
| 5 | ` sh` | 15.055 | 15.664 | -0.609 | 14.445 | no | —(rejected) |  |
| 6 | ` neck` | 14.578 | 15.648 | -1.070 | 13.508 | no | —(rejected) |  |
| 7 | ` suit` | 14.016 | 14.203 | -0.188 | 13.828 | no | —(rejected) |  |
| 8 | ` dress` | 13.484 | 14.195 | -0.711 | 12.773 | no | —(rejected) |  |
| 9 | ` p` | 13.438 | 12.820 | +0.617 | 14.055 | no | —(rejected) |  |
| 10 | ` color` | 13.273 | 13.250 | +0.023 | 13.297 | no | —(rejected) |  |


### image 6763 — word "woman" (node: person) — step 30

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` woman` | 22.375 | 22.062 | +0.312 | 22.688 | yes | 22.688 | **YES** |
| 2 | ` two` | 18.797 | 19.047 | -0.250 | 18.547 | no | —(rejected) |  |
| 3 | ` couple` | 18.656 | 18.375 | +0.281 | 18.938 | no | —(rejected) |  |
| 4 | ` pair` | 17.625 | 17.719 | -0.094 | 17.531 | no | —(rejected) |  |
| 5 | ` lady` | 17.609 | 17.156 | +0.453 | 18.062 | no | —(rejected) |  |
| 6 | ` young` | 16.406 | 17.125 | -0.719 | 15.688 | no | —(rejected) |  |
| 7 | ` both` | 16.109 | 15.977 | +0.133 | 16.242 | no | —(rejected) |  |
| 8 | ` tie` | 15.922 | 15.672 | +0.250 | 16.172 | no | —(rejected) |  |
| 9 | ` girl` | 15.836 | 16.531 | -0.695 | 15.141 | no | —(rejected) |  |
| 10 | ` sm` | 15.812 | 15.109 | +0.703 | 16.516 | no | —(rejected) |  |


### image 6763 — word "tv" (node: tv) — step 63

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` TV` | 20.984 | 20.891 | +0.094 | 21.078 | yes | 21.078 | **YES** |
| 2 | ` television` | 20.016 | 19.953 | +0.062 | 20.078 | yes | 20.078 |  |
| 3 | ` d` | 19.469 | 18.609 | +0.859 | 20.328 | yes | 20.328 |  |
| 4 | ` flat` | 17.547 | 17.062 | +0.484 | 18.031 | no | —(rejected) |  |
| 5 | ` large` | 16.844 | 17.016 | -0.172 | 16.672 | no | —(rejected) |  |
| 6 | ` small` | 16.688 | 16.062 | +0.625 | 17.312 | no | —(rejected) |  |
| 7 | ` person` | 16.672 | 15.766 | +0.906 | 17.578 | no | —(rejected) |  |
| 8 | ` c` | 16.656 | 16.828 | -0.172 | 16.484 | no | —(rejected) |  |
| 9 | ` bar` | 16.125 | 14.977 | +1.148 | 17.273 | no | —(rejected) |  |
| 10 | ` group` | 16.062 | 15.648 | +0.414 | 16.477 | no | —(rejected) |  |


### image 6763 — word "cell phone" (node: cell phone) — step 117

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 18.219 | 17.703 | +0.516 | 18.734 | yes | 18.734 |  |
| 2 | ` hand` | 17.984 | 17.891 | +0.094 | 18.078 | yes | 18.078 |  |
| 3 | ` cell` | 17.609 | 16.234 | +1.375 | 18.984 | yes | 18.984 | **YES** |
| 4 | ` clock` | 17.594 | 17.844 | -0.250 | 17.344 | yes | 17.344 |  |
| 5 | ` d` | 17.281 | 16.203 | +1.078 | 18.359 | yes | 18.359 |  |
| 6 | ` book` | 16.938 | 16.781 | +0.156 | 17.094 | yes | 17.094 |  |
| 7 | ` cup` | 16.656 | 16.281 | +0.375 | 17.031 | yes | 17.031 |  |
| 8 | ` bow` | 16.422 | 15.867 | +0.555 | 16.977 | no | —(rejected) |  |
| 9 | ` remote` | 16.391 | 15.539 | +0.852 | 17.242 | no | —(rejected) |  |
| 10 | ` tie` | 16.156 | 16.375 | -0.219 | 15.938 | no | —(rejected) |  |


### image 6763 — word "cell phone" (node: cell phone) — step 118

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` phone` | 24.969 | 25.391 | -0.422 | 24.547 | yes | 24.547 | **YES** |
| 2 | `phone` | 20.984 | 21.219 | -0.234 | 20.750 | no | —(rejected) |  |
| 3 | `ular` | 17.391 | 17.578 | -0.188 | 17.203 | no | —(rejected) |  |
| 4 | `-` | 13.766 | 13.875 | -0.109 | 13.656 | no | —(rejected) |  |
| 5 | ` Phone` | 13.523 | 13.859 | -0.336 | 13.188 | no | —(rejected) |  |
| 6 | ` ph` | 12.945 | 13.305 | -0.359 | 12.586 | no | —(rejected) |  |
| 7 | ` device` | 12.859 | 12.594 | +0.266 | 13.125 | no | —(rejected) |  |
| 8 | ` or` | 12.703 | 12.758 | -0.055 | 12.648 | no | —(rejected) |  |
| 9 | ` tele` | 12.414 | 12.445 | -0.031 | 12.383 | no | —(rejected) |  |
| 10 | ` is` | 12.391 | 12.414 | -0.023 | 12.367 | no | —(rejected) |  |


### image 2261 — word "man" (node: person) — step 6

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 23.109 | 23.359 | -0.250 | 22.859 | yes | 22.859 | **YES** |
| 2 | ` boy` | 21.312 | 19.891 | +1.422 | 22.734 | no | —(rejected) |  |
| 3 | ` male` | 19.797 | 19.609 | +0.188 | 19.984 | no | —(rejected) |  |
| 4 | ` person` | 19.766 | 20.359 | -0.594 | 19.172 | no | —(rejected) |  |
| 5 | ` sur` | 19.688 | 20.266 | -0.578 | 19.109 | no | —(rejected) |  |
| 6 | ` individual` | 17.453 | 17.969 | -0.516 | 16.938 | no | —(rejected) |  |
| 7 | `,` | 17.359 | 17.391 | -0.031 | 17.328 | no | —(rejected) |  |
| 8 | ` child` | 17.250 | 16.297 | +0.953 | 18.203 | no | —(rejected) |  |
| 9 | ` adult` | 17.234 | 18.125 | -0.891 | 16.344 | no | —(rejected) |  |
| 10 | ` te` | 16.000 | 15.602 | +0.398 | 16.398 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 13

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sur` | 21.406 | 22.609 | -1.203 | 20.203 | yes | 20.203 | **YES** |
| 2 | ` blue` | 19.422 | 20.109 | -0.688 | 18.734 | no | —(rejected) |  |
| 3 | ` bo` | 18.812 | 19.031 | -0.219 | 18.594 | no | —(rejected) |  |
| 4 | ` body` | 18.469 | 18.391 | +0.078 | 18.547 | no | —(rejected) |  |
| 5 | ` small` | 17.250 | 18.172 | -0.922 | 16.328 | no | —(rejected) |  |
| 6 | ` board` | 16.641 | 17.172 | -0.531 | 16.109 | no | —(rejected) |  |
| 7 | ` wave` | 16.078 | 17.203 | -1.125 | 14.953 | no | —(rejected) |  |
| 8 | ` large` | 16.047 | 16.281 | -0.234 | 15.812 | no | —(rejected) |  |
| 9 | ` white` | 15.617 | 17.297 | -1.680 | 13.938 | no | —(rejected) |  |
| 10 | ` bright` | 15.461 | 16.094 | -0.633 | 14.828 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 14

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `f` | 27.203 | 27.516 | -0.312 | 26.891 | yes | 26.891 | **YES** |
| 2 | `board` | 17.625 | 17.672 | -0.047 | 17.578 | no | —(rejected) |  |
| 3 | `fer` | 16.938 | 16.906 | +0.031 | 16.969 | no | —(rejected) |  |
| 4 | `ft` | 14.000 | 14.172 | -0.172 | 13.828 | no | —(rejected) |  |
| 5 | `fers` | 13.500 | 13.562 | -0.062 | 13.438 | no | —(rejected) |  |
| 6 | `fo` | 12.867 | 12.773 | +0.094 | 12.961 | no | —(rejected) |  |
| 7 | `face` | 12.406 | 12.180 | +0.227 | 12.633 | no | —(rejected) |  |
| 8 | `fc` | 12.250 | 12.203 | +0.047 | 12.297 | no | —(rejected) |  |
| 9 | `fac` | 11.930 | 11.883 | +0.047 | 11.977 | no | —(rejected) |  |
| 10 | `fin` | 11.727 | 11.789 | -0.062 | 11.664 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 15

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 25.703 | 25.703 | +0.000 | 25.703 | yes | 25.703 | **YES** |
| 2 | ` board` | 17.656 | 17.266 | +0.391 | 18.047 | no | —(rejected) |  |
| 3 | `ing` | 16.141 | 16.109 | +0.031 | 16.172 | no | —(rejected) |  |
| 4 | `boards` | 14.977 | 14.805 | +0.172 | 15.148 | no | —(rejected) |  |
| 5 | `ba` | 13.773 | 13.438 | +0.336 | 14.109 | no | —(rejected) |  |
| 6 | `bo` | 13.039 | 12.453 | +0.586 | 13.625 | no | —(rejected) |  |
| 7 | `-` | 12.867 | 12.430 | +0.438 | 13.305 | no | —(rejected) |  |
| 8 | ` sur` | 12.742 | 12.406 | +0.336 | 13.078 | no | —(rejected) |  |
| 9 | `able` | 12.711 | 12.320 | +0.391 | 13.102 | no | —(rejected) |  |
| 10 | ` mat` | 12.039 | 10.695 | +1.344 | 13.383 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 28

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sur` | 20.328 | 20.438 | -0.109 | 20.219 | yes | 20.219 | **YES** |
| 2 | ` board` | 19.000 | 19.078 | -0.078 | 18.922 | yes | 18.922 |  |
| 3 | ` blue` | 17.547 | 17.000 | +0.547 | 18.094 | no | —(rejected) |  |
| 4 | ` wave` | 17.297 | 17.781 | -0.484 | 16.812 | no | —(rejected) |  |
| 5 | ` water` | 16.141 | 16.375 | -0.234 | 15.906 | no | —(rejected) |  |
| 6 | ` white` | 15.727 | 15.945 | -0.219 | 15.508 | no | —(rejected) |  |
| 7 | ` small` | 15.539 | 15.695 | -0.156 | 15.383 | no | —(rejected) |  |
| 8 | ` bo` | 15.148 | 14.930 | +0.219 | 15.367 | no | —(rejected) |  |
| 9 | ` top` | 14.898 | 14.852 | +0.047 | 14.945 | no | —(rejected) |  |
| 10 | ` large` | 14.570 | 14.461 | +0.109 | 14.680 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 29

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `f` | 26.141 | 26.203 | -0.062 | 26.078 | yes | 26.078 | **YES** |
| 2 | `board` | 16.875 | 16.812 | +0.062 | 16.938 | no | —(rejected) |  |
| 3 | `fer` | 16.688 | 16.656 | +0.031 | 16.719 | no | —(rejected) |  |
| 4 | `ft` | 13.742 | 13.742 | +0.000 | 13.742 | no | —(rejected) |  |
| 5 | `fo` | 13.602 | 13.531 | +0.070 | 13.672 | no | —(rejected) |  |
| 6 | `face` | 13.195 | 13.234 | -0.039 | 13.156 | no | —(rejected) |  |
| 7 | `fc` | 13.180 | 13.070 | +0.109 | 13.289 | no | —(rejected) |  |
| 8 | `ge` | 12.961 | 12.938 | +0.023 | 12.984 | no | —(rejected) |  |
| 9 | `fers` | 12.953 | 12.945 | +0.008 | 12.961 | no | —(rejected) |  |
| 10 | `fac` | 12.602 | 12.656 | -0.055 | 12.547 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 30

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 25.234 | 25.344 | -0.109 | 25.125 | yes | 25.125 | **YES** |
| 2 | `ing` | 16.172 | 16.156 | +0.016 | 16.188 | no | —(rejected) |  |
| 3 | ` board` | 16.031 | 15.930 | +0.102 | 16.133 | no | —(rejected) |  |
| 4 | `boards` | 14.758 | 14.844 | -0.086 | 14.672 | no | —(rejected) |  |
| 5 | `,` | 13.859 | 13.984 | -0.125 | 13.734 | no | —(rejected) |  |
| 6 | `acing` | 13.391 | 13.422 | -0.031 | 13.359 | no | —(rejected) |  |
| 7 | `ba` | 13.094 | 12.945 | +0.148 | 13.242 | no | —(rejected) |  |
| 8 | ` as` | 12.797 | 12.898 | -0.102 | 12.695 | no | —(rejected) |  |
| 9 | ` and` | 12.680 | 12.750 | -0.070 | 12.609 | no | —(rejected) |  |
| 10 | `bo` | 12.516 | 12.414 | +0.102 | 12.617 | no | —(rejected) |  |


### image 1425 — word "table" (node: dining table) — step 35

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` table` | 27.672 | 27.344 | +0.328 | 28.000 | yes | 28.000 | **YES** |
| 2 | ` surface` | 21.578 | 21.609 | -0.031 | 21.547 | no | —(rejected) |  |
| 3 | ` room` | 18.984 | 18.797 | +0.188 | 19.172 | no | —(rejected) |  |
| 4 | ` or` | 18.969 | 18.719 | +0.250 | 19.219 | no | —(rejected) |  |
| 5 | ` area` | 18.172 | 18.156 | +0.016 | 18.188 | no | —(rejected) |  |
| 6 | ` counter` | 17.703 | 17.625 | +0.078 | 17.781 | no | —(rejected) |  |
| 7 | ` setting` | 17.203 | 16.938 | +0.266 | 17.469 | no | —(rejected) |  |
| 8 | ` des` | 16.938 | 16.969 | -0.031 | 16.906 | no | —(rejected) |  |
| 9 | ` set` | 15.984 | 15.883 | +0.102 | 16.086 | no | —(rejected) |  |
| 10 | ` top` | 15.969 | 15.906 | +0.062 | 16.031 | no | —(rejected) |  |


### image 1425 — word "bowl" (node: bowl) — step 45

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bow` | 17.609 | 15.992 | +1.617 | 19.227 | yes | 19.227 | **YES** |
| 2 | ` cup` | 17.016 | 16.812 | +0.203 | 17.219 | yes | 17.219 |  |
| 3 | ` small` | 16.594 | 14.062 | +2.531 | 19.125 | yes | 19.125 |  |
| 4 | ` white` | 15.805 | 14.570 | +1.234 | 17.039 | no | —(rejected) |  |
| 5 | ` container` | 14.656 | 11.984 | +2.672 | 17.328 | no | —(rejected) |  |
| 6 | ` sau` | 14.586 | 12.844 | +1.742 | 16.328 | no | —(rejected) |  |
| 7 | ` glass` | 14.578 | 12.578 | +2.000 | 16.578 | no | —(rejected) |  |
| 8 | ` sp` | 14.305 | 14.922 | -0.617 | 13.688 | no | —(rejected) |  |
| 9 | ` jar` | 13.906 | 10.203 | +3.703 | 17.609 | no | —(rejected) |  |
| 10 | ` smaller` | 13.742 | 13.562 | +0.180 | 13.922 | no | —(rejected) |  |


### image 1425 — word "bowl" (node: bowl) — step 46

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `l` | 26.000 | 25.125 | +0.875 | 26.875 | yes | 26.875 | **YES** |
| 2 | `el` | 14.789 | 14.812 | -0.023 | 14.766 | no | —(rejected) |  |
| 3 | `ls` | 13.102 | 12.562 | +0.539 | 13.641 | no | —(rejected) |  |
| 4 | `,` | 12.625 | 13.242 | -0.617 | 12.008 | no | —(rejected) |  |
| 5 | ` of` | 12.148 | 12.273 | -0.125 | 12.023 | no | —(rejected) |  |
| 6 | `led` | 11.648 | 12.859 | -1.211 | 10.438 | no | —(rejected) |  |
| 7 | `ler` | 11.570 | 12.625 | -1.055 | 10.516 | no | —(rejected) |  |
| 8 | `ling` | 11.203 | 11.977 | -0.773 | 10.430 | no | —(rejected) |  |
| 9 | ` or` | 11.086 | 11.156 | -0.070 | 11.016 | no | —(rejected) |  |
| 10 | ` and` | 10.836 | 12.156 | -1.320 | 9.516 | no | —(rejected) |  |


### image 1425 — word "table" (node: dining table) — step 73

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` table` | 24.422 | 24.219 | +0.203 | 24.625 | yes | 24.625 | **YES** |
| 2 | ` left` | 21.922 | 21.266 | +0.656 | 22.578 | no | —(rejected) |  |
| 3 | ` d` | 20.766 | 20.688 | +0.078 | 20.844 | no | —(rejected) |  |
| 4 | ` right` | 20.516 | 21.438 | -0.922 | 19.594 | no | —(rejected) |  |
| 5 | ` plate` | 19.953 | 20.359 | -0.406 | 19.547 | no | —(rejected) |  |
| 6 | ` side` | 19.484 | 19.516 | -0.031 | 19.453 | no | —(rejected) |  |
| 7 | ` edge` | 19.219 | 19.250 | -0.031 | 19.188 | no | —(rejected) |  |
| 8 | ` top` | 18.844 | 17.875 | +0.969 | 19.812 | no | —(rejected) |  |
| 9 | ` far` | 18.625 | 18.734 | -0.109 | 18.516 | no | —(rejected) |  |
| 10 | ` upper` | 18.594 | 17.453 | +1.141 | 19.734 | no | —(rejected) |  |


---
# SID


## SID — HALLUCINATED OBJECTS


### image 724 — word "people" (node: person) — step 86

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 20.844 | 21.375 | -0.531 | 20.312 | yes | 20.312 | **YES** |
| 2 | ` ped` | 20.594 | 20.969 | -0.375 | 20.219 | yes | 20.219 |  |
| 3 | ` other` | 19.703 | 19.562 | +0.141 | 19.844 | yes | 19.844 |  |
| 4 | ` more` | 19.062 | 19.156 | -0.094 | 18.969 | no | —(rejected) |  |
| 5 | ` traffic` | 18.125 | 17.250 | +0.875 | 19.000 | no | —(rejected) |  |
| 6 | ` ben` | 16.688 | 15.594 | +1.094 | 17.781 | no | —(rejected) |  |
| 7 | ` individuals` | 16.312 | 16.594 | -0.281 | 16.031 | no | —(rejected) |  |
| 8 | ` smaller` | 16.297 | 15.945 | +0.352 | 16.648 | no | —(rejected) |  |
| 9 | ` birds` | 16.266 | 15.719 | +0.547 | 16.812 | no | —(rejected) |  |
| 10 | ` objects` | 15.664 | 16.312 | -0.648 | 15.016 | no | —(rejected) |  |


### image 724 — word "person" (node: person) — step 94

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 23.328 | 23.719 | -0.391 | 22.938 | yes | 22.938 | **YES** |
| 2 | ` standing` | 20.203 | 20.406 | -0.203 | 20.000 | no | —(rejected) |  |
| 3 | ` individual` | 19.188 | 19.547 | -0.359 | 18.828 | no | —(rejected) |  |
| 4 | ` near` | 19.156 | 19.078 | +0.078 | 19.234 | no | —(rejected) |  |
| 5 | ` close` | 18.797 | 18.906 | -0.109 | 18.688 | no | —(rejected) |  |
| 6 | ` of` | 18.672 | 18.734 | -0.062 | 18.609 | no | —(rejected) |  |
| 7 | ` on` | 18.641 | 18.641 | +0.000 | 18.641 | no | —(rejected) |  |
| 8 | ` located` | 18.594 | 18.625 | -0.031 | 18.562 | no | —(rejected) |  |
| 9 | ` closer` | 17.688 | 17.719 | -0.031 | 17.656 | no | —(rejected) |  |
| 10 | ` position` | 17.516 | 17.578 | -0.062 | 17.453 | no | —(rejected) |  |


### image 724 — word "person" (node: person) — step 102

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 21.594 | 21.797 | -0.203 | 21.391 | yes | 21.391 | **YES** |
| 2 | ` closer` | 19.312 | 19.547 | -0.234 | 19.078 | no | —(rejected) |  |
| 3 | ` near` | 19.141 | 19.234 | -0.094 | 19.047 | no | —(rejected) |  |
| 4 | ` one` | 18.922 | 19.078 | -0.156 | 18.766 | no | —(rejected) |  |
| 5 | ` standing` | 18.812 | 19.000 | -0.188 | 18.625 | no | —(rejected) |  |
| 6 | ` close` | 18.781 | 19.031 | -0.250 | 18.531 | no | —(rejected) |  |
| 7 | ` walking` | 18.359 | 18.422 | -0.062 | 18.297 | no | —(rejected) |  |
| 8 | ` two` | 18.266 | 18.438 | -0.172 | 18.094 | no | —(rejected) |  |
| 9 | ` further` | 18.109 | 18.141 | -0.031 | 18.078 | no | —(rejected) |  |
| 10 | ` individual` | 17.656 | 17.734 | -0.078 | 17.578 | no | —(rejected) |  |


### image 1584 — word "truck" (node: truck) — step 56

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` car` | 21.000 | 21.469 | -0.469 | 20.531 | yes | 20.531 |  |
| 2 | ` tr` | 20.875 | 21.156 | -0.281 | 20.594 | yes | 20.594 | **YES** |
| 3 | ` few` | 20.016 | 20.812 | -0.797 | 19.219 | yes | 19.219 |  |
| 4 | ` couple` | 19.781 | 20.234 | -0.453 | 19.328 | yes | 19.328 |  |
| 5 | ` smaller` | 19.156 | 19.219 | -0.062 | 19.094 | no | —(rejected) |  |
| 6 | ` bus` | 18.828 | 17.656 | +1.172 | 20.000 | no | —(rejected) |  |
| 7 | ` small` | 17.141 | 17.203 | -0.062 | 17.078 | no | —(rejected) |  |
| 8 | ` train` | 17.078 | 16.953 | +0.125 | 17.203 | no | —(rejected) |  |
| 9 | ` motor` | 17.016 | 17.141 | -0.125 | 16.891 | no | —(rejected) |  |
| 10 | ` person` | 16.719 | 17.422 | -0.703 | 16.016 | no | —(rejected) |  |


### image 1584 — word "truck" (node: truck) — step 57

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `uck` | 28.031 | 27.984 | +0.047 | 28.078 | yes | 28.078 | **YES** |
| 2 | `ash` | 16.453 | 15.500 | +0.953 | 17.406 | no | —(rejected) |  |
| 3 | `icy` | 16.453 | 16.406 | +0.047 | 16.500 | no | —(rejected) |  |
| 4 | `oupe` | 15.586 | 15.281 | +0.305 | 15.891 | no | —(rejected) |  |
| 5 | `ump` | 15.508 | 14.938 | +0.570 | 16.078 | no | —(rejected) |  |
| 6 | `unk` | 15.094 | 14.805 | +0.289 | 15.383 | no | —(rejected) |  |
| 7 | `ucker` | 15.031 | 14.578 | +0.453 | 15.484 | no | —(rejected) |  |
| 8 | `uc` | 14.656 | 14.281 | +0.375 | 15.031 | no | —(rejected) |  |
| 9 | `amp` | 14.641 | 13.641 | +1.000 | 15.641 | no | —(rejected) |  |
| 10 | `end` | 14.258 | 13.969 | +0.289 | 14.547 | no | —(rejected) |  |


### image 1584 — word "backpack" (node: backpack) — step 83

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` back` | 23.609 | 23.562 | +0.047 | 23.656 | yes | 23.656 | **YES** |
| 2 | ` hand` | 23.344 | 23.469 | -0.125 | 23.219 | yes | 23.219 |  |
| 3 | ` um` | 20.062 | 19.719 | +0.344 | 20.406 | no | —(rejected) |  |
| 4 | ` personal` | 19.359 | 19.422 | -0.062 | 19.297 | no | —(rejected) |  |
| 5 | ` b` | 18.906 | 18.484 | +0.422 | 19.328 | no | —(rejected) |  |
| 6 | ` items` | 18.578 | 18.312 | +0.266 | 18.844 | no | —(rejected) |  |
| 7 | ` various` | 17.734 | 17.391 | +0.344 | 18.078 | no | —(rejected) |  |
| 8 | ` their` | 17.156 | 16.906 | +0.250 | 17.406 | no | —(rejected) |  |
| 9 | ` a` | 17.062 | 16.406 | +0.656 | 17.719 | no | —(rejected) |  |
| 10 | ` suit` | 16.781 | 16.875 | -0.094 | 16.688 | no | —(rejected) |  |


### image 1584 — word "backpack" (node: backpack) — step 84

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `pack` | 26.688 | 27.031 | -0.344 | 26.344 | yes | 26.344 | **YES** |
| 2 | ` pack` | 17.297 | 17.016 | +0.281 | 17.578 | no | —(rejected) |  |
| 3 | `back` | 16.000 | 15.688 | +0.312 | 16.312 | no | —(rejected) |  |
| 4 | `b` | 14.172 | 13.789 | +0.383 | 14.555 | no | —(rejected) |  |
| 5 | ` back` | 13.547 | 13.375 | +0.172 | 13.719 | no | —(rejected) |  |
| 6 | `s` | 13.531 | 13.391 | +0.141 | 13.672 | no | —(rejected) |  |
| 7 | `-` | 13.211 | 13.016 | +0.195 | 13.406 | no | —(rejected) |  |
| 8 | ` b` | 12.969 | 12.750 | +0.219 | 13.188 | no | —(rejected) |  |
| 9 | `p` | 12.297 | 11.742 | +0.555 | 12.852 | no | —(rejected) |  |
| 10 | `Pack` | 12.117 | 12.094 | +0.023 | 12.141 | no | —(rejected) |  |


### image 1584 — word "backpack" (node: backpack) — step 85

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `s` | 28.438 | 27.938 | +0.500 | 28.938 | yes | 28.938 | **YES** |
| 2 | ` and` | 17.125 | 16.359 | +0.766 | 17.891 | no | —(rejected) |  |
| 3 | ` stra` | 15.781 | 15.258 | +0.523 | 16.305 | no | —(rejected) |  |
| 4 | ` or` | 15.703 | 14.961 | +0.742 | 16.445 | no | —(rejected) |  |
| 5 | ` b` | 15.422 | 14.930 | +0.492 | 15.914 | no | —(rejected) |  |
| 6 | ` pack` | 15.281 | 14.859 | +0.422 | 15.703 | no | —(rejected) |  |
| 7 | `.` | 14.633 | 14.156 | +0.477 | 15.109 | no | —(rejected) |  |
| 8 | `,` | 14.508 | 13.742 | +0.766 | 15.273 | no | —(rejected) |  |
| 9 | `pack` | 14.406 | 14.211 | +0.195 | 14.602 | no | —(rejected) |  |
| 10 | `-` | 13.828 | 13.766 | +0.062 | 13.891 | no | —(rejected) |  |


### image 6763 — word "bottle" (node: bottle) — step 74

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bott` | 20.000 | 19.031 | +0.969 | 20.969 | yes | 20.969 | **YES** |
| 2 | ` other` | 18.469 | 17.719 | +0.750 | 19.219 | yes | 19.219 |  |
| 3 | ` ch` | 18.453 | 17.406 | +1.047 | 19.500 | yes | 19.500 |  |
| 4 | ` people` | 18.141 | 16.688 | +1.453 | 19.594 | no | —(rejected) |  |
| 5 | ` wine` | 16.719 | 16.234 | +0.484 | 17.203 | no | —(rejected) |  |
| 6 | ` books` | 16.688 | 17.750 | -1.062 | 15.625 | no | —(rejected) |  |
| 7 | ` d` | 16.531 | 15.133 | +1.398 | 17.930 | no | —(rejected) |  |
| 8 | ` cu` | 16.469 | 16.188 | +0.281 | 16.750 | no | —(rejected) |  |
| 9 | ` objects` | 15.617 | 14.750 | +0.867 | 16.484 | no | —(rejected) |  |
| 10 | ` pictures` | 15.156 | 14.117 | +1.039 | 16.195 | no | —(rejected) |  |


### image 6763 — word "bottle" (node: bottle) — step 75

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `les` | 28.672 | 28.750 | -0.078 | 28.594 | yes | 28.594 | **YES** |
| 2 | `le` | 20.312 | 19.953 | +0.359 | 20.672 | no | —(rejected) |  |
| 3 | `led` | 20.156 | 20.234 | -0.078 | 20.078 | no | —(rejected) |  |
| 4 | `LES` | 15.984 | 16.047 | -0.062 | 15.922 | no | —(rejected) |  |
| 5 | `l` | 15.578 | 15.250 | +0.328 | 15.906 | no | —(rejected) |  |
| 6 | `ling` | 15.570 | 15.383 | +0.188 | 15.758 | no | —(rejected) |  |
| 7 | `ls` | 14.305 | 13.695 | +0.609 | 14.914 | no | —(rejected) |  |
| 8 | `es` | 14.297 | 14.148 | +0.148 | 14.445 | no | —(rejected) |  |
| 9 | `lers` | 14.211 | 13.820 | +0.391 | 14.602 | no | —(rejected) |  |
| 10 | `lets` | 13.914 | 13.445 | +0.469 | 14.383 | no | —(rejected) |  |


### image 6763 — word "chair" (node: chair) — step 115

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` chair` | 19.375 | 16.625 | +2.750 | 22.125 | yes | 22.125 | **YES** |
| 2 | ` hand` | 17.984 | 17.141 | +0.844 | 18.828 | yes | 18.828 |  |
| 3 | ` d` | 17.922 | 15.180 | +2.742 | 20.664 | yes | 20.664 |  |
| 4 | ` cell` | 17.469 | 16.094 | +1.375 | 18.844 | no | —(rejected) |  |
| 5 | ` car` | 16.406 | 14.422 | +1.984 | 18.391 | no | —(rejected) |  |
| 6 | ` couple` | 15.992 | 14.602 | +1.391 | 17.383 | no | —(rejected) |  |
| 7 | ` clock` | 15.992 | 15.883 | +0.109 | 16.102 | no | —(rejected) |  |
| 8 | ` tie` | 15.922 | 15.812 | +0.109 | 16.031 | no | —(rejected) |  |
| 9 | ` cup` | 15.875 | 15.688 | +0.188 | 16.062 | no | —(rejected) |  |
| 10 | ` third` | 15.820 | 15.336 | +0.484 | 16.305 | no | —(rejected) |  |


### image 1425 — word "doughnut" (node: donut) — step 24

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` d` | 16.812 | 16.562 | +0.250 | 17.062 | yes | 17.062 | **YES** |
| 2 | ` don` | 15.570 | 16.031 | -0.461 | 15.109 | yes | 15.109 |  |
| 3 | ` m` | 14.852 | 14.188 | +0.664 | 15.516 | no | —(rejected) |  |
| 4 | ` sand` | 14.828 | 12.883 | +1.945 | 16.773 | no | —(rejected) |  |
| 5 | ` pas` | 14.344 | 14.930 | -0.586 | 13.758 | no | —(rejected) |  |
| 6 | ` cre` | 14.242 | 14.820 | -0.578 | 13.664 | no | —(rejected) |  |
| 7 | ` c` | 13.656 | 13.984 | -0.328 | 13.328 | no | —(rejected) |  |
| 8 | ` roll` | 13.594 | 12.641 | +0.953 | 14.547 | no | —(rejected) |  |
| 9 | ` dess` | 13.578 | 14.227 | -0.648 | 12.930 | no | —(rejected) |  |
| 10 | ` cro` | 13.570 | 13.875 | -0.305 | 13.266 | no | —(rejected) |  |


### image 1425 — word "doughnut" (node: donut) — step 25

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ough` | 25.641 | 25.312 | +0.328 | 25.969 | yes | 25.969 | **YES** |
| 2 | `ought` | 18.594 | 18.062 | +0.531 | 19.125 | no | —(rejected) |  |
| 3 | `oug` | 18.156 | 17.375 | +0.781 | 18.938 | no | —(rejected) |  |
| 4 | `ome` | 15.984 | 16.547 | -0.562 | 15.422 | no | —(rejected) |  |
| 5 | `um` | 15.742 | 16.078 | -0.336 | 15.406 | no | —(rejected) |  |
| 6 | `ish` | 15.680 | 15.984 | -0.305 | 15.375 | no | —(rejected) |  |
| 7 | `oun` | 15.664 | 14.727 | +0.938 | 16.602 | no | —(rejected) |  |
| 8 | `ess` | 15.344 | 15.891 | -0.547 | 14.797 | no | —(rejected) |  |
| 9 | `unk` | 15.141 | 13.586 | +1.555 | 16.695 | no | —(rejected) |  |
| 10 | `une` | 14.625 | 13.961 | +0.664 | 15.289 | no | —(rejected) |  |


### image 1425 — word "doughnut" (node: donut) — step 26

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `nut` | 20.312 | 21.000 | -0.688 | 19.625 | yes | 19.625 | **YES** |
| 2 | ` nut` | 13.586 | 13.891 | -0.305 | 13.281 | no | —(rejected) |  |
| 3 | ` pas` | 12.227 | 12.203 | +0.023 | 12.250 | no | —(rejected) |  |
| 4 | ` or` | 11.664 | 11.773 | -0.109 | 11.555 | no | —(rejected) |  |
| 5 | `ut` | 11.625 | 11.836 | -0.211 | 11.414 | no | —(rejected) |  |
| 6 | `,` | 11.477 | 11.906 | -0.430 | 11.047 | no | —(rejected) |  |
| 7 | `n` | 11.344 | 11.242 | +0.102 | 11.445 | no | —(rejected) |  |
| 8 | ` filled` | 11.328 | 11.117 | +0.211 | 11.539 | no | —(rejected) |  |
| 9 | ` b` | 10.961 | 10.367 | +0.594 | 11.555 | no | —(rejected) |  |
| 10 | `-` | 10.688 | 10.352 | +0.336 | 11.023 | no | —(rejected) |  |


### image 1425 — word "fork" (node: fork) — step 80

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` fork` | 17.641 | 15.172 | +2.469 | 20.109 | yes | 20.109 | **YES** |
| 2 | ` person` | 17.562 | 17.031 | +0.531 | 18.094 | yes | 18.094 |  |
| 3 | ` sp` | 17.000 | 16.672 | +0.328 | 17.328 | yes | 17.328 |  |
| 4 | ` kn` | 15.984 | 16.031 | -0.047 | 15.938 | no | —(rejected) |  |
| 5 | ` cup` | 15.641 | 15.727 | -0.086 | 15.555 | no | —(rejected) |  |
| 6 | ` wine` | 14.648 | 15.094 | -0.445 | 14.203 | no | —(rejected) |  |
| 7 | ` bott` | 14.602 | 14.516 | +0.086 | 14.688 | no | —(rejected) |  |
| 8 | ` pair` | 14.391 | 13.625 | +0.766 | 15.156 | no | —(rejected) |  |
| 9 | ` glass` | 14.320 | 14.531 | -0.211 | 14.109 | no | —(rejected) |  |
| 10 | ` hand` | 13.883 | 12.469 | +1.414 | 15.297 | no | —(rejected) |  |


## SID — REAL OBJECTS


### image 776 — word "teddy bear" (node: teddy bear) — step 7

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` ted` | 17.094 | 16.969 | +0.125 | 17.219 | yes | 17.219 | **YES** |
| 2 | ` brown` | 16.469 | 16.203 | +0.266 | 16.734 | yes | 16.734 |  |
| 3 | ` stuff` | 15.977 | 15.578 | +0.398 | 16.375 | yes | 16.375 |  |
| 4 | ` large` | 14.992 | 15.234 | -0.242 | 14.750 | no | —(rejected) |  |
| 5 | ` c` | 14.828 | 13.516 | +1.312 | 16.141 | no | —(rejected) |  |
| 6 | ` pl` | 13.977 | 13.008 | +0.969 | 14.945 | no | —(rejected) |  |
| 7 | ` fl` | 13.836 | 12.445 | +1.391 | 15.227 | no | —(rejected) |  |
| 8 | ` small` | 13.805 | 12.734 | +1.070 | 14.875 | no | —(rejected) |  |
| 9 | ` different` | 13.633 | 13.047 | +0.586 | 14.219 | no | —(rejected) |  |
| 10 | ` old` | 13.352 | 13.469 | -0.117 | 13.234 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 8

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `dy` | 27.062 | 28.031 | -0.969 | 26.094 | yes | 26.094 | **YES** |
| 2 | `d` | 17.391 | 17.047 | +0.344 | 17.734 | no | —(rejected) |  |
| 3 | ` be` | 13.688 | 14.484 | -0.797 | 12.891 | no | —(rejected) |  |
| 4 | `di` | 13.469 | 13.492 | -0.023 | 13.445 | no | —(rejected) |  |
| 5 | `b` | 13.336 | 12.664 | +0.672 | 14.008 | no | —(rejected) |  |
| 6 | `die` | 13.055 | 12.414 | +0.641 | 13.695 | no | —(rejected) |  |
| 7 | ` bear` | 12.500 | 12.125 | +0.375 | 12.875 | no | —(rejected) |  |
| 8 | ` ted` | 12.297 | 12.547 | -0.250 | 12.047 | no | —(rejected) |  |
| 9 | `der` | 11.312 | 11.117 | +0.195 | 11.508 | no | —(rejected) |  |
| 10 | `ious` | 11.172 | 11.938 | -0.766 | 10.406 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 9

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` be` | 22.969 | 23.109 | -0.141 | 22.828 | yes | 22.828 | **YES** |
| 2 | ` bear` | 18.312 | 17.906 | +0.406 | 18.719 | no | —(rejected) |  |
| 3 | ` ted` | 14.039 | 13.484 | +0.555 | 14.594 | no | —(rejected) |  |
| 4 | ` brown` | 13.922 | 13.805 | +0.117 | 14.039 | no | —(rejected) |  |
| 5 | `b` | 13.914 | 13.758 | +0.156 | 14.070 | no | —(rejected) |  |
| 6 | `-` | 12.648 | 12.203 | +0.445 | 13.094 | no | —(rejected) |  |
| 7 | ` stuff` | 12.562 | 11.875 | +0.688 | 13.250 | no | —(rejected) |  |
| 8 | ` animals` | 12.234 | 12.359 | -0.125 | 12.109 | no | —(rejected) |  |
| 9 | ` b` | 11.555 | 11.172 | +0.383 | 11.938 | no | —(rejected) |  |
| 10 | ` and` | 11.266 | 10.906 | +0.359 | 11.625 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 10

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ars` | 32.938 | 31.969 | +0.969 | 33.906 | yes | 33.906 | **YES** |
| 2 | `ats` | 17.047 | 16.328 | +0.719 | 17.766 | no | —(rejected) |  |
| 3 | `ams` | 16.688 | 16.047 | +0.641 | 17.328 | no | —(rejected) |  |
| 4 | `ers` | 16.672 | 16.344 | +0.328 | 17.000 | no | —(rejected) |  |
| 5 | `es` | 16.391 | 15.977 | +0.414 | 16.805 | no | —(rejected) |  |
| 6 | `ans` | 15.484 | 14.328 | +1.156 | 16.641 | no | —(rejected) |  |
| 7 | `er` | 14.469 | 13.336 | +1.133 | 15.602 | no | —(rejected) |  |
| 8 | `a` | 14.344 | 13.133 | +1.211 | 15.555 | no | —(rejected) |  |
| 9 | `ads` | 14.289 | 12.539 | +1.750 | 16.039 | no | —(rejected) |  |
| 10 | `ar` | 14.078 | 13.148 | +0.930 | 15.008 | no | —(rejected) |  |


### image 776 — word "bed" (node: bed) — step 15

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bed` | 20.375 | 20.281 | +0.094 | 20.469 | yes | 20.469 | **YES** |
| 2 | ` blue` | 19.062 | 17.688 | +1.375 | 20.438 | yes | 20.438 |  |
| 3 | ` surface` | 18.078 | 18.656 | -0.578 | 17.500 | no | —(rejected) |  |
| 4 | ` c` | 17.828 | 18.938 | -1.109 | 16.719 | no | —(rejected) |  |
| 5 | ` p` | 17.047 | 16.625 | +0.422 | 17.469 | no | —(rejected) |  |
| 6 | ` blank` | 16.844 | 16.469 | +0.375 | 17.219 | no | —(rejected) |  |
| 7 | ` soft` | 16.750 | 16.219 | +0.531 | 17.281 | no | —(rejected) |  |
| 8 | ` table` | 16.125 | 17.281 | -1.156 | 14.969 | no | —(rejected) |  |
| 9 | ` light` | 16.047 | 14.609 | +1.438 | 17.484 | no | —(rejected) |  |
| 10 | ` white` | 16.000 | 16.391 | -0.391 | 15.609 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 29

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` be` | 21.297 | 21.484 | -0.188 | 21.109 | yes | 21.109 |  |
| 2 | ` ted` | 21.188 | 21.016 | +0.172 | 21.359 | yes | 21.359 | **YES** |
| 3 | ` stuff` | 18.328 | 17.688 | +0.641 | 18.969 | no | —(rejected) |  |
| 4 | ` bear` | 16.984 | 16.766 | +0.219 | 17.203 | no | —(rejected) |  |
| 5 | ` larger` | 16.766 | 17.359 | -0.594 | 16.172 | no | —(rejected) |  |
| 6 | ` brown` | 16.625 | 15.852 | +0.773 | 17.398 | no | —(rejected) |  |
| 7 | ` smaller` | 16.000 | 16.266 | -0.266 | 15.734 | no | —(rejected) |  |
| 8 | ` to` | 15.758 | 15.875 | -0.117 | 15.641 | no | —(rejected) |  |
| 9 | ` pl` | 14.695 | 14.227 | +0.469 | 15.164 | no | —(rejected) |  |
| 10 | ` small` | 14.594 | 14.266 | +0.328 | 14.922 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 30

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `dy` | 25.359 | 26.672 | -1.312 | 24.047 | yes | 24.047 | **YES** |
| 2 | `d` | 19.891 | 19.922 | -0.031 | 19.859 | no | —(rejected) |  |
| 3 | `b` | 15.438 | 15.164 | +0.273 | 15.711 | no | —(rejected) |  |
| 4 | `die` | 13.875 | 13.648 | +0.227 | 14.102 | no | —(rejected) |  |
| 5 | `di` | 13.250 | 13.203 | +0.047 | 13.297 | no | —(rejected) |  |
| 6 | `der` | 12.836 | 12.812 | +0.023 | 12.859 | no | —(rejected) |  |
| 7 | ` be` | 12.180 | 12.859 | -0.680 | 11.500 | no | —(rejected) |  |
| 8 | `be` | 12.180 | 11.844 | +0.336 | 12.516 | no | —(rejected) |  |
| 9 | `ds` | 11.977 | 11.516 | +0.461 | 12.438 | no | —(rejected) |  |
| 10 | `ders` | 11.969 | 11.344 | +0.625 | 12.594 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 31

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` be` | 25.969 | 25.828 | +0.141 | 26.109 | yes | 26.109 | **YES** |
| 2 | ` bear` | 19.047 | 18.625 | +0.422 | 19.469 | no | —(rejected) |  |
| 3 | `-` | 14.539 | 13.719 | +0.820 | 15.359 | no | —(rejected) |  |
| 4 | `'` | 14.188 | 13.430 | +0.758 | 14.945 | no | —(rejected) |  |
| 5 | ` b` | 14.109 | 13.531 | +0.578 | 14.688 | no | —(rejected) |  |
| 6 | `b` | 14.062 | 13.297 | +0.766 | 14.828 | no | —(rejected) |  |
| 7 | ` ears` | 13.852 | 13.641 | +0.211 | 14.062 | no | —(rejected) |  |
| 8 | ` animals` | 13.453 | 12.836 | +0.617 | 14.070 | no | —(rejected) |  |
| 9 | ` balls` | 13.305 | 12.688 | +0.617 | 13.922 | no | —(rejected) |  |
| 10 | ` to` | 12.844 | 12.312 | +0.531 | 13.375 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 32

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ars` | 34.125 | 34.250 | -0.125 | 34.000 | yes | 34.000 | **YES** |
| 2 | `ans` | 18.297 | 18.000 | +0.297 | 18.594 | no | —(rejected) |  |
| 3 | `ams` | 18.250 | 18.281 | -0.031 | 18.219 | no | —(rejected) |  |
| 4 | `ers` | 17.781 | 17.344 | +0.438 | 18.219 | no | —(rejected) |  |
| 5 | `ats` | 16.406 | 16.109 | +0.297 | 16.703 | no | —(rejected) |  |
| 6 | `es` | 15.836 | 15.547 | +0.289 | 16.125 | no | —(rejected) |  |
| 7 | `as` | 14.852 | 14.344 | +0.508 | 15.359 | no | —(rejected) |  |
| 8 | `eds` | 14.500 | 14.273 | +0.227 | 14.727 | no | —(rejected) |  |
| 9 | `ads` | 14.336 | 13.531 | +0.805 | 15.141 | no | —(rejected) |  |
| 10 | `ared` | 13.906 | 13.250 | +0.656 | 14.562 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 48

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` ted` | 16.562 | 16.250 | +0.312 | 16.875 | yes | 16.875 | **YES** |
| 2 | ` be` | 16.234 | 16.109 | +0.125 | 16.359 | yes | 16.359 |  |
| 3 | ` arrangement` | 14.844 | 14.695 | +0.148 | 14.992 | no | —(rejected) |  |
| 4 | ` bed` | 14.250 | 14.516 | -0.266 | 13.984 | no | —(rejected) |  |
| 5 | ` scene` | 13.984 | 14.188 | -0.203 | 13.781 | no | —(rejected) |  |
| 6 | ` colors` | 13.516 | 13.102 | +0.414 | 13.930 | no | —(rejected) |  |
| 7 | ` variety` | 13.398 | 13.375 | +0.023 | 13.422 | no | —(rejected) |  |
| 8 | ` various` | 13.289 | 13.195 | +0.094 | 13.383 | no | —(rejected) |  |
| 9 | ` stuff` | 13.102 | 12.398 | +0.703 | 13.805 | no | —(rejected) |  |
| 10 | ` group` | 12.969 | 12.539 | +0.430 | 13.398 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 49

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `dy` | 26.172 | 26.516 | -0.344 | 25.828 | yes | 25.828 | **YES** |
| 2 | `d` | 19.906 | 19.594 | +0.312 | 20.219 | no | —(rejected) |  |
| 3 | `b` | 15.641 | 15.250 | +0.391 | 16.031 | no | —(rejected) |  |
| 4 | `die` | 13.211 | 12.789 | +0.422 | 13.633 | no | —(rejected) |  |
| 5 | `der` | 13.078 | 12.781 | +0.297 | 13.375 | no | —(rejected) |  |
| 6 | `di` | 12.805 | 12.633 | +0.172 | 12.977 | no | —(rejected) |  |
| 7 | ` be` | 12.398 | 12.406 | -0.008 | 12.391 | no | —(rejected) |  |
| 8 | ` bear` | 12.008 | 12.078 | -0.070 | 11.938 | no | —(rejected) |  |
| 9 | `s` | 11.805 | 12.039 | -0.234 | 11.570 | no | —(rejected) |  |
| 10 | `be` | 11.797 | 11.156 | +0.641 | 12.438 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 50

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` be` | 25.984 | 26.281 | -0.297 | 25.688 | yes | 25.688 | **YES** |
| 2 | ` bear` | 19.953 | 19.891 | +0.062 | 20.016 | no | —(rejected) |  |
| 3 | `-` | 14.367 | 14.000 | +0.367 | 14.734 | no | —(rejected) |  |
| 4 | ` animals` | 14.242 | 13.977 | +0.266 | 14.508 | no | —(rejected) |  |
| 5 | ` b` | 14.211 | 13.883 | +0.328 | 14.539 | no | —(rejected) |  |
| 6 | ` collection` | 13.781 | 13.477 | +0.305 | 14.086 | no | —(rejected) |  |
| 7 | ` to` | 13.633 | 13.328 | +0.305 | 13.938 | no | —(rejected) |  |
| 8 | ` balls` | 13.594 | 13.492 | +0.102 | 13.695 | no | —(rejected) |  |
| 9 | `'` | 13.109 | 12.586 | +0.523 | 13.633 | no | —(rejected) |  |
| 10 | ` bunch` | 13.008 | 12.414 | +0.594 | 13.602 | no | —(rejected) |  |


### image 776 — word "teddy bear" (node: teddy bear) — step 51

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ars` | 32.156 | 31.766 | +0.391 | 32.547 | yes | 32.547 | **YES** |
| 2 | `ans` | 17.906 | 17.125 | +0.781 | 18.688 | no | —(rejected) |  |
| 3 | `ers` | 17.016 | 16.453 | +0.562 | 17.578 | no | —(rejected) |  |
| 4 | `ams` | 16.469 | 15.914 | +0.555 | 17.023 | no | —(rejected) |  |
| 5 | `es` | 16.438 | 16.203 | +0.234 | 16.672 | no | —(rejected) |  |
| 6 | `ats` | 16.188 | 15.992 | +0.195 | 16.383 | no | —(rejected) |  |
| 7 | `ar` | 14.875 | 14.039 | +0.836 | 15.711 | no | —(rejected) |  |
| 8 | `ared` | 14.016 | 12.672 | +1.344 | 15.359 | no | —(rejected) |  |
| 9 | `as` | 13.875 | 13.328 | +0.547 | 14.422 | no | —(rejected) |  |
| 10 | `a` | 13.859 | 13.617 | +0.242 | 14.102 | no | —(rejected) |  |


### image 776 — word "bed" (node: bed) — step 68

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bed` | 15.594 | 15.641 | -0.047 | 15.547 | yes | 15.547 | **YES** |
| 2 | ` arrangement` | 15.047 | 14.898 | +0.148 | 15.195 | yes | 15.195 |  |
| 3 | ` overall` | 14.180 | 14.703 | -0.523 | 13.656 | yes | 13.656 |  |
| 4 | ` scene` | 13.875 | 14.055 | -0.180 | 13.695 | no | —(rejected) |  |
| 5 | ` be` | 13.789 | 13.164 | +0.625 | 14.414 | no | —(rejected) |  |
| 6 | ` p` | 13.047 | 12.289 | +0.758 | 13.805 | no | —(rejected) |  |
| 7 | ` largest` | 12.961 | 12.094 | +0.867 | 13.828 | no | —(rejected) |  |
| 8 | ` close` | 12.820 | 13.023 | -0.203 | 12.617 | no | —(rejected) |  |
| 9 | ` group` | 12.750 | 12.094 | +0.656 | 13.406 | no | —(rejected) |  |
| 10 | ` ted` | 12.516 | 11.695 | +0.820 | 13.336 | no | —(rejected) |  |


### image 4765 — word "man" (node: person) — step 5

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 18.859 | 18.828 | +0.031 | 18.891 | yes | 18.891 | **YES** |
| 2 | ` person` | 18.453 | 18.297 | +0.156 | 18.609 | yes | 18.609 |  |
| 3 | ` sur` | 18.266 | 18.047 | +0.219 | 18.484 | yes | 18.484 |  |
| 4 | ` young` | 17.938 | 17.594 | +0.344 | 18.281 | yes | 18.281 |  |
| 5 | ` thr` | 17.641 | 17.250 | +0.391 | 18.031 | yes | 18.031 |  |
| 6 | ` male` | 17.219 | 17.125 | +0.094 | 17.312 | no | —(rejected) |  |
| 7 | ` woman` | 16.938 | 15.844 | +1.094 | 18.031 | no | —(rejected) |  |
| 8 | ` dynamic` | 16.875 | 16.969 | -0.094 | 16.781 | no | —(rejected) |  |
| 9 | ` l` | 16.031 | 16.141 | -0.109 | 15.922 | no | —(rejected) |  |
| 10 | ` sk` | 15.906 | 16.016 | -0.109 | 15.797 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 14

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sur` | 23.266 | 23.406 | -0.141 | 23.125 | yes | 23.125 | **YES** |
| 2 | ` white` | 20.953 | 18.234 | +2.719 | 23.672 | no | —(rejected) |  |
| 3 | ` large` | 18.359 | 17.266 | +1.094 | 19.453 | no | —(rejected) |  |
| 4 | ` yellow` | 17.984 | 18.125 | -0.141 | 17.844 | no | —(rejected) |  |
| 5 | ` long` | 17.859 | 15.336 | +2.523 | 20.383 | no | —(rejected) |  |
| 6 | ` wave` | 17.703 | 18.734 | -1.031 | 16.672 | no | —(rejected) |  |
| 7 | ` small` | 17.188 | 17.719 | -0.531 | 16.656 | no | —(rejected) |  |
| 8 | ` bo` | 16.906 | 17.453 | -0.547 | 16.359 | no | —(rejected) |  |
| 9 | ` board` | 16.484 | 16.953 | -0.469 | 16.016 | no | —(rejected) |  |
| 10 | ` bright` | 16.172 | 14.750 | +1.422 | 17.594 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 15

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `f` | 27.750 | 28.844 | -1.094 | 26.656 | yes | 26.656 | **YES** |
| 2 | `board` | 17.594 | 17.609 | -0.016 | 17.578 | no | —(rejected) |  |
| 3 | `fer` | 16.672 | 17.000 | -0.328 | 16.344 | no | —(rejected) |  |
| 4 | `ft` | 14.047 | 14.039 | +0.008 | 14.055 | no | —(rejected) |  |
| 5 | `fers` | 13.422 | 13.680 | -0.258 | 13.164 | no | —(rejected) |  |
| 6 | `fo` | 12.898 | 12.625 | +0.273 | 13.172 | no | —(rejected) |  |
| 7 | `fc` | 11.773 | 11.711 | +0.062 | 11.836 | no | —(rejected) |  |
| 8 | `fin` | 11.609 | 11.672 | -0.062 | 11.547 | no | —(rejected) |  |
| 9 | `face` | 11.578 | 11.336 | +0.242 | 11.820 | no | —(rejected) |  |
| 10 | `ge` | 11.547 | 11.789 | -0.242 | 11.305 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 16

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 25.594 | 25.594 | +0.000 | 25.594 | yes | 25.594 | **YES** |
| 2 | ` board` | 16.891 | 16.062 | +0.828 | 17.719 | no | —(rejected) |  |
| 3 | `ing` | 16.047 | 15.633 | +0.414 | 16.461 | no | —(rejected) |  |
| 4 | `boards` | 14.367 | 14.031 | +0.336 | 14.703 | no | —(rejected) |  |
| 5 | ` sur` | 13.102 | 12.031 | +1.070 | 14.172 | no | —(rejected) |  |
| 6 | `ba` | 12.289 | 11.867 | +0.422 | 12.711 | no | —(rejected) |  |
| 7 | `-` | 12.164 | 11.367 | +0.797 | 12.961 | no | —(rejected) |  |
| 8 | `able` | 12.117 | 11.734 | +0.383 | 12.500 | no | —(rejected) |  |
| 9 | `bo` | 11.898 | 11.312 | +0.586 | 12.484 | no | —(rejected) |  |
| 10 | `acing` | 10.875 | 10.633 | +0.242 | 11.117 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 58

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sur` | 19.828 | 19.797 | +0.031 | 19.859 | yes | 19.859 | **YES** |
| 2 | ` board` | 17.906 | 18.141 | -0.234 | 17.672 | no | —(rejected) |  |
| 3 | ` wave` | 17.859 | 17.688 | +0.172 | 18.031 | no | —(rejected) |  |
| 4 | ` white` | 17.219 | 15.984 | +1.234 | 18.453 | no | —(rejected) |  |
| 5 | ` edge` | 15.523 | 15.352 | +0.172 | 15.695 | no | —(rejected) |  |
| 6 | ` top` | 15.508 | 15.484 | +0.023 | 15.531 | no | —(rejected) |  |
| 7 | ` water` | 15.141 | 15.320 | -0.180 | 14.961 | no | —(rejected) |  |
| 8 | ` waves` | 15.133 | 15.164 | -0.031 | 15.102 | no | —(rejected) |  |
| 9 | ` cr` | 15.062 | 14.664 | +0.398 | 15.461 | no | —(rejected) |  |
| 10 | ` large` | 15.039 | 14.539 | +0.500 | 15.539 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 59

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `f` | 25.828 | 26.391 | -0.562 | 25.266 | yes | 25.266 | **YES** |
| 2 | `board` | 17.125 | 17.062 | +0.062 | 17.188 | no | —(rejected) |  |
| 3 | `fer` | 15.734 | 15.695 | +0.039 | 15.773 | no | —(rejected) |  |
| 4 | `fo` | 13.805 | 13.523 | +0.281 | 14.086 | no | —(rejected) |  |
| 5 | `ft` | 13.695 | 13.375 | +0.320 | 14.016 | no | —(rejected) |  |
| 6 | `face` | 13.016 | 12.875 | +0.141 | 13.156 | no | —(rejected) |  |
| 7 | `fc` | 12.516 | 12.289 | +0.227 | 12.742 | no | —(rejected) |  |
| 8 | `ge` | 12.109 | 11.969 | +0.141 | 12.250 | no | —(rejected) |  |
| 9 | `fers` | 12.086 | 12.055 | +0.031 | 12.117 | no | —(rejected) |  |
| 10 | `fac` | 11.859 | 11.539 | +0.320 | 12.180 | no | —(rejected) |  |


### image 4765 — word "surfboard" (node: surfboard) — step 60

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 26.250 | 26.156 | +0.094 | 26.344 | yes | 26.344 | **YES** |
| 2 | ` board` | 17.422 | 16.734 | +0.688 | 18.109 | no | —(rejected) |  |
| 3 | `ing` | 16.672 | 15.914 | +0.758 | 17.430 | no | —(rejected) |  |
| 4 | `boards` | 15.867 | 15.477 | +0.391 | 16.258 | no | —(rejected) |  |
| 5 | `,` | 14.812 | 14.758 | +0.055 | 14.867 | no | —(rejected) |  |
| 6 | ` and` | 13.555 | 13.195 | +0.359 | 13.914 | no | —(rejected) |  |
| 7 | ` as` | 13.531 | 13.344 | +0.188 | 13.719 | no | —(rejected) |  |
| 8 | `acing` | 13.469 | 13.516 | -0.047 | 13.422 | no | —(rejected) |  |
| 9 | `bo` | 13.148 | 12.539 | +0.609 | 13.758 | no | —(rejected) |  |
| 10 | `ba` | 12.938 | 12.570 | +0.367 | 13.305 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 4

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` stop` | 17.438 | 16.797 | +0.641 | 18.078 | yes | 18.078 | **YES** |
| 2 | ` red` | 17.422 | 17.438 | -0.016 | 17.406 | yes | 17.406 |  |
| 3 | ` street` | 16.062 | 16.156 | -0.094 | 15.969 | yes | 15.969 |  |
| 4 | ` busy` | 15.438 | 15.289 | +0.148 | 15.586 | no | —(rejected) |  |
| 5 | ` large` | 15.172 | 15.219 | -0.047 | 15.125 | no | —(rejected) |  |
| 6 | ` city` | 15.008 | 15.023 | -0.016 | 14.992 | no | —(rejected) |  |
| 7 | ` unique` | 14.680 | 12.719 | +1.961 | 16.641 | no | —(rejected) |  |
| 8 | ` traffic` | 14.492 | 13.266 | +1.227 | 15.719 | no | —(rejected) |  |
| 9 | ` road` | 14.016 | 13.023 | +0.992 | 15.008 | no | —(rejected) |  |
| 10 | ` tall` | 13.406 | 13.570 | -0.164 | 13.242 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 5

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sign` | 24.328 | 23.984 | +0.344 | 24.672 | yes | 24.672 | **YES** |
| 2 | `-` | 16.781 | 16.781 | +0.000 | 16.781 | no | —(rejected) |  |
| 3 | ` street` | 15.945 | 16.375 | -0.430 | 15.516 | no | —(rejected) |  |
| 4 | ` and` | 15.570 | 15.586 | -0.016 | 15.555 | no | —(rejected) |  |
| 5 | ` or` | 15.531 | 15.750 | -0.219 | 15.312 | no | —(rejected) |  |
| 6 | `light` | 15.484 | 15.133 | +0.352 | 15.836 | no | —(rejected) |  |
| 7 | ` signs` | 14.969 | 14.625 | +0.344 | 15.312 | no | —(rejected) |  |
| 8 | ` signal` | 14.625 | 14.477 | +0.148 | 14.773 | no | —(rejected) |  |
| 9 | ` road` | 14.328 | 14.508 | -0.180 | 14.148 | no | —(rejected) |  |
| 10 | ` light` | 14.133 | 13.820 | +0.312 | 14.445 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 26

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` stop` | 17.516 | 17.109 | +0.406 | 17.922 | yes | 17.922 | **YES** |
| 2 | ` sign` | 16.469 | 15.953 | +0.516 | 16.984 | yes | 16.984 |  |
| 3 | ` street` | 14.891 | 14.328 | +0.562 | 15.453 | no | —(rejected) |  |
| 4 | ` scene` | 14.414 | 13.641 | +0.773 | 15.188 | no | —(rejected) |  |
| 5 | ` red` | 14.352 | 13.867 | +0.484 | 14.836 | no | —(rejected) |  |
| 6 | ` word` | 13.805 | 13.773 | +0.031 | 13.836 | no | —(rejected) |  |
| 7 | ` traffic` | 13.781 | 13.242 | +0.539 | 14.320 | no | —(rejected) |  |
| 8 | ` intersection` | 13.484 | 13.109 | +0.375 | 13.859 | no | —(rejected) |  |
| 9 | ` "` | 12.766 | 12.539 | +0.227 | 12.992 | no | —(rejected) |  |
| 10 | ` road` | 12.727 | 12.266 | +0.461 | 13.188 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 27

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sign` | 25.406 | 25.344 | +0.062 | 25.469 | yes | 25.469 | **YES** |
| 2 | ` signs` | 18.047 | 17.688 | +0.359 | 18.406 | no | —(rejected) |  |
| 3 | ` signal` | 15.758 | 15.242 | +0.516 | 16.273 | no | —(rejected) |  |
| 4 | ` is` | 15.562 | 15.242 | +0.320 | 15.883 | no | —(rejected) |  |
| 5 | `-` | 15.406 | 15.070 | +0.336 | 15.742 | no | —(rejected) |  |
| 6 | `light` | 14.180 | 13.602 | +0.578 | 14.758 | no | —(rejected) |  |
| 7 | ` s` | 14.133 | 13.859 | +0.273 | 14.406 | no | —(rejected) |  |
| 8 | ` light` | 14.086 | 13.320 | +0.766 | 14.852 | no | —(rejected) |  |
| 9 | ` and` | 13.984 | 13.508 | +0.477 | 14.461 | no | —(rejected) |  |
| 10 | ` has` | 13.070 | 12.484 | +0.586 | 13.656 | no | —(rejected) |  |


### image 724 — word "truck" (node: truck) — step 56

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` tr` | 21.000 | 21.391 | -0.391 | 20.609 | yes | 20.609 | **YES** |
| 2 | ` car` | 19.891 | 20.875 | -0.984 | 18.906 | yes | 18.906 |  |
| 3 | ` bus` | 18.812 | 19.234 | -0.422 | 18.391 | no | —(rejected) |  |
| 4 | ` couple` | 18.312 | 18.750 | -0.438 | 17.875 | no | —(rejected) |  |
| 5 | ` few` | 17.547 | 18.219 | -0.672 | 16.875 | no | —(rejected) |  |
| 6 | ` large` | 16.609 | 16.656 | -0.047 | 16.562 | no | —(rejected) |  |
| 7 | ` white` | 16.078 | 14.586 | +1.492 | 17.570 | no | —(rejected) |  |
| 8 | ` small` | 15.969 | 16.297 | -0.328 | 15.641 | no | —(rejected) |  |
| 9 | ` mix` | 15.883 | 16.734 | -0.852 | 15.031 | no | —(rejected) |  |
| 10 | ` motor` | 15.852 | 17.234 | -1.383 | 14.469 | no | —(rejected) |  |


### image 724 — word "truck" (node: truck) — step 57

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `uck` | 26.234 | 26.734 | -0.500 | 25.734 | yes | 25.734 | **YES** |
| 2 | `ash` | 17.594 | 16.906 | +0.688 | 18.281 | no | —(rejected) |  |
| 3 | `ump` | 14.625 | 14.602 | +0.023 | 14.648 | no | —(rejected) |  |
| 4 | `unk` | 14.617 | 14.266 | +0.352 | 14.969 | no | —(rejected) |  |
| 5 | `icy` | 14.516 | 14.953 | -0.438 | 14.078 | no | —(rejected) |  |
| 6 | `ucker` | 14.461 | 13.789 | +0.672 | 15.133 | no | —(rejected) |  |
| 7 | `uc` | 14.445 | 13.898 | +0.547 | 14.992 | no | —(rejected) |  |
| 8 | `oupe` | 13.523 | 13.250 | +0.273 | 13.797 | no | —(rejected) |  |
| 9 | `amp` | 13.172 | 12.641 | +0.531 | 13.703 | no | —(rejected) |  |
| 10 | `uce` | 12.797 | 12.602 | +0.195 | 12.992 | no | —(rejected) |  |


### image 724 — word "car" (node: car) — step 60

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` car` | 21.281 | 22.531 | -1.250 | 20.031 | yes | 20.031 | **YES** |
| 2 | ` bus` | 20.234 | 20.625 | -0.391 | 19.844 | yes | 19.844 |  |
| 3 | ` couple` | 19.625 | 21.156 | -1.531 | 18.094 | no | —(rejected) |  |
| 4 | ` few` | 19.188 | 20.953 | -1.766 | 17.422 | no | —(rejected) |  |
| 5 | ` motor` | 17.250 | 18.109 | -0.859 | 16.391 | no | —(rejected) |  |
| 6 | ` van` | 16.922 | 16.547 | +0.375 | 17.297 | no | —(rejected) |  |
| 7 | ` small` | 16.734 | 17.344 | -0.609 | 16.125 | no | —(rejected) |  |
| 8 | ` smaller` | 16.734 | 17.859 | -1.125 | 15.609 | no | —(rejected) |  |
| 9 | ` tr` | 16.641 | 15.742 | +0.898 | 17.539 | no | —(rejected) |  |
| 10 | ` number` | 15.633 | 17.125 | -1.492 | 14.141 | no | —(rejected) |  |


### image 724 — word "truck" (node: truck) — step 98

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` left` | 19.594 | 19.734 | -0.141 | 19.453 | yes | 19.453 |  |
| 2 | ` tr` | 19.469 | 19.469 | +0.000 | 19.469 | yes | 19.469 | **YES** |
| 3 | ` center` | 19.391 | 19.344 | +0.047 | 19.438 | yes | 19.438 |  |
| 4 | ` stop` | 18.797 | 19.109 | -0.312 | 18.484 | yes | 18.484 |  |
| 5 | ` middle` | 18.312 | 18.359 | -0.047 | 18.266 | yes | 18.266 |  |
| 6 | ` right` | 17.922 | 17.859 | +0.062 | 17.984 | no | —(rejected) |  |
| 7 | ` park` | 17.781 | 17.859 | -0.078 | 17.703 | no | —(rejected) |  |
| 8 | ` front` | 17.609 | 17.750 | -0.141 | 17.469 | no | —(rejected) |  |
| 9 | ` car` | 17.375 | 17.266 | +0.109 | 17.484 | no | —(rejected) |  |
| 10 | ` intersection` | 17.234 | 16.984 | +0.250 | 17.484 | no | —(rejected) |  |


### image 724 — word "truck" (node: truck) — step 99

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `uck` | 26.812 | 26.797 | +0.016 | 26.828 | yes | 26.828 | **YES** |
| 2 | `unk` | 16.953 | 17.047 | -0.094 | 16.859 | no | —(rejected) |  |
| 3 | `ash` | 15.070 | 14.992 | +0.078 | 15.148 | no | —(rejected) |  |
| 4 | `ump` | 13.875 | 13.711 | +0.164 | 14.039 | no | —(rejected) |  |
| 5 | `uc` | 13.438 | 12.906 | +0.531 | 13.969 | no | —(rejected) |  |
| 6 | `ucker` | 13.297 | 12.469 | +0.828 | 14.125 | no | —(rejected) |  |
| 7 | `end` | 12.422 | 12.367 | +0.055 | 12.477 | no | —(rejected) |  |
| 8 | `ough` | 12.367 | 11.914 | +0.453 | 12.820 | no | —(rejected) |  |
| 9 | `uk` | 12.250 | 11.508 | +0.742 | 12.992 | no | —(rejected) |  |
| 10 | `amp` | 12.242 | 11.867 | +0.375 | 12.617 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 106

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` stop` | 17.766 | 17.859 | -0.094 | 17.672 | yes | 17.672 | **YES** |
| 2 | ` car` | 17.344 | 17.141 | +0.203 | 17.547 | yes | 17.547 |  |
| 3 | ` center` | 17.000 | 16.922 | +0.078 | 17.078 | yes | 17.078 |  |
| 4 | ` intersection` | 16.703 | 16.406 | +0.297 | 17.000 | yes | 17.000 |  |
| 5 | ` right` | 16.422 | 16.281 | +0.141 | 16.562 | yes | 16.562 |  |
| 6 | ` park` | 16.094 | 16.109 | -0.016 | 16.078 | no | —(rejected) |  |
| 7 | ` left` | 15.938 | 15.945 | -0.008 | 15.930 | no | —(rejected) |  |
| 8 | ` middle` | 15.578 | 15.469 | +0.109 | 15.688 | no | —(rejected) |  |
| 9 | ` edge` | 15.578 | 15.352 | +0.227 | 15.805 | no | —(rejected) |  |
| 10 | ` cars` | 15.359 | 15.188 | +0.172 | 15.531 | no | —(rejected) |  |


### image 724 — word "stop sign" (node: stop sign) — step 107

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sign` | 27.047 | 27.016 | +0.031 | 27.078 | yes | 27.078 | **YES** |
| 2 | ` signs` | 17.578 | 17.328 | +0.250 | 17.828 | no | —(rejected) |  |
| 3 | ` signal` | 16.641 | 16.094 | +0.547 | 17.188 | no | —(rejected) |  |
| 4 | `-` | 16.328 | 15.812 | +0.516 | 16.844 | no | —(rejected) |  |
| 5 | `.` | 15.727 | 15.195 | +0.531 | 16.258 | no | —(rejected) |  |
| 6 | ` pole` | 14.344 | 13.883 | +0.461 | 14.805 | no | —(rejected) |  |
| 7 | ` and` | 14.281 | 13.828 | +0.453 | 14.734 | no | —(rejected) |  |
| 8 | ` s` | 14.250 | 13.758 | +0.492 | 14.742 | no | —(rejected) |  |
| 9 | ` intersection` | 14.172 | 13.672 | +0.500 | 14.672 | no | —(rejected) |  |
| 10 | `light` | 14.109 | 13.383 | +0.727 | 14.836 | no | —(rejected) |  |


### image 2473 — word "skier" (node: person) — step 10

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 21.156 | 21.453 | -0.297 | 20.859 | yes | 20.859 |  |
| 2 | ` sk` | 21.125 | 21.281 | -0.156 | 20.969 | yes | 20.969 | **YES** |
| 3 | ` man` | 20.547 | 20.625 | -0.078 | 20.469 | yes | 20.469 |  |
| 4 | ` snow` | 20.531 | 21.000 | -0.469 | 20.062 | yes | 20.062 |  |
| 5 | ` ski` | 18.719 | 18.375 | +0.344 | 19.062 | no | —(rejected) |  |
| 6 | ` young` | 18.234 | 18.703 | -0.469 | 17.766 | no | —(rejected) |  |
| 7 | ` male` | 17.734 | 17.547 | +0.188 | 17.922 | no | —(rejected) |  |
| 8 | ` winter` | 16.578 | 16.484 | +0.094 | 16.672 | no | —(rejected) |  |
| 9 | ` professional` | 16.422 | 15.922 | +0.500 | 16.922 | no | —(rejected) |  |
| 10 | ` fre` | 16.266 | 15.938 | +0.328 | 16.594 | no | —(rejected) |  |


### image 2473 — word "skier" (node: person) — step 11

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ier` | 24.547 | 24.672 | -0.125 | 24.422 | yes | 24.422 | **YES** |
| 2 | `ate` | 19.562 | 21.109 | -1.547 | 18.016 | no | —(rejected) |  |
| 3 | `illed` | 19.562 | 20.312 | -0.750 | 18.812 | no | —(rejected) |  |
| 4 | `ater` | 17.250 | 18.422 | -1.172 | 16.078 | no | —(rejected) |  |
| 5 | `is` | 16.734 | 17.000 | -0.266 | 16.469 | no | —(rejected) |  |
| 6 | `ied` | 14.914 | 14.719 | +0.195 | 15.109 | no | —(rejected) |  |
| 7 | `yd` | 14.633 | 16.469 | -1.836 | 12.797 | no | —(rejected) |  |
| 8 | `il` | 14.086 | 14.141 | -0.055 | 14.031 | no | —(rejected) |  |
| 9 | `ij` | 13.961 | 14.422 | -0.461 | 13.500 | no | —(rejected) |  |
| 10 | `ating` | 13.945 | 14.711 | -0.766 | 13.180 | no | —(rejected) |  |


### image 2473 — word "skis" (node: skis) — step 21

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sk` | 21.578 | 20.188 | +1.391 | 22.969 | yes | 22.969 | **YES** |
| 2 | ` legs` | 17.844 | 17.906 | -0.062 | 17.781 | no | —(rejected) |  |
| 3 | ` snow` | 17.672 | 17.562 | +0.109 | 17.781 | no | —(rejected) |  |
| 4 | ` body` | 17.625 | 17.438 | +0.188 | 17.812 | no | —(rejected) |  |
| 5 | ` ski` | 17.500 | 16.141 | +1.359 | 18.859 | no | —(rejected) |  |
| 6 | ` feet` | 16.406 | 16.609 | -0.203 | 16.203 | no | —(rejected) |  |
| 7 | ` arms` | 16.266 | 16.625 | -0.359 | 15.906 | no | —(rejected) |  |
| 8 | ` pol` | 15.320 | 13.102 | +2.219 | 17.539 | no | —(rejected) |  |
| 9 | ` two` | 14.867 | 12.617 | +2.250 | 17.117 | no | —(rejected) |  |
| 10 | ` long` | 14.844 | 13.344 | +1.500 | 16.344 | no | —(rejected) |  |


### image 2473 — word "skis" (node: skis) — step 22

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `is` | 26.359 | 26.062 | +0.297 | 26.656 | yes | 26.656 | **YES** |
| 2 | `ies` | 19.062 | 19.000 | +0.062 | 19.125 | no | —(rejected) |  |
| 3 | `ate` | 16.609 | 17.703 | -1.094 | 15.516 | no | —(rejected) |  |
| 4 | `ates` | 16.172 | 16.594 | -0.422 | 15.750 | no | —(rejected) |  |
| 5 | `ier` | 15.391 | 15.898 | -0.508 | 14.883 | no | —(rejected) |  |
| 6 | `ib` | 14.484 | 14.609 | -0.125 | 14.359 | no | —(rejected) |  |
| 7 | `ating` | 14.461 | 14.188 | +0.273 | 14.734 | no | —(rejected) |  |
| 8 | `ied` | 14.359 | 14.391 | -0.031 | 14.328 | no | —(rejected) |  |
| 9 | `id` | 13.570 | 13.672 | -0.102 | 13.469 | no | —(rejected) |  |
| 10 | `ids` | 13.398 | 12.961 | +0.438 | 13.836 | no | —(rejected) |  |


### image 2473 — word "skier" (node: person) — step 33

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sk` | 18.844 | 18.859 | -0.016 | 18.828 | yes | 18.828 | **YES** |
| 2 | ` person` | 15.953 | 15.875 | +0.078 | 16.031 | no | —(rejected) |  |
| 3 | ` scene` | 15.641 | 15.602 | +0.039 | 15.680 | no | —(rejected) |  |
| 4 | ` main` | 15.211 | 15.422 | -0.211 | 15.000 | no | —(rejected) |  |
| 5 | ` j` | 14.344 | 13.938 | +0.406 | 14.750 | no | —(rejected) |  |
| 6 | ` snow` | 14.242 | 14.695 | -0.453 | 13.789 | no | —(rejected) |  |
| 7 | ` ski` | 14.039 | 13.875 | +0.164 | 14.203 | no | —(rejected) |  |
| 8 | ` man` | 13.719 | 12.773 | +0.945 | 14.664 | no | —(rejected) |  |
| 9 | ` jump` | 13.648 | 13.352 | +0.297 | 13.945 | no | —(rejected) |  |
| 10 | ` background` | 13.594 | 13.820 | -0.227 | 13.367 | no | —(rejected) |  |


### image 2473 — word "skier" (node: person) — step 34

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ier` | 24.312 | 24.391 | -0.078 | 24.234 | yes | 24.234 | **YES** |
| 2 | `is` | 18.547 | 18.562 | -0.016 | 18.531 | no | —(rejected) |  |
| 3 | `illed` | 17.422 | 17.359 | +0.062 | 17.484 | no | —(rejected) |  |
| 4 | `ater` | 16.625 | 16.562 | +0.062 | 16.688 | no | —(rejected) |  |
| 5 | `ate` | 16.359 | 16.344 | +0.016 | 16.375 | no | —(rejected) |  |
| 6 | `ies` | 15.820 | 15.836 | -0.016 | 15.805 | no | —(rejected) |  |
| 7 | `ied` | 14.859 | 14.734 | +0.125 | 14.984 | no | —(rejected) |  |
| 8 | `il` | 13.305 | 13.359 | -0.055 | 13.250 | no | —(rejected) |  |
| 9 | `ie` | 13.188 | 13.047 | +0.141 | 13.328 | no | —(rejected) |  |
| 10 | `yer` | 12.734 | 12.477 | +0.258 | 12.992 | no | —(rejected) |  |


### image 2473 — word "man" (node: person) — step 65

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 19.203 | 18.625 | +0.578 | 19.781 | yes | 19.781 |  |
| 2 | ` group` | 19.016 | 18.188 | +0.828 | 19.844 | yes | 19.844 |  |
| 3 | ` man` | 18.938 | 18.016 | +0.922 | 19.859 | yes | 19.859 | **YES** |
| 4 | ` couple` | 17.453 | 16.891 | +0.562 | 18.016 | no | —(rejected) |  |
| 5 | ` few` | 17.375 | 16.922 | +0.453 | 17.828 | no | —(rejected) |  |
| 6 | ` sk` | 16.172 | 15.992 | +0.180 | 16.352 | no | —(rejected) |  |
| 7 | ` young` | 16.156 | 14.773 | +1.383 | 17.539 | no | —(rejected) |  |
| 8 | ` small` | 16.000 | 15.398 | +0.602 | 16.602 | no | —(rejected) |  |
| 9 | ` pair` | 15.641 | 14.648 | +0.992 | 16.633 | no | —(rejected) |  |
| 10 | ` snow` | 15.586 | 16.172 | -0.586 | 15.000 | no | —(rejected) |  |


### image 2473 — word "skier" (node: person) — step 75

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sk` | 19.297 | 19.578 | -0.281 | 19.016 | yes | 19.016 | **YES** |
| 2 | ` jump` | 16.891 | 17.031 | -0.141 | 16.750 | no | —(rejected) |  |
| 3 | ` action` | 16.578 | 16.766 | -0.188 | 16.391 | no | —(rejected) |  |
| 4 | ` j` | 16.141 | 16.188 | -0.047 | 16.094 | no | —(rejected) |  |
| 5 | ` main` | 15.594 | 15.758 | -0.164 | 15.430 | no | —(rejected) |  |
| 6 | ` ski` | 15.195 | 15.250 | -0.055 | 15.141 | no | —(rejected) |  |
| 7 | ` thr` | 14.961 | 15.008 | -0.047 | 14.914 | no | —(rejected) |  |
| 8 | ` performance` | 14.719 | 14.969 | -0.250 | 14.469 | no | —(rejected) |  |
| 9 | ` aer` | 14.719 | 14.758 | -0.039 | 14.680 | no | —(rejected) |  |
| 10 | ` air` | 14.633 | 14.609 | +0.023 | 14.656 | no | —(rejected) |  |


### image 2473 — word "skier" (node: person) — step 76

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `ier` | 25.109 | 25.422 | -0.312 | 24.797 | yes | 24.797 | **YES** |
| 2 | `illed` | 19.875 | 19.828 | +0.047 | 19.922 | no | —(rejected) |  |
| 3 | `is` | 17.109 | 17.234 | -0.125 | 16.984 | no | —(rejected) |  |
| 4 | `ied` | 16.172 | 16.141 | +0.031 | 16.203 | no | —(rejected) |  |
| 5 | `ate` | 15.461 | 15.000 | +0.461 | 15.922 | no | —(rejected) |  |
| 6 | `ater` | 15.258 | 14.906 | +0.352 | 15.609 | no | —(rejected) |  |
| 7 | `il` | 14.805 | 14.977 | -0.172 | 14.633 | no | —(rejected) |  |
| 8 | `ies` | 13.961 | 14.000 | -0.039 | 13.922 | no | —(rejected) |  |
| 9 | `ie` | 13.375 | 13.383 | -0.008 | 13.367 | no | —(rejected) |  |
| 10 | `immer` | 13.375 | 13.266 | +0.109 | 13.484 | no | —(rejected) |  |


### image 2473 — word "people" (node: person) — step 82

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 22.047 | 22.188 | -0.141 | 21.906 | yes | 21.906 | **YES** |
| 2 | ` individuals` | 20.359 | 20.453 | -0.094 | 20.266 | no | —(rejected) |  |
| 3 | ` spect` | 19.406 | 19.562 | -0.156 | 19.250 | no | —(rejected) |  |
| 4 | ` obser` | 18.703 | 18.750 | -0.047 | 18.656 | no | —(rejected) |  |
| 5 | ` by` | 18.312 | 18.641 | -0.328 | 17.984 | no | —(rejected) |  |
| 6 | ` on` | 18.172 | 18.406 | -0.234 | 17.938 | no | —(rejected) |  |
| 7 | ` view` | 17.656 | 17.625 | +0.031 | 17.688 | no | —(rejected) |  |
| 8 | ` ski` | 17.016 | 17.500 | -0.484 | 16.531 | no | —(rejected) |  |
| 9 | ` persons` | 16.188 | 16.031 | +0.156 | 16.344 | no | —(rejected) |  |
| 10 | ` sk` | 15.789 | 16.172 | -0.383 | 15.406 | no | —(rejected) |  |


### image 5529 — word "man" (node: person) — step 5

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 19.906 | 19.734 | +0.172 | 20.078 | yes | 20.078 |  |
| 2 | ` man` | 19.891 | 19.438 | +0.453 | 20.344 | yes | 20.344 | **YES** |
| 3 | ` snow` | 19.516 | 18.719 | +0.797 | 20.312 | yes | 20.312 |  |
| 4 | ` sk` | 18.234 | 17.969 | +0.266 | 18.500 | no | —(rejected) |  |
| 5 | ` ski` | 17.156 | 16.531 | +0.625 | 17.781 | no | —(rejected) |  |
| 6 | ` winter` | 17.031 | 16.922 | +0.109 | 17.141 | no | —(rejected) |  |
| 7 | ` l` | 16.703 | 16.672 | +0.031 | 16.734 | no | —(rejected) |  |
| 8 | ` male` | 16.453 | 15.609 | +0.844 | 17.297 | no | —(rejected) |  |
| 9 | ` scene` | 16.156 | 15.977 | +0.180 | 16.336 | no | —(rejected) |  |
| 10 | ` thr` | 16.000 | 16.219 | -0.219 | 15.781 | no | —(rejected) |  |


### image 5529 — word "man" (node: person) — step 44

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 20.188 | 19.125 | +1.062 | 21.250 | yes | 21.250 | **YES** |
| 2 | ` sk` | 19.859 | 18.797 | +1.062 | 20.922 | yes | 20.922 |  |
| 3 | ` snow` | 17.672 | 17.500 | +0.172 | 17.844 | no | —(rejected) |  |
| 4 | ` ski` | 17.031 | 16.359 | +0.672 | 17.703 | no | —(rejected) |  |
| 5 | ` slope` | 16.984 | 17.266 | -0.281 | 16.703 | no | —(rejected) |  |
| 6 | ` person` | 16.844 | 15.273 | +1.570 | 18.414 | no | —(rejected) |  |
| 7 | ` scene` | 16.609 | 16.578 | +0.031 | 16.641 | no | —(rejected) |  |
| 8 | ` trees` | 14.984 | 15.031 | -0.047 | 14.938 | no | —(rejected) |  |
| 9 | ` mountain` | 14.273 | 13.977 | +0.297 | 14.570 | no | —(rejected) |  |
| 10 | ` main` | 14.227 | 13.172 | +1.055 | 15.281 | no | —(rejected) |  |


### image 5529 — word "skis" (node: skis) — step 65

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` scene` | 17.375 | 17.500 | -0.125 | 17.250 | yes | 17.250 |  |
| 2 | ` sk` | 17.297 | 15.836 | +1.461 | 18.758 | yes | 18.758 | **YES** |
| 3 | ` ski` | 16.953 | 16.266 | +0.688 | 17.641 | yes | 17.641 |  |
| 4 | ` snow` | 16.250 | 16.125 | +0.125 | 16.375 | yes | 16.375 |  |
| 5 | ` man` | 15.695 | 13.984 | +1.711 | 17.406 | no | —(rejected) |  |
| 6 | ` slope` | 15.453 | 15.250 | +0.203 | 15.656 | no | —(rejected) |  |
| 7 | ` image` | 15.203 | 14.750 | +0.453 | 15.656 | no | —(rejected) |  |
| 8 | ` view` | 14.047 | 13.820 | +0.227 | 14.273 | no | —(rejected) |  |
| 9 | ` environment` | 13.594 | 13.883 | -0.289 | 13.305 | no | —(rejected) |  |
| 10 | ` trees` | 13.594 | 13.711 | -0.117 | 13.477 | no | —(rejected) |  |


### image 5529 — word "skis" (node: skis) — step 66

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `is` | 25.547 | 25.797 | -0.250 | 25.297 | yes | 25.297 | **YES** |
| 2 | `ier` | 24.297 | 24.281 | +0.016 | 24.312 | yes | 24.312 |  |
| 3 | `ies` | 20.844 | 21.141 | -0.297 | 20.547 | no | —(rejected) |  |
| 4 | `ate` | 16.250 | 16.344 | -0.094 | 16.156 | no | —(rejected) |  |
| 5 | `ib` | 15.852 | 15.508 | +0.344 | 16.195 | no | —(rejected) |  |
| 6 | `ied` | 15.547 | 14.977 | +0.570 | 16.117 | no | —(rejected) |  |
| 7 | `ir` | 15.523 | 15.328 | +0.195 | 15.719 | no | —(rejected) |  |
| 8 | `id` | 15.180 | 14.836 | +0.344 | 15.523 | no | —(rejected) |  |
| 9 | `ie` | 14.961 | 14.719 | +0.242 | 15.203 | no | —(rejected) |  |
| 10 | `ing` | 14.828 | 14.422 | +0.406 | 15.234 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 14

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 22.281 | 22.469 | -0.188 | 22.094 | yes | 22.094 | **YES** |
| 2 | ` tour` | 19.328 | 19.359 | -0.031 | 19.297 | no | —(rejected) |  |
| 3 | ` passenger` | 17.125 | 16.672 | +0.453 | 17.578 | no | —(rejected) |  |
| 4 | ` city` | 16.875 | 17.281 | -0.406 | 16.469 | no | —(rejected) |  |
| 5 | ` London` | 16.406 | 15.695 | +0.711 | 17.117 | no | —(rejected) |  |
| 6 | ` sight` | 16.391 | 16.781 | -0.391 | 16.000 | no | —(rejected) |  |
| 7 | ` double` | 16.234 | 15.492 | +0.742 | 16.977 | no | —(rejected) |  |
| 8 | ` public` | 15.359 | 16.203 | -0.844 | 14.516 | no | —(rejected) |  |
| 9 | ` trans` | 14.820 | 14.922 | -0.102 | 14.719 | no | —(rejected) |  |
| 10 | ` red` | 14.578 | 13.461 | +1.117 | 15.695 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 21

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bus` | 21.047 | 21.000 | +0.047 | 21.094 | yes | 21.094 | **YES** |
| 2 | ` large` | 17.531 | 17.406 | +0.125 | 17.656 | no | —(rejected) |  |
| 3 | ` double` | 17.000 | 16.625 | +0.375 | 17.375 | no | —(rejected) |  |
| 4 | ` red` | 16.781 | 14.352 | +2.430 | 19.211 | no | —(rejected) |  |
| 5 | ` top` | 16.391 | 17.281 | -0.891 | 15.500 | no | —(rejected) |  |
| 6 | ` street` | 16.312 | 16.594 | -0.281 | 16.031 | no | —(rejected) |  |
| 7 | ` icon` | 16.000 | 15.508 | +0.492 | 16.492 | no | —(rejected) |  |
| 8 | ` first` | 15.969 | 16.094 | -0.125 | 15.844 | no | —(rejected) |  |
| 9 | ` two` | 15.734 | 15.727 | +0.008 | 15.742 | no | —(rejected) |  |
| 10 | ` city` | 15.203 | 15.422 | -0.219 | 14.984 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 46

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` car` | 19.500 | 20.484 | -0.984 | 18.516 | yes | 18.516 |  |
| 2 | ` bus` | 19.359 | 17.750 | +1.609 | 20.969 | yes | 20.969 | **YES** |
| 3 | ` tr` | 19.297 | 20.500 | -1.203 | 18.094 | yes | 18.094 |  |
| 4 | ` few` | 19.203 | 19.766 | -0.562 | 18.641 | yes | 18.641 |  |
| 5 | ` couple` | 19.109 | 19.328 | -0.219 | 18.891 | yes | 18.891 |  |
| 6 | ` smaller` | 18.422 | 18.422 | +0.000 | 18.422 | yes | 18.422 |  |
| 7 | ` small` | 16.875 | 16.719 | +0.156 | 17.031 | no | —(rejected) |  |
| 8 | ` white` | 16.750 | 15.000 | +1.750 | 18.500 | no | —(rejected) |  |
| 9 | ` second` | 16.625 | 16.297 | +0.328 | 16.953 | no | —(rejected) |  |
| 10 | ` mix` | 16.562 | 17.797 | -1.234 | 15.328 | no | —(rejected) |  |


### image 1584 — word "people" (node: person) — step 68

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 24.719 | 24.297 | +0.422 | 25.141 | yes | 25.141 | **YES** |
| 2 | ` ped` | 24.531 | 24.250 | +0.281 | 24.812 | yes | 24.812 |  |
| 3 | ` cars` | 18.969 | 19.141 | -0.172 | 18.797 | no | —(rejected) |  |
| 4 | ` individuals` | 18.641 | 18.188 | +0.453 | 19.094 | no | —(rejected) |  |
| 5 | ` passengers` | 18.578 | 18.125 | +0.453 | 19.031 | no | —(rejected) |  |
| 6 | ` traffic` | 18.031 | 18.109 | -0.078 | 17.953 | no | —(rejected) |  |
| 7 | ` vehicles` | 16.750 | 16.719 | +0.031 | 16.781 | no | —(rejected) |  |
| 8 | ` b` | 16.672 | 16.609 | +0.062 | 16.734 | no | —(rejected) |  |
| 9 | ` other` | 16.344 | 16.219 | +0.125 | 16.469 | no | —(rejected) |  |
| 10 | ` pass` | 16.172 | 15.516 | +0.656 | 16.828 | no | —(rejected) |  |


### image 1584 — word "bus" (node: bus) — step 103

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` for` | 21.203 | 21.578 | -0.375 | 20.828 | yes | 20.828 |  |
| 2 | ` bus` | 21.141 | 21.062 | +0.078 | 21.219 | yes | 21.219 | **YES** |
| 3 | ` front` | 19.891 | 20.203 | -0.312 | 19.578 | yes | 19.578 |  |
| 4 | ` camera` | 19.031 | 19.297 | -0.266 | 18.766 | no | —(rejected) |  |
| 5 | ` left` | 18.875 | 18.812 | +0.062 | 18.938 | no | —(rejected) |  |
| 6 | ` edge` | 18.203 | 17.938 | +0.266 | 18.469 | no | —(rejected) |  |
| 7 | ` center` | 18.094 | 18.516 | -0.422 | 17.672 | no | —(rejected) |  |
| 8 | ` road` | 17.391 | 17.234 | +0.156 | 17.547 | no | —(rejected) |  |
| 9 | ` vehicles` | 17.328 | 17.797 | -0.469 | 16.859 | no | —(rejected) |  |
| 10 | ` middle` | 17.234 | 17.641 | -0.406 | 16.828 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 7

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 17.422 | 17.062 | +0.359 | 17.781 | yes | 17.781 | **YES** |
| 2 | ` passenger` | 16.672 | 16.219 | +0.453 | 17.125 | yes | 17.125 |  |
| 3 | ` comm` | 16.203 | 15.484 | +0.719 | 16.922 | yes | 16.922 |  |
| 4 | ` sub` | 15.039 | 14.320 | +0.719 | 15.758 | no | —(rejected) |  |
| 5 | ` electric` | 14.867 | 13.305 | +1.562 | 16.430 | no | —(rejected) |  |
| 6 | ` met` | 14.820 | 13.703 | +1.117 | 15.938 | no | —(rejected) |  |
| 7 | ` light` | 13.984 | 13.172 | +0.812 | 14.797 | no | —(rejected) |  |
| 8 | ` city` | 13.656 | 13.055 | +0.602 | 14.258 | no | —(rejected) |  |
| 9 | ` public` | 13.648 | 12.836 | +0.812 | 14.461 | no | —(rejected) |  |
| 10 | ` tram` | 13.016 | 12.852 | +0.164 | 13.180 | no | —(rejected) |  |


### image 6040 — word "people" (node: person) — step 16

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 18.125 | 15.484 | +2.641 | 20.766 | yes | 20.766 | **YES** |
| 2 | ` passengers` | 17.734 | 14.766 | +2.969 | 20.703 | yes | 20.703 |  |
| 3 | ` cars` | 16.047 | 14.883 | +1.164 | 17.211 | no | —(rejected) |  |
| 4 | ` windows` | 15.492 | 12.984 | +2.508 | 18.000 | no | —(rejected) |  |
| 5 | ` train` | 13.773 | 11.516 | +2.258 | 16.031 | no | —(rejected) |  |
| 6 | ` commut` | 13.516 | 10.258 | +3.258 | 16.773 | no | —(rejected) |  |
| 7 | ` open` | 13.406 | 10.227 | +3.180 | 16.586 | no | —(rejected) |  |
| 8 | ` individuals` | 13.227 | 10.703 | +2.523 | 15.750 | no | —(rejected) |  |
| 9 | ` passenger` | 13.125 | 11.391 | +1.734 | 14.859 | no | —(rejected) |  |
| 10 | ` large` | 13.062 | 11.156 | +1.906 | 14.969 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 22

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 20.016 | 19.328 | +0.688 | 20.703 | yes | 20.703 | **YES** |
| 2 | ` vehicle` | 16.938 | 16.125 | +0.812 | 17.750 | no | —(rejected) |  |
| 3 | ` car` | 15.102 | 14.758 | +0.344 | 15.445 | no | —(rejected) |  |
| 4 | ` passenger` | 14.289 | 13.500 | +0.789 | 15.078 | no | —(rejected) |  |
| 5 | ` area` | 13.719 | 12.938 | +0.781 | 14.500 | no | —(rejected) |  |
| 6 | ` station` | 13.695 | 13.062 | +0.633 | 14.328 | no | —(rejected) |  |
| 7 | ` sub` | 13.648 | 12.117 | +1.531 | 15.180 | no | —(rejected) |  |
| 8 | ` comm` | 13.375 | 11.883 | +1.492 | 14.867 | no | —(rejected) |  |
| 9 | ` rail` | 12.977 | 12.375 | +0.602 | 13.578 | no | —(rejected) |  |
| 10 | ` platform` | 12.922 | 11.875 | +1.047 | 13.969 | no | —(rejected) |  |


### image 6040 — word "people" (node: person) — step 31

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 21.391 | 21.766 | -0.375 | 21.016 | yes | 21.016 | **YES** |
| 2 | ` passengers` | 20.750 | 20.906 | -0.156 | 20.594 | yes | 20.594 |  |
| 3 | ` individuals` | 20.016 | 20.312 | -0.297 | 19.719 | yes | 19.719 |  |
| 4 | ` visible` | 18.219 | 18.250 | -0.031 | 18.188 | no | —(rejected) |  |
| 5 | ` persons` | 16.328 | 16.562 | -0.234 | 16.094 | no | —(rejected) |  |
| 6 | ` commut` | 15.500 | 15.172 | +0.328 | 15.828 | no | —(rejected) |  |
| 7 | ` on` | 14.922 | 14.344 | +0.578 | 15.500 | no | —(rejected) |  |
| 8 | ` train` | 14.781 | 15.109 | -0.328 | 14.453 | no | —(rejected) |  |
| 9 | ` different` | 14.727 | 15.016 | -0.289 | 14.438 | no | —(rejected) |  |
| 10 | ` distinct` | 14.578 | 15.219 | -0.641 | 13.938 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 52

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 19.188 | 19.609 | -0.422 | 18.766 | yes | 18.766 | **YES** |
| 2 | ` passengers` | 15.297 | 15.164 | +0.133 | 15.430 | no | —(rejected) |  |
| 3 | ` people` | 14.195 | 14.188 | +0.008 | 14.203 | no | —(rejected) |  |
| 4 | ` scene` | 13.336 | 13.891 | -0.555 | 12.781 | no | —(rejected) |  |
| 5 | ` blue` | 13.289 | 13.836 | -0.547 | 12.742 | no | —(rejected) |  |
| 6 | ` view` | 13.211 | 13.539 | -0.328 | 12.883 | no | —(rejected) |  |
| 7 | ` majority` | 12.727 | 12.758 | -0.031 | 12.695 | no | —(rejected) |  |
| 8 | ` image` | 12.500 | 12.984 | -0.484 | 12.016 | no | —(rejected) |  |
| 9 | ` setting` | 12.430 | 13.117 | -0.688 | 11.742 | no | —(rejected) |  |
| 10 | ` first` | 12.367 | 12.477 | -0.109 | 12.258 | no | —(rejected) |  |


### image 6040 — word "people" (node: person) — step 67

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 15.055 | 15.031 | +0.023 | 15.078 | yes | 15.078 | **YES** |
| 2 | ` train` | 14.875 | 14.695 | +0.180 | 15.055 | yes | 15.055 |  |
| 3 | ` passengers` | 14.531 | 14.320 | +0.211 | 14.742 | yes | 14.742 |  |
| 4 | ` tracks` | 13.578 | 13.344 | +0.234 | 13.812 | yes | 13.812 |  |
| 5 | ` scene` | 13.086 | 12.938 | +0.148 | 13.234 | no | —(rejected) |  |
| 6 | ` presence` | 12.914 | 12.781 | +0.133 | 13.047 | no | —(rejected) |  |
| 7 | ` blue` | 12.281 | 11.297 | +0.984 | 13.266 | no | —(rejected) |  |
| 8 | ` sky` | 12.039 | 11.531 | +0.508 | 12.547 | no | —(rejected) |  |
| 9 | ` surrounding` | 11.820 | 11.766 | +0.055 | 11.875 | no | —(rejected) |  |
| 10 | ` view` | 11.742 | 11.273 | +0.469 | 12.211 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 72

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 19.938 | 20.000 | -0.062 | 19.875 | yes | 19.875 | **YES** |
| 2 | ` scene` | 18.047 | 18.359 | -0.312 | 17.734 | no | —(rejected) |  |
| 3 | ` image` | 15.305 | 15.328 | -0.023 | 15.281 | no | —(rejected) |  |
| 4 | ` interior` | 15.109 | 15.203 | -0.094 | 15.016 | no | —(rejected) |  |
| 5 | ` vehicle` | 15.055 | 14.961 | +0.094 | 15.148 | no | —(rejected) |  |
| 6 | ` area` | 14.656 | 14.688 | -0.031 | 14.625 | no | —(rejected) |  |
| 7 | ` car` | 14.453 | 14.336 | +0.117 | 14.570 | no | —(rejected) |  |
| 8 | ` picture` | 14.281 | 14.430 | -0.148 | 14.133 | no | —(rejected) |  |
| 9 | ` space` | 14.062 | 14.227 | -0.164 | 13.898 | no | —(rejected) |  |
| 10 | ` frame` | 13.797 | 14.156 | -0.359 | 13.438 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 88

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 16.156 | 15.836 | +0.320 | 16.477 | yes | 16.477 | **YES** |
| 2 | ` scene` | 14.906 | 14.914 | -0.008 | 14.898 | yes | 14.898 |  |
| 3 | ` overall` | 14.430 | 14.633 | -0.203 | 14.227 | no | —(rejected) |  |
| 4 | ` passengers` | 13.906 | 14.008 | -0.102 | 13.805 | no | —(rejected) |  |
| 5 | ` atmosphere` | 13.500 | 13.602 | -0.102 | 13.398 | no | —(rejected) |  |
| 6 | ` presence` | 13.359 | 13.336 | +0.023 | 13.383 | no | —(rejected) |  |
| 7 | ` view` | 12.531 | 12.344 | +0.188 | 12.719 | no | —(rejected) |  |
| 8 | ` combination` | 12.523 | 12.461 | +0.062 | 12.586 | no | —(rejected) |  |
| 9 | ` majority` | 12.336 | 12.531 | -0.195 | 12.141 | no | —(rejected) |  |
| 10 | ` image` | 12.273 | 12.336 | -0.062 | 12.211 | no | —(rejected) |  |


### image 6040 — word "train" (node: train) — step 95

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` train` | 22.109 | 22.734 | -0.625 | 21.484 | yes | 21.484 | **YES** |
| 2 | ` or` | 19.719 | 20.391 | -0.672 | 19.047 | no | —(rejected) |  |
| 3 | ` rail` | 17.578 | 17.844 | -0.266 | 17.312 | no | —(rejected) |  |
| 4 | ` vehicle` | 17.484 | 17.531 | -0.047 | 17.438 | no | —(rejected) |  |
| 5 | ` car` | 17.312 | 17.234 | +0.078 | 17.391 | no | —(rejected) |  |
| 6 | `-` | 16.391 | 16.859 | -0.469 | 15.922 | no | —(rejected) |  |
| 7 | ` line` | 15.875 | 15.688 | +0.188 | 16.062 | no | —(rejected) |  |
| 8 | ` bus` | 15.766 | 15.789 | -0.023 | 15.742 | no | —(rejected) |  |
| 9 | ` trans` | 15.648 | 15.250 | +0.398 | 16.047 | no | —(rejected) |  |
| 10 | `,` | 15.547 | 15.812 | -0.266 | 15.281 | no | —(rejected) |  |


### image 6040 — word "passenger" (node: person) — step 102

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` passengers` | 17.469 | 17.766 | -0.297 | 17.172 | yes | 17.172 | **YES** |
| 2 | ` people` | 16.781 | 17.141 | -0.359 | 16.422 | yes | 16.422 |  |
| 3 | ` city` | 14.484 | 14.023 | +0.461 | 14.945 | no | —(rejected) |  |
| 4 | ` individuals` | 14.461 | 15.008 | -0.547 | 13.914 | no | —(rejected) |  |
| 5 | ` travel` | 14.008 | 14.242 | -0.234 | 13.773 | no | —(rejected) |  |
| 6 | ` commut` | 13.883 | 13.883 | +0.000 | 13.883 | no | —(rejected) |  |
| 7 | ` many` | 13.570 | 13.664 | -0.094 | 13.477 | no | —(rejected) |  |
| 8 | ` public` | 13.422 | 13.117 | +0.305 | 13.727 | no | —(rejected) |  |
| 9 | ` rid` | 13.312 | 13.328 | -0.016 | 13.297 | no | —(rejected) |  |
| 10 | ` residents` | 13.234 | 12.773 | +0.461 | 13.695 | no | —(rejected) |  |


### image 6763 — word "man" (node: person) — step 4

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 21.328 | 20.219 | +1.109 | 22.438 | yes | 22.438 | **YES** |
| 2 | ` couple` | 18.094 | 17.828 | +0.266 | 18.359 | no | —(rejected) |  |
| 3 | ` young` | 18.000 | 17.250 | +0.750 | 18.750 | no | —(rejected) |  |
| 4 | ` happy` | 17.672 | 16.844 | +0.828 | 18.500 | no | —(rejected) |  |
| 5 | ` sm` | 17.516 | 16.344 | +1.172 | 18.688 | no | —(rejected) |  |
| 6 | ` woman` | 15.250 | 15.609 | -0.359 | 14.891 | no | —(rejected) |  |
| 7 | ` well` | 15.000 | 13.492 | +1.508 | 16.508 | no | —(rejected) |  |
| 8 | ` beautiful` | 14.906 | 14.250 | +0.656 | 15.562 | no | —(rejected) |  |
| 9 | ` middle` | 14.805 | 14.188 | +0.617 | 15.422 | no | —(rejected) |  |
| 10 | ` hand` | 14.625 | 13.156 | +1.469 | 16.094 | no | —(rejected) |  |


### image 6763 — word "woman" (node: person) — step 7

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` woman` | 25.938 | 26.531 | -0.594 | 25.344 | yes | 25.344 | **YES** |
| 2 | ` beautiful` | 21.125 | 19.656 | +1.469 | 22.594 | no | —(rejected) |  |
| 3 | ` sm` | 20.453 | 20.484 | -0.031 | 20.422 | no | —(rejected) |  |
| 4 | ` young` | 19.812 | 19.172 | +0.641 | 20.453 | no | —(rejected) |  |
| 5 | ` lady` | 19.531 | 19.250 | +0.281 | 19.812 | no | —(rejected) |  |
| 6 | ` pretty` | 18.578 | 17.188 | +1.391 | 19.969 | no | —(rejected) |  |
| 7 | ` girl` | 18.562 | 17.922 | +0.641 | 19.203 | no | —(rejected) |  |
| 8 | ` women` | 18.484 | 18.484 | +0.000 | 18.484 | no | —(rejected) |  |
| 9 | ` pre` | 17.000 | 18.297 | -1.297 | 15.703 | no | —(rejected) |  |
| 10 | ` female` | 16.406 | 15.234 | +1.172 | 17.578 | no | —(rejected) |  |


### image 6763 — word "man" (node: person) — step 20

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 22.969 | 22.641 | +0.328 | 23.297 | yes | 23.297 | **YES** |
| 2 | ` woman` | 21.266 | 22.047 | -0.781 | 20.484 | no | —(rejected) |  |
| 3 | ` couple` | 20.000 | 20.203 | -0.203 | 19.797 | no | —(rejected) |  |
| 4 | ` young` | 18.172 | 17.562 | +0.609 | 18.781 | no | —(rejected) |  |
| 5 | ` sm` | 18.156 | 17.062 | +1.094 | 19.250 | no | —(rejected) |  |
| 6 | ` two` | 17.578 | 17.734 | -0.156 | 17.422 | no | —(rejected) |  |
| 7 | ` bar` | 17.156 | 18.109 | -0.953 | 16.203 | no | —(rejected) |  |
| 8 | ` lady` | 16.891 | 16.797 | +0.094 | 16.984 | no | —(rejected) |  |
| 9 | ` happy` | 16.719 | 16.328 | +0.391 | 17.109 | no | —(rejected) |  |
| 10 | ` pair` | 16.656 | 16.828 | -0.172 | 16.484 | no | —(rejected) |  |


### image 6763 — word "tie" (node: tie) — step 33

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` tie` | 16.656 | 16.656 | +0.000 | 16.656 | yes | 16.656 | **YES** |
| 2 | ` neck` | 15.250 | 15.258 | -0.008 | 15.242 | yes | 15.242 |  |
| 3 | `,` | 14.328 | 13.945 | +0.383 | 14.711 | no | —(rejected) |  |
| 4 | ` strip` | 14.156 | 14.055 | +0.102 | 14.258 | no | —(rejected) |  |
| 5 | ` bow` | 12.164 | 11.922 | +0.242 | 12.406 | no | —(rejected) |  |
| 6 | ` pattern` | 12.078 | 11.867 | +0.211 | 12.289 | no | —(rejected) |  |
| 7 | ` sil` | 11.797 | 11.062 | +0.734 | 12.531 | no | —(rejected) |  |
| 8 | ` and` | 11.406 | 10.914 | +0.492 | 11.898 | no | —(rejected) |  |
| 9 | ` kn` | 11.344 | 11.477 | -0.133 | 11.211 | no | —(rejected) |  |
| 10 | ` key` | 10.719 | 8.539 | +2.180 | 12.898 | no | —(rejected) |  |


### image 6763 — word "woman" (node: person) — step 37

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` woman` | 24.656 | 24.656 | +0.000 | 24.656 | yes | 24.656 | **YES** |
| 2 | ` lady` | 19.750 | 19.391 | +0.359 | 20.109 | no | —(rejected) |  |
| 3 | ` sm` | 18.672 | 17.641 | +1.031 | 19.703 | no | —(rejected) |  |
| 4 | ` young` | 18.391 | 17.500 | +0.891 | 19.281 | no | —(rejected) |  |
| 5 | ` girl` | 18.078 | 17.156 | +0.922 | 19.000 | no | —(rejected) |  |
| 6 | ` female` | 16.859 | 16.094 | +0.766 | 17.625 | no | —(rejected) |  |
| 7 | ` women` | 16.797 | 16.453 | +0.344 | 17.141 | no | —(rejected) |  |
| 8 | ` beautiful` | 16.719 | 15.422 | +1.297 | 18.016 | no | —(rejected) |  |
| 9 | ` attract` | 15.203 | 14.312 | +0.891 | 16.094 | no | —(rejected) |  |
| 10 | ` pretty` | 14.984 | 14.078 | +0.906 | 15.891 | no | —(rejected) |  |


### image 6763 — word "tv" (node: tv) — step 66

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` TV` | 20.953 | 20.656 | +0.297 | 21.250 | yes | 21.250 | **YES** |
| 2 | ` television` | 20.078 | 19.188 | +0.891 | 20.969 | yes | 20.969 |  |
| 3 | ` d` | 19.172 | 18.953 | +0.219 | 19.391 | no | —(rejected) |  |
| 4 | ` flat` | 18.000 | 16.453 | +1.547 | 19.547 | no | —(rejected) |  |
| 5 | ` large` | 17.391 | 15.836 | +1.555 | 18.945 | no | —(rejected) |  |
| 6 | ` small` | 16.625 | 16.016 | +0.609 | 17.234 | no | —(rejected) |  |
| 7 | ` person` | 16.594 | 16.125 | +0.469 | 17.062 | no | —(rejected) |  |
| 8 | ` c` | 16.125 | 17.125 | -1.000 | 15.125 | no | —(rejected) |  |
| 9 | ` group` | 16.000 | 15.172 | +0.828 | 16.828 | no | —(rejected) |  |
| 10 | ` bar` | 15.945 | 15.250 | +0.695 | 16.641 | no | —(rejected) |  |


### image 6763 — word "people" (node: person) — step 90

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` people` | 24.188 | 23.953 | +0.234 | 24.422 | yes | 24.422 | **YES** |
| 2 | ` individuals` | 20.500 | 20.250 | +0.250 | 20.750 | no | —(rejected) |  |
| 3 | ` pat` | 19.078 | 18.641 | +0.438 | 19.516 | no | —(rejected) |  |
| 4 | ` ch` | 18.734 | 18.594 | +0.141 | 18.875 | no | —(rejected) |  |
| 5 | ` persons` | 18.453 | 18.125 | +0.328 | 18.781 | no | —(rejected) |  |
| 6 | ` objects` | 17.312 | 17.203 | +0.109 | 17.422 | no | —(rejected) |  |
| 7 | ` guests` | 17.109 | 16.750 | +0.359 | 17.469 | no | —(rejected) |  |
| 8 | ` customers` | 17.031 | 17.047 | -0.016 | 17.016 | no | —(rejected) |  |
| 9 | ` items` | 16.672 | 16.844 | -0.172 | 16.500 | no | —(rejected) |  |
| 10 | ` bar` | 16.375 | 16.031 | +0.344 | 16.719 | no | —(rejected) |  |


### image 6763 — word "person" (node: person) — step 97

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 22.125 | 22.438 | -0.312 | 21.812 | yes | 21.812 | **YES** |
| 2 | ` standing` | 19.188 | 19.484 | -0.297 | 18.891 | no | —(rejected) |  |
| 3 | ` individual` | 18.766 | 19.312 | -0.547 | 18.219 | no | —(rejected) |  |
| 4 | ` on` | 17.859 | 17.875 | -0.016 | 17.844 | no | —(rejected) |  |
| 5 | ` of` | 17.812 | 17.906 | -0.094 | 17.719 | no | —(rejected) |  |
| 6 | ` near` | 17.672 | 17.812 | -0.141 | 17.531 | no | —(rejected) |  |
| 7 | ` located` | 17.625 | 17.797 | -0.172 | 17.453 | no | —(rejected) |  |
| 8 | ` sitting` | 17.469 | 16.984 | +0.484 | 17.953 | no | —(rejected) |  |
| 9 | ` close` | 17.453 | 17.625 | -0.172 | 17.281 | no | —(rejected) |  |
| 10 | ` man` | 17.188 | 17.219 | -0.031 | 17.156 | no | —(rejected) |  |


### image 6763 — word "person" (node: person) — step 108

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` person` | 21.453 | 21.672 | -0.219 | 21.234 | yes | 21.234 | **YES** |
| 2 | ` on` | 19.016 | 19.266 | -0.250 | 18.766 | no | —(rejected) |  |
| 3 | ` near` | 18.625 | 18.812 | -0.188 | 18.438 | no | —(rejected) |  |
| 4 | ` one` | 18.562 | 18.797 | -0.234 | 18.328 | no | —(rejected) |  |
| 5 | ` closer` | 18.547 | 18.797 | -0.250 | 18.297 | no | —(rejected) |  |
| 6 | ` towards` | 18.500 | 18.828 | -0.328 | 18.172 | no | —(rejected) |  |
| 7 | ` further` | 18.422 | 18.609 | -0.188 | 18.234 | no | —(rejected) |  |
| 8 | ` in` | 18.406 | 18.625 | -0.219 | 18.188 | no | —(rejected) |  |
| 9 | ` two` | 18.172 | 18.250 | -0.078 | 18.094 | no | —(rejected) |  |
| 10 | ` standing` | 18.094 | 18.531 | -0.438 | 17.656 | no | —(rejected) |  |


### image 6763 — word "cell phone" (node: cell phone) — step 125

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` hand` | 18.531 | 17.453 | +1.078 | 19.609 | yes | 19.609 |  |
| 2 | ` cell` | 18.484 | 16.391 | +2.094 | 20.578 | yes | 20.578 | **YES** |
| 3 | ` d` | 17.703 | 17.484 | +0.219 | 17.922 | yes | 17.922 |  |
| 4 | ` remote` | 17.141 | 13.844 | +3.297 | 20.438 | yes | 20.438 |  |
| 5 | ` clock` | 17.016 | 16.984 | +0.031 | 17.047 | yes | 17.047 |  |
| 6 | ` cup` | 16.672 | 16.094 | +0.578 | 17.250 | no | —(rejected) |  |
| 7 | ` tie` | 16.672 | 16.266 | +0.406 | 17.078 | no | —(rejected) |  |
| 8 | ` c` | 16.328 | 16.703 | -0.375 | 15.953 | no | —(rejected) |  |
| 9 | ` car` | 16.234 | 14.031 | +2.203 | 18.438 | no | —(rejected) |  |
| 10 | ` bow` | 16.219 | 15.117 | +1.102 | 17.320 | no | —(rejected) |  |


### image 6763 — word "cell phone" (node: cell phone) — step 126

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` phone` | 25.203 | 25.422 | -0.219 | 24.984 | yes | 24.984 | **YES** |
| 2 | `phone` | 21.094 | 20.906 | +0.188 | 21.281 | no | —(rejected) |  |
| 3 | `ular` | 17.125 | 17.000 | +0.125 | 17.250 | no | —(rejected) |  |
| 4 | `-` | 13.797 | 13.320 | +0.477 | 14.273 | no | —(rejected) |  |
| 5 | ` Phone` | 13.688 | 13.812 | -0.125 | 13.562 | no | —(rejected) |  |
| 6 | ` ph` | 12.953 | 13.031 | -0.078 | 12.875 | no | —(rejected) |  |
| 7 | ` or` | 12.805 | 11.352 | +1.453 | 14.258 | no | —(rejected) |  |
| 8 | ` device` | 12.742 | 11.773 | +0.969 | 13.711 | no | —(rejected) |  |
| 9 | ` tele` | 12.297 | 12.062 | +0.234 | 12.531 | no | —(rejected) |  |
| 10 | ` is` | 12.055 | 11.469 | +0.586 | 12.641 | no | —(rejected) |  |


### image 2261 — word "man" (node: person) — step 6

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` man` | 23.109 | 21.984 | +1.125 | 24.234 | yes | 24.234 | **YES** |
| 2 | ` boy` | 21.312 | 21.250 | +0.062 | 21.375 | no | —(rejected) |  |
| 3 | ` male` | 19.797 | 18.609 | +1.188 | 20.984 | no | —(rejected) |  |
| 4 | ` person` | 19.766 | 20.062 | -0.297 | 19.469 | no | —(rejected) |  |
| 5 | ` sur` | 19.688 | 19.594 | +0.094 | 19.781 | no | —(rejected) |  |
| 6 | ` individual` | 17.453 | 17.625 | -0.172 | 17.281 | no | —(rejected) |  |
| 7 | `,` | 17.359 | 16.969 | +0.391 | 17.750 | no | —(rejected) |  |
| 8 | ` child` | 17.250 | 18.453 | -1.203 | 16.047 | no | —(rejected) |  |
| 9 | ` adult` | 17.234 | 16.438 | +0.797 | 18.031 | no | —(rejected) |  |
| 10 | ` te` | 16.000 | 15.430 | +0.570 | 16.570 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 13

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sur` | 21.406 | 22.797 | -1.391 | 20.016 | yes | 20.016 | **YES** |
| 2 | ` blue` | 19.422 | 14.484 | +4.938 | 24.359 | no | —(rejected) |  |
| 3 | ` bo` | 18.812 | 19.000 | -0.188 | 18.625 | no | —(rejected) |  |
| 4 | ` body` | 18.469 | 18.219 | +0.250 | 18.719 | no | —(rejected) |  |
| 5 | ` small` | 17.250 | 17.672 | -0.422 | 16.828 | no | —(rejected) |  |
| 6 | ` board` | 16.641 | 17.203 | -0.562 | 16.078 | no | —(rejected) |  |
| 7 | ` wave` | 16.078 | 17.531 | -1.453 | 14.625 | no | —(rejected) |  |
| 8 | ` large` | 16.047 | 16.922 | -0.875 | 15.172 | no | —(rejected) |  |
| 9 | ` white` | 15.617 | 18.094 | -2.477 | 13.141 | no | —(rejected) |  |
| 10 | ` bright` | 15.461 | 13.656 | +1.805 | 17.266 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 14

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `f` | 27.203 | 28.312 | -1.109 | 26.094 | yes | 26.094 | **YES** |
| 2 | `board` | 17.625 | 17.766 | -0.141 | 17.484 | no | —(rejected) |  |
| 3 | `fer` | 16.938 | 17.156 | -0.219 | 16.719 | no | —(rejected) |  |
| 4 | `ft` | 14.000 | 13.922 | +0.078 | 14.078 | no | —(rejected) |  |
| 5 | `fers` | 13.500 | 13.836 | -0.336 | 13.164 | no | —(rejected) |  |
| 6 | `fo` | 12.867 | 12.430 | +0.438 | 13.305 | no | —(rejected) |  |
| 7 | `face` | 12.406 | 11.914 | +0.492 | 12.898 | no | —(rejected) |  |
| 8 | `fc` | 12.250 | 12.008 | +0.242 | 12.492 | no | —(rejected) |  |
| 9 | `fac` | 11.930 | 11.641 | +0.289 | 12.219 | no | —(rejected) |  |
| 10 | `fin` | 11.727 | 11.445 | +0.281 | 12.008 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 15

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 25.703 | 25.984 | -0.281 | 25.422 | yes | 25.422 | **YES** |
| 2 | ` board` | 17.656 | 16.594 | +1.062 | 18.719 | no | —(rejected) |  |
| 3 | `ing` | 16.141 | 15.945 | +0.195 | 16.336 | no | —(rejected) |  |
| 4 | `boards` | 14.977 | 14.680 | +0.297 | 15.273 | no | —(rejected) |  |
| 5 | `ba` | 13.773 | 12.875 | +0.898 | 14.672 | no | —(rejected) |  |
| 6 | `bo` | 13.039 | 11.812 | +1.227 | 14.266 | no | —(rejected) |  |
| 7 | `-` | 12.867 | 11.750 | +1.117 | 13.984 | no | —(rejected) |  |
| 8 | ` sur` | 12.742 | 11.680 | +1.062 | 13.805 | no | —(rejected) |  |
| 9 | `able` | 12.711 | 11.875 | +0.836 | 13.547 | no | —(rejected) |  |
| 10 | ` mat` | 12.039 | 10.094 | +1.945 | 13.984 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 28

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sur` | 20.328 | 20.578 | -0.250 | 20.078 | yes | 20.078 | **YES** |
| 2 | ` board` | 19.000 | 19.453 | -0.453 | 18.547 | yes | 18.547 |  |
| 3 | ` blue` | 17.547 | 13.383 | +4.164 | 21.711 | no | —(rejected) |  |
| 4 | ` wave` | 17.297 | 17.625 | -0.328 | 16.969 | no | —(rejected) |  |
| 5 | ` water` | 16.141 | 16.453 | -0.312 | 15.828 | no | —(rejected) |  |
| 6 | ` white` | 15.727 | 15.734 | -0.008 | 15.719 | no | —(rejected) |  |
| 7 | ` small` | 15.539 | 14.961 | +0.578 | 16.117 | no | —(rejected) |  |
| 8 | ` bo` | 15.148 | 14.602 | +0.547 | 15.695 | no | —(rejected) |  |
| 9 | ` top` | 14.898 | 14.883 | +0.016 | 14.914 | no | —(rejected) |  |
| 10 | ` large` | 14.570 | 14.531 | +0.039 | 14.609 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 29

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `f` | 26.141 | 27.172 | -1.031 | 25.109 | yes | 25.109 | **YES** |
| 2 | `board` | 16.875 | 16.578 | +0.297 | 17.172 | no | —(rejected) |  |
| 3 | `fer` | 16.688 | 17.031 | -0.344 | 16.344 | no | —(rejected) |  |
| 4 | `ft` | 13.742 | 13.531 | +0.211 | 13.953 | no | —(rejected) |  |
| 5 | `fo` | 13.602 | 13.219 | +0.383 | 13.984 | no | —(rejected) |  |
| 6 | `face` | 13.195 | 13.117 | +0.078 | 13.273 | no | —(rejected) |  |
| 7 | `fc` | 13.180 | 12.992 | +0.188 | 13.367 | no | —(rejected) |  |
| 8 | `ge` | 12.961 | 12.961 | +0.000 | 12.961 | no | —(rejected) |  |
| 9 | `fers` | 12.953 | 13.164 | -0.211 | 12.742 | no | —(rejected) |  |
| 10 | `fac` | 12.602 | 12.516 | +0.086 | 12.688 | no | —(rejected) |  |


### image 2261 — word "surfboard" (node: surfboard) — step 30

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `board` | 25.234 | 25.344 | -0.109 | 25.125 | yes | 25.125 | **YES** |
| 2 | `ing` | 16.172 | 15.359 | +0.812 | 16.984 | no | —(rejected) |  |
| 3 | ` board` | 16.031 | 14.930 | +1.102 | 17.133 | no | —(rejected) |  |
| 4 | `boards` | 14.758 | 14.258 | +0.500 | 15.258 | no | —(rejected) |  |
| 5 | `,` | 13.859 | 13.742 | +0.117 | 13.977 | no | —(rejected) |  |
| 6 | `acing` | 13.391 | 13.562 | -0.172 | 13.219 | no | —(rejected) |  |
| 7 | `ba` | 13.094 | 12.062 | +1.031 | 14.125 | no | —(rejected) |  |
| 8 | ` as` | 12.797 | 12.602 | +0.195 | 12.992 | no | —(rejected) |  |
| 9 | ` and` | 12.680 | 12.141 | +0.539 | 13.219 | no | —(rejected) |  |
| 10 | `bo` | 12.516 | 11.445 | +1.070 | 13.586 | no | —(rejected) |  |


### image 1425 — word "table" (node: dining table) — step 6

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` table` | 26.625 | 25.859 | +0.766 | 27.391 | yes | 27.391 | **YES** |
| 2 | ` setting` | 20.391 | 19.609 | +0.781 | 21.172 | no | —(rejected) |  |
| 3 | ` scene` | 20.141 | 19.875 | +0.266 | 20.406 | no | —(rejected) |  |
| 4 | ` room` | 19.453 | 19.328 | +0.125 | 19.578 | no | —(rejected) |  |
| 5 | ` area` | 18.922 | 19.562 | -0.641 | 18.281 | no | —(rejected) |  |
| 6 | ` setup` | 18.797 | 18.109 | +0.688 | 19.484 | no | —(rejected) |  |
| 7 | ` set` | 17.203 | 16.641 | +0.562 | 17.766 | no | —(rejected) |  |
| 8 | ` plate` | 17.172 | 17.016 | +0.156 | 17.328 | no | —(rejected) |  |
| 9 | ` d` | 17.016 | 17.172 | -0.156 | 16.859 | no | —(rejected) |  |
| 10 | ` experience` | 16.719 | 17.062 | -0.344 | 16.375 | no | —(rejected) |  |


### image 1425 — word "sandwich" (node: sandwich) — step 29

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` sand` | 13.969 | 13.828 | +0.141 | 14.109 | yes | 14.109 | **YES** |
| 2 | ` c` | 13.539 | 13.828 | -0.289 | 13.250 | yes | 13.250 |  |
| 3 | ` ch` | 13.398 | 13.422 | -0.023 | 13.375 | yes | 13.375 |  |
| 4 | ` m` | 12.953 | 12.812 | +0.141 | 13.094 | yes | 13.094 |  |
| 5 | ` brow` | 12.773 | 13.273 | -0.500 | 12.273 | yes | 12.273 |  |
| 6 | ` piece` | 12.578 | 13.234 | -0.656 | 11.922 | yes | 11.922 |  |
| 7 | ` cro` | 12.312 | 11.883 | +0.430 | 12.742 | no | —(rejected) |  |
| 8 | ` cre` | 12.141 | 12.523 | -0.383 | 11.758 | no | —(rejected) |  |
| 9 | ` don` | 12.133 | 12.578 | -0.445 | 11.688 | no | —(rejected) |  |
| 10 | ` dess` | 12.117 | 12.133 | -0.016 | 12.102 | no | —(rejected) |  |


### image 1425 — word "sandwich" (node: sandwich) — step 30

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `wich` | 20.672 | 20.844 | -0.172 | 20.500 | yes | 20.500 | **YES** |
| 2 | `which` | 13.531 | 13.805 | -0.273 | 13.258 | no | —(rejected) |  |
| 3 | ` d` | 12.727 | 12.812 | -0.086 | 12.641 | no | —(rejected) |  |
| 4 | `y` | 12.000 | 12.445 | -0.445 | 11.555 | no | —(rejected) |  |
| 5 | `-` | 11.758 | 11.805 | -0.047 | 11.711 | no | —(rejected) |  |
| 6 | ` which` | 11.016 | 11.180 | -0.164 | 10.852 | no | —(rejected) |  |
| 7 | `ed` | 10.766 | 10.875 | -0.109 | 10.656 | no | —(rejected) |  |
| 8 | `castle` | 10.562 | 10.961 | -0.398 | 10.164 | no | —(rejected) |  |
| 9 | `w` | 10.430 | 10.430 | +0.000 | 10.430 | no | —(rejected) |  |
| 10 | `bag` | 10.383 | 10.617 | -0.234 | 10.148 | no | —(rejected) |  |


### image 1425 — word "table" (node: dining table) — step 53

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` table` | 22.781 | 22.438 | +0.344 | 23.125 | yes | 23.125 | **YES** |
| 2 | ` d` | 21.172 | 21.750 | -0.578 | 20.594 | yes | 20.594 |  |
| 3 | ` wooden` | 21.094 | 17.781 | +3.312 | 24.406 | no | —(rejected) |  |
| 4 | ` surface` | 18.734 | 18.750 | -0.016 | 18.719 | no | —(rejected) |  |
| 5 | ` edge` | 18.359 | 18.438 | -0.078 | 18.281 | no | —(rejected) |  |
| 6 | ` dark` | 17.828 | 15.633 | +2.195 | 20.023 | no | —(rejected) |  |
| 7 | ` wood` | 17.781 | 13.953 | +3.828 | 21.609 | no | —(rejected) |  |
| 8 | ` brown` | 17.766 | 15.227 | +2.539 | 20.305 | no | —(rejected) |  |
| 9 | ` left` | 17.094 | 18.156 | -1.062 | 16.031 | no | —(rejected) |  |
| 10 | ` right` | 16.953 | 17.969 | -1.016 | 15.938 | no | —(rejected) |  |


### image 1425 — word "bowl" (node: bowl) — step 63

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` bow` | 18.016 | 15.312 | +2.703 | 20.719 | yes | 20.719 | **YES** |
| 2 | ` cup` | 17.328 | 16.391 | +0.938 | 18.266 | yes | 18.266 |  |
| 3 | ` small` | 16.484 | 14.219 | +2.266 | 18.750 | yes | 18.750 |  |
| 4 | ` white` | 15.961 | 13.812 | +2.148 | 18.109 | no | —(rejected) |  |
| 5 | ` glass` | 15.016 | 15.641 | -0.625 | 14.391 | no | —(rejected) |  |
| 6 | ` sp` | 14.719 | 15.914 | -1.195 | 13.523 | no | —(rejected) |  |
| 7 | ` container` | 14.672 | 10.352 | +4.320 | 18.992 | no | —(rejected) |  |
| 8 | ` sau` | 14.656 | 11.438 | +3.219 | 17.875 | no | —(rejected) |  |
| 9 | ` jar` | 13.945 | 9.133 | +4.812 | 18.758 | no | —(rejected) |  |
| 10 | ` smaller` | 13.859 | 13.070 | +0.789 | 14.648 | no | —(rejected) |  |


### image 1425 — word "bowl" (node: bowl) — step 64

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | `l` | 26.000 | 26.203 | -0.203 | 25.797 | yes | 25.797 | **YES** |
| 2 | `el` | 15.055 | 15.031 | +0.023 | 15.078 | no | —(rejected) |  |
| 3 | `ls` | 12.977 | 12.664 | +0.312 | 13.289 | no | —(rejected) |  |
| 4 | `,` | 12.727 | 13.016 | -0.289 | 12.438 | no | —(rejected) |  |
| 5 | `ler` | 11.727 | 12.352 | -0.625 | 11.102 | no | —(rejected) |  |
| 6 | ` of` | 11.531 | 12.219 | -0.688 | 10.844 | no | —(rejected) |  |
| 7 | `led` | 11.445 | 12.336 | -0.891 | 10.555 | no | —(rejected) |  |
| 8 | `ling` | 11.297 | 11.570 | -0.273 | 11.023 | no | —(rejected) |  |
| 9 | ` or` | 11.227 | 11.320 | -0.094 | 11.133 | no | —(rejected) |  |
| 10 | ` and` | 11.086 | 11.383 | -0.297 | 10.789 | no | —(rejected) |  |


### image 1425 — word "table" (node: dining table) — step 86

| rank(E) | token | expert E | amateur A | d=E-A | cd pre-APC | survives APC? | cd post-APC | chosen? |
|---|---|---|---|---|---|---|---|---|
| 1 | ` table` | 25.906 | 25.922 | -0.016 | 25.891 | yes | 25.891 | **YES** |
| 2 | ` left` | 21.719 | 21.125 | +0.594 | 22.312 | no | —(rejected) |  |
| 3 | ` d` | 21.672 | 21.359 | +0.312 | 21.984 | no | —(rejected) |  |
| 4 | ` right` | 20.438 | 21.281 | -0.844 | 19.594 | no | —(rejected) |  |
| 5 | ` side` | 20.188 | 20.016 | +0.172 | 20.359 | no | —(rejected) |  |
| 6 | ` plate` | 19.766 | 20.500 | -0.734 | 19.031 | no | —(rejected) |  |
| 7 | ` edge` | 18.797 | 18.750 | +0.047 | 18.844 | no | —(rejected) |  |
| 8 | ` top` | 18.094 | 17.422 | +0.672 | 18.766 | no | —(rejected) |  |
| 9 | ` same` | 18.078 | 18.078 | +0.000 | 18.078 | no | —(rejected) |  |
| 10 | ` scene` | 18.031 | 17.078 | +0.953 | 18.984 | no | —(rejected) |  |
