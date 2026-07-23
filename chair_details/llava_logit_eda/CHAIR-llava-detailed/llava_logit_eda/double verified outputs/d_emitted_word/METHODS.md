# Methods — d graphs for the EMITTED word

**Files:** `d_vcd_hall.png`, `d_vcd_real.png`, `d_sid_hall.png`, `d_sid_real.png`

This is the cleanest of the three d-graph sets: it uses only the words the model
**actually wrote**, so every real/hallucinated label is exact.

## 1. What the graph shows

- **y-axis = d = E − A** for one object word (defined below).
- **x-axis = n** = a simple running index that just lines the data points up side
  by side. It has **no meaning** (not time, not caption position) — it only lets
  us see the whole spread of d values in one picture.
- **Colour:** a bar is **red when d > 0** and **blue when d < 0**. The solid line
  is d = 0; the dashed line is the mean.

## 2. What d means

At each generation step the model produces a score (a *logit*) for every possible
next word, computed twice:

- **E = expert logit** — from the **real, clean image**.
- **A = amateur logit** — from the **degraded image** (VCD: the image with added
  Gaussian noise; SID: the image with most visual tokens pruned by attention).

Contrastive decoding's final score is **C = (1+α)E − αA = 2E − A = E + (E − A)**,
so the quantity **d = E − A is exactly the amount CD adds to (or removes from) the
plain expert score.** Therefore:

- **d > 0 (red):** CD pushes the word **up** (amplifies it).
- **d < 0 (blue):** CD pushes the word **down** (suppresses it).

If CD worked as intended, hallucinated objects should have **negative** d
(suppressed). The graphs test whether that is true.

## 3. How we decide which words are real vs. hallucinated

We use the **standard CHAIR protocol** (Rohrbach et al., 2018) — the same
benchmark VCD/SID themselves are evaluated on — applied to the model's full
generated caption:

1. Build each image's **ground-truth object set** = the COCO objects actually in
   the image, taken from two sources combined: the COCO **instance-segmentation
   labels** and the objects named in the image's **five human reference captions**
   (mapped through CHAIR's synonym list).
2. Parse the model's caption into words, **singularise** them, **collapse
   multi-word objects** ("teddy bear", "dining table", "hot dog") so they are not
   mis-split, and map each word through the CHAIR synonym dictionary to its
   canonical COCO category.
3. For every object word the model wrote: it is **real** if its category is in the
   image's ground-truth set, and **hallucinated** if it is not.

Because this runs on the *whole written word*, it cannot be fooled by subword
fragments — e.g. the "ship" inside "companion·ship" is never counted as a boat,
and "bear" is correctly read as part of "teddy bear".

## 4. From a labelled word to a point on the graph

For each object word the model emitted, we take the step where its first token was
produced, read off that token's **E** (from the clean-image pass) and **A** (from
the degraded-image pass), and plot **d = E − A**. Hallucinated words go in the
`_hall` graphs, real words in the `_real` graphs.

(If the emitted token happens not to appear in both the top-30 expert and top-30
amateur lists we saved, that point is skipped — this is rare.)

## 5. Sample sizes and headline numbers

| | VCD real | VCD halluc | SID real | SID halluc |
|---|---|---|---|---|
| n (words) | 3213 | 577 | 3227 | 566 |
| mean d | +0.264 | **+0.789** | +0.296 | **+0.684** |

**How to read it:** hallucinated words have a **large positive** mean d — CD
*amplifies* them, by roughly **3× more** than it amplifies real objects. This is
the opposite of the suppression CD is supposed to perform.

## 6. Honest notes for a reviewer

- These are **greedy-decoded** captions on 500 COCO images with LLaVA-1.5-7B
  (deterministic, reproducible).
- This set is conditioned on words that were **actually emitted** (they "won" the
  step). That is exactly the population of interest for "what happens when a
  hallucination is produced," but it is selection-biased, so we report it
  alongside the candidate-level top-10 / top-30 sets (neighbouring folders) which
  are not conditioned on winning.
- The real/hallucinated labels here carry no subword or multi-word error, because
  they are assigned at the whole-word (caption) level, not per token.
