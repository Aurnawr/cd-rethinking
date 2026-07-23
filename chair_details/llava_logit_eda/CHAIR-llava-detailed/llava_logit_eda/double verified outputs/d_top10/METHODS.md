# Methods — d graphs for the TOP-10 candidates

**Files:** `d_top10_vcd_hall.png`, `d_top10_vcd_real.png`, `d_top10_sid_hall.png`, `d_top10_sid_real.png`

This set looks at object words among the **top-10 candidates the expert
considered at each step** — not only the word it finally wrote. It is broader than
the emitted-word set (neighbouring `d_emitted_word/` folder) and is **not**
conditioned on a word "winning", so it removes the selection bias of that set.

## 1. What the graph shows

- **y-axis = d = E − A** for one object candidate.
- **x-axis = n** = a plain running index that lines the data points up side by
  side; it has **no meaning** (not time, not order) — it just lets us see the full
  spread of d in one picture.
- **Colour:** **red when d > 0**, **blue when d < 0**. Solid line = 0, dashed line
  = mean.

## 2. What d means

At each step the model scores every possible next word twice:

- **E = expert logit** — from the **clean image**.
- **A = amateur logit** — from the **degraded image** (VCD: Gaussian-noised;
  SID: attention-pruned).

CD's final score is **C = 2E − A = E + (E − A)**, so **d = E − A is exactly what CD
adds to the plain expert score**:

- **d > 0 (red):** CD amplifies the word.
- **d < 0 (blue):** CD suppresses the word.

A working suppression mechanism should give hallucinated objects **negative** d.

## 3. Which candidates are counted, and how they are labelled

For each step we take the **10 highest-scoring words by the expert logit** (the
top-10). Among those 10 we keep the ones that are COCO objects, using two rules
that make the labelling trustworthy:

1. **Word-initial filter.** A token is treated as an object **only if it is the
   start of a word** (in the LLaMA/SentencePiece tokenizer, a word-initial token
   carries the "▁" space marker) **and** its text is a COCO object word. This
   removes subword fragments — e.g. the "ship" piece inside "companion·ship" is
   *not* word-initial, so it is correctly ignored.
2. **Real vs. hallucinated.** The token's word is mapped (via the CHAIR synonym
   list) to its canonical COCO category, and that category is checked against the
   image's **ground-truth object set** — the COCO **instance-segmentation labels**
   plus the objects named in the **five human reference captions**. In the set →
   **real**; not in the set → **hallucinated** (goes in the `_hall` graph).

We need the amateur score A for the candidate too; if a top-10 expert candidate is
not also present in the amateur's top-30 list we saved, that point is skipped
(documented; it is a small fraction).

## 4. Sample sizes and numbers

| | VCD real | VCD halluc | SID real | SID halluc |
|---|---|---|---|---|
| n (candidates) | 9432 | 3461 | 9498 | 3476 |
| mean d | +0.124 | +0.148 | +0.215 | +0.122 |

**How to read it:** hallucinated candidates are **not** suppressed — their mean d
is positive, and for VCD it is even slightly *higher* than for real objects
(0.148 vs 0.124), i.e. CD amplifies hallucinations a touch more. For SID the order
is reversed and small. Either way there is no clean negative (suppressive) signal
on hallucinations.

## 5. Honest notes for a reviewer

- **Greedy** decoding, 500 COCO images, LLaVA-1.5-7B (reproducible).
- **One residual labelling caveat** (candidate-level only): a word-initial "bear"
  could be the start of "teddy bear", which our per-candidate rule would read as
  the animal "bear". This affects well under 1% of candidates and hits the expert
  and amateur branches equally, so it cannot bias the d comparison. The
  emitted-word set in the neighbouring folder has **no** such caveat (it labels at
  the whole-word level) and shows the same conclusion.
- Compare this folder with `d_top30/` (a wider candidate window): if the
  conclusion is real it should not depend on the exact cut-off — and indeed both
  show hallucinations are not suppressed.
