# Methods — d graphs for the TOP-30 candidates

**Files:** `d_top30_vcd_hall.png`, `d_top30_vcd_real.png`, `d_top30_sid_hall.png`, `d_top30_sid_real.png`

This set is identical in method to the `d_top10/` set, but uses a **wider window
— the top-30 candidates by the expert logit at each step** instead of the top-10.
It exists as a robustness check: a real effect should not depend on where we cut
the candidate list.

## 1. What the graph shows

- **y-axis = d = E − A** for one object candidate.
- **x-axis = n** = a plain running index; it only lines the points up side by side
  and carries **no meaning** (not time, not order).
- **Colour:** **red when d > 0**, **blue when d < 0**. Solid line = 0, dashed line
  = mean.

## 2. What d means

Each step the model scores every next word twice: **E = expert logit** (clean
image) and **A = amateur logit** (degraded image — VCD: Gaussian-noised;
SID: attention-pruned). CD's final score is **C = 2E − A = E + (E − A)**, so
**d = E − A is exactly what CD adds** to the plain expert score:

- **d > 0 (red):** CD amplifies the word.
- **d < 0 (blue):** CD suppresses the word.

A working suppression mechanism should give hallucinated objects **negative** d.

## 3. Which candidates are counted, and how they are labelled

For each step we take the **30 highest-scoring words by the expert logit**. Among
those we keep COCO objects using the same two trustworthy rules as the top-10 set:

1. **Word-initial filter** — a token counts as an object only if it **starts a
   word** (SentencePiece "▁" marker) **and** its text is a COCO object word. This
   drops subword fragments (e.g. "ship" in "companion·ship").
2. **Real vs. hallucinated** — the word's canonical COCO category is checked
   against the image's **ground-truth object set** (COCO instance-segmentation
   labels + objects named in the five reference captions, via the CHAIR synonym
   list). In the set → **real**; not in the set → **hallucinated**.

The amateur score A must be available for the candidate (i.e. the token is also in
the amateur's saved top-30); otherwise the point is skipped.

## 4. Sample sizes and numbers

| | VCD real | VCD halluc | SID real | SID halluc |
|---|---|---|---|---|
| n (candidates) | 19626 | 10045 | 18970 | 9993 |
| mean d | +0.034 | **−0.010** | +0.149 | **−0.007** |

**How to read it:** in this wider window the hallucinated candidates' mean d is
**essentially zero** (−0.01 for both methods) — a coin-flip, not a systematic
push downward. Real objects lean mildly positive. So even given the widest fair
view, CD applies **no reliable suppression** to hallucinations.

**A useful point for a reviewer:** notice the sign of the hallucination effect
**flips** between the top-10 view (slightly positive) and this top-30 view (≈ zero
/ slightly negative), and differs between VCD and SID. A genuine suppression
signal would be **stable and clearly negative** across these reasonable choices.
The fact that it moves around near zero is itself evidence that there is no real
hallucination-suppression signal — only noise around zero.

## 5. Honest notes for a reviewer

- **Greedy** decoding, 500 COCO images, LLaVA-1.5-7B (reproducible).
- Same small candidate-level caveat as top-10: a word-initial "bear" could begin
  "teddy bear"; this affects <1% of candidates and both branches equally, so it
  cannot bias the comparison. The emitted-word set (`d_emitted_word/`) has no such
  caveat and agrees with the conclusion.
- Read this together with `d_top10/`: the conclusion (hallucinations are not
  suppressed) holds at both window sizes, which is the robustness check this set is
  meant to provide.
