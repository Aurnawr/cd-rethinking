# Claim Validation at Scale: 500-Image Experiment

This document checks your claim — *"VCD/SID pump up (amplify) the logit
score of object-words by roughly the same amount whether the object is
really in the picture or not; after the APC filter is applied, it basically
comes down to which words happen to survive, and you can end up with a real
object or a made-up one either way"* — using the full 500-image COCO set
(the same 500 images as your own CHAIR study), not just the earlier
10-image test run.

**What "decoding" setup was used**: greedy for all three methods (the model
always picks its single top choice at each step, no randomness) — this
matches the smaller 10-image test, confirmed with you before running. Your
own separate 500-image CHAIR run used real randomness (sampling) instead;
those numbers are shown too for comparison, in §5.

**How much data**: 500 images processed; 1 image (id 104669) was dropped
from the token-level statistics because of a data-integrity issue during a
pause-and-resume of the job (negligible loss, 1 out of 500). Every number
below is recomputed directly from the raw, saved, per-token records
(114,180 individual logged token decisions) — not from a running counter,
so there's no risk of a silent bug inflating the numbers.

---

## What is a "decoding step," exactly?

The model writes a caption **one word-piece at a time**. Each single
word-piece it writes is one "decoding step." So a 90-word caption is made
of roughly 90-120 decoding steps (some words are one step, some longer
words get split into 2-3 pieces).

At every single step, before the model picks the next word-piece, it has
computed a score (called a **logit**) for **every one of the 32,000 words
in its vocabulary** — basically "how much do I want to say this word right
now." The word with the highest score normally wins. All the statistics in
this document are about what happens to those scores, at every one of these
steps, across all 500 images.

---

## 1. Your exact claim, checked

**The formula in plain terms**: at each step, the model computes two scores
for a candidate word — one using the real picture (call this the "expert"
score, $E$), and one using a deliberately worsened version of the picture
(call this the "amateur" score, $A$ — for VCD this is a noised/blurred
picture, for SID it's the same picture but with most of it hidden from
attention). Then it combines them:
$$C = E + (E - A)$$
So the final combined score $C$ is just the expert score, **plus a bonus**
equal to how much higher the expert rated the word than the amateur did.
Call that bonus $d = E - A$. A bigger $d$ means a bigger bonus — i.e. more
amplification, not less.

| method | # real-object word instances checked | # made-up-object word instances checked | avg bonus $d$, real objects | avg bonus $d$, made-up objects |
|---|---|---|---|---|
| VCD | 33,755 | 23,175 | 0.1754 | **0.2334** |
| SID | 33,850 | 23,400 | **0.3273** | 0.2614 |

**In plain English**: both methods give SOME kind of bonus to object-words
in general (see §1b below for the comparison against non-object words). But
does that bonus land specifically on the FAKE objects (which is what you'd
want, if the goal is spotting hallucinations) or does it land roughly
equally on real ones too?

- **VCD** gives fake objects a *slightly bigger* bonus than real ones
  (0.233 vs 0.175, about 33% more) — the "correct" direction if you wanted
  a hallucination detector, but a small effect.
- **SID** does the opposite — real objects get the *bigger* bonus (0.327 vs
  0.261) — the "wrong" direction.

**Important correction from the smaller (10-image) test**: at only 10
images, VCD showed the opposite pattern from what's shown above, and SID's
pattern was much weaker. At 500 images, *both* directions flip compared to
the small test. **This means the small test's specific numbers were noise,
not a real pattern — the 500-image numbers above are the trustworthy
ones.** The lesson: even though each image contributes many word-level data
points, those points aren't independent of each other (they all come from
the same caption/image), so you need many *images*, not just many *word
instances*, to get a reliable answer.

### 1b. Does this bonus only happen for object-words, or for everything?

You asked specifically to check what happens to non-object words too (words
like "the," "with," "standing," "near"). Using the same 500-image data:

| method | avg bonus $d$, object words | avg bonus $d$, everything else | object words get how much more bonus? |
|---|---|---|---|
| VCD | 0.2018 (n=55,990) | 0.0846 (n=3,074,327) | **2.39x more** |
| SID | 0.3028 (n=56,224) | 0.2095 (n=3,069,248) | **1.45x more** |

**Yes — this confirms your proxy idea.** Object-words consistently get a
noticeably bigger bonus than everything else, for both methods, across
three million+ non-object word instances. In plain terms: the mechanism
correctly notices "this is a claim about an object, which needs the picture
to back it up" and boosts it accordingly — that part makes sense, since
seeing a blurred/hidden picture genuinely should shake the model's
confidence in object claims more than in filler words like "the" or "with."

**What it can't do is tell a real object-claim from a fake one** — from
§1's table, the bonus for real vs. fake objects is much closer to each
other (0.175 vs 0.233 for VCD, 0.327 vs 0.261 for SID — a 25-33% difference)
than the bonus for object-words vs. everything-else is (a 45-139%
difference). **So: VCD/SID amplify "this sounds like an object claim"
strongly and fairly reliably. They do not reliably amplify "and this
particular object claim happens to be false" — that second, finer
distinction is much weaker and even points the wrong way for SID.**

**What stays rock-solid at both 10 and 500 images — the expert's own head
start:**

| method | avg raw expert score, real objects | avg raw expert score, fake objects | gap |
|---|---|---|---|
| VCD | 14.4628 | 13.0627 | 1.4001 |
| SID | 14.3734 | 13.0327 | 1.3407 |

This gap (about 1.3-1.4 points) is almost identical whether you look at 10
images or 500 — this part of the picture is stable and real. **Before any
of the amateur/bonus machinery even runs, the plain expert model already
rates fake objects about 1.3-1.4 points lower than real ones, on its own.**
That's a real, if weak and imperfect, signal the model already had.

---

## 2. The APC filter collapses almost everything to 1-2 choices — the most solid finding

**What APC does, in plain terms**: after computing the combined score for
every one of the 32,000 words, the model throws away any word whose ORIGINAL
expert score (not the combined score — just the plain "does the real
picture support this" score) was too far below the best expert score at
that step. Only the survivors are allowed to be picked.

| method | decoding steps checked | avg words still standing after the filter (out of 32,000) | % of steps where 2 or fewer words survive | % of steps where the final answer matches what you'd get from the expert alone, ignoring the amateur entirely |
|---|---|---|---|---|
| VCD | 59,122 | 2.309 | 69.8% | 93.9% |
| SID | 57,973 | 2.328 | 69.4% | 91.7% |

Nearly identical whether you look at 10 images or 500. **At roughly 7 out of
every 10 decoding steps, only 1 or 2 words out of 32,000 are even allowed to
be considered — and because this filter only looks at the expert's score
(never the amateur's), the final output matches what plain, ordinary,
amateur-free decoding would have produced over 90% of the time regardless.**

---

## 3. Does the model "notice" hallucination steps more? Only sometimes, and noticing isn't fixing

| method | type of word being decided | how many such steps | % of the time the final answer differs from what the expert alone would have picked |
|---|---|---|---|
| VCD | fake object | 928 | **15.6%** |
| VCD | real object | 5,205 | 4.8% |
| VCD | ordinary/filler word | 51,640 | 5.7% |
| VCD | number word ("two," "several") | 918 | 14.9% |
| VCD | colour word ("red," "blue") | 431 | **27.6%** |
| SID | fake object | 895 | **16.4%** |
| SID | real object | 5,292 | 6.6% |
| SID | ordinary/filler word | 50,278 | 7.8% |
| SID | number word | 955 | 18.2% |
| SID | colour word | 553 | **38.3%** |

**Correction to the small test**: at 10 images, SID looked like it didn't
pay any special attention to fake-object steps. At 500 images (and nearly
900 fake-object steps instead of 15), it turns out SID DOES change its
answer at fake-object steps about 2-3x more often than at ordinary steps —
same pattern as VCD. That small-test finding was also noise.

Colour words get the biggest change-of-answer rate of any category, for
both methods — makes sense, since colour is a very literal, pixel-level
property, so a blurred/hidden picture genuinely should change the model's
opinion about colour more than about most other things.

**But changing the answer is not the same as fixing it.** These are, by
definition, steps where the FINAL caption still ends up mentioning a fake
object — so "the model changed its mind here" just means it swapped from
one guess to a different guess, and the different guess was *still wrong*
(see §4 for concrete examples of this swap happening).

---

## 4. What actually happens to the captions — corrects the small test's apparent SID win

| method | avg caption length (words) | real objects mentioned per 100 words | fake objects mentioned per 100 words | share of mentions that are fake |
|---|---|---|---|---|
| plain (no CD) | 90.4 | 2.690 | 0.937 | 25.8% |
| VCD | 91.8 | 2.642 | **0.975** | 27.0% |
| SID | 90.2 | 2.736 | **0.979** | 26.4% |

**Both VCD and SID make hallucination slightly WORSE at 500-image scale**,
even with no randomness involved. The small 10-image test had made SID look
like a clear improvement (fake-object mentions dropping by over a quarter)
— that was luck of which 10 images got picked, not a real effect.

**The "whack-a-mole" pattern, now backed by real numbers (counted across
499 images):**

| method | total fake-object mentions | how many of plain decoding's fake objects got removed | how many brand-new fake objects got added instead | net change |
|---|---|---|---|---|
| plain (no CD) | 423 | — | — | — |
| VCD | 447 | 209 | 233 | **+24 (worse)** |
| SID | 441 | 222 | 240 | **+18 (worse)** |

Both methods genuinely do get rid of a lot of the original hallucinations
(209, 222 of them) — but they replace them with even more brand-new ones
(233, 240). Net result: slightly more total hallucination than doing
nothing at all. The small test's finding that SID was a net win (removing 4
fake objects, adding only 1) does not hold up at proper scale — it flips to
a small net loss.

---

## 5. Cross-check against your own real (randomness-based) 500-image numbers

| method | this test (no randomness) | your own run (with randomness, matching the original papers) |
|---|---|---|
| plain (no CD) | 13.88% | 13.42% |
| VCD | 14.61% | **17.05%** |
| SID | 13.89% | **16.27%** |

Same direction both ways (VCD/SID both worse than plain decoding), but much
worse under real randomness. Makes sense given §2: the filter already
narrows the choice down to about 2 options; plain (no-randomness) decoding
always takes the single best of those 2, while adding randomness on top
means sometimes picking the *second-best* of an already-poor, non-selective
shortlist — which can only add more noise, not remove it.

---

## Bottom line

Your claim holds up, now backed by proper statistical power: VCD and SID
correctly amplify "this sounds like it needs the picture to confirm it" for
object-words in general (2.39x/1.45x more bonus than non-object words) —
but that amplification does not reliably tell real objects from fake ones
(the effect even points the wrong way for SID). The one part of the whole
system that consistently, reliably favours real objects over fake ones is
something the amateur model has nothing to do with: the plain expert
model's own, pre-existing, imperfect ~1.3-1.4 point head start for real
objects. And the real-world result, measured directly on the captions, is
that both methods end up with slightly *more* total hallucination than
doing nothing, mostly by trading one wrong guess for a different wrong
guess rather than getting it right.

---

## In plain English: "the only stable real/fake signal is the expert's own pre-existing confidence gap, which alone governs survival"

Think of it like a job interview with two rounds:

- **Round 1 (the survival filter, APC)**: only the candidate's *original*
  resume score decides who even gets an interview. This score was written
  entirely by one specific person — the "expert" — before anyone else's
  opinion is asked. On average, candidates who are a genuinely good fit for
  the job score a little higher on this resume than candidates who aren't
  — but it's not a perfect rule; it's just a mild tendency.
- **Round 2 (the amateur's bonus)**: for whichever candidates *did* make it
  to round 2 (almost always only 1 or 2 people, out of 32,000 who applied),
  a second person — the "amateur," who only got a blurry photocopy of each
  resume — gives feedback. If the amateur is noticeably less impressed than
  the expert was, that candidate gets a bonus added to their score.

The catch: **whether you even get to Round 2 at all was decided entirely by
Round 1** — and Round 1 was run by the expert alone, using their own
opinion, before the amateur was ever consulted. The amateur's bonus in
Round 2 can only shuffle the order among the 1-2 people who already made
the cut; it can never bring back someone the expert rejected outright, no
matter how strongly the amateur would have vouched for them.

So if you ask "why did this caption end up with a real object instead of a
fake one" (or vice versa), the honest answer is almost never "because the
blurry-photo amateur was clever enough to catch the fake one." It's almost
always "because the expert's own first-round resume score for the real
object happened to be higher than for the fake one" — which is true on
average, but only a mild, imperfect tendency, not a reliable rule. The
amateur genuinely tries to help (it does give bigger bonuses to
object-words that need visual proof, per §1b) — it just never gets the
chance to reverse a decision the expert already made in Round 1, and even
its Round-2 bonus doesn't reliably tell a true object from a false one
(§1). That is the whole mechanism, end to end.
