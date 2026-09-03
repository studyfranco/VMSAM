# 008 — A third speed relation: measure the coefficient, do not assume it

**Agent:** `vmsam-dev-1` (measurement). **Pairs with 009.** **State:** open.
**Campaign objective 3, the measurement half.**

## The signal

```
(0.58, 65, -8125)   (0.61, 382, -47750)
```

Low fidelity **throughout**, and an offset that **grows along the file**. That is
a speed relation, not a bad measurement — and the low fidelity is a consequence of
the drift inside each window, not evidence against the reading.

Check the arithmetic yourself before trusting the example: 65 points × 125 ms =
8125 ms, 382 × 125 = 47750 ms. Both delays are exact multiples of the chromaprint
frame, which is what a quantised measurement of a continuous drift looks like.

## Goal

**Measure the coefficient. Do not assume PAL's 1.042709.** Two windows give it as
the change in delay over the change in position. Report what you measured, with
its uncertainty — the delay is quantised to 125 ms, so a two-window estimate has a
floor you can state exactly, and more windows lower it.

Then say **which relation it is**, or that it is none you recognise. `1000/1001`,
`24/23.976`, `25/24`, `30/29.97` and their inverses are all candidates; so is
"nothing standard", and that is a legitimate answer.

## What the campaign wants from it

`CAMPAIGN.MD`: treat it **exactly as PAL is treated — same method, different
coefficient. Audio and subtitles are both taken.** `docs/AUDIO_SPEED_POLICY.MD`
rules `asetrate` for PAL and NTSC and explains why the *right* answer inverts if
the source was already pitch-corrected at origin. **Read the inverting case before
proposing a method for a third relation** — the ruling is about the physics of how
the speed change was introduced, not about the factor.

That document is **frozen**. A third relation confirmed by measurement **extends**
it, and reaches the owner as a finding in `AGENT_NOTES.MD`, never as an edit.

## Expect both at once

Insertions and deletions can sit **on top of** a speed relation. A residual
staircase after removing the slope is 006's problem, not a refutation of the
slope. Say which you are looking at.

## Target-first — binding

Build one correct answer by hand, on one real error file, before writing code.
**State the coefficient you expect and what result would refute you, before the
run.** Campaign 1 has the counter-example: a method tuned until its spread was the
tightest of a whole benchmark, 0.008 ms, while it correlated the wrong language and
was wrong by 21 ms.

## What changed / measured / dropped / blocked
*(fill as you go)*
