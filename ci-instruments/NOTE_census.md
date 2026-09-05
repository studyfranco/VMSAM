# The refused corpus cannot be read as a census

**PRIVATE.** Written for objective 3, whose closing condition is three counts over
the refused corpus — *repaired, damaged, declined*. **Those counts carry an unbounded
term, and the owner should know before reading them rather than after.**

This is not a claim that the counts are wrong. It is a claim about what kind of
number they are.

---

## The claim

The stage-1 record is the only corpus-wide account of what the pipeline saw. **It is
not a sample of the media; it is a sample of the media through a fixed geometry**,
and that geometry has two independent blind spots. Neither is estimable from the
record, because the record does not contain what it did not sample.

## Blind spot 1 — position. Computed per file, not estimated

`prepare_get_delay_sub` starts the first window at `begin_in_second` = 120 s, and each
window is twice the spacing. A step changes a reading only if one window is
majority-pre and the next majority-post, which requires `T > begin + L/2`.

Across the 306 corpus files long enough to use the ten-window geometry:

| | |
| :--- | :--- |
| first visible change point | **median 228 s**, range 171–488 s |
| **fraction of each file structurally unsampled** | **median 15.9 %**, range 11.4–21.2 % |

**About one sixth of every episode is never looked at.** Not sampled sparsely —
not sampled.

## Blind spot 2 — magnitude. Independent of position

A step must also exceed roughly **half a quantum** to move any window's reading. The
quantum recovered corpus-wide is 125 ms (271 files), 124 ms (27), 138 ms (3).

> **Any step below ~62 ms is invisible anywhere in the file, at any position.**

## Both are measured, not inferred

| | |
| :--- | :--- |
| position | three files carry head steps at 146–160 s and 184 s — all below their own first-visible threshold. Their records show a flat plateau **or no matrix at all** |
| magnitude | one file carries an **89 ms** step at **1170 s** — late enough, but 0.7 of a point, so every 194 s window reads the same plateau either side |

The second is the important one: it shows the two conditions are **independent**. A
step can be well past the position horizon and still invisible.

## Why this lands on objective 3 specifically

*Declined* is the count that proves a decision procedure exists. But a file the
geometry cannot see is not **declined** — it is **never presented**. It will be
counted as a clean constant offset, or as nothing at all, and it will look like
agreement.

**Window agreement is evidence about the windows, not about the media.** One file
demonstrates it exactly: its offset departs at ~56 s and *returns* at ~147 s, so any
window spanning both transitions reads constant. Every window the geometry produces
starts after the first one.

So the risk is not that *repaired* is too low. It is that **declined is inflated by
files that were never examined**, and nothing in the record distinguishes them.

## What would bound it

Stated as a request, since the point of naming a gap is that someone can fill it.

**Neither of these alone is sufficient. Together they are.**

### A — the rate: how often do real files fall in the blind region

Profile a random sample of refused files **end to end** with an instrument that has
**no privileged regions**, and count how many carry structure the record missed. That
gives a miss rate with a confidence interval.

The instrument exists — a whole-file profiler was built today and has already moved
one file out of *constant offset* into *trimmed edges* on exactly this basis. What is
missing is that it has only been pointed where someone else's result suggested. **A
random sample is the difference between an anecdote and a rate.**

Cost: N files × profiling time, no container runs, read-only.

### B — the shape: which (position, magnitude) pairs are invisible

Take content-matched pairs, inject steps at **known** T and known size, run the
pipeline's own geometry, and map detection probability over that plane. This measures
the record's **sensitivity directly** rather than estimating its miss rate.

**Stated limit, and it is the fixture trap:** injected content is constructed, so its
expected answer is one I supplied. It is admissible here only because the thing under
test is *the geometry's sensitivity*, not the media — the record is produced by the
pipeline, not by me. But a sensitivity surface measured on synthetic content is a
surface for synthetic content, and that limit travels with the number.

### Why both

**B** gives the blind region's shape; **A** gives the density of real steps inside it.
The bound is the product. B alone tells you what *could* be missed; A alone tells you
what *was* missed in one sample, by one instrument, with its own coverage.

## What would NOT bound it

- **Reading the record more carefully.** The information is absent, not obscured.
- **Denser sampling within the same geometry.** It converges on a displaced peak,
  stably and confidently — that is the contamination-versus-resolution result, and it
  is why more probes of the same kind make the answer *more* confident and no more
  correct.
- **More instruments that share the geometry.** Three components independently
  inherited the same 120 s blind spot today — a locator, a verifier and my own
  bisection — because each used the pipeline's window layout as its map of where to
  look. **An independent instrument aimed by a dependent one is not independent.**

## What I am not claiming

That any published count is wrong. That the blind spot is large in effect — its
*extent* is measured (15.9 % of file duration, and everything under 62 ms), its
*consequence* is not. And nothing about how campaign 1 classified anything; I have
not read it.

**The honest form for objective 3: report the three counts with the statement that
the corpus they are drawn from is a sample through a geometry with two measured blind
spots, and that the miss rate is not yet bounded.** A count with a stated unbounded
term is usable. A count that looks complete is not.
