# The miss-rate bound assumed files are independent. They are not.

**Self-audit, unprompted.** Triggered by catching the same error in someone else's
numbers: three files whose figures matched to the millisecond turned out to be one
season sampled three times. That obliged me to check my own.

## The corpus is folder-structured and I never looked

| | |
| :--- | :--- |
| corpus | **315 files across 38 folders** |
| files per folder | median 2, **max 49** |
| share of files sharing a folder with another | **95 %** |

## What that does to the published bound

The seeded sample is 98 files — **spanning 19 folders**, mean cluster 5.2, max 12.

```
FILES    2/98  = 2.0 %   Wilson [0.6 %, 7.1 %]   assumes files independent
FOLDERS  2/19  = 10.5 %  Wilson [2.9 %, 31.4 %]  assumes perfect within-folder correlation
```

**The truth is between these.** The published interval is the optimistic endpoint of
that range, not a bound over it.

## What I am NOT doing

**I am not retroactively refuting the frozen prediction.** The estimand was frozen as a
per-file rate, the refutation condition was 10 % on that rate, and it scored 2.3 % and
NOT REFUTED. Rescoring on a unit chosen after seeing the data is exactly the move that
makes frozen tests worthless. **It stands as scored.**

## What I am saying

1. **The interval is too narrow.** It assumes 98 independent draws; there are 19
   clusters. The published `[1.3 %, 7.2 %]` overstates precision.
2. **The conclusion is unit-dependent and that was never stated.** On the folder unit
   the point estimate is **10.5 %, sitting exactly on the frozen 10 % condition**. A
   reader was given no way to know the verdict turned on the choice of unit.
3. **The favourable accident.** The two misses (id 108, id 691) are in **different
   folders**. Had they shared one, the folder-unit estimate would be 1/19 = 5.3 %.
   The spread of the misses is doing work no one chose.

## The rule this produces

> **Report the unit before the fraction, and choose the unit before seeing the data.**

`ACCEPTED_POPULATION.md` already does this — its unit is *folders*, one file per
folder, stated up front. That was right for reasons I had not generalised. The
miss-rate bound predates the lesson.

## Status

**The bound is not withdrawn.** It is annotated: correct as scored, on an estimand
whose precision was overstated and whose verdict is unit-dependent. Anyone quoting
`2.3 % [1.3 %, 7.2 %]` should quote the folder structure with it.
