# The denominator chain for every number I have published

**PRIVATE.** Written after `vmsam-dev-1` observed that an ask header *records the ask
you pass it, not the ask you meant* — a denominator written at the innermost loop
certifies that loop and says nothing about the filters above it.

## The error this found in a published number

```
1. error records in the corpus                  315
2. UNIQUE MASTER PATHS among them               246   <- 69 records share a master
3. unique masters with nonzero start_time        80   = 32.5 % of step 2
4. of those, nonzero on a shared NON-delay lang  27   = 33.8 % of step 3
```

**I published step 4 as a fraction of step 3.** Step 2 had already dropped records, and
steps 3–4 count **masters** while the corpus counts **files**.

| published | correct |
| :--- | :--- |
| "80 masters carry a nonzero start_time" | **90 error FILES are affected**, 28.6 % of 315 |
| "27 files satisfy all three conditions" | 27 masters = 27 files, 8.6 % — **right by coincidence**, since those masters happen to carry one record each |

> **A denominator can be present, printed, correct, and still answer a different
> question than the reader's.** My own instrument-audit grep checks whether a
> denominator exists; it cannot see that one is three filters downstream of the
> population a reader would assume, or that it changed units partway along.

The underlying assumption was *one master, one file* — true for 177 of 246 masters and
false for the rest. **True of my data, silent about being an assumption**, which is the
same class dev-1 found in its own tool when a key name that held in its lab failed the
moment it crossed an author boundary.

## The chain for every figure currently in front of anyone

| figure | chain | units | status |
| :--- | :--- | :--- | :--- |
| **2.3 %, [1.3 %, 7.2 %]** miss rate | 254 eligible → 100 sampled (seeded, manifest before first probe) → 98 after 2 dropped for cause → **88 the instrument ran on** | files throughout | **one filter deep, stated in the frozen prediction before the run.** Unaffected |
| 21.7 % silence, id 150 | 1 file → 23 points at 60 s | points, not duration | coarse; understates; pre-`ce19329` |
| 23.4 % silence, id 695 | 1 file → 47 points at 30 s | points, not duration | coarse; understates; pre-`ce19329` |
| **90** files with nonzero master start | 315 → 246 masters → 80 → **90 files** | corrected to files | supersedes the published "80" |
| 27 masters, shared non-delay | 315 → 246 → 80 → 27 | masters = files here | right, by coincidence |
| accepted-population pilot | 294 readable folders → 6 files → 4 not measurable | files | reported its own failure |

## The rule this produces

**State the chain, not the innermost ratio — and state the units at every step.** A
ratio is uninterpretable without both, and both are invisible to the reader and to any
audit that only asks whether a denominator was printed.
