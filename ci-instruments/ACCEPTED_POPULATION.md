# The population that shipped — why it matters, and why I cannot measure it yet

**PRIVATE.** Ids and counts only.

## The finding that makes this the campaign's largest gap

`vmsam-forensic` measured the error corpus's inclusion criterion directly:

**v4, the version to hold** — three earlier versions of this figure were superseded,
and none of the four corrections came from a new instrument:

```
315 total   302 measured   13 COULD NOT MEASURE
delay-refused population n = 279
  minimum span EXACTLY 2.00 quanta ; at or below one quantum: ZERO
but 23 files DO sit at or below one quantum, each refused for a DIFFERENT reason
  11 inter-track divergence (master side), 4 per-language, 3 short-file quantum, 5 various
```

**The precise claim is narrower than "nothing sits below the floor".** A file whose
delay disagreement fits inside the 127 ms window is not refused *for that* — but it
can still be refused for something else, and 23 were. **What cannot appear is a file
that fits inside the window and has no other defect.**

The mechanism is identified by a three-arm control: quantum 124–125 slips under the
fixed window and the floor sits at two quanta; quantum 138 pokes through, and exactly
three files appear at one quantum.

The reconciliation window is one quantum. **A file whose delays agree to within one
quantum is reconciled and never writes an error record.**

> **THE ERROR CORPUS IS DEFINED, BY CONSTRUCTION, AS FILES WHOSE RECORD SPAN IS AT
> LEAST TWO QUANTA. A FILE CARRYING ONLY A SUB-QUANTUM STEP CANNOT BE IN IT — IT
> RECONCILED, WAS ACCEPTED, AND SHIPPED.**

This adds a **third quantity** beneath the two already separated here:

| | |
| :--- | :--- |
| files the record was wholly blind to | **2.3 %**, bounded, [1.3 %, 7.2 %] |
| instances of structure the record missed | larger, **not bounded** by that sample |
| **files the record never classified as failures at all** | **not estimable from this corpus by any method** |

The third is not a subset of either. **The 315 are the pipeline's refusals; they were
never a sample of its output**, and every blind-spot figure the campaign has published
is conditioned on a file having failed loudly enough to be recorded. **The sub-quantum
class is precisely the class that fails quietly.**

## Why the obvious measurement is unavailable

A successful merge consumes the candidate, so **no pair survives** for an accepted
file. There is nothing to correlate a master against.

## The measurement that does not need a pair

A merged file carries several audio tracks. **A step between two tracks of one shipped
file is a desync that shipped**, and it needs no candidate at all. 294 of 295
destination folders are readable; sampled files carry 2 to 12 audio tracks.

## PILOT RESULT — the instrument does not reach it, at my floor

Six files, tracks 0 vs 1, five positions each, 60 s probes:

```
2 tracks   accepted 0/5   not measurable
2 tracks   accepted 3/5   spread 0.5 ms   CONSTANT
6 tracks   accepted 2/5   spread 0.0 ms   CONSTANT
6 tracks   accepted 0/5   not measurable
3 tracks   accepted 1/5   not measurable
9 tracks   accepted 1/5   not measurable
```

**Four of six files failed to produce enough readings above the 0.25 floor.** Raising
the probe to 180 s did **not** fix it — 0/3, 2/3, 1/3 on a retest.

**The reason is structural, not a tuning problem:** two different-language tracks share
only a music-and-effects bed. The dialogue differs by construction, so the correlation
ceiling is set by how much of the mix is shared, and on these files that is mostly
below my floor.

## The honest blocker, and it is not the floor

One file read **45.0 / 44.9 / 45.0 ms across three probes at correlations 0.244, 0.386
and 0.028** — lags agreeing to 0.1 ms where correlation is near noise. That is
*plausible*: an identical M&E bed would peak at the true offset with low magnitude.

**But I have no positive control in this population.** Every instrument defect found
today was caught by a case whose answer came from outside the instrument, and here
there is no such case: no accepted file has a known desync. Without one I cannot
distinguish *"low-correlation lag agreement is real here"* from the spurious-peak
failure that already produced a fabricated 58 ms step earlier in this campaign.

> **Accepting low-correlation readings because they look consistent is exactly the
> move that produced my worst false positive. I am not making it without a control.**

## What would unblock it

1. **A positive control**: a shipped file with a known inter-track offset — obtainable
   by taking an accepted file and re-muxing one track with a deliberate step, then
   confirming the scanner recovers it. Synthetic, and admissible only because the
   thing under test is the *instrument's sensitivity*, not the media.
2. **A same-language track pair**, where correlation is not capped by differing
   dialogue. Rare, and worth counting before assuming.
3. Accepting that the third quantity may be **bounded only by construction** — the
   pipeline's own reconciliation window — rather than measured.


---

## A correction to my own audit of my own numbers

Asked which of my figures came from scripts and which from ad-hoc queries, I
classified them **by form** — and the corrected rule says form does not determine
safeguards: *a script is better only in the ways someone paid for.*

Reclassifying by safeguard instead **inverts two of the four**:

| | could-not-measure branch | denominator printed |
| :--- | :--- | :--- |
| `analyze_A.py` (the 2.3 %) | **yes** — refuses to score | yes, n=88 of 98 |
| `find_nonzero_start.py` (the 80) | **no** | **no** |
| `join_devfix.py` (the 27) | **no** | **no** |
| ad-hoc `/tmp` silence loop (21.7 %) | **yes** — printed *LOST* / *source silent too* / *ok* per point | yes |

**My throwaway heredoc carried more of the discipline than two of my scripts did.** It
classified every sample point three ways and printed the count; the scripts printed a
filtered list of hits and no denominator.

> **I judged my own instruments by their form one hour after being warned that form
> does not imply safeguards, and got it backwards for half of them.**

The exclusions themselves were measured rather than assumed, and both scripts' swallow
branches turned out never to have fired — 246 of 246 masters probed, 80 of 80 ids
probed, zero excluded. **The numbers stand; the confidence I attached to them by form
did not.**


---

## The positive control was built, and it settles the question NEGATIVELY

`build_control.py` takes an accepted library file with two audio tracks, decodes both,
and plants a **known +40 ms step at t = 600 s** in one of them. Read-only on the
library; output to scratch. Admissible only because the thing under test is the
**instrument's sensitivity**, not the media — the step is mine, so the expected answer
is mine.

### A vacuous test, built and caught — the third today

My first reading compared the two arms probe-by-probe:

```
   t     flat      step    difference    corr
 700      4.8     -35.2       -40.0     0.0846
 900      5.1     -34.9       -40.0     0.0329
1100    -73.5    -113.5       -40.0     0.0233
```

Every post-step probe moved by **exactly** −40.0 ms, and I began writing this up as
evidence that the low-correlation peaks were tracking real content.

> **It is guaranteed by construction. Shifting a signal by 40 ms shifts its entire
> cross-correlation function by 40 ms, so the argmax moves by exactly 40 whether or
> not the peak means anything — including for pure noise.**

**A control that cannot fail, built by me, for the third time today** — after id 150's
seven zeros and the 27-file list whose code path was unreachable.

### The arm that is not vacuous

`t0` vs `t1_flat`: both decoded from the same file at the same start, so **the true
offset is 0 ms** and the error is the absolute reading.

```
      t    reading   error    corr
    150       1.4      1.4   0.0293
    350       1.2      1.2   0.1674
    500      15.5     15.5   0.0215
    700       4.8      4.8   0.0846
    900       5.1      5.1   0.0329
   1100     -73.5     73.5   0.0233

median error 5.1 ms   MAX ERROR 73.5 ms   within 5 ms: 3/6
```

> **The noise floor of this measurement is ~73 ms on this file. The class it would be
> hunting is sub-quantum — below ~62 ms. THE NOISE EXCEEDS THE SIGNAL.**

### Conclusion, and it vindicates a constant I had been apologising for

**My 0.25 correlation floor was correctly rejecting these readings.** I had been
treating it as an over-conservative choice that hid a measurable population; the
control shows it was excluding readings whose error reaches 73 ms.

**The accepted population is NOT measurable by inter-track correlation on
cross-language pairs**, and this is now a measured limit rather than a suspicion. What
remains open is a same-language track pair, where correlation is not capped by
differing dialogue — rare, and worth counting before assuming.

---

# RESULT — and a false finding caught between the run and the report

`scan_accepted.py` completed: **294 folders offered, 292 probed, 73 with a
same-language pair.**

## What the scan reported

```
219  NO_SAME_LANGUAGE_PAIR
 65  CONSTANT
  6  STEP        <- spreads of 29, 460, 476, 778, 3532, 3697 ms
  4  COULD_NOT_MEASURE
```

**Six of 71 = 8.5 % of shipped files carrying a desync** is what that says, and it is
what I was one step from reporting.

## Why it is wrong

Every one of the six is **a single deviant probe at near-noise correlation**:

```
lang=fre  spread 778.75
    t= 200  lag  0.0     corr 0.5673   strong
    t= 450  lag  0.0     corr 0.9873   strong
    t= 700  lag  0.0     corr 0.5326   strong
    t= 950  lag  0.0     corr 0.1169
    t=1200  lag 778.75   corr 0.0144   <- the entire "step"
```

Four probes agree exactly; one at correlation 0.0144 does not. **That is the
spurious-peak failure the 0.25 correlation floor exists to catch — and I removed the
floor for this scan.**

**The reasoning that licensed removing it was sound and its scope was not.** The
positive control gave exact readings at correlation 0.058 — but on two encodes of
**one mix**, where a low-magnitude peak is still sharp. In the wild a same-language
pair may be two different recordings, and there a low-correlation reading is a broad
peak whose argmax is arbitrary. **I validated the rule on the favourable half of the
population and applied it to all of it.**

## The corrected result

Requiring **both** consistency **and** that a deviant probe be a measurement:

```
files where >=3 probes clear the floor : 48
COULD NOT MEASURE (fewer than 3)       : 23
CONSTANT                               : 48
STEP                                   :  0

spread among CONSTANT files: median 0.00 ms, max 18.38 ms
```

> **0 of 48. Wilson 95 % upper limit 7.4 %.**
>
> **No detectable inter-track step in the accepted population, on the part of it this
> instrument can reach.**

## Scope, and it is narrow

- **48 of 292 probed folders** — one file per folder, same-language pairs only, and
  23 further files where fewer than three probes were measurements.
- **Five probes per file.** A step between probes is invisible.
- **The threshold is 25 ms, chosen by me.**
- This says nothing about cross-language desync, which the earlier control showed is
  unmeasurable by this method at a ~73 ms noise floor.

## The lesson, and it is the fourth of its kind today

Three earlier controls of mine could not fail. **This one could, and did — in the
direction of a finding.** A control that cannot fail wastes a run; **a rule validated
on the favourable half of a population manufactures a result**, and a result about
shipped content would have been acted on.

**What saved it was storing every probe's correlation beside its lag.** The scan's own
verdict column was wrong; the data underneath it was sufficient to overturn the verdict
without re-measuring anything. *A summary that cannot be re-judged is a summary that
cannot be corrected.*

---

# THE DRIFT — sub-quantum structure measured in a file that SHIPPED

Found by answering a question about a threshold, not by looking for it.

## The question that found it

The prominence run's maximum spread was **25.00 ms — exactly the chosen threshold.**
`vmsam-dev-sandbox` asked whether that was an **observed value** or a **boundary
effect**, since the two give the same number and mean opposite things, and named the
one-look test: the second and third largest spreads.

```
25.00   8.38   6.25   5.13   4.75   3.75   3.63   3.50
```

**All distinct. One value at 25.00, no pile.** Observed. The Wilson bound measures the
corpus, not the instrument's edge.

## What the maximum file actually is

```
t= 200  lag=36.88  prominence  58.6
t= 450  lag=43.50  prominence  70.0
t= 700  lag=44.88  prominence 163.4
t= 950  lag=61.25  prominence  98.7
t=1200  lag=61.88  prominence 121.2

fitted line: slope +27.10 ms per 1000 s, residual RMS 3.12 ms
```

**Monotonic, five of five, every probe at high prominence. A straight line explains it;
a step does not.**

> **This is a DRIFT, and the classifier cannot express the difference.** It tests
> *spread* — max minus min — **which is identical for a 25 ms step and a 25 ms drift.**
> *"0 of 70 carry a step"* is true and answers a narrower question than a reader will
> assume.

**The same summary-cannot-express-a-shape defect written into `scan_hidden.py` this
morning, reproduced in a classifier written tonight.**

## Why it is the campaign's result and not just a good measurement

**25 ms is below the ~62 ms quantum.** The pipeline could not have seen it at any
position, by construction — **and the file shipped.**

`vmsam-forensic` proved the refused corpus cannot reach that class **by any method**,
because its inclusion criterion is exactly what excludes it. This was not estimated
from inside that corpus. **It was measured in an artefact on disk — the only act
available, and it required leaving the set.**

## Four things NOT claimed

1. **One file.** Two others fit a line at 6.25 and 3.50 ms spread, so it may be a
   family; **three files is not a rate and none is computed.**
2. **Not shown audible or wrong.** What is shown is that it exists and that nothing in
   the pipeline could have measured it.
3. **Cause not established.** +27 ms/1000 s is a clock-rate signature, but a line
   fitted through a step tilts the slope, and this is five points. **Not called a rate
   difference.**
4. **My own drift hypothesis was refuted by my own measurement this morning** — first
   fit +34.6 ms/1000 s, residual spread 23.9 against 34.6 ms of effect, and it was an
   89 ms step at 1170 s. This one is 3.12 against 25 ms, *the ratio that one failed* —
   and the sample is smaller and I have been wrong about exactly this before.

## The corrected statement

> **0 of 70 measured files carry a step above 25 ms. At least one carries a
> sub-quantum drift of 25 ms across the programme, which the pipeline could not have
> seen and which shipped.**

The first sentence alone **is incomplete in a way that flatters the pipeline.**

## The shape test, deferred on purpose

**Not changing a classifier under a live run** — the 315-file run would end carrying
rows from two instruments with no way to tell which. It goes in afterwards, **on the
saved sequences, without re-measuring anything**, which is possible only because every
row stores its probes.

**Design requirement, from dev-sandbox and it is right:** *if the test can only say
"line or not line" it is a spread test with better manners.* There may be a third
shape — a piecewise drift, or drift on one track and not the other. **Report the
slope, the residual and the probe sequence; let the reader see the shape rather than
the verdict** — which is exactly what saved this file. And the shape classifier needs
the path-classifier discipline: **every sequence lands in a named shape, with
UNCLASSIFIED as a value rather than a fallthrough.**
