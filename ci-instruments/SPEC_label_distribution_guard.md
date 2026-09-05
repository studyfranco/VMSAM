# Spec — the guard that would have caught tonight's worst defect in its first minute

**Not built. Specified.** The 315-file run is live and instrumenting under a live run
is the thing I have twice refused for good reasons.

## What it would have caught

`run_full.py` hardcoded `ms=0, cs=0`. On 25 randomly sampled pairs, **stream 0 carries
a different language on each side for 18 — 72 % of the corpus.** The run reached 91
files reading **67 % instrument-did-not-run**, against **10 %** on the sampled run over
the same population.

**I caught it by looking at the labels while waiting for something else.** Two of
today's saves came from noticing rather than measuring — this, and a spurious peak
landing 0.5 ms from a true answer. **Two saves by attention is not a method, it is the
absence of one.**

## The guard

> **Compare a run's label distribution against a prior run of the same instrument on
> the same population. Fire when they diverge beyond chance.**

```
prior:   runs/scanA3-out.json + runs/pass3-out.json   (98 files, same corpus)
live:    runs/full-out.jsonl                          (appends as it goes)

after every N rows (N ~ 20):
    p = share of DID-NOT-RUN labels in the prior run
    k = DID-NOT-RUN count in the live run's first n rows
    two-sided binomial tail of k given n, p
    fire when it falls below, say, 1e-3
```

Tonight: prior `p = 0.10`, live `k/n = 61/91`. **The binomial tail is far below any
threshold anyone would choose — it fires on the first check, at n = 20.**

## Requirements, each from a defect paid for today

- **DID-NOT-RUN is the union of `partial-coverage`, `baseline-not-found`, `unreadable`,
  `instrument-failed`, `no-shared-language`** — the third-outcome set, which is exactly
  what a stream-pairing defect inflates.
- **Fire is a WARNING that names both distributions, never an automatic kill.** A real
  population difference looks the same as a defect, and the difference is a judgement.
  A guard that kills a run on a legitimate corpus shift is an obstacle, and obstacles
  get bypassed.
- **State the prior run's identity and its n.** A comparison against an unnamed
  baseline is a number without a denominator.
- **`COULD NOT COMPARE` when no prior run exists on that population** — never a silent
  pass. The first run of any instrument has no baseline and must say so.
- **It cannot detect a defect that was present in the prior run too.** A baseline is
  not ground truth; two runs agreeing means they share whatever is wrong. **State this
  in the output, not the docstring.**

## Why this one and not a general harness

It is **one comparison, on data already on disk, against a run that already exists.**
The rule earned tonight is dev-1's: *the second time a class catches you, build the
thing that would have caught it.* **For the wrong-stream-pair defect this was the
third** — `hunt_subquantum` at ~17:00, then `run_full`, having written `matched_pair()`
in between and not used it.

> **The guard was not merely known. It was written, named, and in scope, and I
> re-derived the wrong answer from a file I had already loaded.**
