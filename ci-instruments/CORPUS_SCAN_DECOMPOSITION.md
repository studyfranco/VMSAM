# Why 23 % of the corpus could not be measured — three causes, separated

**PRIVATE.** Ids and folder ids only.

## The observation

The 315-file corpus scan reported **72 of 315 = 23 % instrument-did-not-run**, against
**10 of 98 = 10 %** on the earlier sampled run of the same corpus.

## Three hypotheses, and what happened to each

**H1 — the sample was drawn from files eligible for the ten-window geometry.**
**REFUTED BY TEST.** Short files are 67 % did-not-run but there are 9 of them; the 306
long files are still 22 %.

**H2 — my baseline search is a fixed ±60 s and cannot find a larger offset.**

```
recorded |delay| < 10 s    n=222   did-not-run  4 %
recorded |delay| 10-30 s   n= 13   did-not-run  8 %
recorded |delay| 30-60 s   n= 13   did-not-run 15 %
recorded |delay| >= 60 s   n= 53   did-not-run 98 %
52 of the 72 failures in that top bucket
```

**A monotone dose-response across four buckets, 98 % in the top one, 72 % of failures
accounted for. I would have published it as the cause.**

So the baseline search was sized from the record's own delays — up to ±300 s, giving
id 46 a ±259 s search against a recorded 181 s offset — and the 54 affected files
re-run. **Four for four still failed, at 0.005–0.013 correlation.**

> **CONFIRMED BY CORRELATION AND REFUTED BY INTERVENTION.** The association was real;
> the mechanism was not the one it named. **The only thing that caught it was fixing
> the thing and watching nothing happen.**

**H3 — the failures are three different things.** Measured on fields already in
`audio-durations.json`:

| group | n | speed_candidate | median duration ratio |
| :--- | ---: | ---: | ---: |
| folder 103, known misfiled | 26 | **0** | 0.99965 |
| folder 110 | 11 | **11** | **1.0425** |
| folder 172 | 7 | **7** | **1.04244** |
| other | 10 | 1 | 1.00001 |

`25 / 23.976 = 1.04271`.

- **26 MISFILED** — same rate, different content. Nothing to correlate.
- **18 RESAMPLED** — correct content at a 4.27 % rate difference. A constant-lag
  correlator **cannot** work on these: the probe drifts five seconds internally over
  120 s, so no single offset fits the window. **A structural limit of the instrument,
  correctly reported.**
- **10 residue** — ratio ~1.0, not explained here.

## The derived finding, which outranks the rate

> **THE INSTRUMENT'S FAILURE IS A DETECTOR.** A `did-not-run` on a **same-rate** pair
> means the two files are not the same programme. **26 of 26** known-misfiled files
> produce it — and the original misfiling analysis needed three independent methods
> and a folder-level investigation to reach those same 26.

**The converse is NOT claimed.** Ten files fail for reasons not established, so a
`did-not-run` is not proof of misfiling. **One direction measured, the other not.**

## Two errors on the way, both mine

**I tested H3 against a population from which its own cases had been deleted.** The
first run of that test gave 23 % of 13 speed candidates — a weak signal I set aside.
It ran on the rows *remaining after I removed the 54 for re-scanning*, which is exactly
the set with the resampled files taken out. **The subset was the answer and I had built
it**, twenty minutes earlier, and had stopped thinking of the removal as a filter.

**Folders 110 and 172 have a signature identical to folder 103** — clustered,
whole-folder, non-correlating. **Reporting "the misfiling extends to two more folders"
would have been wrong in the most plausible possible way**, and one column already in
the data splits them.

That is the fourth time in one session that a cluster looked like a finding and was a
second mechanism: the four-bucket dose-response, six false `STEP` rows, a 67 %
did-not-run rate that was my own stream pairing, and this.
