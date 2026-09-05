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

**State: coefficient measured, verdict reached, both on two real error files.
Not yet applied — that is dev-2's 009 and it closes when a track exists.**

### Measured — the coefficient, and it is NOT PAL

`vmsam-forensic` named ids 33 and 31 (folder 164, eng+fre, eac3). Their ten
recorded delays are **monotone decreasing** (id 33: 3968 -> 2108 ms) with a gcd of
**124 ms**, which matches my duration->quantum model for their 418 s / 454 s
windows. Regressing delay against window centre:

| id | slope | `speed_ratio` = master/candidate | ~95 % |
| ---: | ---: | ---: | :--- |
| 33 | -0.992431 ms/s | **1.000993** | [1.000953, 1.001034] |
| 31 | -0.993192 ms/s | **1.000994** | [1.000954, 1.001034] |

**Residual scatter about the fit is 38.3 and 41.0 ms against 124/sqrt(12) = 35.8 ms
predicted by pure quantisation.** The residuals are therefore almost entirely the
quantum, the linear model is not strained, and **no staircase is hiding under the
slope** — which is the thing this task warns about.

**The relation is `1001/1000`.** It lies inside both intervals; PAL's 1.042709 is
excluded by four orders of magnitude. **`1001/1000` and `24/23.976` are the same
relation, not two candidates** — 23.976 rounds 24000/1001, and 24/(24000/1001) =
1001/1000 exactly. This confirms `vmsam-forensic`'s NTSC finding on two files, and
I carry its hedge: 2.5 % of the corpus is a **floor**, not a count, because a
PAL-rate file drifts ~9 s inside a 214 s window, collapses to weak correlation, and
is structurally invisible to the method that found these.

### Measured — the verdict, by the policy's own method

`docs/AUDIO_SPEED_POLICY.MD` rules that the verdict comes from applying both
corrections and fingerprinting each against the master, at a 0.85 bar, with three
outcomes and a decline. Four arms, 1200 s window (0.099 % needs room: 0.41 s of
drift over the pipeline's 418 s, 1.19 s over 1200 s):

| arm | id 33 | id 31 |
| :--- | ---: | ---: |
| uncorrected (control) | 0.7411 | 0.7374 |
| **asetrate CORRECT** | **0.9588** | **0.9551** |
| asetrate WRONG (sign control) | 0.6569 | 0.6562 |
| rubberband | 0.9484 | 0.9453 |

**Verdict: `asetrate`**, both files, clearing the bar.

**The sign control is the point.** Weakness 3 of the policy records a sweep that
used `* r` instead of `/ r`, landed twice as wrong, and had rubberband win 33 of 33
with nothing internal able to catch it. Here the wrong-direction arm lands **0.084
below the uncorrected control** — it makes things worse, as a doubled error must.
So the direction handed to dev-2 is demonstrated, not asserted.

### Finding for the owner — the detector's margin does not carry from PAL to NTSC

`AUDIO_SPEED_POLICY.MD` measures rubberband paying a **~0.109** processing handicap
and losing by 0.278-0.313 against real masters. **Measured here the gap is 0.0104
and 0.0098** — an order of magnitude smaller.

Consistent with the document rather than against it: it already records that
against a real master `asetrate` reaches only 0.94-0.98 *"because real pairs carry
content and codec differences no correction removes."* Both arms press against that
same ceiling, which compresses the distance between them. At 4.27 % the losing arm
had far to fall; at 0.099 % it does not.

**The ruling stands** — `asetrate` wins on both files, and its justification is
physical (the exact inverse of the transfer) rather than statistical. **The limit
is that the margin which would detect an inverting case in this corpus is ~0.010,
not ~0.109**, so a detector built to the document's expected separation could miss
one. Two files agreeing in direction and magnitude is evidence the margin is real;
it is not a rate, and I am not reporting it as one.

`AUDIO_SPEED_POLICY.MD` is frozen. This **extends** it by measurement and reaches
the owner as a finding in `AGENT_NOTES.MD`, never as an edit.

### What dev-2 gets

`verdict: "asetrate"`, `speed_ratio: 1001/1000` **as the exact rational**, not my
point estimate — my measurement is consistent with it and *less precise than it*,
so propagating my noise into a file would be worse than using the known relation.
`ratio_precision: 3 decimals (1.001)`; the interval is +/-4e-5, so the fourth
decimal is already uncertain. dev-2 tags the factor it **applied** to full
precision (`SPEC_ZONE_A.MD` §4, correct as written) and carries my interval in the
repair report, where it can say what the tag cannot.

### Strengthened to all eight files, from forensic's final table

`forensic/data/classified_v2.json` classes **8** files as `speed mismatch`, all
`family_v2 = "speed mismatch (NTSC)"`, **all folder 164**. Regressing delay on
window centre, with the uncertainty floor taken from the **quantum** (`q/sqrt(12)`)
rather than from the residuals alone — the floor my own quantum finding implies:

| id | slope ms/s | ratio | resid sd | q-floor |
| ---: | ---: | ---: | ---: | ---: |
| 26 | -1.022438 | 1.001023 | 39.8 | 35.8 |
| 27 | -1.014954 | 1.001016 | 55.2 | 35.8 |
| 29 | -0.981684 | 1.000983 | 35.5 | 35.8 |
| 31 | -0.993192 | 1.000994 | 41.0 | 35.8 |
| 33 | -0.992431 | 1.000993 | 38.3 | 35.8 |
| 35 | -1.015700 | 1.001017 | 39.2 | 35.8 |
| 37 | -0.984921 | 1.000986 | 38.3 | 35.8 |
| 113 | -0.995384 | 1.000996 | 38.0 | 35.8 |

**Mean 1.001001, 95 % [1.000990, 1.001012].** `1001/1000` lands essentially dead
centre; PAL's 1.042709 is excluded by four orders of magnitude.

**Two caveats that travel with the number.** All eight are **one series**, so eight
files are not eight independent observations — the family is one show's encoding
habit seen eight times. And the **between-file scatter is 2.8x the pooled standard
error**, so the eight disagree by more than quantisation explains: the honest
interval is the scatter-based `[1.000990, 1.001012]`, not the tighter pooled one.
Within each file the residual sd sits at or just above the 35.8 ms quantisation
floor, so quantisation dominates there, as expected.

### CORRECTION after the promotion — the corpus is NOT NTSC-only

`vmsam-forensic` has corrected its own earlier finding: there are **19 PAL files**
in the corpus, roughly **twice** the NTSC population. Everything measured above
covers the **8-file NTSC subfamily only**, and the mean of 1.001001 is that
subfamily's coefficient, not the corpus's.

**The hedge carried with the original result is what turned out to be load-bearing**
and it is restated rather than quietly dropped: *2.5 % is a floor, not a count,
because a PAL-rate file drifts ~9 s inside a 214 s window, its correlation
collapses, and the method that found the NTSC eight is structurally unable to see
one.* PAL files were hiding exactly where that hedge said they would be.

**A stated limit of my own instrument.** forensic reports the 19 carry no usable
delay measurement — a 4.27 % relation destroys correlation, so stage 1, forensic's
prominence measure and **`change_point_locator` alike find nothing on them**. The
duration ratio is the only evidence. My locator will decline them, and that decline
is correct rather than a gap: it cannot measure what does not correlate. dev-2's
speed path handles them from the duration ratio, with a synthesised full-coverage
segment, because demanding segments would make the intended case look like failure.

So the honest scope of this task's number: **`1001/1000`, `asetrate`, for the NTSC
family. PAL is a separate population, larger, and measured by a different route.**

### Blocked

Nothing. The measurement half of objective 3 is done on two files. It closes when
dev-2's 009 produces a resampled track that its post-mux verifier accepts.

---

## Objective 4 reframed: does a relation exist that is neither PAL nor NTSC

The owner's question became *"find whether one exists"* rather than *"measure a
third coefficient"*. A classification tolerates scatter that an estimate does not.

### The instrument, and why it is not one already used

`vmsam-forensic`: *"the PAL ones are invisible to every slope- or correlation-based
method here, including mine — at 4.27 % a 60 s window accumulates 2.56 s of internal
drift and aligns with nothing."* True of every audio correlator in the campaign.

**A pHash video probe is not blind to them.** A PAL transfer preserves frames and
moves their timestamps, so content still matches exactly and only *where* it matches
changes. Displacement is measured at twelve positions spread across the whole file
and fitted:

    speed relation      -> displacement grows LINEARLY, slope = 1/r - 1
    content difference  -> displacement FLAT, whatever the duration ratio says

**Validated against ratios chosen by hand, at both relevant frame rates:**

| control | true r | measured | error | rms |
| :--- | ---: | ---: | ---: | ---: |
| `setpts` only, output 24000/1001 | 1.042709 | 1.042442 | **-0.016 %** | 0.07 s |
| `setpts` + `-r 25`, output **25/1** | 1.042709 | 1.042442 | **-0.026 %** | 0.07 s |

The second control exists because the first did not reproduce the corpus's
condition: ffmpeg kept the `setpts` output at 23.976 fps while every real PAL
candidate is 25 fps, and **a frame-rate-dependent systematic would be constant
across releases** — indistinguishable from the finding it would have manufactured.

### The answer

**19 PAL files are three releases, not nineteen samples.** Measured across the whole
file:

| release | n | mean r | sd | vs PAL |
| :--- | ---: | ---: | ---: | ---: |
| folder 110 | 11 | 1.044397 | 0.000701 | **+0.162 %** (8.0 sigma) |
| folder 172 | 7 | 1.042347 | 0.000541 | **-0.035 %** (1.8 sigma) |
| folder 54 | 1 | 1.043056 | — | *no percentage: n < 3* |

**No third relation.** Folder 172 reproduces nominal 25/23.976 to 0.035 %. Folder 110
sits **0.162 % above it, reproducibly, across eleven episodes with a spread of
0.0007** — but a relation would be shared between releases and this is not. **It is a
property of one release that nominal PAL does not describe.**

Neither implied frame rate is standard (source 23.938 fps, or target 25.04 fps), so
folder 110 is **not a different frame-rate pair** — which is what both entries in
`docs/AUDIO_SPEED_POLICY.MD` are.

### The finding that reaches the repair

**Resampling folder 110's eleven files by 1.042709 leaves 0.162 % residual drift —
about 2.3 s over a 1400 s episode**, against `TASKS/009`'s 100 ms tolerance. The
speed path has been verified only on a synthetic fixture built at one exact rate,
where the duration ratio and the content rate are the same number by construction,
so that test could not have detected this.

**The duration ratio is not a measurement of the content** (`vmsam-dev-2`'s wording):
it measures the container's endpoints and equals the content rate only under the
assumption that the file is the master at one rate with nothing added or removed —
the assumption in question. On id 46 a pure resample at the fitted rate predicts
1267.37 s against an actual 1270.64 s: **+3.27 s of extra content.**

**The file is one rate, so `resampled:<factor>` is a truthful shape.** Head [100,800]
gives 1.046110 +/- 0.0018 and tail [850,1250] gives 1.043569 +/- 0.0033 — **0.62
sigma apart**. A varying rate would have required a tail near 1.037 to average out to
the duration ratio; that prediction was written first and missed by 1.8 sigma.

### The band edge

The three files 1-2 milli outside `vmsam-forensic`'s PAL band read **flat**
(0.7-1.2 sigma from flat). All three are folder 103, and each master declares a video
duration **64-66 s longer than its own audio** — +4.47/+4.50/+4.59 %, against +0.00 %
for a real PAL master. Their PAL-shaped container ratio is that internal mismatch,
not a relation to the candidate.

**So the high edge bounds a gap rather than cutting a population — but the gap
immediately beyond it is populated by metadata artefacts**, and the edge clears its
highest member by 0.12 milli. Three non-speed files would have entered the PAL count
had the band been drawn slightly wider.

### Capability statement — carry this with any result from this instrument

**It classifies decisively and it does not produce trustworthy intervals on a single
file.** Per-file standard errors came out narrower than the spread of the same
quantity measured a second way **three times**: coarse versus fine on id 46, per-file
versus between-file across a release, and head-only versus full-range. Separations of
9-62 sigma between "speed" and "flat" are sound; anything resting on a single file's
error bar is not.

It also **cannot treat episodes as independent samples** — folder 110's 8.0 sigma is
one encoding chain measured eleven times — and **cannot resolve NTSC**, whose 0.0999 %
sits inside real-media scatter.

### What was dropped

- **An earlier claim that two releases agree at +0.24 %.** It was fitted over
  [100, 800] s, a range chosen so a fixed +/-45 s search window would never clip —
  **an instrument convenience that became an analysis choice without being decided.**
  Worse, it was a worse measurement: at t=100 that window was wide enough to find
  spurious matches. Scaling the window to position (+/-max(20 s, 6 % of t)) dropped
  id 46's fit residual from 0.97 s to 0.255 s, and folder 172's mean moved from
  +0.215 % onto nominal.
- **Re-running a file to check it.** id 289 returned a bit-identical value:
  **re-running a deterministic pipeline is not an independent measurement.**
- **A refined PAL constant**, per the capability statement above.

### What stays blocked

1. **Whether folder 110's +0.162 % has a cause.** It is real and reproducible; no
   standard frame-rate pair produces it.
2. **Which factor the repair should apply to folder 110** — 1.042709 leaves 2.3 s of
   drift, and the measured rate has not been through `vmsam-dev-2`'s verifier.
3. **NTSC-scale relations**, unreachable by this instrument.

---

## RETRACTED: folder 110's +0.162 %. It was a straight line fitted through a splice.

The section above reports folder 110 at 1.044397, +0.162 % above nominal, 8.0 sigma.
**That figure is withdrawn. The rate is nominal 25/23.976 and the apparent offset was
a modelling error, not a measurement.**

### The measurement that settled it — residual after correction, not rate

Every rate measurement in this campaign fitted a slope and was limited by its own
fit. This one does not fit a slope at all: **correct the candidate by a trial rate
and measure what is left.** A correct rate leaves a flat line, so the precision comes
from *flatness* rather than from a fitted gradient.

id 46, candidate audio corrected by nominal PAL, twelve positions:

    plateau 1   5 points   mean -1879.3 ms   spread 9.6 ms over 391 s
    plateau 2   7 points   mean -3176.5 ms   spread 7.7 ms over 586 s

**Two flat plateaus and a step, not a drift.** Flat to 7.7 ms over 586 s bounds the
rate to **0.0013 %** — two orders of magnitude better than any slope fit produced on
the same media with the same tooling.

| id | folder | step | bracket (s) | plateau flatness | bounds rate to |
| ---: | ---: | ---: | :--- | ---: | ---: |
| 46 | 110 | **-1297.2 ms** | (488.7, 586.5) | 9.6 / 7.7 ms | 0.0025 % |
| 51 | 110 | **-1272.1 ms** | (584.9, 682.4) | 2.6 / 3.4 ms | 0.0005 % |
| 63 | 110 | **-1672.2 ms** | (561.4, 654.9) | 3.5 / 3.7 ms | 0.0007 % |
| 288 | 172 | **+195.8 ms** | (532.7, 639.2) | 6.0 / 86.8 ms | 0.0014 % |

**If the rate were the 1.045 that three line fits produced**, correcting by nominal
would leave ~1.1 s of drift across plateau 1. Observed: 2.6-9.6 ms. **Ruled out by
92x to 424x on every file.**

### Why three independent instruments agreed and were all wrong

| instrument | signal | result |
| :--- | :--- | ---: |
| `vmsam-dev-1` pHash video displacement | video frames | 1.045008 |
| `vmsam-dev-2` loudness-envelope correlation | audio energy | 1.045066 |
| `vmsam-forensic` chromaprint rate search | spectral fingerprint | nominal, but see below |

The first two agree to **0.006 %** on the same file and share no code, no library,
no feature and no signal path. **They share a model: both fit ONE LINE.** A -1626 ms
step across a 1075 s span tilts a single-line fit by **-0.151 %**, and the disputed
effect was +0.162 %.

**`vmsam-dev-2`'s formulation, which is the durable one: INDEPENDENCE OF INSTRUMENTS
IS NOT INDEPENDENCE OF ASSUMPTIONS.** A single-segment fit is not a property of pHash
or of envelopes — it is a property of deciding in advance that the answer is a line.
Three instruments, three modalities, same error, same direction.

`vmsam-forensic` reached nominal and separately withdrew its own numbers: its trial
grid held only two nodes in the relevant region, `1/PAL` and `1/duration_ratio`, so
ten files returned a hard-coded constant and ten returned the duration ratio.
**Neither method had measured the rate. Its answer was right for a reason that could
not have been wrong.**

### What this replaces the finding with — a campaign first

`TASKS/009` defines `chimeric+resampled` and records that it had never been observed
on a real file. **id 46 is one**: nominal PAL plus a ~1.30 s splice. Three of three
folder-110 files carry one, all bracketed between 490 and 680 s — a consistent
position, so it reads as an act break common to the release rather than per-episode
damage. Folder 172 carries a much smaller one.

**This dissolves folder 110 versus folder 172 with no relation involved.** 110 tilts
because its splices are large; 172 barely tilts because its are small. **Objective
4's null stands and is better supported: there is no third relation, and the
apparent one was a splice.**

### The order of operations, which is the operational finding

**These steps are invisible before correction.** At 4.27 % every correlator in the
campaign collapses — stage 1, the prominence measure, and this locator — which is
why `TASKS/009` records that PAL files carry no usable delay measurement, and why
nobody had seen the splices. **The change point becomes findable only after
resampling.**

    resample by the measured rate  ->  THEN locate change points  ->  THEN splice

`vmsam-dev-2` confirms this lands entirely on the measurement side: its assembler
already applies a scalar ratio first and cuts the resampled timeline, and its
single-full-coverage-segment branch fires only when a plan carries **no** segments.
So id 46 needs `verdict: asetrate`, `speed_ratio: 1.042709`, and **two segments at
-1879.3 and -3176.5 ms**, which runs through the existing path unmodified.

**Not yet a capability.** The locator has never been run on a corrected candidate:
it needs the resampled file as input and that file is currently built inside the
assembly and never exposed. Requested as a remuxed candidate, which needs no code
change on either side. **Until it has run end to end this is a design.**

### The defect in my own analysis, recorded because it recurred at three scales

A single-segment model fitted through a piecewise structure — the same defect as
id 266's short segment this morning, and the same as a verifier that probes only the
four longest pieces. **The evidence was in my own residuals**: a -2.70 s outlier at
t=600 sitting exactly at the step, which I explained away as a tie-heavy point.

The parameter version of this has a test — *what would it have to be for the answer
to change, and did anyone compute that.* **The model version does not, because
nobody chooses it.** I fitted one line because one line is what the instrument
emits, and never asked what the data would look like if the file were not one line.
