# Agent notes — logbook

One entry per decision that a later reader would otherwise have to re-derive.
Newest first.

**What goes here:** an architectural choice and why the alternatives lost; a
measurement that changed a decision; a defect found and what it taught; a blocked
need and what was done instead.

**What does not:** progress reports, task status, anything regenerable from the
code or from `git log`. If it is a measured fact rather than a decision, it belongs
in a findings document.

**Format.** `## YYYY-MM-DD — one line saying what changed`, then the decision, then
*why the alternatives lost*. Numbers carry the command that produced them. Files are
cited by opaque id, never by media name (`AGENT.MD` → Privacy).

Entries are **append-only**. Superseding an earlier one means adding a new entry that
says so and why — never editing the old one. A record that can be rewritten is not a
record.

---

## 2026-09-03 — an independent instrument aimed by a dependent one is not independent

**My own blind spot, one layer above the one I had just finished documenting.**

I built an FFT bisection specifically to check the pipeline's measurements from
outside it. Then I probed a file **only from 120 s onward — because 120 s is
`begin_in_second`, where the pipeline's own windows start.** Another agent profiled
the same file from zero and found a change point at ~56 s that I had walked past.

The instrument was independent. **Its aim was not.** I used the pipeline's geometry
as my map of where the interesting parts were, and so inherited exactly the blind
spot I had spent the afternoon documenting in someone else's module.

> **Independence of method does not survive dependence of aim.** Ask separately: is
> my instrument independent, and is my *search* independent?

**And the file was worse than either of us thought.** It has **two** change points and
**the offset returns to where it started** — plateau A, an island of ~91 s at a
different offset, then plateau A again. Confirmed by bisection from both sides;
transition 1 narrowed to a 4-second bracket, transition 2 to two seconds, the step
501 ms at both.

**The consequence is the sharp one:** a file whose offset departs and returns reads
**constant** to any window spanning both transitions — and every window the pipeline
produces starts after the first one. So **window agreement is evidence about the
windows, not about the media.** Two of my own classification buckets
(`constant-offset`, `floor-constant`) were built on exactly that inference and are
unreliable wherever the geometry cannot resolve the region.

That is a second, independent reason the corpus record cannot be read as a census —
not only are head regions unsampled, but a returning offset is actively invisible.

## 2026-09-03 — an instrument that cannot fail loudly will fail quietly

Four instrument defects in one day, three another agent's and one mine, with a single
shape: **the instrument returned a plausible number rather than an error.**

| defect | what it returned |
| :--- | :--- |
| a rounding limit stored to 4 decimals | "ratio exactly zero" — meaning *cannot see a ramp*, read as *no ramp* |
| a CSV parse tripping on trailing commas | plausible durations |
| a packet-count cap misread as a stream ending | **49 defective masters**, of which the dense files were all false positives |
| mine: a 4 s probe with a 4 s max-lag, slicing out of bounds | correlation exactly `0.0000` at every probe, lag pinned at the search bound |

**Mine was the easy one** — eight identical zeros are not a plausible measurement.
Theirs were expensive precisely because the numbers looked reasonable.

**The defence that worked in every case was the same: a cross-check that could
contradict the instrument.** They distrusted their 49 because their own probes had
earlier found matches *past* where the sweep claimed the audio stopped — two of their
own results could not both be true. That is what caught it, not inspection.

> **Build at least one path to the same fact that does not share code with the
> instrument, and compare them before believing either.**

And a retraction that arrives **before** the other agent acts is worth more than a
correct result that arrives late: I would have started extracting frames from eight
masters, found audio exactly where the sweep said there was none, and might have
concluded my *frame method* was unreliable rather than their sweep.

## 2026-09-03 — the cure for the fixture trap: ground truth that does not share an author with the instrument

**The most reusable thing this campaign has produced, above any individual defect.**
Two agents converged on it after four separate fixture failures in one day; the
question at the end is dev-1's formulation and it is sharper than the observation
that produced it.

Every one of the four failures had the same shape: a control whose pass was
guaranteed by how it was built rather than by the thing under test.

> **The antidote is ground truth that exists independently of the thing being
> tested.** A constructed fixture cannot have that property *by construction*: a file
> against a copy of itself has an expected answer **you supplied**, so the fixture and
> the instrument share an author — and therefore share his blind spots.
>
> **The test to apply before trusting any validation set: WHO PRODUCED THE EXPECTED
> ANSWER, AND COULD THEY HAVE PRODUCED IT WRONGLY IN THE SAME DIRECTION AS THE THING
> UNDER TEST?**

That question would have caught all four before any of them ran — a clean-edge
estimator needing correlation to reach 1.0, a fixture with a zero base offset, a
content screen whose positive control had zero offset, and a bisection with reversed
bounds. **It is a check, where the list of instances is only a catalogue.**

**Demonstrated the same day it was written.** Of three early-step files, two had
ground truth from four independent instruments and one rested on the measurement of
an agent whose absolute values had just been shown to carry a bias — i.e. the
expected answer could have been wrong *in the same direction* as the thing it would
test. That file was bisected on the strength of the question alone, and **its ground
truth moved**: my direct measurement and the pipeline's own recorded residual agree at
**837 ms and 833.5 ms** by wholly separate routes, against the earlier **750 ms**.

**Run the check on the weakest leg, not the most interesting one.**

**And it corrects `AGENT.MD` § Target-first in one sentence.** That section already
says to build the expected file by hand first. Another agent's fixture error showed a
hand-built target can itself be wrong, and this says why: **the hand that builds the
target is the hand that builds the instrument.** Both corrections fit in one line.
`AGENT.MD` is frozen — **a finding for the owner.**

## 2026-09-03 — the stage-1 record is structurally blind to the first ~227 s of every episode

dev-1's computation, checked against my parse of all 315 logs. **The most
consequential structural finding of the day, and it invalidates a class of
conclusion rather than a number.**

The ten-window geometry starts at `begin_in_second` = 120 s and each window is twice
the spacing. A step is visible only if some window is majority-pre-change while the
next is majority-post — which requires `T > begin + L/2`:

| shortest audio | first visible change point |
| ---: | ---: |
| 1195.7 s | **206.0 s** |
| 1420.1 s | **227.0 s** |
| 2545.5 s | **329.0 s** |

So the first ~227 seconds of a 24-minute episode are **structurally invisible** to
stage 1 — the 120 s before any window starts, plus the first half of window 0. **A
trimmed head is precisely a change point near the start.**

**Checked in the records**, parsing all 315 stage-1 matrices with window order kept:

```
294  step somewhere in the body (fully visible)
  4  flat, no step visible at all
  2  WINDOW 0 ONLY differs from a constant rest
  1  last window only differs
```

**Two files corpus-wide** show the head-step signature. That is implausibly few for a
family another agent measures at ~10 % — and the reason is that the signature mostly
*cannot appear*.

**The decisive detail.** The three files whose head steps were independently measured
(501, 750, 861 ms) appear **nowhere** in that count, because their logs carry **no
stage-1 matrix at all**: they were refused at stage 2, so the dump was never written.
Their step is visible only in the stage-2 residual's window 0.

**The consequence, and it is the reusable part:** *any corpus classification built on
the stage-1 delay matrix inherits a blind spot over the head of every episode.* Mine
did. It cannot be fixed by reading the record more carefully, because the record does
not contain the information. Only an instrument with its own geometry — a probe grid,
a bisection — can see these.

**Stated as a hypothesis, not a result:** this would undercount *trimmed edges*
specifically, by construction rather than by chance, which is the direction of the one
gap another agent flagged as uninvestigated. **I have not counted how many trimmed
heads exist and cannot**, from a record that structurally cannot show them — which is
the whole point. The direction is explained; the magnitude is not.

**Why this entry exists rather than a number:** I spent the day building
classifications on that record, and three separate findings of mine rested on it. The
finding is not "the count is wrong" but "a whole region of every file was never
sampled, and nothing in the record says so." A reader who trusts the matrix has no way
to discover that from the matrix.

## 2026-09-03 — CORRECTION: the "one bad window" reading was wrong. The outliers are real steps.

**This supersedes the stage-2 entry below, and the remedy I endorsed would have
produced silently wrong files.**

I reported four files as *"nine windows agreeing to two milliseconds, refused because
one bad window dominates the variance"*, and endorsed the remedy of **dropping the
outlier window**. Another agent pushed back; I tested it and they are right.

**The outlier window is not a bad measurement. It is a genuinely different delay,
because content is trimmed or added at the head of the file.** Bisecting one file's
first comparison window with an independent FFT correlation, streams pinned, no
shift:

```
[120,132] +1453.9 ms  corr 0.8182   PRE
[140,146] +1453.9 ms  corr 0.7821   PRE
[144,150]  +949.5 ms  corr 0.0585   spans the boundary
[148,154]  +952.9 ms  corr 0.5291   POST      (as do all later windows)

CHANGE POINT 146-148 s.   STEP = 501.0 ms.
```

Not noise: the pre-step correlations of **0.71–0.82 are the highest anywhere in that
pair** — the body only reaches 0.44–0.50.

**Three instruments sharing no code now agree**: the pipeline's own stage-2 log
(window 0 sitting 501 ms from the other nine), my bisection (501.0 ms), and the other
agent's probe grid (~567 ms).

**Why the remedy was harmful.** Dropping the window and merging on the remaining nine
gives a uniform +952.9 ms — correct after ~147 s and **wrong by 501 ms before it**. A
plausible file, silently wrong at the head, passing every per-plateau check. Exactly
the failure shape this campaign exists to prevent.

**The sentence that cost me this**, and it is the transferable part:

> *"Nine windows agreeing to 2 ms is an excellent delay."* True, and irrelevant.
> **They agree because they are all on the same side of the step.** A tight cluster
> is evidence of CONSISTENCY, not of COMPLETENESS, and I read the first as the
> second.

The same error in general form: a statistic computed over samples that all sit on one
side of a boundary describes that side, not the population. Tightness was measuring
my sampling, not the media.

**What survives, in the other agent's framing, which is better than mine:** stage 2's
variance test **cannot distinguish a real step from a bad window and refuses both
identically**. For these files the refusal is **correct** by `AGENT.MD`'s standard —
there genuinely is no single constant offset — and it is precisely the case zone A
exists to take over. The defect is not that they are refused; it is that **nothing
downstream is told which of the two it was**, so a repairable trimmed edge and an
unrepairable mess leave through the same door.

**Consequence for classification:** "measured, bad answer, **unrepairable**" and
"measured, bad answer, **repairable**" are different rows. My five-way split of the
no-matrix bucket collapsed them; the six-way version is theirs.

## 2026-09-03 — the stream the comparison uses is not the first stream, and all three of us assumed it was

**Three separate wrong-stream incidents in one day, by three agents, on three
different instruments.** That is a pattern, not three mistakes.

| agent | instrument | what was assumed | consequence |
| :--- | :--- | :--- | :--- |
| forensic | `fpcalc` default track | the default is the comparable track | "no match" on a file that genuinely matches — *"I could not measure"* misread as *"different content"* |
| me | `a:0` for a per-window energy measurement | the first audio stream is the compared one | **a clean negative I was one check away from reporting** |
| dev-1 | a single-stream locator | one stream per side | two agents measured 27.5 ms apart and each thought the other was wrong; it was two same-language streams in one file |

**Why it keeps happening.** The engine picks a *common language* and compares tracks
of that language on both sides. On a master carrying `eng, jpn, chi`, the compared
stream can be `a:2` while every convenience tool — `ffprobe` defaults, `fpcalc`,
`-map 0:a:0` — reaches for `a:0`. The tools' default and the engine's choice
coincide often enough to feel safe and differ exactly when a file has several
languages, which is most of this corpus.

**The rule:** *before measuring anything about "the audio", resolve which stream the
engine would compare and pin it explicitly.* It is two `ffprobe` calls — the
folder's `original_language`, and the language tags on both sides.

**What caught mine, and it is the transferable habit:** I was about to report *no
spike on any of three files*, which would have closed a live question in the wrong
direction with nothing downstream to reopen it. The check that saved it was asking
"which stream would the comparison actually use?" **before** sending a negative. A
negative result deserves the same verification as a positive one — arguably more,
because nobody re-opens a closed question.

## 2026-09-03 — measured: stage 2 overruled a correct stage 1 and refused the file

Recorded because it **inverts a design intent stated in a frozen document**.

`AGENT.MD` says stage 2 exists to *correct* stage 1: fingerprint peaks put stage 1
off by roughly ±250 ms, stage 2 runs unconditionally and fixes them, and a residual
beyond 500 ms means stage 1 was not merely imprecise but wrong.

On one file, measured by dev-1 absolutely and unshifted, per window:

- **stage 1 (chromaprint) was right in all ten windows**, holding one plateau, with
  its *lowest* fidelity on the disputed window — it knew it was less sure and was
  still correct;
- **stage 2 (FFT) was wrong in exactly one window, by ~850 ms**;
- that single wrong value raised the variance past the threshold and **refused the
  file**, whose other nine windows agree to ~2 ms.

**The correction ran backwards.** Whatever the cause, the refusal on that file is
not evidence about the media.

**The cross-check that makes it solid**, and it is joint: their unshifted −567 ms
minus chromaprint's −1375 ms predicts a residual of **+808 ms**; the recorded log I
had parsed independently says **+788 ms**. Two agents, two extractions, two code
paths, agreeing to 20 ms.

**What is NOT established.** dev-1 proposed the cause was FFT being dragged by the
loudest passage, having measured 2.5× the energy in that window. I checked the other
two files in the symptom group on the correct comparison streams: one shows 1.53×
and one shows **0.77×** — a window *below* median energy carrying the outlier, which
that mechanism cannot explain. So: **mechanism established for one file, symptom
shared by three, cause of the other two unknown.** They accepted the refutation and
recorded it.

The engine site is `mergeVideo.py:582–594`, outside every tagged zone — **a finding
for the owner, not an edit.**

## 2026-09-03 — retraction: I reported a checker defect that was not one. The rule was right.

**I told the Lead twice that `check_write_zones.py` was wrongly blocking a
`config.ini` addition, and that was false.** I also told the agent affected that
their reading "matches mine" and called it a checker artefact. I endorsed a
diagnosis I had not tested.

**What is actually true.** `src/config.ini` has **no trailing newline** — its last
byte is `c`, the last line being the `audio_sync` entry. So an **append** rewrites
that line: git records the old one removed and a newline-terminated copy added.
`WRITE_ZONES.MD` §3 says existing variables are frozen — *not their value, not their
name, not their position* — so a modified last line is a **real violation**, and the
checker refusing it is the checker working.

**Verified myself, in a throwaway clone, rather than taking anyone's word a second
time:**

| approach | checker | removed lines | last 40 bytes |
| :--- | ---: | ---: | :--- |
| append the new section | **exit 1** | 1 (`audio_sync=…`) | changed |
| **insert before `[software]`** | **exit 0** | **0** | **byte-identical** |

Parsing both files confirms the two existing sections are semantically unchanged and
the new section is present. So the addition was always shippable; only the *method*
was wrong. The fix came from another agent, was verified by the agent affected, and
I then verified it a third time independently — which is the only reason this entry
can state it as fact rather than as another relayed claim.

**Why I got it wrong, and it is the reusable part.** The failure mode is the one I
had already recorded twice today in other people's work: I accepted a plausible
mechanism without running the two-line experiment that would have separated it from
the alternative. Worse, I *amplified* it — a blocked agent's guess became, in my
report, a defect in the campaign's own tooling. **A relayed diagnosis is not
evidence, and passing it upward launders it into one.**

**The rule I am taking from it:** before reporting that a shared instrument is
wrong, reproduce the failure against the instrument yourself. The cost here was one
throwaway clone and ninety seconds.

## 2026-09-03 — correction: two of my visual verifications were wrong, and "same" hid the reason

I gave another agent nine visual verifications as ground truth for their classifier.
**Two were wrong.** They challenged them instead of adopting them, which is the only
reason it was caught.

**The error.** For one file I had written *"the same scenes moments apart, an
elevator interior in both, the same character against the same background"* —
**scene-level**, and unambiguous. For another I wrote *"same setting and character
designs across three timestamps"* — **series-level**, which holds for *any* episode
of that show. I gave the two the same weight. Both came out as the word **"same"**.

Re-tested by extracting the candidate at three timestamps and the master at **both**
alignments the other agent's audio had matched: **not one scene matched at either**.
Same series, different episode. A second file behaved identically.

**A third family, previously unnamed.** Both files are the same folder, consecutive
episodes:

| what is wrong | |
| :--- | ---: |
| wrong **series** filed into a folder | 26 |
| wrong **series**, filename also lies | 1 |
| **right series, wrong EPISODE paired** | 2 verified |

*Same series, different episode* is invisible to every instrument built this
campaign: the names match, the series matches, the character designs match. Only the
scenes and the dialogue audio disagree. The other agent's **theme-only** signature —
shared opening and ending music with zero dialogue correlation — is the sharpest
detector anyone has for it, because that is exactly what two episodes of one series
look like.

**Why it belongs with misfiled content and not with weak correlation:** no constant
offset exists, no repair can succeed, and a repair that appeared to succeed would
splice **the wrong episode** into the library. *Weak correlation* means same content
that is hard to correlate — the opposite handling. How many more exist is
**unmeasured**; two are verified and nothing has swept for the pattern.

**The decision this produces**, and it is the one instrument failure today that was
my own judgement rather than code:

> **Record a verdict's evidence LEVEL beside its conclusion, not just the
> conclusion.** "Same series" and "same scene" are different claims. Written down as
> "same", they are indistinguishable a day later — including to the person who wrote
> them.

Had each verification been labelled `scene-level` or `series-level` when it was
made, the weak ones would have been visible without anyone having to challenge them.
The other agent's classifier scores 9 of 9 against the corrected set; both
"conflicts" were mine.

## 2026-09-03 — a correct observation generalised one step too far deleted the instrument that would have caught the next error

The sharpest failure mode any of us hit today, and the hardest to see, because the
mistake was **evidence-based**.

The forensic agent had measured that duration ratio is unreliable as a speed
indicator: on the NTSC files, durations differ by 4.7–6.9 % while the content
drifts only 0.1 %. **That observation is correct.** They then generalised it into a
blanket withdrawal of the duration ratio as a speed indicator — and thereby deleted
the one cheap instrument that would have caught the error they made next, reading
eleven PAL-sped files as having no shared content.

**The correct scope, which they have now reinstated:** the ratio is usable for PAL,
where the 4.27 % rate difference is large against a few percent of content noise,
and useless for NTSC, where 0.1 % sits far below it.

**Why this is not the fixture trap** recorded above. That one was a control whose
pass was guaranteed by construction. This one is a *true measurement*, correctly
made, whose scope was widened by one step — and the widening removed a defence
rather than adding a false claim. Nothing in the reasoning looks wrong when read
back; only the boundary of the claim is.

**The decision it produces:** when a measurement retires an instrument, record the
**scope** in which it retires it, not the instrument. A withdrawal is a claim like
any other and inherits the limits of the evidence behind it.

## 2026-09-03 — the duration-ratio rule needs both directions, and my own count was over-inclusive

I published: *compute the duration ratio before concluding "different content" from
any correlation measure.* One-directional, and it invited the opposite error.

The forensic agent supplied the missing half. Both are needed:

> 1. **A collapsed correlation does not imply different content** — a speed relation
>    destroys correlation by construction, so check the ratio.
> 2. **A speed-like ratio does not imply a speed relation** — content differences
>    change duration too, so check that the correlation *actually collapsed*.
>
> Call it a speed pair only when **both** hold.

**Applied to my own numbers it corrected them.** I had reported 30 speed candidates
at a ratio tolerance of 0.004; with the guard, **19** — exactly the other agent's
set, reached independently, with no disagreement in either direction.

The instructive loss is a nine-file cluster sitting *tightly* at a consistent
ratio, which looks precisely like a rate — but correlating at 0.95+. A genuine rate
difference of that size could not do that, so the duration difference is **content**
(Blu-ray against streaming, where opening and ending lengths differ). **A tight
ratio cluster is not by itself a speed signature**, which is the caveat my
one-directional rule invited someone to miss.

**And the same agent's spread test became a fourth independent leg under my
misfiling finding**, which I had not thought to make: a speed relation is *one
rate*, so a speed population clusters. The PAL folder spans 0.00127; the folder I
identified as misfiled spans 0.0458, thirty-six times wider. It cannot be a speed
population.

## 2026-09-03 — no correlation instrument can separate "different content" from "speed mismatch", and three of us proved it

**A speed relation destroys correlation by construction**, so a genuinely
repairable speed pair reads exactly like unrelated media. Three agents hit this in
one day with three different instruments, and it nearly cost the campaign a whole
repair family.

| instrument | verdict it gave | what was true |
| :--- | :--- | :--- |
| the pipeline's own stage-1 fidelity | a 0.558–0.562 floor | `docs/AUDIO_SPEED_POLICY.MD` records that exact band for **uncorrected PAL** — the same floor unrelated content sits on |
| my full-file chromaprint peak | 18 of 45 floor rows "different content" | **speed pairs**, caught only because another agent warned me before I published |
| the forensic agent's 60 s-chunk prominence, 12 probes, two independent languages | one folder's 11 files at 0/12, "no shared audio" | **speed pairs — verified visually** |

**The decision, and it is nearly free:** compute the **duration ratio** before
concluding "different content" from any correlation measure. One `ffprobe` per
file. PAL/film is 25/23.976 = **1.042709**; the eleven flagged files all sat within
0.001 of it, while every correctly-behaving file beside them sat at ~1.002. The
observation offered as evidence of missing content — "runs ~53 s shorter than its
master" — *is* the arithmetic of a 4.27 % speedup.

**Why the obvious defences do not work.** A longer correlation window makes it
worse, not better: at 4.27 % a 60 s chunk accumulates 2.56 s of internal drift and
can align with no position in the master. And testing a second language does not
help — a French dub and an English dub of a sped episode are *both* sped, so both
read as absent. That control tests for a dub, not for a speed relation.

**Verified the way the rest of this was verified**, by a prediction stated before
looking: if it is a speed relation of the same content, the candidate at `t` must
show the same frame as the master at `t × 1.042709`, and the master at a naive `t`
must not. It did — the scaled frame was identical, the naive one a different shot.

**What it cost to nearly miss.** The corpus figure would have been ~14 % instead of
8.6 %, and eleven files that the speed-repair task exists to fix would have been
written off as junk. The cross-check that caught it was one agent telling another
that their floor might not mean what they thought.

## 2026-09-03 — the pattern behind three instrument failures in one day: a fixture that cannot fail

Two agents, three refuted instruments before midday, and they share one form. Worth
one entry because the pattern is reusable and the three instances are not.

> **A fixture built by copying one file cannot test anything that depends on the
> two files being different.**

| instance | the control | why its pass was guaranteed by construction |
| :--- | :--- | :--- |
| my sparse-frame content screen | a pair that was **frame-identical at zero offset** | sparse sampling only fails when there IS an offset; the control had none |
| dev-1's clean-edge estimator | a file against a copy of itself with a cut | it tested for correlation reaching **1.00000**, which only identical content can reach. On a real pair of different encodes, aligned content reads 0.93–0.96 and the competing hypotheses were not separable. Refuted on the first real file |
| dev-1's gap-start bisection | reversed `lo`/`hi` bounds | returned its own input bounds; caught because the number was implausible, not because the code objected |

Each guard fired, returned both answers, and was believed. Each was then wrong on
real input.

**The decision this produces**, and it is why the entry exists rather than three
apologies: `AGENT.MD` says *an armed guard that has never fired is
indistinguishable from a working one*. That is necessary and not sufficient. The
corollary we paid three times to learn:

> **A guard fired only on the easy case is indistinguishable from one that works.
> Before believing a control, ask what property of the FIXTURE guarantees the pass,
> and whether real input has that property.**

For a content or alignment instrument the property is almost always the same one:
the two sides being genuinely different encodes rather than one file and a copy.

**Why the alternatives lost.** More controls would not have helped — my screen had
two, both answers, a wide margin. A stricter threshold would not have helped
either. The only thing that would have caught any of the three is asking what the
fixture made impossible, which costs one sentence and no compute.

## 2026-09-03 — an instrument's answer is worth more when it does not share a code path with the thing it checks

Recorded as a decision because it changed what I chose to rest a headline on.

The misfiling conclusion could have been drawn from the recorded delay matrices
alone — they were available for all 315 files, they were free, and they agreed. It
was instead rested on three sources with **no shared code path**: release names and
video frames I looked at; audio fidelity computed by another agent from the
pipeline's own logs without reading a filename or a frame; and my own delay-shape
classifier.

That turned out to matter. **My delay-shape classifier had a defect** (it pooled
track pairs) and **the recorded delays themselves turned out to be residuals rather
than absolute offsets**, so two of the three legs were compromised within hours of
being used — and the answer did not move, because the leg that carried it was
frames on a screen.

**The rule:** when a headline number is going to be quoted by other people, do not
let its evidence share an implementation with the part of the analysis most likely
to be wrong. Cheap redundancy is not redundancy.

## 2026-09-03 — correction: my step classifier pooled track pairs, and I shipped exemplars from it

**Supersedes the classification in the misfiling entry below, not its conclusion.**

My first parser of the recorded stage-1 matrices flattened every
`(fidelity, offset_points, delay_ms)` triple into one list **across all track
pairs**. On four files each track pair was internally *constant* and the two
values were the **master's own two audio tracks sitting 125 ms apart** — a property
of the master's track layout, not a change point in the media. Pooled, that reads
as a textbook step, and **I recommended those four to the measurement agent as the
best change-point exemplars.** dev-1 caught it by parsing per track pair.

**Why the alternative lost, and it was not a judgement call — it was a bug.** There
was no reason to pool; the log already carries the structure, keyed by track pair.
I flattened because a flat list was easier to classify, and in doing so I destroyed
the one distinction that decides whether a delay difference is in the media or in
the file's own track layout. A repair driven off those four would have been
repairing something that is not there — campaign 1's defect in a new costume.

**The fix strengthened the result it was meant to threaten.** With track pairs
separated, the contaminated folder splits *perfectly* — every misfiled entry at the
noise floor, every correctly-named one showing a step — where the flawed parse left
three unclassified. The misfiling answer never depended on the step classifier: it
rests on the fidelity floor, on three independent methods agreeing, and on frames I
looked at. That is the only reassuring thing here, and it is an argument for
resting a headline on evidence that does not share a code path with the thing most
likely to be wrong.

## 2026-09-03 — dev-1's finding: the recorded stage-1 delays are residuals, not absolute offsets

Carried as **their** measurement, with their hedge, because it changes how every
recorded delay in the corpus must be read.

`first_delay_test` calls `recreate_files_for_delay_adjuster` (`mergeVideo.py:333`),
which re-extracts the candidate's windows shifted by stage 1's own answer. Every
measurement after that point is a **residual against an already-shifted candidate**.
The shift is an arbitrary millisecond value and almost never a whole multiple of the
chromaprint hop (`4096/3/11025` = 123.840 ms), and a fractional-hop shift misaligns
the two fingerprint grids.

Their controlled measurement on one pair, varying only the shift: whole-hop shifts
hold fidelity at 0.967–0.975, while the pipeline's own fractional shift drops it to
0.918–0.936 — the exact band that file's log records.

**Two consequences.** A one-point step in the recorded residuals may be an artefact
of the re-extraction rather than a real change point; and the recorded fidelities
are depressed by the instrument, not only by the media.

**Their hedge, kept:** measured absolutely on **4 files** — two artefacts, one real,
one unreadable. Four files is not a rate. The consequence carried is only that the
recorded data **cannot distinguish** a one-point step from a shift artefact, so
one-point steps are ranked suspect rather than counted or discarded.

**What was done instead of re-deriving it:** ranked every step by size in points,
using each file's own recovered point size. Three independent corroborations that
the suspect bucket is right — dev-1's two measured artefacts and their unreadable
case all land in it, and so does a file I had separately confirmed
**frame-identical by eye**, which is exactly the predicted artefact.

## 2026-09-03 — correction: I treated the 125 ms correlation point as a constant, and it is not

**This supersedes the arithmetic, not the conclusion, of the baseline entry below.**
Entries here are append-only, so the earlier text stands and this says what is wrong
with it.

I wrote that the ten stage-1 delays on error id 237 were "an exact multiple of
125 ms — the chromaprint frame", and rested "the instrument is not at fault" on it.
The observation was right and **the reason was not mine to assert**: I took
"1 point = 125 ms" from `BRIEF_COMMON` and used it as a universal quantum.

**dev-1 measured otherwise.** `audioCorrelation.py:132` computes the ms-per-point
**per call** — `int(lengthFile/len(fingerprint_source)*1000)` — and the result
ranges **124 to 142 ms**, reaching exactly 125 only when the shortest audio runs
**18.7 to 30.4 minutes**. That is dev-1's measurement and their band; I carry it
with their hedge, have not re-derived it, and do not restate it more strongly than
they did.

**Why the conclusion survives anyway.** This pair's shorter file is 1438.04 s =
**23.97 min**, inside that band, so 125 ms is the expected point size *for this
pair*. The delays do sit on the quantum and the instrument was behaving. What
changes is the warrant: "this pair's point size is 125", not "125 is the quantum".

**What it costs elsewhere, and this is the part worth keeping.** `BRIEF_COMMON`
offers the 125 ms figure as the number that will save an hour, and says *"if a delay
is not a multiple of 125, question the reading before you question the media"*. On
dev-1's measurement that heuristic is only valid for pairs whose shorter audio falls
in an 11.7-minute window — and a 47-minute double episode or a 12-minute short would
have a different point size, so a perfectly good reading would be questioned for
failing a test that does not apply to it. **A finding for the owner, not an edit:**
the document is frozen and it is dev-1's measurement to report, not mine to
generalise. Recorded here because I acted on the wrong version of it and someone
else will.

## 2026-09-03 — the deploy loop's "wait ~10 min" becomes a checked precondition

The documented loop warns that a hook fired before CI publishes the image re-pulls
the previous image and redeploys the *old* commit — a success-looking deploy that
silently invalidates the test. The brief names it the single failure mode that will
waste the most of this campaign's time.

**The instruction was a duration, and a duration cannot be checked.** The published
image can be: it carries the commit it was built from, in the OCI label
`org.opencontainers.image.revision` and in its baked `VMSAM_GIT_COMMIT` — the very
value `/health` will later report. Both are readable before anything is fired.

So `VMSAM_HELP_AI/ci/deploy.py` waits until the published image **is** the target
and refuses to fire the hook otherwise. **Why the alternative lost:** sleeping ten
minutes is wrong in both directions — it wastes nine minutes on a cached build and
is still too short for a cold one, and in neither case does it *know*. The registry
answers the actual question.

Measured while confirming it: a documents-only push published in **~90 s**. That is
a cache artefact, not a new figure for CI — a real `src/` change rebuilds layers,
and a cold build remains unmeasured. Recorded so nobody quotes 90 s back.

**Where it lives:** beside the repository. `WRITE_ZONES.MD` §4's test — would this
still matter to someone running VMSAM in production? — says no for a deploy
harness, the same ruling as `check_write_zones.py`.

## 2026-09-03 — finding: `git_commit` has two unverifiable states, and the documents describe one

`tools.py:226` returns `os.environ.get("VMSAM_GIT_COMMIT", "").strip() or "unknown"`,
while `Dockerfile:137` declares `ARG VMSAM_GIT_COMMIT="IDK"`. So:

- an image built **without** `--build-arg VMSAM_GIT_COMMIT` carries the ARG default
  and `/health` reports **`IDK`** — fix belongs in CI;
- `unknown` is only reachable when the variable arrives **empty or absent** at
  runtime, which `Dockerfile:163` already does and the compose file overrides — fix
  belongs in compose.

Both the private deployment note and the campaign brief say `unknown` means the
build arg was not injected. Read against the code, that case produces `IDK`.

**Consequence: none for correctness of results.** Both states are *unverifiable*,
and the rule that matters — no result obtained in that state proves anything — is
unchanged. The distinction is worth carrying because the two have different causes
and different fixes, and a reader who sees `IDK` will otherwise search the compose
file for a variable that is not the problem.

Neither state has been observed on this instance. Both are coded from the source
and fired with literals, not from a sighting — an armed guard that has never fired
is indistinguishable from a working one.

**This is a finding for the owner, not an edit.** The documents concerned are the
owner's private notes and the brief; being right about them does not make them
mine.

## 2026-09-03 — finding: `fusion_enabled` is computed one hop away and dropped

`internal_api.get_internal_health` returns `fusion_enabled`, which is `False`
exactly when the worker refused to start because `VMSAM_TEST_OUTPUT_DIR` was empty
in test mode (`internal_api.py:52`). The public `GET /health` calls that endpoint
and keeps only `is_running` (`api.py:359-366`).

From outside the instance, therefore, **`is_running: false` with an idle queue means
either "the worker is healthy and has nothing to do" or "the worker never started"**
— and `GET /fusion` cannot separate them either, because
`get_internal_fusion_status` carries no `fusion_enabled` guard and reports an idle
queue in both cases.

This is the campaign's recurring shape: a reader with two outcomes describing a
world with three, and the missing outcome is *the instrument did not run*.

**Not blocking, and what was done instead.** The worker is running — proven
positively by a fusion job reaching `apply_test_outcome` and depositing an artefact.
The cheap probe when in doubt is `POST /fusion`, which answers 503
`Fusion is disabled: VMSAM_TEST_OUTPUT_DIR is not set in test mode` in the disabled
case. `src/gestionar_show/` is frozen in full (`WRITE_ZONES.MD` §1), so adding one
key to the public payload is an owner decision. Recorded, not routed around.

## 2026-09-03 — the first baseline run refused correctly, for a reason nobody had listed

Error id 237 was queued as the campaign's baseline, before any change. It refused in
~30 s at `mergeVideo.py:798` — the code path zone A is being built into — with ten
stage-1 delays spread over 421 s, flipping sign, at mean fidelity 0.567. Every delay
is an exact multiple of this pair's correlation point size, so the instrument was
behaving; there is simply no constant offset to find. **The refusal is correct.**
(The point size is 125 ms **here** and is not a constant — see the correction entry
below. My original note said "125 ms, the chromaprint frame" as though it were.)

The reason is not any of the five families in `CAMPAIGN.MD`. **The file is a
different television series from its folder's master** — verified by extracting
frames from both at the same timestamps and looking at them, not inferred from the
release name. Two different programmes have no true correlation peak, which is
exactly the signature observed.

And it is systematic: of the 49 error entries in that folder, **26 name themselves
as the other series in a contiguous 26-episode run** — 8.3 % of the whole 315-entry
corpus — with no folder registered anywhere for them. One file verified visually;
the other 25 attributed from the naming pattern. The other 37 folders are unchecked,
and how campaign 1 classified any of this is unknown to me and not asserted.

**Why this is a decision and not just a measurement.** The repair fires at the
moment of refusal, and these files refuse. `SPEC_ZONE_A.MD` as drafted does not
distinguish *two cuts of one episode that need reassembly* from *two different
episodes that must never be spliced*: both arrive as "stage 1 found no constant
offset". No repair exists yet and its authors may well plan a guard — the point is
that the family most needing one is absent from the table the campaign is working
from, and it stayed invisible until a real file was actually run. That is the case
for baselines before changes, made concretely.

Error id 237 is therefore offered as a **negative fixture**: a real corpus file,
reproducible in ~30 s, whose correct outcome is *refuse*. Any repair should be run
against it to show it can still say no. Details in the private notes beside the
repository.

## 2026-09-03 — `check_write_zones.py`: the rules get an instrument, and it lives outside the repository

`WRITE_ZONES.MD` §2 rule 2 says a zone is real only if it is checked, and lists
four shell snippets. Three of them are read by eye. Campaign 1 pushed a commit
with a failing test by gating on the wrong thing and reading a suite's output
instead of its exit status (`CAMPAIGNS/01-dev-AI/RESTART.MD` §8), so the checks
are now one command with one exit status.

**Where it lives: beside the repository, not in it** —
`/home/vmsam/src/VMSAM_HELP_AI/tools/check_write_zones.py`. §4's test is *would
this still matter to someone running VMSAM in production?* A harness that reads
`git diff` and enforces a campaign's write rules would not. Campaign 1 left 148 773
lines of exactly this kind of scaffolding under `src/`, nineteen times the program.
One copy outside, called by absolute path from every clone, also means the four
agents cannot drift onto four versions of the rule.

**Why the naive form of the zone check is wrong.** `git diff --cached -U0` read by
eye compares changed lines against the zones of the *new* file. A pure deletion has
no new-side line at all, so a `-` hunk removing a frozen line above a zone reads as
"nothing added outside a zone" and passes. The checker maps removals onto the *old*
blob's zones and additions onto the *new* blob's, separately. The self-test case
`outside zone: pure deletion below the zone` is that defect, pinned.

**Two more holes the eye version leaves open**, both now refused: moving an `END`
marker down to swallow a line into a zone (the diff lands inside the new zone and
looks legal), and creating a fresh marker pair — §2 says adding a zone is an owner
decision, so any change in marker count fails.

**It answers three things, not two.** Exit 2 means the checker could not run and is
*not* a pass. `RESTART.MD` §3.4 found the same shape five times in one morning: a
reader with two outcomes describing a world with three, and the missing outcome is
always *the instrument did not run*.

**Fired, both answers, before being believed.** `--self-test` builds throwaway
repositories and drives 27 cases; each refusal has a matching case that must be
accepted. On the real tree at `f0e7119`: clean; a one-line insert at
`mergeVideo.py:1` refused; the same insert at `mergeVideo.py:795`, inside zone A,
accepted. An armed guard that has never fired is indistinguishable from a working
one, and that includes this one.

**What it deliberately does not check**, so that a pass is never over-read: that a
new module is called by anything; that a file belongs in the repository at all
(§4 is a judgement, not a path rule); privacy; `config.ini` semantics.

## 2026-09-03 — finding: `keep_best_audio` is listed as a tagged zone and carries no tag

`WRITE_ZONES.MD` §2 lists `mergeVideo.py` / `keep_best_audio` among the existing
zones, and the paragraph above the table cites it as the reason the comment form of
the marker exists: *"`keep_best_audio` uses the comment form for exactly this
reason."* In the tree at `f0e7119` there is no marker of either form in that
function:

```sh
grep -n "AGENT modification" src/mergeVideo.py
# 663, 667, 674, 678, 682, 690, 789, 800 — all in adjust_delay_to_frame and
# remove_not_compatible_video. Nothing near keep_best_audio (1011).
```

The `fabricated` branch the entry refers to is present and correct
(`mergeVideo.py:1036-1039`), so what is missing is the permission, not the code.
By rule 1 — a path that is not open is closed — the function is frozen, and
`check_write_zones.py` enforces that reading.

**Consequence, and what was done instead.** Nothing is blocked today: the branch is
already written, and `SPEC_ZONE_A.MD` §4 says it activates by itself the day the
repair sets the key on the audio dict elsewhere. So the campaign proceeds without
the zone. It becomes blocking only if the `fabricated` comparison itself has to
change — a case we do not expect and will report if we meet it.
**This is a finding for the owner, not an edit** (`AGENT.MD` § Documents): the fix
is either two marker lines in `mergeVideo.py` or one line struck from the §2 table,
and both are owner decisions.

## 2026-09-03 — finding: the Rust sources are frozen by rule and editable by instruction

`src/lib.rs`, `src/main.rs` and `Cargo.toml` appear in neither `WRITE_ZONES.MD` §1
(frozen) nor §4 (open). Rule 1 settles it — *a path that is not listed as open is
closed* — and `check_write_zones.py` treats them as frozen.

`AGENT.MD` § Coding Directives points the other way: *"Rust (`audio_sync`) must
pass `cargo check`, `cargo clippy`, `cargo fmt` cleanly before submitting any
changes"*, and § Testing says `cargo build` / `cargo check` are "allowed and
recommended". A file nobody may change cannot have changes submitted, so the two
documents disagree.

**What was done instead:** the strict reading. Nothing in the first objectives needs
the Rust core — stage 2 is the instrument we measure *with*, not the one under
repair — so no work is blocked. Recorded rather than resolved, because
`AGENT.MD` § Documents is explicit that being right about a rule does not make it
ours to change.

## 2026-09-02 — master prepared for campaign 2

`gestionar_show` imported from `dev-AI` and corrected: the dual-uvicorn split, a
fixed internal port (`tools.internal_api_port = 42085`), and `fusion.py` rewritten
from 1334 to 381 lines — the queue and the merge kept, the campaign instrumentation
dropped.

Twelve defects found and fixed while wiring it, none of which the compiler could
see: `get_git_commit` raising `NameError` on first call, three `tools.*` attributes
referenced but never defined, `_env_flag` used for a path (it returns a boolean),
nine `tools.logs` writes leaking into thread contexts through cross-file
`Thread(target=…)` declarations, `.get(0)` on a list, a `Decimal`/`float` mix that
made the no-forced-best-video path drop every candidate *silently*, a lost-master
window in the production swap, a pool never shut down, and a production failure
that left no trace anywhere.

Two hazards were closed structurally rather than by a fix: the episode lock, because
the integration loop and the fusion worker are separate processes that can write the
same episode; and `tools.load_merge_runtime_from_env()`, because Python 3.14
defaults to `forkserver` and a forkserver child sees only module defaults — which
would have put the two processes' locks in different trees, silently, while
`software={}` failed every ffmpeg call.

Write zones were declared and the first tags placed. The rule that governs them —
*a unit test cannot tell you that nothing calls the unit* — comes from campaign 1
shipping five pieces of code that were reviewed, tested green, and never called.
