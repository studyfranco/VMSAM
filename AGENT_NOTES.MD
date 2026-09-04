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

## 2026-09-03 — pHash IS resolution-independent, and both my corpus controls for 010 were the wrong controls

Two results from `TASKS/010`, the second more useful than the first.

### The `CAMPAIGN.MD` open question, answered

*"nor how it behaves between sources of different resolution"* — measured, before
any cross-resolution result was trusted. One source at three heights, frames hashed
by `frame_compare`'s own front end (32x32 grey, 64-bit DCT pHash), 200 frames each:

| | mean Hamming /64 | max |
| :--- | ---: | ---: |
| 1080 vs 720 | **0.46** | 4 |
| 1080 vs 480 | **0.61** | 4 |
| control, self vs self | **0.00** | 0 |
| control, vs shuffled frames of the same programme | **28.69** | — |

**Resolution-independent to within 0.61 bits of 64 across a 2.25x scale change**,
against ~29 for unrelated content — a separation of about 47x, so a threshold
anywhere between 4 and 28 works and the choice is not delicate. Both controls
fired. **Limits:** 480p is the lowest tested; all three arms are one encode
re-scaled, so this measures the scale part and not codec differences on top.

### Both controls I chose from the corpus were wrong, and running them is what showed it

**id 108 as a positive control found nothing** — median distance 2.0, zero points
over threshold. Two reasons, both mine. `SEARCH=6` frames at 10 fps is ±0.6 s of
slack against a 503 ms island, so **the search absorbed exactly the structure it
was meant to detect** — a control that cannot fail, in my own harness, for the
second time today. And worse: **108 was never a control for modality blindness at
all.** Its island is a *timing* difference and it is audio-visible; my own locator
finds both transitions. I picked a control for the wrong blind spot, and a null
result would have been indistinguishable from a broken instrument.

**id 266 as a negative control was mislabelled.** It has four real change points,
so one constant offset cannot align it and high video distance is *correct*.

### The control that works had to be constructed, and that is not a weakness

A pair whose video differs while its audio does not is **precisely what the run is
hunting**, so no corpus example can be known in advance. Constructed: 120 s of real
video with **video 40-60 s replaced and the audio left untouched**, verified
**byte-identical** by `cmp` on decoded PCM — so the audio path cannot see it.

| | |
| :--- | :--- |
| audio offset measured | **-0.00 ms** |
| coverage | [0.0, 118.0], 20/20 readable |
| median distance | **0.0** |
| over threshold | **43, 49, 55 s — 3/3 inside the planted [40, 60]**, distance 26-31 |

Detected at both `SEARCH=6` and `SEARCH=2`, so it does not rest on the slack.

**This is the fixture rule pointing the other way.** All day the lesson has been
that a constructed fixture proves less than a real one. Here the reverse holds: for
a blind spot defined by *nobody having seen an instance*, the constructed case is
the only one whose answer is known before the run, and the corpus cases are the
ones that cannot serve as controls.

### On waits, and a document that named the wrong consequence

I wedged ten shells on `until ! pgrep -f <script>` — the pattern matches the shell
running the loop, so the predicate can never become false. `COMING_BACK.MD` §3
already warns that `pgrep -f` matches its own shell, and I had read it. It warns
against **misreading** the answer; the failure that costs hours is **blocking** on
it. A rule that names a defect by the wrong consequence inoculates the reader
against noticing the right one.

The general form, which covers `ci`'s six-hour wait on a file whose producer had
died and which my version would not have prevented: **a wait needs a timeout, and a
check that the thing it waits for can still happen.** Wait on a handle you own —
`kill -0 $PID` on a pid you captured — never on a name, a pattern, or a file
someone else produces; bound it; report the timeout as an outcome distinct from
success; and print progress, because a silent wait and a dead wait look identical.
## 2026-09-03 — "repaired" is not one thing, and the corpus says so where two exemplars did not

The splice was validated on ids 266 and 108, which land within ~1 ms of the
master. Over the first twelve repaired files of the corpus sweep the worst lag
runs **1.50, 5.88, 10.62, 17.50, 21.00, 24.25, 32.62, 33.00, 52.38, 61.25, 63.25,
65.00 ms** — median 27, top three at one and a half video frames. Every one is
inside the 100 ms tolerance, so every one counts as `repaired`.

**Decision: the tolerance is not moved.** It is the only thing standing between
`repaired` and `damaged`, and adjusting it after seeing the distribution destroys
the distinction the sweep exists to measure. Its value was derived before any of
this from measured spread — correct plans 0.5–2.8 ms, a missed change point
503 ms, an inverted sign 16 146 ms — and that derivation does not change because
the corpus is untidier than the exemplars.

*Why the exemplars misled:* they were screened for cleanliness before being
chosen. A sample selected for being easy to measure tells you what the method does
at its best, not what it does.

**Second decision: report master fill as a column beside the count.** Measured
over the same twelve: median **8.4 %** of the produced track is master content,
min 1.2 %, max **43.9 %**. Zero files where the master supplies more than half. A
file at 1.2 % and a file at 43.9 % are both `repaired` and they are not the same
result, and a count that hides that is a count nobody can act on.

*Why no threshold:* no measurement justifies one. Inventing a cutoff here is the
defect `SPEC_ZONE_A.MD` §3 records about `get_less_channel_number` fabricating
`"2"` and `"44100"` — a plausible number is indistinguishable from a measured one
downstream. The distribution is reported; the owner can set a cutoff against it,
and cannot un-see a number never reported.

## 2026-09-03 — the fast path failed nine times and only the log knew

`audio_sync`, the Rust stage-2 binary, returned `empty input for correlation` nine
times during the sweep. All nine on **one** file, id 49 — not nine files.
`audioCorrelation.py:262-280` catches it and falls through to the Python
implementation, so a measurement was produced every time and no repair was built
on nothing. `AGENT.MD` is explicit that the Rust rewrite buys memory, not
accuracy, so the fallback answer is the same answer more slowly.

**The finding is not the failure, it is its invisibility.** A bare
`except Exception` treats a genuine binary fault and a transient identically, and
the only record either leaves is one stderr line — on a stream both this agent and
the Lead had classified as instrument chatter when the sweep was set up. Nine
failures, invisible to every consumer.

*What was nearly concluded instead:* that a repaired track had been built where
stage 2 gave nothing. That would have been the most serious finding of the sweep
and it is false. It came from an attribution error of mine — an `awk` that named
the last result line printed, when results print *after* a file completes, so it
named the previous file. Checking the line numbers rather than trusting the
attribution is what corrected it.

*The real lead is the file, not the binary.* id 49's candidate runs 918.4 s
against a 1452.4 s master — **534 seconds, 37 % missing** — with nine E-AC-3
master tracks against one candidate track. It is the only file of the fourteen
shaped like that, and it is classified `different cuts`. Relayed to
`vmsam-forensic`, whose classification it bears on.

## 2026-09-03 — 006 is green on committed code, and the 26.6% it leaves behind is my defect

`vmsam-dev-2` ran error id 266 through `change_point_locator` on committed code —
no hand plan, no hand patch — and it repaired: 2 audio and 21 subtitle tracks, 0
declined, 0 failed, both tracks aligned to **worst 1.12 ms and 1.00 ms** over six
probes each at r 0.961-0.995, with `VMSAM_FABRICATED=chimeric` on the jpn eac3 in
the final mkv. The locator's log read *"2 segment(s) dropped as unusable, 3 kept"*.

**First outside evidence on my brackets, and it is favourable.** dev-2 profiled the
real pair independently — 8 s windows at 2 s steps, none of my numbers — and got
T1 ∈ (628, 632) s where I report **(629.0, 631.5)**. Its bracket contains mine,
from a different instrument in a different direction. **Fenced as it fenced it:**
one transition of several, one instrument, and its 2 s step makes its bracket
*coarser* than mine, so it says my centre is not wrong and says nothing about my
precision. Given the previous bracketing was 35× over-claimed, that is worth a lot
and is still not confirmation.

### The 380 s is mine, not the media's

dev-2 reports 26.6 % of the master timeline (380 s of 1428) filled from the master
because of the two dropped segments. One is genuine — segment 0 at offset -819 ms
is unusable by construction. **The other is my arithmetic.**

Segment 1 came out as master `[100000, 80000]` — inverted. That start is
`change_points[0]["gap_end_ms"]`; that end is `change_points[1]["bracket_low_ms"]`.
So **my brackets overlap and are out of order**, and cp0's gap swallows the whole
of plateau 1. Cause is the bound-only fallback in `_bracket_transition`:

```python
return region_start, region_end + PROBE_WINDOW_SECONDS, False
```

`region_end` is the *next* run's first probe start, and adding a full 60 s window
pushes the high bound past it — here, through the following plateau. **A bracket
honestly labelled `bracket_is_bound_only` can eat an entire segment**, and dev-2's
`continue` then correctly refuses to place a segment my own bracket made
impossible.

**Deliberately not fixed today.** dev-2 is mid-validation building an independent
target against this exact block. Moving the bracket arithmetic underneath it would
mean no disagreement could be attributed — to its target, to my brackets, or to my
having moved them. That is dev-2's own point-3 finding applied forward: a
comparison whose two sides move together proves nothing. Fix after its run, then
one rerun, so only one thing has moved.

### `vmsam-dev-2`'s transport finding, and it is the sharpest of the day

> **A field whitelist in a transport is a defect by construction.** It makes every
> future extension of the format inert.

It was not consuming `candidate_offset_ms_by_stream` at all: `parse_segments`
rebuilt each segment from three named keys and dropped the rest. **Emitter correct,
consumer correct, transport silently lossy** — and the symptom was *nothing
changing*, which is the one symptom nobody investigates. Its unit test passed
throughout because it handed the consumer a hand-built dict and never crossed the
transport.

That is the control-that-cannot-fail family in a place none of us had looked: not
the instrument, not the fixture, but **the wire between two correct things**.
Consequence taken: anything I add to the block from now on, I ask dev-2 to echo
back from *inside* its parser rather than assume it arrives.

The fix moved track 2 from **28.88 ms to 1.00 ms** — the per-stream split I found
in the input, closing in a produced file.

## 2026-09-03 — finding for the owner: a third speed relation, and the margin that does not carry

`docs/AUDIO_SPEED_POLICY.MD` is frozen. This **extends** it by measurement and
reaches the owner as a finding, never as an edit.

**The relation is `1001/1000` and the verdict is `asetrate`.** Measured over the
8 files `vmsam-forensic` classes as speed mismatch: **mean 1.001001, 95 %
[1.000990, 1.001012]**. `1001/1000` = 1.001000 lies essentially dead centre; PAL's
1.042709 is excluded by four orders of magnitude. `1001/1000` and `24/23.976` are
the **same** relation, not two candidates — 23.976 rounds 24000/1001.

Verdict by the document's own three-arm method, 1200 s window:

| arm | id 33 | id 31 |
| :--- | ---: | ---: |
| uncorrected (control) | 0.7411 | 0.7374 |
| **asetrate, correct direction** | **0.9588** | **0.9551** |
| asetrate, WRONG direction | 0.6569 | 0.6562 |
| rubberband | 0.9484 | 0.9453 |

The wrong-direction arm lands **0.084 below the uncorrected control**, so the sign
is demonstrated rather than asserted. That column exists because weakness 3 records
a sweep that used `× r` for `÷ r` and had rubberband win 33 of 33 with nothing
internal able to catch it.

**Three hedges, none optional.**
1. All eight files are **one series**. Eight files are not eight independent
   observations.
2. **Between-file scatter is 2.8× the pooled standard error**, so the
   scatter-based interval above is the honest one and the tighter pooled one must
   not be quoted.
3. The finding against the document is **narrow**. It measures rubberband losing
   by 0.278-0.313 with a ~0.109 handicap; here the gap is **0.0104 and 0.0098**,
   an order of magnitude smaller, because both arms press on the 0.94-0.98 ceiling
   the document itself records for real masters. **THE RULING STANDS**, and its
   justification is physical rather than statistical. What does **not** carry is
   the expected margin: a detector built to ~0.109 separation could miss an
   inverting NTSC case. Two files agreeing in direction and magnitude is evidence
   the margin is real; **it is not a rate.**

**A fourth hedge that turned out to be the finding.** I insisted 2.5 % was a
**floor, not a count**, because a PAL-rate file drifts ~9 s inside a 214 s window,
its correlation collapses, and the method that found these eight is structurally
unable to see one. forensic has since found **19 PAL files**, roughly twice the
NTSC population, exactly where the hedge said they would be. So the number above
is the **NTSC subfamily's** coefficient, not the corpus's.

**Objective 4 asks for audio AND subtitles both taken.** The coefficient applies to
both: a subtitle re-time is a multiplication of every cue timestamp by the same
ratio, and `docs/SUBTITLE_CODECS.MD` decides which formats can be re-timed at all —
an image format cannot, and saying so beats shipping a plausible file.

dev-2 is given **`1001/1000` as the exact rational**, not my point estimate: the
rational lies inside my interval and is *more precise than my measurement*, so
propagating measurement noise into a file would be strictly worse.

## 2026-09-03 — the §2 claim on id 266 is withdrawn, and the target shared the plan's blind spot

`vmsam-dev-2`'s verifier now takes **two probes per piece** and treats disagreement
*within* a piece as a third outcome. On error id 266 it refuses: *"piece 1 probes
disagree by 999.0 ms"*, at master 2.0 s and 609.0 s, both inside what the plan
called one plateau. **The plan was incomplete.** My full-file scan reached the same
conclusion from the other side — four change points where two were known, the two
extra in the head that neither instrument had scanned.

**So `SPEC_ZONE_A.MD` §2 is NOT met on id 266, and I am recording the withdrawal
rather than letting the earlier report stand.** What is still true: the repair
produced a merged output where the baseline produced only a `.log.error`, the
repaired track is present and tagged `VMSAM_FABRICATED=chimeric`, and the splice is
faithful to its plan. What is not true is that this validated anything.

**Why the check failed to catch it, and this is the important part.** §2 point 3 —
*the agent produced the expected file itself and compared it against the output* —
was performed correctly and still proved nothing, because **the hand-built target
was built from the same plan the splice used.** Target and instrument shared an
author, so both were wrong in the same direction and agreed perfectly.

That is exactly the question `vmsam-ci` and I arrived at hours earlier and did not
then apply to this result: *who produced the expected answer, and could they have
produced it wrongly in the same direction as the thing under test?* Here the answer
was yes and nobody asked it. It is the tenth instance of the pattern and the most
expensive, because unlike the others it produced a **claim of success** rather than
a wrong number.

**Consequence for `SPEC_ZONE_A` §2 as written.** Point 3 says the agent must build
the expected file and compare. It does not say the target must be derived
**independently of the plan under test**, and on this evidence it must, or the
comparison is a control that cannot fail. A finding for the owner, not an edit.

**What the two sides now agree on for id 266:** four change points, two of them in
the head, and the corrected plan is the one my rewritten locator produces. dev-2's
rerun against the committed module is the campaign's critical path and it is now
unblocked in both directions — its verifier can detect an incomplete plan, and my
scan no longer produces one.

## 2026-09-03 — corrections I owe after the promotion: PAL exists, and a figure I quoted is withdrawn

Three things I carried that are wrong or narrower than I stated. Recorded before
they propagate further.

**1. The speed family is not NTSC-only.** `vmsam-forensic` has corrected itself:
there are **19 PAL files**, roughly twice the NTSC population. My eight-file
measurement (`TASKS/008`) covers the **NTSC subfamily only**. The hedge I insisted
on carrying is what proved load-bearing — *2.5 % is a floor, not a count, because a
PAL-rate file collapses to weak correlation and is structurally invisible to the
method that found these* — and PAL files were hiding exactly there. **A stated limit
of my own instrument:** a 4.27 % relation destroys correlation, so
`change_point_locator` finds nothing on those 19 and will decline them. That
decline is correct, not a gap.

**2. The "250 ms" I quoted is withdrawn.** I cited *"forensic's eleven at 250 ms"*
as a third instance of same-language tracks a constant distance apart. Never
measured: it is the **difference of two readings on a 125 ms grid**, which inherits
the full grid step as uncertainty — two readings differ by "250 ms" only to ±125.
The phenomenon those eleven files show is real; the figure is not, and I repeated
it as though it were.

**3. My reconciliation of forensic's 750 against ci's 837 was wrong**, and its
standing note `reports/NOTE_contamination_vs_resolution.md` says why in general:
contamination error is bounded by the span between features, which you do not know;
it is arbitrary in sign, can exceed both blended values, and **does not shrink with
denser sampling** — a denser grid of straddling probes converges on a displaced
peak, stably and confidently. Modelling it as resolution error gives error bars
*too tight and the wrong shape*, which is exactly what I did.

**The rule, adopted verbatim into the module:** before differencing two
measurements, establish that neither window spans a feature boundary; if you
cannot, the difference is a **bound**, and say which.

**It cost me within the hour.** My own full-file scan bisected 60 s probes under a
majority model — assuming a straddling window returns the majority side's value —
and placed error id 108's transitions **19 s early and 24 s late while reporting a
625 ms bracket**, an over-claim of about 35× against two instruments that agreed
with each other. The plateau offsets were right throughout. Replaced with
clean-probe bracketing: a probe matching the before-plateau proves `T > p+w`, one
matching the after-plateau proves `T < p`, and anything between spans the boundary
and is **discarded rather than interpreted**.

## 2026-09-03 — declining to comply with a false instruction is what found the real defect

**The clearest argument this campaign has produced for pushing back rather than
complying**, and the outcome was not the one I expected.

I was instructed four times to rewrite 24 commits to remove a privacy leak. I had
already removed it — once, 22 commits earlier, by amend — and each time I answered
with the verification command rather than complying. The fourth instruction added
*"this is now the only thing blocking your commits from landing"* and, in the same
message, *"re-ran over the same range in all four clones: clean."*

**Those two sentences contradict each other**, and that is what I answered.

The audit: 927 lines of commit text across 24 commits — `session` 0, `http` 0, `://`
0. The only matches were 24 `Co-Authored-By` trailers, which are attribution, not
session links. The offending object survives only in my reflog: not an ancestor of
HEAD, on zero branches.

**Why I refused rather than complying to be safe.** Rewriting 24 commits is
irreversible and would have discarded the exact history — three of my own retractions
among it — that makes the record auditable. *Complying with a harmless-looking
instruction is not the cautious option when the instruction destroys evidence.*

### And the problem was real — in the other three clones

The instructing agent checked, retracted in full, and then measured across every
clone:

| clone | commits carrying a session link |
| :--- | ---: |
| mine | **0 of 24** |
| forensic | 8 of 8 |
| dev-1 | 22 of 22 |
| dev-2 | 15 of 15 |

**45 commits across three agents**, verified independently by me afterwards. Their
words: *"I would not have found it if you had complied instead of pushing back."*

> A false instruction is often a **true observation attached to the wrong target**.
> Complying would have destroyed my history *and* left the actual leak in place.

They had read the pre-amend message once and repeated that reading for three ticks
without re-checking — while quoting a clean checker run over the same range. The same
failure the campaign documents warn about, committed by the agent enforcing them.

## 2026-09-03 — the checker's verdicts moved once, and only in the loosening direction

Before certifying any build I asked whether the corrected `check_write_zones.py`
changes the verdict on commits **already in the clones** — *"clean now"* and *"clean
before"* being different statements.

Measured per commit, both classifiers over all four clones:

| clone | verdicts changed |
| :--- | :--- |
| mine | 0 of 24 |
| forensic | 0 of 8 |
| dev-1 | 0 of 22 |
| dev-2 | **1 of 15** — old FAIL, new PASS |

**Nothing that previously passed now fails.** The single mover is the commit carrying
the repair's self-verification, editing two runtime modules after their creating
commits — which is exactly the harm the freeze was diagnosed as causing, in the files
predicted.

> Recorded as: **verdicts unchanged except one strictly-loosening move.** That is the
> form a deploy path can certify against; "the checker is fixed" is not.

## 2026-09-03 — a category error between two quantities sharing a unit has no smell at all

Another agent's formulation, and the sharpest of the day's "silent failure" family.

Their own brief warns that campaign 1 kept three stores on three key spaces, and a
lookup in the wrong one returned *"a clean, confident, entirely wrong absence that
looks exactly like a finding."* They quoted that rule at themselves when checking file
paths that morning and printed one key from each side before believing anything. Then
they compared **two magnitudes of different kinds** — a single-track offset against a
bound on an intra-track gap — and did not notice.

> **Both were in milliseconds, both were about the same file. The units matched; the
> quantities did not.** A key-space error eventually looks odd because the shapes
> differ. **A category error inside one unit looks like arithmetic.**

The list, as it now stands, of ways a measurement fails without announcing itself:

| failure | why it is missed |
| :--- | :--- |
| an instrument that cannot fail loudly | returns a plausible number rather than an error |
| a quantised difference | reported exactly, so it looks measured |
| a statistic over a biased subset | tight, so it looks precise |
| **a comparison between two quantities sharing a unit** | **nothing about it resembles a type mismatch** |

### And the corollary on wrong-track errors

**Five wrong-track incidents across four agents in one day. Four were found by a
second party; one was caught in flight** — and that one was caught by a **structural**
tell, not a numerical one: the log's 2×2 pair grid implied the comparison language, and
a German reference contradicted it immediately.

> **A wrong track returns a plausible number.** The only cheap check is a structural
> one that does not depend on the number looking wrong.

That is why the other four needed someone else: nothing in the value itself was
available to their author as evidence.

### One thing that went the other way

My cross-check of the quantised point-counts against direct bisection — 501.0 vs 500,
840 vs 875, 837 vs 750 — was received as more than bookkeeping, and correctly. All
three sit within ±125 of the direct value **and the residuals spread across the
interval rather than clustering at one end**. That is what two resolutions of the same
quantity look like, and not what a systematic offset looks like.

> The rule says the quantised values are **bounds**. The check says they are **honest
> bounds**. Different statements, and the second is the one that matters to anyone
> downstream who only has the log.

## 2026-09-03 — a difference of two quantised readings is a bound, not a measurement — and check the two figures measure the same thing

Two findings that arrived together, the first another agent's and the second a check
on their conclusion.

### A quantised difference inherits the whole quantum

A figure the team had quoted all day as *"the master's two same-language tracks are
250 ms apart"* was never measured. It is the difference of two readings each snapped
to a **125 ms grid** (−125 and +125), so from the log alone the true value is
**250 ± 125 — anywhere in [125, 375].**

Measured directly and differentially, it is **126–138 ms**: still over the 127 ms
reconciliation limit, but **by about 10 ms rather than by 123**. *"These files sit
just over the line, not far over it"* is a materially different finding.

> **Any delay difference read from this log at the one-quantum scale is a bound.**

Same class as *"a tight cluster is consistency, not completeness"* — a number that
looked exact because it was **reported** exactly.

**Checked against my own work:** every step size I quoted for the early-step family
came from direct FFT bisection, not from differencing log values, so none is
affected. And the quantised point-counts quoted alongside them behave as the rule
requires — 501.0 against 500, 840 against 875, 837 against 750, **each within ±125 of
the direct measurement.** That is a three-file consistency check, and the first
positive evidence that the quantised and direct readings describe the same thing.

### The check that mattered more: are these figures even the same quantity?

The same agent presented **three** numbers as competing estimates — 250 (log),
126–138 (theirs), 84.5 (a third agent's) — and declined to reconcile them by
averaging, which was right. But two of the three **are not measurements of the same
thing**:

| quantity | values |
| :--- | :--- |
| **difference between the master's two tracks** | 137.8, 126.0 |
| **single-track candidate-vs-master offset** | 124.0, 70.4, 261.8, 196.4 … **and 84.5** |

The third agent's 84.5 is a *candidate-versus-master* offset, and it falls **inside**
the first agent's own single-track span of 70.4–124.0. **No disagreement existed.**
Comparing it against a bound on the intra-master gap is a category error.

> Before reconciling — or refusing to reconcile — two numbers, establish that they
> measure the same quantity. A disagreement between different quantities is not
> evidence about either.

Which value you get depends on **which candidate track against which master track** —
the same 2×2 grid that has now produced five wrong-track incidents across four agents
in one day.

## 2026-09-03 — a stated gap is a request someone else can fill: the other half of hedging

**The most transferable thing to come out of today's cross-agent work**, and it is
not about being cautious.

Hedging discipline is usually framed as *not overstating*. This is its other half:
**an unresolved question, stated as unresolved, is actionable.** A guess gives the
reader something to agree with. Silence gives them nothing to check. Only *"I have
not determined which of these two, and I am not claiming to"* tells them exactly
which half is missing — and if they hold that half, they can close it.

**Demonstrated in a single exchange.** Another agent read the code and found **two**
distinct paths to one refusal, said explicitly they had not determined which applied,
and named whose ground it was to settle. I had the logs and had stopped reading the
code; they had the code and no logs. **Neither of us could have closed it alone**, and
the only reason it closed is that they said out loud which half they were missing —
which made me re-open a question I believed my own analysis had already settled.

It cost me a published conclusion, correctly.

> **State the gap, not just the hedge.** "I could not measure X" is a finding.
> "I measured A but not B, and B would change the answer" is a request.

### And a second one from the same exchange: fixing the instrument does not fix the habit

Earlier the same day another agent caught my parser **pooling** evidence across track
pairs. I recorded the lesson and rewrote the parser to segment by pair. Then I
analysed **one pair** and said "the file".

> **The defect moved from the code into the reading of its output, which is not a
> place a code fix reaches.** A tool that makes the right distinction available does
> not make you use it.

Both of that day's corrections between us were symmetrical and each was one check
away: they generalised from *"stage 1 is blind here"* to *"the pipeline is blind
here"* without checking the later stage; I generalised from *"this pair alternates"*
to *"this file is refused for that reason"* without checking the other three pairs.

## 2026-09-03 — CORRECTION: the quantisation noise is real but it does NOT cause the refusal, and I made the same mistake twice

**Supersedes the causal half of the entry below. The discriminator survives; the
conclusion drawn from it does not.**

I claimed the alternating one-point rounding *"is itself what refuses the file"* —
ten self-inflicted refusals. **Wrong.** Another agent noted there are two code paths
to that raise and said it needed the logs to settle, which is my ground. I looked,
and the answer is the other path.

Those files have **four** track pairs, not one:

```
0-0: [0,-125,0,-125,-125,-125,-125,-125,-125,-125]   3 transitions   fid 0.72-0.84
0-1: [-125] x10                                       constant        fid 0.92-0.93
1-0: [+125] x10                                       constant        fid 0.72-0.85
1-1: [+125] x10                                       constant        fid 0.92-0.93
```

**Three of the four are perfectly constant.** One file carries two same-language
tracks **250 ms apart**, so `delay_detected` = {−125, +125}, a gap of 250 which
exceeds the 127 reconciliation limit, and *that* fires the raise. The alternation
sits in the single lowest-fidelity pair and refuses nothing.

**So the other agent's original account was right and mine was wrong.** The two
phenomena coexist on the same file and fire at different lines, exactly as a third
agent predicted without being able to adjudicate.

### The part worth keeping is how I got it wrong

**I read one track pair and generalised to the file — the same defect I was corrected
on this morning**, when my parser pooled triples *across* pairs and turned a master's
own track layout into a phantom step. I recorded the lesson, wrote a parser that
segments by pair, and then **analysed only the first varying pair anyway.**

> Fixing the *instrument* did not fix the *habit*. The parser separated the pairs
> correctly; I still looked at one of them and said "the file".

The general form, and it is the third time today in a different costume: **a
conclusion about a file drawn from a subset of its evidence is a conclusion about the
subset.** Tight windows on one side of a step; one stream of several; one track pair
of four.

**What survives:** the contiguity discriminator (one transition = step, three or more
= sub-quantum rounding) and the 8-file identification, both confirmed against an
independent measurement. **What falls:** that the rounding causes the refusal, and
the phrase "self-inflicted refusal", which I had already put in front of the Lead.

## 2026-09-03 — most "one-point steps" are not steps: a sub-quantum offset causes its own refusal

Twenty files carried a stage-1 record whose delay takes two values one point apart. I
had them flagged as suspect with no account; another agent guessed a re-extraction
artefact. **Both were wrong, and a third agent's independent measurement supplied the
answer before I looked.**

They measured the true offset on ten of them at **+84.5 ms**. The point size is
125 ms, so that is **0.68 of a point — near a rounding boundary.**

The records are then unmistakable:

```
id 6   [0, -125, 0, -125, -125, -125, -125, -125, -125, -125]
id 7   [-125, 0, -125, 0, 0, -125, -125, 0, 0, -125]      <- alternates six times
id 19  [0, 0, 0, 125, 125, 0, 125, 125, 0, 0]
```

**A step has ONE transition. These alternate.** Each window rounds a sub-quantum
offset up or down on its own noise.

> **The discriminator is CONTIGUITY, not the number of distinct values.** Count
> transitions. One transition is a step; three or more is quantisation noise.

Corpus-wide, applying it to every varying stage-1 record:

| | n |
| :--- | ---: |
| multi-point — real structure | 277 |
| one-point, 1–2 transitions — ambiguous | 12 |
| **one-point, alternating — quantisation noise** | **8** |

The 8 are precisely the files the other agent had measured. Two routes, same set.

### The part that matters more than retiring the bucket

The inconsistent rounding **is itself what refuses the file.** Stage 1 requires its
windows to agree; a sub-quantum offset makes them disagree; the pair is dropped
before stage 2 ever runs. **Verified: none of those ten has a method-2 record in its
log at all.**

**So a self-inflicted refusal.** And it corrects a claim made from the same
measurement — that 84.5 ms is *"structurally invisible to the pipeline and would ship
as-is"*. `mergeVideo.py:243` is
`calculated_delay = delay_first_method + delay_second_method*1000`, and stage 2 is an
unquantised FFT: 84.5 ms is exactly what it exists to measure, and it is **added**.

> The offset is invisible **to stage 1**, not to the pipeline. It never reaches stage 2
> because stage 1's rounding disagreement refuses it first. Those are different
> statements and only the first is supported.

## 2026-09-03 — a linear fit cannot tell a drift from a step, and three of us produced a spurious rate on one file within an hour

**Three independent agents, three instruments, same file, same hour, same wrong
answer shape.**

| who | what their fit reported | what it actually was |
| :--- | :--- | :--- |
| another agent | *"POST drift −92.77 ms over 1000 s = 9.28 % of playback"* | an 89 ms **step** at 1170 s |
| me | +34.6 ms per 1000 s, looked like support for a drift hypothesis | the same step |
| a third agent | 45 ms of "drift" across the episode | almost certainly the same step |

> **A rate fitted across a range containing a step becomes a spurious slope. The fit
> cannot distinguish them; only the RESIDUALS can.**

What saved me was noticing the **residual spread (23.9 ms) was the same size as the
total effect (34.6 ms)**. Refitting without the one probe carrying it gave −7.45 ms
per 1000 s, spread 6.4 ms — flat.

**And it makes an existing measurement trustworthy for a stated reason rather than
by luck.** The speed coefficients in the campaign's speed work come from exactly this
kind of fit over ten windows. They survive because their author checked **residual
scatter against the quantisation floor**: 38.3 and 41.0 ms measured against
`124/√12 = 35.8 ms` predicted by pure quantisation. **Residuals AT the floor mean
nothing structural is hiding under the slope.** Had a multi-step file been in that
set, its residuals would have blown well past it. The check was load-bearing, not
decoration — and nobody could have said so this morning.

## 2026-09-03 — the geometry's blind spot has TWO conditions, not one

The position condition was recorded earlier: a step is invisible unless
`T > begin + L/2`, so the head of every episode is unsampled.

**There is a second condition, in MAGNITUDE, that neither of us had written down.**
A step must also exceed roughly **half a quantum** to change any window's reading at
all. The transition found at 1170 s is 89 ms — **0.7 of a point** — so every 194 s
window reads the same plateau on both sides of it. It fails the magnitude condition,
not the position one.

> **A step is visible only if it is both late enough AND large enough.** The head was
> never the special thing; it was just the part noticed first, because the windows
> start at 120 s.

Together the two conditions mean a family count from the stage-1 record has **two
independent unbounded error terms**, and neither is estimable from the record.

## 2026-09-03 — my own run duration is not a duration: the box was 2.8x oversubscribed

The first container job to reach stage 2 has run **over 70 minutes**, against under a
minute for every job that refused inside stage 1. I was about to record that as the
stage-2 cost for the run service.

Measured before doing so: **load average 46.88 on 17 cores**, with four agents
running concurrent `ffmpeg` decodes and FFT correlations — **including me**, doing
heavy bisection work on the same box throughout the run.

> **A duration measured under 2.8x CPU oversubscription is not a duration**, and I
> caused part of the contention I would have been measuring.

So the run-service table still says *stage-2 cost: unmeasured*. What is recorded
instead is the **only** safe reading: stage 2 is the expensive stage, by a factor of
at least tens, and a clean figure needs a quiet box. Quoting 70 minutes would have
put a number into a document that the next agent would have planned around.

## 2026-09-03 — "single step near the head" was an artefact of everyone probing near the head

**The three files this campaign adopted as its change-point acceptance set are not
what any of us described.** Profiling one of them end to end instead of near its
known step:

```
[  20,  91]   +672 ms    corr 0.91-0.97   <- previously unreported by anyone
  ~91 s        TRANSITION   step ~1499 ms
[  91, 184]  +2171 ms    corr 0.88-0.93
 ~184 s        TRANSITION   step  ~837 ms  <- the one four instruments measured
[ 184,1170]  +3008 ms    corr 0.83-0.94   flat
~1170 s        TRANSITION   step   ~89 ms
[1185,1290]  +3097 ms    corr 0.88-0.94   three probes within 0.5 ms
```

**At least four change points**, where all four instruments had reported one. A
second file has two with the offset **returning**. The third has been profiled to
760 s and no further.

**Two things follow, and the second is the larger.**

**1. It resolves a 76 ms disagreement, with nobody wrong.** Another agent's
pre-step probe spanned the ~91 s transition, so their value was a blend of two
plateaus — the same contamination they had already correctly diagnosed in their
*other* probe. Both instruments were sound; both of their probes happened to straddle
transitions nobody had detected.

**2. The head blind spot is a special case of something wider.** That third
transition is at 1170 s with a step of **89 ms** — nowhere near the head, and equally
invisible, because every stage-1 window is 194 s long and reads the same plateau on
both sides of an 89 ms step.

> **The geometry cannot see a step smaller than its own resolution anywhere in the
> file, and cannot see any step in the head at all.**

Every family count built on the stage-1 record inherits both, and **neither can be
bounded from the record** — the information is absent, not merely hard to read.

**And a refuted hypothesis of my own, worth recording for how it nearly passed.** I
proposed that a 45 ms discrepancy was drift. Eleven probes gave a first fit of
+34.6 ms per 1000 s, which looked like support — but the **residual spread was
23.9 ms against 34.6 ms of total effect**. Refitting without the one probe carrying it
gave −7.45 ms per 1000 s, spread 6.4 ms. Flat. The 45 ms was the 1170 s step.

*What stopped me reporting the first fit was that the scatter was the same size as
the effect* — the "plausible number rather than an error" pattern, appearing in a
regression rather than a parser.

**The method point, which is the previous entry turned on all of us.** Every one of
these transitions was found by someone widening their search **after another agent's
result pointed at a region**. Not one was found by an instrument choosing its own
coverage. So a "head scan" is the wrong fix: the head was only ever special in being
the part the windows could not see. **The fix is a full-file scan with no privileged
regions.**

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
## 2026-09-03 — a linear fit cannot tell a drift from a step, and three of us produced a spurious rate on one file within an hour

**My id 13 reconciliation is withdrawn.** A chromaprint reading is
`offset_points × sizePoint` — an exact integer number of milliseconds by
construction. `vmsam-forensic`'s values are 2236.8, 2986.9, 2992.6: one decimal
place, and none is near an integer at 125, 129 or 124 ms. **Its instrument is not
quantised at all**, so my "18 points and 24 points" arithmetic explains *my*
chromaprint reading and cannot explain *theirs*. Two mechanisms landing on 750 was
a coincidence I presented as an explanation. `vmsam-ci` was right to be suspicious.

That is the **second** time in one day I offered an account that fits my own data
and assumed it fits someone else's — the RMS story for family B was the first.

### What the number actually was

ci profiled the whole of id 13 and found **at least four change points**:

```
[ 20,  91]   +672 ms    corr 0.91-0.97   <- a plateau nobody had reported
~91 s        step ~1499 ms
[ 91, 184]  +2171 ms    corr 0.88-0.93
~184 s       step ~837 ms                <- the one all of us had measured
[184,1170]  +3008 ms    corr 0.83-0.94   flat
~1170 s      step ~89 ms
[1185,1290] +3097 ms    corr 0.88-0.94
```

Forensic's PRE probe is [66, 126], which **straddles the ~91 s transition**, so its
2236.8 blends +672 and +2171. Its own straddle diagnosis was right and it applied
it to one probe when it applied to two. **Nobody's instrument was wrong.**

### Independently confirmed here, by probes placed before I read ci's model

| probe | measured | ci's plateau |
| :--- | ---: | :--- |
| [20, 80] | **-671.84** | +672 — exact |
| [60, 120] | -687.07 | straddles ~91, blends |
| [100, 160] | **-2179.09** | +2171, within 8 ms |
| [200…1000] | -3000.45 to -3017.14 | flat, 16.7 ms across 800 s |
| [1200, 1260] | **-3096.85** | +3097 — exact, step 89.87 vs ~89 |

### The trap, and my own output contains it

My script printed *"POST drift −92.77 ms over 1000 s = 9.28 % of playback"*. That is
nonsense produced in a specific way: **I fitted a rate across a range containing a
step**, and the 89 ms step at ~1170 became a spurious slope. ci's first fit did the
same (+34.6 ms/1000 s, residual spread 23.9 against 34.6 of "drift", dragged by one
probe), and forensic's 45 ms POST drift is very likely the third instance.

**A linear fit cannot distinguish a drift from a step unless you look at the
residuals.** Three agents produced a spurious rate on one file within an hour.

**This lands on `TASKS/008` and 008 survives, for a stated reason rather than by
luck.** Those coefficients come from exactly this kind of fit over ten windows.
What protects them is the residual check: 38.3 and 41.0 ms measured against
`124/√12` = 35.8 ms predicted by pure quantisation. **Residuals at the floor mean
nothing structural is hiding under the slope.** Had id 13 been in that set its
residuals would have blown far past it. The check was load-bearing, not decoration.

### Two consequences for the locator fix

1. **Not a head scan — a full-file scan with no privileged regions.** id 13's
   third transition is at 1170 s and was equally invisible. The head was never
   special; it was only the part noticed first.
2. **My blind-spot arithmetic was right and too narrow.** `T > begin + L/2` is the
   condition on **position**. There is a second condition on **magnitude** I had not
   written: a step must also exceed roughly half a quantum to change any window's
   reading. id 13's 89 ms step is 0.7 of a point and fails the *second* condition,
   not the first.

### The acceptance set is downgraded, deliberately

We called these three files "early-step" or "trimmed edge". One has at least four
transitions spread across the whole file; the other two are profiled only to 260 s
and 760 s. **They are three files with *some* transitions confirmed by several
instruments and an unknown number unconfirmed** — still far better than a
constructed fixture, but not the clean ground truth I called it, and not usable as
an acceptance set until each is profiled end to end.

### A ninth pattern instance, and it is `vmsam-ci`'s

> **An independent instrument aimed by a dependent one is not independent.**

ci built a separate instrument to check the pipeline, then pointed it only from
120 s onward because `begin_in_second` is 120 — using the pipeline's geometry as its
map of where to look. This is a *different* failure from the other eight: the
instrument was sound and the construction was sound, and the **aiming** inherited
the blind spot. It applies to me directly — every scan I have run outside the ten
windows was aimed where another agent's result told me to look.

## 2026-09-03 — two instruments, one number: forensic's 750 and ci's 837 were never in conflict

`vmsam-ci` bisected error id 13 and got a step of **837 ms**; the pipeline's own
stage-2 log gives **833.5 ms** by a wholly separate route; `vmsam-forensic`'s probe
grid gives **750 ms**. ci was about to report the 87 ms gap to forensic as a
discrepancy. **It is not one.**

Measured here on one pair of probes with one decode, using **160 s windows so the
quantum is 125 ms** — the pipeline's own, avoiding the short-window bias ci had
just caught in me:

| | FFT (unquantised) | chromaprint |
| :--- | ---: | :--- |
| PRE [20, 180] | **-2171.84 ms** | 0.7635 → **-18 pts** = -2250 ms |
| POST [200, 360] | **-3004.08 ms** | 0.9558 → **-24 pts** = -3000 ms |
| **step** | **832.24 ms** | **6 points = 750 ms** |

**Both answers, from one measurement.** 832.24 agrees with ci's 837 and the log's
833.5 to within 5 ms; 750 agrees with forensic's 750 exactly.

The whole gap is **one plateau misquantised by 0.63 of a point**. True PRE is
17.37 points and chromaprint read 18 — 78 ms high, which is *more* than half a
quantum and so a genuine misread rather than rounding. It compresses the step from
6.66 points to 6. And the misquantised plateau is the low-confidence one: 0.7635
against 0.9558.

**So forensic's instrument is doing what it does and its number is right for a
quantised measurement.** The finding is not a defect in anyone's grid; it is that
**a step must be reported as points plus the quantum, never as a bare millisecond
figure** — the same rule that governs exchanging offsets with dev-2, and the same
one that caught my own short-window bias an hour earlier. Three separate occasions,
one rule.

### And id 108 has TWO change points, with the offset returning

`vmsam-dev-2` profiled its first 260 s:

```
[  0,  52)  -950.0     r 0.37-0.65
[ 52,  58)  TRANSITION 1        <- previously unreported
[ 58, 146)  -1453.9    r 0.79-0.94
[146, 148)  TRANSITION 2        <- ci's bisection and my bracket, reproduced exactly
[148, 260)  -952.9     r 0.42-0.59
```

Its transition-2 bracket reproduces ci's exactly from a third instrument, which is
what makes transition 1 credible. **Carry dev-2's hedges:** one instrument, the
outer regions correlate weakly (0.32-0.65) so -950 rests on consistency across
twelve windows rather than any strong reading, brackets limited by a 2 s step
rather than bisected, and nothing looked at past 260 s.

**This corrects two things I said.**

1. I told dev-2 and the Lead that a container delay on 108 *"would be wrong by
   500 ms for the first 146 seconds"*. Measured, -950 is **correct** for [0, 55)
   and after 148 s, and wrong only inside **[55, 147)**. dev-2's reading is right
   that this is *worse* rather than better: a 0.5 s desync that appears a minute in
   and resolves itself ninety seconds later is the kind of fault a listener
   dismisses rather than reports.
2. **The blind spot is worse than I stated.** I had it that both change points sit
   in the head and are therefore invisible to the ten-window geometry. The sharper
   form is dev-2's: **because the offset returns, any window spanning both reads
   constant**, so they are invisible to each other's absence too. A head scan
   looking for *a* step would find the pair and report nothing — a constraint on
   the fix, not merely a reason for it.

**And the quantisation cost is now concrete.** My chromaprint points give -1500 and
-1000 where the true offsets are -1453.9 and -952.9: **46 and 47 ms out, each more
than a third of a point and each larger than a video frame at 23.976 fps.** A plan
built from quantised plateaus would be a frame out on both segments before any
splice error. That makes the sub-quantum refinement part of the head-scan work
rather than a later improvement.

## 2026-09-03 — RETRACTION, and the locator's head blind spot: the first ~227 s are structurally invisible

**I retract the entry below.** Its headline — *stage 2 is wrong where stage 1 is
right* — is **backwards on id 108**, and unverified on id 12. `vmsam-ci` found it
by bisection and `vmsam-forensic` reached it independently by another instrument.

**What is actually there.** id 108's window 0 spans [120, 334] and contains a
**real 500 ms step at ~146 s** — 26 s of the window pre-step, 188 s post-step.
Stage 2's window-0 value of +1453.9 ms is the **correct** offset for the content at
the start of that window. Chromaprint returned -1000 on all ten windows and **did
not detect the step at all**.

**Neither instrument is broken.** A single number for a window that spans a step is
ill-defined, and the two resolved the ambiguity differently: chromaprint by
majority (188 s against 26 s), the FFT by the short strong segment, because the
pre-step 26 s correlate at 0.8 while the post-step 188 s correlate at 0.45.

**id 12 now measured, and it is the same thing.** Scanning its head region with
both window lengths:

| | last OLD | first NEW | bracket |
| :--- | ---: | ---: | :--- |
| W = 30 s (q 135) | p = 142.5 | p = 145.0 | **T ∈ (157.5, 160.0)** |
| W = 15 s (q 150) | p = 150.0 | p = 152.5 | **T ∈ (157.5, 160.0)** |

Identical brackets. The step is **5 → 12 points = 7 points**, which is 875 ms at
that pair's pipeline quantum of 125 ms — against forensic's 861 ms and the direct
FFT difference of 850 ms (-567 against -1417). So **id 12 holds a real early step
too, and my RMS explanation for it was wrong twice over**: it was never about
energy, and the outlier was never a bad measurement.

All three of ci's family-B files (12, 13, 108) are therefore early-step cases, and
**every one of them sits inside the head blind spot**.

**RESOLVED by `vmsam-ci`: explanation (b), short-window bias.** It probed straight
across the 220-230 boundary I named and well past it — ten probes from 160 s to
760 s all read **+1412 to +1428 ms**, a 16 ms spread across ten minutes at
correlations 0.86-0.97. **There is no second step.** So my short-window absolute
read of 12 points was ~83 ms high against a true 11.3 points, and the ten-window
scan's 11 was right.

The bias is in the **absolute value only**. The location was identical at both
window lengths and the step was 7 points at both — and the step in points matches
ci's 840 ms (6.7 points) to within a rounding. So the rule I had already given
dev-2 — *exchange point counts and locations, never absolute milliseconds across
window lengths* — protected me from my own defect before I knew I had it. Standing
constraint for the module: **do not quote an absolute offset from a window of
30 s or less.**

ci's full 8 s bisection also closes id 12's ground truth: change point **158-160 s**
against my bracket of (157.5, 160.0), step **840 ms** against my 875, forensic's
861 and the direct FFT's 850 — four figures within 35 ms. Its boundary-spanning
probe collapses to **0.267 correlation** between neighbours at 0.89-0.92, the same
signature id 108 showed, which is independent evidence the transition is sharp
rather than gradual.

**One thing I could not resolve and flagged rather than choosing.** In the
post-step region [152, 220] the short windows read **12 points**, while the
ten-window scan reads **11 points** (-1375 at q=125) over [227, 1066] and the FFT
reads -1417 there. One to two points apart, which is more than quantisation
explains. Either a further change point sits between 220 and 227, or short-window
point counts are biased in a way I have not characterised. **Unresolved.** It does
not touch the location, which two window lengths agree on exactly, nor the step
size in points, which is 7 at both.

**The remedy ci and I had converged on would have shipped a defect.** Dropping the
outlier window and merging on the remaining nine gives a uniform offset that is
correct after the step and **wrong by 500 ms before it** — a plausible file,
silently wrong at the head, with every per-plateau check passing.

### The defect this exposed in `change_point_locator.py`, and it is the worst of the four

On id 108 my ten-window scan reads **-1000 on all ten**, so the locator returns
`kind: "constant"` with no change point. **Confidently wrong**, where the other
three defects merely refuse.

Pointed at the region, my own method finds it — **fourth instrument, agreeing**:

| | last OLD | first NEW | bracket |
| :--- | ---: | ---: | :--- |
| W = 30 s (q 135) | p = 130.0 | p = 132.5 | **T ∈ (145.0, 147.5)** |
| W = 15 s (q 150) | p = 137.5 | p = 140.0 | **T ∈ (145.0, 147.5)** |

Both lengths give the same bracket; ci's FFT bisection says 146-148 s, forensic's
probe grid ~567 ms, the pipeline's own stage-2 log 501 ms. Step **12 → 8 points =
4 points**, which is exactly 500 ms at the pipeline's 125 ms quantum — and 540 ms
at W=30, 600 ms at W=15, the point count portable and the millisecond not.

**The blind spot is exactly computable, not incidental.** A window reads the old
plateau while more than half of it is pre-change, so a step is visible only if some
window is majority-pre while the next is majority-post. **Window 0 is the first, so
nothing can be majority-pre before it.** A step is invisible unless
`T > begin + L/2`:

| shortest audio | begin | window L | first visible T |
| ---: | ---: | ---: | ---: |
| 1195.7 s | 120 | 172 | **206.0 s** |
| 1420.1 s | 120 | 214 | **227.0 s** |
| 2545.5 s | 120 | 418 | **329.0 s** |

So on a 24-minute episode the **first ~227 seconds** are invisible: the 120 s before
any window starts, plus the first half of window 0. `CAMPAIGN.MD` puts **trimmed
edges at 18 %** of the corpus, and a trimmed head *is* a change point near the
start — the blind spot sits squarely on the second-largest failure family.

**Fix: scan `[0, begin + L/2]` explicitly**, with the same two-window-length
agreement test. Same code path, pointed at a region the pipeline's geometry never
covers. Fourth on the list of fixes the module freeze blocks.

**Told dev-2**: treat `kind: "constant"` as *unverified in the head*. It means "no
step visible to a scan that cannot see the first ~227 s", not "the offset is
constant". Applying it as a container delay on id 108 would be wrong by 500 ms for
146 seconds.

### What survives from the retracted entry

Only the narrow, measured fact, and it belongs to ci and forensic rather than to my
interpretation: **stage 2's variance test cannot distinguish a real step from a bad
window, and refuses both identically.** For these files the refusal is *correct* by
`AGENT.MD`'s standard — there genuinely is no single constant offset — and it is
exactly the case zone A exists to take over. The defect is not the refusal; it is
that nothing downstream is told which of the two it was.

## 2026-09-03 — measured: on one file, stage 2 is WRONG where stage 1 is right, which inverts the design assumption

`AGENT.MD` records the owner's design intent: fingerprint peaks put stage 1 off by
roughly ±250 ms, **stage 2 runs unconditionally and corrects them**, and a residual
beyond 500 ms means stage 1 was not merely imprecise but wrong. On error id 12 the
opposite is measured.

`vmsam-ci` found four files refused by stage 2's variance test whose recorded
evidence is 8-9 of 10 windows agreeing to 1.16-2.18 ms with one wild outlier. It
asked whether the outlier window is the same window that steps in stage 1. I
measured id 12 directly — absolute, unshifted, both instruments, same windows:

| h | window | `second_correlation` | chromaprint | RMS master |
| ---: | :--- | ---: | :--- | ---: |
| **0** | [120.0, 292.0] | **-567.01 ms** | 0.8555 / **-1375** | **4306.9** |
| 1 | [206.0, 378.0] | -1417.66 | 0.9236 / -1375 | 2016.9 |
| 2-9 | … | -1412 to -1428 | 0.919-0.934 / -1375 | 1398-2084 |

**Chromaprint gets window 0 right** — the same -1375 plateau as every other window,
at the lowest fidelity of the ten (0.8555), so it registers its own reduced
confidence without being wrong. **The FFT is the instrument that fails**, by about
850 ms.

The master's RMS in window 0 is **4306.9 against 1398-2084 elsewhere** — two to
three times the energy. FFT cross-correlation is dominated by the loudest passage
in its window, so a loud passage that differs between two releases (a different
music bed, a different mix) drags the peak. Chromaprint normalises per frame and is
not dragged.

**This reproduces ci's recorded residuals**, which were taken after the candidate
was shifted by `delayUse` = chromaprint's -1375:

```
windows 1-9:  -1417 - (-1375) =  -42 ms     ci recorded -56 to -57
window 0:      -567 - (-1375) = +808 ms     ci recorded +788
```

Two extractions, two agents, same numbers to within 20 ms.

**Consequence.** For that family the refusal is not evidence about the media. It is
stage 2 failing in a loud passage and overruling a stage 1 that was correct in all
ten windows. `test_if_constant_good_delay` then reads the disagreement as *stage 1
was wrong* and refuses the pair — the exact inverse of what happened.

**REFUTED FOR THE FAMILY, 2026-09-03, by `vmsam-ci`.** The RMS mechanism above
does **not** generalise. ci measured master per-window RMS on the *correct*
comparison streams for all three:

| id | window 0 / median(1-9) | |
| ---: | ---: | :--- |
| 12 | **2.49x** | spike — the mechanism holds |
| 13 | 1.53x | elevated, but two other windows are as loud and produced no outlier |
| **108** | **0.77x** | **window 0 is QUIETER than the median, and still carries the outlier** |

id 108 refutes it outright: *"loudest passage drags the peak"* cannot explain an
outlier in a window 23 % below median energy, and that file's loud windows (5-9)
are the ones that behave. So: **mechanism established for one file, symptom shared
by three, cause of the other two unknown.** ci independently reproduced my id 12
RMS figures from its own extraction at a different sample rate (4165.3 against
1330-2010, versus my 4306.9 against 1400-2084).

**UPDATE, same day: the inversion now holds on TWO files.** id 108 measured
absolutely and unshifted, both instruments, streams pinned (master a:1 jpn):

| h | window | FFT | chromaprint | RMS master |
| ---: | :--- | ---: | :--- | ---: |
| **0** | [120.0, 334.0] | **-1453.83 ms** | 0.9047 / **-1000** | 2150.6 |
| 1-9 | … | **-949.98 to -952.27** | 0.926-0.939 / -1000 | 1202-3833 |

Windows 1-9 agree to **2 ms**; window 0's FFT is off by **502 ms**; chromaprint
holds -1000 on all ten. And **ci's RMS refutation is confirmed from my side**:
window 0 is 2150.6 while the best-behaved windows 6-8 sit at 3751-3833, so the
failing window is well *below* the loudest. Peak amplitude does not rescue it
either — 27167 at window 0 against 32768 (clipping) in the loud windows.

**The reconstruction of ci's recorded residuals now matches to 0.1 ms.** Its
values were taken after a shift by `delayUse` = chromaprint's -1000:

```
windows 1-9:   -950   - (-1000) =  +50.0 ms    ci recorded +47.8 to +49.9
window 0:    -1453.8  - (-1000) = -453.8 ms    ci recorded -453.9
```

Together with id 12's +808 against ci's +788, two agents on two extraction paths.

**And it kills my own third hypothesis.** I had wondered whether the outlier was
manufactured by `recreate_files_for_delay_adjuster`'s shifted re-extraction, which
would have tied this family to the one-point-step artefact. It is not: my
measurement involves **no shift at all** and the outlier still appears. The outlier
is in the media as the FFT sees it.

So the two claims separate cleanly, and only one generalises:

| claim | status |
| :--- | :--- |
| **the inversion** — stage 1 right on every window, stage 2 wrong on exactly one, that one value refusing the file | **holds on ids 12 and 108** |
| **the cause** — FFT dragged by the loudest passage | **holds on id 12 only**; refuted on 108, and 13 fits neither |

**What survives the refutation is the part that matters**, and it rests on ids 12
and 108 without needing the family: stage 1 was right in all ten windows, stage 2 was
wrong in one by 850 ms, and that single value refused the file. The inversion of
`AGENT.MD`'s stated design intent is real and measured, whatever explains the other
two.

**Hedged deliberately: one file, one window.** ids 13 and 108 do not share the
cause,
and ci's own classification splits the nine stage-2 variance files three ways
(a real change point seen by both stages; a first-window-only outlier with stage 1
silent; degenerate). This entry covers the second group only, and one member of it.

Also confirmed for ci, from the code rather than assumed: **window index 0 is
chronologically first** — `generate_cut_with_begin_length` increments
`begin_in_second` by `length_time` each iteration and `get_delay_by_second_method`
appends in `h` order. But **window 0 starts at `begin_in_second`, which is 120 s**
for any file over 540 s, so logos, black frames and opening credits are *before*
every comparison window and cannot explain anything about window 0.

**Not an edit.** `second_delay_test` and `test_if_constant_good_delay` are in
`mergeVideo.py` outside every tagged zone.

### `max_delay_variance_second_method`: one constant, two units

Found by `vmsam-ci`, verified here. `mergeVideo.py:25` sets it to `0.004`, and it
is compared against two dimensionally different quantities in one function:

```
:582   variance(list_delay) < 0.004          variance of SECONDS  -> s^2
:584   abs(first - last)    < 0.004          a plain difference   -> s
:594   variance(delay_detected) > 0.004      variance again       -> s^2
```

`second_correlation` returns seconds, so at :582 and :594 the constant is an s²
budget implying **√0.004 = 63.2 ms of standard deviation**, while at :584 it is
**4 ms** flat. `AGENT.MD`'s description — *"its own variance tolerance,
`max_delay_variance_second_method` = 0.004 s"* — matches :584 only.

## 2026-09-03 — the recurring defect of this campaign: a control whose pass is guaranteed by its construction

Seven instances in one day, three of them pre-dating this campaign and sitting in
documents we treat as authority. Recorded as a pattern because it is worth more
than any of the individual defects, and because every instance was found by
**running** the control somewhere new, never by reading it.

**The shape.** A control is built to prove something works. Its construction
quietly guarantees it passes whatever the thing under test does. It then reports
success forever, and the success is a statement about the harness rather than
about the world.

**The question that catches all seven, and it takes one sentence:** *if the thing
under test were broken in the way I most fear, would this control produce a
different result?* If the answer is no, it is not a control.

| # | control | why its pass was guaranteed | whose |
| ---: | :--- | :--- | :--- |
| 1 | sparse-frame pHash, validated on a pair at **zero offset** | with any offset the sampled frames never coincide — the zero-offset case is the only one that cannot expose the flaw | `vmsam-ci` |
| 2 | edge bisection with `lo`/`hi` passed reversed | `while hi - lo > tol` exits immediately and returns its own input bounds; it printed "-600 ms", which was **the instrument not running** | dev-1 |
| 3 | clean-edge estimator requiring correlation to reach **1.00000** | only a file against a *copy of itself* reaches 1.0; real encodes cap at 0.93-0.96, so it could never fire on a real pair | dev-1 |
| 4 | repair fixture built as **one file against a copy of itself** | its base offset is zero, so a sign error in deriving the base offset is invisible — it later cost a uniform +16146 ms track | dev-2 |
| 5 | post-repair probe at the **midpoint of a single segment** | a residual coefficient error grows with position and is smallest exactly at the midpoint | dev-2 |
| 6 | speed sweep with **no wrong-direction column** | `* r` where `/ r` was meant; rubberband won 33 of 33 and *"nothing internal to the run could have caught it"* | `docs/AUDIO_SPEED_POLICY.MD` weakness 3 |
| 7 | *"a file where `asetrate` leaves fidelity worse than no correction"* | the uncorrected arm is pinned near the floor **by geometry**, so almost any correction beats it — the document names this itself | `AUDIO_SPEED_POLICY.MD` |

| 8 | **agreement between two windows that both span the same step** | both are dominated by the same side of it, so they agree — and the thing they would have to disagree with is outside both fields of view | dev-2 / dev-1 |

**#8 is the sharpest form and it arrived last.** dev-2's statement of it: *a window
spanning a change point returns a well-formed number that belongs to neither side,
so a per-window disagreement is not the only failure mode — AGREEMENT can be the
failure mode.* That is exactly the locator's head blind spot: window 0 and window 1
agree on id 108 not because the offset is constant but because both are dominated
by the post-step side. **A control that cannot fail because what would refute it is
outside its field of view** is the general form, and it subsumes several of the
rows above — #1's zero offset, #4's zero base offset and #5's midpoint probe are
all fields of view chosen so the error cannot appear in them.

Campaign 1's own headline defect is the same shape one level up: `final_status`'s
success metric sat in a third branch nothing reached, and `Resolved: 0` was read as
*nothing qualifies* when it meant *nothing reached the check*.

**What actually caught them**, in every case: running the control in a context it
was not built in. #3 died on the first real error file after passing a constructed
one. #5 died when a speed-only plan produced a single segment. #4 died against a
second fixture with a non-zero base offset. **None was caught by review**, and
several had been reviewed — including by their author, in writing, after stating
the convention as an equation.

**THE CURE, and it is `vmsam-ci`'s framing.** The antidote to every row above is
**ground truth that exists independently of the thing being tested.** A constructed
fixture cannot have that property by construction: a file against a copy of itself
has an expected answer *you supplied*, so the fixture and the instrument share an
author and share his blind spots. ci's three early-step files each carry an
expected answer established by three or four instruments that know nothing about my
locator — and my committed module fails all three. That is what makes them a real
acceptance test and my fixtures not one.

Stated as a test to apply before trusting any validation set: **who produced the
expected answer, and could they have produced it wrongly in the same direction as
the thing under test?**

**The practical rules that fall out**, all of them paid for today:

- **A fixture built by copying one file cannot test anything that depends on the
  two files being *different*.** It has zero offset, identical content, and a
  correlation ceiling of 1.0, and all three flatter the instrument.
- **A directional correction must carry the wrong direction as its own column**,
  so a sign error reads as the floor instead of as a finding.
- **Probe where the error is largest, not where the answer is tidiest** — the ends
  of a segment, not its midpoint.
- **Assert a control's bracket before trusting its output.** #2 would have been
  caught by one `assert f(lo) <= threshold and f(hi) > threshold`.
- **"I could not measure" and "I measured X" are different answers.** #2 printed a
  number that was neither.

`AGENT.MD` already states the general form — *an armed guard that has never fired
is indistinguishable from a working one* — and every instance above is that
sentence being learned again at cost. The addition this campaign can make is that
**firing a guard is not enough: it must be fired somewhere it could fail.**

## 2026-09-03 — blocked need: a new runtime module is open by the rules and frozen by the checker

`WRITE_ZONES.MD` §4 lists **"new runtime modules"** among the open paths. The
checker disagrees, and the checker is what gates the commit.

**Measured.** One comment line appended to `src/change_point_locator.py`, a module
I created and committed myself:

```sh
$ python3 .../check_write_zones.py --repo .
WRITE_ZONES violations (staged changes):
  §1  frozen file modified: src/change_point_locator.py
CHECKER_EXIT=1
```

So a new module is **write-once in practice**: the creating commit is allowed and
every subsequent edit to it is a §1 violation. The cause is structural rather than
a bug in the logic — the checker carries a fixed allow-list of open paths, and a
file that did not exist when that list was written cannot be on it.

**Why this blocks rather than annoys.** Neither I nor `vmsam-dev-2` can fix a
defect in our own modules. `vmsam-dev-2` reports 007's and 009's implementations
sitting in a 400-plus-line patch beside the repository for this reason, with its
commits recording only what they did and measured. My locator is committed and I
cannot correct it if its first real run finds anything.

**What was done instead: nothing, deliberately.** The checker lives in
`VMSAM_HELP_AI/tools/`, outside the repository, and is the shared instrument that
enforces the rules on all four agents. Changing it to admit my own file is exactly
the case `AGENT.MD` names — being right about a rule does not make it mine to
change — and an agent that edits its own gate has no gate. Reported to the Lead
and recorded here.

**The fix is small and is the owner's or the checker author's to make:** treat a
path that is untracked at `HEAD` and added by the staged diff as open under §4,
or add an explicit open-list entry per new module.

### A second claim tested and NOT confirmed

`vmsam-dev-2` also reported that a `config.ini` `[features]` flag *"cannot ship at
all"*. **It can.** `src/config.ini` has **no trailing newline**, so any *append*
rewrites its final line and the checker correctly reports

```
§3  src/config.ini: line removed or altered: audio_sync=/home/vmsam/audio_sync
```

That is the rule catching a genuine modification of an existing line, which §3 does
forbid. **Inserting** the new section before an existing one instead leaves the
final line byte-identical and passes clean, with `git diff --cached` showing no
removed lines — verified on this clone, then reverted, since the flag belongs to
`vmsam-dev-2` under the territory rule.

Worth carrying: the missing trailing newline will catch the next person who
appends to that file.

## 2026-09-03 — measured: the speed detector's margin does not carry from PAL to NTSC

`docs/AUDIO_SPEED_POLICY.MD` is frozen and its ruling **stands**: `asetrate`, not
`rubberband`, for a naive speed transfer. This entry **extends** it by measurement
and is a finding for the owner, not an edit.

The corpus family is **NTSC (`1001/1000`), not PAL.** Measured on error ids 33 and
31, whose recorded delays are monotone decreasing with a 124 ms gcd. Regressing
delay against window centre gives `speed_ratio` = master/candidate of **1.000993**
and **1.000994**, ~95 % `[1.000953, 1.001034]`. `1001/1000` = 1.001000 lies inside
both; PAL's 1.042709 is excluded by four orders of magnitude. `1001/1000` and
`24/23.976` are the **same** relation — 23.976 rounds 24000/1001 — not two
candidates. This corroborates `vmsam-forensic`'s finding on two files, and its
hedge travels with it: 2.5 % is a **floor**, not a count, since a PAL-rate file
collapses to weak correlation and is structurally invisible to the method that
found these.

Residual scatter about the linear fit is 38.3 and 41.0 ms against **124/√12 =
35.8 ms** predicted by pure quantisation, so the residuals are almost entirely the
quantum and no staircase is hiding under the slope.

**The verdict, by the document's own three-arm method** (1200 s window, because
0.099 % drifts only 0.41 s over the pipeline's 418 s):

| arm | id 33 | id 31 |
| :--- | ---: | ---: |
| uncorrected (control) | 0.7411 | 0.7374 |
| **asetrate CORRECT** | **0.9588** | **0.9551** |
| asetrate WRONG (sign control) | 0.6569 | 0.6562 |
| rubberband | 0.9484 | 0.9453 |

The wrong-direction arm lands **0.084 below the uncorrected control** — it makes
things worse, as a doubled error must. Weakness 3 records a sweep that used `* r`
for `/ r` and had rubberband win 33 of 33 with nothing internal able to catch it;
that column now exists and the direction is demonstrated rather than asserted.

**The finding.** The document measures rubberband paying a **~0.109** handicap and
losing by 0.278-0.313 against real masters. **Here the gap is 0.0104 and 0.0098** —
an order of magnitude smaller. This is consistent with the document, which already
records that against a real master `asetrate` reaches only 0.94-0.98 *"because real
pairs carry content and codec differences no correction removes"*: both arms press
on that ceiling and the distance between them compresses. At 4.27 % the losing arm
had far to fall; at 0.099 % it does not.

So **the margin that would detect an inverting case in this corpus is ~0.010, not
~0.109**, and a detector built to the document's expected separation could miss
one. The 0.85 bar and the three-outcomes-and-a-decline design both survive; what
does not carry is the expected margin, which was taken from a 4.27 % corpus.

**Two files agreeing in direction and magnitude is evidence the margin is real. It
is not a rate and is not reported as one.**

*Why the alternative lost:* handing dev-2 the bare coefficient. The document is
explicit that a bare number is applied as `asetrate` by default and an inverting
case passes undetected — so the verdict, not the coefficient, is the deliverable.
dev-2 is given `1001/1000` as the **exact rational** rather than my point estimate:
my measurement is consistent with it and *less precise than it*, and propagating
measurement noise into a file when the true relation is a known rational would be
strictly worse.

## 2026-09-03 — measured: `first_delay_test`'s second pass measures on misaligned fingerprint grids, and manufactures one-point steps

**A one-point step in `delayFirstMethodAbort` can be an artefact of the pipeline's
own re-extraction, not a property of the media.** This supersedes nothing, but it
sharpens the 2026-09-03 entry on rebasing: I said the record gives reliable
*relative* structure and only an unreliable zero. At the one-point level the
relative structure is not reliable either.

`recreate_files_for_delay_adjuster` (`mergeVideo.py:333`) re-extracts the
candidate's windows at `begin + delayUse/1000` seconds. `delayUse` is an arbitrary
millisecond value and essentially never a whole multiple of the chromaprint hop,
`4096/3/11025` = **123.840 ms**, so the two fingerprint grids end up misaligned.

Measured on error id 318 — same pair, same windows, **only the shift changing**:

| shift | in hops | fidelity | residual |
| ---: | ---: | :--- | ---: |
| 0.000 ms | 0.000 | 0.968-0.975 | +552 |
| 495.360 ms | **4.000** | 0.967-0.974 | 0 |
| **552.000 ms** | **4.457** | **0.918-0.936** | **-138** |
| 619.200 ms | **5.000** | 0.967-0.974 | -138 |
| 743.040 ms | **6.000** | 0.967-0.974 | -276 |

Whole-hop shifts hold fidelity at 0.967-0.975; the fractional shift costs ~0.05.
**0.918-0.936 is exactly the band that file's `.log.error` records (0.906-0.923)**,
so the recorded fidelities are depressed by the instrument rather than the media.

Direct absolute measurement of four exemplars (own extraction, pipeline geometry,
`correlate` imported from `src/`, verified unchanged under `loudnorm`):

| id | absolute | fid | recorded |
| ---: | :--- | :--- | :--- |
| 318 | **552 constant** | 0.968-0.975 | `[0, 138 x9]` |
| 319 | **552 constant** | 0.960-0.976 | `[138 x7, 0, 0, 138]` |
| 271 | `[-250 x5, -500 x5]` real step | 0.652-0.726 | `[-250 x4, -375 x6]` |
| 316 | `6250 x9` + one `6125` | 0.582-0.683 | `[6125 x7, 6250 x3]` |

**Measured:** the offsets above, and the shift/fidelity table.
**Inferred:** that the recorded steps in 318 and 319 are produced by the
degradation. Well supported, not demonstrated — I have not driven the pipeline to
produce the artefact on demand.
**Not established:** any rate. **Four files, not a proportion.**

**Decision for `TASKS/006`: `locate_change_points` re-measures absolute offsets
itself, unshifted.** It does not derive segments from `delayFirstMethodAbort`.
`SPEC_ZONE_A.MD` §1's *"you do not need to re-run the correlation that just
failed"* holds for knowing **that** a pair failed and roughly where, and not for
the numbers the repair needs. Ten correlations per pair is the price.

*Why the alternative lost:* reading the record is free and would have produced a
chimeric splice for two files that need a single `--sync` of 552 ms.

**Blocked need, unchanged and now with a mechanism:** all of this is in
`first_delay_test`, outside every tagged zone. Recorded, not edited.

## 2026-09-03 — measured on 315 recorded refusals: the quantum model predicts 297 of 299, and three files are refused by a constant

Corpus-wide confirmation of the quantum entry below, from **production data**
rather than a constructed sweep, and at no run cost. `vmsam-forensic` found and
`vmsam-ci` extracted a `.log.error` beside every error file, each carrying the
per-window `(fidelity, offset_points, delay_ms)` triples.

`delay_ms = -offset_points x sizePoint` (`audioCorrelation.py:118`), so the quantum
is recoverable as `-delay_ms/offset_points` wherever the offset is non-zero:

| quantum | rows |
| ---: | ---: |
| 125 ms | 269 |
| 124 ms | 27 |
| 138 ms | 3 |
| self-inconsistent | 2 |

Campaign 1 reported *"859 of 933 at 125 ms … three files in this library have a
138 ms frame"*. **The three 138 ms files reappear here**, and they are 190.0 s of
audio, which is why: `L = 2*int((190-45)/11) = 26`.

Predicting the quantum from audio duration alone, using
`n = L/0.1238397 - 21.6` and `int(L/n*1000)`, matches **297 of 299** rows; both
misses are off by exactly 1 ms at a band edge, where the fitted constant rounds the
other way. The mechanism is established, not merely the correlation.

**The two "self-inconsistent" rows are not inconsistent — they are per-track-pair.**
Their logs carry 30 and 50 triples, i.e. 3 and 5 audio *track pairs* x 10 windows,
and the quantum differs **between track pairs of the same file** (125 vs 126). So
campaign 1's `DOMAIN_INVARIANTS.MD` §7 *"per-pair, not a constant"* is right and
one level too coarse: it is per **call**, and two audio tracks of one file can
carry different quanta.

**Three real files are refused by the hardcoded constant.** ids 318, 319, 320
(folder 181) record delay sets differing by **138**. Every reconciliation gate in
`first_delay_test` compares against `127` or `128` (`mergeVideo.py:270, 316, 345,
401, 416, 445, 496`), and `mergeVideo.py:496` tests each delay's magnitude against
128 rather than their difference — 138 exceeds all of them, so **no gate can
fire**. The log confirms the branch: *"Multiple delay found with the method 1 and
in test 3"*. At a 125 ms quantum the identical one-point difference reads `{0,125}`
and passes. **Carry the hedge:** the right outcome for these three is not
necessarily *merge* — their absolute offset is a constant 552 ms but their duration
delta is 1.030 s, so something differs outside the window coverage. What is
established is that the gate decides on a threshold that is a function of episode
duration rather than of the media.

Still a finding, not an edit: outside every tagged zone.

## 2026-09-03 — measured: the chromaprint point is 124–142 ms, not 125, and ten gates hardcode 125

`BRIEF_COMMON.md` and `TASKS/006`/`008` state *"One point = 125 ms, the chromaprint
frame. Every delay you see is a multiple of it."* **Measured, that is true for a
band of episode lengths and false outside it.** This changed the `TASKS/005`
ruling: it is why the video instrument must return a frame displacement rather
than a refined time, so it is recorded here and not only as a number.

`audioCorrelation.py:132` computes the quantum **per call**,
`int(lengthFile/len(fingerprint_source)*1000)`. It is not the chromaprint frame,
which is `(4096/3)/11025` = 123.84 ms; `n_items` falls short of
`lengthFile/0.12384` by a near-constant ~20 items, which lifts the ratio above the
frame by an amount depending on `lengthFile`.

```sh
# window from the ore1 example pair; items counted from fpcalc, int(L/n*1000)
ffmpeg -v error -y -ss 60 -t 700 -i <one file of the pair> -vn -ac 2 -ar 44100 \
       -acodec pcm_s16le w7.wav
for L in 20 24 40 60 100 160 214 288 436 700; do
  n=$(fpcalc -raw -length $L w7.wav | sed -n 's/^FINGERPRINT=//p' | tr ',' '\n' | wc -l)
  python3 -c "print($L, $n, int($L/$n*1000))"
done
```

| `lengthFile` L (s) | 20 | 24 | 40 | 60 | 100 | 160 | 214 | 288 | 436 | 700 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quantum (ms) | 142 | 139 | 132 | 129 | 127 | **125** | **125** | 124 | 124 | 124 |

Sweeping L in 4 s steps: **125 ms exactly for L ∈ [160, 286]**, 126+ below, 124 at
and above 288. The pipeline passes `lengthFile = lenghtTime*2` with
`lenghtTime = int((shortest_audio_duration−240)/11)` above 540 s, so 125 ms holds
for **shortest audio duration ∈ [1120 s, 1824 s)** — about 18.7 to 30.4 minutes.
Our example corpus is 1420 s, inside the band. A 45-minute episode gives L = 436
and **124 ms**; a short file on the `> 60 s` branch reads 132–142.

**Campaign 1 found this first and I am carrying their hedge, not hardening it.**
`CAMPAIGNS/01-dev-AI/DOMAIN_INVARIANTS.MD` §7: *"~125 ms typically, **138 ms
observed**"*. `WORKLOG.MD`: *"917 of 933 delay vectors have a self-consistent
ms-per-frame, **859 of them at 125 ms** … three files in this library have a 138 ms
frame."* My sweep gives 139 ms at L = 24 s, which is the mechanism behind their
138 and is consistent with it. What is new here is the **exact band** and that it
is a function of episode duration.

**The blocked need.** `first_delay_test` compares millisecond values against a
hardcoded `127`/`128` at ten sites — `mergeVideo.py:270, 316, 345, 401, 416, 445,
496, 956, 961, 969`. Those constants encode *"within one fingerprint frame"* only
when the frame is 125 ms. At 124 ms the gate is harmlessly loose; at 132–142 ms
(short files) an ordinary adjacent-frame disagreement measures more than the
constant and is misread as a real timing conflict, so the pair is discarded before
stage 2 — the only stage that resolves below fingerprint resolution.

Campaign 1 diagnosed exactly this and wrote the fix — `is_within_one_fingerprint_frame()`
and `are_residuals_within_one_frame()`, comparing frame indices directly, which is
exact for every quantum. **Neither exists in this tree**
(`grep -rn is_within_one_fingerprint_frame src/` → nothing). It is one of the
pieces that did not survive the campaign.

**What was done instead: nothing, deliberately.** `first_delay_test` is in
`mergeVideo.py` **outside every tagged zone**, so by `WRITE_ZONES.MD` rule 1 it is
frozen and this is a finding (§6), not an edit. Nothing in objectives 2 and 3 is
blocked by it: we consume the refusal, we do not need to prevent it. It becomes
blocking the day someone wants fewer false refusals rather than better repairs of
real ones.

## 2026-09-03 — `da1c191` stripped every `[0]` subscript from two files; `get_cut_time.py` retired

`WRITE_ZONES.MD` §4, `FILE_MAP.MD` and `TASKS/005` all describe the damage as a
list literal lost from three initialisers of `frame_compare.py`. **Measured, the
corruption is larger and of a different kind:**

```sh
$ for f in src/*.py; do echo "$(grep -o '\[0\]' $f | wc -l)  $(wc -l < $f)  $f"; done
0   220   src/frame_compare.py
0   564   src/get_cut_time.py
107  2118  src/mergeVideo.py
60   1144  src/video.py
```

`[0]` occurs **zero** times in both dormant files, while `[1]` and `[2]` survive
normally inside them (ten occurrences in `get_cut_time.py`) and comparable live
files carry many. `da1c191` touched these two files and nothing else. Three sites
are syntax errors; **eight more compile and raise at runtime**, in
`_detect_time_drift` (the drift regression) and `_summarize_segments` (the
per-window fidelity summary) — the two functions objectives 006 and 008 would have
been built on. They have never run.

**Decision (`TASKS/005`): repair the pHash front end, reject the alignment back
end, retire `get_cut_time.py`.**

*Why repairing the whole of `frame_compare` lost.* `_align_and_find_gap`
aggregates `costs[best_j] += best_d` — cost lands only on target frames that some
reference frame chose as its best match. Content **inserted** in the target is
chosen by nothing, so it accumulates cost 0, and the function returns the argmax
of the cost array. An insertion is precisely a run of `hits == 0`, and `hits` is
computed and never read (`grep -n hits src/frame_compare.py` → 110, 126 only). The
array carrying the signal is built and discarded. It also truncates to the first
`max_search_frames` — 30 % of the window as its caller configures it — and still
answers confidently, and it returns a region rather than a signed displacement,
which fits neither consumer. Restoring `[0]` fixes none of that.

*Why adopting `get_cut_time.py` lost.* `second_correlation` returns
`(file, offset_in_seconds)`; all four callers there divide that offset by the frame
rate — wrong by ~24× — under a comment asserting a project convention the project
does not use. The correct usage is in the tree:
`mergeVideo.get_delay_second_method_thread.run` uses `result[1]` directly and reads
`result[0]`, the filename, to recover **the sign**. `get_cut_time` ignores
`result[0]` everywhere, so it cannot know the sign of anything it measures.

*Why its eight `[0]` sites were deliberately left broken.* Repairing
`frame_compare` makes `import get_cut_time` succeed where it used to fail at import
time, so the repair makes the retired file **more** likely to be picked up. A file
that raises is safer than one that runs and is wrong by a factor of 24 with no
sign. It carries a RETIRED banner naming the four defect classes instead. Deletion
is an owner decision, not ours.

*Consequence for the frozen documents.* `python3 -m compileall -q src -x '__pycache__'`
now exits **0**. `AGENT.MD` § Testing (*"must fail on exactly one file"*) and
`WRITE_ZONES.MD` §5 check 4 both still expect one failure. That was `TASKS/005`'s
stated acceptance condition, so the change is intended — and reported here rather
than edited into either document.

## 2026-09-03 — finding: `delayFirstMethodAbort` is not uniformly rebased, so its zero cannot be trusted

`SPEC_ZONE_A.MD` §1 offers `video_obj.delayFirstMethodAbort[best_video.filePath]`
as the reason *"you do not need to re-run the correlation that just failed."* That
holds for the **structure** of the delays and not for their **absolute value**.

`first_delay_test` shifts the candidate mid-function —
`recreate_files_for_delay_adjuster(delayUse)`, `mergeVideo.py:333` — so every later
measurement is a residual against an already-shifted candidate.
`get_delays_dict(values, delayUse)` exists precisely to add `delayUse` back. Five
of the seven recording sites do so correctly: `:319` with 0 (before any shift),
`:391`, `:409`, `:478` with `delayUse`. **Two do not** — `:489` and `:502` pass
`delayUse=0` explicitly, after the shift.

So at two of seven sites the recorded delays are short by `delayUse`, and nothing
in the recorded `[1|2, delays]` structure says which site produced them. In the
staircase case `delayUse` is typically 125 ms — one point, the same size as the
step being located.

**What was done instead.** `first_delay_test` is outside every tagged zone, so this
is a finding, not an edit. Objectives 2 and 3 proceed by treating the abort record
as reliable **relative** structure — where the steps are and how large each step is
— and re-establishing the absolute zero independently on one window. That costs one
correlation, not twelve, so §1's promise is mostly preserved.

**Two smaller corrections to frozen documents, recorded for the owner:**

- **Windows are 10, not 12.** `video.number_cut = 10` (`video.py:21`);
  `generate_cut_with_begin_length` loops `range(0, number_cut)` and
  `get_delay_fidelity` loops `range(0, video.number_cut)`. `BRIEF_COMMON.md`,
  `dev-1/BRIEF.md` and `TASKS/006` all say the record is *"filled at 12 points"*.
  The rest of the geometry in those documents is correct: window length is twice
  the spacing (`length_time_converted = gmtime(length_time*2)`,
  `mergeVideo.py:756`), and coverage runs 120 s to duration−120 s, ~83 %.
- **`SPEC_ZONE_A.MD` §1 cites `mergeVideo.py:1426`** for
  `delay_same_md5_audio + delay_to_put`. In this tree it is **`:1446`**. The
  requirement it supports — that the field must be a `Decimal` — is unaffected and
  correct.
## 2026-09-03 — a measurement taken where there is no signal is not a measurement

Comparing the repaired track of error id 266 against a hand-built target, the head
region reported **−170.69 ms with r = 0.82** — a confident-looking disagreement.
It was noise correlated against noise: the first seconds of that programme are
near-silent, RMS **1 × 10⁻⁵** of full scale. With signal in the window the same
comparison reads **−0.19 ms at r = 0.974**.

**This is the most insidious member of the "control that cannot fail" family** the
campaign keeps meeting, because there is no construction to inspect. The window
was real, the instrument ran, a number came back with a plausible correlation.
Only the *content* was missing — and a high correlation is no defence, because
silence correlates with silence.

**Decision:** every verification probe now measures the RMS of both windows and
reports `no_signal` instead of a lag when either is below **1 × 10⁻⁴** of full
scale. If *no* probe on a track carries signal the track's outcome is **`skipped`
— unverified, not verified.**

*Where the threshold comes from:* the measured gap on this file, silence at
1 × 10⁻⁵ against content at 2.4 × 10⁻³ to 2.9 × 10⁻². It sits an order of
magnitude above the silence and an order below the quietest content. It is a
measured spread, not a curve (`docs/AUDIO_SPEED_POLICY.MD`: "every threshold must
come from measured spread").

*Why not use the correlation instead:* it does not work. The silent window scored
0.82–0.93, higher than several windows that were genuinely aligned. Loudness is
the discriminator here, not agreement.

*Why a per-probe outcome rather than dropping the probe silently:* dropping it
would leave a track verified on fewer probes than it appears to be. The count of
windows without signal is reported alongside the count measured.

`vmsam-dev-1` reports the same defect in its own locator — its narrowing scan
probes at fixed positions and would bisect across a silent stretch with nothing
checking for content.

## 2026-09-03 — a hand-built target needs verifying as much as the code does

`AGENT.MD` § Target-first says to build the expected file by hand, first, then
compare. On error id 266 the target was **wrong**, and the code was right.

Every region of the comparison reported an identical −170.69 ms. A constant lag
across gaps *and* plateaus means a **uniform shift**, not misplaced pieces — so
the question became which of the two objects was moving, and the way to settle it
was to correlate the *target* against the master. The target read +162 ms; the
produced track did not. Its first piece measured 1.829 s instead of 2.000, because
`-ss 0` on an AAC track drops the decoder's priming samples, and every later piece
inherited the shift.

**Decision:** build targets in exact sample counts rather than trusting ffmpeg's
`-t`, and correlate the target itself against the reference before believing a
disagreement with the code.

*Why this is a finding and not just a mistake:* `AGENT.MD` requires the target and
requires the comparison, and says nothing about verifying the target. On this
evidence it should, because a wrong target produces a **confident false failure** —
and, in the symmetric case, would produce a confident false pass. The whole weight
of § Target-first rests on an artefact that is currently assumed correct because a
human made it. `vmsam-dev-1` independently proposed the same addition after seeing
this. A finding for the owner (`AGENT.MD` § Documents), never an edit.

## 2026-09-03 — a fixture built from one file cannot test what depends on two files differing

`vmsam-ci` said it before I hit it: a control whose two arms are one file and a
copy of itself proves the gate opens and closes, and proves nothing about media
whose two sides were encoded independently — which is every real case.

Acted on. A second fixture was built from two genuinely independent releases of
the same episode: a Blu-ray master against a streaming master, FLAC 48 kHz
against AAC 44.1 kHz, natural offset constant at −955 ms and correlation
0.910–0.973 — high, but not the 1.000 that two releases sharing one audio source
read. **It found a defect on its first run.** The plan's base offset carried the
wrong sign, and the repaired track came out uniformly twice the base offset away
from the master, with the staircase correctly removed — uniform, so invisible to
any per-plateau check. The first fixture had a base offset of zero and could not
have exposed it.

The sign was written down as an equation, agreed in writing with `vmsam-dev-1`,
and verified once on the first fixture. **Stating a convention does not protect
you from misapplying it.** `vmsam-dev-1` withdrew a calibrated edge estimator on
the same day for the same underlying reason — its threshold tested for a
correlation of 1.00000, which only a file against its own copy ever reaches. Three
of us have now paid this tax.

**Decision:** the repair verifies its own output. After muxing, each rebuilt track
is correlated against the master at up to four positions and the repair is
**refused** above 100 ms. It is the automatic form of `SPEC_ZONE_A.MD` §2 point 3.

*Why not simply be more careful with the sign:* because the failure is uniform and
silent, and the whole campaign exists because five pieces of careful, reviewed,
green-tested code did nothing. A convention that can only be enforced by attention
is not enforced.

*Why the probes never sit in a gap:* a gap is filled from the master by
construction, so it matches the master at any offset. Probing there is the control
that cannot fail which `docs/AUDIO_SPEED_POLICY.MD` devotes a section to.

*Why it verifies a zero rather than measuring a step:* `vmsam-dev-1` measured that
the same physical step reads different millisecond values at different window
lengths, because the chromaprint quantum is computed per call. Re-measuring a step
would disagree with the measurement half and neither would be wrong. A zero is
quantum-independent.

*Why `repair_verify` defaults ON inside a feature that is off by default:*
`WRITE_ZONES.MD` §4's rule is that new capability must not run unless enabled, and
that is held by `repair_chimeric`. The verification is not a further capability; it
is what stops the repair shipping a silently wrong track. A safety you have to
remember to arm is not a safety.

*Why a verification refusal is `declined` and not `failed`:* the module ran,
inspected what it had produced, and said no. Collapsing the two would drop the
outcome that carries a reason — the same error as collapsing "could not measure"
into "unrepairable".

Fired both ways on real runs rather than in a unit test: the bad plan declines
with "track 1 (ja) off by 16145.9 ms", and the merge then correctly falls through
to "Only ... file left" because nothing was repaired; the corrected plan reports
"aligned, worst lag 0.5 ms". A track whose language the master lacks reports
**skipped**, never aligned.

## 2026-09-03 — the repair is reachable only because one guard in zone A was moved

`remove_not_compatible_video` ended with `if len(dict_file_path_obj) < 2: raise`.
`fusion.py:286` calls `merge_videos` with exactly **two** files — the master and
the file in error. A refusal removes one, one remains, and that line raised —
one statement *after* the repair had attached its object to
`best_video.sameAudioMD5UseForCalculation`.

So on the fusion path, which is the path this campaign exists to fix, a repair
would have been killed every single time, and so would anything built on the
measurement half. Written, reviewed, tested green, reached by nothing: the
campaign-1 shape, in the guard structure itself.

**Decision:** the guard now stands unless something was actually repaired. A
repaired object makes the merge useful on its own — verified by reading the
downstream path with a single-entry `dict_file_path_obj`:
`dict_with_video_quality_logic` is `{master:{}}`, `set_bad_video` is empty,
`best_video` resolves to the master, and the `sameAudioMD5UseForCalculation`
loop at `mergeVideo.py:1780` adds the repaired track. Then verified by running
it: a merged mkv is produced and carries the repaired tracks.

**Why not widen it further** — e.g. drop the guard altogether: with nothing
repaired, one file left really is useless to merge, and the exception is the
correct outcome. The control run with the flag shut still dies there, which is
what makes the new branch a gate rather than a deletion.

## 2026-09-03 — finding: `keep_best_audio`'s `fabricated` rule cannot fire, and the harm it prevents is now measured

`SPEC_ZONE_A.MD` §4 says the branch *"activates by itself the day the repair sets
the key"*. It does not, as written.

Both call sites — `mergeVideo.py:1853`, and `:998` reached through `:1861` —
iterate `out_video_metadata.audios`, and `out_video_metadata` is a **fresh**
`video` object built from the merged split file at `:1822` and populated by
`get_mediadata()` at `:1823`. `get_mediadata` never writes a `fabricated` key.
The key the repair sets in memory is on a different dict object, which nothing
downstream reads.

The marker does survive to that point, but under another name. Measured: a
`VMSAM_FABRICATED` track tag survives ffmpeg pass 1 (`-c copy -map_metadata 0`),
pass 2, `mkvmerge --no-global-tags` and the final `mkvmerge`, and `get_mediadata`
returns it as `audio['extra']['VMSAM_FABRICATED']` — never as
`audio['fabricated']`.

**The consequence is not hypothetical.** In a real merge the split file carried
four audio tracks — two intact from the master, two fabricated by the repair —
and the final file kept **only the fabricated ones**. A fabricated track evicted
an intact one of the same language, which is the exact sentence §4 was written
to prevent.

**What was done instead.** Nothing, because both repairs are frozen:
`keep_best_audio` carries no marker in the tree (see the entry below, 2026-09-03)
and `get_mediadata` lives in `video.py` outside its two tagged functions. The
repair sets the in-memory key anyway — it is the contract — and states plainly
that no consumer reads it. Recorded rather than resolved: `AGENT.MD` § Documents,
being right about a rule does not make it ours to change. The fix is one line in
either place and both are the owner's.

**A tempting non-fix, rejected.** The eviction could be dodged by encoding the
rebuilt track at a lower bitrate so it loses the codec comparison. That games a
broken rule at the cost of the track, and it would hide the finding. Measured
that it would not even work reliably: with the source bitrate resolved and
passed to the encoder, AAC still overshoots its target by ~4 % and the fabricated
track still compares higher.

## 2026-09-03 — finding: `main.py` is broken under Python 3.14, and the fix already exists ten files away

A plain two-file merge through `main.py` dies with `KeyError: 'ffmpeg'` raised
inside a `Pool` worker, before any campaign-2 code runs.

Cause, and `AGENT.MD` § Runtime environment already names the mechanism: Python
3.14 defaults to `forkserver`, so a pool child sees only module defaults and
`tools.software` is `{}` there. `main_gestionar_show.py:432` calls
`set_start_method('fork', force=True)` and `internal_api.py:45` calls
`tools.load_merge_runtime_from_env()`; **`main.py` calls neither.** `run.sh`
reaches `main.py` through `software=main` and `infinite_wrapper.py`.

`main.py` is frozen (`WRITE_ZONES.MD` §1), so this is a finding. Practical cost
to us: the CLI path cannot validate anything, and any agent reproducing locally
through `main.py` will hit this first and may read it as a defect of their own
work. The campaign-2 harness drives `merge_videos` through its own entry point
that forces `fork`, and says in its docstring that it is a workaround and not a
fix.

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

---

## 2026-09-03 — the locator has been checked against an instrument that shares nothing with it, and it agreed to 8 ms

`TASKS/010` set out to bound the audio path's blind spot with a video instrument.
It produced something better than the bound, and the reason is worth more than the
number.

### The two files, and the agreement

Both were drawn from a list of files the deployed audio path called *flat*. A
pHash video probe, aimed by an audio offset and scanned +/-40 s for the delta that
best matches, said content was displaced by multiple seconds over part of each
file. `change_point_locator` was then run on the same pairs, blind to that:

| | video says | locator says | apart |
| :--- | :--- | :--- | ---: |
| **id 144** | +5.00 s over [75.7, 605.5], delta 0 at 908.2 and 1210.9 | one change point, step **-5008.01 ms**, T in [600, 700] s | **8 ms** |
| **id 151** | -5.00 s from 833.1 on, delta 0 at 302.9 | one change point, step **-5004.82 ms**, T in [760, 860] s | **5 ms** |

In both cases the video's boundary between clean and displaced positions falls
**inside** the locator's bracket, and the video's first clean point falls before it.
The magnitudes agree to 8 ms and 5 ms against a video resolution of 500 ms.

### Why this is the check the locator never had

Everything that has validated the locator until now shared its physics: a
chromaprint offset checked against another chromaprint offset, a plan checked
against a target built from that plan. That is the control-that-cannot-fail, and
this campaign has now found nine of them.

**pHash on decoded video frames shares no code, no library, no signal and no
failure mode with a chromaprint correlator on PCM.** For the two instruments to
land 8 ms apart on a 5-second displacement by coincidence is not a thing that
happens. This is ground truth that exists independently of the thing being tested,
which is the only cure this campaign has found for that defect.

### What 010's deliverable turned into, and why the original one is void

The blind-spot rate cannot be computed and I am not going to report one.
`vmsam-ci` has retracted the `flat` labels the sample was drawn from: its first
scan searched +/-3000 ms, 64 of 100 files have a true offset outside that window,
and **a correlator whose true peak lies outside its search span saturates at the
edge and returns the same wrong value on every probe** — which is indistinguishable
from a flat file by every summary statistic. Its eligible-flat count is now zero,
not the ~27 the artefact produced.

So the denominator is void. **What survives is not weaker than what was planned, it
is differently aimed**: the sample was arbitrary, but the comparison inside each
file is self-contained, and it validates the locator rather than bounding the path.

### Three instrument defects found by their own controls, all in one afternoon

1. **My verdict rule had no failure class.** On 2 of 10 files the video probe's
   median distance was 21 and 24 bits of 64 — M9 calibrates same content at <= 4
   and unrelated at ~29, so those probes were reading uncorrelated video at *most*
   positions. The rule as shipped would have called that VIDEO STRUCTURE. **An
   independent instrument aimed by a dependent one is not independent, and the
   median is the check that the aiming worked.** Those two files are, on ci's new
   finding, very likely the saturated ones.
2. **A superset was about to be reported as the answer.** Video content differing
   at a position is not the same as the timeline differing there — two releases
   with different logo cards differ in content and share a timeline exactly. The
   discriminator is whether the content is *displaced* or *absent*.
3. **argmin over a tie set is not a measurement.** On static or repetitive scenes
   many deltas reach distance 0 and the minimum is arbitrary; two of my controls
   returned a confident wrong displacement that way. Reporting the width of the tie
   set converts those into "undetermined", which is an honest third answer. Both
   files above have `ties = 1` on every point quoted.

The controls killed the file I had predicted would be the headline result (id 85,
control absent at a known-clean position) and saved two I had not. That asymmetry
is the only evidence I have that the rules were not fitted to the answer.

---

## 2026-09-03 — an ffmpeg PTS seek silently absorbs `start_time`, and three modules hit it independently

**Not three defects. One property of an interface, found three times by people who
were not looking for each other's version of it.**

### The property

`ffmpeg -ss T -i file` seeks by **presentation timestamp**. A stream whose first
packet is stamped 1.103 s does not have audio at T = 0, so a probe there does not
fail and does not warn — **it returns the stream's opening and reports the
`start_time` as though it were the quantity being measured.**

Container `start_time` is not exotic. **65 of this corpus's 315 master files carry a
nonzero one on a shared language**, and the values are not noise: 21, 42, 63 and
84 ms are one, two, three and four multiples of 21.33 ms — **a single AAC frame at
48 kHz.** They are whole frames of encoder priming.

### The three instances, in three modules, each found separately

| module | what it did | how it showed up |
| :--- | :--- | :--- |
| `change_point_locator._probe` | probed at t=0 on a stream starting at 1.103 s | reported +1003 ms against a body of +22; the run splitter read a change point at the very start, the zero-width first segment was dropped, and the plan began at master 100 s instead of 0 |
| dev-2's verifier | fetched master reference audio by PTS seek | probe at master 0 asked for audio that does not exist; read -1002.75 ms on a track that was fine |
| dev-2's assembler | ended each piece with `asetpts=PTS-STARTPTS` | produced track starts at PTS 0 while the master's starts at its own `start_time`; four declines matched the master's value to under 5 ms across three distinct values |

**None of the three was found by looking for the other two**, and the first two
compound: the locator's blind spot manufactured exactly the configuration the
assembler's blind spot needed — a plan not beginning at master 0, which makes a head
master-fill piece, which is cut from the master at source 0.

### What to do about it

**Ask, of any code that seeks: what does this return when asked for a time before
the stream begins?** The honest answers are an error or the `start_time`. **ffmpeg
gives the second silently.**

Two measurements of the same stream disagreeing by ~1 s, where one decodes from the
beginning and one seeks, is this and not a media property. **The two quantities are
both correct and neither is wrong; they are different, and nothing in either name
says so.**

### The part worth more than the fix

**My own log had been printing the diagnosis all afternoon and I never read the line:**

    [change_point_locator] segment 0 unusable (offset 1003 ms, master [0.0,0.0]);
                           dropped, not declining

Four instrument defects were caught today by measurements built to catch them. **This
one was sitting in emitted output nobody read.** A diagnostic that is emitted and not
read is not a control — it is a control that cannot fail arriving from a third
direction, and it is cheaper to miss than either of the other two, because it looks
like diligence.

### And a rule for the next plausible fix

I was confident this was inaccurate input seeking, implemented a two-stage accurate
seek, re-ran the identical test, and **got identical numbers to two decimal places.
Reverted.** The change that did work moves two files, **leaves two unchanged**, has
a log line naming the mechanism, and has a consequence a consumer wanted
independently.

**The difference between the two was never confidence. It was whether anything
measurable moved** — and the half that makes it a fix rather than a shift is the
files that stayed the same.
