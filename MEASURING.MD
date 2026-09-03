# Measuring — read this when you have a result and are about to believe it

**Not before you measure. After**, when a number exists and you are deciding what it
licenses. Every entry below is a way a result can look sound and be worthless, and
each was paid for on this corpus rather than reasoned about.

**Which paths hold test material, where outputs go and what survives a restart are
facts about the machine, not about VMSAM** — they change without the code changing,
and a stale path in a tracked file is worse than none. They live outside the
repository, with the rest of the environment. `cargo` and `compileall` belong to the
pre-commit checks in `WRITE_ZONES.MD` §5, which is where they are run.

What is left here is the part that stays true whatever machine you are on: **the ways
a result can look sound and be worthless.** Each of these was paid for.

**A unit test cannot tell you that nothing calls the unit.** Campaign 1 shipped five
pieces of code — a guard, a refined measurement, a policy branch, an order check, a
subtitle repair — that were written, reviewed and unit-tested green, and that nothing
ever called. Every suite passed. A test calls the function directly, which is exactly
what production did not do. Any new guard needs a test that pins its **call site**.

**One turn further: the unit was called, and the transport between them dropped the
field.** `parse_segments` rebuilt each segment from three named keys and discarded
the rest, including the per-stream offsets. Emitter correct, consumer correct,
integration inert — and the unit test passed throughout, because it handed the
consumer a hand-built dict and never crossed the transport.

The cost was **28.88 ms of misalignment**, which passed under four gates: below the
video frame, below `adjust_delay_to_frame`'s snapping, below `mkvmerge`'s integer
milliseconds, and below the repair's own 100 ms tolerance. Only a per-track check
saw it. It would have shipped in silence.

**So: a field whitelist in a transport is a defect by construction.** It makes every
future extension of the format inert, and the only symptom is that nothing changes —
which is the one symptom nobody investigates. Carry the whole structure and let the
consumer ignore what it does not need.

**A single-label classifier turns every percentage into a proportion of files, not
of structure.** One family per id means a file with two defects gets one label, so
*"10.2 % trimmed edges"* does not say that a tenth of this corpus's defects are
trimmed edges. Measured: 11 of 20 files labelled *speed* also carry a splice, with
total separation on both statistics and no overlap — and **applying one repair
without the other produces a file wrong in a way a structural verifier passes.** State
the unit of every share.

**A threshold that falls on a structure in the data stops being a threshold** — it
partitions the measurement space instead of scoring it. 150 ms sat between one
quantum and two and manufactured islands out of grid rounding; two findings were
retracted. Express a threshold in the data's own units, or measure where the
distribution is empty before choosing one.

**Withdraw a criterion for the case that refuted it, not beyond it.** A duration
ratio was withdrawn on NTSC evidence, where 0.0999 % does not clear content noise —
and the withdrawal took PAL with it, where 4.27 % dwarfs it. **That cost 19 files**
and a peer had to find them. A retraction has a scope and it is as easy to overstate
as a claim.

**And an instrument that reports a constant is right whenever the constant is right.**
Ten of twenty files snapped to a hard-coded value, and the tight spread read as
accuracy. A number that cannot disagree with itself is not a measurement — the same
defect as a target derived from the plan under test, arriving through a constant
instead of a plan.

**An independent instrument aimed by a dependent one is not independent.** Three
components inherited the pipeline's own 120 s blind spot because each used its window
layout as a map of where to look. The instrument was different; the geometry was the
same, and the geometry was the thing under test. Aim on an **observed symptom** — poor
coverage, a spread that will not close — never on the record you are auditing.

**The sharpest instance: three instruments, three modalities, one shared model.** A
video pHash, a loudness envelope and an audio fingerprint each fitted **one straight
line** through a displacement that was actually stepped, and agreed with each other
to **0.006 %** — all wrong in the same direction. A −1626 ms step across 1075 s tilts
a single-line fit by −0.151 %, and the effect under dispute was +0.162 %. Three
independent instruments cannot rescue a shared assumption; they only make it look
corroborated.

**And a per-file interval from one method is not an interval.** Three times in a day
an error bar came out narrower than the spread of the same quantity measured a second
way. An interval describes the scatter a method can see, and the method's own blind
directions contribute nothing to it.

**And independence of instruments is not independence of assumptions.** On one file
two methods returned 5997 s and 1377 s; the truth was 12799. **Two uncorrelated
errors bracketing a right answer look exactly like corroboration**, and what settled
it was a third party using neither method. Agreement between two instruments is
evidence only once you have said what they share.

**A null result under perturbation can be a property of the gap you perturbed in.**
A sweep over eleven constants moved almost nothing and was read as robustness. The
deviations were bimodal — 525 below 5 ms, 439 above 100 ms, **twelve in the whole
10–60 ms band and all twelve one file** — so the sweep was perturbing inside an empty
region. Check the distribution before reading a flat response as stability.

**Evidence about an intermediate is not evidence about the deliverable — name the
span yours covers.** A repair sweep measured 42 successes by reading back the repaired
*intermediate*, and never ran the merge. Everything between attaching the object and
the muxer — the file generation, the MD5 dedup, `keep_best_audio`, `mkvmerge` — sat
outside its evidence entirely. A deployed run then produced an output whose log said
one audio track rebuilt, whose subtitles carried the marker, and **whose audio track
was the master's, correlating 1.0000 with it.** Forty-two measured repairs could not
speak to it, because they measured tracks and the deliverable is a file.

**A corpus file cannot isolate one variable, because everything else about it also
differs. For *"does X alone cause Y"*, construct the case.** A container-offset
hypothesis was first tested on a real corpus file drawn from the affected list; it
returned a quality score of ~9 out of 100, because that file is from a *different cuts*
folder and the two sides are not two encodes of one programme. **The verdict did not
flip, and the test could not have detected a flip if one had existed** — caught only
because the absolute value was absurd, not because the file had been checked as
suitable. The synthetic control — one file and a copy of itself given a 21 ms offset by
`-c copy -output_ts_offset`, content bit-identical, 1517 packets each — should have been
the first run and the corpus file the confirmation (`vmsam-dev-1`, reporting the order
it got wrong alongside the result).

**A caveat in a docstring is read once, by its author. A caveat printed with the number
is read by whoever holds the number.** The cadence checker prints its
fails-toward-permitting limitation on **every run** rather than documenting it, and
names its unclassified paths with a count rather than absorbing them silently — so the
classifier's gaps are visible to whoever runs it instead of to whoever wrote it
(`vmsam-ci`). **A caveat travels with the number or it does not travel.**

**A tool that can only reach one conclusion makes every ambiguous input look like that
conclusion.** A hop-freshness script compared two refs for *inequality* and reported
STALE whenever they differed — but inequality has two directions, and only one of them
is staleness: a hub whose local branch is AHEAD of its own cache has simply promoted and
not re-fetched, which is healthy. It also printed the older commit as *"the hop holds"*,
naming the parent as the newer revision. **A false alarm and a backwards attribution, in
one report.** Its author's own framing is the rule: *I did not read the evidence and
conclude wrongly — I ran a tool that could only conclude one way, and reported what it
said.* The repair is ancestry rather than inequality, with a fifth state for **"I do not
hold both commits, so the direction is unknown"** — because direction needs both objects
and the honest answer to not having them is *unknown*, not a guess (`vmsam-dev-1`).

**And a grep counts text; parsing counts code.** Three call sites of one literal were
reported by two agents and carried into a published page by the architect. **Parsing the
module's syntax tree found two.** The third sits inside a triple-quoted block after an
unconditional `raise` — present to a text search, absent from the program. The same parse
showed the two live sites do **different things**: one routes to a heavier method and
only above a fidelity threshold, the other calls a reconciler with no threshold at all.
So the finding was not *"one constant duplicated three times"* but **one constant taking
two decisions of different kinds, with nothing at either site saying which** — a worse
defect, found only by reading the tree instead of the text.

**A grep against these documents is a measurement, and it under-reports by
construction.** The governing files are hard-wrapped, so **any phrase spanning a line
break fails a line-oriented search** — and the phrases worth checking are the long ones.
Reproduced on this revision:

    line-grep 0   normalised 1   "a hypothesis wearing an observation's label"
    line-grep 1   normalised 1   "observation from a run or a reading of the code"
    line-grep 1   normalised 1   "A labelled hypothesis beats silence"

The first is a rule the lead had quoted verbatim to the agent that then grepped zero
for it and reported the rule missing. **Normalise the whitespace before matching:**

```sh
cat *.MD | tr '\n' ' ' | tr -s ' ' | grep -o "<the phrase>"
```

**And search for the rule's own words, not your paraphrase of them.** The same report
grepped three restatements of a rule — none of them quotations — and concluded it was
absent, forty lines after writing *"a grep is not a reading"* about frozen code
(`vmsam-dev-1`, reporting its own miss). **A confident negative about a document,
reached through a proxy for reading it, is the same class as an audit reporting zero
missing rows against a denominator it could not recover.** If the answer matters, open
the file.

**A name that fits two objects will be read as the nearer one.** *"`b4c829d` carries
dev-1's four locator commits and dev-2's three repair fixes"* — true of the **range**
ending there, false of the **object**, which is one file and 68 lines. Nothing wrong had
landed and the range was right; an auditor opening the SHA would have found a single
documentation file. Self-reported by its author hours after it began relaying other
people's instances of the same class, which is how often this one recurs.

**And a reading of the code is not a run of it.** The span was read and pronounced
safe — *"nothing between the attach and the muxer looks at where the track came
from"*, hedged honestly as read-not-run. It was run, and the reading was wrong. A
hedge is not a substitute for the run; it only tells you which claim to retract first.

**A mutation placed before code that can raise makes the caller's error message
false.** A repair attached its result to the shared object, then raised three lines
later on a name lookup, outside the `try`. The caller caught it and logged *"the
repair raised and was abandoned"* — while the object was already attached. State
mutated, result discarded, and the one word in the log that would stop anyone looking
further was the wrong one. Either the mutation is the last statement before the
return, or it lives inside the `try` that reports on it; anything else and the
handler describes a world it did not restore.

That defect also shows the guard rule from its other side: every refusal branch was
correct and only the success path was broken, so **fifty-six green checks had never
walked it.** An armed guard that has never fired is one failure; everything *but* the
guard having never fired is the same failure wearing the opposite face.

**Which rules deserve to be mechanical, and it is not the important ones.** The rules
that survived today are the ones whose violation produces an immediate, local,
unambiguous failure in the actor's own frame. The checker refuses a commit and you
feel it at once — it is the only rule nobody has broken. A waiter never returns and
you feel it six hours later, which is why it needed a check rather than a sentence. A
report written but not sent produces **nothing at all**: no error, no delay, no
symptom in the writer's world, which is why no wording fixes it.

**So: mechanise the rules whose failure is silent in the actor's own frame — not the
ones that matter most, the ones the actor cannot feel going wrong.** Everything else
can be a sentence.

**And an armed guard that has never fired is indistinguishable from a working one.**
Campaign 1's success metric sat in the third branch of a function nothing reached:
315 rows exited earlier, and `Resolved: 0` was read as *nothing qualifies* when it
meant *nothing reached the check*. Fire every guard deliberately, with literals,
plus a case proving it can return the **other** answer.

**A threshold in fixed units, compared against an instrument whose resolution
varies, is a different threshold on every file.** The reconciliation window is the
literal `< 127` ms. The delay grid is `int(lengthFile / n_points * 1000)` — file
dependent, measured at **124–138 ms across the corpus**, never the fixed 125 everyone
assumed. So on a long file one quantum is *under* the window and a one-quantum
disagreement reconciles silently; on a short file one quantum is *over* it and the
same disagreement is refused. **The corpus contains both arms of the experiment**: the
error record's floor sits at exactly 2.00 quanta with 0 of 279 files below it, and the
only files present at one quantum are the three whose quantum is 138 ms (`vmsam-forensic`,
`vmsam-ci`). The defect is not the value 127; it is a millisecond compared against a
quantum, with nothing at the three call sites saying so.

**Its consequence is the largest blind spot the campaign has found.** A file whose only
defect is sub-quantum reconciles, is accepted and ships — so the error corpus *cannot
contain* that population. Every blind-spot figure measured on refusals is conditioned
on a file having failed loudly enough to be recorded, and none of them says anything
about what the pipeline accepts.

**Controls that cannot fail, three more, all found by their own authors.** The armed
guard above was the first; these are the family.

- **Both arms of an A/B taking the same path agree trivially.** A blanket
  `except Exception` around an optional correlator degrades to the fallback for *any*
  reason — absent, crashed, timed out, malformed — with the same log line either way,
  so an agent switched instruments mid-campaign without noticing. Its first comparison
  of the two paths would have run the fallback twice. **It only became a test when each
  arm printed which path it had actually taken** (`vmsam-dev-1`; the two paths then
  agreed to 0.00 ms on six probes, including a −29.5 s offset — which makes the earlier
  numbers lucky, not designed).
- **A control drawn from the favourable half of the population validates nothing about
  the rest.** A correlation floor was removed on a control reading r = 0.058 — but that
  control used two encodes of *one mix*, where a weak peak is still sharp. In the wild a
  same-language pair may be two different recordings, where a weak peak is broad and its
  argmax arbitrary. Six apparent desyncs, spreads to 3.7 s, were each **one deviant probe
  at near-noise correlation** against four agreeing exactly. Corrected: 0 of 48
  (`vmsam-ci`).
- **A control that passes when its input is absent is not a control.** A guard reading
  a clock file permits when the file is missing or unreadable; an audit reports zero
  missing against a denominator it could not recover. Named three times in one day by
  three different agents, each about a different mechanism. **Ask what your check does
  when its evidence is gone — if the answer is "passes", it is decoration.**
- **A verifier that checks the artefact it built, and not the one that shipped, cannot
  report damage.** *"Damaged is 0"* stayed a non-measurement for that structural reason,
  not for want of running it again (`vmsam-forensic`).

**Store the raw quantity beside the verdict, and the verdict becomes reversible.** The
scan above kept every probe's correlation next to its lag. **Its own verdict column was
wrong and the data underneath overturned it without re-measuring anything** — the
difference between a bad afternoon and a lost week.

**A prefix of the identifiers is not a sample.** A first pass over the first 80 ids
gave 71 % for a structural defect; the full corpus gave **22.6 %**, because the low ids
are the divergence cluster. Ordering carries information about content far more often
than it looks like it does (`vmsam-forensic`, caught within minutes of its own result).

**What absence proves depends on which direction you read it.** A commit missing from
your object store proves it is not an ancestor of anything you hold — that settled a
provenance question. **It proves nothing about a claim relating two commits you do not
hold**: two agents could not verify a byte-identical claim about two commits neither
had fetched, and correctly said so rather than assuming either way (`vmsam-ci`).

**An interface that answers a question you did not ask, without erroring, is the
hardest instrument to distrust.** `ffmpeg -ss T -i file`, asked for a time before the
stream begins, does not fail and does not warn: it returns the stream's opening and
reports `start_time` as though it were the measured quantity. **65 of 315 masters carry
a nonzero one**, in whole AAC frames of encoder priming (21, 42, 63, 84 ms — 1–4 ×
21.33 ms at 48 kHz). Three consumers hit it independently in one day, and in every case
**the number came back clean and wrong** (`vmsam-dev-1`, `vmsam-dev-sandbox`).
