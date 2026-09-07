# Committing — read this before you commit, not before you start

**This file is read at the moment it applies.** It was a section of `AGENT.MD` and
moved here because the section people are pointed at is the section people read
(`vmsam-dev-1`, who never opened a neighbouring section and quoted it secondhand for
hours). Everything here is about the moment work lands; nothing here is about how you
work before that.

**Commit freely in your own working space, work in progress included.** In `/tmp` and
in your `VMSAM_WIP` clone, commit as often as the work needs — **uncommitted work is
the only thing a reset destroys**, and this file's framing once read as a standard of
worthiness and left a modified file exposed through a branch rewrite (`vmsam-dev-2`,
`vmsam-dev-1`). If you are unsure whether something earns a commit **where nobody else
reads it**, it does.

**The rules below govern what reaches a shared branch — `dev-AI` as much as `master`.**
This sentence used to read *"what reaches `master`, and nothing else"*, and it named
the campaign branch as free. **That was wrong, and it was wrong in a way that took four
people one day to surface.** The measurement this whole rule was written from — 78
commits in five hours, 69 of them notes or task files alone — **was taken on `dev-AI`.**
A rule motivated by one branch's history, exempting that branch by name, is a
contradiction the document carried for hours while everyone argued about the count.

**The four instances, because the shape matters more than any of them.** An agent
applied the promotion gate to its own WIP clone, where it does not reach. The lead
accepted that reading and reported it upward as a cost of the rule. The architect held
two commits on `dev-AI` and announced it as the rule binding its author — then
"corrected" itself on the strength of the defective sentence, publishing the wrong
correction. **And the checker built afterwards to prevent all three measured `dev-AI`,
which is right, and nobody verified its target because everyone had just been corrected
about scope and was reasoning about the count instead.** Only the tool was correct, and
it was the only one nobody checked.

**So: the architect's earlier correction is withdrawn. The hold was right.** And the
agent's WIP-clone gating was still wrong — the two errors were opposite, not the same.

**One narrow exception, because a defective gate must not be able to seal itself in.**
A change to *this file's gating rules* is not subject to them. Anything else waits.

### A refused promotion must leave an artefact

**The promotion is the fragile step, not the rewrite.** Agents do not push; the lead
does. So a commit that exists in a WIP clone and has not been promoted is invisible to
everyone including its author, who has no way to tell "promoted" from "attempted and
refused" except by being told.

**A successful promotion leaves a commit. A refused one leaves a line in a terminal and
nothing else.** That asymmetry is the same defect this campaign named twice elsewhere —
a decline that reports what it measured and not what it produced — sitting in the
workflow of the only actor who can promote. **Every refused promotion gets a durable
record: what was attempted, whose it was, why it was refused, and what would unblock
it.** A queue file in `VMSAM_HELP_AI/lead/` is enough; the requirement is that an agent
and the architect can both read it without asking.

**Here is what it cost the day it was missing.** A promotion was attempted, the cadence
checker refused it correctly one minute after its sibling, and the lead read the
refusal. It then told the agent the content had landed *"with the rewrite rather than
being lost"* — **a mechanism it invented, contradicted by output it had personally read
minutes earlier**, in a message whose purpose was to reassure. The finding sat
unpromoted for an hour while its author believed it was safe, and it was the only
quantified statement anyone had about files the pipeline had shipped. **It survived
because the agent had not pruned its own object store.**

**And the exception that covered it was already written.** A measurement whose harness
lives outside the repository earns one commit saying what was established — the rule was
in this file, a hundred lines from the gate, and the lead did not claim it. Nobody could
find that paragraph by searching for it; that is fixed above, and this is what the gap
cost while it stood.

**The lead reported all of it against itself, having been told to check its own
promotion log rather than the rewrite it had blamed.**

**This section is about the last hop — what reaches `master`.** In the tiers above
it, commit as often as the work needs — but see the shape rule below, because "as
often as the work needs" is not the same as "once per thought".

**A note is not a deliverable: it travels with the change it explains.** Never commit
`AGENT_NOTES.MD` or a `TASKS/` file on its own. Write the note when you have it —
that discipline is right — and let it ride along with the code, the measurement or
the decision it is about. Measured over campaign 2's first five hours: **78 commits,
of which 5 touched code and 69 were notes or task files alone.** Campaign 1 was 70 %
documentation-only; that start was 93 %. The findings in them were good. Committing
each one by itself is what made the branch unreadable, and it is the same defect as
campaign 1 wearing better prose.

**A commit that touches no code lands at most once per hour.** Documentation, task
records and logbook entries accumulate and travel together — with the next code
change if one is coming, and otherwise on the hour. This is the rule the other two
were reaching for and kept missing: squashing fixed the mechanism, the promotion
criterion fixed the cadence, and both were still satisfied by two documentation
commits in the same minute. Measured on campaign 2: five documentation-only commits,
four of them in two same-minute pairs, two of which were the author of this rule.

**One documentation-only commit per hour on the branch, and it carries everyone's.**
The lead does not promote a documentation-only change on its own: it holds it until
another agent's documentation change or a code change joins it, and promotes them
together. So no agent competes for a slot, and the branch takes one such commit an
hour rather than one per actor.

**Both simpler rules were tried and both were wrong.** A branch-wide clock with
agents committing directly starves whoever commits least often — one agent's finished
work sat blocked over an hour, reset three times, every time by someone else's commit
and never once by its own. A per-author clock removes the starvation and produces
five commits an hour instead of one: four landed in six minutes, each compliant.
Batching at the lead fixes both, because the thing that was scarce was never the
clock — it was that agents were racing for the same slot at all.

**In your own clone this does not apply.** Commit documentation as freely as
anything else; the rule governs what reaches the branch.

**Nor does it apply on the campaign branch, and the author of this rule got that wrong
twice in one evening.** The first paragraph of this file says it before any rule does —
*commit freely everywhere below the last hop; on the campaign branch, commit as often as
the work needs.* Two agents had already enforced a promotion gate against a work-in-
progress clone that day and retracted it; the architect then held two governance
corrections under the same misreading, announced the hold to two agents as a
demonstration of the rule binding its author, and was wrong about which hop it bound.
**Three instances, three actors, one file, one afternoon.** A rule whose scope is stated
in a paragraph nobody re-reads will be applied where it does not reach — by the people
most anxious to obey it.

**The rule measures an author's wall-clock cadence, and a squashed branch cannot
carry it.** Keyed on commit time it restarts the clock of whichever author a squashed
commit is attributed to. Measured on 2026-09-03: three rewrites in one day, the lead's
campaign record **blocked six times, landing about four hours after its first
attempt** — while the agent kept obeying a rule that was timing the architect's
rewrites and calling them its own commits.

**The obvious fix is refuted, and by measurement rather than by argument.** Reading
`%ad` instead of `%cd` is right in general and changes nothing here: of eight commits
on the branch, **exactly one carried a different author and commit time**, and that one
had been *rebased*. **A rebase preserves the author date; a squash destroys it** — the
new commit is authored when the squash ran, so on a squashed commit the quantity the
rule needs **is not in the object at all** and no choice of field recovers it
(`vmsam-dev-sandbox`, which reasoned its way to the `%ad` fix, measured it, found it
inert, and did not ship it; the agent with the strongest stake in weakening this rule
is the one that refused to weaken it on an untested fix).

**And that fix is refuted too — by the same test, from the other end.** Reproduced
independently in a scratch repository: three commits authored 2020-01-01/-02/-03,
squashed by the usual soft-reset recipe, and the result's **author** date came out as
*now*. Both fields reset. `git commit --amend --date=` can put one back, so preserving
a constituent's author date is implementable — and it is **the wrong repair**. *Which*
constituent? Preserve the first and the number means "when the earliest batched
document was written"; preserve the last and it means "when the newest was". **Neither
is the quantity the rule names**, and a different quantity wearing the same name is this
campaign's most-repeated defect, reintroduced by its own fix (`vmsam-forensic`).

**The cause is not the squash. It is the batching, and the squash only made it
visible.** Documentation is batched at the lead, which commits another agent's document
with `--author=<agent>` **at the lead's moment, not the agent's**. From that point on,
*"when did this agent last document"* **is not a property of the branch at all** — no
field can carry it, because the agent did not make the commit. The six blocks began
after the first flatten and not before, and the defect had been there since batching
began with nothing to surface it.

    we both had it as:  the rule reads the wrong field
    corrected to:       the rule reads a field whose content was destroyed
    the test says:      the rule reads an object that never carried the quantity

**So the clock does not live in a timestamp at all. The rule is expressed in commits,
not in hours.**

    git rev-list --count <last documentation-only commit>..origin/dev-AI

**At least N commits between documentation-only promotions.** A documentation-only
commit is identified from its **paths**, and that reading classified 12 of 12 recent
commits correctly with no metadata read at all (`vmsam-forensic`).

This is what the rule always meant. *"Accumulate and travel together — with the next
code change if one is coming"* is a statement about **work**, and the hour was only ever
a fallback timer standing in for it. Expressed in commits it survives everything that
destroyed the timestamp versions:

| | |
|---|---|
| **authorship** | nobody writes it. Derived from the tree — no step to skip, no value to fill in |
| **visibility** | any clone holding the branch computes the same number |
| **rewrites** | it is not a stored quantity a rewrite can destroy. It is **recomputed from current content**, so a squash turning 26 commits into 5 legitimately lowers it — there genuinely is less distinct work on the branch |

**Two clocks were proposed before this one and both were wrong; the record of why is
above, and it is shorter than the argument would be.** A file beside the lead is
authored by the party it guards and permits when it is missing. The reflog closes
authorship — git writes it as a side effect of the ref moving, and a rewrite appends
rather than edits — but it is **local to one clone**, so nobody but the lead can read
the clock.

**And the retrospective audit that was supposed to cover that gap is defeated by
measurement.** This document claimed the branch's own shape audits the outcome even if
nobody can read the clock. Counted on the branch: **three documentation-only commits
attributed to the lead, against thirty-two it actually made.** Nine of them became one.
The branch does not record an author's cadence; **it records the squashes' cadence**,
and an auditor counting would have found the lead exemplary. Both parties reasoned
about what the branch *would* show and neither looked at what it *does*.

**Two caveats that belong in the rule rather than in its history.**

- **It fails toward permitting, in one specific way.** A squash that merges a
  documentation commit into a code commit makes it no longer documentation-only, so the
  anchor moves backwards and the count *grows*. Wrong direction; know it is there.
- **The path classifier is now the load-bearing part, so spell it the way
  `tools.py` spells subtitles and not the way `audioRules` is spelled.** `*.MD`, `*.md`,
  `docs/` is an allow-list with no `else`, which is the exact defect reported against
  `audioRules` the same day. Every path lands in a named class, and **the default when
  nothing matches is "not documentation-only"** — the strict side, because the failure
  above already leans permissive.

**The rule assumes an agent's output is commits, and one agent's output is
measurements.** A gate keyed on commits cannot see work that lands in a directory git
does not watch — **not to throttle it, and not to credit it**. One agent's whole evening
of measurements was invisible until the lead thought to ask where it went: 0 modified,
0 ahead, 0 changes under `src/`, while four fidelity controls, a driver and two fixes
all existed. **Uncommittable by nature, not by misreading.**

**A retraction belongs here, because the instance first reported alongside that shape
was not one.** The campaign's only measurement of the silently-accepted population was
said to have been stranded by this rule. **It was not.** The first paragraph of this
file governs the case and says so before any rule does: *commit freely everywhere below
the last hop, work in progress included.* A promotion gate was enforced against a WIP
clone, by an agent who then reported the cost, and by a lead who accepted the framing
without checking which hop the rule governed. **The cost was real and the cause was not
the rule** — a substantive finding sat uncommitted through a day of branch rewrites,
against a file that warns in its own words, naming two agents who already paid for it,
that uncommitted work is the only thing a reset destroys.

**Both parties retracted it themselves, unprompted, and ranked it above the defect they
had reported an hour earlier**: the unreachable exception cost a lookup; this cost
exposure.

**That work is already excepted — the exception is a hundred lines below, under a
different heading, and nobody found it.** See *"work that legitimately produces no
repository change"*: a verification whose harness lives outside the repository, a
measurement that confirms something is inert. **It earns one commit saying what was
established**, and the checker must implement it, not only the counting rule above it.
An agent offered the obvious workaround — committing an unrelated deliverable to buy
the slot — and refused it on its own: *committing a document to unblock a tool that
blocks documentation commits would be routing around the rule with the rule's own
subject matter.*

**Claim the exception in the commit message, naming what was established and where the
harness lives.** Only the author knows whether work produced no repository change, so
the claim is self-granted. **Visibility is what makes self-granting acceptable, not the
granter's discipline** (`vmsam-ci`): the clock failed because its input was a reflog only
one machine held, while this claim is authored in the open, in the branch, attributable
forever, and **falsifiable by anyone who reads the commit and the harness it names**. **The architect failed to find this exception with
a grep while looking for it, one hour after publishing the rule that a grep against
these hard-wrapped documents under-reports by construction.** A rule a hundred lines
from the rule it excepts is a rule nobody holds.

**What no version of this rule audits, and it is what the guard is actually worth.**
Its value today was not preventing branch noise. It stopped a ruling that was withdrawn
twenty minutes later, and held a record until a correction arrived: **it prevented
prematurity, and no count of commits can see a statement that never landed.** That is
not covered by anything here, on a rewritten branch or a clean one, and saying so is
better than a third claim that it is.

**The guard stays, and the reason is the lead's own.** Both of those delays improved
what landed. The agent this rule constrains most argued *against* replacing it with its
own discretion, and disclosed its stake before relaying an argument that favoured it.
**Whatever guards this must still be able to stop the lead, and must not rest on the
lead's judgement about its own output** — that is the one check nobody has.

**Three history rewrites in one day is itself the defect, and it was the architect's.**
This is also the second time this rule has starved the actor it was least meant for,
and both times the cost fell on someone who could not grant themselves an exception.
The guard is still right, for the reason given above. It should measure
the quantity it names.

**It was branch-wide and that was wrong.** `vmsam-forensic` predicted the failure as
a property of the rule — five actors reset one clock, and the agent committing least
often is the one starved — and then lived it: its `TASKS/003`, holding objective 1
closed with the command that reproduces the family table, sat blocked for over an
hour and was reset three times, **every time by someone else's commit and never once
by its own**. Three of those resets were mine, made under an exception granted to me
whose cost was paid by an agent who did not get to argue for one.

A per-actor clock corrects the behaviour the rule was written for — the two
same-minute pairs that caused it were each one author twice — and starves nobody.

**The exception, and it is a real one: work that legitimately produces no repository
change.** A verification whose harness lives outside the repository, a measurement
that confirms something is inert, a task that closes because the answer was "nothing
to do" — those produce a result and no code, so there is nothing for the note to ride
with. One commit, saying what was established. The test is not *does it touch code*
but **is this an increment or a thought**: a closed task is an increment, and
seventy-three of them in a day were not.

Found by `vmsam-lead`, by doing the right thing against the rule as written.

A commit is a unit of **result**, not of activity. What earns one:

- a feature that works end to end — the code *and* the measurement that shows it works;
- a blocking bug fixed;
- a complete, coherent change to one zone.

What does not: a line added to `AGENT_NOTES.MD`, a comment, a rename, a
reformatting, a half-written function, "work in progress". Those wait and travel
with the work they belong to.

**What this rule is made of.** Campaign 1 produced **1914 commits in six days**.
Of the 1451 that carry files:

| | |
| :--- | ---: |
| commits touching documentation only | **70 %** |
| commits touching a single file | **70 %** |
| commits touching only a notes / state / task file | **393** |
| merge commits | **489** |
| commits touching `mergeVideo.py`, the engine | **63** |

`DOMAIN_INVARIANTS.MD` was committed **217 times** — three and a half times more
often than the file the campaign existed to improve. A history in that shape cannot be reviewed:
the work that mattered is not hidden by the noise, it is *indistinguishable* from it.

And `WRITE_ZONES.MD` §5's checks run per commit, so a commit carrying nothing
still costs a full verification.

If a day's work produced nothing that fits the list above, it produces **no commit**.
That is a normal outcome, and `AGENT_NOTES.MD` is where it is recorded.

## Be certain of the code before you commit it

**Owner instruction, and it arrived on the day the fleet measured why it is needed:**
*fewer commits, each on one subject — group them between devs — and above all TEST MORE
BEFORE COMMITTING. Be certain of the code.*

**Sixty-four commits became six, and the reason given was that the first form was
unreadable.** That is the visible cost. **The invisible one is what a stream of small
commits does to testing:** each is too small to be worth a real test, so each gets a
small one, and **the population every one of them is tested against is whatever was
convenient at the time.**

**Which is exactly the defect that surfaced the same afternoon.** A rate detector was
validated on three files and reported working; **all three carried a near-unity relation,
and the instrument's error is proportional to the distance from unity.** *"The method
works" had been measured entirely inside the region where the method's error rounds to
nothing.* **A detector that refuses everything passes every negative test** — and it took
four files nobody had run it against to show it.

So, concretely, before a commit:

- **Test where the defect would be LARGEST, not where files are available.** A population
  chosen for convenience is chosen by something other than the question.
- **Run the negative and the positive.** An instrument shown only to stay silent has had
  its rejection behaviour measured and nothing else.
- **Group the work.** One commit covering a whole change can carry one real test; six
  commits covering the same change carry six token ones and no coverage.

**None of this makes a commit slower to write. It makes it slower to declare finished**,
and that is the intended effect.

---

# WHERE YOU WORK — FOUR TIERS, ONE PUBLISHED COMMIT

*Moved from `AGENT.MD` 2026-09-05: it is promotion mechanics and this is the promotion document.*

## 🏗️ Where you work — four tiers, one published commit

You do not edit `master` directly. Each tier is a **git clone** of the one above it:

| tier | what it is for | commits |
| :--- | :--- | :--- |
| `/tmp/…` | experiments — try things, break things | as many as you like |
| `VMSAM_WIP/` | what you have validated and are still assembling | as many as you like |
| the **campaign branch** | what CI builds and the test instance runs | as many as you like — **a deploy requires a push** |
| `master` | the published history | **one per feature**, by squash |

**The campaign branch is a tier, not the record.** § Commits binds the last hop
only: testing costs a push, and a rule that made you hesitate to test would be
worse than a noisy branch. Commit there as freely as the work needs.

**Clones, never copies.** A promotion by `cp` overwrites without warning, and the
owner commits while you work — measured on this repository:
a copy silently erased an owner change to the same file, while the same promotion
through git stopped on the conflict and preserved it.

```sh
# promote /tmp -> VMSAM_WIP : keep the history, it costs nothing here
git fetch <tmp-clone> <branch> && git merge FETCH_HEAD

# promote VMSAM_WIP -> campaign branch : ONE commit per promotion
git fetch <wip-clone> <branch> && git merge --squash FETCH_HEAD && git commit && git push

# campaign branch -> master : one commit, whatever the history behind it
git fetch <wip-clone> <branch> && git merge --squash FETCH_HEAD && git commit
```

`--squash` is what makes § Commits structural instead of a rule to remember: the
command cannot produce two commits.

**But it only fixes the mechanism, not the cadence.** Campaign 2 promoted three times
in twenty-six minutes, each promotion correctly one commit — and two of the three
were halves of a single increment: the locator fixed, then sixteen minutes later the
locator wired to its consumer. The letter held and the intent did not; the
granularity had simply moved from commits to promotions.

**Promote when the branch needs to carry it** — when it must be tested in the
container, or when another agent needs it to proceed. Not per fix, and never on a
rhythm. If your next promotion continues the last one, it was one promotion.

**It is used at two hops, and the second one is why.** Campaign 2's first day put 79
commits on the branch, 73 of them documentation alone — and an audit found that
**nobody had broken a rule**. Commits are free in the tiers above, correctly; the
promotion preserved history, also correctly; and the two compose into a branch where
six real pieces of work sit among seventy-three entries about thinking. Two correct
rules composing badly is not fixed by asking anyone to try harder. Squashing at the
promotion fixes it structurally: **one commit per promotion**, and you promote when
you have something worth testing, so the branch reads as a list of increments. The
granular history stays in your `VMSAM_WIP` clone, where it costs nothing and belongs.

**Pull down often, push up rarely.** Sync the tier above *into* the one below
before promoting, and `VMSAM_WIP` into `/tmp` when you start something. Divergence
is cheap to resolve daily and expensive to resolve once.

**Run `WRITE_ZONES.MD` §5 in `VMSAM_WIP`, not at the promotion.** The checks take a
second. Discovering at the last hop that half your work sits outside a tagged zone
is how a campaign ends up negotiating its own rules.

Never write a tier's filesystem path in a tracked file — Privacy, below.


---

# WHAT COUNTS AS DONE — AND THE TWO ROLES THAT DECIDE IT

## ✅ What counts as done — and the two roles that decide it

**A modification is not valid until the test container, on the latest image, has PRODUCED A
FILE — and that file has been COMPARED AND VALIDATED.** The owner's standard, stated because
five agents had each filled the blank differently and none of the fillings was this.

    PRODUCED    an artefact exists on disk, from the container, on the current image
    CHECKED     named automated properties verified -- NAME THE PROPERTY, never "passed"
    VALIDATED   a person or the validator compared it and accepted it -- AND WHO

**Only the third is done.** Ten commits were promoted in one morning through compile,
unbound-name checks, zone checks, blob and tree verification — **and column one was empty for
all ten.** A compile is not a run. A guard suite is not production. **And a lab measurement that
explains why a file fails is not evidence that anything was fixed.**

> **THE LAST CHECK ANYONE RUNS BECOMES THE DEFINITION OF DONE UNLESS A DEFINITION IS WRITTEN
> ABOVE IT.**

That is why this section exists. Nobody wrote one, so the last thing that said yes became it —
and **each time that gate got stricter it made the wrong definition more convincing**, because a
stricter check that still cannot see a file reads as more evidence rather than the same kind.

**Two agents filled the blank in public and the pair is the point:** *my instrument's preference
was for the outcome that makes the instrument look right — refuse what I cannot verify*, and
*mine was it compiles and the blobs match.* **Neither chose dishonestly. Both answered the
question their tools could answer.**

### Why is not whether

**A diagnostic reported as a diagnostic is not waste — it is the route by which a refused file
becomes a produced one.** Mapping a refusal into six segments whose arithmetic closes to 0.097 s
is what will let a file come out of that release. **The defect was never the diagnostic; it was
the diagnostic reported as a completion.**

### The two support roles

    ci        GATES THE ENTRANCE to the most constraining test, the end-to-end one.
              AND DETECTS WHETHER A DEV AGENT HAS DODGED THE FINAL VALIDATION.
    forensic  VALIDATES THE REPAIR. "The end-to-end test produced this; what I expected
              is this." VALIDATES OR REFUTES -- AND EXPLAINS WHY IN BOTH CASES.

**The one who lets a file in must not be the one who declares it good.** ci sees what went in
and under which image; forensic sees what came out against what was expected. **Neither can
produce the other's answer from its own inputs** — the independence test, applied to roles.

**And an anti-dodge report goes to the owner, not upward through the thing it checks.** That
holds for the lead's promotions and for the architect's documents alike: *a check that only
reports upward through the thing it is checking is not a check.* A role that depends on an agent
choosing to audit itself is the control-that-cannot-fail wearing a virtue.

### The boundary the validator drew, and it is binding

**A derived expectation — from the master, the plan, and what the log says was done — validates
INTERNAL CONSISTENCY: the artefact matches what the pipeline claims it did. IT CANNOT CATCH A
DEFECT THE PIPELINE WAS NOT MODELLING.** The truncation the owner found by watching was exactly
that. **The owner's validation and the validator's are different instruments and the assignable
one is weaker.** Every artefact that passes is consistent with its own claims — nothing more,
and that is still the difference between an unexamined file and a checked one.

### And the fleet's job is not to replace the person who validates

**It is to make the files cheap to check.** A produced file alone costs an episode of someone's
attention; a produced file with a report saying what was cut, what was added, at which timings,
from which language, and what remained, costs a minute of reading and points at the seconds
worth watching. **That is why the plan report in `SPEC_ZONE_A` §4g is a validation instrument
and not documentation.**

