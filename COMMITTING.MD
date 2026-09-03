# Committing — read this before you commit, not before you start

**This file is read at the moment it applies.** It was a section of `AGENT.MD` and
moved here because the section people are pointed at is the section people read
(`vmsam-dev-1`, who never opened a neighbouring section and quoted it secondhand for
hours). Everything here is about the moment work lands; nothing here is about how you
work before that.

**Commit freely everywhere below the last hop, work in progress included.** The rules
below govern what reaches `master`, and nothing else. In `/tmp`, in your `VMSAM_WIP`
clone and on the campaign branch, commit as often as the work needs — **uncommitted
work is the only thing a reset destroys**, and this file's framing once read as a
standard of worthiness and left a modified file exposed through a branch rewrite
(`vmsam-dev-2`, `vmsam-dev-1`). If you are unsure whether something earns a commit
where you are, it does.

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
