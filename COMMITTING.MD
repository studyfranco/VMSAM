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

**So the obligation lands on whoever rewrites history, which is the architect.** A
squash must carry the author date of one of its constituent commits — otherwise the
compression silently charges its own timestamp to an agent who was not committing. And
if a rewrite has displaced your commits, the rule does not apply to you until you have
re-landed what it moved.

**Three history rewrites in one day is itself the defect, and it was the architect's.**
This is also the second time this rule has starved the actor it was least meant for,
and both times the cost fell on someone who could not grant themselves an exception.
The guard is still right: it stopped that same agent committing a ruling it withdrew
twenty minutes later, and held a record until a correction arrived. It should measure
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
