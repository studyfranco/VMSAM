# Committing — read this before you commit, not before you start

**This file is read at the moment it applies.** It was a section of `AGENT.MD` and
moved here because the section people are pointed at is the section people read
(`vmsam-dev-1`, who never opened a neighbouring section and quoted it secondhand for
hours). Everything here is about the moment work lands; nothing here is about how you
work before that.

**Commit freely everywhere below the shared branches, work in progress included.** In
`/tmp` and in your `VMSAM_WIP` clone, commit as often as the work needs — **uncommitted
work is the only thing a reset destroys**, and if you are unsure whether something
earns a commit there, it does. **But the rules below govern every SHARED branch —
`dev-AI` included — not only `master`.** A shared branch is one other agents fetch
from, and what lands on it is read as an increment by every one of them: a cadence
that is free in your clone is noise the moment it is published.

**This section is about what reaches a shared branch.** In the tiers below, commit as
often as the work needs — but see the shape rule below, because "as often as the work
needs" is not the same as "once per thought".

**A note is not a deliverable: it travels with the change it explains.** Never commit
a `TASKS/` file or a findings note on its own. Write the note when you have it —
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

## The final form of the cadence rule — measured in commits, not on a clock

**Every clock form above was an approximation of this.** A documentation-only commit
may land on a shared branch only when at least one commit that is NOT
documentation-only has landed since the previous documentation-only one:

    git rev-list --count <last-doc-only-commit>..HEAD    # must be >= 1 before yours

**The classifier is by path, and the default is NOT documentation-only.** A commit is
documentation-only when every path it touches is a `.MD`, task or notes file; anything
unclassified — a script, a fixture, a config — makes the commit count as work.

**One exception, and it is claimed in clear text.** A measurement that legitimately
changes nothing in the repository lands as one commit whose message says so and
**names the harness that produced the measurement** — an unclaimed exception is a
documentation-only commit like any other.

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

What does not: a line added to a task file, a comment, a rename, a
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
That is a normal outcome, and your task file in `CAMPAIGNS/<campaign>/TASKS/` is where it is recorded.

## Be certain of the code before you commit it

**Owner instruction, and it arrived on the day the fleet measured why it is needed:**
*fewer commits, each on one subject — group them between devs — and above all TEST MORE
BEFORE COMMITTING. Be certain of the code.*

**A stream of small commits degrades testing:** each is too small to be worth a real
test, so each gets a small one, and **the population every one of them is tested
against is whatever was convenient at the time.**

**Which is exactly the defect that pattern produces.** A rate detector was validated on
three files and reported working; **all three carried a near-unity relation, and the
instrument's error is proportional to the distance from unity.** *"The method works"
had been measured entirely inside the region where the method's error rounds to
nothing.* **A detector that refuses everything passes every negative test** — and it
took four files nobody had run it against to show it.

So, concretely, before a commit:

- **Test where the defect would be LARGEST, not where files are available.** A population
  chosen for convenience is chosen by something other than the question.
- **Run the negative and the positive.** An instrument shown only to stay silent has had
  its rejection behaviour measured and nothing else.
- **Group the work.** One commit covering a whole change can carry one real test; six
  commits covering the same change carry six token ones and no coverage.

**None of this makes a commit slower to write. It makes it slower to declare finished**,
and that is the intended effect.
