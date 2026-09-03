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

It is mechanically checkable — the timestamp of the last commit touching no code —
so it does not depend on anyone remembering it. **The clock is branch-wide, and every
actor resets it**: the commit that starts your hour may be someone else's
(`vmsam-forensic`, held by another agent's commit). Harmless while code is flowing, a
starvation risk when it is not — so a per-actor clock in the checker is a better fix
than this sentence.

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
