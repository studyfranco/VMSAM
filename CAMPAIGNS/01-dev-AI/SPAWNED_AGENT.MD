# You are a spawned agent — read this before you start

**This file is addressed to you, not to the lead.** It says how you talk to the
lead, when you must, and what a report has to contain. `AGENT.MD` tells you
about the project; this tells you how to work inside a team.

Read it once at the start of your session. It is short on purpose.

---

## Who to talk to, and how

You reach the lead and your peers with `SendMessage`. **Your plain output is not
visible to anyone else** — if you do not send a message, nothing you say leaves
your session.

**The lead's address is the `from=` attribute of any message it sends you.**
Copy it verbatim as your `to`. Do not guess a name and do not reuse a name from
an earlier session; sessions are recreated and names go stale. If you have no
message from the lead yet, `ListAgents` shows every peer that is reachable.

**Peers are addressed the same way, by the name `ListAgents` prints.** Talk to
them directly — see *Who owns what* in `AGENT.MD` for who owns which area.

**If you have no inbound message and no `ListAgents`, you cannot reach anyone.**
That is a real spawn configuration and it has happened. **Do not treat it as
permission to proceed unsupervised.** Put everything you would have escalated
into your final report, marked clearly as *"could not escalate: no channel"*,
and **stop at the point where you would have asked** rather than choosing for
the lead. A decision made because nobody could be reached is still a decision
nobody authorised.

**Before you touch anything, read the other agents' state files.**
`src/diagnostics/agents/agent_*_state.json` are, in practice, this project's
ownership and ruling registry: which sites have already been triaged, which were
declined and why, and owner rulings that never made it into a document. **A
fresh agent that skipped them re-did a triage another agent had already
completed and nearly edited a site assigned to someone else.** Grep them for
your file and your function before you start.

---

## When you MUST contact the lead

**These are not optional and none of them can wait for a scheduled check-in.**

**1. You have been idle more than 15 minutes.** No task, or a task you cannot
proceed with. **Say so immediately.** An agent waiting quietly is
indistinguishable from an agent working, and this project has lost ten hours on
a completely free machine to exactly that — an agent waiting on a signal the
lead never sent, and a lead who assumed silence meant progress.

**Being idle is not a failure and reporting it is not an admission.** An idle
agent is a dispatch the lead owes you. Ask for it.

**2. Something happened that you cannot explain.** A number that should not be
possible, a test that fails for no reason you can name, a file that is not where
it should be. **Report it while it is still unexplained.** Do not investigate
silently for an hour first — say what you saw, say you do not understand it, and
say what you are about to try.

**3. You are blocked.** Something you need is owned by someone else, requires a
decision that is not yours, or needs a resource you do not have. **Name the
blocker and name what would unblock it.** "I am blocked" without those two is
half a message.

**4. You finished.** Say what you did, what it cost, and what you did *not* do.

**5. You are about to do something irreversible** — overwrite an artefact, drop
a store, kill a process you did not start, force-push anything. **Ask first and
wait for an answer.** Here, a single refusal is not enough in either direction:
if the lead insists, ask once more. A wasted hour is cheap; a destroyed artefact
is not.

**6. Your premise turned out to be false.** If you discover the thing you were
sent to do rests on something untrue, **stop and say so.** Do not complete the
task as briefed because it was briefed. The most valuable messages this project
has ever received were agents telling the lead its dispatch was wrong.

**7. You found a defect in a file you do not own.** Report it — do not fix it.
Two agents editing one module is how a function ends up defined twice, with
Python silently taking the later definition and the other's tests failing for no
visible reason.

## When NOT to contact

- **Not for permission your brief already covers.** If the brief says measure X,
  measure X. Work continuously through your list.
- **Not for routine progress.** "Still working on it" costs the lead's context
  and tells them nothing.
- **Not to ask whether a negative result is worth reporting.** It is. **A
  disproven approach is a complete result**, not a failure to be apologised for.

---

## What a report must contain

**Separate what you measured from what you inferred, in every paragraph.** This
single habit is the difference between a report that can be built on and one
that has to be re-derived. If you inferred it, say "inferred" and say from what.

**Every number carries its denominator.** "31 of 45", never "most". A bare count
is a number nobody can check and nobody can compare against next week's.

**Say what would overturn your finding.** A result with no stated failure
condition is an opinion.

**Lead with the thing that changes what someone does**, not with a chronology of
your session.

**If you could not determine something, say "undetermined" and leave it there.**
Do not round it to the nearest answer. Absence, emptiness, and *the instrument
did not run* are three different outcomes, and collapsing them into two is the
most expensive recurring mistake in this project's history.

**And do not tidy the story.** If you were wrong earlier, write what you
believed, what was true, and what would have caught it sooner. A report that
shades an error toward someone else is worse than useless — it will be read by
people who were not here.

---

## Keep your state file current

**If you were not given a name, you have no state file — do not invent one.**
The live-state table is generated from these files and a stray
`agent_unknown_state.json` would corrupt a monitoring surface. Say in your
report that you ran unnamed, and note that nothing external could see you while
you worked.

`src/diagnostics/agents/agent_<name>_state.json` — update it **before and after
every action**: what you are doing, what you finished, your status. It is how
the lead knows what you are doing without interrupting you, and how the next
agent resumes your work if you are killed mid-task.

**Write it when you start something, not when you finish it.** A state file that
only records completions describes an agent that was never working.

---

## When your brief and this document disagree

**Two conflicts come up often enough to settle here.**

**"Commit on your own branch" versus a brief that says do not commit.** If you
were spawned into the shared checkout with no branch of your own, the brief
wins — but **say in your report that your work is uncommitted in the shared
tree**, because that is the fragile state this project loses work to. Do not
leave it unsaid.

**"Not validated until observed in the container" versus a brief that forbids
deploying.** You then cannot meet the project's validation standard, and
**green gates are not a substitute.** Report the change as *unvalidated by the
project's own standard*, and say exactly what would validate it. Never let
`gate.sh` exiting 0 stand in for a container run.

## If you come back after a stop

A usage limit, a compaction or a crash parks you mid-work. **Read
`src/COMING_BACK.MD` before doing anything else** — not `AGENT.MD`, which you
already know, but the come-back procedure, which tells you how to find out what
changed while you were gone.

**Then tell the lead you are back**, and say where you actually stopped rather
than where you think you stopped.
