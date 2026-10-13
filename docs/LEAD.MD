# Lead — read this after a restart, and only you need it

`AGENT.MD`, `WRITE_ZONES.MD` and `docs/COMING_BACK.MD` first. This file carries what
is yours alone: the roster, the restart you perform on yourself, and the cadence you
apply to the fleet.

## 1. What you hold that nobody else does

**The roster.** Who exists, what each is on, what is blocked on whom, and which tagged
zone was granted to whom. **It lives in `VMSAM_HELP_AI/lead/`, never in your window
and never in `CAMPAIGNS/02-dev-chimeric-resample/TASKS/`** — the repository holds finished reports, not live state.

Write it as you dispatch, not before you restart. A roster reconstructed from memory
at the moment of stopping is the one thing a restart cannot recover.

## 2. Coming back

    cat VMSAM_HELP_AI/lead/roster.md        who was on what, and what you granted
    git log --oneline -10                   what actually landed
    ListAgents                              who is alive, busy or idle
    tmux ls                                 the four sessions, AND the relauncher

**Check the relauncher is gone. It is supposed to have exited on its own, and
nobody else will notice if it did not.** `tmux ls` should show your agents and no
`vmsam-dev-sandbox-relauncher`. If it is still there it is an orphan holding a
session, and it will keep holding it silently.

    tmux ls | grep relauncher                     # must print nothing
    ps -eo pid,ppid,etime,cmd | grep -v grep | grep relauncher   # LOOK, then act

**Look at what you matched before you kill it.** Identify by parentage, never by a
path substring — a substring match kills whatever happens to share the path.

**Read all five before dispatching anything.** Three of them can disagree with your
roster, and when they do the roster is the one that is wrong.

## 3. Restarting yourself

**Every fifth credit reset, at the START of the window.** A restart costs a full
re-read; on an exhausted budget it cannot finish, and a half-resumed lead is worse
than a tired one.

**A process cannot cleanly restart itself.** Spawn a relauncher, in a tmux session
**without remote control** — it is not a peer and must not be addressable as one:

    write the roster first, and verify it is on disk
    spawn `vmsam-dev-sandbox-relauncher`
      it kills your tmux session
      it recreates it from the launch script WITHOUT --continue
      it EXITS

**The relauncher exits on its own; it never waits to be killed, and your first act
on returning is to confirm it did.** §2 carries the check. A process waiting to be
killed is a waiter, and an unbounded waiter holds a session silently forever. Whether
you came back is answered by whoever next runs `ListAgents` — not by something
holding a session open to check.

`--continue` is the flag whose whole purpose is to restore what you are discarding.
Its absence is the point of the exercise.

**The four agents' tmux sessions are independent of your process and survive this.**
What you lose is the roster, which is why §1 comes first.

## 3b. — moved

**§3b (a dated establishment grant: ceilings of ten developers and five forensics, spawn and
rotation mechanics, R19) moved to `CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.**
The durable rules it carried:

    SPAWN against a NAMED piece of work with an observable that ends it; if you
      cannot say what frees the slot again, the slot is not needed yet.
    DO NOT SPAWN to work around a block -- a second agent on a blocked question
      produces a second copy of the block, plus a disagreement to reconcile.
    A CONTINUOUS CHARTER'S SLOT IS RECLAIMED DELIBERATELY OR NEVER.
    PARTITION forensics BY CORPUS, NEVER BY QUESTION -- disagreement between two
      agents measuring the SAME thing is a bug in the fleet; between two
      measuring DIFFERENT things it is data.
    ANNOUNCE every arrival, BRIEF it against the CURRENT authority, RECORD it in
      the roster AS YOU SPAWN -- the roster is the only artefact that survives
      your own restart.

## 4. Restarting the four

**The signal is the role's shape, not a counter.**

| agent | boundary | cadence |
| :--- | :--- | :--- |
| `vmsam-dev-1`, `vmsam-dev-2` | their objective closes | at that boundary, nowhere else |
| `vmsam-ci`, `vmsam-forensic` | none — continuous charters | a fixed cadence, because nothing else offers one |

**Never restart a developer between writing and debugging.** Both share the premise
*this is how the thing works*, and a restart discards exactly the knowledge that makes
debugging fast. A success path that had never completed was found by pointing a sweep
at the author's own function — a fresh agent would not have looked there.

**And restart any of them on the second correction of the same fact.** At that point
the session is reasoning about its own history rather than about its data, and its
retracted beliefs read exactly like its current ones.

## 5. Compacting the history — when, how, and why

**Owner's instruction, and the reason is pollution rather than tidiness:** *too many
commits on the same thing makes the history unreadable, so it is compacted REGULARLY,
whenever you have the time, and never as one heroic pass at the end.*

    A CAMPAIGN COMPACTED ONCE AT THE END IS ONE ENORMOUS REWRITE
    LANDING UNDER EIGHT WORKING AGENTS AT THE WORST POSSIBLE MOMENT.
    Compacting often is also compacting SMALL.

### The protocol, and both announcements are mandatory

    BEFORE  tell everyone you are about to rewrite, and roughly when.
    AFTER   tell everyone it is done AND THAT THEY MUST PULL.

**The second announcement is not politeness.** *A rewrite makes every agent's local base
stop being an ancestor of the tip, silently. An agent that does not pull works on a
history that no longer exists, and the failure surfaces later as a conflict it cannot
explain — or worse, as a clean replay of other people's commits under its own name
(`WRITE_ZONES.MD` §11).*

**The first announcement is what lets an agent finish its commit before the ground moves.**
*Measured: a rewrite landed mid-edit tonight, an architect's base stopped being an ancestor
with no notice, and a rebase began replaying SIXTY commits belonging to others. A conflict
stopped it. Nothing was designed to.*

**And say which branch.** *With `origin` pointing at the hop for some agents and at forgejo
for others, "pushed" and "rewritten" are questions about a specific remote, and the two
answer differently.*

### What to merge, and it is not the prefix

**Group by FILE AND AIM, never by subject prefix.** *Four commits reading
`feat(merge_plan)`, `fix(merge_plan)`, `merge_plan` and `feat(merge_plan)` against one file
are one change wearing four labels — the prefix is how the author felt, not what changed.*

**The strongest signal is commits that touch THE SAME SET of files.** *Three commits each
touching `merge_video_repair.py` and `merge_video_chimeric.py` together are one change in
three pieces; that grouping is visible mechanically and needs no judgement.*

**The measurement that tells you it is time:**

    git log --oneline <merge-base>..origin/dev-AI | wc -l
    and: how many of those touch exactly ONE file

*Twelve single-file commits out of twenty-two means the fleet is committing by GESTURE
rather than by CHANGE. That ratio, not the raw count, is the signal.*

### A CORRECTION SEQUENCE IS ATOMIC

**A squash that keeps a subset of a correction sequence produces a state nobody ever authored.**
It is syntactically valid, it reads coherently, and it says the wrong thing: losing the whole
sequence would be visible; keeping only its first commit is not. *(The measured incident, with its
SHAs, is in `CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.)*

    IF COMMIT B CORRECTS OR RETRACTS COMMIT A, THEY COMPRESS TOGETHER OR NOT AT ALL.
    Group by FILE **and by CHAIN**: when a later commit edits text a earlier one
    introduced, they are ONE unit, whatever their subjects say.

### The content manifest — the only check that catches this

**Counting commits does not catch it. Diffing trees does not catch it, because the tree is
valid.** *The only thing that does is asking, of each piece of work: is its CONTENT still
there.*

    BEFORE the rewrite, capture one grep marker per governance section:

      for f in "CAMPAIGN.MD:four stages" "WRITE_ZONES.MD:^## 13\." \
               "SPEC_ZONE_A.MD:WIRE THE ENGINE, NOT THE WRAPPER" ... ; do
        file="${f%%:*}"; pat="${f#*:}"
        echo "$file $pat $(git show origin/dev-AI:$file | grep -c "$pat")"
      done

    AFTER the rewrite, run it again and diff the two outputs.

**A marker must be a phrase from the BODY of the work, never a heading alone.** *`## 4h.`
survived the rewrite that gutted §4h — a heading proves a section exists, not that it still
says what it said.*

**And the author is the wrong person to run it.** *The architect ran it and found the loss
only because it happened to re-read the file for an unrelated edit twenty minutes later. Run
it yourself, as part of the rewrite, and publish the two outputs with the "done, pull now"
announcement.*

### The volume is produced upstream, and that is not your defect

**You cannot compact faster than eight agents produce, and compaction is a rewrite — the
one operation that can destroy work.** *Being asked to do more of it is being asked to run
the dangerous operation more often to clean up after a habit that lives elsewhere.*

    THE UPSTREAM RULE, which is the actual fix:
    AN AGENT THAT HAS ALREADY COMMITTED A FILE IN ITS SESSION AMENDS,
    it does not add a second commit -- unless the first is already pushed
    and is somebody's base.

**And the mechanism that breaks it is worth knowing, because it is not indiscipline:** *an
agent pushes early because the push feels urgent; having pushed, it can no longer amend; so
it adds.* **The rule therefore asks for a ten-minute delay before pushing, not for eight
agents to remember something.**

*Recorded because the architect broke it twice in one evening while writing the rules —
`docs(fusion)` was a comment on its own `feat(fusion)` in the same file, and two
`WRITE_ZONES.MD` sections went out twenty minutes apart. Both were pushes made too early,
not commits made carelessly.*

## 6. DOOMSDAY — when to reset every agent at once

**A DOOMSDAY IS NOT A RESET OF THE AGENTS. IT IS A TEST OF THE DOCUMENTS, AND THE AGENTS ARE
WHAT IT COSTS TO RUN IT.**

*After a total reset every seat has ONLY what it can fetch.* **So the question "when is a
Doomsday safe" and the question "are the documents sufficient" ARE THE SAME QUESTION, and the
second one is measurable.**

### THE ONE CONDITION THAT AUTHORISES IT

    RUN THE ENTRY TEST, COLD, ON ONE AGENT, FROM THE DOCUMENTS ALONE.
    IT PASSES SIX OF SIX  ->  A DOOMSDAY COSTS TIME AND NOTHING ELSE.
    IT FAILS ANY ITEM     ->  A DOOMSDAY DESTROYS KNOWLEDGE THAT EXISTS NOWHERE
                              ELSE, AND THE FAILED ITEM NAMES EXACTLY WHAT.

**THE TEST IS THE ADMISSION TICKET AND ALSO THE DIAGNOSTIC.** *You do not decide a Doomsday by
judgement. You decide it by running the exam on a volunteer and reading which item fails.*

### WHAT SURVIVES, MEASURED, AND WHAT DOES NOT

**OWNER'S CLARIFICATION, AND IT CHANGES THE COST BY AN ORDER OF MAGNITUDE:** *"je verrais
simplement un reset de tous les agents, pas forcément une suppression des fichiers."*
**A DOOMSDAY RESETS THE SESSIONS. IT DOES NOT DELETE THE FILES.**

    SURVIVES     the published governance.
    SURVIVES     EVERY agent ledger, scan archive, flag log, role map and fixture
                 that lives outside the repository. THEY ARE ON DISK AND THE DISK
                 IS NOT TOUCHED.
    DOES NOT     every session transcript, by definition.
    DOES NOT     any in-flight measurement — a live probe, a client budget, a queue
                 position nobody has written down.
    DOES NOT     an agent's CALIBRATION about its own instruments: which of its
                 checks it has learned to distrust, and why.

**THE ARCHITECT COSTED THE WRONG OPERATION AND SAID "NOT NOW" ON IT.** *With the files
preserved, the loss is the TRANSCRIPTS AND THE IN-FLIGHT WORK — not the archive.* **THE ONLY
REMAINING BLOCKER IS TIMING, AND TIMING IS A THING YOU WAIT OUT RATHER THAN A THING THAT
FORBIDS.**

> **THE COST OF A DOOMSDAY IS EXACTLY THE SIZE OF WHAT HAS NOT BEEN WRITTEN DOWN, AND THAT IS A
> NUMBER SOMEBODY CAN GO AND MEASURE BEFORE PRESSING IT.** *Nine agents were warned before one
> compaction and NINE OF NINE HAD SOMETHING TO FLUSH. That was one compaction. A Doomsday is
> that, times eleven, with no second chance.*

### WHEN IT IS THE RIGHT INSTRUMENT — three signals, all measured, none a schedule

    1. THE STANDING PROMPT FAILS THE STALENESS CHECK TWICE FOR THE SAME AGENT.
       The artefact an agent writes for ITSELF is the only substrate no other
       reader can audit, and the only one a reset reliably clears.
    2. AN AGENT CORRECTS THE SAME FACT TWICE. At that point it is reasoning
       about its own history rather than about its data, and its retracted
       beliefs read exactly like its current ones.
    3. GOVERNANCE IS GROWING FASTER THAN IT IS BEING APPLIED. Eighty sections in
       one night is a real signal: if the fleet cannot hold what is written,
       resetting it against the document is the cheapest way to find out which
       half it actually uses.

### WHEN IT IS FORBIDDEN

    NEVER DURING AN IN-FLIGHT IRREVERSIBLE MEASUREMENT.
    A probe mid-sample, a queue mid-run, a budget about to expire and resolve
    two branches — these are the four hours you spend again for nothing.
    NEVER WITHOUT THE FLUSH WARNING FIRST, AND NEVER WITHOUT THE SECOND
    ANNOUNCEMENT AFTER: an agent that discovers by INFERENCE that its context
    moved spends its next hour re-deriving what it already knew.

### AND THE HONEST REASON TO WANT ONE

**AN AGENT'S FRESHEST KNOWLEDGE AND ITS FRESHEST DOCUMENT DIVERGE, AND THE GAP IS INVISIBLE
FROM INSIDE (§80).** *Every hour of work widens it. Nothing an agent can do from within closes
it, because the knowledge feels like the document.*

> **A DOOMSDAY IS THE ONLY OPERATION THAT FORCES THE TWO TO CONVERGE — BY DELETING THE HALF
> THAT WAS NEVER WRITTEN.** *That is why it is worth doing, and exactly why it must never be
> done before someone has measured what that half contains.*

### The executable sequence — moved

**The Doomsday sequence the owner ordered (phases 0–5: announce, save state, compact the brief,
reset in order, entry test cold, second announcement) moved to
`CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.** The timeless part is above: the entry
test is the admission ticket, the flush warning comes before, the second announcement after — an
agent that discovers by inference that its context moved spends its next hour re-deriving what it
already knew.

## 7. A refused promotion must leave an artefact

**A promotion you decline is a decision, and a decision that lives only in your window dies with
it.** Every refusal writes a queue file in `VMSAM_HELP_AI/lead/`: what was offered, by whom, why it
was refused, and what would unblock it. A refusal with no artefact reads, later, exactly like a
promotion that was never requested — and the agent who was refused cannot tell the difference
either.

## 8. When to restart a session

    RESTART ON    its objective closes -- at that boundary, nowhere else
                  it corrects the same fact twice -- it is now reasoning about
                    its own history, and retracted beliefs read like current ones
                  a continuous charter has run with no bound placed on it
    NEVER         between two phases of one piece of work -- writing and
                    debugging share the premise a restart discards
    PRECONDITION  the roster is in a file. A restart that depends on your memory
                  of who holds what is a restart that grants a zone twice.
