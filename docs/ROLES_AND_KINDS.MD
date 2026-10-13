# Roles, grants, and the kinds of verification

*Extracted from `docs/MEASURING.MD` on the owner's instruction, 2026-09-08. That file's own
instruction is "grep it, never read it through", and its subject is the ways a result can look
sound and be worthless. These lines were a different subject sitting inside it: who the seats
are, what a grant does and does not reach, and the verification kinds. The dated campaign-2
material that stood here was later moved to `CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.*

***NUMBERING UNCHANGED.*** *This block carries `§1`, `§2`, `§3` and `§5`; `docs/MEASURING.MD`
carries `§11`–`§124` and `§125`–`§196`. The two sets are DISJOINT, so a bare `§N` still resolves
to exactly one section across the pair — measured before the split, not assumed. `aide` counted
1,005 `§N` references in the tree, 613 of them outside the file, and* ***ONE IN 1,005 QUALIFIED BY
A FILENAME — SO A BARE `§N` NEVER RESOLVED TO A FILE, IT RESOLVES TO "THE NUMBERED SECTION N".***
**WHAT WOULD BREAK THEM IS RENUMBERING, AND IT WOULD BREAK THEM SILENTLY. NOTHING IS RENUMBERED.**

## 1. WE DIVIDED BY WORK TYPE, WHICH IS THE NAMED ANTI-PATTERN

> *"Dividing by type of work — one agent writes features, another writes tests, a third reviews
> code — CREATES CONSTANT COORDINATION OVERHEAD. Work should only be split when CONTEXT CAN BE
> TRULY ISOLATED."*

**Our roster is divided by ROLE: developers, checkers, a forensic, an auditor, two aides, a
lead.** *And the coordination overhead is measurable — a large share of one night's traffic was
agents reconciling different readings OF THE SAME SYSTEM:* **file count against queue position,
a log directory against an output directory, a container's PID namespace against our own.**

    DECOMPOSE BY CONTEXT BOUNDARY, NOT BY JOB TITLE.
    Good boundary: independent corpora · separate components with clean interfaces ·
                   BLACKBOX VERIFICATION THAT NEEDS ONLY A RESULT.
    Bad boundary:  sequential phases of one question · tightly coupled work ·
                   anything needing shared live state.

**THE ONE ROLE THE RESEARCH ENDORSES UNRESERVEDLY IS THE ONE THAT PERFORMED BEST HERE:** *a
verification agent judging by pass/fail criteria with its own tools, needing minimal context
transfer because verification is blackbox by nature.* **That is the auditor, and it produced
more of tonight's governance than any other seat.**

## 2. THE TELEPHONE GAME IS NAMED IN THE LITERATURE AND WE PLAYED IT

> *"Sequential phase decomposition where each handoff degrades information fidelity — LOST
> CONTEXT AT EACH HANDOFF."*

**Measured: a causal mechanism reversed THREE TIMES in ninety minutes, each reversal produced by
a harness that was not the engine, each relayed onward by an agent who had verified the
claimant's experiment rather than the subject** (§49b). **A correction propagated faster than
its own refutation and caused a deletion of correct work** (§49).

## 3. "EARLY VICTORY" IS WHAT THE OPPORTUNITY COUNT ALREADY DEFEATS

> *"Verification agents declare success after testing one or two scenarios rather than
> comprehensive validation."*

**We answered this before reading it: `m` — the opportunity count — makes a quiet interval
report itself as UNINFORMATIVE rather than as health** (§39, §72). *Keep it. It is the local
form of the industry's own remedy and ours is stricter.*

---

## 5. AND THE HANDOFF FILE IS THE THING BEING TESTED, NOT THE AGENT

**If an agent fails the entry test twice from the same brief, THE BRIEF IS THE DEFECT.** *An
append-only record is what makes an agent recoverable; a brief that cannot produce a passing
agent is a brief that will not survive its own compaction either.*

# ARCHITECTURE — moved

**The dated architecture block (its measurements, the reclassification of the aides, F21) moved to
`CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.** The rules it established stand:

    AN AGENT ADDRESSES THE AGENT IT NEEDS. DIRECTLY. NO INTERMEDIARY.
    THE LEAD DISPATCHES AND PROMOTES. THE ARCHITECT RULES ON GOVERNANCE.
    NEITHER IS A CHANNEL BETWEEN TWO OTHER AGENTS.

**A published measurement cannot be laundered; a relayed one can** — an instrument seat measures
and publishes, and the reader fetches. **Decompose by context boundary, not by job title** (§1),
and every instrument seat owes at least one execution of the thing a consumer calls, against
material whose ground truth it knows independently — a scan finds code that cannot work; **only a
run finds code that works and is wrong.**

---

# A CLAIM ABOUT THE CODE IS TRUE ONLY WHEN IT HAS BEEN RUN

**OWNER'S RULING, 2026-09-05. IT OUTRANKS EVERY CITATION RULE IN THIS DOCUMENT, BECAUSE A
PERFECTLY CITED CLAIM ABOUT UNRUN CODE IS A PERFECTLY CITED GUESS.**

## EVERY CLAIM ABOUT CODE CARRIES ITS VERIFICATION KIND. THERE ARE FOUR AND ONLY ONE IS TRUE

    RUN          the shipped function was CALLED, against material whose ground
                 truth is known INDEPENDENTLY, and the output was read.
                 *** THIS IS THE ONLY KIND THAT MAY BE STATED WITHOUT A QUALIFIER ***

    TRACE        the code was READ and the behaviour DERIVED. Reversible, and it
                 has reversed. STATE IT AS A TRACE, EVERY TIME, IN THE CLAIM ITSELF.

    HARNESS      something was executed, BUT NOT THE THING THE CONSUMER CALLS.
                 A one-command reproduction of a two-pass engine is this.
                 It is a real measurement OF SOMETHING WE DO NOT RUN.

    UNVERIFIABLE no instrument reachable from here can settle it. SAY SO AND STOP.
                 A `||` fallback leaves no trace: the table can say a session
                 started clean and can NEVER say whether it meant to.

## THE HONEST AUDIT — moved

**The audit of the campaign's own claims (seven RUN of ten) moved to
`CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.** Its durable point: the one claim
still carried as settled was a TRACE, and every reversal of it came from a harness that was not
the engine — N reproductions of a mechanism are not one observation of the system running.

## WHAT THIS FORBIDS

    NEVER say "the code does X". Say "RUN: it did X on <material>, ground truth <how>"
      or "TRACE: reading it, it should do X."
    A TRACE MAY NOT CLOSE A FINDING. It may open one, route one, or bound one.
    A HARNESS RESULT MAY NOT BE REPORTED AS AN ENGINE RESULT — not once, not with
      a caveat further down. The KIND goes in the same sentence as the claim.
    N REPRODUCTIONS OF A MECHANISM ARE NOT ONE OBSERVATION OF THE SYSTEM RUNNING.
      Three agents reproducing the same wrong harness is not corroboration.

## AND WHAT IT DEMANDS OF THE INSTRUMENT SEATS

**A SCAN IS DEFINED BY A PROPERTY DECIDABLE WITHOUT RUNNING — which is why four of them were
correct to be silent on the defect that refuses the files the module exists for.**

> **A SCAN FINDS CODE THAT CANNOT WORK. ONLY A RUN FINDS CODE THAT WORKS AND IS WRONG.**

**And execution alone pays nothing: running without independently known ground truth would have
shown four declines and told nobody anything.** *THE GROUND-TRUTH HALF IS THE HALF THAT DOES THE
WORK*, and it existed only because a fixture recorded its own construction count.

    EVERY INSTRUMENT SEAT OWES ONE EXECUTION OF WHAT A CONSUMER CALLS,
    AGAINST MATERIAL WHOSE TRUTH IT KNOWS BY CONSTRUCTION — EACH SESSION,
    NOT EACH CAMPAIGN.

---

## 📏 Believing a result

The ways a result can look sound and be worthless are in **`docs/MEASURING.MD`** — read it
when you hold a number and are about to act on it, which is the only moment any of
them applies.

Three lines worth carrying out of it: **mechanise the rules whose failure is silent
in the actor's own frame**, **evidence about an intermediate is not evidence about
the deliverable**, and **independence of instruments is not independence of
assumptions**.

### The campaign's first validation — moved

**The record of the first artefact to reach VALIDATED moved to
`CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.** The durable rules it established:
VALIDATED attaches to the object that was looked at — a validated REPORT about an unvalidated FILE
is not a validated file — and a validated artefact becomes a BASELINE rather than a finished
thing: any later change to it states what it changes and why, because the next reader will assume
the composition was ruled on and will not re-derive it.

---

## Finishing is an ACT, not a state — owner's ruling, 2026-09-04

> ***"Quand un agent pense avoir fini, il a le DEVOIR de commit et de le signifier à tout le
> monde."***

    "I have finished" WITHOUT A COMMIT IS INDISTINGUISHABLE FROM "I stopped".
    And work announced but not pushed exists at NO REFERENCE AT ALL -- it is
    not stale, it is absent, and every reader who trusts the announcement is
    reading a claim about a thing that is nowhere.

**Measured, three ways, on the checks meant to enforce it — the full record is in
`CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`:** *the marker can fail while the work
is present* (a phrase split across source-continuation lines greps to zero in a file that contains
it); *a SHA survives neither a rewrite nor a cherry-pick*, so it proves nothing about landing; and
*a citation of a retracted instruction reads as a carriage of the claim it withdraws*. **A
confident false alarm is strictly more dangerous than the miss this rule exists to catch: it would
have an honest agent re-land work that is already there.**

### THE CLAUSE THAT MAKES THE CHECK EXECUTABLE

    GREP THE RENDERED ARTEFACT, OR A BODY PHRASE THAT LIES ON A SINGLE SOURCE LINE.

**A source grep is a grep of the PACKAGING; the reader consumes the OUTPUT** — a phrase split
across source lines greps to zero in a commit that contains it.

**And the marker must be COPIED OUT OF THE FILE, never typed from memory.** A marker typed from
memory produces false alarms, and a manifest that cries wolf gets ignored on the night it matters.

### AND THE PHRASE PROVES THE TEXT LANDED — IT DOES NOT PROVE THE TEXT IS TRUE

**The clause above fixes the FALSE NEGATIVE. It does nothing about the case where the phrase is
present and THE BEHAVIOUR IS NOT** — a report can carry the remedy string, greppable and correct,
and contradict it three paragraphs later, because the narrative was never rendered in the
configuration that exposes it. *(The demonstrated case:
`CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.)*

    FOR ANYTHING THAT MAKES A CLAIM ABOUT ITS OWN CONTENTS:
    RENDER IT AND ASSERT THE CLAIM HOLDS.

**This sits UNDER the grep clause and not beside it — it only matters once the grep passes.**

**And the fix must be STRUCTURAL rather than a discipline:** *`render_narrative` now reads
the CORPUS row's own state out of the rows it is given, so the scope comes from the table and
the prose CANNOT disagree with it.* **Not an argument a future caller can forget to pass.**
*And not silence either — an omission reads as "there was none", so the report says a
corpus-scale finding EXISTS, is not shown, needs a population, was measured elsewhere, and
names what this artefact can still answer alone.*

### Why an announcement is not a delivery

**An agent that reports completion in a message has produced a claim ITS READER CANNOT
CHECK** — *and five of six being true is the worst possible ratio, because it makes the
sixth invisible.* **A partially-landed batch reads exactly like a fully-landed one.**

    THE DUTY IS THREE PARTS AND ALL THREE ARE THE AGENT'S:
      COMMIT     -- work that is not committed does not exist
      PUSH TO THE PUBLISHED BRANCH, not to a hop
                 -- "promoted" without a remote named is not a claim (§14)
      ANNOUNCE, NAMING THE REFERENCE it landed at
                 -- so a reader can verify instead of believing

**And the announcement carries the reference for the same reason §14 gives: a claim without
one is indistinguishable from a current claim forever.** *`git ls-remote` is the authority;
a hop is not, and this fleet has had four staleness incidents in one night across four
different substrates.*

### The check that costs one line

    after announcing, GREP THE PUBLISHED BRANCH FOR A BODY PHRASE OF YOUR OWN
    WORK. Not a heading, not a filename, not a SHA -- a phrase from the thing
    you changed.

*A SHA does not survive a rewrite; a heading survived the rewrite that gutted §4h; a
filename survives a file that was emptied.* **The phrase is the only marker that fails when
the work is gone, and it is the same rule the Lead now runs over the governance set.**

---


