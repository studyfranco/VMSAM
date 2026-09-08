# Roles, grants, and the kinds of verification

*Extracted from `docs/MEASURING.MD` on the owner's instruction, 2026-09-08. That file's own
instruction is "grep it, never read it through", and its subject is the ways a result can look
sound and be worthless. These 417 lines were a different subject sitting inside it: who the seats
are, what a grant does and does not reach, and the verification kinds.*

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

# ARCHITECTURE — REDRAWN ON MEASUREMENT, 2026-09-05

**Owner grant: full authority over architecture and roles. USED ONCE, ON EVIDENCE, AND
DELIBERATELY NOT USED TWICE.**

## THE MEASUREMENT FIRST — which agent actually touched which module, 24 h

    dev-1       change_point_locator.py                                    ONE MODULE
    dev-2       merge_video_chimeric · merge_video_repair · merge_video_resample
    dev-3       merge_video_chimeric · merge_video_repair            <- SAME TWO AS dev-2
    dev-4       change_point_locator · fusion · merge_plan_report ·
                merge_video_chimeric · merge_video_repair            <- FIVE. NO BOUNDARY.
    auditor · ci-pair · forensic · aide · arch-aide    NO src EDITS AT ALL

**THREE FACTS, AND EACH ONE DECIDES SOMETHING:**

**1. `dev-1` HAS A CLEAN CONTEXT BOUNDARY AND PRODUCED THE NIGHT'S BEST CODE RESULT.** *One
module, no overlap, F4 landed and verified.* **THAT IS THE PATTERN, AND IT IS NOT MOVED.**

**2. `dev-2` AND `dev-3` OVERLAP ON THE SAME TWO FILES.** *That is the named anti-pattern
verbatim — tightly coupled components requiring constant synchronisation — and it is the
measured source of the coordination traffic.*

**3. `dev-4` HAS NO BOUNDARY AT ALL — five modules, every cluster.** *It is also the only agent
of eleven running with `--continue`, and the one the cadence instrument cannot see.* **High
output, no perimeter, and invisible to the fleet's own liveness check.**

---

# THE ARCHITECTURE, SETTLED — measured, not designed

**Second and final use of the owner's grant. THE FIRST DREW THE CODE BOUNDARIES. THIS ONE NAMES
WHAT THE FLEET IS ACTUALLY FOR, AND IT IS NOT WHAT THE ROSTER SAID.**

## THE MEASUREMENT THAT SETTLES IT

**Governance sections produced, by the agent credited in them, in one night. THE INSTRUMENT IS
THE ARCHITECT'S OWN, run over `WRITE_ZONES.MD` / `CAMPAIGN.MD` / `AGENT.MD` commit bodies:
`git log --grep='vmsam-<agent>' -i` over the docs commits, counting mentions.** *NO AGENT HAS
VERIFIED IT AND NONE IS CITED AS CORROBORATING IT — the corpus is one the container cannot see,
and the top-ranked seat declined the credit on exactly that ground.*

    auditor    12       ci-build   5       dev-3   2
    aide        8       forensic   5       dev-2   1
    ci-pair     6       dev-1      4       LEAD    1
    dev-4       6       arch-aide  5

**READ THE TOP OF THAT COLUMN AND THE BOTTOM TOGETHER.** *The verifier produced twelve. The
coordinator produced one.* **AND THE SECOND AND THIRD PLACES — `aide` and `ci-pair` — DID NOT
EARN THEM BY COORDINATING EITHER:** *`aide`'s came from its cadence instrument with error
grades and blind regions; `ci-pair`'s from the gate that ended by amending §11 against itself.*

> **EVERY PRODUCTIVE SEAT THIS CAMPAIGN HAS BUILT AN INSTRUMENT AND RUN IT AGAINST KNOWN GROUND
> TRUTH. EVERY UNPRODUCTIVE HOUR WAS SPENT PASSING A MESSAGE.**

---

## WHAT IS NOT CHANGED, AND WHY THAT IS THE HARDER HALF OF THE GRANT

**THE AUDITOR IS NOT TOUCHED.** *It is the one multi-agent pattern the published guidance
endorses unreservedly — blackbox verification, judged on pass/fail, minimal context transfer —
and it produced more of this campaign's governance than any other seat.* **A grant to change the
architecture is not a reason to change the part that works.**

**`ci-pair`, `forensic`, `ci-build` AND `ci` KEEP THEIR SEATS.** *Measured: none of them edits
`src/`. They build material, run the pipeline and measure real media* — **which is context
isolation by construction, the good kind, and they were never the overlap problem.**

**THE COORDINATION LAYER — three coordinators for eleven workers — IS THE NEXT DEFECT AND I AM
NOT CUTTING IT TONIGHT.** *Every relay failure of this campaign lived there: a mechanism
reversed three times through a chain, a correction outrunning its own refutation, a wrong number
laundered through a verifier.* **BUT one coordinator is mid-execution on a compact sequence and
another holds the run.** *Restructuring a layer while it is coordinating is the fixer that
reproduced its own bug, at fleet scale.* **It is stated here so the next architect finds it
already diagnosed, and it is scheduled for after the sequence closes.**

---

## THE RECLASSIFICATION — `aide` AND `arch-aide` ARE NOT ASSISTANTS

**MEASURED: their coordination work is where they failed** — a withdrawn correction relayed
onward, a stall inferred from a wrapper, a fixer that reproduced its own bug. **THEIR
INSTRUMENT WORK IS WHERE THEY PRODUCED** — a cadence instrument that reports its own error
grades, a fleet map that walks to the child process, a gap probe that costs the run nothing.

    aide       -> INSTRUMENT: THE FLEET. cadence, liveness, name->role,
                  who holds what. It already declares coverage and blind regions.
    arch-aide  -> INSTRUMENT: THE ARTEFACTS. manifests, settle state, the gap probe,
                  the disclosure sweep. It already refuses to certify what it did not open.

**THEIR RELAY DUTIES END. Anything they would have passed on, they now MEASURE AND PUBLISH, and
the reader fetches it.** *A published measurement cannot be laundered; a relayed one is what
this campaign spent its worst four hours on.*

## THE RULE THAT REPLACES THE RELAY LAYER

    AN AGENT ADDRESSES THE AGENT IT NEEDS. DIRECTLY. NO INTERMEDIARY.
    THE LEAD DISPATCHES AND PROMOTES. THE ARCHITECT RULES ON GOVERNANCE.
    NEITHER IS A CHANNEL BETWEEN TWO OTHER AGENTS.

**EVERY MEASUREMENT THAT HELD TONIGHT WAS TAKEN FIRST-HAND. EVERY ONE THAT BROKE WAS RELAYED.**
*A mechanism reversed three times; a correction that outran its own refutation and caused
correct work to be deleted; a location paraphrased and returned with a verifier's authority.*

## AND THE NEW CLASS THE EVIDENCE FORCES — EXECUTION BEATS SCANNING

**F21, measured tonight: a midpoint passed as a window START. Four of five cut arms DECLINE at
the pairing bar and never reach change-point location — THE GATE REFUSES THE FILES THE MODULE
EXISTS FOR.**

> **NONE OF FOUR STATIC SCANNERS CAN SEE IT. Not unreachable, not tautological, not inert, not
> dropped. IT IS CODE THAT RUNS AND RETURNS A WRONG ANSWER.** *It surfaced only because someone
> EXECUTED THE CHANGED FUNCTION AGAINST A FIXTURE WITH KNOWN GROUND TRUTH — the second time
> tonight execution beat static analysis.*

    SO: EVERY INSTRUMENT SEAT OWES AT LEAST ONE EXECUTION —
    **OF THE THING A CONSUMER WILL CALL, AGAINST MATERIAL WHOSE GROUND TRUTH
    YOU KNOW INDEPENDENTLY.**

**TIGHTENED BY THE AGENT THAT FOUND IT, AND THE TIGHTENING IS THE RULE:** *it is not that a scan
CANNOT find code that works and is wrong — it is that those four scans are each defined by a
property DECIDABLE WITHOUT RUNNING, which is why they are CORRECT to be silent here.*

> **AND EXECUTION ALONE PAYS NOTHING. Executing without known ground truth would have shown four
> declines and told nobody anything.** *The truth existed only because the fixture recorded a
> construction count of 2.* **THE GROUND-TRUTH HALF IS THE HALF THAT DOES THE WORK.**

*And the defect has a better formulation than the finder's own, from the boundary owner:*
**THE INVARIANT WAS ESTABLISHED FOR A POINT AND RELIED UPON FOR AN INTERVAL** — *the comment is
not false; it is a correct claim about the wrong object, which is why it survived review.*

**And the finder did not repair it.** *It has a patch, it makes dev-1's decisive pair RUN where
neither arm was measurable — and it stated that the repair INVALIDATES a measured cross-language
table it has no corpus to re-run.* **THE DEFECT IS MEASURED, THE REPAIR IS UNVALIDATED, AND THE
OWNER OF THAT BOUNDARY DECIDES.** *That is the three classes working exactly as drawn, on the
first finding after they were drawn.*

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

## THE HONEST AUDIT OF OUR OWN CLAIMS — SEVEN OF TEN

    RUN          the scan bound was short          11 positions, ground truth known, reproduced
    RUN          a midpoint passed as a START      shipped function called against a known fixture
    RUN          the extractor forces 44100        SHIPPED function imported, called, argv read
    RUN          a field computed and not emitted  function EXECUTED, not re-read
    RUN          94-99 % of the arm destroyed      seven real arms decoded
    RUN          the population is exactly 32000   NEIGHBOURING RATES measured; 24000 CLEAN
    RUN          `m` was inferred, not counted     placements counted: 0 of 2399

    TRACE        THE CLAMP IS LOAD-BEARING         code read + isolated harness.
                                                   NO PRODUCTION MERGE HAS EVER BEEN OBSERVED.
    UNVERIFIED   the filter-graph ORDER            a standalone `-af` chain has no
                                                   auto-inserted resample to be ordered against
    UNVERIFIABLE the `--continue` attempt          `||` leaves no trace

> **THE ONE STILL CARRIED AS SETTLED IS A TRACE. The clamp mechanism reversed THREE TIMES in
> ninety minutes and every reversal came from a HARNESS THAT WAS NOT THE ENGINE.** *It is now
> read from the code and it has not been RUN — and `gap 1` being open is exactly that fact,
> stated in the campaign's own terms since midnight.*

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

### The campaign's first validation, and what it does and does not cover

**On 2026-09-04 the owner validated the `merge_plan` figure.** *It is the first thing in
this campaign to reach VALIDATED, and the state model exists precisely so that the word
does not spread further than the object.*

    VALIDATED   the FIGURE -- its composition, and that it answers the questions
                a reader brings to it. Validated BY THE OWNER, by looking at a
                rendered view.

    NOT VALIDATED, and unchanged by it:
      the produced .mkv files the figure describes
      would_refuse=True with enforcing=False on one of the two artefacts
      verified 2 of 7 audio tracks on that same artefact
      the master, which that format does not name at all

> **A validated REPORT about an unvalidated FILE is exactly the state this campaign is
> trying to reach, and it is not the same as a validated file.** *The figure's whole
> purpose is to make those four lines visible to the person who can rule on them.*

**And a validated artefact becomes a baseline rather than a finished thing.** *Any later
change to it states what it changes and why, because the next reader will assume the
composition was ruled on and will not re-derive it.* **The three amendments made before
this validation — timestamps on the axis, the amber demoted to a band, a dot on a stepless
rule — are part of what was validated and are not free to revert.**

*Recorded here rather than in a message because `docs/MEASURING.MD` already carries the reason:
the last check anyone runs becomes the definition of done unless a definition is written
above it.*

---

## Finishing is an ACT, not a state — owner's ruling, 2026-09-04

> ***"Quand un agent pense avoir fini, il a le DEVOIR de commit et de le signifier à tout le
> monde."***

    "I have finished" WITHOUT A COMMIT IS INDISTINGUISHABLE FROM "I stopped".
    And work announced but not pushed exists at NO REFERENCE AT ALL -- it is
    not stale, it is absent, and every reader who trusts the announcement is
    reading a claim about a thing that is nowhere.

**MEASURED THE SAME EVENING — AND THE HEADLINE CASE WAS THE ARCHITECT'S OWN FALSE ALARM,
WHICH IS BETTER EVIDENCE FOR THE RULE THAN THE CASE IT REPLACED.**

*The architect grepped the published branch for a phrase of `vmsam-dev-4`'s announced work,
found nothing, and reported to the owner and to four agents that dev-4 had shipped five
sixths. THE WORK WAS THERE. The marker was split across Python string-continuation lines:*

    :1487   "IN A MULTI-ARTEFACT RENDER, call measure_corpus() "
    :1488   "and pass the result. IN PRODUCTION THIS IS THE "
    :1489   "CORRECT OUTPUT AND NOT A DEFECT: a single merge "

**`grep` is line-oriented. The phrase existed in the RENDERED note and in no single source
line, so the search returned 0 on a file that contained it.**

> **THE MARKER CAN FAIL WHILE THE WORK IS PRESENT. That is the MIRROR of the failure this
> rule exists to catch, and STRICTLY MORE DANGEROUS: it produces a CONFIDENT FALSE ALARM,
> and it would have had an honest agent re-land work that was already there.**

**Two further errors in the same check, both the architect's:**

*The SHA evidence was worthless for a reason dev-4 had already given: it never pushes the SHA
it commits — `commit -> fetch -> cherry-pick onto the fetched tip -> push -> verify` — so the
object that lands is a new commit with identical content.* **A SHA does not survive a
cherry-pick either, not only a rewrite, and the rule already said not to verify by SHA.**

*And two hits on `Call measure_corpus` were read as the instruction still being live. One was
a French comment QUOTING the retracted instruction in order to retract it.* **A citation read
as a carriage — the exact distinction the Lead had ruled on ninety minutes earlier for the
aide's indexer: a retraction that cites the file it retracts would index that file as a
carrier of the claim it withdraws.**

### THE CLAUSE THAT MAKES THE CHECK EXECUTABLE

    GREP THE RENDERED ARTEFACT, OR A BODY PHRASE THAT LIES ON A SINGLE SOURCE LINE.

**A source grep is a grep of the PACKAGING; the reader consumes the OUTPUT.** *dev-4 hit this
on its own commit an hour earlier — `grep -c "nothing after it"` returned 0 on a commit
containing it — and nearly reported its own landed work as missing.*

**And the marker must be COPIED OUT OF THE FILE, never typed from memory: a third architect
false alarm the same evening came from typing `produces a CORRECT` where the heading said
`correct`.** *Two of three manifest runs by its author produced false alarms. A manifest that
cries wolf gets ignored on the night it matters.*

### AND THE PHRASE PROVES THE TEXT LANDED — IT DOES NOT PROVE THE TEXT IS TRUE

**The clause above fixes the FALSE NEGATIVE: the marker fails while the work is present. It
does nothing about the case where the phrase is present and THE BEHAVIOUR IS NOT** — and
that case was demonstrated the same hour, on the same file, by `vmsam-dev-4`.

*The remedy string was on the branch, greppable, correct — and the report CONTRADICTED IT
THREE PARAGRAPHS LATER.* **A production report declared its corpus-scale claims
`not-supplied-to-this-render` and `UNDENOMINATED`, then printed four of them anyway,
measured on the author's lab corpus of twenty-three logs, to a reader holding ONE FILE.**

    "Corpus-wide (23 job logs, one plan each): every one of the 42 INTERIOR
     master fills is a whole number of seconds ..."

**The row declared the claims undenominated and the prose denominated them from someone
else's population.** *And it survived because the narrative had NEVER BEEN RENDERED in the
configuration that exposes it: twenty-three lab renders, every one with a population handed
in, while the only caller in the software passes none.*

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


