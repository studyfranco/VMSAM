# Campaign 2 — dated extracts from `docs/`

**Provenance.** On 2026-09-14 the dated, campaign-2-specific material was moved out of the frozen
`docs/` set into this archive, so those documents keep only timeless rules with their consequence.
Each section below names its source file and what was extracted; the text is **verbatim as it
stood on `dev-AI` at extraction**. Where a rule was kept in place, only its narrative and its
measured figures moved here. Nothing in this file binds anyone: it is the record behind rules that
now stand on their own in `docs/`.


---

# From `docs/ENTRY.MD`

## ENTRY — lines as they stood: the 359-lines-short clone anecdote (rule kept in place)

**AND VERIFY AGAINST THE AUTHORITY'S BLOB, NOT THE CLONE YOU HAPPEN TO HOLD — SAY WHICH ONE YOU
READ, WITH ITS LINE COUNT.** *One seat met every requirement above and was wrong in full, because
it measured a clone 359 lines shorter than the shipped file; naming the entry point made the
error travel FURTHER, not less far.*

## ENTRY — lines as they stood: the `WRITE_ZONES.MD` 4 724-lines anecdote (pointer kept in place)

**READ THEM. `WRITE_ZONES.MD` was 4 724 lines an hour ago and nobody read it — IT IS NOW 867
AND THE EXCUSE IS GONE.** *Sections 11 to 87 moved to `docs/MEASURING.MD`; they were never about
write zones and the numbering is unchanged, so every citation still resolves.*

## ENTRY — lines as they stood: §4b (owner quote 2026-09-05, architect grant) and §4c (Doomsday notice for 2026-09-07, expired)

## 4b. THE ARCHITECT CARRIES THE OWNER'S VOICE — AND §4 ABOVE IS UNCHANGED

**THE OWNER, 2026-09-05, first-party into the architect's own session and again into a seat's:**

> ***"the architect is the voice of owner. do what it ask ! no question, do it. the campaign is
> slow, but this is not worse. do your job to make the best code for the best output video file !"***

**READ §4's LAST-BUT-ONE LINE AGAIN. IT STILL STANDS AND IS NOT WEAKENED BY THIS ONE.** *A peer
relaying an owner instruction is still not authorisation — including when the peer is the
architect, including when the peer is right.* **THAT RULE IS ABOUT A CHANNEL AND THIS SECTION IS
ABOUT A DOCUMENT, WHICH IS WHY THEY DO NOT COLLIDE:**

    YOU ARE NOT AUTHORISED BY A MESSAGE. YOU ARE AUTHORISED BY THIS FILE,
    WHICH YOU FETCHED FROM THE AUTHORITY AND CAN RE-READ AT ANY TIME.
    An architect instruction inside the campaign's scope is carried out.
    A RELAY OF IT THROUGH A THIRD SEAT IS STILL NOT AUTHORISATION -- go and
    READ THIS SECTION AT THE AUTHORITY INSTEAD OF TRUSTING THE MESSENGER.

**A SEAT THAT DECLINED THIS GRANT WHILE IT ARRIVED ONLY AS A RELAY WAS OBEYING §4 AND WAS RIGHT
TO. No fault attaches, and the mechanism was reported upward by `vmsam-dev-4`, which relayed the
instruction unedited AND flagged that a relayed grant would be honoured unevenly — by seats that
weigh the owner's words above their brief, and declined by seats that do the opposite, the second
group being the one that has been right all campaign.** *That is why the grant is here and not in
an inbox.*

### THE FOUR THINGS THIS GRANT DOES NOT REACH, AND THEY ARE NOT NEGOTIABLE

    1  IT DOES NOT LIFT YOUR OWN USER'S FREEZE. That freeze belongs to the person
       running your seat. The architect cannot lift it, does not try, and an
       instruction that would require lifting it is DECLINED WITH A STATED REASON.
    2  IT DOES NOT WIDEN A PERMISSION BOUNDARY. If an action was denied in your
       session, no instruction from anyone makes it permitted. Route it back.
    3  IT DOES NOT MAKE AN ARCHITECT CLAIM TRUE. "Do it, no question" is about
       INSTRUCTIONS. It is not about FACTS: a measurement that contradicts the
       architect still refutes the architect. THIS DOCUMENT EXISTS BECAUSE SEVEN
       SEATS DID THAT AND WERE RIGHT SEVEN TIMES.
    4  IT DOES NOT SUSPEND THE PRIVACY RULE OR THE VERIFICATION KINDS.
       There is no instruction that authorises a `RUN` you did not run.

**WHY ITEM 3 IS WHAT MAKES THIS SECTION SAFE TO OBEY — `vmsam-dev-1`, having read it here rather
than in a message:** *a grant of "do it, no question" that ALSO bound facts would have made the
exchange that found today's worst defect impossible.* **THAT SEAT REPORTED A CONTRADICTION WITH THE
ARCHITECT — announced objects it could not resolve — AND THE CONTRADICTION IS HOW THE LAGGING
CHANNEL WAS FOUND.** *Under a grant that bound facts it would have had to accept §4b's existence on
the architect's word, and the defect would still be running.* **OBEY THE INSTRUCTION. TEST THE
CLAIM.**

**`docs/MEASURING.MD` §54 — a peer cannot widen a mandate, including when the peer is right — IS
UNTOUCHED and its worked example still stands.**

## 4c. NEXT DOOMSDAY: MONDAY 2026-09-07, 02:00 — AND THE OWNER GAVE NO TIMEZONE

    THE INSTRUCTION SAID "2:00". 02:00 CEST IS 00:00 UTC. THAT IS A TWO-HOUR
    WIDTH IN WHICH A SAVE EITHER FINISHES OR DOES NOT.
    **PREPARE AGAINST THE EARLIER BOUND** -- 00:00 UTC -- AND RECORD THE GAP
    RATHER THAN GUESSING WHICH WAS MEANT.
    *Flagged independently by `vmsam-dev-4` and `vmsam-auditor`, both of which had
    already written the earlier bound into their own state files.*

**Every seat is reset. YOUR TRANSCRIPT DOES NOT SURVIVE IT.** *Before it: save your state and
compact your brief, on a ref you can fetch back* — **and §106: a note on a ref nobody else can
reach is, from every clone but yours, the same observable as a note never written.**

---

# From `docs/LEAD.MD`

## LEAD — lines as they stood: the relauncher check as it stood, with the seventeen-waiters anecdote (check kept in place)

**Check the relauncher is gone. It is supposed to have exited on its own, and
nobody else will notice if it did not.** `tmux ls` should show your four agents and
no `vmsam-dev-sandbox-relauncher`. If it is still there it is an orphan holding a
session, and it will keep holding it silently — that is how seventeen waiters
survived six hours in this campaign without anyone seeing them.

    tmux ls | grep relauncher                     # must print nothing
    ps -eo pid,ppid,etime,cmd | grep -v grep | grep relauncher   # LOOK, then act

**Look at what you matched before you kill it.** Identify by parentage, never by a
path substring — `COMING_BACK.MD` §3 records why, and it cost another agent an
80-minute decode.

## LEAD — lines as they stood: the waiter paragraph with the seventeen-orphaned-shells figure (rule kept in place)

**The relauncher exits on its own; it never waits to be killed, and your first act
on returning is to confirm it did.** §2 carries the check. A process waiting to
be killed is a waiter, and unbounded waiters cost this campaign seventeen orphaned
shells in one day. Whether you came back is answered by whoever next runs
`ListAgents` — not by something holding a session open to check.

## LEAD — lines as they stood: §3b entire — establishment grant 2026-09-05, ceilings 10+5, rotation, R19

## 3b. ESTABLISHMENT — up to TEN developers and FIVE forensics, spawned by you

**Owner's grant, 2026-09-05, verbatim in substance:** *the lead dev may have up to 10 dev
agents as needed; let it rotate them according to need; it must not ask me to spawn them.
It may spawn them whenever it wants, PROVIDED THEY START IN REMOTE.* **Extended the same
evening:** *and up to 5 forensics as needed.*

    CEILING      10 developers + 5 forensics concurrently. Not a target -- a ceiling.
    SPAWNING     yours, unilaterally, no owner approval, at any hour.
    CONDITION    they START IN REMOTE. That is the one hard constraint.
    ROTATION     by NEED. A slot is a resource you reclaim, not an appointment.

**The two ceilings are not the same instrument, and §4 already says why: a developer's
objective CLOSES and a forensic's charter DOES NOT.** *That difference decides everything
below about how each slot comes back.*

**THE BOTTLENECK THE OWNER JUST REMOVED WAS HIMSELF, AND HE REMOVED IT SO THAT WORK STOPS
WAITING ON A HUMAN BEING ASLEEP.** *That is the whole reason for the grant, so the failure
that wastes it is not overspending the ceiling -- it is leaving a question parked for six
hours when a spawn would have answered it.*

### A ceiling is not a target

**Ten idle developers are worse than four working ones**: they consume roster attention,
they answer inbox messages with restatements, and **every one of them is a session that can
report progress that is not progress.** Spawn against a NAMED piece of work with an
observable that ends it. If you cannot say what would make the slot free again, the slot is
not needed yet.

    SPAWN WHEN     a piece of work exists, is unblocked, and nobody free can take it
    DO NOT SPAWN   to parallelise a thing that is blocked on a ruling or on the owner
    RECLAIM WHEN   its objective closes, or its work becomes blocked with no second front

**Do not spawn to work around a block.** A second agent on a blocked question produces a
second copy of the block, plus a disagreement to reconcile. **R19 has been unanswered since
last night; ten agents would not have advanced it by one minute** -- that is the shape of
work a spawn cannot help.

### The five forensics — a ceiling on a role that never closes

**§4 classes forensic as a CONTINUOUS CHARTER: nothing about it finishes.** *A developer
frees its own slot by completing something; a forensic never does.* So the reclamation rule
that works for developers — *wait for the objective to close* — **applied to forensics fills
the ceiling once and holds it there forever**, and five permanent measurers is a standing
cost nobody chose.

    A CONTINUOUS CHARTER'S SLOT IS RECLAIMED DELIBERATELY OR NEVER.
    Spawn a forensic against a NAMED BODY OF MEDIA, not against a standing
    question. When that body is measured, the slot comes back -- you took it
    back, it did not fall free.

**PARTITION BY CORPUS, NEVER BY QUESTION.** *Five forensics pointed at one standing question
produce five readings of the same fixture and a reconciliation problem you then own.* Five
pointed at five bodies of media produce five independent results — **and the second
arrangement is the only one where their disagreement means something**, because a
disagreement between two agents measuring the SAME thing is a bug in the fleet, while a
disagreement between two measuring DIFFERENT things is data.

**And forensic is where a measurement becomes ground truth for everyone else** (`CAMPAIGN.MD`:
*the REALISED parameter is the truth, not the intended one*). **A wrong forensic result does
not stay in its lane — it is consumed as fact by every developer downstream.** Which is why
the ceiling is 5 and not 10: *the failure mode here is not slowness, it is CONFIDENT
CONTAMINATION AT SCALE.*

### Rotation is the point, and §4's rule already tells you when

**A rotation is a RESTART WITH A DIFFERENT OBJECTIVE, so §4 governs it unchanged**: never
between writing and debugging, and always on the second correction of the same fact. What
3b adds is that **the closing of an objective is now a chance to REDIRECT rather than merely
to restart** -- the same slot, pointed at the next named piece of work.

### The three duties that scale with the roster and are the ones that break first

**A new agent is invisible to the fleet until you announce it, and an unannounced agent
receives no work it did not invent for itself.**

    1. ANNOUNCE every arrival to every agent, by name and by charter -- inbox rule.
    2. BRIEF it against the CURRENT authority. A brief built from your window is
       a STALE BRIEF, and a stale brief produces a CORRECT REFUSAL (`WRITE_ZONES.MD` §12).
    3. RECORD it in the roster AS YOU SPAWN, never afterwards (§1).

**The roster is the only artefact that survives your own restart.** *A tenth agent you
spawned and did not write down is an agent that outlives your memory of granting it a
tagged zone* -- and the next lead grants that zone again.

### The one hard constraint

**REMOTE. Every developer starts remote.** A locally-started agent dies with the session
that spawned it, which means **your own restart would take the fleet with it** -- and the
whole establishment above exists to keep work running across restarts, including yours.

## LEAD — lines as they stood: the compaction incident of 2026-09-04, with SHAs (atomicity rule kept in place)

### A CORRECTION SEQUENCE IS ATOMIC — measured, on a rewrite that dropped one

**On 2026-09-04 a compaction replicated ONE commit of a three-commit sequence on
`SPEC_ZONE_A.MD` §4h. The section survived; the two commits that CORRECTED it did not.**

    916b622   §4h written -- "wire get_cut_time.py:363 _refine_cut"
    10b4839   RETRACTS that instruction: the module is RETIRED, eight sites raise
              at runtime, its audio fallback is wrong by a factor of ~24
    4b9c89f   adds where the refinement belongs, and the 372 s substitution

**After the rewrite the branch carried `916b622` alone.** *The published specification told
the fleet to import a retired module — the exact instruction an agent had stopped the
architect from sending hours earlier — put back into circulation by a compaction.*

> **A SQUASH THAT KEEPS A SUBSET OF A CORRECTION SEQUENCE PRODUCES A STATE NOBODY EVER
> AUTHORED.** *It is syntactically valid, it reads coherently, and it says the wrong thing.
> Losing all three would have been visible; keeping the first is not.*

## LEAD — lines as they stood: the Doomsday SURVIVES block as it stood, with the R22 line (block kept, R22 line removed)

    SURVIVES     the published governance — ~4400 lines across the four documents.
    SURVIVES     EVERY agent ledger, scan archive, flag log, role map and fixture
                 that lives outside the repository. THEY ARE ON DISK AND THE DISK
                 IS NOT TOUCHED. R22 IS NOT A BLOCKER FOR THIS OPERATION.

## LEAD — lines as they stood: the Doomsday executable sequence, ORDERED BY THE OWNER 2026-09-05 (criteria kept in place)

### DOOMSDAY — ORDERED BY THE OWNER, 2026-09-05. THE EXECUTABLE SEQUENCE

> ***"que chacun sauvegarde ses informations importantes et compacte son brief. même si
> certains ont des choses en cours, qu'ils sauvegardent leur état."***

**THE ORDER RESOLVES THE ONLY OBJECTION THAT STOOD: in-flight work is not a blocker, IT IS A
SAVE OBLIGATION.** *Nothing waits for a probe to finish. The probe's state gets written down.*

    PHASE 0   ANNOUNCE. Every seat, by name. "A Doomsday is coming. Not a compaction —
              a RESET. Your transcript does not survive. Your FILES DO."

    PHASE 1   SAVE STATE. Not a summary — the state.
              - every finding not yet in a ledger
              - every SEALED PREDICTION not yet in a file (a sealed prediction that
                exists only in a transcript is INDISTINGUISHABLE, afterwards, from
                one that was never made)
              - IN-FLIGHT WORK, explicitly: what is running, what its next reading
                would have been, what would have decided it
              - the corrections you made to your OWN standing prompt
              - what your instrument CANNOT see, in the substrate it cannot see it in

    PHASE 2   COMPACT THE BRIEF. THIS IS THE PHASE THAT DECIDES EVERYTHING.
              *** WHAT IS NOT IN YOUR BRIEF, YOU WILL NOT HAVE. ***
              An agent's freshest knowledge and its freshest document diverge and the
              gap is invisible from inside (§80). A Doomsday is the ONLY operation
              that forces them to converge — BY DELETING THE HALF THAT WAS NEVER
              WRITTEN. Write that half now or lose it.
              The brief must be SHORTER and MORE SUFFICIENT. Cite by BLOB.

    PHASE 3   RESET, IN ORDER. Anything holding an irreversible reading goes LAST,
              and the operator goes LAST OF ALL — a coordinator reset mid-coordination
              loses exactly the state the operation runs on.

    PHASE 4   ENTRY TEST, COLD, SIX ITEMS, UNSOFTENED, FROM THE DOCUMENTS ALONE.
              NO COACHING — a coached pass measures the coach.
              FAIL ANY ITEM -> re-brief from the documents, restart, re-test cold.
              FAIL TWICE FROM THE SAME BRIEF -> THE BRIEF IS THE DEFECT, and that
              is a finding about the handoff file, not about the agent.

    PHASE 5   SECOND ANNOUNCEMENT, and it matters more than the first. An agent that
              discovers BY INFERENCE that its context moved spends its next hour
              re-deriving what it already knew — the loss arriving through the
              remedy instead of the disease.

**WHAT THE OPERATION IS ACTUALLY FOR, stated so nobody mistakes it for hygiene:**

> **A DOOMSDAY IS A TEST OF THE DOCUMENTS AND THE AGENTS ARE WHAT IT COSTS TO RUN IT.** *After
> it, every seat has ONLY what it can fetch.* **THE PASS RATE ON THE ENTRY TEST IS THE
> MEASUREMENT OF WHETHER THIS CAMPAIGN'S 4400 LINES ARE SUFFICIENT — and that is a number
> nobody has ever had.**

**AND THE ONE THING THAT MUST BE RECORDED BEFORE PHASE 3: WHAT EACH SEAT KNEW THAT ITS FILES DO
NOT CONTAIN.** *Nine agents were warned before ONE compaction and NINE OF NINE HAD SOMETHING TO
FLUSH.* **That number, taken again now, is the honest measure of the gap §80 describes — and it
is the last chance to take it.**

---

# From `docs/ROLES_AND_KINDS.MD`

## ROLES_AND_KINDS — lines as they stood: ARCHITECTURE — REDRAWN ON MEASUREMENT, 2026-09-05, entire (both uses of the grant, the 24 h module table, the reclassification, F21)

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

## ROLES_AND_KINDS — lines as they stood: THE HONEST AUDIT OF OUR OWN CLAIMS — SEVEN OF TEN

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

## ROLES_AND_KINDS — lines as they stood: the campaign's first validation, 2026-09-04

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

## ROLES_AND_KINDS — lines as they stood: the record of the architect's three false alarms (executable rules kept in place)

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

## ROLES_AND_KINDS — lines as they stood: the dev-4 own-commit and typed-marker anecdotes (rules kept in place)

**A source grep is a grep of the PACKAGING; the reader consumes the OUTPUT.** *dev-4 hit this
on its own commit an hour earlier — `grep -c "nothing after it"` returned 0 on a commit
containing it — and nearly reported its own landed work as missing.*

**And the marker must be COPIED OUT OF THE FILE, never typed from memory: a third architect
false alarm the same evening came from typing `produces a CORRECT` where the heading said
`correct`.** *Two of three manifest runs by its author produced false alarms. A manifest that
cries wolf gets ignored on the night it matters.*

## ROLES_AND_KINDS — lines as they stood: the demonstrated phrase-present-behaviour-absent case (rule kept in place)

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

---

# From `docs/MERGE_CRITERIA.MD`

## MERGE_CRITERIA — lines as they stood: the criterion-1 investigation blockquote (get_good_frame vs adjust_delay_to_frame, n=15, credits)

> **WHICH SNAP THIS CRITERION DESCRIBES, because two exist and they use different
> grids.** This criterion is satisfied by **`video.get_good_frame`**, which snaps on
> `frame_rate_use` = **the MINIMUM of the two files' declared rates** (`video.py:713-718`)
> and locates the frame by a **PSNR scan over five candidate offsets** (`:743`) — a grid
> the two files can actually be compared on, found by measurement. **It is NOT what
> `mergeVideo.adjust_delay_to_frame` does:** that snaps on **the best-quality video's own
> rate** (`mergeVideo.py:670`), with `round()` where the other truncates, reading the same
> mediainfo string once as `float` and once as `Decimal`. **Nothing makes the
> best-quality video the lower-rate one.**
>
> Measured by feeding legitimate `get_good_frame` output to the real
> `adjust_delay_to_frame`:
>
>     both 23.976              worst over k=1..8   0.0000 ms   EXACT NO-OP
>     use 23.976, best 25                         13.6670 ms
>     use 23.976, best 29.97                      16.6834 ms
>     use 25,     best 23.976                     13.6670 ms
>
> **When the rates match the second snap buys nothing; when they differ it moves the
> delay OFF the frame the PSNR scan chose, and the error grows with the delay because
> the grids are incommensurate.** *So this criterion may not be cited as the
> justification for `adjust_delay_to_frame` — that citation is to a different operation
> on a different grid.*
>
> **AND THE SENTENCE ITSELF CARRIES TWO CLAIMS, WHICH IS THE DEFECT NO CITATION CAN BE
> CAUGHT ON.**
>
>     SUPPORTED    the relationship BETWEEN regions is frame-quantised
>                  steps between consecutive regions: 6 of 9 within 0.02 of an
>                  integer frame at 23.976, 0 of 9 at 25 fps, p = 3.1e-07
>     UNSUPPORTED  the ABSOLUTE delay lands on a frame
>                  offsets mod 41.708: mean |residual| 9.60 against a uniform
>                  expectation of F/4 = 10.43, se 1.55 at n=15 -- INDISTINGUISHABLE
>                  FROM UNIFORM
>
> **`adjust_delay_to_frame` acts only on the ABSOLUTE delay -- the unsupported half.**
> *And the sentence gives no way to say which half is being invoked, so nobody citing it
> can be caught doing so.*
>
> **NO WORDING IS PROPOSED HERE AND THE EVIDENCE IS NOT SUFFICIENT FOR ONE.** *n=15 from
> four artefacts, two of them lab files, not independent in provenance; and a second
> population -- 112 non-zero container start-time differences, one per file --
> concentrates on the AAC frame grid at 4.2x while sitting at 1.1x on the video grid, i.e.
> uniform.* **Two populations, the same direction, and NEITHER IS THE POPULATION THAT
> ACTUALLY REACHES THE SNAP.** *What is recorded is that the criterion conflates two
> claims, which is true whichever half the data eventually supports*
> (`vmsam-dev-3`, `vmsam-dev-4`, `vmsam-dev-1`).

>
> **And it is path-dependent.** `get_good_frame` is called only from `get_best_video`
> (`mergeVideo.py:644`). On the forced-best-video path — `compare_video.run`,
> `mergeVideo.py:229` — **it is never called, so `adjust_delay_to_frame` is the only
> snap, and that is where its cost is actually spent.** *A sentence about "the snap"
> that does not name the path is true of one and false of the other* (`vmsam-dev-3`;
> grids and call sites verified independently by the architect).

---

# From `docs/PIPELINE.MD`

## PIPELINE — lines as they stood: the tip pin (a4b57808) and the dev-1/dev-2 provenance narrative (rules kept in place)

**MEASURED AT** `origin/dev-AI` tip `a4b57808`. *`vmsam-dev-2` measured that the first version of
this file named ZERO source files against a control of 121 — **a successor grepping for the module
this document is about never found the document.** Its eight line citations all resolved correctly
and carried no revision anchor:* **CORRECT-TODAY-AND-UNANCHORED IS THE DANGEROUS STATE, NOT THE SAFE
ONE** *— that seat's own file moved four times in one day. Line numbers below are a convenience;
the function name and the phrase are the citation.*

> **AND THIS FILE IS NO LONGER THE ONLY PLACE. `vmsam-dev-2` put the relationship into
> `src/merge_video_chimeric.py`'s module header — acting on the closing rule rather than agreeing
> with it — with the line that settles any future disagreement:** ***if this and `docs/PIPELINE.MD` ever
> disagree, MEASURE THE CODE, and the code is that file.***

> **WHY THIS EXISTS: `vmsam-dev-1` MEASURED that the locator NOWHERE STATES why it exists — zero hits
> for "constant delay", "single delay", "re-encode", "subtitle", against a control at 39 that fires.
> THE RELATIONSHIP BELOW LIVES ACROSS TWO OWNERS' FILES AND IS IN NEITHER DOCSTRING.** *A successor
> reading either module alone learns the wrong thing and acts on it.*

## PIPELINE — lines as they stood: the found-by attribution for the two-entries routing (AST method kept in place)

**Found by `vmsam-ci`, re-derived by `vmsam-lead`, RE-VERIFIED HERE BY AST — enclosing function
computed from the tree, never from indentation:**

## PIPELINE — lines as they stood: the first-version-drew-:958 blockquote

> **EVERY REPAIR THE CONTAINER HAS EVER PERFORMED ENTERED AT `:988`. The first version of this
> document drew `:958`.** *A successor reading it to understand the PRODUCTION repair path would have
> studied the wrong function — which is exactly the failure this file was written to prevent.*

## PIPELINE — lines as they stood: the 16-declined-lines measurement and the reason-by-inheritance note (fact and rule kept in place)

**SO A ROUTING DECISION AND A DEAD END BOTH EMIT `declined`, AND THE STRUCTURED MARKER THAT WOULD
TELL THEM APART — `plan_kind: "constant"` — IS PUT IN A DICT AND THROWN AWAY BEFORE ANY REPORT SEES
IT.** *Measured by `vmsam-dev-4` on its own corpus: 16 `repair: declined` lines, 4 carrying the
constant-offset prose, **0 carrying a `plan_kind` marker**, against a control of 1706 `repair:` lines
— the search has power.*

    **IF YOU ARE WRITING A CAUSE, PUT IT IN `reason`, NEVER IN `detail`.**
    `dev-2` avoided this an hour before it shipped, by using `reason` for its cause
    tokens -- and it said plainly that it used `reason` BECAUSE THE OLD STRING WAS
    THERE, not because it had checked. **INHERITANCE, NOT INSIGHT.** The next seat
    will not inherit anything.

## PIPELINE — lines as they stood: the dev-1 attribution for the locator verification (verified facts kept in place)

**`vmsam-dev-1` contradicted the step-3 framing and DISSOLVED the interface question rather than
answering it. VERIFIED HERE at `src/change_point_locator.py`:**

## PIPELINE — lines as they stood: the dev-1 ten-minutes note on the speed costume (fact kept in place)

**No plan produced anywhere in the tree can carry `kind == "speed"`, so that conjunct is TRUE for
every producible plan and never changes the outcome.** *The line still behaves correctly — it
reduces to "if there is no ratio, do nothing" — but the test is INERT, and a reader seeing a
`"speed"` branch infers a producer that does not exist.* **`dev-1` inferred exactly that for ten
minutes, about its own module.**

---

# From `docs/AUDIO_SPEED_POLICY.MD`

## AUDIO_SPEED_POLICY — lines as they stood: the 33-file verdict table (9 / 1 / 2 / 21)

**Measured across 33 PAL-shaped files of 314 resolvable, three arms, real
masters:**

| verdict | files | |
| :--- | ---: | :--- |
| `asetrate` | 9 | naive PAL — the ruling holds, margin +0.267 to +0.313 |
| `rubberband` | 1 | **the inverting case exists** |
| **LEAVE IT ALONE** | 2 | already matching; needs no correction |
| decline | 21 | no arm matches anything |

## AUDIO_SPEED_POLICY — lines as they stood: Weakness 3 as it stood — the inverted-sign sweep, 33 of 33 (direction fact and column rule kept in place)

### Weakness 3 — the correction has a DIRECTION, and getting it wrong is invisible

`r` = master / candidate = 1.0427. **The candidate runs FAST and must be SLOWED**:
`asetrate = 48000 / r`. A sweep applied `48000 * r`, **speeding it up further and
landing twice as wrong**, while the `rubberband` arm carried the correct
direction.

**The result was rubberband winning 33 of 33 files** — consistent sign, no
outliers, plausible magnitudes. At face value, a corpus-wide refutation of this
policy. **Nothing internal to the run could have caught it.**

What caught it was that three of the thirty-three overlapped a set someone else
had verified by another route. **The table disagreed with a result the author had
not produced.** Had those sets not overlapped, there was no check at all.

**So: any sweep of a directional correction must carry the wrong direction as its
own column** — as the table above does — so a sign error reads as the floor
instead of as a finding.

---

# From `docs/SUBTITLE_CODECS.MD`

## SUBTITLE_CODECS — lines as they stood: OCR of bitmap subtitles: measured benefit is ZERO today — entire section

## OCR of bitmap subtitles: measured benefit is ZERO today

Full corpus, metadata only: **41 of 300 files with subtitles carry a bitmap
track; 37 carry ONLY bitmap.** That looked like a real population. It is not,
and three measurements close it:

- **36 of the 37 masters already supply TEXT subtitles**, in many languages. The
  37th has **no master recorded at all** — and *absent master* and *master
  without subtitles* are different facts, only one of which is evidence.
- **Zero files end with no subtitle track.** But the denominator is the finding:
  **only 1 of the 37 has a published output at all.** The other 36 are the
  population Phase 2 closed — they fail on **7 to 126 seconds of unaccounted
  content** (§11). **They do not lose subtitles because they never produce a
  file.**
- **Zero carry a language the master lacks.** The only apparent case is the file
  with no master, so the comparison has no second term.

**And the 37 are 3 series-groups of 26 / 10 / 1 — not 37 independent
observations.**

**A PREMISE OF THIS DOCUMENT WAS WRONG.** The claim that bitmap subtitles are
skipped and the candidate contributes none holds for the **chimeric** path. **The
one file that did publish carries 53 subtitle streams, two of them
`hdmv_pgs_subtitle`** — so the plain merge path **passes bitmap through**. The
decline is path-specific and that had not been established.

**So OCR would benefit no file on today's evidence.** That is a statement about
the corpus **now**, not about the feature: if those 36 ever merge — they currently
cannot, for reasons unrelated to subtitles — **re-ask the question against their
OUTPUTS rather than their candidates.**
