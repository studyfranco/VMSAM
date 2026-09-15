# DOCKET — campaign 2, the single source of open work

Maintained by the Lead Dev (`BRIEF.md` §1). **If it is not here, it is not being
worked on.** Ticket bodies live in `CAMPAIGNS/02-dev-chimeric-resample/TASKS/`;
measured findings live in that directory's `FINDINGS.MD`. **This file carries
state, owner and blocker only** — never a copy of a number, because a number
copied into a table reproduces itself without anyone re-running the command that
made it.

Privacy: opaque error ids only, never a filename, title or path.

Baseline measured 2026-09-15 on `523ec97e`:

    python3 VMSAM_HELP_AI/tools/check_write_zones.py --self-test   -> exit 0, 35/35
    python3 -m compileall -q src -x '__pycache__'                  -> exit 0

The checker was fired in both directions before being trusted: an untracked file
under `src/gestionar_show/` and an appended line in `src/audioCorrelation.py` each
produced **exit 1** with the matching `§1` message, and the tree verified clean
after both probes were reverted.

## 1. Stage map (`CAMPAIGN.MD` §5)

| stage | what | carrier | state |
|---|---|---|---|
| 0 | reread TASKS/005-010 against `CAMPAIGN.MD` §2 | Lead + Architect | **read done 2026-09-15, two independent passes** — nine findings, `FINDINGS.MD` §"Stage 0". Verdicts: **005·006·007 CONTRADICT**, **008·009·010 INCOMPLETE**. Rewrites not done: delegated to the Architect |
| 1 | frame-accuracy blockers: the four `ARCH_FRAME_ACCURATE.MD` defects + the three locator defects of TASKS/013 | split, see §2 | **in progress** — `vmsam-dev-frame` on the four; the three locator defects unassigned |
| 2 | corpus ground truth in master frame numbers | TASKS/014 | open, unassigned; part 1 runnable now |
| 3 | repair chain with the "no difference found" verdict; ids 12, 13, 45, 108 through it | TASKS/015 | open, unassigned, blocked on stage 1 |
| 4 | corpus sweep in the test container | CI agent | not started; gated by 1-3 |
| 5 | error-tree sweep, the three numbers, the owner's yes | CI + Forensic | not started; gated by 4 |

Stages 1-3 may interleave. 4 gates 5.

## 2. Seats and assignments

| seat | model | scope | state |
|---|---|---|---|
| `vmsam-lead` | opus | docket, gate, commit, push | running |
| `vmsam-architect` | opus | governance, stage-0 rewrites, the missing pHash ticket | running |
| `vmsam-dev-frame` | sonnet | **the four `ARCH_FRAME_ACCURATE.MD` frame-accuracy defects** — NOT TASKS/013's three locator defects | **code landed `3f1f2987` at exit 0**; parked, held for the forensic comparison (R6) |
| `vmsam-ci` | haiku | the deploy chain and the run log; fires the hook, verifies what is running, queues fusion jobs | spawned 2026-09-15; task 0 is the open build incident (§6) |
| `vmsam-forensic` | sonnet | hand-built targets vs the system's products: stage-1 post-dev check, TASKS/014 ground truth, the no-video-master count | spawned 2026-09-15 |

**TASKS/013's three locator defects (alias · mis-modelling · empty-vs-decline)
remain unassigned and need a seat.** They are a different scope from
`vmsam-dev-frame`'s four; do not let the stage-1 row hide that half.
Tickets 014, 015 and 016 have no seat either.

Spawning is the Lead's or the Architect's act, and a seat's `BRIEF.md` is written
before it starts — an agent never writes its own brief.

## 3. Lead's rulings this round

**R1 — `vmsam-dev-frame`'s defects 2 and 3 are not to be written into
`get_cut_time.py`.** Asked 2026-09-15 whether to (a) fix them in place, (b) land
the corrected logic as a spec note, or (c) hold. **Ruling: (b).** The file's own
header retires it and forbids it an importer, `TASKS/005` §4 ruled the same, and
`SPEC_ZONE_A.MD` retracted the instruction to wire `_refine_cut`. Fixing a defect
in a file that may never acquire a caller produces exactly the campaign-1 failure
`BRIEF_COMMON.md` rule 2 names: reviewed, tested, green, and never called. The
corrected logic is recorded so it is not re-derived, and it is **implemented by
whoever builds the refiner's real consumer** — which is the ticket the Architect is
opening (`FINDINGS.MD` S0-3). This is an option ruled on, not a licence to go
build the consumer: that scope is not `vmsam-dev-frame`'s.

Defects 1 and 4 are unaffected — `frame_compare.py` and `merge_video_chimeric.py`
are open under `WRITE_ZONES.MD` §4 and proceed.

**R3 — `vmsam-dev-frame`'s defects 1 and 4 are not accepted yet.** One blocking
finding, measured by firing the new guard on literals (five of six cases correct,
which is what makes the sixth credible): `master_frame_grid_is_unmeasured` tests
`frame_rate_original is None`, but the assignment above it can set that field to
`""`, so a VFR master with a **blank** `FrameRate_Original` is accepted on a grid
nobody measured — the defect the guard exists to stop, one field over. Named
correctly by the F1 contract's rule 6: an unknown grid must be *unrepresentable*,
so "unmeasured" is a property of the value, not of its absence. Second finding,
to be fixed or stated: dropping the `fps=` filter moved extraction from `fps=10`
to native cadence, so `max_search_frames=50` fell from 5.0 s of content to
**2.09 s** and `band_width=20` from ±2.0 s to **±0.83 s** — a behaviour change
riding inside a fix for something else. Sent back with a target-first requirement
on the fix.

*Update, same day, second review.* Both closed, **verified by the Lead rather than
accepted on report**: the guard now declines on empty, blank, zero, negative,
zero-numerator, zero-denominator, malformed and wrong-type rates — 20 of 21 cases
in the Lead's own table, five of which the seat had not tried. The 21st is a leak:
`"inf"` is accepted, because the predicate decides by `float(x) > 0`. Low severity
— MediaInfo will not emit it — but it names the shape: the Architect ruled a
**normaliser returning an exact rational or `None`**, and a float-comparison
predicate both admits `inf` and discards the exactness `RULING_20260915`'s rule 2
requires. Returning `Fraction | None` makes `inf` unrepresentable instead of
merely rejected. The seconds-based defaults are confirmed restored: 48 / 120
frames at 24000/1001, measured coverage 2.002 s / 5.005 s against the old 2.0 /
5.0. Constructor guard fires both ways on literals.

**R5 — a precondition is probed at admission, not at delivery** (Architect,
2026-09-15, raised from R4). The grid probe must move to the entry of
`assemble_on_master_timeline` and the `mark_output` REFUSED-rename block for that
path must be **deleted, not mirrored** — with nothing yet written there is nothing
to un-say. The deeper reason: under the landed F1 contract an unmeasured grid
cannot legitimately reach the assembler at all, so the late gate is
dead-by-construction. Measured as not yet done: the call still sits at
`merge_video_chimeric.py:2020` in a function beginning at `:1648`. Directed at
the seat; open-zone code, no governance gesture needed. **The Lead's first review
told the seat this was not theirs to act on — that reversal is the Lead's to
carry, not the seat's.**

**R4 — an architecture finding held, not assigned.** The grid refusal fires
*after* the mux completes, which is the only reason it needs the `mark_output`
REFUSED-rename mirror. `FrameRate_Mode` is knowable before any work starts;
moving the probe forward would make the refusal cheap and delete the mirrored
crash path. Larger than the seat's ticket — routed to the Architect.

**R6 — a seat is not purged until its code has run in a container.** The purge
protocol says closed AND merged. I require, in addition, that the code has
executed in the test container and been compared against a hand-built target.
**This is my ruling layered on the protocol, not a reading of it** — recorded that
way so no successor cites a rule that does not say it. `vmsam-dev-frame` is held
under it despite its ticket being complete.

**R6a — R6 amended, because I measured a case it cannot decide.** R6 said a seat
is not purged until its code has run in a container. **Defect 1's code cannot run
in any container**: `frame_compare.FrameComparer` is imported only by
`src/get_cut_time.py`, which is retired and has no importer of its own, so
nothing in the deployed image calls it. Measured, not inferred — `grep` over
`src/` and `src/gestionar_show/`. No number of runs changes that, so R6 as
written would hold `vmsam-dev-frame` forever on a condition no action can satisfy.

**Amended:** a seat is released when its code has **either** (a) run in a
container and been compared against a hand-built target, **or** (b) been shown by
measurement to be **unreachable**, with the import evidence recorded and the
dead-code state opened as a tracked finding. **(b) releases the seat; it does not
close the defect.** The fix stays correct and stays uncalled until the frame
refiner's consumer exists — which `TASKS/006`'s amendment reinstates and which no
ticket yet owns. This is the campaign-1 failure mode (`BRIEF_COMMON` rule 2)
caught at the moment it was about to be papered over with a container run that
would have proved nothing.

**Standing consequence: defect 1 is dead code in production today.** It is listed
in §5 as a block so it is not mistaken for delivered work.

**R7 — two briefs were corrected before placement, and one was contradicting
itself.** The Architect drafted `ci` and `forensic`. `forensic` was placed
unchanged. `ci` carried two defects: its chain section says *"never fire on a
clock, fire on a measured gate"* while its own procedure step 1 said *"confirm
>= 10 min have passed"* — **a literal-minded seat following the procedure would
have done exactly what the day's incident proves is wrong** — and a section
header plus its first item had been lost in a rewrite. Both fixed in place, with
the superseded instruction named in the file so the seat can recognise it if
anyone repeats it. *The test that caught it: read every brief as a literal-minded
reader who will follow the most concrete instruction on the page.*

**R2 — the seat reporting a stale brief is right and the brief is not corrected by
the seat.** `vmsam-dev-frame` measured that its brief omits `get_cut_time.py`'s
retirement and calls `change_point_locator.py` "a new runtime module" when it
already exists at ~1993 lines with a live consumer. Both stale, both verified
against the files. A brief is written by the seat that spawned it: routed to the
Architect, not edited.

## 4. Gate log — what this seat has validated and pushed

| date | subject | checker | compileall | pushed |
|---|---|---|---|---|
| 2026-09-15 | baseline at open, nothing staged | exit 0, 35/35 | exit 0 | — |
| 2026-09-15 | stage-0 corrections + the frame-indexed boundary contract | **exit 1, six §1 lines, every one an authorised path** — the owner's one-time bypass, spent | exit 0 | `1ea300f1` |
| 2026-09-15 | `vmsam-dev-frame` defects 1 and 4 | **not gated — sent back**, see R3 | exit 0 | — |

The `1ea300f1` row is the only commit in this campaign that landed at exit 1, and
it did so under an explicit single-use owner authorisation recorded in
`FINDINGS.MD` ("Governance bypass"). **It is not a precedent and may not be cited
as one.** Procedure actually run: staged exactly the authorised paths; `--staged`
gave six violations, all §1, all on the list, none frozen/tagged/config/compile;
re-run on the landed rev gave the identical six — no drift, nothing rode along.
`vmsam-dev-frame`'s two `src/` files were deliberately left out: the exceptional
and the normal do not share a commit.

## 5. Standing blocks that are nobody's ticket yet

1. **`merge_video_repair.py` and `merge_video_resample.py` are closed** pending the
   owner (`RULING_20260908_REPAIR_RESAMPLE.MD`). Work on them is built inert in a
   lab and does not land. Stage 3 touches this.
2. **The four outlier ids (12, 13, 45, 108) are out of every denominator** until
   TASKS/015 rules — recorded so the sweep does not silently absorb them.
3. **The resampled candidate is never exposed as a file** (`TASKS/008`): the
   locator has never run on a corrected candidate. Until it has run end to end,
   `resample -> locate -> splice` is a design, not a capability — its author's
   hedge, carried unsharpened.
4. **Two frozen-file defects, unowned since 2026-09-08** (`FINDINGS.MD` S0-6): the
   `127`/`128` one-frame gates in `first_delay_test`, and the `delayUse=0` calls at
   `mergeVideo.py:489`/`:502`. Findings, not edits, and **explicitly not licensed**
   by the owner's 2026-09-15 quantum ruling.
5. **`BRIEF_COMMON.md` is frozen carrying a sentence measured false** — "one point
   = 125 ms" — **by the owner's explicit choice** (`FINDINGS.MD` S0-5), not by
   oversight. `AGENT.MD` now carries the measured statement instead.
6. **A master with no video stream has no master frame grid** — the F1 contract
   anchors every boundary on it. The question is a count and belongs to the corpus
   instruments: **does the error tree contain pairs whose MASTER carries no video
   stream?** Zero means the contract is complete as written and a guard must still
   DECLINE on the case; non-zero means a second clock must be stated, and that is
   an owner question. Unassigned, needs a forensic/corpus seat, gates nothing today.
7. **The write-zone checker forbids what `WRITE_ZONES.MD` §1 permits.** §1 allows
   the Architect or the Lead Dev to edit governance documents **with the owner's
   authorisation**; the checker implements no authorisation path of any kind. Run
   against landed history it returns **exit 1 on five of the last five governance
   commits** (`523ec97e`, `7068b07e`, `47817068`, `c6f73d12`, `1a6a290b`). So the
   instrument and the practice have disagreed for the whole restart. **Blocking:**
   the `AGENT.MD` quantum correction and the stage-0 `FINDINGS.MD` entry are
   authorised by the owner and still cannot reach exit 0, so both sit uncommitted.
   Put to the owner; **the Lead does not choose among the options** — the gatekeeper
   is the seat with the most obvious motive to loosen its own gate.
8. **Stale text in two frozen briefs**, reported by the seats that hit them, routed
   not edited: `architect/BRIEF.md:69` still assigns dev-seat spawning to the
   Architect (superseded by the owner, 2026-09-15 — it is the Lead's);
   `architect/BRIEF.md:53` says agent directories are root-owned, which is false on
   this machine — no root is available to this fleet at all; and
   `dev-frame/BRIEF.md` omits `get_cut_time.py`'s retirement and calls
   `change_point_locator.py` "a new runtime module" when it already exists at
   ~1993 lines with a live consumer.
9. **`config.ini [features]` has never existed** (`FINDINGS.MD` S0-7) although every
   brief points agents at it. Creating it is an addition, allowed by
   `WRITE_ZONES.MD` §3. Nobody owns writing the section.


## 6. OPEN INCIDENT — the image build has been failing since overnight

**Nothing this round has ever executed.** Both containers
(`showgestionar-test`, `showgestionar-test2`) served `e2691f63`; one hook firing
moved them to `523ec97e`, the pre-round tip, **committed 547 minutes earlier**.
Neither `1ea300f1` nor `3f1f2987` has an image.

Measured, in order:

    hook                         status=SUCCESS exit_code=0    (it ran)
    both containers              e2691f63 -> 523ec97e          (one firing covers BOTH)
    origin/dev-AI                3f1f2987                      (the push is fine)
    GitHub mirror                3f1f2987, 9 s after the push  (the mirror is fine)
    Actions 3f1f2987 / 1ea300f1  FAILURE, "Build and push Docker image"
    Actions 523ec97e             success, the previous evening
    registry dev-AI revision     523ec97e

**Cause, confirmed against `sources.debian.org`:** `Dockerfile` line 56 pinned
`python3-pydantic=2.13.4-4` and `python3-pydantic-core=2.46.4-3`. Testing now
carries **2.13.5-1** and **2.46.5-1** — *the pinned versions have left the
archive*, and `apt install pkg=version` on a vanished version fails hard. The
pins were deliberate (`a8342d02`, *"force the version"*).

**The decisive argument, and it is what rules out our own code:** `1ea300f1` is
**docs-only**, and it failed the build. The `Dockerfile` `COPY`s `Cargo.toml`,
`src/*.py`, `src/*.ini`, the two json files, the shell scripts and the two
package directories — **no markdown file enters the image at all**. A commit that
changed nothing the build reads cannot have broken it. Any build after the
archive moved would have failed, whatever its diff.

**Two corrections on the record, one from each seat that diagnosed this.** Both
the Architect and the Lead reported *"nothing in the Dockerfile is pinned"*. Both
were wrong, and wrong the same way: each scanned `FROM` lines and `dist-upgrade`
and never read the `apt install` line, then reported a conclusion the search
could not have supported. **Line 56 was the only pinned thing in the file, and it
was the whole cause.**

**The lesson, replacing both earlier phrasings:** on a moving-target distribution,
**an unpinned dependency drifts silently at runtime; a version pin expires loudly
at build** — the archive deletes old versions, so a pin is a time-bomb with an
unknown fuse. Neither direction is safe on `testing`. This time it failed loudly,
which is the lucky one.

**And it was invisible because nothing watches the build.** It would have stayed
invisible until someone needed a container. That is now `vmsam-ci`'s standing
job, and the reason its brief fires on a *measured registry gate* rather than a
clock.

**Status: a one-line fix is prepared in the tree and NOT committed**, awaiting
the owner's confirmation directly to the Lead. The authorisation was given to the
Architect and relayed; the owner's standing instruction to this seat is *"always
ask me next time"*, and `Dockerfile` is frozen by `WRITE_ZONES.MD` §1. Verified
while waiting: the diff is exactly one line, the two pins removed and the other
four packages untouched, and nothing else is modified in the tree.

**Residual risk, flagged not resolved:** those pins were added deliberately. If a
newer pydantic breaks VMSAM at runtime, removing them trades a loud build failure
for a quiet runtime one. The owner pinned them and has read the log; the call is
his.

## 6b. The corpus cannot exercise the campaign's own instrument

**Measured 2026-09-15 by `vmsam-forensic`, verified independently by the Lead:**
`VMSAM_CORPUS/` holds **142 media files; 5 carry a video stream; 137 are
audio-only.** The five are `corpus-from-KEEP/*.mkv` finished merge **outputs** —
not divergence pairs, and none carries a recorded expected boundary.

The consequence chain, in one place:

- the requirement is repair **to the frame**, validated by **pHash on decoded
  video frames**;
- `RULING_20260915_FRAME_INDEXED_BOUNDARY.MD` keys every boundary on a **master
  video frame index** with a mandatory exact-rational grid;
- `TASKS/016` must calibrate that pHash window and floor against **known frame
  boundaries**;
- 016 calibrates against `TASKS/014`'s rows;
- **014 can produce no row at all: there is no video to index against.**

Synthetic fixtures are audio-only **by the owner's stated construction
constraint**; curated real-id cases are audio-only too — every `CASE.json`
extraction runs `-vn`, so the corpus stores per-language FLAC and never the
original video. Nobody decided this: the corpus was built for campaign 1's
**audio correlation** work, the campaign pivoted to **frame accuracy**, and the
corpus never followed. It stayed invisible until something finally tried to
measure a frame boundary.

**With the owner. A costed proposal exists** (`VMSAM_HELP_AI/forensic/`, lab-only,
nothing landed): all-intra fixtures, **~232 MB per boundary case**, **~1-1.5 GB**
for a suite against the existing 16 GB, with a built-and-measured accept/reject
pair proving byte-identity up to the claimed boundary and first divergence
exactly at it. Nothing enters the corpus until he rules.

## 6c. The decline arm of defect 4 cannot be fired on real media

`merge_video_chimeric.py`'s own comment records a sweep of **561 files: 559 CFR,
2 VFR**. The two VFR ids would be the decline case. **The per-id data is gone** —
it lived only in dev-1's lab; searched across `VMSAM_HELP_AI/`, `VMSAM_CORPUS/`
and both campaign archives, only the aggregate survives, because the aggregate
had been written into the code. Media is mounted read-only and a VFR master will
not be fabricated.

So the container leg covers **the pass arm only**, and the honest form of the
claim is: **the guard was fired in both directions LOCALLY** (33 hand-built cases
from forensic, 21 from the Lead, zero mismatches) **and in the container in one
direction only.** Stronger than a unit test, weaker than both-directions in
production. The word *locally* travels with that claim or the claim is wrong.

## 7. Still unseated

TASKS/013's three locator defects, 014, 015, 016. **016 gates the most** — every
boundary acceptance in the contract landed at `1ea300f1` cites its placeholder —
but it waits on 014's rows, because calibration without ground truth is the
instrument calibrating itself. The no-video-master count rides with the forensic
seat as one `mediainfo` sweep.


---

# THE STATE AT 2026-09-15 13:00 — read this section first after a restart

Everything above is the day's accumulation. This section is what a successor
needs and is kept short deliberately.

## The mill — the owner's protocol, end to end

    CI random-feeds (5/container, both containers) and TRIAGES every .log.error
      -> EXPECTED (do-not/cannot-help refused with a named reason): forensic confirms, row closes
      -> UNEXPECTED-DECLINE / CRASH: one message per row to the Lead
    -> the Lead DISPATCHES an investigator per SIGNATURE (not per id)
    -> investigator: raw fpcalc PER-POINT vector over the bounded zone, then pHash
       on the frames, then ITS OWN VISION on extracted images, then a REPORT:
       zone in master frame numbers, evidence trail, diagnosis
    -> the Lead hands the report to a DEV seat with the problem details
    -> the dev hardens the GENERIC method; the Lead writes no production code
    -> normal gate at exit 0 -> CI redeploys -> the id resubmits -> the row flips
       or its refusal is ruled correct

**The dev seats' standing objective (owner, verbatim):** *"surtout rendre plus
robuste la methode de detection des differences, et d'adaptation des candidats
sur le master."* **Every patch aims at generic robustness of (1) difference
detection and (2) candidate-onto-master adaptation. A patch that makes one id
pass without strengthening the METHOD on its family is refused on that ground
alone** — it succeeds for the wrong reason.

## The measurement-retention invariant (Architect ruling, 2026-09-15)

Sibling to *"`None` means could not measure"*:

> **A measurement, once made, is never silently discarded.**

1. A filter dropping a measured run/segment/probe **records the drop per item,
   with a named reason**, in the same structured stream as the measurements.
2. The comparison stage either runs on **all** measured runs before usability
   filtering, or **declines explicitly with the drop-list attached**. It may
   never compare only the survivors and present the result as complete.
3. The usability filter is not the enemy — its defect is **position** (before
   comparison) and **silence** (unnamed drops).

**Its floor, which is as important as the invariant:** *retaining a measurement
cannot recover resolution the instrument never had.* See the sub-quantum entry
below before building any discrimination on fine offset differences.

**Pre-registered acceptance test for the retention patch** — a test, never a
target, and both numbers are reported whichever way they fall:

    PREDICTION  master fill on the id-12 family collapses toward the true gap size
    REFUTATION  fill stays at TASKS/007's 21.9 / 31.6 / 22.4 % -- the fill really
                was gap, and the retention defect is real but small

**Why it outranks everything else open:** end-condition **clause 2** (untouched
zones bit-identical to their source) is **unreachable while matching content is
being re-filled from the master**. The defect is on the campaign's critical path.

## What id 12 measured — the campaign's first product, and what it cost

The chimeric repair produced a **1.27 GB file**, the first ever. Its log says
`change_points=0`, `offset_monotone=true`, a single constant **-1412.34 ms** —
and the assembler discarded **515.7 s of candidate audio (43 %)** and filled it
from the master.

**The mechanism, measured by `vmsam-dev-frame` with `tools.dev` instrumentation,
pair digest reconciled to `e39737967605`:** detection **worked** — 8 runs
correctly separated, including a settling head and a 22-probe stable body. The
per-segment usability filter then dropped **6 of 8**, and change-points survive
only between two runs that **both** survived. All three head runs went, so
**nothing ever compared the head's ~-570 ms against the body's -1412 ms** — an
845 ms difference, measured and discarded.

> *"It didn't fail measuring. It failed keeping what it measured."* — dev-frame

**The 140-second chain, end to end:** one spurious probe at **t=520 s** sits
inside the 480-620 s interior fill; it splits the body plateau; the transition
around it brackets to nothing; the segment is dropped; **140 s of matching
content is replaced by master fill.** One noise probe, 140 seconds.

**Forensic's independent measurement** (its own RMS-envelope instrument, firing
control 0.13 against 0.94): head ~**-640 ms to ~65 s**, ~**-570/-590 ms to
~158 s**, transition at **158-165 s**, then **-1430 ms**. So ~158 of the 220 s
head fill also has matching content, under offsets the single-constant model
never considered. **The interior 140 s is the cleanest measured waste and does
not depend on the head question.**

## The sub-quantum limit — the owner's two-instrument division, demonstrated

Id 12's head spread is ~**70 ms**. The chromaprint quantum **on this file, from
its own log**, is **129 ms**. So the head structure is **sub-quantum: `fpcalc`
cannot resolve it even in principle** — dev-frame's "settling head" of -642 /
-588 / -567 was real structure, not noise.

70 ms is ~**1.7 video frames**: invisible to chromaprint, squarely visible to
pHash. **`OWNER_SPEC_FRAME_REPAIR.MD` asserted that division of labour; id 12 is
the first measured case on real media demonstrating why it is necessary.** It is
the empirical answer to any future seat proposing to collapse the two
instruments into one.

## Two instrument defects, kept separate on purpose

1. **Peak-finder miss** (forensic, self-found and self-reported): a brute-force
   argmax missed **0.99 at -567 ms sitting inside its own search window**. Same
   class as `TASKS/013` §1's unresolved **alias defect** — the first evidence
   anyone has put under that ticket. Hedged: the bug is not fully isolated.
2. **Sampling-density trap** — the more dangerous, and a **class rather than an
   instance**: raw-waveform correlation on this pair peaks only a few ms wide, so
   **0.9899 at exactly -567.0 collapses to 0.18 at -570**. Any correlation-shaped
   instrument sampling coarser than its own peak width produces **confident
   nonsense**. The pattern-check — *what is your peak width against your sampling
   step?* — is worth running against `audioCorrelation` itself.

## Named, not solved — the drift chicken-and-egg

If forensic's 0-158 s zoom returns **drift** rather than a second step, the head
carries a **speed relation on a SEGMENT, not the whole file**. The campaign's
order is `resample -> locate -> splice`, which handles a segment-scoped drift
only if the drift zone is **located first**. **Named today so nobody discovers it
inside an implementation.** If the answer is drift, it goes to the Architect
before anyone designs.

## Reproducibility — the finding that bit three times in one day

A production run **does not record which streams it selected**, so nobody can
reproduce it from outside. It cost a false reconciliation on id 108, a
stream-selection doubt on id 12, and required `tools.dev=True` plus hand
instrumentation merely to see which segments were dropped. **The end condition
rests on hand-built targets compared against products: if the production
instrument's own inputs are not reproducible, every such comparison is unreliable
by construction.** Fix, generic and cheap, folded into the patches: emit the
resolved master/candidate stream pair and language code on the **success** path
as the decline path already does, plus `segments_dropped_unusable` with
positions, offsets and reason.

## Open with the owner

1. **The corpus has no video-bearing divergence pairs** (142 media files, 5 with
   video, none of them pairs). Blocks TASKS/014, TASKS/016, and calibration for
   **two** patches. A costed proposal exists: all-intra, **~232 MB per boundary
   case**, **~1-1.5 GB** per suite, with a built-and-measured accept/reject pair.
   Nothing has been written to the corpus.
2. **The pairing patch's cross-language risk.** Finished, gated **exit 0**, not
   landed. Condition 1 could not be measured — the controls die at an earlier
   gate and never reach the changed code. Lead's recommendation on record: land
   it, because `VMSAM_MODE=test` and forensic's five-leg validation are what
   protect the library today, **but it must not reach production mode until
   condition 1 closes.**
3. **No read-only path from `folder_id` to a master** — blocks the
   no-video-master count.
4. **`actions:read` token** — deprioritised; buys log lines on a monthly-at-most
   failure.

## A lesson of mine, recorded because it cost 74 minutes

**A gate nobody revisits is not a gate, it is a stop.** I held the wide sweep on
a condition batch 1 had already satisfied, and did not go back to check it while
dispatching elsewhere. The sweep is also what **generates** the material the
remaining open questions need, so holding it was circular. Both containers had
been idle for 74 minutes against a standing owner instruction.
