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
| `vmsam-dev-frame` | sonnet | **the four `ARCH_FRAME_ACCURATE.MD` frame-accuracy defects** — NOT TASKS/013's three locator defects | running; 2 of 4 in flight, 2 ruled out (§3) |

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
| 2026-09-15 | docket + stage-0 findings + the `AGENT.MD` quantum correction | see the commit | see the commit | this commit |

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
