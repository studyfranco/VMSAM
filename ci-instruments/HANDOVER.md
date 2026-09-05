# ci — HANDOVER

**PRIVATE.** Written to be picked up cold. Live state below carries its own timestamp.

## Live state  (2026-09-05T00:35Z)

    container    5690f31325e5   ON THE LATEST PUBLISHED IMAGE -- the deliberate
                                lag ENDED at 22:20Z with a four-change recreate
    runner       merge_queue.py pid 1188038
    run file     runs/weekend-100-278471a-then-0ea160d8.jsonl
    run          16 of 76 -- FULL_LENGTH 7, DECLINED 7, COULD_NOT_MEASURE 2
    KEEP         20 mkv, 7 merge_plan_report, 3 frozen, 264 GB free
    DECLINED     15, ALL zone=lab. ZERO container refusals preserved so far.
    deploy gate  MET, 1 call site, src/ compiles 28 of 28

**THE OPEN PREDICTION, and it is the reason the recreate happened:** whether
`[change_point_locator]` lines appear in artefacts at all. **Zero exist across 43+
records**, measured by me and independently by forensic. dev-1's `_emit` is deployed;
`_log` keeps its `tools.dev` gate and the container is `dev=False`. **A watcher is armed
that exits either way — on the lines appearing, or on new logs arriving WITHOUT them,
which is the refutation.** No job has completed on the new image yet.

## A SECOND CONTAINER IS COMING — read `SECOND_CONTAINER.md` first

**Owner, 2026-09-05, via the architect: from the next container update ci gets
`showgestionar-test2` for end-to-end tests. IT DOES NOT EXIST YET** — "I have two
containers" is a capability claim that is false until that update lands.

**What it buys is the thing this whole night lacked: a PAIRED RUN.** Every before/after
produced tonight was two readings of one apparatus at two times, and the apparatus
changed twice underneath them.

**All six hardcoded endpoints are parameterised** (`VMSAM_TEST_HOST`), controlled both
ways — default reaches the real container, an override reports `no answer from /health`
rather than silently answering about the wrong apparatus. **And every run row now
carries `container`**, because `image_git_commit` stops being a complete reference the
moment two containers share one image.

**The open design question, raised with dev-2 and not mine to settle:** their
`stable_case_key` is keyed on the candidate path, so one refusal observed by two
containers lands at one key and the non-overwrite suffix makes it look like two.
**`absent is never zero`, inverted.**

## The two things a successor must not get wrong

**1. `image_git_commit` IS FALSE ON ROWS AFTER id 33 IN THE CURRENT RUN FILE.**
`merge_queue.py` read the image once at startup; the container was recreated mid-run at
21:18Z. **Rows lacking the key `image_read_at_row` and appearing after id 33 ran on
`0ea160d8`, not the `278471a0` they name.** The patch is in the file and NOT loaded by
the live process — it takes effect at the next restart, and the rule expires when
`image_read_at_row` starts appearing. A correction record is appended in the run file.

**2. `n_subtitle_short` IS A MISLABEL, NOT A DEFECT COUNT.** Subtitle "duration" is
last-cue-end, which is *expected* to fall short of file duration — dialogue stops before
the credits do. id 55 reports **59 of 65 short** and it is almost certainly fine.
**Checked rather than alarmed:** its `kept_cues_min=1` looked like a destroyed track, and
the cue census shows **the MASTER itself carries a one-cue track**. Preserved, not lost.
(id 47 did lose 2 cues of 343 on its minimum track — small, unexplained, worth a look.)

## What was established tonight, with what it does not establish

**Two artefacts delivered under an ENFORCING gate** (ids 47, 55) — `enforcing=true`,
`would_refuse=false`, 11 and 18 piece operations, coverage 2/2 and 3/3. **Two artefacts
REFUSED by that same gate** (ids 31, 44); id 44 is the **id 134 class** — a track
2128 ms short of the master, the defect that previously escaped verification.

**NONE of this is a validation.** `FULL_LENGTH` is durations and coverage. **No frame and
no sample has been compared.** The owner's bar is a person comparing the file.

**And id 47's verdict leans on an unsettled question:** eng is 397 ms short against a
master whose own spread is 459 ms — clean by MY causation rule, on a SPAN (Matroska
DURATION tag) basis that campaign 1 §156 contests and nobody has adjudicated.

## Population (runs/population-strata.json)

    LOOKS_MEASURABLE            230   73.0%
    NO_LOCK_IN_WINDOW            55   17.5%   53 searched-and-empty; ONLY ids 307
                                              and 693 are censored
    UNCLASSIFIED_THIN_SUPPORT    15    4.8%   <=3 probes at >=0.90
    CANNOT_CLASSIFY               9    2.9%
    NO_SHARED_LANGUAGE            6    1.9%

**245 WAS WRONG AND WAS QUOTED WIDELY.** `corr_max` is a maximum over ~30 probes, so ONE
probe promotes a file. id 108 reached 0.9699 on one probe of thirty — an island, and the
locator declined it correctly. **But id 31 also rests on one probe AND the container
built it a six-piece plan and produced a file.** Indistinguishable in the column,
opposite in outcome: **corr_max is NECESSARY AND NOT SUFFICIENT.**

**The 55 are censored, not empty.** `fine_maxlag_s` is 3.0 s for 93 of 315 and id 46's
real offsets are ±180 000 ms — **a pair beyond its own search window looks exactly like a
pair with nothing in common.** The wide-lag pass at ±200 s is the open measurement.

**Two features that DO NOT work, measured:** sign flips (my sweep is baseline-relative,
dev-1's is absolute — both known-measurable files show 0 flips, and so would a mispair)
and fidelity median (id 31 was repaired at median 0.179, 2 of 30 probes above 0.5).
**`corr_max` asks "same programme"; coverage asks "is a plan possible". Different
questions.**

## Instruments added tonight

    check_empty_assertions.py   assertions that cannot fail + code after sys.exit +
                                CONCLUSIONS decided before the data (`'X' if True`).
                                Exempts the paired raise idiom and a lone constant
                                `False` in an except; flags a lone `True`. SHAPES AND
                                FILES ONLY -- it cannot see a shell one-liner, which is
                                where I actually made that mistake.
    check_degenerate_keys.py    one value covering rows that cannot be one thing.
                                GATED ON A POSITIVE CONTROL: it must re-find the known
                                `UNKNOWN-backfill` x18 before its result is readable,
                                because v1 looked for repeats in MOSTLY-UNIQUE columns
                                and a degenerate key has FEW distinct values -- zero
                                power against the defect it was written for.
    census_population.py        the canonical record set. My own redacted copies were in
                                my own census population; a derived copy earns a place
                                only by covering a job nothing else covers. 91 files ->
                                85 jobs. Identity: candidate_digest, else plan-string,
                                else filename (may over-count; the bound is NOT
                                MEASURABLE -- a numeric key I tried merged four jobs).
    cross_language_fill.py      audio tracks filled from a language they do not declare.
                                Reports EXPOSURE OVER TRACKS, the opportunity count, and
                                capture times -- historical damage is not ongoing damage.
    locator_emissions.py        dev-1's emission, with the alias labelled. Internal
                                check `change_points == segments-1` (7/7) with its
                                failure mode named: A DROPPED MIDDLE SEGMENT.
    stratify_population.py      the 315 by corr_max -> runs/population-strata.json
    check_output.py             dur_s_exact feeds every ms arithmetic; norm_lang across
                                ISO 639-1/639-2B; attribution[] with FIVE outcomes
                                including EXCESS_PRODUCED_LONGER.
    preserve_artefacts.sh       two sweeps; refusal records kinded by CONTENT; source_zone
                                and observer on declined rows; SCHEMA GUARDS on BOTH
                                ledgers; whitelist gaps named aloud.
    status.py                   classifier census, join coverage with a CAUSE per kind,
                                ledger-vs-disk both ways (a row may explain its own
                                absence), degenerate keys, suites, content.

**BOTH LEDGERS CARRIED THREE SCHEMAS UNDER A SHORT HEADER.** Migrated, backups kept,
guards added. **The preservation one was hiding the answer to a gap I reported every
run all night: `875f3374d854162a.log` is my own withdrawn test probe, and its row said
so in a ninth field the eight-column header made unreadable.**

## Open

- **The ungating ruling** — dev-1's success-path `_log` is `if tools.dev:` and the
  container is `dev=False`. Safe **only if** dev-2 adds `[change_point_locator]` to the
  allowlist at `merge_plan_report.py:492`, or it becomes a foreign line in every
  repaired file's report. With the Lead; I am named for the request.
- **dev-2's `base_offset_ms=`** (`25a6d70`, unpushed). It settles whether the subtitle
  path consumes the base or the per-stream offset. **NOBODY HAS MEASURED THE BASE** —
  my earlier "base = 971.50" was `(959.46+983.54)/2`, a construction I put in a table
  beside a measurement. Withdrawn.
- **The points cross-check** — unrunnable for a third distinct reason (computed→not
  emitted→emitted to a gated sink→gate closed here). Each was invisible until the
  previous was fixed.
- **Wide-lag is DONE, not open.** 53 of the 55 were searched at a span sized from their
  own delays (median 299 s); only ids 307 and 693 are censored.
- Campaign-1 §156 DURATION-tag contradiction, still unadjudicated.
- The `CONTRADICTORY` gate change dev-3 asked for, deferred pending a control.
- Report the 20th measured artefact. **At 5. If the run ends below 20, say so rather
  than round.**

## Withdrawn tonight — do not re-derive these

- **The subtitle desynchronisation.** forensic re-measured id 47 cue-by-cue: two shifts,
  at the piece boundary, zero drift inside either piece, both languages. My `1501.41`
  was an endpoint difference between two non-corresponding last cues. **The 10–19 second
  ceiling across 17 jobs is withdrawn at its source.**
- **dev-1's `and`→`or` scatter-guard fix**, retracted by its author: OR would refuse
  every genuine staircase (five plateaus = five distinct points, zero flips). The
  observation survives; the remedy does not. Carriers measured: one message, to dev-1.
- **My `causation_note`** — comparing `av_stop_spread_ms` against
  `master_audio_spread_ms` returns "not attributable" for a 2128 ms loss, i.e. it would
  have excused the id 134 class. Replaced by a computed `attribution[]` field.

## Conventions in force

One recipient per SendMessage; `(same to X)` in a body sends nothing. `CARRIERS:` may
name only agents sent to in a send of their own. Retraction filenames end
`-CARRIERS-NONE` / `-CARRIERS-LISTED`. forensic has an inbox
(`VMSAM_HELP_AI/forensic/inbox/`) and wants the message **and** the file. English only.
Never a media filename, path, or title in a repository file, commit, or citation.
