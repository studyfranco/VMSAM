# VMSAM — Task state & resume

**Purpose**: the single place to find *what is being done, what is finished, and
what to pick up next*. Read it at the start of a session; update it the moment a
task changes state. It is a condensed resume, not a worklog — depth belongs in
`AGENT_NOTES.MD`.

**Last updated**: 2026-08-29 21:20Z.

---

## Mission

Every one of the **309 error files** in `<ERROR_TREE>` must
end in exactly one of two states:

- **Resolved** — passed BOTH validations: Step 1 (local audit, cross-provenance
  audio drift **< 15 ms**) AND Step 2 (re-merged on `showgestionar-test:8080`
  with output in `/config/output`).
- **Unmergeable_Verified** — proven impossible, with the measurement that proves it.

Plus: the **Master Summary Table** covering all **315** files, in `AGENT_NOTES.MD`. (309 was the count before six files that had arrived ~35 h earlier were found to be in the errors table but absent from the private mapping.)

---

## Progress at last update

| Metric | Value |
| :--- | :--- |
| Processed through the live API | **179 / 309** |
| MERGED_IN_SYNC (Step 2 + Step 1 verified) | **38** |
| FAILED | 115 |
| NO_OUTPUT | 20 |
| MERGED_OUT_OF_SYNC | **0** (err_9 was a false positive — see below) |
| First-pass diagnostics written | **309 / 309** ✅ |
| Deep forensic reports written | **126 / 309** |
| Deployed SHA | `0f426fc` (= `dev-AI` HEAD at the time) |
| `/config/output` free | 392 GB of 932 GB (58% used) |

Batch position: **C_cut 50/180** (classes done: E_corr, B_fps,
D_trim, A_no_lang).

---

## Coming back after an interruption

Follow **`src/COMING_BACK.MD`**,
in order. The short version:

1. read this file, then `git log` and **`git status`**
2. **rescue uncommitted work first** — a killed agent's deliverable lives in the
   working tree and nothing else protects it
3. `ps` (not a bare `pgrep`) to see what is alive; restart the dead from the
   registry below
4. `GET /health` — `git_commit` must equal HEAD and `dev` must be true, or
   nothing the container produces means anything
5. tests green, privacy check clean, counts from `master_summary.py` — never
   from memory
6. report where you are before doing more

## ⛔ Deliberately stopped — do NOT restart these

The hourly refresh says to restart any job showing `—`. **Check here first.** A
service stopped on purpose looks identical to one that died, and restarting it
can be worse than the outage — the entry below was stopped to stop it recreating
the container in the middle of a running test.

| Service | Stopped by | When | Why | Restart when |
| :--- | :--- | :--- | :--- | :--- |
| ALL JOBS | Lead, on the owner's instruction | 2026-09-02 ~08:5xZ | **The owner stopped the campaign to restart the project on a new basis.** Sweep (`deploy_queue.sh` -> `full_test.sh` -> `batch_verify.py run C_cut`), the second `deploy_queue.sh`, `watchdog_loop.sh` and `compact_sessions.sh serve` were all stopped by pid after inspecting the process tree. No `ffmpeg`/`ffprobe`/`fpcalc` was running at the time and the container finished its in-flight merge and went idle, so **nothing was killed mid-write.** C_cut stopped at item **55 of 179** -- past err_150 at item 52, so T4 completed before the stop and nothing was lost to it. **The kill ORDER was wrong**: the Lead signalled `deploy_queue.sh` (the sweep's parent) first, which orphans the sweep rather than stopping it; it came out clean only because all signals landed inside one second with no ffmpeg mid-file. Correct order for any future teardown: watchdog, then `full_test.sh`, then SIGTERM the python so it completes its `batch_state` write, then the queue. | **Do not restart any of these.** The restart is a new basis, not a resume. |
| `full_test.sh` (merge sweep) | Agent 3 | 2026-08-30 ~17:0xZ | T3 runs the 23 files SERIALLY against the container's one fusion worker. Two concurrent `batch_verify` runs both POST /fusion and both snapshot the output directory, so each attributes the other's new files to its own submission — and they read-modify-write one `batch_state.json`, the race that lost 33 verdicts | T3's 23 files have finished |
| `correction_loop.sh` | Agent 3 (implicitly) | 2026-08-30 ~17:0xZ | submits merges to the same serial fusion worker T3 is driving; restarting it contends for the container mid-run | with the sweep |
| `hourly_deploy.sh` | Agent 3 | 2026-08-30 ~17:15Z | dev-AI HEAD had moved ahead of the container, so its next hourly tick would have deployed, killed the batch and recreated the container **mid-merge** during the T3 serial run — booking results against a build that never produced them | T3's 23 files have finished |

Delete the row when the service is back up. An empty table is the normal state.

## 👥 Agent roster — DISBANDED 2026-09-02

**All four agent sessions were destroyed on the owner's instruction after their
handovers landed.** Nothing was unlanded and no worktree was dirty when they were
killed — verified per branch before signalling. **Do not respawn from the table
below**; it is kept only so the restart can see how the team was arranged. The
handovers in `task_archive/20260902-HANDOVER-*.md` and `RESTART.MD` are what
survives.

## 👥 Agent roster (historical) — who *was* alive

Hand-maintained; the generated block below says who *is* alive. **The wake
procedure is the diff between the two** — see `AGENT.MD` → "Waking the team
after a usage limit". A session missing here has no owner; a session listed here
and absent below is dead and needs respawning, not messaging.

| # | Session & remote-control name | Owns | Brief | Restart |
| ---: | :--- | :--- | :--- | :--- |
| 1 | `vmsam-agent-c-cut` | C_cut, structural edits, chimeric assembly | `briefs/t2-verify.txt` | `spawn_agent.sh c-cut <brief> --resume` |
| 2 | `vmsam-agent-d-trim` | D_trim, B_fps, E_corr, A_no_lang | `briefs/d-trim.txt` | `spawn_agent.sh d-trim <brief> --resume` |
| 3 | `vmsam-agent-ci` | test queue, deployment loop, suite, bug hunting | `briefs/ci.txt` | `spawn_agent.sh ci <brief> --resume` |
| 4 | `vmsam-agent-forensics` | evidence — `src/diagnostics/`, `anonymise_reports.py` | `briefs/forensics.txt` | `spawn_agent.sh forensics <brief> --resume` |

Briefs live in `/home/vmsam/src/claude_agents/briefs/`, the spawner beside them
— **outside the repository**, so a container recreate does not take them with it.
`--resume` continues that agent's own history; **never `--continue`**, which
resumes the Lead's conversation instead because history is keyed by directory.

## 📋 Current assignments — owner decisions of 2026-08-31T15:40Z

**TWO OWNER RULINGS, both acted on:**

1. **The cut-point peak filter goes ON by default.** Evidence: 189 of 546
   boundaries unsupported by their own windows; 150 of 159 agreement with a
   corrected re-scan on the only reproducible subset; **7 files recovered,
   0 regressions** across the 49 that can exercise it. It adds no constant —
   `CHANGE_POINT_MIN_PEAK` has one home and the pipeline already applies it at
   the tail. `VMSAM_CUT_POINT_PEAK_FILTER=false` restores the old behaviour and
   is how every pre-decision table can be reproduced.
2. **The rebuild campaign is STOPPED.** All 14 refusals reproduce on HEAD with
   byte-identical fidelity values, and the comparator found **0 of 18 files both
   measurable and attributable**, with the 12 `unchanged` confirmed
   **bit-identical by packet MD5** — the re-merge did not touch the audio.

| Agent | Now | Then |
| :--- | :--- | :--- |
| 1 `c-cut` | flip the filter default; understand why `test_17_tail_plateau` fails with it on | err_117 |
| 2 `d-trim` | find where a *successful* merge records its delay vectors, then run the `sizePoint` arithmetic over the 27 | MD5 census (deferred) |
| 3 `ci` | **stage** the 113-file C_cut run — batch list, queue path, ordering, wall-clock estimate; do not start it | start it once the filter flip lands |
| 4 `forensics` | the `Unmergeable_Verified` criterion for the different-recording class | — |

**Next campaign: the 119 never-processed files** — all C_cut, artefacts on disk,
folders present, episodes in range. **Nothing is structurally blocked**; they
were never run for a scheduling reason. Master confirmed for 14; the other 99
need an enqueue to resolve, 87 distinct (folder, episode) pairs across 16
folders.

**119 never-processed, of which 113 carry a cut-point record** and 6 do not
(err_694–699). The two numbers being equal is a coincidence and was briefly
propagated as one figure.

**RESCAN FIRST, then merge — authorised 2026-08-31.** All 387 change points in
those 113 records are pre-fix: `lo_s`, `hi_s`, `refine_iters`, `located` and
`on_bracket_edge` are absent from every one, so they cannot be screened for the
edge-convergence defect at all — the field that identifies it is the field the
defective scanner did not write. Costed from recorded elapsed times: **rescan
62 min against a 9.8 h merge run, 11%**. Rescanning only the suspects is not a
smaller job, it is an impossible one.

Every file in the run records `samples` **per language** — err_214's figure was
measured on one language and understated its drift by half.

**Mission counts, corrected.** `Resolved` was reported as 71 until
`sync_verdict`'s 50 ms threshold was found being counted against the mission's
15 ms bar. The band was then probed per file against each master's own frame
duration: **22 genuine desync, 5 undetermined at the instrument's 10 ms
resolution, 0 correctly snapped.** The quantisation escape hatch rescues none of
them.

## 🎚️ Case 3 — speed relations: the owner's audio and subtitle policy

Ruled 2026-08-31, after measurements contradicted the first version of the rule.

**The first rule was "reject resampled audio on non-PAL stretches", on
artefact and pitch-degradation grounds. Both premises measured false:**

    asetrate precision, PAL 1.042708   0.7 ppm  = 1.0 ms over a 24 min episode
    asetrate precision, NTSC 1.001001  0.1 ppm
    a deliberately hostile 1.0173      5.9 ppm  = 8.5 ms
    -> every ratio inside 8.5 ms; the non-PAL ones are the MOST accurate

    pitch shift, PAL   4.27%  = +72.4 cents   <- the rule PERMITTED this
    pitch shift, NTSC  0.10%  = +1.7  cents   <- the rule REJECTED this

A factor of forty in the direction nobody would choose. The owner's answer was
to find a method that does not degrade the audio, and one exists — three do,
already in `merge_video_prebuild.py`, with `atempo` and `rubberband` both
present in the configured ffmpeg (9.0.1, `--enable-librubberband`).

**THE POLICY:**

1. **Master audio is always preferred, always kept, never stretched.**
2. **Candidate audio whose interpretation matches the master's is REMOVED, not
   stretched** — the master already carries that content. The dub gate's
   `same=True` at the 0.90 fidelity bar is the test.
3. **Distinct candidate audio is kept and stretched**, method by relation kind:
   **PAL → `asetrate`**, because the original PAL speedup raised pitch too, so
   this is the *faithful inverse* and `atempo` would leave those 72 cents
   permanently shifted; **non-PAL → `rubberband`**, falling back to `atempo`,
   because no pitch shift was applied at source and none should be introduced.
   Check the running build for the filter rather than assuming.
4. **If no pitch-preserving method is available, remove the candidate audio.**
5. **Subtitles are always retimed and kept** — retiming is lossless and
   `-itsscale:s` is already independent of audio resampling.
6. **Every stretched track is TAGGED**, following the existing `CHIMERIC=`
   precedent: method, factor, kind, and whether pitch was shifted or preserved
   — "resampled" alone cannot distinguish `asetrate` from `rubberband`, and
   those are different facts about the audio. **The tag must be read back off
   the finished output**, since `mkvmerge` runs after ffmpeg and silently drops
   custom per-stream metadata.

**Known gap, recorded rather than closed**: `speed_factor_applied` is written
only when `detect_pal_relation` returns a factor, and it returns one only for
PAL-shaped duration ratios — so **no table in this project can count a non-PAL
relation**, because the recording path excludes them by construction. Six NTSC
files (err_29, 31, 33, 35, 37, 113, all 1.0010 within 0.00015% of each other)
are real, measured out of band, and invisible to every table.

## ✅ Phase 1 complete — every file in the corpus has been attempted

**315 of 315 carry a `batch_state` row. The never-processed bucket is empty**,
for the first time in this mission.

**Final shape, by GROUP rather than by file** — episodes of one series are one
observation with a large denominator, not N findings:

    repair_attempted_rejected                     47 files /  7 groups
    repair_declined_record_not_piecewise          30 files /  4 groups  (27 = 1 series)
    merged                                        23 files /  9 groups
    merged_unverifiable_candidate_only_languages  13 files /  4 groups
    err_694-699                                    6 files /  1 group

**THE PHASE-2 DECISION, in its honest form.** The negative half is strong:
**47 files across SEVEN independent series-groups** where repair ran and the
chimeric build **rejected itself**. Seven groups agreeing is a result, and no
rescan changes any of them.

The recovery half looks like 30 files, but **27 of the 30 are one series** plus
three singletons. So it is one measurement with a large denominator — **if the
fresh scan finds the same non-piecewise structure there, the recovery population
is 3, not 30.** That is the number to decide on.

**THE CAVEAT THAT LEADS THE REPORT.** Step 2 verifies the `*.chimeric.mkv`
**intermediate**, not the published file. Every Step-2 pass in this project is a
statement about a file that does not ship. The passes are not false — the
intermediate really was in sync — they are **silent about the artefact**.

**The six newest files carry three qualifiers of their own**: they had **no
cut-point record in any table**, so structural repair was unavailable **by
construction** — which is not the same result as repair attempted and
self-rejected, and the seven-group negative depends on that distinction. They
ran on a different build from the 119. And they were invisible because they were
in the errors table but in **neither** the private mapping **nor**
`classified.json`; regenerating the mapping was necessary and not sufficient.

**Two follow-ups, scoped and not started:**
1. **`classified.json` has no producer anywhere in this tree** and **seven
   consumers** — `batch_verify`, `master_summary`, `verify_no_output`,
   `build_state`, `step2_bar_audit`, `cross_series_check`,
   `unverifiable_reason_scan`. It was written once and never regenerated.
2. **`src/diagnostics/tools/corpus_delta.py` already exists** and asks *"has
   anything changed under us?"* — the standing reconcile invariant 75 calls for.
   **Run it rather than rebuild it.**

<!-- LIVE STATE: generated by src/tools_batch/refresh_task_state.py -- do not hand-edit -->

## Live state — regenerated hourly, do not hand-edit

_Generated 2026-09-02T13:48Z · 9 processes · load 3.85_

| Job | Alive | Running for |
| :--- | :--- | ---: |
| merge sweep | — | — |
| watchdog | — | — |
| hourly deploy | — | — |
| redeploy+sweep | — | — |
| error correction | — | — |
| gccphat sweep | — | — |
| deploy queue | — | — |
| agent sessions | — | — |
| session compactor | — | — |

| Production coverage | |
| :--- | ---: |
| ever run in the container | 196 of 315 |
| **never run at all** | **119** |
| of those, still `pending` | 96 |

> ⚠️ **Rostered but not running: `vmsam-agent-c-cut`, `vmsam-agent-ci`, `vmsam-agent-d-trim`, `vmsam-agent-forensics`** — a message will not reach a dead session. Respawn from the roster's restart command (it carries `--resume`).

| Agent | Role | Status | In progress | State file |
| ---: | :--- | :--- | ---: | ---: |
| 1 (C_cut) | C_cut and structural edits | idle -- context compacted, AGENT.MD/TASK.MD re-read, tree clean and branch landed; holding for dispatch | 0 | 4h36m ago |
| 2 | D_trim, B_fps and E_corr | READING C_cut artefacts (read-only). Claim settled in advance; watching for a flagged record. | 0 | 6h27m ago |
| 3 | testing, hourly deployment and watchdo | T4 CLOSED -- err_150 measured, fre in_sync 3.3 ms, matching Agent 1's C3 exactly; orphan proven stale by CHIMERIC tag presence. Free. Sweep still owns the container, so the err_211 shape batch stays queued behind the SWEEP, not behind T4. | 0 | 4h37m ago |
| forensics (Agent 4) | forensic audio/video inspection -- evi | HOLDING at Lead's instruction -- delay-ambiguity and fps censuses landed; awaiting next dispatch | 0 | 6h01m ago |

_These rows come from the state files the agents write themselves, not from the process table — an agent that is alive but wedged still shows its last self-report, which is why each age is printed. Compare the session list above against the roster to find a dead agent._

| Deployment | |
| :--- | :--- |
| HEAD | `0138157` |
| deployed | `e75c783` |
| dev gate | `True` |
| ⚠️ | container is behind HEAD — results from it prove nothing about HEAD |

| Mission | Count |
| :--- | ---: |
| Failed | 174 |
| Step 2 verified; no Step 1 record | 75 |
| Step 2 only; Step 1 could not measure | 32 |
| Merged, unverifiable | 31 |
| Out of sync | 1 |
| Not processed | 1 |
| MERGED_OUT_OF_SYNC | 1 |

| Side ledgers | Judged |
| :--- | ---: |
| external verdicts | 86 |
| no-output verdicts | 22 |
| correction state | 91 |

<!-- END LIVE STATE -->

## 🚦 Test queue — requests for Agent 3 (dev agents do not deploy)

| # | Requested by | What to test | Judge it by | Status |
| :-- | :--- | :--- | :--- | :--- |
| T1 | Agent 1, 2026-08-30 | `src/merge_video_chimeric.py` — the fpcalc same-recording gate on both gap-patch selectors, the shared-reference fallback, and the `CHIMERIC` stream tag. Uncommitted in the working tree; needs the Lead to commit by path before it can ship. | Success criterion 4. Re-run the C_cut class. Every audio stream of a chimeric output must carry a `CHIMERIC=` tag (`ffprobe -show_entries stream_tags=CHIMERIC`); no track may be patched from a source the log did not report as `same=True`; a build that logs `refusing to stitch` must fall back to the unrepaired candidate and **not** produce an output. Expect `err_196` to refuse (its only track fingerprints at 0.606 against the master's) and `err_220` / `err_15` to build. | deployed 3bd4694 — **gate CONFIRMED LIVE**, build blocked elsewhere: the dub gate fired in the real path (`same=True fidelity=0.9876` / `0.9890`) and the chimeric build completed, but `verify_chimera` rejected it on **one tail checkpoint of five** (2629.9 ms at peak 0.474; the other four read −0.4/0.3/0.2/0.5 ms at peaks 0.73–1.03). Not a dub-integrity failure. Needs the tail-plateau question, not a gate change. |
| T2 | Agent 1, 2026-08-30 | `src/merge_video_chimeric.py` — `drop_uncorroborated_tail` (a final plateau resting on one window below `VERIFY_MIN_PEAK` and below its predecessor is dropped and the predecessor extended) and the railed-residual guard in `verify_chimera` (a residual at 0.85 of the ±3000 ms representable range fails, and is never skipped as low-peak). Uncommitted; Lead commits by path. | Re-run the C_cut class. Error id 120 is the case: it must now BUILD and pass, with its 1405 s checkpoint reading near 0 ms at a high peak instead of 2629.9 ms at 0.474. Locally the corrected plan measures **0.8 ms at peak 1.002** against a 1.012 self-correlation ceiling. 28 of 180 C_cut records change plateau tables, so watch for any file that previously passed and now fails — that is the regression to report. No `CHIMERIC`/gate behaviour changed; T1's criteria still apply unchanged. | deployed badf272 (re-verify: the queue re-deployed it every 120 s until the marker was fixed; results from that window are untrustworthy) |
| T3 | Agent 2, 2026-08-30 | The two flag-gated features on their 21 files, which have never been through the container. `VMSAM_MULTITRACK_REFERENCE` (the reference is the master's uniquely default-flagged track in the correlation language, else the first) and `VMSAM_LANGUAGE_TAG_FALLBACK` (measure instead of raising when the pair shares no audio language tag and each side carries exactly one non-`und` track). Both now default ON under `VMSAM_MODE=test` via `tools._feature_default`, so no stack variable is needed — that was the blocker and it is gone. Nothing is opt-in in production: an explicit env value still wins in either direction. | Re-run the 21: error ids 6, 7, 10, 11, 14-24, 678-682, 684. **Expect to BUILD and verify in sync**: the 15 multitrack pairs that reduced to one constant delay locally — ten at +80.0 ms and five at 0.0 ms, spread 0.0 ms over three windows each — and the 6 `A_no_lang` pairs 678, 679, 680, 681, 682, 684, whose language-tag override measured -4430..-4470 ms with spread <= 20 ms and median peaks 0.90-0.96. **Expect to REFUSE, and a build from either is the finding**: error id 170 (piecewise -2010/-2020/-3020 ms, so a constant delay is wrong for it) and error id 349 (no window clears the 0.30 peak floor). Check the delay actually applied lands on a master frame boundary — 41.708 ms at 23.976 fps — per success criterion 1; a delay that measures well but does not sit on a frame is wrong. The language-tag override is in memory and covers the delay key only, so **no output may show a rewritten media file or a changed database row**. Compare against a run with both flags explicitly set to something other than their ON value: the 21 must then behave exactly as they do today, which is what proves the default is the only thing that changed. | deployed 5306582 |

| T4 | Agent 4, 2026-08-30 | **`err_150` — a merge that produced an output no ledger records.** Nothing in the merge code is under test here; the request is to make one file's run *recorded*. Reconciling the mirrored output tree found a 3.90 GB output whose path is derived by no `batch_state` row, written 2026-08-30 12:43:22 inside a gap between two deployed builds. It matches `err_150` by episode, and `err_150`'s ledger record reads `state: pending`, `container_outcome: None` — never processed. The output is nobody's recorded `merged_file`. Two known write-path mechanisms produce this and **neither is fixed**: `retest_ids.py` posts to `/fusion` and never writes `batch_state`, and `batch_state.json` has a single read-modify-write writer that has already lost 33 verdicts to a second one. Re-running rather than back-filling the ledger by hand is the whole point: a recorded run yields a `db_master`, an outcome and a `(git=)`-stamped log, none of which a hand-written entry produces. | Re-run `err_150` alone through the container and confirm `batch_state` gains a row for it. **Stated in advance so the result cannot be fitted afterwards:** it should **BUILD** — an output already exists, so the merge demonstrably completes — and it is one of the few files in this population that is genuinely **checkable**, because master and candidate both carry `fre` and `jpn`, giving a same-language reference. Contrast the eight `pending` files that returned `unverifiable`: all eight share `jpn` with their master, but the output's `jpn` track is the master's own. So for `err_150` expect a real number rather than another `unverifiable`, and if it comes back `unverifiable` that is the finding, not the expected case. Also compare the new output against the existing 3.90 GB one: if the sizes or stream layouts differ, the unrecorded run was made by different code and the orphan should be treated as stale rather than as evidence. | deployed e75c783 — after T3. **Expectation amended 2026-08-30**: Agent 3 found `batch_verify.py:282` reads `lang = item.get("lang_pick") or "jpn"`, chosen once and never retried, so an `unverifiable` result may mean only that the selection reached a language the candidate did not contribute. `err_150` carries `fre` and `jpn` on **both** sides, so with the selection fix in place a number should be available — but if it returns `unverifiable`, first check which language was actually compared before treating that as the finding. **Scope corrected 2026-09-02 by Agent 3, who wrote the amendment above:** it does NOT apply to `err_150`. Its ledger row, recorded 2026-09-01T00:22:35Z under build `afa72e7`, shows `languages_attempted ['jpn','fre']`, `lag_ms_by_language {}` and `selection_reason every_language_both_files_share_was_already_checked` — so the selection exhausted the pair rather than stopping at one. **Corrected 2026-09-02, same author, after reading the row itself rather than a summary of it:** the two per-language verdicts read `unverifiable`, not `indeterminate`, and each carries the note *external probe failed: `tools.software['ffmpeg']` is not configured; refusing to fall back to PATH for a measurement*. **Neither language was measured.** Both were *attempted* and both hit an instrument that was not configured, which is upstream of language selection entirely. The conclusion is unchanged and in fact stronger — the `batch_verify.py:282` selection amendment cannot explain this row, because no selection this file could have made would have produced a number. Read the re-run as three distinct outcomes, never two: samples present, the not-configured note, or a blank. A two-valued reading cannot tell a refusal to measure from a measurement that came back empty (invariant 157). **Pinned 2026-09-02T08:01:59Z, with the sweep at C_cut 40/179 and err_150 at item 52 — so the pin is dated by the sweep's own position, not only by a clock, with the instrument that will read the result — `reconcile_verdicts.evidence_class`, chosen now so it cannot be picked to fit the answer:** `recorded_utc 2026-09-01T00:22:35Z`, `build afa72e7587f1`, `outcome MERGED_UNVERIFIED`, `evidence_class "instrument fault, not a file verdict"`. The re-run has moved the row only when `recorded_utc` differs from that value; it has produced a *finding about the file* only when `evidence_class` stops saying instrument fault. Those are two separate questions and the second does not follow from the first.

**T4 ANSWERED 2026-09-02T08:50:12Z — both pinned questions came back yes.** `recorded_utc 2026-09-01T00:22:35Z → 2026-09-02T08:50:12Z`, `build afa72e7587f1 → e75c78386dd0`, `outcome MERGED_UNVERIFIED → MERGED_IN_SYNC`, and the pinned instrument moved from `"instrument fault, not a file verdict"` to `"multiple windows, agreeing (spread 2.7 ms)"`. **`fre` in_sync, max 3.3 ms** over three windows — 0.6 / −1.0 / −3.3 ms at peaks 0.991 / 0.990 / 0.969, `compared = output vs master file`, well inside the 15 ms bar. **This is the number T4 predicted in advance and it matches Agent 1's independent C3 reading exactly (fre in_sync, worst 3.3 ms)** — two instruments, separately arrived at, same value.

**The `jpn` leg is the control that makes the `fre` number mean something.** It returned `unverifiable` and said why: *the only `jpn` track is bit-identical to the master's; this compares the master with itself* — 0.0 ms at peak 1.000 across all three windows. So the multi-track trap in `DOMAIN_INVARIANTS` §the-two-verification-traps **fired and named itself** rather than passing silently, and it doubles as a negative control: a self-comparison here reads exactly 0.0/1.000, so `fre`'s 0.6/−1.0/−3.3 at peaks below 1.0 cannot be one. The eight other `pending` files this row was contrasted against fail on precisely this leg; err_150 is the case where the second language rescues it.

**The orphan is stale, and the mechanism is now known rather than inferred.** T4 asked to compare the new output against the unrecorded 3.90 GB one and to treat the orphan as stale if the layouts differ. Both were probed. **53 streams each, identical order, identical codecs, languages, titles, dispositions and `duration=1432.014000`; sizes differ by 1082 bytes on 3.9 GB.** Exactly one structural difference across all 53: **the new output carries `CHIMERIC` tags on 29 streams and the orphan carries none.** The orphan therefore predates T1's tagging work — different code, so stale, as the conditional anticipated. The tagging is also **discriminating rather than blanket**, which is the strongest available evidence it is correct: the one audio track that came from the master untouched (`jpn`) is the one track with no tag.

**Not marked `step2_passed` here.** The queue marks rows `deployed`, never `passed`; the Lead judges the output against the request's criteria. Reported, not graded. T4's pre-registered finding therefore stands on its own merits and is not an artefact of the `batch_verify.py:282` selection bug. The amendment is retained for the eight other `pending` files it was written about and withdrawn for this one. Note also that the run T4 asks for already happened under `afa72e7`, an ancestor of the image it was served on, so the remaining value of a re-run is a build stamp, not a row. |
### T1 — what to look for in the job log

The gate prints one line per decision:

```
chimeric: dub gate a3 vs master a3: same=True fidelity=0.9913
chimeric: dub gate a0 vs master a2: same=False fidelity=0.606
```

Cost measured locally: **1.6–5.8 s per merge, 1–5 fingerprint decisions**, on
candidates carrying 1–11 audio streams. If a job shows more than ~10 decisions
or more than ~30 s in the gate, the per-pair cache is not working — say so
rather than raising the budget.

## Standing briefs — what was asked, and where it stands

Verified against the repository, not recalled. Every "done" row names the
evidence you can re-check.

| Brief | Status | Evidence |
| :--- | :--- | :--- |
| Deploy on GitHub Actions build state, never a static sleep | **done** | `src/tools_batch/ci_ready.py`; AGENT.MD "Ask CI, do not sleep". Measured builds take 1m09s–1m49s, not the documented ~10 min |
| Agent 3 as an always-on CI/CD queue service | **done** | `deploy_queue.sh` running; serves the queue in this file |
| `fpcalc` fidelity ≥ 0.90 gate on both patch selectors | **done** | `DubGate` + `pick_patch_source`; bar sits in an empty band — same recording 0.904–0.993, different 0.563–0.847 over 217 pairs |
| `chimeric` stream metadata tag on every altered track | **done** | `describe_chimeric_track`; `ffprobe -show_entries stream_tags=CHIMERIC` |
| Dev-mode logging across the 9 merge-path modules | **done** | `tools.dev` only, 0 `DEV_MODE`; commit "dev-mode diagnostics across the nine merge-path modules" |
| Structured dev logging inside `merge_video_chimeric.py` | **done** | `_dev()` helper; timeline, plateaus, gaps, patch reasons, per-checkpoint verification |
| Tail checkpoint at 1405 s — diagnose before fixing | **done** | Verdict **(A)**: genuinely misaligned. Refuted (B) with a self-correlation ceiling — the failing checkpoint reached 47 % of achievable while four passing ones reached 89–100 % |
| Tail fix + no re-judgment drift | **done** | `drop_uncorroborated_tail` (28 of 180 records, A/B-correct 10/10); `VERIFY_RAIL_FRACTION` a measured no-op on all 481 recorded checkpoints |
| Master Summary Table in `AGENT_NOTES.MD` | **done** | **315** rows, generated by `master_summary.py`, keyed by error id |
| **T2 — container verification of the tail fix** | **OPEN** | its row is marked `deployed` but **flagged untrustworthy**: a queue defect re-deployed it every 120 s for ~20 min while ten sweeps wrote one state file. Job 9 is re-running it |
| **100 % of `STATE.json` at `step2_passed` / `unmergeable_verified`** | **OPEN** | 150 of 309 still `pending` — the mission |

Everything above the last two rows is shipped. **The definition of done rests on
those two**, and both need container runs rather than more code.

### Case 3 — non-PAL speed stretch: what is built, and what is deliberately not

Built and flag-gated, default off (`speed_relation_kind`, `method_for_relation`,
`build_ffmpeg_command(include_audio=…, relation_kind=…)`, `audio_track_plan`):

* the relation is classified from the **measured** factor, never from the gross
  duration ratio — `detect_pal_relation` returns a factor only when the ratio is
  already PAL-shaped, so a genuine non-PAL relation never reached a measurement;
* **PAL → `asetrate`**, which is the *faithful inverse*: the original speedup
  raised pitch too, so restoring the timeline this way restores the pitch. A
  pitch-preserving stretch on a PAL file would leave it ~72 cents out forever;
* **non-PAL → `rubberband`**, falling back to `atempo`, checked in the running
  build rather than assumed. No pitch-preserving method available → drop the
  audio, keep the subtitles;
* subtitles are **always** retimed and kept — `-itsscale:s` is lossless, there
  is no signal to degrade, and it is worth most where a track is distinct;
* every stretched track carries `VMSAM_RESAMPLED={method,factor,kind,pitch}`,
  verified end to end through ffmpeg **and** `mkvmerge`, plus an appended title
  marker. `pitch` is derived from `method` so the two cannot drift apart.

**Rule 2 does not delete on the audio test, by owner's ruling.** Fidelity cannot
separate a duplicate from a different regional dub: labelled dubs run
0.833–0.9922 and duplicates 0.927–0.9894, so the dub range reaches *higher* and
no threshold exists. At the shipped 0.90 gate **18 of 19 known distinct dubs
clear the bar**. Removal is therefore gated on metadata positively identifying a
duplicate; metadata silence means **keep**, because an absent label identifies an
unlabelled file and not a duplicate.

**Non-dialogue tracks are excluded by DISPOSITION FLAG, never by title.**
err_349's master carries a jpn track titled `[ Audio Guide ]` with
`visual_impaired` set; comparing a performance against a narration of it produced
a "the master is defective" reading that wrongly disqualified the file. Titles
are not dispositions.

**UNBUILT, NOT UNCONSIDERED — what would actually separate a dub from a
duplicate.** Two regional dubs share their entire music-and-effects bed and
differ only in dialogue, which is a small fraction of what Chromaprint
fingerprints — that is *why* the statistic cannot carry the distinction. A
measurement isolating **dialogue** from the shared bed would: centre-channel
extraction on 5.1, or source separation. Nothing in this repository does that
today and it was not built for Case 3. "We cannot tell a dub from a duplicate"
is a permanent limit **only until someone measures the right thing**.

## Active jobs (registry — every launch is recorded here BEFORE it starts)

**Audited against ground truth 2026-08-30.** Six entries said `running` for work
that had finished or been killed. A stale registry is worse than none: an
unreported death looks exactly like progress.

### 1. Main merge batch — `full_test.sh`
- **What & why**: drive all **315** error files through `POST /fusion` and judge the output. Validation Step 2.
- **Parameters**: classes in order `E_corr, B_fps, D_trim, A_no_lang, C_cut`, then failed-only sweeps; queue wait `timeout=5400 s`, poll 15 s; sync bar 50 ms.
- **Where**: `src/tools_batch/full_test.sh` → `batch_verify.py`; state `/config/output/vmsam_agent/batch_state.json`; outputs under `<TEST_OUTPUT>`
- **Check**: `python3 src/tools_batch/master_summary.py` (NOT `batch_verify.py report`, which does not merge the external verdicts)
- **Restart** (resumable): `cd src/tools_batch && setsid nohup ./full_test.sh > /tmp/fulltest.log 2>&1 < /dev/null &`
- **Status**: `running`

### 2. Watchdog — `watchdog_loop.sh`
- **What & why**: the AGENT.MD 1-hour kill rule, with a py-spy dump before every kill so a timeout names the frame it is stuck in.
- **Where**: `src/tools_batch/watchdog_loop.sh`; diagnostics `/config/output/vmsam_agent/timeouts/`
- **Restart**: `cd src/tools_batch && setsid nohup ./watchdog_loop.sh > /tmp/watchdog.log 2>&1 < /dev/null &`
- **Status**: `running`

### 3. Hourly deploy — `hourly_deploy.sh`
- **What & why**: ship `dev-AI` HEAD when it differs from the deployed SHA.
- **Parameters**: 3600 s period; **fires only when HEAD is ≥ 780 s old**, because CI needs ~10 min and an early hook re-pulls the previous image while looking successful; prints `DEPLOY DID NOT LAND` when it fails.
- **Restart**: `cd src/tools_batch && setsid nohup ./hourly_deploy.sh > /tmp/hourly_deploy.log 2>&1 < /dev/null &`
- **Status**: `running`

### 4. Error-file correction run — `error_correct.py` + agent
- **What & why**: close the loop per file — read the log, pick the repair the evidence supports, build, verify, apply the residual back onto the plateau that produced it, rebuild. Plus the 6 `out_of_sync` and the cross-language verification question.
- **Parameters**: resumable, state written after every file; 5-probe cap; **`apply` on the retirement manifest is NOT authorised** — the dry run stands at 1 of 128 and the owner has not approved deletion.
- **Where**: `src/tools_batch/error_correct.py`; state `/config/output/vmsam_agent/correction_state.json`; log `/tmp/errcorrect.log`
- **Check**: `python3 src/tools_batch/error_correct.py report`
- **Restart**: `cd src/tools_batch && setsid nohup python3 error_correct.py run > /tmp/errcorrect.log 2>&1 < /dev/null &`
- **Scope**: 89 of the 129 logs carry a strategy this loop can execute (chimeric
  plateau, or a language-tag override); the other 40 are handled by repair paths
  that already run inside the merge, and the loop does not claim them.
- **Progress**: the first 27 verdicts were **withdrawn and re-measured** after
  three defects were found in how they were judged — see
  `src/task_archive/20260830-0230Z-verification-cross-language.md`. Previous
  state kept as `correction_state.pre-reference-fix.*.json` beside the live one.
  At 49 of 89: 41 `in_sync` (worst 10.4 ms, median 0.5 ms), 6
  `mergeable_after_language_override`, 1 `out_of_sync`, 1 `unverifiable`.
- **Do not poll this job.** It is resumable and writes state after every file;
  read `error_correct.py report` when you next act. Poll waiters accumulated to
  1200+ processes once already.
- **Status**: `superseded by entry 6` — this row described one BATCH of `error_correct.py`, which judges a bounded set and exits. Entry 6 is the loop that drives it to completion and is the one to restart.
  `CORRECTION_DONE` after a pass that judged nothing new. Final: **89 of 89
  claimable logs judged** (77 `in_sync`, 6 `mergeable_after_language_override`,
  4 `out_of_sync`, 2 `unverifiable`). The 40 unclaimed logs are handled by repair
  paths inside the merge. Needs moving out of this table with an archive.

### 5. Per-cut GCC-PHAT head-to-head — agent
- **What & why**: the measurement half is committed and green; the comparison against the incumbent is not done. The verdict that matters is whether the two methods AGREE A FILE IS STRUCTURALLY EDITED, not sub-millisecond agreement.
- **Parameters**: control corpus 9 pairs; error corpus = every log whose terminal mechanism is `REC_DECLINE_NONUNIFORM`, **105 of 129**, taken from `error_log_triage.survey()` and named by the `GET /errors` database id. Three measurements per pair on the SAME cuts: `incumbent` (Chromaprint `correlate` on the extracted cuts), `gcc_files` (GCC-PHAT on the same extracted bytes), `gcc_native` (GCC-PHAT reading the sources directly). Both judged by `is_within_one_fingerprint_frame`, unchanged.
- **Where**: harness `src/tools_batch/gccphat_compare.py`, reporting + independent arbiter `src/tools_batch/gccphat_report.py`; results `/config/output/vmsam_agent/gccphat_cuts/`, arbitration `.../gccphat_cuts/arbiter/`
- **Check**: `python3 src/tools_batch/gccphat_report.py summary`
- **Restart** (resumable, one JSON per pair): `cd src/tools_batch && setsid nohup bash -c 'python3 -u gccphat_compare.py run --corpus control --force; python3 -u gccphat_compare.py run --corpus errors' > /tmp/gccphat_compare.log 2>&1 < /dev/null &`
- **Correction on record**: `decode_to_raw` built its ffmpeg command with no `-map`, so the direct-from-source variant decoded whatever track ffmpeg defaults to. On a control release carrying French first and Japanese second, that correlated French against the master's Japanese and returned a confident −933 ms at peak 0.61 where the language-matched answer is −955 ms at peak 0.98 — and nothing in the output said a different track had been read. `decode_to_raw`/`measure_cut_vector` now take a `StreamOrder`, the harness passes the one `extract_audio_in_part` uses, and `test_09` covers it. Production was never affected: it calls `measure_cut_vector_from_files`, which reads the pipeline's own per-cut files.
- **Wiring correction on record**: `test_if_constant_good_delay_new_method` let `is_within_one_fingerprint_frame` overrule GCC-PHAT's own `stepped` verdict. Measured on error-102: a real 167 ms edit is 1.34 fingerprint frames, lands on two adjacent frames, and the gate calls it constant -- so the merge would have shipped one delay and left the last third of the episode 167 ms out. `describe_cut_vector` now reports `step_confirmed` (>= 2 cuts either side, both sides flat within tolerance) and the call site declines on a confirmed step, while still letting the gate decide an unconfirmed one. Six tests in `test_09`.
- **Rate diagnosis (AGENT.MD 1-hour rule)**: the batch is 105 bounded sub-jobs, each with a 900 s deadline, not one long job. Per pair: 70 s on the control corpus, ~180 s on the error corpus, and the box is at **load average 45** because the main merge sweep and the correction loop run alongside. The cost is the shared cut extraction (68 s median unloaded), which both methods need for an apples-to-apples comparison. Expected finish ~05:00Z; every pair is written as its own JSON, so a kill costs one pair.
- **Reporting**: `python3 src/tools_batch/gccphat_report.py summary --corpus errors` regenerates every table from the per-pair JSONs at any time, so the sweep needs **no watcher** -- do not arm one. `percut --pair <id>` prints one pair side by side; `arbitrate` runs the independent correlator on the disagreements; `arbiter-report` tabulates who was right.
- **Known blind spot**: no `sizePoint = 138 ms` pair is exercised, and none will be -- the three that exist are all one show and are named in `20260829-0100Z-gate-regression-test.md`. Run those three explicitly to close it.
- **Status**: `done` — 105/105 error pairs measured; the agent completed and reported. Verdict recorded here rather than left in a message: **GCC-PHAT cannot replace the incumbent** — its lag axis is ±5 s and 8 of 16 error pairs have true offsets of 23–212 s, which Chromaprint has no ceiling on.

### 6. Correction loop — `correction_loop.sh`
- **What & why**: `error_correct.py` judges a bounded batch and exits by design, so one invocation is one batch, not the job. This drives it to completion and stops when a pass judges nothing new.
- **Where**: `src/tools_batch/correction_loop.sh`; state `/config/output/vmsam_agent/correction_state.json`; log `/tmp/errcorrect.log`
- **Check**: `python3 src/tools_batch/error_correct.py report`
- **Restart** (resumable): `cd src/tools_batch && setsid nohup ./correction_loop.sh > /tmp/errcorrect.log 2>&1 < /dev/null &`
- **Status**: `running` — restarted 14:27Z. The 07:58Z 'finished' was one BATCH, not the job: error_correct.py judges a bounded batch and exits, so 89 of 131 were judged and 14 were still waiting. That is what this loop exists to drive.; stopped on
  "no progress on pass 2", which is its designed exit. Re-running it judges
  nothing new until new logs appear.

### 7. Deep forensic search — **CLOSED 2026-08-30, do not resume as-is**
Archived: `src/task_archive/20260830-0830Z-deep-forensic-search-verdict.md`.
Costed and scoped down rather than resumed. The count was **126 done / 183
remaining** (the directory's 128 entries include a `README.md` and `tools/`).

- **Why not resumed**: compute is negligible — **2.7 h wall at 2 slots, 6.2
  CPU-hours, 240 MB**, measured, not guessed. The binding cost is the **agent
  budget**: this is `agent()` per file at `effort: high`, and 183 more reports
  ≈ **1.5 session windows**, during which no other agent runs.
- **Why it is not pure waste either**: 115 of 126 (91 %) only confirmed the
  first-pass bucket, but **11 (8.7 %) changed the answer**, and it is the *only*
  producer of `unmergeable_verified` evidence in the project.
- **Scope of the 183**: 82 already `step1/step2_passed` (no value), 21 answered
  mechanically by the script below, 4 already answered by the correction loop,
  leaving **76 genuinely open** (74 `C_cut`). A targeted 76-file pass is 0.6 of
  a session window and 1.1 h at 2 slots — **queued behind the merge sweep, not
  urgent.**
- **Done instead**: `src/diagnostics/tools/cross_series_check.py` — resumable,
  2-slot pool, one JSON per pair, self-calibrating (aborts unless a file against
  itself measures peak 1.0 at 0 ms). 26 pairs in **3 minutes**; 23 proved
  `different_programme`, agreeing 5 of 5 with the deep reports that had measured
  the same pairs by hand.
- **Check / re-run**: `python3 src/diagnostics/tools/cross_series_check.py --report`

### 8. Redeploy
- **What & why**: the container runs `65f1530` and reports `dev=False`. HEAD carries the `tools.dev` fix (uvicorn's `workers=5` public instance never saw `tools.dev` set in `__main__`). Until this ships, **the episode logging cannot appear**.
- **Where**: `src/tools_batch/redeploy_and_sweep.sh` — waits out the CI window, stops the batch, deploys, verifies `dev`, restarts the sweep; refuses to sweep if the deploy does not land.
- **Restart**: `cd src/tools_batch && setsid nohup ./redeploy_and_sweep.sh > /tmp/redeploy.log 2>&1 < /dev/null &`
- **Status**: `pending — owner's call`

### 9. C_cut agent — spawned session `vmsam-agent-c-cut`
- **What & why**: re-verify **T2** cleanly and then work the 101 pending `C_cut` files. T2's row is marked `deployed badf272` but **flagged untrustworthy**: a queue defect re-deployed it every 120 s for ~20 minutes while ten concurrent sweeps wrote one state file, so nothing from that window can be believed.
- **Form**: a **spawned Claude session**, not an in-process sub-agent — the owner asked twice to be able to watch and steer agents, and an in-process agent has no pid for `ps` or `tmux` to find. See `AGENT.MD` → "Three ways to run an agent".
- **Where**: session and remote-control name `vmsam-agent-c-cut`; brief `/home/vmsam/src/claude_agents/briefs/t2-verify.txt`
- **Watch**: `tmux attach -t vmsam-agent-c-cut` (detach `Ctrl-b d`) · read without attaching: `tmux capture-pane -t vmsam-agent-c-cut -p`
- **Restart**: `/home/vmsam/src/claude_agents/spawn_agent.sh c-cut /home/vmsam/src/claude_agents/briefs/t2-verify.txt` — add `--resume` to continue its own history rather than start fresh. **Never `--continue`**: history is per directory and it would resume the LEAD's conversation.
- **Acceptance**: error id 120 builds with its 1405 s checkpoint near 0 ms, not 2629.9 ms; every audio stream of a chimeric output carries a `CHIMERIC=` tag; no track patched from a source not logged `same=True`. **The regression to watch is any file that previously passed and now fails** — `drop_uncorroborated_tail` changed 28 of 180 plateau tables.
- **Status**: `running`

### 10. Five-hourly wake-up — cron `a4f59635`
- **What & why**: the usage window resets every ~5 h and agents die mid-task when it closes. Fires the come-back procedure so nothing sits dead until the owner notices.
- **Parameters**: `13 2,7,12,17,22 * * *` local. **Session-only — it dies with this Claude session and auto-expires after 7 days.**
- **Status**: `scheduled`

### 11. D_trim/B_fps/E_corr agent — spawned session `vmsam-agent-d-trim`
- **What & why**: the **49 pending files nobody was working** — E_corr 24, B_fps 15, A_no_lang 6, D_trim 4. Agent 2's slot had been idle ~6 h. First job is the **28 files with a step profile and no plateau record**, which nothing can repair until the cut-point scan runs.
- **Form**: a **spawned Claude session** — the owner asked to watch and steer every agent, and an in-process agent has no pid. See `AGENT.MD` → "Three ways to run an agent".
- **Where**: session and remote-control name `vmsam-agent-d-trim`; brief `/home/vmsam/src/claude_agents/briefs/d-trim.txt`
- **Watch**: `tmux attach -t vmsam-agent-d-trim` (detach `Ctrl-b d`) · read without attaching: `tmux capture-pane -t vmsam-agent-d-trim -p`
- **Restart**: `/home/vmsam/src/claude_agents/spawn_agent.sh d-trim /home/vmsam/src/claude_agents/briefs/d-trim.txt` — add `--resume` for its own history. **Never `--continue`**.
- **Acceptance**: a plateau record for each of the 19 handed-over ids; the B_fps record measured against a master the **database** agrees with (the current one is the inverse PAL ratio misfiled as 8.26 % from theory); the 21 flag-gated files through the container.
- **Status**: `running`

### 12. Forensic evidence agent — spawned session `vmsam-agent-forensics`
- **What & why**: `unmergeable_verified` is the **only state that closes a file without a merge**, so it must carry our best evidence and currently carries our weakest — 6 of the first 7 such marks were later **refuted**. Also audits whether the 56 `step2_passed` files actually meet the bar, and whether the 435 anonymised reports kept their numbers.
- **Form**: spawned Claude session. **Evidence, not repair** — it reports defects in other agents' work and does not edit their files.
- **Where**: session and remote-control name `vmsam-agent-forensics`; brief `/home/vmsam/src/claude_agents/briefs/forensics.txt`
- **Watch**: `tmux attach -t vmsam-agent-forensics` · `tmux capture-pane -t vmsam-agent-forensics -p`
- **Restart**: `/home/vmsam/src/claude_agents/spawn_agent.sh forensics /home/vmsam/src/claude_agents/briefs/forensics.txt`
- **Acceptance**: a per-id verdict on all 26, distinguishing **never mergeable** from **merely different** (a sequel filed into the wrong folder is a cataloguing defect, not an unmergeable pair); a sampled figure for how many `step2_passed` files hold up.
- **Status**: `running`

### 13. Test orchestration and bug hunting — spawned session `vmsam-agent-ci`
- **What & why**: Agent 3's work was being done by **scripts with no judgement**. The suite is 551 checks and reported green, which is the exact condition `DOMAIN_INVARIANTS` §12 warns about — its first job is to find the checks that **cannot fail** (two are known: a source-text assertion, and a closure checker that once reported `intact`/exit 0 with the protected root deleted).
- **Form**: spawned Claude session. It **supervises** the running host loops — `deploy_queue.sh`, `hourly_deploy.sh`, `watchdog_loop.sh`, `correction_loop.sh` — and must not duplicate them.
- **Where**: session and remote-control name `vmsam-agent-ci`; brief `/home/vmsam/src/claude_agents/briefs/ci.txt`
- **Watch**: `tmux attach -t vmsam-agent-ci` · `tmux capture-pane -t vmsam-agent-ci -p`
- **Restart**: `/home/vmsam/src/claude_agents/spawn_agent.sh ci /home/vmsam/src/claude_agents/briefs/ci.txt`
- **Acceptance**: each unfailable check **shown** to pass on broken code before it is repaired; the queue contract **shown** failing before it is called fixed; the two known merge-path bugs quantified on real data (they sit inside the protected closure, so the fix is the owner's call, not the agent's).
- **Status**: `running`

### 14. Session compaction on a timer — `compact_sessions.sh`
- **What & why**: the owner's standing rule is that **every session is compacted about every five hours**, so a long-running agent stops working from a summary of a summary. It cannot be done by an agent to itself — `/compact` is typed into a pane, not a tool any session holds — so it was passed agent-to-agent and **three sessions refused in a row**. They were right to: a peer saying "the owner asked for this" is unverifiable from inside another session, and a request declined elsewhere and then handed onward is the shape of routing around a refusal. The fix is not to convince agents but to **remove the agent from the loop**.
- **Form**: a host loop, `setsid`/`nohup`-detached, reparented to init and ignoring `SIGHUP` — so it outlives the session that started it, which a service enforcing a rule about sessions has to.
- **Parameters**: `INTERVAL=18000` (5 h) · state `/tmp/.vmsam_compact_state` · log `/tmp/compact_sessions.log`
- **Restart**: `cd src/tools_batch && setsid nohup bash compact_sessions.sh serve > /dev/null 2>&1 < /dev/null &`
- **Inspect without sending anything**: `bash src/tools_batch/compact_sessions.sh --status` · **stop**: `pkill -f 'compact_sessions.sh serve'`
- **What it will not do**: it never sends to a **busy** pane (read from the pane's own `esc to interrupt` marker, not guessed from timing); it never sends the follow-up unless it has **observed** the compaction finish over consecutive identical captures; on timeout it logs `UNVERIFIED`, withholds the follow-up, and does **not** record the session as compacted, so the pane is retried rather than counted done for five hours.
- **Fired and verified end-to-end, not by exit code**: `07:29:07Z` sent to the Lead's pane, `07:36:28Z` compaction observed to finish and follow-up sent — a **7m21s** wait, which is what proves it releases on observation rather than on a timer. Confirmed by finding the follow-up text in the pane and seeing the session act on it. It then compacted `vmsam-agent-ci` at `07:41:58Z`/`07:44:14Z`, and correctly skipped the three busy panes.
- **Status**: `running` (pid 2839867 at 07:36:58Z)

### Finished jobs
Moved out of this list into `src/task_archive/` — one file per job, with the
parameters, the numbers, what was corrected and the decision taken. Read it
before re-running anything that may already have been answered.

| Archive file | Job |
| :--- | :--- |
| `20260829-0100Z-gate-regression-test.md` | adversarial test of the frame-space gate widening |
| `20260829-0200Z-residual-risks.md` | dead consensus stub + failure-path recovery cost |
| `20260829-0230Z-fusion-thinning.md` | moving all merge intelligence into `mergeVideo.py` |
| `20260829-0310Z-cut-point-mapping.md` | plateau/change-point tables (57 + 180 files) |
| `20260829-0310Z-pal-speed-analysis.md` | PAL factor recovery (35 triaged, 21 measured) |
| `20260829-1520Z-forensics-first-pass.md` | first-pass diagnostics, 309/309 |
| `20260830-0100Z-stage1-tests-rebuild.md` | rebuilding the Stage-1 suites the container wipe destroyed |
| `20260830-0130Z-error-log-lifecycle.md` | error-file triage, the staleness rule, and the correction loop |
| `20260830-0200Z-offset-benchmark.md` | offset-method RAM curve, incumbent baseline, where the RAM actually goes |
| `20260830-0215Z-verification-corrections.md` | two instrumentation bugs that misfiled 35 merges as failures |
| `20260830-0230Z-privacy-cleanup.md` | removing the media library from the repository and its history |
| `20260830-0132Z-branch-salvage.md` | the six abandoned ML/scene-detection branches: intent, verdict and reasoning for each. **Nothing ported** — one idea (visual sequence alignment with gaps) written up as a brief for an improvement agent. Read before restoring any of them. |
| `20260830-0230Z-verification-cross-language.md` | cross-language reads were condemning every chimeric build |
| `20260830-0245Z-task-sprawl.md` | 1212 of 1246 processes were poll waiters, and why it stayed invisible |
| `20260830-0430Z-visual-alignment.md` | pHash + scene-cut video alignment: phase-1 verdict (video is exact but measures a different quantity) and phase-2 plateau cross-check |
| `20260830-0840Z-multitrack-reference-and-language-tag.md` | the 17 multi-track reference conflicts (15 resolved, 2 correctly declined) and the measured language-tag fallback for the 6 `NO_COMMON_LANGUAGE` pairs. Both opt-in, both off by default; **Step 2 blocked until the container sets the two env vars** |
| `20260830-1030Z-refined-plateaus-reach-the-container.md` | why the 77 refined repairs reached no container, and the two fixes that let 34 of them through |
| `20260830-0830Z-deep-forensic-search-verdict.md` | deep search costed and retired as a blanket sweep; 26 files reached `unmergeable_verified`, and why the first pass's 7 could not be trusted |
| `20260830-2100Z-xcorr-impact-table.md` | Impact of replacing `_xcorr_offset` with an overlap-normalised coefficient |
| `20260830-2130Z-frame-relative-tolerance.md` | `ALIGN_TOLERANCE_MS` at 0.5 and 1.0 frames — corpus report |
| `20260830-2200Z-err20-desync.md` | err_20 — 180 ms desync, root-caused |
| `20260831-0000Z-per-language-delay-design.md` | One delay for every stream — incidence, mechanism, and a design to decide |
| `20260831-0015Z-measure-offset-ms-overlap-bias.md` | `measure_offset_ms` is biased against large lags |
| `20260831-0100Z-xcorr-full-corpus.md` | The statistic swap on the full corpus — 530 checkpoints, 53 files |
| `20260831-0130Z-delay-path-seeding-is-impossible.md` | You cannot seed `delays[language]` from outside the delay path |
| `20260831-0200Z-a-no-lang-population.md` | `MergeVideoVisual` has no population — the six A_no_lang files are all mislabelled tags with a 4.4 s offset |
| `20260831-0300Z-tolerance-production-audit.md` | The 0.5-frame tolerance, confirmed in production — 7 rejections, not 13 |
| `20260831-1200Z-mechanism-b-resolved.md` | Mechanism B resolved — the assembler is faithful; the *plan* is wrong for fre |
| `20260831-1300Z-cross-encode-ceiling.md` | The cross-encode ceiling is real after all — and 3 % of correct alignments fall below `VERIFY_MIN_PEAK` |
| `20260831-1400Z-single-language-ceiling.md` | A non-circular ceiling for single-language pairs — there isn't one, and the reason is the corpus itself |
| `20260831-1500Z-per-segment-ceiling.md` | Per-segment consistency works — and the ceiling on C_cut segments is uniformly high |
| `20260831-1600Z-content-relative-floor-proposal.md` | Proposal: a content-relative floor for `verify_chimera` — costed, and not recommended |
| `20260831-0758Z-correcting-by-an-estimate-does-not-make-the-next-measurement-a-residual.md` | Correcting by an estimate does not make the next measurement a residual |
| `20260831-1032Z-the-trusted-band-was-never-trustworthy.md` | The trusted band was never trustworthy |
| `20260831-1040Z-the-pal-gate-fails-open-one-way-and-closed-the-other.md` | The PAL gate corroborates before it accepts, and nothing before it refuses |
| `20260831-1210Z-err9-bounded-rerun-prediction.md` | err_9 bounded re-run: both outcomes written down BEFORE the run |
| `20260831-1330Z-campaign-before-after-instrument.md` | The campaign before/after comparison, and the defect that made its first fifteen results worthless |
| `20260831-1141Z-err9-result-both-registered-outcomes-were-wrong.md` | err_9 result: both registered outcomes were wrong |
| `20260901-0430Z-asetrate-beats-rubberband-at-both-magnitudes.md` | asetrate beats rubberband at BOTH magnitudes: PAL 0.9686 vs 0.6707 (23/23), NTSC 0.9997 vs 0.8553 (6/6). Refutes the '+/-1.7 cents is unresolvable' claim three agents had asserted without measuring |
| `20260901-1220Z-slope-vs-ratio-the-first-external-check.md` | first EXTERNAL check of the slope instrument, on Agent 4's four files: slope lands 0.0001-0.0004% from exact PAL against ratios up to 0.0844% off, 19-238x closer. The deviation is a discrete content step, and deviation/content equals 1/duration on every file |
| `20260901-1240Z-folder172-prediction-registered.md` | both folder-172 outcomes registered BEFORE measuring, with an appended correction: the files sit near FORWARD PAL, not inverse, so the premise of the test was wrong. Superseded by 20260902-0530Z |
| `20260901-1300Z-pitch-correction-detector-design.md` | design for the pitch-corrected-source detector, recording that it is biased AGAINST the case it exists to find, and that enabling the speed policy arms the unguarded delay branch |
| `20260901-1420Z-the-arm-ran-backwards-and-the-known-answer-caught-it.md` | the corpus sweep ran asetrate BACKWARDS and returned a clean 33-of-33 refutation of the owner's ruling. Nothing internal could have caught it; only an external known answer did |
| `20260901-1520Z-the-inverting-case-exists-and-so-does-a-worse-one.md` | constraint 2 answered: the inverting case exists (n=1, err_293) and a commoner DESTRUCTIVE case sits beside it -- err_26/27 match uncorrected and collapse under either correction |
| `20260901-1600Z-production-correlate-cannot-see-a-pal-pair.md` | production's own correlate scores an uncorrected PAL pair AT OR BELOW the null band for unrelated recordings, at three windows across two agents. The finding that survived every geometry correction |
| `20260901-1700Z-the-delay-is-not-applied-it-is-structurally-unrepresentable.md` | trace: the wrong delay is never applied, the pair raises. SUPERSEDED IN PART by 20260902-0100Z -- the 'refused for its shape' conclusion was measured at a geometry production does not use and is withdrawn |
| `20260901-1800Z-bounding-the-slope-refusal-population.md` | 95 of 195 failures come through the multiple-delay branch but only 5 are confirmed speed relations. Ceiling stated as 16 ratio-visible, with the direction the screen cannot see |
| `20260901-1900Z-the-delay-column-is-not-a-confirmation.md` | the delay column runs 4.28-4.88% steeper than the relation at 9.27 sigma. Four explanations tested and refuted, a fifth eliminated by arithmetic. OPEN |
| `20260901-2000Z-correction-refused-for-its-score-not-its-shape.md` | CORRECTION: these pairs are refused for their SCORE, not their shape. The earlier 'a slope cannot be represented' reading was measured at a 60 s geometry I chose myself; at production's 178 s the delays are saturated at the search ceiling |
| `20260901-2200Z-subtitle-native-wins-minimum-member-scope.md` | subtitle native-wins needs ONE closure member and it is none of the four obvious ones -- those are quota reducers that do not run under the cap. Records the risk audio does not have: subtitle tracks of one language are different KINDS, not substitutes |
| `20260902-0030Z-correction-the-22-all-carry-steps.md` | WITHDRAWAL: all 22 'single-regime' files carry a fitted step. The claim came from recording `factor` and `residual_rms_ms` while discarding `content_step_ms` -- one omission, two wrong conclusions |
| `20260902-0100Z-production-is-right-the-files-step.md` | production's per-cut delay spread EQUALS the fitted step (median ratio 1.006, 21 of 22 within 25%). The files step, production refuses correctly, and the defect report is withdrawn |
| `20260902-0200Z-where-the-piecewise-routing-decision-is.md` | the piecewise routing gate and what it tests. The gate is NOT what blocked the 22 -- 21 of 22 carry a piecewise record. A 6 s step IS expressible; `plan_segments` has no magnitude cap |
| `20260902-0300Z-err205-is-refused-by-conservation.md` | WITHDRAWAL: err_205's record is not stale -- it measured the step to 0.4 ms. The reason it is not flagged piecewise is CONSERVATION: 4 s of content unaccounted for |
| `20260902-0430Z-the-48-incoherent-are-two-populations.md` | the 48 'no coherent slope' files are TWO populations: 26 at median residual 497 ms with 7.2 s duration deltas (the Phase 2 shape), and 22 at 39332 ms with 78.6 s (do not correspond) |
| `20260902-0530Z-folder172-is-a-clean-pal-relation.md` | folder 172: all four are genuine PAL relations, slope within 0.0002% of the constant, lags monotonic 4 of 4. The `tha` question is malformed -- the masters have no `tha` track |
| `20260901-2100Z-truncation-impact-zero-verdicts-move.md` | the int() scale truncation moves ZERO recorded verdicts, on two independent arguments. The empirical leg was later withdrawn (computed on 220-char-truncated records); the structural leg stands |
| `20260901-2230Z-slope79-prediction-registered.md` | the 79-file prediction registered BEFORE measuring, including the saturation rule that any file with lags at the band edge is reported SATURATED rather than as a slope |
| `20260901-2330Z-the-79-carry-no-hidden-speed-relation.md` | 0 of 79 show a clean slope near a known constant. CLOSES the population bound: 16 ratio-visible IS the ceiling, so the largest failure family is not speed |
| `20260902-0700Z-corpus-slope-screen-suspended-at-12.md` | Corpus slope screen: suspended at 12 of 178, with its finding already in hand |
| `20260902-0800Z-native-wins-on-real-metadata.md` | Native-wins on a real file's metadata, and what I could NOT demonstrate |
| `20260902-1030Z-the-26-unvalidatable-keyframe-pairs.md` | The 26 keyframe pairs nothing can check: none is structurally unaskable |
| `20260902-1130Z-bfps-is-at-least-four-things.md` | B_fps is not one class: at least four behaviours, and only 21 of 35 are rates |
| `20260902-1230Z-dur-delta-is-sound-and-err110-is-not.md` | `dur_delta` is sound; my sign claim was wrong; err_110 is genuinely mis-classified |
| `20260902-1330Z-g24-run-registered-before-any-result.md` | The six unmeasured 23.976->24.0 files: acceptance registered BEFORE the run |
| `20260902-1500Z-g24-result-8-of-8.md` | The 23.976->24.0 group: 8 of 8 carry the relation |
| `20260902-1600Z-sections-23-and-24-are-separate.md` | §23 and §24 are separate mechanisms: a single `-t` cannot make two tracks differ |
| `20260902-1700Z-sizing-the-over-10s-class.md` | The >10 s shortfall class: 7 outputs, 14 tracks, and no single shared property |
| `20260902-1800Z-reencode-is-not-the-cause.md` | The truncated tracks are NOT the re-encoded ones |
| `20260902-1900Z-no-track-starts-at-1000s.md` | The start-timestamp prediction is refuted: short and full tracks start identically |
| `20260902-2000Z-stopped-not-cut-corpus-wide.md` | "Stopped short" separates cleanly from "cut at the limit", corpus-wide |
| `20260902-2100Z-widening-the-join-and-a-double-count.md` | Widening the join: the reconstruction validates vacuously, and my earlier counts double-counted |
| `20260902-2200Z-what-the-stderr-tail-must-show-registered-first.md` | What the err_211 `stderr_tail` must show: registered before the record exists |
| `20260902-2330Z-the-stop-is-silent.md` | err_211: the stop is SILENT. ffmpeg emitted no diagnostic while six streams ended 1000 s early. |
| `20260902-2350Z-launcher-leaks-a-process-and-the-record-corpus-cannot-fail.md` | Two defects found while auditing the instrument behind err_211 |
| `20260903-0020Z-the-gate-could-not-see-a-skip.md` | "A skipped test counts as a defect" was enforced by nothing |
| `20260903-0045Z-copyts-prediction-preregistered.md` | PRE-REGISTERED: what the retained first-pass intermediate must show |
| `20260903-0110Z-copyts-refuted-and-the-loss-is-inside-the-second-pass.md` | RESULT: the `-copyts` mechanism is refuted, and the question gets sharper |
| `20260903-0200Z-completeness-column-and-the-scan-that-measured-the-masters.md` | The completeness column, and the scan that measured the wrong files |
| `20260903-0230Z-the-12ms-lead-is-retired-and-the-test-had-no-power.md` | The 12 ms lead is retired, and the test that seemed to confirm it had no power |
| `20260903-0300Z-retention-bounded-and-a-test-that-proved-nothing.md` | Retention, bounded — and a safety test that was inert until mutation caught it |
| `20260903-0330Z-the-truncation-is-NON-DETERMINISTIC.md` | The truncation is NON-DETERMINISTIC: identical input, identical command, different stop points |
| `20260903-0400Z-the-master-pass-is-the-control-and-it-is-deterministic.md` | The master's own passes are a within-run control, and nobody had used them |
| `20260903-0430Z-the-slope-screen-at-12-with-its-denominator.md` | The slope screen at 12 of 178, written with its denominator — and my note miscounted its own data |
| `20260903-0530Z-invariants-audit.md` | DOMAIN_INVARIANTS.MD audited end to end — report, no edits |
| `20260903-0620Z-applying-155-finds-a-contradicted-inference-in-85.md` | Applying 155 to the file it was written for: two live instances, and one is a contradicted inference |
| `20260903-0700Z-85s-provenance-claim-is-the-error-86-describes.md` | 85's provenance claim is false, and 86 — the next entry — is the rule that says why |
| `20260903-0740Z-two-of-the-seven-incomplete-are-inherited-not-truncated.md` | Two of the seven "incomplete" outputs were never truncated — the master's own track was short |
| `20260903-0820Z-inherited-from-master-shipped.md` | `inherited_from_master` shipped — and the scare on the way was my own label |
| `20260902-0615Z-my-own-stamps-were-invented-too.md` | I reported the field-name problem while my own values were fabricated |
| `20260902-0640Z-reading-the-sweep-no-new-truncation-yet.md` | Reading the sweep: 32 new outputs, no new truncation — and C_cut is 2 of 179 |
| `20260902-0740Z-step1-coverage-is-a-four-day-old-ledger-snapshot.md` | Step 1's filter scope is a four-day-old ledger snapshot: 36 of 40 rows changed outcome under it, 46 eligible never audited |
| `20260902-0800Z-is-per-language-a-step-1-verdict.md` | Is `per_language` a Step 1 verdict? Only 1 of 64 in_sync results is verifiably cross-provenance |
| `20260902-1030Z-the-guard-is-not-what-it-reads-as.md` | The copy guard's real thresholds are 0.05 ms and 0.9985 — rounding four lines above redefines the literals |
| `20260902-1230Z-can-provenance-be-recovered-retrospectively.md` | Retrospective provenance: 37 recoverable / 24 undeterminable / 4 lost, and the store cannot self-heal |
| `20260902-HANDOVER-agent1-c-cut.md` | HANDOVER Agent 1 — the mission bar, the residue method, C_cut, the decidability count |
| `20260902-HANDOVER-agent2-d-trim.md` | HANDOVER Agent 2 — the guards and their real thresholds, provenance recovery, the join traps |
| `20260902-HANDOVER-agent3-ci.md` | HANDOVER Agent 3 — infrastructure, the instrument-fault population, teardown order |
| `20260902-HANDOVER-agent4-forensics.md` | HANDOVER Agent 4 — the stores, what can and cannot be dated, the evidence layer |
| `20260903-0930Z-i-recorded-the-docstring-lesson-and-wrote-it-again.md` | I recorded the docstring lesson, then wrote the same bug again — audited under 157 |

---

## Done ✅

- [x] **Fusion worker deadlock fixed** — `Pool.terminate()` joined its handler
      threads unbounded from the worker thread; one failed job wedged the queue
      for the container's lifetime. Bounded cleanup (`run_with_deadline`,
      `close_ffmpeg_pools`, `kill_pool_processes`). Verified live.
- [x] **Frame-space delay gates** — the hardcoded `127`/`128` encoded "one
      Chromaprint frame" for a 125 ms frame only; `sizePoint` is per-pair and
      138 ms occurs. Three 0-saiji files now merge, sync-verified at −10 ms.
- [x] **Gate guards after adversarial testing** (`6b90a08`) — two-distinct-delays
      requirement and a 200 ms raw-span cap. Testing found 7 of 20 one-frame
      pairs are genuine edits, but all 7 were already admitted by master's `<127`.
- [x] **PAL prebuild + content-step chain** — theoretical factor 1.042708, never
      the endpoint ratio (which absorbs trims: 2.66 s residual drift vs 2 ms).
- [x] **Chimeric structural repair** — plateau tiling, gap patching from master,
      subtitle cue shifting. Synthetic 23/23; real pair −0.5 ms per plateau.
- [x] **Per-track reference disambiguation** — for masters with several tracks of
      the correlation language (own tracks 131 ms apart).
- [x] **`fusion.py` thinned to a pure queue worker** (`0336c84`) — all merge
      intelligence moved to `mergeVideo.py` behind
      `merge_videos_with_repair(master, candidate, …)`. Verified: protected
      closure 30→30 with identical AST bodies.
- [x] **Docs**: `DOMAIN_INVARIANTS.MD`, `MERGE_VIDEO_ARCHITECTURE.MD`,
      `API_REFERENCE.MD`, `SCHEMA.MD`, `NEED_TOOLS.MD`, AGENT.MD hierarchy table.
- [x] **First-pass forensics 309/309**.
- [x] **5h01m milestone #1** written (`fcfac5c`).

## In progress 🔄

> **Archive reconciliation, 2026-09-01T14:40Z**: 57 files in `src/task_archive/`, of which
> **~10 are not referenced anywhere in this file** — all dated today, all from
> the PAL/speed line of work. A finished job with no registry row is as
> invisible as a registry row with no archive file. Listed in
> `src/diagnostics/reports/open_defects_register.md`; not repaired here
> because the rows they belong to are another agent's to write.


- [ ] **Main batch** — C_cut 34/179. **RUNNING as of 2026-09-02T07:45Z**, verified in the process table and not assumed: `full_test.sh` pid 2692018 (parent 2692015) with `batch_verify.py run C_cut` beneath it, log `/tmp/fulltest.log`. The earlier `NOT RUNNING as of 2026-09-01T14:39Z` on this line was true when written and had gone stale in the other direction — a dead entry for a live job hides progress exactly as a live entry for a dead job invents it. Restart if it dies: `cd src/tools_batch && setsid nohup ./full_test.sh > /tmp/fulltest.log 2>&1 < /dev/null &`
- [x] ~~**Deep forensic search**~~ — **closed**, costed and scoped down. 126 done,
      183 remaining of which only **76** are worth an agent; the rest are already
      passed, or answered by a script. Do **not** relaunch the 5-shard workflow.
      See `src/task_archive/20260830-0830Z-deep-forensic-search-verdict.md`.
- [ ] **Step-1 verification** — `verify_unverified.py`; resolves the true master
      from each job log and upgrades `MERGED_UNVERIFIED` verdicts.
      **NOT RUNNING as of 2026-09-01T14:39Z** — verified against the process table, not assumed.

## Next up 📋

1. ~~Diagnose err_9~~ ✅ **Resolved as a false positive.** Merged,
   master and candidate all carry exactly eng+jpn, so the candidate contributed
   no track and the check compared the master against *itself* — the
   `DOMAIN_INVARIANTS` §6 trap. It scored OUT_OF_SYNC on one weak window
   (peak 0.607, −1020 ms) while the two strong windows agreed at −20 ms
   (peaks 0.75 / 0.98). `verify_unverified.py` now returns `unverifiable` when
   the merged output has no more tracks in the language than the master, and the
   report no longer maps every non-`in_sync` verdict to OUT_OF_SYNC.
2. **Apply the safe residual-risk steps** (report:
   `/config/output/vmsam_agent/residual_risks_report.md`):
   delete the dead `get_consensus_delay` / `recover_delay_by_consensus` stubs and
   the now-unused `from collections import Counter`; add the early-exit in
   `recover_by_direct_correlation` at the **current 90 s** window (decision-identical
   by superset argument). ⚠️ Do **not** revive the consensus vote — it rescues 0
   of 52 failed files and is wrong on 3 of 4 candidates.
3. **Failed-only sweeps** — re-run files that failed under older code, now that
   `5fe3f75` / `f709c5f` / `6b90a08` are deployed. Loop until no file changes state.
4. **Finish C_cut** (150 remaining) then the remaining classes.
5. **Master Summary Table** — all 309 rows, into `AGENT_NOTES.MD`.
6. **Milestone #2** at ~00:21Z.

---

## How to restart everything

```bash
cd /tmp/vmsam_tests                 # harness + scripts live here (tmpfs — may be GONE after a container update)
nohup ./full_test.sh > fulltest.log 2>&1 &          # main batch, resumable via batch_state.json
nohup python3 verify_unverified.py > v.log 2>&1 &   # Step-1 upgrades
./watchdog_loop.sh &                                 # 1-hour kill rule
./hourly_deploy.sh &                                 # deploy when HEAD != deployed SHA
```

> ⚠️ **`/tmp` is tmpfs and is wiped by `init.sh` on container start.** The test
> harness, `batch_verify.py`, `forensic.py`, `verify_unverified.py` and the
> watchdog live there and will be **lost**. Persistent state survives in
> `/config/output/vmsam_agent/` (btrfs) and in git. Rebuilding the harness is the
> first task after a container update.

**Persistent state** (survives, in `/config/output/vmsam_agent/`):
`batch_state.json` (per-file outcomes — the batch resumes from it),
`classified.json` (309 files by class), `cut_points.json` + `cut_points_c.json`
(plateau tables, 204 records), `pal_factors.json` (21 PAL files, 14 step models),
`gate_regression_test.json`, `residual_risks_report.md`, `step1_audits.json`.

**Deploy loop**: push to `dev-AI` → wait ~10 min for CI → `cd /home/vmsam/src && bash ./hook_executor_parameters`
→ poll `GET /health` until `git_commit` equals the pushed SHA. Credentials live
only in `/home/vmsam/src/hook_executor_parameters` — **never** commit them.

---

## Standing rules (full text in `AGENT.MD`)

- **Double-validation**: Step 1 local (< 15 ms) **and** Step 2 container. A
  produced file alone is never success.
- **Protected**: `generate_launch_merge_command` and its 30-function closure;
  `init.sh`, `run.sh`, `Dockerfile`, `.github/`, `.forgejo/`; the DB schema.
- **1-hour kill rule**: any job over 60 min is killed and diagnosed as a defect.
- **5-file cap** on concurrent media probing.
- **Run to completion**: a checkpoint never stops the run.
