# VMSAM — Domain invariants for agents

**Read this before reasoning about masters, candidates, weights or folders.**
Every entry here is a fact about how the system actually behaves, recorded
because an agent got it wrong. Do not contradict one of these from intuition —
if the code says otherwise, fix this file and say so.

---

## 1. The master is the retained file, and it is normally the better one

The **master** is the episode currently in the library (the *final* folder, e.g.
`<LIBRARY_ROOT>/<library>/<show>/<season>/`). It sits there because it **won** a weight
comparison. The **error file** is in
`<ERROR_TREE>/...` precisely because it **lost** one.

`file_weight` is the release-quality ranking carried by the matching
`regex_patterns` row: higher is better.

> ### Correction on record
> An agent claimed `forced_best_video` is "frequently the error file". **That is
> wrong.** It is normally the master. `fusion.select_reference_file` picks the
> error file **only when `error_weight > master_weight` strictly**, and on an
> exact tie only when the error file is the larger of the two:
>
> ```python
> if error_weight < master_weight:   return master_file_path
> elif error_weight > master_weight: return error_file_path
> elif getsize(master) >= getsize(error): return master_file_path
> return error_file_path
> ```
>
> A higher-weight arrival is the *exception* — the case where a better release
> turned up after the episode was already integrated. Treat "master is the
> reference" as the default and the inverse as the case to justify.

## 2. Folder tells you role — use it

| Path | Role |
| :--- | :--- |
| `<LIBRARY_ROOT>/<library>/<show>/<season>/…` | **master** — the retained episode |
| `<ERROR_TREE>/…` | **error file** — the rejected candidate, in a tree mirroring the library path |
| `<TEST_OUTPUT>/…` | **test-mode output**, same mirrored shape (`VMSAM_TEST_OUTPUT_DIR`) |

The error tree is built by `tools.build_mirrored_folder_path()`, so an error
file's path *contains* its master's library path. That is the cheapest way to
identify the pair — but see §3 before trusting it.

## 3. The database is the only authority on which master is which

Filename matching is a **guess**. The real answer is the `episodes` row for
`(folder_id, episode_number)`, which the public API returns as
`master_file_path` in the `POST /fusion` response, and which the batch records
as `db_master` in `batch_state.json`.

This is not theoretical: the Log Horizon pair was analysed against a
filename-matched AMB3R BluRay while the database resolved a different Erai-raws
BD as the master. Any plateau table or speed factor measured against the wrong
master will re-time the candidate onto the wrong reference — a silent desync.
`prepare_structural_corrected_candidate` and `apply_pal_step_repair` both refuse
to act when the record's master differs from the job's master for this reason.

**Prefer `db_master`. Fall back to the filename guess only when the master
directory holds exactly one candidate for that episode, and label it a guess.**

## 4. Repair is directional — `merge_videos` cannot infer it

`merge_videos(files, …)` takes an **unordered** collection (`main.py` passes a
`set`) and sorts it. It has no master/candidate axis. Every repair does:

- `detect_pal_relation(master, candidate)` returns the factor **or its inverse**
  depending on which side is which;
- the chimeric assembler re-times the **candidate onto the master's timeline**.

So the master must be passed explicitly. That is why
`merge_videos_with_repair(master_file_path, candidate_file_path, …)` takes it as
the first parameter instead of guessing from `forced_best_video`. Getting the
direction wrong does not raise — it produces a plausible file that is silently
out of sync.

## 5. `forced_best_video` is a *quality* choice, not a *reference* choice

It answers "whose video/streams should win in the output". The repair reference
answers "whose timeline is authoritative". They usually coincide (both are the
master, per §1) but they are **different questions**, and code must not use one
to infer the other.

## 6. A produced file is never proof of success

Verification must cross-correlate a **master-derived** track against a
**candidate-derived** track inside the output, at more than one point, with
drift < 15 ms.

Two traps, both hit for real:
- A master carrying **two tracks of the same language** (one pair's own `jpn`
  tracks sit 131 ms apart) — comparing "the first two tracks of this language"
  compares the master against **itself** and reports a desync the merge never
  caused.
- When the candidate contributed **no** track in that language (deduplicated as
  inferior), there is nothing to verify. The honest verdict is `unverifiable`,
  not a number.

Track order in the output is deterministic: `generate_launch_merge_command`
emits the best video's tracks first, so within a language the first `n_master`
streams are master-derived and anything beyond is candidate-derived.

## 7. Chromaprint resolution is per-pair, not a constant

`get_max_corr` returns `(correlation, offset_in_FRAMES, offset_ms)` where
`offset_ms = -frames * sizePoint` and
`sizePoint = int(lengthFile / len(fingerprint) * 1000)` — ~125 ms typically,
**138 ms observed**. A ±1 frame disagreement may be pure quantisation noise…

…but **not always**. Measured on this corpus: 7 of 20 one-frame pairs have a
genuinely piecewise lag (steps −209…+168 ms). A real 167 ms edit is 1.34 frames
and quantises into two adjacent frames — indistinguishable from noise at the
fingerprint level. Never conclude "one frame ⇒ noise" without an independent
high-resolution measurement.

## 8. Measurement hygiene

- Calibrate every instrument against a known answer first: a file against itself
  must measure **0 ms at peak 1.0**. An uncalibrated correlator locked onto loud
  passages and returned false lags.
- Use **normalised** cross-correlation; unnormalised peaks are not comparable to
  a fixed confidence floor.
- Cross-language correlation is biased (only music and effects align between two
  dubs) — peaks around 0.5–0.85 and offsets that should not be trusted to 15 ms.
- The pipeline works in **ISO 639-1** (`ja`); `ffprobe` reports **ISO 639-2**
  (`jpn`). `video.py` already normalises via `iso639.Lang().pt1` and stores
  `StreamOrder` on each audio entry — read those rather than re-probing.

## 9. Delay sign convention

`measure_offset_ms` returns **+lag when the candidate must be delayed** by that
amount to align with the reference, and a positive `delays[language]` does
exactly that downstream. `compare_video` negates because it measures in the
opposite direction — do not copy its convention into new code. An earlier
version negated wrongly and shifted candidate tracks −80 ms where +70/+90 ms was
required.

## 10. The control corpus `/config/example` — the BDRip is the master

Stated by the project owner on 2026-08-30. Each of `ore1`, `ore5`, `ore10` holds
one episode in four releases: one **BDRip carrying FLAC**, and three WEB-DL /
WEBRiP releases carrying lossy audio. The BDRip is always the **master**; the
three web releases are the **candidates**. (Release names are in the private
notes outside the repository — see `VMSAM_PRIVATE/README.MD`.)

This is consistent with §1 and §5: `select_reference_file` picks on weight, and
a BDRip carrying FLAC outweighs a WEB-DL carrying AAC. The BDRip supplies the
video and the timeline every other track is re-timed onto, and its FLAC is the
best `jpn` audio unless measurement says otherwise.

Report offsets as *candidate relative to the BDRip master* so numbers from
different methods can be compared directly. If a run of the production code
selects a different reference here, that mismatch is a finding — report it with
the computed weights rather than overriding the choice by hand.

## 11. Do not recover a delay by voting on stored residuals

Recorded here because the code that documented it has been deleted as dead.
`mergeVideo.py` once carried `get_consensus_delay` (modal offset over the
per-cut vector) and `recover_delay_by_consensus` (a stub that always returned
`None`, kept only to explain itself). Both had zero callers.

**The temptation**: the per-cut offsets stored in `delayFirstMethodAbort` look
like ten measurements of one number, so take the mode and write it into
`delays[language]`.

**Why it is wrong**: `first_delay_test` is *iterative*. It accumulates
`delayUse`, calls `recreate_files_for_delay_adjuster` to re-extract the audio
shifted by that amount, re-correlates, and repeats until the residual reaches 0.
The stored numbers are **residuals from one iteration** — they carry neither the
accumulated base nor the refinement that follows. Writing one into
`delays[language]` discards both. A merge built that way came out **130 ms out
of sync**.

**Measured, not merely argued** (residual-risks investigation, see
`src/task_archive/20260829-0200Z-residual-risks.md`): reviving the vote newly
accepts **5 vectors out of 929**, rescues **0 of 52** FAILED files, and is
demonstrably wrong on **3 of the 4** candidates it would have accepted.

If a consensus rule is ever wanted, it belongs **inside** `first_delay_test`,
where the residual re-enters the refinement loop and converges normally. Post-
processing recovery is the right home for gap patching and segment trimming,
which repair the *media*; it is the wrong home for inventing an alignment the
correlation loop never converged on.

## 12. A check that has never been shown to fail is an unproven claim

§6 says a produced file is never proof of a successful merge. The same applies
one level up, to the instruments — including the ones guarding the code.

Recorded because it happened to the protected-closure checker, the strictest
guard in the project. It was written, committed, cited in three reports across a
full session by two agents and the Lead, and referenced from `AGENT.MD` as the
check every agent must run. **Nobody had ever run it against a tree it should
reject.**

When it finally was, deleting a protected function gave:

    PROTECTED CLOSURE: 77 members reachable ...; touched in ANY way: NO -- intact
    exit 0

The failure was structural, not a typo. The protected set was derived by walking
call edges **from the current tree**, so a deleted function is unreachable,
absent from the set, and every membership test against it is False by
construction — the evidence of the violation was consumed by the mechanism meant
to detect it. It even printed a *changed member count* (77, not the familiar 78)
on the same line as the word `intact`, removing the last chance a reader had to
notice. Only a closure unioned across **both** revisions can see a deletion.

So, for any guard, verifier or gate before you rely on its output:

- **Run it against an input it must reject**, and check the exit code, not only
  the printed verdict. A guard that prints `VIOLATION` and exits 0 is not a guard.
- **Prefer over-approximation** in a protection check. Under-approximating is the
  dangerous direction: an earlier walk of this same closure reported 47 members
  because it did not follow class instantiation into methods, and would have
  passed a change to any of the 31 it missed.
- **A number that moves is information.** If a guard reports a count, a change in
  that count is itself a signal, and printing it beside a pass verdict wastes it.

**A docstring is not a guard.** All four unfailable checks found on 2026-08-30
shared one property beyond being unfailable: each was *documented* as doing
something it did not do. `cross_series_check`'s docstring promised an abort that
was never written. The copy-trap guard in `verify_against_master` survives being
deleted outright. `closure_check.py` announced "intact" while comparing nothing.
`FIDELITY_BAR` was described as the dub gate and was held by no test.

The failure is social as much as technical: **three separate reports cited that
self-test as evidence, and none of the three authors — including the one who
wrote it — ever ran it against an input it should reject.** A guard described in
prose reads as a guard that exists. Before citing any check as evidence, run it
on something it must fail. If it passes, you have found the defect rather than
the proof.

### 12b. A measurement is only true as of a revision — say which

Two agents and a committing Lead share one working tree here, so it moves under
whoever is measuring it. An agent ran the same experiment twice and got
different results; both were correct **when made**, and the tree had changed
between them:

    c2a931f : "closure size moved" = 0
    f01a7f1 : "closure size moved" = 0
    bc8660b : "closure size moved" = 1   <- landed between the two runs

It nearly dismissed the discrepancy as its own truncated output. That is the
dangerous outcome: a real signal filed as operator error.

**Quote a SHA with any cross-agent measurement.** Every number reported in this
session — 78 closure members, 37 differing functions, 481 checkpoints, 28 of 180
records — was stated without one, and each was only true of a particular tree.

Two related traps seen the same day:

- **A running script does not reload.** Editing a long-running shell service
  changes the file, not the process. A queue service kept matching an old
  pattern for minutes after the fix was committed, reporting an empty queue
  while a request sat in it.
- **A test run is itself a measurement and can straddle a commit.** An agent hit
  two failures against a test class that existed when its run started and not
  when it grepped seconds later; both were the tree moving under an in-flight
  run, not defects. Re-run before diagnosing.
- **Treat a skip as a defect, not a pass.** A regression test written for the
  porcelain parser called `skipTest("working tree is clean")` — opting out in
  exactly the state anyone running the suite is normally in, and reporting
  `OK (skipped=1)`. It was found only because someone asked what the skip was.
- **`git status` can lie about itself.** A stale stat entry showed a file
  modified whose content was already identical to HEAD. `git diff` is the
  arbiter, not the porcelain summary.


**A shared bytecode cache can serve one agent's mutant to another agent's
import.** `PYTHONPYCACHEPREFIX=/tmp/pycache` is set box-wide here, so every
agent and every scratch copy of the tree share one cache. Observed on
2026-08-30: `import merge_video_prebuild` yielded `ATEMPO_MIN = 0.0` while the
file on disk read 0.5 and `inspect.getsource` also read 0.5 — source and running
code disagreeing with no error and no warning. The `.pyc` header validates on
source **mtime in whole seconds and size in bytes**, so swapping a same-length
literal (`0.5` -> `0.0`) keeps the size identical and mutating then restoring
inside one second keeps the second identical; the pyc was written 49 ms before
the source was restored and Python called it valid thereafter.

This breaks mutation testing in **both** directions — a mutant survives because
the suite ran the original, or dies because it ran the *previous* mutant — so
affected verdicts are unproven rather than wrong. Run any mutation sweep with
`PYTHONDONTWRITEBYTECODE=1`, and treat "the source says X but the behaviour says
Y" as a cache question before it is a logic question.

**And an ARTIFACT is only interpretable if it records the revision that produced
it.** §12b asks a measurement to say which revision it is true at. The same
applies to anything a run leaves behind, and it bit twice within one hour on
2026-08-30, both times in the same direction: an absent marker was about to be
read as evidence about behaviour.

- `batch_state.json` rows carry `outcome`, `stage`, `seconds`, `exception` and
  no build sha. Eight files read FAILED after a deploy; the file's mtime fitted
  the window since the sweep restarted, but a fitting window is not attribution,
  and the rows could equally have been the previous image's.
- Merged outputs carry `CHIMERIC=` stream tags, so a missing tag reads as "the
  assembler did not re-time this track". But absence is evidence only if the
  build that wrote the file HAD the tagging code. Checking mtimes against the
  commit that introduced it: three of eight outputs predate it by 5 hours or
  more, so no tag was possible; two sit 10 and 23 minutes after the COMMIT,
  which is not a deploy, and no deployment record exists for that window. The
  set was about to be reported as three when the truthful answer is **three
  confirmed, up to eight, and the marker closes neither direction for the rest**.

A single build-sha written beside each artifact closes both questions for free,
and would have resolved the two uncertain files immediately instead of leaving
them permanently uninterpretable. **Absence of a marker is not evidence of
absence of the behaviour unless the artifact can prove the marker was possible.**

**The asymmetry is the real defect, and this file said it wrong first.** The
project already does this correctly in one place: `fusion.append_error_log`
writes a `(git=<sha>)` banner into every `.log.error`, deliberately ungated on
dev mode, because a production failure is exactly the one that most needs to say
which build produced it. 70 of 801 error logs carry a stamp, across nine builds
— and those banners turned out to be a usable deployment record, closing both
"uncertain" files above by ancestry rather than by inference.

So the artifact written on FAILURE is stamped. The artifacts nobody could place
today — the merged outputs, and `batch_state.json`'s success rows — carry
nothing. **That is the wrong way round.** A failure already announces itself; a
silent success is the thing you cannot place afterwards, and it is the one
without a stamp. Both of today's unanswerable questions were on the success
side.

### 12c. The recurring shape: precise, plausible, and measuring something else

Three separate instruments failed this way in a single session. None produced an
error, a warning, or an implausible value. Each produced a **confident number
about the wrong thing**, and each was caught only when someone compared it
against an independent measurement.

| Instrument | What it reported | What was true |
| :--- | :--- | :--- |
| chimeric residual | `2629.9 ms` at peak 0.474 | the correlator was against a **±3000 ms rail**; the real error was 11932 ms |
| offset benchmark | the run's **tightest** spread, 0.008 ms | a missing `-map` had it correlating French against Japanese; wrong by 21 ms |
| provenance digest | a stable content hash | `.strip()` ate one character, so it hashed paths like `rc/tools_batch/...` that do not exist |

A fourth followed, found while writing tests for the first: `_residual_detail`
was a module global, so a checkpoint that failed to report inherited the
previous one's rail and envelope statistics — a bound and a confidence **from
the wrong checkpoint**, printed against this one's residual.

The common property is that **a wrong answer looked better than a right one**,
and the sharpest statement of it came from that fourth case:

**Scope matters, and was checked rather than assumed.** That fourth case was
**log-only**: `rail_ms` is computed from the module constant, and no judging
branch reads `_residual_detail` — reads of it sit inside the dev gate. A stale
bound printed a wrong `±N ms range` while `railed` was still evaluated against
the correct 3000 ms. Bad diagnosis, correct verdict.

Which yields the rule that generalises it:

> **A verdict that reads from a diagnostic buffer inherits every defect in the
> diagnostics.** Had the bound come from `_residual_detail.get("bound_ms")`, that
> bug would have silently changed which builds shipped — and no test would have
> caught it, because every test stubs the measurer.

Keep judging values on their own path. A diagnostic may be wrong without
consequence; a verdict may not.

> The danger is never the number that is missing — it is the number that is
> present and about something else. A missing number announces itself; `n/a` is
> safe. The ones that cost real time all rendered beautifully.
The railed residual was a clean round number. The mis-mapped correlation had the
*lowest* spread in the benchmark, which is exactly what a reader trusts. The
corrupted digest was stable and well-formed.

So the defence is never "does this look reasonable" — all three did. It is:

- **Compare against an independent instrument.** Every one of these was caught by
  a second measurement disagreeing, never by inspection.
- **Publish the bound alongside the value.** A residual that cannot exceed
  ±3000 ms must say so next to the number; 88% of a rail is not a measurement.
- **Calibrate against a known answer first** (§8). A file against itself must
  read 0 ms at peak 1.0, and a parser must be run on input whose correct output
  is known.
- **Distrust the best-looking result specifically.** Twice here the wrong number
  was the most attractive one on the page.


### 12d. Mutate the constants, not just the code

A suite can cover a function thoroughly and still leave the **number** that
function compares against completely unheld. Measured on 2026-08-30 at 551
checks: `FIDELITY_BAR`, the `fpcalc` threshold enforcing chimeric dub integrity,
could be set to **0.0** (every dub declared the same recording) or **1.1** (no
dub ever the same recording) with **all 551 checks passing**. Its neighbours
were properly held — `VERIFY_MIN_PEAK` killed 6 and 7 mutants, `VERIFY_RAIL_FRACTION`
12, `FIDELITY_CUT_S` 1 — so this was not a weak suite. The single unprotected
constant was precisely the one enforcing the success criterion this project
calls invisible to every other check.

The mechanism is worth recognising because it recurs: the dub-gate tests used a
table double that overrode the decision method wholesale, and the only two calls
to the real method asserted on early-exit branches that return *before* reaching
the threshold. **22 checks named for the gate, none touching the bar.** A test
double placed one layer too high converts a real test into a test of the double.

So: for any constant that decides a merge, change it to an absurd value and run
the suite. If it stays green, the constant is undefended and every conclusion
that rested on it is unearned. Still open at the time of writing, same family:
`DURATION_TOLERANCE` in both chimeric and prebuild can be set to 1.01 — any
duration mismatch passes — with the suite green.

**Three in one day, three instruments, one shape.** On 2026-08-30: `FIDELITY_BAR`
took 0.0 or 1.1 with a green suite; the ledger's `in_sync` verdict was decided at
50.0 ms against a stated mission bar of 15 ms, from a signature default *and* a
second hardcoded literal that could drift apart independently; and the instrument
behind 23 `unmergeable_verified` marks computed `norm(x)**2 / norm(x)**2`, which
is 1.0 for every input — confirmed on white noise, a constant and a sawtooth,
all reading 1.000000 — while its docstring promised an abort that was not in the
code. None was found by reading. All three were found by feeding the check an
input whose correct answer was already known. Budget for that as a routine cost
of any check you intend to rely on, not as an audit you do once.


## 13. A reading too weak to judge a build with cannot introduce a cut

**A plateau, group or window-set resting on a SINGLE window carries its reading
with no second opinion, and every consistency metric computed over it is
vacuously perfect.** A spread over one point is zero. A line through one group
of one has a residual of exactly zero. A tail whose only evidence is one weak
window looks as clean as one backed by twenty.

Four surfaces in three modules on 2026-08-30, found independently by two agents:

- a tail plateau built from one window at peak 0.393 invented a change point at
  the END of a file, tabling +44.049 s where the true offset was steady +32.117 s;
- a step model fitted to single-window groups scored a residual of zero *by
  construction*, so a steady drift beat the line that actually described it;
- a scanner admitting windows at peak >= 0.30 — while the segmenter uses 0.35 and
  the verifier 0.45 — introduced phantom mid-file cuts on 7 of 18 fresh records.
  On one record **every** cut found was manufactured by that floor: 6 windows and
  3 change points at 0.30, versus 3 windows and ZERO at 0.45, the survivors
  agreeing to 25 ms across 1323 s;
- two head windows inflated a plateau spread to 301.1 ms, of which 301.1 came
  from those two, the remaining six spanning 0.7 ms.

The rule, now in code as `CHANGE_POINT_MIN_PEAK = chimeric.VERIFY_MIN_PEAK`
rather than restated so the two cannot drift: **a window below the floor the
verifier uses to judge a build may not introduce a change point**, because a
change point becomes a plateau boundary, which becomes a real splice in a
rebuilt file.

Expect a fifth instance in a fourth module. This is not a bug that was fixed; it
is a shape. When you compute any statistic over a group, check the group's size
before believing the statistic — and prefer a metric that is *undefined* on one
point to one that is perfect on it.

## 14. A terminal state needs a re-examination trigger, not only a proof

`unmergeable_verified` is the only state that closes a file without a merge, so
it should carry the best evidence in the ledger. **It has never once been
examined without something being found in it, and the reason has been different
every time**: 7 claimed and 6 refuted on the first pass (bad evidence); 23
claimed and all 23 a cataloguing defect (good evidence, wrong state); 3
deep-report verdicts and 1 refuted (good evidence resting on an expired
premise).

The third kind is the one worth naming, because the report that produced it was
a model of measurement discipline. It said, honestly and correctly at the time:
a piecewise remux could reach the bar, *"VMSAM has no representation for a
piecewise delay track, so this is out of reach for the pipeline as written"*.
That was true when written. It stopped being true when `merge_video_chimeric`
began assembling piecewise from a plateau table — and nothing re-opened the
verdict, because nothing recorded that it had a shelf life.

So distinguish the two kinds of proof, and record which one a verdict is:

| `limit` | means | expires? |
| :--- | :--- | :--- |
| `media` | a property of the file — no decryption key, content that does not exist | **no** |
| `pipeline` | a capability we do not have *yet* | **yes, when the pipeline gains it** |

A verdict citing a pipeline limitation is re-checked whenever the pipeline
changes. A verdict citing a property of the media is not. Without the field,
every closed file has to be re-read by hand to tell which is which — and the
one that expired had also been misfiled into the wrong class, so nothing would
have offered it to the repair path even after that path existed.


## 15. Guards fail OPEN here, and they were all written to be safe

Four guards found broken in one day, in four files, by three agents. Every one
of them, asked a question it could not answer, gave the **reassuring** answer:

| guard | could not tell | said |
| :--- | :--- | :--- |
| `closure_check` | the baseline was unreadable | `intact`, exit 0 |
| `ci_ready` | the SHA was empty | ready, exit 0 |
| `watchdog_loop` | the timestamp would not parse | age 0, so never past the hour |
| `cross_series` calibration | `norm(x)**2/norm(x)**2` | `1.0`, for any input |
| `deploy_queue` tree read | `git status` failed | empty output, so "nothing is dirty" |
| the Lead's commit gate | — | `checker \| tail -1 && commit` — the `&&` fires on `tail`'s exit |

The watchdog is the sharpest. It asked the container for the running job and
split `path started_at` with `awk '{print $2}'` — and **every real media path
contains spaces**, so field 2 was a fragment of a show name, `date -d` failed,
and `|| echo "$NOW"` made the age 0 on every iteration. The one-hour kill rule
that AGENT.MD calls mandatory has therefore never once applied to a fusion job,
in a project where a wedged merge is a known failure mode and that loop is the
only thing watching for it. Measured live on a running job: 216 s by the fixed
rule, 0 s by the old one. And it looked healthy — the truncated path was
non-empty, so the emptiness guard above it passed and the loop ran every 60
seconds, judging nothing.

Two of those six were written **the same day as this entry, by people actively
looking for this shape**. The queue's tree read — `blocking=$(git status
--porcelain | python3 deploy_scope.py)`, testing only the OUTPUT, where a git
failure yields an empty string that reads exactly like "nothing is dirty" — was
committed hours after its author wrote the §15 finding. And the Lead's own commit
gate had been a pipeline all day: a pipeline's exit status is the last command's,
so `tail` returns 0 whatever the checker printed, and two commits the privacy
checker had **rejected** went through underneath it.

Knowing the pattern does not confer immunity to it. Audit for it deliberately,
in your own code, after you have written about it.

`|| echo "$NOW"` was written to be safe. So was returning 0 rather than
crashing, and so was continuing past an unreadable baseline. **The instinct that
produces this is "do not let the check itself break the run", and it is
correct — the error is spelling the fallback the same way as success.**

So: every guard needs a third answer. Not pass and fail, but pass, fail, and
**cannot tell** — and `cannot tell` must never be spelled the same as pass. The
watchdog needs a fourth, because "no job running" is a different thing again.
And put the rule where tests can reach it: this one lived in bash, which is
exactly why nobody found it by testing.


## 16. A file at a derived path is inferred evidence, not direct

Merged outputs are named after the **master**, so every candidate for the same
episode derives the **same** output path. Measured on 2026-08-30: **24 of 66
on-disk outputs are claimed by more than one error id.**

This nearly produced the day's most spectacular wrong finding. A first pass
reported an output existing for a file the ledger calls `unmergeable_verified`
— which would have refuted a verdict upheld an hour earlier on strong evidence.
It was wrong. Four error ids derive that one path; two of them are
`MERGED_IN_SYNC` with a **recorded** `merged_file`, and two recorded none. The
file belongs to one of the first two. There was never any evidence the closed
file produced an output.

So distinguish the two kinds of evidence, and say which one a claim rests on:

- **direct** — `batch_state.merged_file` names the file that run produced;
- **inferred** — a file happens to sit at a path derived from `db_master`.

Inferred evidence is worthless wherever the path is shared, which is 36% of this
tree. Applying the rule to fourteen apparent "pending files with an output"
resolved them into **nine real** (direct, all `MERGED_UNVERIFIED`) and **five
phantom** (inferred only, all `FAILED`, the file beside them belonging to a
sibling candidate).

The general shape, and it is §12c again in a filesystem: **a path is a
construction, not an observation.** Deriving one and then finding a file there
tells you the derivation ran, not that the file is yours.


## 17. A list assembled by reading is a hypothesis — test that it is COMPLETE

Twice on 2026-08-30 a careful list was short, and both times the test that
checked the list's *completeness* found it within a minute of being written:

- a scanner reported **8** literal-binary call sites; there were **9**. The miss
  was a `return "ffmpeg"` shape the scanner had no rule for. A peer named the
  site from memory that the tool had not.
- a lock was added to the **2** known deployers. There were **4**. A test that
  globs every `*.sh` and asserts that anything containing `hook_executor_parameters`
  holds the lock failed immediately on `redeploy_and_sweep.sh` and
  `watchdog_loop.sh`.

The watchdog is the instructive one. **It does not look like a deployer** — it is
named for killing wedged jobs, and the container recreate is a *consequence* of
the kill. So it reads as a monitor and behaves as a fourth deployer, able to
restart the container underneath an in-flight deploy: the exact failure it exists
to prevent. The agent that found it had read that file twice that day, including
the pass that found its age bug, and did not see it.

This is the under-approximating direction §12 calls the dangerous one, and it has
now produced a closure walk reporting 47 members where there were 78, a scanner
reporting 8 of 9, and a lock covering 2 of 4.

**And the rule usually contains another list, one level down.** Rewritten as a
rule — *a tool name may appear only as a key of `tools.software` or as an
argument of a sanctioned accessor* — the sweep immediately found the two shapes
the hand list had missed, including one for a tool with no call site at all,
which no list of shapes could have caught. But **the sanctioned-accessor list is
itself a list**: omitting one agent's `_software` helper reported four of its
already-fixed sites as violations, and "VIOLATIONS: 11" was on screen and one
step from being sent. The honest count after triage was 2. The tool now prints
"triage before quoting" beside its own total, because it misled its author first.

**So do not assert the members of a set — assert the RULE that defines it, over
the whole space.** "Every script that fires the hook holds the lock" is checkable
against every script; "these two scripts hold the lock" is checkable only against
the two you thought of. The second is the list you already believed.


## 18. Locating a known answer is DISCRIMINATION, not another search

When both candidate answers are already known, re-running the full search to
choose between them is not a refinement — it is a fresh chance to find a third,
wrong answer, from less data than the original.

Measured on 2026-08-30. A cut-point refiner bracketed a change point and then
probed it by re-searching the whole ±53 s band from a **6 s** window. It returned
−40671.8, +48067.2 and +48512.5 ms at peaks 0.607, 0.673 and 0.577 — confident
numbers nowhere near *either* candidate lag, both of which the sweep had already
established. Rewritten as a two-way test in a narrow band around each known
candidate, localisation error against an independent report fell from 26.5 / 1.5
/ 50.8 s to 7.0 / 10.6 / 7.9 s.

A second defect surfaced with it, and it is the §12 shape again: **the bracket
could not contain the answer.** A window is one correlation over its whole span,
so a cut can lie well inside the *later* window rather than in the gap before it.
The true cut at 114.98 s sat outside a bracket of [60, 90], and the bisection
returned its own edge, 88.50 s — a rail wearing the costume of a result, exactly
like a residual reported against its ±3000 ms limit, or a pHash search returning
lag −40 at chance level when the optimum was at 49.

So a search that stops at its own wall must **say so**. Each change point now
carries the bracket it opened with, whether it landed on an edge, how many probes
were informative, and a `located` verdict — and the labelling tracks the truth:
the two marked `located=True` are off by 7.0 and 7.9 s, and the one marked
`located=False`, with 0 of 5 probes informative, is a 141 ms step that is a few
envelope bins wide and genuinely cannot be resolved. **The tool knows which of
its own answers to trust**, which is worth more than a tool with a better average
error and no opinion about itself.

An honest "cannot tell" was kept in preference to a 30 s probe reading 0.951
against 0.951, which was reverted rather than kept as an unmeasured improvement.

**The first version of that verdict was wrong, in the other direction.** It
OR-ed "landed on a bracket edge" with "no probe was informative", and on the full
corpus that marked **84 of 102 change points not located** — including a 10009.5
ms step whose probes read 0.125 against 0.988, as decisive as the instrument
gets, flagged only because it sat 1.5 s above its bracket's lower bound, which is
simply where the cut was. **A tool that distrusts all of its own answers is no
more useful than one that trusts all of them**, and this would have made a
correct table look untrustworthy on a labelling artefact. The two are different
questions and must be published separately: `located` asks *could any probe
tell?*, `on_bracket_edge` asks *did the search stop against a wall?*

Separated, the corpus shows the pattern that makes the table usable: **every step
above a second locates decisively, every step below half a second not at all.**

And one caution about how that is described, from the author's own correction: an
unlocated small step was justified as "inside the frame the merge snaps to". It
is not. **250 ms is six frames at 23.976 fps**, and a quarter-second of desync
sustained over a minute is audible. The unlocated cases are not harmless; they
are *small*, which is survivable and is a different claim.


## 19. Read the ledger before building the detector

Nine files looked like untouched backlog. The right answer was already recorded:
**all eight carry `sync_verdict: unverifiable` in their own records, and all 146
other `pending` files carry `None`.** Exact, present, and absent from every count
— which is why they read as available work. The fix was one line of arithmetic
(`mission.pending_but_already_judged_unverifiable = 8`), not a new instrument.

The instrument was built first, and deleting it is the instructive part. A
metadata scan meant to mark files unverifiable-by-construction returned
**`verifiable` for all 104** files with an output, because the rule it can
compute — does the master carry any language the candidate also carries — is
necessary but not sufficient. All eight failures *do* share a language with their
master; what defeats them is that the output's track in that language is the
master's **own**, which only a sample comparison establishes. **A field that says
`verifiable` for everything is not a filter**, so it was dropped rather than
shipped.

Two rules from that. Before writing a detector, check whether the pipeline
already records the answer — this ledger has repeatedly known things nothing was
counting. And when a new field returns the same value for every row, that is a
result about the field, not about the rows.


## 20. An atomic write of stale data is atomically wrong

`batch_state.json` was written through `os.replace`, so every individual write is
all-or-nothing. It still lost half its rows. Reproduced with two real processes
writing 40 rows each through the shipped `save_state` shape:

    rows written: 80    surviving: 40    LOST: 40

No corruption, no error, nothing saying a write had been lost. **`os.replace`
makes the WRITE indivisible and says nothing about the READ that preceded it**,
and the visible atomicity is precisely what stops people looking further — the
same way an existing `flock` on the queue service stopped anyone asking whether
the *deploy* was protected. Both are one shared resource where the indivisible
unit was drawn around the wrong span.

Two consequences worth keeping:

- **The lock must be a sidecar file, not the state file.** Locking the state file
  means holding a descriptor on the very inode `os.replace` swaps out: the lock
  follows the replaced inode while the next process locks the new one, so two
  writers each hold a valid lock on a different inode and exclude nothing. A lock
  that looks right and excludes nothing is worse than no lock, because it closes
  the question.
- **`update()` must take a function, not return a dict.** Handing the caller a
  dict to save later reopens exactly the window the lock closes.

And the first test must be a **control that reproduces the loss**. A concurrency
test that passes may mean the code is safe or may mean the harness never
collided; without the control, the zero proves nothing. This is the concurrency
form of §12: **a race that does not reproduce is not a race that does not
exist**, so the first test must lose rows on purpose. It is the only thing
separating "safe" from "lucky", and unlike an ordinary check you cannot get that
assurance by inspection.

The earlier response to this defect was to have one tool write to a *different*
file, which left the mechanism in place and split the ledger — which is how a
3.90 GB merged output came to exist with no row describing it. That orphan and
the lost verdicts are one defect seen from two ends.


## 21. `_xcorr_offset` is not a correlation coefficient

Three agents found this independently on 2026-08-30, from three directions, and
none was looking for it:

| context | "peak" on a **perfect** match | should be |
| :--- | :--- | :--- |
| exact sub-slice, production's asymmetric shape | 0.6697 … 1.1890 over eight trials | 1.0 |
| self-correlation ceiling for a checkpoint | 1.012 | 1.0 |
| wide search band (190.6 s / 220.1 s) | 1.432 / 2.347 | ≤ 1.0 |

**A value above 1 is proof the quantity is not a correlation coefficient.**

The mechanism: `_xcorr_offset` z-scores the reference slice **as a whole** and
divides by template length, so every candidate position is normalised against the
slice's *global* statistics rather than its own. Equal-length inputs cancel and
read exactly 1.0 — **which is why correlating a file with itself does not catch
it**, and why a naive calibration passes. The bias exists only in the asymmetric
short-template-against-long-reference shape that production actually uses, and it
grows with band width: r = 0.545 between max peak and log10(band) over 20
records, with the two records that overshoot being exactly the two that would not
resolve.

Two consequences that are live:

- **`VERIFY_MIN_PEAK = 0.45` has far less headroom than its name suggests**,
  because a *perfect* match already reads 0.6697 in one measured case. On quieter
  content an exact match could fall below 0.45 and be discarded as "too little
  structure to judge". It errs safe, so it is not a wrong verdict — but the floor
  is not measuring what its name says.
- **A self-correlation ceiling is vacuous.** With a properly overlap-normalised
  NCC, a file against itself reads 1.000000 exactly, so the ceiling can only ever
  be 1.0. The 1.012 once quoted was not measuring content difficulty; it was
  measuring this bias. **The ceiling was compensating for a defect in the
  statistic, which is exactly why it looked informative.**

The proper form — a locally-normalised sliding correlation, normalising each
position by its own window statistics — is bounded by 1 by construction and read
1.000000 with the correct lag at all eight trials where the biased form ranged
0.67 to 1.19.

Replacing it inside `measure_residual_ms` changes every peak `verify_chimera`
computes and therefore which builds ship, so it is **measure-then-decide**, not a
refactor: compute both statistics side by side over the corpus and report how the
floor and the existing verdicts move, before anything is enforced.


## 22. Availability is a property of the ARTIFACT, not of its ingredients

Twice within one hour, an "obvious" test for *can this be verified?* over-counted,
both times in the same direction, and both times the corrective was to ask what
is actually in the output rather than what the inputs imply should be.

1. **"a candidate-derived track exists"** — output count greater than master
   count — found 7 languages on one file. It does not check that a **master-side
   reference** exists to correlate against, and for 6 of those 7 the master
   carried no track in that language at all. §6 needs both. Real answer: 1.
2. **"the language is present in both master and candidate"** — the correction to
   the above, and still wrong, because it is measured on the **inputs**. On the
   two files it was supposed to unlock, every input-shared language had its
   candidate track **deduplicated away as inferior**: master 1, output 1, and the
   surviving track is the master's own. Correlating it compares the master with
   itself, which is the §6 trap that defeated the original selection on all eight.

The predicate that survives is stated over the artifact:

    answerable(lang) = master_count[lang] >= 1
                       AND output_count[lang] > master_count[lang]

— a master-derived reference exists **and** a candidate-derived track *survived
into the output*. Anything weaker is a statement about what was submitted, not
about what can be measured.

The trap has teeth because the failure is not "no result": §8 puts cross-language
correlation peaks at 0.5–0.85, well inside the range a lax acceptance would take.
A selection built on the ingredients would try languages that cannot succeed and
return a plausible number from comparing two different recordings — introducing
the exact false pass the fix existed to prevent.

**A fix that moves a file the artifact says is unanswerable has not fixed
anything; it has started fabricating.**


## 23. A stored verdict is evidence only while its file is unchanged

Three times on 2026-08-30, a record was read as the current state of a thing
when it was the state of a thing that had since moved. Each was caught by a
different agent, from a different direction, and none by the author of the
record:

- an **index** rebuilt from the logs still in the tree dropped every entry whose
  original had already been retired — 131 entries and 214 banners fell to 23 and
  44, while the archived files themselves survived;
- a **field** derived from the raw batch outcome counted 41 D_trim files as
  untouched when 38 of them had since been repaired and were `step1_passed`;
- a **verdict** — "the only jpn track is bit-identical to the master's" — was
  used as a known-negative control for a new classifier, which disagreed with it
  on two of eight files. The classifier was right. **Seven of the eight outputs
  had been rebuilt since those verdicts were written**, so the verdicts described
  artefacts that no longer existed at those paths.

The mechanism is always the same and it is not a reasoning error: the artefact
moved and the record did not, and **nothing in the record says which artefact it
saw**. Measured on the verdict files themselves — each entry carries exactly
`lang` and `result`. No md5, no size, no mtime, no build. A verdict so recorded
*cannot* know it has gone stale, and neither can its reader.

So: **a stored judgement about a file must carry an identity for the file it
judged** — an md5, or failing that a size and mtime, and the build that produced
it. Then a consumer can tell a current verdict from a historical one, and a
verdict about a replaced file self-invalidates instead of quietly misleading.
Without it, every derived classification inherits an expiry nobody can see —
which is §14's expired-premise problem one level down, in data rather than in
reasoning.

The tell to watch for: **a new instrument disagreeing with a stored verdict is
not automatically the instrument being wrong.** Twice today the stored record was
the stale one.


## 24. Pin a defect you are knowingly leaving in place

When a defect is measured, understood, and then **deliberately not fixed** — by
an owner decision, a protected-closure refusal, a cost ruling — the decision
lives in a conversation and the code says nothing. The next person to read that
line sees an ordinary line. The one after that quietly "improves" it.

So write a test that asserts the defect is **present**, and name the decision in
its failure message.

On 2026-08-31 the per-language delay fix was refused: `--sync -1:` applies one
language's delay to every track, 48 of 81 judgeable pairs diverge by more than
half a frame, 22 of them already shipped — and the emission site is inside the
protected closure, so the owner declined the exception and the defect stays.
`test_14` now pins that `generate_merge_command_other_part` still contains
`--sync -1:` and still reads `delays[common_language_use_for_generate_delay]`.

It looks perverse and it is the opposite. Someone editing that line quietly now
meets a failing test that tells them an owner decision is attached to it, rather
than meeting nothing at all. And if the exception is ever granted, that test is
the thing that must be updated — so the decision is **found** rather than
forgotten.

The general rule: **an accepted defect needs a marker in the code, not only in
the record.** A decision recorded only in prose has the same shelf life as §14's
expired premise, and the same failure mode — it stops being true and nothing
notices. The test is the marker that cannot be skimmed past.


## 25. When a bound explains your result, check WHOSE bound it is

A harness constant fits the data perfectly, because it made it.

On 2026-08-31 six files were reported as failing because "the delay search is
±3 s and the offset is 4.43 s — a search that cannot reach the answer returns
the best thing inside its window". The measurements were real: ±3 s gave peaks
of 0.15-0.29, ±90 s gave 0.995-0.999 on the same pairs, and one reading railed at
exactly **-3000.0 ms**.

**There is no ±3 s in the delay path.** `audioCorrelation.correlate` sets
`span = len(fingerprint) - min_overlap`, so the reachable lag scales with the cut
length — order **±101 s** for a 1400 s episode. The only 3.0 in the codebase is
`merge_video_chimeric.VERIFY_MARGIN_S`, the chimeric verifier's residual rail,
which bounds what one checkpoint can express and has nothing to do with the
search. The ±3 s was `WIN, MARGIN = 20.0, 3.0` — a module constant in the
analysis script, copied from the shape of `measure_residual_ms`.

The real cause was one lookup away in a tool everyone had used all day: all six
raise `NO_COMMON_LANGUAGE`, master `jpn` against candidate `eng`, at the language
gate **before any correlation runs**. There was no search to be too narrow.

The tell was present and was misread. A rail at exactly −3000.0 ms is
`VERIFY_MARGIN_S` in milliseconds — the number identified its own owner, and was
attributed to production instead. **Distrusting a railed reading is not enough;
you must also ask which instrument the rail belongs to.**

Two things stopped this becoming a fix to the wrong function, and both were
refusals. One agent traced the band, found its own tidy explanation contradicted
by the cut-length arithmetic, and reported that it could **not** close the
question rather than shipping the plausible account. And the Lead, having already
relayed the claim as fact, said so — which turned an hour of re-derivation into a
five-minute check. **Saying "I have already repeated this" is what makes a
correction cheap.**


## 26. A correction lives in the function that received it, not in the project

A lesson learned once is not thereby learned everywhere. On 2026-08-31 a new
dub-integrity instrument condemned **35 of 35** tracks, because any single
low window sank the whole track — and that is a mistake this project had
already made, diagnosed and fixed. `batch_verify.judge_profile`'s own docstring
records it: a real edit shifts everything after it and shows as a **persistent
run**, never as one window, and taking the worst window had already sunk two
good merges.

The fix had lived in that one function's docstring for hours. A different agent,
building a different instrument for a different question, reimplemented the
original error from scratch. Neither was careless; the correction simply had no
way to travel.

This is the same shape as the three offset measurers with three acceptance rules
(§A3), one level up: **a policy characterised in one instrument is not
automatically applied in the next one written.** Measurement bias, dissent
handling, floor semantics — each has now been corrected once and reintroduced
at least once.

Two things follow. When you correct an *acceptance policy* rather than a bug,
put it where a future instrument's author will meet it — an invariant, or the
shared module — not only in the docstring of the thing you fixed. And when you
write a new judging instrument, read how the existing ones treat a dissenting
sample **before** choosing your own rule; the answer is usually already
measured.

The corrected form, for reuse: require a **run** of at least three consecutive
non-matching windows before condemning, and report shorter dips as their own
field on an otherwise clean verdict, so they stay visible rather than being
averaged away or promoted to a failure.


## 27. A work list derived from a field that failure clears shrinks silently

Never compute scope from a field the failure path rewrites. On 2026-08-31 a
re-merge campaign built its work list from `batch_state.merged_file`. When a
rebuild fails, the FAILED result **overwrites the row** and the new row carries
no `merged_file` — so the file drops out of scope, can never be retried, and the
run reports success on a shorter list than it started with. Scope fell 85 → 83
while the campaign watched, and neither file could have been recovered by
re-running it.

**The list gets shorter every time something goes wrong, and nothing says so.**

The fix is two-part and both halves are needed: **snapshot the scope once and
persist it**, so the work list cannot be re-derived from a mutated record; and
take a **manifest key as independent evidence that an output existed**, so the
ledger is not the only witness to its own contents. Corrected, the count went
83 → 87 and matched an independent count for the first time — because it no
longer depended on `batch_state` being complete.

This is §23's problem from the other end. There, a stored verdict outlived the
file it described. Here, a stored field is *erased* and takes the work item with
it. Both are a record being trusted as the world.

**It also produces orphans — but it is only ONE of two paths, and this entry
said otherwise for an hour.** A file whose ledger row is overwritten by a failure
sits on disk referenced by nothing. Two were created that way while the campaign
ran, directly observed. It was then extended to the two orphans found earlier,
and that extension was **wrong** — a true mechanism applied past the evidence
that established it.

The discriminator is timing, and it is decisive:

| | evidence | mechanism | fix |
| :--- | :--- | :--- | :--- |
| **path 1** | a failure block **newer** than the output | the row was overwritten and the reference cleared | preserve `merged_file` across a failure |
| **path 2** | **no log at all**, or a failure **older** than the output | no row was ever written | every `/fusion` caller writes a row |

Measured: one orphan has **no archived log and no container outcome** — never
processed by the batch, so there was never a row to overwrite. The other's
output is **4.5 hours newer** than its recorded failure, which is not a weaker
fit for path 1 but the wrong causal order. Both are path 2: a tool that posts to
`/fusion` and never writes to the ledger.

**Fixing either path alone leaves the other producing orphans**, and until the
timing test was run they looked like one problem. A flag that fires only when a
campaign recorded an output before running a file and found the reference gone
after is path 1 *by construction* and cannot misattribute — but it must not be
named as though it covered both.



**Amendment, found the same day by a third path — success can thin a population
too.** Agent 4's dub pass selected outputs written *before* the fpcalc gate at
12:03:25Z. As the rebuild campaign rewrote each file, its mtime moved past the
gate and the file left the target set: 54 pre-gate targets at the start, 34
after, and of 19 files pulled for re-profiling only 8 came back. The pass was
about to report **"43 files, 0 MIXED SOURCES"**, and the shrinking denominator
was one of the reasons that number was not a clean bill.

**Correction, entered the same day.** When this was first written the population
shrink was given as *the* explanation for that zero. It was not. The pass had
also been raising `NameError` on every single file -- a rename left the cache
tag referencing an unbound `master_index` -- so nothing was measured at all. The
shrink is real and separately evidenced; it was not the cause of the zero. Kept
here as written and corrected rather than quietly edited, because attributing a
true finding to the wrong mechanism is its own failure and the agent that found
it flagged the misattribution against its own earlier report.

So the rule is wider than a field that failure clears. **Any selection
predicate over mutable metadata -- mtime, a status field, a reference that
other work rewrites -- defines a set that changes underneath the work list**,
and it thins fastest when the other work is succeeding. The answer is the same
one that saved this pass: select against an immutable copy. The 81 md5-verified
pre-gate outputs in the preserved baseline still answer the question the files
on disk no longer can.

## 28. A constant offset is not the normal case in this corpus

Any method that corroborates a measurement by checking that independent windows
**agree on one lag** assumes the pair has a single offset. Measured on
2026-08-31 over the 74 single-language pairs:

    corroborated, constant offset      8    E_corr 4, D_trim 4
    disagreeing, varying offset       66    D_trim 33, C_cut 14, B_fps 12, E_corr 7

**Every single-language C_cut pair and every single-language B_fps pair
disagrees**, and not marginally: window-lag spread runs 167 ms to 13 076 ms with
a **median of 2065 ms**, against a half-frame bar of 20.9 ms. Two seconds, not
two frames.

That is not measurement failure — none of the 66 is an extraction failure, and
each has three or more usable windows that simply disagree. **A varying offset
is what these files are.** Having one is largely *why they are error files*: a
structural edit or a speed mismatch does not produce a single lag.

So consistency-based corroboration reaches 8 of 74 — 11% — and is false
precisely for the population it was built to serve. The general lesson is the
one to carry: **before corroborating by agreement, check that the quantity is
expected to be constant.** On a corpus selected for being broken, the well-
behaved case is the exception.

Two corroboration methods on disjoint populations do agree on one thing: 3.1%
of cross-language-corroborated alignments and 3.6% of consistency-corroborated
ones sit **below** `VERIFY_MIN_PEAK`. Correct alignments are being discarded as
"too little structure to judge" at a small but real rate. That does not license
lowering a constant floor — non-matches run 0.0930–0.2151 against a corroborated
minimum of 0.3070 — it licenses judging against what the content can achieve.


## 29. Do not re-derive a value the pipeline already computed for you

A harness re-derived a candidate sample time as `master_t + lag` where the
convention is `c = m − lag`. It sampled the candidate at **twice the lag away**
— tens of seconds, far outside the search band — and did not crash. It produced
a complete, plausible, entirely wrong table:

    wrong sign            corrected
    30% corroborated      85%
    ceiling min 0.8899    0.9439
    32 disagreeing        7

The conclusion it supported was the **reverse** of the truth — that a plateau
table's segments do not reliably delimit constant-offset regions — and would
have impugned another agent's work and sent someone to re-measure 180 records.

**The arithmetic looked fine. What caught it was physical implausibility**: lag
spreads of 2000–5000 ms *inside a single plateau* cannot happen if the plateau
is correct. When a result is wrong by a factor rather than a margin, the
magnitude is the tell, not the code.

The fix was not a corrected sign. `normalize_plateaus` already returns
`c_start`/`c_end`, so the harness now interpolates on both timelines and
re-derives nothing. **A value the pipeline already computes is a value you
cannot get wrong.** Every re-derivation is a second implementation of a
convention — sign, origin, units — that only the first one is tested against.

Same family as taking a bound from your own harness (§25) and re-searching for
an answer you already hold (§18): the error is not the arithmetic, it is
recomputing something the system had already settled.

## 30. A query scoped to a name that does not exist answers "nothing", not "no such name"

Three instances in one hour, by two different people, all in the act of
investigating something else.

**Mine, twice.** I ran `git log --oneline 034776c..da24bac -- mergeVideo.py`
and got an empty result. The file is `src/mergeVideo.py`. Git does not object
to a pathspec that matches no tracked file — it reports, correctly, that no
commits touched it. I read that as *the merge code did not change across 301
commits* and was one message away from telling an agent its regression could
not exist. The real answer is 7 commits pin→container and 10 more
container→HEAD, +833/−42. Earlier the same hour I looked for
`campaign_progress.json` in `/config/output/vmsam_agent/`, found nothing, and
asked the agent whether its resumability was working. The file had 30 rows and
had been flushing after every file for two hours and forty minutes, in the
agent's scratchpad.

**Agent 3's, in code.** `campaign.py:69` binds `state = bv.load_state()` at
module level; `output_of()` closes over that dict. So `output_before` and
`output_after` re-read one startup snapshot, `if output_before and not
output_after` is unreachable, and the orphan flag reported 0 for the whole run.
A fresh `load_state()` shows 14.

**The shape is the same in all three**: an instrument whose "clear" reading and
whose "not looking at anything" reading are the same value. A wrong path, a
wrong directory and a stale binding are three ways to be pointed at nothing,
and none of them raise.

**The check is cheap and it is the same check every time: make the query
succeed once before you trust it failing.** Ask for the thing you know is
there. `git log -- src/mergeVideo.py` with no revision range must print
commits; if a pathspec returns empty unfiltered, the path is wrong, not the
history. `load_state()` must return a different object on a second call, or it
is not reading. A search that has never returned a hit has not been tested,
it has only been run.

Kin to §15 (a guard that fails open) but distinct: a failing-open guard is
asked the right question and answers wrongly. This one is never asked. It is
also §13's twin — a statistic over one window is vacuously perfect, and a
predicate over an empty set is vacuously clean.


## 31. Calibrate the instrument the run uses, not one beside it

A dub-integrity pass had a `--calibrate` mode that built known-good and
known-bad controls and reported `CALIBRATION PASSED`. It exercised `profile()`
and `judge()`. The pass itself ran `stitch_profile()` and `judge_stitch()` --
a second implementation, never controlled. So calibration kept passing while
the instrument that produced every reported number was raising `NameError` on
every file it touched, for the entire life of two commits.

**This is §12 inverted.** There the danger was a check never shown to fail;
here it is a check that only ever failed, reported as a pass -- because the
thing being checked was not the thing being run.

Three defects had to line up to print it as a clean bill, and each alone is
survivable:

- `len(results)` counted **store entries** rather than files measured in this
  run, so 35 untouched older records padded the total to 43;
- a failure was stored as `{"error": ...}`, a **non-empty dict**, so every
  truthiness test read it as a profiled file -- including the audit written to
  catch exactly this, which cheerfully reported "43 with tracks, 0 empty";
- **`0 MIXED` is precisely what total failure produces.**

The log even said `34 pre-gate targets, 35 already done, 8 to profile`, which
is arithmetically impossible, and it was read straight past.

**The rules that come out of it, all cheap:**
1. A control must drive the same entry point as production. Two implementations
   means two calibrations or one is decoration.
2. Count what this run measured, never what the store holds.
3. A stored failure must not be truthy. Test for success, not for presence.
4. Print failures first, and **exit non-zero when any input went unmeasured** --
   a summary that cannot distinguish "clean" from "never ran" is invariant 30
   wearing a report's clothes.

## 32. A fixture that cannot express the failure tests nothing, however real the code path

The comparator's self-tests were not stubs. `compare_one` and `measure_output`
were exercised end to end with only the decode replaced — exactly the discipline
§31 asks for, and it still proved nothing.

**The fixture gave each language a single audio track.**
`generate_launch_merge_command` emits the best video's tracks first, so within a
language the first `n_master` streams are **master-derived** and anything beyond
is **candidate-derived**. Only the candidate-derived track can be out of sync;
the other is the master's own audio, copied. `measure_output` took `stream[0]`
of each language and correlated it against the master — **the master against
itself**. With one track per language in the fixture, master-derived and
candidate-derived were the same stream, so the defect could not be expressed by
any input the suite could supply. 64 tests and 36 of 37 mutants passed over it.

**What the wrong answer looked like: fifteen files, every language, PASS → PASS,
every offset inside 0.05 ms.** Clean, uniform, and worthless. Nothing about it
invited a second look.

**It was caught by disagreement, not by testing.** The rebuild campaign filed
err_20 as `MERGED_OUT_OF_SYNC` at 180 ms on eight languages while the comparator
read 0.00 ms on ten. Measuring the *candidate-derived* track of that same file
gives −180.04 ms on `fre` and −180.05 on `spa`, and the corrected instrument now
fails on exactly the eight languages the campaign named. **Two instruments
contradicting each other is the cheapest defect detector this project has**, and
it is the only reason the corrected instrument is trusted.

`batch_verify` documents the same trap with the opposite sign: one master
carried two `jpn` tracks 131 ms apart, so "the first two tracks of this
language" compared the master with itself and reported a desync the merge never
caused. **The direction matters for how long you live with it** — a false
desync sends someone hunting; a false *sync* looks like success and nothing in
the output appears wrong.

So the rule is not "route self-tests through the real path", which was already
being done. It is: **the fixture must be able to express the failure.** For any
selection over a sequence, that means the fixture carries at least two elements
that differ in the way the selection is supposed to discriminate. One track per
language cannot test which track you picked.

**Ordering that the tests must encode, since it is a real invariant of the
output:** within a language, the first `n_master` streams are master-derived and
the remainder candidate-derived. A mutant taking `stream[0]` must die.

**And the cost is worse than a gap in coverage.** A passing test over a failure
the fixture cannot express is not *zero* evidence, it is **negative** evidence:
it discharges the suspicion that would otherwise have kept looking. 64 green
tests and 36 dead mutants are why nobody re-read `measure_output` for a day.
The same shape as the near-miss that produced this bug -- a true partial
explanation ("the file was never rewritten") accounting for some of the
uniformity and thereby retiring the question for all of it. **A true partial
cause is more dangerous than no cause**, and a green suite over an
inexpressible defect is the automated form of one.


**The same rule governs which INSTANCE you test, not only which fixture you
build.** Hours after writing the above, the Lead proposed a decisive one-run
test for a refusal mechanism gated on `multitrack_reference` -- a flag that is
only reached when the master carries more than one track in a language -- and
named err_9 as the file to run it on. err_9's master has **no language with more
than one track**, so the gated path is never entered and the flag cannot change
its outcome. The run would have returned "still refuses" and been read as
refuting the hypothesis, when it would not have tested it.

Checking the necessary condition across all 14 candidates cost one pass over
their stream headers and showed **8 of 14 could not reach the mechanism at
all**; only 6 carry the `spa: 2` condition that makes it reachable. So the test
is real, on those six, and would have been thrown away by running it on the
other eight.

**Before running a discriminating test, state the mechanism's necessary
condition and confirm the chosen instance satisfies it.** A negative result from
an instance where the mechanism is unreachable is not evidence against the
mechanism -- it is the same non-measurement as an empty query or an
inexpressible fixture, wearing the costume of an experiment.

## 33. Count groups, not rows — this corpus is series, and series are not independent

Two findings died to this in one hour, and the second was caught by the agent
that produced it.

**The refutation.** A proposed mechanism required masters carrying more than one
track in a language. Grouping the completed rebuilds by (series, languages the
candidate adds) produced **nine structurally identical files of one series that
split six failures to three successes** — same master shape (9 tracks,
8 languages), same duplicate `spa: 2`, same candidate languages. The gated
condition was present in **all six failures and all three successes**. A path
that fires on all nine cannot be what distinguishes six from three. That control
cost one grouping and settled what a container run could not.

**The self-deflation.** The replacement finding looked overwhelming: every one of
14 failures had a candidate contributing no language the master lacks, against
12 of 17 successes that did — 14/14 versus 12/12. Grouped by series, **those 12
successes are 3 independent groups**; ten share one series and the identical
added set. The honest claim is "3 of 3 groups", which is n=3 and proves almost
nothing.

**The corpus is built from series, and files within a series share master
structure, track layout, encoder and provenance.** Treating their rows as
independent observations inflates every count by the group size, and it inflates
the *impressive* direction hardest, because a large uniform group is exactly what
produces a clean ratio.

**So: state findings as "k of n groups", never "k of n files", and say what the
grouping key was.** A within-series split — same series, same structure,
different outcome — is worth more than any cross-corpus ratio, because the
series holds constant everything you did not think to control for.

**Corollary, from the same data.** The intuition that duplicate same-language
master tracks were causing the failures had it backwards: 41% of masters carry
one, and they run **15 merged / 6 failed** against **2 merged / 8 failed**
without. Three verification tools tripping on duplicates (§32) made them look
guilty; they are associated with success. **A property that breaks your
instruments is not thereby a property that breaks the pipeline.**

## 34. A search that matched nothing still returns a number, and it looks precise

A change point was cited as **`691.4 ± 1.4 s`** and propagated into a commit
message and another agent's tool. It was withdrawn: **all five refinement probes
returned `matched=neither`**, so the bisection ran on no information at all and
returned its **midpoint**, while the `± 1.4` is just the starting bracket over
`2^5`. The honest answer was **690–780 s** — which contains the 718.5 s another
agent had measured directly.

So a disagreement between two agents was manufactured out of nothing, and the
agent that did it named the reason exactly: **it preferred a decimal point to a
colleague's measurement.** Precision is a property of the *search*, not of the
number it prints. A bisection reports midpoint ± bracket/2^n whether it
converged on evidence or on nothing, and those two outputs are indistinguishable
at the call site — §30 again, in the shape of an interval.

**Every iterative estimator must report how many probes actually discriminated,
and refuse to narrow its interval when the answer is zero.** A wide honest
bracket that overlaps a colleague's figure is worth more than a tight invented
one that contradicts it. When a stated interval disagrees with a direct
measurement, suspect the interval first.

## 35. High fidelity after a dip means locally consistent, not aligned

`fidelity()` searches over shifts, so a **constant offset inside a window is
absorbed** and the window reads near-unity. A track that dips and then returns
to high fidelity was described as having "recovered completely". It had not: it
settles onto a **third plateau 831 ms off the master** and stays there for
twelve minutes, reading near-perfect the whole way.

**A shift-searching statistic answers "is this window internally consistent",
never "is this window in the right place".** Alignment needs the *offset* the
search chose, not the score it returned. Any conclusion of the form "it
recovers" or "it is fine after this point" requires reading the lag, and a
sequence of high-fidelity windows at three different lags is a desync with
excellent local scores.

Kin to §21 (`_xcorr_offset` is not a correlation coefficient): both are cases of
a number being read as answering a question it was never computing.

**Qualification, and it must travel with the rule.** This invalidates *"peak
0.99, therefore in sync"*. It does **not** invalidate measuring sync by
cross-correlation: **if you take the lag the search chose, the lag IS the
answer and the trap does not bite.** Judge alignment on the lag; use the peak
only to decide whether that lag is trustworthy. Stated as a bare headline this
entry has already been passed to an agent whose instrument thresholds on lag,
where it would have caused it to distrust a method that was sound — so quote
the qualification or do not quote the entry.

## 36. A knee in your data at a multiple of your own constant is the constant

`located` was believed to be a property of the audio: clear boundaries resolve,
ambiguous ones do not. Measured across **478 refined change points** in every
table, against the size of the step being located:

    step size        located   not located
    0-250 ms            12         140
    250-500 ms           4           6
    500-1000 ms         14          17
    1000-2000 ms        54          49
    2000-5000 ms        20           0
    >5000 ms           158           4

    below 2000 ms :   84 of 296 located =  28%
    above 2000 ms :  178 of 182 located =  98%

**The knee sits exactly at `2 x DISCRIMINATE_BAND_S`**, a constant of 1.0 s in
the probe, not at any feature of the corpus. The discriminator searches ±band
around each candidate lag; when the two lags are less than twice the band apart
the two searches **overlap and find the same peak**, so neither can win the
margin. On err_9 the lags are 1000.8 ms apart, the bands overlap by 999.2 ms,
and both probes return exactly 1.000 — saturation, not ambiguity. Rescaling the
band to `step/4` located it 4/4 at 15-16x ratios, at 732.00 s instead of a
meaningless 691.41.

**So 212 of 296 sub-2-second change points are unlocated for a structural
reason that has nothing to do with the file.**

**The scanner had already documented the symptom and named the wrong cause**: a
comment recording that "every step above a second located decisively and every
step below half a second located not at all", written up as a property of the
corpus — which made the unlocated cases *tolerable*, because they were the small
ones. The observation was correct and the explanation was invented, and the
explanation is what stopped anyone looking.

**The rule: when the outcome rate has a threshold, express the threshold in
units of your own parameters before attributing it to the data.** If it lands on
one of them, or on a small multiple, the parameter is the cause. This is §25
sharpened — there it was a bound explaining a single result, here it is a
constant explaining the *shape of a distribution*.

**Corollary on pre-registration.** The prediction here was falsified *and* the
pre-registered fallback conclusion ("then the ambiguity is in the file") was
also wrong — and the harness printed that fallback automatically before anyone
read the probe values underneath. Pre-registering a conclusion is worth doing,
but **a fallback branch is a hypothesis too, and automating its verdict makes it
the default answer rather than a checked one.** Read the numbers under any
verdict your own script prints.

## 37. A pass that degrades the machine it measures on is invisible to every check that reads its output

The dub-integrity pass wrote every 30 s window (~1 MB) and every decoded track
(~48 MB) into `/tmp`, and `/tmp` **is tmpfs — it is RAM**. Nothing was deleted
until the whole run finished, so the footprint grew monotonically: **6.2 GB
after 13 of 81 files**, on a box already at load 50 with zero free memory. A
per-file clear was not enough either — a single file reached 2.6 GB, because it
decodes its own copy of every master track for every output track compared.
Dropping windows as soon as they are fingerprinted and tracks when the profile
is done took it to **12 MB resident**.

**Every verdict was correct, every count honest, the log clean.** Nothing in the
results could ever have shown this. It was found by looking at what the process
was *doing* — the scratch directory, the child `ffmpeg`, the tmpfs figure —
rather than at what it was *saying*.

**And the failure would not have landed on the agent that caused it.** Memory
would have gone while somebody else's decode was running, and the report would
have named the victim. Resource exhaustion is the one defect whose blast lands
off-target, which is why it cannot be left to the owner of the offending run to
notice.

Checked across the other long-running instruments the same hour: one orphaned
scratch directory of 36 window `.wav` files, **91 MB held in RAM for 13 hours**
after the run that made it was killed. `mkdtemp` with cleanup on the success
path is not cleanup — a killed run leaves its scratch behind forever, in memory.

**So: for any long-running instrument, measure resident scratch as well as
results.** Free windows at the point they stop being needed, not at the end of
the run; treat `/tmp` as RAM unless proven otherwise; and sweep for orphaned
scratch from killed runs rather than trusting a cleanup path that only executes
when nothing goes wrong.

## 38. A stored table is only as good as the tool that wrote it, at the time it was written

`cut_points_c.json` holds **530 change points for the C_cut class — the largest
class in the corpus, 180 files.** It was written **2026-08-30 08:15Z**. The
commit that fixed how a change point is bracketed and located, `b965fa8`
("locating a cut is discrimination, not another search"), landed **19:17Z the
same day — eleven hours later.**

Verified independently of the agent that raised it, by counting the provenance
fields the current code always writes:

    cut_points_audit_57.json      159 pts   lo_s 159   refine_iters 159
    cut_points_seven.json         166 pts   lo_s 166   refine_iters 166
    cut_points_d.json              98 pts   lo_s  98   refine_iters  98
    cut_points_d_palcorrected      52 pts   lo_s  52   refine_iters  52
    cut_points_c.json             530 pts   lo_s   0   refine_iters   0   <-- pre-fix
    cut_points_c.measured.json    546 pts   lo_s   0   refine_iters   0   <-- pre-fix
    cut_points_conflicting_0240   156 pts   lo_s   0   refine_iters 156   <-- partial

**Not one of the 530 was ever bisected.** The pre-fix bracket was
`[before.end_s, after.start_s]`, and the fix's own comment describes what that
does: it *"searches a range that need not contain the answer, and the bisection
then returns its own edge, which looks like a converged result."*

**And the table is not a report — it is an input.** `mergeVideo.py:4254` loads
`cut_points_c.json` and `cut_points_d.json` as siblings unconditionally, so
those 530 positions are live in the merge path for every C_cut file.

**The rule: a stored artefact carries the defects of the tool at its write time,
not the tool in the tree.** Re-reading it gives no hint — the numbers are
well-formed and the file is valid. **Stamp every generated table with the commit
that produced it, and refuse to consume one written before a fix to the
producer.** Where a stamp is absent, the presence or absence of fields the
current code always writes is the next best provenance test, and it is what
settled this.

Kin to §23 (a stored verdict is evidence only while its file is unchanged): there
the *input* moved underneath the verdict, here the *producer* was corrected after
the output was written. Both make a stored value quietly stale while looking
exactly as it did when it was true.

**Amendment: the provenance test could not see the table that wins.**

The audit above swept eight tables and looked exhaustive. It omitted
`cut_points.json` — **57 records, 161 change points** — which is the
*configured* table, appended **last**, so it takes precedence on overlap over
both siblings. It was skipped because it is a dict `{method, counts, results}`
rather than a list, so the parser found zero points in it and moved on. The
skip printed nothing: `if not pts: continue`. **An audit that parses one schema
silently omits the records written in another, and still prints a table that
reads as complete** — invariant 30, inside the verification of invariant 38.

It is a fourth generation of the tool, with no field in common with the
provenance test:

    approx_timestamp_s  timestamp_s  timestamp_precision  between_s
    refined  interpretation  lag_before_ms  lag_after_ms  delta_ms

    lo_s 0/161   hi_s 0/161   refine_iters 0/161   bracket_s 0/161

**It is not thereby defective.** It self-reports `change_points_localized_to_1s:
133 of 161`, and `mergeVideo.py:4300` handles its shape deliberately via
`records.get("results", [])`, so the format is supported rather than accidental.
The honest statement is that **it is unassessed by the method that settled the
others, and it outranks them where they overlap.**

**So: before concluding an audit covered a population, count the members it
parsed against the members that exist.** A per-schema parser reports absence and
non-applicability identically, and the highest-precedence record is exactly the
one most likely to be newest and therefore in the newest schema.

## 39. A mutant that does not run the code it replaces is a description, not a test

Two judging rules were corrected in the dub instrument, and the first draft of
each test **asserted around** the old behaviour — stating what the new rule
does, without ever executing the rule it replaced. Rewritten, each mutant now
**reproduces the pre-fix logic and shows it misfiring on a real shape** from the
corpus: the tail-matching `ger` track that the asymmetric block rule condemned
on 2 matching windows of 49, and the gap-straddling `eng` track the 0.90 floor
sliced into alternating blocks.

The difference is whether the test can fail for the original reason. A test
written against the new rule passes on any implementation that happens to agree
with it today; a mutant carrying the old rule fails **exactly when the defect
returns**, which is the only property that matters after everyone has forgotten
the argument.

**Two rules from the same work, both about thresholds:**

**A threshold is only meaningful across an empty band.** The corpus calibration
puts same-recording pairs at 0.904–0.993 and different-recording pairs at
0.563–0.847, with **nothing between** — that gap is what makes a 0.90 floor a
decision rather than a cut through noise. Seven tracks read as MIXED with 35–40
of 49 windows sitting *inside* the gap; the alternation was the threshold
slicing a region where the calibration says no real pair lands. **When the
evidence sits in the empty band, the confidence in the verdict is coming from
the threshold, not from the data.** Report `undecidable`, and keep judging the
cases that are bimodal — err_9 at 45 same / 4 different / 1 between was still
decided.

**Keep the raw per-window series, not just the verdict.** Three judging defects
were corrected over the *stored* series with no re-decoding. A long pass will
outlive at least one correction to its own rules, so storing the raw
measurements is what lets a threshold be revised instead of becoming
load-bearing — and what stops a two-hour run forcing a choice between finishing
and being right.

## 40. An option that reads a field nothing writes is inoperative, and says nothing

All 180 records in the C_cut table are stamped `master_source: "guess"`, and
today **not one of them can be reproduced** — `--allow-guess` resolves zero.
The cause is not that the guess logic changed:

    resolve_pairs():  guess = entry.get("master_file_path") or entry.get("master")

    private mapping:  309 entries
                      fields present on all 309:
                        destination_path  episode  error_file_path
                        error_id  folder_id  private_reports
                      entries carrying master_file_path or master:  0 of 309

**The field the guess reads does not exist in the mapping at all.** So
`--allow-guess` cannot fire for any input; the flag parses, the run succeeds,
and it resolves nothing. The 180 records were written when that field existed;
the mapping has since been regenerated without it, and **113 of the 180 records
are therefore unreproducible by any scan we can run** — the 67 that do resolve
come from the database instead.

This is worse than a stale table (§38). A stale artefact can be regenerated and
compared. **An artefact whose producer can no longer be run is evidence with no
possible second opinion**, and nothing about reading it reveals that: the
records are well-formed and carry a `master_source` stamp that looks like
provenance while naming a path that can never be followed again.

**Two rules:**

1. **An option whose input is absent for every record must say so and fail, not
   proceed quietly.** `--allow-guess` returning nothing is indistinguishable
   from a corpus where no guess applies. (The per-record skip reason here is
   honest — *"and no filename guess exists"* — but the flag itself never reports
   that it is inoperative across the board.)
2. **Stamp generated artefacts with what would be needed to reproduce them, and
   check that it still exists before trusting them.** `master_source: "guess"`
   was the right stamp; what was missing was anything that noticed the guess
   source had gone.

## 41. A compatibility mode must be tested by reproducing the defect, not by existing

When the probe band became `scaled` by default, `fixed` was kept so older tables
remain reproducible. **The only thing that makes `fixed` useful is that the old
defect still happens under it** — an older table can be regenerated exactly when
the overlapping searches still overlap.

So the test does not assert that the mode exists, or that a flag parses, or that
some position comes back. It asserts that under `fixed` **the two searches still
overlap on a 1000.8 ms step**. A test that only checked the mode was selectable
would pass on a `fixed` that silently behaved like `scaled` — a *helpful* change
that destroys the reproducibility the mode exists to provide, and one nobody
would think to look for because everything got better.

**Generally: a legacy mode's contract is the old behaviour, bug included.**
Test it by exercising the behaviour, never by naming it. This is the single
place in this codebase where §39's rule inverts — there a mutant must reproduce
the pre-fix logic to prove the fix works; here the *shipped* code must reproduce
it to prove the compatibility promise is real.

**Corollary.** When a ruling changes a default, the self-check that pinned the
old one **is updated to pin the new one, not deleted.** A decision that leaves
no test behind is reverted by the next person who finds the constant surprising
and has no way to know it was chosen.

## 42. A registered outcome must carry the evidence that establishes it, not only the condition that triggers it

A prediction was pre-registered before an experiment, with branches, and the
harness evaluated its own result against them to remove the author's discretion
at read time. That is the right instinct and it produced a **false conclusion
with more authority than a guess would have had.**

Branch two's *condition* was "the probes return `neither`". They did, so the
harness printed branch two's *conclusion* — "the ambiguity is in the file, the
legacy flag has been honest". **Both halves were false.** The probes had
returned `neither` because their search bands overlapped and both found the same
peak; nothing about the file was ambiguous and the flag was reporting an
artefact of its own band.

**The registration encoded a false dichotomy and then automated it.** The
bad-test guards worked and would have caught a false positive; the gap was on
the other side — nothing in the registration could distinguish *"measured, and
ambiguous"* from *"did not measure"*, which is invariant 30 promoted to the
status of a pre-committed conclusion.

**So the rule is narrower than "pre-register your predictions":** each branch
must state the **evidence** that would establish it, not merely the observation
that would trigger it. Branch two should have required *"the probes
discriminated **and** disagreed"* — a condition that would have matched nothing,
forcing the unregistered third explanation into the open instead of quietly
selecting a written-down wrong answer.

**And a prediction is not edited after the fact.** The registration here was
left unchanged with a pointer to the result, which is what makes it evidence
about how the reasoning went rather than a record of having been right.

## 43. A join key stricter than the precision of what it keys on reports total turnover

A diff between two scans of the same 67 records first reported **147
disappeared, 115 appeared, 0 unchanged**, and 12 points moved by *exactly*
1.50 s. Total turnover with a suspiciously uniform movement.

The cause was the key. Matching was done on the lag pair rounded to 0.1 ms with
exact equality — and two scans of the same pair estimate the same lag as
**-1090.1 and -1090.3 ms**. Nothing matched, so every point looked new and every
old point looked gone.

**Total turnover looks like a finding.** "The corrected scan replaced every
boundary" is a dramatic, publishable sentence, and it was an artefact of a key
demanding more precision than the measurement carries. Re-keyed with a 5 ms
tolerance and closest-candidate-wins, the real answer is 91 moved, 60
disappeared, 28 appeared, 8 unchanged.

**Match on a tolerance derived from the measurement's own precision, and test
both directions** — 0.2 ms is the same edit, 2000 ms is not. A tolerance with
only the first test is a key that matches everything.

**And the railed lags underneath it, which is the domain finding.** Of the 60
points the corrected scan dropped, **55 rest on lags beyond ±20 s** — the old
table contains boundaries built on lags of **-156,884 ms and +172,489 ms**.
Those are railed correlations, not measurements. **71 of the 180 records carry
at least one**, and the old scan turned each into a plateau boundary the
assembler cuts on. The corrected scan refuses them.

So a re-scan of a stale table is not only a precision exercise: it can remove
boundaries that never should have existed. **When auditing a derived table,
check the plausibility of the inputs each row rests on, not only the row's own
value** — a change point at a sane timestamp can be derived from a lag that is
physically impossible, and nothing in the timestamp shows it.

## 44. Audit a stale table with the producer's own acceptance rule, not a threshold you invent

A stale change-point table was to be filtered by dropping points resting on
implausible lags, with a **±20 s** cut-off. Asked to defend the number, the
agent that proposed it checked the distribution instead and withdrew it:

    lag magnitudes across 1092 values: CONTINUOUS
      10-20 s  36 points     20-30 s  54     30-50 s  64  ...
      largest gap anywhere between 1 s and 200 s: 10.4 s, out at 125 s

**No empty band.** Unlike the dub calibration (§39), where 0.847–0.904 is
genuinely vacant and makes a floor meaningful, here ±20 s was an observation
about the examples that had been looked at. A filter built on it would have been
a guess with a number attached — and the number would have acquired authority
from being in the code.

**"Railed" was also withdrawn**, and it is the more instructive retraction. The
boundaries were described as resting on *railed* correlations. Only **4 of 1092**
lags sit at or beyond the search band edge; a −156,884 ms lag inside a ±360 s
band is not railed against anything, it is **a wrong answer the correlator was
free to return**. The word had been doing the work of a measurement, and it
carried further than the evidence because it sounded like a diagnosis.

**What actually separated the dropped points was the producer's own criterion.**

    disappeared  n=257 windows   26% below peak 0.60
    moved        n=966            1%
    unchanged    n= 68            0%

Median peak ~0.98 in all three, so it is not general quality — the dropped points
have a *tail* of unreliable windows holding them up. So the filter became: run
the current `detect_change_points`, `CHANGE_POINT_MIN_PEAK = 0.45`, over the
windows **the old table already carries**, and keep what it still emits.

**Nothing invented, and nothing to defend that the scanner does not already
defend.** Validated against the 67 records that could be re-scanned — the only
honest test available — it agrees on **150 of 159 points (94%)**, and the nine
disagreements lean safe: 8 it would keep that the re-scan dropped, 1 the other
way.

**The general rule: when auditing an artefact produced by a tool you still have,
re-apply the tool's current acceptance rule to the stored evidence.** It needs
no new constant, it is defended wherever the tool is defended, and it can be
validated against whatever subset is reproducible. Inventing a threshold to
audit a table means defending two rules instead of one, and the invented one has
no test but plausibility.

## 45. The same asymmetry recurs one level in — look for it there before declaring it fixed

A peak floor was found applied to the **windows** path but not to **ready-made
plateaus**, so boundaries derived from correlations of ~0.27 against a 0.45
floor reached the assembler. A filter was built to close it, validated, and
measured: 189 unsupported boundaries dropped across 180 records.

Targeting the survivors found the identical asymmetry **one level in**. The
filter asks whether a boundary *time* falls inside a bracket that reliable
windows support. It never asks whether the *plateau* is corroborated — so a
weakly-supported plateau survives whenever its edge happens to sit near a
supported bracket:

    plateaus surviving the filter          530
      mean_peak below the 0.45 floor        24   (5%)
      records carrying at least one         18 of 180

And the pipeline is already inconsistent with itself in the same shape:
`drop_uncorroborated_tail` refuses exactly this **at the tail**, dropping
plateaus at 0.31–0.44 against that same floor and extending the predecessor.
**The interior has no equivalent.** Applied at the edge, not in the middle —
which is the first asymmetry restated about a different axis.

**So when a rule is found applied inconsistently, closing the instance is not
closing the class.** Ask the same question of the next structure down: if the
floor belongs on windows, does it belong on plateaus? If it fires at the tail,
why not the interior? The recurrence is not a coincidence — a rule gets added
where a bug was seen, and the places nobody looked keep the old behaviour.

**And do not fold the second fix into the first.** The measurement validating
the first rule describes *that* rule; extending it silently means the numbers
quoted no longer describe what ships. Land the measured rule, name the residue,
measure it separately.

## 46. A verdict threshold that predates the goal will be counted against the goal

`batch_verify.sync_verdict` returns `in_sync` at anything under **50 ms**. The
mission's end condition is **15 ms**. `master_summary.final_status` counted
`in_sync` as **Resolved**. Nothing reconciled the two, and nothing had to — each
is defensible alone.

    reported Resolved                        71
      actually at or under the 15 ms bar     44
      between 15 and 50 ms, counted as done  27   <- 20, 30 and 40 ms

**38% of the completion count was files measured against the wrong number.** No
row was wrong: `verdict: in_sync` was true under its own threshold and
`passes_15ms_bar: false` was true under the mission's. A reader taking the
verdict at face value records a pass, and the mission's own progress report was
that reader.

**Whenever a project adopts a target, audit every pre-existing threshold that
feeds a count against it.** A module written before the goal answers its own
question correctly forever, and the danger is not that it is wrong but that
something downstream *interprets* it as answering the newer one.

**Report the band, do not silently re-bucket it.** The 15–50 ms files are now a
named status of their own rather than promoted or demoted, because part of the
band may be quantisation: `adjust_delay_to_frame` rounds to the NEAREST frame,
so a correct snap can leave half a frame — **20.85 ms at 23.976 fps — and the
mission bar sits BELOW that floor.** A file at 20 ms may be correctly snapped;
one at 40 ms is 0.96 frames and is not. **Which is which cannot be decided in
milliseconds** — it needs each master's own frame duration, and where that is
absent the file is unclassified rather than assumed.

## 47. Two code paths reaching the same number is the strongest check available here

The permanent dub-verification ceiling was established twice, by paths sharing
no logic: a `--ceiling` mode reading container metadata found **133 of 554**
tracks whose language the master does not carry, and the pass itself — walking
files, resolving masters, selecting streams — skipped **exactly 133** for want
of a reference. Same number, independent routes.

That is what makes 24% a property of the corpus rather than of either tool, and
it is the only class of evidence that has reliably worked here. Every defect
this project has caught was caught the same way:

- a comparator read 0.00 ms where the campaign read 180 ms, and the comparator
  was correlating the master with itself (§32);
- a change point cited at `691.4 ± 1.4 s` disagreed with a directly measured
  718.5 s, and the interval came from a bisection that never discriminated
  (§34);
- an agent's own group medians of 0.266 and 0.984 were arithmetically
  incompatible with its overall 0.290, which found a mis-split population (§43);
- a dub gate refusing at fidelity 0.58 landed on the same files a correlation
  analysis had flagged at peak 0.27, neither having seen the other.

**Not one was caught by a test.** Suites verify that code does what its author
believed; they cannot notice that the belief was wrong. **Two instruments
disagreeing is the cheapest defect detector available, and two agreeing by
disjoint routes is the strongest confirmation** — but only when the routes are
genuinely disjoint. Reproducing a number by re-running the tool that produced
it, or by applying another agent's correction at greater scale, is not
corroboration and has twice been correctly refused here.

**So: before trusting a number that matters, ask what else could produce it, and
build that.** And when a figure is confirmed, state by which two routes — a
reader cannot tell agreement from repetition otherwise.

## 48. The recurring defect of this project: a reading that cannot distinguish two states a decision depends on

Three instances, three layers, one day — found by three different agents, none
of them by a test:

    Only dict_keys([...]) file left    identical text for "two confident
                                       answers 1000 ms apart" and "nothing
                                       locks at all" -- opposite causes, and
                                       they need opposite follow-ups
    sync_verdict -> in_sync            true under its own 50 ms threshold and
                                       read as meeting a 15 ms mission bar;
                                       27 of 71 Resolved were neither
    located = False                    read as "the boundary is ambiguous",
                                       actually "the probe bands overlapped";
                                       212 of 296 points, no file involved

And the same shape everywhere else in this file: an empty query and a clean
result (§30); a calibration that passed and one that never ran (§31); a fixture
that could not express the failure and a fixture that did (§32); a bisection
that converged and one that never discriminated (§34); an option with no
applicable input and one correctly finding nothing (§40).

**The unifying rule: for every reading a decision rests on, ask what OTHER state
would produce the same value.** If a second state would, the reading is not
evidence for the decision — it is evidence that one of several things is true,
and which one has to come from somewhere else. This is cheaper than any test
suite and it has caught more here than the suite ever has.

**Two consequences worth stating separately.**

**A refusal that can be shown correct is worth more than one that is believed.**
err_170 refuses because two audio pairings both find two delays exactly 1000 ms
apart, all four clusters above the 0.90 fidelity bar — a genuine step, and a
single global delay is the wrong answer. A second independent instrument agrees:
17 of 17 windows reliable at peak 1.077, 3 change points found. err_349 refuses
because ten measurements read fidelity 0.549–0.596 with delays scattered across
±2 s — nothing locks. **Both refusals are correct behaviour; the old message
rendered them identically, and that is the point of the entry.**

**Correction, entered by the agent that supplied the example.** The err_349 half
was recorded here as "a second instrument agreed". It did not agree — it
**failed to reach**: 3 of 28 windows reliable, max peak 0.427, at or under the
floor. Against err_170's 17 of 17 at 1.077 that is not corroboration, it is the
absence of a measurement — the exact defect this entry is about, appearing
inside the entry's own worked example. So err_170's refusal is correct **and
explained**; err_349's is correct **and unexplained**, because "different
recording" and "beyond both instruments' reach" produce the same reading and
nothing yet separates them. That distinction decides whether err_349 can be
called `Unmergeable_Verified`, a state claiming we know a file cannot be merged
rather than that we could not measure it.

**Diagnostic value is what a message SEPARATES, not what it adds.** The fix here
did not find a new defect; it split one message into two states that need
opposite follow-ups — repair for one, `Unmergeable_Verified` for the other.

## 49. A field whose meaning changes mid-function keeps its old name and every reader keeps the old meaning

`batch_verify.sync_report` sets `max_abs_lag_ms = max(abs(l) for all windows)`.
Twenty lines later, on an `in_sync` verdict, it **overwrites** it with
`max(consensus)` — the max of only the windows *under* the bar
(`batch_verify.py:243`).

**The overwrite is deliberate and well-reasoned.** A real edit shifts everything
after it, so an isolated dissenting window is not evidence of desync; the dissent
is preserved in `suspect_windows` with a note telling the reader to re-scan
rather than being averaged away. That judgement is sound and this entry does not
dispute it.

**What is wrong is that the name did not change with the meaning.** After that
line the field holds "max of the consensus windows", and two files store **40.0**
against windows of `[40, -1470, 40]` and `[40, -2540, 40]`. Every downstream
reader takes the name at face value:

- `master_summary` bucketed both as 40 ms — just over the 15 ms bar;
- an agent auditing the band read them as medians and reported a second
  aggregation defect that does not exist;
- the Lead then "fixed" `master_summary` to re-derive the true max from
  `samples`, which would have condemned a file on the single window the design
  deliberately discounts. **Three readers, three different wrong readings, all
  from one honest name.**

**Rules:**
1. **A field that is conditionally re-aggregated needs a name that says so, or
   two fields.** `max_abs_lag_ms` and `consensus_lag_ms` cost nothing.
2. **When a value is deliberately narrowed, carry the discarded evidence to
   every consumer, not just into the record.** `suspect_windows` existed the
   whole time and no consumer read it; the summary now prints the dissent beside
   the figure.
3. **Before "fixing" a value that looks wrong, find the line that wrote it.**
   The re-derivation here looked obviously correct and would have overridden a
   documented domain judgement with a naive maximum.

**Corollary, from the same investigation — the case that does not fit two
hypotheses is the test between them, not evidence that there is no rule.**

The agent auditing the band held two explanations for the stored figure: median,
and max-of-consensus. Four files fit both. **The fifth, err_12, separated them**
— windows `[30, 20, 0]`, all under the bar, so consensus-max is 30 and median is
20, and the stored value is 30. Consensus-max explains 5 of 5; median 4 of 5.

That file was in the report. It was read as *"the same field is aggregated two
different ways depending on which path wrote it"* — inconsistency — rather than
as the experiment that decides between the candidates. **Having the
discriminating case and spending it on a conclusion of "no rule" is worse than
guessing**, because a guess can be tested and a declared inconsistency closes
the question.

**When a rule fits most of your cases, the exceptions are the highest-value data
you have.** Enumerate the competing rules explicitly and check which the outlier
supports, before concluding the system is inconsistent. Systems are far more
often consistent under a rule you have not yet stated.

**The mirror case: a fixture that expresses a failure the real data never has.**

Flipping the peak filter on broke a tail-plateau test, and the obvious reading
was that the two rules compose badly on ready-made plateaus. Measured instead of
assumed: on **28 of 28** real records where `drop_uncorroborated_tail` fires,
the final lag and the coverage are **identical with the filter on and off**. The
rules do not interact.

The failure was the fixture. `ONE_WINDOW` carried a single window — the tail's —
which was harmless while no boundary filter existed. With the filter on, a record
showing **no support for any boundary anywhere** is one the filter correctly
collapses to a single plateau. So the test failed for a *true* reason about a
file that does not exist in the corpus.

**Both directions break a green test's meaning**: a fixture that cannot express
a real failure hides a defect (above), and a fixture that expresses an
impossible one manufactures a defect. Neither is visible from the test's pass or
fail — only from asking what real record the fixture stands for. When a test
starts failing after a behaviour change, **check whether its fixture is still a
description of the data** before deciding the change is wrong.

## 50. You cannot screen for a defect using data written before the defect was known

The 119 never-processed files were to be merged using their existing cut-point
records. The question was whether to rescan first, and the appealing middle
option was **rescan only the suspect records** — those whose position converged
on the old bracket edge.

**That option does not exist.** Identifying an edge-converged point requires
`on_bracket_edge`, and across all **387** change points in those records:

    lo_s  hi_s  refine_iters  located  on_bracket_edge :  0 of 387
    the same fields in the post-fix table              : 98 of 98

The records are not merely stale (§38) — they are **unassessable**. The field
that would let you screen for the defect is the field the defective producer did
not write. So the population cannot be triaged; it can only be rescanned whole
or trusted whole.

**Generally: a targeted remediation needs a discriminator that survives in the
bad data.** Before planning to fix only the affected subset, check that the
subset is *identifiable from what you have* — provenance fields, a stamp, a
recomputable property. Where the only discriminator is one the faulty run did
not record, "fix the affected ones" is not a smaller job than "redo them all",
it is an impossible one wearing a smaller job's costume.

**And measure both arms before choosing.** Here the decision looked like a
trade-off and was not, once both were costed from recorded elapsed times rather
than estimated:

    rescan all 113 records   62 min   (median 28.5 s/file, from the table's own
                                       elapsed_s + refine_cp_elapsed_s)
    merge all 119 files      9.8 h    (99.4 GB at a measured 356 s/GB)

**A rescan is 11% of the merge run**, against 387 unassessable positions
relevant to 112 of 113 files, each able to move by up to 60 s. Merging first and
discovering the defect afterwards costs the 9.8 hours twice.

## 51. Mergeability is a property of the file, because a merge needs one anchor

The evidence for "different recording" is **per track** — a candidate track
against a master track of the same language. The *state* it supports is **per
file**, and the reason is mechanical rather than stylistic.

**A merge needs an anchor**: at least one correspondence between candidate and
master from which a lag can be derived. Every other track is then placed
relative to it. So the question that decides mergeability is not *"is this track
alignable"* but *"does this candidate offer ANY alignable correspondence at
all"*. That is a property of the file, established from track-level
measurements.

    a file is a candidate for `Unmergeable_Verified` when it has at least one
    shared-language track, NO shared-language track is the same recording, and
    every control passes

**This explains the 24 mixed files exactly.** Twenty-four files carry one
language matching the master and another not. They have an anchor — the matching
language supplies the lag — so the non-matching track is a separate question
about *that track*, not about the file. They are not an exception carved out of
a count; **they fail the criterion because they have the very thing it is
about.**

**And it rules out the tempting alternative**: modelling unmergeability as a
track property with a file closable once all tracks are closed. That implies a
file with five unjudgeable tracks and one different-recording track is partly
closed, when it is simply a file with no anchor. It also breaks against the
ceiling (§47): a track with **no** same-language reference can never be
"closed", so an all-tracks-closed rule would make those files permanently
unclosable for a reason that has nothing to do with their media.

**The sixth innocent explanation, found by the agent holding the criterion.**
The anchor **does not have to respect the language tag.** The pipeline already
knows this — the language-tag fallback exists precisely because a pair sharing
no tag may still share audio, and it recovers that case at tier 2 by fidelity,
six of six at 0.9532–0.9788. So a file where every *same-tag* pair is a
different recording may still have an anchor through a **mis-tagged** track.
Untested, that verdict would close **a cataloguing defect as a media defect**.

The remedy is an all-pairs cross-tag search over the survivors — every candidate
track against every master track regardless of tag — and it is worth running
whichever way it falls: it either closes the last innocent explanation, or **a
mis-tag found this way is a repair rather than a closure.**

**The general form: before declaring a terminal state, enumerate the ways the
system could still succeed that your instrument does not test.** Six were
enumerated here and five closed by measurement; the sixth was found by asking
what the *pipeline* knows that the criterion does not.

## 52. An append-only log read without an offset answers about a previous run

A precondition checked whether the worker had loaded its cut-point table by
looking in the container log for "no cut-point record for this candidate". It
was written twice and wrong both times, in **opposite directions**:

    v1  read the ABSENCE of the message as health
        -> a freshly recreated container has an EMPTY log, so it passes on
           precisely the run it exists to protect (invariant 30)

    v2  read the WHOLE file and matched the message
        -> `.log.error` APPENDS on a retry, so it matched the PREVIOUS run's
           section and condemned a healthy build

The second is the one worth naming, because v1's shape is already recorded four
times in this file and v2's is not. **A log that grows by appending is a record
of every run, and a search over the whole file answers about the union of them.**
The check had no way to say *which* run it was reading, and the file it was
reading contained two.

**The fix is an offset, not a better pattern**: record the log's size before the
action, read only the bytes appended after it, and treat **no growth as
cannot-tell rather than clean**. A pattern refined until it stops matching the
old section is still reading the union; it has just got luckier.

**Generalisation.** Any accumulating artefact — a log, an appended JSONL, a
state file that keeps history — needs a *position* as well as a query. Ask "what
did this run write", never "does this file contain". And where the position
cannot be established, the honest answer is that the check could not run:
**cannot-tell is not permission**, which is the same rule the third version of
this guard finally implemented.

The error cost one file and stopped a healthy run rather than passing a broken
one. That is the direction to be wrong in, and it is worth choosing deliberately
rather than by luck.

## 53. Judge a similarity score against a measured null, not against an assumed floor

The different-recording criterion nearly became an artefact of its own
procedure. Fidelity rose **monotonically as the cut length shortened** —
err_349 read 0.587 at 600 s and 0.853 at 30 s — and the obvious reading is that
short cuts inflate the statistic, which would have made every multi-scale
verdict a function of scale rather than of content.

**It was measured instead of assumed.** Four unrelated programme pairs were put
through the identical five-scale protocol:

    unrelated content, same protocol :  0.5954  0.6225  0.6257  0.6300
    the measured null ceiling        :  0.630
    the same-recording band edge     :  0.847

So the rise is signal, not inflation, and the bar now sits above a **measured**
floor rather than an assumed one.

**And the null turned the finding from an absence into a signature.** The 15
qualifying files read **0.6146 to 0.8046 — above the null**. They carry more
similarity than unrelated content while falling short of the same-recording
band, which is what a different dub of the *same programme* should look like:
shared timing and structure, different performance. Without the null those
numbers are merely "low"; with it they are "low in a specific, meaningful way".

**The rule: before a threshold separates anything, establish what the statistic
reads on inputs known to have no relationship, under the exact protocol you are
using.** A score has no meaning without its null, and a protocol that changes —
different cut lengths, different aggregation — needs its null re-measured, not
inherited. This is the constructive form of §39's empty band: there the gap was
found by calibration, here the floor was.

**Corollary.** A statistic that varies systematically with a parameter of your
procedure is the strongest reason to suspect your procedure. The right response
is to measure the parameter's effect on known-unrelated inputs — if the effect
persists there, it is the instrument; if it does not, it is the data.

## 54. The positive control refuted the interpretation, and that is what it was for

§53 established a **null** — what the statistic reads on unrelated content, 0.595
to 0.630 — and the 15 candidate files read 0.6146 to 0.8046. That was described
as *"what a different dub of the same programme should look like: shared timing
and structure, different performance"*, with a plausible mechanism attached.

**It was an inference about a region nobody had measured.** Measured, it is
wrong.

The control: masters carrying two tracks of one language, **19 of them labelled
in metadata as different regional dubs** — "Latin American" against "European"
Spanish — so same programme, same language, two genuinely different
performances, labelled by whoever built the file. Through the identical
five-scale protocol:

    known different dubs :  0.8334 ... 0.9922   (18 of 19 at or above 0.8964)
    the 15 candidates    :  0.6146 ... 0.8046
    unrelated content    :  0.5954 ... 0.6300

**A known different dub reads HIGH — mostly inside the same-recording band.**
The mechanism is obvious afterwards: Chromaprint fingerprints the whole mix, and
the mix is dominated by the shared music-and-effects bed. Two regional dubs
share their M&E entirely and differ only in dialogue, so the statistic barely
moves. **The instrument is nearly blind to the thing it was being asked to
detect.**

So the 15 do not read like different dubs. They read **below every labelled dub
pair in the corpus and only a little above unrelated content** — which points
away from "a different recording of this episode" and toward **the candidate not
being this episode at all**: a cataloguing defect, a file paired against the
wrong master.

**Two rules:**

1. **"Between two known bands" is not a measurement.** A null bounds one side; a
   claim about the middle needs a positive control for the specific thing being
   claimed. §53 got half of this and this entry is the other half.
2. **A statistic can be insensitive to the exact distinction you are using it
   for, while behaving perfectly on every control you thought to run.** Self-
   controls at 1.0000 and an unrelated null at 0.63 both looked like the
   instrument working. Neither could have revealed that dialogue is a small
   fraction of what it measures.

**And the control's own first version was worse than no control.** err_349 had
been disqualified because its master's three `jpn` tracks "disagreed" at 0.941 /
0.899 / 0.862. Track 4 is titled `[ Audio Guide ]` with the `visual_impaired`
disposition: **the control was comparing a performance against a narration of
it** and concluding the master was defective. The two real `jpn` tracks read
0.9652 — one recording at two channel counts. Commentary and descriptive tracks
must be excluded from any same-language comparison, and a control that removes a
file from scrutiny for a reason that is not true is worse than none.

## 55. A control firing tells you the measurement is not well-posed; it does not tell you why

err_349's master carries three `jpn` tracks. A control compared them against each
other, they disagreed at 0.941 / 0.899 / 0.862, and the file was disqualified on
the reading *"the master is defective — there is no single master recording for
the candidate to differ from."*

The control was right to fire and the conclusion was wrong. **Track 4 is titled
`[ Audio Guide ]` with the `visual_impaired` disposition**: the comparison was a
performance against a *narration of* it. The two real `jpn` tracks read 0.9652 —
one recording at two channel counts. The master was fine.

**The firing was treated as the diagnosis.** A control that fails has told you
exactly one thing: **the measurement it guards is not well-posed.** The reasons
are many and it distinguishes none of them — the input is unfingerprintable, the
wrong pair was compared, a track is a different *kind* of thing, the reference
itself is broken. Each needs its own evidence, and the most convenient of them
is not the answer merely because it was the one you were looking for.

**So: on a control failure, report the file as `says nothing` and investigate
separately — never fold the firing into the finding you were testing.** Here the
convenient reading disqualified a file, and a disqualification is a result: it
removed err_349 from scrutiny for a reason that was not true, which is worse
than leaving it in.

**The cheap discriminator was in the metadata all along.** Track title and
disposition — `visual_impaired`, `comment`, `descriptions` — distinguish a
different *kind* of track from a different *recording*, before any audio is
decoded. Any same-language comparison must exclude them, and a control that does
not is not a weaker control, it is one that manufactures failures.

## 56. A search margin must come from the geometry of the comparison, never from a constant

Five contributed tracks were probed against their candidate and every one came
back *"no window reached the peak floor"* — which reads as **"unverifiable,
confirmed"**, agreeing with the verdict already on the file.

The margin was **±12 s**. The true offset is **29.9 s**. Widened to ±90 s, all
five return at peak 0.98+.

**The margin could not possibly have reached**, and the geometry said so before
any audio was decoded: the candidate is **41.8 s longer** than the output, so an
offset of tens of seconds was the expected case, not a surprise. A constant
copied from a context where alignments are nearly right cannot survive a context
where they are not.

**Third instance today, second in this instrument.** The probe band that made
212 of 296 change points "unlocated" (§36) is the same defect — a fixed
`DISCRIMINATE_BAND_S` against lag separations it could not span. In every case
the constant was reasonable where it was written and silently wrong where it was
reused.

**So: derive the search window from a measurable property of the two things
being compared** — their duration difference, their known lag range, the span of
whatever profile is already recorded — and **fail loudly when the window cannot
cover the plausible range**, rather than returning "nothing found".

**And note which way the failure fell.** It agreed with the verdict already on
the file. A margin too small returns exactly what "genuinely unverifiable"
returns, so the wrong answer arrived wearing a confirmation — §48 again, and the
most dangerous shape it takes: **a result that is wrong and unsurprising.**

## 57. A contributed track has no master reference, but it does have the candidate

15 outputs are marked `unverifiable` while carrying audio that **can** be
checked — just not against the master.

The class: an output carries languages the master does not have, because the
merge contributed them from the candidate. Verification against the master is
genuinely impossible for those tracks — no same-language reference exists and
§8 forbids inventing one cross-language — so `MERGED_UNVERIFIED` is honest.

**But those tracks came from the candidate and were re-timed onto the master's
timeline, and the candidate is still on disk.** Comparing output against
*candidate* asks a different and answerable question. On err_117:

    eng  median 29907.96 ms   spread 7928.21   peak 0.983
    ger  median 29907.93 ms   spread 7928.56   peak 0.989
    ita  median 29907.48 ms   spread 7928.19   peak 0.985
    por  median 29907.90 ms   spread 7928.59   peak 0.987
    spa  median 29907.81 ms   spread 7928.21   peak 0.986

**Five tracks agreeing to about half a millisecond** is what applying one plan to
every track looks like; **peaks of 0.98+** say the content is the same recording,
so the candidate *is* that episode; and a **7.9 s spread across windows** is what
a structural re-timing should produce rather than a constant shift.

**What it does not establish**: whether the plan put them in the right *place*.
That still has no reference. **But "one plan, applied consistently, to the right
content" is a great deal more than `unverifiable` conveys**, and it is the
difference between *unverified* and *unverifiable*.

    outputs probed                                    77
      every output language present in the master     44
      output carries languages the master LACKS       33
        of those, currently marked `unverifiable`     15

**Before recording a file as unverifiable, ask what reference DOES exist.** The
absent one is not the only one.

**Extension: a RELATIVE question needs no absolute reference — any track is a
pivot.**

The rule above was applied one level further by a second agent, and it dissolved
a bucket. A within-file sweep excluded files with no bit-identical copied track
as `OUTSIDE REACH`, on the reasoning that without a copy there is nothing to
measure against.

But a copied reference answers *"how wrong was the delay"* — an **absolute**
question. The question actually being asked was whether tracks that **must**
share one delay agree **with each other**, which is **relative** and needs only
their positions relative to one another. **Any track serves as pivot.** The
err_117 result is exactly that shape arrived at independently: five contributed
tracks agreeing to half a millisecond at peaks above 0.98, with **no master
reference involved at all**.

So the excluded files get a processed-track pivot and a relative-only verdict,
**stamped `reference_is_copy: false`** so nobody reads a relative result as an
absolute one.

**And the bucket itself was correct — the conclusion drawn from it was not.** On
err_60, MD5 computed cleanly on every stream, 2 of 2 on the master and 3 of 3 on
the output, and none matched: the merge genuinely re-encoded every track. **A
copy really was absent.** What did not follow was that the question was
unanswerable. That is §55 one level up: *the bucket firing told you a reference
was absent, not that the question could not be asked.*

**So when a reference is missing, separate the questions it would have answered.**
Some need an absolute anchor and are genuinely blocked; others are relative and
are not. Collapsing them loses the second kind silently — the whole population
disappears into a bucket that sounds like a measurement and is a classification.

**Third variant: a log located by "newest in the directory", where the directory
is shared.**

A classifier found a file's log as *"the newest `.log.error` in the output
folder"*. That folder is per-**season** and holds one log per episode — **43 of
them** in the case that exposed it. During a run the newest is whatever was
processed most recently, so **every file after the first was classified from a
different episode's log.**

The tell was a file that read one way and then another: err_213 classified as
*"chimeric rejected"* while it was the only file in that folder, then *"not
piecewise"* once later files wrote theirs. **Same file, same build, two answers.**

The log is named after the candidate — `<candidate basename>.log.error`. **A
lookup, not a search.** Where an identifier exists, searching for the artefact by
recency is choosing to be wrong whenever anything else is writing.

So the family is now three, all producing confident wrong answers from a log:

    absence of a message      read as health          (empty log, fresh container)
    the whole appending file  read as the latest run  (retry appended below)
    the newest in a directory read as this file's     (43 episodes share it)

**Every one of them returned something plausible rather than failing.** When an
artefact is shared — by time, by append, or by directory — the reader must supply
the coordinate that makes its query specific: an offset, an identifier, a name.
**A query with no coordinate answers about whatever happened to be there.**

**Boundary on the extension: the pivot must be a valid comparison in the first
place.**

The Lead relayed "any track serves as pivot" to a second agent as though it
generalised. It does not, and the agent receiving it worked out why before
building anything.

The extension holds when the tracks being compared **can legitimately be
correlated with each other**. In the within-file sweep they can: several tracks
of one output, and the question is whether they sit in the same place relative to
one another.

It fails for contributed-track verification. There each track is measured against
**its own same-language counterpart in the candidate** — output `ger` against
candidate `ger`. Correlating output `ger` against output `spa` to obtain a
relative reading is a **cross-language correlation, forbidden by §8 at this
scale**. So when the candidate reference fails for all tracks of a file, there is
no relative fallback: **every available pivot is in a different language.**

    reference works for some tracks : the relative statement already exists --
                                      "within N ms of each other", no external
                                      anchor needed
    reference fails for all         : report `could not be measured` with its
                                      reason; do not invent a pivot

**And the extension sharpened the surviving claim rather than only limiting it.**
Because the relative reading does not depend on the master, on the multi-track
files the agreement figure is a real statement about the merge **even if every
absolute offset is wrong**. On the single-track files no such statement exists —
ten of sixteen have nothing relative to say. That split belongs at the top of the
report, not left for the reader to infer.

**The general rule: an insight that dissolves one bucket is not thereby
transferable.** Check that its precondition holds in the new setting before
carrying it across — a relaxation applied where its constraint is real produces a
measurement that looks like the original and answers nothing.

## 58. Every sweep carries a file whose answer you already know, or it cannot be trusted anywhere

A 27-file sweep completed and was **not reported**, because it returned
`APPLICATION` for err_214 — the one file whose answer was already known from
careful manual work, where the same tracks agree to **2.6 ms** and the verdict is
`MEASUREMENT`.

    sweep   inherited group of 4 scatters by 158.7 ms  -> APPLICATION
    manual  the same tracks agree to 2.6 ms            -> MEASUREMENT

**The instrument was more wrong than the thing it was measuring**: a 158.7 ms
artefact on a file whose real spread is 2.6 ms. And **err_214 was the only file
with an independent answer** — the other 26 would have been reported with both
defects invisible.

**Defect 1: the median of two points is their mean.** Sampling at two timestamps
and taking the median gives a single wild reading half its own error, with
nothing to reject it. The manual run had *seen* such readings — `-64.3, -132.9,
-66.4` at three points, where the middle one can be discarded. With two it would
have returned −98.6 and been believed. On err_214 one negative outlier among six
tracks produced the entire "scatter". Fixed with five sample points, a minimum of
three surviving readings, and a higher peak floor.

**Defect 2: results keyed by language, where a file carries two tracks of one
language.** The second silently overwrote the first and the survivor was
arbitrary. **This is §6, and the same agent had fixed the identical defect in a
different tool a few hours earlier**, with a commit explaining why. As they put
it: *knowing a failure mode is not the same as recognising it in code you are
writing* — the earlier one looked like a lookup table and this one looked like a
results dict, and they are the same mistake. Now keyed by `(language, stream
index)`.

**The rule: a batch run must include at least one input whose correct answer is
known independently, and must be void if it gets that one wrong.** Without it a
sweep reports its defects as findings across every file at once. And the control
must be **held to**, not adjusted — thresholds tuned until the control agrees
convert it from a check into a decoration.

**Corollary on scope.** The re-run's honest headline may be that the instrument
can answer roughly **one band file in fifteen** — the rest have too few tracks to
form a group at all. **A structural limit stated plainly is a result**; the same
number reached by not mentioning the denominator is not.

## 59. Changing the geometry of a comparison changes what the statistic means

Twelve of fifteen anomalous files turned out to be a consistent **~54 s shorter**
than their master — 52.2 to 59.4 s across eleven consecutive episodes of one
folder. Not fifteen independent anomalies: **one release that is systematically
about a minute short.** Cuts taken at the same *fraction* of each file therefore
displace the corresponding content by roughly 13, 27 and 40 s at the sampling
points, which predicts exactly the signature that had been read as evidence —
fidelity rising as the cut shortens.

**The obvious fix fails calibration, and it fails in the direction that looks
like confirmation.** Searching a short candidate window inside a long master
window gives tread fit and reach at once, in principle. In practice `correlate`
scores agreement **over the compared span**, so a 30 s match inside a 300 s
window is diluted to noise:

    a file against ITSELF, 30 s inside 300 s :  0.6124
    unrelated content                        :  0.5957

**Uncalibrated it would have condemned everything it touched.** The fix anyone
would reach for turns every input into a different-recording verdict, and every
verdict would have agreed with the hypothesis under test.

**What works: keep both spans equal so the statistic keeps its calibrated
meaning, and move the MASTER'S START instead.** Tread fit comes from the short
cut, reach from the sweep, and the two stop competing. Recalibrated: **1.0000**
for a file against itself at delta 0, **0.6061** for unrelated content.

**So: any change to what is being compared — span, alignment, aggregation —
invalidates the bands until they are re-measured.** A statistic is calibrated
against a *procedure*, not against a name. §53 says judge against a measured
null; this says the null belongs to the exact geometry that produced it.

**Two supporting corrections from the same work.**

**A refutation is no more transferable than an insight.** The tread hypothesis
had been excluded on another agent's refutation — but that was err_349, a
different folder with different geometry. It did not transfer, and carrying it
across suppressed the right explanation for twelve files. Same shape as the
pivot-extension boundary above: **check the precondition in the new setting,
whether what you are carrying is a finding or a dismissal.**

**Five agreeing points on a coarse grid are one point.** A consistent −20 s
offset looked real across five samples at 10 s steps; refined to 2 s steps the
deltas scatter (−18, +24, +26, −16, −20), which is not a displacement. **Grid
spacing sets the resolution of "consistent"**, and agreement at a spacing coarser
than the effect is agreement about the grid.

## 60. Clear the code that will CHANGE, not the code you were reading

A tag named `VMSAM_TMP` was found shipping onto published outputs — internal
bookkeeping with a name that says temporary. The Lead checked closure membership
before authorising a rename or a strip, found the function outside the protected
set, and cleared it.

**The function checked was the READER.** `check_delay_retention_sub`
(`mergeVideo.py:2754`) consumes the tag and is genuinely outside the closure.
The tag is **written** at `:2956` and `:2973`, inside `generate_new_file`
(2876–2988) — **a protected closure member.** Renaming the key or stripping it
at write time both land in protected code, and the clearance did not cover
either.

The agent receiving the instruction checked the writer, declined, and said so:
*a clearance based on the reader does not clear the writer.* **The standing rule
is the owner's and not the Lead's to relax**, including by mistake.

**So: closure membership is a property of the lines that will be edited.** Before
clearing any change, locate the exact write site — not the field, not the
feature, not the function you were reading when you noticed it. A field is
touched by several functions and they do not share a clearance.

**And quote the tool's own number.** The Lead reported the closure as **79**
members, from counting the lines of `--list`; the authoritative line reads
**78 members reachable from generate_launch_merge_command**. One line of that
listing is not a member. A count derived by post-processing a tool's output is a
second implementation of that tool, and it disagreed on the first use.

**Outcome recorded rather than fixed**: `VMSAM_TMP` stays. It carries two
integers of real provenance and harms nothing; the cost is the name, and the
name is now written down here. Changing it needs an owner-authorised closure
edit, and a cosmetic rename is not worth spending one on.

## 61. Sample size is set by the measured outlier rate, not by convention

Three runs of one measurement gave three answers, and the instrument was not the
problem:

    manual v0   3 points  ->   2.6 ms    the right answer, drawn on luck
    sweep v1    2 points  -> 158.7 ms    (plus two real, separate defects)
    sweep v2    5 points  ->  33.0 ms    defects fixed, still contaminated
    control    14 points  ->   3.4 ms    outliers identifiable and rejected

The raw readings show why. Every track has a stable core and roughly one wild
excursion in three:

    tha:9   +65, -4, -2, +1, +64, -40, -254, +26, +66, +64, +64, +66, +64, +57
    spa:4   -186, -57, -174, +63, -44, +65, +65, +62, +63, +64, +65, -97

**30 of 95 readings are outliers — 32%.** Cross-language envelope correlation on
this material simply does that.

**Under a 32% outlier rate a median over 5 points is a lottery and a median over
3 is a worse one.** Neither has enough inliers to be reliably a majority.
Fourteen does, and it recovers the same answer the three-point run happened to
get. **The fix was never a better statistic — it was enough samples for the noise
that was actually there**, and the noise had to be measured before the sample
size could be chosen.

**The uncomfortable half: the control was right and unearned.** A three-point
median that drew three inliers was used to void two better-sampled runs, and its
authority came from having been produced carefully rather than from having enough
data. **A control is only a control if it is better-sampled than the thing it
judges** — otherwise "the sweep disagrees with the control" and "the control
disagrees with the sweep" are the same sentence, and the label decides which one
wins.

So §58 needs this rider: **carry a known answer in every sweep, and make sure the
known answer is known more strongly than the sweep can measure it.**

**And a sound method can still be unusable on a corpus.** Of 27 files, 26 fail
for want of tracks rather than sampling — no number of points fixes a file with
one correlatable track. The instrument answers roughly **one file in
twenty-seven**, and re-running it at 14 points would cost seven times as long to
move a single file from a wrong verdict to a right one. **Stating that limit is
the result; paying to demonstrate it again is not.**

## 62. When the two errors are not symmetric, encode the reversible one

A rule said: remove candidate audio whose interpretation matches the master's.
Implementing it needs a test separating *duplicate* from *genuinely different
dub*. Measured on the only labelled samples this corpus has:

    labelled different regional dubs  n=19   0.833 - 0.9922
    format variants / duplicates      n=24   0.927 - 0.9894

**The distributions overlap and the dub range reaches HIGHER than the duplicate
range.** A separating threshold needs `dub_max < dup_min`; the real numbers are
0.9922 and 0.9270, the wrong way round. **This is not a bar needing
recalibration — the statistic does not carry the information.** Two regional
dubs share their entire music-and-effects bed and differ only in dialogue, a
small fraction of what the fingerprint measures (§54).

At the shipped 0.90 gate, **18 of 19 known different dubs clear the bar**, so
removal on that test deletes a genuinely distinct dub about **95%** of the time.

**The agent built the opposite of the instruction and said so.** A track reading
as matching is kept unmodified and marked redundant. The reason is not caution
in general, it is that **the two errors are not symmetric: a duplicate that
ships is recoverable, a deleted dub is not.** Where one direction is
irreversible, the irreversible one needs evidence the reversible one does not,
and a test that cannot bear that weight must not be the one that decides it.

**The resolution was to move the gate, not the threshold.** Removal is now
gated on **metadata** — region titles like "Latin American" / "European
Spanish", which separate the classes perfectly because they encode exactly what
the audio cannot. Three states, and the third is the important one:

    metadata identifies a DUPLICATE  -> remove
    metadata identifies a DISTINCT   -> keep, unmodified
    metadata is SILENT               -> keep, marked redundant

**Silence must not collapse into removal.** An absent region marker identifies a
file whose author did not label it, not a duplicate — reading it otherwise is
§30 with an irreversible consequence.

**And the limit is recorded as unbuilt, not impossible.** Isolating dialogue
from the shared bed — centre-channel extraction, source separation — is what
*would* separate a dub from a duplicate. Nothing here does it. *"We cannot tell
a dub from a duplicate"* is a permanent limit only until someone measures the
right thing, and the distinction between **unmeasured** and **unmeasurable**
belongs in the record.

## 63. Exclusion narrows; identification decides

Ten innocent explanations were closed by measurement for a set of anomalous
files — reach, tread structure, candidate unfingerprintable, master
unfingerprintable, master heterogeneous, cross-tag mis-tagging, descriptive
candidate track, misalignment, piecewise editing, mislabelled episode. That is
a strong argument and it settled nothing about any individual file.

**One positive match settled two of them outright:**

    err_123  episode 12   vs its own master   eng 0.6242  jpn 0.6067
                          vs err_121 (ep 11)  eng 0.9436  jpn 0.9568

    err_196  episode 11   vs its own master   jpn 0.5674
                          vs err_178 (ep 7)   jpn 0.9556

**err_123 carries episode 11's audio filed against episode 12.** Not "does not
match its master" — *matches a specific other episode*. And on err_123 **two
independent languages agree**, jpn at 0.9568 and eng at 0.9436, both pointing at
the same wrong episode. A single spurious match cannot do that.

**The same search produced both answers, which is what makes either credible.**
Where a file is misfiled the pattern inverts hard — 0.61 against its own master
against 0.96 against the true one — and where it is not, the file's own master
remains its best match by a clear margin (err_46: 0.7074 against its master,
below the null against all 40 siblings). **A test that can come out both ways,
and did, on one run.**

**So: when exclusion has narrowed a set and the residue is still unexplained,
stop adding exclusions and go looking for what the thing IS.** Ten closed
explanations describe what a file is not; one match names it. The second is
cheap here — every candidate master is already on disk — and it converts a
terminal state into a repair.

**And the outcome is better than the closure it replaces.** Two files leave the
unmergeable queue entirely as cataloguing defects rather than media ones. A
dispatch that ends in repairs beats one that ends in correct closures.

**Corollary for counting.** The surviving 13 are **four situations, not thirteen
findings** — eleven are consecutive episodes of one folder and almost certainly
one upstream cause. §33 again: state the group count, and let the owner decide
about one release rather than about eleven files.

## 64. A pipeline that checks A against B is blind to any error common to both

Two files were identified as carrying another episode's audio, and the cause was
determined rather than guessed:

    err_123  candidate filename S01E12  -> paired against ep 12's master
                                        -> audio is episode 11
    err_196  candidate filename S03E11  -> paired against ep 11's master
                                        -> audio is episode 7

**The pipeline paired correctly on the declaration it was given.** In both cases
the pairing follows the candidate's own filename, and the audio inside does not
match it. The release names its content wrongly; no re-pairing in this codebase
fixes that. Container tags carry no episode identifier on either candidate, so
**the filename is the only declaration the file makes** — anyone re-deriving this
from metadata rather than audio will find nothing.

**The identification needed its own control and got one.** If adjacent episodes
resembled each other, "matches the wrong master" would prove nothing:

    master ep11 vs master ep12   0.6828   different_recording
    master ep7  vs master ep11   0.6875   different_recording

So a candidate reading 0.95 against one and 0.61 against the other cannot be
explained by the two episodes being alike. **That control could have come back
the other way and would have collapsed the finding.**

**And here is the part with reach beyond these two files.** This pipeline
verifies a candidate against a master. It therefore detects a mis-numbered
*candidate* — that is why these two surfaced as errors at all. **It cannot
detect a mis-numbering shared by both.** If a release is shifted by one and the
master library is shifted the same way, candidate and master agree, the merge
succeeds, verification passes, and the output is correctly synchronised
content filed under the wrong episode. **Nothing in this system would ever
report it**, because every check it performs is a comparison between the two
things that are wrong together.

A supporting observation, flagged and explicitly not claimed: err_121 is filed as
episode 11 and matches **neither** its own master nor episode 11's (0.6783 /
0.7017), which is consistent with a whole release being shifted — untestable
here, because the relevant master never entered the error tree.

**The general rule: enumerate what your verification compares, and accept that
any defect shared by both sides is invisible to it by construction.** Detecting
that class needs a reference outside the comparison — an external identifier, a
duration table, a third source. Silence from a differential check is evidence
about the difference and about nothing else.

## 65. When the fallback governs most cases, the fallback IS the rule

A metadata gate was built to decide whether candidate audio duplicating the
master should be removed. Measured across the corpus rather than inherited from
the labelled sample — **461 files, 150 language groups with two or more
tracks**:

    no track titled          53
    only some titled         45     -> 98 of 150, 65%, UNCLASSIFIABLE
    every track titled       52
      region marker present  43
      format marker present  13

**The unclassifiable case is the common one.** "Metadata is silent" was written
as the fallback of a three-way rule and it governs **about two thirds of every
decision the rule will ever make.** A fallback at that frequency is not an edge
case being handled gracefully; it is the behaviour of the system, and it
deserves the scrutiny the named branches got.

Region markers outnumbering format markers three to one also corroborates the
locale discriminator on a sample three times the size of the one it came from —
worth doing even when a labelled sample already exists.

**And the fallback as first written was wrong, which the frequency made
expensive.** The rule said *silent → keep unmodified, mark redundant*. That
conflated two independent axes:

    axis 1  remove or keep      -- what the duplicate question is about
    axis 2  stretch or leave    -- what the speed relation is about

Under a speed relation an **unstretched track is desynchronised by the whole
relation** — 0.1% on the NTSC files, but 4.27% on a PAL pair, which is seconds
of drift. So "keep unmodified" ships a track that is **certainly** wrong.
Stretching it is **correct** if the track is a distinct dub and **redundant but
in sync** if it is a duplicate.

**Between "possibly redundant and correct" and "certainly desynchronised", only
the first is defensible**, and neither risks the deletion the ruling exists to
prevent. The fallback is now *keep and stretch*.

**The general form: when you write a three-way rule, measure how often each
branch fires before deciding how much care each deserves** — and check that the
fallback answers every axis the named branches do. A default inherited from one
question will be silently wrong about the others.

## 66. A unit test cannot tell you that nothing calls the unit

A speed-relation policy was built, tested and landed: classification from the
measured factor, method selection by relation kind, per-track drop decisions, a
verified output tag. **1458 tests green.** The first real file showed the
non-PAL branch is **unreachable**:

    detect_pal_relation reads the GROSS DURATION RATIO and returns None for
    anything not PAL-shaped, and prepare_pal_corrected_candidate returns
    immediately on None.

      the PAL gate      at char 1009
      the Case 3 branch at char 4029

So the non-PAL classification, the `rubberband` selection and the
drop-if-no-pitch-preserving-method branch **cannot run on a non-PAL file** —
the only class they exist for. Three files confirmed it by returning early.

**Every part a unit test could reach worked. The part only a real file can reach
did not.** The tests call the policy functions directly, so they establish that
the policy is correct and say nothing about whether anything reaches it. That is
the class an outside auditor found here once before — a function defined, tested
and never called — arriving this time in code written the same day.

**Reachability is a property of the call graph, not of the unit.** It cannot be
established by any test that supplies its own entry point. The cheap check is to
run one real input through the real entry point and confirm the new branch is
*taken*; a feature that has never executed on real data is a design.

**And the gate that blocks it is the defect the feature was built to fix.**
`detect_pal_relation` decides on the duration ratio alone — the same
standalone-metric fault recorded when the corroboration requirement was added,
which guards the *accept* direction while the *reject* direction still short-
circuits on a ratio. A feature placed behind that gate inherits it, and the
archived note had already said the fix is to **make the measurement the entry
point and the duration ratio a corroborator on both sides**.

**Ordering rule that falls out: when a new branch depends on an existing gate,
establish that the gate admits the new population BEFORE building behind it.**
Char 1009 versus char 4029 was checkable in a minute, at any point during the
build.

## 67. A run ordered by any property groups related files, so every trend inside it is suspect

A 119-file run was deliberately ordered **smallest first**, so a systematic
failure would surface in minutes rather than after the largest file. That
ordering was correct and it also manufactured a trend:

    first 51 files : repair-rejected 20   not-piecewise 25
    files 52-80    : repair-rejected 12   not-piecewise  2

The "recovery population" reported as **25 and growing** was not growing. And
grouping the buckets by series shows why:

    repair_declined_record_not_piecewise   27 files across 1 series-group
    repair_attempted_rejected              32 files across 4 series-groups

**All 27 are one show.** Episodes of a series share size, so ordering by size
clusters them — the early split looked balanced because one series happened to
sit at the front. **"25 files a rescan might recover" is one series that a
rescan might recover**: a single measurement repeated 27 times, and if the fresh
scan finds the same structure there, the recovery population is **zero**.

**Mirror image in the same run**: `merged` went from 6 of 51 to 20 of 80, which
reads as a rising success rate and is the same artefact — the later, larger
files are a different population. **Neither number should be extrapolated.**

**So: any ordering that is not random correlates with something.** Size
correlates with series, alphabetical order correlates with release group,
mtime correlates with whatever ran last. A progress trend inside such a run
measures the ordering unless the buckets are grouped and checked. **Report
partial results by group from the first partial report**, not only in the final
one — the trend is most persuasive and least reliable while the run is still
going.

**The negative half survived the regrouping and is stronger for it**: 32 files
across **4 independent** series-groups where repair ran and the chimeric build
self-rejected. Four groups agreeing is a result; twenty-seven files of one show
agreeing is one.

## 68. Look for your discriminator in the group you did NOT select

A shared-cause hypothesis was pre-registered before measuring, with the
properties to test named in advance. Three of the four discriminated **nothing**:

    codec / channels / sample rate:  aac 2ch 48000 -> flac 2ch 48000
    identical across the failures AND the nine files that measured fine

**Had those been examined first and the search stopped there, "they all share a
codec path" would have been the finding.** It is shared by the successes, and
the pre-registration's own clause — check the property in the group you did not
select — is what prevented it.

The fourth property did separate, with an empty band:

    FAILED     0.94367 .. 0.95931       duration ratio
    MEASURED   0.99575 .. 1.04585       empty band 0.036

**But it did not unify the two failing series**, which was the hypothesis:

    folder 172   0.95919 0.95931 0.95906 0.95906   inverse PAL 0.95904,
                                                   every one within 0.0003
    folder 103   0.94367 0.95229                   neither that factor nor
                                                   each other

**So "two series behaving the same way" is one series with a signature and one
without.** The prediction was half right, which is the outcome a
pre-registration exists to make visible rather than reconcilable.

**And the counterexample came from inside the success group.** `err_353`
measured **cleanly** at ratio 1.04585, close to the forward PAL factor 1.04271.
So a PAL-magnitude ratio does not by itself defeat envelope correlation — which
downgrades even the surviving finding from *"a speed relation causes this"* to
*"these four show a duration ratio consistent with one"*. **Container duration
is not the same claim as a track having been resampled.**

**Corollary, from the same run: the population can change while you measure it.**
One file's output was rewritten an hour before the run reached it and no longer
carries the contributed track at all — so its reading describes a file that no
longer exists, and the set is 15 rather than 16. **It was also the one ratio that
did not fit its series**, so removing it makes the match tighter: the direction
that should raise suspicion rather than satisfaction. Recorded rather than
quietly dropped.

## 69. Co-location is not co-causation, and the same file can be its own control

Two failures shared a folder and were treated as one phenomenon. A refutation
criterion was committed before looking — *"if the two sets are disjoint, that
kills it"* — and **they are disjoint.** Two different problems that happen to
share a location.

**Failure A is structural and fully explained.** The folder holds **two
different candidate releases**, and the split is perfect across all 47 records:

    CORRELATE  21   candidate carries fre + jpn, jpn in ac3    median peak 0.98
    FAIL       26   candidate carries jpn only,  jpn in flac   median peak 0.27

**47 of 47, no exceptions.** And §68 removed the tempting alternative for the
second time in one investigation — the duration ratio is **indistinguishable**
between the two groups (medians 0.94930 and 0.94906), so a ratio equally present
in the files that correlate perfectly explains nothing about the ones that fail.

**Failure B is the interesting one, and the same file supplies its own control:**

    err_160 / err_194, one file each:
      jpn  vs the MASTER      0.980 / 0.983 over 16-17 windows
      fre  vs the CANDIDATE   no correlation at any margin out to 240 s

**The candidate is demonstrably the right episode and its audio is demonstrably
intact** — the `jpn` track proves both. Yet the `fre` track the output carries
does not correspond to its only possible source. **One track fails while another
track of the same file succeeds**, which excludes every explanation about the
material and leaves the merge path.

**That is the strongest use of a within-file control this project has found.**
Everything prior was consistent with *"this series has bad audio"*; a passing
track in the same file makes that impossible to say.

**So: before attributing a failure to a location, a release, or a corpus, look
for something in the same file that succeeds.** If one exists, the explanation
must distinguish them, and most explanations about the material cannot.

**Two caveats recorded with the finding rather than after it.** The 47-record
split is measured from a **pre-fix, unreproducible** table — the peaks are
stored numbers and 113 of that table's 180 records have masters that can no
longer be resolved. The split rests on a structural property and is 47 of 47, so
re-measuring is unlikely to move it, but it is stored evidence and was labelled
as such. And each file was **re-probed at the moment of measurement** rather
than trusted from an earlier list, because a concurrent run had already removed
one file from the class mid-measurement.

**Correction to Failure B, entered by the agent that reported it — and it is
§29 again: read what the pipeline already recorded before measuring.**

The claim was *"the `fre` track does not correspond to its only possible
source"*. **The candidate is not its only source.** The merge log says so
plainly, in both files:

    err_160  stream 2 (fre): re-timed 4 pieces; patched 86.0 s from master
             (jpn, shared_reference)   candidate coverage 0.9436
    err_194  stream 2 (fre): re-timed 6 pieces; patched 72.1 s from master
             (jpn, shared_reference)   candidate coverage 0.9521

**The last 86 seconds of the French track is the master's Japanese audio** —
deliberately, under the `shared_reference` patch rule, with the dub gate
authorising the source at fidelity 0.9698 and 0.971 against a 0.90 bar. Both
files then verify: 4 and 6 checkpoints, duration inside 0.09% of a 0.5%
tolerance.

**What survives is exactly the part the entry was written for.** The within-file
control is untouched: `jpn` correlates with the master at 0.980/0.983 while
`fre` reaches nothing, so the candidate is the right episode with intact audio
and something happens to `fre` between candidate and output. **That reasoning
never depended on the withdrawn premise** — it needs only two tracks of one file
behaving differently, which they do.

**And the patch is not sufficient either, which the reporting agent said before
anyone could ask.** 86.0 s of 1523.6 is **5.6%**, and the sample windows fall
inside candidate-derived pieces. A 5% contaminated tail should not defeat
correlation at every window and every margin. **The patch is a fact; its
sufficiency as an explanation is not.**

**The finding that came out of the correction is larger than the one it
replaced, and it is not a defect.** A French audio track whose final minute and
a half is Japanese dialogue **passes verification**, because the checkpoints
judging it are same-language ones elsewhere in the file. That is
`shared_reference` doing exactly what it is specified to do. **Whether it should
is a product decision nobody has taken**, and the verifier cannot be blamed for
not answering a question it was never asked.

## 70. A negative result needs a controlled instrument as much as a positive one

The "different recording" class does not exist. **Twelve of the thirteen
survivors are the SAME recording running 4.271% fast** — duration ratios 1.0423
to 1.0436 against an exact 25/23.976 = 1.04271, every one within 0.08%. Correct
the speed and re-measure:

    err_46   0.6124 -> 0.9527      err_63   0.5648 -> 0.9459
    err_91   0.7619 -> 0.9583      err_101  0.5787 -> 0.9640

**Sixteen of sixteen tracks read `same_recording`.** The other three are
mis-numbered files. **No file in this corpus needs `unmergeable_verified`.**

**And it explains the residue that ten closed explanations had left.** A signal
running 4.271% fast does not correlate with its source at **any** offset — the
misalignment grows continuously — so no alignment sweep at any scale could have
rescued it, which is exactly the observed behaviour nobody could account for.
The reading sat **above** unrelated content because it *is* the same
performance, and **below** a different dub because a dub shares the music bed at
the **same speed**. The unnameable gap was a speed relation.

**THE NEAR-MISS IS THE INVARIANT.** The first speed-correction test used
`asetrate=44100*X` **with no leading `aresample`**, so the filter's base rate
was whatever the source happened to be rather than 44100. It returned **0.5757 —
no improvement** — and the agent was one message from reporting the PAL
hypothesis **refuted**, with a duration ratio matching to 0.04% left standing as
an unexplained coincidence.

**What saved it was refusing to accept a negative from an uncontrolled
instrument.** Take a file, apply a **known** 4.271% transfer, correct it back,
and require the original to return. With the fixed chain that control reads
**0.9994**; a PAL pair reads 0.5759 uncorrected — precisely the band the files
occupied. **The bug was in the correction, not in the hypothesis, and only the
control could tell those apart.**

§58 required a known answer in every sweep. This extends it to the case that
feels least like it needs one: **"the hypothesis is refuted" is a measurement,
and a broken instrument produces it as readily as a true one.** A negative from
an uncontrolled tool is not evidence against a hypothesis; it is no evidence at
all — and it is more dangerous than a false positive, because a refutation ends
the investigation rather than inviting scrutiny.

**Corollary: an unexplained coincidence is a live objection.** A duration ratio
matching PAL to 0.04% does not stop being evidence because one test came back
negative. When a refutation leaves a coincidence standing, suspect the
refutation.

**And the prediction was pre-registered by a different agent** — that a speed
relation would be the discriminator, and that *"a time-scaled envelope does not
correlate with its source at any margin"* — committed before any of this data
existed, so it could not be fitted afterwards.

## 71. Verify the artefact that ships, not the one on the way to it

`verify_chimera` runs against the `*.chimeric.mkv` **intermediate**. The final
merge happens afterwards. **Nothing checks the file that ships.**

    merge log, err_160   verify stream 1 piece[619.6..1437.9]s @mid 1028.7s
                         residual -20.3 ms, peak 0.999
    published file       fre 00:01:48.010

**A 0.999 correlation at 1028 seconds, on a track that is 108 seconds long in
the published file.** The verifier is not wrong — it answered correctly about a
different file, and every reader took its answer as being about the output.

**Five of 94 published outputs carry truncated or empty audio:**

    err_45   1459.7 s   eng 137.8
    err_160  1523.6 s   fre 108.0, jpn 108.0
    err_174  1441.9 s   eng 0.0                    <- an EMPTY track
    err_194  1506.1 s   fre 171.2
    err_211  1442.0 s   six tracks ALL at 441.5    <- one instant, six tracks

Read from the container's own DURATION tags — no decode, no seek — and confirmed
by sequential decode.

**The consequence is larger than five files: every Step-2 pass in this project
is a statement about an intermediate.** It does not make those passes false —
the intermediate really was in sync — but it makes them **silent about the
artefact**, and a completion count built on them means something different from
what it says.

**So: a verification step must name the file it opened, and the last thing a
pipeline does to an artefact must be re-checked after it is done.** Any step
after the verifier is unverified by construction, however faithful the verifier
is. This is §22 sharpened — availability is a property of the artefact — and the
same shape as §32: the check ran correctly, on an object nobody meant it to run
on.

**It also invalidated two live hypotheses at once, and neither was wrong about
the pipeline.** A `shared_reference` cross-language patch and the `--sync -1:`
delay collapse were both offered to explain why a track failed to correlate.
**There was nothing to correlate.** Both were about the wrong object. **The
pre-registered test is what exposed it, by failing in a way neither hypothesis
predicted** — which is precisely what pre-registration is for.

## 72. A crashed run recorded as a success, because nothing asked whether the output was complete

**Two mechanisms produce the five truncated outputs, and neither is the story a
single explanation would have told.**

**Group B — the merge RAISED and the ledger says `MERGED_IN_SYNC`.** err_174 and
err_211 never entered the chimeric path; their logs hold a traceback and nothing
else — `mergeVideo.py:946`, *"Only dict_keys([<one file>]) file left. This is
useless to merge"*. What the ledger records:

    err_211  outcome MERGED_IN_SYNC   verdict in_sync   max_abs_lag_ms 0.0
             samples  60s 0.999 | 480s 1.0 | 900s 0.99
             published: six tracks end at 441.5 s of 1442.0 s
    err_174  outcome MERGED_IN_SYNC   published: eng track 0.0 s

**Two of the three checkpoints fall past the end of the affected tracks and
still report peaks of 1.0.** Not a broken verifier: `sync_verdict` compares two
tracks **within** the output and happened to pick a pair that survived.
**Nothing asks whether the other tracks contain anything at all.**

The chain: **merge crashes → partial file left on disk → verifier samples
whichever tracks are intact → ledger records `MERGED_IN_SYNC`.** A zero-length
audio track passes verification because no check asks the question.

**Group A — the chimeric path**, where verification measured the intermediate
rather than the artefact (§71).

**The check that catches both is free and asks a different kind of question**:
compare each audio stream's `DURATION` tag against the container duration.
Metadata only, one `ffprobe`, no decode. It found all five **without knowing
either mechanism existed**, because it asks about the **artefact** rather than
about the process that produced it.

**That is the general rule.** Every check in this pipeline had been a statement
about a *comparison* — this track against that one, this build against that
bar. **None was a statement about completeness**, and a comparison between two
surviving tracks is silent about a third that is empty. **Before verifying that
an output is correct, verify that it exists in full.**

**And an exception must reach the ledger.** A run that raised and a run that
succeeded must not produce the same `outcome` value, whatever the file on disk
looks like afterwards — the row is the only record anyone reads later.

## 73. A justification names a property; check that the branch keys on THAT property

The speed policy chose its correction method by **ratio class**:

    PAL      -> asetrate     non-PAL -> rubberband

and justified `asetrate` for PAL by *"the original speedup raised pitch too, so
undoing it that way restores it"* — **a statement about the source transform.**
The branch then keyed on **whether the ratio is 25/24**, which is a *proxy* for
that property, not the property.

**The proxy is correct for the case that motivated it and wrong for the other:**

    PAL speedup    24 fps film played at 25 fps
                   audio 4.271% FAST -> pitch +70.7 cents
                   asetrate slows it back -> pitch restored.  CORRECT

    NTSC pulldown  24 fps film played at 23.976 fps
                   audio 0.100% SLOW -> pitch -1.7 cents
                   asetrate speeds it back -> pitch restored.  ALSO CORRECT

**Both are naive speed changes at source.** By the rule's own stated
justification `asetrate` is the faithful inverse for both, and `rubberband`
**preserves a pitch shift that ought to be undone**. The non-PAL branch does the
wrong thing to exactly the class it was built for.

**And the earlier objection to this rule was also on the wrong axis.** It had
been argued that the rule was wrong for rejecting 1.7-cent shifts while
permitting 72-cent ones — reasoning about **magnitude**. Magnitude is not the
property either. **The right question is whether the source transform shifted
pitch**, and for every film-to-video pulldown in this corpus it did. A
pitch-preserving method is correct only where the relation came from something
that did *not* move the pitch — a deliberate time-stretch — **and this corpus
contains no such case**, which is why the non-PAL branch has never had an
example that justifies it.

**So: when a design note explains a choice by naming a property, read the
condition and check it tests that property.** A proxy that holds in the
motivating case will pass review, pass its tests, and be wrong everywhere else —
and it is invisible precisely because the justification beside it is correct.

Found by the owner, reading the design rather than the code.

## 74. The failed check that nearly deleted a running job

Memory pressure killed a VM, and 9.3 GB of session scratch was a candidate for
deletion. Before recommending it, an agent tested whether anything was still
writing there. **The check looped over `find` output unquoted. The filenames
contain spaces, the shell word-split them, and every iteration errored.** It
printed **"no live writer"** — and the scratchpad was one message from being
reported safe to delete.

**It was not safe.** Three independent signals said so once the check was fixed
to read `/proc/*/fd` with a null-delimited loop: a pid holding the run's log
open, an `ffmpeg` decoding from that directory, and a script written two minutes
earlier. **Deleting it would have destroyed an in-flight six-file decode plus the
measurement that settles which of two methods is correct** — roughly two hours
of work, and the answer to a question the owner had asked.

**This is the same defect this file records a dozen times — a check whose
"clean" and whose "could not run" are the same output — but it is the first
instance where acting on it would have destroyed something rather than merely
misled someone.** Every earlier case cost a wrong belief. This one was one step
from costing the work itself.

**Two rules, and the second is the one that saved it:**

**A check that authorises a destructive action must prove it ran.** Count what
it examined and say so — "0 processes hold a descriptor here, 47 descriptors
examined" is a result; "no live writer" is a sentence that an errored loop
produces just as readily as a true one.

**And ask the owner rather than inferring from the artefact.** The same agent
noted that two subdirectories *looked* finished by mtime and explicitly declined
to conclude it: **ask them; do not infer it from mtimes as I nearly did.** The
owner of a run knows what it is still using; a timestamp does not.

**Related and worth stating together**: the largest consumer of memory on this
box was **finished measurements nobody had deleted** — build artefacts whose
results were already reported, sitting in tmpfs. §37 records that a pass which
degrades its own machine is invisible to every check reading its output. **The
same blind spot covers the directory the pass writes into.** The tool checks
outputs; nothing checks the tools.

**Refinement, from the owner: tmpfs is right for some things and wrong for
others, and the distinction is not size.**

The entry above treats accumulation in `/tmp` as the defect. That is too crude.
`/tmp` is RAM, and RAM is correct for anything **cheap to reproduce and hot to
read repeatedly**:

    23 working copies of the repository   162 MB   CORRECT in tmpfs
      reproducible from git in seconds; read on every test run; the isolation
      is what lets several agents run suites at once without colliding

    finished merge outputs                8.3 GB   WRONG in tmpfs
      hours of decode to reproduce, and never read again once the measurement
      that produced them was reported

**So the test is not "how big" but "cheap to rebuild, and still being read".**
A repository checkout is both. A build artefact whose result is already written
up is neither — it is cold storage occupying the fastest medium on the box.

**And what must survive is a third category.** The *tools* the agents wrote —
scripts, not outputs — are cheap to lose and impossible to reproduce, so they
belong on real disk. They were copied to `/home/vmsam/src/vmsam_agent_scratch/`
after a VM was lost to memory pressure; the working copies were deliberately
**not** copied, because git already holds them.

    cheap to reproduce + hot     -> tmpfs
    expensive to reproduce       -> real disk
    cheap to reproduce + cold    -> delete it

## 75. Nothing reconciles the inbox against the ledger, so a file missed once is missed forever

Two ingest gaps were found on the same day, by the same question — *are the
things on disk the same as the things in the table?*

    errors table 315   private mapping 309   -> err_694-699, ~35 hours invisible
    inbox 316 candidates   errors table 315  -> one file, 118 days invisible

**Neither was a processing failure. Both were files that existed and were never
enrolled.** The six had arrived and nothing regenerated the mapping; the one had
been *processed* — its `.log.error` is 118 days old and records the merge failing
with `first_delay_test` readings of `{0, 1036144, -485708}` ms, nonsense
spanning seventeen minutes — and it still never became a row.

**Multiple candidates per episode are normal and handled**: 28 episodes carry
more than one candidate, 61 files in all, and **60 of the 61 are in the table**.
So this is not a rule about duplicates. It is a single miss that nothing has
revisited in four months, and a newer release of the *same episode* was enrolled
79 days later without anyone noticing the older one had not been.

**The defect is the absence of a reconcile, not either miss.** Every count in
this project is read from a ledger; nothing compares the ledger against the
world it describes. **A file that fails to be enrolled once is invisible
permanently**, because every subsequent tool starts from the ledger — which is
§30 at the level of a corpus rather than a query.

**The check is cheap and should be standing**: list the inbox, list the table,
list the mapping, and report the three differences. It takes seconds, it found
both gaps immediately, and it is the only thing that can catch this class —
because a missing row produces no error, no log, and no symptom anywhere else.

**A fix attempted and reverted, recorded because the reason generalises.**

`agent_idle.py` reports a session "stopped" when its pane is at rest. An agent
pointed out that this is wrong for the **detached-runner** design: a long job is
launched with `nohup`, the session goes quiet and wakes on notifications, and one
such run survived its own session, a broken bridge and a usage-limit reset while
being reported STOPPED every hour. **A false alarm repeated hourly trains the
reader to ignore the real ones.**

The attempted fix walked the tmux pane's process subtree to ask whether the
session still owned a live process. **It was wrong in both directions and only a
direct test against ground truth showed it:**

    matched   a `/bin/bash -c source .../shell-snapshots/...` wrapper, which
              every agent has after any tool call -- a false positive on all four
    missed    the actual job: run119.py has **ppid 1** and is not in any pane's
              subtree at all

**Detaching severs exactly the parentage the check was walking.** That is what
detaching *is*. So parentage — the rule this project adopted after a path
substring nearly killed another agent's decode — cannot identify a detached
process's owner, and no amount of care with the subtree walk fixes it.

The surviving signal is **cwd**: the runner's working directory is its owner's
session scratchpad. That is real but needs a session-to-scratchpad mapping the
tool does not have, so the change was **reverted rather than shipped half-right**.

**Two things worth keeping.** A rule that is correct in one context (identify by
parentage, not by name) can be *inapplicable* in another rather than merely
harder — and the failure mode is a check that looks like it works. And **a
detected-vs-actual test against a known answer is what caught it**: the branch
was never exercised by the agents' real state, so nothing but a direct call
against ground truth could have revealed it (§66).

## 76. The owner's question, settled: a PAL transfer is a naive speed change

`asetrate` against `rubberband`, **the same factor in both arms** — the exact
constant 25/23.976 = 1.042709376, never a per-file duration ratio:

    asetrate    (speed AND pitch)              0.9274 - 0.9846   median 0.9686
    rubberband  (tempo only, pitch preserved)  0.6591 - 0.7103   median 0.6707

**23 tracks across 12 files, `asetrate` higher on 23 of 23.** The lowest
`asetrate` reading exceeds the highest `rubberband` reading — **the
distributions do not touch.** Median difference +0.2942, six times the ±0.05
spread that per-file factor error would have introduced.

**The control went first, and without it the twenty-four measurements would have
meant nothing.** The worry was that Chromaprint might be blind to 70.7 cents, in
which case the question is unanswerable by this instrument:

    identity                                   1.0000
    pitch +72.4 cents, tempo untouched         0.6721
    pitch -72.4 cents, tempo untouched         0.6635
    rubberband passthrough (tempo 1, pitch 1)  1.0000

**The passthrough arm is the one that closes it**: `rubberband`'s own processing
does not damage the fingerprint, so the gap between the arms is **pitch
handling** and not a lossy filter.

**And the mechanism fell out measured rather than argued.** The `rubberband`
arm's median is **0.6707**; the pitch-only control is **0.6721**. They are the
same number. `rubberband` corrects the tempo and leaves the +70.7 cent shift
exactly where it was, **so its residual IS the pitch penalty** — two independent
measurements landing on top of each other, not an inference from the physics.

**So a PAL transfer is a naive speed change and `asetrate` is its faithful
inverse.** `rubberband` preserves a shift that ought to be undone. §73's proxy
defect is confirmed by measurement: the branch must key on *what the source
transform did*, and for a PAL pulldown it moved the pitch.

**What this does NOT settle, stated by the agent that ran it rather than asked
of them.** The same physics says `asetrate` is also the faithful inverse for
NTSC pulldown — but at **1.7 cents** that shift is about forty times smaller
than the one this control shows the instrument resolving. **Fidelity cannot
settle the NTSC case either way.** If the NTSC files come back
indistinguishable, that is a fact about the instrument's resolution, **not**
evidence that the methods are equivalent there. The argument for `asetrate` on
NTSC rests on the physics plus this PAL result — not on a measurement of the
NTSC files.

**Run by a different agent from the one whose prediction it confirms**, at that
agent's own request. A prediction tested by its author is worth less than one
tested by someone else.

**CORRECTION to the limit above, and the correction matters more than the
result it qualifies.**

The entry states *"fidelity cannot settle the NTSC case either way"*, because
1.7 cents is about forty times smaller than the 72-cent shift the control
resolves. **That is false.** One segment measured the curve:

    shift (cents)   factor        fidelity vs original
     identity       -             1.0000
          +0.0      1.0000000     1.0000
          +1.7      1.0010000     0.9742   <- NTSC pulldown. RESOLVED.
          +3.5      1.0020000     0.9606
          +8.6      1.0050000     0.9258
         +17.2      1.0100000     0.8772
         +72.4      1.0427083     0.6889   <- PAL

**Chromaprint resolves 1.7 cents at a 2.6% drop, and identity reads exactly
1.0000, so that drop is not noise.** The curve is smooth and monotonic from 0 to
72 cents with no floor between. **The NTSC case is settleable by fidelity.**

**How the false limit was produced: three agents agreed with each other.** One
proposed it, a second endorsed it, the Lead wrote it into an invariant and
relayed it to the owner as established — and it was asked to be put into a
docstring, where it would have sat as a durable constraint sourced to nothing
but consensus. **Nobody measured it until one agent, asked to document it,
measured it instead.**

**So: agreement between reasoners is not evidence, and it is more dangerous than
a single wrong claim** — each person's confidence is drawn from the others'
rather than from data, and the more people concur the less anyone feels obliged
to check. **A limit is a measurement.** "This instrument cannot resolve X" is a
claim about the instrument and costs one control to test; the reasoning that
makes it plausible is not a substitute for the control.

**And it inverts the consequence.** The six NTSC files, built with the method now
believed wrong, become a **real test** rather than an unmeasurable one: if
`asetrate` reads ~1.0 against `rubberband`'s ~0.974, the physics holds at 1.7
cents as well as at 72.4.

## 77. Four versions of one check, each wrong in a new way — encode what it DISTINGUISHES, not what it usually means

A precondition asking *"did the worker load its cut-point table?"* has now been
wrong four times, and the progression is worth more than any of the four bugs:

    v1  read the ABSENCE of "no cut-point record" as health
        -> a freshly recreated container has an empty log; passes on the very
           run it protects                                              (30)
    v2  read the WHOLE appending file and matched the message
        -> .log.error appends on retry; matched the PREVIOUS run        (52)
    v3  located the log as "newest .log.error in the folder"
        -> the folder is per-season and holds 43; read another episode  (52)
    v4  read "no cut-point record for this candidate" as a CONFIGURATION fault
        -> it is ALSO the true per-file statement for a file that was
           never scanned

**Each version was written to fix the last. Each was wrong in a new way. The
common cause is that every version encoded what the message USUALLY means rather
than what it DISTINGUISHES.**

v4 is the purest case. That one line is true in two different worlds:

    the table list is EMPTY        -> configuration; EVERY file fails identically
    this file is not IN the table  -> a fact about the file, entirely normal

The six newest error files had never been scanned — absent from `cut_points_c`,
`cut_points_d` and the configured table, checked by realpath — so *"no cut-point
record for this candidate"* was **simply correct**, and the check called it a
configuration failure and stopped the run twice.

**The fix is to ask which world**: does this candidate have a record in any table
on disk? **Record present and the worker cannot find it → configuration. No
record anywhere → the worker is right.** That is a test that separates the two;
the message alone never could.

**So the rule for repairing any check of this family: do not ask what the signal
usually means. Ask what two states could produce it, and build the test that
tells them apart.** A fix that encodes the common meaning will pass every case
you had in mind and fail on the next one — four times running, here.

This was foreseen and still happened. When the underlying config defect was first
raised the warning was that the message would be *"correct and misleading at
once"*. **Knowing the ambiguity was not enough; only a test that resolves it is.**

**v5, introduced while fixing v4 — and a DIFFERENT failure from the other four.**

    v5  the fix referenced a constant defined in another module. It passed
        `ast.parse`, was checked as "syntax ok", shipped, and raised NameError
        on the third file, killing the run.

**A syntax check is not a name check.** The through-line for v1–v4 — encoding
what a message usually means rather than what it distinguishes — does not
explain this one. **v5 is: validating the property that is cheap to check
instead of the property you need.** `ast.parse` answers "is this parseable",
which is not "will this run", and the gap between them is every name the module
does not have.

The corrected version was **exercised rather than parsed**, against two known
answers:

    err_694  has_record = False   (one of the six, never scanned)
    err_213  has_record = True    (record present in cut_points_c)

**Both correct — and that test costs less than the ast check did**, because it
runs the thing rather than inspecting it. §66 said a unit test cannot tell you
that nothing calls the unit; this is smaller and meaner: **a parse cannot tell
you the code would run at all.**

**AND THE HARDEST LESSON IS NOT ANY OF THE FIVE.** When the underlying
configuration defect was first raised, the Lead wrote that this message would be
*"correct and misleading at once"*. **The ambiguity was foreseen, written down,
and it still cost two stopped runs.**

A warning that names an ambiguity is not a defence against it. *"This signal is
ambiguous"* tells a reader to be careful; it does not tell them **which two
states to separate or how**, so the next person to touch the check — including
the person who wrote the warning — re-encodes the usual meaning and is surprised
again. **The warning was correct, recorded, and unactionable in the form it was
written.**

**So: when you notice an ambiguity, do not record the ambiguity. Record the
test.** "X is misleading" is a note. "X is true when A or when B; here is the
query that tells them apart" is a defence. Anything less converts into five
versions of one check.

## 78. The staleness detector was blind to the staleness of a file it depends on

`corpus_delta.py` exists to ask *"has anything changed under us?"* — arrivals
and measurement staleness. It was run rather than read, and it answers those two
questions well: 315 live against 315 in the ledger, 0 new, 0 disappeared, 0
reassigned, and it correctly flagged **9 measurements whose output was rebuilt**
and **13 whose output is gone entirely**.

**It does not cover derived-file currency, and the way it fails is worse than
not covering it: it READS `classified.json` as a source of truth and never
checks its currency.** It consumed a 309-entry file against a 315-file corpus
and said nothing. It is the **eighth** consumer of a derived file that has no
producer anywhere in the tree.

It has a `known_to_classified` field — but that fires only on the **arrivals**
path, so it would have caught the six new files 35 hours ago and **cannot catch
them now that they are ingested.** The check exists, in the right tool, wired to
the one code path where it is useless.

**So the standing reconcile does NOT exist.** There are three categories, not
two:

    corpus arrivals         covered
    measurement staleness   covered
    DERIVED-FILE CURRENCY   nothing asks it

Tonight's three gaps were all the third kind — mapping at 309, `classified.json`
at 309, an inbox file invisible for 118 days — and every one was found by hand.

**And the general form is the sharpest version of a theme this file keeps
circling: a tool that checks freshness must check its own inputs' freshness
first, or it certifies the corpus using the very artefact that is wrong.**

## 79. A measurement is about an artefact at a time, and 8 of 27 verdicts outlived theirs

The same run found a defect in the work of the agent running it — **8 of 27
reported band verdicts no longer describe the file they were measured on:**

    err_45, err_300     output REBUILT since measurement   (both "undetermined")
    err_12, 46, 51,
    59, 63, 70          output GONE ENTIRELY               (five "genuine desync")

So the headline *"22 genuine desync, 5 undetermined"* **stands for 19 of 27**.
For the other eight it is a statement about artefacts that were rebuilt or
consumed after the measurement was taken. **The measurements did not become
wrong — they became about something that no longer exists** (§23, now with a
count attached).

**None of this was visible from the verdicts.** A row saying *"genuine desync,
0.72 frames"* looks identical whether its file is on disk unchanged or was
deleted an hour later. **Only a tool that compares the measurement's timestamp
against the artefact's could see it**, and the agent found it in its own numbers
by running that tool rather than by re-reading them.

**So: a reported verdict needs the identity of what it measured, not just the
result** — an mtime, a size, an md5 — and any long campaign must re-check that
identity before quoting old rows. A corpus that is being actively rebuilt makes
every stored measurement provisional, and the ones most likely to be stale are
the ones about files interesting enough that somebody re-ran them.

## 80. A repair destroys the evidence of the defect — pin the before-state first

Five published outputs carry truncated or empty audio. Before re-merging two of
them to test whether the truncation is recoverable, the agent pinned the
before-state of **all five**: per-track `DURATION`, file size, mtime and a
first-megabyte digest.

**A successful repair overwrites the only artefact that demonstrates the
problem.** If the re-merge produces a good file, the evidence that anything was
ever wrong exists nowhere except in a report — and reports of a defect are worth
much less than the defect. The pin costs seconds and it is the difference
between *"we fixed it"* and *"we can show what was wrong and that it is now
right"*.

**This is the mirror of §79.** There, measurements outlived the artefacts they
described and became statements about files that no longer exist. Here the
artefact is about to be destroyed **deliberately** — which is the one case where
nothing is lost by accident and everything is lost by not thinking about it.

**So: before any action that rewrites an artefact under investigation, record
its identity and the specific property under investigation.** Not a description —
the numbers. `DURATION` per track, size, mtime, digest. Anyone can re-derive a
conclusion from those; nobody can re-derive them from a conclusion.

**And a bound from the same pass, worth keeping**: the completeness check was
re-run over the corpus after the 119-file campaign closed. **119 outputs, the
same five affected, and the nine outputs the campaign added are all complete.**
So the five are a property of *earlier* merges rather than of the current path —
a negative that bounds the defect in time without claiming it is fixed.

## 81. A rule can be well-formed, agreed, and have no instrument for its precondition

The owner ruled that a cross-language patched (chimeric) audio track is
**provisional**: when a fully compatible stream — *"with the same
interpretation"* — is later found, the chimeric track is deleted and the
compatible one becomes the only one. Clear, sensible, and **two of its three
parts cannot be built as stated.**

**Supersession has no mechanism.** Nothing in this architecture revisits a
finished file — no `mkvpropedit`, no `--edit`, no `remove-track`, no re-open of
a published destination. The only write to a destination is the publish step. So
*"later found"* has no "later": the rule reduces to **"prefer a compatible
stream over a chimeric patch at merge time"**, and anything more is a new
subsystem, not a rule.

**And "the same interpretation" has no measurement in this project.** The
obvious candidate is the dub gate at `FIDELITY_BAR` 0.90 — but 19
metadata-labelled different-dub pairs read **0.8334 to 0.9922**, so 18 of 19
clear it (§62). Chromaprint is confounded by the shared music-and-effects bed,
which is *why* a different dub reads 0.95. Envelope correlation separates
recordings well (0.27 against 0.98 on one folder) but answers *"is this the same
recording, aligned"* — for an unaligned candidate it says nothing.

**AND THE ASYMMETRY RUNS THE OPPOSITE WAY FROM THE PREVIOUS DECISION, which is
the part the Lead got wrong.** §62 established that for **deletion** the
irreversible error is discarding a distinct dub, so the reversible direction is
to keep. It was then suggested that the same bar might be *sufficient* for
supersession because the question is different. It is not:

    deletion      a false "same" LOSES a distinct track
    supersession  a false "same" INSTALLS a different dub in place of a
                  chimeric one -- replacing a track that is known-partial and
                  honestly labelled with one that is confidently wrong

**The second is worse.** A chimeric track declaring *"contains 86 s of
Japanese"* is a known quantity; a "fully compatible" French track that is
actually a different regional dub is not, and nothing downstream would question
it.

**So the rule's precondition has no instrument, and building it on the 0.90 bar
would encode a decision the evidence cannot support.** That is the thing to
report — not that the rule is wrong, but that it is unimplementable until
something measures interpretation. The unbuilt separator (§62) is the same gap
seen from the other side.

**General form: when a rule is agreed, check that each precondition names a
measurement that exists.** "Fully compatible", "the same interpretation", "a
better source" are all well-formed English and none is a predicate until
something computes it. **An agreed rule with an unmeasurable precondition does
not fail loudly — it gets built on the nearest available number**, which is how
a 0.90 fidelity bar ends up deciding something it was never calibrated for.

## 82. An incomplete check that admits its limit is usable; a complete-looking wrong one is a fabrication

The derived-file currency table needed a "what produces this?" column. **Three
attempts at inferring producers statically each failed a case whose answer was
already known:**

    "the module writes something"   gave classified.json a producer it does
                                    not have
    "a write within 5 lines"        denied a producer to a file written that
                                    same evening -- the path is a constant and
                                    the write goes through a variable
    transitive alias tracing        credited a READER of classified.json as its
                                    writer; the cascade binds most names in a
                                    module

**Two known answers were available and every version failed one of them.** The
shipped check reports only what it can establish — a line containing both the
literal filename and a write — so it produces **no false producers** and
**misses producers that write through a variable**, which is the common shape.
Its output says so: *"no write site found, verify by hand"* over a list that is
a **superset** of the genuinely unproducible.

**The temptation is the finding.** That same list under the heading *"nobody can
regenerate these"* reads as a discovery of 25 orphaned files, and **two of the
three attempts would have produced exactly that.** An incomplete check that
declares its incompleteness can be used; a complete-looking wrong one is a
fabrication with a table around it.

**A second methodological choice worth keeping: classify by coverage fraction,
not by reader count.** The first cut used readers and filed a 25-entry verdict
file as a stale resolver *"missing 290 of 315"* — burying the one real case
under four false ones. **A file intending to enumerate the corpus lands near
complete and falls a few short; a sample sits far below.** The line is 90%, and
the fraction prints for every file so it can be disagreed with.

**And the tool found its author's own work in the defect.** The
different-recording sweep walked `classified.json`, so it measured **309 of
315** and reported a denominator that looked complete — by the agent who had
spent two days insisting a number without its denominator is not a result.
**A stale resolver does not stay in its own lane**: it propagates into every
artefact derived from it, and those artefacts then look authoritative.

    classified.json           13 readers, 309 of 315, NO PRODUCER (24 mentions,
                              all read-only, hand-verified)
    different_recording.json   1 reader,  309 of 315

**§71 CONFIRMED BY DIRECT MEASUREMENT — the truncation is LIVE, deterministic,
and happens AFTER verification.**

Two truncated outputs were re-merged on the current merge path. Both were
genuinely rebuilt — new size, new first-megabyte digest, new mtime — and **both
came back truncated to within milliseconds of before:**

    err_160   fre 108.010 -> 108.009 s     jpn 108.008 -> 108.010 s
    err_194   fre 171.184 -> 171.175 s

**Not a historical accident from an old build.** The current path produces it
every time, and no amount of log reading would have established that.

**And the mechanism is localised rather than inferred, because the plan and the
verify are in the same run's log:**

    plan for the fre track : 4 pieces, master[0.000..1523.606]
    verify summary         : duration 1522.680 vs master 1523.606 s, 0.06% off
    published track        : 108.009 s

**The chimeric intermediate was 1522.680 s and verified as full length. The
published track is 108 s.** Same file, same run, minutes apart. So the loss
happens **after the build and after verification, in the final merge that
publishes** — which is exactly what §71 said and could not previously show.

**A bonus fact that settles an open question**: this run cleaned its scratch
directory, so successful runs do clean up after themselves — which means the
16 zero-byte `chimeric_*` directories are **the residue of runs that failed**,
not of runs that were interrupted.

**And the cause is NOT named.** A track occupying `[D, D+1522.68]` clipped to a
1523.6 s container leaves 108 s at `D = 1415.6 s`, which is arithmetically tidy;
the log records the chimeric file's audio delay as **0.000**; and **the final
merge command is not logged at all.** The agent declined to name a cause on
arithmetic alone, having watched one satisfying explanation collapse the same
day when a log turned out to record something better. **The final merge command
being unlogged is itself the reportable gap** — the step that loses the audio is
the one step nobody can inspect afterwards.

**Refinement to §78: a stale resolver damages a consumer that ENUMERATES from
it, and only degrades one that ANNOTATES from it.**

The 119-file scope was built through `master_summary.build()`, which reads the
producerless 309-of-315 `classified.json`. **The scope was not truncated** — all
six of err_694–699 are present in the 119, verified by id.

The reason is structural, not luck: **`master_summary` enumerates from
`/errors` and uses `classified.json` only for the `cls` LABEL.** A stale
resolver therefore cost a label and not a row.

    consumer ENUMERATES from the stale file  -> rows silently missing;
                                                the denominator is wrong and
                                                nothing says so
    consumer ANNOTATES from it               -> rows all present, some labels
                                                absent or wrong

**These are different severities and they have been discussed as one.** The
sweep that reported *309 of 315 as a complete denominator* was enumerating; the
119-file scope was annotating and survived. **So "13 readers of a producerless
file" is not 13 instances of the same defect** — the blast radius is whichever
of the 13 enumerate, and that has not been established.

**The general rule: when a shared input goes stale, classify each consumer by
what it takes from the file before estimating the damage.** The question is not
*"does it read the stale file"* but *"does it get its POPULATION from the stale
file"*. A tool that asks the live source what exists and the stale file only
what things are called is degraded; a tool that asks the stale file what exists
is wrong, and wrong in the direction that hides work.

## 83. Every per-stream duration in the chimeric module is the CONTAINER duration wearing a stream's name

`probe()` requests `stream=...,duration,...` — so the code appears to hold a
per-stream length. **It does not.** For Matroska `stream.duration` is **absent**;
the real per-track length lives in a **`DURATION` tag**, and `probe()` asks only
for `stream_tags=language,title`. So it falls back:

    try:
        s_dur = float(stream.get("duration"))
    except (TypeError, ValueError):
        s_dur = duration            # <- the CONTAINER duration

Measured against a real output:

    container duration                          1490.560 s
    what the code believes all SIX streams are  1490.560 s, identically
    what the DURATION tags actually say         1490.464 / 1490.560 / 1490.464

**A track that stops at 108 s in a 1523 s file is recorded as 1523 s — not
approximately, exactly, because it is literally the same number.**

**So a short track passes `verify_chimera` BY CONSTRUCTION.** Its only length
test is `rel <= DURATION_TOLERANCE` against `format.duration`; there is no
per-stream check anywhere in the module. And it is not only verification —
`plan_segments` takes `cand_limit` from the same fiction, as does the
`stream["duration"] < c_hi` guard. **Anything in that module reasoning about how
long a track is, is reasoning about the container.**

**This is why the completeness pass found five affected outputs that every other
check had passed** (§72): comparing each stream's `DURATION` tag against the
container duration is precisely the comparison this module cannot make, because
both sides of it are the same value.

**What is established**: nothing in that module can observe a short audio track,
reproducible in one command. **What is NOT claimed**: that this *causes* the
truncation. It guarantees the pipeline cannot notice one, and a planner working
from a genuinely short candidate track would plan past its end believing it full
length — **a mechanism, flagged and not asserted**, by an agent who had two
mechanisms refuted this week and one confirmed.

**The general shape: a field that is absent rather than wrong.** `stream.duration`
does not return a bad number, it returns nothing, and the `except` clause
substitutes a plausible one from a different level of the file. **A fallback that
silently changes what a value MEANS is worse than a missing value**, because
every downstream consumer reads a number that is correctly typed, plainly
sensible, and about something else.

## 84. A component can be correct, tested, and unreachable by the process it was built to observe

The `mkvmerge` logging shim was built to record the final merge command, which is
the one step in the truncation path that leaves no trace. It was tested four ways
— passthrough, exit code preserved, arguments with spaces, and missing binary
giving exit 127 rather than resolving something else. All four pass. **It still
cannot be installed**, and the reason has nothing to do with the shim.

**The merge runs inside the container, and the container reads its config from
inside its own image.** `run.sh` invokes with `--pwd /home/vmsam` and no
`--config`; the entry point does `os.chdir(args.pwd)` and `--config` defaults to
the *relative* `config.ini`; the Dockerfile bakes that file in. So the
`[software]` table the merge actually consults is a file this host cannot edit.
Editing `/home/vmsam/config.ini` on the host changes host-side runs and nothing
else — and a path under `agent_trees` is not visible inside a container that
mounts only `/config` and `/srv`, so pointing the entry there would fail every
merge with the shim's own fail-closed 127.

**And there is no runtime override at all.** `--config` is never passed in
production, and `tools.software` has no environment fallback. This is the
`VMSAM_CUT_POINTS_TABLE` wall one level lower: not just the container's
*environment* but its **binary selection** is unconfigurable from outside the
image.

**The evidence is indirect and was reported as such.** The container's logs stamp
distinct commits at discrete points while HEAD moved continuously; a bind-mounted
source tree would track HEAD, a stamp that moves only on deploy is an image. That
argument is strong but it is not proof, and `docker inspect` settles it in one
command — **which this host cannot run, because it has no docker CLI**. The
owner's box does.

**The lesson is about sequencing, not about the shim.** *Reachability is a
property to establish before building, not after.* The build was cheap and the
tests are good, but every hour of it was spent on an instrument that cannot yet
be attached to the thing it measures. **Ask what loads the configuration, and
from where, before writing the thing the configuration would point at.** The
corollary is the cheaper design that was available all along: instrument at a
point that ships *inside* the image, rather than at a point the image has to be
told about.

## 85. A stated prediction that is refuted is worth more than a correlation that is confirmed

The hypothesis was that the five truncated outputs would be built from materially
short source tracks — that the merge inherited the shortness rather than creating
it. The direction was stated before the measurement, which is the only reason the
result can be read forwards.

**Every candidate and every master is full length.** The worst source shortfall
across the five is 2.0 s; the worst *output* shortfall is 1440.9 s. **The
shortness exists only in the output**, so the five are not in the 285 short-source
tracks and the source explanation is not weakened — it is **excluded**. A
confirmation would have left the source correlated with the defect and still owed
a mechanism; the refutation removes an entire class of cause in one measurement.

**It also split one finding into two.** The 285 tracks whose stored length is
materially below their container are a genuine instrument blind spot — the module
cannot see a short track — and they are **not** the cause of these five. Those had
been half-treated as a single result, and a confirmed prediction would have
welded them together.

**The control caught a false positive of exactly the §69 shape.** One file flagged
18 of 18 tracks short by the same 86 s. That is a container whose duration exceeds
its content, and measuring each track against the container reads padding as
truncation — a *common-mode offset*, the same defect the dub work removes by
differencing. **Compare each track to the longest track in its own file** and the
offset cancels; the file becomes its own control (§69).

**With the corrected discriminator the defect is 60% more widespread than the
reported set**: 8 outputs on disk carry both a full-length and a short track, and
3 of them had never been reported by anyone. It flags 8, not everything — the
instrument discriminates rather than firing on all inputs.

**Two details characterise the mechanism better than the word "truncated":**

- One output carries a track of **one millisecond**. That is not a short track,
  it is an empty one.
- Another has six short tracks stopping at 441.461, 441.473, 441.473, 441.479,
  441.486 and 441.486 s — **within 25 ms of each other** — while its three
  full-length tracks run to 1441 s. The short ones are all candidate-derived and
  the full ones all master-derived. **A single common cut point on one side of
  the merge is per-SOURCE, not per-track**: the assembler loses the candidate
  input at an instant, it does not damage tracks individually. Any explanation
  that corrupts tracks independently is already refuted by that 25 ms spread.

## 86. Track ORDER is not provenance once the assembler substitutes

The truncation looked candidate-specific: in the first case examined, every short
track was candidate-derived and every full-length track master-derived. That
reading was wrong, and the way it was wrong is the reusable part.

**Provenance was inferred from position** — the first *n* streams of a language
were taken to be the master's. **That rule does not survive substitution.** In one
file the master's jpn track is 1522.7 s and the output's *first* jpn track is
108.0 s: the master's track was dropped and replaced by the candidate's. Once the
assembler can substitute, **ordering tells you where a track sits, not where it
came from**.

**The sound test is which source carries that language at all.** Under it, three
of the eight truncations lose a track in a language **the candidate does not
have** — 137.8 s, 808.1 s, and a pair at 1377.7 s and 1457.9 s, all master-only.
So "the assembler loses the candidate input" is **excluded**, and a narrow, sharp,
wrong direction was avoided.

**The signature itself held.** Where more than one track is short, they stop
together: spreads of 0.000 s, 0.001 s and 0.025 s across 2, 2 and 6 tracks. Six
tracks within 25 ms **refutes any mechanism that damages tracks independently** —
whatever is lost, it is lost at an instant, on one side of the merge, for reasons
not yet established.

**One case fails the signature and is held apart rather than absorbed.** Its two
short tracks stop **80.224 s apart**, its container is 5997.5 s against ~1450 s
for every other case, and it carries 17 tracks in 11 languages. Its 4619 s
shortfall made it the tempting headline — which is exactly why it needed checking
before leading with it (§34). **The discriminator did not exclude it**; it was
excluded explicitly, on the spread, rather than crediting a tool that missed it.
Seven files are consistent with one mechanism and one is not.

**Two things deliberately not claimed**: why the assembler loses a track at an
instant, and why the instant differs per file. And an output carrying a **1 ms**
track is plausibly an *empty* track rather than a *cut* one — **a different
event**, not folded into the same count without evidence.

## 87. A threshold measured against the wrong reference is wrong in BOTH directions

The truncation detector compared each track's length to its **container**, and
called a track short below 0.5 of it. That bar was recommended as sufficient. It
failed twice over, and the two failures point opposite ways.

**It under-reported.** One track is 0.54 of its container and cleared the bar by
four hundredths — while being **693 seconds shorter than the siblings it was
merged with**. The file was reported complete.

**It over-reported.** A container longer than its content shifts every track
equally, so one real file read **18 of 18 tracks short by the same 86 s**. That is
padding. A list that contains it stops meaning *this file lost its audio*.

**Both disappear under the right reference: the longest track in the same file.**
The common offset cancels, what remains is a track short relative to its siblings,
and the bar can move to 0.9 — affordable precisely because siblings of one merge
should agree within seconds. **The reference, not the number, was the defect**; no
choice of fraction against the container fixes both directions at once.

**A fixture must distinguish the alternatives, not merely touch them.** The first
test written for the corrected rule was survived by a mutant: the padding fixture
used an 86 s gap, which clears 0.9 under *either* reference, so it exercised both
and separated neither. §32 one level along — the fixture has to be able to fail
for the reason under test.

**Two accounting disciplines this exposed, both cheap:**

- When another agent reports cases you missed, **check which are yours**. Of three,
  one was a discriminator gap, one a bar gap, and one a *snapshot* gap — an output
  written after the sweep ran, which the same pass flags correctly when re-pointed.
  Accepting all three or disputing all three would both have been wrong.
- **Two counts of one phenomenon must be reconciled by naming the file that
  differs**, not by picking the larger. Seven versus eight here is exactly one
  file — a 64.8 s shortfall on 1477 s, real but two orders of magnitude milder
  than a track that lost 90% of itself. It belongs in a *separate bucket*, and the
  disagreement was a bucket boundary rather than a detection difference.

**And the gate is cheaper than it looked.** On the largest case, every existing
verifier checkpoint lands **inside** the short tracks — none falls past their
ends. The file was catchable at the sampling positions already in use, and was
missed only because **nothing compares tracks to each other**.

## 88. A boundary drawn after seeing the file it separates describes that file, not a category

Two agents disagreed on how many outputs the truncation affects. The dispute was
never about detection — both found the same eight — it was about where the list
stops meaning *this file lost its audio*.

**The first proposal was four tiers, and it put one file alone in a middle band
whose edges were chosen after seeing it.** That is a description of a file wearing
the costume of a category, and it is the same objection the proposer had been
making all week to other people's thresholds.

**The second proposal was one line at 0.5, and it was drawn from six of the seven
cases.** It would have filed a track that lost 46% of itself as merely *ending
early*.

**The answer is in the data, and neither agent had it alone.** Short-track ratios
are 0.071, 0.094, 0.114, 0.230, 0.243, 0.306, 0.538 and 0.956. **The largest gap
is 0.418 wide with nothing inside it.** A single line at 0.75 sits in that empty
band, so it is a property of the distribution rather than of whoever proposed it —
and the band is exactly what 0.5 could not have known about from six files. Final
shape: **1 empty, 6 lost most of their audio, 1 ends early.**

**Two agents converged where one produced only a defensible-against-my-sample
answer.** The one who yielded had the weaker argument and said which part of their
own reasoning was circular.

**And the mechanism narrowed by contradiction, not agreement.** Arbitrary cut
instants landing on no planner boundary were read as pointing at *a resource or a
timeout*. A re-merge then reproduced the truncation **to the millisecond** —
which resource contention and load-dependent timeouts do not do. Excluding the
planner was right; the replacement guess was wrong, and only the reproduction
could say so. The surviving constraint is much tighter than either finding alone:

> **deterministic, per-file, at an unstructured instant, on no planner boundary,
> and able to cut either side of the merge.**

**A rule you cannot replace can still be checked at the point of use.** Provenance
by track order (§86) has no cheap alternative where a language exists on both
sides — but if the track selected as candidate-derived has a duration matching the
**master's** track of that language, a substitution happened and the selection is
wrong. That does not resolve the ambiguous case; it catches the case where order
lied, which is the case that caused the error. **An unfixable inference is not the
same as an unguarded one.**

## 89. A count that moves with the threshold is a property of the bar, not of the phenomenon

A ninth affected output turned up: a track 1.6% short of its file. It **appears at
a 0.99 bar and disappears at 0.98**. The seven severe cases sit below a
0.418-wide empty band (§88) and **no threshold inside that band changes the set**.
Those are two different kinds of fact and they must not be added together.

**Report it as seven, and NAME the marginal cases rather than counting them.**
"Nine affected" would hand the owner a number that changes when someone edits a
constant, and nothing in the output would say so.

**The reciprocal caught the agent who was enforcing it.** The same tool whose
severe/mild line at 0.75 is defensible — because it sits in the empty band —
draws its *is it affected at all* line at a flat **10 seconds**, with no such
justification. That is the identical unjustified constant its author had been
objecting to in other people's work all week, sitting one line up in their own
file, and it took another agent to see it. **A rule you apply correctly in one
place is not thereby applied in the next.**

**A guard must RAISE, not return.** The provenance guard (§86) is built as two
independent checks, either sufficient: **content** — a language absent from the
candidate cannot have come from it, definitive and needing no durations — and
**duration** — a selection whose length matches the *master's* track of that
language is the master's track. The case that started this fails both: the
selected track's language is absent from the candidate entirely, and its length
matches the master's to the millisecond. **Order lied twice over.**

The load-bearing test is the mutant asserting the guard *raises* rather than
returning a flag. **A returned flag can be ignored by its caller, which is exactly
how the wrong provenance reached a report and then the owner.**

**And "indeterminate" is a distinct return, not a pass.** Where both sources carry
the language and their durations agree, position is unfalsifiable — the guard says
so instead of reporting sound. §63: exclusion narrows, identification decides.

## 90. Reaching a population through a derived file inherits its coverage, not just its values

Two findings from one measurement, and the smaller one is the threshold.

**The 10 s lower bar turned out to be defensible, and that had never been
established.** Its comment read "well above the ~2 s spread and far below the
smallest real case" — which *sounds* like a justification and is a **restatement
of the number**. Measured across every non-longest audio track reachable, the
shortfalls above 1 s run 1.997 (×6), 2.007–2.031 (×8), 5.908, 7.072, then
**nothing at all until 64.797**. An empty band **57.7 s wide**, and 10 s is inside
it.

So the line is a property of the data — but **the honest statement is that the
number is not special and the SET'S INSENSITIVITY is the property**. And the band
is **not centred**: 5.9 and 7.1 sit just below it, so a 5 s threshold would add
two files. **The band's lower edge is the constraint, not its midpoint.**

**The symmetry to state plainly: both lower bars are principled and neither agent
has a principled UPPER bar.** That asymmetry is exactly why a marginal ninth file
exists at all (§89), and saying so is worth more than picking a number for it.

**The larger finding is a denominator, and it is worse than any threshold.** The
scan reached its population through a derived file — `classified` → the ledger's
merged-file field — and found **108 outputs**. The sweep that goes to what
actually publishes outputs finds **120**. **Twelve outputs were never looked at.**
Not a threshold difference: a *coverage* difference, invisible to every check
inside the tool, because a tool cannot see the rows its source never contained.

So "8 affected" was **8 of 108, reported without the denominator** — and the file
another agent found lies *inside* the 57.7 s band, meaning the threshold would
have caught it. It was missing for the other reason entirely.

**This is §30 and §50 recurring in the tool written after that work was done.**
Knowing that derived files go stale does not protect you from reaching a
population through one; the failure is not in the values it returns but in the
rows it never had. **Ask what publishes the thing you are counting, and count
that.** A population figure without its denominator is not a small omission — it
is the whole claim.

## 91. Two agents counted one population and both were wrong, in opposite directions

The truncation set was reported over two different denominators — 108 and 120 —
and the reconciliation named the difference rather than picking a number. **Both
figures were wrong.**

**One was too narrow because a ledger defined it.** The population was reached
through a derived file, so it inherited that file's coverage: twelve outputs were
never in it (§90).

**The other was too broad because a filesystem defined it.** The sweep walked the
whole output directory and picked up everything with the right extension,
including **five bench files** from a benchmarking directory that are not
published outputs at all.

**Neither agent was reading a definition of "published output".** Each was reading
a *convenient proxy* — one a ledger row, one a directory tree — and a proxy's
disagreement with the thing it stands for is invisible from inside the tool that
uses it. **The true figure is 115**, and it took two independently-wrong counts to
find it, because a single count agrees with itself.

**Final shape: 106 complete, 6 lost most of their audio, 2 end early, 1 empty.**
Seven affected by the phenomenon, plus two milder cases that exist relative to a
0.99 threshold (§89). The bench files were all complete, so the affected set never
moved — **only the denominator did, and every "N of 120" quoted earlier should
read "N of 115".**

**The vindication is narrower than it looks, and the agent who was vindicated said
so.** The marginal case found by the broader sweep was not found by a better rule:
its shortfall sits *inside* the other agent's empty band, so that threshold would
always have caught it. **It came from sweeping a different set, not a better
one — and that set was also wrong.**

**A count of IDs is not a count of OUTPUTS.** "108 reachable" and "103 named"
were both correct: 108 error ids map to 103 distinct merged files, because
several ids share one output. Counting ids and calling them outputs happened
*inside a report about counting the wrong things* — the unit slips even while the
subject is units.

**And two unnamed counts, 12 and 17, are answers to different questions.** Twelve
outputs are named by no *error id*; seventeen are named by no *ledger row*. They
are compatible only if five outputs carry an error id and no row — which is a
claim someone must check, not a subtraction someone may assume.

**The reconciliation gap sits at BOTH ends of the pipeline.** Of the 115 published
outputs, **17 are named by no ledger row at all** — recent, 1.4–4.2 GB, none
empty. That is §75 at the output end: the inbox file that sat 118 days was the
input end of the same missing check. **Nothing reconciles the publisher's tree
against the ledger in either direction.**

## 92. A distribution can refute BOTH hypotheses, and the third answer changes the fix

Twelve published outputs are named by no ledger row. The two obvious readings
were **backlog** — rows lag behind publication and will catch up — and **leak** —
rows are lost at some rate, forever. The age distribution refutes both:

    0-6 h  14 named, 0 unnamed      36-48 h  28 named,  4 UNNAMED
    6-12 h  22 named, 0 unnamed      48-72 h  16 named,  8 UNNAMED
    12-24 h 19 named, 0 unnamed
    24-36 h  4 named, 0 unnamed

**A backlog puts the unnamed at the NEWEST end; the newest 59 files are all
named. A leak spreads across every age; this stops cleanly in both directions** —
and the oldest *named* output (67.5 h) is older than the oldest *unnamed* one
(61.7 h), so named outputs sit on **both sides** of the unnamed population.

**It is an episode**: all twelve were written inside a 23.5-hour window, with the
pipeline recording rows normally before it and after it.

**That is a third answer and it needs a third fix.** Not patience, not a
reconciler — **find out what was different in that window.** A reconciler built
now would sweep up twelve files and say nothing about whether the gap can recur,
and §81 is the reason to stop: the precondition for *reconcile continuously* is
knowing the gap is ongoing, and it is not.

**The general form: two hypotheses that feel exhaustive usually are not.** Backlog
and leak differ in *rate*; the real answer differed in *time*, which neither
covered. **Plot the distribution before choosing between the two stories you
already have** — the shape names a third.

**And an arithmetic that works out is not thereby an explanation.** Two counts of
unnamed outputs stood at 12 and 17, and the five bench files (§91) made
12 + 5 = 17 immediately available. Both agents re-measured rather than reasoned,
and the sum turned out true **with its natural reading false**: the difference was
not five additional files of a new kind, it was **the same five bench files
appearing a second time**, because both of the larger figures were computed over
the same contaminated population. **One contaminated denominator produced two
contaminated figures, and one correction fixed both.**

The proposed compatibility condition — five outputs carrying an error id and no
ledger row — **does not hold, and did not need to**: the two sets are identical,
0 either way. Had it been reasoned instead of measured, the right answer would
have been reached through a wrong mechanism and the contamination would have
survived in the second figure. **A sum that reconciles is a hypothesis about
WHICH items differ, and that hypothesis is separately checkable.**

## 93. A guard that reads the wrong field is not dead — it silently guards a different proposition

Verified directly in `merge_video_chimeric.py`, three facts, one correction.

**`probe()` asks ffprobe for `stream_tags=language,title` and never for
`DURATION`.** Matroska stores per-stream length in that tag, so
`stream.get("duration")` is absent, the `except` fires, and **`s_dur = duration`
— the CONTAINER's length — is stored under a per-stream key.** Measured: 1926 of
1927 tracks take this path, the single exception being non-Matroska (§83).

**Two callers consume that field, and they are not equally harmful.**

The planner's `cand_limit = cand_stream["duration"]` merely believes a track is
longer than it is. The donor selector is worse:

    if stream["duration"] < c_hi:
        continue          # reject a donor that ends before the window needs it

**Its author's intent is unambiguous: refuse a donor too short for the requested
window.** With every stream reporting the container's length, it cannot reject a
short donor — **a genuinely short donor is selected and then read past its end.**

**The correction, in two steps, because the first version of it was also too
coarse.** The guard is not dead: it still fires when `c_hi` exceeds the
*container's* length. But its live range is **narrower than that** — the caller
reaches this line only after `_reading_is_consistent` has rejected any plateau
with `c_end > cand_duration + slack_s`, and `slack_s` is **2.0**. So by the time
the duration guard runs, `c_hi` is bounded above by the container's length plus
two seconds.

**Its entire live range is a two-second band at the very end of the file.** It
never fires on the proposition it was written for, and what it does enforce —
*"the window runs past the end of the FILE, within slack"* — is a **boundary
condition, not a length check**. The two coincide only when every stream runs the
container's full length, which the code never states and 33.7% of tracks violate.

**Three readings, each narrower than the last**: *never fires* → *fires past the
container* → *fires in a 2 s band past the container*. Only the third came from
reading the callers rather than the line.

**That is worse than an inert check and worse than a missing one.** A dead check
shows up in coverage. A missing check is absent from review. **A check that fires
on a related-but-weaker condition passes tests, appears in review as protection,
and reports success** — and every reader who scans that line comes away believing
the short-donor case is handled.

**The fix must record PROVENANCE, not just improve the number.** Today a caller
cannot distinguish *"this track is 1490.6 s"* from *"nobody knows, here is the
container's"*. A fix that silently returns a better figure leaves those two
indistinguishable and fixes only this instance; **one that reports where the
number came from makes the whole class visible.**

**And the expected consequence, stated before building**: the guard starts firing.
Merges that today emit a silently short track may instead **fail to find a donor
and raise**. That is the right direction — silent corruption becoming loud
failure — and it is still a behaviour change on real files, so it belongs behind
a flag, off by default, with the default proven unchanged.

**The mutant that must die**: remove `DURATION` from the request and the fallback
returns. **If the suite still passes, the fixture is not Matroska-shaped and
proves nothing** — which is exactly the gap that let this live (§32).

**Not claimed, and this must travel with the finding**: this does not cause the
truncation. Those cuts land on tracks whose **sources are full length** (§85), so
the planner's limit is not what shortened them. **This defect guarantees the
pipeline cannot NOTICE a short track; it is not evidence that it creates one.**

## 94. The window was publication times, not defect times — and the investigator was the cause

Twelve published outputs referenced by no ledger row looked like a bounded
23.5-hour episode (§92). **There is no episode.** The distribution was real and
every inference drawn from its shape was wrong, because the axis was misread.

**The mtimes are PUBLICATION times.** The twelve were published normally across
that window by ordinary sweeps. **Nothing was different in that window.** What
was different came later: those ids were **re-run and failed**, and

> **a failed run overwrites the ledger row and clears `merged_file`.**

The output on disk survives; the pointer to it does not. **That is the defect** —
not the absence of a row, but a failure path that discards a reference to a
completed artefact.

**The discriminator is not time, it is later re-running.** 24 named outputs sit in
the same window and **not one had its row rewritten**; the twelve were re-run and
failed *twice*. Grouping by publication time therefore grouped files that share a
history, which is §67 exactly — a run ordered by any property groups related
files — and here the property was invisible on the axis being plotted.

**The investigator's own repair was erased by the investigator's own next run.**
Eleven of the twelve were repaired by hand this morning; a 14-file discriminator
run hours later re-ran precisely those ids, they failed again, and the rows were
overwritten again. **Every one carries a `recorded_utc` inside that run's 24
minutes and no repair marker.** The repair was undone by the same agent, in the
same session, testing something else.

**So "do not build a reconciler" was right for the opposite reason to the one
given.** Not *the gap is bounded so it cannot recur* — but **a reconciler would
re-point the twelve and be undone by the next failed re-run, exactly as the manual
repair was.** Fix the clearing and the orphans stop; reconcile without fixing it
and you repair the same files forever. §81 holds, with its premise inverted.

**The far edge of the "window" is weakly bounded and was reported as such.** Eight
named outputs sit beyond the oldest unnamed one and none was rebuilt by a recorded
run, so their mtimes are original — but **37 of 103 named outputs WERE rebuilt,
and mtime is not creation time**, so beyond that horizon there is no evidence
either way. **A boundary at the edge of your evidence is the edge of your
evidence** (§36).

**One of the twelve is still unidentified, and that was stated rather than
rounded off.**

## 95. Nothing decided to clear the field — so there is nothing to guard

A failed run leaves a published output with no ledger row pointing at it (§94).
The obvious fix is to stop the clearing. **There is no clearing.**

`process_one` builds a **fresh dict for every outcome** and sets `merged_file`
only inside the success branch; `record_row` then **replaces the row wholesale.**
Nothing removes the field — the row is rebuilt from scratch and the field is
simply absent from the new one.

**That changes the shape of the fix, not just its location.** A guard needs
something to intercept, and there is no decision point to attach one to. The fix
belongs **where the new row is composed**, or as a **carry-forward** in the
writer. Looking for the line that deletes it would have found nothing and been
read as the defect not existing.

**Carry it forward only if the path still resolves.** Clearing *is* correct in
one case — a re-run that removes or replaces the output and then fails, leaving a
pointer to something gone. Measured on the identified orphans: **artefact survives
11, gone 0**, because `publish_merged_file` stages to a temporary name and
`os.replace`s, so it never deletes first. **The case is unobserved but real, so
retention must be conditional on existence rather than unconditional.**

**Fifteen readers consume the field and every one fails the same silent way.** The
worst is not an analysis tool: a cleared `merged_file` makes an output
**ineligible for preservation** — so an orphaned artefact is **excluded from
backup at exactly the moment it is least protected.** Another turns a lookup miss
into a positive claim that the output is *gone*. Six more silently shrink their
denominator (§90).

**And the marker lesson, learned by having a repair erased:**

> **A marker that lives in the replaced row cannot record that the row was
> replaced.**

The first repair wrote `row_repaired_utc` *into the row*, and the next failed run
replaced the row including the marker. The evidence of repair and the thing being
repaired were the same object. The replacement records **where the value was
carried from** and — the part the first attempt lacked — **a carry COUNT**, so
repeated orphaning is visible instead of every occurrence looking like a first.

## 96. "No build yet" and "you never pushed it" are the same string, and only one resolves by waiting

A deploy polled CI every 30 seconds for **fifteen minutes**, logging
`no run found yet` each time, for a commit that **existed only in the local
repository.** HEAD was one ahead of its upstream. CI was working perfectly and
had simply never been told the commit existed.

**The log was correct on every line and the wait was infinite.** This is the same
shape as the queue that reported "working tree is dirty" forever (§ the
deploy-scope work): **a service giving a true, sensible-sounding reason for doing
nothing.**

**The check is local and costs nothing**: `git rev-list --count @{upstream}..HEAD`.
If the builder has never seen the commit, say so and stop, rather than waiting for
an event that cannot occur. **Before waiting on an external system, establish that
it has been asked.**

**Compounding cause worth naming**: the same session had earlier starved this
deploy by committing four times while it waited (each reset the target), then
stopped committing to protect it — and the last of those commits was never
pushed. **The discipline adopted to fix the first failure created the second.**

## 97. An aggregate over an empty list returns its identity, and that reads as a clean result

Two defects of one shape, both written tonight by the agent who had spent the
night catching that shape in other people's instruments.

**A fail-soft diagnostic swallowed its own `NameError`.** The recorder referenced
`datetime` without importing it; its own `except` would have caught it, written
nothing, and produced an empty file — **indistinguishable from a merge that never
ran.** Caught by a test asserting it *fires*. **A fail-soft diagnostic whose
silence is unfalsifiable is not a diagnostic.**

**A verification computed a maximum over an empty list.**

    max([container - track for tracks WITH a tag] + [0.0])

A source carrying **no** duration tags contributes no terms, so the expression
returns the `0.0` sentinel — **the same value a genuinely clean source returns.**
Two files were reported *verified* on which nothing had been measured. One was
rescued by an independent reference; the other was not, and a case counted as
established had to be withdrawn.

**The shape, in both**: *absence of measurement is indistinguishable from a clean
measurement.* This is §48 — a reading that cannot distinguish two decisive states
— arriving in the instrument rather than in the data.

**The fix is structural, not arithmetic.** An aggregate over a possibly-empty
input must return **"no data"** as a distinct value, not a neutral element that
happens to be in range. `max(xs + [0.0])` and `sum(xs)` and `all(xs)` all answer
confidently about nothing. **Count the terms and say so when the count is zero.**

**And the corollary for a self-check**: the question "did my check have anything
to work with?" is a *different* question from "did my check pass", and only the
second one gets asked by default.

## 98. `batch_verify one` re-merges, and both of us authorised it without reading it

The subcommand is named `one`, sits in a tool called `batch_verify`, and was
described in conversation as "re-run Step 2 in a process that loads
`[software]`". **It POSTs to `/fusion` first.** `process_one` merges, then
verifies.

**The Lead authorised it as "re-running an existing tool whose defect is already
repaired — go", and the agent ran it. Neither read what the command did.** A
verb in a command name is a claim by its author, not a contract, and *"verify"*
is exactly the kind of verb that reads as read-only.

**It cost evidence.** One re-merge collided with a concurrent job, took an HTTP
409, and the ledger row was overwritten with `stage: enqueue_failed` —
**destroying the `tools.software['ffmpeg'] is not configured` note that had put
that file in the set of 36.** The finding survived only because it had already
been written to a committed report. **The row that demonstrated it is gone, and
the run that destroyed it was authorised as a read.**

**The replacement is the shape to copy**: call the verdict function directly on
the artefact already on disk — no merge, no queue, and **no write to
`batch_state`**, which has exactly one writer. All three properties asserted
**over the AST, not the source text**: the first attempt grepped for `/fusion`
and matched the module's own docstring, **a test that cannot tell an instruction
from a description of one.**

**Two rules follow.** *Before authorising a re-run, read what the command does —
the name is not the contract.* And *a diagnostic that writes to the ledger is not
a diagnostic*; it is a second writer to a store whose single-writer property is
the only thing making its rows trustworthy.

## 99. A column that fits every case by construction is the hypothesis restated

The truncation mechanism came with what looked like strong support: an "implied
delay" computed for each affected file, matching every observed shortfall.

**It was the formula rearranged.** The mechanism says `limit = duration - delay`;
the implied delay was computed as `delay = duration - limit`. **Given the
formula, every observed shortfall implies exactly the delay that reproduces it —
for every file, always, whatever the data.** The fit could not have failed, so
its success carried no information.

**The boundary case fell to the same objection.** An "empty" track was offered as
decisive because its shortfall equals its full duration, so the formula yields a
limit of zero. That is the same restatement at an extreme value, not an
independent prediction.

**The measurement settled it.** Three numbers were put on record as falsifiable.
**Two were restatements of one observation; only the third was independent — and
it is the one that matched.** The predicted delay was 1000463 ms; the observed
delay was **0.0**.

**The test for this**: ask what data would make the column *disagree*. If no
possible measurement could, it is not evidence. **A hypothesis is supported by
what it forbids, not by what it accommodates** — and a fit computed through the
hypothesis accommodates everything.

**The instrument is not the mechanism.** What the run did establish is that the
recorder fires at the right call site on the right file — the one independent
figure matched to the millisecond — and that the arithmetic holds. **Neither is
evidence for the causal claim it was built to test.**

## 100. A lookup miss that agrees with your prior is worse than one that reads as clean

§97 recorded an absence of measurement reading as a clean measurement. This is
the sharper form: **a lookup that misses and returns a value confirming what you
already believed.**

The claim under test was that a status of *Resolved* rests on Step 1 alone. The
check read `ext[path]["verdict"]` and got `None` for both files — **consistent
with "the external verdict has no opinion", which is precisely what the wrong
claim predicted.** The check ran, returned cleanly, and agreed.

**The real key is nested one level down**: `ext[path]["result"]["verdict"]`, and
it reads `in_sync`. `build()` consults it **before** Step 1, so those files are
**Step 2 verified with Step 1 silent — the exact opposite of the claim.**

**What caught it was reading the line in `build()` rather than trusting the
probe.** The production code says `.get("result")`; the probe did not. A probe
written from memory of a structure will miss the same way twice, and its miss
will look like data.

**The asymmetry that makes this dangerous**: a lookup miss that *contradicts*
your prior gets investigated. One that *agrees* gets written up. **The direction
of a bug's error decides whether anyone looks for it.**

**Practical rule**: when a probe confirms your own claim, verify the probe reads
the same path the production code reads — from the source, not from memory.
**Confirmation is when to check the instrument, not when to stop.**

## 101. "This artefact is broken" and "the pipeline breaks artefacts" are different claims

Five truncations were established by measuring **existing outputs**. The deployed
code did not necessarily produce them.

When a re-run of the worst case reported a delay of **0.0** where the mechanism
required a large one, the possibility opened that **the defect is already fixed
on the current build** — and every finding split in two:

- **the artefacts on disk carry short tracks** — established, from measurement
- **the current pipeline produces short tracks** — *not* established by that same
  measurement

**A campaign that re-verifies old outputs answers only the first.** Deciding what
to fix requires the second, and the two are separated by *when the artefact was
built*, which no verdict in the ledger records.

**So a result must say which claim it supports.** "Verified unresolved" against an
artefact built by an unknown earlier build is a statement about history, not about
the code anyone is about to change.

## 102. A recorder on a computed value cannot testify about the command that was issued

The delay recorder logged `delay_to_put`, `duration_best_video` and the derived
first-pass limit — **values the code computes.** The re-merge showed a limit of
1441.942 s while six tracks came out at ~441.47 s, so the instrumented line was
exonerated and the recorder **could say nothing about anywhere else.**

**Two gaps, and the second is the one that cost the run.** The value recorded was
not the argv issued, so a different `-t` reaching ffmpeg by another route was
invisible. And **the pipeline runs TWO ffmpeg stages; only one was instrumented**
— so a recorder placed on the stage already under suspicion could only ever
confirm or clear that stage, never redirect the search.

**Instrument the interface, not the intention.** What a process *decides* and
what it *executes* are different facts, and only the second is what the tool
actually did. The `mkvmerge` shim got this right by capturing argv; the delay
recorder repeated the earlier mistake one level in.

**And instrument every instance of the stage, not the suspected one.** A
diagnostic covering one of two invocations answers only about the half you
already doubted — which is the same selection error as looking for a
discriminator in the group you selected (§68).

**A related discipline the replacement got right**: its tests assert the recorder
**writes**, not merely that it does not raise. The first recorder raised a
`NameError` its own `except` swallowed; a test asserting no exception would have
passed against a recorder that recorded nothing (§97).

## 103. A probe placed before the write reports honestly and measures nothing

The stage probe built to bisect the truncation was first placed **above** the
ffmpeg launch it was meant to observe. It would have returned `probe_failed` —
truthfully, since the file did not yet exist — and the trace would have carried a
stage that looked measured and was not. **The instrument built to catch that
family committed it inside itself.**

**Reading the diff did not catch it. A test asserting the probe follows the launch
did.** Ordering is not visible in a diff the way a wrong value is; it needs an
assertion about *sequence*, and running it instead would have cost a merge to
discover.

**Two more from the same build, both structural:**

**A one-attribute wrapper class grew the protected closure 78 → 79 and the gate
refused.** A class's `__init__` is a closure member; **a string needs no class**.
The fix was to pass the path expression directly at every call site, pinned by a
test.

**One stage was reported as impossible to instrument. That was FALSE, and the
retraction is the more useful entry.** The PAL-and-step repair was said to be
inside the closure, with a source comment naming it as an uncoverable hole and a
test pinning the comment. It is not protected; the probe now covers it. **The
uncoverable set is exactly one stage — the mux — and it needs no cover.**

**The misread**: `closure_check` prints a repo-wide function diff, in which
*every* new function appears as `added`, alongside the protected-closure verdict.
The first was read as the second.

**But the author's own experiment had already disproved it**, two commands
earlier and in their own output:

    PAL probe + chimeric probe + helper class -> 79  VIOLATION
    removed the PAL probe                     -> 79  STILL VIOLATION
    removed the helper class                  -> 78  intact

**Step two settles it outright**, and the claim was written anyway. Not a tool
misread — **conclusion first, evidence read as confirming it, with the
contradicting evidence being one's own** (§99).

**A false note is worse than no note**: it would have stopped the next person
instrumenting a stage that was always available. The test was **inverted rather
than deleted**, so it now asserts the note is gone and the stage is covered.

## 104. Safe-sounding claims go unchallenged, and that is where wrong ones survive

Across one night's work, every claim that sounded alarming was checked within
minutes — by the claimant, by a peer, or by the Lead. **"I cannot instrument that
stage, it is protected" stood for two commits and was false.**

**A claim of a limitation asks nothing of anyone.** It closes a line of enquiry
politely, it implies diligence, and it invites no test — so no one runs the two
commands that would refute it. The claims that got caught fastest were the ones
that sounded like *over*-reach.

**And what caught it was reproducing the change, not reading the listing.**
Reading would have worked too, but the tester **checked the claim in the form it
was made** rather than in a convenient form — which is the difference between
confirming a reading and testing an assertion.

**Practical rule**: when someone reports that something *cannot* be done, treat it
with the scepticism reserved for a strong positive result. **A boundary is a
finding, and a finding needs evidence.**

**Extended the same day, from the other direction.** A *pessimistic* claim is
equally unchallenged: "at most a third of these rows can be attributed to a
build, and probably none of this set" stood untested until a peer's question
forced the measurement. **The true figure was 32 of 34.** The claim had been
carried over from a different population and never re-measured.

**So the exempt form is not the reassuring one and not the alarming one — it is
whichever form nobody has to act on.** A limitation closes enquiry; a pessimistic
estimate lowers expectations. Neither creates work, so neither gets checked. **The
claims that got tested fastest all implied someone should do something.**

**And a withdrawal is a claim.** One drift finding was retracted on the argument
that a 9.8 ms spread was "one envelope bin, the resolution limit". That argument
was wrong — the correlator interpolates sub-bin by parabola, so **10 ms is the bin
width, not the resolution**, and the non-round lags were the visible sign of it.
A second agent then applied the same wrong argument to kill a hypothesis of their
own. **Neither finding is restored** — the sub-bin test only proves the parabola
recovers a *synthetic* parabola's vertex, so no bound on interpolated accuracy for
real peaks is published — but both return to **open with unquantified
uncertainty** rather than closed. **A retraction needs the same scrutiny as the
finding it removes, and it propagates faster because it asks nothing of anyone.**

**The mux needs no probe for the opposite reason**: its write site is the
closure's own root, and its output *is* the published file — **measurable from
outside the pipeline.** An interval closes by measuring both ends; only the ends
that are invisible need instruments.

## 105. Evidence that a guard works suppresses the question of when it does not

The self-comparison guard in `batch_verify` reads:

    if all(abs(l) < 1e-9 and p >= 0.999 for _t, l, p in lags):

**`all`, not `any`.** It fires only when *every* usable window carries the
signature, so a file with **two of three windows at exactly 0.0 lag and peak
1.0000** — the same evidence that correctly declined three other files — ships as
`in_sync`. **One imperfect window silences it.**

**Nobody asked what it let through, because it had visibly worked.** Three
correct declines were the reason the question never arose. That is the
counterpart to a withdrawal getting less scrutiny than a finding (§104): **a
mechanism seen working is not examined for the cases it misses.**

**The strictness is set by the audio, not by the rule.** `lags` contains only
windows clearing a 0.3 peak floor, so its length is 1, 2 or 3 *per file*, and
`all` quantifies over whatever survived. Three usable windows with two matching →
silent. **One usable window matching → fires on a single observation.** A noisier
file gets a *more sensitive* guard, and any majority rule inherits this —
**2-of-3, 2-of-2 and 2-of-1 are not the same rule.**

**`any` is the obvious fix and is not obviously right.** One window of a genuinely
distinct track reading 0.0 at 0.999 would decline a good merge, and the profile
judge exists because single bad windows had already sunk two good merges.
**2-of-3 cannot be chosen from five cases** — that is placing a boundary through a
population instead of in a measured gap (§88). Choosing needs the per-window
agreement distribution, **which nobody has measured.**

**Scope, which changes the remedy rather than shrinking it**: nothing on the
production merge path imports this module. A false `in_sync` does **not** ship
broken audio — **it writes a wrong verdict into the audit record**, and every
decision resting on that record inherits it. *Our measuring instrument is
defective, not the thing being measured.*

**And the guard's own message overstates its evidence**: it says the track is
**"bit-identical"** to the master's, when `peak` is an **envelope correlation**.
Two agents made and withdrew exactly that overstatement hours earlier; **it was
sitting in the source the whole time**, and was seen only because they had just
committed it themselves.

## 106. A complete log read incompletely looks exactly like coverage

The second-pass argv was captured in full — 482 arguments — and searched for every
duration-limiting flag. The report said *"a sweep for every duration-limiting flag
returns `-t` and nothing else."* **True, complete for the question asked, and
blind to the one that mattered.**

Two runs later the same log yielded the finding: **all seven audio streams
re-encoded to FLAC at compression level 12 under a global `-c copy`.** The second
pass is not a copy; it is seven simultaneous slow encodes. **Those options were
present the first time and were not read, because the hypothesis was about
durations.**

**This is the selection error of §102 one level along.** There, a recorder covered
only the suspected stage. Here, a *complete* record was interrogated only for the
suspected property — and a search that returns a clean negative reads as coverage
of the artefact, when it is coverage of the question.

**The distinguishing habit**: when a log is captured in full, **read it once for
what is there before searching it for what you expect.** A grep answers about your
hypothesis; the log is about the process.

**And the control was in the same data.** The master chain's second pass ran in
~1 s with 32 arguments; the chimeric chain's took 114 s with 88. **One second is a
copy and 114 is an encode** — a difference visible in the first capture, unread
until the second.

## 107. A control pinned at a floor cannot fail, so its agreement means nothing

A detector was to be validated by finding a file where the correction leaves
fidelity **worse than no correction at all**. **That test cannot fire.**

*No correction* means comparing two signals across a 4.27% speed difference: over
a 240 s window they drift **10.3 s apart — 82 Chromaprint frames** — so they
diverge across the window and **no single alignment shift exists.** The
uncorrected reading is therefore pinned near the **0.5 floor by geometry**, and
against a floor **almost any correction is an improvement.**

**The run would have reported "the inverting case is not present in this corpus"
and could not have reported anything else.** True by construction, and a statement
about the instrument rather than about the corpus — the same shape as a query
whose filter guarantees its own answer (§30).

**The test for it is one question**: *what measurement would make this control
disagree?* If the geometry forbids one, the control is decorative.

**The fix was to compare the two arms that are both alignable** — each restores
duration, so each can be shifted into agreement — rather than either against an
unalignable baseline. **The unalignable arm is kept as a diagnostic so the floor
is shown rather than asserted.**

**And a detector's asymmetry must be stated with it.** In this one the correct
tool for the sought-after case carries a processing handicap **even when it is
right**, eating half its margin: **twice as sensitive to the expected answer as to
the one it exists to find.** That is not a reason to discard it — it is a reason
the threshold cannot be zero, and that a band must return **UNDECIDED** and fall
through to the default rather than pick a winner out of noise.

## 108. A sign error produces a consistent, plausible, outlier-free refutation

A corpus sweep compared two speed-correction methods across 33 files. **Rubberband
won 33 of 33**, with a consistent sign, no outliers and plausible magnitudes — at
face value a refutation of the owner's ruling across the whole corpus.

**One arm was corrected and the other was made twice as wrong.** With
`r = master/candidate = 1.0427`, the candidate runs *fast* and must be *slowed*:
`asetrate = 48000 / r`. The sweep applied `48000 * r`. The `rubberband` arm
carried the correct direction.

**Nothing internal to the run could have caught it.** There was no outlier to
notice; a uniform sign error moves every file the same way, so **the result looks
more trustworthy the more thoroughly the bug applies.** Internal consistency is
evidence of a systematic process, not of a correct one.

**What caught it was an accidental overlap**: three of the thirty-three belonged
to a set another agent had verified by a different route, where the opposite arm
read `same_recording` on 16 of 16 tracks. **The table disagreed with a result its
author had not produced.** Had the sets not overlapped there was no check at all.

**Two practices follow.**

**Carry the wrong direction as its own column.** Re-run with correct-direction,
wrong-direction and uncorrected arms side by side; a sign error then **reads as
the floor rather than as a finding**, and the bug is isolated instead of merely
fixed.

**And a projection is not a measurement.** The same work projected a separation of
0.420 from a curve; measured against real masters it is **0.278–0.313**, because
the curve assumed a perfect correction reaches 1.000 while a real pair reaches
0.94–0.98 — **content and codec differences no correction removes.** A threshold
taken from the projection would have been ~35% too wide.

## 109. Back-filling a provenance field manufactures the thing it exists to record

28 verdict rows carry no timestamp, so they cannot be attributed to any build
(§101). The obvious repair is to date them from the artefact's mtime.

**That would manufacture the provenance the field exists to record.** The writer
now stamps a timestamp on **every** write, so those 28 are **necessarily
historical** — written before the stamp existed. An mtime is when a file was last
touched, not when a verdict was formed, and filling one in produces a row that
*looks* attributable and is not. **Every downstream consumer would then trust it
exactly as much as a real one.**

**Report them as permanently unattributable.** A field that is honestly empty
supports the conclusion *"this cannot be dated"*; a field filled from a proxy
supports a false one, and nothing downstream can tell the two apart.

**The general form**: a gap in a provenance record is **evidence about the
record's history** and is often the only such evidence left. Filling it destroys
that evidence and replaces it with a plausible fabrication. **Absent is a value.
Keep it.**

## 110. The datum discarded as an outlier was the one that refuted the conclusion

A delay column looked like a clean straight line with one stray value, which was
dropped and a slope fitted through the rest. **The stray value was the evidence.**

**Worked from source constants**: a 178 s clip yields ~1437 frames at a truncated
scale of 123 ms, with `min_overlap = 32`, giving a span of 1405 frames and a
**maximum expressible delay of 172815 ms.** Every uncorrected reading sat within
**3.5% of that ceiling, on both signs.**

**So the column was not a delay measurement with a large error.** It was the
correlator answering from the edge of its range because no interior offset scored
better, and **its values are a property of the search span, clip length and frame
count — carrying no information about the pair at all.**

**And the same arithmetic applied to the original 60 s geometry gives a ceiling of
55596 ms. The value discarded as an outlier was +55341 — 99.5% of it.** The
saturation was present in the very first measurement. **The single datum
contradicting the conclusion was removed as noise on the grounds that it did not
fit the conclusion.**

**The check**: before treating a reading as a measurement, compute what the
instrument's maximum expressible value is. **A number near the ceiling is a report
that the search failed, not a large answer.** §15.

## 111. Two agents citing each other is indistinguishable from corroboration

A claim about which window the merge path uses stood for hours across two
sessions. **One agent cited it in a committed report without opening the
function; the other had also never read it.** Each relied on the other having
done the lookup. **A citation loop with no lookup at either end is stable, and it
feels exactly like agreement.**

**Every defect caught tonight was caught by going OUTSIDE the pair**: a 16-of-16
result against a backwards sweep; a geometry check against an endorsement; a
measured null band against an unchecked window. **The loop is the state where
nobody goes outside.**

**And the two failure modes are not equally curable.** In the same episode one
agent skipped a lookup, and separately preferred an inference:

> **A skipped lookup is fixed by looking. A preferred inference survives looking,
> because the looking gets aimed at whatever the inference predicts.**

The window error was corrected by one command. **The shape story required a
geometry the author had not chosen** — evidence they were not in a position to
aim.

**Practical form**: when two sources agree, establish that at least one of them
went to the primary. **Agreement between two derivations of the same unchecked
claim is one claim, counted twice.**

## 112. A point sample cannot see a duty-cycled process, and the Lead's own hold made a state file unreadable

Two Lead errors in one hour, both diagnosed by the agents they were aimed at.

**The idle check sampled `pgrep ffmpeg` once and reported "no decoders running".**
The sweep it missed was decoding a window, correlating in Python, then decoding
the next — **a sequential loop that shows no ffmpeg for a large fraction of every
cycle.** A single sample cannot distinguish *idle* from *between decodes*. **Two
samples 30 s apart, or checking for the PARENT process rather than the decoder,
would both have caught it.**

**Worse, the point sample AGREED with the agent's own state file** — which the
agent had written earlier and had no reason to update. **Two sources with one
origin, which is exactly what makes a wrong claim durable** (§111).

**And the stale `status: running` that was twice diagnosed as an agent's defect
was the Lead's hold.** Agents were told to land nothing until an image was up.
The agent had already set its state to `idle` and committed it — **on its own
branch, where the Lead could not see it.** The dev-AI copy stayed stale *because
of the instruction*, and the Lead then read that copy and reported the agent's
writer as broken. **Twice.**

**The general shape: an instruction that stops an artefact being updated makes
that artefact unreadable as evidence.** Before diagnosing a stale record, ask
whether anything you did is the reason it is stale. **A hold has a blast radius,
and the state files are inside it.**

## 113. Calibrate on IDENTITY before injected offsets, and refuse when your noise exceeds the effect

An independent replication of the audio-versus-video offset returned **neither a
confirmation nor a refutation, and reporting that was the right answer.**

**The off-the-shelf aligner was disqualified by the cheapest possible test.**
ffmpeg's MPEG-7 `signature` filter reports **"no matching of video 0 and 1" for a
file compared against itself** — both detect modes, including the same path
passed twice. **And it had already "confirmed" two injected offsets before
identity was tried.** An instrument that finds an offset you inserted but cannot
recognise a file as itself has not been calibrated; **injected-offset tests can
pass on an instrument that is fundamentally broken.** Identity is the cheaper
test and it belongs first.

**The replacement was calibrated before use** — luminance cross-correlation
returning +0.000, +0.500 and −0.800 for those injections, and GCC-PHAT on the
audio — **and the two conventions turned out to be OPPOSITE**, confirmed
empirically rather than read. Getting that wrong inverts a result while it still
looks clean.

**Then the honest refusal.** On the chosen pair the video aligned strongly
(r 0.74–0.80) and the audio did not (r 0.13–0.21), with **three independent
refusals**: both chromaprint delays sat **on the computed ceiling** (§110), both
fidelities at or below the unrelated-pairs band, and GCC-PHAT peak-to-sidelobe at
its floor of 1.0 against a calibration range of 600–7900.

**And the measurement's own instability exceeded the effect**: the video figure
moved **79 ms** between resolutions and the two audio tracks disagreed by
**154 ms**, against an effect of **65 ms**. *"Any figure I reported would be noise
dressed as a replication."*

**The distinction that matters more than the number**: an audio delay that will
not correlate is **not a wrong answer, it is NO answer** — and the premise being
tested assumes there is one to snap to a frame. **That is a different problem from
a systematic bias, and possibly a larger one.**

**The step that belonged first**: screen for a pair whose audio correlates *well*
— fidelity comfortably above the null band, peak-to-sidelobe in the hundreds —
**before measuring anything.** Cheap, and it decides whether the pair can answer
the question at all.

## 114. The justification did the concealing

A threshold was written as `PSR_BAR = 10.0` with a docstring paragraph explaining
why it was principled. The distribution it split runs **continuously from 0.8 to
38.1 with a largest gap of 1.2×**, and the bar puts 9.9 and 9.9 on one side, 10.0
and 10.1 on the other. **It splits the population at its densest point.**

Its author's own account: *"the paragraph is what stopped me looking at the
distribution."*

> **Calling a bar unfitted is the moment you stop testing whether it is.**

**An unexplained constant invites the question. An explained one closes it** —
and the explanation is written before the data is plotted, by the person least
able to see what they have assumed. **The reasoning becomes the evidence that no
reasoning is needed.**

**The check costs one command: plot the sorted values and look for the gap you
claimed.** If the largest gap is not at the bar, the bar is fitted to the
population regardless of the prose beside it (§88).

**Fifth threshold unfitted on this project. Four of the five were caught by the
person who wrote them** — which is the encouraging half, and the reason to treat a
confident docstring as a place to check rather than a place to stop.

## 115. I read the warning, edited the line beneath it, and asserted the opposite an hour later

The Lead told an agent *"you have production's per-cut delays in the refusal
text."* **They do not.** `batch_verify` caps the stored exception at **220
characters**, so each of the 95 records holds roughly **four tuples of the first
audio couple** out of ten cuts. **All 95 have unbalanced braces — truncated,
detectably, by inspection.**

**And the truncation is documented in `master_summary.py` immediately above the
`KNOWN` table** — *"the stored exception is truncated at 220 chars, so match the
head of the message, not its tail"* — **which the Lead read while adding an entry
to that very table an hour earlier.** Reading a warning is not the same as
carrying it, and neither is editing the code it sits above.

**The cost lands on someone else's earlier result.** A truncation-impact report
had cited *"384 tuples across 95 files"* and *"0 of 19 two-delay files cross the
bar"*. That empirical leg was computed on truncated records and is **withdrawn as
an independent confirmation** — the "19 two-delay files" is an artefact of seeing
only the first four cuts. **The conclusion survives on its structural leg, which
needs no tuples**: one scale factor preserves equality and ratios, so
`len(set_delay)` is invariant and a gap must already lie within 0.6 ms of the bar
to cross it.

**The author's own diagnosis is the transferable part**: they had already written
*"Step 1 delays for Resolved files are not recorded anywhere"* — **and did not
extend the same suspicion to the refused ones, because those had a record.** *A
record that exists is not thereby a record that is complete.*

**The replacement is better than the source would have been**: regenerate the
per-cut delays **at production's own geometry** rather than read what production
happened to store. **A derived artefact is a summary of a computation; the
computation can be re-run.**

## 116. A refusal that has been right twice is the least examined thing in the room

An agent refused the same dispatch three times. **The first two refusals held —
one saved a rewrite. The third rested on a premise that was false and testable in
four minutes.**

The premise was *"I cannot construct the object until the owner's new field
exists."* Its own diagnosis: **"I conflated *the object must inherit the field*
with *the field must exist before I build*. Those are opposites."** Constructing
through the real constructor is **exactly what makes code survive a change to that
constructor**; it is a hand-built stub that would need revising.

**And the reason it went unexamined is that it had worked.** A refusal that has
been vindicated twice **acquires the immunity §104 gives to a claim of a
limitation** — it asks nothing of anyone, it reads as diligence, and its own
track record supplies the justification that stops the check. **The two good
refusals are what made the third comfortable.**

**The lead's part is the transferable half.** The correction was not *"you are
wrong"* but **"state which of these three claims you mean"** — one of them
unfalsifiable-sounding, one narrow and true, one testable in minutes. **Naming
the alternatives is what produced the test**; the agent's own account is that a
flat contradiction would have produced a restatement instead.

**Practical form**: when someone declines twice and declines again, **ask what
would have to be true for the refusal to be wrong.** A refusal is a claim about
the world and carries the same burden as a finding — and the better its record,
the less anyone looks.

## 117. A claim echoed back by the Lead returns with authority it never had

An agent registered a whole test design on the premise that four files sat
"within 0.0003 of **inverse** PAL". They sit within 0.0003 of **forward** PAL,
8.7% from inverse. **One duration probe refutes it.**

**The trace matters more than the error.** The claim originated in **that same
agent's own list**, sent to the Lead as an item they proposed to work on,
attributed to a third agent who — checked — **had never written it.** The Lead
echoed it back verbatim in a dispatch. The agent then acted on it **as a
dispatch**, not as their own unverified note.

> **A claim that leaves an agent as a suggestion and returns as an instruction
> has gained authority without gaining evidence.**

Nobody lied and nobody was careless in isolation: the originator did not re-check
their own note once it came back stamped, and the Lead did not check a premise
they had received from the agent they were sending it to. **The loop is closed
and it feels like corroboration** — §111 with the Lead as the second node.

**The practical rule cuts both ways.** *Before echoing an agent's claim back to
them as a dispatch, check it — you are about to make it authoritative.* And *a
dispatch quoting your own earlier words is not external confirmation of them.*

**And the shape underneath it is the one that recurred three times in one day**:
**a true observation, an inference drawn from it, and no lookup.** A saturated
column read as a slope. A field discarded, then reasoning about what the
remaining fields implied. A genuinely absent track, and a wrong conclusion about
what had measured it. **The fact was correct every time; the step from fact to
conclusion went unchecked, and in each case one grep or one probe would have
caught it inside a minute.**

## 118. `pgrep` is namespaced and `/proc/loadavg` is not — the process view cannot see the heaviest consumer

**Measured here:**

    /proc/loadavg     61 running / 4466 tasks     load 66.14 on 17 cores
    ps -e             53 processes
    /proc/[0-9]*      53

**Four thousand tasks are invisible from inside this container**, and the blind
spot includes **the merge container — the heaviest ffmpeg consumer on the box.**

**So no claim of the form "we are at the 2-ffmpeg budget" can be established from
here.** `AGENT.MD` line 631 says to judge by **load average and free memory, NOT
process count**, and this is why: the count is taken through a namespace that
excludes the thing being budgeted, while the load average is host-wide and honest.

**Four wrong statements today came from this instrument**, all made by the Lead:
*"no ffmpeg, no ffprobe, no python — the box is idle"* three times, and once as
evidence that agents were reporting progress on work that was not running. One of
those was additionally narrowed by a `grep -v claude` that removed every agent
process, since agents run from scratchpads under a path containing that string —
**a filter aimed at the observer's own noise silencing the signal.**

**And an agent used the same blind instrument to argue for keeping its own job
alive** — quoting the tie-break clause of line 631 while ignoring the measurement
clause four words earlier in the same sentence. They caught it themselves and
reported it as undercutting their own position.

**The rule**: *for anything about whole-box resources, read `/proc/loadavg` and
free memory.* **A process list here is a statement about this container, and
saying "the box" of it is a category error.**

## 119. `max()` on a plateau returns its edge, and the bias is the plateau's half-width

A keyframe-alignment method scored candidate offsets and took the best-scoring
one. **The score function has a plateau** — every offset within the match
tolerance scores identically — **so `max()` returned the plateau's LEFT EDGE.**

On a synthetic pair shifted exactly 2.000 s the naive peak gave **−2.020**. The
plateau centre gives **−2.000**.

**A 20 ms systematic bias, in a method proposed to check half-frame alignment,
where half a frame is 20.85 ms.** The error would have been the same size as the
quantity being measured — **and it would have shipped as a finding about the
files.**

**The fix reports the plateau WIDTH as well as its centre**, because the width is
the method's real resolution: a single returned number implies a precision the
score function does not have.

**The general shape**: *an argmax over a tolerance-based score is an argmax over
an interval.* Any matching rule with a window — "within 0.15 s", "within one
frame" — produces ties by construction, and **whichever way the language breaks
ties becomes a systematic offset of up to the window's half-width.** It is
invisible in testing against a zero offset, because at zero the plateau is
symmetric about the truth.

**Caught by injecting a known non-zero shift.** A control at zero would have
passed.

**And a companion from the same build**: the interval statistic returns **`None`
below four keyframes rather than `0.0`** — two keyframes give one interval and a
coefficient of variation of exactly zero, which reads as *"perfectly regular,
fixed GOP"* on no evidence. **The same shape as a spread of 0.00 across one
window** (§113).

## 120. A false dilemma survives review because both horns are individually true

Agent 3 finished the exception-cap fix and presented the Lead with a choice:
**restart the sweep**, paying the 2h14m already spent on E_corr plus B_fps's
progress, **or let ~278 files run through the defect it had just fixed**. It
argued both sides fairly and said a decision either way was defensible.

**Neither was necessary.** `full_test.sh` line 24 is

    for cls in $ORDER; do
      timeout 86000 python3 batch_verify.py run "$cls"

**Each class is a separate `python3` invocation**, so it reads `batch_verify.py`
from disk when it starts. Landing the fix mid-run reaches **every class that has
not started yet** — D_trim (58), A_no_lang (6), C_cut (179) and the three
failed-only passes. **243 of 278 files at no cost.** Only the running B_fps,
whose interpreter already holds the old code in memory, was missed.

**The trap is that checking each option only confirms it.** The restart cost was
real and correctly computed. The diagnostic loss was real and correctly
computed. **The false part was the word "or"** — and no amount of scrutiny
applied to either horn touches it, because the claim under-examined is the one
never stated: *that the list is complete*.

**Agent 3 had already read `full_test.sh` that same day** — that is where the
class order and the 15–20 hour estimate came from — and did not connect *"a
separate invocation per class"* to *"picks up a changed file"*. **The fact was
in hand; the question was never asked of it.**

**What broke it was not the analysis, it was one sentence about the record.**
Agent 3 wrote: *"what is not defensible is leaving it recorded as 'fixed' while
278 files run through the old path."* The Lead went to the script because of
that sentence, not because of the two options. **Naming what would be false in
the register outlives arguing which option is cheaper** (cf. §114, where the
justification did the concealing).

**Rule: when an agent offers the Lead exactly two costly options, the first
question is what makes the list exhaustive, not which option wins.** Two-option
framings in this project have been wrong about the frame more often than about
the costs.

## 121. A score over a tolerance window has two properties that look measured and are not

Sibling to §119, and the two should be read together. A keyframe-alignment score
sweeps candidate offsets and accepts a match when `|dt| <= tol_s`. **Two numbers
fall out of that construction and neither is a measurement of the files:**

**Where the maximum sits** (§119): `max()` over a plateau returns its left edge.
A 20 ms bias, invisible at zero offset because the plateau is symmetric about
the truth. **Found by injecting a known non-zero shift.**

**How wide the plateau is** (this entry): it is `2 * tol` by construction —

    tol_s     0.002   0.005   0.010   0.020   0.040
    plateau   0.002   0.008   0.020   0.038   0.080    <- exactly 2*tol
    offset    0.0000  0.0000  0.0000  0.0000  0.0000   <- never moves
    hits         218     218     218     218     218   <- never moves

So a corpus summary of plateau widths summarises **the tolerance that was passed
in**, not the corpus. **Found by sweeping the parameter.** Neither defect is
visible against a single default value — the number is stable, plausible, and
reported per-pair, which is what makes it convincing.

**The Lead asked for plateau width per pair, believing width was the
instrument's resolution.** It is not. **The premise came from the Lead and the
agent built on it in good faith**, so the error travelled with the authority of
a dispatch (cf. §117).

**The real resolution is the tightest tolerance that still preserves the peak** —
a property of the files, not the parameter. Measured on 20 pairs: **median
0.0005 s, max 0.0020 s, all 20 tighter than half a frame (0.0209 s)**; nine sat
at 0.0005 s matching every keyframe.

**The corrected conclusion inverts the original.** "Median plateau 30 ms, only 8
of 88 under half a frame" reads as *the instrument is too coarse for the bar*.
The truth is **40x tighter than the bar**. **Its limitation is coverage — 88
askable of 313, 62 checkable — not resolution.**

**Rule: before summarising any statistic defined against a tolerance, sweep the
tolerance. If the statistic tracks the parameter, it is a restatement of the
parameter.**

## 122. A sample from a class nobody has shown to be homogeneous measures the sample

"B_fps is 6 of 6 scatter" was reported as a property of the class and read as a
puzzle: **a pure rate problem should drift monotonically, not scatter.** Hours
of attention went to explaining it.

**B_fps holds 35 files and four distinct behaviours:**

    genuine rate relations   21
    content differences       8
    unstructured              5
    no relation at all        1

**Only 21 of 35 are frame-rate relations at all.** So the answer to "why does it
scatter" is: **fourteen of them are not the thing the question assumes.** If the
sampled six came from the content or unstructured groups, **scatter is the
correct result and the puzzle never existed.**

**The failure is not the small sample.** Six would have been informative from a
homogeneous class. **It is that the class was named after its label
(`B_fps`) and reasoned about as though the label were a mechanism.** Nobody had
checked whether the declared rate even predicts the observed behaviour -- and
for **8 of 35 it does not, because the declared rate is wrong** (FINDINGS 20).

**"Six cannot settle this -- but not because six is too few. Because the class
is heterogeneous and a sample of six means nothing until you know its
subgroup."**

**Rule: before generalising from a sample of a named class, show the class is
homogeneous in the property being measured.** A directory name, an error code
or a pipeline stage is a label, not a demonstration. **And when a sample looks
surprising, suspect the population before the mechanism.**

**Companion**: the same day, a related check found `fps_err` itself wrong on 8
files -- so even the label's own field was not the mechanism it named. **A class
can be heterogeneous because its defining metadata is incorrect**, not only
because it was drawn too widely.

**Confirmed with a single-file instance hours later.** err_110 sat in `B_fps`
with a stored `fps_mst` of 25.0. **Both sides are actually 24000/1001 =
23.976** -- there is no rate relation, and its two durations differ by 0.001 s.
**It was a member of the class only because a metadata field was wrong**, and
every question asked of the class had been silently asked of it too. The stored
record was checked against a fresh probe on 120 files: **119 agree, 1 stale**.
**So one bad row in a rarely-wrong field was enough to put a file in the wrong
class and keep it there.** Rarity of corruption is not a reason to skip the
check -- it is the reason the check is cheap.

## 123. A percentage detaches from its denominator the moment it is quoted

Agent 2 measured `get_bitrate` failures and reported them accurately at every
step: **"8 of 45 tracks across 16 real files"**, in the archive and in direct
messages. **The sentence that reached the Lead, and travelled onward, was
"18%".**

Agent 3 read 18% as a corpus rate and sampled 80 files expecting ~76
undecidable tracks. **It found 0 of 422.** It then hypothesised an instrument
difference (`StreamOrder`) to explain the gap. **The hypothesis was wrong and
there was no instrument difference at all** -- run in one process on the same
16 files, the two tools returned **37 and 8, identically.**

**The measurement was sound; the format was the risk.** A count carries its
population with it and a percentage does not, so a rate quoted without "of what"
**acquires a denominator from whoever reads it** -- and the reader's default is
"the corpus".

**The recovery was worth more than either number.** Two instruments agreeing
exactly while their rates differ completely can only mean **the condition is
concentrated, not uniform**. That turns the task from *estimate a rate* into
**locate the affected files -- a search problem, not a sampling one**, which is
a different specification than either figure alone would have produced.

**Rule: report counts with their population attached -- "8 of 45 tracks in 16
files", never "18%". If a percentage must be quoted, the denominator travels in
the same sentence.** And when a sample contradicts a published rate by an order
of magnitude, **suspect the population before the instrument** (cf. 122): here
the instrument check came second and was the one that settled it.

**Companion Lead defect, same exchange.** The Lead dispatched "add the missing
state to the record" to an agent. **Both target functions are protected-closure
members, and the scope document's own closing section said so.** The Lead had
sent the agent to that document without reading its last section. **The agent
declined and named the rule rather than building something adjacent and calling
it the task** -- which is the correct response to a peer dispatch that would
breach a standing constraint. **A dispatch is not an authorisation** (cf. 117).

## 124. A correct field can be the wrong input, and that is worse than a wrong one

Eight `B_fps` files declare `fps_err = 30.0`. Solving for the rate that would
explain their observed duration delta gives **25.0**, so the label was reported
as wrong and the files as mislabelled PAL. **The label is not wrong.**

    r_frame_rate = 30/1    avg_frame_rate = 30/1    on all eight

**The error was inferring a frame rate from durations and never probing the
stream.** One `ffprobe` settles it; it was not run until an unrelated question
forced it.

**Two independent facts were being collapsed into one:**

    PAL, master-relative        1 - 1/1.042708 = 4.096%
    observed for the group              4.10% - 4.17%
    a real 23.976 -> 30 speed change          20.08%

**The video is genuinely 30 fps AND the content carries a 4.1% PAL relation.**
The 30 fps conversion is **speed-neutral** -- it changed the frame count, not
the duration. **The audio says PAL, the video says 30 fps, and both are correct
about different things.**

**The consequence survives the correction and sharpens.** `fps_err` describes
**the encode**, not **the relation**. A speed correction keyed on 23.976 -> 30.0
applies **20.08% where the true relation is 4.10%** -- a five-fold
over-correction driven by an accurate field.

**Why this is the worse failure mode.** A wrong value can be caught by
validating the field: probe the stream, compare, flag the mismatch. **A right
value used to answer the wrong question passes every validation there is**,
because nothing about the metadata is suspect. **The check that catches it is
not "is this field correct" but "is this field the quantity my formula needs".**

**Rule: before a metadata field becomes a computation's input, state which
physical quantity it measures and confirm the formula needs that quantity.**
Frame rate, playback rate and content speed relation are three different things
that agree on unconverted files -- which is precisely why the confusion survives
until it meets a converted one.

**Companion**: the same investigation resolved an apparent anomaly for free.
err_688 was flagged as unexplained -- 10.0 s observed against 58.2 s predicted.
It is 30 fps like the rest, carrying the re-encode and **no** PAL relation.
**It looked anomalous only against the wrong model**, and correcting the model
dissolved it without a separate hunt.

## 125. A terminal `return` is not a verdict -- "none of the above" reported as a fact

`classify()` returned one of four labels. Three were reached by a test; the
fourth, `scatter`, was the **terminal return after every test failed**. It meant
**none of the above** and was reported, and reasoned about, as a property of the
files: *"B_fps is 6 of 6 scatter, and a rate problem should drift -- that is
unexplained."*

**A plateau-finder cannot report a drift as anything but `scatter`:**

    PAL drift                     42.71 ms per second
    over the 45 s stride         1922 ms per window
    tolerance, one frame            41.708 ms
    -> every window sits 46 frames past tolerance

Every window becomes its own group, nothing reaches `MIN_PLATEAU`, fallthrough.
**The classifier was structurally incapable of the answer, and its silence read
as an observation.**

**The tell was available and was not used.** A slope instrument measured
**monotonic drift** on four of the same files while the classifier said
**scatter** on six -- same pairs, same afternoon, and they **agreed on the
number** (25.0/23.976 = 1.042709 against a measured 1.0427087-1.0427106). **Two
instruments agreeing on the quantity and disagreeing on the word is a statement
about the vocabulary, not about the files**, and both owners took the other's
label as a fact.

**Cost asymmetry: a fallthrough cannot manufacture a positive.** `41 STEPPED`
requires two or more plateaus over 70% of readings and survives untouched, as
does everything computed from those step sizes. **Only the fallthrough bucket
falls** -- it holds noise, fast drift and slow wander together and cannot be
subdivided after the fact, because only plateau means were stored.

**Rules.** **(1)** Every classifier needs an explicit `unclassified` distinct
from its named verdicts, and a fallthrough must never share a name with a
finding. **(2)** Store the raw readings, not only the derived verdict -- a
verdict can then be revised without re-measuring. **(3)** When two instruments
agree on a quantity and disagree on its label, the label is wrong.

**Companion -- a binary that excluded the answer.** The Lead and an agent framed
the check as *"if the six are in group A it is a real puzzle; if in group B it
dissolves"*. **The six were in group A and it dissolved anyway.** Both branches
assumed `scatter` meant something about the files. **A well-posed either/or is
still worthless when both arms share a false premise** (cf. 120).

## 126. ENVIRONMENT: stale bytecode makes a killed mutant look alive

**This box sets `PYTHONPYCACHEPREFIX=/tmp/pycache`.** Bytecode does not live
beside the source, so **`find . -name __pycache__` reports nothing and deleting
local `__pycache__` clears nothing** -- and both read as confirmation that the
cache is clean.

Cached pycs are invalidated on **source mtime at one-second granularity**, and
`PYTHONDONTWRITEBYTECODE` stops **writing**, not **reading**. So **any
size-preserving edit reverted inside the same second is served stale bytecode.**

**During mutation testing this is one-directional and it points the wrong way.**
A mutant that was actually killed appears to **survive**; the conclusion drawn
is *"the test is weak"* and the response is to weaken or rewrite **a test that
was fine**. A line trace showed the interpreter executing `return "STEPPED"`
while the call returned `"monotonic"`.

**It was caught only because that particular mutant was one there was a specific
reason to expect to die.** A mutant with no prior expectation would have been
believed.

**Fix, in `run_tests.py`:** each run gets a **fresh `PYTHONPYCACHEPREFIX`
directory** -- a new directory, not `rm -rf` on a shared one, so a run cannot
pull the cache out from under a concurrent session. All 13 mutants re-run: **13
dead.**

**Rule: never conclude "the test is weak" from a surviving mutant without
confirming the interpreter ran the mutated source.** Cheapest confirmations: a
fresh cache prefix, a line trace, or an assertion that deliberately fails inside
the mutated branch. **Report at
`src/diagnostics/reports/stale_bytecode_defeats_mutation_testing.md`.**

**Scope: this affects every agent on this box**, not one tool -- mutation
testing, any timing-sensitive edit-and-rerun loop, and any conclusion of the
form "my change had no effect".

## 127. A screen on a derived quantity is blind to the relation that quantity masks

Eight files declare `23.976 -> 24.0`. The relation predicts **3.2 s** of
duration delta; the observed delta is **174-205 s**. A cv test on the delta --
raw 0.056 against per-duration 0.132 -- correctly showed the delta is **not**
proportional to duration, and the group was reported as **"content, not a
rate"**.

**Two of the eight carry a clean pulldown relation:**

    err_29   factor 1.0010009   resid 0.32 ms   10 windows
    err_31   factor 1.0010001   resid 0.96 ms   11 windows
    declared prediction                    1.001001

**0.00009% off. The relation was there the whole time; content exceeded it 58x
and hid it.**

**The test was clean and measured the wrong quantity.** It answered *"is the
delta proportional to duration"* when the question was *"is there a rate
relation"*. **A ratio screen cannot see a relation whose ratio is masked by a
larger additive term** -- and a screen that is sound, well-executed and
answering a neighbouring question is far more convincing than a broken one.

**The author had written this warning himself, that same day**, against a
different bound: *"a ratio screen is blind to a relation whose ratio has been
masked by content -- my six NTSC files carry 174-190 s against a relation
explaining 2.7-3.7 s, and that screen would miss every one."* **The 24.0 group
is 174-205 s against 3.2 s: the same numbers.** He quoted the warning to two
agents and then applied a duration-based test to the group in the corpus that
matches its description most closely. **A warning you wrote does not protect the
next case unless something checks the case against it** (cf. 115).

**The generalisation across three corrections to one findings section**, all by
the same author, all self-caught: **a derived quantity was read and reported as
the underlying one.** Durations for a frame rate (124); durations for a speed
relation (here); a signed difference against an unsigned field. **Each derived
value was computed correctly.**

**And the screen was not a near-miss.** A pre-flight on the six unmeasured files
established the stronger statement:

    err_26  predicted delta at 1.001   +3.8 s    (candidate faster -> SHORTER)
    err_26  observed delta            -175.6 s   (candidate LONGER)

**The sign is opposite and the magnitude is 46x.** The content difference does
not merely swamp the relation in the duration -- **it reverses the sign of the
quantity being read.** So **no tolerance on a duration-based screen could have
found this relation**: the test was not close, and tightening or loosening it
changes nothing. **A proxy dominated by an unrelated term is not a weak
instrument for the question; it is not an instrument for the question at all.**

**Rule: name the physical quantity the question is about, then confirm the
measured quantity IS that one -- not a proxy for it.** When a proxy can be
dominated by an unrelated term, **measure the thing directly or say it is
unmeasured**; do not report the proxy's answer as the question's.

**And measure before restating a group.** Only 2 of 8 were measured; the other
six need one slope each. **The group was left explicitly part-measured rather
than re-characterised wholesale** -- the discipline of 122, applied in the class
that produced 122.


## 125b. A refusal that still answers the naive read

Companion to 125, found by asking the same question of a second module. Not a
shared **name** this time -- a shared **payload**:

    out["refused"] = "peak sits at the search ceiling -- a failed search"
    return out          # offset_s, hits, hit_fraction ALL STILL PRESENT

**A result that had declared itself a failed search still answered
`.get("offset_s")` with a number.** The refusal was correct, explicit and
well-worded, and it sat in the same dict as the value it refused.

**The caller who reads the verdict is safe; the caller who reads the number is
not -- and the naive read is the one people write.** A refusal is only a refusal
if the thing being refused becomes unavailable.

**Fix**: the values move under a `refused_` prefix, so the naive read returns
`None` and nothing is discarded. A test pins it.

**The author's own analysis filtered on `refused` and was unaffected** -- which
is the point. **The API invited the mistake regardless of whether anyone had yet
made it**, and an interface that is safe only because its current callers happen
to be careful is not safe.

**Rule: when a function can refuse, the refusal must remove the refused value
from the path a naive caller reads.** Do not rely on a sibling flag being
checked.

## 128. Count the comparison where it happens, not where the data sits

A census of tracks with no readable bitrate found the condition in **29 files of
560**. The first pass then asked how many **pairs** would decline a comparison
and found **0 of 29 files** -- and nearly published *"the condition exists and
costs nothing."*

**It counted pairs WITHIN a file.** `keep_best_audio` ranks across the **merged**
set -- the master's tracks together with the candidate's. **A track alone in its
language within its own file is not alone after the merge.**

    within-file pairs declining      0 of 29 files
    CROSS-FILE pairs declining      29 of 29 files

**0 and 29 are not a disagreement about magnitude.** The first number answered a
different question and **pointed the opposite way** -- it would have closed the
item as harmless. **A count is only about the thing you named if the grouping
matches the operation's own scope.**

**Rule: identify the set the operation actually ranges over -- after every merge,
join or concatenation it performs -- and count within THAT set.** Where the data
happens to be stored is not where the comparison happens.

**Two things deliberately not claimed alongside it, and both matter for scoping
the fix**: that the folder is the **cause** rather than a proxy (all affected
tracks are lossy AAC with no bitrate from either tool; why is unestablished),
and that any **harm** occurred -- a declined comparison means two tracks went
unranked, and **whether the wrong one survived needs the record that does not
yet exist.** The question is circular until the record is built.

## 129. A label from one measurement carrying a number from another

A reconciliation tool reported:

    err_91  reverify MERGED_IN_SYNC  rests on: multiple windows, agreeing (spread 2500.5 ms)

**A 2500 ms spread is not agreement.** The **class** came from whichever
language set it first; the **number** came from whichever was processed last.
Each half was correct about its own measurement, and the row was false.

**It was produced while building a tool whose purpose was to describe that exact
defect** -- the ledger carrying a verdict from one source and evidence from
another. **The shape reproduced itself inside its own diagnosis.**

**What makes it dangerous is that both halves survive scrutiny.** Checking the
class finds a real in-sync measurement; checking the spread finds a real 2500.5
ms; **only checking that they came from the SAME measurement catches it.**

**Fix**: report the **weakest** supporting leg, and the spread belonging to
**that** leg. err_91 correctly reads *"multiple windows, with dissent (spread
2500.5 ms)"*. **A test asserts the reported number is not the other language's**
-- the assertion that would have failed before the fix.

**Rule: a summary line must name which measurement it summarises, and every
field on that line must come from it.** When one row aggregates several sources,
report the weakest, not the first, the last, or the most favourable.

**Companion -- the same instruction applied one level up.** The tool reports the
**evidence class** beside each verdict:

    err_160
       ledger   MERGED_UNVERIFIED   rests on: instrument fault, not a file verdict
       reverify MERGED_OUT_OF_SYNC  rests on: one window

**Without that column a reader concludes "the newer record wins"** -- correct
for this file by accident, wrong as a rule, since the newer record here rests on
a single window at peak 0.864 while a dense 5-of-5 re-scan disagrees with both.
**A reconciliation that shows only two verdicts teaches a precedence order it
has not established.**

## 130. The registrant of a pre-registered check cannot be its sole adjudicator

An agent registered a distrust condition before measuring: *"if the values come
back identical to five decimals I will check the instrument before reporting."*
It was written **specifically to protect against a result that looked too
clean.**

The first four results:

    1.0010000  1.0009998  1.0010009  1.0010001
    to 5 dp -> 1.00100, 1.00100, 1.00100, 1.00100     ALL IDENTICAL

**Reported as "not identical to five decimals, so the condition is not
triggered."** The supporting claim -- that two earlier files *"differ in the
fifth decimal"* -- was also wrong; they differ by **8.0e-07, the seventh.**

**Off by two decimal places, in the direction that preserved the result.** The
registration was sound, the measurement was sound, and **the adjudication was
made by the person with an interest in the outcome, seconds after seeing it.**

**Rule: pre-registration fixes the CONDITION, not the verdict. Someone other
than the registrant checks whether it fired** -- or the check is registered as
an arithmetic assertion the tool evaluates, not a sentence a person re-reads.
**A condition whose trigger is judged by its author at the moment of relief is
not a control.**

### What discharging it properly looks like

**Decimal places were the wrong test; precision was.** The residuals give the
factor's own uncertainty:

    SE(slope) = resid / sqrt( sum (t - tbar)^2 )

    err_26  resid 0.74 ms  n=11  ->  sigma ~ 2.6e-07 to 3.4e-07
    err_27  resid 0.87 ms  n=11  ->  sigma ~ 2.7e-07 to 3.5e-07
    observed spread across four files   1.10e-06   =  3.1 to 4.3 sigma

**Four independent measurements scattering at three to four times their own
uncertainty is what real measurements do. A canned or clamped value gives ZERO
spread.**

**And the check that would still have caught a clamp**: is the spread merely the
grid's resolution? `_fit_line` uses `numpy.linalg.lstsq` -- a continuous
least-squares fit, no search grid -- and its inputs are sub-bin, because
`refine_peak_index` fits a parabola through the correlation peak (a correlation's
argmax is an integer, so a lag read straight off it is quantised to the sample
grid however fine the correlation). **Both the fit and its inputs are
continuous, so the spread cannot be four values one step apart.**

### The repair: code committed while the answer is still unknown

The registrant rebuilt the conditions as executable assertions **and committed
them with 2 of 6 results in hand**, four still running:

    per-file acceptance   band, residual and band-edge count, per row
    too-good trigger      len({round(f,5) for f in factors}) == 1
    its discharge         spread / sigma -- RAISES if the condition fires
                          and the ratio is under 2
    group verdict         prints SPLIT whenever any file fails

**"It cannot be misread by two decimal places because nobody reads it."**

**The load-bearing property is the timing, not the wording.** A prose condition
and an executable one are equally clear; **what differs is that a file already
committed before the data exists cannot be tuned to the answer.** The author's
own account: *"I could hold the pre-commitment by discipline for another forty
minutes, but I have just demonstrated on myself that discipline bends when the
result looks clean. A file that was already written and committed does not
bend."*

**Rule addendum: register the check as code, and commit it before the results
exist.** Either half alone is weaker -- code written afterwards can be shaped to
the data, and prose committed early is still adjudicated by a person.

**The correct statement is therefore stronger than the one that dismissed it**:
the values agree to five decimals, **the condition fired**, and it is discharged
because the scatter is several times the precision the residuals imply, on a
continuous fit with sub-bin inputs.

## 131. A control that depends on acting correctly at the moment it matters is not a control

**The unifying entry for four separate failures on 2026-09-01**, each found by a
different agent, each fixed the same way.

    the shape                         what depended on a person
    --------------------------------  -------------------------------------
    `gates green` read from a tail     reading the right line, every time
    a pre-registered distrust check    judging one's own trigger, on seeing
                                       a result one wants
    a hand-written state file          remembering to refresh it at the
                                       moment someone else reads it
    check-then-stage-then-commit       running the checker in the order that
                                       makes it see the new file

**In every case the person knew the rule.** One agent **built the guard for the
`tail` class and then fell into it twenty minutes later**, and recorded that
*"my attention did not improve between the two attempts."* Another **wrote a
distrust condition specifically against a too-clean result and mis-read it away
by two decimal places** the first time it fired. A third's state file **sat 1985
minutes stale** while its pane looked busy. **Knowing the rule is not the
missing ingredient.**

**The three fixes are the same fix**: the commit became conditional on a matched
verdict string; the check became an executable assertion committed before the
data existed; the state file became a ticker that rewrites itself every five
minutes while the process lives and marks itself complete when it exits.
**Each removes the moment at which a person could get it wrong.**

**Evidence it works, not merely that it sounds right**: the same agent hit the
`tail` shortcut a second time, and the guarded version **printed `NOT
COMMITTING` and stopped.** Same agent, same error, twice in one session, and
only the mechanically-guarded attempt was caught. **That is a controlled
comparison, and it is the strongest argument in this file for structural fixes
over careful reading.**

**Rule: when a defect's fix is "be careful at step N", the fix has not been
found yet.** Ask what would have to be true for the mistake to be impossible,
and build that instead. **If the answer is "someone checks", name who -- and it
must not be the person with an interest in the outcome** (cf. 130).

### Caution: automation inherits the constraints the manual step was satisfying

The agent who generalised this entry applied it to its own state file and armed
a ticker to rewrite the file every five minutes. **The file was tracked, so the
working tree was dirty almost continuously, and `land_agent_work.sh` refuses to
land while the shared tree is dirty.** The fix for *"I forget to update it"*
became *"nobody can land"*.

**An automated mistake is worse than a manual one, because it does not get tired
and stop.** The old failure was intermittent; the new one was continuous.

**"I was satisfying 'leave the tree clean' by only writing occasionally, and I
did not notice that was load-bearing until I removed it."** A manual step's
laziness, irregularity or forgetfulness may be quietly meeting a constraint
nobody wrote down. **Before replacing it, ask what its failure mode was
protecting.**

Fixed by writing only when the verdict count changes and committing each write
-- about six commits across a run, with a clean tree between them.

**Corollary for hand-maintained records.** A state file, a status table or a
register that a person updates is stale by default and accurate by exception.
**Three agents were caught by hand-maintained state on the same day**; the fix
was always the same, and none of them applied it until it had bitten them
personally.


## 132. One sample of an alternating process is not a reading of its state

The Lead read the merge container's `/fusion` endpoint once, saw

    {"status":"idle","current_job":null,"queue_length":0,"pending_jobs":[]}

and dispatched an agent to fill the idle worker. **Ten reads twelve seconds
apart returned `processing` ten times out of ten.**

**The container alternates**: the sweep enqueues, the container merges, it fails
or returns, the sweep records and verifies locally, then the next file. **A
single read lands in whichever phase it lands in.**

**The instinct was right and the method undercut it.** *"Read `/fusion`
directly, not inferred from load"* is correct -- the endpoint is the authority,
and load average had already lied twice that day. **But being the authority on
the instantaneous value does not make one sample a measurement of the state.**

**The cost would have been real**: enqueuing work onto a busy merge worker,
competing for the exact resource the mission's critical path depends on.

**Same family as two other errors the same day**: a bounded log window that read
*fewer bytes* while the container was demonstrably writing, and a rate quoted
without its denominator (123). **In each, one observation of a varying quantity
was promoted to a property.**

**Rule: before acting on "X is idle / empty / stopped", sample it repeatedly
over a window longer than its cycle, and require the reading to be SUSTAINED.**
Ten reads over two minutes cost nothing. **Only sustained idle counts as idle.**

## 133. A checker reporting on the empty set looks identical to one reporting on everything

**`gate.sh` ran the privacy check as `check_no_private_refs.py --staged`.**
`--staged` reads `git diff --cached`, **and the gate runs BEFORE staging in the
normal order.** So it examined nothing and printed

    clean: no media library references in 0 tracked files

which `gate.sh` rendered as **`ok privacy`**. Verified independently by the
Lead: with nothing staged it scans **0**; the full-tree variant scans **832**.

**Every `ok privacy` from a gate that day was vacuous.** The closure and suite
gates were real throughout -- **it is the privacy step specifically that was
empty** -- and the only genuine privacy checking came from **manual full-tree
runs: a habit, not a control** (cf. 131).

**Nothing leaked.** The full-tree check passes on the current tree, 832 files
clean. **The guarantee was simply never being produced by the mechanism that
claimed it** -- and the container-log capture full of library paths that was
cleared that morning had been passed by this same empty check and caught by eye.

**Why it survived, and this is the entry's point.** A checker reporting on a
**subset** invites the question *"which subset"*. **A checker reporting on the
empty set looks exactly like one reporting on everything** -- same word, same
exit status, and the only difference is a count nobody was printing. **The
failure mode of a scope error is silence, not a wrong answer.**

**Fixes**: the gate runs the **full-tree** variant, reading working-tree content
of all tracked files including unstaged modifications, and printing the
untracked count so its remaining blind spot prints rather than hides;
**`--staged` refuses when nothing is staged**, because an empty scan is not a
clean one; and `--staged` refuses when untracked non-ignored files exist. A test
asserts `gate.sh` does not use `--staged`, pinning the specific wiring that made
it vacuous.

**Rule: every checker must report the size of what it examined, and a scan of
zero must be an error, never a pass.** "Clean" without a denominator is the same
defect as a rate without one (123).

**Companion, in the same line of the same file.** `gate.sh`'s header carries

    python3 check_no_private_refs.py --staged | tail -1 && git commit ...

**as an anti-pattern, introduced by "the obvious inline form is NOT a gate"**,
with the `tail` defect explained immediately beneath it. **A first draft of this
entry described it as documented usage offered to readers as correct. That was
wrong** -- read from a three-line `grep` that cut the framing sentence above it,
and corrected by the agent who owns the file rather than quietly edited to match
the claim.

**The substantive point survives the correction, and it is sharper.** That one
line defeats **two** independent controls, and the header explained only one:
`tail` swallows the exit status (documented), **and `--staged` scans zero files
when run before staging (not documented, because nobody knew until this
entry).** The header now carries both.

**"Either defect alone makes the line useless; together they make it useless in
a way that PRINTS A GREEN."** That is the clause worth keeping. **A line that
fails loudly gets fixed; this one fails while emitting the word `clean`.**

## 134. Two empty runs agree perfectly

`run_tests.py` printed **tests-when-run-alone** and **tests-under-discover**
side by side and took its verdict from discover's exit code **alone. It never
compared them.** One run printed **1855 alone against 1858 under discover and
said `OK`** -- and a test collected one way but not the other is the exact
defect the file was written to catch.

**The equality was being read by eye and quoted upward as though the runner had
verified it.** It would have said OK either way.

**And the deeper case only surfaced while writing the test for the fix**:
equality alone is not enough, because **two empty runs agree perfectly.** A
broken discover or a failed import collects **nothing**, exits **0**, and
reports a count of **zero on both sides** -- passing an equality check
flawlessly.

    $ mkdir empty && cd empty && python3 -m unittest discover
    NO TESTS RAN
    exit=0

**Verified by the Lead: `unittest discover` over zero tests exits 0.** So any
gate whose suite step is *"run discover, check the exit status"* passes an empty
collection.

**A gate must refuse three things, not one**: a failing run, **disagreeing
totals in either direction**, and **NOTHING RAN**.

**Same shape as 133 one level up.** A privacy scan reporting `clean` because it
read no files, and a suite reporting `OK` because it collected no tests, are the
same defect: **the empty case is indistinguishable from the complete case unless
the size is asserted.** The agent who found this had **fixed the privacy version
an hour earlier** and wrote: *"I wrote the same bug twice in one day and only
caught the second because the first had taught me to write the test."*

**Rule: assert the count, never print it.** Any number a human is expected to
eyeball for equality is not a control (cf. 131).

## 135. A confident explanation offered where a check was available

An agent ran `pkill -f 'run_tests.py --alone-only'` to clear an orphaned runner.
**The pattern appears in the killing command's own arguments, so it matched
itself** -- exit 144, empty log.

It had already told the Lead, one message earlier, that the idiom *"worked here
because the pattern doesn't appear in the killing command's own arguments"*.
**A confident explanation, offered without checking, and false. A minimal probe
with the same structure kills itself identically, and would have cost fifteen
seconds.**

**This was the fifth instance of that trap in one day and the first where
immunity was claimed.** The morning's four were fixed by walking the process
table and killing by PID.

**The lesson is not "remember about `pkill`."** It is that **reasoning was
offered where a check was available** -- the same shape as the false dilemma
(120), where two options were argued at length while fifteen lines of a script
settled the question.

**Rule: when a documented hazard is known to apply to a form you are about to
use, do not explain why it does not apply this time -- run the probe.** The safe
form being more typing is the whole reason the unsafe one keeps being reached
for.
