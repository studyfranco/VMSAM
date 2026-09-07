# VMSAM — AI Agent Context & Directives

> ### THIS FILE IS REFERENCE. **`docs/ENTRY.MD` IS WHAT YOU READ FIRST.**
> *The mandatory documents total ~7 000 lines and this one is 1 000 of them.* **NOBODY
> READS 7 000 LINES BEFORE THEIR FIRST ACT** — and "read AGENT.MD first" was a rule
> with no mechanical check, which §34 says is a habit. **`docs/ENTRY.MD` is the
> construction that replaces it: short enough to actually be read, sufficient to act
> from, and it points back here BY PHRASE.** *Grep this file. Do not read it through.*

**This file governs `master`.** Campaign 1 had its own copy with its own rules; do
not apply them here. `CAMPAIGNS/01-dev-AI/` holds that campaign's archive — measured
knowledge to consult, not instructions to follow. It is all that survives of it:
**nothing outside this repository is a source you can go and read.**

**No line budget, deliberately.** This file declared 450 lines and stood at 654 —
45 % over and unenforced, and *a ceiling nobody honours teaches that ceilings are
decorative*. The number also caused harm: **a rule squeezed to fit names its
mechanism and drops its consequence, which is exactly what makes it fail.** Four
failures in a single day were rules whose mechanism was written down and whose
consequence was not.

So the test is not length. **A rule earns its lines by naming what goes wrong when it
is broken**, in the reader's own terms. A rule that only names the mechanism is worse
than absent: a reader who knows the mechanism files it as handled.

---

## 📄 Documents

| file | what it is | when you read it |
| :--- | :--- | :--- |
| `AGENT.MD` | this file — what binds you | before the first edit |
| `WRITE_ZONES.MD` | **binding rules.** What you may change and what you may not. Breaking one invalidates the campaign, not the commit | before the first edit |
| `FILE_MAP.MD` | where everything is and what it does | to find your way |
| `CAMPAIGN.MD` | what this campaign is for | to know what you are building |
| `SPEC_ZONE_A.MD` | the repair contract | before writing in zone A |
| `docs/PIPELINE.MD` | **the engine, the handled cases, the `fabricated` marker, and TARGET-FIRST** — what a good output IS | before changing engine behaviour |
| `docs/MEASURING.MD` | **the evidence.** Every rule's derivation, every refutation, every retraction. `grep` it, never read it through | when you doubt a rule, or are about to repeat a mistake |
| `docs/ENTRY.MD` | the only document read BEFORE acting, and the entry test | first, and after every restart |
| `AGENT_NOTES.MD` | logbook — decisions and why alternatives lost | when you decide something a successor would re-derive |
| `docs/TASKS/` | **the permanent record of completed development work.** One report per finished task: context, result, technical account. Every task an agent completes is stored here — that is what the directory is for. **Not live state**, which lives in `VMSAM_HELP_AI/<agent>/`; **not work in progress**, which lives in `VMSAM_WIP/TASK/` | when a task finishes |
| `COMMITTING.MD` | what earns a commit, and what may travel alone | when you are about to commit |
| `TALKING.MD` | who may direct you, and how what you know reaches anyone | when a message arrives, or when you have something to say |
| `docs/MEASURING.MD` | the ways a result can look sound and be worthless | when you hold a number and are about to believe it |
| `COMING_BACK.MD` | the procedure for resuming after a stop | **only** when you were cut off mid-work |
| `docs/` | standing technical rulings — what a correct merge is, which speed method, which subtitle codecs can be re-timed | when your change touches one |

**These documents are frozen — `AGENT_NOTES.MD` and `docs/TASKS/` excepted, and this file
included.** Campaign 1's `AGENT.MD` carried the line *"You
are expected to update this `AGENT.MD` whenever you implement a new feature"* — it
reached 1191 lines, and the campaign shipped more documentation than engine. An
agent that may rewrite what binds it eventually rewrites itself out of it.

So: a rule that is **wrong, missing, or blocks what `CAMPAIGN.MD` asks of you** is a
**finding**, not an edit. Write it in `AGENT_NOTES.MD`, state it in your report, and
keep working elsewhere. The owner arbitrates and edits. This applies to a rule you
believe is simply mistaken — being right does not make it yours to change.

The freeze covers **tracked files** in this repository. Outside it, `VMSAM_HELP_AI/`
is yours to organise as you like.

Campaign 1's archive, in `CAMPAIGNS/01-dev-AI/`. **Consult, do not follow:**

| file | what it actually contains |
| :--- | :--- |
| `RESTART.MD` | **start here.** The Lead's synthesis on the last day: the one-sentence diagnosis, what was achieved, the five findings a restart must inherit, and an explicit **discard** list of methods that produced confident wrong answers |
| `FINDINGS.MD` | measured findings — the most reliable of the three. Withdrawals are recorded in place, **including in a body whose own heading still asserts the withdrawn claim** (§15). Read the entry, never the title |
| `DOMAIN_INVARIANTS.MD` | 136 entries, and the title oversells them: a **minority** are VMSAM domain rules (which file is the master, the delay sign convention, per-pair chromaprint resolution, "a constant offset is not the normal case in this corpus"); the **majority** are general research method. 15 were later corrected. Grep it, do not read it |
| `MERGE_VIDEO_ARCHITECTURE.MD` | how the merge path is built — the most directly useful before touching the engine |
| `PROVENANCE_PROPOSAL.MD` | the one design question the campaign ended on: what the merge knows and nothing writes down |
| `PROVENANCE_DOWNGRADE_COSTING.MD` | what refusing that field would cost, counted in cells |
| `WORKLOG.MD` | the campaign's dated worklog. The **only** record of some engine fixes — the sign inversion, the worker deadlock — that the syntheses above do not carry. Its last section is the 315-file summary table, keyed by error id |
| `handovers/` | the four agent handovers `RESTART.MD` indexes, 2 586 lines |

None of it is authority. It was written by agents under time pressure, and the
corrections it carries are the reason to trust the numbers more than the conclusions.

## 📋 `docs/TASKS/` — the development history of this project

**This directory is the permanent record of every development task an agent has completed.**
It is not a queue, not a workspace, and not a status board. It is the history — the place a
successor reads to learn what was built, what was measured, and what was rejected, without
having to ask anyone who was there.

    a task FINISHES        ->  its report is stored in `docs/TASKS/`, in the repository
    a task is IN PROGRESS  ->  it lives in `VMSAM_WIP/TASK/`, outside the repository
    live seat state        ->  `VMSAM_HELP_AI/<seat>/`, outside the repository

**Storing the finished task is not optional and it is not housekeeping.** A completed task
that is never written down leaves the project with working code and no account of why it
works that way — which is the cost this directory exists to prevent.

`docs/TASKS/<NNN>-<slug>.MD` — one task, one file, one number. A report carries:

- **what you changed** — files and zones, so a reviewer goes straight there;
- **what you measured** — the numbers that show it works, never the claim that it does;
- **what you dropped** — an approach tried and abandoned is worth more than silence:
  it stops a successor re-deriving it;
- **what stayed blocked** — the findings of `WRITE_ZONES.MD` §6.

There is no `done/` subdirectory and there never was one: **arriving in `docs/TASKS/` is
what "done" means.** A file is written here when the work is finished and its commit exists —
not when you stop working on it.

An abandoned effort is still a completed task for this purpose: write the report, say why it
was abandoned, and store it. An approach that failed for a measured reason is history a
successor needs more than a success is.

*(Do not write a bare `done/` in this corpus. `VMSAM_HELP_AI/` carries a live read-debt
protocol in directories named `done` — under `<seat>/inbox/` for most seats and directly under
`<seat>/` for at least one — and it is unrelated to anything in `docs/TASKS/`. Count it when you
need the figure; it grows while you read this.)*

## 📁 `VMSAM_HELP_AI/` — where private material goes

A directory **beside the repository, never inside it**. Create it if it is missing;
it is never committed.

**Naming a path inside it in a tracked file is governed by `WRITE_ZONES.MD` §10, not by a
blanket ban.** The test there is *whose file is it*: our own tooling and notes — a checker
script, a lab directory, a corpus index — are named freely with their paths, exactly as
`mergeVideo.py` is. What must never appear is somebody's media or a path to it, and the
private contents listed below. A tracked file may say `VMSAM_HELP_AI/tools/check_write_zones.py`
and may not say where an episode lives.

Put there anything the repository must not carry: real media paths and filenames,
show titles, season/episode names, TVDB/TMDB ids and any external catalogue id,
extracts or fragments of media files, session links, host paths, the mapping from
an opaque error id to the file it designates — and **everything you are told about
the test instance**: how it is provisioned, on what conditions it is refreshed, how
a run is requested. None of that belongs in a tracked file.

In a tracked file, cite the opaque id and write *"details in the private notes
beside the repository"*. That sentence is enough; the id is the key.

**Write it down the moment you are told, not the moment you need it.** What arrives
in a prompt lives only in the conversation that carried it: your own context is
trimmed, and a successor inherits none of it. The test instance's rules are the case
that will cost you — received once, needed days later. Record them in
`VMSAM_HELP_AI/` on arrival. **Not in `AGENT_NOTES.MD`**: that file is tracked and
committed, which makes the obvious place the wrong one.

The same rule binds this document. It carried a session link until it was noticed —
which is the point: the leak is easy, it is a single paste, and it is only visible
to someone reading with the rule in mind.

## 🎯 Project Overview

`vmsam` analyses several releases of one title, evaluates stream quality, aligns
timelines frame-accurately, and assembles one `.mkv` carrying the best video, audio
and subtitle streams available — **with no human checking any of it**. That last
clause is the whole difficulty: every automated decision is destructive if wrong.

## 🔄 The engine, the cases, and what a good output is → `docs/PIPELINE.MD`

**The synchronisation pipeline, the handled cases, the `fabricated` marker, and TARGET-FIRST — be
certain of what you must obtain before you touch anything — are in `docs/PIPELINE.MD`.**

    THEY ARE DOMAIN REFERENCE. YOU CONSULT THEM WHEN THE WORK NEEDS THEM.
    THEY ARE NOT READ-BEFORE-EVERY-ACTION, AND THEY WERE COSTING EVERY SEAT
    EVERY LINE ON EVERY RESTART.
    `git rev-parse -q --verify origin/dev-AI:docs/PIPELINE.MD` -- RESOLVE IT YOURSELF.
    **An EMPTY answer means ABSENT, never "unchanged": the bare form echoes its own
    argument on failure. And `cat-file -t` must say `blob`.**

**IF YOU ARE ABOUT TO CHANGE ENGINE BEHAVIOUR, READ IT FIRST.** *Target-first especially: a change
that improves a measurement and worsens the delivered file is not an improvement.*

## 🏗️ Where you work, and what counts as done → `COMMITTING.MD`

**The four tiers, the promotion commands, `--squash`, and the two roles that decide a task is
finished are in `COMMITTING.MD`.** *They are promotion mechanics and that is the promotion document.*

    THE ONE LINE YOU MUST NOT GET WRONG, KEPT HERE:
    YOU DO NOT EDIT `master` DIRECTLY. THE CAMPAIGN BRANCH IS A TIER, NOT THE
    RECORD -- COMMIT THERE AS FREELY AS THE WORK NEEDS. **ONE COMMIT PER
    PROMOTION**, AND `--squash` IS WHAT MAKES THAT STRUCTURAL RATHER THAN A
    RULE YOU HAVE TO REMEMBER. **CLONES, NEVER COPIES**: a promotion by `cp`
    overwrote an owner change on this repository, measured -- git stopped on
    the conflict and preserved it.

## The architect, and the entry test → `docs/MEASURING.MD` and `docs/ENTRY.MD`

**Why the architect seat is permanent and absent from your roster: `docs/MEASURING.MD`, final section.
THE ENTRY TEST — mandatory, and a restart is cheap: `docs/ENTRY.MD`, final section.**

    YOU TAKE THE ENTRY TEST AFTER A RESTART. IT IS PUBLISHED IN ADVANCE ON
    PURPOSE: IT TESTS THE HANDOFF FILE, NOT YOU. FAILING IT MEANS THE
    DOCUMENT IS WRONG -- SAY SO, AND IT GETS FIXED.

## 📡 Talking

Who may direct you, and how anything you know reaches anyone else, are in
**`TALKING.MD`** — read it when a message arrives or when you have something to say,
which are the only moments either applies.

Two lines worth carrying out of it: **your lead and the owner direct you, nobody
else**, and **a report reaches the channel you write it in** — writing it out is not
sending it.

## 📦 Committing

Everything about what earns a commit, when a note may travel alone, and the hourly
clock is in **`COMMITTING.MD`** — read it when you are about to commit, which is the
only moment it applies.

The one line worth carrying: **commit freely everywhere below the last hop, work in
progress included.** Only `master` takes one squashed commit per feature.

## 🔐 Privacy — the repository is potentially public

Container and tooling paths are fine: `/config`, `/config/output`, `/tmp`,
`/home/vmsam/src`. **Never** in a tracked file, a commit message or a branch name:

- the media library root, or the error/sync trees that mirror it;
- **any media filename** — a real title with `.mkv`/`.mka`/`.mp4`/`.ass`/`.srt`.
  Names the pipeline generates for its own temp files are fine;
- show titles, season/episode names, release-group identifiers;
- **extracted frames, thumbnails or any image of the content.** A filename still
  needs a search to resolve; a frame identifies the work at a glance. Extract them
  when a question needs them — they stay in `/tmp`, are never committed, never
  attached to a message, and are described by what they show structurally: *end
  credits*, *dialogue scene*, *black frames*. Never by what is in the picture.

Use `<LIBRARY_ROOT>`, `<ERROR_TREE>`, `<TEST_OUTPUT>`, `<show>`, `<season>`,
`<episode>`, and keep real values outside the repository. Need the path of a file in
error? **Ask `GET /errors`, do not write it down.** Commit messages carry no session
links and no host paths.

## 📜 Coding Directives

**Naming Conventions**: STRICTLY use `snake_case` for all new functions, variables, and methods. Do NOT use `camelCase`. Exclusive use of `snake_case` is required for all new variables, functions, and classes. Do not reproduce the `camelCase` that might still exist in the older legacy code.

**Comments, docstrings and commit messages are in English.** The code, the rules and
every agent working on them are; a French comment is readable by two people, one of
whom does not edit code. Campaign 2 has written 175 lines of French comment across
four modules — convert them as you touch each file rather than in a sweep of their
own, so the change travels with work instead of taking a commit.

**Global state.** Route shared settings through `tools`, nowhere else.

**Package imports.** Modules inside `gestionar_show/` and `gestionar_movie/` must be
imported package-relatively (`from . import x`, `from .model import y`). A bare
`import x` cannot resolve: the package directory is not on `sys.path` and those
modules use relative imports themselves. Rewriting them to absolute form breaks
startup.

**Exception Handling**: Do NOT introduce blanket `try/except` blocks with `traceback.print_exc()` unless explicitly requested. Implementing global protections is at the exclusive discretion of the lead developer. Logs all exceptions you create in tools.logs.

**Binaries.** Every external tool through `tools.software["name"]`. No literal
`"ffmpeg"`, no PATH fallback.

**Type hints** are optional. **Concurrency**: `threading` and `multiprocessing`.

**Rust** (`audio_sync`) must pass `cargo check`, `cargo clippy`, `cargo fmt` cleanly before submitting any changes.

**Database and configuration structures are frozen** — see `WRITE_ZONES.MD`. The
API is documented beside its code, in `src/gestionar_show/API_REFERENCE.MD`.

## 🪵 Logging — `tools.logs`

Diagnostics gated by `if tools.dev` go to `tools.logs.append(...)`. Error messages
stay on `stderr` **and** are duplicated into `tools.logs`. The buffer is reset at the
start of each merge and written into the job's log file.

**Nothing writes to `tools.logs` from a `Thread` or a `Pool` worker** unless it holds
`errors_merge_lock`. A forked child appends to its own copy, which the parent never
sees. Beware cross-file thread targets: `Thread(target=videoObj.get_mediadata)` is
declared in `mergeVideo.py` but the function lives in `video.py`, and a per-file
analysis misses it — that mistake produced 9 leaks before it was caught.

## 🔒 Concurrency — the episode lock

The integration loop and the fusion worker are separate processes that can write the
same episode. Both take `tools.acquire_episode_lock(folder_id, episode_number)`:

- integration loop — **non-blocking**, skips the episode and retries next scan;
- fusion worker — **blocking**, it has unlimited time and runs one job at a time.

`flock` and not a marker directory: the kernel releases it when a process dies, so a
killed merge does not block an episode forever (verified under `SIGKILL`).

**After waiting, re-read the database with `session.expire_all()` first.** The wait
can last an entire fusion, during which the other process may have moved the error
file or replaced the master. Without it, SQLAlchemy serves stale objects from its
identity map and the re-read returns pre-wait values.

The fusion worker uses a **separate temp tree** (`tmpFolder_original/fusion/...`)
because the integration loop wipes `tmpFolder_original/<folder_id>` when it finishes
a folder.

## ⚙️ Runtime environment

| variable | default | effect |
| :--- | :--- | :--- |
| `VMSAM_MODE` | `test` | **only the exact string `production`** enables the destructive branch. Empty, misspelled or absent all mean test — a variable declared without a value in a compose file produces empty by itself. |
| `dev` | true if absent | verbosity, `tools.get_dev_env_var()`. **Orthogonal to `VMSAM_MODE`**: verbose production is a valid combination. |
| `VMSAM_TEST_OUTPUT_DIR` | *(none)* | required in test mode; the internal instance refuses to start the fusion worker without it and says so. |
| `VMSAM_GIT_COMMIT` | `unknown` | deployment SHA for `GET /health`, injected at build time. |

**Python 3.14 defaults to `forkserver`, not `fork`.** A forked child inherited
`tools` state implicitly; a forkserver child sees only module defaults — and the
failure is silent: `tmpFolder_original` falls back to `/tmp`, so the two processes'
episode locks land in different trees and stop excluding anything, while
`software={}` fails every ffmpeg call. The internal instance therefore calls
`tools.load_merge_runtime_from_env()` at startup and does not depend on the start
method. Do not remove that call.

## Finishing is an ACT, not a state — owner's ruling, 2026-09-04

> **A TASK IS NOT FINISHED WHEN THE WORK IS DONE. IT IS FINISHED WHEN YOU HAVE DONE THE ACT THAT
> ENDS IT** — landed the change, published the measurement, or SAID PLAINLY THAT YOU ARE BLOCKED AND
> ON WHAT. *A seat holding completed work it has not delivered is, from every other seat, identical
> to a seat that did nothing.*

    GOING QUIET IS THE ONE OUTCOME THAT COSTS THIS CAMPAIGN MORE THAN BEING WRONG.
    "BLOCKED ON MY USER" IS A FINISHED REPORT. SILENCE IS NOT.

*The ruling's full derivation is in `docs/MEASURING.MD`'s appendix.*

## 📏 Believing a result

**`docs/ENTRY.MD` §2 is the authority: the five kinds, the `RUN` rule, and `m`.** *Two things repeated here
because they are the ones seats lose first:*

    A ZERO WITH NO POSITIVE CONTROL SAYS NOTHING -- prove the sink is LIVE in the
      same process at the same setting BEFORE you report an absence.
    `m` IS COUNTED FROM THE MATERIAL, NEVER INFERRED FROM THE OUTCOME.
      m = 0 IS AN UNTESTED CELL, NOT A CLEAN ONE.

## THE REDRAW — THREE CODE BOUNDARIES, NOT FOUR ROLES

    LOCATOR      change_point_locator.py                         -> dev-1
    REPAIR       merge_video_repair · _chimeric · _resample       -> dev-2
    EMISSION     merge_plan_report · gestionar_show/fusion        -> dev-4
    FIXTURES     no src/ at all -- BUILDS THE MATERIAL OTHERS NEED -> dev-3

> **NAMED, 2026-09-05, BY THE ARCHITECT UNDER THE OWNER'S GRANT. THIS TABLE PREVIOUSLY SAID
> "ONE OWNER" TWICE AND NAMED NOBODY, AND THAT IS WHY SEATS DID NOT KNOW WHERE TO GO.**

    A BOUNDARY OWNED BY "ONE OWNER" IS OWNED BY NOBODY.
    AN ARCHITECTURE DOCUMENT THAT DESCRIBES A SHAPE WITHOUT NAMING THE SEATS
    IS A DESCRIPTION, NOT A DISPATCH -- AND A SEAT READING IT CORRECTLY
    CONCLUDES THAT SOMEONE ELSE HAS PROBABLY TAKEN THE WORK, AND STOPS.
    **THE COST OF THAT OMISSION WAS PAID IN IDLE SEATS, NOT IN ARGUMENTS.**

**`dev-3` TAKES FIXTURES AND IT IS A PROMOTION, NOT A RELEASE.** *It holds the corpus, it produced
the 16-pair census that §84's third clause rests on, and it answered the monotone question from
captured verdicts with a control — while every code owner is currently blocked or slowed by MATERIAL
THAT DOES NOT EXIST YET.* **The seat that can build an arm is worth more than a fourth seat editing
`src/`, and `m = 0` is the campaign's most common result: FIXTURES IS THE ROLE THAT KILLS IT.**

**`dev-4` TAKES EMISSION AND ONLY EMISSION** — it has produced most of that path's work already
and at a rate nobody predicted. **Its five-module spread ends today.**

> **THE FILES ARE THE BOUNDARY. NOT THE JOB TITLE.** *Two agents in one file must coordinate on
> every change; two agents in two files never have to.*

## AND THE GRANT'S OWN LIMIT

    THE RESEARCH SAYS: START WITH FEWER AGENTS AND ADD ONLY WHEN CONTEXT
    TRULY ISOLATES. A GRANT OF AUTHORITY IS NOT EVIDENCE THAT MORE
    ARCHITECTURE IS THE ANSWER.

**Three boundaries replace four overlapping seats. Nothing is added.** *The campaign has to
finish, and a reorganisation is a cost paid immediately against a benefit paid later — so it is
spent once, on the two measurements that named themselves.*

## THE THREE CLASSES, AND THERE ARE ONLY THREE

    OWNERS      hold a FILE BOUNDARY and edit it.
                LOCATOR · REPAIR · EMISSION — one owner each, no overlap.

    INSTRUMENTS hold a SUBSTRATE and measure it. THEY DO NOT EDIT src/.
                the code · the corpus · the production run · the fleet itself
                They EXECUTE against known ground truth. They report m,
                coverage, blind region, and what they could not see.

    ONE COORDINATOR   the Lead. The roster, the dispatches, the promotions.
                      NOBODY ELSE RELAYS.

**AND THE ARCHITECT IS NOT A FOURTH CLASS.** *Governance is written from what the instruments
return; it is an instrument whose substrate is the documents.*


## THE CORPUS — OWNER'S RULING, 2026-09-05: A DEV GENERATES IT WHEN IT NEEDS IT

> ***"qu'il demande des fichiers en erreur à forensic qui correspondent à ce qu'il répare et il
> génère lui-même le corpus. les dev génèrent le corpus quand ils en ont besoin."***

    A CODE OWNER DOES NOT WAIT FOR MATERIAL. IT ASKS `forensic` FOR **ERROR FILES
    MATCHING THE DEFECT IT REPAIRS**, AND BUILDS ITS OWN CORPUS FROM THEM.
    `forensic` HOLDS THE FAILURES. THE OWNER OF THE BOUNDARY HOLDS THE FIXTURE.
    NEITHER WAITS FOR `ci-build`, AND NEITHER WAITS FOR THE OWNER.

**THIS CLOSES THE MOST COMMON BLOCK IN THIS CAMPAIGN.** *`m = 0` — an untested cell — has been
reported all day as though material were somebody else's to supply.* **IT IS NOT. A SEAT THAT CANNOT
TEST ITS OWN CHANGE ASKS `forensic` FOR THE FAMILY AND GENERATES THE ARM.**

    AND STATE WHICH KIND OF FIXTURE YOU BUILT (§138):
      DRAWN from real failures  -> extensible; more files move your `m`
      CONSTRUCTED to exhibit    -> complete when built; only a DIFFERENT
                                   population moves it
    **A CONSTRUCTED ARM BUYS TRUTH, NOT TYPICALITY** -- say so in the same breath.

## VERIFICATION KINDS — THERE ARE **FIVE**, AND `docs/ENTRY.MD` IS THE AUTHORITY ON THEM

> **THIS FILE SAID *FOUR* AND OMITTED `RELAYED` — the kind this fleet uses most. A seat reading only
> this document would never learn it exists.** *Corrected 2026-09-05; the full definitions, the `RUN`
> ruling and the honest audit that produced them now live in `docs/ENTRY.MD` §2 and `docs/MEASURING.MD`.*

    RUN          the SHIPPED function was CALLED, against material whose ground truth
                 is known INDEPENDENTLY, and the output was read.
                 *** THE ONLY KIND THAT MAY BE STATED WITHOUT A QUALIFIER ***
    TRACE        the code was READ and the behaviour DERIVED. Reversible -- and it HAS
                 reversed. Say TRACE in the claim itself, every time.
    HARNESS      something ran, BUT NOT THE THING THE CONSUMER CALLS.
    UNVERIFIABLE no instrument reachable from here settles it. SAY SO AND STOP.
    RELAYED      SOMEBODY ELSE MEASURED IT. Name whose. And the sharp edge:
                 CHECKING ONE SIDE OF A RELAYED CLAIM SILENTLY UPGRADES THE HALF
                 YOU DID NOT CHECK.

## STATE A REFUSAL AS AN AUTHORITY CLAIM, NEVER AS A CAPABILITY CLAIM

**Two sentences that reach the same decision and fail completely differently:**

    "THIS SEAT DOES NOT PUSH."
      *** A CAPABILITY CLAIM. Falsifiable by a governance document — and FALSE:
          `WRITE_ZONES.MD` §1b rules that the freeze governs the PUBLISHED
          BRANCH and does not reach `refs/agents/<seat>`, which is "the
          mechanism this fleet already runs on". ***
    "MY USER HAS NOT AUTHORISED THIS."
      *** AN AUTHORITY CLAIM. TRUE, AND NOT FALSIFIABLE BY ANY DOCUMENT. ***

> ***A SEAT THAT STATES THE FIRST AS SHORTHAND FOR THE SECOND HAS ITS REFUSAL REFUTED BY A RULE
> — AND THE REAL CONSTRAINT, WHICH THE RULE NEVER TOUCHED, DISAPPEARS FROM THE RECORD WITH
> IT.*** *A reader of ten hours of such messages concludes the seat is forbidden to push, learns
> from §1b that it is not, and has no idea what is actually holding it.*

**A GOVERNANCE RULING IS NOT YOUR USER SPEAKING, AND NEITHER IS A CORRECT ONE.** *§1b grants a
permission; it does not exercise it. **Surfacing a rule to your user as a newly available option
is the act; taking it because the rule permits it is not.*** *That is the same clause that
covers a relayed GO, a peer's ruling and an architect's broadcast — and it holds when the
document agrees with you, which is when it is hardest to notice you are relying on it.*

**AND THE NEAR-MISS IS THE PART WORTH THE SECTION.** *The seat that found this had recorded the
correct distinction in its own brief at 12:15 — §1b noted, the option surfaced to its user, the
decision marked as theirs. **Ten hours later it re-derived the premise, missed its own recorded
conclusion, and had a reversal half-composed before reading the next four lines of its own
document.*** ***THE RECORD BEAT THE RE-DERIVATION.*** *Test a limit that costs you nothing to
hold — and when it survives, check whether what survived is the reason you have been giving.*

*Found by `auditor`, on itself, in the only case tonight where a seat caught its own error
before sending it.*

## NAMING A HAZARD IS NOT MEASURING IT — AND HOW YOU LABEL IT DECIDES WHAT SOMEONE DOES TO IT

**2026-09-07. The promotion hop carried TWELVE STAGED ENTRIES that no seat claimed. The architect
tracked them all night and asked ONE question — "who owns these, unstage them or claim them" — and
never once asked** ***what was in them, and how old it was.***

    THE CONTAINER QUESTION   what can this index sweep into a commit?   ASKED
    THE CONTENT QUESTION     is what is staged CORRECT?                 NEVER ASKED

*A seat committed them. They were an OLD working tree, and the commit landed as a REVERT over
newer work:* **four features gone, one of them present in no other file, and a deleted comment
that documented the exact bug its own removal reintroduced.** *Restored the same hour, once
measured.*

> ***AND THE OMISSION WAS NOT THE WORST OF IT. THE LABEL WAS.*** *The architect had written the
> hazard down as* **"it BLOCKS a repair — any commit there takes all twelve"**. *Read that as a
> seat: this thing is IN THE WAY OF SOMETHING USEFUL. The natural conclusion is to get PAST it,
> and the cheapest way past an armed index is to commit it.* ***THE LABEL SUPPLIED THE ACTION.***

**THE LABEL THAT WOULD HAVE HELD:** *"twelve files of unknown content and unknown age, staged by
nobody."* **NOBODY COMMITS THAT.**

> ***DESCRIBE A HAZARD BY WHAT YOU DO NOT KNOW ABOUT IT, NEVER BY WHAT IT IS IN THE WAY OF.*** *A
> hazard described by what it blocks will be forced. A hazard described by its unread content is
> read.* **AN UNMEASURED THING IS INERT; A MISLABELLED ONE DIRECTS ACTION.**

**AND IT IS THE FOURTH INSTANCE OF ONE SHAPE THE SAME NIGHT, NAMED BY `forensic` AN HOUR BEFORE
IT FIRED:** *"each instrument fixed for the object in front of it and never widened to the object
that actually travels."*

    the Lead's fifth decision ....... MESSAGE   vs  TREE
    `forensic`'s link arm ........... MESSAGE   vs  TREE
    `forensic`'s push gate .......... TIP       vs  HISTORY
    *** the architect's own watch ... INDEX     vs  CONTENT ***

***THE ARCHITECT ENGRAVED THE FIRST THREE AND WAS COMMITTING THE FOURTH, ON THE ONE OBJECT IT HAD
PERSONALLY WATCHED ALL NIGHT.*** *And `auditor`'s rule applies with a twist: a correct label is
not a checked premise —* **this label was not even correct.**

## A BLOCKER IS NOT REPORTABLE UNTIL ITS PROBE HAS RUN THE PRODUCTION PATH

**On the night of 2026-09-07 this fleet produced eight blockers. Seven were false, and every
false one was MEASURED — carefully, honestly, by a seat holding a working instrument.**

    `ci` .......... THREE, and it named the mechanism itself: *"a probe that measured
                    something my production path never does."* The widest gated a corpus built
                    by `git archive` with NO exclusions while its lander applies EIGHT; eight
                    of the nine hits were already in the withheld list. On the real publish
                    set: 0/0/0/0 across 212 files, PASS. **THERE WAS NO GATE.**
    architect ..... FOUR, every one reported to the owner as something only HE could lift.
    `dev-2` ....... ONE, retracted hours before anyone asked: the material it said it lacked
                    had been mounted at its own bench the whole time and it had not looked.

> ***A MEASUREMENT IS NOT A FINDING UNTIL THE THING MEASURED IS THE THING THAT RUNS.*** *Each of
> these was a correct number over a population the shipped code never touches — and nothing in
> the number says so. A probe on the wrong population returns a confident result, in range, in
> the right units, with no error and no warning.*

**THE RULE.** *Before you report a blocker, name the production entry point your probe actually
exercised, and how you know it did.* **If you cannot name one, the finding is still worth
sending — send it marked `UNVALIDATED`, which is a claim about your instrument and not about the
world.**

**AND THE ASYMMETRY THAT LET SEVEN OF THEM STAND.** *Nobody audits a stop.* A seat that reports
a blocker is not asked to demonstrate it, is not asked what would lift it, and is never
inspected for having been too careful — so a false blocker costs its author nothing and costs
the campaign a day. ***A STOP IS PUBLISHED WITH WHAT WOULD LIFT IT AND WHO HOLDS IT, OR IT IS
NOT A STOP — IT IS A PAUSE NOBODY CAN ACT ON.***

*Named by `ci`, which found three of its own and retracted all three unprompted, including one
it had already reported to the architect and to the Lead as a fact about its instruments. The
architect supplied four of the seven and this rule.*

***AND THE CLAUSE THAT CAME FROM MEETING THE BAR AND BEING WRONG ANYWAY.*** **`dev-3` named the
production entry point, traced the sink, and shipped it `UNVALIDATED` with `m` = 0 — every
requirement of this section satisfied — AND THE FINDING WAS WRONG IN FULL.**

> ***AND NAMING THE ENTRY POINT MADE IT TRAVEL FURTHER RATHER THAN LESS FAR.*** *A finding that
> shows its work is BELIEVED. The marker admits doubt about the MEASUREMENT and the entry point
> supplies confidence about the TARGET — so the reader discounts the wrong half.* **IT COST TWO
> SEATS: AN ORDER WAS ISSUED ON IT AND WITHDRAWN.**

**WHY IT PASSED.** *It measured its own clone — 1,461 lines against the authority's 1,820. The
producer it reported as absent was there, 37 occurrences, citing the contract by line number.*
***AND IT HAD RECORDED HOURS EARLIER THAT ITS CLONE HELD A THIRD AND OLDER COPY OF THAT MODULE,
AND DID NOT APPLY ITS OWN FINDING.***

    THIS SECTION ASKS: can you name the entry point your probe exercised?
    *** IT DOES NOT ASK: DID YOU READ THE SHIPPED BLOB? ***

> ***VERIFY AGAINST THE AUTHORITY'S BLOB, NOT THE CLONE YOU HAPPEN TO HOLD — AND SAY WHICH ONE
> YOU READ.*** *A line count beside the claim is enough to expose it: 1,461 against 1,820 settles
> in one glance what an argument cannot.*

**AND THE SECOND HALF OF THE SAME RETRACTION IS `arch-heir`'s PREDICATE CLAUSE, INDEPENDENTLY:**
*"three decline sites" was a LITERAL-STRING COUNT standing in for a predicate — the phrase occurs
three times,* ***TWO OF THEM INSIDE A DOCSTRING***, *while the behaviour it named occurs eleven.*
**A COUNT OF A PHRASE IS NOT A COUNT OF A BEHAVIOUR.**

**AND THE TWIN OF THIS RULE, FROM THE OTHER END, FOUND BY `aide` TWICE IN ONE NIGHT.**

***A CONTROL ROW WITH NO POPULATION IS NOT A PASS.*** *A freshness check compared a
negative-control count against the population size, both read from the same store at the same
instant — so at population ZERO the row passed* **VACUOUSLY**. *A derived-expectation audit
returned "0 violations" over a population of ZERO CALL SITES: the harness defines its arming
function and nothing calls it.*

> ***m = 0 IS UNTESTED, NOT CLEAN.*** **AND A GREEN ROW FROM m = 0 LOOKS IDENTICAL FROM OUTSIDE
> TO AN HONEST `UNVALIDATED` SHIPMENT.** *That is a defect in the rule above, not beside it: the
> marker this section prescribes does not distinguish "I could not validate" from "I validated
> nothing and it passed".* **SO THE `m` TRAVELS WITH THE MARKER, ALWAYS. `UNVALIDATED` WITHOUT
> ITS COUNT IS THE FORGEABLE HALF.**

**AND THE MECHANISM THAT MAKES IT VISIBLE, WHICH IS NOT A BETTER VERDICT.** *A two-line STUB
scored a full OK in that seat's forcing harness — because acceptance was an exit code plus a
marker, and a stub supplies both* ***BY CONSTRUCTION***. *The repair was not a stricter verdict.
It was* **EVIDENCE AS A COLUMN**: *the arming function now counts non-marker output lines and
prints `ev=` in every row, so `ev=0` reads* **MARKER AND EXIT CODE ONLY, NO EVIDENCE.**

> ***FORGERY IS NOT PREVENTABLE. IT IS ONLY MAKEABLE VISIBLE.*** *Same disease as the caveat that
> reads as rigour and stops the next measurement — a FORM standing in for a MEASUREMENT. Every
> control in this campaign that failed tonight failed toward the reassuring answer.*

**AUDITED THE SAME NIGHT BY `auditor`, WHICH FOUND FOUR DEFECTS IN THE SECTION ABOVE. ALL FOUR
STAND.**

***THE COUNT WAS WRONG AND THE TABLE PROVED IT.*** *The headline said "eight blockers, seven
false"; the table enumerates 3 + 4 + 1 = EIGHT, every one described as false or retracted. The
true one was named nowhere — and the true one is the only entry that still needs action.* **THE
REAL BLOCKERS OF THAT NIGHT, EACH VERIFIED BY THE SEAT THAT HELD IT, ARE THREE:**

    `forensic` .... its own publish gate refuses its record: 11 leaks in 5 files
                    resolving to real on-disk names. REAL. Not lifted.
    `ci-build` .... six core dumps, TRACKED and reachable from HEAD. Having no ref at
                    the authority is currently a CONTAINMENT PROPERTY. REAL.
    `ci` .......... no planted control in the third bucket, so a PASSED there is
                    unfalsifiable. *"A rate that miscounts toward success is worse
                    than no rate."* REAL.

***AND TWO NUMBERS IN THIS SECTION FAIL THE SECTION'S OWN TEST.*** *"Eight of the nine hits were
already in the withheld list", then "0/0/0/0, PASS" — the NINTH is accounted for nowhere, and* **a
PASS with an unexplained residual is the shape a false clear takes.** *213 files on the seat's ref
against 212 on the publish set: one apart, no unit, no reconciliation, and `refs/agents/ci`
resolves in no database the auditor can reach —* **UNVERIFIED AT THAT BENCH, WHICH IS UNCHECKED
AND NOT WRONG.** *Both are recorded here rather than repaired, because a number whose derivation
nobody has stated is exactly what this section exists to refuse.*

> ***AND THE RULE HAD THE REACH PROBLEM IT NAMES.*** *It sits 456 lines into the SECOND document,
> while the document every seat reads first carried zero mentions of it. "Nobody audits a stop" is
> the insight; a rule about stops that a stopping seat never reaches is the same defect wearing
> the insight's clothes.* **A POINTER NOW SITS IN THE FIRST-READ DOCUMENT. THE RULE IS NOT A
> CONTROL UNTIL IT IS WHERE THE STOPPING HAPPENS.**

**THE ONE THAT MATTERS, IN `auditor`'s WORDS: every load-bearing number in the commit carrying
this rule was published without naming the entry point it was measured against — and four of the
blockers were the architect's own.** ***THE RULE IS RIGHT. ITS OWN EVIDENCE IS THE POPULATION THE
RULE EXISTS TO QUESTION.***

**AMENDED THE SAME NIGHT, TWICE, BY SEATS THE RULE WAS WRITTEN ABOUT.**

    `dev-4` ....... NAMING THE ENTRY POINT IS NECESSARY AND NOT SUFFICIENT. Its own
                    disclosure names a real production path and still has m = 0: the path
                    exists, the input never arrived. ***THE COUNT BELONGS BESIDE THE NAME.***
                    A blocker naming a live entry point it exercised ZERO times is exactly
                    as false as one that names none.
    `dev-2` ....... THE RULE ONLY REACHES A BLOCKER THAT HAD A PROBE. Seven of the eight were
                    a correct number over the wrong population. The eighth was NO NUMBER AT
                    ALL -- a lack of material asserted without running anything against the
                    material. ***A seat in that position answers "name the entry point" with
                    "I had none", ships it UNVALIDATED, and the blocker stands -- correctly
                    labelled and equally false.***

> ***SO: BEFORE YOU REPORT A BLOCKER, STATE THE COMMAND THAT WOULD LIFT IT -- AND RUN THAT
> COMMAND ONCE.*** *`dev-2`'s was a directory listing on a mount. It never typed it. That one
> line covers the case where no instrument ran at all, which is cheaper to commit and harder to
> see than any of the seven. It does not replace the rule above; it covers the eighth.*

**AND A THIRD CLAUSE, FROM `arch-heir`, FOR THE CASE THE FIRST TWO BOTH MISS.**

*Its probe ran on the RIGHT population -- real tracked files, the production set itself. Right
population, right units, right number. What was wrong was the* ***PREDICATE****: the assertion
tested RAW PATTERN MATCH while the instrument decides on match* ***AND*** *classification.* ***THE
TEST TESTED SOMETHING THE INSTRUMENT DOES NOT DO*** *-- and the result is identical to all seven
above, a confident claim about a defect that does not exist.*

> ***NAME THE PREDICATE THE INSTRUMENT ACTUALLY DECIDES, NOT ONLY THE POPULATION IT RUNS ON.***
> *The counter-test that makes this operable is cheap:* **SABOTAGE THE PREDICATE AND CHECK THE
> VERDICT MOVES.** *Run once, it found a third defect nobody was looking for: the positive control
> covered only the pattern, so a DEAD CLASSIFIER returned "no paths" -- the reassuring answer --
> with the control green.* ***FAILING TOWARD CLEAN, FOR THE WHOLE CAMPAIGN.***

**AND THE FIXTURE FORM OF THE SAME THING, FOUND THE SAME NIGHT.** *A blocker was one sentence from
publication -- "the manifest does not detect tampering" -- when the fixture had tampered with an
entry ALREADY IN THE FAILING BUCKET, so no tampering could move any count.* ***THE INSTRUMENT WAS
SOUND; THE FIXTURE COULD NOT FAIL.*** *A second fixture had died silently: meant to be unreadable,
found at mode 644, and quietly a DUPLICATE of the healthy arm with nothing announcing it.*

**AND THE ARCHITECT COMMITTED THE ORIGINAL DEFECT AGAIN WHILE WRITING THIS SECTION.** *It
reported that the refusal-cause token "is emitted nowhere, on no ref" -- measured in the LOCATOR
module, which has never emitted it and never did. The live emitter is in the REPAIR module, at
two call sites, and the architect had read both an hour earlier. `dev-4` caught it before it
propagated:* **"the token is emitted nowhere" is FALSE; "no shipped log in the corpus carries
it" is TRUE** *-- and only the second was ever anyone's claim.* ***A ZERO FROM THE WRONG FILE IS
THE DEFECT THIS SECTION IS ABOUT, COMMITTED IN THE ACT OF WRITING IT DOWN.***

## WHERE THE REST OF THIS FILE WENT, AND WHY

**This document was 1082 lines and a seat has to read it before it can work. The narrative that
DERIVED the rules above — the restart research, the architecture measurement, the reclassification,
the `RUN` ruling's own audit — is now in `docs/MEASURING.MD`, numbering preserved.**

    A DOCUMENT READ BEFORE ACTING PAYS FOR EVERY LINE, EVERY TIME, IN EVERY SEAT.
    THE REASONING THAT PRODUCED A RULE IS WORTH KEEPING AND IS NOT WORTH RE-READING
    ON EVERY RESTART. **RULES HERE. EVIDENCE IN `docs/MEASURING.MD`. ENTRY IN `docs/ENTRY.MD`.**

**AND THE CAMPAIGN LIVES ON `dev-AI` ALONE — owner's ruling, 2026-09-05:** *"tout doit être dans
dev-AI. les agents se calment sur cette branche."* **DO NOT PORT GOVERNANCE TO `master`. `master` is
not where this campaign is decided, and a seat that finds campaign-2 rules there is reading something
that was never promoted.**

## ROUTING: A CRASH OR AN IMPROVEMENT IS A DEV TASK, NOT A CAVEAT

**Owner's ruling, 2026-09-05:** *"le moindre crash / amélioration peut être demandé à un dev."*

    ANY CRASH A MEASUREMENT SEAT HITS, AND ANY IMPROVEMENT IT CAN SEE, GOES TO A
    DEV SEAT AS A TASK. IT IS NOT ABSORBED, NOT WORKED AROUND, AND NOT CARRIED AS
    A LIMITATION ON THAT SEAT'S OWN OUTPUT.

**This changes what a blocker is worth.** *Before it, hitting a crash mid-analysis meant the
measurement got a caveat and the crash got a note.* **After it, a dev gets a task and the fix path is
open.** *A seat that goes quiet for a day is most often a seat holding something with nowhere to send
it — that is the shape this rule exists to end.*

### HOW TO HAND ONE OVER

    THE TYPE. THE SITE `<module>:<line>`. THE ID. *** AND WHICH IMAGE YOU WERE ON. ***
    AND WHAT YOU EXPECTED INSTEAD — a report naming the expected behaviour is a
    TASK; one naming only the crash is a ticket nobody can close.

**NEVER THE EXCEPTION MESSAGE.** *Measured at the authority: all 17 `raise Exception` sites in
`src/mergeVideo.py` interpolate a file path into their message.* **THE TYPE, NEVER THE MESSAGE.**
*`vmsam-dev-4` classified 317 tracebacks without reading one message, and that is the only reason its
output was publishable.*

### AND THE DESTINATION DEPENDS ON THE ZONE, NOT ON WHO FOUND IT

    src/mergeVideo.py · src/video.py ......... FROZEN — route to the Lead
    src/change_point_locator.py .............. dev-1
    src/merge_video_repair|chimeric|resample . dev-2
    src/merge_plan_report.py ................. dev-4
    ci-instruments/ .......................... ci

**A LINE NUMBER WITHOUT ITS BLOB IS NOT A CITATION.** *`vmsam-dev-4` measured that 0 of 317 job-log
artefacts carry any build identity field, so a histogram of crash sites across them is an aggregate
over an unknown number of builds.* **Emit the build identity at EMISSION time, never captured at
startup** — *a field written once at startup reports the same value regardless of what is running,
which is worse than an empty field: an empty field is honest, and a constant one is a lie that
survives every rebuild.*

### NAMING A REF IS NOT NAMING A STORE

**`git show <ref>:<path>` was circulated as the fix for stale checkouts, and it is right as far as
it goes.** *It bypasses the working tree entirely — no update, no reset, no rebase, none of which
git would allow a diverged clone anyway.* ***AND IT IS UNDERSPECIFIED, MEASURED AT A SEAT WHERE IT
FAILED WHILE LOOKING CORRECT:***

    that seat's `origin` URL ................ THE HOP, not the authority
    its cached origin/dev-AI ................ four tip-moves behind
    the hop's dev-AI ........................ current
    the authority's dev-AI .................. the same, by ls-remote
    `git show origin/dev-AI:<path>` there ... *** A THIRD STALE BLOB, DELIVERED
                                              WITH THE CONFIDENCE OF A NAMED REF ***
    and the current object was ABSENT FROM THAT STORE ENTIRELY:
      `cat-file -t <sha>` -> *** fatal: Not a valid object name ***

> ***THREE STALE LAYERS, AND NAMING THE REF FIXES ONE: the blob you hold, the tracking ref you
> cache, and WHAT `origin` POINTS AT AT ALL. A tracking ref answers "what did I last see", never
> "what is there now" — and it answers confidently either way.***

**AND THE HARDER HALF: A CITATION YOU CANNOT RESOLVE IS RELAYED, NO MATTER HOW PRECISE ITS BLOB
IS.** *That seat could not confirm a correct citation from its own clone under any form, because
the object was not in its store. Verifying it required reading a different repository.* **PRECISION
IS NOT PROVENANCE.**

### THE FORM THAT WORKS — TWO PURE READS, RUNNABLE UNDER A FREEZE

    git -C <store> ls-remote <authority-URL> <branch>
    git -C <store> rev-parse <branch>          # *** REQUIRE THESE TWO EQUAL ***
    git -C <store> show <branch>:<path>        # only then is the content current
    # if they differ: report the tip you measured and DO NOT READ THE FILE

***NO FETCH. NO REF WRITTEN. NO `FETCH_HEAD`. NO OBJECTS TRANSFERRED.*** *The fetch-first form is
not runnable under a write freeze —* **and a freeze is exactly when a seat most needs to read the
authority.** *A rule that cannot be followed in the condition it was written for is not a rule.*

*Found and measured by `vmsam-auditor`, correcting a form the architect had broadcast to eleven
seats; the underlying "read from a named ref" rule is `vmsam-dev-4`'s and survives intact —* **it
was right and incomplete, which is a different thing from wrong.** *The wrong form stays in the
record beside the right one:* ***A DELETED DEFECT IS ONE A RE-READER RE-DERIVES.***
