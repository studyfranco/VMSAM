# VMSAM — AI Agent Context & Directives

> ### THIS FILE IS REFERENCE. **`ENTRY.MD` IS WHAT YOU READ FIRST.**
> *The mandatory documents total ~7 000 lines and this one is 1 000 of them.* **NOBODY
> READS 7 000 LINES BEFORE THEIR FIRST ACT** — and "read AGENT.MD first" was a rule
> with no mechanical check, which §34 says is a habit. **`ENTRY.MD` is the
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
| `PIPELINE.MD` | **the engine, the handled cases, the `fabricated` marker, and TARGET-FIRST** — what a good output IS | before changing engine behaviour |
| `MEASURING.MD` | **the evidence.** Every rule's derivation, every refutation, every retraction. `grep` it, never read it through | when you doubt a rule, or are about to repeat a mistake |
| `ENTRY.MD` | the only document read BEFORE acting, and the entry test | first, and after every restart |
| `AGENT_NOTES.MD` | logbook — decisions and why alternatives lost | when you decide something a successor would re-derive |
| `TASKS/` | **finished reports** — one per closed group of actions: context, result, technical account. **Not live state**, which lives in `VMSAM_HELP_AI/<agent>/` | when a group of actions closes |
| `COMMITTING.MD` | what earns a commit, and what may travel alone | when you are about to commit |
| `TALKING.MD` | who may direct you, and how what you know reaches anyone | when a message arrives, or when you have something to say |
| `MEASURING.MD` | the ways a result can look sound and be worthless | when you hold a number and are about to believe it |
| `COMING_BACK.MD` | the procedure for resuming after a stop | **only** when you were cut off mid-work |
| `docs/` | standing technical rulings — what a correct merge is, which speed method, which subtitle codecs can be re-timed | when your change touches one |

**These documents are frozen — `AGENT_NOTES.MD` and `TASKS/` excepted, and this file
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

## 📋 `TASKS/` — the work, and its history

```
TASKS/
├── <NNN>-<slug>.MD     an open task: the owner states the goal, you fill in the rest
└── done/               closed tasks, moved here — the campaign's history
```

One task, one file, one number. You fill it in as you go, not at the end:

- **what you changed** — files and zones, so a reviewer goes straight there;
- **what you measured** — the numbers that show it works, never the claim that it does;
- **what you dropped** — an approach tried and abandoned is worth more than silence:
  it stops a successor re-deriving it;
- **what stayed blocked** — the findings of `WRITE_ZONES.MD` §6.

A task moves to `done/` when its goal is met **and** its commit exists — not when you
stop working on it. An abandoned task stays open and says why it was abandoned.

## 📁 `VMSAM_HELP_AI/` — where private material goes

A directory **beside the repository, never inside it**. Create it if it is missing;
it is never committed and never referenced by path in a tracked file.

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

## 🔄 The engine, the cases, and what a good output is → `PIPELINE.MD`

**The synchronisation pipeline, the handled cases, the `fabricated` marker, and TARGET-FIRST — be
certain of what you must obtain before you touch anything — are in `PIPELINE.MD`.**

    THEY ARE DOMAIN REFERENCE. YOU CONSULT THEM WHEN THE WORK NEEDS THEM.
    THEY ARE NOT READ-BEFORE-EVERY-ACTION, AND THEY WERE COSTING EVERY SEAT
    EVERY LINE ON EVERY RESTART.
    `git rev-parse -q --verify origin/dev-AI:PIPELINE.MD` -- RESOLVE IT YOURSELF.
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

## The architect, and the entry test → `MEASURING.MD` and `ENTRY.MD`

**Why the architect seat is permanent and absent from your roster: `MEASURING.MD`, final section.
THE ENTRY TEST — mandatory, and a restart is cheap: `ENTRY.MD`, final section.**

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

*The ruling's full derivation is in `MEASURING.MD`'s appendix.*

## 📏 Believing a result

**`ENTRY.MD` §2 is the authority: the five kinds, the `RUN` rule, and `m`.** *Two things repeated here
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

## VERIFICATION KINDS — THERE ARE **FIVE**, AND `ENTRY.MD` IS THE AUTHORITY ON THEM

> **THIS FILE SAID *FOUR* AND OMITTED `RELAYED` — the kind this fleet uses most. A seat reading only
> this document would never learn it exists.** *Corrected 2026-09-05; the full definitions, the `RUN`
> ruling and the honest audit that produced them now live in `ENTRY.MD` §2 and `MEASURING.MD`.*

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

## WHERE THE REST OF THIS FILE WENT, AND WHY

**This document was 1082 lines and a seat has to read it before it can work. The narrative that
DERIVED the rules above — the restart research, the architecture measurement, the reclassification,
the `RUN` ruling's own audit — is now in `MEASURING.MD`, numbering preserved.**

    A DOCUMENT READ BEFORE ACTING PAYS FOR EVERY LINE, EVERY TIME, IN EVERY SEAT.
    THE REASONING THAT PRODUCED A RULE IS WORTH KEEPING AND IS NOT WORTH RE-READING
    ON EVERY RESTART. **RULES HERE. EVIDENCE IN `MEASURING.MD`. ENTRY IN `ENTRY.MD`.**

**AND THE CAMPAIGN LIVES ON `dev-AI` ALONE — owner's ruling, 2026-09-05:** *"tout doit être dans
dev-AI. les agents se calment sur cette branche."* **DO NOT PORT GOVERNANCE TO `master`. `master` is
not where this campaign is decided, and a seat that finds campaign-2 rules there is reading something
that was never promoted.**
