# VMSAM — AI Agent Context & Directives

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
| `AGENT_NOTES.MD` | logbook — decisions and why alternatives lost | when you decide something a successor would re-derive |
| `TASKS/` | the work assigned to you, and what you did with it | at the start and at the end of every task |
| `COMMITTING.MD` | what earns a commit, and what may travel alone | when you are about to commit |
| `TALKING.MD` | who may direct you, and how what you know reaches anyone | when a message arrives, or when you have something to say |
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

## 🔄 Synchronization pipeline

```
merge_videos(files, out_folder, merge_sync=True)
  └─ sync_merge_video                       language selection, MD5 audio grouping
       ├─ get_delay_and_best_video          no reference imposed: tournament of pairs
       │    or get_delay                    forced_best_video: every file against it
       │      └─ prepare_get_delay          extraction of the comparison windows
       │      └─ compare_video (Thread)     one per pair
       │           └─ test_if_constant_good_delay
       │                ├─ first_delay_test        1 · Chromaprint, per window
       │                └─ second_delay_test       2 · FFT / Rust, one long block
       │           └─ get_best_video                3 · PSNR + VMAF, only if no
       │                                                reference is imposed
       │           └─ adjust_delay_to_frame         frame snap
       │      └─ remove_not_compatible_video        4 · refusal  ← repair zone A
       └─ generate_launch_merge_command     ffmpeg passes, then mkvmerge
```

**Why stage 2 is in Rust** (owner's design intent, absent from the code): the FFT
correlation was Python, and it ran out of memory on long files. `audio_sync` is its
optimised rewrite; the Python implementation survives as a fallback. Rust buys
memory, not accuracy — the tolerances above are unchanged by it.

**The rule the whole design rests on: one constant offset for the pair.** It is
enforced in three places, not one — do not mistake the last for the rule:

- `first_delay_test` requires the delays to agree **across windows and across track
  pairs**. Pairs that disagree go to `ignore_audio_couple`; too wide a disagreement
  raises. This is where constancy is actually established.
- `second_delay_test` has its own variance tolerance,
  `max_delay_variance_second_method` = **0.004 s**.
- `test_if_constant_good_delay` then checks that **stage 2's correction stays under
  500 ms**. Stage 2 re-extracts the candidate already shifted by stage 1's answer,
  so it measures a *residual*, not a second independent offset: the check
  `abs(calculated_delay - delay_first_method) < 500` reduces algebraically to
  `abs(delay_second_method * 1000) < 500`.

**Why 500, and why stage 2 at all** (owner's design intent, recorded here because the
code does not say it): fingerprint peaks were observed putting stage 1 off by
roughly **±250 ms**. Rather than trying to filter those peaks, stage 2 runs
unconditionally and *corrects* them — a residual inside the band is applied, one
beyond it means stage 1 was not merely imprecise but wrong, and the pair is refused.
So the threshold is not a confidence margin: it is the width of the error stage 2 is
expected to absorb.

The side effect is precision. Stage 2's own tolerance is 4 ms against a frame of
41.7 ms, so the residual is fine enough to expose **a cut that is one frame longer
than its counterpart** — which is how a structural difference first becomes visible.

Everything downstream assumes that single number is right.

**A delay measured by comparison is snapped to a whole video frame** when the
reference is CFR — 41.708 ms at 23.976 fps, via `adjust_delay_to_frame`. Sub-frame
precision is discarded by design, so never sell a method on it. Three limits go with
that sentence, and each has been mistaken for the rule before:

- **the snap does not survive to the container.** `mkvmerge` is given
  `--sync -1:<int ms>`, so a delay snapped to 792.459 ms ships as 792 — half a
  millisecond off the boundary. Far below the frame quantum, but it means "lands
  exactly on a frame" is not literally true;
- **only comparison delays are snapped.** `adjust_delay_to_frame` is called from
  `compare_video` alone. The MD5-group delays (`delay_same_md5_audio`) are derived
  from container delays and go through unsnapped;
- **VFR is not refused, it is unsnapped.** The non-CFR branch returns the delay as
  measured, with a `TODO`. Nothing rejects a variable-frame-rate reference.

**A refusal is a correct outcome, not a failure.** No constant offset means the
files differ in a way a single delay cannot express, and merging anyway produces a
plausible file that is silently wrong.

What each stage actually does, read from the code, is in `FILE_MAP.MD`.

## 📊 Handled Cases & Core Features

- **Audio Stream Deduplication & Ranking**: evaluates stream properties (codec,
  bitrate, sampling rate, channel layout) against **configurable** priority rules in
  `config.ini` `[mergerules]` — e.g. `Flac>AAC`, `Flac>MP3`, `DTS>E-AC-3*1.1` — to
  retain only the optimal tracks **per language**. `keep_best_audio`.
- **Subtitle Cleaning & Deduplication**: prioritises styled formats (ASS over a raw
  SRT carrying the same text — `not_keep_ass_converted_in_srt`), cleans text
  streams, classifies hearing-impaired and dubtitle tracks
  (`test_if_hearing_impaired`, `test_if_dubtitle`), and drops identical streams by
  MD5. Text formats hash on **text**, image formats (PGS/VOBSUB) on packet bytes.
- **Track Metadata & Flag Assignment**: detects and tags default, forced,
  commentary and audio-description streams — from the container flags *and* from
  the title (`commentary`, `audio description`, `audio guide`, `descriptive`) — with
  ISO 639 compliance via `iso639.Lang`, falling back to the macrolanguage.
- **Multi-Pass Remuxing Pipeline**: FFmpeg extracts and loudness-normalises the
  comparison audio (`loudnorm`, −23 LUFS), applies offsets with `aresample` and
  `-itsoffset`, then `mkvmerge` assembles the container and receives the final
  per-track `--sync` in integer milliseconds.

## 🏷️ The `fabricated` marker

A track that the pipeline **built** is not a track it **copied**, and after the mux
nothing tells them apart. One key carries the difference, on the audio stream dict:

| `fabricated` | |
| :--- | :--- |
| absent or empty | the track is untouched — it came from a source file as it was |
| a non-empty string | the track was **built**, and the string says how: `chimeric`, `resampled:<factor>`, `chimeric+resampled:<factor>` |

**Test truthiness, never equality.** A value you do not recognise still means
fabricated, so a method added later needs no change at any consumer.

On the produced file the same string is a Matroska track tag, `VMSAM_FABRICATED`,
so the origin survives the merge and is recoverable months later:

```sh
ffprobe -v error -show_entries stream_tags=VMSAM_FABRICATED -of compact <file>
```

**Nothing sets the key today** — `keep_best_audio` already reads it (a fabricated
track loses to an intact one before any codec rule) and the branch is inert until
the repair marks its tracks. The full contract is `SPEC_ZONE_A.MD` §4.

**Why it is defined before it exists.** Provenance is knowable only at the moment
of production; after that it is gone. Campaign 1 measured the consequence: **zero of
110 measurements could be attributed to the code that produced them**, and three
separate tools reconstructed track provenance from emission order, from track
counts and from codec — three methods, three answers, no tiebreak.

## 🎯 Target-first — be certain of what you must obtain

You are free on the method. You are not free on knowing what a correct result is
**before** you produce one.

1. **Build the expected file by hand, first — and state what it shares with the code
   under test before believing any agreement.** *A target that shares an input with
   the code under test cannot disagree with it*. That is checkable in advance, which
   "build it independently" is not. Campaign 2 met every clause of
   `SPEC_ZONE_A.MD` §2 on a real file and proved nothing, because the hand-built
   target came from the same plan as the splice; the claim was refused on its own
   result rather than left to stand. Whatever the change, produce the
   output you say the code should produce — manually, with whatever tools you like
   — and check it yourself. This is the target, and it exists before the code does.
2. **Run the pipeline locally against it.** This pass hunts bugs, not correctness:
   crashes, files that never appear, a path never taken.
3. **When the code holds, ask for a run in the test instance.** A local run proves
   your code executes; only a real run proves the pipeline produces the result. You
   do not operate that instance. **Its rules are deliberately absent from this
   repository — you are given them in the prompt.** Copy them into `VMSAM_HELP_AI/`
   the moment you receive them; do not plan around them, and do not infer any of
   them from the runs you have seen.
4. **Compare.** The produced file and your target must be **identical to what you
   expected**. Not close, not plausible — identical on the properties you named in
   step 1.

**A test run is batched, exactly like a commit.** A pass costs the same whether it
carries one line or a finished feature — so ask when you hold something worth the
pass, and state what you expect it to show *before* it runs. And never assume the
instance carries your latest code: name the revision you believe is under test, and
when a result contradicts what you read in the source, question that assumption
before you question the code.

**Why in this order.** A target built afterwards is fitted to whatever came out,
and then any result looks like a success. Campaign 1 has the example: a method was
tuned until its numbers looked excellent — the tightest spread of a whole benchmark,
0.008 ms — while it correlated the wrong language and was wrong by 21 ms. A file you
built and verified yourself cannot be argued with by a plausible number.

**And it is the only check dead code cannot pass.** Five times, code was written,
reviewed, unit-tested green, and never called by anything: every suite passed. A
unit test calls the function directly, which is exactly what production does not do.
Comparing a produced artefact against a target you built proves the path ran.

State the target **and the acceptance condition before running** — including what
result would refute you. A prediction written after the fact is not a prediction.

**And if you dispatch work, the refutation is yours to supply, not to hope for.**
This clause addresses whoever does the work, so a dispatcher can satisfy every word of
it and still hand out a task with no refutation in it — which happened, and the agent
supplied its own, covering for the dispatch. A task without a stated refutation is a
task nobody can lose.

## 🏗️ Where you work — four tiers, one published commit

You do not edit `master` directly. Each tier is a **git clone** of the one above it:

| tier | what it is for | commits |
| :--- | :--- | :--- |
| `/tmp/…` | experiments — try things, break things | as many as you like |
| `VMSAM_WIP/` | what you have validated and are still assembling | as many as you like |
| the **campaign branch** | what CI builds and the test instance runs | as many as you like — **a deploy requires a push** |
| `master` | the published history | **one per feature**, by squash |

**The campaign branch is a tier, not the record.** § Commits binds the last hop
only: testing costs a push, and a rule that made you hesitate to test would be
worse than a noisy branch. Commit there as freely as the work needs.

**Clones, never copies.** A promotion by `cp` overwrites without warning, and the
owner commits while you work — measured on this repository:
a copy silently erased an owner change to the same file, while the same promotion
through git stopped on the conflict and preserved it.

```sh
# promote /tmp -> VMSAM_WIP : keep the history, it costs nothing here
git fetch <tmp-clone> <branch> && git merge FETCH_HEAD

# promote VMSAM_WIP -> campaign branch : ONE commit per promotion
git fetch <wip-clone> <branch> && git merge --squash FETCH_HEAD && git commit && git push

# campaign branch -> master : one commit, whatever the history behind it
git fetch <wip-clone> <branch> && git merge --squash FETCH_HEAD && git commit
```

`--squash` is what makes § Commits structural instead of a rule to remember: the
command cannot produce two commits.

**But it only fixes the mechanism, not the cadence.** Campaign 2 promoted three times
in twenty-six minutes, each promotion correctly one commit — and two of the three
were halves of a single increment: the locator fixed, then sixteen minutes later the
locator wired to its consumer. The letter held and the intent did not; the
granularity had simply moved from commits to promotions.

**Promote when the branch needs to carry it** — when it must be tested in the
container, or when another agent needs it to proceed. Not per fix, and never on a
rhythm. If your next promotion continues the last one, it was one promotion.

**It is used at two hops, and the second one is why.** Campaign 2's first day put 79
commits on the branch, 73 of them documentation alone — and an audit found that
**nobody had broken a rule**. Commits are free in the tiers above, correctly; the
promotion preserved history, also correctly; and the two compose into a branch where
six real pieces of work sit among seventy-three entries about thinking. Two correct
rules composing badly is not fixed by asking anyone to try harder. Squashing at the
promotion fixes it structurally: **one commit per promotion**, and you promote when
you have something worth testing, so the branch reads as a list of increments. The
granular history stays in your `VMSAM_WIP` clone, where it costs nothing and belongs.

**Pull down often, push up rarely.** Sync the tier above *into* the one below
before promoting, and `VMSAM_WIP` into `/tmp` when you start something. Divergence
is cheap to resolve daily and expensive to resolve once.

**Run `WRITE_ZONES.MD` §5 in `VMSAM_WIP`, not at the promotion.** The checks take a
second. Discovering at the last hop that half your work sits outside a tagged zone
is how a campaign ends up negotiating its own rules.

Never write a tier's filesystem path in a tracked file — Privacy, below.

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

## 🧪 Believing a result

**Which paths hold test material, where outputs go and what survives a restart are
facts about the machine, not about VMSAM** — they change without the code changing,
and a stale path in a tracked file is worse than none. They live outside the
repository, with the rest of the environment. `cargo` and `compileall` belong to the
pre-commit checks in `WRITE_ZONES.MD` §5, which is where they are run.

What is left here is the part that stays true whatever machine you are on: **the ways
a result can look sound and be worthless.** Each of these was paid for.

**A unit test cannot tell you that nothing calls the unit.** Campaign 1 shipped five
pieces of code — a guard, a refined measurement, a policy branch, an order check, a
subtitle repair — that were written, reviewed and unit-tested green, and that nothing
ever called. Every suite passed. A test calls the function directly, which is exactly
what production did not do. Any new guard needs a test that pins its **call site**.

**One turn further: the unit was called, and the transport between them dropped the
field.** `parse_segments` rebuilt each segment from three named keys and discarded
the rest, including the per-stream offsets. Emitter correct, consumer correct,
integration inert — and the unit test passed throughout, because it handed the
consumer a hand-built dict and never crossed the transport.

The cost was **28.88 ms of misalignment**, which passed under four gates: below the
video frame, below `adjust_delay_to_frame`'s snapping, below `mkvmerge`'s integer
milliseconds, and below the repair's own 100 ms tolerance. Only a per-track check
saw it. It would have shipped in silence.

**So: a field whitelist in a transport is a defect by construction.** It makes every
future extension of the format inert, and the only symptom is that nothing changes —
which is the one symptom nobody investigates. Carry the whole structure and let the
consumer ignore what it does not need.

**Which rules deserve to be mechanical, and it is not the important ones.** The rules
that survived today are the ones whose violation produces an immediate, local,
unambiguous failure in the actor's own frame. The checker refuses a commit and you
feel it at once — it is the only rule nobody has broken. A waiter never returns and
you feel it six hours later, which is why it needed a check rather than a sentence. A
report written but not sent produces **nothing at all**: no error, no delay, no
symptom in the writer's world, which is why no wording fixes it.

**So: mechanise the rules whose failure is silent in the actor's own frame — not the
ones that matter most, the ones the actor cannot feel going wrong.** Everything else
can be a sentence.

**And an armed guard that has never fired is indistinguishable from a working one.**
Campaign 1's success metric sat in the third branch of a function nothing reached:
315 rows exited earlier, and `Resolved: 0` was read as *nothing qualifies* when it
meant *nothing reached the check*. Fire every guard deliberately, with literals,
plus a case proving it can return the **other** answer.

