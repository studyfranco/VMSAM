# VMSAM — AI Agent Context & Directives

**This file governs `master`.** The `dev-AI` campaign had its own copy with its own
rules; do not apply them here. Documents ending in `_DEV_AI` are that campaign's
archive — measured knowledge to consult, not instructions to follow.

Budget: **250 lines.** A rule that does not fit is not a rule, it is a finding, and
it belongs in the archive. This ceiling exists because the previous campaign's
equivalent reached 1191 lines and stopped being read before acting.

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

**You write exactly one of these: `AGENT_NOTES.MD`.** Every other document here is
the owner's, **this file included**. Campaign 1's `AGENT.MD` carried the line *"You
are expected to update this `AGENT.MD` whenever you implement a new feature"* — it
reached 1191 lines, and the campaign shipped more documentation than engine. An
agent that may rewrite what binds it eventually rewrites itself out of it.

So: a rule that is **wrong, missing, or blocks what `CAMPAIGN.MD` asks of you** is a
**finding**, not an edit. Write it in `AGENT_NOTES.MD`, state it in your report, and
keep working elsewhere. The owner arbitrates and edits. This applies to a rule you
believe is simply mistaken — being right does not make it yours to change.

Campaign 1's archive, suffixed `_DEV_AI`. **Consult, do not follow:**

| file | what it actually contains |
| :--- | :--- |
| `FINDINGS_DEV_AI.MD` | 23 measured findings — the most reliable of the three. 13 were withdrawn or superseded and say so |
| `DOMAIN_INVARIANTS_DEV_AI.MD` | 136 entries, and the title oversells them: a **minority** are VMSAM domain rules (which file is the master, the delay sign convention, per-pair chromaprint resolution, "a constant offset is not the normal case in this corpus"); the **majority** are general research method. 15 were later corrected. Grep it, do not read it |
| `MERGE_VIDEO_ARCHITECTURE_DEV_AI.MD` | how the merge path is built — the most directly useful before touching the engine |

None of it is authority. It was written by agents under time pressure, and the
corrections it carries are the reason to trust the numbers more than the conclusions.

## 📁 `VMSAM_HELP_AI/` — where private material goes

A directory **beside the repository, never inside it**. Create it if it is missing;
it is never committed and never referenced by path in a tracked file.

Put there anything the repository must not carry: real media paths and filenames,
show titles, season/episode names, TVDB/TMDB ids and any external catalogue id,
extracts or fragments of media files, session links, host paths, and the mapping
from an opaque error id to the file it designates.

In a tracked file, cite the opaque id and write *"details in the private notes
beside the repository"*. That sentence is enough; the id is the key.

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

## 🎯 Target-first — be certain of what you must obtain

You are free on the method. You are not free on knowing what a correct result is
**before** you produce one.

1. **Build the expected file by hand, first.** Whatever the change, produce the
   output you say the code should produce — manually, with whatever tools you like
   — and check it yourself. This is the target, and it exists before the code does.
2. **Run the pipeline locally against it.** This pass hunts bugs, not correctness:
   crashes, files that never appear, a path never taken.
3. **When the code holds, run it in a test instance.** A local run proves your code
   executes; only a real run proves the pipeline produces the result.
4. **Compare.** The produced file and your target must be **identical to what you
   expected**. Not close, not plausible — identical on the properties you named in
   step 1.

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

## 🔐 Privacy — the repository is potentially public

Container and tooling paths are fine: `/config`, `/config/output`, `/tmp`,
`/home/vmsam/src`. **Never** in a tracked file, a commit message or a branch name:

- the media library root, or the error/sync trees that mirror it;
- **any media filename** — a real title with `.mkv`/`.mka`/`.mp4`/`.ass`/`.srt`.
  Names the pipeline generates for its own temp files are fine;
- show titles, season/episode names, release-group identifiers.

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

**Database and configuration structures are frozen** — see `WRITE_ZONES.MD`.

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

## 🌐 API architecture (`gestionar_show` — frozen)

Two uvicorn instances, both forked by `main_gestionar_show.py`:

| instance | bind | workers | owns |
| :--- | :--- | :--- | :--- |
| public | `0.0.0.0:8080` | 5 | external routes, database validation. Stateless. |
| internal | `127.0.0.1:42085` | **1** | the fusion queue, the worker, execution settings |

`workers=1` on the internal instance is **mandatory**: the queue lives in process
memory, and at that setting uvicorn serves in-process so the app inherits the forked
`tools` globals. Raising it splits the queue *and* empties `tools.software`.

The public app must not import `fusion`, directly or transitively — it would pull
`mergeVideo` → `audioCorrelation` → numpy/scipy into all five workers. Verify with an
import-closure walk, not by grepping one file: the first violation hid one module
deep, inside the proxy client.

The proxy uses stdlib `urllib` with an **empty `ProxyHandler`** — the image ships no
HTTP client library, and bare `urlopen` honours `http_proxy` even for loopback.

## 🧪 Testing and Execution

Validate your code with files in `/config/example/`. Each folder is a couple of files to sync together. Save your tests file in `/config/output` and yout temporary files in `/tmp`
Rust compilation via `cargo build` / `cargo check` is allowed and recommended.

`python3 -m compileall -q src -x '__pycache__'` must fail on exactly one file:
`frame_compare.py`, pre-existing since `da1c191`. Any second failure is yours.

**A unit test cannot tell you that nothing calls the unit.** Campaign 1 shipped five
pieces of code — a guard, a refined measurement, a policy branch, an order check, a
subtitle repair — that were written, reviewed and unit-tested green, and that nothing
ever called. Every suite passed. A test calls the function directly, which is exactly
what production did not do. Any new guard needs a test that pins its **call site**.
