# Agent notes — logbook

One entry per decision that a later reader would otherwise have to re-derive.
Newest first.

**What goes here:** an architectural choice and why the alternatives lost; a
measurement that changed a decision; a defect found and what it taught; a blocked
need and what was done instead.

**What does not:** progress reports, task status, anything regenerable from the
code or from `git log`. If it is a measured fact rather than a decision, it belongs
in a findings document.

**Format.** `## YYYY-MM-DD — one line saying what changed`, then the decision, then
*why the alternatives lost*. Numbers carry the command that produced them. Files are
cited by opaque id, never by media name (`AGENT.MD` → Privacy).

Entries are **append-only**. Superseding an earlier one means adding a new entry that
says so and why — never editing the old one. A record that can be rewritten is not a
record.

---

## 2026-09-03 — correction: I treated the 125 ms correlation point as a constant, and it is not

**This supersedes the arithmetic, not the conclusion, of the baseline entry below.**
Entries here are append-only, so the earlier text stands and this says what is wrong
with it.

I wrote that the ten stage-1 delays on error id 237 were "an exact multiple of
125 ms — the chromaprint frame", and rested "the instrument is not at fault" on it.
The observation was right and **the reason was not mine to assert**: I took
"1 point = 125 ms" from `BRIEF_COMMON` and used it as a universal quantum.

**dev-1 measured otherwise.** `audioCorrelation.py:132` computes the ms-per-point
**per call** — `int(lengthFile/len(fingerprint_source)*1000)` — and the result
ranges **124 to 142 ms**, reaching exactly 125 only when the shortest audio runs
**18.7 to 30.4 minutes**. That is dev-1's measurement and their band; I carry it
with their hedge, have not re-derived it, and do not restate it more strongly than
they did.

**Why the conclusion survives anyway.** This pair's shorter file is 1438.04 s =
**23.97 min**, inside that band, so 125 ms is the expected point size *for this
pair*. The delays do sit on the quantum and the instrument was behaving. What
changes is the warrant: "this pair's point size is 125", not "125 is the quantum".

**What it costs elsewhere, and this is the part worth keeping.** `BRIEF_COMMON`
offers the 125 ms figure as the number that will save an hour, and says *"if a delay
is not a multiple of 125, question the reading before you question the media"*. On
dev-1's measurement that heuristic is only valid for pairs whose shorter audio falls
in an 11.7-minute window — and a 47-minute double episode or a 12-minute short would
have a different point size, so a perfectly good reading would be questioned for
failing a test that does not apply to it. **A finding for the owner, not an edit:**
the document is frozen and it is dev-1's measurement to report, not mine to
generalise. Recorded here because I acted on the wrong version of it and someone
else will.

## 2026-09-03 — the deploy loop's "wait ~10 min" becomes a checked precondition

The documented loop warns that a hook fired before CI publishes the image re-pulls
the previous image and redeploys the *old* commit — a success-looking deploy that
silently invalidates the test. The brief names it the single failure mode that will
waste the most of this campaign's time.

**The instruction was a duration, and a duration cannot be checked.** The published
image can be: it carries the commit it was built from, in the OCI label
`org.opencontainers.image.revision` and in its baked `VMSAM_GIT_COMMIT` — the very
value `/health` will later report. Both are readable before anything is fired.

So `VMSAM_HELP_AI/ci/deploy.py` waits until the published image **is** the target
and refuses to fire the hook otherwise. **Why the alternative lost:** sleeping ten
minutes is wrong in both directions — it wastes nine minutes on a cached build and
is still too short for a cold one, and in neither case does it *know*. The registry
answers the actual question.

Measured while confirming it: a documents-only push published in **~90 s**. That is
a cache artefact, not a new figure for CI — a real `src/` change rebuilds layers,
and a cold build remains unmeasured. Recorded so nobody quotes 90 s back.

**Where it lives:** beside the repository. `WRITE_ZONES.MD` §4's test — would this
still matter to someone running VMSAM in production? — says no for a deploy
harness, the same ruling as `check_write_zones.py`.

## 2026-09-03 — finding: `git_commit` has two unverifiable states, and the documents describe one

`tools.py:226` returns `os.environ.get("VMSAM_GIT_COMMIT", "").strip() or "unknown"`,
while `Dockerfile:137` declares `ARG VMSAM_GIT_COMMIT="IDK"`. So:

- an image built **without** `--build-arg VMSAM_GIT_COMMIT` carries the ARG default
  and `/health` reports **`IDK`** — fix belongs in CI;
- `unknown` is only reachable when the variable arrives **empty or absent** at
  runtime, which `Dockerfile:163` already does and the compose file overrides — fix
  belongs in compose.

Both the private deployment note and the campaign brief say `unknown` means the
build arg was not injected. Read against the code, that case produces `IDK`.

**Consequence: none for correctness of results.** Both states are *unverifiable*,
and the rule that matters — no result obtained in that state proves anything — is
unchanged. The distinction is worth carrying because the two have different causes
and different fixes, and a reader who sees `IDK` will otherwise search the compose
file for a variable that is not the problem.

Neither state has been observed on this instance. Both are coded from the source
and fired with literals, not from a sighting — an armed guard that has never fired
is indistinguishable from a working one.

**This is a finding for the owner, not an edit.** The documents concerned are the
owner's private notes and the brief; being right about them does not make them
mine.

## 2026-09-03 — finding: `fusion_enabled` is computed one hop away and dropped

`internal_api.get_internal_health` returns `fusion_enabled`, which is `False`
exactly when the worker refused to start because `VMSAM_TEST_OUTPUT_DIR` was empty
in test mode (`internal_api.py:52`). The public `GET /health` calls that endpoint
and keeps only `is_running` (`api.py:359-366`).

From outside the instance, therefore, **`is_running: false` with an idle queue means
either "the worker is healthy and has nothing to do" or "the worker never started"**
— and `GET /fusion` cannot separate them either, because
`get_internal_fusion_status` carries no `fusion_enabled` guard and reports an idle
queue in both cases.

This is the campaign's recurring shape: a reader with two outcomes describing a
world with three, and the missing outcome is *the instrument did not run*.

**Not blocking, and what was done instead.** The worker is running — proven
positively by a fusion job reaching `apply_test_outcome` and depositing an artefact.
The cheap probe when in doubt is `POST /fusion`, which answers 503
`Fusion is disabled: VMSAM_TEST_OUTPUT_DIR is not set in test mode` in the disabled
case. `src/gestionar_show/` is frozen in full (`WRITE_ZONES.MD` §1), so adding one
key to the public payload is an owner decision. Recorded, not routed around.

## 2026-09-03 — the first baseline run refused correctly, for a reason nobody had listed

Error id 237 was queued as the campaign's baseline, before any change. It refused in
~30 s at `mergeVideo.py:798` — the code path zone A is being built into — with ten
stage-1 delays spread over 421 s, flipping sign, at mean fidelity 0.567. Every delay
is an exact multiple of this pair's correlation point size, so the instrument was
behaving; there is simply no constant offset to find. **The refusal is correct.**
(The point size is 125 ms **here** and is not a constant — see the correction entry
below. My original note said "125 ms, the chromaprint frame" as though it were.)

The reason is not any of the five families in `CAMPAIGN.MD`. **The file is a
different television series from its folder's master** — verified by extracting
frames from both at the same timestamps and looking at them, not inferred from the
release name. Two different programmes have no true correlation peak, which is
exactly the signature observed.

And it is systematic: of the 49 error entries in that folder, **26 name themselves
as the other series in a contiguous 26-episode run** — 8.3 % of the whole 315-entry
corpus — with no folder registered anywhere for them. One file verified visually;
the other 25 attributed from the naming pattern. The other 37 folders are unchecked,
and how campaign 1 classified any of this is unknown to me and not asserted.

**Why this is a decision and not just a measurement.** The repair fires at the
moment of refusal, and these files refuse. `SPEC_ZONE_A.MD` as drafted does not
distinguish *two cuts of one episode that need reassembly* from *two different
episodes that must never be spliced*: both arrive as "stage 1 found no constant
offset". No repair exists yet and its authors may well plan a guard — the point is
that the family most needing one is absent from the table the campaign is working
from, and it stayed invisible until a real file was actually run. That is the case
for baselines before changes, made concretely.

Error id 237 is therefore offered as a **negative fixture**: a real corpus file,
reproducible in ~30 s, whose correct outcome is *refuse*. Any repair should be run
against it to show it can still say no. Details in the private notes beside the
repository.

## 2026-09-03 — `check_write_zones.py`: the rules get an instrument, and it lives outside the repository

`WRITE_ZONES.MD` §2 rule 2 says a zone is real only if it is checked, and lists
four shell snippets. Three of them are read by eye. Campaign 1 pushed a commit
with a failing test by gating on the wrong thing and reading a suite's output
instead of its exit status (`CAMPAIGNS/01-dev-AI/RESTART.MD` §8), so the checks
are now one command with one exit status.

**Where it lives: beside the repository, not in it** —
`/home/vmsam/src/VMSAM_HELP_AI/tools/check_write_zones.py`. §4's test is *would
this still matter to someone running VMSAM in production?* A harness that reads
`git diff` and enforces a campaign's write rules would not. Campaign 1 left 148 773
lines of exactly this kind of scaffolding under `src/`, nineteen times the program.
One copy outside, called by absolute path from every clone, also means the four
agents cannot drift onto four versions of the rule.

**Why the naive form of the zone check is wrong.** `git diff --cached -U0` read by
eye compares changed lines against the zones of the *new* file. A pure deletion has
no new-side line at all, so a `-` hunk removing a frozen line above a zone reads as
"nothing added outside a zone" and passes. The checker maps removals onto the *old*
blob's zones and additions onto the *new* blob's, separately. The self-test case
`outside zone: pure deletion below the zone` is that defect, pinned.

**Two more holes the eye version leaves open**, both now refused: moving an `END`
marker down to swallow a line into a zone (the diff lands inside the new zone and
looks legal), and creating a fresh marker pair — §2 says adding a zone is an owner
decision, so any change in marker count fails.

**It answers three things, not two.** Exit 2 means the checker could not run and is
*not* a pass. `RESTART.MD` §3.4 found the same shape five times in one morning: a
reader with two outcomes describing a world with three, and the missing outcome is
always *the instrument did not run*.

**Fired, both answers, before being believed.** `--self-test` builds throwaway
repositories and drives 27 cases; each refusal has a matching case that must be
accepted. On the real tree at `f0e7119`: clean; a one-line insert at
`mergeVideo.py:1` refused; the same insert at `mergeVideo.py:795`, inside zone A,
accepted. An armed guard that has never fired is indistinguishable from a working
one, and that includes this one.

**What it deliberately does not check**, so that a pass is never over-read: that a
new module is called by anything; that a file belongs in the repository at all
(§4 is a judgement, not a path rule); privacy; `config.ini` semantics.

## 2026-09-03 — finding: `keep_best_audio` is listed as a tagged zone and carries no tag

`WRITE_ZONES.MD` §2 lists `mergeVideo.py` / `keep_best_audio` among the existing
zones, and the paragraph above the table cites it as the reason the comment form of
the marker exists: *"`keep_best_audio` uses the comment form for exactly this
reason."* In the tree at `f0e7119` there is no marker of either form in that
function:

```sh
grep -n "AGENT modification" src/mergeVideo.py
# 663, 667, 674, 678, 682, 690, 789, 800 — all in adjust_delay_to_frame and
# remove_not_compatible_video. Nothing near keep_best_audio (1011).
```

The `fabricated` branch the entry refers to is present and correct
(`mergeVideo.py:1036-1039`), so what is missing is the permission, not the code.
By rule 1 — a path that is not open is closed — the function is frozen, and
`check_write_zones.py` enforces that reading.

**Consequence, and what was done instead.** Nothing is blocked today: the branch is
already written, and `SPEC_ZONE_A.MD` §4 says it activates by itself the day the
repair sets the key on the audio dict elsewhere. So the campaign proceeds without
the zone. It becomes blocking only if the `fabricated` comparison itself has to
change — a case we do not expect and will report if we meet it.
**This is a finding for the owner, not an edit** (`AGENT.MD` § Documents): the fix
is either two marker lines in `mergeVideo.py` or one line struck from the §2 table,
and both are owner decisions.

## 2026-09-03 — finding: the Rust sources are frozen by rule and editable by instruction

`src/lib.rs`, `src/main.rs` and `Cargo.toml` appear in neither `WRITE_ZONES.MD` §1
(frozen) nor §4 (open). Rule 1 settles it — *a path that is not listed as open is
closed* — and `check_write_zones.py` treats them as frozen.

`AGENT.MD` § Coding Directives points the other way: *"Rust (`audio_sync`) must
pass `cargo check`, `cargo clippy`, `cargo fmt` cleanly before submitting any
changes"*, and § Testing says `cargo build` / `cargo check` are "allowed and
recommended". A file nobody may change cannot have changes submitted, so the two
documents disagree.

**What was done instead:** the strict reading. Nothing in the first objectives needs
the Rust core — stage 2 is the instrument we measure *with*, not the one under
repair — so no work is blocked. Recorded rather than resolved, because
`AGENT.MD` § Documents is explicit that being right about a rule does not make it
ours to change.

## 2026-09-02 — master prepared for campaign 2

`gestionar_show` imported from `dev-AI` and corrected: the dual-uvicorn split, a
fixed internal port (`tools.internal_api_port = 42085`), and `fusion.py` rewritten
from 1334 to 381 lines — the queue and the merge kept, the campaign instrumentation
dropped.

Twelve defects found and fixed while wiring it, none of which the compiler could
see: `get_git_commit` raising `NameError` on first call, three `tools.*` attributes
referenced but never defined, `_env_flag` used for a path (it returns a boolean),
nine `tools.logs` writes leaking into thread contexts through cross-file
`Thread(target=…)` declarations, `.get(0)` on a list, a `Decimal`/`float` mix that
made the no-forced-best-video path drop every candidate *silently*, a lost-master
window in the production swap, a pool never shut down, and a production failure
that left no trace anywhere.

Two hazards were closed structurally rather than by a fix: the episode lock, because
the integration loop and the fusion worker are separate processes that can write the
same episode; and `tools.load_merge_runtime_from_env()`, because Python 3.14
defaults to `forkserver` and a forkserver child sees only module defaults — which
would have put the two processes' locks in different trees, silently, while
`software={}` failed every ffmpeg call.

Write zones were declared and the first tags placed. The rule that governs them —
*a unit test cannot tell you that nothing calls the unit* — comes from campaign 1
shipping five pieces of code that were reviewed, tested green, and never called.
