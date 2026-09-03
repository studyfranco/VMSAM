# 002 — The test container: deploy loop, fusion queue, results

**Agent:** `vmsam-ci`. **State:** open.

## Goal

Own the path from a pushed commit to a comparable artefact, and keep it working
for the other three. Nobody else fires the hook or queues a fusion.

1. **Confirm the deploy loop end to end** and write the confirmed procedure into
   `VMSAM_HELP_AI/ci/`. The rules are in the private notes beside the repository.
2. **Confirm `VMSAM_TEST_OUTPUT_DIR`.** `/config/output` is an *inference* from
   the empty `/config/output/srv/` and from how the output path is derived. Not a
   measurement. A run settles it.
3. **A baseline run before anyone changes anything.** Pick one error file, queue
   it, collect what comes back, and record what the pipeline does *today*. State
   what you expect before the run.
4. **Serialise the queue.** One job at a time; the others ask you.
5. **Any bug visible enough to stop the others** is yours to fix or to route.

## Standing rules

- CI takes **~10 min** from push to image. A hook fired early redeploys the old
  image and looks like a success — that is the failure mode that silently
  invalidates a test.
- Never read a SHA mismatch as a failure of the code under test.
- A test run is batched like a commit: ask for it when it carries something worth
  the pass, and **state what you expect it to show before it runs**.

---

## What changed

Nothing in the repository except this file and `AGENT_NOTES.MD`. **No engine code
was touched**, which is the point: goals 1–3 exist to establish what the pipeline
does *before* the first change lands.

The instruments live beside the repository (`WRITE_ZONES.MD` §4 — a deploy harness
is not something a production run of VMSAM needs):

| file | what |
| :--- | :--- |
| `VMSAM_HELP_AI/ci/deploy.py` | the deploy loop as one command with one exit status |
| `VMSAM_HELP_AI/ci/DEPLOY_LOOP.md` | the confirmed procedure, each claim marked measured / from-document / from-code |
| `VMSAM_HELP_AI/ci/RUN_SERVICE.md` | how the other three ask for a run (goal 4) |
| `VMSAM_HELP_AI/ci/runs/` | per-run prediction and result files |

## What was measured

All 2026-09-03, against deployed commit `db92f0e`.

### Goal 1 — the deploy loop, confirmed end to end

Deployed `6f71af4` → `db92f0e` and verified it. The pushed change was
documents-only, so the loop was exercised without risking a conclusion:
`git diff --name-only 6f71af4 db92f0e` touches no `src/`, `Dockerfile`, `run.sh`,
`init.sh`, `Cargo.toml`, `.github/` or `.forgejo/` path.

| | |
| :--- | :--- |
| `/health` before | `git_commit 6f71af4…`, `mode test`, `dev false`, `is_running false` |
| `/health` after | `git_commit db92f0e…`, same mode/dev |
| `GET /errors` before / after | **315 entries, 38 folder ids, identical id sets** |
| fusion queue before / after | idle, empty |

**The corpus survives a container recreate** — same 315 ids on both sides. The
database is not in the image. Worth knowing before anyone reads a corpus change as
a side effect of a deploy.

### Goal 1a — step 2 is now a measured precondition, not a timer

The documented loop says "CI takes ~10 min; do not fire the hook early", and a
hook fired early redeploys the previous image and **looks like a success**. A
duration cannot be checked. The published image can: it carries the commit it was
built from, in `org.opencontainers.image.revision` and in its baked
`VMSAM_GIT_COMMIT`, both readable before firing anything.

`deploy.py` waits until the published image **is** the target and refuses to fire
otherwise. The campaign's most expensive failure mode is now a refusal with a
message instead of a plausible-looking wrong result.

Measured: a documents-only push published in **~90 s**, not ~10 min — the CI
workflow uses a layer cache and that push changed no layer. **This is not "CI takes
90 s".** A real `src/` change rebuilds layers and the ~10 min figure is the one to
expect. A cold build has not been measured. The point is that the wait no longer
has to be guessed in either direction.

### Goal 1b — step 4 has four outcomes, not two

`tools.py:226` is `os.environ.get("VMSAM_GIT_COMMIT", "").strip() or "unknown"`;
`Dockerfile:137` is `ARG VMSAM_GIT_COMMIT="IDK"`. So there are **two distinct
unverifiable states with different causes**:

| `git_commit` | verdict | cause | fix |
| :--- | :--- | :--- | :--- |
| the target SHA | deployed | — | — |
| `IDK` | **unverifiable** | built with no `--build-arg VMSAM_GIT_COMMIT` | CI |
| `unknown` | **unverifiable** | the variable arrived **empty** at runtime | compose |
| another SHA | **not deployed** | image not published when the hook fired | re-run, do not re-push |
| no answer | container did not come back | — | re-fire |

Neither unverifiable state has been observed here. They are coded from the source
and fired with literals, **not** from a sighting. See the finding in
`AGENT_NOTES.MD`.

### Goal 1c — the guards, fired both ways

*An armed guard that has never fired is indistinguishable from a working one*, and
that includes mine. Eight guards fired deliberately, each refusal paired with a
case that must be accepted:

| guard | refuses | accepts |
| :--- | :--- | :--- |
| image precondition | target not the published tag → **hook not fired**, exit 1 | target is the published tag → exit 0 |
| target validation | a short SHA → exit **2** | a 40-char SHA → proceeds |
| health verdict | `unknown` / `IDK` / mismatch / silence → exit 1, each with its own message | target SHA → exit 0 |

Exit **2** means the instrument could not run, and is **not** a pass.

### Goal 2 — `VMSAM_TEST_OUTPUT_DIR` = `/config/output`, confirmed

No longer an inference. Two independent steps:

1. **It is set at all.** `internal_api.py:52` refuses to start the fusion worker in
   test mode when the variable is empty, and the endpoint then answers 503
   `Fusion is disabled`. The `POST` returned **200 and queued**, so the worker had
   started, so the variable was non-empty at container start.
2. **Its value.** `fusion.apply_test_outcome` writes to
   `os.path.join(get_test_output_dir(), destination_path.lstrip(os.sep))`. The
   artefact appeared at `/config/output/` + that suffix, so `get_test_output_dir()`
   returned `/config/output`. `/config/output/srv` held **0** entries before the
   run and **5** after.

**Hedge kept:** this proves the tree is reachable at `/config/output/...` *from
this container*. If the test container mounts the same volume at a different
internal path, the variable's literal value could differ while every observation
holds. Operationally irrelevant; one inference-step beyond what was measured.

Both outcome branches write there — success deposits the mkv and a `.log`, failure
deposits only a `.log.error` — which is why a single job settles it regardless of
whether the merge works.

### Goal 3 — the baseline: what the pipeline does today

Error id **237** (folder id 103, episode 1). Prediction written and frozen
**before** queueing; result scored against it. Both beside the repository.

The job finished in **~30 s** — far too short for a merge, and that is the first
result. It died in stage 4, on the exact code path zone A is being built into:

```
mergeVideo.py:798  remove_not_compatible_video
  Exception: Only …1 file left. This is useless to merge files
```

Stage 1 refused the pair, it was dropped, and with one file left the function
raised. Recorded stage-1 evidence: **ten windows, ten delays spread over 421 s and
flipping sign, mean fidelity 0.567.** Every delay is an exact multiple of this
pair's correlation point size, so **the instrument is not at fault**; the readings
sit on the quantum and are consistent with the sign convention. **The refusal is
correct: no constant offset exists.**

**Correction:** I first wrote "125 ms — the chromaprint frame", treating it as a
constant. It is not. `audioCorrelation.py:132` computes it per call and dev-1
measured it ranging **124–142 ms**, hitting exactly 125 only when the shortest audio
runs **18.7–30.4 min** — their measurement, their band, carried unhardened. This
pair's shorter file is 23.97 min, inside it, so 125 is right *here* and the
conclusion stands on a narrower warrant. See `AGENT_NOTES.MD`.

All five predicted points held, including the artefact being a `.log.error` rather
than an mkv. **My stated reason for the refusal was wrong**, which matters more
than the prediction being right — see below.

### Goal 3a — a corpus finding the baseline surfaced

I predicted a refusal because the file was already in `incompatible_files` and
nothing had changed. That reasoning treated it as an ordinary desync awaiting
repair. It is not one.

**The file in error id 237 is a different television series from its folder's
master.** Verified by extracting frames from both files at the same two timestamps
and looking at them — unrelated content in both pairs. That is measured, not
inferred from the release name. It also explains the stage-1 signature: two
different programmes have no true correlation peak, hence fidelity 0.567 and
delays that scatter and flip sign.

It is not one file. In folder id 103 — the largest, **49 of the 315 entries** —
**26 name themselves as that other series, a contiguous run of 26 episodes**, and
23 are correctly named. No folder for the other series is registered among the 295.
So **26 of 315 entries — 8.3 % of the corpus — are, on this evidence, misfiled
content rather than a synchronisation failure.**

Hedges, carried deliberately:

- **one file verified visually (id 237)**; the other 25 are attributed from the
  release name plus the contiguous-run pattern. I did not open them;
- I have **not** checked the other 37 folders. A corpus-wide count is task 003's;
- I make **no claim** about how campaign 1 classified these. I have not read its
  per-file classification and will not assert one.

Details, filenames and the frames are in the private notes beside the repository.
Cited here by id, per `AGENT.MD` § Privacy.

**Why it matters beyond a count.** `CAMPAIGN.MD`'s five families are *different
cuts, trimmed edges, speed mismatch, weak correlation, no shared language*.
"Different content entirely" is not among them, and it is not a degree of any of
them — the correct handling is the opposite of repair. The repair fires **at the
moment of refusal**, and these files refuse. Nothing in `SPEC_ZONE_A.MD` as drafted
distinguishes *two cuts of one episode that need reassembly* from *two different
episodes that must never be spliced*; both arrive as "stage 1 found no constant
offset". **I am not claiming a repair would mishandle them** — no repair exists yet
and its authors may plan a guard. I am saying the family that most needs one is not
in the table, and it was invisible until a real file was run.

Error id 237 is offered as a **negative fixture**: a real corpus file, reproducible
in ~30 s, where the correct answer is *refuse*. A repair should be run against it
to show it can still say no.

### Goal 4 — the run service

Published beside the repository. What a requester must state **before** a run:
what is under test, which error ids (by opaque id), what they expect, the
acceptance condition, **what would refute them**, and which artefact they want
back. I freeze that in a timestamped prediction file before queueing and score the
result against it afterwards.

Self-imposed: one job at a time; **no deploy while the queue is busy** (recreating
the container under a live merge kills it mid-write); the SHA is verified before
*and* after every job, and a run whose SHA moved is discarded rather than reported.

## What was dropped

- **Waiting a fixed ten minutes after a push.** Replaced by polling the registry
  for the target commit. The timer cannot distinguish "image ready" from "build
  still running"; the registry can.
- **The documented one-line hook-output filter.** Run verbatim it dies —
  `json.decoder.JSONDecodeError: Extra data: line 1 column 31572`. The response is
  one JSON object followed by trailing data. `deploy.py` uses `raw_decode()` and
  keeps the full response on disk. Reported, not edited: the document is the
  owner's.
- **Reading `GET /fusion` as proof the worker is alive.** That endpoint has no
  `fusion_enabled` guard, so an idle queue is returned whether or not the worker
  started. The probe that actually distinguishes them is `POST /fusion`.
- **Settling `VMSAM_TEST_OUTPUT_DIR` by inspecting the container.** Not reachable
  from here, and no endpoint exposes it. A run settled it instead — and the failure
  branch writes an artefact too, so it did not need the merge to succeed.

## What stayed blocked

1. **`fusion_enabled` is computed and then discarded.** `/internal/health` returns
   it; the public `/health` fetches that payload and keeps only `is_running`
   (`api.py:359-366`). From outside, a healthy idle worker and one that never
   started are indistinguishable — exactly the missing-third-outcome shape the
   campaign warns about. Not blocking today (the worker is running, proven by a job
   completing) and there is a cheap probe. `src/gestionar_show/` is frozen, so
   surfacing the field is an owner decision. **Finding, not an edit.**

2. **The `IDK` / `unknown` distinction is documented in two places as one state.**
   Both the private workflow note and the brief say `unknown` means the build arg
   was not injected; read against the code, that produces `IDK`. Both are
   unverifiable, so no result is misread — but the causes and the fixes differ.
   Owner's documents; reported.

3. **A full merge has never been timed.** No run has reached the mux, so the run
   service quotes ~30 s for an early refusal and *unmeasured* for a real merge. The
   first change that gets past stage 1 supplies that number; until then I will not
   quote one.
