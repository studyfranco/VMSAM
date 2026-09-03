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

### Goal 3b — the corpus-wide misfiling sweep (the Lead's priority)

**Answer: real, and NOT corpus-wide. 96 % of it is one folder.**

| | entries | |
| :--- | ---: | :--- |
| folder id 103 | **26** | three independent methods agree on the same ids |
| the other **37 folders** | **1** | confirmed by looking at frames |
| **total** | **27** | **8.6 % of 315** |

The 8 % survives and its operational meaning inverts: it is **one contaminated
folder plus one stray entry**, not a distributed family. Folder 103 holds 49 of
the 315 entries and more than half are junk; its other **23 entries are genuine
repair targets** (high fidelity, clean plateaus). Clean that folder and the family
essentially disappears.

**Why the folder-103 number is trustworthy — three methods, one answer.** Release
names plus video frames (mine); audio fidelity from the pipeline's own refusal
logs (the forensic agent, independently, never reading a filename or a frame); and
delay-pattern shape (mine, after the fact). The first two produced the **identical
26-id set** without either seeing the other's inputs.

**Two methods refuted before use, each against a case whose truth I already knew:**

- **filenames** — folders legitimately mix title conventions for one work (a
  romaji-named folder holding an English- or Chinese-titled release). Mostly false
  positives.
- **durations** — in folder 103 the misfiled and correctly-named entries show the
  *same* delta against their master. Different cuts produce duration mismatch too.
- **sparse frame pHash** — I built it, it separated my controls cleanly, and it was
  **wrong**: 20 frames over 22 min is 66 s apart, so any offset makes the samples
  miss. Its positive control had a **zero** offset — the one case that cannot
  expose the flaw. A control whose pass was guaranteed by the setup.

**The error this nearly produced.** `floor-scatter` (every window at chance,
delays scattered) looks exactly like "no shared content", and there are 45 such
rows — reporting them as misfiled gives ~14 %. **18 of the 45 are PAL-ratio speed
pairs**: same content, and precisely what `TASKS/008` exists to repair. dev-1
predicted this before I measured it, citing `docs/AUDIO_SPEED_POLICY.MD`'s
**0.558–0.562** for uncorrected PAL — the same floor a different-series pair sits
on. Verified at the source. **Fidelity level cannot separate "different content"
from "speed mismatch"**; the separators are the duration ratio and the delay
pattern. Reporting the floor as misfiling would have told dev-2 to stop repairing
18 files that are its target.

**My own screen produced three false positives out of four.** Of the four
non-speed floor-scatter rows outside folder 103, three are the same content on
inspection — one of them scoring low on a language-matched pair. Only one is
genuinely different, and its *own filename claims the same title as its master*,
so no naming heuristic could have found it. **The screen narrows candidates; the
eye decides.** Every "different content" count above was confirmed by looking.

### Goal 3c — recorded delays already exist for the whole corpus

A `.log.error` sits beside **every** error file, **315/315**, each carrying the
pipeline's own per-window `(fidelity, offset_points, delay_ms)` matrix. The
forensic agent found this; I verified it independently and extracted all 315.

**I had told a developer the opposite** — that delays exist only where a fusion has
run — and corrected it directly. **No container run is needed to obtain recorded
per-window delays for any error entry.** That materially changes what the queue is
for: it is needed to test *new code*, not to observe *existing behaviour*.

Three caveats travel with every figure from these logs: they were written by an
**older revision** (their traceback reads `mergeVideo.py:818/:946/:2050`, the
deployed commit reads `:798/:919/:2057`), through the **integration loop** rather
than the fusion path, and the candidate path inside them is sometimes historical.
The delays measure the media, which has not changed; they do not measure the code
under test.

Corpus shape, **after the correction below**: `within-step` 247, `floor-scatter`
50, `no-matrix` 12, `between-pair-offset` 4, and **2 `cmd-error`: the instrument
did not run**, kept on their own line and never folded into a family. PAL-ratio
speed candidates: **30 (9.5 %)**.

### Goal 3d — correction: my shape classifier had a defect, and the headline survived it

dev-1 found two faults. Both recorded rather than edited away; the second is a trap
for anyone who reads these logs.

**Fault 1, mine.** My first parser pooled every `(fidelity, offset, delay)` triple
**across track pairs**. On four files each track pair was internally *constant* and
the two values were the **master's own two audio tracks 125 ms apart** — not a
change point at all. Flattened, they read as a clean step, and **I recommended
those exact four as the best change-point exemplars.** A repair driven off that
would have repaired something that is not there. The rewritten parser separates
within-pair from between-pair variation and identifies exactly the four dev-1
named, reached independently.

**That correction made the misfiling answer stronger.** Folder 103 now splits
**perfectly** — 26 at the noise floor, 23 showing a step — where the flawed parse
gave 23 plus 3 unclassified. **27 of 315 stands.**

**Fault 2, dev-1's measurement.** The recorded delays are **residuals, not absolute
offsets**: `first_delay_test` re-extracts the candidate shifted by stage 1's own
answer (`mergeVideo.py:333`), and that shift is almost never a whole multiple of
the chromaprint hop (123.840 ms). A fractional-hop shift misaligns the fingerprint
grids. Their controlled measurement on one pair, varying only the shift, shows
whole-hop shifts holding fidelity at 0.967–0.975 while the pipeline's own
fractional shift drops it to 0.918–0.936 — the exact band that file's log records.
**So a one-point step may be an artefact of the instrument, and the recorded
fidelities are depressed by it rather than by the media.**

That is **4 files of 247** — two artefacts, one real, one unreadable. **Four files
is not a rate and is not carried as one.** The consequence carried is narrower:
the recorded data *cannot distinguish* a one-point step from a shift artefact.

So steps are now **ranked, not counted**: 20 are one-point (suspect), 17 are two
points, 23 are 3–5, and 187 are larger. Three independent corroborations that the
suspect bucket is the right one — dev-1's two measured artefacts and their
unreadable case all fall in it, and so does the file I had confirmed
**frame-identical by eye**, which is precisely the predicted artefact.

**What it did not touch.** The misfiling answer rests on the fidelity floor, on
three agreeing methods, and on frames I looked at — not on the step classifier. A
defect there could not move it, which is the one reassuring thing in the episode. These sit near `CAMPAIGN.MD`'s campaign-1 shape,
but **I do not claim to have reproduced campaign 1's classification** — the buckets
are mine, and 16 % sits in `mixed`, which its table has no equivalent for.

**What is still not established.** 25 of the 26 folder-103 files are attributed
from the release name, the contiguous run and the fidelity floor — only one was
opened. `mixed` (51) and `no-matrix` (14) were not individually examined, so **27
is a floor, not a ceiling**. The 30 speed candidates are identified by duration
ratio alone and none has been shown to correct.

## What was dropped
- **My own chromaprint screen as a corpus-wide detector.** It was validated on
  three eye-verified controls and still produced three false positives out of four
  checked. Kept as a candidate-narrower, dropped as a verdict. The corpus-wide run
  was killed part-way once that was known — it was consuming CPU shared with two
  other agents for a number I had shown I could not trust.
- **The container-duration table.** `format=duration` is set by the longest stream,
  usually video: on one master the container reads 1501.25 s while every audio
  stream reads 1436.977 s. dev-1 caught it. Re-probed on the audio axis, recording
  which of three sources answered, since Matroska usually leaves the per-stream
  `duration` field empty and uses a tag.

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
