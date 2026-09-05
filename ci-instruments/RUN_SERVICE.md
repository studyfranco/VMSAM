# The run service — how to get something tested

**PRIVATE.** Names the API host. Never copied into the repository.

Operated by `vmsam-ci`. **Nobody else fires the deploy hook or queues a fusion
job** (`BRIEF_COMMON` § Territory). Come to me; the queue is serialised, one job at
a time, and I am started first precisely so that ten concurrent sweeps do not hit
one state file.

---

## Before you ask: is a run what you actually need?

`AGENT.MD` § Target-first is a four-step order and a run is **step 3**:

1. build the expected output **by hand** and check it yourself;
2. run the pipeline **locally** — this hunts crashes and never-taken paths;
3. **then** ask me;
4. compare the produced file against your target on the properties you named in 1.

A run costs the same whether it carries one line or a finished feature, so ask when
you hold something worth the pass. A local run proves your code executes; only a
real run proves the pipeline produces the result.

## What I need from you, in one message

A request missing any of these gets handed back — not to be difficult, but because
a run whose expected result was written afterwards cannot be read.

| # | field | why |
| ---: | :--- | :--- |
| 1 | **What is under test** — the commit, or "nothing, this is a baseline" | so the result can be attributed to something |
| 2 | **Which error ids**, by opaque id | never a filename; `AGENT.MD` § Privacy |
| 3 | **What you expect to happen** — stated *before* the run | a target built afterwards is fitted to whatever came out |
| 4 | **The acceptance condition** — what makes this a pass | |
| 5 | **What would refute you** — what result would say your change is wrong | this is the field people skip, and it is the one that makes the run worth its cost |
| 6 | **Which artefact you want back** — the mkv, the `.log`, the `.log.error`, or a named property of the output | so I collect the right thing |

I write 3–5 into a prediction file, timestamped, **before** I queue anything, and I
do not edit it afterwards. The result goes in a separate file and is scored against
it. That is not ceremony: it is the only check that dead code cannot pass.

## What I hand back

| | |
| :--- | :--- |
| the deployed SHA | verified on `/health` **before and after** the job |
| the queue transitions | with timestamps |
| every artefact | path under `/config/output/…`, and the `.log` or `.log.error` inline |
| the prediction, scored | point by point, including the ones you got wrong |

**If the SHA moved during your run, I discard the result and tell you so.** I do
not report it as a failure of your change (`BRIEF`: a SHA that does not match says
nothing about the code under test — it says the deployment did not happen).

## Where artefacts land

`VMSAM_TEST_OUTPUT_DIR` = **`/config/output`** (confirmed by measurement
2026-09-03; see `private/ENVIRONMENT.md` for how, and for the one hedge on it).

```
/config/output/<destination_path minus its leading />/
```

`/config/output` is btrfs, survives a restart, and is the **only** read-write
channel shared with `showgestionar-test`. Both outcome branches write there:

| outcome | what lands |
| :--- | :--- |
| merge succeeded | the `.mkv` **and** a `.log` beside it |
| merge failed or refused | only a `.log.error` |

So **every** completed job leaves a trace. A job that leaves nothing did not run.

### House rules for that directory

- Drop the target you built by hand there too — it is also the channel *to* the
  test container.
- `example_bench` (14 G) and `vmsam_agent` (50 G) are **campaign 1's leftovers and
  not ours**. Do not delete them without the owner; ask the Lead and the Lead asks.
  404 G is free, so there is no pressure.
- Put your artefacts under a path that says whose they are. Do not write at the top
  level.

## Timing, measured

| step | measured 2026-09-03 |
| :--- | :--- |
| push → image published | **~90 s** for a documents-only change (full layer cache). Expect the documented **~10 min** for a real `src/` change; a cold build is not yet measured |
| hook → container recreated | ~30 s |
| recreate → `/health` answers the new SHA | < 6 s |
| a fusion job that **refuses early** | ~30 s (measured on id 237) |
| a fusion job that runs a full merge | **not yet measured** — no run has reached the mux |

**I no longer wait a fixed ten minutes.** `deploy.py` polls the registry until the
published image carries the target commit, then fires the hook. The
fired-too-early failure mode — which redeploys the old image and looks like a
success — is now a refusal with a message rather than a silent bad result. See
`DEPLOY_LOOP.md`.

## Rules I enforce on myself, so you can rely on them

- **One job at a time.** The worker is sequential; I do not overlap requests.
- **I do not deploy while the queue is busy.** Recreating the container under a
  live merge would kill it mid-write. If you need a deploy, it waits for idle.
- **I state what I expect before every run**, including my own.
- **I report the third outcome.** "I could not measure" and "this is unverifiable"
  are different answers from "it failed", and I will give you whichever is true.
  A `git_commit` of `IDK` or `unknown` means the deployment is *unverifiable* and
  **no result obtained in that state proves anything** — I will throw it away
  rather than hand you a number.

## Known limitation you should know about

From outside, **you cannot tell a healthy idle worker from one that never
started.** `/internal/health` computes `fusion_enabled`, and the public `/health`
fetches that payload and keeps only `is_running` (`api.py:359-366`), so
`is_running: false` + an idle queue is ambiguous.

It does not currently bite: the worker is running, proven by a job completing. The
cheap probe if you ever doubt it is to `POST /fusion` and read the status — 503
`Fusion is disabled` is the disabled case. `src/gestionar_show/` is frozen, so
surfacing the field is an owner decision, recorded as a finding, not an edit.
