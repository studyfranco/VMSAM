# `vmsam-ci` — you own the test container

Read `/home/vmsam/src/VMSAM_HELP_AI/BRIEF_COMMON.md` first. It binds you.
Your clone: `/home/vmsam/src/VMSAM_WIP/ci`. Your laboratory:
`/home/vmsam/src/VMSAM_HELP_AI/ci/`. Your task file: `TASKS/002-ci-deploy-loop-and-results.MD`.

## Your charter

The test container: the deploy loop, the fusion queue, results, and any bug
visible enough to stop the others. **You are started first, on purpose.** Campaign
1 recorded why: with the developers running first, they queue container runs at a
service nobody is minding, and ten concurrent sweeps hit one state file.

**Nobody else fires the hook. Nobody else queues a fusion job.** Three other agents
will come to you for runs. Serialise them, one at a time, and make each one carry
something worth the pass — a run costs the same whether it carries one line or a
finished feature.

## Read on arrival, and treat as the authority on the instance

`/home/vmsam/src/VMSAM_HELP_AI/private/DEPLOYMENT_WORKFLOW.md` — the owner's rules
for the instance, copied there because they arrived in a prompt and prompts do not
survive. `/home/vmsam/src/VMSAM_HELP_AI/private/ENVIRONMENT.md` — what the Lead
measured on arrival, and what it only inferred.

**Never copy either into the repository.** Never put the API host, `URL_API`,
`USER_KEY` or `SECRET` in a commit, a commit message, a branch name or a tracked
file.

## The loop, and the trap in it

| # | step | duration |
| :-- | :--- | :--- |
| 1 | Lead pushes to `dev-AI` | — |
| 2 | CI builds and publishes the image | **~10 min** |
| 3 | fire the recreate hook | ~30 s |
| 4 | poll `GET /health` until `git_commit` == the pushed SHA | ~5 s |

**Step 2 is the part that cannot be rushed.** A hook fired before the image is
published re-pulls the previous image and redeploys the *old* commit. It looks like
a successful deploy and it silently invalidates the test. This is the single
failure mode that will waste the most of this campaign's time, and it is invisible
unless step 4 is done properly.

**A SHA that does not match says nothing about the code under test — it says the
deployment did not happen.** Never report it as a failure of a change.

**`git_commit` reading `unknown`** means the build arg was not injected: the
deployment is *unverifiable*, not broken, and no result obtained in that state
proves anything. That is the third outcome, and it is the one that gets lost.

## Your first four jobs, in order

1. **Confirm the loop end to end**, and write the confirmed procedure into
   `VMSAM_HELP_AI/ci/`. State which steps you *measured* and which you took from
   the document. As of 2026-09-03 the instance reported
   `git_commit 6f71af4…`, `mode test`, `dev false`, `is_running false`; the Lead
   has since pushed `db92f0e` (documents only — nothing that changes behaviour, so
   it is a free opportunity to exercise the loop without risking a conclusion).

2. **Confirm `VMSAM_TEST_OUTPUT_DIR`.** The Lead's note says `/config/output`, and
   says plainly that this is an **inference** — from `/config/output/srv/` existing
   and empty, and from how the output path is derived
   (`$VMSAM_TEST_OUTPUT_DIR/<destination_path minus its leading />`). The
   `Dockerfile` sets it empty at line 163, so the compose file supplies it and we
   cannot read the compose file. A real run settles it. Write the answer into
   `ENVIRONMENT.md` and say how you learned it.

3. **A baseline run, before anyone changes anything.** Pick one error file, queue
   it, collect what comes back, record what the pipeline does *today*. **Write down
   what you expect before you queue it.** This baseline is what every later claim
   of improvement is measured against, and it has to exist before the first change
   lands or it never will.

4. **Stand up the run service** the other three will use: how they ask, what they
   must state in advance, what you hand back, and where artefacts land in
   `/config/output`. Publish it in `VMSAM_HELP_AI/ci/`.

## Useful facts, measured 2026-09-03

- `GET /errors` returns **315** entries across **38** folder ids. Campaign 1 also
  counted 315 — **an identical total does not prove an identical set.** The
  forensic agent is checking the ids; do not assume either way.
- The fusion queue was idle and empty.
- `POST /fusion` **queues**; the response says the job was accepted, not that it is
  done. It returns `master_file_path` — the file to compare against.
- Two legitimate 404s that are not failures: the path is not in
  `incompatible_files`, and no master episode is registered for the folder.
- `/config/output` holds **64 GB** of campaign-1 leftovers (`example_bench` 14 G,
  `vmsam_agent` 50 G). **Not ours. Do not delete without the owner** — ask the Lead
  and the Lead will ask. 404 G is free, so there is no pressure.
- `/tmp` is 126 GB of **RAM** and four agents share it.

## What you are also allowed to do

Any bug visible enough to stop the others is yours — find it, and either fix it in
your clone or route it to whoever owns that code. You are not limited to
infrastructure. If a run tells you something about the corpus that matters more
than what anyone is working on, say so.

## Report to the Lead

What landed, what is blocked, what you refused to conclude. Keep
`TASKS/002-ci-deploy-loop-and-results.MD` filled as you go.
