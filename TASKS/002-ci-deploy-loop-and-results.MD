# 002 — The test container: deploy loop, fusion queue, results

**Agent:** `vmsam-ci`. **State:** open.

## Goal

Own the path from a pushed commit to a comparable artefact, and keep it working
for the other three. Nobody else fires the hook or queues a fusion.

1. **Confirm the deploy loop end to end** and write the confirmed procedure into
   `VMSAM_HELP_AI/ci/`. The rules are in `VMSAM_HELP_AI/private/DEPLOYMENT_WORKFLOW.md`.
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

## What changed
*(fill as you go)*

## What was measured
*(fill as you go)*

## What was dropped
*(fill as you go)*

## What stayed blocked
*(fill as you go)*
