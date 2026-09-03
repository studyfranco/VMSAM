# 002 — The test container: deploy loop, fusion queue, results

**Agent:** `vmsam-ci`. **State:** open.

## Goal

Own the path from a pushed commit to a comparable artefact, and keep it working for
the other three. Nobody else fires the hook or queues a fusion.

1. Confirm the deploy loop end to end; write the confirmed procedure into `VMSAM_HELP_AI/ci/`.
2. Confirm `VMSAM_TEST_OUTPUT_DIR`.
3. A baseline run before anyone changes anything, with the expectation stated first.
4. Serialise the queue. One job at a time; the others ask.
5. Any bug visible enough to stop the others is mine to fix or route.

## What changed

**No engine code.** The instruments live beside the repository (`WRITE_ZONES.MD` §4):
`deploy.py` (the loop as one command, one exit status), `DEPLOY_LOOP.md`,
`RUN_SERVICE.md`, and `runs/` (a frozen prediction and a scored result per run).
Repository-side: this file and `AGENT_NOTES.MD`.

## What was measured

| | |
| :--- | :--- |
| deploy loop | confirmed end to end, four times; eight guards fired both ways; exit 2 means the instrument could not run and is **not** a pass |
| `VMSAM_TEST_OUTPUT_DIR` | **`/config/output`** — measured twice over, no longer an inference |
| CI, pure-Python `src/` change | **≈1.9 min** commit→image (2009 lines). Only the final `COPY src/` layer invalidates |
| CI, Dockerfile/Cargo/dependency change | **unmeasured**. The documented ~10 min belongs here |
| stage-2 job cost | **unmeasured** — the one job that reached it ran under 2.8× CPU oversubscription, so its 137 min is not a duration |
| corpus across a recreate | **unchanged**, 315 entries / 38 folders, identical ids, every deploy |
| fusion worker on the first `config.ini` build | **started** — `POST /fusion` 200, no 503 |
| stage 1 across an engine change | **deterministic** — same file, two builds, ten delays bit-identical |
| zone A with its flag off | **inert** — same file refuses identically with the call site present |
| misfiled content in the corpus | **27 of 315 (8.6 %)**, and **26 of the 27 are one folder** |

## What was dropped

- **Waiting a fixed ten minutes after a push.** Replaced by polling the registry for the
  target commit; a duration cannot be checked, the published image can.
- **The documented one-line hook filter.** Dies on the real response
  (`JSONDecodeError: Extra data`); `raw_decode` used instead.
- **`GET /fusion` as proof the worker is alive.** No `fusion_enabled` guard — the decisive
  probe is `POST`.
- **Three screens for the misfiling question**: filenames (folders mix title conventions
  for one work), durations (misfiled and correct files show the same delta), and a
  frame-pHash instrument I built whose positive control had a **zero offset** — the one
  case that could not expose its flaw.
- **My own chromaprint screen as a verdict.** Three false positives in four checked; kept
  as a candidate-narrower only.
- **The container-duration table.** `format=duration` is set by the longest stream; re-probed
  on the audio axis.
- **"Drop the outlier window"** as a remedy, and **"the rounding causes the refusal"** —
  both mine, both retracted after measurement.

## What stayed blocked

1. **`fusion_enabled` is computed and discarded** before the public `/health`, so a healthy
   idle worker and one that never started are indistinguishable from outside.
   `src/gestionar_show/` frozen — owner's call.
2. **`dev:false` disarms `SPEC_ZONE_A.MD` §3's detector.** Those fallback log lines *cannot*
   appear, so their absence is guaranteed by configuration rather than health. Measured:
   artefacts end `Logs:` empty while the non-gated `Merged errors:` is populated.
3. **Stage 2's variance test cannot distinguish a real step from a bad window** and refuses
   both identically. The refusal is correct; nothing downstream is told which it was.
   `mergeVideo.py:582–594`, outside every tagged zone.
4. **The stage-1 record is not a census.** A step is invisible unless it is both late enough
   (`T > begin + L/2`) and larger than roughly half a quantum. Two independent unbounded
   error terms, neither estimable from the record. Every family count inherits them.
5. **Objective 2 waits on the owner's re-download** and has not been started.
6. **No inbound message has ever been delivered to this session.** Outbound works; every
   instruction acted on today arrived by relay.

Narrative, method and the day's corrections: `AGENT_NOTES.MD`, and
`VMSAM_HELP_AI/ci/` (full archive at `002-full-narrative-archive.MD`).
