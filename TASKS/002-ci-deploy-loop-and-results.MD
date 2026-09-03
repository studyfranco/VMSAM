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
7. **A broken instrument produced false CLEANLINESS, not noise — and it propagated.**
   The first whole-file scan searched offsets within ±3000 ms of zero; 64 of 100 sampled
   files carry a recorded delay outside that span. Outside its window the correlation
   saturates at the edge and returns the *same wrong value every probe*, so the classifier
   saw a constant sequence and correctly called it flat. That run reported 28 % flat.
   The corrected run reports **zero eligible-flat in the first 35** — every flat file is a
   master-side pair, and 247 of 315 corpus files are already `within-step` in the record,
   so a large flat population was never plausible.
   Noise would have been discarded in a minute. Saturation yields stable, repeatable,
   confident agreement that no summary statistic separates from a clean file — and the
   clean set is the worst place for it, because clean is what everyone else samples *from*.
   The defect damaged another agent's plan, silently, through a list.
8. **The miss rate must be scored as a JOIN, not a marginal count.** The sample is drawn
   from the *refused* corpus, so a measured step is usually the very step the record shows.
   Counting labels would have reported ~48 % as a "miss rate" — a large, quotable number
   that counts the record's successes as its failures. A miss requires both halves: the
   record blind **and** structure found. Estimand, decision boundary and the two defects
   found while scoring are frozen in `VMSAM_HELP_AI/ci/runs/SCANA3-PREDICTION.md`.

9. **The miss rate is bounded: 2 of 88 = 2.3 %, Wilson 95 % [1.3 %, 7.2 %].** 98-file
   seeded sample of the refused corpus, uniform 30-probe instrument, manifest written
   before the first probe; estimand and a 10 % refutation condition frozen in advance.
   **Not refuted.** Two of three frozen predictions were wrong and are recorded as wrong.
   **The sentence that must travel with it: 2.3 % is the rate of FILES the record was
   wholly blind to, not the rate of STRUCTURE it missed.** One sampled file carries a
   confirmed 24.7 ms sub-quantum step and scores as *agreement*, because its other,
   larger step is visible. The second number is larger and this sample does not bound it.
10. **The repair's verifier checks the track it built, not the track that shipped.**
    Two container runs on the current build, predictions frozen first:
    error id 144 **declined** with its own measurement — *rebuilt track off by 1103.4 ms,
    tolerance 100 ms* — on a file whose correct answer is known from three independent
    directions. That is the decline path working as `SPEC_ZONE_A` requires.
    Error id 352 **logged `1 audio and 4 subtitle track(s) rebuilt, 0 declined`**, and the
    artefact carries the four subtitles tagged while **no audio track carries the marker**;
    the output's audio is the master's at correlation 1.0000. **A track reported as rebuilt
    is absent from the artefact, and the verifier cannot see that** — it runs before the
    merge. Owned jointly with the repair's author; the mechanism is not yet settled.
11. **Freezing a prediction does not protect the interpretation.** The id 144 refutation
    table said *"no mkv → over-conservative"*. There was no mkv, and the reading was wrong:
    the decline was correct because the built track really was off. **The rows were
    verdicts, not observations, and pre-registering a reading makes it harder to revise
    because it carries the authority of having been frozen.** Freeze the observation and
    the decision rule; never the conclusion — and check the rule can come out both ways on
    the space actually searched.

Narrative, method and the day's corrections: `AGENT_NOTES.MD`, and
`VMSAM_HELP_AI/ci/` (full archive at `002-full-narrative-archive.MD`).
Method write-up for the algorithms index: `VMSAM_HELP_AI/ci/sections/SCANNER.md`.
