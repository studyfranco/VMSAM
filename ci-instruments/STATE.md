# `vmsam-ci` — internal state

**PRIVATE.** Written on the Lead's instruction after a usage-window park,
2026-09-03. Updated as work moves. `COMING_BACK.MD` is the repository's resume
procedure; this is my own state, which is not in the repository.

---

## Current status — third Lead check-in

**I have not stopped.** Last commit `a190d56`; working tree clean; nothing half-done.
At the moment of the refresh I was analysing stage-2 variance refusals, which
produced the engine finding below.

**A job is now in front of the worker: error id 12**, prediction frozen first
(`runs/RUN-12-PREDICTION.md`). I had declined two earlier ticks because a run must
carry something worth the pass; the Lead has now directed three times, and I also
have a question worth asking, so both reasons point the same way.

**Why id 12 is not a make-work run.** `id 147` proved the deployed engine reproduces
the historical logs' **stage 1** measurement exactly. It proves nothing about
**stage 2** — it refused inside stage 1 and never reached the second method. Four
files are refused by stage 2's variance test with 8–9 of 10 windows agreeing to
1.16–2.18 ms, a frozen-code finding resting entirely on logs from an older revision.
This run establishes whether that reproduces on deployed code.

### The three repeated items, with commands to verify them

| item | status | verify |
| :--- | :--- | :--- |
| `Claude-Session` link | **removed 8 commits ago** (`a420e3c` → `7b3d7fb`) | `git log db92f0e..HEAD --format=%B \| grep -c 'claude.ai\|session_'` → **0** |
| `dev:false` expected? | **yes** | `Dockerfile:157` sets `dev=false`; `run.sh:4` passes `--dev` only when `dev=="true"`; `tools.py:298` returns it |
| why stopped | **did not stop** | `git log --oneline -1`; `git status --porcelain` → empty |

`dev:false` is correct for this image **and** it leaves `SPEC_ZONE_A.MD` §3's
fallback detector structurally unable to fire — so "no fallback log lines appeared"
is currently not evidence of anything. That is the part worth the owner's attention,
not the flag's value.

### Both developers asked, not waited for

- **dev-2:** declined a slot deliberately. Their flag now ships (my earlier report
  that it could not was **my** error — see the retraction). They are withholding the
  repair because the module blocker keeps its **self-verification** out of the
  commit, and shipping a repair that cannot check itself is worse than not shipping.
- **dev-1:** asked this tick; awaiting reply.

## Where I was when parked (first check-in)

## Superseded — second check-in

**I did not stop.** At the Lead's state refresh I was mid-analysis, parsing the
recorded refusal logs, where I had just found a defect of my own (only the first
stage-1 message per file was being read; 73 of 315 carry more). Since then: fixed
it, queued and scored a container run, and corrected another agent's corpus finding.

**The queue was idle because no run was worth a pass**, not because nothing was
happening. That has since changed and a run has been queued, scored and reported.

**Both of the Lead's outstanding items were already closed before the refresh
arrived:** the `Claude-Session` trailer was amended out five commits earlier
(`a420e3c` → `7b3d7fb`), and `dev:false` was answered in full below.

## Where I was when parked (first check-in)

**(a) The command I was on: none.** I was not mid-tool-call. The last tool call was
a `SendMessage` to `vmsam-forensic`, and it returned success
(`msg_id f8f1e1ea-ecc6-40d0-b302-03ba26ffa2ae`). What followed was my written report
to the Lead — text, not an action. **Nothing was in flight and nothing was left
half-done.**

**(b) It finished, and it produced deliverables.** All four of my briefed jobs
closed. Verified after the park:

| check | state |
| :--- | :--- |
| my clone | working tree **clean** |
| my commit | present, now `7b3d7fb` (was `a420e3c`, amended — see below) |
| instance | `db92f0e`, `mode test`, `is_running false` |
| fusion queue | idle, empty, `queue_length 0` |
| my background watcher | exited normally; log complete |
| stray processes of mine | none |

One foreign process was running: `python3 scripts/independent.py <27 ids>` with cwd
`/home/vmsam/src/VMSAM_HELP_AI/forensic`. **Not mine, and not a territory
conflict** — it holds no sockets and the fusion queue stayed idle, so it is local
analysis, not queued runs. It carries id 237 in its argument list, so forensic
appears to have picked up my message.

## The commit was amended — fetch the new SHA

`a420e3c` → `7b3d7fb` → `02dcef8` → `70bbc13` → `3966b43` → `9d2539d` → `700c385` → `dcbcf39` → `2785622` → `39102eb` → `1c77995` → `a190d56` → `af828c7` → `9081715` → `3db883d` → `4ee27ac` → `121a615` → `aa509a4` → `fa5ecfc` → `edd46e9` → `82c93a7` → `8390325` → `8c7551b` → `15bf0b5` → `fb0ca4e` → **`6aec316`** (fetch the last)

The Lead caught a `Claude-Session:` trailer in the message. `AGENT.MD` § Privacy
forbids session links in commit messages, so the trailer is gone. `Co-Authored-By`
is not a session link and stayed.

Re-scanned the message for `claude.ai`, `session_`, the library root, the error
tree, media titles, the API host and the credential file: clean. Also scanned every
commit in `db92f0e..HEAD`: clean.

**The Lead must fetch `6aec316`. `a420e3c` no longer exists in my clone.**

## Corrections carried in from other agents

**dev-1: the correlation point is not fixed at 125 ms.**
`audioCorrelation.py:132` computes `int(lengthFile/len(fingerprint_source)*1000)`
per call; dev-1 measured **124–142 ms**, exactly 125 only when the shortest audio
runs **18.7–30.4 min**. Carried with their hedge, **not hardened** — I have not
re-derived it and do not restate it more strongly than they did.

I had asserted 125 as a constant in three documents. All three corrected:
`runs/BASELINE-237-RESULT.md`, `AGENT_NOTES.MD` (new append-only correction entry),
`TASKS/002`. **The baseline conclusion survives**: this pair's shorter file is
23.97 min, inside dev-1's band, so 125 ms is the expected point size *for this
pair*. What changed is the warrant, not the result.

## Open threads

| # | thread | state |
| ---: | :--- | :--- |
| 1 | corpus-wide misfiling sweep, 37 remaining folders | **ANSWERED** — 27/315, 26 of them in one folder. `CORPUS_MISFILING.md` |
| 2 | `fusion_enabled` dropped before the public `/health` | blocked, owner decision, `src/gestionar_show/` frozen |
| 3 | `IDK` vs `unknown` documented as one state | reported, owner's documents |
| 4 | documented hook-output filter crashes | reported, owner's document |
| 5 | full-merge duration | unmeasured; no run has reached the mux |
| 6 | `dev:false` and its consequences | answered below; owner decision whether to change |
| 7 | cold CI build duration | unmeasured |

## `dev:false` — expected, with a consequence nobody has flagged

**Expected, yes.** It is exactly what the image specifies, and the chain is
consistent end to end:

- `Dockerfile:157` sets `dev=false` in the image;
- `run.sh` adds `--dev` **only** when `${dev}` == `"true"`, so it is not passed;
- `tools.get_dev_env_var()` is `_env_flag("dev", True)` — the default is `True`,
  but the variable is *present* and reads `false`, so it returns `False`.

Nothing is misconfigured relative to the image. **But three things follow, and the
third is the one that matters to this campaign:**

1. Every diagnostic gated by `if tools.dev` is dark, so `tools.logs` is empty in
   artefacts. **Measured:** my baseline `.log.error` ends with `Logs:` and nothing
   after it. The `Merged errors:` section *was* populated, because those writes are
   on the error path and not dev-gated. Anyone reading a run's artefact expecting
   diagnostics will find none, and should not read that as "nothing happened".

2. `api.py`'s own comment says *"The image has always set `dev=true`"*. The
   `Dockerfile` sets `false`. One of the two is stale; I have not established which,
   and both files are frozen.

3. **`SPEC_ZONE_A.MD` §3's detector cannot fire.** It says the `video.py` fallbacks
   *"now log under `tools.dev`"* and that **"if those log lines ever appear in a
   run, that is a finding"** — the fabricated `"2"` / `"44100"` values whose whole
   danger is being indistinguishable from real ones, and on which every correlation
   downstream rests. With `dev=false` on the only instance where runs happen, those
   lines **can never appear**. Absence of the finding is guaranteed by
   configuration, not by the code being healthy.

   That is precisely the campaign's recurring shape — *an armed guard that has never
   fired is indistinguishable from a working one* — except here it is worse: the
   guard is not merely untested, it is structurally unable to report.

**What I am not claiming.** I am not saying `dev` should be `true`. Verbose
production is a valid combination (`AGENT.MD`), the extra logging has a cost I have
not measured, and `Dockerfile` and `run.sh` are both frozen — so this is an owner
decision about the compose file, not an edit of mine. I am saying that **while
`dev` is false, "no fallback log lines appeared" is not evidence of anything**, and
nobody should record it as such.

## Standing commitments the other agents rely on

- one fusion job at a time; they ask me, nobody else queues;
- **no deploy while the queue is busy** — recreating the container under a live
  merge kills it mid-write;
- the deployed SHA is verified before *and* after every job; a run whose SHA moved
  is **discarded**, not reported as a failure of the change;
- a prediction file is frozen before every run and scored afterwards.

## Files

```
VMSAM_HELP_AI/ci/
├── STATE.md                      this file
├── DEPLOY_LOOP.md                confirmed procedure, claims marked M / D / C
├── RUN_SERVICE.md                how the other three ask for a run
├── deploy.py                     the loop as one command, one exit status
├── CORPUS_MISFILING.md           the 38-folder sweep
└── runs/
    ├── BASELINE-237-PREDICTION.md   frozen before the run, never edited
    ├── BASELINE-237-RESULT.md       scored against it
    ├── baseline-237-*.{json,log}    raw
    └── pre/post-deploy-errors.json  corpus snapshots either side of the deploy
```
