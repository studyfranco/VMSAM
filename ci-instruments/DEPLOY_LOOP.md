# The deploy loop — confirmed procedure

**PRIVATE.** Names the API host and the credential file. `VMSAM_HELP_AI/` is
outside the repository; this file is never copied into it.

Owner: `vmsam-ci`. Confirmed end to end on 2026-09-03 against commit `db92f0e`.
Authority on the instance remains `private/DEPLOYMENT_WORKFLOW.md`; this file
records **what was measured**, and says so line by line.

---

## Provenance of every claim below

| mark | meaning |
| :--- | :--- |
| **[M]** | measured by me, 2026-09-03, with the command shown |
| **[D]** | taken from `private/DEPLOYMENT_WORKFLOW.md`, not independently re-measured |
| **[C]** | read from the code or the CI workflow in the repository |

---

## The one command

```sh
/home/vmsam/src/VMSAM_HELP_AI/ci/deploy.py --target <sha>
/home/vmsam/src/VMSAM_HELP_AI/ci/deploy.py --check      # what is running now
```

**Read its exit status, never its output.** 0 deployed and verified, 1 not
deployed, **2 the instrument could not run — which is not a pass.**

## What changed against the documented loop

### Step 2 is now a measured precondition, not a wait **[M]**

`DEPLOYMENT_WORKFLOW.md` step 2 says CI takes ~10 min and warns that a hook fired
early re-pulls the previous image and redeploys the old commit — a success-looking
deploy that silently invalidates the test. The brief names this as the single
failure mode that will waste the most of the campaign's time.

**A duration cannot be checked. The registry can.** The published image carries the
commit it was built from, in two independent places, and both are readable
anonymously from this container:

```sh
# anonymous pull token, then manifest -> config blob
curl -s "https://ghcr.io/token?scope=repository:studyfranco/vmsam:pull&service=ghcr.io"
# config blob carries:
#   Labels["org.opencontainers.image.revision"] = <sha>
#   Env: VMSAM_GIT_COMMIT=<sha>          <- exactly what /health will report
```

Measured 2026-09-03 for `dev-AI`:

```
created  : 2026-09-03T02:43:15.182343687Z
revision : db92f0e726b744b43af4760e2cfabc60fee38faf
version  : dev-AI
```

So `deploy.py` **waits until the published image is the target**, and refuses to
fire the hook otherwise. The dangerous case is now a refusal with a message
instead of a plausible-looking success.

### The ~10 min figure is a worst case, not a constant **[M]**

`db92f0e` was pushed and the image was published **about 90 seconds later**
(`org.opencontainers.image.created` 02:41:48Z, image `created` 02:43:15Z). The
workflow uses `cache-from: type=gha` **[C]** and that push touched documents only,
so every layer was a cache hit.

**Do not read this as "CI takes 90 s".** A push that changes `src/`, the
`Dockerfile` or the Rust crate rebuilds real layers and the ~10 min figure **[D]**
is the one to expect. The point is not that it is fast — it is that **you no
longer have to guess either way.**

### The documented output filter does not work **[M]**

`DEPLOYMENT_WORKFLOW.md` prints:

```sh
bash ./hook_executor_parameters 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); ...'
```

Run verbatim, 2026-09-03, this **dies**:

```
json.decoder.JSONDecodeError: Extra data: line 1 column 31572 (char 31571)
```

The response is one JSON object followed by trailing data, so `json.load` refuses
it. `deploy.py` uses `json.JSONDecoder().raw_decode()`, which reads the leading
object and ignores the rest, and keeps the full response at
`runs/hook-last.json` because it embeds the host firewall log.

**This is a finding about the document, not a fix to it.** The document is the
owner's; it is reported, not edited.

## Step 4 — four outcomes, not two **[C]**, all four fired **[M]**

`tools.py:226` is `os.environ.get("VMSAM_GIT_COMMIT", "").strip() or "unknown"` and
`Dockerfile:137` is `ARG VMSAM_GIT_COMMIT="IDK"`. That yields **two distinct
unverifiable states with different causes**, which the brief and
`DEPLOYMENT_WORKFLOW.md` both describe as one:

| `/health` `git_commit` | verdict | cause | exit |
| :--- | :--- | :--- | ---: |
| the target SHA | **deployed** | — | 0 |
| `IDK` | **unverifiable** | image built with **no** `--build-arg VMSAM_GIT_COMMIT`; it carries the `Dockerfile` ARG default. **Fix in CI** | 1 |
| `unknown` | **unverifiable** | the variable arrived **empty** at runtime — the compose file overrode it (`Dockerfile:163` already sets it empty). **Fix in compose** | 1 |
| another SHA | **not deployed** | the image was not published when the hook fired | 1 |
| no answer at all | **container did not come back** | re-fire, read `status`/`exit_code` | 1 |

Both documents say `unknown` means "the build arg was not injected". Read against
the code, build-arg-not-injected produces **`IDK`**; `unknown` means it arrived
empty. Both are unverifiable, so no *result* is misread — but they have different
causes and different fixes, so the distinction is worth carrying. **Finding, not an
edit.**

Neither state has been observed on this instance. They are coded from the source
and fired with literals, not from a sighting.

## The guards, fired deliberately, both answers **[M]**

*An armed guard that has never fired is indistinguishable from a working one*
(`AGENT.MD`) — that includes these.

| guard | fired with | result |
| :--- | :--- | :--- |
| image precondition **refuses** | `--target 6f71af4…` (not the published tag), budget 0 | `IMAGE NOT READY`, **hook not fired**, exit 1 |
| image precondition **accepts** | `--target db92f0e…` (is the published tag) | `IMAGE READY`, exit 0 |
| target validation | `--target deadbeef` | `not a full 40-character SHA`, exit **2** |
| health verdict `ok` | literal `{"git_commit": <target>}` | `DEPLOYED`, exit 0 |
| health verdict `unknown` | literal | `UNVERIFIABLE` + compose cause, exit 1 |
| health verdict `IDK` | literal | `UNVERIFIABLE` + CI cause, exit 1 |
| health verdict mismatch | literal `{"git_commit": 6f71af4…}` | `NOT DEPLOYED`, exit 1 |
| health verdict silent | literal `None` | `NO ANSWER`, exit 1 |

## The confirmed run — 2026-09-03

| field | value |
| :--- | :--- |
| deployed SHA before | `6f71af4cec0f05d65745bbd23857994e18362643` **[M]** |
| target | `db92f0e726b744b43af4760e2cfabc60fee38faf` **[M]** |
| what the diff touched | documents only — no `src/`, `Dockerfile`, `run.sh`, `init.sh`, `Cargo.toml`, `.github/`, `.forgejo/` **[M]** |
| image published | 02:43:15Z, revision = target **[M]** |
| hook | fired 02:46Z; **the documented parser crashed**, the hook itself succeeded **[M]** |
| deployed SHA after | `db92f0e726b744b43af4760e2cfabc60fee38faf` **[M]** |
| `/health` after | `mode test`, `dev false`, `is_running false` **[M]** |
| `GET /errors` before / after | 315 entries / 38 folders, **identical id sets** **[M]** |
| fusion queue before / after | idle, empty **[M]** |

**The corpus survives a container recreate** **[M]** — same 315 ids on both sides.
The database is not in the image. Worth knowing before anyone reads a corpus change
as a side effect of a deploy.

## Who pushes

**Not me, and not any developer.** `vmsam-ci` does not push (BRIEF_COMMON: the Lead
fetches and promotes). The clone at `/home/vmsam/src/VMSAM_WIP/ci` has `origin` =
`/home/vmsam/src/VMSAM` **[M]**, the Lead's clone, whose own `origin` is forgejo
**[M]**. So the loop is:

```
agent clone --(Lead fetches)--> /home/vmsam/src/VMSAM --(Lead pushes)--> forgejo
   --> CI --> ghcr --> [deploy.py] --> showgestionar-test
```

`deploy.py` with no `--target` reads `git -C /home/vmsam/src/VMSAM rev-parse HEAD`,
i.e. what the Lead last promoted — **not** what any agent has locally.

## Untested, and deliberately so

- The `IDK` and `unknown` runtime states. Coded from the source, fired with
  literals, never observed. If one ever appears, it is a finding.
- A **cold** CI build. Every observation here is of a fully cached documents-only
  build. The first real `src/` change is the measurement that settles the true
  build duration — and `deploy.py` does not care either way, which is the point.
- Re-firing the hook while a fusion job is running. Not attempted: it would
  recreate the container under a live merge. **Do not deploy while the queue is
  busy** — see `RUN_SERVICE.md`.
