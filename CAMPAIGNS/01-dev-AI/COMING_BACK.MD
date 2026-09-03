# Coming back — read this ONLY when returning from a stop

**Do not read this file on a normal session start.** It is for one situation:
you were cut off mid-work -- a usage limit, a container update, a context
compaction, a killed session -- and you are picking up again. `AGENT.MD` is what
you read to know the project; **this is what you read to find out what happened
while you were gone.**

**Nothing below is optional, and the order matters.** Every step exists because
skipping it has already cost this project work.

**The one-line version**: your own memory of what was happening is the least
reliable source in the room. The tree, the registry, the process table and the
invariants have all moved. Ask them, do not recall them.

---

A session ends without warning: a usage limit, a container update, a context
compaction, a killed agent. **Nothing below is optional, and the order matters.**
Every step exists because skipping it has already cost this project work.

### 1. Read the state before touching anything

```bash
cat src/TASK.MD                      # mission, counts, the job registry
git -C . log --oneline -8            # what actually landed
git status --porcelain               # what did NOT land -- see step 2
```

### 2. Rescue uncommitted work FIRST

An agent killed mid-task leaves its work in the **working tree**, uncommitted
and unattributed. It is the most fragile thing in the repository: one careless
`git checkout` or a history rewrite destroys it. Before anything else, look at
what is modified and untracked and decide what it is. Two agents died to a usage
limit on 2026-08-30 with finished, correct, unpushed code in the tree.

Untracked files you did not create are almost always a dead agent's deliverable.
Read them before assuming they are junk.

### 3. Check what is still alive, and do not trust a bare `pgrep`

```bash
ps -eo pid,etime,cmd | grep -E "full_test|batch_verify|watchdog_loop|hourly_deploy" | grep -v grep
```

`pgrep -f <name>` **matches its own shell** and has twice reported a dead job as
alive. Confirm with `ps`, and confirm the job's log file exists *and is growing*
before believing it. Restart anything dead using the exact command in the
`src/TASK.MD` job registry — every job there is resumable or idempotent by
design.

**Never kill by path substring. Identify your processes by PARENTAGE.**
`/config/output/vmsam_agent/` is shared, so any grep on it matches every agent's
decoder. An agent cleaning up its own `ffmpeg` matched a substring including
`vmsam_agent` and was one step from killing another agent's 80-minute decode of
the preserved baseline — a run that had already been restarted once. The two
processes were indistinguishable to a path filter and obviously different by
`ppid`:

```bash
ps -eo pid,ppid,etime,cmd | grep ffmpeg | grep -v grep   # LOOK, then act
```

What saved it was printing the full cmdline before acting instead of piping the
match into `kill`. Do that every time: **look at what you matched before you
kill it.** In the same minute, `pgrep -f "cut_point_scan.py"` also returned two
pids that were only that agent's own shell pipeline containing the literal
string — the self-match trap above, firing simultaneously.

**And `kill $!` on a wrapper does not stop the job.** Launching under
`nohup nice -n 15 python3 ...` makes `$!` the pid of `nice`, not of python.
Killing it **reports success and leaves the job running, reparented to init** —
so an agent that trusts it will report the box free while still holding a
decoder. Verify the work actually stopped by looking for the real process, not
by trusting the kill's exit status.

### 4. Wake the other agents — they do not come back on their own

A usage limit closes **every session on the account at once**, the Lead's and
the agents' alike. It does not kill them: the tmux session and the `claude`
process survive, the agent is simply blocked mid-turn. So when the window
reopens the team is not dead, it is **parked** — and it stays parked
indefinitely, because nothing external pokes it. Waking it is the Lead's job and
belongs in this procedure, not in the Lead's memory.

```bash
tmux ls                                    # sessions that survived
```

Then compare that against the **roster in `src/TASK.MD`** — the hand-maintained
table of who *should* be alive. The generated block below it says who *is*. The
whole procedure is the diff between the two:

- **Listed and present** → it is parked. Send it a message with `SendMessage`,
  addressed by its exact session name. The message *is* the wake-up: delivery
  resumes the session at its next tool round.
- **Listed and absent** → it is dead, and a message will not reach it. Respawn
  from the roster's restart command, which carries `--resume` so it continues
  its own history. **Never `--continue`** — history is keyed by working
  directory and every agent starts in the repository, so `--continue` resumes
  the *Lead's* conversation instead.

**Wake `vmsam-agent-ci` first.** It owns the test queue and the deployment loop;
if the others come back first they will queue container runs at a service with
nobody minding it, and that queue has already run ten concurrent sweeps against
one state file.

**Say what changed, not "are you awake".** An agent resumes believing whatever
was true when it was frozen, and acts on it. Give it the delta: the current
`HEAD`, which jobs died and were restarted, what the mission counts are now, and
anything landed by another agent that touches its files. A wake-up that is only
a greeting produces an agent confidently continuing a superseded plan — cheaper
to prevent in one sentence than to unpick afterwards.

Finally, if the roster and the live block agree and everything is running, say
so and change nothing. Restarting a healthy loop is how this project ended up
with two owners of one sweep.

### 5. Compare deployed against HEAD

```bash
curl -s http://showgestionar-test:8080/health
git rev-parse HEAD
```

`git_commit` must equal `HEAD` before any result from the container means
anything. `dev` must be `true` or every diagnostic is silently dark. A container
running older code will happily produce results that look fine and prove
nothing. CI needs ~10 min after a push; firing the deploy hook earlier re-pulls
the previous image and *looks* successful.

### 6. Re-verify, do not re-derive

```bash
cd src/tools_batch/tests && python3 -m unittest discover   # must be green
python3 src/tools_batch/check_no_private_refs.py           # must be clean
python3 src/tools_batch/master_summary.py                  # current counts
```

The suites are pure-unit and run in under a second; there is no excuse for
starting work without them green. Counts come from the generator, never from
memory or from an earlier message — they have changed under us more than once.

### 7. Two traps specific to resuming

- **A later definition silently wins.** If you and an agent both add a function
  with the same name to one module, Python takes the last one and the other's
  tests fail mysteriously. On resume, `grep -n "^def <name>" <file>` any module
  an agent was editing.
- **Nothing that validates the pipeline may live outside the repository.** A
  container update wiped `/tmp` and destroyed the entire test harness once.
  Scratch goes in `/tmp`; anything you would miss goes in `src/tools_batch/`.

### 8. Say where you are before doing more

Post the counts, what is running, what died, and what is blocked — then continue.
Do not silently resume: the owner cannot see the job list, and an unreported
death looks identical to progress.
