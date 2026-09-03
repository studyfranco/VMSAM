# Coming back — read this ONLY when returning from a stop

**Not on a normal session start.** This is for one situation: you were cut off
mid-work — a usage limit, a context compaction, a killed session, a container
update — and you are picking up. `AGENT.MD` tells you what the project is; **this
tells you what happened while you were gone.**

**The one line**: your own memory is the least reliable source in the room. The
tree, the processes and the numbers have all moved. **Ask them, do not recall
them.** Every step below exists because skipping it has already cost this project
work — including, once, by the author of this file.

---

## 1. Read the state before touching anything

```sh
git log --oneline -8         # what actually landed
git status --porcelain       # what did NOT land — see step 2
cat AGENT_NOTES.MD           # what you were deciding, and why
ls TASKS/                    # what you were asked for
```

## 2. Find your work — it is not where you left it

Three tiers (`AGENT.MD` § Where you work), and a stop hits them differently:

- **`/tmp/…` may simply be gone.** A container update wipes it. That is designed:
  everything you had validated is one tier up, and what was not validated was an
  experiment you can redo cheaply. Do not spend an hour mourning it.
- **`VMSAM_WIP/` is where your real state lives.** `git log` there, not your
  memory — it is the only account of what you actually finished.
- **The main repository should be clean.** If it is not, you were interrupted
  mid-promotion: read what is in the tree before touching it.

**Being level with the branch does not mean holding the current rules, and being
behind does not mean holding stale ones.** Two agents sat a commit behind with
byte-identical documents, because that commit touched none. Only the document's own
revision answers it, and *"have you read it"* cannot:

```sh
git diff HEAD origin/dev-AI -- AGENT.MD WRITE_ZONES.MD COMING_BACK.MD   # must be empty
```

```sh
git -C <wip-clone> log --oneline -10      # what you had actually validated
git -C <wip-clone> status --porcelain     # what was not even committed there
git log --oneline -8                      # what landed in the main repository
```

**Uncommitted work is still the fragile thing** — one careless `checkout` and it is
gone. But the answer is no longer a copy taken on the side: commit it **in the tier
where commits are free**. That is what those tiers are for, and campaign 1 lost two
agents' finished, correct code to a usage limit for want of it.

**Before any `reset --hard`, take a snapshot you can come back from.** Two artefacts,
neither touching the index, and a rewritten branch is exactly when you need them:

```sh
D=/home/vmsam/src/VMSAM_HELP_AI/wip/$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p "$D"
git diff HEAD --binary > "$D/tracked.patch"
git ls-files --others --exclude-standard | tar czf "$D/untracked.tgz" -T -
```

Restore with `git apply` and `tar xzf`. **The path is absolute on purpose**: a
relative `../` resolves differently from a `VMSAM_WIP/<agent>/` clone than from the
main one, so the snapshots would scatter into per-clone directories nobody looks in.
Found by `vmsam-dev-1`. This costs seconds and makes a destructive
resync reversible — which is the whole difference between an instruction someone can
safely obey and one they have to be trusted with.

**Every wait needs a bound and a liveness check — and this is here, in the section
you were sent to, because the section people are pointed at is the section people
read.** `vmsam-dev-1` never opened §3: a `sed` range printed §2 and stopped at §3's
heading, and every later reference to it was quoting someone's summary. Seventeen
unbounded waiters across four agents in one day, the oldest six hours old.

    until ! pgrep -f <script>          # never false: pgrep matches the waiting shell
    until [ -f runs/pass1.json ]       # never true: the producer died six hours ago

Neither believed anything false. Both simply had no terminating condition and nothing
checking that what they waited for could still happen. **And when you kill a producer,
kill its waiters** — nothing warns you at the moment your own action makes a wait's
premise false (`vmsam-ci`, which killed the producer correctly and stayed wedged on
its own waiter).

**Files you did not create are a deliverable until proven otherwise.** Read them
before assuming they are junk.

## 3. Look at what you matched before you kill it

If you left processes running, identify them by **parentage**, never by a path
substring — the temp trees are shared, so a grep on a path matches other work
too. Campaign 1 came one step from killing another agent's 80-minute decode; the
two processes were indistinguishable to a path filter and obviously different by
`ppid`.

```sh
ps -eo pid,ppid,etime,cmd | grep ffmpeg | grep -v grep   # LOOK, then act
```

Two traps that have each fired here:

- **`pgrep -f <name>` matches its own shell.** This paragraph used to warn only that
  it reports a dead job as *alive* — the mechanism, with the wrong consequence
  attached. The failure that actually cost four agents hours is **blocking** on it:
  `until ! pgrep -f <x>` can never become false, because the waiter is the match. A
  rule that names a defect by the wrong symptom is worse than no rule; it inoculates
  the reader against noticing. Confirm with `ps`, and confirm the log exists *and is
  growing*.
- **`kill $!` on a wrapper does not stop the job.** Under `nohup nice -n 15
  python3 …`, `$!` is the pid of `nice`. Killing it **reports success and leaves
  python running**, reparented to init. Verify by finding the real process, never
  by trusting the kill's exit status.

## 4. Nothing the test instance produced is evidence until you check what ran

A stale instance produces results that look fine and prove nothing. Before
reading any result: confirm the revision under test is the revision you mean, the
way its own rules say to. `AGENT.MD` § Target-first — you do not operate it, and
what you do not remember about it is in `VMSAM_HELP_AI/`, not here.

## 5. Re-verify, do not re-derive

```sh
python3 -m compileall -q src -x '__pycache__'   # one expected failure, see AGENT.MD
```

Then the checks in `WRITE_ZONES.MD` §5, before anything else. **Numbers come from
the source that generates them, never from memory and never from an earlier
message** — counts have moved under this project more than once, and a figure
carried forward in prose is how a wrong one survives.

## 6. Two traps specific to resuming

- **A later definition silently wins.** If two edits added a function of the same
  name to one module, Python keeps the last and the other's tests fail
  mysteriously. `grep -n "^def <name>" <file>` on any module you were editing.
- **Scratch goes in `/tmp`; anything you would miss goes in `VMSAM_HELP_AI/`.** A
  container update wiped `/tmp` and destroyed a whole harness once. `/tmp` is not
  storage, and the repository is not the answer either — see `WRITE_ZONES.MD` §4.

## 7. Say where you are before doing more

Post what landed, what died, what is blocked, and what you are resuming — then
continue. **Do not silently resume.** An unreported stop looks exactly like
progress from the outside, and it is the difference between a delay and a
surprise.

## 8. If you lead a team, quiet is ambiguous

**An agent that has stopped looks exactly like an agent that is thinking.**
`SendMessage` is push-only: nothing tells you when a peer finishes its turn, and
nothing tells you when one is parked mid-turn. The session stays alive and blocked
**indefinitely, because nothing external pokes it** — a message resumes it at its
next tool round, and until someone sends one it waits forever.

- **Poll rather than assume.** `ListAgents` reports each peer as busy or idle. Idle
  after a dispatch means *finished* or *never started* — it never means "still
  working on it".
- **Make the reply part of the task.** Require a message when a task completes **and**
  when it blocks. Then silence always means a fault, instead of meaning either.
- **A usage limit parks every session on the account at once**, yours included. When
  yours resumes, theirs do not: they are waiting for a message only you can send.

This cost campaign 2 its first hours: four agents idle, the lead running and unaware,
each side waiting for the other.
