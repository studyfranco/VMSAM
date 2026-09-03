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

```sh
git -C <wip-clone> log --oneline -10      # what you had actually validated
git -C <wip-clone> status --porcelain     # what was not even committed there
git log --oneline -8                      # what landed in the main repository
```

**Uncommitted work is still the fragile thing** — one careless `checkout` and it is
gone. But the answer is no longer a copy taken on the side: commit it **in the tier
where commits are free**. That is what those tiers are for, and campaign 1 lost two
agents' finished, correct code to a usage limit for want of it.

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

- **`pgrep -f <name>` matches its own shell** and has reported a dead job as
  alive. Confirm with `ps`, and confirm the log file exists *and is growing*.
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
