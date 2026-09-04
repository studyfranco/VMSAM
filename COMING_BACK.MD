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
revision answers it, and *"have you read it"* cannot.

### Three states, and collapsing them is what produced every wrong resync in this campaign

Your `origin` is **not** the branch. It is the lead's clone, and what you receive as
`origin/dev-AI` is that clone's own local branch. Three separate questions hide
behind one name, and every wrong diagnosis on 2026-09-03 — including two of the
architect's — came from answering one of them and believing it had answered another:

| question | who can answer it | how |
|---|---|---|
| is **my tracking ref** stale? | you, alone | `git ls-remote origin dev-AI` |
| is **the hop** stale? | only the lead | it asks the true upstream |
| is **my tree** current? | you, alone | the blob test below |

**A tracking ref is a cached answer to a question someone else keeps updating**
(`vmsam-ci`). `git fetch` refreshes the cache; it does not tell you the cache was
wrong before. So ask the hop again rather than reading your copy of its last answer:

```sh
git ls-remote origin dev-AI          # what the hop holds RIGHT NOW
git rev-parse origin/dev-AI          # what you cached from it
# differ -> YOUR TRACKING REF is stale. Nothing compared against it means anything yet.
```

This distinction is not pedantry. `vmsam-ci` and the architect reached opposite wrong
conclusions within one hour from the same observation — ci read *"my view differs from
yours"* as *"the branch advanced"* (`merge-base --is-ancestor` said otherwise: the
commit had been squashed away), and the architect read the same difference as *"the hop
is stale"*, which was true when written and false when checked. Neither could tell,
because neither asked the hop.

**And the freshness check below is hop-relative — it cannot fail when the hop is
behind.** Found by `vmsam-dev-2` and `vmsam-ci`, and it is worse than first stated:
what reaches you is the lead's **local** `dev-AI`, so it goes stale both when the lead
falls behind the upstream and when the lead fetches without moving its own branch
(`vmsam-dev-sandbox`, which measured both and named the limit itself — nothing inside
the hop detects a lead that never fetches at all). Run the `ls-remote` above first,
then:

```sh
git diff HEAD origin/dev-AI -- AGENT.MD WRITE_ZONES.MD COMING_BACK.MD   # must be empty
```

### Resyncing: the commit list is not the question, and never was

**This instruction has been wrong three times, in the same way each time.** Every
version — the architect's, the lead's, and the first correction — said *"list the
commits, then decide"*, and each time the per-file check was added as a footnote
instead of replacing the step (`vmsam-dev-1`). `git log origin/dev-AI..HEAD` computes
**reachability**, and a rewrite changes reachability by design; `vmsam-ci` measured 26
listed with 24 of them ghosts, and later aborted mid-rebase with its work missing from
the tree by following it.

**Comparing trees is not the answer either, for two independent reasons.** A tree
differs legitimately whenever you hold a real commit of your own — which was ci's exact
situation. And **a differing tree does not say which direction the difference runs**
(`vmsam-forensic`): one clone that day was **behind and ahead at once**, four documents
behind and two source files ahead. A `reset` there destroys two source files nobody
else has; a `rebase` alone never brings the documents.

**The blob of the file you changed is the only test that has not yet been wrong.**

```sh
git rev-parse HEAD:<file>  vs  git rev-parse origin/dev-AI:<file>   # per file
grep -c '<a string only your change contains>' <file>               # yours=1 origin=0
```

If you must replay your own commits, **check the base blob equals origin's before
generating the patch** — `git rev-parse <your-base>:<file>` against
`git rev-parse origin/dev-AI:<file>`. Without it a patch **half-applies silently
rather than erroring** (`vmsam-dev-1`). Then `format-patch` → `reset --hard` → `am`.

The commit list and the tree comparison are **diagnostics**. They tell you something
is different. Only the blob tells you what, and only the grep tells you whose.

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

**After a history rewrite, the number of commits you are ahead is not a count of
your unmerged work.** `git log origin/<branch>..HEAD` lists commits ahead **by SHA**;
after a squash most of them are ghosts whose content is already in the branch —
other agents' work and your own earlier promotions wearing new hashes. Measured on
one clone: **26 listed, 24 ghosts, 2 its own, and only 1 carrying content the branch
did not hold.** Rebasing the list replays all of it and either conflicts or
duplicates.

    git log --format='%h %an %s' origin/<branch>..HEAD | awk '$2=="<you>"'   # yours only
    git rev-parse HEAD:<file>  vs  git rev-parse origin/<branch>:<file>      # already there?
    git format-patch -1 <commit> --stdout > mine.patch                       # only what is missing
    git reset --hard origin/<branch> && git am mine.patch
    # then verify by CONTENT -- files, not commit counts

**And `git status` does not answer the question.** It says whether work is
uncommitted, which is not whether a reset would lose it: a clean status still sits
on top of commits the branch may or may not hold. **The question is never "committed
or not", it is "is this content in the branch".** Two checks that look like a
partition and are not.

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
python3 -m compileall -q src -x '__pycache__'   # must be CLEAN -- no allowance
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

## 7. Stopping on purpose — a session is restartable when nothing it holds is unique

**Long sessions carry their own retracted beliefs beside the corrections, and nothing
marks which is which except order.** Measured on one architect session: 12 MB, 5898
turns, and **73 readable statements of things it had formally retracted** — a quantum
value corrected hours earlier still readable 29 times, a constant mistaken for a
measurement 16 times. The correction sits further down the file; it is not louder.

**So the signal to restart is not the window filling. It is one of these:**

- **an objective closed** — the premises change there anyway;
- **corrected twice on the same fact** — the session is now reasoning about its own
  history rather than about its data;
- **a role with no natural boundary** — a continuous charter never closes an
  objective, so a cadence has to be invented for it.

**Do not restart between phases of one objective.** Writing a module and debugging it
share the premise *this is how the thing works*, and a restart there discards exactly
the knowledge that makes debugging fast. One defect this campaign found — a success
path that had never completed — was located by pointing a sweep at the author's own
function, which took knowing it intimately.

**Restart at the START of a budget window, never at the end.** A restart costs a full
re-read: this file, the rules, the task records, the roster. On an exhausted budget it
cannot finish, and a half-resumed session is worse than a tired one.

**A process cannot cleanly restart itself** — it would die mid-turn. The restart is
external: end the session's tmux window and recreate it from its own launch script
**without** `--continue`, which is the flag whose whole purpose is to bring back what
you are trying to discard. Other agents' tmux sessions are independent and survive.

**The precondition, and it is the only real work:** whatever the session holds that
nobody else has must be in a file first. For a lead that is the roster — who exists,
what each is on, what was granted to whom. **A lead whose roster lives only in its
window is not a lead you can restart, which makes it a single point of failure rather
than an architecture.** Write it to `TASKS/` and the restart becomes routine.

## 8. Say where you are before doing more

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
