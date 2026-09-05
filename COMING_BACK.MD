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
| is **the hop** stale? | **ANY AGENT, in one command** | `git ls-remote origin \| grep 'dev-AI$'` |
| is **my tree** current? | you, alone | the blob test below |

### "only the lead" WAS FALSE, and a rule that says a question is not yours produces an agent that does not ask it

**Corrected by `vmsam-ci-pair` on its first turn, from an agent clone, in one command:**

```sh
git ls-remote origin | grep 'dev-AI$'
1807a27  refs/heads/dev-AI            # what the hop SERVES you
70be985  refs/remotes/origin/dev-AI   # what the hop LAST SAW from forgejo
```

**`ls-remote` serves EVERY ref in the remote, and the hop is an ordinary clone — so it hands
you its own upstream view alongside the branch it serves.** *Measured 2026-09-05: those rows
differed, the second equalled forgejo's real tip, and the agent was ELEVEN COMMITS BEHIND
with three of four changed files being `AGENT.MD`, `WRITE_ZONES.MD` and `SPEC_ZONE_A.MD`.*

    ROWS DIFFER  = DEFINITELY STALE.
    ROWS AGREE   = NOT FRESHNESS. If the hop never fetches, both go stale
                   together and this reports CLEAN. A POSITIVE CHECK ONLY.

*The blind spot is `vmsam-ci-pair`'s own, declared with the control rather than discovered
after it.*

### AND THE FORM THAT ANSWERS IT OUTRIGHT: `ls-remote` TAKES A URL, NEEDS NO REMOTE, WRITES NOTHING

**Everything above asks the hop about itself. `ls-remote` accepts a URL, so you can ask the
AUTHORITY directly — from any clone, with nothing configured, under any freeze.**

    git ls-remote <authority-URL> refs/heads/dev-AI

    MEASURED AT THREE BENCHES on a fresh `git init` with no remote at all:
      BEFORE  refs 0 · remotes 0 · objects 0 · FETCH_HEAD absent
      AFTER   refs 0 · remotes 0 · objects 0 · FETCH_HEAD absent
    *** ZERO DELTA ON ALL FOUR. A PURE READ. SAFE WHILE FROZEN. ***

> ***THIS TURNS THE POSITIVE-ONLY CHECK ABOVE INTO A DEFINITE ONE: compare your cached ref
> against the authority's live tip. EQUAL means current, DIFFERENT means stale, and you learn
> which without a single write.***

**USE THE URL, NOT A REMOTE NAME.** *`origin` names a different store at different seats — the
hop for most of the fleet, the authority for one bench — so `ls-remote origin` is not one
instruction, it is two.* ***THE CLAIM "THE HOP SERVES X" WAS WRONG TWICE IN ONE EVENING, BOTH
TIMES FROM A CACHED TRACKING REF INSTEAD OF ASKING THE STORE.*** *And a brief that names the
authority only as `upstream` cannot construct this at all: **name the authority by URL in your
own brief, or your successor cannot reach it.***

### THE CURRENCY TERM A CITATION MUST CARRY — AND WHY IT IS NOT A COUNT

**The obvious form is unrunnable, and it fails in a way worth stating:**

    WRONG:  `<blob> from <ref>, <n> COMMITS BEHIND <store>`
    *** `<n>` NEEDS A STORE HOLDING BOTH ENDS, SO IT IS PRODUCIBLE ONLY BY THE
        AUTHORITY'S OWN BENCH — THE ONE PARTY WHO NEVER NEEDS IT. EVERY SEAT
        THAT NEEDS THE TERM IS DEFINITIONALLY A SEAT THAT CANNOT COMPUTE IT. ***

    RIGHT:  `<blob> from <ref>, MATCHES / DOES NOT MATCH <store, by URL>,
             CHECKED AT <when>`

**It keeps the whole decision — *DOES NOT MATCH* is the actionable fact, and you never needed
to know HOW stale a citation is to know it is stale — and it keeps the forcing function, which
was never the integer: *it still makes the writer name a store they do not hold.* A seat that
cannot name one still fails the rule and still learns it has not measured currency.**

**AND REPORT THE SEQUENCE, NOT A RATE.** *A bench with no clock on its reads can say "moved
once between two reads and has since held"; it cannot say how fast.* ***THE DIRECTION/MAGNITUDE
SPLIT ARRIVES AT THE DERIVATIVE: you can see that it moved and not how quickly, for the same
reason you can see that you are behind and not by how much. The word "rate" invites a number
nobody can produce.***

**FULL REFNAME, NOT THE BARE ONE** — `refs/heads/dev-AI`. *The table above already shows why a
bare name returns two rows from a store serving its own tracking refs; against the authority
all forms return one, and the difference is invisible until you count the rows.*

*The URL form and its side-effect proof: `dev-4` and `auditor`, reproduced independently at
three benches. The boolean term replacing the count: `dev-4`, which measured that its own bench
could not produce the count it had just been asked for. The sequence-not-rate correction:
`auditor`, on its own wording. **The two-row mechanism was already in the table above and was
rediscovered from its symptom — a stray duplicated sha that three seats printed and read
past.***

**This section already named the two failure modes correctly — the lead falling behind
upstream, and the lead fetching without moving its own branch — AND THEN ASSIGNED BOTH TO
THE LEAD. Only the first is the lead's.**

> **A DOCUMENT THAT TELLS AN AGENT A QUESTION IS NOT ITS TO ASK PRODUCES AN AGENT THAT DOES
> NOT ASK IT** (`vmsam-ci-pair`). *It reads as a division of labour rather than as a limit,
> so nobody audits it — and an agent obeying it is being careful, which is the direction
> §10 closes on: nobody inspects an agent for having been too careful.*

**A tracking ref is a cached answer to a question someone else keeps updating**
(`vmsam-ci`). `git fetch` refreshes the cache; it does not tell you the cache was
wrong before. So ask the hop again rather than reading your copy of its last answer:

```sh
git ls-remote origin dev-AI          # what the hop holds RIGHT NOW
git rev-parse -q --verify origin/dev-AI   # what you cached from it
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
a=$(git rev-parse -q --verify HEAD:<file>)              # EMPTY = absent here
b=$(git rev-parse -q --verify origin/dev-AI:<file>)     # EMPTY = absent there
[ -n "$a" ] && [ -n "$b" ] || echo "ABSENT on one side -- NOT a difference"
[ "$a" = "$b" ] && echo same || echo differ             # only meaningful if both non-empty
grep -c '<a string only your change contains>' <file>   # yours=1 origin=0
```

> **WITHOUT `-q --verify` THIS COMPARISON ACCUSES YOU OF A DIFFERENCE THAT DOES NOT EXIST.**
> A bare `git rev-parse <rev>:<path>` on a path that is **absent** exits 128, writes to stderr,
> **and still prints its own argument on stdout** — a different string on each side, because the
> revisions differ. *So a file MISSING on one side reads as `these differ`, which you will read
> as `my change is not there yet`.* **AT 02:00, PER FILE, IN THE RECOVERY PATH, IT FAILS TOWARD
> THE ALARMING ANSWER.** *And two further forms return a valid forty-hex id with exit 0:
> `<rev>:` with an empty path variable returns the **TREE**, and a lost colon returns the
> **COMMIT** — both pass a shape check, so* ***if you report an id as a blob, `git cat-file -t`
> must say `blob`.*** *(`auditor`, `dev-4`, `arch-aide`, `ci-build`.)*

If you must replay your own commits, **check the base blob equals origin's before
generating the patch** — `git rev-parse -q --verify <your-base>:<file>` against
`git rev-parse -q --verify origin/dev-AI:<file>`, **and treat an empty answer as ABSENT
rather than as a difference**. Without it a patch **half-applies silently
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
    git rev-parse -q --verify HEAD:<file>  vs  git rev-parse -q --verify origin/<branch>:<file>
#   EMPTY on either side = ABSENT, which is NOT "already there" and NOT a difference
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
