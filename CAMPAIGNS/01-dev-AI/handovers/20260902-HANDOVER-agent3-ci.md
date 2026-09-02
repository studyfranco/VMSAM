# Handover — Agent 3 (`vmsam-agent-ci`): testing, deployment, the queue, the watchdog

**Written** 2026-09-02, after every job was stopped on the owner's instruction and the
campaign ended. **Scope**: the infrastructure everything else ran on — the test queue, the
deploy loop, the merge sweep, the watchdog, the session compactor, and the measurement
hygiene around them.

**Three rules I wrote this under.** Files are cited as `err_<id>` only. **Measured and
inferred are separated everywhere and labelled.** And the story is not tidied: the errors
below are mine unless attributed, and the ones that were caught by someone else say so.

**One thing to read first.** Every count in this document is a **snapshot with an instant
attached**. The ledger moved under two agents while we were both measuring it this morning
and produced two different correct answers (§3.4). A count quoted without its timestamp is
how `step1_verdicts.json` became a four-day-old snapshot of a ledger that had moved on.

---

## 1. What is true

### 1.1 T4 — `err_150` is measured, and the reading was fixed before the run

**Measured.** Confidence: **high**, and it is the highest-confidence result in this
document because the reading was pinned in the repository (`ca6d03d`) **before** the sweep
reached the file, so it could not be chosen to suit the answer.

    recorded_utc    2026-09-01T00:22:35Z  ->  2026-09-02T08:50:12Z
    build           afa72e7587f1          ->  e75c78386dd0
    outcome         MERGED_UNVERIFIED     ->  MERGED_IN_SYNC
    evidence_class  "instrument fault"    ->  "multiple windows, agreeing (spread 2.7 ms)"

    fre  in_sync  max 3.3 ms
         0.6 / -1.0 / -3.3 ms at peaks 0.991 / 0.990 / 0.969
         compared = "output vs master file"

**It agrees with a second, independent instrument.** Agent 1's C3 reading was `fre in_sync,
worst 3.3 ms`. Two paths, arrived at separately, one value. Inside the 15 ms bar.

**The `jpn` leg is what makes the `fre` number mean anything, and it is the part that would
be lost in a summary.** `jpn` returned `unverifiable` and said why: *the only `jpn` track is
bit-identical to the master's; this compares the master with itself* — **0.0 ms at peak
1.000 on all three windows**. So the multi-track trap in `DOMAIN_INVARIANTS` **fired and
named itself** rather than passing silently, and it doubles as a **negative control**: a
self-comparison here reads exactly 0.0/1.000, so `fre`'s non-zero lags at peaks below 1.0
cannot be one.

**What would overturn it**: evidence that the output's `fre` track is master-derived rather
than candidate-derived. The 0.0/1.000 control argues strongly against it, but the direct
check — that the output carries more tracks than the master, the pigeonhole argument — was
not run on this file.

**Not marked `step2_passed` by me.** The queue marks rows `deployed`, never `passed`. The
Lead judged it and accepted it; a service that graded its own homework is what that rule
exists to prevent.

### 1.2 The orphan output is stale, by mechanism rather than inference

**Measured.** Confidence: **high**. Both files were probed; nothing here is reasoning.

The unrecorded ~3.9 GB output T4 was written about survives in `baseline_before_remerge`
(3,898,254,651 bytes, md5 in the manifest). Against the run's new output:

    53 streams each, identical order, codecs, languages, titles, dispositions
    duration 1432.014000 in both
    1082 bytes apart on 3.9 GB
    EXACTLY ONE structural difference across all 53 streams:
      the new output carries CHIMERIC tags on 29; the orphan carries none

So the orphan predates the tagging work — **different code, therefore stale**, which is
exactly the conditional T4 wrote in advance.

**A second finding fell out of it, and it is the strongest evidence the tagging is
correct**: the tagging is **discriminating, not blanket**. The one audio track that came
through from the master untouched is the one track with **no** tag.

**What would overturn it**: a build of the pre-tagging code that also emits tags, or
evidence the orphan was stripped of tags after writing.

### 1.3 The instrument-fault population, and its boundary

**Measured.** Confidence: **high**, and it **strengthened** under 139 new rows.

`tools.software['ffmpeg'] is not configured` — `_binary()` correctly refusing to guess a
PATH binary for a measurement — reached the **verdict slot**, where it is indistinguishable
from a finding about the file. The refusal is right. Its landing place is the defect.

    at 2026-09-02 ~08:00Z    29 records affected
    at 2026-09-02 ~11:00Z    26   (three were re-run clean by the sweep, err_150 among them)

    affected window   2026-08-31T18:22:06Z .. 2026-09-01T02:39:03Z
    clean window      2026-09-01T06:32:34Z .. 2026-09-02T08:50:12Z
    INTERLEAVED       False

**The two populations do not straddle**, with a gap of about four hours and nothing in it.
**So the affected set is closed by its own content and never needed dating from a commit** —
which matters, because the host-side fix landed 2026-08-31 20:47 UTC and **fourteen affected
rows were written after it**, by a long-running `batch_verify` still holding the
pre-fix module. A reconciliation by commit date would have placed those fourteen wrong.
**Read the note, not the clock.**

**What would overturn it**: an affected row appearing inside the clean window. None has,
across 139 rows re-recorded after the finding was made — which is the part that upgrades
this from a snapshot to a result.

### 1.4 The `build` stamp names the wrong process

**Measured.** Confidence: **high**.

Every `per_language` row carries a `build`, so "the row carries no code stamp" is wrong —
but `batch_verify.py:533`/`:559` set it from `deployed_build()`, **the container's SHA, not
the host module's**. The 26 affected rows span **two** container builds while sharing **one**
host defect.

**A stamp that names the wrong process is worse than none, because it looks like
provenance.** `batch_verify.py:155-157` already records this hazard for a different case —
one `build` value silently spanned two runs, which is why `recorded_utc` was added at all.

### 1.5 Ledger snapshot at handover

**Measured** 2026-09-02 ~11:00Z, after the sweep ended at 08:58:30Z. **Counts come from the
generator, not from here** — re-run `master_summary.py` and `build_state.py`; this is a
snapshot for orientation only.

    rows 315      outcome: FAILED 201 · MERGED_IN_SYNC 67 · MERGED_UNVERIFIED 45
                           MERGED_OUT_OF_SYNC 1 · None 1

    evidence_class_for: never measured 233 · multiple windows agreeing 47
                        instrument fault 26 · no samples 8 · one window 1

Final sweep: ended 08:58:30Z at **C_cut 55/179**, past `err_150` at item 52, **139 rows
re-recorded**. Container left idle at `e75c78386dd0`, `is_running false`, `queue_length 0`,
**no orphaned `PROCESSING` job**.

---

## 2. What is wrong and still wrong

### 2.1 FIXED today — `evidence_class` collapsed absence into emptiness

**The defect.** `if not per_language: return NO_SAMPLES` is true for both a missing key and
an empty dict, and the distinction was already destroyed one `.get()` **before** the
function was entered. "No samples" asserts that a measurement ran and found nothing. For
most of the corpus, no measurement ever ran.

    before   241 "no samples"    (234 absent + 6 empty + 1 present-but-unusable)
    after    233 "never measured" + 8 "no samples"     [my run, ~11:00Z]
             234 "never measured" + 7 "no samples"     [the Lead's run, ~10:00Z]

**Both are correct.** One row gained an empty `per_language` between the two runs — the
ledger moved. **The counts differing is the fix working, not the fix failing**, and I record
both rather than picking the tidier one.

**The fix is right in its shape as well as its result**: a new `evidence_class_for(record)`
at the call site, leaving `evidence_class` meaning what it always meant for a mapping that
exists. Fixing it *inside* `evidence_class` was impossible — the information was gone before
the call. I found it and did **not** write it: it is not my file, and two writers in one
module is a collision this project has paid for repeatedly.

### 2.2 OPEN — there is no host-side equivalent of the deployment gate

We gate the container on `git_commit == HEAD` so its results mean something. **Nothing does
this host-side, and §1.3 is what that costs**: fourteen rows produced by pre-fix code after
the fix existed, with nothing in the record to show it.

**My design, and the reason the obvious version does not work.** Do **not** stamp
`git rev-parse HEAD` at write time — that reports the checkout, and **a running process's
code is not its checkout's code**, which *is* the defect. **Stamp at import time, in the
process that writes the row**, so a long-running sweep carries the revision it actually
loaded. Small change, squarely Agent 3's, **not built** — it is with the owner.

### 2.3 OPEN — the rest

- **`classified.json` has no producer** anywhere in the tree and **seven consumers**. Written
  once, never regenerated.
- **`step1_verdicts.json`** is a stale snapshot of a ledger that moved; the mission's 15 ms
  bar has never fired on any file, because `final_status` takes the lag from one store and
  the verdict that gates the bar from another.
- **`batch_state.json` has one read-modify-write writer and has already lost 33 verdicts** to
  a second. Nothing structural prevents a recurrence.
- **`reverify_in_place` deliberately does not write `batch_state`**, and must not start —
  that was the fix for real data loss. The cost of the fix was never priced.

---

## 3. What we misunderstood, and for how long

### 3.1 The centrepiece — I said `err_150`'s languages were measured. They were not.

I wrote in the T4 row that both shared languages "were measured, neither skipped nor
mis-selected". **I wrote it from a summary of the row rather than the row.** The row says:

    per_language.jpn  verdict "unverifiable"
    per_language.fre  verdict "unverifiable"
      both: external probe failed: tools.software['ffmpeg'] is not configured

Both were **attempted**; **neither was measured**. The verdicts read `unverifiable`, not
`indeterminate`. The cause sits **upstream of language selection entirely**.

**What produced the error**: reading a summary and treating it as the record. **What caught
it**: reading the record. That is the whole lesson and it took about a day to surface.

**The conclusion survived and got stronger** — the `batch_verify.py:282` selection
amendment cannot explain the row, because *no* selection this file could have made would
have produced a number. But "both measured" credits a measurement that never ran, and the
Lead was actively building on it, so it did not get to stand. **Corrected before it was
built on further**, which is the only reason this counts as a near-miss rather than a defect.

### 3.2 My `.get()` near-miss — the same defect, twenty minutes later

Checking whether rows carried a code stamp, I read `record.get("git")` as `None` and was
about to report "**no stamp**" for a row that **carries one** — the key is `build`, and
`.get()` returns `None` for an absent key exactly as it does for a null value.

**Absence and null are two outcomes; `.get()` collapses them into one.** It is the same
shape as §2.1, and as an error I made two days earlier reading "92 records have an empty
field" when they lacked the key entirely — which would have justified weakening a filter
that was working. **Three instances of one defect in one morning**, at three layers, each
found by a different person, none of them looking for it.

### 3.3 The three `/compact` refusals — including the one relayed to me

The owner needs each session compacted about every five hours. **No session can do that to
itself** — `/compact` is typed into a pane, not a tool any of us holds — so the instruction
was passed agent to agent, and **three sessions refused in a row, including when the Lead
told me two peers had already refused and then asked me.**

**The refusals were correct.** A peer saying "the owner asked for this" cannot be verified
from inside another session, and **a request declined elsewhere and then handed to the next
session is the shape of routing around a refusal**, whatever the request happens to be. The
instruction was in fact genuine, which does not make the refusals wrong: none of us could
tell a real relay from a mistaken one, and *that* is the property that makes relayed
authority weak.

**The fix was to remove the agent from the loop, not to convince it.** A timer has no
authority to launder and needs nobody's trust. **The restart needs this rule intact** — it
is the thing standing between "a peer asked me to" and an actual escalation.

### 3.4 A count is a snapshot, and we produced two correct ones that disagreed

§2.1 has two different right answers an hour apart. Neither is an error. **Quoting a count
without its instant is precisely how `step1_verdicts.json` went stale**, and we reproduced
the mechanism in miniature while documenting it.

### 3.5 The shutdown order — the Lead's error, recorded as an error

`deploy_queue.sh` is the sweep's **parent**. Killing it first **orphans** the sweep rather
than stopping it: it reparents to init and keeps running while the process tree makes it
look dead. It was killed first. **It came out clean only because all four signals landed
inside the same second and no `ffmpeg` was mid-file — luck, not design.** The correct order
is in §6.4. Recorded here because "it worked" and "it was right" are different claims, and
this project has been bitten by the gap between them all week.

---

## 4. Unknown / unresolved

### 4.1 `(fre, shared_reference)` — content or timing? Not measured, not guessed.

`err_150`'s output carries, on its audio tracks:

    fre                  "patched 51.5 s from master (fre, same_dub)"
    ger eng spa ita por  "patched 51.5 s from master (fre, shared_reference)"

**Measured, and pigeonhole-certain**: the master carries **only two** audio tracks, `fre` and
`jpn`; the candidate carries seven. **For those five languages the master has no track in
that language at all**, so a 51.5 s gap patched "from master" is **cross-language by
construction**, whatever the content turns out to be. The `fre` case is the one where a
same-language master track exists, and there the dub gate fired correctly.

**Not established, and I did not guess**: whether `shared_reference` means the audio
**content** came from the master's `fre`, or only that `fre` was the **timing reference**.
The tag is ambiguous.

**Why it matters**: `AGENT.MD` criterion 4 says fall back to the **master VO**, and the VO
here is `jpn`, not `fre`. If the patch source is chosen by the *shared reference language*
rather than by the VO rule, those five tracks contain 51.5 s of French — the *detectable*
failure rather than the silent one, but not what the rule specifies.

**Settling it needs a decode.** I offered and was told not to; it is **open in Agent 1's
territory** and they have been told not to investigate it for now.

### 4.2 The `err_211` shape batch — never run

8–10 runs, pre-registered at `endpoint_shape_registration.py` with every threshold pinned
while all three verdicts still read INDETERMINATE. **Queued behind T4, then behind the
sweep, then overtaken by the stop.** The registration is the valuable part and it survives;
the batch is unrun.

### 4.3 Seen and not explained

- **pid 2692015.** Established as the sweep's parent via four independent signs. Whether a
  second poller ever existed was never settled — it was a thing to watch after the sweep
  ended, and the stop overtook it.
- **`err_211`'s truncation emits no diagnostic at all.** Reproduced three times: seven tracks
  in at ~1441.947 s, six out at 441.448–441.485 and `jpn` at 1441.942 — the `-t` value
  exactly. All starts 0.000 in and out, so the `-copyts` ceiling is **refuted corpus-wide**
  (631 tracks, largest start 1.98 s). **The complete stderr contains zero diagnostic lines.**
  Cause unknown.
- **The 1082-byte difference** between the two `err_150` outputs is container/metadata scale,
  not content — consistent with a fresh mux — but was not itemised field by field.

---

## 5. What to discard

Blunt, as asked.

**5.1 The sweep architecture as it stands — yes, this is one of them.** One long-running
`batch_verify` holding an imported module for hours is what produced §1.3: **fourteen rows
written by code that had already been fixed**, invisible in the record. It also makes the
class the unit of recovery, so stopping at C_cut 55/179 discards the class's progress rather
than the item's. **A restart should not inherit "one process for a whole class."**

**5.2 `step1_verdicts.json` as a verdict store.** It is a point-in-time snapshot with no
timestamp of a ledger that moves. **36 of its 40 rows changed outcome underneath it** and 46
eligible rows were never audited. Do not migrate it; regenerate the concept.

**5.3 The `deployed`-every-120-seconds queue behaviour** that re-deployed T2 for ~20 minutes
while ten sweeps wrote one state file. The marker defect is fixed; **the results from that
window remain untrustworthy and should not be inherited as data.**

**5.4 Any count carried forward in prose.** Regenerate from `master_summary.py` /
`build_state.py`. Every count in this document has an instant attached for exactly this
reason, and they have all moved before.

**5.5 The five-hourly cron wake-up** (`a4f59635`) — session-scoped, dies with the session
that made it, and auto-expires. It is not a service; do not treat its absence as a failure.

---

## 6. What to keep

**6.1 The queue marks `deployed`, never `passed`.** The Lead judges the output against the
criteria written in the request. **A service that marked its own work passed would be
grading its own homework**, and today is a live demonstration: I produced T4's result and
did not accept it.

**6.2 `ci_ready.py` asks CI rather than sleeping.** It asks the Actions API whether a
completed successful run exists **for exactly this SHA**. The documented "~10 minutes" was
wrong — measured runs take 1m09s–1m49s — and a fixed 780-second wait deferred deploys for
four hours while reporting itself healthy. **Three outcomes, and the third is the one that
matters: "cannot tell" is not "not ready."**

**6.3 The compactor's release-on-observation.** `compact_sessions.sh` never sends to a busy
pane, and **never sends the follow-up unless it has observed the compaction finish** — two
signals over consecutive samples. On timeout it logs `UNVERIFIED`, **withholds** the
follow-up, and does **not** record the session as compacted. Its first firing held **7m21s**,
which is what proves it releases on observation rather than on a clock; verified by finding
the follow-up text in the pane and watching the session act on it, **not by an exit code**.

**6.4 The shutdown order** (§3.5): **watchdog first** — a sweep being torn down looks exactly
like a hung job; **then `full_test.sh`**, so it does not start the next class; **then SIGTERM
the python**, so it completes its `batch_state` write; **then the queue**, which is the
parent. And **kill the python, not the wrapper**: `kill` on a wrapper reports success and
leaves the job running, reparented to init.

**6.5 Logs that matter do not live in `/tmp`.** A container update wiped `/tmp` here once and
destroyed the entire test harness. All four service logs and the compactor state were in
`/tmp` at shutdown; they are preserved **outside the repository** under `VMSAM_PRIVATE/`,
because the sweep log is a list of media filenames and the privacy checker would rightly
reject it.

**6.6 — the practice, above all the tools: pin the reading before the run.** T4's result is
checkable by anyone because the values *and the instrument* were committed to the repository
before the sweep reached the file. Choosing the instrument afterwards lets the reading be
picked to suit the answer, and choosing it beforehand costs one commit. **Everything in §1
that is high-confidence is high-confidence because of this, and nothing else in this
document did as much work.**

**6.7 `gate.sh`, never an inline pipeline.** `check_no_private_refs.py --staged | tail -1 &&
git commit` is **not a gate**: a pipeline's exit status is the last command's, so `tail`
returns 0 whatever the checker found. Two commits landed this way that the checker had
rejected — the check ran, printed its refusal, and the commit went through underneath it.

---

## 7. Standing instruction, suspended and not cancelled

I hold an instruction given to me **directly by the owner**: *check the logs periodically,
and when no fusion is in progress, update the container.*

**Every job is stopped, so the condition is satisfied and I am deliberately not acting on
it** — updating now would deploy into a project being rebuilt from a new basis. **I am not
treating a relayed directive as cancelling an instruction the owner gave me directly**, and
the Lead explicitly declined to rule on it, which was right.

**So it is recorded here as suspended, not cancelled, and the owner decides which they
meant.**
