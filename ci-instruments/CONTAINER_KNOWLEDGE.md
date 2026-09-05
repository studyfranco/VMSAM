# The test containers — what ci is the only one who knows

**PRIVATE.** Written because the owner ruled ci is the **entry point and single
authoritative reader** of the test containers, and *"quand il est reset, c'est le seul à
se souvenir des infos"*.

    ANYTHING I AM THE ONLY ONE WHO REMEMBERS MUST LIVE IN A FILE, NOT IN MY CONTEXT.
    A RESET ERASES THE SECOND AND NOT THE FIRST.

**If this file is thinner than it was, the fleet has lost its container knowledge — and
no `curl` will tell you, because a container answers whether or not anybody knows what
it is.**

---

## 1. The four references, and being current on one is not being current

**There is no single thing to be "up to date" with.** This is the whole of `WRITE_ZONES.MD`
§14 applied to my own zone:

| reference | what it is | how to read it |
| :--- | :--- | :--- |
| forgejo `dev-AI` | the published branch | `git ls-remote <URL> refs/heads/dev-AI` — **live, never a tracking ref** |
| the hop | `/home/vmsam/src/VMSAM`, what several agents' `origin` points at | it can be AHEAD of forgejo; check both directions |
| the registry tag | what an image was built from | `deploy.py --check` prints `published:` |
| the running container | **what the code IS** | `GET /health` → `git_commit` |

**These diverge routinely.** Tonight the hop was ten commits stale for an hour; my own ci
clone was 57 behind and produced a confident false negative about a rule that existed.

## 2. The containers

    showgestionar-test    http://showgestionar-test:8080     TODAY'S ONLY CONTAINER
    showgestionar-test2   ANNOUNCED, DOES NOT EXIST YET      see SECOND_CONTAINER.md

**Endpoints I use, and what each is worth:**

    GET  /health    git_commit, mode, dev, is_running
    GET  /fusion    status, queue_length, current_job.started_at
    GET  /errors    the corpus -- `incompatible_files` is a LIST (not a dict),
                    315 entries, 38 distinct `folder_id`
    POST /fusion    the decisive probe -- and it costs a real job

**`is_running` from `/health` is not trustworthy for waiting:** it reported `JOB_DONE` at
t+271 s on a job that ran another 105 minutes. **Poll for the ARTEFACT and the queue's
status instead.**

**`HTTP 200` on `GET /fusion` does NOT prove the worker started** —
`get_internal_fusion_status` carries no `fusion_enabled` guard, so it reports an idle
queue whether or not the worker is running. **The first `POST /fusion` is the real probe.**

**All six endpoint call sites are parameterised on `VMSAM_TEST_HOST`** (`deploy.py`,
`merge_queue.py`, `start_when_idle.sh`, `targeted_probe.py`,
`test_merge_plan_report.py`). Default is today's container. **Controlled both ways: a
wrong aim reports `no answer from /health` rather than answering about the wrong
apparatus.**

## 3. The deploy sequence, and what each guard means

`deploy.py --target <40-char sha>` does, in order:

1. **wait for the published image to carry the target** — firing early re-pulls the
   *previous* image and redeploys the old commit, **which looks like a successful deploy**.
2. **`OVERTAKEN`** — if the registry tag has been rebuilt to a *different* commit while
   waiting, it **refuses** rather than deploying a stale target. It fired tonight and the
   re-target picked up a fourth agent's change for free.
3. fire the recreate hook · 4. verify `/health` against the target.

**`deploy_condition.py` answers MET / NOT MET / **CANNOT TELL** — never a boolean.** It:

- resolves the tip from a **live `ls-remote`**, never a tracking ref;
- checks freshness **symmetrically** (the hop can be ahead of forgejo);
- **compiles every file in `src/`** — measured on the poison ref `f99bb98`, which had a
  call site and a locator that died at line 159. *A call site is an existence check on a
  token; compiling interrogates the thing.*
- matches wiring as `import merge_plan_report` or `merge_plan_report.<name>(`, **not the
  bare name** — a docstring citation counted as a call site and moved the number.

**Credentials live only in `/home/vmsam/src/hook_executor_parameters`.** Reference by
path; **never inline, never in a commit, a branch name or a tracked file.**

## 4. THE RUNNER MUST BE RESTARTED AT A RECREATE, AND ONLY THERE

`merge_queue.py` reads the container image **once at startup** and stamps every row.
Recreate mid-run and every subsequent row names the wrong image.

**A mid-run restart risks a double-POST** — the container keeps working on the job the old
runner was tracking. **That cost is already paid at a recreate: the hook has just killed
that job.** So a recreate is the only free moment, and `deploy.py` prints a reminder
saying so.

**Two clients POSTing to one container voids a run** (`runs/merge-queue-VOID-two-clients.jsonl`).
**Never fire a targeted probe while the corpus run is live.**

## 5. What the sweeps cover, and why those extensions

`preserve_artefacts.sh`, two sweeps:

    MAIN     root /config/output/srv
             *.mkv *.log *.log.error *.REFUSED.* *.NOVERDICT.* *.merge_plan.html
    SECOND   root /config/output, pruning KEEP and DECLINED
             *.REFUSED.* *.NOVERDICT.* only -> /config/output/DECLINED

**The second sweep exists because every declined artefact on this disk is OUTSIDE the
main root.** Rooting the main sweep one level up would drag **509** of other agents' files
into KEEP.

**The whitelist names any extension under its root it does not cover** — *an empty line
there is a measurement; the absence of a line was nothing at all.*

**Retention, per the owner: delete under DISK PRESSURE (30 GB floor), never after
verification.** `.log` and `.log.error` are **never** evicted. Eviction order:
validated → unread → **REFUTED LAST**.

**A produced `.mkv` is named from the MASTER's basename; a `.log.error` from the ERROR
file's.** Those two bases structurally cannot join — which is why `error` joins 3/24 and
that is a limitation, not a fault.

## 5b. MY OWN COPIES ARE IN MY OWN POPULATION — use `census_population.py`

**I am the only agent that makes copies, so I am the only one who can count them twice.**

    /config/output/KEEP/*.error       THE CONTAINER'S OWN RECORD -- canonical
    runs/decline-<id>.log.redacted    MY REDACTED COPY OF ONE -- derived

**A raw glob over both reported REFUSED 4 where there were two jobs.** The rule is not
*exclude mine* — that loses coverage:

    CANONICAL FIRST. A DERIVED COPY EARNS A PLACE ONLY BY COVERING A JOB
    NOTHING ELSE COVERS, AND IT SAYS SO.        91 files -> 85 jobs

**Identity is `candidate_digest`, else the plan-piece string, else the filename.** The
last case may over-count and **the bound is NOT MEASURABLE** — I tried a
redaction-surviving numeric key and **it merged four distinct jobs** (`decline-20/21/22/23`,
four contents, four ids, one signature). *Over-counting gets questioned; merging is
invisible and reads as a cleaner corpus.*

**And the first deduplicator I wrote keyed on `candidate_digest` ALONE and skipped what
it could not key — turning a double count into a disappearance while the output looked
cleaner.** *A deduplicator that drops what it cannot key is a filter pretending to be a
counter.*

## 6. The gates I run, and the rule under all of them

    THREE OUTCOMES, NEVER A BOOLEAN. "cannot tell" is a result.

- **A checker refuses to report a count until it has produced every outcome on cases
  whose answer is known.** `check_pairing.py` self-tests AGREE / DISAGREE / UNRESOLVABLE
  before it will score anything.
- **A positive control on the FINDER, not the data:** verify the query would find
  something if something were there, before believing it found nothing. **A query built
  on an unverified structural assumption returns a clean, confident, well-formed ZERO.**
- **Process identity is `basename(argv[1])` from `/proc/PID/cmdline`, excluding a match's
  own children** — never a substring of `ps`. Substring matching killed my own shell
  three times and invented a duplicate preserver twice.
- **Cite by PHRASE, never by line number.** A line number does not survive a commit.
- **TWO OPPOSITE KEY DEFECTS, AND EACH PASSES THE CHECK WRITTEN FOR THE OTHER.**
  *A key that matches NOTHING joins nothing — the symptom is an empty join.* **A key
  that matches EVERYTHING joins everything, and reads as a healthy coverage figure.**
  My ledger had both: a report stem that could never join its artefact, and 18 rows
  sharing the literal value `UNKNOWN-backfill` counted as 18 mutual joins.
  **Only asking what the matched rows ARE separates them.**
  The test, with no threshold and no format assumption: **a job produces at most one
  artefact of each kind, so a valid stem covers at most one row per kind — PER INODE,
  not per name.** The inode form survives a ledger merge; the per-name form is safe
  only while `ci-preservation-ledger` and `ci-declined-ledger` stay apart, and three
  container refusals already exist under two names on one inode.
- **A SWEEP CAN BE BLIND TO THE DEFECT IT WAS WRITTEN FOR.** My first degeneracy sweep
  looked for repeats in *mostly-unique* columns; a degenerate key has *few* distinct
  values. It reported three false positives and could not see the real one.
  **`check_degenerate_keys.py` now refuses to report until it re-finds a defect whose
  answer is known.**
- **A CLEAN RESULT NEEDS ITS OPPORTUNITY COUNT.** `0 of N carry the defect` means
  nothing without `the condition was present in M of N`. **With M = 0 it is UNTESTED,
  not PASSED** — a green light with nothing behind it. Measured: 12 of 12 delivered
  artefacts carried no wrong-language fill, **and the condition arose in ZERO of them**
  because eight were single-language and four were fully covered. **The pipeline was
  never asked the question.**
- **A LIVE CHECK IS READ AS ONGOING.** Both wrong-language artefacts were captured
  five hours before the current run, by an earlier image. *"Two artefacts are
  wrong-language"* and *"this run is shipping them"* are different claims. **A standing
  PROBLEMS line must carry WHEN, or it asserts the second.**

## 7. Standing limits — things no gate here can see

    a uniformly shifted subtitle track   correct duration   currently EMPTY
    an unnecessary interior fill         correct duration   OCCUPIED
    a WRONG-LANGUAGE audio fill          correct duration   OCCUPIED, AND IT SHIPPED

**`FULL_LENGTH` is durations and coverage. No frame and no sample has ever been
compared** — by me or by anyone. **The owner's bar is a person comparing a produced file,
and it has never been met.**

## A TABLE LIVES IN THE ZONE IT DESCRIBES, AND IS NOT A MEMBER OF IT

`ci-declined-ledger.tsv` sits in `/config/output/DECLINED/` beside the artefacts it
records. A count of "files in DECLINED" therefore counts the ledger, and the ledger is
not a declined artefact. That reads as `12 files, 11 rows` -- a missing row that does
not exist.

THIS IS THE THIRD TIME. The degenerate-key sweep flagged my own `-` markers; the orphan
check flagged `vmsam-forensic`'s `.bak`; now the reconciliation counts my own ledger.
Every instrument I write deposits its output inside its own input.

**Rule: any count over a zone filters to the artefact extension first** (`grep '\.mkv$'`),
never `ls | wc -l`. A discrepancy of exactly the number of tables in the directory is the
signature -- check that before looking for a lost artefact.

And the count that raised it was taken mid-cycle, so it carried a second error at the
same time: a file is hard-linked before its row is appended, and a reconciliation run
between those two moments reports a gap that closes by itself. **Reconcile against a
quiet directory, or report the gap as a time, not a fault.**

## 8. A SHARED FIELD NAME IS NOT A SHARED QUESTION

`runs/` holds 24 files. Four are `-VOID-`. Of the rest, **only 9 speak the delivery
vocabulary** (`FULL_LENGTH`/`DECLINED`/`COULD_NOT_MEASURE`/`TRUNCATED`). The others reuse
the field `verdict` for a different question entirely:

    pair-sweep-trace.jsonl   AMBIGUOUS . CONTENT PRESENT . WRONG PAIR
    probe-trace.jsonl        CARRIES x1.001001 . DOES NOT CARRY x1.042709

**id 33 reads `CONTENT PRESENT`, `CARRIES x1.001001` and `COULD_NOT_MEASURE` across
three files.** Three questions, not three answers. Anyone scanning `runs/*.jsonl` for
`verdict == 'FULL_LENGTH'` merges a content check, a framerate probe and a delivery
outcome into one count that looks perfectly well-formed. Use `accepted_set.py`, which
classifies by the file's WHOLE vocabulary and excludes rather than includes on doubt.

**And accepted is a property of `(id, build)`, not of `id`.** Five ids disagree across
runs; four are `DECLINED` under one build and `FULL_LENGTH` under another. A manifest
with one row per id asserts something no run ever measured. The tool reports the
disagreements and refuses to resolve them — the later run *looks* better, and looks
better is not a measurement.

**The accepted set is 28 ids of 67 ever given a delivery verdict, out of a 315-entry
corpus.** The other 248 are **not "not accepted"** — they are unattempted.

## 9. MY LEDGER IS NOT THE ACCEPTED SET. IT IS THE REPAIR PATH OVER THE REFUSAL CORPUS.

`vmsam-ci-pair` joined my 28 `FULL_LENGTH` ids against their 315-entry `/errors`
snapshot: **28 of 28 present, zero absent.**

    EVERY ID I RECORD AS ACCEPTED IS INSIDE THE REFUSAL CORPUS.

Which means this ledger does not measure "files the pipeline acted on in normal
operation". It measures **previously-refused files that a repair run, in a TEST
container, subsequently delivered**. Those are not the same population and the second
one is not a sample of the first.

**The production accepted set remains unmeasured by anything I hold.** Every statement
I make about "what ships" is scoped to the repair path unless it says otherwise --
including the wrong-language fill finding, which is a defect in *repair output*, not a
measured rate over production deliveries.

Neither side could see this alone: I had the verdicts and no corpus membership, they had
the corpus and no verdicts. **It came from putting two populations beside each other
that had never been beside each other** — which is the only operation that catches a
substituted population, and it is cheap exactly once, before the column is written.

## 10. WHAT AN IMAGE CARRIES IS A QUESTION ABOUT THE BLOB, NOT THE BRANCH

A commit being "on `dev-AI`" does not put it in the running container, and a monitor
reporting a blob landed does not mean the image was rebuilt. Ask the image:

    git grep -c '<token>' <image-commit> -- '*.py'

Measured on `368ce460` (registry, built 00:21:03Z, **35 commits AHEAD of the live
`5690f313`** — a real upgrade, unlike `575f5572` which was its parent):

    _emit             present   dev-1's locator ungating IS in
    base_offset_ms    present   dev-2's field IS in
    PREDICTED_REFUSAL ABSENT    dev-2's newest gate work is NOT in
    agreement=        ABSENT    likewise
    offset_last_ms    ABSENT    dev-1's rename is on a review branch, not dev-AI
    offset_ms=        present   the alias is still what emits

**So a single recreate can land one agent's work and not another's, and each of them
will be watching for their own token in the same artefacts.** Say which, per token,
rather than announcing "the new image".

### 10b. ASK THE BLOB, NOT THE COMMIT — ancestry gives a FALSE NEGATIVE here

Measured on `368ce460`, and it contradicted itself until the right question was asked:

    token `fill source itself short by`   PRESENT in the image
    its commit c43ab47                    NOT an ancestor of the image
    dev-2's commits 25a6d70/283dab7/c43ab47   NONE are ancestors

    the image's blobs:
      src/merge_video_repair.py    558dee7e53e7c8f8   dev-2 named 558dee7e53e7c8f8
      src/merge_video_chimeric.py  b34b3bae12143db3   dev-2 named b34b3bae12143db3
                                                      BYTE-IDENTICAL, BOTH

The Lead **promotes by blob** — squashing and fast-forwarding onto the hop — so an
author's commit hash does not survive promotion even though their content does.
`git merge-base --is-ancestor <their-commit> <image>` therefore answers **"was this
commit object promoted"**, which nobody asked, and reads as **"is my work in"**, which
is false.

    A COMMIT IS A CONTAINER FOR CONTENT. PROMOTION KEEPS THE CONTENT AND DISCARDS THE
    CONTAINER, SO ASK ABOUT THE CONTENT.

Use `git rev-parse <image>:<path>` against the blob the author names, or grep the image
for the author's token. **Two routes agreeing (my token grep, dev-2's blob list) is what
makes the answer trustworthy; either alone would have been a guess.**
