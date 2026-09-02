# Handover — Agent 4, forensics / evidence layer / stores / provenance

**Written 2026-09-02 at `b4a2923` on `agent/forensics`, on the owner's stop
order.** Everything here is reading and writing only; no gate was run for this
document, no decode was spent, the container was not touched.

**Three rules held throughout: files are named `err_<id>` and never by path;
every paragraph says whether it is MEASURED or INFERRED; the false starts are
left in, because in this territory they are the content.**

---

## 1. WHAT IS TRUE

### 1.1 The artefact population is not missing — MEASURED

    verdict rows examined                                       106
      artefact present at a RECORDED path                        83
      artefact present at the path the project DERIVES           23
      unlocatable                                                 0
    rows in any store carrying a recorded `merged_file` that is missing   0
    ledger rows FAILED with genuinely no output anywhere         141
      of those, carrying a verdict in any store                    0

**Confidence: high.** Three independent routes (recorded path, derived path,
ledger cross-check) and they agree. **What would overturn it:** a store I did
not read that carries verdicts keyed differently — I read `external_verdicts`,
`no_output_verdicts`, `step1_verdicts`, `batch_state` and the ledger. A fourth
verdict store keyed on something else would not have been seen.

**The claim this replaced was mine to check and not mine to originate.** The
Lead had put *"a verdict outliving the thing it was about"* to the owner on the
strength of 11 files reported unmeasurable and 7 ids shared between two
populations. **All seven — err_9, err_12, err_46, err_51, err_59, err_63,
err_70 — have an output on disk at the derived path.** 185 `FAILED` rows have no
`merged_file` **field**; 44 of them have an output. **It was a field's absence
read as a file's absence.**

### 1.2 Which store dates what — MEASURED

    stores surveyed                                              48
      that can date any row                                       15
      that cannot date any row                                    33

    batch_state.recorded_utc          250 of 315   a true row date
    band_eleven_results.measured_utc   11 of 11    a true row date
    output_provenance.output_mtime_utc  80         dates the ARTEFACT, not the row
    correction_state.started                       dates the CORRECTION, not the row

**Two of the four apparently-dated stores date the SUBJECT, not the row.** A
reader taking `output_mtime_utc` as *when this was judged* is reading a fact
about the file as a fact about the verdict.

**Confidence: high, with one method caveat that is itself a finding — see §3.3.**
Date fields were detected by the **shape of the value**, not the name of the key.
**What would overturn it:** a store dating rows in a format my shape test does
not recognise (epoch integers were covered; a bespoke encoding would not be).

### 1.3 `written_utc` prevents nothing — it lets a reader notice — MEASURED

This is about my own landed change, and it is the answer I was least keen to
write.

    the 27 stale verdicts        YES, all 27   every ledger row carries recorded_utc
    step1_verdicts               PARTLY, 25 of 40   15 ledger rows have no
                                 recorded_utc on the far side to compare against
    the "artefact gone" 11       NO — nothing about them was a dating problem

**A date does not prevent a stale verdict. It lets a reader see that one might be
stale.** That is worth having and it is a different claim. **A partial remedy,
recorded as a partial remedy.** The third case needs artefact identity, which is
equally prospective.

### 1.4 The `error_id` parity work — MEASURED, and landed

`reverify_external.py` and `verify_no_output.py` were the two verdict-store
writers and they could not be joined: one keyed by path, the other by error id.
Both now carry identical helpers — `artefact_identity()`, `load_error_index()`,
`error_lookup()`, `error_fields()` — and `error_id` **is omitted, not nulled**,
when there is no id. Both writers are host-side only; neither is referenced by
the Dockerfile, `run.sh`, `init.sh` or any `main*.py`. **Verified before
editing**, because a host-side tool and a container-side tool are not the same
risk.

`artefact_identity()` is three-valued (`recorded` / `absent` with a reason) and
records path, size and mtime **at the moment of judging**.

**Confidence: high — it is code with tests, not an inference.** The `error_id`
omission property is tested at runtime, and the reason it is testable at all is
§3.1.

### 1.5 The delay-ambiguity population — MEASURED

The exception text `Multiple delay found ... in test 1` **names the wrong
quantity.**

    rows characterised                                          170
    of 115 with an extractable measurement, the set was EMPTY   106
    genuinely multiple                                            9

    shapes: agreed / bimodal / drift / disordered / few_valued
    the bimodal group                                            53

**Confidence: high on the count, and the 53 bimodal are CONSISTENT WITH but NOT
DEMONSTRATED to be a mid-file structural difference** — that needs a decode at
the changeover cut and was never spent. **Recorded as consistent, not
demonstrated, and it must stay that way in any restart.**

**A short count that must carry its caveat:** 55 delay rows were unreadable at
the time of measurement and heal as the sweep reprocesses them. **Any count taken
from that exception text before they heal is short by 55 and must say so.**

### 1.6 The fps census and the NTSC family — MEASURED

    file pairs probed                                           173
    master and candidate at IDENTICAL rates                     148
    distinct rate relations present                              25
    relations the production detector can see                     10

`detect_pal_relation` matches PAL-shaped ratios only, so **no table in the
project can count a non-PAL relation.** Confirmed with two independent
instruments (a ratio screen and a slope fit).

**A correction that belongs here, because I got the standing of this wrong.** I
reported eight files as a new NTSC 1000/1001 finding. **TASK.MD Case 3 had
already recorded six of those exact error ids at that exact rate a day
earlier.** Only two members and the two-instrument confirmation were new. **Owner
rulings live in `src/TASK.MD` and measurements live in `FINDINGS.MD`, and the two
do not cross-reference each other** — work driven by the stores does not see a
ruling recorded in TASK.MD. Grep TASK.MD before calling anything new.

### 1.7 The costing of the reconstructions — MEASURED, landed at `b4a2923`

If a merge-time provenance field is refused, marking the downstream
reconstructions unverifiable is **three different decisions**:

    downgrade the ORDER-based claims                    2 cells
    downgrade the one claim the evidence contradicts    1 cell
    downgrade every verdict resting on a reconstruction 109 cells

    footing of every verdict record in the corpus
      in_sync       BYTE-identity copy guard            121
      in_sync       footing NOT RECORDED                 61
      in_sync       ORDER                                 1
      unverifiable  footing NOT RECORDED                 66
      unverifiable  COUNT                                51
      unverifiable  BYTE-identity copy guard             37

**The order-based premise every document has been written about — mine included —
carries two records in the entire corpus: err_78 and err_293.** The price sits on
the byte-identity copy guard, which carries 121 `in_sync` records.

**The soundness is one-directional in every case, and only the two directions
that ASSERT can produce a false pass:**

| Reconstruction | Sound | Unsound | What the unsound direction does |
| :--- | :--- | :--- | :--- |
| COUNT `out > n_master` | existence, by pigeonhole | — | permits, correctly |
| COUNT `out <= n_master` | — | "candidate contributed none" | **withholds** |
| ORDER `streams[n_master]` | — | "this stream is the candidate's" | **asserts** |
| BYTE bit-identical | "this is the master's copy" | — | withholds, correctly |
| BYTE differs | — | "not the master's copy" | **asserts** |

**Existence is sound; identification is not.** The pigeonhole proves a
candidate-derived track is in there somewhere and says nothing about which stream
it is.

**err_152 is the one demonstrated failure, and it is worth more than the
50-file figure I nearly reported:** its `eng` is master-derived by construction
(the candidate carries no `eng`), its single `jpn` track correlates at peak
**1.0000** against the master, and `batch_state` records `verdict in_sync,
compared "output vs master file", max_abs_lag_ms 0.0`. **There is no
candidate-derived track anywhere in that output and the verdict is a 0.0 ms
measurement of the master against itself.** It renders in the Master Summary
Table as a Step 2 pass.

**A screen offered as a screen, INFERRED not established:** 29 of the 96
`in_sync` files read exactly 0.0 ms. A well-aligned merge also reads 0.0, so this
names a population to look at, not a population that is wrong.

---

## 2. WHAT IS WRONG AND STILL WRONG

### 2.1 `src/diagnostics/tools/provenance_classify.py` — two defects, PARTLY fixed today and NOT VERIFIED

**Found by Agent 2 while answering a different question; they flagged rather than
patched, which was right.**

**Defect A — bare binary literals, against the owner's standing rule.**
`streams()`, `pcm()` and `pcm_digest()` invoked bare `ffprobe` / `ffmpeg`
instead of `tools.software[...]`. **Every measurement this module ever took used
whichever binary was first on PATH, which cannot now be shown to be the one the
merges being graded had used.** The probe matters as much as the decode: a
different `ffprobe` reporting stream order or language tags differently changes
**which** tracks are compared.

**Defect B — `resolve()` did not check the candidate.** `classify()` then read an
unreadable candidate as an **empty stream list**, so every language fell through
the structural gate and returned `master_derived_by_construction` with
`judgeable: false`. **A missing file rendering as a confident verdict about
provenance.** Invariant 157 inside my own module. **Latent, not active** — no file
in the current set is affected, and `--calibrate` happened to check the candidate
before calling while `--all` did not. **Latent is a property of today's inputs,
not of the code.**

**Defect C — the untyped absence.** `resolve()` returned a bare `None` for four
different reasons (no classified row, no `merged_file`, no `db_master`, either
file absent) and `main` did `if not paths: continue`, writing nothing. **A file
that was never eligible and a file the tool could not resolve are the same
non-entry.** Same shape as `no_record` beside `index_unavailable`.

**WHAT I ACTUALLY DID TODAY, PRECISELY — AND THEN REVERTED.** I wrote an edit to
that file and **reverted it before handing over**, deliberately. **The file on the
branch is unchanged.** The edit did the following, described here in enough detail
to redo in one sitting:

* adds `_load_software()` and `_binary()` in the shape of
  `batch_verify._binary` — **including the refusal to fall back to PATH, and the
  config load that makes the refusal safe. Both halves, because the refusal
  alone is what put 31 instrument faults into the verdict slot;**
* routes the three call sites through `_binary("ffprobe")` / `_binary("ffmpeg")`;
* makes `classify()` **refuse** an unreadable candidate, returning
  `state: "candidate_unavailable"` and **`judgeable: None`, never `False`** —
  "I could not look" and "I looked and there is nothing" must not share a value;
* makes `resolve()` return `(paths, "resolved")` or `(None, <one of six typed
  reasons>)`, and **check the candidate**;
* adds `state: "classified"` to the normal return.

**IT WAS NOT FINISHED, WHICH IS WHY IT IS NOT ON THE BRANCH.** `main()` still
called `resolve()` expecting the old single return value, so **the edited module
was broken at `--all` and `--calibrate`** — a landmine to hand to a restart. And
**the firing test does not exist** —
the Lead asked for the missing-candidate cell to be fired deliberately with
literals **plus a capacity check** (a gate that can only ever say one thing would
explain the absence without any candidate being missing), and that test was never
written. **An armed guard that has never fired is indistinguishable from a
working one.** **Both defects are therefore STILL OPEN on the branch.** A restart
should redo this properly: the two helpers, the three call sites, the two
`resolve()` callers, and one test file that fires the missing-candidate cell with
literals and includes the capacity check.

### 2.2 The classification store is append-only and cannot self-heal — MEASURED

`--all` does `if str(eid) in results: continue`, so a run today would **add 47
rows and refresh none of the 82.** Meanwhile **25 of the classified outputs have
been rebuilt since the store was written** (2026-08-30T22:12Z). **Every record
backing an `in_sync` verdict may describe an artefact that no longer exists at
that path.**

**Making it refreshing is the owner's decision, not a repair** — re-running a
classification over rows that already have verdicts changes what the corpus says.
**What it would take:** a `--refresh` that rewrites a row only when the artefact's
identity differs from the one recorded — which requires the artefact identity
field that does not exist. **The store cannot currently tell whether it needs
refreshing.**

### 2.3 The reconstructions themselves — still wrong, still in use

`sync_verdict`'s ORDER branch, its COUNT note, and the byte-identity copy guard.
**All three are in production and all three are unsound in one direction.** The
costing at `b4a2923` says what changing them costs. **Nothing was changed.**

### 2.4 A guard written for a case that has never seen it — MEASURED

`master_summary.py:65` maps the phrase *"tracks are master-derived"* to a table
cell — **and it fires on zero rows**, because the claim lives in a nested
`per_language` record and in a store the detail column does not read. The
mapping is correct and unreachable.

---

## 3. WHAT WE MISUNDERSTOOD, AND FOR HOW LONG

### 3.1 The surviving mutant, and why the remedy was a smaller function

**This is the centrepiece and it is about my own test.**

A mutation run against the artefact-identity work left one mutant alive: the
writer emitting **`"error_id": None`** instead of omitting the key. The test that
should have caught it was an AST check asserting that `error_id` **appeared** in
the write site's dict literal. **It appears in both versions.** The assertion was
satisfied by the defect.

**What was believed:** that an AST assertion over the source could hold a
property the runtime could not reach. **What was true:** the property — *the key
is omitted, not nulled* — is a property of the **value produced**, and it was
only unreachable because the code that produced it was inlined at the write site
with nothing to call.

**How long:** hours, and it survived a deliberate mutation run before it fell.

**The remedy was not a better assertion. It was a smaller function.** Extracting
`error_fields(path, index)` made the property a return value, and a two-line
runtime test then killed the mutant outright. **A property that can only be
checked by parsing source is usually a property that has nowhere to live.**
Recorded as **invariant 162: a check can answer a weaker question than the one it
is trusted for — then the design is right and the test is wrong.**

**What would have caught it sooner:** asking, before writing the AST check, *what
would I call to observe this?* If the answer is "nothing", that is the finding.

### 3.2 `git stash` on a conflicted tree — a merge commit with one parent

Mid-merge, with conflicts in the tree, I ran `git stash`. **It cleared
`MERGE_HEAD`.** The subsequent commit `123bdd5` was titled as a merge, exited 0,
and **had exactly one parent** — dev-AI was dropped entirely while the operation
reported success.

**What was believed:** that a clean exit status from a commit meant the merge had
been recorded. **What was true:** the merge state had been silently discarded and
the commit was an ordinary one wearing a merge's message. **How long:** minutes,
because I read the parent list; **it would have been permanent if I had trusted
the exit code.** Recovery was `reset` to `181b14b`, redo the merge, re-apply
three edits from saved text.

**What would have caught it sooner — and this is the rule:** **verify a merge by
reading `git log --format=%P` on the result, never by the exit status of the
command that made it.** Exit status reports whether a command ran, not whether it
did what its name says.

### 3.3 My own name test reported `batch_state` as dated by its outcome

The first datability survey detected date fields by **key name**. `outcome`
contains the substring `utc`. **`batch_state` was duly reported as dated by its
outcome field.**

**What was believed:** that a key-name heuristic was good enough for a survey.
**What was true:** it produced a confident wrong answer on the largest store in
the corpus. **How long:** one pass — caught before reporting, because the result
looked too convenient. Redone by detecting **timestamp-shaped values**.

**What would have caught it sooner:** the general rule that **a name is a claim
and a value is evidence**, which this project has now hit from at least four
directions (a file's name asserting currency; a `build` field naming the wrong
half; `CHIMERIC` tags absent from files that predate the tag).

### 3.4 The Lead's framing, which I corrected, and which was already on record

The Lead wrote *"a verdict outliving the thing it was about"* and put it to the
owner. **I measured it and every artefact was there.** A field's absence read as a
file's absence.

**And the same mistake was already on record in `step2_only_class.md`, where all
27 turned out to have published outputs — in a document the Lead had read.**

**How long:** it reached the owner before it was measured. **What would have
caught it sooner:** the measurement itself is cheap — `os.stat` on a derived path
— and it was never run because the reading felt conclusive. **The agent who
reported the 11 said explicitly it could not measure in either direction and was
right to; the error entered when that honest non-measurement was carried forward
as a finding.**

### 3.5 Reading `judgeable: false` as a negative — mine, today, in the costing

The cross-check against `provenance_classify` first reported **50 in_sync
verdicts contradicted by content**. That was wrong.

**err_95 is the file that caught it.** Its `eng` has one master track against two
in the output; one is `master_derived_shifted` and **one is `indeterminate`**. The
classifier did not refute the pigeonhole — **it could not decide.**
`judgeable: false` folds *"there is no candidate side"* together with *"I could
not tell"*.

**Checking the leaves instead of the flag:**

    candidate side CONFIRMED      5      (err_83, err_87, err_98, err_157, err_173)
    NO candidate side, refuted    1      (err_152)
    COULD NOT TELL               48
    never classified             38

**5 confirmed, 1 refuted, 89 undetermined.** **This is invariant 157 committed by
the auditor, in the same document that cites invariant 157.** How long: one pass.
What would have caught it sooner: **check the leaves, not the entry point** —
which is the amendment I wrote to 157 myself.

### 3.6 Four smaller ones, kept because they are the same shape

* **A brace-balance extractor** ran to end-of-string on filenames containing
  `{edition-…}`, sinking 59 rows into a bucket labelled "no brace" that was
  itself wrong. Fixed by bounding on `} for `.
* **A silent 220-character truncation** that I looked for by grepping for the
  word "TRUNCAT" and did not find. **Looking at the data directly found it
  immediately** — a legacy `[:220]`, splitting perfectly temporally.
* **A false coverage gap**: `different_recording.json` showed 0 of 111 overlap. It
  was keyed by error id, not by path. **111 of 111.** Ruled out the key-format
  mismatch before concluding, which is the only reason it did not become a
  finding.
* **Two ANCHOR FAILED results** in a mutation run that I first read as surviving
  mutants. **My anchor text was wrong, not the code.** A mutation that does not
  apply looks exactly like one that survives; the harness must assert the anchor
  matched **exactly once** AND that the file changed.

---

## 4. UNKNOWN / UNRESOLVED

**The central one, and it is the thing to carry forward:**

> **Every artefact is present, and nothing in any store records what that
> artefact WAS at judging time — no path, size, mtime or hash. "The artefact
> still exists" and "the verdict still describes it" are different claims, and
> only the first is checkable.**

**The 27 `in_sync`-beside-`FAILED` rows are both true, of different runs.** That
dissolves the apparent contradiction and replaces it with an unanswerable
question: **which run the surviving file came from.** No store can say.

Also open:

* **The 22-file Step 1 backlog. Still held, never started** — it needs decode and
  C_cut owns the box.
* **The 53 bimodal delay rows** — consistent with a mid-file structural
  difference, **not demonstrated**. Needs a decode at the changeover cut.
* **89 of 96 `in_sync` verdicts are undetermined on provenance**, and the
  instrument that would settle them (`provenance_classify` re-run over current
  artefacts) exists and was not run.
* **Agent 2 counted the master-derived claim 66 times; I count 51** (12 in
  `batch_state.per_language`, 39 in `step1_verdicts`). **The direction is
  identical and independently reached; the magnitude differs by 15 and I cannot
  account for it.** Two counts of one phenomenon are reconciled by naming the
  file that differs, and that was not done.
* **Not explained:** why 29 `in_sync` verdicts read exactly 0.0 ms. A screen, not
  a finding.
* **Not explained:** why `master_summary`'s master-derived phrase was written at
  all, given the claim it maps has never been stored where the mapping can see
  it.

---

## 5. WHAT TO DISCARD

**Blunt, as asked.**

* **Do not inherit any count taken from the `Multiple delay found` exception text
  before the 55 unreadable rows healed.** It is short by 55 and the shortfall is
  invisible in the number.
* **Do not inherit `provenance_classification.json` as a description of current
  artefacts.** It is a snapshot of 2026-08-30T22:12Z, it is append-only, 25 of its
  subjects have been rebuilt since, and **it carries no per-row artefact
  identity, so it cannot say which of its rows are stale.**
* **Do not inherit any derived artefact whose only currency claim is its name.**
  `limit_vs_end.json` cost a full reconstruction by being a superseded join with
  a current-looking name and no marker.
* **Do not inherit the 41 of 64 `per_language` results that "cannot be assessed
  at all".** They are undistinguished, which is more recoverable than wrong, and
  they must not be promoted on a restart because they look like data.

**AND, VERBATIM, THE RULE THAT MATTERS MOST HERE:**

> **Back-filling manufactures the thing the field records; absent is a value.**

Specifically: **do not date the 106 verdict rows from an mtime, do not date the
110 lag figures from a container build, and do not write a timestamp onto any
historical row.** A field that is honestly empty supports *"this cannot be
dated"*; a field filled from a proxy supports a false conclusion, and **nothing
downstream can tell the two apart.** A gap in a provenance record is evidence
about the record's history and is often the only such evidence left.

**Re-running is not back-filling.** A re-verification produces a **new
observation** with a real stamp; it is not a repair of the old row and must not be
recorded as one.

---

## 6. WHAT TO KEEP

### 6.1 `src/PROVENANCE_PROPOSAL.MD` — start here

**The four asks from four agents are three fields, and only one needs the
owner.**

1. **Writer stamp** — `{utc, rev, dirty, by}` on every row a host-side tool
   writes. Agent 1's measured gap, Agent 3's design and my `written_utc` are
   **one write at one moment**; adopting them separately is three writes where
   one belongs.
2. **Artefact identity** — path, size, mtime of the thing judged, recorded at
   judging time. **The same write as (1) but a different subject, so it keeps its
   own name.** Folding it in would repeat the error it exists to catch.
3. **Track provenance** — which source contributed each output track, **recorded
   by the assembler that knows.**

**`rev` must be captured at IMPORT time, not write time.** A long-running sweep
holds the module it imported, so `git rev-parse HEAD` at write time reports the
checkout the process is **not** running — **reproducing exactly the defect the
field exists to record.** `provenance.py::_commit()` does it the wrong way today.
`dirty` is there because host-side sweeps run from worktrees with uncommitted
edits.

**Field 3 is on the protected closure and there is no way around it.** Provenance
exists for the length of one function call: `generate_new_file`,
`generate_merge_command_common_md5` and `generate_merge_command_other_part` each
know the source file they contribute; the moment the first mux runs, everything
downstream reads a probe of the **combined** intermediate — which is why
`keep_best_audio` cannot see it either. **All five candidate sites are among the
78 closure members.** The technique is already proven on that path
(`keep_best_audio` reads `CHIMERIC` / `VMSAM_RESAMPLED` tags today and
`native_wins` depends on them); **what is missing is the site, not the
technique.**

**Two implementation facts found by reading, not assumed.** `ffmpeg_cmd_dict` is
already threaded through all three contributors, so the record costs **one key
and no change to any command string** — but **`keep_best_audio` drops tracks
between the two muxes**, so a record taken at the first mux describes the
intermediate and must be resolved through the final `--track-order`. **The join
is part of the field, not an afterthought.** And field 2 is below zero in one
place: `batch_verify` already snapshots `(size, mtime)` for every output file and
**discards the mtime**.

**RULE ON FIELD 3 FIRST — and the reason is timing, not importance.** Fields 1
and 2 can land at any moment and a re-verification then produces a stamped row,
because **verification is repeatable and cheap. A merge is neither.** Every merge
that runs without field 3 produces an output whose provenance can never be
recovered. **The cost of waiting is asymmetric and it accrues while the sweep
runs.**

**And none of the three repairs a written row.** All prospective, per store, as
recorded in §5.

### 6.2 Invariant 162

**A check can answer a weaker question than the one it is trusted for — then the
design is right and the test is wrong.** The concrete pair is `"error_id": 152`
against `"error_id": None`, both satisfying an assertion that the key appeared.
**The fix was a smaller function, not a cleverer assertion.**

### 6.3 The firing discipline

**An armed guard that has never fired is indistinguishable from a working one;
the remedy is to fire it deliberately** — with literals, in a test named for the
cell, **plus a capacity check**, because a gate that can only ever say one thing
explains an absence without the absence being real. `no_record` versus
`index_unavailable` is the worked example. **`FireTheCellThatNeverFires` in
`test_65_error_parity.py` is the pattern to copy.**

Also keep: **invariant 157 and its amendment — a three-valued tool is not
three-valued if anything under it is; check the leaves, not the entry point.**
§3.5 is what happens when the author of the amendment forgets it.

### 6.4 The two costed documents

`src/PROVENANCE_DOWNGRADE_COSTING.MD` (`b4a2923`) — what refusing field 3 costs,
in cells the owner would actually see: **2, or 1, or 109**, and they are three
different decisions. **The count-based claim everyone discussed changes 0 cells.**

---

## Branch state at handover

`agent/forensics` at **`b4a2923`**, merged up to `dev-AI` (`603aa8d`), **nothing
pushed, working tree clean apart from this report.**

**`src/diagnostics/tools/provenance_classify.py` is UNCHANGED and both its
defects are still open.** I wrote a fix today and reverted it rather than hand a
restart a module that was broken at both entry points with no firing test — §2.1
describes the change fully enough to redo. **That is a deliberate choice to lose
work rather than land it unverified, and it should be read as such and not as an
oversight.**
