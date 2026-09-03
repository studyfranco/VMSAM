# VMSAM — what to know before starting again

**Written 2026-09-02, on the owner's instruction to stop the campaign and
restart on a new basis.** This is the Lead's synthesis of four agent handovers
totalling 2586 lines. Read this first; it tells you which of them to open and
why.

> **Archived from campaign 1.** Content unchanged; the handover paths in the table
> below were rewritten to their location here.
>
> Three documents this file cites are deliberately not archived: the campaign's own
> `AGENT.MD` (1191 lines of superseded rules), a generated diagnostics report. Where a claim turns on one of them, treat the
> claim as unsupported rather than going looking.

| Document | Lines | Open it for |
| :--- | ---: | :--- |
| `handovers/20260902-HANDOVER-agent1-c-cut.md` | 833 | the mission bar, the store-dating method, C_cut, the decidability count |
| `handovers/20260902-HANDOVER-agent2-d-trim.md` | 807 | the guards and their real thresholds, provenance recovery, the join traps |
| `handovers/20260902-HANDOVER-agent3-ci.md` | 388 | infrastructure, the sweep, the instrument-fault population, teardown order |
| `handovers/20260902-HANDOVER-agent4-forensics.md` | 558 | the stores, what can and cannot be dated, the evidence layer |
| `PROVENANCE_PROPOSAL.MD` | 264 | **the single design decision the restart turns on** |
| `PROVENANCE_DOWNGRADE_COSTING.MD` | 246 | what to do if that decision is *no* |

---

## 1. The one-sentence diagnosis

**The merge knows facts at merge time that nothing writes down, and every tool
downstream reconstructs them differently — so where the tools disagree there is
no tiebreak, because none of them observed the fact.**

Four agents reached this independently, from four directions, none of them
looking for it. It is the root cause of nearly every specific defect below.
`sync_verdict` reconstructs provenance from emission order; `verify_lang` from
track counts; `provenance_classify` from codec and correlation. **Three methods,
three answers, no authority.**

## 2. What was actually achieved

Say it plainly, because the rest of this document is about what went wrong.

- **315 error files classified and put through a first pass.** The corpus is
  mapped.
- **The container path works.** T4 proved it end to end on a file with a pinned
  prediction written before the run: `err_150` went `MERGED_UNVERIFIED` →
  `MERGED_IN_SYNC`, `fre in_sync` worst **3.3 ms**, inside the bar — and matched
  a second agent's independent reading exactly.
- **Real defects found and fixed**: the cut-point peak filter (7 files recovered,
  0 regressions), the chimeric dub gate, the tail-plateau drop, the `evidence_class`
  absence/emptiness collapse.
- **A test suite of ~2300 that catches things.** More importantly, a *habit* of
  firing guards deliberately and killing mutants, which caught defects the suite
  structurally could not.

## 3. The five findings a restart must inherit

### 3.1 The mission's own bar had never been applied to anything

`final_status` reaches the 15 ms comparison only in its **third** branch, and
that branch needs a Step 1 verdict of `in_sync` or `out_of_sync`. Every Step 1
verdict in the corpus said `unverifiable`. **So every one of 315 rows exited at
branch one or two, and `Resolved: 0` read as *nothing qualifies* when it meant
*nothing reached the check*.** Both are now proven to render (`test_65`).

**The general form is the lesson: an armed guard that has never fired is
indistinguishable from a working one.** Fire every guard deliberately, with
literals, plus a capacity check proving it can return the *other* answer.

### 3.2 Nothing dates a measurement to the code that produced it

**Zero of 110 lag figures can be attributed to the code that measured them.**
Every version stamp in the corpus dates the container's *merge*; every lag is
measured *host-side*, where nothing is stamped. 71 of 110 rows carry no
row-level version or timestamp at all.

Worse than none: `batch_state.build` exists and is `deployed_build()` — **the
container's SHA, not the host module's.** A stamp naming the wrong process looks
like provenance.

**And a fix does not reach a running process.** `7744aa3` fixed an unconfigured-
binary defect at 20:47 UTC; **24 records across 22 files written after that still
carry the pre-fix behaviour**, because a long-lived sweep holds the imported
module until it restarts. The container is gated on `git_commit == HEAD` for
exactly this reason. **Nothing gates the host side.**

### 3.3 A refusal that reaches the verdict slot becomes a finding about the file

**31 of 45 per-language `unverifiable` records were an instrument that could not
run** — `tools.software['ffmpeg'] is not configured` — not a measurement. Only
14 were real structural refusals.

The refusal itself was correct: declining to guess a PATH binary for a
measurement is the right behaviour. **The defect is that "I could not measure"
was written where "this file is unverifiable" is read.**

### 3.4 A check that answers a weaker question than the one it is trusted for

**Found five times in one morning, at five different layers**, each by a
different person, none of them looking for it:

| Where | The weaker question |
| :--- | :--- |
| An AST test | asserted `error_id` *appeared*; `error_id: None` appears |
| The Lead's gate | read the suite's *output* by eye, not its *exit status* |
| A `.get()` | returns `None` for absent **and** for null |
| `evidence_class` | `not per_language` is true for missing **and** empty — **241 rows** |
| The privacy checker | claimed "no media library references"; it checks extensions |

**Every one is the same shape: a reader with two outcomes describing a world
with three, and the missing outcome is always *the instrument did not run*.**

**The remedy is usually a smaller function, not a cleverer assertion.** Where a
property can only be checked by parsing source, the property has nowhere to
live — extract it so it becomes a return value.

**And only a mutant reliably detects it.** A test cannot report that it is
asking the wrong question; it can only pass.

### 3.5 Absence is not emptiness, and a field's absence is not a file's absence

Both directions cost real time here. **11 files were reported as artefacts gone;
all 11 had outputs on disk** — 185 `FAILED` rows have no `merged_file` *field*
and 44 have a file at the derived path. The same mistake was already on record
in `step2_only_class.md` with all 27, **in a document the Lead had read.**

**Corollary that is still open**: every artefact is present, and nothing records
what it *was* at judging time. **"The artefact still exists" and "the verdict
still describes it" are different claims and only the first is checkable.**

## 4. What to discard — do not inherit these

- **`step1_verdicts.json`.** A four-day-old snapshot of a ledger that moved: 36
  of its 40 rows changed outcome underneath it, and it covers 4 of the 50 rows
  currently eligible.
- **`provenance_classification.json` as a description of current artefacts.**
  Every record backing an `in_sync` verdict describes a file rewritten since.
  The store is append-only and **cannot refresh**.
- **Any count taken from `external_verdicts` as one population.** It provably
  contains two code eras; its mtime tells a reader nothing about any row.
- **The residue/grid method as an instrument.** It is calibrated and honest
  (bound 0.0698 per entry, minimum n=2) and its own author recommends against
  relying on it: it dates a code *era*, not a tool, and just under half the
  entries are unclassifiable in principle. **Keep it as evidence FOR the missing
  field, never as a substitute.**
- **`SAME_RECORDING_PEAK = 0.98` on codec-differing pairs.** Measured: same-source
  across a lossy transcode reads **0.869–0.963**; different-source over a shared
  music bed reads up to **0.987**. **The populations overlap and no threshold on
  this quantity orders them.**
- **Peak values between 0.91 and 0.98 as evidence of a shared source.** Two
  signals with **no sample in common**, sharing only an amplitude envelope, read
  **0.979**. Two dubs of one programme have exactly that structure.
- **`new_files[0] == basename(db_master)` as a master/candidate discriminator.**
  True for every record. It has now produced a confident wrong answer twice, in
  two different contexts, months apart in effect. It is on the permanent discard
  list because **it will come back.**
- **Back-filling any of the above.** *Back-filling manufactures the thing the
  field records; absent is a value.*

## 5. What to keep

- **`PROVENANCE_PROPOSAL.MD`.** Begin here.
- **The discipline of firing a guard deliberately** with literals plus a capacity
  check (invariant 141).
- **Mutation-checking a new test before trusting it.**
- **Pinning a prediction before the run.** T4 is the model: two questions stated
  in advance, so nobody had to take anyone's word for which answer arrived.
- **The structural-versus-numerical framing.** Step 1 refuses on *structure* (no
  surplus tracks ⇒ both compared tracks are master-derived); `verify_against_master`
  refuses on a *number*. They disagree because they are answering different
  questions. **This generalises further than anything else produced.**
- **Separating measured from inferred, in every paragraph.** This is the single
  habit that made the last day productive.
- **The error string as an accidental version stamp.** Comparing a *stored* error
  message against the text current source emits separated "the fix is wrong" from
  "the fix was never loaded", without touching the process.

## 6. The decision the restart turns on

`PROVENANCE_PROPOSAL.MD` reduces four asks to **three fields**:

1. **Which code wrote this row** — captured **at import time, not write time**.
   `git rev-parse HEAD` at write time reports the checkout the process is not
   running, which reproduces the exact defect the field exists to record.
2. **What the artefact was when it was judged** — size and mtime at minimum.
   `batch_verify` already computes both and **throws the mtime away**; this field
   is below zero in cost.
3. **Where each output track came from** — `source` (master | candidate), source
   stream index, applied delay, `copied | re-encoded`.

**Field 3 is on the protected closure and there is no site outside it.**
Provenance exists for the duration of one function call; after the first mux
everything reads a probe of the combined intermediate. `keep_best_audio` drops
tracks *between* the two muxes, so a record taken at the first mux describes the
intermediate and **must be resolved through the final `--track-order` — the join
is part of the field, not an afterthought.**

**Rule on field 3 first.** Fields 1 and 2 can land any time, because
verification is repeatable and cheap. **A merge is not.** Every merge that runs
without field 3 produces an output whose provenance can never be recovered.

**If the answer is no**, `PROVENANCE_DOWNGRADE_COSTING.MD` prices the
alternative: marking the three reconstructions as unverifiable where they are
used. **Do not apply that blanket** — where the store says master-derived it says
so 66 times and never under an `in_sync` verdict, and that absence is informative.

## 7. Open questions nobody answered

- **`(fre, shared_reference)`** — five language tracks patched from the master's
  `fre` while the master carries only `fre` and `jpn`. Cross-language by
  pigeonhole. **Unknown whether the tag names the content source or the timing
  reference.** If content, five tracks carry ~51.5 s of French against criterion
  4's VO rule. **One decode settles it. Highest-value open question.**
- **Whether the 20 codec-differing rows are already decided at stage one.**
  `pcm_digest` hashes decoded PCM at 8 kHz mono, **not container bytes**, so the
  stated reason for skipping them is about a comparison the code does not make.
  The classifier's own calibration records `master_derived_shifted n=27` with
  **25 reading exactly 1.0000**, which supports it. **One decode. Cheapest
  high-value check available.**
- **`err_112`** — three windows at peaks 0.999/0.999/1.000, so one window
  satisfies the self-comparison guard and two do not. Needs a decoded-sample
  comparison.
- **The 11 "artefact gone" files.** Two agents measured opposite answers, almost
  certainly by probing different trees. **Deliberately left unreconciled rather
  than resolved in prose.** Treat as unverified in both directions.
- **`err_06f8e07512fd9caa`** — stamped `MERGED_IN_SYNC` on one language 3.1 ms
  inside the bar, while another language of the same file is a confirmed
  master-against-itself. Nobody acted on it.
- **The 15 ms boundary itself.** The code passes a row at exactly 15.0
  (`abs(v) > 15` fails); `AGENT.MD` says "under 15 ms", which is strict.
  **Fourteen rows turn on `<` versus `<=`.** Nobody ruled on it.

## 8. Process failures worth not repeating

- **The Lead amplified an agent's "unsupported" into "false"** and dispatched
  thirteen files on the stronger claim. The agent retracted; the premise was
  wrong. **Do not restate a finding more strongly than its author did.**
- **The Lead pushed a commit with a failing test**, having gated on the privacy
  check and read the suite by eye. **A pipeline's exit status is the last
  command's.** Use `gate.sh`; never an inline `check | tail && commit`.
- **The Lead routed around a refusal** — told an agent that two peers had
  declined a request, then asked it. It declined and named it correctly.
  **A dispatch is not permission, and a peer cannot relay the owner's authority.**
- **Teardown order matters.** Killing the queue first **orphans** the sweep
  rather than stopping it; it reparents to init and looks dead. Correct order:
  watchdog, `full_test.sh`, SIGTERM the python so it completes its `batch_state`
  write, then the queue.
- **`git stash` on a conflicted tree clears `MERGE_HEAD`**, producing a
  one-parent commit titled as a merge, exit 0. Caught by reading the parent
  list, not the exit status.
- **Before believing an absence, print one key from each side.** Three stores use
  three key spaces — absolute path, error id as string, and a list of rows. A
  lookup in the wrong space returns a clean, confident, entirely wrong absence
  that looks exactly like a finding. **This misled the project twice.**

## 9. Housekeeping the restart inherits

- **2575 show identifiers (`tvdb-<id>` and kin) across 438 tracked files.** A show
  id resolves to a title in one lookup. The privacy checker now **reports** these
  and does not refuse — failing on them would block every landing. **This needs
  an owner decision.**
- **The privacy checker's guarantee is narrower than it sounded.** Read every
  past "clean" as *no library path and no filename-with-an-extension*. A bare
  title in prose was never checked and cannot be.
- **`retained_first_pass`** accumulates ~3.9 GB intermediates, capped at 10 with a
  30 GB free-space floor. Those bounds are the terms of a narrow approval.
- **A clean sweep at this size is the expected outcome, not evidence of
  improvement.** At the sizes actually reached, a wholly clean run happens
  **82–86%** of the time with the failure rate unchanged.

## 10. On scene detection

The image already provides `scdet`, `select='gt(scene,0.4)'`, `silencedetect`
and `ffprobe -show_frames`. **Nothing new is needed to find cuts within one
file.**

**But that is not the project's problem.** Finding where two *encodes* diverge
is a correspondence problem, and scene scores do not solve it: the same cut
scores differently at different bitrates, so two cut lists do not align. What
would actually help is **perceptual hashing** (`opencv-python` or `imagehash`) —
hash one frame per second of each file and match the sequences. It survives
re-encoding, resolution changes and PAL speed differences, and it answers
*which part of A is which part of B*, which is the question.

`NEED_TOOLS.MD` asks for `dtw-python` and `librosa` for the same problem from
the audio side. **Pick one modality and do it properly rather than three
partially.**
