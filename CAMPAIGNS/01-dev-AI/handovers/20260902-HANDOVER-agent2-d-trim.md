# HANDOVER — Agent 2 (d-trim): D_trim, B_fps, E_corr, A_no_lang, the guards, provenance

Written 2026-09-02 on the owner's stop-and-restart directive. Work stopped where
it stood; the C_cut watcher is disarmed. Nothing here is tidied — the false
starts are the content.

**Reading convention.** Every paragraph separates MEASURED (a number I produced
and can point at) from INFERRED (a reading of code or a model). Files are cited
as `err_<id>` only. Where a claim is a model rather than a corpus fact it says so
in the same sentence, not in a footnote.

---

# 1. WHAT IS TRUE

## 1.1 The copy guard's real thresholds are 0.05 ms and 0.9985

`batch_verify.verify_against_master` refuses a self-comparison on

    if all(abs(l) < 1e-9 and p >= 0.999 for _t, l, p in lags):

Four lines above, the samples are stored rounded:

    lags.append((t, round(l, 1), round(p, 3)))

and the guard reads those. MEASURED, by calling the function from outside with
literals: `abs(l) < 1e-9` applied to a value rounded to one decimal is
`abs(l) < 0.05`; `p >= 0.999` applied to a value rounded to three decimals is
`p >= 0.9985`. Boundary probes: lag 0.0499 refuses, 0.05 passes; peak 0.9985
refuses, 0.9984 passes.

**Confidence: certain.** It is pinned by `test_66_the_copy_guard.py`, which
calls the real function and asserts both boundaries, and asserts the two `round`
calls off the AST so the rounding cannot be moved silently.

**What would overturn it:** nothing about the reading. Removing or changing
either `round` changes the thresholds, and the test fails loudly if it happens.

**Consequence, MEASURED on literal arrays through the module's own arithmetic:**
a bit-identical pair reads exactly 0.0 at peak 1.0; a same-source pair
requantised to 8 bits reads |lag| 0.0056 at peak 0.999525, which still refuses.
So the guard catches a copy **and a gentle same-source re-encode**. I had
reported the opposite (see §3.2).

## 1.2 The guard has fired, twice, and both firings were correct refusals

MEASURED from `batch_state.json`: `err_fc149a010e30747b` (jpn) and
`err_06f8e07512fd9caa` (jpn), each with samples `[0.0, 1.0]` in all three
windows and the note "the only '<lang>' track is bit-identical to the master's".

This answers the invariant 141 question from the ledger rather than only from a
test: the refusal is producible, so its absence elsewhere carries information.

**Also MEASURED, and it is a defect nobody has acted on:**
`err_06f8e07512fd9caa` had its jpn refused as master-against-itself and the file
is stamped `MERGED_IN_SYNC` anyway, on its `fre` reading **46.9 ms against a
50 ms bar**. A file-level pass resting on one language 3.1 ms inside the bar,
with another language known unverifiable.

## 1.3 A peak of 0.91–0.98 is not evidence of a shared source

MEASURED. Two signals with **no sample in common**, sharing only a slow
amplitude envelope, read peak **0.979** through `batch_verify`'s arithmetic
(10 ms envelope, z-scored, sub-bin refined). Two dubs of one programme share
exactly that structure: one music-and-effects bed, one loudness contour,
different dialogue.

**Confidence: high, and it is a model, not a corpus measurement.** The signals
are synthetic. What makes it trustworthy is that the test was written asserting
the opposite (`assertLess(peak, 0.5)`) and failed at 0.979; I did not go looking
for this number.

**What would overturn it:** a decode of two real dubs showing envelope
correlation well below 0.9 in this corpus. That decode has not been done.

**Consequence:** the 15 stored `in_sync` rows reading 0.0 ms in every window at
peaks 0.912–0.996 are **not** identifiable as self-comparisons from their peaks.
I was tempted to read them as such and did not.

## 1.4 `SAME_RECORDING_PEAK = 0.98` was calibrated only on matching codecs, and the band does not survive being widened

The classifier's own docstring records the band:

    master_derived_shifted   n=27   0.9833 .. 1.0000
    candidate_derived        n=18   0.2588 .. 0.9718

empty from 0.9718 to 0.9833, threshold inside it. MEASURED (by me, from the
corpus): all 45 of those tracks reached stage two, which is only reachable when
the codecs match. **The limitation was never recorded with the band.**

MEASURED, on literal arrays through `provenance_classify.same_recording`'s exact
arithmetic (1 ms envelope hop, mean-removed, normalised dot, max over ±2000 ms):

    SAME SOURCE, lossy transcode        14 bit/3.8 kHz  0.9627
                                        12 bit/3.5 kHz  0.9401
                                        10 bit/3.0 kHz  0.9078
                                         8 bit/3.0 kHz  0.9075
                                         7 bit/2.5 kHz  0.8690
                                        range 0.8690 .. 0.9627   0 of 8 above 0.98

    DIFFERENT SOURCE, shared music bed  bed x1.0        0.4732
                                        bed x2.0        0.8637
                                        bed x3.0        0.9465
                                        bed x4.0        0.9717
                                        bed x6.0        0.9868
                                        range 0.1386 .. 0.9868   ceiling ABOVE 0.98

    SAME SOURCE, lossless codec change (flac <- pcm/truehd)      1.0000

**The two populations overlap.** A same-source pair across a lossy codec change
reads *below* the threshold in every case I tried; a different-source pair over a
dominant shared bed reads *above* it. **No threshold on this quantity separates
them.**

**Confidence: high for the direction, moderate for the exact numbers.** The
transcode and bed models are synthetic and their parameters are mine. What is
robust is the crossing: the same-source ceiling (0.9627) is below the
different-source ceiling (0.9868), so no cut point orders the two populations
correctly.

**What would overturn it:** a decode-based measurement on real codec-differing
pairs in this corpus showing same-source readings clustering above 0.98. That is
the experiment that was never run.

**But see §1.5** — the population this band would be newly applied to may be
empty, which changes what the overlap costs.

## 1.5 The 20 codec-differing rows are a skipped branch, and the stated reason for the skip is false

MEASURED: of the 33 `in_sync` rows the provenance store reaches, 20 are
all-indeterminate and **every one gives the same reason** — `codec differs from
the master's track` — all with `out=1, mst=1`, across ten files and three
languages (`err_68a4b68d072cfc6e`, `err_42a4e8c4228cc200`,
`err_4a16427bf85774bd`, `err_3940aadc3d799acf`, `err_5b5f7376ef0787d8`,
`err_50424f23ebac25de`, `err_d71fbf6e40eefe99`, `err_fb8de8f9fe21453e`,
`err_40c13d1c66233741`, `err_14747cc8b8e7121f`, `err_e6ed0776bbfae2f6`).

The code:

    if track["codec"] != master_codec:
        verdicts.append({... "indeterminate", "why": "codec differs ..."})
        continue                      # <- never reaches stage two

**INFERRED from the code, and I consider it solid:** the docstring justifies this
as "a byte comparison across codecs proves nothing either way." That would be
true if stage one compared container bytes. **It does not.** `pcm_digest` runs
ffmpeg to `-ac 1 -ar 8000 -f s16le` and md5s the **decoded PCM**. A lossless
codec change leaves the decoded samples identical, so the digest still matches.
The justification is correct about a comparison the implementation does not
make. Invariant 159's shape: the principle is right and the implementation it
guards is a different thing.

**INFERRED, and this is the part a restart should check rather than trust:**
reading `mergeVideo.generate_new_file_audio_config_second_pass` against its base
command (`-c copy` at line ~3415), the only way a **master-derived** track
changes codec is a lossless non-flac source re-encoded to flac — which is
bit-transparent, so it reads 1.0 and the digest already matches. Every other
master-derived path either copies the stream or re-encodes to the same codec
name. If that holds, the "same source across a *lossy* codec change" population
is empty in this pipeline, and the overlap in §1.4 costs nothing **for these 20
rows** — they would be decided at stage one, not stage two.

**I could not verify this**, because verifying it needs a decode. It is the
single highest-value thing a restart could check cheaply.

## 1.6 Provenance recovery: 37 recoverable / 24 undeterminable / 4 lost

MEASURED, over all 64 `in_sync` rows, by comparing the output's mtime against
the verdict's `recorded_utc`:

    artefact predates its verdict, output+master+candidate all present   37   (28 files)
    NO recorded_utc -- cannot show the artefact matches the verdict      24   (16 files)
    artefact REBUILT after its verdict (+0.1 h)                           4    (2 files)

**The 24 are a third state, not a negative.** They are a missing timestamp, not
a missing file. Collapsing them into either bucket is the error this project
keeps paying for.

**Confidence: high.** mtime and the stored stamp are both direct reads.

**What would overturn it:** an mtime that does not track rebuilds — e.g. files
copied or moved with timestamps preserved. I did not check for that and it is a
real hole in the method.

## 1.7 Every provenance record backing an in_sync verdict is stale

MEASURED. `provenance_classification.json` was last written
**2026-08-30T22:12:28Z**. Of the 32 `in_sync` rows that have a record, **32** sit
on outputs whose mtime is later. The store is two days behind everything it
grades.

**Confidence: certain, subject to the same mtime caveat as §1.6.**

## 1.8 Coverage of the store is a scheduling fact, not a physical limit

MEASURED: the 32 `in_sync` rows with no record cover 22 files; **all 32 resolve
today** (output and master both on disk), **all 22 have their candidate on disk
too**, and **all 22 outputs were built after the store's last run**.

Across all 315 classified rows, what `resolve()` would return today: 129
resolvable, 185 with no `merged_file` in `batch_state`, 1 with no `db_master`.
82 are already in the store, and `--all` skips those, so a run today would add
47 and refresh none.

## 1.9 Step 1's filter scope is a four-day-old ledger snapshot

MEASURED (earlier in the session): the 40 entries in the Step 1 store are a
snapshot at store mtime `2026-08-29T22:33:48Z` against a `batch_state` at
`2026-09-02T07:28:06Z`. Coverage of the currently eligible set is **4 of 50**.
The gap is **46**, not 275 — 229 of the 275 are outcomes the filter never
targets. **39 of the 40 recorded verdicts are the `shared[0]` defect** (`fre` 34,
`eng` 5): `verify_unverified.py:38` builds `shared = [l for l in ("fre","eng","jpn") if ...]`
and then uses `shared[0]`, so the language is chosen by list order, not by
evidence.

**Not distinguishable from the data:** "not attempted since" versus "attempted
and crashed before the write". I reported those undistinguished and they still
are.

## 1.10 The err_211 truncation is silent and non-deterministic

MEASURED. Complete stderr for the stopping pass is 18 lines with **zero**
diagnostic vocabulary. Three runs on byte-identical inputs produced three
different endpoint sets; all six truncated tracks vary and `jpn` never does — it
lands exactly on `-t 1441.942`. Magnitudes 1000.494 / 1000.487 / 1000.456 are
**three samples of a moving quantity, not one measurement**.

**INFERRED:** the mechanism is a class (a race or resource condition), not a
candidate. Nothing in the corpus contradicts this and nothing confirms it.

Screen output for the two known cases:

    err_211  trk 7  in 1441.959  out 1441.942  DROP 0.017  INTRA 1000.469
    err_160  trk 2  in 1522.680  out  108.012  DROP 1414.668  INTRA 0.003

Baseline: 5 truncations in 103 screened merges = **4.85%**. **60 records** are
needed before a clean run is surprising; the sweep reached 7 this morning.

## 1.11 The launcher leaks a process

MEASURED, reproduced with `sleep 8.37`, timeout 1, max_restart 1: one process
still alive after the raise. `tools.launch_cmdExt_with_timeout_reload` (line 92)
kills on `TimeoutExpired`, `Popen`s a replacement at line 111, then falls out of
the `except` with `unpocessed` still true, so line 95 `Popen`s again and orphans
the line-111 process.

**This is inside the protected closure and was NOT touched.** It is with the
owner.

**Not err_211:** `exceeded_timeout` is false on 96 of 96.

---

# 2. WHAT IS WRONG AND STILL WRONG

Ordered by whether a restart must fix it before trusting the tool.

## 2.1 LOAD-BEARING — `provenance_classify.py` invokes bare binaries

`src/diagnostics/tools/provenance_classify.py`, in `streams()`, `pcm()` and
`pcm_digest()`: `["ffprobe", ...]` and `["ffmpeg", ...]` as literals.

The owner's standing rule is that every binary goes through
`tools.software["name"]`. `batch_verify._binary` exists precisely to enforce it,
and its docstring records that measurements taken before it "used whichever
binary was first on PATH — which we cannot now show was the same one the merges
being graded had used." **The same argument voids every measurement this
classifier has taken.** That is not a style issue; it is a provenance issue about
the provenance tool.

**Routed to Agent 4, which owns `src/diagnostics/`. Not fixed here** — two
writers in one module is a collision this project keeps paying for.

## 2.2 LOAD-BEARING — an absent candidate reads as a master-derived verdict

`resolve()` checks `merged_file` and `db_master` exist; **it does not check the
candidate**. `classify()` then does
`cnd_s = streams(candidate) if candidate and os.path.isfile(candidate) else []`,
so with the candidate absent every language falls into the structural gate and
reads `master_derived_by_construction` with `judgeable: False`.

**A missing file is silently indistinguishable from a real finding.** MEASURED:
no file among the 22 is affected today, so the hazard is latent, not active. It
would become active the moment the corpus is pruned.

Routed to Agent 4.

## 2.3 LOAD-BEARING — absence in the store is untyped

`resolve()` returns `None` for four distinct reasons (no classified row, no
`merged_file`, no `db_master`, either file absent) and `main` does
`if not paths: continue`, **writing nothing**. A restart reading the store cannot
tell "never attempted" from "attempted and unresolvable". Invariant 157, in a
tool none of us wrote.

## 2.4 LOAD-BEARING — the store cannot self-heal

`if str(eid) in results: continue`. Records are append-only by id and are never
refreshed, which is why §1.7 happened and will happen again. **Any restart that
re-runs the classifier without clearing the store gets the same stale answers.**

## 2.5 The verdict schema has no build identity

24 of 64 `in_sync` rows carry no `recorded_utc`, and none carries an identity for
the artefact it graded. §1.6's whole method is an mtime comparison standing in
for a fact nobody recorded.

## 2.6 Two paths grade the same question by different rules

`sync_verdict` (Path A) uses a bare `worst <= 50`, with no `judge_profile`, no
dissent note and no copy guard. `verify_against_master` (Path B) uses
`judge_profile` with all three. **Deliberately left unequal** — changing Path A's
rule would reclassify recorded verdicts, which is the owner's call. Recorded here
so a restart does not mistake it for an oversight.

## 2.7 Step 1's language pick is positional

`verify_unverified.py:38`, `shared[0]`. 39 of 40 stored verdicts carry it. Not
fixed; it changes recorded verdicts.

## 2.8 Reported and already actioned by someone else

The 24 post-fix `tools.software['ffmpeg'] is not configured` probe failures and
the duplicated `return binary` in `batch_verify._binary` were flagged from here
and picked up at `f65f8d6`. Noted so the restart does not re-report them.

---

# 3. WHAT WE MISUNDERSTOOD, AND FOR HOW LONG

This is the section with the most value in it. Two of these I withdrew myself
today and both withdrawals made the picture larger, not smaller.

## 3.1 WITHDRAWN — "the merge re-encodes audio to flac, so a master-derived track is never a byte copy"

**Believed:** roughly two hours, and it was load-bearing in a report to the Lead.

**True:** the base command carries `-c copy`
(`mergeVideo.py` ~3415). `generate_new_file_audio_config_second_pass` overrides
it in exactly two cases — a lossless source (to flac, which is *lossless*, so the
decoded samples are unchanged) and a lossy source **with a negative delay**.
Every other track is a straight stream copy.

**So it was wrong twice**: flac does not destroy sample identity, and most tracks
are not re-encoded at all.

**What would have caught it sooner:** reading the *caller* before reasoning about
the callee. I read the function that sets the audio codec and never looked at the
command it was extending, so I never saw the `-c copy` that decides the majority
case. A function that *overrides* a default cannot be understood without the
default.

## 3.2 WITHDRAWN — the copy guard read as a bit-identity test

**Believed:** about three hours, and I built a whole argument on it — that the
guard "tests for the one form of self-comparison this pipeline does not produce."

**True:** §1.1. The rounding four lines above redefines both thresholds, the lag
one by eight orders of magnitude, and the guard therefore *does* catch the
gentle same-source re-encode I had said it could not.

**What would have caught it sooner:** calling it. I read the predicate carefully
and correctly and still got the wrong answer, because the predicate is not where
the threshold lives. The test that found it was written asserting my own reported
behaviour, and two of its assertions failed on the first run. **Writing the test
as a restatement of the claim is what turned a belief into a measurement.**

Filed as **invariant 162**: a literal is a claim about the comparison, never
about the quantity — rounding, unit conversion, clamping and serialisation each
silently redefine every threshold downstream, and none is visible from the line
that states it.

**And 162 itself had to be narrowed.** I wrote that `refine_peak_index` landed
"upstream of a `round(l, 1)` that discards it before any decision reads it." The
Lead caught it: `round(l, 1)` **keeps** a tenth of a millisecond, so the
refinement survives for every decision above 0.1 ms and the residue method built
on it is sound. What the rounding destroys reaches exactly one comparison. **I
committed the same error the entry warns against, inside the entry.** The
narrowed form is the useful one: *a decision taken near the floor of its input's
stored precision is the one to check.*

## 3.3 WITHDRAWN — the provenance split I reported as fact

**Believed:** about four hours. I reported "2 store-confirmed cross-provenance /
11 structural-only / 20 all-indeterminate / 31 no record" as a statement about
the corpus.

**True:** §1.7. The store predates every output it grades. The split is an
accurate reading of the store and **not** a statement about the artefacts.

**What would have caught it sooner:** applying to my own source the caution I had
already accepted about someone else's. The Lead warned that classifying today's
file says nothing about the run the verdict came from. The identical argument
applies to a classification already in hand — one `getmtime` call — and I did not
make it until asked a different question.

**The general form, which is worth carrying:** a caution accepted about a
proposed measurement is almost never applied to the stored measurement it is
being compared against.

## 3.4 The completeness scan measured the masters

**Believed:** long enough to produce and report a full scan. `new_files[0]` **is**
`basename(db_master)` on 110 of 110 records, so the scan resolved to the master
file and measured it against itself.

**What would have caught it sooner:** the mtimes were already in hand — a median
198 days older than the outputs — and unexamined. It was caught only because
Agent 4 got a different count. **A number that disagrees with a peer's is worth
more than a number that agrees.**

## 3.5 The same degenerate predicate, a second time, today

MEASURED: when I re-armed the C_cut watcher I tried to split master from
candidate passes with `new_files[0] == basename(db_master)` and got **64 master /
0 candidate**. That is the *same predicate* from §3.4. This time I recognised the
output as degenerate rather than as a result — a predicate true for every record
is not a discriminator — and threw it away within one step.

**Kept in this report deliberately.** The lesson is not "I fixed it"; it is that a
known-bad inference reappeared hours later in a new context, unprompted. It will
reappear again.

## 3.6 The C_cut denominator

**Believed:** minutes, but it was sent to the Lead. I reported "64 records / 0
flagged" beside a figure the Lead was tracking at 4-to-5. **64 is the whole
historical class ledger; the current sweep is 6.** An order of magnitude
overstatement of the evidence base, placed directly next to the number it would
be compared with.

**What would have caught it sooner:** stating the population in the same sentence
as the count. "64" alone is not a claim; "64 C_cut records with a verdict, of
which 6 are from this sweep" is.

## 3.7 A test that matched its own docstring

`assertIn("load_software", source)` matched the real call **and** the function's
docstring, so deleting the call left the suite green. Found by mutation.

**What would have caught it sooner:** nothing textual. The fix was structural —
`_astcheck.py`, whose every text search runs over `code_text()` with all
docstrings and comments stripped. **A note about the hazard did not stop me
writing the hazard again hours after documenting it.** The only thing that
stopped it was not having the wrong form to hand.

## 3.8 The one the Lead caused, and my account of it

The Lead asked for a landing test and I **wrote the test as a docstring** and
implemented two magnitude tests instead — the stated principle and the
implemented check were different things, and Agent 3 found it.

I later described this as the Lead having asked me to write a principle into a
docstring. **The Lead corrected that account**, and the correction was right: the
docstring-instead-of-implementation was mine. Recorded here in the Lead's favour
because a handover that shades a shared error toward the other party is worse
than useless.

## 3.9 A wrong statement to a peer, corrected by data

I told Agent 3 that `_stderr_diagnostics` and `stderr_tail` were "separately
buffered". They are not — both derive from one `decoded` blob. Proved by data
after I fixed a hole in my own filter. Also: I read the function name
`_stderr_diagnostics` as the JSON key, with an `or` fallback to `stderr_tail`,
which would have produced a silently wrong answer. Root cause: **both spellings
are real in this codebase**, so the wrong one does not look wrong.

## 3.10 I audited a stale checkout

I declared "INVARIANT 152 DOES NOT EXIST" while 13 commits behind. It existed.
Damage bounded by diff; only section 0 of that audit was void.

## 3.11 My own timestamps were invented

10 of 10 archive-note stamps were suspiciously round and **+23 to +28 hours ahead
of their own commits**. Fixed by adopting `agent_state.write()`, which takes no
stamp argument.

## 3.12 Master-pairing by adjacency

Correct for `err_211`, **backwards for `err_160`** — its master pass ran 21 s
*after*. Fixed by identity from `batch_state` rather than by time order. Related:
the truncation denominator double-counted master passes; 45 of 90 records are
master passes and both known truncations are in candidate passes.

## 3.13 A mutation that never applied looked like a surviving mutant

My `sed` targeted `return joined[-cap:]`, a line that does not exist. Mutations
now verify the file changed. Agent 1's refinement, which is the correct rule: **a
non-applying mutation yields a survival, never a kill.**

---

# 4. UNKNOWN / UNRESOLVED

## 4.1 The band on codec-differing pairs — measured, and the news is bad

Unlike the rest of this section, this one **was** measured before the stop; see
§1.4. The result is the honest negative the Lead said would be better than a
wider band: **the same-source and different-source populations overlap, and no
threshold on this quantity separates a shared-envelope pair from a same-source
pair across a lossy codec change.**

What is still unknown: whether that matters. If §1.5's inference holds — the only
master-derived codec change this merge produces is lossless and reads 1.0 at
stage one — then stage two is never needed for these rows and the overlap is
harmless *here*. **That inference is unverified and needs one decode.**

**If it does not hold, the correct conclusion is that the measurement is the
wrong one, not that the band should be widened.**

## 4.2 The 22-file backlog

22 files, 32 `in_sync` rows, all resolvable today, none classified. Blocked only
by §2.4 (the store cannot refresh) and §2.1 (the classifier's binaries are
unconfigured, so its output may not be trustworthy anyway).

## 4.3 Things I saw and cannot explain

- **Ten pre-provenance `in_sync` rows, all `eng`, all reading 0.0 ms in every
  window across ten different files.** The store labels stream 0
  `master_derived_shifted` or `master_copy` and stream 1 `indeterminate`. Under
  the emission-order rule stream 1 is candidate-derived, so these are
  structurally cross-provenance; under the store they are undecided. Two
  instruments agreeing about one track and disagreeing about the other. Invariant
  86 is exactly about the emission-order inference failing when the assembler
  substitutes, so I did not resolve them. **Ten files aligning to better than
  0.05 ms across genuinely different sources is either a real and impressive
  result or a self-comparison, and I could not tell which.**
- **`tha` at ratio 0.482** — the PAL-correction anomaly recorded in
  `contributed_track_check.py`'s known-limitation section. Untouched.
- **26 files needing a cut-point scan**, and slope coverage at 166 of 178
  unscreened. Both need a decode and a script that does not exist.

---

# 5. WHAT TO DISCARD

Blunt, as asked.

1. **`provenance_classification.json` in its entirety.** Every record backing an
   `in_sync` verdict is stale (§1.7), and every measurement in it was taken with
   unconfigured binaries (§2.1). **Do not inherit a single verdict from it.**
   Its *method* is worth keeping; its contents are not.

2. **My "2 / 11 / 20 / 31" provenance split.** Withdrawn (§3.3). If it survives
   anywhere in the worklog it should be struck.

3. **Any statement that the copy guard is a bit-identity test**, including my own
   earlier report and any reasoning built on it (§3.2).

4. **Any claim that a master-derived track is never a byte copy** (§3.1).

5. **The `new_files[0] == basename(db_master)` predicate, permanently.** It is
   true for every record and has now produced two wrong answers on two different
   days (§3.4, §3.5). It should not exist in any tool.

6. **The Step 1 store's 40 verdicts as a coverage figure.** They are a four-day-old
   snapshot covering 4 of 50 eligible files, and 39 of the 40 carry the `shared[0]`
   defect (§1.9).

7. **The err_211 magnitude as a single number.** 1000.494 / 1000.487 / 1000.456
   are three samples of a moving quantity (§1.10). Quoting one is a fabrication.

8. **"64 C_cut records" as an evidence base** (§3.6). The sweep is 6–7.

9. **`SAME_RECORDING_PEAK = 0.98` applied to codec-differing pairs.** Not the
   threshold value — which is the owner's, and defensible on the population it
   was measured on — but its *extension* (§1.4).

**Do NOT discard:** the two copy-guard firings (§1.2), which are real refusals on
real files, and the 37/24/4 recovery split (§1.6), which is a direct read.

---

# 6. WHAT TO KEEP

## 6.1 The structural-versus-numerical framing — page one

Two tools guard against the same trap by incompatible means:

- **Step 1 (`verify_unverified`) guards STRUCTURALLY.** No more tracks in the
  output than in the master means the candidate contributed none, therefore
  unverifiable, citing invariant 6. It refuses on a **count**.
- **`verify_against_master` guards NUMERICALLY.** It refuses only when every
  window is `abs(l) < 1e-9 and p >= 0.999` — in practice 0.05 ms and 0.9985.

On the same configuration one refuses and the other passes. **Promoting
`per_language` to Step 1 status would resolve precisely the cases Step 1 exists
to refuse.** MEASURED: of 64 `in_sync` results, 22 are "output vs master file"
with `out <= ref`, the exact configuration Step 1 refuses.

The general lesson, and the reason this belongs on page one: **a structural guard
and a numerical guard for the same hazard will disagree, and the disagreement is
not a bug in either — it is the absence of the fact they are both reconstructing.**

## 6.2 The recommendation this all converges on

**Write provenance at merge time instead of reconstructing it afterwards.**

- **What:** per audio track in the output — `source` (`master` | `candidate`),
  the source's stream index, the applied delay in ms, and `copied | re-encoded`.
- **Where:** by the assembler as it builds the merge command, into the per-output
  record beside `merged_file`. **Not** a container tag — the question is asked by
  tools reading the ledger, and a tag needs a decode to read.
- **What it settles:** the 51 undistinguished `in_sync` rows become readable
  rather than inferable; the 22 rows where Step 1 and `verify_against_master`
  disagree stop needing a rule; the copy guard becomes unnecessary rather than
  approximate; `master_derived_shifted` stops being a category because a shift is
  a recorded delay; and the `codec differs` reason is answered directly by
  `copied | re-encoded`.

**The strongest argument for it is the one I did not expect:** every row that is
recoverable retrospectively is recoverable only because an mtime happens to
predate a timestamp. **That is luck about what has been rebuilt, not a property
of the system.** A recorded `source` needs no artefact to still exist, no mtime
comparison and no threshold.

Three tools reconstruct provenance three different ways — `sync_verdict` by
emission order, `verify_lang` by counts, `provenance_classify` by codec and
correlation — and where they disagree there is no tiebreak, because none of them
observed the fact.

## 6.3 `test_66_the_copy_guard.py` — 18 tests

Keep it whole. It fires the guard deliberately, requires the same call to return
`unverifiable`, `in_sync` **and** `out_of_sync` (invariant 141 capacity), pins
both written literals **and** the rounding that redefines them off the AST, and
carries the shared-envelope result. Its docstring records that two of its
assertions were written the other way round first and failed.

## 6.4 The literal-array method

Measure a guard **from outside**, by calling it with literals, and measure the
instrument behind it by reproducing its arithmetic on synthetic arrays. No
decode, no container, no fixture. It found §1.1, §1.3 and §1.4 in one morning and
it is the only reason any of them are numbers rather than readings.

**The rule that makes it work: write the test as a restatement of what you
believe, then run it.** A test written to confirm cannot fail informatively; a
test written to restate can.

## 6.5 `_astcheck.py`

The module that refuses the vacuous form. Every text search runs over
`code_text()`, with all docstrings and comments stripped, so prose describing an
expression cannot satisfy a check about it (§3.7). Roughly 86 raw-source
assertion lines remain tree-wide and are owed by other agents; mine are
converted.

## 6.6 The invariants filed from this territory

- **149** — compute what the null returns before crediting a ratio.
- **160** — a value separated from its neighbours deserves a check even under the bar.
- **161** — a conclusion that survives its method being refuted is not the same as one that was checked.
- **162** — a threshold is whatever it is applied to, not what it is written as; and a decision taken near the floor of its input's stored precision is the one to check.

## 6.7 Standing operational rules that earned their place

- **A non-applying mutation is a survival, never a kill** (Agent 1's rule).
- **A skipped test is a defect, not a pass.**
- **State the population in the same sentence as the count** (§3.6).
- **`err_<id>` only; never a filename or a library path.**
- **The protected closure is 78 members; verify with `closure_check.py`.** It was
  intact at every commit from this worktree, including the leak in §1.11 which
  was reported and not fixed.

---

## Status at handover

Branch `agent/d-trim`, not pushed. Last committed work `bc51dc6` on top of
`da5dac5`. Suite was 2308 green with no skips at that commit; **no gate was run
for this handover, as instructed** — it adds one document and changes no code.
C_cut watcher disarmed. The band recalibration in §1.4 was completed before the
stop arrived; nothing else was in flight.

---

# 7. ADDENDUM — things in my head that were not in this document

Added after the report was committed, in answer to "say now anything that would
be lost." These are not conclusions; they are operational facts and gotchas that
cost time to learn and are written nowhere else.

## 7.1 The join between the three stores, which is not obvious and is the one that bit me

Three stores, three different key spaces:

    batch_state.json                  keyed by ABSOLUTE SOURCE PATH
    classified.json                   a LIST of rows, each with "id" and "file_path"
    provenance_classification.json    keyed by ERROR NUMBER as a STRING ("25", "104")

The join is `{r["file_path"]: str(r["id"]) for r in classified}`. **My first
lookup used the path md5 and returned "not present" for all twelve rows I was
checking** — a clean, confident, entirely wrong absence. It looked exactly like a
finding.

**The general rule this earns:** before believing an absence, print one key from
each side and confirm they are the same kind of string. It costs one line and it
is the second time this project has been misled by a key space (the first was the
completeness scan resolving to the master).

## 7.2 How I proved a running process was holding stale code, cheaply

The 24 post-fix probe failures were established by comparing the **stored error
text** against the **error text the current source emits**:

    stored   "tools.software['ffmpeg'] is not configured; refusing to fall back ..."
    current  "tools.software['ffmpeg'] is not configured and <config> has no
              readable [software] section; refusing to fall back ..."

Different wording, post-fix timestamps. A long-lived process was still running
the old module.

**Keep the method.** When a fix is in the tree and the symptom persists, the error
string is a version stamp nobody thinks to read, and it distinguishes "the fix is
wrong" from "the fix was never loaded" without touching the process.

## 7.3 `final_status` is deliberately NOT fed the completeness result

`master_summary.py:275` computes `final_status(outcome, verdict, step1_ms, step1_verdict)`
and puts `complete` and `complete_state` **beside** it, with a comment saying
`final_status` is not passed the completeness result.

**This is deliberate and a restart should not "fix" it.** A file can be in sync
on every measured language and still be missing content; folding completeness
into the status would let one answer mask the other, which is the collapse that
let err_20's eight bad languages hide behind one good reading. I nearly wrote
this up as a gap and checked the comment first.

## 7.4 The merge writes to disk outside its output, and it is bounded

`mergeVideo.py:5298` onwards. First-pass intermediates are retained under
`/config/output/vmsam_agent/retained_first_pass`:

    RETAIN_FIRST_PASS_MAX          = 10
    RETAIN_FIRST_PASS_FLOOR_BYTES  = 30 GB

These run about 3.9 GB each. The cap and the free-space floor **are the terms of
a narrow approval**, not polish: it was approved because two questions in one
night were decidable from a single metadata read of one intermediate and both
times the artefact had been deleted. It was **not** approved as an always-on
behaviour.

`_evict_oldest_retained` never follows or removes symlinks. Bounds are asserted
in `test_59_retain_first_pass_bounded.py`, including that retention does not
alter the pass it retains and does not by itself cause a merge to run.

**A restart that does not know this exists will find several gigabytes
accumulating in a directory it did not create.**

## 7.5 A detail that corroborates §1.5 and that I did not connect at the time

The classifier's own calibration records `master_derived_shifted   n=27
0.9833 .. 1.0000`, and **25 of those 27 read exactly 1.0000**.

That is independent support for the inference in §1.5: master-derived tracks in
this corpus are overwhelmingly **bit-transparent** — copies, or lossless
conversions — rather than merely similar. It makes "the codec-differing rows
would be decided at stage one" more likely than I claimed, and it was sitting in
the docstring I quoted the band from.

**It is still an inference and still needs the one decode.**

## 7.6 The C_cut power numbers, which the report gives only as a threshold

Baseline 5 truncations in 103 screened merges = 4.85%. §1.10 gives the 60-record
threshold; these are the numbers behind it:

- at the projected ceiling of ~8 candidate passes, a wholly clean run happens
  **67%** of the time **with the failure rate unchanged**;
- at the n=3–4 the sweep actually reached, **82–86%**.

**So a clean sweep at this size is the expected outcome, not evidence of
improvement.** Anyone tempted to read the sweep's silence as progress needs these
two numbers in front of them.

## 7.7 Smaller items, each of which cost time

- **`-t` under `-copyts` cuts at an absolute TIMELINE POSITION**, not after that
  much elapsed content. Several early readings were wrong because of this.
- **The `DURATION` tag is a timeline END, not a length** (invariant 156). Length
  is `tag - start_time`. The landing question wants the ends, not the lengths,
  and the two are easy to swap.
- **`file_id = md5(subject_path)[:16]` is stable for durable library paths and
  MOVES for per-run temp paths.** Anything keyed by it across a temp-path run is
  silently a different record.
- **`_stderr_diagnostics` filters and then slices** (`joined[-cap:]`), so the
  stored field is a true suffix of the filtered stream, not of the raw one. It
  and `stderr_tail` derive from **one** `decoded` blob; I told a peer they were
  separately buffered and that was wrong.
- **`judge_profile` condemns only on a MAJORITY of windows over the bar.** A lone
  dissenting window is reported as a suspect window on an otherwise in-sync
  verdict. Two outputs were once filed out_of_sync on single windows at -1470 ms
  and -2540 ms with peaks of 0.59 and 0.47 while every other window sat at
  +40 ms.
- **`run_tests.py` runs discover with `-v` so a skip can be NAMED**, and
  `gate_result` takes `skipped=()`. A skipped test is a defect here, not a pass,
  and the gate can now say which one.

## 7.8 The thing I would most want a restart to know, in one sentence

**Every disagreement between the verification tools in this repository is the
same disagreement**: three of them reconstruct, by three different methods, a
fact the merge knew and did not write down — and where reconstructions disagree
there is no tiebreak, because none of them observed it.
