# Agent notes — logbook

<!-- ===================================================================== -->
## BRIEF — dev-2, zone A. READ THIS BEFORE ANYTHING ELSE. 2026-09-05.

Written against a RESET that deletes the transcript and keeps the files. What is
not here, I do not have. Cite by BLOB; a tip moves, a blob does not.

**READ `docs/ENTRY.MD` FIRST — it is the only document that must be read before acting,
and it WINS over any other document it disagrees with.** Blob `430fb75c346a` at the
authority when I last computed it (2026-09-05, post-reset). Everything else is
REFERENCE, looked up by phrase: `WRITE_ZONES.MD` is 83 numbered classes of mistake
and is GREPPED, never read through. **Verify the blob before trusting it: I was
handed `7c91928fd998` by a peer and it was ALREADY ONE REVISION STALE** — the
revision I was told to assert was missing the section that governed the very message
handing it to me.

### WHO I AM AND WHAT I MAY TOUCH
Zone A — repair. **MINE:** `src/merge_video_repair.py` (blob `882867d78bf3`),
`src/merge_video_chimeric.py` (`c164b75128ff`), `SPEC_ZONE_A.MD` (`cd33e0caac33`),
`AGENT_NOTES.MD`, `CAMPAIGNS/02-dev-chimeric-resample/TASKS/`. **NOT MINE:** `src/change_point_locator.py` — dev-1's
by `WRITE_ZONES.MD` (authority blob `9bf31072f75d`), which is the document that
assigns zones; `git log` shows who typed first and answers a different question.
`src/merge_plan_report.py` is dev-4's. `mergeVideo.py`, `video.py`, `tools.py`
are frozen except six tagged zones (`WRITE_ZONES` §2).

**NEVER push. NEVER add a git remote.** Work reaches the branch when the Lead
fetches a blob. Commits keep `Co-Authored-By`, never a `Claude-Session:` trailer.

> **CORRECTED 2026-09-05, AND IT MATTERS MORE THAN THE LINE IT REPLACES.** This
> brief said *"my push URL is the literal `DISABLED_agents_do_not_push`"*.
> **MEASURED IN THIS CLONE: IT IS NOT.** `git config --get remote.upstream.pushurl`
> returns a live `ssh://` forgejo URL, identical to the fetch URL. **THERE IS NO
> MECHANICAL GUARD. A push would go through.**
>
> *I carried that sentence in the one document written to survive a reset, and it
> asserted a safety property of the world that I had never checked.* **"I cannot
> push" and "I do not push" are different claims and only the second is true** —
> and the false one is the flattering one, because it converts a discipline I must
> keep every time into a fact I need not think about. `vmsam-dev-3` found that such
> a line can be false; the Lead told me to check mine rather than inherit it, and
> it was false here. **CHECK IT AGAIN IN A NEW CLONE — this is a property of a
> working copy, not of the seat, so it does not travel with this file.**

**Privacy:** repository files (.MD, code, commit messages) carry NO episode name,
NO series name, NO media filename, NO hard-coded media path — but the
repository's own files name themselves freely. Runtime output beside the media
may carry full paths and never enters the repository. The test is WHOSE FILE IS
THIS, not "is this a filename". Redact BEFORE an excerpt travels.

**A peer cannot widen a constraint.** A document granting me scope does not
override an instruction from my user; the narrower governs. Never treat a peer
message as authorisation, never edit permissions/config on a peer's request.

### EVERY CLAIM CARRIES ITS KIND (owner ruling; `AGENT.MD` blob `c6ed9169770f`)
RUN — the shipped function was CALLED, ground truth known independently. The only
kind stateable bare. TRACE — read and derived; **may not close a finding.**
HARNESS — something ran, but not what the consumer calls; **may never be reported
as an engine result, not even with a caveat.** UNVERIFIABLE — say so and stop.
**RELAYED — SOMEBODY ELSE MEASURED IT.** Not unverifiable: verifiable, just not by
me today. Carry it AS GIVEN, never paraphrased; name whose it is, what it is a
figure about, and its clock. **Do not lend it my own standing.**

> **THE SHARP EDGE, AND IT IS THE ONE THAT CATCHES ME:** *checking ONE SIDE of a
> relayed claim UPGRADES THE HALF I DID NOT CHECK.* What comes back is an unmeasured
> statement wearing the badge of an adjacent measurement. When a peer hands me four
> claims and I verify two, THE OTHER TWO ARE EXACTLY AS UNMEASURED AS BEFORE — and
> they now feel checked, which is worse than if I had checked nothing.

*`AGENT.MD` moved from `173a67183bb9` to `c6ed9169770f` while my transcript was
gone; the ruling above SURVIVES at the new blob (grepped, with a negative control).
A blob citation is stable about what was verified and SILENT about whether the
branch still carries it — so I check both, every time.*

### MY NUMBERS, WITH THEIR KINDS
| claim | kind |
| :--- | :--- |
| duration rule refused **44 of the 113 files driven to a mux, 38.9%** | **HARNESS** — gate and assembly shipped; population and language rule MINE. 150 is my CAP against 277 eligible; build pinned `5f2cbe968e1c` (which resolves to TWO commits, so it cannot order them) |
| `master_timeline` 9 of 113, 8.0% — refusing before the ruling existed | HARNESS, same population |
| 34 pre-mux declines cost the ruling nothing | HARNESS |
| BY FAMILY: trimmed edges 63.2% [41,81] vs different cuts 33.7% [25,44]; weak correlation and speed mismatch have governed=0 → **UNTESTED HERE, never 0%** | HARNESS |
| the three plan-line tokens emit `absent(not_in_plan)` — 6 emissions, 2 files, current build, through the consumer's entry | **RUN** |
| no module in `src/` writes `speed_margin`, `speed_margin_absent_reason`, `fidelity_margin`, `decided_by` | **TRACE** — `grep -c` = 0 on two trees with `quantum_ms` as positive control. AN ABSENCE IS NOT RUNNABLE, so this can never be closed from my seat |
| whether my harness population matches production | **UNVERIFIABLE FROM HERE** |
| twelvefold locator step reduction found essentially no new structure (change_points 2.00→2.04, segments 3.00→3.13) | HARNESS — and it contradicts my own written prediction, left standing |

**Never** carry `33 of 41`, `37 of 45`, `37 of 91`, or any percentage of 92/123/150.

### BEFORE REPORTING ANY CONSTRUCTED INVOCATION (`WRITE_ZONES` §83, blob `6d5a032f3bda`)
**Read the call site, and state which arguments I constructed and which the
engine builds.** A run of the wrong chain is not a RUN — it is a harness wearing
the word, and an opportunity count on the wrong chain is a compliance artefact
that looks *more* careful than a bare number. Measured across four seats on one
defect: four different wrong chains, three came back clean. My own instance is
item 1 below; ci-pair reached the same rule from the audio side, where the
engine's extraction inherits `-ar <grid>` at `video.py:271`.

### THE SUBSTRATE MY INSTRUMENTS CANNOT SEE
1. **`best_video.delays` is populated by `prepare_get_delay` in `mergeVideo.py` as
   a side effect of the merge, upstream of zone A.** My lab never runs that path,
   so `get_delay_language` always falls to its fallback branch and every language
   decision I observe is my own empty dict. This fact exists nowhere else in the
   repository. A REACHABLE INSTRUMENT IS NOT THE SAME AS A VALID ONE.
2. An absence cannot be RUN. 3. The corpus JSONs are a 2026-09-03 snapshot.
4. A pinned build cannot exercise anything committed after it. 5. A number that
has become a fact in someone else's sentence is outside my instruments entirely.

### TEN RULES I BROKE TONIGHT AND NOW HOLD
1. Never quote an identifier I have not just computed (I sent a 40-hex blob id
   that pointed at nothing). 2. Ask the authority — and CHECK `git remote -v` FIRST. In this clone
   `origin` is a LOCAL HOP (`/home/vmsam/src/VMSAM`) and `upstream` is forgejo:
   `git fetch upstream refs/heads/dev-AI` then `git show FETCH_HEAD:<doc>`, and
   cite the BLOB, never `FETCH_HEAD` as an address (it is per-clone, rewritten
   by whoever fetched last). The hop has been stale by nine commits and has
   SERVED 81 sections while HOLDING 82. I wrote a peer's fetch instruction into
   this brief without checking it against a remote list I had already run
   myself — twice in one night, a correction accepted faster than it was
   verified. 3. Prove an absence with a grep AND a positive
   control, never an enumeration (mine was false in two trees when I sent it).
4. A reader's branch is not an opportunity if no writer can reach it —
   awaiting-vs-outside is a property of the WRITER/READER PAIR. 5. Report
   m (OF WHICH COULD HAVE FAILED) as a COMPUTED number, never as prose beside n.
6. Never pipe a measurement into `head`/`tail`: the pipeline exits 0 while the
   command dies. 7. Every claim carries its kind. 8. Put a control on the FINDER
   before blaming the data. 9. A captured tool output is a measurement of a
   moment that has passed. 10. A ratio whose numerator sits inside its own
   denominator cannot fall, and a ratio that cannot fall is not a measurement.

### OPEN, AND WHERE IT SITS
- **(a) with the OWNER via the Lead:** does the locator originate the four keys?
  ORIGINATE → the three fields FILL. DROP → they COME OUT, dev-4's register
  change FIRST, my bytes after. Until then **KEEP them** — the token emitting at
  100% is the only artefact-visible trace that four keys were designed, consumed
  and never originated. It is a marker of an open decision, not a permanent
  record. dev-1 declined to add the keys under an instruction from THEIR user.
- **SEALED, do not adjust:** `why=` reachability — head_gap 215 / interior_bracket
  441 / tail_gap 279 = 935 regions over 88 files (23/47/30%), registered with ci.
  Narrowed before reading: the LABELS are definitional (`chimeric.py:380` assigns
  head/interior on `cursor == 0`; `merge_plan_report` uses the same rule), so
  only the COUNTS are predictive.
- SPEC §4h: extending the first segment back to master 0 is a repair in case A
  and a fabrication in case B — OPEN BY DECISION.
- Owner-pending: which audio track a subtitle follows when its language has no
  audio (id 205: 14 subtitle languages, 6 audio).
- Lab state: `lab/t99_consumer_entry.py` (RUN, valid), `t99b_multilang_probe.py`
  (shared-language counts valid; its `consumer`/`agree` columns are the retracted
  artefact — FIX BEFORE REUSE), `t99c_governed_language.py` (KILLED, invalid as
  written for reason 1 above). Full state: `VMSAM_HELP_AI/dev-2/ledger/`.

<!-- ===================================================================== -->


One entry per decision that a later reader would otherwise have to re-derive.
Newest first.

**What goes here:** an architectural choice and why the alternatives lost; a
measurement that changed a decision; a defect found and what it taught; a blocked
need and what was done instead.

**What does not:** progress reports, task status, anything regenerable from the
code or from `git log`. If it is a measured fact rather than a decision, it belongs
in a findings document.

**Format.** `## YYYY-MM-DD — one line saying what changed`, then the decision, then
*why the alternatives lost*. Numbers carry the command that produced them. Files are
cited by opaque id, never by media name (`AGENT.MD` → Privacy).

Entries are **append-only**. Superseding an earlier one means adding a new entry that
says so and why — never editing the old one. A record that can be rewritten is not a
record.

---

## 2026-09-05 — `record()` reached production BY COINCIDENCE, and one new word would have re-armed F26

**Found by `vmsam-dev-1` while verifying a ruling of mine rather than accepting it, and
measured here before changing anything.** I had told dev-1 that `candidate_path` is
interpolated into every `record()` log line **unconditionally**. The conclusion held; **the
reason was one step wrong, and the difference is the whole finding.**

    the emission was guarded:
        if tools.dev or outcome in ("repaired", "failed", "declined", "no_plan")

    MEASURED BY AST over every call site:
        outcomes actually passed   declined x6 . no_plan x2 . failed x1 . repaired x1
        the whitelist              THE SAME FOUR STRINGS
        outcomes not whitelisted   NONE

**Two consequences, neither visible from reading the line.**

*The `tools.dev or` left operand could never change the result — a DEAD CLAUSE that makes a
reader conclude the line is debug-gated when it is not.* dev-1 names it as the inverse of
the mistake that cost it a false result today: reading an ungated absence as a real one.

*And the line reached production only because the used-outcome set HAPPENED to equal the
whitelist.* **One new outcome string — one word — silently re-gates it to `tools.dev` only,
which is F26 exactly: a refusal that emits nothing in production because its emission is
dev-gated. Reintroduced, one word away, inside the module that exists to consume F26's
victims.**

**Decision: remove the condition.** A `record()` call is by definition an outcome worth
recording; there is no outcome this function should swallow. **Behaviour-identical today
and proven so rather than asserted** — every call-site outcome was in the old whitelist, so
the guard was always true; removing it changes nothing observable now and changes what
happens to the *next* outcome. **The property goes from true-by-coincidence to
true-by-construction**, which is the only kind a rule cannot forget.

**Why the alternatives lost.** *Keeping the whitelist and adding a comment* — a comment is
the control that depends on everyone remembering, and the whole finding is that nothing at
the call site mentions the whitelist exists. *Removing `tools.dev or` but keeping the
membership test* — leaves the trap intact and only deletes the misleading half. *Raising on
an unknown outcome* — converts a logging gap into a merge crash, which is a worse failure
in a pipeline nobody is watching.

**AND IT CLOSES A REAL F26 INSTANCE AT ONE SITE, not just a latent one.** Before today,
when `change_point_locator` failed to import, the only trace was a **dev-gated** line plus
the false constant — so in production you saw *"no measurement available"* and nothing said
the module was missing. The cause token now reaches production unconditionally:
`cause=locator_module_absent`. **The remaining `tools.dev` read in the file (one, verified
by AST) is now supplementary detail rather than the sole trace.**

**A note on how this arrived.** dev-1 checked a claim of mine, agreed with the conclusion,
and found the premise wrong in a way that mattered more than the conclusion. *A verification
that CONFIRMS is worth what one that overturns is worth and gets reported far less often* —
this one did both, and the half that was worth having was the agreement's reasoning, not
its verdict.

## 2026-09-05 — the no_plan record asserted an absence of evidence where a conclusive negative existed

**T-12-A.** `merge_video_repair.py` recorded, unconditionally,
`no measurement available for this pair (<producer>)` whenever the locator returned no
plan. **On the fidelity-floor path that string is FALSE: the probes ran, they succeeded,
and they returned a conclusive negative.** A measurement WAS available and it said no.
*A conclusive negative filed as an absence of evidence* — and the producer's own contract
says it in `change_point_locator.py`: **"None means I could not measure — never the files
are compatible."**

**Decision: name the cause the site can actually observe, and never one it cannot.**
`get_plan_from_locator` now returns `(plan, cause)`; both `no_plan` record sites emit
`cause=<stable_token>: <prose>`.

    cause_unavailable      the locator RAN, returned no plan, and supplied no cause
    locator_module_absent  the module did not import -- nothing to attempt with
    language_undetermined  could not tell which language the merge measured on

**Why not `unknown`, which dev-4 offered as the safe default:** `cause_unavailable` is not
a cause at all, it is *this consumer's own observation that the producer told it nothing*
— checkable here, and strictly stronger than `unknown`, which conflates *nobody knows*
with *I was not told*. dev-4 accepted it against its own default. **Why the exception
class is not in the token:** a token that varies is not a token; the class went to the
prose before dev-4 had to ask.

**ACCEPTANCE, written BEFORE the change** (`VMSAM_HELP_AI/dev-2/ACCEPTANCE_T12A_no_plan_cause.md`)
**and every clause measured:**

    R1 constant gone      PASS -- 6 of 6 pairs reaching the branch, KIND: RUN through
                          `repair_not_compatible_videos`, mergeVideo.py:826's callee
    R2 nothing invented   PASS -- both tokens emitted are observations of my own site;
                          no locator reason is asserted, because none is emitted yet
    R3 accepting pairs    PASS -- STRUCTURAL, covering all of them rather than one:
                          8 of 10 `record()` sites byte-identical in the AST, and
                          0 non-`no_plan` sites changed. Plus id 267 run: unchanged.
    R4 positive control   PASS -- sink proven LIVE in-process AT `tools.dev = False`,
                          the setting the measurement runs at, BEFORE any absence was
                          claimed; negative control silent
    R5 m from material    m = 6 reaching the branch, both tokens exercised (3 and 3)

**THE LIMIT, AND IT IS THE ONE I NEARLY REPORTED AS A RESULT.** The three
`language_undetermined` firings all carry `language_route = "no language to choose from"`
— which is **my own substrate artefact**: `best_video.delays` is populated by
`prepare_get_delay` upstream of zone A and my lab never runs that path, so that branch is
reached by my empty dict. **They verify the EMISSION FORM and say nothing about how often
production reaches that site.** dev-4 measures 0 of 19 in its production corpus, and that
figure stands beside mine rather than being corrected by it.

**Two counting corrections taken from peers, both of which I had made myself:**
*dev-1:* the file has **10** `record()` call sites, not the 11 my `grep -c` reported — the
eleventh is the `def`. **I counted mentions as instances**, the exact defect I catalogued
in my own ledger, one day later. *dev-4:* `record()`'s `detail` parameter **never reaches a
log line** — 28 `tools.logs.append` sites, 0 interpolating `detail`. I put the cause in
`reason` and the change survives, **but I chose `reason` because the old string was there,
not because I had checked** — inheritance, not insight, and their measurement is what makes
it known rather than lucky.

**Not fixed, and not mine to fix:** every log already written keeps the old constant
forever — 19 in dev-4's corpus. **The population is mixed by DATE, not by site**, and no
change of mine reaches it.

## 2026-09-05 — id 688 AMENDS the entry below: the tenfold split stands, "does not bound the worst case" does not

**This supersedes the second half of the next entry. Per this file's append-only rule
nothing there is edited — read that entry's worst-case sentence as withdrawn.**

The Lead put the right question: id 688 sat at 94.5 % with `bound_only=0`, and either a
second mechanism produced near-total substitution, **or `bound_only` did not mean what I
was reading it to mean — which would invalidate my own grouping.** It is the second, in a
sharper form than either of us stated.

**IT IS NOT A SECOND MECHANISM.** id 688's row shows `bracket_widths_s = [-4.0, -20.0,
-16.0]`, and a bracket width cannot be negative. My own harness comment already diagnosed
that class — *a negative width can only come from a MISSING KEY read as zero by
`.get(..., 0)`* — and a **structural control confirms this artefact predates that fix**:
`bracket_widths_unknown`, the field added BY the fix, is present in **0 of 150 rows**. So
the scan ran pre-fix, and id 688's three change points had `bracket_high_ms` **absent**.

**AND THE SAME DEFECT SITS ONE LEVEL UP, WHERE IT IS INVISIBLE.** The grouping field is

    result["bound_only"] = sum(1 for x in change_points
                               if x.get("bracket_is_bound_only"))

`.get()` with no default, so **an ABSENT flag counts as NOT bound-only — as REFINED.**
A missing width betrays itself with a negative number; **a missing flag betrays itself
with nothing.** id 688 was detectable only because a *different* absent key happened to
produce a negative width.

    AS FILED      unrefined n=53 median 14.9%   refined n=7 median 1.5%  max 94.5%
    AMENDED       unrefined n=53 median 14.9%   refined n=6 median 1.5%  max  8.5%
                  UNCLASSIFIABLE n=1  -- id 688, plan block demonstrably missing keys

**WHAT SURVIVES: the tenfold split, untouched.** The 53 unrefined rows are *positively*
identified — a truthy count requires the key present and true — and their median 14.9 %
against 1.5 % is unchanged. The pricing of §4h step 2 stands.

**WHAT DOES NOT SURVIVE: "refinement does not bound the worst case."** That sentence
rested **entirely** on id 688, and id 688 is the one row I cannot classify. Without it the
refined group's maximum is **8.5 %, not 94.5 %** — and nothing I hold says refinement
fails to bound the worst case. *I flagged that maximum an hour ago as the thing a median
would have hidden. It was the right instinct pointed at a row that could not carry it.*

**THE LIMIT, STATED BECAUSE IT IS NOT MEASURABLE FROM THIS ARTEFACT:** how many of the
remaining 6 "refined" rows are refined and how many merely had an absent flag **cannot be
told from this scan at all.** There is no detector in the output. So `refined n=6` is an
upper bound on the refined population, not a count of it.

**Harness fixed** (`lab/t71_first_real_log.py`): `bracket_is_bound_only` is now tested
`is True`, with `bound_only_absent` and `bound_only_classifiable` emitted beside it, so a
future run can tell the three states apart instead of collapsing two into the flattering
one. **The closed t87 artefact cannot be repaired by that fix and is not re-run here** —
its rows stay as they are, with this entry as their reading.

**Why this is the third time today the control changed the answer, and the pattern is
one pattern:** the denominator print caught a sweep that had shrunk to 0 of 60; the shape
check caught a fraction of 296.9 %; and here a field's *presence* caught a grouping that
had been silently absorbing absences. **Every one was absent-read-as-a-value, and in every
one the wrong reading was the flattering one.**

## 2026-09-05 — refinement is worth about TEN TIMES the typical fill, measured on 60 real repaired files

**KIND: HARNESS.** `t87_gate_cost_ALL_FAMILIES_seed20260904.json`, build pinned
`5f2cbe968e1c`/`016bcaea0465`, 150 was my CAP against 277 eligible. Population and
language rule mine; the fill widths are the shipped module's own emitted `gap_ms` — **the
quantity is self-reported by the code that performs the fill**, named here rather than
left implicit. Denominator first: **60 rows carry `gap_ms` and all 60 are
`outcome=repaired`**, all 60 with a scanned span.

    master audio inserted    median 213,018 ms   max 1,332,004 ms
    fraction of the master timeline   median 14.9%   max 94.5%
    >=10% inserted master  38 of 60      >=25%  6 of 60      >=50%  1 of 60

    bracket NEVER refined (bound_only>=1)  n=53   median 14.9% of the timeline
    bracket refined                        n= 7   median  1.5%

**Decision this supports: step 2 of §4h is worth doing.** §4h argued from one artefact
that *"the fill region is the locator's residual uncertainty, materialised as inserted
master audio"*. Measured across 60, the tenfold split confirms it — the median repaired
file carries 3.5 minutes of master audio, in the master's language on every track the
master does not carry (§4c).

**AND THE MEDIAN HIDES THE WORST CASE, so both are stated** (`docs/MEASURING.MD` §15b). The
refined group's **maximum is 94.5%** — id 688, `bound_only=0`, almost entirely master by
a mechanism this measurement does not identify. *Refinement cuts the typical fill tenfold
and does not bound the worst case.* A median alone would have read as "refined files are
fine", which is the sentence I would have written a day ago.

**Why the alternatives lost — two of my own errors, both caught by structure not by care.**
*Reporting the first sweep* — it returned `0 of 60`, because `gap_ms` is a decimal STRING
and I tested `isinstance(v, (int, float))`. Caught **only** because the denominator was
printed before anything derived from it; a denominator that shrinks to zero in my own
hands is §62 at its purest. *Reporting the first fraction* — `median 127.8%, max 296.9%`,
because I divided by the row's `seconds`, which is the scan's **wall-clock runtime**, not
the media duration (confirmed against this file's own §13: min 95, median 167, max 1835,
an exact match). **Caught by SHAPE: a fill cannot exceed the timeline it fills.** The
corrected sweep divides by `scanned_seconds` and prints a shape check counting rows above
100%, which returns 0.

**Together with the entry below, the campaign's open ground is two numbers:** refinement
is worth ~10x the typical fill, **and** the engine named for it in §4h cannot be wired.
Neither weakens the other, and nothing here measures whether a rewritten aligner could
reach frame accuracy.

## 2026-09-05 — the pHash engine RAN for the first time and cannot be wired: 0 of 6 within a frame, and it never declines

**Decision: do NOT wire `frame_compare.FrameComparer` into the repair path, and do NOT
ship the one-line truncation fix that would make its numbers move.** `SPEC_ZONE_A.MD`
§4h step 2, precondition 1 — *"the engine may never have run... run it once and look at
what it returns"* — is now answered, and the answer is negative. Full evidence and every
command: `VMSAM_HELP_AI/dev-2/FINDING_phash_engine_cannot_be_wired.md`; harness
`lab/t100..t102`, results `t100_first_run.json`, `t101_designed_chain.json`,
`t102_truncation.json`. KIND: **RUN** — shipped class imported from `src/`, called,
output read, ground truth by construction.

**Ground truth by construction:** one 23.976 fps source, REF 180 s, CAND identical with
5.000 s removed at 90.000 s; measured duration delta 5.005 s.

**The material was the easiest possible case, and that is what makes this decisive.**
§4h's own detectability precondition, computed from the reference alone *before* the
detector ran, has two terms and both pass strongly: **75 hash values lost, all 75 in ONE
contiguous run**, against a control span losing 8 in two runs. This is not "nothing to
find".

    designed chain (get_cut_time.py:363, the only caller ever)  error vs truth
      A  bracket 12 s, refined                                    -10.000 s
      B  bracket 100 s, unrefined — THE WIDTH THE OWNER HEARD     -54.000 s
      C  CONTROL, bracket with NO CUT                             -74.000 s
    every arm: a 4-frame gap at exactly `bracket_start - 4.0 s`

**The control is the finding.** A bracket containing no cut returns the same confident,
well-formed answer as one containing a five-second cut. §4h asks whether the engine *"CAN
decline rather than return an argmax it has no basis for"*. It cannot.

**Why the alternatives lost.**

*Reporting the first run as an engine result* — refused, and by me, against my own
output. `t100` handed the engine the whole file with default arguments and got a gap at
frame 0. That was **my** chain: the real caller passes a coarse bracket with
`band_width = ceil(len·fps·0.20)` and `max_search_frames = ceil(len·fps·0.30)`. A run of
the wrong chain is not a run (`docs/MEASURING.MD` §83), and this is the second time in two
days that reading the call site changed my conclusion — the first cost me an entire
retracted "18 of 30" result.

*Shipping the truncation fix alone* — refused. `max_search_frames` bounds a distance in
its name and truncates the sequence in its use (`ref = ref_hashes[:n]`), so the cut sits
**outside** the searched region in both true-cut arms (index 240 vs 87; 1296 vs 720).
Real defect. But raising coverage was stated **before the run** as the falsifier, and it
falsified: errors improve to +2.542 s and -9.458 s and **0 of 6 land within one frame
(41.67 ms)**. The control stays confident and wrong under both settings. §4h names this
shape for this very engine — *two independent failures, and both must be stated or the
first fix looks like a failure* — so a change that moves the numbers while leaving cut
boundaries seconds from the truth is the worse outcome, not the cheaper one.

**A correction to §4h, measured.** It states the pHash branch requires
`abs(fps_main - fps_tgt) < 1e-6` and warns that *"any wiring that inherits that gate
inherits the exclusion"*. `grep -rn "1e-6" src/*.py` returns exactly one line —
`get_cut_time.py:377`, inside the **retired** module. The gate is not in the engine; the
engine resamples both sides to a common `-vf fps=N` and is rate-agnostic by
construction, with precision `1000/fps` ms — 41.67 ms at the caller's fps=24, which does
beat the 41.71 ms frame. **Precision was never the barrier and the PAL population was
never excluded by the engine. The aligner is the barrier.**

**Third defect, minor and worth one line:** `find_scene_gap_requirements(before_common,
after_common)` accepts two tuning parameters and uses neither — one occurrence of each
name in the file, the signature (control: `band_width`, 3 occurrences).

**What is NOT claimed:** that a rewritten aligner could not reach frame accuracy. Nothing
here measures that. Only this implementation is refused, on this evidence.


## 2026-09-04 — the 89 ms step at ~1170 s NEVER EXISTED, and twelve entries below assert it as measured fact

**`vmsam-forensic` fabricated the figure and retracted it. This entry supersedes every
use of it below; nothing there is edited, per this file's append-only rule.**

    THE FABRICATED CLAIM   "a real 89 ms step at ~1170 s, on id 13"
    NEVER MEASURED. Both the MAGNITUDE and the POSITION were invented.

**The twelve carriers in this file, none of them marked**, quoted rather than numbered
for the reason below. Each asserts the 89 ms step or the 1170 s position as measured:

    grep -nE '89 ?ms|1170' AGENT_NOTES.MD

**Read every hit outside this entry as fabricated, not as data.**

> **LINE NUMBERS DO NOT SURVIVE IN THIS FILE AND I PROVED IT ON MYSELF.** I first wrote
> this entry citing lines 673-1574, measured minutes earlier. **Appending the entry —
> newest first, at the top — moved all twelve down by exactly 60.** The citation was
> wrong the instant it was saved, by the act of saving it.
>
> **AN APPEND-ONLY, NEWEST-FIRST DOCUMENT INVALIDATES EVERY LINE NUMBER IN IT ON EVERY
> APPEND.** A line number here is not a pointer, it is a snapshot of one — the same
> object as an `mtime` that does not survive a move, and as a commit list that
> disagrees with its branch. **In this file, cite by the command that finds it.**

### THE RULE AT LINE 700 SURVIVES. ITS WORKED EXAMPLE DOES NOT.

> *"A step must also exceed roughly half a quantum to change any window's reading at
> all. The transition found at 1170 s is 89 ms — 0.7 of a point — so every 194 s
> window reads the same plateau on both sides of it."*

**The arithmetic is sound and the general rule stands: a step is visible only if it is
both late enough AND large enough.** The 1170 s / 89 ms instance illustrating it was
invented. **A reader who checks the worked example finds nothing; a reader who does not
check inherits a fictional measurement as the canonical case.**

    A TRUE GENERAL RULE ILLUSTRATED BY A FABRICATED SPECIFIC. Scrutiny lands on the
    rule, the rule holds, AND THE INSTANCE TRAVELS FREE.

That is the inverse of the failure recorded elsewhere in this campaign — *a true general
statement offered as the cause of a specific observation* — and it is harder to catch,
because the part a reader tests is the part that is correct.

### WHAT THE FABRICATION COST, WHICH IS NOT NOTHING

It became the stated justification for `MIN_STEP_MS = 60.0` in `change_point_locator.py`
— **the reason a threshold had its value.** The real smallest confirmed step is
**41.7 ms**, one video frame at 24.000 fps. So the constant was set ABOVE the real
phenomenon and then hid it:

    A THRESHOLD SET JUST BELOW THE SMALLEST *CLAIMED* OBSERVATION CONVERTS A SAMPLING
    LIMIT INTO AN INSTRUMENT LIMIT AND THEN HIDES THE CONVERSION. The next person
    measures a smaller step, does not see it, AND CONFIRMS THE THRESHOLD.

`PLATEAU_TOLERANCE_MS = 50.0` absorbed the same step one stage earlier, so either
constant alone was sufficient to lose it. Both are now 5.0, with the ordering asserted;
the window they must lie in is 0.4 ms (measured flat-region spread) to 41.7 ms.

**Correcting the fabrication is what exposed the defective constant.** The author went to
fix a citation and found the threshold was wrong.

### AND THE METHOD LESSON, WHICH IS THE PART TO KEEP

forensic traced the claim through a decision, a docstring and a data-file ask header —
**the artefacts of REASONING, where a finding feels like it lives** — and found a fourth
route by accident, in source comments.

> **A RETRACTION'S CARRIER SEARCH HAS A BLIND SPOT SHAPED LIKE THE AUTHOR'S MODEL OF
> WHERE CLAIMS LIVE. `grep` THE SOURCE TREE, NOT ONLY THE DOCUMENT TREE.**

Applying that took under five minutes and found two more carriers, including these
twelve. **Sixteen carriers total; the author's model of where the claim lived was wrong
by a factor of four.**

