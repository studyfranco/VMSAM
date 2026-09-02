# HANDOVER — Agent 1, C_cut and structural edits

Written 2026-09-02 at `92d05cc` on `agent/c-cut`, on the owner's directive to
stop and hand over. **Reading and writing only: no gates, no decode, no
container work.** The one exception I declare below.

**Its purpose is negative.** Someone restarting this project should be able to
read it and *not repeat what we did*. The false starts are the content; I have
not tidied them.

**Convention used throughout: MEASURED means I ran something and read a number.
INFERRED means I reasoned from records to a conclusion no instrument returned.**
Where a paragraph mixes the two it says so. This distinction is the single most
valuable habit this team developed, and it is the one thing I would insist the
restart inherit before any tool or any test.

**Privacy:** files are `err_<id>` only. No paths, no titles. **But see §2.1 — the
checker that enforces this is blind to a class of leak, and there is a live one
in the tree right now.**

---

# 1. WHAT IS TRUE

Each item: the claim, whether it is measured or inferred, my confidence, and
**what would overturn it**. The last field is the important one. Nothing here is
worth inheriting without the condition that would kill it.

## 1.1 The mission's 15 ms bar has never fired, and could not

**MEASURED.** `"Resolved"` appears 0 times in 315 rows; `"In sync, over the
15 ms bar"` appears 0 times. Neither is "no file qualifies" — **neither branch is
reachable**. `final_status` tests the bar only in its third branch, which
requires `sync == "in_sync"` **and** a Step 1 verdict of `in_sync` or
`out_of_sync`. `step1_verdicts.json` holds 40 entries; **all 40 are
`unverifiable`; 0 are `in_sync` or `out_of_sync`.** Every row therefore exits at
branch one (75 rows, no Step 1 record) or branch two (32 rows, Step 1 could not
measure).

**Confidence: certain.** It is a structural reading of the code plus a count of
the store, not a sample.

**This is invariant 158 in its purest form.** The bar is armed, correct, and has
never once been evaluated. Both wrong branches render; the right one does not.
An armed guard that has never fired is indistinguishable from a working one, and
this one had been quoted as a working one for the length of the campaign.

**What would overturn it:** a single `step1_verdicts` entry with an `in_sync` or
`out_of_sync` verdict. That is all it takes. **So the fix is not in
`final_status` — it is in whatever was supposed to write a decisive Step 1
verdict and never did.** A restart that "fixes" the branch order will have
silenced the alarm without repairing the thing it was alarming about.

**And there is something for it to catch.** MEASURED: 48 measurements over 31
distinct files fall in the 15–50 ms band, which `batch_verify.sync_verdict` calls
`in_sync` because its own bar is 50 ms. Probed against each master's own frame
duration, **31 of the 48, over 22 files, are above half a frame** — so the
quantisation excuse does not cover them. INFERRED from that: the corpus contains
real desync that no current instrument reports as desync.

## 1.2 What the bar could decide if it did fire — 5 undecidable rows

**MEASURED, 2026-09-02, and the last analysis I completed.** Full working in
`src/diagnostics/reports/what_the_15ms_bar_can_decide.md`.

Of 315 rows, **110 carry a surviving lag figure**; 205 do not. Applying the bar
by hand to the 110, using the code's own inclusive convention:

    decidable PASS    77
    decidable FAIL    28
    UNDECIDABLE        5

Over all 315, with the fourth category kept separate:

    decidable PASS                                        77
    decidable FAIL                                        28
    UNDECIDABLE                                           35   (5 at the bar, 30 merged but never measured)
    the question does not arise -- no artefact exists    175   (174 Failed, 1 Not processed)

**The undecidable count is small. Say so.** The team's working assumption was
that this corpus is riddled with unmeasurable rows; on the mission's own
question it is five.

**And all five are undecidable because of a boundary convention nobody ruled
on.** `MISSION_BAR_MS` is applied as `abs(v) > 15.0` — a fail — so the code
**passes** a row at exactly 15. Its own comment cites AGENT.MD as *"Step 1 under
15 ms"*, which is strict. MEASURED:

    convention                  pass  fail  undecidable
    <= 15 ms  (the code)          77    28      5    all at |v| = 20
    <  15 ms  (the wording)       68    33      9    all at |v| = 10

**The two undecidable sets are disjoint — no row is undecidable under both.**
Fourteen rows change status on `<` versus `<=`. **The undecidable count is not a
property of the corpus. It is a property of an unwritten convention.** A restart
should rule on `<` or `<=` in writing before it measures anything.

**What would overturn it:** a ruling that changes the convention (moves which
rows, not how many), or re-measuring the five with post-refinement code (decides
them outright).

## 1.3 The residue method: what it establishes, and its exact bound

**MEASURED.** The envelope correlation bins at `w = sr // 100` = 10 ms.
`refine_peak_index` (commit `6dbbc8c`, 2026-08-31T18:12:58+02:00) added sub-bin
interpolation. **So a lag value that is a NON-ZERO multiple of 10 was written by
code without refinement.**

Calibrated against a control of known-refined entries: **0 non-zero grid hits in
127 samples over 43 entries.** Rule of three, 95%:

    per sample   3/127 = 0.0236
    per ENTRY    3/43  = 0.0698   <- USE THIS ONE

**Use the per-entry bound.** Three windows of one file share a true lag; they
are not three independent draws, and treating them as independent overstates the
method threefold. That is the exact mistake this project kept making in other
places.

**A single non-zero grid value is 93.0% — short of 95%.** Derived minimum:
`log(0.05)/log(0.0698) = 1.125` → **n = 2 entries**, at both 95% and 99%. Below
n = 2 the method must report "cannot tell", and it must be allowed to.

**What would overturn it:** any second mechanism that produces exact multiples of
10 ms (a rounding step, a snap, a different hop size). I did not find one; I did
not prove there is none. **That gap is the method's real weakness and it is an
absence-of-evidence argument, which is the kind this project has been wrong
about before.**

## 1.4 `no_output_verdicts` is pre-refinement; `external_verdicts` is two eras

**MEASURED, applying 1.3.**

    store                entries   REFINED   PRE-REFINEMENT   UNCLASSIFIABLE
    no_output_verdicts       19        0            10              9
    external_verdicts        77       22            17             38

**`no_output_verdicts`: 10 qualifying, 0 contradicting → `0.0698^10 ≈ 3e-12`.
Pre-refinement, decisively.** Its distinct values are `{0, 10, 20, 30, 40, 1470,
2540}` and nothing else. Also MEASURED: the undated `batch_state` population has
14 qualifying entries → `6.5e-17`, likewise decisive.

**`external_verdicts` contains BOTH ERAS, and the method proves it.** 22 entries
must be refined, 17 must not be. **So the store's mtime tells a reader nothing
about any individual row in it.** That is the method's best result — it disproved
the assumption that a store has one era, which no metadata could have done.

**47 of 96 entries are all-exactly-zero and are unclassifiable at any sample
size**, because zero is identical under both instruments.

**What would overturn it:** as 1.3.

## 1.5 The ±5 ms bound, and what it does and does not reach

**MEASURED for the bin width; INFERRED for the interval.** The refined value
cannot be recovered — `refine_peak_index` interpolates from the correlation
array, and the array is not stored, only the resulting lag. **But a
pre-refinement value `v` does establish that the true lag lay in `v ± 5 ms`.**

    NOT recoverable   the refined figure
    IS establishable  a +/- 5 ms interval around the recorded one

**Practical consequence, MEASURED:** the interval straddles the 15 ms bar only
for `10 < |v| <= 20`. Outside that band both eras give the same verdict, so the
dating question does not arise at all.

**Which corrects a framing I gave the Lead and the Lead repeated back to me.**
The 47 all-zero entries were called the sharp case. They are the opposite:
MEASURED, 22 rows rest on an all-exactly-zero store entry and **all 22 are
decidable-PASS** under both eras and both conventions, because `0 ± 5` is
entirely below 15. **Unclassifiable is not undecidable.** They are permanently
undateable and permanently harmless to this question at the same time.

## 1.6 Zero of 110 lag figures can be dated to the code that produced them

**MEASURED, and this one survived a real challenge on 2026-09-02 that I expected
it to fail.**

`batch_state` records carry a `build` field, and **every one of its six distinct
shas resolves to a commit in this repository**, orderable against `6dbbc8c` by
ancestry. That looks exactly like the field I said did not exist.

**It is not, and the reason is the trap the field was created to catch.**
`build` is written from `deployed_build()`, which reads the **container's**
`/health` sha. The lag beside it is produced by `sync_verdict`, which is
**host-side code in `batch_verify.py`**. **The field dates the merge that made
the artefact, not the code that measured it** — and the container was behind
HEAD for the entire relevant window.

**So it dates the wrong thing precisely, which is worse than dating nothing
vaguely.** A reader who ordered lags by `build` would get a confident answer to a
question they did not ask, with no field anywhere able to contradict them.

**Confidence: high.** It is a reading of where each value is assigned, not a
sample.

**What would overturn it:** finding that `sync_verdict` runs container-side for
some population, or a host-side stamp I did not find. I grepped for one; I did
not exhaust the tree.

**And note what this costs at the bar: nothing.** MEASURED — four of the five
undecidable rows are already classified pre-refinement, so their undecidability
is the 10 ms bin and not an unknown era. Only `err_45` is undateable, and the
residue method says its population is pre-refinement at `6.5e-17`. **A commit
stamp would rescue none of the five. Only a re-measurement decides them.** That
narrows what I told the Lead an hour earlier: the missing field is worth having,
and it is not what stands between us and these rows.

## 1.7 `verify_unverified.py` abandons a verifiable pair on 7 of 28

**MEASURED by ffprobe stream counts only. No decode, no correlation.**

`verify_unverified.py:38` picks `shared[0]` from a fixed order `("fre","eng",
"jpn")` and, if that one language does not qualify, records `unverifiable` for
the whole file — **without looking at the other shared languages it already
computed.**

    3 shared languages   19
    2 shared languages    9
    0 shared languages    1
    exactly 1 shared      0    <- not one file

**No probed file has exactly one shared language, so `shared[0]` discards
information on every single file it runs on.** The split:

    a shared language QUALIFIES (output has MORE tracks than the master)    7
    no shared language qualifies -- genuinely unverifiable                 21
    no shared language at all                                              1   err_13
    artefact missing at the path I probed                                  11

**21 of 28 are right for the right reason.** The seven that are wrong —
`err_83, 87, 91, 95, 98` and two others — all share the shape `fre 1/1, eng 2/1`:
**`fre` is picked first and never qualifies; `eng` would have.**

**Confidence: high on the seven, because the qualifying test is a stream count I
read directly.** **INFERRED, and NOT verified: that comparing the `eng` pair
would have produced a usable verdict.** It would have produced a *comparison*.
Whether it correlates is unmeasured, and the box came down before I could.

**What would overturn it:** the `eng` pairs failing to correlate, which would
make the seven unverifiable for a different reason and the defect cosmetic.

## 1.8 err_78's precedence is incidental

**MEASURED.** And it opens with a correction of my own: **I told the Lead that
`external_verdicts` overrides err_78's lag. It does not — err_78 has no
`external_verdicts` entry at all.** The override is `no_output_verdicts`, which
is consulted **last** and therefore wins over everything.

    LEDGER (batch_state)   eng  16.7 ms  output-stream vs output-stream
                                samples -8.6 / -9.2 / -16.7  peaks .993 .987 .988
                           fre   0.6 ms  output vs master file
    no_output_verdicts     fre   0.0 ms  output vs master file
                                samples 0.0 / 0.0 / 0.0      peaks .979 .968 .978

**The override differs from the ledger in four ways at once**: different
language, weaker comparison path, four days older, and coarser (all 57 of that
store's sample lags are multiples of 10, across 7 distinct values).

**The ledger had already measured `fre` at 0.6 ms by the same fallback path and
still reported 16.7, because it took its own worst.** The override replaces a
worst-case with a best-case and the row cannot show that it did.

**Applying 1.5 makes it sharper:** the overriding `0.0` means "within ±5 ms", and
the ledger's `16.7` is **outside that interval** — so the two are irreconcilable
as one measurement before the language difference is even considered.

**Confidence: certain that the precedence produces this. INFERRED: that the
ledger's figure is the better one.** It is on the stronger path and it is newer;
neither is proof.

## 1.9 The chimeric same-dub rule, and why it was safe to ship

**MEASURED, and this is the strongest empirical result in my territory.** Mean
fpcalc fidelity over 3 × 30 s cuts on the longest plateau, bar 0.90 — the
instrument and bar `find_differences_and_keep_best_audio` already used. Over
**217 real stream pairs across 15 master-patched builds**:

    same recording       n=32    0.904 - 0.993
    different recording  n=185   0.563 - 0.847
    empty band                   0.847 - 0.904

**The band is empty — the separation is not a chosen threshold, it is a gap in
the data.** And **9 pairs carry the SAME language tag and are NOT the same
recording** (0.606–0.811), which is why a language tag cannot be the gate.

**Exposure MEASURED 2026-08-30:** 15 of 29 produced chimeras draw a gap patch
from the master; **12 of those patch a track with the master's audio in a
DIFFERENT LANGUAGE**. **Container-side exposure was 0** — the chimeric path never
fired in any deployed merge, confirmed by correlating a shipped output whose
`spa` track matches the raw candidate at 0.95 with constant lag.

**What would overturn it:** a same-language different-recording pair scoring
above 0.904, which would close the empty band. n=185 on the low side makes that
unlikely but not impossible.

---

# 2. WHAT IS WRONG AND STILL WRONG

Ordered by whether a restart is load-bearing on it.

## 2.1 LOAD-BEARING — the privacy checker is blind to a bare title, and there is a live leak

**MEASURED, 2026-09-02, and found by accident while reading for §2.3.**

`check_no_private_refs.py` reports **"clean: no media library references in 950
tracked files"** while a media title sits in a source comment in
`verify_unverified.py` around line 48, inside the note about a pair scored
`OUT_OF_SYNC` on a single weak window.

**Cause, read from the checker:** it matches (a) library path prefixes and
(b) `MEDIA_NAME`, a regex that requires a **media extension** (`mkv|mka|mp4|...`).
**A bare title with no extension cannot match either.** So the instrument tests
"no path, and no filename-with-extension" while its output line claims "no media
library references" — a strictly stronger claim than it evaluates.

**This is invariant 157 arriving through a gate.** The checker is two-valued
where the subject has three states: *leak found*, *no leak*, and **/*no leak of
the kind I can see*/**. It has passed every commit this project made.

**For the restart: this is load-bearing and it is the first thing I would fix.**
Every privacy assurance the campaign gave — mine included, in every report header
— rests on an instrument that was never tested against a title. **I did not fix
it: `verify_unverified.py` is not a file I own (AGENT.MD rule 4), and I was
directed to write only.** It is a one-line deletion in the comment plus a
title-aware rule in the checker.

## 2.2 LOAD-BEARING — the lag precedence at `master_summary.py:259-264`

    step1_ms, dissent = worst_lag(record)
    for source in (ext, no_output):
        if isinstance(source.get(path), dict):
            got, got_dissent = worst_lag(source[path].get("result") or {})
            if got is not None:
                step1_ms, dissent = got, got_dissent

**Last writer wins, and the order is `batch_state` → `external` → `no_output`.**
So the store proven coarsest and oldest (§1.4) overrides the two better ones,
silently, on every row where it has an entry. **MEASURED: it is the source of the
surviving lag on 18 rows.** §1.8 is one worked example.

**Why it is load-bearing:** it is not a bug in a number, it is a bug in an
*authority order*, and the module's docstring states a different order from the
one the code implements ("Sources, in order of authority" lists `external`,
`batch_state`, `step1` — the loop does not match it). **A restart that copies
this file inherits a precedence nobody chose.**

**Not fixed** because I was told not to edit `master_summary.py`, correctly — it
is the file every count in the project is read from, and changing it mid-campaign
would have made every prior number unreproducible.

## 2.3 LOAD-BEARING — `verify_unverified.py:38`, `shared[0]`

§1.7. The one-line shape of the fix is to iterate the shared languages and keep
the first that qualifies, rather than the first that exists. **Held for an owner
ruling and never given one.** A restart should not reuse this file as-is.

## 2.4 NOT load-bearing — `chimeric.probe` has no third state

The probe returns a two-valued answer where the subject has three: *patched*,
*not patched*, and *the path never ran*. **Container-side exposure is 0 (§1.9),
so nothing has been mis-reported yet.** It becomes load-bearing the moment the
chimeric path first fires in a deployed merge, and not before. **Fix it before
re-opening that path, not after.**

## 2.5 NOT load-bearing today — K2, K3, K6 and K7 in `keyframe_align`

I fixed the leaf and the refusals in `keyframe_align.py`: `keyframe_times` now
returns `(times, reason)` with three outcomes rather than two, `segment_offset`
returns four named refusals instead of a bare `None`, and
`step_at_change_point` builds its message from both sides instead of a constant.

**K2, K3, K6 and K7 remain unfixed and are marked as such in the file.** They are
not load-bearing because keyframe alignment is not on the merge path that
produced any shipped output. **They become load-bearing if a restart makes
keyframe alignment a primary instrument**, which was under discussion and never
decided.

## 2.6 Cosmetic but worth one line

`describe_chimeric_track` always emits "re-timed N pieces", so an **untagged**
stream means the assembler never touched it. **The acceptance criterion "every
audio stream carries a CHIMERIC tag" is mis-stated** and should read "every
*altered* stream". MEASURED on err_120: 6 of 7 tagged, the untagged one being the
master's own `flac` track, which is correct behaviour failing a wrong criterion.

---

# 3. WHAT WE MISUNDERSTOOD, AND FOR HOW LONG

**This is the section that matters.** Each entry: what was believed, what was
true, how long it stood, and what would have caught it sooner. **The ones where
the Lead or I was the source of the error are included with attribution, because
an error that travelled is worse than an error that sat still.**

## 3.1 The `jpn` default that was in a different file — stood ~4 days

**Believed:** err_678's `external_verdicts` note ("the only `jpn` track is
bit-identical") described a property of err_678.

**True:** `reverify_external.py` read the language from a regex over the entry's
`note` field and fell back to a **hardcoded `jpn`** when the regex missed.
err_678's ledger row **has no top-level `note` field at all**, so the regex found
nothing and the tool judged a `jpn` track on a file whose correlation language is
not `jpn`.

**Blast radius, MEASURED:** of 86 `external_verdicts` entries, **72 took the
hardcoded default**; of those, **24 are files whose `lang_pick` is not `jpn`**
(eng 11, fre 10, chi 2, none 1). **28% of the store was judging the wrong
language and the store cannot say which rows.**

**And the nine C1 nulls are exactly this.** I first reported that
`external_verdicts` "carries no fault at all" for the C1 nine. **True but
incomplete** — the null result *is* the fault: the hardcoded `jpn` looked for a
`jpn` track on `eng` files and found nothing, and "found nothing" was recorded as
a verdict about the file rather than about the query.

**What would have caught it sooner:** the store recording **which language it
actually used and where that came from**. It now does — `select_language` returns
`(lang, source)` and the record carries `lang_source` as one of `note`,
`lang_pick`, `none`. **A field that names its own provenance is the cheapest
defence this project found.**

## 3.2 err_76: "unsupported" amplified into "false" — stood hours, cost 13 files

**Believed** (mine, then the Lead's, then the owner's): the ledger note on err_76
is FALSE.

**True:** the analysis in `deep/err_76.md` compares **candidate against master**
and never touches the merged output. So the note is **unsupported by that
evidence — not shown false.** Those are different claims and only one of them
licenses a re-run.

**How it travelled:** I wrote "FALSE". The Lead amplified it verbally to the
owner and **dispatched thirteen files on it**. I corrected it in place, at the
line where the wrong value could still be lifted (invariant 155); the Lead
corrected it with the owner.

**What would have caught it sooner:** asking, before writing the stronger word,
*what would this evidence have to show for the note to be false, and does it show
that?* **It would have taken one sentence.** The three-state discipline of §2.1
is the same lesson in a different costume: *contradicted*, *supported*, and
*not addressed by this evidence* are three states, and I collapsed them to two.

**This is the single most expensive error in my territory and it was a word
choice.**

## 3.3 The C2 thirteen — and the Lead's "12 + 1" is also not right

**Believed** (the Lead's framing in the handover dispatch): the C2 thirteen were
12 bit-identity and 1 count-based.

**True, MEASURED from the per-language notes:** **13 carry a bit-identity note**,
and **err_691 additionally carries a count-based note on its `eng` leg only** —
its primary language is bit-identity too. So it is 13 and 1 **overlapping**, not
12 and 1 disjoint. **The count-based population is a different set entirely:**
err_6, 7, 10, 11, 14–19, err_76, and err_691's `eng`.

**Why the distinction earns its keep — and this reverses my own earlier claim
that it does not.** I argued the measurement-vs-inference split does not hold
here, because bit-identity is *also* an inference: the guard never compares
bytes, it fires on `all(abs(l) < 1e-9 and p >= 0.999 ...)`, and its docstring
reasons in the wrong direction (identical samples imply `(0, 1.0)`; `(0, 1.0)`
does not imply identical samples). **That part stands.** What distinguishes them
is the **failure mode**:

    bit-identity note wrong about provenance -> the CONCLUSION SURVIVES
                                                (the track carries the master's audio whatever file it came from)
    count-based note wrong                   -> the CONCLUSION IS FALSE
                                                (the output does carry independent audio)

**How long the wrong split stood:** the "12 + 1" phrasing was live in the Lead's
dispatch on the day of the shutdown. It had not yet been acted on.

## 3.4 The eleven "artefact gone" files — UNRECONCILED, and I am leaving it that way

**I reported 11 files as "artefact missing, not measurable in either direction"**
(§1.7). **The Lead reports that all eleven were on disk.**

**I did not reconcile this and cannot now** — the box is coming down and I was
directed to stop. **MEASURED on my side: `os.path.isfile` returned False at the
path I probed.** **INFERRED, untested: the probable cause is which tree I probed
— the mirrored output tree versus the library tree, which is exactly the
distinction `mirrored()` exists to manage and exactly the kind of thing that
looks like absence.**

**I am recording it as an open discrepancy rather than resolving it in the
report, because resolving it in prose is how §3.2 happened.** A restart should
treat the "11 missing" figure as unverified in both directions.

**What would have caught it sooner:** the probe naming *which tree* it looked in,
in its own output. It did not.

## 3.5 My "87 of 199 on the grid" — contaminated denominator, stood ~1 day

**Believed:** `batch_state` shows 87 of 199 lags on the 10 ms grid, a ~44% rate,
suggesting the store is a noisy mixture.

**True:** the figure mixed two eras in one denominator. Partitioned by
`recorded_utc` against `6dbbc8c`:

    AFTER the commit -- KNOWN REFINED    n=127 samples    on grid  15   0.118
    undated (older era)                  n= 72 samples    on grid  72   1.000

**And all 15 of the refined store's grid hits are exactly 0.0. Non-zero grid hits
under refinement: ZERO in 127.**

**So the discriminating feature is not "a multiple of ten" but "a NON-ZERO
multiple of ten"** — an exact zero is zero at every resolution and carries no
information. **The correction did not weaken the instrument, it created it:** the
raw rate looked like 44% and the real one is 0%.

**What would have caught it sooner:** partitioning before counting, always. **A
rate over a population known to contain two mechanisms measures neither.** This
is the same error as §3.1's blast radius and as the "ALL folder 86" below.

## 3.6 "ALL folder 86" — falsified within the day

**Believed** (mine): a defect applied to all of folder 86.

**True, MEASURED:** 3 folders, 3 languages, 3 builds. The population was not the
folder.

**Related and worse:** I attributed ten folder-86 files to
`retest_ids.GROUPS["multitrack"]` **by id-list membership**. That is the
id-signature trap: the same ids appear in several lists for unrelated reasons.
**The operative cause was the ledger's own note**, which said so directly.

**What would have caught it:** reading the record that states the cause instead
of matching an id against a list that happens to contain it. **Membership is not
causation, and an id list is the easiest false friend in this project.**

## 3.7 The `updated_utc` field — two failure modes, never once correct, ~2 days + 14 commits

**Believed:** the agent state files carried real timestamps.

**True, MEASURED against each carrying commit:**

    stamp                  commit time (UTC)     stamp - commit
    2026-09-02T08:10:00Z   2026-09-02T06:01:47Z       +128 min
    2026-09-02T09:40:00Z   2026-09-02T05:54:39Z       +225 min
    2026-09-02T08:35:00Z   2026-09-02T05:16:02Z       +199 min
    2026-09-02T07:40:00Z   2026-09-02T05:06:50Z       +153 min
    2026-09-02T06:55:00Z   2026-09-02T04:07:07Z       +168 min
    2026-09-02T06:20:00Z   2026-09-02T04:01:33Z       +138 min

**The spread is 97 minutes, so it is not a timezone bug** (the box is +0200;
local-time-labelled-Z would be +120 every time). **The tell is in the values: all
seven end `:00Z` on a five-minute boundary.** A clock read seven times does not
do that. **Before that the same field was FROZEN** — one value carried unchanged
through fourteen commits over two days.

**Why it mattered to someone else:** a stale stamp reads as "idle"; a future
stamp reads as "recently active" for as long as it is in the future. **The Lead's
roster rendered the future case as `?` — the same glyph it used for a file it
could not read — so an untrustworthy clock and a missing reading looked identical
from outside.** Invariant 157 again, arriving through a timestamp.

**The fix, and the general principle: `agent_state.write()` takes no timestamp
argument.** There is nothing to pass wrongly. **Removing the parameter beat
validating the parameter**, and I would apply that shape wherever it fits.

## 3.8 A mutation SURVIVED — the tests were testing the producer, not the record

**Believed:** 13 green tests covered `reverify_external`'s written record.

**True, MEASURED:** deleting `lang_source` from the written dict left **all 13
green**. The tests exercised the function that computes the value and never
asserted the value reached the file.

**Fix:** 4 AST-based tests that assert on the literal written dict. **The lesson,
and I would put it in the invariants: a test of the producer is not a test of the
record.**

**What would have caught it sooner:** exactly what did — deliberately breaking
the code and checking the suite notices. **We did that once, late, and it found
something immediately.** See §6.

## 3.9 Smaller ones, recorded because the pattern repeats

- **I called err_316's guard "the copy guard".** It is the **self-comparison
  (bit-identity)** guard. Two different guards, adjacent code, one name.
- **`languages_inconclusive` reads 46; 12 survive.** MEASURED. The count was
  four times the real one because rows were counted that had a named reason
  elsewhere. **A count is not a finding until you have read what it counted.**
- **`cut_points_refined.json` was on no path the fusion worker reads**, and
  `VMSAM_CUT_POINTS_TABLE` could not fix it — the hardcoded `cut_points_c.json`
  sibling is merged last and overwrote all 39 refined records. **Refinement had
  been "done" and was reaching nothing.** Same shape as §2.2: a load order
  nobody chose, silently winning.
- **Repeated bash CWD drift** after `cd` into a test directory produced results
  attributed to the wrong tree more than once. Fixed by prefixing every command
  with the absolute worktree path. **Trivial, and it corrupted real measurements.**

## 3.10 The pattern behind almost all of the above

**Every entry in this section is one of three shapes**, and a restart that learns
only this much has taken the useful part:

1. **A two-valued instrument over a three-valued subject** (§2.1, §2.4, §3.1,
   §3.2, §3.7). Absence and negative result rendered identically.
2. **A rate or a count over a population containing two mechanisms** (§3.5,
   §3.6, §3.9's 46-vs-12). It measures neither.
3. **An authority order nobody chose, winning silently** (§2.2, §3.9's cut
   points). Last writer wins is a decision even when it was an accident.

---

# 4. UNKNOWN / UNRESOLVED

Recorded as open. **None of these were investigated after the stop order.**

## 4.1 err_112's decode — the C2 near miss

**MEASURED:** peaks 0.999 / 0.999 / 1.000 at lags 0.2 / 0.3 / 0.0. **Its third
window meets the self-comparison guard's condition while the other two do not**,
so the signature occurs outside the guard. Strong, not certain.

**What would settle it: a byte or decoded-sample comparison of the two tracks. A
decode. Never run.**

**And a caution for whoever picks it up.** On 2026-09-02 a `reverify_in_place`
run on err_112 completed with exit 0 — `jpn` attempted and passed, `eng` not
measurable. **That is not the measurement err_112 was waiting on.** It re-derived
a language verdict from the same correlation instrument, so it cannot distinguish
*the output carries the master's own audio* from *it carries independent audio
that correlates well*. **A passing test on the wrong instrument looks exactly
like an answer.**

## 4.2 `(fre, shared_reference)` — content or timing? OPEN, NOT INVESTIGATED

Raised by Agent 3 in my state file on the day of the stop. **Recorded verbatim as
open on the owner's instruction; I did not investigate it.**

**MEASURED and on record for err_120:** only the `fre` track is `same_dub`
(candidate `fre` patched from master `fre`). **The other five — ger, eng, spa,
ita, por — are `shared_reference` from the master's `fre` track**, which was
documented as the safe fallback because `fre` is the only recording both releases
demonstrably share.

**The open question:** does `shared_reference` mean the patch borrowed the `fre`
track's *timing*, or its *content*? **If content, then five language tracks carry
French audio across the patched span — reported as ~51.5 s — which would violate
criterion 4's VO rule.** The fallback was documented and reviewed as a timing
reference. **Nobody has confirmed which it is in the produced artefact.**

**This is the highest-value open question in my territory.** It is answerable
from one artefact with one decode.

## 4.3 The C1 / C2 / C3 ledger reconciliation

Held for the owner and never ruled on. **The blocking fact, MEASURED: 80 files
have been re-verified in place against the artefact on disk and 42 carry a
different verdict — more than half of everything comparable.** A single
reconciled number would repeat the original error with fresher data, which is why
`master_summary` prints both with the evidence class beside each.

## 4.4 Question 1 — should host-side tools stamp their commit?

Raised, never answered. **§1.6 narrows it: the stamp is worth having and it would
not have decided any of the five undecidable rows.** So it is a hygiene question,
not a blocker.

## 4.5 The 18 merged rows with no lag and no reason on file

**MEASURED:** 30 rows merged and produced an artefact yet carry no lag figure. Of
these, **12 record "the only track is a copy of the master's"** — a structural
refusal, correct, there is nothing independent to compare. **The other 18 record
nothing at all.** An artefact exists, nothing measured it, no reason is on file.
**I never found out why, and no residue method reaches them.**

## 4.6 Whether a second mechanism produces exact 10 ms multiples

§1.3. **I looked and did not find one. I did not prove there is none.** The whole
residue method rests on this and it rests on an absence.

---

# 5. WHAT TO DISCARD

Blunt, as asked.

## 5.1 DELETE `step1_verdicts.json` rather than trust it

**MEASURED: 40 entries, 40 `unverifiable`, 0 decisive.** It is consulted as the
third authority for a sync verdict and **has never once supplied one**. Worse, it
is the store whose emptiness makes the mission bar unreachable (§1.1) — so its
presence in the precedence chain reads as coverage while contributing nothing.
**A restart should either delete it or make it the thing it claims to be. Do not
inherit it as-is.**

## 5.2 DO NOT BUILD ON THE RESIDUE METHOD

**I recommended this before the shutdown and the Lead agreed. It stands.** It
survives its own calibration — clean control, honest bound, derived minimum n,
and it produced a finding no metadata could (`external_verdicts` is two eras).
**And it should still not be the answer:**

- **It dates a code ERA, not a tool.** It cannot say which tool, which run, or
  which host.
- **Just under half the entries are unclassifiable in principle**, and not a
  random half — they are the ones reporting zero, the value most likely to be
  relied on as a pass.
- **It is an inference standing in for a field.** Its value is as **evidence FOR
  the missing field, not a substitute for it.**

**Keep the report as a worked example of calibrating an instrument. Do not put
it in a pipeline.**

## 5.3 Discard `no_output_verdicts` as an authority

**MEASURED (§1.4, §1.8):** it is the coarsest store, proven pre-refinement, four
days stale on the rows I checked, and it **wins the precedence** (§2.2). Either
remove it from the chain or move it last in *authority* rather than last in
*assignment order*. **A restart that copies the loop copies the defect.**

## 5.4 Discard `external_verdicts` as a single store

It is two eras (§1.4). **Any statistic over it as a whole measures nothing.** If
it is inherited, it must be split first, and 47 of its entries cannot be split.

## 5.5 Discard id-list membership as evidence of cause

§3.6. **`retest_ids.GROUPS` are convenience lists. An id being in one says
nothing about why it is there.** Read the record.

## 5.6 Discard the inline pipeline gate

`check_no_private_refs.py --staged | tail -1 && git commit` **is not a gate** — a
pipeline's exit status is `tail`'s, so the `&&` always fires. **The Lead and one
agent each landed a commit the checker had rejected this way.** Use
`src/tools_batch/gate.sh`. **And see §2.1: even the real gate is weaker than its
output line claims.**

## 5.7 Discard "measurement at scale by fan-out"

Recorded in AGENT.MD and worth repeating: a 12-shard agent fan-out drove load to
163 on 17 cores, and a 6-shard one exhausted a session budget and returned
nothing. **Measurement at scale is a script's job. Spend agents on
interpretation.**

---

# 6. WHAT TO KEEP

## 6.1 `test_65_fire_the_mission_bar.py` — 11 tests

**Keep it and run it first.** It is the test that proves §1.1: it fires the
mission bar deliberately and asserts on which branch is reached. **A test that
proves a guard CAN fire is worth more than any number of tests that pass while it
never does.** Every project accumulates armed-and-never-fired guards; this is the
shape of test that finds them.

Also keep `test_50_agent_state.py` (9) and
`test_64_reverify_external_language.py` (17, including the 4 AST-based ones).

## 6.2 The mutation check — deliberately break it and see if the suite notices

**It found a real hole on its first use** (§3.8). **Cheap, and it is the only
thing that distinguishes a test suite from a test suite's reputation.** Apply it
to any test you are about to rely on, especially one that guards a written
record.

## 6.3 The capacity check (invariant 141)

Before believing a count, ask what the instrument *could* have returned. Half the
errors in §3 die instantly under it. **A count of zero from an instrument that
cannot return non-zero is not a measurement.**

## 6.4 Invariants 155, 157, 158, 159 — earned, all four

- **155 — a correction must sit where the error is quoted from.** The test is
  *can the wrong value still be lifted?* Correcting a conclusion in a summary
  while the wrong number stays in the table is not a correction.
- **157 — a two-valued instrument cannot distinguish an absence from a result.**
  §2.1, §2.4, §3.1, §3.7. **The most productive single line in the project.**
- **158 — an armed guard that has never fired is indistinguishable from a
  working one.** §1.1.
- **159 — a taxonomy complete over the subject has no cell for the instrument.**
  Every "unverifiable" bucket needs to say whether the *file* or the *tool* was
  the obstacle.

**Add one from this handover:** *a test of the producer is not a test of the
record* (§3.8).

## 6.5 A field that names its own provenance

`lang_source` ∈ `{note, lang_pick, none}` (§3.1), and `verdict: "not_attempted"`
as a **named** refusal rather than a null. **The cheapest defence found in this
project, by a distance.** Any value that could have come from more than one place
should carry where it came from. `build` (§1.6) is the counter-example that
proves the rule: it names a provenance, just not the one the reader needs.

## 6.6 `agent_state.write()` taking no timestamp

§3.7. **Remove the parameter rather than validate it.** Also keep
`is_suspiciously_round()` — reported, never enforced, because a genuine
`12:35:00Z` exists.

## 6.7 The reporting discipline itself

Numbers rather than adjectives; state what you are about to run before running
it; **say plainly when a measurement contradicts what you were told**; and a
negative result is a complete result — measure it, write it down, close the
question.

**The owner's observation, which I can now confirm from my own record: seven
defects in this project were found by one agent attacking another's assumptions,
and none by an agent agreeing with its brief.** §3.2, §3.3 and §3.4 in this
document are all cases where the Lead and I agreed with each other and were
jointly wrong. **§1.6 is the one where I expected my own claim to fall and
checked anyway, and it held.** Both outcomes came from the same habit.

---

# CLOSING — the state I am leaving

- **Branch `agent/c-cut` at `92d05cc` plus this commit. Not pushed.**
- **Tree otherwise clean.** No edits to `master_summary.py`, none to the
  protected closure, none to any file outside my ownership.
- **Two reports written today and committed with this handover:**
  `src/diagnostics/reports/what_the_15ms_bar_can_decide.md` (§1.2, finished
  before the stop order) and this file.
- **Gates: NOT RUN, on the Lead's explicit instruction, because the box is coming
  down.** The suite last ran green at 2283 tests with no skips; the closure last
  verified at 78 members; the tree has not changed in a way that could affect
  either. **The privacy check I did run, because it is an owner hard rule and it
  is a local read — it reports clean, and §2.1 explains why that word is worth
  less than it looks.**
- **One live privacy leak is in the tree and I did not fix it** (§2.1). It is in
  a file I do not own and I was directed to write only. **It should be the first
  commit of the restart.**
