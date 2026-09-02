# If field 3 is refused: what marking the reconstructions "unverifiable" would cost

**Read at `603aa8d`** (invariant 154). Written by Agent 4 (forensics) to cost my
own second question in `PROVENANCE_PROPOSAL.MD` — *if a merge-time provenance
field is refused, should the downstream reconstructions be marked unverifiable in
the reports that use them?* **This is a costing, not an edit. No report, no
generator and no wording was changed. Nothing was re-run and no decode was
spent.** The protected closure is intact: 78 members, 0 differing.

## The answer in three numbers, because it is three different decisions

| Decision | Cells the owner would see change |
| :--- | ---: |
| Mark the **order**-based claims undetermined | **2** |
| Mark the **evidence-contradicted** claim undetermined | **1** |
| Mark **every** verdict resting on a reconstruction undetermined | **109** |

**And the count-based claim — the one this discussion started from — changes 0
cells in the owner's table.** It changes 51 reason strings inside stores.

## The claim inventory: four sites, and they are not comparable in size

The Master Summary Table in `AGENT_NOTES.MD` has **315 rows**. **109 of them
carry a Final Status that is a Step 2 statement** — 75 *"Step 2 verified; no Step
1 record"*, 32 *"Step 2 only; Step 1 could not measure"*, 1 *"Out of sync"*,
1 `MERGED_OUT_OF_SYNC`. Every one of those rests on a provenance reconstruction
or on a footing the row does not record.

Across every verdict store, by what actually carried the record:

    in_sync       BYTE-identity copy guard      121
    in_sync       footing NOT RECORDED           61
    in_sync       ORDER                           1
    unverifiable  footing NOT RECORDED           66
    unverifiable  COUNT                          51
    unverifiable  BYTE-identity copy guard       37
    out_of_sync   ORDER / BYTE / not recorded      3

(Records, not files: a file-level row and its per-language rows describe one
file. **96 distinct files carry an `in_sync` verdict.**)

### Site 1 — ORDER. `batch_verify.sync_verdict`, the two-track branch

**What it says.** Nothing in prose. It picks `a = streams[0]` and
`b = streams[n_master]` and returns a number. The claim is *that pair is
cross-provenance*, and it is carried by the verdict itself.

**Footing.** Emission order. Invariant 86: once the assembler substitutes,
**ordering tells you where a track sits, not where it came from.**

**Where it surfaces.** `compared: "output stream vs output stream"` appears on
**exactly two records in the whole corpus** — **err_78** (`in_sync`, rendering as
*"Step 2 verified; no Step 1 record"*) and **err_293** (`out_of_sync`, rendering
as `MERGED_OUT_OF_SYNC`).

    has:    in_sync, 0 ms
    honest: in_sync between two output streams chosen by emission order;
            whether they came from different sources is not recorded

**Cost: 2 cells.** This is the reconstruction the documents — including my own
proposal — have been written about, and it is the smallest of the four by two
orders of magnitude.

### Site 2 — COUNT. `sync_verdict`'s note, and `verify_unverified`'s

**What it says**, in two wordings:

    all 2 'jpn' tracks are master-derived (master has 2); candidate contributed none
    candidate contributed no track in this language; both compared tracks are master-derived

**Footing.** `len(streams) <= n_master`.

**Where it surfaces.** 12 records in `batch_state.per_language`, 39 in
`step1_verdicts` — **51 in total, every one of them under `unverifiable`, and not
one under an `in_sync`.** That absence is the important part and it is
informative, not empty: **this claim can only ever withhold a measurement. It
cannot produce a false pass.**

**And it reaches the owner's table on zero rows.** `master_summary.py:65` carries
a phrase written for exactly this note — *"no candidate-derived track to compare
against"* — and it **fires on nothing**, because the claim lives in a nested
`per_language` record and in a store the detail column does not read. A guard
written for a case, that has never seen it (invariant 158).

    has:    unverifiable -- candidate contributed no track in this language
    honest: unverifiable -- the counts are consistent with the candidate having
            contributed none, which this store cannot distinguish from the merge
            keeping the candidate's track and dropping the master's

**Cost: 0 cells in the owner's table, 51 reason strings in the stores.**

**A count I have not reconciled, stated rather than smoothed over.** Agent 2
reported this claim appearing **66** times and never under an `in_sync` verdict.
**I measure 51** across the two stores that carry it — 12 in
`batch_state.per_language`, 39 in `step1_verdicts`. **The direction is identical
and independently reached; the magnitude differs by 15** and I cannot account for
it from here — a different snapshot, or a wider scan than the two stores I read.
Two counts of one phenomenon are reconciled by naming the difference, not by
picking one (invariant 88), and the difference is not named yet.

### Site 3 — BYTE-IDENTITY. `verify_against_master`'s copy guard

**What it says.** *"the only 'jpn' track is bit-identical to the master's; this
compares the master against itself"* → refuse. Otherwise: compare the output
track against the master **file** and return a verdict.

**Footing.** Byte comparison, **in a pipeline that re-encodes and applies
`delays[language]`.**

**Where it surfaces.** **This is the dominant footing in the corpus** — 121
`in_sync` records and 37 `unverifiable`. It reaches the table on 13 rows as *"the
only track is a copy of the master's"*, and it carries the great majority of the
109 Step 2 statuses.

    has:    in_sync, worst 0.0 ms
    honest: in_sync between an output track and the master file; whether that
            output track is candidate-derived is not recorded

**Cost: up to 109 cells** — and this, not the ordering premise, is where the
entire price of the owner's decision sits.

### Site 4 — CONTENT. `provenance_classify.py`

Not a reconstruction from metadata: it decides by decoded samples, then by
correlation, and it **reports `indeterminate` rather than guessing** when a codec
mismatch makes the byte comparison meaningless. It is the instrument that can
audit the other three, and its stored results are the basis of the next section.

## The soundness is one-directional, and that decides the whole question

Every one of the three reconstructions is sound in one direction and unsound in
the other:

| Reconstruction | Sound direction | Unsound direction | What the unsound direction does |
| :--- | :--- | :--- | :--- |
| COUNT `out > n_master` | a candidate-derived track **exists** — pigeonhole, no ordering assumption | — | permits a measurement, correctly |
| COUNT `out <= n_master` | — | *"the candidate contributed none"* | **withholds** — cannot produce a false pass |
| ORDER `streams[n_master]` | — | *"this stream is the candidate's"* | **asserts** a number that may be master-against-itself |
| BYTE bit-identical | *"this is the master's copy"* | — | **withholds**, correctly |
| BYTE differs | — | *"not the master's copy"* | **asserts** a number that may be master-against-itself |

**Existence is sound; identification is not.** The pigeonhole proves a
candidate-derived track is in there somewhere and says nothing about **which**
stream it is — so a file can be correctly judged verifiable and still be measured
on the wrong pair.

**Only two of the five rows can produce a false pass**, and they are the two that
assert rather than withhold. That is the set worth downgrading, and it is not the
set the discussion has been about.

## Cross-checked against the content instrument — and my first cut of this was wrong

`provenance_classification.json` has an opinion on 58 of the 96 files carrying an
`in_sync` verdict. **My first pass read `judgeable: false` as "no candidate-derived
track" and reported 50 contradicted verdicts.** That is wrong, and **err_95 is the
file that caught it**: its `eng` has 1 master track against 2 in the output, one
resolved as `master_derived_shifted` and **one `indeterminate`** — the classifier
did not refute the pigeonhole, it **could not decide**. `judgeable: false` folds
*"there is no candidate side"* together with *"I could not tell"*, and reading it
two-valued is invariant 157 committed by the auditor. **Checking the leaves
instead of the flag:**

    candidate side CONFIRMED, artefact unchanged since the classification     5
    candidate side CONFIRMED, artefact rebuilt since                          2
    candidate side CONFIRMED, no path recorded                                1
    COULD NOT TELL (at least one indeterminate track)                        48
    NO CANDIDATE SIDE -- refuted, artefact unchanged                          1
    NO CANDIDATE SIDE -- refuted, artefact rebuilt since                      1
    not classified at all                                                    38

**5 confirmed, 1 refuted, 89 undetermined.** The reconstruction is demonstrably
right on err_83, err_87, err_98, err_157 and err_173, and demonstrably wrong on
**one** file. **Everything else is undetermined by the instrument that would
settle it — not by the reconstruction.**

**And this cross-check has the defect it is measuring.** The classification store
was written 2026-08-30T22:12Z and **carries no per-row artefact identity**, so
"unchanged since" is bounded by the store file's mtime, which dates the last write
to the whole file. 25 of the classified files have an output rebuilt since. **My
own audit cannot say which artefact it compared against — which is field 2, in the
instrument being used to argue for field 2.**

## err_152: the one demonstrated case, in full

    eng   the candidate does not carry eng      -> master-derived by construction
    jpn   1 master track, 1 output track, correlates at peak 1.0000 vs the master
                                                -> the master's own audio, shifted
    batch_state:  verdict in_sync, compared "output vs master file",
                  max_abs_lag_ms 0.0, outcome MERGED_IN_SYNC
    renders as:   "Step 2 only; Step 1 could not measure", Step 2 = pass, complete

**There is no candidate-derived track anywhere in that output, and the verdict is
a 0.0 ms measurement of the master against itself.** The byte-identity guard let
it through because a shifted copy differs byte-wise. **One file, named, current,
and it is worth more than the 50-file figure I nearly reported.**

**A screen, offered as a screen and not as a finding:** 29 of the 96 `in_sync`
files read **exactly 0.0 ms**. A well-aligned merge can also read 0.0, so this
identifies a population to look at, not a population that is wrong.

## Which reports the owner actually reads — the part with a price

- **`AGENT_NOTES.MD`, the Master Summary Table.** 315 rows, 109 of them a Step 2
  statement. **This is the only surface where a downgrade changes what the owner
  is looking at.**
- **`OWNER_QUESTIONS.MD` / `FINDINGS.MD` / `TASK.MD`** — prose the owner reads.
  The count-based claim already appears there **with its own limit stated**:
  *"the 21 are unverifiable by the test Step 1 uses; they are not proven
  master-derived."* Nothing to downgrade; it is already honest.
- **The JSON stores and `src/diagnostics/*.md`** — agent-facing. A downgrade here
  costs nothing and buys nothing.

## The three decisions, costed

1. **Downgrade ORDER only.** 2 cells (err_78, err_293). Nearly free, and it
   removes the reconstruction that is unsound with no sound direction at all.
2. **Downgrade what the evidence contradicts.** 1 cell today (err_152), and the
   honest form is a re-measurement rather than a downgrade. **Do not extend this
   to the 48 "could not tell"** — that would be reading an instrument's refusal as
   a negative, which is the error this document had to correct in itself.
3. **Downgrade every verdict resting on a reconstruction.** 109 cells — the whole
   Step 2 column. **This is the honest state of the evidence and it is also a
   report that says almost nothing.** It would discard the 5 confirmed cases and
   the pigeonhole's sound direction along with everything else.

**My recommendation: 1 and 2, not 3** — and instead of the blanket downgrade,
**one column, not a restatement of 109 cells**: a *footing* column saying which
reconstruction carried each Step 2 verdict (`order`, `byte`, `count`, `not
recorded`). It costs one column, changes no verdict, and puts the reader in a
position to discount a cell themselves. That is the same move as the completeness
column, which answers beside the verdict rather than folding into it.

## What would settle the rest, and what I did not do

**89 of 96 are undetermined, and the instrument that settles them already
exists.** Re-running `provenance_classify` over the current artefacts converts
"could not tell" and "not classified" into an answer for every file whose codecs
allow a byte comparison. **It needs decode, C_cut owns the box, and I have not
started it** — this is a costing, and starting a decode sweep is not.

**Two things it still would not fix, both of which are the proposal.** The
classifier's own store has no per-row artefact identity, so its next run ages the
same way this one did (field 2). And correlating at peak 1.0 tells you a track is
the master's *recording*; it cannot tell you which **run** put it there
(field 3). **A content instrument can recover provenance for a file. Only the
merge can record it for the corpus.**
