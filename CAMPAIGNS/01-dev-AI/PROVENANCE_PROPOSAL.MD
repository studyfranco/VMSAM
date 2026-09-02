# One ruling on provenance: three fields, and only one of them is on the merge path

**Read at `aedf112`** (invariant 154). Written by Agent 4 (forensics) at the
Lead's request, consolidating proposals from four agents so the owner rules once
rather than reconciling four. **Nothing here is implemented. No code changed, no
row was written, nothing on the merge path was touched.** The protected closure
is intact at this revision: 78 members, 0 differing.

## The one sentence

**The merge knows things at merge time that nothing writes down, and every tool
afterwards reconstructs them differently.** Four agents reached that from four
directions in one day. The reconstructions are not merely expensive; three of
them are demonstrably wrong on shapes that occur in this corpus.

## The four proposals are three fields

| Raised by | As | Field |
| :--- | :--- | :--- |
| Agent 1 | *zero of 110 lag figures can be dated to the code that produced them* | **1. Writer stamp** |
| Agent 3 | a row-level stamp of the writer, taken at import time | **1. Writer stamp** |
| Agent 4 | `written_utc` on a verdict row (landed) | **1. Writer stamp** |
| Agent 4 | an identity record for the artefact judged | **2. Artefact identity** |
| Agent 2 | a provenance field on the tracks, written by the assembler | **3. Track provenance** |

**Three of the four are the same write.** Agent 1 measured the gap, Agent 3
designed the stamp, and my `written_utc` is its other half — *when* and *by what
code*, taken by one process at one moment. Adopting them as separate fields
would be three writes where one belongs. **Fields 1 and 2 are also one write**
(same writer, same moment) but **two names**, because they answer questions about
different subjects: field 1 is about the writer, field 2 is about the thing
judged. Folding them into one name would repeat the error field 2 exists to
catch.

**Field 3 shares nothing with the other two** — different process, different
moment, different subject, and it is the only one that needs your authority.

---

## Field 1 — Writer stamp (host-side, per row)

**What.** On every row a host-side tool writes into a verdict or state store:

    "written": {"utc": "...Z", "rev": "<commit of the loaded module>",
                "dirty": true|false, "by": "<tool>"}

**Who writes it and at what moment.** The process writing the row, with `rev`
and `dirty` captured **at import time, not at write time.** This is the whole
point of the field and the one detail that must not be got wrong: a long-running
`batch_verify` holds the module it imported, so `git rev-parse HEAD` called at
write time reports the checkout the process is *not* running. That is not a
hypothetical — it is the measured defect. `src/diagnostics/tools/provenance.py`
already has `_commit()`, and **calling it at write time would reproduce exactly
the fault this field exists to record.** `dirty` is there because host-side
sweeps run from worktrees with uncommitted edits, and a bare HEAD then asserts a
revision the running code did not equal.

**Attachment points.** `batch_state` needs nothing new structurally: `record_row`
is already the single row writer and already stamps `recorded_utc` — the writer
revision goes beside it. `reverify_external.py` and `verify_no_output.py` carry
`written_utc` already and need only `rev`. `verify_unverified.py` (step1) writes
`{lang, lags, worst_ms, verdict}` with no date at all.

**Which of today's findings it would have settled.**

- **Agent 1's zero of 110**, prospectively and completely. Today: which image
  performed the merge, 92 of 110; roughly when the merge ran, 74 of 110; **which
  code measured the lag, 0 of 110.**
- **The 14 rows produced by pre-fix code after the fix landed** (`7744aa3`,
  2026-08-31 20:47 UTC; affected rows run to 2026-09-01T02:39:03Z). The container
  is gated on `git_commit == HEAD`; **a host-side gate cannot copy it, because a
  running process's code is not its checkout's code — that is the defect.** A
  stamp records what happened instead of trying to prevent it.
- **err_78**, which was settled by reasoning from delay residues mod 10 ms — a
  store whose every value lands on the envelope bin was written without
  `refine_peak_index`, whatever its timestamp says. One field, one line.
- **The `build` field's false provenance.** `build` is `deployed_build()`, the
  **container's** SHA. It is accurate about the merge and silent about the
  host-side measurement, and it currently reads as provenance for both. A
  separate writer field makes `build` correct rather than misleading.

**Which it would NOT settle.** It does not say the file on disk is the file the
verdict described (field 2), or where a track came from (field 3). It does not
improve any measurement. And — measured, on my own landed `written_utc` — **a
date prevents nothing; it lets a reader notice.** Case by case: the 27 stale
verdicts, yes, all 27; `step1_verdicts`, 25 of 40, because 15 ledger rows have no
`recorded_utc` on the far side to compare against; the 11 reported as
artefact-gone, no — nothing about them was a dating problem.

**Cost.** Near zero. Host-side only, off the merge path, no decode, no probe. An
import-time capture and one dict key. **No owner authority is needed for this
field**; it is here so the ruling is complete, not because it is blocked.

---

## Field 2 — Artefact identity (host-side, per verdict row)

**What.** On every row carrying a verdict about an output file, the identity of
the file **as it was when judged**: path, size, mtime, and an explicit `state`
that distinguishes *recorded* from *absent* with a reason. Landed in
`reverify_external.py` and `verify_no_output.py` as `artefact_identity()`;
missing from `batch_state`'s per-language rows and from `step1_verdicts`.

**Who writes it and at what moment.** The verdict writer, at the moment the
verdict is formed — not later, and never back-filled.

**Which of today's findings it would have settled.**

- **The 27 `in_sync`-beside-`FAILED` rows.** Both are true, of different runs.
  Which run the surviving file came from is the question no store can answer, and
  it is the only thing standing between those rows and a resolution.
- **The whole "a verdict outliving the thing it was about" framing**, which was
  wrong: 106 verdict rows, 83 artefacts at a recorded path, 23 at the derived
  path, **0 unlocatable.** *"The artefact still exists"* and *"the verdict still
  describes it"* are different claims and **only the first is checkable today.**
- **The stale-artefact class at row level** — the same hazard the provenance
  sidecars fixed for derived JSON, unfixed for the media the verdicts are about.

**Which it would NOT settle.** Size and mtime detect **replacement by a later
run**, which is the case that actually occurs here. They do not detect an
in-place rewrite that preserves both, and they cannot help if a re-run lands in
the same second. **mtime is not the merge time** and must never be read as one.
A content hash would close that gap; on a 3.9 GB output it is not free, and I do
**not** recommend it unless you want tamper-evidence rather than run-identity.

**Cost.** Effectively zero, and in one place **below** zero: `batch_verify`
already snapshots `(size, mtime)` for every file in the output directory before
and after a run, and already records `merged_file` and `size_mb`. **The mtime is
in memory and is discarded.** Keeping it is a smaller change than the code that
computes it.

---

## Field 3 — Track provenance (merge time, per output track) — THE ONE THAT NEEDS YOU

**What.** For each audio and subtitle stream in a merged output, **which source
file contributed it.** Agent 2's phrasing, which I have not improved on:
*recorded by the assembler that knows.*

**Who writes it and at what moment — and the site matters more than the field.**
Provenance exists in exactly one window. `generate_launch_merge_command` builds
the first mux from three contributors — `generate_new_file` (the best video),
`generate_merge_command_common_md5` (its same-md5 companions), and
`generate_merge_command_other_part` (the other file) — and **each one knows the
source file it is contributing.** The moment that mux runs, the knowledge is
gone: everything after it reads `out_video_metadata`, which is a probe of the
**combined** intermediate. `keep_best_audio` at the same function runs on that
combined list, and **not one of the fields it compares records which file a track
came from.**

**So this cannot be recovered downstream, and three tools currently try:**

- **`sync_verdict`'s note** *"all N '<lang>' tracks are master-derived; candidate
  contributed none"* is a **count comparison carrying a provenance claim**. Its
  premise (the master's tracks are muxed first) is correct at mux time and silent
  about the dedup that runs between the mux and the count. If a master track
  loses to a candidate one, the output still holds `n_master` tracks and one of
  them is candidate-derived: **the condition fires and the note is wrong.**
  Reachable on precisely the shape all eleven count-based files have.
- **Position.** Invariant 86: in one truncation case the master's track was
  dropped and replaced by the candidate's, so the output's *first* track of that
  language is the candidate's 108.0 s one. **Ordering tells you where a track
  sits, not where it came from.**
- **Byte-identity.** The self-comparison guard tests for byte-identity **in a
  pipeline that re-encodes.**

**Which of today's findings it would have settled.** The `per_language`
population, exactly: **1 of 64 verifiably cross-provenance, 22 in the
configuration Step 1 refuses, and 41 that cannot be assessed at all because the
fields that would say were never written.** Not one of the 43 could be certified
except the single Path A result. It also settles the master-derived note for the
twelve, and removes the position inference from the eight truncation files.

**Which it would NOT settle.** Nothing about *when* or *by what code* a verdict
was formed, nothing about the artefact's identity afterwards, and nothing about
why a track is short. It makes a provenance claim checkable; it does not make a
measurement better.

**Cost, stated plainly.** All three contributing functions and the mux itself are
**inside the 78-member protected closure** — `generate_launch_merge_command`,
`generate_new_file`, `generate_merge_command_common_md5`,
`generate_merge_command_other_part`, `keep_best_audio`. **There is no site
outside the closure where the answer still exists.** The mechanism, however, is
already proven on this exact path: `keep_best_audio` reads `CHIMERIC` and
`VMSAM_RESAMPLED` stream tags today, `native_wins` depends on them, and the
Matroska tag case-folding hazard is already handled there. **What is missing is
the site, not the technique.**

**The cheapest shape, if you grant it.** `ffmpeg_cmd_dict` is already threaded
through all three contributors as a shared accumulator. One additional key
mapping intermediate stream order to source file, plus one write beside the
output, **changes no command string.** The record must then be resolved through
the final `--track-order` list, because `keep_best_audio` drops tracks between
the two muxes — a record captured at the first mux describes the intermediate,
not the output, and joining them is part of the field, not an afterthought.

---

## Recommendation

**Two changes, not four.**

1. **One host-side row header** carrying fields 1 and 2 together, written by the
   host-side verdict writers -- `batch_verify` (`record_row`), `reverify_external`,
   `verify_no_output` and `verify_unverified`. No merge path, no closure, no decode. This
   needs no ruling — it is listed so you can see the whole shape and object if
   you want it shaped differently.
2. **One merge-time track provenance record.** This needs your authority and is
   the only item here that does.

**If you rule on only one thing, rule on field 3**, for a reason that is about
timing rather than importance: fields 1 and 2 can be added at any moment and a
re-verification will then produce a stamped row, because verification is
repeatable and cheap. **A merge is neither.** Every merge that runs without field
3 produces an output whose track provenance can never be recovered — the
information exists for the duration of one function call and is then gone for
that file forever. **The cost of waiting is asymmetric, and it accrues while the
sweep runs.**

---

## What this does NOT do for what is already written

**Every field here is prospective. None of it repairs a single existing row**,
and the temptation to make it look retrospective is the failure mode invariant
109 was written about: back-filling a provenance field manufactures the thing the
field exists to record. **Absent is a value. Keep it.**

Concretely, and per store:

- **The 106 verdict rows** carry no artefact identity and never will. They can be
  **re-verified**, which produces a new row about a new observation — **not a
  repair of the old one.**
- **The 110 lag figures**: 0 are datable to the code that produced them, and that
  is permanent. 71 have no row-level dating of any kind. Attribution from the
  data — the 10 ms envelope residue — **remains the only method, and it yields a
  bound, not an identification.**
- **33 of 48 stores** cannot date any row. Of the four that look dated, two date
  the **subject** rather than the row: `output_provenance.output_mtime_utc` dates
  the artefact and `correction_state.started` dates the correction. A reader
  taking `output_mtime_utc` as *when this was judged* is reading a fact about the
  file as a fact about the verdict.
- **Every published output to date** has no track provenance. The one partial
  exception is real but narrow: tracks the repair paths touched carry `CHIMERIC`
  or `VMSAM_RESAMPLED` tags, so *those* are identifiable. **A track that came
  straight from either source carries nothing that distinguishes it**, which is
  the majority and is exactly the population the three broken reconstructions are
  applied to.

**And none of the three improves a verdict.** They are provenance instruments.
They let a reader tell which run, which code, and which source — they do not make
a measurement more likely to be right.

## The ruling requested

1. **Field 3, track provenance at merge time: granted or refused?** It requires
   editing protected-closure members. If granted, may it be the accumulator-key
   form above — no change to any command string, one record written beside the
   output?
2. If refused, **do you want the three downstream reconstructions marked as
   unverifiable in the reports that use them**, rather than silently carrying a
   provenance claim they cannot support?
3. **Fields 1 and 2 as one host-side row header** — any objection, and do you
   want a content hash (field 2) or is run-identity enough?
