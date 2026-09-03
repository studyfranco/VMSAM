# Open questions for the owner

Asked when the owner returns, not while they are away. Answered items move to
the bottom with their ruling, so the record shows what was decided and when.

## Open

**1. Phase 2 authorisation.** Rescan of the one 27-file series plus 3
singletons that refused as `repair_declined_not_piecewise`. Pending since
before the 2026-09-01 compaction. It is the largest block of the corpus that is
neither Resolved nor proven unmergeable. Needs container time we may not have
before the token limit.

**2. Are the new error files the same series as the 27?** The owner mentioned
new error files arriving and named the series. If it is the same series as the
27 that refuse as not-piecewise, then Phase 2 and the new-file ingest are one
job rather than two, and I would merge them.

**3. Should the hourly refresh detect a usage park and run the come-back
procedure itself?** The 5-hourly wake-up fires on a fixed clock; the credit
window resets a floating five hours after first use, so they drift apart by
construction and the wake-up has landed mid-park. If the hourly instead ran the
come-back whenever the gap since the last refresh exceeds ~45 minutes, it would
fire exactly when a reset actually happened and cost nothing on a normal hour.
Cheap to build; not built.

**4. The cron prompt still says 309 error files.** The corpus is 315 — the 309
came from a stale mapping, corrected 2026-09-01. The wake-up and hourly prompt
files live outside the repository, so I have not edited them.

**5. May an agent add a locale gate and a drop record to `keep_best_audio`?
BOTH functions are inside the protected closure, so this needs your authority.**

`keep_best_audio` decides which audio track survives using **only**
`Format`, `Channels`, `SamplingRate` and bitrate. It never reads the track
Title, never reads the language, and computes no md5. **So two different
regional dubs of one language, same codec and same channel count, are
indistinguishable to it and one is dropped on bitrate.** That is the mechanism
behind the single case where a distinct dub was demonstrably lost.

You have already ruled on this decision **in a different place**: for the speed
policy you ruled that removal is gated on locale metadata and never on the audio
test, because measured fidelity cannot separate a regional dub from a format
variant (0.8334–0.9922 against 0.9270–0.9894, overlapping). **This is the same
decision at a site with no gate.**

Two separable changes, and they should not be conflated:

- **The gate** — do not drop a track whose Title names a locale the survivor's
  does not. Prevents future loss. This is the fix.
- **The record** — one line per dropped track with both sides' metadata. Makes
  the decision auditable. This is the instrument.

Everything needed is already in scope at the line that sets `keep=False`: both
tracks' language, Title, Format, Channels, SamplingRate, bitrate, and which side
lost. A few lines at one site, the same shape the subtitle path already
produces. **No new probe, no decode, no extra pass.**

**The record is cheaper and strictly safer — it changes no behaviour — so if
only one lands it should be that one.** The gate without the record is
unverifiable; the record without the gate documents the loss without preventing
it.

**NOT CLAIMED:** that any of the undecidable drops were wrong. This explains how
a wrong drop would happen and why nothing would show it.

**6. Is "Resolved" meant to require Step 1 AND Step 2, or does a Step 2 pass
suffice?** Ten files carry `MERGED_UNVERIFIED` in their ledger row while the
container-side Step 2 verdict reads `in_sync`; the summary reports two of them
as Resolved and eight as over the bar, on the Step 2 measurement, without the
row's own outcome contradicting it. That is defensible — Step 2 in the container
is the stronger check — but the row and the status disagree in a reader's eyes.
I have made the summary say which step carried the verdict rather than change
the rule.

**7. The truncation repair needs the closure or the image. Your "adapt around it"
ruling has no reachable implementation, and this is now a bounded fact rather
than an open search.**

Traced from the merge log, no merges spent:

- **Stripping subtitles from the re-encode pass is reachable** — the chimeric
  build is outside the closure.
- **Nothing that could re-attach them is.** The path runs through the function you
  put off limits and two protected-closure members, and **both merge commands are
  built at the closure root** (`mergeVideo.py:3249` and `:3302`).
- **The chimeric is not an escape route**: the final mux takes `-A -S -D
  --no-chapters` on it and `--track-order` names only the other inputs, so **it
  contributes attachments alone and its subtitles are discarded by explicit
  flag.** The published subtitles reach the output *through* the second pass, and
  nothing else supplies them — confirmed twice, by stream arithmetic and by
  argv.

**So the five known files stay broken unless you choose one of:**

- **grant closure access for this one change** — the smallest edit is not mapping
  subtitle streams into the audio re-encode pass, then re-attaching them at the
  mux;
- **accept the gap-filling variant** (question 8) — reachable today, no closure
  access, but it alters subtitle content in a published artefact;
- **wait for an upstream ffmpeg fix** — we have a reproduction from primitives
  that uses nothing from your library;
- **leave them.** Five files, and the defect is silent: the container duration
  follows the longest subtitle while the audio follows the gap.

**8. May a subtitle packet be inserted inside a long gap, to break the trigger
without changing which streams exist? NOT RECOMMENDED, and not designed.**

It is reachable at the chimeric build with no closure access, and it works
because the trigger is the *gap*, not the stream. **It alters subtitle content in
a published artefact**, which is a different question from the one you ruled on.
Found and declined by the agent that found it.

**9. One ruling on provenance — `src/PROVENANCE_PROPOSAL.MD`.** Four agents
independently reached the same place from four directions: the merge knows things
at merge time that nothing writes down, and every tool afterwards reconstructs
them differently. The four proposals are **three fields**, and **only one needs
your authority** — a per-track record of which source contributed each output
stream, written by the assembler. There is **no site outside the protected
closure where that answer still exists**: after the first mux everything reads a
probe of the combined intermediate. Three downstream tools currently reconstruct
it and all three are wrong on shapes this corpus contains. The other two fields
(a writer stamp, an artefact identity record) are host-side, off the merge path,
and need no ruling — they are in the document so the shape is complete. **All of
it is prospective; none of it repairs a written row**, and the document says so
per store.

**Costed, if you refuse it — `src/PROVENANCE_DOWNGRADE_COSTING.MD`.** Marking the
downstream reconstructions unverifiable is **three different decisions**: the
order-based claims are **2 cells** (err_78, err_293), the one claim the evidence
contradicts is **1** (err_152, an `in_sync` at 0.0 ms measuring the master against
itself), and a blanket downgrade is **109** — the whole Step 2 column of the
Master Summary Table. **The count-based claim everyone has been discussing changes
0 cells**, because it only ever withholds a verdict and never asserts one.
**Recommended instead of the blanket downgrade: one footing column** saying which
reconstruction carried each Step 2 verdict, which changes no verdict and lets a
reader discount a cell themselves.

## Answered

### 2026-09-01, four rulings

**Retain the merge intermediate: ADD A RETAIN FLAG, default off.** Mirrors
`VMSAM_RETAIN_CHIMERIC`; skips `tools.file_remove` for `first_pass_out` at
`mergeVideo.py:3314`. **Agents prepare; the owner rebuilds the image.** This is
the only remaining way to observe the offset that feeds the confirmed truncation
cut -- it is not in the inputs and not in the output, so if it exists it is
created and consumed inside one ffmpeg invocation.

**Phase 2: RUN IT NOW, pause the sweep.** The 27-file series plus 3 singletons
refusing as `repair_declined_not_piecewise` -- the largest block that is neither
Resolved nor proven unmergeable.

**`keep_best_audio`: THE OWNER DOES ALL OF IT.** All three converging items --
the drop record's `could-not-compare` state, the locale gate, and the 42-output
byte-identical selection question -- **are the owner's work. No agent proposes,
scopes or edits any part of it.** Agents keep the measurements current here and
nothing more.

**Deploy: NOW.** `77ebfc5` carries `21f79cf` (repair enabled); the deployed
image `8e1b0e3` carried the build without the enable, so
`repair_not_compatible_candidate` returned False and **every prior
container-side repair result was vacuous.**


**REPAIR BELONGS IN `remove_not_compatible_audio`, NOT IN `fusion.py`**
(2026-09-01, owner). Superseding the earlier `remove_not_compatible_video`
sketch:

> `fusion.py` does not create the repair file. Repair happens **after** a file is
> found incompatible, inside
> `remove_not_compatible_audio(video_obj_path_file, already_compared)`. **If it
> cannot be repaired, raise an error.** If it can: **remove the incompatible
> entry from the list and add the repaired file — audio and subtitles — to
> `videoObj.delay_same_md5_audio`.** `fusion.py` then simply tries again to
> produce a file with `mergeVideo.py`.

**So the repair is local to the point of rejection**, and the retry path stays a
retry rather than becoming a second implementation.

**The entry point must be recorded** (2026-09-01, owner). `batch_state` writes
`stage: "ran"` and does not say **which of the three entry points** produced a
result, so a verdict cannot name the code path behind it. **One field at the point
of decision** — the same shape as the build stamp and the audio-drop record, and
unrecoverable afterwards.


**The subtitle native-wins rule is the OWNER'S to write, not ours** (2026-09-01).
**No closure grant is needed for subtitles**, and the audio exception stays at its
original single member.

> **HANDS OFF: `clean_number_stream_to_be_lover_than_max` and the subtitle
> selection path are the owner's active working area.** They are moving the list
> creation and adding a method to remove chimeric/resampled subtitles on demand,
> which changes a lot inside that function. **No agent edits it, and no agent
> proposes edits to it.**

**Their first design, recorded because it is sound and may return** even though
they chose a different route:

- add a boolean to `video.video.__init__` — *does this file carry a
  chimeric/resampled subtitle* — **and the same for audio**;
- set it true in `get_mediadata()` when one is found;
- test it in `clean_number_stream_to_be_lover_than_max`, **before the first
  `if number_sub_will_be_copy > number_max_sub_stream`** — i.e. at the always-on
  point, above the quota gate.

**That placement is the part worth keeping**: it is exactly the always-on site the
scoping identified independently, reached from the other direction.

**COLLISION WARNING for the repair-upstream work**: the owner may add fields to
`video.video.__init__` and `get_mediadata()`. **Any temporary `video.video` built
for the md5 list must be constructed through the real constructor and probed the
real way** — never assembled field by field — or it will be missing whatever they
add and will fail somewhere far from the cause.

**The scoping done for it still stands and is handed over**, because it cost real
lookups:

- The four functions that first looked like the site — `remove_sub_language`,
  `keep_one_ass`, `sub_group_id_detector_and_clean_srt_when_ass_with_test`,
  `clean_srt_when_ass` — are **quota reducers**, every call site nested inside
  `if number_sub_will_be_copy > number_max_sub_stream` (verified at
  `mergeVideo.py:2551` and `:2560`). **They do not run on a file under the cap.**
- `clean_number_stream_to_be_lover_than_max` **always runs** its counting loop,
  which already drops MD5 duplicates unconditionally — an always-on selector
  rather than a quota-only path.
- **There is no home outside the closure**: the orchestrator's only caller is
  `generate_launch_merge_command` itself, and the subtitle list is constructed
  inside it.
- **The `CHIMERIC` marker reaches subtitle streams and survives a `-c copy`
  remux**, verified end to end rather than read from source.
- **Subtitle tracks of one language are often not substitutes** — forced,
  hearing-impaired, dub-titles and full are different kinds. A language-only rule
  could drop a re-timed *forced* track because an untimed *full* track exists.
  `test_if_forced(sub)` and the probe's `codec_name` are both available at the
  always-on site, so scoping by (language, forced-ness, codec class) needs no
  group machinery and no second member.

**pHash prior art: assess before deleting** (2026-09-01). `src/get_cut_time.py`
(574 lines) and `src/frame_compare.py` (242 lines) implement pHash DCT frame
comparison with a scene-detection fallback — the method asked for in the
frame-alignment task. **Both parse; nothing imports either.** To be evaluated on
a known-good pair before any decision to delete or revive.


**The `int()` delay-scale truncation: MEASURED, and no verdict moves.** The owner
ruled "measure the impact first" rather than fix it. Result: **zero**, for two
independent reasons.

**1. It cannot change a refusal.** The truncation scales every delay in a file by
**one** factor, so equal stays equal and distinct stays distinct — `len(set_delay)`
is invariant *by construction*. The only reachable test is the 127 ms two-delay
window: of 95 files carrying the multiple-delay exception, 19 have exactly two
distinct delays, and **0 cross the bar even at the worst case the truncation could
ever produce.**

**2. Method 2 absorbs it.** The second method is **additive** on the first, so it
measures and removes whatever the first got wrong — a scale error included —
accepted whenever the refinement is under 500 ms. That bar is reached at a delay
of ~108.7 s. Of 384 recorded method-1 delays, **276 (72%) sit under the crossover
and are absorbed entirely**; the other **108 are all above 159 s** — saturated
output of failed correlations pinned near the search ceiling, already refused for
reasons unrelated to this.

**No Resolved file can stop being one, and the direction is why**: correcting makes
the applied delay **larger, i.e. closer to truth**, so the residual it leaves
*shrinks*. Movement is only ever toward passing. Corroborated from the residual
side — 82 files with a recorded worst residual, **median 0.7 ms**, none within a
truncation-width of the bar. **A median of 0.7 ms is itself evidence the
truncation never reaches the output**: an uncorrected 0.46% on a 3 s delay would
leave 13.8 ms.

**Stated limit**: Step 1 delays for *Resolved* files are **recorded nowhere** —
they survive only inside failure exception text, so all 384 tuples come from
refused files. **The Resolved claim rests on the structural argument and the
residual distribution, not on a per-file de-truncation**, which would need a
re-run.

**So the fix is optional rather than urgent.** It remains a real defect —
understating every delay by ~0.46% in a fixed direction — but nothing recorded
depends on it.


**CLOSURE EXCEPTION, granted 2026-09-01 — narrow, and it does not open the
closure.** The owner authorised editing exactly two protected members:

    mergeVideo.keep_best_audio
    mergeVideo.find_differences_and_keep_best_audio

**Only to add the native-wins selection rules** for the repair-upstream design:
a chimeric or resampled track is removed when an unmodified native exists, for
audio and for subtitles. **No other closure member is authorised, and this
exception does not extend to any other change to these two.** The closure check
still runs and still guards the other 76.

**Why it had to be explicit**: the design *required* the edit, and the agent
scoping it refused to treat that as permission — **a standing rule is not
repealed by a design that happens to need it.** That refusal was correct and the
grant is recorded here rather than living in a conversation.

**`VMSAM_CHIMERIC` was authorised as a prerequisite — AND IT DOES NOT NEED
BUILDING. The tag already exists and ships today.** `merge_video_chimeric._build`
writes `CHIMERIC=<description>` onto every assembled audio track, and repeats it
in the subtitle mux rather than trusting `-c copy`. Verified on a published
output: master-derived tracks carry **no** marker; every chimeric-derived track
carries e.g. *"re-timed 3 pieces; patched 5.0 s from master (eng, same_dub)"* —
**what was patched, how much, from which source, and under which rule.** Exactly
the content specified.

**So the rule should point at `CHIMERIC`, not at a prefixed twin.** Renaming
orphans the marker on every file already published with it, and a reader written
against the new name sees nothing on those. **Two tags for one fact is how they
drift** — the same objection that ruled out inferring membership from build
records: the answer is already in the file.

**One open detail, deliberately not decided by writing a second tag**:
`VMSAM_RESAMPLED` is JSON with four fields; `CHIMERIC` is prose. **Presence is
sufficient for native-versus-modified, so nothing is blocked.** A rule wanting the
piece count as a *number* would be parsing a sentence. **If parity of format is
wanted, that is a change to decide on its own**, not one to arrive at sideways.



**config.ini may name only tools present in the environment** (2026-09-01).
The agent container is built from the same image as the service, so the rule is
checkable from inside it. Now enforced by a test: a bare name must resolve on
PATH, an absolute path must exist or be COPYed into the image by the Dockerfile.

**The 15 ms Step 1 bar becomes half a frame, per file** (2026-09-01), keeping
15 ms for VFR references. `adjust_delay_to_frame` rounds to the nearest frame,
so a perfectly snapped delay still carries up to 20.85 ms at 23.976 fps: the bar
sat below the quantisation floor and was unachievable by construction. NOT a
lowered standard — each file must still be shown snapped to the NEAREST frame
rather than an adjacent one before it counts as Resolved.

**Priority for the remaining budget: the 55 merged-unverifiable** (2026-09-01).
They already merged; only Step 2 verification failed, so they are the best
chance of moving files per token spent.

## The audio-drop record needs a "could not compare" state -- and only you can add it

**Status: OPEN -- requires an owner edit. No agent can build this.**

**The defect.** The scoped record logs which of two same-language tracks was
dropped and why. `get_bitrate` **raises on 8 of 45 tracks measured across 16
real files**. Built to the original scope, the record would **log a decision for
a comparison that never ran** -- which reads as a judgement rather than as a
failure, and is worse than no record at all.

**What is needed.** Three states rather than two: **compared-and-dropped,
compared-and-kept, and could-not-compare (with the reason)**.

**Why it is yours.** `keep_best_audio` (2286-2333) and
`find_differences_and_keep_best_audio` (2204-2284) are both **protected-closure
members**, verified from `closure_check --list`. Neither the log nor the locale
gate can be added by an agent under the standing rule. **The Lead dispatched
this to an agent in error; the agent declined and said why rather than building
something adjacent and calling it the task.**

**What is already done, so the edit is small.** `audio_comparability.py`
(`d50df4e`) measures **whether the comparison can run**, never what it would
decide -- classifying each track as `FROM_FFPROBE`, `FROM_MEDIAINFO` or
**`UNAVAILABLE`**. It is validated against the real function in one process:

    real video.get_bitrate on the real dicts   ok 37   RAISED 8
    audio_comparability on the same files      37 comparable   8 UNAVAILABLE

**Identical.** So the missing state is **specified, named, tested and
measurable on real files** before any closure member is touched. It does not
reimplement the selector, deliberately: a reimplementation agreeing on every
file here would be indistinguishable from the real thing **until the first file
where it was not.**

**CENSUS COMPLETE -- the scope is now known and it is not a completeness case.**

Of **1921 audio tracks across 560 files, 35 (1.8%) carry no readable bitrate**
from either tool -- the exact condition on which `get_bitrate` raises. They sit
in **29 files of 560**, and they cluster hard:

    folder 172   15 files      folder 109    6 files
    folder 159    3 files      folder 181    3 files
    folder 111    1 file       folder 248    1 file

**Six folders, one holding over half.** Declining language: `jpn` in 24 files,
`eng` in 5. **The earlier "8 of 45" and "0 of 422" were one phenomenon seen from
inside and outside a single folder** -- clustering, with the alternative named
in advance.

**29 of 29 affected files lose a REAL comparison.** A first pass counted pairs
**within** a file and found 0, which nearly published as *"the condition exists
and costs nothing"*. `keep_best_audio` ranks across the **merged** set -- the
master's tracks together with the candidate's -- so **a track alone in its
language within its own file is not alone after the merge**:

    within-file pairs declining      0 of 29 files
    CROSS-FILE pairs declining      29 of 29 files

**The first number was not a smaller version of the second; it answered a
different question and pointed the opposite way.**

**Not claimed**: that the folder is the cause rather than a proxy -- all
affected tracks are lossy AAC with no bitrate from either tool, but why is
unestablished. **And no harm is demonstrated** -- a declined comparison means
two tracks went unranked; **whether the wrong one survived needs the record that
does not exist**, which is what makes this question circular until you make the
edit.

## 42 of 114 outputs carry 3+ byte-identical audio tracks -- a selection question, not a defect

**Status: OPEN -- needs a decode slot to settle, and then a ruling from you.**

**What is measured.** Across 114 merged outputs, **42 carry a group of three or
more byte-identical audio tracks**, up to **11 tracks over 10 languages**:

    eac3   35        aac    7        flac   0

**What is NOT claimed.** That those tracks are duplicates. At constant bitrate,
size and frame count are exact functions of duration alone, so **byte identity
carries no information about the audio either way** (invariant 138). The FLAC
column being zero says the signature tracks the **rate model**, not the content.

**Worked example.** One output holds eight byte-identical `eac3` tracks. Its
candidate already carried **nine** such tracks -- ger, eng, spa, spa, fre, ita,
jpn, por, tha -- and the merge kept eight, replacing only `jpn` with the
master's own. **Nothing in the metadata separates them**: no `TITLE` tag on any
of the eleven audio tracks, identical channels, layout and sample rate, and the
two `spa` tracks agree in every readable field. **Releases do legitimately ship
Latin-American and Castilian Spanish.**

**Why this is not a pipeline defect.** Even if the tracks are duplicates, **it
is a property of the incoming release, not something the merge produced.**

**Why it still matters, and what the question actually is.** The merge **kept
eight of the nine.** If they are copies, the published file carries eight
language labels over one recording -- **a `keep_best_audio` selection question,
a different family from the truncations, applying to 42 of 114 outputs rather
than to one file.**

**What would settle it**: a hash of **decoded samples**, not of the compressed
stream. That needs decode and is not queued.

**The ruling needed from you** is what `keep_best_audio` should do when several
tracks in different languages are plausibly the same recording -- which is
inside your active working area, so no agent has proposed a change.

## Did the audio selection run before or after the truncation? (err_211)

**Status: OPEN -- only you can answer it, because settling it means reading
`keep_best_audio`'s ordering, which is your working area.**

**The observation, from the artefact on disk:**

    MASTER   eng 1440.832   fre 1441.001   ger 1441.001   jpn 1441.001
    OUTPUT   eng  441.479   fre 1441.0     ger 1441.0     jpn 1441.942

**A complete `eng` track existed in the master. The output holds one `eng`
track and it is the truncated one at 441.5 s.** The master's is absent entirely.

**The other five short tracks have no master counterpart** -- nothing was
available to lose. **`eng` is the single case where a complete track existed and
the truncated one was kept in its place.**

**The selection is not obviously at fault.** `jpn` shows it working as intended:
master 1441.001 dropped, candidate 1441.942 kept, both complete.

**The two readings, and they differ enormously:**

    BEFORE   the selector chose `eng` from the candidate, and that track was
             truncated afterwards. The selector is blameless and the fix is
             entirely in the encode.
    AFTER    the selector compared a 441 s track against a 1440 s one and kept
             the short one. That is a selection defect as well as an encode one.

**No agent has inferred which**, deliberately -- an ordering cannot be read off
a result. **And no agent has scoped a change**, per your ruling that
`keep_best_audio` is yours.

**Context**: the merge recorded this file `MERGED_IN_SYNC 17.7` with ffmpeg
exiting 0 and no diagnostic anywhere (FINDINGS 28). **The truncation reproduces
on demand on the current image**, so whichever reading is right can be tested
against a live case.

## URGENT-ish: a timeout leaks a second process onto the merge output path

**Status: OPEN -- `tools.launch_cmdExt_with_timeout_reload` is a PROTECTED
CLOSURE member (confirmed by `closure_check --list`), so no agent may touch it.
This one is yours and it is the only defect found tonight that can CORRUPT an
output rather than mis-describe one.**

**`src/tools.py:92-113`**, verified by reading the source:

    while unpocessed:
        cmdDownload = Popen(cmd, ...)                  # line 95
        try:
            ... communicate(timeout=timeout)
            unpocessed = False
        except TimeoutExpired:
            cmdDownload.kill()
            max_restart -= 1
            if max_restart < 0: raise
            else:
                cmdDownload = Popen(cmd, ...)          # line 111  REPLACEMENT

**On timeout it kills the original, starts a replacement at line 111, then falls
out of the `except` with `unpocessed` still True.** The loop repeats, **line 95
starts another process and rebinds over the replacement**, which is then **never
communicated with, never killed, never waited on.**

**Reproduced**: `sleep 8.37`, timeout 1, `max_restart 1` -- **one process still
alive after the raise.**

**Why it matters more than the rest.** Every other finding tonight loses
information: a capped field, a missing marker, a starved queue, a verdict on a
rolling log. **This changes what happens.** An orphaned `ffmpeg` still writing
the merge output path **alongside its replacement** is a second writer to a file
that is later measured, verified and given a verdict -- **and nothing we record
would show it.**

**It is LATENT, not active.** `exceeded_timeout` is false in **96 of 96**
records and err_211's encoding pass ran **114 s against a 3600 s timeout.** **It
has not fired in this corpus and is not the truncation mechanism.**

**The fix is small** -- the replacement at 111 is redundant, since the loop
already restarts; deleting it, or setting `unpocessed = False` around it, closes
the leak. **But it is inside the closure and no agent has proposed a diff.**

## CLOSED -- no action needed: retain ONE first-pass intermediate

**Superseded by FINDINGS 36 before this was ever run. Do not act on it.** The
measurement it would have produced **was already on disk**, written by the very
passes in question, and it **refuted** the hypothesis it was meant to test: input
and output both start at 0.000000 on all seven tracks, so the mechanism's
precondition is absent.

**A retained copy would have come from a LATER run and been weaker evidence than
what already existed.** The request was raised by one agent, corrected to the
right flag by another, and then cancelled by that same agent on finding the data.

**The bounded implementation is kept, off and prepared, on its own merits.**

**Original request, retained for the reasoning only:**

### Retain ONE first-pass intermediate -- the last standing mechanism needs it

**Status: APPROVED, then CLOSED -- superseded by FINDINGS 36; DO NOT ACT ON IT.**
The reasoning below is live and correct; only this status value changed.

**Original status: APPROVED by the Lead as a bounded, one-file copy-aside. Raised here
because it touches the merge path and you authorised the flag originally.**

**Six artefacts were never retained during this campaign. Four described the
past. TWO would settle questions that are live now, and both need the same
file** -- the **FIRST-PASS intermediate**, which is the **second pass's input**,
written to the container's `/tmp` and discarded.

**CORRECTION: the Lead first approved retaining the CHIMERIC intermediate. That
is a different artefact at a different stage and cannot bear on the
hypothesis.** There are two flags and two directories:

    VMSAM_RETAIN_FIRST_PASS  -> retained_first_pass    0 files, NEVER USED
    VMSAM_RETAIN_CHIMERIC    -> retained_chimeric     10 files, used

`mergeVideo.py:5265` and `:5313` -- different functions, different artefacts.
**The run is one-shot and gated, so retaining the wrong one would have produced
an artefact that cannot answer the question.**

**What it would settle.** Both passes carry `-t 1441.942` with `-copyts`, under
which `-t` caps the **timestamp** -- so a stream starting near 1000.49 s would
receive ~441.45 s **and stop silently.** That hypothesis was refuted on
**delivered** files, correctly and by its own author, who then declined to
withdraw it: **the second pass's input is the one file that could carry a
non-zero start, and nobody has it.**

**It is the only surviving candidate mechanism.** The encoder's **complete**
stderr for the truncating invocation was captured and is silent -- established
by direct observation, not inference -- which closed the entire family of
explanations that assumed a message existed.

**The instrument was BUILT FOR THIS EXACT QUESTION and never switched on.**
`_retain_first_pass_for_diagnosis`'s own docstring, in the tree today, states
the mechanism and carries the worked measurement:

> *"This is the artefact that would settle FINDINGS 24. The second pass runs
> `-t duration_best_video` with `-copyts`, and that combination cuts at an
> absolute timestamp -- measured, not assumed: an input starting at 40 s with
> `-t 50` yields 9.985 s of content, and `-t 40` yields an empty file. So the
> length that survives is `limit - start`, and the START is a property of THIS
> file."*

**So this is not new work.** It is **running an instrument built for this
question and never used** -- the directory is empty. **The criterion should be
read against that docstring rather than restated.**

**And mkvmerge was already excluded in writing**, by whoever wrote the chimeric
docstring at `:5316`: the final mkvmerge carries **no `--split`, no `--append`,
no duration limit and no value resembling 441 across 219 arguments.** Consistent
with everything measured since, and stated before any of it.

**The bound**: `VMSAM_RETAIN_FIRST_PASS` exists, is off by default, and **has
never been used once.** Approved as **one file, on the next run of a known
reproducer (err_211 truncates on demand and has three times), as a copy-aside
with an explicit delete once the questions are read** -- **not** a retention
default. *"I would rather it be scoped to one file and removed than become a
setting nobody revisits."*

**Cost is knowable in advance**: one episode's chimeric intermediate, ~3.9 GB,
against **143 GB free of 932 GB (85% used)** and a standing 30 GB floor.
**Unbounded retention across a sweep would breach that floor; one file does
not.**

**Conditions carried, two of them set by the requesting agents on themselves:**
the flag must be **proven by test not to alter the pass it retains**; it must
**not by itself justify a merge that would not otherwise run**; and the artefact
is **deleted once read.**

**It reaches the container only at the next deploy that happens anyway.**

---

## The Master Summary Table asks two facts about two events to share one cell

**Status: OPEN — owner's ruling needed on the table's shape. No count has been
changed and no classifier touched.**

**Measured**, from `/config/output/vmsam_agent/`, joining verdicts the way
`master_summary.py` does: **205 rows carry `outcome: FAILED`. 27 of them are
classified into a Step 2 verified bucket**, because `final_status` tests
`sync == "in_sync"` before it tests `outcome == "FAILED"`. The 27 partition
cleanly across two verdict stores, 15 from `external_verdicts.json` and 12 from
`no_output_verdicts.json`, with zero overlap.

**What the 27 are.** An earlier run merged and published; a later run failed at
the delay test and produced only a `.error` log. **The published output is still
on disk** -- checked at the publisher's mirrored path, confirmed to be an output
rather than a master reached through a path that resolves back, and larger than
its master by 144-614 MB, consistent with added audio tracks. So:

    the file            has a published output, verified in_sync against its master
    the latest attempt  failed at the delay test

**Both are true and neither corrects the other.** Reporting them as "27 failed
files counted as verified" is true of the **rows** and unfair to the **files**.

**Why this is not a tie-break to be fixed in the classifier.** `sync` before
`outcome` discards the newer event; `outcome` before `sync` would discard the
older one. Either ordering loses a fact that is true. **The state "verified
output, failed latest attempt" was never representable**, and no ordering
creates it.

**Why it also cannot be inferred today.** The verdict stores retain
`lang`, `result`, and -- in one of the two -- `error_id`. **No path, no size, no
mtime, no hash, no time of measurement.** `compared: "output vs master file"`
names the *kind* of comparison, not the file. So nothing inside `final_status`
can distinguish a verdict about the artefact now on disk from one about an
artefact since replaced. **The information was never recorded.**

**The question for the owner:** should the table gain a state for a file whose
published output is verified but whose latest attempt failed -- and does such a
file count toward the end condition, or against it? It bears directly on the
mission's terminal states: these files may be closer to Resolved than the
counts say, or they may be files whose good output is about to be overwritten
by a run that keeps failing.

**Not claimed:** that the 27 are Resolved. They lack Step 1 entirely, and all
ten of one subgroup have a measurable language Step 1 never tried. **Backlog,
not ceiling.**

**Proceeding meanwhile, as it prejudges nothing:** the two verdict stores will
record **path plus one identity field at the moment the verdict is taken**, so a
future reader can *test* the join rather than assume it. Additive, local
writers only, **changes no count today** -- which is the argument for doing it
before the table moves rather than after. Old rows will lack the field, so the
reader carries `absent` as its own outcome from the first version.

---

## The master-derived note has a demonstrated mechanism for being wrong

**Status: OPEN — reading only. No edit made and none proposed; `keep_best_audio`
already has open owner items.**

`sync_verdict` writes *"all N '<lang>' tracks are master-derived; candidate
contributed none"* when `len(streams) <= n_master`. **A count comparison
carrying a provenance claim.** Its docstring supplies the missing step as an
ordering assumption -- the master's tracks are muxed first.

**The ordering premise is correct at mux time. It is silent about the
deduplication that runs between the mux and the count.** Traced by reading, five
steps, all in `mergeVideo.py`: the intermediate is muxed from BOTH files; the
selection object probes that COMBINED file, so its per-language track list spans
both by construction; the comparison is symmetric on Format, Channels,
SamplingRate, bitrate and `audioRules` -- **not one of those fields records
which file a track came from**; and the losing track's StreamOrder becomes the
mux exclusion list.

**So if a master track loses to a candidate one, the output still holds
`n_master` tracks of that language and one of them is candidate-derived. The
condition fires and the note is wrong.**

**Reachable on exactly the affected shape**: it needs two tracks of one language,
at least one per side, differing on codec/channels/rate/bitrate. All eleven
count-based files with recorded counts carry two of the correlation language on
both sides, and their diagnostics establish the codecs differ -- master
HE-AAC + E-AC-3 against candidate E-AC-3 + AAC-LC. Different formats is the
branch that consults `audioRules` and can drop either side.

**NOT claimed: that it fired on any of the twelve.** That needs the merged
output's provenance, which is not recorded -- `out_streams_in_lang` /
`ref_streams_in_lang` are absent on all twelve, and no `deep/` report examines
the output. **What changed is the note's standing: from an unverified
assumption to one with a demonstrated mechanism for being wrong on precisely
this shape.**

**Cutting the other way, and found while looking for the opposite:** a track
whose content MD5 was already added is dropped, and the master's tracks are
added first -- so where the candidate's track is byte-identical to the master's,
**the master's survives structurally, by ordering, with no comparison
involved.** That is independent corroboration from the code path for the
thirteen files whose conclusion previously rested only on a correlation
inference. **Two routes, and they agree.**

**What reading cannot settle:** whether the swap ever occurred on a given file.
That is run-time data flow plus `audioRules` configuration. The code says the
swap is reachable; only a run says whether it happened, and executing would be a
new run rather than evidence about the old ones.

---

## Fifteen more rows render as a Step 2 pass, through a second store

**Status: OPEN — same class as the 27 above, different store. Nothing changed.**

`reverify_external.py` took only rows that were `MERGED_UNVERIFIED` with a
`merged_file`. **Fifteen have since been re-merged and now read `FAILED`; their
verdict was never rewritten.** Five of the fifteen now record no `merged_file`
at all.

All fifteen render as **"Step 2 only; Step 1 could not measure"**, because
`final_status` tests `sync == "in_sync"` before `outcome == "FAILED"` and the
FAILED branch is therefore unreachable for them.

**No completion count is inflated** -- the bucket is the one honest about a
missing Step 1. What is wrong is that **the row says the merge failed and the
table says Step 2 verified**, on a verdict describing a previous generation of
the artefact.

**NOT claimed**: that any of the fifteen is out of sync, or that the earlier
artefact was bad. **The verdict may have been true of the file it measured.
That file is not the one the row now describes.**

They were findable only by joining on `outcome`, because `reverify_external`
entries carry **no timestamp** -- so nothing in the record distinguishes a
current verdict from one describing a replaced artefact. A timestamp is being
added for future entries; it cannot say anything about these fifteen.

---

## The mission's 15 ms bar has never fired, and cannot fire today

**Status: OPEN — the most consequential finding of the session. Report only;
`final_status` not edited and no diff proposed.**

    "Resolved"                      0 of 315 rows
    "In sync, over the 15 ms bar"   0 of 315 rows

**Neither figure means "no file qualifies". Neither branch is reachable.**
`final_status` tests the 15 ms threshold only in its third branch, and reaching
it requires `step1_verdict` to be `in_sync` or `out_of_sync`. Verified
independently at this read:

    step1_verdicts.json entries          40
    of which verdict == "unverifiable"   40
    of which in_sync or out_of_sync       0

**Every one of the 315 rows exits at branch one or two. The mission's own
threshold is never applied to anything.** Its silence has been reading as
*"nothing is over the bar"* when the truth is *"nothing has reached the
check"* -- indistinguishable from the table (invariant 158, in its purest
form: a guard that has never fired).

**And there IS something for it to catch.** 48 measurements over 31 distinct
files sit in the 15-50 ms band, **every one stamped `in_sync`** (10 ledger, 15
external, 23 re-verification). Against each master's own frame duration, where
half a frame = `500/fps_mst` is the floor `adjust_delay_to_frame` can leave:

    ABOVE half a frame          31   over 22 files
    at or below half a frame    17

**About a third of the band is inside the quantisation floor**, so the band
must NOT be reported as uniformly bad. **The other 31 cannot be explained that
way.**

**Two shapes, needing different repairs, sharing one verdict word:**

    constant   34      monotonic  12      scattered  2

    err_316   -46.9 / -46.9 / -46.7 ms   spread  0.2   CONSTANT
    err_124     0.5 / -43.7 / -48.3 ms   spread 48.8   MONOTONIC

An offset and a ramp at the same magnitude, both `in_sync`. **Constant
outnumbers monotonic 2.8 to 1, so the COMMON member of this class is repairable
by a single delay** -- and that is precisely what the verdict field hides.

**Seventeen of the 22 display a 30-70 ms lag in the table's `Step 1` column**,
beside a status that never applied the bar to it. **A rendering wrinkle worth
knowing**: for err_316 that 47 ms is not a Step 1 measurement at all --
`worst_lag` reads `batch_state`'s `max_abs_lag_ms`, written by `batch_verify`
in the container, which is Step 2, while `step1_verdicts` holds null. The column
headed `Step 1` shows 47 ms for a file whose status says there is no Step 1
record, **and both are right about their own source.**

**No completion count is inflated** -- none of the 22 is `Resolved`. What is
absent is any signal that 22 files carry a measured lag above the quantisation
floor, and the branch built to give that signal is unreachable.

**THE PRECONDITION IS COVERAGE, NOT THE THRESHOLD.** The bar cannot fire until
some file gets a Step 1 verdict of `in_sync` or `out_of_sync`; all 40 that exist
say `unverifiable`. **These 22 are not waiting for a better bar -- they are
waiting for a Step 1 that returns a verdict**, which puts them beside the
22-file Step 1 backlog already held for decode.

**Bearing on the END CONDITION**: "Resolved (Step 1 under 15 ms AND Step 2 in
the container)" is currently unachievable for every file in the corpus, because
no file has a Step 1 verdict the bar can be applied to. **158's remedy applies
literally: fire the branch deliberately.** A synthetic row with
`step1_verdict = "in_sync"` and a 40 ms lag would prove it renders. **Its
absence today proves nothing about the corpus.**

## Step 1 computes every shared language, then discards all but the first

**Status**: measured, cause located, no code changed. Awaiting your ruling.

All 40 rows in `step1_verdicts.json` say `unverifiable`. That is the reason the
mission's 15 ms bar is applied to nothing (previous entry). The reason *behind*
it is one line.

    39  candidate contributed no track in this language; both compared tracks are master-derived
     1  no shared language

    language chosen:  fre 34   eng 5   None 1

`src/tools_batch/verify_unverified.py:38`

    shared = [l for l in ("fre", "eng", "jpn") if l in langs_m and l in langs_a]
    lang = shared[0]

The comprehension finds **every** language present in both files. The next line
keeps **one**. If the candidate contributed no track in that one -- deduplicated
as inferior, so both compared tracks are master-derived -- the tool writes
`unverifiable` and moves to the next file. **It never tries the second or third
shared language.** `fre` leads the tuple, which is why 34 of 40 chose it.

**The refusal itself is correct and should not be removed.** Comparing the master
against itself yields a drift near zero, and a near-zero drift would render
`Resolved` -- a false pass inside the bucket the both-steps ruling exists to
protect. The tool declines to measure what it cannot measure. The question is
only whether it declines too early.

**What is NOT yet known, and is being measured now**: whether any of the 39 have a
*second* shared language in which the candidate did contribute a track. If most
carry exactly one shared language, this line discards nothing and the 39 are
genuinely unverifiable. Agent 1 is counting audio streams per shared language,
ffprobe only, no decode and no ledger write.

**Two questions, and both are yours because either answer re-decides every Step 1
verdict in the corpus:**

1. Should Step 1 try **all** shared languages before returning `unverifiable`,
   rather than only the highest-priority one?
2. Is the fixed `fre > eng > jpn` priority the intended order, or should the pick
   prefer the language in which the **candidate** actually contributed a track --
   which is the only kind of pair Step 1 can measure at all?

**A correction to the record, since it was repeated in earlier notes**: this was
previously attributed to a `lang_pick or "jpn"` default. There is no jpn default
on this path. The order is fixed at `fre > eng > jpn` and the defect is
abandoning after the first pick, not the pick itself.

### Measured since the entry above was written

**The `shared[0]` reading is confirmed, with scope.** Agent 1 probed audio-stream
counts per shared language for the 29 files whose artefacts still exist, using
the tool's own probe rather than a reimplementation:

    3 shared languages  19      2 shared   9      0 shared   1

**Not one file has exactly one shared language**, so `shared[0]` discarded a real
alternative on 28 of 29. The split:

    a shared language QUALIFIES (output has more tracks than the master)    7
    no shared language qualifies -- unverifiable BY THIS TEST              21
    no shared language at all (err_13) -- genuinely impossible              1
    artefact gone, not measurable in either direction                      11

**On all seven the first pick was `fre`, and `fre` never qualified.** Not "the
first pick sometimes misses" but "on every file where a verifiable pair exists,
the first pick is not it".

**The two directions are not equally sound, and this bounds the 21.**
`out > n_master` implies at least one track is candidate-derived -- a pigeonhole
argument needing no ordering assumption. `out <= n_master` does **not** prove the
candidate contributed nothing, because the merge can drop a master track and keep
a candidate one. **The 21 are unverifiable by the test Step 1 uses; they are not
proven master-derived.**

**Four files carry the whole chain**: err_83, err_87, err_95, err_98 -- a
verifiable `eng` pair exists, Step 1 stopped at `fre`, the verdict became
`unverifiable`, and the mission bar's third branch was never reached.

**Correction to the correction.** The entry above said "there is no jpn default on
this path", which is right for `verify_unverified.py` and was stated too broadly.
The `lang_pick or "jpn"` default is real and lives in `reverify_external.py`,
fixed at `769fb80`. **Two language-selection defects, two files, not one defect.**

## The Step 1 verdict is read from a store four days older than the lag beside it

**Status**: measured. No code changed. This supersedes the entry above as the
larger blocker.

`master_summary.py:275`

    final_status(outcome, verdict, step1_ms, (step1.get(path) or {}).get("verdict"))

**Argument 3, the lag, comes from `batch_state.json` -- written today. Argument 4,
the verdict that decides whether the 15 ms bar is consulted at all, comes from
`step1_verdicts.json` alone.**

    step1_verdicts.json    40 rows, all unverifiable      last written 2026-08-30 00:33
    batch_state.json      315 rows, 81 with per_language  written today

`step1_verdicts.json`'s only writer is `verify_unverified.py`, which selects
targets by `outcome == "MERGED_UNVERIFIED"` **at the moment it runs**. The ledger
has moved out from under it:

    entries                        40        still MERGED_UNVERIFIED today   4
    became MERGED_IN_SYNC since    18        became FAILED since            18

**Coverage of the currently eligible set is 4 of 50, and 46 eligible rows have
never been audited.** (Not 275 -- 229 rows are outside the filter's scope by
design, and reporting them as a gap would inflate it sixfold.)

Meanwhile `batch_state` carries, today:

    rows with a per_language report   81     in_sync 43   out_of_sync 1   unverifiable 37
    rows with a real numeric lag      44     of those, above the 15 ms bar   10

### The obvious repair is the wrong one, and it was measured before being proposed

**Do NOT join `batch_state`'s `per_language` verdict into argument 4.** The
counterfactual table was computed read-only, with `master_summary.py` unedited:

    status                                   now   joined   delta
    Step 2 verified; no Step 1 record         75      22     -53
    Step 2 only; Step 1 could not measure     32      42     +10
    In sync, over the 15 ms bar                0       8      +8
    Resolved                                   0      35     +35

**35 files would become `Resolved` and 32 of them for one reason: `per_language`
is written by `batch_verify` at line 723, inside the row saved after
`api("POST", "/fusion", ...)` -- it is the container-side STEP 2 correlation.**
Argument 2 is already Step 2, falling back to `in_sync` whenever `outcome ==
"MERGED_IN_SYNC"`. **For those 32, both arguments would come from the same Step 2
run, and `Resolved` would mean "Step 2 passed, and Step 2 passed."**

`master_summary.py:150-158` records this exact failure as having happened once
already -- ten files counted Resolved or over-bar on the container's Step 2
verdict while their own Step 1 could not measure. **The join would re-create it
at 35 instead of ten.**

**17 rows move the other way and those are sound**: `MERGED_UNVERIFIED` rows whose
`per_language` says `unverifiable` go from *"no Step 1 record"* to *"Step 1 could
not measure"* -- strictly more accurate, and moving no file toward the end
condition. **A join that only improved counts would be suspect; this one demotes
17.**

**The source choice is moot**: the row's top-level `verdict` and the
`per_language` aggregate agree on all 75 rows where both exist.

### Twelve rows carry a lag over 15 ms; eight would render over-bar

    err_45 20.0   err_211 17.7   err_218 32.2   err_290 30.0
    err_299 20.0  err_300 20.0   err_302 30.0   err_316 46.9

The four that do not, failing three different ways:

- **err_78** -- a ledger lag of 16.7 is **overridden to 0.0 by
  `external_verdicts`**, so it renders `Resolved`. **The larger measurement loses
  to the smaller by source precedence, not by evidence**, and the override comes
  from the store whose language may have been the guessed `jpn`. This is the one
  to look at hardest.
- **err_83 (40.0) and err_101 (20.0)** -- no `per_language` at all, so nothing
  reaches them and their lags stay unchecked.
- **err_293 (70.1)** -- exits at the `out_of_sync` branch. Correct.

**The membership path stays unreachable.** err_293 is the only `out_of_sync`
`per_language` row and its argument 2 is `out_of_sync` too, so `final_status`
returns before the bar. The path `test_65` pinned needs argument 2 `in_sync`
beside argument 4 `out_of_sync`, and **no row in the corpus has that
combination** -- before the join or after it.

**So the question for you is not whether to join, but what a real Step 1 costs.**
46 eligible rows have never been audited and 36 of the 40 audited have changed
outcome since. **Nothing has been joined and nothing has been written.**

**One defect in two stores, found by two agents who were not looking for it**:
`step1_verdicts` is a four-day-old snapshot of a ledger that moved, and the 15
stale external verdicts are a store with no timestamp that cannot tell a current
verdict from one about a replaced artefact.

**CORRECTED, measured afterwards: the artefacts are not gone.** This was written
as *"a verdict outliving the thing it was about"* on the strength of 11 files
reported unmeasurable and 7 ids shared between two populations. Measured three
ways across 106 verdict rows: **83 artefacts present at a recorded path, 23 at
the path the project derives, 0 unlocatable. Zero rows in any store carry a
recorded `merged_file` path that is missing**, and the 141 ledger rows that are
`FAILED` with genuinely no output anywhere carry no verdict in any store. **All
seven named ids -- err_9, 12, 46, 51, 59, 63, 70 -- have an output on disk.**

**It was a field's absence read as a file's absence** -- 185 `FAILED` rows have no
`merged_file` *field*, and 44 of them have an output at the derived path. **The
same mistake is already on record in `step2_only_class.md`, where all 27 turned
out to have published outputs.** The agent that reported the 11 said explicitly it
could not measure in either direction, and was right to; this supplies the
measurement rather than contradicting the report.

**The real defect is one step over.** Every artefact is present, and **nothing in
any store can show that the file there now is the file the verdict described** --
no path, size, mtime or hash recorded at judging time. *"The artefact still
exists"* and *"the verdict still describes it"* are different claims and only the
first is checkable. The 27 `in_sync`-beside-`FAILED` rows are both true, of
different runs; which run the surviving file came from is the question no store
can answer.

**And `written_utc` would prevent none of it.** A date lets a reader *notice*,
which is worth having and is a different claim. Case by case: the 27 stale
verdicts, **yes, all 27** -- every ledger row carries `recorded_utc`, so the
comparison is possible on both sides. `step1_verdicts`, **partly, 25 of 40** --
the other 15 ledger rows have no `recorded_utc` to compare against. The
"artefact gone" 11, **no** -- nothing about them was a dating problem. **A partial
remedy, recorded as one.** The third case needs artefact identity, not a
timestamp.

## Two thirds of "could not measure" is an unconfigured instrument, and a fix does not reach a running sweep

**Status**: measured. No code changed.

`batch_state`'s per-language `unverifiable` records are two different facts under
one word:

    31 records / 29 files   external probe failed: tools.software['ffmpeg'] is not configured
    14 records              structural refusals (all tracks master-derived, bit-identical)

**The refusal is correct** -- `_binary()` declining to guess a PATH binary for a
measurement is the owner's standing rule working. **The defect is that the
refusal reached the verdict slot**, where it is indistinguishable from a finding
about the file. `reconcile_verdicts.py` already documents the historical half.

**What is new is the timing.** The fix is `7744aa3`, 2026-08-31 20:47 UTC.
Affected rows:

    2026-08-31T20:48:57Z   one minute AFTER the fix landed
    ... 22 files ...
    2026-09-01T02:39:03Z   the last one, six hours later
    2026-09-02             15 rows with per_language, 0 affected

**Twenty-four language-records across 22 files were produced by pre-fix code
after the fix existed** (7 records / 7 files precede it), and all 24 carry the
PRE-FIX wording of the message, which the current code no longer emits --
independent confirmation that the running process, not the checkout, is what
produced them. **An earlier draft of this entry said "fourteen"; that was a
miscount of the same list.** A
long-running `batch_verify` holds the imported module in memory; the commit lands
and the sweep goes on refusing until it is restarted. The config loads correctly
today -- verified directly, `_binary('ffmpeg')` resolves.

**The deployed container is gated on `git_commit == HEAD` for exactly this
reason. There is no equivalent gate host-side.**

**Correction, measured after the above was written.** A per-language row *does*
carry a code stamp -- `build`, non-null on all 75 -- so "no stamp" was wrong.
But `batch_verify.py:533`/`:559` set it from `deployed_build()`: **it is the
CONTAINER's SHA, not the host module's.** The affected 29 span two container
builds (`afa72e7` x27, `0419069` x2) while sharing one host defect. **The stamp
that exists is the wrong one, which is worse than none, because it looks like
provenance.** `batch_verify.py:155-157` already records this hazard for a
different case -- `build == da24bac` silently spanned two runs, which is why
`recorded_utc` was added at all.

**And the affected set does not need dating from a commit at all.** The two
populations do not interleave:

    affected window   2026-08-31T18:22:06Z .. 2026-09-01T02:39:03Z   29 rows
    clean window      2026-09-01T06:32:34Z .. 2026-09-02T07:38:49Z   30 rows
                      a 3h53m gap, nothing in it

**Read the note, not the clock.** 16 further clean rows predate the stamp and are
unplaceable in time, but are clean by their own content and cannot widen the set.
**The last clean row is from the sweep running now**, so today's `batch_verify` is
post-fix in the only way the record can show.

**A host-side gate cannot copy the container's.** Gating on `git_commit == HEAD`
does not work here, because a running process's code is not its checkout's code
-- that is the defect itself. What would work is a **row-level stamp of the
writer, taken at import time by the process writing the row**, so a long-running
sweep carries the revision it actually loaded.

**Two questions:**

1. Should a host-side sweep record the commit it is running, the way the
   container's `/health` reports `git_commit`?
2. Should an instrument fault be a distinct verdict rather than `unverifiable`?
   `reconcile_verdicts.py` already keeps `FAULT_MARKERS` to recover the
   distinction after the fact -- **a reader reconstructing at analysis time what
   the writer knew at write time**, which is the same shape as the missing
   provenance field on the merge path.

### The answer to question 1, measured: zero of 110

    lag-carrying rows                                                  110
    whose surviving figure can be dated to the code that produced it     0

**Every version stamp in this corpus dates the container's MERGE. Every lag
figure is produced HOST-SIDE. No host-side tool records what it was running.**

    figure's source        n     row `build`   row `recorded_utc`   job log `git=`
    batch_state            39        33              19                  16
    external_verdicts      53        44              42                   0
    no_output_verdicts     18        15              13                   1

**The two right-hand columns are not what they look like.** `batch_state.build`
is `deployed_build()`, whose own docstring says *"from the CONTAINER's `/health`,
never from local HEAD"*. The `(git=)` stamp referred to in TASK.MD's T4 row is the
container's job-log line -- 70 of 131 archived logs, three distinct stamps, all
from the 2026-08-30 window -- and it dates the same event. **`batch_verify` POSTs
to `/fusion`, waits, then runs `lag_ms`/`lag_ms_across` with ffmpeg locally: the
correlation is host-side. Both stamps are real, both accurate, and both date the
wrong half.**

**71 of the 110 have no row-level dating at all.** `external_verdicts` entries
carry `{lang, result}`; `no_output_verdicts` carry `{error_id, lang, merged,
result}`. No version, no timestamp. They can be placed only by the store file's
mtime, which dates the last write to the whole file. **The `build` and
`recorded_utc` counts beside them above belong to the `batch_state` row for the
same file -- a different measurement, at a different time, by a different
process.**

**What a reader may conclude:**

    which image performed the merge      92 of 110
    roughly when the merge ran           74 of 110
    which code measured the lag           0 of 110

**A reader comparing two figures -- which `master_summary.py:259-264` does
silently -- has no field that would let them prefer either. The precedence is
store order because store order is all there is.**

**The container records its build on every job by three independent routes for
the half of the work that is not in question. The half that produces every number
in the summary table has no equivalent.**

**Attribution from the data is currently the only method available**, and it works
where metadata does not:

    no_output_verdicts    57 of 57 sample lags on multiples of 10 ms,  7 distinct
    external_verdicts    167 of 226                                   48 distinct
    batch_state           87 of 199                                   47 distinct

A store whose every value lands on the 10 ms envelope bin was written without
`refine_peak_index`, whatever its timestamp says. **One field would have settled
err_78 in a line, instead of by reasoning from residues mod 10.**

## "No samples" was asserting a measurement that never ran, on 76% of the corpus

**Status**: FIXED and landed. Reported because it changes a count you have been
shown.

`reconcile_verdicts.evidence_class` was called as
`evidence_class(record.get("per_language"))`. `.get` returns `None` for a missing
key and `not per_language` is true for both `None` and `{}`, so a record that was
**never measured** and one that **measured and found nothing** returned the same
label.

    per_language   ABSENT 234    NULL 0    EMPTY 6    PRESENT 75

    before   241  no samples
    after    234  never measured
               7  no samples

**`no samples` asserts that a measurement ran.** For 234 of 315 rows it never
did. The other counts are unchanged: 44 multiple-windows-agreeing, 29 instrument
fault, 1 one-window.

**The fix is a function that still holds the record, not a cleverer test.** The
distinction was destroyed at the call site, one `.get` before `evidence_class`
was entered, so it had nowhere to live inside it. `evidence_class` keeps its
meaning for a mapping that exists.

**This is the fourth instance of the same defect found today, and the one with
scale**: an AST check satisfied by `error_id: None`; a test suite read by eye
reporting green while a test failed; a `.get()` that would have reported "no
stamp" for an absent key; and now the dominant cell of the ledger's own summary.
**Each time a reader with two outcomes described a world with three, and the
missing outcome was always *the instrument did not run*.**

## The residue method: calibrated, and its author recommends against relying on it

**Status**: measured, reported, nothing built on it.

Dating a store by whether its values land on the 10 ms envelope bin was the only
attribution available for the 71 rows with no metadata. Calibrated against a
store known to be refined:

    AFTER 6dbbc8c -- KNOWN REFINED     127 samples, on the grid  15   0.118
    undated (older era)                 72 samples, on the grid  72   1.000

**All 15 grid hits in the refined store are exactly 0.0.** Non-zero grid hits
under refinement: **zero in 127**. So the discriminator is *a NON-ZERO multiple of
ten*; an exact zero is zero at every resolution and carries no information. **The
raw rate looked like 44% because it mixed both eras in one denominator.**

    false-positive bound (rule of three, 95%)   per sample 0.0236   per ENTRY 0.0698
    minimum n, derived                          2 entries, at both 95% and 99%

**The per-entry bound is the correct one** -- three windows of one file share a
true lag, and treating samples as independent overstates the method threefold.
**A single non-zero grid value carries 93.0% confidence, short of 95%.**

Applied:

    store                entries   REFINED   PRE-REFINEMENT   UNINFORMATIVE
    no_output_verdicts       19        0            10               9
    external_verdicts        77       22            17              38

**`no_output_verdicts` is pre-refinement decisively** (`0.0698^10` ~ 3e-12).
**`external_verdicts` contains BOTH eras**, so its mtime tells a reader nothing
about any individual row. **47 of 96 entries are all-exactly-zero and
unclassifiable at any sample size -- and not a random half: they are the ones
reporting zero, the value most likely to be relied on as a pass.**

**Inversion: flag only, but the flag carries a bound.** The refined figure cannot
be recovered -- `refine_peak_index` interpolates from a correlation array that is
not stored -- but the bin is 10 ms, so a pre-refinement value `v` means the true
lag lay in `v ± 5 ms`. **For err_78 the overriding `0.0` means "within ±5 ms" and
the ledger's `16.7` is outside that interval: irreconcilable as one measurement
even before the language difference.** The 71 are not recoverable data, they are
**boundable** -- and the bound is wide enough that anything within 5 ms of the 15
ms threshold cannot be placed on either side of it.

**The agent that built it recommends against relying on it**, and I agree: it
dates a code *era*, not a tool; just under half the entries are unclassifiable in
principle; and it is an inference standing in for a field, which is the shape of
every defect found today. **Its value is as evidence FOR the field, not as a
substitute for it. It measured how bad the gap is; it should not be used to live
with it.**

## Retrospective provenance reaches 37 of 64, and the store grading the corpus is two days behind it

**Status**: measured, nothing run or re-classified. One standing-rule violation
found, routed to its owner.

**A finding reported this morning is WITHDRAWN by the agent that made it.**
`provenance_classification.json` was last written **2026-08-30T22:12:28Z**, and
**all 32 `in_sync` rows that have a record sit on outputs whose mtime is later.**
Every provenance record backing an `in_sync` verdict describes an artefact that
has since been rewritten at that path. The split *2 store-confirmed / 11
structural-only / 20 all-indeterminate* is an accurate reading of the store and
**not a statement about the files on disk**.

**How much is genuinely recoverable**, asking not whether the artefact exists but
whether it is the artefact the verdict graded:

    artefact predates its verdict, all three files present   37   (28 files)
    no recorded_utc -- cannot show the artefact matches      24   (16 files)
    artefact REBUILT after its verdict                        4    (2 files)

**37 recoverable, 24 undeterminable, 4 lost.** The 24 are a missing timestamp
rather than a missing file -- a third state, counted as neither.

**Coverage is a scheduling fact, not a physical limit.** All 32 unrecorded rows
resolve today: output, master and candidate all present, all 22 files built after
the store's last run. **But `--all` does `if str(eid) in results: continue`, so a
run today would add 47 and refresh none of the 82.** The store is append-only and
cannot self-heal. That is the one mechanical thing between this corpus and an
answer.

**The 20 "all indeterminate" rows are the MOST recoverable, not the least.** Every
one gives the same reason -- `codec differs from the master's track` -- and it is
a skipped branch:

    if track["codec"] != master_codec:
        verdicts.append({... "indeterminate", "why": "codec differs ..."})
        continue                      # never reaches stage two

The justification is sound **about stage one**, a byte digest: a byte comparison
across codecs proves nothing. **Stage two is an envelope correlation on decoded
PCM at 8 kHz mono -- the codec is gone by then.** A stage-one argument applied to
a stage-two measurement.

**But the `continue` must not simply be deleted.** `SAME_RECORDING_PEAK = 0.98`
sits in an empty band from 0.9718 to 0.9833 measured over 45 tracks **that all had
matching codecs** -- and two signals with **no sample in common**, sharing only an
amplitude envelope, measure **0.979**, inside that band. **The band may be empty
because of what was in it, not because the quantity separates.** Opening those
rows needs the band re-established on the codec-differing population first.

**Two defects in `src/diagnostics/tools/provenance_classify.py`, routed to the
agent that owns it, neither fixed here:**

1. `resolve()` checks the output and the master but **not the candidate**;
   `classify()` then treats an absent candidate as an empty stream list, so every
   language falls to the structural gate and reads `master_derived_by_construction`
   with `judgeable` false. **A missing candidate is silently indistinguishable
   from a genuine master-derived verdict.** Latent -- no file in the 22 is
   affected today.
2. The module invokes **bare `ffprobe` and `ffmpeg` literals rather than
   `tools.software[...]`** -- your standing rule, and the same defect
   `batch_verify._binary` exists to prevent.

**What this does to the merge-time proposal: strengthens it, for an unexpected
reason.** Every recoverable row is recoverable only because an mtime happens to
predate a timestamp -- **luck about what has been rebuilt, not a property of the
system.** The 24 unstamped rows are lost for want of a field nobody wrote; the 4
stale ones for want of a build identity on the verdict. **A `source` field written
by the assembler would need no artefact to still exist, no mtime comparison and no
threshold.**

**Recommendation**: run the retrospective classification on the 37 -- it is cheap
and it is the only evidence available for verdicts already issued -- **but its
coverage must not be read as a fix.** Carry "37 recoverable, 24 undeterminable,
4 lost", never a hopeful 106.
