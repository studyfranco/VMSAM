# Zone A — repair specification

**What to build.** Permission to build it is in `WRITE_ZONES.MD`; this file does
not grant any.

Draft: written from campaign 1's measured record and the owner's design decisions.
The owner owns it — correct it rather than working around it.

---

## 1. The contract

Repair happens **when a file is rejected**, inside `remove_not_compatible_video`.
Not before the merge, not during it. The rejected file is repaired against
`best_video` and rejoins the merge through the MD5 group.

`best_video` is a **video object** on both call sites.

Available without any change:

```python
video_obj.delayFirstMethodAbort[best_video.filePath]   # -> [1|2, delays]
```

`first_delay_test` fills it at 12 points. It tells you *why* the pair was refused
and which delays were seen — you do not need to re-run the correlation that just
failed.

Rejoining the merge: attach the repaired object to
`best_video.sameAudioMD5UseForCalculation`. It is then consumed by
`generate_merge_command_common_md5`, which never goes through the delay machinery.

**Three requirements, each verified against the code:**

1. `delay_same_md5_audio` **must be a `Decimal`** — `Decimal('0')` when the repair
   already aligned the file. It defaults to `None` (`video.py:55`) and
   `mergeVideo.py:1426` computes `delay_same_md5_audio + delay_to_put`:
   `None + Decimal` raises `TypeError`.
2. The object must have had `get_mediadata()` — `generate_new_file` reads
   `audios` and `video['Duration']`.
3. The temporary file must survive until `mkvmerge` runs. Do not put it anywhere
   that is cleaned between the repair and the merge.

## 2. What closes the zone

**Not** "the code is in the zone and the tests are green." The previous campaign
produced five pieces of code that were written, reviewed and unit-tested green,
and that **nothing ever called** — a guard, a refined measurement, a policy
branch, an order check, a subtitle repair. Every suite passed. A unit test calls
the function directly, which is exactly what production did not do.

Zone A closes when, on a real error file that used to fail:

1. a merged output is produced, and
2. the repaired track is **present in the final mkv**, and
3. the agent produced the expected file itself and compared it against the output.

Point 3 is the one that cannot be faked by dead code.

## 3. Why two lines in the tagged zones exist

**`get_good_frame` — `calculated_delay = float(calculated_delay)`.**
`test_if_constant_good_delay` returns a `Decimal`, and the division by 1000 in
`get_best_video` keeps that type, while every timeline value in `get_good_frame` is
a float seeded by `get_fps()`. The first addition raised `TypeError`,
`compare_video.run` caught it and recorded the pair as *not compatible* — so the
whole no-forced-best-video path silently dropped every candidate. Converting the
rest to `Decimal` buys nothing: `1/23.976` is not exact in decimal either, `int()`
truncations dominate, and the result is snapped to a frame by `round()` downstream.

**`video.py` fallbacks — the `BaseException` re-raise.** `get_less_channel_number`
and `get_less_sampling_rate` answer a failure with a fabricated plausible value
(`"2"`, `"44100"`) from a bare `except:`. Those values are indistinguishable from
real ones everywhere downstream, and the sampling rate decides how the comparison
audio is extracted — every correlation then rests on it. Both now log under
`tools.dev` and re-raise `KeyboardInterrupt` / `SystemExit`. If those log lines ever
appear in a run, that is a finding.

### The suites cannot see a join, and that was demonstrated twice in one evening

**§2 exists because five pieces of code were written, reviewed and unit-tested green
while nothing called them.** Two instances arrived the same night, from the other side of
the same seam, and both were caught by reading rather than by any test:

- A call passed `reference_stream` to a function that did not accept it. `py_compile`
  passed and **all four guard suites passed**, because the suite stubs the assembler and
  never calls it with a plan. **Production would have called it and raised on every
  file.**
- `verify_output_file` was **defined, tested, and never called** — in the function whose
  entire purpose is to stop unnoticed nothing-happened. Found by grepping for the call
  site.

**A unit test calls the function directly, which production did not; a guard suite stubs
the boundary, which production does not.** Both failures live at the join, and neither
kind of test is built to see one. **Check cross-module calls by signature, not by
compile** — and know that signature-checking verifies arity and names, not types.

## 3b. The reconciliation window is a coarse sort, and it must say so in quanta

**The `< 127` comparison is stage one of two, by design.** The owner's framing: the
first pass exists to make a large, cheap split between compatible and incompatible
files; **the second and heavier method then sorts the groups that carry a very small
offset.** So a fixed window letting a sub-quantum disagreement through is not, by
itself, the defect — *the open question is whether stage two catches what stage one
waves past, and nobody has measured that.*

**What is a defect is that the comparison is invisible and expressed in the wrong
unit.** It is written three times in `first_delay_test` as a bare millisecond literal,
judged against a grid step of `int(lengthFile / n_points * 1000)` that is **file
dependent — measured 124, 125 and 138 ms, never a fixed 125**. Nothing at any call site
says so, and nothing is logged.

**Requirement, in the owner's own shape:** log the comparison as
`(concordance, windows of difference, quantum)` — for example `(0.97, -1, 125)`.
**Count windows, not milliseconds.** A threshold in windows is the same threshold on
every file; a threshold in milliseconds is a different one on each. And **which files
are affected must be answerable from the logs**, not reconstructed afterwards.

## 4. Same tag is not same recording — check concordance before building

**A language tag groups recordings that are not interchangeable.** VFQ, VFF and VFI
all carry `fre`. So do a dub and a redub. Two rules follow, and the second is a
defect in the code today.

**Before building, ask whether the master already carries the same recording.** If it
does, the chimeric track is work that will be thrown away — `keep_best_audio` drops a
fabricated track in favour of an intact one, which is the correct outcome and means
the assembly was wasted. **Test for concordance on the two tracks rather than on
their tags.** High correlation with the master's same-language track means the same
recording: do not build.

**And a fabricated track may only lose to an intact one carrying the SAME
recording.** The rule in `keep_best_audio` fires before every codec, channel and
bitrate comparison and asks nothing about content:

    elif audio_1.get('fabricated') and (not audio_2.get('fabricated')):
        audio_1['keep'] = False

Its two call sites do not behave alike. One passes tracks already established
compatible by a `validation` step; the other passes **every audio of one language,
unvalidated**. On that second path a chimeric VFQ and the master's intact VFF are
both `fre`, and the chimeric one is dropped — **losing a recording the output does
not otherwise carry.** The rule was written to stop a fabricated track winning a
codec comparison. It also silently destroys a variant, which was never intended.

Until the fabricated rule is conditioned on sameness, the concordance test is not an
optimisation but a guard: **not building a track the tag-blind rule would delete is
the only thing keeping the variant.**

## 4b. When the streams of one file disagree, refuse — owner's ruling

**Ratified.** `SPEC_ZONE_A` defined the contract for a *track* and said nothing about
a file carrying several. Faced with that silence an agent chose to **decline repair
when a file's streams disagree**, knowing it was the cautious reading and that it cost
yield, and reported that no rule had asked for it and no rule would have stopped the
opposite choice. The owner has ruled the choice correct and it is now the rule:

> **When there are two comparison audios and the method that would repair the
> analogous audios diverges, do not merge — a divergence means there is a problem.**

**And the owner states it a second time as a validation criterion, which is stronger
than a refusal rule:**

> **The repair plans must be identical to validate a repair.**

**Maintained at absolute rigour, and the owner has priced it explicitly.** The rule
refuses the file the campaign had been leading with — two of its language plans differ
by 41 ms, and it verified at 1.125 ms and 8.25 ms after per-stream offsets were applied.
**The agent that produced it said the price out loud before it was paid rather than
letting the ruling land quietly, and that is why the ruling is informed.**

**But it was not a validated success, and the owner says so from his side:** it never
went through the complete pipeline in the test container, so nothing end-to-end ever
accepted it. **41 ms is a few frames**, and he suspects the underlying cause is PAL —
25 fps against 24000/1001 — which would make it a rate problem wearing a divergence's
clothes. *A result nobody put through the pipeline is not a result the pipeline
produced.*

**Read that before treating per-stream plans as the fix.** Two files decline because
one plan is correct on the language it was measured on and wrong on another; they do
repair when each stream gets its own plan. **That is not the ruling.** Identical plans
across streams is the *condition of validity*, so a file needing different plans per
stream is a file whose streams disagree — which is the signature this section says must
not be merged over. **The per-stream capability is not the remedy for divergence; the
divergence is the finding.**

**The owner's documented case, and it carries an open verification.** In error 24 the
two Japanese tracks are **250 ms apart**. That is attributed to a bug of the time which
produced a drift of a few frames. **Whether that bug still exists has not been
established, and establishing it is a task, not an assumption.** Until it is, a
divergence between streams is exactly the signature that must not be merged over: the
repair would be built on top of a defect rather than around it.

Two files declined on this judgement do repair correctly with per-stream plans. **That
is not an argument against the rule** — it is the yield the rule knowingly costs, and
the owner has priced it.

## 4c. Filling a gap: the comparison language, never silence — MANDATORY

**When the master carries no track in the candidate's language, fill from the best
track of the comparison language.** Not silence. The owner's word is *mandatory*, and
he gave it knowing the cost.

**The measured exposure was total.** `find_master_audio_for_language` looked the
candidate's language up in the master and returned `None` otherwise; **the
comparison-language branch was never implemented at all.** So the three outcomes the
specification describes were only ever two — same-language, or silence. Of the tracks
filled with silence, **42 of 42 had a master track in a language shared with the
candidate.** Every one of them had a source and got silence instead. It was not a rare
fallback; it was the only thing that could happen once the languages differed.

**And the cost is stated here because it was stated to him before he ruled.** A French
gap filled with Japanese **is audible as a language change**. The agent that wrote the
silence behaviour wrote it deliberately for that reason, recorded in its own module that
a cross-language fill *"passes every structural check and is indefensible on
listening"*, and put that in front of the owner rather than implementing quietly.
**He ruled for the fill anyway.** That is his call to make, and it is written down so
nobody re-opens it as a defect.

**Which track of the comparison language:** call `keep_best_audio`'s existing ranking.
One definition of *best* in the system is worth more than a locally better second one —
two parallel subtitle classifications already diverged here for exactly that reason.

## 4h. The fill must EQUAL the gap, to the frame — owner's ruling, and the project's objective

### The objective of the chimeric path, in the owner's words

> ***"Repérer À LA FRAME PRÈS les zones ajoutées et enlevées d'un fichier candidat par
> rapport au master. Si c'est une zone supprimée, on supprime la zone à la frame. Si c'est
> une zone ajoutée, on utilise soit la même langue soit la langue de comparaison pour
> combler."***

**Three operations, and the first one governs the other two.** *Locate to the frame; then
cut at the frame, or fill at the frame. A boundary that is not frame-accurate makes both
of the following operations wrong by the same amount, in opposite directions — the cut
removes material that was good, and the fill covers material that was good.*

### WHERE the refinement belongs — owner's ruling

> ***"Le code de la création des fichiers chimériques doit AFFINER ce que les étapes de
> comparaison ont vu."***

**The comparison stage produces COARSE ZONES. The chimeric builder NARROWS them to the
frame. It does not adopt them.** *That places the refinement in `merge_video_chimeric.py`,
downstream of `change_point_locator`.*

**The output contract: the produced timeline IS the master's timeline at zero offset.
Candidate material lying OUTSIDE the master's extent is removed — head and tail. Candidate
material lying INSIDE it is USED.**

### MEASURED: the builder discards usable candidate audio and substitutes the master

    HEAD   true excess, outside the master        22 815.13 ms   remove: correct
           candidate INSIDE the master's extent   220 000.00 ms   DISCARDED
           master inserted in its place           220 000.00 ms

    TAIL   true excess, beyond 1432056            13 947.01 ms   remove: correct
           candidate INSIDE the master's extent   152 056.00 ms   DISCARDED
           master inserted in its place           152 056.00 ms

> **BOTH EQUALITIES ARE EXACT. This is not a rounding artefact, it is a SUBSTITUTION.**

**372 SECONDS of candidate audio that lay inside the master's extent were discarded and
exactly as much master audio inserted** — *six minutes of twenty-four, and under §4c those
six minutes are in another language on every track the master does not carry.*

**A SECOND defect, not the bracket one.** *The interior gap of exactly 100 000 ms is a
bracket never refined. The head and tail lie OUTSIDE every retained plateau — there is no
change point there to have a bracket.*

### MEASURED: the fill region is not a measurement of the media

**`change_point_locator.py:845-848` builds the plan from the bracket, not from a point:**

    end_ms  = change["bracket_low_ms"]      the candidate segment ENDS here
    widened = change["bracket_high_ms"]     the next one BEGINS here
    boundary = widened

**Everything between the two is a gap, and a gap is filled from the master.**

> **THE FILL REGION IS THE LOCATOR'S RESIDUAL UNCERTAINTY, MATERIALISED AS INSERTED MASTER
> AUDIO. The less sure the locator is, the more is inserted** (`vmsam-dev-3`).

**And the floor is audible even when everything works.** *`bracket_high = first_after +
REFINE_WINDOW(8)` with `first_after >= last_before + REFINE_STEP(4)`, so a SUCCESSFULLY
refined bracket already yields a 12-second gap. An unrefined one is `region_end +
PROBE_WINDOW(60)` against coarse probes `PROBE_STEP(40)` apart — about 100 seconds.*

**On the first artefact the owner listened to, the interior master region is EXACTLY
100 000 ms:**

    master fill regions, read off the emitted plan
      head        0 - 220000        220 000 ms
      INTERIOR  680000 - 780000     100 000 ms   == 40 000 + 60 000, TO THE MILLISECOND
      tail     1280000 - 1432056    152 056 ms

**`PROBE_STEP_SECONDS(40) + PROBE_WINDOW_SECONDS(60)` is the signature of a bracket that was
NEVER REFINED.** *One hundred seconds of the master's language inserted into the middle of
the candidate's timeline, and its width is a property of the instrument rather than of the
media. That is what the owner heard.*

### The flag that says so is written and never read

    change_point_locator.py:803    "bracket_is_bound_only": not narrowed

**One occurrence in `src/`. The assignment. Nobody reads it.** *The plan therefore cannot
distinguish a 12-second refined bracket from a 100-second unrefined one, and neither can
any reader of the log: the emitted `plan` line carries the piece boundaries and NOT the
bracket widths, NOT `bracket_is_bound_only`, NOT `step_ms`.*

> **THE LOCATOR KNOWS WHEN ITS ANSWER IS COARSE. IT WRITES IT DOWN. NOBODY ASKS.**

*And `change_point_locator.py:518` already describes the class — "every bracket collapsed to
bound-only and all precision was lost". The owner's file is the real instance that comment
was missing.*

### Required, in order

1. **EMIT the bracket.** `bracket_low_ms`, `bracket_high_ms`, `bracket_is_bound_only`,
   `step_ms`, per change point, on the `plan` line. *A fill region whose width is the
   instrument's uncertainty must declare that width, or a listener is the first to learn it.*
2. **REFINE TO THE FRAME — AND WIRE THE ENGINE, NOT THE WRAPPER.**

       src/frame_compare.py   THE pHash ENGINE. 64-bit DCT, band alignment,
                              scene fallback. NOT retired. Clean.
       src/get_cut_time.py    RETIRED. Its own header: "Do not import, do not
                              build on this file... it has no importer and MUST
                              NOT ACQUIRE ONE." Ruled dead weight by TASKS/005.

   **An earlier revision of this section told the fleet to wire
   `get_cut_time.py:363 _refine_cut`. That was wrong and is retracted**
   (`vmsam-dev-3`, against a plan already in flight). *That module has eight sites that
   raise at runtime after `da1c191` stripped their `[0]` subscripts — deliberately left
   raising, on the principle that a file which raises is safer than one that runs and is
   wrong — and `_refine_cut`'s own audio fallback divides a `second_correlation` offset
   that is ALREADY IN SECONDS by the frame rate, wrong by a factor of about 24.*

   **"This is wiring, not research" is true of the ENGINE and false of the CALLER.**

   **THE REFUSAL CRITERION EXISTS, AND IT IS A PROPERTY OF THE INPUT — measured
   2026-09-04, and it moves the verdict on this engine upward for the third time.**

       REAL   removed span S=98 :   3 distinct hash values,   0 LOST
       SYNTH  removed span S=98 :  93 distinct hash values,  25 LOST

   > **A CUT IS DETECTABLE ONLY IF IT REMOVES A HASH VALUE THAT EXISTS NOWHERE
   > ELSE IN THE FILE.**

   *One fixture's cut removes NO information the hash can represent: all three
   values still exist elsewhere, so every reference frame finds a distance-0
   partner and deposits nothing.* **Invisible at any band, under any criterion —
   not a band effect, not a threshold effect, not a statistic effect. And the
   join IS a real edge, Hamming 34: the discontinuity is there and the frames
   either side are individually unremarkable.**

       DETECTABILITY IS COMPUTABLE BEFORE THE DETECTOR RUNS, FROM THE REFERENCE
       ALONE: take the span you intend to detect and count how many of its hash
       values exist nowhere else. ZERO MEANS NO AMOUNT OF TUNING WILL HELP.

       AND THE LOSSES MUST BE CONCENTRATED -- R9, closed the same hour:
         dev-1's span loses 8, ALL EIGHT CONSECUTIVE in the last 0.3 s before
           the join. They pile into one index and make the spike.
         dev-3's real arm loses 0        -> invisible under any criterion
         dev-3's synthetic arm loses 25, SPREAD THINLY, and still does not
           separate: peak 12 against 12, argmax 94 s from the cut.

       SO THE CRITERION HAS TWO TERMS, NOT ONE: how many values are lost, AND
       whether they are contiguous. Counting alone would have passed the
       synthetic arm, which does not work.
       Generation, seam, keyframe alignment, band and statistic were each
       eliminated before this, and none of them was it.

   **So the engine CAN decline rather than return an argmax it has no basis for
   — and the test is cheap and runs before a single hash is compared.** *Every
   search for a refusal STATISTIC failed because the answer is not in the cost
   field; it is a property of the input.*

   **TWO INDEPENDENT FAILURES, AND BOTH MUST BE STATED OR THE FIRST FIX LOOKS
   LIKE A FAILURE:**

       1  the cut must REMOVE SOMETHING THE HASH CAN REPRESENT
       2  the criterion must find deposits that are SMALL AND LOCALISED

   *The synthetic arm loses 25 values and STILL does not separate — peak 12
   against 12, argmax 94 s from the cut.* **A position-sensitive criterion fixes
   only the second, and neither fix alone makes the engine work on a source
   where there is nothing to find.**

   **THREE THINGS TO ESTABLISH BEFORE WIRING ANYTHING:**

   - **The engine may never have run.** *Its only importer is the retired module.* **Run it
     once on an equal-FPS pair and look at what it returns.** *Wiring a never-executed
     engine into the repair path is the same bet as a report module with no caller.*
   - **THE FPS GATE EXCLUDES THE POPULATION THAT NEEDS THIS MOST.** *The existing pHash
     branch requires `abs(fps_main - fps_tgt) < 1e-6` — EXACT equality. 23.976 against
     23.976 refines; 23.976 against 25.0 never does.* **On a PAL-versus-film pair the
     frame-accurate branch is never taken, and files with a speed relation are exactly the
     ones a frame-accurate refiner is for.** *Any wiring that inherits that gate inherits
     the exclusion.*
   - **What the achievable precision is when the gate fails.** *Scene detection, then
     nothing usable. State it; do not let a fallback silently define the tolerance.*
3. **Then shrink the analysis margins.** *Not before: shrinking them while the boundary is
   the instrument's uncertainty changes the uncertainty without measuring it.*


## 4d. A repair proves the file it produced, not the track it built

**A repair that reports success and ships a truncated file has not succeeded.** An
observed run reported *7 audio and 24 subtitle tracks rebuilt, 0 declined, 0 failed* and
produced a file with **nothing after 21:21** — audio and subtitles truncated. Nothing in
the repair noticed.

**This is the structural gap, arriving as an instance.** The verifier checks *the track
it built*, not *the file that shipped*, and it does not probe master-filled regions at
all. That is why *"damaged is 0"* was never a measurement, and it is why it stayed
uncontradicted: **the instrument that decides "repaired" is the same one that would have
to detect "damaged", and it never looks at the artefact.**

**So the acceptance condition is on the output file**: every track present, every track
running to the expected duration, no truncation. A count of rebuilt tracks is a
statement about work done, not about a file.

## 4e. What a repair must log

**The repair currently reports counts and nothing else, and counts cannot be audited.**
The owner's requirement: *who, what, how — what was done to the file, and at which
timings.* A line saying *"1 audio and 4 subtitle tracks rebuilt"* leaves no way to tell
a correct repair from the truncating one above.

At minimum, per track: what was cut and what was added, **with the timings**; where each
filled region came from — same-language master, comparison language, or silence — and
the language actually used; the plan applied, and the residual measured after applying
it. **A skip is a decision and gets a line too** — one observed run skipped an element
as *"empty"* with no indication of what was skipped or why that mattered.

## 4f. The repair is TWO STAGES and a set of metrics — not a tree of cases

**The owner's correction, and it is aimed at how this campaign has been working.** Three files
of one release were measured as three investigations and produced three beautiful maps and
**zero general criterion**. His instruction: *stop handling each case as an exception. Metrics,
and simplify.*

### Stage 1 — find the rate transform. Stage 2 — handle the additions and deletions.

**Do not read the container's frame rate to choose.** Test the transforms and let the
measurement decide:

    PAL candidate against 24000/1001 master   slow by   × 0.95904095904
    the inverse                               speed by  × 1.04270833333
    NTSC-to-NTSC and PAL-to-PAL cases likewise, and NTSC against PAL

**Reading the declared rate would have missed a whole release.** Eight episodes of one folder
carry steps that are integer frames at **29.97** while every container on both sides reports
`24000/1001`, and nothing anywhere declares it. **The unit is in the EDIT, not in the
container.** The owner's instinct to test rather than read is measured-correct.

**TAKE THE BEST TRANSFORM ABOVE 90 % AUDIO SIMILARITY — AND REPORT BY HOW MUCH IT WON.** If two
hypotheses sit above the bar within a margin too small to separate them, that is
**INDETERMINATE, not PAL**. Choosing the argmax of a broad peak is the error that produced two
wrong maps in one night: a scan whose top three positions sat within 0.028 was read as a
location. **The winner and the margin are two facts and only reporting one of them is the
count-not-a-membership defect wearing a verdict.**

**When the margin cannot separate them, escalate to a different modality rather than to a
tie-break rule.** The owner's proposal is scene detection, and its value is precisely that it
does not share an input with the audio comparison — *neither could produce the other's answer
from its own inputs.* **A tie-break computed from the same correlations is not a third opinion.**

### The two thresholds are different quantities and must not be merged

    90 %   AUDIO SIMILARITY   -- "is this the same programme"
    ~80 %  FRACTION IN COMMON -- "is there enough continuous material to be worth merging"

The second is a range, deliberately: **the agent holding the corpus finds the value**, and it
is the point at which the pipeline stops trying rather than a property of any file.

**AND THE VALUE MUST BE CALIBRATED AGAINST FILES THAT MUST NOT MERGE, NOT ONLY AGAINST FILES
THAT MUST.** A control population is not validated by existing — *it is validated by containing
the case that breaks you.* A bar set this campaign had an adversarial population and it was the
**wrong adversary**: it tested against cross-language coincidence and never against a probe
landing on a transition, and it is refusing good files in production as a result.

**The adversarial material already exists.** One corpus file carries regions where a wide scan
finds **no correspondence at any offset** — those are what "should not merge" looks like, and
they are measured rather than constructed.

### Small missing pieces must not break the sync

**A candidate missing short pieces is the ordinary case, not a failure.** Where a large
continuous fraction is common, attempt the merge. **But the missing pieces must be LOCATED**,
and that is where the visual methods earn their place — audio correlation says a piece is
absent, frame comparison says where.

## 4g. The head, and what the merge_plan must emit

### At the head of the file

**If there is an offset at the start: take from the master if the master has it, otherwise
insert silence — so that the output begins at 0 of the master.** One rule, no case analysis.

### Every produced file gets a plan report beside it

    THE PRODUCED FILE goes where the merged file will live -- not to a scratch output
                      directory that something later deletes
    THE REPORT        <candidate filename>.merge_plan.log, beside it

**This is not bookkeeping. It is what makes a produced file checkable at all**, and the
campaign has a number for what its absence costs: a manifest records **195 artefacts produced
and since deleted, 45 of them merged and never verified by anything, and one merged while the
pipeline itself recorded it as OUT OF SYNC.** None can now be examined.

**The report carries, per file:**

- **a timeline diagram, generated automatically** — master time along the axis with
  **timestamps**, one bar per segment at its offset, **lost regions as a red line ABOVE the
  offset values**, and **every offset value actually used**, borrowed ones included
- **a human-readable account beside the diagram**, in plain sentences: *took from the candidate
  here, deleted there, took from the master here to fill.* A reader must be able to see what
  was done to the file without reading a table
- **the borrowed placements marked on the track itself**, with a tag **distinct from
  `VMSAM_FABRICATED`** — that key has a consumer that evicts assembled tracks, and *intact but
  derived placement* is not *assembled from pieces*. Keep it simple and light.

**The diagram and the sentences are the same information twice on purpose.** The owner's own
validation is watching files, and *the log is read by whoever opens the log; the tag and the
report are read by whoever opens the file.*

## 5. Commentary tracks are not repaired — owner's ruling

**No chimeric version of an audio track tagged as commentary.** Pass it through
untouched or leave it out; never assemble one, and never mark one.

The repair rests on *what was done to the file was done to all its streams* — true of
a cut, true of a resample. **A commentary track is a separate recording over the same
picture**, not a translation of the programme's own audio, so it inherits neither the
premise nor the plan: the master may carry no commentary to fill from, and there is
no reason its edit points sit where the main track's do.

Three places in the current code walk `commentary` and each must stop:

    iterate_candidate_audios   builds a chimeric track for every commentary stream
    find_master_audio_for_language   would fill a gap from the master's commentary
    mark_audio_dicts   stamps `fabricated` on commentary and audio-description alike

**Audio description is the adjacent question and this ruling does not decide it.**
It is the same shape — a separate recording over the same picture — and it is not
what was ruled on. Raise it rather than extend this by analogy.

## 6. Track marking — `fabricated`

**Defined here, before the repair exists, so the repair implements a contract
rather than inventing one.** Nothing sets it today; every consumer must therefore
tolerate its absence.

### In memory

On the audio metadata dict — the same dicts `keep_best_audio` iterates:

| key | `fabricated` |
| :--- | :--- |
| type | `str` |
| absent or empty | the track is **untouched**: it came from a source file as it was |
| present | the track was **built** by the repair, and the string says how |

Defined values, `method` optionally followed by `:detail`:

| value | meaning |
| :--- | :--- |
| `chimeric` | assembled from two or more sources on the master's timeline |
| `resampled:<factor>` | stretched by that factor, e.g. `resampled:1.042708` |
| `chimeric+resampled:<factor>` | both, in that order |

Consumers test **truthiness**, never equality: a value they do not recognise still
means fabricated. Adding a method is therefore backward-compatible.

### On the produced file

The same string, as a Matroska **track tag** named `VMSAM_FABRICATED`, so the
origin of a track survives the merge and is recoverable afterwards:

```sh
ffprobe -v error -show_entries stream_tags=VMSAM_FABRICATED -of compact <file>
```

Set it on the first ffmpeg pass (`-metadata:s:a:<n> VMSAM_FABRICATED=...`, next to
the existing `-map_metadata 0`); `mkvmerge` preserves track tags — the merge command
passes `--no-global-tags`, not `--no-track-tags`.

An output whose tracks were rebuilt and which carries no such tag is a defect, not
a detail: campaign 1 spent real work dating 67 outputs by mtime because nothing in
them said what produced them.

### The consumer that already exists — and it is not a defect

`keep_best_audio` prefers an untouched track over a fabricated one of the same
language, before any codec rule applies — a FLAC assembled from three sources must
not evict an intact AAC. The branch is already in place and inert: with the key
absent everywhere, `.get()` returns `False` on both sides and the current behaviour
is unchanged. It activates by itself the day the repair sets the key.

**This was reported as the campaign's largest finding and the owner has ruled it a
design intention. It is closed.** Measured first: `keep_best_audio` ranks on codec,
channels, sample rate and bitrate, and alignment is not in the comparison — a master
DTS 6ch at 1509 kbps beats a repaired E-AC-3 2ch at 224 kbps, so of 42 correctly
repaired tracks only **12 could reach a file**, those whose language the master lacks.

**The owner's answer: that is the point.**

> **The repair exists to add languages in the cases where they are absent.**

**Concretely, in the owner's own example: it repairs a VFF where there would otherwise
only be a VFQ.** A dialect absent from the artefact is the case; a dialect already
present is not. **For now it is only that**, and the boundary is deliberate. The owner
names one thing it might become and explicitly defers it: software that, given two
tracks from different encodes — an AAC and an E-AC-3 — could produce a good-quality
Opus from both. **That is not this campaign and must not be built toward.**

So twelve of forty-two is not a loss rate, it is the specification being met. A track
whose language the master already carries has nothing to add, and the master's own
recording is the better artefact — it is also the timeline everything else is measured
against. **Do not "fix" this, and do not put alignment into the ranking.** The owner
adds two facts that follow from it: **a chimeric VFQ is kept against a VFF** — a
different dialect is not the same language for this purpose — and a chimeric audio is
generally integrated **under the comparison language**.

**What this changes for whoever reports next:** an observation of a repaired track
losing to the master's own track is *the system working*, not evidence of a defect.
That distinction was already made once here — observations consistent with a defect
are not evidence of it — and this is the case it was made for.
