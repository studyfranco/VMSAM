# VMSAM — campaign 2: chimeric repair and resampling

> ***THE GOAL IS FRAME-ACCURATE. Not "close enough", not "the right stretch" — the exact zone, to
> the frame.*** *The owner's requirement, 2026-09-07.*

**This file carries what BINDS. What was MEASURED to establish it is in
`CAMPAIGNS/02-dev-chimeric-resample/FINDINGS.MD`, and the zone contract is in
`CAMPAIGNS/02-dev-chimeric-resample/SPEC_ZONE_A.MD`. Task reports are in that directory's `TASKS/`.**

## THE NEED — the decision chain, in the owner's own order

    FIDELITY OK, DELAY DIFFERENT ...... edit the candidate to match the master
    FIDELITY NOT OK ................... test resampling
    STILL CUT AFTER THAT .............. hand it to the chimeric path

### 1. Fidelity holds, the delay does not — edit to match the master

*The pair agrees on content; only the timing is wrong.* **This is the constant-offset case and it
is what the locator is for.** The measurement returns a plan whose `kind` is `constant`, and the
repair applies the offset. Nothing is resampled and nothing is spliced.

    the floor .............. `MIN_MEDIAN_FIDELITY = 0.70` in `change_point_locator.py`
    the plan ............... `{"kind": "constant", ...}` — one segment, one offset
    where the repair runs ... at the moment of refusal, inside `remove_not_compatible_video`,
                             against `best_video`. Not before the merge, not during it.

***AND THE INDEX IS IN FRAMES, NEVER IN MILLISECONDS.*** *A speed relation changes the duration and
not the frame count, so the frame index is invariant under a resample while a millisecond boundary
is silently wrong by an amount that grows with position in the file.* **Rules and derivation:
`FINDINGS.MD`.**

### 2. Fidelity does not hold — test resampling

*A rate relation degrades the correlation, which is why fidelity fails.* **The two constants the
engine knows:** `1.042709` for PAL against film (25 / 23.976), `1.001` for NTSC.

> ***AND THE VERDICT DECIDES, NEVER THE NUMBER.*** *`merge_video_repair.py`'s own docstring records
> why: resampling a file that needed nothing was* **TWICE AS FREQUENT** *as the inverting case the
> detector exists to find. Nothing is applied without an explicit verdict, and a ratio derived from
> a duration match is forbidden — a file whose only defect is carrying more content would be
> resampled by one.*

***AND THE RATE MUST BE MEASURED BY SOMETHING THAT IS NOT THE CORRELATION, BECAUSE THE CORRELATION
IS WHAT THE RATE RELATION BREAKS.*** *See the ruling below. A pitch-based measurement covers the
RESAMPLE family; the STRETCH family preserves pitch and needs a content-anchored timing measurement
that does not exist yet.* **Verdict vocabulary and its exclusions: `FINDINGS.MD`.**

### 3. Still cut — the chimeric path

*The candidate has material the master lacks, or the reverse, and no single offset or ratio
reconciles them.* **The chimeric path assembles on the master's timeline and has already taken the
delay problem into account** — it is reached from the repair, and it is wired into the guard's own
escape clause rather than around it:

    `mergeVideo.py`  `if len(dict_file_path_obj) < 2 and not len(repaired_videos):`
                     raise "only one file left, useless to merge"

*That `and not len(repaired_videos)` is deliberate and recent: without it the repair path was
UNREACHABLE — two files in, the refusal removes one, one remains, and the guard raised just after
the repair had attached its object.* **Detail: `FINDINGS.MD`; contract: `SPEC_ZONE_A.MD`.**

## WHERE YOU MAY WRITE — the tagged zones, and what is in them

`src/mergeVideo.py`, `src/video.py` and `src/tools.py` are frozen **except** between
`BEGIN: AGENT modification ok` and `END: AGENT modification`. Everything else in them is closed.
The full freeze table is in `WRITE_ZONES.MD` between `<!-- FROZEN-TABLE:BEGIN -->` and `:END`.

### `src/mergeVideo.py` — four zones

    ~681  and  ~692  and  ~700   THE FRAME ARITHMETIC. `Decimal(number_frame * framerate_duration_ms)`
                                 and a standing TODO for variable-frame-rate. *** THIS IS WHERE
                                 FRAME-ACCURACY IS COMPUTED, SO IT IS THE ZONE THE GOAL DEPENDS ON. ***
    ~807 to ~896                 THE REPAIR CALL SITE. The late import of `merge_video_repair`, the
                                 narrow-by-scope except, the five-outcome log, and the guard above.
                                 *** THE REPAIR IS A CAPABILITY CLOSED BY DEFAULT, GRAFTED ONTO A
                                 PATH THAT WAS GOING TO REJECT THE FILE ANYWAY. IT MAY NOT BRING
                                 DOWN A MERGE THAT WAS WORKING. ***

### `src/video.py` — two zones

    ~616 to ~685    `get_best_quality_video` — the VMAF comparison that picks the better video.
    ~690 to ~786    `get_good_frame` — *** THE FRAME SNAP. This is where a millisecond delay becomes
                    a frame index, and it is the other half of the frame-accuracy goal. ***

**Everything outside those markers in those two files is frozen, and `src/audioCorrelation.py`,
`src/gestionar_show/` and `Dockerfile` are frozen entirely — the freeze forbids MODIFYING, never
CALLING.**


`AGENT.MD` says how to work here and never changes. This file says what to build,
and is replaced at the next campaign.

---

## The objective

Files the pipeline **refuses** are repaired instead of parked. A refusal means
stages 1 and 2 found no constant offset — which is correct, and which is precisely
the case the repair takes over.

The repair happens **at the moment of refusal**, inside
`remove_not_compatible_video`, against `best_video`. Not before the merge, not
during it. Contract: `SPEC_ZONE_A.MD`.

### And the repair targets the defective zone, to the frame

The owner's requirement, 2026-09-07: *do not replace the whole problematic stretch —
repair the exact zone, to the frame.* Two instruments, because neither reaches the
other's resolution:

    fpcalc   locates the ZONE      1 fingerprint point = 126.6 ms = 3 to 4 frames
    pHash    decides the FRAME     `frame_compare.py`, DCT 8x8, 64 bits

Three constraints, all measured, none of them optional:

- **Work from the raw fingerprint list, not from `correlation()`.** That function
  averages the per-point Hamming similarity and returns one number, and the position is
  gone with it. **The per-point vector locates the zone in a single pass**, which is why
  the iterative descent looked necessary and is not. *Measured in `docs/MEASURING_NOTES.MD`.*
- **Index in frame numbers, not milliseconds.** A speed relation changes the duration and
  not the frame count, so **the frame index is invariant under the resample** while a
  millisecond boundary needs a conversion factor and is silently wrong without it, by an
  amount that grows with position in the file. *Measured in `docs/MEASURING_NOTES.MD`.*
- **Do not route through `cross_correlation`.** Below `min_overlap` it returns without a
  value and without a message, and that `None` has the shape of "nothing found". Both the
  32-point floor and the silent failure belong to the layer the new method stops using.

Window floor: **30 s**, as the main software bounds its cut-time search. The hard wall is
4.05 s, so the margin is sevenfold.

## What is refused today, and why

> **ONE ROW IS NO LONGER CAMPAIGN 1's. `vmsam-forensic` RE-ESTABLISHED THE SPEED FAMILY ON
> 2026-09-03 FROM THE PIPELINE'S OWN REFUSAL RECORDS — 303 files with a computable duration ratio,
> PAL 6.3 % + NTSC 2.5 % = **8.9 %**, with campaign 1's classification columns DELIBERATELY NEVER
> READ.** *The number moves DOWN, not away, and the table's warning is retired for that row alone.*

**The remaining proportions are campaign 1's and have not been re-measured.** They are read
from that campaign's documents, over a corpus of 315 files that has since grown —
new errors keep arriving. Treat the table as the **shape** of the problem, never as
a count: it says which failures exist and what they look like, not how many there
are today.

Re-establishing them from zero is the forensic agent's first job, and its report
lands in `VMSAM_HELP_AI/`. Until it does, a figure taken from this table into any
other document is a figure with no measurement behind it.

| family | share (campaign 1) | shape of the failure |
| :--- | ---: | :--- |
| different cuts | **57 %** | the offset is piecewise-constant — a staircase. Content one side has and the other does not |
| trimmed edges | 18 % | the same, with a single step at one end |
| speed mismatch | **8.9 %** — *re-measured* · PAL 6.3 · **NTSC 2.5** *(parts rounded; they do not sum to the total as printed)* | the offset drifts linearly. **The two halves behave oppositely and must not be treated as one family** — see below |
| weak correlation | 10 % | heavy re-encodes, differently mixed dubs, music-only stretches: no confident peak |
| no shared language | 2 % | no audio language in common, or the tags lie |

**The largest family is not speed.** Over 79 measured files, **zero** showed a clean
slope near a constant (`CAMPAIGNS/01-dev-AI/FINDINGS.MD` §15). A repair that only handles speed
addresses a tenth of the problem.

### PAL and NTSC are one number and two problems

The speed row is 6.3 % PAL and 2.5 % NTSC — **rounded, and they do not sum to the 8.9 %
printed beside them.** The unrounded values are in the forensic agent's report; a reader
checking the arithmetic here gets a discrepancy that this document cannot resolve.
The ratios differ by forty-two times.
Measured drift, same relation applied to a 30 s window and to a 24-minute episode:

    PAL   25 / 23.976  = 1.042709    1281 ms per window   30.7 frames   ->  61.5 s per episode
    NTSC  1000 / 999   = 1.001001      30 ms per window    0.72 frames  ->   1.44 s per episode

**PAL breaks loudly.** Thirty frames of drift inside one window smears the correlation
peak, the offsets scatter, and the file is refused. A refused file is parked, not
delivered wrong.

**NTSC passes silently.** Under one frame of drift per window, so every window-scoped
check agrees and the peak stays sharp — and the episode still ends 34 frames out of
sync. *It is the case that satisfies every control we have and produces a bad file.*

    A per-window check cannot see a drift smaller than its own window.

So the two halves need opposite work. PAL needs a branch that is currently unreachable.
**NTSC needs a check that spans the whole file, because no windowed instrument will ever
report it.** The owner has NTSC material in the error tree; it is the half that can be
worked on today, and the half a delivered file can carry.

Two files were proven genuinely unrepairable: one truncated at source, one
encrypted. Whatever the corpus size turns out to be, the ceiling is far away.

## What campaign 1 established, and what it did not

**Established, and worth reusing.** Cut repair by plateau reassembly works: the
candidate's audio is cut at the jumps, placed at its master-timeline positions, and
the gaps filled from the master itself. Speed measurement can reach 0.3 ms of drift
error over a whole episode when refined iteratively — single-pass carries ~0.1 %,
which is the instrument, not the media.

**Not established.** That any of it improves the library — and the campaign could
not have found out, because **its own success metric never ran**. `final_status`
reaches the 15 ms comparison only in its third branch, which needs a Step 1 verdict
of `in_sync` or `out_of_sync`; every Step 1 verdict in the corpus said
`unverifiable`. All 315 rows exited at branch one or two, and `Resolved: 0` was read
as *nothing qualifies* when it meant *nothing reached the check*
(`CAMPAIGNS/01-dev-AI/RESTART.MD` §3.1).

**The general form is the lesson: an armed guard that has never fired is
indistinguishable from a working one.**

**The defect that cost the most.** A guard, a refined measurement, a policy branch,
an order check and a subtitle repair were each written, reviewed, unit-tested green
— and never called by anything. Five times. Every suite passed each time.

## What closes a piece of work here

`AGENT.MD` § Target-first, applied to this campaign: on a real error file that used
to fail, an output is produced and the repaired track is **present in the final
mkv**, identical to the target you built by hand.

Not "the code is in the zone and the tests are green".

## The four stages, as the owner defined them on 2026-09-04

**The objective changed shape and the sentence that says how is his:**

> ***"VMSAM regardait ce qui était pleinement commun. Maintenant il règle les divergences."***

**The old target was the INTERSECTION. The new one is the DIFFERENCE, characterised to the
frame.** *Every tool in the repository was built to answer "where does it match"; it is now
being asked "where exactly does it STOP matching", and that question needs three orders of
magnitude more resolution than the first.*

### The chain

    1  COMPATIBILITY   are the files compatible as a whole
    2  IDENTITY        are these the same episode -- audio similarity, and a
                       resample exists to make that comparison possible
    3  CHIMERA         adapt the candidate TO THE MASTER: cut at the exact place,
                       add at the exact place
    4  PRODUCE         the best master

### Stage 2 — the identity gate, specified

> ***"Si ≥ 50 % du candidat ne correspond pas au master, on ne traite pas."***
>
> ***"On relance la première audio correlation mais avec MAX 50 FICHIERS qui font MINIMUM
> 1 MIN 30 et CHEVAUCHANT LE SUIVANT DE 50 %. Ça permet une première cartographie du
> fichier. Et il faut que 50 % DE CES FICHIERS MATCH avec une similarité ≥ 90 %."***

    at most 50 probes | each at least 90 s | each overlapping the next by 50 %
    PASS if at least 50 % of the probes reach similarity >= 0.90

**This is a CARTOGRAPHY, not a delay measurement.** *Its output is a map of where the two
files agree, and the gate is a count over that map rather than a mean over a correlation.*
**A mean fidelity hides a bimodal file; a count of probes above a bar does not.**

*Note on the denominator, stated because the overlap makes it ambiguous: with 50 % overlap
every second is covered by two probes, so "50 % of the probes" is not "50 % of the
duration". The rule is written on PROBES.*

### Stage 3 — repair is for everyone, and the FFT is not the limitation

> ***"Détrompe-toi. Les outils en place détectent presque à la frame de divergence. La FFT
> est tellement puissante pour ça. Donc à l'étape 3 ON RÉPARE TOUT LE MONDE."***

**The owner is right, and tonight's measurements support him rather than contradict him.
The distinction the fleet had collapsed:**

    the CORRELATION resolves    0.0-0.4 ms      flat-region spread, measured by dev-1
    the PROBE GRID resolves     123.840 ms      the chromaprint quantum
    the SEARCH PROCEDURE        12 000 ms       the refined-bracket floor
                                100 000 ms      when the refinement never narrows

> **THE MEASUREMENT IS PRECISE. THE SEARCH THAT PLACES THE MEASUREMENT IS COARSE.** *A
> sub-millisecond spread cannot come out of a 124 ms-quantised series, so the instrument is
> not the ceiling — the procedure that decides WHERE to point it is.*

**And repair must therefore be reachable for EVERY file that passes stage 2, not only for
files stage 1 rejected.** *Today the only entry point is `remove_not_compatible_video`, so a
file whose divergence stage 1 TOLERATES is merged with its divergence and never repaired.
That is a defect of reachability, not of policy.*

### Stage 3 — the chimera is alignment, and only alignment

> ***"La chimère est un alignement. Un ajustement pour rentrer dans le master."***

**No quality selection. Which stream is better belongs upstream, to `best_video`.** *The
chimera makes the candidate fit the master's timeline at zero offset and does nothing else,
so a defect in output quality is never the chimera's and a defect in placement always is.*

### THE END CONDITION, SETTLED BY THE OWNER, 2026-09-05 — AND IT IS SIMPLER THAN WHAT WE HAD

> ***"la condition de fin c'est `ci` qui l'a donnée, lorsqu'il a réussi à fusionner tous les
> fichiers. Sauf ceux complètement différents du master."***

    THE CAMPAIGN ENDS WHEN `ci` HAS MERGED EVERY FILE,
    EXCEPT THOSE COMPLETELY DIFFERENT FROM THE MASTER,
    *** WITH THE LATEST VERSION OF THE CODE ***

**OWNER, SHARPENING IT:** *"quand je dis tous, c'est avec la dernière version du code. Quitte à
refaire 1000 fois le même fichier."*

> **A MERGE PRODUCED BY AN OLDER REVISION DOES NOT COUNT. RE-RUNNING THE SAME FILE A THOUSAND
> TIMES IS ACCEPTABLE AND EXPECTED.**

    THE END CONDITION IS A STATE OF THE CURRENT CODE, NOT AN ACCUMULATION OF
    HISTORY. EVERY CODE CHANGE INVALIDATES EVERY EARLIER MERGE.
    THE LEDGER IS NOT A TALLY THAT GROWS. IT IS A SNAPSHOT THAT RESETS.

**AND THAT MAKES ARTEFACT PROVENANCE LOAD-BEARING, WHICH IT WAS NOT AN HOUR AGO.** *To evaluate
the condition you must answer, per artefact:* **which revision produced this?** *And it is
measured that you cannot:*

    the engine emits digests of its OWN manifest, in its OWN hashing scheme.
    TWELVE LOWERCASE HEX CHARACTERS BESIDE A FILENAME, AND NOT ONE IS A GIT OBJECT.

**[CORRECTED BY `vmsam-ci`, VERIFIED HERE] THE ARCHITECT WROTE "NOTHING THE CONTAINER EMITS CAN
BE JOINED TO THE PUBLISHED BRANCH AT ALL". THAT IS FALSE AND IT IS A §97 INSTANCE OF ITS OWN —
a true statement about the ENGINE'S MANIFEST DIGESTS, published as a claim about EVERY
EMISSION.**

    all 40 client rows carry `image_git_commit`, 3 distinct values,
    AND ALL THREE RESOLVE — `git cat-file -t` answers `commit` for each.
    It comes from `/health`, injected at build. A DIFFERENT FIELD.
    BOTH THINGS ARE TRUE AT ONCE: the join EXISTS on the client's path
    and DOES NOT EXIST on the artefact's own.

**AND THE REAL LIMIT IS SHARPER AND BELONGS TO THE CLIENT, NOT THE ENGINE:** *only 14 of 40 rows
read the revision AT THE ROW; the other 26 carry a value stamped once at startup, and the
client's own correction record says it is wrong for every row after a known id.* **THOSE 26
CANNOT BE ATTRIBUTED TO ANY REVISION — and that is one seat's own defect to fix, not an emission
anyone owes it.**

**So the end condition currently cannot be evaluated for a second, independent reason** — *not
only because the exclusion class is silent, but because* **an artefact cannot be dated against
the code that made it.** *Two emissions close both: one line naming the exclusion cause, one line
carrying a resolvable revision.*

## OBJECTIVE 3 IS NOT DOWNSTREAM OF OBJECTIVE 2 — ARCHITECT'S RULING, 2026-09-08

***A RATE RELATION CANNOT BE DETECTED BY CORRELATION, BECAUSE CORRELATION IS WHAT IT BREAKS.***
*`dev-2`'s sentence, and it is the whole argument:* **"IF A PAIR IS RATE-RELATED, THE MEASUREMENT
THAT WOULD PROVE IT IS THE ONE THE RATE RELATION BREAKS."**

    VERIFIED AT `change_point_locator.py`, blob `88b9f7cadf89be3b`, 1992 lines:
      `speed_relation_suspected` ..... 3 occurrences, ALL in the locator. *** ZERO ELSEWHERE ***
      `median_fidelity_below_floor` .. 2, both in the locator. *** ZERO ELSEWHERE ***
      control `quantum_ms` ........... 13 + two other modules, so the search discriminates
      the monotone discriminator ..... 3 kept probes: *** 33.3% OF RANDOM ORDERINGS PASS ***
                                       4: 8.3% · 5: 1.7%

**THREE FACTS DECIDE IT, AND THE THIRD ALONE WOULD:**

*1. THE HANDOFF CHANNEL DOES NOT EXIST.* The docstring promises to refuse on monotone drift and
hand the pair to objective 3; the refusal is a token with no reader. ***A DOCUMENTED CONTRACT WITH
NO CHANNEL IS NOT A HANDOFF, IT IS A NOTE.***

*2. WHERE IT CAN FIRE, THE EVIDENCE IS AT ITS WORST.* PAL at 4.27% reads fidelity 0.602 and 0.669
on two independently constructed pairs — sub-floor, declined, and `merge_video_repair` has no
fallback. Fewest probes means the monotone test sits at 33.3% false-pass.

> ***3. AND EVEN WITH A CHANNEL IT WOULD NEVER FIRE FOR THE POPULATION THE OWNER NAMED. NTSC AT
> 0.1001% READS FIDELITY 0.9614 — THE 0.70 GATE IS NEVER ENTERED AND NOTHING DECLINES THE PAIR.***
> **THE CAPABILITY COVERS THE RELATION THAT BARELY MATTERS AND MISSES THE ONE IT WAS BUILT FOR, AND
> NOBODY CHOSE THAT.**

***THE RULING: MEASURE THE RATE BY A METHOD THAT DOES NOT DEPEND ON CORRELATION FIDELITY. OBJECTIVE
3 RUNS BEFORE OR BESIDE OBJECTIVE 2, NEVER AFTER IT.*** *A pitch-based arm separating resample from
stretch is a DIRECT measurement rather than an inference from a degraded one.*

**AND WHAT THE RULING DOES NOT DO, stated because a design ruling is read as permission otherwise:**
***IT LIFTS NO `no-go`. IT AUTHORISES NO RUN AGAINST THE ERROR CORPUS OR AGAINST REAL MEDIA.*** *The
owner's gate stands and `dev-2`'s own result is why —* **RESAMPLING A FILE THAT NEEDED NOTHING WAS
TWICE AS FREQUENT AS THE INVERTING CASE, SO NOTHING APPLIES WITHOUT AN EXPLICIT VERDICT, AND NO
PRODUCER OF THAT VERDICT EXISTS EITHER.** *Building the measurement does not build the permission.*

***AMENDED THE SAME DAY: THE RULING COVERS THE RESAMPLE FAMILY AND NOT THE STRETCH FAMILY, AND THAT
IS A SCOPE AND NOT A GAP.*** *`dev-2` wrote the verdict contract BEFORE building and the writing
surfaced the conflict:*

    a RESAMPLE moves pitch WITH speed .... `k` is readable from the audio alone. Clean.
    a STRETCH preserves pitch ............ *** k_pitch = 1.000, AND THE ONLY REMAINING
                                          SIGNAL IS HOW LONG THE FILE IS. ***

> ***SO FOR THE STRETCH FAMILY THE ONLY AVAILABLE RATIO IS THE DURATION-DERIVED ONE — WHICH IS
> EXACTLY THE RATIO THE BOUNDARY FORBIDS.*** *`merge_video_repair.py:93-100` states it at the site:
> the verdict decides and not the number, because* **RESAMPLING A FILE THAT NEEDED NOTHING WAS TWICE
> AS FREQUENT AS THE INVERTING CASE THE DETECTOR EXISTS TO FIND.**

**AND ONE MEASURED PAIR SHOWS THE BOUNDARY IS RIGHT RATHER THAN MERELY CAUTIOUS:** *pitch preserved
at k = 1.000131, duration off by +0.4649%,* ***AND THE EXCESS WAS CONTENT.*** *A duration-derived
ratio would have resampled a file whose only defect was carrying more material — the exact
fabrication this campaign is convened against.*

    SCOPE, STATED RATHER THAN PATCHED AROUND: 12 RESAMPLE COVERED · 8 STRETCH NOT.
    The stretch family needs a CONTENT-ANCHORED timing measurement -- *** WHERE EVENTS
    FALL, NOT HOW LONG THE FILE IS *** -- and that instrument does not exist.

> ***A NAMED EXCLUSION IS A RESULT. A LADDER THAT LOOKS COMPLETE BECAUSE A DURATION RATIO WAS LET IN
> THROUGH THE BACK IS THE FABRICATION DEFECT WEARING A COMPLETENESS COSTUME.*** *`dev-2` refused
> that before anyone offered it.*

***AMENDED AGAIN, BY THE RUN. "12 COVERED, 8 NOT" IS NOT SUFFICIENT AS A SCOPE — AND THE REASON IS
THE THIRD APPEARANCE OF THIS CAMPAIGN'S OLDEST DEFECT.*** *`dev-2` sealed its verdict contract and
its gate before the runner existed, then ran six rungs on constructed material:*

    1 identity ......... no_rate_relation        k=1.000000  spread 0.000000
    2 pure delay ....... no_rate_relation        k=0.999996  spread 0.000005
    3 PAL resample ..... rate_relation_measured  k=1.042482  spread 0.000389
    5 NTSC resample .... rate_relation_measured  k=1.001024  spread 0.000234
    6 no shared content  *** ambiguous, NO k, spread 0.005288 -- THE MUST-CATCH HOLDS ***
    *** 4 PAL STRETCH, A 4.27% RELATION, PITCH PRESERVED ->
        `no_rate_relation`, k=0.999986, spread 0.000072 ***

> ***IT IS NOT BLIND. IT EMITS AN AFFIRMATIVE VERDICT THAT A PAIR WITH A 4.27% RATE RELATION HAS
> NONE — AT THE TIGHTEST AGREEMENT OF ANY RUNG EXCEPT IDENTITY. A CONFIDENT, WELL-MEASURED, WRONG
> ANSWER.*** *An instrument that can only speak about PITCH, making a claim about RATE. This is
> `no_shared_content` a third time, and `dev-2` named it as its own.*

***THE FIX IS IN THE VOCABULARY, NOT THE THRESHOLD: A PITCH-BASED PRODUCER MAY NEVER EMIT
`no_rate_relation`. ITS CLEAN ANSWER IS `no_pitch_relation` — A STATEMENT ABOUT WHAT IT MEASURED,
NEVER ABOUT THE WORLD.*** **AN UNCOVERED FAMILY THAT RECEIVES A CLEAN VERDICT IS WORSE THAN ONE
THAT RECEIVES NOTHING, AND A SCOPE NOTE IS A RULE SOMEONE HAS TO REMEMBER WHILE THE VOCABULARY IS A
RULE NOBODY CAN FORGET.**

**AND THE RULE `dev-2` ASKED BE CARRIED INSTEAD OF ITS VERDICT TABLE, BECAUSE IT NEARLY COST THE
FINDING:** ***EXCLUDING A FAMILY FROM COVERAGE IS NOT EXCLUDING IT FROM THE TEST.*** *Its own
amended scope put stretch out, so an honest application of that amendment would have DELETED THE
RUNG THAT FOUND THIS. The runner still required rung 4 and the run followed the runner.* **IT
RECORDS THAT AS LUCK RATHER THAN METHOD, AND IT IS RIGHT TO.**

*And on rung 6, which decides whether the other five are believable: prominence ran 1.10–1.14
against 1.10–1.45 on the passing rungs —* ***IT DOES NOT SEPARATE. A PROMINENCE THRESHOLD WOULD
HAVE BEEN THE WRONG GATE, AND THAT IS KNOWN ONLY BECAUSE THE GATE WAS SEALED BEFORE THE ROW WAS
SEEN.***

*`dev-2`'s limit, recorded rather than softened, and stated by it before anyone could state it:*
***TWO CONSTRUCTED PAIRS CONDEMN THE GATE AND DO NOT CERTIFY THAT OPENING IT PRODUCES A CORRECT
REPAIR.*** *Monotone arithmetic derived independently by `dev-2` — dated in a sealed header written
before its run — and by `dev-1`; the record first showed only `dev-1` and that was my error.
Boundary raised by the Lead.*

## Open questions the owner has not decided

### On the pipeline itself

- **Stage 3 has never been validated.** Nothing shows the PSNR scan locates the
  right frame, nor how it behaves between sources of different resolution. Worth
  validating, and worth generalising to the forced-reference path if it holds. A
  scene-detection tool is available and may be faster or better; the comparison has
  not been made.
- **Choosing the best video is unsettled.** VMAF is what the code uses. No method
  has been shown to be the right one.
- **Per-language delay.** The merge measures a divergence across languages, records
  it, and applies one delay to every track. Measured: 48 of 81 judgeable pairs
  diverge by more than half a frame, median 1.06 frames, worst 8.70 — and 22 such
  files already shipped. The emission site is frozen; this is an owner decision.
- **Whether a sharper offset instrument is worth it.** GCC-PHAT was built in
  campaign 1 and never shown to beat the existing pair on real error files. Bring
  examples before proposing it again.


### THE OWNER'S RULINGS OF 2026-09-05, RECORDED WHERE THEY BIND

    R20   **DERIVE `-ar` FROM THE SOURCE PAIR. DO NOT PIN 44100.**
          *"avoid unnecessary audio resampling and preserve the original sample
          rate whenever possible."*  -> This SUPERSEDES the convergence argument
          the R21 comment rests on, and it is the reason the one-sided clamp must
          not be "repaired" into a two-sided one.

    R-alias  **THE `offset_ms=` ALIAS REMOVAL IS AUTHORISED**, to be validated
             against a REAL RECORD rather than against source.

    `record()` OUTPUT  **MUST NEVER REACH A REPOSITORY FILE.** It stays beside the
             media or in a local ignored directory, **AND MUST BE COVERED BY
             `.gitignore`** -- the owner's words. A ruling inherited from another
             seat is now the owner's own, and the seat that declined to inherit it
             was right to ask.

    THE HOLD ON THE LOCATOR IS **LIFTED**. Two prepared diffs land in one batch,
    **HEADER FIRST** -- the seat's own ordering, which ranks the successor's
    understanding above its own blocking change.

    AND AFTER THAT, IN ORDER: the ~20-file sizing run that collapses a 12x
    uncertainty (22-268 files, 4-44 h), then CONSTRUCTED MATERIAL FOR THE SIX
    SILENT REFUSAL PATHS THAT HAVE NEVER BEEN RUN -- both authorised, both
    buildable under report-only.
