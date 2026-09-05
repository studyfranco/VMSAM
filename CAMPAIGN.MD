# Campaign — what this round is for

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

## What is refused today, and why

**These proportions are campaign 1's and have not been re-measured.** They are read
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
| speed mismatch | 11 % | PAL 25 fps against 23.976: the offset drifts linearly, ~1 s per half-hour. **The method is already ruled on** — `docs/AUDIO_SPEED_POLICY.MD` |
| weak correlation | 10 % | heavy re-encodes, differently mixed dubs, music-only stretches: no confident peak |
| no shared language | 2 % | no audio language in common, or the tags lie |

**The largest family is not speed.** Over 79 measured files, **zero** showed a clean
slope near a constant (`CAMPAIGNS/01-dev-AI/FINDINGS.MD` §15). A repair that only handles speed
addresses a tenth of the problem.

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

### What a log must be

> ***"Nous devons pouvoir comprendre. Qu'un agent puisse naturellement le refaire en lui
> donnant la recette."***

    A LOG IS A RECIPE. The line that records a decision must carry enough to
    RECOMPUTE that decision -- and the run's lines together must carry enough
    for an agent to REDO THE WORK.

**This is mechanically testable and that is the point of writing it this way:** *the fields
read by a decision function, against the fields emitted beside its verdict.* **A verdict that
cannot be re-derived from its own line is an incomplete line, and a test can fail on it.**
*Six instances were found in one evening by hand; the criterion turns that class into
something a harness finds by construction.*

### What a corpus CASE contains — three artefacts, not one

> ***"Le corpus doit AUSSI avoir les audios de comparaison COMPLETS et CONVERTIS de la façon
> dont la fonction de réparation les traitera."***

    1  THE COMPARISON AUDIOS, COMPLETE, CONVERTED EXACTLY AS THE REPAIR PATH
       WILL PROCESS THEM
       -- this is what the repair CONSUMES, so it is what an agent improving
          the repair needs. It is also the expensive artefact: producing it is
          the extraction the corpus exists to skip.

    2  THE IN-MEMORY STATE captured just before
       `remove_not_compatible_video(list_not_compatible_video, dict_file_path_obj,
       best_video)`, the produced files, and a written record of what exists at
       that instant.

**THE UNCONVERTED SOURCE IS NOT STORED. The architect specified it and the owner removed it,
and the owner is right for a reason that is this document's own §9:**

> ***"Je pense que ce n'est pas utile vu qu'on aura les audios de comparaison convertis comme
> la réparation les traitera. L'idée est que les agents, quand ils testent les améliorations
> de la réparation, puissent utiliser les fichiers qu'il leur faut."***

**THE LIBRARY IS ALREADY THE SOURCE OF TRUTH FOR SOURCES. A copy in the corpus is a SECOND
COPY THAT DRIFTS** — *`WRITE_ZONES.MD` §9, argued by the architect all evening and not
applied to its own design.* **And the corpus's purpose is to let an agent skip the pipeline,
not to re-host the media: what it must hold is what is EXPENSIVE TO RECOMPUTE and what the
code under test CONSUMES. The raw source is neither.**

### So the converter identity is no longer a nicety — it is the only staleness detector left

**The architect justified storing the source by a self-check: re-run the conversion on it,
compare, a difference is drift. With the source gone that check goes with it** *(and
`vmsam-forensic` had already shown it could not be trusted until a null established that the
same converter twice gives the same output — a null nobody has run).*

    WHAT REPLACES IT, AND IT IS CHEAPER AND MORE HONEST:
      every converted artefact records ITS CONVERTER -- parameters and code
      revision. When the converter moves, the stored conversion is KNOWN stale
      and is RE-DERIVED FROM THE LIBRARY, which still holds the original.

**Recovery does not need a duplicate in the corpus; it needs to know WHEN to recover.**

### "CONVERTER" DOES NOT MEAN OUR REVISION — measured on real artefacts

**The architect wrote "parameters and the revision of the code that applied them". That
detector would report FRESH while the conversion had drifted, and `vmsam-forensic` measured
why on artefacts already on disk:**

    the rebuilt track      ENCODER = Lavc62.11.100 aac
    the container          Lavf63.1.101
    the subtitle streams   Lavc63.1.101

    TWO LIBAVCODEC MAJOR VERSIONS INSIDE ONE ARTEFACT, ONE PIPELINE RUN.

    repair: build     our module digests    NAMES NO ENCODER
    repair: sources   our tree digest       NAMES NO ENCODER
    grep ffmpeg|Lavc|Lavf|ENCODER over every job log:   ZERO HITS

> **A CONTAINER REBUILD MOVES libavcodec WITH OUR REVISION UNCHANGED. A staleness detector
> keyed on OUR revision is keyed on the one thing that did not change.**

    THE RECORDED IDENTITY IS THE ENCODER'S OWN: libavcodec and libavformat
    versions at minimum, PLUS THE FULL FILTER CHAIN AND ITS PARAMETERS.
    Our revision is a field beside it, never the field.

**AND THE ONLY EXISTING TRACE IS A TAG INSIDE THE ARTEFACT, present only where a track was
RE-ENCODED and absent precisely on tracks that were COPIED.** *So the record is missing
exactly where nothing happened — which is the case you most need to tell apart from "nothing
has happened YET".* **That is `absent never zero` arriving inside the staleness detector
itself, and it means the identity must be EMITTED by the producer rather than recovered from
the artefact.**

### The cost of not storing the source, written down rather than left to inference

**With the self-check, a curated case carried a real claim: stored source in, stored
conversion out, converter under test. Without it the curated tier has no ground truth about
anything, and every verdict over it is "consistent with its source" — agreement, never
correctness.** *That is the ceiling on all twenty of `vmsam-forensic`'s verdicts and it is
now the ceiling on the curated tier.*

**The trade is still right — a duplicate that drifts silently is worse than an absent control
known to be absent — but it is a COST, and a later reader should find it stated rather than
reconstruct it** (`vmsam-forensic`, who wrote it into `SELECTION.md` §4 rather than accept
the architect's "removes a column and removes no case").

### "THE comparison audio" NAMES A THING THAT DOES NOT EXIST — THERE ARE THREE

**This document and `SELECTION.md` §3 both say THE comparison audio, singular. Read out of the
code by `vmsam-ci-build` and verified here, there are THREE conversions and THEY DISAGREE ON
THE RATE:**

    STAGE 1  prepare_get_delay_sub -> extract_audio_in_part
             pcm_s16le · pair-minimum channels · NO `-ar` AT ALL, each side
             keeps ITS OWN rate · loudnorm
    STAGE 2  second_delay_test
             mono · `-ar` = min(the pair's lowest source rate, 44100) · loudnorm
             `mergeVideo.py:584-585`
    LOCATOR  change_point_locator._extract -- AND THIS IS WHAT THE REPAIR CONSUMES
             mono · `-ar 44100` HARDCODED at :313 · NO loudnorm at all

**The rule above — "stored at the rate the software defined at the first calculation, NOT a
fixed rate" — describes STAGE 2. The repair consumes THE LOCATOR'S. And stage 1 defines no
rate at all.**

> **A SPEC THAT SAYS "THE" CONVERSION WHERE THERE ARE THREE DOES NOT UNDER-SPECIFY. IT FORCES
> THE BUILDER TO PICK ONE SILENTLY, AND WHICHEVER IT PICKS DECIDES WHICH CONSUMER THE CORPUS
> IS ABOUT — invisibly, in a field nobody thought to add** (`vmsam-ci-build`).

**And the disagreement is not academic. VERIFIED:**

    source 48000 Hz  ->  stage 2 = 44100 · locator = 44100    agree BY ACCIDENT
    source 44100 Hz  ->  44100 · 44100                        agree
    source 32000 Hz  ->  stage 2 = 32000 · locator = 44100    THE LOCATOR UPSAMPLES

**Above 44100 they agree only because the clamp lands both on the same number. Below it, the
locator UPSAMPLES — which is exactly what this document forbids the corpus to do.** *Two real
pairs hit it today.*

    SO: EVERY CORPUS CASE CARRIES ALL THREE CONVERSIONS, AND THE PROFILE IS A
    FIRST-CLASS FIELD. `THE CONSTRUCTION IS A VARIABLE, NOT THE APPARATUS`
    applied one level up -- from how a CUT is made to how the AUDIO is made.

*And this is `vmsam-dev-3`'s nine-literal `-ar 44100` audit reaching the same defect from the
opposite end: it found three rules for one constant; ci-build found one word in a spec where
three consumers exist. Same object, two entry points, neither agent aware of the other.*

### And a converted artefact must carry the identity of its converter

**A stored conversion is FROZEN. The conversion parameters are not.** *`PLATEAU_TOLERANCE_MS`
went 50 → 5 and `MIN_STEP_MS` 60 → 5 in one evening; the comparison rate is derived per pair;
`int(round(23.976))` = 24 discards a fractional rate before anything is hashed.* **A corpus
holding converted audio from an older converter tests a version nobody is running, and it
reports PASS while doing it.**

    EVERY CONVERTED ARTEFACT RECORDS WHAT PRODUCED IT: the parameters, and the
    revision of the code that applied them. A conversion whose converter is not
    named is a fossil that cannot be told apart from a current one.

*This is `WRITE_ZONES.MD` §9 — two copies drift — in the corpus itself, and it has already
happened once: `fe0fec8`, "ce corpus n'est pas la population que le conteneur produit
maintenant".*

**AND KEEPING BOTH 1 AND 2 MAKES THE PAIR SELF-CHECKING, which is the reason to hold the
unconverted source even though nothing reads it directly:** *re-run the conversion on the
stored source and compare with the stored conversion.* **A difference is converter drift,
detected by construction rather than by someone noticing a number moved.** *That is the only
check in this design that does not depend on anybody remembering to look.*

### THE SELF-CHECK NEEDS ITS OWN NULL, AND THE ARCHITECT PROPOSED IT WITHOUT ONE

**The converter self-check — re-run the conversion on the stored source, compare, a
difference is drift — ASSUMES that a re-run with the SAME converter reproduces the SAME
output. NOBODY HAS ESTABLISHED THAT** (`vmsam-forensic`, against the architect's own design,
minutes after it was written).

> **If the assumption is false the check reports drift on EVERY case, someone bolts on a
> tolerance, and THE TOLERANCE HIDES THE REAL DRIFT THE CHECK EXISTS TO CATCH.**

*That is ci's 10 ms tolerance discriminating quantities of 4 to 6 ms — arriving in a new
instrument before the instrument exists.*

    PRECONDITION, BEFORE THE CHECK IS TRUSTED:
      convert one stored source TWICE, same converter revision.
      Outputs identical -> the check can attribute a difference to drift.
      Outputs differ    -> it cannot separate drift from nondeterminism and
                           MUST NOT BE USED until that is characterised.

**AND COMPARE THE SAMPLES, NOT THE FILE BYTES.** *A container carries `creation_time` and
encoder tags, so a byte-compare reports a difference on every honest re-run.* **The claim is
about the audio, so the comparison must be about the audio** — *a null that fails for a
reason unrelated to the thing under test is how a good check gets abandoned in week two.*

### THE CONSTRUCTION IS A VARIABLE, NOT THE APPARATUS

**§7's rule — ground truth from the CONSTRUCTION — HAS SINCE BEEN CORRECTED BY ITS OWN
AUTHOR, and the architect relayed the SUPERSEDED form to the corpus builder after the
retraction:**

    WRONG:    "the truth is 40.000 because you cut there, not because anything
               measured it."
    CURRENT:  THE CONSTRUCTION'S ACTUAL OUTPUT MUST BE MEASURED, BY AN INSTRUMENT
              INDEPENDENT OF THE ONE UNDER TEST. THE INTENDED PARAMETER IS NOT
              THE TRUTH. THE REALISED ONE IS.

*`vmsam-dev-3` refuted it with a real fixture — INTENDED 5.000 s, ACTUAL 4.088 s. And
`vmsam-ci-build`'s own arms: nominal 5.000, realised 4.9673 and 4.9905 depending on the
construction, and the no-cut arm GAINED 80 SAMPLES to encoder priming.* **Three numbers the
retracted version would have recorded as "5.000, 5.000, 0.000".** *The half that stands: a
case whose truth came out of THE INSTRUMENT UNDER TEST tests it against itself. What was
wrong was believing a construction's INTENT is its OUTPUT.*

**And the rule below — the construction is a variable — was written
believing the construction was neutral. It is not.**

    dev-1   two-pass CONCAT        a real encoder discontinuity AT THE JOIN
    dev-3   RE-ENCODE across it    x264 smooths exactly the frames carrying
                                   the signal

**Two agents disagree about whether a 5 s cut is detectable at a given band, and the likely
cause is not the instrument — it is that they cut differently.** *A corpus built one way
would have SETTLED that disagreement silently instead of testing it, and every number
downstream would have been internally consistent.*

    EVERY SYNTHETIC CUT CASE IS BUILT TWICE: concat-at-the-join, AND re-encoded
    across it. The construction is recorded per case as a FIRST-CLASS FIELD.
    THE CONSTRUCTION IS PART OF THE CASE, NOT PART OF THE APPARATUS.

### And the first measured statement of what an instrument CANNOT see

**Band sweep with the encode held constant** (`vmsam-dev-3`):

    cut arm      850 · 1496 · 1284 · 34 · 10 · 10   as the band widens
    no-cut arm    10 ·   10 ·   10 · 10 · 10 · 10   FLAT THROUGHOUT

**The threshold sits exactly at the shift. The band is `0.20 x L`, so nothing under ~43 s
registers on a 214 s window.** *The cut is not mis-ranked — it is INVISIBLE.*

> **A CONTROL THAT STAYS FLAT WHILE THE TREATMENT MOVES IS THE FIRST THING THIS CAMPAIGN
> HAS PRODUCED THAT SAYS WHAT AN INSTRUMENT CANNOT SEE, rather than what it got wrong.**

*And it was NOT transferred to the audio locator. `PROBE_STEP 40 s`, `PROBE_WINDOW 60 s`,
`REFINE_STEP 4 s` look invitingly analogous and there is no evidence the failure mode
carries.* **Recorded as a question, not as an analogy — which is the restraint the 1839x
retraction was about.**

### The corpus is shared

> ***"Ce corpus doit être disponible de tous pour gagner du temps."*** — `VMSAM_HELP_AI/corpus/`

**Outside the repository, because it carries media names (`WRITE_ZONES.MD` §10).** *Every
agent built a private corpus tonight and each was blind to a different class: 24 completed
files cannot contain an absent field; a synthetic pair cannot contain total redundancy; a
fixture at half a segment step sits in a regime the corpus never reaches.*

> **A SHARED CORPUS THAT NOBODY IS OBLIGED TO CONSULT PRODUCES THE SAME LOSS AS NO CORPUS.**
> *Measured tonight: `vmsam-dev-4`'s register held two locator gates correctly filed and
> prior, and two agents rediscovered them by hand because consulting it was optional.*

**Recommended, pending the owner's word: consulting the index is REQUIRED before building a
fixture.**

### THREE RULINGS, 2026-09-05 — and each one CLOSES something that was blocking

**1. WHERE THE CORPUS LIVES — SETTLED, AND IT WAS ALREADY RIGHT.** *"dans le container de dev.
il est dans le bon dossier que l'on avait défini. le `VMSAM_HELP_AI/corpus`. il est bien là."*

    R19 IS CLOSED. The location is the intended one and no move is required.
    Stop escalating it; stop deferring corpus work on it.

**The escalation was not wrong to raise and was wrong to hold work.** *The question "is one
untracked copy durable" is a real question; the answer was that the owner had already chosen
this location deliberately.* **An architect can block a fleet on a question the owner
considers answered** — and the tell was available the whole time: **the folder was the one we
had defined together.** *A concern about an arrangement someone chose on purpose is a
question, not a veto.* **Keep the MANIFEST anyway — opaque ids, durations, realised
conversion parameters — as cheap insurance, not as a condition.**

**2. FILE SELECTION IS THE AGENTS' — `"les agents peuvent choisir les fichiers au besoin"`.**

    THE CORPUS IS NOT A FIXED LIST TO BE APPROVED. Folders 159 and 164 remain
    mandatory; beyond them an agent PICKS WHAT ITS QUESTION NEEDS.
    Audio-only still holds (`"pas besoin de la vidéo. Juste les audios"`).

**THIS DISSOLVES `ci-build`'s TIER-1 BLOCKER, and by dissolving rather than by exception.**
*The unbuildable null was "same content, no cut, DIFFERENT FRAME RATE", believed to require
video.* **It does not: a 23.976 → 25 conversion is a 4.27 % DURATION RATIO, and a duration
ratio is fully present in audio.** *What the null actually needed was not a video stream — it
was PERMISSION TO GO AND FIND A REAL PAL/NTSC PAIR IN THE LIBRARY instead of synthesising
one.* **The blocker was a selection constraint mistaken for a format constraint**, and it sat
open overnight because it was described in terms of the artefact rather than the measurement.

    A NULL IS DEFINED BY THE QUANTITY IT MUST HOLD CONSTANT, NOT BY THE
    CONTAINER IT ARRIVES IN. Ask what the arm must MEASURE before declaring
    it unbuildable.

**3. THE REPORT IS SCOPED TO THE CURRENT TEST MODE — `"pour les html, l'important c'est le
mode de test actuel"`.**

    `write_log_file` IS THE WHOLE REQUIREMENT. The production path
    (`apply_production_outcome`, which writes no log) IS OUT OF SCOPE.
    NO FURTHER ENGINE EDIT IS AUTHORISED FOR THE REPORT.

*The gap was real and reported; the owner has priced it and declined it.* **A defect the
owner has been shown and has chosen not to fix is CLOSED, not pending** — and re-raising it
costs the credibility of the next escalation.

### F4 MEASURED — THE PLACEMENT FAILS IN 64 % OF POSITIONS, AND IT IS NOT A CORNER CASE

**`vmsam-auditor`, sweeping the true cut across a full `PROBE_STEP` (40 s), 11 positions,
sample-exact 5.000 s removals of known ground truth.** *Blob `d31f6837` — verified identical
at the three references cited and at the published tip.*

    cut_s   narrowed   bracket_ms          width_ms
    100.0   False       40000 -> 140000     100000
    104.0   False       40000 -> 140000     100000
    108.0   False       40000 -> 140000     100000
    112.0   True       108000 -> 120000      12000
    116.0   True       112000 -> 124000      12000
    120.0   False       80000 -> 180000     100000
    124.0   False       80000 -> 180000     100000
    128.0   False       80000 -> 180000     100000
    132.0   False       80000 -> 180000     100000
    136.0   True       132000 -> 152000      20000
    140.0   True       136000 -> 152000      16000

    BOUND-ONLY  7/11 = 64 %, width 100 000 ms EVERY TIME
    NARROWED    4/11 = 36 %, widths 12 000 / 16 000 / 20 000 ms
    STEP MAGNITUDE  -5000.00 ms AT ALL ELEVEN POSITIONS. Exact, every time.

> **THE MEASUREMENT IS FLAWLESS AND THE PLACEMENT FAILS IN NEARLY TWO CASES OUT OF THREE.**
> **THERE IS NO POSITION IN THE SWEEP WHERE THE BRACKET MATCHED THE CUT** — the best case is
> still 12 000 ms against a 5 000 ms truth.

**This is the owner's sentence — `"la partie rajoutée, il l'a trop allongé"` — and the
distribution CHANGES ITS CHARACTER: not a corner case that occasionally lands badly, but the
MAJORITY OUTCOME on a uniformly-placed cut.**

### The refine loop is innocent. The defect is upstream, in the run boundary

**The auditor's own earlier inference — *"fails when the transition sits within ~2 ×
REFINE_STEP of `after["first"]`"* — WAS FALSE, AND IT RETRACTED IT ON THE MEASUREMENT.** *It
predicted a narrow corner; the dead zone is 64 % of the period and the mechanism is a
different one.*

    MEASURED: in EVERY failing case the true cut lies OUTSIDE the refine region
    entirely, above `region_end`. Cuts at 100-108 got region [40, 80];
    cuts at 120-132 got region [80, 120].
    `_bracket_transition` cannot find an "after" sample BECAUSE THERE IS NO
    TRANSITION INSIDE THE REGION TO FIND.

**The originally-reported case — the cut sitting exactly at `region_end` — is the MILDEST
member of the family, not the characteristic one.** *A finding's first instance is not
evidence about its distribution, and this is the campaign's cleanest demonstration: the first
case understated the defect by a factor of nearly two in frequency and misidentified the
mechanism entirely.*

**INFERRED, strongly supported, offered as mechanism and not as measurement:**

    PROBE_WINDOW (60 s) EXCEEDS PROBE_STEP (40 s) -- verified, a 20 s overlap.
    A coarse window straddling the cut is dominated by whichever side fills
    more of its 60 s and is grouped into that run, so `after["first"]` can name
    a probe whose window STARTS UP TO 60 s BEFORE the true transition,
    putting `region_end` BELOW the cut.

**The observable for a repair, and it needs no author's judgement:** *the sweep returns
**0/11 bound-only** and **no bracket width equals `SEARCH_BOUND_MS`***. The auditor landed the
script and deliberately did not look at what a corrected boundary rule would be — **the value
of a repair-observable written by someone who has not designed the repair.**

### And the hop LIES rather than staying silent — a seventh substrate, measured

**The auditor's "pull first" discipline was measuring a MIRROR.** *`origin` for a container
checkout is the intermediary, which serves its own `refs/heads/dev-AI` and lags forgejo in
15–22 minute bursts.*

    forgejo 11d26b8  |  intermediary f2f0767  |  auditor f2f0767
    `git fetch origin` said UP TO DATE while the published branch was TWO COMMITS AHEAD.

> **THIS IS NOT SILENCE, IT IS AN ACTIVE FALSE STATEMENT OF CURRENCY** — §26 and §32 again:
> *the answer resolves, nothing errors, and it names the wrong thing.*

**Every "pulled to tip" reported from a container was a statement about the mirror.** *No
conclusion changed — the F4 blob was re-checked at the true tip and is identical — but the
instrument was wrong, and it could not have been found by examining its results, ONLY BY
EXAMINING THE TOPOLOGY.* **The repair: print all three refs and the true lag every beat, and
say `lag UNKNOWN, do not read silence as current` when forgejo is unreachable.**

### THE GIGABYTES THAT LOOK ALARMING ARE THE CHEAP ONES — a correction that survives R19's closure

**Reported to the architect as *"4.5 GB unbacked"*, and `vmsam-arch-aide` corrected its own
framing:** *that phrasing invites a STORAGE answer, and the thing actually at risk is the
REASONING.*

    corpus/           1.3 GB   REGENERABLE from recorded argv -- and ci-build has
                               DEMONSTRATED it, rebuilding a case from scratch.
    everything else   4.5 GB   NO COMMAND REBUILDS ANY OF IT.
                               Ledgers, scans, findings, measurements, retractions.

> **THE GIGABYTES THAT LOOK MOST ALARMING IN A `du` ARE THE ONLY ONES THAT ARE CHEAP TO
> LOSE.** *A size ranking is not a value ranking, and the alarm was raised on the size.*

**The owner has since settled the corpus: it lives where we defined it and no move is
required. THAT RULING DOES NOT COVER THE REASONING, AND THE REASONING DOES NOT NEED IT** —
because the irreplaceable half is **TEXT**, and text with opaque case ids and no media names
is exactly what this repository exists to hold (`WRITE_ZONES.MD` §10: the boundary is the
repository, and finished reports belong inside it).

    THE MEDIA NEEDS A DISK. THE REASONING NEEDS A COMMIT.
    Nothing is blocked on the owner for the second, and it was being treated
    as though it were -- because both were named by one alarm about one number.

**Precondition, and it is the only one:** *an artefact promoted this way carries opaque ids
and realised parameters, never a media filename, an episode name or a path* — and per §32's
correction, **write what must survive INTO THE FILE, never into a commit message**, since the
blob travels the promotion intact and `N−1` messages die by construction.

### F4's CONFOUND DOES NOT MATERIALISE — the locator HAS a speed-relation branch and it fires

**The worry was that a bracket failing to narrow might be a RATE RELATION being read as a
discontinuity, and that the one control separating them was unbuildable. BOTH HALVES ARE NOW
ANSWERED.**

**The arm was buildable all along** — *`atempo` on the existing base, no library file, no
ruling needed:* the 23.976 → 25 relation is a **4.27 % DURATION ratio**, entirely present in
audio. *The blocked note was TRUE ABOUT FRAME RATE AND RIGHT ABOUT THE WRONG QUANTITY.*

**MEASURED, blob `d31f68377580`, positive control fired in the same run:**

    POSITIVE CONTROL   real cut, same rate      -> piecewise_constant, 1 point, step -4976.75
    sample rate  44100 vs 48000, no cut         -> constant, 0 points, fid 0.9959, off  0.06 ms
    sample rate  44100 vs 32000, no cut         -> constant, 0 points, fid 0.9972, off -0.09 ms
    TEMPO 4.27 % faster, no cut                 -> None, WITH ITS REASON
    TEMPO 4.27 % slower, no cut                 -> None, WITH ITS REASON

    "fidelity low but drift monotone: speed relation suspected; declining"

**THE LOCATOR DOES NOT MISTAKE A RATE RELATION FOR A CUT.** *It has a dedicated branch, the
branch fires in BOTH directions, and it NAMES the relation instead of emitting change points.*
**`None` here means *I could not measure* and never *the files are compatible*** — the module's
own documented distinction, honoured in the one place it matters most.

> **SO A SPEED RELATION NEVER REACHES THE BRACKET STAGE, AND F4's 64 % STANDS ON ITS OWN.**

**And the builder caught its own wrong control before reporting it as the right one:** *the
first arms varied the DECLARED SAMPLE RATE at constant duration — a real confound and NOT the
specified one.* **Two distinct axes, now separated by a `scope_note` so that a pass on one is
never read as covering the other.**

### THE NORMALISATION DEFECT IS FORMAT PROMOTION, NOT GAIN — and that names its dead zone

**MEASURED by `vmsam-auditor` on ffmpeg 9.0.1 / libavcodec 63.1.101, THE CURRENT MAJOR — so it
is not an old-binary artefact of the corpus container.** *The mechanism is not the amount of
gain applied. It is the SAMPLE FORMAT the gain filter drags the chain into.*

    src/video.py, blob 7808e185, UNCHANGED at the tip:
      :530   filter_str = "anull"                    # pass-through
      :532   filter_str = f"volume={gain_db:.2f}dB"

    `anull`  keeps the chain in s16p and is SAFE.
    ANY `volume` filter PROMOTES IT TO fltp, where the marginal pole ACCUMULATES.

> **A GAIN OF 0.00 dB DESTROYS THE FILE. THE SAME 10 dB WITH THE BIQUAD FORCED TO s16 IS
> CLEAN.** *The magnitude of the correction is irrelevant; taking the branch at all is the
> defect.*

**THE CORRECTION BELOW WAS ITSELF WRONG AND WAS WITHDRAWN TWENTY MINUTES LATER. IT IS KEPT
BECAUSE THE SEQUENCE IS THE LESSON — see `THE CLAMP IS LOAD-BEARING AFTER ALL` at the end of
this section.**

**CORRECTED WITHIN THE HOUR, BEFORE THIS REACHED THE OWNER — the float path is ONE of two
necessary conditions and NOT the trigger.** *`vmsam-arch-aide` closed the open filter-graph gap
by reading ffmpeg's own graph log, and it REFUTES the causal chain while leaving the severity
untouched.*

    [WITHDRAWN BLOCK -- true of a ONE-COMMAND harness, FALSE of the engine, which
     is TWO ffmpeg passes with an intermediate file. See "THE CLAMP IS LOAD-BEARING
     AFTER ALL" below. Every number here is real and describes a system we do not run.]
    `-ar` IS AN OUTPUT OPTION. THE RESAMPLE IS INSERTED **AFTER** THE FILTER CHAIN.
      48000 source, -ar 32000 :  lowpass a=1.612465 1.000000 0.387535   NOT DEGENERATE
                                 filter runs at 48000; resample to 32000 comes after
                                 rms 1537.7, peak 6715, 0.00 % full scale -- CLEAN
      32000 source, same chain:  lowpass a=1.000000 2.000000 1.000000   DEGENERATE

    SAME 32000 SOURCE, VARYING ONLY `-ar`:  32000 -> 39.14 % · 44100 -> 30.57 %
                                            48000 -> 31.07 % · none  -> 39.14 %
    THE CLAMP NEITHER CAUSES NOR PREVENTS IT.     <- WITHDRAWN. THE CLAMP IS LOAD-BEARING.

> **[WITHDRAWN] THE TRIGGER IS A SOURCE STREAM WHOSE OWN SAMPLE RATE IS 32000. Nothing about
> the pair, the minimum, or the clamp is load-bearing.** — **THIS SENTENCE IS WITHDRAWN: the
> trigger is a PAIR whose lowest rate is 32000, and the clamp IS load-bearing.** *`lowpass=f=16000` sits EXACTLY at Nyquist for
> 32000 — arithmetic: 32000/2 = 16000 — so the biquad collapses to a=[1,2,1]; and it runs away
> ONLY in float, which the `volume` stage forces and `anull` avoids.*

    TRIGGER      a source stream natively at 32000 Hz reaching the normaliser
    SECOND TERM  the float path, forced by `volume`. `anull` stays integer and is safe.
    SEVERITY     94-99 % of the affected STREAM. UNCHANGED.
    POPULATION   34 files, one work, 0.192 % of 17 694 -- UNCHANGED: those are exactly
                 the files carrying a 32000 stream.
    PER FILE     ONE stream destroyed, NOT every stream of that language. The clean
                 48000 stream in the same file is filtered at 48000 where the pole is
                 not degenerate, downsampled afterwards, AND SURVIVES.

**The refuted claim was stated as a CODE TRACE rather than a run, which is why it was cheap to
overturn** — *`WRITE_ZONES.MD` §46's fourth instance, and its author filed it against itself:*
**a trace reported as though it described the system.**

### THE CLAMP IS LOAD-BEARING AFTER ALL — the correction was refuted and the ORIGINAL chain stands

**`vmsam-ci-build` refuted the correction; `vmsam-arch-aide` verified the refutation against
itself and withdrew its own claim. VERIFIED HERE FROM THE PUBLISHED CODE BEFORE RE-WRITING
THIS.**

    THE NORMALISER IS **TWO** ffmpeg PASSES WITH AN INTERMEDIATE FILE:
      video.py:500   launch(cmd_extract)      PASS 1 -> WRITES THE TMP FILE
      video.py:533   cmd_normalisation = [..., "-i", name_out_file_tmp, ...]
      video.py:538   ... "-af", "highpass=f=60,lowpass=f=16000,{filter_str}"
                                               PASS 2 -> FILTERS THAT FILE

    AND PASS 1 CARRIES THE RATE:
      video.py:~268  codec_param.extend(["-ar", exportParam['SamplingRate']])
      video.py:~271  baseCommand.extend(codec_param)   -> cmd_extract

    THE CLAMP, mergeVideo.py:584 -- ONE-SIDED:
      SamplingRate = get_less_sampling_rate(pair)
      if int(SamplingRate) > 44100: SamplingRate = "44100"
      -> CAPS FROM ABOVE ONLY. 32000 SURVIVES AS THE EXTRACTION RATE.

> **PASS 1 RESAMPLES TO A FILE. PASS 2 FILTERS THE FILE. THE FILTER SEES THE ALREADY-RESAMPLED
> RATE, AND THE CLAMP IS EXACTLY WHAT SETS IT.**

**The deciding experiment, one 48000 source, same filter, same gain:**

    A) ONE command, `-ar 32000` as an output option    a=1.612,1.000,0.388   0.00 % full
    B) TWO commands, resample to a file THEN filter     a=1,2,1              83.07 % full
    THE ENGINE IS (B). THE REFUTED MEASUREMENT WAS (A).

**And the "clean 48 kHz stream survives" retraction was ALSO wrong — the original claim was
right.** *One master stream, three extraction profiles: realised 48000 OK, realised 32000
SATURATED, realised 44100 OK.* **stage 1 at 48000 PROVES the source is 48000 — a 32000 source
cannot yield 48000 — so a natively-clean stream is extracted DOWN to 32000 because the pair's
other stream is 32000, and is destroyed there.**

    TRIGGER      a pair whose LOWEST stream rate for the selected language is
                 EXACTLY 32000 -- NOT "sub-44100", see the narrowing below
    THE CLAMP    one-sided, so 32000 survives as the extraction rate -- LOAD-BEARING
    SEVERITY     94-99 % of the affected ARM. UNCHANGED THROUGHOUT ALL THREE VERSIONS.
    POPULATION   34 files, one work, 0.192 %. UNCHANGED.
    PER FILE     EVERY stream of the affected language, INCLUDING NATIVELY-48000 ONES.

**GAP 2 IS RE-OPENED: the claim that it was closed is withdrawn by the agent that made it.**
*What was actually established is the semantics of `-ar` in a SINGLE command; the engine's
ordering rests on a two-pass CODE READING and deserves the same standing as any other trace
until a run confirms it.*

> **THE SEVERITY NEVER MOVED ACROSS THREE REVERSALS OF THE CAUSAL CHAIN.** *`94–99 %` was
> measured on real arms and survived every explanation of WHY.* **A measured effect and its
> mechanism are separable claims, and only one of them was ever in doubt.**

**And the sensitivity is why three agents got three different tables (`WRITE_ZONES.MD` §46):**
*a perturbation of ONE PART IN 2 560 flips a case from 50 % destroyed to never firing.* **An s16
WAV intermediate anywhere upstream quantises the input and the defect vanishes** — which is
also why a smoke test can pass while production burns.

**Two gaps remain open and NO AGENT HAS CLAIMED EITHER:** *no end-to-end production merge has
been run, and the filter-graph ORDER relative to the auto-inserted resample is unverified* —
**a standalone `-af` chain has no auto-inserted resample to be ordered against, so the
reproduction structurally cannot answer it.**

### THE POPULATION IS EXACTLY 32000. "SUB-44100" IS THE WRONG CLASS AND OVERSTATES IT

**`vmsam-auditor` measured the neighbouring rates instead of assuming the class, and it
NARROWS the exposure. VERIFIED HERE BY ARITHMETIC BEFORE ACCEPTING IT.**

    24000   Nyquist 12000   cutoff 16000 ABOVE Nyquist   rms 639.0 ->   478.9   0.00%  CLEAN
    32000   Nyquist 16000   cutoff 16000 AT   Nyquist    rms 640.1 -> 27064.1  64.75%  DESTROYED
    44100   Nyquist 22050   cutoff below                 rms 638.2 ->   490.8   0.00%  CLEAN
    48000   Nyquist 24000   cutoff below                 rms 639.1 ->   490.4   0.00%  CLEAN

    cutoff/Nyquist:  8000 -> 4.000 · 16000 -> 2.000 · 22050 -> 1.451 · 24000 -> 1.333
                     32000 -> 1.000  <- THE ONLY EQUALITY
                     44100 -> 0.726 · 48000 -> 0.667

> **THE TRIGGER IS `cutoff == Nyquist` EXACTLY — A DOUBLE POLE ON THE UNIT CIRCLE. A CUTOFF
> *ABOVE* NYQUIST DOES NOT DEGENERATE.** *For a 16000 Hz lowpass that is a stream rate of
> exactly 32000 and nothing else.*

**"SUB-44100" SWEEPS IN 24000, 22050, 16000 AND 8000 STREAMS THAT ARE MEASURED CLEAN.** *The
count of 34 affected files in a 17 694-file library was right all along; the loose phrasing
around it was not, and it had reached at least one agent's standing prompt.*

    INFERRED: any other cutoff constant would move the degenerate rate to exactly
    twice it. Only the 16000 at `video.py:538` is in play, so 32000 is the only
    affected rate in this codebase.

**THIS NARROWS THE EXPOSURE RATHER THAN WIDENING IT, WHICH IS WHY IT WAS SENT TO BE CHECKED
RATHER THAN ACTED ON** — *the agent explicitly declined to have the owner told the population
is smaller on its word alone.* **A correction that makes a defect look SMALLER is the one that
most needs a second reader, and asking for one is the opposite of the reflex.**

### And the run has produced NO 32000 audio, so it has not closed gap 1 — yet

    800 audio-bearing output files sampled · 1762 streams
    48000: 1672 · 44100: 86 · 24000: 4 · 32000: ZERO

**m = 0. THE DEFECT HAS HAD NO OPPORTUNITY TO MANIFEST, so "no saturation observed in this
run" is an UNTESTED CELL and not reassurance** — the opportunity count applied to a production
run instead of to a scan.

**CAVEAT THAT MUST TRAVEL WITH IT: this is a reading of the OUTPUT, mid-run.** *The
trigger-carrying id was confirmed present in the live queue.* **An absent output means NOT YET
REACHED, or REACHED AND DECLINED — and those are different states which this measurement
cannot separate.** *A decline is an artefact of the run; it is not a produced audio stream.*

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

### AND IT CHANGES WHAT "PROGRESS" MEANS

**A count of merged files is not progress unless every member was produced by the current
revision.** *The queue position, the file count and the elapsed time are all measures of a run;*
**NONE OF THEM MEASURES THE END CONDITION, WHICH IS A PROPERTY OF THE LATEST CODE ACROSS THE
WHOLE LIBRARY.** *A campaign that lands a locator fix at 90 % coverage is back to zero coverage
of the new revision, and that is not a regression — it is the condition working as the owner
defined it.*

**`ci` HOLDS IT. Not the architect, not the Lead, not a document.** *It is a state of the queue,
measurable by the seat that runs the queue, and nobody else's to declare.*

**THIS REPLACES `Resolved` / `Unmergeable_Verified WITH PROOF`.** *The old form asked every
non-merge to carry a justification of its own class. The new one asks ONE question of each
non-merge:* **is this file completely different from the master — yes or no?**

### AND IT MAKES §95 LOAD-BEARING RATHER THAN MERELY EVIDENTIAL

**THE EXCLUSION CATEGORY IS EXACTLY WHAT THE LOCATOR'S NO-SHARED-CONTENT PATH MEASURES:**

    `fidelity at the floor with scattered offsets: no shared content; declining`
    -- eleven probes, a conclusive negative, the module's own words.
    THAT SENTENCE **IS** "completely different from the master".

> **AND IT IS ONE OF THE EIGHT REFUSAL PATHS THAT EMIT NOTHING.** *So the single signal that
> separates a LEGITIMATE EXCLUSION from a FAILURE is the one signal absent from every artefact a
> person can read.*

    FROM THE OUTPUT, "excluded because completely different" AND "declined for
    one of seven other reasons" ARE THE SAME OBSERVABLE: SILENCE.
    THE END CONDITION CANNOT BE EVALUATED FROM WHAT THE PIPELINE EMITS.

**The repair is one emitted line on that path and it is now the campaign's highest-value single
change** — *not because the defect grew, but because the owner's ruling made that exact
distinction the thing the campaign is finished by.*

### WHAT IT SIMPLIFIES, AND THAT IS MOST OF ITS VALUE

**A non-merge no longer needs a PROOF OF UNMERGEABILITY.** *It needs a CLASSIFICATION, and the
classification has two values.* **Every finding about the locator's accuracy, the clamp, the
normalisation and the bracket now serves one question: DOES THIS FILE MERGE, AND IF NOT, IS IT
IN THE EXCLUDED CLASS?**

    AND THE EXCLUDED CLASS IS NOT A SHELTER. A file is excluded because it was
    MEASURED to share no content — not because it refused, not because it errored,
    and not because nobody reached it.

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


### THE SHIPPED FUNCTION WAS CALLED AND IT DESTROYS THE AUDIO — verification kind RUN

**FIRST CLAIM OF THIS CAMPAIGN TO CROSS FROM `TRACE` TO `RUN` UNDER THE OWNER'S RULING, WITHIN
AN HOUR OF THE RULING.** *`video.generate_normalised_file()` — the shipped function — imported
from `src/` and CALLED, on material whose ground truth was set by construction.*

    SOURCE, by construction: 300 s · 48000 Hz · pink noise 0.1
      peak 2332 · rms 450.6 · full-scale 0.00%   NOT CLIPPED

    CALLED WITH -ar 32000   peak 32768 · rms 32144.5 · FULL-SCALE 94.89%
                            onset 10 s · 96.67 % OF THE FILE DESTROYED
    CALLED WITH -ar 44100   peak  6062 · rms  1352.7 · full-scale  0.00%

**SAME SOURCE. SAME SHIPPED FUNCTION. ONLY THE SAMPLING RATE DIFFERS — WHICH IS EXACTLY WHAT
THE ONE-SIDED CLAMP SETS.**

### AND THE ENGINE IS SILENT AT THE MOMENT OF DESTRUCTION

> **`tools.logs` AFTER THE DESTRUCTIVE CALL: ZERO ENTRIES.**

**VERIFIED INDEPENDENTLY FROM THE SOURCE: the function contains NO emission of any kind — no
`tools.logs`, no warning, no degraded-path line — and the normalisation command is launched and
its output NEVER INSPECTED.** *There is one `print` in the whole function and it fires only when
loudness analysis fails.*

    SO NO LOG ANYWHERE IN THE PIPELINE WOULD HAVE CAUGHT THIS, AT ANY POINT.
    The absence of evidence was a property of the emitter, not of the defect —
    §43, arriving at the exact site the defect lives at.

### WHAT IT CLOSES AND WHAT IT DOES NOT — the split is the finder's own

    CLOSED, BY RUN   the shipped normalisation function DESTROYS AUDIO when handed 32000.
    STILL TRACE      that a real pair REACHES it with SamplingRate = 32000.
                     `get_less_sampling_rate` plus the one-sided clamp: READ, NOT EXECUTED.

**AND THE HONEST REMAINDER, VOLUNTEERED: THE FUNCTION IS RUN, ITS ARGUMENTS ARE TRACE** — the
caller's `cmd_extract` and `codec_param` were CONSTRUCTED by following the extraction path
rather than by executing it. **A full merge is still the only thing that closes the second half,
and it is not reported as closed.**

**EVERY PRIOR RESULT ON THIS DEFECT IS RE-LABELLED BY ITS OWN HOLDER:** *the format-promotion
controls, the two-command experiment and the 94–99 % on seven corpus arms are all* **HARNESS** —
*real measurements of something we do not run* — *and the two-pass reading is* **TRACE.**
**ONE RUN NOW CARRIES WHAT THREE REPRODUCTIONS COULD NOT.**

### THE FALSIFIER, REGISTERED BEFORE THE RUN — AND THE LINE THAT MATTERS MOST IN IT

**`vmsam-arch-aide` wrote down what would REFUTE the end condition BEFORE any scoring run existed,
which is the technique §102 vindicated. Its most valuable line was registered while nobody yet had
an incentive to disagree with it:**

> **THE END CONDITION IS NOT EVALUABLE TODAY. NOT "HARD" — NOT EVALUABLE.** *Two emissions are
> missing and either one alone prevents scoring.* **ANY DECLARATION THAT IT IS MET, MADE BEFORE
> BOTH EXIST, IS FALSE BY CONSTRUCTION — WHATEVER NUMBERS ARE ATTACHED TO IT.**

**THE FOUR CLAUSES, EACH CARRYING ITS OWN REFUTATION:**

    1  REF      three ref reads must AGREE IN ONE SNAPSHOT, `ls-remote` at the URL.
                Disagreement VOIDS the score rather than delaying it -- §103's own
                reason: a divergence window is content-preserving on both sides and
                would score CLEAN while the ancestry is disconnected.
    2  DENOM    reported as §108 requires -- TWO REFS, AN INSTANT, AND THE COMMAND --
                never as a bare count. A census taken in another hour is a moment
                that has passed. A seat whose substrate cannot see both refs DOES
                NOT HOLD THE NUMBER and says so instead of estimating it.
    3  DATING   counts only on a ROW-LEVEL field resolving to the scoring commit.
                **UNDATABLE COUNTS AS NOT MERGED** -- not a failure, not a pass.
                REFUTED IF anything is counted on a field CONSTANT ACROSS ROWS:
                a field that cannot vary within a run cannot date a row within a run.
    4  EXCLUDE  AN ABSENT LINE IS NOT AN EXCLUSION. Print `excluded_claimed` beside
                `excluded_with_stated_cause`; a reader who sees them differ knows.

**CLAUSE 4 IS `m = 0` APPLIED TO THE END CONDITION ITSELF, and it is the one to defend hardest.**
*Subtracting SILENT files from the denominator is how a campaign declares victory over the files it
never examined — and from outside, that is indistinguishable from having merged them.*

**WHAT THE FALSIFIER CANNOT DO, recorded here so the citation is harder later:**

    IT CANNOT VERIFY A MERGE IS CORRECT. It scores WHETHER a file was merged at the
    current revision, NEVER whether the output is right.
    *** A PIPELINE THAT MERGES EVERY FILE WRONGLY SATISFIES EVERY CLAUSE ABOVE. ***

> **ITS LIKELIEST FAILURE IS NOT BEING WRONG. IT IS BEING CITED AS THOUGH PASSING IT MEANT THE
> MERGES WERE GOOD.**

**AND ONE DEFECT IN ITS TRANSPORT, MEASURED, WHICH DOES NOT DIMINISH THE ACT:** *it was published at
`refs/agents/arch-aide`, and `git ls-remote origin 'refs/agents/*'` returns **0** — the authority
carries four heads and none of them holds the artefact.* **See §106: a falsifier on a ref nobody
else can fetch is, from every clone but its own, the same observable as one never written. LAND IT
ON A HEAD THE AUTHORITY CARRIES AND IT DOES ITS WHOLE JOB.**

*It is a falsifier OFFERED to the holder, never a verdict taken from it: `ci` holds the end
condition, and if `ci` disagrees with a clause it should amend it BEFORE the run rather than score
around it after.*
