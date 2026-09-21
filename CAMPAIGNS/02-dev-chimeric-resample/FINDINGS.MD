# Campaign 2 — findings, derivations and refutations

*Moved out of `CAMPAIGN.MD` on the owner's instruction, 2026-09-08, so that document could be
lightened without losing anything. Every line below stood in `CAMPAIGN.MD` and is unchanged.
`CAMPAIGN.MD` keeps what BINDS; this file keeps what was MEASURED to establish it.*

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

> ***RETRACTED 2026-09-08 — THE MEASUREMENT STANDS AND THE GENERALISATION DOES NOT.*** *The
> evidence above is* **ONE RATIO IN TWO SIGNS**: *4.27 % faster and 4.27 % slower. One point,
> measured twice, is not a band — and the sentence generalises it to every speed relation.*

    `dev-1`'s ladder, 0.0002 -> 0.02 ..... THE PLAN IS *** ACCEPTED *** FROM 0.0002 TO 0.005
    `ci`'s sealed eight-rung ladder ...... the arm's threshold sits in [2.491 %, 4.267 %],
                                          and its rung at 2.49 % FIRES NOTHING

> ***SO A SPEED RELATION DOES REACH THE BRACKET STAGE ANYWHERE BELOW ~2.5 % — WHICH IS THE WHOLE
> 0.1 % NTSC FAMILY, AND 24 AGAINST 23.976 IS THE COMMONEST FILM-TO-VIDEO RELATION THERE IS.***
> **THE ARM IS TRUE AT 4.27 % AND FALSE WHERE REAL NTSC MATERIAL LIVES, SO "F4's 64 % STANDS ON
> ITS OWN" DOES NOT FOLLOW FOR THAT POPULATION.**

*Corroborated at the architect's bench from the other side, and it is the same fact:*
***`speed_relation_suspected` IS REACHABLE ONLY UNDER `median_fidelity < MIN_MEDIAN_FIDELITY`,
WHICH IS 0.70 — AND NTSC AT 0.1001 % READS 0.9614. THE GATE IS NEVER ENTERED, SO THE BRANCH
CANNOT FIRE FOR THE POPULATION THE OWNER NAMED.***

> ***AND THE CLASS IS THE ONE THIS CAMPAIGN HAS NOW MET THREE TIMES: A ONE-POINT TEST GENERALISED
> TO A FAMILY.*** *`dev-1`'s own retracted 8.28x separation, the byte table, and this.* **THREE
> INSTANCES, THREE SEATS, AND EACH TIME THE POINT WAS MEASURED CORRECTLY.**

*`dev-1` found it and declined to adjudicate it —* **"not mine, it is F4's owner's; but they cannot
know without the ladder, and the ladder exists"** *— and the architect could not determine who owns
F4 from any tracked file. The retraction is landed by the seat that landed the document.*

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

### WHERE IT IS, AND HOW TO GET IT — SUPERSEDES THE PARAGRAPH THAT STOOD HERE

    git ls-remote <authority> 'refs/agents/*'
      ->  **refs/agents/arch-aide-artefacts**
    *** NO SHA IS GIVEN HERE ON PURPOSE. *** The one written into this line an
    hour ago was ALREADY STALE when it was read back -- the ref was republished
    in between. **A SHA IN A DOCUMENT IS A PIN (§105). THE REF NAME AND THE
    COMMAND ARE THE CITATION; THE SHA IS THE READER'S TO TAKE.**
    `END_CONDITION_FALSIFIER.md` IS IN THAT REF'S PUBLISH SET.
    TO READ IT:
      git fetch <authority> refs/agents/arch-aide-artefacts \
  && git show FETCH_HEAD:END_CONDITION_FALSIFIER.md

> **THE AUTHORITY *DOES* SERVE IT.** *This document previously said it did not — measured when that
> was true, left standing after it stopped being true, and corrected by its author with a `RUN`.*

**AND THE PRECISE FACT IS MORE USEFUL THAN EITHER VERSION: the default refspec is
`+refs/heads/*:refs/remotes/origin/*`, so NOTHING WALKS `refs/agents/*`.**

> **THIS DOCUMENT PREVIOUSLY PRESCRIBED `+refs/agents/*:refs/remotes/agents/*` HERE — SIX LINES
> ABOVE THE SENTENCE THAT EXPLAINS WHY IT IS PERMANENT.** *No configured refspec matches such a
> namespace, so `--prune` never visits it and no fetch ever updates it: it is immortal and
> invisible.* **MEASURED, AFTER SOMEONE FOLLOWED IT: one clone holds `refs/remotes/aa` frozen at
> a commit the authority has moved past, and `git fetch --all --prune` leaves it untouched.**
> *Found and retracted by its own author (`arch-aide`), who asked for the line it had asked for.*
> **A RULE AND ITS VIOLATION CAN SIT SIX LINES APART AND READ AS A DOCUMENTED FEATURE.** *The end-condition holder
can read the falsifier today with ONE NAMED FETCH and will never trip over it otherwise.*

    **SERVED AND FETCHED ARE DIFFERENT PROPERTIES.** THIS WAS A DISCOVERABILITY GAP,
    NOT A REACHABILITY ONE -- AND THE REMEDY WAS THIS LINE, NOT A NEW LANDING.

*It is a falsifier OFFERED to the holder, never a verdict taken from it: `ci` holds the end
condition, and if `ci` disagrees with a clause it should amend it BEFORE the run rather than score
around it after.*

### THE SPEED BRANCH DOES NOT CATCH THE SPEED FAMILY — AT THE FAMILY'S OWN RATIO

**`vmsam-dev-1` built the arm nobody in the fleet had, swept eight rate errors across two decades,
and closed its own `m = 0` — then found something larger than the gap it was filling.**

     pct  drift/window  fid_median  monotone   branch taken
    0.05          30ms       0.955     False   ACCEPTED
    0.10          60ms       0.953     False   ACCEPTED
    0.25         150ms       0.910     False   ACCEPTED
    0.50         300ms       0.851     False   other refusal (pairing bar)
    1.00         600ms       0.754     False   other refusal (pairing bar)
    2.00       1200ms        0.653      True   *** SPEED -- the only hit ***
    3.00       1800ms        0.597     False   scattered
    4.27       2562ms        0.573     False   scattered

> **4.27 % IS THE 23.976 → 25 PAL RELATION. IT LANDS ON `scattered`, NOT ON `SPEED`. Measured twice,
> two independent constructions.** *That is the canonical speed family — **11 % of the corpus** by
> this document's own table — and it does not reach the branch written to protect it.*

**THE MECHANISM, AND IT IS ARITHMETIC RATHER THAN A BUG:** *at 4.27 % a 60 s window drifts 2562 ms
INTERNALLY, which destroys the correlation INSIDE the window.* **THE SAME RATE ERROR THAT PUSHES
FIDELITY UNDER THE FLOOR DESTROYS THE MONOTONICITY THAT WOULD IDENTIFY IT AS SPEED.**

    below ~2 %   fidelity stays ABOVE the floor  -> never refuses on this path
    at   ~2 %    below floor AND drift coherent  -> SPEED. THE ONLY HIT.
    above ~2 %   drift scrambles the offsets     -> scattered

**The branch's own comment at `change_point_locator.py:924` — verified verbatim — reads *"refusing on
fidelity alone would refuse the whole family the speed repair exists for."* THE INTENTION IS RIGHT
AND THE ARITHMETIC DOES NOT DELIVER IT AT THE FAMILY'S OWN RATIO.** *Same shape as F21: an invariant
established for one object and relied upon for another.*

### CONFIRMED ON REAL FAILURES, 2026-09-05 — AND THE REFUSAL MESSAGE IS FALSE

**`vmsam-forensic` supplied 19 real PAL pairs; `vmsam-dev-3` ran six arms against a SEALED
preregistration and wrote the outcome beside the seal without touching it.**

    case         lang  master/cand  dev from 25/23.976  fid_median  branch
    curated-46   eng     1.042260        0.000449         0.601     scattered
    curated-46   fre     1.041945        0.000764         0.616     scattered
    curated-51   eng     1.043478        0.000769         0.609     scattered
    curated-51   fre     1.042768        0.000059         0.618     scattered
    curated-91   eng     1.042888        0.000179         0.608     scattered
    curated-91   fre     1.042113        0.000597         0.610     scattered

    SIX OF SIX match the PAL ratio to 0.0008.  SIX OF SIX took `scattered`.
    **ZERO took `speed_relation_suspected`.**
    THE VOID CONDITIONS DO NOT APPLY: every arm REACHED the sub-floor branch --
    that branch PRINTED the verdict -- and all six sat BELOW the 0.70 floor.

> **THE MODULE SAYS *"no shared content"* ABOUT PAIRS THAT SHARE ALL THEIR CONTENT AT A 4.27 % RATE
> RATIO.** *The decline may still stand — a speed relation may be genuinely unrepairable here — but
> **THE STATED REASON IS FALSE.***

**TWO INDEPENDENT ROUTES, NEITHER SEAT BUILDING THE OTHER'S ARM:** *`dev-1` reached it synthetically
at 4.27 %; this is the real-material counterpart.* **AND IT IS NON-CIRCULAR ONLY BECAUSE `forensic`'s
classifier NEVER READS THE LOCATOR'S BRANCH** — ffprobe durations and a logged correlation collapse.
*Had it labelled by the branch, the fixture could not have disagreed with the code.*

**THE RETRACTION IT FORCES, PUBLISHED TO FOUR SEATS AND WITHDRAWN BY ITS AUTHOR:** *"seven of eight
`floor_scattered` declines look correct" —* **SIX OF THOSE SEVEN ARE PAL PAIRS.**

**AND THE SEAT NAMED ITS OWN FAILURE:** *it had the duration gaps in its own table — 1324 against
1270 — and read them as "large divergence, plausibly no shared content" **without ever computing the
ratio**.* ***A 53-SECOND GAP AND A 4.27 % RATE RELATION ARE THE SAME OBSERVABLE AT THAT
RESOLUTION.*** *It chose the first reading and measured it accurately.*

**FIXTURE KIND, STATED IN THE SAME BREATH: DRAWN from real failures, therefore EXTENSIBLE** — 16 more
PAL ids remain with both sides on disk, taking `m` from 6 arms to roughly 38 without one
construction. *Not a rate: 3 error ids of 19, of 303 with a computable ratio; the PAL label is a
`TRACE` from 2026-09-03 and was not re-executed; and whether these pairs still fail TODAY was
explicitly not checked.*

**THE LIMITS OF THE SYNTHETIC RUN, STATED BY THAT SEAT ITSELF AND KEPT:** *the grid samples ONE point between 1.00 and 3.00,
so the window is bracketed only as (1.00, 3.00) and is NOT claimed narrower. Synthetic mono 44100, one
source, one 60 s window.* **GROUND TRUTH BY CONSTRUCTION BUYS TRUTH, NOT TYPICALITY. What settles it
is a real PAL pair, and nobody in the fleet holds one.**

**AND IT CHANGES WHAT `dev-3`'s ZERO MEANS — THE TWO NUMBERS MUST TRAVEL TOGETHER:** *`dev-3` measured
0 of 7 monotone across the real corpus, a property of that corpus.* **IF REAL PAL MATERIAL ALSO LANDS
ON `scattered`, THE ZERO IS NOT A SAMPLING ACCIDENT — the branch is not being missed by the sample, it
is being missed by the ARITHMETIC.** *Neither figure is sufficient alone and the seat declined to
report its own without the other. That is §85 obeyed rather than violated: not a join of two relayed
figures, but two independent measurements each stating what it cannot conclude.*

### THE LOCATOR'S REFUSAL CONTRACT — WRITTEN 2026-09-05, BECAUSE NOBODY HAD WRITTEN IT

**`vmsam-dev-4` found the gap under the blocker and it is the architect's to close:**

> ***"What is the locator supposed to emit, and has anyone written it down? THE END CONDITION TURNS
> ON A PRODUCER EMISSION I CAN FIND NO SPECIFICATION FOR."***

**It is right. Every seat has been waiting on a vocabulary NOBODY EVER SPECIFIED — and a boundary
owner cannot emit one it was never given.** *That is a gap BETWEEN two boundaries, invisible from
inside either, which is why it reached the architect rather than the module's author.*

**MEASURED FIRST, so the contract is written against the code and not against an idea. 14 bare
`return None` in `src/change_point_locator.py`:**

    IN THE PUBLIC ENTRY `locate_change_points`   10 sites
      NINE already call `_log(` immediately before returning -- THE CAUSE IS
      ALREADY COMPUTED AND ALREADY PHRASED. It is written to a dev-gated log
      and thrown away at the boundary.
      ONE (:1025) returns with no log at all.
    IN THE HELPERS `_probe` (2) and `_audio_duration_seconds` (2)  4 sites
      INTERNAL. Their `None` PROPAGATES; they are not the boundary.

#### THE CONTRACT

    `locate_change_points` RETURNS `(plan, cause)`.
      plan  == the block already specified, OR None.
      cause == None WHEN A PLAN IS RETURNED.
               A STABLE TOKEN, `snake_case`, WHENEVER plan IS None.
               **NEVER FREE TEXT. NEVER AN EXCEPTION MESSAGE.**

    EVERY ONE OF THE TEN BOUNDARY RETURNS CARRIES A TOKEN. NO EXCEPTIONS,
    INCLUDING THE PATHS A READER BELIEVES CANNOT BE REACHED --
    **`m = 0` IS AN UNTESTED CELL, AND AN UNTOKENED PATH IS INDISTINGUISHABLE
    FROM ONE THAT NEVER RAN.**

    THE NINE PATHS THAT ALREADY LOG: THE TOKEN IS THE LOG LINE'S SUBJECT,
    COMPRESSED. Do not invent a new vocabulary -- the phrasing already exists
    and was written by the seat that knows what the branch means.
    THE TENTH (:1025): its author names it, because nobody else can.
    THE FOUR HELPER RETURNS: the CALLER assigns the token. A helper's `None`
    is not a refusal, it is a missing measurement, and the boundary decides
    what that means.

#### THE TOKEN MUST REACH A LOG LINE — ADDED BY `vmsam-dev-4`, AND THE CONTRACT WAS WRONG WITHOUT IT

**The first version specified a RETURN VALUE. THE END CONDITION DEPENDS ON AN EMISSION.**

> *"My substrate is the log. A return value I cannot see is a return value that does not exist for
> me."*

    THE TOKEN MUST APPEAR IN AN **EMITTED LOG LINE**. A cause that reaches only an
    in-memory structure SATISFIES NOTHING.

**AND THIS IS NOT A HYPOTHETICAL RISK — THE ENGINE HAS FAILED THIS WAY TWICE, BOTH FOUND TONIGHT, IN
THE SAME FUNCTION THE NEW TOKEN WILL FLOW THROUGH:**

    `detail`      accepted, stored, interpolated by **0 of 28** log sites --
                  and its store has **ZERO READERS ANYWHERE**.
    `plan_kind`   computed at the constant-offset refusal, passed as `detail`,
                  **DROPPED** -- which is precisely why the report cannot tell a
                  ROUTING DECISION from a DEAD END.

> **A CONTRACT THAT PERFECTLY SPECIFIES A VALUE AND IS SILENT ON ITS EMISSION HAS ALREADY FAILED
> TWICE HERE.** *If a future consumer routes the cause into `detail` in good faith — and `detail` is
> the parameter whose NAME invites exactly that — **TEN PERFECTLY SPECIFIED TOKENS PRODUCE NOTHING
> ANY READER CAN SEE, AND THE CONTRACT READS AS SATISFIED.***

**IT IS TESTABLE BY GREP WITH A POSITIVE CONTROL, WHICH IS WHY IT IS WRITTEN RATHER THAN ASSUMED.**

*And one line reserved, not required tonight: the contract makes tokens stable and does not say the
vocabulary is CLOSED or enumerated anywhere. A consumer accounting against a known set widens
silently as tokens accrete.* **NAME WHERE THE ENUMERATION LIVES, ONCE IT EXISTS.**

#### WHAT THE TOKEN MUST AND MUST NOT SAY

    IT NAMES **WHAT THE MEASUREMENT FOUND**, never what the consumer should do.
      GOOD   `no_shared_content` · `speed_relation_suspected` · `fidelity_below_floor`
      WRONG  `cannot_repair` · `give_up` -- those are the CONSUMER's conclusions.

    AND IT MUST DISTINGUISH THE TWO SENTENCES `docs/PIPELINE.MD` §3 SEPARATES:
      A MEASUREMENT THAT RAN AND SAID NO  is not
      A MEASUREMENT THAT COULD NOT RUN.
    Today both arrive as `None` and the consumer records the second for both --
    which is the defect this whole contract exists to close.

    **NO MEDIA PATH, NO FILENAME, NO EXCEPTION TEXT.** *An arbitrary exception
    message is not a vocabulary anyone owns, and it travels with whatever the
    library called the file.*

#### WHAT THIS DOES NOT DECIDE

**IT DOES NOT NAME THE TEN TOKENS.** *Only the seat that owns the module knows what each branch
measured, and inventing them from outside would be exactly the "invented, not read" failure the
consumer already refuses.* **THE ARCHITECT SPECIFIES THE SHAPE; THE OWNER OF THE BOUNDARY FILLS IT.**

**AND IT DOES NOT GROW THE CONSUMER'S VOCABULARY BY CORRECTION.** *The three tokens live today are
all authored by the consumer describing its own ignorance, and they are TRUE.* **When these land, the
vocabulary GROWS. `cause_unavailable` does not become wrong — it becomes one branch among several,
and it keeps its meaning: THE PRODUCER SAID NOTHING.**

### R21 — ADJUDICATED BY THE ARCHITECT, 2026-09-05. **YES, IT IS A PROBLEM.**

**The owner routed it: *"Ask directly to the architect if it is a problem."* MEASURED, END TO END:**

    `src/video.py:914-919` -- **A COMMENT, NOT AN `assert`.** It reasons that a
    fallback is harmless because:
      *"mergeVideo clamps the result to 44100 on the very next line, so the
        fallback and the correctly-parsed value CONVERGE for any source at 44100
        OR ABOVE ... It would only matter for a SUB-44100 SOURCE, WHICH THIS
        CORPUS DOES NOT CONTAIN."*

    `src/mergeVideo.py:584` -- **`if int(...) > 44100:` — THE CLAMP IS ONE-SIDED.**
    IT CAPS FROM ABOVE ONLY. **A 32000 SOURCE IS NOT RAISED. IT PASSES THROUGH.**

    THE FACT: `ci-pair` measures **34 files below 44100** in its population;
    `dev-1` names two ids at 32000. **THE CORPUS DOES CONTAIN THEM.**

> **THE COMMENT'S REASONING IS CORRECT AND ITS FACT IS FALSE — AND ITS OWN ESCAPE CLAUSE NAMES
> EXACTLY THE CASE THAT EXISTS.** *"It would only matter for a sub-44100 source" is precise, careful,
> and the sentence that stopped anyone looking.*

**AND THE CONSEQUENCE IS NOT HYPOTHETICAL: `video.py:538` APPLIES `lowpass=f=16000` TO THE
NORMALISED STREAM.** *At 32000 Hz that is **exactly Nyquist** — the degeneracy this campaign already
measured and recorded in the emission module's own text.* **A SOURCE THE COMMENT DECLARES ABSENT IS
THE ONE SOURCE FOR WHICH THE FILTER IS DEGENERATE.**

    THE RULING:
    1. **THE COMMENT IS THE DEFECT.** Not the fallback, which is instrumentation and
       is correctly reasoned. **A COMMENT THAT RULES A CASE OUT IS A GUARD MADE OF
       PROSE, AND IT GUARDS THE READER RATHER THAN THE CODE.** It must be corrected
       to state that sub-44100 sources EXIST AND ARE MEASURED.
    2. **THE ONE-SIDED CLAMP IS A SEPARATE QUESTION AND IT IS NOT MINE TO DECIDE.**
       Raising 32000 to 44100 is a resample the owner has just ruled AGAINST in
       general (R20: derive the rate, preserve the original). **DO NOT "FIX" THE
       CLAMP BY MAKING IT TWO-SIDED WITHOUT SETTLING R20 FIRST -- THE TWO RULINGS
       POINT IN OPPOSITE DIRECTIONS AND ONLY ONE OF THEM IS ABOUT THIS FILE.**
    3. **`video.py` IS FROZEN AND IS NOBODY'S BOUNDARY.** The correction is one
       comment. It goes to whoever the owner assigns, or it waits -- but it is
       recorded here so the next reader does not inherit the false fact.

#### IT IS NOT A TRACE. SOMEBODY RAN IT.

**`vmsam-arch-aide` EXECUTED the shipped function at 11:0x, BEFORE any of this discussion, ON A PAIR
SHAPED LIKE THE MEASURED CASE:**

    `get_less_sampling_rate()` on a sub-44100 pair -> returns **'32000'**
    AND THE CLAMP LEAVES IT.
    CONTROLS: 48000/48000 -> '48000' -> clamped to '44100' · 96000 -> '44100'.

**AND THE CLAMP, RE-MEASURED BY THE ARCHITECT ACROSS THE WHOLE FILE:**

    `> 44100` -> **1**    `< 44100` -> **0**    CONTROL `SamplingRate` -> 15
    *** ONE-SIDED, CONFIRMED. ***

> **A FALSE COMMENT, A ONE-SIDED CLAMP, AND AN EXECUTED COUNTER-EXAMPLE.** *Three seats found the
> false claim independently BY READING; a fourth had already run the mechanism.* **NEITHER METHOD
> NEEDED THE OTHER.**

#### AND THE COMMENT'S OWN `m` IS THE DEFECT THIS CAMPAIGN SPENT THE DAY FINDING IN ITS INSTRUMENTS

    THE COMMENT'S POPULATION ...... 1,312 audio streams in 359 files
    THE MEASURED POPULATION ....... 17,694 readable files, **34 CARRYING 32000**

> ***`m` STATED FROM THE SAMPLE RATHER THAN FROM THE MATERIAL — SITTING IN `src/` SINCE BEFORE THIS
> CAMPAIGN STARTED.***

**AND THE COMMENT SAYS WHY IT EXISTS:** *"Logged to find out whether that ever changes, not because
it is currently wrong."* **IT WAS ALREADY WRONG WHEN WRITTEN. THE LOG WAS WATCHING FOR A CHANGE THAT
HAD ALREADY HAPPENED.**

#### AND IT IS UNASSIGNABLE, WHICH IS WHY IT REACHED THE OWNER

    `WRITE_ZONES.MD` §1 freezes `src/video.py` EXCEPT TWO TAGGED FUNCTIONS.
    THIS IS NOT ONE OF THEM -- it appears in `WRITE_ZONES.MD` **0** times,
    against a control of 5 for the file itself.
    **IT IS NOT AN UNMADE DISPATCH. IT IS A FALSE COMMENT IN A FROZEN FUNCTION,
    IN NO SEAT'S BOUNDARY.**

*One seat carried it as "UNOWNED" in nearly every tick today, and named the reason it never moved:*
**"I reported a status line accurately for eleven hours and never tested the premise underneath it —
that it was somebody's to take."**

**AND THE GENERAL FORM, WHICH IS WHY THIS IS IN THE GOVERNING DOCUMENT RATHER THAN IN A TICKET:**

> **A COMMENT THAT SAYS *"THIS CANNOT HAPPEN ON THIS CORPUS"* IS A MEASUREMENT WITH NO DATE, NO
> METHOD, AND NO `m`. IT WAS TRUE WHEN WRITTEN AND NOTHING RE-CHECKS IT — AND ITS ONLY EFFECT IS TO
> STOP THE NEXT PERSON MEASURING.**

*Every such sentence in this repository is a claim inheriting §105: a pin, in prose, with no revision
beside it.*


---

## Carriers — where campaign 2's record lives

Pointers only; nothing below is duplicated here.

- `CAMPAIGN.MD` (this directory) — what BINDS: the owner rulings of 2026-09-05 and 2026-09-08,
  the stage-2 gate, and the campaign's end condition.
- `AGENT_NOTES_DEV2.MD` — the developer's own record: the 89 ms retraction, the pHash verdict,
  and the (plan, cause) contract.
- `MEASURING.MD` (this directory) — the complete measurement journal.
- `DOCS_EXTRACTS.MD` — the dated campaign-2 material extracted from `docs/` (grants, incidents,
  measured tables), verbatim, with provenance headers.

## Owner ruling, 2026-09-14 — repair scope: the divergent zone only

Relayed by the Architect, verbatim intent: the campaign is stalling on the
precision of the requirement. **The repair replaces only the divergent zone,
never the whole comparison zone.** A splice that rebuilds the entire compared
span where releases only diverge for a segment is over-repair: it discards
master-quality material the ruling says to keep. Every repair plan and every
target built by hand must state which span is divergent and leave the rest
untouched.

## Salvaged from the retired labs, 2026-09-14 (Architect)

The old labs were emptied before deletion. Durable facts not carried anywhere
else in the tree, condensed; full notes die with the labs, their unique git
history is bundled beside the repository.

- The fingerprint quantum is 124–142 ms in practice, not a constant 125 ms;
  threshold 127 encodes "one frame plus two", not a magic number.
- `lowpass=f=16000` on a 32000 Hz source saturates (~25x) — a Nyquist bug that
  poisons correlation upstream of any repair (dev-3, confirmed).
- A PAL constant rounded 0.200 ms wrong turned a test green that should have
  failed: rate constants must be written at full precision (dev-2).
- Duration ratio alone cannot separate resample from time-stretch; pitch
  separates them perfectly (4.25 % gap) — hence the `no_pitch_relation` verdict
  family (dev-2).
- Delay-to-frame architecture: SHIFT, not ROUND — settled by three independent
  routes (dev-3).
- `max(...) or [0.0]` conflates "measured zero lag" with "never measured" — the
  campaign's `None`-means-could-not-measure rule, violated in one expression
  (dev-2, latent in production).
- Four contract keys (`speed_margin`, `fidelity_margin`, `decided_by`, …) have
  no producer anywhere and always read `absent` (dev-4).
- Chimeric fill on 42 files: median 200 s, max 352 s — the scale of what "only
  the divergent zone" saves is measured, not hypothetical (dev-2).
- Two unmerged pipeline variants (`audio_offset_gccphat.py`,
  `candidate_repair.py`) are staged beside the repository for the Lead Dev's
  ruling — GCC-PHAT offset and post-correlation repair decision with speed
  compensation, both absent from `src/`.

## Recovered 2026-09-14 — the owner's frame-accurate spec was lost, and is back

`OWNER_SPEC_FRAME_REPAIR.MD` (owner, direct, 2026-09-07 — repair the exact zone
TO THE FRAME; fpcalc's per-point vector locates the zone at ~3-4 frames, pHash
decides the frame; index in FRAME NUMBERS, invariant under resample; never route
through cross_correlation) and `ARCH_FRAME_ACCURATE.MD` (its transposition onto
the existing code, naming four defects that defeat frame accuracy: integer-coerced
fps, pHash gated on fps equality, fabricated default frame rate, no VFR gate)
survived only in a retired lab. They are the precision requirement the campaign
has been missing, and they now live here. The 2026-09-14 "divergent zone only"
ruling is this spec's first line restated.

Related instruments beside the repository: `VMSAM_CORPUS/docs/REPAIR_TARGETING.md`
(the id-level denominator: attempt/do-not/cannot-help per file) and
`VMSAM_CORPUS/docs/CORPUS_MISFILING.md` (the content-identity instrument and its
refuted alternatives). Open owner item from the latter: four files (ids 12, 13,
45, 108) are refused by a single outlier window while nine of ten windows agree
to ~2 ms — `mergeVideo.py:582-594`, frozen, awaiting a ruling; note the variance
constant 0.004 is seconds-squared (sigma ~63 ms), not 4 ms.

## Owner rulings, 2026-09-14 — the frame-accurate spec is confirmed and extended

1. `OWNER_SPEC_FRAME_REPAIR.MD` is confirmed **as is**: it is THE precision
   reference of the campaign. Tasks 005-010 are to be reread against it.
2. The corpus ground truth is to carry the expected boundary **as a master video
   frame number** — for the ENTIRE corpus, not only the synthetic cases (real
   pairs get annotated once the pHash instrument can measure them). See TASKS/014.
3. An audio-only divergence is still bounded **in master video frames**: one
   reference clock for everything, consistent with frame-index invariance under
   resample.

## Owner ruling, 2026-09-14 (addendum) — the 50-frame pHash validation

Chromaprint's ~125 ms resolution is not a limitation to overcome but the
targeting stage: it bounds the divergent zone cheaply. pHash then validates the
cut by proving the 50 frames before and the 50 frames after it are identical
between master and candidate. Recorded in OWNER_SPEC_FRAME_REPAIR.MD as a dated
addendum; TASKS/013/014 acceptance criteria inherit it.

## Owner ruling, 2026-09-14 — the outlier-window cases are settled by experiment, not by edit

The stage-2 check stays frozen. Ids 12, 13, 45, 108 (nine windows agreeing to
~2 ms, refused on one outlier) are routed through the finished chimeric path,
which rules: no divergence found on audio AND images means the refusal was
unjustified; a found divergence (owner expects 1-2 frames) means it was right.
The repair path must return "no difference found" as a first-class verdict —
same contract as TASKS/013 §3. Details: TASKS/015.

## Carriers addendum, 2026-09-14 — recovered task results beside the repository

`VMSAM_HELP_AI/salvage-labs-20260904/` holds two recoveries the tree lacked:
the MEASURED RESULTS of forensic tasks 003 and 004 (TASKS/003-* and 004-* here
carry only the briefs), and the dev-3 "shift, do not round" decision series
(T01-T07, three independent routes) behind the delay-application rule already
summarized above. Consult them there; they stay beside the repository.

## Measured 2026-09-15 — the deploy chain works end to end (C2 closed)

The chain push -> CI image -> recreate hook -> verified SHA, which the previous
campaign left untested with a live reason to expect failure, was exercised once
deliberately. The test container's /health moved from git_commit b1954d45 to
e2691f639b5df620235a43030d158fcba2630fca (the pushed tip) within ~30 s of the
hook. Command and workflow: the deployment document beside the repository.
Caveat that stands: fire the hook only ~10 min after a push — early firing
redeploys the previous image and looks like success.

## Owner rulings, 2026-09-15 — end-condition parameters closed, calibration delegated

1. **Denominator**: the content of `/errors` at the instant of the final
   validation — deliberately unfrozen so the owner can add test files;
   reclassified by the REPAIR_TARGETING method at that date.
2. **Validation**: the Forensic agent builds the expected file BY HAND, with
   its own vision and tools, never the pipeline under test; the product is
   compared to that target. Bit-exactness not required.
3. **Delegation**: the 50-frame boundary window (~2 s) is an arbitrary
   placeholder — window length, similarity floor, and the nearest-cut
   alternative are delegated to measurement (TASKS/016).

## Precision, 2026-09-15 (owner concern) — the bit-exactness waiver is narrow

"Bit-exactness is not required" was a negative licence an agent could stretch.
Restated positively in CAMPAIGN.MD §4: byte identity is waived only for
streams a named transform re-encoded; untouched zones must be bit-identical to
their source; structure, boundary pHash, flat correlation, and attributability
of every remaining difference are all required; an unexplained difference of
any size is a FAIL. Skipping a leg because of the waiver is a broken
validation, not a pass.

## Stage 0, 2026-09-15 — TASKS/005-010 reread against the frozen requirement

Two independent passes: the Lead's, and a sub-agent launched by the Architect on
the six tickets **before** the Lead's findings reached it, so it never saw them.
Verdicts: **005, 006, 007 CONTRADICT** the requirement; **008, 009, 010 are
INCOMPLETE** against it. The tickets are dated 2026-09-08 and parts of the
requirement are owner-dated 2026-09-14, so some of this is staleness — but
006's dropped refiner, 007's fill policy and 007's flag removal are **positive
statements that say the opposite of the frozen text**, and staleness does not
cover those.

### S0-1 — the boundary key is specified in milliseconds; the requirement says frame numbers

`TASKS/005` L287-289 fixes the locator-to-splicer contract as
`{master_start_ms, master_end_ms, candidate_offset_ms}`; `TASKS/006` carries the
same convention; `TASKS/007` works to a **100 ms** alignment tolerance
(L205, L399), which is ~2.4 frames at 23.976 fps — **no frame-accuracy claim can
rest on it**. `CAMPAIGN.MD` §2 and `OWNER_SPEC_FRAME_REPAIR.MD` constraint 2
require master video frame numbers, because the frame index is invariant under
resample and a millisecond boundary needs a factor to be right at all.

`TASKS/005` **contradicts itself** on this: L67-69 states the deliverable is *"a
signed displacement in whole reference frames"*, L287-289 delivers the millisecond
dict. A dev cannot satisfy both. Reached by both passes independently; the 007
extension and the self-contradiction are the Architect's sub-agent's.

### S0-2 — the requirement's second instrument was dropped as useless

`TASKS/006` §"What was dropped": *"**The frame-level video refiner.** dev-2's
repair fills the bracket from the master, correct by definition, at 0.49 % of an
episode against 15 %. **An instrument with no consumer.**"* Verified verbatim by
two readers. This is the highest-order item of the pass: the refiner is the only
component that can satisfy *"to the frame"*, and the only one that can run the
pHash boundary validation. It was dropped on a consumer argument, and §2 then made
it mandatory — so the drop must be **reversed, not extended**.

### S0-3 — nothing owns the pHash boundary check

None of the six tickets contains the 50-frames-each-side validation (owner
addendum, 2026-09-14). `TASKS/016` calibrates its numbers; no ticket implements
it. `TASKS/007` L83-84 substitutes audio alignment within 100 ms as its closure
evidence and L115 already declines to claim the boundaries are right. A ticket is
missing between 013 and 016; the Architect opens it.

### S0-4 — the repair is specified to rebuild far more than the divergent zone

`CAMPAIGN.MD` §2: repair **only the divergent zone, never the whole comparison
zone**. `TASKS/007` L8-11 places the candidate's pieces on the master timeline and
**fills the gaps from the master itself**; L126 *"filled from the master, never
spliced"*; L131-133 measures master fill of **21.9 %, 31.6 % and 22.4 %** of the
output on real ids 8, 12, 13. Verified.

Correction to the Lead's first pass, recorded because the first pass was read by
others: the Lead wrote that no ticket proposes replacing the whole comparison
zone. Literally true — 007 replaces no zone wholesale — and it **missed the
finding**, which is that repairing 22-32 % of a timeline is not repairing the
divergent zone either. The Architect's sub-agent caught it. The consequence
reaches `CAMPAIGN.MD` §4 clause 5: master fill is not a *named transform*, so
every filled second is an unexplained difference, and clause 5 makes an
unexplained difference of any size a FAIL.

### S0-5 — the chromaprint point is not 125 ms (owner ruled 2026-09-15)

`BRIEF_COMMON.md` L81 states *"One point = 125 ms … if a delay is not a multiple
of 125, question the reading before you question the media"*, and `TASKS/006`
L15-16 states it as a general rule. **Measured false as a general rule**:
`audioCorrelation.py:132` computes the quantum per call as
`int(lengthFile/len(fingerprint_source)*1000)`, giving **124-142 ms**, and 125
exactly only for a window of 160-286 s (`TASKS/005` §5). `TASKS/008` L66
independently measures a **gcd of 124 ms** on real files. Campaign 1 recorded
*"138 ms observed"*; both figures are the same mechanism seen from different
window lengths, and this entry carries that hedge rather than sharpening it.

**Correction to the strength of this finding, and to its arithmetic.** Two
things were wrong in the first telling, both the Lead's to own. (i) *"Measured
false"* was too strong — the Architect re-verified against 005's table and the
owner's *"125 is the norm at +/- 1"* holds for every window the pipeline passes
at ordinary episode length. (ii) The episode-duration mapping was inherited from
005 §5 and 005 §5 mis-transcribed the code. The real function is `video.py:474-485`
with `number_cut = 10` (`:21`), doubled at `mergeVideo.py:774`:
`L = 2*int((duration - 90)/11)` on the **`> 300 s`** branch — not
`int((duration - 240)/11)` on `> 540 s` as the ticket states. Recomputed, the
125 ms band is an episode of **16.2 to 27.9 minutes** (005 says 18.7 to 30.4),
124 ms from **27.9** minutes up, and 005's own measured `L = 100 -> 127 ms` is a
**~10.7-minute** episode, not a 13.2-minute one.

So *"+/- 1 for every episode-length file"* is over-strong at the short end: **it
holds from about 16 minutes up; a ~10.7-minute episode reads 127**, and between
L=100 and L=160 nothing was measured, so 126 there is interpolation and is **not
claimed**. Short-format episodes are an ordinary case, not a corner one.

**Where the universal quantifier came from, since it matters more than the
number.** The owner's words were *"125 ms est **généralement** la norme à +/- 1
ms"*. He carried the hedge; the Architect dropped it in relaying, and the
unhedged form reached `BRIEF_COMMON.md`. The Architect identified this as their
own violation of rule 4 — *never restate a finding more strongly than its author
did* — committed against the owner himself, and has corrected that file under
the freeze instrument that governs it, citing the real formula by line and naming
the withdrawn version. So the narrower statement here is **not a softening of the
owner's ruling; it is his ruling, restored.**

A third artefact of the same mis-read line, found by the Architect on re-reading
the source: 005 §5's *"a 45-minute episode gives L = 436"*. Under the real formula
a 44-minute file gives `length_time = 231`, so **L = 462**. The conclusion
survives — the table reads 124 from L >= 288 either way — but the figure is
wrong. It sits in `TASKS/005` §5 prose and is corrected there in place; the
`L = 436` in that section's sweep *table* is a legitimate swept value and stands.

**The method finding underneath the number, which outlasts it.** Not "the ticket
was wrong". This: **a measured table and a transcribed formula sat side by side
in the same section of `TASKS/005` §5, carrying the same apparent authority, and
the formula poisoned every consumer for hours — the Architect, the Lead,
`BRIEF_COMMON.md`, `AGENT.MD`, the 006 amendment — while the table beside it
stayed true.** Nobody checked it precisely *because* half the section was
genuinely measured. The rule with its consequence attached: **a number copied
from code is not a measurement, and it inherits none of the authority of the
measurements beside it. Cite code by `file:line`, or re-read it.**

Both seats re-derived the real formula independently from
`src/video.py:475-477` (`number_cut = 10` at `:21`) and `src/mergeVideo.py:774`,
and agreed to the minute on all three band edges: L=160 -> 970 s (16.2 min),
L=288 -> 1674 s (27.9 min), L=100 -> 640 s (10.7 min). `grep` for `-240` or `540`
in either file returns nothing: **005's formula exists nowhere in the code.**

**Owner, 2026-09-15**, asked whether to correct the documents: *"Agent.md have
right! But not commit the brief_common. Save this finding. 125ms is the common
value, but this will not change the algorithm."* So:

- `AGENT.MD` is corrected and now states the quantum as per-pair with the band.
  **It never carried the false claim** — the Lead's first report said it did, and
  that limb was wrong; the correction is an addition, not a repair.
- `BRIEF_COMMON.md` is **left as it stands** and is now frozen mechanically. It
  therefore freezes a sentence measured false, **by the owner's explicit choice
  and not by oversight** — recorded here so no successor reads it as authority.
- **No code change is licensed by this.** The owner bounds the consequence:
  125 ms is the common value and the algorithm does not change. In particular
  this does **not** authorise touching the `127`/`128` gates of S0-6.

### S0-6 — two frozen-file defects, unowned since 2026-09-08

From `TASKS/005` §Blocked, reached by the Lead's pass only: the ten hardcoded
`127`/`128` one-frame gates in `first_delay_test`, which encode "within one frame"
only at a 125 ms quantum; and `get_delays_dict(..., delayUse=0)` at
`mergeVideo.py:489` and `:502`, which record a post-shift residual as if it were
absolute. Both sit outside any tagged zone: findings by `WRITE_ZONES.MD` rule 1,
frozen, and no ticket carries them. **Still findings after this entry.**

### S0-7 — `config.ini [features]` has never existed

Every brief, `CAMPAIGN.MD` §6 and `WRITE_ZONES.MD` §4 send agents to
`config.ini` `[features]`. `TASKS/007` L69 records removing it *"entirely"* citing
§4, while `TASKS/009` L32 and L211 still block waiting on it — two seats on
opposite contracts. Measured by the Lead, with a firing control:

    git ls-files | grep -i 'config\.ini'          -> src/config.ini only
    git log -S'[features]' --oneline -- src/config.ini   -> EMPTY, exit 0
    git log -S'mergerules' --oneline -- src/config.ini   -> c5a73159  (control fires)
    src/config.ini holds [mergerules] and [software], nothing else

So it was never removed and §3's *additions only* was never breached — **the
section was never implemented**, which is a different and more honest finding than
the one 007 recorded against itself. Creating `[features]` is an **addition**,
expressly allowed by §3. Note also that 007 cited **§4** for a `config.ini` rule
and §4 governs source modules, not configuration: the justification did not
support the act even where the act was permissible.

### S0-8 — the stage-3 validation instrument cannot see a frame

`TASKS/010` L68 and L111-113: the video probe's resolution is **500 ms**, and
*"below 25 ms nothing in this campaign can see anything"*. 500 ms is twelve frames
at 23.976 fps. The instrument meant to validate stage 3 is two orders of magnitude
coarser than the quantity the requirement is about, and 010 states no
frame-resolution acceptance criterion. Architect's sub-agent; verified in the text.

### S0-9 — `no_pitch_relation` is in no ticket's verdict vocabulary

A standing ruling absent from the verdict sets of `TASKS/008` and `TASKS/009`
(L195). Architect's sub-agent. Unresolved.

### What held clean under both passes

No ticket routes through `cross_correlation` — zero mentions across all six.
`TASKS/008` L390-398 independently derives **resample -> locate -> splice** and
honestly flags that it has never run end to end, because the resampled candidate
is built inside the assembly and never exposed as a file. `WRITE_ZONES.MD` §5
check 4 is already reconciled with 005's compileall result.

### Open against this entry

`TASKS/005` §6 and `TASKS/006` are **not rewritten**. The owner, asked, answered
*"Ask to the architect. I will reply if needed."* — the ms-to-frame contract
question is delegated to the Architect, who is drafting corrected text for 005,
006, 008 and 009 in the lab, unlanded.

## Governance bypass, 2026-09-15 — one commit, owner-authorised, spent on use

`WRITE_ZONES.MD` §1 permits the Architect or the Lead Dev to edit governance
documents **with the owner's authorisation**. §5 check 1b requires the staged
governance diff to be **empty, unconditionally**, and `check_write_zones.py`
faithfully implements 1b: `GOVERNANCE` is a flat tuple, `classify()` prefix-matches
it, and the violation is appended with no authorisation path of any kind. **The
document contradicts itself**, and no change to the checker could fix it —
loosening the instrument would breach §5, which names the instrument the authority.

Measured before raising it (Lead), against landed history:

    --rev 523ec97e -> exit 1   --rev 7068b07e -> exit 1   --rev 47817068 -> exit 1
    --rev c6f73d12 -> exit 1   --rev 1a6a290b -> exit 1

**Correction to the Lead's first telling of this, recorded because it was
reported before it was refuted.** The Lead wrote "five of five, the whole
restart". The Architect refuted the scope: `check_write_zones.py` is **not under
version control** (`git log --all -- '*check_write_zones*'` is empty), its only
date is an mtime of 2026-09-14 23:27, so `1a6a290b` (14 15:04) predates the
instrument's last write by eight hours and its content at that moment is recorded
nowhere. The defensible claim is **four commits, from the last hour of 09-14
onward**. And the reason the fifth is unmeasurable is itself a finding: **the
mechanical authority on write zones has no auditable history of its own.**

### The owner's answer — a fourth shape, not one of the three options

Asked to choose between teaching the checker an authorisation path, letting
governance ride with code, or committing the files himself, the owner chose none.
Verbatim, to the Architect: *"il faut juste faire attention au autorisation.
bypass juste cette fois ci le checker. il faut faire en sorte que ces
modifications exceptionnelle ne viennent pas perturber la suite du travail."* And
independently, in the same minutes, to the Lead: *"Only this time you can modify
the governance files. Always ask me next time."*

Two seats put the question separately and received the same ruling in two
languages. **The checker does not change. The owner is the authorisation path,
manually, per occasion.** This resolves §1-vs-§5 without editing `WRITE_ZONES.MD`:
§1's "with the owner's authorisation" has a concrete form — an explicit owner
order per batch — and §5 check 1b remains the mechanical default that makes every
*unauthorised* governance diff fail.

**This bypass is spent on use. Every future governance edit needs a fresh,
explicit owner authorisation. No seat may cite this entry as precedent** — citing
it is the failure it was designed to prevent.

### What was authorised, exactly

    AGENT.MD
    CAMPAIGNS/02-dev-chimeric-resample/FINDINGS.MD
    CAMPAIGNS/02-dev-chimeric-resample/RULING_20260915_FRAME_INDEXED_BOUNDARY.MD
    CAMPAIGNS/02-dev-chimeric-resample/TASKS/005-dev1-frame-compare-ruling.MD
    CAMPAIGNS/02-dev-chimeric-resample/TASKS/006-dev1-locate-change-points.MD
    CAMPAIGNS/02-dev-chimeric-resample/TASKS/007-dev2-chimeric-splice.MD

`TASKS/DOCKET.md` rides at the repository root, which is open zone and needs no
authorisation. **Not covered, and not touched:** `WRITE_ZONES.MD`, `docs/`,
`TASKS/008`/`009`/`010` (gaps, not contradictions — the landed ruling corrects
them at the contract level), anything under `src/`, and `BRIEF_COMMON.md`.

The procedure followed: stage exactly those paths; run `--staged`; **expect exit
1 and require every violation line to be a §1-governance line naming a path on
the list** — one violation of any other kind aborts; re-run on the landed rev
afterwards and confirm the same governance-only set. The captured output of both
runs is recorded in the task file rather than the commit message (privacy: commit
messages stay minimal and carry no links or ids).

**The exceptional and the normal did not share a commit.** `vmsam-dev-frame`'s
work on `src/frame_compare.py` and `src/merge_video_chimeric.py` was already in
the working tree when this batch was staged, and was deliberately left out of it:
it lands separately, through the normal gate, at exit 0, or it does not land.
That separation is what "ne pas perturber la suite du travail" compiles to.

### What this batch carries

The stage-0 entry above (S0-1 to S0-9); the `AGENT.MD` quantum correction; the F1
frame-indexed boundary contract as a campaign ruling; and amendments to 005, 006
and 007 pointing at it — 006's dropped refiner reinstated, 007's whole-track
re-encode and 100 ms tolerance superseded, its master fill **retained** as a
legitimate repair of a divergent zone.

### Owner policy recorded in passing, 2026-09-15 — flags are a last resort

*"je pense qu'il faut le moins de flag possible. vous pouvez rajouter qu'en
dernier recours."* **As few flags as possible; a flag is a last resort, not a
default pattern.** This settles the 007-vs-009 contradiction in a direction
neither ticket took: 007's no-flag instinct was the closer of the two. In review,
a seat proposing a flag must first show why the behaviour cannot simply be correct
unconditionally. `config.ini [features]` may be created when that case is made —
it has never existed (S0-7) — and creating it is an addition, allowed by §3.

## Governance gesture #2, 2026-09-15 — owner: *"j'autorise"*

Second and last authorised governance batch of the day. **Spent on use**, like
the first: a third batch needs a fresh, explicit authorisation.

Contents: the two `AGENT.MD` corrections below, the `WRITE_ZONES.MD` §1 revocation,
`RULING_20260915_MEASUREMENT_RETENTION.MD`, `RULING_20260915_DECISION_ROUTING.MD`,
the F1 ruling's blocked need closed by measurement, `TASKS/017`, and these entries.

**`TASKS/017` is the reason this batch could not wait.** Its final 106 lines
existed **only on disk** from the moment root `TASKS/` was closed until now — two
copies beside the repository, nothing in git. That is the largest loss risk the
campaign carried today, and it is closed by this commit.

## Owner rulings recorded verbatim, 2026-09-15

**Privacy, stated at its cleanest** — *"Les log.error sont bien tels quels. Ils
nous aident et restent internes. La privacy s'applique qu'au repo."*
**The repository is the privacy surface** — tracked files, commit messages,
branch names. **Internal artefacts carry full paths and names by design, because
that is what makes them useful.** The surviving nuance: `VMSAM_TEST_OUTPUT_DIR`'s
tree mirrors library paths, so anything that **publishes or commits from** it
still crosses the boundary. This killed a parked log-sanitizer proposal outright.

**An intact track always wins** — *"une piste intacte de même langue gagne
toujours contre une piste réparée."* Unconditional, **no quality override.**
With `SPEC_ZONE_A` §6 this closes the `fabricated` question entirely. **No code
changes: it is what the code already does.**

**Language policy is intentional** — *"Faut qu'il comprenne que c'est bien
intentionnel ! Et que c'est totalement voulu. Qu'il fasse avec les
paramétrages."* Seats work **with** the configuration, never around it.

**Flags are a last resort** — *"je pense qu'il faut le moins de flag possible.
vous pouvez rajouter qu'en dernier recours."*

**Repair only the divergent zone, to the frame** — *"Je veux que le script
prenne du master QUE la partie problématique, À LA FRAME PRÈS"* — and its
sharpening: ***"les 140 secondes ne sont PAS la cible de la réparation — elles
indiquent OÙ CHERCHER."*** **Indication is not localisation.**

**The two repair verbs** — *"On utilise le master pour les rajouts. Si c'est 3
secondes EN PLUS, on rajoute. Si c'est 3 secondes EN TROP, on coupe."* A
conforming plan contains **passthrough, ADD-from-master, and CUT. Nothing else.**

**`AGENT.MD` is read IN FULL** — no `head`, no truncation. The onboarding read of
a binding document is integral; *"grep, don't load"* governs reference lookups
only. **You cannot be bound by rules you have not read.**

## The same wrong rule, invented twice in one day, in opposite directions

**The Lead** read `AGENT.MD` to line 200 only, invented *"no host paths in
tracked files"*, and made a seat redact **five lines the document expressly
permits** (`AGENT.MD`:379 — *"container and tooling paths are fine"*).
**The Architect**, independently and hours later, drafted the **same invented
rule** into the `AGENT.MD` amendment — where it would have become the document's
own text, and then been **uncatchable by reading**, because the two sections
would have agreed with each other.

**It was caught only because the full-read order forced both of us over the
sections we had skipped.** The rule that survives is the real one: **container
and tooling paths are fine; media identity in a tracked file is not.**

## Two self-indictments, recorded beside the cases that produced them

**The Lead: a gate nobody revisits is not a gate, it is a stop.** The wide sweep
was held on a condition that batch 1 had already satisfied, and never
re-examined while attention was elsewhere. **Both containers idled 74 minutes
against a standing owner instruction.** Compounding it: the sweep is what
**generates** the material the open questions need, so holding it was circular.

**The Lead again, on parking:** the pairing patch sat most of a day because
*"the cross-language risk call"* was routed upward **with no named decider, no
stated default, and no date.** The decision had nowhere to go. That is the case
the anti-parking rule in `RULING_20260915_DECISION_ROUTING.MD` exists for.

**The Architect: a mechanism read before its ruling nearly became a defect
report.** Having traced the `fabricated` branch, it drafted the question to the
owner as an (a)/(b) choice **with a lean toward "fix it"** — when
`SPEC_ZONE_A` §6 already answered it in the owner's own words, including *"do
not 'fix' this."* It retracted before the owner acted. **The discipline is the
full-read order pointed at `CAMPAIGNS/`: a mechanism finding checks the rulings
before it grows a verdict.**

## Method findings paid for today

**A measurement of an intermediate is not a measurement of the product.** Two
Lead escalations died on this: 43 % master fill that **never shipped** (the
chimeric track lost to ranking), and four subtitle languages *"silently lost"*
that were **configured policy**. Both corrections came from seats that went and
looked at the **delivered file** instead of trusting the log.

**A guard that silently matches nothing looks exactly like a guard that passes.**
`send_to_agent.sh` accumulated **five defects of one family** — checking a proxy
instead of the thing: text *arrived* for text *submitted*; a `\xNN` escape that
matched **zero** lines; the message's **first** 24 characters, which scrolled out
of view when the opener grew; five Enters where a fresh TUI needs more; and
`[[:space:]]` for "empty" when the empty prompt is `❯` + a **non-breaking
space**. Every one was caught by firing the check in **both** directions against
real state.

**Salvage the method, never the material.** Applied twice: `vmsam-dev-frame`'s
hand-built targets and firing tests reached `VMSAM_CORPUS/`; `vmsam-bench-sync`'s
three independent instruments reached `tools/` and its err_12 derivation reached
`VMSAM_CORPUS/docs/`. The media, extracts and scratch went with the seats.

## Gesture #3 — 2026-09-15 evening + 2026-09-16, synthesized under the owner's carte-blanche authorization of 2026-09-16

This gesture lands the rules held in trust since 2026-09-15 evening
(`VMSAM_HELP_AI/lead/RULES_EARNED_20260915_EVENING.md`, Lead seat #2, measured
~18:16–20:00Z that evening, checkpointed ~21:00Z), plus the dated findings
measured across 2026-09-15 evening and 2026-09-16 by the Lead, the Architect
and the seats under them. Evidence pointers are given as `file:line` where a
specific line was read; a bare filename means the citation is to the document
as a whole. Per this campaign's own rule, nothing below is restated more
strongly than its author stated it.

---

### PART 1 — RULES HELD IN TRUST (46 rules, `lead/RULES_EARNED_20260915_EVENING.md`, Lead seat #2, 2026-09-15 evening)

Grouped by the Architect into seven themes for landing. Each rule is carried
**verbatim, bold kept**, with the one-line evidence context the source file
gave it. Per the source file's own instruction: *"When FINDINGS carries them,
delete this file — a rule in two places drifts."* `RULES_EARNED_20260915_EVENING.md`
is deleted in this same gesture (see MANIFEST).

#### 1a. Constitutional

- **THE OWNER'S ORDER IS THE AUTHORITY. The checker and the frozen text are
  both RECORDS of it, and either can lag. When they disagree, the question is
  never *which artefact wins* — it is WHICH ONE MATCHES THE OWNER'S MOST
  RECENT ORDER.** (`RULES_EARNED_20260915_EVENING.md:19-21`) Evidence: one
  evening produced both lag directions on the same two owner-closed modules —
  at ~18:00 the text matched the order and the checker lagged; at ~21:00 the
  checker matched the order and the text lagged. **"The checker is the
  authority" resolves both the same way and is therefore wrong about the
  first.**
- **Practical form for any brief: never write "X is superseded because the
  checker is the authority." Write the order (quoted), that the checker has
  been updated and verified, and that the text is stale-not-binding.**
  (`:43-46`) Evidence: `WRITE_ZONES.MD` §5's "it is the authority" is scoped
  to *which procedure to run*, not *which record wins* — a neighbouring
  question that looks like the one being asked.
- **THE LAYER RULE (Architect, ratified 2026-09-15): the EXTRACTOR never
  compensates; the PLANNER decides. A projection overrun is not a
  demonstration — it is an unread defect... a full-length file is only ever
  produced by a plan that SAYS it filled a tail and why.** (`:220-227`)
  Evidence: fill legitimacy requires a demonstrated divergence, named
  decision, frame-bounded, pHash-checked boundary — never an extractor
  compensating silently.
- **Authorization relayed THROUGH a peer licenses non-actions and
  independently evidenced actions; anything irreversible resting on the
  authorization ALONE waits for the owner's own words.** (`:248-250`)

#### 1b. Guards

- **A GUARD THAT SPEAKS BOTH WAYS MUST BE TRIAGED ON ITS VERDICT TOKEN, NEVER
  ON THE LINE'S PRESENCE.** (`:52-58`) Evidence: the extraction-bound audit
  line fires on the passing path too; CI counted a clean PASS as one of the
  two expected refusals.
- **THE ABSENCE OF A GUARD'S LOG LINE IS A DIFFERENT FACT FROM A PASS.**
  (`:155-158`) Evidence: the extraction guard emits its audit line for
  passing tracks too, so `grep` returning nothing means the guard never
  ran — not that everything passed.
- **A scanner whose hits are never looked at is worse than no scanner: it
  manufactures the feeling of having checked.** (`:198-200`) Evidence: the
  Lead's privacy regex alarmed four times on `erai` inside the French word
  *serait*.
- **"0 false refusals in 59 was true and not good enough."** (dev-extent.)
  **A false refusal inside a guard added to remove a false pass** is the
  trade to refuse. (`:212-218`) Evidence: 16 of 59 tracks declare a
  `Duration` shorter than their real packet extent (worst −120 ms); one
  surviving plan sits at exactly 0.0 ms headroom.
- **"A correct bound does not move the plan — it REFUSES it."** (dev-extent.)
  (`:228-229`)
- **Three ways a guard lies** — all three measured the same day: the
  UNFIRED guard; the HALF guard (checks one side of a two-sided operation,
  reports silence on the side it never looks at); and the WRONG-INPUT
  guard — *a guard written in full awareness of the hazard, reaching for the
  wrong number, is worse than no guard, because the comment persuades the
  next reader the case is handled.* (`:254-258`)
- **AN INSTRUMENT FAITHFULLY RECORDING A STATE THAT CANNOT CHANGE IS THE
  UNFIRED-GUARD PROBLEM WEARING A TIMESTAMP.** (`:275-279`) Evidence: a
  60-second watcher logged `E04=no E06=no` for an hour; "the queue is deep"
  and "these two are not in the queue at all" produce identical evidence
  from the watcher's position.
- **A PANE'S `❯` LINE IS AN INPUT BUFFER — what was typed AT a seat, never
  what the seat did.** Corollary: **a `❯` line neither the Lead nor the
  Architect authored is probably the OWNER typing directly into the sessions
  he watches** — telling a seat to disregard it puts the Lead's convenience
  above the owner's direction. (`:323-331`) Evidence: four instances the same
  evening of reading an input buffer as a state, including an Architect
  relaying queued instructions to the owner as completed decisions.
- **AND ONE THING THAT DOES NOT SCALE WITH DELEGATION: the gate.** **"Verified
  by a sub-agent" never means verified on this seat.** (`:352-357`) Evidence:
  four checker changes on 2026-09-15, four Lead-verified exit statuses; the
  fourth caught a scope bug the self-test alone had passed.
- **MUTATION-TEST A GUARD OR YOU HAVE NOT TESTED IT.** (`:380-383`) Evidence:
  dev-cause's sentinel checker proves it can fail — exit 0 on the real file,
  exit 1 with the two parentheses deleted.
- **THE EXIT CODE TRAP, COMMITTED BY THE SEAT WHOSE BRIEF OPENS WITH IT.**
  **"Never use `cmd | filter` then `$?`. A piped command hides the exit code
  of the first command."** **Knowing a rule, having authored guidance about
  it, and having enforced it on others all evening are not the same as
  following it.** (`:405-423`) Evidence: `git push -q 2>&1 | grep -v
  "^remote:" ; echo "push EXIT=$?"` read grep's status (1) as a failed push;
  the push had succeeded. `lead/BRIEF.md` IRON LAW #1 is this defect by name.
  See PART 2, "the pipe-trap" entry, for the second half of this family.
- **TEST A FIX AGAINST THE FAILURE CLASS, NOT THE FAILURE INSTANCE — "the
  same failure mode wearing the fix."** (`:425-433`) Evidence: `compileall`
  exits 0 on a missing directory, so §5's check 4 was a green gate over zero
  files from a wrong cwd; the proposed `test -d "$(git rev-parse
  --show-toplevel)/src"` closed the wrong-cwd instance and left the class
  open (any repo carrying a `src/` passes, green, on a stranger's tree). The
  fix pins IDENTITY over SHAPE: `test -f "$R/WRITE_ZONES.MD" -a -d
  "$R/src"` — and a git WORKTREE of VMSAM passes, and should. Lands in
  `WRITE_ZONES.MD` §5 in this same gesture (see PART 4).
- **A DOCUMENTED COMMAND IS A COMMAND SOMEONE WILL RUN.** **"A disclaimer
  people read past is not a control." Every snippet left in a governance
  document must FAIL CLOSED as written, or become prose.** (`:434-438`)
  Evidence: §5's snippets carry the disclaimer "nobody runs them" — a seat
  ran them, and so did the Lead, repeatedly. Lands in `WRITE_ZONES.MD` §5 in
  this same gesture (see PART 4).

#### 1c. Transport and silence

- **WHEN THE BRIDGE IS DOWN, DROP A FILE IN THE SEAT'S LAB — and the
  successor must be able to tell it from an impersonation.** **A note
  dropped into a lab must name its author, its seat and the time in its own
  text.** (`:68-81`) Evidence: an unsigned note appeared in inv-fill's lab at
  18:10:40; Lead seat #2 spawned six minutes later and could not immediately
  tell who had written it.
- **THE TRANSPORT FAILURE IS OLDER THAN IT LOOKED. A known-broken bridge must
  be handed over explicitly; it was not in the handover.** (`:82-85`)
- **BEFORE YOU READ A SILENCE AS A RESULT, PROVE THE CHANNEL CAN SPEAK.**
  (`:87-98`) Evidence: dev-extent found the test instance runs `dev: false`
  before trusting an absence-based validation clause; verified the ungated
  channel could carry an artefact and examined the one control that could
  have refuted it.
- **ENVIRONMENT FACT with consequences past one ticket: the test instance
  runs `dev: false`.** (`:99-104`) Every `if tools.dev` diagnostic is
  invisible there.
- **A SEAT'S SILENCE IS NOT EVIDENCE THAT IT IS WORKING — the transport can
  fail silently and the Lead cannot tell the difference.** **PROCEDURE: never
  infer a seat's state from the absence of messages.** **AND THIS IS WHY THE
  FILES RULE PAYS: "a seat's numbers exist when they exist as FILES."**
  (`:123-137`) Evidence: `vmsam-inv-fill` finished its partition, the
  remote-control bridge failed, and its report sat unread for ~90 minutes
  while `ListAgents` showed it "offline" though its tmux session was alive.
- **A seat's numbers exist when they exist as FILES.** (`:235-238`) Evidence:
  measured twice the same evening — one seat kept 11 harnesses and zero saved
  results; another's crown-jewel instrument sat in `/tmp`, which is RAM here
  and dies on restart.
- **NOTHING AN INSTRUMENT DEPENDS ON MAY LIVE IN `/tmp`** — not its fixture,
  not its baseline, not its raw output. (`:186-194`) Evidence: nearly cost
  three artefacts in one evening (the log-mel NCC tool, derived pairing
  positions, a test-suite baseline); fix: extract the baseline from git, and
  an unresolvable revision exits 2/SKIP/"the instrument did not run" — never
  a pass, never a fail.

#### 1d. Records and provenance

- **A RECORD THAT IS CORRECT GOING FORWARD AND WRONG BEHIND IS WORSE THAN ONE
  THAT IS UNIFORMLY WRONG.** **Annotate the boundary; never retro-edit an
  audit trail.** (`:106-114`) Evidence: CI's RUNLOG carried CEST values
  labelled UTC; fixing the format going forward without marking the
  changeover would make the log silently mixed-unit.
- **"NEVER RESTATE A FINDING MORE STRONGLY THAN ITS AUTHOR DID" APPLIES TO
  YOUR OWN FINDINGS.** (`:139-146`) Evidence: dev-extent caught the Lead
  leaking a prediction into an inbox, then went back and weakened its own
  record accordingly, dropping the bigger claim that existed only in
  messages.
- **A PREDICTION THAT LIVES IN A MESSAGE CANNOT BE CHECKED BY WHOEVER SEES
  THE RUN.** (`:148-154`) Evidence: the Lead promised an expectation would
  travel "verbatim" into a documented open item while it existed only in an
  inbox; caught by the seat.
- **A DEAD HYPOTHESIS NEEDS AN ARTEFACT MORE THAN A LIVE ONE DOES.**
  (`:172-178`) Evidence: two refutations existed only inside messages to the
  Lead — negative results are the likeliest to live only in prose.
- **A sub-agent's return is a REPORT to assess, not an analysis to relay.**
  (`:244-247`) Provenance before relay; what cannot be sourced travels as
  "reported, unverified" or not at all.
- **The era key of a row is the DEPLOYED COMMIT SHA.** (`:251-253`) Evidence:
  per-module `build=` stamps hash one file — a week-old locator stamp beside
  a fresh chimeric stamp is consistent inside one clean deploy, which is
  precisely why it deceives.
- **PROVENANCE MUST SURVIVE EVERY HOP — AND WITH BOTH PERMANENT SEATS
  DELEGATING, THERE ARE NOW MORE HOPS.** **THE RULE: a claim that has passed
  through two or more synthesis layers without a principal reading its
  source travels as "reported, unverified," however confident it sounds.**
  **THE CHEAP FIX: name the source `file:line`.** (`:333-351`) Evidence: the
  "coincidentally clean quantum" travelled unreviewed through the Lead to the
  Architect and nearly reached FINDINGS; the id-196 cross-reference came from
  the Lead's own sub-agent and made a seat disprove a claim the Lead had
  never checked.
- **A PATH THAT HAS NEVER FIRED IS THE ONE PLACE YOU CANNOT RECOVER THE DATA
  LATER.** (`:368-373`) Evidence: `record()` site 2225 passes `str(error)`
  only, so the exception class name reaches no artefact; the site has zero
  production lines in 59 artefacts.

#### 1e. Instruments and measurement

- **A PREDICTION INSTRUMENT AND A VALIDATION INSTRUMENT ARE DIFFERENT THINGS,
  and "re-run the tool" does not distinguish them.** **Label every instrument
  by the question it can answer, and name the revision under test BEFORE
  reading any artefact as a verdict on it.** (`:160-170`) Evidence:
  `tools/blast_radius.py` predicts what a guard would do from plans already
  in `/config/output`; on a growing population it returns a fresh, correct
  and completely irrelevant number for a build that never deployed.
- **A RE-RUN IS NOT THE ORIGINAL, and on a LIVE population that matters...
  if a later run disagrees with the record, the re-run is the NEWER FACT —
  go and look at what changed in the POPULATION before concluding anything
  about the CODE.** (`:179-184`) Evidence: `/config/output` grew 58→59 pairs
  mid-session.
- **"The instrument did not run" binds the REVIEWER too.** (`:195-197`)
  Evidence: the Lead ran a seat's tests without `src` on the path, got 4 of 5
  `ModuleNotFoundError`, and nearly reported a bad invocation as a defect.
- **A count whose author did not look behind it is a claim wearing a
  number's clothes.** (`:201`)
- **A SAMPLE SELECTED BY RELEVANCE TO YOUR TICKET IS THE SAMPLE LEAST LIKELY
  TO ANSWER YOUR QUESTION.** **Two artefacts agreeing is not a population.**
  (`:265-274`) Evidence: dev-extent's first check of "does the sweep re-run
  existing entries" looked only at the two artefacts carrying its own audit
  line, both new paths; checking all six found the rewrite sitting among the
  four it had no reason to open.
- **AN INSTRUMENT THAT STOPS LOOKING MUST SAY SO IN ITS OWN TRAIL.**
  (`:281-288`) Evidence: dev-extent's watcher carried a 3300 s deadline
  expiring 69 minutes before the wait it was watching could close; fixed by
  writing the stop and its reason into the trail.
- **A COUNTER OF FAILURES IS NOT A COUNTER OF WORK — AND IT INVERTS.**
  (`:297-307`) Evidence: `.log.error` population counts failing jobs only,
  so it is a floor on completion divided by an unmeasurable failure
  fraction; measured 6 artefacts in 11 min then zero across 7 consecutive
  minutes while 5 `.mkv` files were produced in the same window.
- **AND IT STILL COULD NOT ANSWER THE QUESTION ASKED.** **The ordering of a
  queue is only visible by asking what the queue does with an insertion —
  never from watching what comes out of it.** (`:308-312`)
- **THE SCAR IS ON THE WRONG LIMB — a fix added to expose a divergence can
  guard a different process than the one that was wounded.** **Consequence
  that survives regardless: `/health` proves the FLAG, never that a line
  reached a READER.** (`:385-403`) Evidence: `/health`'s `dev` field is exact
  for the fusion worker (same expression as the assignment) and can diverge
  from the CLI-launched daemon, the process the original incident bit.
- **A MEASURED COUNT BEATS AN INHERITED ONE, AND THE SAFE DIRECTION IS
  FINER.** (`:374-379`) Evidence: the Lead specified 9 distinct decisions at
  `get_speed_ratio` (inherited); dev-cause measured 11 — erring fine is free,
  erring coarse is invisible.

#### 1f. Verdict vocabulary

- **A NUDGED EXECUTION IS NOT THE MECHANISM WORKING.** (`:115-121`) Evidence:
  CI's auto-deploy rule had never fired unprompted in 80 cycles; the
  2026-09-15 fire came after the Lead asked why it had not.
- **A verdict name that cannot distinguish a correct refusal from a failure
  inflates every failure count the campaign produces.** (`:262-263`)
- **The prediction corollary: the reason to state a prediction is not to be
  right — a wrong one gets killed in one measurement instead of surviving as
  a plausible story.** (`:259-261`) Evidence: three predictions preceded the
  Lazarus trace; all three were wrong.
- **An instrument's output is not a media fact — and the error recurs in the
  VOCABULARY used to describe its own fix.** (`:230-233`) Evidence: the
  saturation rail was ruled a no-measurement marker and then called "the PAL
  detector" within the hour, by the same seat.
- **A SENTINEL THAT MEANS "DEFECT" MUST NEVER BE SPENT ON "KNOWN GAP" — the
  second use destroys the first's meaning.** (`:359-367`) Evidence:
  `merge_video_repair.py`'s `(unstated)` sentinel means producer-contract
  violation; 22 authorized-scope raise sites left untokened would have
  stamped 22 authorized decisions as bugs if reused. dev-cause proposed
  `(untokened_raise_site)` instead: **uncounted, not falsely counted.**
- **ONE IDEA AT THREE LAYERS IN A SINGLE ASSIGNMENT.** **The value of a third
  state is that it makes "I did not run" unspellable as "I passed."**
  (`:446-451`) Evidence: seven missing `cause=` stamps were refusals that
  could not spell which refusal they were; `ran_conclusive_negative` vs
  `could_not_run` is the same defect one layer down; a gate that cannot spell
  that it did not run, found on the last pass.

#### 1g. Process ops

- **ABSOLUTE PATHS ONLY FOR ANYTHING THAT DELETES.** (`:203-211`) Evidence:
  dev-extent ran `rm -rf src` from a shell a previous `cd` had moved into the
  repository, deleting the common repo's entire `src/` tree; fully
  recovered, independently verified, left no trace — reported anyway.
- **Salvage-first, verify, then delete.** (`:239-240`) Backup before salvage;
  deletion is the one action in the Lead's mandate with no undo.
- **The successor-gate rule.** **"A gate I did not personally read the exit
  status of is a gate I cannot vouch for."** (`:241-243`) Every new Lead
  re-reads the last landed gesture's gate with their own eyes.
- **STATE A BOUNDARY AS A NUMBER, BECAUSE A NUMBER IS CHECKABLE BY PEOPLE YOU
  ARE NOT TALKING TO ABOUT IT.** (`:289-295`) Evidence: the Lead computed
  "~110 minutes worst case" only to bound a hold; the seat checked its own
  watcher's deadline against that figure and found a 69-minute gap the Lead
  did not know existed.
- **A HOLD MUST BE RE-DECIDED WHEN THE THING THAT JUSTIFIED IT DISSOLVES.**
  (`:314-322`) Evidence: the Lead held a seat from purge for context it
  uniquely held; the seat wrote that context down and the hold was never
  re-examined against its own dissolved justification.
- **THE COMPOSITION RULE'S REAL JOB, three-for-three: "not catching wrong
  claims, but finding the EDGE of right ones."** (`:439-445`) Evidence:
  §5's "it is the authority" scoped to procedure-vs-record; `/health`'s `dev`
  field exact for the worker, divergent for the daemon; the compile guard
  closing two cases of three — every claim made in good faith, the value was
  the boundary, not the correction.

---

### PART 2 — DATED FINDINGS ENTRIES (2026-09-15 evening – 2026-09-16)

#### 2.1 The floor check declines on a statistic while a better one sits unused beside it

Carried whole, at the weight its own author gave it. Full text:
`VMSAM_HELP_AI/architect/drafts/FINDING_median_ignores_consistency_20260916.md`
(Architect, 2026-09-16). Summary, hedges preserved:

`median_fidelity_below_floor` (change_point_locator, decline branch) reads
only the fidelity distribution and never consults the offset-consistency
evidence the same probe scan already built and printed on the same line. A
pair whose offsets are tightly self-consistent can be declined on median
fidelity alone even though the agreement proving the alignment is real sits
two fields away. Measured pair: `errid:307` — 35 of 37 probes span 131.8 ms
(one quantum) at ~6.03–6.10 s, fidelity_max 0.9527, median 0.6039, declined;
two independent instruments (production locator, forensic's log-mel NCC)
agree on the offset and on which two probes are outliers. Control:
`errid:250` — scattered throughout, fidelity_max 0.6266 (never clears the
floor), decline faultless. A second, related defect at the same site
(`change_point_locator.py:1699-1752`): the block claims `ran_conclusive_negative`
— the strongest token the measurement vocabulary has — in the same call that
prints `distinguishes_unrelated_from_unmeasurable="no"`, i.e. the code claims
it concluded while the field admits it cannot tell "unrelated" from
"could not measure." True for the genuinely-scattered case (250), false for
the tight-plateau case (307). Fix has two independent halves (consult
offset consistency before declining; stop asserting `ran_conclusive_negative`
where the field says `=no`, using the existing `could_not_run` token — no new
vocabulary). Explicitly NOT established by this finding: whether stage 1
(first_delay_test / chromaprint) is separately saturated on this pair (a real
open question about a different stage, not the cause of 307's refusal); how
many other refusal-family ids carry the same locator/stage-1 disagreement (not
surveyed, S2 sampling stopped at three exemplars). Collision noted and
respected: the decline site is also the Architect's reserved speed-family
refusal block; the median fix may restructure the fidelity decision and
correct the measurement token, but must not change the monotone→
`speed_relation_suspected` routing and must leave room for the PAL saturation
screen to be inserted before the fidelity check later.

#### 2.2 The governance gate deadlock, and its one-time resolution

`WRITE_ZONES.MD` §1 permits governance edits "with the owner's authorisation";
§5 check 1b requires the staged governance diff to be empty, unconditionally,
with no authorisation path of any kind — `check_write_zones.py` implements
1b faithfully. **The document contradicts itself, and no change to the
checker could fix it**, because loosening the checker would itself breach
§5 (`FINDING_GOVERNANCE_GATE_DEADLOCK.MD` §§1-4, Architect, measured against
`tools/check_write_zones.py:45-51,179-181,309-310`). The datation the Lead
first reported ("five of five, the whole restart") was one too many:
`check_write_zones.py` is not tracked by git, so only four commits from the
last hour of 09-14 onward demonstrably crossed the gate; the fifth (`1a6a290b`,
09-14 15:04) is unmeasurable, which is itself a finding — the mechanical
authority on write zones has no auditable history of its own
(`FINDING_GOVERNANCE_GATE_DEADLOCK.MD` §2).

**Owner's resolution, 2026-09-15, a fourth shape not among the three
options offered:** *"il faut juste faire attention au autorisation. bypass
juste cette fois ci le checker..."* / to the Lead, independently, in the
same minutes: *"Only this time you can modify the governance files. Always
ask me next time."* (`CAMPAIGNS/02-dev-chimeric-resample/FINDINGS.MD:1444-1463`).
**The checker does not change. The owner is the authorisation path, manually,
per occasion.** The bypass is spent on use; no seat may cite it as precedent.
A second, last, bypass was granted and spent the same day ("Governance
gesture #2... owner: *j'autorise*", `FINDINGS.MD:1510-1517`). This present
gesture (#3) proceeds under a fresh, explicit carte-blanche authorization of
2026-09-16, per the owner's own doctrine that every governance edit needs its
own authorization and no seat may treat a prior bypass as standing.

**§5's check-1b text is corrected in this same gesture** — not by adding an
authorisation path to the checker (rejected: any file-based token can be
minted by any agent in this environment and is not access control; the
Architect's option (A) recommendation was never taken up), but by adding the
fail-closed identity guard and framing sentence the Lead's own testing
already earned (PART 1b, "TEST A FIX AGAINST THE FAILURE CLASS" and "A
DOCUMENTED COMMAND IS A COMMAND SOMEONE WILL RUN"). See PART 4 for the exact
edit; it does not touch §1 or check 1b's substance, only the snippet
apparatus around it.

#### 2.3 Documentation that did not travel with the revert (two instances, one day)

**Instance #1 — the void `last_failures` deploy detector.** `a3ee9dbf`
reverted `891243ce` (the `last_failures` ring buffer); the docket's own §6
description of GET /fusion carrying `last_failures`/`last_failures_since`
was left standing after the revert, uncorrected (`lead/DOCKET.md:20-23,1281-1290`).

**Instance #2 — the `fusion.py` swallow's stale "resolved via scoped
exception."** A docket entry recording the swallow as "RESOLVED via an
owner-authorized scoped exception (`891243ce`)... the write-zone checker
fails on this file ON PURPOSE as the audit trail of that exception" was
**false at the tip, both halves**, once `a3ee9dbf` reverted `891243ce`:
`git grep last_failures HEAD -- src/gestionar_show/fusion.py` returns empty
(exit 1, absent); no `fusion` special case exists in `check_write_zones.py`
at all. The authorization itself never existed — the owner's *"developpe ce
point stp. en suivant le process"* equally reads as "elaborate on this
point," and the owner confirmed that second reading. **The authorization
never existed, the code is gone, and the defect it made visible is still
there** (row 13 — no pool rebuild after `BrokenProcessPool`). Left in place
rather than deleted, per the docket's own rule: read the entry, never the
title (`lead/DOCKET.md:2283-2303`).

**Ratified by the Architect as a measured family: A REVERT'S LANDING DUTY
INCLUDES SWEEPING EVERY RECORD THAT CITES THE REVERTED COMMIT AS STANDING.**
A revert that removes code but leaves its documentation is a revert only
half-done, and the half left behind is the half people read
(`lead/DOCKET.md:2301-2303`).

#### 2.4 Its mechanisation: `tools/revert_sweep.sh`, and the steady-state-exit-1 doctrine

Built, force-tested and run for real before the push that landed it
(`lead/DOCKET.md:796-839`, `tools/TOOLS.MD:25`). Finds every record still
citing a reverted commit as standing, across repo docs and the fleet's own
records; does not decide — each hit is corrected or marked
deliberately-standing-history. Exit status only: 0 clean, 1 cited (triage
every hit), 2 could-not-run (not a pass). Force-tested 9/9 across 6 arms,
including a firing control (planted citation → 1 CITED; unrelated sha on the
same tree → 0 CLEAN, proving both answers reachable). The force-test itself
caught a real defect on its first run: an empty `find` result fed one blank
filename to `grep`, which exits 2, so an empty tree reported COULD-NOT-RUN —
same family as a gate passing because nothing was staged (see PART 2.7).
Fixed; the arm is kept. First real run against `891243ce`: 55 hits across 6
files, four never checked by anyone; triage found 3 stale, 1 legitimate
history.

**THE DOCTRINE, now standing convention:** after a correct triage the tool
still exits 1 and the hit count *goes up* (55/6 before, 57/7 after), because
a correction block naming the SHA is itself a citation, and so is its own
documentation row. The tool enumerates; it does not classify, deliberately —
a checker that guessed which citations were fine would return a confident
wrong answer, worse than a list. **Exit 1 means "go triage," not "still
broken." The steady state after a landed revert is exit 1 with every hit
deliberate. Anyone who drives it to 0 will do it by deleting history — the
exact failure this convention exists to prevent** (`lead/DOCKET.md:832-839`).

#### 2.5 Four rules earned 2026-09-16, for the gesture batch (`lead/DOCKET.md:887-900`)

1. **`era` IS A UNIQUELY DANGEROUS SUBSTRING IN A FRENCH-COMMENTED CODEBASE.**
   `grep -i era` matches `serait`, `compteraient`, `tolerance_ms`,
   `format_generation`, `enumerates`. Exact tokens with word boundaries.
   Measured: a substring grep reported "1 era-bearing line" in a plan that
   contains zero, and would have turned "did not land" into "partially
   landed."
2. **PRESENCE SHOULD BE WHOLESALE** — a feature that should be
   absent-or-everywhere showing up ONCE is a MISREAD, not a partial landing.
   Generalised: **when a count comes back as 1 for something that has no
   reason to be 1, suspect the instrument first.**
3. **A CRITERION LOOSENED AFTER A FAILURE IS WORTH NOTHING IF IT CAN NO
   LONGER FAIL.** After correcting `era_check.sh`'s denominator (the
   candidate-duration clamp could not bind at all — the denominator was
   wrong, not the clamp, `lead/DOCKET.md:1729`), all four original arms were
   re-fired (4/4) — the correction is only admissible because the tool can
   still return every other answer.
4. **THE JUNIOR SEAT REFUSING TO IMPLEMENT OVER A SPEC CONTRADICTION IS THE
   CONSTITUTION WORKING FROM BELOW.** `dev-subcue` flagged `SPEC_ZONE_A` §4c
   while the Lead was dispatching the reversal and the Architect had
   authored it. It caught BOTH principals. Citable.

   **The flag and its withdrawal, both at full strength, author = the
   Architect.** `dev-subcue` flagged that the dispatched reversal would have
   overturned `SPEC_ZONE_A` §4c, a mandatory owner ruling, without citing it
   (`tools/TOOLS.MD:195`, `lead/DOCKET.md:858-859`). The ruling drafted on
   that reversal — `RULING_20260916_FILL_LANGUAGE_IDENTITY.MD` — **was
   withdrawn by its own author the same day it was issued** (`tools/TOOLS.MD:195`:
   "`RULING_20260916_FILL_LANGUAGE_IDENTITY.MD` was withdrawn the day it was
   issued — it reversed `SPEC_ZONE_A.MD` §4c, a MANDATORY owner ruling,
   without citing it; cross-language master fill stays legal"). It does not
   land in this gesture (see MANIFEST, do-not-land list).

#### 2.6 A negative control needs a known-WRONG PAIRING, not a known answer

Ruled by the Lead applying written rules, 2026-09-16 (`lead/DOCKET.md:861-898`).
Raised by `dev-subcue`, half right, and it raised the question instead of
silently hardcoding — that is the behaviour worth recording, not the error.
Its argument: a withdrawn ruling said "the bar comes from its controls,
never hardcoded," but a live production job cannot fire a control needing a
*known answer*, so the bar (0.85) must be a constant. **Correct for a
positive control, wrong for a negative one: a negative control needs a
known-WRONG pairing, which is always constructible** — forensic built one (a
span 130 s away, same track, same language → NCC 0.3645) without knowing the
right answer, only by deliberately pairing two things that should not match.
Ruled: keep 0.85 as a constant floor; add a per-region negative control
against a deliberately mismatched window of the same claimed master source,
with a stated separation required between the reading and that control.
**The outcome vocabulary gains a third state**: when the negative control is
not well below the reading, the answer is neither `content_verified` nor
`content_mismatch` — it is `content_indiscriminate`, an honest *I could not
measure* (implemented and fired on synthetic corpus, `tools/TOOLS.MD:195` —
all four arms fired: PASS/content_verified, PASS/no_fill_regions_to_check,
FAIL/content_mismatch, BROKEN/content_indiscriminate; not yet fired on a
real production artefact). Generalises beyond this ticket: any live
similarity/threshold check can carry its own negative control at negligible
cost, and without one its passes are uninterpretable on unfamiliar material.
The rule was then proven load-bearing by construction: dev-subcue's
verifier harness includes a deliberate `content_indiscriminate` case built
from TONAL MATERIAL — a pure tone correlates high against nearly anything,
so without the negative control the check would pass it; with the control,
the third state fires (dev-subcue's own report and the Lead's review,
2026-09-16 — the Lead's phrasing: *"a 440 Hz tone reads 1.0 against
anything and would have passed without it"*; Architect's session record).

#### 2.7 The empty-population family (three instances)

**"The verdict describes the mechanism, not the thing the mechanism was
pointed at."**

1. `revert_sweep.sh`'s own force-test defect: an empty `find` result fed one
   blank filename to `grep`, which exits 2, so an empty tree reported
   COULD-NOT-RUN instead of CLEAN — "an empty population reaching a check
   produces a verdict about the check, not about the tree" (`lead/DOCKET.md:819-821`).
2. dev-subcue's self-caught `--staged` defect, same day: a gate's first run
   used the tool's default `--staged` mode and trivially passed because
   nothing was staged; it re-ran `--worktree` before reporting — cited
   directly as the sibling case by the revert-sweep entry above
   (`lead/DOCKET.md:540-543,819`).
3. **A heartbeat that beats hardest when every cycle fails.** The `POSO`
   banner in the fusion worker's logs prints after the `except` block, in the
   same iteration — **POSO counts CYCLES ATTEMPTED, not SUCCEEDED**: a cycle
   that throws still prints POSO because the Error line precedes it in the
   same iteration. Zero POSO has two separate causes (loop stalled, or loop
   never started) that must not be conflated. Never report POSO alone; pair
   it with the Error line count (`tools/TOOLS.MD:24`, "POSO REFINEMENTS").

#### 2.8 Inspection bias: in-flight observation over-samples long jobs

The Lead's own "jobs exceed 2.5 hours" claim, retracted as the biased one:
two right-censored observations (>187 min, >167 min) caught at poll time are
not a random sample — **live polling over-samples long jobs by
construction, because the longest jobs are the ones still running when you
look.** Corrected duration methodology: build distributions **only from
COMPLETED boundaries** (`error_file_path` change plus artefact completion on
disk); keep censored observations in a **separate column, never averaged
in**. Actual pre-deploy rate measured from archive completion timestamps:
**~5.2 jobs/hour ≈ 23 minutes per job** — the two watched jobs were outliers
that seized both worker slots (`VMSAM_HELP_AI/supervisor/RUNLOG.md:154-230`,
`VMSAM_HELP_AI/supervisor/JOB_TRACKING.md:39-62`, `VMSAM_HELP_AI/supervisor/STATUS.md:257-265`).
Context (Architect's session record): the bias surfaced as a push-hold
question — two watched in-flight jobs at >187/>167 min against post-deploy
completions of 5-11 min read as a "twenty-to-thirtyfold" behaviour change;
the archive-throughput measurement (~5.2 completions/hour pre-deploy)
dissolved it as sampling artifact, the push proceeded, and the Lead
retracted the biased claim to the Supervisor with the reasoning.

#### 2.9 RAW vs RECONSTRUCTED (standing rule, arose with CI, binds every seat)

`VMSAM_HELP_AI/lead/DOCKET.md:311` — standing rule issued and binding on
every seat: a measurement built from a RECONSTRUCTED intermediate (an
inference, a re-derivation, a replay) must never be represented, filed, or
scored as if it were the RAW measurement the pipeline or instrument actually
took. The distinction travels with the number, not just in the seat's head.

#### 2.10 The instrument moved the population, then measured the absence it made

**"THE POPULATION WAS NEVER SHOWN TO THE MECHANISM AT ALL"** — named as a
family the same day, sibling of the empty-population family
(`lead/DOCKET.md:625-630`). Two grains of the same error, side by side: **a
seat pointing a probe at the wrong TREE — CI probing the archive it had just
created itself**, rather than the tree the population actually lived in; and
a principal (the Lead) pointing a probe at the wrong EXTENSION — a `*.log`
glob that could never match the `.mkv.log.error` files the token actually
lived in, producing a reported "zero current-era `fill_source_too_short`"
that was retracted as the Lead's own filter artefact, not a measurement
(`lead/DOCKET.md:602-616`: `--include='*.log'` → 0 vs `--include='*.error'`
→ 19, of which 4 current-era). **Rule: before trusting any zero from a
FILTERED search, fire the same search against material you KNOW carries the
thing.**

#### 2.11 `fusion.py`'s bundled blocked need, now smaller

The blocked need on `src/gestionar_show/fusion.py` (frozen) originally
bundled two problems: no visibility into a silently-dead fusion job, and no
rebuild of a `ProcessPoolExecutor` after a `BrokenProcessPool`. **The
visibility half is closed without touching the frozen file**, via the
owner's logs endpoint: `POST /webhook/showgestionar-test-recreate-logs`
(container 1 only; HMAC-signed; envelope kept lab-side, never
tracked/committed, cite opaque ids only) reads the swallowed-exception line,
worker death and the POSO heartbeat on demand (`tools/TOOLS.MD:24`).

**The pool-rebuild half remains open, and is owner-level.** `fusion.py:159`
submits into a module-level `ProcessPoolExecutor`; one `BrokenProcessPool`
invalidates the pool for every later submit and nothing ever rebuilds it, so
a single child death can turn the worker into a machine that accepts jobs
and instantly fails all of them, indefinitely, until someone restarts the
container. `891243ce` (now reverted, see PART 2.3) did not touch this — it
only made the failures visible instead of silent. **The rebuild is future
work with its own acceptance conditions; it is never a rider on another
landing** (Architect's ruling, 2026-09-16). Un-owned; no seat, no ticket.
Acceptance will have to distinguish a pool that is broken from a pool that
is merely busy (`lead/DOCKET.md:1805-1817`).

#### 2.12 Two target-first acceptance records that predate their events

Both pre-registered before the event they judge, per this campaign's own
target-first discipline:

- **"PRE-REGISTERED ACCEPTANCE — THE DEPLOY'S OWN PROOF"**, written before
  the push and before the recreate, naming the literal expected `VMSAM_ERA`
  tag value rather than "a tag," with three distinct, non-collapsible
  refutation arms (tag absent / tag present but `unknown` / tag present but
  a different SHA), each with its own cause and its own first check
  (`lead/DOCKET.md:743-784`). This one landed and passed as written: reviewed
  at the diff, six gate checks read by the Lead, and the tag reproduced —
  18/18 streams tagged, one distinct value, and 18/18 `git_commit=unknown`
  on the independently-fired REFUSED-rename arm (`lead/DOCKET.md:528-543`).
- **"PRE-REGISTERED ACCEPTANCE — THE PLAN-SIDE ERA STAMP,"** written before
  the code existed and before the mission was dispatched, so the implementer
  inherits an acceptance instead of authoring one (`lead/DOCKET.md:841-885`).

**One target-first criterion in the same family FAILED AS WRITTEN and was
corrected code-derived, with a 4-arm regression proving the correction still
falls back to every other answer.** `era_check.sh`'s candidate-duration
clamp could not bind at all as originally written — **the denominator was
wrong, not the clamp** (`lead/DOCKET.md:1729`). Corrected, then **all four
original arms were re-fired (4/4)** before the correction was admitted —
"a criterion loosened after a failure is worth nothing if it can no longer
fail" (`lead/DOCKET.md:895-897`, see also PART 2.5, rule 3).

#### 2.13 `verified=N/M` means a verdict was reached, not that N/M passed

`repair.py:1856`'s `verified_count` counts `outcome not in (None,
"skipped")` — a real alternative outcome closes the denominator through the
**existing** mechanism, no new denominator logic needed. Traced by a seat:
`verified=1/5` on a produced artefact was structural, not a bug —
`verify_on_master_timeline` (`chimeric.py:3766`) checks TIMING via a
same-language-only lookup, so every cross-language-filled track hits
`master_audio==None` and is `skipped` by construction, because there is
nothing to align on a fill whose position is asserted rather than measured
(`lead/DOCKET.md:906-911`). **E2 row specification: this table must aggregate
`verify=` OUTCOMES, never raw verified COUNTS** — a bare `N/M` invites
reading "M−N failed" when the true reading is "M−N reached a different,
equally valid outcome token." *(The exact citation `repair.py:1710` named in
the assigning brief was not located; `repair.py:1856` is the line this
mechanism was actually found at in the sources searched, and is cited in its
place.)*

#### 2.14 Class-3 framing: the cue-drop mechanism is sound, its input is wrong

Cue drops up to 96/track, tied to inflated interior-bracket/tail-gap spans.
**"H-A2 drops are named correctly; the spans are too wide. So the drop
mechanism is sound and its input is not."** (`lead/DOCKET.md:392-393`)
Pre-registered outcomes for the forensic leg that must run first before any
dev dispatch on this class: the filled span correlates with the master's
other-language track (wrong-language audio shipped, a product defect); the
span is silence/correlates with nothing (a different, silent-fill defect,
must not collapse into the first); or the span correlates with the
compared track's own material (no product defect — an artefact of reading
the plan, and the plan's vocabulary needs repair). All three outcomes are
pre-registered as acceptable; outcome three is explicitly not a failure.

#### 2.15 Archive relocation is non-uniform (open, CI, cause not established)

Archive-by-era relocation to `/config/output/archive/f42e1ed5/` measured
independently by the Lead: 121 files total (CI's RUNLOG logged 114 — the
later, larger count is the one to quote). **Two shapes from one operation**:
92 files FLAT (`f42e1ed5/<basename>`, depth 1) and 29 files SUBTREE
(`f42e1ed5/Plex/<lib>/<show>/Season N/<basename>`, depth 5). The relocation
was described to the fleet as deterministic subtree mapping — **true for 29
files, false for 92**. **Cause NOT ESTABLISHED**; assigned to CI to find the
branch. Until then, anything resolving an archived path must search by
quoted basename (`find /config/output/archive -type f -name '<name>'`), not
by the assumed subtree shape. Forensic's target-side evidence (the real
library under `/srv/...`) is untouched by this defect — it was never in this
store and never moved (`lead/DOCKET.md:572-587`).

#### 2.16 Brief-freeze scope gap closed at source, confirmed twice in production

**"THE INSTRUMENT DID NOT FAIL — IT WAS NEVER ASKED."** `check_briefs_frozen.sh`'s
scope was `BRIEF_COMMON.md`, `architect/BRIEF.md`, `lead/BRIEF.md` — 3 of 6
briefs; `supervisor/BRIEF.md` sat at mode 644 while every other brief was
444, undetected because the guard was never asked about it. Root cause,
measured by grep: **`spawn_agent.sh` never chmods a brief** — the only
`chmod` in it targets a test stub; `--bless` chmods only its own three named
files. So nothing in the spawn path sets 444; the other five briefs were 444
only because an earlier Lead had done it by hand, and a freshly written
brief arrives 644 by default (`lead/DOCKET.md:435-460`). **Design caution
preserved: do NOT hash-freeze seat briefs** — an ephemeral seat's brief
legitimately changes as the Lead dispatches new missions, and hash-freezing
it would make the Lead's own dispatch path a violation. Two properties, two
treatments: MODE-freeze (444) on ALL briefs, enforcing that an agent never
modifies its own brief; HASH-freeze (manifest) on the permanent three only.
**Closed at source**: `chmod 444` the brief inside `spawn_agent.sh` at spawn
time, proposed to and implemented by the Architect, so the property holds by
construction. **Confirmed twice in production**: `ci/BRIEF.md` came back 444
straight out of `spawn_agent.sh` on CI's recycle (`lead/DOCKET.md:554-556`);
`corpus-pairs/BRIEF.md` came back 444 and was verified still 444 after spawn
on the corpus-pairs seat's launch (`lead/DOCKET.md:930`).

#### 2.17 Undelivered disposition: the owner's word settled the principle, not the individual dispositions

Owner ruling, verbatim: *"Je ne vois pas a quoi ce dossier sert dans le
developpement. Soit c'est un corpus et il est dans le mauvais dossier, soit
c'est bon pour la poubelle."* — corpus or trash, no third state
(`lead/DELETION_LOG_20260916.md:1-9`). This settled the **principle**; it did
not settle which of the 40 `undelivered/` cases was which. Of 40 case
directories (47 files, 9.35 GB), 20 were deleted outright (no hit across the
search list), 18 named-evidence cases were moved to
`VMSAM_CORPUS/refused-products/`, and **2 were left UNDETERMINED** — a
prior-campaign forensic register used the identical `case_key` grammar but
its own README disclaimed reuse against the current disk, so the classifier
correctly refused to guess either way.

**The Lead was inclined to delete both, and did not decide alone**: it asked
forensic — the seat that would actually use the material — instead of
ruling from absence-of-evidence. Forensic named a use for one of the two:
`970948223b8144e8` graduated to corpus (a second, independent,
prior-build 2026-09-05 exemplar of cross-language borrowed fill, r=0.996–0.999
at lag 0, negative-controlled r=0.01–0.21 elsewhere — evidence the defect
class predates today's case, directly relevant to the §4c reversal question
then with the owner). The other, `648dfc1cea9d6d28`, was deleted on
forensic's own word: "I do not have a use for it"
(`lead/DELETION_LOG_20260916.md:42-52`). **The check that changed an outcome
was asking the seat that would use the material — a principal deciding alone
would have been wrong about one case in two.**

#### 2.18 The pipe-trap, both halves, and its near-miss in CI's own procedure

**Half 1 — exit status through a pipe.** `git push -q 2>&1 | grep -v
"^remote:" ; echo "push EXIT=$?"` reads grep's exit status, not git's; `grep
-v` returns 1 when it filters everything away, exactly what happens on a
successful push whose entire output is `remote:` lines. The Lead read this
as a failed push; the push had succeeded (`RULES_EARNED_20260915_EVENING.md:405-423`,
carried in PART 1b above).

**Half 2 — an unquoted multi-line variable through `echo`.** CI's own
drain-defect procedure captures an HTTP response with `response=$(curl -v ...
2>&1)`, then reads it back as `http_status=$(echo $response | grep "< HTTP"
| cut -d' ' -f3)` and `response_body=$(echo $response | tail -1)` — **both
`echo`s are unquoted against a multi-line, verbose-curl capture**, which
word-splits and collapses the variable before `grep`/`tail` ever see it
structured (`VMSAM_HELP_AI/ci/RUNLOG.md:5719-5729`). **Measured near-miss,
not yet a landed defect**: the procedure is written this way in CI's own
RUNLOG at the line cited; whether it has actually produced a wrong
`http_status` or `response_body` in a real run was not established in the
sources checked here and is not asserted.

**Placement rationale, verbatim: "true, useful, and in the wrong room."**
Both halves are real, general shell-scripting hazards this campaign has now
measured twice — but neither is a governance-document control, a spec, or a
ruling; they belong in FINDINGS as measured operational hazards, not in §5 of
`WRITE_ZONES.MD` as procedure. (This is why PART 4 of this gesture edits §5
only for the fail-closed identity guard and the framing sentence, and does
not fold the pipe-trap in as a third snippet.)

#### 2.19 Two signed Lead-#4 failures, kept at full strength at their author's own request

Both from the Lead's own reports to the Architect, 2026-09-16 (Architect's
session record; the Lead verifies his own ledger lines at the gate):

1. **The 21-minute bound slip.** The Lead set a push bound of 11:45Z with
   explicit arithmetic, then pushed at 12:06:14Z — he estimated elapsed time
   instead of measuring it, the identical error he had corrected in the
   Supervisor at its cycle 2 that same morning, committed on the decision he
   had built the entire window around. Self-caught while verifying an
   unrelated timestamp; it cost nothing (no boundary fired in the overrun),
   and he reported it rather than letting it be inferred from timestamps.
2. **The miscalibrated pre-registered era criterion.** His acceptance said
   `VMSAM_ERA` must sit on "EVERY stream"; the first real artefact returned
   ERA-PARTIAL (22 of 91) — **instrument-caught**: the check's own exit
   failed the criterion as written. The 22 were exactly the FABRICATED
   streams; the denominator was wrong, not the code (an era tag on a copied
   stream would lie about provenance). The correction was derived from the
   code without looking at any result, written into the tool beside the rule
   with the admission, and all four refutation arms re-fired (4/4) before
   the re-score — see PART 2.5 rule 3 and PART 2.12.

His own framing, kept: *"every one was caught by an instrument or a seat"*
(the second literally by the instrument; the first by his own re-check
discipline) — *"I would rather that be the record than a tidier one."*

#### 2.20 Stability is not correctness, and the wrong-question family (added by the Architect at review — measured after this draft's cutoff)

**Two identical readings prove STABILITY, NOT CORRECTNESS — a wrong parser
is perfectly stable.** The Lead's `/errors` parser did `d.get('errors', d)`
against an endpoint whose key is `incompatible_files`, fell back to counting
the dict's keys, and returned **1 — stably, on every reading**. It would
have passed the fleet's own "two consecutive identical readings" rule all
afternoon; a repopulation stall was reported that never existed (the base
held 317 throughout — corroborated by the Architect's independent
measurement, which survived by luck of parser choice, not virtue). **Rule
amendment: before trusting a stable value, check the SHAPE of what was
parsed, not the repetition of the number** — issued to CI before it used
317 as the stage-5 denominator, the one number a whole sweep rests on.

**The family it closes out, named by its author (Lead #4), four instances in
one day, one principal, all self-caught: "I asked the instrument the wrong
question and believed the answer."** An estimated timestamp instead of a
measured one (the bound slip, 2.19); a `*.log` glob that excluded the
`.log.error` class the token lived in (2.10); an n=1-derived duration figure
amplified into a fleet claim (2.8); a dict key that counted keys instead of
rows (this entry). (Architect's session record; ledger lines verified by the
Lead at the gate.)

#### 2.21 The quantum carries its window: two correct, non-interchangeable quanta in one pipeline (added at review)

dev-pal's consumer census (its lab, `dev-pal/001-quantum-ms-census.MD`)
measured **`quantum_ms = 129`, invariant, across 118+ readable plan
emissions** — zero at 125, zero at None. Not a defect: `change_point_locator.py:186`
sets `PROBE_WINDOW_SECONDS = 60.0`, a fixed window, and the quantum is a
property of the window actually passed — a fixed window yields a fixed
quantum, which is the rule OBEYED. **The unwritten consequence, now
written: the same pair carries TWO different quanta** — the locator's 60 s
probes → 129 ms; the pipeline's merge window (`L = 2*int((dur-90)/11)`,
`>300 s` branch) → ~125 ms at ~24 min, ~124 ms at ~44 min. Both correct for
their own window, **not interchangeable: mixing them carries a systematic
~3.2 % per-point error**. Unit clause appended to the three-argmaxes
ruling: *the paths do not even share a unit — any cross-path comparison
converts through each path's own window, or it is not a comparison.* The
quantum band table's applicability boundary (already queued for this
gesture) is subsumed here. Hedge kept verbatim from the seat: **zero
PAL-family ids in that sample** — the population the E3 mission targets was
still being copied; the saturation-orthogonality argument is labelled
inference until the pairs land.

**And it convicts the `else 125` fallback twice** (`change_point_locator.py:1660`):
it renders "no quanta were measured" as a VALUE — inside the exact constant
AGENT.MD singles out — and the value it substitutes is the PIPELINE's
quantum, inconsistent with every one of the module's own 118 measured
emissions. Ruled (Architect, 2026-09-16): the fallback becomes
decline-at-admission — dev-pal's census shows four unguarded division sites
immediately downstream, so the shape is structural, not stylistic — under a
NEW granted token **`no_quantised_points`** (probes ran, energy floor
cleared, every kept probe's quantised `correlate()` returned `points == 0`),
always paired with `measurement=could_not_run`, both arms fired with
literals before landing, and the token entering `ARTEFACT_FORMATS.md` §9c's
canonical grammar table **in the same diff** — a token born outside the
grammar table is the two-grammars defect at birth.

#### 2.22 The era vocabulary: measured / reconstructed / inferred (ruled at review, 2026-09-16)

A verification pass pinned a pre-tag artefact's producing commit EXACTLY by
matching **two** independent content-derived build digests jointly against
the digest trajectory of candidate commits (the chimeric digest changed at
every commit after the match, making the equivalence class a singleton).
This is not a breach of the `ARTEFACT_FORMATS` trap ("`build=` is a module
identity, never an era key") — the trap forbids READING a digest as a
LABEL; a joint reconstruction against history is a per-artefact MEASUREMENT.
Nor is it "retro-guessing" under the transitional rule — it is
retro-MEASURING, which the rule did not distinguish. **Ruling — three era
states, the label travelling with every number:**

- **era-MEASURED** — the `VMSAM_ERA` tag: raw, artefact-internal, no lookup.
- **era-RECONSTRUCTED** — ≥2 independent content-derived digests whose JOINT
  match resolves to a UNIQUE commit class within the artefact's timestamp
  window; recorded with its command and the class stated; a class holding
  more than one commit classes the artefact to the CLASS, never to a picked
  commit. RAW-vs-RECONSTRUCTED applies literally: this is reconstructed
  evidence and says so wherever it is used.
- **era-INFERRED** — everything else (mtime, directory label, deploy clock):
  permanent, as before.

No blanket upgrades: archived pre-tag artefacts stay INFERRED until a row
needs an era and pays for its own reconstruction. Pre-registered predictions
that specified THE TAG score on the tag only. And the Lead's line enters the
record beside it: **"a pre-registered prediction updated when new
information arrives is not a prediction, it is a commentary."**

---

### PART 3 — DELEGATION

**The junior seat refusing to implement over a spec contradiction is the
constitution working from below.** `dev-subcue` flagged `SPEC_ZONE_A` §4c
while the Lead was dispatching the reversal and the Architect had authored
it — catching both principals in the same motion (`lead/DOCKET.md:857-859`,
PART 2.5 rule 4). The ruling built on the reversal,
`RULING_20260916_FILL_LANGUAGE_IDENTITY.MD`, was **withdrawn by its own
author — the Architect — the same day it was issued**, because it reversed a
mandatory owner ruling without citing it (`tools/TOOLS.MD:195`). Both the
flag and the withdrawal are carried here at full strength: the flag,
because a junior seat's refusal caught what two principals had not; the
withdrawal, because the Architect's own retraction of his own authored
ruling — searched, found wrong, reversed, same day — is exactly the
self-correction discipline this campaign measures and rewards elsewhere
(compare PART 1e's `/health`-scoping retraction and PART 2.19's
model-selection candidate). Cross-language master fill **stays legal**;
`SPEC_ZONE_A` §4c stands uncontradicted going forward.
`RULING_20260916_FILL_LANGUAGE_IDENTITY.MD` does not land in this gesture
(see MANIFEST).

---

### PART 4 — BLOCKED NEEDS

- **`fusion.py`'s pool-rebuild half** — see PART 2.11. Owner-level; no seat,
  no ticket; acceptance conditions not yet written; explicitly never a rider
  on another landing.
- **`BrokenProcessPool`'s general acceptance question** — distinguishing a
  pool that is broken from a pool that is merely busy — is named as future
  work belonging to whoever eventually owns the rebuild (`lead/DOCKET.md:1814-1817`).
- **Archive relocation's cause** (PART 2.15) — assigned to CI, not yet
  established; a frozen-file/no-frozen-file determination was not part of
  this search and is not claimed here.
- **The two-stage disagreement** noted inside the median-fidelity finding
  (PART 2.1): stage 1 (chromaprint) vs the locator producing offsets three
  orders of magnitude apart on the same pair, with nothing comparing them —
  held as an open question pending a generalisation check, not promoted to a
  defect.

---

*Evidence pointers above that name a file without a line number are
citations to the document as a whole, read in full at drafting time. Where a
specific claim from the assigning brief could not be located in the sources
searched (PART 2.6's tone-frequency illustration, PART 2.8's "E1"/"30x"
labelling, PART 2.13's exact line number, PART 2.19's two Lead failures),
that is stated in place rather than invented, per instruction.*

## Gesture #3, additive completion — 2026-09-16 evening (authorised under the same carte-blanche as 8d31c553)

`8d31c553` landed the 46-rule synthesis and declared
`lead/RULES_EARNED_20260915_EVENING.md` fully carried. **The premise was
false, and the deletion was correctly blocked on it** (the Lead, measuring
before deleting): one rule of 55 was entirely absent, and a per-body sweep
ordered by the Architect then found the synthesis had thinned 16 more —
titles carried, load-bearing substance dropped. This entry restores all of
it, near-verbatim, because the lab file is deleted once this lands and
whatever this entry does not carry stops existing. Nothing below revises
`8d31c553`'s entries; every block is additive to the entry whose title it
names.

### The absent rule, restored in full (PART 1, theme: dispatch and briefing)

**THE LEAD-SIDE HALF, and it is the real lesson: I gave the asymmetry
warning to ONE seat and not the other.** dev-extent was told
presence-proves-new-era-absence-proves-nothing and read the artefact
correctly; CI was told to classify by digest and was not told how to read
the line, and misfiled it. Same artefact, two seats, and the difference was
which briefing I wrote. A caveat given to the seat that already understands
it, and withheld from the seat about to act on it, is a caveat that does no
work. **When a property of an instrument is subtle enough to need
explaining once, it needs explaining to EVERY consumer of that instrument.**

*(Recorded with its own irony, because the irony is the finding: this is
the rule about a caveat delivered to one consumer and withheld from
another — and it was the one rule dropped from the document that exists so
nobody has to be told twice. Its pairing partner — "a guard that speaks
both ways must be triaged on its verdict token" — was carried; the pair was
split and the second half lost.)*

### Restored particulars of sixteen thinned rules

Each block names the landed rule it completes and restores only what the
synthesis dropped.

**"A guard that speaks both ways…" — the triage mechanism itself.**
Presence = the guard RAN; `REFUSED` + `decided_by=` = it refused. The
extraction-bound audit line fires on the PASSING path too — that is what
makes its absence readable as "the guard never ran", and the same property
made CI count a clean PASS as one of two expected refusals. Two facts in
one artefact (a passing extraction bound and a decline from the fill-source
verifier) were merged because they shared a file.

**"When the bridge is down, drop a file in the seat's lab…" — the
corollary that shows the safeguard working.** The note must name its
author, its seat and the time IN ITS OWN TEXT (the incident note said
"your /rc registration failed, so I cannot reach you by message" — which is
how authorship was eventually reconstructed). And the receiving seat,
inv-fill, treated the unsigned note as UNVERIFIED DATA, checked every claim
against the real logs before acting, and corrected two of its three
inferences — the only reason an unsigned instruction did not land
unchecked.

**"The transport failure is older than it looked" — the grounding
instance.** The bridge was down at 18:10 and known to Lead #1; Lead #2
learned it at 19:40 and in between read the seat's silence as
work-in-progress. A known-broken bridge must be handed over explicitly; it
was not in the handover.

**"Before you read a silence as a result, prove the channel can speak" —
the evidence.** Verified before opening a single artefact: zero `tools.dev`
hits in the module; the same ungated `tools.logs` channel demonstrably
reaches existing artefacts (`repair: USED audio track`); and the control
that could have refuted it — no artefact in the corpus contains ANY
`chimeric:` line — was examined rather than waved away, and explained
(pre-existing `chimeric:` emissions are rare exception paths; the new line
fires per track, so if the channel carries it, it will be abundant).

**"The test instance runs dev: false" — the consequence past one ticket.**
That is why production never logs per-position/minority pairing scores, and
why any future "the log does not show X" reasoning about that instance must
first establish that X is ungated. A property of the instrument — one that
silently converts "not logged" into "did not happen".

**"A nudged execution is not the mechanism working" — the warning that is
the point.** Recording the post-question fire as "the first automatic fire"
would have taught the next Lead the mechanism works and to stop checking.
The authority is confirmed; the mechanism's silence is the finding; the
question is narrow: why does a rule a seat holds, with standing authority,
at the top of every cycle, not run?

**"A seat's silence is not evidence that it is working" — the procedure.**
Never infer a seat's state from the absence of messages: check the pane
(`tmux capture-pane`), the session (`spawn_agent.sh --list`) and the lab's
file mtimes. `ListAgents` "offline" means the BRIDGE is down, not that the
process is dead. And this is why the files rule pays: had the seat held its
finished partition in context and sent it only as a message, the transport
failure would have destroyed a full cluster's work invisibly.

**"A prediction that lives in a message cannot be checked" — the rule and
the fix.** The rule binds the person STATING the prediction, not only the
person measuring it. Every expectation now lives in the docket (§6) AND in
the lab of whoever will read the run. An expectation held in one place is
held nowhere.

**"An instrument's output is not a media fact" — the correction, not just
the error.** The saturation rail is a SCREEN, not "the PAL detector":
29/29 content-mismatch ids saturate too; the duration ratio is the
discriminator.

**"A sub-agent's return is a report to assess" — the extension and the
tally.** Apply to your own sub-agents the standard you apply to seats. Two
retractions that night, both self-caught before reaching FINDINGS.

**"A counter of failures is not a counter of work — AND IT INVERTS" — the
whole body (only the title had landed).** The only running instrument
counted `.log.error` population — produced by FAILING jobs only; a
successful merge writes none. So the count is a FLOOR on completion divided
by an unmeasurable failure fraction, and a run of SUCCESSES is
indistinguishable from a STALL in it. Measured: 6 artefacts in 11 min
(0.55/min), then zero across 7 consecutive minutes — while 5 `.mkv` were
produced in the same window; read as throughput, that zero says "hung", it
actually says "working, and working WELL". **Before using a count as
progress, establish whether it counts the thing or its complement.** The
seat supplied the floor and explicitly REFUSED to convert it into an ETA —
a refused number is worth more than a plausible one.

**"A hold must be re-decided when the thing that justified it dissolves" —
the words and the mirror.** The seat's own charge, against the Lead, in its
pulse to the owner: "it stopped being re-decided after the thing that made
it optional got built" — the context had become a validation table and
per-track predictions sharp enough for a stranger to score. This campaign
had already earned "a gate nobody revisits is not a gate, it is a stop" —
applied to everyone except the seat that wrote it down.

**"A pane's ❯ line is an input buffer" — the other three of four
instances.** A seat reported "closed for tonight" on a sentence describing
a ruling rather than its own status; the Lead read a prompt's
absence-from-his-own-outbox as evidence of its origin; and the corollary
the Lead got wrong: a ❯ line neither principal authored is probably the
OWNER typing directly into a watched session — his instruction in flight,
and telling a seat to disregard it puts the Lead's convenience above the
owner's direction.

**"The gate does not scale with delegation" — the scope bug, named.** The
fourth verification caught what the self-test alone passed: a bare
assignment creating a Python local, leaving the module tuple polluted for
any process that self-tested first — and the commit path runs both.
"Verified by a sub-agent" never means verified on this seat.

**"Mutation-test a guard" — the second property and the close.** The
checker also reads both the accepted class and the sentinel literal FROM
SOURCE rather than hardcoding them. Proving the guard can fail is the half
nobody does.

**"The scar is on the wrong limb" — the checkable pointers.** The incident
and fix live at `api.py:394-396`; three processes set `tools.dev` from TWO
sources — `main.py:30` and `main_gestionar_show.py:440` take `args.dev`
(CLI), `fusion.py:318` takes `get_dev_env_var()` (ENV) — and `api.py:396`
reports `get_dev_env_var()`. Exact for the fusion worker, divergent for the
CLI-launched daemon, which is the process the incident bit. `/health`
proves the FLAG, never that a line reached a reader.

*(Also lightly thinned, restored in one line each: the sentinel rule's
count — reusing `(unstated)` would have given the loud signal 22 false
instances, after which nobody reads it.)*

### Method entry: how the loss was found, and what the finders got wrong

**A byte- or line-count match on a SYNTHESIS proves nothing about its
completeness.** Both principals verified gesture #3's item 1 by size
(903 lines) — the third instance in one day of the family "a verification
that fires on a proxy of the thing" (the Lead's spawn-gate substring, the
Architect's first-line anchor grep, and this — the strongest, because the
proxy was agreed by both principals independently). The only check that
could find the loss was content-level, per rule, and it ran only because a
deletion forced it.

The per-body sweep then added two of its own lessons: **a whole-file grep
can clear a phrase through unrelated reuse** (one search phrase matched
only a coincidentally identical quote from an earlier gesture — the check
had to be re-run against the gesture-3 slice alone); and the sweep's own
margin note flagged two bullets its verdict list omitted — one proved
fine, one proved the worst thinning of the set. **A verifier's summary is
itself a synthesis; its margins are part of the report.**

## Relocated from BRIEF_COMMON.md by the owner, 2026-09-21 — "these are findings"

The owner removed the "domain invariants (Architect's reminder)" section
from `BRIEF_COMMON.md` because its content is MEASURED FINDINGS, not
standing brief material. Carried here so the measurements survive:

`audioCorrelation` returns `(correlation, offset_in_points, delay_ms)`.
One point = 125 ms, the chromaprint frame — generally the norm, to
+/- 1 ms (exact domain below). Every delay seen is a multiple of the
quantum (8125 ms = 65 points; 47750 ms = 382). A delay that is not a
multiple of the quantum questions the reading before the media.

The quantum is `lengthFile / n_items`, so it MOVES with the window
(owner-authorised correction 2026-09-15; measured: TASKS/005 §5's fpcalc
table + the real window formula, `video.py:474-485` with
`mergeVideo.py:774` — `L = 2*(duration-90)/11` on the `>300 s` branch):
125 ms exactly for L in [160, 286] s — episodes of ~16.2 to ~27.7 min;
124 ms from L = 288 (~27.9 min) up, 44-min episodes included. Short
formats are the exception: a ~10.7-min episode (L = 100) reads 127 ms,
and between L = 100 and 160 nothing was measured — do not interpolate a
claim there. So 125 +/- 1 holds from ~16 minutes up; below that it does
not. NEVER hardcode 125: derive the quantum from the window actually
passed (`TASKS/008` measured a real pair at gcd 124 ms — a hardcoded 125
would have mis-scaled it).
