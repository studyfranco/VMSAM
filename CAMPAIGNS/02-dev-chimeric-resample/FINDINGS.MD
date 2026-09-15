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
