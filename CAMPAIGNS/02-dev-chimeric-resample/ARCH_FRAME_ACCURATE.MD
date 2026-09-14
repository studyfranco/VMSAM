> **BLOBS THIS DOCUMENT IS WRITTEN AGAINST** — corrected 2026-09-07 after a clone
> census showed my working copy was stale. **Authority `dev-AI` resolved as
> `72738949` at 2026-09-07T22:16Z** — the Lead's repaired obligation 2: a blob
> citation carries the instant the authority was resolved, or it is not a citation.
>
> *I would go one step further than the instant, because I have measured a
> timestamp lying: **NAME THE AUTHORITY TIP ID, NOT ONLY THE CLOCK.** A tip id is
> verifiable by any reader forever and needs no shared clock; an instant needs a
> clock, and this seat has already recorded ledger headings drifting 8 h against
> git's own times. Carry both; the id is the one that survives.*
>
>     src/frame_compare.py          **d18e8bbd**  220 lines   == AUTHORITY
>     src/get_cut_time.py           **8928e3ea**  593 lines   == AUTHORITY
>     src/merge_video_chimeric.py   **b282768f** 3051 lines   == AUTHORITY
>     src/change_point_locator.py   **28a076f9** 1848 lines   == AUTHORITY (re-derived)
>
> *** **THE FIRST VERSION OF THIS DOCUMENT READ `change_point_locator.py` AT
> `814aebb7` — 1461 LINES, 387 SHORT — THE SAME STALE BLOB I RETRACTED A CLAIM
> OVER EARLIER THE SAME NIGHT. Every VALUE I cited was right; every LINE NUMBER
> was wrong; and one structural claim had already been fixed at the authority.** ***
> Three of the four files matched; the one that did not is the one the design turns on.

# FRAME-ACCURATE CHIMERAS — the architecture, transposed onto the code that exists

Owner's order, 2026-09-07: *transpose the architecture to obtain frame-accurate
chimeras per my proposal.* Owner's spec: **do not replace the whole problematic
stretch — repair the exact zone, to the frame.** fpcalc locates the ZONE, pHash
decides the FRAME. Three constraints: raw fingerprint list not `correlation()`;
frame numbers not milliseconds; do not route through `cross_correlation`.

Measured against `origin/dev-AI`. I do not edit `src/`. This is a design and a
set of measurements, handed to whoever owns the modules.

---

## THE HEADLINE: BOTH HALVES EXIST. THE JOIN DOES NOT.

    src/frame_compare.py       FrameComparer, DCT 8x8 -> 64 bits, banded Hamming
                               alignment, returns start_frame/end_frame in TARGET
                               coordinates.  *** This is the owner's stage 2,
                               already built, and correctly shaped. ***

    its only importer          src/get_cut_time.py
    that file's line 3         "get_cut_time.py -- RETIRED. Do not import, do not
                                build on this file."
    live importers of it       *** ZERO ***

    the LIVE locator           src/change_point_locator.py
    what it calls at :659      audioCorrelation.second_correlation   <- FORBIDDEN
    and at :667                audioCorrelation.correlate            <- FORBIDDEN

*** **THE OWNER IS NOT ASKING FOR A NEW ALGORITHM. HE IS ASKING FOR A REWIRE.
THE FRAME STAGE IS WRITTEN AND REACHABLE ONLY FROM A FILE THE REPO FORBIDS
IMPORTING; THE LIVE PATH USES THE EXACT ROUTE HE RULED OUT.** ***

---

## WHY THE TWO-STAGE DESIGN IS A NECESSITY, NOT AN OPTIMISATION

The code already carries the number that proves it.

    CHROMAPRINT_HOP_SECONDS = 4096/3/11025          = **123.840 ms**   (the hop)
    the module's own comment: int(lengthFile/n_items*1000) is **124-142 ms**
                                                     depending on window length
    smallest CONFIRMED real step, id 13              = **41.7 ms = ONE FRAME @ 24.000**

    one fingerprint point = 123.84 / 41.7  =  **2.97 frames**
                          = 142.0  / 41.7  =  **3.41 frames**

**The owner's "1 point = 3-4 frames" is exactly right.** And the module also
states a step must *"exceed roughly half a quantum"* to be seen at all — **62-71 ms,
1.5-1.7 frames.** *** **A ONE-FRAME ERROR IS BELOW THE AUDIO STAGE'S DETECTION
FLOOR. THE ZONE STAGE CANNOT SEE THE THING THE OWNER IS ASKING US TO FIX. ONLY
THE FRAME STAGE CAN.** ***

### sizePoint is a PARAMETER and the owner is right to call it one

`60000/463 = 129.59 ms` is an **empirical** per-point size for one window. It is
not the hop (123.840) and it is not a constant: the code's own band is 124-142 ms
**and it varies with window length**. Hard-coding 129.6 costs up to **12.4 ms per
point — 0.3 frames per point, accumulating across the window.** Compute it as
`window_ms / n_items` per window, or the frame stage inherits a drifting bracket.
*(The `126.6` in the spec text is a transposition of `129.6`.)*

---

## FOUR DEFECTS THAT DEFEAT FRAME ACCURACY IN THE STAGE THAT IS SUPPOSED TO PROVIDE IT

### 1. THE FRAME RATE IS COERCED TO AN INTEGER, AND THE VIDEO IS RESAMPLED TO IT

    frame_compare.py:29   self.fps = int(max(1, fps))
    frame_compare.py:49   "-vf", f"scale={w}:{h},fps={self.fps},format=gray"
    get_cut_time.py:384   fps=int(round(fps_use))

`23.976 (24000/1001) -> 24`. ffmpeg's `fps` filter **duplicates frames** to make
up the rate. Over the owner's **30 s window floor**:

    30 s x 23.976 = **719.28 real frames**  ->  **720 frames emitted**
    *** ONE DUPLICATED FRAME INSIDE EVERY WINDOW, AT AN UNPREDICTABLE POSITION. ***

and the index is then converted back on the same invented grid:

    frame_compare.py:214   start_frame = int(round(start_time * self.fps))

*** **ON EVERY 1000/1001 RATE -- WHICH IS ALL NTSC CONTENT -- THE RETURNED FRAME
NUMBER CAN BE OFF BY ONE. ONE FRAME IS THE ENTIRE TOLERANCE THE OWNER IS ASKING
FOR. THE FRAME-ACCURATE STAGE IS NOT FRAME-ACCURATE ON NTSC.** ***

FIX: carry the rate as an exact rational (`r_frame_rate` num/den, or MediaInfo
`FrameRate_Original`), do not resample, and return the index **with the grid it
is expressed on**. A frame number without its grid is the same error as a count
without its surface.

### 2. THE pHASH PATH IS GATED ON FPS EQUALITY -- WHICH THE CHIMERA CASE VIOLATES

    get_cut_time.py:377   if fps_main is not None and fps_tgt is not None \
                              and abs(fps_main - fps_tgt) < 1e-6:

The owner's own demonstration is a PAL/NTSC speed change: **240 frames stay 240
while the duration goes 10.010 s -> 9.600 s.** That is *different rates by
construction*. On unequal rates this falls through to scene detection and then to
`_refine_cut_audio` -> `second_correlation`.

*** **THE FORBIDDEN ROUTE IS REACHED EXACTLY IN THE CASE THE OWNER CARES ABOUT.
THE FRAME STAGE IS SKIPPED PRECISELY WHEN THE SOURCES DIFFER -- WHICH IS WHAT
MAKES IT A CHIMERA.** ***

Either the candidate is resampled to the master grid **before** refinement, and
the equality test then holds by construction, or the frame stage must work across
two grids and say which one its output is on. **It cannot silently do neither.**

### 3. A FABRICATED DEFAULT FRAME RATE

    get_cut_time.py:369   fps_use = fps_tgt if fps_tgt is not None \
                              else (fps_main if fps_main is not None else **25.0**)

When neither probe yields a rate the code **invents 25.0** and every frame number
downstream is expressed on a grid nobody measured. **A frame-accurate pipeline
must DECLINE here.** `m=0` is untested, not clean; an invented denominator is the
same failure as an empty population reported as PASS.

### 4. NOTHING GATES ON VARIABLE FRAME RATE

`merge_video_chimeric.py:1941-1945` reads `FrameRate_Mode` and
`FrameRate_Original` and **nothing decides on them**. *** A frame number on a
variable-rate file is not a coordinate. *** Refuse, or resolve through
`FrameRate_Original`.

---

## THE OWNER'S TWO GUARDS ARE NOT IN THE CODE AT ALL

    window floor 30 s   PROBE_WINDOW_SECONDS = 60.0 exists; **no floor is named**
    hard wall 4.05 s    *** 0 occurrences of 4.05 or 4050 in the locator or the
                        repair module ***

Both are spec-only. They must land as named constants **with the owner's
derivation written beside them**, or they will drift the way the two step gates
did -- those were found INVERTED, and the module records that nothing in the code
said which way round they belonged, which is why it could invert unnoticed.

---

## THE TRANSPOSITION

    STAGE 0  ADMISSIBILITY  (new -- it REFUSES where the code currently defaults)
      exact rational rate for BOTH sides, or DECLINE. No 25.0.
      CFR, or FrameRate_Original present, or DECLINE.
      every number downstream carries its grid (num/den).

    STAGE 1  ZONE -- from the RAW FINGERPRINT LIST            (owner constraint 1)
      compare fpcalc items item-by-item. NOT correlation(), NOT cross_correlation().
      sizePoint = window_ms / n_items, computed PER WINDOW. Never a literal.
      resolution 3.0-3.4 frames => the output is a BRACKET, never a boundary.
      window floor 30 s; the existing 60 s / 40 s geometry already satisfies it.
      emits (start_point, end_point, sizePoint_ms, window_ms, n_items)
            -- the points AND the constant that converts them, together.

    STAGE 2  FRAME -- pHash over the bracket                  (owner constraint 2)
      FrameComparer at the NATIVE rational rate, no fps resampling.
      DCT 8x8 -> 64 bits, banded Hamming: already implemented and correct.
      if the bracket exceeds the 4.05 s hard wall -> DECLINE, do not refine.
      emits (start_frame, end_frame, grid_num, grid_den, method="phash_frames")

    STAGE 3  CARRY THE FRAME, NOT THE MILLISECOND             (owner constraint 2)
      the plan/segment structures in merge_video_repair.py and
      merge_video_chimeric.py carry MILLISECONDS throughout.
      At 23.976, 1 ms = 0.024 frames. *** A frame converted to ms and back is NOT
      the same frame whenever the boundary sits within ~20.8 ms of a half-frame. ***
      The frame index must travel as the PRIMARY value; ms becomes derived and
      display-only.

    STAGE 4  THE FORBIDDEN ROUTE, STRUCTURALLY REFUSED        (owner constraint 3)
      change_point_locator.py:659 must not be the AUTHORITY for a frame boundary.
      Keep second_correlation as a cross-check that cannot be promoted to the answer.

---

## DO NOT COPY THE CALLER

`get_cut_time.py` is retired for **four documented defect classes**, including:
*"`second_correlation` returns (file, offset_in_SECONDS). All four callers here
divide that offset by the frame rate, wrong by a factor of ~24, under a comment
asserting a project convention the project does not use."*

**The pHash class is sound. Its caller is not.** Take `frame_compare.py` and the
shape of the call; take nothing else from that file.

---

## WHAT I HAVE NOT MEASURED

- No real PAL/NTSC pair has been run through this. **The owner states both his
  demonstrations are synthetic and real content is unmeasured; I confirm I have
  added nothing to that.**
- The off-by-one from `fps` coercion is derived from the filter's documented
  behaviour and the arithmetic above, **not observed on a file**. It is a
  prediction. It is falsifiable by extracting a 30 s window from a 23.976 source
  twice -- once with `fps=24`, once native -- and comparing frame counts. **720 vs
  719 confirms it; 720 vs 720 refutes it and this section is wrong.**
- The 0-to-end unbounded cut is not done, per the owner.


---

## CORRECTION, 2026-09-07: `no_shared_content` NO LONGER EXISTS, AND I CAUSED THAT

This document and my message to `dev-2` both reasoned about the decline token
`no_shared_content`. *** **IT HAS BEEN RETIRED AT THE AUTHORITY.** *** At blob
`28a076f9` lines 1267-1282 the branch now returns:

    _decline("speed_relation_suspected" if monotone else "median_fidelity_below_floor", ...)

and the comment above it records the reasoning, including:

> *"`no_shared_content` RETIRED: A CLAIM ABOUT THE WORLD FROM AN INSTRUMENT THAT
> CAN ONLY SPEAK ABOUT ITS OWN MEASURABILITY ... NOISE IS NOT MONOTONE — THE GUARD
> FOR THE SPEED FAMILY IS DEFEATED BY THE SPEED RELATION BEING LARGE ENOUGH TO
> DESTROY THE MEASUREMENT THAT WOULD DETECT IT. (mechanism `vmsam-arch-aide`;
> constructed pair `vmsam-dev-2`.)"*

**The mechanism I proposed was adopted into the code verbatim, and the row now
states what it does not know instead of asserting the files are unrelated.**

*** **SO MY OWN CORRECT FINDING IS WHAT MADE MY DOCUMENT STALE, AND I WENT ON
CITING THE RETIRED TOKEN FOR THREE MORE MESSAGES BECAUSE I NEVER RE-FETCHED THE
FILE I HAD JUST CAUSED A CHANGE IN. A seat is least likely to re-read the file it
believes it understands, and MOST likely to have changed it.** ***

### What survives unchanged, re-derived at `28a076f9`

    CHROMAPRINT_HOP_SECONDS = 4096/3/11025      :177   value identical
    "124-142 ms depending on window length"     :121   identical
    41.7 ms = ONE FRAME @ 24.000                :212   identical
    PROBE_WINDOW_SECONDS 60.0 / STEP 40.0       :186-187
    "no rate correction"                        :199   identical
    second_correlation / correlate              :659 / :667   (were :466 / :475)

**Every arithmetic conclusion in this document stands. Only the coordinates moved,
and one token died.**

### And a caveat at the authority that qualifies my own derivation

Line **221**, absent from the blob I first read:

> *"AND 41.7 ms IS A FLOOR ON WHAT HAS BEEN SEEN, NOT ON WHAT EXISTS."*

**I built "one point = 2.97 to 3.41 frames" on 41.7 ms as THE frame duration, which
is sound, but I presented it as the smallest step that CAN exist. The module says
it is the smallest CONFIRMED one.** The two-stage argument does not depend on it —
a smaller true step makes the case for the frame stage stronger, not weaker — but
**the sentence I wrote claimed more than the source does.**
