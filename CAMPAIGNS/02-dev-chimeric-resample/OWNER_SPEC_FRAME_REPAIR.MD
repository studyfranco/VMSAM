# OWNER SPECIFICATION — frame-accurate repair and chimeras
# Received DIRECTLY from the owner 2026-09-07, not by relay. Original French;
# he instructed that everything he says be TRANSLATED before reaching an agent.
# This file is the irrecoverable half: the verification is in SCANS.md, the
# SPEC ITSELF existed only in a transcript.

## THE REQUIREMENT
Do not replace the whole problematic stretch — REPAIR THE EXACT ZONE, TO THE FRAME.

## TWO INSTRUMENTS (neither reaches the other's resolution)
    fpcalc   locates the ZONE     1 fingerprint point ~ 3-4 frames
    pHash    decides the FRAME    src/frame_compare.py, DCT 8x8, 64 bits

## THREE CONSTRAINTS — all measured, none optional
1. WORK FROM THE RAW FINGERPRINT LIST, NOT FROM correlation().
   It averages per-point Hamming similarity into one number: a 2 s defect in a
   60 s window moves it 2% and the position is gone. Keeping the per-point vector
   locates the same defect to 27.7-30.8 s for an injection at 28.0-30.0 s, in ONE
   PASS. The averaging is why a descent looked necessary.
2. INDEX IN FRAME NUMBERS, NOT MILLISECONDS.
   PAL relation on 240 frames -> 240 frames, duration 10.010 s -> 9.600 s. The
   frame index is invariant under the resample; a millisecond boundary needs a
   factor and is wrong without it.
3. DO NOT ROUTE THROUGH cross_correlation.
   Below min_overlap it returns without a value and without a message, and that
   None has the shape of "nothing found". Both the 32-point floor and the silent
   failure belong to the layer the new method stops using.

WINDOW FLOOR: 30 s, as the main software bounds its cut-time search.
HARD WALL: 4.05 s. Margin sevenfold.

## HIS TWO DEMONSTRATIONS (his numbers)
### 1. The per-point vector localises the zone
Synthetic pink noise, 60 s, 463 fingerprint points, sizePoint 129.6 ms.
Defect injected by cutting volume 28.00 s -> 30.00 s.
    correlation() returns the mean ......... 0.9798 ; defect moves it 2.02%
      one value for 463 points; position lost
    what correlation() DISCARDS — per-point similarity
      24 contiguous points below threshold
      *** ZONE LOCALISED 27.73 s -> 30.84 s ***
    Three points margin each side = the instrument's own resolution.
    ONE 60 s PASS, NO DESCENT.

### 2. The frame index is invariant under resampling
Test pattern, 240 frames at 24000/1001, PAL relation applied with -c copy
(no frame re-encoded):
    origin  duration 10.010 s   frames 240
    PAL     duration  9.600 s   frames 240
Duration changes by ratio 1.042709; the FRAME COUNT DOES NOT.

## WHAT HE STATES AS NOT ESTABLISHED — do not soften
- Both demonstrations are SYNTHETIC.
- Real PAL and real NTSC from the error set: UNMEASURED. He says that is the dev
  seats' work.
- The unbounded 0 -> end cut he asked for is NOT DONE. Every figure is over a
  whole window.

## MY VERIFICATION AGAINST THE CODE (authority 0aa9e71)
    correlation() averages          CONFIRMED  audioCorrelation.py:53, :66
                                    covariance / float(len(listx))
    cross_correlation silent        CONFIRMED AND WORSE  :79 floor, :82 BARE
                                    return, :83 the raise IS COMMENTED OUT --
                                    the message was written and disabled
    min_overlap = 32                CONFIRMED  :29
    pHash instrument exists         CONFIRMED  src/frame_compare.py:2, :75
                                    dct(arr, norm='ortho', axis=0)

    *** ONE FIGURE DOES NOT VERIFY: his spec says 126.6 ms per fingerprint point;
    his demonstration says 129.6; 60000/463 = 129.59. A DIGIT TRANSPOSITION.
    At 129.6 ms a point is 3.24 frames at 25 fps, 3.11 at 24000/1001 -- so
    "3 to 4 frames" holds either way. Figure wrong, conclusion unaffected. ***
    *** AND sizePoint IS A PARAMETER of get_max_corr(:109), NOT A CONSTANT. The
    spec states it as a property of the instrument when its value arrives from
    the caller. A constant that is an argument is a claim about ONE CALL SITE and
    the spec does not name which. ***

## CURRENT TASK, IN FLIGHT AT THE RESET
*** "Transpose the architecture to obtain FRAME-ACCURATE CHIMERAS per my
proposal." *** Owner, directly, minutes before Doomsday Phase 0.
NOT STARTED beyond reading. What I had established:
  - the existing architecture doc is CAMPAIGNS/01-dev-AI/MERGE_VIDEO_ARCHITECTURE.MD,
    335 lines, 10 sections (entry points · execution flow · module state contract ·
    Stage C delay resolution · Stage A/B simple path · Stage D protected closure ·
    not_compatible_video_list · extracted legacy · defects observed · master/candidate
    repair prebuilds)
  - *** THREE MODULES ALREADY EXIST AND I HAD NOT READ THEM:
    src/merge_video_chimeric.py · src/merge_video_repair.py ·
    src/merge_video_resample.py *** — the chimera path may already be partly built.
    READ THESE FIRST; do not design what exists.
  - mergeVideo.py already carries adjust_delay_to_frame(:677) — a frame-rounding
    step exists on the delay path.

## MY SCOPE ON THIS TASK
I DO NOT EDIT src/. Charter. I produce ARCHITECTURE and VERIFICATION; the dev
seats implement. Reading src/ is in scope; writing it is not.

## DISCLOSURE CONDITION THAT BINDS ANY REAL-FILE WORK
The error set is 172-of-172 disclosure-bearing (Lead, measured). Localisation
output is a set of timestamps AGAINST A FILENAME. Shape and count, never content,
including in whatever the dev seats report back.

## ADDENDUM — owner, 2026-09-14, direct

The two instruments' division of labour, restated and extended:
chromaprint correlation DETECTS the divergences — at ~125 ms, which is enough
to TARGET the zone between master and candidate and avoid excess computation.
Then **pHash must VALIDATE that the 50 frames before and the 50 frames after
the cut are the same** on both sides. The zone comes from chromaprint; the
cut's frame and its validation come from pHash, over that 50-frame window.

### Note on the 50-frame figure (owner, 2026-09-15)

The 50 frames (~2 s) are an arbitrary placeholder. The owner does not know the
right window length, the minimum similarity, nor whether the window should
extend to the nearest frame change in both directions — he delegated those
numbers to measurement: TASKS/016.
