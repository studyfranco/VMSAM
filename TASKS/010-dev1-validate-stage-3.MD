# 010 — Stage 3 has never been validated

**Agent:** `vmsam-dev-1` (measurement). **State:** open, lower priority than
005–008 — start it when one of those is waiting on someone else.

## Goal

`CAMPAIGN.MD` § Open questions: *nothing shows the PSNR scan locates the right
frame, nor how it behaves between sources of different resolution.* Stage 3 is
`video.get_good_frame` (PSNR over ±2 frames) and `video.get_best_quality_video`
(the VMAF vote that names the reference), and it runs **only when no reference is
imposed** — with a forced reference `compare_video.run` takes the `else` branch,
snaps the delay and stops.

Two questions, in this order:

1. **Does the PSNR scan find the right frame?** Construct a pair where you know
   the answer — take one source, produce a copy displaced by a known whole number
   of frames, and see whether the scan returns that number. Then repeat across a
   resolution change, which is the case nobody has looked at.
2. **Is it worth generalising to the forced-reference path?** Only if 1 holds.

A scene-detection tool is available and may be faster or better; the comparison
has not been made. Making it is in scope. Choosing between them without measuring
is not.

## The trap this task exists inside

Stage 3 is where campaign 1's success metric sat — in the third branch of a
function nothing reached, so `Resolved: 0` read as *nothing qualifies* when it
meant *nothing reached the check*. Before measuring what the scan answers,
**prove the scan runs at all** on the input you give it, and prove it can return
the other answer.

Both zones are `video.py`'s (`VMAF` 616–685, `PSNR` 690–786) and are **grantable,
one agent at a time** — ask the Lead. `get_good_frame` opens with
`calculated_delay = float(calculated_delay)`; removing it is allowed, understanding
what it protected first is not (`SPEC_ZONE_A.MD` §3).

## What changed / measured / dropped / blocked
*(fill as you go)*
