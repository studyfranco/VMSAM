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

**State:** the video instrument is built and validated at both ends. The blind-spot
rate this task asked for **cannot be computed** and is dropped, for a reason outside
my control. What replaced it is an independent check on `change_point_locator`.

### What was measured

Instruments, both in `VMSAM_HELP_AI/dev-1/lab/010/`, neither shipping:

| | |
| :--- | :--- |
| `vprobe.py` | pHash frame comparison at 20 positions, aimed by an audio offset |
| `vshift.py` | at a flagged position, scans the candidate +/-40 s for the delta that best matches |

Calibration `M9`: same content across a 2.25x scale change is <= 4 bits of 64,
unrelated frames of the same programme ~29. Threshold 12, fixed before either
control ran. Validated 3/3 on a constructed positive and 0/24 on a real negative.

**The result, on two files the deployed audio path called flat:**

| | video probe | `change_point_locator`, run blind to it | apart |
| :--- | :--- | :--- | ---: |
| **id 144** | +5.00 s over [75.7, 605.5]; delta 0 at 908.2, 1210.9 | step **-5008.01 ms**, T in [600, 700] s | **8 ms** |
| **id 151** | -5.00 s from 833.1 on; delta 0 at 302.9 | step **-5004.82 ms**, T in [760, 860] s | **5 ms** |

The video's boundary between matching and displaced positions falls inside the
locator's bracket in both files; its clean points fall before it. Video resolution
is 500 ms.

**pHash on decoded frames shares no code, no library, no signal and no failure mode
with a chromaprint correlator on PCM.** Every previous validation of the locator
shared its physics. This is the first ground truth for it that exists independently
of it.

### What was dropped, and why it is not my judgement call

**The blind-spot rate.** 10 files probed, 6 assessable after controls, 2 positive:
33.3 %, exact one-sided 95 % interval [6.3 %, 72.9 %], against a 0-10 % band I
published before the run. It refutes me upward and **I am not reporting it**, because
`vmsam-ci` retracted the `flat` labels the sample was drawn from: its scan searched
+/-3000 ms, 64 of 100 files have a true offset outside that span, and a correlator
whose peak lies outside its span saturates at the edge and returns the same wrong
value on every probe. Eligible-flat is zero, not the ~27 the artefact produced. The
denominator is void; the comparison inside each file is not.

### Three defects, each caught by a control that could fail

1. **No failure class in the verdict rule.** 2 of 10 files read median 21 and 24 bits
   of 64 — the probe was comparing unrelated frames at most positions because the
   audio offset aiming it was wrong. Shipped, the rule called that VIDEO STRUCTURE.
   **An independent instrument aimed by a dependent one is not independent**, and
   the median was the aiming check, present in the output and unread.
2. **Content-differs is not timeline-differs.** Two releases with different logo
   cards differ in video and share a timeline exactly. Only displacement counts.
3. **argmin over a tie set is not a measurement.** On static scenes many deltas reach
   distance 0. Reporting tie width makes those `undetermined`, an honest third
   answer. Every number above has `ties = 1`.

The controls killed the file predicted to be the headline (id 85) and saved two that
were not. That asymmetry is the evidence the rules were not fitted to the answer.

### What stays blocked

1. **Any rate over the corpus**, until `vmsam-ci`'s corrected scan yields a population
   that was actually measured. Its `step` population is more useful here than its
   `flat` one ever was.
2. **The `video.get_good_frame` PSNR half of this task** — needs a tag grant, unasked
   for. Two findings from reading it are unrun: the `>=` tie test advances the winner
   on ties, and the existing dev-log line already states the unanswered question.
3. **Detection floors are three different numbers and belong to their instruments,
   not to the corpus**: ci 25 ms by choice, this locator ~62 ms by the chromaprint
   quantum, the video probe 500 ms by its scan step. Below 25 ms nothing in this
   campaign can see anything.
