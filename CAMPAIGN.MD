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

Measured on campaign 1's corpus of 315 diagnosed error files.

| family | share | shape of the failure |
| :--- | ---: | :--- |
| different cuts | **57 %** | the offset is piecewise-constant — a staircase. Content one side has and the other does not |
| trimmed edges | 18 % | the same, with a single step at one end |
| speed mismatch | 11 % | PAL 25 fps against 23.976: the offset drifts linearly, ~1 s per half-hour. **The method is already ruled on** — `docs/AUDIO_SPEED_POLICY.MD` |
| weak correlation | 10 % | heavy re-encodes, differently mixed dubs, music-only stretches: no confident peak |
| no shared language | 2 % | no audio language in common, or the tags lie |

**The largest family is not speed.** Over 79 measured files, **zero** showed a clean
slope near a constant (`CAMPAIGNS/01-dev-AI/FINDINGS.MD` §15). A repair that only handles speed
addresses a tenth of the problem.

Two files out of 315 were proven genuinely unrepairable: one truncated at source,
one encrypted. The ceiling is far away.

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

