# 006 — Locate the change points inside a staircase

**Agent:** `vmsam-dev-1` (measurement). **Pairs with 007.** **State:** open.
**Campaign objective 2, the measurement half.**

## The signal

A window sequence reading

```
(0.92, -1, 125)  (0.93, 0, 0)  (0.90, -1, 125)
```

is **one content with a small insertion or deletion**: correlation stays high
throughout while the offset steps by whole points. One point = **125 ms**, the
chromaprint frame — every delay in this corpus is a multiple of it.

That is the shape. `first_delay_test` sees the disagreement across windows and
refuses the pair, which is correct: no single constant offset exists. It records
what it saw in `video_obj.delayFirstMethodAbort[best_video.filePath]` →
`[1|2, delays]`, filled at 12 points. **You do not need to re-run the correlation
that just failed** (`SPEC_ZONE_A.MD` §1).

## Goal

Build the method that turns that staircase into **change points**: where, in
master-timeline seconds, does the offset step, and by how much. Windows are 12,
their length is twice their spacing so consecutive ones overlap by half — a step
is bracketed to roughly one spacing by the window data alone. Getting from there
to a frame is the part that needs an instrument, 005's.

## Target-first — binding, `AGENT.MD` and `CAMPAIGN.MD` both

1. **Build one correct answer by hand first**, on one real error file, with
   whatever tools you like. The located change point, in seconds, and how you
   know. This exists **before** the code.
2. **State what result would refute you, before the run.** A prediction written
   afterwards is not a prediction.
3. Only then write code, and check it reproduces the hand answer.

## The rule that decides whether this counts

**Your work counts when dev-2 calls it, not when it passes a test.** Five pieces
of campaign-1 code were written, reviewed and unit-tested green and never called
by anything. Agree the interface with dev-2 **before** you build behind it, and
pair this with 007 before either is declared done.

## Ask the Lead for

- a tagged call site, if you need one — one agent at a time, one edit
- a container run, through `vmsam-ci` — never directly
## What changed

| file | what |
| :--- | :--- |
| `src/change_point_locator.py` | new module. Full-file probe grid, plateau grouping, clean-probe transition bracketing, per-stream offsets, bounds clamp. Reads no configuration |

Called by `vmsam-dev-2`'s repair. Returns numbers, never files.

## What was measured

**The sign convention holds.** `candidate_time_ms = master_time_ms + candidate_offset_ms`,
derived from `audioCorrelation`'s algebra and confirmed on a constructed 1000 ms
deletion (read -1000) and independently by dev-2 from the opposite direction.

**Plateau offsets are confirmed against outside instruments.** On error id 108,
-950.68 / -1453.83 / -951.47 against `vmsam-ci`'s and dev-2's -950.0 / -1453.9 /
-952.9 — agreement to ~1.5 ms.

**On error id 266, committed code, no hand plan or patch** (dev-2's run): repaired,
2 audio and 21 subtitle tracks, 0 declined, 0 failed, both tracks aligned to worst
1.12 ms and 1.00 ms over six probes at r 0.961-0.995, `VMSAM_FABRICATED=chimeric`
in the final mkv. Locator log: 35 probes over [0,1428] s, median fidelity 0.969,
zero sign flips, five plateaus.

**All branches fired both ways on real files**: id 237 declines (fidelity 0.576,
31 distinct, 18 flips); id 318 returns `constant`; id 147 returns two change
points. The drop-vs-decline branch fired with literals in both directions.

**Transitions: ONE independent check only.** dev-2's 8 s / 2 s profile of id 266
puts one transition in (628, 632) s, containing my (629.0, 631.5). One transition,
one instrument, one file, and a coarser step — favourable, **not confirmation**.

## What was dropped

- **The frame-level video refiner.** dev-2's repair fills the bracket from the
  master, correct by definition, at 0.49 % of an episode against 15 %. An
  instrument with no consumer.
- **`_align_and_find_gap`** (`frame_compare`): aggregates cost only where a
  reference frame matched, so inserted content is invisible to its argmax.
- **`get_cut_time.py`** as a starting point: divides `second_correlation`'s
  already-in-seconds offset by the frame rate at all four call sites.
- **Reading `delayFirstMethodAbort`**: those are residuals, and a fractional-hop
  re-extraction manufactures one-point steps.
- **The clean-edge estimator and the majority-model bisection**, both refuted.

## What is blocked

Nothing is blocked. Two known defects, both mine, neither fixed yet and both
deliberate:

1. **Transition brackets are unconfirmed at scale.** The only bracket numbers I
   produced before the current method came from a version later found 35x
   over-claimed. One outside check exists.
2. **Bound-only brackets overlap.** `_bracket_transition`'s fallback returns
   `region_end + PROBE_WINDOW_SECONDS`, which can push past the next run's first
   probe and swallow a plateau. On id 266 this dropped 2 segments and left 26.6 %
   of the timeline filled from the master — waste, not damage. Not fixed while
   dev-2 validates against this exact block; changing it mid-validation would make
   any disagreement unattributable.

Narrative, measurement logs and the full history of what was refuted:
`AGENT_NOTES.MD` and `VMSAM_HELP_AI/dev-1/lab/`.
