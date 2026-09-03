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

## What changed / measured / dropped / blocked
*(fill as you go)*
