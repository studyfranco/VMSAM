# ARCHITECT RULING — the boundary key is a master frame index

**F1, delegated by the owner 2026-09-15** — *"Ask to the architect. I will reply
if needed."* Authored by the Architect; landed by the Lead under the owner's
one-time governance authorisation of 2026-09-15 (`FINDINGS.MD`, "Governance
bypass"). **In force.** It supersedes the millisecond boundary convention of
`TASKS/005` §6, `TASKS/006` and `TASKS/007`, each of which now carries an
amendment pointing here.

## What is replaced

`TASKS/005 §6` delivers `{master_start_ms, master_end_ms, candidate_offset_ms}`
under `candidate_time_ms = master_time_ms + candidate_offset_ms`. `TASKS/006`
carries the same millisecond convention (§L27, L35), and `TASKS/007` consumes it
(L485) with a **100 ms tolerance** (L205, L399).

That convention is void. A millisecond boundary is not invariant under the
resample that the campaign order puts BEFORE the locate step, so every ms figure
produced on the pre-resample timeline is wrong by the rate factor. And 100 ms is
**~2.4 frames at 24000/1001**: no frame-accuracy claim can be built on it.

## The contract

    {
      "grid":                    {"num": <int>, "den": <int>},   # MANDATORY
      "master_start_frame":      <int>,      # inclusive
      "master_end_frame":        <int>,      # EXCLUSIVE — half-open [start, end)
      "candidate_offset_frames": <int>,      # on the SAME grid
      "similarity":              <float>,
      "margin":                  <float>,
      "derived_ms":              {...}       # OPTIONAL, derived, never authoritative
    }

Eight rules govern it.

1. **The key is the frame index.** Milliseconds may appear only inside
   `derived_ms`, computed from the index and the grid. No consumer may read a ms
   figure back as authority. A ms field that is not under `derived_ms` is a defect.

2. **The grid travels with the index, always.** `grid` is the master's exact
   rational rate (`r_frame_rate` num/den, or MediaInfo `FrameRate_Original`), two
   integers, never a float, never rounded. *A frame number without its grid is a
   count without its surface.* A payload missing `grid` is malformed, not
   defaulted.

3. **The interval is half-open, `[start, end)`.** Stated because off-by-one at a
   boundary is precisely the failure this ruling exists to remove. An
   implementation that treats `end` as inclusive is wrong by one frame, which is
   the entire tolerance the owner asked for.

4. **The grid is the MASTER's, in every case.** Including audio-only divergences
   (owner ruling 2026-09-14 §3: one reference clock for everything), and including
   pairs whose native rates differ — because the candidate is resampled to the
   master grid BEFORE the locate step. This is not a convention, it is what makes
   the order `resample → locate → splice` load-bearing.

5. **This makes the fps-equality gate hold by construction.**
   `get_cut_time.py:377` gates the pHash path on `abs(fps_main - fps_tgt) < 1e-6`
   and therefore skips the frame stage exactly in the chimera case. Under rule 4
   the candidate is already on the master grid when refinement runs, so the gate
   is satisfied rather than bypassed. Defect 2 of `ARCH_FRAME_ACCURATE.MD` is
   resolved by the contract, not by deleting the gate.

6. **An unknown grid is UNREPRESENTABLE, not merely forbidden.** `grid` is
   mandatory and has no default, so `get_cut_time.py:369`'s invented `25.0` cannot
   be expressed in this contract at all. Where the master's rate is not probeable
   as an exact rational, the producer **DECLINES**. Same for variable frame rate
   (defect 4): refuse, or resolve through `FrameRate_Original` and say so. The
   illegal state is removed from the type, which is stronger than a rule against it.

7. **`None`/absent means "I could not measure".** It never means "compatible",
   "zero", or "no difference". **"no difference found" is a distinct, first-class
   verdict value** with its own name, and a producer emitting it must have proven
   it can emit a located zone instead. Every verdict carries `similarity` AND
   `margin`. A pitch-based producer's refusal is `no_pitch_relation`, never
   `no_rate_relation`.

8. **Applying a delay is SHIFT, not ROUND** (settled by three independent routes,
   dev-3). This ruling changes the unit of the boundary; it does not reopen that.

## The consequence for `TASKS/006` — the dropped refiner comes back

`TASKS/006 L84-88` dropped the frame-level video refiner with the justification
"**An instrument with no consumer**". That justification is **voided by this
ruling**: a contract keyed on frame indices has no producer of frame-level
precision other than the refiner, and no validator of a boundary other than pHash.
The refiner is now the contract's consumer. It is reinstated, and `TASKS/006` is
to be rewritten rather than extended.

`TASKS/007`'s 100 ms tolerance is likewise replaced: boundary acceptance is the
50-frame pHash identity check at each cut (placeholder until `TASKS/016` rules the
window and floor), **not** a millisecond window.

## The one case this contract cannot express — a blocked need, not a guess

A master with **no video stream at all** has no master frame grid, so rule 4 has
nothing to anchor on. I am not inventing a fallback clock for it: that is exactly
the class of invention rule 6 exists to forbid.

**Blocked need — CLOSED by measurement (forensic, 2026-09-15): ZERO of 5956.**
Every media file in every one of the 315 resolvable destination folders was
probed for video streams — **over-inclusive by design, so that an empty result
answers the narrower question with no identity assumption anywhere**: the masters
are a subset of what was swept. Distribution: 4897 files with one video stream,
1034 with two, 25 with four, **none with zero**. A failed probe returns no output
rather than a spurious positive, so the zero is signal and not silence.

**So the contract is complete as written and no second clock is needed.** The
guard must still **DECLINE** on the case per rule 6 if it ever appears —
**rarity is not absence**, and the state stays unrepresentable rather than
defaulted.

*Coverage bound, stated by the measurer rather than discovered later: ids
694-699 postdate the mapping snapshot and were not swept; a live pull the same
day showed 317+ ids. It does not move the verdict — a growing error tree does
not make a video library start containing audio-only masters — but it is the
honest edge of what was checked.*

## What would refute this ruling

The owner's own invariant is the test: 240 frames at 24000/1001, duration
10.010 s; apply the PAL relation with `-c copy`; duration becomes 9.600 s and the
frame count stays 240. **Indices emitted under this contract must be identical
before and after that transform.** If any index moves, the contract is wrong, not
the media.
