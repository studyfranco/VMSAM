# RULING 2026-09-15 — measurement retention, and its floor
# Architect. Companion to "None means could not measure". Queued for the next
# authorised governance gesture; binding on the retention patch already via the
# Lead's dispatch (message trail, 2026-09-15).

## The finding that forces it (dev-frame, id-12 pair, opaque pair id e39737967605)

Detection worked: eight delay runs correctly separated, including the settling
head. The per-segment usability filter then dropped six of eight BEFORE any
comparison ran — so nothing ever compared −567 ms against −1412 ms, and the
verdict "no divergence found" was computed over survivors presented as the
whole. One spurious probe at t=520 s split a matching plateau, its transition
bracketed to nothing, the segment was dropped, and **140 seconds of MATCHING
content were filled from the master**.

> "It didn't fail measuring. It failed keeping what it measured." — dev-frame

## The invariant

**A measurement, once made, is never silently discarded.** For every instrument
in this pipeline, present and future:

1. A filter that drops a measured run, segment, or probe records the drop **per
   item, with a named reason**, in the same structured stream as the
   measurements (`dropped: reason=... span=...` beside `declined: reason=...`).
2. A comparison or verdict stage either **runs on ALL measured items before
   usability filtering**, or **declines explicitly with the drop-list
   attached**. It never computes over survivors and presents the result as
   complete — that is the id-12 failure shape, and it is a verdict defect even
   when the verdict happens to be right.
3. Usability filtering is legitimate — unusable segments are real. Its two
   defects here were POSITION (before comparison) and SILENCE (unnamed drops).
   A conforming filter runs after the verdict stage has seen everything, and
   leaves a trace a reader can audit.

4. **The mirror image (Lead, caught at the gate 2026-09-15): emitting the
   container without the contents is the same defect one notch smaller.** The
   standalone retention split left `pairing_position_scores` read-and-emitted
   while its producer went out with the pairing revert — every decline line
   would have carried `=None`, and a reader concludes "could not be measured"
   when the scores were measured and simply no longer recorded. `None` means
   "I could not measure", never "the producer was reverted in another patch".
   When a patch removes a producer, it removes the field's emission in the
   same diff — a field that can only ever read None is a lie wearing the
   None discipline as a costume.

## The floor (Lead + forensic measurement, folded in at the Lead's request)

**Retaining a measurement cannot recover resolution the instrument never had.**
Id 12's head structure spans roughly −640 → −570/−590 ms across ~158 s: a
spread of ~70 ms, on a file whose chromaprint quantum is 129 ms (its own log).
The "settling head" of −642/−588/−567 was real structure, **definitionally
unresolvable by the audio instrument** — not noise, not drift artefact, not a
retention loss. A seat reading rule 1 must not conclude that keeping every
probe licenses discrimination finer than one quantum: below the quantum, the
honest outputs are "structure present, sub-quantum" and a handoff to the
instrument that can resolve it.

That handoff is not hypothetical. 70 ms is ~1.7 video frames — invisible to
fpcalc at 129 ms, squarely within pHash's per-frame resolution. **This is the
first measured case on real media demonstrating the owner's two-instrument
division** (fpcalc targets the zone, pHash decides the frame,
OWNER_SPEC_FRAME_REPAIR.MD): the spec asserted it; id 12 demonstrates it.

## The acceptance test of the retention patch (pre-registered, before any code)

**The owner has reaffirmed the requirement on this exact case (2026-09-15,
verbatim intent):** the t=520 chain — one noise probe splits a matching
plateau, the segment is dropped, 140 s of matching content are master-filled —
is *"typiquement le problème"*: **the script takes from the master ONLY the
problematic part, TO THE FRAME.** This is OWNER_SPEC_FRAME_REPAIR's first line
applied to the fill path, and it makes the acceptance criterion sharper than a
percentage:

1. After the patch, every master-fill span in the plan is bounded in **master
   frame numbers** (F1 contract, grid attached, half-open) and covers ONLY a
   zone where divergence is DEMONSTRATED. And the owner's correction
   (2026-09-15) names the principle that governs how such a zone is found:
   **INDICATION IS NOT LOCALISATION.** A coarse signal — a variance outlier,
   a dropped segment, a low-similarity window, the 140 s at t≈480-620 — is
   never a repair boundary: it defines a SEARCH INTERVAL. Inside that
   interval the fine instruments (the raw per-point vector, then pHash at the
   frame) localise the actual divergence — perhaps 3 seconds inside 140 — and
   **the repair covers only what was localised, to the frame**. The rest of
   the search interval stays candidate content; if the search finds nothing,
   the verdict for that interval is "no difference found" and ALL of it stays
   candidate. Repairing at the granularity of the indicator is the same
   defect as filling on absent evidence — both replace content nobody proved
   divergent.

   **And once localised, the repair's DIRECTION follows the sign (owner,
   2026-09-15): the master is the reference for additions.** Candidate is
   MISSING content the master has → ADD it from the master — exactly the
   localised span, no more. Candidate has EXTRA content the master lacks →
   CUT it — exactly the localised span, no more. Both operations produce
   frame-indexed boundaries (F1) and every one of them is a repair boundary
   carrying the pHash check. Add-from-master and cut-from-candidate are the
   only two repair verbs at a localised divergence; both adapt the candidate
   onto the master's timeline, never the reverse.

   **Id 12 is the grammar's worked example, measured (2026-09-15):** its
   remaining 17.0 % master fill decomposes as 4.38 s genuine difference
   (0.37 %) plus 199.06 s localisation debt (16.61 %) — two 100 s bound-only
   brackets each known to contain a step the audio tier cannot narrow. It
   carries a genuine ADD at the head (candidate missing 591 ms + an −821 ms
   step) and a genuine CUT at the tail (+120.13 ms of extra candidate
   content at 1040-1140 s) — both verbs, one real file. **A ~46× fill
   reduction sits behind one unwired instrument** (the frame tier,
   `FrameComparer`, no live caller). Hedge carried as stated by the Lead:
   the ~0.4 % floor assumes the frame tier narrows each bracket essentially
   to its step; the true floor is 4.38 s plus whatever boundary width
   TASKS/016 rules. 46× is the honest headline; 0.4 % is its optimistic
   end.
2. Every fill boundary is a repair boundary and passes the pHash check
   (placeholder 50 frames / threshold 12 until TASKS/016 rules).
3. The before/after fill measurement stands as the magnitude test: 007
   measured 21.9 / 31.6 / 22.4 % on real ids; prediction: a large share was
   retention loss, not gap. **Measure fill on id 12 (minimum) before and
   after; report both numbers whichever way they fall; do not tune toward the
   prediction.** Fill collapsing toward true gap size confirms; fill staying
   refutes the magnitude while leaving the invariant intact. Either answer
   counts; an unmeasured patch does not — and a patch that reduces fill by
   keeping mismeasured content would fail leg 2, which is why legs 1-2 come
   first.

**Why this outranks everything else open:** end-condition clause 2 (zones the
repair did not touch are bit-identical to their source) is unreachable while
matching content is being re-filled from the master. This defect sits on the
campaign's critical path, not on id 12's.

## Related, recorded separately (not part of this ruling)

- Forensic's peak-finder bug (missed 0.99 at −567 inside its own window):
  retracted finding, instrument corrected under guard-firing discipline — the
  fixed instrument must find that peak AND still reject a true negative.
- The sharper trap found in the same pass: raw-waveform correlation peaks only
  a few ms wide on this pair — 0.9899 at exactly −567.0 collapses to 0.18 at
  −570. **A correct instrument produces confident nonsense at the wrong
  sampling density.** Distinct failure class; owns its own FINDINGS entry.
