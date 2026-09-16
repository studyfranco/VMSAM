# RULING 2026-09-15 (night) — the guards family: three ways a guard lies
# Architect, from the evening's three measured guard defects. Queued for the
# next gesture; binding by message now.

A guard protects only if all three hold: it FIRES, it checks BOTH SIDES, and
it checks THE RIGHT INPUT. The campaign measured all three failures in one day:

1. **The unfired guard** (AGENT.MD rule 3, long-standing): armed but never
   exercised — indistinguishable from a working one. Fire every guard both
   ways, with literals, before trusting it. (Measured today: the policy-drop
   line, later found fired 4+58 in production.)
2. **The half guard** (Lazarus, structural): `fill_short_by_ms` checked
   master-sourced pieces only — nothing examined the candidate side of the
   same two-sided operation. **Every guard on a two-sided operation names
   which sides it checks**; an extent check without its twin is half a guard,
   and its silence on the unchecked side reads as a pass.
3. **The wrong-input guard** (Lazarus, root cause — the most expensive form):
   `merge_video_chimeric.py:1661` takes the MAX duration across all audio
   tracks; the extraction bound at `:453` therefore tests the plan against
   the LONGEST track, not the track being spliced (E06: two `en` tracks —
   even language-aware selection picks wrong). **And the docstring names the
   exact hazard it fails to guard.** The Lead's sentence, kept verbatim: *"A
   guard written in full awareness of the hazard, reaching for the wrong
   number, is worse than no guard: the comment persuades the next reader the
   case is handled."* Rule: **a guard's check names, in its output or its
   log line, WHICH concrete input it tested** (which track, which stream,
   which file) — a guard that cannot say what it checked cannot be audited
   against what it should have checked.

Corollary on predictions, from the same trace (three pre-registered
predictions, all three wrong, truth found in one measurement): the reason to
state a prediction is not to be right — a wrong one gets killed in one
measurement instead of surviving as a plausible story.
