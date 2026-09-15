# RULING 2026-09-15 — decision routing, and the end of parked decisions
# Trigger: the pairing patch sat parked for hours on a "risk call" that was
# (a) already measurable and partly measured, and (b) not routed to anyone.
# Owner delegation, verbatim intent: "Tu devrais pouvoir débloquer ce genre de
# chose. Investigue et solutionne le blocage pour qu'il ne se reproduise plus."

## Who rules what — three classes, no fourth

1. **OWNER ONLY**: his frozen invariants and their exceptions; governance
   documents (per-occasion authorisation, "always ask me next time"); privacy
   boundaries; the end-condition's semantics; anything that could damage the
   library or publish data.
2. **ARCHITECT**: design trade-offs, RISK ACCEPTANCE within the campaign's
   standing rules, instrument calibrations and their interim placeholders,
   contract semantics, zone questions inside src/. Duty attached: every such
   ruling is reported to the owner with its grounds, hedges, and refutation
   condition — he overrules at will, but he is never the bottleneck.
3. **LEAD**: sequencing, seats, commit windows, review standards, grants.

## The anti-parking rule (same disease as the CI stalls, decision-shaped)

**No decision may sit parked without all three of: a NAMED decider (one of
the three classes above), a stated DEFAULT if unruled, and a date.** A parked
decision missing any of the three is a stall, and any seat that notices one
reports it as such. "Waiting on a risk call" with no named decider is how a
finished, measured patch sat idle while the campaign asked why nothing moved.

## Applied immediately — the pairing patch (class 2, ruled today)

LANDS. Grounds: (1) the calibrated min-of-two statistic and the 0.85 bar are
untouched; (2) forensic's two-arm measurement: the hardest AVAILABLE negative
class (commentary — a more plausible negative than cross-language; Lead's
correction 2026-09-15: available, not hardest possible — commentary maxes at
0.665 while the original calibration's hardest negative reached 0.8477,
0.0023 from the bar, from a population recorded as not recoverable; the
0.80-0.85 band is untested by anything in existence) rejects 14/14 under
both geometries with >=0.19 margin, controls accept 11/11;
(3) cross-language material refuses at an earlier gate on every available
proxy. Hedges carried, not softened: the commentary evidence is one show /
11 episodes; forensic's lab instrument does not reliably reproduce values in
the near-bar band (its commentary conclusion is unaffected, 0.19+ away).
Compensating tripwire, standing: the sweep logs both probe scores on every
acceptance; **any ACCEPT whose minimum falls in [0.85, 0.87] is flagged by
CI triage as a forensic review row** — the near-bar band stays under watch at
population scale, which is the measurement the one-show bound lacks.
Refutation condition: one measured cross-language or commentary acceptance
under the patched geometry reopens the ruling and parks the patch class.


## Corpus go/no-go (#2) — ruled the same day, class 2

**BUILD IT — with the hard-negative band as a stated requirement.** The spec
(forensic, SPEC_video_bearing_ground_truth.md) is extended by what the
commentary measurement taught: the suite MUST include negatives ENGINEERED
into the 0.80-0.85 fidelity band, not only clean accept/reject cases — a
corpus of easy negatives would reproduce the gap we already have and fail
the one thing it is most needed for. Grounds for building over
tripwire-forever: the tripwire only ever observes ACCEPTS in production —
it can flag a wrong acceptance but can never demonstrate the reject arm on
the band that matters; engineered band cases fire that arm deliberately,
and 016's boundary calibration needs the same material. Cost measured and
small (~1-1.5 GB per suite against 16 GB standing). Named decider: the
Architect. Default executed: build. Date: 2026-09-15. **Stated fallback,
decided now so it never parks:** if fidelity proves not engineerable into
the target band after honest attempts (a real possibility — fidelity is a
measured outcome, not a set parameter), the fallback IS the tripwire as
permanent mitigation, recorded as such — a bounded risk, not a closed one.