# Lessons extracted from the master governance documents

Extracted 2026-09-14, during the compression of `AGENT.MD` and `WRITE_ZONES.MD`
after their reimport from `master` (audit of the diff `fe0c34e..master`). The
**principles stayed in those two files**; this file holds the measured narratives,
numbers and attributions that backed them. Consult it when a compressed rule needs
its evidence — none of this is instruction, all of it was paid for.

Section numbers below are the ones the governance files cite.

---

## §1 — The line budget that taught the wrong lesson (from AGENT.MD, header)

`AGENT.MD` once declared a 450-line budget and stood at 654 — 45 % over and
unenforced. A ceiling nobody honours teaches that ceilings are decorative. The
number also caused harm directly: **four failures in a single day were rules whose
mechanism was written down and whose consequence was not** — squeezed to fit, each
rule kept its mechanism and dropped what goes wrong when it is broken, which is
exactly the part that makes a reader act on it.

## §2 — A target that shares an input with the code under test (from AGENT.MD, Target-first §1)

Campaign 2 met **every clause of `SPEC_ZONE_A.MD` §2 on a real file and proved
nothing**: the hand-built target came from the same plan as the splice, so target
and output could not disagree whatever the code did. The claim was refused on its
own result rather than left to stand. The check that survives compression: state
what the target shares with the code under test *before* believing any agreement —
that is verifiable in advance, which "build it independently" is not.

## §3 — The dispatch with no refutation in it (from AGENT.MD, Target-first)

The target-first clause addresses whoever does the work, so a dispatcher satisfied
every word of it and still handed out a task with **no refutation in it** — and the
receiving agent supplied its own, covering for the dispatch. That is why the rule
now binds the dispatcher explicitly: a task without a stated refutation is a task
nobody can lose.

## §4 — Promotion cadence: the letter held, the intent did not (from AGENT.MD, tiers)

Two measured drifts, both with every individual rule obeyed:

- **Three promotions in twenty-six minutes**, each correctly one commit — and two
  of the three were halves of a single increment: the locator fixed, then sixteen
  minutes later the locator wired to its consumer. The granularity had simply moved
  from commits to promotions.
- **Campaign 2's first day put 79 commits on the branch, 73 of them documentation
  alone** — and an audit found that *nobody had broken a rule*. Commits are free in
  the tiers above, correctly; the promotion preserved history, also correctly; and
  the two composed into a branch where six real pieces of work sat among
  seventy-three entries about thinking. Two correct rules composing badly is not
  fixed by asking anyone to try harder — squashing at the promotion fixes it
  structurally.

## §5 — The transport that dropped the field: `parse_segments`, 28.88 ms (from AGENT.MD, Believing a result)

`parse_segments` rebuilt each segment from three named keys and discarded the rest,
including the per-stream offsets. **Emitter correct, consumer correct, integration
inert** — and the unit test passed throughout, because it handed the consumer a
hand-built dict and never crossed the transport.

The cost was **28.88 ms of misalignment**, which passed under four gates: below the
video frame, below `adjust_delay_to_frame`'s snapping, below `mkvmerge`'s integer
milliseconds, and below the repair's own 100 ms tolerance. Only a per-track check
saw it. It would have shipped in silence — hence the rule: a field whitelist in a
transport is a defect by construction; carry the whole structure.

## §6 — Which rules deserve to be mechanical (extracted section, from AGENT.MD)

The rules that survived are the ones whose violation produces an **immediate,
local, unambiguous failure in the actor's own frame**. The checker refuses a commit
and you feel it at once — it is the only rule nobody has broken. A waiter that
never returns is felt six hours later, which is why it needed a check rather than a
sentence. A report written but not sent produces **nothing at all**: no error, no
delay, no symptom in the writer's world, which is why no wording fixes it.

So: **mechanise the rules whose failure is silent in the actor's own frame — not
the ones that matter most, the ones the actor cannot feel going wrong.** Everything
else can be a sentence.

## §7 — Capability flags: the measured versions (from WRITE_ZONES.MD §4)

- Campaign 2 measured its own version of the flag defect without naming it:
  *"zone A is measurably inert with its flag off"* is a true statement about work
  that reaches no library.
- The gate-is-the-measurement example in full: **error id 237 was refused on a
  median fidelity of 0.576 with 18 sign flips**, not on a flag — a capability
  declining on its own measurements, which is a correct outcome on a service where
  nobody is present to flip anything.

## §8 — The sub-option ruling and who found it (from WRITE_ZONES.MD §4)

The original ruling named a capability and its flag; it said nothing about the
options *inside* a capability whose flag is already gone — and every key removed
after the two headline ones was of that kind. **`vmsam-dev-2` applied the ruling's
reason one level down three times while the document merely failed to contradict
it**: a plan supplied by a human is the deferral in different clothes; a switch
disabling the only source makes the capability inert; a verifier you can turn off
is not a safety. Anything that can make the capability not act, or act unchecked,
is the same defect one level down.

## §9 — Why reading a diff by eye is not enough (from WRITE_ZONES.MD §5)

`git diff --cached -U0` shows changed lines, and a reader maps them onto the zones
of the file **as it now is**. Three holes follow:

- **A pure deletion has no new-side line at all** — removing a frozen line *above*
  a zone reads as "nothing added outside a zone" and passes. Removals have to be
  mapped onto the **old** blob's zones and additions onto the new one's, separately.
- **Moving an `END` marker down** swallows a line into a zone while the diff lands
  inside it and looks legal.
- **A fresh marker pair** grants a zone that §2 reserves to the owner — so any
  change in marker count must fail.

Attributions the compressed text no longer carries: the holes were found by the
campaign-2 lead against the very section its author wrote; the divergence between
the documented snippets and what agents actually run (the checker) was noticed by
`vmsam-ci`, not by the document; and check 4's old allowance for `frame_compare.py`
became false when `vmsam-dev-1` fixed the file — reported as a finding rather than
edited out by the agent that outgrew it, exactly as the freeze demands.

## §10 — The self-editable governance file (from AGENT.MD, document freeze)

Campaign 1's `AGENT.MD` carried the line *"You are expected to update this
`AGENT.MD` whenever you implement a new feature"* — it reached **1191 lines**, and
the campaign shipped more documentation than engine. That is the measurement behind
the freeze: an agent that may rewrite what binds it eventually rewrites itself out
of it.

## §11 — Provenance cannot be reconstructed (from AGENT.MD, the `fabricated` marker)

Campaign 1 measured the cost of not recording provenance at production time:
**zero of 110 measurements could be attributed to the code that produced them**,
and three separate tools reconstructed track provenance from emission order, from
track counts and from codec — three methods, three answers, no tiebreak.

## §12 — The target fitted after the fact (from AGENT.MD, Target-first)

A campaign-1 method was tuned until its numbers looked excellent — the tightest
spread of a whole benchmark, **0.008 ms** — while it correlated the **wrong
language** and was wrong by **21 ms**. A target built afterwards is fitted to
whatever came out, and then any result looks like a success.

## §13 — Five pieces of dead code, every suite green (from AGENT.MD, Believing a result)

Campaign 1 shipped five pieces of code — a guard, a refined measurement, a policy
branch, an order check, a subtitle repair — that were written, reviewed and
unit-tested green, **and that nothing ever called**. Every suite passed, because a
unit test calls the function directly, which is exactly what production did not do.

## §14 — `Resolved: 0`, the guard that never fired (from AGENT.MD, Believing a result)

Campaign 1's success metric sat in the third branch of a function nothing reached:
**315 rows exited earlier**, and `Resolved: 0` was read as *nothing qualifies* when
it meant *nothing reached the check*.

## §15 — The `cp` promotion that erased an owner change (from AGENT.MD, tiers)

Measured on this repository: a promotion by copy silently erased an owner change to
the same file, while the same promotion through git stopped on the conflict and
preserved it. Hence clones, never copies.

## §16 — The cross-file thread target (from AGENT.MD, logging)

`Thread(target=videoObj.get_mediadata)` is declared in `mergeVideo.py` while the
function lives in `video.py`; a per-file analysis of who writes `tools.logs` misses
it, and that mistake produced **9 leaks** before it was caught.
