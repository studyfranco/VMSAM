# MEASURING — durable rules only. The full campaign-2 journal, whose `§N` numbering (~1,005 references across the tree) is cited everywhere, lives at `CAMPAIGNS/02-dev-chimeric-resample/MEASURING.MD` — bare `§N` citations resolve THERE, never here.

## Opportunity count `m`
Report `m` by counting the inputs on which the check COULD have gone the other way — never by inferring it from the outcome. Where that count is not computable, do not assert `m`; report UNTESTED. A quiet cell with `m = 0` is an untested cell, not a healthy one. (§78, §56)

## Denominator beside the verdict
Print `0 of N`, never a bare `0` — a zero prints the same whether it means "looked, found nothing" or "nothing to look at". A sweep without a denominator certifies what it never examined; a sweep that silently shrinks its own denominator is worse. (§196, §35, §62)

## Positive control before any null
A detector that says 0 must first prove it can say >0, on the axis the defect lives on — nothing in a control says which axis it exercises. A null from an instrument never shown to fire is not evidence. (§31, §41, §25)

## RELAYED is a marker, not a weakness
A relayed claim travels with its author's hedge and the label RELAYED — transmission is mediation, and verbatim does not fix a relay. A join between two relayed figures is weaker than either; resolve by direct address, not another hop. (§93, §85, §162 preamble)

## "I could not measure" ≠ "absent"
`None`/`unknown` is a property of the instrument, not of the material — never map it to a clean value. Before filing UNVERIFIABLE, name which observable the null was taken on. (§43, §110, §91, §156)

## Cite the blob, not the tip
Cite by content (`<blob>:<path>`), not by role (branch, HEAD, "latest") — a tip moves under the citation. Corollary: a blob is silent about whether any branch still carries it; a citation is only as reachable as the ref that carries it. (§33, §33b, §106)

## A recorded number does not re-run; a predicate does
`N files merged` ages and cannot be checked; `at T, <ref>:<path> = X while <other> = Y` is checkable forever. Record the refs, the instant, and the command — only the predicate is falsifiable. (§108, §27)

## Exit codes are answers, not errors — and pipes eat them
`grep -c` prints its count on stdout and its outcome in the exit status; `|| echo 0` fires on the honest zero and corrupts it. Never guard a count with `||`; every `|| true` in an instrument voids every zero it ever produced. (§169, §193)

## Printing a verdict is not adjudicating it
A score the harness prints but never acts on is a report, not a gate. If you want a guard, demand "exit non-zero", not "print" — an instruction naming an output will be satisfied by an output. (§162, §161)

## Instrument before alarm
Every broken instrument produces a PLAUSIBLE number; none is caught by looking at the number. Audit the instrument (and show the disagreement to someone who would notice) before believing — or relaying — its alarm. A detector's own alarms are not evidence of normality. (§191, §25)

## Retraction: grep the source tree, not only the documents
A carrier search is blind in the shape of the author's model of where claims live — decisions, docstrings, and notes miss source comments and constants. Grep the whole tree; a retraction must be local to every point a reader can arrive at. (§58; campaign-2 AGENT_NOTES retraction)

## Line numbers die in append-only files
In a newest-first append-only file, every append invalidates every line number below it — the citation is wrong the instant it is saved. Cite by the command that finds the text. A line number anywhere is a coordinate in a tree, not a property of a defect. (§27; campaign-2 AGENT_NOTES retraction)

## A threshold calibrated to a claimed observation inherits its fiction
Setting a cutoff just below the smallest CLAIMED observation converts a sampling limit into an instrument limit and then hides the conversion: the next smaller real event is invisible and "confirms" the threshold. Derive limits from measured floors, not from the smallest thing so far reported. (campaign-2 AGENT_NOTES retraction; §149)

## Compare the name set, not the count
Two changes cancel and the counts match; an equal count is compatible with total turnover. Diff identities, not totals. (§64, §137)

## Reproducing a result is not verifying it
Re-deriving a figure from the same source — or agreement between reads that share a source — adds nothing. Independence is a property of failure modes, not of mechanisms. (§49b, §112, §24)

## A correction is an artefact — verify it like the claim it corrects
Corrections propagate faster than their refutations and are accepted on credibility, not evidence; a correction that deletes must clear a higher bar than one that adds. (§48, §49, §49c)

## Name the quantity beside the number
State what was measured and the bridge to the quantity the claim is about — a figure true of a configuration travels as a measurement of the effect, and one phrase can cover five different quantities, wrong a different way each time. An integer without its counting rule (unit + boundary) is not comparable. (§97, §46, §167, §196)

## A pass indistinguishable from "nothing examined" is not a pass
A gate satisfied vacuously — no pairs near the bar, no input that could fail — certifies nothing. Put the vacuity check in the output: print two quantities that must agree next to each other, from different sources. (§28, §84, §196)

## Zeros need their path proven live
A zero from a dead sink and a zero from a silent path are the same observation; `flips = 0` is not `monotone`; a guard that has never fired is indistinguishable from a working one — prove it can fire, once, on purpose. (§109, §116, §111, §163)

## Every denominator has an owner — usually unknowingly
Every denominator that failed was owned by whoever happened to run the command, and none of them knew. Before dispatching on a population figure, name who owns its definition; a seat whose substrate cannot see the population must say so rather than estimate. (§194, §196, §178)
