# RULING 2026-09-15 — every artefact-derived claim carries its build digest
# Architect. Trigger: inv-pairing found one of "its 4 cases" carrying
# build=6fd445935aef — an artefact predating the patch under evaluation,
# swept into a retroactive tally that checked neither build nor timestamp.
# The audit that followed found NO check anywhere comparing an artefact's
# build to the deployed code. Queued for the next governance gesture; binding
# on every live seat since 2026-09-15 evening by message.

## The rule

**Every claim derived from a produced artefact (a .log.error, a merged file,
a REFUSED product, a verification row) carries the BUILD DIGEST of the
artefact it derives from.** The logs already print `build=` — reading it
costs a grep. Consequences, each naming its failure mode:

1. **A tally that mixes digests is not a tally — it is a mixture presented
   as one.** Counts, proportions, and before/after comparisons are computed
   PER DIGEST, or they are not computed. The campaign's own 17:49 signature
   aggregate (24/10/4/3/1) is the type specimen: era-blind across at least
   three deploy eras, demoted to indicative the moment this rule existed.
2. **A retroactive sweep filters by digest FIRST.** An artefact whose build
   is not the code under evaluation is evidence about THAT build only — it
   may inform history, never a current-code verdict.
3. **Investigator partitions carry the digest as a mandatory column.** A
   partition without it does not land in a report.
4. **CI's triage records the digest per row** at detection time — the era
   boundary becomes a property of every row instead of a narrative line in
   a RUNLOG.
5. **FINDINGS admits no count without its digest column.** The gatekeeper
   refuses the entry, not the finding.

## The per-module refinement (inv-norepair, 2026-09-15 late — in the rule, not a message)

**A `build=` stamp on a log line dates ONE MODULE, not the pipeline.** Measured:
one repair run logged `merge_video_chimeric.py` at `465aa391a12b` while the
same run's `change_point_locator.py` stamp sat a week older — consistent,
because module stamps are content hashes: an unchanged file keeps its digest
across deploys. Therefore:

- **The authoritative ERA KEY of a row is the DEPLOYED COMMIT SHA**
  (`VMSAM_GIT_COMMIT`, what `/health` reports) — per-run, injected at build,
  dating the whole pipeline.
- Per-module `build=` stamps are DIAGNOSTIC — they date a module's content,
  and a mismatch between a module stamp and the era key's expectation is
  itself a finding (a stale module in a fresh image).
- A row is era-clean under the COMMIT key only. A tally keyed on one
  module's stamp can mix pipelines and believe itself clean — the exact
  failure this ruling exists to prevent, one level deeper.

## Why a rule and not a habit

Two seats independently produced digest-blind counts in one evening (the
architect's aggregate; the cycles 6-10 retroactive tally). When independent
actors repeat an error the same day, the fix is mechanical recording, not
vigilance — AGENT.MD's closing line applies: mechanise the rules whose
failure is silent in the actor's own frame. A stale artefact looks exactly
like a current one from inside a grep; only the digest tells them apart.
