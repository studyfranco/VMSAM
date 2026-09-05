# ci-instruments — the measuring apparatus, saved

Everything here ran from `VMSAM_HELP_AI/ci/`, which **is not a git repository**: no
toplevel, no remote, no history. One copy, one machine. `vmsam-dev-sandbox` raised it as
a standing risk and this commit answers it.

## What this is

The instruments, not the evidence. `status.py` (one-command self-check),
`preserve_artefacts.sh` (the two-sweep preserver), `check_output.py` (duration and
attribution), and the checkers written when a specific failure justified one:
`check_degenerate_keys.py`, `check_empty_assertions.py`, `cross_language_fill.py`,
`check_pairing.py`, `census_population.py`, `publish_families.py`, `settle_sizes.py`,
`save_state_scan.py`.

## What is deliberately NOT here, and why

- **`runs/`** — raw container logs. They carry real library paths as the whole point of
  their existence; the redacted copies beside them are what may be cited. Excluded, not
  forgotten: the evidence stays on the machine and only the apparatus travels.
- **`*.bundle`** — binary git bundles of earlier states. Already history somewhere.
- **The ledgers** (`ci-preservation-ledger.tsv`, `ci-declined-ledger.tsv`,
  `ci-artefact-families.tsv`, `ci-settled-sizes.tsv`) — they live in `/config/output`
  beside the artefacts they describe. **A ledger away from its artefacts is a claim
  nobody can check.**

## One change was needed to make this committable

`test_merge_plan_report.py` asserted that the redactor strips a library path — and used
**the real library root** as the input. The test that proves redaction works was the
file carrying the thing it strips. It now uses `/srv/SYNTHETIC-LIBRARY-ROOT`, which the
generic `r'/srv/[^\s\'"\]},]*'` pattern matches identically, so the test keeps every bit
of its power. Verified after the change: **18 passed, 1 failed**, and the one failure is
the standing deliberate red (`title words survive in tail -- do not cite it`).

`save_state_scan.py` is the gate that found it. It reports **file and count, never the
matching text** — a scanner that prints its findings publishes the thing it exists to
protect, into a terminal and a transcript. It checks two independent sources, because
they fail differently: `redact.py`'s patterns, and the credential file's actual values
read at scan time and never printed. **A pattern cannot know a secret; only the secret
knows itself.**

## Standing limits of what is here

- `status.py` reads its run denominator from the live runner's `ONLY_IDS`, never from a
  literal. It printed `of 100` for hours because I took the number from the run FILE'S
  NAME. **The name records what was asked for; the environment records what was given.**
- Counts over a zone must filter to the artefact extension. My own tables live in the
  directories they describe, and a discrepancy of exactly the number of tables in a
  directory is that, not a lost artefact.
- Artefacts are hard-linked eagerly, on purpose, because the runner deletes its outputs.
  A link taken mid-write records a size that was true at that instant and is not the
  artefact's size. `settle_sizes.py` is the second reading; the link is not the problem.
