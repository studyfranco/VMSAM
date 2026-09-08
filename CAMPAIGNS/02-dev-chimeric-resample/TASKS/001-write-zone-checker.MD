# 001 — `check_write_zones.py`: one command that enforces `WRITE_ZONES.MD`

**Owner of the task:** Lead. **State:** delivered, kept open until it has caught
something it was not written for.

## Goal

One command, one exit status, that refuses: a write outside a tagged zone, an
edited governance document, a removed `config.ini` line, a file created outside
the open paths. `WRITE_ZONES.MD` §2 rule 2 — a zone is real only if it is checked.

## What changed

Nothing in the repository. The tool lives **beside** it, at
`VMSAM_HELP_AI/tools/check_write_zones.py` (`WRITE_ZONES.MD` §4: a harness is not
something a production run needs). Reasoning in `AGENT_NOTES.MD`, 2026-09-03.

## What was measured

| check | fired | shown able to answer the other way |
| :--- | :--- | :--- |
| line changed outside a zone (add / edit / **pure deletion**) | yes | yes |
| `END` marker moved down to swallow a line | yes | — |
| marker pair invented | yes | — |
| governance document edited, or a new file under `docs/` | yes | yes |
| frozen file modified or deleted | yes | yes (open file, both) |
| `config.ini` line removed, and value changed | yes | yes (addition) |
| file created outside the open paths (4 shapes) | yes | yes (2 shapes) |
| tree stops compiling beyond `frame_compare.py` | yes | yes |

`--self-test`: **27/27**. On the real tree at `f0e7119`: clean; insert at
`mergeVideo.py:1` refused; the same insert at `mergeVideo.py:795`, inside zone A,
accepted.

## What was dropped

The four shell snippets of §5 as the working procedure. Three of them are read by
eye, and the `-U0` zone check is wrong for a pure deletion: it compares changed
lines against the *new* file's zones, and a removal has no new-side line. Kept as
the specification, not as the gate.

## What stayed blocked

Two findings, both in `AGENT_NOTES.MD` 2026-09-03, both awaiting the owner:

1. `keep_best_audio` is listed as a tagged zone and carries no marker. Not
   blocking today — the `fabricated` branch is already written and inert.
2. `src/lib.rs`, `src/main.rs`, `Cargo.toml` are frozen by `WRITE_ZONES.MD` rule 1
   and editable by `AGENT.MD` § Coding Directives. Not blocking today.
