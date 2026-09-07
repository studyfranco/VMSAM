# 000 — Tag write-access grants (ledger)

**Never closed.** This file is the history of who was allowed to write where.

Only **one agent at a time** may write inside a given tagged zone, and **only
`vmsam-dev-2` writes in zone A**. New capability goes in its own module: each
agent owns its files and functions and touches nothing of another's. What is
shared is the *call site* inside a tag — so an agent that needs one **asks the
Lead**; it does not take it. A grant covers **one edit** and expires when that
edit is promoted or abandoned.

Zones that can be granted (`WRITE_ZONES.MD` §2), at `f0e7119`:

| id | file | lines | function | note |
| :--- | :--- | ---: | :--- | :--- |
| `A` | `src/mergeVideo.py` | 789–800 | `remove_not_compatible_video` | zone A, dev-2 only |
| `SNAP-CFR-1` | `src/mergeVideo.py` | 663–667 | `adjust_delay_to_frame`, before the CFR maths | |
| `SNAP-CFR-2` | `src/mergeVideo.py` | 674–678 | `adjust_delay_to_frame`, **after the `return`** | dead zone — restructure, do not write into it |
| `SNAP-VFR` | `src/mergeVideo.py` | 682–690 | `adjust_delay_to_frame`, the non-CFR `else` | the VFR `TODO` |
| `TOOLS-MOD` | `src/tools.py` | 389–393 | module level | shared helpers |
| `VMAF` | `src/video.py` | 616–685 | `get_best_quality_video` + the two fallbacks | |
| `PSNR` | `src/video.py` | 690–786 | `get_good_frame` | keep the `float()` coercion, `SPEC_ZONE_A.MD` §3 |

`keep_best_audio` is listed in `WRITE_ZONES.MD` §2 but **carries no marker** in the
tree — see `AGENT_NOTES.MD`, 2026-09-03. It is frozen until the owner rules.

## Ledger

| # | date | zone | agent | for | released |
| ---: | :--- | :--- | :--- | :--- | :--- |
| — | — | — | — | *(no grant issued yet)* | — |
