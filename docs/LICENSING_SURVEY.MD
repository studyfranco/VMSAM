# Licensing survey — comparable synchronisation projects

Survey recorded during campaign 2 and extracted from `WRITE_ZONES.MD` on
2026-09-14. Licences are **as declared by each project at the time of the survey**
— re-verify at the source before relying on any row.

The binding rule lives in `WRITE_ZONES.MD` §4: **reimplement an idea; do not copy
the code.** This file only records what the surveyed projects declare.

| project | licence |
| :--- | :--- |
| `ffsubsync` | MIT |
| `tympanix/subsync` | Apache-2.0 |
| `alass`, `sc0ty/subsync` | **GPL-3.0** |
| `VideoDiff` | **GPL-2.0** |
| `video-offset-finder` | **none declared** |

Reading the table:

- The last row is the strictest, not the most permissive: **no licence means no
  permission to copy at all** by default.
- The two permissive licences (MIT, Apache-2.0) still require their notice to
  travel with any copied code.
- VMSAM carries no `LICENSE` file of its own, so what copied copyleft code
  (GPL-2.0/GPL-3.0) would oblige is an **owner question, not an agent's** — which
  is the whole reason to reimplement instead and never reach it.
