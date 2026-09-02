# Agent notes — logbook

One entry per decision that a later reader would otherwise have to re-derive.
Newest first.

**What goes here:** an architectural choice and why the alternatives lost; a
measurement that changed a decision; a defect found and what it taught; a blocked
need and what was done instead.

**What does not:** progress reports, task status, anything regenerable from the
code or from `git log`. If it is a measured fact rather than a decision, it belongs
in a findings document.

**Format.** `## YYYY-MM-DD — one line saying what changed`, then the decision, then
*why the alternatives lost*. Numbers carry the command that produced them. Files are
cited by opaque id, never by media name (`AGENT.MD` → Privacy).

Entries are **append-only**. Superseding an earlier one means adding a new entry that
says so and why — never editing the old one. A record that can be rewritten is not a
record.

---

## 2026-09-02 — master prepared for campaign 2

`gestionar_show` imported from `dev-AI` and corrected: the dual-uvicorn split, a
fixed internal port (`tools.internal_api_port = 42085`), and `fusion.py` rewritten
from 1334 to 381 lines — the queue and the merge kept, the campaign instrumentation
dropped.

Twelve defects found and fixed while wiring it, none of which the compiler could
see: `get_git_commit` raising `NameError` on first call, three `tools.*` attributes
referenced but never defined, `_env_flag` used for a path (it returns a boolean),
nine `tools.logs` writes leaking into thread contexts through cross-file
`Thread(target=…)` declarations, `.get(0)` on a list, a `Decimal`/`float` mix that
made the no-forced-best-video path drop every candidate *silently*, a lost-master
window in the production swap, a pool never shut down, and a production failure
that left no trace anywhere.

Two hazards were closed structurally rather than by a fix: the episode lock, because
the integration loop and the fusion worker are separate processes that can write the
same episode; and `tools.load_merge_runtime_from_env()`, because Python 3.14
defaults to `forkserver` and a forkserver child sees only module defaults — which
would have put the two processes' locks in different trees, silently, while
`software={}` failed every ffmpeg call.

Write zones were declared and the first tags placed. The rule that governs them —
*a unit test cannot tell you that nothing calls the unit* — comes from campaign 1
shipping five pieces of code that were reviewed, tested green, and never called.
