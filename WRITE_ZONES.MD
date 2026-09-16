# VMSAM — Write Zones

**What an agent may change, where, and how compliance is checked.**

This file is the authority. A campaign that violates it is not validated, whatever
its results. Read it before the first edit, not after a review.

This file says **what may be changed and why**. It says nothing about what to
build — that is `CAMPAIGNS/02-dev-chimeric-resample/SPEC_ZONE_A.MD` and the
campaign's own task documents.

Two rules govern everything below:

1. **Default is frozen.** A path that is not listed as open is closed. Silence is
   not permission.
2. **A zone is real only if it is checked.** Every rule here names the command
   that verifies it. A rule nobody can verify is a wish.

---

## 1. Frozen — no modification, ever

| path | why |
| :--- | :--- |
| `src/gestionar_show/` (all of it) | the API and the fusion worker: destructive paths, database writes, atomic file replacement |
| `src/main_gestionar_show.py` | the daemon, its loop and its two uvicorn instances |
| `src/audioCorrelation.py` | the delay measurement the whole pipeline trusts |
| `src/main.py`, `src/infinite_wrapper.py` | the other entry points |
| `src/tools/` | the database module |
| `src/video.py` | **except the two tagged functions below** — media handling, ffmpeg pools, MD5 threads: the part that touches disk and processes |
| `init.sh`, `run.sh` | production entry points |
| `Dockerfile` | the runtime image |
| `.github/`, `.forgejo/` | CI/CD |
| `AGENT.MD`, `WRITE_ZONES.MD`, `docs/` (all of it) | the rules that bind you and the standing rulings — modified **only by the Architect or the Lead Dev, with the owner's authorisation**. A wrong rule is a finding, see §6 |
| `CAMPAIGNS/` | the campaign records — `CAMPAIGN.MD`, `SPEC_ZONE_A.MD`, `FINDINGS.MD`, `MEASURING.MD` and the archives. Written by the **Architect and the Lead Dev only**; history is even less yours to rewrite than a rule. **In-progress task files live in THEIR AGENT'S folder** — `VMSAM_HELP_AI/<agent>/` — so a file's location is its ownership (owner, 2026-09-15: the old *"`TASKS/` at the repository root is yours"* rule is revoked, and the root `TASKS/` directory is closed and removed). The repository carries only finished, synthesized work, landed by governance gesture; no document here is open to an agent |

`src/gestionar_show/fusion.py` deserves its own line: it replaces the master file
in the library and deletes the source. It has been reviewed and corrected — atomic
`os.replace`, episode lock, pool lifecycle, silent-failure trace. **Do not touch it.**

Need something from a frozen file? Do not edit it: open a question for the owner
and keep working elsewhere.

## 2. Tagged zones — the only writable places in shared engine files

`src/mergeVideo.py`, `src/video.py` and `src/tools.py` are frozen **except** between these markers:

```python
"""
BEGIN: AGENT modification ok
"""
    ...your code...
"""
END: AGENT modification
"""
```

**One modification outside a tagged zone invalidates the campaign.** Not the
commit — the campaign. This is deliberate: the cost has to be visible before the
edit, not after.

Existing zones:

| file | function | purpose |
| :--- | :--- | :--- |
| `mergeVideo.py` | `remove_not_compatible_video` | **zone A** — repair of rejected files |
| `mergeVideo.py` | `adjust_delay_to_frame` | **the whole function is open**, not only its inner tags: frame snapping, and the VFR case that is still a TODO |
| `tools.py` | module level | shared helpers for the above |
| `video.py` | `get_best_quality_video` | the quality vote that names the reference — verify it really returns the best video |
| `video.py` | `get_good_frame` | PSNR frame scan; already carries a `float()` coercion fix, see below |
| `mergeVideo.py` | `keep_best_audio` | the `fabricated` rule — inert until the repair sets the key (`CAMPAIGNS/02-dev-chimeric-resample/SPEC_ZONE_A.MD` §4) |

Adding a new zone is an owner decision, not an agent one.

### Two tag forms — the string form is not always legal

The `"""BEGIN: AGENT modification ok"""` block is a **statement**, so it is illegal
wherever a statement is: between two `elif` branches, between `except` clauses,
between a `try` and its `finally`. Use the comment form there:

```python
            # BEGIN: AGENT modification ok
            elif ...
            # END: AGENT modification
```

Both forms carry the same text, so the checks below find either. `keep_best_audio`
uses the comment form for exactly this reason.

### A tag placed after a `return` is a dead zone

`adjust_delay_to_frame` currently contains one: the `return` on the CFR branch
short-circuits the tag that follows it, so anything written there never runs.
That is the campaign-1 defect — code that nothing calls — built into the tag
structure itself. It is harmless here only because the whole function is open,
so restructure it rather than writing inside the dead block.

Check before writing in any zone:

```sh
python3 - <<'PY'
import ast
src = open('src/mergeVideo.py').read()
for n in ast.walk(ast.parse(src)):
    body = getattr(n, 'body', None)
    if not isinstance(body, list):      # lambda/IfExp carry an expression here
        continue
    for i, st in enumerate(body[:-1]):
        if isinstance(st, ast.Return):
            rest = [x for x in body[i+1:]
                    if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant))]
            if rest:
                print("unreachable after return line", st.lineno, "->", [x.lineno for x in rest])
PY
```

### Two lines inside open zones that are not free to remove

`get_good_frame` opens with `calculated_delay = float(calculated_delay)`, and the
two frozen fallbacks in `video.py` re-raise `KeyboardInterrupt` / `SystemExit`
before returning their fabricated value. Both fix defects that produced **silent**
wrong behaviour. Removing either is allowed — understanding what it protected
first is not optional. See `CAMPAIGNS/02-dev-chimeric-resample/SPEC_ZONE_A.MD`.

## 3. Configuration

**`config.ini` — additions only.** New lines may be added and are then freely
editable by the agent that added them. **Existing variables are frozen**: not
their value, not their name, not their position. The merge rules encode measured
codec relationships; changing one silently changes every merge.

**`*.json` (`config.json`, `param_template.json`, `titles_subs_group.json`)** —
frozen unless there is an imperative need, which means an owner decision recorded
before the edit.

## 4. Open

`src/frame_compare.py`, `src/get_cut_time.py`, `src/folder_generator.py`,
`src/gestionar_movie/`, **every `src/merge_video_*.py`**, and **new runtime
modules** — code that ships and runs in production.

**Owner rulings on the `merge_video_*` family** (superseding the 2026-09-08
closure, which `RULING_20260908_REPAIR_RESAMPLE.MD` records as history):
`merge_video_chimeric.py` opened 2026-09-08; `merge_video_resample.py` opened
2026-09-15 (*"modifiable, améliorez comme vous voulez"*);
`merge_video_repair.py` opened in full 2026-09-15 (*"adaptez-le pour avoir vos
informations en mode tools.dev et permettant le traitement du maximum de
fichiers en erreur"*); and 2026-09-16 the owner generalised: *"les
merge_video_*.py sont ouverts"* — this line exists so the information is never
again lost to a stale closure note. The rule that generalises stays in force:
**a permission must rest on a property checkable at the tip — never on a
property of the history, which any rewrite can flatten.** The checker
(`OWNER_CLOSED_MODULES`, empty as of this writing) is the current record; when
this text and the checker disagree, the owner's most recent order is the
authority and the disagreement is a finding.

### What does not go in the repository

This repository holds **VMSAM's code**. Nothing else. Batch scripts, verification
harnesses, corpus sweeps, forensic reports, per-agent state files, task archives,
scratch analysis: `VMSAM_HELP_AI/`, beside the repository, never inside it.

Campaign 1 is the measurement. It left behind, under `src/`:

| directory | files | lines |
| :--- | ---: | ---: |
| `diagnostics/` | 624 | 88 394 |
| `tools_batch/` | 173 | 47 349 |
| `task_archive/` | 101 | 13 030 |
| **total** | **898** | **148 773** |

VMSAM itself is 7 225 lines of Python and 569 of Rust. The scaffolding outweighed
the program **nineteen to one**, and someone opening the repository could no longer
tell what VMSAM *is*.

The test, applied to every file you are about to create: **would this still matter
to someone running VMSAM in production?** If it exists to help *you* work — to
measure, to sweep, to remember, to report — it goes beside the repository.

### Files with no importer

Open like the rest, listed because their state is not obvious from reading them.

| file | state |
| :--- | :--- |
| `frame_compare.py` | pHash/DCT frame comparison. Compiles since campaign 2's repair (§5, check 4 tolerates no failure there). |
| `get_cut_time.py` | multi-range cut detection, drift, audio chimera. No importer; imports `frame_compare`. Homonym: `mergeVideo` used to define a *class* of the same name, now in `merge_video_unused.py`. |
| `merge_video_unused.py` | `find_a_cut_for_not_compatible` and the old `get_cut_time` class, extracted from `mergeVideo.py`. No importer. |

New capability goes in **its own module**, called from a tagged zone, with a test
that pins the call site.

**It must not be conditioned on a parameter.** VMSAM runs with nobody watching —
that is the premise the whole design rests on, and why a wrong automated decision
is destructive. A repair that waits for a human to switch it on is a feature that
will never run, on a service where nobody is there to run it.

**The gate is the measurement, not a setting.** A capability decides for itself,
from what it measured, whether it should act — and when it cannot establish that it
should, it **declines**, which is already a correct outcome here (error id 237 was
refused on its measurements, not on a flag). Measured cases:
`CAMPAIGNS/02-dev-chimeric-resample/LESSONS_FROM_MASTER_DOCS.MD` §7.

**Reimplement an idea; do not copy the code.** Comparable projects exist and reading
them is encouraged, but an idea described in a README and implemented yourself is
unencumbered, and copied lines are not. **If you do copy anything, say so in a
comment at the copy** — project, file, revision, licence: that records provenance,
it does not grant permission. The licences of the projects surveyed are in
`docs/LICENSING_SURVEY.MD`; several are copyleft or unlicensed, which is the whole
reason to reimplement instead and never reach the question.

**A sub-option of an unconditional capability is a condition on it.** Anything that
can make the capability **not act, or act unchecked** — a supplied plan, a source
switch, a disableable verifier — is the same defect one level down. Measured cases:
`LESSONS_FROM_MASTER_DOCS.MD` §8.

**A flag may exist while a capability is being built**, so that half-finished
behaviour does not reach a library — and **removing it is part of finishing**. It
belongs in `config.ini`, never in the environment: `run.sh` passes no `--config` and
the image bakes its variables into the frozen `Dockerfile`, so a new `VMSAM_<NAME>`
is absent at runtime and cannot be turned on at all. Campaign 1 measured the
consequence — **the code read 18 `VMSAM_` names and the image set two** — so every
feature built to a ruling that day was unreachable by construction. `config.ini`
additions are allowed by §3; a scaffolding flag left behind at the end is not.

**A removed `config.ini` line is judged against `origin/master`**: a line master
carries stays frozen, a line this campaign added is this campaign's to take back out.
That satisfies §3 and §4 both rather than overriding one — found while removing the
flags, and it is why the removal needed no ruling.

## 5. Verification

**Run the write-zone checker** — `VMSAM_HELP_AI/tools/check_write_zones.py`, one
command with one exit status, and **exit 2 means it could not run, which is not a
pass**. It is the authority; the snippets below are what it replaced, kept because
knowing what they missed is the point.

**Reading a diff by eye is not enough — the checker covers holes the eye misses**:
pure deletions of frozen lines, moved `END` markers, fresh marker pairs.
The full reasoning is in `CAMPAIGNS/02-dev-chimeric-resample/LESSONS_FROM_MASTER_DOCS.MD` §9.

**The snippets below are not the procedure and nobody runs them** — every agent runs
the checker instead. They are kept as the reasoning, not as the steps: an agent
following them literally would do weaker checks and believe it had done the
campaign's.

Before every commit:

```sh
# 1. no write outside the open paths
git diff --cached --name-only

# 1b. no governance document rewritten
git diff --cached --name-only -- AGENT.MD WRITE_ZONES.MD docs/ CAMPAIGNS/   # must be empty

# 2. in the two tagged files, every changed line sits inside a zone
#    (BEGIN: AGENT modification ok ... END: AGENT modification)
git diff --cached -U0 -- src/mergeVideo.py src/tools.py

# 3. config.ini: additions only, no changed or removed line
git diff --cached -- src/config.ini | grep '^-' | grep -v '^---'   # must be empty

# 4. the tree still compiles -- cleanly, no allowance
python3 -m compileall -q src -x '__pycache__'
# 5. Rust, when you touched it
cargo check && cargo clippy && cargo fmt --check
```

Checks 1b and 3 must print nothing. **Check 4 must pass cleanly**: it used to be
allowed one failure on `frame_compare.py`, broken since `da1c191`, and campaign 2
repaired it — so any failure at all is now yours.

## 6. Reporting a blocked need

A frozen file that stands in your way is a finding, not an obstacle to route
around. Record what you wanted to change, why, and what you did instead. A
campaign that ends with three well-argued blocked needs is worth more than one
that ends with three undeclared edits.
