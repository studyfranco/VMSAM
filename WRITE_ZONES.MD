# VMSAM — Write Zones

**What an agent may change, where, and how compliance is checked.**

This file is the authority. A campaign that violates it is not validated, whatever
its results. Read it before the first edit, not after a review.

This file says **what may be changed and why**. It says nothing about what to
build — that is `SPEC_ZONE_A.MD` and the campaign's own task documents.

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
| `AGENT.MD`, `WRITE_ZONES.MD`, `CAMPAIGN.MD`, `FILE_MAP.MD`, `SPEC_ZONE_A.MD`, `*_DEV_AI.MD` | the rules that bind you, and the campaign-1 archive. Not yours to edit — a wrong rule is a finding, see §6. **`AGENT_NOTES.MD` and `TASKS/` are yours**, and no other document is frozen |

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
| `mergeVideo.py` | `keep_best_audio` | the `fabricated` rule — inert until the repair sets the key (`SPEC_ZONE_A.MD` §4) |

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
first is not optional. See `SPEC_ZONE_A.MD`.

## 3. Configuration

**`config.ini` — additions only.** New lines may be added and are then freely
editable by the agent that added them. **Existing variables are frozen**: not
their value, not their name, not their position. The merge rules encode measured
codec relationships; changing one silently changes every merge.

**`*.json` (`config.json`, `param_template.json`, `titles_subs_group.json`)** —
frozen unless there is an imperative need, which means an owner decision recorded
before the edit.

## 4. Open

`src/frame_compare.py`, `src/get_cut_time.py`, `src/merge_video_unused.py`,
`src/folder_generator.py`, `src/gestionar_movie/`, and **new runtime modules** —
code that ships and runs in production.

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
| `frame_compare.py` | pHash/DCT frame comparison. **Does not compile** — `frame_compare.py:109`, pre-existing since `da1c191`. |
| `get_cut_time.py` | multi-range cut detection, drift, audio chimera. No importer; imports `frame_compare`. Homonym: `mergeVideo` used to define a *class* of the same name, now in `merge_video_unused.py`. |
| `merge_video_unused.py` | `find_a_cut_for_not_compatible` and the old `get_cut_time` class, extracted from `mergeVideo.py`. No importer. |

New capability goes in **its own module**, called from a tagged zone, behind a
flag that is **off by default**, with a test that pins the call site.

## 5. Verification

Before every commit:

```sh
# 1. no write outside the open paths
git diff --cached --name-only

# 1b. no governance document rewritten
git diff --cached --name-only -- AGENT.MD WRITE_ZONES.MD CAMPAIGN.MD FILE_MAP.MD SPEC_ZONE_A.MD '*_DEV_AI.MD'   # must be empty

# 2. in the two tagged files, every changed line sits inside a zone
#    (BEGIN: AGENT modification ok ... END: AGENT modification)
git diff --cached -U0 -- src/mergeVideo.py src/tools.py

# 3. config.ini: additions only, no changed or removed line
git diff --cached -- src/config.ini | grep '^-' | grep -v '^---'   # must be empty

# 4. the tree still compiles
python3 -m compileall -q src -x '__pycache__'
```

Checks 1b and 3 must print nothing. Check 4 fails on `frame_compare.py` until someone
fixes it; that failure is expected and is the only one allowed.

## 6. Reporting a blocked need

A frozen file that stands in your way is a finding, not an obstacle to route
around. Record what you wanted to change, why, and what you did instead. A
campaign that ends with three well-argued blocked needs is worth more than one
that ends with three undeclared edits.
