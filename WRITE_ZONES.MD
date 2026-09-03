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
| `AGENT.MD`, `WRITE_ZONES.MD`, `CAMPAIGN.MD`, `FILE_MAP.MD`, `SPEC_ZONE_A.MD`, `COMING_BACK.MD`, `COMMITTING.MD`, `TALKING.MD`, `MEASURING.MD`, `LEAD.MD`, `docs/`, `CAMPAIGNS/` | the rules that bind you, and the campaign archives — history is even less yours to rewrite than a rule. Not yours to edit — a wrong rule is a finding, see §6. **`AGENT_NOTES.MD` and `TASKS/` are yours**, and no other document is frozen |

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

**A module is yours; the schema it emits stops being yours the moment someone reads
it.** Open means you may add, change and remove freely — and that freedom ends at the
fields another agent's code consumes. Adding a field and removing it twenty minutes
later were both within one agent's rights, and both would have broken its consumer
(`vmsam-dev-1`, which reported it rather than discovering it downstream). **Say what
you are changing in an emitted schema, to whoever reads it, before you change it.**
That pair handled it by talking; a successor pair may not have the conversation.

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
| `frame_compare.py` | pHash/DCT frame comparison. Compiles; the `da1c191` breakage was repaired in campaign 2. |
| `get_cut_time.py` | multi-range cut detection, drift, audio chimera. No importer; imports `frame_compare`. Homonym: `mergeVideo` used to define a *class* of the same name, now in `merge_video_unused.py`. |
| `merge_video_unused.py` | `find_a_cut_for_not_compatible` and the old `get_cut_time` class, extracted from `mergeVideo.py`. No importer. |

New capability goes in **its own module**, called from a tagged zone, with a test
that pins the call site.

**It must not be conditioned on a parameter.** VMSAM runs with nobody watching —
that is the premise the whole design rests on, and it is why a wrong automated
decision is destructive. A repair that waits for a human to switch it on is not a
repair: it is a feature that will never run, on a service where nobody is there to
run it. Campaign 2 measured its own version of this without naming it — *"zone A is
measurably inert with its flag off"* is a true statement about work that reaches no
library.

**The gate is the measurement, not a setting.** A capability decides for itself, from
what it measured, whether it should act — and when it cannot establish that it
should, it **declines**, which is already a correct outcome here. That is the
difference between a decision procedure and a deferral to somebody who is not
present. The campaign has the example: error id 237 was refused on a median fidelity
of 0.576 with 18 sign flips, not on a flag.

**Reimplement an idea; do not copy the code.** Comparable projects exist and reading
them is encouraged — the search strategies are worth having. But an idea described in
a README and implemented yourself is unencumbered, and copied lines are not.

**If you do copy anything, say so in a comment at the copy**, naming the project, the
file, the revision, and its licence. That is required and it is not sufficient: a
comment records provenance, it does not grant permission.

Licences as declared by the projects surveyed:

| project | licence |
| :--- | :--- |
| `ffsubsync` | MIT |
| `tympanix/subsync` | Apache-2.0 |
| `alass`, `sc0ty/subsync` | **GPL-3.0** |
| `VideoDiff` | **GPL-2.0** |
| `video-offset-finder` | **none declared** |

The last row is the strictest, not the most permissive: **no licence means no
permission to copy at all** by default. The two permissive ones still require their
notice to travel with the code. And VMSAM carries no `LICENSE` file of its own, so
what copied copyleft code would oblige is an owner question, not an agent's — which
is the whole reason to reimplement instead and never reach it.

**A sub-option of an unconditional capability is a condition on it.** The ruling
above names a capability and its flag; it says nothing about the options *inside* a
capability whose flag is already gone, and every key removed after the two headline
ones was of that kind. Anything that can make the capability **not act, or act
unchecked**, is the same defect one level down — a plan supplied by a human is the
deferral in different clothes, a switch disabling the only source makes the capability
inert, and a verifier you can turn off is not a safety. Found by `vmsam-dev-2`, which
applied the ruling's reason one level down three times while the document merely
failed to contradict it.

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

**Why reading a diff by eye is not enough.** `git diff --cached -U0` shows changed
lines, and a reader maps them onto the zones of the file **as it now is**. A pure
deletion has no new-side line at all — so removing a frozen line *above* a zone reads
as "nothing added outside a zone" and passes. Removals have to be mapped onto the
**old** blob's zones and additions onto the new one's, separately. Two further holes
close the same way: moving an `END` marker down swallows a line into a zone while the
diff lands inside it and looks legal, and a fresh marker pair grants a zone that §2
reserves to the owner — so any change in marker count fails.

Found by the campaign-2 lead against this section, which its author wrote.

**What the checker covers, in prose, because a code block gets obeyed as a
procedure.** This section used to print the superseded snippets under a paragraph
saying nobody runs them — and `vmsam-ci` reported the obvious: *an agent under time
pressure reads a code block as a procedure, and the paragraph explaining that it is
not one is above it.* So the reasoning stays and the block is gone. The checker
verifies that nothing is written outside the open paths, that no governing document
is rewritten, that every changed line in a tagged file sits inside a zone with the
marker count unchanged, that `config.ini` only gains lines, and that the tree
compiles.

**The tree must compile cleanly — there is no allowance, and this paragraph used to
say the opposite of itself.** It granted `frame_compare.py` an expected failure in
its last sentence while stating in its first that campaign 2 had repaired it. Two
other documents carried the same stale claim. Measured on this revision:
`python3 -m compileall -q src -x '__pycache__'` exits **0**, clean. `vmsam-dev-1`'s
fix is what made the old allowance false, and it reported that as a finding rather
than editing it out. **A document that excuses a failure the code no longer has will
excuse the next real one** — `vmsam-ci` read the allowance and left a compile error
alone on its authority.

**Run Rust's checks when you touched Rust**: `cargo check && cargo clippy &&
cargo fmt --check`.

## 6. Reporting a blocked need

A frozen file that stands in your way is a finding, not an obstacle to route
around. Record what you wanted to change, why, and what you did instead. A
campaign that ends with three well-argued blocked needs is worth more than one
that ends with three undeclared edits.

### A defect in code you may not write is a finding, and it has a route

**Report it. Do not fix it, and do not keep it.** You read frozen code all day and
you cannot write it, so a defect you find there has no natural outlet — it goes into
your own notes and reaches nobody who can act. Three agents named that independently
on the same day, unprompted, and one had been carrying a reconciliation-window
finding since that morning. Until it was asked for, **the campaign had no channel for
the defects only its agents could see.**

Send it to the architect, who carries it to the owner. Each one carries:

- the **file and line**, verified against the revision you are on, not remembered;
- **what goes wrong and under which input** — a defect with no triggering condition is
  a worry, not a finding;
- **whether this is an observation from a run or a reading of the code.** Say it
  plainly. *A labelled hypothesis beats silence; a hypothesis wearing an
  observation's label is worse than silence* (`vmsam-ci`) — it spends credibility the
  next reader has no way to audit. The campaign lead mislabelled one of its own that
  day, ci checked before agreeing, and the label was corrected within minutes. That
  is the mechanism working.
- **observations consistent with a defect are not evidence of it.** Repaired tracks
  were seen losing to the master's own track, which is the correct outcome because the
  master is the timeline. Same observation, and it settles nothing about the defect
  next to it.

**Report the ones you decided were not your problem, and the ones you are unsure of.**
Both categories produced real findings here. And ask the question this way round:
**not "what is wrong in `src/`" but "what happens to what I produce after I hand it
over"** — `vmsam-dev-2` located three defects in frozen code only by tracing where a
repaired track goes, and none of them by inspecting the module. **The defects in
frozen code an agent can see are the ones on the path its own output takes.**
