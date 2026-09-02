# VMSAM — Write Zones

**What an agent may change, where, and how compliance is checked.**

This file is the authority. A campaign that violates it is not validated, whatever
its results. Read it before the first edit, not after a review.

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

Adding a new zone is an owner decision, not an agent one.

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

### What is already fixed in the tagged functions

`get_good_frame` starts with `calculated_delay = float(calculated_delay)`.
`test_if_constant_good_delay` returns a `Decimal` and the division by 1000 keeps
that type, while every timeline value in the function is a float seeded by
`get_fps()`. The first addition raised `TypeError`, `compare_video.run` caught it
and recorded the pair as *not compatible* — so the whole no-forced-best-video
path silently dropped every candidate. Do not remove that line without
understanding what it protects.

`get_less_channel_number` and `get_less_sampling_rate` are **frozen**, but read
them: both answer a failure with a fabricated plausible value (`"2"`, `"44100"`)
from a bare `except:`, and both now log it under `tools.dev`. If those lines ever
appear in a run, that is a finding worth reporting — the value is
indistinguishable from a real one everywhere downstream.

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
`src/folder_generator.py`, `src/gestionar_movie/`, new modules, campaign tooling,
tests, diagnostics and documents.

### Dormant capabilities — this is the moment to wake them

Three files hold work that nothing calls today. They are not debris to step
around: they are capability waiting to be used, and the owner expects them to be
picked up.

| file | state |
| :--- | :--- |
| `frame_compare.py` | pHash/DCT frame comparison. **Does not compile** — `frame_compare.py:109`, `costs =  * m`, pre-existing since `da1c191`. Fix that first. |
| `get_cut_time.py` | multi-range cut detection, drift, audio chimera. Its only importer is nobody; it imports `frame_compare`. Beware the homonym: `mergeVideo` used to define a *class* of the same name, now in `merge_video_unused.py`. |
| `merge_video_unused.py` | `find_a_cut_for_not_compatible` and the old `get_cut_time` class, extracted from `mergeVideo.py`. Nothing imports it. |

Read them, reuse them, rewrite them, delete what you do not want.

New capability goes in **its own module**, called from a tagged zone, behind a
flag that is **off by default**, with a test that pins the call site.

## 5. Zone A — the contract

Repair happens **when a file is rejected**, inside `remove_not_compatible_video`.
Not before the merge, not during it. The rejected file is repaired against
`best_video` and rejoins the merge through the MD5 group.

`best_video` is a **video object** on both call sites.

Available without any change:

```python
video_obj.delayFirstMethodAbort[best_video.filePath]   # -> [1|2, delays]
```

`first_delay_test` fills it at 12 points. It tells you *why* the pair was refused
and which delays were seen — you do not need to re-run the correlation that just
failed.

Rejoining the merge: attach the repaired object to
`best_video.sameAudioMD5UseForCalculation`. It is then consumed by
`generate_merge_command_common_md5`, which never goes through the delay machinery.

**Three requirements, each verified against the code:**

1. `delay_same_md5_audio` **must be a `Decimal`** — `Decimal('0')` when the repair
   already aligned the file. It defaults to `None` (`video.py:55`) and
   `mergeVideo.py:1426` computes `delay_same_md5_audio + delay_to_put`:
   `None + Decimal` raises `TypeError`.
2. The object must have had `get_mediadata()` — `generate_new_file` reads
   `audios` and `video['Duration']`.
3. The temporary file must survive until `mkvmerge` runs. Do not put it anywhere
   that is cleaned between the repair and the merge.

## 6. What closes zone A

**Not** "the code is in the zone and the tests are green." The previous campaign
produced five pieces of code that were written, reviewed and unit-tested green,
and that **nothing ever called** — a guard, a refined measurement, a policy
branch, an order check, a subtitle repair. Every suite passed. A unit test calls
the function directly, which is exactly what production did not do.

Zone A closes when, on a real error file that used to fail:

1. a merged output is produced, and
2. the repaired track is **present in the final mkv**, and
3. the agent produced the expected file itself and compared it against the output.

Point 3 is the one that cannot be faked by dead code.

## 7. Verification

Before every commit:

```sh
# 1. no write outside the open paths
git diff --cached --name-only

# 2. in the two tagged files, every changed line sits inside a zone
#    (BEGIN: AGENT modification ok ... END: AGENT modification)
git diff --cached -U0 -- src/mergeVideo.py src/tools.py

# 3. config.ini: additions only, no changed or removed line
git diff --cached -- src/config.ini | grep '^-' | grep -v '^---'   # must be empty

# 4. the tree still compiles
python3 -m compileall -q src -x '__pycache__'
```

Check 3 must print nothing. Check 4 fails on `frame_compare.py` until someone
fixes it; that failure is expected and is the only one allowed.

## 8. Reporting a blocked need

A frozen file that stands in your way is a finding, not an obstacle to route
around. Record what you wanted to change, why, and what you did instead. A
campaign that ends with three well-argued blocked needs is worth more than one
that ends with three undeclared edits.
