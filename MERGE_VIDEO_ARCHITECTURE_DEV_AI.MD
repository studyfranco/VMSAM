# `mergeVideo.py` — Architecture & Function Reference

Structural map of the merge engine: execution flow, module-state contract, and
every function's inputs, outputs and side effects. Derived from an AST audit of
the file plus reading of the spine functions. Update whenever a function is
added, removed, or has its contract changed.

- **File**: `src/mergeVideo.py` — 2082 lines, 52 module-level functions,
  3 classes, 9 module variables.
- **Extracted**: `src/merge_video_legacy.py` holds two unreachable definitions
  removed from this file. Nothing imports it. See § 8.
- **Protected**: `generate_launch_merge_command` and its 34-function downstream
  closure must not be modified. See § 6.

---

## 1. Entry points and consumers

`merge_videos` and `merge_videos_with_repair` are the only functions called
from outside. Three modules import this one:

| Consumer | Names used |
| :--- | :--- |
| `main.py` | `merge_videos`, `cut_file_to_get_delay_second_method` |
| `main_gestionar_show.py` | `merge_videos`, `cut_file_to_get_delay_second_method`, `default_audio`, `errors_merge`, `show_not_compatible_error` |
| `gestionar_show/fusion.py` | `merge_videos_with_repair` plus the same four globals |

Everything else in the file is internal. That surface is the module's real
public contract — four of the five names are *module globals the caller writes
before calling*, not parameters, and `merge_videos_with_repair` resets none of
them either.

`merge_videos_with_repair(master_file_path, candidate_file_path, out_folder,
merge_sync=True, in_folder=None)` is a thin layer over an unmodified
`merge_videos`: it repairs the candidate first (§10) and returns the path it
actually merged. The master is the FIRST parameter and is never inferred —
`merge_videos` itself takes an unordered `files` collection and has no
master/candidate axis, while every repair is directional. Only the caller can
resolve which file is the master (fusion reads it from the database), so the
decision is layered on top rather than pushed inside.

---

## 2. Execution flow

```
merge_videos(files, out_folder, merge_sync, inFolder=None)
│
├─ build video objects, disambiguate duplicate basenames with _1, _2 …
├─ per file (threaded): video.get_mediadata()  →  video.calculate_md5_streams()
├─ resolve forced_best_video from tools.special_params
├─ audioRules = decript_merge_rules(tools.mergeRules['audio'])
│
├─ merge_sync = False ──→ simple_merge_video()      no alignment, straight to assembly
│
└─ merge_sync = True  ──→ sync_merge_video()
   │
   ├─ STAGE A  language selection
   │    video.get_common_audios_language() → drop "und"
   │    if no common language: fall back to the most frequent one and DROP every
   │    video lacking it; if even that is unique to one file → raise
   │    pick common_language_use_for_generate_delay
   │    (prefers tools.special_params["original_language"])
   │
   ├─ STAGE B  identical-audio deduplication
   │    group videos by concatenated audio-stream MD5; a video whose audio is
   │    byte-identical to another is removed from the working set and carried as
   │    sameAudioMD5UseForCalculation with a precomputed delay_same_md5_audio
   │
   ├─ STAGE C  delay + best-video resolution        ◄── the audit focus
   │    forced_best_video is None → get_delay_and_best_video(...)
   │    forced_best_video is set  → get_delay(...)
   │    both return `already_compared`, bound at the call site as
   │    `dict_with_video_quality_logic`
   │
   └─ STAGE D  assembly                             ◄── PROTECTED, do not modify
        generate_launch_merge_command(dict_with_video_quality_logic,
                                      dict_file_path_obj, out_folder,
                                      common_language_use_for_generate_delay,
                                      audioRules)
```

`merge_videos` returns `None`. The merged file appears in `out_folder` named
`<reference basename>_merged.mkv`, with `_1` appended when two inputs shared a
basename. Callers resolve it by glob rather than by assuming the name.

---

## 3. Module state contract

Nine module-level variables. Four are the caller-writable surface; the rest are
internal.

| Variable | Written by | Purpose |
| :--- | :--- | :--- |
| `errors_merge` | **caller**, reset to `[]` before each run | Accumulated non-fatal merge errors. Read back by the caller for the `.log.error` entry. Guarded by `errors_merge_lock`. |
| `default_audio` | **caller** | Whether to assign a default audio flag. Declared `global` inside the assembly stage. |
| `show_not_compatible_error` | **caller** | Gates every diagnostic explaining *why* a pair was rejected. Set `False` in production, which is why `.log.error` files never carry the reason. |
| `cut_file_to_get_delay_second_method` | **caller** | Seconds per cut for the second delay method. Also declared `global` inside `compare_video`. |
| `not_compatible_video_list` | internal | Files dropped as incompatible. See § 7. |
| `not_compatible_video_list_lock` | internal | `RLock` guarding the above. |
| `max_delay_variance_second_method` | internal, mutated by `compare_video` | Variance ceiling for accepting a second-method delay. |
| `max_stream` | internal (`80`) | Hard cap on output track count. |
| `errors_merge_lock` | internal | `RLock` for `errors_merge`. |

> **These are process-wide.** `merge_videos` resets none of them. Every caller
> must reset the ones it reads before invoking a merge, or state leaks between
> runs. `main_gestionar_show.process_episode` and `fusion.prepare_merge_context`
> both do this for `errors_merge`; neither yet does it for
> `not_compatible_video_list`.

---

## 4. Stage C — the delay resolution path

The audit focus. Two mutually exclusive entry points, both returning the same
shape.

### 4.1 `get_delay(videosObj, language, audioRules, dict_file_path_obj, forced_best_video)`

Used when the caller already knows which file should win. Compares the forced
best video against every other candidate, pairwise, in a pipelined loop: thread
*n+1* starts before thread *n* is joined.

| | |
| :--- | :--- |
| **Returns** | `already_compared` — `{forced_best_video: {other_path: True, …}}`. Only *successful* comparisons are recorded. |
| **Mutates `videosObj`** | Removes `dict_file_path_obj[forced_best_video]`, then appends it back at the end. **The list is reordered** — the forced best video ends up last. |
| **Mutates `dict_file_path_obj`** | Indirectly, via `remove_not_compatible_video`, which deletes failed candidates. |
| **Mutates video objects** | `compare_video` writes `delays[language]` on each candidate. |
| **Raises** | Nothing directly. `remove_not_compatible_video` raises when fewer than two files survive. |
| **Degenerate case** | If `forced_best_video` is the only file, `videosObj` is empty after the removal and the function returns `{forced_best_video: {}}` without comparing anything. |

A candidate lands in `list_not_compatible_video` when
`launched_compare.video_obj_with_best_quality is None` — i.e. `compare_video`
could not establish a reliable delay.

### 4.2 `get_delay_and_best_video(videosObj, language, audioRules, dict_file_path_obj)`

Used when no best video is forced. Runs a tournament: candidates are compared in
adjacent pairs, winners advance to the next round, and the survivor is the best
video. Substantially more complex than `get_delay` — it additionally calls
`can_always_compare_it`, `get_waiter_to_compare`, `was_they_not_already_compared`
and `remove_not_compatible_audio` to decide which pairings are still meaningful.

Same return shape, same mutation profile, same failure route.

### 4.3 `remove_not_compatible_video(list_not_compatible_video, dict_file_path_obj)`

The chokepoint. **96% of all recorded merge failures terminate here.**

1. Writes the diagnostic to `stderr` — only if `show_not_compatible_error`.
2. Records the paths via `record_not_compatible_videos` (§ 7).
3. Deletes each failed path from `dict_file_path_obj`.
4. If fewer than two entries remain, raises
   `Exception(f"Only {dict_file_path_obj.keys()} file left. This is useless to merge files")`.

Step 4 is the message that dominates the error tree. It reports the *consequence*
of dropping candidates, never the reason — which step 1 holds and production
discards.

### 4.4 Supporting cast

| Function | Role |
| :--- | :--- |
| `prepare_get_delay` / `prepare_get_delay_sub` | Compute cut positions, pick the worst shared audio quality, and extract audio parts on every video object. Heavy I/O. |
| `get_good_parameters_to_get_fidelity` | Choose the audio parameters used for correlation. Raises if none is workable. |
| `get_delay_fidelity` / `get_delay_fidelity_thread` | Threaded Chromaprint-based coarse delay + fidelity score. |
| `get_delay_by_second_method` / `get_delay_second_method_thread` | Threaded FFT cross-correlation fallback (the Rust `audio_sync` path). |
| `compare_video` (class, 470 lines) | The heaviest unit in the file. Owns the full compare of one pair: coarse delay, fine correlation, variance checks, quality comparison. Sets `video_obj_with_best_quality` to `None` on failure — the sole signal consumed by both Stage C entry points. |
| `remove_not_compatible_audio` | Recursively removes audio tracks transitively marked incompatible. Raises on unexpected `True`/`False` states. |
| `was_they_not_already_compared`, `can_always_compare_it`, `get_waiter_to_compare` | Small predicates scheduling the tournament in 4.2. |

---

## 5. Stage A/B and the simple path

| Function | Contract |
| :--- | :--- |
| `merge_videos(files, out_folder, merge_sync, inFolder=None)` | Entry point. Builds `video` objects, threads metadata + MD5 extraction, resolves `forced_best_video`, mutates `tools.language_to_completely_remove` (copies it, then removes the original language). Returns `None`. |
| `sync_merge_video(videosObj, audioRules, out_folder, dict_file_path_obj, forced_best_video)` | Stages A–D. Raises `Exception("No common language between …")` when no language is shared by at least two files. Returns `None`. |
| `simple_merge_video(...)` | Skips alignment entirely; calls `print_forced_video` then goes straight to assembly. Used when `merge_sync=False`. |
| `decript_merge_rules(stringRules)` / `decript_merge_rules_bester` | Parse the `[mergerules]` codec ranking from `config.ini` into the `audioRules` structure. Recursive. |
| `print_forced_video(forced_best_video)` | `stderr` only, gated on `tools.dev`. |

---

## 6. Stage D — protected closure

`generate_launch_merge_command(dict_with_video_quality_logic, dict_file_path_obj,
out_folder, common_language_use_for_generate_delay, audioRules)` and everything
it reaches. **Do not modify any of these 35 names:**

```
generate_launch_merge_command · generate_merge_command_common_md5
generate_merge_command_other_part · generate_merge_command_insert_ID_sub_track_set_not_default
generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language
generate_merge_command_insert_ID_audio_track_to_remove_and_new_und_language_set_not_default_not_forced
generate_new_file · generate_new_file_launch_cmd · generate_new_file_audio_config
generate_new_file_audio_config_second_pass · get_relative_id_track
check_delay_retention · check_delay_retention_sub
find_differences_and_keep_best_audio · keep_best_audio
clean_number_stream_to_be_lover_than_max · remove_sub_language · keep_one_ass
sub_group_id_detector · sub_group_id_detector_and_clean_srt_when_ass_with_test
clean_srt_when_ass · not_keep_ass_converted_in_srt · get_sub_title_group_id
insert_type_in_group_sub_title · clean_title · clean_dubtitle_title
clean_hearing_impaired_title · clean_forced_title
test_if_dubtitle · test_if_hearing_impaired · test_if_forced
prepare_get_delay_sub · get_delay_fidelity · get_delay_fidelity_thread
get_good_parameters_to_get_fidelity
```

Note the overlap: `prepare_get_delay_sub`, `get_delay_fidelity`,
`get_delay_fidelity_thread` and `get_good_parameters_to_get_fidelity` are reached
from **both** Stage C and Stage D. Touching them for a Stage C change would
modify protected behaviour — treat them as protected.

Broadly this closure does: pick the best audio per language, collapse subtitle
tracks into groups and clean their titles, cap the output at `max_stream` tracks,
build the per-file `ffmpeg` normalisation commands, run a two-pass conversion,
and emit the final `mkvmerge` command.

---

## 7. `not_compatible_video_list` — the incompatibility record

Populated in exactly one place — `remove_not_compatible_video` (§ 4.3) — with
the paths of candidates that failed correlation.

```python
reset_not_compatible_video_list()          # call before each merge_videos run
get_not_compatible_video_list()  -> list   # snapshot, safe to keep
record_not_compatible_videos(paths)        # internal; used by remove_not_compatible_video
```

All three take `not_compatible_video_list_lock`. `get_…` returns a **copy**, so a
later merge cannot mutate a snapshot a caller is still holding.

**Scoping is the caller's job**, matching the existing `errors_merge` convention.
`merge_videos` does not reset it. Until a caller calls
`reset_not_compatible_video_list()` the list accumulates for the lifetime of the
process and no longer describes a single merge.

Prior to this refactor the list was appended to and **never read by anything** —
no return path, no consumer, no reset. The accessors above exist to make it
usable for post-processing of problem cases without any caller reaching into the
global.

---

## 8. Extracted legacy — `merge_video_legacy.py`

Two definitions were removed from `mergeVideo.py` after an AST call-graph walk
from every real entry point proved them unreachable, and a text search found no
dynamic or string reference anywhere in `src/`.

| Name | Was | Why it is dead |
| :--- | :--- | :--- |
| `class get_cut_time` | L688-731, 44 lines | `run()` dereferences `self.video_obj_1`, `self.video_obj_2`, `self.process_to_get_best_video` and `self.get_best_video` — none assigned by `__init__`, none defined on the class. Instantiating and starting it raises `AttributeError`. An abandoned refactor of `compare_video`. Unrelated to the module `src/get_cut_time.py`; `mergeVideo.py` never imported it, so nothing was shadowed. |
| `find_a_cut_for_not_compatible` | L812-822, 11 lines | Coherent but never called. Raises the cut count to an odd number ≥ 15 and re-extracts audio for the failed candidates — i.e. retries alignment at higher resolution instead of dropping them. |

`find_a_cut_for_not_compatible` is worth keeping in view: it is the
second-chance pass that `remove_not_compatible_video` currently skips, and 96%
of recorded failures die at exactly that point.

---

## 9. Defects observed during the audit

Recorded, not fixed — all sit in active legacy code.

1. **Index-shifting deletion in `sync_merge_video`.** When no common language
   exists, incompatible videos are collected as ascending indices and then
   removed with `for i in list_video_not_compatible: del videosObj[i]`. Each
   deletion shifts every later index down by one, so with two or more
   incompatible files the wrong elements are deleted. The parallel
   `dict_file_path_obj` cleanup uses paths and is correct, leaving the two
   structures inconsistent.

2. **`get_delay` reorders `videosObj`.** The forced best video is removed and
   re-appended, so it ends up last rather than at its original position. Any
   caller depending on input order sees it silently changed.

3. **The rejection reason is unrecoverable in production.** Every explanation of
   *why* a pair was incompatible is written to `stderr` behind
   `show_not_compatible_error`, which production sets to `False`. The
   `.log.error` files therefore record only the terminal
   `"Only … file left"` message.

4. **`max_delay_variance_second_method` and `cut_file_to_get_delay_second_method`
   are mutated at runtime** by `compare_video` via `global`. Two merges in one
   process do not start from the same thresholds.

---

## 10. Master/candidate repair prebuilds

Appended after the protected closure and after `merge_videos`, so nothing above
it is touched. This section owns every decision about HOW a mismatched
candidate is repaired before merging — the intelligence that used to live in
`gestionar_show/fusion.py`, which is now only a queue worker that relaunches
merges.

| Name | Role |
| :--- | :--- |
| `pal_factor`, `pal_ratio_tolerance` | `25/(24000/1001)` = 1.042708, ±0.5%. The **theoretical** factor is always used, never a measured endpoint ratio. |
| `probe_duration_seconds(file_path)` | Container duration via `ffprobe`, or `None`. |
| `detect_pal_relation(master, candidate)` | `pal_factor`, `1/pal_factor`, or `None`. Directional — master first. |
| `load_pal_factors_record(candidate)` | Record from `tools.get_pal_factors_table()`; accepts the bare-list and `{results: [...]}` shapes. Cached in `pal_factors_cache`. |
| `apply_pal_step_repair(corrected, master, candidate, factor, prebuild_dir)` | PAL speed **plus one content step**: chains a chimeric re-assembly after the speed correction. Only for `LINEAR_SPEED_PLUS_CONTENT_STEP`. |
| `prepare_pal_corrected_candidate(master, candidate, prebuild_root=None)` | Speed-corrected prebuild via `merge_video_prebuild`, then the step chain. |
| `load_cut_points_record(candidate)` | Record from `tools.get_cut_points_table()` merged with its `cut_points_c.json` sibling. Cached in `cut_points_cache`. |
| `prepare_structural_corrected_candidate(master, candidate, prebuild_root=None)` | Chimeric rebuild via `merge_video_chimeric` for piecewise-constant lag. |
| `prepare_repaired_candidate(master, candidate, prebuild_root=None)` | Composes the two: PAL first; structural **only** if PAL returned the candidate unchanged. |
| `merge_videos_with_repair(...)` | `prepare_repaired_candidate` → `merge_videos` → returns the merged input path. |

**Guards.** Both table-driven paths compare `record['master']` against the
caller's master and refuse when they disagree: plateaus fitted to another
file's timeline would re-assemble the candidate against the wrong reference,
which is how a silent desync happens. `detect_pal_relation` has no such guard —
it is directional by construction, so the argument order is the contract.

**Prebuild root.** Defaults to `tools.tmpFolder_original` (stable for the run),
deliberately **not** `tools.tmpFolder`, which the caller rewrites per episode
and deletes in its `finally`. The `prebuild_root` keyword exists only so tests
can redirect it.

**Imports.** `merge_video_prebuild` and `merge_video_chimeric` are imported at
module scope. Neither imports `mergeVideo`, `video`, or any dependant of them,
so no cycle exists — do not add such an import to either.

**Not inherited by `main.py` / `main_gestionar_show.py`.** They still call
`merge_videos` directly, because they have no resolved master to repair
against. Every function above is public, so they can opt in later once they
decide which side is the master.
