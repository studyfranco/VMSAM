# PIPELINE.MD — what the engine does, and what a good output is

**Moved out of `AGENT.MD` 2026-09-05 at the owner's instruction — that file is read before a seat can
work and every line in it is paid for in every seat, on every restart.** *This is DOMAIN reference:
consult it when you need it, not before every action.* **`AGENT.MD` keeps the rules; this keeps the
subject matter.**

# THE CONTROL FLOW — READ THIS BEFORE YOU TOUCH THE ENGINE

**FILES THIS DESCRIBES:** `src/mergeVideo.py` · `src/change_point_locator.py` ·
`src/merge_video_repair.py` · `src/merge_video_chimeric.py` · `src/merge_video_resample.py` ·
`src/gestionar_show/fusion.py`
**MEASURED AT** `origin/dev-AI` tip `a4b57808`. *`vmsam-dev-2` measured that the first version of
this file named ZERO source files against a control of 121 — **a successor grepping for the module
this document is about never found the document.** Its eight line citations all resolved correctly
and carried no revision anchor:* **CORRECT-TODAY-AND-UNANCHORED IS THE DANGEROUS STATE, NOT THE SAFE
ONE** *— that seat's own file moved four times in one day. Line numbers below are a convenience;
the function name and the phrase are the citation.*

> **AND THIS FILE IS NO LONGER THE ONLY PLACE. `vmsam-dev-2` put the relationship into
> `src/merge_video_chimeric.py`'s module header — acting on the closing rule rather than agreeing
> with it — with the line that settles any future disagreement:** ***if this and `PIPELINE.MD` ever
> disagree, MEASURE THE CODE, and the code is that file.***

> **WHY THIS EXISTS: `vmsam-dev-1` MEASURED that the locator NOWHERE STATES why it exists — zero hits
> for "constant delay", "single delay", "re-encode", "subtitle", against a control at 39 that fires.
> THE RELATIONSHIP BELOW LIVES ACROSS TWO OWNERS' FILES AND IS IN NEITHER DOCSTRING.** *A successor
> reading either module alone learns the wrong thing and acts on it.*

## 1. THERE IS **ONE CYCLE**, AND THE REPAIR IS A DETOUR INSIDE IT

**Most files correlate on a SINGLE CONSTANT DELAY, applied as a container offset — no re-encode,
bitmap subtitles preserved. THE REPAIR EXISTS ONLY FOR FILES WHERE NO SINGLE DELAY HOLDS.**

    merge_videos :2141  ->  sync_merge_video :2050
                                    |
                                    v
    +--- get_delay :961  *** THE PRODUCTION PATH *** -----------------+
    |     mergeVideo.py -- THE STAGE WHERE COMPATIBILITY IS DECIDED    |
    |                                                                 |
    |     compare_video -> correlation                                |
    |          |                          |                           |
    |          | ONE DELAY HOLDS          | NONE HOLDS                |
    |          |                          v                           |
    |          |     mergeVideo.py:remove_not_compatible_video :798   |
    +----------|--------------------------|---------------------------+
               |                          |  *** LEAVES THE CYCLE ***
               |                          v
               |            repair_not_compatible_videos :1980
               |                          |
               |            locator -> PLAN   |   None -> record()
               |                          |
               |            RESAMPLE **THEN** CUT  ->  track rebuilt
               |                          |
               |<-------------------------+  *** RE-ENTERS AT THE SAME POINT ***
               |                             hung on the master:
               |                             best_video.sameAudioMD5UseForCalculation
               v
    ############ FROM HERE, NO DIFFERENCE AT ALL ############
               |
        generate_launch_merge_command :1815
               |
        find_differences_and_keep_best_audio :1935
               |
        the produced file

### THERE ARE **TWO** ENTRIES, AND PRODUCTION TAKES THE ONE THIS DIAGRAM DID NOT DRAW

**Found by `vmsam-ci`, re-derived by `vmsam-lead`, RE-VERIFIED HERE BY AST — enclosing function
computed from the tree, never from indentation:**

    mergeVideo.py:remove_not_compatible_video  DEFINED 798-868, TWO call sites:
      :958  enclosing `get_delay_and_best_video`  -- no reference imposed. CLI / tournament.
      :988  enclosing `get_delay`                 -- *** THE FUSION PATH ***

    THE ROUTING, mergeVideo.py:2127-2131:
      if forced_best_video == None:  -> get_delay_and_best_video
      else:                          -> get_delay
    AND gestionar_show/fusion.py SETS IT IN **BOTH ARMS** OF ONE if/else, :343 and :348.
    **NO PATH THROUGH THE FUSION WORKER LEAVES IT UNSET.**

> **EVERY REPAIR THE CONTAINER HAS EVER PERFORMED ENTERED AT `:988`. The first version of this
> document drew `:958`.** *A successor reading it to understand the PRODUCTION repair path would have
> studied the wrong function — which is exactly the failure this file was written to prevent.*

**THE SHAPE IS UNCHANGED AND HOLDS FOR BOTH ENTRIES:** *both call the same function at `:798`, and
both hang the result on `best_video.sameAudioMD5UseForCalculation`.*

> **A REPAIRED FILE GOES BACK THROUGH THE SAME CYCLE AS A NORMAL ONE.** *It is not a parallel lane
> that rejoins later — it leaves at the refusal, inside `get_delay_and_best_video`, and re-enters at
> the point it left.* **DOWNSTREAM — track choice, subtitles, the merge command — NOTHING
> DISTINGUISHES THE TWO.**

**AND THAT IS WHY `len(dict_file_path_obj) < 2` USED TO RAISE ON THE VERY PATH THE REPAIR EXISTS
FOR:** *the check counted the files REMAINING without seeing that one had just come back in.*
**Campaign 1's defect exactly — code written, reviewed, tested green, that nothing ever reaches.**

## 2. THE CUT PLAN, AND THE ORDER THAT MATTERS

    STEP 1 -- WHAT THE MEASUREMENT RETURNS: one segment
      master_start_ms       where, on the MASTER timeline
      master_end_ms         until where
      candidate_offset_ms   by how much to shift the candidate

    STEP 2 -- WHAT GETS GLUED: one piece      normalize_segments, chimeric.py:267
      source = "candidate" | "master" | "silence"
      GAPS IN THE PLAN BECOME **MASTER** PIECES, never emptiness.
      IT REFUSES RATHER THAN CLAMPS -- and not by discipline:
        7 raise sites · 1 return · **ZERO min()/max() calls**
        A CLAMP WOULD NEED ONE. THERE IS NO CLAMPING MACHINERY AT ALL.
      Clamping would produce a file "plausible et silencieusement" wrong.

    STEP 3 -- THE FILTER CHAIN: **THE SLOPE FIRST, THE STAIRCASE ON TOP**

      [0:candidate] --speed_chain--> [spd] --asplit--> pieces --concat--> rebuilt
                     ^^^^^^^^^^^^^^                    ^^^^^^
                     THE SLOPE IS UNDERNEATH           THE STAIRCASE IS ON TOP

      *** THE RESAMPLE IS APPLIED **BEFORE** THE CUTTING. THEY ARE NOT TWO ROUTES. ***
      src/merge_video_chimeric.py -- confirmed by its owner from the code, not the
      comment: `candidate_entry` is REBOUND to `[spd]` BEFORE the `asplit`.
      The candidate's `start_time` is MULTIPLIED BY THE RATIO first, and the slice
      times are read on the **ALREADY-RESAMPLED** candidate.
      `SPEC_ZONE_A.MD` §4 says `chimeric+resampled` IN THAT ORDER for this reason.

    STEP 4 -- THE END: THE REPAIRED OBJECT REJOINS THE MERGE
      hung on `best_video.sameAudioMD5UseForCalculation`
      consumed by `generate_launch_merge_command` WITHOUT the delay machinery
      -- because it no longer needs one. IT IS A FILE LIKE THE OTHERS AGAIN.

## 2b. THE MARKER THAT WOULD SEPARATE THEM IS COMPUTED AND THEN DROPPED

> **`record(candidate_path, outcome, reason, detail)` TAKES A FOURTH ARGUMENT AND
> ***NEVER PUTS IT IN A LOG LINE.*** VERIFIED BY AST AT THE AUTHORITY:**

    record()'s only `tools.logs.append` interpolates  outcome · candidate_path · reason
    it does NOT interpolate                           **detail**
    `plan_kind` is passed ONLY as `detail`, at :2075 and :2080
    -> `detail` reaches `last_repair_report` IN MEMORY and reaches NO READER.

**SO A ROUTING DECISION AND A DEAD END BOTH EMIT `declined`, AND THE STRUCTURED MARKER THAT WOULD
TELL THEM APART — `plan_kind: "constant"` — IS PUT IN A DICT AND THROWN AWAY BEFORE ANY REPORT SEES
IT.** *Measured by `vmsam-dev-4` on its own corpus: 16 `repair: declined` lines, 4 carrying the
constant-offset prose, **0 carrying a `plan_kind` marker**, against a control of 1706 `repair:` lines
— the search has power.*

    **IF YOU ARE WRITING A CAUSE, PUT IT IN `reason`, NEVER IN `detail`.**
    `dev-2` avoided this an hour before it shipped, by using `reason` for its cause
    tokens -- and it said plainly that it used `reason` BECAUSE THE OLD STRING WAS
    THERE, not because it had checked. **INHERITANCE, NOT INSIGHT.** The next seat
    will not inherit anything.

*The value exists, it is one interpolation away from the log line, and it is inside the REPAIR
boundary.*

## 2c. A LATENT HAZARD: THE PLAN'S OFFSETS ARE MEASURED ON THE **UNRESAMPLED** PAIR

**`vmsam-dev-1` contradicted the step-3 framing and DISSOLVED the interface question rather than
answering it. VERIFIED HERE at `src/change_point_locator.py`:**

    speed_ratio 0 · asetrate 0 · atempo 0 · resample 0
    CONTROL `quantum_ms` -> 12, FIRES.   its only kinds: "constant" | "piecewise_constant"

**THE LOCATOR EMITS NO RATIO AT ALL. When it detects a monotone drift it DECLINES — it never
produces a plan describing a speed relation.** *So there is no ordering question at its interface.
There is a stronger constraint:*

> **EVERY MILLISECOND IN A PLAN IS THE RELATIONSHIP BETWEEN MASTER AND CANDIDATE *AS THEY ARE ON
> DISK*. IF A RATIO IS APPLIED BEFORE THOSE SLICE TIMES ARE READ, THE OFFSETS ARE NOT REORDERED —
> THEY ARE ***INVALIDATED***, because they were measured against a timeline that no longer exists.**

    THAT IS COMPATIBLE WITH §2 STEP 3 **ONLY BECAUSE A PLAN AND A SPEED RATIO NEVER
    COEXIST TODAY** -- and they never coexist only because the locator DECLINES the
    speed family. **THE ORDERING IS SAFE BY ACCIDENT OF A REFUSAL, NOT BY DESIGN.**
    AND WE NOW KNOW THAT REFUSAL IS ISSUED FOR A **FALSE STATED REASON** (CAMPAIGN.MD:
    six of six real PAL pairs took `scattered`, saying "no shared content" about files
    that share everything at 4.27 %).
    *** IF THAT BRANCH IS EVER FIXED SO PAL PRODUCES A PLAN, THIS HAZARD GOES LIVE. ***
    ANYONE TOUCHING THE SPEED BRANCH MUST SETTLE THIS FIRST.

### And a costume: `kind == "speed"` has ZERO producers

    the ONLY occurrence in src/ is the READER, merge_video_repair.py:101
      if plan.get("kind") != "speed" and plan.get("speed_ratio") == None: return None, None

**No plan produced anywhere in the tree can carry `kind == "speed"`, so that conjunct is TRUE for
every producible plan and never changes the outcome.** *The line still behaves correctly — it
reduces to "if there is no ratio, do nothing" — but the test is INERT, and a reader seeing a
`"speed"` branch infers a producer that does not exist.* **`dev-1` inferred exactly that for ten
minutes, about its own module.**

### And a source comment that names the wrong module

**`src/merge_video_repair.py:130` attributes a `speed_ratio` emission to the locator's author.** *At
the current blob that module emits no such field in either convention.* **THE ACTUAL PRODUCER IS
`src/merge_video_resample.py:11`, and it uses the convention the comment says is correct.** *The
defect the comment records is real and well documented; it is FALSE OF THE CODE AS IT NOW STANDS and
sends a live reader to the wrong file.* **A citation with a stale half, in a source comment rather
than in a document.**

## 3. THE TWO SENTENCES A SUCCESSOR MUST NOT CONFUSE

    "THIS FILE CANNOT BE HELPED"            -> a dead end
    "THIS FILE GOES BACK TO THE CHEAP PATH" -> a ROUTING DECISION

**A plan that is a CONSTANT OFFSET WITH NO GAP is refused by the repair — and that refusal is the
SECOND sentence, not the first.** *Rebuilding would cost an encode per track and destroy the bitmap
subtitles a container offset keeps.* **THE REPAIR HANDS THE FILE BACK. It does not give up on it.**

---

## 🔄 Synchronization pipeline

```
merge_videos(files, out_folder, merge_sync=True)
  └─ sync_merge_video                       language selection, MD5 audio grouping
       ├─ get_delay_and_best_video          no reference imposed: tournament of pairs
       │    or get_delay                    forced_best_video: every file against it
       │      └─ prepare_get_delay          extraction of the comparison windows
       │      └─ compare_video (Thread)     one per pair
       │           └─ test_if_constant_good_delay
       │                ├─ first_delay_test        1 · Chromaprint, per window
       │                └─ second_delay_test       2 · FFT / Rust, one long block
       │           └─ get_best_video                3 · PSNR + VMAF, only if no
       │                                                reference is imposed
       │           └─ adjust_delay_to_frame         frame snap
       │      └─ remove_not_compatible_video        4 · refusal  ← repair zone A
       └─ generate_launch_merge_command     ffmpeg passes, then mkvmerge
```

**Why stage 2 is in Rust** (owner's design intent, absent from the code): the FFT
correlation was Python, and it ran out of memory on long files. `audio_sync` is its
optimised rewrite; the Python implementation survives as a fallback. Rust buys
memory, not accuracy — the tolerances above are unchanged by it.

**The rule the whole design rests on: one constant offset for the pair.** It is
enforced in three places, not one — do not mistake the last for the rule:

- `first_delay_test` requires the delays to agree **across windows and across track
  pairs**. Pairs that disagree go to `ignore_audio_couple`; too wide a disagreement
  raises. This is where constancy is actually established.
- `second_delay_test` has its own variance tolerance,
  `max_delay_variance_second_method` = **0.004 s**.
- `test_if_constant_good_delay` then checks that **stage 2's correction stays under
  500 ms**. Stage 2 re-extracts the candidate already shifted by stage 1's answer,
  so it measures a *residual*, not a second independent offset: the check
  `abs(calculated_delay - delay_first_method) < 500` reduces algebraically to
  `abs(delay_second_method * 1000) < 500`.

**Why 500, and why stage 2 at all** (owner's design intent, recorded here because the
code does not say it): fingerprint peaks were observed putting stage 1 off by
roughly **±250 ms**. Rather than trying to filter those peaks, stage 2 runs
unconditionally and *corrects* them — a residual inside the band is applied, one
beyond it means stage 1 was not merely imprecise but wrong, and the pair is refused.
So the threshold is not a confidence margin: it is the width of the error stage 2 is
expected to absorb.

The side effect is precision. Stage 2's own tolerance is 4 ms against a frame of
41.7 ms, so the residual is fine enough to expose **a cut that is one frame longer
than its counterpart** — which is how a structural difference first becomes visible.

Everything downstream assumes that single number is right.

**A delay measured by comparison is snapped to a whole video frame** when the
reference is CFR — 41.708 ms at 23.976 fps, via `adjust_delay_to_frame`. Sub-frame
precision is discarded by design, so never sell a method on it. Three limits go with
that sentence, and each has been mistaken for the rule before:

- **the snap does not survive to the container.** `mkvmerge` is given
  `--sync -1:<int ms>`, so a delay snapped to 792.459 ms ships as 792 — half a
  millisecond off the boundary. Far below the frame quantum, but it means "lands
  exactly on a frame" is not literally true;
- **only comparison delays are snapped.** `adjust_delay_to_frame` is called from
  `compare_video` alone. The MD5-group delays (`delay_same_md5_audio`) are derived
  from container delays and go through unsnapped;
- **VFR is not refused, it is unsnapped.** The non-CFR branch returns the delay as
  measured, with a `TODO`. Nothing rejects a variable-frame-rate reference.

**A refusal is a correct outcome, not a failure.** No constant offset means the
files differ in a way a single delay cannot express, and merging anyway produces a
plausible file that is silently wrong.

What each stage actually does, read from the code, is in `FILE_MAP.MD`.

## 📊 Handled Cases & Core Features

- **Audio Stream Deduplication & Ranking**: evaluates stream properties (codec,
  bitrate, sampling rate, channel layout) against **configurable** priority rules in
  `config.ini` `[mergerules]` — e.g. `Flac>AAC`, `Flac>MP3`, `DTS>E-AC-3*1.1` — to
  retain only the optimal tracks **per language**. `keep_best_audio`.
- **Subtitle Cleaning & Deduplication**: prioritises styled formats (ASS over a raw
  SRT carrying the same text — `not_keep_ass_converted_in_srt`), cleans text
  streams, classifies hearing-impaired and dubtitle tracks
  (`test_if_hearing_impaired`, `test_if_dubtitle`), and drops identical streams by
  MD5. Text formats hash on **text**, image formats (PGS/VOBSUB) on packet bytes.
- **Track Metadata & Flag Assignment**: detects and tags default, forced,
  commentary and audio-description streams — from the container flags *and* from
  the title (`commentary`, `audio description`, `audio guide`, `descriptive`) — with
  ISO 639 compliance via `iso639.Lang`, falling back to the macrolanguage.
- **Multi-Pass Remuxing Pipeline**: FFmpeg extracts and loudness-normalises the
  comparison audio (`loudnorm`, −23 LUFS), applies offsets with `aresample` and
  `-itsoffset`, then `mkvmerge` assembles the container and receives the final
  per-track `--sync` in integer milliseconds.

## 🏷️ The `fabricated` marker

A track that the pipeline **built** is not a track it **copied**, and after the mux
nothing tells them apart. One key carries the difference, on the audio stream dict:

| `fabricated` | |
| :--- | :--- |
| absent or empty | the track is untouched — it came from a source file as it was |
| a non-empty string | the track was **built**, and the string says how: `chimeric`, `resampled:<factor>`, `chimeric+resampled:<factor>` |

**Test truthiness, never equality.** A value you do not recognise still means
fabricated, so a method added later needs no change at any consumer.

On the produced file the same string is a Matroska track tag, `VMSAM_FABRICATED`,
so the origin survives the merge and is recoverable months later:

```sh
ffprobe -v error -show_entries stream_tags=VMSAM_FABRICATED -of compact <file>
```

**Nothing sets the key today** — `keep_best_audio` already reads it (a fabricated
track loses to an intact one before any codec rule) and the branch is inert until
the repair marks its tracks. The full contract is `SPEC_ZONE_A.MD` §4.

**Why it is defined before it exists.** Provenance is knowable only at the moment
of production; after that it is gone. Campaign 1 measured the consequence: **zero of
110 measurements could be attributed to the code that produced them**, and three
separate tools reconstructed track provenance from emission order, from track
counts and from codec — three methods, three answers, no tiebreak.

## 🎯 Target-first — be certain of what you must obtain

You are free on the method. You are not free on knowing what a correct result is
**before** you produce one.

1. **Build the expected file by hand, first — and state what it shares with the code
   under test before believing any agreement.** *A target that shares an input with
   the code under test cannot disagree with it*. That is checkable in advance, which
   "build it independently" is not. Campaign 2 met every clause of
   `SPEC_ZONE_A.MD` §2 on a real file and proved nothing, because the hand-built
   target came from the same plan as the splice; the claim was refused on its own
   result rather than left to stand. Whatever the change, produce the
   output you say the code should produce — manually, with whatever tools you like
   — and check it yourself. This is the target, and it exists before the code does.
2. **Run the pipeline locally against it.** This pass hunts bugs, not correctness:
   crashes, files that never appear, a path never taken.
3. **When the code holds, ask for a run in the test instance.** A local run proves
   your code executes; only a real run proves the pipeline produces the result. You
   do not operate that instance. **Its rules are deliberately absent from this
   repository — you are given them in the prompt.** Copy them into `VMSAM_HELP_AI/`
   the moment you receive them; do not plan around them, and do not infer any of
   them from the runs you have seen.
4. **Compare.** The produced file and your target must be **identical to what you
   expected**. Not close, not plausible — identical on the properties you named in
   step 1.

**A test run is batched, exactly like a commit.** A pass costs the same whether it
carries one line or a finished feature — so ask when you hold something worth the
pass, and state what you expect it to show *before* it runs. And never assume the
instance carries your latest code: name the revision you believe is under test, and
when a result contradicts what you read in the source, question that assumption
before you question the code.

**Why in this order.** A target built afterwards is fitted to whatever came out,
and then any result looks like a success. Campaign 1 has the example: a method was
tuned until its numbers looked excellent — the tightest spread of a whole benchmark,
0.008 ms — while it correlated the wrong language and was wrong by 21 ms. A file you
built and verified yourself cannot be argued with by a plausible number.

**And it is the only check dead code cannot pass.** Five times, code was written,
reviewed, unit-tested green, and never called by anything: every suite passed. A
unit test calls the function directly, which is exactly what production does not do.
Comparing a produced artefact against a target you built proves the path ran.

State the target **and the acceptance condition before running** — including what
result would refute you. A prediction written after the fact is not a prediction.

**And if you dispatch work, the refutation is yours to supply, not to hope for.**
This clause addresses whoever does the work, so a dispatcher can satisfy every word of
it and still hand out a task with no refutation in it — which happened, and the agent
supplied its own, covering for the dispatch. A task without a stated refutation is a
task nobody can lose.

