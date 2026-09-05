# PIPELINE.MD — what the engine does, and what a good output is

**Moved out of `AGENT.MD` 2026-09-05 at the owner's instruction — that file is read before a seat can
work and every line in it is paid for in every seat, on every restart.** *This is DOMAIN reference:
consult it when you need it, not before every action.* **`AGENT.MD` keeps the rules; this keeps the
subject matter.**

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

