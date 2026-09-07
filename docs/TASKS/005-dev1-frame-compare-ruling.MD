# 005 — `frame_compare.py`: repair or replace, and say which before writing

**Agent:** `vmsam-dev-1`. **State:** ruled, edit applied. **Do this before 006.**

## Goal

`src/frame_compare.py` does not compile — `frame_compare.py:109`, pre-existing
since `da1c191` — and **nothing in the live pipeline imports it or
`get_cut_time.py`**. Frame comparison is one of the two instruments that can
locate a cut. Decide whether to repair it or replace it, **and say which and why
before writing a line.**

The break is not subtle:

```python
costs =  * m          # line 109
hits  =  * m          # line 110
smoothed =  * m       # further down
```

A list literal was lost from three initialisers. So the *cost of repair* is
minutes; the ruling is not about that. It is about whether the method the file
implements — pHash on 32×32 grey frames, banded alignment, worst-cost window,
ffmpeg scene detection as a fallback — is the instrument we want, or whether it
should be replaced.

## What the ruling has to weigh

- `CAMPAIGNS/01-dev-AI/RESTART.MD` §10 argues **against** scene scores for this
  problem and **for** perceptual hashing, on the grounds that the same cut scores
  differently at two bitrates so two scene lists do not align. Note that
  `frame_compare.py` already *is* perceptual hashing, with scene detection only as
  a fallback — so §10 is closer to an argument for repairing it than against.
  That is my reading, not a ruling; make your own and give your reasons.
- `RESTART.MD` §10 is a campaign-1 opinion, and consulting it is not following it.
- The audio side may be the better instrument for a change point measured in
  points of 125 ms. Video is what pins it to a **frame**. Say which question you
  are answering.
- **Whatever you choose must be reachable.** A module nothing imports is exactly
  the campaign-1 failure. Name, in this file, the caller you intend — and note
  that the caller is dev-2's, so the two of you agree on the signature before
  either writes it.
- `get_cut_time.py` (565 lines, imports `frame_compare`) already contains
  `_refine_cut`, `_scene_refine_window`, `_refine_cut_audio`,
  `_build_chimeric_audios`. Read it before writing anything new. It is open, it
  has no importer, and it may be a better starting point than a blank file — or
  it may be dead weight. Rule on that too.

## Acceptance

The ruling, written here, **before** the first edit. Then: `python3 -m compileall
-q src -x '__pycache__'` fails on nothing, or fails only where you say it should
and you have said why.

## The ruling

**Written before the first edit.** Everything marked *measured* carries the
command that produced it; everything else is marked *inferred* and is a reading of
the source, not a run.

### Which question I am answering

The task offers a choice and I take the second: **video is the instrument that
pins a change point to a frame; audio brackets it.** That decides the shape of
what 005 must produce. The audio quantum is not a frame and is not fine enough —
see the correction below — so the video instrument's job is *not* to find the
change point from nothing. Its job is: **given a bracket from audio, return a
signed displacement in whole reference frames.** Anything that returns a *region*
rather than a *displacement* is answering a different question.

### Ruling in one line

**Repair the compile break in `frame_compare.py`; keep and reuse its pHash
front end; do NOT adopt its alignment back end. Retire `get_cut_time.py` — it is
dead weight, and its defects are worse than its absence.**

### 1. The break is not what the task description says it is

The task, `WRITE_ZONES.MD` §4 and `FILE_MAP.MD` all describe a lost list literal in
three initialisers of one file. **Measured, that is not the shape of the damage:**

```sh
$ for f in src/*.py; do echo "$(grep -o '\[0\]' $f | wc -l)  $(wc -l < $f)  $f"; done
0   220   src/frame_compare.py
0   564   src/get_cut_time.py
107  2118  src/mergeVideo.py
60   1144  src/video.py
4    308   src/audioCorrelation.py
```

```sh
$ grep -on '\[[0-9]\]' src/frame_compare.py src/get_cut_time.py
src/get_cut_time.py:174:[1]   …:177:[1]  …:208:[1]  …:216:[1]  …:218:[1]
src/get_cut_time.py:224:[1]   …:246:[2]  …:252:[2]  …:321:[1]  …:431:[1]
```

**Every `[0]` subscript is absent from both files, while `[1]` and `[2]` survive
normally.** `da1c191` touched exactly these two files and nothing else
(`git show --stat da1c191`). This is one mechanical corruption across both files,
not three typos in one.

**Why that changes the ruling and not just the patch.** In `frame_compare.py` the
three sites are syntax errors, so the file announces its own damage. In
`get_cut_time.py` **eight sites compile and fail at runtime**, silently, and they
sit in the two functions objectives 006 and 008 would have depended on. Line
numbers below are as of `db92f0e`, before the RETIRED banner this task added
shifted them by +24; the banner itself names functions instead, so it cannot go
stale:

| line | as written | intended | consequence |
| ---: | :--- | :--- | :--- |
| 215 | `sum(p for p in points)` | `p[0]` | `0 + tuple` → `TypeError` |
| 217 | `sum(p * p for p in points)` | `p[0]*p[0]` | same |
| 218 | `sum(p * p[1] …)` | `p[0]*p[1]` | same |
| 224 | `slope * p + intercept` | `slope * p[0]` | same |
| 239 | `delay_fidelity_values[keys]` | `keys[0]` | unhashable `list` |
| 248 | `float(v)` | `float(v[0])` | `float(tuple)` |
| 286-7 | `_hhmmss_to_seconds(self.list_cut_begin_length[seg0])` | `[seg0][0]` | passes a `list` where a `str` is expected — `video.py:491` builds each entry as `[begin_str, length_str]` |

So `_detect_time_drift` (the drift regression — objective 3's instrument) and
`_summarize_segments` (the per-window fidelity summary — objective 2's instrument)
**both raise on their first call.** They have never run.

### 2. Why I keep the pHash front end

`RESTART.MD` §10's argument — the same cut scores differently at two bitrates, so
two scene lists do not align — I find sound, and it argues for content-derived
hashing over per-file scene scores. I reach the same conclusion by a different
route: a scene score is a property of *one* file, so two scene lists are two
independent quantisations of the same event and there is no reason their errors
cancel. A pHash distance is a property of *a pair of frames*, so it is a
comparison in the first place. That is the reason to keep hashing, and it is
mine rather than inherited.

`frame_compare.py:40-93` — the ffmpeg 32×32 grey extraction and the 64-bit DCT
pHash — is sound and I keep it. I read it line by line and found no defect.

### 3. Why I reject the alignment back end (`_align_and_find_gap`)

Four defects, none of which restoring `[0]` repairs. **Inferred from the source**;
(a) I regard as decisive.

**(a) It aggregates the wrong statistic, and discards the one that carries the
signal.** The loop is

```python
costs[best_j] += best_d      # line 125
hits[best_j]  += 1           # line 126   <- written, never read
...
center = max(range(m), key=lambda j: smoothed[j])    # line 141
```

Cost lands only on target frames that some reference frame *chose as its best
match*. Content **inserted** in the target is chosen by no reference frame at all,
so it accumulates cost 0 and `hits` 0 — and the function then returns the argmax
of the **cost** array. An insertion is precisely a run of `hits == 0`, and `hits`
is computed and never read anywhere in the file (`grep -n hits src/frame_compare.py`
→ lines 110, 126 only). The array that carries the answer is built and thrown
away, while the argmax runs on an array in which an insertion is indistinguishable
from a perfect match. This is not a typo; the aggregation is aimed at the wrong
quantity.

**(b) It silently examines only the opening of the window it was given.**
`n = min(len(ref_hashes), self.max_search_frames)` takes the **first** N frames.
`get_cut_time._refine_cut` passes `max_search = ceil(length*fps*0.30)`, so 70 % of
the window that was extracted, decoded and hashed is discarded and only its first
30 % is examined — and a confident answer is still returned. A change point in the
last 70 % cannot be found, ever. That is `AGENT.MD`'s armed guard that cannot fire.

**(c) The band is not a band.** `_refine_cut` passes `band_width =
ceil(length*fps*0.20)` while the sequences are truncated to 30 % of the window by
(b). A ±20 % band across a 30 %-long sequence admits nearly every `j` for every
`i`, so "banded alignment" degenerates into an unconstrained nearest-neighbour
search — which is where pHash false matches come from, and the band existed to
prevent them.

**(d) It returns the wrong type.** `find_scene_gap_requirements` returns
`{start_frame, end_frame, start_time, end_time}` — *where* it thinks something is
odd, in target coordinates. It never returns *by how many frames the two timelines
differ*. 006 needs the step size and 007 needs it to place the pieces, so the
output type does not fit either consumer. Also `start_frame = round(start_time *
self.fps)` uses the **decimation** rate: at the class default `fps=10` a "frame" is
a 100 ms tick, not a video frame. Nothing ties the output to the reference's frame
grid.

### 4. Why `get_cut_time.py` is retired rather than adopted

Beyond the six runtime breaks in §1:

**(a) It is wrong about `second_correlation`'s return, at four sites.**
`audioCorrelation.second_correlation` returns `(file, offset)` where `offset` is
already **seconds** (`data['offset_seconds']`, or `xmax/fs` in the fallback,
`audioCorrelation.py:267,292,295`). `get_cut_time` does
`float(res[1]) / float(fps)` at lines 177, 208, 321 and 431 — dividing seconds by
~24, wrong by that factor — under a comment (line 174) asserting a *"convention
projet"* the project does not use. **The live consumer proves the correct usage:**
`get_delay_second_method_thread.run` (`mergeVideo.py:161-168`) uses `result[1]`
directly and reads `result[0]` — the filename — to recover **the sign**.
`get_cut_time` ignores `res[0]` at all four sites, so it cannot know the sign of
any offset it measures.

**(b) It re-runs the correlation that just failed**, which `SPEC_ZONE_A.MD` §1
exists to make unnecessary, and re-runs it *differently*: `_detect_global_offset`
and `_detect_time_drift` extract raw, **un-normalised** audio, while every audio
the pipeline correlates is loudness-normalised to −23 LUFS
(`audioCorrelation.generate_norm_cmd`). And it calls `get_delay_fidelity(main,
target, self.lenghtTime)` where the live caller passes `self.lenghtTime*2`
(`mergeVideo.py:262`) — a different window length, therefore a different quantum
(§5), therefore different numbers.

**(c) Its repair half is out of contract and is not mine.**
`_build_chimeric_audios` hardcodes a 5.0 s "after" segment (line 490) and a 0.5 s
"before", so it builds a ~5.5 s stub rather than a track; `_build_final_mkv` writes
its own container, bypassing zone A's requirement to rejoin through
`best_video.sameAudioMD5UseForCalculation`. That territory is dev-2's (007/009)
in any case.

**What is worth salvaging:** `_scene_refine_window` (15 lines, a usable scene
probe) and the *shape* of `_find_all_bad_ranges`. Both are cheaper to rewrite than
to inherit.

**A consequence I am creating and will not hide.** Repairing `frame_compare.py`
makes `import get_cut_time` succeed where it previously failed at import time —
so the repair makes the retired file *more* likely to be picked up, not less. I
have therefore added a RETIRED banner at the top of `get_cut_time.py` naming the
four defects, and I have **not** restored its eight `[0]` sites: a file that
raises is safer than one that runs and is wrong by a factor of 24 with no sign.
Whether it is deleted is the owner's call, not mine.

### 5. The correction that made me rule this way — the audio quantum is not 125 ms

`BRIEF_COMMON.md`, `TASKS/006` and `TASKS/008` all state *"One point = 125 ms, the
chromaprint frame. Every delay you see is a multiple of it."* **Measured, that is
true for this corpus's episode length and false as a general rule.**

`audioCorrelation.py:132` computes the quantum per call:

```python
return get_max_corr(corr, source, target, span,
                    int(lengthFile/len(fingerprint_source)*1000))
```

It is not the chromaprint frame, which is `(4096/3)/11025` = **123.84 ms**. It is
`lengthFile / n_items`, and `n_items` is short of `lengthFile/0.12384` by a
roughly constant ~20 items, which pulls the ratio *above* the frame by an amount
that depends on `lengthFile`. **Measured** on a window from the `ore1` example
pair (`fpcalc -raw -length L`, items counted, `int(L/n*1000)`):

| `lengthFile` L (s) | 20 | 24 | 40 | 60 | 100 | 160 | **214** | 288 | 436 | 700 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quantum (ms) | 142 | 139 | 132 | 129 | 127 | 125 | **125** | 124 | 124 | 124 |

Band edges, measured by sweeping L in 4 s steps: the quantum is **125 ms exactly
for L ∈ [160, 286]**, 126+ below it, 124 at and above 288.

The pipeline passes `lengthFile = lenghtTime*2` and
`lenghtTime = int((shortest_audio_duration − 240)/11)` for durations > 540 s
(`video.generate_begin_and_length_by_segment`), so L ∈ [160, 286] corresponds to
**shortest audio duration ∈ [1120 s, 1824 s)** — about 18.7 to 30.4 minutes. The
`ore1` pair is 1420 s, inside that band, so it reads 125. A 45-minute episode
(≈2640 s) gives L = 436 and a quantum of **124 ms**; a short file on the
`> 60 s` branch reads 132–142 ms.

This is **not a new discovery** — campaign 1 recorded it, and I am carrying their
hedge rather than restating it more strongly. `CAMPAIGNS/01-dev-AI/DOMAIN_INVARIANTS.MD`
§7: *"~125 ms typically, **138 ms observed**"*; `WORKLOG.MD`: *"917 of 933 delay
vectors have a self-consistent ms-per-frame, **859 of them at 125 ms** … three
files in this library have a 138 ms frame."* My sweep gives 139 ms at L = 24 s,
which is the mechanism behind their 138 and is consistent with it. What I add is
the **exact band** and the fact that it is a function of episode duration.

**Why this belongs in the 005 ruling.** It is the reason the video instrument has
to return a *frame displacement* rather than a refined time: the audio bracket's
width is a per-pair quantity that must be read from the pair, and at 124–142 ms it
is 3 to 3.4 frames at 23.976 fps. Any code — mine or dev-2's — that hardcodes 125
is right for 24-minute episodes and quietly wrong elsewhere.

### 6. The caller I intend, agreed with dev-2

`AGENT.MD`'s failure mode is a module nothing calls, so the caller is named before
the code. dev-2 proposed the interface first
(`VMSAM_HELP_AI/dev-2/INTERFACE_dev1_dev2.md`, v1) and I accept it with the
corrections in §5 above and in my reply. Shape:

- **new module, mine**, `src/` — imported by dev-2's repair module, lazily, behind
  the off-by-default `config.ini` `[features]` key that is dev-2's to add;
- it returns **numbers, not files**: a partition of the master's timeline as
  `{master_start_ms, master_end_ms, candidate_offset_ms}` under the stated
  equation `candidate_time_ms = master_time_ms + candidate_offset_ms`;
- exactly **one caller**, and it is dev-2's zone-A path. Nothing else imports it.

`frame_compare.FrameComparer`'s pHash front end is called by that module. That is
what makes repairing it worth doing rather than merely tidy.

## What changed / measured / dropped / blocked

### Changed

- `src/frame_compare.py` — restored the three `[0]` list literals at lines 109,
  110, 130. Open file, `WRITE_ZONES.MD` §4. Nothing else touched.
- `src/get_cut_time.py` — added a RETIRED banner to the module docstring naming
  the four defect classes. Open file. **No code changed**, deliberately: see §4.

### Measured

- `[0]` absent from both dormant files, present at normal density elsewhere —
  commands in §1.
- The chromaprint quantum as a function of window length, and the exact band in
  which it is 125 ms — table and commands in §5.
- `python3 -m compileall -q src -x '__pycache__'` → **exit 0**, no failures.
  This changes the expectation stated in `AGENT.MD` § Testing (*"must fail on
  exactly one file: `frame_compare.py`"*) and in `WRITE_ZONES.MD` §5 check 4.
  Those documents are frozen; the change is intended and is this task's stated
  acceptance condition, so it is recorded here and reported, not edited.

### Dropped

- **`_align_and_find_gap` as the alignment method** — §3. Its pHash front end is
  kept.
- **`get_cut_time.py` as a starting point for 006** — §4. I will not inherit its
  drift regression, its segment summary, or its chimera builder.
- **Restoring `get_cut_time.py`'s eight `[0]` sites** — §4, last paragraph.
  Making it run without fixing the `/fps` and sign errors would make it more
  dangerous, not less.

### Blocked

Nothing in 005 is blocked. Two findings raised for the owner rather than edited,
both recorded in `AGENT_NOTES.MD`:

1. the ten hardcoded `127`/`128` one-frame gates in `first_delay_test`, which
   encode "within one frame" only for a 125 ms quantum;
2. `get_delays_dict(…, delayUse=0)` at `mergeVideo.py:489` and `:502`, which
   record a post-shift residual as if it were absolute.

Both sit in `mergeVideo.py` **outside any tagged zone**, so by `WRITE_ZONES.MD`
rule 1 they are frozen and are findings, not edits (§6).
