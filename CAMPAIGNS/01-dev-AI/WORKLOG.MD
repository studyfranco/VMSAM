# VMSAM — Agent Worklog

> **Campaign 1's worklog, archived.** Content unchanged except one library path,
> replaced by `<LIBRARY_ROOT>/<show>/<season>` under `AGENT.MD` § Privacy.

Chronological log of architectural decisions taken by AI agents. Newest entry first.

---


## 2026-08-30 — Chimeric integrity: the gap patch was chosen by language tag, and 12 shipped builds carry the wrong one (Agent 1)

**Exposure, measured before anything was changed.** Both gap-patch selectors in
`merge_video_chimeric.py` matched on the language tag alone, and
`_pick_master_fallback` fell through to `master["audio"][0]` when the tag missed.
Replaying `plan_segments` over the plateaus each build actually used — no
re-merge, the plans are deterministic — gives:

| Surface | Records | Draw a gap patch from the master | Patch a track with a DIFFERENT language |
| :--- | ---: | ---: | ---: |
| Chimeras `error_correct.py` actually built | 29 | **15** | **12** |
| C_cut plateau table, gated as production gates it | 149 of 180 | 133 | 63 |
| Merges produced by the deployed container | 20 | **0** | **0** |

The container column is zero and that is not luck: the chimeric path never fired
in any deployed merge (`Recovery declined … is not uniform` in the retry logs),
confirmed independently by correlating a shipped C_cut output — its Spanish
track reads 0.95 against the raw candidate at constant lag, so no re-assembly
happened. **The defect is in the local repair tooling today and would ship the
moment the cut-point table becomes reachable in the container**, which is the
next step on this agent's roadmap.

The 12 cross-language builds patch, for example, a Latin-American Spanish track
and a Brazilian Portuguese track with the master's *German* audio, for 30–360 s
each. Every one of them passed `verify_chimera`.

**The rule, and why it is not a metadata rule.** `title` is empty on most of
these tracks and inconsistently punctuated on the rest; codec and layout are
shared across unrelated dubs. The codebase already answers the question by
signal: `find_differences_and_keep_best_audio` decides whether two same-language
tracks are the same recording with Chromaprint through
`audioCorrelation.correlate`, and its bar is `mean(fidelity) >= 0.90`. That
instrument and that bar are reused — 3 cuts of 30 s taken on the longest
plateau, so both streams are compared where they carry the same content.

Measured over **217 real stream pairs across the 15 master-patched builds**:

```
same recording      n=32    0.904 .. 0.993
different recording n=185   0.563 .. 0.847
```

an empty band 0.057 wide around the inherited bar. **Nine of those pairs carry
the same language tag and are not the same recording** (0.606–0.811) — exactly
the cases a tag rule gets wrong.

*False negatives*: a same recording re-encoded so hard it drops below 0.90, or a
plateau shorter than 120 s to sample on. Both refuse a patch rather than take a
wrong one. *False positives*: two releases carrying the identical dub — which is
not an error, it is the same recording. The residual risk is a region with no
dialogue: end credits fingerprint alike across dubs because only the music is
there. That is why the cuts are taken on the longest plateau rather than at the
gap.

**What changed** (`src/merge_video_chimeric.py`, nothing else):

- `DubGate` — the fpcalc comparison, cached per stream pair, **fails closed**;
- `pick_patch_source` — confirmed same dub → a recording both releases
  demonstrably share in *another* language → refuse. Never a same-language track
  the fingerprint did not confirm;
- `_alt_candidate_source` gated the same way, and inert without a gate;
- `_pick_master_fallback` kept **only** as `verify_chimera`'s measurement
  reference, which is a different question and was always correct there;
- `describe_chimeric_track` writes a `CHIMERIC=` stream tag on every altered
  audio and subtitle track, naming the composition and carrying no path or title;
- `verify_chimera` now reports `candidate_coverage`.

**Second finding, from the same replay.** `verify_chimera` can pass a build that
is mostly master audio, because a master-patched region correlates perfectly
against the master by construction. `err_196` was recorded `in_sync` with
**879 s — 62 % of its only track — of straight master audio**, on a plateau table
whose peaks are 0.377/0.392. Candidate coverage is now reported; no bar was set,
because the fingerprint gate already refuses that build (0.606) and setting a new
pass bar without measuring it against the 89 judged files is how verdicts drift.

**Target-first, `err_220`.** The expected plan was written out by hand from the
measured fidelities before the assembler was run; the code reproduced it on all
five streams. Built in `/tmp` and verified: same-language residuals −7.2 and
−9.8 ms at peaks 1.01/1.06, duration error 3.5e-5, `CHIMERIC=` present on all
five audio streams, output-spa vs master-eng 0.994 inside the patch and
output-spa vs candidate-spa 0.998 outside it.

**Scene-cut cross-validation of the golden pair (success criterion 2).** Scene
cuts either side of `err_220`'s change point, mapped through each plateau's lag:

| Region | Lag that fits the video | Per-cut error |
| :--- | ---: | ---: |
| before the change point | 0.000 s | 1 ms |
| after the change point | +5.006 s | 1 ms |

Audio says −0.133 s and +4.872 s, so **the two instruments agree on the step to
1 ms** (5.005 vs 5.006) while their absolute offsets differ by a constant
133 ms — the audio-to-video skew this file documents at 0.4–99 ms, here 133 ms,
constant across the change point and therefore cancelling in the step. Exactly
the behaviour criterion 2 predicts.

*Instrument note*: `showinfo` logs at INFO level, so `ffmpeg -v error … select
… showinfo` silently reports **zero** scene cuts everywhere. The first pass did
exactly that and read as "no cuts to cross-validate". Calibrate against a busy
stretch (17 cuts in 60 s here) before believing an empty result.

*Open finding, not fixed here*: the table places `err_220`'s change point at
898–903 s, while the video puts it between 914 s and 932 s. Up to ~30 s of the
assembled track therefore reads the wrong plateau, and nothing looks there —
`verify_chimera` samples plateau midpoints (449 s and 1158 s on this file), never
the change point. Cross-validating change-point *location* against scene cuts is
the obvious next refinement to the plateau tooling.

**Both halves, `err_15`.** Candidate a7/jpn (0.808 against the master's jpn) is
blocked and rerouted; candidate a8/jpn (0.985) keeps its same-dub master patch.
Same language, same file, opposite verdicts, decided by signal. `err_196` now
refuses outright.

**Cost, measured not estimated.** 1.6–5.8 s per merge, 1–5 fingerprint
decisions — never the full stream cross-product, because verdicts and extracted
cuts are cached per pair. 4.0 s / 2 decisions on a 5×4 stream pair; 5.8 s /
5 decisions on 11×3. Roughly 0.3 % of a chimeric build.

Tests 427 → 449 (`src/tools_batch/tests/test_15_dub_gate.py`, 22 cases). Privacy
clean. All 35 protected closure names byte-identical to HEAD by AST body
comparison; `src/mergeVideo.py` not touched. Not committed — Agent 3 request T1
is queued in `src/TASK.MD`.


## 2026-08-30 08:40Z — The multi-track reference cannot be measured; the language tag can

Agent 2 (`D_trim` / `B_fps` / `E_corr`). Full detail and every number in
`src/task_archive/20260830-0840Z-multitrack-reference-and-language-tag.md`.

### 1. The 17 `REC_DECLINE_MULTITRACK` pairs — the conflict is real

Verified before choosing a rule. On err 6 the master's own two `ja` tracks are
**130 ms apart at peak 0.98** — the same recording muxed twice — so the
candidate correlates with **both** at ≈0.99 (+70 ms and −60 ms, spread 0.0 over
three windows). The instrument read 0.0 ms at peak 1.0 against every file
itself, on every window, before any of this was believed.

**Three arbiters were tried and all three failed.** *Peak height*: 0.995 vs
0.99. *A second common language*: works on err 6, and on err 20 the master's
**seven** dubbed languages agree at **−180 ms**, matching neither `ja` reading —
because that master's `ja` pair is displaced from its own dubs (cross-checked
inside each file: the candidate is coherent at 0 ms, the master is not). *The
picture*: `phash_offset` reads **+5.0 ms** on the pair whose two audio readings
are +70 and −60 — their exact midpoint, and inside the 0.4–99 ms audio-to-video
skew the 04:45Z entry already measured.

So the reference is **declared, not measured**: the master's uniquely
`default`-flagged track in the correlation language, falling back to the first —
which is the track `get_stream_order` already returns and the one the verifier
compares against. Not merely "the first": on **17 of 17** masters the default
flag and the first entry agree.

Measured end to end on the real media through the production functions:
**15 of the 17 reduce to one constant delay** (ten at +80.0 ms, five at 0.0 ms,
every one with a spread of exactly **0.0 ms** across three windows). err 170 and
err 349 still decline, **correctly** — 170 has a genuinely piecewise lag
(−2010 / −2020 / −3020 ms) and 349's candidate never reaches the 0.30 peak floor.
Naming the reference removes only the requirement that the master's own tracks
agree with each other, which they cannot; it does not lower the uniformity bar.

### 2. The 6 `NO_COMMON_LANGUAGE` pairs — the proposal holds, with one correction

Each side carries exactly one audio track, tagged `ja` on the master and `en` on
the candidate, and they are the **same recording**: a constant −4430 to
−4470 ms across three windows, spread ≤ 20 ms, median peaks **0.90–0.96**,
calibration 0.0 ms at peak 1.0 on all 18 windows. Independent corroboration the
correlator did not supply: the measured lag matches each pair's **duration
difference** (−4.4 to −4.6 s) — the candidate carries a leader the master lacks.
A real dub rides on music and effects and peaks at 0.5–0.85
(`DOMAIN_INVARIANTS` §8); these do not.

**The correction**: the proposed flat "≥3 probes at peak ≥ 0.85" **rejects
err 679**, whose middle window peaks at 0.77. Shipped as **≥3 windows at
≥ 0.75, median peak ≥ 0.85, spread ≤ 40 ms** — admits all six, still declines a
synthetic dub at 0.79/0.80/0.81. Second correction: the incumbent 60/480/900 s
window starts fit **one** window into a six-minute short, and one window cannot
show a lag is constant, so the fallback sizes its windows to the file.

The override is exactly as narrow as proposed: the candidate's single audio
entry is re-keyed onto the master's language **in the in-memory `video` object**
and nothing else. No media is written, no database row is touched, no delay is
set here. The output's tags come from `get_mediadata` re-probing the produced
file, so the track is still published as what its container says it is.

### 3. Both are opt-in and off by default

`tools.multitrack_reference` (`VMSAM_MULTITRACK_REFERENCE=default`) and
`tools.language_tag_fallback` (`VMSAM_LANGUAGE_TAG_FALLBACK=measured`), in the
shape of `new_delay_method` / `video_align_method`. With them unset the old
branches run unchanged — asserted **behaviourally**, not just structurally:
a test drives the real `recover_not_compatible_videos` over the err 6 numbers
with the flag off and requires it to still decline as `reference is ill-defined`.
46 new pure-unit checks in `src/tools_batch/tests/test_13_multitrack_reference.py`.
Protected closure re-verified: **35/35 names present, 0 AST-changed**, 30
reachable from `generate_launch_merge_command`, nothing newly reachable.

### 4. What blocks these 21 files from `Resolved`

**Step 2 is unreachable from inside the repository.** Both flags come from the
environment and `showgestionar-test` has no way to set them: the `Dockerfile` is
read-only and the compose environment lives outside the tree. Re-triggering
these files against the deployed image proves nothing about the new paths until
those two variables are present in the container. That is the owner's call.

---

## 2026-08-30 04:45Z — Video alignment measured against audio: exact, unbiased, and still the wrong number

### 1. The question, and the answer

The project owner asked whether a perceptual-hash alignment of the *picture*
could give a better offset than the audio correlators, especially where they are
weak; mid-task he added **scene-change detection** as a second candidate in the
same phase. Both were built, calibrated and measured on identical spans against
the offset the incumbent and GCC-PHAT already agree on.

**The picture measures the offset exactly and it is still not the number the
merge needs.** On the control corpus pHash returns **1001.0 ms — exactly 24
frames — on every window of every pair, spread 0.0 ms**, and the scene vote,
which uses no grid at all, independently lands within 0.9 ms of it. The audio
answer is 954.6–955.9 ms. The gap is a constant **45–46 ms**, and it is not
instrument error: on the `webweb` control, where the true offset is 0 by
construction, pHash reads **exactly 0.0 ms** and the scene vote −0.9 to 0.0.

The gap is the release pair's own **audio-to-video skew**. `generate_launch_merge_command`
re-times *audio* tracks; the number it needs is an audio-to-audio delay, and
nothing keeps one release's audio in the same phase with its picture as
another's. Measured across ten unrelated release families, that skew ranges from
**0.4 ms to 99 ms**. A video-derived delay carries it as a straight error.

### 2. The cross-language case, which was the whole argument for video, does not rescue it

One control candidate carries **eleven dubs** against a Japanese-only master, so
every dub is a cross-language read whose correct answer is known — the
Japanese-against-Japanese read on the same files and windows. Over **15 reads /
11 distinct language tags, 9 distinct offsets**: median |error| **21.4 ms**,
p90 33.4, **max 44.0**, ten of fifteen over the 15 ms bar. Peak-to-sidelobe falls from 51 to 9–25 but never
collapses, which is exactly the trap `DOMAIN_INVARIANTS` §8 describes.

So audio at its *worst* (44.0 ms) is still better than video at its *best*
(45.1 ms) — and when the languages match, audio is right to under a millisecond
while video is still ~46 ms out. On the five `REC_DECLINE_MULTITRACK` files
where the pictures are frame-identical, the skew is **84.5 ms**, reproducible to
0.04 ms. **Phase 1: no.**

The six `NO_COMMON_LANGUAGE` files (error ids 678–682, 684) are the only pairs
where the audio path yields nothing — and that is a *gating* failure, not a
measurement one: all six are the same recording with a wrong language tag. On
this corpus there is no pair whose audio genuinely cannot be correlated because
the dubs differ.

### 3. Three defects found by calibrating instead of assuming

- **`src/frame_compare.py` did not compile.** `costs =  * m` — a starred-expression
  `SyntaxError` from `da1c191`, on `dev-AI` **and** `master`, unnoticed for a
  year because the module is dead code: nothing imports `src/get_cut_time.py`,
  its only importer. Fixed, and now covered by a test.
- **ffmpeg's `fps` filter anchors on the first decoded frame, not on `-ss`.** A
  window asked for at 600.000 s of 23.976 fps content starts at 600.016 s, so
  two windows a non-integer number of frames apart are sampled on grids that do
  not line up. Cost, measured: four of nine control pairs disagreed with
  themselves by one frame and two reported 1042.7 ms where the answer is 1001.0.
  `snap_to_frame_grid` fixes it; afterwards every window of every pair agrees
  exactly. **I first blamed this on animation shot "on twos" and was wrong** —
  recorded as wrong in the module and in `GridSnapTests`.
- **A frame grid does not always start at zero.** 6 of the 34 media files
  touched here report `start_time=0.023000` on their video stream, so their
  whole frame grid is shifted 23 ms and a phase-blind snap lands between two
  frames. The symptom was the two video methods disagreeing by exactly one frame
  where they had agreed everywhere else. `snap_to_frame_grid` now takes the
  stream's `start_time`. The scene-cut method never had the problem, because it
  compares real PTS rather than a resampled grid — an advantage of the sparse
  method that nobody predicted.
- **An off-by-one in the new DP's insertion pointer.** `prefix[d]` is
  `opening[d-1]`, so the band column that opened a candidate-side gap is `d-1`;
  storing `d` shifted every *insertion* by exactly one frame while leaving
  *deletions* correct. Caught only by adding the mirror test. It had already
  contaminated a phase-2 result: error id 44's video step read 667.3 ms against
  the audio's 627.4, and after the fix reads **625.6 ms** — 39.9 ms of
  disagreement that was entirely mine.

### 4. Phase 2 was earned, on a different property

The skew cancels in a **step** (`lag_after − lag_before`) whenever the edit takes
the same duration out of both modalities, so the picture can cross-check a
structural edit even though it cannot supply a delay — and where it does not
cancel, that is itself an observation about the edit. The DP is
banded, affine-gapped, vectorised, and **returns the path**; the abandoned
branch's version could not gap at all, because `64 − hamming` is never negative
against a gap penalty of −1. On an unedited control pair it returns one plateau
and zero gaps; on 2877 frames with a ±3 s band it runs in 0.06 s.

Against the plateau tables, over 19 change points on 8 error files (15 of which
produced a comparable video step), it reproduces the audio's step to a **median
of 3.0 ms** — 9 of the 15 land inside 3.5 ms, 10 inside one frame — and
**contradicts it at 5**, twice with evidence far stronger than the table's. The two
sharpest: **error id 82** — the table says a 6-frame edit at ~691.5 s, the
picture says a **4-frame edit at 680.01 s**, one clean step across a 240 s
bracket at mean peak 0.994/0.995; and **error id 97**, the same shape in the
other direction (table 4 frames at ~889.5 s, picture **6 frames at 893.56 s**).
Both audio cut times came from a coarse 90 s scan and are not localised. **Error
id 129** confirms a step's size to 1.4 ms while placing it 35 s later than the
table does.

`path_to_plateaus` emits the `{start_s, end_s, lag_ms, mean_peak}` shape
`merge_video_chimeric.normalize_plateaus` already consumes, and a test runs the
real `normalize_plateaus` over the real DP output.

**One overclaim withdrawn before publishing it.** I first read error ids 82 and
97 as having the *wrong edit size* in `cut_points_c.json`. The table's own
evidence does not support that: its lags there are 17 windows agreeing to 0.5 ms.
Both measurements are strong and both are right — the audio removed 250.1 ms and
the picture removed exactly 4 frames (166.8 ms), so the release's own A/V sync
moves 83.3 ms across that edit. **The merge needs the audio number**, so the
table is right for what it is used for. What stands is the *location*: those cut
times are coarse 90 s-scan estimates flagged `localised: false`, and the picture
puts a single unambiguous step at 680.01 s and 893.56 s. Error id 129 @1025.6 s
is a genuine conflict between two localised estimates (size right to 1.4 ms,
location 35 s apart) and needs a third observation. `gaps --ids <id>` is a
~20 s-per-change-point, language-free gate on any plateau table.

### 5. Cost, and a hypothesis that did not survive

Scene detection was expected to be dramatically cheaper on RAM because it is
sparse. It is not: **225 MB (pHash) vs 224 MB (scene)** median peak RSS over 29 pairs, including
ffmpeg. Both must decode every frame; the hash storage is 46 KB. The real
difference is 20 % of wall time from a simpler filter graph. Against the audio
path the video path costs ~2.3× the RAM and about the same wall time once the
audio path's shared cut extraction is counted.

The scene detector was also calibrated: two decodes of the same window return
**identical** cut lists, and the recovered offset is **stable at 1001.0 ms across
thresholds 0.25–0.70** — only the supply of cuts changes. Supply is not a
constraint on this corpus: over 29 pairs / 116 windows, a 60 s window carries a
median of **15** hard cuts (min 3, p90 40) and only **1 window of 116** has fewer
than four, with the pooled vote's peak standing a median of **52×** above its
background.

### 6. Status

Opt-in and off by default (`tools.video_align_method`, `VMSAM_VIDEO_ALIGN=phash`).
`video_align_visual` has **zero importers** in `src/`. The
`generate_launch_merge_command` closure is 51 functions at HEAD and 51 in the
tree with identical AST bodies. Suite 307 → **348**, green. Full detail,
including what was not measured, in
`src/task_archive/20260830-0430Z-visual-alignment.md`.

---

## 2026-08-30 00:45Z — The per-cut head-to-head, and a decode that read the wrong track

### 1. What is being compared, and why the constancy verdict is the only question

VMSAM does not decide "is this a delay or an edit?" by looking at an offset. It
cuts both files into `video.number_cut` parts, measures one offset per part, and
asks whether that vector is CONSTANT. So a fair comparison between the incumbent
correlator and GCC-PHAT has to be per-cut, on the SAME cuts, judged by the SAME
gate. `src/tools_batch/gccphat_compare.py` does exactly that: the cut table comes
from the real `mergeVideo.prepare_get_delay_sub`, the cuts are extracted by the
real `video.extract_audio_in_part`, and both vectors go through
`is_within_one_fingerprint_frame` unchanged. Three measurements per pair:

| name | what it reads |
| :--- | :--- |
| `incumbent` | `audioCorrelation.correlate` on the extracted cut files |
| `gcc_files` | GCC-PHAT on the SAME extracted cut files -- identical bytes in |
| `gcc_native` | GCC-PHAT reading the two source files directly at the same timestamps |

A disagreement on the constancy verdict is a behaviour change. A disagreement of
three milliseconds is a performance note. The tables are built by
`src/tools_batch/gccphat_report.py`, which also carries an independent arbiter
(plain normalised cross-correlation on the waveform, plus a 10 ms RMS envelope)
so a disagreement can be settled by measurement rather than by preferring the
newer method.

### 2. A decode with no `-map` reads whatever track ffmpeg feels like

`audio_offset_gccphat.decode_to_raw` built its ffmpeg command without `-map`, so
the direct-from-source variant decoded ffmpeg's default audio track. On a
control-corpus release that carries **French first and Japanese second**, that
correlated French against the master's Japanese and reported a perfectly
confident **-933.26 ms with a spread of 0.008 ms across ten cuts** -- a number
that looks like the tightest measurement in the whole run. The language-matched
answer is **-954.61 ms**, which is what the incumbent (Chromaprint), `gcc_files`
(GCC-PHAT on the extracted cuts) and the independent arbiter all say; the
arbiter puts it at **-954.607 ms, spread 0.0016 ms**.

The two numbers are 21.6 ms apart, which is a fifth of a fingerprint frame and
enough to move `adjust_delay_to_frame` by a whole video frame at 23.976 fps. The
first hypothesis was a bias in the pipeline's own extraction; that was tested by
extracting the same window three ways (full decode, input `-ss`, and the
pipeline's `-copyts` + output `-ss`) and the three agreed to within 0.5 ms. The
extraction is innocent. Only the track choice was wrong.

Two things make this worth recording beyond the one-line fix:

- **The confidence was not merely high, it was the highest.** Peak-to-sidelobe
  said the French-vs-Japanese lock was excellent, because music and effects DO
  align between two dubs of one episode (DOMAIN_INVARIANTS 8). The normalised
  peak tells the real story -- 0.61 cross-language against 0.98 same-language --
  and a phase-transform confidence does not.
- **Nothing in the output said which track had been read.** A measurement that
  cannot name its own input cannot be audited.

Fixed: `decode_to_raw` takes a `StreamOrder` and emits `-map 0:<n>`;
`measure_cut_vector` and `refine_step_seconds` forward `stream_a`/`stream_b`; the
harness passes the `StreamOrder` that `extract_audio_in_part` puts at index 0.
`test_09_gccphat_cuts.TrackSelection` covers the argv, the no-stream default and
the forwarding. **Production was never affected**: `compare_video.
test_if_constant_good_delay_new_method` calls `measure_cut_vector_from_files`,
which reads the pipeline's own per-cut files and never opens a source file.

### 3. The control corpus is three pairs, not nine

Each `/config/example/<episode>` folder holds one BDRip and three WEB releases,
and the harness pairs each web release against the BDRip -- nine pairs. **They
are not nine measurements.** Decoded to mono 8 kHz, the correlation-language
track of all three web releases in a folder is *bit-identical*:

| folder | web1 / web2 / web3 audio MD5 | master audio MD5 |
| :--- | :--- | :--- |
| A | `7cbaabc8...` (all three) | different |
| B | `ffe4735b...` (all three) | `6a66c74e...` |
| C | `c872be14...` (all three) | `29babef3...` |

The three releases differ in video bitrate and in how many dubs they carry; they
mux the same audio. So the control corpus supplies **n = 3 independent audio
pairs**, and any accuracy or constancy claim quoted "over 9 pairs" is three
answers written down three times. The cost figures are still nine real runs --
extraction and correlation genuinely happened nine times -- but the *answers* are
three. Anything that pairs web against web is 0 ms by construction and is not a
measurement at all.

### 4. error-102: the incumbent is right, and the new method as wired would ship the bug

The first disagreement in the error corpus is the exact case DOMAIN_INVARIANTS 7
warns about, caught live.

| cut (start s) | incumbent (125 ms grid) | GCC-PHAT raw |
| ---: | ---: | ---: |
| 0-5 (120-670) | 0 | -20.00 ms |
| 6,7,9 (780-1110) | -125 | -186.85 ms |
| 8 (1000) | **-250** | -186.85 ms |

An independent instrument -- normalised waveform cross-correlation, no phase
whitening, plus a 10 ms RMS envelope leg -- puts the same file at **-20.0005 ms
before and a step of -166.84 ms after window 5**, envelope leg -20.34 ms and
167.93 ms. The file is genuinely edited: ~167 ms of content differs between
670 s and 780 s.

- The **incumbent declines it, and is right**, though its third distinct offset
  (-250 at cut 8) is a fingerprint error, not the edit: -186.85 ms quantises to
  -125, not -250.
- **GCC-PHAT measures the edit correctly and then throws the answer away.**
  `is_within_one_fingerprint_frame` sees frames {0, -1}, span 125 ms, and says
  "constant"; `test_if_constant_good_delay_new_method` treats that gate as the
  deciding vote and returns the median -20 ms. The merge would shift the whole
  candidate by -20 ms and leave its last third 167 ms out of sync.

The gate is not wrong -- it exists to tolerate *fingerprint quantisation*, and a
167 ms edit is 1.34 frames, which at Chromaprint resolution is indistinguishable
from noise. GCC-PHAT has no fingerprint and no quantisation: its own
`describe_cut_vector` already said `stepped, step_ms=-166.85, between 670 and
1000 s`. **Deferring to a coarser instrument's tolerance when you hold the finer
measurement is the defect.** A method that can see 0.005 ms should not ask a
125 ms grid whether it is allowed to believe itself.

**Fixed in the tree** (not committed). `describe_cut_vector` now reports
`step_confirmed`: True only when at least two cuts sit either side of the change
point and each side is flat to within the same tolerance the uniform test uses.
`test_if_constant_good_delay_new_method` declines on a confirmed step regardless
of the frame gate, and still lets the frame gate decide an UNconfirmed one --
one cut disagreeing with nine is exactly the noise that gate exists to forgive.
Six tests in `test_09`, including the error-102 vector reproduced synthetically
and the decision rule evaluated directly. The default path is untouched: none of
this is reachable unless `VMSAM_OFFSET_METHOD=gccphat`.

### 4b. Both methods are wrong somewhere, and the failures do not overlap

Three disagreements decided by the independent arbiter in the first 13 error
pairs. They point in opposite directions, which is the useful part.

| error id | incumbent | GCC-PHAT | arbiter (waveform / envelope) | who is right |
| :--- | :--- | :--- | :--- | :--- |
| 102 | declines (3 offsets, span 250 ms) | constant by the frame gate, `stepped` by its own reading | −20.0005 ms then a **−166.84 ms step** after window 5; envelope −20.34 / 167.93 | **incumbent** |
| 104 | declines (3 offsets, span 250 ms) | same shape | −20.0003 ms, step **−166.84 ms** after window 5 | **incumbent** |
| 110 | declines (4 offsets, span **219 125 ms**) | `uniform`, +23.235 ms, spread 0.005 ms | **+23.2325 ms, spread 0.0033 ms** over 7 trusted windows | **GCC-PHAT** |

error-110 is the case for the new method. Seven of the incumbent's ten cuts read
0 ms and three read ~217-219 **seconds** -- gross false locks, not quantisation.
The file is a 23 ms delay, comfortably inside one frame, and the merge declined
it. GCC-PHAT and two independent correlators agree to 0.005 ms.

error-102/104 is the case against it, and both share a master with error-48, so
they are one episode measured twice rather than two independent results.

### 4c. GCC-PHAT cannot see the offsets this corpus actually contains

The Welch frame is 10 s zero-padded to 20 s, so the lag axis runs **+/-5 s**.
In the first 13 error pairs, **7 produced no usable measurement at all**, and
only one of those was a confidence problem:

| cause | n | what it means |
| :--- | ---: | :--- |
| `out_of_range` | 6 | the incumbent measured 23-30 s, once 212 s. Not a weak peak -- off the axis. No confidence floor can rescue it. |
| `low_confidence` | 1 | a **drifting** lag: within one 604 s cut the offset moves 250 ms, every 10 s Welch frame sees a different lag, the average smears and the peak-to-sidelobe ratio falls under 2.0. |

The incumbent has no such ceiling: its Chromaprint pass searches the whole
fingerprint. Whatever else is true, **the new method cannot replace it on this
corpus without a coarse stage in front of it** -- the very structure the
incumbent already has (fingerprint coarse, fine correlator after). The
`low_confidence` behaviour, by contrast, is correct: a window whose lag drifts
does not have one lag, and refusing to name one beats naming the wrong one.

### 4d. What this comparison does NOT exercise: the 138 ms fingerprint frame

Stated up front so nobody quotes these tables wider than they reach. Measured
`sizePoint` across the pairs this run has completed: **125 ms on all but one,
124 ms on one**. That is not a new distribution -- the gate-regression run
already measured all 929 recorded vectors (125 x857, 124 x49, 126 x6, 138 x3),
see `src/task_archive/20260829-0100Z-gate-regression-test.md`, which also records
that **all three 138 ms vectors belong to the same show**.

So the 138 ms case -- the one the frame gates were widened *for* -- is
**unexercised by this comparison**, exactly as it was unexercised by the earlier
offset benchmark. Any claim here about how the two methods behave on a 138 ms
frame is an extrapolation. The quantisation code is covered by unit test
(`test_09.FrameQuantisation` asserts the grid changes the answer, which is why it
is a parameter), but no 138 ms *pair* has been measured end to end by either
method. Closing that needs the three files named in the gate-regression archive
run through `gccphat_compare.py` explicitly; it will not happen by sampling more
of the error corpus, because the corpus does not contain them.

### 5. Default path, verified rather than asserted

- `generate_launch_merge_command`'s closure -- 77 members by call-graph including
  classes, a superset of the 35 documented names -- has **byte-identical AST
  bodies** against `b7dd5cb` (the last commit before the GCC-PHAT work) and
  against HEAD. Script: closure walk over `mergeVideo`/`video`/`tools`/
  `audioCorrelation`, `ast.unparse` round-trip, `ast.dump` compare.
- `mergeVideo.py` differs from `b7dd5cb` by exactly **two hunks**: the
  `if tools.new_delay_method:` branch in `compare_video.run`, and the new method
  appended after it. Everything the default path executes is untouched.
- With `VMSAM_OFFSET_METHOD` unset: `tools.new_delay_method` is `False`,
  `audio_offset_gccphat.method_selected()` is `"default"`, and
  `audioCorrelation.second_correlation` falls through to the Rust `audio_sync`
  path.

---

## 2026-08-30 02:00Z — The `.log.error` corpus: what it really says, and a loop that closes it

### 1. The corpus is nearly monolithic, and its exception is uninformative

128 files, 146 retry blocks. **121 of 128 end on the same exception** -- every
candidate was dropped, nothing left to merge. Classifying on that string yields
one bucket and no insight. The signal is the last `Recovery declined` line in
`Merged errors`:

| Terminal mechanism | Files |
| :--- | ---: |
| offset profile not uniform across the timeline | 104 |
| master carries several tracks of the language, reference ill-defined | 17 |
| no common audio language | 6 |
| an `ffmpeg` command failed | 1 |

### 2. "Old" is not a date

Every retirement rule that fires on age was rejected. The corpus is younger than
15 hours and inside it a fresh file can be stale and an old one current. The rule
implemented is *can the pair still produce this record?* -- master replaced,
candidate consumed, a recorded in-sync merge for **this** candidate, or every
block predating a deployed fix that names that mechanism.

Two tempting signals were measured and rejected:

- **A merged output beside the log proves nothing.** 10 files have one. In all
  10 it came from a *different* candidate of the same episode: the output is
  named after the MASTER and siblings share it.
- **No deployed fix names any of the four mechanisms.** The `float`/`Decimal`
  frame-arithmetic fix would, but it is not deployed and no block shows that
  crash.

**Dry run: 1 of 128.** Its master was replaced in the library the same day. The
other 127 reproduce today -- and 18 of them prove it directly, having been
retried ~13 h apart with an identical mechanism and an identical measured
profile both times.

### 3. Why the structural repair never shows up in these logs

The declined non-uniform profile is *correct behaviour*: one delay cannot absorb
a step. The chimeric repair is supposed to take over. It never appears in any of
the 128 logs -- no chimeric path, no chimeric error. Two candidate explanations,
and the evidence points at the first:

1. **The structural path is not executing in the container.**
   `tools.get_cut_points_table()` returns the env value and an empty
   `VMSAM_CUT_POINTS_TABLE` disables the whole path silently. Decisive evidence:
   one pair in the corpus makes `pysubs2` raise `FormatAutodetectionError` during
   the chimeric build (reproduced locally). That exception is **not** a
   `ChimeraError`, so it escapes `build_chimeric_candidate` and would abort the
   fusion job with that traceback. No log shows it. So that build never ran.
   The PAL prebuild *did* run for 3 files from the sibling table in the same
   directory, so the directory itself is reachable -- check the env var.
2. **It runs and fails silently.** `build_chimeric_candidate` deletes its output
   and returns `None` when its own verification fails, and
   `prepare_structural_corrected_candidate` then falls back to the unrepaired
   candidate, printing the reason to stdout only. Nothing reaches `errors_merge`,
   so the corpus cannot show it either way.

Either way the same one-line change makes the corpus self-describing: append the
chimeric outcome to `errors_merge`. Not made -- another job owns the merge path
this round.

### 4. The correction loop, and the four defects it had to fix first

`src/tools_batch/error_correct.py`: read the log, choose the repair the evidence
supports, build, verify, apply the residual back onto the plateau that produced
it, rebuild. **The residual IS the lag error of the plateau it falls in**, so the
correction is `lag += residual` -- one measurement per round, no search. Two
rounds have sufficed on every file that converged.

Four defects found on the way in, each measured, each unit-tested:

1. `plateaus_from_cut_points` peak-filters a `windows` list but uses a stored
   `plateaus` list **verbatim**, so plateaus measured at correlation 0.33 with
   lags of five minutes go straight into the plan.
2. Plateau hygiene must compare master **starts**, never ends. Consecutive
   plateaus overlap in master space and that overlap *is* the content cut. An
   end-based rule kept 2 of 5 good plateaus on one pair and produced a
   "repaired" candidate whose first eleven minutes were master audio -- in sync,
   and worthless.
3. `plateau_timeline="auto"` guesses from the last `end_s`; dropping one tail
   plateau flips the guess and reinterprets every bound. State it.
4. `verify_chimera` reports `streams x plateaus` checks and falls back to the
   master's first audio stream for languages the master lacks, i.e. it
   correlates across languages at peak ~0.5. Feeding those raw to a refinement
   applies the same correction once per stream: **a plateau moved 395 ms when
   seven streams each contributed the same ~50 ms cross-language read.**
   Consolidate to one median per checkpoint over same-language streams only.

Acceptance is the project bar: only pieces carrying candidate content are judged
(a master-patched piece would compare the master with itself), at every plateau
midpoint, drift < 15 ms; `unverifiable` is not a failure.

### 4b. What the loop actually did — first 10 files

| Outcome | Files | Worst residual |
| :--- | ---: | :--- |
| `in_sync` | 7 | 0.40 / 0.70 / 0.90 / 1.10 / 1.25 / 1.50 / 4.60 ms |
| `out_of_sync` | 2 | 22.85 and 24.0 ms |
| `error` | 1 | subtitle path raised |

Every one of the seven converged in **exactly two rounds**, over 3-5 checkpoints
spanning the episode, at 54-168 s per file. All ten produced **no output at all**
under the deployed build.

The two failures share one signature and it is worth recording, because "still
23 ms out after 3 rounds" reads like it might converge with more rounds and it
will not. One checkpoint **flips sign every round while keeping its size**:
`+25.0 / -21.6 / +22.85` on one pair, `+23.75 / +22.0 / -24.0` on the other, with
the correlation peak dropping each time it flips (0.838 -> 0.664 -> 0.695). That
is a checkpoint sitting **on a plateau boundary**: the 20 s verification window
straddles the change point, the correlator locks alternately onto the content
each side of it, and what it reports is the two plateaus' lag *difference* with
alternating sign. Correcting the lag by that amount only moves the ambiguity.
The loop now recognises it, stops instead of paying for another rebuild, and
reports the floor it reached rather than the last round it happened to end on.

One file recorded `error`: `pysubs2` raises `FormatAutodetectionError` while
shifting an extracted `subrip` track. That is not a `ChimeraError`, so it escapes
`build_chimeric_candidate` entirely -- in the fusion worker it would abort the
whole job **inside the repair**, before the merge ever runs. The loop now retries
once with subtitles disabled and flags the result, because the audio question can
still be answered and dropping the candidate's subtitles is a real loss, not a
clean pass.

Refined plateaus for the seven are in
`/config/output/vmsam_agent/cut_points_refined.json`, in the shape
`prepare_structural_corrected_candidate` already reads.

### 5. Two buckets that are not what their message says

- **"No common language" is about the tags, not the audio.** One pair measured
  (instrument calibrated first: master against itself, 0.0 ms at peak 1.027):
  the two single tracks correlate at peak 0.998 and 1.157 with lags -4450.5 and
  -4449.9 ms -- the same recording, 0.6 ms apart over 300 s, tagged `jpn` on one
  side and `eng` on the other. Six files, all one tag override from merging, and
  the pipeline raises before it ever correlates.
- **"Reference is ill-defined" declines with TWO uniform matches, not zero.**
  Measured: the master's own two tracks of the correlation language are
  **132.0 ms apart** (peak 1.086 -- the case `DOMAIN_INVARIANTS` §6 records at
  "131 ms"), and the candidate correlates uniformly with both: +71.9/+71.4/+70.2
  ms against one (spread 1.7 ms) and -59.9/-60.1/-60.5 ms against the other
  (spread 0.6 ms). Neither is wrong. A delay only means something relative to a
  chosen master track, and which track survives into the output is decided later
  inside the protected merge closure -- so the recovery cannot know, and
  declines. Fix the reference and each of these is a single constant delay.
  17 files, 15 of them one show.

### 6. Tooling

`src/tools_batch/error_log_triage.py` (`report` / `plan` / `apply --yes`) and
`src/tools_batch/error_correct.py` (`plan` / `run` / `report`). Both resumable,
both dry-run by default, both writing only under `/config/output/vmsam_agent`.
`correction_state.json` is the correction loop's own file -- `batch_state.json`
keeps its single writer. Refined plateaus go to `cut_points_refined.json`, never
back into the measured tables. `apply` copies the log into
`error_log_archive/` and fsyncs a manifest line **before** unlinking, so a crash
duplicates evidence rather than losing it; the whole path was exercised in a
sandboxed tree. Stage-1 suite: 143 -> 222 checks, green.

## 2026-08-30 01:00Z — Episode-treatment logging in `fusion.py`, and a build stamp on every `.log.error`

### What was added

`src/gestionar_show/fusion.py` now emits **one delimited block per episode**,
every line prefixed `FUSION|` so a single grep pulls one episode out of
interleaved merge output. It is buffered during the job and emitted once at the
end, so it stays contiguous in `docker logs`.

Sections: **job** (id, start, mode, build, error file) · **database**
(folder id, episode number, destination path, `original_language`, `number_cut`,
`cut_file_to_get_delay_second_method`) · **pairing** (both paths, both weights,
both sizes, which file is the reference and **which branch of
`select_reference_file` fired**, phrased so MASTER reads as the normal case and
CANDIDATE as the inverse per `DOMAIN_INVARIANTS` §1/§5) · **media** (duration
for both plus the signed delta, and per-language audio/subtitle counts side by
side) · **paths** (temp, out, merge input, whether a repair fired, merged file,
published destination, `.log.error`) · **timing** (resolve / probe / setup /
merge / publish / cleanup) · **failure** (the exception, the traceback, and
`mergeVideo.errors_merge` — which previously only ever reached the `.log.error`
file and never the console).

**Gate**: `tools.dev` **and** `execution_mode == tools.mode_test`, written to
**both stdout and stderr**. Proven closed both ways by unit test.

**Discipline**: purely observational. It computes nothing the merge consumes;
the repair is named by reading the value `merge_videos_with_repair` already
returns rather than re-deriving it; the reference branch is *described*, never
decided, and a disagreement with `select_reference_file` is printed instead of
hidden. Every probe and the whole emission path swallow their own exceptions —
a unit test makes a probe raise and asserts the merge outcome is unchanged.

Probing is one cheap `ffprobe` per file (container metadata, no decoding), run
sequentially from the single worker thread: at most one file at a time, far
under the 5-file cap.

### `.log.error` now names the build

`append_error_log` writes
`===== Fusion retry <ts> (VMSAM_MODE=<mode>) (git=<sha>) =====`. The SHA comes
from `tools.get_git_commit()` — the **same source** `GET /health` reports, so
the two cannot disagree — and it reads only an environment variable, so it
cannot raise and needs neither the `git` CLI nor a `.git` directory. Unlike the
episode block this is **ungated**: it is written in production too, because a
production failure is the one that most needs attributing to a build. An
undeterminable version writes `git=unknown`; the field is never omitted. The
`(VMSAM_MODE=...)` group is byte-identical to the old banner so existing
parsers keep working.

Before this, a failure recorded under old code was indistinguishable from one
recorded after the fix that addresses it — the whole error corpus was ambiguous.

### Tests

`src/tools_batch/tests/test_06_fusion_logging.py`, 54 checks (suite total
83 → 137, still under two seconds, still no media decoded). Covers the pure
formatters, the ffprobe summariser, every branch of `explain_reference_choice`
**cross-checked against `select_reference_file` itself**, the gate open and
closed both ways, the `.log.error` banner, and `run_fusion_job` end-to-end with
the merge engine, the session and the pools stubbed.

### ⚠️ The gate is currently closed in the deployed container

`run.sh` launches `main_gestionar_show.py` **without `--dev`**, so `tools.dev`
is `False` in `showgestionar-test` and this block will not appear in its
`docker logs`. The `Dockerfile` sets `ENV dev=true`, but nothing reads that
variable — `run.sh` never expands `$dev`. Both files are read-only for agents.
Until the owner wires one to the other, every `tools.dev`-gated diagnostic in
this project is dark in the deployed container.

---

## 2026-08-29 19:20Z — 5h01m milestone checkpoint #1

**Corpus**: 309 error files. **Processed through the live API**: 158. **Remaining**: 151.

### Live batch outcomes

| Outcome | Count |
| :--- | --: |
| FAILED | 109 |
| MERGED_UNVERIFIED | 29 |
| NO_OUTPUT | 14 |
| MERGED_IN_SYNC | 6 |

### By error class

| Class | Total | Processed | In sync | Failed |
| :--- | --: | --: | --: | --: |
| A_no_lang | 6 | 6 | 0 | 6 |
| B_fps | 35 | 35 | 2 | 15 |
| C_cut | 180 | 29 | 0 | 24 |
| D_trim | 58 | 58 | 2 | 41 |
| E_corr | 30 | 30 | 2 | 23 |

### Diagnostics written

- First-pass recipe reports: **309/309** (complete)
- Autonomous deep reports: **126/309** (in progress; two agent fan-outs were cut short by session limits)
- Files already proven `Unmergeable_Verified` in the first pass: **7**

### Double-validation status

Step 2 (live container) is running continuously. Step 1 (independent local audit,
strict <= 15 ms) has confirmed every merge examined so far: 23 outputs audited in
one pass returned **13 at 0.0 ms, worst 40 ms, zero failures**, and the Rick and
Morty PAL family verified at 7.1-15.8 ms. `MERGED_UNVERIFIED` means the automatic
check could not find a same-language cross-provenance pair, not that the merge is
suspect — those go through a second pass that resolves the true master from the
job log.

### Code shipped this period

| Commit | Change |
| :--- | :--- |
| `6b90a08` | Two guards on the frame-space gates after adversarial testing (see below) |
| `0336c84` | `fusion.py` thinned to a pure queue worker; all merge intelligence moved to `mergeVideo.py` behind `merge_videos_with_repair` |
| `633f442` | `src/DOMAIN_INVARIANTS.MD` — facts agents kept getting wrong |
| `0f426fc` | `AGENT.MD` document hierarchy as a table |

### Findings worth carrying forward

1. **The one-frame gate does mask real edits — but not because of my change.**
   Adversarial testing over 929 recorded delay vectors found 7 of 20 one-frame
   pairs have genuinely piecewise lag (steps -209 to +168 ms). All 7 are 125 ms
   spreads that master's `< 127` already admitted; the change newly admits only
   3 pairs, all measured constant to 0.2-0.4 ms. The test did find a real defect
   of mine in the fourth gate (frames in {-1,0,1} describe three delays spanning
   two frames) — guarded in `6b90a08`.
2. **Reviving the consensus vote is high risk, not a missed opportunity**: it
   rescues 0 of 52 failed files and is measurably wrong on 3 of 4 candidates.
3. **A verifier corrected an investigator's headline numbers**: a claimed -63.9%
   speedup was ~2x inflated by warm page cache (true figure -31.6%), and a
   "0/25 decision changes" gate was vacuous because all 50 values were null.
4. **Disk is the real constraint, not tooling.** `/config/output` at 57%
   (405 GB free), 64 GB written. The chimeric path emits FLAC (945 MB for one
   22-minute episode).

### Queue state

Batch on C_cut 30/180. Watchdog and hourly deploy alive. Deployed SHA equals
`dev-AI` HEAD. Load average 38 on 17 cores.

**This checkpoint does not stop the run** — sweeps continue per the
run-to-completion rule.

---

## 2026-08-29 (third pass) — Team campaign: PAL, chimeric, and the full validation loop

### Delivered and Stage-1 validated this pass

| Capability | Evidence |
| :--- | :--- |
| PAL speed prebuild (fusion-integrated) | Log Horizon S02E23: corrected 1472→1411.7 s vs master 1408.1 s; detection positive/negative controls pass. |
| Chimeric structural repair (fusion-integrated) | Synthetic ground truth 23/23 (residuals +0.1/−1.8/−1.8 ms, gap proven master-sourced); Rick & Morty S01E09 via the fusion path: built in 42 s, duration exact. |
| PAL + content-step chain | R&M S01E01: 1.296 s step at cand_t 508 s repaired; alignment −40/0/0 ms at 120/600/1100 s. |
| Sign-fixed measured recovery | Ragnarok S02E03 merged; cross-provenance same-language check +0/+0/+10 ms (was −170 ms). |

### Measurement corpus (sub-agent team)

- `pal_factors.json`: all 21 PAL files match 1.042708 within 0.006% by slope; 12
  are PAL **plus one content step** (all 11 R&M + Suicide Squad ISEKAI S01E06);
  2 files where the endpoint duration ratio lies (content, not speed).
- `cut_points.json`: 57/57 pairs mapped, 156–161 change points, 96% within ≤5 s
  brackets (133 to ~1 s), median step ~1.0 s. Noir E23 reclassified
  (not pure tail; correlation-ambiguous A→B→A signature — excluded from
  chimeric). 0-saiji S01E04 reclassified: constant −476 ms, not structural.
- Log Horizon verdict: master is a pitch-preserved phase-vocoder stretch whose
  Chromaprint fidelity floor (~0.57) is unreachable for ANY clean source;
  envelope NCC is the only viable alignment signal for it. `asetrate`
  correction leaves +0.7 semitone error on such masters (classic PAL masters
  pitch-up, so asetrate stays the default).

### Shared-state lessons (three incidents, all fixed)

1. `batch_state.json` corrupted by two concurrent writers → per-PID temp names +
   corruption quarantine in `load_state`.
2. `cut_points.json` / `pal_factors.json` overwritten by two sub-agent
   generations with different schemas → consumers (`plateaus_from_cut_points`,
   `load_pal_factors_record`) accept every observed shape; conflicting
   generations preserved as `*_conflicting_*.json`.
3. A silent `s.replace` no-op left the batch runner misclassifying appended
   `.log.error` files as NO_OUTPUT → detection now tracks file growth, and the
   Edit-verify habit replaces blind string patching.

### Campaign state at write time

E_corr pass under the new build was cut short by the runner bug above (3 files
correctly IN_SYNC from the earlier pass; 0-saiji ×3 remain the verified
resolutions there, plus local Ragnarok). B_fps running live against `7b81ee7`
with PAL prebuild + step chain; D_trim (chimeric) follows; E_corr re-run last.
Results land in `batch_state.json` and the next worklog entry.

---

## 2026-08-29 (second pass) — Sign inversion found and fixed; PAL prebuild integrated

### 1. Provenance investigation settled the recovery question

A dedicated investigator established per-stream provenance of the merged
Ragnarok output by correlating every merged stream against both sources
(peaks 0.998–1.000): **the applied shift lands in the output** — all
candidate-derived tracks moved exactly −80 ms — but the required shift was
+70/+90 ms. **Correct magnitude, inverted sign**; same-language check
(candidate eng vs master eng) confirmed −170 ms net desync.

Cause: `recover_not_compatible_videos` negated the measured lag, copying
`compare_video`'s convention for *its own* measurement direction.
`measure_offset_ms` returns +lag when the candidate must be *delayed* — which is
what a positive `delays[language]` already does. **Fix: no negation.** Recovery
re-enabled. The prior worklog entry claiming the delay "does not reach the
output" is corrected by this one: it reaches it, negated.

### 2. Redundant language normalisation removed (user review)

`normalise_language()`/`find_audio_stream_index()` duplicated what `video.py`
already does — its `audios` dicts are keyed ISO 639-1 and each entry carries
`StreamOrder`. Replaced with `get_stream_order(video_obj, language)` reading the
parsed metadata; measurement functions now take explicit stream indices.

### 3. PAL prebuild integrated (Stage 1 validated)

- `src/merge_video_prebuild.py` — ports the validated prebuild
  (`asetrate`+`aresample:soxr` audio, `-itsscale:s` for subtitle starts, FLAC
  out, duration verified ±0.5%). Known limitation, documented: `-itsscale`
  scales cue starts, not cue durations (~4.3% long display on text subs).
- `mergeVideo.py` — `detect_pal_relation()` (duration ratio vs
  theoretical 1.042708 ±0.5%, both directions) and
  `prepare_pal_corrected_candidate()`, composed by `prepare_repaired_candidate()`
  and reached through `merge_videos_with_repair()`, which `run_fusion_job` calls
  in place of `merge_videos`. (These lived in `gestionar_show/fusion.py` until
  the fusion-intelligence move; fusion is now only a queue worker.) On any failure it falls back to the original file. Positive
  control (Log Horizon S02E23 → 0.959041) and negative control (0-saiji → None)
  both pass. **The theoretical factor is always used, never the endpoint ratio**
  — the endpoint absorbs trims and left 2.66 s residual drift vs 2 ms.

### 4. OPEN — Chromaprint disagrees with envelope NCC on the corrected pair

End-to-end PAL merge of Log Horizon S02E23: prebuild correct (1472→1411.7 s vs
master 1408.1 s), but the fingerprint stage reports fidelity 0.56–0.59 with
scattered offsets (0 / −198.9 s / −207.0 s) and rejects the pair — while
envelope NCC on the same corrected audio measures +17 ms flat. Under
investigation (full-duration lag profile, audio-master comparison, per-cut
reproduction of the pipeline method). The 3.6 s residual duration difference is
unaccounted for and may be a leading trim or structural difference.

---

## 2026-08-29 — Worker deadlock fixed; the one-frame constant found and corrected

### 1. PRIORITY 0 — fusion worker deadlock, fixed and verified live

`run_fusion_job` creates its `multiprocessing.Pool` objects from the worker
*thread*, under the forked start method. `Pool.terminate()` joins its handler
threads with an effectively unbounded timeout, and those handlers block when a
worker is stuck mid-task or was forked while another thread held a lock. The
`finally` never returned, so `worker_loop` never cleared `current_job`.

Observed before the fix: a job whose `.log.error` records failure at
`20:57:42Z` was still `"processing"` more than an hour later, with a second file
queued behind it that never started.

**Fix** (`gestionar_show/fusion.py`): `run_with_deadline()` runs each cleanup
step in a throwaway daemon thread and returns at the deadline regardless;
`close_ffmpeg_pools()` calls `close()` then `terminate()` under a 30 s deadline
and, on overrun, SIGKILLs the workers via `kill_pool_processes()` and clears the
module attributes so the next job builds fresh pools. `tools.remove_dir` in the
same `finally` is bounded too.

**Verified live on `showgestionar-test` at commit `672911d`:** submitted the exact
file that previously wedged the queue; status went `idle → processing → idle` and
the queue drained. Stage 1 `test_06_deadlock.py` 13/13 reproduces the hang with a
pool whose `terminate()` never returns and asserts the worker still drains.

### 2. Root cause of the one-frame rejections — a hardcoded frame duration

`audioCorrelation.get_max_corr` returns
`(correlation, offset_in_frames, -offset_in_frames*sizePoint)`, where `sizePoint`
is the Chromaprint frame duration **for that pair**,
`int(lengthFile/len(fingerprint_source)*1000)`.

Measured across the recorded failures: **917 of 933 delay vectors have a
self-consistent ms-per-frame, 859 of them at 125 ms.** But it varies with content
length, and three files in this library have a 138 ms frame.

Three gates in `first_delay_test` compare the **millisecond** values against a
hardcoded `127`/`128`, which encodes "within one frame" only when the frame is
125 ms. On a 138 ms frame an ordinary adjacent-frame disagreement measures 138,
exceeds the constant, and is misread as a real timing conflict — so the pair is
discarded before reaching the second method, the only stage that resolves below
fingerprint resolution.

Added `is_within_one_fingerprint_frame()` (spread between cuts) and
`are_residuals_within_one_frame()` (each residual's magnitude), both comparing
frame indices directly, which is exact for every `sizePoint`. Wired in as
alternatives beside the existing constants at the three gates — tests 2, 3 and 4
— so nothing that passed before changes.

**Result:** `0-saiji ... S01E07`, the file that motivated this, now merges. It
needs **no recovery at all** — the standard refinement loop converges once the
gates stop rejecting it. Locally: 94 s, one merged file produced. The same three
files (S01E04, S01E07, S01E08) are the only ones in the library with a 138 ms
frame, and they are exactly the ones the constant rejected.

### 3. Sound replacement for the disproven consensus vote

`recover_by_direct_correlation()` re-derives the offset **from the audio** at
10 ms resolution (`measure_offset_ms`, envelope cross-correlation) at three
points along the timeline, and only returns a delay when the windows agree within
40 ms. A profile that varies is a structural edit and is declined for the
trimming/patching strategies.

Measured on the Ragnarok S02E03 pair: **+80.0 ms at 60 s, 480 s and 900 s** —
perfectly uniform. The fingerprint method could only say "0 or 1 frame" because
80 ms sits *inside* one 125 ms frame. That is precisely why voting on quantised
residuals produced the 130 ms desync recorded in the previous entry.

### 4. VERIFIED RESOLVED — 3 files, sync-checked on showgestionar-test

Deployed `4732a2b`, reprocessed through `POST /fusion`, and cross-correlated the
master-derived and candidate-derived tracks inside each merged output:

| File | Outcome | Max lag | Peak |
| :--- | :--- | :--- | :--- |
| `0-saiji … S01E04` | **MERGED_IN_SYNC** | 10.0 ms | 0.999 |
| `0-saiji … S01E07` | **MERGED_IN_SYNC** | 10.0 ms | 0.999 |
| `0-saiji … S01E08` | **MERGED_IN_SYNC** | 10.0 ms | 0.999 |

10 ms is one bin of the measurement envelope. Locally the same S01E07 pair
measures **−480 ms between the two sources** and **−10 ms inside the merged
output**, with the master track bit-aligned to its source (+0.0 ms, peak 0.997).

These are exactly the three files with a 138 ms fingerprint frame — the ones the
hardcoded `127`/`128` constants rejected. Root cause found, fixed, deployed,
reprocessed and sync-verified.

**A merged file appearing is not success.** Every claim above is gated on the
cross-correlation check, because an earlier attempt produced a 1.4 GB output that
was a constant 130 ms out of sync.

### 4b. CORRECTION — the "130 ms desync" was a measurement error of mine

Two earlier entries state that a recovered delay produced a merge that was a
constant 130 ms out of sync, and that the recovery delay "does not reach the
output". **Both conclusions were wrong, and the cause was my own measurement.**

The sync check compared "the first two tracks of the merged file in the shared
language". For the Ragnarok pair the **master itself carries two `jpn` tracks**
(`aac` and `eac3`) and those two are **+130 ms apart in the source file**:

```
master source, stream 2 (aac jpn) vs stream 3 (eac3 jpn)
  @00:01:00   +130.0 ms   peak 0.984
  @00:10:00   +130.0 ms   peak 0.962
```

Those are the same numbers and the same peaks I read off the merged output. The
check was comparing the master against **itself**, carried through the merge
unchanged. The +130 ms was pre-existing in the source and had nothing to do with
any delay the pipeline applied — which is also why two very different applied
values produced an identical residual. That identical-residual observation should
have been read as "the measurement is not sensitive to the delay", not as
"the delay is being discarded".

**Fix to the method** (`batch_verify.sync_verdict`): establish provenance before
judging. `generate_launch_merge_command` emits the best video's tracks first, so
within a language the first `n_master` streams are master-derived and anything
beyond is candidate-derived; the check now compares `streams[0]` against
`streams[n_master]`. When the candidate contributed no track in that language —
deduplicated away as an inferior copy — the honest answer is `unverifiable`, not
a number.

**What survives.** The three `0-saiji` resolutions in §4 are unaffected: that
master has exactly one `jpn` track, so the merged file's two `jpn` tracks are
genuinely one per source. Their −10 ms stands.

**What is now reopened.** Whether the measured-offset recovery works is *undecided*
again, not disproven. It was disabled on the strength of the wrong conclusion. The
Ragnarok merged output's only cross-source comparison available is cross-language
(candidate `fre` against master `jpn`), which measured +160 ms at peaks
0.56–0.85 — too biased to rule on, since only music and effects align between two
different dubs.

### 5. Batch analysis (sub-agents) — two corrections to earlier claims

**PAL / frame rate, 35 files** (`/config/output/vmsam_agent/pal_analysis.json`):

| Verdict | Count |
| :--- | :--- |
| `linear_speed` (correctable) | 21 |
| `cut_difference` | 14 |
| `header_lie` | **0** |

> **Correction.** An earlier note claimed the container frame rate "lies" for the
> 30 fps and 29.97 fps groups. It does not. Timestamp-derived fps matched the
> declared `r_frame_rate` **exactly for all 69 files**. The 7 Suicide Squad
> ISEKAI files genuinely are 30 fps — they carry a PAL-sped timeline
> frame-rate-converted into a 30 fps container, so `fps_mst/fps_err` = 0.799
> misses them while the duration ratio (0.9583–0.9593 vs 23.976/25 = 0.95904)
> identifies them. A naive fps-ratio test would have skipped 7 correctable files.

All 21 `linear_speed` cases are the same PAL relation, factor **1.042708**.
`Lazarus S01E01` is 918 s against a 1452 s master (ratio 0.632) — almost certainly
a truncated file, not a merge problem.

**Cut / trim, 58 files** (`/config/output/vmsam_agent/trim_analysis.json`):

| Verdict | Count |
| :--- | :--- |
| MIDDLE (interior divergence) | 51 |
| TAIL (trailing content only) | 6 |
| INCONCLUSIVE | 1 |
| HEAD | 0 |
| **UNIFORM offset** | **0** |

> **Correction.** I had expected a share of these to be plain constant offsets
> recoverable by a delay. **None are.** Every window pair was probed at the same
> absolute timestamps in both files, so a head insertion would have shown as an
> equal non-zero lag in both windows; none did. The 6 TAIL cases are the only
> cheap ones (body aligned, 80–380 ms constant offset, difference entirely after
> the body). The INCONCLUSIVE file (`Secrets of the Silent Witch S01E12`) has a
> best peak of 0.42 with no consistent lag across five windows and all stream
> combinations — it looks **mis-paired**, not merely trimmed.

---

## 2026-08-28 (fourth pass) — Deferred recovery scaffold; a measured wrong turn on delay consensus

### Scope

Built the deferred-exception / recovery scaffold requested for post-processing,
then attempted the correlation-failure fix on top of it. **The fix was measured
to be wrong and was removed.** The scaffold is sound and is kept; no file is
claimed as resolved.

### Delivered and kept

| Change | Status |
| :--- | :--- |
| `remove_not_compatible_video(..., defer_exhaustion=, reference_video=)` | Returns `True`/`False` instead of raising when deferred. Default unchanged, so every existing call site behaves as before. |
| `not_compatible_video_list` entries are dicts | `{file_path, reason, metadata}`; `metadata` carries `delayFirstMethodAbort`, `delays`, `audio_languages`. `get_not_compatible_video_paths()` added for path-only callers. |
| `recover_not_compatible_videos(...)` in `sync_merge_video` | Runs after `get_delay`, before the original exception is raised by `raise_not_enough_file`. The extension point constraint 3 asked for. |
| `merge_videos` resets the accumulator | Scopes it to one run, matching the `errors_merge` convention. |
| `get_consensus_delay(...)` | Majority vote over per-cut offsets; accepts both the raw `(fidelity, ?, offset)` tuples and the flat int lists `get_delays_dict` produces. |

### The wrong turn, and the measurement that caught it

**Hypothesis.** 98% of `.log.error` files carry per-cut delay vectors in
`Merged errors`. Many show a lopsided split — e.g. `{138: 9, 0: 1}` at fidelity
0.91 — which looked like one outlier cut (a silent intro correlating to 0)
sinking an otherwise unanimous result. `first_delay_test` already tolerates 25%
dissent but only when the first and last cut agree, so an outlier at either end
is never forgiven.

**What was built.** A majority vote, used post-hoc: read the vectors stored in
`delayFirstMethodAbort`, take the modal offset, write it to `delays[language]`,
re-insert the candidate.

**Result.** A pair that production rejected merged, producing a 1.4 GB file.
Cross-correlating the master-derived Japanese track against the candidate-derived
Japanese track *inside the merged output* gave:

```
00:01:00  best lag = +130.0 ms   (peak 0.984)
00:10:00  best lag = +130.0 ms   (peak 0.959)
00:20:00  best lag = +130.0 ms   (peak 0.977)
```

**Constant +130 ms across the whole episode — the merge was silently desynced.**
Strictly worse than the failure it replaced.

**Root cause of the wrong turn.** `first_delay_test` is *iterative*. It
accumulates `delayUse`, calls `recreate_files_for_delay_adjuster` to re-extract
the audio shifted by that amount, re-correlates, and repeats until the residual
reaches 0. The numbers in `delayFirstMethodAbort` are **residuals from a single
iteration**, carrying neither the accumulated base nor the refinement that
follows. Writing one into `delays[language]` discards both.

A second error compounded it: the success path writes
`delays[lang] += (delay * -Decimal('1.0'))`, negated and as a `Decimal`. The
first attempt assigned a positive int.

**Correction applied.** `recover_delay_by_consensus` now always returns `None`
and carries the explanation, so the dead end is not re-attempted. Consensus was
moved *inside* `first_delay_test` (both passes) where a residual re-enters the
refinement loop and converges through the normal path. Both branches are gated
on a strict majority and fidelity ≥ 0.85 — stricter than the 0.70 branch they
sit beside.

**Standing conclusion: post-processing recovery is the wrong home for delay
correction.** It is the right home for gap patching and segment trimming, which
repair the media. It cannot invent an alignment the correlation loop never
converged on.

### Why no file is claimed as fixed

On the one pair carried end to end, the in-loop consensus **did not fire**:
fidelities were 0.78–0.82, below the 0.85 bar, so every cut counted as dissent.
The pair now fails cleanly instead of merging out of sync. That is the correct
outcome and it is not a resolution.

The second method's own diagnostic for that pair is
`{-0.0189, 0.1219, 0.1228, …}` — two clusters ~140 ms apart, in seconds. Combined
with the constant +130 ms residual, the pair carries a **uniform** offset the
first method could not converge on, not a piecewise splice.

### BLOCKING — the fusion worker hangs after a failed job

`showgestionar-test` was given one file. Its `.log.error` records the failure at
`20:57:42Z`, 15 s in. Over an hour later `GET /fusion` still reported:

```
status: processing
current_job.started_at: 2026-08-28T20:57:27Z
```

A second file enqueued behind it went to `queue_position 1` and never started.
`worker_loop` clears `current_job` in a `finally`, so the hang is inside
`run_fusion_job`'s own `finally` — the `Pool.close()` / `Pool.terminate()` pair,
or `tools.remove_dir`. A plausible mechanism: those `multiprocessing.Pool`
objects are created from the worker **thread** under `set_start_method('fork')`,
so a child can be forked while another thread holds a lock and then deadlock.

**Consequence: batch reprocessing is impossible until this is fixed.** One
failure wedges the queue for the container's lifetime; only a recreate clears it.
This is the single highest-priority item — every other batch fix is untestable at
scale without it. Not fixed here: `fusion.py` is outside the protected closure,
but the diagnosis needs to be confirmed against a live hang rather than guessed.

### Verification

- `test_04_consensus.py` 30/30 — vote accepts 9-of-10 and 8-of-10, rejects
  7-of-10, ties, low fidelity and scattered offsets; defer/raise both correct;
  recovery declines to fabricate a delay.
- `test_01` 24/24, `test_02` 8/8, `test_03` 12/12 — known-good pair still
  yields delay `0E-9` with an empty incompatible list.
- Two real merges executed locally in `/tmp`; the library and error tree are
  read-only mounts and were never written to.

---

## 2026-08-28 (third pass) — Deployment validation rule, mergeVideo audit, architecture doc

### Scope

Established the deploy-and-verify loop as a mandatory rule, extracted the dead
code from `mergeVideo.py`, documented its architecture, and built an isolated
test harness against the `/config/example` samples.

### Files touched

| File | Change |
| :--- | :--- |
| `AGENT.MD` | **New mandatory section** — a change is not validated until observed running in `showgestionar-test`. Records the shared `/config/output` volume and the secrets rule. |
| `src/mergeVideo.py` | Two unreachable definitions removed (55 lines); `not_compatible_video_list` given a locked accessor API. 2106 → 2082 lines. |
| `src/merge_video_legacy.py` | **New.** The extracted definitions, verbatim, imported by nothing. |
| `src/MERGE_VIDEO_ARCHITECTURE.MD` | **New.** Flow, module-state contract, per-function contracts, protected closure, observed defects. |
| `FILE_MAP.MD` | Two rows added; the stale `get_cut_time` note corrected. |

Untracked, outside the repository: `/home/vmsam/src/DEPLOYMENT_WORKFLOW.md`.

### Decisions

**1. Dead code was proven dead, not assumed.** An AST call-graph walk from every
real entry point — `merge_videos` plus every `mergeVideo` attribute touched by
`main.py`, `main_gestionar_show.py` and `gestionar_show/fusion.py` — reached only
two module-level definitions out of 54. A text search confirmed no dynamic or
string reference. Those two, and only those two, were extracted.

**2. Extracted rather than deleted.** `find_a_cut_for_not_compatible` retries
alignment at a higher cut count instead of dropping a candidate — precisely the
second-chance pass `remove_not_compatible_video` skips, where 96% of recorded
failures die. Deleting it would discard the most relevant prior art for the
cut-difference work. `class get_cut_time` is unsalvageable (its `run()`
dereferences four attributes the class never defines, so it raises
`AttributeError` on start) but is kept alongside for the same archaeological
reason.

**3. `not_compatible_video_list` was made readable without changing its
lifecycle.** It was append-only with no consumer and no reset. Added
`reset_not_compatible_video_list()`, `get_not_compatible_video_list()` (returns a
copy) and `record_not_compatible_videos()`, all under a new `RLock`. The single
existing call site now goes through the recorder; nothing else in active code
changed. Scoping stays the caller's job, matching the `errors_merge` convention
already used in `process_episode` and `prepare_merge_context` — deviating from
that would have been the more invasive choice.

**4. The protected closure was computed, not eyeballed.**
`generate_launch_merge_command` reaches 35 names. Four of them —
`prepare_get_delay_sub`, `get_delay_fidelity`, `get_delay_fidelity_thread`,
`get_good_parameters_to_get_fidelity` — are *also* reached from the Stage C delay
path, so a future Stage C change must treat them as protected. Recorded in
`MERGE_VIDEO_ARCHITECTURE.MD` § 6.

### Verification

Deployment loop tested end to end: `d7aa850` pushed → hook fired
(`status=SUCCESS exit_code=0 duration_ms=29887`) → `GET /health` reported
`d7aa8507…`, up from `1689552…`.

Isolated tests under `/tmp/vmsam_tests/` (harness reproduces `main.py`'s
initialisation; `set_start_method('fork')` is required because Python 3.14
defaults to `forkserver`, which fails in this container):

| Script | Result |
| :--- | :--- |
| `test_01_pure.py` | 24/24 — merge-rule parsing, the accessor contract, `remove_not_compatible_video` raise/no-raise boundary, extracted names absent, five-name public surface intact. |
| `test_02_mediadata.py` | 8/8 — 4 releases parsed, all share `ja`, per-pair shared languages enumerated. |
| `test_03_get_delay.py` | 12/12 — real `get_delay` on the two CR WEB-DL releases: 164 s, returned `{forced: {other: True}}`, delay `0E-9`, empty incompatible list, and both documented side effects observed. |

### Defects recorded, not fixed

All in active legacy code; see `MERGE_VIDEO_ARCHITECTURE.MD` § 9.

1. **Index-shifting deletion** in `sync_merge_video`: incompatible videos are
   collected as ascending indices then removed with `del videosObj[i]` in
   ascending order, so with two or more the wrong elements are deleted. The
   parallel `dict_file_path_obj` cleanup uses paths and is correct, leaving the
   two structures inconsistent.
2. **`get_delay` reorders `videosObj`** — the forced best video is removed and
   re-appended, ending up last. Confirmed by test 03.
3. **The rejection reason is unrecoverable in production** —
   `show_not_compatible_error` is `False` there, so `.log.error` records only the
   terminal `"Only … file left"` message.
4. **Runtime mutation of module thresholds** — `compare_video` rebinds
   `max_delay_variance_second_method` and `cut_file_to_get_delay_second_method`
   via `global`, so two merges in one process do not start from the same values.

### API surface note

Re-mapped `showgestionar-test:8080` against the live `/openapi.json`: **10 routes,
exactly those in `API_REFERENCE.MD`.** There is no dedicated reprocessing or
"validated files" endpoint. The re-run loop is
`GET /errors` → `POST /fusion` → poll `GET /fusion` → inspect
`VMSAM_TEST_OUTPUT_DIR`; there is no API route that lists successful merges, so
success is observed on the filesystem rather than through the API.

---

## 2026-08-28 (second pass) — Unique constraints, SQLAlchemy 2.0 type map, API key normalisation

### Scope

Fixed the three ORM/API defects recorded in the previous entry, and audited the
`web-interface` branch for fallout from the key rename.

### Files touched

| File | Change |
| :--- | :--- |
| `src/gestionar_show/model.py` | `folders.__table_args__` gains `uq_folders_destination_path`; `incompatible_files.__table_args__` gains `uq_incompatible_files_file_path`; `Base.type_annotations` → `type_annotation_map`. |
| `src/gestionar_show/api.py` | `POST /regex/` insert branch returns `"folder_id"` instead of `"folder id"`. |
| `src/gestionar_show/SCHEMA.MD` | Types now `TEXT`, constraints documented, §10 migration added. |
| `src/gestionar_show/API_REFERENCE.MD` | Key rename, plus the new `IntegrityError` path on `POST /folders/`. |
| `FILE_MAP.MD` | `model.py` row notes the constraints and the type map. |

### Decisions

**1. The bare `UniqueConstraint` had to move into `__table_args__`, not just be
renamed.** A free-standing `UniqueConstraint(...)` in a declarative class body is
an expression whose value is discarded — the `Table` never sees it. `episodes`
already showed the working pattern with its composite `Index`.

**2. `type_annotation_map` is keyed by the type object, not its name.** The old
`type_annotations = {'str': Text}` was wrong twice over: unrecognised attribute
name, and a string key where SQLAlchemy expects `str` itself. Both had to change
for `Mapped[str]` to resolve to `Text`. Confirmed: every string column now emits
`TEXT` instead of `VARCHAR`. No runtime effect on PostgreSQL, where the two are
the same type.

**3. `episodes` was deliberately left unconstrained.** Its
`(folder_id, episode_number)` pair is read with `.first()` and treated as unique
by the application, and `episodes.file_path` can legitimately repeat. Neither was
in scope; adding a constraint there would be an unrequested schema change. A
regression test asserts `episodes` still accepts duplicate `file_path`.

**4. The redundant `index=True` on `folders.destination_path` was kept.**
PostgreSQL builds an implicit unique index to back the constraint, so
`ix_folders_destination_path` is now redundant. Dropping it is a separate schema
change and was not requested.

### Frontend audit — `origin/web-interface`

Ten files on that branch. Result: **no frontend change was needed.**

- `static/app.js:631` calls `await api.createRegex(payload)` and **discards the
  response** — it is never destructured, stored or read.
- `src/api.rs:139-155` (`proxy_create_regex`) forwards the upstream body through
  as opaque bytes; it never parses the JSON, so no Rust struct binds the key.
- The only `folder_id` occurrence on the branch is `static/app.js:97`, a **query
  parameter** on `GET /regex_folder/`, unrelated to the response key.
- Grepped every non-CI file on the branch for `"folder id"`: zero hits.

`"folder id"` also appears on `origin/master` and `origin/final`, but only in
their own backend copies of `api.py` / `gestionar_show_api.py` — not in any
consumer.

### Verification

- Emitted DDL re-generated with `CreateTable`/`CreateIndex` for the PostgreSQL
  dialect: both `CONSTRAINT ... UNIQUE` clauses present, all string columns `TEXT`.
- Behavioural test on a throwaway SQLite database: duplicate `destination_path`
  rejected, duplicate `file_path` rejected, distinct `file_path` accepted,
  `episodes` duplicates still accepted.
- AST import-closure walk: `api` → `internal_client`, `model`, `settings` with
  **no `fusion`** and no merge engine; `internal_api` → `fusion`, `model`,
  `settings`. Matches `AGENT.MD` § 2.
- `GET /health`, `GET /errors`, `GET /fusion` on `showgestionar-test:8080` still
  answer normally. The test container runs the previous image, so it does not yet
  exercise this commit.

### Pending — needs a lead-developer decision

**1. The constraints will not reach the running database on their own.**
`create_all()` skips existing tables and there is no Alembic, so the deployed
schema keeps its old shape until the `ALTER TABLE` in `SCHEMA.MD` § 10 is run by
hand. That migration also requires de-duplicating `incompatible_files` first: the
live test database holds 310 rows over 309 distinct paths and would reject the
constraint as-is.

**2. `IntegrityError` on the rejected-file path is caught but not recovered.**
`process_rejected_files()` (`main_gestionar_show.py:82-83`) moves the media into
the error tree and *then* inserts the row, with no pre-check. Once the constraint
is live, re-rejecting a same-named release raises inside
`insert_incompatible_file()`'s own `commit()`. The exception is swallowed by the
`try/except Exception: pass` at lines 161-169, so nothing crashes — but the
session is left in an aborted transaction (on PostgreSQL every later statement in
that `process_episode` call then fails), and the file sits in the error tree with
no row, invisible to `GET /errors` and unretryable via `POST /fusion`.

Both fixes mean adding exception handling to the scan loop, which `AGENT.MD`
§ "Exception Handling" reserves to the lead developer. Not changed here.

---

## 2026-08-28 — API and schema reference documents

### Scope

Reverse-engineered the `gestionar_show` HTTP surface and database schema into two
reference documents. **Documentation only — no behaviour, route, model or
configuration key was changed.**

### Files touched

| File | Change |
| :--- | :--- |
| `src/gestionar_show/API_REFERENCE.MD` | **New.** All 10 public and 3 internal endpoints: payloads, status codes, queue semantics, verification log. |
| `src/gestionar_show/SCHEMA.MD` | **New.** All 5 tables: columns, types, constraints, indexes, emitted DDL, relationships, write scope by mode. |
| `FILE_MAP.MD` | Two rows added to §4. |

### Method

Endpoints were read from the handlers, then confirmed against the live test
deployment at `http://showgestionar-test:8080` (commit `03d28b40…`,
`VMSAM_MODE=test`). The route inventory in `API_REFERENCE.MD` is diffed against
`/openapi.json` programmatically: no live route is undocumented and no documented
public route is absent from the spec.

Only read-only and non-mutating rejection paths were exercised. The mutating
success paths (`POST /folders/`, `POST /regex/`, `POST /incrementaller/`, a
successful `POST /fusion`) were deliberately **not** run — they write to the
shared test database and `POST /folders/` also creates directories on disk. Their
contracts are transcribed from source and flagged as such.

The schema was derived from `Base.metadata` via `CreateTable`/`CreateIndex`
compiled for the PostgreSQL dialect, since `DATABASE_URL` lives only in the
deployed container. The dialect was established indirectly: `GET /folders_infos/`
matches `Anime` (291 rows) but not `anime` (`404`), and a case-sensitive `LIKE`
rules out SQLite.

### Findings recorded (not fixed — schema is protected)

**1. `folders.destination_path` has no unique constraint.** The
`UniqueConstraint("destination_path")` at `model.py:23` sits bare in the class
body and is never passed to `__table_args__`, so SQLAlchemy discards it. The
`folders` table carries a `PrimaryKeyConstraint` and nothing else; the emitted
DDL has only a non-unique index. Uniqueness rests entirely on the
`get_folder_by_path()` check in `POST /folders/`, which two concurrent requests
can race. `episodes` declares its composite index correctly and shows the working
pattern. See `SCHEMA.MD` §7.1.

**2. `Base.type_annotations` is a no-op.** SQLAlchemy 2.0 reads
`type_annotation_map`; the name used is unrecognised, so `Mapped[str]` falls back
to `VARCHAR` instead of the intended `Text`. No runtime consequence on
PostgreSQL, where unbounded `VARCHAR` and `TEXT` are the same type. See
`SCHEMA.MD` §7.2.

**3. `incompatible_files.file_path` is not unique and does collide in practice.**
`get_incompatible_file_by_path()` resolves with `.first()`, and the live test
database holds 310 rows over 309 distinct paths (ids `349`/`350` share one), so
which row a fusion request resolves to is planner-dependent. The same table also
lacks the composite `(folder_id, episode_number)` index that `episodes` has,
though `get_incompatible_files_data()` filters on exactly that pair.

**4. Two response-shape quirks** worth knowing before writing a client: the
insert branch of `POST /regex/` returns the key `"folder id"` with a space rather
than an underscore, and the Pydantic defaults on `Folder` (`10` / `2` / `12`)
differ from the `insert_folder()` fallbacks (`5` / `2.5` / `12`).

### Pending

Items 1–3 are latent defects. Each needs an explicit instruction before any fix,
since all three are schema changes and the database is protected. Item 4 is an
API contract change and would break existing clients.

---

## 2026-08-27 (third pass) — Full public/worker decoupling, env-only git detection

### Scope

Removed every dependency of the public API on `fusion.py`, direct and transitive,
and reduced `get_git_commit()` to a build-metadata lookup.

### Files touched

| File | Change |
| :--- | :--- |
| `src/gestionar_show/settings.py` | **New.** `Settings` extracted so the internal app stops importing the public app. |
| `src/gestionar_show/api.py` | `fusion` import deleted; mode/git read via `tools`; `Settings` imported. |
| `src/gestionar_show/internal_client.py` | `fusion` import replaced by `tools`. |
| `src/gestionar_show/internal_api.py` | Owns execution-settings validation; `tools` for mode/commit. |
| `src/gestionar_show/fusion.py` | Mode/port settings removed, delegated to `tools`. |
| `src/tools.py` | `get_git_commit()` simplified; gained `get_execution_mode()`, `get_test_output_dir()`, `get_internal_port()`, `mode_*`, `supported_modes`. |
| `src/main_gestionar_show.py` | Uses the `tools` accessors; startup line logs mode and commit. |

### Decisions

**1. The settings moved to `tools`, they were not duplicated.** The zero-import rule
could have been satisfied by re-declaring `VMSAM_MODE` handling inside `api.py`, but
that would leave two definitions of the default and of `supported_modes` to drift
apart. `AGENT.MD` already requires shared runtime parameters to route through
`tools`, and `tools` is light (os/shutil/subprocess/psutil), so `get_execution_mode`,
`get_test_output_dir`, `get_internal_port` and the mode constants now live there.
`fusion.py`, `api.py` and `internal_api.py` all read the same source.

**2. The transitive edge mattered more than the direct one.** Deleting
`from . import fusion` in `api.py` was not sufficient: `internal_client.py` imported
`fusion` for `get_internal_port()`, so the public app still pulled the merge engine.
That edge is now `tools`. Enforced closure, verified by AST analysis:

```
gestionar_show.api          → internal_client, model, settings, tools
gestionar_show.internal_api → fusion, model, settings, tools
```

**3. `Settings` was split into its own module.** `internal_api` previously imported it
from `api`, which built a second unused `FastAPI` object in the internal process and
registered routes it never serves. `settings.py` removes that and makes the two apps
genuinely independent — neither imports the other.

**4. Validation is split by authority, not duplicated arbitrarily.** The public app
screens an unknown `VMSAM_MODE` (a cheap string check, avoids a pointless DB round
trip and a pointless hop) and resolves the two DB rows it alone has a session for.
The internal worker owns everything execution-related, including the
`VMSAM_TEST_OUTPUT_DIR` check that `api.py` used to perform — it is the process that
would fail on it. A caller still gets an immediate `400`, forwarded through the proxy.

**5. `get_git_commit()` is env-only.** The `.git/HEAD`, `refs/` and `packed-refs`
parsing is gone: the runtime image contains no `.git` directory, so that path was
~40 lines that could only ever fire in a dev checkout. Now
`os.environ.get("VMSAM_GIT_COMMIT", "").strip() or "unknown"`, cached in a module
global. **Operational consequence:** `VMSAM_GIT_COMMIT` must be injected via Docker
build arg or Compose, and it cannot be declared in the read-only `Dockerfile`. Until
it is, `/health` reports `"unknown"` — accurate, not an error.

**6. Path mirroring unchanged.** `tools.build_mirrored_folder_path()` remains the only
`lstrip(os.sep)` in the project, used by both `process_rejected_files` (error tree)
and `fusion.get_test_output_folder()` (test tree).

### Database access patterns

- `GET /errors` — one unfiltered `SELECT` over `incompatible_files`, ordered
  `folder_id, episode_number, file_weight DESC`. The table only holds unresolved
  rejects; add pagination if that changes.
- `POST /fusion` (public) — two indexed point lookups on the request-scoped session:
  `incompatible_files.file_path`, then `ix_folder_episode`. Fails fast on an unknown
  path or a missing master, and supplies `master_file_path` for the response.
- `POST /internal/fusion` — **no DB access.** The public instance has already resolved
  the rows, and the worker redoes full resolution at execution time; querying here
  would only add a round trip on the enqueue path.
- Worker (`run_fusion_job`) — its **own** session per job, from the `sessionmaker`
  handed to `fusion.start_worker()` at startup, closed in a `finally`. Never a
  request-scoped session: a merge runs for minutes to hours and would pin a pooled
  connection for its whole duration.
- Production commit is single-transaction: `episodes.file_weight` is left dirty and
  flushed by the `session.commit()` inside `delete_incompatible_file()`, so the weight
  update and the reject removal land together or not at all.
- `GET /fusion` and `GET /health` touch **no** database on either instance.

### API contracts

`GET /health` → `200`
```json
{"status": "ok", "git_commit": "<sha|unknown>", "mode": "test", "is_running": false}
```
`status` is `"degraded"` when `/internal/health` is unreachable or non-200; `mode` and
`git_commit` are read locally, so they stay accurate in that case.

`GET /errors` → `200` `{"incompatible_files": [{id, folder_id, episode_number, file_path, file_weight}]}`

`GET /fusion` → `200` `{status, current_job, queue_length, pending_jobs}` · `503` worker down

`POST /fusion` `{"error_file_path": "..."}` →
`200` `{message, error_file_path, master_file_path, mode, queue_position}` ·
`400` unknown `VMSAM_MODE` (public) or missing `VMSAM_TEST_OUTPUT_DIR` (worker, forwarded) ·
`404` unknown error path or no master episode ·
`409` already queued or processing · `503` worker unreachable

Internal, loopback only: `GET /internal/health` → the public payload plus
`queue_length`; `GET /internal/fusion` → queue status; `POST /internal/fusion` →
`{message, error_file_path, mode, queue_position}`.

### Known issues

- **`api.py` imports keep being rewritten to bare form** (`import fusion`,
  `import internal_client`). Inside a package those cannot resolve — the modules are
  not on `sys.path` and use relative imports themselves. `internal_client` was set
  back to `from . import internal_client`; the `fusion` one is now moot, deleted.
  Worth checking whether a formatter or hook is doing this.
- `frame_compare.py:109` does not compile (pre-existing, commit `da1c191`). Dead code
  — see `FILE_MAP.MD` §9.

### Not done / open

- `VMSAM_*` variables remain undeclared in the read-only `Dockerfile`;
  `VMSAM_GIT_COMMIT` in particular now needs a build-arg wiring to be useful.
- No cross-process throttle between the fusion worker and the integration loop.
- No cancel / bulk re-enqueue endpoint.
- Verification was `python -m py_compile` plus AST import-closure and cross-module
  reference checks. No runtime execution, no commit, per instruction.

---

## 2026-08-27 (later) — Dual uvicorn architecture, `/health`, shared path resolution

### Scope

Split the single constrained API process into a public multi-worker instance and a
loopback-only single-worker fusion instance, added `GET /health`, and collapsed the
two copies of the error/test path logic into one shared helper.

### Files touched

| File | Change |
| :--- | :--- |
| `src/gestionar_show/internal_api.py` | **New.** Internal worker app (`127.0.0.1`, `workers=1`). |
| `src/gestionar_show/internal_client.py` | **New.** stdlib-urllib public → internal client. |
| `src/gestionar_show/api.py` | `/health`; `GET`/`POST /fusion` now proxy; worker lifecycle removed. |
| `src/gestionar_show/fusion.py` | Lazy merge-engine imports, `is_job_running()`, `get_internal_port()`. |
| `src/tools.py` | `build_mirrored_folder_path()`, `get_git_commit()`. |
| `src/main_gestionar_show.py` | Forks two uvicorn processes; `process_rejected_files` uses the shared helper. |

### Decisions

**1. Topology.** Public `0.0.0.0:8080` `workers=5` serves `/health`, `/errors`,
`GET|POST /fusion` and the pre-existing folder/regex routes. Internal
`127.0.0.1:8081` `workers=1` serves `/internal/health` and `/internal/fusion`, and
owns the only queue and worker thread in the deployment. `main_gestionar_show`
starts the internal instance first, then the public one, and terminates both.

**2. `GET /fusion` proxies too, not just `POST`.** The brief only specifies
forwarding for `POST`, but the queue lives exclusively in the internal process — a
public worker has no queue state to report. Serving `GET /fusion` locally would
return a permanently empty queue. Same for `is_running` in `/health`.

**3. Validation stays public, execution stays internal.** `POST /fusion` checks
`VMSAM_MODE`, `VMSAM_TEST_OUTPUT_DIR`, the `incompatible_files` row and the master
`episodes` row *before* forwarding, so bad requests never reach the worker and the
caller gets a precise 400/404. The internal endpoint does not re-validate — the
worker redoes full resolution at execution time anyway, and duplicating it would
just add a second DB round trip on the hot path.

**4. The proxy uses `urllib`, not `httpx`/`requests`.** Neither is installed in the
runtime image (`Dockerfile` installs `python3-fastapi`, `python3-uvicorn`, … and no
HTTP client), and the `Dockerfile` is read-only per `AGENT.MD`, so a dependency
cannot be added. `internal_client` therefore builds on the standard library. It
uses an opener with an **empty `ProxyHandler`**: bare `urlopen` honours
`http_proxy`/`HTTP_PROXY`, which would route loopback traffic through a proxy if the
container defines one.

**5. Merge-engine imports are lazy.** `fusion.py` no longer imports `mergeVideo` /
`video` at module scope — `load_merge_modules()` does it on demand.
> **SUPERSEDED 2026-08-27 (third pass):** the public app no longer imports `fusion`
> at all, so laziness is no longer what protects it. The lazy import is kept anyway:
> it still shortens internal-instance startup.

**6. Unreachable-worker handling.** `GET`/`POST /fusion` return **503** with the
transport error when the internal instance is down. `/health` instead returns
`status: "degraded"` with `is_running: false`, because the public instance is still
correctly serving its own routes and a health check should say so rather than fail.
Only `urllib.error.URLError` / `OSError` are caught — not bare `Exception`, per the
exception-handling directive in `AGENT.MD`.

**7. `git_commit` resolution order:** `VMSAM_GIT_COMMIT` → `.git/HEAD` (following the
ref, including `packed-refs`) → `"unknown"`.
> **SIMPLIFIED 2026-08-27 (third pass):** the `.git` parsing was removed. See the
> newest entry — the env var is now the sole source.

**8. Path resolution is now single-source.** `tools.build_mirrored_folder_path()`
holds the only `lstrip(os.sep)` in the project. Both `process_rejected_files` (error
tree) and `fusion.get_test_output_folder()` (test tree) call it, so the two trees
cannot drift.

**9. `internal_api` reuses `Settings` from `api.py`.** That imports the public app
module, constructing a second unused `FastAPI` object in the internal process.
Harmless and cheap; the alternative was duplicating the settings class or adding a
third module for one declaration.

### API contracts

`GET /health` → `200`
```json
{"status": "ok", "git_commit": "<sha|unknown>", "mode": "test", "is_running": false}
```
`status` is `"degraded"` when the internal worker is unreachable.

`GET /errors` → `200` `{"incompatible_files": [{id, folder_id, episode_number, file_path, file_weight}]}`

`GET /fusion` → `200` `{status, current_job, queue_length, pending_jobs}` · `503` worker down

`POST /fusion` `{"error_file_path": "..."}` →
`200` `{message, error_file_path, master_file_path, mode, queue_position}` ·
`400` bad `VMSAM_MODE` / missing `VMSAM_TEST_OUTPUT_DIR` ·
`404` unknown error path or no master episode · `409` already queued/processing · `503` worker down

### Known issues found during this pass

- **`api.py` had been edited to `import fusion` (bare).** That cannot resolve:
  `fusion.py` sits inside the `gestionar_show` package, not on `sys.path`, and uses
  relative imports itself, so it is only importable as `gestionar_show.fusion`. It
  was restored to `from . import fusion`, which is required for the app to start.
- `frame_compare.py:109` does not compile (pre-existing, commit `da1c191`). Dead code
  today — see `FILE_MAP.MD` §9.

### Not done / open

- No cross-process throttle between the fusion worker and the integration loop; a
  retry can still oversubscribe CPU against an automatic merge.
- No cancel / bulk re-enqueue endpoint.
- `VMSAM_*` variables are undeclared in the read-only `Dockerfile`.
- Verification was `python -m py_compile` plus an AST cross-module reference check
  only. No runtime execution, no commit, per the operator's instruction.

---

## 2026-08-27 — `/errors`, queued `/fusion` endpoints and dual-mode execution engine

### Scope

Added a manual retry path for files the integration pipeline rejected: list them
(`GET /errors`), enqueue a re-merge (`POST /fusion`), and watch the queue drain
(`GET /fusion`). Retries run through the same `mergeVideo.merge_videos` code path
as the automatic pipeline, but their side effects are gated by `VMSAM_MODE`.

### Files touched

| File | Change |
| :--- | :--- |
| `src/gestionar_show/fusion.py` | **New.** Queue, worker thread, dual-mode execution engine. |
| `src/gestionar_show/api.py` | 3 endpoints, `FusionRequest` model, worker lifecycle wiring. |
| `src/gestionar_show/model.py` | 3 read/delete helpers. **No schema change.** |
| `src/main_gestionar_show.py` | `uvicorn` `workers=5` → `workers=1`. |

### Decisions

**1. `uvicorn` dropped from 5 workers to 1 — this is load-bearing, do not revert.**
> **SUPERSEDED 2026-08-27 by the dual-instance split below.** The reasoning still
> holds, but it now applies only to the *internal* instance; the public one is
> back to 5 workers.
The queue is a `queue.Queue` in process memory. With 5 workers each process would
hold its own queue and its own worker thread, so `POST /fusion` and `GET /fusion`
would routinely hit different processes: the status endpoint would report a queue
the caller never wrote to, and up to 5 merges would run concurrently — exactly the
CPU/IO overload the sequential worker exists to prevent.

A second benefit falls out of this: with `workers=1` uvicorn skips its multiprocess
supervisor and serves the app in-process. `run_uvicorn` is `fork`ed from `__main__`
*after* `tools.software`, `tools.mergeRules`, `tools.core_to_use`, `tools.language_*`
and `tools.tmpFolder_original` are populated, so the API process inherits a fully
initialised `tools` module and can run merges. Under a spawn-based supervisor those
globals would be empty and every fusion job would fail. `prepare_merge_context()`
still `setdefault`s the `special_params` booleans so the module degrades rather than
`KeyError`s if the API is ever launched standalone.

Throughput is not a concern here: the endpoints are `def` (not `async def`), so
FastAPI runs them in its threadpool and they still serve concurrently.

**2. Volatile queue is intentional; only `QUEUED` and `PROCESSING` are tracked.**
No persistence table, no schema change. A restart drops pending jobs and orphans the
in-flight one — but the `incompatible_files` rows are only deleted on success, so a
lost job simply stays listed in `GET /errors` and can be re-enqueued. The database
remains the durable source of truth; the queue is scheduling state only.

**3. `VMSAM_MODE` defaults to `test`, and is read per job.**
An unset or misspelled variable must never silently take the destructive branch, so
the default is `test` and anything outside `{test, production}` is rejected with a
400 at `POST` time and re-checked in the worker. The value is read at execution time
(`get_execution_mode()`), so switching modes needs a container env change, not a code
change. Neither variable is declared in the `Dockerfile` (read-only per `AGENT.MD`) —
they must be supplied at container run time.

**4. Reference selection reuses the pipeline's existing rule.**
`select_reference_file()` picks the higher `file_weight`, and falls back to
`os.path.getsize` on a tie — identical to `process_rejected_files`, so a manual retry
and an automatic one choose the same `forced_best_video`.

**5. The merge result always lands on the master episode's registered path.**
`episodes.file_path` is the canonical library name, so a retry overwrites
the master's registered filename rather than promoting the error file's name; only `file_weight` moves,
to `max(master, error)`. This keeps the row's path stable for the folder scanner.

**6. Test-mode output mirrors the full destination tree.**
`${VMSAM_TEST_OUTPUT_DIR}/<destination_path stripped of its leading slash>/`, i.e.
`<LIBRARY_ROOT>/<show>/<season>` → `${VMSAM_TEST_OUTPUT_DIR}<LIBRARY_ROOT>/<show>/<season>`. The task brief's example
implied a shorter `Anime/x/s1`, which would require stripping an arbitrary first
component and can collide across roots.
> **REFINED 2026-08-27:** the trimming was duplicated in `fusion.py`. It now lives
> once in `tools.build_mirrored_folder_path()`, called by both trees.

**7. `.log.error` files are appended to, never truncated.**
Each attempt writes a `===== Fusion retry <ts> (VMSAM_MODE=…) =====` banner followed by
the exception, traceback and `mergeVideo.errors_merge`, so successive retries accumulate
a readable history. (The legacy `process_rejected_files` opens with `"w"` and loses it.)
This is the one file write that happens on failure in **production** mode: the media
files and the database rows are left strictly untouched, so the entry stays unprocessed
and re-enqueueable. On production success the stale `.log.error` is deleted along with
the error file.

**8. Publication is atomic.**
`publish_merged_file()` stages the merged file as `<destination>.tmp` — same directory,
therefore same filesystem — then `os.replace()`s it over the master in one syscall. The
legacy `os.remove(master)` + `shutil.move()` sequence leaves a window where the episode
has no file at all; this closes it. `shutil.move` handles the cross-filesystem copy from
the temp folder into the staging name.

**9. Output discovery has a glob fallback.**
`mergeVideo` names its output `<reference basename>_merged.mkv`, but it appends `_1` when
two inputs share a basename. `find_merged_file()` tries the expected name, then falls back
to a unique `*_merged.mkv` in the output folder, and raises with both the expected path and
the candidate list when it is ambiguous.

### Database access patterns

- `GET /errors` — one `SELECT` over `incompatible_files`, ordered
  `folder_id, episode_number, file_weight DESC`. Unfiltered full-table scan by design:
  the table only holds unresolved rejects, so it stays small. Add pagination if that stops
  being true.
- `POST /fusion` — two indexed point lookups (`incompatible_files.file_path`, then
  `ix_folder_episode`) purely to fail fast on an unknown path or a missing master, using
  the request-scoped `Depends(get_session)`.
- Worker — opens its **own** session per job from a `sessionmaker` handed to
  `fusion.start_worker()` at startup, and closes it in a `finally`. A request-scoped
  session must never be used here: a merge runs for minutes to hours and would pin a
  pooled connection for the whole time.
- The production commit is single-transaction: the `episodes.file_weight` update is left
  dirty and flushed by the `session.commit()` inside `delete_incompatible_file()`, so the
  weight change and the reject removal land together or not at all.
- `GET /fusion` touches **no** database at all — pure in-memory read under a short lock.

### Performance considerations

- `GET /fusion` is O(pending) over a `deque` copy taken under `state_lock`; the lock is
  never held across IO, so status polling cannot stall the worker.
- The worker rebuilds `video.ffmpeg_pool_audio_convert` (`core_to_use / 1.6`, min 1) and
  `video.ffmpeg_pool_big_job` (1) per job and terminates both in a `finally`, matching
  `process_episode`. Pools are not reused across jobs — a hung ffmpeg child cannot poison
  the next retry.
- The API process now imports `mergeVideo` / `video` / `audioCorrelation` transitively via
  `fusion`. Import cost is one-off at startup and these were already loaded by
  `main_gestionar_show`.
- **Known contention:** the fusion worker and the main integration loop are separate
  processes with no shared scheduler, so a retry can run alongside an automatic merge and
  oversubscribe the CPU. Sequencing is only guaranteed *within* the fusion queue. A
  cross-process lock is the obvious next step if this bites.

### Not done / open

- `VMSAM_MODE` and `VMSAM_TEST_OUTPUT_DIR` are undocumented in the `Dockerfile` (read-only).
- No cross-process throttle between the fusion worker and the integration loop (see above).
- No endpoint to cancel a queued job or to re-enqueue in bulk.
- Endpoints were **not** executed: validation was `python3 -m py_compile` only, per the
  testing directive in `AGENT.MD` and the operator's instruction to defer runtime testing
  to a dedicated environment.

---

<!-- MASTER SUMMARY TABLE: generated by src/tools_batch/master_summary.py -->

## Master Summary Table — all 315 error files

Keyed by the error file's **database id** (`GET /errors`). Resolve an id to a
file through `corpus_index.tsv` in the private notes outside the repository;
names never appear here. `Resolved` requires **both** a produced merge and a
passed sync check (Step 1 local, Step 2 in the container).

| Status | Count |
| :--- | ---: |
| Failed | 173 |
| Step 2 verified; no Step 1 record | 75 |
| Step 2 only; Step 1 could not measure | 32 |
| Merged, unverifiable | 32 |
| Out of sync | 1 |
| Not processed | 1 |
| MERGED_OUT_OF_SYNC | 1 |

## Where the ledger and the re-verification disagree

The status table above counts the **ledger**. 99 files have also been re-verified
in place against the artefact on disk, and **41 carry a different verdict**.

**Both counts are shown. A single reconciled number would repeat the original error**
**with fresher data**, so the evidence behind each is named too.

| Ledger says | Re-verify says | Files | Re-verify rests on |
| :--- | :--- | ---: | :--- |
| MERGED_UNVERIFIED | MERGED_IN_SYNC | 30 | multiple windows, agreeing (29); multiple windows, with dissent (1) |
| MERGED_UNVERIFIED | MERGED_OUT_OF_SYNC | 1 | one window (1) |

**10 further rows are NOT counted as resolved.** They are `FAILED` rows whose
`merged_file` was carried forward from an earlier successful run, so the
re-verify measured an output written **two to three days before the attempt that**
**failed**. Ids: err_25, err_87, err_91, err_95, err_98, err_117, err_127, err_214, err_217, err_406.

*Nothing here writes the ledger.* `reverify_in_place` refuses to, and that refusal
was the fix for real data loss -- a re-merge once collided with another agent's job
and overwrote a row with `enqueue_failed`. **This section prices that cost; it does
not pay it.**

### Completeness — is every audio track as long as the programme?

**Beside the sync verdict, never inside it.** Alignment and completeness are
different questions: a track that is absent has no lag to be wrong about, so a
sync check can pass on a file that is missing most of its audio. No verdict in
the table above is derived from this column.

`not assessed` means **no completeness record for that row** — most often
because it never merged, so there is no output to measure. It does **not** mean
complete.

| Completeness | Count |
| :--- | ---: |
| not assessed | 212 |
| complete | 96 |
| incomplete | 5 |
| inherited_from_master | 2 |

**5 file(s) are recorded as a passing Step 2 while missing a track**, and **4 of those carry a sync verdict that approved them.** That is the condition this column exists to surface: an absent track has no lag to be wrong about, so the check can pass precisely because the audio is missing.

| Class | Count |
| :--- | ---: |
| `C_cut` | 180 |
| `D_trim` | 58 |
| `B_fps` | 35 |
| `E_corr` | 30 |
| `A_no_lang` | 6 |
| `None` | 6 |

| Error ID | Folder | Ep | Class | Step 1 | Step 2 | Final Status | Complete | Detail |
| ---: | ---: | ---: | :--- | ---: | :--- | :--- | :--- | :--- |
| 5 | 143 | 1 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 6 | 86 | 1 | `E_corr` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 7 | 86 | 2 | `E_corr` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 8 | 153 | 24 | `C_cut` | 1 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 9 | 151 | 1 | `D_trim` | 0 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 10 | 86 | 3 | `E_corr` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 11 | 86 | 4 | `E_corr` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 12 | 135 | 2 | `D_trim` | 30 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the second delay test varied too much |
| 13 | 135 | 15 | `D_trim` | 10 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 14 | 86 | 5 | `E_corr` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 15 | 86 | 6 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 16 | 86 | 7 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 17 | 86 | 8 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 18 | 86 | 9 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 19 | 86 | 10 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 20 | 86 | 11 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 21 | 86 | 12 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 22 | 86 | 13 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 23 | 86 | 14 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 24 | 86 | 15 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 25 | 161 | 1 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 26 | 164 | 2 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 27 | 164 | 1 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 29 | 164 | 3 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 31 | 164 | 4 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 33 | 164 | 5 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 35 | 164 | 6 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 37 | 164 | 7 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 44 | 54 | 1 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 45 | 72 | 23 | `D_trim` | 20 ms | pass | Step 2 verified; no Step 1 record | **INCOMPLETE** 1/5 tracks, worst 1322 s |  |
| 46 | 110 | 1 | `B_fps` | 40 ms (a window dissents at 1470) | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 47 | 110 | 1 | `D_trim` | 40 ms (a window dissents at 1470) | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 48 | 54 | 2 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 49 | 54 | 1 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 50 | 54 | 1 | `C_cut` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 51 | 110 | 2 | `B_fps` | 20 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 52 | 110 | 2 | `D_trim` | 20 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 53 | 54 | 3 | `C_cut` | 0 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 54 | 54 | 4 | `C_cut` | 0 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 55 | 54 | 4 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 56 | 54 | 5 | `C_cut` | 0 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 57 | 54 | 5 | `B_fps` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 58 | 54 | 5 | `C_cut` | 0 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 59 | 110 | 3 | `B_fps` | 40 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 60 | 110 | 3 | `D_trim` | 40 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 61 | 54 | 6 | `C_cut` | 0 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 62 | 54 | 6 | `C_cut` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 63 | 110 | 4 | `B_fps` | 40 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 64 | 110 | 4 | `D_trim` | 40 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 65 | 74 | 1 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 67 | 54 | 7 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 69 | 54 | 7 | `E_corr` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 70 | 110 | 5 | `B_fps` | 40 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 71 | 54 | 7 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 72 | 110 | 5 | `D_trim` | 40 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 73 | 74 | 2 | `C_cut` | 1 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 74 | 54 | 7 | `E_corr` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 75 | 54 | 8 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 76 | 54 | 8 | `E_corr` | 1 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 77 | 110 | 6 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 78 | 110 | 6 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 79 | 74 | 3 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 80 | 54 | 9 | `C_cut` | 10 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 81 | 54 | 10 | `C_cut` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 82 | 54 | 10 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 83 | 110 | 7 | `B_fps` | 40 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 84 | 110 | 7 | `D_trim` | 40 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 85 | 54 | 11 | `C_cut` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 86 | 54 | 11 | `C_cut` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 87 | 110 | 8 | `B_fps` | 30 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 88 | 110 | 8 | `D_trim` | 30 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 89 | 54 | 12 | `C_cut` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 90 | 54 | 12 | `E_corr` | 1 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 91 | 110 | 9 | `B_fps` | 40 ms (a window dissents at 2540) | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 92 | 110 | 9 | `D_trim` | 40 ms (a window dissents at 2540) | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 93 | 54 | 13 | `C_cut` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 94 | 54 | 13 | `C_cut` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 95 | 110 | 10 | `B_fps` | 40 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 96 | 110 | 10 | `D_trim` | 40 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 97 | 54 | 9 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 98 | 110 | 11 | `B_fps` | 40 ms | fail | Step 2 only; Step 1 could not measure | not assessed | the delay differs across cuts; the test accepts only one |
| 99 | 110 | 11 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 100 | 54 | 3 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 101 | 54 | 1 | `B_fps` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 102 | 54 | 2 | `E_corr` | 1 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 104 | 54 | 2 | `E_corr` | 0 ms | pass | Step 2 verified; no Step 1 record | complete ⚠ stale artefact |  |
| 108 | 159 | 3 | `C_cut` | — | fail | Failed | not assessed | the second delay test varied too much |
| 109 | 159 | 13 | `C_cut` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 110 | 101 | 23 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 111 | 159 | 23 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 112 | 111 | 2 | `E_corr` | 0 ms | pass | Merged, unverifiable | undeterminable ⚠ stale artefact |  |
| 113 | 164 | 8 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 114 | 62 | 14 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 117 | 62 | 15 | `C_cut` | — | fail | Failed | complete | the delay differs across cuts; the test accepts only one |
| 118 | 154 | 1 | `C_cut` | 2 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 119 | 154 | 1 | `E_corr` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 120 | 62 | 16 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 121 | 40 | 11 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 122 | 62 | 17 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 123 | 40 | 12 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 124 | 62 | 18 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 125 | 32 | 1 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 126 | 153 | 1 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 127 | 62 | 20 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 128 | 32 | 2 | `C_cut` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 129 | 153 | 2 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 130 | 32 | 3 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 131 | 62 | 21 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 132 | 109 | 4 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 133 | 153 | 3 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 134 | 103 | 12 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 135 | 103 | 12 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 136 | 62 | 22 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 137 | 32 | 4 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 138 | 109 | 5 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 139 | 74 | 4 | `C_cut` | 1 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 140 | 153 | 4 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 141 | 103 | 26 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 142 | 62 | 23 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 143 | 103 | 26 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 144 | 32 | 5 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 146 | 74 | 5 | `C_cut` | 1 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 147 | 153 | 5 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 148 | 103 | 1 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 149 | 60 | 2 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 150 | 62 | 24 | `C_cut` | — | pass | Merged, unverifiable | complete ⚠ stale artefact |  |
| 151 | 32 | 6 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 152 | 109 | 20 | `E_corr` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 153 | 153 | 6 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 154 | 103 | 2 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 155 | 60 | 3 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 156 | 62 | 25 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 157 | 32 | 7 | `C_cut` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 158 | 109 | 7 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 159 | 153 | 7 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 160 | 103 | 3 | `C_cut` | 84 ms | pass | Out of sync | **INCOMPLETE** 2/4 tracks, worst 1416 s | Step 2 verified in the container; Step 1 could not |
| 161 | 62 | 26 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 162 | 60 | 4 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 163 | 32 | 8 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 164 | 109 | 8 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 165 | 153 | 8 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 166 | 103 | 4 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 167 | 60 | 5 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 168 | 62 | 19 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 169 | 32 | 9 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 170 | 109 | 9 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 171 | 153 | 9 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 172 | 103 | 6 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 173 | 60 | 6 | `C_cut` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 174 | 32 | 10 | `C_cut` | 10 ms | pass | Step 2 verified; no Step 1 record | **INCOMPLETE** 1/5 tracks, worst 1442 s |  |
| 175 | 109 | 10 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 176 | 153 | 10 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 177 | 103 | 7 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 178 | 60 | 7 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 179 | 32 | 11 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 180 | 109 | 11 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 181 | 153 | 11 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 182 | 103 | 8 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 183 | 60 | 8 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 184 | 32 | 12 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 185 | 153 | 12 | `D_trim` | 1 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 186 | 103 | 10 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 187 | 60 | 9 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 188 | 32 | 13 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 189 | 153 | 13 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 190 | 103 | 11 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 191 | 60 | 10 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 192 | 32 | 14 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 193 | 153 | 14 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 194 | 103 | 13 | `C_cut` | 20 ms | pass | Step 2 verified; no Step 1 record | **INCOMPLETE** 1/4 tracks, worst 1335 s | Step 2 verified in the container; Step 1 could not |
| 195 | 60 | 11 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 196 | 60 | 11 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 197 | 32 | 15 | `C_cut` | 10 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 198 | 153 | 15 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 199 | 103 | 14 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 200 | 60 | 12 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 201 | 32 | 16 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 202 | 153 | 17 | `D_trim` | 1 ms | pass | Step 2 verified; no Step 1 record | not assessed |  |
| 203 | 103 | 16 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 204 | 60 | 13 | `C_cut` | — | pass | Merged, unverifiable | complete | the only track is a copy of the master's |
| 205 | 32 | 17 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 206 | 153 | 18 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 207 | 103 | 17 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 208 | 32 | 18 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 209 | 153 | 19 | `D_trim` | 1 ms | pass | Step 2 verified; no Step 1 record | not assessed |  |
| 210 | 103 | 18 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 211 | 32 | 19 | `C_cut` | 18 ms | pass | Step 2 verified; no Step 1 record | **INCOMPLETE** 6/9 tracks, worst 1000 s ⚠ stale artefact |  |
| 212 | 153 | 20 | `D_trim` | 1 ms | pass | Step 2 verified; no Step 1 record | not assessed |  |
| 213 | 103 | 19 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 214 | 32 | 20 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 215 | 153 | 21 | `D_trim` | 2 ms | pass | Step 2 verified; no Step 1 record | not assessed |  |
| 216 | 103 | 20 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 217 | 32 | 21 | `D_trim` | 10 ms | fail | Step 2 only; Step 1 could not measure | complete | the delay differs across cuts; the test accepts only one |
| 218 | 153 | 22 | `D_trim` | 32 ms | pass | Step 2 verified; no Step 1 record | not assessed |  |
| 219 | 103 | 21 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 220 | 32 | 22 | `C_cut` | 10 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 221 | 153 | 23 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 222 | 103 | 22 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 223 | 32 | 23 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 224 | 103 | 23 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 225 | 32 | 24 | `C_cut` | 0 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 229 | 81 | 4 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 230 | 81 | 5 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 231 | 81 | 6 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 232 | 81 | 7 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 233 | 81 | 8 | `C_cut` | — | — | Not processed | not assessed |  |
| 234 | 81 | 9 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 235 | 81 | 10 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 236 | 81 | 11 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 237 | 103 | 1 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 238 | 103 | 2 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 239 | 103 | 3 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 240 | 81 | 1 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 241 | 103 | 4 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 242 | 81 | 2 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 243 | 103 | 5 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 244 | 81 | 3 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 245 | 103 | 6 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 246 | 103 | 7 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 247 | 103 | 8 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 248 | 103 | 9 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 249 | 103 | 10 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 250 | 103 | 11 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 251 | 103 | 13 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 252 | 103 | 12 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 253 | 103 | 15 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 254 | 103 | 14 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 255 | 103 | 17 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 256 | 103 | 16 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 257 | 103 | 19 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 258 | 103 | 18 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 259 | 103 | 20 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 260 | 103 | 21 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 261 | 103 | 22 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 262 | 103 | 23 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 263 | 103 | 24 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 264 | 103 | 25 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 265 | 103 | 26 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 266 | 153 | 16 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 267 | 172 | 6 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 268 | 172 | 6 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 269 | 173 | 1 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 270 | 173 | 2 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 271 | 172 | 9 | `E_corr` | 30 ms | fail | Step 2 verified; no Step 1 record | not assessed | the delay differs across cuts; the test accepts only one |
| 272 | 173 | 3 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 273 | 172 | 5 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 274 | 173 | 4 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 275 | 172 | 7 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 276 | 173 | 5 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | not assessed |  |
| 277 | 172 | 10 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 278 | 173 | 6 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 279 | 173 | 7 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | not assessed |  |
| 280 | 173 | 8 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 281 | 173 | 9 | `D_trim` | 0 ms | pass | Step 2 verified; no Step 1 record | not assessed |  |
| 282 | 173 | 11 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 283 | 173 | 12 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 284 | 173 | 13 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 285 | 173 | 14 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 286 | 173 | 15 | `C_cut` | — | pass | Merged, unverifiable | complete |  |
| 287 | 173 | 16 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 288 | 172 | 6 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 289 | 172 | 7 | `B_fps` | 0 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 290 | 172 | 5 | `B_fps` | 30 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 291 | 172 | 2 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 292 | 172 | 3 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 293 | 172 | 4 | `B_fps` | 70 ms | pass | MERGED_OUT_OF_SYNC | complete ⚠ stale artefact |  |
| 294 | 172 | 1 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 295 | 172 | 1 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 296 | 172 | 2 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 297 | 172 | 3 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 298 | 172 | 4 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 299 | 172 | 10 | `B_fps` | 20 ms | pass | Step 2 only; Step 1 could not measure | complete |  |
| 300 | 172 | 8 | `B_fps` | 20 ms | pass | Step 2 only; Step 1 could not measure | complete ⚠ stale artefact |  |
| 301 | 172 | 8 | `C_cut` | 0 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 302 | 172 | 9 | `B_fps` | 30 ms | pass | Step 2 only; Step 1 could not measure | complete ⚠ stale artefact |  |
| 303 | 172 | 9 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 305 | 172 | 9 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 306 | 172 | 4 | `C_cut` | 0 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 307 | 176 | 10 | `C_cut` | — | fail | Failed | not assessed | the two delay methods disagree |
| 316 | 177 | 12 | `C_cut` | 47 ms | pass | Step 2 verified; no Step 1 record | complete ⚠ stale artefact |  |
| 318 | 181 | 7 | `E_corr` | 10 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 319 | 181 | 4 | `D_trim` | 10 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 320 | 181 | 8 | `E_corr` | 10 ms | pass | Step 2 verified; no Step 1 record | complete |  |
| 349 | 17 | 8 | `C_cut` | — | fail | Failed | not assessed | every candidate dropped as incompatible; 1 file left |
| 352 | 196 | 2 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 353 | 196 | 13 | `C_cut` | 1 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 375 | 122 | 9 | `D_trim` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 403 | 242 | 1 | `C_cut` | 1 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 406 | 248 | 11 | `E_corr` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 407 | 219 | 3 | `C_cut` | 14 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 422 | 219 | 4 | `C_cut` | 20 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 428 | 219 | 5 | `C_cut` | 8 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 431 | 219 | 6 | `C_cut` | 10 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 433 | 219 | 7 | `C_cut` | 6 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 675 | 219 | 8 | `C_cut` | 8 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 677 | 219 | 9 | `C_cut` | 9 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 678 | 230 | 3 | `A_no_lang` | — | pass | Merged, unverifiable | complete |  |
| 679 | 230 | 4 | `A_no_lang` | — | pass | Merged, unverifiable | not assessed |  |
| 680 | 230 | 5 | `A_no_lang` | — | pass | Merged, unverifiable | not assessed |  |
| 681 | 230 | 6 | `A_no_lang` | — | pass | Merged, unverifiable | not assessed |  |
| 682 | 230 | 7 | `A_no_lang` | — | pass | Merged, unverifiable | not assessed |  |
| 683 | 219 | 10 | `C_cut` | 5 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 684 | 230 | 8 | `A_no_lang` | — | pass | Merged, unverifiable | not assessed |  |
| 686 | 219 | 11 | `C_cut` | 6 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 687 | 219 | 12 | `C_cut` | 6 ms | pass | Step 2 verified; no Step 1 record | complete | Step 2 verified in the container; Step 1 could not |
| 688 | 283 | 1 | `B_fps` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 691 | 146 | 1 | `C_cut` | — | pass | Merged, unverifiable | undeterminable ⚠ stale artefact |  |
| 692 | 225 | 11 | `C_cut` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 693 | 211 | 17 | `E_corr` | — | fail | Failed | not assessed | an ffmpeg command failed |
| 694 | 62 | 42 | `None` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 695 | 62 | 41 | `None` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 696 | 62 | 43 | `None` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 697 | 62 | 44 | `None` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 698 | 62 | 45 | `None` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |
| 699 | 62 | 46 | `None` | — | fail | Failed | not assessed | the delay differs across cuts; the test accepts only one |

<!-- END MASTER SUMMARY TABLE -->
