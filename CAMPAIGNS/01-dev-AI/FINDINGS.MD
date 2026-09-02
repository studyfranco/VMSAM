# Findings — living report

Updated as results land, not at the end of a cycle. Every number here is
measured and carries its denominator. Anything not established is marked
**NOT CLAIMED**. Files are named by error id; resolve ids through the private
notes outside the repository.

Last updated 2026-09-01.

---

## 1. Output truncation — seven files, mechanism narrowed, cause not found

**Established.** Over **115 published outputs** (the publisher's tree, not a
ledger-derived list):

| | count |
| :--- | ---: |
| complete | 106 |
| lost most of its audio | 6 |
| ends early | 2 |
| empty | 1 |

**CORRECTED 2026-09-01: the phenomenon is FIVE files, not seven.** The
discriminator is *short beside full in the same output* — and that **also matches
an output whose MASTER was already short.** Five files were verified against their
sources; the set was then extended by three **without re-applying the test that
qualified it.** On checking, two of those three carry tracks that are the same
length in their own master: the merge copied a short track faithfully and
truncated nothing. The third has no master recorded and is **undetermined**.

**CORRECTED AGAIN: FOUR are established, not five.** These are the numbers that
supersede every earlier count in this file's history:

| | count | ids |
| :--- | ---: | :--- |
| truncation, **established** | **4** | err_45, err_160, err_174, err_211 |
| consistent with the mechanism, not established | 1 | err_194 |
| inherited — the master's own track is short | 2 | err_112, err_691 |
| undetermined, no master recorded | 1 | err_233 |

**Not 8, not 9, not 7-plus-2. Four.**

**The detector needs a master comparison it does not have.** A short track is only
evidence of truncation if the source was long.

**And the check that was supposed to supply it silently measured nothing.** The
"sources are clean" test computed

    max([container - track for tracks WITH a tag] + [0.0])

so **a source carrying no duration tags at all returns 0.0 and reads as CLEAN.**
Two candidates are untagged. One survives because its master settles it
independently; the other does not — its only short track is in a language **the
master does not carry**, and its candidate has no tags, so **both references are
silent.** It remains consistent with the mechanism — its shortfall is exactly what
the formula predicts — but consistent is not established.

Settling it needs a per-track length that does not depend on the `DURATION` tag:
packet counting or a decode. **Not run.**

**What the mechanism is not.** Each of these was a live hypothesis and each was
refuted by measurement:

- **Not inherited from short sources.** Every candidate and master behind the
  seven is full length: worst source shortfall 2.0 s against a worst output
  shortfall of 1440.9 s. **The merge creates it.**
- **Not candidate-specific.** Three of the affected outputs lose a track in a
  language **the candidate does not carry at all**.
- **Not per-track damage.** Where several tracks are short they stop together —
  spreads of 0.000, 0.001 and 0.025 s. Six tracks within 25 ms means the
  assembler loses one *input* at an instant.
- **Not the planner.** No cut lands on a recorded plan boundary; nothing within
  5 s. A round-number coincidence was tested against its own
  multiple-comparisons budget (1.83 hits expected by chance, 2 found) and
  **not upgraded**.
- **Not a resource or load-dependent timeout.** A re-merge reproduces the
  truncation **to the millisecond**.

**Surviving constraint:** deterministic, per-file, at an unstructured instant,
on no planner boundary, and able to cut either side of the merge.

### The defect is upstream of `mkvmerge`

**Established 2026-09-01, from the shim's first capture on a known case.** The
merge was re-run and **reproduced exactly**: six tracks short at 441.473–441.486
s, **spread 0.013 s**, three tracks full at 1441 s.

**Both `mkvmerge` commands exited 0, and neither was told to truncate.** The
219-argument final insert carries only `--default-track-flag`,
`--forced-display-flag`, `--hearing-impaired-flag`, `--no-chapters`,
`--original-flag`, `--track-order`, `-A -D -S -a -o`. **No `--split`, no
`--append`, no duration limit, no `--sync`, and no value resembling 441 anywhere
in the argv.** It is a pure track-selection and flag-setting step, and it copied
what it was given.

**So the short tracks were already short on the way in.** The chain, with every
stage reporting success:

1. chimeric assembly emits an intermediate whose candidate-side tracks stop at
   441.5 s
2. `verify_chimera` passes it — **its only length test is container against
   container**, and the container is full length
3. `mkvmerge` copies the short tracks and exits 0
4. publication moves the result into place

**Nothing in that chain can notice**, because the only per-stream duration the
code has is the container's (§2). This is the first point at which §1 and §2
connect.

**The chain above was an inference, it was marked NOT CLAIMED, and MEASUREMENT
REFUTED IT.** With retention enabled, one re-run's chimeric intermediate was
weighed: container 1441.961 s and **seven of seven audio tracks at 1441.960 s.
Zero short.** The prediction on record was ~441.5 s and it was wrong.

**So chimeric assembly is exonerated and the loss is downstream of it.** Step 1
of that chain is dead.

**And the refutation sharpens the constraint rather than reopening the
question.** The intermediate carries exactly the seven candidate-side languages;
the published output carries nine — those seven plus two master-side. Of the
seven, **six come out at 441.5 s and one comes out full at 1441.942 s.**

**The spared track is the language the plan was fitted to.** So the defect is
**not per-file, not per-side, and not uniform across the candidate: it spares
exactly one language, and that language is the one with a reference.** That is
the sharpest constraint obtained on this defect.

### The mechanism, found at the extraction stage

**`mergeVideo.generate_new_file_launch_cmd` (line 2780) cuts the prepared stream
to `duration - delay`:**

    cmd_first_pass.extend(["-strict","-2","-t",
        str(Decimal(duration_best_video) - (Decimal(delay_to_put)/Decimal(1000))), ...])

`-t` is an ffmpeg **output duration limit**. Units are consistent — `delay_to_put`
is milliseconds and the same value is handed to `mkvmerge` as `--sync` — so this
is **not a unit bug. The arithmetic is right; the input is not.**

**WITHDRAWN — that fit was circular, and it was reported here as evidence.** The
"implied delay" for each file was computed as *shortfall = delay*, which is the
formula rearranged. **Given the formula, every observed shortfall implies exactly
the delay that reproduces it, for every file, whatever the data.** A column that
fits every case by construction is the mechanism restated, not support for it.
The boundary case — an "empty" track being this mechanism with the delay equal to
the duration — is the same restatement and carries no independent weight either.

**AND THE MEASUREMENT DOES NOT SUPPORT IT.** With the recorder deployed, one
re-run of the worst case observed:

| | predicted | observed |
| :--- | ---: | ---: |
| `delay_to_put_ms` | 1000463 | **0.0** |
| `duration_best_video_s` | 1441.942 | 1441.942 |
| `first_pass_limit_s` | 441.479 | **1441.942** |

**`duration_best_video_s` was the only independent number of the three, and it is
the one that matched** — to the millisecond, which establishes the recorder fires
at the right call site on the right file. **The delay is zero.**

**Why one language survives.** `delay_to_put` is per `video_obj` — **per source
file, not per track.** One prepared stream per input, so the candidate's stream is
cut by the candidate's delay and **all its tracks go short together** (the 25 ms
spread), while the master's stream keeps its own delay and stays full. The output's
surviving tracks are the master-derived ones.

**Why it reproduces to the millisecond.** The delay is measured deterministically
from the same media: same delay, same `-t`, same instant, across runs and builds.

**Why nothing catches it.** `mkvmerge` is handed an already-short input and copies
it faithfully, exit 0; `verify_chimera` compares container against container;
every per-stream duration in the module is the container's (§2); and
`delay_to_put` is written with a **bare `print` to stdout**, which does not
survive the job.

**ESTABLISHED by reading source**: the `-t` limit exists and is computed from
`delay_to_put`; units are consistent; and `generate_new_file_launch_cmd` is
**not** closure-protected, so it is editable. **Established by measurement**: the
recorder fires at that site, and `limit = duration - delay/1000` holds
arithmetically. **That is the instrument working, not the mechanism.**

**BOTH ffmpeg STAGES ARE RULED OUT**, on a prediction committed before the log
existed. All four invocations carry `-t 1441.942` and nothing else: **no `-ss`, no
`-itsoffset`, and nothing in 482 arguments containing `441.4` or `1000.4`.** A
sweep for every duration-limiting flag returns `-t` alone, no filters.

**The inputs tighten the interval further**: the first pass reads the master and
the chimeric; the second reads each first pass's own output. **The chimeric enters
at `-t 1441.942` with no seek**, and the retained chimeric measured 1441.960 with
every track full, starting at 0. **So nothing between the chimeric and the end of
the second pass removes 1000 s.**

**The remaining interval cannot be bisected from the host**: every intermediate
lives in the container's own `/tmp`, and the tree is cleaned on completion. The
next instrument is **a probe of each intermediate's per-track durations recorded
as each stage writes it** — the recorder pattern applied to the *artefact* rather
than the command, naming the site by bisection instead of by reading code.

### The site is the SECOND PASS, by bisection across a closed interval

**Measured, one merge, all four stages:**

| stage | result |
| :--- | :--- |
| chimeric_out | 7 tracks, 1441.960–1441.961, all start 0 |
| first_pass_out | 7 tracks, 1441.947–1441.959, all start 0 |
| **second_pass_out** | **jpn 1441.942, and six at 441.473–441.485** |
| mux output = published file | same six short |

**The second pass receives a full-length input and emits six short tracks.** The
mux is exonerated. **The ffmpeg stages are back in scope** — which was the
committed refuting branch, and it is the one that happened.

**And the argv for that same invocation carries something the first read did not
draw out.** Alongside `-t 1441.942`, no `-ss`, no `-itsoffset`:

    -c:a:0 flac -compression_level:a:0 12 -sample_fmt:a:0 s32
    -exact_rice_parameters:a:0 1        ... identically for a:1 through a:6

**Every one of the seven audio streams is re-encoded to FLAC at compression level
12**, under a global `-c copy -copyts`. **The second pass is not an audio copy at
all — it is seven simultaneous slow encodes, six of which stop at ~441 s while the
seventh completes.** Those options were in the log two runs earlier; only the
duration-limiting flags were read, because those were what the hypothesis was
about.

**NO MECHANISM IS NAMED INSIDE THAT STAGE.** Input full, `-t` full, every stream
starting at 0, six of seven outputs short. **That is a bounded reproducible fact
and the evidence stops there.**

### ffmpeg exits 0, in 114 seconds, and reports success on truncated output

**The timeout reading is dead.** The second-pass argv was written at 08:03:55Z and
its output probed at 08:05:49Z — **elapsed ~114 s against a 3600 s timeout**, so
no timeout fired and no restart fired. And the launcher **raises on any non-zero
exit**; the merge continued past that call and published, so **the exit code was
0.**

> **ffmpeg exited 0, in 114 seconds, and produced six tracks truncated to ~441 s
> from a full-length input, reporting success.**

**The master chain is the control and it fits.** Its second pass took **~1 s for 4
tracks with 32 arguments**; the chimeric chain took **114 s for 7 tracks with 88
arguments**, carrying FLAC level 12 on every stream. **One second is a copy, 114
is an encode — and only the encoding chain truncates.**

**This also settles the surviving stream.** Since the run was not cut short,
*"jpn finished first"* is not available: **jpn is not the stream that got further
before a kill, it is the stream that came out whole from a run that completed.**
A third statement, distinct from both the per-source explanation and the
finished-first one.

**The command alone does NOT reproduce it.** The recorded 88-argument second pass
was replayed **verbatim** against a synthetic 7-track FLAC input of the same
length, on this host, with no container and no deploy: **exit 0, 121 s elapsed
(against 114 s in the pipeline), all seven tracks full at 1441.942.** The branch
genuinely ran — the output is `s32` throughout.

**A second variable was eliminated in the same session.** The pipeline's surviving
track was also its longest (1441.959 against six at 1441.947); reproducing that
12 ms asymmetry changed nothing — all seven full again. **The survivor being the
longest is a coincidence of that file, not a mechanism.**

**Eliminated, none of which truncates a clean multi-track input**: FLAC level 12,
`s32`, seven simultaneous streams, `-c copy` with per-stream overrides,
`-copyts`, `-max_interleave_delta 0`, `-max_muxing_queue_size 16384`, `-t`, and
unequal stream durations.

**So a property of the real input is required**, which is a much narrower search
than "somewhere in this stage". The leading candidate: the real input carries
video and a large number of subtitle and attachment streams — the mux track order
runs to `1:77` — and `-map 0 -copy_unknown` maps them all while `-vn` drops only
video. **Sparse subtitle packets under `-max_interleave_delta 0` are the least
synthetic-like thing about it**, and the muxing-queue cap exists for a reason.

**And the tags were honest.** Every truncation figure had come from a `DURATION`
tag; the published output was decoded to check. The short track yields PCM at 100
and 400 s and **none at 500, 1000 or 1400**, while the intact one yields PCM at all
five. **The audio really does stop.**

### Reproduced from scratch: four necessary conditions

**Nothing from the library is involved.** Four commands build it:

    ffmpeg -f lavfi -i "sine=frequency=440:duration=1441.96:sample_rate=48000" \
           -af asetnsamples=n=304:p=0 -ac 2 -sample_fmt s32 -c:a flac DENSE
    printf '1\n00:07:20,000 --> 00:07:21,735\nx\n' > CUE
    ffmpeg -i DENSE -i CUE -map 0:a -map 1:s -c copy INPUT
    ffmpeg -i INPUT -map 0 -c copy -c:a:0 flac OUTPUT

    audio DURATION 00:07:21.971 instead of 00:24:01.960 — three runs, identical

**Four conditions, all necessary:**

1. the audio is **re-encoded**, not copied
2. an **early-ending subtitle** is mapped into the same output
3. the audio has **small frames** — ~304 samples, against the 4096 default
4. the **duration is long**

**Condition 3 was reached with `-af asetnsamples`.** The FLAC encoder has no
`-block_size` option, so an earlier attempt to set it was **silently ignored — a
missing option reported as acceptance**, the same shape as everything else in this
document, this time inside ffmpeg's own argument handling.

**Condition 4 is why a 60 s test came back clean and was reported as eliminating
density.** Matched density at 60 s: full length. Matched density at 1441.96 s:
truncated. **The elimination was correct at the scale tested and wrong as a
general claim.**

The densities line up with the real file: 229,948 packets (real, truncates),
227,678 (synthetic matched, truncates), 67,592 (synthetic default, full).

### CORRECTED: the trigger is the START of a long subtitle GAP, not the end of a stream

**A prediction was committed for a second file — that its 108.010 s truncation
would match a subtitle *ending* near 105 s — and it was refuted.** Both of that
file's subtitle streams end at 1406.457 and 1436.617 s.

**What was there instead**: one stream with 35 packets and **a single gap of
1222.5 s**, running from 107.902 to 1330.387. **The audio truncated at 108.009 and
108.012 — 0.107 s past the START of that gap.**

| file | gap | audio stops |
| :--- | :--- | ---: |
| first | 1000.2 s from 441.735 | 441.899 (+0.164) |
| second | 1222.5 s from 107.902 | 108.009 (+0.107) |

**The first file's subtitle simply stopped and never resumed — a gap running to
the end of the file.** That is a special case of this rule, and it was mistaken
for the rule itself.

**The synthetic sweep re-reads consistently under it**, as gap length rather than
cue position:

| cue ends | gap | audio |
| ---: | ---: | :--- |
| 2 s | 1439.9 s | 2.229 — at the gap start |
| 121 s | 1320.9 s | 121.232 — at the gap start |
| 442 s | 1000.0 s | 441.971 — at the gap start |
| 721 s | 720.9 s | **1441.960 — full** |

**Four of six fit.** The threshold is bracketed between **720.9 s (full)** and
**1000.0 s (truncated)** by exactly one pair, and **no number is fitted inside
it.** The remaining two points — small gaps near the end — destroyed the audio and
varied run to run: **a different, non-deterministic failure this rule does not
cover.**

**Two real files with different cut instants, one rule, plus a reproduction from
primitives.** The second file was spent on a prediction that failed, and that was
worth more than confirming the one already held.

**And the reason nothing downstream ever sees it**: on the second file the
second-pass container came out at **1436.617 — the end of the *other* subtitle
stream.** **The container follows the longest subtitle while the audio follows the
gap**, so the container duration looks entirely plausible.

### It is a reproduction, not a law

Varying **only** where the single cue ends, same dense audio:

| cue ends | audio result |
| ---: | :--- |
| 2 s | 2.229 — just past the cue |
| 121 s | 121.232 — just past the cue |
| 442 s | 441.971 — just past the cue |
| **721 s** | **1441.960 — FULL** |
| 1201 s | 0.025 / 0.006 / 0.012 — **nearly empty, different each run** |
| 1439 s | 0.019 — nearly empty |

**Three regimes, not one.** Below ~442 s it truncates to just past the cue,
deterministically. At 721 s it is full. Past ~1200 s the audio is destroyed and
**the result varies run to run — a race, not a rule.**

**The regime matching every real observation in this corpus is the first.** **No
threshold is fitted to six non-monotonic points**, and no mechanism is offered for
the other two regimes.

**An upstream report is now supportable** — it rests on nothing from the library —
**but it must carry the non-monotonic sweep**, or it reports a rule that does not
hold.

**Still not claimed**: the mechanism inside ffmpeg.

**The missing measurement WAS whether the process succeeded.** The argv recorder
does not capture an exit code, and the launcher's result is examined only by a
keyword scan writing to container stderr, which does not survive the job. **So
"ffmpeg finished and produced this" and "ffmpeg was killed partway and this is
what had been written" are indistinguishable in everything logged** — the same
defect as §97 and §102, now in a third instrument. The launcher is called with a
retry count and a 3600 s timeout, which makes the second reading testable rather
than speculative.

**THE MECHANISM IS UNESTABLISHED.** One ffprobe separates the two live cases: with
a delay of 0 this merge should not truncate anything, so **a full-length output
means the run did not reproduce the failing condition — unestablished, not
refuted — while a still-short output with the limit equal to the full duration
REFUTES it outright.**

**And a live possibility either way**: if `delay_to_put` is genuinely 0 on the
current build for a file that previously lost 1000 s of audio, **something
changed the delay between then and now** — which would make this file already
fixed, and would make the other four a question about the *current* build rather
than the one that produced the artefacts. **The four established truncations were
measured on EXISTING outputs, which the deployed code did not necessarily
produce.** Not asserted.

**INFERRED, NOT MEASURED**: that these files' delays hold those values. The
arithmetic fits all seven and the boundary case fits a prediction it was not
fitted to — but **no delay value has been observed, because the only channel
carrying it evaporates.** Closing it needs `delay_to_put` persisted, or one run's
worker stdout captured.

**This does NOT say the delay measurement is wrong.** It says **whatever delay is
measured is subtracted from the duration with no sanity bound.** A delay of
1 000 463 ms on a 1441 s file is not a plausible audio offset, and a bound
rejecting a delay comparable to the runtime would have refused it. **Whether the
delay is spurious, and why, is a separate question at a different site.**

**One case is held apart rather than absorbed**: its two short tracks stop
**80.224 s** apart against 0.000–0.025 s for every other multi-track case, and
its container is 5997.5 s against ~1450 s elsewhere. Two agents disagree on
whether it belongs; the spread is the only discriminator and neither converged
by agreement.

**NOT CLAIMED:** why the assembler loses an input at an instant, or why the
instant differs per file. An output carrying a **1 ms** track is plausibly an
*empty* track rather than a *cut* one — a different event, not folded into the
count.

**New instrument, working end to end.** `mkvmerge` now runs through a logging
shim shipped inside the container image, capturing argv, exit status and elapsed
time for every invocation. First capture on a known truncation case: a split
merge (argc 10, 4 s) and a final insert (argc 219, 5 s), **both exit 0**. That
record has never existed before. Caveat for whoever reads the log: the `-o`
target is a **staging name** that publication has already renamed away, so shim
lines cannot be matched to published outputs without following the move.

---

## 2. Per-stream durations are the container's, for 1926 of 1927 tracks

**Established.** `probe()` **does** request `stream=duration` — the code looks
correct on inspection — **but Matroska does not answer it.** Per-stream length in
Matroska lives in the `DURATION` **tag**, and `stream_tags=language,title` is
requested without it. Measured on one real file: `stream=duration` **absent**,
`stream_tags=DURATION` **1443.904 s**, container **1444.192 s** — a 288 ms gap.

**The mechanism matters because it is invisible at the call site**: anyone reading
`probe()` sees it ask for a per-stream duration and stops there. **This is a
fallback that looks like a field, not a missing request.** The `except`
substitutes the **container's** duration under a per-stream key. Measured over
549 files and 1927 audio tracks: **1926 take the fallback**; the single exception
is non-Matroska. **626 tracks (33.7%)** differ from their container by more than
0.1 s and **285 by more than ten seconds**.

**Consequence 1, the planner** believes tracks are longer than they are.

**Consequence 2, worse: a donor-selection guard enforces the wrong
proposition.** `if stream["duration"] < c_hi: continue` was written to reject a
donor too short for the requested window. It cannot: every stream reports the
container's length. It is **not dead** — the caller reaches it only after a
consistency check bounds `c_hi` to the container plus `slack_s = 2.0`, so its
entire live range is a **two-second band at the end of the file**. What it
actually enforces is a boundary condition, not a length check. **A genuinely
short donor is selected and read past its end.**

**Fix scoped, not built.** One probe argument plus a parser, with a hard
requirement: **record which source the number came from**, so a caller can
distinguish a measured length from a substituted one. Expected consequence,
stated before building: **the guard starts firing**, so merges that today emit a
silently short track may instead raise — the right direction, but a behaviour
change, so it goes behind a flag, off by default, with the default proven
unchanged.

**NOT CLAIMED:** this does not cause the truncation in §1. Those cuts land on
full-length sources. It guarantees the pipeline cannot **notice** a short track;
it is not evidence that it creates one.

---

## 3. A failed run silently orphans a completed output

**Established and FIXED (landed).** Twelve published outputs were referenced by
no ledger row. Cause: `process_one` builds a fresh row per outcome and sets
`merged_file` only on success; the writer then replaces the row wholesale.
**Nothing decides to clear the field — it is simply absent from the new row.**
The artefact survives on disk; the pointer does not.

**The discriminator was not time but later re-running**: 24 outputs published in
the same window had no row rewritten, while the twelve were re-run and failed
twice. An apparent "23.5-hour episode" was **publication times, not defect
times**.

**Fifteen readers consume the field and all fail the same silent way.** The worst
is not an analysis tool: a cleared `merged_file` makes an output **ineligible for
preservation**, so an orphaned artefact is excluded from backup at exactly the
moment it is least protected.

**Fix landed**: the writer carries `merged_file` forward when the new row has
none, **conditional on the path still resolving**, plus `merged_file_carried_from`
and `merged_file_carried_count`. The count matters — an earlier hand repair was
erased by the next failed run, because **a marker that lives in the replaced row
cannot record that the row was replaced.**

**NOT CLAIMED:** one of the twelve is unidentified — 4.15 GB, written
2026-08-30T14:39:25Z, matching no manifest entry. Not guessed at.

---

## 4. The 15 ms Step 1 bar was below the quantisation floor

**Established.** `adjust_delay_to_frame` does `round(delay / frame_ms)` on CFR
references, so a **correctly** snapped delay still carries up to half a frame:
**20.85 ms at 23.976 fps**, 20.0 ms at 25 fps. A mission bar of 15 ms was
therefore unachievable by construction for CFR audio.

**Owner ruling 2026-09-01:** the bar becomes **half a frame, per file**, keeping
15 ms for VFR references.

**This is not a lowered standard.** A delay can be snapped to the **wrong** frame
and still read under half a frame. `record_frame_snap` stores `residual_frames`
for exactly this, and **each affected file must be shown snapped to the NEAREST
frame** before it counts as Resolved. Files on an adjacent frame stay over the
bar. **Open:** how many of the 27 have a usable snap record — if most do not, the
ruling cannot be applied to them and that is itself the answer.

---

## 5. Counting defects found in our own instruments

Recorded because they distorted reported numbers, not as process notes.

- **A population reached through a derived file inherits its coverage.** One scan
  found 108 outputs where the publisher's tree holds 115; twelve were never in
  the population. Eleven of the twelve were clean, so the gap cost exactly one
  finding.
- **A population reached by walking a directory is too broad.** Another sweep
  counted 120 by including five benchmark artefacts. **Both counts were wrong,
  in opposite directions**, and neither agent was reading a definition of
  *published output* — each was reading a convenient proxy. It took two
  independently-wrong counts to find 115, because a single count agrees with
  itself.
- **A count of ids is not a count of outputs**: 108 error ids map to 103 distinct
  merged files.
- **A threshold measured against the wrong reference is wrong in both
  directions**: a fraction-of-container bar both cleared a track 693 s shorter
  than its siblings and flagged 18 padded tracks as truncated. Comparing each
  track to **the longest track in its own file** removes the common offset.
- **A count that moves with the threshold is a property of the bar**, not of the
  phenomenon. Report the stable set; name the marginal cases rather than adding
  them.

---

## 6. The merged-unverifiable files are a CEILING, not a backlog

**Established.** These merged; only Step 2 verification failed. They were chosen
as the best chance of moving files per token spent. **They cannot be moved, and
the reason is not compute.**

Bucketed by per-language track counts across candidate, master and output, over
the files carrying an unverifiable verdict with an output on disk:

| | count |
| :--- | ---: |
| the candidate's shared-language track was DROPPED | 58 |
| the candidate's track was KEPT | 6 |
| contributed-only languages | 1 |

**CORRECTED: all 58 dropped cases are undecidable. There is no actionable
residue.** One file was reported as the single case where locale metadata proved a
distinct dub was dropped. **It proves nothing.** The locale keyword list contained
`'us '` for the US locale and matched *inside* `"op-us 2.0"` — the candidate's
title is *"Japanese - Opus 2.0"* and the master's is *"PCM 2.0"*. **Both are format
names; no locale is involved.**

**This strengthens the finding rather than weakening it**: the ceiling is complete
rather than nearly complete, and it removes the one file anyone might have spent a
day on.

**The general form, because it will recur**: a keyword list matched by substring
will do this again — `uk`, `us `, `spain` are all short enough to appear inside
unrelated words. **A locale test needs word boundaries and a closed vocabulary,
not `in`.**

**And audio cannot break the tie.** 19 metadata-labelled different regional dubs
on this corpus read **0.8334–0.9922**; within-master format variants of the same
recording read **0.9270–0.9894**. The ranges **overlap, with the dub range
extending higher**. No fidelity threshold separates the two cases.

**So the limit is informational, not computational.** For 57 files the drop
decision cannot be re-evaluated at all — not merely that re-merging would not
help. **No amount of compute resolves it, because the information is not in the
artefacts.**

**NOT CLAIMED:** that those 57 drops were wrong. They may all have been correct
deduplications. **The finding is that nobody can tell** — which is more useful
than a suspicion, and is what should be planned against.

**The only thing that would move this class is at ingest time.** The merge
records SUBTITLE decisions per track with an md5, and records the audio decision
as the single line *"Keep the best audio"*. **If audio drops were logged with the
same detail as subtitle drops, the question would be answerable for every future
merge.** That is a fix for what comes next, not a recovery of these.

**Open denominator**: two agents count this population as 65 and 55, and bucket
it 58/6/1 and 35/18/2 respectively — the second by output-side provenance, which
is a different question. Two files land in the KEPT bucket under both methods
independently. **Not reconciled; do not quote a single figure yet.**

---

## 7. Nothing dates an artefact to the build that produced it

**Measured over the 108 ledger rows whose output exists on disk:**

| | count |
| :--- | ---: |
| no `build` field at all | 44 |
| `build` recorded, no timestamp on the row | 26 |
| **row written BEFORE the file it names** | 5 |
| `build` and a row written within an hour after the file | 33 |

**CORRECTED: tested properly — the output's mtime falling inside its own job's
execution window, rather than timestamp adjacency — 134 of 136 datable rows are
coherent, and 32 of the 34 re-verified files ARE attributable to a build.** The
earlier "at most 33 of 108, and probably none of this set" was carried from a
different population and never re-measured (§104).

**The field is broadly usable.** The two failures name a build committed **7.2 and
7.7 hours after their rows were written**, which is a timestamp ordering rather
than an inference.

**What remains true**: nothing in the store *states* that a file was produced by a
build — the coherence above is inferred from execution windows, and 28 rows in the
unverifiable class carry no timestamp at all.

**The `build` field is weaker than the count suggests.** It is stamped when a
**row** is written, and a row can be written by a sweep that merely *verified* a
file. Measured on a separate question the same night: **18 rows carried a
campaign's build for files whose output was never rewritten** — identical to the
byte, with a preserved copy whose mtime was later. So a row with a build and a
plausible timestamp is evidence about **when something looked at the file**, not
about what made it. **Five rows were written before the file they name**, which is
a timestamp ordering rather than an inference: those cannot describe the artefact
that now exists.

**Why this matters more than it sounds (§101).** Every finding in this document
about a merge defect was measured on **outputs already on disk**. *"This artefact
is broken"* is established. *"The current pipeline breaks artefacts"* is a
separate claim, and the step between them was taken repeatedly tonight without
support. The two are separated by when the artefact was built — **the one thing
nothing records.**

**This does not make the findings wrong.** The five truncations were measured
against their own sources and stand as facts about those files.

**The fix is cheap at ingest and impossible in retrospect**: stamp the producing
commit at the moment of production, the way the resampling tag already stamps a
transformation onto the tracks it changes. **It pairs with the audio-drop record
(§6) — both are one line at the point of decision, and both are unrecoverable
afterwards.**

---

## 8. An entire outcome class has no member recorded by a working instrument

The fix that made the verifier load `[software]` landed at **2026-08-31 20:47Z**.
The build deployed at that moment was **not** a descendant of it, and did not
become one for **10h19m** — during which the sweep kept merging and every verdict
was recorded by an instrument that **could not verify**.

**68 of 136 dated ledger rows fall inside that window**: 38 `FAILED`,
27 `MERGED_UNVERIFIED`, 2 `MERGED_IN_SYNC`. The 38 failures are the queue's own,
and they are the larger half.

**And the `MERGED_UNVERIFIED` class has no clean member at all:**

| | count |
| :--- | ---: |
| `MERGED_UNVERIFIED` rows | 63 |
| dated | 35 |
| inside the gap (all on the pre-fix build) | 27 |
| dated outside the gap — **every one predating the fix commit** | 8 |
| **recorded by a build carrying the fix** | **0** |
| undated | 28 |

**That claim is WITHDRAWN — it was made before the experiment that tests it.**
34 re-verifications under a build carrying the fix, all previously
`MERGED_UNVERIFIED`: **30 passed, 1 failed, 2 inconclusive, 1 mixed.** The
remainder is not empty; it has four members, and the genuine *could-not-decide*
case does exist. **Roughly 4 in 34 on this sample**, not extrapolated to the 63.

**What survives is that the instrument, not the media, was the blocker for most
of them** — the verdict was a statement about the verifier.

**Measured, and the qualifier is stronger than the headline.** Of the 30 that
passed, **exactly one measured two languages; the other 29 rested on a single
language each.** Across the 34: **37 language tracks attempted, 75 not
measurable — 67% never compared.**

**The number to publish is 25, not 30.** Five of the 30 sit at or above the
verifier's own copy guard — three indistinguishable from the master measured
against itself, two near-identical content displaced ~20 ms — and each attempted
one language, so the whole file-level pass rests on that comparison.

**But 67% unmeasured is not 67% exposure.** 58 of the 75 are `no_candidate_track`:
the candidate carries no track in that language, so the output's is master-derived
and unaltered — **declining is correct and nothing was missed.** The real exposure
is the **17 `no_master_reference`** cases: candidate-contributed languages the
master never had, **unverifiable by construction**, which no amount of container
time touches. That class is 46 tracks across the wider set.

**One convergence worth recording**: the single failure fails on the same language
that an unrelated instrument had already flagged as one of the five established
truncations — 108.010 s against a master's 1522.7 s. **Two instruments with
nothing in common, agreeing.**

**This invalidates the SUBJECT of the ceiling finding in §6, not its reasoning.**
That analysis treated the unverifiable verdict as a statement about the media.
The bucketing logic stands; what it was about does not. **The buckets are not
restated until the class has one member from a build carrying the fix.**

**One re-verification under the current image settles it**, and it needs no merge:
the in-place tool loads the config itself.

**The generalisable part, beside §102**: **28 of the 63 rows carry no timestamp at
all**, so they cannot be attributed to any build. **A verdict that cannot name the
build that produced it is not a measurement.** And nothing anywhere compared the
instrument's build against the fix's build — the same absent reconciliation as
the manifest against its directory (§7), the inbox against the ledger, and the
publisher's tree against the ledger.

---

## 9. Rubberband's processing cost does not scale with the stretch

**Measured on six files, 6 of 6:**

| method | fidelity range | median |
| :--- | :--- | ---: |
| `asetrate` | 0.9996 – 0.9999 | **0.9997** |
| `rubberband` | 0.8454 – 0.8655 | 0.8553 |

**`asetrate` wins at 1.7 cents as it did at 72.4 cents**, so the policy — asetrate
for both PAL and non-PAL — is unchanged and better supported.

**The prediction was refuted, and that is the finding.** Rubberband was expected to
read ~0.974, matching the pitch-only curve at 1.7 cents. **It reads 0.855.**

| | |
| :--- | ---: |
| pitch-only control at 1.7 cents | 0.9742 |
| rubberband at 1.7 cents | 0.8553 |
| **unexplained** | **0.119** |

At 72.4 cents rubberband's residual matched its pitch penalty to **0.0014**. At
1.7 cents it is **0.12 worse than the penalty**. So **rubberband's own processing
degrades the fingerprint, and that cost does not scale down with the stretch** — a
0.1% correction pays nearly what a 4.3% one does.

**A passthrough control does not cover this.** A `tempo=1:pitch=1` arm reading
1.0000 shows rubberband is clean when it **does nothing**; it is not evidence that
it is clean when it **works**.

**A design departure, stated rather than hidden**: these six were not measured
against their masters. They carry **174–190 s of content difference** against a
relation explaining 2.7–3.7 s — which is why they are unrepairable — so a
corrected-candidate-versus-master fidelity **would have been dominated by content
and produced a number that looks like a pitch answer while measuring something
else.** The transfer was simulated on real audio from the same six, where the
ground truth is exact.

### Which is better for the EAR, not the fingerprint

**The question is not which tool sounds better. It is which transformation the
source actually underwent**, because only one of the two is an inverse.

**PAL speed-up is normally a naive speed change**: 24 fps film run at 25 fps, so
the audio plays 4.27% fast **and its pitch rises 72.4 cents** — about three
quarters of a semitone, plainly audible. Industry practice is to leave it that
way; **pitch-correcting on the way in adds its own artefacts**, so most studios do
not.

**If the source was NOT pitch-corrected** — the usual case — then:

- **`asetrate` is the exact inverse.** It restores both duration *and* the
  original pitch, and introduces no artefacts of its own, because it is the same
  operation run backwards. **Best for the ear and best for the fingerprint.**
- **`rubberband` fixes the duration and LEAVES THE PITCH 72.4 CENTS SHARP**,
  permanently, while adding stretch artefacts. **It preserves the defect it was
  asked to remove.**

**If the source WAS pitch-corrected**, the reverse holds: `asetrate` would drag
the pitch 72.4 cents flat, and a time-stretch is the correct repair.

**So the tool follows from a property of the file, and that property is
measurable.** Our own evidence says the naive model holds here: of twelve files
measured against real masters, **16 of 16 tracks read `same_recording` after
`asetrate` correction.** Had they been pitch-corrected at source, that correction
would have made them *worse*.

**Two caveats stated rather than buried.** The fidelity comparison used
`rubberband` at default settings; higher-quality settings might narrow the
0.12 gap, though they cannot fix a pitch that is wrong by construction. And the
1.7 cents NTSC case is **inaudible either way** — there the choice is fidelity
only, and `asetrate` still wins 6 of 6.

**Recommendation**: keep `asetrate` for both, and **add a pitch check rather than
assuming the naive model** — if a candidate's pitch already matches its master
while its duration does not, it was pitch-corrected at source and must be
time-stretched instead. **Nothing currently detects that case.**

---

## 10. WITHDRAWN — measured at a geometry production does not use

**This section claimed that a speed-related pair is refused for the SHAPE of its
delays rather than its score. That is withdrawn by its author.**

The trace used **60 s clips from t=0**, taken from a function that only validates
that fingerprinting works. **Production's real cuts come from elsewhere**: 178 s
clips spaced 89 s from t=120, overlapping. At that geometry the delays are

    -172500  +145250  +166875  +163625  +169375  +171500  -172375  +161375  -170875  -172625

**alternating sign, saturated at the edge of the search span. Not a slope.** 7.6 s
of in-clip drift over 178 s does not degrade the correlation, it **destroys** it.
**These pairs are refused because the correlation genuinely fails — for their
score.**

**The clean line came from clips the author chose.** Their own account: *"the
window was a lookup I skipped; this was an inference I preferred."*

**What survives, and it is stronger.** At the real geometry:

| | uncorrected | `asetrate` |
| :--- | ---: | ---: |
| one | 0.5796 | 0.9337 |
| other | 0.5831 | 0.9119 |

**Both uncorrected means fall BELOW the unrelated-pair null band (0.5954–0.6300)**
— so the merge path still cannot distinguish a speed-related pair from two
recordings that share nothing. **The error had flattered the pipeline; correcting
it makes the case worse for it.**

Also intact: **no wrong delay is applied** (the pair raises); the unguarded
single-delay branch still cannot fire; and the delay-slope anomaly is now measured
at **two** geometries — +4.88% at 60 s, +4.28% at 178 s — robust and unexplained.

**And the population bound weakens with it.** The 95 multiple-delay refusals are
**correlation failures, not slope-shaped refusals**. The **5 confirmed speed
relations stand**, because those came from a three-arm sweep rather than this
reasoning. **The scoped structural fix now solves a problem production has not
been shown to have.**

**How it surfaced is worth recording**: one agent cited the other's window claim in
a committed report without opening the function, while that agent had also not
read it. **A two-agent citation loop with no lookup at either end, standing for
hours because each relied on the other having done it.**

---

## 11. Phase 2 rescan: 30 of 30, zero recovery — and the reason generalises

**Measured, whole population, ~8 minutes of container time**: 29 held at
not-piecewise, 1 flipped, 0 unscanned. **The single flip is not a recovery** — it
has 0 change points and 1 reliable window of 17, so it is piecewise *by default*
because nothing contradicted it, and the corrected scanner found **less** signal
there than the old one.

**The recovery population is measured at ZERO.**

**Why, and this is the transferable part: these files do not fail on cut-point
POSITION. They fail on content that is not there.**

| unaccounted content | tolerance |
| ---: | ---: |
| 124 656 ms | 6 321 ms |
| 126 375 ms | 6 591 ms |
| 57 310 ms | 7 668 ms |
| 7 858 ms | 786 ms |
| 7 087 ms | 711 ms |

**Mismatches of 7 to 126 seconds.** A scan repositions edits; **it cannot conjure
two minutes of missing programme.** The three singletons predicted exactly this
and the series confirmed it.

**"We rescanned and nothing recovered" is now measured across a whole series plus
three independent singletons** — a far stronger statement than *"never tried"*,
for eight minutes of box time.

**A defect found and fixed mid-run**: a classifier's early return omitted two keys
that its caller reads unconditionally, so **any file with no reliable window
crashed the scanner** — one did, thirteen files in, taking fourteen with it. Fixed
with a **key-parity** test rather than a two-name test, so a third return path is
covered. The 15 lost files were re-run and are in the 30.

---

## 12. WITHDRAWN — it measured a quantity the project already documents as differing

**Everything measured was the GLOBAL offset.** `AGENT.MD`'s own "Definition of
success" §2 records that audio-to-video skew across ten release families runs
**0.4–99 ms**, that the two instruments answer **different questions**, and that
**cross-validation belongs at CHANGE POINTS, not at the global offset.**

**So the reported ~1.6-frame audio/video disagreement sits inside the documented
skew.** Expected behaviour, not a finding — and the corpus figure, already reduced
from *20 of 29* to a single observation, measured the same wrong quantity.

**The endorsed comparison was then run**, on stored plateaus with no decoding:

    step agreement over n=15 change points:  median |error| 3.0 ms
    within one 23.976 fps frame:             10 of 15

**3.0 ms at change points against ~65 ms at the global offset.** The skew cancels
exactly as documented and AGENT.MD's own cited figure reproduces.

**And the structural consequence for the half-frame ruling**: half a frame is
**20.85 ms**; the documented skew reaches **99 ms**. **A video-derived offset
cannot validate a global frame snap — the skew exceeds the bar by up to a factor
of five.** Video evidence works **only at change points**, i.e. on `C_cut` files,
**not the drift files the ruling concerns.**

### No independent video evidence is obtainable for the over-bar files

Both routes are closed, measured rather than assumed:

- **6 of the 10 ids are absent from the harness's 111-entry pair list**, and the
  runner **skips an unknown id with a bare `continue` — no message, exit 0.**
  Six dropped and the run reports success; caught only by counting output files
  against ids passed. **Silent-skip class again.**
- **The 4 that did run are PAL pairs** — master 23.976 fps against candidate
  25.000 — where the method cannot work: **worst Hamming 30.5–31.2 of 64 against
  32 for chance.** Window spreads reach **11.9 seconds**, offsets cluster near
  6000 ms **because they are pinned at the ±6 s search ceiling**, and the scene
  vote contradicts itself within a file (+2347 and −2058 ms).
- **0 of the 10 have a change-point table entry**, so the endorsed route is
  unavailable too.

**The frame-snap persistence scope is the only route that changes this, and it is
prospective — it fixes the next run and never these 26.**

## 13. The two audio instruments disagree about which pairs are measurable

**Pearson correlation between chromaprint fidelity and GCC-PHAT peak-to-sidelobe,
over 40 pairs: 0.154.** They are essentially uncorrelated about *whether a pair
can answer at all*.

**Four contradict outright:**

| fidelity | PSR | delay | % of ceiling |
| ---: | ---: | ---: | ---: |
| 0.8889 | 8.4 | −630 ms | 1% |
| 0.8853 | 9.9 | −3402 ms | 3% |
| 0.8212 | 1.5 | −252 ms | 0% |
| 0.8188 | 1.2 | −3402 ms | 3% |

**Chromaprint says "same recording, small sensible delay". GCC-PHAT says "no
peak".** Those fidelities are same-recording territory on the dub calibration, and
the delays are 0–3% of the search ceiling — **so neither is a failed search by the
exact test** (§110).

> **"Can the audio answer this?" has no single answer. It depends which instrument
> you ask.**

**That is harder than a low pass rate and it is not fixed by measuring more
pairs.** A screen cannot be built from one instrument, and building it from two
needs a rule for the contradictions — **which nobody has, and which should not be
invented.**

**What survives exactly, needing no threshold**: **3 of 40 pairs return a delay on
the computed ceiling** — a failed search by construction.

**The pass rate is WITHDRAWN.** The screen's headline 35% answerable rested on a
peak-to-sidelobe bar of 10.0 that **splits the population at its densest point**:
sorted values run continuously 0.8 → 38.1 with a largest gap of 1.2×, and the bar
puts 9.9 and 9.9 on one side, 10.0 and 10.1 on the other. Fidelity is continuous
too, largest gap 0.064 and that in the tail. **The bar was described as "not
fitted" in its own docstring** — the fifth threshold here to be unfitted, and the
second written by the same author the same day.

**Validated on identity first**, which is the lesson from disqualifying an
off-the-shelf aligner: master against itself gives fidelity 1.0000, delay 0.0,
PSR 1.9e10. **And one known negative reads fidelity 0.6163 — inside the null band
and ambiguous — while sitting at 98.8% of the ceiling.** That is the argument for
three gates rather than one: **the ceiling gate catches what the fidelity gate
cannot.**

---

## 14. A delay that steps mid-file has no single value to snap to a frame

**ONE file's audio delay is not one number — and the count was corrected downward
from three.** Two other files reported as stepped are **not**: their delays read
as **scatter at the PSR floor** (+1487, +862, −5016, −1683, +71, −5019 …), which
is audio that **stops correlating partway through**, not a second regime. **A
spread says a pair answers more than once; it does not say the answers are each
stable.** Only the file below has genuine plateaus.

**Measured per window:**

| window | delay | windows | spread |
| :--- | ---: | ---: | ---: |
| t ≤ 780 s | −32.011 ms | 6 | 0.003 ms |
| t 810–930 s | −198.864 ms | 4 | 0.004 ms |
| t 1290–1350 s | −209.500 ms | 2 | — |

**Each regime is internally stable to a ten-thousandth of a frame. These are
STEPS, not drift.**

    step 1   166.8524 ms  =  4.0005 frames of 41.708 ms
    step 2    10.6363 ms  =  0.2550 frames

**The first step is four frames to within half a thousandth of one** — that is an
**edit**, not a sync error. **This is now a single file, not three**, and the
prediction that ≥40 of 79 would show steps was built partly on the two examples
since withdrawn. **It stands as committed and the 79 will settle it, but its
support is thinner than when it was written.** The second is not a frame multiple and is harder to
account for.

**This is a third failure mode**, beside *no answer* and *a biased answer*: **an
answer that is correct in each half and different between them.** **Nothing in the
pipeline asks whether a delay is constant across the file before applying it** —
and a file like this has no single delay to snap to a frame, which is the premise
the mission rests on.

**It also reframes the largest refusal family.** `first_delay_test` refuses a pair
whose cuts disagree, and **a genuine step discontinuity is exactly that** — so
some of those refusals may be the test working correctly on files that cannot be
merged with one delay, rather than an instrument limitation.

### The screen promoted the worst possible pair

`audio_answerability` scores each pair on its **best** window. This file scored
**`answerable` at PSR 38.1 — the strongest in the run — and that 38.1 came from
the second regime.**

> **Best-window scoring answers "can this correlate somewhere". It cannot answer
> "does this give one answer", and those are different questions.**

**For this failure the screen is worse than no screen**, because it hands the
reader a confident verdict on precisely the pair where a single audio delay is
meaningless. The fix is free — **score the SPREAD across windows rather than the
best of them**; the windows are already measured and already discarded.

### Peak-to-sidelobe is not a confidence measure, from three sides

    it does not track the INPUT transformation   matched pair, psr 1.5 vs 38.1
    it does not track the OUTPUT correctness     five episodes, one series
    it varies 13.8x WITHIN one stable regime     delay stable to 0.003 ms

**So PSR cannot be used as confidence for either thing anyone has used it for** —
a constraint on the screen, on any stratification built from it, and on every
future reader of that return value.

---

## 15. The largest failure family is NOT speed — and 22 files have a constant delay production said it could not find

**Registered before measuring**: fewer than 10 of the 79 would show a clean slope
near a known constant. **Measured: zero.**

| | |
| :--- | ---: |
| no coherent slope | 48 |
| flat — no speed relation | 31 |
| **clean slope near a constant** | **0** |
| clean slope, no constant | 0 |
| saturated | 0 |

All 79 measurable, nothing lost to missing data.

**The bound is closed.** It stood at **5 confirmed of 195 failures, ceiling 16**,
with the ceiling explicitly unknown above that because a ratio screen is blind to
a relation the ratio no longer shows. **The 79 a ratio screen could not classify
contain no masked relation at all — so 16 is the ceiling, supported rather than
bounded.**

### The 22 that matter

**31 files run at identical rates to a part in a million while differing in length
by up to 9%** — median |factor − 1| of **0.00009%**, median residual **0.62 ms**,
median duration ratio **1.0559**. That length difference is **entirely content**.

**22 of them appeared to fit a clean CONSTANT delay within 5 ms while production
refused them for finding MULTIPLE delays. THAT IS WITHDRAWN — production is
right.**

Production's per-cut delays were **regenerated at production's own geometry**,
with saturated cuts removed by the per-file ceiling, and compared against the step
the slope instrument had fitted on the same file:

    clean spread   fitted step    ratio
      1000 ms       -1001.4 ms    0.999
      5000 ms       +5007.7 ms    0.998
      6000 ms       -6001.8 ms    1.000
      ...
    median ratio 1.006     21 of 22 within 25%     11 within 2%

**Production's spread in per-cut delays IS the step.** Two instruments, different
window schemes, one number, 22 files — and the residual scatter is quantisation:
every clean spread is a multiple of **125 ms**, the fingerprint frame.

**So both halves of the original claim were wrong.** The delay is **not constant**
— two regimes separated by up to 6 seconds. And Step 1 **did** find it: it found
both regimes, reported that they disagree, and refused. `first_delay_test` accepts
one delay or two within 127 ms; **a 6000 ms step is neither. There is no defect in
these 22.**

**The cause was one discarded field.** The "constant delay within 5 ms" was a
constant delay **plus a step the instrument had already modelled and removed** —
`factor` and `residual_rms_ms` were recorded from a report that also carries
`content_step_ms`, and the small residual was read as a single regime. **One
omission produced two wrong conclusions, the second with a defect report
attached.**

**The saturation check was load-bearing, not ceremonial**: 12 of the 22 had at
least one cut pinned at the search ceiling and raw spreads ran to **420 750 ms**.
Without the per-file ceiling filter the agreement would have been buried under
ratios of 300–1000, and the report would have read *"production disagrees
wildly"* — **the second time today the ceiling discipline changed an answer rather
than confirming one.**

**And the step phenomenon is not rare: 66 of 79 files carry a fitted step**, with
one file agreeing to **0.1 ms across two instruments.** The ≥40 prediction is
**supported by this data, not contradicted.**

**The other 48 are genuinely incoherent** — residual median **2919 ms**, max
**86 seconds**. Not a lag with error: **content that does not correspond.**

**The saturation check fired on nothing — 0 of 79 — and that is worth
distinguishing from not having run it** (§110). It is the check whose absence cost
an hour earlier the same day.

### A tolerance wider than the gap between two hypotheses

The "near a known constant" window was **0.15%**. **NTSC pulldown sits 0.100% from
unity.** So a factor of **exactly 1.000000 — no relation at all — matched as
"clean slope near NTSC pulldown"**, and four files were labelled that way before
the run was stopped.

**Left running it would have reported most of 79 files as NTSC relations, blown
the registered prediction in the wrong direction, and printed the factor to six
decimals while doing it.** The fix is not a smaller number: **unity is tested
first and is its own answer.**

---

## 16. SUPERSEDED -- the steps are frame-quantised in about half of files, not as a law

**The heading below claimed a law. On clean data it is a tendency, and the
difference decides whether a whole class of repair is safe.**

Measured 2026-09-01 on the language-matched re-run (95 files, **zero
cross-language pairs**, jpn 84 / eng 11 -- the population the earlier claim
lacked):

    STEPPED 41    scatter 54          stepping is 43%, not a rare pathology
                  ^^^^^^^^^^ NOT a finding -- see the withdrawal below
    steps within 0.01 of an integer frame     40 of 63   (63%)
    files where EVERY step is integer         20 of 41   (49%)
    the 23 non-integer steps: median 0.200 frames, max 0.411

**The non-integer steps are not noise around an integer.** A median offset of
0.200 frames and a maximum of 0.411 is a real sub-frame quantity; measurement
slop at this scale is an order of magnitude smaller.

**Consequence for repair, and this is the reason the correction matters:**
**any repair that rounds a step to the nearest frame boundary will silently
corrupt the roughly half of files whose steps are not integer** -- and it will
do so without raising anything, because rounding always succeeds.

**Distribution by class**: C_cut **39 stepped / 42 scatter**.

### WITHDRAWN: "B_fps is 6 of 6 scatter, and a rate problem should drift"

**That was the classifier, not the files, and it is withdrawn.** `classify()`
held four verdicts and **no way to say "this drifts"**; `scatter` was its
terminal `return`, so it meant **none of the above** and was reported as though
it meant something about the files.

    PAL drift                     42.71 ms per second
    over the 45 s stride         1922 ms per window
    tolerance, one frame            41.708 ms
    -> every window sits 46 frames past tolerance

**Every window becomes its own group, nothing reaches MIN_PLATEAU, verdict
`scatter`. A plateau-finder CANNOT report a drift as anything else.** Confirmed
on a synthetic ramp at exactly that rate before the withdrawal was made.

**The six were err_46, 51, 59, 63, 70 and 77 -- all in the declared-25.0
genuine-PAL group**, so this was the branch in which the puzzle was real. **It
dissolved anyway, for a reason neither branch allowed for.**

**What survives**: **41 STEPPED of 95 stands** -- two or more plateaus over 70%
of readings is not something a fallthrough can manufacture -- and **the
integer-frame result stands**, being computed from those step sizes. **The 54
`scatter` does not stand as a finding**: it now contains genuine noise, fast
drift and slow wander together, and cannot be subdivided from what was stored,
because the sweep recorded plateau means rather than per-window readings.

**The disagreement was available before either claim was published.** A slope
instrument measured **monotonic drift** on four of these exact files while the
plateau classifier said **scatter** on six -- **same pairs, same afternoon.
25.0/23.976 = 1.042709 against a measured 1.0427087-1.0427106: the two
instruments agreed on the number and disagreed on the word, and neither owner
treated that as information.**

**Provenance of the population**: 95 rather than 89 because the exception-cap
fix surfaced **six records whose `Multiple delay found` text had been truncated
away**. The defect had been removing evidence from the corpus, not only from
the reports.

**A superseded prediction, recorded as unsettled.** Agent 4 predicted >=40
stepped and measured 41 -- but predicted it **of 79**, and this is 95. **A
prediction is not confirmed on a denominator it did not name.** The 43% is the
finding; the prediction stays open.

**Everything derived from the 89-file run remains WITHDRAWN** (82%
cross-language), including the earlier 74%, which was a biased subset.

---

### Original claim, superseded above

Scanned five multiple-delay refusals at 45 s windows, classified by **regime
structure** rather than by fit residual:

    two files STEPPED (2 and 4 regimes)      three files SCATTER

**Every step is an integer frame count:**

| step | ms | frames |
| :--- | ---: | ---: |
| A→B | +1001.31 | **24.0075** |
| A→B | −1000.95 | **−23.9988** |
| B→C | −1001.20 | **−24.0048** |
| C→D | +959.30 | **23.0002** |
| earlier file | +166.85 | **4.0005** |

**Five steps, three files, integer to eight parts in ten thousand.**

**BOTH THE GENERALISATION AND ITS CORRECTION ARE UNSETTLED — the population run
they rest on was 82% CROSS-LANGUAGE.** The classifier compared `a:0` against
`a:0` without checking languages, so on one pair it correlated a German master
against a Japanese candidate. §8 says cross-language correlation is untrustworthy,
**so the whole classification rested on comparisons this project already
distrusts and reported them as verdicts about the files.**

    a:0 language mismatch, master vs candidate
      STEPPED   27 of 37   (73%)
      scatter   46 of 52   (88%)
      overall   73 of 89   (82%)

**Void, and not replaced by a new number:**

- **37 of 89 stepped** — withdrawn.
- **The ≥40 prediction was refuted on that figure, so the refutation is void and
  the prediction returns to UNSETTLED, not lost.**
- **The correction relayed as "frame-quantised in about two in five, 22 of 56"
  was computed on the same contaminated set.**

**On the 16 pairs whose languages did match: 17 of 23 steps within 0.01 of an
integer frame — 74%, against the 39% reported.** So the correction may have been
wrong **in the direction of under-claiming**. **74% is not asserted either**:
those 16 are the pairs whose two sides happen to lead with the same language,
which is not a random sample. **Hold both the original finding and its correction
as unsettled rather than swapping one for the other.**

**The fix costs nothing**: every one of the 89 pairs shares at least one audio
language, so requiring a shared language loses no file. The scan now picks one on
both sides and **records which**; pairs with none get their own verdict instead of
being measured anyway. **Contaminated entries are dropped on startup** — *a
verdict from a distrusted comparison is not a weaker verdict, it is a different
measurement wearing this one's name.*

**The six hand measurements stand throughout** — each is a difference of two
plateau means on a language-matched pair.

**And the operational claim survives the withdrawal**, because it does not depend
on the step/scatter split: **none of the 89 is a clean single regime**, so the
delay test refuses all of them correctly.

**And the 23-frame step is what settles it.** 1001 ms is *both* 24 frames **and**
about one second, so every 24 is ambiguous between a frame-quantised edit and a
whole-second one. **959.30 ms is 23.0002 frames and is not a round number of
seconds.** The quantum is the frame — **and the discriminating datum is the single
step that broke the tidy pattern**, the second time today the untidy point carried
the answer (§110).

**The operational consequence holds whatever caused them: these files cannot be
merged with ONE delay.** A single delay is wrong on one side of the step by 4, 23
or 24 frames. **So part of the multiple-delay refusal family is the test working
correctly** on material that genuinely has no single answer.

**NOT established: that an edit caused them.** Frame-quantised discontinuities are
what content trimmed at frame boundaries produces — **consistent, not
demonstrated.** Four mechanisms died today; a fifth was not named.

**Three of the five are SCATTER, not steps** — a stable region then disorder, the
same shape as two earlier step claims that had to be withdrawn. **So the
phenomenon is real and narrower than predicted**: 2 of 5 projects to ~36 of 89,
**below the ≥40 committed in advance, and the prediction was left unrevised.**

**And the 22 turn out to be the same phenomenon, not a counterexample.** Their
apparent flatness was a residual measured *after* a step had been modelled and
removed; regenerating production's own per-cut delays shows **the spread IS the
step**, 22 for 22.

### The truncated record is unfindable, not merely short

**149 of 196 stored exceptions are exactly 220 characters, 104 with unbalanced
brackets** — five tuples of one couple out of ten cuts. **And the reason is cut
too**: this population could not be found by grepping for the function that raises
it, only by printing raw text and matching the message. **A truncated record that
loses its own classification is not a short record, it is an unfindable one.**

---

## 17. Three entry points, and only one of them repairs

**The Lead's hypothesis — that stepped files are blocked by a routing gate keyed
on something they fail — is REFUTED. The gate would have passed 21 of the 22.**

    of the 22 stepped files
       no cut-point record    0
       have a record         22
          flagged piecewise   21
          not piecewise        1

**And the tables predate the failures** — written 2026-08-29 to 08-31 07:29Z,
failures recorded 08-31 10:41Z to 09-01 01:29Z. **The records existed and the gate
was not the obstacle.**

**What the read did find is simpler and larger:**

    fusion.py:1234              merge_videos_with_repair(...)   <- repairs first
    main_gestionar_show.py:98   merge_videos(...)               <- NO repair
    main.py:72                  merge_videos(...)               <- NO repair

**Two of the three entry points have no repair at all.** A stepped candidate
arriving on either goes **straight to `first_delay_test`**, which correctly
refuses it.

**Which entry point these 22 took cannot be recovered.** `batch_state` records
`stage: "ran"` and **does not name the entry point**; three builds are
represented. That is §101 again — **a record that cannot say which code path
produced it** — and it is one field at the point of decision.

**The one apparent stale record was WITHDRAWN — the record was right and measured
the step better than the agent disputing it:**

    stored record   delta 5004.2 ms at 642.0 s
    45 s scan       step -5004.59 ms
    joint 12-window fit  +5007.7 ms      <- the outlier of the three

**0.4 ms between the record and an independent scan.** *"Wrong"* had been inferred
from *"disagrees with my classification"*, which is not evidence.

**The real reason it declines is CONSERVATION, and it is a better objection:**

> the lag changes by −4994 ms across the file, but the master is only −995 ms
> longer — **−3999 ms of content unaccounted for** (tolerance 500 ms)

**A lag stepping by five seconds should be matched by about five seconds of
duration difference. The durations differ by one.** The step is real and well
measured; **it does not account for where the content went.** That is the same
check that accounted for all 50 refusals in an earlier audit, and it is the same
shape as Phase 2's 7–126 seconds of unaccounted content (§11).

**And the piecewise flag is not a change-point count**: three other files carry
two change points and one carries three, **all four flagged piecewise.** The flag
is *no reasons*, and here the reason is conservation. **So there is no stale-data
defect among the 22 at all.**

**A methodological note worth carrying: step MAGNITUDES are comparable across
instruments; LAGS are not.** One record reads −1002.1 → +4002.1 where a scan reads
+133.07 → −4871.52 — different references, different audio tracks, **absolute
values sharing nothing while the deltas agree to 0.4 ms.**

### A 6-second step IS expressible

`plan_segments` gives every plateau its **own** `lag` and computes
`c_start = m_start − lag`. **There is no magnitude cap on the lag anywhere in the
planner** — a 6000 ms lag is arithmetically identical to a 60 ms one, changing
only where in the candidate a piece is read from. The only bounds are a 10 ms
minimum piece and the candidate's duration.

**What it cannot do is invent content.** If the step means the candidate is
*missing* six seconds rather than displaced by them, the planner emits a **gap**
piece. **So the discontinuity is expressible; whether the output is right depends
on which of those two it is, and that is not a property of the planner.**

---

## 18. Almost every verdict in the ledger is partial, and the damage only made it visible

**Of every output carrying a `MERGED_*` verdict, exactly ONE has a verdict
covering every language present in it — and that file contains one language.**

    verdict covers every language present    1   (a single-language file)
    partial                                 21
    no per-language record at all           85

**Undamaged files sit at 1-of-8 and 3-of-8 languages judged.** So the three
outputs found stamped `MERGED_IN_SYNC` with broken audio were not a special
class: **the damage never made these comparisons partial — it made the
partiality visible.**

### But narrow is not wrong, and the narrowness is the MATERIAL

**All 35 thin-verdict files were re-verified with the widened comparison. ZERO
verdicts changed.** Every one remains `MERGED_IN_SYNC`.

**That corrects the natural reading — including the Lead's.** Finding three
damaged `MERGED_IN_SYNC` outputs invited the inference that thin verdicts are
unreliable **as verdicts**. They are not. **They are narrow, which is a different
fault** — and the three damaged files were caught by a **per-track audit**, not by
a wider correlation.

**And widening is mostly not available.** Across the 35 files there are **218
language-slots**:

     85  no_master_reference     the master lacks that language
     71  no_candidate_track      the candidate lacks it
     43  indeterminate
     19  answerable              measured

**156 of 218 — 72% — are unmeasurable because one side does not carry the
language at all.** So *"widen the comparison"* is not a remedy that exists here:
**the verifier already measured everything the material allows.**

**AND "19 answerable" WAS NOT THE MEASURED COUNT — the measured count is 50.**
`answerable` and `indeterminate` are statements about the **counts**, not about
whether a measurement happened: *indeterminate* means the output holds as many
tracks of a language as the master, so the counts decide nothing — **and it is
then resolved downstream by correlation.** 32 of the 43 were compared and passed.

    218  language-slots
    156  no counterpart on one side   (72%) -- excused by the ruling
     62  comparable
     50  MEASURED                     (81% of the comparable set)
     12  comparable but unresolved

**"81% of what can be compared" and "19 of 218" describe the same data and support
opposite conclusions.** The second was reported.

**The other 11 are legitimately declined**: 10 where the candidate contributed no
track, 1 by the copy guard. **So none of the 43 represents lost coverage, and
under the owner's ruling these files are verified to the extent their material
allows.**

**One thing does not close.** `sync_verdict` returns *"candidate contributed
none"* whenever the output holds no more tracks of a language than the master —
**a claim the count cannot support, and the same module's own `indeterminate()`
docstring says so**: *"The output's track may be the candidate's … Counts cannot
distinguish them."* **The message asserts as fact the exact thing its neighbour
documents as undecidable.**

Measured on two of the ten, the conclusion is **correct** — every output track
corresponds to a master track, two of them bit-identical at peak 1.0000, and the
candidate's own two tracks were dropped. **Correct by assumption, verified on 2 of
10** — the other eight share a folder and a structure, **which is a reason to
expect the same answer, not evidence of it.**

**And a third sighting of ~131 ms**: two masters here each carry two `jpn` tracks
**130.6 ms apart**, after the copy-guard docstring recorded it for one file and
another master reproduced it. **Three files, three episodes — a property of the
material.**

**Three verdicts rest on a single window, and they are one folder, not three
cases** — each file holds one language, so one language on one window is the
entire verdict, with peaks of 0.999 at ~10 ms lag: **just under the copy guard,
which needs exactly-zero lag AND ≥0.999 in every window.** The detector was
validated against a known single-window case first — *"a broken detector also
returns three, and it would have returned zero"*.

**And the 35 are 11 folders, three of which hold 25 of them.**

**The 21 files with no per-language record carry no `language_states` at all** —
they **predate** the every-language verifier rather than having run it and found
nothing. **Never asked, not asked-and-refused.**

### The three false stamps, answered on the artefacts

**One of them is the whole defect in a single row.** Its `eng` track ends at
441.5 s, so the **480 s and 900 s windows fall past its end — only the 60 s
window survives, and the verdict passes on that one window.** Its `jpn` reads
−0.1, 0.1 and −1.3 ms at peaks 0.999, 1.0, 0.99: **two of three windows carry the
master-against-itself signature, and the guard requires all three** (§105). **Two
known defects in one verdict, neither constructed.**

The empty-track file is now explicitly `unverifiable` — **the verdict did not
flip; the claim became narrower and honest.** The third's damaged language was
never attempted at all.

### Two comparison paths, and the common one is weaker

`sync_verdict` branches on whether the output holds 2+ tracks of a language.
**The ≥2 case is the ordinary result of a merge, and it is the weaker branch**: a
bare `worst <= 50` — **the exact max rule that `judge_profile`'s own docstring
says it was written to replace after it produced two false condemnations** — with
**no dissent reporting, no provenance, and no copy guard at all.** All four of
those exist only on the fallback path.

**So "the copy guard is all-or-nothing" understates it: on the common path there
is no copy guard.** Found by *running* it, not reading it — two files returned
with provenance absent and one populated, and only two code paths explain that
split.

### A drift finding raised and killed by its own selection

A possible drift defect measured across 156 series showed a ~2σ excess. Split by
whether a file had been *selected* for re-verification:

    unselected (ledger)     n=103   excess -2.3   -0.9 sigma
    selected (re-verified)  n= 53   excess +8.3   +2.9 sigma

**The unselected population is slightly BELOW chance. All the signal is in files
chosen because they were problematic** — the pooled figure was the average of a
signal and its absence. **Withdrawn.**

**Two method notes**: the naive baseline is 33%, the correct **tie-aware** one is
**13%**, because 95 of 156 series contain a repeated value and can never be
monotonic — **using 33% would have inverted the conclusion.** And the analyst's
own expected value moved 20.3 → 23.0 mid-analysis because a cherry-pick changed
the input file underneath them, **recorded as a correction rather than restated
as a refinement.**

---

## 19. The merge measured a 180 ms divergence, labelled it, and applied nothing

**From the merge's own log:**

> *"Per-language offsets (RECORDED ONLY — the merge applies one delay to every
> track): Per-language divergence: 10 languages disagree by 183.6 ms (4.40
> frames); the merge applies one delay to all of them"*

**It measured it, named it, and the field has no consumer.** Then a recovery path
returned `[0.0, 0.0, 0.0]` — *"spread 0.0 ms against a 40.0 ms tolerance →
uniform"* — on a file whose candidate sits **+180 ms from its master**, measured
independently at peak 0.98. **A confident zero from a different comparison
overrode a divergence the same log had recorded three lines earlier.**

### The mechanism, also from the log

> *"Recovery reference resolved: master has 2 'ja' tracks, track 9 chosen by
> default_flag, offset 0.0 ms"*

**The master carries two Japanese tracks and the recovery picked its reference by
a metadata FLAG, not by measurement.** Against that track the offset genuinely
*is* 0.0. **The recovery was not wrong about `ja` — it was wrong to generalise**,
because the merge applies one delay to every track and the other ten languages are
180 ms out.

**The corroborating number was already in our own source**: the copy-guard
docstring records *"one master carried two jpn tracks 131 ms apart"*, and this
file's two audio couples disagree by exactly that — **−66 ms and +65 ms, each with
variance 0.000000.** Internally certain, mutually contradictory. **The master's own
duplicate tracks are the ambiguity, and the recovery resolves it by flag.**

**This is the day's dominant defect class in the MERGE path rather than the
reporting path** — the first instance that changed what was *produced* rather than
how it was described: **the recovery cannot distinguish "no delay" from "I
compared the wrong pair", and both return 0.0, which passes a 40 ms tolerance.**

### Sizing, with the limit stated

    61  merge logs exist (against 315 corpus files)
    44  carry the RECORDED ONLY header
     5  carry a MEASURED divergence      -- 180.1 / 180.2 / 180.1 / 183.6 / 184.3 ms
     5  recovery returned a uniform zero
     5  intersection                     -- one folder, five consecutive ids

**The 44 are not suppressed divergences**: what follows the header reads *"not
measured, present on one side only"* — **files where the languages were never
compared at all. A count of the header is not a count of the defect.** And one
file matching a text search for the same phrase carries a *different* message —
**a search stopping at the phrase would have reported six.**

**The precondition is common.** Every resolved master probed, metadata only, 245
distinct, zero failures: **59 (24%) carry at least one language on 2+ tracks** —
spa 35, jpn 27, fre 22, eng 18 — and **27 share this file's exact shape.**

**The discriminating condition is now MEASURED, and it is met almost always.**
185 pairs correlated across the 59 masters:

    genuinely offset      159        median 37.4 ms, 75th 83.1, max 209.8
    aligned (< 1 ms)        7        134 of 159 exceed half a frame
    rejected as unreliable 19

**Seven of 166 usable pairs are actually aligned.** So `default_flag` is choosing
between **materially different references in nearly every case where it has a
choice**, and the upper bound of 59 is a measured condition rather than a
possibility.

**The distribution is lumpy but continuous — there is no gap to cut at**, mass at
25–49 ms just above half a frame with a tail to 210. **No threshold proposed.**

**The instrument was proved before its zeros were trusted** — self-pair 0.0 at
peak 1.0000 against a genuine pair at 130.9 ms, same file — **and it paid for
itself immediately.** The largest raw offset was **218 ms at a correlation peak of
0.227**, below production's own 0.3 usable floor, and its **confirming window put
the same pair at 0.01 ms at peak 0.98.** It would have been the headline. **14
more pairs died the same way on confirm disagreement** — the n=1 degeneracy again,
here manufacturing a finding rather than losing one.

**And the 131 ms in the copy-guard docstring is a PATTERN, not one file**:
measured 130.9 / 130.4 / 130.8 at peaks 0.98. A third track in the same master
sits at **180 ms with peaks 0.62–0.70** — a different *recording* as well as a
different offset.

**The grouping changes the headline.** 159 offset pairs reads as corpus-wide;
honestly it is **42 files across 8 folders, with three folders holding 145 of the
159 pairs:**

    folder A   108 pairs / 10 files   8 languages   11-210 ms   <- larger, unexamined
    folder B    26 pairs / 15 files                  1-184 ms   <- the confirmed one
    folder C    11 pairs /  7 files                  7-191 ms

**The largest group has no merge log in the tree**, so the log scan that found the
confirmed case could not have seen it.

**Harm is still unextrapolated**: the third condition — *whether other languages
then needed a different delay* — remains unmeasured.

**AND THE LARGEST UNEXAMINED GROUP DID NOT CONFIRM THE MECHANISM.** Checked in
order: *"no merge log"* turned out to mean **no OUTPUT**, not never-ran — 10 of 11
read `stage=ran, outcome=FAILED` with no product. **So on the merge side that
folder is exposure without harm**; nothing shipped wrong from it. (Ten files that
cannot merge at all is a different cost, already counted elsewhere.)

**All ten failed with the same exception as the confirmed folder** — and **every
one is exactly 220 characters, with the second audio couple cut off on all ten.**
With no merge log there is no second copy: **the truncation removed the diagnostic
content from precisely the files being diagnosed, and it is gone rather than
hidden.**

**But the offset is ~9 SECONDS, not 180 ms**: −9133.6 and −8900.4 ms at peaks
0.76–0.91. **That is a different edit, not a sync offset, and the delay test
refused correctly.** So the folder shares the **precondition and the exception,
not the defect.**

**The shape recurred and the mechanism did not — and only the measurement
separates those.** The confirmed defect stands at **one group.**

**Two things the negative result still yields.** The per-language divergence is
present there too and **dwarfed** — `jpn` −9133.6 against `eng` −8900.4, **233 ms
apart** — so *"one delay for every track"* is wrong there as well, **invisible
next to nine seconds.** That is exactly how a file carries this condition and is
never noticed for it. And one file's windows disagree while its peaks fall with
them — −9135 → −3130 as the peak drops 0.821 → 0.558 — **either the offset is not
constant or the correlation loses lock, and two windows cannot separate those.**
Flagged, not guessed at.

---

## Current state — 315 error files

| Status | Count |
| :--- | ---: |
| Failed | 182 |
| Merged, unverifiable | 55 |
| Resolved | 44 |
| In sync, over the 15 ms bar | 27 |
| MERGED_OUT_OF_SYNC | 5 |
| No output | 2 |

The corpus is **315**, not 309; the 309 came from a stale mapping, corrected
2026-09-01. The full mission end condition is **not reachable** within the
current token budget. Priority in force: the **55 merged-unverifiable**, on the
grounds that they already merged and only Step 2 verification failed.

## 20. B_fps is not one class, and a CORRECT frame-rate field is the wrong repair input

**The class was being reasoned about as a single phenomenon. It is four, and
only 21 of 35 files are frame-rate relations at all.**

Tested by asking whether the **declared** rate predicts the observed duration
difference (metadata only, no decode):

    declared          n    obs/pred            reading
    23.976 -> 25.0   13    0.99 - 1.11         EXACT, 11 of 13 at 1.00-1.02
    23.976 -> 30.0    8    0.20 consistently   off by 5x, but very consistent
    23.976 -> 24.0    8    -47 to -69          wrong sign AND 50x too large
    23.976 -> 29.97   5    -0.29 to 1.84       no agreement
    25.0 -> 23.976    1    0.00                durations identical

### The 8 declared `30.0`: a PAL relation carried by a real 30 fps encode

**An earlier revision of this section said these eight were mislabelled -- that
the declared `fps_err=30.0` was wrong and the files were really 25.0. That is
WITHDRAWN. The declared rate is correct.**

    err_288 289 290 293 299 300 302 688
    r_frame_rate = 30/1    avg_frame_rate = 30/1    on ALL EIGHT

The withdrawn claim came from **solving for the rate that would explain the
observed duration delta, getting 25.0, and concluding the label was wrong** --
**a frame rate inferred from durations, with the stream never probed.** One
`ffprobe` settles it, and it was not run until a different question forced it.

**What is actually there is two independent facts, both true:**

    PAL, master-relative            1 - 1/1.042708 = 4.096%
    observed for the group                  4.10% - 4.17%
    a real 23.976 -> 30 speed change              20.08%

**The video is genuinely encoded at 30 fps, and the content carries a 4.1% PAL
speed relation.** The 30 fps conversion is **speed-neutral** -- it changed the
frame count, not the duration -- and the PAL relation is what moved the
timeline. **The audio says PAL, the video says 30 fps, and both are correct
about different things.**

**Corroborated independently by audio**: slopes of **1.0427087 to 1.0427106 on
err_289, 290, 299 and 302, monotonic 4 of 4**, within 0.0002% of the
25/23.976 constant.

**err_688 is not an anomaly and needs no separate explanation.** It is 30 fps
like the rest with a delta of 0.7%: it carries the re-encode and **no** PAL
relation. It looked odd only against a model in which 30 fps implied a speed
change.

### The repair consequence survives, on better grounds

**`fps_err` is accurate. It describes the ENCODE, not the relation.** Anything
keying a speed correction on 23.976 -> 30.0 applies **20.08% where the true
relation is 4.10%**.

**A correct field can still be the wrong input, and that is worse than a wrong
label, because nothing about the metadata looks suspect.** A wrong value can be
caught by validating the field; a right value used for the wrong question
cannot.

### The 8 declared `24.0`: content DOMINATES, but a real relation is underneath

**An earlier revision called these eight "a content difference, not a rate".
That is WITHDRAWN: two of them carry a clean, confirmed pulldown relation.**

    err_29   factor 1.0010009   resid 0.32 ms   10 windows   0 at the band edge
    err_31   factor 1.0010001   resid 0.96 ms   11 windows   0 at the band edge
    declared 23.976 -> 24.0 predicts        1.001001

**0.00009% and 0.00001% off -- agreeing with the declared metadata to five
decimals.**

**How both statements are true at once.** The relation predicts **3.2 s** of
duration delta on err_29. Observed delta: **188.2 s. Content exceeds the
relation 58x and masks it entirely.**

**The withdrawn test measured the duration delta -- a content-dominated
quantity -- and read "the delta is not rate-shaped" as "there is no rate".**
The cv comparison was **a clean test of the wrong quantity**: it correctly
showed the delta is not proportional to duration, which was never the question.
**A ratio screen is blind to a relation whose ratio has been masked by
content**, and at 174-205 s against 3.2 s this group is the corpus's clearest
instance of exactly that.

**And the screen was not a near-miss that a tighter bar would have caught.** On
err_26 the relation predicts **+3.8 s** (candidate faster, therefore shorter)
against an observed **-175.6 s** (candidate longer): **the sign is opposite and
the magnitude 46x. The content difference reverses the sign of the quantity the
screen reads**, so **no tolerance on a duration-based test could have found
these relations.**

**Only 2 of 8 are measured; the group is NOT restated wholesale.** The other six
need one slope each, which is decode. **Assuming the group is homogeneous is
invariant 122 -- the rule that came out of this very class.**

    genuine rate relations                21   unchanged
    declared 24.0: relation CONFIRMED      2   err_29, err_31
    declared 24.0: unmeasured              6
    unstructured                           5
    mis-classified by metadata             1

**The ~180 s excess is still real** and still the same shape as Phase 2's zero
recovery (FINDINGS 11) -- unaccounted content. **What changed is that content
and a rate relation are both present, not that one replaced the other.**

### Why the class scatters instead of drifting

    genuine rate relations   21   (13 declared 25.0 + 8 mislabelled 30.0)
    content differences       8
    unstructured              5   (cv 1.3, no model fits)
    no relation at all        1   (err_110, identical durations)

**Monotonic drift is what the 21 should show and what the other 14 cannot.**
The earlier "B_fps is 6 of 6 scatter" was a sample of six from this population;
**if those six came from the 24.0 or 29.97 groups, scatter is the correct result
and there is no puzzle.** Which six is not yet known.

### WITHDRAWN caution, and the narrower true statement that replaces it

An earlier revision of this section warned that `dur_delta` "disagrees in sign
with freshly probed durations". **That was wrong and is withdrawn.**
`dur_delta` is **`|master - cand|`, unsigned**; the author had compared a signed
difference against an absolute one. **The field is sound and may be compared
against a fresh probe, taking the absolute value.**

    err_26   stored 175.616   master-cand -175.6   |diff| matches
    err_46   stored  53.771   master-cand  +53.8   |diff| matches

**Staleness is rare, not systemic** -- the opposite of what the withdrawn
caution implied. Final over the corpus: **307 agree, 1 disagree, 6 with no
stored value, 1 unresolvable -- sound on 307 of 308.** The
recorded `master` still matches what `resolve_pairs` returns on **314 of 315**.

**The one genuine error is err_110, and it is wrong on two fields:**

    stored dur_delta   63.926     actual  1472.012 vs 1472.013  ->  0.001 s
    stored fps_mst     25.0       actual  BOTH sides 24000/1001 = 23.976

Not a sign, not a rewrite, not a master change -- the candidate's mtime is
untouched. **The record is simply wrong about that file.**

**This changes what err_110 is.** It is listed above as "no relation at all,
durations identical", which is correct -- **and the reason it appeared in
`B_fps` at all is that its stored `fps_mst` of 25.0 is wrong. Both sides are
23.976. It is not a frame-rate file and never was.** The class breakdown is
therefore:

    genuine rate relations   21
    content differences       8
    unstructured              5
    mis-classified into the class by a wrong metadata field   1

**The eight are "a PAL relation carried by a 30 fps encode", not "mislabelled
PAL". The heterogeneity result and invariant 122 stand; only the account of
this one subgroup was wrong.**

## 21. The keyframe route corroborates the audio at change points -- on the half of cuts where both can speak

**A second instrument, sharing no mechanism with the audio route, agrees with it
to better than half a frame on about half the cuts it can reach.** One reads
audio envelopes; the other reads container keyframe timestamps.

**The honest chain, every narrowing named:**

    313 pairs -> 88 askable -> 64 with change points -> 113 cuts
        -> 61 where both give a plausible step -> 29 inside half a frame

**Sign was measured, not derived**: 22 same-sign against 80 opposite of 113, so
the keyframe step is **inverted** relative to the table's convention. Expected
from the definitions (`before - after` against `after - before`) but
**established from the population**. All figures apply the measured inversion.

### Where both instruments give a plausible step

Restricting to cuts where **both** report under 10 s -- a step larger than that
inside a 120 s half-window is a content edit, not a delay:

    n = 61      median |difference|  33.3 ms
    within half a frame (20.85 ms)   29
    within one frame  (41.71 ms)     35

    err_218  cut  747.5s   kf +1000.0   audio +1000.1   diff 0.1 ms
    err_60   cut  534.8s   kf +2000.0   audio +2000.2   diff 0.2 ms
    err_181  cut  241.0s   kf +1000.0   audio +1000.3   diff 0.3 ms
    err_96   cut  800.0s   kf +2040.0   audio +2040.7   diff 0.7 ms
    err_64   cut  621.1s   kf +2680.0   audio +2679.1   diff 0.9 ms
    err_185  cut  311.4s   kf +1000.0   audio +1001.0   diff 1.0 ms

**Sub-millisecond agreement between two instruments sharing no mechanism is the
strongest corroboration this campaign has produced.**

**The 10 s bound is the only threshold in the result**, and it is physical
rather than fitted -- but it is chosen, and the 29 must not be quoted without
it.

### The disagreements are predominantly THIS instrument, not the audio record

68 of 113 differ by more than two frames. **An initial reading blamed the audio
side, from sorting by absolute difference -- which puts the largest audio values
at the top by construction.** Counted across all 68 instead:

    KEYFRAME step exceeds 10 s     37
    AUDIO step exceeds 10 s         7
    both                            0
    keyframe step ~0 while audio claims one   11

**A population inferred from its extreme tail** -- the summary view had a shape
the data did not.

**The 37 blow-ups are the method's honest limit.** A 120 s half-window often
holds too few scene-driven keyframes to vote, and the histogram then peaks on a
coincidence instead of refusing. `segment_offset` requires six points and a peak
height of three; **that is not enough, and the bar was deliberately NOT raised
here -- tuning it to remove exactly these 37 would be fitting it to them.**

**Conclusion: complementary, not a substitute.** It corroborates the audio at
change points to better than half a frame on about half the cuts where both can
speak, and it refuses or blows up on the rest.

## 22. The ledger is the least accurate of the four records of this corpus

**Of 62 files in the `MERGED_UNVERIFIED` bucket, 36 failed with
`tools.software['ffmpeg'] is not configured`** -- `_binary()` correctly refusing
to guess a path in a run where `config.ini`'s `[software]` section never loaded.
**Those verdicts were about the instrument, not the files.**

All 34 that can be matched have since been re-verified:

    31  in_sync
     2  still unverifiable for real reasons
     1  OUT OF SYNC

**And all 34 still read `MERGED_UNVERIFIED` in the ledger**, because
`reverify_in_place` deliberately does not write `batch_state`.

### The refusal to write is sound and must not be reversed

Its docstring records why: `batch_verify one` **re-merges** before verifying,
and doing so once **collided with another agent's job and overwrote a ledger row
with `enqueue_failed`, destroying the evidence that put the file in the set.**
Not writing was the fix for real data loss.

**The cost of that fix was never priced, and it runs the dangerous way.**

### err_160: four instruments, three findings, and the ledger carries none

    ledger            MERGED_UNVERIFIED
    re-verify         out of sync, 84.1 ms
    C_cut dense scan  5 of 5 windows agreeing to 3.6 ms at ~85 ms
    per-track audit   two tracks end at 108.0 s of a 1523.6 s file

**The ledger is wrong; the re-verify is right for an unreliable reason** -- its
verdict rests on a **single window at peak 0.864**, the n=1 case -- **and only
the dense re-scan is both right and reliable.**

### Measured: 42 of 80 comparable files disagree

    files present in both records   80
      agree                         38
      DISAGREE                      42

**The divergence is not a handful of edge cases.** More than half of every file
that can be compared carries a different verdict in the two records.

### Consequence for every count published from the ledger

**Anything counting "unverified" from `batch_state` alone -- `master_summary`,
`build_state`, any status total, including counts this Lead has reported -- is
contradicted by files already on disk.** The Master Summary Table's
merged-unverifiable figure is an upper bound containing at least 31 files
already shown to be in sync.

**The cheap intermediate is a read-only reconciliation** reporting where the two
sources disagree: nothing written, nothing lost, and the divergence no longer
invisible.

## 23. A truncation nobody could see, because every track's duration WAS the container's

**err_691 retains about 24% of a hundred-minute programme on two tracks:**

    programme        5997.498 s
    audio tracks     17
    por short by     4539.610 s
    spa short by     4619.834 s

**The next largest shortfall anywhere in 141 measured tracks is 2.009 s** --
three orders of magnitude away. This is not a marginal call.

### Why it was invisible until today

**`probe` substituted the container length on 1926 of 1927 tracks.** So every
track's duration *was* the programme duration, and the question *"is this track
short"* answered **no by construction, for every file, always.** The
`duration_source` provenance added this afternoon is what made the question
askable -- **the finding is a consequence of the fix, not of a new search.**

**It sits as `MERGED_UNVERIFIED`, and an in-place re-verify called it
inconclusive and was right to.** That instrument asks about **sync**; these
tracks are not out of sync, they are **absent**. **Two instruments, two
questions** -- and the ledger recorded the answer to the one that was asked.

### Two files were unexamined rather than unverifiable

**err_95 and err_98** record `needs 2 'fre' tracks in the output, found 1` --
true. But the same output holds **two `eng` tracks against a master with one**,
which is exactly the structure the verifier looks for. **It fixed on `fre` and
never looked at `eng`.** err_91 had the identical shape and has since moved to
FAILED under the live sweep.

**Bounded**: the precondition is metadata-measured; whether those two `eng`
tracks are genuinely different sources needs decode. **So this is "the verifier
had a subject and did not try it", not "these files are in sync". The missing
piece is language selection, not a new instrument.**

### The remaining 26 share one missing measurement, and it is not a bug

**The candidate contributed no track in any language the master also has** --
the surplus languages are ones the master lacks entirely. **There is no pair
because the merge never made one.**

Conservation narrows without settling: **20 conserved, 4 end_shortfall.**
**Conserved is not sync** -- a track can run full length and be offset
throughout -- so they move from *unverifiable* to **"not truncated, still
unverified for sync"**. Settling them needs a comparison against the **source**
rather than a sibling track.

### The population moves while it is being read

**`batch_state.json` is written by the running sweep.** `verdict: unverifiable`
went **62 -> 60 between two queries ten minutes apart**, and a first pass
silently mixed rows read either side of that write -- a file selected on
`verdict == unverifiable` read back as `outcome: FAILED`. **Every list of this
population needs a snapshot timestamp or it is not reproducible.** The figures
above are one frozen snapshot, **2026-09-01T18:31:18Z**: **29 open at that
snapshot against 31 dispatched -- drift, not disagreement.**

**Open and not chased**: a **~2 s cluster** -- 13 tracks across err_127,
err_168 and err_406 ending consistently two seconds early.
