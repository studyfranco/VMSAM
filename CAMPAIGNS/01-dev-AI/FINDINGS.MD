# Findings — living report

Updated as results land, not at the end of a cycle. Every number here is
measured and carries its denominator. Anything not established is marked
**NOT CLAIMED**. Files are named by error id; resolve ids through the private
notes outside the repository.

Last updated 2026-09-02 — see **A2** and **B2** for the final day.

---

# Reader's guide

**For someone opening this cold.** Nothing here is new: every line points at a
section that already contains it. Sections are not renumbered or moved —
correction placement is itself a defect class in this project, so this is
additive.

## A. ESTABLISHED — measured, reproduced, not withdrawn

| | finding | strongest number |
| :-- | :--- | :--- |
| **§37** | **The truncation is non-deterministic.** Three runs, byte-identical input, three different answers | 6 of 7 tracks move between runs; `jpn` never does |
| **§28** | It reproduces on demand, and the pipeline stamped the truncated output **in sync** | full 7 tracks in, 6 short out, verdict `in_sync` |
| **§36** | The `-t`/`-copyts` ceiling is **refuted** — from the truncating invocation's own records | every `start_time` 0.000000, in and out |
| **§23** | A shortfall nobody could see, because every track's duration *was* the container's | 1926 of 1927 tracks carried a substituted container length, so "is this track short" answered *no by construction*. **Not err_691** — that file is retired from *truncation* in the section's own bounded correction |
| **§2** | Per-stream durations were the container's | **1926 of 1927 tracks** |
| **§31** | A one-millisecond track stamped in sync — and **coverage is not the fix** | 2 of 3 concealed files *were* verified and passed |
| **§32** | Four in-sync files are incomplete; a sign-loss bug ships in the merge path | 5 genuine merge truncations of 7 incomplete; 4 carry a sync verdict |
| **§26** | No track ever **exceeds** its limit | max over-run **42.0 ms** |
| **§34** | A timeout leaks a second process onto the same output path, **inside the closure** | reproduced: `sleep 8.37`, one process still alive |
| **§22** | The ledger is the least accurate of the four records of this corpus | — |

**The encoder's complete stderr for a truncating invocation contains no error, no
warning, and nothing naming a stream** (§28, §36). That is a measurement, not an
absence: the capture is the whole run, not a suffix.

## A2. ESTABLISHED 2026-09-02 — the last day of the campaign

**Added, not merged into table A.** These are new results, and this file's rule
is that sections are additive because correction placement is itself a defect
class here. Full working is in the four handovers in `task_archive/` and the
synthesis in `RESTART.MD`.

| | finding | strongest number |
| :-- | :--- | :--- |
| **bar** | **The mission's 15 ms bar had never been applied to any file.** `final_status` reaches the comparison only in its third branch, which needs a Step 1 verdict of `in_sync`/`out_of_sync` | **40 of 40** Step 1 verdicts said `unverifiable`, so **all 315 rows** exited at branch one or two. Both branches now proven to render (`test_65`) |
| **date** | **Nothing dates a measurement to the code that produced it** — extends §7 from artefacts to measurements | **0 of 110** lag figures attributable; **71 of 110** carry no row-level version or timestamp at all |
| **stamp** | The stamp that exists names the **container**, not the host module that took the measurement | affected rows span **two container builds** while sharing **one** host defect |
| **reach** | **A source fix does not reach a running process** | **24 records / 22 files** written after `7744aa3` still carried pre-fix behaviour, and the pre-fix wording proves it |
| **instr** | **A refusal that reaches the verdict slot is read as a finding about the file** | **31 of 45** per-language `unverifiable` records were `tools.software['ffmpeg'] is not configured`; only **14** were structural refusals |
| **absent** | **`no samples` was asserting a measurement that never ran** | **241** rows under one label became **234 never measured + 7 no samples**. FIXED |
| **guard** | The copy guard's real thresholds are **0.05 ms and 0.9985**, not `1e-9`/`0.999` — `round(l,1)` and `round(p,3)` sit four lines above the comparison | eight orders of magnitude, found by writing the test as reported and watching it fail |
| **peak** | **A peak of 0.91–0.98 is not evidence of a shared source** | two signals with **no sample in common**, sharing only an amplitude envelope, read **0.979** |
| **band** | **`SAME_RECORDING_PEAK = 0.98` does not survive codec-differing pairs** — the populations overlap and no threshold on this quantity orders them | same-source lossy transcode **0.869–0.963**; different-source shared bed up to **0.987** |
| **prov** | Retrospective provenance recovery | **37 recoverable / 24 undeterminable / 4 lost**, of 64 |
| **arte** | **The artefacts are not missing** — a field's absence was read as a file's absence | **0 of 106** verdict rows unlocatable; 185 `FAILED` rows lack the *field*, 44 have the file |

**The pattern across all eleven, and the reason this day is worth reading:** in
every case a reader with **two** outcomes was describing a world with **three**,
and the missing outcome was always *the instrument did not run*. Found five
times in one morning at five different layers, each by a different person, none
of them looking for it.

## B2. SUPERSEDED 2026-09-02

| what is superseded | the correction |
| :--- | :--- |
| **"Eleven artefacts are gone"**, and the Lead's framing of *a verdict outliving the thing it was about* | **All present.** A field's absence read as a file's absence — the same mistake already on record in `step2_only_class.md` with all 27. **Two agents then measured opposite answers on these eleven** and it is left **unreconciled deliberately**: treat as unverified in both directions |
| The **2 / 11 / 20 / 31** provenance split | **Withdrawn by its author**: an accurate reading of a store last written 2026-08-30T22:12:28Z, whose every record backing an `in_sync` verdict describes an artefact rewritten since |
| **`lang_pick or "jpn"`** as the cause of the Step 1 failures | **Two different defects in two files.** The `jpn` default is in `reverify_external.py` (fixed `769fb80`); `verify_unverified.py:38` has a fixed `fre > eng > jpn` priority and abandons after the first pick |
| **"87 of 199 land on the grid"** as a discriminator rate | Mixed two eras in one denominator. The discriminator is a **NON-ZERO** multiple of ten; an exact zero is zero at every resolution. **0 of 127** non-zero grid hits under refinement |
| Any count taken from **`external_verdicts` as one population** | It provably contains **both** code eras; its mtime tells a reader nothing about any individual row |

## B. SUPERSEDED — do not quote these without reading the correction

**A number below is still readable in place. Each one is wrong.**

| section | what is superseded | where the correction is |
| :-- | :--- | :--- |
| **§10, §12** | **PARTLY withdrawn** — about ⅗ and ¾ of each section is current, and §12's surviving result is positive: median \|error\| 3.0 ms over n=15 change points | in the headings, which now carry the replacement |
| **§16** | "steps are frame-quantised" as a **law** | same section: true of about half of files |
| **§13** | the 35% answerable pass rate | same section, marked WITHDRAWN |
| **§15** | that production could not find a constant delay | same section |
| **§20** | that `fps_err=30.0` was mislabelled 25.0 | same section: **the declared rate is correct** |
| **§24** | two eliminated mechanisms; err_112 as a second head | same section, both marked |
| **§26** | track counts — **325 → 238 → 205**, moved three times | §26 and §27 |
| **§28, §34** | control denominators — **NOT one chain.** `2 of 92` is over `stderr_tail` rows (comparable full merges); `ZERO of 96` is over `stderr_diagnostics` **key presence**, a different field 92 records lack entirely | §28 for the first, **§34** for the second |
| **§32** | err_233 listed as incomplete | **§33** — it is a content gap, not a defect |
| **§1** | a circular fit reported as evidence | §1, line marked WITHDRAWN |

**Pattern worth knowing before reading any count here: every quantity that moved
was a count. Nothing that was a bound or an event count ever moved** — 42.0 ms,
1441.942 s and the stop-event set held throughout -- EXCEPT its stated VALUE, which moved 6 -> 3 when invariant 144's own key was applied, so even this pattern has an exception, while track totals,
row totals and denominators each moved two or three times.

## C. OPEN — known to be unknown, and what would close each

| open question | what would close it |
| :--- | :--- |
| **Why the six streams stop.** The mechanism is a **class** — a race or resource condition — not a candidate (§37) | not a metadata question; four candidate mechanisms are already eliminated in §24, §36, §37 |
| **The magnitude.** 1000.494 / 1000.487 / 1000.456 are **three samples of a quantity that moves**, not three measurements of one constant (§37) | repeat runs; the spread is the observable, not the value |
| **Is the non-determinism general?** §37 establishes it on one file; **§39** measures the repeat population | **repeat runs of one subject**, retained deliberately rather than by accident |
| **Candidate-side identity.** `file_id` is stable for durable library paths and unstable for per-run temp copies, with no field saying which (§37 and the `record_entry_point` docstring) | a stable candidate-side identifier; counting distinct `file_id`s overcounts by the number of runs |
| **Sub-millisecond structure** | not reachable from container tags — **§38** measures the floor; needs decoded sample counts |

## D. NEEDS THE OWNER

**Not restated here — see `OWNER_QUESTIONS.MD`.** The process leak (§34), the
selection ordering, the sign fix (§32), and the `keep_best_audio` items are all
recorded there with their status.

**Three claims this guide could not make have since been written into the body,
as §38 and §39.** They were reported as absent here first, because a guide is
the wrong place for a first appearance — the guide now cites them rather than
introducing them.

**RESOLVED since this guide was written, and recorded rather than dropped:**
the retention item in `OWNER_QUESTIONS.MD` is now **CLOSED** -- see its heading
*"CLOSED -- no action needed: retain ONE first-pass intermediate"* --
superseded by §36 -- the measurement was already on disk, written by the passes
in question, and no retention was needed. **No action is required of the owner
or the Lead.** The original request is retained under that CLOSED heading for
its reasoning only.

## What resists summary, and why

**§24 says two things** — the correspondence between cut points and an unrelated
track's end, *and* the elimination of two mechanisms — and a one-line summary of
it will misrepresent one of them. Read it whole.

**§1 needs a forward pointer, NOT a superseded banner — and my earlier line here
said otherwise.** I wrote that §23, §28, §36 and §37 "supersede most of its
conclusions". **Reading it, that is too strong.** Its counts are corrected in
place and current — `CORRECTED` and `CORRECTED AGAIN` bring it to **4
established** — and it carries material that appears nowhere else: the delay
subtracted from a duration with **no sanity bound**, err_691 held apart on an
**80.224 s** spread, and the mkvmerge logging shim's first capture.

**What a reader of §1 alone would not know is that its open questions were later
answered**: the mechanism narrowed further in §24, the stop reproduced in §28,
the `-copyts` ceiling was refuted in §36, and §37 established the
non-determinism. **A blanket banner would mark current material as stale; a
pointer at the head would not.** That is the edit I would make, and it is 429
lines — the longest section in the file — so the pointer matters more than for
any other.

---

## 1. Output truncation — FOUR established (the heading said seven; see
CORRECTED and CORRECTED AGAIN in the body), mechanism narrowed, cause not found

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

## 10. PARTLY WITHDRAWN — measured at a geometry production does not use, but
the surviving three fifths stand (see "What survives, and it is stronger")

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

## 12. PARTLY WITHDRAWN — it measured a quantity the project already documents
as differing, but the step-agreement result stands: median |error| 3.0 ms over
n=15 change points, 10 of 15 inside one 23.976 fps frame

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

### FINAL: 8 of 8 carry the relation

All six remaining files were measured against acceptance criteria **committed
before the results existed** (`g24_check.py`):

    err_26   1.0010000  dev 0.00010%  resid 0.74 ms  edge 0/11
    err_27   1.0009998  dev 0.00012%  resid 0.87 ms  edge 0/11
    err_33   1.0009995  dev 0.00015%  resid 0.87 ms  edge 0/11
    err_35   1.0009994  dev 0.00016%  resid 0.55 ms  edge 0/11   step 1.9 ms
    err_37   1.0009995  dev 0.00015%  resid 0.43 ms  edge 0/10
    err_113  1.0009997  dev 0.00013%  resid 0.33 ms  edge 0/11   step 1.8 ms

**8 of 8, every file inside a 0.02% band, every residual under 1 ms against a
50 ms bar, zero windows at the search edge.**

**The too-good condition fired and was discharged by assertion, not by
judgement**: factors identical to five decimals, **spread 1.53e-06 against a
worst-case sigma of 6.07e-07 -- ratio 2.5**, where a clamped value gives zero
spread. **Quote 2.5, not the 4.3 from an earlier hand calculation** -- that used
each file's own sigma; the committed checker uses the worst case across all
files, which is the conservative denominator.

**The group is not ONLY a rate.** Duration deltas remain **174-205 s with the
opposite sign** where the relation explains 3.2 s. **These files carry a genuine
relation AND a large content difference** -- the content is what a duration
screen sees, which is exactly why it saw nothing else.

**Two files fit sub-frame steps** (1.9 ms, 1.8 ms) -- **not the ~180 s content
difference**, so that content is **spread through the file rather than sitting
at a discontinuity.** That distinguishes them from the C_cut family.

    genuine rate relations                21
    declared 24.0: relation CONFIRMED      8   ALL of them
    unstructured                           5
    mis-classified by metadata             1

**The ~180 s excess is still real** and still the same shape as Phase 2's zero
recovery (FINDINGS 11) -- unaccounted content. **What changed is that content
and a rate relation are both present, not that one replaced the other.**

### CAUTION: a declared rate relation is not a prediction that the file cannot merge

**A reframing that "~35 B_fps files cannot merge until the speed policy lands"
was falsified by the sweep within the hour it was published.** err_289 --
one of the eight carrying a measured PAL slope above -- **merged IN SYNC at 2.8
ms despite a declared 1.25 ratio.**

**A declared frame-rate difference is a fact about the VIDEO. Whether the audio
carries a matching relation, and whether it survives the merge, must be measured
per file.** The 21 "genuine rate relations" counted below are a statement about
the duration arithmetic, **not a count of files that will fail to merge.** Treat
it as an **upper bound**, not a prediction.

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

    BEFORE the tie refusal
    n = 61      median |difference|  33.3 ms
    within half a frame (20.85 ms)   29        48%
    within one frame  (41.71 ms)     35

    AFTER the tie refusal (current)
    n = 43      median |difference|   6.5 ms
    within half a frame (20.85 ms)   27        63%
    within one frame  (41.71 ms)     32

**The median improves five-fold and the raw count falls, because refusals
replaced numbers.** As a rate the method improves: **48% -> 63% inside half a
frame.** The count inside half a frame falls by **exactly 2** -- the cost
predicted in the characterisation and accepted in advance, two cuts whose
correct answer had won a coin-flip. **Both figures are given so neither can be
quoted alone.**

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

### The tie refusal: 37 blow-ups become 9

    cut-level outcomes   before  140 numbers, 23 refusals
                         after    76 numbers, 87 refusals

**Slightly more than half of all cut measurements were tiebreak results** --
invisible because a tiebreak and a measurement returned the same shape of
answer. **The 28 exact ties are gone and the nine already recorded by id are
untouched**, which is what a rule targeting only ties should do.

**Eight of the nine sit at a peak/runner-up ratio of exactly 1.03 -- one count
of separation.** Catching them needs a ratio bar that **8 of the 32 correct cuts
fall inside**, so they stay open: **nine wrong numbers left visible rather than
a bar fitted to them.**

**The 10 s plausibility bound remains the only chosen threshold in the entire
result.** The tie rule adds none -- equality only, pinned by a test that fails
if a ratio or margin ever enters it.

**The 37 blow-ups were the method's honest limit.** A 120 s half-window often
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

### The cap fix landed mid-sweep and is CONFIRMED firing, not merely present

The exception cap (§22's sibling defect) was raised while the sweep ran, on the
argument that `full_test.sh` starts **a separate `python3` per class**, so any
class not yet begun would import the changed module. **That was an account of a
mechanism, and four checks had confirmed only that the code was PRESENT in the
file the sweep imports. None had observed it FIRE.**

    D_trim boundary reached           19:29:56Z
    D_trim exception rows on 8e1b0e3   1
    over 220 chars                     1
    longest                          754

**B_fps finished on the old cap at 220; D_trim's first failure stored 754** --
same sweep, same process tree, one class boundary between them. **Both ends of
the message survived**: the head carries the phrase `master_summary` triages on,
the tail carries the second file path the old cap cut away.

**So the ~243 files still ahead record complete diagnostics, and the 35 B_fps
files are the only ones that missed it.**

**Two design choices made it a test rather than a confirmation.** It scoped to
**D_trim rows on the current build** -- a long exception from another class or an
earlier build could not have satisfied it. And it distinguished **UNTESTED from
REFUTED**, because a D_trim run with no failures stores no exceptions and
reading that as "the cap did not fire" would be absence arriving as evidence.
**That third outcome was built before the data existed, which is the only time
the decision can be made honestly.**

### Consequence for every count published from the ledger

**Anything counting "unverified" from `batch_state` alone -- `master_summary`,
`build_state`, any status total, including counts this Lead has reported -- is
contradicted by files already on disk.** The Master Summary Table's
merged-unverifiable figure is an upper bound containing at least 31 files
already shown to be in sync.

**The cheap intermediate is a read-only reconciliation** reporting where the two
sources disagree: nothing written, nothing lost, and the divergence no longer
invisible.

## 23. A shortfall nobody could see, because every track's duration WAS the
container's — and its opening example, err_691, is later retired from
*truncation* (its own master carries two of the tracks)

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

### FINDINGS 24's mechanism does NOT explain this -- they are two defects

**Settled structurally rather than numerically**, so it does not depend on a
delay record that does not exist:

    tmp_file_first_pass = md5(video_obj.filePath)[:16] + "_first_pass.mkv"
    one delay_to_put per video_obj
    one "-t first_pass_limit" on that whole output

**The 24 truncation is per SOURCE FILE, not per track.** Every stream from that
source is cut at the same instant. **err_691's `por` and `spa` differ from each
other by 80.2 s, and both are the master's own languages** -- same source, same
`video_obj`, same `-t`. **A per-file truncation cannot produce a per-track
difference.**

**Supporting magnitudes**: `delay_to_put` across 52 recorded runs spans **-4421
to +1980 ms** -- a maximum of 4.4 s. The formula would need **~1000x the largest
value ever recorded**, and `limit_le_zero` never fired in 52 records.

**And the shapes differ**: above 10 s, **12 of 14 tracks match no master track
at all**, where **16 of 16 did in the ~2 s band.** A class landing on no master
boundary is not the class produced by subtracting a delay from a
boundary-aligned duration.

**The delay is ABSENT, not small.** err_691 records `lag_ms_by_language: {}`,
`languages_attempted: ['jpn','eng']` only, with `por` and `spa` in
`languages_not_measurable`. **They were never attempted** -- not measured and
failed.

**An absence that is NOT evidence, and was nearly reported as one**:
`first_pass_limits.jsonl` holds nothing for err_691. **That is timing, not a
property of the file** -- the artefact spans 06:53Z-19:41Z and err_691 ran at
**02:39Z**, four hours before the instrumentation existed. **The merge's applied
delay for this file is not recorded anywhere reachable**, and
`lag_ms_by_language` is a *verification* measurement, not the merge input; the
two must not be substituted for one another.

**Scope**: this settles err_691 and the mechanism's shape. **The structural
point generalises to any file** -- a per-file truncation can never produce a
per-track difference. **Whether the rest of the >10 s class shares one cause is
not settled by one member.**

### The >10 s class, sized -- and the per-file cut is ruled out for ALL of it

    merged outputs in batch_state                116
    db_master present on disk                    115
    master recorded but GONE from disk             1    <- own category, not a pass
    successfully probed                          113

    short by MORE than 10 s        7 outputs / 14 tracks
    short by 0.5 - 10 s           22 outputs
    short by <= 0.5 s             84 outputs

    >10 s shortfalls: min 693.4   median 1334.9   max 4619.8 s

**In every one of the seven, other tracks run FULL LENGTH while the short ones
are cut:**

    out 1   1 short of 3     others 1501.3, 1501.5
    out 4   6 short of 9     others 1441.0, 1441.9
    out 7   2 short of 17    the two short ends DIFFER

**A single `-t` truncates every stream in its output; it cannot leave some at
full length.** So the per-file cut explains **none of the seven** -- the argument
made for err_691 now covers the class.

**Where the ends land: 12 of 14 on nothing** (11 nothing, 1 empty track, 1
master_min, 1 master_a0) -- reproducing an earlier sample result on the full
set.

**A false positive caught in the author's own test**: the empty track first
scored as landing on a round number, **because 0.0 is a multiple of 60.** A
zero-length track is not landing on anything.

**Refuted for 11 of 14 tracks: "the short tracks come from a short source."**
**BOUNDED CORRECTION -- this was tested against CANDIDATE durations only.**
err_691's own master carries `spa` at 1377.664 and `por` at 1457.888, **both
matched exactly by the delivered tracks** (FINDINGS 32), so those two ARE
inherited; err_112 is a third. **11 of 14 untouched, and every structural
argument in this section stands whole.** What dies is the word *truncation*
applied to err_691. Below: no short-track end
matches any candidate duration, and **every candidate runs full length** -- 1434
to 6015 s against short ends of 0 to 1458 s.

**Codec is associated but not determining:**

    short >10 s   flac 12, eac3 2
    NEVER short   aac 185, dts 6, ac3 3
    flac 217 total, 12 short (5.5%)    eac3 297 total, 2 short (0.7%)

**No aac, dts or ac3 track is ever short** -- but 94.5% of flac is fine, and ten
flac tracks run full length **inside the affected outputs themselves**.

**REFUTED as a class property: "the short tracks are the master's own
languages."** True in **1 of 7**, and that one is err_691, from which the pointer
was raised. **Four of seven record no `selection_reason` at all.**

**Established for all seven**: the shortfall is **per-track-subset within an
output, never whole-file**; it lands on nothing measurable in 12 of 14; and it
is **not inherited from a short source.**

**Open lead, not yet tested**: the codecs that are never short (aac, dts, ac3)
are plausibly **stream-copied**, while flac is the **re-encode** target -- err_127
and err_168's tracks arrived as aac and left as flac. **If the truncated tracks
are exactly the re-encoded ones, that unifies both bands and locates the defect
in the encode rather than in any duration arithmetic.**

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

## 24. The merge re-encodes complete audio and cuts it short at an unrelated track's end

**The ~2 s shortfall cluster is two mechanisms, and only one of them is ours.**
Widened to every merged output with a master still on disk: **16 tracks in 6
files** in the 1.9-2.1 s band, and **16 of 16 end exactly where a master audio
track ends.** Above 10 s, 12 of 14 match nothing -- so the err_691 class
(FINDINGS 23) is cleanly separate.

### INHERITED -- 4 tracks, not a merge defect

Each is **byte-for-byte identical to a track in its own master** -- same
`NUMBER_OF_BYTES`, same `NUMBER_OF_FRAMES`, same codec. **Copied, and the master
was already short.** Nothing was lost that was not already missing, and no merge
change would fix it.

### PROPAGATED -- 12 tracks, and this one IS the merge

    err_127   ger eng spa ita por fre    2.007-2.009 s short
    err_168   ger eng spa ita por fre    1.997 s short

**Distinct encodes, not copies** -- all seven byte counts in each file differ.
They arrived from the candidate as **aac** and left as **flac**, so they were
**re-encoded**. And **their sources were full length for their own programme**:
err_127's candidate tracks run 1474.560 s of a 1474.598 s programme.

**Yet every one lands on exactly the master's shortest audio track:**

    output tracks end   1430.048 s / 1430.059 s
    master's shortest   1430.048 s / 1430.059 s     to the millisecond
    frame counts        14897 against 14897          exact
    programme runs      1432.056 s

The `jpn` track in each output is the master's own, **copied, and full length.**

**So the merge re-encoded audio that was complete and cut it ~2 s short of the
programme, at the end of an unrelated master track.**

### Not determined: FIRST audio track or SHORTEST

**In both files `fre` is both**, so these twelve cannot separate the two rules.

**An earlier pass appeared to separate them and was wrong.** Four tracks matched
"shortest" while not matching "first" -- which looked decisive, and those four
are the **inherited** cases: **they match the master's shortest track because
they ARE that track.** A copied track matches its own source trivially and
discriminates nothing. **The apparent evidence came entirely from conflating the
two mechanisms this work exists to separate.**

### MECHANISM CONTESTED -- the site is real, its causal role is refuted twice

**Read this before the section below.** The code site is correctly identified
and the arithmetic is confirmed. **What is refuted is that this line is what
shortened these files.** Two agents, two instruments, independently:

**Direction 1 -- the truncations have no delay to explain them.**

    err_211   three recorded delays, ALL 0.0 ms   -> predicts 0.000 s
              measured shortfall                             1000.487 s

**A negative delay lengthens the limit and cannot truncate at all.** err_211 has
**three independent records agreeing on zero.**

**WITHDRAWN from this refutation: err_112.** It was cited here as a second
instance, and **it was never truncated** -- its source is already short (see
below). **Its recorded delay proves nothing about a mechanism that had nothing
to do.** The refutation stands on err_211 one way and err_25 the other, **on one
file fewer than first reported.**

**And err_112 was one of only TWO truncated files thought to carry a recorded
delay at all. Removing it leaves exactly ONE file where the first-pass formula
can be tested against a real truncation** -- which must be known before this is
counted as two independent refutations.

**Direction 2 -- the one large positive delay cut nothing.**

    err_25    recorded +1980.0 ms   -> predicts a 1.980 s shortfall
              measured shortfall                        0.005 s

**The strongest single test available, and the formula fails it.** (err_25 also
carries a 0.0 ms record from a second merge, so the surviving output could be
from either run -- **stated because it weakens the test, not because it rescues
the formula.**)

**Denominators**: 315 ledger rows; **115 pairs with both master and output on
disk** -- 199 have no output on disk and are **ABSENT, not passing.** Only **22
error ids carry a recorded `delay_to_put` at all, and 31 of 35 records are
exactly 0.0 ms.** So the mechanism is **recorded 35 times and inert in 31 of
them.**

**Five of the seven large truncations have no recorded delay**, so for those it
is **untested rather than refuted** -- the honest limit on the refutation.

**Implied delays for the >60 s class run 693 s to 11,422 s.** A delay of eleven
thousand seconds is not an audio delay; **the absurdity of the number is itself
evidence against the mechanism.**

**One measurement excluded from every conclusion**: err_691's master reads
12,799 s -- three and a half hours -- against 1,378 s of audio, **almost
certainly a master/candidate pairing mismatch.** Flagged because its 11,422 s
figure is the largest in the set and would otherwise anchor the result.

### A located candidate: the SECOND pass `-t`, ungated, alongside `-copyts`

    mergeVideo.py ~3280   base_cmd.extend(["-map","0", ..., "-c","copy","-copyts"])
    mergeVideo.py  3294   base_cmd.extend(["-strict","-2","-t", duration_best_video, ...])

**Every other duration path is gated on `delay_to_put`; this one is not:**

    first pass   -t duration - delay/1000        gated   (refuted above)
    second pass  -ss when delay < 0              gated
    second pass  aresample async first_pts=0     gated
    second pass  -t duration_best_video          NOT GATED   <- every merge

**err_211 is what makes this decisive.** Its recorded delay is **0.0 ms three
times over**, so **no gated path can fire** -- and it is still short by 1000.487
s. **The truncation is not delay-dependent at all**, which eliminates the entire
gated family rather than only the first-pass `-t`.

**The arithmetic.** The same command carries **`-copyts`**, preserving input
timestamps. If `-t` is then measured against those timestamps it becomes an
**absolute end time**, predicting `output_length = duration - content_start`:

    err_174    master 1441.909   audio   0.001   implied start 1441.908
    err_211    master 1441.960   audio 441.473   implied start 1000.487
    err_112    master 1501.500   audio 808.128   implied start  693.372
                                                 ^ NOT A TRUNCATION -- its
                                                   source is already short; see
                                                   the withdrawal below. Listed
                                                   only as it appeared in the
                                                   original fit.

**err_174 is the fit that makes this more than arithmetic**: a shortfall of
1441.908 s against a master duration of **1441.909 s -- equal to three decimals**
-- with an audio track of **0.001 s**. An absolute cut at `t = duration`, on
content whose timestamps begin at that duration, **produces exactly an empty
track.** That is the `limit_le_zero` boundary the recorder's docstring names,
reached by a different route than the one it was written for.

**`-shortest` is REFUTED**: absent from `src/` entirely, literally and
dynamically. `video.get_shortest_audio_durations` exists and is in the protected
closure, but **its single caller bounds delay-measurement windows and writes
temp WAVs -- it never touches the output.**

**VERIFIED BY READING**: the ungated `-t` beside `-copyts`; no other
unconditional duration operation in either pass; `-shortest` absent; the gated
paths inert at delay 0.

**TESTED AND CONFIRMED** -- `-copyts` makes `-t` an ABSOLUTE timeline end:

    ffmpeg -y -f lavfi -i sine=frequency=1000:duration=60 -c:a flac BASE
    ffmpeg -y -itsoffset 40 -i BASE -c copy -copyts OFFSET
    ffmpeg -y -i OFFSET -map 0 -c copy -copyts -strict -2 -t <T> OUT

    input packets    40.000 .. 99.977   span 59.977 s

    -t 50   packets 40.000 .. 49.985   CONTENT SPAN  9.985 s   predicted 10
    -t 45   packets 40.000 .. 44.992   CONTENT SPAN  4.992 s   predicted  5
    -t 40   "Output file is empty, nothing was encoded"        predicted ~0

**The boundary case reproduces err_174**: an input whose timestamps begin at the
limit yields an empty file -- that file's 0.001 s audio track against a 1441.909
s master.

**A near-miss that nearly refuted a correct hypothesis.** `format=duration`
reported **50.008 and 45.015** for those runs. **Read as content length that says
RELATIVE**, and the refutation was being written before the packet timestamps
were checked. **It is the absolute timeline END, not the content length.** That
is the defect `record_first_pass_limit`'s own docstring names -- *"every
per-stream duration in the pipeline is really the container's"* -- **hit while
testing the mechanism that docstring describes. A duration field is not a length
when the start is not zero.**

**Superseded, for the record**: that `-copyts` makes that `-t` absolute. **The
arithmetic fits six files and fits err_174 exactly. That is strong and it is not
a test.** The direct test is one ffmpeg invocation on a synthetic input with
offset timestamps -- no library file, no decode slot.

**NOT EXPLAINED**: why the content carries a non-zero start timestamp at all,
and why implied starts range from 693 s to 1442 s. **This locates the cut; it
does not explain the offset that feeds it.**

### The candidate ALSO explains the per-track contrast that killed everything else

**Every mechanism so far died on the same objection: `delay_to_put` is per
`video_obj`, so it cannot make two tracks of one output differ.** An absolute
`-t` under `-copyts` **has no such problem -- each track carries its own start
timestamp**, so one unconditional `-t` produces per-track cuts.

**Output 4, nine tracks, one recorded candidate -- the control is INSIDE a single
output, needing no cross-file comparison:**

    jpn  flac  <- aac        1441.9   FULL    re-encoded
    eng  flac  <- aac,eac3    441.5   SHORT   re-encoded
    spa/spa/ita/por/tha flac <- aac  441.5  SHORT  re-encoded

**Six flac tracks cut to exactly 441.5 s and a seventh flac track in the same
output at 1441.9 s.** Same output, same codec, same conversion, opposite
outcomes.

**The arithmetic closes:**

    absolute -t at the master duration        1441.9
    short tracks end                           441.5  -> implied start 1000.4
    err_211 implied start (measured earlier)  1000.487, audio 441.473
    full jpn track ends                       1441.9  -> implied start    0.0

**A track whose timestamps begin at zero survives whole; a track whose
timestamps begin at ~1000 s is cut to the remainder.** That is one rule producing
both outcomes in one command, and it is what no gated mechanism could do.

### ELIMINATED TOO -- the truncation exists BEFORE the second pass runs

**The absolute-cut mechanism is refuted on direct evidence, from data already on
disk.** `record_stage_artefact` has been recording `first_pass_out` -- **with
per-track `start_time`** -- the moment the first pass writes it. **50 records
exist.**

**Every audio track's `start_time` in all 50 falls between 0.000 and 0.959 s.**
None at 693 s or 1000 s. **The absolute cut is real ffmpeg behaviour and its
precondition is absent.**

**err_112 is NOT A TRUNCATION AT ALL, and an earlier revision of this section
used it as though it were.** Corrected:

    err_112 master   eng 00:13:28.128
            first_pass_out / second_pass_out / final output   eng 00:13:28.128

**808.128 s, identical at every stage INCLUDING THE SOURCE. The master's own
`eng` track is short; the pipeline copied it faithfully end to end and truncated
nothing.**

**Across the six examined, one is not a truncation:**

    err_174   source 1439.911 s   output   0.001 s
    err_160   source 1522.681 s   output 108.008 s
    err_194   source 1504.794 s   output 171.175 s
    err_45    source 1458.976 s   output 137.824 s
    err_211   source 1440.832 s   output 441.473 s
    err_112   source   808.128 s  output 808.128 s   <- SOURCE ALREADY SHORT

**Five genuine truncations with full sources. The population is five, not six.**

**err_211's records were misread as a different run. They are the SAME run, the
other side of it.** All three stage records belong to the **master** (or to the
candidate's chimeric):

    chimeric_out     subject CANDIDATE   7 tracks   all 24:01.960   FULL
    first_pass_out   subject MASTER      4 tracks   all full
    second_pass_out  subject MASTER      4 tracks   all full

**And the output splits cleanly by source:**

    full  in output:  ger, fre, jpn                  <- MASTER's, minus eng
    short in output:  eng, spa, spa, ita, por, tha   <- CANDIDATE's, minus jpn

**Every short track comes from the candidate. Its chimeric is recorded full. Its
two passes have no record at all.** The master side was probed end to end and is
full throughout, which is what made the records look like a different run.

**The missing artefact was on disk.** Filed under a file_id that resolved
against nothing, because the subject is **the chimeric-corrected candidate's
temp path** -- a fourth path kind the reverse map does not cover. Its timestamps
sit seconds after the master records: **the same run, the third side of it.**

    first_pass_out    container 1441.959
        eng 24:01.947  spa 24:01.947  spa 24:01.947  ita 24:01.947
        jpn 24:01.959  por 24:01.947  tha 24:01.947      ALL SEVEN FULL
        every start_time 0.000000

    second_pass_out   container 1441.942
        eng 07:21.485  spa 07:21.485  spa 07:21.479  ita 07:21.473
        por 07:21.479  tha 07:21.485                     SIX SHORT
        jpn 24:01.942                                    ONE FULL

### RESOLVED: the truncation is in the SECOND pass

**Against a prediction registered at `f1b0f3b` before the record was found** --
*"candidate `first_pass_out` FULL at ~1441.9 s, `second_pass_out` short at
~441.5 s"* -- **measured: seven full at 1441.947-1441.959, then six at
441.473-441.485 with `jpn` alone surviving. Confirmed including the count.**

**The named falsifying observation is the one that occurred**: candidate
`first_pass_out` showing all seven tracks full. **The truncation is not in the
first pass.**

**And the absolute cut is eliminated a second time, independently.** Every track
on both sides reads **`start_time = 0.000000`**, so `-t duration_best_video`
under `-copyts` cuts at 1441.9 s of content and **cannot** produce 441.5 s. It
was already out on recorded delays; the start times rule it out from the other
direction.

### The six did not get CUT -- they STOPPED. The `-t` demonstrably worked.

**All seven tracks receive byte-identical encoder options** -- `-c=flac
-compression_level=12 -sample_fmt=s32 -exact_rice_parameters=1` -- with `jpn` as
`a:4` carrying exactly what the six cut tracks carry. **No filter, no seek, no
per-stream duration, no codec difference.** The command distinguishes nothing.

**And the `-t` acted -- on the track that survived:**

    -t                    1441.942000000
    jpn   first_pass_out  1441.959  ->  second_pass_out  1441.942   trimmed 17 ms
    the six              1441.947  ->   441.473 - 441.485

**`jpn` was trimmed to exactly the `-t` value, so the limit is working
correctly. The other six never reached it.**

**A `-t` cannot produce output shorter than its limit. Only the encoder stopping
can.** The search has been for something that **cuts**; the evidence says
something **stopped**.

### SETTLED from the ledger and the artefact together -- no slot required

**For err_211 the limit is not inferred, it is recorded:**

    first_pass    -map 0:a?  -map 0:s?    -t 1441.942000000    exactly one -t
    second_pass   -map 0                  -t 1441.942000000    exactly one -t

**`-map 0:a?` takes EVERY audio stream of input 0 in a single command.** And the
artefact's ends fall in four groups:

    441.473   441.479   441.486     six tracks, 13 ms apart
    1441.942                        three tracks, EXACTLY the limit

**Three tracks reached 1441.942 -- precisely the `-t` -- on that command, beside
the ones that did not.** The limit demonstrably worked on their siblings under
the same `-t`, and **a `-t` truncates and cannot lengthen.**

**So the six did not hit their limit. Established from records already on disk**,
which is what a queued container run was going to settle.

**Ten instances of the same structure exist**, found by counting only tracks
**not** byte-identical to a source (a copied track says nothing about an encoder
that never wrote it): 60 outputs with 2+ propagated tracks, 16 with 2+ distinct
end groups, **11 that actually merged, 10 with spread beyond encoder rounding.**

**err_211 is the only one whose limit is on disk** -- the other nine predate the
argv ledger, so they give *"at least one track per file is short of its limit"*
without a value.

**NOT claimed: that the ten share a cause.** Ten files showing one structure are
ten instances of a pattern, **and the only mechanism evidence available is a
silence.**

**Three claims, one closed**: *which* tracks stopped short is settled; **why**
is untouched, and **the magnitude** -- 1000.456/463/469 against a derived
1000.487 -- is untouched.

**And they stopped together**: 441.473, 441.479, 441.479, 441.485, 441.485,
441.486 -- **13 ms of spread across six independent streams. One event in the
invocation, not six coincident failures.**

**The record that would say why postdates the run.** `ffmpeg_results.jsonl`
carries exit code, elapsed and `stderr_tail`, and holds **no record for this
file_id**:

    results recorder earliest entry   2026-09-01T14:59:10Z
    the truncating second pass ran    2026-09-01T08:03:55Z

**Seven hours late** -- the same shape as `first_pass_limits` not covering
err_127 and err_168. **The instrument arrived after the event it was built for.**
All 92 records it does hold read **`exit_code: 0`**, consistent with this class
of failure reporting success.

**This `exit_code` claim STANDS and is not affected by the stderr withdrawal
elsewhere in this document.** The same 92 rows are used for two different
things: **the exit codes are complete and untouched by the 4000-character cap;
the `stderr_tail` field is truncated on all 92 and yields only 2 comparable
controls.** Different fields of the same rows -- one withdrawn, one not.

**Not explained: why `jpn` survives.** All seven took the same code path -- the
delay is 0.0 ms, so the `delay < 0` branch never fires and every track went
through the FLAC branch. **Six cut, one not, under a `-t` that applies to the
whole invocation.** One fact recorded without being built on: in
`first_pass_out`, **`jpn` is the LONGEST track** -- 1441.959 against the others'
1441.947, a 12 ms difference -- **and it is the one that survives.**

### Where §24 stands: two mechanisms located, both eliminated by data

    first-pass  -t duration - delay/1000     eliminated by recorded delays
    second-pass -t under -copyts             eliminated by recorded start times

**The `-copyts` behaviour is confirmed and real; it is simply not what happened
here.** The truncation in err_112 exists **before the first pass completes**,
which places it in the first pass itself or in what the first pass read.

**A note on the tooling.** A retain flag (`VMSAM_RETAIN_FIRST_PASS`, default
off) was authorised and built to capture this artefact. **It turned out to be
unnecessary: the measurement was already being recorded.** A duplicate recorder
line was written and **rejected by an existing stage-enumeration test**, which is
what such tests are for. **The flag is harmless while off and its justification
is now much weaker than when it was authorised.**

### Superseded: what remained unmeasured when the artefact was thought deleted

**Published outputs start at ~0** (0.959, 1.079, 0.000), so **the offset the
mechanism needs is invisible in the final file.** It would live in
`first_pass_out`, which `tools.file_remove` deletes at the end of the run.

**And err_211 needs PER-TRACK offsets, not one file-wide offset:**

    a:1 ger  24:01.000  full     a:2 eng  07:21.486  short
    a:5 fre  24:01.000  full     a:3 spa  07:21.486  short
    a:7 jpn  24:01.942  full     a:6 ita  07:21.473  short

**One absolute cut truncates every track at the same timestamp**, so tracks
ending at 24:01 and 07:21 in one file can only come from one cut **if they began
~1000 s apart. Consistent with the mechanism, not evidence for it** -- nothing
on disk records those starts.

**The measurement that settles it**: capture `first_pass_out`'s per-track start
timestamps **before** `tools.file_remove` at `mergeVideo.py:3314`.
`record_stage_artefact` is already called on the adjacent line for
`second_pass_out`.

**Confirmed**: the `-copyts` reading is now tested. **But it now
explains the per-track contrast as well as the frame-exact ends**, and the
per-track contrast is the fact that refuted the previous three candidates.

### What still needs a mechanism

**The frame-exact correspondence in the ~2 s band is unexplained again.** Twelve
tracks landing on the master's shortest audio track **to the millisecond, with
frame counts 14897 against 14897**, is a fact. The `-t` formula predicted it
elegantly and now fails a direct test. **Do not let the elegance of the refuted
explanation stand in for a surviving one.**

### Code site found -- and it reverses an earlier exoneration

`mergeVideo.py:3226`, in `generate_new_file_launch_cmd`:

    first_pass_limit = Decimal(duration_best_video) - (Decimal(delay_to_put)/Decimal(1000))
    cmd_first_pass.extend(["-strict", "-2", "-t", str(first_pass_limit), ...])

**`duration_best_video` is the best video's VIDEO track duration**, traced to
`best_video.video['Duration']` and threaded unchanged through three functions.
**The second pass is excluded structurally**: at 3295 it applies `-t
duration_best_video` **unmodified**, and takes one input -- the first-pass temp
file. Anything shorter than the video duration was already short on arrival.

**Earlier in this campaign the `-t` in `generate_new_file_launch_cmd` was
EXONERATED. That exoneration is withdrawn.** The inspection read line 3228's
`-t` and the name `first_pass_limit`, and **the subtraction is on line 3226, one
line above.** The name is accurate about where the number starts and **silent
about what happens to it.**

    err_127   master video 1432.056   output ends 1430.048   difference 2.008 s
    err_168   master video 1432.056   output ends 1430.059   difference 1.997 s

**The formula requires `delay_to_put` = 2008 ms and 1997 ms** -- ordinary
magnitudes in this corpus.

### The FIRST-or-SHORTEST question has a third answer: neither

**No audio duration is read at that site at all.** The limit is
`video_duration - delay`, and **the exact correspondence with the master's
shortest audio track is a CONSEQUENCE, not a cause**: the delay was derived by
aligning against that track, so the cut lands where that track ends. **The
correspondence is exact precisely because the same track determines both sides
of it.**

**That explains an exactness a duration-copy could not.** Frame counts matching
**14897 against 14897** requires the cut on the same sample -- and a limit
computed from a delay measured against that track achieves it **without anyone
reading its duration.**

### Not yet established, and cheap to close

**The formula is confirmed end-to-end -- on OTHER files.**

    first_pass_limit records                              50
      formula  video - delay/1000 == limit  holds        50/50
      and that limit appears as the ACTUAL -t in the
      recorded ffmpeg command for the same file_id       49/50

**Two independent records agree**: the computed value and the command line that
actually ran. The single unmatched row has **no `ffmpeg_argv` entry at all**,
not a disagreeing one. Worked example: video 3354.021, delay 1980 ms, limit
3352.041 -- **and both 3354.021 and 3352.041 appear as real `-t` values for that
file**, as expected when different tracks carry different delays.

**What is NOT confirmed is the application to these two files.**
`first_pass_limits.jsonl` holds 50 records and **neither err_127 nor err_168 is
among them** -- both instrumentation files postdate those merges. The 26
distinct recorded `-t` values span 358.358 to 3354.021 and **include none of
1430.048, 1430.059 or 1432.056.**
The relation `video - limit == delay/1000` holds on **every recorded row**, but
the specific delays for err_127 and err_168 are **inferred from the arithmetic,
not read.** **CORRECTION: no re-run can close it while these files fail where they do.**
err_127 fails with `Multiple delay found` **at the delay test, which raises
BEFORE `generate_new_file_launch_cmd` is reached.** `first_pass_limit` is
therefore **never computed and never written**.

**"One re-run closes it, no decode" was stated twice -- in a report and in a
message -- and is withdrawn.** Not *did not* produce the record: **cannot**,
while the file fails at that stage.

**So 2008 ms and 1997 ms remain inferred from arithmetic with NO PATH TO
CONFIRMATION** short of the file getting past the delay test. **A re-run was
attempted and returned UNTESTED**, which the three-way verdict correctly
distinguished from a refutation:

    UNTESTED -- the re-run wrote no first_pass_limit record.
      No record means the first pass did not run, which is not evidence
      about the delay either way.

**Without that third outcome it would have read as a refutation of the causal
claim above. It is not one.**

### The mechanism was already written down

`record_first_pass_limit`'s own docstring:

> **THIS IS THE NUMBER THAT TRUNCATES TRACKS AND IT HAS NEVER BEEN WRITTEN
> DOWN.** ... so a delay comparable to the runtime removes most of the file --
> and every downstream stage then reports success.

**Whoever added that instrumentation knew the mechanism.** What was missing was
the connection to twelve measured tracks. **A correct description of a defect,
sitting in the source, does not find the defect** -- it waits for someone to
arrive carrying the matching measurement.

### Original note, superseded above

`video.get_shortest_audio_durations` is the obvious suspect **by name** and is
**ruled out by the call graph**: exactly one caller, in `prepare_get_delay_sub`,
which cuts segments for delay measurement and does not build the output encode.
`first_pass_limit` at `mergeVideo.py:3230` is the **video** duration -- 1441.942
s on err_211 -- so it is not the source of a 1430.048 s cut either.

**Correspondence established to the millisecond and the frame; site not found.**

### One hypothesis killed cheaply

Six tracks sharing a duration and frame count across six languages **is exactly
what a duplicated-and-relabelled stream looks like**, which would be worse than
a truncation. **All seven byte counts differ**, so they carry different audio.
Refuted from metadata, before it could become a scare.

**Noticed, not chased**: err_19's output holds **eight `eac3` tracks with
byte-identical sizes** (25,449,472). Constant-bitrate eac3 at one duration gives
identical sizes, so this is probably arithmetic -- **but it is the same
signature as the case just refuted, and it has not been checked.**

**Scope**: 113 master/output pairs probed; masters no longer on disk are absent,
so **this is a floor, not a census.**

## 25. The repair fires and declines -- and the ledger records a different failure

**First observed execution of `repair_not_compatible_candidate`.** On every
prior image it returned False without running, so **every earlier container-side
repair result was vacuous.** On `c7c9c3f`, from the container log for err_108:

    FUSION| repair applied      : (not recorded)
    FUSION| exception           : Exception: The incompatible candidate cannot be repaired:
                                  neither the speed nor the structural chain applied to it
    FUSION|   File "mergeVideo.py", line 5431, in merge_videos_with_repair
    FUSION|     repair_not_compatible_candidate(not_compatible_video, ...)
    FUSION|   File "mergeVideo.py", line 1294, in repair_not_compatible_candidate
    FUSION|     raise Exception("The incompatible candidate cannot be repaired: ...")

**It was entered, executed, and raised from inside itself.** That is a stack
frame, not an inference. **The repair fires and declines for a stated reason --
a finding about the repair, not a deployment failure.**

### The durable record carries the LATER, less informative failure

    container log : "The incompatible candidate cannot be repaired..."   <- the repair's
    ledger row    : "Variance delay in the second test is to big [...]"  <- the delay test's

**The merge attempted the repair, was declined, fell through to the ordinary
path, and failed there -- and only the second failure reached the ledger.**

**A reader of `batch_state` alone would conclude err_108 failed on delay
variance and would never learn the repair had been tried and refused.** The log
says so in its own words: **`repair applied : (not recorded)`.**

**Same shape as the folder-86 `RECORDED ONLY` defect and as the 34
known-but-unrecorded verdicts (FINDINGS 22): the informative event happens and
the durable record carries the later, less informative one.** Every failure-
reason tally drawn from `batch_state` -- including the Master Summary Table --
is a tally of **last** failures, not of **causes**.

### The repair's verdict cannot be counted for any run longer than the log window

**A per-file fired/declined count was requested for Phase 2's 30 files. It is
not obtainable -- not by any agent, with the current instrumentation.**

The container log is a **bounded 201-line rolling window**. Measured while
Phase 2 was running:

    frames entering repair_not_compatible_candidate   4
    "cannot be repaired" declines                     6
    "repair applied : (not recorded)" blocks          2

**Two FUSION blocks -- roughly two files of the twelve attempted. The other ten
had rolled off.** The window covered about **17%** of the question, **and the
missing part is gone rather than unread.**

**By the time all 30 finish, the window will hold the last two of thirty.**

**This is `repair applied : (not recorded)` doing exactly what it says.** The
repair's verdict is emitted to a rolling log and **never made durable**, so the
count is unobtainable for any run longer than the window. **The question is not
answerable until something records it** -- which is a more useful result than
the number would have been.

**PHASE 2 COMPLETE: 30 of 30 attempted, 30 FAILED, three failure heads.**

    28  "Multiple delay found with the method 1..."   the 27-file series
     1  "Delay found between ..."                     err_307, a singleton
     1  "Variance delay in the second test is to big" err_108, a singleton

**The 27-file series IS uniform. Each of the three appended singletons fails its
own way** -- which is neither of the two shapes reported while it ran.

**Both earlier statements in this section were wrong, and each had more data
than the last:**

    at 12 files   "not uniform, two heads"   -- one head was a class of one
    at 22 and 24  "uniform, one head"        -- the 28th file had another
    at 29         correct

**The correct statement at 12, 22 and 24 was "n of 29 so far, heads so far",
and it was not made.** A population described from a prefix of itself is not a
population claim.

**Superseded, for the record -- Phase 2 at 22 of 29 started** (`batch_state` is keyed by FILE PATH, not
`error_id`): **21 FAILED, and all 21 carry ONE failure head** -- *"Multiple
delay found with the method 1"*.

**An earlier revision said "two distinct failure heads, so Phase 2 is not a
uniform population." That is WITHDRAWN.** The second head belongs to **err_108
alone -- one of the three singletons, not part of the 27-file series.**

**The 29 ARE uniform.** The claim was drawn from a sample of twelve that
happened to include the singleton run first, **and generalised to the
population. Twelve was enough to see two heads and not enough to see that one of
them was a population of one.**

**Correct shape: the 27-file series fails one way, and at least one singleton
fails another.**

### The stderr tail structurally CANNOT hold the diagnostic -- and the controls are 2, not 92

**Verified by the Lead after asserting the wrong figure:**

    rows                             92   (46 first_pass + 46 second_pass, 44 files)
    tail length min / med / max    4000 / 4000 / 4000   <- EVERY tail truncated
    elapsed < 10 s                   74 of 92
    clean file AND invocation >= 10 s  2   <- the usable controls

**"All 92 are successful merges giving a real denominator" was wrong. 74 of 92
ran under ten seconds**, so most are not full merges and are not comparable
controls. **The denominator is two.**

**And the window covers the wrong part of the run. For every comparable run it
holds NOTHING ELSE:**

    progress-meter share of the 4000-character window
      long runs  (>= 10 s, n=18)   median 100.0%   max 100.0%   (3998 of 4000)
      short runs (<  10 s, n=74)   median  10.1%

    long runs                                18
    with ANY non-progress line in the tail    2   <- the usable controls

**Not ~94% -- for every run of ten seconds or more it is 100.0%.** The window
contains nothing but the progress meter on all 18. **The tail holds the last
~20 seconds of runs lasting 97 to 227 seconds.**

**So the "zero genuine diagnostics across 92 tails" result was STRUCTURAL, not
empirical: it measured a window that, for every comparable run, could not have
contained one. The instrument had no capacity to disagree.**

**A diagnostic emitted when a stream stops early happens mid-run by definition**,
and the progress spam that follows pushes it out before the process exits. **A
re-merge run against this instrumentation would return no diagnostic, and that
would be a fact about the cap, not about the encoder.**

**This is the pre-registered "absence is not evidence" condition, known in
advance to be GUARANTEED rather than possible.** The experiment would have spent
a decode slot to produce an uninterpretable null.

**Fix**: `_stderr_diagnostics`, recorded **beside** the tail. ffmpeg rewrites its
progress line with a carriage return, so the noise is trivially identifiable;
dropping it leaves the stream listing and any real diagnostic. **`stderr_tail`
is unchanged, so nothing that reads it moves.** Applied to the 92 existing
already-truncated tails it removes 33% of the characters and retains **5,820
non-progress lines**.

**Privacy note**: dropping the progress meter promotes ffmpeg's `Input #N ...
from '<path>'` lines into the kept text, so the field carries media paths --
same class as `stderr_tail` already held, same destination outside the
repository. **Cite by `file_id` only.**

### The stderr cap is pinned by code AND by test

`mergeVideo.py:2951` -- `stderr_bytes.decode(...)[-4000:]` -- is
**container-side**, so raising it means edit, land, CI build, deploy: **a full
cycle, not a flag.** `test_44_ffmpeg_result.py:82` asserts the length is
**exactly 4000**, so it is a two-file change. **Its docstring gives a privacy
reason for the cap: the exception's message embeds the full command.**

**So the queued re-merges will be read knowing the limit, and a line absent from
a capped tail is UNRECORDED, not absent.**

**The exception-cap fix carried across the deploy**: err_108's stored exception
is **621 characters with head and tail intact**, independently reconfirming on a
second build what the class boundary demonstrated on the first.

## 26. No track exceeds its limit by more than 42 ms -- and 12 never reach it

**The stopped-not-cut reading, tested corpus-wide.** Joining recorded first-pass
limits to per-track ends: **205 tracks across 21 outputs** (of 113 probed --
bounded by which files have a recorded limit, not by the corpus).

    RECORDED limits      21 outputs, 205 tracks
      exceed the limit (by 1-42 ms)             130
      exactly on the limit                       28
      SHORT of the limit                         47
        of which SHORT by more than 1 s          12   <- the signature
      LONGER by more than 1 s                     0
      over-run: min 1.0 ms  median 25.0 ms  max 42.0 ms

    RECONSTRUCTED limits 84 outputs, 470 tracks
      at the limit                              292
      within 4.5 s                              172
      SHORT                                       6
      LONGER by more than 1 s                     0

**No track exceeds its limit by more than 42 ms** across 105 outputs and **675** tracks
(205+470; an earlier revision said 708, computed from the superseded 238). **The short ones never reached the bound**, which a limit cannot cause.

**An earlier revision said "OVER = 0" and "no track anywhere exceeds its limit".
That is FALSE as written and is corrected here.** It holds only at second scale:

    tracks with end > limit at all     204 of 325   <- DENOMINATOR IS THE
                                                       INFLATED JOIN, see below
    by how much: min 1.0 ms, median 25.0 ms, max 42.0 ms
    over-runs exceeding one frame       11   (all at 42.0 ms, 0.3 ms past)

**The 325 here is the pre-correction count and the ratio has NOT been
re-derived.** **Note that BOTH terms move, not just the denominator** -- the
corrected total is 205, and `204 of 205` is not a possible answer. Duplicated rows inflate a count and a median; **they cannot
inflate a maximum**, so **42.0 ms stands unchanged** and the headline bound is
safe. **The "204 of" fraction and the median should not be quoted until the join
is re-run on the deduplicated set.**

**The over-runs are container rounding, not a failure of `-t`.** The finding
survives; **the absolute phrasing does not, and an absolute claim is exactly the
kind that travels without its threshold.** The Lead called `OVER = 0` "the
cleanest single number in the campaign", which is how it spread.

**The null itself passes the capacity test (invariant 141): the instrument CAN
produce a value on the other side of the bound, and does so on a large majority
of tracks**, so `OVER = 0` at the 1 s threshold is a real measurement rather
than a structural shape. **The capacity claim needs only "not all one sign" and
does not depend on the exact fraction** -- which is as well, since that fraction
is the one marked not-yet-re-derived above. **Deduplication removes whole rows,
so the sign distribution survives it.**

**The track count in this section has been wrong twice and is now 205.**

    325   double-counted via batch_state rows (once per candidate)
    238   double-counted AGAIN via limit_scan.json's duplicate rows
    205   correct

`limit_scan.json` holds 113 rows over 105 distinct masters -- **eight appear
twice** -- so the first correction iterated its rows and re-entered those eight.
**One duplication was fixed and another walked into, inside the commit whose
subject was fixing a duplication.**

**205 is confirmed by two methods sharing no assumptions**: identity-based dedup,
and a value-based reconstruction that infers each output's fan-out as the
**minimum tuple multiplicity** and divides. **The second names the case that
would break it** -- an output whose tracks are all genuine pairs, which would
double its `N` -- **and that case does not occur here**, since the eight
duplicates are repeated rows of one output rather than two candidates with
fully-duplicated track sets.

**The event count was unchanged by the deduplication**, because an event keys on
(output, end-time) and duplicates collapse. **Its VALUE is 3, not 6** -- six is
the track count, and invariant 144 records that using it here was the error this
rule warns against.

### What moved and what did not -- the pattern is the finding

**Only the TOTALS moved, and they moved three times. Everything that survived
did so because it was counted as an EVENT or bounded by an EXTREMUM:**

    42.0 ms maximum      invariant under duplication by construction
    3 stop-events        keyed on (output, end-time) -- SIX is the track
                         count, and using it here was the error this rule
                         warns against; see the correction in invariant 144
    capacity claim       needs only that both signs occur -- 130 over, 47 under
    track totals         325 -> 238 -> 205

**Totals are what duplication reaches.** A count is the one statistic that a
repeated row changes, and it is the one this section reported three times.

**Reconstructed and recorded rows are kept separate and never pooled** -- a
reconstructed bound is weaker evidence. **The reconstruction is bounded by the
largest delay ever recorded, 4.42 s**, so a >4.5 s shortfall survives the worst
case and **a ~2 s one does not: the ~2 s band cannot be judged from
reconstructed limits at all**, and was not.

**Provenance**: 54 records over 50 distinct file_ids -- 27 master, 9 candidate,
**18 UNRESOLVED**, kept as its own category. An unresolved subject is not an
absent record (invariant 139).

### Multi-track stops agree to milliseconds

    out 5   6 short   441.473 - 441.486   spread 13 ms
    out 6   4 short   108.008 - 108.012   spread  4 ms

**WITHDRAWN -- refuted on its own example.** The sentence read *"encoders
failing independently do not land within 4 ms; one event per output, not several
coincident failures."* **FINDINGS 37 measured that same output three times**:
three distinct endpoints in every run, per-track deltas
`[-0.006, -0.012, 0.000, +0.012, +0.006, 0.000]`, **"the movement is per-track,
not per-invocation"**, and between 3 and 6 stop times.

**Tracks landing within 13 ms WERE failing independently.** This is an argument
from tightness, so **no marker on the 4 ms or 13 ms figures reaches it** -- the
numbers are correct and the inference drawn from them is dead.

**What survives**: the clustering is real and it distinguishes *"the assembler
lost one input"* from *"six unrelated failures"*. **What does not**: that
tightness proves a single event.

### Mixed outputs -- the prediction that killed every cutting hypothesis

    out 1   1 short, 21 AT the limit
    out 2   1 short, 21 AT the limit
    out 4   1 short,  2 AT the limit
    out 5   6 short,  1 AT the limit

**One track stops at 441.5 s while twenty-one others in the same output run to
the bound**, under the same command and byte-identical options. **Four
independent outputs.** A per-file limit cannot make two tracks differ -- that
objection eliminated three mechanisms, and it is exactly what the stopped
reading predicts.

### COUNT EVENTS, NOT TRACKS

**Three different quantities have been called "the event count" in this document
and they are not the same thing:**

    magnitudes seen in the pre-correction set   4   (~2 s, 693, 1000.46, 1415.6)
    events on the deduplicated recorded set     6
    outputs in the >10 s class (FINDINGS 27)    7   <- outputs, NOT events

**The 4 below belongs to the pre-correction table and is retained to show what
the double-count did.** The current figure on the recorded set is **6 events**.
**FINDINGS 27's "seven" counts OUTPUTS, not events**, and citing it as an event
count was an error of this document.

    1.763, 1.809, 1.967      the ~2 s band
    693.372                  alone
    1000.456 - 1000.469      SIX TRACKS, ONE OUTPUT
    1415.594 - 1415.598      FOUR TRACKS, ONE OUTPUT

**Both the 4 and the 14 in this block are pre-correction figures**, retained to
show what the double-count did; the deduplicated set gives **12 short tracks and
6 events**.

**6 of the 14 sit at 1000.46 +/- 0.01, matching err_211's 1000.487 -- but that
is one output's six tracks, not six independent files.** **As events: one at ~2
s, one at 693, one at 1000.46, one at 1415.6. Four magnitudes, four events.**

**Counting the six as six data points would manufacture a cluster from a single
file.**

**So the ~2 s band and the large band are the same SIGNATURE at different
magnitudes -- both are stops below a correctly-applied limit -- but nothing here
says they share a cause, and they must not be merged on this evidence.**

### Widening the coverage adds NO new event

    recorded       6 events from 12 tracks
    reconstructed  6 events from  6 tracks

**Going from 21 outputs to 84 more adds no new event** -- so **widening the
RECONSTRUCTED-LIMIT coverage adds none.**

**This does NOT close the population.** FINDINGS 27 later proves three further
tracks short of their limit **from ordering alone, on files where no limit was
ever recorded** -- exactly the case this join cannot reach. **The stop-event
population is larger than what this section measures.**

**Not reached**: why a stream stops. Nothing here touches the encoder's own
reason.

## 27. The 0.5-10 s band: three quarters inherited, and three tracks proven short without a limit

**26 outputs / 106 tracks** at snapshot (`batch_state` is rewritten by the
sweep, so the population drifts). Every shortfall is a track's own `DURATION`
tag against the output's **video** tag -- never a container length.

    inherited                              78 tracks / 22 outputs
    propagated, ends SHORT of the limit     3 tracks /  3 outputs   PROVEN
    propagated, ends AT the limit           0 tracks                none provable
    undeterminable, limit never recorded   25 tracks / 10 outputs

**Three quarters of the band is not a merge defect.** The 78 are **byte-for-byte
identical to a track already in the master or candidate** -- same
`NUMBER_OF_BYTES`, same `NUMBER_OF_FRAMES`. Copied, not produced. **Consistent
with err_112 at 693 s: the band boundary does not predict the mechanism and only
the copy test does.**

**Of the 28 propagated, every one is flac (therefore re-encoded) and 27 end
within 50 ms of an audio track's end in their own master** -- the err_127 /
err_168 correspondence generalised across the band. `err_80` alone lands on
nothing.

### Three tracks are provably short of their limit WITHOUT recovering any limit

From the argv ledger's structure: a first-pass invocation carries up to 36
`-map` arguments and **exactly one `-t` -- 52 of 52 recorded invocations.** So
every stream from one pass shares one limit.

    err_217   1440.857  vs  1440.896     39 ms apart
    err_197   1441.043  vs  1441.058     15 ms
    err_214   1440.976  vs  1441.016     40 ms

**A `-t` truncates and cannot lengthen. With ends `E1 < E2` from one invocation,
`E2 <= t`, therefore `E1 < t` strictly.** **The earlier track is short of its
limit whatever the limit was.**

**So propagated-short is non-empty, and the stop-event population is larger than
the seven of FINDINGS 26.**

### Nothing goes in "ends at the limit"

**The 25 undeterminable are kept first-class, not folded in.** 25 dwarfs the 3
that were provable, and assigning them to either verdict would claim the
opposite of what was measured. **No correctly-trimmed file is confirmed in this
band, so the band does not shrink on this evidence.**

### Three kinds of absence, and only one is fixable

    ffmpeg_argv.jsonl        earliest  2026-09-01T07:15:54Z
    first_pass_limits.jsonl  earliest  2026-09-01T06:53:13Z
    the ten outputs          written   2026-08-29 .. 2026-09-01T04:05:15Z

**All ten artefacts predate both recorders.** This is **not** a file_id
resolution failure and **no resolver fix recovers it**. **"Unresolved", "never
recorded", and "written before the recorder existed" are three different
things**, and only the first is worth fixing.

**Not claimed**: that the 27 were cut **by** the master audio track they land
on. The correspondence is exact and repeated; **the code site is unidentified**
-- `get_shortest_audio_durations` has one caller, in the delay-measurement path.
**Correspondence, not mechanism.** And `err_80` landing on nothing is not
claimed as a distinct defect; one file may be a measurement gap.

> **MERGE-PATH BOUNDARY, 2026-09-03 00:51Z.** The container moved `5b1cb82` ->
> `e75c783`, which carries **48 lines of merge-path code**, not docs and not
> tooling: `src/mergeVideo.py` (+40) and `src/merge_video_chimeric.py` (+8) --
> the shape-based progress filter and the completeness check. **Everything in
> sections 24 to 34 was measured on `5b1cb82` or earlier. Measurements taken
> after this boundary are under different merge-path code and MUST NOT be pooled
> with them without saying so.** The Lead described this image as carrying
> tool-only changes, twice; that was wrong and is corrected here.

## 28. The stop REPRODUCES on demand -- and the pipeline stamped the truncated output IN SYNC

**err_211 re-merged on `0149a52`. The truncation recurred, identically:**

    first_pass_out   container 1441.959
      eng 24:01.947  spa 24:01.947  spa 24:01.947  ita 24:01.947
      jpn 24:01.959  por 24:01.947  tha 24:01.947      ALL SEVEN FULL
      every start_time 0.000000

    second_pass_out  container 1441.942
      eng 07:21.479  spa 07:21.473  spa 07:21.479  ita 07:21.485
      por 07:21.485  tha 07:21.485                     SIX SHORT
      jpn 24:01.942                                    ONE FULL

**Same six tracks, same ~441.48 s, same survivor within the candidate's set,
same 13 ms of spread. The defect is a reproducer now, not a historical
artefact** -- any future instrumentation can be tested against it on demand.

### The output has THREE complete tracks, from TWO sources -- and that sharpens it

An earlier revision of this section said *"jpn is the only survivor"*. **The
final output holds three complete tracks**, and an audit of the artefact on disk
is what caught it:

    ger   1441.0     complete   <- from the MASTER's second pass
    fre   1441.0     complete   <- from the MASTER's second pass
    jpn   1441.942   complete   <- from the CANDIDATE's second pass
    eng spa spa ita por tha   441.473 - 441.486   SHORT, all from the CANDIDATE

**Two second-pass invocations ran, on two different subjects:**

    master's    ger 24:01.959  eng 24:00.880  fre 24:01.959  jpn 24:01.959   all full
    candidate's eng 07:21.479  spa 07:21.473  spa 07:21.479  ita 07:21.485
                jpn 24:01.942  por 07:21.485  tha 07:21.485   SIX SHORT, ONE FULL

**`ger` and `fre` never passed through the invocation that truncated.** They are
survivors of the output, not of the event.

**So the thing to explain is unchanged and is now stated precisely: within the
CANDIDATE's seven tracks, `jpn` alone survived.** Anyone reasoning from "three
survivors" would look for what `ger` and `fre` share with `jpn`; **the answer
must account only for `jpn` against its six siblings in one command.** The
master's own `eng` and `jpn` were dropped from the output entirely.

**This does not soften the finding** -- six tracks still hold 31% of the
programme and the file is recorded verified.

### For `eng`, the loss was RECOVERABLE: a complete track existed and was dropped

    MASTER   eng 1440.832   fre 1441.001   ger 1441.001   jpn 1441.001
    OUTPUT   eng  441.479   fre 1441.0     ger 1441.0     jpn 1441.942

**The master carried a complete `eng` track at 1440.8 s. The output holds one
`eng` track and it is the 441.5 s one.** The master's is not in the output at
all.

**The other five short tracks have no master counterpart, so nothing was
available to lose. `eng` is the one case where a complete track existed and the
truncated one was kept in its place** -- **recoverable loss, not merely
unrecorded loss.**

**The selection mechanism is not itself the defect.** `jpn` shows it working:
master 1441.001 dropped, candidate 1441.942 kept, **both complete**. It is doing
what it is for.

**NOT ESTABLISHED -- and this is the whole question: whether the selection ran
before or after the truncation.**

    BEFORE   the selector chose a track that was subsequently truncated,
             and cannot be blamed
    AFTER    it compared a 441 s track against a 1440 s one and kept the
             short one -- a different and much worse finding

**This cannot be told from the artefact, and an ordering must not be inferred
from a result.** `keep_best_audio` is the owner's territory; this is recorded as
an observation about the output, not as a scoped defect.

### The verdict on that merge was `MERGED_IN_SYNC 17.7`

**The pipeline produced a file with six audio tracks holding 31% of the
programme, and recorded it as merged and in sync.** Not failed, not flagged --
**verified.**

**And every stage agreed:**

    ffmpeg exit codes, all four records          0
    lines matching error/warning/failed          0 of 4 records
    merge verdict                                MERGED_IN_SYNC 17.7

**Four independent reporting surfaces -- the encoder's exit status, its stderr,
the merge outcome and the sync check -- and none of them registers that two
thirds of the audio is missing.** The sync check is not wrong on its own terms:
it compares what is present, and what is present is aligned.

### `stderr_diagnostics` is capped identically and fills with metadata

**All four new records are exactly 4000 characters.** Dropping the progress
meter created no headroom -- **per-stream metadata blocks expanded to fill it**,
exactly as pre-registered:

    rec 1   79 metadata lines,  8 Stream # declarations, 10 other
    rec 4   94 metadata lines, 17 Stream # declarations,  2 other

**But the null is STRONGER than the pre-registered exclusion, and the reason
matters.** The old tail held the **last** 4000 characters -- ~94% progress meter,
covering roughly the final 20 seconds. **Stripping the meter moves the window to
run from mid-listing to the final mux summary, so it covers the ENTIRE ENCODE
PHASE.**

Between the end of the stream listing and the final summary, all four
invocations emitted:

    first_pass     nothing
    second_pass    "Press [q] to stop, [?] for help"

**No error. No warning. Nothing naming a stream.** The only other non-listing
content is the `muxing overhead:` summary, present in all four and therefore
background by the registered criterion.

**So for the encode phase this is "no error", not "no record" -- the absence is
about the ENCODER, not about the instrument.** *"The encoder stops silently" is
a different and harder finding than "the message was in a log we truncated"*,
and it **eliminates the whole family of hypotheses that assumed a message
existed to be found.**

**The limitation that remains, stated rather than glossed**: all four fields are
exactly 4000 characters, so all four are truncated **at the start**, and for a
nine-track file the stream listing consumes **82-94% of the budget**. **A
diagnostic emitted during input parsing, before or within the listing, would
still be lost.** The null is sound for the encode phase and silent about the
parse phase. **The listing is as filterable as the meter was.**

**And the magnitude is NOT settled by the reproduction.** Observed shortfalls
**1000.456 / 1000.463 / 1000.469** against the independently derived
**1000.487** -- 20-30 ms apart. **A record naming nothing settles neither the
mechanism nor the magnitude**, and the reproduction of 1000.456 must not stand
in for agreement with 1000.487.

**The privacy rule applies to the field** -- it carries stream titles and
language tags for a named series, so it stays on disk and is cited by
`file_id` and record number, never quoted.

## 29. Five outputs conceal shortfalls up to 1440.9 s — and the count of those
claiming to be in sync is FOUR by ledger outcome and FOUR by owner-table status,
with an intersection of three (see the correction in the body)

**A ceiling of 1.92 s was reported for shortfalls hidden inside outputs marked
complete. That ceiling was a property of the SELECTION.** Re-run over every
output with a `MERGED_*` verdict, damaged or not:

    filtered (74 "complete" outputs)   ANSWERED 59  UNDETERMINABLE 15  max    1.136 s
    unfiltered (all MERGED_*)          ANSWERED 95  UNDETERMINABLE 16  max 1440.9 s

    above 1.92 s: 5
      err_174  1440.9 s     err_194  1333.6 s     err_45  1321.9 s
      err_211  1000.5 s     err_112   693.3 s

**Three orders of magnitude.**

**Capacity established before either run** (invariant 141): on err_211 -- known
to hold six tracks a thousand seconds short -- longest 1441.942, shortest
441.473, **gap 1000.5 s, detected.** The method can return the other answer.
**Zero tracks lacked a `DURATION` tag across both runs**, so no duration is a
container substitution.

**No new damage: all five were already identified.** What is new is **the
magnitude each partial verdict concealed.** These are not "files with short
tracks" -- they are **verdicts resting on intact tracks while up to 1440.9 s was
missing from the same file.**

**CORRECTED -- the sentence below named the wrong field and there are TWO live
fours, not one three.** Verified against `batch_state.json`:

    batch_state outcome MERGED_IN_SYNC   err_45, err_112, err_174, err_211
    owner-table "Step 2 verified"        err_45, err_174, err_194, err_211
    intersection                         err_45, err_174, err_211

**err_112's OUTCOME is `MERGED_IN_SYNC`** -- the original sentence said the
opposite. **And err_194, whose outcome is `MERGED_UNVERIFIED`, is presented to
the owner as "Step 2 verified" while missing 1334.9 s.**

**Both fours are correct about their own quantity.** `master_summary.final_status`
prefers `external_verdicts` and falls back to the ledger row only when nothing
else spoke, **so a `MERGED_IN_SYNC` file can render "Merged, unverifiable" and a
`MERGED_UNVERIFIED` one can render "Step 2 verified".** Ten files are in that
state.

**The original claim, retained because its reasoning is sound where it applies**:
*a file that admits it could not be verified is not making a false statement
about itself* -- **true, and it does not apply to err_112 or err_194, neither of
which is in the state the sentence assumed.**

## 30. The "two-method agreement gate" is a one-sided threshold on the second method

`mergeVideo.py:441` and `:452`, reached only **after** method 1 has produced a
single delay:

    calculated_delay = delay_first_method + (delay_second_method * 1000)
    if abs(calculated_delay - delay_first_method) < 500: accept

**`(d1 + d2*1000) - d1` is `d2*1000`. The first method's value cancels
exactly.** Verified against the source:

    d1=  100.0  d2=0.2 -> |calc-d1| 200.00  accept
    d1= 5000.0  d2=0.2 -> |calc-d1| 200.00  accept   <- d1 changed 50x, same result
    d1=-3000.0  d2=0.6 -> |calc-d1| 600.00  REFUSE

**The condition reduces to `abs(d2) < 0.5 s` -- a magnitude test on the
correlation pass alone.** A pair is refused when the second method returns half a
second or more, **whatever the first method said and however well the two
agree.**

**The name, the comment and the exception message all describe an agreement
test.** The exception reads *"unexpected between the two methods"*; the code
never compares them. **Consequences run both ways: two methods agreeing closely
on a large delay are REFUSED; two disagreeing wildly with a small `d2` are
ACCEPTED.**

**The sibling site at `:898` does the same test honestly.** It computes
`calculated_delay = mean_between_delay + round(d2*1000)` -- note the changed
first term, with `delay_first_method+...` left commented beside it -- and then
gates on **`abs(delay_second_method) < 0.125` directly.** **Same quantity, stated
plainly, no cancelling expression.** One site says what it does; the other does
not.

**Not proposing a threshold change**: the population reaching this gate is **one
file** (err_307), and a bar tuned on n=1 is worse than a misnamed one reported.

### err_307 is a singleton by folder and by nothing else

It is the only file in folder 176, so series membership is **undefined rather
than false.** Its two-track shared language looked distinctive -- err_100 has
`eng x2, fre x2, jpn x2` and failed at the **earlier** gate, so the same shape
appears on both sides of the split.

**Its only distinguishing feature is WHICH GATE IT REACHED** -- a fact about how
far it got, not about what kind of file it is. **err_307 passed the gate that
stopped the other 28.**

## 31. A one-millisecond track stamped in sync -- and coverage is not the fix

**Capacity established first**: on err_211 the instrument returns a **1000.469 s**
shortfall, so every null below is a measurement.

**The 20 zero-coverage `MERGED_IN_SYNC` rows resolve to one cause, not three:**

    A: no per-language record at all      20
    B: record exists, nothing attempted    0
    C: comparison ran, matched nothing     0

    ANSWERED 20   UNDETERMINABLE 0   conserved 14 | end_shortfall 5 | TRUNCATED 1

**err_174: programme 1441.901 s, `eng` short by 1441.900 s.** The track is **one
millisecond long**, in a row stamped `MERGED_IN_SYNC`, verdict `in_sync`, **with
no language ever examined.**

### The overlap is exactly one, and that is the finding

    err_160   MERGED_UNVERIFIED   1415.6 s   zero coverage
    err_174   MERGED_IN_SYNC      1441.9 s   ZERO COVERAGE   <- the only overlap
    err_194   MERGED_UNVERIFIED   1334.9 s   zero coverage
    err_211   MERGED_IN_SYNC      1000.5 s   HAS coverage
    err_45    MERGED_IN_SYNC      1321.9 s   HAS coverage
    err_691   MERGED_UNVERIFIED   4619.8 s   zero coverage

**Three `MERGED_IN_SYNC` files conceal a >=1000 s shortfall; only ONE is
attributable to zero coverage.**

**Two of the three were verified and PASSED.** err_211 attempted `jpn` and `eng`
and **both passed, at 1.3 ms and 17.7 ms, while six tracks were missing 1000.5
s.**

**So zero coverage is not the risk -- it is one route to the same blind spot.**
Closing it catches err_174 and leaves err_211 and err_45 exactly as they are.
**"1 of 20 truncated" would read as "coverage is the fix", and it is not.**

**A language passing at 1.3 ms is not evidence the file is whole.** The verifier
measures **alignment between two tracks**; an absent track has no lag to be
wrong about, and a present aligned track says nothing about its siblings.

**Not claimed**: the 14 conserved files are **conserved, not verified** -- a
track can run full length and be offset throughout.

## 32. Four in-sync files are incomplete -- and a sign-loss bug ships in the merge path

**A completeness check (each audio track's `DURATION` against the output's video
`DURATION`, metadata only) over 117 outputs on disk:**

    109 complete       8 incomplete   <- SUPERSEDED: err_233 withdrawn as a
                                        false positive, err_112 and err_691
                                        inherited from an already-short master.
                                        5 genuine truncations. See below.

    err_691   MERGED_UNVERIFIED   4619.8 s
    err_174   MERGED_IN_SYNC      1441.9 s
    err_160   MERGED_UNVERIFIED   1415.6 s
    err_194   MERGED_UNVERIFIED   1334.9 s
    err_45    MERGED_IN_SYNC      1321.9 s
    err_211   MERGED_IN_SYNC      1000.5 s
    err_112   MERGED_IN_SYNC       693.4 s   <- NEW to this count, and INHERITED:
                                              the MASTER's own eng track is 808 s
                                              in a 1501 s file. The pipeline copied
                                              a short source; it truncated nothing.
                                              See the inherited/propagated split.
    err_233   (no outcome)          86.2 s   <- WITHDRAWN, see below

**CORRECTED: 5 of the 7 incomplete outputs carry a genuine merge truncation, not
7.** Two are **inherited** -- the master's own track was already short and the
merge carried it forward:

    err_112   master's own eng   808.128 s vs its longest 1501.452 -> short 693.324
              DELIVERED shortfall                                     693.372
    err_691   master's own spa 1377.664 and por 1457.888, both matched exactly

**48 ms apart on err_112; exact on err_691. The merge did not truncate these.**

**The headline is UNTOUCHED, and that was checked first**: both inherited cases
are `Merged, unverifiable`, **neither is among the four carrying a sync verdict
that approved them.** *A correction that quietly deflates the finding it corrects
is the one to distrust.*

**This is a GAP in the completeness column, not a bug.** It asks *"is every audio
track as long as the programme"* and answers correctly -- **what it conflates is
the cause.** And `explained_by_content_gap` does not reach it: that compares the
**candidate's container against the master's**, a pair-level difference, while
**this is a within-master defect -- one track short inside an otherwise
full-length file**, which no container-level comparison can see.

**Four of 52 `MERGED_IN_SYNC` rows conceal an incomplete track, not three --
but they are not four of a kind.** **err_112 is INHERITED**: its master's own
`eng` track is already short, so the pipeline copied a short source rather than
producing one. **Three are propagated; one is inherited.** The distinction is the
one drawn earlier in this document and it must not be lost by listing them in a
single undifferentiated column.
err_112 sits **below the >=1000 s band everyone had been searching and above
anything ordinary** -- it was missed because the threshold, not the file, was
where attention stopped.

**`duration_source` is reported throughout**, with the container fallback named
as `container_fallback` rather than applied silently -- the substitution that
made this question unanswerable on 1926 of 1927 tracks.

**Capacity is a TEST, not a habit**: the first test class proves the detector
fires, using err_174 and err_211 as fixtures, **before anything asserts a null**
-- and proves it can also return `complete`, so it is not answering yes to
everything.

### The bug the new tests found, and it ships

    "-00:00:01"  ->  +1.0

**`-00` is negative ZERO, so the hours term contributes nothing and the sign
vanishes.** Verified independently:

    -00:00:01 -> +1.0      <- sign lost
    -01:00:00 -> -3600.0   <- sign survives

**The existing negative test used `-01:00:00`, where the sign survives into the
total, so it passed throughout** while a small negative duration returned
positive.

**Present in all three copies of the parser, including the one that ships in the
merge path.** **Exposure is bounded, with one qualification the author volunteered**: the
merge-path copy is reachable only with `VMSAM_STREAM_DURATION_TAG` **on**, and
the gate is **off by default** -- so that copy is real, shipping, and **not yet
live.** **But the same bug was in `completeness_check` and `track_conservation`,
diagnostic tools run directly -- so it WAS live in this session's own
measurements until all three were fixed.** No reported result depended on it,
since no real `DURATION` tag carries a leading sign. Fixed by rejecting a leading sign in the raw text -- **checking the
result cannot catch it**, because the result is indistinguishable from a
genuine positive.

**A fixture exercising the easy half of a condition is the recurring shape of
this session; this is the fourth.**

### err_214 did not test the discriminator -- it never reached it

**A third outcome now distinguishes "never reached the encoder".** err_214 failed
at the delay test before any merge command and wrote **zero** ffmpeg records.

**Filing that beside err_211's "recorded and silent" would treat opposites as
comparable**: one is an encoder that ran and said nothing, the other an encoder
that never ran. **The structural discriminator -- seven propagated tracks with
two distinct end times -- remains UNTESTED and needs a file that actually
merges.**


## 33. err_233 is a false positive, and the fix is `all`, not `any`

**err_233 never ran.** `stage: enqueue_failed`, `http: 409`; the classified row
agrees from the other side -- `status: pending`, master
`never_run_so_never_looked_up`.

**Its absent outcome is not a ledger gap.** `outcome` is absent **because there
was no outcome, and the cause is recorded.** That is a **fourth kind of absence**
beside *unresolved*, *never recorded* and *written before the recorder existed*
-- **and it is the benign one.** A file with no verdict is not a file that
passed; this one is also not a file that was silently dropped.

**The artefact is not this row's product.** `merged_file` was **carried forward**
from an observation five hours earlier, and the file's mtime (06:24:41Z) is
**later than the row pointing at it** (06:13:03Z). **Nothing in a completeness
measurement would have shown that -- only the ledger says the two are
unconnected.**

### The 86.2 s was the pair's own content gap

    measured shortfall    86.164 s
    recorded dur_delta    86.176 s
    apart                  0.012 s

**For C_cut a genuine content difference is the normal case**, so calling it a
truncation files an expected property of the pair as a merge fault.
`completeness_check` now takes `dur_delta` and reports
`explained_by_content_gap`.

### `all`, not `any` -- and err_691 is why

Every short track must be explained. **One track short by the content gap and
another short by a thousand seconds is still a finding**, and letting the
explained one excuse the rest is how a real defect hides behind an expected one.

    err_233   short   86.2   dur_delta   86.176   MATCHES
    err_112   short  693.4   dur_delta    0.007
    err_45    short 1321.9   dur_delta    4.756
    err_174   short 1441.9   dur_delta   27.031
    err_211   short 1000.5   dur_delta   33.024
    err_194   short 1334.9   dur_delta   71.859
    err_160   short 1415.6   dur_delta   85.825
    err_691   short 4619.8   dur_delta 6783.840   <- gap EXCEEDS the shortfall

**Exactly one matched.** The other seven exceed their content gap by ten to a
hundred times, **so the four `MERGED_IN_SYNC` result is unchanged** -- and
err_233 was never among those four, having no outcome at all.

**err_691's `dur_delta` of 6783.8 s exceeds its 4619.8 s shortfall, so a loose
tolerance would absolve the worst file in the corpus.**

**Left open, deliberately**: whether the 409 was retried -- one refusal recorded,
nothing after, **a service question rather than a forensic one.** And **16 tracks
short by ~21.4 s in the carried artefact** -- real, unexplained, and belonging to
whichever run produced that file. **Better attached to nothing than to the wrong
row.**

## 34. A timeout leaks a second process onto the same output path -- and it is in the closure

**`tools.launch_cmdExt_with_timeout_reload`, `src/tools.py:92-113`. Verified by
reading the source, not inferred:**

    while unpocessed:
        cmdDownload = Popen(cmd, ...)                  # line 95
        try:
            ... communicate(timeout=timeout)
            unpocessed = False
        except TimeoutExpired:
            cmdDownload.kill()
            max_restart -= 1
            if max_restart < 0: raise
            else:
                cmdDownload = Popen(cmd, ...)          # line 111  REPLACEMENT

**On timeout it kills the original, starts a replacement at 111, then falls out
of the `except` with `unpocessed` STILL TRUE.** The `while` repeats, **line 95
starts a third process and rebinds `cmdDownload` over the replacement.**

**The process started at 111 is never communicated with, never killed, never
waited on. It keeps running.**

**Reproduced**: `sleep 8.37`, timeout 1, `max_restart 1` -- **one process still
alive after the raise.**

### Why this outranks everything else found tonight

**Every other defect in this document loses information about what happened** --
a capped field, a missing marker, an unread key, a starved queue, a verdict on a
rolling log. **This one changes what happens.**

**An orphaned `ffmpeg` still writing the merge output path, alongside its
replacement, is a second writer to a file we later measure, verify and record a
verdict against -- and no record we keep would show it.** Same shape as the
`batch_state` read-modify-write that lost 33 verdicts, **except the contended
resource is the media output rather than the ledger.**

### Latent, not active, and not the truncation mechanism

**`exceeded_timeout` is false in 96 of 96 records**, and err_211's encoding pass
ran **114 s against a 3600 s timeout.** **So it has not fired in this corpus and
is not attached to the stop question.**

### And the corpus cannot calibrate against a failure

    exit_code 0        96 of 96
    failure None       96 of 96
    exceeded_timeout   false 96 of 96

The exception path exists -- `mergeVideo.py:3322` and `:3387` record with
`exit_code None` and `b""` -- but **no record carries `exit_code None`, so it has
never fired. The corpus is 96 successes by construction, with no failing run
anywhere in it to calibrate against.**

### The encoder's ENTIRE stderr is now captured, and it is silent

**Third reproduction, then the record.** Checked in that order, because silence
from a run that did not truncate proves nothing:

    441.45 441.45 441.45 441.47 441.47 441.48 | 1441.0 1441.0 1441.94
    6 of 9 streams short by ~1000 s, container 1441.94 s, MERGED_IN_SYNC 17.7

**The marker is deployed and firing**, stating its own arithmetic at the front
of the field:

    rec 1  first_pass    2.962s   4063 chars  [TRUNCATED: 4000 of 8295 kept, 1022 noise lines dropped]
    rec 3  first_pass    8.593s   4062 chars  [TRUNCATED: 4000 of 7009 kept,  589 noise lines dropped]
    rec 2  second_pass   0.818s   1205 chars  MARKER: NONE
    rec 4  second_pass 110.571s   1698 chars  MARKER: NONE

**Record 4 -- the 110-second invocation that truncated -- begins:**

    ffmpeg version 9.0.1 Copyright (c) 2000-2026 the FFmpeg developers

**That is the first line ffmpeg ever emits. No marker, 18 lines. THE FIELD IS
THE COMPLETE OUTPUT OF THE RUN** -- parse phase, encode and completion, entire.
The previous record began `": 00:23:55.687000000"`, a severed `DURATION` tag.

**Its full contents:**

     1-9   version banner and library versions
    10-16  [aost#0:N] Multiple -codec options specified for stream N,
           only the last option '-codec:a:N flac' will be used.   (seven, streams 0-6)
    17     Press [q] to stop, [?] for help
    18     [out#0/matroska] ... muxing overhead: 0.506978%

**Lines carrying ffmpeg's diagnostic vocabulary -- error, invalid, corrupt,
non-monotonic, discarding, out of order: ZERO.**

**So six audio streams ended a thousand seconds early and ffmpeg's entire stderr
for the invocation that produced them contains no error, no warning about
duration, and no mention of any stream ending.**

### Checked: the filter could not have removed the answer

**Two keep-guards existed when these records were written, and they were not the
same list** -- one omitted **`non monoton`** and **`out of order`**, precisely
the timestamp diagnostics this investigation turns on. **Whichever ran first
would have decided the vocabulary silently.** Now folded into one module-level
constant holding the union, with a test.

**That was a latent risk, not an actual loss, and the reason is structural.**
The filter drops only what it RECOGNISES as noise -- lines beginning `size=`,
`frame=`, `sed=`, `peed=`, and the indented stream listing. **Anything
unrecognised is kept, so a column-zero diagnostic survives regardless of the
guard.** Verified directly against the shipped function:

    [out#0/matroska] Non-monotonic DTS in output stream 0:3   -> SURVIVES
    size=  100kB time=00:00:01.00 speed=1x                    -> dropped

**An allow-list of noise fails safe; a deny-list of diagnostics would not
have.** The silence in record 4 is the encoder's, not the filter's.

**The finding is now complete rather than positional.** The encode-region
argument rested on the field being a suffix that happened to cover the encode;
**it now rests on the field being the whole thing. There is nothing outside the
window to appeal to, because there is no window.**

**And it is not empty, which is stronger: 1698 characters of real content, all
of it benign.**

### The only warning ffmpeg emits RESTATES the population; it does not explain it

Seven streams draw *"Multiple -codec options specified ... only the last option
'-codec:a:N flac' will be used"* -- **the only thing ffmpeg complains about in
the whole run.**

**There is no duplicated `-c:a:N`.** All twelve recorded argv rows carry **zero
duplicated targets**. What ffmpeg counts is the **global `-c copy` together with
the per-stream `-c:a:N flac`** -- two `-codec` options applying to one stream.
**The ordinary idiom: set a default, override per stream.** The discarded option
is `copy` and the kept one is `flac`, **which is the intended one**; nothing
meant to apply is lost.

**And the warned set is the six truncated tracks PLUS ONE -- but it still cannot
be the mechanism:**

    re-encoded to flac   7 tracks   6 truncated, 1 full
    copied as aac        2 tracks   0 truncated, 2 full

**`jpn` survived at exactly 1441.942 -- the `-t` -- and drew the IDENTICAL
warning in the same words.** So the warning is present on the track that
survived and on the six that did not. **It cannot separate them, and separating
them is the entire question.**

**What it marks is "was given an explicit codec", which is the same set as "was
re-encoded"** -- and truncation being confined to re-encoded tracks was already
established. **It restates the population.**

**Stated as a qualified relation rather than a bare "no", deliberately**:
"unrelated" would be wrong and would invite someone to re-derive the relation
later and think they had found something.

**Structural fact kept**: truncation is confined to the re-encoded set and the
two copied tracks are untouched, consistent with every propagated short track in
the corpus being flac.

### ffmpeg's output stream numbering is NOT the final file's

**A first pass compared warned-vs-truncated BY INDEX and produced sets that
crossed in both directions -- a clean-looking "no" that would have agreed with
the Lead's stated expectation.**

**It was unsound and the evidence was in its own output**: `a:0` and `a:4` are
**aac in the final file despite carrying `-c:a:0 flac`.** The published artefact
goes through a later **mkvmerge step that reorders streams**.

**Redone by CODEC** -- which is what those options actually determined -- the
sets do not cross. **The index reading would have handed over a false negative
dressed as a clean one**, and its author's note is the rule: *"it would have
agreed with your expectation, which is exactly when I should check hardest."*

### Correction: the control denominator is ZERO, not 4 and not 96

**The 92 do not have the key AT ALL:**

    KEY PRESENT 8   KEY MISSING 92   PRESENT AND EMPTY 0
    last record without it 20:23:53Z, first with it 22:55:46Z -- a clean boundary

**An earlier revision said 92 were empty and inferred the filter had consumed
everything.** That came from `.get(key, "")`, **which returns `""` for absence
exactly as for emptiness** -- and the inference would have justified weakening a
filter that was working correctly.

**The honest control denominator is ZERO.** Every record that has ever carried
this field is err_211's; **the instrument has never been pointed at another
file.** Any sentence of the form *"N records show no error lines"* should be
**retired rather than re-based.**

**So "quiet across 96 records" was 4 observations and 92 blanks.** A blank field
and a quiet encoder are not the same thing. **What the 92 mean is not claimed** --
filter behaviour and genuinely progress-only output are not separable from the
record alone.

**And the two fields are not separately buffered**: `mergeVideo.py:3032-3035`
derives both from **one decoded blob** -- `text = decoded[-4000:]`,
`diagnostics = _stderr_diagnostics(decoded)`. **One buffer, two windows.**

## 35. The owner's table now says which verified files are incomplete

**A completeness column, beside the sync verdict and never inside it:**

    rows in the table                  315
    rows gaining a real assessment     110   (every MERGED row)
      complete                         103
      INCOMPLETE                         7
    not assessed                       205   (never merged, no artefact)

**All 7 incomplete are recorded as a PASSING Step 2, and 4 carry a sync verdict
that approved them.**

    err_691   2/17 tracks, worst 4620 s   Merged, unverifiable      stale artefact
    err_174   1/5  tracks, worst 1442 s   Step 2 verified           <- 1 ms of audio
    err_160   2/4  tracks, worst 1416 s   Out of sync
    err_194   1/4  tracks, worst 1335 s   Step 2 verified
    err_45    1/5  tracks, worst 1322 s   Step 2 verified
    err_211   6/9  tracks, worst 1000 s   Step 2 verified           stale artefact
    err_112   1/3  tracks, worst  693 s   Merged, unverifiable      stale artefact

**No existing verdict moved, proven rather than asserted**: an A/B over all 315
rows with the artefact present and absent gives `id, folder, episode, cls,
step1, step2, status, detail` **identical on every row**, and `final_status` is
not passed the completeness result -- asserted from source by a test. **With no
artefact every row reads `not assessed`, never `complete`.**

**Why beside and not inside**: in err_211 the verifier **ran and passed**, at 1.3
and 17.7 ms, **because an absent track has no lag to be wrong about.** The check
can pass *precisely because* the audio is missing.

**13 rows carry `artefact_older_than_record`, including three of the seven
incomplete** -- so those three measurements may describe a previous generation
of the output. **Flagged in the table, neither silently trusted nor silently
dropped.**

### The near-miss: a fully populated scan of the wrong files

**The first scan reported 108 of 110 complete. It was measuring the MASTERS.**

Joining the output beside its subject finds **zero** files -- a loud failure. So
the resolver joined `new_files[0]` to `dirname(db_master)` and **all 110
resolved.** They resolved because **`new_files[0]` IS `basename(db_master)` on
110 of 110.** The path built was a path to the master.

**Masters are complete because they were never merged**, so the scan returned a
clean, plausible, meaningless result. **An empty scan would have been obvious; a
fully populated scan of the wrong files was not.**

**A resolver that finds a file is not a resolver that found the right file.**

**What caught it was another agent's count.** **What should have caught it was
evidence already in hand**: every artefact so resolved was **older than the
record that supposedly produced it, by a median of 198 days** -- a number
computed *before* the discrepancy existed and not looked at.

### Two counters that would have reported clean forever

**`not assessed` was counted as `not`** -- the counter split the **rendered
cell** on a space. **A category named by the renderer is not a category.**

**The concealed counter keyed on `status == "Resolved"`** -- a real
`final_status` value with a count of **zero** in this corpus. **It would have
printed "0 concealed" indefinitely while four such rows sat in the table.** It
now keys on `step2 == "pass"`, the claim actually being contradicted.

**A counter keyed to a value that never occurs is indistinguishable from a clean
result** -- the three-state collapse of invariant 148, arriving through the
**counter** rather than the vocabulary.

## 36. The `-t`/`-copyts` ceiling is REFUTED from the truncating invocation itself

**No retention was needed. The measurement was already on disk**, written by the
passes in question -- `stage_artefacts.jsonl`, records at 00:22:19Z and 00:24:10Z,
matching the ffmpeg records for the 8.593 s first pass and the **110.571 s
second pass** exactly.

**Verified directly:**

    INPUT   first_pass_out   00:22:19   7 audio   start_time 0.000000 on ALL
            ends  24:01.947 x6,  24:01.959 (jpn)

    OUTPUT  second_pass_out  00:24:10   7 audio   start_time 0.000000 on ALL
            ends  07:21.448 x3, 07:21.473 x2, 07:21.479,  24:01.942 (jpn)

**The mechanism requires the truncated streams' timestamps to begin near 1000.49
s, so that `limit - start` yields 441.45. They begin at 0.000.**

**With start 0.000 and `-t 1441.942`, all seven should have survived. `jpn` did,
at exactly the `-t` value to the millisecond** -- so `-t` is behaving as
documented. **Six others were cut from identical start times. Same file, same
`-t`, same start, seven streams, two outcomes.**

**This was the hypothesis's own pre-registered branch, and the one its author
said would be a real result rather than a null.**

### Two things it settles as a by-product

**The truncation occurs INSIDE the ffmpeg second pass** -- input whole, output
cut. **Not mkvmerge, not the first pass.** That confirms **by artefact** the
exclusion already written at `mergeVideo.py:5316` (no `--split`, no `--append`,
no duration limit, no value resembling 441 across 219 arguments).

**It is not a uniform cut**: 441.448 x3, 441.473 x2, 441.479 -- a **31 ms
spread.** **A single hard limit applied to seven streams does not scatter.**

### And the data was flagged in the source, four lines from the flag

`mergeVideo.py:3423-3428`, beside the retention call:

> *"NO SECOND PROBE HERE. `first_pass_out` is already recorded by
> `record_stage_artefact` the moment the first pass writes it, and that record
> ALREADY carries per-track `start_time` -- which is the measurement this line
> was going to add."*

**Someone declined to add a redundant probe and wrote down that the data was
already there. That note sits four lines above the flag we were about to switch
on to obtain it.**

**And it satisfies the hardest bad-test condition by construction**: the
intermediate, the argv and the stderr all come from the **same invocation**,
because the record was written **by** that pass. **A retained copy would have
come from a LATER run and reintroduced exactly the drift that condition
warned about** -- the ends already differ run to run (441.448-441.486).

**No mechanism is proposed.** There are measurements and no surviving candidate,
**and inventing one now is how the `-t` ceiling was adopted in the first place.**

## 37. THE TRUNCATION IS NON-DETERMINISTIC -- same input, same command, three answers

**The single most eliminating result of this campaign.** Three second-pass
invocations over **byte-identical inputs** -- same seven languages in order,
same seven durations, same container, every `start_time` 0.000000 -- verified
directly from `stage_artefacts.jsonl`:

    track   run A 09-01 08:05   run B 09-01 22:57   run C 09-02 00:24
    eng          441.485             441.479             441.479
    spa          441.485             441.473             441.473
    spa          441.479             441.479             441.448
    ita          441.473             441.485             441.473
    jpn         1441.942            1441.942            1441.942
    por          441.479             441.485             441.448
    tha          441.485             441.485             441.448

**All six truncated tracks vary between runs. `jpn` never does -- 1441.942 every
time, the `-t` value to the millisecond.**

### What this eliminates, all at once

**Anything deterministic in the inputs or the command.** Packet and codec-frame
boundaries, the input durations, the start timestamps, `-t`, `-copyts`,
mkvmerge, and the first pass. **A deterministic function of identical inputs
cannot return three different answers.**

**`-t` works exactly as documented. It is simply not what stops the other six.**

**And it confirms the stopped-not-cut reading with a property rather than an
inference: they are not being cut at a limit, they are STOPPING, at a point that
moves.**

### THE CONTROL: the master's own second pass is bit-stable in the same runs

**Verified from the same file.** The master's second pass, same three runs, same
box, minutes from the candidate's:

    08:03:50   ger 1441.959   eng 1440.880   fre 1441.959   jpn 1441.959
    22:55:48   ger 1441.959   eng 1440.880   fre 1441.959   jpn 1441.959
    00:22:14   ger 1441.959   eng 1440.880   fre 1441.959   jpn 1441.959

**Identical to the millisecond, all THREE runs. 12 track-instances, zero
variation, zero truncation.** The run-A row was missing from an earlier revision
while the text beside it said "three runs" -- the record exists and matches, so
the control is one run stronger than it was written, not weaker.

**So the non-determinism is NOT "the second pass is flaky."** The same code, in
the same three runs, produced a **bit-stable** result on one file and a **moving**
one on the other, **minutes apart.** **Whatever moves is a property of THAT
invocation** -- its input, its width, or its timing -- **not of the stage.**

### And the start-offset ceiling fails again, with the sign reversed

    master     ger/fre/jpn start 0.959000, eng 0.048000   NONE truncated
    candidate  all seven   start 0.000000                 SIX truncated

**The tracks with a NON-ZERO start are the ones that survived.** The ceiling
predicts the opposite -- a stream beginning at 0.959 s should lose 0.959 s. **The
master's tracks move 1441.960 -> 1441.959: one millisecond.**

**A second, independent refutation with the sign reversed**, in the same file as
the first. **Caveated**: the master pass carries its own `-t`, not read from the
argv, so this is not a controlled comparison of the limit. **What it establishes
is that a non-zero start did not cost this pipeline the offset** -- the behaviour
the mechanism requires.

### A NON-finding, recorded so it is not later cited as corroboration

**The master's `eng` track reads 1440.880 -- 1.08 s shorter than its three
siblings -- deterministically, in all three runs, in both passes.** It is a
**pre-existing property of the master's own track**, present before either pass.
**It is not a stop and it is not related.** A 1.08 s shortfall on a track named
`eng` in this file is exactly the shape someone cites in a week.

### MORE THAN ONE stop event -- but not necessarily six

**Answered from the three runs already on disk, with no new measurement.**
Verified directly:

    run A  [441.485, 441.485, 441.479, 441.473, 441.479, 441.485]  3 distinct
    run B  [441.479, 441.473, 441.479, 441.485, 441.485, 441.485]  3 distinct
    run C  [441.479, 441.473, 441.448, 441.473, 441.448, 441.448]  3 distinct

**A single stop event would give all six the same endpoint. They land on THREE,
in every run, three times independently.** Within-run spread 12 / 12 / 31 ms.

**And they do not move together.** Per-track deltas A -> B:

    [-0.006, -0.012, 0.000, +0.012, +0.006, 0.000]

**Two down, two up, two unchanged.** Lockstep would move them all one way. **The
movement is per-track, not per-invocation.**

**The NEGATIVE half only.** Six tracks landing on three endpoints means **between
3 and 6 distinct stop times** -- three if tracks genuinely share a stop, six if
they merely coincide at 1 ms resolution. **"More than one" is established;
"six" is not**, and an earlier revision of this section said six. **Two agents
corrected their own framing to this before it was challenged.**

### The spacing is NOT a container artefact -- checked, and it cost nothing

**If the muxer rounded durations to a grid, the ~6 ms spacing would be an
artefact and the question void.** Over **all 631 recorded track durations**:

    mod 2: 2 distinct residues    mod 3: 3 distinct    mod 6: 6 distinct

**Every residue is occupied, so the container does not round to a 6 ms grid.**
The spacing among the truncated endpoints is real. **A negative result that makes
the shape question a real question rather than a confusion** -- and the fifth
time in this session the answer was already on disk.

### Recorded and explicitly NOT built on

**Runs A and B draw from the same three values `{441.473, 441.479, 441.485}` and
only reassign them between tracks. Run C draws from `{441.448, 441.473,
441.479}` -- the same spacing, shifted one step down.**

**Either that is structure or it is two samples of a small discrete range, and
n=3 cannot separate them.** It is **the most inviting thing in the data and the
least supported** -- the same temptation as the ~6.25 ms grid inferred from four
values, flagged by a second agent for the same reason.

### The LOCALISATION generalises -- err_160 shows the same shape (one run, so it
says nothing about non-determinism), and it has no survivor

**Two of six remaining incomplete outputs have usable stage records.** The other
four -- err_691, err_174, err_194, err_45 -- have **none**, which is *"never
recorded"* and a **floor, not a null.**

### The four are PERMANENTLY irrecoverable -- a third category, not a gap

    err_691   batch_state row ONLY   output 2026-09-01T02:35Z
    err_174   batch_state row ONLY   output 2026-08-30T11:18Z
    err_194   batch_state row ONLY   output 2026-09-01T04:05Z
    err_45    batch_state row ONLY   output 2026-08-31T09:34Z

**Zero records in every source**, checked on both the durable master `file_id`
and the candidate's: `entry_points` 51, `ffmpeg_argv` 112, `ffmpeg_results` 100,
`first_pass_limits` 58, `stage_artefacts` 120 -- **0 hits each.**

**All four predate every recorder.** The earliest instrumentation anywhere is
`first_pass_limits` at **2026-09-01T06:53:13Z**; the four outputs precede it by
**4h 18m, 2h 48m, 21h and 2 days.**

**So this is the third category and nothing reaches it**: not *unresolved
subject*, which a resolver fix recovers; not *never recorded*, which a future
sweep recovers. **The runs are over and were never observed.** A re-merge
produces **a new run, not evidence about the old one** -- and for a defect
established to be non-deterministic, **that is a different subject.**

**Establishable for all four**: the output ends short of a **reconstructed**
limit, since reconstruction is bounded by the largest recorded delay (4.42 s)
and all four shortfalls are 693.4-4619.8 s.

**Establishable for NONE**: *input whole, output cut at the second pass.* **That
needs the first pass's output, which exists in no record, no retained artefact
and no reconstruction. The intermediate is the thing never observed, and it is
exactly what the shape is about.**

**err_691 is a permanent limit** -- the largest damage in the corpus at 4619.8 s,
nothing established beyond the shortfall, **no master-side record to anchor the
temporal-adjacency method that resolved err_160.**

**Noticed, not pursued**: err_194's ledger row is dated **2026-08-31T18:22:06Z
against an output written 2026-09-01T04:05:15Z** -- **the row predates the
artefact it names by 9h 43m.** Filesystem mtimes were used above for that reason.


**err_160, verified directly:**

    09:54:56  first_pass_out    fre 25:22.680   jpn 25:22.680      BOTH WHOLE
    09:55:30  second_pass_out   fre 01:48.009   jpn 01:48.012      BOTH CUT
    09:54:46 / 09:55:51         the MASTER's three tracks 25:22.681 in BOTH passes

**Whole in, cut out, inside the second pass** -- different file, different
languages, different magnitude (**1414.7 s lost against err_211's 1000.5 s**).
**The localisation holds on both cases where it can be tested.**

### And the survivor is NOT a language property

    err_211   7 candidate tracks in, 6 cut, jpn SURVIVES at the -t value
    err_160   2 candidate tracks in, 2 cut, NOTHING survives

**err_160's second-pass container reads 1436.617 s and neither track comes near
it** -- both stop at 108 s. **The `-t` was applied and no track reached it.**

**err_211's survivor is `jpn`; err_160's `jpn` was cut.** So **whatever spares a
track, it is not which language it is** -- the most obvious remaining hypothesis,
removed without a run.

**And it is not which SOURCE either.** Delivered `jpn` reads **1441.942** -- the
candidate's second-pass `-t` to the millisecond -- while the master's own tracks
have a real length of **1441.000** (tag 1441.959 less a 0.959 start). **So the
survivor is candidate-derived, from the SAME input as the six that stopped.**

**And err_160 alone cannot distinguish a hard limit from the stop**, since
everything was cut. **Only err_211's survivor excludes a single limit**, so one
of the two testable cases is uninformative on that question.

### The id trap fired again, in a second form

**Matching unresolved records by track-language signature is ALSO ambiguous:**

    err_174 and err_211 both:  eng,ita,jpn,por,spa,spa,tha
    err_160 and err_194 both:  fre,jpn

**The same records match two files each**, and either attribution would have
looked clean. **What disambiguates is temporal adjacency to a durably-resolved
MASTER record of the same run** -- err_211's candidate record sits 6 s after its
master's; err_160's two sit **between** its master's two.

**err_691's `eng,jpn` signature matches three record pairs and it has no master
anchor. Those are unattributable and were left unassigned.**

### What the next sweep needs, stated as a requirement

**The candidate side under a stable identifier.** The master side is already
durable **and is not where the loss happens** -- in both testable cases the
master's tracks pass through both passes untouched. **Retaining or probing the
candidate under something that survives the temp path is the whole of what is
missing.**

### Is it general? NEITHER outcome -- the limit is the instrument

**Measured in full at §39, which is self-contained. This is a pointer, not a
summary.**

**Two copies of one measurement had already drifted** -- this block called the
marginal 6 ms *"above noise"* while §39 called it *"the least supported value"*,
and reconciling two copies only leaves them to drift again. **The numbers live in
one place.**

**The answer: neither outcome.** Of 54 records only **five subjects have a repeat
run at all**, two of them vary, and the five exist **by accident rather than by
design**. **So it establishes neither a pipeline property nor that err_211 is
peculiar** -- and the limiting fact is the instrument, not the media.

### TWO INSTRUMENTATION REQUIREMENTS FOR THE NEXT SWEEP

Both fall out of this section and neither is a wish:

**1. The candidate side needs a stable identifier.** The master side is already
durable **and is not where the loss happens** -- in both testable cases the
master's tracks pass through both passes untouched. `file_id` is the md5 of a
temp path; matching by track-language signature is ambiguous (two file pairs
share a signature); only temporal adjacency to a master record disambiguates,
and **err_691 has no master anchor at all, so its records are unattributable.**

**2. Repeat runs of the same subject.** Every repeat that exists today is an
accident. **The question "does this vary" is unanswerable for 49 of 54 records**,
not because the analysis is weak but because the data was never collected.

### The class it attaches

**A race or resource condition** -- buffering, muxing queue, interleaving,
scheduling. **No specific mechanism is proposed**; three have died on evidence
already and the honest state is a class, not a candidate.

### The caveat that nearly sank it

**The three records carry DIFFERENT `file_id`s**, because `file_id` is the md5
of the subject path and the subject is a **temp path that changes between
runs.**

**Corrected and generalised: `file_id` is not unstable. It is STABLE for the
master and UNSTABLE for the candidate** -- a durable library path versus a temp
path -- **and both kinds live in one store with no field saying which a record
is.** The master's own records share one id (`f4e1d0b98d`) across runs and were
matched by key; the candidate's cannot be. **Content fingerprint and master key
are two independent routes that agreed, neither knowing the other's answer.** They are matched as one logical merge by an **eight-number input
fingerprint**, not by an identifier -- **and were nearly published as three runs
of one file before the ids were checked.**

**Recorded and NOT built on**: the four values sit on a ~6.25 ms grid, and run C
is both the fastest and the lowest-stopping. **And the A -> B per-track deltas
`[-0.006, -0.012, 0.000, +0.012, +0.006, 0.000]` are symmetric about zero, with
the magnitudes pairing** -- **which is exactly what reshuffling six tracks over
three values produces**, and is not promoted. **A four-point grid from four
values fits anything, and elapsed time is n=2.**

### And the magnitude question dissolves rather than resolving

**`1000.494`, `1000.487`, `1000.456` are three samples of a quantity that is not
constant** -- not three measurements of one that is. **They were never going to
merge, and now there is a reason.**

---

## 38. The endpoint measurement has a 1 ms floor, and it is the container's

**Established.** §36 cites all 631 recorded track durations; this establishes
their **precision**, which that section does not.

    all 631 duration_tag values      fractional digits in the format   9
                                     exact to the millisecond        631
                                     exact to the microsecond        631
    all 631 start_time values        exact to the millisecond        631

**Nanosecond format, millisecond resolution, zero exceptions.**

**The rounding is not the recorder's.** `record_stage_artefact` stores
`tags.get("DURATION")` verbatim — the string as ffprobe read it — so the floor is
the muxer's tag. Checked in the line rather than assumed, because *"my instrument
rounds"* and *"the container rounds"* have different fixes.

**Both consequences matter and they point opposite ways.** The ~6 ms endpoint
spacing in §37 is **six times the floor**, so it is real rather than an artefact
of the measurement. And **sub-millisecond structure is unobservable through this
store at any sample size** — a continuum quantised to 1 ms by the container
cannot be distinguished from a 1 ms grid. **More runs sharpen the distribution;
they cannot sharpen the ruler.** Getting below it means counting decoded samples,
which is a decode and a different experiment.

**NOT CLAIMED:** that the 6 ms spacing means anything. The four distinct
endpoints are 441.448, 441.473, 441.479, 441.485 — gaps of exactly 25.000 ms and
6.000 ms, which at 48 kHz are 1200 and 288 samples. **A FLAC-frame explanation
was tried and dropped**: 304 samples at 48 kHz is 6.333 ms against a measured
6.000, and AAC 1024-frame (21.333 ms), eac3 1536-frame (32.000 ms) and FLAC
4608-block (96.000 ms) fit no better. **The gaps are round in milliseconds and
not obviously round in samples**, which is what one would expect if the tag's own
rounding is what makes them look regular. **A pattern that is round in the units
of the instrument is suspect until measured in the units of the thing.**

---

## 39. Only five subjects have a repeat run at all — and two of the five vary

**Established.** The measured limit behind §37's question of whether the
non-determinism is general.

    second_pass_out records                       54
    groups with 2 or more records                  8
    CONFIRMED to be runs of ONE file               5
    of those, varying                              2

**Three of the eight groups are not repeat runs.** They share a key — track
count, language multiset, container duration to 0.1 s — while having **different
first-pass inputs**, so they are distinct files of the same shape. *"5 of 8 vary"*
was the answer before the inputs were checked, and it was wrong.

**Both matching routes fail, in opposite directions.** Grouping by `file_id`
finds **3** subjects, grouping by shape finds **13**, and the answer is **5** —
because candidate `file_id`s move between runs while shape over-groups. Neither
route alone is sufficient.

    tracks  n  varies  moving/total  span    stable
    7       3  YES     6/7            37 ms  jpn        <- err_211, and §37
    2       2  YES     1/2             6 ms  fre
    4       3  no      0/4              —    (all)
    9       3  no      0/9              —    (all)
    2       2  no      0/2              —    (all)

**Partial stability occurs in both varying subjects**, so it is not peculiar to
err_211 — with the caveat that the second is **n=2**.

**A citation corrected here rather than carried forward**: an earlier revision
said the 6 ms is *"the value §38 flags as the least supported in the data"*.
**§38 says no such thing.** §38 establishes the **1 ms floor**, which makes 6 ms
**six times the floor and therefore above noise** — the opposite reading. The
*"least supported"* phrase is **§37's**, and there it describes the **shifted-
triple structure**, not this value. **The caution about the 6 ms rests on n=2,
and on nothing else.**

**The stable subjects share nothing measurable.** Track counts 2, 4, 9 against
varying 2 and 7; identical run counts; and **cut magnitude fails to separate
them** — a stable subject was cut by 29 ms while a varying one was cut by 1 ms.

**NOT CLAIMED:** that the non-determinism is a pipeline property, or that it is
peculiar to one file. **Neither is established**: one clear case, one marginal,
three counterexamples. **The limiting fact is the instrument** — five subjects
have a repeat, and they exist by accident. A sweep that wants this answered
should retain repeat runs deliberately.

## 40. 80% of the corpus fails on one message, and that message names the wrong quantity

`Multiple delay found with the method 1 and in test 1` is the single largest
failure population: **170 of 199 FAILED rows**, measured with
`src/diagnostics/tools/delay_ambiguity.py` at 2026-09-02T07:05Z. **These counts
move while the sweep runs** -- 169/58 truncated when this section was started,
170/55 when it was finished, as reprocessed rows come back readable. The
*shapes* below did not move. Every
one of them is method 1, test 1 -- the raise at the end of `first_delay_test`.
**None** comes from either of the test-2 raise sites, although those sites emit
the same phrase, so a search on the phrase alone would have pooled two
different faults.

### 40.1 The condition that fires is not the condition the message describes

The raise is guarded by `len(delay_detected) != 1`. That is true in two opposite
situations: **several offsets survived the gates, and none did.** The message
then interpolates `delay_Fidelity_Values` -- the raw per-cut measurements, which
by construction contain more than one offset whenever anything is wrong -- so
both situations print as "Multiple delay found".

Replaying test 1's branch over the recorded measurements, using the merge path's
own `is_within_one_fingerprint_frame` rather than a copy of it:

| what actually fired | rows |
|---|---|
| `no_offset_survived` -- the set was **empty** | 106 |
| `several_offsets_survived` | 5 |
| `handed_to_second_method` | 4 |

**In 106 of 115 readable rows nothing was found at all.** The population that
reads as "several candidate delays, disambiguate them" is overwhelmingly
"every candidate was discarded and the set came out empty". This is
[INVARIANT 157](#invariant-157) in its plainest form: one message, two
populations, and the more numerous one is the one the wording denies.

*No change to the delay test is proposed here.* This section reports what the
records say the code did; the gates it describes are measurement code in the
merge path and their thresholds may be right.

**This is not the first look at this family.** §16 already established that part
of the multiple-delay refusal population is *the test working correctly on
material that genuinely has no single answer*, and already recorded the 220-char
truncation. What is new below is the split by *which condition fired*, the
location of the changeover, and the discovery that the residual bucket §16 warns
about is hiding a second population here too.

### 40.2 57 rows cannot be characterised at all, and that is a date, not a property

Exception text was stored under a `[:220]` rule until it was replaced (§ the
`store_exception` work). 55 rows still carry the old rule's damage and stop **mid-
dict, with no marker of any kind** -- `{` present, `}` absent, length exactly
220. Nothing in the text says it was cut.

The split is perfectly temporal and there is no overlap:

```
truncated  n=57   2026-08-31T10:52:21Z .. 2026-09-01T03:02:58Z
analysable n=113  2026-09-01T20:29:38Z .. 2026-09-02T06:40:47Z
```
(the snapshot at which the boundary was checked; 55/115 an hour later)

So truncation records **when a row was written, never what it contains**. The
count fell from 58 to 55 during the measurement itself, as the sweep rewrote
rows under the new rule -- these heal on reprocessing and need no separate
repair. Until a row is reprocessed, `record_truncated` is its own outcome and
must not be counted as "no ambiguity found".

### 40.3 The readable rows hold at least three different faults

Of 123 audio couples across the 115 readable rows (10 cuts each, without
exception):

| shape | couples | mean fidelity | median spread | reading |
|---|---|---|---|---|
| `bimodal` | 53 | 0.939 (min 0.904) | 9 frames | two confident offsets |
| `disordered` | 28 | 0.574 (max 0.590) | 3373 frames | never locked |
| `few_valued` | 24 | 0.880 (min 0.619) | 24 frames | mixed |
| `agreed` | 10 | 0.989 | 0 | disagrees only with the *other* couple |
| `drift` | 8 | 0.839 (min 0.813) | 21 frames | a one-way ramp |

**These are not degrees of one problem. They are different problems.**

**The bimodal couples do not look like noise.** **All 53 change offset exactly
once, in sample order -- zero interleaved, not one exception** -- and the
changeover clusters hard at the middle of the file: **48 of 53 at cut 4, 5 or 6
of 10**. Median jump 1375 ms, and 1000 ms is the single most common value. A
one-time, mid-file, whole-second offset change at fidelity 0.95 is **consistent
with** a real structural difference between the two files rather than a
measurement that wobbled. **It is not demonstrated, and §16 is the reason to say
so explicitly**: frame-quantised discontinuities are what content trimmed at
frame boundaries produces -- "consistent, not demonstrated" -- and two earlier
step claims in this file had to be withdrawn.

**The `scatter` bucket is not one population, and §16 predicted exactly this
mistake.** There, `scatter` was a classifier's terminal `return` -- "none of the
above" -- reported as though it said something about the files, and the claim
built on it was withdrawn: *a plateau-finder cannot report a drift as anything
else*. `scatter` here is the same kind of residual bucket, so it was split
before being described. **The separation is total:**

| | couples | mean fidelity | offsets over the ten cuts |
|---|---|---|---|
| disordered | 28 | **max 0.590** | no order |
| drift | 8 | **min 0.813** | a clean one-way ramp |

**There is not one crossing case, and the gap is 0.223 wide** -- test 1's own
0.70 fidelity bar falls inside it, without having been chosen for that. **The 28 are a measurement that never locked** --
calling them "ambiguous" overstates them, nothing was measured. **The 8 are a
drift**, and a drift is not an ambiguity at all:

```
err_26   2976 ms over 9 cuts   per-cut step 248 / 372 ms   = 2 / 3 frames
err_27   3348 ms               248 / 372 / 496 ms          = 2 / 3 / 4
err_29   2356 ms               248 / 372 ms                = 2 / 3
err_31   1984 ms               124 / 248 ms                = 1 / 2
err_33   1860 ms               124 / 248 ms                = 1 / 2
err_35   2604 ms               248 / 372 ms                = 2 / 3
err_37   2604 ms               248 / 372 ms                = 2 / 3
err_113  2728 ms               248 / 372 ms                = 2 / 3
```

Every one has sizePoint 124 ms, every per-cut step is an **integer number of
frames**, and every ramp runs one way for all ten cuts. That is a constant rate
quantised to fingerprint resolution. **These eight are NOT the declared-25.0
genuine-PAL group of §16** (err_46, 51, 59, 63, 70, 77) -- no overlap at all.

**The rate is not computable from records on disk** -- cut spacing is
`(duration-240)/11` for a file over 540 s and no id-keyed store carries these
durations -- so it was measured with a metadata-only probe of the eight pairs.
**It is NOT PAL. It is the 1000/1001 NTSC film relationship, and two independent
instruments agree.**

Every one of the eight pairs is `24000/1001` against `24/1` -- 23.976 fps against
24.000 -- with **no other fps pair among them**. From the container metadata
alone, with no audio involved, that predicts a drift of

    (1 - 23.976/24) x 1000  =  0.9990 ms per second

The audio correlation, measured independently, gives:

```
err       ms/s   deviation   one frame   within?
err_26   1.0366     0.0376      0.0432      yes
err_27   1.0109     0.0119      0.0374      yes
err_29   0.9660     0.0330      0.0508      yes
err_31   0.9711     0.0279      0.0607      yes
err_33   0.9888     0.0102      0.0659      yes
err_35   0.9875     0.0115      0.0470      yes
err_37   0.9875     0.0115      0.0470      yes
err_113  1.0037     0.0047      0.0456      yes
```

mean 0.9940 ms/s against a predicted 0.9990. **8 of 8 agree within one
fingerprint frame of quantisation**, which is the floor of the measurement --
the residual spread is the quantisation, not a disagreement.

**PAL is refuted, not merely unsupported.** 25-vs-24 predicts 40.0 ms/s and
25-vs-23.976 predicts 41.0 ms/s; the measured values are **forty times smaller**.
So this is a *second and independent* rate family, not a duplicate of §16's
declared-25.0 genuine-PAL group -- consistent with there being no overlap in
membership.

**CORRECTION -- SIX OF THE EIGHT WERE ALREADY KNOWN, AND I REPORTED THEM AS IF
THEY WERE NOT.** TASK.MD's Case 3 ruling of 2026-08-31 already records
*"Six NTSC files (err_29, 31, 33, 35, 37, 113, all 1.0010 within 0.00015% of
each other) are real, measured out of band, and invisible to every table."*
That is the same rate, the same six error ids, ruled a day before I measured
them. **New here are err_26 and err_27**, and the independent confirmation: the
owner's six were measured out of band, while these are derived from the
container headers and then corroborated by Chromaprint drift, two instruments
that were not compared before. The finding is a second route to a known result
plus two members, not a discovery -- and the reason it read as new to me is that
Case 3 lives in TASK.MD while I was working from FINDINGS and the stores.

**What this settles.** These eight are not ambiguous, not unmeasurable, and not
a refusal on bad input: they are one well-defined rate relationship between the
two files, visible in the container headers before any audio is read. Whether
anything should be *done* about them is the owner's call and no change is
proposed here.

**Five rows are a one-frame disagreement between two couples.** Each couple
agrees with itself perfectly across all ten cuts at fidelity ~0.98; they differ
from each other by exactly 125 ms, one fingerprint frame. All five are
consecutive episodes of one series. This is the `several_offsets_survived`
group, and it is the sub-population most likely to be resolution rather than
signal -- flagged, not acted on.

### 40.4 What the records cannot settle

* **Whether the bimodal changeover is real.** The measurements are consistent
  with a genuine mid-file offset change, and cannot prove it. Confirming means
  decoding at the changeover cut, which is held.
* **Why the 28 disordered couples fail.** "Different source recording" is the
  obvious hypothesis and **the records refute it as the general explanation**:
  joined on error id, `different_recording.json` calls **27 of 36** scatter rows
  `same_recording`, and whole-track fidelity is indistinguishable between the
  shapes (0.853 scatter vs 0.855 bimodal). Whatever separates them is not
  visible to a whole-track instrument.
* ~~The drift rate of the eight.~~ **ANSWERED above by metadata probe: the
  1000/1001 NTSC film relationship, 8 of 8, PAL refuted by a factor of 40.**
* **Anything about the 57 truncated rows.**

### 40.5 Two joins that silently return nothing

Both cost real time here and both look like clean negatives:

* `batch_state` is keyed by **library path**; the diagnostic stores
  (`different_recording`, `step_classify`, ...) are keyed by **error id**.
  Joining on either alone returns *zero overlap* for a pair of stores that in
  fact overlap completely -- 0 of 111, then 111 of 111.
* Extracting the interpolated dict by scanning braces to end-of-string fails on
  every library name containing `{edition-...}` or `{tmdb-...}`. It reported 59
  rows as having no measurement when all of them had one, under a bucket
  labelled "no brace" that was itself wrong.

## 41. A rate relationship explains one file in seven, and the detector sees only the largest family

Frame rates read from the container headers of every `Multiple delay found` row,
both files, metadata only, no decode. **173 rows, 173 measured, 0 unmeasurable.**

| relation between master and candidate | rows |
|---|---|
| **identical frame rates -- no rate relation at all** | **148** |
| PAL-from-film, 25 / 23.976 | 10 |
| 1000/1001 NTSC pulldown, 24 / 23.976 | 8 |
| 5/4 telecine, 29.97 / 23.976 | 5 |
| 5/4 x 1000/1001, 30.000 / 23.976 | 2 |

**148 of 173 -- 86% -- have identical frame rates**, every one of them 23.976 on
both sides. Whatever those files fail on, it is not a rate relationship, and
that count is reported as its own outcome rather than left as the remainder of a
table about rates.

**This census can see what no table in the project can.** TASK.MD records the
gap: `speed_factor_applied` is written only when `detect_pal_relation` returns a
factor, and it returns one only for PAL-shaped duration ratios, so **no table
here can count a non-PAL relation.** Reading the frame rates directly is not
bound by that. It also reads **the row** -- its key and `db_master` -- rather
than the exception text, so the 220-char truncation that blinds §40 for 55 rows
does not apply: 173 of 173 are covered, including every truncated one.

### 41.1 The detector matches exactly one family, and the evidence is direct

`pal_matched.json` contains **all ten** of the 25/23.976 rows and **none** of
the other fifteen. So:

    rate relation present and DETECTED       10
    rate relation present and NOT detected   15   <- 8 NTSC + 7 telecine
    no rate relation                        148

**Fifteen files in this population carry a real, named rate relationship that
the existing detector cannot see** -- the recorded gap, now with a number
against it. This is a measurement of the detector's coverage, not a proposal to
change it.

### 41.2 Refusing to guess is what found the fourth and fifth families

Seven rows first landed in `other`. A nearest-match classifier would have filed
them as PAL -- 1.25 against 1.0427 is the closest listed family -- and they
would never have been looked at again. They are a ratio of **exactly 5/4**,
23.976 to 29.97, which is a real relation and now two named families. **The
refusal was the instrument**; naming the hazard would not have been.

**And one family was removed before the first run for the opposite reason.**
`30/29.97` and `24/23.976` are both exactly 1000/1001, so listing them
separately would have left the second **permanently unfired while looking like
coverage** (§158). The ratio genuinely cannot separate them; only the absolute
rates can, so both are recorded on every row.

### 41.3 A refuted guess, stated because it was mine

I expected the fast relations to explain §40's 28 `disordered` couples -- PAL
drifts ~40 ms/s, so between cuts hundreds of seconds apart the offset would jump
far enough that correlation could never lock. **It is refuted.** Cross-tabulated:

```
relation                      drift  disordered  bimodal  few_valued  agreed  truncated
identical                         0          28       38          19       5         55
1000/1001 (NTSC)                  8           0        0           0       0          0
PAL-from-film 25/23.976           0           0        9           1       0          0
5/4 telecine                      0           0        4           3       0          0
```

**Every one of the 28 disordered rows has identical frame rates**, and not one
rate-related row is disordered. The two phenomena do not overlap at all.

**What the table does show is a clean split by whether the relation is
detected.** The eight NTSC rows are 8-for-8 `drift` -- an uncorrected ramp -- and
the seventeen PAL and telecine rows are 13 `bimodal` and 4 `few_valued`, with
**no drift among them**. That is consistent with the PAL ones having been
detected and corrected before test 1 ran, leaving a small residual, while the
undetected ones drift the whole way. **Consistent, not demonstrated**: it is
inferred from which rows the detector matched, and confirming it means observing
the correction actually run.

### 41.4 Which cells here have never fired

On this corpus every probe succeeded, so `probe_failed`, `no_video_stream`,
`ffprobe_missing` and `probe_timeout` have **never fired against real data**.
Their silence is not evidence they work (§158). Each is fired deliberately in
`test_64`, along with `other`, which fired once on the first run and may not
fire again now that the 5/4 family is named.

## 42. No verdict has outlived its artefact — the stores just cannot prove which artefact they judged

Measured 2026-09-02 by reading records only. No decode, no container, nothing
back-filled, and **no timestamp written onto any historical row** — a stamp
invented now would be a verdict about when we noticed wearing the clothes of a
verdict about when it was measured, which is the same defect one level up and
undetectable afterwards.

### 42.1 Which stores can date their rows, and what a reader may conclude

Detected by the SHAPE OF THE VALUE, not the name of the key. A name test is
worthless here: `outcome` contains the substring `utc`, and a first pass duly
reported `batch_state` as dated by its `outcome` field.

| store | rows | dates the row? |
|---|---|---|
| `batch_state` | 315 | `recorded_utc` on **250 of 315** — partial |
| `band_eleven_results` | 11 | `measured_utc`, 11 of 11 |
| `correction_state` | 91 | `started` — dates the CORRECTION, not the row |
| `output_provenance` | 80 | `output_mtime_utc` — dates the **artefact**, not the row |
| **33 other stores** | — | **nothing at all** |

**Two of the four apparently-dated stores do not date their rows.**
`output_mtime_utc` is the artefact's mtime and `started` is when a process
began; a reader who takes either as "when this was judged" is reading a fact
about the subject as a fact about the verdict. **Only `measured_utc` and
`recorded_utc` are row dates**, and one of those covers 79% of its store.

**What a reader may conclude from an undated store: nothing about currency.**
Not "it is current", not "it is stale". The file's mtime bounds the newest row
in it and says nothing about any individual row, because these stores are
appended to over days.

### 42.2 The population of verdicts whose artefact is gone is EMPTY

```
verdict rows across external_verdicts, no_output_verdicts, step1_verdicts   106
   artefact present at a RECORDED path                                       83
   artefact present at the path the project DERIVES for that row             23
   artefact not locatable                                                     0
rows anywhere with a recorded merged_file path that is missing                0
```

**Zero, on every route tested.** And separately: 141 ledger rows are `FAILED`
with genuinely no output anywhere — **not one of them carries a verdict in any
store.** The stores are not holding verdicts about deleted files.

#### The "artefact gone" reading was a FIELD's absence read as a FILE's absence

The reported population — ledger `FAILED`, no `merged_file`, artefact gone —
does not survive measurement. 185 ledger rows are `FAILED` with no
`merged_file` **field**, and of those **44 have an output on disk** at the path
this project derives for them. Every one of the seven ids named as gone
(err_9, 12, 46, 51, 59, 63, 70) **has a file there**.

This is the same mistake recorded earlier against `step2_only_class.md`, where
`batch_state`'s missing `merged_file` field was reported as "no output on
disk" and all 27 turned out to have published outputs. **The agent who reported
it was careful to say it could not measure in either direction, and that was
correct** — this supplies the measurement, it does not contradict the report.

**And the caution holds in the other direction too**: an artefact absent because
its re-merge failed is the pipeline working. The store's fault was never that
the file vanished.

### 42.3 The real defect: present is not the same as *the one that was judged*

Every artefact is on disk. **Nothing in any store can show that the file there
now is the file the verdict described.** No path, size, mtime or hash was
recorded at judging time, so "the artefact still exists" and "the verdict still
describes it" are different claims and only the first is checkable.

    27 rows carry an `in_sync` verdict beside a ledger outcome of FAILED
       external_verdicts   15   err_9,12,46,51,53,54,59,61,63,70,87,91,95,98,217
       no_output_verdicts  12   err_47,52,56,58,60,64,72,84,88,92,96,271

Both are true of different runs. Which run the surviving file came from is the
question no store can answer.

**`step1_verdicts` is the same defect with nothing left to hold on to**: 40
rows, **no per-row date, no artefact identity**, and every one of them says
`unverifiable`. Their ledger outcomes today are 18 `MERGED_IN_SYNC`, 19
`FAILED`, 3 `MERGED_UNVERIFIED`. A snapshot with neither a date nor a subject
identity cannot be aged or checked, only believed.

### 42.4 Would `written_utc` have prevented these? Two of three, and only by DETECTING

**It prevents nothing.** A verdict goes stale when the world moves; a date lets
a reader *notice*, which is worth having and is not the same claim.

| case | would a row date have helped? |
|---|---|
| the 27 stale verdicts | **YES, all 27** — every one of their ledger rows carries `recorded_utc`, so the comparison is possible on both sides |
| `step1_verdicts` snapshot | **PARTLY — 25 of 40.** The other 15 ledger rows have no `recorded_utc`, so there is nothing on the far side to compare against |
| the "artefact gone" 11 | **NO.** Nothing about them is a dating problem. The row lost its pointer to the artefact; a date on the verdict says nothing about which file it judged |

**So it is a partial remedy and is recorded as one.** A date on one store
half-answers: it takes a date on BOTH sides to age a verdict, and
`batch_state` covers 250 of 315. **The third case needs artefact identity, not a
timestamp** — which is exactly what landed alongside it, and which is equally
prospective: it can say nothing about the 106 rows already written.
