# Agent notes — logbook

One entry per decision that a later reader would otherwise have to re-derive.
Newest first.

**What goes here:** an architectural choice and why the alternatives lost; a
measurement that changed a decision; a defect found and what it taught; a blocked
need and what was done instead.

**What does not:** progress reports, task status, anything regenerable from the
code or from `git log`. If it is a measured fact rather than a decision, it belongs
in a findings document.

**Format.** `## YYYY-MM-DD — one line saying what changed`, then the decision, then
*why the alternatives lost*. Numbers carry the command that produced them. Files are
cited by opaque id, never by media name (`AGENT.MD` → Privacy).

Entries are **append-only**. Superseding an earlier one means adding a new entry that
says so and why — never editing the old one. A record that can be rewritten is not a
record.

---

## 2026-09-03 — measured: on one file, stage 2 is WRONG where stage 1 is right, which inverts the design assumption

`AGENT.MD` records the owner's design intent: fingerprint peaks put stage 1 off by
roughly ±250 ms, **stage 2 runs unconditionally and corrects them**, and a residual
beyond 500 ms means stage 1 was not merely imprecise but wrong. On error id 12 the
opposite is measured.

`vmsam-ci` found four files refused by stage 2's variance test whose recorded
evidence is 8-9 of 10 windows agreeing to 1.16-2.18 ms with one wild outlier. It
asked whether the outlier window is the same window that steps in stage 1. I
measured id 12 directly — absolute, unshifted, both instruments, same windows:

| h | window | `second_correlation` | chromaprint | RMS master |
| ---: | :--- | ---: | :--- | ---: |
| **0** | [120.0, 292.0] | **-567.01 ms** | 0.8555 / **-1375** | **4306.9** |
| 1 | [206.0, 378.0] | -1417.66 | 0.9236 / -1375 | 2016.9 |
| 2-9 | … | -1412 to -1428 | 0.919-0.934 / -1375 | 1398-2084 |

**Chromaprint gets window 0 right** — the same -1375 plateau as every other window,
at the lowest fidelity of the ten (0.8555), so it registers its own reduced
confidence without being wrong. **The FFT is the instrument that fails**, by about
850 ms.

The master's RMS in window 0 is **4306.9 against 1398-2084 elsewhere** — two to
three times the energy. FFT cross-correlation is dominated by the loudest passage
in its window, so a loud passage that differs between two releases (a different
music bed, a different mix) drags the peak. Chromaprint normalises per frame and is
not dragged.

**This reproduces ci's recorded residuals**, which were taken after the candidate
was shifted by `delayUse` = chromaprint's -1375:

```
windows 1-9:  -1417 - (-1375) =  -42 ms     ci recorded -56 to -57
window 0:      -567 - (-1375) = +808 ms     ci recorded +788
```

Two extractions, two agents, same numbers to within 20 ms.

**Consequence.** For that family the refusal is not evidence about the media. It is
stage 2 failing in a loud passage and overruling a stage 1 that was correct in all
ten windows. `test_if_constant_good_delay` then reads the disagreement as *stage 1
was wrong* and refuses the pair — the exact inverse of what happened.

**REFUTED FOR THE FAMILY, 2026-09-03, by `vmsam-ci`.** The RMS mechanism above
does **not** generalise. ci measured master per-window RMS on the *correct*
comparison streams for all three:

| id | window 0 / median(1-9) | |
| ---: | ---: | :--- |
| 12 | **2.49x** | spike — the mechanism holds |
| 13 | 1.53x | elevated, but two other windows are as loud and produced no outlier |
| **108** | **0.77x** | **window 0 is QUIETER than the median, and still carries the outlier** |

id 108 refutes it outright: *"loudest passage drags the peak"* cannot explain an
outlier in a window 23 % below median energy, and that file's loud windows (5-9)
are the ones that behave. So: **mechanism established for one file, symptom shared
by three, cause of the other two unknown.** ci independently reproduced my id 12
RMS figures from its own extraction at a different sample rate (4165.3 against
1330-2010, versus my 4306.9 against 1400-2084).

**What survives the refutation is the part that matters**, and it rests on id 12
alone without needing the family: stage 1 was right in all ten windows, stage 2 was
wrong in one by 850 ms, and that single value refused the file. The inversion of
`AGENT.MD`'s stated design intent is real and measured, whatever explains the other
two.

**Hedged deliberately: one file, one window.** ids 13 and 108 do not share the
cause,
and ci's own classification splits the nine stage-2 variance files three ways
(a real change point seen by both stages; a first-window-only outlier with stage 1
silent; degenerate). This entry covers the second group only, and one member of it.

Also confirmed for ci, from the code rather than assumed: **window index 0 is
chronologically first** — `generate_cut_with_begin_length` increments
`begin_in_second` by `length_time` each iteration and `get_delay_by_second_method`
appends in `h` order. But **window 0 starts at `begin_in_second`, which is 120 s**
for any file over 540 s, so logos, black frames and opening credits are *before*
every comparison window and cannot explain anything about window 0.

**Not an edit.** `second_delay_test` and `test_if_constant_good_delay` are in
`mergeVideo.py` outside every tagged zone.

### `max_delay_variance_second_method`: one constant, two units

Found by `vmsam-ci`, verified here. `mergeVideo.py:25` sets it to `0.004`, and it
is compared against two dimensionally different quantities in one function:

```
:582   variance(list_delay) < 0.004          variance of SECONDS  -> s^2
:584   abs(first - last)    < 0.004          a plain difference   -> s
:594   variance(delay_detected) > 0.004      variance again       -> s^2
```

`second_correlation` returns seconds, so at :582 and :594 the constant is an s²
budget implying **√0.004 = 63.2 ms of standard deviation**, while at :584 it is
**4 ms** flat. `AGENT.MD`'s description — *"its own variance tolerance,
`max_delay_variance_second_method` = 0.004 s"* — matches :584 only.

## 2026-09-03 — the recurring defect of this campaign: a control whose pass is guaranteed by its construction

Seven instances in one day, three of them pre-dating this campaign and sitting in
documents we treat as authority. Recorded as a pattern because it is worth more
than any of the individual defects, and because every instance was found by
**running** the control somewhere new, never by reading it.

**The shape.** A control is built to prove something works. Its construction
quietly guarantees it passes whatever the thing under test does. It then reports
success forever, and the success is a statement about the harness rather than
about the world.

**The question that catches all seven, and it takes one sentence:** *if the thing
under test were broken in the way I most fear, would this control produce a
different result?* If the answer is no, it is not a control.

| # | control | why its pass was guaranteed | whose |
| ---: | :--- | :--- | :--- |
| 1 | sparse-frame pHash, validated on a pair at **zero offset** | with any offset the sampled frames never coincide — the zero-offset case is the only one that cannot expose the flaw | `vmsam-ci` |
| 2 | edge bisection with `lo`/`hi` passed reversed | `while hi - lo > tol` exits immediately and returns its own input bounds; it printed "-600 ms", which was **the instrument not running** | dev-1 |
| 3 | clean-edge estimator requiring correlation to reach **1.00000** | only a file against a *copy of itself* reaches 1.0; real encodes cap at 0.93-0.96, so it could never fire on a real pair | dev-1 |
| 4 | repair fixture built as **one file against a copy of itself** | its base offset is zero, so a sign error in deriving the base offset is invisible — it later cost a uniform +16146 ms track | dev-2 |
| 5 | post-repair probe at the **midpoint of a single segment** | a residual coefficient error grows with position and is smallest exactly at the midpoint | dev-2 |
| 6 | speed sweep with **no wrong-direction column** | `* r` where `/ r` was meant; rubberband won 33 of 33 and *"nothing internal to the run could have caught it"* | `docs/AUDIO_SPEED_POLICY.MD` weakness 3 |
| 7 | *"a file where `asetrate` leaves fidelity worse than no correction"* | the uncorrected arm is pinned near the floor **by geometry**, so almost any correction beats it — the document names this itself | `AUDIO_SPEED_POLICY.MD` |

Campaign 1's own headline defect is the same shape one level up: `final_status`'s
success metric sat in a third branch nothing reached, and `Resolved: 0` was read as
*nothing qualifies* when it meant *nothing reached the check*.

**What actually caught them**, in every case: running the control in a context it
was not built in. #3 died on the first real error file after passing a constructed
one. #5 died when a speed-only plan produced a single segment. #4 died against a
second fixture with a non-zero base offset. **None was caught by review**, and
several had been reviewed — including by their author, in writing, after stating
the convention as an equation.

**The practical rules that fall out**, all of them paid for today:

- **A fixture built by copying one file cannot test anything that depends on the
  two files being *different*.** It has zero offset, identical content, and a
  correlation ceiling of 1.0, and all three flatter the instrument.
- **A directional correction must carry the wrong direction as its own column**,
  so a sign error reads as the floor instead of as a finding.
- **Probe where the error is largest, not where the answer is tidiest** — the ends
  of a segment, not its midpoint.
- **Assert a control's bracket before trusting its output.** #2 would have been
  caught by one `assert f(lo) <= threshold and f(hi) > threshold`.
- **"I could not measure" and "I measured X" are different answers.** #2 printed a
  number that was neither.

`AGENT.MD` already states the general form — *an armed guard that has never fired
is indistinguishable from a working one* — and every instance above is that
sentence being learned again at cost. The addition this campaign can make is that
**firing a guard is not enough: it must be fired somewhere it could fail.**

## 2026-09-03 — blocked need: a new runtime module is open by the rules and frozen by the checker

`WRITE_ZONES.MD` §4 lists **"new runtime modules"** among the open paths. The
checker disagrees, and the checker is what gates the commit.

**Measured.** One comment line appended to `src/change_point_locator.py`, a module
I created and committed myself:

```sh
$ python3 .../check_write_zones.py --repo .
WRITE_ZONES violations (staged changes):
  §1  frozen file modified: src/change_point_locator.py
CHECKER_EXIT=1
```

So a new module is **write-once in practice**: the creating commit is allowed and
every subsequent edit to it is a §1 violation. The cause is structural rather than
a bug in the logic — the checker carries a fixed allow-list of open paths, and a
file that did not exist when that list was written cannot be on it.

**Why this blocks rather than annoys.** Neither I nor `vmsam-dev-2` can fix a
defect in our own modules. `vmsam-dev-2` reports 007's and 009's implementations
sitting in a 400-plus-line patch beside the repository for this reason, with its
commits recording only what they did and measured. My locator is committed and I
cannot correct it if its first real run finds anything.

**What was done instead: nothing, deliberately.** The checker lives in
`VMSAM_HELP_AI/tools/`, outside the repository, and is the shared instrument that
enforces the rules on all four agents. Changing it to admit my own file is exactly
the case `AGENT.MD` names — being right about a rule does not make it mine to
change — and an agent that edits its own gate has no gate. Reported to the Lead
and recorded here.

**The fix is small and is the owner's or the checker author's to make:** treat a
path that is untracked at `HEAD` and added by the staged diff as open under §4,
or add an explicit open-list entry per new module.

### A second claim tested and NOT confirmed

`vmsam-dev-2` also reported that a `config.ini` `[features]` flag *"cannot ship at
all"*. **It can.** `src/config.ini` has **no trailing newline**, so any *append*
rewrites its final line and the checker correctly reports

```
§3  src/config.ini: line removed or altered: audio_sync=/home/vmsam/audio_sync
```

That is the rule catching a genuine modification of an existing line, which §3 does
forbid. **Inserting** the new section before an existing one instead leaves the
final line byte-identical and passes clean, with `git diff --cached` showing no
removed lines — verified on this clone, then reverted, since the flag belongs to
`vmsam-dev-2` under the territory rule.

Worth carrying: the missing trailing newline will catch the next person who
appends to that file.

## 2026-09-03 — measured: the speed detector's margin does not carry from PAL to NTSC

`docs/AUDIO_SPEED_POLICY.MD` is frozen and its ruling **stands**: `asetrate`, not
`rubberband`, for a naive speed transfer. This entry **extends** it by measurement
and is a finding for the owner, not an edit.

The corpus family is **NTSC (`1001/1000`), not PAL.** Measured on error ids 33 and
31, whose recorded delays are monotone decreasing with a 124 ms gcd. Regressing
delay against window centre gives `speed_ratio` = master/candidate of **1.000993**
and **1.000994**, ~95 % `[1.000953, 1.001034]`. `1001/1000` = 1.001000 lies inside
both; PAL's 1.042709 is excluded by four orders of magnitude. `1001/1000` and
`24/23.976` are the **same** relation — 23.976 rounds 24000/1001 — not two
candidates. This corroborates `vmsam-forensic`'s finding on two files, and its
hedge travels with it: 2.5 % is a **floor**, not a count, since a PAL-rate file
collapses to weak correlation and is structurally invisible to the method that
found these.

Residual scatter about the linear fit is 38.3 and 41.0 ms against **124/√12 =
35.8 ms** predicted by pure quantisation, so the residuals are almost entirely the
quantum and no staircase is hiding under the slope.

**The verdict, by the document's own three-arm method** (1200 s window, because
0.099 % drifts only 0.41 s over the pipeline's 418 s):

| arm | id 33 | id 31 |
| :--- | ---: | ---: |
| uncorrected (control) | 0.7411 | 0.7374 |
| **asetrate CORRECT** | **0.9588** | **0.9551** |
| asetrate WRONG (sign control) | 0.6569 | 0.6562 |
| rubberband | 0.9484 | 0.9453 |

The wrong-direction arm lands **0.084 below the uncorrected control** — it makes
things worse, as a doubled error must. Weakness 3 records a sweep that used `* r`
for `/ r` and had rubberband win 33 of 33 with nothing internal able to catch it;
that column now exists and the direction is demonstrated rather than asserted.

**The finding.** The document measures rubberband paying a **~0.109** handicap and
losing by 0.278-0.313 against real masters. **Here the gap is 0.0104 and 0.0098** —
an order of magnitude smaller. This is consistent with the document, which already
records that against a real master `asetrate` reaches only 0.94-0.98 *"because real
pairs carry content and codec differences no correction removes"*: both arms press
on that ceiling and the distance between them compresses. At 4.27 % the losing arm
had far to fall; at 0.099 % it does not.

So **the margin that would detect an inverting case in this corpus is ~0.010, not
~0.109**, and a detector built to the document's expected separation could miss
one. The 0.85 bar and the three-outcomes-and-a-decline design both survive; what
does not carry is the expected margin, which was taken from a 4.27 % corpus.

**Two files agreeing in direction and magnitude is evidence the margin is real. It
is not a rate and is not reported as one.**

*Why the alternative lost:* handing dev-2 the bare coefficient. The document is
explicit that a bare number is applied as `asetrate` by default and an inverting
case passes undetected — so the verdict, not the coefficient, is the deliverable.
dev-2 is given `1001/1000` as the **exact rational** rather than my point estimate:
my measurement is consistent with it and *less precise than it*, and propagating
measurement noise into a file when the true relation is a known rational would be
strictly worse.

## 2026-09-03 — measured: `first_delay_test`'s second pass measures on misaligned fingerprint grids, and manufactures one-point steps

**A one-point step in `delayFirstMethodAbort` can be an artefact of the pipeline's
own re-extraction, not a property of the media.** This supersedes nothing, but it
sharpens the 2026-09-03 entry on rebasing: I said the record gives reliable
*relative* structure and only an unreliable zero. At the one-point level the
relative structure is not reliable either.

`recreate_files_for_delay_adjuster` (`mergeVideo.py:333`) re-extracts the
candidate's windows at `begin + delayUse/1000` seconds. `delayUse` is an arbitrary
millisecond value and essentially never a whole multiple of the chromaprint hop,
`4096/3/11025` = **123.840 ms**, so the two fingerprint grids end up misaligned.

Measured on error id 318 — same pair, same windows, **only the shift changing**:

| shift | in hops | fidelity | residual |
| ---: | ---: | :--- | ---: |
| 0.000 ms | 0.000 | 0.968-0.975 | +552 |
| 495.360 ms | **4.000** | 0.967-0.974 | 0 |
| **552.000 ms** | **4.457** | **0.918-0.936** | **-138** |
| 619.200 ms | **5.000** | 0.967-0.974 | -138 |
| 743.040 ms | **6.000** | 0.967-0.974 | -276 |

Whole-hop shifts hold fidelity at 0.967-0.975; the fractional shift costs ~0.05.
**0.918-0.936 is exactly the band that file's `.log.error` records (0.906-0.923)**,
so the recorded fidelities are depressed by the instrument rather than the media.

Direct absolute measurement of four exemplars (own extraction, pipeline geometry,
`correlate` imported from `src/`, verified unchanged under `loudnorm`):

| id | absolute | fid | recorded |
| ---: | :--- | :--- | :--- |
| 318 | **552 constant** | 0.968-0.975 | `[0, 138 x9]` |
| 319 | **552 constant** | 0.960-0.976 | `[138 x7, 0, 0, 138]` |
| 271 | `[-250 x5, -500 x5]` real step | 0.652-0.726 | `[-250 x4, -375 x6]` |
| 316 | `6250 x9` + one `6125` | 0.582-0.683 | `[6125 x7, 6250 x3]` |

**Measured:** the offsets above, and the shift/fidelity table.
**Inferred:** that the recorded steps in 318 and 319 are produced by the
degradation. Well supported, not demonstrated — I have not driven the pipeline to
produce the artefact on demand.
**Not established:** any rate. **Four files, not a proportion.**

**Decision for `TASKS/006`: `locate_change_points` re-measures absolute offsets
itself, unshifted.** It does not derive segments from `delayFirstMethodAbort`.
`SPEC_ZONE_A.MD` §1's *"you do not need to re-run the correlation that just
failed"* holds for knowing **that** a pair failed and roughly where, and not for
the numbers the repair needs. Ten correlations per pair is the price.

*Why the alternative lost:* reading the record is free and would have produced a
chimeric splice for two files that need a single `--sync` of 552 ms.

**Blocked need, unchanged and now with a mechanism:** all of this is in
`first_delay_test`, outside every tagged zone. Recorded, not edited.

## 2026-09-03 — measured on 315 recorded refusals: the quantum model predicts 297 of 299, and three files are refused by a constant

Corpus-wide confirmation of the quantum entry below, from **production data**
rather than a constructed sweep, and at no run cost. `vmsam-forensic` found and
`vmsam-ci` extracted a `.log.error` beside every error file, each carrying the
per-window `(fidelity, offset_points, delay_ms)` triples.

`delay_ms = -offset_points x sizePoint` (`audioCorrelation.py:118`), so the quantum
is recoverable as `-delay_ms/offset_points` wherever the offset is non-zero:

| quantum | rows |
| ---: | ---: |
| 125 ms | 269 |
| 124 ms | 27 |
| 138 ms | 3 |
| self-inconsistent | 2 |

Campaign 1 reported *"859 of 933 at 125 ms … three files in this library have a
138 ms frame"*. **The three 138 ms files reappear here**, and they are 190.0 s of
audio, which is why: `L = 2*int((190-45)/11) = 26`.

Predicting the quantum from audio duration alone, using
`n = L/0.1238397 - 21.6` and `int(L/n*1000)`, matches **297 of 299** rows; both
misses are off by exactly 1 ms at a band edge, where the fitted constant rounds the
other way. The mechanism is established, not merely the correlation.

**The two "self-inconsistent" rows are not inconsistent — they are per-track-pair.**
Their logs carry 30 and 50 triples, i.e. 3 and 5 audio *track pairs* x 10 windows,
and the quantum differs **between track pairs of the same file** (125 vs 126). So
campaign 1's `DOMAIN_INVARIANTS.MD` §7 *"per-pair, not a constant"* is right and
one level too coarse: it is per **call**, and two audio tracks of one file can
carry different quanta.

**Three real files are refused by the hardcoded constant.** ids 318, 319, 320
(folder 181) record delay sets differing by **138**. Every reconciliation gate in
`first_delay_test` compares against `127` or `128` (`mergeVideo.py:270, 316, 345,
401, 416, 445, 496`), and `mergeVideo.py:496` tests each delay's magnitude against
128 rather than their difference — 138 exceeds all of them, so **no gate can
fire**. The log confirms the branch: *"Multiple delay found with the method 1 and
in test 3"*. At a 125 ms quantum the identical one-point difference reads `{0,125}`
and passes. **Carry the hedge:** the right outcome for these three is not
necessarily *merge* — their absolute offset is a constant 552 ms but their duration
delta is 1.030 s, so something differs outside the window coverage. What is
established is that the gate decides on a threshold that is a function of episode
duration rather than of the media.

Still a finding, not an edit: outside every tagged zone.

## 2026-09-03 — measured: the chromaprint point is 124–142 ms, not 125, and ten gates hardcode 125

`BRIEF_COMMON.md` and `TASKS/006`/`008` state *"One point = 125 ms, the chromaprint
frame. Every delay you see is a multiple of it."* **Measured, that is true for a
band of episode lengths and false outside it.** This changed the `TASKS/005`
ruling: it is why the video instrument must return a frame displacement rather
than a refined time, so it is recorded here and not only as a number.

`audioCorrelation.py:132` computes the quantum **per call**,
`int(lengthFile/len(fingerprint_source)*1000)`. It is not the chromaprint frame,
which is `(4096/3)/11025` = 123.84 ms; `n_items` falls short of
`lengthFile/0.12384` by a near-constant ~20 items, which lifts the ratio above the
frame by an amount depending on `lengthFile`.

```sh
# window from the ore1 example pair; items counted from fpcalc, int(L/n*1000)
ffmpeg -v error -y -ss 60 -t 700 -i <one file of the pair> -vn -ac 2 -ar 44100 \
       -acodec pcm_s16le w7.wav
for L in 20 24 40 60 100 160 214 288 436 700; do
  n=$(fpcalc -raw -length $L w7.wav | sed -n 's/^FINGERPRINT=//p' | tr ',' '\n' | wc -l)
  python3 -c "print($L, $n, int($L/$n*1000))"
done
```

| `lengthFile` L (s) | 20 | 24 | 40 | 60 | 100 | 160 | 214 | 288 | 436 | 700 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quantum (ms) | 142 | 139 | 132 | 129 | 127 | **125** | **125** | 124 | 124 | 124 |

Sweeping L in 4 s steps: **125 ms exactly for L ∈ [160, 286]**, 126+ below, 124 at
and above 288. The pipeline passes `lengthFile = lenghtTime*2` with
`lenghtTime = int((shortest_audio_duration−240)/11)` above 540 s, so 125 ms holds
for **shortest audio duration ∈ [1120 s, 1824 s)** — about 18.7 to 30.4 minutes.
Our example corpus is 1420 s, inside the band. A 45-minute episode gives L = 436
and **124 ms**; a short file on the `> 60 s` branch reads 132–142.

**Campaign 1 found this first and I am carrying their hedge, not hardening it.**
`CAMPAIGNS/01-dev-AI/DOMAIN_INVARIANTS.MD` §7: *"~125 ms typically, **138 ms
observed**"*. `WORKLOG.MD`: *"917 of 933 delay vectors have a self-consistent
ms-per-frame, **859 of them at 125 ms** … three files in this library have a 138 ms
frame."* My sweep gives 139 ms at L = 24 s, which is the mechanism behind their
138 and is consistent with it. What is new here is the **exact band** and that it
is a function of episode duration.

**The blocked need.** `first_delay_test` compares millisecond values against a
hardcoded `127`/`128` at ten sites — `mergeVideo.py:270, 316, 345, 401, 416, 445,
496, 956, 961, 969`. Those constants encode *"within one fingerprint frame"* only
when the frame is 125 ms. At 124 ms the gate is harmlessly loose; at 132–142 ms
(short files) an ordinary adjacent-frame disagreement measures more than the
constant and is misread as a real timing conflict, so the pair is discarded before
stage 2 — the only stage that resolves below fingerprint resolution.

Campaign 1 diagnosed exactly this and wrote the fix — `is_within_one_fingerprint_frame()`
and `are_residuals_within_one_frame()`, comparing frame indices directly, which is
exact for every quantum. **Neither exists in this tree**
(`grep -rn is_within_one_fingerprint_frame src/` → nothing). It is one of the
pieces that did not survive the campaign.

**What was done instead: nothing, deliberately.** `first_delay_test` is in
`mergeVideo.py` **outside every tagged zone**, so by `WRITE_ZONES.MD` rule 1 it is
frozen and this is a finding (§6), not an edit. Nothing in objectives 2 and 3 is
blocked by it: we consume the refusal, we do not need to prevent it. It becomes
blocking the day someone wants fewer false refusals rather than better repairs of
real ones.

## 2026-09-03 — `da1c191` stripped every `[0]` subscript from two files; `get_cut_time.py` retired

`WRITE_ZONES.MD` §4, `FILE_MAP.MD` and `TASKS/005` all describe the damage as a
list literal lost from three initialisers of `frame_compare.py`. **Measured, the
corruption is larger and of a different kind:**

```sh
$ for f in src/*.py; do echo "$(grep -o '\[0\]' $f | wc -l)  $(wc -l < $f)  $f"; done
0   220   src/frame_compare.py
0   564   src/get_cut_time.py
107  2118  src/mergeVideo.py
60   1144  src/video.py
```

`[0]` occurs **zero** times in both dormant files, while `[1]` and `[2]` survive
normally inside them (ten occurrences in `get_cut_time.py`) and comparable live
files carry many. `da1c191` touched these two files and nothing else. Three sites
are syntax errors; **eight more compile and raise at runtime**, in
`_detect_time_drift` (the drift regression) and `_summarize_segments` (the
per-window fidelity summary) — the two functions objectives 006 and 008 would have
been built on. They have never run.

**Decision (`TASKS/005`): repair the pHash front end, reject the alignment back
end, retire `get_cut_time.py`.**

*Why repairing the whole of `frame_compare` lost.* `_align_and_find_gap`
aggregates `costs[best_j] += best_d` — cost lands only on target frames that some
reference frame chose as its best match. Content **inserted** in the target is
chosen by nothing, so it accumulates cost 0, and the function returns the argmax
of the cost array. An insertion is precisely a run of `hits == 0`, and `hits` is
computed and never read (`grep -n hits src/frame_compare.py` → 110, 126 only). The
array carrying the signal is built and discarded. It also truncates to the first
`max_search_frames` — 30 % of the window as its caller configures it — and still
answers confidently, and it returns a region rather than a signed displacement,
which fits neither consumer. Restoring `[0]` fixes none of that.

*Why adopting `get_cut_time.py` lost.* `second_correlation` returns
`(file, offset_in_seconds)`; all four callers there divide that offset by the frame
rate — wrong by ~24× — under a comment asserting a project convention the project
does not use. The correct usage is in the tree:
`mergeVideo.get_delay_second_method_thread.run` uses `result[1]` directly and reads
`result[0]`, the filename, to recover **the sign**. `get_cut_time` ignores
`result[0]` everywhere, so it cannot know the sign of anything it measures.

*Why its eight `[0]` sites were deliberately left broken.* Repairing
`frame_compare` makes `import get_cut_time` succeed where it used to fail at import
time, so the repair makes the retired file **more** likely to be picked up. A file
that raises is safer than one that runs and is wrong by a factor of 24 with no
sign. It carries a RETIRED banner naming the four defect classes instead. Deletion
is an owner decision, not ours.

*Consequence for the frozen documents.* `python3 -m compileall -q src -x '__pycache__'`
now exits **0**. `AGENT.MD` § Testing (*"must fail on exactly one file"*) and
`WRITE_ZONES.MD` §5 check 4 both still expect one failure. That was `TASKS/005`'s
stated acceptance condition, so the change is intended — and reported here rather
than edited into either document.

## 2026-09-03 — finding: `delayFirstMethodAbort` is not uniformly rebased, so its zero cannot be trusted

`SPEC_ZONE_A.MD` §1 offers `video_obj.delayFirstMethodAbort[best_video.filePath]`
as the reason *"you do not need to re-run the correlation that just failed."* That
holds for the **structure** of the delays and not for their **absolute value**.

`first_delay_test` shifts the candidate mid-function —
`recreate_files_for_delay_adjuster(delayUse)`, `mergeVideo.py:333` — so every later
measurement is a residual against an already-shifted candidate.
`get_delays_dict(values, delayUse)` exists precisely to add `delayUse` back. Five
of the seven recording sites do so correctly: `:319` with 0 (before any shift),
`:391`, `:409`, `:478` with `delayUse`. **Two do not** — `:489` and `:502` pass
`delayUse=0` explicitly, after the shift.

So at two of seven sites the recorded delays are short by `delayUse`, and nothing
in the recorded `[1|2, delays]` structure says which site produced them. In the
staircase case `delayUse` is typically 125 ms — one point, the same size as the
step being located.

**What was done instead.** `first_delay_test` is outside every tagged zone, so this
is a finding, not an edit. Objectives 2 and 3 proceed by treating the abort record
as reliable **relative** structure — where the steps are and how large each step is
— and re-establishing the absolute zero independently on one window. That costs one
correlation, not twelve, so §1's promise is mostly preserved.

**Two smaller corrections to frozen documents, recorded for the owner:**

- **Windows are 10, not 12.** `video.number_cut = 10` (`video.py:21`);
  `generate_cut_with_begin_length` loops `range(0, number_cut)` and
  `get_delay_fidelity` loops `range(0, video.number_cut)`. `BRIEF_COMMON.md`,
  `dev-1/BRIEF.md` and `TASKS/006` all say the record is *"filled at 12 points"*.
  The rest of the geometry in those documents is correct: window length is twice
  the spacing (`length_time_converted = gmtime(length_time*2)`,
  `mergeVideo.py:756`), and coverage runs 120 s to duration−120 s, ~83 %.
- **`SPEC_ZONE_A.MD` §1 cites `mergeVideo.py:1426`** for
  `delay_same_md5_audio + delay_to_put`. In this tree it is **`:1446`**. The
  requirement it supports — that the field must be a `Decimal` — is unaffected and
  correct.

## 2026-09-03 — `check_write_zones.py`: the rules get an instrument, and it lives outside the repository

`WRITE_ZONES.MD` §2 rule 2 says a zone is real only if it is checked, and lists
four shell snippets. Three of them are read by eye. Campaign 1 pushed a commit
with a failing test by gating on the wrong thing and reading a suite's output
instead of its exit status (`CAMPAIGNS/01-dev-AI/RESTART.MD` §8), so the checks
are now one command with one exit status.

**Where it lives: beside the repository, not in it** —
`/home/vmsam/src/VMSAM_HELP_AI/tools/check_write_zones.py`. §4's test is *would
this still matter to someone running VMSAM in production?* A harness that reads
`git diff` and enforces a campaign's write rules would not. Campaign 1 left 148 773
lines of exactly this kind of scaffolding under `src/`, nineteen times the program.
One copy outside, called by absolute path from every clone, also means the four
agents cannot drift onto four versions of the rule.

**Why the naive form of the zone check is wrong.** `git diff --cached -U0` read by
eye compares changed lines against the zones of the *new* file. A pure deletion has
no new-side line at all, so a `-` hunk removing a frozen line above a zone reads as
"nothing added outside a zone" and passes. The checker maps removals onto the *old*
blob's zones and additions onto the *new* blob's, separately. The self-test case
`outside zone: pure deletion below the zone` is that defect, pinned.

**Two more holes the eye version leaves open**, both now refused: moving an `END`
marker down to swallow a line into a zone (the diff lands inside the new zone and
looks legal), and creating a fresh marker pair — §2 says adding a zone is an owner
decision, so any change in marker count fails.

**It answers three things, not two.** Exit 2 means the checker could not run and is
*not* a pass. `RESTART.MD` §3.4 found the same shape five times in one morning: a
reader with two outcomes describing a world with three, and the missing outcome is
always *the instrument did not run*.

**Fired, both answers, before being believed.** `--self-test` builds throwaway
repositories and drives 27 cases; each refusal has a matching case that must be
accepted. On the real tree at `f0e7119`: clean; a one-line insert at
`mergeVideo.py:1` refused; the same insert at `mergeVideo.py:795`, inside zone A,
accepted. An armed guard that has never fired is indistinguishable from a working
one, and that includes this one.

**What it deliberately does not check**, so that a pass is never over-read: that a
new module is called by anything; that a file belongs in the repository at all
(§4 is a judgement, not a path rule); privacy; `config.ini` semantics.

## 2026-09-03 — finding: `keep_best_audio` is listed as a tagged zone and carries no tag

`WRITE_ZONES.MD` §2 lists `mergeVideo.py` / `keep_best_audio` among the existing
zones, and the paragraph above the table cites it as the reason the comment form of
the marker exists: *"`keep_best_audio` uses the comment form for exactly this
reason."* In the tree at `f0e7119` there is no marker of either form in that
function:

```sh
grep -n "AGENT modification" src/mergeVideo.py
# 663, 667, 674, 678, 682, 690, 789, 800 — all in adjust_delay_to_frame and
# remove_not_compatible_video. Nothing near keep_best_audio (1011).
```

The `fabricated` branch the entry refers to is present and correct
(`mergeVideo.py:1036-1039`), so what is missing is the permission, not the code.
By rule 1 — a path that is not open is closed — the function is frozen, and
`check_write_zones.py` enforces that reading.

**Consequence, and what was done instead.** Nothing is blocked today: the branch is
already written, and `SPEC_ZONE_A.MD` §4 says it activates by itself the day the
repair sets the key on the audio dict elsewhere. So the campaign proceeds without
the zone. It becomes blocking only if the `fabricated` comparison itself has to
change — a case we do not expect and will report if we meet it.
**This is a finding for the owner, not an edit** (`AGENT.MD` § Documents): the fix
is either two marker lines in `mergeVideo.py` or one line struck from the §2 table,
and both are owner decisions.

## 2026-09-03 — finding: the Rust sources are frozen by rule and editable by instruction

`src/lib.rs`, `src/main.rs` and `Cargo.toml` appear in neither `WRITE_ZONES.MD` §1
(frozen) nor §4 (open). Rule 1 settles it — *a path that is not listed as open is
closed* — and `check_write_zones.py` treats them as frozen.

`AGENT.MD` § Coding Directives points the other way: *"Rust (`audio_sync`) must
pass `cargo check`, `cargo clippy`, `cargo fmt` cleanly before submitting any
changes"*, and § Testing says `cargo build` / `cargo check` are "allowed and
recommended". A file nobody may change cannot have changes submitted, so the two
documents disagree.

**What was done instead:** the strict reading. Nothing in the first objectives needs
the Rust core — stage 2 is the instrument we measure *with*, not the one under
repair — so no work is blocked. Recorded rather than resolved, because
`AGENT.MD` § Documents is explicit that being right about a rule does not make it
ours to change.

## 2026-09-02 — master prepared for campaign 2

`gestionar_show` imported from `dev-AI` and corrected: the dual-uvicorn split, a
fixed internal port (`tools.internal_api_port = 42085`), and `fusion.py` rewritten
from 1334 to 381 lines — the queue and the merge kept, the campaign instrumentation
dropped.

Twelve defects found and fixed while wiring it, none of which the compiler could
see: `get_git_commit` raising `NameError` on first call, three `tools.*` attributes
referenced but never defined, `_env_flag` used for a path (it returns a boolean),
nine `tools.logs` writes leaking into thread contexts through cross-file
`Thread(target=…)` declarations, `.get(0)` on a list, a `Decimal`/`float` mix that
made the no-forced-best-video path drop every candidate *silently*, a lost-master
window in the production swap, a pool never shut down, and a production failure
that left no trace anywhere.

Two hazards were closed structurally rather than by a fix: the episode lock, because
the integration loop and the fusion worker are separate processes that can write the
same episode; and `tools.load_merge_runtime_from_env()`, because Python 3.14
defaults to `forkserver` and a forkserver child sees only module defaults — which
would have put the two processes' locks in different trees, silently, while
`software={}` failed every ffmpeg call.

Write zones were declared and the first tags placed. The rule that governs them —
*a unit test cannot tell you that nothing calls the unit* — comes from campaign 1
shipping five pieces of code that were reviewed, tested green, and never called.
