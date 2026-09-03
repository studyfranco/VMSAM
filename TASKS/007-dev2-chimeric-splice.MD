# 007 — Turn a located change point into a track on the master's timeline

**Agent:** `vmsam-dev-2` (repair). **Pairs with 006.** **State:** open.
**Campaign objective 2, the repair half.**

## Goal

Given a change point located by 006, produce a **chimeric** audio or subtitle
track on the master's timeline: the candidate's content cut at the step, each
piece placed at its master-timeline position, and the gaps filled from the master
itself. Campaign 1 established that plateau reassembly works (`CAMPAIGN.MD`), so
the method is not the open question — landing it in a real merge is.

## Where it happens

Inside `remove_not_compatible_video`, **at the moment of refusal**, against
`best_video`. Not before the merge, not during it. `SPEC_ZONE_A.MD` §1, and
its three verified requirements:

1. `delay_same_md5_audio` **must be a `Decimal`** — `Decimal('0')` when the repair
   already aligned the file. It defaults to `None` and `mergeVideo.py:1426`
   computes `delay_same_md5_audio + delay_to_put`; `None + Decimal` raises.
2. The object must have had `get_mediadata()`.
3. The temporary file must survive until `mkvmerge` runs.

Rejoin the merge by attaching the repaired object to
`best_video.sameAudioMD5UseForCalculation` — consumed by
`generate_merge_command_common_md5`, which never goes through the delay machinery.

Mark the track: `fabricated = "chimeric"` on the audio dict, and
`VMSAM_FABRICATED` as a Matroska track tag on the first ffmpeg pass
(`SPEC_ZONE_A.MD` §4). Consumers test **truthiness, never equality**.

## Zone A is yours, and only yours

You are the only agent who writes in zone A (`mergeVideo.py:789-800`). The
capability itself goes in **its own module**, called from the zone, behind a
`config.ini` `[features]` key that is **off by default** — not an environment
variable: `run.sh` passes no `--config`, the `Dockerfile` is frozen, and a
default-off env flag is therefore permanently off. Campaign 1 measured that: the
code read 18 `VMSAM_` names and the image set two.

## What closes this

`SPEC_ZONE_A.MD` §2, and nothing less: on a **real error file that used to fail**,
a merged output is produced, the repaired track is **present in the final mkv**,
and you built the expected file yourself and compared against it. Point 3 is the
one dead code cannot fake.

## Ask the Lead for

- any tag other than zone A — one agent at a time, one edit
- a container run, through `vmsam-ci` — and state what you expect before it runs

## What changed / measured / dropped / blocked

**State: the path is built and lands a repaired track in a real merge. It has
NOT yet run on a real error file** — that is what still separates this from
`SPEC_ZONE_A.MD` §2.

### What changed — commit `3747488`

| file | zone | what |
| :--- | :--- | :--- |
| `src/mergeVideo.py` | zone A only | the call, and one guard fix (below) |
| `src/merge_video_chimeric.py` | new module | the assembly on the master's timeline |
| `src/merge_video_resample.py` | new module | the `asetrate` chain (serves 009) |
| `src/merge_video_repair.py` | new module | the orchestrator, five outcomes |

**The guard fix is the important half of the zone A change.**
`len(dict_file_path_obj) < 2` raised one line *after* the repair attaches its
object, and `fusion.py:286` calls `merge_videos` with exactly **two** files — the
master and the file in error. So on the path this campaign exists to fix, the
repair would have been killed every single time, and so would anything built on
dev-1's measurement. That is the campaign-1 defect shape reproduced exactly:
code written, reviewed, tested green, reached by nothing. The guard now stands
unless something was actually repaired.

### What was measured

Each against a target built by hand **before** the run.

| # | what | result |
| :--- | :--- | :--- |
| T1 | `VMSAM_FABRICATED` survives ffmpeg pass 1, pass 2, `mkvmerge --no-global-tags` and the final `mkvmerge`; mediainfo returns it under the track's `extra` dict | holds at all four stages |
| T3 | a candidate cut from the master's own samples, re-spliced | **byte-identical** to the target: 3 840 000 bytes, md5 `6cd624c1432422610c657d2f0e46cefd`. Sample-exact, not close |
| T4 | six subtitle cues, expected values computed by hand | all six land; 4 kept, 2 dropped as predicted |
| T5 | every guard fired with literals, each paired with a case proving the other answer | **37 checks, exit 0** |
| T8-A | control: flag shut | merge still dies at "useless to merge files"; outcome recorded `disabled` |
| T8-B | flag open, real merge, real audio | merged mkv produced; 2 audio + 2 subtitle tracks rebuilt, 0 declined, 0 failed; the final mkv carries `VMSAM_FABRICATED=chimeric` on audio **and** subtitle streams |
| T9 | is the repaired track on the master's timeline? | **−1.2, −0.2, +1.0, −0.4 ms** at 100/250/350/500 s, r = 0.993–0.998 |
| T9 control | the same instrument on the *unrepaired* candidate | −0.9, −0.4 ms before the cut, **+8000.2, +7999.8 ms** after |
| T9b | the same, at 44 100 Hz with parabolic interpolation (0.0227 ms/sample) | repaired: **−0.000, −0.000, +0.998 ms**; candidate: −0.658, +7999.252, +8000.998 ms |

T9's ±1.2 ms was the 8 kHz correlator's own quantisation, not error: at full
rate the repaired track is sub-millisecond at two of three positions.

The T9 control is what makes T9 mean anything: without it, "0 ms everywhere"
could equally have meant the correlator measures nothing. The correlator is
numpy cross-correlation, deliberately **not** `audioCorrelation` — otherwise the
instrument would be testing itself.

Fixture: a 600 s extract of a real example file with 8 s of content removed at
t = 300 s. Built by `lab/t6_build_fixture.sh` in the private laboratory. Details
in the private notes beside the repository.

**"Known by construction" had to be checked, and the obvious check is the wrong
one here.** `vmsam-dev-1` warned that a filter-based cut can silently remove the
wrong amount — its own first fixture used `aselect`, which cuts on audio *frame*
boundaries, and removed 1021.678 ms instead of 1000. Checked: mine uses `atrim`
on seconds, and the decoded sample counts are 26 491 904 against 26 139 648, a
difference of **7987.664 ms**, not 8000. That is **not** a cut error. The
candidate's audio is re-encoded, and 12.3 ms of AAC priming and trailing padding
sits between the duration difference and the content offset. The content offset
measures 8000 ms to within 1 ms (T9b). So the general rule holds — never trust
"known by construction" — but **the instrument must match the fixture**: a byte
count is ground truth for a PCM-exact fixture and is wrong by 12 ms on a
re-encoded one, where correlation is the only honest measure.

### On what the flag-shut control does and does not prove

`vmsam-ci` raised this and it is right. My T8-A control runs on a
**content-matched** pair, so shut-versus-open separates "the gate is shut" from
"the gate is open" on a file where repair is genuinely possible. It would prove
nothing on a file where refusal is the correct final answer — error id 237 is
such a file, and the same exception fires there for reasons unrelated to the
gate. A control taken on an unrepairable file cannot distinguish *repair
declined* from *repair impossible*.

### What an unrepairable refusal looks like, from a real one

From `vmsam-ci`'s baseline on error id 237 (details in the private notes beside
the repository), carried with its author's framing: stage 1 raised, ten windows,
delays `{−209625, −211000, −195125, −200500, +210125, −191250, −199375, −207375,
+208375, +203000}` ms, spread 421 s, **sign flipping between windows**, fidelity
flat at 0.5590–0.5781 with no window standing out. `vmsam-ci` established by
extracting and looking at frames — not from the filename — that the file is a
different television series from its folder master. So this is the shape of *no
shared content*: a floor, not a staircase buried in noise. A staircase has
plateaus that agree within a window group and high fidelity on them. My repair
must never accept this shape, and it cannot today, because it only ever executes
a plan somebody measured.

### The second fixture — independent provenance, and what it caught

`vmsam-ci` named the limit of the first fixture before I hit it: both its arms
share one provenance — a file and a copy of itself — so it proves the splice
arithmetic is exact and can prove nothing about media whose two sides were
encoded independently, which is every real case. Acted on rather than filed.

Measured first, to pick a genuinely independent pair from `/config/example`:
two of the four releases carry the **same** Japanese audio (lag 0.0/0.2/0.0/0.1 ms,
r = **1.000** at four positions), so they are one observation, not two. The BDRip
against the WEB-DL is independent — Blu-ray master against streaming master, FLAC
48 kHz against AAC 44.1 kHz — with a constant natural offset of −955.6/−955.1/
−955.4/−954.5 ms and r = **0.910–0.973**. High, but not 1.000.

Fixture 2 is that pair, with 8 s removed from the candidate. It exercises three
paths fixture 1 could not: a **non-zero base offset underneath the staircase**, a
gap filled **across codec and sample rate**, and the **silence** path, because the
BDRip carries only Japanese while the candidate carries jpn/kor/eng.

**It found a real defect on its first run, and the defect was mine.** I derived
the plan's base offset **with the wrong sign**. The repaired track came out
uniformly **+16 146 ms** off — exactly twice the 8072.85 ms base offset — with the
staircase correctly removed. Uniform, therefore invisible to every per-plateau
check, and visible only against the master. `vmsam-ci` predicted that failure
shape out loud before the run, and the diagnostic it proposed is what found it.

The sign I got wrong is the one written down as an equation, agreed in writing
with `vmsam-dev-1`, and verified once already on fixture 1. **Stating a convention
does not protect you from misapplying it.** Only a measurement against the master
does. Fixture 1 could not have caught it: its base offset was zero.

| | |
| :--- | :--- |
| A2 the gap holds **master** content | lag **0.3 ms**, r = 0.997 |
| A3 the same gap in kor/eng holds **silence** | RMS **0.00000000**, peak 0.0, against 0.064–0.110 elsewhere |
| A4 the master's FLAC jpn beats the fabricated AAC jpn | predicted, and confirmed — the intact track survived untagged |

A4 is the counterpart to finding 1: the eviction runs the **right** way round
whenever the codec rule genuinely favours the intact track. The bad case is two
tracks of the same codec where only the bitrate separates them.

### The check that should have caught it, now built in

`merge_video_chimeric` verifies its own output: after muxing it correlates each
rebuilt track against the master at up to four positions and **refuses the
repair** above a 100 ms tolerance. This is the automatic form of
`SPEC_ZONE_A.MD` §2 point 3 — produce the target and compare against it.

Two constraints shaped it, both from teammates:

- **The probes sit strictly inside candidate-sourced pieces, never in a gap.** A
  gap is master content by construction and would match at any offset — *a
  control that cannot fail is not a control* (`docs/AUDIO_SPEED_POLICY.MD`).
- **It verifies a zero, it does not measure a step**, so it is quantum-independent
  and does not violate `vmsam-dev-1`'s constraint that a step must never be
  re-measured at a different window length.

**Fired both ways on real production runs, not in a unit test:**

| plan | result |
| :--- | :--- |
| base offset with the wrong sign | **declined** — `track 1 (ja) off by 16145.9 ms. Tolerance 100 ms`, and the merge then correctly falls through to `Only ... file left` because nothing was repaired |
| corrected plan | **repaired** — `aligned, worst lag 0.5 ms` (probes at 148.6 s → 0.0 ms, 453.0 s → 0.5 ms) |

And a third outcome rather than a false pass: kor and en came back **`skipped` —
"the master has no track in this language"**. A track the master cannot verify is
not a verified track.

A verification refusal is now recorded as **`declined`**, not `failed`: the module
ran, looked at what it had produced, and said no. Collapsing those two would have
lost the outcome that carries a reason.

**Default.** `repair_verify` defaults **on**, inside a feature that is off by
default. `WRITE_ZONES.MD` §4's rule — new capability must not run unless enabled —
is held by `repair_chimeric`. The verification is not a further capability; it is
what stops the repair delivering a silently wrong track, and a safety you have to
remember to arm is not one.

### An uncertain change point costs master content, not correctness

`vmsam-dev-1` measured that audio brackets a change point to about **2.5 s**, not
to a frame, and asked whether it should build a video refiner to pin one. Neither
of the two options it offered was needed.

**Given a bracket, I do not choose a change point — I replace the bracket with
master content.** For a bracket `(lo, hi)` with offset `A` before and `B` after:

| case | safe gap | why |
| :--- | :--- | :--- |
| offset decreases, `D = A − B` — the master has content the candidate lacks | `[lo, hi + D)` | every `m < lo` is certainly before `T`; every `m ≥ hi + D` is certainly past `T + D` |
| offset increases — the candidate has content the master lacks | `[lo, hi)` | no master gap exists, so one is created and the candidate's extra content is dropped |

Everything inside the gap comes from the master, which is **correct by
definition** — not "inaudibly wrong". `vmsam-dev-1` supplies the gap already
computed, plus the raw bracket and step so the arithmetic stays checkable; any
margin **beyond** that is mine, because it depends on whether the fill is master
audio or silence, which is resolved per track at splice time.

**Verified rather than accepted.** Re-derived from the convention independently,
checked against fixture 1 where the answer was already known by another route
(`A=0`, `B=−8000`, `T=300000` → `[300000, 308000)`, which is the plan that had
already run), then swept 2001 values of `T` per bracket confirming
`g0 ≤ T` and `T + D ≤ g1` for every one, in both the deletion and insertion cases.

**The cost, and why the video refiner is not needed.** On id 266, two change
points with 1000 ms steps:

| bracket | substituted | share of the episode |
| :--- | ---: | ---: |
| `vmsam-dev-1`'s measured 2.5 s | 7.0 s | **0.49 %** |
| the ~108 s I had assumed | 218.0 s | 15.24 % |

At 0.49 % the substitution is not the deciding cost even if every gap is silence.
So the recommendation is *do not build the video refiner* — not "build it if the
silence population is large". It would only matter on a file with roughly twenty
change points, where silence reaches 5 % of a track.

**One asymmetry, stated because it is unwelcome:** when the fill is silence the
gap **cannot be narrowed** to reduce the cost — narrowing takes candidate content
from the wrong side of the change point, which is a real error rather than a
cosmetic one. On a silence-fill track the gap is a hard cost: accept it, or
decline the track. There is no middle.

Each rebuilt track therefore reports `gap_filled_ms`, `silence_filled_ms` and
`silence_fraction`, so the population and its cost come out of real runs as
numbers. A `repair_max_silence_fraction` exists and is **unset by default**: no
one has measured how much silence makes a track unusable, and putting a figure
there would be the plausible invented value `SPEC_ZONE_A.MD` §3 records about
`get_less_channel_number` fabricating `"2"` and `"44100"`.

Note for reading the first real result: `vmsam-forensic` reports that id 266
shares **jpn only**, master-only eng/por/spa, candidate-only none — so both
rebuilt tracks fill from the master and **nothing on that file exercises the
silence path**.

### Findings that are not mine to fix

1. **`keep_best_audio`'s `fabricated` rule cannot fire, and the harm it exists to
   prevent is now measured.** Both call sites (`mergeVideo.py:1853`, and `:998`
   via `:1861`) iterate the audio dicts of `out_video_metadata` — a fresh object
   built from the merged split file at `:1822` and populated by `get_mediadata()`
   at `:1823`, which never writes a `fabricated` key. The marker comes back from
   the file under `extra['VMSAM_FABRICATED']` instead. So `SPEC_ZONE_A.MD` §4's
   *"it activates by itself the day the repair sets the key"* does not hold as
   written. **Measured consequence:** in T8-B the split file carried all four
   audio tracks — the master's two intact and my two fabricated — and the final
   mkv kept **only the fabricated ones**. A fabricated track evicted an intact
   one, which is the exact sentence the rule was written to prevent.
2. **A re-encode cannot be made to lose a bitrate comparison against its own
   source, so that comparison is not a safe discriminator** — the general form of
   finding 1. First run: the encoder was left to pick its own bitrate and chose
   132 300 = 2 × 44 100 × 1.5, above the master's 127 997 / 127 999. That part
   was my defect and is fixed — the bitrate is now resolved from the source by
   four routes and the origin of the number is recorded. **The fix did not cure
   the eviction.** Re-run with `-b:a 130 741` / `130 271` resolved by an exact
   copy pass: my intermediates measure 135 672 / 134 749 bps (the AAC encoder
   overshoots its target by ~4 %), `keep_best_audio` still sees 132 300 against
   127 997, and the fabricated tracks still evict the intact ones. I have **not**
   established where mediainfo's 132 300 comes from — it is identical for two
   tracks encoded at different targets, so it is not a measurement of my output.
   What is established is that it is the number the rule compares, that it is
   above the intact track's, and that the fabricated track wins. Making a rebuild
   lose this comparison reliably would mean deliberately degrading it, which is
   gaming a rule rather than repairing a file. Only finding 1's rule can decide
   this correctly.
3. **`main.py` is broken under Python 3.14.** It calls neither
   `set_start_method` nor `tools.load_merge_runtime_from_env`, the default start
   method is `forkserver`, so a Pool worker sees `tools.software == {}` and every
   ffmpeg call raises `KeyError: 'ffmpeg'`. Measured on a plain two-file merge,
   before any repair code ran. `main_gestionar_show.py:432` already carries the
   one-line fix (`set_start_method('fork', force=True)`); `main.py` is frozen, so
   this is a finding. `run.sh` reaches it through `software=main`.
4. `not_compatible_video_list` (`mergeVideo.py:32`) is written at `:811` and read
   nowhere. Observation only; left untouched.

### What was dropped

- **WAV intermediates per spliced piece.** First design cut each piece to WAV and
  concatenated. Dropped for a single `filter_complex` per track: a 5.1 24-bit
  episode is ~1.2 GB of intermediates and `/tmp` is RAM shared by four agents.
  The cost is a constraint that is now enforced rather than assumed — the plan
  must read the candidate **monotonically**, or `concat` would make ffmpeg buffer
  an episode. A backwards read is refused, not worked around.
- **Encoding every chimeric track to FLAC.** Rejected precisely because finding 1
  makes the `fabricated` rule inert: a lossless rebuild of a lossy source would
  beat an intact track on the codec rules. The module now never upgrades a lossy
  source, and declines a codec it cannot re-encode in family (`dts`) by name.
- **Letting the encoder choose its bitrate.** That is what produced finding 2's
  132 300 bps. Replaced by a four-route resolution — `video.get_bitrate`,
  `BitRate_Maximum`, `StreamSize/Duration`, then an exact `-c copy -f null` pass
  (0.46 s for 592 s of AAC) — and a **decline** if all four fail, rather than a
  plausible invented number (`SPEC_ZONE_A.MD` §3).

### What stays blocked

1. ~~The `[features]` flag cannot be committed.~~ **Resolved — and I had the
   diagnosis wrong.** `src/config.ini` has no trailing newline, so an **append**
   rewrites the last line: git records `audio_sync=…` (no newline) removed and
   `audio_sync=…` (newline) added. The checker was not refusing my addition; it
   was correctly catching a real modification of a frozen line, which §3 does
   forbid. My append genuinely altered an existing line, just not in the way I
   intended.

   `vmsam-dev-1` found the fix and tested it on its own clone: **insert, do not
   append.** Placing `[features]` before `[software]` leaves the final line
   byte-identical. Verified independently here: pure addition with no `-` line,
   last 40 bytes identical, checker exit 0. And verified for meaning, which the
   checker explicitly does **not** do — parsing before and after gives identical
   `[mergerules]` and `[software]` sections, `repair_chimeric` reads shut,
   `repair_verify` reads open, and the silence budget reads unset.

   **One ambiguity, recorded rather than resolved.** §3 says existing variables
   are frozen — "not their value, not their name, not their **position**".
   Inserting a section above shifts every existing line's *number*, though no
   variable moves relative to another or changes section, and INI section order
   is semantically irrelevant. I read "position" as ordering rather than line
   number, and the rule's stated purpose supports that — it exists because "the
   merge rules encode measured codec relationships; changing one silently changes
   every merge", and none changes here. But the owner may read it the other way,
   and it is a finding rather than something for me to settle.

   **A trap left for the next person:** `config.ini` still has no trailing
   newline, so anyone who *appends* hits exactly this again.
2. **A new runtime module cannot be modified after its creating commit — now the
   only blocker, and independently confirmed.** `vmsam-dev-1` reproduced it on its
   own clone by appending one comment line to its already-committed
   `src/change_point_locator.py`: `§1 frozen file modified`, exit 1. It cannot fix
   a defect in its own locator either. The
   checker classifies `src/<new>.py` as frozen and exempts only git status `A`.
   Measured both ways: creating exits 0, modifying exits 1 with `§1 frozen file
   modified`. `WRITE_ZONES.MD` §4 opens "new runtime modules"; the instrument
   opens only their first commit. **This is now blocking a correctness fix**, not
   a hypothesis: the bitrate work in finding 2 is written and verified in the
   working tree and cannot be committed.
3. **A real error file.** Needs an exemplar id from `vmsam-forensic` and a
   container run from `vmsam-ci`. Error id 237 is ruled out — `vmsam-ci` found it
   is a different series from its folder master.
4. **Where does a test live?** `WRITE_ZONES.MD` §4 sends verification harnesses
   outside the repository and the checker allows new files only at `src/*.py`, so
   the test that `AGENT.MD` requires to pin the call site cannot travel with the
   code it pins. Everything above lives in the private laboratory instead.
