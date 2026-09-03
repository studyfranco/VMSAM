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

**State:** the repair runs unconditionally, gated by the measurement. `SPEC_ZONE_A.MD`
§2 met on real error id 266 against committed code, point 3 included. The corpus
sweep — `repaired` / `damaged` / `declined` — is the remaining work.

### What changed

| file | zone | what |
| :--- | :--- | :--- |
| `src/mergeVideo.py` | zone A only | the call, and the `< 2` guard that made the repair unreachable on the two-file fusion path |
| `src/merge_video_chimeric.py` | new module | assembly on the master's timeline, self-verification |
| `src/merge_video_resample.py` | new module | the `asetrate` chain and the factor actually applied |
| `src/merge_video_repair.py` | new module | the orchestrator, four outcomes, no flag |
| `src/config.ini` | removal | `[features]` removed entirely — `WRITE_ZONES.MD` §4 |

### What was measured

Each against a target built by hand **before** the run. Commands and full detail in
the private notes beside the repository.

| what | result |
| :--- | :--- |
| splice, synthetic, sample-exact target | **byte-identical**, md5 `6cd624c1…` |
| subtitles, six cues hand-computed | all six land; 4 kept, 2 dropped |
| speed, right direction | 62.562625 s / 959.039 Hz vs 62.562624 / 959.039 predicted |
| speed, **wrong** direction | 1042.709 Hz — 83.7 Hz away; refused twice over |
| tag factor | `1.042710`, not the 1.042709 requested; residual 1.981 ms per episode |
| **id 266, locator-driven** | **repaired**; both jpn tracks **aligned at 1.12 and 1.00 ms**, six probes each |
| **id 266, point 3** | matches a target built from **my own** boundaries at 13 positions incl. all three gaps: **0.00–2.31 ms, r 0.900–0.999** |
| **id 266, the plan itself** | independently confirmed by `vmsam-dev-1`'s pHash **video** probe — cumulative displacement across the staircase agrees to **3.17 ms** |
| id 108, two change points one in the head | repaired, aligned 0.5 ms; the same plan minus one transition **declined at 503.0 ms** |
| **id 237, negative fixture** | **declined on the measurement** — `fid 0.576, 31 points, 18 flips`; zero output; ten stage-1 delays identical to ci's deployed-build baseline |
| guards | 43 structural + 18 orchestrator branches, each refusal paired with its opposite |

### The weak joint in the §2 claim, and what closed it

Both targets built for id 266 — the piece-by-piece one and the one from my own
full-file profile — were compared against a track built from a plan, and the
second shared no numbers with the plan but the *comparison* still only ever tested
the splice's fidelity to boundaries nothing outside the audio correlator had seen.

`vmsam-dev-1` closed it with a **pHash video probe**: decoded frames, 64-bit DCT
hash, sharing no code, library, signal or failure mode with the chromaprint
correlator the plan comes from. Holding one segment's offset fixed and asking the
video what displacement it measured at five positions, the **cumulative
displacement across the staircase agrees to 3.17 ms**.

**Its limits, carried as it stated them, not generously.** Four steps confirmed,
not five — the −168 ms step at ~30 s is below the video's floor at either
resolution, so **segment 1 is not independently confirmed** and the staircase is
confirmed from 60 s on. Three determined readings carry it, not five: two
positions were undetermined by tie width and one match quality sits exactly at its
threshold. The two apparent step errors, −248 and +251 ms, cancel exactly — the
arithmetic of one misread position rather than two wrong steps — and a re-run of
that segment alone at 125 ms quantum put all three determined readings within one
quantum.

So the honest form is: the splice is faithful to its plan, and **the plan's
staircase is now corroborated from 60 s on by an instrument outside the one that
produced it.** Not "the boundaries are right", and not confirmed at the head.

### An unverified offset becomes a gap — and why the lag figures barely move

`vmsam-dev-1` flags a segment whose span is shorter than its 60 s probe window:
no clean probe fits inside it, every window overlapping it straddles the
transition, and a peak-picking correlator on a straddling window returns a
**displaced peak** — arbitrary in sign and unbounded by the grid. On id 266 that
was 168 ms, and 168 is an instance rather than a ceiling, so the 100 ms
verification tolerance cannot rescue the case.

Such a segment is therefore **filled from the master, never spliced**, and the
pair declines only when *every* segment is flagged. First two on real media:

| id | master | unverified | of output | total gap | master fill | worst ms | was |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1426.0 s | 0.00 | 0.0 % | 312.0 s | 21.9 % | 1.50 | 1.50 |
| 12 | 1198.6 s | 75.68 s | 6.3 % | 378.6 s | **31.6 %** | 17.88 | 17.50 |
| 13 | 1319.2 s | 79.32 s | 6.0 % | 295.6 s | 22.4 % | 52.00 | 52.38 |

**The worst lag barely moves, and that is correct rather than disappointing.**
Dropping an unverified segment removes candidate content whose offset was never
bounded and replaces it with master content. The probes sample *candidate*-sourced
pieces, so the region that changed is no longer probed at all, and the lag figures
come from pieces that were already sound. **The fix does not improve the measured
alignment; it removes content that could not be measured.** Only the second was
claimed.

The cost is visible where it belongs: id 12's master fill went 25.3 % → 31.6 %.

### What was dropped

- **The feature flags.** `WRITE_ZONES.MD` §4: a repair conditioned on a parameter
  is not a repair. With them the campaign's product did nothing as shipped.
- **WAV intermediates per piece** — one `filter_complex` instead; a 5.1 episode is
  ~1.2 GB of intermediates and `/tmp` is RAM shared by four agents.
- **Encoding chimeric tracks to FLAC** — never upgrade a lossy source; a
  fabricated lossless track evicts an intact one and the `fabricated` rule is inert.
- **Letting the encoder pick its bitrate** — it chose 132 300 against a source's
  128 000. Four resolution routes then a decline.
- **Absorbing a uniform residual into `delay_same_md5_audio`** — the residual is
  not one number (+8.9 ms on one track, −18.6 on the other), and a uniform residual
  is also the signature of a plan sign error.
- **A hand-plan file** — a plan supplied by a human is the deferral §4 forbids.

### What stays blocked or open

1. **The corpus sweep.** 250 ATTEMPT files, with `vmsam-forensic`'s 52 CANNOT-HELP
   excluded and named — 27 wrong series, 11 where the defect is in the **master**,
   7 no shared language, 2 wrong episode, 2 defective master, 2 unknown, 1 lying tags.
2. **The three counts carry an unbounded term.** `vmsam-ci` measured the corpus
   record is a sample through a geometry with two blind spots: first visible change
   point median 228 s, 15.9 % of each file unsampled, and a step under ~62 ms
   invisible anywhere. `declined` is the most exposed of the three.
3. **`keep_best_audio`'s `fabricated` rule cannot fire** — both call sites iterate
   a freshly probed merged file, where the marker arrives as
   `extra['VMSAM_FABRICATED']`. Measured: fabricated tracks evicted intact ones.
4. **`main.py` is broken under Python 3.14** — no `set_start_method`, no
   `load_merge_runtime_from_env`; a Pool worker sees `tools.software == {}`.

### Owner's ruling — commentary tracks are not repaired (`caa49fd`)

`SPEC_ZONE_A.MD` §4. **No chimeric version of an audio track tagged as
commentary.** Pass it through untouched or leave it out; never assemble one,
never mark one. Dispatched to `vmsam-dev-2` as a behaviour change due before its
next promotion, not as a documentation change.

Three call sites, all confirmed present in the tree at `caa49fd`:

    src/merge_video_chimeric.py:586   iterate_candidate_audios
    src/merge_video_chimeric.py:390   find_master_audio_for_language
    src/merge_video_repair.py:396     mark_audio_dicts

`mark_audio_dicts` stops stamping `commentary` and **keeps** stamping
`audiodesc`. **Audio description is deliberately left open** — the spec says to
raise it rather than extend by analogy. The resulting asymmetry in the code is
correct until the owner rules on it.

The ruling costs nothing already shipped: the commentary path has never run on a
delivered artefact. `vmsam-ci`'s id 150 output carries five `fabricated` tags and
all five are candidate-only languages on the main audio holder.

### Ordering decision — the lead's, recorded because it moved an owner ruling

`vmsam-dev-1`'s point 3 is ahead of the ruling above and ahead of ids 44 and 48:

> Do any of the 42 repaired files carry a nonzero `start_time` difference on a
> rebuilt track?

It is the only open branch whose answer is that **something already shipped
wrong**; a small difference passes under the 100 ms tolerance silently, the same
shape as id 266's 27.5 ms per-stream split. Everything else outstanding concerns
files not yet produced. It is also cheap — one container field per rebuilt track
against its source, no re-runs, no correlation.

### Contract ruling — the plan must declare its reference frame

`vmsam-dev-1` measured that the 1103 ms on release 32 is a container
`start_time`: `ffmpeg -ss t -i file` seeks by presentation timestamp and absorbs
it, decoding from zero and cutting at `t` does not. The two quantities differ by
982.5 ms on `jpn` and 51.5 ms on `eng`, and the produced file's measured offset
matches the decode-from-zero figure to 0.06 ms. **A frame mismatch across the
handoff, not a wrong number on either side.**

Which framing the assembly wants is `vmsam-dev-2`'s to answer — it owns the mux.
What is ruled here is narrower and independent of that answer: **the plan carries
its reference frame as a key**, not as a convention and not as a sentence in a
document. A number that could be either quantity, in a field named neither,
consumed by a module assuming the other, is a defect whichever reading wins.

`vmsam-dev-1` has a one-flag change ready and has deliberately not made it:
re-framing the field would silently move every offset the module has emitted,
including plans already repaired against and the id 46 plan that verified at
1.125 ms. Holding it is correct.

#### What the `start_time` finding does **not** do — `vmsam-forensic`

The container offset **cannot dissolve the `different cuts` family**, and the
argument is arithmetic rather than judgement. The two measurements are different
quantities:

    vmsam-dev-1     an ABSOLUTE offset of ~983 ms between two measurement methods
    vmsam-forensic  a SPREAD of 5000 ms between two plateaus inside one file

A constant container offset shifts both plateaus equally and **cancels in a
spread** — it cannot produce a spread at all, and 983 ≠ 5000. Two plateaus plus a
change point at a different position in every episode remain evidence of a
genuine per-episode cut. **`different cuts` stands.**

What the finding *is* about: **why two measurements of the same file disagree by
~1 s depending on method.** That is a live hazard on the seam between `vmsam-dev-2`'s
extraction and `vmsam-dev-1`'s probe, which is why the 42-file check comes first.

`vmsam-forensic` reproduced `vmsam-dev-1`'s figures independently on its own data
to 0.1 ms — id 125, candidate minus master: `jpn −983.0`, `eng −48.0`.

**Unconfirmed, and flagged as such:** `vmsam-dev-1` wrote that *every* episode of
release 32 carries the same `jpn start_time`. Across twelve episodes
`vmsam-forensic` measures −983 −983 −983 −959 −983 −911 −983 −959 −959 −935 −772
−791 ms — **clustered, range 211 ms**, six of twelve at −983. The reconciliation
to test is whether the four files `vmsam-dev-1` measured all drew from the −983
cluster; if so the mechanism is untouched and only *every* was too strong. Put to
`vmsam-dev-1`; not resolved.

**Correction to this file's own arithmetic:** folder 32 holds **24** files — 19
`different cuts` plus 5 `trimmed edges` — not the 19 previously relayed.

#### `holder[language][0]` — narrowed by `vmsam-forensic`

The defect costs nothing where the fill and the alignment resolve to the **same**
index: consistently wrong together cancels. **The cost is incurred only where the
two resolve differently** — a condition checkable in the code, rather than a
corpus property needing a survey. Worst case is id 24: 14 streams with **three**
`jpn`, so `[0]` is one choice of three.

The divergence is **not** itself a container artefact. All eleven masters probed:
ten show 0.0 ms same-language `start_time` spread, id 24 shows 16.0 ms against a
126–138 ms gap. Real audio misalignment. The test was run by its own author
looking for the outcome that would have cost it the finding.

#### The contract ruling above is **withdrawn** — `vmsam-dev-2` measured it away

The ruling "the plan must declare its reference frame as a key" rested on there
being two frames. **There are not.** `vmsam-dev-2` had inferred from a correlation
that its extraction does not honour `start_time`; the direct test says the
opposite:

    same file, same stream, same window, two extractions
    ffmpeg -ss T -i           (PTS seek, absorbs start_time)
    [0:N]atrim=start=T        (what the assembler does)
    stream start_time  120.0 ms
    difference          +0.0 ms over 20 s

Same frame. **No conversion is needed between the plan and the assembly**, and
`vmsam-dev-1` has been told not to touch its probe. A key naming a distinction
that does not exist is not hygiene, it is a second thing to keep true. Withdrawn
for the same reason `vmsam-dev-1` reverted its accurate-seek change: **a
defensible improvement with no evidence behind it is not a fix.**

Recorded rather than deleted because the ruling was dispatched to two agents.

**The cadence guard is why this file never carried the error.** The ruling was
staged for commit and refused — *a commit touching no code landed 5 min after the
last one* — and `vmsam-dev-2`'s correction arrived while it sat unstaged. The
guard was written for a different reason and caught this one.

#### The real defect — `vmsam-dev-2`'s, and it is not the one reported

The assembler ends every piece with `asetpts=PTS-STARTPTS` and concatenates, so
**the produced track starts at PTS 0 while the master's track starts at its own
`start_time`.** The misalignment is **the master stream's `start_time`**, not the
master-minus-candidate difference previously reported.

    id    decline      master jpn start_time    apart
    144   1103.4 ms          1103.0 ms          0.4 ms
    157   1103.8             1103.0             0.8
    174   1059.4             1055.0             4.4
    225    887.6              887.0             0.6

Three different values, agreeing to under 5 ms.

**Two figures withdrawn by their author.** "26 of 79 repaired files" counted the
wrong quantity; counted correctly it is **7 of 79, largest 74 ms, all under the
100 ms tolerance**. The "constant 120 ms remainder" across release 32 **does not
exist** — an artefact of subtracting the difference instead of the master's
`start_time`. There is no second constant.

The finding survives in kind and **shrinks by a factor of four**: something would
have gone out with a silent systematic offset, seven files rather than
twenty-six, mechanism fully explained rather than partly.

**Nothing has shipped wrong.** Id 150, the only file actually delivered, has
master `start_time` 0.0 on both languages; `vmsam-ci`'s artefact is unaffected.
This closes `vmsam-dev-1`'s point 3.

**The fix is not made, deliberately.** Carrying the master track's `start_time`
moves every produced track by tens of milliseconds. Held until there is a control
that can fail — *what `mkvmerge` actually does with a zero-based track muxed
beside a `start_time`-bearing one.* Author's words: **"I would rather be a day
slower than ship a correction I inferred."**

#### Ids 44 and 48 repaired — the decline count was not a corpus property

    id 44   en 19.75 ms   en 4.625 ms   fr 5.0 ms   ja 2.625 ms
    id 48   en 17.0 ms    fr 5.0 ms     ja 1.5 ms

Both had declined in the sweep at ~196 and ~211 ms. Every track now lands inside
20 ms. **The two streams of id 44 sharing the `eng` tag come out at 19.75 and
4.625** — different from each other, as `vmsam-dev-1`'s measurement predicted, and
**a language-keyed plan would have been 238 ms wrong on one of them.**

Fifteen declines now resolve as: **2** fixed by per-stream plans (44, 48), **4**
explained by the `start_time` defect and pending the fix (144, 157, 174, 225),
**2** locator nulls, **7** not yet diagnosed. **The decline count was measuring two
missing features and one defect of the repair, not a property of the corpus.**

### The two defects, separated — and the exposure population

**Both were real, both were `vmsam-dev-2`'s, and their signatures were identical:**
each equal to the master stream's `start_time`.

| | where | mechanism |
| :--- | :--- | :--- |
| assembler | `atrim` + `asetpts=PTS-STARTPTS` + `concat` | a piece asked for `[0, 120]` on a stream beginning at 1.103 s returns **118.900 s**, so everything after the head moves up by 1.103 s |
| verifier | master reference fetched by PTS seek | a probe at position 0 on that stream returns content from 1.103 s — **the reference itself was shifted** |

**They were separable only because the first was caught by a LENGTH and not by a
lag.** Measured by lag alone the two are one number forever, each confirming the
other, and every correction appears to half-work. A second *quantity*, not a
second opinion — distinct from the day's other instrument defects, which all fell
to a second instrument.

`no_reference` is now a fourth probe outcome: a position earlier than the master
stream's start is **an absent comparison, not a failed one.** This is the
campaign's recurring defect answered directly — a missing thing reported as a
measured thing.

**Withdrawn:** "7 of 79 repaired files carry a systematic offset, largest 74 ms."
Measured by the shifted verifier; the split between assembler and output is being
re-run, not reasoned about.

**Survives on independent grounds:** nothing has shipped wrong. Id 150 is the only
delivered file and its master `start_time` is 0.0 on both languages, so neither
defect had anything to act on.

#### id 144 — the first refused file taken end to end

    track 1 (en)   ALIGNED   worst  5.625 ms   (was 48.25)
    track 5 (ja)   ALIGNED   worst 12.125 ms   (was inconsistent by 1099 ms)

**No third term.** The head probes now sit at **48 ms and 1103 ms, not 0** — the
two master streams' own `start_time`s — and both read within a millisecond of
zero. The probe producing a 1103 ms error had been asking for audio that does not
exist. **`vmsam-dev-1`'s plan was correct throughout, on both languages, with no
per-stream plan needed.**

#### The exposure population — `vmsam-dev-1`, `0cbeac8`

Re-sliced on the master's own `start_time` rather than the mismatch. 315 pairs,
ffprobe only. **The earlier 112 and 77 counted mismatches and are superseded by
their author.**

    master start_time nonzero on a shared language   65  (21 %)
      >= 100 ms    visible, declines                 25  (24 of them release 32)
      20 - 100 ms  UNDER the 100 ms tolerance        33  <- the silent band
      < 20 ms                                         7

**The defect was never release-specific; its visibility was.** The visible
declines cluster in one release because that release's masters carry the large
values; the rest of the corpus carries the same defect where no instrument looked.

`vmsam-ci`'s control file **id 695 falls in the silent band**, reached
independently from a different starting set — the corroboration the id 150
selection never had.

#### `vmsam-dev-1`'s correction, made against itself

Release 32's 24 episodes carry **ten distinct** master `jpn start_time`s spanning
216 ms (887 … 1103), not one value. The four declines sit at 1103, 1103, 1055 and
887 — so the errors **agreed across the pair sharing a value and tracked the input
on the pair that does not.**

**The corrected version is the stronger evidence.** An identical input predicting
an identical error fits any constant, including a fixed 1103 ms bug. A clustered
input predicting a clustered error that tracks it file by file across three
distinct values **cannot be a constant** — matched at 0.4, 0.8, 4.4 and 0.6 ms.

157, 174 and 225 are **not** counted as fixed. Same signature and same release is
the reasoning that produced two confident errors today.

### The silence is inherent, not a fallback — and it is the repair's own population

`vmsam-dev-2` split the fourteen silence-filled tracks. **There was nothing to
split: all fourteen are inherent.** For every one, the filled track's language is
**absent from the master** — thirteen of the fourteen are French against a master
carrying Chinese, English and Japanese.

    id 695   candidate eng,jpn   master jpn
    id 216   candidate fre,jpn   master chi,eng,jpn
    id 353   candidate fre,jpn   master eng,jpn
    id 267   candidate fre,jpn   master eng,jpn,por,spa

So silence is **the only honest option short of not producing the track**. The
alternative is not "fill it better", it is "decline it".

**This answers the owner's older open question — what the repair is *for*.** A
shared language either loses to the master in `keep_best_audio` or is partitioned
away by `find_differences_and_keep_best_audio`; the tracks that survive to a
delivered artefact are exactly the **candidate-only** languages. The repair's
entire delivered value is *add a language the master lacks* — **and that is
precisely the population whose gaps cannot be filled.** The value and the defect
are the same files. Declining every track whose language the master lacks does not
trim the repair, it deletes it.

The question, in its author's words: **at what fraction is a track with no source
for its gaps still worth producing?**

Measured: 14 tracks, median **12.9 %**, max **24.5 %**, `silence_ms` equals
`gap_ms` **exactly on all fourteen** — the mechanism is fully accounted for with no
unexplained remainder. Confirmed from outside the producing stage: `vmsam-ci`
sampled the delivered file at 30 s spacing and got **23.4 %** against the
assembler's 24.5 %. Two methods sharing nothing, one point apart.

### The artefact does not record it — open with the owner

The mux writes one tag per stream, `VMSAM_FABRICATED={marker}`. `gap_filled_ms`,
`silence_filled_ms` and `head_pad_ms` exist only in the in-memory report. **A
delivered track that is a quarter silence says "I was built" and nothing else.**

Recommended: **a second per-stream tag, not an extended marker.** `chimeric:silence=21.7`
looks in-grammar under §5's `method[:detail]`, but `chimeric+resampled:<factor>`
already puts a float after the colon — a consumer splitting on `:` to read a
resample factor would silently read a silence fraction as one. Overloading the
field consumers are told to test for truthiness is how a marker stops being a
marker.

Prepared, not shipped, pending the owner: §5's vocabulary is his.

### Separability of the two defects — decidable per file

`vmsam-dev-2` checked `vmsam-dev-1`'s arithmetic, found it sound, and located the
gap **in its own description**: a master fill piece at the head is *also* cut from
a stream, so when the plan does not begin at master 0 that piece carries the
master's `start_time` as an **assembler** term — the same number as the verifier
term, same file, inseparable.

    plan begins at master 0       -> separable, verifier-only
    plan begins after master 0    -> both terms equal the master's start_time

**Id 375 — the file reached for first because its number was largest — is the one
file where the two cannot be told apart.** Its plan begins at master 100000 ms;
id 173's begins at 0. The other six carry the answer, decidably, from the plan's
own first segment.

**Withdrawn:** id 375's 74.88 → 1.38 ms. Two copies of the re-run were racing on
one output path, the second truncating the first — caught when a just-read log came
back empty. Not reconciled: every output both had produced was deleted and a single
instance relaunched behind a **PID lock**, a check on the process rather than on a
command-line pattern, because *a pattern matches the shell that searches for it*.
Two copies of the measurement meant to settle whether anything shipped wrong,
racing on one path.

### Survivorship — the ceiling on what the seven can settle

`vmsam-ci`'s general shape: **a defect's best test cases are systematically absent
from any sample collected while the defect was live**, because the "repaired" set
is by construction the set where the defect did not fire.

Applied to the seven by their own author: they are repaired files drawn from the
set a **shifted verifier passed**, so the files where the assembler defect fired
hardest are in the **declined** set — release 32. A clean result therefore licenses
only *"the files that passed a defective check were not additionally damaged."*
Close to circular. **It is not "nothing was about to ship wrong"**, which is how
both `vmsam-dev-1` and the lead had been framing it.

    ARM A   the seven      did the defect harm what the old verifier PASSED?      ceiling above
    ARM B   the declines   does what it REJECTED now pass, and at what alignment?  load-bearing

`id 144` is arm B's first member: aligned at 5.625 and 12.125 ms.

**Arm A results so far, reported per file as decidable or not:**

    id 375   master start_time 74 ms   74.88 -> 1.38 ms   plan @ master 100000  NOT DECIDABLE (both terms are 74)
    id 173   master start_time 42 ms   39.62 -> 2.88 ms   plan @ master 0       DECIDABLE, verifier-only

`id 173` is the first separable case and falls where `vmsam-dev-1` predicted: no
head master piece, candidate `start_time` zero, so the assembler term is
structurally zero and the whole 39.62 ms was the shifted reference.

### The fourteen — clean on one axis, conditioned on another

**Clean:** `silence_filled_ms` is computed **at assembly**, per track, before any
selection. The only rule that could select on it is `max_silence_fraction`, which
is `None` on every production path — nothing has ever been dropped for being too
silent. **The fourteen are not conditioned on their own value.**

**Not clean:** they come from files that **passed verification**, because the sweep
harness records fill columns only on a repaired row. A declining file *builds its
tracks* — assembly completes before verification runs — and then discards the
counts with the exception. **The distribution is over survivors of an alignment
rule.** Direction of bias unmeasured and not guessed.

**Honest form: median 12.9 %, max 24.5 %, _over tracks that verified_.**

Now measurable rather than merely flagged: a refusal carries the audio reports it
built, so re-running the declines yields their fill fractions and the distribution
can be computed over **everything that was built** rather than everything that
passed. Folded into arm B, not run separately.

**A decline was reporting what it measured and not what it produced** — the same
defect as the probes, one field along.
