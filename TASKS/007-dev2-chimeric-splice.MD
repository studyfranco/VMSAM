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
| id 108, two change points one in the head | repaired, aligned 0.5 ms; the same plan minus one transition **declined at 503.0 ms** |
| **id 237, negative fixture** | **declined on the measurement** — `fid 0.576, 31 points, 18 flips`; zero output; ten stage-1 delays identical to ci's deployed-build baseline |
| guards | 43 structural + 18 orchestrator branches, each refusal paired with its opposite |

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
