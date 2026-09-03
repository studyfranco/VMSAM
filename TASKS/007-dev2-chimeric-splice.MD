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
*(fill as you go)*
