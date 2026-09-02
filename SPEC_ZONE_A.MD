# Zone A — repair specification

**What to build.** Permission to build it is in `WRITE_ZONES.MD`; this file does
not grant any.

Draft: written from campaign 1's measured record and the owner's design decisions.
The owner owns it — correct it rather than working around it.

---

## 1. The contract

Repair happens **when a file is rejected**, inside `remove_not_compatible_video`.
Not before the merge, not during it. The rejected file is repaired against
`best_video` and rejoins the merge through the MD5 group.

`best_video` is a **video object** on both call sites.

Available without any change:

```python
video_obj.delayFirstMethodAbort[best_video.filePath]   # -> [1|2, delays]
```

`first_delay_test` fills it at 12 points. It tells you *why* the pair was refused
and which delays were seen — you do not need to re-run the correlation that just
failed.

Rejoining the merge: attach the repaired object to
`best_video.sameAudioMD5UseForCalculation`. It is then consumed by
`generate_merge_command_common_md5`, which never goes through the delay machinery.

**Three requirements, each verified against the code:**

1. `delay_same_md5_audio` **must be a `Decimal`** — `Decimal('0')` when the repair
   already aligned the file. It defaults to `None` (`video.py:55`) and
   `mergeVideo.py:1426` computes `delay_same_md5_audio + delay_to_put`:
   `None + Decimal` raises `TypeError`.
2. The object must have had `get_mediadata()` — `generate_new_file` reads
   `audios` and `video['Duration']`.
3. The temporary file must survive until `mkvmerge` runs. Do not put it anywhere
   that is cleaned between the repair and the merge.

## 2. What closes the zone

**Not** "the code is in the zone and the tests are green." The previous campaign
produced five pieces of code that were written, reviewed and unit-tested green,
and that **nothing ever called** — a guard, a refined measurement, a policy
branch, an order check, a subtitle repair. Every suite passed. A unit test calls
the function directly, which is exactly what production did not do.

Zone A closes when, on a real error file that used to fail:

1. a merged output is produced, and
2. the repaired track is **present in the final mkv**, and
3. the agent produced the expected file itself and compared it against the output.

Point 3 is the one that cannot be faked by dead code.

## 3. Why two lines in the tagged zones exist

**`get_good_frame` — `calculated_delay = float(calculated_delay)`.**
`test_if_constant_good_delay` returns a `Decimal`, and the division by 1000 in
`get_best_video` keeps that type, while every timeline value in `get_good_frame` is
a float seeded by `get_fps()`. The first addition raised `TypeError`,
`compare_video.run` caught it and recorded the pair as *not compatible* — so the
whole no-forced-best-video path silently dropped every candidate. Converting the
rest to `Decimal` buys nothing: `1/23.976` is not exact in decimal either, `int()`
truncations dominate, and the result is snapped to a frame by `round()` downstream.

**`video.py` fallbacks — the `BaseException` re-raise.** `get_less_channel_number`
and `get_less_sampling_rate` answer a failure with a fabricated plausible value
(`"2"`, `"44100"`) from a bare `except:`. Those values are indistinguishable from
real ones everywhere downstream, and the sampling rate decides how the comparison
audio is extracted — every correlation then rests on it. Both now log under
`tools.dev` and re-raise `KeyboardInterrupt` / `SystemExit`. If those log lines ever
appear in a run, that is a finding.

## 4. Track marking — `fabricated`

**Defined here, before the repair exists, so the repair implements a contract
rather than inventing one.** Nothing sets it today; every consumer must therefore
tolerate its absence.

### In memory

On the audio metadata dict — the same dicts `keep_best_audio` iterates:

| key | `fabricated` |
| :--- | :--- |
| type | `str` |
| absent or empty | the track is **untouched**: it came from a source file as it was |
| present | the track was **built** by the repair, and the string says how |

Defined values, `method` optionally followed by `:detail`:

| value | meaning |
| :--- | :--- |
| `chimeric` | assembled from two or more sources on the master's timeline |
| `resampled:<factor>` | stretched by that factor, e.g. `resampled:1.042708` |
| `chimeric+resampled:<factor>` | both, in that order |

Consumers test **truthiness**, never equality: a value they do not recognise still
means fabricated. Adding a method is therefore backward-compatible.

### On the produced file

The same string, as a Matroska **track tag** named `VMSAM_FABRICATED`, so the
origin of a track survives the merge and is recoverable afterwards:

```sh
ffprobe -v error -show_entries stream_tags=VMSAM_FABRICATED -of compact <file>
```

Set it on the first ffmpeg pass (`-metadata:s:a:<n> VMSAM_FABRICATED=...`, next to
the existing `-map_metadata 0`); `mkvmerge` preserves track tags — the merge command
passes `--no-global-tags`, not `--no-track-tags`.

An output whose tracks were rebuilt and which carries no such tag is a defect, not
a detail: campaign 1 spent real work dating 67 outputs by mtime because nothing in
them said what produced them.

### The consumer that already exists

`keep_best_audio` prefers an untouched track over a fabricated one of the same
language, before any codec rule applies — a FLAC assembled from three sources must
not evict an intact AAC. The branch is already in place and inert: with the key
absent everywhere, `.get()` returns `False` on both sides and the current behaviour
is unchanged. It activates by itself the day the repair sets the key.
