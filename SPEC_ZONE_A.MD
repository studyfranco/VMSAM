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

## 4. Same tag is not same recording — check concordance before building

**A language tag groups recordings that are not interchangeable.** VFQ, VFF and VFI
all carry `fre`. So do a dub and a redub. Two rules follow, and the second is a
defect in the code today.

**Before building, ask whether the master already carries the same recording.** If it
does, the chimeric track is work that will be thrown away — `keep_best_audio` drops a
fabricated track in favour of an intact one, which is the correct outcome and means
the assembly was wasted. **Test for concordance on the two tracks rather than on
their tags.** High correlation with the master's same-language track means the same
recording: do not build.

**And a fabricated track may only lose to an intact one carrying the SAME
recording.** The rule in `keep_best_audio` fires before every codec, channel and
bitrate comparison and asks nothing about content:

    elif audio_1.get('fabricated') and (not audio_2.get('fabricated')):
        audio_1['keep'] = False

Its two call sites do not behave alike. One passes tracks already established
compatible by a `validation` step; the other passes **every audio of one language,
unvalidated**. On that second path a chimeric VFQ and the master's intact VFF are
both `fre`, and the chimeric one is dropped — **losing a recording the output does
not otherwise carry.** The rule was written to stop a fabricated track winning a
codec comparison. It also silently destroys a variant, which was never intended.

Until the fabricated rule is conditioned on sameness, the concordance test is not an
optimisation but a guard: **not building a track the tag-blind rule would delete is
the only thing keeping the variant.**

## 4b. When the streams of one file disagree, refuse — owner's ruling

**Ratified.** `SPEC_ZONE_A` defined the contract for a *track* and said nothing about
a file carrying several. Faced with that silence an agent chose to **decline repair
when a file's streams disagree**, knowing it was the cautious reading and that it cost
yield, and reported that no rule had asked for it and no rule would have stopped the
opposite choice. The owner has ruled the choice correct and it is now the rule:

> **When there are two comparison audios and the method that would repair the
> analogous audios diverges, do not merge — a divergence means there is a problem.**

**The owner's documented case, and it carries an open verification.** In error 24 the
two Japanese tracks are **250 ms apart**. That is attributed to a bug of the time which
produced a drift of a few frames. **Whether that bug still exists has not been
established, and establishing it is a task, not an assumption.** Until it is, a
divergence between streams is exactly the signature that must not be merged over: the
repair would be built on top of a defect rather than around it.

Two files declined on this judgement do repair correctly with per-stream plans. **That
is not an argument against the rule** — it is the yield the rule knowingly costs, and
the owner has priced it.

## 5. Commentary tracks are not repaired — owner's ruling

**No chimeric version of an audio track tagged as commentary.** Pass it through
untouched or leave it out; never assemble one, and never mark one.

The repair rests on *what was done to the file was done to all its streams* — true of
a cut, true of a resample. **A commentary track is a separate recording over the same
picture**, not a translation of the programme's own audio, so it inherits neither the
premise nor the plan: the master may carry no commentary to fill from, and there is
no reason its edit points sit where the main track's do.

Three places in the current code walk `commentary` and each must stop:

    iterate_candidate_audios   builds a chimeric track for every commentary stream
    find_master_audio_for_language   would fill a gap from the master's commentary
    mark_audio_dicts   stamps `fabricated` on commentary and audio-description alike

**Audio description is the adjacent question and this ruling does not decide it.**
It is the same shape — a separate recording over the same picture — and it is not
what was ruled on. Raise it rather than extend this by analogy.

## 6. Track marking — `fabricated`

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

### The consumer that already exists — and it is not a defect

`keep_best_audio` prefers an untouched track over a fabricated one of the same
language, before any codec rule applies — a FLAC assembled from three sources must
not evict an intact AAC. The branch is already in place and inert: with the key
absent everywhere, `.get()` returns `False` on both sides and the current behaviour
is unchanged. It activates by itself the day the repair sets the key.

**This was reported as the campaign's largest finding and the owner has ruled it a
design intention. It is closed.** Measured first: `keep_best_audio` ranks on codec,
channels, sample rate and bitrate, and alignment is not in the comparison — a master
DTS 6ch at 1509 kbps beats a repaired E-AC-3 2ch at 224 kbps, so of 42 correctly
repaired tracks only **12 could reach a file**, those whose language the master lacks.

**The owner's answer: that is the point.**

> **The repair exists to add languages in the cases where they are absent.**

So twelve of forty-two is not a loss rate, it is the specification being met. A track
whose language the master already carries has nothing to add, and the master's own
recording is the better artefact — it is also the timeline everything else is measured
against. **Do not "fix" this, and do not put alignment into the ranking.** The owner
adds two facts that follow from it: **a chimeric VFQ is kept against a VFF** — a
different dialect is not the same language for this purpose — and a chimeric audio is
generally integrated **under the comparison language**.

**What this changes for whoever reports next:** an observation of a repaired track
losing to the master's own track is *the system working*, not evidence of a defect.
That distinction was already made once here — observations consistent with a defect
are not evidence of it — and this is the case it was made for.
