# What a correct merge is

**Durable facts about merging, not campaign goals.** They survive the campaign that
happens to cite them: this is step 1 of `AGENT.MD` § Target-first — the properties a
target must satisfy before any code is written.

Four criteria, ordered by how hard a defect in them is to detect.

**1 · Frame-accurate alignment when the FPS matches.** The delay locks to a video
frame boundary — 41.7 ms at 23.976 fps. Sub-frame precision is discarded by design,
so do not sell a method on it; a delay that does not land on a frame is wrong even
when it measures well.

> **Two snaps exist and they use different grids.** `video.get_good_frame` snaps on the MINIMUM
> of the two files' declared rates and finds the frame by PSNR; `mergeVideo.adjust_delay_to_frame`
> snaps on the best-quality video's own rate — so this criterion may not be cited as the
> justification for `adjust_delay_to_frame`: that citation is to a different operation on a
> different grid, and when the rates differ the second snap moves the delay OFF the frame the
> PSNR scan chose.
>
> **And only half of this criterion is supported by measurement**: the relationship BETWEEN
> regions is frame-quantised; the claim that the ABSOLUTE delay lands on a frame is unsupported —
> and `adjust_delay_to_frame` acts only on the absolute half. *(The investigation, its n=15
> population and its credits: `CAMPAIGNS/02-dev-chimeric-resample/DOCS_EXTRACTS.MD`.)*

**2 · Cross-validation when FPS or content differs.** Audio correlation alone cannot
separate *shifted* from *edited*. Scene-cut detection must agree with the audio
before a boundary is trusted. Measured: audio-to-video skew is 0.4–99 ms across ten
release families, so the two instruments answer different questions — they agree on
**steps**, where the skew cancels, which is why cross-validation belongs at change
points and not at the global offset.

**3 · Distinct dubs preserved, subtitles deduplicated.** Two French dubs from
different distributors are two tracks, not a duplicate. Subtitles deduplicate on
*text* for encodable formats, on packet bytes for image formats (PGS/VOBSUB).

**4 · Chimeric integrity — dub matching before stitching.** A stitched track must
come from one dubbing source end to end. A correlation check **cannot** catch a
mismatch: two dubs of one language align in time perfectly well. Use `title`,
`codec`, `channels`, `channel_layout`. A track that changes voice cast mid-episode
passes every check in this repository.


## What counts as done — the three states

    PRODUCED    an artefact exists on disk, from the container, on the current image
    CHECKED     named automated properties verified -- NAME THE PROPERTY, never "passed"
    VALIDATED   a person or the validator compared it and accepted it -- AND WHO

**Only the third is done.** A compile is not a run; a guard suite is not production; and a lab
measurement that explains why a file fails is not evidence that anything was fixed.

> **THE LAST CHECK ANYONE RUNS BECOMES THE DEFINITION OF DONE UNLESS A DEFINITION IS WRITTEN
> ABOVE IT.** And each time that last gate gets stricter it makes the wrong definition more
> convincing — a stricter check that still cannot see a produced file reads as more evidence
> rather than as the same kind. This section is the written definition.
