# What a correct merge is

**Durable facts about merging, not campaign goals.** They survive the campaign that
happens to cite them: this is step 1 of `AGENT.MD` § Target-first — the properties a
target must satisfy before any code is written.

Four criteria, ordered by how hard a defect in them is to detect.

**1 · Frame-accurate alignment when the FPS matches.** The delay locks to a video
frame boundary — 41.7 ms at 23.976 fps. Sub-frame precision is discarded by design,
so do not sell a method on it; a delay that does not land on a frame is wrong even
when it measures well.

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

