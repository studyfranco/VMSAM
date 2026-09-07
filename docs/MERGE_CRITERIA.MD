# What a correct merge is

**Durable facts about merging, not campaign goals.** They survive the campaign that
happens to cite them: this is step 1 of `AGENT.MD` § Target-first — the properties a
target must satisfy before any code is written.

Four criteria, ordered by how hard a defect in them is to detect.

**1 · Frame-accurate alignment when the FPS matches.** The delay locks to a video
frame boundary — 41.7 ms at 23.976 fps. Sub-frame precision is discarded by design,
so do not sell a method on it; a delay that does not land on a frame is wrong even
when it measures well.

> **WHICH SNAP THIS CRITERION DESCRIBES, because two exist and they use different
> grids.** This criterion is satisfied by **`video.get_good_frame`**, which snaps on
> `frame_rate_use` = **the MINIMUM of the two files' declared rates** (`video.py:713-718`)
> and locates the frame by a **PSNR scan over five candidate offsets** (`:743`) — a grid
> the two files can actually be compared on, found by measurement. **It is NOT what
> `mergeVideo.adjust_delay_to_frame` does:** that snaps on **the best-quality video's own
> rate** (`mergeVideo.py:670`), with `round()` where the other truncates, reading the same
> mediainfo string once as `float` and once as `Decimal`. **Nothing makes the
> best-quality video the lower-rate one.**
>
> Measured by feeding legitimate `get_good_frame` output to the real
> `adjust_delay_to_frame`:
>
>     both 23.976              worst over k=1..8   0.0000 ms   EXACT NO-OP
>     use 23.976, best 25                         13.6670 ms
>     use 23.976, best 29.97                      16.6834 ms
>     use 25,     best 23.976                     13.6670 ms
>
> **When the rates match the second snap buys nothing; when they differ it moves the
> delay OFF the frame the PSNR scan chose, and the error grows with the delay because
> the grids are incommensurate.** *So this criterion may not be cited as the
> justification for `adjust_delay_to_frame` — that citation is to a different operation
> on a different grid.*
>
> **AND THE SENTENCE ITSELF CARRIES TWO CLAIMS, WHICH IS THE DEFECT NO CITATION CAN BE
> CAUGHT ON.**
>
>     SUPPORTED    the relationship BETWEEN regions is frame-quantised
>                  steps between consecutive regions: 6 of 9 within 0.02 of an
>                  integer frame at 23.976, 0 of 9 at 25 fps, p = 3.1e-07
>     UNSUPPORTED  the ABSOLUTE delay lands on a frame
>                  offsets mod 41.708: mean |residual| 9.60 against a uniform
>                  expectation of F/4 = 10.43, se 1.55 at n=15 -- INDISTINGUISHABLE
>                  FROM UNIFORM
>
> **`adjust_delay_to_frame` acts only on the ABSOLUTE delay -- the unsupported half.**
> *And the sentence gives no way to say which half is being invoked, so nobody citing it
> can be caught doing so.*
>
> **NO WORDING IS PROPOSED HERE AND THE EVIDENCE IS NOT SUFFICIENT FOR ONE.** *n=15 from
> four artefacts, two of them lab files, not independent in provenance; and a second
> population -- 112 non-zero container start-time differences, one per file --
> concentrates on the AAC frame grid at 4.2x while sitting at 1.1x on the video grid, i.e.
> uniform.* **Two populations, the same direction, and NEITHER IS THE POPULATION THAT
> ACTUALLY REACHES THE SNAP.** *What is recorded is that the criterion conflates two
> claims, which is true whichever half the data eventually supports*
> (`vmsam-dev-3`, `vmsam-dev-4`, `vmsam-dev-1`).

>
> **And it is path-dependent.** `get_good_frame` is called only from `get_best_video`
> (`mergeVideo.py:644`). On the forced-best-video path — `compare_video.run`,
> `mergeVideo.py:229` — **it is never called, so `adjust_delay_to_frame` is the only
> snap, and that is where its cost is actually spent.** *A sentence about "the snap"
> that does not name the path is true of one and false of the other* (`vmsam-dev-3`;
> grids and call sites verified independently by the architect).

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

