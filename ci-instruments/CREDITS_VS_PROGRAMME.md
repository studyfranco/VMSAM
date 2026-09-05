# The unmatched master tails: credits or programme?

**PRIVATE.** Frames were extracted to `/tmp` only, never committed, never attached to
a message, and are described **by category only** — a frame identifies the owner's
media faster than a filename does, and `AGENT.MD`'s privacy section names filenames,
paths, titles and catalogue ids but **not frames or extracted media fragments**. That
omission has been reported as a finding rather than left to the next agent to infer.

## The question

`vmsam-forensic` measured 16 files in a 5–20 % duration-shortfall band, 14 of them
with an **unmatched master tail** — the master continues 68–125 s past the point
where the candidate's audio ends. Audio cannot answer what is in that tail:
**content missing at the end produces no offset change at all**, so there is no
signal for a correlation instrument to find.

If **credits**, the 14 are ordinary files whose masters carry more outro. If
**programme**, 14 candidates are each missing over a minute of content and a repair
would silently fill it from the master.

## Method

Three timestamps per file at **15 / 50 / 85 %** across each unmatched tail, reported
per timestamp and **never averaged**. One sample was not enough because *an ending
sequence and a quiet closing scene look alike at a single point* — the same failure
that produced two wrong visual calls earlier in the campaign, where a **series-level**
observation was given **scene-level** weight.

## A defect in the duration data, found while doing this

`mst_audio_s` is the **minimum across the master's audio streams, not the master's
length.** id 143's master decodes to 1537.37 / 1537.37 / **1434.84** s and the stored
value is 1434.836 — the shortest of three. id 233's decodes to 1477.65 / **1499.05** /
1499.04 and the stored value is 1477.63 — the shortest of nineteen.

**Every tail length in the shortfall sweep is therefore a lower bound**, by a
per-file amount:

| id | stored tail | real tail | understated by |
| ---: | ---: | ---: | ---: |
| 233 | 64.8 s | 86.2 s | 21.4 |
| 234 | 66.6 s | 86.1 s | 19.5 |
| 244 | 64.7 s | 84.1 s | 19.4 |
| **143** | **−0.5 s** | **102.0 s** | **102.5** |

The container tag `mst_dur` **equals** the longest audio stream (id 233: 1499.051 vs
a decoded 1499.05), so `mst_dur` is the true end here and `mst_audio_s` is not. That
*inverts* the refinement made after id 691 — which needed the audio because its
container lied, where these need the container because the audio minimum lies.
**Neither is globally safe; the safe form is the decoded max over audio streams.**

**The consequence larger than any verdict:** band membership is computed from the
understated length. A file whose apparent shortfall fell below the 5 % cut could
exceed it once the true length is used — id 143 moved by 102.5 s, which is 6.6 % of
its master. **The band may be missing files, in the direction of understating the
problem.**

### A retraction, caught before it was reported

The first extraction gave id 143 a tail of **−0.5 s**, and the conclusion one step
away was *"id 143 has no unmatched tail; its prominence was 0.142 because there was
nothing there to match."* That would have been elegant, wrong, and hard to dislodge
**because it explained the weak measurement perfectly.**

What stopped it: the three frames at 15/50/85 % of a *negative* interval all landed
within half a second of each other. Checking whether the master carried video past
its stated audio end — rather than accepting the tidy answer — showed that it did.

## Result — it is not binary. It splits by folder.

| id | tail | 15 % | 50 % | 85 % | verdict |
| ---: | ---: | :--- | :--- | :--- | :--- |
| 233 | 86.2 s | credits | credits | credits | **CREDITS — benign** |
| 234 | 86.1 s | credits | credits | credits | **CREDITS — benign** |
| 244 | 84.1 s | credits | credits | credits | **CREDITS — benign** |
| 203 | 120.9 s | programme | programme | programme | **PROGRAMME** |
| 160 | 84.9 s | programme | programme | programme | **PROGRAMME** |
| 143 | 102.0 s | programme | programme | programme | **PROGRAMME** |
| 49 | 534.0 s | programme | programme | credits | **MIXED** |

**No disagreement within any file** for 233, 203 or 160. Only id 49 changed category
across its tail — programme through 50 %, credit roll by 85 %.

> **id 49 is the file where a single sample would have decided the answer, and which
> sample you took would have decided it.** It is the direct vindication of taking
> three, and it was found by the design rather than by luck.

The split follows the folder split exactly:

- **folder 81 → credits → the 11 files there are benign** — measured on **three**
  files (233, 234, 244) spanning forensic's ordering, not one
- **folder 103 → programme → those 3 files are the worst case** — and it is
  **uniform**: 143, 160 and 203 are all programme

Both halves of the original dichotomy are true, of different files.

## The limit on the benign half — now smaller, but not gone

Three of the eleven folder-81 files are measured. The other **eight** remain
forensic's attribution from a shared signature, stated by its author as such and
**not laundered into a measurement here by agreement**.

## id 143 resolved, and it is not the exception

Excluded from the audio sweep at prominence **0.142** against a noise floor reaching
0.141. Measured by pictures instead, across its real 102.0 s tail: **programme at all
three points.** It is the third member of a now-uniform folder 103.

This also offers a testable alternative to *"id 143 sits at the noise floor"*: if the
mapping of the candidate's final 30 s used the short stream's end as the master's
end, the search region was displaced by 102 s. **Not asserted** — the mapping's
internals are not mine — but it predicts the prominence recovers once corrected.

## Caveats carried with the result, not discovered after it

1. **The window assumes the two files start aligned.** It is
   `[candidate_audio_end, master_audio_end]`, which is the master's unmatched tail
   only if there is no head offset — and these are *refused* files, where a constant
   head offset is exactly what the pipeline could not establish. All four tails are
   long (64.8–534.0 s), so a head offset of a second or two moves the window by a few
   percent and cannot change any verdict here. **On a short-tail file this caveat
   stops being a footnote and decides whether the answer means anything.**
2. **Three points of a long tail.** A short programme fragment inside a long credit
   sequence would be missed. For id 233 it would have to fall between samples; id 49
   is the reverse case, and its transition was found only because three were taken.
