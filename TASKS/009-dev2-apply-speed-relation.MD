# 009 — Apply a measured speed relation to audio and subtitles

**Agent:** `vmsam-dev-2` (repair). **Pairs with 008.** **State:** open.
**Campaign objective 3, the repair half.**

## Goal

Take the coefficient 008 measured and produce a track on the master's timeline:
audio resampled, **and subtitles re-timed by the same coefficient**. `CAMPAIGN.MD`
is explicit that both are taken.

Method: `docs/AUDIO_SPEED_POLICY.MD`, which is a ruling and not a suggestion —
`asetrate`, not `rubberband`, not `atempo`, and read the inverting case before you
assume it carries over to a coefficient it does not name.

Subtitles: `docs/SUBTITLE_CODECS.MD` says which formats can be re-timed at all.
An image format is not one of them; say so rather than shipping a plausible file.

## Marking

`fabricated = "resampled:<factor>"`, or `"chimeric+resampled:<factor>"` in that
order when a staircase sat on top of the slope (`SPEC_ZONE_A.MD` §4). The same
string as a `VMSAM_FABRICATED` Matroska track tag. Consumers test truthiness.

Write the factor you actually applied, at the precision you actually applied it.
A tag reading `resampled:1.042709` on a track resampled by something else is worse
than no tag: campaign 1 spent real work dating 67 outputs by mtime because nothing
in them said what produced them.

## Same structure as 007

Own module, called from zone A, behind an **off-by-default `config.ini`
`[features]` key**. Closes on `SPEC_ZONE_A.MD` §2 — real error file, repaired
track present in the final mkv, compared against a target you built yourself.

## What changed / measured / dropped / blocked
*(fill as you go)*
